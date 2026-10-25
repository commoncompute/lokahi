#include "strata_session.h"
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include <type_traits>

struct Slot {
    bool occupied = false;
    uint64_t ordinal = 0, deadline = 0;
    strata_session_snapshot_v1 value{};
};
struct strata_session_engine {
    strata_session_config_v1 config{};
    std::array<Slot, 64> slots{};
    std::mutex mutex;
    uint64_t ordinal = 0, reserved_bytes = 0;
    uint32_t queued = 0, active = 0, records = 0;
    bool draining = false;
};
static_assert(std::is_standard_layout<strata_session_request_v1>::value, "C ABI layout");
static_assert(sizeof(strata_session_config_v1) == 32, "ABI v1 config");
static_assert(sizeof(strata_session_request_v1) == 200, "ABI v1 request");
static_assert(sizeof(strata_session_snapshot_v1) == 208, "ABI v1 snapshot");

namespace {
bool identifier(const char *id) {
    if (!id) return false;
    for (size_t i = 0; i < 128; ++i) {
        const unsigned char c = static_cast<unsigned char>(id[i]);
        if (!c) return i > 0;
        const bool alpha = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
        const bool digit = c >= '0' && c <= '9';
        if (!alpha && !digit && (i == 0 || (c != '.' && c != '_' && c != ':' && c != '-'))) return false;
    }
    return false;
}
bool terminal(uint32_t state) { return state >= STRATA_STATE_COMPLETED && state <= STRATA_STATE_CANCELLED; }
Slot *find(strata_session_engine *e, const char *id) {
    for (auto &slot : e->slots)
        if (slot.occupied && std::strcmp(slot.value.request_id, id) == 0) return &slot;
    return nullptr;
}
void finish(strata_session_engine *e, Slot &slot, uint32_t state, uint32_t reason) {
    if (slot.value.state == STRATA_STATE_QUEUED) --e->queued;
    else --e->active;
    e->reserved_bytes -= slot.value.reserved_bytes;
    slot.value.reserved_bytes = 0;
    slot.value.state = state;
    slot.value.reason = reason;
    ++slot.value.revision;
}
template<class Function> strata_session_result guard(Function fn) noexcept {
    try { return fn(); } catch (...) { return STRATA_SESSION_INTERNAL; }
}
template<class Function> strata_session_result request_op(strata_session_engine *e, const char *id, Function fn) {
    if (!e || !identifier(id)) return STRATA_SESSION_INVALID;
    return guard([&]() -> strata_session_result {
        std::lock_guard<std::mutex> lock(e->mutex);
        Slot *slot = find(e, id);
        if (!slot) return STRATA_SESSION_NOT_FOUND;
        if (terminal(slot->value.state)) return STRATA_SESSION_TERMINAL;
        return fn(*slot);
    });
}
}

extern "C" strata_session_result strata_session_create_v1(const strata_session_config_v1 *c, strata_session_engine **out) {
    if (!out) return STRATA_SESSION_INVALID;
    *out = nullptr;
    if (!c || c->abi_version != 1 || c->struct_bytes != sizeof(*c) || !c->memory_limit_bytes ||
        !c->max_queued || c->max_queued > 63 || c->max_active != 1 ||
        c->max_records < c->max_queued + 1 || c->max_records > 64 || c->reserved) return STRATA_SESSION_INVALID;
    return guard([&]() -> strata_session_result {
        auto *e = new (std::nothrow) strata_session_engine();
        if (!e) return STRATA_SESSION_INTERNAL;
        e->config = *c;
        *out = e;
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_destroy_v1(strata_session_engine **handle) {
    if (!handle || !*handle) return STRATA_SESSION_INVALID;
    auto *e = *handle;
    // Destruction requires the trusted owner to join callers first. A mutex
    // cannot make use-after-destroy safe for a foreign caller retaining a pointer.
    return guard([&]() -> strata_session_result {
        {
            std::lock_guard<std::mutex> lock(e->mutex);
            if (e->active || e->queued) return STRATA_SESSION_BUSY;
        }
        delete e;
        *handle = nullptr;
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_submit_v1(strata_session_engine *e, const strata_session_request_v1 *r, uint64_t now) {
    if (!e || !r || r->abi_version != 1 || r->struct_bytes != sizeof(*r) || !identifier(r->request_id) ||
        r->reserved || !r->reserve_bytes || r->deadline_monotonic_ns <= now || !r->prompt_tokens ||
        !r->max_output_tokens || r->max_output_tokens > 16384 || r->context_tokens > 65536 ||
        uint64_t(r->prompt_tokens) + r->max_output_tokens > r->context_tokens) return STRATA_SESSION_INVALID;
    bool digest = false;
    for (auto byte : r->model_digest) digest |= byte != 0;
    if (!digest) return STRATA_SESSION_INVALID;
    return guard([&]() -> strata_session_result {
        std::lock_guard<std::mutex> lock(e->mutex);
        if (find(e, r->request_id)) return STRATA_SESSION_DUPLICATE;
        if (e->draining || e->queued >= e->config.max_queued || e->records >= e->config.max_records ||
            r->reserve_bytes > e->config.memory_limit_bytes - e->reserved_bytes || e->ordinal == UINT64_MAX)
            return STRATA_SESSION_NO_CAPACITY;
        for (auto &slot : e->slots) if (!slot.occupied) {
            slot = Slot{};
            slot.occupied = true;
            slot.ordinal = ++e->ordinal;
            slot.deadline = r->deadline_monotonic_ns;
            slot.value.abi_version = 1;
            slot.value.struct_bytes = sizeof(slot.value);
            std::memcpy(slot.value.request_id, r->request_id, sizeof(r->request_id));
            std::memcpy(slot.value.model_digest, r->model_digest, sizeof(r->model_digest));
            slot.value.reserved_bytes = r->reserve_bytes;
            slot.value.state = STRATA_STATE_QUEUED;
            slot.value.revision = 1;
            slot.value.prompt_tokens = r->prompt_tokens;
            slot.value.max_output_tokens = r->max_output_tokens;
            e->reserved_bytes += r->reserve_bytes;
            ++e->queued; ++e->records;
            return STRATA_SESSION_OK;
        }
        return STRATA_SESSION_INTERNAL;
    });
}
extern "C" strata_session_result strata_session_next_v1(strata_session_engine *e, uint64_t now, strata_session_snapshot_v1 *out) {
    if (out) *out = {};
    if (!e || !out) return STRATA_SESSION_INVALID;
    return guard([&]() -> strata_session_result {
        std::lock_guard<std::mutex> lock(e->mutex);
        Slot *next = nullptr;
        for (auto &slot : e->slots) if (slot.occupied && slot.value.state == STRATA_STATE_QUEUED) {
            if (slot.deadline <= now) finish(e, slot, STRATA_STATE_CANCELLED, STRATA_REASON_DEADLINE);
            else if (!next || slot.ordinal < next->ordinal) next = &slot;
        }
        if (e->active >= e->config.max_active) return STRATA_SESSION_BUSY;
        if (!next) return STRATA_SESSION_NOT_FOUND;
        --e->queued; ++e->active;
        next->value.state = STRATA_STATE_RUNNING;
        ++next->value.revision;
        *out = next->value;
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_read_v1(strata_session_engine *e, const char *id, strata_session_snapshot_v1 *out) {
    if (out) *out = {};
    if (!e || !out || !identifier(id)) return STRATA_SESSION_INVALID;
    return guard([&]() -> strata_session_result {
        std::lock_guard<std::mutex> lock(e->mutex);
        Slot *slot = find(e, id);
        if (!slot) return STRATA_SESSION_NOT_FOUND;
        *out = slot->value;
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_add_tokens_v1(strata_session_engine *e, const char *id, uint32_t delta) {
    return request_op(e, id, [&](Slot &slot) -> strata_session_result {
        if (slot.value.state == STRATA_STATE_CANCEL_PENDING) return STRATA_SESSION_CANCEL_PENDING;
        if (slot.value.state != STRATA_STATE_RUNNING || !delta ||
            delta > slot.value.max_output_tokens - slot.value.completion_tokens) return STRATA_SESSION_INVALID;
        slot.value.completion_tokens += delta;
        ++slot.value.revision;
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_cancel_v1(strata_session_engine *e, const char *id) {
    return request_op(e, id, [&](Slot &slot) -> strata_session_result {
        if (slot.value.state == STRATA_STATE_QUEUED) finish(e, slot, STRATA_STATE_CANCELLED, STRATA_REASON_USER_CANCEL);
        else if (slot.value.state == STRATA_STATE_RUNNING) {
            slot.value.state = STRATA_STATE_CANCEL_PENDING;
            slot.value.reason = STRATA_REASON_USER_CANCEL;
            ++slot.value.revision;
        }
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_check_v1(strata_session_engine *e, const char *id, uint64_t now) {
    return request_op(e, id, [&](Slot &slot) -> strata_session_result {
        if (slot.deadline <= now && slot.value.state != STRATA_STATE_CANCEL_PENDING) {
            if (slot.value.state == STRATA_STATE_QUEUED) finish(e, slot, STRATA_STATE_CANCELLED, STRATA_REASON_DEADLINE);
            else { slot.value.state = STRATA_STATE_CANCEL_PENDING; slot.value.reason = STRATA_REASON_DEADLINE; ++slot.value.revision; }
        }
        return slot.value.state == STRATA_STATE_RUNNING ? STRATA_SESSION_OK :
            slot.value.state == STRATA_STATE_QUEUED ? STRATA_SESSION_BUSY : STRATA_SESSION_CANCEL_PENDING;
    });
}
extern "C" strata_session_result strata_session_finish_v1(strata_session_engine *e, const char *id, uint32_t state, uint32_t reason) {
    return request_op(e, id, [&](Slot &slot) -> strata_session_result {
        if (slot.value.state != STRATA_STATE_RUNNING && slot.value.state != STRATA_STATE_CANCEL_PENDING) return STRATA_SESSION_INVALID;
        const bool cancelled = slot.value.state == STRATA_STATE_CANCEL_PENDING;
        if (cancelled && (state != STRATA_STATE_CANCELLED || reason != slot.value.reason)) return STRATA_SESSION_INVALID;
        if (!cancelled && !((state == STRATA_STATE_COMPLETED &&
              (reason == STRATA_REASON_STOP || (reason == STRATA_REASON_LENGTH && slot.value.completion_tokens == slot.value.max_output_tokens))) ||
              (state == STRATA_STATE_FAILED && reason == STRATA_REASON_EXECUTION_FAILED))) return STRATA_SESSION_INVALID;
        finish(e, slot, state, reason);
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_forget_v1(strata_session_engine *e, const char *id) {
    if (!e || !identifier(id)) return STRATA_SESSION_INVALID;
    return guard([&]() -> strata_session_result {
        std::lock_guard<std::mutex> lock(e->mutex);
        Slot *slot = find(e, id);
        if (!slot) return STRATA_SESSION_NOT_FOUND;
        if (!terminal(slot->value.state)) return STRATA_SESSION_BUSY;
        *slot = Slot{}; --e->records;
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_drain_v1(strata_session_engine *e) {
    if (!e) return STRATA_SESSION_INVALID;
    return guard([&]() -> strata_session_result {
        std::lock_guard<std::mutex> lock(e->mutex);
        e->draining = true;
        return STRATA_SESSION_OK;
    });
}
extern "C" strata_session_result strata_session_totals_v1_read(strata_session_engine *e, strata_session_totals_v1 *out) {
    if (out) *out = {};
    if (!e || !out) return STRATA_SESSION_INVALID;
    return guard([&]() -> strata_session_result {
        std::lock_guard<std::mutex> lock(e->mutex);
        *out = {e->config.memory_limit_bytes, e->reserved_bytes, e->queued, e->active, e->records, e->draining ? 1u : 0u};
        return STRATA_SESSION_OK;
    });
}
