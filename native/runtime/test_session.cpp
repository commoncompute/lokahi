#include "strata_session.h"
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

static int checks = 0;
#define CHECK(expression) do { ++checks; assert(expression); } while (0)
static strata_session_config_v1 config(uint64_t memory = 1000, uint32_t queued = 4, uint32_t records = 8) {
    return {1, sizeof(strata_session_config_v1), memory, queued, 1, records, 0};
}
static strata_session_request_v1 request(const char *id, uint64_t bytes = 100, uint64_t deadline = 1000) {
    strata_session_request_v1 value{};
    value.abi_version = 1; value.struct_bytes = sizeof(value);
    std::strncpy(value.request_id, id, sizeof(value.request_id) - 1);
    value.model_digest[0] = 7;
    value.reserve_bytes = bytes; value.deadline_monotonic_ns = deadline;
    value.prompt_tokens = 64; value.max_output_tokens = 32; value.context_tokens = 128;
    return value;
}
static strata_session_snapshot_v1 read(strata_session_engine *e, const char *id) {
    strata_session_snapshot_v1 value{};
    CHECK(strata_session_read_v1(e, id, &value) == STRATA_SESSION_OK);
    CHECK(value.native_runtime_qualified == 0);
    return value;
}
static strata_session_totals_v1 totals(strata_session_engine *e) {
    strata_session_totals_v1 value{};
    CHECK(strata_session_totals_v1_read(e, &value) == STRATA_SESSION_OK);
    CHECK(value.reserved_bytes <= value.memory_limit_bytes);
    return value;
}

int main() {
    strata_session_engine *e = nullptr;
    auto cfg = config();
    CHECK(strata_session_create_v1(nullptr, &e) == STRATA_SESSION_INVALID && !e);
    for (unsigned which = 0; which < 6; ++which) {
        auto bad = cfg;
        switch (which) { case 0: bad.abi_version = 2; break; case 1: bad.struct_bytes = 0; break;
            case 2: bad.max_active = 2; break; case 3: bad.max_queued = 64; break;
            case 4: bad.reserved = 1; break; default: bad.memory_limit_bytes = 0; }
        CHECK(strata_session_create_v1(&bad, &e) == STRATA_SESSION_INVALID && !e);
    }
    CHECK(strata_session_create_v1(&cfg, &e) == STRATA_SESSION_OK);
    auto a = request("A", 600), b = request("B", 300), c = request("C", 200);
    CHECK(strata_session_submit_v1(e, &a, 1) == STRATA_SESSION_OK);
    CHECK(strata_session_submit_v1(e, &a, 1) == STRATA_SESSION_DUPLICATE);
    CHECK(strata_session_submit_v1(e, &b, 1) == STRATA_SESSION_OK);
    CHECK(strata_session_submit_v1(e, &c, 1) == STRATA_SESSION_NO_CAPACITY);
    a.model_digest[0] = 99; // submission is copied, never a borrowed descriptor
    CHECK(read(e, "A").model_digest[0] == 7);
    CHECK(totals(e).reserved_bytes == 900);
    strata_session_snapshot_v1 out{};
    CHECK(strata_session_next_v1(e, 2, &out) == STRATA_SESSION_OK);
    CHECK(std::strcmp(out.request_id, "A") == 0 && out.state == STRATA_STATE_RUNNING);
    CHECK(strata_session_next_v1(e, 2, &out) == STRATA_SESSION_BUSY && !out.abi_version);
    CHECK(strata_session_add_tokens_v1(e, "A", 8) == STRATA_SESSION_OK);
    CHECK(read(e, "A").completion_tokens == 8 && read(e, "B").completion_tokens == 0);
    CHECK(strata_session_add_tokens_v1(e, "A", 25) == STRATA_SESSION_INVALID);
    CHECK(strata_session_add_tokens_v1(e, "B", 1) == STRATA_SESSION_INVALID);
    CHECK(strata_session_forget_v1(e, "A") == STRATA_SESSION_BUSY);
    CHECK(strata_session_cancel_v1(e, "A") == STRATA_SESSION_OK);
    CHECK(read(e, "A").state == STRATA_STATE_CANCEL_PENDING);
    CHECK(totals(e).reserved_bytes == 900 && totals(e).active == 1);
    CHECK(strata_session_destroy_v1(&e) == STRATA_SESSION_BUSY && e);
    CHECK(strata_session_add_tokens_v1(e, "A", 1) == STRATA_SESSION_CANCEL_PENDING);
    CHECK(strata_session_finish_v1(e, "A", STRATA_STATE_COMPLETED, STRATA_REASON_STOP) == STRATA_SESSION_INVALID);
    CHECK(strata_session_finish_v1(e, "A", STRATA_STATE_CANCELLED, STRATA_REASON_USER_CANCEL) == STRATA_SESSION_OK);
    CHECK(read(e, "A").completion_tokens == 8 && totals(e).reserved_bytes == 300);
    const auto revision = read(e, "A").revision;
    CHECK(strata_session_finish_v1(e, "A", STRATA_STATE_CANCELLED, STRATA_REASON_USER_CANCEL) == STRATA_SESSION_TERMINAL);
    CHECK(strata_session_cancel_v1(e, "A") == STRATA_SESSION_TERMINAL);
    CHECK(read(e, "A").revision == revision);
    CHECK(strata_session_submit_v1(e, &a, 1) == STRATA_SESSION_DUPLICATE);
    CHECK(strata_session_drain_v1(e) == STRATA_SESSION_OK);
    CHECK(strata_session_submit_v1(e, &c, 1) == STRATA_SESSION_NO_CAPACITY);
    CHECK(strata_session_next_v1(e, 2, &out) == STRATA_SESSION_OK && std::strcmp(out.request_id, "B") == 0);
    CHECK(strata_session_add_tokens_v1(e, "B", 32) == STRATA_SESSION_OK);
    CHECK(strata_session_finish_v1(e, "B", STRATA_STATE_COMPLETED, STRATA_REASON_LENGTH) == STRATA_SESSION_OK);
    CHECK(totals(e).reserved_bytes == 0 && totals(e).active == 0 && totals(e).records == 2);
    CHECK(strata_session_forget_v1(e, "A") == STRATA_SESSION_OK);
    CHECK(strata_session_destroy_v1(&e) == STRATA_SESSION_OK && !e);

    cfg = config(1000, 1, 2);
    CHECK(strata_session_create_v1(&cfg, &e) == STRATA_SESSION_OK);
    a = request("A", 100, 10); b = request("B", 100, 20); c = request("C", 100, 30);
    CHECK(strata_session_submit_v1(e, &a, 1) == STRATA_SESSION_OK);
    CHECK(strata_session_next_v1(e, 2, &out) == STRATA_SESSION_OK);
    CHECK(strata_session_submit_v1(e, &b, 2) == STRATA_SESSION_OK);
    CHECK(strata_session_submit_v1(e, &c, 2) == STRATA_SESSION_NO_CAPACITY);
    CHECK(strata_session_next_v1(e, 21, &out) == STRATA_SESSION_BUSY);
    CHECK(read(e, "B").state == STRATA_STATE_CANCELLED && read(e, "B").reason == STRATA_REASON_DEADLINE);
    CHECK(strata_session_check_v1(e, "A", 21) == STRATA_SESSION_CANCEL_PENDING);
    CHECK(totals(e).reserved_bytes == 100 && totals(e).active == 1);
    CHECK(strata_session_finish_v1(e, "A", STRATA_STATE_CANCELLED, STRATA_REASON_USER_CANCEL) == STRATA_SESSION_INVALID);
    CHECK(strata_session_finish_v1(e, "A", STRATA_STATE_CANCELLED, STRATA_REASON_DEADLINE) == STRATA_SESSION_OK);
    CHECK(strata_session_destroy_v1(&e) == STRATA_SESSION_OK);

    CHECK(strata_session_create_v1(&cfg, &e) == STRATA_SESSION_OK);
    for (unsigned which = 0; which < 9; ++which) {
        auto bad = request("bad");
        switch (which) { case 0: bad.abi_version = 2; break; case 1: bad.struct_bytes = 0; break;
            case 2: bad.reserve_bytes = 0; break; case 3: bad.deadline_monotonic_ns = 1; break;
            case 4: bad.reserved = 1; break; case 5: bad.model_digest[0] = 0; break;
            case 6: bad.prompt_tokens = UINT32_MAX; break; case 7: bad.max_output_tokens = 16385; break;
            default: std::memset(bad.request_id, 'a', sizeof(bad.request_id)); }
        CHECK(strata_session_submit_v1(e, &bad, 1) == STRATA_SESSION_INVALID);
    }
    CHECK(strata_session_destroy_v1(&e) == STRATA_SESSION_OK);

    // Competing reservations are serialized in native code, not a Python GIL.
    cfg = config(1000, 63, 64);
    CHECK(strata_session_create_v1(&cfg, &e) == STRATA_SESSION_OK);
    std::vector<std::thread> threads;
    std::array<strata_session_result, 32> results{};
    for (unsigned i = 0; i < results.size(); ++i) threads.emplace_back([&, i] {
        const std::string name = "parallel-" + std::to_string(i);
        const auto value = request(name.c_str(), 100);
        results[i] = strata_session_submit_v1(e, &value, 1);
    });
    for (auto &thread : threads) thread.join();
    unsigned admitted = 0;
    for (auto result : results) { CHECK(result == STRATA_SESSION_OK || result == STRATA_SESSION_NO_CAPACITY); admitted += result == STRATA_SESSION_OK; }
    CHECK(admitted == 10 && totals(e).reserved_bytes == 1000);
    for (unsigned i = 0; i < results.size(); ++i) if (results[i] == STRATA_SESSION_OK) {
        const std::string name = "parallel-" + std::to_string(i);
        CHECK(strata_session_cancel_v1(e, name.c_str()) == STRATA_SESSION_OK);
    }
    CHECK(totals(e).reserved_bytes == 0);
    CHECK(strata_session_destroy_v1(&e) == STRATA_SESSION_OK);
    std::cout << checks << " native lifecycle/accounting checks passed; synthetic descriptors, no model execution\n";
}
