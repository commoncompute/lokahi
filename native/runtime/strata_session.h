#ifndef STRATA_SESSION_H
#define STRATA_SESSION_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Trusted, already-authorized host descriptors ONLY. No customer JSON, paths,
 * plaintext, signing keys, weights or tensor pointers cross this ABI.
 * This is lifecycle accounting, not LLM execution or backend qualification. */
typedef struct strata_session_engine strata_session_engine;
typedef enum {
    STRATA_SESSION_OK = 0, STRATA_SESSION_INVALID = 1,
    STRATA_SESSION_NO_CAPACITY = 2, STRATA_SESSION_DUPLICATE = 3,
    STRATA_SESSION_NOT_FOUND = 4, STRATA_SESSION_BUSY = 5,
    STRATA_SESSION_TERMINAL = 6, STRATA_SESSION_CANCEL_PENDING = 7,
    STRATA_SESSION_INTERNAL = 8
} strata_session_result;
typedef enum {
    STRATA_STATE_QUEUED = 1, STRATA_STATE_RUNNING = 2,
    STRATA_STATE_CANCEL_PENDING = 3, STRATA_STATE_COMPLETED = 4,
    STRATA_STATE_FAILED = 5, STRATA_STATE_CANCELLED = 6
} strata_session_state;
typedef enum {
    STRATA_REASON_NONE = 0, STRATA_REASON_STOP = 1, STRATA_REASON_LENGTH = 2,
    STRATA_REASON_EXECUTION_FAILED = 3, STRATA_REASON_USER_CANCEL = 4,
    STRATA_REASON_DEADLINE = 5
} strata_session_reason;

typedef struct {
    uint32_t abi_version; /* exactly 1 */
    uint32_t struct_bytes;
    uint64_t memory_limit_bytes;
    uint32_t max_queued; /* 1..63 */
    uint32_t max_active; /* exactly 1: no continuous batching claimed */
    uint32_t max_records; /* max_queued + max_active .. 64, includes terminals */
    uint32_t reserved; /* zero */
} strata_session_config_v1;
typedef struct {
    uint32_t abi_version;
    uint32_t struct_bytes;
    char request_id[128]; /* ASCII [A-Za-z0-9][A-Za-z0-9._:-]*, NUL terminated */
    uint8_t model_digest[32]; /* nonzero identity, NOT signature verification */
    uint64_t reserve_bytes; /* complete, validated worst-phase demand */
    uint64_t deadline_monotonic_ns; /* trusted host conversion, no wall-clock mix */
    uint32_t prompt_tokens;
    uint32_t max_output_tokens;
    uint32_t context_tokens; /* prompt + output <= context, max 65536 */
    uint32_t reserved;
} strata_session_request_v1;
typedef struct {
    uint32_t abi_version;
    uint32_t struct_bytes;
    char request_id[128];
    uint8_t model_digest[32];
    uint64_t reserved_bytes; /* live reservation; zero after safe terminal */
    uint64_t revision; /* per-request monotonic snapshot revision, not wire sequence */
    uint32_t state;
    uint32_t reason;
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t max_output_tokens;
    uint32_t native_runtime_qualified; /* always zero */
} strata_session_snapshot_v1;
typedef struct {
    uint64_t memory_limit_bytes;
    uint64_t reserved_bytes;
    uint32_t queued;
    uint32_t active;
    uint32_t records; /* terminal tombstones retained until explicit forget */
    uint32_t draining;
} strata_session_totals_v1;

strata_session_result strata_session_create_v1(const strata_session_config_v1 *, strata_session_engine **);
/* Refuses destruction while work is in flight. Pointer cleared on success. */
strata_session_result strata_session_destroy_v1(strata_session_engine **);
strata_session_result strata_session_submit_v1(strata_session_engine *, const strata_session_request_v1 *, uint64_t now_ns);
/* FIFO. Expire queued deadlines first. Returned snapshot is a value copy. */
strata_session_result strata_session_next_v1(strata_session_engine *, uint64_t now_ns, strata_session_snapshot_v1 *);
strata_session_result strata_session_read_v1(strata_session_engine *, const char *request_id, strata_session_snapshot_v1 *);
strata_session_result strata_session_add_tokens_v1(strata_session_engine *, const char *request_id, uint32_t delta);
/* Queued cancellation terminates immediately; running cancellation retains
 * ALL bytes/active ownership until executor confirms a safe terminal point. */
strata_session_result strata_session_cancel_v1(strata_session_engine *, const char *request_id);
strata_session_result strata_session_check_v1(strata_session_engine *, const char *request_id, uint64_t now_ns);
strata_session_result strata_session_finish_v1(strata_session_engine *, const char *request_id, uint32_t state, uint32_t reason);
strata_session_result strata_session_forget_v1(strata_session_engine *, const char *request_id);
/* Reject new submissions; existing FIFO still runs. No forceful GPU teardown. */
strata_session_result strata_session_drain_v1(strata_session_engine *);
strata_session_result strata_session_totals_v1_read(strata_session_engine *, strata_session_totals_v1 *);
#ifdef __cplusplus
}
#endif
#endif
