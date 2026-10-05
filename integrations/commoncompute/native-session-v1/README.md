# CommonCompute native-session integration foundation

Miki Engine's current native control core is in `native/runtime`. This package
wraps it; **the CommonCompute app has not been modified or deployed**.

## Current integration boundary (inspection 2026-10-04)

The local CommonCompute checkout uses `XPCMLXLLMRunner` with a fresh assignment
runner, and `SealedXPCMLXLLMContract` with signed authorization/model manifests
and ciphertext input. Plaintext decryption/tokenization belongs inside the
authorized worker, not the provider host. Model artifacts are reverified before
execution. The unrelated dirty checkout was preserved.

```text
CommonCompute sealed request / signed model authorization (existing)
  -> authorized isolated worker verifies, decrypts, tokenizes (existing MLX lane)
     -> validated opaque identity / complete byte demand (integration target)
        -> Miki Engine C++ session owner (implemented, fixture linked)
           -> MLX/native GPU/ANE executor (not connected here)
     -> sealed/output transport and durable receipt (existing owner, preserve)
```

Do not send this ABI through generic RPC or use the older `strata-wire-v1`
plaintext fixture package as customer transport. This metadata core is not
cryptography, tenant authorization, artifact staging, network input decoding,
sandboxing or proof of complete-model memory safety.

## Files and readiness

| File | Implemented scope | Not implemented |
| --- | --- | --- |
| `StrataSessionOwner.swift` | Swift actor, C ABI owner, trusted descriptors, cancellation/token totals | XPC runner, decryption, inference, model residency |
| `VerifySession.swift` | Executable linked to the actual native library | App acceptance, distribution/signing, fleet compatibility |
| `session.ts` | Strict lossless diagnostic decoder and monotonic per-request observer | Node FFI, customer input, RPC or token stream |
| `verify-diagnostic.ts` | Tests snapshots emitted by C++ through Python | TypeScript native inference or MLX equivalence |

Keep legacy Clang module `StrataSession`, `strata_*` symbols, Swift owner name,
and stored runtime IDs unchanged during the product rename. Python callers may
use the new `miki_engine.native_session` facade; it reuses the same binding.

Swift 6.2 isolated `deinit` is required by the current owner. It compiles with
the local Swift 6.2.4 toolchain; CommonCompute's deployment/toolchain matrix
must be qualified before adding it to app targets. No unsafe Sendable bypass
is used. Explicit joined `close` is required: deinitialization will not force
free queued/active work and can leak metadata if ownership is violated.

The diagnostic format uses decimal **strings** for uint64 reservations and
snapshot revisions so JavaScript never rounds them. Request IDs/digests are
stable opaque identity; revision and counts must not regress. Terminal values
are immutable; an identical repeated snapshot is harmless. Revisions can skip
between observations and are not a reliable ordered customer token stream.

## Acceptance before provider canary

- Keep `mlx_llm` the oracle/fallback and all new routing disabled by default.
- Verify signed runtime/model identity, ciphertext-only ingress and worker-only
  plaintext/token counts. Add no arbitrary path, URL, module or code authority.
- Prove resident model memory and worst-phase KV/temp reservations on the
  physical device; do not trust caller-declared byte counts.
- Test queued/active cancel, deadline, worker death, crash/restart, bounded queue,
  immutable terminal accounting and host anti-replay. Reject native owner
  destruction until all GPU/ANE callbacks have been joined.
- Compare exact prompt/greedy output tokens and numerical logits/KV with MLX,
  then serialize whole-model cold/warm TTFT, prefill/decode and peak memory runs.
  Never replay after customer-visible partial output; retries before output
  require an authorized policy.
- Gate any speed preference by matched physical-hardware evidence, thermal/
  reliability limits and independent replication. A library written in C++
  does not by itself beat the MLX native kernels.
- SSD residency remains opt-in, disabled for resident maximum-speed comparisons;
  project storage on the HDD does not qualify runtime SSD paging.

Bindings are currently integration groundwork, not a provider release.
