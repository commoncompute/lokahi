# Miki bootstrap: validation and blocked physical baseline

Date: October 3, 2026. Controller checkpoint:
`4184607c062b273bb325dacb87222b259043f06e`, local branch
`codex/miki-empirical-loop-20261003`, based on published Strata native checkpoint
`76b8d0d3f4238dd3f35ffbf2a56179ac2bc51858`. No remote push, production
promotion, CommonCompute release, or measured runtime acceleration.

Design/run guide: [Miki empirical loop](../design/MIKI_EMPIRICAL_LOOP.md).
Hardware discovery: [M1 atlas](../../research/hardware_atlas/apple_m1_20261003.md).
Existing Strata native/ANE work and MLX-LM oracle/fallback are preserved.

## Implemented

Project-local commands for hardware discovery, pinned offline baseline,
bounded evolution, experiment inspection, ancestry and reproduction. Protected
source/checkpoint/diff verification; structured Astra proposer and independent
critique; serial fresh-worker correctness/performance evaluation; immutable
workloads/statistics/policy; append-only SQLite history and hashed external raw
artifacts; local compare-and-swap retention and re-profiling.

Only three initial native MLX compiler MLP plans are mutable. No arbitrary
generated source, new serving daemon, integrated ANE path, production pager,
or universal model-family support is claimed. The native MLX compiler routes
have not yet been exercised against real model weights in this checkpoint.

## Local validation

Host: physical M1, 16 GiB unified memory; macOS 26.6.2 / 25G83;
Xcode 26.3 / 17C529. Existing Python 3.13.11, MLX 0.32.2 and MLX-LM 0.31.3.
No package or model downloads/installations performed.

| Gate | Result | What it establishes |
| --- | --- | --- |
| Full regression | 396 tests, 22 subtests passed; no skips | Existing regression plus 37 new synthetic/controller contracts |
| Controller generations | Synthetic two-generation acceptance/ancestry/re-profile test passed | Control flow, not a real improvement |
| Immutable MLX comparison | Missing and slower reference rejected in later-generation evaluator tests | Parent-only wins cannot be accepted |
| Proposer contract | Synthetic JSONL/schema/read-only/forbidden-tool tests passed | No live Astra response or model availability proof |
| C/NEON layout | 400 cases passed under UBSan | Bit-preserving CPU layout correctness |
| C++ admission | 1,837 cases passed under UBSan | Synthetic budget arithmetic, not fleet capacity |
| Packed Q4 CPU contract | 75 cases passed under UBSan | Bounds/numerical contract, not Metal dispatch |
| Objective-C reverse-engineered ANE tools | Both executables compiled, warnings as errors; CLI rejection tests passed | No valid ANE inference dispatched by these checks |
| Packed Q4 Metal/Objective-C++ host | Compiled and linked, warnings as errors | No valid GPU dispatch/readback or throughput measurement |
| FP16 fused SwiGLU Metal entries | Compiled and linked to metallib | No BF16 JIT or real-model execution |
| Syntax / whitespace | `compileall` and Git diff checks passed | Syntax/patch formatting |

These are local correctness and compilation checks, not measured speed claims.

## Real failure evidence, preserved rather than retried indefinitely

Target: local pinned `Llama-3.2-3B-Instruct-4bit`. Admission runs before artifact
hashing/model loading. Therefore no local weight verification, valid model
timings, correctness traces or bottleneck profile was obtained in these attempts.

| Experiment | Internal free bytes observed | Outcome | Raw artifact SHA-256 |
| --- | --- | --- | --- |
| `481f1558eaa04dde97b1f366005b8646` | 42,480,365,568 | FAILED: <40 GiB reserve | `02dd740450a2363693ea76de124c6831e2cb765d5a43c8541348b79ff84a2c38` |
| `f736a514aa97407abf6aa96c1840fd73` | 30,103,490,560 | FAILED: <40 GiB reserve | `fc8a009b8b5affe01360a6f824ae2703fcd5a1d103576c7bd05a5f67cdfbac81` |

Reserve: 42,949,672,960 bytes (40 GiB). The second attempt used the frozen
controller checkpoint above. Its STARTED event records hardware class, Git SHA,
protected-source digest and full manifest. The first attempt was a bootstrap
failure before those fields were added; its original record is unchanged.
Raw traces are private on the approved X9, referenced by hash/path in local
APFS SQLite. Public report deliberately omits private absolute paths and DB.

Accepted real candidates: **0**. Retained real plans: **0**. Astra requests: **0**.
Full-model Strata/MLX ratio, TTFT, decode, energy and full-runtime residency:
**unknown**, not zero and not a win. Historical ANE precision rejection remains.

## Storage operation and remaining dependency

The user explicitly approved archiving only the older `native-mlx-xpc` build.
Its metadata-preserving tar is on the X9; every one of 25,909 source entries,
regular-file content hash, link, ownership/mode, nanosecond mtime and extended
attribute was checked. No nonempty ACL or nonzero file flag was admitted.
Source stability and open-file safety were rechecked, then only that original
was removed via its exact quarantine path. A private verified/completion receipt
and recovery instructions accompany the tar. Original regular data totals
3,149,538,928 bytes; archive totals 3,248,936,448 bytes. This does not guarantee
an equivalent increase in APFS free space or remove Time Machine snapshots.

Models, Python, current native builds and unrelated data remain untouched.
The X9 is exFAT: it holds archive/artifact files, while Git, Python, active builds
and SQLite remain internal. The failed HDD was not accessed. Other disk activity
coincided with falling internal headroom; its cause is unverified. No global
cleanup, process termination or reserve reduction was attempted.

Next required external condition: restore **stable >=40 GiB internal free space**
and safe current memory pressure. At the second attempt this requires about
12 GiB additional free space; more headroom is advisable because usage changed
during the session. Then run the baseline/profile and the bounded Astra-driven
loop. Stop if those guards or proposal/compiler/correctness gates fail. Do not
advance to ANE/AMX/fleet work merely to bypass this missing first-loop evidence.

## Reproduce validation

Use an existing environment and an approved internal-SSD build directory on
your machine; never copy literal placeholder paths or install silently. Full
commands for native builds are in the
[October 1 validation guide](PUBLICATION_VALIDATION_20261001.md). This session
used those same build-only/arithmetic gates, with a fresh Miki build directory.

```sh
STRATA_ANE_EXPERIMENT_BUILD='/path/to/approved/native-build/ane' \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /path/to/existing/python -m pytest -q -p no:cacheprovider
PYTHONPYCACHEPREFIX='/path/to/approved/native-build/pycache' \
  /path/to/existing/python -m compileall -q src tests benchmarks
git diff --check
```

Use Miki `experiment show` for preserved failures and `experiment reproduce`
for a new controlled attempt once admission is safe. Reproduction verifies the
protected source and never rewrites the previous result or auto-promotes a plan.
