# Miki Engine rename and CommonCompute native foundation

Work begun 2026-10-04; handoff 2026-10-05. Evidence scope: local CPU lifecycle
correctness, cross-language linkage and source/storage migration. **No new
complete-model performance result or production promotion.**

## Changes

- Product/source distribution renamed to **Miki Engine / `miki-engine`** as
  approved. `miki_engine` is a lazy Python facade; `ollm`, the `miki` optimizer
  CLI, C ABI `strata_*`, existing runtime/wire IDs and historical artifacts
  remain compatible. Version remains 0.5.0; no registry publication.
- Native C++ fixed-capacity session ownership/admission/cancel/deadline/token
  accounting with a versioned C ABI, plus thin Python/Swift wrappers and a
  TypeScript metadata codec. See `native/runtime/README.md` for precise limits.
- Current CommonCompute sealed contract was inspected read-only. The older
  plaintext integration playbook now carries a historical-boundary warning.
  CommonCompute's dirty source and existing MLX lane remain untouched.
- Existing Metal Q4/fused FFN, CPU/NEON/planner and reverse-engineered private
  ANE code are preserved, not newly qualified or claimed as a complete model.

## Local validation

Physical Apple M1, 16 GiB, macOS 26.6.2 (25G83), Xcode 26.3, Swift 6.2.4,
Python 3.13.11, MLX 0.32.2 and MLX-LM 0.31.3. No dependencies or models were
downloaded. Model storage remains the existing internal-SSD Strata directory.

- C++ UBSan lifecycle executable: **151 checks passed**, including competing
  reservations, FIFO, cancellation, deadline and terminal accounting.
- Python ctypes and Swift linked native fixtures pass. TypeScript consumes
  real native-produced diagnostic values; strict TypeScript checking passes.
- Combined ASan/UBSan attempt hung **before `main`** in ASan initialization
  on this machine. The sampled test process was terminated; ASan coverage is
  not claimed. UBSan was run separately, not substituted as an ASan result.
- Full Python regression: **402 tests plus 22 subtests passed** both from the
  internal source and from the migrated HDD source (12.46 s / 12.57 s). Native
  UBSan also passes from the migrated source. `compileall`, build-path dry-run
  contracts and `git diff --check` pass. CPU tests do not establish LLM latency
  or throughput.

The protected 40 GiB internal-SSD reserve initially failed admission. Available
space fluctuated below the floor during other machine activity; no real-model
benchmark was forced through and no guard was weakened. Existing baseline
failures/history remain unchanged. A valid matched MLX/model benchmark is the
next gate, not an inferred speedup from native linkage.

Fresh Qwen3-0.6B control attempt at clean source commit
`375c3077eaafd72fe7fc79bc077a9644a042e218`:

- Experiment: `bd83120b799940ba82a2af497810b684`.
- Outcome: admission failure before loading weights; no measurements/winner.
- Available internal bytes: 41,808,539,648 (about 38.94 GiB), below the
  42,949,672,960-byte reserve. The guard was preserved.
- Private raw failure artifact on the mounted replacement HDD; SHA-256:
  `14170d7d5da8ea99c8064c7a40b4ee847003abff8d4cf685deb9be5cddf501bf`.
- Existing private SSD SQLite history received append-only STARTED/FAILED
  entries. Past results were not rewritten; production eligibility is false.

## Storage and Git boundary

The mounted replacement drive is `/Volumes/Tyler HDD 6TB` (APFS HDD). The
approved new source checkout is `/Volumes/Tyler HDD 6TB/miki-engine`; migrate
only into a previously absent path, verify the committed source and preserve
the older `/Volumes/Tyler HDD 6TB/strata` checkout. Never access the failed
`/Volumes/Tyler HDD` or follow old symlinks.

Migration completed into the previously absent new path using a local Git
clone with no hardlinks. Git object checking found no corruption (an existing
dangling local commit is preserved). The 224-file protected source inventories
match exactly between source and destination; protected-manifest SHA-256:
`6b812e509e887949a14d60d41ae5208618b2a6ca6f46f704c520b7f9626edc3d`.
The older HDD checkout remains clean at `76b8d0d`. The new checkout's public
origin was configured and fetched read-only; the feature branch has no upstream
because its new local commits have not been pushed.

Keep the validated environment at
`$HOME/Documents/strata-recovered/.venv`, and models, active native
builds and SQLite on internal SSD. Cold project caches/archives can be moved
only with verified recovery copies. X9 was not mounted during this iteration;
do not create its mount path or pretend HDD capacity restores SSD headroom.

The cold project `.cache/pip` was moved to the mounted HDD with 671 verified
entries and 100,839,840 file bytes (about 96 MiB). SHA-256, metadata/xattrs,
ACLs, links, source stability and no-open-files checks passed before removing
only that exact SSD cache. The private recovery directory contains an immutable
verification manifest and removal receipt. Unrelated CommonCompute Xcode caches
remain untouched pending the separate permission question.

The existing public origin `https://github.com/Ikaikaalika/strata.git` remains
unchanged. Work is locally committed after validation; no remote repo rename,
push, package publication, provider-app change or deployment is authorized by
the naming answer. Previous empirical histories are not rewritten after the
protected source/branding changes; a future experiment freezes a new identity.

## Next development order

1. Restore stable SSD headroom without moving weights/environments or active
   compiler state to HDD; retain reserve and memory/thermal admission.
2. Freeze the new source checkpoint and run the pinned small Qwen MLX control
   with serialized fresh-process pairs. Reject noisy/incorrect candidates.
3. Connect lifecycle ownership to a bounded MLX executor in an isolated worker,
   then qualify one native prefill/FFN family-specific segment end to end.
4. Requalify private ANE compile/dispatch/readback precision for that exact
   segment; use ANE only if whole-model correctness and a matched measured win
   pass. Keep CPU/GPU fallbacks; no blanket ANE acceleration claim.
5. Only after lifecycle, correctness and performance evidence, propose a disabled
   CommonCompute provider canary. Full model-family/VLM coverage, continuous
   batching, SSD paging and signed fleet releases remain later work.
