# Lōkahi agent instructions

Lōkahi (formerly Strata) is an Apple-Silicon LLM inference runtime and learning
laboratory owned by Common Compute LLC. The Python package is `lokahi`; the old
`ollm` namespace was removed without a compatibility alias. The Miki empirical
optimizer now lives at `src/lokahi/lab` (`python -m lokahi.lab`); the
`miki_engine` facade was dropped. Keep `strata_*` C ABI/wire IDs and the
on-disk lab identifiers (`miki_candidate.json`, `miki/verified-*` refs,
`~/Library/Application Support/Strata/miki/`): recorded history refers to them.

## Engineering boundaries

- The primary integrator freezes shared contracts before parallel work.
- Specialist agents receive exclusive writable paths. Do not edit outside the
  assigned paths or commit from a specialist lane.
- Preserve unrelated dirty work. Do not reset, clean, stash, or rebase it.
- Lōkahi is built in public at `https://github.com/commoncompute/lokahi`.
  Publish reviewed source, tests, designs and non-sensitive experiment evidence.
  Keep private fleet records, credentials, model weights, environments and
  unaudited recovery candidates out of Git. See `docs/BUILD_IN_PUBLIC.md`.
- Do not download packages, repositories, models, or large data until the user
  chooses the destination.
- Use generated small fixtures before real model weights.
- The original `/Volumes/Tyler HDD` failed. Do not access it, including through
  preserved symlinks. `/Volumes/Tyler HDD 6TB` replaces it for source checkouts
  and cold archives only: it is an HDD. Never use it as an SSD offload target,
  place model weights or caches there to represent SSD behavior, or run storage
  qualification benchmarks against it. If it is unmounted, stop and report
  rather than creating a fake mount directory.
- Keep builds, active caches and performance storage on the internal SSD. The
  approved Lōkahi SSD root is `~/lokahi-ssd` (60 GB cap); the Strata model root
  is `~/Library/Application Support/Strata/models`. Keep a 40 GiB free-space
  reserve. SSD placement does not by itself qualify production model paging.
  Other contributors choose their own SSD locations.
- `/Volumes/X9 Pro/Strata-Miki` is approved for project artifacts when
  mounted. It is exFAT: keep Git worktrees and SQLite on APFS, and Python and
  active native builds on the internal SSD. Do not format it, move unrelated
  data, or count its free space toward the internal-SSD reserve.

## Lōkahi lab empirical loop

- Correctness precedes performance. LLMs propose and critique; protected native
  workers and deterministic software measure, judge and retain local checkpoints.
- Freeze evaluation infrastructure in a clean local commit before evolving.
  Candidates may change only the candidate file in isolated Git worktrees.
  Benchmark logic, references, workloads, statistics, test expectations, policy,
  controller and history are not optimization targets. Changes invalidate a run.
- Preserve failures, inconclusive results, candidate diffs and raw hashed records.
  Never rewrite historical evidence. Keep SQLite/history private and append-only.
- Require real physical macOS hardware, matched artifact/software identities,
  3 warmups, at least 10 interleaved pairs, independent fresh-process replication,
  fixed correctness limits and the protected noise/memory/confidence gates.
  No simulated-chip, microbenchmark-only, fallback-only or native-language win.
- The initial 2% lab acceptance threshold applies only to local experimental
  plans. It does not weaken the full-runtime production/fleet gates.
- Native/private ANE, AMX and cross-device work remain separate research lanes.
  Record source URL, revision/license and documented/reported/measured/inferred
  status. Never redistribute proprietary Apple binaries, firmware or code.
- Keep evolution finite, serialized and interruptible. Stop on resource guards,
  instability or budgets; never change machine-wide security/firmware settings.
- Research providers are replaceable. Model inference itself must remain
  offline and independent of the proposer.

## Evidence rules

Label results as correctness, synthetic, or hardware evidence.

- Correctness evidence compares deterministic outputs to an independent oracle.
- Synthetic evidence cannot support hardware throughput, energy, or SSD claims.
- Hardware evidence records chip, macOS build, shapes, dtype, warmup, iterations,
  units, numerical error, and timing boundaries.
- Core ML configuration or private-framework discovery does not prove ANE
  execution. Require compile, dispatch, readback, and numerical verification.
- An on-disk ANE compile cache is not live weight paging.
- Apple Silicon storage remains SSD to unified memory to the compute engine;
  never claim direct SSD execution without measured proof.

## Local gates

The current known MLX-capable test environment is:

```sh
LOKAHI_TEST_BACKENDS=cpu,metal PYTHONPATH=src ~/lokahi-ssd/venvs/mlx/bin/python -m pytest -q
```

The environment is local, not a portable installation requirement. Do not
install dependencies silently when it is absent; report the gap and obtain a
destination first. Also run:

```sh
PYTHONPATH=src python3 -m compileall -q src tests benchmarks
git diff --check
```

Native probes must be isolated executables and fail closed. Hardware performance
runs are serialized; agents may code in parallel but must not run competing GPU,
ANE, or storage benchmarks concurrently.
