# Miki small-family iteration — 2026-10-05

Physical Apple M1, 16 GiB, macOS 26.6.2 (25G83), Xcode 26.3,
MLX 0.32.2 / MLX-LM 0.31.3. **No accepted full-model speedup in this round.**
Historical evidence remains immutable; later source changes are new checkpoints.

## Real Qwen control and bounded proposal round

Clean source: `1b4805112123f8156d44a9495a59400a871352c8`.
Pinned `mlx-community/Qwen3-0.6B-4bit`, revision
`73e3e38d981303bc594367cd910ea6eb48349da8`, resident internal-SSD weights,
offload disabled, batch 1, exactly 32 generated tokens per request, three
warmups and ten measured samples per workload. Public repeated-text prompts;
MLX-LM `generate_step` through final synchronization, without chat template,
token-to-text, CommonCompute queue/network or energy measurement.

Control experiment `3586ecb1b84d43b1a4a72366fc0a1617`:

| Prompt tokens | Median TTFT | Median decode | TTFT CV |
|---|---|---|---|
| 64 | 103.18 ms | 74.94 tokens/s | 0.357; noisy |
| 256 | 231.50 ms | 83.14 tokens/s | 0.055 |

The load boundary was 1,356.70 ms, excluding artifact hashing. Separate
fresh-cache model-call prefill probes were 82.04 ms / 264.91 ms; these are
different boundaries and cannot be added to or substituted for TTFT.
Peak MLX memory medians were 515,932,648 / 620,070,132 bytes. Process lifetime
high-water was 500,711,424 bytes, not total unified-memory residency.
Baseline self-consistency is not an independent model-quality evaluation.

Raw control SHA-256:
`234a06c64fec59f498184fd468b5f23a0c9ed78dd59d6ee13067a84c141e520c`.

Bounded session `4dfe66bf0cdf437d92eb6f31a0705a61` ran a fresh control,
Codex/Astra hypothesis generation and independent adversarial critique, then
two candidates with fresh-process replication and protected deterministic gates.
Population 2, one generation, TTFT objective, 600-second maximum;
actual elapsed 281.41 seconds. Inference remained local and offline.

| Candidate | Plan | Deterministic outcome |
|---|---|---|
| `0d0f87af33de4935ac089fbaa6565928` | `compile_decode` | REJECT: fresh-process peak memory regression |
| `be49ff5951b0420da33ddc2fa8c7fb88` | `compile_prefill` | REJECT: major request latency regression |

Both parent and unchanged-MLX comparisons rejected these plans. Accepted 0,
retained 0; verified ref still identifies the unchanged control checkpoint,
not a speedup. No thresholds, workloads, reference or histories were weakened.
Initial evaluation hashes respectively:
`2d39ec5a0b00f9e123d9e1a67e31733ee9e24652815babcc05a26fbb0bbb1eb6`,
`df5fc07d23beacd86ca7f2d5265bcc758d86575a6eded56e66a9a2b2cdf86a7c`.
Ambient swap was about 14 GiB and some timings were noisy; these are local
observations, not an idle-machine maximum or Darkbloom comparison.

## Separately requalified reverse-engineered ANE component

The current Objective-C probe was rebuilt on SSD with the local SDK. Existing
private API/MIL/IOSurface code is informed by MIT-licensed
[maderix/ANE](https://github.com/maderix/ANE) and
[skyfallsin/ane.cpp](https://github.com/skyfallsin/ane.cpp); attribution remains
in `native/ane/THIRD_PARTY_NOTICES.md`. No Apple proprietary code/binary was
redistributed, and no machine-wide security setting was changed.

The fixed FP16 projection `[64,256] @ [256,256]` passed compile, model load,
dispatch, readback and CPU numerical verification with maximum absolute error
0 (tolerance 0.002). Five warmups, 50 resident dispatches, reused IOSurfaces:
median **0.2815 ms**, p95 0.3370 ms, dispatch boundary `evaluateWithQoS` only.
Compile/load, tensor layout copies, GPU handoff, sampling and full LLM phases
are excluded. Planner eligibility remains false. Prior real FFN/chain
precision failures are not waived by this simple projection result.

Private component receipt SHA-256:
`b00fb16b06fdaf3ba4fc753fca06dd692dc287bb9f0011184abfcd858e12ef3b`.

## Family coverage and staging

The [canary design](../design/SMALL_FAMILY_CANARIES.md) separates exact new
Qwen3.8/V4 releases from older tiny proxies. Read-only catalog coverage sees
59 rows and 29 `(kind, modelType)` lanes in the current local CommonCompute
generated catalog, SHA-256
`fe471b6461e78495a6cdc414e2794a6539f25acf26fecfdc61f5228e2e771c68`.
This is a dirty-checkout source snapshot, not deployed provider availability.
No CommonCompute file was edited. Qwen3.8's pinned remote weights total
16,054,541,349 bytes, contradicting its 4.67 GB catalog estimate.

New weights download to `/Volumes/Tyler HDD 6TB/Miki-Engine-models`, with HF
and Xet caches also explicitly on that drive. Cold Gemma4 E2B contains eight
selected metadata/weight files, 3,583,086,498 bytes, all verified against the
exact revision and remote SHA/Git-blob provenance. Hub verification warnings
refer to intentionally omitted README/.gitattributes and local HF bookkeeping,
not omitted weights. The full 3.55 GB weights are present.

Llama3.2 1B also downloaded cold and was staged into a previously absent real
internal-SSD directory. Its cold and staged content/provenance records match.
The reserve passed before and after staging; no existing model was removed.
Staging receipt SHA-256:
`aab8a71c6b605f3bebf2d9906fcd81918ef2ca5e89c3e0de3565234dbc86b577`.
Gemma4 SSD staging is pending reserve headroom. No native V4, Qwen3.8,
GLM chat or GPT-OSS weights were downloaded or inferred on this M1.

## Cache recovery work completed

Three specifically approved inactive CommonCompute Xcode DerivedData folders
were cold-archived and removed from SSD only after content hashes,
metadata/xattrs/ACLs/links, source-stability and no-open-files checks passed.
100,570 entries / **7,571,623,801 logical file bytes (about 7.05 GiB)**.
APFS sharing/snapshots and unrelated machine activity mean this is not a
claim of equal net free-space gain. Active caches, Python and models stayed.

Recovery root:
`/Volumes/Tyler HDD 6TB/Miki-Xcode-cache-archive-20261005.Ev0GiB`.
The first two recovery copies are qualified directories. The third preserves
the original provenance xattr in a verified PAX archive: the OS reissued that
xattr in the auxiliary directory copy, so that copy was not treated as exact.
The private removal receipt identifies each original and verified manifest.
Archive SHA-256:
`4ad0e875ebe21b9ba00ca8d093b89548146eb7299abb678b74d13ba138fab36a`.
All three approved originals are absent and recoverable from that root.

## Next gates

New baseline-only controls do not extend the candidate allowlist. Freeze and
validate the new source before attempting Llama1B/Gemma4/R1-distill controls.
Gemma4 text must retain PLE/local-global/shared-KV semantics; image/audio
paths need separate VLM oracle/fixtures. Native Metal Q4/FFN, CPU layout/cache
reuse and coarse ANE subgraphs remain the low-level search priorities.
Changed shapes/quantization/state/chips require new evidence. Never assume
small-model wins scale, enable SSD paging in resident results, or promise
fastest inference across all models from this machine.
