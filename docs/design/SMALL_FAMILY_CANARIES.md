# Small-family canaries on the 16 GiB M1

2026-10-05. Goal: short, repeatable iterations on small real models, then
qualify transfers by actual computation/state semantics. **Scaling well is a
hypothesis, never an assumed benchmark result or automatic fleet promotion.**

`miki family-plan` inventories the local SSD snapshot directories and reports
the pinned canary matrix. Directory presence is not artifact validity; a nominal
catalog memory tier is not live admission. The worker still verifies artifact
hashes, offline provenance, software, pressure and the 40 GiB SSD reserve.
The existing optimizer continues to allow only exact dense Qwen3/Llama Q4
group64 models and its three bounded MLX compiler plans. New rows do not
silently extend optimization, download models or change acceptance thresholds.
Unchanged-MLX baseline-only controls additionally admit pinned Llama1B, Gemma4
E2B text and R1-Distill-Qwen1.5B. These are fresh qualification attempts, not
implemented family-specific native acceleration. Gemma4 vision/audio still
needs MLX-VLM, image/audio fixtures and independent correctness gates.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $HOME/Documents/strata-recovered/.venv/bin/python -m lokahi.lab family-plan
# Optional read-only inventory of every architecture in a generated catalog:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $HOME/Documents/strata-recovered/.venv/bin/python -m lokahi.lab family-plan --catalog $HOME/Documents/commoncomputeai/packages/workloads/src/mlxCatalog.gen.ts
```

## Canary ladder

The CommonCompute checkout was inspected read-only. Its dated catalog includes
small Qwen3/Llama/Gemma, GLM-OCR and larger GPT-OSS/GLM chat lanes. Catalog
availability means a pinned Swift type can be constructed, **not** that Miki
implements that family or that all workloads fit this machine. No catalog
entries or provider app files were changed.

| Family/semantics | Local small canary | What it covers | What it does not prove |
| --- | --- | --- | --- |
| Qwen dense | Qwen3 0.6B Q4, already downloaded | Existing compiler-plan loop, dense GQA/SwiGLU | Qwen hybrid recurrent state, MoE, VLM |
| Qwen hybrid | Qwen3.5 0.8B OptiQ Q4, already downloaded | Future hybrid/recurrent text qualification | Same cache/rollback semantics as Qwen3; Qwen3.8/MoE; vision |
| Qwen3.8 | 27B Q4; **larger-Mac target** | Exact current-family target, hybrid DeltaNet/attention | Resident M1 results from the older 0.8B proxy |
| Llama dense | Llama3.2 1B Q4; existing 3B Q4 control | Dense text path plus first size transfer within the same architecture | Llama4 MoE or 70B/M-series speed |
| Gemma3 text | Catalog Gemma3 1B QAT Q4 | Small catalog text canary | Gemma4 PLE, larger/VLM paths |
| Gemma4 dense | Catalog E2B Q4 | PLE plus local/global attention qualification | 26B MoE or encoder-free 12B/multimodal paths |
| DeepSeek distill | R1-Distill-Qwen 1.5B Q4 research canary | Qwen2 dense distill only | Native DeepSeek MLA, MoE, expert paging or V4 |
| DeepSeek V4 | Flash, 284B total / 13B active; **component/fleet target** | Future CSA/HCA, mHC and mixed FP4/FP8 MoE qualification | Native V4 correctness/performance from a Qwen distill |
| GLM OCR | Catalog GLM-OCR 0.9B Q4, separate VLM oracle/fixtures | OCR encoder/connector/decoder qualification | GLM chat or MoE/MLA/DSA |
| GLM chat | Catalog GLM4.7 Flash Q4; **32 GiB tier** | Fleet target; small shape/component fixtures locally | Full-model M1 result from reduced fixtures |
| GLM5 current | GLM5.3 Flash, 320B total / 18B active; **component/fleet target** | Future sparse/linear attention and mHC native VLM qualification | GLM5 coverage from GLM-OCR or GLM4.7 |
| GPT-OSS | Catalog GPT-OSS20B MXFP4-Q8; **32 GiB tier** | Fleet target; bounded MXFP4/MoE/Harmony fixtures locally | MXFP4 parity from affine Q4, or 120B performance |

This uses the smallest useful **catalog/architecture representative**, not the
lowest parameter count from every historical release. Gemma3 270M exists but
is not the current CommonCompute catalog canary. A modern family's smallest
model can still exceed this machine; OCR/distills must never substitute for
its native chat architecture in the scorecard.

Official model cards confirm Qwen3.5 0.8B and Llama3.2 1B as small releases;
Gemma4 E2B uses effective parameters and has about 5.1B parameters including
embeddings. GLM-OCR is an OCR-specific 0.9B model; GLM4.7 Flash is a 30B-A3B
MoE. These distinctions are why a family name/active-parameter count alone
cannot determine memory or transfer eligibility. Sources:
[Qwen](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[Llama](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct),
[Gemma4](https://huggingface.co/google/gemma-4-E2B-it),
[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR),
[GLM4.7](https://huggingface.co/zai-org/GLM-4.7-Flash),
[GPT-OSS](https://huggingface.co/openai/gpt-oss-20b),
[DeepSeek distill architecture](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit).
Community quantized snapshots use exact 40-hex revisions in
`src/lokahi/lab/bench/families.py`; native benchmark admission rechecks actual files.

## Exact current releases and catalog coverage

The publisher inventory on 2026-10-05 contains Qwen3.8 27B, Flash-Next and
2.4T-A95B general-purpose checkpoints, plus FP8 versions. **No tiny Qwen3.8
chat release was found.** Qwen3.5 0.8B is useful for related hybrid-state work,
but cannot be renamed Qwen3.8 in results. See
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
[the publisher inventory](https://huggingface.co/Qwen/models).

Exact pinned remote file metadata gives **16,054,541,349 weight bytes** for
`mlx-community/Qwen3.8-27B-4bit`, not the current catalog estimate of 4.67 GB.
Its three shards total about 14.95 GiB before runtime/KV/OS memory. The existing
24 GiB catalog tier already excludes this M1. The read-only coverage report
flags the sizing discrepancy; it does not edit CommonCompute or silently lower
admission. Model parameter counts in packed-U32 Hub metadata are not file sizes.

[DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) is
the smaller native V4 release in the checked publisher family: 284B total,
13B active, mixed FP4/FP8 weights. The pinned release has **159,617,149,040
weight bytes**. Active parameters describe per-token work, not total residency.
Neither this M1 nor its ANE can turn that into a full-resident 16 GiB model.
The 256 GiB research planning tier in Miki is not a production fit certification.

[GLM5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is a current native
GLM5 target (320B total / 18B active), not a tiny chat model. Its pinned FP8
release has 328,337,455,672 weight bytes; the 512 GiB research planning tier is
not local support or a fleet fit certification. The smaller GLM4.7 Flash chat
lane and OCR0.9B model cannot qualify its sparse/linear/mHC state semantics.

`family-plan --catalog` covers every `(llm/vlm, modelType)` lane in the supplied
generated TypeScript data without evaluating JavaScript. The current local
CommonCompute snapshot yields **59 rows / 29 lanes**. It selects the lightest
catalog artifact estimate per lane, not the latest or smallest Hub-wide model.
All lanes retain `miki_execution_qualified=false` and
`performance_qualified=false`. In particular, a catalog `loadable` constructor
or nominal memory tier is not a Miki executor/performance qualification.

Additional catalog-fit controls for later adapter work include SmolLM3 3B,
Phi4-mini 3.8B, Qwen3-VL2B, Qwen2-VL2B, Qwen2.5-VL3B, Liquid LFM2-VL1.6B,
GLM-OCR0.9B and SmolVLM2 2.2B. OCR, VLM, hybrid and text workloads remain
distinct. Current MiniCPM4, Mistral/Devstral, GPT-OSS and native GLM chat lanes
have higher catalog memory tiers and remain component/fleet targets here.

The user approved **6 TB HDD cold downloads, then real copies staged on SSD
one at a time before testing**. Keep explicit HF cache/Xet cache locations on
that HDD as well; `hf --local-dir` and `--cache-dir` cannot be combined in the
installed CLI. Verify every selected file against its pinned provenance and
require the 40 GiB reserve after staging. HDD presence never admits inference.

See [the dated iteration evidence](../experiments/MIKI_SMALL_CANARIES_20261005.md)
for actual baselines, rejected candidates and separately measured ANE work.

## Iteration and transfer contract

1. Freeze a clean source/policy checkpoint. Verify a pinned offline artifact,
   tokenizer and chat template before using it. Use no remote custom code.
2. Measure unchanged MLX-LM with resident/offload-disabled weights: cold load,
   warm TTFT, separate prefill/decode, complete fixed-work wall time, peak
   MLX/process memory, pressure/swap and thermal state. Energy is unavailable
   unless separately measured; do not infer it from speed or process CPU time.
3. Compare a small bounded variant with identical tokens/work. The initial
   compiler-plan loop uses exact greedy/logit/KV parity, ≥10 interleaved pairs,
   fresh-process replication, confidence/noise/memory/p95 gates. A failure,
   unsupported state or inconclusive noisy result is preserved, not a winner.
4. Native CPU/Metal/ANE kernel candidates specialize by operation, dimensions,
   quantization, phase and chip. Keep wrappers thin. Reverse-engineered ANE
   must pass compile, dispatch, readback and numerical error limits for the exact
   segment before timing; existing chain precision failures are not waived.
5. Record transfer eligibility for shared primitives only. Recurrent state,
   sliding/local/global attention, PLE, MLA, MoE expert routing, MXFP4, multimodal
   encoders and rollback each require their own correctness gate.
6. Rebenchmark after changing size, prompt/batch/context, quantization, backend,
   chip, macOS, software or residency. Test at least two same-architecture
   sizes before saying a local optimization transfers. This is still not
   evidence for 512 GiB Macs, larger native architectures or production serving.
7. Qualify large weight traffic, KV growth, expert locality, SSD staging and
   ANE handoff/compile amortization on suitable provider hardware. Any reduced
   local fixture is labeled component/synthetic and cannot promote a model.

Every result key includes model/revision/tokenizer identity, architecture and
state semantics, dimensions, quantization/group, workload/context/batch, backend
and dispatch proof, chip/software/OS, residency/cache mode, synchronization and
timing boundary. Never extrapolate an M1 token/s number by RAM size or core count.

Downloads require the user's chosen destination and a measured byte-budget
check. Keep active weights/builds/Python/SQLite on SSD. HDD copies are cold
archives; never silently use them for a resident maximum-speed comparison.
