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
silently extend execution, download models or change acceptance thresholds.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/tylergee/Documents/strata-recovered/.venv/bin/python -m miki family-plan
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
| Llama dense | Llama3.2 1B Q4 pending destination; existing 3B Q4 control now | Dense text path plus first size transfer within the same architecture | Llama4 MoE or 70B/M-series speed |
| Gemma3 text | Catalog Gemma3 1B QAT Q4 | Small catalog text canary | Gemma4 PLE, larger/VLM paths |
| Gemma4 dense | Catalog E2B Q4 | PLE plus local/global attention qualification | 26B MoE or encoder-free 12B/multimodal paths |
| DeepSeek distill | R1-Distill-Qwen 1.5B Q4 research canary | Qwen2 dense distill only | Native DeepSeek MLA, MoE, expert paging or V4 |
| GLM OCR | Catalog GLM-OCR 0.9B Q4, separate VLM oracle/fixtures | OCR encoder/connector/decoder qualification | GLM chat or MoE/MLA/DSA |
| GLM chat | Catalog GLM4.7 Flash Q4; **32 GiB tier** | Fleet target; small shape/component fixtures locally | Full-model M1 result from reduced fixtures |
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
`src/miki/bench/families.py`; native benchmark admission rechecks actual files.

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
