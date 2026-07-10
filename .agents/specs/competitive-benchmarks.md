# Spike: cross-backend competitive benchmark matrix

**Rows:** `BENCH-*`, `BACKEND-*` · **state:** baseline protocol ready; concrete
backend/model recipes are leaf spikes.

The vLLM oracle remains mandatory. This spike adds native engines so a backend
cannot be called complete while materially slower than the ecosystem's strong
implementation for that hardware.

## Reference set

| ID | Target | Reference | Why it is in the matrix | Source/tool |
|---|---|---|---|---|
| `BENCH-CUDA-VLLM` | CUDA, all concurrency | pinned production vLLM | compatibility + large-concurrency floor | `vllm bench throughput`, `vllm bench serve` |
| `BENCH-CUDA-SGLANG` | CUDA, concurrency 1/2/4/8/16 | SGLang `v0.5.13` (`28b095c`) initially | current stable low-concurrency candidate; Qwen3.6 entered v0.5.11 and v0.5.13 adds faster Qwen3.5 Blackwell GDN kernels | [release](https://github.com/sgl-project/sglang/releases/tag/v0.5.13), [official bench serving guide](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md) |
| `BENCH-CPU-LLAMA` | CPU/GGUF | same-file llama.cpp | quantized CPU and GGUF maturity floor | `llama-bench`, server completion benchmark |
| `BENCH-VK-LLAMA` | Vulkan/GGUF | same-file llama.cpp Vulkan | Vulkan maturity floor | `llama-bench` + server |
| `BENCH-APPLE-OMLX` | Apple MLX | oMLX `v0.5.0rc1` pinned initially | continuous batching, tiered KV and MLX-native serving | [release](https://github.com/jundot/omlx/releases/tag/v0.5.0rc1), [repo](https://github.com/jundot/omlx) |
| `BENCH-APPLE-MLXLM` | Apple MLX | pinned MLX-LM | primitive/model numerics and offline baseline | [MLX-LM](https://github.com/ml-explore/mlx-lm) |
| `BENCH-APPLE-LLAMA` | Apple Metal/GGUF | same-file llama.cpp Metal | alternate file-compatible native floor | `llama-bench` + server |

Pin exact commits/tags in each result. A release candidate stays labeled as
such and is re-evaluated when a stable oMLX release supersedes it.

## Canonical workloads

| Workload | Concurrency | Requests | Input/output | Purpose |
|---|---|---:|---|---|
| `serve-low` | 1, 2, 4, 8, 16 | at least `5 * concurrency`, preferably sustained multi-wave | 1024 / 128, ignore EOS, seed 0 | low-concurrency TTFT/TPOT/ITL and scheduler overhead |
| `serve-gate` | existing per-model large-concurrency points | existing gate counts | 1024 / 128, ignore EOS, seed 0 | production throughput floor |
| `chat-short` | 1, 4, 8 | at least 40 | fixed tokenized chat prompt / 128 | interactive latency and template parity |
| `prefix` | 1, 8 | cold + warm repeats | fixed shared prefix + suffix / 128 | prefix-cache behavior; cold and warm never mixed |
| `offline` | 1 and saturation | enough to reach steady state | same token IDs/lengths | kernel/model runner without HTTP tax |
| `serve-ctx` (adopted from spark-bench tier2) | 1 | ≥3 per context | 1k / 8k / 32k input, 512 output | single-stream TTFT/prefill-tps/decode-tps vs context length; catches decode/prefill degradation at long context that the fixed-1024 workloads hide |

SGLang uses `python -m sglang.bench_serving` with JSON output, streaming, fixed
request rate or infinite-rate closed-loop as declared, and the same OpenAI
endpoint/prompt corpus as our/vLLM arms. Its own guide requires enough prompts
for steady state; this protocol uses at least five waves.

## Required metrics

| Higher is better | Lower is better | Preconditions |
|---|---|---|
| req/s, input/output/total tok/s | mean/P50/P90/P99 TTFT, TPOT, ITL, E2E; peak memory | identical prompt tokens, generated token count, correctness/quality |

Report raw per-repetition JSON plus mean, median, range, and ours/reference
ratio. Do not reduce results to one throughput number.

## Per-backend leaf spike requirements

| Item | Required decision/evidence |
|---|---|
| Model | Exact shared checkpoint/quant, or conversion equivalence and why it is non-binding |
| Server | Exact command, port, API flags, cache size and concurrency controls |
| Warmup | JIT/compile/model warmup excluded consistently |
| Profiling | nsys both CUDA references and ours; native profiler equivalent on Apple/Vulkan |
| Isolation | whole multi-arm series under one GPU/resource lock |
| Acceptance | binding floors versus informational leaders declared before seeing results |

## Current execution notes

- DGX: `dgx.casa`, GB10/sm121. All GPU work uses `flock /tmp/gpu`.
- Apple: `192.168.68.103`, M4/16 GB. It is valid for op/small-model and
  representative serving gates, not the 27B/35B scale claim.
- SGLang was not installed at the initial 2026-07-10 audit; provisioning must
  use its own environment and must not mutate `~/venvs/vllm-oracle`.
- The exact SGLang recipe is [spiked](cuda-sglang-low-concurrency.md). Its
  preflight waits for the current PR #3 GPU claim; binding TTFT/ITL execution
  also depends on `SERVE-ASYNC-LLM`. Overlapping data is void.

## Folded: spark-bench (external GB10 operator bench) — what we adopt / ignore

[Weschera/spark-bench](https://github.com/Weschera/spark-bench) (@ `d340b6e`,
2026-07-09, MIT) is a third-party benchmark run on a 4× DGX Spark cluster —
exactly our `dgx.casa` hardware (GB10, 128 GB unified). It is primarily a
model-QUALITY bench ("TrueScore": 74 agentic/tool/code scenarios), but its
non-scoring serving tiers, deployment recipes, and multi-engine matrix scripts
are directly useful to us: it publishes GB10 serving numbers for our EXACT gate
checkpoints (`nvidia/Qwen3.6-35B-A3B-NVFP4`, `nvidia/Qwen3.6-27B-NVFP4`) under
vLLM, SGLang, llama.cpp, and a third-party native GB10 engine ("Atlas",
`avarok/atlas-gb10`). Sources cited below: `README.md`, `spark_bench.py`
(`run_tier2`, `_concurrency_run`), `recipes/qwen36-35b-nvidia-nvfp4-mtp3.md`,
`recipes/qwen36-27b-nvidia-nvfp4-dflash.md`, `recipes/README.md`,
`recipes/mimo-v25-nvfp4-tp2-1m.md`, `archive/one-off-runs/matrix_driver.sh`,
`results/spark_bench.csv` (every number carries a `run_id`). Live board:
[wesche.com/dgx](https://wesche.com/dgx).

### External GB10 reference envelope (informational, NEVER a floor)

Their tier2 workload shape differs from ours: single closed-loop wave of `n`
streaming chat requests (`_concurrency_run`, ThreadPoolExecutor), ~4 chars/token
filler prompts (~1024 prompt tokens), 512 generated tokens, "agg decode tok/s" =
total output ÷ wall (includes TTFT), no sustained multi-wave. So these numbers
are an ENVELOPE/smell-test for our own GB10 results — never a denominator or
binding floor (our floors stay same-box, same-workload, fresh re-measures).

| Config (single Spark) | c1 decode tok/s | c16 agg tok/s | Source / run_id |
|---|---:|---:|---|
| 35B-A3B **NVFP4** vLLM (eugr 07-06 nightly: marlin MoE, flashinfer attn, fp8 KV, async-sched, util 0.6, 32k ctx, seqs 16), no spec | 78.0 | 357.3 | `recipes/qwen36-35b-nvidia-nvfp4-mtp3.md`; `…eugr0706-parity-throughput-20260708-170657` |
| same, **MTP-3** (checkpoint's built-in heads) | 105.3 | 450.2 | `…recipe-mtp3-throughput-20260708-171225` |
| same, **DFlash K10** (z-lab drafter) | 76.3 | 301 | `…dflash-k10-throughput-20260708-183230` |
| 35B-A3B **FP8** vLLM (3-engine matrix, snapshot `95a723d0`) | 53.5 | 255.0 | `matrix_driver.sh`; `…vllm-aeon023-throughput-20260702-222509` |
| 35B-A3B FP8 **SGLang 0.5.12**, no spec / MTP-NEXTN | 50.9 / 60.4 | — / 251.6 (279 @ c8) | `…sglang0512-nospec…192419`, `…sglang0512-mtp-NEXTN…193115` |
| 35B-A3B FP8 **Atlas** (`avarok/atlas-gb10`) | 52.7 | 54.7 (collapses) | `…atlas-throughput-20260702-232024` |
| 27B **NVFP4** vLLM nightly + DFlash k=10, 262k-ctx config | 34.4 (warm probe) | — | `recipes/qwen36-27b-nvidia-nvfp4-dflash.md` |

Notable evidence for our own tracks:
- **Spec decode (`ROAD-V1-C3`):** on the 35B, MTP-3 beat DFlash-K10 decisively
  (105.3 vs 76.3 c1; DFlash ≈ no-spec baseline at only ~19–36% acceptance) —
  independent confirmation of our MTP→GDN-spec→DFlash routing. Their DFlash k
  sweep on 27B (k∈{8,10,12,15}, k=10 best) shows acceptance-% alone is the
  wrong tuning target; end-to-end tok/s is. Compile+CUDA graphs was worth ~7%.
- **Spec decode is the production operating point on this hardware:** external
  GB10 operators serve MTP-3 as the default config. Once C3 lands, the gate
  bench gains spec-decode-ON arms, and the vLLM denominator must then run ITS
  best spec-decode config on the same checkpoint (mirror + every-axis rule,
  per-config).
- **Denominator hygiene:** their tuned community config
  ([eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker):
  `VLLM_MARLIN_USE_ATOMIC_ADD=1`, `--moe-backend marlin`, `--async-scheduling`,
  fp8 KV, fastsafetensors) is a config our vLLM oracle arm should be
  sanity-checked against — if community-tuned vLLM beats our oracle arm on the
  same checkpoint/workload, our denominator is stale, not our engine fast.

### Adopted methodology (into the leaf-spike requirements above)

From `matrix_driver.sh` (their same-snapshot, same-node, sequential 3-engine
matrix — vLLM vs SGLang vs Atlas):

1. **Cross-engine parity probe before measuring:** identical
   messages+tools payload to every engine; compare `prompt_tokens` fingerprint,
   structured `tool_calls` parseability, and think-tag leakage. A mismatched arm
   is reported per-domain, not silently averaged. We add this to the per-backend
   leaf-spike "Model/Server" evidence for multi-engine arms (SGLang, llama.cpp,
   oMLX).
2. **Warmup shadow pass explicitly labeled `warmup-discard`** and kept in the
   raw record (not deleted) — auditable warmup exclusion. We already exclude
   warmup; adopt the labeled-shadow-run convention so exclusion is visible in
   artifacts.
3. **Teardown-verify between engine legs:** kill containers, verify available
   memory returns to a recorded baseline (leak/hang precursor check), then
   `drop_caches`, before the next leg. Adopt for our multi-engine series.
4. **Provenance snapshot per series:** container image digests, harness git
   SHA, and per-leg `nvidia-smi -q -d TEMPERATURE,POWER` snapshots recorded
   with the results (see benchmark-protocol.md note).
5. **SGLang-on-GB10 provisioning lead** for `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT`:
   the `scitrera/dgx-spark-sglang:0.5.12` image served the 35B FP8 with MTP
   (NEXTN) and DFlash on GB10; their fallback flags when default startup failed
   were `--attention-backend flashinfer --disable-piecewise-cuda-graph`. Our
   pin remains v0.5.13 per the [SGLang spike](cuda-sglang-low-concurrency.md);
   this is an existence proof + known-good container lineage, not a pin change.
6. **Atlas** (`avarok/atlas-gb10`): track as an informational native-GB10
   competitor only — closed provenance, and their data shows it collapsing under
   concurrency (c16 agg 54.7 tok/s vs vLLM 255). Not a floor candidate.
7. **Multi-node RoCE recipes** (`recipes/mimo-v25-nvfp4-tp2-1m.md`,
   `deepseek-v4-flash-2spark-1m.md`): working NCCL/Gloo/Ray RoCE-NIC pinning for
   2–4 Sparks, plus their `tier1` ib_write_bw/nccl-tests interconnect tier.
   Not actionable on single-Spark `dgx.casa`; bookmark for `ROAD-V1-D2`
   (tensor parallelism) hardware acquisition.

### Ignored (with reasons)

- **TrueScore / eval scenario suite** — model-quality benchmark (agentic/tool
  quality of the MODEL); our engine-correctness bar is token-exact vs the vLLM
  oracle, which is stronger for engine work and model-independent.
- **~4 chars/token filler prompts** — tokenizer-dependent prompt sizing; we keep
  the fixed token-ID corpus (their own `prompt_tokens` drift, 1017 vs 1024,
  shows why).
- **Single-wave concurrency as acceptance** — one wave of n requests hides
  sustained-load amortization; our protocol mandates sustained multi-wave. Kept
  only as the reading key for their envelope numbers above.
- **Size tiers / leaderboard mechanics / OpenRouter arms / pixel-grading** —
  operator model-selection concerns, not engine parity.
- **Their "one model per Spark" isolation rule** — already covered (stricter)
  by our `flock /tmp/gpu` whole-series protocol.

## Tests and artifacts to retain

Keep harness scripts, tokenized prompt corpus/hash, raw JSON, server logs,
environment/package lock, nsys summaries, memory traces and exact commands under
the claim's evidence directory. Only concise results and reproduction paths go
in the append-only ledger.
