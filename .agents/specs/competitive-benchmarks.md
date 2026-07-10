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

## Tests and artifacts to retain

Keep harness scripts, tokenized prompt corpus/hash, raw JSON, server logs,
environment/package lock, nsys summaries, memory traces and exact commands under
the claim's evidence directory. Only concise results and reproduction paths go
in the append-only ledger.
