# Features

What vllm.cpp supports, next to the engines it is measured against. This page is
a **keyed table**: one row per feature, kept current. It is not a changelog.

For measured speed see [BENCHMARKS.md](BENCHMARKS.md); for per-capability
lifecycle state and the caveats behind each row see [STATUS.md](STATUS.md); for
the agent-facing parity inventory with upstream file references see
[.agents/feature-matrix.md](../.agents/feature-matrix.md).

**Legend.** ✅ supported and gated. ◐ partial, usable with named gaps. ☐ not yet.
n/a means the feature does not apply to that engine's design.

Reference versions: vLLM 0.26.0.dev0, SGLang v0.5.15, llama.cpp `237ad9b96`,
MLX-LM as of 2026-07. Competitor columns describe what those projects ship, and
are our reading of their documented behavior, not measurements.

## At a glance

| | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Language | C++20 | Python + CUDA | Python + CUDA | C/C++ |
| Runtime deps | none | PyTorch | PyTorch | none |
| Install size | **66 MiB** | 9.1 GiB | comparable to vLLM | comparable to us |
| Embeddable behind a C ABI | ✅ | ☐ | ☐ | ✅ |
| Weight formats | Safetensors + GGUF | Safetensors | Safetensors | GGUF |
| Correctness gate | token-exact vs vLLM | reference | own | own |
| Architectures | 25+ gated | 130+ | 100+ | 100+ |

## Serving and scheduling

| Feature | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Continuous batching | ✅ | ✅ | ✅ | ◐ |
| Chunked prefill | ✅ | ✅ | ✅ | ☐ |
| Automatic prefix caching | ✅ | ✅ | ✅ (radix) | ◐ |
| Preemption and recompute | ✅ | ✅ | ✅ | ☐ |
| Priority scheduling | ◐ gating | ✅ | ✅ | ☐ |
| LPM cache-aware admission | ✅ | ☐ | ✅ | ☐ |
| In-batch prefix de-prioritization | ✅ | ☐ | ✅ | ☐ |
| Async / overlap scheduling | ✅ default on (UAF-safe drain; device token-ids mirror on gate + classic-dense models; opt-in `VT_ASYNC_EXECUTOR` out-of-capture H2D staging) | ✅ | ✅ | ☐ |
| CUDA graph decode capture | ◐ per-family | ✅ | ✅ | ✅ |
| Partial-prefill concurrency | ☐ | ✅ | ✅ | ☐ |
| Cascade attention | ☐ | ✅ | ◐ | ☐ |

## KV cache and memory

| Feature | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Block-paged KV with refcount and LRU evict | ✅ | ✅ | ✅ | ◐ |
| Hybrid KV groups (full attention + GDN/Mamba) | ◐ | ✅ | ◐ | ◐ |
| Sliding-window and chunked-local attention | ◐ | ✅ | ✅ | ✅ |
| fp8 KV cache | ◐ CPU only | ✅ | ✅ | ✅ |
| KV offload to host memory | ✅ | ✅ | ✅ | ☐ |
| External KV provider ABI (LMCache) | ☐ | ✅ | ◐ | ☐ |
| KV events (block create / evict publish) | ◐ no transport | ✅ | ☐ | ☐ |
| Prefix-cache matching unit | ◐ resolver only | ✅ | ☐ | ☐ |
| Compute directly on quantized blocks | ✅ | ☐ | ☐ | ✅ |

## Quantization and weight formats

| Format | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| NVFP4 (W4A4 and W4A16 Marlin) | ✅ | ✅ | ✅ | ☐ |
| GGUF k-quants and i-quants | ✅ | ☐ | ☐ | ✅ |
| AWQ | ◐ CPU dequant | ✅ | ✅ | ☐ |
| GPTQ | ◐ CPU dequant | ✅ | ✅ | ☐ |
| MXFP4 compressed-tensors | ◐ W4A16 Marlin, mem 2.63x less. gate_up FUSION + decode-graph default-ON; #44 3/3, 32B 6/6. **`VT_MARLIN_DENSE` DEFAULT-ON** (`KERNEL-MARLIN-DENSE-EXEC`): dense marlin 48-CTA, byte-faithful, beats MoE (c8 0.969) | ✅ | ✅ | ☐ |
| fp8 weights | ✅ | ✅ | ✅ | ☐ |
| bf16 / fp16 | ✅ | ✅ | ✅ | ✅ |
| Safetensors direct load, no conversion | ✅ | ✅ | ✅ | ☐ |

## Model coverage

| Family | vllm.cpp | Gate |
|---|---|---|
| Qwen3.6 27B and 35B-A3B (NVFP4, GDN MoE) | ✅ | token-exact, at or above vLLM speed |
| Qwen3 / Qwen2 dense (bf16) | ✅ | token-exact, speed-pending |
| Qwen3-Coder, Qwen3-VL | ✅ | token-exact |
| Llama, Mistral, InternLM2/3 | ✅ | token-exact |
| Gemma 1 / 2 / 3 / 4 | ✅ | token-exact, Gemma-4 multimodal near-tie |
| DeepSeek-V2 (MLA), DeepSeek-V4-Flash | ✅ | token-exact; **DS4-Flash decode beats ds4 1.144x by default** (`VT_V4_RESIDENT_W` default-ON, byte-exact); Phase-2 routed-expert residency measured −3.4%, held default-OFF |
| GLM-4, GLM-4.7-Flash | ✅ | token-exact |
| OPT, Phi, MiniCPM, OLMo-2, StableLM, Granite, Yi | ✅ | token-exact |
| Laguna-S / Laguna-XS 2.1 | ✅ | near-tie; **vLLM parity+ (1.03x) by default** (`VT_LAGUNA_RESIDENT_BF16W` default-ON) |
| Whisper, Voxtral (audio) | ✅ | token-exact |
| Embedding and reranking models | ☐ | engine-side pooler landed, no model registered |

25+ architectures are gated today against vLLM's 130+ registered text
architectures. The full per-model state, including which are oracle-blocked, is
in [STATUS.md](STATUS.md) and `.agents/model-matrix.md`.

The marks above are the accurate ones. The 2026-08-06 live-state audit moved
**10** rows off a stale `ACTIVE` claim to `READY`: 3 model, 3 engine, 2 kernel,
1 quantization and 1 backend. For the MODEL rows `check-model-checklist.py` then
demotes the INTERNAL mark in `.agents/model-matrix.md` from `✅` to `🚧`,
because `✅` is illegal at `READY`. That is a lifecycle-contract artifact:
`READY` is the state contracts' legality floor for a row with no Git-visible
claim, never a statement about capability. Every one of the ten carries in-row
anchors asserting a passing gate. No capability changed, no gate was lost, and
these public marks are deliberately NOT demoted to match.

The same reasoning covers the Multimodal marks below: `Video ✅
correctness-gated` and `Audio ✅ correctness-gated` rest on
`ENG-MM-VIDEO-FORWARD` and `ENG-MM-AUDIO-ENCODER`, two of the three engine rows
the audit moved. Their gates (video-processor 41/41, A2 encoder-tower 203/203)
still pass and are still anchored in `.agents/engine-matrix.md`. No checker
couples this page to that matrix, so the marks stay `✅` by decision, not by
oversight.

## Multimodal

| Input | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Image | ✅ correctness-gated | ✅ | ✅ | ◐ |
| Video | ✅ correctness-gated | ✅ | ✅ | ☐ |
| Audio | ✅ correctness-gated | ✅ | ◐ | ◐ |
| Video+audio GENERATION (MiniMax-H3 DiT, vLLM-Omni lane) | ◐ portable path complete; e2e on real weights (Thor); FP4-resident Marlin-W4A16 routing CPU-landed, GB10 speed pending | ✅ (vllm-omni, BF16-only, no quantized H3 arm) | ☐ | ☐ |
| Multimodal over the OpenAI server | ☐ | ✅ | ✅ | ◐ |

Image, video and audio are correct through the CLI and library. Serving them
over the HTTP API is the named open gap: the vision tower is not yet folded into
the registered engine forward.

## Speculative decoding

| Speculator | vllm.cpp | vLLM | SGLang |
|---|---|---|---|
| MTP (multi-token prediction) | ✅ token-identical, ~4% faster at c1 | ✅ | ✅ |
| Draft model | ◐ CPU brick | ✅ | ✅ |
| Medusa | ☐ spike only | ✅ | ✅ |
| EAGLE / EAGLE3 | ☐ | ✅ | ✅ |
| DFlash block diffusion | ✅ 2.9x over spec-off, at/above vLLM DFlash-on | ✅ | ☐ |
| n-gram / prompt lookup | ✅ 27B 5/5 strict vs vLLM | ✅ | ✅ |
| Other methods (ngram-gpu, suffix, dspark, custom-class, dynamic-k, mlp-speculator) | ☐ inventoried | ✅ | ◐ |

## Structured output and tool calling

| Feature | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| JSON schema constrained decode | ✅ | ✅ | ✅ | ✅ |
| Regex constrained decode | ✅ | ✅ | ✅ | ✅ |
| GBNF grammars | ✅ | ☐ | ☐ | ✅ |
| xgrammar backend | ✅ | ✅ | ✅ | ☐ |
| Jump-forward decoding | ✅ opt-in | ☐ | ✅ | ☐ |
| Tool-call parsers | ✅ 36 families | ✅ | ✅ | ◐ |
| Reasoning-content parsers | ✅ | ✅ | ✅ | ☐ |
| Custom logits processors | ◐ CPU-verified | ✅ | ✅ | ☐ |

## Backends and hardware

| Backend | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| CUDA | ✅ sm_80 to sm_121a | ✅ | ✅ | ✅ |
| CPU (x86, Arm i8mm; A76 assembly correct/default, llama speed gate open) | ✅ | ◐ | ☐ | ✅ |
| Metal (Apple Silicon) | ✅ | ☐ | ☐ | ✅ |
| Vulkan | ◐ | ☐ | ☐ | ✅ |
| ROCm | ☐ | ✅ | ✅ | ✅ |
| XPU / TPU | ☐ | ✅ | ◐ | ☐ |

CUDA runtime-verified on GB10 (sm_121a), Jetson Thor (sm_110) and Jetson AGX
Orin (sm_87). sm_110 is a correctness venue only: CUTLASS has no FP4 tensor-core
kernels for it.

## Serving, API and operations

| Feature | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| OpenAI-compatible `/v1/chat/completions` | ✅ | ✅ | ✅ | ✅ |
| Streaming (SSE) | ✅ | ✅ | ✅ | ✅ |
| Offline batch API | ✅ | ✅ | ◐ | ☐ |
| Prometheus metrics | ✅ | ✅ | ✅ | ◐ |
| Plugin / out-of-tree model registration | ✅ in-tree factory `DONE` + plugin seam | ✅ | ◐ | ☐ |
| LoRA adapters | ☐ CPU brick only | ✅ | ✅ | ✅ |
| Embedding / pooling endpoints | ◐ engine only | ✅ | ✅ | ✅ |
| Flat C ABI for embedding in other languages | ✅ versioned | ☐ | ☐ | ✅ |

## Parallelism and scale-out

Single-GPU today. Every mode below is scoped against one `vt::Communicator`
abstraction, and `world_size == 1` stays byte-identical.

| Mode | vllm.cpp | vLLM | SGLang |
|---|---|---|---|
| Tensor parallel (TP) | ◐ CPU-gated, no 2-GPU run | ✅ | ✅ |
| Collective / process-group abstraction | ✅ CPU + NCCL transport | ✅ | ✅ |
| Pipeline parallel (PP) | ☐ spike written | ✅ | ✅ |
| Expert parallel (EP) + EPLB | ☐ spike written | ✅ | ✅ |
| Data parallel (DP) | ☐ spike written | ✅ | ✅ |
| Context parallel (PCP / DCP) | ☐ scoped | ✅ | ◐ |
| Multi-node | ☐ spike written | ✅ | ✅ |
| PD disaggregation | ☐ | ✅ | ✅ |

## Not supported yet

| Gap | State | Detail |
|---|---|---|
| Kimi-Linear-48B-A3B (KDA + NoPE-MLA + MoE hybrid) | Full-model GB10 e2e RUNS (bf16-resident §13), NEAR-TIE 106/128, pool math CLOSES; default OFF | Full model RUNS on GB10 (bf16-resident): host RSS 1.7 GiB, min-avail 21 GiB, no OOM. Token NEAR-TIE 106/128 (6/8 prompts exact, numerics vs deterministic oracle); STRICT path = device islands + bf16 stream |
| Multi-GPU execution | Hardware-blocked | TP proven equal to tp=1 on CPU; no 2-GPU box to run it |
| LoRA end to end | CPU brick landed | Unwired standalone; not usable through the server |
| Multimodal over HTTP | Architecturally blocked | Vision tower lives outside the registered engine forward |
| Embedding / reranking models | Engine side only | Pooler and runner path landed, no model architecture registered |
| ROCm, XPU, TPU | Not started | CUDA, CPU, Metal and Vulkan only |
| Custom logits processors on CUDA | Open, not root-caused | Segfaults in a CUDA build, 232/232 green on CPU |

## How to read this page

A ✅ means the feature is implemented **and** carries a gate: for model rows that
is a token-for-token comparison against the pinned vLLM oracle on the same
workload, and for engine rows it is a named test in the tree. A ◐ means the code
path exists and works within stated limits, and the limits are named in
[STATUS.md](STATUS.md) rather than glossed. We do not mark a row ✅ because the
code compiles, and we do not mark a competitor ☐ to flatter a column.

Feature parity is not the same as speed parity. Most architectures here are
correctness-complete and speed-pending, and [BENCHMARKS.md](BENCHMARKS.md) says
which is which.

The marks track implementation and gates, not who is working on something. The
2026-08-04 claim triage moved 58 agent-record rows out of `ACTIVE` because
nobody is flying them; the 2026-08-05 device inventory put 11 llama.cpp ggml
backends in scope as inventoried rows. Neither changed a capability, so **no
mark on this page moved**. An inventoried backend is not a supported one, and the same
holds for the 31 architectures inventoried on 2026-08-05. A row's lifecycle state and its support mark
are independent: see [STATUS.md](STATUS.md).
