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
| Architectures | 30 registered, 25+ gated | 130+ | 100+ | 100+ |

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
| Automatic memory sizing (no hand-tuned budget) | ☐ hand-typed block count | ☐ percent, hand-tuned | ☐ | ◐ |
| Memory cap with a pre-flight error instead of an OOM | ☐ | ◐ KV pool only | ◐ | ☐ |

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

The supported set is exactly what the C++ registry registers: every
architecture self-registers from its own translation unit via
`REGISTER_VLLM_MODEL`, and `scripts/check-supported-models.py` gates this list
against the source so it can never drift. Today that is **30 registered
architectures**. Each row names the concrete checkpoint it was gated against and
the honest verdict; per-arch lifecycle caveats are in [STATUS.md](STATUS.md) and
the agent-facing detail is in `.agents/model-matrix.md`.

Gate words: **strict** is token-for-token identical to the vLLM oracle;
**near-tie** is the ratified distributional gate used where vLLM's own greedy is
bf16-non-deterministic; **scaffold** means registered and config/loader-gated
but the forward is not yet a real-checkpoint run. Speed is a separate bar (match
or beat the reference on every axis); most rows are correctness-complete and
speed-pending, which [BENCHMARKS.md](BENCHMARKS.md) tracks.

### Registered architectures

<!-- supported-arch-table:begin -->
| Architecture | Tested checkpoint(s) | Correctness gate | Speed vs reference |
|---|---|---|---|
| `Qwen3_5ForConditionalGeneration` | Qwen3.6-27B (NVFP4, GDN hybrid) | strict 235/235 text, image+video 32/32 vs vLLM 0.25.0 | gate model: at or above vLLM |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.6-35B-A3B (NVFP4, GDN MoE) | strict 315/315 text vs vLLM 0.25.0 | gate model: 0.93x to 1.03x grid |
| `Qwen3ForCausalLM` | Qwen3 dense 0.6B/1.7B/4B/32B, NVFP4A16 | near-tie strict 16/16 vs vLLM 0.25.0 | c1 every-axis parity, c8 decode residual |
| `Qwen3MoeForCausalLM` | Qwen3-Coder-30B-A3B | strict 6/6 vs vLLM 0.25.0 | 11/16 grid cells at or above graphed vLLM |
| `Qwen3VLForConditionalGeneration` | Qwen3-VL-4B-Instruct (image + video) | image strict 32/32, video near-tie vs vLLM 0.25.0 | vision tower 0.57x vs vLLM encode; umbrella pending |
| `LlamaForCausalLM`, `InternLM3ForCausalLM` | Llama-3.2-1B, 01-ai/Yi-Coder-1.5B-Chat, internlm3-8b-instruct | strict 16/16 each vs vLLM 0.25.0 | pending |
| `InternLM2ForCausalLM` | internlm2-chat-1_8b | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `MistralForCausalLM` | Mistral-7B-v0.3 | strict 16/16 vs vLLM 0.25.0 | pending |
| `OPTForCausalLM` | facebook/opt-125m | strict 6/6 vs vLLM 0.25.0 | pending |
| `PhiForCausalLM` | microsoft/phi-2 | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `Phi3ForCausalLM` | microsoft/phi-4 (14B), Phi-3 | strict 16/16 vs vLLM 0.25.0 | pending |
| `GemmaForCausalLM` | google/gemma-1.1-2b-it, unsloth/gemma-2b | near-tie 48/48 vs vLLM 0.25.0 | pending |
| `Gemma2ForCausalLM` | google/gemma-2-2b-it | near-tie 48/48 vs vLLM 0.25.0 | pending |
| `Gemma3ForCausalLM` | google/gemma-3-1b-it | strict 48/48 vs vLLM 0.25.0 | pending |
| `Gemma4ForConditionalGeneration` | Gemma-4 multimodal (unsloth/gemma-4-E4B-it) | text strict, image mm near-tie; audio pending | pending |
| `GraniteForCausalLM` | ibm-granite/granite-3.3-2b-instruct | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `StableLmForCausalLM` | stabilityai/stablelm-2-1_6b | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `MiniCPMForCausalLM` | openbmb/MiniCPM-2B-sft-bf16 | strict 16/16 vs vLLM 0.25.0 | pending |
| `MiniCPM3ForCausalLM` | openbmb/MiniCPM3-4B (MLA) | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `Olmo2ForCausalLM`, `Olmo3ForCausalLM` | allenai/OLMo-2-0425-1B; OLMo-3 (Olmo2 factory alias) | OLMo-2 strict 16/16; OLMo-3 oracle-blocked (vLLM 0.25.0 cannot build it) | pending |
| `DeepseekV2ForCausalLM` | DeepSeek-V2-Lite (MLA) | strict 8/8 vs vLLM 0.25.0 | speed short, attributed |
| `DeepseekV4ForCausalLM` | DeepSeek-V4-Flash GGUF (ds4 q2-imatrix, UD-IQ2) | coherent near-tie vs ds4 oracle (vLLM cannot fit one GB10) | decode beats ds4 1.144x, default on |
| `Glm4ForCausalLM` | GLM-4-9B-0414 | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `Glm4MoeLiteForCausalLM` | zai-org/GLM-4.7-Flash (31.2B, MLA MoE) | near-tie 8/8 vs vLLM 0.25.0 | pending |
| `LagunaForCausalLM` | poolside/Laguna-S-2.1-NVFP4, GGUF-Q4_K, Laguna-XS | byte-exact near-tie (distributional vs vLLM) | vLLM parity+ 1.03x, default on |
| `KimiLinearForCausalLM` | Kimi-Linear-48B-A3B (KDA + NoPE-MLA + MoE) | e2e runs bf16-resident; bf16-regime knobs 106→120/128 (7/8 exact), not STRICT | 1.30 tok/s, default off; residual = device islands (§14) |
| `KimiK3ForConditionalGeneration` | Kimi-K3 (2.8T MoE) | scaffold: registry+config+enumeration gated, forward refuses | HW-infeasible (~1.56 TB); no run |
| `CohereForCausalLM` | Command-R / Cohere (and Cohere2) | scaffold: W0 tiny-random oracle run-verified; real-checkpoint gate blocked | no run |
<!-- supported-arch-table:end -->

### Standalone and non-registered lanes

These run through dedicated forwards, not the `REGISTER_VLLM_MODEL` registry, so
they sit outside the gated list above.

| Lane | Tested checkpoint(s) | Correctness gate | Speed vs reference |
|---|---|---|---|
| Voxtral audio (`VoxtralForConditionalGeneration`) | Voxtral-Mini-3B-2507 | near-tie-robust 16/16 vs vLLM 0.25.0 | decode 0.97x (beats vLLM); encoder TTFT ~17x, pending |
| Whisper audio encoder | openai/whisper-small; whisper-large-v3 (Voxtral cfg) | encoder tower 77/77; large-v3 tower 203/203 | pending |
| MiniMax-H3 DiT (`MiniMaxH3DiTModel`, vllm-omni lane) | MiniMax-H3 (33.1B video+audio) | portable 72/72; t2va+fl2va COHERENT; ref2va NVFP4 grid = the community checkpoint's own quant fidelity, NO loader bug (§8.12); loads GGUF + NVFP4, STREAMS the bf16 13-shard release to device | FP4/Marlin landed; ref2va NVFP4 render blocked on checkpoint quant (needs official modelopt NVFP4), speed pending; no bf16 render yet |
| MTP speculator | Qwen3.6-27B, Qwen3.6-35B-A3B | token-identical to vLLM `mtp` at c1 | ~4% faster c1; +16% output tput (MoE) |
| DFlash block-diffusion | Qwen3 (DFlash draft) | near-tie e2e 27/27 vs vLLM | 2.9x over spec-off, 1.003x vs vLLM DFlash-on |
| DeepSeek-V4 MTP | DeepSeek-V4-Flash (nextn head) | lossless 5/5; real-model weight-blocked | pending |

### Inventoried but blocked

Enumerated in `.agents/model-matrix.md`, not registered, no runnable GB10 gate:

| Architecture | Model | Why blocked |
|---|---|---|
| `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM` | DeepSeek-V3 / V3.2 | 671B, ~642 GiB fp8 vs 119 GiB unified; V3.2 also DSA-indexer dep-blocked |
| `GlmMoeDsaForCausalLM` | GLM-5 (DSA) | ~1404 GiB bf16; dep-blocked (GLM-5.x is DeepSeek-V3.2 verbatim) |
| `MiniMaxM2ForCausalLM` | MiniMax-M2 | ~230B, ~428 GiB bf16, ~4x over the unified pool |

25 of the 30 registered architectures carry a passing correctness gate today;
the rest are honestly marked scaffold or blocked above. vLLM registers 130+ text
architectures, so this is a curated, gated subset, not a breadth claim. Embedding
and reranking models are not yet registered: the engine-side pooler landed, no
model architecture is wired.

## Multimodal

| Input | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Image | ✅ correctness-gated | ✅ | ✅ | ◐ |
| Video | ✅ correctness-gated | ✅ | ✅ | ☐ |
| Audio | ✅ correctness-gated | ✅ | ◐ | ◐ |
| Video+audio GENERATION (MiniMax-H3 DiT, vLLM-Omni lane) | ◐ t2va+fl2va COHERENT on GB10; ref2va NVFP4 grid = the community checkpoint's own quant fidelity, NO loader bug (§8.12); DiT loads GGUF or NVFP4, and streams the bf16 13-shard release to device | ✅ (vllm-omni, BF16-only, no quantized H3 arm) | ☐ | ☐ |
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
| CPU (x86 AVX-512, Arm i8mm) | ✅ | ◐ | ☐ | ✅ |
| Metal (Apple Silicon) | ✅ | ☐ | ☐ | ✅ |
| Vulkan | ◐ | ☐ | ☐ | ✅ |
| ROCm | ☐ (W0 skeleton, HIP never compiled) | ✅ | ✅ | ✅ |
| XPU / TPU | ☐ | ✅ | ◐ | ☐ |

CUDA runtime-verified on GB10 (sm_121a), Jetson Thor (sm_110) and Jetson AGX
Orin (sm_87). sm_110 is a correctness venue only: CUTLASS has no FP4 tensor-core
kernels for it.

Vulkan **runs a model end to end**: `opt-125m` greedy is STRICT token-exact, 6/6
prompts / 96/96 tokens vs the vLLM 0.25.0 oracle, all nine of that model's ops
dispatched natively with **zero provider declines**, on llvmpipe (no Vulkan GPU is
reachable here). Still partial: **16 native kernels**, the other 71 ops fall back
to the portable CPU tier, and quant/MoE/MLA/linear-attention have none at all (MLA
is refused at the platform seam, not mis-routed). **No speed number is owed.**
Build with `-DVLLM_CPP_VULKAN=ON`; off by default.

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
| OpenAI video generation `/v1/videos` (Sora shape) | ✅ `model`/`size`/`seconds` aliases + `GET /{id}/content`; `input_reference` and the `metadata` video/audio references condition the render | ◐ (vllm-omni, its own request shape) | ☐ | ☐ |
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
| Kimi-Linear-48B-A3B (KDA + NoPE-MLA + MoE hybrid) | e2e RUNS (bf16-resident §13); bf16-regime knobs 106→120/128 (7/8 exact), NOT STRICT; default OFF | bf16 residual+island-inputs → 120/128 best (§14); 1 near-tie left. STRICT+speed residual = device islands (per-channel GDN kernel + paged FA2). 1.30 tok/s; vLLM HW-can't-serve bf16 on 1 GB10 |
| Multi-GPU execution | Hardware-blocked | TP proven equal to tp=1 on CPU; no 2-GPU box to run it |
| LoRA end to end | CPU brick landed | Unwired standalone; not usable through the server |
| Multimodal over HTTP | Architecturally blocked | Vision tower lives outside the registered engine forward |
| Embedding / reranking models | Engine side only | Pooler and runner path landed, no model architecture registered |
| ROCm | W0 skeleton, unbuilt | Backend + platform + 1 op (RmsNorm); the HIP sources have never been compiled by anyone (no AMD board here). Open: [ROCM.md](ROCM.md), [#41](https://github.com/mudler/vllm.cpp/issues/41) |
| XPU, TPU | Not started | CUDA, CPU, Metal and Vulkan are the built backends |
| Custom logits processors on CUDA | Open, not root-caused | Segfaults in a CUDA build, 232/232 green on CPU |
| Memory budgeting (`ROAD-V1-MEM`, #83) | Scoped, spike owed | No profiling; KV pool is a hand-typed `--num-blocks`. Target: auto-size to the declared workload, optional total-footprint cap, refuse before allocating |

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
