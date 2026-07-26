# vllm.cpp

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team**, the folks behind LocalAI, the open-source AI engine that runs any model (LLMs, vision, voice, image, video) on any hardware, no GPU required.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![LocalAI](https://img.shields.io/badge/LocalAI-Run_Locally-orange)](https://github.com/mudler/LocalAI)

vllm.cpp is a from-scratch C++20 port of [vLLM](https://github.com/vllm-project/vllm) that runs large language models with no Python, PyTorch, or ggml at inference time. It mirrors vLLM's V1 / Model Runner V2 architecture one-to-one (scheduler, paged KV cache, sampler, engine step loop) on top of its own portable tensor runtime (`vt::`), and ships llama.cpp-style as a library, a stable C API, an example CLI, and an OpenAI-compatible server. It loads Hugging Face **safetensors** and **GGUF** checkpoints, and runs on CUDA, CPU, Metal, and Vulkan from one source tree.

> **Pre-release, under heavy development.** The engine is correctness-complete and speed-competitive for a specific set of models on one GPU (NVIDIA GB10 / DGX Spark, sm_121a), plus a CPU path that matches or beats llama.cpp on GGUF. Read the honest per-feature state in [Features](#features) and [Supported models](#supported-models) below, and the full evidence in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Capabilities are labelled: *correctness-complete* (token-exact vs the vLLM oracle), *speed-pending* (correct, throughput work in progress), *build-only* (compiles for a target with no runtime proof here), and *hardware-blocked* (cannot run on the hardware available).

The reference numbers below are measured against the pinned **vLLM 0.25.0** oracle on GB10, greedy, same workload, same tokens. Where a claim cannot be measured on the hardware here, it is stated as such rather than implied.

## Features

vllm.cpp implements an intentionally focused subset of vLLM, held to token-for-token correctness against the pinned oracle. One binding current-state line per capability follows; the detailed chronology and evidence live in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and the [`.agents/`](.agents/) record.

| Capability | State | Notes |
|---|---|---|
| Qwen3.6-27B (NVFP4) text generation | Correctness-complete, at/above vLLM speed | Token-exact greedy on GB10; beats vLLM 0.25.0 total throughput at every concurrency (1.007-1.045x), effective parity 115/124 axes |
| Qwen3.6-35B-A3B (NVFP4, GDN MoE) | Correctness-complete, decode at-parity, prefill speed-pending | Token-exact greedy; decode at or beyond vLLM, remaining gap is prefill TTFT |
| Qwen3 / Qwen2 dense (BF16) | Correctness-complete, speed-pending | Near-tie-robust token-exact vs vLLM (Qwen3-0.6B, Qwen3-4B); c1 effective parity, c8 decode residual |
| Qwen3.5-4B plain BF16 direct loading on discrete CUDA | Correctness-complete, speed-pending | Direct ON and OFF are token-identical; direct loading cuts peak/stable host PSS by 72.0%/91.2%. The pre-transplant H32 AOT, decode-graph and ratio-4 FA2 result reached 0.9864x vLLM 0.25.0; the current-main development branch is pending revalidation |
| Qwen3-Coder-30B-A3B MoE (BF16) | Correctness-complete, speed-pending | Near-tie-robust token-exact 6/6; 11 of 16 binding grid cells at or above vLLM |
| Llama-3.x dense (BF16) | Correctness-complete, speed-pending | Near-tie-robust token-exact 16/16 (Llama-3.2-1B); llama3 RoPE scaling |
| Mistral dense (BF16) | Correctness-complete, speed-pending | Paged-engine token-exact 16/16 (Mistral-7B-v0.3) |
| OPT (learned pos-emb, cross-family) | Correctness-complete, speed-pending | Strict token-exact 6/6 (OPT-125m); additivity canary across model families |
| DeepSeek-V2 MLA | Correctness-complete, speed-pending | Token-exact 8/8 (DeepSeek-V2-Lite); 0.86-0.95x vLLM output rate, TTFT faster at c4/c8 |
| GLM-4 dense (sandwich norms, partial rope) | Correctness-complete, speed-pending | Token-exact 16/16 (GLM-4-9B-0414); first GLM-family model; partial interleaved RoPE + Gemma2 sandwich norms + biased qkv |
| GLM-4.7-Flash (MLA + GLM MoE) | Correctness-complete, speed-pending | Token-exact 8/8 (GLM-4.7-Flash, 31.2B); reuses the DeepSeek-V2 MLA stack; first e2e coverage of the q_lora query branch + noaux_tc sigmoid router with routed-scaling |
| Gemma-3 dense (GeGLU, dual rope, sandwich norms) | Correctness-complete, speed-pending | STRICT token-exact 48/48 greedy (gemma-3-1b-it); first Gemma-family model; GeGLU (gelu_pytorch_tanh) + dual per-layer RoPE theta + Gemma-RMSNorm sandwich norms + sqrt(hidden) embed-scale + query_pre_attn_scalar scaling |
| Gemma-2 dense (attn + final logit soft-cap) | Correctness-complete, speed-pending | Near-tie-band 48/48 (gemma-2-2b-it): 44/48 strict on vLLM's greedy + 4/48 at 0.0-nat ties in vLLM's own logits; proves the attention + final logit soft-cap primitives (attn_logit_softcapping 50 + final 30); the inverse of Gemma-3 (both soft-caps, no QK-norm) |
| Gemma-1 dense (the original Gemma) | Correctness-complete, speed-pending | STRICT token-exact 48/48 greedy (gemma-2b); two fused norms/layer, head_dim scale, GeGLU + sqrt(hidden) embed-scale, tied lm_head; no soft-cap/QK-norm/sliding |
| OLMo-2 dense (pure post-norm, full-width QK-norm) | Correctness-complete, speed-pending | Token-exact 16/16 (OLMo-2-0425-1B); first OLMo-family model; ZERO new compute kernel (pure post-norm `norm_after` + full-width QK-norm reuse existing ops); real ByteLevel-tokenizer gate (no BOS) |
| Phi-3 / Phi-4 dense (Llama subclass, LongRoPE) | Correctness-complete, speed-pending | Token-exact (near-tie-robust) 16/16 (Phi-4-mini-instruct: 7 strict + 9 near-tie, 0 divergent) by the ratified root-divergence gate; bigger-dense STRICT anchor phi-4 (14B) 14/16 fully token-exact + 2 exact-tie (0.0 nats); pre-fused qkv/gate_up loader + LongRoPE cache bit-identical to vLLM's |
| Granite-3 dense (Llama + 4 scalar multipliers) | Correctness-complete, speed-pending | Token-exact 16/16 (granite-3.3-2b-instruct); first IBM Granite model; ZERO new compute kernel; embedding/residual/attention/logits multipliers (attention scale 1/64, not 1/sqrt(head_dim)) threaded over the shared dense path |
| StableLM dense (LayerNorm, partial rope, qkv bias) | Correctness-complete, speed-pending | Token-exact 16/16 (stablelm-2-1_6b): 14/16 strict on vLLM's greedy + 2/16 at bf16 near-ties (max gap 0.438 nats), 0 forward-divergent; first Stability AI model; ZERO new compute kernel (nn.LayerNorm with weight+bias + partial NeoX rope 16/64 + merged qkv bias, all reuse existing ops) |
| OLMo-3 dense (dual rope, interleaved sliding window) | Implemented, oracle-blocked | Loads + runs in our engine (dual rope: plain sliding + YaRN full-attn, per-layer sliding window); no SACRED gate: vLLM 0.25.0 oracle cannot run OLMo-3-1025-7B (`KeyError: 'rope_theta'`; transformers 5.13.1 nests `rope_parameters` per layer-type, no flat `rope_theta`; run-verified W0 2026-07-26) |
| InternLM2 dense (fused-`wqkv` interleaved split) | Correctness-complete, speed-pending | Token-exact 16/16 (internlm2-chat-1_8b): 12/16 strict + 4/16 bf16 near-tie (max gap 0.0 nats), 0 divergent; first InternLM model; ZERO new compute kernel (reuses the Llama dense forward; the only delta is a loader-side de-interleave of the fused `wqkv`, which packs q/k/v interleaved by KV-group) |
| Command-R / Cohere dense (`CohereForCausalLM`) | Implemented, gate-blocked | ZERO-new-kernel port grounded in vLLM `commandr.py`: weight-only Cohere LayerNorm + GPT-J full-width RoPE + PARALLEL residual + `logit_scale` + tied embeddings, all reuse; compiles, links, self-registers. No SACRED gate yet (real checkpoints HF-gated, ungated ones tiny-random, GPU box disk-full); oracle run-verified at W0. See docs/BENCHMARKS.md |
| Phi-1 / Phi-2 dense (`PhiForCausalLM`, parallel residual) | Correctness-complete, speed-pending | Token-exact 16/16 (microsoft/phi-2): 9/16 strict + 7/16 bf16 near-ties (max gap 0.25 nats), 0 forward-divergent; the OLDER Microsoft Phi arch, DISTINCT from Phi-3/Phi-4; ZERO new compute kernel (GPT-J parallel residual, LayerNorm-with-bias, biased qkv/dense, partial NeoX rope 32/80, non-gated NewGELU MLP reusing `vt::GeluTanh`, untied biased lm_head); F16 dtype-aware loader |
| Safetensors loading | Supported | Both gate models plus every registered dense/MoE family |
| GGUF loading (F32/F16/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K) | Supported; compute-in-quant on CPU | Weights in six block encodings stay compressed from file to matmul on CPU (no BF16 expansion) |
| CPU backend vs llama.cpp | At or ahead on every axis (GGUF) | Prefill 1.18x ahead, decode at parity, peak memory 1.01x, byte-identical greedy tokens |
| Paged KV cache + prefix caching | Supported | Block-paged full attention, hybrid full-attention + GDN state groups, automatic prefix caching (APC) on by default for dense models |
| KV offload to CPU / disk | Built, opt-in, off by default; the disk connector is engine-refused | CPU and disk tiers with identity-checked blocks, selected by `--kv-transfer-config` (or programmatically) over one abstract KVConnector ABI. Worker-side KV store/load is implemented for the LMCache connector only; the CPU/disk connector is scheduler-side only, so the engine now REFUSES it at construction (a loud error, not silently wrong output). Guide: [docs/KV-OFFLOAD.md](docs/KV-OFFLOAD.md) |
| LMCache client (`lm://` remote KV) | Built, opt-in, off by default; a working, verified external KV cache | Pure-C++ `lm://` client wired as an `LMCacheConnector`, no `lmcache` in-process; keys agree byte-for-byte with a real vLLM+LMCache peer, mismatched blocks refused. Proven in a real OPT-125m loop vs a live `lmcache.v1.server`: connector-ON tokens are BIT-IDENTICAL to the connector-OFF cold run (both after an in-process restart and from a cold second process). See docs/BENCHMARKS.md |
| Sampling | Supported | Greedy, temperature, top-k, top-p, min-p, penalties, allowed-token / bad-word masks |
| Structured output | Supported (subset), engine-enforced | JSON schema, JSON object, regex, choice, GBNF grammar. Constrained decoding runs in the production engine (native grammar backend, per-step logits bitmask) and is reachable from OpenAI `response_format` and the C ABI (ABI v2 `structured_*` fields) |
| Tool-call parsing | 36 parser families / 40 accepted names, streaming | Every vLLM tool parser at the pin except the three Rust/Harmony-backed ones: pure-text parsers ported 1:1, the six engine-backed families reimplemented from their wire formats, all held to the upstream test suites. Selection via `--tool-call-parser` (server), `tool_parser` (C ABI), or template auto-detection; native-syntax forced tool_choice where expressible. Tables: docs/BENCHMARKS.md |
| Reasoning parsing | 7 parsers, streaming | think_auto (auto-detect default: content unless markers appear), deepseek_r1, mistral ([THINK]), minimax_m2 (+append_think), step3, olmo3 - reasoning split engine-side BEFORE tool parsing, streamed as `reasoning` deltas in the chat chunks |
| OpenAI server | Supported (subset) | `/v1/completions`, `/v1/chat/completions`, streaming SSE, `/v1/models`, `/health`, `/version` |
| Tokenizers | Supported | Byte-level BPE (Qwen/Llama-3/OPT/GPT-2/DeepSeek/OLMo-2) and SentencePiece BPE (Mistral/Gemma), plus GGUF vocab; added-token `lstrip`/`rstrip` whitespace semantics (e.g. Phi-4-mini's special tokens); byte-exact vs the vLLM oracle |

Speculative decoding (MTP, k=1) is available on the Qwen3.5/3.6 checkpoints via `--speculative-config`, at concurrency 1 and above. Not yet supported: multimodal / vision, LoRA, multi-GPU, and the full tool-calling template surface. See [Serving and API notes](#serving-and-api-notes).

## Supported models

Every model below passes a token-for-token correctness gate against the pinned vLLM 0.25.0 oracle on GB10. Where vLLM's own greedy is deterministic the bar is strict token-exact; where vLLM is self-inconsistent at bf16 near-ties, the bar is a near-tie-robust check (our token within half a nat of vLLM's own teacher-forced argmax on our prefix). "Speed" is a separate bar (match or beat vLLM on every axis), tracked in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Architecture | Example checkpoints | Safetensors | GGUF | Correctness | Speed |
|---|---|:---:|:---:|---|---|
| Qwen3.5/3.6 hybrid (GDN + MoE) | Qwen3.6-35B-A3B, Qwen3.6-27B | yes | 35B only | Token-exact | 27B at/above vLLM; 35B prefill-pending |
| Qwen3 / Qwen2 dense | Qwen3-4B, Qwen3-0.6B, Qwen3-32B | yes | dense qwen35 | Token-exact (near-tie-robust) | Speed-pending |
| Qwen3-MoE | Qwen3-Coder-30B-A3B | yes | - | Token-exact (near-tie-robust) | Speed-pending |
| Llama-3.x dense | Llama-3.2-1B | yes | - | Token-exact (near-tie-robust) | Speed-pending |
| Mistral dense | Mistral-7B-v0.3 | yes | - | Token-exact | Speed-pending |
| OPT | OPT-125m | yes | - | Strict token-exact | Speed-pending |
| DeepSeek-V2 (MLA) | DeepSeek-V2-Lite | yes | - | Token-exact | Speed-pending |
| GLM-4 dense | GLM-4-9B-0414 | yes | - | Token-exact | Speed-pending |
| GLM-4.7-Flash (MLA MoE) | GLM-4.7-Flash | yes | - | Token-exact (near-tie-robust) | Speed-pending |
| Gemma-3 dense | google/gemma-3-1b-it | yes | - | Strict token-exact (48/48) | Speed-pending |
| Gemma-2 dense | gemma-2-2b-it | yes | - | Token-exact (near-tie-robust, 48/48) | Speed-pending |
| Gemma-1 dense | gemma-2b | yes | - | Strict token-exact (48/48) | Speed-pending |
| OLMo-2 dense | OLMo-2-0425-1B | yes | - | Token-exact (near-tie-robust) | Speed-pending |
| Granite-3 dense | granite-3.3-2b-instruct | yes | - | Token-exact (16/16) | Speed-pending |
| Phi-3 / Phi-4 dense | Phi-4-mini-instruct, phi-4 (14B) | yes | - | Token-exact (near-tie 16/16; 14B strict 14/16) | Speed-pending |
| Phi-1 / Phi-2 dense (`PhiForCausalLM`) | microsoft/phi-2 | yes | - | Token-exact (16/16; 9 strict + 7 near-tie) | Speed-pending |
| OLMo-3 dense | OLMo-3-1025-7B | yes | - | Implemented, oracle-blocked | Speed-pending |

Compressed-tensors NVFP4A16 (W4A16) dense weights also load and compute natively (RedHatAI/Qwen3-32B-NVFP4A16), correctness-complete and speed-pending.

The **Gemma family**: **Gemma 3** (`Gemma3ForCausalLM`, `google/gemma-3-1b-it`) is the **first landed member** (correctness-complete, speed-pending): STRICT token-exact 48/48 greedy vs the vLLM 0.25.0 oracle. It reuses infrastructure already in the tree (gemma-RMSNorm `(1+w)`, the GLM sandwich norms, the shared dense-attention path, sliding-window attention, tied embeddings) plus one genuinely-new compute kernel, GeGLU (`gelu_pytorch_tanh`), a bf16 embedding-scale multiply, dual per-layer RoPE theta, and query_pre_attn_scalar scaling. **Gemma 2** (`Gemma2ForCausalLM`, `gemma-2-2b-it`) and **Gemma 1** (`GemmaForCausalLM`, `gemma-2b`) have since landed too, both correctness-complete and speed-pending at 48/48 greedy vs the same oracle ([spike](.agents/specs/sweep-gemma.md)): Gemma 2 proves the attention + final logit soft-cap primitives, Gemma 1 the original two-fused-norms block. Only **Gemma 4**, the newest registered variant, remains unimplemented: it needs a large new primitive stack (per-layer embeddings, YOCO KV-sharing, a Gemma-4 MoE) and has only multimodal-wrapped checkpoints, so it is recorded as blocked rather than supported. Its multimodal path (image + video + **audio**, the only audio-capable model in the pin) has been assessed ([spec](.agents/specs/gemma4-multimodal.md)) and is **oracle-blocked**: the Gemma-4 vision and audio towers load through Transformers `AutoModel`, but the pinned oracle's Transformers has no `gemma4` module, so the multimodal path cannot be constructed for a gate; audio, the genuinely-new modality, is instead staged first on the smallest oracle-runnable audio model (Whisper, then Voxtral-Mini-3B on the already-landed Mistral backbone). The **OLMo-2 family** (`Olmo2ForCausalLM` / `Olmo3ForCausalLM`) is the **first landed OLMo member** (correctness-complete, speed-pending): token-exact 16/16 greedy vs the vLLM 0.25.0 oracle on `allenai/OLMo-2-0425-1B` (STRICT 13/16 + near-tie-band 3/16, max gap 0.094 nats, 0 forward-divergent). It is the cleanest dense bring-up yet, needing no new compute kernel: its two distinctive traits both reuse existing infrastructure. The reordered post-norm placement (`norm_after`) is a subset of the GLM/Gemma sandwich norms (the same standalone output-norm plus a plain residual add, without the pre-norms), and its QK-norm is a full-width RMSNorm reusing the existing norm op at a new shape. Its GPT-NeoX ByteLevel tokenizer (which prepends no BOS) makes it a real tokenizer-inclusive gate. `Olmo3ForCausalLM` rides the same class (the 0.25.0 oracle constructs it); the Olmo-3 interleaved sliding-window path has since landed and runs, but is oracle-blocked for a gate (see the table above). Larger DeepSeek / GLM / MiniMax / Gemma-4 variants are recorded as **hardware-blocked** (they do not fit 119 GiB of unified memory on this box) or **spiked-only**, per the [model matrix](.agents/model-matrix.md). The frontier families Kimi / MiniMax / GLM-latest are scoped for mechanical porting in [a dedicated spike](.agents/specs/sweep-kimi-minimax-glm-latest.md): Kimi-Linear-48B is the one that fits GB10 (91.5 GiB) and is e2e-gateable, while MiniMax-M2 (214.3 GiB fp8), Kimi-K2, MiniMax-M3 and GLM-5 remain hardware- or dependency-blocked (honesty-pass only). The matrix opens with an architecture-support checklist (a per-architecture status roll-up covering every engaged model) that a CI checker keeps in lockstep with the detailed rows.

From a **next-tier batch of recent dense text families** ([spike](.agents/specs/sweep-recent-dense-batch.md)), the top three are now implemented (correctness-complete or gating): **Granite-3** (`GraniteForCausalLM`, `granite-3.3-2b-instruct`) is correctness-complete, token-exact 16/16 vs the vLLM 0.25.0 oracle (Llama plus four scalar multipliers, no new compute kernel); **Phi-3 / Phi-4** (`Phi3ForCausalLM`, `Phi-4-mini-instruct`) is implemented and runs but is **not** recorded as a clean pass: it scores 15/16, and although its LongRoPE cos/sin cache is bit-identical to the oracle's, two positions on one prompt fall outside the near-tie band as a cascade after an exact-tie divergence, so closing it needs a cascade-aware gate or a strict check on the larger phi-4 (14B); **OLMo-3** (`Olmo3ForCausalLM`, dual rope plus interleaved sliding window) is implemented and runs in our engine but has no SACRED gate because the pinned vLLM 0.25.0 oracle cannot run the `OLMo-3-1025-7B` checkpoint (its transformers predates OLMo-3's per-layer-type rope schema). **Command-R / Cohere** (`CohereForCausalLM`, parallel-residual with a weight-only LayerNorm and a logit scale) is now **implemented** (a faithful zero-new-kernel port that compiles, links and self-registers) but has **no SACRED gate** yet: every real small `CohereForCausalLM` checkpoint is HF-gated with no token on the GPU box, the only ungated checkpoints are tiny-random (head_dim 8/2, outside the validated attention path), and the GPU box is disk-full; the oracle was run-verified at W0 (arch confirmed `CohereForCausalLM`, not `Cohere2ForCausalLM`). **Phi-1 / Phi-2** (`PhiForCausalLM`, `microsoft/phi-2`) is now **correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 9 strict + 7 bf16 near-ties, max gap 0.25 nats, 0 forward-divergent), the OLDER Microsoft Phi architecture, DISTINCT from Phi-3/Phi-4: a GPT-J parallel-residual decoder (one nn.LayerNorm with bias feeds both attention and MLP), biased q/k/v/dense projections, partial NeoX RoPE, a non-gated NewGELU MLP and an untied biased lm_head, all reusing landed ops (its `gelu_new` maps to the landed `vt::GeluTanh`, so no new compute kernel), with an F16-checkpoint dtype-aware loader mirroring vLLM's f16->bf16 cast. The remaining families are **spiked-only**, scoped and ranked for rapid one-at-a-time bring-up: **MiniCPM** (Llama plus scale-depth and embedding scalars) and **MiniCPM3** (reuses the landed MLA path).

## Performance

Numbers are measured on NVIDIA GB10 (DGX Spark, sm_121a) against the pinned vLLM 0.25.0 oracle, greedy, closed loop, input 1024 tokens to output 128, three interleaved repetitions, ratios direction-normalized so 1.0 or higher passes. The full per-axis grids, memory tables, and exact reproduction recipes are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Qwen3.6-27B (NVFP4), total throughput vs vLLM 0.25.0:**

| Concurrency | Ours (tok/s) | vLLM (tok/s) | Ratio |
|---:|---:|---:|---:|
| 1 | 86.05 | 82.32 | 1.045x |
| 2 | 159.68 | 158.03 | 1.011x |
| 4 | 292.34 | 290.31 | 1.007x |
| 8 | 508.77 | 505.46 | 1.007x |
| 16 | 801.76 | 789.16 | 1.016x |
| 32 | 1095.01 | 1076.25 | 1.017x |

We beat vLLM on total throughput at every concurrency; effective parity is 115/124 per-axis metrics (two-grid totality), with the residuals being noise-band coin-flips or a favorable determinism tradeoff described in the benchmark record. Peak host memory also passes (24.88 GiB vs vLLM 28.18 GiB). The 35B decode path is at or beyond vLLM everywhere; its remaining gap is prefill time-to-first-token, tracked as active work.

**Local Qwen3.5-4B plain BF16 direct loader, speed-pending:** the measured
pre-transplant repair on an RTX 5070 Ti reached 5769.99 total tok/s versus vLLM
0.25.0 at 5849.80 tok/s (0.9864x) on the identical 128-request workload. Direct
loading reduced peak PSS from 8.59 to 2.41 GiB and stable PSS from 8.59 to 0.76
GiB. Mean TPOT/ITL was 43.72 ms versus vLLM's 38.55 ms. Profiling attributes
the residual to the discrete-CUDA sampled-token D2H path synchronizing the main
stream instead of retaining vLLM's event-overlapped device mapping. The repair
has since been transplanted onto current `main`; that development branch is
pending the same-series revalidation, so these numbers do not yet bind to it.
This local 4B diagnostic does not establish 27B/35B support. Exact evidence and
reproduction:
[Qwen3.5-4B main repair](docs/bench-evidence/qwen35-4b-main-repair-20260725.md).

**CPU vs llama.cpp (GGUF, same file, single binary):** prefill 223.8 tokens/s vs llama.cpp 177.3 (1.18x ahead), decode at parity (24.7 vs 25.4 tokens/s), peak memory 2.83 GiB vs 2.80 (1.01x), and the output tokens are byte-identical to llama.cpp's greedy decode.

There is no front-page race clip yet; when one is produced it will follow the LocalAI house style (side-by-side, identical output, honest measured ratios). Until then the numbers above and in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) are the reference.

## Build

vllm.cpp uses CMake (>= 3.24) and a C++20 compiler (gcc 13/14 and clang are exercised; the tree builds -Werror-clean on gcc 14.2). The core has no ML dependencies; the OpenAI server uses a vendored header-only HTTP transport (cpp-httplib).

```sh
# CPU build (the correctness / CI reference). The server is ON by default.
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

```sh
# NVIDIA GB10 build with the vendored fast GDN path.
# Triton-AOT cubins are vendored: Python/Triton is only needed to regenerate
# them (VLLM_CPP_TRITON_REGEN), never to build or run them.
cmake -S . -B build-cuda \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_TRITON=ON
cmake --build build-cuda -j
```

The example binaries land under `build/examples/`: `vllm-cli`, `server`, `vllm-bench`, and `tokenize`.

### CMake options

Read from [`CMakeLists.txt`](CMakeLists.txt). Defaults shown are the shipped defaults.

| Option | Default | Purpose |
|---|---|---|
| `VLLM_CPP_CUDA` | `AUTO` | Build the CUDA backend: `ON`, `OFF`, or `AUTO` (on when a CUDA toolchain is found) |
| `VLLM_CPP_CUDA_ARCHITECTURES` | `121a` | Target CUDA arch(s): `121a` (GB10), `120a` (consumer Blackwell), `120a;121a` (same-family fat binary), `90a` (Hopper, portable-only). The `a` suffix is required for the native fp4 MMA |
| `VLLM_CPP_METAL` | `AUTO` | Build the Metal backend: `ON`, `OFF`, or `AUTO` (on for an Apple host with an ObjC++ compiler) |
| `VLLM_CPP_VULKAN` | `AUTO` (= `OFF`) | Build the Vulkan backend. Opt-in with `-DVLLM_CPP_VULKAN=ON`; headers are vendored and SPIR-V is committed, so no graphics toolchain is needed |
| `VLLM_CPP_MLX` | `OFF` | Build the optional MLX GEMM provider for Metal (needs `-DMLX_ROOT=<mlx install>`) |
| `MLX_ROOT` | (empty) | Root of an MLX install (`include/` + `lib/`) for `VLLM_CPP_MLX` |
| `VLLM_CPP_SERVER` | `ON` | Build the OpenAI HTTP server (needs `third_party/httplib/httplib.h`; disables itself with a warning if absent) |
| `VLLM_CPP_TRITON` | `OFF` | Consume the vendored per-arch Triton-AOT GDN cubins (CUDA only; no Python needed) |
| `VLLM_CPP_TRITON_REGEN` | `OFF` | Maintainer knob: regenerate the AOT cubins with Python + Triton |
| `VLLM_CPP_CUTLASS_DIR` | `third_party/cutlass` | CUTLASS source root (>= 4.5.0) for the sm120a NVFP4 GEMM |
| `VLLM_CPP_CUTLASS_FETCH` | `OFF` | FetchContent CUTLASS 4.5.0 if not found locally |
| `VLLM_CPP_MARLIN` | `ON` | Build the vendored Marlin NVFP4 W4A16 MoE GEMM (sm_12xa) |
| `VLLM_CPP_BUILD_TESTS` | `ON` | Compile and register ctest targets |
| `VLLM_CPP_BUILD_EXAMPLES` | `ON` | Build the example CLI, server, and bench binaries |
| `VLLM_CPP_BENCH_PROFILE_CONTROL` | `OFF` | Trace-only profiler replay control (never for production timing builds) |

Only GB10 / sm_121a is a runtime-gated CUDA target today. `120a` and `90a` are build-supported (they compile and emit real machine code) but unproven here, and non-Apple / non-NVIDIA backends run a subset of operations. See [Acceleration](#acceleration) and the [backend matrix](.agents/backend-matrix.md).

## Running inference (CLI)

`vllm-cli` runs a one-shot completion through the C ABI. Source: [`examples/cli/main.cpp`](examples/cli/main.cpp).

```sh
build/examples/vllm-cli \
  --model /path/to/Qwen3.6-27B \
  --prompt "The capital of France is" \
  --max-tokens 64
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (config.json + tokenizer.json + safetensors) |
| `--prompt "<text>"` | (required) | Prompt text |
| `--tokenizer-config <path>` | (none) | Override `tokenizer_config.json` |
| `--max-tokens N` | `16` | Max tokens to generate |
| `--temperature T` | `0.0` | Sampling temperature (`<= 0` means greedy) |
| `--top-p P` | `1.0` | Nucleus cutoff |
| `--top-k K` | `0` | Top-k (`0` means all) |
| `--seed S` | (unset) | RNG seed (enables seeded sampling) |
| `--stream` | off | Stream token deltas to stdout |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's `--speculative-config` (e.g. `'{"method":"mtp","num_speculative_tokens":1}'`). Unset means no speculation. See [docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md) |
| `-h`, `--help` | | Print usage and exit |

A throughput/latency harness, `vllm-bench` ([`examples/bench/main.cpp`](examples/bench/main.cpp)), takes `--model`, `--dataset-path`, `--num-prompts`, `--input-len`, `--output-len`, `--concurrency`, `--max-num-batched-tokens`, and `--num-blocks`. A tokenizer smoke tool, `tokenize` ([`examples/tokenize/main.cpp`](examples/tokenize/main.cpp)), takes `<tokenizer.json | model.gguf> <corpus.txt>`.

## OpenAI-compatible server

`server` is a small HTTP server speaking the OpenAI API, so any OpenAI client works by pointing its `base_url` at it. Source: [`examples/server/main.cpp`](examples/server/main.cpp) and [`src/vllm/entrypoints/openai/`](src/vllm/entrypoints/openai/).

```sh
build/examples/server \
  --model /path/to/Qwen3.6-35B-A3B \
  --port 8000 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192
```

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.6-35B-A3B", "prompt": "The capital of France is", "max_tokens": 64}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-35B-A3B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

Endpoints (registered in [`src/vllm/entrypoints/openai/api_server.cpp`](src/vllm/entrypoints/openai/api_server.cpp)):

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/completions` | Text completion (JSON or `text/event-stream`) |
| POST | `/v1/chat/completions` | Chat completion (JSON or streaming SSE) |
| GET | `/v1/models` | List the served model |
| GET | `/health` | Process liveness (200) |
| GET | `/version` | Engine version |

Server flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (safetensors or `.gguf`) |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size |
| `--num-blocks N` | `256` | KV blocks |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `8` | Max concurrent sequences (also sizes the HTTP worker pool) |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--scheduling-policy fcfs\|priority` | `fcfs` | Scheduler policy |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--tool-call-parser <name>` | `hermes` | Tool-call dialect (40 accepted names over 36 parser families). `auto` detects it from the chat template, `none` disables tool parsing. An unknown name aborts startup listing every registered name |
| `--reasoning-parser <name>` | `none` | Reasoning parser (`think_auto`, `deepseek_r1`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`). `auto` detects from the chat template, `none` disables. Unknown names abort startup |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's own flag. Unset means no connector. See [docs/KV-OFFLOAD.md](docs/KV-OFFLOAD.md) |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's own flag, e.g. `'{"method":"mtp","num_speculative_tokens":1}'`. Unset means no speculation (byte-identical to the non-spec engine). MTP only, on the Qwen3.5/3.6 checkpoints that ship an `mtp.*` head (safetensors). A malformed or unsupported document aborts startup. Guide: [docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md) |
| `-h`, `--help` | | Print usage and exit |

For a production deployment, use [LocalAI](https://localai.io), which can embed engines like this behind a model gallery, multi-model serving, the full OpenAI API surface, auth, and metrics.

## Consuming it as a library (C API and C++)

Link `libvllm` (static or shared) and include [`include/vllm.h`](include/vllm.h). It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 6`, 19 exported symbols) suitable for `dlopen` / FFI / LocalAI integration.

```c
#include "vllm.h"

vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&mp, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
    return 1;
}

vllm_sampling_params sp = vllm_sampling_params_default();
sp.max_tokens = 64;               // sp.temperature = 0.0 means greedy

vllm_completion out;
if (vllm_complete(engine, "The capital of France is", &sp, &out) == VLLM_OK) {
    printf("%s\n", out.text);
    vllm_completion_free(&out);
}
vllm_engine_free(engine);
```

The ABI covers lifecycle (`vllm_engine_load` / `vllm_engine_free`), blocking and streaming completion (`vllm_complete`, `vllm_complete_stream`), non-blocking concurrent requests (`vllm_request_submit` / `_cancel` / `_wait` / `_done` / `_error` / `_free`), memory helpers, and diagnostics (`vllm_last_error`, `vllm_version`, `vllm_abi_version`). Structured output (ABI v2): set at most one of `structured_json` (JSON-Schema string), `structured_regex`, `structured_choice`/`n_structured_choice`, `structured_grammar` (GBNF), or `structured_json_object` on `vllm_sampling_params` and generation is grammar-constrained per step on every completion entry point. Chat (ABI v3): `vllm_chat` / `vllm_chat_stream` take one OpenAI chat-completions request JSON (messages, tools, tool_choice, sampling) and run the SAME engine-side pipeline as the bundled server: the model's chat template (tokenizer_config.json, or the GGUF `tokenizer.chat_template` metadata) renders the prompt, tool_choice lowers to the structural-tag decode constraint (auto is LAZY: the engine decides when a tool call engages), and tool-call output is parsed engine-side into structured `tool_calls` deltas; the stream callback receives one `chat.completion.chunk` JSON per delta. Multi-turn tool conversations round-trip: assistant-history tool_calls and the tool reply's tool_call_id/name reach the template context. Tool-parser selection (ABI v4): `vllm_model_params.tool_parser` names the tool-call dialect explicitly, and when unset the engine AUTO-DETECTS it from the chat template via an ordered marker table (the llama.cpp `common/chat.cpp` detection idea); an unknown explicit name fails the first chat call with `VLLM_ERR_INVALID_ARGUMENT`. Reasoning selection (ABI v5): `vllm_model_params.reasoning_parser` picks the chain-of-thought splitter the same way (auto-detect: `[THINK]` selects mistral, `<think>` selects think_auto - markerless answers stay pure content, marked ones split R1-identically; nothing detected leaves reasoning parsing off; "none" force-disables). Speculative decoding (ABI v6): `vllm_model_params.speculative_config` takes the same JSON object vLLM's `--speculative-config` does (e.g. `'{"method":"mtp","num_speculative_tokens":1}'`); NULL or empty disables speculation and is byte-identical to the non-speculative engine, and a malformed or unsupported document fails `vllm_engine_load`. Chat templates render through the vendored google/minja engine (the same renderer llama.cpp ships), so full-surface Jinja templates (namespace, macros, filters - e.g. the complete Qwen3.5 template) render faithfully; template resolution still PROBES a render once, and a genuinely broken template degrades - with a stderr witness, never silently per request - to a Hermes-aware plain prompt that keeps structural-tag tool engagement working.

For C++ consumers, the higher-level surface lives under [`include/vllm/`](include/vllm/): `LoadedEngine::FromModelDir(...)` ([`entrypoints/model_loader.h`](include/vllm/entrypoints/model_loader.h)) hands back the synchronous `LLMEngine` ([`v1/engine/llm_engine.h`](include/vllm/v1/engine/llm_engine.h)) or the async `AsyncLLM` ([`v1/engine/async_llm.h`](include/vllm/v1/engine/async_llm.h)) the server itself uses, plus `SamplingParams` and `RequestOutput`. The underlying portable tensor runtime is `vt::` ([`include/vt/`](include/vt/): `tensor.h`, `dtype.h`, `ops.h`, `backend.h`, and friends), which carries no ggml or PyTorch dependency.

## Quantization

| Format | State |
|---|---|
| NVFP4 W4A4 / W4A16 | Both gate-model paths run on GB10, token-exact. FP4 tactics match vLLM; Marlin NVFP4 W4A16 grouped-MoE is the 35B expert path |
| compressed-tensors NVFP4A16 (W4A16), dense | Correctness-complete via the Marlin weight-only path (the same kernel vLLM forces for `use_a16` on sm_121); speed not yet measured |
| GGUF F32 / F16 / Q4_0 / Q8_0 / Q3_K / Q4_K / Q5_K / Q6_K | Supported. On CPU, the six block encodings compute directly on the compressed blocks (no BF16 expansion), byte-identical to the reference path (`VT_GGUF_KEEP_QUANT=0` disables it). GPU builds still expand GGUF weights |
| FP8 (W8A8) | The 35B ModelOpt static per-tensor projection slice is implemented; generic FP8 modes and FP8 KV remain open |
| MXFP4 / MXFP8 | Planned |

## Acceleration

| Backend | Hardware | State |
|---|---|---|
| CPU | x86-64 and arm64 | Correctness / CI reference; at or ahead of llama.cpp on every GGUF axis (prefill 1.18x, decode parity, memory parity), with an Arm i8mm quant-GEMM tier (runtime-detected via auxv on Linux, sysctl on Apple Silicon) |
| CUDA | GB10 / DGX Spark, sm_121a | Gate-model correctness passes; 27B at/above vLLM throughput, 35B prefill-pending. The only runtime-gated CUDA target |
| CUDA | Consumer Blackwell, sm_120a | Build-supported (compiles, emits real sm_120a code, all fast paths resolve) but not runtime-proven here (no such card) |
| CUDA | Hopper, sm_90a | Build-supported, portable-kernels-only (accelerated paths disabled, no Hopper kernel bodies); not runtime-proven here |
| Metal | Apple Silicon | Two models run end to end and pass correctness (OPT-125m, Qwen3-0.6B); 18 of 75 ops native, the rest fall back to CPU on unified memory. No general model support yet; MLX is the named speed floor |
| Vulkan | Portable GPU | Skeleton: 8 ops plus the fusion catalogue run and cross-check against CPU and CUDA. No model runs yet; off unless `-DVLLM_CPP_VULKAN=ON` |
| Intel XPU | Intel GPUs | Spiked, hardware-blocked |
| ROCm / ANE | AMD GPUs / Apple Neural Engine | Post-parity roadmap |

## Serving and API notes

- **Automatic prefix caching (APC)** is implemented and on by default for dense models (hybrid / GDN and attention-free default off, mirroring vLLM). Hit-rate statistics are counted per vLLM's own counters. Some block-hash extra keys (multimodal / LoRA / `cache_salt`) are stubbed, and there is no `/metrics` endpoint yet.
- **KV persistence to disk / CPU offload** is built (CPU and disk tiers, identity-checked blocks, a size-budgeted disk tier) and wired opt-in into the scheduler through an abstract `KVConnector` ABI selected by a `KVTransferConfig` (a disk-offload connector and the LMCache `lm://` client, over the same seam), off by default. The LMCache `lm://` client interoperates with a real vLLM+LMCache peer (its cache keys are byte-for-byte identical to LMCache's own `ChunkedTokenDatabase`, chunk 256, vLLM's `sha256_cbor`, and a KV prefix a real peer wrote is found and loaded back byte-identically over the wire), and it is now proven END TO END inside a real generation loop: with the connector ON against a live `lmcache.v1.server`, an OPT-125m run loads a previously stored prefix's KV and shortcuts prefill, generating tokens BIT-IDENTICAL to the connector-OFF cold run (verified both after an in-process restart and from a genuinely cold second process). The worker-side KV store/load is implemented **for the LMCache connector only**: the CPU/disk `OffloadingConnector` implements the scheduler half alone (it shortcuts prefill for matched blocks, but nothing moves their bytes into the KV pages), so the engine now REFUSES to wire it at construction, on every device, naming the connector and the device, rather than serving output computed over KV that was never written. Both connectors are selectable from the server with `--kv-transfer-config '<json>'`, mirroring vLLM's own flag and JSON shape. The disk connector's worker half remains unimplemented and is the named remaining step. Usage guide: [docs/KV-OFFLOAD.md](docs/KV-OFFLOAD.md).
- **Tool calling** covers the dialect set in the Features table above, including the Qwen3-Coder XML parser (`qwen3_coder` / `qwen3_xml` / `mimo`). Selection over HTTP is `--tool-call-parser` and `--reasoning-parser`, mirroring vLLM's own flag names, alongside the C ABI's `vllm_model_params.tool_parser` / `.reasoning_parser`. Both flags keep the server's previous behaviour as their default (`hermes`, reasoning off), take `auto` to opt into the same chat-template detection the C ABI uses, take `none` to disable, and abort startup listing every registered name if given an unknown one.
- `/health` reports process liveness rather than a full engine-health probe.
- **Speculative decoding (MTP, k=1)** is shipped and user-facing on the Qwen3.5/3.6 gate checkpoints, enabled with `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` on the OpenAI server, the example CLI, and the C API (ABI v6 `speculative_config`). Unset by default and byte-identical to the non-speculative engine when unset (the 27B, 35B, and Coder greedy gates still pass token-exact with the speculative code present). At concurrency 1 our speculative-on greedy output is token-for-token identical to both our own speculative-off output and vLLM's own MTP speculative greedy output, and it is measured about 1.04x faster than vLLM's speculative-on decode (time per output token 66.2 ms prose / 62.95 ms code vs vLLM 69.1 / 65.3), with matching drafter acceptance. The concurrent (multi-request) path, where a speculative request shares a scheduler step with an ordinary prefill, is implemented via a GDN split-merge over disjoint per-request state (proven bit-exact against the two requests run alone). Honest caveats: only MTP at k=1, only the Qwen3.5/3.6 checkpoints that ship an `mtp.*` head (safetensors), and at concurrency above 1 the 27B's greedy output is not bit-stable across batch shapes, so exact token agreement between spec-on and spec-off is a concurrency-1 property only (correctness above that rests on the model-independent bit-exact GDN proof and on matching acceptance, not identical token streams). Usage guide: [docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md); measured A/B in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- **LoRA and multi-GPU** are not supported yet, and **multimodal** is correctness-complete on Qwen3-VL-4B and on our own gate model Qwen3.6-27B (image and video generate end-to-end and match vLLM) but not yet speed-optimized or wired into the OpenAI server. Multimodal (image/video/audio) is the top roadmap priority and the input pipeline has landed (M0/M1): the C++ Qwen3-VL image processor (resize/normalize/patchify to `pixel_values` + `image_grid_thw`), placeholder-token expansion, and the `MultiModalHasher` mm-hash are all bit/byte-identical to the vLLM 0.25.0 oracle on a fixed image, and the engine-side encoder-cache manager plus the inert multimodal request fields are in place (with no image input every path is byte-identical, proven by the unchanged text gates). The **Qwen3-VL vision tower** (`Qwen3_VisionTransformer`: patch embed, 24 ViT blocks with the vision RoPE, patch merger, and DeepStack multi-level features) has since landed (M2a) and is proven faithful to the vLLM 0.25.0 tower on the fixed image, matching every stage (patch-embed, one ViT block, the merger, the DeepStack taps, and the full output) within a measured bf16 tolerance against a dumped reference. The **vision-conditioned text-backbone numeric contracts** (M2b/M2c) have since landed and are unit-green against the vLLM 0.25.0 reference: the multimodal RoPE positions (`get_rope_index`) are bit-exact, the three-section multimodal RoPE application reuses the existing cached-RoPE kernel and is faithful for Qwen3-VL's exact interleaved configuration, and the DeepStack decoder-injection scatter plus the embed-merge masked scatter are bit-exact. **Image-to-text generation now works end-to-end on Qwen3-VL-4B (M2c landed, correctness-complete, speed-pending).** The Qwen3-VL weight loader (`language_model.*` text backbone + `visual.*` vision tower) plus a forked greedy decode (embed the text ids, scatter the vision tower's merger output into the image-placeholder positions, apply the three-section multimodal RoPE, inject the DeepStack features after decoder layers 0/1/2, then run the paged decode) produce, for the fixed image and "What is in this image?" prompt, greedy tokens that match the vLLM 0.25.0 golden **strict token-exact, 32 of 32** (the whole pipeline, from the C++ image processor through the tower and backbone, on-device). This is the first multimodal capability in the tree; it is image-only on this vehicle, speed is not yet measured, and the M3 step reuses this same tower, backbone, and loader to complete Qwen3.6 image and then video. **Image-to-text generation now also works end-to-end on our own gate model, Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`, the GDN-hybrid, M3-b landed, correctness-complete, speed-pending).** Using the vision-inclusive checkpoint (`Qwen/Qwen3.6-27B`, a uniform-bf16 release that includes the vision tower, unlike the text-only NVFP4 quant we run for text), the full pipeline (the C++ image processor, the M2a vision tower with the 27B config, then the forked GDN-hybrid decode) produces, for the fixed image and "What is in this image?" prompt, greedy tokens that match the vLLM 0.25.0 golden **strict token-exact, 32 of 32**. The fork adds only two multimodal-conditioned points to the shared hybrid forward, both gated on multimodal input so a text-only request is byte-identical: it scatters the vision tower's merger output into the image-placeholder positions (the 27B tower has no DeepStack), and it feeds the three-section multimodal RoPE (`mrope_section [11,11,10]`, interleaved) as a per-token cosine/sine cache into the sixteen full-attention layers (the linear-attention layers carry no RoPE). The text gates are unchanged (Qwen3.6-27B 235/235, Qwen3.6-35B 315/315, Qwen3-Coder 138/138 re-run alongside), proving the fork is inert on text. **Video understanding (M3c) has since landed as preprocessing plus full wiring on Qwen3-VL-4B (the new pieces proven correct, end-to-end token-exactness pending).** The genuinely-new piece is video preprocessing (frame sampling, the temporal grid, and the timestamp-interleaved placeholder), since the same vision tower already handles temporal patches and the multimodal RoPE already handles the temporal axis. The C++ video processor's `pixel_values_videos` and `video_grid_thw` are bit-exact to the vLLM 0.25.0 oracle on a fixed clip (41 of 41, red-first proven against a wrong temporal mapping that mismatches ~50% of values), the video multimodal-RoPE positions are bit-exact, and the per-frame windowed vision tower is faithful within the same bf16 envelope as the image tower. **Video-to-text generation now works end-to-end on Qwen3-VL-4B (correctness-complete, speed-pending).** The full greedy run matches the vLLM 0.25.0 golden token-for-token except at a single position; teacher-forcing vLLM on our own generated sequence proves that position is a genuine bf16 near-tie (our token is vLLM's own second choice, 0.125 nats behind, among four tokens tied within a quarter of a nat), and every token after it is exactly vLLM's teacher-forced argmax (the remaining differences versus the greedy golden are just the one-token shift that single tie causes). The vision tower already accumulates in float32 everywhere vLLM does (float32 matrix-multiply accumulation and float32 attention softmax), so this residual is the irreducible bf16 rounding envelope, not a fixable numeric gap; the video gate therefore uses the same near-tie-robust form as the small-dense text models (accept a divergence only when vLLM's own logits cannot separate it from vLLM's argmax by more than half a nat), and it passes. The image gate stays strict token-exact (32 of 32) as the deterministic proof the forward is correct. **Video-to-text generation now also works end-to-end on Qwen3.6-27B (M3d landed, correctness-complete, speed-pending), completing the Qwen video modalities on our own gate model (image and video; Qwen has no audio path).** This is a pure reuse increment: the 27B video driver reuses the video preprocessing, the per-frame windowed vision tower, and the temporal multimodal RoPE proven on Qwen3-VL-4B, and shares the exact GDN-hybrid decode with the 27B image driver (the two differ only in the placeholder mask and the position builder). For the fixed clip and "What is happening in this video?" prompt, greedy tokens match the vLLM 0.25.0 golden strict token-exact, 32 of 32 (here vLLM's own greedy is deterministic, so the gate is strict, and teacher-forcing confirms zero divergence at every position); the 27B image gate re-runs strict 32 of 32 (the shared-driver refactor is byte-identical on image) and the text gates are unchanged by construction. Speed remains unmeasured. **The audio input pipeline has now also landed (audio-track A0/A1), the first step toward audio understanding (the genuinely-new modality; Gemma-4 audio is oracle-blocked, so audio is stood up on Whisper first).** The C++ Whisper-class audio processor (canonical PCM16 WAV decode, the log-mel `input_features` spectrogram, the audio placeholder-token expansion, and the audio mm-hash) is bit/near-bit-identical to the vLLM 0.25.0 / transformers 5.13.1 oracle on a fixed clip for `openai/whisper-small`: the log-mel matches to a relative L2 of about 2e-7 (well inside the float FFT-versus-DFT envelope), and the placeholder ids and mm-hash match exactly. It is inert when no audio is present (the text, image, and video paths are byte-identical). audio encoder tower has since landed too (audio-track A2), proven faithful in isolation.** The C++ encoder (a two-layer convolutional frontend that halves the 3000 mel frames to 1500 tokens, a fixed sinusoidal position embedding, twelve pre-norm bidirectional transformer blocks, and a final layer norm) reproduces the transformers 5.13.1 `WhisperEncoder` on the same fixed clip within a measured bf16 tolerance against a dumped reference: the convolutional frontend and first block match tightly (relative L2 about 0.5%), and the final 1500x768 encoder output matches to about 3% (the smooth bf16 accumulation envelope across twelve layers between two independent kernel stacks, the same per-layer envelope the vision tower measured), with red-first proof that a wrong convolution stride, a missing position embedding, or a skipped final layer norm each blow the tolerance by 70x or more. It reuses the existing matrix-multiply and attention ops (the convolutions are expressed as im2col plus the existing GEMM, so no new kernel), and is additive, so the text, image, video, and audio-input gates are byte-identical by construction. **End-to-end audio-to-text generation now works too (audio-track A3, correctness-complete, speed-pending): the first actual audio understanding in the tree, on `mistralai/Voxtral-Mini-3B-2507`.** The full C++ pipeline (the log-mel spectrogram, the Whisper-large-v3-class encoder at Voxtral's config, a four-frame downsample plus the audio-language adapter projector, a masked-scatter merge into the already-landed Mistral/Llama decoder, then a forked greedy decode) transcribes a fixed clip and matches the vLLM 0.25.0 greedy golden token-for-token up to the first genuine bf16 near-tie: the decoder is proven token-exact (feeding vLLM's own audio embeddings reproduces all 48 tokens), and because our audio encoder uses different bf16 matmul/attention kernels than vLLM (so bit-exact is infeasible, exactly as the encoder tolerance notes say), the binding gate is the same near-tie-robust form used for the video and small-dense text models: teacher-forcing vLLM on our own generated sequence shows every one of our tokens is vLLM's own argmax (the single point where greedy branches is a four-way exact tie), so every divergence is within a hundredth of a nat. Loading the model required reproducing the one transform vLLM applies to the Mistral consolidated checkpoint that its own code does not spell out: the query/key weights are permuted from the interleaved rotary layout to the NeoX layout (verified bit-exact against vLLM's loaded weights), without which the decoder produces the wrong tokens. It is additive and gated on audio input, so the Mistral text gate and the image/video/audio-input gates are byte-identical (re-run and unchanged). **Multimodal speed has now been measured at concurrency 1 against vLLM 0.25.0's production graphed config ([spec](.agents/specs/multimodal-speed.md)):** on Qwen3.6-27B image, the decode is already at vLLM parity (225 vs 227 ms/token, marginally faster) and the language-model prefill is at parity, so the measured gap was the vision encoder tower (about 2.1 seconds versus vLLM's sub-quarter-second encode), which dominated time-to-first-token; the mm decode is not host-bound at this concurrency because a 27B decode is memory-bandwidth-bound and the per-token weight streaming hides the driver overhead. That tower gap has since been closed and beaten: profiling attributed 99% of the tower time to the naive attention kernel (not weight marshalling or the GEMMs), and a warp-scoped online-softmax attention kernel plus a one-time resident-weight load bring the per-image tower from 2114 ms to 148 ms (14.3x), which is 0.59x of vLLM's eager encode (faster), while the strict image and video token-exact gates still hold 32 of 32 (the new attention kernel is a separate op, so the text and audio paths are byte-identical and the 27B text gate re-runs 235 of 235). Video shares the same decode (at parity) and tower (the gap). For audio the vLLM denominator is captured (Voxtral-Mini-3B at about 43 ms first token, 41 ms per token), but the our-side audio timing is not yet measured (build-blocked on a full disk); because that decode is cheap, audio is where the eager single-sequence driver's per-token host overhead could actually matter. The mm paths are single-sequence eager drivers (no batched or graphed mm serving yet), so higher-concurrency throughput is not yet measurable on our side. Design and status in [.agents/specs/multimodal-track.md](.agents/specs/multimodal-track.md), [.agents/specs/audio-track.md](.agents/specs/audio-track.md), and [.agents/specs/multimodal-speed.md](.agents/specs/multimodal-speed.md).
- **Environment variables.** A handful of behavior-changing knobs (CPU thread count, prefix-cache hash seed, LMCache host/port, the default-on async and CUDA fast paths and their rollbacks, the portable-reference bisect switch, GGUF loading behavior, the Vulkan device selector) are documented in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). A CI check (`scripts/check-env-doc.py`) keeps that reference honest: any new production env var must be documented or explicitly classified kernel-internal.

## Verification and parity

Every model is gated token-for-token against the pinned vLLM 0.25.0 oracle on the same workload, and every change that could affect correctness or performance is compared apples-to-apples against vLLM with both numbers and the ratio recorded. Behavioral CPU tests run under CTest; CUDA correctness, sanitizer, trace, and performance evidence is recorded per feature rather than inferred from source. The protocol is in [`.agents/gates.md`](.agents/gates.md) and [`.agents/benchmark-protocol.md`](.agents/benchmark-protocol.md). A CI check (`scripts/check-fusion-consistency.py`) additionally keeps model forwards routing their fusable add+RMSNorm glue through the portable fusion catalog rather than hand-fusing it.

## Why vllm.cpp

vLLM is an excellent serving framework, but running it drags in a heavy Python / PyTorch / CUDA stack. vllm.cpp is a from-scratch C++20 port focused on inference:

- **No Python at inference.** A single `libvllm` behind a flat C ABI (`include/vllm.h`), easy to embed from C, C++, Go, or Rust, or to `dlopen` from LocalAI.
- **One source tree, many backends.** CUDA, CPU, Metal, and Vulkan from the same code, loading safetensors and GGUF.
- **Faithful to vLLM.** The V1 / Model Runner V2 architecture is mirrored one-to-one so upstream vLLM changes port mechanically, and correctness is held token-for-token against the pinned oracle.
- **Honest numbers.** Every capability is labelled correctness-complete, speed-pending, build-only, or hardware-blocked, and the evidence is in the record.

## Project record

The canonical project record lives under [`.agents/`](.agents/), indexed by [AGENTS.md](AGENTS.md). This README is a human-readable current-state snapshot, not a chronological log: detailed status and evidence live in [`.agents/state.md`](.agents/state.md), the [parity ledger](.agents/parity-ledger.md), the [area matrices](.agents/model-matrix.md), and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Citation

If you use vllm.cpp, please cite this repository and the upstream vLLM project:

```bibtex
@software{vllm_cpp,
  title  = {vllm.cpp: a C++ inference engine porting vLLM},
  author = {Di Giacinto, Ettore},
  url    = {https://github.com/mudler/vllm.cpp},
  year   = {2026}
}
```

vLLM is by the vLLM project ([vllm-project/vllm](https://github.com/vllm-project/vllm)). Model weights are governed by their own licenses, so check each model card.

## Author

Ettore Di Giacinto ([@mudler](https://github.com/mudler)).

## License

vllm.cpp is released under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for third-party attributions. The model weights keep their own licenses.

---

Built by the [LocalAI](https://github.com/mudler/LocalAI) team. If you want to run LLMs (and vision, voice, image, and video models) locally on any hardware with an OpenAI-compatible API, [give LocalAI a star](https://github.com/mudler/LocalAI).
