# Feature matrix — the feature-level breakdown of the roadmap

**This document is the feature-level breakdown of the project roadmap
([roadmap_v1.md](roadmap_v1.md) — this matrix is its §E).**
One row per vLLM feature, grounded in [porting-inventory.md](porting-inventory.md)
(the per-subsystem parity record), the B1–B6 research reports, and direct source
verification (2026-07-10, worktree @ a82d262).

**Convention (delegation unit):** every row carries a **Spec** link to
`.agents/specs/<feature-slug>.md`. A gap row becomes delegable work by WRITING the
spec at that path (scope, upstream file:line to mirror, gates, A/B plan) and handing
it to a sub-agent. Most spec files do not exist yet — writing one is how work starts.

**Status legend:** ✅ merged + gated · 🟡 partial (what's missing is in Notes) ·
🚧 spec written / agent running · ☐ gap (not started). Tier (T0–T3) per
porting-inventory; Wave per [specs/expansion-map-2026-07-10.md](specs/expansion-map-2026-07-10.md).

**Mirror policy:** vLLM parity is the FLOOR for every row (mirror-vllm-always);
the surpass track (roadmap_v1.md "Protocol evolution") builds beyond it, never
instead of it.

---

## 1. Engine core & scheduling

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Continuous batching / unified scheduler (token-budget, no prefill/decode split) | `v1/core/sched/scheduler.py` | ✅ T0 | running-first + FCFS preemption + update_from_output | [specs/unified-scheduler.md](specs/unified-scheduler.md) |
| Chunked prefill (on by default) | `config/scheduler.py` | ✅ T0 | chunked==one-shot gated | [specs/chunked-prefill.md](specs/chunked-prefill.md) |
| Prefix caching (APC) | `v1/core/kv_cache_utils.py` | ✅ T0 | sha256_cbor byte-exact; hash_block_size≠block_size + partial-block behind 1:1 stubs | [specs/prefix-caching.md](specs/prefix-caching.md) |
| Preemption (FCFS tail pop, recompute) | `v1/core/sched/scheduler.py` | ✅ T0 | | [specs/preemption.md](specs/preemption.md) |
| CUDA graphs (decode capture/replay, no torch) | `config/compilation.py::cudagraph_mode` | ✅ T0 | batched-graph gates green on both models | [specs/cuda-graphs.md](specs/cuda-graphs.md) |
| **Async/overlap scheduling** | `v1/core/sched/async_scheduler.py` | ☐ T1 (**promoted**) | B3 finding: vLLM's DEFAULT at our pin ⇒ unmet MIRROR obligation (we run synchronous); latency lever | [specs/async-scheduling.md](specs/async-scheduling.md) |
| Priority scheduling (`policy="priority"`) | `v1/core/sched/request_queue.py` | ☐ T1 | | [specs/priority-scheduling.md](specs/priority-scheduling.md) |
| Partial-prefill concurrency (`max_num_partial_prefills`, long-prefill limits) | `config/scheduler.py` | ☐ T1 | | [specs/partial-prefill-concurrency.md](specs/partial-prefill-concurrency.md) |
| `step_with_batch_queue` (pipelined batch queue) | `v1/engine/core.py` | ☐ T1 | | [specs/batch-queue-step.md](specs/batch-queue-step.md) |
| `scheduler_reserve_full_isl`, pluggable `scheduler_cls`, `stream_interval` | `config/scheduler.py` | 🟡 T1 | reserve_full_isl field ✅; pluggable cls → C ABI registry (§9.4) ☐ | [specs/scheduler-knobs.md](specs/scheduler-knobs.md) |
| Cascade attention (shared-prefix batches) | `config/model.py::disable_cascade_attn` | ☐ T2 | | [specs/cascade-attention.md](specs/cascade-attention.md) |
| DBO / ubatch overlap | `config/parallel.py::enable_dbo` | ☐ T2 | | [specs/dbo-ubatch.md](specs/dbo-ubatch.md) |

## 2. KV cache & memory

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Paged KV: BlockPool (free list, refcount, LRU evict) | `v1/core/block_pool.py` | ✅ T0 | | [specs/block-pool.md](specs/block-pool.md) |
| KVCacheManager (allocate_slots, watermark, admission) | `v1/core/kv_cache_manager.py` | ✅ T0 | | [specs/kv-cache-manager.md](specs/kv-cache-manager.md) |
| Hybrid KV coordinator (full-attn + GDN/mamba state groups) | `v1/core/kv_cache_coordinator.py` | ✅ T0 | cross-group MIN-intersection prefix hit; `align` mode stubbed | [specs/hybrid-kv-coordinator.md](specs/hybrid-kv-coordinator.md) |
| Mamba/GDN prefix-cache retention (`mamba_cache_mode=align`, vllm#45845) | `v1/core/` | ☐ T1 | 1:1 stub in place | [specs/mamba-align-retention.md](specs/mamba-align-retention.md) |
| SlidingWindowSpec + ChunkedLocalAttentionSpec | `v1/kv_cache_interface.py` | ☐ T1 | stubs in place | [specs/sliding-window-kv-spec.md](specs/sliding-window-kv-spec.md) |
| fp8 KV cache (`cache_dtype=fp8*`) | `layers/quantization/kv_cache.py` | ☐ T1 | | [specs/fp8-kv-cache.md](specs/fp8-kv-cache.md) |
| nvfp4 / per-token-head / turboquant KV | `config/cache.py` | ☐ T2 | | [specs/nvfp4-kv-cache.md](specs/nvfp4-kv-cache.md) |
| KV offload (CPU tiering, LRU/ARC) | `v1/kv_offload/` | ☐ T2 | | [specs/kv-offload.md](specs/kv-offload.md) |
| KV connectors (Nixl/LMCache/Mooncake, **PD disaggregation**) | `distributed/kv_transfer/` | ☐ T2 | | [specs/kv-connectors-disagg.md](specs/kv-connectors-disagg.md) |
| KV events (block create/evict publish) | `config/kv_events.py` | ☐ T2 | | [specs/kv-events.md](specs/kv-events.md) |
| MLAAttentionSpec (latent KV) | `v1/kv_cache_interface.py` | ☐ T2 | with DeepSeek family | [specs/mla-kv-spec.md](specs/mla-kv-spec.md) |
| Sizing: `gpu_memory_utilization`, block overrides | `config/cache.py` | ✅ T0 | peak mem BEATS vLLM ~25-35% (surpass-track datum) | [specs/kv-sizing.md](specs/kv-sizing.md) |

## 3. Parallelism & scale-out

No distributed runtime in-tree yet (verified: no NCCL/`tensor_parallel` in `src/`).
Upstream's Ray/multiproc executor is replaced by the in-process seam (§9.2) —
multi-process/multi-GPU re-enters through that seam.

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Tensor parallel (TP) | `distributed/`, `config/parallel.py` | 🚧 T2 (spec written, task #50) | weight-stacking semantics already ported (single-GPU shape); ⚠ impl needs a 2-GPU box (GB10 is single-GPU) | [specs/tensor-parallelism.md](specs/tensor-parallelism.md) |
| Pipeline parallel (PP) | same | ☐ T2 | | [specs/pipeline-parallel.md](specs/pipeline-parallel.md) |
| Expert parallel (EP) + EPLB | `v1/worker/gpu/eplb_utils.py` | ☐ T2 | | [specs/expert-parallel.md](specs/expert-parallel.md) |
| Data parallel (DP) | `config/parallel.py` | ☐ T2 | | [specs/data-parallel.md](specs/data-parallel.md) |
| Multi-node (Ray / multiproc executor) | `v1/executor/` | ☐ T3 | re-enters via §9.2 seam; not as-is | [specs/multi-node.md](specs/multi-node.md) |

## 4. Model families

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| **Qwen3.5/3.6 hybrid** — 35B-A3B MoE + 27B dense (gate models) | `models/qwen3_next.py` family | ✅ T0 | MVP gate PASSED both (35B 1.02×, 27B 1.007× vs graphed vLLM, token-exact) | [specs/qwen36-gate-models.md](specs/qwen36-gate-models.md) |
| Model registry (`architectures[]` → impl) | `models/registry.py` | ✅ T0 | arch-select incl. dense/MoE dispatch | [specs/model-registry.md](specs/model-registry.md) |
| Dense decoders (Llama 3.x, Qwen2/3, Mistral, Gemma 2/3, Phi) | `models/` | ☐ T1 | GQA+RoPE+SwiGLU+RMSNorm = subset of T0 layer set | [specs/dense-decoders.md](specs/dense-decoders.md) |
| MoE decoders (Mixtral, Qwen3-MoE, GLM-4-MoE, OLMoE) | `models/` | ☐ T1 | FusedMoE layer exists (35B-proven) | [specs/moe-decoders.md](specs/moe-decoders.md) |
| Qwen3-Next | `models/qwen3_next.py` | ☐ T1 | same stack, interleaved-GQA weight layout | [specs/qwen3-next.md](specs/qwen3-next.md) |
| Hybrid others (Jamba, Bamba, NemotronH, FalconH1, Zamba2, LFM2, Kimi-Linear) | `models/` | ☐ T2 | needs mamba1/2 backends | [specs/hybrid-others.md](specs/hybrid-others.md) |
| MLA family (DeepSeek V3/V3.2/V4, Kimi K2.5) | `models/deepseek_*.py` | ☐ T2 | MLA wrapper + latent KV + sparse indexer | [specs/mla-family.md](specs/mla-family.md) |
| Embedding / pooling / reranker models (BERT-family, GTE, Qwen embed) | `models/` | ☐ T2 | bidirectional attn + poolers | [specs/pooling-models.md](specs/pooling-models.md) |
| Multimodal (Qwen3-VL, vision towers, mRoPE full, encoder cache) | `models/`, `multimodal/` | ☐ T3 | 27B VL wrapper runs TEXT path only today (ViT/merger stubbed); revisit after T1 | [specs/multimodal.md](specs/multimodal.md) |
| Encoder-decoder / ASR (Whisper) | `models/whisper.py` | ☐ T3 | | [specs/encoder-decoder-asr.md](specs/encoder-decoder-asr.md) |

## 5. Quantization

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| NVFP4 modelopt (35B: W4A16 experts via vendored Marlin + fp8/bf16 dense) | `quantization/modelopt.py` | ✅ T0 | Marlin bit-exact vs vLLM's own kernel; gate-passing config | [specs/nvfp4-modelopt.md](specs/nvfp4-modelopt.md) |
| NVFP4 compressed-tensors W4A4 (27B dense, cutlass sm120a GEMM) | `compressed_tensors/`, `csrc/.../fp4/` | ✅ T0 | vendored `cutlass_scaled_fp4_mm_sm120a` 1:1; per-shape autotune beats flashinfer on several shapes (surpass datum) | [specs/nvfp4-w4a4.md](specs/nvfp4-w4a4.md) |
| GGUF quants (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/F16/F32 + NVFP4 ext types) | `quantization/gguf.py` | 🟡 T0 | dequant byte-exact vs ggml + loader ✅; **35B real-file GPU greedy parity PASSED (A2)**; no 27B file, NVFP4 type-40 dequant, and IQ2_S/IQ4_XS remain | [specs/gguf-quants.md](specs/gguf-quants.md) |
| fp8 W8A8 (e4m3) | `quantization/fp8.py` | ☐ T1 | | [specs/fp8-w8a8.md](specs/fp8-w8a8.md) |
| MXFP4 / MXFP8 | `quantization/mxfp4.py`, modelopt | ☐ T1 | | [specs/mxfp4-mxfp8.md](specs/mxfp4-mxfp8.md) |
| AWQ / GPTQ (+ int Marlin) | `quantization/{awq,gptq}*.py` | ☐ T2 | Marlin template already vendored (nvfp4 instantiation) — int paths are new instantiations | [specs/awq-gptq.md](specs/awq-gptq.md) |
| compressed-tensors int schemes | `compressed_tensors/` | ☐ T2 | | [specs/ct-int-schemes.md](specs/ct-int-schemes.md) |
| bitsandbytes, torchao, quark, INC, … | various | ☐ T3 | niche/other-vendor | [specs/quant-longtail.md](specs/quant-longtail.md) |

## 6. Sampling & generation controls

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Core sampler pipeline (temperature, top-k/p, min-p, penalties, seed, n, stop, min/max_tokens, output_kind) | `v1/sample/sampler.py` | ✅ T0 | full ordered pipeline, on-GPU; greedy bit-exact gate | [specs/core-sampler.md](specs/core-sampler.md) |
| torch-Philox bit-exact random parity | `v1/sample/ops/` | ☐ T1 | current RNG gumbel-max distribution-correct only | [specs/philox-rng-parity.md](specs/philox-rng-parity.md) |
| logprobs payload end-to-end | `v1/sample/`, serving | 🟡 T1 | sampler gather_logprobs+ranks ✅; protocol parses `logprobs`/`top_logprobs`; **serving payload NOT wired** (verified: zero logprobs refs in serving_chat.cpp) | [specs/logprobs-payload.md](specs/logprobs-payload.md) |
| prompt_logprobs | `v1/sample/` | ☐ T1 | protocol field parsed; no engine path | [specs/prompt-logprobs.md](specs/prompt-logprobs.md) |
| logit_bias / allowed_token_ids / bad_words | `sampling_params.py`, `v1/sample/` | 🟡 T1 | sampler OPS ✅ (M1.7); SamplingParams fields + OpenAI protocol wiring ☐ (verified absent in protocol.h/sampling_params.h) | [specs/logit-bias-bad-words.md](specs/logit-bias-bad-words.md) |
| best_of / echo / suffix / user fields | `protocol.py` | 🟡 T1 | `echo` parsed; best_of/suffix/user deferred (protocol.h header note) | [specs/completions-longtail-fields.md](specs/completions-longtail-fields.md) |
| Beam search wrapper | `beam_search.py` | ☐ T1 | | [specs/beam-search.md](specs/beam-search.md) |
| Reasoning parsers (+ reasoning-gated grammar) | `reasoning/` | ☐ T1 | | [specs/reasoning-parsers.md](specs/reasoning-parsers.md) |
| Thinking budget | `v1/sample/` | ☐ T1 | stub marked | [specs/thinking-budget.md](specs/thinking-budget.md) |
| Repetition detection | `v1/sample/` | ☐ T1 | | [specs/repetition-detection.md](specs/repetition-detection.md) |
| Custom logits processors (plugin point) | `v1/sample/logits_processor/` | ☐ T2 | → C ABI callback registry (§9.4) | [specs/custom-logits-processors.md](specs/custom-logits-processors.md) |

## 7. Structured outputs & tool calling

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Structured outputs: `json`(schema)/`json_object`/`regex`/`choice`/`grammar` + `response_format` | `v1/structured_output/` | ✅ T0 | integration layer 1:1; grammar ENGINE = native backend (§9 ORIGINAL) behind upstream's pluggable-backend seam | [specs/structured-outputs.md](specs/structured-outputs.md) |
| xgrammar C++ core as 2nd backend | `structured_output/backend_xgrammar.py` | ☐ T1 | same proven seam; closes whitespace/key-order/exotic-schema parity | [specs/xgrammar-backend.md](specs/xgrammar-backend.md) |
| STRUCTURAL_TAG (full) | `structured_output/` | 🟡 T1 | lazy TriggeredTag-equivalent matcher ✅ (powers tool_choice=auto); full structural-tag surface ☐ | [specs/structural-tag.md](specs/structural-tag.md) |
| guidance / outlines backends | `structured_output/` | ☐ T2 | | [specs/guidance-outlines-backends.md](specs/guidance-outlines-backends.md) |
| Tool calling: `tools`/`tool_choice` auto+required+named, streaming deltas, Hermes + Qwen3 parsers | `entrypoints/openai/tool_parsers/` | ✅ T0 | lazy grammar for auto; forced grammar for required/named | [specs/tool-calling.md](specs/tool-calling.md) |
| Tool parser breadth (Qwen-Coder XML, Mistral, pythonic, …) | `tool_parsers/` | ☐ T1 | abstract parser seam in place | [specs/tool-parser-breadth.md](specs/tool-parser-breadth.md) |

## 8. Speculative decoding

Scoping done → [specs/spec-decode-scoping-2026-07-10.md](specs/spec-decode-scoping-2026-07-10.md)
(B5). Route: MTP k=1 on 27B → GDN spec path → DFlash. Both gate checkpoints SHIP
MTP heads (we currently skip `mtp.*` at load).

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| MTP (Qwen3.6 MTP heads, k=1 first) | `v1/worker/gpu/spec_decode/`, `models/qwen3_5_mtp.py` | 🚧 **spec written** | biggest single-model win; NB both gates are GDN hybrids so the GDN spec path is on milestone 1 | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) |
| Rejection sampler | `v1/worker/gpu/spec_decode/rejection_sampler.py` | 🚧 covered by MTP spec | prerequisite for all spec paths | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) (2.4) |
| GDN spec segments (metadata + slot-snapshot rollback) | `v1/attention/backends/gdn_attn.py`, `fla/ops/fused_sigmoid_gating.py` | 🚧 covered by MTP spec | on milestone 1 (both gates hybrid) | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) (3) |
| DFlash (block-diffusion drafter) | in-pin + published drafts for our models | 🚧 **spec written** (after MTP) | DGX-Spark community container exists; GDN slot memory at k=15 flagged | [specs/dflash-spec-decode.md](specs/dflash-spec-decode.md) |
| ngram | `v1/spec_decode/ngram_proposer.py` | ☐ T2 | | [specs/spec-decode-ngram.md](specs/spec-decode-ngram.md) |
| EAGLE3 | `v1/spec_decode/eagle.py` | ☐ T2 | | [specs/spec-decode-eagle3.md](specs/spec-decode-eagle3.md) |

## 9. Serving surface (OpenAI API, endpoints, CLI, library)

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| `/v1/chat/completions` + `/v1/completions` (SSE streaming) | `entrypoints/openai/` | ✅ T0 | cpp-httplib; error shapes mirrored | [specs/chat-completions-endpoints.md](specs/chat-completions-endpoints.md) |
| `/v1/models`, `/health`, `/version` | same | ✅ T0 | verified routes in api_server.cpp | [specs/models-health-version.md](specs/models-health-version.md) |
| **`/metrics` Prometheus, names 1:1** (`vllm:*`) | `v1/metrics/` | ☐ T0-core | verified ABSENT in-tree; core set was T0, deferred at server land — **oldest open T0 debt**; full set T1 | [specs/prometheus-metrics.md](specs/prometheus-metrics.md) |
| `stream_options` / `include_usage` | `protocol.py` | ☐ T1 | verified absent | [specs/stream-options.md](specs/stream-options.md) |
| `/tokenize`, `/detokenize`, `/ready`, `/ping`, `/server_info`, `/reset_prefix_cache` | various routers | ☐ T1 | | [specs/utility-endpoints.md](specs/utility-endpoints.md) |
| Chat templating (Jinja subset, minja-style) | `entrypoints/chat_utils.py` | ✅ T0 | Qwen3.6 byte-identical vs jinja2; tool/think branches included | [specs/chat-templating.md](specs/chat-templating.md) |
| AsyncLLM-equivalent streaming engine API | `v1/engine/async_llm.py` | ☐ T1 | today: sync engine + server-side engine mutex (verified) | [specs/async-llm-api.md](specs/async-llm-api.md) |
| C API library (llama.cpp-style, 11-symbol ABI) | — (our packaging) | ✅ T0 | dlopen-proven for LocalAI | [specs/c-api-library.md](specs/c-api-library.md) |
| Rich C++ API (`LLM`/`AsyncLLM` mirror) | `entrypoints/llm.py` | ☐ T1 | | [specs/cpp-api.md](specs/cpp-api.md) |
| CLI: `serve` + `bench {latency,throughput,serve}` | `entrypoints/cli/` | ✅ T0 | examples/server + examples/bench; serve-latency A/B vs `vllm serve` = closing track A1 | [specs/cli-serve-bench.md](specs/cli-serve-bench.md) |
| CLI: `chat`, `complete` | `entrypoints/cli/` | ☐ T1 | examples/cli covers basic complete | [specs/cli-chat-complete.md](specs/cli-chat-complete.md) |
| `/v1/embeddings`, `/pooling`, `/score`, `/rerank` | pooling routers | ☐ T2 | with pooling models (§4) | [specs/pooling-endpoints.md](specs/pooling-endpoints.md) |
| `/v1/responses`, `/v1/messages` (Anthropic-style), audio | responses/messages routers | ☐ T2 | | [specs/responses-messages-endpoints.md](specs/responses-messages-endpoints.md) |
| Sleep/pause/resume, profiling, RL weight-update endpoints | various | ☐ T2–T3 | | [specs/admin-endpoints.md](specs/admin-endpoints.md) |
| OTLP tracing | `config/observability.py` | ☐ T2 | | [specs/otlp-tracing.md](specs/otlp-tracing.md) |

## 10. LoRA & adapters

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| LoRA runtime (punica-style batched apply) | `lora/`, `v1/worker/lora_model_runner_mixin.py` | ☐ T2 | no runtime in-tree (verified: field placeholders only) | [specs/lora-runtime.md](specs/lora-runtime.md) |
| LoRA dynamic load/unload endpoints | `entrypoints/openai/` | ☐ T2 | after runtime | [specs/lora-endpoints.md](specs/lora-endpoints.md) |

## 11. Long context & attention breadth

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| YaRN rope scaling | `layers/rotary_embedding/` | ☐ T1 | Qwen3.6 long-context uses it — first of the rope family | [specs/yarn-rope.md](specs/yarn-rope.md) |
| llama3 / longrope / dynamic-NTK scaling | same | ☐ T1–T2 | | [specs/rope-scaling-family.md](specs/rope-scaling-family.md) |
| Sliding-window attention backend | `v1/attention/backends/` | ☐ T1 | pairs with §2 KV spec row | [specs/sliding-window-attention.md](specs/sliding-window-attention.md) |
| MLA backends (latent KV, MQA decode) | `v1/attention/backends/mla/` | ☐ T2 | | [specs/mla-backends.md](specs/mla-backends.md) |
| Mamba1/Mamba2/short-conv/linear backends | `v1/attention/backends/` | ☐ T2 | | [specs/mamba-backends.md](specs/mamba-backends.md) |
| Encoder / cross-attention | backends | ☐ T2 | | [specs/encoder-cross-attention.md](specs/encoder-cross-attention.md) |

## 12. Platforms & hardware

Per [specs/expansion-map-2026-07-10.md](specs/expansion-map-2026-07-10.md) (B1/B2) and
[backends.md](backends.md). Portability confined to vLLM's own seams
(platforms/ + attention backends + vt:: op tables).

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| CUDA GB10 (sm_121) | `platforms/cuda.py` | ✅ T0 | the gate platform | [specs/cuda-gb10.md](specs/cuda-gb10.md) |
| NVIDIA fan-out: A100/H100/4090 (sm_80+) | same | ☐ Wave 1 | B2: nearly FREE — bf16 path already sm_80+, ~no new kernels; needs per-arch Triton-AOT cubins (A4) + CI matrix | [specs/nvidia-fanout.md](specs/nvidia-fanout.md) |
| ROCm (gfx942 first) | `platforms/rocm.py` | ☐ Wave 2 | B2: Triton GDN AOT retargets via one flag; rest = HIP port of vt:: | [specs/rocm.md](specs/rocm.md) |
| Apple Metal via MLX (vendored under vt::) | — (§9.8 extension) | ☐ Wave 2 | B1: MLX wins vs native MSL; M4 Mac mini/16 GB reachable for op + small-model bring-up, but gate-model memory testing still needs a larger Mac | [specs/metal-mlx.md](specs/metal-mlx.md) |
| Vulkan | — (§9.8 extension) | ☐ Wave 3 | B2 call: Vulkan before SYCL | [specs/vulkan.md](specs/vulkan.md) |
| Intel XPU | `platforms/xpu.py` | ☐ Wave 3+ | loyal port (not a deviation) | [specs/intel-xpu.md](specs/intel-xpu.md) |
| CPU production path (threadpool + compute-in-quant) | `platforms/cpu.py`, `csrc/cpu/` | 🚧 B4 | today: single-threaded scalar reference (correctness-grade, CI parity); B4: threadpool = 1-wk prereq, compute-in-quant ~3.3× decode / ~10× prefill on GGUF; local Qwen3.5-2B bench = decision measurement (in flight) | [specs/cpu-production.md](specs/cpu-production.md) |
| De-Python the build (vendor Triton-AOT cubins per-arch) | — (build infra) | ✅ A4/sm_121a | vendored artifacts + manifest build without Python; `VLLM_CPP_TRITON_REGEN` is maintainer-only; additional arches land with NVIDIA/ROCm fan-out | [specs/depython-build.md](specs/depython-build.md) |

## 13. Loading, tokenizer, config

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| safetensors loader + stacked-params mapping + WeightsMapper | `model_loader/` | ✅ T0 | | [specs/safetensors-loader.md](specs/safetensors-loader.md) |
| GGUF loader (name mapping, transform inversions, expert split) + embedded vocab | `model_loader/gguf_loader.py` | 🟡 T0 | loader + 35B real-file GPU parity ✅; dense-27B loader/file and remaining quant breadth are open | [specs/gguf-loader.md](specs/gguf-loader.md) |
| HF `tokenizer.json` byte-level BPE + incremental detok | `transformers_utils/tokenizer.py` | ✅ T0 | | [specs/hf-tokenizer.md](specs/hf-tokenizer.md) |
| SentencePiece tokenizer | same | ☐ T1 | needed for Llama-2-era / Gemma checkpoints | [specs/sentencepiece.md](specs/sentencepiece.md) |
| Config surface: dataclass-for-dataclass structs, `vllm serve`-compatible flags | `vllm/config/` | 🟡 T0/T1 | T0 set ✅; remainder lands with each feature | [specs/config-surface.md](specs/config-surface.md) |
| Sharded-state / tensorizer / runai / BnB loaders | `model_loader/` | ☐ T3 | | [specs/loader-longtail.md](specs/loader-longtail.md) |

---

## Reading the matrix into work

1. Pick the highest-leverage ☐/🟡 row (tier order, then roadmap_v1.md §C/§D queue).
2. Write its spec at the linked `specs/<slug>.md` path: scope, upstream file:line
   to mirror, gates (token-exact / conformance), same-box A/B plan.
3. Delegate the spec to a sub-agent (worktree isolation; GPU work under the flock
   mutex). Row flips ☐ → 🚧.
4. Merge + gate ⇒ ✅ here, marker flip in porting-inventory.md, row in
   parity-ledger.md — same change.

Currently open T0 debt (pre-existing, tracked): `/metrics` core set (§9),
serve-latency A/B (A1), and e2e suites (A5). GGUF real-file parity A2 passed;
its remaining format/model breadth stays tracked in §5/§13.
