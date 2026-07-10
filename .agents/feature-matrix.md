# Feature matrix — cross-cutting feature parity

**This document is the broad parity coverage view for cross-cutting engine, KV,
scale-out, sampling, serving, spec-decode, long-context, LoRA and loading.** The
stable-ID execution rows and exact code/test anchors live in
[engine-matrix.md](engine-matrix.md). The comprehensive model,
quantization, kernel and platform inventories live in
[model-matrix.md](model-matrix.md),
[quantization-matrix.md](quantization-matrix.md),
[kernel-matrix.md](kernel-matrix.md), and
[backend-matrix.md](backend-matrix.md). The ordered portfolio is
[roadmap_v1.md](roadmap_v1.md).

**Convention (delegation unit):** a planned-but-missing spec is not a spike.
Rows are being migrated to the stable-ID/evidence contract in
[coordination.md](coordination.md); a legacy implemented row without exact code,
test and real-spec anchors is `ANCHOR-BACKFILL`, not protocol-complete `DONE`.

**Lifecycle:** `INVENTORIED -> SPIKE -> READY -> ACTIVE -> GATING -> DONE`, with
`PARTIAL`, `BLOCKED`, and `ANCHOR-BACKFILL` as explicit non-done states. Legacy
emoji rows retain runtime meaning during the evidence migration but cannot be
used as new claims. Tier (T0–T3) remains per porting-inventory.

**Mirror policy:** vLLM parity is the FLOOR for every row (mirror-vllm-always);
the surpass track (roadmap_v1.md "Protocol evolution") builds beyond it, never
instead of it.

---

## 1. Engine core & scheduling

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Continuous batching / unified scheduler (token-budget, no prefill/decode split) | `v1/core/sched/scheduler.py` | `ANCHOR-BACKFILL` T0 | proven text-only running-first/FCFS/token-budget slice; broad scheduler parity open | `planned: specs/unified-scheduler.md` |
| Chunked prefill (on by default) | `config/scheduler.py` | `ANCHOR-BACKFILL` T0 | basic token-budget chunking; partial-prefill concurrency/MM modes open | `planned: specs/chunked-prefill.md` |
| Prefix caching (APC) | `v1/core/kv_cache_utils.py`, `v1/core/kv_cache_coordinator.py` | `PARTIAL` T0 | W0 ports no-prefix coordination, hybrid/attention-free default-off resolution, explicit CLI override, and cache-off benchmark contracts; hashing/lookup/reuse subset exists, while salt/partial/retention/connector breadth remains | [prefix-caching.md](specs/prefix-caching.md) |
| Preemption (FCFS tail pop, recompute) | `v1/core/sched/scheduler.py` | `ANCHOR-BACKFILL` T0 | FCFS recompute slice | `planned: specs/preemption.md` |
| CUDA graphs (decode capture/replay, no torch) | `config/compilation.py::cudagraph_mode` | `PARTIAL` T0 | Qwen-specific capture; explicit 35B assertion, generic modes and direct 27B evidence open | `planned: specs/cuda-graphs.md` |
| Opt-in batch-invariant execution (`VLLM_BATCH_INVARIANT=1`) | `envs.py`, `model_executor/layers/batch_invariant.py`, `csrc/.../nvfp4_scaled_mm_sm120_kernels.cu` | ☐ T1 | v0.25 production default is off; the opt-in mode changes matmul/norm/attention/collective and NVFP4 dispatch to make one request invariant to neighboring batch rows. W3-C3R executes the default-off contrast for ours and vLLM but does not implement the opt-in feature | `planned: specs/batch-invariant-execution.md` |
| **Async/overlap scheduling** | `v1/core/sched/async_scheduler.py` | ☐ T1 (**promoted**), spike accepted | B3 finding: vLLM's DEFAULT at our pin ⇒ unmet MIRROR obligation; W2 supplies the asynchronous frontend only, while execution row `ENG-ASYNC-SCHED` remains `READY` for the W3 scheduler/runner overlap | [async-serving.md](specs/async-serving.md) |
| Priority scheduling (`policy="priority"`) | `v1/core/sched/request_queue.py` | 🚧 T1 `GATING` | W4 queue, priority preemption and request/config plumbing implemented; CPU suites green, GB10 G1 token-exact A/B pending | [async-serving.md](specs/async-serving.md) |
| Partial-prefill concurrency (`max_num_partial_prefills`, long-prefill limits) | `config/scheduler.py` | ☐ T1 | | `planned: specs/partial-prefill-concurrency.md` |
| `step_with_batch_queue` (pipelined batch queue) | `v1/engine/core.py` | ☐ T1 | | `planned: specs/batch-queue-step.md` |
| `scheduler_reserve_full_isl`, pluggable `scheduler_cls`, `stream_interval` | `config/scheduler.py` | `PARTIAL` T1 | fields/internal handling exist; scheduler registry and public stream-interval wiring are absent | `planned: specs/scheduler-knobs.md` |
| Cascade attention (shared-prefix batches) | `config/model.py::disable_cascade_attn` | ☐ T2 | | `planned: specs/cascade-attention.md` |
| DBO / ubatch overlap | `config/parallel.py::enable_dbo` | ☐ T2 | | `planned: specs/dbo-ubatch.md` |

## 2. KV cache & memory

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Paged KV: BlockPool (free list, refcount, LRU evict) | `v1/core/block_pool.py` | `ANCHOR-BACKFILL` T0 | core free-list/refcount/cache/LRU slice; connector/partial primitives separate | `planned: specs/block-pool.md` |
| KVCacheManager (allocate_slots, watermark, admission) | `v1/core/kv_cache_manager.py` | `ANCHOR-BACKFILL` T0 | named allocation/admission/lookahead slice | `planned: specs/kv-cache-manager.md` |
| Device-resident attention KV + indexed GDN state I/O | `v1/worker/gpu/attn_utils.py`, `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `ACTIVE` T0 | W0 stable allocations/fallback and W1 indexed BF16↔F32 gather/scatter remain correctness/safety green with separate 1.021239×/1.006246× component A/Bs and structural copy reduction. Clean `b5c6e4f` re-ranks the exact residual: W2 FP4 improves every concurrency and wins c16/c32 throughput, but FP4 tactic-selection parity, c1-c8, TPOT/ITL and host-memory axes remain ahead of device-residency W2. Inherited pools still fail zero-leak | [device-resident-kv-gdn-state.md](specs/device-resident-kv-gdn-state.md) |
| Hybrid KV coordinator (full-attn + GDN/mamba state groups) | `v1/core/kv_cache_coordinator.py` | `PARTIAL` T0 | cross-group MIN-intersection prefix hit; align/retention modes absent | `planned: specs/hybrid-kv-coordinator.md` |
| Mamba/GDN prefix-cache retention (`mamba_cache_mode=align`, vllm#45845) | `v1/core/` | ☐ T1 (**promoted into competitor gate**) | vLLM hybrid defaults off but explicit cache-on selects align for Qwen3.5/3.6; local 1:1 stub exists, full retention/runtime path is required for matched vLLM/SGLang shared-prefix gating | `planned: specs/mamba-align-retention.md` |
| SlidingWindowSpec + ChunkedLocalAttentionSpec | `v1/kv_cache_interface.py` | `PARTIAL` T1 | Both execution leaves are implemented and `GATING`: W1 sliding-window and W3 chunked-local sizing, registry/grouping, manager prefix/recycling policy, admission and hybrid-disabled conversion pass their ported CPU/property/sanitizer gates; attention/model/oracle/runtime closure remains | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| fp8 KV cache (`cache_dtype=fp8*`) | `layers/quantization/kv_cache.py` | ☐ T1 | | `planned: specs/fp8-kv-cache.md` |
| nvfp4 / per-token-head / turboquant KV | `config/cache.py` | ☐ T2 | | `planned: specs/nvfp4-kv-cache.md` |
| KV offload (CPU tiering, LRU/ARC) | `v1/kv_offload/` | ☐ T2 | | `planned: specs/kv-offload.md` |
| External KV-cache provider ABI + LMCache (MP service and in-process connectors) | `config/kv_transfer.py`, `distributed/kv_transfer/kv_connector/v1/{base,lmcache_connector,lmcache_mp_connector}.py` | ☐ T2 | explicit roadmap outcome `KV-EXTERNAL-CACHE`: mirror `kv_producer`/`kv_consumer`/`kv_both`, scheduler/worker metadata, async layer load/store, dynamic external connector modules, failure policy, metrics and cache-lifecycle ownership; gate the official LMCache shared-prefix quickstart plus Qwen3.6 hybrid behavior | `planned: specs/external-kv-cache-lmcache.md` |
| KV connector breadth (NIXL/Mooncake/MultiConnector, **PD disaggregation**) | `distributed/kv_transfer/` | ☐ T2 | builds on the external-provider base seam; remains a separate breadth/scale-out leaf | `planned: specs/kv-connectors-disagg.md` |
| KV events (block create/evict publish) | `config/kv_events.py` | ☐ T2 | | `planned: specs/kv-events.md` |
| MLAAttentionSpec (latent KV) | `v1/kv_cache_interface.py` | ☐ T2 | with DeepSeek family | `planned: specs/mla-kv-spec.md` |
| Sizing: `gpu_memory_utilization`, block overrides | `config/cache.py` | `PARTIAL` T0 | watermark/fixed loader inputs exist; public utilization/cache-byte/override policy absent | `planned: specs/kv-sizing.md` |
| Weight CPU offload (`cpu_offload_gb` UVA per-parameter + layer-group `PrefetchOffloader`) | `config/offload.py`; `model_executor/offloader/` | ☐ T2 | v1-supported at the pin; blanket/name-targeted, NOT router-aware; mirror floor for expert streaming (engine row `ENG-WEIGHT-OFFLOAD`) | `planned: specs/weight-offload-uva.md` |
| Expert streaming from disk (routed-MoE experts paged NVMe→GPU on router output, budgeted resident cache) | absent in-pin (surpass-track); design ref antirez/ds4 | ☐ T2 | corrected engine row `ENG-EXPERT-STREAM` READY: bank-only loader, fixed contiguous Marlin slots, logical→slot remap, explicit router D2H, chunked prefill; W0 trace/baseline first | [expert-streaming.md](specs/expert-streaming.md) |

## 3. Parallelism & scale-out

No distributed runtime in-tree yet (verified: no NCCL/`tensor_parallel` in `src/`).
Upstream's Ray/multiproc executor is replaced by the in-process seam (§9.2) —
multi-process/multi-GPU re-enters through that seam.

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Tensor parallel (TP) | `distributed/`, `config/parallel.py` | 🚧 T2 (spec written, task #50) | weight-stacking semantics already ported (single-GPU shape); ⚠ impl needs a 2-GPU box (GB10 is single-GPU) | [specs/tensor-parallelism.md](specs/tensor-parallelism.md) |
| Pipeline parallel (PP) | same | ☐ T2 | | `planned: specs/pipeline-parallel.md` |
| Expert parallel (EP) + EPLB | `v1/worker/gpu/eplb_utils.py` | ☐ T2 | | `planned: specs/expert-parallel.md` |
| Data parallel (DP) | `config/parallel.py` | ☐ T2 | | `planned: specs/data-parallel.md` |
| MoE sequence parallelism without DP | `config/parallel.py::use_sequence_parallel_moe`, `distributed/parallel_state.py` | ☐ T2 | v0.25.0 adds the non-DP path and reports 1.9–5.0% E2E throughput; inventory row `PAR-SEQUENCE-MOE` | `planned: specs/sequence-parallel-moe.md` |
| Multi-node (Ray / multiproc executor) | `v1/executor/` | ☐ T3 | re-enters via §9.2 seam; not as-is | `planned: specs/multi-node.md` |

## 4. Model families

The current pin contains 353 unique static architecture IDs; the audited
v0.25.0 target adds three, for 356 after pin advancement, plus a dynamic
Transformers-compatibility path. The complete alias-preserving inventory is
[model-matrix.md](model-matrix.md); this summary is only the roadmap roll-up.

| ID | Category / mechanism | Pinned upstream | State | Grounded summary | Detailed evidence / spike |
|---|---|---|---|---|---|
| `MODEL-GATE-QWEN35` | Qwen3.5/3.6 dense + MoE wrappers | `models/registry.py:556-560` | `PARTIAL` | 27B/35B **text submodels** pass token/perf gates; local plain-BF16 4B is diagnostic at 15/16 representative greedy sequences; both upstream IDs are multimodal wrappers and their ViT/merger paths are absent | model matrix + existing Qwen specs |
| `MODEL-FACTORY` | architecture -> model factory + reject unknown | `models/registry.py:998-1404` | `PARTIAL` | central ordered type-erased registry now covers both implemented Qwen IDs; live loading resolves the full `architectures` list and rejects unknown/previous/OOT IDs with pinned messages instead of using `num_experts`; execution row `MODEL-FACTORY-registry` is `GATING` on the two-model GPU no-regression campaign | [model-factory-registry.md](specs/model-factory-registry.md) |
| `MODEL-TEXT` | 130 text-generation IDs | `models/registry.py:71-208` | `INVENTORIED` | 0/130 static text-generation IDs directly supported | model matrix |
| `MODEL-POOLING` | embedding, late-interaction, reward, token/sequence classification | `models/registry.py:210-329` | `INVENTORIED` | five distinct work fronts; no pooling runtime | model matrix |
| `MODEL-MM` | 115 v0.25.0-target multimodal IDs | `models/registry.py:331-583` | `PARTIAL` | 2 wrappers run text-only; encoder/processors/cache absent; MOSS-Transcribe-Diarize is target-pending | model matrix |
| `MODEL-SPEC` | 46 v0.25.0-target speculative-draft IDs | `models/registry.py:585-638` | `READY` | MTP/DFlash specs exist; Laguna DFlash and Bailing hybrid MTP are target-pending; no draft model runs | model matrix + [spec-decode specs](specs/mtp-spec-decode.md) |
| `MODEL-TRANSFORMERS` | 14 static aliases + dynamic compatible classes | `models/registry.py:635-680,1096-1164` | `INVENTORIED` | C++ policy/factory not spiked | model matrix |

## 5. Quantization

The canonical leaf inventory is
[quantization-matrix.md](quantization-matrix.md). It separately tracks format
recognition, materialization/repack, native quantized compute, real-model gates
and reference-engine performance.

| ID | Block | State | Grounded summary | Detailed evidence / spike |
|---|---|---|---|---|
| `QUANT-CUDA-GATES` | NVFP4 W4A16, NVFP4 W4A4, gate-specific FP8 W8A8 | `DONE` | support/correctness stays closed; performance remains `ACTIVE` at `3f256ab` **55/124**. W3-C runtime/cache gates and six-process stability pass: immutable `d211b8f` loads one byte-identical 64/64 native map with zero tuning/misses and passes both 27B arms. C3R proves direct/fallback 6/6 x 128-token equality under identical sequential batch shape and shows the earlier independent-run 2/6 condition was an invalid default-mode batch-invariance predicate; stopped partial performance stays void and corrected C3 is ready. Exact grid and 35B performance remain open ([persistent-cache spike](specs/nvfp4-persistent-plan-cache.md)) | quant matrix §2 + [coverage spike](specs/quantization-coverage.md) |
| `QUANT-GGUF` | llama.cpp encodings and output presets | `PARTIAL` | F32/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K materialize; F16 was corrected to reader-only; CPU threadpool/chunked dispatch is correctness-gated but its B4 speed/RSS checkpoint is pending; no direct compute-in-quant or llama.cpp speed parity | quant matrix §1 |
| `QUANT-VLLM-BREADTH` | generic FP8/MX, AWQ/GPTQ, CT integer, vendor methods, KV | `PARTIAL` | gate-specific implementations exist; generic dispatch/modes remain inventoried | quant matrix §§2-3 |
| `QUANT-MLX` | affine Q2-8, MXFP4/MXFP8/NVFP4, QQ, mixed recipes/imports | `INVENTORIED` | required for Apple backend; no MLX runtime yet | quant matrix §4 |

## 6. Sampling & generation controls

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Core sampler pipeline (temperature, top-k/p, min-p, penalties, seed, n, stop, min/max_tokens, output_kind) | `v1/sample/sampler.py` | `PARTIAL` T0 | greedy and bounded controls work; random uses an on-GPU multi-block exponential-race reduction, but `n` is parsed and not executed and token/logprob results still synchronize to host | `planned: specs/core-sampler.md` |
| torch-Philox bit-exact random parity | `v1/sample/ops/` | ☐ T1 | current RNG gumbel-max distribution-correct only | `planned: specs/philox-rng-parity.md` |
| logprobs payload end-to-end | `v1/sample/`, serving | `PARTIAL` T1 | sampler tensors/protocol fields exist; engine/output/OpenAI payload absent | `planned: specs/logprobs-payload.md` |
| prompt_logprobs | `v1/sample/` | ☐ T1 | protocol field parsed; no engine path | `planned: specs/prompt-logprobs.md` |
| logit_bias / allowed_token_ids / bad_words | `sampling_params.py`, `v1/sample/` | `PARTIAL` T1 | internal ops exist; request metadata and OpenAI wiring absent | `planned: specs/logit-bias-bad-words.md` |
| best_of / echo / suffix / user fields | `protocol.py` | `PARTIAL` T1 | echo parses but behavior is deferred; remaining fields absent | `planned: specs/completions-longtail-fields.md` |
| Beam search wrapper | `beam_search.py` | ☐ T1 | | `planned: specs/beam-search.md` |
| Reasoning parsers (+ reasoning-gated grammar) | `reasoning/` | ☐ T1 | | `planned: specs/reasoning-parsers.md` |
| Thinking budget | `v1/sample/` | ☐ T1 | stub marked | `planned: specs/thinking-budget.md` |
| Repetition detection | `v1/sample/` | ☐ T1 | | `planned: specs/repetition-detection.md` |
| Custom logits processors (plugin point) | `v1/sample/logits_processor/` | ☐ T2 | → C ABI callback registry (§9.4) | `planned: specs/custom-logits-processors.md` |

## 7. Structured outputs & tool calling

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Structured outputs: `json`(schema)/`json_object`/`regex`/`choice`/`grammar` + `response_format` | `v1/structured_output/` | `PARTIAL` T0 | seam + native backend work for a bounded JSON-schema subset; upstream backend matrix is unported | `planned: specs/structured-outputs.md` |
| xgrammar C++ core as 2nd backend | `structured_output/backend_xgrammar.py` | ☐ T1 | same proven seam; closes whitespace/key-order/exotic-schema parity | `planned: specs/xgrammar-backend.md` |
| STRUCTURAL_TAG (full) | `structured_output/`, `tool_parsers/structural_tag_registry.py` | `PARTIAL` T1 | tool-choice subset works; full response-format/tag surface absent | `planned: specs/structural-tag.md` |
| guidance / outlines backends | `structured_output/` | ☐ T2 | | `planned: specs/guidance-outlines-backends.md` |
| Tool calling: `tools`/`tool_choice` auto+required+named, streaming deltas, Hermes + Qwen3 parsers | `tool_parsers/`, `entrypoints/openai/chat_completion/` | `PARTIAL` T0 | Hermes core works; local Qwen3 parser is a Hermes alias, not upstream Qwen3Engine parity | `planned: specs/tool-calling.md` |
| Unified Streaming Parser Engine | `parser/engine/` | ☐ T1 | one reasoning/tool-call streaming event engine, adapters, token-ID scanning and replay; inventory row `TOOLS-STREAMING-PARSER` | `planned: specs/streaming-parser-engine.md` |
| Tool parser breadth (Qwen-Coder XML, Mistral, pythonic, …) | `tool_parsers/` | ☐ T1 | abstract parser seam in place | `planned: specs/tool-parser-breadth.md` |

## 8. Speculative decoding

Scoping done → [specs/spec-decode-scoping-2026-07-10.md](specs/spec-decode-scoping-2026-07-10.md)
(B5). Route after speed parity: MTP k=1 on 27B → GDN spec path → DFlash →
DSpark. Both gate checkpoints SHIP
MTP heads. The optional safetensors head loader/standalone forward now exists;
normal target loading still leaves `mtp.*` unloaded until speculative decoding
is configured, exactly as upstream loads its draft model on demand.

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| MTP (Qwen3.6 MTP heads, k=1 first) | `v1/worker/gpu/spec_decode/`, `models/qwen3_5_mtp.py` | 🚧 **GATING: M-mtp-0** | dense + MoE safetensors loaders, shared embed/lm-head and standalone forward are CPU-tested; both-checkpoint DGX oracle head parity is queued behind `CLAIM-SERVE-GATE-1`. No scheduler/rejection/GDN-spec runtime path is claimed yet | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) |
| Rejection sampler | `v1/worker/gpu/spec_decode/rejection_sampler.py` | 🚧 covered by MTP spec | prerequisite for all spec paths | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) (2.4) |
| GDN spec segments (metadata + slot-snapshot rollback) | `v1/attention/backends/gdn_attn.py`, `fla/ops/fused_sigmoid_gating.py` | 🚧 covered by MTP spec | on milestone 1 (both gates hybrid) | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) (3) |
| DFlash (block-diffusion drafter) | in-pin + published drafts for our models | 🚧 **spec written** (after MTP) | DGX-Spark community container exists; GDN slot memory at k=15 flagged | [specs/dflash-spec-decode.md](specs/dflash-spec-decode.md) |
| DSpark (semi-autoregressive block drafter) | `v1/worker/gpu/spec_decode/dspark/`, `models/{qwen3_dspark,deepseek_v4/nvidia/dspark}.py` | ☐ T1 (**user-promoted**) | DeepSeek-V4 and Qwen3 draft layouts, reduced-vocab mapping, Markov sampling and full-CUDA-graph behavior inventoried as `SPEC-DSPARK`; dedicated spike follows parity/MTP | `planned: specs/dspark-spec-decode.md` |
| TLI heterogeneous-vocabulary spec decode | `v1/spec_decode/vocab_mapping.py`, `config/speculative.py` | ☐ T1 | target↔draft ID mapping and shared-token constrained logits; current upstream validation is greedy draft only; inventory row `SPEC-TLI` | `planned: specs/tli-spec-decode.md` |
| ngram | `v1/spec_decode/ngram_proposer.py` | ☐ T2 | | `planned: specs/spec-decode-ngram.md` |
| EAGLE3 | `v1/spec_decode/eagle.py` | ☐ T2 | | `planned: specs/spec-decode-eagle3.md` |

## 9. Serving surface (OpenAI API, endpoints, CLI, library)

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| `/v1/chat/completions` + `/v1/completions` (SSE streaming) | `entrypoints/openai/` | `ANCHOR-BACKFILL` T0 | basic transport/framing plus W2 live incremental AsyncLLM delivery and disconnect abort are CPU/TSan-proven; full protocol and GB10 online gates remain in leaf rows | `planned: specs/chat-completions-endpoints.md` |
| `/v1/models`, `/health`, `/version` | same | `PARTIAL` T0 | routes work; `/health` always returns 200 instead of checking engine health | `planned: specs/models-health-version.md` |
| **`/metrics` Prometheus, names 1:1** (`vllm:*`) | `v1/metrics/` | ☐ T0-core | verified ABSENT in-tree; core set was T0, deferred at server land — **oldest open T0 debt**; full set T1 | `planned: specs/prometheus-metrics.md` |
| Per-request timing metrics in chat/completion response bodies | `entrypoints/generate/base/serving.py`, OpenAI protocols/serving | ☐ T1 | v0.25.0 opt-in surface for TTFT/prefill/decode timing, streaming/non-streaming and invalid multi-output suppression; inventory row `SERVE-RESPONSE-METRICS` | `planned: specs/per-request-response-metrics.md` |
| `stream_options` / `include_usage` | `engine/protocol.py`, completion/chat `protocol.py` + `serving.py`, `serve/utils/api_utils.py` | 🚧 **GATING** T1 | native-ID final/continuous completion+chat frames, non-stream validation and force mode are implemented; CPU 105/105, focused ASan+UBSan 3/3 and TSan 1/1 pass. Fresh merged-SHA 27B→35B online evidence remains before `DONE` | [stream-options.md](specs/stream-options.md) |
| `/tokenize`, `/detokenize`, `/ready`, `/ping`, `/server_info`, `/reset_prefix_cache` | various routers | ☐ T1 | | `planned: specs/utility-endpoints.md` |
| Chat templating (Jinja subset, minja-style) | `renderers/hf.py`, `entrypoints/chat_utils.py` | `ANCHOR-BACKFILL` T0 | bounded Qwen3.6 minja/Jinja subset; generic template parity open | `planned: specs/chat-templating.md` |
| AsyncLLM-equivalent streaming engine API | `v1/engine/async_llm.py` | 🚧 **T0 `GATING`** | W2 live SSE/concurrent abort behavior remains CPU/TSan-green. Fixed `max_num_seqs + 4` capacity, the legacy toggle and 32-client/control reserve are implemented. Exact fixed/legacy c32 GPU means are 1097.031/1097.290 tok/s (0.999764×), 8/20 fixed axes, all 1,152 requests/six lifecycles and no sampled stall; the fresh fixed ladder is healthy at c32. Broader every-axis parity remains before `DONE` | [async-serving.md](specs/async-serving.md) |
| C API library (llama.cpp-style, 17-symbol ABI) | — (our packaging) | `ANCHOR-BACKFILL` T0 | dlopen/FFI proof now includes six additive nonblocking request submit/cancel/wait/done/error/free symbols over AsyncLLM; CPU/TSan green, not a claim of LocalAI integration | `planned: specs/c-api-library.md` |
| Rich C++ API (`LLM`/`AsyncLLM` mirror) | `entrypoints/llm.py` | ☐ T1 | | `planned: specs/cpp-api.md` |
| CLI: `serve` + `bench {latency,throughput,serve}` | `entrypoints/cli/` | `PARTIAL` T0 | separate server/bench binaries and one in-process benchmark; server now exposes `max_num_seqs`/`max_num_batched_tokens` for reproducible operating points, but no matching command family | `planned: specs/cli-serve-bench.md` |
| CLI: `chat`, `complete` | `entrypoints/cli/` | ☐ T1 | examples/cli covers basic complete | `planned: specs/cli-chat-complete.md` |
| `/v1/embeddings`, `/pooling`, `/score`, `/rerank` | pooling routers | ☐ T2 | with pooling models (§4) | `planned: specs/pooling-endpoints.md` |
| `/v1/responses`, `/v1/messages` (Anthropic-style), audio | responses/messages routers | ☐ T2 | | `planned: specs/responses-messages-endpoints.md` |
| Sleep/pause/resume, profiling, RL weight-update endpoints | various | ☐ T2–T3 | | `planned: specs/admin-endpoints.md` |
| OTLP tracing | `config/observability.py` | ☐ T2 | | `planned: specs/otlp-tracing.md` |

## 10. LoRA & adapters

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| LoRA runtime (punica-style batched apply) | `lora/`, `v1/worker/lora_model_runner_mixin.py` | ☐ T2 | no runtime in-tree (verified: field placeholders only) | `planned: specs/lora-runtime.md` |
| LoRA dynamic load/unload endpoints | `entrypoints/openai/` | ☐ T2 | after runtime | `planned: specs/lora-endpoints.md` |

## 11. Long context & attention breadth

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| YaRN rope scaling | `layers/rotary_embedding/` | `GATING` T1 | W5 implements typed modern/legacy config, memoized f32/bf16 cache/factory, plain YaRN and its `mrope_section` branch, plus NeoX/GPT-J and 1-D/3-axis supplied-cache CPU/CUDA code. Six pinned-source oracle fixtures and CPU/sanitizer gates pass; CUDA compile/runtime, feature-positive Nomic/Qwen-VL e2e, trace and every-axis closure remain. Audited Qwen3.6 gates use **default interleaved MRoPE, not YaRN**, so they are regression-only | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| llama3 / longrope / dynamic-NTK scaling | same | `GATING` T1–T2 | all three independently claimable leaves are implemented: W6 Llama 3 low/mid/high, W7 Phi-3 global short/long selection, and W8 dynamic factor/alpha formulas pass nine combined pinned-source f32/bf16 fixtures plus CPU/sanitizers. CUDA apply, Llama-3.1/Phi-3/Hunyuan e2e and G9 remain gated | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| Sliding-window attention backend | `v1/attention/backends/` | `GATING` T1 | W2 implements the shared semantic window, model-config normalization, CPU mask, separately specialized full/local portable-CUDA kernels, vendored FA2 local dispatch, and upstream-derived tests. CPU + sanitizer gates and sm_121a compile-only checks pass; GPU runtime, supported-model e2e, oracle, trace, performance and memory closure remain gated | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| Chunked-local attention wrapper/backend | `model_executor/layers/attention/chunked_local_attention.py` | `GATING` T1 | W4 cached wrapper, six exact virtual-batch shapes, reusable block-table update plan, cudagraph rejection, spec emission and ordinary-backend delegation pass CPU/reference/sanitizer gates; Llama4/model, oracle, GPU trace and every-axis closure remain | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| MLA backends (latent KV, MQA decode) | `v1/attention/backends/mla/` | ☐ T2 | | `planned: specs/mla-backends.md` |
| Mamba1/Mamba2/short-conv/linear backends | `v1/attention/backends/` | ☐ T2 | | `planned: specs/mamba-backends.md` |
| Encoder / cross-attention | backends | ☐ T2 | | `planned: specs/encoder-cross-attention.md` |

## 12. Platforms & hardware

The exhaustive platform and 13-target CUDA inventory is
[backend-matrix.md](backend-matrix.md). Build-source availability is not runtime
support; every target needs build, correctness, trace, performance and memory
evidence.

| ID | Platform block | State | Grounded summary | Detailed table / spike |
|---|---|---|---|---|
| `BACKEND-CUDA-SM121` | GB10/sm121a | `PARTIAL` | gate workload built, traced, token/perf gated; full component-family coverage is open | [backend row](backend-matrix.md#cuda-target-rows) |
| `BACKEND-CUDA-OTHER` | vLLM sm70/75/80/86/87/89/90/100/101/103/110/120 targets | `INVENTORIED` | our global build defaults to 121a; no other target is validated | [backend matrix](backend-matrix.md), [CUDA inventory](specs/cuda-architecture-inventory.md) |
| `BACKEND-CPU` | production CPU | `PARTIAL` | persistent threadpool + chunked GEMM/row dispatch is 1/3/20-thread bit-identical and TSAN-clean; idle-host performance/RSS gate and compute-in-quant remain open | [backend matrix](backend-matrix.md) |
| `BACKEND-ROCM` | ROCm | `INVENTORIED` | source/dispatch spike required; no "one flag" support claim | [backend matrix](backend-matrix.md) |
| `BACKEND-MLX` | Apple Metal through MLX | `INVENTORIED` | M4/16 GB host available; runtime absent | [backend matrix](backend-matrix.md) |
| `BACKEND-VULKAN` | Vulkan | `INVENTORIED` | runtime absent | [backend matrix](backend-matrix.md) |
| `BACKEND-XPU` | Intel XPU | `INVENTORIED` | loyal upstream-platform port, runtime absent | [backend matrix](backend-matrix.md) |
| `BACKEND-ANE` | encoder/pooling accelerator | `INVENTORIED` | specialized CoreML route only | [backend matrix](backend-matrix.md) |

## 13. Loading, tokenizer, config

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| safetensors container + model-specific loading | `model_loader/default_loader.py`, `models/utils.py` | `PARTIAL` T0 | Qwen-specific resolvers work; no generic WeightsMapper/stacked-parameter framework | `planned: specs/safetensors-loader.md` |
| GGUF reader + Qwen MoE transform mapping + embedded BPE | no pinned-vLLM implementation; llama.cpp-compatible deviation | `PARTIAL` T0 | 35B real-file parity works; dense and format/compute breadth remain open | `planned: specs/gguf-loader.md` |
| HF `tokenizer.json` byte-level BPE + incremental detok | `tokenizers/registry.py`, `tokenizers/hf.py` | `ANCHOR-BACKFILL` T0 | bounded Qwen byte-BPE slice; deviations need leaf accounting | `planned: specs/hf-tokenizer.md` |
| SentencePiece tokenizer | same | ☐ T1 | needed for Llama-2-era / Gemma checkpoints | `planned: specs/sentencepiece.md` |
| Config surface: dataclass-for-dataclass structs, `vllm serve`-compatible flags | `vllm/config/` | `PARTIAL` T0/T1 | gate-model structs and a small CLI subset incl. explicit scheduler sequence/token budgets only | `planned: specs/config-surface.md` |
| Sharded-state / tensorizer / runai / BnB loaders | `model_loader/` | ☐ T3 | | `planned: specs/loader-longtail.md` |

---

## Reading the matrix into work

1. Follow the ordered portfolio in roadmap_v1.md and select an unblocked
   `INVENTORIED` row or row block.
2. Claim its stable ID in coordination.md and write the real
   `specs/<slug>.md` spike. A plain `planned: specs/...` cell is deliberately
   not a link and does not make the row `READY`.
3. Merge the spike, change the row to `READY`, then delegate implementation in
   an isolated worktree. GPU jobs use one `flock /tmp/gpu` across the whole run.
4. `DONE` requires exact local code, ported-test/evidence and real-spec anchors,
   plus the roadmap, README, inventory, ledger and state updates in the same
   change. Legacy emoji rows are migrated by the anchor-backfill workstream.

Currently open T0 debt (pre-existing, tracked): `/metrics` core set (§9),
`SERVE-GATE-ONLINE`, and `SERVE-E2E-NIGHTLY`. GGUF real-file parity A2 passed;
its remaining format/model breadth stays tracked in §5/§13.
