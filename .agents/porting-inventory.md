# vLLM → vllm.cpp Porting Inventory (living parity record)

**Upstream reference:** `/home/mudler/_git/vllm` @ sync point in
[upstream-sync.md](upstream-sync.md).
**Rule:** every item lists its upstream path. We port features/capabilities 1:1 —
same algorithms, same semantics, same names — so that future upstream PRs can be
translated mechanically. Deviations (there are few, all forced by the no-Python
constraint) are recorded in §9.

**This document is kept up to date at all times** — it is the static record of
what vLLM has vs what we have:

- **Status markers**, appended inline to any item as work lands:
  (none) = not started · 🚧 `<our-commit>` = in progress · ✅ `<our-commit>` =
  ported & parity-tested. Optionally suffix upstream references:
  `(vllm#12345)` for the upstream PR(s) an item's behavior comes from.
- **Every PR/change we introduce** must (a) flip the relevant markers here and
  (b) append a row to [parity-ledger.md](parity-ledger.md) recording what the
  change does compared to vLLM, with upstream paths/PR references where they
  exist. No change to engine behavior merges without its ledger row.
- When the upstream sync point advances and brings new features, add them here
  (with their vLLM PR references) in the right tier — the inventory tracks what
  vLLM has, even for things we haven't scheduled.

**Tiers**
- **T0 — the Gate.** Everything required to (a) serve Qwen3.6-35B-A3B (NVFP4) and
  Qwen3.6-27B (NVFP4) on the DGX Spark (GB10, sm_121) at **vLLM-parity prefill and
  decode throughput under large concurrency**, from both safetensors and **GGUF**
  checkpoints; (b) ship as a **library** (llama.cpp-style) with an example
  CLI + OpenAI server; (c) pass e2e test suites.
- **T1 — serving completeness.** Single-GPU text serving at feature parity for
  everyday use.
- **T2 — breadth.** Model zoo, spec decode, LoRA, offload, multi-GPU.
- **T3 — not ported as-is.** Python/torch-specific machinery replaced by C++
  equivalents, or out of scope; each entry says what replaces it.

---

## 1. Engine core & scheduling (`vllm/v1/`)

| Item | Upstream | Tier |
|---|---|---|
| EngineCore + `step()` (schedule → execute → sample → update) | `v1/engine/core.py` | T0 ✅ `88821f3` (step loop + add/abort; batch-queue/async/DP/grammar deferred) |
| Executor pass-through seam (over the runner; no collective_rpc/Ray/multiproc) | `v1/executor/{abstract,uniproc_executor}.py` | T0 ✅ `88821f3` (ModelRunnerBase ABC + direct pass-through) |
| LLMEngine (add_request/step/generate; offline C++ LLM) | `v1/engine/llm_engine.py` | T0 ✅ `c1859d9` (synchronous text path; e2e greedy/concurrent/streaming/stop) |
| `SamplingParams` (+ Verify/PostInit == `__post_init__`, eos on params) | `vllm/sampling_params.py` | T0 ✅ `b888645`/`fabf48f` (structural port) + PostInit/Verify now RUN by the InputProcessor `73a9509` (M1.8 constructing unit — carry closed; NOTE: the `verify(model_config)` validator family — only `_validate_logprobs` matters — is dropped at T0, inert since logprobs tracking is deferred) |
| `Request` + `RequestStatus` (12-status ordering, IsFinished, FinishReason map) | `v1/request.py` | T0 ✅ `4320dae`/`a43eaf8` (behavioral unit tests; eos moved to SamplingParams per upstream) |
| EngineCore I/O types (`EngineCoreRequest`/`EngineCoreOutput(s)`, `ModelRunnerOutput`, `SamplerOutput`) | `v1/engine/__init__.py`, `v1/outputs.py` | T0 ✅ `cd13ec3` (structural port) |
| `RequestOutput` / `CompletionOutput` (public result carriers; FinishReason→string) | `vllm/outputs.py` | T0 ✅ `4d477eb` (+ prompt_logprobs opaque placeholder at M1.1 close-out; logprobs payloads deferred to sampler unit) |
| `step_with_batch_queue` (pipelined batch queue, deferred sampling) | `v1/engine/core.py` | T1 |
| Busy loop + input/output queue split (in-proc analog of ZMQ boundary) | `v1/engine/core.py`, `core_client.py` | T0 ✅ `core_proc.{h,cpp}` + `core_client.{h,cpp}` (W1 `ENG-CORE-BUSY-LOOP`: EngineCoreProc busy loop, shutdown drain/abort, WAKEUP + ENGINE_CORE_DEAD sentinels, InprocClient engine thread; tests `test_engine_core_proc.cpp` ← upstream `tests/v1/engine/test_engine_core_client.py`; GPU G1/G4 gating pending; UTILITY/DP/aborts-queue/batch-queue deferred) |
| InputProcessor (validate, tokenize, build EngineCoreRequest) | `v1/engine/input_processor.py` | T0 ✅ `73a9509` (text path; runs PostInit/Verify + max_tokens default + eos/stop wiring; mm/lora/embeds/pooling deferred) |
| OutputProcessor + RequestState + incremental Detokenizer | `v1/engine/output_processor.py`, `detokenizer.py` | T0 ✅ `c7ba3a5` baseline + W2 `GATING`: process_outputs detokenize/string-stop/DELTA-CUMULATIVE-FINAL_ONLY plus thread-safe single-slot `RequestOutputCollector`, per-request queue handoff, abort-final output and error propagation; logprobs/pooling/parallel-sampling remain deferred |
| AsyncLLM-equivalent streaming API + sync LLM API | `v1/engine/async_llm.py`, `llm_engine.py` | T0 🚧 W2 `ACTIVE`: `AsyncLLM` owns the EngineCoreProc/output-handler threads, concurrent add/generate/abort, collector streams and clean shutdown. The c32 cpp-httplib defect now has fixed `max_num_seqs + 4` delivery workers plus a legacy A/B toggle; the 32-client control-reserve test is 100×/sanitizer-green. Exact GPU G1/G3-G6 remain. Synchronous `LLMEngine` stays for offline/compatibility use |
| Unified scheduler: token-budget, **no prefill/decode distinction** | `v1/core/sched/scheduler.py` | T0 ✅ `4f12158` (schedule() running-first + chunked prefill + FCFS preemption; update_from_output + check_stop; priority/spec/structured/async deferred behind 1:1 stubs) |
| Chunked prefill (`enable_chunked_prefill`, on by default) | `config/scheduler.py` | T0 ✅ `4f12158` |
| Budgets: `max_num_batched_tokens`, `max_num_seqs`, `max_num_scheduled_tokens` | `config/scheduler.py` | T0 ✅ `2f0ea69` |
| Preemption (FCFS tail pop, recompute) + `SchedulerOutput` new/cached diff protocol | `v1/core/sched/{scheduler,output}.py` | T0 ✅ `4f12158` (`c65e650` SchedulerOutput/NewRequestData/CachedRequestData in the MRV2 shape — prefill_token_ids + resumed-as-new fold) |
| FCFS request queue | `v1/core/sched/request_queue.py` | T0 ✅ `2f0ea69` |
| Priority scheduling (`policy="priority"`) | same | T1 🚧 GATING (W4 `ENG-PRIORITY-SCHED`): `PriorityRequestQueue` (heap by `Request.__lt__`) + priority preemption (victim = max `(priority, arrival_time)`) + `priority` plumbed through `Request`/`EngineCoreRequest`/OpenAI field + `SchedulerPolicyFromString`; default stays FCFS. Tests ported: `tests/vllm/v1/test_scheduler.cpp` (12 `test_priority_scheduling_*` cases ← `tests/v1/core/test_scheduler.py:2382,2978`) + `tests/vllm/v1/test_request_queue.cpp` (priority-queue cases + seeded random property ← `tests/v1/core/test_priority_scheduler_random.py`). CPU 93/93 green; GPU G1 token-exact A/B deferred (GPU held) |
| Partial-prefill concurrency (`max_num_partial_prefills`, long-prefill threshold/limits) | `config/scheduler.py` | T1 |
| Async scheduling (overlap schedule with execution) | `v1/core/sched/async_scheduler.py` | T1 `READY` (W3; vLLM default-ON mirror obligation, distinct from W2's asynchronous frontend) |
| `scheduler_reserve_full_isl`, pluggable `scheduler_cls`, `stream_interval` | `config/scheduler.py` | T1 |
| Cascade attention (shared-prefix batch attention) | `config/model.py::disable_cascade_attn` | T2 |
| DBO / ubatch overlap | `config/parallel.py::enable_dbo` | T2 |
| MoE sequence parallelism without requiring DP (v0.25.0, 1.9–5.0% reported E2E gain) | `config/parallel.py::use_sequence_parallel_moe`, `distributed/parallel_state.py` | T2 |
| Spec-decode scheduling hooks (`spec_token_ids`, lookahead slots) | `v1/core/sched/scheduler.py` | T2 (with §6) |

## 2. KV cache management (`vllm/v1/core/`)

| Item | Upstream | Tier |
|---|---|---|
| ✅ `5ee2301` BlockPool: free list, ref counts, LRU eviction, `BlockHash→block` map | `v1/core/block_pool.py` | T0 |
| ✅ `75caf38` KVCacheManager: `allocate_slots` (accounting/watermark/OOM→nullopt/admission-cap), `get_computed_blocks`, `free`; hybrid prefill | `v1/core/kv_cache_manager.py` | T0 |
| ✅ `5ee2301` Prefix caching (parent-chained block hashing; sha256_cbor byte-exact vs upstream — default deviates to sha256_cbor since upstream sha256/pickle default isn't cross-language reproducible; align path hash_block_size≠block_size + cache_partial_block deferred behind 1:1 stubs) | `v1/core/kv_cache_utils.py`, `config/cache.py` | T0 |
| ✅ `75caf38` **Hybrid KV coordinator + per-group managers** (full-attn group + GDN/mamba state group — the gate models are hybrid): SingleTypeKVCacheManager (FullAttention left→right multi-block prefix + Mamba right→left single recurrent state), HybridKVCacheCoordinator cross-group MIN-intersection prefix hit. Mamba `align` mode deferred behind a 1:1 stub | `v1/core/kv_cache_coordinator.py`, `single_type_kv_cache_manager.py` | **T0** |
| ✅ `75caf38` Specs: `FullAttentionSpec`, `MambaSpec` (GDN state), `KVCacheGroupSpec`, `KVCacheConfig` — `page_size_bytes` byte-exact vs upstream. Local-attention spec extensions are tracked by the T1 row below; `MLAAttentionSpec` remains deferred | `v1/kv_cache_interface.py` | T0 |
| Mamba/GDN prefix-cache retention (`mamba_cache_mode` align, PR #45845) | `v1/core/` | T1 |
| 🚧 W1 `SlidingWindowSpec`/manager and W3 `ChunkedLocalAttentionSpec`/manager are implemented/`GATING`: concrete sizing/grouping, registry dispatch, local prefix/recycling policies, admission caps, hybrid-disabled conversions and ported deterministic CPU properties pass; attention/model/oracle/runtime closure remains | `v1/kv_cache_interface.py`, `v1/kv_cache_spec_registry.py`, `v1/core/single_type_kv_cache_manager.py`, `v1/core/kv_cache_utils.py` | T1 |
| `MLAAttentionSpec` (+ latent KV) | same | T2 (with DeepSeek-family) |
| `CrossAttentionSpec`, `EncoderOnlyAttentionSpec` | same | T2 |
| KV quantization: fp8 KV cache (`cache_dtype=fp8*`) | `config/cache.py`, `layers/quantization/kv_cache.py` | T1 |
| KV quantization: nvfp4 / per-token-head / turboquant modes | `config/cache.py` | T2 |
| KV offload (CPU tiering, LRU/ARC policies) | `v1/kv_offload/` | T2 |
| External KV-cache provider ABI + LMCache MP/in-process interoperability | `config/kv_transfer.py`; `distributed/kv_transfer/kv_connector/v1/{base,lmcache_connector,lmcache_mp_connector}.py` | T2 |
| KV connector breadth (NIXL/Mooncake/MultiConnector/PD-disaggregation) | `distributed/kv_transfer/` | T2 |
| KV events (ZMQ publish of block create/evict) | `config/kv_events.py` | T2 |
| Sizing: `gpu_memory_utilization`, `num_gpu_blocks_override`, `kv_cache_memory_bytes` | `config/cache.py` | T0 |

## 3. Model runner / worker (`vllm/v1/worker/`)

We port the **Model Runner V2** (`v1/worker/gpu/` package — `model_runner.py`,
`input_batch.py`, `block_table.py`, `attn_utils.py`, `states.py`), not the legacy
`gpu_model_runner.py`: MRV2 is already the default for all dense models at the
current pin, and features such as DSpark are V2-only. v0.25.0 deletes the legacy
libtorch PagedAttention v1/v2 CUDA kernels; our live `vt::PagedAttention` is the
backend-neutral paged-KV contract backed by current FA2/fallback kernels, not a
port of those deleted kernels, so there is no corresponding local code to retire.

| Item | Upstream | Tier |
|---|---|---|
| Persistent InputBatch (incremental add/diff/swap-remove) | `v1/worker/gpu/input_batch.py` | T0 ✅ `62fdfca` (add/remove/condense, V1 host-array algorithm; staged device storage deferred M2) |
| BlockTable tensors + slot mapping | `v1/worker/gpu/block_table.py` | T0 ✅ `62fdfca` (BlockTable+MultiGroupBlockTable host-array, slot_mapping=block_id*bs+offset; staged tensors deferred M2) |
| Step input build: `query_start_loc`, `seq_lens`, positions, logits indices | `v1/worker/gpu/model_runner.py` | T0 ✅ `62fdfca` (update_states+prepare_inputs matched 1:1 vs `_prepare_inputs`; LoRA/spec/mm slot state deferred M2/T1) |
| `CommonAttentionMetadata` contract → per-backend builders | `v1/attention/backend.py`, `v1/worker/gpu/attn_utils.py` | T0 ✅ `bd47ce3` (T0 field set + MakeCommonAttentionMetadata from step-inputs; FastPrefill/CrossAttn/dcp/sparse fields deferred) |
| Split execute/sample (`ExecuteModelState`) for deferred sampling | `v1/worker/gpu/model_runner.py` | T0 ✅ `9949f87` (GPUModelRunner: execute_model→nullopt / sample_tokens; decode-first reorder + InputBatch::swap_states; logits_indices gather; sampled-token write-back; four-way ordering identity gated) |
| KV tensor allocation from `KVCacheConfig` | same | T0 ✅ `9949f87` (per-full-attn-layer (nb,2,bs,H,D) + per-GDN-layer ssm/conv-state buffers, f32 at T0; bf16 cache = M2) |
| Batched paged forward (Qwen3.6: ReshapeAndCache+PagedAttention + batched GDN persistent-state + GDN-zeroing) | `models/qwen3_next.py`, `mamba/qwen_gdn_linear_attn.py` | T0 ✅ `f1ae018` (paged==dense bit-exact; M0.9 dense kept as ForwardDense) |
| CUDA graph capture/replay for decode (our own capture; no torch) | `v1/worker/gpu/cudagraph_utils.py`, `config/compilation.py::cudagraph_mode` | **T0** (parity at high concurrency needs it — MoE decode is launch-bound) |
| Dummy runs / warmup / memory profiling | same | T0 |
| LoRA runner mixin | `v1/worker/lora_model_runner_mixin.py` | T2 |
| ubatching, DP/PP utils, EPLB | `v1/worker/gpu/{dp,pp,eplb}_utils.py`, `ubatching.py` | T2 |

## 4. Attention & hybrid-layer backends (`vllm/v1/attention/`)

| Item | Upstream | Tier |
|---|---|---|
| Backend interface: `AttentionBackend/Impl/MetadataBuilder` | `v1/attention/backend.py` | T0 ✅ `bd47ce3` (ABCs + flash NHD get_kv_cache_shape) |
| `reshape_and_cache` (write K/V into paged NHD cache at slot_mapping) | `csrc/.../cache_kernels.cu::reshape_and_cache_flash` | T0 ✅ `e231196`→`7de4f0c` (vt::ReshapeAndCache, stride-based NHD write CPU+CUDA; GB10 gates pass; other CUDA targets unvalidated) |
| Paged attention for full-attn layers on sm_121 (bf16, GQA 16/2, partial RoPE) — FlashInfer-class performance is the bar; strategy in §9 | ref: `v1/attention/backends/{flashinfer,triton_attn,flash_attn}.py` | T0 🚧 on GB10 gate workloads (CPU/CUDA parity + full 27B/35B paged-engine greedy correctness gates pass. Exact clean pushed-`b5c6e4f` 27B online evidence is valid but below the every-axis floor: median total ratios c1→c32 are 0.9933/0.9520/0.9657/0.9760/1.0213/1.0218×, with 4/4/5/4/17/14 of 20 performance and 2/4 memory axes. W0/W1 device-residency component A/Bs remain 1.021239×/1.006246× with structural copy reduction; inherited pools fail zero-leak. HTTP is healthy/neutral. FP4 W2 materially improves the grid and passes focused correctness/sanitizer gates, but its trace shows the next gap is runtime tactic selection; W3 precedes device-residency W2 re-ranking and any 35B run); broader attention/backend/SM coverage lives in `kernel-matrix.md` and remains open |
| **GDN backend**: metadata segmentation (prefill/decode/spec) | `v1/attention/backends/gdn_attn.py` | T0 ✅ `370ddaf` (GDNAttentionMetadata decode/prefill split + has_initial_state mask + prefill rebasing; spec segments + align col-gather deferred; GDN-state zeroing = caller obligation, see state.md) |
| GDN chunked-scan prefill kernel (chunk gated delta rule) | `layers/fla/ops/chunk.py` (Triton ref), `flashinfer.gdn_prefill` (Blackwell) | T0 🚧 `ead59d6` (correctness-grade sequential; chunked perf kernel M2.3) |
| GDN fused decode recurrence (sigmoid-gating delta rule update) | `layers/fla/ops/{fused_sigmoid_gating,fused_recurrent}.py` | T0 ✅ `ead59d6` (correctness-grade) |
| GDN post-conv prep (q,k,v,g,beta + L2 norm) + causal conv1d fn/update | `layers/fla/ops/fused_gdn_prefill_post_conv.py`, `layers/mamba/ops/causal_conv1d.py` | T0 ✅ `ead59d6` (l2norm + conv1d only; packed q/k/v split + g/beta prep is M0.9 — gdn-semantics §6) |
| Sliding-window attention | `model_executor/layers/attention/attention.py`; `v1/attention/backends/flash_attn.py`; pinned FA2 local mask chain | T1 🚧 W2 `GATING`: generic config/semantic window + CPU/portable-CUDA/FA2 operator port; CPU/sanitizer and sm_121a compile-only checks pass, while GPU runtime, positive-model/oracle/trace/performance/memory gates remain pending |
| Chunked-local attention wrapper | `model_executor/layers/attention/chunked_local_attention.py`; `v1/attention/backends/utils.py` | T1 🚧 W4 `GATING`: cached wrapper, exact virtual Q/K batches, reusable block-table transforms, cudagraph rejection, spec emission and ordinary-backend delegation pass CPU/reference/sanitizer gates; W3 KV is also `GATING`, while Llama4/model/oracle/GPU gates remain separate |
| YaRN + long-context RoPE factory/cache/apply | `model_executor/layers/rotary_embedding/{__init__,base,yarn_scaling_rope,mrope,llama3_rope,phi3_long_rope_scaled_rope,dynamic_ntk_scaling_rope,dynamic_ntk_alpha_rope}.py` | T1 🚧 W5-W8 `GATING`: typed modern/legacy parameters, memoized f32/bf16 cache/factory, shared supplied-cache apply, YaRN/MRoPE, exact Llama 3 bands, Phi-3 globally selected short/long caches, and both dynamic-NTK transforms pass fifteen pinned-source CPU oracle fixtures with sanitizers. CUDA compile/runtime, positive-model e2e, trace and every-axis gates remain |
| MLA backends (latent KV, MQA decode) | `v1/attention/backends/mla/` | T2 |
| Mamba1/Mamba2/short-conv/linear backends | `v1/attention/backends/{mamba*,short_conv,linear}_attn.py` | T2 |
| Encoder / cross-attention | backends | T2 |
| CPU attention backend (scalar/SIMD reference, for CI parity) | `v1/attention/backends/cpu_attn.py`, `csrc/cpu/` | T0 (correctness-grade only) |

## 5. Model architectures (`vllm/model_executor/models/`)

The current pin has **353 unique static architecture IDs**. The audited v0.25.0
target adds MOSS-Transcribe-Diarize, Laguna DFlash and Bailing MoE v2.5 MTP,
bringing the target inventory to **356 IDs** (373 category memberships, 310
implementation targets, 261 modules) plus a dynamic Transformers-compatible
path. The generic ordered architecture-to-type-erased-
factory contract is now implemented for the two architectures whose text paths
exist locally: `Qwen3_5ForConditionalGeneration` and
`Qwen3_5MoeForConditionalGeneration`. Live loading consumes the full
`config.architectures` list in order and mirrors pinned unknown,
previously-supported, and out-of-tree rejection; unimplemented IDs remain
unsupported rather than being misclassified by `num_experts`. The execution row
is `GATING` on its deferred two-model GPU no-regression campaign; see
[model-factory-registry.md](specs/model-factory-registry.md) and
`model-matrix.md`. Dynamic Transformers/terratorch/OOT runtime loading and
Python lazy-import/subprocess caching remain explicitly deferred.

| Family | Marquee members | Needs | Tier |
|---|---|---|---|
| **Qwen3.5/3.6 hybrid (incl. MoE)** | `Qwen3_5ForConditionalGeneration` (27B dense-hybrid, **VL multimodal** wrapper — vision_config present), `Qwen3_5MoeForConditionalGeneration` / `qwen35moe` (35B-A3B) | GDN layers ×3 : 1 gated full-attn (qk-norm, partial RoPE 64d, output gate), MoE 256e top-8 + shared expert (35B) / **dense SwiGLU MLP** (27B), GemmaRMSNorm-style `(1+w)` — ✅ `25326fc` (35B forward correctness-grade, **safetensors**; 16/16 greedy on GB10 = M0 exit; GGUF k-quant load M0.10). **27B (co-equal gate):** CPU-first scaffolding started — arch/quant surveyed (`.agents/specs/qwen27b-w4a4-notes.md`: dense hybrid, W4A4, **and a VL wrapper — text path first, ViT deferred**), CPU W4A4 emulation reference + skipping greedy-parity gate landed; dense loader + single-seq `ForwardDense` + batched PAGED `Qwen3_5DenseModel::Forward` (paged==dense CPU-anchored) + `GPUModelRunner` dense route all landed CPU-green; W4A4 GPU GEMM + oracle golden pending (GPU-gated). serving M1–M3 | **T0 (the gate)** |
| Dense decoders | Llama 3.x, Qwen2/3, Mistral, Gemma 2/3, Phi | GQA + RoPE + SwiGLU + RMSNorm (subset of T0 layer set) | T1 |
| MoE decoders | Mixtral, Qwen3-MoE (30B-A3B), GLM-4-MoE, OLMoE | FusedMoE 🚧 `65788b3` (correctness-grade eager; grouped-GEMM perf M2.2) | T1 |
| Qwen3-Next | `Qwen3NextForCausalLM` | same stack, interleaved-GQA weight layout | T1 |
| Hybrid others | Jamba, NemotronH, FalconH1, Zamba2, LFM2, Kimi-Linear | mamba1/2 kernels | T2 (`Bamba` is previously-supported, not in the pinned live registry) |
| MLA family | DeepSeek V3/V3.2/V4, Kimi K2.5 | MLA wrapper, latent KV, sparse indexer | T2 |
| MTP/EAGLE draft models | `Qwen3_5MTP`, `Qwen3_5MoeMTP`, EAGLE3 | spec decode (§6) | T2 |
| Embedding/pooling/reranker | BERT-family, GTE, Qwen embed | bidirectional attn, poolers | T2 |
| Multimodal / ASR / encoder-decoder | Qwen3-VL, Whisper, … | vision/audio towers, mRoPE full, encoder cache | T3 (deferred; revisit after T1) |

## 6. Layers, quantization, sampling

**Layer library** (`model_executor/layers/`) — T0 set: `RMSNorm` + fused-residual,
`GemmaRMSNorm`, `RMSNormGated` (GDN), `silu_and_mul`, RoPE `default` NeoX +
partial-rotary (+ MRoPE sections parsing; full MRoPE T3 with multimodal), linear
layers with the `QKVParallelLinear`/`MergedColumnParallel` **weight-stacking
semantics** (`qkv_proj`, `gate_up_proj` fusion — needed for checkpoint mapping even
single-GPU), `VocabParallelEmbedding`/`ParallelLMHead`, `LogitsProcessor` (gather →
lm_head), FusedMoE layer with router/top-k/shared-expert. Fused qk-norm+RoPE+gate
kernel (`layers/fused_qk_norm_rope.py`) — T0 as perf lever. T1: remaining
activations, YaRN/llama3/longrope/dynamic-NTK rope scaling (YaRN is T1 — Qwen3.6
long-context uses it). T2: the rest.

**Quantization** (`layers/quantization/`, registry in `__init__.py`):

| Method | Upstream | Tier |
|---|---|---|
| **NVFP4 gate slices** — ModelOpt W4A16 experts (35B) + compressed-tensors W4A4 dense (27B) | `quantization/modelopt.py`, `compressed_tensors/` | **T0 ✅ correctness/support; performance W3 ACTIVE** — both native CUDA paths and full gate workloads pass on GB10. W2 includes distinct hybrid buckets, complete key/single-flight/capture handling, all 32 SM12 tactics, merged gate/up CT semantics, and fused one-input SiLU→NVFP4. W3-A delayed timing and W3-B pre-serve all-bucket warmup are correctness/safety-gated; W3-B's repeated component is **1.000293×** but strict-fails **15/20 timing + 2/4 memory**, keeps only **20/80** tactic IDs stable and therefore earns no speed credit. Its corrected old-oracle trace closes the original wide-tactic structural mismatch. `b5c6e4f` is historical and `3cc490c` is VOID. Canonical vLLM 0.25.0 + FlashInfer 0.6.13 is active; immutable `9cc7191` source/build/oracle/exact-corpus preflight passes, while its locked model gate/grid/trace has not run. No ratio binds and no topology repair is selected before that trace; W3-C remains optional. The accepted [small-M dispatch spike](specs/nvfp4-small-m-dispatch.md) keeps iterations independent; generic quant-config/backend breadth remains in `quantization-matrix.md` |
| **GGUF materialization** — F32/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K | **vllm.cpp deviation**: pinned vLLM has no GGUF load format; llama.cpp is the container/quant reference | **T0 🟡** loader + synthetic per-layout tests + real APEX Q3/Q4/Q5/Q6/Q8 greedy parity pass. The llama.cpp-derived CPU threadpool/chunked-op prerequisite is implemented and correctness-gated (1/3/20 full suites + TSAN), but its B4 speed/RSS gate is pending. All weights still expand to bf16; no compute-in-quant/llama.cpp speed parity. F16 is reader-only, not executable; BF16/Q2_K/IQ/TQ/Q1/MXFP4/NVFP4 execution remains open. |
| fp8 (W8A8, e4m3) | `quantization/fp8.py`, ModelOpt | **T0 gate slice ✅ / generic T1 🟡** — 35B static per-tensor W8A8 projections are native and gated; other scale/activation/config/KV modes remain open |
| MXFP4 / MXFP8 | `quantization/mxfp4.py`, modelopt | T1 |
| AWQ/GPTQ (+Marlin), compressed-tensors int schemes | various | T2 |
| bitsandbytes, torchao, quark, INC, … | various | T3 (niche/other-vendor) |

Threadpool test-port mapping (row `QUANT-GGUF-CPU-THREADPOOL`): llama.cpp
`tests/test-barrier.cpp` → `tests/vt/test_cpu_threadpool.cpp:63` (repeated
barrier/chunk reach-all-threads), and `tests/test-thread-safety.cpp` → the
reduced concurrent-submit case at `:133`; the full multi-model/context case is
checked in at `:173` with a real doctest skip assigned to
`SERVE-E2E-NIGHTLY`. The local 1/3/20 byte-determinism battery is at `:399`.

**Sampling** (`vllm/sampling_params.py`, `v1/sample/`):
T0 fields: `n`, `temperature`, `top_p`, `top_k`, `min_p`, `presence/frequency/
repetition_penalty`, `seed`, `stop`/`stop_token_ids`, `include_stop_str_in_output`,
`ignore_eos`, `max_tokens`, `min_tokens`, `logprobs`, `detokenize`,
`skip_special_tokens`, `output_kind` (CUMULATIVE/DELTA/FINAL_ONLY); pipeline order
exactly as `v1/sample/sampler.py` (fp32 logits → processors → greedy short-circuit
→ temperature → penalties → top-k/top-p → seeded sample → logprobs), GPU top-k/
top-p kernels (`v1/sample/ops/`).
**✅ Sampler ported (M1.7, `38a8846`):** the full ordered `Sampler.forward`
pipeline (raw-logprobs snapshot → allowed-ids → bad-words → min-tokens/logit-bias →
penalties → greedy/temperature/min-p/top-k/top-p/random → where(temp<eps) merge →
gather_logprobs+ranks), `SamplingMetadata`/`LogprobsTensors`/`make_sampling_metadata`,
core ops (ApplyTemperature/GreedyArgmax/ApplyTopKTopP/ComputeProbs/ComputeLogprobs/
RandomSample) + penalties/min-p/logit-bias/token-mask/allowed-ids, CPU+CUDA (CUDA
dgx-pending). **logit_bias/allowed_token_ids/bad_words landed at T0** (moved up
from T1 below — the OpenAI-serving MVP needs them). Greedy = bit-exact parity gate;
random RNG = exponential-noise gumbel-max, distribution-correct, **torch-Philox
bit-exact parity deferred to T1**. Deferred (marked stubs): logprob_token_ids
(generative-scoring), spec-decode bonus-token, thinking-budget, logprobs_mode
variants beyond raw/processed. **InputBatch-side tracking of seeds/min_p/min_tokens/
logit_bias/allowed/bad_words + num_logprobs is an M1.8 wiring dependency**
(make_sampling_metadata emits empty defaults today — the InputBatch doesn't store
them yet; SamplingMetadata carries the fields ready to populate).
**T0 (MVP, user-mandated): grammars/structured outputs** — ✅ **`a66eef6`** (M3.4).
The INTEGRATION layer ported 1:1 (`v1/structured_output/` manager + backend_types
ABCs + request key; `get_grammar_bitmask` scheduler integration + `GrammarOutput`
+ `apply_grammar_bitmask` runner masking, set=allowed→-inf; EngineCore→sample_tokens
seam) — the parity surface. The grammar ENGINE is a from-scratch **NATIVE backend
(§9, ORIGINAL)** behind that seam: GBNF/EBNF parser + push-down FSM + token-byte
trie (sub-O(vocab) fill; fill==accept invariant guarded by an exhaustive
differential test), covering `json` (schema→GBNF), `json_object`, `regex`,
`choice`, `grammar`(EBNF/GBNF) + OpenAI `response_format`. **The xgrammar C++ core
is DEFERRED to a later parity-completion milestone** (a 2nd backend behind the SAME
proven seam — mirrors upstream's own 4 pluggable backends; not a parity deviation).
Deferred: STRUCTURAL_TAG, reasoning-gating, spec-decode multi-row, key-order
flexibility, whitespace-flexibility/exotic-schema parity (xgrammar-only until vendored).
T1: `prompt_logprobs`, `logprob_token_ids`, additional backends
(guidance/outlines), reasoning parsers, beam search wrapper, thinking budget,
repetition detection, torch-Philox bit-exact random parity. T2: rejection
sampler (spec decode), routed-experts return. (`logit_bias`/`allowed_token_ids`/
`bad_words` primitives were ported at M1.7 `aac5138`; their SamplingParams and
OpenAI request/payload wiring remain T1.)

**Spec decode** (`v1/worker/gpu/spec_decode/`): after speed parity, T1 starts
with Qwen3.5/3.6 MTP (the gate checkpoints ship their heads), then DFlash and
the user-promoted DSpark path. Tokenizer-agnostic TLI heterogeneous-vocabulary
mapping is a distinct T1 row; ngram, EAGLE3 and suffix remain T2. M-mtp-0 is
`GATING`: the optional BF16 `mtp.*`
safetensors loader (`src/vllm/model_executor/models/qwen3_5_mtp.cpp:271`) and
standalone dense/MoE head (`src/vllm/model_executor/models/qwen3_5.cpp:3359`)
mirror `models/qwen3_5_mtp.py:63-165`, share the target embedding/lm-head, and
have ported loader/direct-hidden tests
(`tests/vllm/v1/spec_decode/test_mtp_speculator.cpp:201,225,299,331`) plus a
two-checkpoint oracle dump/runner (`tools/parity/dump_qwen3_5_mtp.py:1`,
`tests/parity/test_op_parity.cpp:1226`). GPU head parity is still pending; no
spec-token scheduler, rejection sampler, GDN snapshot rollback, GGUF MTP, or
user-visible speculative-decoding path is claimed. Those land in M-mtp-1+
before this becomes supported.

## 7. Serving surface (`vllm/entrypoints/`)

| Item | Upstream | Tier |
|---|---|---|
| Basic `/v1/completions`, `/v1/chat/completions` SSE transport, `/v1/models`, `/health`, `/version` | `entrypoints/openai/` | T0 **partial** `23d9f2c` + W2 `GATING`: live AsyncLLM streams, disconnect abort and concurrent scheduling remain; deterministic fixed delivery capacity for all `max_num_seqs` streams plus four control workers is CPU/sanitizer-green and GPU-classified healthy/steady-state-neutral. `/health` is liveness-only; `/metrics`, full fields, logprobs and remaining GB10 gates stay open |
| Completion/chat `stream_options` final + continuous usage and force mode | `entrypoints/openai/{engine,completion,chat_completion}/`, `entrypoints/serve/utils/api_utils.py` | T1 **GATING**: native prompt/output-ID counts, empty-choice terminal usage before `[DONE]`, continuous choice/role usage, non-stream validation and `--enable-force-include-usage` are CPU/ASan/UBSan/TSan-gated; fresh 27B+35B online closure remains |
| Chat templating (bounded Qwen3.6 Jinja subset; engine: minja-style) | `renderers/hf.py`, `entrypoints/chat_utils.py` | T0 **anchor-backfill** `a99a65e` (original minja-subset engine; generic template/parser breadth remains open) |
| **Tool/function calling** (user-mandated MVP): `tools`/`tool_choice` in chat API, auto-tool-choice, streaming tool-call deltas, Hermes parser first; upstream Qwen3Engine and other parser families remain T1 | `tool_parsers/`, `entrypoints/openai/chat_completion/` | T0 **partial** `18e3efb` (Hermes parser + streaming; local `qwen3` is a Hermes alias, not upstream Qwen3Engine parity; **tool_choice=auto RELAXED via a native LAZY grammar matcher behind vLLM's STRUCTURAL_TAG seam** — free text until `<tool_call>`, then constrain; required/named forced; Coder-XML/Mistral/pythonic parsers deferred) |
| Unified Streaming Parser Engine for tool calls + reasoning | `parser/engine/` | T1 (v0.25.0 inventory; token-ID scanner, event stream, serving adapters and replay tests) |
| Opt-in per-request timing metrics in chat/completion responses | `entrypoints/generate/base/serving.py`, OpenAI protocol/serving | T1 (v0.25.0 inventory; streaming/non-streaming and multi-output suppression) |
| `/tokenize`, `/detokenize`, `/ready`, `/ping`, `/server_info`, `/reset_prefix_cache` | various routers | T1 |
| `/v1/embeddings`, `/pooling`, `/score`, `/rerank` | pooling routers | T2 |
| `/v1/responses`, `/v1/messages` (Anthropic-style), audio endpoints | responses/messages routers | T2 |
| Sleep/pause/resume, LoRA load/unload, profiling, RL weight-update endpoints | various | T2–T3 |
| CLI: `serve`, `bench {latency,throughput,serve}`, `chat`, `complete` | `entrypoints/cli/` | T0 **partial** (separate example server + one in-process bench), T1 matching command families/rest |
| Offline API: `generate`, `chat`, streaming enqueue; `get_metrics` | `entrypoints/llm.py` | T1 parity surface; bounded synchronous C/C++ entry points exist |
| Prometheus metric names **1:1** (`vllm:num_requests_running`, `vllm:time_to_first_token_seconds`, `vllm:kv_cache_usage_perc`, …) | `v1/metrics/` | T0 (core set), T1 (full set) |
| OTLP tracing | `config/observability.py` | T2 |

**Library packaging (llama.cpp-style) — T0 ✅ `0b252ec` baseline + W2 `GATING`:** core built as `libvllm`
(static + shared; the shared lib's linker version-script exports ONLY the 17
`vllm_*` C ABI symbols — nm-verified + ctest-enforced) with a stable **C API**
(`include/vllm.h`: `vllm_engine_load`/`free`, `vllm_complete` [blocking],
`vllm_complete_stream` + `vllm_token_callback` [streaming/early-stop],
`vllm_request_submit`/`cancel`/`wait`/`done`/`error`/`free` [additive
nonblocking callback delivery over the shared AsyncLLM],
`vllm_string_free`/`vllm_completion_free`, `vllm_last_error`, `vllm_version`;
opaque handles, no-throw-across-ABI, thread-local error, unique per-call request
ids — cgo/purego-friendly for LocalAI; W2 CPU/TSan/dlopen gates green, GB10
G1/G3-G6 pending). The
richer C++ API (`include/vllm/*.hpp` mirroring `LLM`/`AsyncLLM`) is later.
Examples: `examples/cli` ✅ (C-API client), `examples/server` ✅ (OpenAI server),
`examples/bench` (M2). DoD (LocalAI-style dlopen consumption) MET.

## 8. Config, loading, tokenizer

- **Config surface** (`vllm/config/`): port dataclass-for-dataclass as structs with
  identical field names/defaults — T0: `ModelConfig`, `CacheConfig`,
  `SchedulerConfig` (✅ `2f0ea69` — T0 field set incl. un-deferred watermark +
  scheduler_reserve_full_isl), `DeviceConfig`, `LoadConfig`, `VllmConfig`
  container + CLI arg names matching `vllm serve` flags. T1/T2: the rest as
  features land.
- **Weight loading**: safetensors iterator + `stacked_params_mapping`
  (`qkv_proj`/`gate_up_proj`) + `WeightsMapper` renaming — T0. GGUF loader
  (name mapping GGUF→vLLM params) — T0 (gate). Sharded-state, tensorizer,
  runai, BnB loaders — T3.
- **Tokenizer**: HF `tokenizer.json` byte-level BPE + Qwen/Llama-3 pretokenizers,
  incremental detokenization, special-token handling — T0 ✅ `0baa46e`.
  GGUF-embedded vocab (llama.cpp `tokenizer.ggml.*` metadata) — T0 (gate models
  must run from bare GGUF) ✅ `0baa46e`. SentencePiece — T1.

## 9. Deviations from upstream (forced, recorded)

1. **Compute layer**: torch/torch.compile/Triton/inductor (`vllm/ir/`,
   `vllm/kernels/`, `config/compilation.py`) are replaced by `vt::` — an eager,
   arena-allocated C++ tensor runtime with explicit CUDA kernels (ggml-inspired
   in spirit, not a ggml dependency). Kernel sources, in preference order:
   port vLLM/FLA Triton semantics to CUDA; adapt csrc/cpu SIMD templates for the
   CPU reference; vendor specific CUTLASS/FlashInfer kernels only when a
   benchmark proves we can't match them. GDN + NVFP4-MoE prior art exists in
   the killgate/phase llama.cpp patch series on dgx.casa.
2. **Process model**: ZMQ multi-process split becomes an in-process
   thread + queue boundary with the same interface shape (multi-process remains
   possible later).
3. **FastAPI/pydantic** → cpp-httplib + our JSON schema validation, preserving
   endpoint paths, request/response JSON, and error shapes exactly.
4. **Python plugin points** (custom logits processors, scheduler_cls, connectors)
   → C ABI callback registries, added per tier.
5. **GGUF as a first-class input** (upstream treats it as one loader among many;
   for us it is a gate, including NVFP4 GGUF extension types).
7. **Explicit owned-tensor model weights** (no `nn.Module`/`AutoWeightsLoader`).
   The dense 27B (`Qwen3_5ForConditionalGeneration`, text_config `qwen3_5_text`)
   loads through `LoadQwen3_5Dense` into `Qwen3_5DenseWeights` — mirroring the
   35B's `LoadQwen3_5Moe` — with the MoE block replaced by a `DenseMlpWeights`
   SwiGLU MLP. Two recorded remaps: (a) the compressed-tensors NVFP4 W4A4 tensor
   NAMES (`weight_packed`/`weight_scale`/`weight_global_scale`/`input_global_scale`)
   vs the 35B modelopt names (`weight`/`weight_scale`/`weight_scale_2`/`input_scale`);
   (b) on the CPU correctness path every W4A4 Linear is MATERIALIZED to bf16 at
   load (`MaterializeCtNvfp4Bf16Transposed` → `DequantCtNvfp4WeightToF32`) so the
   existing bf16 forward carries it — the fp4-resident tensor-core GEMM reuse is
   the later GPU step (qwen27b-w4a4-notes.md §5 steps 6-7), not a permanent
   deviation. Per-Linear bf16-vs-W4A4 routing is `IsQwen27QuantizedLinear`
   (encodes the checkpoint `ignore` list, §3.6). Text path only; the ViT/merger
   and MTP head are deferred stubs. The batched PAGED 27B forward is
   `Qwen3_5DenseModel::Forward` — same signature/structure as the 35B
   `Qwen3_5Model::Forward`, reusing the file-local `GdnBlockPaged`/
   `FullAttnBlockPaged`/paged machinery VERBATIM via `RunDenseLayerPaged` (the
   only delta vs `RunLayerPaged` is `DenseMlpBlock` in place of `MoeBlock`).
   Registry/type-erasure update (`c707602`): `GPUModelRunner` now carries one
   `LoadedModel*` and calls the registration's type-erased prepare/forward hooks;
   model-specific weights and both existing decode-graph objects live behind the
   concrete `LoadedModel`. Its concrete-weight constructor overloads are only
   compatibility adapters for synthetic tests. `LoadedEngine` likewise owns one
   `std::unique_ptr<LoadedModel>` instead of the `{moe,dense}` optional pair and
   obtains its KV spec plus already-gated dense/MoE scheduler policy through the
   factory. `FromModelDir` resolves `config.architectures` in declaration order
   before tokenizer/weight work, then calls the matched safetensors/GGUF loader;
   unrelated dense configs now receive the pinned reject-unknown error instead
   of falling through `num_experts==0`. The GGUF Qwen loader publishes the
   canonical MoE registration ID while retaining `qwen35moe`/`qwen3next` as its
   container `model_type`. Everything between runner and LLMEngine remains
   architecture-agnostic. CPU evidence: `test_model_registry.cpp` (112 active
   assertions + one tracked second-family skip), the live reject case in
   `test_model_loader_gguf.cpp`, deterministic dense `LoadedEngine`, runner and
   GGUF tests, and the full 94/94 suite. The implementation is `GATING`, not
   `DONE`, until both real gate-model greedy and adjacent-commit every-axis
   no-regression runs complete on dgx.
8. **Extension platforms** (T2): Apple Metal and Vulkan backends — upstream has
   no equivalent under `vllm/platforms/`; we add them through the mirrored
   Platform/AttentionBackend/vt-op seams so they behave as vLLM platforms
   would. Intel is NOT a deviation (upstream `platforms/xpu.py` is ported
   loyally). Strategy, explorations (MLX, ANE), and binding vt:: interface
   requirements: `backends.md`.
9. **Vendored CUTLASS (sm120a NVFP4 GEMM)**: `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu`
   is a 1:1 lift of vLLM's `cutlass_scaled_fp4_mm_sm120a`
   (`csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu` @
   e24d1b24) — the near-peak Blackwell block-scaled fp4xfp4 GEMM (cutlass example
   79b). The ONLY change is the host surface: `torch::stable::Tensor` → `vt::Tensor`
   (raw pointers), `torch::stable::empty`/DeviceGuard → `cudaMallocAsync` + our
   stream; the CollectiveBuilder config + Fp4GemmSm120 are verbatim. `vt::SwizzleBlockscale`
   is the lift of vLLM `swizzle_blockscale` (`nvfp4_utils.py:13-53`). This is the
   "vendor a CUTLASS kernel only when a benchmark proves we can't match it" clause
   of §9.1: our hand-written fp4 GEMM ran ~15% of peak, cutlass ~300 TFLOPS on the
   27B prefill projections (measured GB10). CUTLASS v4.4.2 is header-only (torch-free),
   provided via `-DVLLM_CPP_CUTLASS_DIR` (default `third_party/cutlass`), mirroring
   vLLM's own FetchContent(v4.4.2). The TU is isolated (own `-isystem` cutlass tree
   + `--expt-relaxed-constexpr`, sm_12xa only, ~37s compile) and gated by
   `VT_CUTLASS_NVFP4`; the op is opt-in (`VT_NVFP4_CUTLASS`) so no existing path
   changes. Only the 27B (compressed-tensors W4A4) uses it; the 35B is FP8 + Marlin
   W4A16 (never the sm120a fp4xfp4 kernel — grounded in `modelopt.py` MIXED_PRECISION
   per-layer resolve).

10. **Vendored Marlin (NVFP4 W4A16 grouped-MoE GEMM — the 35B experts)**:
    `src/vt/cuda/marlin/` is a torch-free 1:1 vendor of vLLM's
    `moe/marlin_moe_wna16` @ e24d1b24 (`marlin_template.h`, `marlin.cuh`,
    `marlin_dtypes.cuh`, `dequant.h`, `marlin_mma.h`, `kernel.h`,
    `core/scalar_type.hpp`, + the generated NVFP4-bf16 instantiation). The ONLY
    change is the ONE torch coupling: `STD_TORCH_CHECK` (from
    `<torch/headeronly/util/Exception.h>`) → `vt_marlin_check.h` (a throwing
    macro). `marlin_mm_moe.cu` = vLLM `ops.cu:1-542` (the `marlin_mm` dispatcher);
    the torch host launcher (`moe_wna16_marlin_gemm:543`) is replaced by the
    torch-free `vt::MoeGroupedGemmNvfp4Marlin` (`src/vt/cuda/cuda_moe_marlin.cu`)
    mirroring the NVFP4 branch (`b_type=kFE2M1f`, `s_type=kFE4M3fn`,
    `group_blocks=1`, bf16 act/out). Gated `VT_MARLIN_NVFP4` (CMake
    `VLLM_CPP_MARLIN`, sm_12xa). The two vendored TUs build with
    `-static-global-template-stub=false` (nvcc 13 makes `__global__` template
    instantiations static by default → undefined cross-TU refs at link; verified
    on GB10), `-isystem` (warning isolation), `--expt-relaxed-constexpr`.
    **Verified GB10 sm_121a (2026-07-05): the vendored `marlin_mm` is BIT-EXACT
    vs vLLM's own `moe_wna16_marlin_gemm`** (rel_err=0, `tools/marlin/`). This is
    the "vendor a proven kernel when we can't match it" clause of §9.1: the
    hand-written wmma MoE GEMM ran ~15% of peak; Marlin is vLLM's tuned M-small
    W4A16 kernel. STILL PENDING (branch `feature/marlin-nvfp4`, not merged): the
    C++ load-time repack (mirror `prepare_nvfp4_moe_layer_for_marlin` —
    `gptq_marlin_moe_repack` + `marlin_permute_scales` +
    `nvfp4_marlin_process_scales`/`_global_scale`), the `moe_align_block_size`
    port, the 35B forward wiring, 16/16 parity, and the A/B TFLOPS measurement.

11. **Vendored FlashAttention-2 (the full-attn PREFILL kernel — both gate models'
    head_dim-256 GQA attention)**: `src/vt/cuda/flash_attn/` is a byte-identical,
    torch-free vendor of vllm-project/flash-attention @ 2c839c33 (the exact
    source vLLM 0.24.0 builds as `_vllm_fa2_C`; its
    `flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256,64,64,4,...>>` is what
    vLLM's own profile shows for prefill on GB10). 3 stub headers replace the
    ATen/c10 surface (PhiloxCudaState POD + C10_CUDA_CHECK) and
    `fa2_compat_prelude.h` is force-included; the FA sources stay pristine.
    `flash_api.cpp` (torch-heavy) is replaced by the torch-free launcher
    `src/vt/cuda/cuda_flash_attn_fa2.cu` (fills `Flash_fwd_params` from
    `vt::Tensor` views; paged block_table + varlen cu_seqlens_q/seqused_k;
    num_splits pinned 1 = vLLM's FA-2 varlen behavior,
    flash_attn_interface.py:309-310). Engaged by the LaunchPaged dispatch for
    the natively-bf16 prefill combo only (bf16 q from the fused preamble + bf16
    KV + bf16 out, head_dim 256); decode + every other combo stay on the
    hand-written WMMA/decode kernels, and the CPU reference is untouched. CMake
    `VLLM_CPP_FLASH_ATTN` (default ON with CUTLASS, sm_12xa), runtime
    `VT_FA2_PREFILL` (default ON when compiled; =0 → WMMA for same-binary A/B).
    **Verified GB10 sm_121a (2026-07-10): kernel 3.68× vs our WMMA (475.3→129.2
    ms per profile window), with same-binary 27B e2e +1.52%/+0.54%
    (conc16/conc32); token-exact greedy gates PASS ON and OFF. The historical
    claim that this reached ≥1.0× vs vLLM is non-binding after the 2026-07-11
    sampling/token-budget audit; the component A/B remains valid. The earlier
    −4.3% attempt was the f32↔bf16 cast glue + a per-layer D2H sync, both removed
    (see parity-ledger 2026-07-10).**

12. **Additive drop-in adapter ABI W0 (`BACKEND-ABI-VT`, GATING):** the common
    `vt::` surface now carries upstream-compatible semantic scalar IDs separate
    from storage dtype, layout/shape/element-stride descriptors, typed op
    registration, monotonic queue identity, named workspace roles and explicit
    device-resource free functions. CUDA's new `cuda_dropin.{h,cu}` ports the
    device/current-stream boundary from
    `csrc/libtorch_stable/torch_utils.h:20-82` and binds a test-only Layer A to
    a raw pointer/geometry/scalar/workspace/stream Layer B; its workspace pool
    mirrors `torch::stable::new_empty` caching and FlashInfer's caller-owned
    first-use-zero contract. Ported tests map
    `tests/cuda/test_cuda_context.py:54-83` and
    `tests/v1/cudagraph/test_cudagraph_dispatch.py:271-354` to
    `tests/vt/test_dropin_abi.cpp`. No production family uses this helper yet.
    Narrow-scope deviations/debt are explicit: `W0-SCALAR-FORWARDER` keeps the
    exact `ScalarType::id()` vocabulary in `ops.h` until the first family claim
    may safely replace the vendored Marlin header with a common forwarder;
    `W0-BACKEND-SHIM` leaves old `Backend` virtual calls as index-0 shims while
    new adapters use the device-explicit free functions. CPU 94/94 is green;
    CUDA cross-build/runtime/capture/model/trace evidence remains `W0-GPU`.

9.N **Triton CUDA fast-path for PROVEN codegen-bound kernels (User-sanctioned
    2026-07-09; extends the "vendor a proven kernel when we can't match it" clause).**
    For a kernel where portable C++ is *measured-exhausted* against vLLM's compiler
    codegen, a CUDA-only Triton kernel may be AOT-compiled to cubin at BUILD time and
    dispatched behind `vt::`. Scope, strictly bounded: gated `VLLM_CPP_TRITON` (default
    OFF); the CPU reference + a correct portable hand-C++ CUDA fallback are PRESERVED
    (Metal/Vulkan/ROCm/XPU still port from `vt::`+CPU-ref; CUDA-without-Triton still
    works); the RUNTIME stays Python/Triton-free (cubin via the CUDA driver API);
    token-exact + greedy-16/16-gated. **Trigger case: the GDN linear-attention chunk
    kernels** (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` + `chunk_o` +
    `recompute_w_u`), ~1.9× slower than vLLM's Triton/FLA where the portable path was
    proven exhausted — register-tiling (delta_h −22%), blocked tensor-core inverse,
    bf16 I/O, and BOTH async-pipeline tiers (Rung-1 cp.async + Rung-2 TMA+mbarrier,
    `include/vt/cuda/tile/`) all landed, and the delta_h kernel still sits ~1.9× off
    vLLM on a compute floor → the residual is Triton's WMMA compute codegen, a compiler
    capability. See discipline.md "SANCTIONED EXCEPTION" + mission.md. Toolchain proven
    on branch `perf/triton-fastpath` (cmake/TritonAOT.cmake, `triton.tools.compile/link`,
    token-exact rmsnorm). See discipline.md + mission.md; rationale + evidence in
    `.agents/state.md` (2026-07-09 "portable async-pipeline EXHAUSTED").

    **IMPLEMENTED — delta_h (2026-07-09; branch `perf/gdn-deltah-triton-aot`).**
    The delta_h state recurrence is the first kernel landed on this sanction:
    `triton_kernels/chunk_delta_h.py` (FLA `chunk_delta_h.py:42-315` VERBATIM —
    autotune/heuristics stripped so the constexprs are pinned per-shape via the
    AOT signature; one documented grid-carrier scalar `NH`=N*H since the FLA grid
    needs the sequence count, not a kernel arg). Toolchain `cmake/TritonAOT.cmake`
    (merged from `perf/triton-fastpath`) builds two specializations — H=48 (27B),
    H=32 (35B); both K=V=128, Hg=16, BT=64, BV=64, warps4/stages3 — to embedded
    cubins; dispatch `TryTritonDeltaH` in `cuda_gdn.cu` behind the runtime toggle
    `VT_GDN_DELTAH_TRITON` (opt-in until the win is proven), with the hand
    `GdnChunkDeltaHRegRingKernel` + CPU ref PRESERVED as fallback and the OFF build
    byte-inert. The GDN buffer layout is a verified 1:1 drop-in (strides checked
    stride-for-stride against the FLA pointer arithmetic), so the Triton kernel
    consumes the same device buffers the other chunk kernels produce.
    GATES (GB10 sm_121a): token-exact — `test_ops_gdn` 450/450 (incl. a new
    gate-shape Triton-vs-sequential case at H=48 and H=32) + 27B greedy 16/16 AND
    35B greedy 16/16 single & batched-graph, all through the Triton path.
    MEASURED (GB10, idle-flocked box, 27B NVFP4). delta_h KERNEL (same-binary nsys
    A/B, in1024/out4, 240 launches, control kernels ±1% ⇒ clean isolation): hand
    `GdnChunkDeltaHRegRingKernel` 1685.4 µs → Triton
    `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 1180.7 µs/launch = **−29.9%
    (1.43×)** — the Triton WMMA codegen substantially beats our best hand kernel,
    VALIDATING the sanction's premise (the residual WAS compiler codegen). But
    delta_h is only ~19% of GDN chunk-kernel GPU time (the profile shows WU ~25% and
    ChunkO ~20% are LARGER), so GDN aggregate −5.7% ⇒ GDN µs/tok gap vs vLLM's FLA
    (128 vs 71.7 = ~1.79×) closes to ~1.68×. E2E (same-binary A/B, 3 reps,
    non-overlapping arms) vs FRESH graphed vLLM: conc16/np96 in1024/out128 hand
    712.96 → Triton 716.10 tok/s (+0.44%), vLLM 766.62 ⇒ **0.930×→0.934×**;
    conc32/np192 hand 859.21 → Triton 866.59 (+0.86%), vLLM 1043.17 ⇒
    **0.824×→0.831×**. HONEST VERDICT: delta_h Triton is a real, clean, token-exact
    per-kernel win that PROVES the Triton fast-path closes the codegen gap on the
    kernel it targets, but delta_h alone does NOT reach 27B ≥1.0× MVP parity (e2e
    +0.4–0.9%). Reaching parity needs the SAME AOT treatment on the two LARGER GDN
    kernels — `chunk_o` (~20%) and `recompute_w_u` (~25%) — the next rollout on this
    sanction. Kept default-OFF pending that rollout + the flip-to-default decision.

    **IMPLEMENTED — chunk_o + WU/WY pipeline (2026-07-09; branch
    `perf/gdn-wu-chunko-triton-aot`, stacked on `perf/gdn-deltah-triton-aot`).**
    The two LARGER GDN chunk kernels join delta_h on the sanction:
    `triton_kernels/chunk_o.py` (FLA `chunk_o.py` `chunk_fwd_kernel_o` VERBATIM) and
    the 3 FLA WY kernels our single fused `GdnChunkWUWmmaVecKernel` mirrors —
    `triton_kernels/{chunk_scaled_dot_kkt,solve_tril,wy_fast}.py`
    (`chunk_scaled_dot_kkt_fwd_kernel` → `merge_16x16_to_64x64_inverse_kernel` →
    `recompute_w_u_fwd_kernel`). AOT adaptations: autotune/heuristics stripped +
    constexprs pinned per-shape; one dead grid-carrier `NT` (B*H==H under our varlen
    B=1 packing, baked); solve_tril pins USE_TMA=0 (`is_tma_supported` False on GB10)
    + DOT_PRECISION "ieee". Toolchain builds 8 new specs (4 kernels × H=48/32); dispatch
    `TryTritonChunkO` (`VT_GDN_CHUNKO_TRITON`) + `TryTritonWU` (`VT_GDN_WU_TRITON`, runs
    the 3 WY kernels into scratch A f32 / Ai bf16), device-built FLA `chunk_indices`
    (`GdnBuildChunkIndices`); hand kernels + CPU ref PRESERVED, OFF build byte-inert.
    Two bugs found+fixed (both would-be silent): (1) chunk_o `scale` — Triton AOT
    mis-packs an fp32 scalar as C `double` (kernel reads 4 of 8 bytes → garbage), so
    scale is PINNED to Dk^-0.5 in-kernel and the dispatch guards `args.scale`; (2) the
    solve_tril `Ai` output — FLA does `zeros_like`, solve_tril writes only the 10
    lower 16×16 blocks and `recompute_w_u` dots the FULL block, so `Ai` MUST be zeroed
    (a dirty cudaMallocAsync pool in the busy engine put NaN in the upper triangle →
    35B batched-graph all-zero tokens; the op test + compute-sanitizer both hand back
    a CLEAN pool, which is why it slipped there and surfaced only in the 6-request
    batched gate).
    GATES (GB10 sm_121a, full Triton WU+delta_h+chunk_o): `test_ops_gdn` 31/31 (new
    gate-shape cases incl. H=32 multi-seq + tiny-seq T<16 WY) + compute-sanitizer 0
    errors; 27B greedy gate PASS; 35B greedy single + batched-graph 2/2 (33/33), all
    through the Triton path.
    MEASURED (GB10, idle-flocked, 27B NVFP4). Per-kernel (nsys same-binary A/B,
    in1024/out4 np16, 240 launches/kernel): WU (fused hand `GdnChunkWUWmmaVecKernel`)
    2204.8 → Triton kkt 286.5 + solve_tril 415.7 + recompute_w_u 722.7 = 1424.9
    µs/launch = **−35.4%**; chunk_o hand 1729.5 → Triton 1097.7 = **−36.5%**; delta_h
    hand 1673.9 → Triton 1174.8 = **−29.8%**. **GDN chunk aggregate 5608.2 → 3697.4
    µs/launch = −34.1% (1.52×)** — the full Triton port (delta_h+chunk_o+WU) cuts a
    THIRD of GDN chunk-kernel GPU time, taking the ~1.79× GDN-vs-FLA gap down to
    ~1.18× (the residual is the autotuner's per-shape BK/BV/warps/stages vs our pinned
    BV=64/w4/s3 — a tuning knob, not structure). E2E (same-binary A/B, 3 reps,
    non-overlapping arms) vs FRESH graphed vLLM (766.62 conc16 / 1043.17 conc32,
    same box/config, measured flock-clean by the PR #1 agent minutes earlier — a
    model+config constant, not code-dependent): conc16/np96 in1024/out128 hand 713.38
    → Triton 723.82 tok/s (+1.46%), vLLM 766.62 ⇒ **0.930×→0.944×**; conc32/np192
    hand 856.73 → Triton 876.08 (+2.26%), vLLM 1043.17 ⇒ **0.821×→0.840×**.
    **VENDORED — build-time Python removed (2026-07-10; branch
    `build/vendor-triton-aot`).** The generated AOT artifacts (per-spec C
    launchers with EMBEDDED cubins + the linked stable dispatchers, 12 bases =
    5 kernels × H=48/32 plus the two bf16-output `chunk_o` dispatchers) are now
    COMMITTED per-arch under `src/vt/cuda/triton_aot_vendored/<arch>/` (today
    `sm_121a/`; 48 generated C/H files + MANIFEST — the embedded-cubin C arrays dominate; the
    Marlin/FA-2 vendoring precedent), each with a provenance header and a
    `MANIFEST` (triton/ptxas/python/CUDA versions, sha256 of every
    `triton_kernels/*.py`, per-base generation parameters, date). This NARROWS
    the sanction: `-DVLLM_CPP_TRITON=ON` now consumes the vendored artifacts
    with ONLY a C compiler — **no Python/Triton at build time for normal
    builders**; Python+Triton is a MAINTAINER-only regen dependency
    (`-DVLLM_CPP_TRITON_REGEN=ON` reruns `triton.tools.compile/link` AND
    refreshes the vendored tree + MANIFEST; `scripts/regen-triton-aot.sh` wraps
    it and prints the git diff to review/commit). Staleness is guarded at
    configure time and by a no-Python/GPU CI checker: target, line-info policy,
    generator shim, kernel-source hashes, exact base/signature set, artifact
    inventory, and every artifact hash must match the MANIFEST; drift is FATAL.
    **Cross-worktree byte reproducibility is now resolved (2026-07-10,
    `CLAIM-PR3`).** The initial measurement found that all cross-path deltas were
    embedded Triton line-info source paths in the 12 specialization C arrays.
    Generation now derives the explicit `cuda:121:32` target from `sm_121a`,
    sets `TRITON_DISABLE_LINE_INFO=1`, and pins the repository-owned Triton 3.6
    numeric-target shim by hash. Recovered pre-current-main validation records
    two regenerations from different absolute source roots as byte-identical;
    the current integrated tree's pure drift checker and ten mutation cases
    pass over all 49 files. Current W2 staging now makes BF16 core/z output the
    dense-27B default only (all 35B paths retain f32) because it restores the
    native 16/16 stream. Its exact c16 BF16/f32 AB/BA/AB is
    **789.279/783.023 tok/s = 1.007989×**, 16/20 timing and 2/4 memory axes,
    six returns, summary `ee6d25c…c930b`. This is a positive but strict-gate-red
    `KERNEL-GDN-AOT-BF16` checkpoint; paired trace/pool/fresh-vLLM closure remains
    and is not an A4/reproducibility claim. The OFF build stays byte-inert.
    VALIDATED end-to-end on dgx.casa (2026-07-10, fresh clone, configure pinned
    to `VLLM_CPP_TRITON_PYTHON=/nonexistent/python` so only the vendored
    artifacts could satisfy the build): full `all` builds green with no Python,
    `test_ops_gdn` 31/31 cases (557 assertions), and the 27B W4A4 paged-engine
    greedy gate passes on the vendored cubins.
    HONEST VERDICT: the full GDN Triton port (delta_h+chunk_o+WU) is a real, clean,
    token-exact per-kernel win — −34.1% GDN chunk-kernel GPU time, ~3× the delta_h-
    alone e2e win — but 27B does **NOT** reach ≥1.0× MVP parity (0.944× conc16 /
    0.840× conc32). The residual has MOVED OFF GDN: the GDN chunk kernels are now
    Triton-fast (~1.18× vLLM's autotuned FLA, a BK/BV/warps/stages tuning knob, not
    structure), so the dominant remaining gap (~5.6% conc16 / ~16% conc32) is the
    NON-GDN prefill buckets — vLLM's Inductor whole-graph fusion (rmsnorm+quant,
    silu+quant, run unfused by us; see fusion-architecture + honest-bar notes). GDN
    Triton was necessary but not sufficient for MVP. Kept default-OFF pending the
    flip-to-default decision (+ the per-shape AOT-config tune to squeeze the GDN
    1.18× residual).

## 10. E2E test suites (T0 deliverable)

1. **Op parity**: golden dumps from upstream vLLM (Python, test-time only) →
   C++ kernel tests (CPU ref + CUDA), thresholds per dtype. Ported from the
   relevant `tests/kernels/` cases.
2. **Engine behavioral**: scheduler/BlockPool/prefix-cache/detokenizer suites
   ported from upstream `tests/v1/core/` semantics — chunk splits, preemption,
   hash reuse, eviction, UTF-8 boundaries; hybrid-group allocation for GDN.
3. **Model parity**: per-layer activation + end-to-end logits vs upstream for
   Qwen3.6-27B/35B-A3B (and Qwen3-0.6B for fast CI); greedy decode
   token-for-token match; NVFP4 and GGUF paths each validated.
4. **Server e2e**: OpenAI-endpoint conformance (streaming chunks, stop handling,
   usage accounting, error shapes), health/metrics; runs in CI with the 0.6B
   model on CPU ref backend, nightly on dgx.casa with gate models.
5. **The gate benchmark**: the unmodified pip-vLLM 0.25.0 `bench serve` oracle
   (contract-audited against target `702f481`; porting pin still `e24d1b24`) is
   validated/active on DGX and wrapped by the committed
   `tools/bench/online_gate*.py` and `scripts/dgx-online-serving.sh` harness
   (ported command/schema contracts in
   `tests/tools/test_online_gate_*.py`). It freezes exact 1024-token partitions,
   runs c1/2/4/8/16/32 with three interleaved ours/vLLM repetitions under one
   model-wide lock and an identical 32-sequence/per-model-token-budget
   scheduler configuration, retains detailed TTFT/TPOT/ITL/E2EL arrays, compares
   output exact-match counts diagnostically behind a separate commit-bound model
   correctness gate, samples process-tree/GPU memory and thermal/power,
   verifies memory return, refreshes targets from a clean exact HEAD, hashes the
   executable pip-oracle runtime, and records an ours-nsys/vLLM-torch-profiler
   trace pair. Both arms now explicitly disable prefix caching for the hybrid
   gate comparison; the profiler uses closed-loop c16 admission with production
   max-seqs/model length, and the server exposes the pinned model-default plus
   explicit cache override. The oracle manifest hashes the venv's `ninja` executable and the
   profiler prepends that venv to `PATH`, because FlashInfer can JIT inside a
   spawned EngineCore. Pinned-vLLM producer-ahead DELTA merges remain valid
   inter-chunk timing evidence when native output counts are exact. Profiler
   warmup/measured output digests and their equality flag are retained without
   treating FP4 greedy near-tie branches as a performance failure; any missing, failed,
   native-count-inexact, over-fragmented, hash-drifted, or below-vLLM artifact
   is non-binding. The historical direct-library denominators are also
   non-binding: vLLM used temperature 1 while ours used temperature 0, and the
   27B token budgets differed. CPU harness contracts are green; corrected
   direct-library and online DGX campaigns remain open under
   `BACKEND-GATE-CUDA-VLLM` / `SERVE-GATE-ONLINE`.
