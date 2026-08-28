# vLLM → vllm.cpp Porting Inventory (living parity record)

**Upstream reference:** `${VLLM_SOURCE}` @ sync point in
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
| InputProcessor (validate, tokenize, build EngineCoreRequest) | `v1/engine/input_processor.py` | T0 ✅ `73a9509` (text path; runs PostInit/Verify + max_tokens default + eos/stop wiring; mm/lora/embeds/pooling deferred) + `_validate_prompt_len` decoder arm (`input_processor.py:387-432`) landed: empty prompt and prompt ≥ `max_model_len` raise `InputValidationError` → HTTP 400 (`serve/utils/error_response.py:62-65`); tests `tests/vllm/v1/test_input_processor.cpp`, `tests/vllm/entrypoints/openai/test_api_server.cpp`. Encoder arm + out-of-vocab check still deferred |
| Startup KV sizing: `_check_enough_kv_cache_memory` / `estimate_max_model_len` / `max_memory_usage_bytes` / `_auto_fit_max_model_len` | `v1/core/kv_cache_utils.py:751-788, :791-798, :800-851, :1967-2027` | T0 ✅ `vllm::v1::check_enough_kv_cache_memory` etc. in `v1/core/kv_cache_utils.{h,cpp}`, applied at `LoadedEngine::ResolveMaxModelLen`: a pinned `--max-model-len` the KV pool cannot hold is REFUSED at startup, an unpinned one is auto-fitted down to the pool. Deviations: the two upstream `Callable`s are passed as values (both are pure arithmetic here) and `estimate_max_model_len`'s binary search is written closed-form (our per-block geometry is linear in the block count). Tests `tests/vllm/v1/test_kv_cache_utils.cpp`, `tests/vllm/entrypoints/test_loaded_engine_dense.cpp`. Issue #83 M4 |
| OutputProcessor + RequestState + incremental Detokenizer | `v1/engine/output_processor.py`, `detokenizer.py` | T0 ✅ `c7ba3a5` baseline + W2 `GATING`: process_outputs detokenize/string-stop/DELTA-CUMULATIVE-FINAL_ONLY plus thread-safe single-slot `RequestOutputCollector`, per-request queue handoff, abort-final output and error propagation; logprobs/pooling/parallel-sampling remain deferred |
| AsyncLLM-equivalent streaming API + sync LLM API | `v1/engine/async_llm.py`, `llm_engine.py` | T0 🚧 W2 `ACTIVE`: `AsyncLLM` owns the EngineCoreProc/output-handler threads, concurrent add/generate/abort, collector streams and clean shutdown. The c32 cpp-httplib defect now has fixed `max_num_seqs + 4` delivery workers plus a legacy A/B toggle; the 32-client control-reserve test is 100×/sanitizer-green. Exact GPU G1/G3-G6 remain. Synchronous `LLMEngine` stays for offline/compatibility use |
| Unified scheduler: token-budget, **no prefill/decode distinction** | `v1/core/sched/scheduler.py` | T0 ✅ `4f12158` (schedule() running-first + chunked prefill + FCFS preemption; update_from_output + check_stop; priority/spec/structured/async deferred behind 1:1 stubs) |
| Chunked prefill (`enable_chunked_prefill`, on by default) | `config/scheduler.py` | T0 ✅ `4f12158` |
| Budgets: `max_num_batched_tokens`, `max_num_seqs`, `max_num_scheduled_tokens` | `config/scheduler.py` | T0 ✅ `2f0ea69` |
| Preemption (FCFS tail pop, recompute) + `SchedulerOutput` new/cached diff protocol | `v1/core/sched/{scheduler,output}.py` | T0 ✅ `4f12158` (`c65e650` SchedulerOutput/NewRequestData/CachedRequestData in the MRV2 shape — prefill_token_ids + resumed-as-new fold) |
| FCFS request queue | `v1/core/sched/request_queue.py` | T0 ✅ `2f0ea69` |
| Priority scheduling (`policy="priority"`) | same | T1 🚧 GATING (W4 `ENG-PRIORITY-SCHED`): `PriorityRequestQueue` (heap by `Request.__lt__`) + priority preemption (victim = max `(priority, arrival_time)`) + `priority` plumbed through `Request`/`EngineCoreRequest`/OpenAI field + `SchedulerPolicyFromString`; default stays FCFS. Tests ported: `tests/vllm/v1/test_scheduler.cpp` (12 `test_priority_scheduling_*` cases ← `tests/v1/core/test_scheduler.py:2382,2978`) + `tests/vllm/v1/test_request_queue.cpp` (priority-queue cases + seeded random property ← `tests/v1/core/test_priority_scheduler_random.py`). CPU 93/93 green; GPU G1 token-exact A/B deferred (GPU held) |
| Partial-prefill concurrency (`max_num_partial_prefills`, long-prefill threshold/limits) | `config/scheduler.py` | T1 |
| Async scheduling (overlap schedule with execution) | `v1/core/sched/async_scheduler.py` | T1 `READY` (W3; vLLM default-ON mirror obligation, distinct from W2's asynchronous frontend). Accepted `3812d8` c2 control completes six timings/two traces at **1.002153×** ON/OFF total and **1.002004×** traced GPU time; neutral/no speed credit, W3 stays unclaimed later parity work |
| Opt-in batch-invariant execution (`VLLM_BATCH_INVARIANT=1`): deterministic matmul/norm/attention/collectives and persistent-scheduler NVFP4 across batch shapes | `envs.py`; `model_executor/layers/batch_invariant.py`; `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`; `tests/v1/determinism/` | T1 (v0.25.0 inventory; production default is off. W3-C3R proves default-mode batch-shape dependence in ours and vLLM; local opt-in mode absent) |
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
| KV tensor allocation from `KVCacheConfig` | same | T0 ✅ runner consumes each ordered `MambaSpec` as conv then temporal and allocates the dtypes independently; gate ABI is BF16 conv + FP32 SSM from nested `mamba_ssm_dtype=float32`, with exact page-byte tests and F16/BF16/F32 SSM aliases |
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
| Paged attention for full-attn layers on sm_121 (bf16, GQA 16/2, partial RoPE) — FlashInfer-class performance is the bar; strategy in §9 | ref: `v1/attention/backends/{flashinfer,triton_attn,flash_attn}.py` | T0 🚧 on GB10 gate workloads: correctness passes; immutable `3f256ab` binds at **55/124** with c1→c32 total ratios 0.993504/0.954464/0.966438/0.980678/1.027889/1.039417× and host PSS/RSS red. W3-E/W3-F/W3-G earn no speed credit. W3-H `c498a413` passes status `84d15970…6e66`; FA2 main is only the third positive mapped residual behind fused and normal FP4 production. No attention speed credit or 35B performance follows. Broader coverage remains in `kernel-matrix.md` |
| **GDN backend**: metadata segmentation (prefill/decode/spec) | `v1/attention/backends/gdn_attn.py` | T0 ✅ `370ddaf` (GDNAttentionMetadata decode/prefill split + has_initial_state mask + prefill rebasing; spec segments + align col-gather deferred; GDN-state zeroing = caller obligation, see state.md) |
| GDN chunked-scan prefill kernel (chunk gated delta rule) | `layers/fla/ops/chunk.py` (Triton ref), `flashinfer.gdn_prefill` (Blackwell) | T0 🚧 `ead59d6` (correctness-grade sequential; chunked perf kernel M2.3) |
| GDN fused sigmoid-gating decode recurrence (mixed/spec and packed-disabled fallback) | `layers/fla/ops/fused_sigmoid_gating.py` | T0 ✅ `ead59d6` correctness-grade decomposed recurrence; exact mixed/spec breadth remains under its engine rows |
| GDN packed pure non-spec decode (default-on FP16/BF16/F32 path) | `layers/fla/ops/fused_recurrent.py:255-478`; Qwen dispatch `qwen_gdn_linear_attn.py:1286-1298,1644-1695` | T0 🚧 `KERNEL-GDN-PACKED-DECODE` `ACTIVE`: clean `f344dec` closes W1D2/G2 for exact dispatch, rollback and safety; `7ff713e` + `24cea4f` close exact structure. Clean `d82d282` passed model gates/all c2 legs, then failed incomplete at c16 packed r1 with 96/96 HTTP 500 responses and no marker. Partial legs earn no speed credit |
| GDN post-conv prep (q,k,v,g,beta + L2 norm) + causal conv1d fn/update | `layers/fla/ops/fused_gdn_prefill_post_conv.py`, `layers/mamba/ops/causal_conv1d.py` | T0 ✅ prefill/mixed path; pure decode bypasses materialized q/k/g/beta only after the packed row gates |
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
`Qwen3_5MoeForConditionalGeneration` — plus, since 2026-08-12, their TEXT-ONLY
arms `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`, which are two further
registrations over those SAME two factories rather than new implementations (§9
deviation 17: ahead-of-pin, run gate OWED). Live loading consumes the full
`config.architectures` list in order and mirrors pinned unknown,
previously-supported, and out-of-tree rejection; unimplemented IDs remain
unsupported rather than being misclassified by `num_experts`. The execution row
is `GATING` on its deferred two-model GPU no-regression campaign; see
[model-factory-registry.md](specs/model-factory-registry.md) and
`model-matrix.md`. Dynamic Transformers/terratorch/OOT runtime loading and
Python lazy-import/subprocess caching remain explicitly deferred.

| Family | Marquee members | Needs | Tier |
|---|---|---|---|
| **Qwen3.5/3.6 hybrid (incl. MoE)** | `Qwen3_5ForConditionalGeneration` (27B dense-hybrid, **VL multimodal** wrapper — vision_config present), `Qwen3_5MoeForConditionalGeneration` / `qwen35moe` (35B-A3B), plus the TEXT-ONLY arms `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM` (`qwen3_5_text` / `qwen3_5_moe_text`; motivating checkpoint `Qwen/Qwen3.8-2.4T-A95B`) | GDN layers ×3 : 1 gated full-attn (qk-norm, partial RoPE 64d, output gate), MoE 256e top-8 + shared expert (35B) / **dense SwiGLU MLP** (27B), GemmaRMSNorm-style `(1+w)` — ✅ `25326fc` (35B forward correctness-grade, **safetensors**; 16/16 greedy on GB10 = M0 exit; GGUF k-quant load M0.10). **27B (co-equal gate):** CPU-first scaffolding started — arch/quant surveyed (`.agents/specs/qwen27b-w4a4-notes.md`: dense hybrid, W4A4, **and a VL wrapper — text path first, ViT deferred**), CPU W4A4 emulation reference + skipping greedy-parity gate landed; dense loader + single-seq `ForwardDense` + batched PAGED `Qwen3_5DenseModel::Forward` (paged==dense CPU-anchored) + `GPUModelRunner` dense route all landed CPU-green; W4A4 GPU GEMM + oracle golden pending (GPU-gated). **Ordinary plain-BF16 leaf (`LOAD-SAFETENSORS-DIRECT-DENSE`):** raw-NK stacked/tied 4B loading plus layer-bounded discrete-CUDA staging is implemented; real-weight CPU load passes 1656/1656 and local AOT-CUDA retained-host/direct-device tokens match at 1664/1664. The corrected AOT series completes 18/18 legs: ON/OFF/vLLM total **6155.10/6064.06/6730.46 tok/s**, ON=OFF 128/128, peak PSS **2.405/8.571/7.569 GiB**; current ON is 0.9316x the previous AOT result. Current-v0.25 oracle/sanitizer, strict VRAM and external 27B/35B regressions remain open, so this is not a support claim. **TEXT-ONLY arms (`MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm`, `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm`, issue #490, both `PARTIAL`):** additive registration against the existing dense/MoE factories plus a once-per-checkpoint backbone-namespace resolution (`model.language_model.` or flat `model.`, mixed REFUSED); AHEAD OF THE PIN at `ad5d29db7` / vllm#50210 and the **run gate is OWED** — no Qwen3.8 checkpoint fits GB10, so nothing about generated tokens, memory or speed is claimed. The bf16 / 3-D-stacked MoE routed-expert arm was NOT implemented at that date and such a checkpoint was refused by name; **it is now implemented and ORACLE-GATED (2026-08-15, #740 + #864, §9 deviation 17(e)): 6/7 prompts STRICT 16/16 on the published `Qwen/Qwen3.6-35B-A3B` bf16 repo @`995ad96e`, the seventh an exact logit tie (#910), SACRED 3/3 byte-identical, NO speed number.** The run gate for `Qwen3_5MoeForCausalLM` itself is still OWED on size alone ([spec](specs/qwen38-text-only.md)). serving M1–M3 | **T0 (the gate)** |
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
| **NVFP4 gate slices** — ModelOpt W4A16 experts (35B) + compressed-tensors W4A4 dense (27B) | `quantization/modelopt.py`, `compressed_tensors/` | **T0 ✅ correctness/support; performance ACTIVE** — `3f256ab` binds at **55/124**. FP4 tactics and BA topology/projection are matched. Clean `f344dec` closes default+rollback correctness; packed structure is accepted. Clean `d82d282` failed incomplete at c16 packed r1 after all c2 legs; no terminal result exists. qkvz, exact grid and 35B performance remain unauthorized; `benchmark_binding=false` |
| **GGUF materialization** — F32/Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ2_XXS/NVFP4 | **vllm.cpp deviation**: pinned vLLM has no GGUF load format (0.26 migrated GGUF to the OOT `vllm-gguf-plugin`); llama.cpp is the container/quant reference. Q2_K (id 10) + IQ2_XXS (id 16) ported 1:1 from llama.cpp `ggml-quants.c` @ `237ad9b96` (`dequantize_row_q2_K:903`, `dequantize_row_iq2_xxs:2416` + `iq2xxs_grid`/`ksigns_iq2xs`/`kmask_iq2xs` `ggml-common.h:550,503,499`) — from-necessity port, 2026-07-29 `CLAIM-DSV4-GGUF-LOADER`, the single-Spark DeepSeek-V4-Flash GGUF vehicles | **T0 🟡** loader + synthetic per-layout tests + real APEX Q3/Q4/Q5/Q6/Q8 greedy parity pass. The llama.cpp-derived CPU threadpool/chunked-op prerequisite is implemented and correctness-gated (1/3/20 full suites + TSAN), but its B4 speed/RSS gate is pending. Most weights still expand to bf16 (K-quants keep-quant on CPU); Q2_K/IQ2_XXS are dequant-only (no vec_dot yet). F16 is reader-only, not executable; BF16/IQ (other)/TQ/Q1/MXFP4 execution remains open. |
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
`skip_special_tokens`, `output_kind` (CUMULATIVE/DELTA/FINAL_ONLY),
`stream_interval` (per-request streaming-cadence override, upstream-sync
**vllm#49754** 2026-07-30 — only raises the engine `--stream-interval`, clamped
up in `output_processor` `RequestState::FromNewRequest`; Verify() rejects `<1`);
pipeline order
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
bit-exact parity deferred to T1**. Deferred (marked stubs): spec-decode
bonus-token, thinking-budget. **logprob_token_ids (generative scoring) LANDED
2026-08-10** and is no longer a stub — `SAMPLE-LOGPROB-TOKEN-IDS` in
engine-matrix.md, [specs/logprob-token-ids.md](specs/logprob-token-ids.md).
(The `logprobs_mode` variants left the same stub list when #238 landed; the
sentence is corrected here because this edit rewrites it.) **InputBatch-side tracking of seeds/min_p/min_tokens/
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
`choice`, `grammar`(EBNF/GBNF) + OpenAI `response_format`. **The xgrammar backend
W1 landed 2026-07-29 (`CLAIM-TOOLS-XGRAMMAR`, `TOOLS-XGRAMMAR`→ACTIVE)** as a 2nd
registerable backend behind the SAME seam. **§9 DECISION (recorded): mirror
xgrammar's algorithm PORTABLY, do NOT vendor the mlc-ai/xgrammar C++ library** —
xgrammar IS "grammar → pushdown automaton → per-step token bitmask", already
implemented portably by the native engine; vendoring would duplicate that
machinery + add a heavy dependency against the no-extra-deps posture. So
`XgrammarStructuredOutputBackend` (`backend_xgrammar.cpp`) REUSES the native
matcher and ports only the xgrammar-FAITHFUL front-end where the two diverge — the
JSON-schema→EBNF converter (`xgrammar_json_schema.cpp`, SEMANTICS ported from
xgrammar `cpp/json_schema_converter.cc` @ `a32ac89`): property DECLARATION order
(`nlohmann::ordered_json`, vs the native path's lexicographic sort), the
`any_whitespace` `ws` rule, and the `basic_*` set VERBATIM — closing the key-order
+ whitespace + exotic-schema parity that was xgrammar-only. `auto`→xgrammar
selection mirrored (`sampling_params.py:1031`). Deferred (W2+): STRUCTURAL_TAG-full,
reasoning-gating, spec-decode multi-row, optional object properties, strict-compact
separators, the `has_xgrammar_unsupported_json_features` guard +
`validate_xgrammar_grammar` feeding the `auto` fallback, production wiring, GPU
oracle parity.
T1: `prompt_logprobs` (`logprob_token_ids` LANDED 2026-08-10), additional backends
(guidance/outlines), reasoning parsers, beam search wrapper, thinking budget,
repetition detection, torch-Philox bit-exact random parity. T2: rejection
sampler (spec decode), routed-experts return. (`logit_bias`/`allowed_token_ids`/
`bad_words` primitives were ported at M1.7 `aac5138`; their SamplingParams and
OpenAI request/payload wiring remain T1.)

**Spec decode** (`v1/worker/gpu/spec_decode/`): after speed parity, T1 starts
with Qwen3.5/3.6 MTP (the gate checkpoints ship their heads), then DFlash and
the user-promoted DSpark path. Tokenizer-agnostic TLI heterogeneous-vocabulary
mapping is a distinct T1 row; ngram, EAGLE3 and suffix remain T2.
**Eagle-draft `max_position_embeddings` clamp (upstream-sync vllm#49343,
2026-07-30, incr 2):** `SpeculativeConfig::MaybeOverrideDraftMaxPositionEmbeddings`
(`include/vllm/config/speculative.h`) mirrors
`config/speculative.py::_maybe_override_draft_max_position_embeddings` @ `32e657e68`
— raise an eagle/eagle3 draft's mpe up to the target's `max_model_len` (never
lower) so the draft rotary `cos_sin_cache` (our `RotaryEmbeddingBase`, sized to
`max_position_embeddings_`) is not gathered out of bounds at target-scale
positions. Ported AHEAD of eagle draft loading (which is deferred / EAGLE3 is T2),
so there is NO live call site yet — the eagle loader wires it in when it lands.
Gate `tests/vllm/config/test_speculative_draft_max_position_embeddings.cpp` 4/4
(3 upstream pure-unit cases + a never-lower guard). Upstream's 2
model-integration cases (`ModelConfig(EAGLE3_DRAFT/AR_MODEL)`) are SKIPPED — they
need HF-download + a full draft ModelConfig this T0 subset does not carry; they
land with the eagle loader. The PR's bundled `llm_base_proposer.py` block_size
determinism half is N-A (no ported eagle/llm-base proposer).
**Upstream-sync incr 3 (2026-07-30) — quant/config triage, verified SKIP-all
(records-only, no code):** three ranked items resolved as unported-code-path after
reading the real upstream diffs. **vllm#49483** (`8a7b3c299`,
`compressed_tensors/utils.py:145-150`) reorders `_find_first_match(class)` vs
`_match_fused_layer(layer)` in `find_matched_target`; our tree has NO generalized
matcher — compressed-tensors schemes are resolved per-projection by tensor-name
probing (`IsCtNvfp4Projection`/`LoadCtNvfp4W4A16`, `dense_weight_loaders.h:256,266`;
`ignore` list honored inline), so no fused-vs-class ordering exists to fix (bug
structurally impossible). **vllm#48589** (`948107acf`,
`inc/config_parser.py:142-152`) adds an `endswith` suffix fallback to the Intel-
Neural-Compressor `INCConfigParser` — no INC quant path carried (Intel/XPU, N-A).
**vllm#49134** (`0da6e7f3d`, `config/compilation.py`, `custom_op.py`) validates
`CompilationConfig.custom_ops` — no such config surface (fusion catalog is a
declarative `vt::FusedChain`). The T0 RMSNorm pair (**vllm#49750** uncontiguous-
residual perf, **vllm#48391** batch-invariance) stays GPU-gated INVENTORY: our
`RmsNormRowKernel` (`src/vt/cuda/cuda_ops.cu:62`) indexes contiguously (`row*h`,
no `input_stride`) so 49750's residual-stride relaxation is a multi-backend
structural change with no current strided-residual caller, and 48391 depends on the
`vllm_is_batch_invariant` determinism subsystem not carried here — neither is a
clean 1:1 mirror. M-mtp-0 is
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
| Chat templating (full Jinja surface; engine: vendored google/minja `021c229` + one documented lstrip guard) | `renderers/hf.py`, `entrypoints/chat_utils.py` | T0 (2026-07-24 `CLAIM-CAPI-MINJA`: homegrown subset renderer replaced by vendored google/minja behind the same apply_chat_template contract; 5 subset-era test expectations corrected against real jinja2 3.1.2; real Qwen3.5 template renders incl. tools branch) |
| **Tool/function calling** (user-mandated MVP): `tools`/`tool_choice` in chat API, auto-tool-choice, streaming tool-call deltas, Hermes parser first; upstream Qwen3Engine and other parser families remain T1 | `tool_parsers/`, `entrypoints/openai/chat_completion/` | T0 **partial** `18e3efb` (Hermes parser + streaming; local `qwen3` is a Hermes alias, not upstream Qwen3Engine parity; **tool_choice=auto RELAXED via a native LAZY grammar matcher behind vLLM's STRUCTURAL_TAG seam** — free text until `<tool_call>`, then constrain; required/named forced; Coder-XML/Mistral/pythonic parsers deferred) |
| Unified Streaming Parser Engine for tool calls + reasoning | `parser/engine/` | T1 (v0.25.0 inventory; token-ID scanner, event stream, serving adapters and replay tests) |
| Opt-in per-request timing metrics in chat/completion responses | `entrypoints/generate/base/serving.py`, OpenAI protocol/serving | T1 (v0.25.0 inventory; streaming/non-streaming and multi-output suppression) |
| `/tokenize`, `/detokenize`, `/ready`, `/ping`, `/server_info`, `/reset_prefix_cache` | various routers | T1 |
| `/v1/embeddings`, `/pooling`, `/score`, `/rerank` | pooling routers | T2 |
| `/v1/responses`, `/v1/messages` (Anthropic-style), audio endpoints | responses/messages routers | T2 |
| Sleep/pause/resume, LoRA load/unload, profiling, RL weight-update endpoints | various | T2–T3 |
| CLI: `serve`, `bench {latency,throughput,serve}`, `chat`, `complete` | `entrypoints/cli/` | T0 **partial** (separate example server + one in-process bench); T1 `chat`/`complete` spike accepted at [cli-chat-complete.md](specs/cli-chat-complete.md), correcting the prior false claim that the pin lacked those commands. Remote command implementation remains open under `SERVE-CLI-CHAT` |
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
G1/G3-G6 pending). ABI v2 (2026-07-23, `CLAIM-CAPI-STRUCTURED-V2`): additive
structured-output constraint fields on `vllm_sampling_params`
(`structured_json`/`structured_regex`/`structured_choice`/`structured_grammar`/
`structured_json_object`), lowered to `StructuredOutputsParams` under the
upstream exactly-one rule; `VLLM_ABI_VERSION` bumped 1→2. ABI v3 (2026-07-24, `CLAIM-CAPI-CHAT-V3`): `vllm_chat` /
`vllm_chat_stream` — OpenAI chat request JSON in, response / chunk JSON out,
running the engine-side serving-chat pipeline (chat template incl. GGUF
`tokenizer.chat_template` via `LoadChatTemplateFromGguf`, structural-tag
tool_choice, streaming Hermes tool parsing); 17→19 exported symbols. ABI v4 (2026-07-24, `CLAIM-CAPI-PARSER-SELECT`):
`vllm_model_params.tool_parser` (NULL = auto-detect from the chat template via
the ordered marker table in `tool_parsers/detect.{h,cpp}`, llama.cpp-inspired;
one row per parser family). ABI v5 (2026-07-24, `CLAIM-CAPI-REASONING-V5`):
`vllm_model_params.reasoning_parser` (auto-detect "[THINK]"/"<think>" via
`reasoning_parsers/detect.{h,cpp}`, no-detection = disabled, "none" =
force-off). The
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
   **From-necessity CUDA port (2026-07-29, `KERNEL-QUANT-CIQ-GEMM-CUDA`,
   `CLAIM-CUDA-KEEPQUANT-GEMM`):** the CUDA keep-quant GGUF k-quant GEMM
   (`src/vt/cuda/cuda_quant_dot.cu`, the kCUDA provider for `kMatmulBTQuant`) is a
   from-necessity port — the pinned vLLM (0.26, GGUF moved to the OOT plugin) has
   NO CUDA keep-quant GEMM for any k-quant, so DeepSeek-V4's ~2-3-bit routed
   experts (IQ2_XXS/IQ3_XXS/Q2_K) had no on-GPU compressed-weight GEMM and fell to
   the unified-memory CPU reference tier (the 20 ARM cores). It ports the
   MMVQ-style dequant-in-kernel STRUCTURE of llama.cpp @ `237ad9b96`
   `ggml/src/ggml-cuda/mmvq.cu` + `vecdotq.cuh` (warp-per-row, quantize the
   activation then integer-dot the compressed blocks) but reproduces OUR CPU
   keep-quant NUMERICS (Q8_K activation, `ggml-cpu/quants.c` vec_dots), NOT
   llama.cpp's Q8_1 CUDA numerics — so it stays bit-identical to the landed CPU
   oracle it was gated against (GB10, NMSE ≤1e-6, memcheck 0). The `__constant__`
   IQ codebook tables (`cuda_quant_iq_tables.cuh`) are auto-derived from the CPU
   `cpu_quant_iq_tables.h` (single source). Additive kCUDA provider; the CPU
   oracle is untouched.
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
   **The Platform seam itself is now REALIZED and is a faithful port, NOT a
   deviation (2026-07-18, `BACKEND-PLATFORM`).** `include/vllm/platforms/interface.h`
   + `src/vllm/platforms/{platform,cpu,cuda}.cpp` mirror `vllm/platforms/interface.py:134-229`
   1:1 (`class Platform`: `is_cuda`/`is_cpu`, `get_device_capability`/
   `has_device_capability` :409-439, `supported_dtypes` :181-187), self-registered
   per `DeviceType` via the RegisterBackend/GetBackend static-init idiom
   (`CurrentPlatform()` = accelerator-first / CPU-fallback, mirroring how vLLM
   resolves `current_platform`). A `Platform` COMPOSES `vt::Backend` (the vt
   runtime of §9.1) — memory-model queries (`is_unified_memory`,
   `supports_graph_capture`) delegate to the composed backend, so this is an
   organizing seam over the existing deviations, introducing no new one. The
   discrete-vs-unified `ResidencyPolicy` folds PR #4's host-weight-release +
   DevicePool memory-model debt into one advertised policy object;
   `get_attn_backend_priority()` is a stub for the later attention-registry item.
   **Model self-registration is now REALIZED (2026-07-19, extensibility item 5,
   `MODEL-FACTORY-registry` / `CLAIM-MODEL-SELFREG-1`) — faithful port, one small
   recorded deviation.** The fixed `constexpr std::array<ModelRegistration,2>
   kRegistrations` is replaced by a `REGISTER_VLLM_MODEL(...)` static-`Registrar`
   idiom (`include/vllm/model_executor/models/model_registry.h:167-189`) that
   copies the proven `RegisterOp`/`RegisterBackend`/`RegisterPlatform` static-init
   pattern (`src/vt/ops.cpp`, `src/vt/backend.cpp`,
   `src/vllm/platforms/platform.cpp`): each architecture registers itself from its
   OWN TU into the type-erased `ModelFactory`, mirroring how `_VLLM_MODELS` is
   assembled from per-model registrations (`registry.py:682-693`) rather than a
   fixed in-file array. The Qwen dense/MoE arch-specific entry points moved out of
   the `model_registry.cpp` monolith into per-variant TUs (`qwen3_5_dense.cpp`,
   `qwen3_5_moe.cpp`) over a shared `qwen3_5_common.{h,cpp}` (ModelInfo, config
   hook, KV-cache builder, host-logits carrier, borrowed-weights tag); the heavy
   forward machinery (`Qwen3_5Model::`/`Qwen3_5DenseModel::` + DevicePool/matmul/
   GDN) stays in `qwen3_5.cpp` (that deeper shared-machinery factoring is a
   deferred follow-up — scope-disciplined out of this behavior-preserving pass).
   **Deviation:** C++ does not order static init across TUs, so registration
   arrival order is unspecified; the registry applies a stable canonical sort by
   architecture name on first query so `SupportedArchs()`/the unsupported-arch
   error message stay deterministic (vLLM relies on dict insertion order). This is
   cosmetic only — resolution picks the first CONFIG-architecture match and is
   order-independent, so no model resolves differently. Adding a model =
   one new TU + one `REGISTER_VLLM_MODEL` line, ZERO edit to a shared array.
   **The attention-backend registry + Platform-driven priority is now REALIZED
   (2026-07-19, extensibility item 4, `BACKEND-ATTN-REGISTRY` /
   `CLAIM-ATTN-REGISTRY-1`) — faithful port, no new deviation.**
   `include/vllm/v1/attention/registry.{h,cpp}` mirror
   `vllm/v1/attention/backends/registry.py` (`@register_backend` self-registration)
   1:1: a `(DeviceType, name)` registry (`RegisterAttentionBackend` /
   `AttentionBackendRegistrar`) copying the same `RegisterOp`/`RegisterBackend`/
   `RegisterPlatform`/`REGISTER_VLLM_MODEL` static-init pattern, plus
   `SelectAttentionBackendName` — a faithful port of `cuda.py:361-470`
   `get_valid_backends`/`get_attn_backend_cls` (walk the platform's
   capability-ordered priority, return the first REGISTERED = valid backend). The
   item-1 `Platform::get_attn_backend_priority()` stub is filled with the exact
   `cuda.py::_get_backend_priorities:154-166` non-MLA lists (`CudaPlatform`,
   major-10 vs else) and `cpu.py:75-87` (`CpuPlatform`). FLASH_ATTN/GDN_ATTN
   self-register. It reuses (does not extend) the pre-existing CPU FA-NHD-layout
   deviation (§9.1 / `cpu_paged_attn.cpp`): upstream CPU picks CPU_ATTN's
   `[N,H,block,head]` layout while our CPU paged-attn reuses FlashAttention's NHD
   layout. CPU_ATTN was therefore named-but-unregistered here and the walk fell
   through to FLASH_ATTN. **Since [#1371](https://github.com/mudler/vllm.cpp/issues/1371)
   `CPU_ATTN` IS registered** (`src/vllm/v1/attention/backends/cpu_attn.cpp`,
   ported from `cpu_attn.py:39-110`) and the CPU walk stops on it, which is
   upstream's answer; the layout deviation is unchanged and is now recorded on the
   backend itself rather than being the reason it had no registrar.
   The concrete attention KERNEL stays selected
   at the vt:: op table (§9.1), so this is an organizing engine-SELECTION seam
   over the existing runtime, introducing no new deviation. Adding a backend's
   attention = one self-registering TU + one priority slot, ZERO selector/model/
   runner edit. The three `backends.md` portability seams (Platform, attention
   registry, model self-registration) are now all realized.
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
   27B prefill projections (measured GB10). CUTLASS v4.5.0 is header-only (torch-free),
   provided via `-DVLLM_CPP_CUTLASS_DIR` (default `third_party/cutlass`), mirroring
   the current v0.25 FlashInfer dependency. The TU is isolated (own `-isystem` cutlass tree
   + `--expt-relaxed-constexpr`, sm_12xa only, ~37s compile) and gated by
   `VT_CUTLASS_NVFP4`; the op is opt-in (`VT_NVFP4_CUTLASS`) so no existing path
   changes. Only the 27B (compressed-tensors W4A4) uses it; the 35B is FP8 + Marlin
   W4A16 (never the sm120a fp4xfp4 kernel — grounded in `modelopt.py` MIXED_PRECISION
   per-layer resolve). W3-E now also ports v0.25's direct activation-scale
   address emission: `Fp4ScaleLayout` selects linear or padded CUTLASS layout,
   normal and both fused quant producers zero/write the latter directly, and
   true-W4A4 model dispatch defaults direct. `VT_FP4_DIRECT_SF=0` preserves the
   linear producer plus this standalone swizzle. CPU/CUDA bytes, focused
   sanitizer, 27B direct/fallback and 35B inertness gates pass; paired tracing
   removes 624 activation swizzles. Immutable `53ab149` completes all 12
   c2/c16 component legs, but its 32/40 timing + 6/8 memory result is
   plan-selection-confounded. The corrected frozen-map run from runtime
   `d211b8f` plus gate `69a5c45` completes another 12 legs with identical 64/64
   plans: direct gains 1.004483x/1.005044x mean total throughput, but strict
   acceptance still fails at 39/40 timing + 1/8 memory. No conditional exact
   grid or 35B performance run followed; the performance row remains `ACTIVE`
   without speed credit.

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

    **DENSE Marlin (row `KERNEL-MARLIN-DENSE-PORT`, gated OFF `VT_MARLIN_DENSE`,
    2026-08-06)**: the byte-preserving E=1 route. `src/vt/cuda/marlin/libtorch_stable/
    quantization/marlin/` now also vendors vLLM's OWN dense marlin (a DISTINCT
    kernel from the moe one — direct-A, `lda`, no sorted_token_ids/expert_ids/top_k
    gather, its own par-split fp32 C_tmp reduce), all from vLLM @ `555967922`
    `csrc/libtorch_stable/quantization/marlin/`:
      * `kernel.h` ← `kernel.h` (verbatim; `namespace marlin`, dense `MARLIN_KERNEL_PARAMS` with `lda`)
      * `marlin_template.h` ← `marlin_template.h:1-2081` (verbatim dense kernel; the
        SHARED `marlin.cuh`/`marlin_dtypes.cuh`/`dequant.h`/`marlin_mma.h` it includes
        are byte-identical to our existing vendored copies — diff-verified)
      * `marlin_mm_dense.{h,cu}` ← `marlin.cu:326-541` `marlin::marlin_mm` + config
        helpers (`get_marlin_kernel`/`determine_exec_config`/`is_valid_config`/…);
        the torch::stable `marlin_gemm` host wrapper (`marlin.cu:545-894`) is stripped,
        replaced by the torch-free launcher `vt::MarlinDenseGemm`
        (`src/vt/cuda/cuda_marlin_dense.cu`, mirrors `cuda_moe_marlin.cu`). Only the
        original's redundant inner `is_a_8bit` shadow is dropped (identical value,
        avoids `-Wshadow`). `STD_TORCH_CHECK` → `vt_marlin_check.h` as for the moe TU.
      * `kernel_selector.h`, `sm80_kernel_bfloat16_fe2m1f_bfloat16.cu` ←
        `generate_kernels.py` output. KEY: the dense kernel is the SAME 12-param
        `Marlin<>` template as the moe one, so these are the SAME instantiation set —
        the dense kernel BODY + `namespace marlin` come from the local dense
        `kernel.h`/`marlin_template.h` this TU includes.
    New op `vt::OpId::kMarlinDenseGemm` + `MarlinDenseArgs` (ops.h, appended before
    `kCount` — no id shift); routing in `dense_nvfp4_gemm.h` reuses the EXISTING
    resident weights + workspace (same `marlin_permute` repack for dense and moe —
    CONFIRMED via the shared repack ops; no shim needed) with rank-2 operand views
    and NO moe_align. **Compile-VERIFIED GB10 sm_121a (2026-08-06): all 3 new dense
    `.cu` compile clean under the exact production flags** (`-Werror=all-warnings`,
    `-static-global-template-stub=false`, `--generate-code=…sm_121a`). GPU exec gates
    (unit RED-first battery, strict token battery dense-ON, nsys 48-CTA, binding
    c1..c8) are the dgx follow-up; extracted tree kept at dgx `~/dense_check/vllm.cpp`.

11. **Vendored FlashAttention-2 (head-dim-256 GQA prefill implemented; ratio-6
    split-KV decode `ACTIVE`)**: `src/vt/cuda/flash_attn/` is a byte-identical,
    torch-free vendor of vllm-project/flash-attention @ `2c839c33`, still the
    exact FA2 dependency pinned by vLLM v0.25.0 `702f481`. Three stub headers
    replace the ATen/c10 surface (PhiloxCudaState POD + C10_CUDA_CHECK) and
    `fa2_compat_prelude.h` is force-included; tuned FA sources stay pristine.
    The torch-free `src/vt/cuda/cuda_flash_attn_fa2.cu` fills
    `Flash_fwd_params` from `vt::Tensor` views for paged varlen prefill and the
    bounded pure-decode route. Its
    `num_splits=1` statement is correct for ordinary ragged paged varlen, but
    was incorrectly generalized to pure decode: upstream `flash_api.cpp`
    separately performs `seqlenq_ngroups_swapped`, then applies the split-count
    heuristic and emits split main + combine. The exact v0.25 27B trace proves
    that path executes.

    Prefill remains implemented for the natively-BF16 combination only, under
    CMake `VLLM_CPP_FLASH_ATTN` and runtime `VT_FA2_PREFILL`; every other
    combination retains its fallback. **Verified GB10 sm_121a (2026-07-10):
    prefill kernel 3.68× vs our WMMA (475.3→129.2 ms per profile window), with
    same-binary 27B e2e +1.52%/+0.54% at c16/c32 and token-exact ON/OFF gates.**
    Those component values remain valid but are not the current v0.25 binding
    grid.

    W3-G implements the missing Qwen3.6-27B pure-decode specialization: BF16,
    Hq/Hkv 24/4, D256, paged KV, direct logical swap through supported Q/O
    strides, the exact split heuristic, fixed per-device/stream/shape F32
    scratch released at queue teardown, cast-free model dispatch and
    `VT_FA2_DECODE=0` fallback. Ported `test_flash_attn.py:95-217` semantics
    cover the 523/37/2011 fallback vector, ratio-6 B1--16, invalid ratio/window,
    split arithmetic, capture/replay/capacity and two-queue lifecycle in
    `tests/vt/test_ops_paged_attn.cpp:1354-1555`. Qwen3.6-35B-A3B is ratio 8
    and deliberately inert. Binding diagnostics are local ratio-6 decode **22,893 calls /
    8,793.238 ms** versus vLLM FA2 main **23,616 / 7,061.921 ms** plus combine
    **23,488 / 123.245 ms**; unequal windows make this attribution, not a speed
    ratio. Immutable `ae9e8ff` passes clean sm_121a compile, **20/20 +
    454,323** CUDA assertions, zero-error/zero-leak memcheck, both 27B arms,
    correctness-only 35B and paired node trace. Default/fallback switch exactly
    between **240 main+combine / 0 old** and **0 decode combine / 240 old**;
    both preserve 3,536 FP4 GEMMs/producers and all 64 plans. The short prompt
    is performance-negative (**3.246400/1.395488 ms**) and non-binding. The
    completed frozen c2/c16 component covers all 12 legs and 612 requests,
    reaches **1.017668×/1.006548×** mean total throughput, and strict-fails
    **35/40 timing + 5/8 memory**. W3-G earns no speed credit and no exact grid
    or 35B performance run follows under the [W3-G
    spike](specs/fa2-gqa-split-kv-decode.md).

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

    **IMPLEMENTED — packed pure-decode recurrence (2026-07-16;
    `CLAIM-GDN-DECODE-TRITON`).** The GDN packed pure-decode recurrence
    (`fused_recurrent_gated_delta_rule_packed_decode_kernel`,
    `fla/ops/fused_recurrent.py:256-336`, launch `:439-478`, num_warps=1/
    num_stages=3) joins the sanction as the FIRST decode-side vendored kernel, and
    the FIRST landed under a MEASURED-not-inferred codegen proof.
    `triton_kernels/fused_recurrent_packed_decode.py` (FLA body VERBATIM; AOT
    adaptations: scale pinned to Dk^-0.5 in-kernel — same fp32-scalar mis-pack as
    chunk_o; constexpr dims/strides pinned per-shape to the supported dense call
    sites; one dead
    grid-carrier `NBH`=B*HV; state-index ABI adapter `state_idx < 0` for our
    slot-0-valid cache ABI vs FLA's `<= 0`). Exact specializations
    `gdn_decode_h48` (27B) and `gdn_decode_h32` (dense 4B), both H=16, K=V=128,
    BK=128, BV=32; 35B does NOT select packed decode;
    dispatch `TryTritonPackedDecode` in `cuda_gdn.cu` behind runtime toggle
    `VT_GDN_PACKED_DECODE_TRITON` (**default ON since the 2026-07-16 flip** —
    MIRROR policy: it is vLLM's exact token-identical FLA kernel; `=0` rolls back
    to the hand `GdnPackedDecodeKernel` in the same binary; default-ON predicate
    `src/vt/cuda/gdn_packed_decode_triton.h` mirrors `GdnTritonEnvOn`), guarding
    every dtype/stride/shape and falling back to the hand kernel on any mismatch;
    CPU ref + hand kernel PRESERVED, `=0` rollback + non-Triton build byte-inert.
    WHY (MEASURED, dgx `~/work/vllm.cpp-gdn-recurrence/phase1`, GB10 sm_121a,
    cuobjdump -res-usage on the compiled cubins at the c16 shape): the vLLM FLA
    decode cubin holds the register-resident [BV=32,BK=128] fp32 state tile at
    **REG:205, STACK:0, LOCAL:0 (ZERO spills)**; the byte-for-byte hand-CUDA port
    (`GdnPackedDecodeRegTileKernel`, `float sh[128]` + `#pragma unroll`) compiles to
    **REG:255 (the hard ceiling) + STACK:48 (SPILLS to local)** — fatal for a
    state-bandwidth-bound decode (the naive port measured 700.5 vs 793.6 tok/s and
    FAILED the oracle boundary on DGX 54f0541). The legacy shared-staged hand kernel
    is REG:56 / 8-warp / 0-spill at ~83% of peak BW; vLLM ~92%. The gap is register
    allocation / codegen (structure ported 1:1, still spills) — the sanction's
    exact premise, MEASURED not inferred. DECISION: vendored cubin (a portable
    occupancy-aware redesign would fight NVCC's allocator to match ptxas AND fix the
    naive port's latent correctness bug; codegen-bound, not a config we mis-set).
    GATES (GB10 sm_121a, `~/work/vllm.cpp-gdn-recurrence`): AOT-vs-legacy-vs-CPU op
    test 28/28 (proves default stays legacy, `=1` fires the cubin, both match the
    reference); full `test_ops_gdn` 49/49 (2343 assertions); oracle boundary 12/12
    (legacy path bit-exact preserved); **27B model gate 235/235 token-exact with the
    Triton decode path ON** (bit-identical to vLLM through the full model);
    compute-sanitizer 0 errors / 0 leaks; default-off gate 235/235.
    The 27B acceptance campaign established token exactness and a positive
    c16 A/B, and the later default flip made the vendored selector current
    production behavior (`=0` rollback). Detailed chronology remains in the
    append-only ledger/state record.

    **Dense H32 extension checkpoint (2026-07-25).** The current selector
    defaults the vendored path ON and `=0` is the rollback. The new Hv=32
    specialization is gated by the exact raw-packed/FP32-state ABI and remains
    ineligible for 35B. Local sm_120: flag 10/10, GDN 66/66 (4,242), real 4B
    graph/direct/eager 3/3 (1,672). Same-binary H32 AOT vs hand rollback is
    **+4.5906% total/output throughput**. Final graph-node trace proves the AOT
    specialization executes. The parent direct-loader checkpoint remains
    speed-pending at 0.9864x vLLM total/output throughput; see
    [2026-07-25 evidence](../docs/bench-evidence/qwen35-4b-main-repair-20260725.md).

13. **Vendored BLAKE3 + the LMCache MODE-1 (`lm://`) wire protocol port
    (`KV-EXTERNAL-CACHE` W1):** an LMCache C++ CLIENT keys chunks on LMCache's
    OWN blake3 rolling token hash (`lmcache/v1/multiprocess/token_hasher.py:37-49`
    @ `8570aad`), the hash the `blake3` PyPI package LMCache imports computes.
    blake3 is not mirrorable from a vLLM dep — it is the external cache's own
    hash — so we VENDOR the official BLAKE3-team C reference implementation
    (`third_party/blake3/`, upstream tag **1.5.5** / commit **81f772a**, dual
    CC0-1.0 / Apache-2.0), built as a separate `blake3_vendored` static lib with
    the PORTABLE backend forced (`BLAKE3_NO_AVX512/AVX2/SSE41/SSE2`,
    `BLAKE3_USE_NEON=0`). Cleaner mirror than a from-scratch spec
    re-implementation: the C files are UNMODIFIED (clean re-sync), kept off the
    `-Werror` path, and byte-identical on x86-64 AND aarch64 (BLAKE3 is defined
    little-endian; the portable `load32` assembles words byte-wise —
    endian-independent, VERIFIED: `none_hash`/single/chunk vectors identical on
    the dev box and on the `dgx.casa` aarch64 box, both matching the Python
    `blake3` output). The wire FRAMING (`lmcache/v1/protocol.py:214-321`
    `ClientMetaMessage`/`ServerMetaMessage` fixed `struct`; the `CacheEngineKey`
    string codec `lmcache/utils.py:449-561`; the `MemoryFormat.KV_2LTD`
    `[2,L,T,D]` layout `lmcache/v1/memory_management.py:79-133`) is a DIRECT port
    into `src/vllm/v1/kv_offload/lmcache/{remote_protocol,cache_engine_key,
    token_hasher,memory_format}.{h,cpp}`, gated BYTE/BIT-EXACT against fixtures
    captured from the real Python codec
    (`tests/vllm/v1/kv_offload/lmcache/test_lmcache_codec.cpp` vs
    `tests/fixtures/lmcache/lmcache_fixtures.json`, generator
    `scripts/lmcache/gen_lmcache_fixtures.py`). ENDIANNESS deviation: LMCache's
    `struct.pack("iiiiiiiii150s", …)` is NATIVE-endian (no prefix); we encode
    little-endian EXPLICITLY and require client/server to share byte order (true
    for every realistic `lm://` deployment). Pure CPU codec — NO call site routes
    to it (the connector is W3), so every model path is provably inert. `torch`
    is not installed on the build boxes, so the effective `DTYPE_TO_INT` map (with
    the `torch.half is torch.float16` singleton collision resolving float16→2) is
    mirrored from source by reasoning and documented in `remote_protocol.h`; the
    hashes and framing themselves use the REAL Python codecs.

11. **Plugin system: Python entry points → C++ static-init/`dlopen` registration
    (2026-07-29, `ENG-PLUGIN-SYSTEM` / `CLAIM-PLUGIN-SYSTEM`) — legitimate
    mechanical divergence, no behavioral change.** vLLM discovers plugins through
    `importlib.metadata` entry points (`vllm/plugins/__init__.py:36-42`
    `load_plugins_by_group` reads the `vllm.general_plugins` group; each entry
    point resolves to a `register()` function `load_general_plugins` calls,
    `:77-90`). Pure C++20 has no Python entry points, so the faithful analog is a
    process-global REGISTRATION list an out-of-core translation unit or shared
    object populates at load time — the SAME static-init idiom the project already
    uses for models (`REGISTER_VLLM_MODEL`), platforms (`RegisterPlatform`), and
    vt ops/backends (`RegisterOp`/`RegisterBackend`). The new
    `include/vllm/plugins/plugins.h` / `src/vllm/plugins/plugins.cpp` add ONLY the
    discovery+orchestration layer those seams lacked: `RegisterGeneralPlugin` /
    `REGISTER_VLLM_GENERAL_PLUGIN` (the registration seam, mirror of an entry
    point) and `LoadGeneralPlugins()` (mirror of `load_general_plugins` — the
    `plugins_loaded` load-once latch `:33,82-85`, the `VLLM_PLUGINS` allowlist
    `envs.py:1104-1108` parsed identically incl. `""`→`{""}`→no plugin, and the
    per-plugin `try/catch` failure isolation `:68-72`). The BEHAVIOR is mirrored
    1:1; only the transport (Python entry points vs C++ static-init/`dlopen`)
    differs. A general plugin's callback installs its out-of-tree contribution
    through the EXISTING public registries (`vllm::RegisterModel` for a model
    factory — the C++ analog of `ModelRegistry.register_model`,
    `registry.py:1039-1083` — `RegisterPlatform`, the quant registry), so the core
    engine is untouched and byte-identical (W1 never calls `LoadGeneralPlugins`
    from any production path; the engine/CLI `--load-plugins` wiring is a named W3
    residual). The documented C-ABI entry symbol `vllm_plugin_register` is the
    contract a future `dlopen`-based loader (W2) resolves so the static and dynamic
    paths converge on one registration mechanism. Proven with a toy-model
    out-of-core plugin compiled ONLY into the test executable (not the library, so
    the counted 28-arch registry is untouched): `test_plugin_system` 1 case / 29
    assertions, RED-first. Spec [specs/plugin-system.md](specs/plugin-system.md).

12. **Mirror source is HF transformers, not vLLM, for the Parakeet /
    FastConformer audio encoder** (2026-08-07, `CLAIM-PARAKEET-MODEL-P4`, model
    row `MODEL-AUDIO-PARAKEET-ENCODER`, spike
    [specs/parakeet-conformer-encoder.md](specs/parakeet-conformer-encoder.md)).
    vLLM DOES NOT implement this encoder: `vllm/model_executor/models/
    parakeet.py:14` does `from transformers import ParakeetEncoder` and `:61`
    (`ProjectedParakeet.__init__`) instantiates it, as the audio component of
    `nano_nemotron_vl.py` (`registry.py:511-513`). There is therefore no vLLM
    source to mirror for the encoder, the conformer block, the attention, the
    subsampling stack or the CTC head, and every ported file cites transformers
    5.3.0 `transformers/models/parakeet/modeling_parakeet.py` instead. This is a
    provenance deviation, not a behavioral one: HF IS what vLLM runs, so mirroring
    HF mirrors vLLM's behavior exactly. The vLLM-NATIVE halves ARE mirrored where
    they exist — the log-mel front end follows `ParakeetExtractor:138` and
    `vllm/transformers_utils/configs/parakeet.py ExtractorConfig:41`. Three
    sub-deviations, each stated in the ported file's header: (a) the front end has
    no torch/torchaudio/librosa, so its STFT is a direct DFT of the 257 needed
    bins rather than an FFT (float summation order only — the same deviation, and
    justification, as the Whisper path), and it CONSTRUCTS the slaney mel bank in
    double from transformers `audio_utils.mel_filter_bank:453`, which is exactly
    what vLLM calls, while HF's own `ParakeetFeatureExtractor:94-97` uses
    librosa's float32 bank and says at `:83-93` that the only difference is the
    precision; (b) vLLM's 30-second clip splitting (`parakeet.py:253-284`) belongs
    to the Nemotron-VL token budget, not to the ASR model, so the extractor
    processes one clip and a caller that wants the splitting slices first; (c) the
    RNN-T / TDT transducer is a SEPARATE row,
    `MODEL-AUDIO-PARAKEET-TRANSDUCER`, landed 2026-08-07 as spike work item P6.

    **CORRECTION, 2026-08-07.** Sub-deviation (c) used to read "the RNN-T / TDT
    transducer is deliberately NOT ported: it has no upstream in either vLLM or
    HF and is a product call the spike left open, so this row is CTC only". That
    was measured against the transformers INSTALLED on the box, 5.3.0, which
    ships only `ParakeetForCTC`. Upstream `main` implements the entire transducer
    stack (`modeling_parakeet.py` `ParakeetRNNTDecoder:831`,
    `ParakeetRNNTJointNetwork:879`, `ParakeetForRNNT:922`,
    `ParakeetTDTJointNetwork:1035`, `ParakeetForTDT:1052`, plus the greedy loops
    at `generation_parakeet.py:125` / `:271`), so it was never a product call and
    never a deviation: it is mirror work, and it is now ported. The provenance
    deviation is if anything STRONGER for the transducer than for the encoder:
    vLLM wraps only the encoder and has no transducer call site at all, so HF is
    the sole possible source. **Method rule this earns: a grep against the
    installed package is not evidence about upstream. Record the version you
    measured, and check `main` before writing "no upstream" into the record.**
14. **ROCm integrated-APU managed allocation (`BACKEND-ROCM` W1, approach (b)
    from issue #41 F6, maintainer-ratified 2026-08-08).** On a device probing
    `hipDeviceAttributeIntegrated=1` + `ManagedMemory=1` +
    `ConcurrentManagedAccess=1`, `RocmBackend::Alloc` uses
    `hipMallocManaged(hipMemAttachGlobal)` and `UnifiedMemory()` returns true
    exactly then, so the CPU reference tier's host-dereference contract is
    API-guaranteed on XNACK-less RDNA3 APUs (gfx1151/gfx1103 measure
    `PageableMemoryAccess=0`, vetoing the CUDA-shaped W0 probe even though the
    aliasing holds). **No upstream analog exists to mirror:** allocation is
    torch's job in vLLM, `vllm/platforms/rocm.py` knows the APUs only as
    device-name map entries (`rocm.py:75-77`) plus `is_navi`
    (`rocm.py:909-910`), and `csrc/` has no `hipMallocManaged` call at the
    pin — so this is ADDITIVE, grounded in the issue-41 measurements
    (community F6 report), not in an upstream file. Discrete devices are
    byte-identical to W0 (`Integrated=0` kills the branch). Spec:
    `specs/rocm-unified-memory-b.md`; blind-written, community compile+ctest
    evidence PENDING.
15. **Extension platform: Tenstorrent Blackhole (`BACKEND-TENSTORRENT`,
    `DeviceType::kTENSTORRENT` — deliberately not `kBLACKHOLE`, which collides
    with this codebase's pervasive NVIDIA Blackwell/GB10 references).** No
    upstream analog: vLLM has no Tenstorrent platform anywhere, so this is an
    extension platform in the same sense as item 8's Metal/Vulkan — added
    through the mirrored Platform/vt-op seams so it behaves as a vLLM platform
    would. Strategy: mirror decision E1 ([backends.md](backends.md)) rather
    than E2 — Tenstorrent's Tensix cores are a dataflow multicore chip, not a
    SIMT device, so `vt::tt` is proposed as a thin adapter over **ttnn**
    (Tenstorrent's own C++ tensor-op library, confirmed externally
    consumable via its exported `TT-NN` CMake package), the same move Apple's
    MLX was for Metal, rather than hand-written Tensix kernels. `Backend::
    UnifiedMemory()` is `false` (discrete PCIe device) — this DISABLES the
    portable CPU reference tier (`op_provider.h`), so unlike a unified-memory
    backend there is no partial-coverage safety net; every op the target
    model touches must be registered or the run throws. Full design, the ttnn
    op-coverage evidence, and the one identified open risk (`vt::Tensor` is a
    bare device-pointer view; `ttnn::Tensor`'s device-side constructors take
    no equivalent raw-pointer-attach path) are in
    [tenstorrent-backend.md](specs/tenstorrent-backend.md). **STATUS:
    `ACTIVE` — W0 skeleton landed 2026-08-09 (`vt::tenstorrent::Backend`, a
    `Platform` registrar), growing toward OPT-125m: ALL NINE of OPT-125m's ops now
    registered (`kMatmul`..`kPagedAttention`; attention is a host-staged f32
    oracle while Alloc is host memory), 11/11 test cases passing on real
    Blackhole hardware
    (`tests/vt/test_tenstorrent_backend.cpp`). Not yet reviewed or accepted
    by a maintainer. `ACTIVE` means a gated skeleton here, not a supported
    backend — same caveat Metal/Vulkan's own `ACTIVE` status carries.**

16. **Off-pin upstream anchor: Muse Glimmer is ported from an UNMERGED vLLM PR
    (2026-08-10, `MODEL-MUSE-GLIMMER`, issue
    [#268](https://github.com/mudler/vllm.cpp/issues/268)).** Meta released
    Muse Glimmer on 2026-08-08, well after the parity pin `555967922`
    (2026-07-26). There is no `muse_glimmer` code at the pin — `grep -ril
    'muse\|glimmer' vllm/model_executor/models/` at the pin returns nothing —
    and none on vLLM `main` either. The ONLY upstream implementation is
    [vllm#51655](https://github.com/vllm-project/vllm/pull/51655), OPEN and
    approved but unmerged, with 3 of 20 CI checks red, at head `075d645af`
    (a descendant of the pin). Every `file:line` this row cites therefore points
    at a **branch head, not the pin** — a deliberate exception to "port from the
    pinned oracle", taken on explicit developer direction (2026-08-10). It is
    recorded here, and argued for in the commit that introduced it, because no
    checker enforces the anchor rule and the waiver registry has since been
    retired (`a4f72f86`): an exception now lives in the commit message that
    needs it, attached to the diff it excuses. Consequences, all binding while this stands:
    (a) the anchor is mutable — a force-push or review round on #51655 rewrites
    what we cite, so the fetched ref is kept and re-diffed before every
    re-anchor; (b) upstream's own gates have NOT fully passed, so where our
    HF-reference gate disagrees with #51655 the HF reference wins and the
    divergence is reported upstream rather than mirrored; (c) **no speed axis is
    claimable for this model** — the pinned oracle cannot load `muse_glimmer`
    at all (and the checkpoint wants transformers 5.15.0.dev0 vs the pin's
    5.14.1), so there is no honest denominator and every performance axis is an
    OPEN GAP by construction, not a waived one. The exception is discharged by
    #51655 merging plus a pin advance that includes it; until then the row
    carries this deviation. Scope and gates: [muse-glimmer
    spec](specs/muse-glimmer.md) §0.
17. **From-necessity dense non-causal CROSS attention (`vt::OpId::kAttentionCross`,
    2026-08-11, `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model` phase L2,
    issue [#435](https://github.com/mudler/vllm.cpp/issues/435)).** The pinned vLLM
    (`555967922`, 0.26) has no dense non-causal attention op whose QUERY and KEY
    token counts may differ: every attention entry point it exposes is a paged or
    causal decode/prefill path, and `vt::Attention` inherited that assumption and
    REJECTS `Tq != S` — which no cross-attention can satisfy. LTX-2.5's DiT is
    four cross-attentions per block (text→video, text→audio, audio→video,
    video→audio), so the op is a precondition for the model, not a convenience.
    * **Upstream semantics mirrored:** torch
      `scaled_dot_product_attention(q, k, v, attn_mask=..., is_causal=False)` as
      LTX's `PytorchAttention` calls it — Lightricks/LTX-2 @ `fd4ded7f`,
      `packages/ltx-core/src/ltx_core/model/transformer/attention.py:97-102`. The
      additive-bias contract (both the `(mask - 1) * finfo.max` prompt form and the
      log-space self-attention STRENGTH form, dense `[Tq, S]` and key-only `[1, S]`)
      mirrors `transformer_args.py:199-237`.
    * **Written from scratch** in the sense §9.1 means: the three-pass
      max-subtract / exp / weighted-sum STRUCTURE is this tree's own
      `AttentionKernel` (`src/vt/cpu/cpu_ops.cpp`) with the key extent taken from
      the KEY's token count and the optional bias added before the max, so the two
      agree bit-for-bit on a square unbiased call. No vLLM kernel was ported.
    * **Local anchor:** `include/vt/ops.h` (`OpId::kAttentionCross` appended before
      `kCount` — no id shift — plus `AttentionCrossArgs` and the declaration),
      `src/vt/ops.cpp` (`vt::AttentionCross` validation), `src/vt/cpu/cpu_ops.cpp`
      (`AttentionCrossKernel`, the CPU reference). Sole caller: `vllm::Ltx2Attention`
      (`src/vllm/model_executor/models/ltx2.cpp`), which routes on `context ==
      nullptr` — the call's MEANING — so the op a call site dispatches never depends
      on the prompt length.
    * **Backends:** CPU kernel, and — since phase L8 (2026-08-12) — a NATIVE CUDA
      kernel, `src/vt/cuda/cuda_attention_cross.cu`. The CUDA one was owed to "the
      LTX-2.5 device-resident forward, which is the first caller that would need
      it", and that caller arrived. It had to arrive WITH it: before it the op had
      a CPU kernel only, and on GB10 `Backend::UnifiedMemory()` is true, so
      `RegisterReferenceTier` would have installed the CPU kernel for the CUDA
      device and every cross-attention in the DiT — six per block — would have run
      on the HOST while every gate stayed green and "it ran on the GPU" was false.
      The CUDA kernel is a structural port of `AttentionDenseFlashKernel`
      (`src/vt/cuda/cuda_ops.cu:3229-3318`, itself a 1:1 port of the vendored FA2
      `compute_attn_1rowblock`), generalized on the three axes `AttentionCrossArgs`
      exists for: the key extent is KEY's own `S`, there is no causal mode, and an
      optional f32 additive bias joins the SCALED score before the max-subtraction.
      It uses the online-softmax recurrence where the CPU kernel uses the explicit
      three-pass, so the two agree to f32 summation-order slack and are NOT
      bit-identical — the same relationship `AttentionDenseFast` already has with
      `AttentionKernel`.
    * **Tests and evidence:** `tests/vllm/models/test_ltx2.cpp` — the validation
      refusals, the fully-masked-key softmax, the DENSE `[Tq, S]` per-query bias
      rows, the `Hq > Hkv` GQA broadcast, bit-for-bit agreement with `vt::Attention`
      on a square unbiased call, the no-provider refusal, and the six full-DiT
      forward goldens (`scripts/gen-ltx2-goldens.py`, upstream `fd4ded7f`) that run
      it inside the model. Those are all CPU-only.
      **`tests/vt/test_ops_attention_cross.cpp` is the DIRECT gate on both
      backends (2026-08-12).** It exists because of a finding recorded here so it
      cannot recur: when the CUDA kernel landed, every test that reached it went
      through the LTX-2.5 DiT, and all six distinct call geometries that model
      produces are `tiles=1 npl=1 nblk=1 Hq==Hkv` — so the kernel's whole
      flash-tiling machinery was DEAD CODE in every gate, including the
      `ChooseTileCols` halving whose own comment says it prevents a launch failure
      "on exactly the real geometry". A render puts `S` at prompt length and `Tq`
      in the thousands, straight into that regime. The new suite drives
      `tiles>1`, `nblk>1` with a RAGGED last block, `npl>1`, the `Hq>Hkv` GQA
      broadcast, the `head_dim = 128` f32 case where the tile MUST halve
      (`2*64*128*4` = 64 KiB against a 48 KiB launch limit), a bf16 stream at that
      head_dim, a dense `[Tq, S]` bias carried across a tile boundary, and a fully
      masked key placed in a LATER tile. Each case asserts the tile/block/lane
      counts it reaches before asserting the numbers, and the oracle is an
      INDEPENDENT f64 host reference in the test file — not the CPU kernel — so
      the CUDA arm is not gated against a helper that could be wrong in the same
      direction; the CPU-vs-CUDA comparison is kept as a separate, tighter
      statement.
    * **Spec:** [ltx-2.5 spec](specs/ltx-2-5.md) §1.2 and §7. Lifecycle: shipped
      (CPU + CUDA). Owner: the LTX-2.5 row.
18. **LTX-2.5 pipeline recipes are sourced from the CROSS-CHECK, and three upstream
    guiders are refused (2026-08-12,
    `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model` phase L5, issue
    [#435](https://github.com/mudler/vllm.cpp/issues/435)).** Two deviations, both
    forced, both recorded rather than discovered later.
    * **(a) The 2.4 and 2.5 recipe rows have NO binding-oracle source.** vLLM-Omni
      is this project's binding oracle for LTX (spec §3) and its recipe table stops
      at 2.3: `_PIPELINE_RECIPES` keys on exactly `("one_stage","2")`,
      `("one_stage","2.3")`, `("distilled_two_stage","2")`, `("dmd2","2")`,
      `("dmd2","2.3")` (`vllm_omni/diffusion/models/ltx2/ltx2_recipes.py:161-166` @
      `a4ea67a2`). The three rows this port adds — `("one_stage","2.4")`,
      `("one_stage","2.5")`, `("distilled_two_stage","2.5")` — take their VALUES
      from Lightricks `ltx-pipelines` @ `fd4ded7f`, which is the model author's own
      runtime and the spec's designated cross-check: `utils/constants.py:17-23`
      (the distilled sigmas), `:130-179` (`_PARAMS_SINCE_VERSION` /
      `detect_params`, which is what makes (2,5) inherit `LTX_2_4_PARAMS`), and
      `distilled.py:62-84, 170-185` (the ancestral stage-1 sampler, the single
      thing generation 2.5 changes). The SHAPE of the recipe model stays
      vLLM-Omni's. The refusal on an unknown pair is mirrored exactly and never
      relaxed, because a plausible-but-wrong sigma schedule renders rather than
      failing. **The two references disagree on the default negative prompt** —
      Lightricks' carries five leading tags vLLM-Omni's lacks — so both strings are
      kept, each row takes its own source's, and the disagreement is a gated value
      (`kLtx2NegativePromptsAgree`) rather than a preference. Discharged when
      vllm-omni lands native 2.5 (upstream
      [vllm-omni#6066](https://github.com/vllm-project/vllm-omni/issues/6066)), at
      which point the rows re-anchor to it.
    * **(b) `CFGStarRescalingGuider`, `LtxAPGGuider` and `LegacyStatefulAPGGuider`
      are REFUSED, not ported — because NOTHING UPSTREAM CONSTRUCTS THEM.** All
      three appear in the whole `LTX-2` tree only at their own `class` statements
      (`components/guiders.py:31, 78, 129`); every pipeline builds
      `MultiModalGuider` from `MultiModalGuiderParams`
      (`utils/constants.py:49-68`). They are an unported arm, refused by name and
      recorded as owed. `CFGGuider`, `STGGuider`, `MultiModalGuider` and
      `projection_coef` itself ARE ported.

      **CORRECTION, 2026-08-12.** This entry previously justified the refusal on a
      SHAPE claim — that `projection_coef`'s rank-2 `(B, 1)` result "RAISES at
      every rectangular rank >= 3, i.e. at every real `(B, C, F, H, W)` video
      latent". That premise is FALSE and was recorded as a golden, which is the
      failure spec §7.0(b) exists to prevent. Re-measured against upstream: torch
      right-aligns `(B, 1)` onto the last two axes, so the real predicate is
      `B > 1 && shape[-2] not in {1, B}`. At **B = 1** — the ordinary
      single-request video latent — it composes AND is numerically correct,
      because `(1, 1)` is just a scalar; `(2, 128, 8, 2, 16)` composes too. Where
      it composes with `B > 1` it is silently WRONG, applying the per-batch
      coefficient along axis -2 rather than the batch axis. The
      `norm(dim=[-1,-2,-3])` in the two threshold arms (`:114`, `:205`) is a
      SEPARATE constraint needing rank >= 3. The measured matrix now carries
      `B = 1` and `shape[-2] == B` rows and the `square` abstraction — a
      mis-generalization of those two axes — is gone.
    * **Local anchor:** `include/vllm/model_executor/models/ltx2_pipeline.h`,
      `src/vllm/model_executor/models/ltx2_pipeline.cpp`
      (`ResolveLtx2PipelineRecipe`, `Ltx2Guidance`).
    * **Tests and evidence:** `tests/vllm/models/test_ltx2_pipeline.cpp` — the
      recipe table gated against vLLM-Omni's OWN key list (so a row appearing
      upstream fails this gate instead of going unnoticed), the refusal on six
      unknown pairs, the ancestral-sampler split between the 2.0 and 2.5 distilled
      rows, and `kLtx2GuideProbeComposes`, the shape matrix taken from executing
      upstream that is the evidence behind (b).
    * **Spec:** [ltx-2.5 spec](specs/ltx-2-5.md) §3 and §6 (phase L5). Lifecycle:
      shipped (CPU reference). Owner: the LTX-2.5 row.
19. **torchao NVFP4 group-scale UNSWIZZLE (`Ltx2UnswizzleNvfp4BlockScale`,
    2026-08-12, `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model` phase L6,
    issue [#435](https://github.com/mudler/vllm.cpp/issues/435)).** LTX-2.5's
    shipped Gemma-4 text encoder is quantized by **torchao**, not
    compressed-tensors and not modelopt. Measured from the file's own
    `<module>.torchao_nvfp4` U8[240] marker rather than assumed: `{"format":
    "torchao_nvfp4", "block_size": 16, "is_swizzled_scales": true, ...}`. The
    ENCODING is identical to the modelopt W4A16 path we already have — E2M1 nibble
    pairs, one fp8-e4m3 scale per 16 inputs, one F32 `weight_scale_2` used as a
    MULTIPLIER. The single delta is that the group-scale tensor is stored in the
    cuBLAS block-scaling-factors layout, so it needs inverting before the existing
    dequant can read it. **No new quant scheme was added.**
    * **Upstream semantics mirrored:** the permutation vLLM's own producers apply —
      `swizzle_blockscale`, `vllm/model_executor/layers/quantization/utils/
      nvfp4_utils.py:44-49`, and `to_blocked`,
      `vllm/model_executor/layers/quantization/qutlass_utils.py:165-180`, at the
      parity pin `555967922`. The two are the same permutation written twice;
      `qutlass_utils.py`'s own header records it as a copy of
      `torchao/prototype/mx_formats`, i.e. of the module that quantized this
      checkpoint. NVIDIA's layout reference:
      <https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout>.
    * **Written from scratch** in the sense §9.1 means: only the INVERSE, as an
      explicit index formula. Both upstream writings are forward-only, and
      `swizzle_blockscale` calls `.cuda()` unconditionally, so there is nothing to
      port verbatim. The decode itself is `DequantNvfp4ToBf16`
      (`include/vllm/model_executor/model_loader/nvfp4_dequant.h:59`), UNCHANGED.
    * **Local anchor:** `include/vllm/model_executor/models/ltx2_loader.h`,
      `src/vllm/model_executor/models/ltx2_loader.cpp`
      (`Ltx2UnswizzleNvfp4BlockScale`, `ParseLtx2TorchaoNvfp4Marker`,
      `Ltx2DequantTorchaoNvfp4ToBf16`).
    * **Tests and evidence:** `tests/vllm/models/test_ltx2_loader.cpp`, three ways —
      the five swizzle cases in `tests/vllm/models/ltx2_quant_goldens.inc`; a
      512-byte tile of the SHIPPED text encoder's own `weight_scale` with its values
      decoded by torch's fp8-e4m3; and the two REAL manifests
      (`ltx2_fp8_dit_manifest.inc`, 6124 tensors; `ltx2_nvfp4_te_manifest.inc`,
      1688), captured from the files' own headers with no payload read. The swizzle
      oracle RUNS: `scripts/gen-ltx2-quant-goldens.py` loads vLLM's own `to_blocked`
      out of the pinned checkout with `importlib.util.spec_from_file_location` and
      calls `to_blocked(x, backend="torch")`. The earlier "not importable, so it
      cannot be executed here" limit was WRONG — `import vllm.*` does die in
      `vllm.distributed` on a missing `zmq`, but `qutlass_utils.py` needs only
      torch, `vllm.triton_utils` and `vllm.utils.math_utils`, so loading the file
      directly runs the real producer. Regenerating against it left every golden
      byte IDENTICAL to the transcription's, and upstream's executing `to_blocked`
      reproduces the shipped file's first 512 `weight_scale` bytes from the emitted
      linear tile.
    * **OWED, and precisely:** the `backend="triton"` arm of `to_blocked` and
      `swizzle_blockscale` — vLLM's other two writings of the same permutation —
      are pinned by source fragment only. Neither can execute on this host (no
      active Triton driver; `swizzle_blockscale` calls `.cuda()` unconditionally),
      so running them needs a GPU host. **CLOSED at L7 (2026-08-12):** Lightricks'
      first-party NVFP4 DiT
      (`ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`, 18.72 GB) was behind
      an un-accepted HF gate when this was written and is now on the NAS; the
      SHIPPED file's own header is read by
      `tests/vllm/multimodal/test_ltx2_video.cpp` (`LTX2_CHECKPOINT_ROOT`), which
      resolves it onto the L2 contract and checks its declared config against the
      shape-derived one. Still owed: the torchao arm of the Gemma TOWER itself
      (`Gemma4Weights`) is not
      wired — `ltx2_text_encoder.h` declares no tower contract, so L6 loads the two
      caption projections, the asset pack and the geometry, and VALIDATES every
      tower module without materializing it. **CLOSED at L10 (2026-08-12):**
      `Ltx2LoadGemmaTowerFromSafetensors` (ltx2_text_encoder.h/.cpp) materializes
      the tower onto `Gemma4Weights`, reusing L6's
      `Ltx2DequantTorchaoNvfp4ToBf16` unchanged. It reads the FLAT
      `model.layers.{i}.*` names the LTX file ships — not the
      `model.language_model.layers.{i}.*` form `gemma4_weights.cpp` reads, which
      is why the tower could not simply go through that loader — and resolves
      each layer's geometry from `layer_types`, `global_head_dim` and
      `num_global_key_value_heads` rather than from the stored tensor widths,
      which are HALF the logical ones under NVFP4. It REFUSES by name: a PLE
      config against a checkpoint with no PLE tensors, `num_kv_shared_layers`
      != 0, a `v_proj` present or absent against what `attention_k_eq_v`
      declares, and a module in neither the BF16 nor the torchao-NVFP4 form.
      The Gemma config is an INPUT and is never inferred: the shipped
      `vonkaiser` build has NO `__metadata__` at all, and `layer_types`,
      `global_head_dim`, `num_global_key_value_heads`, `attention_k_eq_v` and
      both `rope_parameters` entries each move every hidden state while leaving
      the tensor set byte-identical.
    * **Spec:** [ltx-2.5 spec](specs/ltx-2-5.md) §1.4 and §6 (L6). Lifecycle:
      shipped (host + load-time device staging). Owner: the LTX-2.5 row.
    * **CLOSED 2026-08-13 by phase L9a — and the diagnosis it closes was WRONG.**
      This entry previously recorded that the FIRST-PARTY
      `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` stores `weight_scale`
      in the LINEAR `[N, K/16]` layout, and that a linear read was the missing
      piece. **It does not, and it was not.** The bytes are SWIZZLED — the same
      permutation this entry already inverts — merely declared in the
      cuBLAS-padded framing `[round_up(N,128), round_up(G,4)]` = `[4096, 256]`
      instead of torchao's `to_blocked` `[32*ceil(N/128), 16*ceil(G/4)]` =
      `[1024, 1024]`. For every layer in that file `N % 128 == 0` and
      `G % 4 == 0`, so the padded framing is NUMERICALLY IDENTICAL to the linear
      shape; that coincidence is what made the wrong diagnosis look right, and it
      is why no shape test could have settled it.

      The file additionally packs element `2j` in the HIGH nibble, which no shape
      encodes at all. Both facts were established by correlating the dequantized
      weights against the `vonkaiser` FP8 DiT of the same base weights (0.9956
      with 9.46% relative rms, against 0.0004-0.26 for every other reading), and
      independently confirmed in Lightricks' own runtime
      (`ltx-kernels/docs/NVFP4.md:27-29`, `csrc/nvfp4/quantize.cu:26-31`,
      `ltx-core/quantization/nvfp4/linear.py:6-7`).
    * **Resolution.** `Ltx2ResolveNvfp4Producer` discriminates on the
      `torchao_nvfp4` marker (torchao always writes one, so its absence excludes
      torchao), corroborates with the framing, and REFUSES by name on any other
      combination. `Ltx2UnswizzleNvfp4BlockScale` was already framing-agnostic and
      is UNCHANGED. The nibble order became an `Nvfp4NibbleOrder` parameter on
      `DequantNvfp4ToBf16`, defaulting to low-first so every pre-existing caller
      is untouched by construction — see
      [nvfp4-nibble-order spec](specs/nvfp4-nibble-order.md), which also records
      why H3's nibble-swap-at-load was NOT reused and when it must be.

19. **LTX-2.5 phase L7 — the family behind `vllm::multimodal::VideoEngine`, and
    the driving loop.**
    * **Upstream source:** Lightricks/LTX-2 @ `fd4ded7`,
      `packages/ltx-pipelines/src/ltx_pipelines/` — `distilled.py:186-300`
      (`DistilledPipeline.__call__`), `utils/blocks.py:500-582`
      (`DiffusionStage.__call__`) and `:212-235` (`_build_state`),
      `utils/samplers.py:39-79` (`euler_denoising_loop`) and `:488-558` (the
      ancestral driver), `utils/helpers.py:428-447` (`create_noised_state`),
      `:462-464` (`post_process_latent`), `:466-503`
      (`modality_from_latent_state`, `timesteps_from_mask`),
      `utils/denoisers.py:214-252` (`SimpleDenoiser`); plus `packages/ltx-core`
      `model/transformer/model.py:590-604` (`X0Model.forward`), `utils.py:38-50`
      (`to_denoised`), `types.py:70,108-123,164-200` (the scale factors and both
      latent-shape derivations), `tools.py:139-184` / `:246-280` (the two
      `LatentTools`), and the four configurators
      (`video_vae/model_configurator.py:21-24,81-94,255-265`,
      `audio_vae/model_configurator.py:13-39,49-88,108-141,184-190`,
      `upsampler/model_configurator.py:12-30`).
    * **Written from scratch**, and recorded as such: the noise STREAM. Upstream
      draws from a seeded `torch.Generator`; reproducing torch's stream
      bit-exactly decides WHICH sample comes out, not whether the pipeline is
      right, so `SplitMixGaussian` is a documented splitmix64 + Box-Muller source
      drawn in upstream's own ORDER (video before audio, one draw per state per
      step) and is NOT torch's. Same call MiniMax-H3 made
      (`minimax_h3.h:1895-1897`). The cost is stated where it is taken: a clip
      rendered here is a different sample from the same distribution, so it is
      not comparable to an upstream render frame by frame, and sample-level
      comparison needs the noise supplied from outside.
    * **Local anchor:** `include/vllm/multimodal/ltx2_video.h`,
      `src/vllm/multimodal/ltx2_video.cpp`; the loaders it needed are
      `Ltx2LoadVaeWeights`, the three `SDOps` key-rule sets and the four config
      parsers in `ltx2_loader.{h,cpp}`.
    * **Tests and evidence:** `tests/vllm/multimodal/test_ltx2_video.cpp` —
      registration, detection by tensor name (prefixed and de-prefixed), a
      STRUCTURAL e2e over a reduced-dimension checkpoint set written in the
      shipped file format, the same generation driven through `include/vllm.h`
      alone, every refusal by name, and the shipped Lightricks checkpoints when
      `LTX2_CHECKPOINT_ROOT` is set.
    * **OWED, and precisely:** (a) the forward on an accelerator — CLOSED by
      phase L8, entry 20 below; (b) the Gemma-4 tower, so a PROMPT
      cannot be encoded and conditioning is prompt-embeds only; (c) image /
      keyframe / reference conditioning, which needs the video VAE's ENCODER;
      (d) the full-scale render, which at 21.00B needs ~76 GB of f32 weights and
      ~2.6e14 FLOPs per denoise step; (e) parity against the BINDING oracle and
      any speed number, both structurally pending per the spec's §0 and §3.
    * **Spec:** [ltx-2.5 spec](specs/ltx-2-5.md) §5 and §6 (L7). Lifecycle:
      shipped (CPU, structural e2e). Owner: the LTX-2.5 row.
20. **LTX-2.5 phase L8 — the DEVICE-RESIDENT DiT forward, and the `vt::OpId::kLtx2`
    glue table (2026-08-12,
    `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model` phase L8, issue
    [#435](https://github.com/mudler/vllm.cpp/issues/435)).** Phase L7 wired
    LTX-2.5 through `vllm::multimodal::VideoEngine` and had to REFUSE `device = 1`,
    because L2's forward was f32-only by declaration and L6's staging was bf16 and
    refused to widen. This is the forward that makes the two meet: the same graph
    with every activation in device memory and the stream in the checkpoint's own
    dtype.
    * **Upstream semantics mirrored:** unchanged from L2 — Lightricks/LTX-2 @
      `fd4ded7f`, `packages/ltx-core/.../transformer/{model,transformer,attention,
      adaln,rope,feed_forward}.py`. This entry adds no new upstream BEHAVIOR; it is
      the same `LTXModel.forward` at a different residency, which is why it is
      gated against the SAME frozen goldens rather than new ones.
    * **Dtype polarity, which is the design decision:** the device stream is
      **bf16**, because upstream resolves ONE model dtype and every layer inherits
      it (`model.py` has no per-layer dtype at all) and bf16 is what
      `Ltx2StreamDitToDevice` already puts on the device. `kF32` is accepted as a
      GATE arm only — it is the L2 parity dtype, and it is what lets this forward
      be compared against `ltx2_goldens.inc` at f32 round-off instead of at a bf16
      band. Nothing widens a bf16 load to reach it. The `scale_shift_table` family
      stays F32 on both arms because the CHECKPOINT stores it F32
      (`ltx2_loader.h:64-66`); those are a few kilobytes against a 21 GB model.
    * **Written from scratch** in the sense §9.1 means: seven small kernels the
      shared `vt::` surface does not express — `ada_value`, `modulate`,
      `add_gated`, `gate_heads`, `rope` (LTX's split and interleaved layouts),
      `output_modulate`, and plain ungated `silu`. Each is a 1:1 transcription of a
      named host helper in `ltx2_dit.cpp` / `ltx2.cpp`, in the same arithmetic
      order. Everything else reuses tuned shared ops (`vt::MatmulBT`, `vt::Add`,
      `vt::RmsNorm`, `vt::LayerNorm`, `vt::GeluTanh`, `vt::Attention`,
      `vt::AttentionCross`); no new GEMM, norm or attention kernel was added.
    * **Local anchor:** `include/vllm/model_executor/models/ltx2_device.h`,
      `src/vllm/model_executor/models/ltx2_device.cpp`,
      `src/vllm/model_executor/models/ltx2_device_resolve.cpp` (a separate TU so
      the `vt::GetOp` cast links in CPU-only builds), `src/vt/cpu/cpu_ltx2.cpp`,
      `src/vt/cuda/cuda_ltx2.cu`, and `vt::OpId::kLtx2` appended before `kCount`
      (no id shift). Registered on BOTH `kCPU` and `kCUDA`, so the port's STRUCTURE
      is covered by CPU CI and a GPU gates the KERNELS.
    * **Tests and evidence:** `tests/vllm/models/test_ltx2_device.cpp`, its own
      target so `test_ltx2`'s 29/1615 baseline does not move. It runs all six
      upstream forward cases device-resident against the SAME `ltx2_goldens.inc`
      the CPU arm meets, the bf16 stream against those goldens at a bf16 band PLUS
      a check that it actually DIFFERS from f32 (a bf16 arm that matched f32 would
      mean the dtype policy silently was not applied), a CUDA-vs-host comparison at
      identical inputs, and every refusal by name.
    * **FOUND BY THE bf16 ARM, which is why that arm exists.** The staging
      predicate matched the `scale_shift_table` SUFFIX, which silently excluded
      `scale_shift_table_a2v_ca_video` / `..._audio`; those were staged bf16 and
      then read through `ada_value`'s `const float*` table parameter. The
      audio<->video cross gate became 2.85e32 and the video stream 6.89e30 after
      one block — while the f32 arm, where the mismatch cannot arise, stayed green
      at 1e-7. The predicate is now a SUBSTRING match and every table read is
      guarded by an explicit `CheckTableF32`, so a future miss is a named refusal
      rather than a reinterpretation.
    * **MEASURED ON GB10 (2026-08-12), and it is where the shipped model stands.**
      A 21.00B DiT stages and runs device-resident: vonkaiser
      `ltx-2.5-22b-distilled-fp8.safetensors` (21.0 GB, 6124 tensors) staged in
      271.5 s and one bf16 forward at 48 layers / inner 4096 / head_dim 128
      produced finite, non-degenerate output. The wall-clock is SIZING ONLY and is
      not a speed result — spec §0, no production-configuration denominator exists.
      The FIRST-PARTY `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`
      (18.72 GB, 7876 tensors) did NOT stage at that time. **The reason recorded
      here was WRONG and is superseded by entry 18 above (phase L9a,
      2026-08-13):** the file does not store a LINEAR `[N, K/16]` scale. Its bytes
      are SWIZZLED, declared in the cuBLAS-padded framing — which for these
      geometries is numerically identical to the linear shape, which is precisely
      why the wrong reading looked confirmed — and it packs the OPPOSITE nibble
      order. The refusal was correct; its diagnosis was not. Loading it needed a
      producer discriminator and a nibble-order arm, not a linear read. Phase L7's
      shipped-checkpoint test only parsed the MANIFEST, which is why this surfaced
      as late as it did: nothing had materialized a tensor from that file before.
    * **CORRECTED 2026-08-12 by an adversarial review of L7+L8 (row
      `LTX25-L8-FIX`). The headline held; five records did not.**
      1. **The FP8 DiT this row RAN carries no `__metadata__` at all.** The engine
         comment claimed "both shipped LTX-2.5 DiTs DO carry one". Read from the
         NAS: the first-party NVFP4 file declares
         `['config','gemma_source_checkpoint','license','model_version']`; the
         vonkaiser FP8 file declares NO `__metadata__` key. So for the copy L1-L6
         gated against and L8 ran on the GPU, the config-adoption branch never
         executed and the DiT silently took `double_precision_rope = false` and
         `av_ca_timestep_scale_multiplier = 1` against LTX-2.5's declared
         `float64` and `1000`. Both move every RoPE angle and every audio<->video
         modulation. The run's own claims survive it — the weights stage, the
         forward executes device-resident, the output is finite and
         non-degenerate, none of which depends on the RoPE precision — but the
         CONFIGURATION was unstated, which is what made it a defect. A DiT that
         declares no config is now REFUSED by the engine unless the caller names
         one through the `dit_config_path` extra, mirroring how `model_version`
         was already handled; the device gate resolves it through the same
         `Ltx2AdoptDeclaredDitParams` and ASSERTS which configuration it ended up
         running under, in both branches.
      2. **The L7 in-flow repair was ungated.** Deleting `im.dit.params =
         declared;` left the whole suite green. `Ltx2VideoEngine::dit_params()`
         now exposes what the ENGINE loaded and `test_ltx2_video.cpp` asserts on
         it against a manifest control; the same deletion now fails 2 cases / 4
         assertions. The disagreement-refusal branch, also unexercised, is gated
         on both the declared and the supplied config.
      3. **`test_ltx2_device`'s recorded GB10 baseline was 12/547; it is 13/547.**
         The env-gated shipped-checkpoint case is still COUNTED when it skips.
         Assertions were right. **After this row it is 13/552 on GB10** — the F9
         cross-backend bf16 arm adds exactly five assertions, and the CPU-backend
         arm stays at its own 13/498.
      4. **"bit-identical" overstated the bf16 evidence.** What was measured is
         that both arms print the same max|diff| against the goldens at six
         significant figures, which is consistent with bit-identity and does not
         establish it; nothing compared the two backends' outputs to each other at
         bf16. That comparison now exists inside the CUDA-vs-host case, and is
         asserted as a bf16 BOUND with the measured value printed.
      5. **`device = N` relabelled a queue instead of selecting one.** The load
         called `Backend::CreateQueue()` — which `backend.h:212-217` records as a
         temporary index-0 shim — and then overwrote `queue->device.index`, so on
         a multi-GPU host `device = 2` would have run on GPU 0 while every
         residency check agreed it was index 1. It now uses the free function
         `vt::CreateQueue(Device)` and refuses an index with no registered
         backend, by name. Unreachable on single-GPU GB10, which is why it had to
         be fixed before a second device exists.
    * **RE-VERIFIED ON GB10 2026-08-12 (row `LTX25-L8-FIX`), one `flock` hold,
      2305 s of waiting behind a foreign holder.** The shipped vonkaiser FP8 DiT
      (6124 tensors) staged in 185.9 s and one bf16 forward at 48 layers / inner
      4096 / head_dim 128 ran device-resident in 0.42 s (SIZING ONLY, not a speed
      result — spec §0). Output finite and non-degenerate: absmax video 0.300781,
      audio 2.14062. `VT_OP_PROVIDER_STATS=1` on that run: **8 distinct ops on
      device=1, all `vt-native`, ZERO `vt-cpu-ref` and ZERO `<none>`** — so the
      forward is device-native and the reference tier never served it.
      `test_ltx2_device` 13/552 SUCCESS.
      **The CUDA `AttentionCross` tiling is now MUTATION-PROVED**, four mutants
      built unlocked and executed inside the hold. Three are INVISIBLE to every
      LTX-2.5 gate and caught only by `test_ops_attention_cross`: the bias read by
      within-tile column instead of absolute key (5 cases red / device gate
      SUCCESS), the GQA kv-head `h/(hq/hk)` replaced by `h%hk` (1 case red /
      device gate SUCCESS — identical whenever `Hq == Hkv`, which is every
      LTX-2.5 geometry, so it is invisible there BY CONSTRUCTION), and the
      shared-memory limit raised so `ChooseTileCols` never halves (device gate
      SUCCESS; the direct gate's assertion count DROPS 63 -> 59 because the
      `d=128` f32 case throws on the launch — which CONFIRMS the halving's own
      comment that a fixed 64-column tile fails "on exactly the real geometry").
      The fourth, removing the online-softmax rescale, is caught by BOTH gates and
      is recorded as such: the rescale runs at every KEY, not only at tile
      boundaries, so it is not evidence about tiling.
    * **FOUND WHILE GATING THE ABOVE, and it is NOT an LTX-2.5 defect: the shared
      `DevicePool` is DEVICE-BLIND.** `vllm::Pool()` (device_pool.h) is a
      process-wide singleton whose free list is
      `unordered_map<size_class, vector<void*>>` — the DEVICE is not part of the
      key. So a block `cudaMalloc`ed for a CUDA-queue forward is handed straight
      back to a CPU-backend `DBuf` of the same size class, and the CPU backend's
      `Copy` is a host `memcpy` on a device pointer. MEASURED on GB10 2026-08-12:
      SIGSEGV in `__memcpy_sve <- UploadStream <- PrepareStreamDev`, with
      compute-sanitizer reporting ZERO device errors because the fault is
      host-side. It had never been reachable because no test had run a bf16
      CPU-backend device forward AFTER a bf16 CUDA one; at f32 the two arms land
      in different size classes and never trade blocks. The pool already carries
      exactly this invariant for STREAMS — `AuxPool()` exists because "two streams
      sharing one pool BREAKS" its reuse ordering — and the first fix used that
      same sanctioned seam: the CPU arm ran under an `ActivePoolScope` with its
      own pool. **That workaround is GONE, and so is the fault it worked around.**
      `POOL-DEVICE-KEY` ([#516](https://github.com/mudler/vllm.cpp/issues/516),
      [`specs/pool-device-key.md`](specs/pool-device-key.md)) states the DEVICE
      half at the pool itself: a `DevicePool` is bound to one backend, `Pool(b)`
      is the only spelling and there is no device-less one, every operation
      throws on a foreign backend, and the per-caller scope in
      `test_ltx2_device.cpp` was DELETED in the same change — because a list of
      remembered callers is what this fault was, and leaving one behind would
      have disarmed the only test that exposes the silent-NaN direction.
    * **OWED, and precisely:** (a) the prompt-K/V cache on the device path, which
      is REFUSED by name rather than ignored; (b) an FP4-RESIDENT arm — the
      `LinearDev` seam is one parameter away from the shared Marlin W4A16
      dispatcher MiniMax-H3 routes through, but `Ltx2StreamDitToDevice` dequantizes
      to bf16 at load, so keeping the packed weights resident is loader work this
      phase did not do; (c) every speed number, structurally pending per the spec's
      §0 — no production-configuration denominator exists.
    * **Spec:** [ltx-2.5 spec](specs/ltx-2-5.md) §6 (L8). Lifecycle: shipped
      (CPU + CUDA, bf16 and f32 streams). Owner: the LTX-2.5 row.

21. **LTX-2.5 phase L9c — the EMBEDDINGS CONNECTOR reaches the render path, and the
    two references disagree about which module owns the sort.**
    2026-08-13, `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model` phase L9c,
    issue [#435](https://github.com/mudler/vllm.cpp/issues/435).
    * **Upstream source:** Lightricks/LTX-2 @ `fd4ded7f`,
      `packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py:70-95`
      (`EmbeddingsProcessor.create_embeddings`) and `:23-48` (its three helpers), plus
      `embeddings_connector.py:194-256` (the two configurators) and
      `encoders/encoder_configurator.py:331-346` (`EMBEDDINGS_PROCESSOR_KEY_OPS`,
      which is what says these weights belong to the TEXT ENCODER even though they
      ship inside the DiT file).
    * **Local anchor:** `Ltx2ConnectorCreateEmbeddings`
      (`src/vllm/model_executor/models/ltx2_connector.cpp`), `Ltx2ParseConnectorConfig`
      and `Ltx2LoadConnectorWeights` (`ltx2_loader.cpp`), and the call in
      `Ltx2VideoEngine::Load` (`src/vllm/multimodal/ltx2_video.cpp`).
    * **What was actually wrong before it.** `Ltx2ConnectorForward` landed at L5 and
      was gated against upstream on five arms — and its ONLY caller was its own test.
      The render handed the prompt-embeds file straight to cross-attention, and the
      two `*_embeddings_connector` families (129 tensors each, present in both shipped
      DiTs) were reported as unported and stepped over. A brick with a golden and no
      caller is indistinguishable, from the outside, from a brick that is wired.
    * **THE TWO REFERENCES DISAGREE, and it is recorded rather than resolved by
      preference.** `ltx_core` right-pad-sorts the features in the PROCESSOR before
      calling the connector ("Connectors expect right-padded input",
      `embeddings_processor.py:80-84`); `diffusers` folds that sort INTO the connector
      (`src/diffusers/pipelines/ltx2/connectors.py`, the
      `torch.argsort(1 - binary_attn_mask, stable=True)` branch) and its comment
      claims that matches "the original LTX implementation" — true only because
      `ltx_core` does it one level up. They compose to the same function and differ in
      which module owns it. This port follows `ltx_core`, so `Ltx2ConnectorForward`
      stays a faithful port of `Embeddings1DConnector`.
    * **WHERE THEY AGREE AND IT LOOKS LIKE A BUG.** `_to_binary_mask` is
      `encoded_mask < 0.000001` (`embeddings_processor.py:46-48`). An additive mask
      holds `0.0` for KEPT and `-finfo(f32).max` for PADDED, and BOTH satisfy it — so
      the mask handed to the DiT is one at every position, and the video-only multiply
      that follows is an identity on every reachable path. `diffusers` writes the
      identical `(video_attn_mask < 1e-6)` and the identical multiply. Checked on both
      BEFORE mirroring it, because the reading a port arrives at by reasoning about
      intent (`>= 0`) is the opposite at padded positions. Gated as the surprising
      behaviour, so "fixing" it REDs: `test_ltx2_pipeline`, case "ltx2 the processor's
      binary mask mirrors a comparison that looks backwards".
    * **THE REGISTER BOUNDARY IS NOW NUMERICALLY REACHABLE (2026-08-13, review
      finding F1 on the L9c PR).** `prompt_embeds_valid_rows` is the single knob
      deciding which positions become the connector's TRAINED registers, and the
      only case covering it asserted that valid=4 renders differently from valid=2.
      Every monotone corruption of that boundary keeps that true, so a fresh
      reviewer's `prompt_valid_rows + 1` left the suite fully green — one padded row
      conditioned on caller junk instead of a register: finite, correctly shaped,
      plausible, wrong. Closed by asserting WHICH positions are registers, from both
      sides: perturbing rows `[valid, N)` must move NOTHING (they are substituted at
      `embeddings_connector.py:148-150`) and perturbing row `valid - 1` must move
      something. `test_ltx2_video`, case "the register boundary sits EXACTLY at the
      valid-row count"; RED under `+ 1`, `- 1`, `0` and `v_rows`.
    * **A CONFIG VALUE THAT IS NOT NEAR ITS DEFAULT.** LTX-2.5 declares
      `connector_positional_embedding_max_pos = [4096]` where `Embeddings1DConnector`'s
      class default is `[1]`, and `get_fractional_positions` DIVIDES the token index by
      it (`rope.py:132-141`). No shape can see the difference; the default is every
      RoPE angle wrong. `positional_embedding_theta` is deliberately NOT read from the
      DiT config even though one is declared, because neither configurator passes it —
      reading it would be a re-invention rather than a port.
    * **ONE KEY WHERE THIS PARSE DIVERGES RATHER THAN MIRRORS (2026-08-13, review
      finding F3).** `Ltx2ParseConnectorConfig` reads
      `connector_num_learnable_registers`, which NEITHER configurator does
      (`embeddings_connector.py:194-219` and `:222-256` both leave it at the class
      default of 128), so "mirrors both configurators key for key" is not literally
      true of this one key. The divergence is kept — a checkpoint declaring something
      else must not be silently run at 128 — and is now ENFORCED rather than asserted:
      `test_ltx2_video`, case "a connector config that disagrees with the FILE is
      refused", subcase "a register count the file's TABLE does not carry is refused",
      which REDs when the read is made inert.
    * **The record changed, not just the code.** The two connector families are no
      longer reported as unported: this port reads them, so naming them would say
      something untrue about the tree and would demand `allow_unported_modules` from a
      caller whose checkpoint is read completely. Asserted as an ABSENCE in
      `test_ltx2_loader`, case "ltx2 loader: the unported families are refused by name,
      not absorbed" (`CHECK(conn_ck.unported.empty())`), so restoring the old behaviour
      REDs.
    * **OWED, and precisely:** the Gemma-4 TOWER, so what ENTERS the connector is still
      whatever the caller put in the prompt-embeds file — the link below the tower is
      real, the tower is not. `prompt_embeds_valid_rows` exists because a file carries
      no tokenizer mask and the connector REPLACES padding with learned registers.
    * **Spec:** [ltx-2.5 spec](specs/ltx-2-5.md) §6. Lifecycle: shipped (CPU host
      module; f32, an annotated escape — it runs once per load and its output is
      narrowed to the stream dtype on upload). Owner: the LTX-2.5 row.

17. **Ahead-of-pin forward port: the Qwen3.5 TEXT-ONLY arms are anchored on a
    POST-PIN upstream commit (2026-08-12,
    `MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm` +
    `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm`, issue
    [#490](https://github.com/mudler/vllm.cpp/issues/490)).** Our parity pin is
    `555967922` (2026-07-26), whose `vllm/model_executor/models/registry.py`
    carries only `Qwen3_5ForConditionalGeneration` and
    `Qwen3_5MoeForConditionalGeneration`. The text-only arms
    `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` arrived upstream AFTERWARDS,
    in [vllm#50210](https://github.com/vllm-project/vllm/pull/50210) at
    `ad5d29db7` (a descendant of the pin). Every `file:line` this row cites —
    `registry.py:202-203`, `qwen3_5.py:296-300`, `qwen3_5.py:439-449` — therefore
    points at a **post-pin head, not at the pin**, which is a deliberate
    exception to "port from the pinned oracle". Same shape as deviation 16 above
    and taken for the same reason: at the pin the feature does not exist at all,
    so there is nothing there to mirror. It is recorded here, and argued for in
    the commit that introduced it, because no checker enforces the anchor rule.
    Consequences, all binding while this stands:
    (a) this row does **not** advance the pin and reconciles nothing else in the
    `555967922..ad5d29db7` range — the next [upstream-sync](upstream-sync.md)
    cycle reconciles it deliberately, and until then a Qwen3.5 change must check
    both anchors; (b) **no token or speed axis is claimable for
    `Qwen/Qwen3.8-2.4T-A95B`** — 2.4T bf16 is ~4.8 TB and the released FP8
    variant ~2.4 TB against GB10's 128 GB unified, so the run gate for that
    checkpoint is OWED (both rows are `PARTIAL`, never `DONE`, and
    the gap is recorded in [BENCHMARKS](../docs/BENCHMARKS.md) §Open gaps). The
    A smaller sibling DOES exist and this row previously claimed otherwise:
    `Qwen/Qwen3.8-27B` (55.6 GB bf16, 18 shards, revision
    `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) fits GB10 and **loads on current
    `main` with no code change** — evidenced by reaching the engine
    constructor's terminal log statement (`entrypoints/model_loader.cpp:1153`),
    which is emitted only after every weight binds and the hybrid KV geometry
    resolves. It does NOT close either text-only gate below: it declares
    `Qwen3_5ForConditionalGeneration`, not `Qwen3_5[Moe]ForCausalLM`. It is the
    already-gated Qwen3.6-27B shape retrained — `config.json` differs in exactly
    one key (`transformers_version`) and the safetensors tensor-name set is
    identical (1199 names, zero difference either direction). Its own
    token-exact gate is OWED and unrun.
    DENSE gate closes when a `Qwen3_5ForCausalLM` checkpoint that fits GB10
    appears; the MoE one, as recorded on 2026-08-12, did not — per (e) below, a
    fitting PUBLISHED MoE checkpoint would still have been refused at load.
    **SUPERSEDED 2026-08-15: (e) is implemented and oracle-gated, so a published
    MoE checkpoint no longer refuses and the MoE gate now needs only a fitting
    `Qwen3_5MoeForCausalLM` checkpoint, exactly like the DENSE one**; (c) **one behavior deliberately diverges from upstream, in the
    strict direction.** Upstream normalizes the two weight namespaces with
    `WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})`, so a
    checkpoint carrying backbone tensors under BOTH spellings LOADS upstream, its
    names silently rewritten; `ResolveQwen3_5BackbonePrefix`
    (`src/vllm/model_executor/models/qwen3_5_weights.cpp:597-615`) REFUSES it.
    The reason is that our loaders resolve the namespace ONCE and a mixed index
    is the one input where a per-lookup mapper binds half a model from each
    namespace and still appears to succeed — a wrong model that loads, which no
    name-mapping test can see. A refusal is strictly safer than upstream's
    rewrite and cannot make a checkpoint upstream rejects load here, so it is
    tracked rather than reconciled; it is revisited if upstream ever publishes a
    genuinely mixed index; (d) **the published config's dtype key is NOT
    consumed, tree-wide.** `Qwen/Qwen3.8-2.4T-A95B` is transformers 4.57.3 and
    spells the model dtype `dtype`; `hf_config.cpp:520-522` reads only the legacy
    `torch_dtype`, so `HfConfig::torch_dtype` is EMPTY on this document. Inert
    today — nothing in the tree reads that field — and deliberately NOT fixed on
    this row, because consuming `dtype` is a behavior change on every model and
    owes its own row, RED-first test and inertness proof. Pinned by an assertion
    in `tests/vllm/models/test_qwen3_8_text_only.cpp` so it cannot drift
    silently; (e) **the bf16 / 3-D-STACKED MoE ROUTED-EXPERT ARM IS NOT
    IMPLEMENTED, and this was recorded INVERTED until 2026-08-12.** The earlier
    text here, and on every other surface, said "MTP, quantized and GGUF arms for
    3.8 are NOT implemented and are recorded as owed". The QUANTIZED arm is the
    only one that IS implemented: `LoadQwen3_5Moe` routes every routed expert
    through `LoadMoeExpertsInto`
    (`src/vllm/model_executor/models/qwen3_5_weights.cpp:519-530`) into
    `LoadNvfp4Raw` (`:433-462`), which hard-requires per-expert
    `experts.<e>.<proj>.weight` = `U8`, `.weight_scale` = `F8_E4M3` and
    `.weight_scale_2`. There is **no stacked branch and no bf16 branch** —
    unlike `gemma4_weights.cpp:326`, which dispatches between layouts. Read live
    2026-08-12, the published indices have neither shape:
    `Qwen/Qwen3.8-2.4T-A95B` has 93x `mlp.experts.gate_up_proj` + 93x
    `.down_proj` (3-D stacked) and ZERO names matching `weight_scale` or
    `input_scale`, with `lm_head.weight` alone; `Qwen/Qwen3.6-35B-A3B` is the
    same under the VL prefix. Our gated 35B row reads the REQUANTIZED
    `nvidia/Qwen3.6-35B-A3B-NVFP4`, so this loader **has never read a published
    Qwen bf16 MoE repo**. What is owed is therefore the stacked/bf16 MoE expert
    arm (with the bf16 shared expert, the FP8-less attention tower and the bf16
    `lm_head` on that path), which needs its own row, spec, RED-first test and
    NVFP4 inertness proof. Until it exists such a checkpoint is REFUSED by name
    (`CheckMoeExpertLayoutSupported`, same file), per AGENTS.md §Shared seams.
    **The DENSE arm is NOT affected and the asymmetry is deliberate record:**
    `LoadQwen3_5Dense` routes BF16 vs FP8 vs NVFP4 per projection by tensor
    presence (`qwen3_5_dense_weights.cpp:355-361,473-504`) and routes the head by
    dtype (`LoadDenseLmHead` / `LoadLmHeadAnyDtype`, `:215-233,515-547`), so it
    may genuinely load a flat bf16 checkpoint. The whole deviation is discharged
    by a pin advance that includes `ad5d29db7`, at which point (a) and the
    anchors become ordinary mirrored behavior and only (c), (d) and (e) survive
    as tracked items. Scope and gates:
    [qwen3.8 text-only spec](specs/qwen38-text-only.md).

    **(e) IS CLOSED as of 2026-08-15, and closed with a token.**
    [#740](https://github.com/mudler/vllm.cpp/issues/740)
    ([spec](specs/moe-bf16-stacked-experts.md)) landed the 3-D stacked bf16
    routed-expert reader and [#864](https://github.com/mudler/vllm.cpp/issues/864)
    ([spec](specs/moe-bf16-tower-arms.md)) the bf16 GDN tower, attention tower,
    shared expert and `lm_head`, each dispatched by tensor presence the way
    `LoadQwen3_5Dense` already was. Both published indices now satisfy the load
    plan completely, so the sentence above — "this loader has never read a
    published Qwen bf16 MoE repo" — no longer holds. The binding token-exact
    greedy gate RAN on `Qwen/Qwen3.6-35B-A3B` bf16
    @`995ad96eacd98c81ed38be0c5b274b04031597b0` against the pinned oracle:
    **6/7 prompts STRICT 16/16**, the seventh one exact logit tie
    (`top2_gap_mnats = 0.0`) our on-device argmax breaks toward the higher id
    where `torch.argmax` takes the lower
    ([#910](https://github.com/mudler/vllm.cpp/issues/910)) — PASS under the
    ratified near-tie doctrine. Only the FIRST divergence per prompt is validly
    adjudicable, so the raw 108/112 position count is not a quality score.
    SACRED inertness 3 of 3 with real counts and byte-identical goldens (27B
    235/235, 35B 315/315, Coder 138/138, 688 assertions). NO throughput, latency
    or memory number exists for that checkpoint. What is still owed is unchanged
    by this: `Qwen/Qwen3.8-2.4T-A95B` remains unrunnable here on size, so (b)
    stands for both classes, and the MTP and GGUF arms for 3.8 stay owed.

18. **Beyond-pin port: `dots3_note` is anchored on vLLM `main`, not on the
    parity pin (2026-08-23, `MODEL-MM-dots3-note-dots3-note-for-causal-lm`,
    issue [#699](https://github.com/mudler/vllm.cpp/issues/699), W1).** Our
    pin is `555967922` (0.26.0.dev0, 2026-07-26), whose checkout carries only
    `dots_ocr.py` — verified, not assumed. `dots3_note` arrived afterwards in
    [vllm#51255](https://github.com/vllm-project/vllm/pull/51255) as the
    platform-split package `vllm/models/dots3_note/{common,nvidia}/`. Every
    `file:line` W1 cites was read at `origin/main` =
    `c205726108df54bb6fbf15b19e725a4a3add2b18`, and the anchors that decide
    correctness are named in `src/vllm/model_executor/models/dots3_note.h`.
    **W2 (2026-08-24) re-read its own anchors at `origin/main` =
    `185cada36b`**, which is 20 days of upstream later, and cites two: the
    tower prefixes in `nvidia/multimodal.py`'s `hf_to_vllm_mapper` (`:70-78`,
    the two prefixes at `:75-76`; `vision_encoder.` -> `visual.` and
    `audio_encoder.` -> `audio_tower.`. First written here as `:53-62`, which is
    W1's `c205726108` line and an INHERITED anchor rather than the re-read one
    this paragraph claims — corrected in place, review F4 on
    [#1847](https://github.com/mudler/vllm.cpp/pull/1847)) and
    the DSA indexer's rope slice in
    `vllm/model_executor/models/deepseek_v2.py` (`:805`, `:814`). Point (b)
    below is why it re-read rather than inheriting W1's revision.
    Same shape as deviations 16 and 17, taken for the same reason: at the pin
    the architecture does not exist, so there is nothing there to mirror. It
    is recorded here, and argued for in the commit that introduced it,
    because no checker enforces the anchor rule. Consequences, binding while
    this stands: (a) this row advances nothing in `555967922..main` and the
    next [upstream-sync](upstream-sync.md) cycle reconciles it deliberately;
    (b) upstream is STILL MOVING here — vllm#52172 landed 2026-08-13 — so a
    dots3 change re-reads its anchors rather than trusting a cited line;
    (c) **no token, throughput, latency or memory number is claimable for
    this model on any axis**, and that is a memory ceiling rather than a
    scheduling gap: `dots-studio/dots3-note-prev` is ~576 GB bf16 and its
    fp8 sibling ~290 GB against a 122 GiB host, so the oracle cannot run
    here at any published precision and there is no smaller checkpoint in
    the org. Spec [dots3-note](specs/dots3-note.md) §6.4 records the
    developer decision (option B) and `## Owed` carries the e2e gate.
    **W3 (2026-08-25) re-read its anchors again, at `origin/main` =
    `06ecec7a84`**, and point (b) is why:
    `git log 185cada36b..06ecec7a84 -- vllm/models/dots3_note/` is EMPTY, so the
    sources are byte-identical to W2's, but the line numbers W0 recorded in spec
    §2.2 are not — `_forward_note_mla` is `model.py:135-201` with the headwise
    gate at `:190-197`, against §2.2's `:246-262`. The re-derived anchors live in
    `src/vllm/model_executor/models/dots3_note_attn.h` beside the code that uses
    them. **W3 also writes maths for the first time on this row, and it is a
    PORTABLE HOST REFERENCE rather than a device forward** — recorded here
    because that is a deviation from how every other model in this tree ships a
    layer. `Dots3NoteModel::ForwardDevice` still refuses by name; nothing in
    `ModelRegistry::Forward` reaches the new code; the shared MLA seam
    (`mla::ForwardMlaAttentionBlock`) is NOT extended, because three of the four
    dots3 deltas sit inside it and adding optional branches to the SACRED
    DeepSeek-V2 path with no device forward to exercise them buys untested code
    and no gate. `deepseek_v4_dsa.{h,cpp}` is the in-tree precedent for the same
    call, taken for the same reason. Consequence (d): the layer is unreached
    debt, W4 owns the wiring and the seam extension, and spec `## Owed` names
    both.

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
   (contract-audited against target `702f481`; porting pin ADVANCED 2026-07-26 to `555967922` / vLLM 0.26.0.dev0 — see specs/pin-advance.md) is
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
