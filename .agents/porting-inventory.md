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
| Busy loop + input/output queue split (in-proc analog of ZMQ boundary) | `v1/engine/core.py`, `core_client.py` | T0 |
| InputProcessor (validate, tokenize, build EngineCoreRequest) | `v1/engine/input_processor.py` | T0 ✅ `73a9509` (text path; runs PostInit/Verify + max_tokens default + eos/stop wiring; mm/lora/embeds/pooling deferred) |
| OutputProcessor + RequestState + incremental Detokenizer | `v1/engine/output_processor.py`, `detokenizer.py` | T0 ✅ `c7ba3a5` (process_outputs: detokenize + string-stop + reqs_to_abort feedback; streaming DELTA/CUMULATIVE/FINAL_ONLY; logprobs/pooling deferred) |
| AsyncLLM-equivalent streaming API + sync LLM API | `v1/engine/async_llm.py`, `llm_engine.py` | T0 |
| Unified scheduler: token-budget, **no prefill/decode distinction** | `v1/core/sched/scheduler.py` | T0 ✅ `4f12158` (schedule() running-first + chunked prefill + FCFS preemption; update_from_output + check_stop; priority/spec/structured/async deferred behind 1:1 stubs) |
| Chunked prefill (`enable_chunked_prefill`, on by default) | `config/scheduler.py` | T0 ✅ `4f12158` |
| Budgets: `max_num_batched_tokens`, `max_num_seqs`, `max_num_scheduled_tokens` | `config/scheduler.py` | T0 ✅ `2f0ea69` |
| Preemption (FCFS tail pop, recompute) + `SchedulerOutput` new/cached diff protocol | `v1/core/sched/{scheduler,output}.py` | T0 ✅ `4f12158` (`c65e650` SchedulerOutput/NewRequestData/CachedRequestData in the MRV2 shape — prefill_token_ids + resumed-as-new fold) |
| FCFS request queue | `v1/core/sched/request_queue.py` | T0 ✅ `2f0ea69` |
| Priority scheduling (`policy="priority"`) | same | T1 |
| Partial-prefill concurrency (`max_num_partial_prefills`, long-prefill threshold/limits) | `config/scheduler.py` | T1 |
| Async scheduling (overlap schedule with execution) | `v1/core/sched/async_scheduler.py` | T1 (perf lever for the gate if needed → may promote to T0) |
| `scheduler_reserve_full_isl`, pluggable `scheduler_cls`, `stream_interval` | `config/scheduler.py` | T1 |
| Cascade attention (shared-prefix batch attention) | `config/model.py::disable_cascade_attn` | T2 |
| DBO / ubatch overlap | `config/parallel.py::enable_dbo` | T2 |
| Spec-decode scheduling hooks (`spec_token_ids`, lookahead slots) | `v1/core/sched/scheduler.py` | T2 (with §6) |

## 2. KV cache management (`vllm/v1/core/`)

| Item | Upstream | Tier |
|---|---|---|
| ✅ `5ee2301` BlockPool: free list, ref counts, LRU eviction, `BlockHash→block` map | `v1/core/block_pool.py` | T0 |
| ✅ `75caf38` KVCacheManager: `allocate_slots` (accounting/watermark/OOM→nullopt/admission-cap), `get_computed_blocks`, `free`; hybrid prefill | `v1/core/kv_cache_manager.py` | T0 |
| ✅ `5ee2301` Prefix caching (parent-chained block hashing; sha256_cbor byte-exact vs upstream — default deviates to sha256_cbor since upstream sha256/pickle default isn't cross-language reproducible; align path hash_block_size≠block_size + cache_partial_block deferred behind 1:1 stubs) | `v1/core/kv_cache_utils.py`, `config/cache.py` | T0 |
| ✅ `75caf38` **Hybrid KV coordinator + per-group managers** (full-attn group + GDN/mamba state group — the gate models are hybrid): SingleTypeKVCacheManager (FullAttention left→right multi-block prefix + Mamba right→left single recurrent state), HybridKVCacheCoordinator cross-group MIN-intersection prefix hit. Mamba `align` mode deferred behind a 1:1 stub | `v1/core/kv_cache_coordinator.py`, `single_type_kv_cache_manager.py` | **T0** |
| ✅ `75caf38` Specs: `FullAttentionSpec`, `MambaSpec` (GDN state), `KVCacheGroupSpec`, `KVCacheConfig` — `page_size_bytes` byte-exact vs upstream. `SlidingWindowSpec`/`ChunkedLocalAttentionSpec`/`MLAAttentionSpec` deferred behind 1:1 stubs | `v1/kv_cache_interface.py` | T0 |
| Mamba/GDN prefix-cache retention (`mamba_cache_mode` align, PR #45845) | `v1/core/` | T1 |
| `SlidingWindowSpec`, `ChunkedLocalAttentionSpec` | `v1/kv_cache_interface.py` | T1 |
| `MLAAttentionSpec` (+ latent KV) | same | T2 (with DeepSeek-family) |
| `CrossAttentionSpec`, `EncoderOnlyAttentionSpec` | same | T2 |
| KV quantization: fp8 KV cache (`cache_dtype=fp8*`) | `config/cache.py`, `layers/quantization/kv_cache.py` | T1 |
| KV quantization: nvfp4 / per-token-head / turboquant modes | `config/cache.py` | T2 |
| KV offload (CPU tiering, LRU/ARC policies) | `v1/kv_offload/` | T2 |
| KV connectors (Nixl/LMCache/Mooncake/PD-disaggregation) | `distributed/kv_transfer/` | T2 |
| KV events (ZMQ publish of block create/evict) | `config/kv_events.py` | T2 |
| Sizing: `gpu_memory_utilization`, `num_gpu_blocks_override`, `kv_cache_memory_bytes` | `config/cache.py` | T0 |

## 3. Model runner / worker (`vllm/v1/worker/`)

We port the **Model Runner V2** (`v1/worker/gpu/` package — `model_runner.py`,
`input_batch.py`, `block_table.py`, `attn_utils.py`, `states.py`), not the legacy
`gpu_model_runner.py`: upstream defaults new architectures to V2 and new features
(e.g. DSpark) are V2-only. This is where upstream is going; PR-portability demands
we mirror it.

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
| `reshape_and_cache` (write K/V into paged NHD cache at slot_mapping) | `csrc/.../cache_kernels.cu::reshape_and_cache_flash` | T0 ✅ `e231196`→`7de4f0c` (vt::ReshapeAndCache, stride-based NHD write CPU+CUDA; CUDA parity dgx-pending) |
| Paged attention for full-attn layers on sm_121 (bf16, GQA 16/2, partial RoPE) — FlashInfer-class performance is the bar; strategy in §9 | ref: `v1/attention/backends/{flashinfer,triton_attn,flash_attn}.py` | T0 🚧 `c244592` (vt::PagedAttention correctness-grade CPU+CUDA, anchored to M0.9 dense; FlashInfer-class perf + CUDA graphs = M2.4; CUDA parity dgx-pending) |
| **GDN backend**: metadata segmentation (prefill/decode/spec) | `v1/attention/backends/gdn_attn.py` | T0 ✅ `370ddaf` (GDNAttentionMetadata decode/prefill split + has_initial_state mask + prefill rebasing; spec segments + align col-gather deferred; GDN-state zeroing = caller obligation, see state.md) |
| GDN chunked-scan prefill kernel (chunk gated delta rule) | `layers/fla/ops/chunk.py` (Triton ref), `flashinfer.gdn_prefill` (Blackwell) | T0 🚧 `ead59d6` (correctness-grade sequential; chunked perf kernel M2.3) |
| GDN fused decode recurrence (sigmoid-gating delta rule update) | `layers/fla/ops/{fused_sigmoid_gating,fused_recurrent}.py` | T0 ✅ `ead59d6` (correctness-grade) |
| GDN post-conv prep (q,k,v,g,beta + L2 norm) + causal conv1d fn/update | `layers/fla/ops/fused_gdn_prefill_post_conv.py`, `layers/mamba/ops/causal_conv1d.py` | T0 ✅ `ead59d6` (l2norm + conv1d only; packed q/k/v split + g/beta prep is M0.9 — gdn-semantics §6) |
| Sliding-window attention | backends | T1 |
| MLA backends (latent KV, MQA decode) | `v1/attention/backends/mla/` | T2 |
| Mamba1/Mamba2/short-conv/linear backends | `v1/attention/backends/{mamba*,short_conv,linear}_attn.py` | T2 |
| Encoder / cross-attention | backends | T2 |
| CPU attention backend (scalar/SIMD reference, for CI parity) | `v1/attention/backends/cpu_attn.py`, `csrc/cpu/` | T0 (correctness-grade only) |

## 5. Model architectures (`vllm/model_executor/models/`)

292 registered upstream. Registry design (`registry.py`: HF `architectures[]` →
implementation) is ported at T0.

| Family | Marquee members | Needs | Tier |
|---|---|---|---|
| **Qwen3.5/3.6 hybrid (incl. MoE)** | `Qwen3_5ForConditionalGeneration` (27B dense-hybrid, **VL multimodal** wrapper — vision_config present), `Qwen3_5MoeForConditionalGeneration` / `qwen35moe` (35B-A3B) | GDN layers ×3 : 1 gated full-attn (qk-norm, partial RoPE 64d, output gate), MoE 256e top-8 + shared expert (35B) / **dense SwiGLU MLP** (27B), GemmaRMSNorm-style `(1+w)` — ✅ `25326fc` (35B forward correctness-grade, **safetensors**; 16/16 greedy on GB10 = M0 exit; GGUF k-quant load M0.10). **27B (co-equal gate):** CPU-first scaffolding started — arch/quant surveyed (`.agents/qwen27b-w4a4-notes.md`: dense hybrid, W4A4, **and a VL wrapper — text path first, ViT deferred**), CPU W4A4 emulation reference + skipping greedy-parity gate landed; dense loader + single-seq `ForwardDense` + batched PAGED `Qwen3_5DenseModel::Forward` (paged==dense CPU-anchored) + `GPUModelRunner` dense route all landed CPU-green; W4A4 GPU GEMM + oracle golden pending (GPU-gated). serving M1–M3 | **T0 (the gate)** |
| Dense decoders | Llama 3.x, Qwen2/3, Mistral, Gemma 2/3, Phi | GQA + RoPE + SwiGLU + RMSNorm (subset of T0 layer set) | T1 |
| MoE decoders | Mixtral, Qwen3-MoE (30B-A3B), GLM-4-MoE, OLMoE | FusedMoE 🚧 `65788b3` (correctness-grade eager; grouped-GEMM perf M2.2) | T1 |
| Qwen3-Next | `Qwen3NextForCausalLM` | same stack, interleaved-GQA weight layout | T1 |
| Hybrid others | Jamba, Bamba, NemotronH, FalconH1, Zamba2, LFM2, Kimi-Linear | mamba1/2 kernels | T2 |
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
| **NVFP4 (modelopt_fp4 + compressed-tensors w4a4_nvfp4)** — gate checkpoints; W4A4 MMA on sm_121 for MoE grouped GEMM + dense | `quantization/modelopt.py`, `compressed_tensors/` | **T0** 🚧 `65788b3`/`bc0a8d7` (35B modelopt W4A16: dequant-to-bf16 + tensor-core GEMM). **27B compressed-tensors NVFP4 W4A4** (the dense gate): CPU dequant + activation-quant emulation reference `nvfp4_emulation.h` ✅ (mirrors `nvfp4_emulation_utils.py`; weight-dequant reciprocal-global fix + the per-token activation fp4 quant; `test_ct_nvfp4_emulation` 81 assert green). Forward + W4A4 GPU GEMM pending — see `.agents/qwen27b-w4a4-notes.md` §5 |
| **GGUF quants** — container + dequant/matmul for types in our benchmark files (Q4_K/Q5_K/Q6_K/Q8_0/F16/F32 + NVFP4 GGUF extension types from the APEX/killgate tooling) | `quantization/gguf.py`, `model_loader/gguf_loader.py` | **T0 (gate)** ✅ `1a4db5c` (dequant F32/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/Q3_K byte-exact vs ggml + the qwen35moe GGUF loader [name mapping, convert-transform inversions, expert split, HfConfig] + model_loader `.gguf` routing; IQ2_S/IQ4_XS i-quants deferred; real-APEX greedy parity dgx-pending) |
| fp8 (W8A8, e4m3) | `quantization/fp8.py` | T1 |
| MXFP4 / MXFP8 | `quantization/mxfp4.py`, modelopt | T1 |
| AWQ/GPTQ (+Marlin), compressed-tensors int schemes | various | T2 |
| bitsandbytes, torchao, quark, INC, … | various | T3 (niche/other-vendor) |

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
`bad_words` promoted to T0 — ported at M1.7 `aac5138`.)

**Spec decode** (`v1/spec_decode/`): all T2, ordered — MTP (Qwen3.6 ships MTP
weights; biggest single-model win) → ngram → EAGLE3 → dspark/suffix. Rejection
sampler + `use_v2_model_runner`-style padded drafting come with it.

## 7. Serving surface (`vllm/entrypoints/`)

| Item | Upstream | Tier |
|---|---|---|
| `/v1/completions`, `/v1/chat/completions` (SSE streaming), `/v1/models`, `/health`, `/version`, `/metrics` | `entrypoints/openai/` | T0 ✅ `23d9f2c` (protocol+serving+cpp-httplib server+SanitizeUtf8+engine-mutex+examples/server; `/metrics` deferred; logprobs payload M3.x) |
| Chat templating (Jinja subset needed by Qwen/Llama templates; engine: minja-style) | `entrypoints/chat_utils.py` | T0 ✅ `a99a65e` (original minja-subset Jinja engine, Qwen3.6 byte-identical to jinja2; tool/think branches M3.3/M3.4) |
| **Tool/function calling** (user-mandated MVP): `tools`/`tool_choice` in chat API, auto-tool-choice, streaming tool-call deltas, tool-call parsers for the gate models (Qwen family + Hermes format first; others T1), grammar-forced JSON for `tool_choice=required`/named | `entrypoints/openai/tool_parsers/`, `chat_completion/` | T0 ✅ `18e3efb` (Hermes/Qwen3 parsers + streaming; **tool_choice=auto RELAXED via a native LAZY grammar matcher behind vLLM's STRUCTURAL_TAG seam** — free text until `<tool_call>`, then constrain [xgrammar TriggeredTag / llama.cpp lazy-grammar equivalent]; required/named forced; `<tool_call>` added-token matchability fixed for the real gate model; Coder-XML/Mistral/pythonic parsers deferred) |
| `/tokenize`, `/detokenize`, `/ready`, `/ping`, `/server_info`, `/reset_prefix_cache` | various routers | T1 |
| `/v1/embeddings`, `/pooling`, `/score`, `/rerank` | pooling routers | T2 |
| `/v1/responses`, `/v1/messages` (Anthropic-style), audio endpoints | responses/messages routers | T2 |
| Sleep/pause/resume, LoRA load/unload, profiling, RL weight-update endpoints | various | T2–T3 |
| CLI: `serve`, `bench {latency,throughput,serve}` (bench is how we measure the gate), `chat`, `complete` | `entrypoints/cli/` | T0 (`serve`+`bench`), T1 rest |
| Offline API: `generate`, `chat`, streaming enqueue; `get_metrics` | `entrypoints/llm.py` | T0 |
| Prometheus metric names **1:1** (`vllm:num_requests_running`, `vllm:time_to_first_token_seconds`, `vllm:kv_cache_usage_perc`, …) | `v1/metrics/` | T0 (core set), T1 (full set) |
| OTLP tracing | `config/observability.py` | T2 |

**Library packaging (llama.cpp-style) — T0 ✅ `0b252ec`:** core built as `libvllm`
(static + shared; the shared lib's linker version-script exports ONLY the 11
`vllm_*` C ABI symbols — nm-verified + ctest-enforced) with a stable **C API**
(`include/vllm.h`: `vllm_engine_load`/`free`, `vllm_complete` [blocking],
`vllm_complete_stream` + `vllm_token_callback` [streaming/early-stop],
`vllm_string_free`/`vllm_completion_free`, `vllm_last_error`, `vllm_version`;
opaque handles, no-throw-across-ABI, thread-local error, unique per-call request
ids — cgo/purego-friendly for LocalAI; ASan-clean + dlopen-smoke-proven). The
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
   Runner deviation: `GPUModelRunner` now carries EITHER arch via a
   `{moe,dense}_weights_` pointer pair (was a single `Qwen3_5MoeWeights&`
   reference) + a `Qwen3_5DenseWeights` constructor overload; `execute_model`
   dispatches to the dense forward when `dense_weights_` is set. `initialize_kv_cache`
   is unchanged (config-driven; same hybrid backbone). Full LoadedEngine dense
   dispatch is now WIRED (CPU): `LoadedEngine` carries the SAME `{moe,dense}_weights_`
   pair (`std::optional<Qwen3_5MoeWeights>` / `std::optional<Qwen3_5DenseWeights>`,
   was a single `Qwen3_5MoeWeights weights_`) + a `Qwen3_5DenseWeights` constructor
   overload driving the runner's dense overload; `FromModelDir` arch-selects on
   `LoadedEngine::IsDenseArch` (`num_experts==0` → `LoadQwen3_5Dense`, else
   `LoadQwen3_5Moe`). Everything between (Executor/EngineCore/processors) was
   already arch-agnostic (touches the runner only via `ModelRunnerBase`), so NO
   change was needed there — the MoE-typing was confined to the runner + the
   loader. CPU-proven by `test_loaded_engine_dense.cpp` (IsDenseArch dispatch +
   dense stack generates deterministically through the full LLMEngine loop). The
   remaining 27B steps are all GPU-gated (oracle golden → W4A4 GEMM 6a → flip
   `kW4A4ForwardReady`, qwen27b-w4a4-notes.md §5 steps 5-7).
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
5. **The gate benchmark**: `bench serve`-equivalent (request-rate sweeps,
   concurrency ladders; prefill-heavy, decode-heavy, mixed) vs vLLM on the same
   GB10 — TTFT, ITL, prompt+gen tokens/s. Regression-tracked per commit.
