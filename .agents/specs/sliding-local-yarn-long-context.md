# Sliding-window KV/attention, chunked-local attention, and long-context RoPE — joint spike

Rows covered: umbrella rows `KV-SLIDING-LOCAL-SPECS` and
`ATTN-ROPE-FAMILY`; claimable leaves `KV-SLIDING-WINDOW-SPEC`,
`KV-CHUNKED-LOCAL-SPEC`, `ATTN-SLIDING-WINDOW`, `ATTN-CHUNKED-LOCAL`,
`ATTN-YARN`, `ATTN-ROPE-LLAMA3`, `ATTN-ROPE-LONGROPE`, and
`ATTN-ROPE-DYNAMIC-NTK`. Roadmap block: `ROAD-V1-C5`. Upstream pin:
`/home/mudler/_git/vllm` @ `e24d1b24fe96`.

The umbrella rows describe the two execution blocks and are not implementation
claims. Implementers claim the eight leaves, in the dependency order below.
This spike performed source/config inspection only; it ran no GPU workload and
does not claim support.

## Scope

This block has two independent chains:

| Chain | In scope | Explicitly out of scope |
|---|---|---|
| Local KV + attention | `SlidingWindowSpec` and `ChunkedLocalAttentionSpec`; memory/admission bounds; spec registry and manager dispatch; cache-hit, skipped-block, eviction and prefix-cache policy; hybrid-manager-disabled conversion to `FullAttentionSpec` while preserving compute locality; causal decoder and symmetric encoder sliding windows; chunked-local virtual batches, local block tables, underlying-backend dispatch, and the pinned compatibility rules | R-SWA, sink attention, MLA/SWA-MLA, cross attention, KV connectors/offload, DCP/PCP, speculative EAGLE compatibility, and quantized KV breadth. Those stay with their own rows; tests that touch them are carried as tracked skips/dependencies |
| Long-context RoPE | common config/factory/cache/apply seam; standard YaRN; YaRN's `mrope_section` branch for 1-D text and 2-D T/H/W positions; Llama 3 banded scaling; Phi-3 LongRoPE short/long factors; dynamic-NTK `alpha` and `factor`; NeoX and GPT-J layout wherever upstream supports them; CPU and CUDA | linear/NTK/deepseek-yarn, proportional/Gemma4, XDRoPE, FoPE, dual-chunk RoPE, MM position construction and model architectures. The MRoPE apply operator is in scope, but multimodal preprocessing is not |

Required behavior is vLLM's behavior, including limitations:

- A decoder window of `W` becomes FlashAttention `(W - 1, 0)`: the current
  token plus at most `W - 1` previous tokens. Unequal Q/K lengths use the
  dependency's bottom-right alignment. Encoder-only attention symmetrizes to
  `(W - 1, W - 1)`.
- `SlidingWindowManager.get_num_skipped_tokens(n)` is
  `max(0, n - W + 1)`. Its admission cap is
  `ceil(min(W - 1 + max_num_batched_tokens, max_model_len) / block_size) + 1`;
  the final block accounts for an unaligned left edge.
- Chunked-local attention never crosses a fixed chunk boundary. Its skipped
  prefix is `floor(n / chunk_size) * chunk_size`, its admission cap is
  `ceil(min(chunk_size + max_num_batched_tokens, max_model_len) / block_size)`,
  and `chunk_size % block_size == 0`.
- With the hybrid manager disabled, local specs become full-allocation specs,
  but `sliding_window` / `attention_chunk_size` remain attached so compute is
  still local.
- LongRoPE selects the short or long cache once from configured
  `max_model_len`; it never changes formula when a sequence crosses the
  original context boundary, because doing so would invalidate cached keys.

### Gate-checkpoint audit

The actual cached configs on `dgx.casa` were inspected, not inferred from model
names:

| Checkpoint | `max_position_embeddings` | `rope_parameters` | SW/local |
|---|---:|---|---|
| NVIDIA Qwen3.6-35B-A3B-NVFP4 | 262144 | `rope_type=default`, `rope_theta=10000000`, `partial_rotary_factor=0.25`, `mrope_section=[11,11,10]`, `mrope_interleaved=true` | none |
| unsloth Qwen3.6-27B-NVFP4 | 262144 | same | none |

Therefore neither MVP gate checkpoint is a positive YaRN or local-attention
case. Text-only Qwen3.6 supplies equal T/H/W position streams, so its default
interleaved MRoPE reduces to the current 1-D result. Both checkpoints remain
mandatory regression gates; feature-positive oracle/model cases are listed
separately below.

## Upstream chain

All vLLM anchors below are at `e24d1b24fe96`.

### 2.1 Sliding-window and chunked-local execution

| # | Upstream source | Runtime role |
|---|---|---|
| K1 | `vllm/config/model.py:542-559,654-660,723-726,1232-1234` | Loads `attention_chunk_size` / `sliding_window`, normalizes zero, and applies the disable-window option |
| K2 | `vllm/model_executor/layers/attention/attention.py:204-400,583-608` | Resolves model/per-layer window, selects the backend, passes the window to the implementation, and emits `SlidingWindowSpec` |
| K3 | `vllm/model_executor/layers/attention/chunked_local_attention.py:30-128` | Selects an ordinary underlying backend, subclasses its metadata builder, disables cudagraph support, transforms metadata, and emits `ChunkedLocalAttentionSpec` |
| K4 | `vllm/v1/kv_cache_interface.py:205-307,480-586` | Full-allocation fallback fields, merge rules, local-spec page size, uniform grouping, memory sizing and single-source admission caps |
| K5 | `vllm/v1/core/single_type_kv_cache_manager.py:669-873` | Sliding-window right-to-left contiguous prefix hit, alignment/EAGLE handling, reachable masks, skipped-token recycling, and cascade disable |
| K6 | `vllm/v1/core/single_type_kv_cache_manager.py:876-1023` | Chunk-boundary prefix lookup/null blocks, restrictions, skipped-token recycling, and cascade disable |
| K7 | `vllm/v1/core/single_type_kv_cache_manager.py:1455-1525` | Spec registry dispatch and recycling-aware admission-cap plumbing |
| K8 | `vllm/v1/core/kv_cache_utils.py:1403-1496` | Hybrid-manager-disabled conversion to `FullAttentionSpec` while preserving the compute window/chunk |
| K9 | `vllm/v1/attention/backends/utils.py:225-420` | Splits each request into chunk-aligned virtual batches; builds local Q/K lengths and block-table gather indices; enforces chunk divisibility |
| K10 | `vllm/config/vllm.py:1494-1528`; `vllm/v1/worker/gpu_model_runner.py:2657-2677` | Mirrors incompatibilities: chunked-local + EAGLE disabled; chunked-local HMA opt-in because of measured latency; cascade disabled for either local type |
| K11 | `vllm/model_executor/models/llama4.py:242-268` | Representative consumer: `attention_chunk_size` selects `ChunkedLocalAttention` |

The CUDA dispatch is dynamic:

1. Non-MLA CUDA priorities are FlashAttention, FlashInfer, Triton, Flex, then
   TurboQuant on sm_121
   (`vllm/platforms/cuda.py:144-160,361-493`; selector validation is in
   `vllm/v1/attention/selector.py:54-143`).
2. `get_flash_attn_version` selects FA3 only on SM90 and FA4 only on SM100;
   sm_121 falls back to FA2
   (`vllm/v1/attention/backends/fa_utils.py:132-180`). The Python backend maps
   decoder `W` to `(W-1,0)`, encoder `W` to `(W-1,W-1)`, propagates it through
   metadata/AOT scheduling, and passes it to the varlen paged call
   (`vllm/v1/attention/backends/flash_attn.py:255-300,411-507,674-717,840-955`).
3. vLLM pins `vllm-project/flash-attention` commit
   `2c839c33742309ec41e620bf837495ec9926c56e`
   (`cmake/external_projects/vllm_flash_attn.cmake:13-58`). At that exact
   dependency pin, `csrc/flash_attn/flash_api.cpp:136-147,398-407,587-611`
   normalizes causal/local flags and disables the GQA head-swap fast path for a
   window; `csrc/flash_attn/src/flash_fwd_launch_template.h:54-99,101-130`
   selects the local specialization; `flash_fwd_kernel.h:80-94,267-300`
   restricts visited KV tiles and applies `Mask`; `mask.h:38-81,111-190`
   defines the exact inclusive, bottom-right-aligned token mask.
4. If FlashInfer is explicitly selected or higher-priority validation fails,
   vLLM maps the same window to `window_left=W-1`
   (`vllm/v1/attention/backends/flashinfer.py:1455-1500`). The installed oracle
   on dgx is `flashinfer-python 0.6.12`; its inspected source specializes on
   `window_left >= 0` and carries the value through plan/forward
   (`flashinfer/decode.py:1046-1152,1363-1368`;
   `flashinfer/prefill.py:2026-2053,2309-2317`), while the CuTe-DSL mask bounds
   live at `flashinfer/cute_dsl/attention/fusion/mask.py:18-54,88-177`.

Chunked-local attention does not require a distinct GPU mask kernel: K9 turns
chunks into independent causal virtual sequences and calls whichever ordinary
backend K3 selected. It does require exact metadata and block-table behavior.

### 2.2 Long-context RoPE execution

| # | Upstream source | Runtime role |
|---|---|---|
| R1 | `vllm/model_executor/layers/rotary_embedding/__init__.py:33-84` | Normalizes list values for the cache key, extracts theta/type/dimension, and memoizes one module per full configuration |
| R2 | same file `:155-171,200-230,243-284,315-335` | Dispatches Llama3, both dynamic modes, plain/MRoPE YaRN, and LongRoPE |
| R3 | `rotary_embedding/base.py:20-131,139-252,298-318` | Builds dtype-specific cos/sin cache and applies NeoX/GPT-J rotation through native or custom-op paths |
| R4 | `rotary_embedding/common.py:32-76`; `yarn_scaling_rope.py:10-84` | YaRN correction dimensions/ramp/mscale, interpolated/extrapolated inverse frequencies, and expanded cache |
| R5 | `rotary_embedding/mrope.py:190-261,263-340` | Reuses YaRN cache math for `mrope_section`; applies plain/interleaved T/H/W sections for 2-D positions and the common path for 1-D text |
| R6 | `rotary_embedding/llama3_rope.py:11-54` | Keeps high frequencies, scales low frequencies, and blends the middle wavelength band |
| R7 | `rotary_embedding/phi3_long_rope_scaled_rope.py:16-159` | Validates NeoX-only layout, derives mscale, builds short+long caches, and selects the cache half globally from runtime max length |
| R8 | `rotary_embedding/dynamic_ntk_scaling_rope.py:30-73`; `dynamic_ntk_alpha_rope.py:9-43` | Implements `factor`/trained-length and Hunyuan `alpha` base transforms |
| R9 | `vllm/_custom_ops.py:200-225`; `csrc/libtorch_stable/pos_encoding_kernels.cu:8-99,103-200`; `csrc/cpu/pos_encoding.cpp:332-365` | Runtime cache lookup and in-place Q/K rotation on CUDA and CPU |

For these one-dimensional scaling modes, FlashInfer is not the active RoPE
dependency: `rotary_embedding/base.py:37-48` explicitly leaves its RoPE path
disabled. The actual CUDA path is vLLM's custom op over the precomputed cache.
The 2-D MRoPE CUDA apply path is vLLM's own Triton implementation
(`mrope.py:14-187`); its formula/cache still comes from R4/R5.

Runtime verification remains mandatory at implementation time: nsys both the
pinned oracle and ours on identical feature-positive workloads, then compare
`cuda_gpu_kern_sum`. Source tells us the available dispatch; the trace proves
what sm_121 actually ran.

## Our baseline

| Area | Local anchor | Honest state |
|---|---|---|
| KV types | `include/vllm/v1/kv_cache_interface.h:36-52,70-176`; `src/vllm/v1/kv_cache_interface.cpp:1-61` | Kinds are enumerated and `FullAttentionSpec` already preserves fallback `sliding_window` / `attention_chunk_size`, but both concrete local specs, their sizing/grouping, and quant-aware page math are explicitly deferred |
| KV manager | `include/vllm/v1/core/single_type_kv_cache_manager.h:54-68,88-179`; `src/vllm/v1/core/single_type_kv_cache_manager.cpp:20-220` | Reusable admission/removal hooks exist; only Full/Mamba subclasses exist. No local prefix-hit/mask/recycling policy |
| Coordinator | `src/vllm/v1/core/kv_cache_coordinator.cpp:16-60` | Constructs Full/Mamba only; spec equality already retains fallback window/chunk fields |
| Allocation order | `src/vllm/v1/core/kv_cache_manager.cpp:178-205` | Already removes skipped blocks before allocation, which is the upstream invariant admission-cap sizing relies on |
| Attention metadata | `include/vllm/v1/attention/backend.h:61-118`; `src/vllm/v1/attention/backend.cpp:10-53` | Common lengths/table/slot metadata exists; no local virtual-batch transform or per-layer window |
| Attention API | `include/vt/ops.h:110-149,757-788` | `PagedAttentionArgs` carries scale/causal only; documented loop is every causal key from 0 through the query |
| CUDA attention | `src/vt/cuda/cuda_flash_attn_fa2.cu:205-245`; `src/vt/cuda/cuda_paged_attn.cu` | FA2 hard-codes `window_size_left=-1`; portable paged kernels have no lower key bound |
| HF config | `src/vllm/transformers_utils/hf_config.cpp:162-200`; `include/vllm/transformers_utils/hf_config.h:46-52` | Explicitly rejects `yarn`, `llama3`, `longrope`, and `dynamic`; stores only theta/partial dimension/max position |
| RoPE apply/cache | `src/vt/cpu/cpu_ops.cpp:308-357`; `src/vt/cuda/cuda_ops.cu:359-460`; fused consumer `cuda_ops.cu:463-590` | Plain NeoX computes default frequencies from `base` in each apply/cache kernel. The fused QK preamble can consume a cache, but cache generation has no scaled-frequency abstraction |
| RoPE tests | `tests/vt/test_ops_rope.cpp`; `tests/vt/test_cuda_ops.cpp:506`; `tests/parity/test_op_parity.cpp` cases `rope_f32_pos_short` / `rope_f32_pos_131k` | Default one-dimensional RoPE only; no config-dispatch or scaling-formula oracle cases |

No present code anchor may be upgraded to support evidence by this spike. In
particular, 262K default MRoPE on Qwen3.6 does not demonstrate any of the four
scaled families.

## Port map

Every implementation file carries its upstream path and pin in the header.
Shared-file ownership follows the work breakdown; downstream claims rebase on
the shared foundation rather than editing it concurrently.

### 4.1 KV and local attention

| Leaf | Upstream | Ours (new/changed) | Required shape |
|---|---|---|---|
| `KV-SLIDING-WINDOW-SPEC` | K4, K5, K7, K8 | `include/src/vllm/v1/kv_cache_interface.*`; `include/src/vllm/v1/core/single_type_kv_cache_manager.*`; narrow registry/coordinator/utility changes | Concrete spec and manager; exact page/admission math, grouping, right-to-left hit, alignment/EAGLE behavior, reachable mask, recycling, no-cascade; preserve full-allocation fallback |
| `KV-CHUNKED-LOCAL-SPEC` | K4, K6, K7, K8 | same mirrored KV files, with separate classes/tests | Concrete spec/manager; chunk-boundary null/hit rules, restrictions, admission cap, recycling, no-cascade; preserve fallback |
| `ATTN-SLIDING-WINDOW` | K1/K2 and the complete FA2/FlashInfer chain | introduce mirrored `include/src/vllm/model_executor/layers/attention/attention.*` if the generic layer seam is still absent; extend `include/vllm/v1/attention/backend.h`, `include/vt/ops.h`, CPU/CUDA paged-attention launchers and FA2 adapter | One optional `(left,right)` semantic value from config through metadata/dispatch; portable kernel lower bound; FA2 parameters; causal and symmetric encoder cases. Do not clone a Qwen-specific path |
| `ATTN-CHUNKED-LOCAL` | K3, K9-K11 | new mirrored `include/src/vllm/model_executor/layers/attention/chunked_local_attention.*` and `include/src/vllm/v1/attention/backends/utils.*`; model/config wiring only through the generic seam | Cached backend wrapper, cudagraph=NEVER, six virtual-batch shapes, async/device index upload equivalent, block-table update callback, underlying backend reuse |

The first attention implementation must expose a backend-neutral window in
`PagedAttentionArgs`; CUDA FA2, portable CUDA and CPU consume the same semantic
field. A backend-specific API hidden in the Qwen3.6 model is rejected.

### 4.2 RoPE family

| Leaf | Upstream | Ours (new/changed) | Required shape |
|---|---|---|---|
| `ATTN-YARN` (foundation lead) | R1-R5, R9 | extend `HfConfig` to retain the complete typed rope-parameter set; add mirrored `include/src/vllm/model_executor/layers/rotary_embedding/{base,common,yarn_scaling_rope,mrope}.*`; extend `vt::` RoPE to apply a supplied cache; add `tools/parity/dump_long_context.py` | Common cache key/factory and dtype cache; plain YaRN; `apply_yarn_scaling`, truncate/ramp/mscale options; 1-D and 2-D/interleaved MRoPE branch. Owns shared API/CMake lines |
| `ATTN-ROPE-LLAMA3` | R2, R6 | new mirrored `llama3_rope.*`, factory case, leaf tests/goldens only | Exact low/mid/high frequency branches, including equal low/high factor |
| `ATTN-ROPE-LONGROPE` | R2, R7 | new mirrored `phi3_long_rope_scaled_rope.*`, factory case, leaf tests/goldens only | Validate factor-array length and NeoX-only mode; build both caches; select one globally from max model length; explicit short/long mscale |
| `ATTN-ROPE-DYNAMIC-NTK` | R2, R8 | new mirrored `dynamic_ntk_scaling_rope.*` and `dynamic_ntk_alpha_rope.*`, factory cases, leaf tests/goldens only | Both mutually exclusive dispatch forms, missing-mode rejection, trained-position default, dimension guard |

The hot apply kernel remains a cache lookup/rotation. Formula code builds the
cache once; it must not add YaRN/Llama3/dynamic branches or transcendental work
to each token/head. LongRoPE is represented as a globally selected cache-row
base, not a per-position switch.

## Tests to port

Upstream tests are the executable specification. Tests blocked by a model,
backend, connector, TP or multimodal dependency are checked in with a named
`SKIPPED` reason and owning row; they are not silently dropped.

### 5.1 KV/spec/manager tests

| Upstream module/cases | Local target | Leaf / obligation |
|---|---|---|
| `tests/v1/test_kv_cache_spec_registry.py:174-306` | extend `tests/vllm/v1/test_kv_cache_interface.cpp` and `test_single_type_kv_cache_manager.cpp` | both KV leaves: built-in registration, custom/unregistered behavior, uniform-base mapping, hybrid conversion fields |
| `tests/v1/core/test_single_type_kv_cache_manager.py:54-536` | `tests/vllm/v1/test_single_type_kv_cache_manager.cpp` | Port all SW/chunked cases: possible prefix, skipped-block removal, allocation/evictable accounting, chunk allocation, admission-cap predictor property |
| `tests/v1/core/test_kv_cache_utils.py:1253-1496,1540-1635,1867-2088,2351-2475` | extend `tests/vllm/v1/test_kv_cache_utils.cpp` | merge/uniform/group/sizing/spec-kind/window extraction, padded page size and hybrid-unify behavior |
| `tests/v1/core/test_prefix_caching.py:733-985,2457-2750,2781-3338,3414-3503,3698-3909` | extend `tests/vllm/v1/test_prefix_caching.cpp` or the current coordinator/manager suites | hybrid SW combinations, EAGLE alignment, different block size, retention/replay tails, admission cap, free ordering and pure-SW policy. Cases needing unported EAGLE/retention are present but SKIPPED against their owner |
| `tests/v1/core/test_contiguous_kv_packing.py:19-86` and SW cases in `tests/v1/kv_offload/` / `tests/v1/kv_connector/` | matching future suites | Inventory-only dependency coverage: check in skips until contiguous packing/offload/connector rows exist; no connector code in C5 |

### 5.2 Attention tests

| Upstream module/cases | Local target | Leaf / obligation |
|---|---|---|
| `tests/v1/attention/test_chunked_local_attention.py:28-204` | new `tests/vllm/v1/attention/test_chunked_local_attention.cpp` | `ATTN-CHUNKED-LOCAL`: all six exact Q/K length and block-table vectors, including clipping, partial first/last chunks, chunk>sequence, and decode in the second chunk |
| `tests/v1/attention/test_attention_backends.py:745-867` | new/extended paged-attention backend suite | `ATTN-SLIDING-WINDOW`: causal prefill/decode/mixed plus symmetric encoder windows across supported local backends |
| `tests/kernels/attention/test_flash_attn.py:95-217` | `tests/vt/test_cuda_ops.cpp` or dedicated `test_paged_attention_window.cpp` | FA2 paged-window parity over query/KV lengths, block tables, GQA and boundary windows |
| `tests/kernels/attention/test_flashinfer.py:296-482` | same semantic suite, backend-selectable | FlashInfer fallback parity when available; otherwise a tracked backend skip, not support evidence |
| `tests/v1/e2e/general/test_correctness_sliding_window.py:19-78` | new feature e2e | `bigcode/starcoder2-3b` SW-only and `google/gemma-3-1b-it` hybrid, hybrid manager on/off, fresh and prefix-hit replay |
| `vllm/model_executor/models/llama4.py:242-268` behavior | synthetic layer e2e + eventual Llama4 model e2e | Chunked-local positive model gate; full Llama4 case SKIPPED until `MODEL-TEXT-llama4-llama4-for-causal-lm` lands |

### 5.3 RoPE tests

| Upstream module/cases | Local target | Leaf / obligation |
|---|---|---|
| `tests/kernels/core/test_pos_encoding.py:66-193` | extend `tests/vt/test_ops_rope.cpp`, `test_cuda_ops.cpp`, `tests/parity/test_op_parity.cpp` | Generic CPU/CUDA apply, Q/K optionality, cache reuse/dtype; dynamic factor=1 case |
| `tests/kernels/core/test_rotary_embedding.py:39-74` | same | Custom-op shape/dtype/layout checks |
| `tests/kernels/core/test_mrope.py:47-235` and `test_apply_rotary_emb.py:52-112` | new MRoPE parity suite | 1-D and 3×T positions, plain/interleaved sections, 11 and 8192 tokens, bf16 tolerances; TP=2 and model-driven cases tracked until their rows land |
| `tests/models/language/pooling/test_nomic_max_model_len.py:30-113` | config test + eventual e2e | YaRN override/max-length legality; pooling e2e waits on `MODEL-EMBED-bert-with-rope-nomic-bert-model` |
| `tests/test_config.py:437-587,645-650` | `tests/vllm/test_hf_config.cpp` | disable-window, dynamic override/max length, nested parameters and MRoPE detection |
| No direct per-class numerical tests at the pin | new oracle-generated goldens from `tools/parity/dump_long_context.py` | Required gap-fill: plain YaRN (truncate on/off, scaling on/off), MRoPE YaRN, Llama3 low/mid/high bands, LongRoPE short/long selections and mscale overrides, dynamic alpha/factor, f32+bf16, NeoX+GPT-J where accepted, partial rotary dim, positions `{0,1,original-1,original,max-1}` |

The golden generator imports the pinned vLLM classes directly and records the
pin, complete parameters, input seed/dtype/layout, cos/sin cache, and Q/K
outputs in each manifest. The absence of direct upstream class tests is not
permission to test only our formula.

## Gates

No row reaches `DONE` from unit tests alone.

| Gate | Command / workload | Pass condition |
|---|---|---|
| G1 CPU build + ported unit tier | `cmake -S . -B build-c5-cpu -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF && cmake --build build-c5-cpu -j20`; then `ctest --test-dir build-c5-cpu --output-on-failure` | All applicable upstream cases green; every skip names the dependency row |
| G2 KV properties | deterministic randomized sequences spanning `W/chunk/block` boundaries, prefix hits/misses, preemption/free/reallocate and `max_num_batched_tokens` | Predicted allocation equals real non-null held blocks; held blocks never exceed the spec cap; no live block freed; manager and full-allocation fallback produce identical tokens |
| G3 RoPE oracle parity | Generate goldens with `~/venvs/vllm-oracle/bin/python tools/parity/dump_long_context.py ...`; run CPU and CUDA consumers | f32 `rtol=atol=1e-5`; bf16 `rtol=1.6e-2, atol=1e-2` (upstream MRoPE tolerance); config/factory class and error parity exact. CPU/CUDA use the same selected cache |
| G4 attention operator parity | On dgx under `flock /tmp/gpu`, run dedicated local-window tests at Q lengths `{1,5,129}`, KV lengths straddling block/window boundaries, W `{1,block-1,block,block+1,4096}`, prefill/decode/mixed, GQA, causal and symmetric encoder | Output/LSE within upstream test tolerance; no key outside the exact mask changes output; FA2 and portable paths agree. FlashInfer selected arm agrees when installed |
| G5 chunked-local metadata/operator | Six ported vectors plus random chunk-aligned reference masks and an underlying-backend run | Q token count preserved; every virtual K/Q length ≤ chunk; block tables exact; outputs equal dense reference with the chunk-boundary mask; cudagraph use is rejected |
| G6 feature-positive e2e | SW: StarCoder2-3B and Gemma3-1B test above, manager on/off. Local: Llama4 or the smallest upstream-compatible checkpoint once its model row lands. RoPE: Llama-3.1 (Llama3), Phi-3-mini-128k (LongRoPE), Nomic/gpt-oss (YaRN), Hunyuan or a pinned config override (dynamic) | Greedy token-for-token vs pinned vLLM on identical prompts/seeds, including a position beyond original context; prefix replay exact. A missing model row leaves the leaf `GATING`, with the test committed SKIPPED |
| G7 runtime dispatch trace | One steady-state nsys capture of vLLM and ours per local-attention model, same workload; `nsys stats --report cuda_gpu_kern_sum` | Trace proves the actual oracle backend/kernel (expected default FA2 on sm_121 unless validation changes it), ours uses the intended local path, and no full-window fallback is hidden |
| G8 performance/memory | Same-binary pre/post plus fresh `vllm bench throughput` / serving denominator on the G6 model at its large-concurrency point, ≥2-3 uncontended repetitions; record total/output throughput, req/s, TTFT, TPOT/ITL and peak memory | Every standing axis meets/beats vLLM; SW/local optimized-manager memory is bounded by the derived cap and ≤ vLLM. Hybrid-disabled arm may allocate full KV but must match vLLM's same arm |
| G9 global regression | Existing 35B and 27B greedy suites plus their exact benchmark recipes, both configs unchanged | 16/16 token exact and no throughput/latency/memory axis below the fresh vLLM denominator. These are regression gates, not feature-positive evidence |

GPU gates use one `flock /tmp/gpu` for each complete A/B or trace series and
run only on an otherwise idle GB10. Results without the lock for the whole
series are void. The implementing change records commands, commit, config,
seed, model revision, output and ratio in the parity ledger.

## Dependencies

| Leaf | Hard predecessor / external dependency | Reason |
|---|---|---|
| `KV-SLIDING-WINDOW-SPEC` | existing block-pool/manager/coordinator baselines | First claimable KV leaf; share-file claims must name its exact class/test slices |
| `ATTN-SLIDING-WINDOW` | semantic `PagedAttentionArgs` work may begin in parallel; e2e waits on `KV-SLIDING-WINDOW-SPEC` | Compute and memory chains meet only at model execution |
| `KV-CHUNKED-LOCAL-SPEC` | existing KV baseline | Independent of SW after shared registry merge |
| `ATTN-CHUNKED-LOCAL` | `KV-CHUNKED-LOCAL-SPEC` for e2e; metadata utility itself can start independently | Needs spec emission/manager for full execution |
| `ATTN-YARN` | none beyond current default RoPE | Foundation lead: common typed config/factory/cache/apply API |
| three other RoPE leaves | `ATTN-YARN` common API merged | Prevents four agents from concurrently inventing/editing the same factory/cache |
| SW positive e2e | `MODEL-HFALIAS-transformers-transformers-for-causal-lm` or `MODEL-TEXT-gemma3-gemma3-for-causal-lm` | Neither architecture is implemented today |
| Chunked-local positive e2e | `MODEL-TEXT-llama4-llama4-for-causal-lm` | Representative upstream consumer |
| YaRN positive e2e | `MODEL-EMBED-bert-with-rope-nomic-bert-model` or `MODEL-TEXT-gpt-oss-gpt-oss-for-causal-lm` | Operator goldens can land first; support closure needs a consumer |
| Llama3 / LongRoPE / dynamic positive e2e | respectively `MODEL-TEXT-llama-llama-for-causal-lm`, `MODEL-TEXT-phi3-phi3-for-causal-lm`, and `MODEL-TEXT-hunyuan-v1-hun-yuan-dense-v1-for-causal-lm` (or MoE sibling) | Model-family dependencies remain honest |
| YaRN MRoPE 2-D e2e | e.g. `MODEL-MM-qwen2-vl-qwen2-vlfor-conditional-generation` / `MODEL-MM-qwen3-vl-qwen3-vlfor-conditional-generation` | C5 owns cache/apply semantics, not MM position construction |

Hardware/data: CPU CI for G1-G3; `dgx.casa` GB10/sm_121 for G3-G9. New model
assets are downloaded only by the implementation owner after checking license,
disk and exact revision; this spike downloaded none. No new runtime library is
required: ours extends the vendored FA2 adapter/portable kernels and does not
add PyTorch, Python or FlashInfer as a dependency.

## Work breakdown

Claim the leaves, not the two umbrellas:

| W | Row | Non-overlapping deliverable | Primary ownership | Close gates |
|---|---|---|---|---|
| W1 | `KV-SLIDING-WINDOW-SPEC` | Spec, registry dispatch, manager, hybrid conversion, manager/prefix tests | SW classes/cases in KV interface/manager/coordinator/utils | G1-G2 |
| W2 | `ATTN-SLIDING-WINDOW` | Generic window plumbing, CPU/portable CUDA lower bound, FA2 adapter, backend/config tests | attention layer/backend/`vt::PagedAttention` window fields and kernels | G1, G4, G6-G9 |
| W3 | `KV-CHUNKED-LOCAL-SPEC` | Spec/manager/registry/hybrid conversion and chunk policy tests | Chunked-local classes/cases in shared KV files, rebased after W1 registry shape | G1-G2 |
| W4 | `ATTN-CHUNKED-LOCAL` | Backend wrapper, virtual batches/block tables, model seam and tests | new chunked-local attention/backend-utils files | G1, G5-G9 |
| W5 | `ATTN-YARN` | Common typed config/factory/cache/apply foundation, plain + MRoPE YaRN, oracle tool and goldens | HF config; new base/common/yarn/mrope files; shared `vt::` cache API/CMake lines | G1, G3, G6, G9 |
| W6 | `ATTN-ROPE-LLAMA3` | Formula builder/factory case and low/mid/high goldens | `llama3_rope.*` and leaf tests only | G1, G3, G6, G9 |
| W7 | `ATTN-ROPE-LONGROPE` | Global short/long cache selection, validation and goldens | `phi3_long_rope_scaled_rope.*` and leaf tests only | G1, G3, G6, G9 |
| W8 | `ATTN-ROPE-DYNAMIC-NTK` | Alpha/factor builders, dispatch/error cases and goldens | two dynamic files and leaf tests only | G1, G3, G6, G9 |

Order: W1 and W3 may start separately with a coordination-designated shared
registry lead; W2 follows W1 for e2e; W4 follows W3 for e2e. W5 lands the RoPE
foundation, then W6-W8 can run in parallel. Model-dependent G6/G8 closures are
handoffs to the named model rows; until then code-bearing leaves remain
`GATING`/`PARTIAL`, never `DONE`.

## Risks/decisions

| # | Risk / decision | Resolution |
|---|---|---|
| D1 | Historical matrix text said Qwen3.6 long context used YaRN | Corrected: both gate configs use default interleaved MRoPE. Keep them as regression gates and require positive family cases |
| D2 | Window off-by-one and unequal Q/K alignment | Store semantic `W`, map once to `(W-1,0)` / symmetric encoder, and port the dependency mask boundary vectors. Do not reinterpret W in each kernel |
| D3 | Reading vLLM source can misidentify the runtime kernel | nsys oracle and ours; source inspection includes the exact pinned FA dependency and installed FlashInfer fallback |
| D4 | Hybrid-manager-disabled mode can look correct while losing memory savings | Separate compute-correctness from memory-policy gates; compare optimized and fallback arms to matching vLLM arms |
| D5 | Chunked-local HMA has an upstream latency regression | Mirror default disable and opt-in environment behavior. Do not advertise memory optimization in the default arm until G8 says otherwise |
| D6 | EAGLE/DCP/PCP/cascade combinations | Mirror pinned assertions/disable rules; port their tests as tracked skips. Supporting them is a separate row, not a local invention |
| D7 | Per-token formula branches would regress every model | Precompute typed caches once and retain the hot lookup/rotate kernel. G8 covers default-RoPE regression |
| D8 | LongRoPE threshold switching corrupts KV consistency | Select short vs long globally from configured max length exactly as R7; test two separate configs/processes |
| D9 | Cache construction precision can differ from vLLM | Oracle manifests include cache tensors and dtype. Match the pinned vLLM result/tolerance, not an independently simplified formula |
| D10 | YaRN `mrope_section` makes one leaf larger | It is part of the factory's YaRN branch and remains in `ATTN-YARN`; 1-D and 2-D apply tests are mandatory. MM preprocessing/model e2e is a named dependency |
| D11 | Pinned vLLM has no direct numerical tests for most scaling classes | The required oracle-golden matrix closes that executable-spec gap and cites the exact upstream classes; no family closes on config parsing alone |
| D12 | Model prerequisites are mostly inventoried | Operator/KV code can land with upstream tests, but the leaf cannot be `DONE` until its feature-positive e2e is unskipped and gated |

No genuine product choice remains. Public behavior, dispatch preference,
formulae, limitations, and fallback modes are all defined by the pinned vLLM
chain.
