ID: ISSUE-GH-1960
Title: **`SlidingWindowMLASpec` is a declared enumerator with no struct behind it, and `MLAAttentionSpec` carries none of the four DeepSeek-V4 fields, so 105 of V4's 167 cache entries cannot be sized at all.** W1 of [#1925](https://github.com/mudler/vllm.cpp/issues/1925). `KVCacheSpecKind::kSlidingWindowMla` is declared at `include/vllm/v1/kv_cache_interface.h:89` and the port's deferral list names the class as omitted (`:46-52`); it is the spec class of the SWA cache (43 entries, `vllm/v1/attention/backends/mla/sparse_swa.py:86-101`) and of both compressor-state populations (41 + 21, `vllm/models/deepseek_v4/compressor.py:188-200`). `MLAAttentionSpec` (`kv_cache_interface.h:242-261`) adds no fields over `FullAttentionSpec` where upstream carries `cache_dtype_str`, `alignment`, `compress_ratio` and `model_version` (`vllm/v1/kv_cache_interface.py:381-388`), so the compressed latent is sized `block_size` rows per page where upstream stores `block_size // compress_ratio`, and the 584-byte `fp8_ds_mla` token (`:396-405`) throws by name instead (`src/vllm/v1/kv_cache_interface.cpp:64-71`). `_apply_alignment_padding` (`:345-351`) has no twin, so no V4 page reaches its 576B/512B alignment. Pure allocation metadata: nothing constructs either spec outside tests, because publishing before W3 would allocate a silent subset (`src/vllm/v1/worker/gpu/runner.cpp:577-597` drops an unmatched group kind with no diagnostic).
Row: KV-DSV4-MULTICACHE
State: UNKNOWN
Kind: bug
GitHub: 1960
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:746`

### Frozen archive evidence

> | [#1960](https://github.com/mudler/vllm.cpp/issues/1960) | `KV-DSV4-MULTICACHE` | **`SlidingWindowMLASpec` is a declared enumerator with no struct behind it, and `MLAAttentionSpec` carries none of the four DeepSeek-V4 fields, so 105 of V4's 167 cache entries cannot be sized at all.** W1 of [#1925](https://github.com/mudler/vllm.cpp/issues/1925). `KVCacheSpecKind::kSlidingWindowMla` is declared at `include/vllm/v1/kv_cache_interface.h:89` and the port's deferral list names the class as omitted (`:46-52`); it is the spec class of the SWA cache (43 entries, `vllm/v1/attention/backends/mla/sparse_swa.py:86-101`) and of both compressor-state populations (41 + 21, `vllm/models/deepseek_v4/compressor.py:188-200`). `MLAAttentionSpec` (`kv_cache_interface.h:242-261`) adds no fields over `FullAttentionSpec` where upstream carries `cache_dtype_str`, `alignment`, `compress_ratio` and `model_version` (`vllm/v1/kv_cache_interface.py:381-388`), so the compressed latent is sized `block_size` rows per page where upstream stores `block_size // compress_ratio`, and the 584-byte `fp8_ds_mla` token (`:396-405`) throws by name instead (`src/vllm/v1/kv_cache_interface.cpp:64-71`). `_apply_alignment_padding` (`:345-351`) has no twin, so no V4 page reaches its 576B/512B alignment. Pure allocation metadata: nothing constructs either spec outside tests, because publishing before W3 would allocate a silent subset (`src/vllm/v1/worker/gpu/runner.cpp:577-597` drops an unmatched group kind with no diagnostic). | bug |

## Resolution

-
