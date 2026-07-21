// The paged KV-cache STORAGE dtype, resolved in ONE place.
//
// No single upstream twin: vLLM carries the KV storage dtype on the cache
// config (`CacheConfig.cache_dtype` -> `AttentionSpec.dtype`, consumed by
// `kv_cache_interface.py:380-398`). We mirror the SHAPE of that contract — the
// KV cache SPEC is the single source of truth for the storage dtype, and the
// allocator sizes buffers from `spec->page_size_bytes()` — while keeping our
// own `VT_KV_CACHE_F32` same-binary A/B as the thing that picks the value.
//
// DEFAULT: bf16 (vLLM's bf16 flash_attn KV store — halves KV memory vs f32).
// `VT_KV_CACHE_F32=1` selects f32 for the A/B. Zero bytes are +0.0f in both.
//
// Every producer of an attention KV-cache spec (the model KV-cache factories
// and the runner tests) MUST build its spec with this dtype, because the runner
// now derives BOTH the allocation size and the cache view from the spec.
#ifndef VLLM_V1_KV_CACHE_DTYPE_H_
#define VLLM_V1_KV_CACHE_DTYPE_H_

#include <cstdlib>

#include "vt/dtype.h"

namespace vllm::v1 {

inline vt::DType ResolveKvCacheDType() {
  const char* kv_f32_env = std::getenv("VT_KV_CACHE_F32");
  return (kv_f32_env != nullptr && kv_f32_env[0] == '1') ? vt::DType::kF32
                                                         : vt::DType::kBF16;
}

}  // namespace vllm::v1

#endif  // VLLM_V1_KV_CACHE_DTYPE_H_
