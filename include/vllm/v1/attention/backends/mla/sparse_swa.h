// Ported from: vllm/v1/attention/backends/mla/sparse_swa.py @ pin e126687a9a
// (`DeepseekSparseSWABackend`, `:121-148`).
//
// WHAT THIS FILE IS FOR. DeepSeek-V4 publishes its sparse sliding-window cache
// at 64 tokens per block (`sparse_swa.py:82-87`, and the comment there derives
// the 64 from the C4A KV block shape the two caches SHARE a physical tensor
// with). No dense backend in this tree accepts 64-only geometry as its own, and
// asking one is what this backend exists to stop: the SWA layer NAMES this
// backend (`sparse_swa.py:116-118`, `get_attn_backend`), it never wins a
// capability walk, and it is in no platform priority list here either.
//
// It carries HOST METADATA ONLY, exactly as upstream's does for this decision:
// the declared block sizes, the declared head sizes and the paged shape. The
// forward is the model's own DSA op path (`deepseek_v4_dsa.cpp`), and this class
// deliberately has no `get_impl_cls()` — see `## Owed` in
// `.agents/specs/dsv4-per-group-attn-backend.md`.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vllm/v1/attention/backend.h"

namespace vllm::v1 {

class DeepseekSparseSWABackend final : public AttentionBackend {
 public:
  // sparse_swa.py:123-125.
  static constexpr const char* kName = "DEEPSEEK_SPARSE_SWA";

  std::string get_name() const override { return kName; }

  // sparse_swa.py:126-128 — `[MultipleOf(64)]`. NOT 16: the two rules are
  // independent, and 64 is a multiple of 16 only by coincidence of this model's
  // page arithmetic. Written as the declared LIST, so the base
  // `supports_block_size` (backend.py:175-192) turns it into upstream's answer.
  std::vector<int> get_supported_kernel_block_sizes() const override {
    return {64};
  }

  // sparse_swa.py:133-135.
  std::vector<int> get_supported_head_sizes() const override { return {512}; }

  // The cache holds ONE vector per token (`sparse_swa.py:101-113` binds
  // `[B, H=1, N, C] -> [B, N, C]`), which is the fused rank-3 MLA view
  // `CheckKvCacheShape` expects for every group this model publishes.
  bool is_mla() const override { return true; }

  // sparse_swa.py:96-98 — the layer is a sliding-window cache.
  bool supports_sliding_window() const override { return true; }

  // The uint8 arm is `fp8_ds_mla`'s UE8M0 paged layout and the bf16 /
  // float8_e4m3fn arms are the contiguous full-cache layout
  // (`sparse_swa.py:88-90`).
  std::vector<std::string> supported_kv_cache_dtypes() const override {
    return {"auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_ds_mla"};
  }

  std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size,
      const std::string& cache_dtype_str = "auto") const override;
};

}  // namespace vllm::v1
