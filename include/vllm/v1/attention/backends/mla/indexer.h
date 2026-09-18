// Ported from: vllm/v1/attention/backends/mla/indexer.py @ pin e126687a9a
// (`DeepseekV4IndexerBackend`, `:183-196`).
//
// The indexer key cache takes the ENGINE block size, which this tree and
// upstream both resolve to 256 for DeepSeek-V4. Upstream declares that as the
// EXACT size `[256]` and not as a multiple-of rule, so 64 and 512 are both
// refused; that exactness is ported rather than softened, because
// `supports_block_size` reads every entry as a MultipleOf and a `{256}` entry
// says "any multiple of 256", which is the nearest expressible statement and is
// what refuses 64.
//
// Host metadata only, for the same reason as `sparse_swa.h`.
#pragma once

#include <string>
#include <vector>

#include "vllm/v1/attention/backend.h"

namespace vllm::v1 {

class DeepseekV4IndexerBackend final : public AttentionBackend {
 public:
  // indexer.py:185-187.
  static constexpr const char* kName = "DEEPSEEK_V4_INDEXER";

  std::string get_name() const override { return kName; }

  // indexer.py:194-196.
  std::vector<int> get_supported_kernel_block_sizes() const override {
    return {256};
  }

  // The indexer key row is one packed vector per token (fp8 key plus its
  // per-128 scales), so the paged view is the fused rank-3 one. Its head size is
  // `index_head_dim + index_head_dim/128*4` and is model-derived
  // (`deepseek_v4_registry.cpp:494-496`), so no head-size list is declared —
  // upstream's `DeepseekV32IndexerBackend` declares none either.
  bool is_mla() const override { return true; }

  std::vector<std::string> supported_kv_cache_dtypes() const override {
    return {"auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_ds_mla"};
  }

  std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size,
      const std::string& cache_dtype_str = "auto") const override;
};

}  // namespace vllm::v1
