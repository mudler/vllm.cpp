// Ported from: vllm/v1/kv_cache_interface.py @ e24d1b24
//
// The page_size_bytes / real_page_size_bytes math for FullAttentionSpec,
// SlidingWindowSpec and MambaSpec. See kv_cache_interface.h for the formulas
// and deferred-spec notes.
#include "vllm/v1/kv_cache_interface.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vllm::v1 {

namespace {

// Upstream KVQuantMode.is_per_token_head / is_nvfp4 (the modes whose page-size
// math is deferred here — see header).
bool needs_deferred_quant_math(KVQuantMode mode) {
  return mode != KVQuantMode::kNone;
}

}  // namespace

int64_t AttentionSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "AttentionSpec: kv_quant_mode != NONE page-size math is deferred (T1)");
  }
  // K + V: 2 * block_size * num_kv_heads * head_size * dtype_size.
  return 2LL * block_size * num_kv_heads * head_size *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int64_t AttentionSpec::page_size_bytes() const {
  const int64_t real_page_size = real_page_size_bytes();
  // (Per-token-head scale bytes for quantized KV cache are deferred; the NONE
  // path adds nothing here.)
  if (page_size_padded.has_value()) {
    if (*page_size_padded < real_page_size) {
      throw std::runtime_error(
          "AttentionSpec: page_size_padded must be >= real_page_size_bytes");
    }
    return *page_size_padded;
  }
  return real_page_size;
}

int64_t FullAttentionSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "FullAttentionSpec: kv_quant_mode != NONE page-size math is deferred "
        "(T1)");
  }
  const int64_t last_dim =
      static_cast<int64_t>(head_size) + static_cast<int64_t>(head_size_v);
  return static_cast<int64_t>(block_size) * num_kv_heads * last_dim *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int64_t SlidingWindowSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "SlidingWindowSpec: kv_quant_mode != NONE page-size math is deferred "
        "(T1)");
  }
  const int64_t last_dim =
      static_cast<int64_t>(head_size) + static_cast<int64_t>(head_size_v);
  return static_cast<int64_t>(block_size) * num_kv_heads * last_dim *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int SlidingWindowSpec::max_admission_blocks_per_request(
    int max_num_batched_tokens, int max_model_len) const {
  const int num_tokens = std::min(
      sliding_window - 1 + max_num_batched_tokens, max_model_len);
  return (num_tokens + block_size - 1) / block_size + 1;
}

int ChunkedLocalAttentionSpec::max_admission_blocks_per_request(
    int max_num_batched_tokens, int max_model_len) const {
  const int num_tokens = std::min(
      attention_chunk_size + max_num_batched_tokens, max_model_len);
  return (num_tokens + block_size - 1) / block_size;
}

int64_t MambaSpec::page_size_bytes() const {
  if (shapes.size() != dtypes.size()) {
    throw std::runtime_error(
        "MambaSpec: shapes and dtypes must have the same length");
  }
  int64_t page_size = 0;
  for (size_t i = 0; i < shapes.size(); ++i) {
    int64_t numel = 1;
    for (int64_t dim : shapes[i]) {
      numel *= dim;
    }
    page_size += numel * static_cast<int64_t>(vt::SizeOf(dtypes[i]));
  }
  if (page_size_padded.has_value()) {
    if (*page_size_padded < page_size) {
      throw std::runtime_error(
          "MambaSpec: page_size_padded must be >= computed page_size");
    }
    return *page_size_padded;
  }
  return page_size;
}

bool KVCacheConfig::has_mamba_layers() const {
  for (const auto& group : kv_cache_groups) {
    if (dynamic_cast<const MambaSpec*>(group.kv_cache_spec.get()) != nullptr) {
      return true;
    }
  }
  return false;
}

}  // namespace vllm::v1
