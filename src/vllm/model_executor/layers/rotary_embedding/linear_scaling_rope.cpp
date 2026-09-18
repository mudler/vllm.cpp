// SPDX-License-Identifier: Apache-2.0
// Ported from vLLM linear_scaling_rope.py:37-127 @ e126687a9a828d51.
// Copyright contributors to the vLLM project; adapted from HuggingFace's
// Apache-2.0 Llama rotary implementation (see upstream source attribution).
#include "vllm/model_executor/layers/rotary_embedding/linear_scaling_rope.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vllm {

LinearScalingRotaryEmbedding::LinearScalingRotaryEmbedding(
    int64_t head_size, int64_t rotary_dim, int64_t max_position_embeddings,
    double base, bool is_neox_style, double scaling_factor, vt::DType dtype)
    : LinearScalingRotaryEmbedding(head_size, rotary_dim, max_position_embeddings,
          base, is_neox_style, std::vector<double>{scaling_factor}, dtype) {}

LinearScalingRotaryEmbedding::LinearScalingRotaryEmbedding(
    int64_t head_size, int64_t rotary_dim, int64_t max_position_embeddings,
    double base, bool is_neox_style, std::vector<double> scaling_factors,
    vt::DType dtype)
    : RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base,
          is_neox_style, dtype, /*init_cache=*/false),
      scaling_factors_(std::move(scaling_factors)) {
  if (scaling_factors_.empty())
    throw std::invalid_argument("linear RoPE requires at least one factor");
  for (double factor : scaling_factors_) {
    if (!(factor > 0.) || !std::isfinite(factor))
      throw std::invalid_argument("linear RoPE factor must be finite and positive");
    const double rows = std::ceil(static_cast<double>(max_position_embeddings) * factor);
    const int64_t limit = std::numeric_limits<int64_t>::max() / rotary_dim;
    if (!std::isfinite(rows) || rows < 1. ||
        static_cast<long double>(rows) > static_cast<long double>(limit - total_rows_))
      throw std::overflow_error("linear RoPE cache size overflow");
    const auto length = static_cast<int64_t>(rows);
    scaling_factor_to_offset_[factor] = total_rows_;
    lengths_.push_back(length);
    total_rows_ += length;
  }
  if (static_cast<uint64_t>(total_rows_) >
      std::numeric_limits<size_t>::max() / sizeof(float) / static_cast<uint64_t>(rotary_dim))
    throw std::overflow_error("linear RoPE cache byte size overflow");
  initialize_cache();
}

std::vector<float> LinearScalingRotaryEmbedding::_compute_cos_sin_cache() const {
  const auto inv_freq = _compute_inv_freq(base_);
  std::vector<float> cache(static_cast<size_t>(total_rows_) * rotary_dim_);
  const int64_t half = rotary_dim_ / 2;
  int64_t offset = 0;
  for (size_t f = 0; f < scaling_factors_.size(); ++f) {
    for (int64_t position = 0; position < lengths_[f]; ++position) {
      // Match upstream's FP32 position division before the outer product.
      const float t = static_cast<float>(position) / static_cast<float>(scaling_factors_[f]);
      const auto row = static_cast<size_t>(offset + position) * rotary_dim_;
      for (int64_t i = 0; i < half; ++i) {
        const float angle = t * inv_freq[static_cast<size_t>(i)];
        cache[row + i] = std::cos(angle);
        cache[row + half + i] = std::sin(angle);
      }
    }
    offset += lengths_[f];
  }
  return cache;
}

}  // namespace vllm
