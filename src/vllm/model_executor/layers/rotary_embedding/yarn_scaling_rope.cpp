// Ported from:
//   vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py:10-84
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "vllm/model_executor/layers/rotary_embedding/common.h"

namespace vllm {

namespace rotary_embedding_detail {

std::vector<float> compute_yarn_inv_freq(
    int64_t rotary_dim, double base, int64_t max_position_embeddings,
    double scaling_factor, double extrapolation_factor, int64_t beta_fast,
    int64_t beta_slow, bool truncate) {
  if (rotary_dim <= 0 || rotary_dim % 2 != 0 ||
      !(scaling_factor > 0.0) || !std::isfinite(scaling_factor)) {
    throw std::invalid_argument("invalid YaRN inverse-frequency arguments");
  }
  const int64_t half = rotary_dim / 2;
  const auto [low, high] = yarn_find_correction_range(
      beta_fast, beta_slow, rotary_dim, base, max_position_embeddings,
      truncate);
  const std::vector<float> ramp =
      yarn_linear_ramp_mask(low, high, half);
  std::vector<float> inv_freq(static_cast<size_t>(half));
  const float base_f = static_cast<float>(base);
  const float factor_f = static_cast<float>(scaling_factor);
  const float extrapolation_f = static_cast<float>(extrapolation_factor);
  for (int64_t i = 0; i < half; ++i) {
    const float exponent =
        static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
    const float pos_freq = std::pow(base_f, exponent);
    const float inv_extrapolation = 1.0F / pos_freq;
    const float inv_interpolation = 1.0F / (factor_f * pos_freq);
    const float mask =
        (1.0F - ramp[static_cast<size_t>(i)]) * extrapolation_f;
    inv_freq[static_cast<size_t>(i)] =
        inv_interpolation * (1.0F - mask) + inv_extrapolation * mask;
  }
  return inv_freq;
}

std::vector<float> compute_yarn_cos_sin_cache(
    int64_t rotary_dim, double base, int64_t max_position_embeddings,
    double scaling_factor, double extrapolation_factor, int64_t beta_fast,
    int64_t beta_slow, bool truncate, float mscale) {
  const long double rows_real =
      static_cast<long double>(max_position_embeddings) *
      static_cast<long double>(scaling_factor);
  if (!(rows_real > 0.0L) ||
      rows_real > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("YaRN cache length overflow");
  }
  const int64_t rows = static_cast<int64_t>(std::ceil(rows_real));
  if (rows > std::numeric_limits<int64_t>::max() / rotary_dim ||
      static_cast<uint64_t>(rows) >
          std::numeric_limits<size_t>::max() /
              static_cast<uint64_t>(rotary_dim)) {
    throw std::overflow_error("YaRN cache size overflow");
  }
  const std::vector<float> inv_freq = compute_yarn_inv_freq(
      rotary_dim, base, max_position_embeddings, scaling_factor,
      extrapolation_factor, beta_fast, beta_slow, truncate);
  const size_t elements =
      static_cast<size_t>(rows) * static_cast<size_t>(rotary_dim);
  std::vector<float> cache(elements);
  const int64_t half = rotary_dim / 2;
  for (int64_t position = 0; position < rows; ++position) {
    const float p = static_cast<float>(position);
    for (int64_t i = 0; i < half; ++i) {
      const float freq = p * inv_freq[static_cast<size_t>(i)];
      cache[static_cast<size_t>(position * rotary_dim + i)] =
          std::cos(freq) * mscale;
      cache[static_cast<size_t>(position * rotary_dim + half + i)] =
          std::sin(freq) * mscale;
    }
  }
  return cache;
}

}  // namespace rotary_embedding_detail

YaRNScalingRotaryEmbedding::YaRNScalingRotaryEmbedding(
    int64_t head_size, int64_t rotary_dim,
    int64_t max_position_embeddings, double base, bool is_neox_style,
    double scaling_factor, vt::DType dtype, double extrapolation_factor,
    double attn_factor, int64_t beta_fast, int64_t beta_slow,
    bool apply_yarn_scaling, bool truncate)
    : RotaryEmbeddingBase(head_size, rotary_dim, max_position_embeddings, base,
                          is_neox_style, dtype, false),
      scaling_factor_(scaling_factor),
      extrapolation_factor_(extrapolation_factor),
      attn_factor_(attn_factor),
      beta_fast_(beta_fast),
      beta_slow_(beta_slow),
      truncate_(truncate),
      mscale_(static_cast<float>(
          (apply_yarn_scaling ? yarn_get_mscale(scaling_factor_) : 1.0) *
          attn_factor_)) {
  if (!(scaling_factor_ > 0.0) || !std::isfinite(scaling_factor_) ||
      !std::isfinite(extrapolation_factor_) || !std::isfinite(attn_factor_)) {
    throw std::invalid_argument("YaRN factors must be finite and positive");
  }
  initialize_cache();
}

std::vector<float> YaRNScalingRotaryEmbedding::_compute_inv_freq(
    double scaling_factor) const {
  return rotary_embedding_detail::compute_yarn_inv_freq(
      rotary_dim_, base_, max_position_embeddings_, scaling_factor,
      extrapolation_factor_, beta_fast_, beta_slow_, truncate_);
}

std::vector<float> YaRNScalingRotaryEmbedding::_compute_cos_sin_cache() const {
  return rotary_embedding_detail::compute_yarn_cos_sin_cache(
      rotary_dim_, base_, max_position_embeddings_, scaling_factor_,
      extrapolation_factor_, beta_fast_, beta_slow_, truncate_, mscale_);
}

}  // namespace vllm
