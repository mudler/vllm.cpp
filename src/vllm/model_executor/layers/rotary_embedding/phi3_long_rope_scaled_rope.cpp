// Ported from:
//   vllm/model_executor/layers/rotary_embedding/
//   phi3_long_rope_scaled_rope.py:16-159
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace vllm {

Phi3LongRoPEScaledRotaryEmbedding::
    Phi3LongRoPEScaledRotaryEmbedding(
        int64_t head_size, int64_t rotary_dim,
        int64_t max_position_embeddings,
        int64_t original_max_position_embeddings, double base,
        bool is_neox_style, vt::DType dtype,
        std::vector<double> short_factor, std::vector<double> long_factor,
        std::optional<double> short_mscale,
        std::optional<double> long_mscale, int64_t max_model_len)
    : RotaryEmbeddingBase(head_size, rotary_dim, max_position_embeddings, base,
                          is_neox_style, dtype, false),
      original_max_position_embeddings_(original_max_position_embeddings),
      max_model_len_(max_model_len),
      use_long_rope_(max_model_len > original_max_position_embeddings),
      short_factor_(std::move(short_factor)),
      long_factor_(std::move(long_factor)),
      short_mscale_(0.0),
      long_mscale_(0.0) {
  if (!is_neox_style_) {
    throw std::invalid_argument(
        "Phi3LongRoPEScaledRotaryEmbedding only supports neox_style");
  }
  if (original_max_position_embeddings_ <= 0 || max_model_len_ <= 0) {
    throw std::invalid_argument(
        "LongRoPE original and runtime max positions must be positive");
  }
  const size_t expected = static_cast<size_t>(rotary_dim_ / 2);
  if (short_factor_.size() != expected || long_factor_.size() != expected) {
    throw std::invalid_argument(
        "LongRoPE short_factor and long_factor must have rotary_dim/2 "
        "entries");
  }

  const double scale = static_cast<double>(max_position_embeddings_) /
                       static_cast<double>(original_max_position_embeddings_);
  double default_mscale = 1.0;
  if (scale > 1.0) {
    default_mscale = std::sqrt(
        1.0 + std::log(scale) /
                  std::log(static_cast<double>(
                      original_max_position_embeddings_)));
  }
  short_mscale_ = short_mscale.value_or(default_mscale);
  long_mscale_ = long_mscale.value_or(default_mscale);
  initialize_cache();
}

std::vector<float> Phi3LongRoPEScaledRotaryEmbedding::ComputeInvFreq(
    const std::vector<double>& rescale_factors) const {
  const int64_t half = rotary_dim_ / 2;
  std::vector<float> inv_freq(static_cast<size_t>(half));
  const float base = static_cast<float>(base_);
  for (int64_t i = 0; i < half; ++i) {
    const float exponent =
        static_cast<float>(2 * i) / static_cast<float>(rotary_dim_);
    const float factor =
        static_cast<float>(rescale_factors[static_cast<size_t>(i)]);
    inv_freq[static_cast<size_t>(i)] =
        1.0F / (factor * std::pow(base, exponent));
  }
  return inv_freq;
}

std::vector<float>
Phi3LongRoPEScaledRotaryEmbedding::_compute_cos_sin_cache() const {
  std::vector<float> cache = build_cos_sin_cache(
      ComputeInvFreq(short_factor_), original_max_position_embeddings_,
      static_cast<float>(short_mscale_));
  std::vector<float> long_cache = build_cos_sin_cache(
      ComputeInvFreq(long_factor_), max_position_embeddings_,
      static_cast<float>(long_mscale_));
  cache.insert(cache.end(), long_cache.begin(), long_cache.end());
  return cache;
}

void Phi3LongRoPEScaledRotaryEmbedding::forward(
    vt::Queue& queue, const vt::Tensor& positions, vt::Tensor& query,
    vt::Tensor* key, const vt::Tensor& cache) const {
  if (key == nullptr) {
    throw std::invalid_argument("Phi-3 LongRoPE requires a key tensor");
  }
  if (cache.rank != 2 || cache.shape[1] != rotary_dim_ ||
      cache.shape[0] !=
          original_max_position_embeddings_ + max_position_embeddings_) {
    throw std::invalid_argument(
        "LongRoPE cache must concatenate [original,max] position rows");
  }
  const int64_t offset =
      use_long_rope_ ? original_max_position_embeddings_ : 0;
  const int64_t rows = use_long_rope_ ? max_position_embeddings_
                                      : original_max_position_embeddings_;
  const vt::Tensor selected = cache.Slice(0, offset, offset + rows);
  RotaryEmbeddingBase::forward(queue, positions, query, key, selected);
}

}  // namespace vllm
