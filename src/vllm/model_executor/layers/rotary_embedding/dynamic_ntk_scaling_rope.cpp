// Ported from:
//   vllm/model_executor/layers/rotary_embedding/
//   dynamic_ntk_scaling_rope.py:30-73
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/rotary_embedding/dynamic_ntk_scaling_rope.h"

#include <cmath>
#include <stdexcept>

namespace vllm {

DynamicNTKScalingRotaryEmbedding::DynamicNTKScalingRotaryEmbedding(
    int64_t head_size, int64_t rotary_dim,
    int64_t max_position_embeddings, int64_t max_trained_positions,
    double base, bool is_neox_style, double scaling_factor,
    vt::DType dtype)
    : RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base,
                      is_neox_style, dtype, false),
      scaling_factor_(scaling_factor),
      max_trained_positions_(max_trained_positions) {
  if (rotary_dim_ <= 2) {
    throw std::invalid_argument("dynamic NTK requires rotary_dim greater than 2");
  }
  if (max_trained_positions_ <= 0) {
    throw std::invalid_argument("max_trained_positions must be positive");
  }
  initialize_cache();
}

std::vector<float>
DynamicNTKScalingRotaryEmbedding::_compute_cos_sin_cache() const {
  const double scale = scaling_factor_ *
                           static_cast<double>(max_position_embeddings_) /
                           static_cast<double>(max_trained_positions_) -
                       (scaling_factor_ - 1.0);
  const double exponent = static_cast<double>(rotary_dim_) /
                          static_cast<double>(rotary_dim_ - 2);
  const double scaled_base = base_ * std::pow(scale, exponent);
  return build_cos_sin_cache(_compute_inv_freq(scaled_base),
                             max_position_embeddings_);
}

}  // namespace vllm
