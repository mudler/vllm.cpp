// Ported from:
//   vllm/model_executor/layers/rotary_embedding/
//   dynamic_ntk_alpha_rope.py:9-43
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/rotary_embedding/dynamic_ntk_alpha_rope.h"

#include <cmath>
#include <stdexcept>

namespace vllm {

DynamicNTKAlphaRotaryEmbedding::DynamicNTKAlphaRotaryEmbedding(
    int64_t head_size, int64_t rotary_dim,
    int64_t max_position_embeddings, double base, bool is_neox_style,
    double scaling_alpha, vt::DType dtype)
    : RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base,
                      is_neox_style, dtype, false),
      scaling_alpha_(scaling_alpha) {
  if (rotary_dim_ <= 2) {
    throw std::invalid_argument("dynamic NTK requires rotary_dim greater than 2");
  }
  initialize_cache();
}

std::vector<float>
DynamicNTKAlphaRotaryEmbedding::_compute_cos_sin_cache() const {
  const double exponent = static_cast<double>(rotary_dim_) /
                          static_cast<double>(rotary_dim_ - 2);
  const double scaled_base = base_ * std::pow(scaling_alpha_, exponent);
  return build_cos_sin_cache(_compute_inv_freq(scaled_base),
                             max_position_embeddings_);
}

}  // namespace vllm
