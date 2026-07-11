// Ported from:
//   vllm/model_executor/layers/rotary_embedding/llama3_rope.py:11-54
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/rotary_embedding/llama3_rope.h"

#include <cmath>
#include <stdexcept>

namespace vllm {

Llama3RotaryEmbedding::Llama3RotaryEmbedding(
    int64_t head_size, int64_t rotary_dim,
    int64_t max_position_embeddings, double base, bool is_neox_style,
    vt::DType dtype, double scaling_factor, double low_freq_factor,
    double high_freq_factor, int64_t orig_max_position)
    : RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base,
                      is_neox_style, dtype, false),
      scaling_factor_(scaling_factor),
      low_freq_factor_(low_freq_factor),
      high_freq_factor_(high_freq_factor),
      orig_max_position_(orig_max_position) {
  if (!std::isfinite(scaling_factor_) || !(scaling_factor_ > 0.0) ||
      !std::isfinite(low_freq_factor_) || !(low_freq_factor_ > 0.0) ||
      !std::isfinite(high_freq_factor_) || !(high_freq_factor_ > 0.0) ||
      orig_max_position_ <= 0) {
    throw std::invalid_argument(
        "Llama 3 RoPE factors and original max position must be finite and "
        "positive");
  }
  initialize_cache();
}

std::vector<float> Llama3RotaryEmbedding::_compute_inv_freq(
    double base) const {
  std::vector<float> inv_freqs =
      RotaryEmbeddingBase::_compute_inv_freq(base);
  const float scaling_factor = static_cast<float>(scaling_factor_);
  const float low_freq_factor = static_cast<float>(low_freq_factor_);
  const float high_freq_factor = static_cast<float>(high_freq_factor_);
  const float original_max = static_cast<float>(orig_max_position_);
  const float low_freq_wavelen = original_max / low_freq_factor;
  const float high_freq_wavelen = original_max / high_freq_factor;
  const float two_pi = static_cast<float>(2.0 * std::acos(-1.0));

  for (float& inv_freq : inv_freqs) {
    const float wave_len = two_pi / inv_freq;
    float smooth = 0.0F;
    if (low_freq_factor_ != high_freq_factor_) {
      smooth = (original_max / wave_len - low_freq_factor) /
               (high_freq_factor - low_freq_factor);
    }
    if (wave_len < high_freq_wavelen) {
      continue;
    }
    if (wave_len > low_freq_wavelen) {
      inv_freq /= scaling_factor;
      continue;
    }
    inv_freq = (1.0F - smooth) * inv_freq / scaling_factor +
               smooth * inv_freq;
  }
  return inv_freqs;
}

}  // namespace vllm
