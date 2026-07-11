// Ported from:
//   vllm/model_executor/layers/rotary_embedding/llama3_rope.py:11-54
// @ e24d1b24fe96.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"

namespace vllm {

class Llama3RotaryEmbedding final : public RotaryEmbedding {
 public:
  Llama3RotaryEmbedding(
      int64_t head_size, int64_t rotary_dim,
      int64_t max_position_embeddings, double base, bool is_neox_style,
      vt::DType dtype, double scaling_factor, double low_freq_factor,
      double high_freq_factor, int64_t orig_max_position);

  std::string type_name() const override { return "Llama3RotaryEmbedding"; }

  double scaling_factor() const { return scaling_factor_; }
  double low_freq_factor() const { return low_freq_factor_; }
  double high_freq_factor() const { return high_freq_factor_; }
  int64_t orig_max_position() const { return orig_max_position_; }

 protected:
  std::vector<float> _compute_inv_freq(double base) const override;

 private:
  double scaling_factor_;
  double low_freq_factor_;
  double high_freq_factor_;
  int64_t orig_max_position_;
};

}  // namespace vllm
