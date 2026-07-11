// Ported from:
//   vllm/model_executor/layers/rotary_embedding/
//   dynamic_ntk_scaling_rope.py:30-73
// @ e24d1b24fe96.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"

namespace vllm {

class DynamicNTKScalingRotaryEmbedding final : public RotaryEmbedding {
 public:
  DynamicNTKScalingRotaryEmbedding(
      int64_t head_size, int64_t rotary_dim,
      int64_t max_position_embeddings, int64_t max_trained_positions,
      double base, bool is_neox_style, double scaling_factor,
      vt::DType dtype);

  std::string type_name() const override {
    return "DynamicNTKScalingRotaryEmbedding";
  }
  double scaling_factor() const { return scaling_factor_; }
  int64_t max_trained_positions() const { return max_trained_positions_; }

 protected:
  std::vector<float> _compute_cos_sin_cache() const override;

 private:
  double scaling_factor_;
  int64_t max_trained_positions_;
};

}  // namespace vllm
