// Ported from:
//   vllm/model_executor/layers/rotary_embedding/
//   dynamic_ntk_alpha_rope.py:9-43
// @ e24d1b24fe96.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"

namespace vllm {

class DynamicNTKAlphaRotaryEmbedding final : public RotaryEmbedding {
 public:
  DynamicNTKAlphaRotaryEmbedding(
      int64_t head_size, int64_t rotary_dim,
      int64_t max_position_embeddings, double base, bool is_neox_style,
      double scaling_alpha, vt::DType dtype);

  std::string type_name() const override {
    return "DynamicNTKAlphaRotaryEmbedding";
  }
  double scaling_alpha() const { return scaling_alpha_; }

 protected:
  std::vector<float> _compute_cos_sin_cache() const override;

 private:
  double scaling_alpha_;
};

}  // namespace vllm
