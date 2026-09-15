// Ported from vLLM linear_scaling_rope.py:37-127 @ e126687a9a828d51.
#pragma once

#include <map>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"

namespace vllm {

class LinearScalingRotaryEmbedding final : public RotaryEmbedding {
 public:
  LinearScalingRotaryEmbedding(int64_t head_size, int64_t rotary_dim,
      int64_t max_position_embeddings, double base, bool is_neox_style,
      double scaling_factor, vt::DType dtype);
  LinearScalingRotaryEmbedding(int64_t head_size, int64_t rotary_dim,
      int64_t max_position_embeddings, double base, bool is_neox_style,
      std::vector<double> scaling_factors, vt::DType dtype);

  std::string type_name() const override { return "LinearScalingRotaryEmbedding"; }
  const std::map<double, int64_t>& scaling_factor_to_offset() const {
    return scaling_factor_to_offset_;
  }

 protected:
  std::vector<float> _compute_cos_sin_cache() const override;

 private:
  std::vector<double> scaling_factors_;
  std::vector<int64_t> lengths_;
  std::map<double, int64_t> scaling_factor_to_offset_;
  int64_t total_rows_ = 0;
};

}  // namespace vllm
