// Ported from:
//   vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py:10-84
// @ e24d1b24fe96.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"

namespace vllm {

namespace rotary_embedding_detail {

std::vector<float> compute_yarn_inv_freq(
    int64_t rotary_dim, double base, int64_t max_position_embeddings,
    double scaling_factor, double extrapolation_factor, int64_t beta_fast,
    int64_t beta_slow, bool truncate);

std::vector<float> compute_yarn_cos_sin_cache(
    int64_t rotary_dim, double base, int64_t max_position_embeddings,
    double scaling_factor, double extrapolation_factor, int64_t beta_fast,
    int64_t beta_slow, bool truncate, float mscale);

}  // namespace rotary_embedding_detail

class YaRNScalingRotaryEmbedding final : public RotaryEmbeddingBase {
 public:
  YaRNScalingRotaryEmbedding(
      int64_t head_size, int64_t rotary_dim,
      int64_t max_position_embeddings, double base, bool is_neox_style,
      double scaling_factor, vt::DType dtype,
      double extrapolation_factor = 1.0, double attn_factor = 1.0,
      int64_t beta_fast = 32, int64_t beta_slow = 1,
      bool apply_yarn_scaling = true, bool truncate = true);

  std::string type_name() const override {
    return "YaRNScalingRotaryEmbedding";
  }

  double scaling_factor() const { return scaling_factor_; }
  double extrapolation_factor() const { return extrapolation_factor_; }
  double attn_factor() const { return attn_factor_; }
  int64_t beta_fast() const { return beta_fast_; }
  int64_t beta_slow() const { return beta_slow_; }
  bool truncate() const { return truncate_; }
  float mscale() const { return mscale_; }

 protected:
  std::vector<float> _compute_inv_freq(double scaling_factor) const override;
  std::vector<float> _compute_cos_sin_cache() const override;

 private:
  double scaling_factor_;
  double extrapolation_factor_;
  double attn_factor_;
  int64_t beta_fast_;
  int64_t beta_slow_;
  bool truncate_;
  float mscale_;
};

}  // namespace vllm
