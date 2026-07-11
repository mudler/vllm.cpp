// Ported from:
//   vllm/model_executor/layers/rotary_embedding/
//   phi3_long_rope_scaled_rope.py:16-159
// @ e24d1b24fe96.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"

namespace vllm {

class Phi3LongRoPEScaledRotaryEmbedding final
    : public RotaryEmbeddingBase {
 public:
  Phi3LongRoPEScaledRotaryEmbedding(
      int64_t head_size, int64_t rotary_dim,
      int64_t max_position_embeddings,
      int64_t original_max_position_embeddings, double base,
      bool is_neox_style, vt::DType dtype,
      std::vector<double> short_factor, std::vector<double> long_factor,
      std::optional<double> short_mscale,
      std::optional<double> long_mscale, int64_t max_model_len);

  std::string type_name() const override {
    return "Phi3LongRoPEScaledRotaryEmbedding";
  }

  void forward(vt::Queue& queue, const vt::Tensor& positions,
               vt::Tensor& query, vt::Tensor* key,
               const vt::Tensor& cache) const override;

  int64_t original_max_position_embeddings() const {
    return original_max_position_embeddings_;
  }
  int64_t max_model_len() const { return max_model_len_; }
  bool use_long_rope() const { return use_long_rope_; }
  double short_mscale() const { return short_mscale_; }
  double long_mscale() const { return long_mscale_; }
  const std::vector<double>& short_factor() const { return short_factor_; }
  const std::vector<double>& long_factor() const { return long_factor_; }

 protected:
  std::vector<float> _compute_cos_sin_cache() const override;

 private:
  std::vector<float> ComputeInvFreq(
      const std::vector<double>& rescale_factors) const;

  int64_t original_max_position_embeddings_;
  int64_t max_model_len_;
  bool use_long_rope_;
  std::vector<double> short_factor_;
  std::vector<double> long_factor_;
  double short_mscale_;
  double long_mscale_;
};

}  // namespace vllm
