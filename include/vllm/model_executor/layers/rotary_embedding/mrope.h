// Ported from:
//   vllm/model_executor/layers/rotary_embedding/mrope.py:14-187,190-340
// @ e24d1b24fe96.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"

namespace vllm {

class MRotaryEmbedding final : public RotaryEmbeddingBase {
 public:
  MRotaryEmbedding(
      int64_t head_size, int64_t rotary_dim,
      int64_t max_position_embeddings, double base, bool is_neox_style,
      vt::DType dtype, std::vector<int64_t> mrope_section,
      bool mrope_interleaved = false,
      std::optional<double> scaling_factor = std::nullopt,
      double extrapolation_factor = 1.0, double attn_factor = 1.0,
      int64_t beta_fast = 32, int64_t beta_slow = 1,
      bool truncate = true);

  std::string type_name() const override { return "MRotaryEmbedding"; }

  const std::vector<int64_t>& mrope_section() const {
    return mrope_section_;
  }
  bool mrope_interleaved() const { return mrope_interleaved_; }
  std::optional<double> scaling_factor() const { return scaling_factor_; }
  float mscale() const { return mscale_; }

  void forward(vt::Queue& queue, const vt::Tensor& positions,
               vt::Tensor& query, vt::Tensor* key,
               const vt::Tensor& cache) const override;
  void forward_native(vt::Queue& queue, const vt::Tensor& positions,
                      vt::Tensor& query,
                      vt::Tensor* key = nullptr) const override;

  static std::array<std::vector<int64_t>, 3> get_next_input_positions(
      int64_t mrope_position_delta, int64_t context_len, int64_t seq_len);

 protected:
  std::vector<float> _compute_inv_freq(double base_or_factor) const override;
  std::vector<float> _compute_cos_sin_cache() const override;
  vt::RopeArgs rope_args() const override;

 private:
  std::vector<int64_t> mrope_section_;
  bool mrope_interleaved_;
  std::optional<double> scaling_factor_;
  double extrapolation_factor_;
  double attn_factor_;
  int64_t beta_fast_;
  int64_t beta_slow_;
  bool truncate_;
  float mscale_;
};

}  // namespace vllm
