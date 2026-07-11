// Ported from:
//   vllm/model_executor/layers/rotary_embedding/__init__.py:30-112,200-230,
//   243-283,315-335
//   vllm/model_executor/layers/rotary_embedding/base.py:13-252,298-318
// @ e24d1b24fe96.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {

class RotaryEmbeddingBase {
 public:
  virtual ~RotaryEmbeddingBase() = default;

  int64_t head_size() const { return head_size_; }
  int64_t rotary_dim() const { return rotary_dim_; }
  int64_t max_position_embeddings() const {
    return max_position_embeddings_;
  }
  double base() const { return base_; }
  bool is_neox_style() const { return is_neox_style_; }
  vt::DType dtype() const { return dtype_; }
  int64_t cache_rows() const { return cache_rows_; }
  size_t cache_bytes() const { return cos_sin_cache_.size(); }
  const void* cache_data() const { return cos_sin_cache_.data(); }

  // Non-owning CPU view over the dtype-specific cache registered by upstream
  // as cos_sin_cache. The embedding object owns the referenced bytes.
  vt::Tensor cos_sin_cache() const;

  virtual std::string type_name() const = 0;

  // Apply a caller-supplied cache on any registered vt backend. This is the
  // C++ equivalent of upstream's _match_cos_sin_cache_dtype followed by its
  // custom op. `key` may be null for the base one-dimensional path.
  virtual void forward(vt::Queue& queue, const vt::Tensor& positions,
                       vt::Tensor& query, vt::Tensor* key,
                       const vt::Tensor& cache) const;

  // CPU-native convenience path using this object's owned cache.
  virtual void forward_native(vt::Queue& queue, const vt::Tensor& positions,
                              vt::Tensor& query,
                              vt::Tensor* key = nullptr) const;

 protected:
  RotaryEmbeddingBase(int64_t head_size, int64_t rotary_dim,
                      int64_t max_position_embeddings, double base,
                      bool is_neox_style, vt::DType dtype,
                      bool init_cache = true);

  virtual std::vector<float> _compute_inv_freq(double base) const;
  virtual std::vector<float> _compute_cos_sin_cache() const;
  virtual vt::RopeArgs rope_args() const;

  std::vector<float> build_cos_sin_cache(
      const std::vector<float>& inv_freq, int64_t rows,
      float magnitude_scale = 1.0F) const;
  void initialize_cache();

  int64_t head_size_;
  int64_t rotary_dim_;
  int64_t max_position_embeddings_;
  double base_;
  bool is_neox_style_;
  vt::DType dtype_;

 private:
  int64_t cache_rows_ = 0;
  std::vector<std::byte> cos_sin_cache_;
};

class RotaryEmbedding : public RotaryEmbeddingBase {
 public:
  RotaryEmbedding(int64_t head_size, int64_t rotary_dim,
                  int64_t max_position_embeddings, double base,
                  bool is_neox_style, vt::DType dtype);
  RotaryEmbedding(int64_t head_size, int64_t rotary_dim,
                  int64_t max_position_embeddings, double base,
                  bool is_neox_style, vt::DType dtype, bool init_cache);
  std::string type_name() const override { return "RotaryEmbedding"; }
};

// Memoized typed factory. The effective key covers every typed RoPE parameter plus
// head/rotary/max/layout/dtype, mirroring pinned get_rope's module cache.
std::shared_ptr<RotaryEmbeddingBase> get_rope(
    int64_t head_size, int64_t max_position, bool is_neox_style,
    const RopeParameters& rope_parameters,
    vt::DType dtype = vt::DType::kF32);

// Additive equivalent of pinned LongRoPE's get_current_vllm_config() lookup.
// The original overload keeps vLLM's default short-cache choice; callers with
// an explicit runtime max length use this overload, and the value joins the
// memoization key so short and long configurations cannot alias.
std::shared_ptr<RotaryEmbeddingBase> get_rope(
    int64_t head_size, int64_t max_position, bool is_neox_style,
    const RopeParameters& rope_parameters, vt::DType dtype,
    int64_t max_model_len);

}  // namespace vllm
