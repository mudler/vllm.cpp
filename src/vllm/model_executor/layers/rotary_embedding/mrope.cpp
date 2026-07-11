// Ported from:
//   vllm/model_executor/layers/rotary_embedding/mrope.py:14-187,190-340
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/rotary_embedding/mrope.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "vllm/model_executor/layers/rotary_embedding/common.h"
#include "vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.h"

namespace vllm {

namespace {

int64_t MropeCacheMax(int64_t max_position_embeddings) {
  if (max_position_embeddings <= 0 ||
      max_position_embeddings > std::numeric_limits<int64_t>::max() / 4) {
    throw std::invalid_argument("invalid MRoPE max_position_embeddings");
  }
  // Qwen2.5-VL may derive indices from video duration; pinned vLLM reserves 4x.
  return max_position_embeddings * 4;
}

}  // namespace

MRotaryEmbedding::MRotaryEmbedding(
    int64_t head_size, int64_t rotary_dim,
    int64_t max_position_embeddings, double base, bool is_neox_style,
    vt::DType dtype, std::vector<int64_t> mrope_section,
    bool mrope_interleaved, std::optional<double> scaling_factor,
    double extrapolation_factor, double attn_factor, int64_t beta_fast,
    int64_t beta_slow, bool truncate)
    : RotaryEmbeddingBase(head_size, rotary_dim,
                          MropeCacheMax(max_position_embeddings), base,
                          is_neox_style, dtype, false),
      mrope_section_(std::move(mrope_section)),
      mrope_interleaved_(mrope_interleaved),
      scaling_factor_(scaling_factor),
      extrapolation_factor_(extrapolation_factor),
      attn_factor_(attn_factor),
      beta_fast_(beta_fast),
      beta_slow_(beta_slow),
      truncate_(truncate),
      mscale_(scaling_factor_.has_value()
                  ? static_cast<float>(yarn_get_mscale(*scaling_factor_) *
                                       attn_factor_)
                  : 1.0F) {
  if (mrope_section_.size() != 3) {
    throw std::invalid_argument("mrope_section must contain T/H/W entries");
  }
  int64_t sum = 0;
  for (int64_t section : mrope_section_) {
    if (section < 0 ||
        section > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      throw std::invalid_argument("mrope_section entries are out of range");
    }
    sum += section;
  }
  if (sum != rotary_dim_ / 2) {
    throw std::invalid_argument(
        "sum(mrope_section) must equal rotary_dim / 2");
  }
  if (scaling_factor_.has_value() &&
      (!(*scaling_factor_ > 0.0) || !std::isfinite(*scaling_factor_))) {
    throw std::invalid_argument("MRoPE YaRN factor must be finite and positive");
  }
  initialize_cache();
}

std::vector<float> MRotaryEmbedding::_compute_inv_freq(
    double base_or_factor) const {
  if (!scaling_factor_.has_value()) {
    return RotaryEmbeddingBase::_compute_inv_freq(base_or_factor);
  }
  return rotary_embedding_detail::compute_yarn_inv_freq(
      rotary_dim_, base_, max_position_embeddings_, *scaling_factor_,
      extrapolation_factor_, beta_fast_, beta_slow_, truncate_);
}

std::vector<float> MRotaryEmbedding::_compute_cos_sin_cache() const {
  if (!scaling_factor_.has_value()) {
    return RotaryEmbeddingBase::_compute_cos_sin_cache();
  }
  return rotary_embedding_detail::compute_yarn_cos_sin_cache(
      rotary_dim_, base_, max_position_embeddings_, *scaling_factor_,
      extrapolation_factor_, beta_fast_, beta_slow_, truncate_, mscale_);
}

vt::RopeArgs MRotaryEmbedding::rope_args() const {
  vt::RopeArgs args = RotaryEmbeddingBase::rope_args();
  for (size_t i = 0; i < 3; ++i) {
    args.mrope_section[i] = static_cast<int32_t>(mrope_section_[i]);
  }
  args.mrope_interleaved = mrope_interleaved_;
  return args;
}

void MRotaryEmbedding::forward(vt::Queue& queue,
                               const vt::Tensor& positions,
                               vt::Tensor& query, vt::Tensor* key,
                               const vt::Tensor& cache) const {
  if (key == nullptr) {
    throw std::invalid_argument("MRotaryEmbedding requires a key tensor");
  }
  if (positions.rank != 1 && positions.rank != 2) {
    throw std::invalid_argument("MRoPE positions must be [T] or [3,T]");
  }
  RotaryEmbeddingBase::forward(queue, positions, query, key, cache);
}

void MRotaryEmbedding::forward_native(vt::Queue& queue,
                                      const vt::Tensor& positions,
                                      vt::Tensor& query,
                                      vt::Tensor* key) const {
  if (key == nullptr) {
    throw std::invalid_argument("MRotaryEmbedding requires a key tensor");
  }
  vt::Tensor cache = cos_sin_cache();
  forward(queue, positions, query, key, cache);
}

std::array<std::vector<int64_t>, 3>
MRotaryEmbedding::get_next_input_positions(int64_t mrope_position_delta,
                                           int64_t context_len,
                                           int64_t seq_len) {
  if (seq_len < context_len) {
    throw std::invalid_argument("seq_len must be at least context_len");
  }
  std::vector<int64_t> values;
  values.reserve(static_cast<size_t>(seq_len - context_len));
  for (int64_t i = context_len + mrope_position_delta;
       i < seq_len + mrope_position_delta; ++i) {
    values.push_back(i);
  }
  return {values, values, values};
}

}  // namespace vllm
