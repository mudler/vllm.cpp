// Ported from:
//   vllm/model_executor/layers/rotary_embedding/__init__.py:30-112,243-283
//   vllm/model_executor/layers/rotary_embedding/base.py:13-252,298-318
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/rotary_embedding/base.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/layers/rotary_embedding/llama3_rope.h"
#include "vllm/model_executor/layers/rotary_embedding/mrope.h"
#include "vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace vllm {

RotaryEmbeddingBase::RotaryEmbeddingBase(
    int64_t head_size, int64_t rotary_dim, int64_t max_position_embeddings,
    double base, bool is_neox_style, vt::DType dtype, bool init_cache)
    : head_size_(head_size),
      rotary_dim_(rotary_dim),
      max_position_embeddings_(max_position_embeddings),
      base_(base),
      is_neox_style_(is_neox_style),
      dtype_(dtype) {
  if (head_size_ <= 0 || rotary_dim_ <= 0 || rotary_dim_ % 2 != 0 ||
      rotary_dim_ > head_size_) {
    throw std::invalid_argument(
        "rotary_dim must be positive, even, and no larger than head_size");
  }
  if (max_position_embeddings_ <= 0) {
    throw std::invalid_argument("max_position_embeddings must be positive");
  }
  if (!(base_ > 0.0) || !std::isfinite(base_)) {
    throw std::invalid_argument("RoPE base must be finite and positive");
  }
  if (dtype_ != vt::DType::kF32 && dtype_ != vt::DType::kBF16) {
    throw std::invalid_argument("RoPE cache dtype must be f32 or bf16");
  }
  if (init_cache) initialize_cache();
}

std::vector<float> RotaryEmbeddingBase::_compute_inv_freq(double base) const {
  const int64_t half = rotary_dim_ / 2;
  std::vector<float> inv_freq(static_cast<size_t>(half));
  const float base_f = static_cast<float>(base);
  for (int64_t i = 0; i < half; ++i) {
    const float exponent =
        static_cast<float>(2 * i) / static_cast<float>(rotary_dim_);
    inv_freq[static_cast<size_t>(i)] =
        1.0F / std::pow(base_f, exponent);
  }
  return inv_freq;
}

std::vector<float> RotaryEmbeddingBase::build_cos_sin_cache(
    const std::vector<float>& inv_freq, int64_t rows,
    float magnitude_scale) const {
  if (rows <= 0 || static_cast<int64_t>(inv_freq.size()) != rotary_dim_ / 2) {
    throw std::invalid_argument("invalid RoPE cache dimensions");
  }
  if (rows > std::numeric_limits<int64_t>::max() / rotary_dim_ ||
      static_cast<uint64_t>(rows) >
          std::numeric_limits<size_t>::max() /
              static_cast<uint64_t>(rotary_dim_)) {
    throw std::overflow_error("RoPE cache size overflow");
  }
  const size_t elements =
      static_cast<size_t>(rows) * static_cast<size_t>(rotary_dim_);
  std::vector<float> cache(elements);
  const int64_t half = rotary_dim_ / 2;
  for (int64_t position = 0; position < rows; ++position) {
    const float p = static_cast<float>(position);
    for (int64_t i = 0; i < half; ++i) {
      const float freq = p * inv_freq[static_cast<size_t>(i)];
      cache[static_cast<size_t>(position * rotary_dim_ + i)] =
          std::cos(freq) * magnitude_scale;
      cache[static_cast<size_t>(position * rotary_dim_ + half + i)] =
          std::sin(freq) * magnitude_scale;
    }
  }
  return cache;
}

std::vector<float> RotaryEmbeddingBase::_compute_cos_sin_cache() const {
  return build_cos_sin_cache(_compute_inv_freq(base_),
                             max_position_embeddings_);
}

void RotaryEmbeddingBase::initialize_cache() {
  const std::vector<float> cache = _compute_cos_sin_cache();
  if (cache.empty() || cache.size() % static_cast<size_t>(rotary_dim_) != 0) {
    throw std::runtime_error("RoPE cache builder returned an invalid shape");
  }
  cache_rows_ =
      static_cast<int64_t>(cache.size() / static_cast<size_t>(rotary_dim_));
  const size_t element_size = vt::SizeOf(dtype_);
  if (cache.size() > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error("RoPE cache byte-size overflow");
  }
  cos_sin_cache_.resize(cache.size() * element_size);
  if (dtype_ == vt::DType::kF32) {
    std::memcpy(cos_sin_cache_.data(), cache.data(),
                cache.size() * sizeof(float));
  } else {
    auto* out = reinterpret_cast<uint16_t*>(cos_sin_cache_.data());
    for (size_t i = 0; i < cache.size(); ++i) {
      out[i] = vt::F32ToBF16(cache[i]);
    }
  }
}

vt::Tensor RotaryEmbeddingBase::cos_sin_cache() const {
  return vt::Tensor::Contiguous(
      const_cast<std::byte*>(cos_sin_cache_.data()), dtype_,
      vt::Device{vt::DeviceType::kCPU, 0}, {cache_rows_, rotary_dim_});
}

vt::RopeArgs RotaryEmbeddingBase::rope_args() const {
  vt::RopeArgs args;
  args.base = static_cast<float>(base_);
  args.rotary_dim = static_cast<int>(rotary_dim_);
  args.is_neox_style = is_neox_style_;
  return args;
}

void RotaryEmbeddingBase::forward(vt::Queue& queue,
                                  const vt::Tensor& positions,
                                  vt::Tensor& query, vt::Tensor* key,
                                  const vt::Tensor& cache) const {
  vt::RopeFromCache(queue, query, key, positions, cache, rope_args());
}

void RotaryEmbeddingBase::forward_native(vt::Queue& queue,
                                         const vt::Tensor& positions,
                                         vt::Tensor& query,
                                         vt::Tensor* key) const {
  vt::Tensor cache = cos_sin_cache();
  forward(queue, positions, query, key, cache);
}

RotaryEmbedding::RotaryEmbedding(int64_t head_size, int64_t rotary_dim,
                                 int64_t max_position_embeddings, double base,
                                 bool is_neox_style, vt::DType dtype)
    : RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base,
                      is_neox_style, dtype, true) {}

RotaryEmbedding::RotaryEmbedding(int64_t head_size, int64_t rotary_dim,
                                 int64_t max_position_embeddings, double base,
                                 bool is_neox_style, vt::DType dtype,
                                 bool init_cache)
    : RotaryEmbeddingBase(head_size, rotary_dim, max_position_embeddings, base,
                          is_neox_style, dtype, init_cache) {}

namespace {

struct RopeCacheKey {
  int64_t head_size;
  int64_t rotary_dim;
  int64_t max_position;
  bool is_neox_style;
  RopeParameters rope_parameters;
  vt::DType dtype;

  friend bool operator==(const RopeCacheKey&, const RopeCacheKey&) = default;
};

int64_t EffectiveRotaryDim(int64_t head_size,
                           const RopeParameters& parameters) {
  if (parameters.rope_dim.has_value() && *parameters.rope_dim != 0) {
    return *parameters.rope_dim;
  }
  if (!(parameters.partial_rotary_factor > 0.0) ||
      parameters.partial_rotary_factor > 1.0) {
    throw std::invalid_argument(
        "partial_rotary_factor must be between 0.0 and 1.0");
  }
  return static_cast<int64_t>(
      static_cast<double>(head_size) * parameters.partial_rotary_factor);
}

}  // namespace

std::shared_ptr<RotaryEmbeddingBase> get_rope(
    int64_t head_size, int64_t max_position, bool is_neox_style,
    const RopeParameters& rope_parameters, vt::DType dtype) {
  const int64_t rotary_dim =
      EffectiveRotaryDim(head_size, rope_parameters);
  const RopeCacheKey key{head_size, rotary_dim, max_position, is_neox_style,
                         rope_parameters, dtype};

  static std::mutex cache_mutex;
  static std::vector<
      std::pair<RopeCacheKey, std::shared_ptr<RotaryEmbeddingBase>>>
      rope_cache;
  std::lock_guard<std::mutex> lock(cache_mutex);
  const auto found = std::find_if(
      rope_cache.begin(), rope_cache.end(),
      [&](const auto& item) { return item.first == key; });
  if (found != rope_cache.end()) return found->second;

  std::shared_ptr<RotaryEmbeddingBase> embedding;
  if (rope_parameters.rope_type == "default") {
    if (!rope_parameters.mrope_section.empty()) {
      embedding = std::make_shared<MRotaryEmbedding>(
          head_size, rotary_dim, max_position, rope_parameters.rope_theta,
          is_neox_style, dtype, rope_parameters.mrope_section,
          rope_parameters.mrope_interleaved);
    } else {
      embedding = std::make_shared<RotaryEmbedding>(
          head_size, rotary_dim, max_position, rope_parameters.rope_theta,
          is_neox_style, dtype);
    }
  } else if (rope_parameters.rope_type == "llama3") {
    if (!rope_parameters.factor.has_value() ||
        !rope_parameters.low_freq_factor.has_value() ||
        !rope_parameters.high_freq_factor.has_value() ||
        !rope_parameters.original_max_position_embeddings.has_value()) {
      throw std::invalid_argument(
          "Llama 3 RoPE requires factor, low_freq_factor, high_freq_factor, "
          "and original_max_position_embeddings");
    }
    embedding = std::make_shared<Llama3RotaryEmbedding>(
        head_size, rotary_dim, max_position, rope_parameters.rope_theta,
        is_neox_style, dtype, *rope_parameters.factor,
        *rope_parameters.low_freq_factor,
        *rope_parameters.high_freq_factor,
        *rope_parameters.original_max_position_embeddings);
  } else if (rope_parameters.rope_type == "yarn") {
    if (!rope_parameters.factor.has_value() ||
        !rope_parameters.original_max_position_embeddings.has_value()) {
      throw std::invalid_argument(
          "YaRN requires factor and original_max_position_embeddings");
    }
    const double factor = *rope_parameters.factor;
    const int64_t original_max =
        *rope_parameters.original_max_position_embeddings;
    if (!rope_parameters.mrope_section.empty()) {
      // Pinned get_rope deliberately drops apply_yarn_scaling for MRoPE.
      embedding = std::make_shared<MRotaryEmbedding>(
          head_size, rotary_dim, original_max, rope_parameters.rope_theta,
          is_neox_style, dtype, rope_parameters.mrope_section,
          rope_parameters.mrope_interleaved, factor,
          rope_parameters.extrapolation_factor, rope_parameters.attn_factor,
          rope_parameters.beta_fast, rope_parameters.beta_slow,
          rope_parameters.truncate);
    } else {
      embedding = std::make_shared<YaRNScalingRotaryEmbedding>(
          head_size, rotary_dim, original_max, rope_parameters.rope_theta,
          is_neox_style, factor, dtype,
          rope_parameters.extrapolation_factor, rope_parameters.attn_factor,
          rope_parameters.beta_fast, rope_parameters.beta_slow,
          rope_parameters.apply_yarn_scaling, rope_parameters.truncate);
    }
  } else {
    throw std::invalid_argument("unknown RoPE scaling type " +
                                rope_parameters.rope_type);
  }

  rope_cache.emplace_back(key, embedding);
  return embedding;
}

}  // namespace vllm
