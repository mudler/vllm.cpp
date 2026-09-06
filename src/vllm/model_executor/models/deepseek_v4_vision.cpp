#include "vllm/model_executor/models/deepseek_v4_vision.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/ops.h"

namespace vllm::multimodal {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::Reshape;
using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

constexpr size_t kGeometryCacheCapacity = 8;

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument(message);
}

int64_t CheckedMul(int64_t left, int64_t right, const char* message) {
  if (left < 0 || right < 0 ||
      (left != 0 && right > std::numeric_limits<int64_t>::max() / left)) {
    throw std::overflow_error(message);
  }
  return left * right;
}

void ValidateConfig(const DeepSeekV4VisionConfig& config) {
  if (config.patch_size <= 0) Invalid("DeepSeek-V4 vision patch size must be positive");
  if (config.hidden_size <= 0) Invalid("DeepSeek-V4 vision hidden size must be positive");
  if (config.num_heads <= 0 || config.hidden_size % config.num_heads != 0) {
    Invalid("DeepSeek-V4 vision head count must divide hidden size");
  }
  if (config.head_dim() % 4 != 0) {
    Invalid("DeepSeek-V4 vision head dimension must be divisible by four");
  }
  if (config.depth <= 0) Invalid("DeepSeek-V4 vision depth must be positive");
  if (config.intermediate_size <= 0) {
    Invalid("DeepSeek-V4 vision intermediate size must be positive");
  }
  if (config.output_size <= 0) Invalid("DeepSeek-V4 vision output size must be positive");
  if (config.downsample_ratio <= 0) {
    Invalid("DeepSeek-V4 vision downsample ratio must be positive");
  }
  if (!(config.norm_epsilon > 0.0f) || !std::isfinite(config.norm_epsilon)) {
    Invalid("DeepSeek-V4 vision norm epsilon must be finite and positive");
  }
  if (!(config.rope_theta > 0.0) || !std::isfinite(config.rope_theta)) {
    Invalid("DeepSeek-V4 vision RoPE theta must be finite and positive");
  }
  if (config.compute_dtype != DType::kBF16) {
    Invalid("DeepSeek-V4 vision compute dtype must be bf16");
  }
  const int64_t patch_square = CheckedMul(config.patch_size, config.patch_size,
                                           "DeepSeek-V4 vision patch dimension overflow");
  static_cast<void>(CheckedMul(3, patch_square,
                               "DeepSeek-V4 vision patch dimension overflow"));
  const int64_t ratio_square =
      CheckedMul(config.downsample_ratio, config.downsample_ratio,
                 "DeepSeek-V4 vision aligner dimension overflow");
  static_cast<void>(CheckedMul(config.hidden_size, ratio_square,
                               "DeepSeek-V4 vision aligner dimension overflow"));
}

void ValidateTensor(const Tensor& tensor, DType dtype,
                    const std::vector<int64_t>& shape, const char* name) {
  if (tensor.dtype != dtype) Invalid(std::string(name) + " has the wrong dtype");
  if (tensor.rank != static_cast<int>(shape.size())) {
    Invalid(std::string(name) + " has the wrong rank");
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (tensor.shape[i] != shape[i]) Invalid(std::string(name) + " has the wrong shape");
  }
  if (!tensor.IsContiguous()) Invalid(std::string(name) + " must be contiguous");
  if (tensor.data == nullptr) Invalid(std::string(name) + " has no storage");
}

void ValidateWeights(const DeepSeekV4VisionConfig& config,
                     const DeepSeekV4VisionWeights& weights) {
  if (weights.blocks.size() != static_cast<size_t>(config.depth)) {
    Invalid("DeepSeek-V4 vision block count does not match depth");
  }
  const int64_t hidden = config.hidden_size;
  const int64_t intermediate = config.intermediate_size;
  const int64_t output = config.output_size;
  ValidateTensor(weights.patch_weight, config.compute_dtype,
                 {hidden, config.patch_dim()}, "DeepSeek-V4 vision patch weight");
  ValidateTensor(weights.patch_bias, config.compute_dtype, {hidden},
                 "DeepSeek-V4 vision patch bias");
  const vt::Device device = weights.patch_weight.device;
  if (!(weights.patch_bias.device == device)) {
    Invalid("DeepSeek-V4 vision weights must share one device");
  }
  for (const DeepSeekV4VisionBlockWeights& block : weights.blocks) {
    ValidateTensor(block.norm1_weight, DType::kF32, {hidden},
                   "DeepSeek-V4 vision norm1 weight");
    ValidateTensor(block.qkv_weight, config.compute_dtype, {3 * hidden, hidden},
                   "DeepSeek-V4 vision qkv weight");
    ValidateTensor(block.qkv_bias, config.compute_dtype, {3 * hidden},
                   "DeepSeek-V4 vision qkv bias");
    ValidateTensor(block.out_weight, config.compute_dtype, {hidden, hidden},
                   "DeepSeek-V4 vision attention output weight");
    ValidateTensor(block.out_bias, config.compute_dtype, {hidden},
                   "DeepSeek-V4 vision attention output bias");
    ValidateTensor(block.norm2_weight, DType::kF32, {hidden},
                   "DeepSeek-V4 vision norm2 weight");
    ValidateTensor(block.mlp_w1_weight, config.compute_dtype,
                   {2 * intermediate, hidden}, "DeepSeek-V4 vision MLP w1 weight");
    ValidateTensor(block.mlp_w2_weight, config.compute_dtype,
                   {hidden, intermediate}, "DeepSeek-V4 vision MLP w2 weight");
    const Tensor* tensors[] = {
        &block.norm1_weight, &block.qkv_weight, &block.qkv_bias,
        &block.out_weight, &block.out_bias, &block.norm2_weight,
        &block.mlp_w1_weight, &block.mlp_w2_weight,
    };
    for (const Tensor* tensor : tensors) {
      if (!(tensor->device == device)) Invalid("DeepSeek-V4 vision weights must share one device");
    }
  }
  ValidateTensor(weights.final_norm_weight, DType::kF32, {hidden},
                 "DeepSeek-V4 vision final norm weight");
  ValidateTensor(weights.aligner_w1_weight, config.compute_dtype,
                 {output, config.aligner_input_size()},
                 "DeepSeek-V4 vision aligner w1 weight");
  ValidateTensor(weights.aligner_w1_bias, config.compute_dtype, {output},
                 "DeepSeek-V4 vision aligner w1 bias");
  ValidateTensor(weights.aligner_w2_weight, config.compute_dtype,
                 {output, output}, "DeepSeek-V4 vision aligner w2 weight");
  ValidateTensor(weights.aligner_w2_bias, config.compute_dtype, {output},
                 "DeepSeek-V4 vision aligner w2 bias");
  const Tensor* tail[] = {
      &weights.final_norm_weight, &weights.aligner_w1_weight,
      &weights.aligner_w1_bias, &weights.aligner_w2_weight,
      &weights.aligner_w2_bias,
  };
  for (const Tensor* tensor : tail) {
    if (!(tensor->device == device)) Invalid("DeepSeek-V4 vision weights must share one device");
  }
}

Tensor RowSlice(const Tensor& source, int64_t first_row, int64_t rows) {
  Tensor result = source;
  result.data = static_cast<char*>(source.data) +
                static_cast<size_t>(first_row * source.stride[0]) * vt::SizeOf(source.dtype);
  result.shape[0] = rows;
  return result;
}

Tensor VectorSlice(const Tensor& source, int64_t first, int64_t size) {
  Tensor result = source;
  result.data = static_cast<char*>(source.data) +
                static_cast<size_t>(first) * vt::SizeOf(source.dtype);
  result.shape[0] = size;
  return result;
}

void LinearBias(Queue& queue, Tensor& output, const Tensor& input,
                const Tensor& weight, const Tensor& bias) {
  vt::MatmulBT(queue, output, input, weight);
  vt::Add(queue, output, output, bias);
}

void CopyTensor(Backend& backend, Queue& queue, Tensor* destination,
                const Tensor& source) {
  if (destination == nullptr || destination->data == source.data) return;
  backend.Copy(queue, destination->data, source.data,
               static_cast<size_t>(source.Numel()) * vt::SizeOf(source.dtype));
}

// Record one internal scratch buffer's declared dtype. See
// DeepSeekV4VisionScratchDType: a value gate cannot see a buffer that is too
// wide, so the memory format is reported and asserted separately.
void RecordScratch(DeepSeekV4VisionCapture* capture, const char* name,
                   const Tensor& buffer) {
  if (capture == nullptr || capture->scratch_dtypes == nullptr) return;
  capture->scratch_dtypes->push_back({name, buffer.dtype});
}

void ValidateCaptureTensor(const Tensor* tensor, DType dtype,
                           const std::vector<int64_t>& shape,
                           vt::Device device, const char* name) {
  if (tensor == nullptr) return;
  ValidateTensor(*tensor, dtype, shape, name);
  if (!(tensor->device == device)) Invalid(std::string(name) + " is on the wrong device");
}

struct PersistentTensor {
  Backend* backend = nullptr;
  void* pointer = nullptr;
  Tensor tensor{};

  PersistentTensor() = default;
  PersistentTensor(const PersistentTensor&) = delete;
  PersistentTensor& operator=(const PersistentTensor&) = delete;
  PersistentTensor(PersistentTensor&& other) noexcept { *this = std::move(other); }
  PersistentTensor& operator=(PersistentTensor&& other) noexcept {
    if (this != &other) {
      Reset();
      backend = other.backend;
      pointer = other.pointer;
      tensor = other.tensor;
      other.backend = nullptr;
      other.pointer = nullptr;
    }
    return *this;
  }
  ~PersistentTensor() { Reset(); }

  void Reset() {
    if (backend != nullptr && pointer != nullptr) backend->Free(pointer);
    backend = nullptr;
    pointer = nullptr;
    tensor = {};
  }

  static PersistentTensor Upload(Backend& backend, Queue& queue, DType dtype,
                                 const std::vector<int64_t>& shape,
                                 const void* host) {
    PersistentTensor result;
    result.backend = &backend;
    int64_t count = 1;
    for (const int64_t dimension : shape) {
      count = CheckedMul(count, dimension,
                         "DeepSeek-V4 vision cached tensor size overflow");
    }
    const size_t bytes = static_cast<size_t>(count) * vt::SizeOf(dtype);
    result.pointer = backend.Alloc(bytes == 0 ? 1 : bytes);
    result.tensor = dense_attn::MakeTensor(result.pointer, dtype, queue.device, shape);
    if (bytes != 0) backend.Copy(queue, result.pointer, host, bytes);
    return result;
  }
};

struct Geometry {
  vt::Device device;
  int64_t height = 0;
  int64_t width = 0;
  PersistentTensor rope;
  PersistentTensor positions;
  PersistentTensor unfold_indices;
};

OwnedTensor BorrowResidentWeight(const Tensor& tensor) {
  OwnedTensor weight;
  weight.dtype = tensor.dtype;
  weight.rank = tensor.rank;
  for (int index = 0; index < tensor.rank; ++index) {
    weight.shape[index] = tensor.shape[index];
  }
  weight.nk = true;
  // CARRY THE LOAD-TIME STORAGE-LAYOUT MARKERS. `repacked`, `q8_0_aligned` and
  // `elem_kn_repacked` say the bytes were rewritten at load into a different
  // block interleave or orientation; the byte count and the [N,K] shape are
  // unchanged, so a borrow that copies dtype, rank, shape and bytes and stops
  // there produces a weight that is wrong only in how the kernel decodes it.
  //
  // This is the defect `main` fixed at `7a937db8a` (#2031) in the SHARED
  // `dense_attn::ResidentWeight`, re-introduced here in a private helper. There
  // it cost a debugging campaign: an i8mm-interleaved `block_q8_0x4` buffer
  // (136-byte blocks) reached `kMatmulBTQuant` flagged as flat `q8_0` (34-byte
  // blocks) and decoded to NaN, then all-zero logits, then token id 0, with no
  // crash and no refusal, because the `lm_head` GEMM swallowed the NaN.
  //
  // PROPAGATE RATHER THAN REFUSE. A fail-closed check here would reject the
  // repacked weight instead of decoding it, which removes the CPU i8mm fast
  // path rather than fixing the loss. `ResidentWeight` re-reads `repacked` and
  // `elem_kn_repacked` off this OwnedTensor on its host-alias arm and keeps its
  // own audit guard for the device-staging arm, so the markers only have to
  // survive the borrow.
  //
  // Host-conditional and therefore invisible here: `vt::cpu::QuantRepackActive()`
  // is true only on an aarch64 i8mm host. W3A's mmproj reader is what makes it a
  // live trigger rather than a latent one, because it can now hand this tower
  // block-quantized weights.
  weight.repacked = tensor.repacked;
  weight.q8_0_aligned = tensor.q8_0_aligned;
  weight.elem_kn_repacked = tensor.elem_kn_repacked;
  const size_t bytes =
      static_cast<size_t>(tensor.Numel()) * vt::SizeOf(tensor.dtype);
  auto keep_alive =
      std::shared_ptr<const void>(tensor.data, [](const void*) {});
  weight.bytes = OwnedBytes::Borrow(
      static_cast<const uint8_t*>(tensor.data), bytes, std::move(keep_alive));
  // The public weight tensor already resides on its declared device. Alias it
  // without taking ownership so ResidentWeight never stages a second copy.
  weight.d_dev = std::shared_ptr<void>(tensor.data, [](void*) {});
  return weight;
}

}  // namespace

int64_t DeepSeekV4VisionConfig::aligned_rows(int64_t height, int64_t width) const {
  if (height <= 0 || width <= 0) {
    Invalid("DeepSeek-V4 vision grid dimensions must be positive");
  }
  if (downsample_ratio <= 0) {
    Invalid("DeepSeek-V4 vision downsample ratio must be positive");
  }
  const int64_t rows = 1 + (height - 1) / downsample_ratio;
  const int64_t columns = 1 + (width - 1) / downsample_ratio;
  return CheckedMul(rows, columns, "DeepSeek-V4 vision aligned row count overflow");
}

void DeepSeekV4VisionRopeCosSin(int64_t height, int64_t width,
                                const DeepSeekV4VisionConfig& config,
                                std::vector<float>* cosine,
                                std::vector<float>* sine) {
  if (cosine == nullptr || sine == nullptr) {
    Invalid("DeepSeek-V4 vision RoPE outputs must not be null");
  }
  ValidateConfig(config);
  if (height <= 0 || width <= 0) {
    Invalid("DeepSeek-V4 vision grid dimensions must be positive");
  }
  const int64_t tokens = CheckedMul(height, width,
                                    "DeepSeek-V4 vision patch count overflow");
  const int64_t rope_width = config.head_dim() / 2;
  const int64_t frequencies = rope_width / 2;
  const int64_t rope_values = CheckedMul(
      tokens, rope_width, "DeepSeek-V4 vision RoPE table size overflow");
  cosine->resize(static_cast<size_t>(rope_values));
  sine->resize(static_cast<size_t>(rope_values));
  std::vector<float> inverse(static_cast<size_t>(frequencies));
  for (int64_t index = 0; index < frequencies; ++index) {
    const double exponent = static_cast<double>(2 * index) /
                            static_cast<double>(rope_width);
    inverse[static_cast<size_t>(index)] =
        static_cast<float>(1.0 / std::pow(config.rope_theta, exponent));
  }
  for (int64_t row = 0; row < height; ++row) {
    for (int64_t column = 0; column < width; ++column) {
      const int64_t token = row * width + column;
      for (int axis = 0; axis < 2; ++axis) {
        const float position = static_cast<float>(axis == 0 ? row : column);
        for (int64_t index = 0; index < frequencies; ++index) {
          const size_t offset = static_cast<size_t>(token * rope_width +
                                                    axis * frequencies + index);
          const float angle = position * inverse[static_cast<size_t>(index)];
          (*cosine)[offset] = std::cos(angle);
          (*sine)[offset] = std::sin(angle);
        }
      }
    }
  }
}

class DeepSeekV4Vision::Impl {
 public:
  Impl(Backend& backend, DeepSeekV4VisionConfig config,
       DeepSeekV4VisionWeights weights)
      : backend_(backend), config_(std::move(config)), weights_(std::move(weights)) {
    ValidateConfig(config_);
    ValidateWeights(config_, weights_);
    mlp_gate_up_weights_.reserve(weights_.blocks.size());
    for (const DeepSeekV4VisionBlockWeights& block : weights_.blocks) {
      mlp_gate_up_weights_.push_back(
          BorrowResidentWeight(block.mlp_w1_weight));
    }
  }

  const DeepSeekV4VisionConfig& config() const { return config_; }
  Backend& backend() { return backend_; }

  size_t cached_geometry_count() const { return geometries_.size(); }

  DeepSeekV4VisionStorageMarkers mlp_gate_up_markers(int64_t block) const {
    if (block < 0 ||
        block >= static_cast<int64_t>(mlp_gate_up_weights_.size())) {
      Invalid("DeepSeek-V4 vision block index is out of range");
    }
    const OwnedTensor& weight =
        mlp_gate_up_weights_[static_cast<size_t>(block)];
    DeepSeekV4VisionStorageMarkers markers;
    markers.repacked = weight.repacked;
    markers.q8_0_aligned = weight.q8_0_aligned;
    markers.elem_kn_repacked = weight.elem_kn_repacked;
    return markers;
  }

  Geometry& GeometryFor(Queue& queue, int64_t height, int64_t width) {
    for (size_t i = 0; i < geometries_.size(); ++i) {
      const Geometry& candidate = geometries_[i];
      if (candidate.device == queue.device && candidate.height == height &&
          candidate.width == width) {
        if (i + 1 != geometries_.size()) {
          std::rotate(geometries_.begin() + static_cast<std::ptrdiff_t>(i),
                      geometries_.begin() + static_cast<std::ptrdiff_t>(i + 1),
                      geometries_.end());
        }
        return geometries_.back();
      }
    }

    if (geometries_.size() == kGeometryCacheCapacity) {
      // The model runner is queue-serial. Drain before releasing an evicted
      // entry so no provider can still read its device allocation.
      backend_.Synchronize(queue);
      geometries_.erase(geometries_.begin());
    }

    Geometry geometry;
    geometry.device = queue.device;
    geometry.height = height;
    geometry.width = width;
    const int64_t tokens = CheckedMul(height, width,
                                      "DeepSeek-V4 vision patch count overflow");
    const int64_t head_dim = config_.head_dim();
    const int64_t half = head_dim / 2;

    std::vector<float> cosine;
    std::vector<float> sine;
    DeepSeekV4VisionRopeCosSin(height, width, config_, &cosine, &sine);
    const int64_t rope_values = CheckedMul(
        tokens, head_dim, "DeepSeek-V4 vision RoPE cache size overflow");
    std::vector<float> rope(static_cast<size_t>(rope_values));
    for (int64_t token = 0; token < tokens; ++token) {
      std::memcpy(rope.data() + token * head_dim,
                  cosine.data() + token * half,
                  static_cast<size_t>(half) * sizeof(float));
      std::memcpy(rope.data() + token * head_dim + half,
                  sine.data() + token * half,
                  static_cast<size_t>(half) * sizeof(float));
    }
    // f32 is deliberate: pinned apply_rotary widens q/k before multiplying by
    // its f32 cos/sin table, then narrows once to the model dtype.
    geometry.rope = PersistentTensor::Upload(backend_, queue, DType::kF32,
                                              {tokens, head_dim}, rope.data());

    if (tokens > std::numeric_limits<int32_t>::max()) {
      Invalid("DeepSeek-V4 vision patch count exceeds i32 positions");
    }
    std::vector<int32_t> positions(static_cast<size_t>(tokens));
    for (int64_t token = 0; token < tokens; ++token) {
      positions[static_cast<size_t>(token)] = static_cast<int32_t>(token);
    }
    geometry.positions = PersistentTensor::Upload(
        backend_, queue, DType::kI32, {tokens}, positions.data());

    const int64_t ratio = config_.downsample_ratio;
    const int64_t aligned = config_.aligned_rows(height, width);
    const int64_t unfold_width = config_.aligner_input_size();
    const int64_t index_count = CheckedMul(
        aligned, unfold_width,
        "DeepSeek-V4 vision unfold index count overflow");
    if (CheckedMul(tokens + 1, config_.hidden_size,
                   "DeepSeek-V4 vision unfold source size overflow") >
        std::numeric_limits<int32_t>::max()) {
      Invalid("DeepSeek-V4 vision unfold source exceeds i32 indexing");
    }
    std::vector<int32_t> indices(static_cast<size_t>(index_count));
    int64_t destination = 0;
    const int64_t block_columns = 1 + (width - 1) / ratio;
    const int64_t block_rows = 1 + (height - 1) / ratio;
    for (int64_t block_row = 0; block_row < block_rows; ++block_row) {
      for (int64_t block_column = 0; block_column < block_columns; ++block_column) {
        for (int64_t channel = 0; channel < config_.hidden_size; ++channel) {
          for (int64_t local_row = 0; local_row < ratio; ++local_row) {
            for (int64_t local_column = 0; local_column < ratio; ++local_column) {
              const int64_t row = block_row * ratio + local_row;
              const int64_t column = block_column * ratio + local_column;
              const int64_t patch =
                  row < height && column < width ? row * width + column : tokens;
              indices[static_cast<size_t>(destination++)] = static_cast<int32_t>(
                  patch * config_.hidden_size + channel);
            }
          }
        }
      }
    }
    geometry.unfold_indices = PersistentTensor::Upload(
        backend_, queue, DType::kI32, {index_count}, indices.data());
    geometries_.push_back(std::move(geometry));
    return geometries_.back();
  }

  void ValidateQueue(Queue& queue) const {
    if (!(queue.device == weights_.patch_weight.device)) {
      Invalid("DeepSeek-V4 vision queue and weights must share one device");
    }
  }

  void ValidateVisionIo(Queue& queue, const Tensor& output,
                        const Tensor& patches, int64_t height,
                        int64_t width, DeepSeekV4VisionCapture* capture) const {
    ValidateQueue(queue);
    if (height <= 0 || width <= 0) {
      Invalid("DeepSeek-V4 vision grid dimensions must be positive");
    }
    const int64_t tokens = CheckedMul(height, width,
                                      "DeepSeek-V4 vision patch count overflow");
    if (patches.rank != 2 || patches.shape[0] != tokens ||
        patches.shape[1] != config_.patch_dim()) {
      Invalid("DeepSeek-V4 vision patches must be [height*width, patch_dim]");
    }
    if (patches.dtype != config_.compute_dtype) {
      Invalid("DeepSeek-V4 vision patch dtype must equal model dtype");
    }
    if (!patches.IsContiguous()) Invalid("DeepSeek-V4 vision patches must be contiguous");
    if (!(patches.device == queue.device)) {
      Invalid("DeepSeek-V4 vision patches and queue must share one device");
    }
    if (output.dtype != config_.compute_dtype) {
      Invalid("DeepSeek-V4 vision output dtype must equal model dtype");
    }
    ValidateTensor(output, config_.compute_dtype, {tokens, config_.hidden_size},
                   "DeepSeek-V4 vision encoder output");
    if (!(output.device == queue.device)) {
      Invalid("DeepSeek-V4 vision output and queue must share one device");
    }
    if (capture == nullptr) return;
    ValidateCaptureTensor(capture->patch_embedding, config_.compute_dtype,
                          {tokens, config_.hidden_size}, queue.device,
                          "DeepSeek-V4 vision patch capture");
    if (!capture->block_outputs.empty() &&
        capture->block_outputs.size() != static_cast<size_t>(config_.depth)) {
      Invalid("DeepSeek-V4 vision block capture count must match depth");
    }
    for (Tensor* block : capture->block_outputs) {
      ValidateCaptureTensor(block, config_.compute_dtype,
                            {tokens, config_.hidden_size}, queue.device,
                            "DeepSeek-V4 vision block capture");
    }
    ValidateCaptureTensor(capture->final_norm, config_.compute_dtype,
                          {tokens, config_.hidden_size}, queue.device,
                          "DeepSeek-V4 vision final norm capture");
  }

  void ValidateAlignerIo(Queue& queue, const Tensor& output,
                         const Tensor& vision, int64_t height,
                         int64_t width, DeepSeekV4VisionCapture* capture) const {
    ValidateQueue(queue);
    if (height <= 0 || width <= 0) {
      Invalid("DeepSeek-V4 vision grid dimensions must be positive");
    }
    const int64_t tokens = CheckedMul(height, width,
                                      "DeepSeek-V4 vision patch count overflow");
    ValidateTensor(vision, config_.compute_dtype, {tokens, config_.hidden_size},
                   "DeepSeek-V4 vision aligner input");
    if (!(vision.device == queue.device)) {
      Invalid("DeepSeek-V4 vision aligner input and queue must share one device");
    }
    if (output.dtype != config_.compute_dtype) {
      Invalid("DeepSeek-V4 vision output dtype must equal model dtype");
    }
    const int64_t rows = config_.aligned_rows(height, width);
    ValidateTensor(output, config_.compute_dtype, {rows, config_.output_size},
                   "DeepSeek-V4 vision output");
    if (!(output.device == queue.device)) {
      Invalid("DeepSeek-V4 vision output and queue must share one device");
    }
    if (capture == nullptr) return;
    ValidateCaptureTensor(capture->aligner_unfold, config_.compute_dtype,
                          {rows, config_.aligner_input_size()}, queue.device,
                          "DeepSeek-V4 vision unfold capture");
    ValidateCaptureTensor(capture->aligner_hidden, config_.compute_dtype,
                          {rows, config_.output_size}, queue.device,
                          "DeepSeek-V4 vision aligner hidden capture");
    ValidateCaptureTensor(capture->aligner_gelu, config_.compute_dtype,
                          {rows, config_.output_size}, queue.device,
                          "DeepSeek-V4 vision aligner GELU capture");
  }

  void VisionForward(Queue& queue, Tensor& output, const Tensor& patches,
                     int64_t height, int64_t width,
                     DeepSeekV4VisionCapture* capture) {
    ValidateVisionIo(queue, output, patches, height, width, capture);
    Geometry& geometry = GeometryFor(queue, height, width);
    Dev device{backend_, queue};
    const int64_t tokens = height * width;
    const int64_t hidden = config_.hidden_size;
    const int64_t heads = config_.num_heads;
    const int64_t head_dim = config_.head_dim();
    const int64_t intermediate = config_.intermediate_size;

    DBuf hidden_state(device, config_.compute_dtype, {tokens, hidden});
    LinearBias(queue, hidden_state.t(), patches, weights_.patch_weight,
               weights_.patch_bias);
    if (capture != nullptr) {
      CopyTensor(backend_, queue, capture->patch_embedding, hidden_state.t());
    }

    // Every per-layer scratch allocation is hoisted and reused for all blocks.
    DBuf normalized(device, config_.compute_dtype, {tokens, hidden});
    DBuf query(device, config_.compute_dtype, {tokens, hidden});
    DBuf key(device, config_.compute_dtype, {tokens, hidden});
    DBuf value(device, config_.compute_dtype, {tokens, hidden});
    DBuf attention(device, config_.compute_dtype, {tokens, hidden});
    DBuf projected(device, config_.compute_dtype, {tokens, hidden});
    // Gate-up scratch is owned by the mandatory MlpGateUpMethodBase seam. Its
    // pooled buffers return after each layer and create no repeat allocation.
    DBuf query_f32;
    DBuf key_f32;
    if (config_.compute_dtype == DType::kBF16) {
      // f32 scratch is required by pinned apply_rotary: q/k are widened before
      // the f32 cos/sin multiply and narrowed exactly once afterward.
      query_f32 = DBuf(device, DType::kF32, {tokens, hidden});
      key_f32 = DBuf(device, DType::kF32, {tokens, hidden});
    }

    // The memory format of the model path, in allocation order. Everything the
    // tower carries between operations is the model dtype; the two rotary
    // buffers are the only f32 entries and they have the reason above.
    if (capture != nullptr && capture->scratch_dtypes != nullptr) {
      capture->scratch_dtypes->clear();
    }
    RecordScratch(capture, "vision.hidden_state", hidden_state.t());
    RecordScratch(capture, "vision.normalized", normalized.t());
    RecordScratch(capture, "vision.query", query.t());
    RecordScratch(capture, "vision.key", key.t());
    RecordScratch(capture, "vision.value", value.t());
    RecordScratch(capture, "vision.attention", attention.t());
    RecordScratch(capture, "vision.projected", projected.t());
    if (config_.compute_dtype == DType::kBF16) {
      RecordScratch(capture, "vision.rope_query_f32", query_f32.t());
      RecordScratch(capture, "vision.rope_key_f32", key_f32.t());
    }

    vt::RopeArgs rope_args;
    rope_args.rotary_dim = static_cast<int>(head_dim);
    rope_args.is_neox_style = true;
    const vt::RmsNormArgs norm_args{config_.norm_epsilon, false};
    // VT-ATTN-NAIVE: W2 is the correctness rung. The generic public op keeps
    // CUDA, ROCm and Vulkan on this composition until a profile selects a rung.
    const vt::AttentionArgs attention_args{
        1.0f / std::sqrt(static_cast<float>(head_dim)), false};

    for (int64_t layer_index = 0; layer_index < config_.depth; ++layer_index) {
      const DeepSeekV4VisionBlockWeights& layer =
          weights_.blocks[static_cast<size_t>(layer_index)];
      vt::RmsNorm(queue, normalized.t(), hidden_state.t(), layer.norm1_weight,
                  norm_args);

      // Three views of the one checkpoint qkv Linear. Separate MatmulBT calls
      // avoid a merged-output split/copy and keep the composition provider-wide.
      const Tensor q_weight = RowSlice(layer.qkv_weight, 0, hidden);
      const Tensor k_weight = RowSlice(layer.qkv_weight, hidden, hidden);
      const Tensor v_weight = RowSlice(layer.qkv_weight, 2 * hidden, hidden);
      const Tensor q_bias = VectorSlice(layer.qkv_bias, 0, hidden);
      const Tensor k_bias = VectorSlice(layer.qkv_bias, hidden, hidden);
      const Tensor v_bias = VectorSlice(layer.qkv_bias, 2 * hidden, hidden);
      LinearBias(queue, query.t(), normalized.t(), q_weight, q_bias);
      LinearBias(queue, key.t(), normalized.t(), k_weight, k_bias);
      LinearBias(queue, value.t(), normalized.t(), v_weight, v_bias);

      Tensor query_heads;
      Tensor key_heads;
      if (config_.compute_dtype == DType::kBF16) {
        vt::CastF32(queue, query_f32.t(), query.t());
        vt::CastF32(queue, key_f32.t(), key.t());
        query_heads = Reshape(query_f32.t(), {tokens, heads, head_dim});
        key_heads = Reshape(key_f32.t(), {tokens, heads, head_dim});
        vt::RopeFromCache(queue, query_heads, &key_heads,
                          geometry.positions.tensor, geometry.rope.tensor,
                          rope_args);
        vt::CastBf16(queue, query.t(), query_f32.t());
        vt::CastBf16(queue, key.t(), key_f32.t());
        query_heads = Reshape(query.t(), {tokens, heads, head_dim});
        key_heads = Reshape(key.t(), {tokens, heads, head_dim});
      } else {
        query_heads = Reshape(query.t(), {tokens, heads, head_dim});
        key_heads = Reshape(key.t(), {tokens, heads, head_dim});
        vt::RopeFromCache(queue, query_heads, &key_heads,
                          geometry.positions.tensor, geometry.rope.tensor,
                          rope_args);
      }
      Tensor value_heads = Reshape(value.t(), {tokens, heads, head_dim});
      Tensor attention_heads = Reshape(attention.t(), {tokens, heads, head_dim});
      // VT-ATTN-NAIVE: W2 is the correctness rung, and the pinned tower is
      // non-causal over a whole image. The generic public op keeps CUDA, ROCm
      // and Vulkan on one composition until a profile selects a faster rung.
      vt::Attention(queue, attention_heads, query_heads, key_heads, value_heads,
                    attention_args);
      LinearBias(queue, projected.t(), attention.t(), layer.out_weight,
                 layer.out_bias);
      vt::Add(queue, hidden_state.t(), hidden_state.t(), projected.t());

      vt::RmsNorm(queue, normalized.t(), hidden_state.t(), layer.norm2_weight,
                  norm_args);
      layers::UnquantizedMlpGateUpMethod gate_up_method(
          &mlp_gate_up_weights_[static_cast<size_t>(layer_index)],
          intermediate);
      DBuf activated = gate_up_method.Apply(device, normalized.t());
      // The shared seam owns this buffer; record it once so its width is
      // asserted beside the buffers this file allocates.
      if (layer_index == 0) {
        RecordScratch(capture, "vision.mlp_gate_up_activated", activated.t());
      }
      vt::MatmulBT(queue, projected.t(), activated.t(), layer.mlp_w2_weight);
      vt::Add(queue, hidden_state.t(), hidden_state.t(), projected.t());

      if (capture != nullptr && !capture->block_outputs.empty()) {
        CopyTensor(backend_, queue,
                   capture->block_outputs[static_cast<size_t>(layer_index)],
                   hidden_state.t());
      }
    }

    vt::RmsNorm(queue, output, hidden_state.t(), weights_.final_norm_weight,
                norm_args);
    if (capture != nullptr) CopyTensor(backend_, queue, capture->final_norm, output);
  }

  void AlignerForward(Queue& queue, Tensor& output, const Tensor& vision,
                      int64_t height, int64_t width,
                      DeepSeekV4VisionCapture* capture) {
    ValidateAlignerIo(queue, output, vision, height, width, capture);
    Geometry& geometry = GeometryFor(queue, height, width);
    Dev device{backend_, queue};
    const int64_t tokens = height * width;
    const int64_t hidden = config_.hidden_size;
    const int64_t rows = config_.aligned_rows(height, width);
    const int64_t unfold_width = config_.aligner_input_size();

    // Append one all-zero patch. The cached scalar gather maps every spatial pad
    // cell to that row and emits torch F.unfold's [channel,dy,dx] order exactly.
    DBuf padded(device, config_.compute_dtype, {tokens + 1, hidden});
    padded.Zero(device);
    backend_.Copy(queue, padded.ptr(), vision.data,
                  static_cast<size_t>(tokens * hidden) *
                      vt::SizeOf(config_.compute_dtype));
    DBuf unfolded(device, config_.compute_dtype, {rows, unfold_width});
    Tensor padded_scalars = Reshape(padded.t(), {(tokens + 1) * hidden, 1});
    Tensor unfolded_scalars = Reshape(unfolded.t(), {rows * unfold_width, 1});
    vt::IndexSelect(queue, unfolded_scalars, padded_scalars,
                    geometry.unfold_indices.tensor);
    if (capture != nullptr) {
      CopyTensor(backend_, queue, capture->aligner_unfold, unfolded.t());
    }

    DBuf hidden_state(device, config_.compute_dtype,
                      {rows, config_.output_size});
    // Appended, never cleared: a whole Forward records the vision stage first
    // and then these, in one sequence. The aligner called on its own appends to
    // whatever the caller's vector already holds.
    RecordScratch(capture, "aligner.padded", padded.t());
    RecordScratch(capture, "aligner.unfolded", unfolded.t());
    RecordScratch(capture, "aligner.hidden_state", hidden_state.t());
    LinearBias(queue, hidden_state.t(), unfolded.t(),
               weights_.aligner_w1_weight, weights_.aligner_w1_bias);
    if (capture != nullptr) {
      CopyTensor(backend_, queue, capture->aligner_hidden, hidden_state.t());
    }
    vt::GeluErf(queue, hidden_state.t(), hidden_state.t());
    if (capture != nullptr) {
      CopyTensor(backend_, queue, capture->aligner_gelu, hidden_state.t());
    }
    LinearBias(queue, output, hidden_state.t(), weights_.aligner_w2_weight,
               weights_.aligner_w2_bias);
  }

 private:
  Backend& backend_;
  DeepSeekV4VisionConfig config_;
  DeepSeekV4VisionWeights weights_;
  std::vector<OwnedTensor> mlp_gate_up_weights_;
  std::vector<Geometry> geometries_;
};

DeepSeekV4Vision::DeepSeekV4Vision(Backend& backend,
                                   DeepSeekV4VisionConfig config,
                                   DeepSeekV4VisionWeights weights)
    : impl_(std::make_unique<Impl>(backend, std::move(config),
                                   std::move(weights))) {}

DeepSeekV4Vision::~DeepSeekV4Vision() = default;
DeepSeekV4Vision::DeepSeekV4Vision(DeepSeekV4Vision&&) noexcept = default;
DeepSeekV4Vision& DeepSeekV4Vision::operator=(DeepSeekV4Vision&&) noexcept = default;

const DeepSeekV4VisionConfig& DeepSeekV4Vision::config() const {
  return impl_->config();
}

void DeepSeekV4Vision::Forward(Queue& queue, Tensor& output,
                               const Tensor& patches, int64_t height,
                               int64_t width,
                               DeepSeekV4VisionCapture* capture) {
  impl_->ValidateQueue(queue);
  if (height <= 0 || width <= 0) {
    Invalid("DeepSeek-V4 vision grid dimensions must be positive");
  }
  const DeepSeekV4VisionConfig& config = impl_->config();
  const int64_t tokens = CheckedMul(height, width,
                                    "DeepSeek-V4 vision patch count overflow");
  static_cast<void>(CheckedMul(tokens, config.hidden_size,
                               "DeepSeek-V4 vision activation size overflow"));
  if (patches.rank != 2 || patches.shape[0] != tokens ||
      patches.shape[1] != config.patch_dim()) {
    Invalid("DeepSeek-V4 vision patches must be [height*width, patch_dim]");
  }
  if (patches.dtype != config.compute_dtype) {
    Invalid("DeepSeek-V4 vision patch dtype must equal model dtype");
  }
  if (!patches.IsContiguous()) {
    Invalid("DeepSeek-V4 vision patches must be contiguous");
  }
  if (!(patches.device == queue.device)) {
    Invalid("DeepSeek-V4 vision patches and queue must share one device");
  }
  if (output.dtype != config.compute_dtype) {
    Invalid("DeepSeek-V4 vision output dtype must equal model dtype");
  }
  ValidateTensor(output, config.compute_dtype,
                 {config.aligned_rows(height, width), config.output_size},
                 "DeepSeek-V4 vision output");
  if (!(output.device == queue.device)) {
    Invalid("DeepSeek-V4 vision output and queue must share one device");
  }
  Dev device{impl_->backend(), queue};
  DBuf vision(device, config.compute_dtype, {tokens, config.hidden_size});
  impl_->VisionForward(queue, vision.t(), patches, height, width, capture);
  // RECORDED AFTER THE VISION STAGE, not before it. This buffer is allocated
  // first, but the vision stage is what CLEARS the list, so a record here would
  // be erased; a review found it as the one scratch buffer the list never
  // described. Widening it is caught either way, by the `Tensor& output` shape
  // and dtype check `VisionForward` runs on it, but the list said it described
  // every buffer and it did not. It sits between the two stages in the sequence,
  // which is also where it sits in the dataflow.
  RecordScratch(capture, "forward.vision", vision.t());
  impl_->AlignerForward(queue, output, vision.t(), height, width, capture);
}

void DeepSeekV4Vision::VisionForward(Queue& queue, Tensor& output,
                                     const Tensor& patches, int64_t height,
                                     int64_t width,
                                     DeepSeekV4VisionCapture* capture) {
  impl_->VisionForward(queue, output, patches, height, width, capture);
}

void DeepSeekV4Vision::AlignerForward(Queue& queue, Tensor& output,
                                      const Tensor& vision, int64_t height,
                                      int64_t width,
                                      DeepSeekV4VisionCapture* capture) {
  impl_->AlignerForward(queue, output, vision, height, width, capture);
}

size_t DeepSeekV4Vision::cached_geometry_count() const {
  return impl_->cached_geometry_count();
}

DeepSeekV4VisionStorageMarkers DeepSeekV4Vision::mlp_gate_up_markers(
    int64_t block) const {
  return impl_->mlp_gate_up_markers(block);
}

}  // namespace vllm::multimodal
