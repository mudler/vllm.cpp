// Native selected-row decode for BACKEND-ROCM-QUANT-GATHER (#3093).
// Primary: plugin d4c1f0d vocal_embeds.py:79-98. The plugin selects packed rows
// then decodes them. This HIP adaptation fuses both operations into caller output.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "vt/rocm/rocm_embedding_quant.h"
#include "vt/rocm/rocm_quant_dequant.h"

namespace vt::rocm {
namespace {

constexpr int kThreads = 128;
constexpr int64_t kMaxBlocks = 1024;

void Check(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: embedding_quant: ") + operation +
                             ": " + hipGetErrorString(error));
  }
}

class DeviceError {
 public:
  DeviceError() {
    Check(hipMalloc(reinterpret_cast<void**>(&pointer_), sizeof(EmbeddingQuantError)),
          "allocate bounds record");
  }
  ~DeviceError() {
    if (pointer_ != nullptr) (void)hipFree(pointer_);
  }
  DeviceError(const DeviceError&) = delete;
  DeviceError& operator=(const DeviceError&) = delete;
  EmbeddingQuantError* get() const { return pointer_; }
  void Free() {
    Check(hipFree(pointer_), "free bounds record");
    pointer_ = nullptr;
  }

 private:
  EmbeddingQuantError* pointer_ = nullptr;
};

// One thread decodes one selected block. The cap retains bounded launch size;
// the grid-stride loop covers every remaining block without truncating IDs.
template <typename Tout, typename Tid, class Codec>
__global__ void EmbeddingQuantGatherKernel(Tout* out, const uint8_t* table,
                                           const Tid* ids, int64_t total,
                                           int64_t blocks_per_row, int64_t width,
                                           int64_t vocab, size_t row_bytes,
                                           EmbeddingQuantError* error) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += step) {
    const int64_t token = index / blocks_per_row;
    const int64_t block = index - token * blocks_per_row;
    const int64_t id = static_cast<int64_t>(ids[token]);
    if (id < 0 || id >= vocab) {
      if (atomicCAS(&error->status, 0, 1) == 0) error->id = id;
      continue;  // Reject before reading any packed bytes for this ID.
    }
    const auto* packed = table + static_cast<size_t>(id) * row_bytes +
                         static_cast<size_t>(block) * Codec::kBytes;
    Codec::template Decode<Tout>(packed, out + token * width + block * Codec::kElems);
  }
}

// The same complete list defines both capability and dispatch. Registration
// enables the existing loader policy, whose device gate has no dtype argument.
#define VT_ROCM_GATHER_TYPES(X) \
  X(kQ4_0, DqQ4_0)             \
  X(kQ5_0, DqQ5_0)             \
  X(kQ8_0, DqQ8_0)             \
  X(kQ2_K, DqQ2_K)             \
  X(kQ3_K, DqQ3_K)             \
  X(kQ4_K, DqQ4_K)             \
  X(kQ5_K, DqQ5_K)             \
  X(kQ6_K, DqQ6_K)             \
  X(kQ8_K, DqQ8_K)             \
  X(kIQ2_XXS, DqIQ2_XXS)       \
  X(kIQ2_XS, DqIQ2_XS)         \
  X(kIQ3_XXS, DqIQ3_XXS)       \
  X(kIQ1_S, DqIQ1_S)           \
  X(kIQ4_NL, DqIQ4_NL)         \
  X(kIQ3_S, DqIQ3_S)           \
  X(kIQ2_S, DqIQ2_S)           \
  X(kIQ4_XS, DqIQ4_XS)         \
  X(kMXFP4, DqMXFP4)           \
  X(kIQ1_XXXS, DqIQ1_XXXS)

template <typename Tout, typename Tid>
void Launch(hipStream_t stream, Tensor& out, const Tensor& table, const Tensor& ids,
            int64_t total, int64_t blocks_per_row, size_t row_bytes,
            EmbeddingQuantError* error) {
  const auto grid = static_cast<unsigned>(std::min((total - 1) / kThreads + 1, kMaxBlocks));
  switch (table.dtype) {
#define VT_ROCM_GATHER_LAUNCH(DT, CODEC)                                             \
    case DType::DT:                                                                 \
      VT_CHECK(BlockBytes(table.dtype) == CODEC::kBytes &&                           \
                   BlockElems(table.dtype) == CODEC::kElems,                       \
                 "rocm embedding_quant: codec geometry mismatch");                 \
      EmbeddingQuantGatherKernel<Tout, Tid, CODEC><<<grid, kThreads, 0, stream>>>(    \
          static_cast<Tout*>(out.data), static_cast<const uint8_t*>(table.data),    \
          static_cast<const Tid*>(ids.data), total, blocks_per_row, table.shape[1], \
          table.shape[0], row_bytes, error);                                        \
      break;
    VT_ROCM_GATHER_TYPES(VT_ROCM_GATHER_LAUNCH)
#undef VT_ROCM_GATHER_LAUNCH
    default:
      throw std::runtime_error("rocm embedding_quant: unsupported table dtype");
  }
  Check(hipGetLastError(), "launch");
}

}  // namespace

bool EmbeddingQuantSupported(DType dtype) {
  switch (dtype) {
#define VT_ROCM_GATHER_SUPPORTED(DT, CODEC) case DType::DT: return true;
    VT_ROCM_GATHER_TYPES(VT_ROCM_GATHER_SUPPORTED)
#undef VT_ROCM_GATHER_SUPPORTED
    default: return false;
  }
}

void EmbeddingQuantKernelRocm(Queue& queue, Tensor& out, const Tensor& table,
                              const Tensor& ids) {
  // vt::Embedding validates ranks, shapes, dtypes, layout, and device equality.
  VT_CHECK(EmbeddingQuantSupported(table.dtype), "rocm embedding_quant: unsupported table dtype");
  const int64_t tokens = ids.shape[0];
  const int64_t width = table.shape[1];
  const int64_t vocab = table.shape[0];
  VT_CHECK(tokens >= 0 && width >= 0 && vocab >= 0, "rocm embedding_quant: negative shape");
  if (tokens == 0 || width == 0) return;
  VT_CHECK(vocab > 0, "rocm embedding_quant: empty table (vocab 0)");
  VT_CHECK(tokens <= std::numeric_limits<int64_t>::max() / width,
             "rocm embedding_quant: output element count overflow");
  const int64_t blocks_per_row = width / BlockElems(table.dtype);
  const int64_t total = tokens * blocks_per_row;
  const size_t row_bytes = RowSizeBytes(table.dtype, width);
  VT_CHECK(row_bytes <= std::numeric_limits<size_t>::max() / static_cast<size_t>(vocab),
             "rocm embedding_quant: table byte count overflow");

  // Allocation, launch, and error cleanup belong to the queue's device even
  // when the caller most recently selected another device on this host thread.
  Check(hipSetDevice(queue.device.index), "bind queue device");
  const auto stream = static_cast<hipStream_t>(queue.handle);
  DeviceError error;
  Check(hipMemsetAsync(error.get(), 0, sizeof(EmbeddingQuantError), stream), "clear bounds record");
  // f32 is an explicit caller-selected primitive mode. Model gathers use bf16.
  if (out.dtype == DType::kF32) {
    if (ids.dtype == DType::kI32)
      Launch<float, int32_t>(stream, out, table, ids, total, blocks_per_row, row_bytes, error.get());
    else
      Launch<float, int64_t>(stream, out, table, ids, total, blocks_per_row, row_bytes, error.get());
  } else {
    if (ids.dtype == DType::kI32)
      Launch<uint16_t, int32_t>(stream, out, table, ids, total, blocks_per_row, row_bytes, error.get());
    else
      Launch<uint16_t, int64_t>(stream, out, table, ids, total, blocks_per_row, row_bytes, error.get());
  }
  EmbeddingQuantError host_error{};
  Check(hipMemcpyAsync(&host_error, error.get(), sizeof(host_error), hipMemcpyDeviceToHost, stream),
        "copy bounds record");
  Check(hipStreamSynchronize(stream), "synchronize bounds record");
  error.Free();
  if (host_error.status != 0) {
    throw std::runtime_error("vt rocm: embedding_quant: id " + std::to_string(host_error.id) +
                             " out of range [0, " + std::to_string(vocab) + ")");
  }
}

#undef VT_ROCM_GATHER_TYPES

}  // namespace vt::rocm
