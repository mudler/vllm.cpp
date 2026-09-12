// Original storage adapter for vLLM's parameter-dtype contract at e126687a9a,
// weight_utils.py:1241. The conversion precedes ordinary ROCm GEMM dispatch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vt/device.h"

namespace vt::rocm {
struct F16ScratchStats {
  size_t capacity_bytes = 0;
  size_t retained_bytes = 0;
  size_t high_water_bytes = 0;
  size_t stream_count = 0;
  std::vector<uintptr_t> allocation_pointers;
};
F16ScratchStats GetF16ScratchStats();
void ResetF16ScratchCapture(const Queue& q);
void RetainF16ScratchCapture(const Queue& q, void* graph);
void ReleaseF16ScratchQueue(const Queue& q);
void ReleaseF16ScratchGraph(void* graph);
}  // namespace vt::rocm

#ifdef VT_ROCM_F16_CONVERSION_IMPLEMENTATION
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include "vt/rocm/rocm_f16_codec.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/tensor.h"

namespace vt::rocm::f16_detail {
inline void Check(hipError_t status, const char* what) {
  VT_CHECK(status == hipSuccess,
           std::string("ROCm F16 conversion: ") + what + ": " + hipGetErrorString(status));
}

// Queue slabs and graphs share allocation ownership. hipFree completes pending
// uses before releasing a final owner, including a replay on another queue.
struct Allocation {
  void* data = nullptr;
  size_t capacity = 0;
  int device = -1;
  ~Allocation() {
    if (!data) return;
    int previous = -1;
    const auto get = hipGetDevice(&previous);
    const auto bind = get == hipSuccess ? hipSetDevice(device) : get;
    const auto release = bind == hipSuccess ? hipFree(data) : bind;
    const auto restore = get == hipSuccess ? hipSetDevice(previous) : get;
    if (release != hipSuccess || restore != hipSuccess) {
      std::fprintf(stderr, "ROCm F16 scratch release failed: %s; restore: %s\n",
                   hipGetErrorString(release), hipGetErrorString(restore));
      std::terminate();
    }
  }
};
using Allocations = std::vector<std::shared_ptr<Allocation>>;
struct Scratch {
  std::mutex mutex;
  std::shared_ptr<Allocation> current;
  size_t high_water = 0;
  Allocations captured;
};
using ScratchKey = std::tuple<int, uint64_t, uintptr_t>;
struct ScratchPool {
  std::mutex mutex;
  std::map<ScratchKey, std::shared_ptr<Scratch>> streams;
  std::map<void*, Allocations> graphs;
};
inline ScratchPool& Pool() {
  // Only the CPU registry lives through runtime teardown. Queue and graph
  // destruction release their entries and device allocations explicitly.
  static auto* pool = new ScratchPool;
  return *pool;
}
inline ScratchKey Key(const Queue& q) {
  return {q.device.index, q.id, reinterpret_cast<uintptr_t>(q.handle)};
}
inline std::shared_ptr<Scratch> ForQueue(const Queue& q, bool create = true) {
  auto& pool = Pool();
  std::lock_guard<std::mutex> lock(pool.mutex);
  const auto key = Key(q);
  auto found = pool.streams.find(key);
  if (found != pool.streams.end()) return found->second;
  if (!create) return {};
  auto entry = std::make_shared<Scratch>();
  pool.streams.emplace(key, entry);
  return entry;
}
inline Allocations TakeCaptured(const Queue& q) {
  auto entry = ForQueue(q, false);
  if (!entry) return {};
  std::lock_guard<std::mutex> lock(entry->mutex);
  Allocations captured;
  captured.swap(entry->captured);
  return captured;
}
inline size_t Aligned(size_t bytes) {
  VT_CHECK(bytes <= SIZE_MAX - 255, "ROCm F16 conversion: size overflow");
  return (bytes + 255) & ~size_t{255};
}
inline size_t ConvertedBytes(const Tensor& tensor, DType dtype) {
  const auto count = static_cast<uint64_t>(tensor.Numel());
  VT_CHECK(count <= SIZE_MAX / SizeOf(dtype), "ROCm F16 conversion: size overflow");
  return Aligned(static_cast<size_t>(count) * SizeOf(dtype));
}

__global__ void Convert(void* dst, DType dst_dtype, const void* src, DType src_dtype,
                        int64_t rows, int64_t columns, int64_t row_stride,
                        bool round_bf16) {
  const int64_t count = rows * columns;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t offset = (i / columns) * row_stride + i % columns;
    float value = src_dtype == DType::kF32 ? static_cast<const float*>(src)[offset]
        : src_dtype == DType::kF16 ? DF16ToF32(static_cast<const uint16_t*>(src)[offset])
                                  : DBF16ToF32(static_cast<const uint16_t*>(src)[offset]);
    if (round_bf16) value = DBF16ToF32(DF32ToBF16(value));
    if (dst_dtype == DType::kF32) static_cast<float*>(dst)[i] = value;
    else static_cast<uint16_t*>(dst)[i] = DF32ToBF16(value);
  }
}
inline void ConvertOn(hipStream_t stream, Tensor& dst, const Tensor& src) {
  const auto grid = static_cast<unsigned>(std::min<int64_t>((src.Numel() + 255) / 256, 65535));
  Convert<<<grid, 256, 0, stream>>>(dst.data, dst.dtype, src.data, src.dtype,
      src.shape[0], src.shape[1], src.stride[0], src.weight_value_dtype == DType::kBF16);
  Check(hipGetLastError(), "launch");
}

inline bool NeedsAdaptation(const Tensor& out, const Tensor& a, const Tensor& b) {
  return b.weight_value_dtype || a.dtype != b.dtype ||
      (out.dtype == DType::kBF16 && a.dtype != DType::kBF16);
}

// The guard spans conversion, GEMM enqueue, and final output conversion. Calls
// sharing a queue cannot overwrite its temporary operands before GEMM uses them.
struct Prepared {
  std::shared_ptr<Scratch> owner;
  Scratch& scratch;
  std::unique_lock<std::mutex> lock;
  Tensor a, b, out;
  Tensor& destination;
  hipStream_t stream;

  Prepared(const Queue& q, Tensor& output, const Tensor& activation, const Tensor& weight)
      : owner(ForQueue(q)), scratch(*owner), lock(scratch.mutex), a(activation), b(weight),
        out(output), destination(output), stream(static_cast<hipStream_t>(q.handle)) {
    DType input_dtype = DType::kF32;
    if (a.dtype == DType::kBF16 &&
        (b.dtype == DType::kBF16 || b.weight_value_dtype == DType::kBF16))
      input_dtype = DType::kBF16;
    else if (a.dtype == DType::kF16 && b.dtype == DType::kF16 && !b.weight_value_dtype)
      input_dtype = DType::kF16;
    const bool cast_a = a.dtype != input_dtype;
    const bool cast_b = b.dtype != input_dtype || b.weight_value_dtype.has_value();
    const bool cast_out = out.dtype == DType::kBF16 && input_dtype != DType::kBF16;
    // F32 scratch is required for unsupported mixed input pairs or a BF16 output
    // paired with homogeneous F16/F32 inputs; it preserves F32 activations exactly.
    const size_t a_bytes = cast_a ? ConvertedBytes(a, input_dtype) : 0;
    const size_t b_bytes = cast_b ? ConvertedBytes(b, input_dtype) : 0;
    const size_t out_bytes = cast_out ? ConvertedBytes(out, DType::kF32) : 0;
    VT_CHECK(a_bytes <= SIZE_MAX - b_bytes && a_bytes + b_bytes <= SIZE_MAX - out_bytes,
             "ROCm F16 conversion: scratch size overflow");
    const size_t needed = a_bytes + b_bytes + out_bytes;
    const size_t previous_capacity = scratch.current ? scratch.current->capacity : 0;
    if (needed > previous_capacity) {
      VT_CHECK(!StreamIsCapturing(stream),
               "ROCm F16 conversion: warm this scratch size before graph capture");
      size_t capacity = std::max<size_t>(256, previous_capacity);
      while (capacity < needed) {
        VT_CHECK(capacity <= SIZE_MAX / 2, "ROCm F16 conversion: capacity overflow");
        capacity *= 2;
      }
      auto allocation = std::make_shared<Allocation>();
      allocation->capacity = capacity;
      allocation->device = q.device.index;
      Check(hipMalloc(&allocation->data, capacity), "scratch allocation");
      scratch.current = std::move(allocation);
    }
    scratch.high_water = std::max(scratch.high_water, needed);
    if (StreamIsCapturing(stream) && scratch.current &&
        std::find(scratch.captured.begin(), scratch.captured.end(), scratch.current) ==
            scratch.captured.end()) {
      scratch.captured.push_back(scratch.current);
    }
    auto* base = static_cast<uint8_t*>(scratch.current ? scratch.current->data : nullptr);
    if (cast_a) {
      a = Tensor::Contiguous(base, input_dtype, q.device, {a.shape[0], a.shape[1]});
      ConvertOn(stream, a, activation);
    }
    if (cast_b) {
      b = Tensor::Contiguous(base + a_bytes, input_dtype, q.device, {b.shape[0], b.shape[1]});
      ConvertOn(stream, b, weight);
    }
    if (cast_out)
      out = Tensor::Contiguous(base + a_bytes + b_bytes, DType::kF32, q.device,
                               {out.shape[0], out.shape[1]});
  }
  void Finish() {
    if (out.data != destination.data) ConvertOn(stream, destination, out);
  }
};
}  // namespace vt::rocm::f16_detail
#endif  // VT_ROCM_F16_CONVERSION_IMPLEMENTATION
