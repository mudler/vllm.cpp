// BACKEND-ROCM-ATTN-GATE-SPLIT (#3106).
// vLLM e126687a9a qwen3_next.py:424-433 splits query/gate within each head.
// This copies the same values to the existing contiguous local destinations,
// following cpu_ops.cpp:3985 and cuda_glue.cu:193. The gate remains F32 because
// the shared SigmoidGateBf16 consumer requires that storage format.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"

namespace vt::rocm {
namespace {
constexpr int kBlock = 256;

__device__ inline float Load(const float* source, int64_t index) { return source[index]; }
__device__ inline float Load(const __hip_bfloat16* source, int64_t index) {
  return __bfloat162float(source[index]);
}
__device__ inline void Store(float* out, int64_t index, float value) { out[index] = value; }
__device__ inline void Store(__hip_bfloat16* out, int64_t index, float value) {
  out[index] = __float2bfloat16(value);
}

template <typename Input, typename Output>
__global__ void AttnGateSplitK(Output* query, float* gate, const Input* packed,
                              int64_t n, int64_t heads, int64_t width) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    const int64_t d = i % width;
    const int64_t head = (i / width) % heads;
    const int64_t token = i / (width * heads);
    const int64_t base = token * heads * 2 * width + head * 2 * width;
    Store(query, i, Load(packed, base + d));
    gate[i] = Load(packed, base + width + d);
  }
}

template <typename Input>
void Launch(Queue& q, Tensor& query, Tensor& gate, const Tensor& packed,
            int64_t n, int64_t heads, int64_t width) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  const unsigned grid = static_cast<unsigned>(blocks > 65535 ? 65535 : blocks);
  const auto stream = static_cast<hipStream_t>(q.handle);
  switch (query.dtype) {
    case DType::kF32:
      AttnGateSplitK<Input, float><<<grid, kBlock, 0, stream>>>(
          query.Ptr<float>(), gate.Ptr<float>(), packed.Ptr<Input>(), n, heads, width);
      break;
    case DType::kBF16:
      AttnGateSplitK<Input, __hip_bfloat16><<<grid, kBlock, 0, stream>>>(
          query.Ptr<__hip_bfloat16>(), gate.Ptr<float>(), packed.Ptr<Input>(), n, heads, width);
      break;
    default: VT_CHECK(false, "attn_gate_split: q_out must be f32 or bf16");
  }
}
}  // namespace

void AttnGateSplitKernelRocm(Queue& q, Tensor& query, Tensor& gate, const Tensor& packed) {
  EnsureQueueDevice(q);
  // The shared wrapper owns rank, shape, dtype, contiguity, and device validation.
  const int64_t n = query.Numel();
  if (n == 0) return;
  const int64_t heads = query.shape[1], width = query.shape[2];
  switch (packed.dtype) {
    case DType::kF32: Launch<float>(q, query, gate, packed, n, heads, width); break;
    case DType::kBF16: Launch<__hip_bfloat16>(q, query, gate, packed, n, heads, width); break;
    default: VT_CHECK(false, "attn_gate_split: qgate must be f32 or bf16");
  }
  const auto error = hipGetLastError();
  if (error != hipSuccess)
    throw std::runtime_error(std::string("vt rocm: attn_gate_split launch: ") +
                             hipGetErrorString(error));
}
}  // namespace vt::rocm
