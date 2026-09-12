// Compiled vLLM residual expression on ROCm. Legacy RmsNorm is unchanged.
// Ported from vllm/ir/ops/layernorm.py:44-62 @
// e126687a9a828d513c01a07cd69f025f27d63280. The executing generated kernels
// ckic6h6:33-52, ctj2x6:33-58, and c5slugd:33-56 fix the order and stores.
// See .agents/specs/rocm-residual-norm.md for the measured compiled lifetime.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {
constexpr int kBlock = 256;

void Check(hipError_t status, const char* operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("rocm residual_rmsnorm: ") + operation + ": " +
                             hipGetErrorString(status));
}

class QueueDeviceScope {
 public:
  explicit QueueDeviceScope(const Queue& queue) {
    auto stream = static_cast<hipStream_t>(queue.handle);
    bool capturing = false;
    if (stream != nullptr) {
      hipDevice_t stream_device = -1;
      Check(hipStreamGetDevice(stream, &stream_device), "read stream device");
      VT_CHECK(stream_device == queue.device.index, "residual_rmsnorm: stream/queue device mismatch");
      hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
      Check(hipStreamIsCapturing(stream, &status), "read capture status");
      capturing = status != hipStreamCaptureStatusNone;
    }
    Check(hipGetDevice(&previous_), "read current device");
    // During capture the validated stream owns the launch. HIP forbids changing
    // the current device during capture. Outside capture, restore it on return.
    changed_ = !capturing && previous_ != queue.device.index;
    if (changed_) Check(hipSetDevice(queue.device.index), "select queue device");
  }
  ~QueueDeviceScope() { if (changed_) (void)hipSetDevice(previous_); }
 private:
  int previous_ = 0;
  bool changed_ = false;
};

__device__ float Expression(const __hip_bfloat16* a, const __hip_bfloat16* base,
                             const __hip_bfloat16* delta, int64_t column) {
  const float attention = __bfloat162float(a[column]) + __bfloat162float(base[column]);
  return delta == nullptr ? attention : __bfloat162float(delta[column]) + attention;
}

__global__ void ResidualRmsNormRowKernel(
    __hip_bfloat16* out, const __hip_bfloat16* a, const __hip_bfloat16* base,
    const __hip_bfloat16* delta, const __hip_bfloat16* weight,
    __hip_bfloat16* residual_out, int64_t rows, int64_t width, int64_t out_stride,
    int64_t a_stride, int64_t base_stride, int64_t delta_stride, int64_t residual_stride,
    float epsilon) {
  __shared__ float partial[kBlock];  // FP32 row reduction scratch, never persistent storage
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto* arow = a + row * a_stride;
    const auto* brow = base + row * base_stride;
    const auto* mrow = delta == nullptr ? nullptr : delta + row * delta_stride;
    float acc = 0.0f;
    for (int64_t column = threadIdx.x; column < width; column += kBlock) {
      const float value = Expression(arow, brow, mrow, column);
      acc += value * value;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride /= 2) {
      if (static_cast<int>(threadIdx.x) < stride)
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      __syncthreads();
    }
    const float inverse = 1.0f / sqrtf(partial[0] / static_cast<float>(width) + epsilon);
    for (int64_t column = threadIdx.x; column < width; column += kBlock) {
      // All variance reads finish before any store. Each lane reloads its own
      // complete expression before writing either exact alias, including tails
      // and rows wider than one block. No lane writes another lane's inputs.
      const float value = Expression(arow, brow, mrow, column);
      const float gamma = __bfloat162float(weight[column]);
      out[row * out_stride + column] = __float2bfloat16((value * inverse) * gamma);
      if (residual_out != nullptr)
        residual_out[row * residual_stride + column] = __float2bfloat16(value);
    }
    __syncthreads();  // protect the reduction scratch when this block serves another row
  }
}

void ResidualRmsNormKernel(Queue& queue, Tensor& out, const Tensor& a, const Tensor& base,
                           const Tensor* delta, const Tensor& weight,
                           const ResidualRmsNormArgs& args, Tensor* residual_out) {
  QueueDeviceScope device(queue);
  const auto rows = static_cast<unsigned>(std::min<int64_t>(a.shape[0], 65535));
  ResidualRmsNormRowKernel<<<rows, kBlock, 0, static_cast<hipStream_t>(queue.handle)>>>(
      out.Ptr<__hip_bfloat16>(), a.Ptr<__hip_bfloat16>(), base.Ptr<__hip_bfloat16>(),
      delta == nullptr ? nullptr : delta->Ptr<__hip_bfloat16>(), weight.Ptr<__hip_bfloat16>(),
      residual_out == nullptr ? nullptr : residual_out->Ptr<__hip_bfloat16>(),
      a.shape[0], a.shape[1], out.stride[0], a.stride[0], base.stride[0],
      delta == nullptr ? 0 : delta->stride[0], residual_out == nullptr ? 0 : residual_out->stride[0],
      args.eps);
  Check(hipGetLastError(), "launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kResidualRmsNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ResidualRmsNormFn>(&ResidualRmsNormKernel)));
  }
} registrar;
}  // namespace
}  // namespace vt::rocm
