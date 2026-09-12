// CPU reference for the compiled vLLM residual expression.
// Ported from vllm/ir/ops/layernorm.py:44-62 @
// e126687a9a828d513c01a07cd69f025f27d63280, with executing generated kernels
// ckic6h6:33-52, ctj2x6:33-58, and c5slugd:33-56 defining the BF16 boundaries.
#include <array>
#include <cmath>

#include "cpu_threadpool.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {
constexpr int kBlock = 256;

float Expression(const Tensor& a, const Tensor& base, const Tensor* delta,
                  int64_t row, int64_t column) {
  const float av = BF16ToF32(a.Ptr<uint16_t>()[row * a.stride[0] + column]);
  const float rv = BF16ToF32(base.Ptr<uint16_t>()[row * base.stride[0] + column]);
  const float attention = av + rv;
  if (delta == nullptr) return attention;
  const float mv = BF16ToF32(delta->Ptr<uint16_t>()[row * delta->stride[0] + column]);
  return mv + attention;
}

void ResidualRmsNormKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& base,
                           const Tensor* delta, const Tensor& weight,
                           const ResidualRmsNormArgs& args, Tensor* residual_out) {
  const int64_t rows = a.shape[0], width = a.shape[1];
  ParallelForRows(CurrentThreadpool(), rows, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      // Row-local FP32 reduction scratch mirrors the deterministic GPU tree.
      // No activation or residual is materialized at FP32 in model storage.
      std::array<float, kBlock> partial{};
      for (int lane = 0; lane < kBlock; ++lane) {
        float acc = 0.0f;
        for (int64_t column = lane; column < width; column += kBlock) {
          const float value = Expression(a, base, delta, row, column);
          acc += value * value;
        }
        partial[static_cast<size_t>(lane)] = acc;
      }
      for (int stride = kBlock / 2; stride > 0; stride /= 2)
        for (int lane = 0; lane < stride; ++lane)
          partial[static_cast<size_t>(lane)] += partial[static_cast<size_t>(lane + stride)];
      const float inverse = 1.0f / std::sqrt(partial[0] / static_cast<float>(width) + args.eps);
      for (int64_t column = 0; column < width; ++column) {
        // Read the complete unrounded value before either permitted alias writes.
        const float value = Expression(a, base, delta, row, column);
        const float gamma = BF16ToF32(weight.Ptr<uint16_t>()[column]);
        out.Ptr<uint16_t>()[row * out.stride[0] + column] = F32ToBF16((value * inverse) * gamma);
        if (residual_out != nullptr)
          residual_out->Ptr<uint16_t>()[row * residual_out->stride[0] + column] = F32ToBF16(value);
      }
    }
  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kResidualRmsNorm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ResidualRmsNormFn>(&ResidualRmsNormKernel)));
  }
} registrar;
}  // namespace
}  // namespace vt::cpu
