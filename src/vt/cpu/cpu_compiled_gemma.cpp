// CPU reference for the executing Gemma compiler boundaries, vLLM e126687a9a.
#include <array>
#include <cmath>

#include "cpu_threadpool.h"
#include "vt/compiled_gemma.h"

namespace vt::cpu {
namespace {
float Load(const Tensor& t, int64_t i) {
  return BF16ToF32(t.Ptr<uint16_t>()[i]);
}
template <class Value>
float SquaredSum(int64_t width, Value value) {
  std::array<float, 1024> partial{};
  for (int64_t j = 0; j < width; ++j) {
    const float v = value(j);
    partial[j % 1024] = std::fma(v, v, partial[j % 1024]);
  }
  std::array<float, 256> lanes{};
  for (int lane = 0; lane < 256; ++lane)
    lanes[lane] = ((partial[4 * lane] + partial[4 * lane + 1]) + partial[4 * lane + 2]) +
                  partial[4 * lane + 3];
  std::array<float, 8> warps{};
  for (int warp = 0; warp < 8; ++warp) {
    for (int half = 0; half < 2; ++half) {
      const int begin = warp * 32 + half * 16;
      for (int stride = 8; stride; stride /= 2)
        for (int lane = 0; lane < stride; ++lane)
          lanes[begin + lane] += lanes[begin + lane + stride];
    }
    warps[warp] = lanes[warp * 32] + lanes[warp * 32 + 16];
  }
  for (int stride = 4; stride; stride /= 2)
    for (int lane = 0; lane < stride; ++lane) warps[lane] += warps[lane + stride];
  return warps[0];
}
float Inverse(const Tensor& x, int64_t row, float eps, float scale = 1.f) {
  const int64_t width = x.shape[1];
  const float sum = SquaredSum(width, [&](int64_t j) { return Load(x, row * width + j) * scale; });
  return 1.f / std::sqrt(sum / static_cast<float>(width) + eps);
}
void Scaled(Queue&, Tensor& out, const Tensor& input, const Tensor& weight,
            const ScaledRmsNormArgs& args, Tensor& residual) {
  const int64_t width = input.shape[1];
  ParallelForRows(CurrentThreadpool(), input.shape[0], [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const float inv = Inverse(input, row, args.eps, args.scale);
      for (int64_t j = 0; j < width; ++j) {
        const auto scaled = F32ToBF16(Load(input, row * width + j) * args.scale);
        residual.Ptr<uint16_t>()[row * width + j] = scaled;
        out.Ptr<uint16_t>()[row * width + j] =
            F32ToBF16((BF16ToF32(scaled) * inv) * (1.f + Load(weight, j)));
      }
    }
  });
}
void Sandwich(Queue&, Tensor& out, const SandwichNormInputs& in, float eps, Tensor* residual) {
  const int64_t width = in.a.shape[1];
  ParallelForRows(CurrentThreadpool(), in.a.shape[0], [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const float ai = Inverse(in.a, row, eps), di = in.delta ? Inverse(*in.delta, row, eps) : 0.f;
      const auto expression = [&](int64_t j) {
        const float a = Load(in.a, row * width + j) * ai;
        const float sum = std::fma(a, 1.f + Load(in.a_weight, j), Load(in.base, row * width + j));
        if (!in.delta) return sum;
        const float delta = Load(*in.delta, row * width + j) * di;
        return std::fma(delta, 1.f + Load(*in.delta_weight, j), sum);
      };
      const float sum = SquaredSum(width, expression);
      const float inv = 1.f / std::sqrt(sum / static_cast<float>(width) + eps);
      for (int64_t j = 0; j < width; ++j) {
        const float value = expression(j);
        out.Ptr<uint16_t>()[row * width + j] =
            F32ToBF16((value * inv) * (1.f + Load(in.weight, j)));
        if (residual) residual->Ptr<uint16_t>()[row * width + j] = F32ToBF16(value);
      }
    }
  });
}
void Gelu(Queue&, Tensor& out, const Tensor& x) {
  const int64_t width = out.shape[1];
  ParallelForRows(CurrentThreadpool(), out.shape[0], [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row)
      for (int64_t j = 0; j < width; ++j) {
        const float gate = Load(x, row * 2 * width + j), up = Load(x, row * 2 * width + width + j);
        const float activated = (0.5f * gate) * (1.f + std::erf(gate * 0.7071067811865476f));
        out.Ptr<uint16_t>()[row * width + j] = F32ToBF16(activated * up);
      }
  });
}
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kScaledRmsNorm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ScaledRmsNormFn>(&Scaled)));
    RegisterOp(OpId::kSandwichRmsNorm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SandwichRmsNormFn>(&Sandwich)));
    RegisterOp(OpId::kCompiledGeluErfMul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GeluAndMulFn>(&Gelu)));
  }
} registrar;
}  // namespace
}  // namespace vt::cpu
