// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/ops.h"

#include <cmath>

namespace vt::cpu {
namespace {

float LoadF32(const Tensor& t, int64_t elem_offset) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_offset];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    default: VT_CHECK(false, "LoadF32: unsupported dtype"); return 0.0f;
  }
}

void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  float* o = out.Ptr<float>();
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += LoadF32(a, i * k + p) * LoadF32(b, p * n + j);
      }
      o[i * n + j] = acc;
    }
  }
}

void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  float* o = out.Ptr<float>();
  for (int64_t i = 0; i < t; ++i) {
    float* res_row = residual ? residual->Ptr<float>() + i * h : nullptr;
    float sumsq = 0.0f;
    for (int64_t j = 0; j < h; ++j) {
      float v = LoadF32(x, i * h + j);
      if (res_row) {
        v += res_row[j];
        res_row[j] = v;  // new residual stream
      }
      sumsq += v * v;
    }
    float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(h) + args.eps);
    for (int64_t j = 0; j < h; ++j) {
      float v = res_row ? res_row[j] : LoadF32(x, i * h + j);
      float wj = LoadF32(w, j);
      if (args.gemma) wj += 1.0f;
      o[i * h + j] = v * inv * wj;
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmul, DeviceType::kCPU, reinterpret_cast<void*>(&MatmulKernel));
    RegisterOp(OpId::kRmsNorm, DeviceType::kCPU, reinterpret_cast<void*>(&RmsNormKernel));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
