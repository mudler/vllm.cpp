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

void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  float* o = out.Ptr<float>();
  for (int64_t i = 0; i < t; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      float gate = LoadF32(x, i * 2 * d + j);
      float up = LoadF32(x, i * 2 * d + d + j);
      float silu = gate / (1.0f + std::exp(-gate));
      o[i * d + j] = silu * up;
    }
  }
}

void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  const int64_t t = ids.shape[0], h = table.shape[1], v = table.shape[0];
  float* o = out.Ptr<float>();
  for (int64_t i = 0; i < t; ++i) {
    int64_t id = ids.dtype == DType::kI32 ? ids.Ptr<int32_t>()[i] : ids.Ptr<int64_t>()[i];
    VT_CHECK(id >= 0 && id < v, "embedding: id out of range");
    for (int64_t j = 0; j < h; ++j) {
      o[i * h + j] = LoadF32(table, id * h + j);
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmul, DeviceType::kCPU, reinterpret_cast<void*>(&MatmulKernel));
    RegisterOp(OpId::kRmsNorm, DeviceType::kCPU, reinterpret_cast<void*>(&RmsNormKernel));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kCPU, reinterpret_cast<void*>(&SiluAndMulKernel));
    RegisterOp(OpId::kEmbedding, DeviceType::kCPU, reinterpret_cast<void*>(&EmbeddingKernel));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
