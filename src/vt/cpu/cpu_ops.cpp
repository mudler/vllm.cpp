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

// Mirror of LoadF32 for outputs: f32 stores directly, bf16 rounds via F32ToBF16.
void StoreF32(const Tensor& t, int64_t elem_offset, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[elem_offset] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[elem_offset] = F32ToBF16(v); break;
    default: VT_CHECK(false, "StoreF32: unsupported dtype");
  }
}

void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += LoadF32(a, i * k + p) * LoadF32(b, p * n + j);
      }
      StoreF32(out, i * n + j, acc);
    }
  }
}

void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
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
      StoreF32(out, i * h + j, v * inv * wj);
    }
  }
}

void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  for (int64_t i = 0; i < t; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      float gate = LoadF32(x, i * 2 * d + j);
      float up = LoadF32(x, i * 2 * d + d + j);
      float silu = gate / (1.0f + std::exp(-gate));
      StoreF32(out, i * d + j, silu * up);
    }
  }
}

void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  const int64_t t = ids.shape[0], h = table.shape[1], v = table.shape[0];
  for (int64_t i = 0; i < t; ++i) {
    int64_t id = ids.dtype == DType::kI32 ? ids.Ptr<int32_t>()[i] : ids.Ptr<int64_t>()[i];
    VT_CHECK(id >= 0 && id < v, "embedding: id out of range");
    for (int64_t j = 0; j < h; ++j) {
      StoreF32(out, i * h + j, LoadF32(table, id * h + j));
    }
  }
}

// In-place rotation of one head starting at element head_off; f32 math,
// stores round back to the tensor's dtype (f32 or bf16).
void RopeRotateHead(const Tensor& t, int64_t head_off, int rot, double base, int64_t pos) {
  const int half = rot / 2;
  for (int i = 0; i < half; ++i) {
    double freq = std::pow(base, -2.0 * i / rot);
    double angle = static_cast<double>(pos) * freq;
    float c = static_cast<float>(std::cos(angle));
    float s = static_cast<float>(std::sin(angle));
    float x = LoadF32(t, head_off + i);
    float y = LoadF32(t, head_off + i + half);
    StoreF32(t, head_off + i, x * c - y * s);
    StoreF32(t, head_off + i + half, x * s + y * c);
  }
}

void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  for (int64_t i = 0; i < t; ++i) {
    int64_t p = pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[i] : pos.Ptr<int64_t>()[i];
    for (int64_t hh = 0; hh < hq; ++hh) {
      RopeRotateHead(qs, (i * hq + hh) * d, args.rotary_dim, static_cast<double>(args.base), p);
    }
    for (int64_t hh = 0; hh < hk; ++hh) {
      RopeRotateHead(ks, (i * hk + hh) * d, args.rotary_dim, static_cast<double>(args.base), p);
    }
  }
}

struct Registrar {
  Registrar() {
    // static_cast against the ops.h aliases ties kernel signatures to the
    // registration contract at compile time.
    RegisterOp(OpId::kMatmul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
