// ROCm GDN norm/gate + fused attention preamble (BACKEND-ROCM-GDN-KERNELS
// family 5, the final slice; issue #41). kRmsNormGated (gdn-semantics §5),
// kSigmoidGateBf16 (ops.cpp contract: out bf16, attn f32/bf16, gate f32 — the
// sigmoid input stays unrounded), kAttnQkNormRopeGate (fused full-attention
// preamble: split q|gate + gemma/plain qk-RMSNorm + partial NeoX RoPE-from-
// cache + gate passthrough). Donors: cuda_gdn.cu RmsNormGatedRowKernel and
// cuda_ops.cu:1429 area; CPU oracles cpu_ops.cpp:1210/:2273/:956.
// kSigmoidGateBf16 has NO CUDA registration — the CPU composite semantics are
// the donor of record (the Vulkan lane's only native sibling).
//
// Gate: the norm-gate and preamble arms of test_backend_cross_device.cpp —
// NMSE <= 5e-4 for the norm/preamble arithmetic; SigmoidGateBf16 bit-exact
// (single multiply, same RNE store on both sides).

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm gdn: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}
inline unsigned GridFor(int64_t n) {
  if (n <= 0) return 1;
  const int64_t g = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(g > 65535 ? 65535 : g);
}

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}
__device__ inline float Silu(float x) { return x / (1.0f + expf(-x)); }
__device__ inline float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float GemmaNormElem(float v, float inv, float w, bool gemma) {
  float wj = w;
  if (gemma) wj += 1.0f;
  return v * inv * wj;
}

// ── Norm/gate (donor cuda_gdn.cu RmsNormGatedRowKernel; CPU :1210/:2273) ─────
template <typename Tx, typename Tout>
__global__ void RmsNormGatedK(Tout* out, const Tx* x, const Tx* gate,
                              const Tx* w, int64_t rows, int64_t d,
                              int64_t gate_group, int64_t gate_outer, float eps,
                              bool sigmoid_gate) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  float sumsq = 0.0f;
  for (int64_t j = 0; j < d; ++j) {
    const float v = Ld(x, i * d + j);
    sumsq += v * v;
  }
  const float inv = 1.0f / sqrtf(sumsq / static_cast<float>(d) + eps);
  const int64_t gbase = (i / gate_group) * gate_outer + (i % gate_group) * d;
  for (int64_t j = 0; j < d; ++j) {
    const float z = Ld(gate, gbase + j);
    const float act = sigmoid_gate ? Sigmoid(z) : Silu(z);
    St(out, i * d + j, Ld(x, i * d + j) * inv * Ld(w, j) * act);
  }
}

// ops.cpp contract: out bf16, attn f32-or-bf16, gate f32 (unrounded sigmoid
// input). Tattn only.
template <typename Tattn>
__global__ void SigmoidGateK(__hip_bfloat16* out, const Tattn* attn,
                             const float* gate, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    St(out, i, Ld(attn, i) * Sigmoid(gate[i]));
  }
}

template <typename Tsrc, typename Tqk, typename Tgate>
__global__ void AttnQkNormRopeGateK(Tqk* q_out, Tqk* k_out, Tgate* gate_out,
                                    const Tsrc* qgate, const Tsrc* kf,
                                    const float* q_norm, const float* k_norm,
                                    const float* cos_sin, int64_t t, int64_t hq,
                                    int64_t hkv, int64_t dh, int64_t qgate_stride,
                                    int64_t kf_stride, int rot, float eps,
                                    bool gemma) {
  const int64_t half = rot / 2;
  const int64_t items = t * (hq + hkv);
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < items; item += step) {
    const int64_t tok = item / (hq + hkv);
    const int64_t h = item % (hq + hkv);
    const float* cs = cos_sin + tok * rot;
    int64_t src_off, out_off;
    if (h < hq) {
      src_off = tok * qgate_stride + h * 2 * dh;
      out_off = (tok * hq + h) * dh;
      for (int64_t j = 0; j < dh; ++j)
        St(gate_out, out_off + j, Ld(qgate, src_off + dh + j));
      float ss = 0.0f;
      for (int64_t j = 0; j < dh; ++j) {
        const float v = Ld(qgate, src_off + j);
        ss += v * v;
      }
      const float inv = 1.0f / sqrtf(ss / static_cast<float>(dh) + eps);
      for (int64_t j = 0; j < dh; ++j) {
        if (j < half) {
          const float ni = GemmaNormElem(Ld(qgate, src_off + j), inv, q_norm[j], gemma);
          const float nih =
              GemmaNormElem(Ld(qgate, src_off + j + half), inv, q_norm[j + half], gemma);
          St(q_out, out_off + j, ni * cs[j] - nih * cs[half + j]);
        } else if (j < rot) {
          const int64_t i = j - half;
          const float ni = GemmaNormElem(Ld(qgate, src_off + i), inv, q_norm[i], gemma);
          const float nih =
              GemmaNormElem(Ld(qgate, src_off + i + half), inv, q_norm[i + half], gemma);
          St(q_out, out_off + j, ni * cs[half + i] + nih * cs[i]);
        } else {
          St(q_out, out_off + j,
             GemmaNormElem(Ld(qgate, src_off + j), inv, q_norm[j], gemma));
        }
      }
    } else {
      const int64_t hk_i = h - hq;
      src_off = tok * kf_stride + hk_i * dh;
      out_off = (tok * hkv + hk_i) * dh;
      float ss = 0.0f;
      for (int64_t j = 0; j < dh; ++j) {
        const float v = Ld(kf, src_off + j);
        ss += v * v;
      }
      const float inv = 1.0f / sqrtf(ss / static_cast<float>(dh) + eps);
      for (int64_t j = 0; j < dh; ++j) {
        if (j < half) {
          const float ni = GemmaNormElem(Ld(kf, src_off + j), inv, k_norm[j], gemma);
          const float nih =
              GemmaNormElem(Ld(kf, src_off + j + half), inv, k_norm[j + half], gemma);
          St(k_out, out_off + j, ni * cs[j] - nih * cs[half + j]);
        } else if (j < rot) {
          const int64_t i = j - half;
          const float ni = GemmaNormElem(Ld(kf, src_off + i), inv, k_norm[i], gemma);
          const float nih =
              GemmaNormElem(Ld(kf, src_off + i + half), inv, k_norm[i + half], gemma);
          St(k_out, out_off + j, ni * cs[half + i] + nih * cs[i]);
        } else {
          St(k_out, out_off + j,
             GemmaNormElem(Ld(kf, src_off + j), inv, k_norm[j], gemma));
        }
      }
    }
  }
}

}  // namespace

void RmsNormGatedKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                            const Tensor& w, const RmsNormGatedArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "rocm rmsnorm_gated: unsupported input dtype (f32/bf16 only)");
  VT_CHECK(gate.dtype == x.dtype && w.dtype == x.dtype,
           "rocm rmsnorm_gated: gate/weight dtype must match x");
  const int64_t d = x.shape[x.rank - 1];
  const int64_t t = d == 0 ? 0 : x.Numel() / d;
  if (t == 0 || d == 0) return;
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  hipStream_t s = AsStream(q);
  const unsigned grid = static_cast<unsigned>(t);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      RmsNormGatedK<float, float><<<grid, 1, 0, s>>>(
          out.Ptr<float>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), t, d,
          gate_group, gate_outer, args.eps, args.sigmoid_gate);
    } else {
      RmsNormGatedK<float, __hip_bfloat16><<<grid, 1, 0, s>>>(
          out.Ptr<__hip_bfloat16>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(),
          t, d, gate_group, gate_outer, args.eps, args.sigmoid_gate);
    }
  } else {
    if (out.dtype == DType::kF32) {
      RmsNormGatedK<__hip_bfloat16, float><<<grid, 1, 0, s>>>(
          out.Ptr<float>(), x.Ptr<__hip_bfloat16>(), gate.Ptr<__hip_bfloat16>(),
          w.Ptr<__hip_bfloat16>(), t, d, gate_group, gate_outer, args.eps,
          args.sigmoid_gate);
    } else {
      RmsNormGatedK<__hip_bfloat16, __hip_bfloat16><<<grid, 1, 0, s>>>(
          out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), gate.Ptr<__hip_bfloat16>(),
          w.Ptr<__hip_bfloat16>(), t, d, gate_group, gate_outer, args.eps,
          args.sigmoid_gate);
    }
  }
  Check(hipGetLastError(), "rmsnorm_gated launch");
}

void SigmoidGateBf16KernelRocm(Queue& q, Tensor& out, const Tensor& attn,
                               const Tensor& gate) {
  // Contract (ops.cpp:3350): out bf16, attn f32/bf16, gate f32.
  VT_CHECK(out.dtype == DType::kBF16, "rocm sigmoid_gate_bf16: out must be bf16");
  VT_CHECK(gate.dtype == DType::kF32, "rocm sigmoid_gate_bf16: gate must be f32");
  const int64_t n = out.Numel();
  if (n == 0) return;
  hipStream_t s = AsStream(q);
  if (attn.dtype == DType::kBF16) {
    SigmoidGateK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__hip_bfloat16>(),
                                               attn.Ptr<__hip_bfloat16>(),
                                               gate.Ptr<float>(), n);
  } else {
    SigmoidGateK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__hip_bfloat16>(),
                                               attn.Ptr<float>(),
                                               gate.Ptr<float>(), n);
  }
  Check(hipGetLastError(), "sigmoid_gate launch");
}

void AttnQkNormRopeGateKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                                  const Tensor& qgate, const Tensor& kf,
                                  const Tensor& q_norm, const Tensor& k_norm,
                                  const Tensor& cos_sin, const RmsNormArgs& na,
                                  const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  if (t == 0) return;
  const int64_t items = t * (hq + hkv);
  hipStream_t s = AsStream(q);
  auto launch = [&](auto src_tag, auto qk_tag, auto gate_tag) {
    using Tsrc = decltype(src_tag);
    using Tqk = decltype(qk_tag);
    using Tgate = decltype(gate_tag);
    AttnQkNormRopeGateK<Tsrc, Tqk, Tgate><<<GridFor(items), kBlock, 0, s>>>(
        q_out.Ptr<Tqk>(), k_out.Ptr<Tqk>(), gate_out.Ptr<Tgate>(), qgate.Ptr<Tsrc>(),
        kf.Ptr<Tsrc>(), q_norm.Ptr<float>(), k_norm.Ptr<float>(), cos_sin.Ptr<float>(),
        t, hq, hkv, dh, qgate.stride[0], kf.stride[0], ra.rotary_dim, na.eps, na.gemma);
  };
  // Dispatch on the OUTPUT dtype like the CUDA lane (cuda_ops.cu
  // LaunchAttnPreambleOut): the caller picks f32 out (token-exact path) or bf16
  // out (FA-2 combos) INDEPENDENTLY of the source dtype — the 0.8B bf16 model
  // runs bf16 src + f32 q/k out + f32 gate on the f32-attention path, which the
  // old src-keyed dispatch silently mis-launched as all-bf16 (writing bf16 bits
  // through the f32 out pointer; issue #41 M4 W2 finding). Validated combos
  // (mirror CUDA): (f32 src, f32 out); (bf16 src, f32 out + f32 gate);
  // (bf16 src, bf16 out + bf16 gate); (bf16 src, bf16 out + f32 gate).
  const bool src_bf16 = qgate.dtype == DType::kBF16;
  if (q_out.dtype == DType::kF32) {
    if (src_bf16) {
      launch(__hip_bfloat16{}, float{}, float{});
    } else {
      launch(float{}, float{}, float{});
    }
  } else {
    VT_CHECK(src_bf16, "rocm attn_qk_norm_rope_gate: bf16 out needs bf16 src");
    if (gate_out.dtype == DType::kF32) {
      launch(__hip_bfloat16{}, __hip_bfloat16{}, float{});
    } else {
      launch(__hip_bfloat16{}, __hip_bfloat16{}, __hip_bfloat16{});
    }
  }
  Check(hipGetLastError(), "attn_qk_norm_rope_gate launch");
}

}  // namespace vt::rocm
