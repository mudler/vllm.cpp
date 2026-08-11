// ROCm GDN post-conv glue (BACKEND-ROCM-GDN-KERNELS family 3; issue #41).
// kGdnPostConv: conv-split + q/k l2norm + g/beta in ONE launch (mirror of fla
// fused_gdn_prefill_post_conv; gdn-semantics.md §4/§6). Hand-translated from
// cuda_gdn.cu:1155, readable side by side against the donor; CPU oracle
// src/vt/cpu/cpu_ops.cpp:2337. This is the fused VT_GLUE_FUSE path the model
// calls by default — the M2 throw after conv lands. Sequential per
// (token, head) thread here, matching the CPU oracle's reduction order; the
// donor's block-tree reduction and the Fast/Split variants are M5 levers.
//
// Gate: the postconv arm of the fused-glue case in
// test_backend_cross_device.cpp — NMSE <= 5e-4 vs the CPU oracle, padded a/b
// row strides included.

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

// ── Fused glue (donors cuda_gdn.cu:1155; cpu oracle :956 for the preamble) ──
template <typename Tqkv, typename Tconv, typename Tgate>
__global__ void GdnPostConvK(Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, float* g_out,
                             float* beta_out, const Tconv* conv, const Tgate* araw,
                             const Tgate* braw, const float* a_log,
                             const float* dt_bias, int64_t t, int64_t hk, int64_t dk,
                             int64_t hv, int64_t dv, int64_t a_row_stride,
                             int64_t b_row_stride, float eps) {
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t items = t * (hk + 1);
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < items; item += step) {
    const int64_t tok = item / (hk + 1);
    const int64_t head = item % (hk + 1);
    const int64_t row = tok * conv_dim;
    if (head < hk) {
      const Tconv* qin = conv + row + head * dk;
      const Tconv* kin = conv + row + key_dim + head * dk;
      float qss = 0.0f, kss = 0.0f;
      for (int64_t j = 0; j < dk; ++j) {
        const float qv = Ld(qin, j);
        const float kv = Ld(kin, j);
        qss += qv * qv;
        kss += kv * kv;
      }
      const float qinv = 1.0f / sqrtf(qss + eps);
      const float kinv = 1.0f / sqrtf(kss + eps);
      Tqkv* qo = q_out + (tok * hk + head) * dk;
      Tqkv* ko = k_out + (tok * hk + head) * dk;
      for (int64_t j = 0; j < dk; ++j) {
        St(qo, j, Ld(qin, j) * qinv);
        St(ko, j, Ld(kin, j) * kinv);
      }
    } else {
      const Tconv* vin = conv + row + 2 * key_dim;
      Tqkv* vo = v_out + tok * value_dim;
      for (int64_t j = 0; j < value_dim; ++j) St(vo, j, Ld(vin, j));
      for (int64_t h = 0; h < hv; ++h) {
        const int64_t idx = tok * hv + h;
        const float av = Ld(araw, tok * a_row_stride + h);
        const float bv = Ld(braw, tok * b_row_stride + h);
        const float x = av + dt_bias[h];
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        g_out[idx] = -expf(a_log[h]) * sp;
        beta_out[idx] = 1.0f / (1.0f + expf(-bv));
      }
    }
  }
}

}  // namespace

void GdnPostConvKernelRocm(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                           Tensor& g_out, Tensor& beta_out, const Tensor& conv,
                           const Tensor& araw, const Tensor& braw, const Tensor& a_log,
                           const Tensor& dt_bias, const L2NormArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  if (t == 0) return;
  VT_CHECK(q_out.dtype == k_out.dtype && q_out.dtype == v_out.dtype,
           "rocm gdn_post_conv: q/k/v out dtypes must match");
  VT_CHECK(q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16,
           "rocm gdn_post_conv: q/k/v out must be f32 or bf16");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "rocm gdn_post_conv: conv must be f32 or bf16");
  const int64_t items = t * (hk + 1);
  hipStream_t s = AsStream(q);
  auto launch = [&](auto qkv_tag, auto conv_tag, auto gate_tag) {
    using Tqkv = decltype(qkv_tag);
    using Tconv = decltype(conv_tag);
    using Tgate = decltype(gate_tag);
    GdnPostConvK<Tqkv, Tconv, Tgate><<<GridFor(items), kBlock, 0, s>>>(
        q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
        beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<Tgate>(), braw.Ptr<Tgate>(),
        a_log.Ptr<float>(), dt_bias.Ptr<float>(), t, hk, dk, hv, dv, araw.stride[0],
        braw.stride[0], args.eps);
  };
  auto launch_gate = [&](auto qkv_tag, auto conv_tag) {
    if (araw.dtype == DType::kBF16)
      launch(qkv_tag, conv_tag, __hip_bfloat16{});
    else
      launch(qkv_tag, conv_tag, float{});
  };
  auto launch_conv = [&](auto qkv_tag) {
    if (conv.dtype == DType::kBF16)
      launch_gate(qkv_tag, __hip_bfloat16{});
    else
      launch_gate(qkv_tag, float{});
  };
  if (q_out.dtype == DType::kBF16)
    launch_conv(__hip_bfloat16{});
  else
    launch_conv(float{});
  Check(hipGetLastError(), "gdn_post_conv launch");
}

}  // namespace vt::rocm
