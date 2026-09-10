// ROCm KDA per-K-channel-decay gated-delta recurrence (BACKEND-ROCM; issue #2942).
// kKdaGatedDeltaRule — hand-translation of cuda_gdn.cu KdaScanKernel (:3139) +
// LaunchKdaScan + KdaGatedDeltaRuleKernelCuda (:3207-3235), readable side by side
// against the donor. CPU oracle: src/vt/cpu/cpu_ops.cpp:2241
// (KdaGatedDeltaRuleKernel; recurrence math in KdaHeadTokenStep :2208).
// Spec: .agents/specs/rocm-kda-gated-delta-rule.md.
//
// Byte-for-byte GdnScanK (rocm_gdn_scan.hip) EXCEPT the state decay is
// per-K-channel: GDN uses one scalar decay = expf(g[t*hv_n+hv]) for the whole
// [Dv,Dk] state; KDA stages a per-K vector decay[ki] = expf(g[(t*hv_n+hv)*dk+ki])
// in shared memory and applies s_row[ki] *= decay[ki]. State is float only (no
// f16/bf16 state dispatch, unlike GDN); the launcher passes nullptr for state_idx
// (decode with state indexing is the GDN decode path, not KDA).
//
// Gate: the recurrence case in test_backend_cross_device.cpp — NMSE <= 5e-4
// vs the CPU oracle on out AND in-place state.

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm kda: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}
// KDA dispatches Tin/Tout in {float, __hip_bfloat16} only (never __half);
// TState is always float.  The __half overloads that GDN carries would be
// dead code here and trip -Werror,-Wunused-function.
__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

// ── Per-K-channel-decay recurrence scan (donor cuda_gdn.cu:3139) ──────────
template <typename Tin, typename Tout, typename TState>
__global__ void KdaScanK(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                         const float* g, const float* beta, TState* state,
                         const int32_t* qsl, const int32_t* state_idx,
                         int64_t state_slots, int64_t hk_n, int64_t dk,
                         int64_t hv_n, int64_t dv, float scale) {
  const int64_t s = blockIdx.y;
  const int64_t hv = blockIdx.x;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t state_slot = state_idx != nullptr ? state_idx[s] : s;
  if (state_slot < 0 || state_slot >= state_slots) {
    const int64_t begin = qsl != nullptr ? qsl[s] : s;
    const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
    for (int64_t t = begin; t < end; ++t)
      for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x)
        St(out, (t * hv_n + hv) * dv + vi, 0.0f);
    return;
  }
  extern __shared__ float smem[];  // [dk] q' then [dk] k then [dk] per-K decay
  float* q_sh = smem;
  float* k_sh = smem + dk;
  float* d_sh = smem + 2 * dk;
  TState* s_head = state + (state_slot * hv_n + hv) * dv * dk;
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Ld(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Ld(k, (t * hk_n + hk) * dk + i);
      d_sh[i] = expf(g[(t * hv_n + hv) * dk + i]);
    }
    __syncthreads();
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
      TState* s_row = s_head + vi * dk;
      float dot = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float decayed = Ld(s_row, ki) * d_sh[ki];
        dot += decayed * k_sh[ki];
      }
      const float vp = (Ld(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float updated = Ld(s_row, ki) * d_sh[ki] + vp * k_sh[ki];
        St(s_row, ki, updated);
        o += updated * q_sh[ki];
      }
      St(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();
  }
}

template <typename Tin, typename Tout>
void LaunchKdaScan(hipStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                    const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                    const int32_t* qsl, int64_t n, const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const dim3 grid(static_cast<unsigned>(hv_n), static_cast<unsigned>(n));
  const size_t shmem = 3 * static_cast<size_t>(dk) * sizeof(float);
  KdaScanK<Tin, Tout, float><<<grid, kBlock, shmem, s>>>(
      out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
      beta.Ptr<float>(), state.Ptr<float>(), qsl, nullptr, state.shape[0], hk_n, dk, hv_n,
      dv, args.scale);
  Check(hipGetLastError(), "kda scan launch");
}

}  // namespace

void KdaGatedDeltaRuleKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                                  const Tensor& v, const Tensor& g, const Tensor& beta,
                                  Tensor& state, const Tensor& qsl, const GdnArgs& args) {
  constexpr const char* name = "kda_gated_delta_rule";
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           std::string("rocm ") + name + ": unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           std::string("rocm ") + name + ": q/k/v dtypes must match");
  const int64_t n = state.shape[0];
  const int64_t hv_n = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  if (n == 0 || hv_n == 0 || dv == 0) return;
  VT_CHECK(3 * static_cast<size_t>(dk) * sizeof(float) <= 48 * 1024,
           std::string("rocm ") + name + ": Dk too large for the shared q'/k/decay staging");
  hipStream_t s = AsStream(q);
  const int32_t* qsl_ptr = qsl.Ptr<int32_t>();
  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32)
      LaunchKdaScan<float, float>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n, args);
    else
      LaunchKdaScan<float, __hip_bfloat16>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n,
                                           args);
  } else {
    if (out.dtype == DType::kF32)
      LaunchKdaScan<__hip_bfloat16, float>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n,
                                           args);
    else
      LaunchKdaScan<__hip_bfloat16, __hip_bfloat16>(s, out, q_in, k, v, g, beta, state,
                                                     qsl_ptr, n, args);
  }
}

}  // namespace vt::rocm
