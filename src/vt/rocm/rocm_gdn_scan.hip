// ROCm GDN recurrence core (BACKEND-ROCM-GDN-KERNELS family 4; issue #41).
// kGdnPrefill + kGdnDecode over the PORTABLE scan — hand-translation of
// cuda_gdn.cu GdnScanKernel (:1856) + its launchers, readable side by side
// against the donor. CPU oracle: src/vt/cpu/cpu_ops.cpp:1331/1368
// (GdnHeadTokenStep); formula reference gdn-semantics.md §7/§8. One kernel
// serves both ops (qsl != null => varlen prefill; qsl == null => single-step
// decode with optional state_idx + NULL-block zero-out).
//
// Deliberately NOT ported (docs/ROCM.md §6): the WMMA chunked prefill
// (`#if __CUDA_ARCH__ >= 800`), the fused/packed/Triton-AOT decode variants —
// NVIDIA-only machinery; the scan is the correctness baseline and the perf
// variants are M5 levers (upstream's ROCm answer, Triton/CK, is a later row).
//
// Gate: the recurrence case in test_backend_cross_device.cpp — NMSE <= 5e-4
// vs the CPU oracle on out AND in-place state; decode NULL-slot zero-out
// EXACT.

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

// ── Recurrence scan (donor cuda_gdn.cu:1856; serves prefill AND decode) ─────
template <typename Tin, typename Tout, typename TState>
__global__ void GdnScanK(Tout* out, const Tin* q, const Tin* k, const Tin* v,
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
  extern __shared__ float smem[];  // [dk] q' then [dk] k
  float* q_sh = smem;
  float* k_sh = smem + dk;
  TState* s_head = state + (state_slot * hv_n + hv) * dv * dk;
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Ld(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Ld(k, (t * hk_n + hk) * dk + i);
    }
    __syncthreads();
    const float decay = expf(g[t * hv_n + hv]);
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
      TState* s_row = s_head + vi * dk;
      float dot = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float decayed = Ld(s_row, ki) * decay;
        dot += decayed * k_sh[ki];
      }
      const float vp = (Ld(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        const float updated = Ld(s_row, ki) * decay + vp * k_sh[ki];
        St(s_row, ki, updated);
        o += updated * q_sh[ki];
      }
      St(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();
  }
}

template <typename Tin, typename Tout>
void LaunchGdnScanState(hipStream_t s, Tensor& out, const Tensor& q_in,
                        const Tensor& k, const Tensor& v, const Tensor& g,
                        const Tensor& beta, Tensor& state, const int32_t* qsl,
                        const int32_t* state_idx, int64_t n,
                        const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const dim3 grid(static_cast<unsigned>(hv_n), static_cast<unsigned>(n));
  const size_t shmem = 2 * static_cast<size_t>(dk) * sizeof(float);
  if (state.dtype == DType::kF16) {
    GdnScanK<Tin, Tout, __half><<<grid, kBlock, shmem, s>>>(
        out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
        g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__half>(), qsl, state_idx,
        state.shape[0], hk_n, dk, hv_n, dv, args.scale);
  } else if (state.dtype == DType::kBF16) {
    GdnScanK<Tin, Tout, __hip_bfloat16><<<grid, kBlock, shmem, s>>>(
        out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
        g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<__hip_bfloat16>(), qsl,
        state_idx, state.shape[0], hk_n, dk, hv_n, dv, args.scale);
  } else {
    GdnScanK<Tin, Tout, float><<<grid, kBlock, shmem, s>>>(
        out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(),
        g.Ptr<float>(), beta.Ptr<float>(), state.Ptr<float>(), qsl, state_idx,
        state.shape[0], hk_n, dk, hv_n, dv, args.scale);
  }
  Check(hipGetLastError(), "gdn scan launch");
}

template <typename Tin>
void LaunchGdnScanIn(hipStream_t s, Tensor& out, const Tensor& q_in,
                     const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, Tensor& state, const int32_t* qsl,
                     const int32_t* state_idx, int64_t n, const GdnArgs& args) {
  if (out.dtype == DType::kBF16)
    LaunchGdnScanState<Tin, __hip_bfloat16>(s, out, q_in, k, v, g, beta, state, qsl,
                                            state_idx, n, args);
  else
    LaunchGdnScanState<Tin, float>(s, out, q_in, k, v, g, beta, state, qsl,
                                   state_idx, n, args);
}

}  // namespace

void GdnPrefillKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta,
                          Tensor& state, const Tensor& qsl, const GdnArgs& args) {
  const int64_t n = state.shape[0];
  if (n == 0) return;
  hipStream_t s = AsStream(q);
  if (q_in.dtype == DType::kBF16) {
    LaunchGdnScanIn<__hip_bfloat16>(s, out, q_in, k, v, g, beta, state,
                                    qsl.Ptr<int32_t>(), nullptr, n, args);
  } else {
    LaunchGdnScanIn<float>(s, out, q_in, k, v, g, beta, state, qsl.Ptr<int32_t>(),
                           nullptr, n, args);
  }
}

void GdnDecodeKernelRocm(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         Tensor& state, const Tensor* state_idx, const GdnArgs& args) {
  const int64_t n = q_in.shape[0];
  if (n == 0) return;
  const int32_t* si = state_idx != nullptr ? state_idx->Ptr<int32_t>() : nullptr;
  hipStream_t s = AsStream(q);
  if (q_in.dtype == DType::kBF16) {
    LaunchGdnScanIn<__hip_bfloat16>(s, out, q_in, k, v, g, beta, state, nullptr, si, n,
                                    args);
  } else {
    LaunchGdnScanIn<float>(s, out, q_in, k, v, g, beta, state, nullptr, si, n, args);
  }
}

}  // namespace vt::rocm
