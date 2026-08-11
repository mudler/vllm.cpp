// ROCm GDN causal conv1d (BACKEND-ROCM-GDN-KERNELS family 2; issue #41).
// Hand-translation of the cuda_gdn.cu scalar donors — CausalConv1dFwdKernel
// (:479, prefill, gdn-semantics.md §2) and CausalConv1dUpdateKernel (:884,
// decode seqlen==1, §3) — readable side by side against them. CPU oracles:
// src/vt/cpu/cpu_ops.cpp:1021/1081. The register-resident (VT_CONV_REG) and
// tiled (VT_CONV_TILED) CUDA variants are perf levers and stay unported
// (docs/ROCM.md §6). The exact-chunks descriptor form (VT_CONV_EXACT_CHUNKS)
// IS supported — same chunk addressing as the donor's Reg kernel over the
// scalar body — because Qwen3.5 prefill passes descriptors on the live path
// (the first draft's loud VT_CHECK guard against them fired in the M2 rerun,
// proving it).
//
// Gate: the "causal conv1d fwd/update" case in test_backend_cross_device.cpp —
// outputs NMSE <= 5e-4 vs the CPU oracle (silu epilogue arithmetic), conv_state
// write-back/roll BIT-EXACT (raw-x byte movement).

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

// ── Conv (donors cuda_gdn.cu:479 scalar fwd, :884 update) ───────────────────
constexpr int kConvExactM = 8;  // upstream compute_causal_conv1d_metadata BLOCK_M

// exact mode (batch_ptr/token_chunk_offset_ptr non-null): blockIdx.y indexes
// PROGRAMS, program p owning sequence batch_ptr[p] and its
// token_chunk_offset_ptr[p]-th 8-token chunk — the upstream
// compute_causal_conv1d_metadata work split, numerically identical to the
// legacy whole-sequence mapping (same tap order; the state write-back belongs
// to the block whose chunk reaches the sequence end).
template <typename Tin, typename Tout, typename THas>
__global__ void CausalConv1dFwdK(Tout* out, const Tin* x, const Tin* w,
                                 const Tin* bias, float* conv_state,
                                 const int32_t* qsl, const THas* his,
                                 int64_t c_dim, int64_t x_row_stride, int64_t k,
                                 bool silu, const int32_t* batch_ptr,
                                 const int32_t* token_chunk_offset_ptr) {
  const bool exact = batch_ptr != nullptr;
  const int64_t s = exact ? batch_ptr[blockIdx.y] : blockIdx.y;
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= c_dim) return;
  const int64_t width = k - 1;
  const int64_t begin = qsl[s];
  const int64_t t_len = qsl[s + 1] - begin;
  const int64_t token_offset =
      exact ? static_cast<int64_t>(token_chunk_offset_ptr[blockIdx.y]) * kConvExactM : 0;
  // Chunks past the sequence end are skipped — except chunk 0, which also owns
  // the t_len == 0 corner.
  if (token_offset > 0 && token_offset >= t_len) return;
  const int64_t token_end =
      (exact && token_offset + kConvExactM < t_len) ? token_offset + kConvExactM : t_len;
  const bool init = his[s] != 0;
  float* srow = conv_state + (s * c_dim + c) * width;
  const float b = bias != nullptr ? Ld(bias, c) : 0.0f;
  for (int64_t t = token_offset; t < token_end; ++t) {
    float acc = b;
    for (int64_t j = 0; j < k; ++j) {
      const int64_t ti = t - (k - 1 - j);
      float v = 0.0f;
      if (ti >= 0) {
        v = Ld(x, (begin + ti) * x_row_stride + c);
      } else if (init) {
        v = srow[width + ti];
      }
      acc += Ld(w, c * k + j) * v;
    }
    St(out, (begin + t) * c_dim + c, silu ? Silu(acc) : acc);
  }
  if (token_end == t_len) {
    for (int64_t j = 0; j < width; ++j) {
      const int64_t tj = t_len - width + j;
      float v = 0.0f;
      if (tj >= 0) {
        v = Ld(x, (begin + tj) * x_row_stride + c);
      } else if (init) {
        v = srow[width + tj];
      }
      srow[j] = v;
    }
  }
}

template <typename Tin, typename Tout, typename TState>
__global__ void CausalConv1dUpdateK(Tout* out, const Tin* x, const Tin* w,
                                    const Tin* bias, TState* conv_state,
                                    const int32_t* cache_idx, int64_t n,
                                    int64_t c_dim, int64_t x_row_stride,
                                    int64_t k, int64_t state_len, bool silu) {
  const int64_t width = k - 1;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += step) {
    const int64_t bt = idx / c_dim;
    const int64_t c = idx % c_dim;
    int64_t srow_off = idx;
    if (cache_idx != nullptr) {
      const int32_t slot = cache_idx[bt];
      if (slot < 0) continue;
      srow_off = static_cast<int64_t>(slot) * c_dim + c;
    }
    TState* srow = conv_state + srow_off * state_len;
    const float xt = Ld(x, bt * x_row_stride + c);
    float acc = bias != nullptr ? Ld(bias, c) : 0.0f;
    for (int64_t j = 0; j < width; ++j) acc += Ld(w, c * k + j) * Ld(srow, j);
    acc += Ld(w, c * k + width) * xt;
    St(out, idx, silu ? Silu(acc) : acc);
    for (int64_t j = 0; j + 1 < width; ++j) St(srow, j, Ld(srow, j + 1));
    if (width > 0) St(srow, width - 1, xt);
  }
}

template <typename Tin, typename Tout>
void LaunchConvFwd(hipStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                   const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                   const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t c = x.shape[1], k = w.shape[1];
  const int64_t x_rs = x.stride[0];
  const int32_t* bp = args.batch_ptr != nullptr ? args.batch_ptr->Ptr<int32_t>() : nullptr;
  const int32_t* tco = args.token_chunk_offset_ptr != nullptr
                           ? args.token_chunk_offset_ptr->Ptr<int32_t>()
                           : nullptr;
  // exact mode: grid.y is the descriptor program count, not the sequence count.
  const int64_t rows = bp != nullptr ? args.batch_ptr->shape[0] : conv_state.shape[0];
  const dim3 grid(static_cast<unsigned>((c + kBlock - 1) / kBlock),
                  static_cast<unsigned>(rows));
  if (his.dtype == DType::kI8) {
    CausalConv1dFwdK<Tin, Tout, int8_t><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
        bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
        qsl.Ptr<int32_t>(), his.Ptr<int8_t>(), c, x_rs, k, args.silu_activation, bp, tco);
  } else {
    CausalConv1dFwdK<Tin, Tout, int32_t><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
        bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<float>(),
        qsl.Ptr<int32_t>(), his.Ptr<int32_t>(), c, x_rs, k, args.silu_activation, bp, tco);
  }
  Check(hipGetLastError(), "causal_conv1d_fwd launch");
}

template <typename Tin, typename Tout, typename TState>
void LaunchConvUpdate(hipStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                      const CausalConv1dArgs& args) {
  const int64_t n = x.shape[0] * x.shape[1], c = x.shape[1], k = w.shape[1];
  const int64_t x_rs = x.stride[0];
  const int64_t state_len = conv_state.shape[2];
  CausalConv1dUpdateK<Tin, Tout, TState><<<GridFor(n), kBlock, 0, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(),
      bias != nullptr ? bias->Ptr<Tin>() : nullptr, conv_state.Ptr<TState>(),
      cache_idx, n, c, x_rs, k, state_len, args.silu_activation);
  Check(hipGetLastError(), "causal_conv1d_update launch");
}

template <typename Tin, typename Tout>
void LaunchConvUpdateS(hipStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                       const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                       const CausalConv1dArgs& args) {
  if (conv_state.dtype == DType::kBF16)
    LaunchConvUpdate<Tin, Tout, __hip_bfloat16>(s, out, x, w, bias, conv_state, cache_idx, args);
  else
    LaunchConvUpdate<Tin, Tout, float>(s, out, x, w, bias, conv_state, cache_idx, args);
}

}  // namespace

void CausalConv1dFwdKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                               const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                               const Tensor& his, const CausalConv1dArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "rocm causal_conv1d_fwd: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(w.dtype == x.dtype && (bias == nullptr || bias->dtype == x.dtype),
           "rocm causal_conv1d_fwd: weight/bias dtype must match x");
  const int64_t n = conv_state.shape[0];
  if (n == 0 || x.shape[1] == 0) return;
  hipStream_t s = AsStream(q);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      LaunchConvFwd<float, float>(s, out, x, w, bias, conv_state, qsl, his, args);
    } else {
      LaunchConvFwd<float, __hip_bfloat16>(s, out, x, w, bias, conv_state, qsl, his, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchConvFwd<__hip_bfloat16, float>(s, out, x, w, bias, conv_state, qsl, his, args);
    } else {
      LaunchConvFwd<__hip_bfloat16, __hip_bfloat16>(s, out, x, w, bias, conv_state, qsl, his, args);
    }
  }
}

void CausalConv1dUpdateKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                                  const Tensor* bias, Tensor& conv_state,
                                  const Tensor* conv_state_indices,
                                  const CausalConv1dArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "rocm causal_conv1d_update: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(w.dtype == x.dtype && (bias == nullptr || bias->dtype == x.dtype),
           "rocm causal_conv1d_update: weight/bias dtype must match x");
  if (x.shape[0] * x.shape[1] == 0) return;
  const int32_t* ci =
      conv_state_indices != nullptr ? conv_state_indices->Ptr<int32_t>() : nullptr;
  hipStream_t s = AsStream(q);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdateS<float, float>(s, out, x, w, bias, conv_state, ci, args);
    } else {
      LaunchConvUpdateS<float, __hip_bfloat16>(s, out, x, w, bias, conv_state, ci, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdateS<__hip_bfloat16, float>(s, out, x, w, bias, conv_state, ci, args);
    } else {
      LaunchConvUpdateS<__hip_bfloat16, __hip_bfloat16>(s, out, x, w, bias, conv_state, ci, args);
    }
  }
}

}  // namespace vt::rocm
