// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA GDN kernels (M0.7): causal conv1d fwd/update, l2norm, gated rmsnorm,
// gated-delta-rule scan. Correctness-grade — plain kernels matching the CPU
// reference math in src/vt/cpu/cpu_ops.cpp element for element; formulas from
// .agents/gdn-semantics.md (§ cited per kernel). Perf-grade (chunked prefill)
// lands M2.3.
//
// Metadata contract (M0.7 Task 2 review): query_start_loc / has_initial_state
// live on the OP'S DEVICE. These kernels read them device-side; the host
// wrappers never copy them back (that would force a stream sync). The CPU
// reference validates qsl bounds/monotonicity host-side; here bad metadata is
// unchecked (correctness-grade — the M0.9 builder owns metadata integrity).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;
// CUDA grid.y is limited to 65535; sequence/batch counts map to grid.y below.
constexpr int64_t kMaxGridY = 65535;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

// f32 load/store overloads: bf16 converts on the way in/out, math is f32.
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, same as host F32ToBF16
}

__device__ inline float Silu(float x) { return x / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// causal_conv1d_fwd (gdn-semantics.md §2): one thread per (sequence, channel),
// sequential over the sequence's tokens. Each thread owns its conv_state row
// [K-1] exclusively, so outputs (which read the OLD state for the first K-1
// tokens) and the write-back are ordered by plain program order — no barriers.
// The write-back needs no old-row buffer: writing ascending j reads old cols
// at index t_len + j >= j, so no already-written slot is ever read (same
// values the CPU reference buffers into old_row).
// Upstream counterpart: layers/mamba/ops/causal_conv1d.py (causal_conv1d_fn
// Triton kernel) — align post-MVP.

template <typename Tin, typename Tout>
__global__ void CausalConv1dFwdKernel(Tout* out, const Tin* x, const Tin* w, const Tin* bias,
                                      float* conv_state, const int32_t* qsl,
                                      const int32_t* his, int64_t c_dim, int64_t k,
                                      bool silu) {
  const int64_t s = blockIdx.y;
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= c_dim) return;
  const int64_t width = k - 1;
  const int64_t begin = qsl[s];
  const int64_t t_len = qsl[s + 1] - begin;
  const bool init = his[s] != 0;
  float* srow = conv_state + (s * c_dim + c) * width;
  const float b = bias != nullptr ? Load(bias, c) : 0.0f;
  for (int64_t t = 0; t < t_len; ++t) {
    float acc = b;
    for (int64_t j = 0; j < k; ++j) {
      const int64_t ti = t - (k - 1 - j);  // token index of window[j]
      float v = 0.0f;
      if (ti >= 0) {
        v = Load(x, (begin + ti) * c_dim + c);
      } else if (init) {
        v = srow[width + ti];  // old state col (K-1)+(t-i); not yet overwritten
      }
      acc += Load(w, c * k + j) * v;
    }
    Store(out, (begin + t) * c_dim + c, silu ? Silu(acc) : acc);
  }
  // State write-back: last K-1 RAW x tokens, left-padded with zeros (no init
  // state) or shifted old state when T < K-1.
  for (int64_t j = 0; j < width; ++j) {
    const int64_t tj = t_len - width + j;  // new state col j holds token tj
    float v = 0.0f;
    if (tj >= 0) {
      v = Load(x, (begin + tj) * c_dim + c);
    } else if (init) {
      v = srow[width + tj];  // shifted old state; index t_len+j >= j (unwritten)
    }
    srow[j] = v;
  }
}

template <typename Tin, typename Tout>
void LaunchConvFwd(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                   const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                   const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t n = conv_state.shape[0], c = x.shape[1], k = w.shape[1];
  const dim3 grid(static_cast<unsigned>((c + kBlock - 1) / kBlock), static_cast<unsigned>(n));
  CausalConv1dFwdKernel<Tin, Tout><<<grid, kBlock, 0, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), bias != nullptr ? bias->Ptr<Tin>() : nullptr,
      conv_state.Ptr<float>(), qsl.Ptr<int32_t>(), his.Ptr<int32_t>(), c, k,
      args.silu_activation);
  Check(cudaGetLastError(), "causal_conv1d_fwd launch");
}

// ---------------------------------------------------------------------------
// causal_conv1d_update (gdn-semantics.md §3): grid-stride, one thread per
// (token, channel). Read-old-then-roll on the thread's own state row.
// Upstream counterpart: layers/mamba/ops/causal_conv1d.py
// (causal_conv1d_update Triton kernel, seqlen==1 path) — align post-MVP.

template <typename Tin, typename Tout>
__global__ void CausalConv1dUpdateKernel(Tout* out, const Tin* x, const Tin* w,
                                         const Tin* bias, float* conv_state, int64_t n,
                                         int64_t c_dim, int64_t k, bool silu) {
  const int64_t width = k - 1;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t c = idx % c_dim;
    float* srow = conv_state + idx * width;  // idx = bt * c_dim + c; row [K-1]
    const float xt = Load(x, idx);
    float acc = bias != nullptr ? Load(bias, c) : 0.0f;
    for (int64_t j = 0; j < width; ++j) acc += Load(w, c * k + j) * srow[j];
    acc += Load(w, c * k + width) * xt;
    Store(out, idx, silu ? Silu(acc) : acc);
    for (int64_t j = 0; j + 1 < width; ++j) srow[j] = srow[j + 1];  // roll left
    if (width > 0) srow[width - 1] = xt;                            // raw x
  }
}

template <typename Tin, typename Tout>
void LaunchConvUpdate(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const Tensor* bias, Tensor& conv_state, const CausalConv1dArgs& args) {
  const int64_t n = x.shape[0] * x.shape[1], c = x.shape[1], k = w.shape[1];
  CausalConv1dUpdateKernel<Tin, Tout><<<GridFor(n), kBlock, 0, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), bias != nullptr ? bias->Ptr<Tin>() : nullptr,
      conv_state.Ptr<float>(), n, c, k, args.silu_activation);
  Check(cudaGetLastError(), "causal_conv1d_update launch");
}

void ConvFwdKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                       const Tensor& his, const CausalConv1dArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda causal_conv1d_fwd: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(w.dtype == x.dtype && (bias == nullptr || bias->dtype == x.dtype),
           "cuda causal_conv1d_fwd: weight/bias dtype must match x");
  const int64_t n = conv_state.shape[0];
  if (n == 0 || x.shape[1] == 0) return;
  VT_CHECK(n <= kMaxGridY, "cuda causal_conv1d_fwd: too many sequences (grid.y limit)");
  cudaStream_t s = AsStream(q);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      LaunchConvFwd<float, float>(s, out, x, w, bias, conv_state, qsl, his, args);
    } else {
      LaunchConvFwd<float, __nv_bfloat16>(s, out, x, w, bias, conv_state, qsl, his, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchConvFwd<__nv_bfloat16, float>(s, out, x, w, bias, conv_state, qsl, his, args);
    } else {
      LaunchConvFwd<__nv_bfloat16, __nv_bfloat16>(s, out, x, w, bias, conv_state, qsl, his,
                                                  args);
    }
  }
}

void ConvUpdateKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                          const Tensor* bias, Tensor& conv_state,
                          const CausalConv1dArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda causal_conv1d_update: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(w.dtype == x.dtype && (bias == nullptr || bias->dtype == x.dtype),
           "cuda causal_conv1d_update: weight/bias dtype must match x");
  if (x.shape[0] * x.shape[1] == 0) return;
  cudaStream_t s = AsStream(q);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdate<float, float>(s, out, x, w, bias, conv_state, args);
    } else {
      LaunchConvUpdate<float, __nv_bfloat16>(s, out, x, w, bias, conv_state, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdate<__nv_bfloat16, float>(s, out, x, w, bias, conv_state, args);
    } else {
      LaunchConvUpdate<__nv_bfloat16, __nv_bfloat16>(s, out, x, w, bias, conv_state, args);
    }
  }
}

// ---------------------------------------------------------------------------
// l2norm (gdn-semantics.md §4): one block per row, shared-memory f32 tree
// reduction (M0.6 rmsnorm pattern). Plain SUM of squares — not a mean.
// Upstream counterpart: layers/fla/ops/l2norm.py (l2norm_fwd_kernel2) —
// align post-MVP.

template <typename Tin, typename Tout>
__global__ void L2NormRowKernel(Tout* out, const Tin* x, int64_t d, float eps) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * d;
  Tout* orow = out + row * d;
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += kBlock) {
    const float v = Load(xrow, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] + eps);
  for (int64_t j = threadIdx.x; j < d; j += kBlock) Store(orow, j, Load(xrow, j) * inv);
}

void L2NormKernelCuda(Queue& q, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda l2norm: unsupported input dtype (f32/bf16 only)");
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  if (rows == 0 || d == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned grid = static_cast<unsigned>(rows);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      L2NormRowKernel<float, float>
          <<<grid, kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<float>(), d, args.eps);
    } else {
      L2NormRowKernel<float, __nv_bfloat16>
          <<<grid, kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<float>(), d, args.eps);
    }
  } else {
    if (out.dtype == DType::kF32) {
      L2NormRowKernel<__nv_bfloat16, float>
          <<<grid, kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<__nv_bfloat16>(), d, args.eps);
    } else {
      L2NormRowKernel<__nv_bfloat16, __nv_bfloat16>
          <<<grid, kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<__nv_bfloat16>(), d,
                                   args.eps);
    }
  }
  Check(cudaGetLastError(), "l2norm launch");
}

// ---------------------------------------------------------------------------
// rmsnorm_gated (gdn-semantics.md §5): one block per row, same reduction; var
// is a MEAN (unlike §4), norm first, then act(gate) (norm_before_gate=True,
// group_size=None — the only configuration Qwen GDN uses, baked in).
// Upstream counterpart: layers/fla/ops/layernorm_guard.py
// (layer_norm_fwd_kernel via RMSNormGated) — align post-MVP.

template <typename Tin, typename Tout>
__global__ void RmsNormGatedRowKernel(Tout* out, const Tin* x, const Tin* gate, const Tin* w,
                                      int64_t d, float eps, bool sigmoid_gate) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * d;
  const Tin* zrow = gate + row * d;
  Tout* orow = out + row * d;
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += kBlock) {
    const float v = Load(xrow, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(d) + eps);
  for (int64_t j = threadIdx.x; j < d; j += kBlock) {
    const float z = Load(zrow, j);
    const float act = sigmoid_gate ? 1.0f / (1.0f + expf(-z)) : Silu(z);
    Store(orow, j, Load(xrow, j) * inv * Load(w, j) * act);
  }
}

void RmsNormGatedKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                            const Tensor& w, const RmsNormGatedArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda rmsnorm_gated: unsupported input dtype (f32/bf16 only)");
  VT_CHECK(gate.dtype == x.dtype && w.dtype == x.dtype,
           "cuda rmsnorm_gated: gate/weight dtype must match x");
  const int64_t t = x.shape[0], d = x.shape[1];
  if (t == 0 || d == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned grid = static_cast<unsigned>(t);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      RmsNormGatedRowKernel<float, float><<<grid, kBlock, 0, s>>>(
          out.Ptr<float>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), d, args.eps,
          args.sigmoid_gate);
    } else {
      RmsNormGatedRowKernel<float, __nv_bfloat16><<<grid, kBlock, 0, s>>>(
          out.Ptr<__nv_bfloat16>(), x.Ptr<float>(), gate.Ptr<float>(), w.Ptr<float>(), d,
          args.eps, args.sigmoid_gate);
    }
  } else {
    if (out.dtype == DType::kF32) {
      RmsNormGatedRowKernel<__nv_bfloat16, float><<<grid, kBlock, 0, s>>>(
          out.Ptr<float>(), x.Ptr<__nv_bfloat16>(), gate.Ptr<__nv_bfloat16>(),
          w.Ptr<__nv_bfloat16>(), d, args.eps, args.sigmoid_gate);
    } else {
      RmsNormGatedRowKernel<__nv_bfloat16, __nv_bfloat16><<<grid, kBlock, 0, s>>>(
          out.Ptr<__nv_bfloat16>(), x.Ptr<__nv_bfloat16>(), gate.Ptr<__nv_bfloat16>(),
          w.Ptr<__nv_bfloat16>(), d, args.eps, args.sigmoid_gate);
    }
  }
  Check(cudaGetLastError(), "rmsnorm_gated launch");
}

// ---------------------------------------------------------------------------
// Gated-delta-rule scan (gdn-semantics.md §7), shared by prefill and decode:
// one BLOCK per (sequence, v-head), threads parallelize over Dv state rows,
// sequential over the sequence's tokens. The f32 state [Dv,Dk] stays in
// GLOBAL memory, updated in place on the state tensor (Dv*Dk*4B = 64KB at
// real dims exceeds the 48KB default shared-memory budget).
//
// Per token: q' = q*scale; S *= exp(g); v' = (v - S@k)*beta;
// S += outer(v',k); o = S@q'. Thread `vi` owns state row vi EXCLUSIVELY:
// S@k is a row-dot with its own row, the decay/rank-1 updates touch only its
// own row, and o[vi] is its own row dotted with q' — no cross-thread hazard
// on S, so no barriers are needed in the row phase. The only barriers guard
// the shared q'/k staging: one after the cooperative load (all threads must
// see the full vectors) and one at the end of the token iteration (no thread
// may overwrite q'/k for token t+1 while another still reads them for t).
//
// qsl == nullptr selects decode: block y is the token index, token range
// [y, y+1), state row y (one single-token sequence per batch row).
// Upstream counterpart: layers/fla/ops/{fused_recurrent,fused_sigmoid_gating}.py
// Triton kernels (chunked variant: chunk.py, lands M2.3) — align post-MVP.

template <typename Tin, typename Tout>
__global__ void GdnScanKernel(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                              const float* g, const float* beta, float* state,
                              const int32_t* qsl, int64_t hk_n, int64_t dk, int64_t hv_n,
                              int64_t dv, float scale) {
  const int64_t s = blockIdx.y;   // sequence (prefill) or token (decode)
  const int64_t hv = blockIdx.x;  // v-head
  const int64_t hk = hv / (hv_n / hk_n);
  extern __shared__ float smem[];  // [dk] q' then [dk] k
  float* q_sh = smem;
  float* k_sh = smem + dk;
  float* s_head = state + (s * hv_n + hv) * dv * dk;  // [Dv, Dk]
  const int64_t begin = qsl != nullptr ? qsl[s] : s;
  const int64_t end = qsl != nullptr ? qsl[s + 1] : s + 1;
  for (int64_t t = begin; t < end; ++t) {
    for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
      q_sh[i] = Load(q, (t * hk_n + hk) * dk + i) * scale;
      k_sh[i] = Load(k, (t * hk_n + hk) * dk + i);
    }
    __syncthreads();
    const float decay = expf(g[t * hv_n + hv]);
    const float beta_t = beta[t * hv_n + hv];
    for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
      float* s_row = s_head + vi * dk;
      float dot = 0.0f;  // (S * exp(g)) @ k, fused with the decay pass
      for (int64_t ki = 0; ki < dk; ++ki) {
        s_row[ki] *= decay;
        dot += s_row[ki] * k_sh[ki];
      }
      const float vp = (Load(v, (t * hv_n + hv) * dv + vi) - dot) * beta_t;
      float o = 0.0f;  // (S + outer(v',k)) @ q', fused with the rank-1 update
      for (int64_t ki = 0; ki < dk; ++ki) {
        s_row[ki] += vp * k_sh[ki];
        o += s_row[ki] * q_sh[ki];
      }
      Store(out, (t * hv_n + hv) * dv + vi, o);
    }
    __syncthreads();  // all reads of q_sh/k_sh done before the next token's load
  }
}

template <typename Tin, typename Tout>
void LaunchGdnScan(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                   const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                   const int32_t* qsl, int64_t n, const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const dim3 grid(static_cast<unsigned>(hv_n), static_cast<unsigned>(n));
  const size_t shmem = 2 * static_cast<size_t>(dk) * sizeof(float);
  GdnScanKernel<Tin, Tout><<<grid, kBlock, shmem, s>>>(
      out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
      beta.Ptr<float>(), state.Ptr<float>(), qsl, hk_n, dk, hv_n, dv, args.scale);
  Check(cudaGetLastError(), "gdn scan launch");
}

// Shared wrapper body: qsl_ptr == nullptr → decode (n = batch = state rows;
// ops.cpp validated state.shape[0] == T for decode / query_start_loc [N+1] on
// the queue's device for prefill).
void GdnScanCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                 const Tensor& g, const Tensor& beta, Tensor& state, const int32_t* qsl_ptr,
                 const GdnArgs& args, const char* name) {
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           std::string("cuda ") + name + ": unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           std::string("cuda ") + name + ": q/k/v dtypes must match");
  const int64_t n = state.shape[0], hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  if (n == 0 || hv_n == 0 || dv == 0) return;
  VT_CHECK(n <= kMaxGridY, std::string("cuda ") + name + ": too many sequences (grid.y limit)");
  VT_CHECK(2 * static_cast<size_t>(dk) * sizeof(float) <= 48 * 1024,
           std::string("cuda ") + name + ": Dk too large for the shared q'/k staging");
  cudaStream_t s = AsStream(q);
  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      LaunchGdnScan<float, float>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n, args);
    } else {
      LaunchGdnScan<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n,
                                          args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchGdnScan<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, qsl_ptr, n,
                                          args);
    } else {
      LaunchGdnScan<__nv_bfloat16, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, qsl_ptr,
                                                  n, args);
    }
  }
}

void GdnPrefillKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const Tensor& qsl, const GdnArgs& args) {
  GdnScanCuda(q, out, q_in, k, v, g, beta, state, qsl.Ptr<int32_t>(), args, "gdn_prefill");
}

void GdnDecodeKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                         const GdnArgs& args) {
  GdnScanCuda(q, out, q_in, k, v, g, beta, state, nullptr, args, "gdn_decode");
}

// Registers the CUDA GDN kernels during static init (pre-main, like the M0.6
// ops in cuda_ops.cu). Filling the op table is harmless on machines without a
// GPU: the kCUDA backend never registers there, so no CUDA queue can exist to
// dispatch with.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&ConvFwdKernelCuda)));
    RegisterOp(
        OpId::kCausalConv1dUpdate, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&ConvUpdateKernelCuda)));
    RegisterOp(OpId::kL2Norm, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<L2NormFn>(&L2NormKernelCuda)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernelCuda)));
    RegisterOp(OpId::kGdnPrefill, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernelCuda)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
