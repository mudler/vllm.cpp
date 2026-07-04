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

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;
// CUDA grid.y is limited to 65535; sequence/batch counts map to grid.y below.
constexpr int64_t kMaxGridY = 65535;
// FLA chunk size (fla/ops/utils.py FLA_CHUNK_SIZE=64). The chunked GDN prefill
// scan (below) processes tokens in chunks of this many; the intra-chunk work is
// parallel across chunks and the cross-chunk state recurrence is sequential.
constexpr int kChunk = 64;
// Head-dim ceiling for the chunked path: local per-column arrays are sized to
// kChunk and the delta-h kernel stages the [Dv,Dk] state in shared memory
// (Dv*Dk*4B). Dk=Dv=128 is the real gate dim; larger dims fall back to the
// sequential scan (which strides arbitrary dims).
constexpr int64_t kChunkMaxDim = 128;

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

// ===========================================================================
// Chunk-parallel GDN prefill scan (gdn-semantics.md §7 "chunked oracle").
//
// Mirrors vLLM's FLA chunked gated delta rule (fla/ops/chunk.py + sub-ops).
// The sequential GdnScanKernel carries the [Dv,Dk] state one token at a time,
// so a T-token prefill has sequential depth T. This path splits the sequence
// into chunks of kChunk tokens: the intra-chunk work (steps A/B/D below) is
// INDEPENDENT across chunks (parallel across the whole grid), and only the
// cross-chunk state recurrence (step C) is sequential — depth ceil(T/kChunk).
//
// Math (derived from the sequential recurrence; exact up to fp reassociation,
// validated chunked-vs-sequential on GB10):
//   G = local cumsum of the log-decay g within each chunk (inclusive).
//   A[i,j] = beta_i * exp(G_i - G_j) * (k_i . k_j),  j<i   (chunk_scaled_dot_kkt)
//   u = (I+A)^{-1} (beta (.) v)                            (solve_tril+recompute_w_u)
//   w = (I+A)^{-1} (beta (.) exp(G) (.) k)                 (fused via fwd-subst)
//   v_new = u - w @ h_start^T                              (chunk_delta_h)
//   h_end = h_start*exp(G_last) + sum_i k_i (x) v_new_i*exp(G_last-G_i)
//   o_i   = scale * exp(G_i) * ( q_i.h_start[v]
//                              + sum_{j<=i} exp(-G_j)(q_i.k_j) v_new_j )  (chunk_fwd_o)
// (I+A) is unit lower-triangular so u,w are obtained by forward substitution,
// which replaces the explicit solve_tril inverse — numerically equivalent, and
// still fully chunk-parallel because each column solves independently.
//
// vllm.cpp original (inventory deviation §9): an original fused CUDA kernel set
// structured to map onto the FLA sub-ops (cumsum / scaled_dot_kkt+solve_tril+
// wy / chunk_delta_h / chunk_fwd_o) so a future csrc/FLA drop-in is mechanical.

bool ChunkedPrefillEnabled() {
  // A/B toggle: default ON. VT_GDN_CHUNKED=0 falls back to the sequential scan.
  // Read each call (prefill is coarse-grained, so getenv cost is negligible and
  // it lets the unit test drive both paths in one process).
  const char* e = std::getenv("VT_GDN_CHUNKED");
  return e == nullptr || e[0] != '0';
}

// Step A — chunk_local_cumsum (fla/ops/cumsum.py): inclusive prefix sum of g
// within each chunk, per (chunk, v-head). One block per (chunk, v-head).
__global__ void GdnChunkCumsumKernel(float* gcum, const float* g, const int32_t* tok0a,
                                     const int32_t* lena, int64_t hv_n) {
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  if (threadIdx.x != 0) return;  // tiny (<=kChunk) serial scan per chunk-head
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  float acc = 0.0f;
  for (int64_t i = 0; i < len; ++i) {
    acc += g[(tok0 + i) * hv_n + hv];
    gcum[(tok0 + i) * hv_n + hv] = acc;
  }
}

// Step B — chunk_scaled_dot_kkt + solve_tril + recompute_w_u fused
// (fla/ops/{chunk_scaled_dot_kkt,solve_tril,wy_fast}.py). One block per
// (chunk, v-head): build K K^T in shared, then forward-substitute the two
// WY columns u (from beta.v) and w (from beta.exp(G).k). Each output column
// (vi for u, ki for w) is an independent forward solve → no cross-thread
// dependency in the substitution.
template <typename Tin>
__global__ void GdnChunkWUKernel(float* u, float* w, const Tin* k, const Tin* v,
                                 const float* beta, const float* gcum, const int32_t* tok0a,
                                 const int32_t* lena, int64_t hk_n, int64_t dk, int64_t hv_n,
                                 int64_t dv) {
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  __shared__ float kk[kChunk * kChunk];  // K K^T (row i, col j)
  __shared__ float bs[kChunk];           // beta
  __shared__ float eg[kChunk];           // exp(G_i)
  __shared__ float egi[kChunk];          // 1/exp(G_i)
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
    bs[i] = beta[(tok0 + i) * hv_n + hv];
    const float e = expf(gcum[(tok0 + i) * hv_n + hv]);
    eg[i] = e;
    egi[i] = 1.0f / e;
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * len; idx += blockDim.x) {
    const int64_t i = idx / len, j = idx % len;
    float dot = 0.0f;
    const int64_t ib = (tok0 + i) * hk_n * dk + hk * dk;
    const int64_t jb = (tok0 + j) * hk_n * dk + hk * dk;
    for (int64_t d = 0; d < dk; ++d) dot += Load(k, ib + d) * Load(k, jb + d);
    kk[i * kChunk + j] = dot;
  }
  __syncthreads();
  // u column vi: u_i = beta_i (v_i - eG_i * sum_{j<i} KK[i,j] * (u_j/eG_j))
  for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
    float ucol[kChunk], us[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float s = 0.0f;
      for (int64_t j = 0; j < i; ++j) s += kk[i * kChunk + j] * us[j];
      const float vi_val = Load(v, (tok0 + i) * hv_n * dv + hv * dv + vi);
      const float ui = bs[i] * (vi_val - eg[i] * s);
      ucol[i] = ui;
      us[i] = ui * egi[i];
    }
    for (int64_t i = 0; i < len; ++i) u[(tok0 + i) * hv_n * dv + hv * dv + vi] = ucol[i];
  }
  // w column ki: w_i = beta_i eG_i (k_i - sum_{j<i} KK[i,j] * (w_j/eG_j))
  for (int64_t ki = threadIdx.x; ki < dk; ki += blockDim.x) {
    float wcol[kChunk], ws[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float s = 0.0f;
      for (int64_t j = 0; j < i; ++j) s += kk[i * kChunk + j] * ws[j];
      const float ki_val = Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki);
      const float wi = bs[i] * eg[i] * (ki_val - s);
      wcol[i] = wi;
      ws[i] = wi * egi[i];
    }
    for (int64_t i = 0; i < len; ++i) w[(tok0 + i) * hv_n * dk + hv * dk + ki] = wcol[i];
  }
}

// Step C — chunk_delta_h (fla/ops/chunk_delta_h.py): the cross-chunk state
// recurrence. One block per (sequence, v-head), SEQUENTIAL over the sequence's
// chunks. The running [Dv,Dk] state lives in shared memory; each chunk snapshots
// its start state into hstate (consumed by step D), emits v_new, and advances
// the state. This is the only sequential-across-chunks kernel.
template <typename Tin>
__global__ void GdnChunkDeltaHKernel(float* state, float* hstate, float* v_new, const Tin* k,
                                     const float* u, const float* w, const float* gcum,
                                     const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                     int64_t dk, int64_t hv_n, int64_t dv) {
  const int64_t n = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  extern __shared__ float hsh[];  // [Dv, Dk] running state
  float* s_head = state + (n * hv_n + hv) * sd;
  for (int64_t e = threadIdx.x; e < sd; e += blockDim.x) hsh[e] = s_head[e];
  __syncthreads();
  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * kChunk;
    const int64_t rem = seqlen - it * kChunk;
    const int64_t len = rem < kChunk ? rem : kChunk;
    float* hstart = hstate + gc * sd;  // chunk-start snapshot for step D
    for (int64_t e = threadIdx.x; e < sd; e += blockDim.x) hstart[e] = hsh[e];
    __syncthreads();
    // v_new[i,vi] = u[i,vi] - sum_ki w[i,ki] * hsh[vi,ki]
    for (int64_t idx = threadIdx.x; idx < len * dv; idx += blockDim.x) {
      const int64_t i = idx / dv, vi = idx % dv;
      float acc = u[(tok0 + i) * hv_n * dv + hv * dv + vi];
      const int64_t wb = (tok0 + i) * hv_n * dk + hv * dk;
      const float* hrow = hsh + vi * dk;
      for (int64_t d = 0; d < dk; ++d) acc -= w[wb + d] * hrow[d];
      v_new[(tok0 + i) * hv_n * dv + hv * dv + vi] = acc;
    }
    __syncthreads();
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    // hsh[vi,ki] = hsh[vi,ki]*exp(G_last) + sum_i k[i,ki]*v_new[i,vi]*exp(G_last-G_i)
    for (int64_t idx = threadIdx.x; idx < sd; idx += blockDim.x) {
      const int64_t vi = idx / dk, ki = idx % dk;
      float acc = hsh[idx] * eglast;
      for (int64_t i = 0; i < len; ++i) {
        const float decay = eglast * expf(-gcum[(tok0 + i) * hv_n + hv]);
        acc += Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki) *
               v_new[(tok0 + i) * hv_n * dv + hv * dv + vi] * decay;
      }
      hsh[idx] = acc;
    }
    __syncthreads();
  }
  for (int64_t e = threadIdx.x; e < sd; e += blockDim.x) s_head[e] = hsh[e];  // final state
}

// Step D — chunk_fwd_o (fla/ops/chunk_o.py): the output. One block per
// (chunk, v-head): the cross-chunk term q.h_start plus the intra-chunk causal
// term sum_{j<=i} exp(G_i-G_j)(q_i.k_j) v_new_j, both scaled.
template <typename Tin, typename Tout>
__global__ void GdnChunkOKernel(Tout* out, const Tin* q, const Tin* k, const float* v_new,
                                const float* hstate, const float* gcum, const int32_t* tok0a,
                                const int32_t* lena, int64_t hk_n, int64_t dk, int64_t hv_n,
                                int64_t dv, float scale) {
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int64_t sd = dv * dk;
  const float* hstart = hstate + gc * sd;
  __shared__ float qk[kChunk * kChunk];  // q_i . k_j
  __shared__ float eg[kChunk];           // exp(G_i)
  __shared__ float egi[kChunk];          // 1/exp(G_i)
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
    const float e = expf(gcum[(tok0 + i) * hv_n + hv]);
    eg[i] = e;
    egi[i] = 1.0f / e;
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * len; idx += blockDim.x) {
    const int64_t i = idx / len, j = idx % len;
    float dot = 0.0f;
    const int64_t ib = (tok0 + i) * hk_n * dk + hk * dk;
    const int64_t jb = (tok0 + j) * hk_n * dk + hk * dk;
    for (int64_t d = 0; d < dk; ++d) dot += Load(q, ib + d) * Load(k, jb + d);
    qk[i * kChunk + j] = dot;
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * dv; idx += blockDim.x) {
    const int64_t i = idx / dv, vi = idx % dv;
    // cross-chunk: q_i . h_start[vi,:]
    float cross = 0.0f;
    const int64_t qb = (tok0 + i) * hk_n * dk + hk * dk;
    const float* hrow = hstart + vi * dk;
    for (int64_t d = 0; d < dk; ++d) cross += Load(q, qb + d) * hrow[d];
    // intra-chunk causal: sum_{j<=i} QK[i,j] * (v_new_j / eG_j)
    float intra = 0.0f;
    for (int64_t j = 0; j <= i; ++j)
      intra += qk[i * kChunk + j] * egi[j] * v_new[(tok0 + j) * hv_n * dv + hv * dv + vi];
    const float o = scale * eg[i] * (cross + intra);
    Store(out, (tok0 + i) * hv_n * dv + hv * dv + vi, o);
  }
}

template <typename Tin, typename Tout>
void LaunchChunkedPrefill(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const std::vector<int32_t>& tok0h,
                          const std::vector<int32_t>& lenh, const std::vector<int32_t>& bohh,
                          const Tensor& qsl, int64_t n_seq, int64_t nt_tot, const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const int64_t t_tot = out.shape[0];

  // Device scratch (stream-ordered; freed after the kernels on the same stream).
  float* gcum = nullptr;
  float* u = nullptr;
  float* w = nullptr;
  float* v_new = nullptr;
  float* hstate = nullptr;
  int32_t* d_tok0 = nullptr;
  int32_t* d_len = nullptr;
  int32_t* d_boh = nullptr;
  Check(cudaMallocAsync(&gcum, static_cast<size_t>(t_tot * hv_n) * sizeof(float), s),
        "gdn chunked gcum alloc");
  Check(cudaMallocAsync(&u, static_cast<size_t>(t_tot * hv_n * dv) * sizeof(float), s),
        "gdn chunked u alloc");
  Check(cudaMallocAsync(&w, static_cast<size_t>(t_tot * hv_n * dk) * sizeof(float), s),
        "gdn chunked w alloc");
  Check(cudaMallocAsync(&v_new, static_cast<size_t>(t_tot * hv_n * dv) * sizeof(float), s),
        "gdn chunked v_new alloc");
  Check(cudaMallocAsync(&hstate, static_cast<size_t>(nt_tot * hv_n * dv * dk) * sizeof(float), s),
        "gdn chunked hstate alloc");
  Check(cudaMallocAsync(&d_tok0, static_cast<size_t>(nt_tot) * sizeof(int32_t), s),
        "gdn chunked tok0 alloc");
  Check(cudaMallocAsync(&d_len, static_cast<size_t>(nt_tot) * sizeof(int32_t), s),
        "gdn chunked len alloc");
  Check(cudaMallocAsync(&d_boh, static_cast<size_t>(n_seq) * sizeof(int32_t), s),
        "gdn chunked boh alloc");
  Check(cudaMemcpyAsync(d_tok0, tok0h.data(), static_cast<size_t>(nt_tot) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, s),
        "gdn chunked tok0 upload");
  Check(cudaMemcpyAsync(d_len, lenh.data(), static_cast<size_t>(nt_tot) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, s),
        "gdn chunked len upload");
  Check(cudaMemcpyAsync(d_boh, bohh.data(), static_cast<size_t>(n_seq) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, s),
        "gdn chunked boh upload");
  // The host index vectors must outlive the async uploads; the caller syncs the
  // stream before they go out of scope.

  const dim3 grid_chunk(static_cast<unsigned>(nt_tot), static_cast<unsigned>(hv_n));
  const dim3 grid_seq(static_cast<unsigned>(n_seq), static_cast<unsigned>(hv_n));

  GdnChunkCumsumKernel<<<grid_chunk, 32, 0, s>>>(gcum, g.Ptr<float>(), d_tok0, d_len, hv_n);
  Check(cudaGetLastError(), "gdn chunked cumsum launch");

  GdnChunkWUKernel<Tin><<<grid_chunk, 128, 0, s>>>(u, w, k.Ptr<Tin>(), v.Ptr<Tin>(),
                                                   beta.Ptr<float>(), gcum, d_tok0, d_len, hk_n,
                                                   dk, hv_n, dv);
  Check(cudaGetLastError(), "gdn chunked wu launch");

  const size_t hsh_bytes = static_cast<size_t>(dv * dk) * sizeof(float);
  auto* delta_kernel = GdnChunkDeltaHKernel<Tin>;
  if (hsh_bytes > 48 * 1024) {
    Check(cudaFuncSetAttribute(delta_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(hsh_bytes)),
          "gdn chunked delta_h shared opt-in");
  }
  delta_kernel<<<grid_seq, 256, hsh_bytes, s>>>(state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(),
                                                u, w, gcum, qsl.Ptr<int32_t>(), d_boh, hk_n, dk,
                                                hv_n, dv);
  Check(cudaGetLastError(), "gdn chunked delta_h launch");

  GdnChunkOKernel<Tin, Tout><<<grid_chunk, 256, 0, s>>>(out.Ptr<Tout>(), q_in.Ptr<Tin>(),
                                                        k.Ptr<Tin>(), v_new, hstate, gcum,
                                                        d_tok0, d_len, hk_n, dk, hv_n, dv,
                                                        args.scale);
  Check(cudaGetLastError(), "gdn chunked o launch");

  Check(cudaFreeAsync(gcum, s), "gdn chunked gcum free");
  Check(cudaFreeAsync(u, s), "gdn chunked u free");
  Check(cudaFreeAsync(w, s), "gdn chunked w free");
  Check(cudaFreeAsync(v_new, s), "gdn chunked v_new free");
  Check(cudaFreeAsync(hstate, s), "gdn chunked hstate free");
  Check(cudaFreeAsync(d_tok0, s), "gdn chunked tok0 free");
  Check(cudaFreeAsync(d_len, s), "gdn chunked len free");
  Check(cudaFreeAsync(d_boh, s), "gdn chunked boh free");
}

void GdnPrefillChunkedCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                           const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                           const Tensor& qsl, const GdnArgs& args) {
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           "cuda gdn_prefill(chunked): unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           "cuda gdn_prefill(chunked): q/k/v dtypes must match");
  const int64_t n_seq = state.shape[0], hv_n = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  if (n_seq == 0 || hv_n == 0 || dv == 0) return;
  const int64_t hk_n = q_in.shape[1];
  VT_CHECK(hk_n > 0 && hv_n % hk_n == 0,
           "cuda gdn_prefill(chunked): Hv must be a multiple of Hk");
  cudaStream_t s = AsStream(q);

  // Host chunk layout from query_start_loc (small D2H copy + sync — negligible
  // vs a whole prefill; the sequential path avoids it, this path needs the
  // per-sequence chunk counts to size scratch and grids).
  std::vector<int32_t> qslh(static_cast<size_t>(n_seq) + 1);
  Check(cudaMemcpyAsync(qslh.data(), qsl.Ptr<int32_t>(),
                        (static_cast<size_t>(n_seq) + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                        s),
        "gdn chunked qsl download");
  Check(cudaStreamSynchronize(s), "gdn chunked qsl sync");

  std::vector<int32_t> tok0h, lenh, bohh(static_cast<size_t>(n_seq));
  int32_t nt_tot = 0;
  for (int64_t n = 0; n < n_seq; ++n) {
    bohh[static_cast<size_t>(n)] = nt_tot;
    const int32_t begin = qslh[static_cast<size_t>(n)];
    const int32_t len = qslh[static_cast<size_t>(n) + 1] - begin;
    const int32_t nt = (len + kChunk - 1) / kChunk;
    for (int32_t it = 0; it < nt; ++it) {
      const int32_t rem = len - it * kChunk;
      tok0h.push_back(begin + it * kChunk);
      lenh.push_back(rem < kChunk ? rem : kChunk);
    }
    nt_tot += nt;
  }
  if (nt_tot == 0) return;

  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32)
      LaunchChunkedPrefill<float, float>(s, out, q_in, k, v, g, beta, state, tok0h, lenh, bohh,
                                         qsl, n_seq, nt_tot, args);
    else
      LaunchChunkedPrefill<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, tok0h, lenh,
                                                 bohh, qsl, n_seq, nt_tot, args);
  } else {
    if (out.dtype == DType::kF32)
      LaunchChunkedPrefill<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, tok0h, lenh,
                                                 bohh, qsl, n_seq, nt_tot, args);
    else
      LaunchChunkedPrefill<__nv_bfloat16, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, tok0h,
                                                         lenh, bohh, qsl, n_seq, nt_tot, args);
  }
  // Keep the host index vectors alive until every async upload has completed.
  Check(cudaStreamSynchronize(s), "gdn chunked launch sync");
}

void GdnPrefillKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const Tensor& qsl, const GdnArgs& args) {
  const int64_t dv = state.shape[2], dk = state.shape[3];
  // Chunked path (default) for real head dims; the sequential scan handles the
  // arbitrary-dim corners (the CUDA op tests exercise Dv=Dk=300) and the A/B
  // fallback (VT_GDN_CHUNKED=0).
  if (ChunkedPrefillEnabled() && dk <= kChunkMaxDim && dv <= kChunkMaxDim && args.scale != 0.0f) {
    GdnPrefillChunkedCuda(q, out, q_in, k, v, g, beta, state, qsl, args);
    return;
  }
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
