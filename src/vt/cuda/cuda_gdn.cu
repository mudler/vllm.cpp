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
#include <mma.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
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

// cache_idx (optional; mirrors mamba causal_conv1d_update conv_state_indices): when
// non-null, token bt's state row is the persistent-cache slot cache_idx[bt] (idx < 0 ==
// NULL block → skip, leaving out untouched), so the caller need not gather/scatter.
template <typename Tin, typename Tout>
__global__ void CausalConv1dUpdateKernel(Tout* out, const Tin* x, const Tin* w,
                                         const Tin* bias, float* conv_state,
                                         const int32_t* cache_idx, int64_t n, int64_t c_dim,
                                         int64_t k, bool silu) {
  const int64_t width = k - 1;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t bt = idx / c_dim;
    const int64_t c = idx % c_dim;
    int64_t srow_off = idx;  // compact: row == bt (idx = bt*c_dim + c)
    if (cache_idx != nullptr) {
      const int32_t slot = cache_idx[bt];
      if (slot < 0) continue;  // NULL block
      srow_off = static_cast<int64_t>(slot) * c_dim + c;
    }
    float* srow = conv_state + srow_off * width;  // row [K-1]
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
                      const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                      const CausalConv1dArgs& args) {
  const int64_t n = x.shape[0] * x.shape[1], c = x.shape[1], k = w.shape[1];
  CausalConv1dUpdateKernel<Tin, Tout><<<GridFor(n), kBlock, 0, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), bias != nullptr ? bias->Ptr<Tin>() : nullptr,
      conv_state.Ptr<float>(), cache_idx, n, c, k, args.silu_activation);
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
                          const Tensor* conv_state_indices, const CausalConv1dArgs& args) {
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda causal_conv1d_update: unsupported x dtype (f32/bf16 only)");
  VT_CHECK(w.dtype == x.dtype && (bias == nullptr || bias->dtype == x.dtype),
           "cuda causal_conv1d_update: weight/bias dtype must match x");
  if (x.shape[0] * x.shape[1] == 0) return;
  const int32_t* ci =
      conv_state_indices != nullptr ? conv_state_indices->Ptr<int32_t>() : nullptr;
  cudaStream_t s = AsStream(q);
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdate<float, float>(s, out, x, w, bias, conv_state, ci, args);
    } else {
      LaunchConvUpdate<float, __nv_bfloat16>(s, out, x, w, bias, conv_state, ci, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdate<__nv_bfloat16, float>(s, out, x, w, bias, conv_state, ci, args);
    } else {
      LaunchConvUpdate<__nv_bfloat16, __nv_bfloat16>(s, out, x, w, bias, conv_state, ci, args);
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

void GdnScanCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                 const Tensor& g, const Tensor& beta, Tensor& state, const int32_t* qsl_ptr,
                 const GdnArgs& args, const char* name);  // fwd decl (corner-dim fallback)

// ---------------------------------------------------------------------------
// Fused single-step DECODE recurrence — the (batch × Hv × value-tile) parallel
// grid mirror of fla fused_recurrent_gated_delta_rule_fwd_kernel at T==1
// (vllm/model_executor/layers/fla/ops/fused_recurrent.py:27-175; grid
// (NK=1, NV, N*HV) at :217; b_h[BV,BK] register-resident, loaded once :103-120,
// single step :122-149). Each block owns one (sequence, v-head, BV-value-tile).
// vs the sequential GdnScanKernel (which streams the [Dv,Dk] state from GLOBAL
// four times per token, strided by Dk so uncoalesced): here the [BV,Dk] slice is
// staged into shared ONCE via a COALESCED load, updated in place, and written
// back ONCE — two coalesced global passes instead of four strided ones. The grid
// spans n*Hv*NV blocks so decode parallelizes across the whole batch.
//
// state_idx (optional; mirrors fla ssm_state_indices IS_CONTINUOUS_BATCHING
// :104-116): when non-null the block reads/writes persistent-cache row
// state_idx[i_n] (idx<0 == NULL block → zero the output, skip), so the caller
// need not gather/scatter per-request state rows. When null the state is the
// compact [n,Hv,Dv,Dk] buffer and the state row == i_n.
// NW warps cooperate on ONE (seq, v-head, BV-tile)'s [BV,Dk] state by SPLITTING
// the Dk contraction: thread (vi, wk) owns value-row vi and the Dk column slice
// [wk*ceil(Dk/NW), ...). The two contractions over Dk (dot = (S·exp g)@k and
// o = S@q') are each a partial-sum over the thread's slice, then reduced across
// the NW lanes of the row's warp-group with a __shfl_xor butterfly. Shared-mem
// footprint is UNCHANGED (still the one [BV,Dk] slice), so with NW warps per
// block the SM packs NW× the resident warps for the same shared budget — the
// occupancy lever. NW==1 is bit-identical to the original single-warp kernel
// (empty reduction, whole-Dk slice per thread). The NW lanes of a row are the
// NW consecutive lanes [floor(lane/NW)*NW, +NW), aligned inside one warp (NW is
// a power of two dividing 32 and the launcher only uses NW>1 when BV==32, so a
// block is always a whole number of warps) — the xor butterfly stays in-warp.
template <typename Tin, typename Tout, int NW>
__global__ void GdnDecodeFusedKernel(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                                     const float* g, const float* beta, float* state,
                                     const int32_t* state_idx, int64_t hk_n, int64_t dk,
                                     int64_t hv_n, int64_t dv, int64_t bv, float scale) {
  const int64_t i_v = blockIdx.x;         // value-dim tile
  const int64_t i_nh = blockIdx.y;        // fused (sequence, v-head)
  const int64_t i_n = i_nh / hv_n;        // sequence == decode token index
  const int64_t hv = i_nh % hv_n;         // v-head
  const int64_t hk = hv / (hv_n / hk_n);  // GQA-mapped k-head
  const int64_t vbase = i_v * bv;
  const int tid = static_cast<int>(threadIdx.x);
  const int vi = tid / NW;          // value-state row within this BV tile
  const int wk = tid % NW;          // Dk-slice owner within the row's warp-group
  const int64_t vrow = vbase + vi;  // this thread's value-state row

  // Persistent-cache indirection (fla ssm_state_indices). idx<0 == NULL block
  // (our state slots are 0-indexed and slot 0 is valid, so the sentinel is < 0).
  // Uniform over the block (i_n is fixed by blockIdx.y) → no shuffle/sync hazard.
  int64_t row = i_n;
  if (state_idx != nullptr) {
    const int32_t si = state_idx[i_n];
    if (si < 0) {
      if (vrow < dv && wk == 0) Store(out, (i_n * hv_n + hv) * dv + vrow, 0.0f);
      return;
    }
    row = si;
  }

  const int64_t sdk = dk + 1;  // padded shared row stride (kills the 32-way conflict)
  extern __shared__ float smem[];
  float* bq = smem;       // [dk]  q' = q*scale
  float* bk = bq + dk;    // [dk]  k
  float* sbh = bk + dk;   // [bv * sdk]  padded [BV,Dk] state slice
  const int64_t vrows = (dv - vbase) < bv ? (dv - vbase) : bv;  // valid rows in tile
  const int64_t tile = vrows * dk;                              // valid elems

  // q'(=q*scale) and k for this (token, head) — broadcast to every lane.
  const int64_t qkbase = (i_n * hk_n + hk) * dk;
  for (int64_t i = tid; i < dk; i += blockDim.x) {
    bq[i] = Load(q, qkbase + i) * scale;
    bk[i] = Load(k, qkbase + i);
  }
  // Coalesced load of the [BV,Dk] state slice into padded shared.
  float* s_head = state + (row * hv_n + hv) * dv * dk + vbase * dk;  // [<=bv, dk]
  for (int64_t e = tid; e < bv * dk; e += blockDim.x)
    sbh[(e / dk) * sdk + e % dk] = e < tile ? s_head[e] : 0.0f;
  __syncthreads();

  // This thread's Dk column slice [c0, c1) of value-row vi (partition of [0,dk)).
  const int64_t ck = (dk + NW - 1) / NW;
  const int64_t c0 = static_cast<int64_t>(wk) * ck;
  const int64_t c1 = (c0 + ck) < dk ? (c0 + ck) : dk;

  // Every lane stays live through the shuffles (tail rows compute on the zeroed
  // slice); only the global v-load and o-store are guarded by vrow < dv.
  const float decay = expf(g[i_n * hv_n + hv]);
  const float beta_t = beta[i_n * hv_n + hv];
  float* r = sbh + static_cast<int64_t>(vi) * sdk;
  float pdot = 0.0f;  // partial (S * exp(g)) @ k over this slice, fused w/ decay
  for (int64_t c = c0; c < c1; ++c) {
    r[c] *= decay;
    pdot += r[c] * bk[c];
  }
  float dot = pdot;  // reduce the partials across the NW lanes of the row-group
#pragma unroll
  for (int off = 1; off < NW; off <<= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
  const float vv = vrow < dv ? Load(v, (i_n * hv_n + hv) * dv + vrow) : 0.0f;
  const float vp = (vv - dot) * beta_t;
  float po = 0.0f;  // partial (S + outer(v',k)) @ q' over this slice, fused w/ update
  for (int64_t c = c0; c < c1; ++c) {
    r[c] += vp * bk[c];
    po += r[c] * bq[c];
  }
  float o = po;
#pragma unroll
  for (int off = 1; off < NW; off <<= 1) o += __shfl_xor_sync(0xffffffffu, o, off);
  if (vrow < dv && wk == 0) Store(out, (i_n * hv_n + hv) * dv + vrow, o);
  __syncthreads();

  // Coalesced write-back of the updated slice.
  for (int64_t e = tid; e < tile; e += blockDim.x) s_head[e] = sbh[(e / dk) * sdk + e % dk];
}

template <typename Tin, typename Tout, int NW>
void LaunchGdnDecodeFusedNW(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                            const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                            const int32_t* state_idx, int64_t n, const GdnArgs& args) {
  const int64_t hk_n = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv_n = v.shape[1], dv = v.shape[2];
  const int64_t bv = dv < 32 ? dv : 32;   // fla BV cap of 32; any dv via tail guard
  const int64_t nv = (dv + bv - 1) / bv;  // value-dim tiles (fla NV)
  const dim3 grid(static_cast<unsigned>(nv), static_cast<unsigned>(n * hv_n));
  const size_t shmem =
      (2 * static_cast<size_t>(dk) + static_cast<size_t>(bv) * (dk + 1)) * sizeof(float);
  GdnDecodeFusedKernel<Tin, Tout, NW><<<grid, static_cast<unsigned>(bv * NW), shmem, s>>>(
      out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
      beta.Ptr<float>(), state.Ptr<float>(), state_idx, hk_n, dk, hv_n, dv, bv, args.scale);
  Check(cudaGetLastError(), "gdn decode(fused) launch");
}

// nw: warps-per-block for the Dk-split (occupancy lever). >1 only when BV==32
// (dv>=32, the real gate dim) so a block is always a whole number of warps and
// the row-group shuffles stay in-warp; smaller dv (test corners) forces nw=1.
template <typename Tin, typename Tout>
void LaunchGdnDecodeFused(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const int32_t* state_idx, int64_t n, const GdnArgs& args, int nw) {
  const int64_t dv = v.shape[2];
  const int nw_eff = dv >= 32 ? nw : 1;
  switch (nw_eff) {
    case 2:
      LaunchGdnDecodeFusedNW<Tin, Tout, 2>(s, out, q_in, k, v, g, beta, state, state_idx, n, args);
      break;
    case 4:
      LaunchGdnDecodeFusedNW<Tin, Tout, 4>(s, out, q_in, k, v, g, beta, state, state_idx, n, args);
      break;
    case 8:
      LaunchGdnDecodeFusedNW<Tin, Tout, 8>(s, out, q_in, k, v, g, beta, state, state_idx, n, args);
      break;
    default:
      LaunchGdnDecodeFusedNW<Tin, Tout, 1>(s, out, q_in, k, v, g, beta, state, state_idx, n, args);
      break;
  }
}

// Decode dispatch. state_idx == nullptr: compact [n,Hv,Dv,Dk] state (row==i_n).
// Falls back to the sequential scan for the rare corner dims whose [BV,Dk]
// shared slice would exceed the 48 KB default budget (the real gate Dk=128 and
// every decode unit dim are well under).
void GdnDecodeFusedCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                        const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                        const int32_t* state_idx, const GdnArgs& args) {
  VT_CHECK(q_in.dtype == DType::kF32 || q_in.dtype == DType::kBF16,
           "cuda gdn_decode: unsupported q dtype (f32/bf16 only)");
  VT_CHECK(k.dtype == q_in.dtype && v.dtype == q_in.dtype,
           "cuda gdn_decode: q/k/v dtypes must match");
  const int64_t n = q_in.shape[0], hv_n = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  if (n == 0 || hv_n == 0 || dv == 0) return;
  VT_CHECK(n * hv_n <= kMaxGridY, "cuda gdn_decode: too many (seq×head) blocks (grid.y limit)");
  const int64_t bv = dv < 32 ? dv : 32;
  const size_t shmem =
      (2 * static_cast<size_t>(dk) + static_cast<size_t>(bv) * (dk + 1)) * sizeof(float);
  if (shmem > 48 * 1024) {  // corner-dim fallback (no decode test hits this; real dims are 128)
    GdnScanCuda(q, out, q_in, k, v, g, beta, state, nullptr, args, "gdn_decode");
    return;
  }
  // Warps-per-block for the Dk-split occupancy lever (default 8 — measured best
  // on GB10 sm_121: raises GdnDecodeFused theoretical occupancy 10.4%→66.7% and
  // cuts per-call time ~2.2× at conc-64; A/B via env, read once per call —
  // negligible vs kernel). NW=1 == the original single-warp kernel bit-for-bit,
  // so VT_GDN_DECODE_NW=1 is the "before" in the same binary.
  int nw = 8;
  if (const char* e = std::getenv("VT_GDN_DECODE_NW")) {
    const int v_nw = std::atoi(e);
    if (v_nw == 1 || v_nw == 2 || v_nw == 4 || v_nw == 8) nw = v_nw;
  }
  cudaStream_t s = AsStream(q);
  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32)
      LaunchGdnDecodeFused<float, float>(s, out, q_in, k, v, g, beta, state, state_idx, n, args, nw);
    else
      LaunchGdnDecodeFused<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, state_idx, n,
                                                 args, nw);
  } else {
    if (out.dtype == DType::kF32)
      LaunchGdnDecodeFused<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, state_idx, n,
                                                 args, nw);
    else
      LaunchGdnDecodeFused<__nv_bfloat16, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state,
                                                         state_idx, n, args, nw);
  }
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
template <typename Tin, typename TSc>
__global__ void GdnChunkWUKernel(TSc* u, TSc* w, const Tin* k, const Tin* v,
                                 const float* beta, const float* gcum, const int32_t* tok0a,
                                 const int32_t* lena, int64_t hk_n, int64_t dk, int64_t hv_n,
                                 int64_t dv) {
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  __shared__ float kk[kChunk * kChunk];  // K K^T (row i, col j)
  __shared__ float ed[kChunk * kChunk];  // exp(G_i - G_j), j<=i (bounded <=1), else 0
  __shared__ float gs[kChunk];           // G_i (local cumsum)
  __shared__ float bs[kChunk];           // beta_i
  __shared__ float eg[kChunk];           // exp(G_i)
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
    bs[i] = beta[(tok0 + i) * hv_n + hv];
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * len; idx += blockDim.x) {
    const int64_t i = idx / len, j = idx % len;
    float dot = 0.0f;
    const int64_t ib = (tok0 + i) * hk_n * dk + hk * dk;
    const int64_t jb = (tok0 + j) * hk_n * dk + hk * dk;
    for (int64_t d = 0; d < dk; ++d) dot += Load(k, ib + d) * Load(k, jb + d);
    kk[i * kChunk + j] = dot;
    ed[i * kChunk + j] = j <= i ? expf(gs[i] - gs[j]) : 0.0f;  // G non-increasing -> <=1
  }
  __syncthreads();
  // u column vi: u_i = beta_i (v_i - sum_{j<i} exp(G_i-G_j) KK[i,j] u_j)
  for (int64_t vi = threadIdx.x; vi < dv; vi += blockDim.x) {
    float ucol[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float s = 0.0f;
      for (int64_t j = 0; j < i; ++j) s += ed[i * kChunk + j] * kk[i * kChunk + j] * ucol[j];
      const float vi_val = Load(v, (tok0 + i) * hv_n * dv + hv * dv + vi);
      ucol[i] = bs[i] * (vi_val - s);
    }
    for (int64_t i = 0; i < len; ++i)
      Store(u, (tok0 + i) * hv_n * dv + hv * dv + vi, ucol[i]);
  }
  // w column ki: w_i = beta_i (exp(G_i) k_i - sum_{j<i} exp(G_i-G_j) KK[i,j] w_j)
  for (int64_t ki = threadIdx.x; ki < dk; ki += blockDim.x) {
    float wcol[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float s = 0.0f;
      for (int64_t j = 0; j < i; ++j) s += ed[i * kChunk + j] * kk[i * kChunk + j] * wcol[j];
      const float ki_val = Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki);
      wcol[i] = bs[i] * (eg[i] * ki_val - s);
    }
    for (int64_t i = 0; i < len; ++i)
      Store(w, (tok0 + i) * hv_n * dk + hv * dk + ki, wcol[i]);
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
  extern __shared__ float hsh[];      // [Dv, Dk] running state
  __shared__ float decay[kChunk];     // exp(G_last - G_i), i<=last (bounded <=1)
  float* s_head = state + (n * hv_n + hv) * sd;
  for (int64_t e = threadIdx.x; e < sd; e += blockDim.x) hsh[e] = s_head[e];
  __syncthreads();
  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * kChunk;
    const int64_t rem = seqlen - it * kChunk;
    const int64_t len = rem < kChunk ? rem : kChunk;
    float* hstart = hstate + (gc * hv_n + hv) * sd;  // chunk-start snapshot for step D
    for (int64_t e = threadIdx.x; e < sd; e += blockDim.x) hstart[e] = hsh[e];
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);
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
    // hsh[vi,ki] = hsh[vi,ki]*exp(G_last) + sum_i k[i,ki]*v_new[i,vi]*exp(G_last-G_i)
    for (int64_t idx = threadIdx.x; idx < sd; idx += blockDim.x) {
      const int64_t vi = idx / dk, ki = idx % dk;
      float acc = hsh[idx] * eglast;
      for (int64_t i = 0; i < len; ++i) {
        acc += Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki) *
               v_new[(tok0 + i) * hv_n * dv + hv * dv + vi] * decay[i];
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
  const float* hstart = hstate + (gc * hv_n + hv) * sd;
  __shared__ float qk[kChunk * kChunk];  // q_i . k_j
  __shared__ float ed[kChunk * kChunk];  // exp(G_i - G_j), j<=i (bounded <=1), else 0
  __shared__ float gs[kChunk];           // G_i
  __shared__ float eg[kChunk];           // exp(G_i)
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * len; idx += blockDim.x) {
    const int64_t i = idx / len, j = idx % len;
    float dot = 0.0f;
    const int64_t ib = (tok0 + i) * hk_n * dk + hk * dk;
    const int64_t jb = (tok0 + j) * hk_n * dk + hk * dk;
    for (int64_t d = 0; d < dk; ++d) dot += Load(q, ib + d) * Load(k, jb + d);
    qk[i * kChunk + j] = dot;
    ed[i * kChunk + j] = j <= i ? expf(gs[i] - gs[j]) : 0.0f;
  }
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < len * dv; idx += blockDim.x) {
    const int64_t i = idx / dv, vi = idx % dv;
    // cross-chunk: exp(G_i) * (q_i . h_start[vi,:])
    float cross = 0.0f;
    const int64_t qb = (tok0 + i) * hk_n * dk + hk * dk;
    const float* hrow = hstart + vi * dk;
    for (int64_t d = 0; d < dk; ++d) cross += Load(q, qb + d) * hrow[d];
    // intra-chunk causal: sum_{j<=i} exp(G_i-G_j) QK[i,j] v_new_j
    float intra = 0.0f;
    for (int64_t j = 0; j <= i; ++j)
      intra += ed[i * kChunk + j] * qk[i * kChunk + j] * v_new[(tok0 + j) * hv_n * dv + hv * dv + vi];
    const float o = scale * (eg[i] * cross + intra);
    Store(out, (tok0 + i) * hv_n * dv + hv * dv + vi, o);
  }
}

// ===========================================================================
// Tensor-core (WMMA) DeltaH + ChunkO — the top prefill cost (nsys: DeltaH ~25%
// + ChunkO ~18% of 35B prefill). These replace the CUDA-core cooperative
// dot-products with WMMA 16x16 matmuls, f32 accumulate, mirroring FLA's tl.dot
// tiling (chunk_delta_h.py:176 v_new = u - w@hᵀ, :278 h += (k⊙decay)ᵀ@v_new;
// chunk_o.py:111 q@hᵀ, :113 q@kᵀ, :137 A@v_new). Templated on the data/scratch
// dtype TD: bf16 inputs use native bf16 fragments (16x16x16); f32 inputs use
// TF32 fragments (16x16x8, mantissa rounded to 19 bits) — the real 35B GDN runs
// f32, so TF32 is the path that actually hits the model. Running state H[Dv,Dk]
// stays f32 in shared (FLA keeps b_h in f32 registers); scratch dtype = TD.
namespace wmma = nvcuda::wmma;
constexpr int kWM = 16;   // WMMA tile M/N
constexpr int kNB = 32;   // DeltaH V1 output column block (Dv must be a multiple)

// Per-dtype WMMA config: fragment types, K-tile width, and a load that rounds
// f32 -> tf32 in place (bf16 loads natively).
template <typename TD>
struct WmmaCfg;
template <>
struct WmmaCfg<__nv_bfloat16> {
  static constexpr int WK = 16;
  using Acc = wmma::fragment<wmma::accumulator, kWM, kWM, WK, float>;
  using Arow = wmma::fragment<wmma::matrix_a, kWM, kWM, WK, __nv_bfloat16, wmma::row_major>;
  using Acol = wmma::fragment<wmma::matrix_a, kWM, kWM, WK, __nv_bfloat16, wmma::col_major>;
  using Brow = wmma::fragment<wmma::matrix_b, kWM, kWM, WK, __nv_bfloat16, wmma::row_major>;
  using Bcol = wmma::fragment<wmma::matrix_b, kWM, kWM, WK, __nv_bfloat16, wmma::col_major>;
  template <typename F>
  __device__ static void load(F& f, const __nv_bfloat16* p, int ld) {
    wmma::load_matrix_sync(f, p, ld);
  }
};
template <>
struct WmmaCfg<float> {
  static constexpr int WK = 8;
  using Acc = wmma::fragment<wmma::accumulator, kWM, kWM, WK, float>;
  using Arow = wmma::fragment<wmma::matrix_a, kWM, kWM, WK, wmma::precision::tf32, wmma::row_major>;
  using Acol = wmma::fragment<wmma::matrix_a, kWM, kWM, WK, wmma::precision::tf32, wmma::col_major>;
  using Brow = wmma::fragment<wmma::matrix_b, kWM, kWM, WK, wmma::precision::tf32, wmma::row_major>;
  using Bcol = wmma::fragment<wmma::matrix_b, kWM, kWM, WK, wmma::precision::tf32, wmma::col_major>;
  template <typename F>
  __device__ static void load(F& f, const float* p, int ld) {
    wmma::load_matrix_sync(f, p, ld);
    for (int i = 0; i < f.num_elements; ++i) f.x[i] = wmma::__float_to_tf32(f.x[i]);
  }
};

// DeltaH (WMMA). One block per (sequence, v-head), SEQUENTIAL over chunks.
// f32 running state Hf[Dv,Dk] resident in dynamic shared; a pool is reused by
// the two matmuls. Per chunk: snapshot start-state -> TD hstate; V1 v_new=u-W@Hᵀ
// (Hᵀ via a bf16/tf32 slice Hb of Hf); V2 folds the per-row decay into K so the
// rank update H = H*exp(G_last) + V_newᵀ@(K⊙decay) reads V_new straight from
// global (no staging). f32 accumulate.
template <typename TD>
__global__ void GdnChunkDeltaHWmmaKernel(float* state, TD* hstate, TD* v_new, const TD* k,
                                         const TD* u, const TD* w, const float* gcum,
                                         const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                         int64_t dk, int64_t hv_n, int64_t dv) {
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  const int64_t n = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, nwarps = static_cast<int>(blockDim.x) / 32;

  extern __shared__ char smem_raw[];
  float* Hf = reinterpret_cast<float*>(smem_raw);       // [Dv,Dk] f32 state
  char* pool = smem_raw + sd * sizeof(float);           // reused by V1/V2
  __shared__ float decay[kChunk];

  float* s_head = state + (n * hv_n + hv) * sd;
  for (int64_t e = tid; e < sd; e += blockDim.x) Hf[e] = s_head[e];
  __syncthreads();

  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * kChunk;
    const int64_t rem = seqlen - it * kChunk;
    const int64_t len = rem < kChunk ? rem : kChunk;
    TD* hstart = hstate + (gc * hv_n + hv) * sd;
    for (int64_t e = tid; e < sd; e += blockDim.x) Store(hstart, e, Hf[e]);
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += blockDim.x)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);
    __syncthreads();

    // --- V1: v_new = u - W @ Hᵀ  ([len,Dk]@[Dk,Dv] -> [len,Dv]) ---
    TD* Hb = reinterpret_cast<TD*>(pool);                        // [kNB,Dk] TD slice of Hf
    float* sTile = reinterpret_cast<float*>(Hb + kNB * dk);      // [kChunk,kNB] f32
    for (int64_t nb = 0; nb < dv; nb += kNB) {
      for (int64_t e = tid; e < kNB * dk; e += blockDim.x)
        Store(Hb, e, Hf[(nb + e / dk) * dk + e % dk]);
      __syncthreads();
      const int ntiles = (kChunk / kWM) * (kNB / kWM);
      for (int t = warp; t < ntiles; t += nwarps) {
        const int rt = t % (kChunk / kWM), ct = t / (kChunk / kWM);
        const int i0 = rt * kWM, vi0 = static_cast<int>(nb) + ct * kWM;
        typename Cfg::Acc acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int64_t kk = 0; kk < dk; kk += WK) {
          typename Cfg::Arow a;
          typename Cfg::Bcol b;
          Cfg::load(a, w + (tok0 + i0) * hv_n * dk + hv * dk + kk, hv_n * dk);
          Cfg::load(b, Hb + (vi0 - nb) * dk + kk, dk);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(sTile + i0 * kNB + ct * kWM, acc, kNB, wmma::mem_row_major);
      }
      __syncthreads();
      for (int64_t e = tid; e < kChunk * kNB; e += blockDim.x) {
        const int64_t i = e / kNB, cvi = e % kNB;
        if (i < len) {
          const float uv = Load(u, (tok0 + i) * hv_n * dv + hv * dv + nb + cvi);
          Store(v_new, (tok0 + i) * hv_n * dv + hv * dv + nb + cvi, uv - sTile[i * kNB + cvi]);
        }
      }
      __syncthreads();
    }

    // --- V2: H = H*exp(G_last) + V_newᵀ @ (K⊙decay)  ([Dv,len]@[len,Dk]) ---
    TD* Kd = reinterpret_cast<TD*>(pool);                       // [kChunk,Dk] TD, decay-scaled
    for (int64_t e = tid; e < kChunk * dk; e += blockDim.x) {
      const int64_t i = e / dk, ki = e % dk;
      Store(Kd, e, i < len ? Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki) * decay[i] : 0.0f);
    }
    __syncthreads();
    const int ntiles2 = (dv / kWM) * (dk / kWM);
    for (int t = warp; t < ntiles2; t += nwarps) {
      const int vt = t % (dv / kWM), kt = t / (dv / kWM);
      const int vi0 = vt * kWM, ki0 = kt * kWM;
      typename Cfg::Acc acc;
      wmma::load_matrix_sync(acc, Hf + vi0 * dk + ki0, dk, wmma::mem_row_major);
      for (int x = 0; x < acc.num_elements; ++x) acc.x[x] *= eglast;
      for (int ii = 0; ii < kChunk; ii += WK) {
        typename Cfg::Acol a;  // A[vi,i] = V_new[i,vi], col-major straight from global v_new
        typename Cfg::Brow b;  // B[i,ki] = Kd[i,ki]
        Cfg::load(a, v_new + (tok0 + ii) * hv_n * dv + hv * dv + vi0, hv_n * dv);
        Cfg::load(b, Kd + ii * dk + ki0, dk);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(Hf + vi0 * dk + ki0, acc, dk, wmma::mem_row_major);
    }
    __syncthreads();
  }
  for (int64_t e = tid; e < sd; e += blockDim.x) s_head[e] = Hf[e];
}

// ChunkO (WMMA). One block per (chunk, v-head). cross = Q@Hstartᵀ; A = Q@Kᵀ
// (decay-weighted, causal-masked); o = scale*(exp(G)*cross + A@V_new). Buffers
// alias: Ks (used only for qk) is reused as outc after qk.
template <typename TD, typename Tout>
__global__ void GdnChunkOWmmaKernel(Tout* out, const TD* q, const TD* k, const TD* v_new,
                                    const TD* hstate, const float* gcum, const int32_t* tok0a,
                                    const int32_t* lena, int64_t hk_n, int64_t dk, int64_t hv_n,
                                    int64_t dv, float scale) {
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int64_t sd = dv * dk;
  const TD* hstart = hstate + (gc * hv_n + hv) * sd;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, nwarps = static_cast<int>(blockDim.x) / 32;

  extern __shared__ char smem_raw[];
  const size_t r2 = static_cast<size_t>(kChunk * dk) * sizeof(TD) > static_cast<size_t>(kChunk * dv) * 4
                        ? static_cast<size_t>(kChunk * dk) * sizeof(TD)
                        : static_cast<size_t>(kChunk * dv) * 4;
  TD* Qs = reinterpret_cast<TD*>(smem_raw);                                    // [kChunk,Dk]
  char* p2 = smem_raw + static_cast<size_t>(kChunk * dk) * sizeof(TD);         // Ks | outc
  TD* Ks = reinterpret_cast<TD*>(p2);
  float* outc = reinterpret_cast<float*>(p2);
  float* qks = reinterpret_cast<float*>(p2 + r2);                             // [kChunk,kChunk]
  TD* As = reinterpret_cast<TD*>(qks + kChunk * kChunk);                       // [kChunk,kChunk]
  __shared__ float gs[kChunk], eg[kChunk];

  for (int64_t e = tid; e < kChunk * dk; e += blockDim.x) {
    const int64_t i = e / dk, d = e % dk;
    const bool in = i < len;
    Store(Qs, e, in ? Load(q, (tok0 + i) * hk_n * dk + hk * dk + d) : 0.0f);
    Store(Ks, e, in ? Load(k, (tok0 + i) * hk_n * dk + hk * dk + d) : 0.0f);
  }
  for (int64_t i = tid; i < len; i += blockDim.x) {
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();

  // qk = Q @ Kᵀ  [kChunk,kChunk]
  const int nqk = (kChunk / kWM) * (kChunk / kWM);
  for (int t = warp; t < nqk; t += nwarps) {
    const int it = t % (kChunk / kWM), jt = t / (kChunk / kWM);
    const int i0 = it * kWM, j0 = jt * kWM;
    typename Cfg::Acc acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t kk = 0; kk < dk; kk += WK) {
      typename Cfg::Arow a;
      typename Cfg::Bcol b;
      Cfg::load(a, Qs + i0 * dk + kk, dk);
      Cfg::load(b, Ks + j0 * dk + kk, dk);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(qks + i0 * kChunk + j0, acc, kChunk, wmma::mem_row_major);
  }
  __syncthreads();
  // A[i,j] = exp(G_i - G_j) * qk[i,j] for j<=i (causal), else 0.
  for (int64_t e = tid; e < kChunk * kChunk; e += blockDim.x) {
    const int64_t i = e / kChunk, j = e % kChunk;
    float a = 0.0f;
    if (i < len && j < len && j <= i) a = expf(gs[i] - gs[j]) * qks[e];
    Store(As, e, a);
  }
  __syncthreads();

  // cross = Q @ Hstartᵀ  [kChunk,Dv] -> outc (aliases Ks, now free)
  const int ncr = (kChunk / kWM) * (dv / kWM);
  for (int t = warp; t < ncr; t += nwarps) {
    const int it = t % (kChunk / kWM), vt = t / (kChunk / kWM);
    const int i0 = it * kWM, vi0 = vt * kWM;
    typename Cfg::Acc acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t kk = 0; kk < dk; kk += WK) {
      typename Cfg::Arow a;
      typename Cfg::Bcol b;
      Cfg::load(a, Qs + i0 * dk + kk, dk);
      Cfg::load(b, hstart + vi0 * dk + kk, dk);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(outc + i0 * dv + vi0, acc, dv, wmma::mem_row_major);
  }
  __syncthreads();
  for (int64_t e = tid; e < kChunk * dv; e += blockDim.x) {
    const int64_t i = e / dv;
    outc[e] *= i < len ? eg[i] : 0.0f;
  }
  __syncthreads();
  // outc = eg*cross + A @ V_new ; o = scale * outc
  for (int t = warp; t < ncr; t += nwarps) {
    const int it = t % (kChunk / kWM), vt = t / (kChunk / kWM);
    const int i0 = it * kWM, vi0 = vt * kWM;
    typename Cfg::Acc acc;
    wmma::load_matrix_sync(acc, outc + i0 * dv + vi0, dv, wmma::mem_row_major);
    for (int jj = 0; jj < kChunk; jj += WK) {
      typename Cfg::Arow a;
      typename Cfg::Brow b;
      Cfg::load(a, As + i0 * kChunk + jj, kChunk);
      Cfg::load(b, v_new + (tok0 + jj) * hv_n * dv + hv * dv + vi0, hv_n * dv);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(outc + i0 * dv + vi0, acc, dv, wmma::mem_row_major);
  }
  __syncthreads();
  for (int64_t e = tid; e < kChunk * dv; e += blockDim.x) {
    const int64_t i = e / dv, vi = e % dv;
    if (i < len) Store(out, (tok0 + i) * hv_n * dv + hv * dv + vi, scale * outc[e]);
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

  // Scratch dtype tracks the input dtype: f32 scratch consumed by TF32
  // fragments (the real 35B GDN, f32), bf16 scratch by native bf16 fragments
  // (the coupled bf16 path). The tensor cores read the scratch natively.
  using TSc = Tin;
  // WMMA is used for both dtypes when the head dims tile cleanly (the routing
  // in GdnPrefillKernelCuda guarantees this for bf16; f32 corners fall back to
  // the CUDA-core chunked kernels below).
  const bool wmma = (dk % kWM == 0 && dv % kNB == 0);
  // u/w/v_new are over-allocated by kChunk rows so the WMMA kernels can read
  // full BT-row tiles past a partial-tail chunk without going out of bounds
  // (the extra rows contribute nothing: masked/discarded).
  const int64_t t_pad = t_tot + kChunk;

  // Device scratch (stream-ordered; freed after the kernels on the same stream).
  float* gcum = nullptr;
  TSc* u = nullptr;
  TSc* w = nullptr;
  TSc* v_new = nullptr;
  TSc* hstate = nullptr;
  int32_t* d_tok0 = nullptr;
  int32_t* d_len = nullptr;
  int32_t* d_boh = nullptr;
  Check(cudaMallocAsync(&gcum, static_cast<size_t>(t_tot * hv_n) * sizeof(float), s),
        "gdn chunked gcum alloc");
  Check(cudaMallocAsync(&u, static_cast<size_t>(t_pad * hv_n * dv) * sizeof(TSc), s),
        "gdn chunked u alloc");
  Check(cudaMallocAsync(&w, static_cast<size_t>(t_pad * hv_n * dk) * sizeof(TSc), s),
        "gdn chunked w alloc");
  Check(cudaMallocAsync(&v_new, static_cast<size_t>(t_pad * hv_n * dv) * sizeof(TSc), s),
        "gdn chunked v_new alloc");
  Check(cudaMallocAsync(&hstate, static_cast<size_t>(nt_tot * hv_n * dv * dk) * sizeof(TSc), s),
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

  GdnChunkWUKernel<Tin, TSc><<<grid_chunk, 128, 0, s>>>(u, w, k.Ptr<Tin>(), v.Ptr<Tin>(),
                                                        beta.Ptr<float>(), gcum, d_tok0, d_len,
                                                        hk_n, dk, hv_n, dv);
  Check(cudaGetLastError(), "gdn chunked wu launch");

  auto opt_in = [](void* kernel, size_t bytes, const char* what) {
    if (bytes > 48 * 1024)
      Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(bytes)),
            what);
  };

  if (wmma) {
    // WMMA tensor-core path (TF32 for f32 in, native bf16 for bf16 in).
    // Zero v_new so the over-allocated tail rows the matmuls may sweep
    // (their A operand is 0 there, but 0*NaN=NaN on the tensor core) are finite.
    Check(cudaMemsetAsync(v_new, 0, static_cast<size_t>(t_pad * hv_n * dv) * sizeof(TSc), s),
          "gdn chunked v_new zero");
    const size_t sz = sizeof(TSc);
    const size_t v1 = static_cast<size_t>(kNB * dk) * sz + static_cast<size_t>(kChunk * kNB) * 4;
    const size_t v2 = static_cast<size_t>(kChunk * dk) * sz;
    const size_t delta_bytes = static_cast<size_t>(dv * dk) * sizeof(float) + (v1 > v2 ? v1 : v2);
    opt_in(reinterpret_cast<void*>(GdnChunkDeltaHWmmaKernel<TSc>), delta_bytes,
           "gdn chunked delta_h(wmma) shared opt-in");
    GdnChunkDeltaHWmmaKernel<TSc><<<grid_seq, 256, delta_bytes, s>>>(
        state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(), d_boh,
        hk_n, dk, hv_n, dv);
    Check(cudaGetLastError(), "gdn chunked delta_h(wmma) launch");

    const size_t r2 = static_cast<size_t>(kChunk * dk) * sz > static_cast<size_t>(kChunk * dv) * 4
                          ? static_cast<size_t>(kChunk * dk) * sz
                          : static_cast<size_t>(kChunk * dv) * 4;
    const size_t chunko_bytes = static_cast<size_t>(kChunk * dk) * sz + r2 +
                                static_cast<size_t>(kChunk * kChunk) * 4 +
                                static_cast<size_t>(kChunk * kChunk) * sz;
    opt_in(reinterpret_cast<void*>(GdnChunkOWmmaKernel<TSc, Tout>), chunko_bytes,
           "gdn chunked o(wmma) shared opt-in");
    GdnChunkOWmmaKernel<TSc, Tout><<<grid_chunk, 256, chunko_bytes, s>>>(
        out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v_new, hstate, gcum, d_tok0, d_len,
        hk_n, dk, hv_n, dv, args.scale);
    Check(cudaGetLastError(), "gdn chunked o(wmma) launch");
  } else if constexpr (std::is_same<Tin, float>::value) {
    // CUDA-core fallback — only reached for f32 corner dims (bf16 non-WMMA dims
    // are routed to the sequential scan upstream).
    const size_t hsh_bytes = static_cast<size_t>(dv * dk) * sizeof(float);
    opt_in(reinterpret_cast<void*>(GdnChunkDeltaHKernel<Tin>), hsh_bytes,
           "gdn chunked delta_h shared opt-in");
    GdnChunkDeltaHKernel<Tin><<<grid_seq, 256, hsh_bytes, s>>>(
        state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(), d_boh,
        hk_n, dk, hv_n, dv);
    Check(cudaGetLastError(), "gdn chunked delta_h launch");

    GdnChunkOKernel<Tin, Tout><<<grid_chunk, 256, 0, s>>>(out.Ptr<Tout>(), q_in.Ptr<Tin>(),
                                                          k.Ptr<Tin>(), v_new, hstate, gcum,
                                                          d_tok0, d_len, hk_n, dk, hv_n, dv,
                                                          args.scale);
    Check(cudaGetLastError(), "gdn chunked o launch");
  } else {
    VT_CHECK(false, "cuda gdn_prefill(chunked): bf16 requires WMMA-friendly head dims");
  }

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
  const int64_t n_seq = state.shape[0], hv_n = state.shape[1], dv = state.shape[2];
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
  // fallback (VT_GDN_CHUNKED=0). The bf16 chunked path is WMMA (tensor-core),
  // which tiles at 16 and 32 — bf16 dims that are not WMMA-friendly fall back
  // to the sequential scan (real gate dims Dk=Dv=128 satisfy both).
  const bool wmma_ok = q_in.dtype != DType::kBF16 || (dk % kWM == 0 && dv % kNB == 0);
  if (ChunkedPrefillEnabled() && dk <= kChunkMaxDim && dv <= kChunkMaxDim && args.scale != 0.0f &&
      wmma_ok) {
    GdnPrefillChunkedCuda(q, out, q_in, k, v, g, beta, state, qsl, args);
    return;
  }
  GdnScanCuda(q, out, q_in, k, v, g, beta, state, qsl.Ptr<int32_t>(), args, "gdn_prefill");
}

void GdnDecodeKernelCuda(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                         const Tensor* state_idx, const GdnArgs& args) {
  const int32_t* si = state_idx != nullptr ? state_idx->Ptr<int32_t>() : nullptr;
  // Fused (batch × Hv × value-tile) parallel decode — the fla fused_recurrent
  // mirror. state_idx != null => in-place on the FULL persistent cache at slot
  // state_idx[i_n] (mirrors fla ssm_state_indices); null => compact state (row ==
  // i_n). Replaces the batch-serial GdnScanKernel decode branch (M2.x GDN decode
  // throughput). A/B: VT_GDN_FUSED_DECODE=0 falls back to the sequential scan for
  // same-binary before/after measurement — but ONLY for the compact path (the scan
  // has no ssm_state_indices indirection), so the indexed path always uses fused.
  const char* e = std::getenv("VT_GDN_FUSED_DECODE");
  if (si == nullptr && e != nullptr && e[0] == '0') {
    GdnScanCuda(q, out, q_in, k, v, g, beta, state, nullptr, args, "gdn_decode");
    return;
  }
  GdnDecodeFusedCuda(q, out, q_in, k, v, g, beta, state, si, args);
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
