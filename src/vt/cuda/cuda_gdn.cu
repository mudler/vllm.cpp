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
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "vt/cuda/tile/cp_async.cuh"
#include "vt/cuda/tile/tma_pipeline.cuh"
#include "vt/ops.h"

#ifdef VLLM_CPP_TRITON
// SANCTIONED CUDA-only Triton AOT fast-path for the GDN delta_h state recurrence
// (see .agents/discipline.md "SANCTIONED EXCEPTION" + porting-inventory.md §9).
// AOT-generated stable dispatchers (triton.tools.link output; see
// cmake/TritonAOT.cmake, triton_kernels/chunk_delta_h.py). Plain C symbols; the
// launcher loads the EMBEDDED cubin via the CUDA driver API — no Triton/Python at
// runtime. One spec per gate-model GDN shape: h48 (Qwen3.6-27B), h32 (35B).
// gdn_deltah_hNN_default(stream, k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens,
//   chunk_offsets, T, NH) launches the FLA chunk_gated_delta_rule_fwd_kernel.
extern "C" {
#include "gdn_deltah_h48.h"
#include "gdn_deltah_h32.h"
// GDN chunk_o (the output kernel): gdn_chunko_hNN_default(stream, q, k, v_new,
// hstate, gcum, out, cu_seqlens, chunk_indices, scale, T, NT). See
// triton_kernels/chunk_o.py. o(=out) is f32 (the default GDN out dtype).
#include "gdn_chunko_h48.h"
#include "gdn_chunko_h32.h"
// GDN WU pipeline (kkt -> solve_tril -> recompute_w_u), 3 stable dispatchers each.
// See triton_kernels/{chunk_scaled_dot_kkt,solve_tril,wy_fast}.py.
#include "gdn_kkt_h48.h"
#include "gdn_kkt_h32.h"
#include "gdn_tril_h48.h"
#include "gdn_tril_h32.h"
#include "gdn_wu_h48.h"
#include "gdn_wu_h32.h"
}
#endif

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

// Rung-2 TMA support (VT_GDN_TMA). Host-side CUtensorMap builder for the delta_h
// chunk streaming: a 3D tiled descriptor over a [tokens, heads, dk] bf16 tensor
// (dk innermost/contiguous), box (dk, 1, BT) — one [BT, dk] chunk tile per TMA.
// cuTensorMapEncodeTiled is a driver symbol; we fetch it once via the runtime's
// cudaGetDriverEntryPoint so no explicit libcuda link is needed (CUTLASS's own
// runtime-linked path). Mirrors the CuTe-DSL kernel_h _make_bf16_tma_args
// (gdn_chunk_cutedsl/kernel_h.py:56-75): 128 K-dim, num_stages ring, bf16 tile.
using PfnTmaEncode = PFN_cuTensorMapEncodeTiled_v12000;
inline PfnTmaEncode GdnTmaEncodeFn() {
  static PfnTmaEncode fn = [] {
    PfnTmaEncode f = nullptr;
    cudaDriverEntryPointQueryResult q{};
    // Versioned entry-point query (non-deprecated; cuTensorMapEncodeTiled_v12000 => 12000).
    cudaError_t e = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", reinterpret_cast<void**>(&f), 12000, cudaEnableDefault, &q);
    if (e != cudaSuccess || f == nullptr) return static_cast<PfnTmaEncode>(nullptr);
    return f;
  }();
  return fn;
}

// Build a 3D TMA descriptor for a bf16 [t_dim, h_dim, dk] tensor (row-major, dk
// contiguous). box (dk,1,BT), swizzle NONE (the WMMA consumer reads plain
// row-major smem, identical to the cp.async ring — this isolates the copy/
// pipeline mechanism from the smem-layout lever). OOB fill NONE => TMA zeros the
// tail beyond t_dim (partial-chunk tail rows are masked downstream regardless).
inline bool BuildGdnTmaDesc3D(CUtensorMap* desc, const void* base, int64_t t_dim, int64_t h_dim,
                              int64_t dk, int64_t bt) {
  PfnTmaEncode encode = GdnTmaEncodeFn();
  if (!encode) return false;
  const uint64_t gdim[3] = {static_cast<uint64_t>(dk), static_cast<uint64_t>(h_dim),
                            static_cast<uint64_t>(t_dim)};
  const uint64_t gstr[2] = {static_cast<uint64_t>(dk) * 2u,
                            static_cast<uint64_t>(h_dim) * static_cast<uint64_t>(dk) * 2u};
  const uint32_t bdim[3] = {static_cast<uint32_t>(dk), 1u, static_cast<uint32_t>(bt)};
  const uint32_t estr[3] = {1u, 1u, 1u};
  CUresult r = encode(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                      const_cast<void*>(base), gdim, gstr, bdim, estr,
                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
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

// ---------------------------------------------------------------------------
// Tiled causal_conv1d_fwd (VT_CONV_TILED, default OFF) — the (token-tile x
// channel-tile) mirror of the FLA/vLLM causal_conv1d Triton kernel. The scalar
// CausalConv1dFwdKernel above rereads each x value K times (once per tap it lands
// on) and is serial per (seq, channel); this stages a [BLOCK_M+width, BLOCK_N]
// x-tile into shared ONCE (each x read ~once vs K times, the halo aside) and lets
// BLOCK_M token rows reuse the same sliding window, with consecutive threads over
// the channel dim so the global x loads and out stores are coalesced.
//
// NUMERICALLY IDENTICAL to the scalar kernel (token-exact): for every output
// (token t, channel c) it accumulates acc = bias; for j in [0,k): acc += w[c*k+j]
// * v_j — the SAME tap ORDER (j=0 oldest .. j=k-1 newest, bias first) over the
// SAME f32 values. v_j is fetched from the shared tile, but the tile is filled by
// the identical Load(x,...) / conv_state reads the scalar kernel does, so each v_j
// is bit-for-bit the same float; a conv is a fixed sum of k<=~4 f32 products, and
// preserving its accumulation order keeps a razor-thin greedy argmax from flipping.
// One block owns a (channel-tile, sequence) and loops token-tiles of BLOCK_M, then
// performs the (K-1) state write-back byte-identically to the scalar kernel
// (ascending j, same read-before-overwrite ordering). Causal masking, the (K-1)
// left-pad / conv_state prepend (has_initial_state), per-channel weight + optional
// bias, silu/identity, and the [token, c_dim] output layout are all preserved.
// PREFILL only — the decode CausalConv1dUpdate path is untouched.
constexpr int kConvTileM = 16;  // token tile (rows per block)
constexpr int kConvTileN = 32;  // channel tile (a warp — coalesced x loads/stores)

template <typename Tin, typename Tout>
__global__ void CausalConv1dFwdTiledKernel(Tout* out, const Tin* x, const Tin* w,
                                           const Tin* bias, float* conv_state,
                                           const int32_t* qsl, const int32_t* his, int64_t c_dim,
                                           int64_t k, bool silu) {
  const int64_t width = k - 1;
  const int64_t s = blockIdx.y;  // sequence
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int64_t c = static_cast<int64_t>(blockIdx.x) * kConvTileN + tx;  // channel
  const bool active = c < c_dim;
  const int64_t begin = qsl[s];
  const int64_t t_len = qsl[s + 1] - begin;
  const bool init = his[s] != 0;
  // Own conv_state row [K-1] for this (seq, channel); prefill state is f32.
  float* srow = active ? conv_state + (s * c_dim + c) * width : nullptr;
  const float b = (active && bias != nullptr) ? Load(bias, c) : 0.0f;

  extern __shared__ float sh[];  // [(kConvTileM + width) * kConvTileN], slot-major

  // t_len is uniform across the block (s == blockIdx.y), so every thread runs the
  // same iteration count and hits every __syncthreads() below in lockstep.
  for (int64_t t0 = 0; t0 < t_len; t0 += kConvTileM) {
    // Cooperative, coalesced load of the tile: (K-1) history halo + BLOCK_M window.
    for (int64_t slot = ty; slot < kConvTileM + width; slot += blockDim.y) {
      const int64_t tok = t0 - width + slot;  // token index within the sequence
      float val = 0.0f;
      if (active) {
        if (tok >= 0 && tok < t_len) {
          val = Load(x, (begin + tok) * c_dim + c);
        } else if (tok < 0 && init) {
          val = srow[width + tok];  // old conv_state col (K-1)+tok, in [0,width)
        }
      }
      sh[slot * kConvTileN + tx] = val;
    }
    __syncthreads();
    // One output token per row: token t reads window slots [ty, ty+k), i.e. the
    // taps ti = t-(k-1)..t, matching the scalar kernel's ti = t-(k-1-j).
    const int64_t t = t0 + ty;
    if (active && t < t_len) {
      float acc = b;
      for (int64_t j = 0; j < k; ++j) {  // SAME order as the scalar kernel
        acc += Load(w, c * k + j) * sh[(ty + j) * kConvTileN + tx];
      }
      Store(out, (begin + t) * c_dim + c, silu ? Silu(acc) : acc);
    }
    __syncthreads();  // all reads of sh done before the next tile overwrites it
  }

  // State write-back (byte-identical to the scalar kernel): last K-1 RAW x tokens,
  // left-padded with zeros (no init state) or shifted old state when T < K-1. One
  // row (ty == 0) writes each channel's state; ascending j reads old col t_len+j
  // (>= j, unwritten), so no already-written slot is reread.
  if (active && ty == 0) {
    for (int64_t j = 0; j < width; ++j) {
      const int64_t tj = t_len - width + j;
      float val = 0.0f;
      if (tj >= 0) {
        val = Load(x, (begin + tj) * c_dim + c);
      } else if (init) {
        val = srow[width + tj];
      }
      srow[j] = val;
    }
  }
}

// Toggle: default OFF (byte-identical scalar dispatch). Cached static reader
// (GdnSlackMemsetEnabled style) — VT_CONV_TILED=1 selects the tiled kernel.
bool ConvTiledEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_CONV_TILED");
    return e != nullptr && e[0] == '1';
  }();
  return on;
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

// Tiled launcher (VT_CONV_TILED=1). Block = (kConvTileN channels, kConvTileM
// tokens); grid = (channel-tiles, sequences) — same sequence-per-grid.y layout as
// the scalar launcher, so the n <= kMaxGridY guard already covers grid.y. Dynamic
// shared holds the [(BLOCK_M+width) x BLOCK_N] x-tile (width = k-1).
template <typename Tin, typename Tout>
void LaunchConvFwdTiled(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                        const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                        const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t n = conv_state.shape[0], c = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  const dim3 grid(static_cast<unsigned>((c + kConvTileN - 1) / kConvTileN),
                  static_cast<unsigned>(n));
  const dim3 block(kConvTileN, kConvTileM);
  const size_t shmem = static_cast<size_t>(kConvTileM + width) * kConvTileN * sizeof(float);
  CausalConv1dFwdTiledKernel<Tin, Tout><<<grid, block, shmem, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), bias != nullptr ? bias->Ptr<Tin>() : nullptr,
      conv_state.Ptr<float>(), qsl.Ptr<int32_t>(), his.Ptr<int32_t>(), c, k,
      args.silu_activation);
  Check(cudaGetLastError(), "causal_conv1d_fwd(tiled) launch");
}

// Dispatch on the toggle. tiled == false forwards to the EXACT scalar launcher
// (byte-identical grid/block/kernel), so VT_CONV_TILED unset/0 is a no-op.
template <typename Tin, typename Tout>
void DispatchConvFwd(bool tiled, cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                     const Tensor* bias, Tensor& conv_state, const Tensor& qsl, const Tensor& his,
                     const CausalConv1dArgs& args) {
  if (tiled) {
    LaunchConvFwdTiled<Tin, Tout>(s, out, x, w, bias, conv_state, qsl, his, args);
  } else {
    LaunchConvFwd<Tin, Tout>(s, out, x, w, bias, conv_state, qsl, his, args);
  }
}

// ---------------------------------------------------------------------------
// causal_conv1d_update (gdn-semantics.md §3): grid-stride, one thread per
// (token, channel). Read-old-then-roll on the thread's own state row.
// Upstream counterpart: layers/mamba/ops/causal_conv1d.py
// (causal_conv1d_update Triton kernel, seqlen==1 path) — align post-MVP.

// cache_idx (optional; mirrors mamba causal_conv1d_update conv_state_indices): when
// non-null, token bt's state row is the persistent-cache slot cache_idx[bt] (idx < 0 ==
// NULL block → skip, leaving out untouched), so the caller need not gather/scatter.
template <typename Tin, typename Tout, typename TState>
__global__ void CausalConv1dUpdateKernel(Tout* out, const Tin* x, const Tin* w,
                                         const Tin* bias, TState* conv_state,
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
    // Persistent conv_state cache is bf16 (vLLM default) or f32 (unit test);
    // Load()/Store() upcast/downcast, the convolution accumulates in f32.
    TState* srow = conv_state + srow_off * width;  // row [K-1]
    const float xt = Load(x, idx);
    float acc = bias != nullptr ? Load(bias, c) : 0.0f;
    for (int64_t j = 0; j < width; ++j) acc += Load(w, c * k + j) * Load(srow, j);
    acc += Load(w, c * k + width) * xt;
    Store(out, idx, silu ? Silu(acc) : acc);
    for (int64_t j = 0; j + 1 < width; ++j) Store(srow, j, Load(srow, j + 1));  // roll left
    if (width > 0) Store(srow, width - 1, xt);                                  // raw x
  }
}

template <typename Tin, typename Tout, typename TState>
void LaunchConvUpdate(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                      const CausalConv1dArgs& args) {
  const int64_t n = x.shape[0] * x.shape[1], c = x.shape[1], k = w.shape[1];
  CausalConv1dUpdateKernel<Tin, Tout, TState><<<GridFor(n), kBlock, 0, s>>>(
      out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), bias != nullptr ? bias->Ptr<Tin>() : nullptr,
      conv_state.Ptr<TState>(), cache_idx, n, c, k, args.silu_activation);
  Check(cudaGetLastError(), "causal_conv1d_update launch");
}

// Persistent conv_state dtype (bf16 cache = vLLM default; f32 = unit test).
template <typename Tin, typename Tout>
void LaunchConvUpdateS(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                       const Tensor* bias, Tensor& conv_state, const int32_t* cache_idx,
                       const CausalConv1dArgs& args) {
  if (conv_state.dtype == DType::kBF16)
    LaunchConvUpdate<Tin, Tout, __nv_bfloat16>(s, out, x, w, bias, conv_state, cache_idx, args);
  else
    LaunchConvUpdate<Tin, Tout, float>(s, out, x, w, bias, conv_state, cache_idx, args);
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
  const bool tiled = ConvTiledEnabled();  // VT_CONV_TILED (default OFF)
  if (x.dtype == DType::kF32) {
    if (out.dtype == DType::kF32) {
      DispatchConvFwd<float, float>(tiled, s, out, x, w, bias, conv_state, qsl, his, args);
    } else {
      DispatchConvFwd<float, __nv_bfloat16>(tiled, s, out, x, w, bias, conv_state, qsl, his, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      DispatchConvFwd<__nv_bfloat16, float>(tiled, s, out, x, w, bias, conv_state, qsl, his, args);
    } else {
      DispatchConvFwd<__nv_bfloat16, __nv_bfloat16>(tiled, s, out, x, w, bias, conv_state, qsl,
                                                    his, args);
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
      LaunchConvUpdateS<float, float>(s, out, x, w, bias, conv_state, ci, args);
    } else {
      LaunchConvUpdateS<float, __nv_bfloat16>(s, out, x, w, bias, conv_state, ci, args);
    }
  } else {
    if (out.dtype == DType::kF32) {
      LaunchConvUpdateS<__nv_bfloat16, float>(s, out, x, w, bias, conv_state, ci, args);
    } else {
      LaunchConvUpdateS<__nv_bfloat16, __nv_bfloat16>(s, out, x, w, bias, conv_state, ci, args);
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
// gdn_post_conv: fused split + q/k l2norm + g/beta gating (mirror of fla
// fused_gdn_prefill_post_conv _fused_post_conv_kernel, grid (L, H+HV)). One
// launch replaces GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta. Bit-for-bit
// equal: the q/k blocks (i_head < Hk) reuse the EXACT L2NormRowKernel shared-mem
// tree reduction; the trailing (i_head == Hk) block does the v copy + the §6
// g/beta gating (softplus threshold 20), identical to GdnConvSplit/GdnGBeta.
// Templated on the q/k/v output dtype (Tqkv): f32 by default, or bf16 for the
// coupled GDN bf16 path (VT_GDN_BF16) — mirrors FLA keeping the matmul-input
// activations (q/k/v) in bf16 while g/beta stay f32 (they gate the recurrence,
// FLA holds them f32). The l2norm reduction math is f32; only the store rounds
// to Tqkv. g_out/beta_out are always f32. Tconv = the conv-output activation
// dtype: f32 (default) or bf16 under the input-side bf16 GDN path (VT_GDN_IN_BF16),
// which mirrors FLA carrying the post-conv activations in bf16 (halved conv-read
// traffic); the reads upcast to f32 via Load() so the norm math is unchanged.
template <typename Tqkv, typename Tconv>
__global__ void GdnPostConvKernel(Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, float* g_out,
                                  float* beta_out, const Tconv* conv, const float* araw,
                                  const float* braw, const float* a_log, const float* dt_bias,
                                  int64_t hk, int64_t dk, int64_t hv, int64_t dv, float eps) {
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t tok = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t row = tok * conv_dim;
  if (head < hk) {
    __shared__ float partial[kBlock];
    const Tconv* qin = conv + row + head * dk;
    const Tconv* kin = conv + row + key_dim + head * dk;
    Tqkv* qo = q_out + (tok * hk + head) * dk;
    Tqkv* ko = k_out + (tok * hk + head) * dk;
    // ---- q l2norm (plain SUM of squares over Dk, §4) ----
    float acc = 0.0f;
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) {
      const float v = Load(qin, j);
      acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float qinv = 1.0f / sqrtf(partial[0] + eps);
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) Store(qo, j, Load(qin, j) * qinv);
    __syncthreads();
    // ---- k l2norm ----
    acc = 0.0f;
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) {
      const float v = Load(kin, j);
      acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s /= 2) {
      if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
      __syncthreads();
    }
    const float kinv = 1.0f / sqrtf(partial[0] + eps);
    for (int64_t j = threadIdx.x; j < dk; j += kBlock) Store(ko, j, Load(kin, j) * kinv);
  } else {
    // v copy + §6 g/beta for this token.
    const Tconv* vin = conv + row + 2 * key_dim;
    Tqkv* vo = v_out + tok * value_dim;
    for (int64_t j = threadIdx.x; j < value_dim; j += kBlock) Store(vo, j, Load(vin, j));
    for (int64_t h = threadIdx.x; h < hv; h += kBlock) {
      const int64_t idx = tok * hv + h;
      const float x = araw[idx] + dt_bias[h];
      const float sp = x > 20.0f ? x : log1pf(expf(x));  // softplus, threshold 20
      g_out[idx] = -expf(a_log[h]) * sp;
      beta_out[idx] = 1.0f / (1.0f + expf(-braw[idx]));
    }
  }
}

void GdnPostConvKernelCuda(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                           Tensor& beta_out, const Tensor& conv, const Tensor& araw,
                           const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias,
                           const L2NormArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  if (t == 0) return;
  VT_CHECK(q_out.dtype == k_out.dtype && q_out.dtype == v_out.dtype,
           "cuda gdn_post_conv: q/k/v out dtypes must match");
  VT_CHECK(q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16,
           "cuda gdn_post_conv: q/k/v out must be f32 or bf16");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "cuda gdn_post_conv: conv must be f32 or bf16");
  dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>(hk + 1));
  cudaStream_t s = AsStream(q);
  // Dispatch over (q/k/v out dtype) x (conv-in dtype). conv is bf16 under the
  // input-side bf16 GDN path (VT_GDN_IN_BF16); the conv read upcasts to f32.
  auto launch = [&](auto qkv_tag, auto conv_tag) {
    using Tqkv = decltype(qkv_tag);
    using Tconv = decltype(conv_tag);
    GdnPostConvKernel<Tqkv, Tconv><<<grid, kBlock, 0, s>>>(
        q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), g_out.Ptr<float>(),
        beta_out.Ptr<float>(), conv.Ptr<Tconv>(), araw.Ptr<float>(), braw.Ptr<float>(),
        a_log.Ptr<float>(), dt_bias.Ptr<float>(), hk, dk, hv, dv, args.eps);
  };
  const bool qkv_f32 = q_out.dtype == DType::kF32;
  const bool conv_f32 = conv.dtype == DType::kF32;
  if (qkv_f32 && conv_f32) {
    launch(float{}, float{});
  } else if (qkv_f32) {
    launch(float{}, __nv_bfloat16{});
  } else if (conv_f32) {
    launch(__nv_bfloat16{}, float{});
  } else {
    launch(__nv_bfloat16{}, __nv_bfloat16{});
  }
  Check(cudaGetLastError(), "gdn_post_conv launch");
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
template <typename Tin, typename Tout, typename TState, int NW>
__global__ void GdnDecodeFusedKernel(Tout* out, const Tin* q, const Tin* k, const Tin* v,
                                     const float* g, const float* beta, TState* state,
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
  // Coalesced load of the [BV,Dk] state slice into padded shared. The persistent
  // cache TState is bf16 (mirrors vLLM's default mamba_cache_dtype=auto→model
  // dtype; fla fused_recurrent reads bf16→f32 registers→writes bf16) or f32
  // (unit test). Load() upcasts to f32; the recurrence below runs in f32.
  TState* s_head = state + (row * hv_n + hv) * dv * dk + vbase * dk;  // [<=bv, dk]
  for (int64_t e = tid; e < bv * dk; e += blockDim.x)
    sbh[(e / dk) * sdk + e % dk] = e < tile ? Load(s_head, e) : 0.0f;
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

  // Coalesced write-back of the updated slice (f32 register state → bf16 cache).
  for (int64_t e = tid; e < tile; e += blockDim.x)
    Store(s_head, e, sbh[(e / dk) * sdk + e % dk]);
}

template <typename Tin, typename Tout, typename TState, int NW>
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
  GdnDecodeFusedKernel<Tin, Tout, TState, NW><<<grid, static_cast<unsigned>(bv * NW), shmem, s>>>(
      out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v.Ptr<Tin>(), g.Ptr<float>(),
      beta.Ptr<float>(), state.Ptr<TState>(), state_idx, hk_n, dk, hv_n, dv, bv, args.scale);
  Check(cudaGetLastError(), "gdn decode(fused) launch");
}

// nw: warps-per-block for the Dk-split (occupancy lever). >1 only when BV==32
// (dv>=32, the real gate dim) so a block is always a whole number of warps and
// the row-group shuffles stay in-warp; smaller dv (test corners) forces nw=1.
template <typename Tin, typename Tout, typename TState>
void LaunchGdnDecodeFused(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const int32_t* state_idx, int64_t n, const GdnArgs& args, int nw) {
  const int64_t dv = v.shape[2];
  const int nw_eff = dv >= 32 ? nw : 1;
  switch (nw_eff) {
    case 2:
      LaunchGdnDecodeFusedNW<Tin, Tout, TState, 2>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args);
      break;
    case 4:
      LaunchGdnDecodeFusedNW<Tin, Tout, TState, 4>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args);
      break;
    case 8:
      LaunchGdnDecodeFusedNW<Tin, Tout, TState, 8>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args);
      break;
    default:
      LaunchGdnDecodeFusedNW<Tin, Tout, TState, 1>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args);
      break;
  }
}

// Pick the persistent-state dtype (bf16 cache = vLLM default; f32 = unit test)
// then forward to the (Tin,Tout)-typed launcher.
template <typename Tin, typename Tout>
void LaunchGdnDecodeFusedS(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                           const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                           const int32_t* state_idx, int64_t n, const GdnArgs& args, int nw) {
  if (state.dtype == DType::kBF16)
    LaunchGdnDecodeFused<Tin, Tout, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, state_idx,
                                                   n, args, nw);
  else
    LaunchGdnDecodeFused<Tin, Tout, float>(s, out, q_in, k, v, g, beta, state, state_idx, n, args,
                                           nw);
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
    VT_CHECK(state.dtype == DType::kF32,
             "cuda gdn_decode: bf16 state cache unsupported on the corner-dim scan fallback");
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
      LaunchGdnDecodeFusedS<float, float>(s, out, q_in, k, v, g, beta, state, state_idx, n, args,
                                          nw);
    else
      LaunchGdnDecodeFusedS<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, state_idx, n,
                                                  args, nw);
  } else {
    if (out.dtype == DType::kF32)
      LaunchGdnDecodeFusedS<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, state_idx, n,
                                                  args, nw);
    else
      LaunchGdnDecodeFusedS<__nv_bfloat16, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state,
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

// 128-bit (int4/float4) staging: one coalesced 16-byte transfer moves N=16/
// sizeof(TD) activations (8 bf16 / 4 f32), addressed ONCE per group instead of
// per element. Uf/Sf unpack/pack to f32 lanes with the SAME __bfloat162float /
// __float2bfloat16(RNE) the scalar Load/Store use, so every staged value is
// bit-identical; Cpy is a raw 16-byte copy (bf16↔f32 round-trip is the identity
// for a value already in TD, so a direct copy reproduces Load→Store exactly).
// This is the DeltaH/ChunkO analogue of the prefill-attention K/V-staging win.
template <typename TD>
struct V128;
template <>
struct V128<__nv_bfloat16> {
  static constexpr int N = 8;
  __device__ static void Uf(const __nv_bfloat16* p, float* f) {
    const int4 v = *reinterpret_cast<const int4*>(p);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = __bfloat162float(b[i]);
  }
  __device__ static void Sf(__nv_bfloat16* p, const float* f) {
    int4 v;
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i] = __float2bfloat16(f[i]);
    *reinterpret_cast<int4*>(p) = v;
  }
  __device__ static void Sf0(__nv_bfloat16* p) { *reinterpret_cast<int4*>(p) = int4{0, 0, 0, 0}; }
  __device__ static void Cpy(__nv_bfloat16* d, const __nv_bfloat16* s) {
    *reinterpret_cast<int4*>(d) = *reinterpret_cast<const int4*>(s);
  }
};
template <>
struct V128<float> {
  static constexpr int N = 4;
  __device__ static void Uf(const float* p, float* f) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    f[0] = v.x;
    f[1] = v.y;
    f[2] = v.z;
    f[3] = v.w;
  }
  __device__ static void Sf(float* p, const float* f) {
    *reinterpret_cast<float4*>(p) = float4{f[0], f[1], f[2], f[3]};
  }
  __device__ static void Sf0(float* p) { *reinterpret_cast<float4*>(p) = float4{0, 0, 0, 0}; }
  __device__ static void Cpy(float* d, const float* s) {
    *reinterpret_cast<float4*>(d) = *reinterpret_cast<const float4*>(s);
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
                                         int64_t dk, int64_t hv_n, int64_t dv, bool vec) {
  using Cfg = WmmaCfg<TD>;
  using V = V128<TD>;
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
  if (vec)
    for (int64_t g = tid; g < sd / 4; g += blockDim.x)
      V128<float>::Cpy(Hf + g * 4, s_head + g * 4);
  else
    for (int64_t e = tid; e < sd; e += blockDim.x) Hf[e] = s_head[e];
  __syncthreads();

  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * kChunk;
    const int64_t rem = seqlen - it * kChunk;
    const int64_t len = rem < kChunk ? rem : kChunk;
    TD* hstart = hstate + (gc * hv_n + hv) * sd;
    if (vec)
      for (int64_t g = tid; g < sd / V::N; g += blockDim.x) {
        float f[V::N];
#pragma unroll
        for (int j = 0; j < V::N; ++j) f[j] = Hf[g * V::N + j];
        V::Sf(hstart + g * V::N, f);
      }
    else
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
      if (vec)
        for (int64_t g = tid; g < (kNB * dk) / V::N; g += blockDim.x) {
          float f[V::N];
#pragma unroll
          for (int j = 0; j < V::N; ++j) f[j] = Hf[nb * dk + g * V::N + j];
          V::Sf(Hb + g * V::N, f);
        }
      else
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
      if (vec)
        for (int64_t g = tid; g < (kChunk * kNB) / V::N; g += blockDim.x) {
          const int64_t base = g * V::N;
          const int64_t i = base / kNB, cvi = base % kNB;
          if (i < len) {
            float uf[V::N];
            V::Uf(u + (tok0 + i) * hv_n * dv + hv * dv + nb + cvi, uf);
#pragma unroll
            for (int j = 0; j < V::N; ++j) uf[j] -= sTile[i * kNB + cvi + j];
            V::Sf(v_new + (tok0 + i) * hv_n * dv + hv * dv + nb + cvi, uf);
          }
        }
      else
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
    if (vec)
      for (int64_t g = tid; g < (kChunk * dk) / V::N; g += blockDim.x) {
        const int64_t base = g * V::N;
        const int64_t i = base / dk, ki = base % dk;
        if (i < len) {
          float kf[V::N];
          V::Uf(k + (tok0 + i) * hk_n * dk + hk * dk + ki, kf);
          const float d = decay[i];
#pragma unroll
          for (int j = 0; j < V::N; ++j) kf[j] *= d;
          V::Sf(Kd + base, kf);
        } else {
          V::Sf0(Kd + base);
        }
      }
    else
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
  if (vec)
    for (int64_t g = tid; g < sd / 4; g += blockDim.x)
      V128<float>::Cpy(s_head + g * 4, Hf + g * 4);
  else
    for (int64_t e = tid; e < sd; e += blockDim.x) s_head[e] = Hf[e];
}

// ----------------------------------------------------------------------------
// DeltaH (REGISTER-tiled, vt::tile Rung-1). FAITHFUL 1:1 port of FLA's
// chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (H held in ACCUMULATOR
// REGISTERS across the chunk loop, NOT in 64 KiB of shared memory) — the true
// CUTLASS-style persistent-accumulator threadblock GEMM. Frees smem for the
// N-stage cp.async input ring (STEP 2 / VT_GDN_TILE_PIPE_CPASYNC).
//
// Ported FROM:
//   vllm/model_executor/layers/fla/ops/chunk_delta_h.py:43-315
//     chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (pin e24d1b24). Grid
//     (cdiv(V,BV), N*H); b_h1/b_h2 = [BV,64] f32 register accumulators (K-split
//     at 64: K=128 -> two halves); per-seq SEQUENTIAL over chunks.
//   include/cute/arch/copy_sm80.hpp:40-193 (the cp.async ring — STEP 2 only).
// Specialized to the gate shape: K=V=128, BT=64, USE_G=1, USE_GK=0,
// USE_INITIAL_STATE=1, STORE_FINAL_STATE=1, SAVE_NEW_VALUE=1, IS_VARLEN=1,
// USE_EXP2=0. Requires dk==128 (two K-halves), dv%BV==0, blockDim.x==(BV/16)*32.
//
// Warp ownership (persistent accumulators): warp w owns v-rows
// [w*16, w*16+16) of this block's v-slice, ALL K columns -> Cfg::Acc
// h1[NK],h2[NK] (NK=64/16=4) per warp = 8 f32 accumulator fragments, live across
// the whole chunk loop. nwarps == BV/16 (each warp = 16 v-rows).
//
// Per chunk: (a) snapshot start-state -> hstate(bf16); (b) V1 b_v = u - W@b_hᵀ
// (b_hᵀ built by storing the accumulators to a transient smem tile and loading
// it col-major — the ONLY smem the persistent state touches); store v_new; (c)
// decay folded into K + b_h *= exp(g_last) (algebraically identical to FLA's
// decay-on-b_v: per-row scalar, same as GdnChunkDeltaHWmmaKernel, ref-verified);
// (e) V2 b_h += b_vᵀ @ (K⊙decay) accumulated DIRECTLY into the persistent
// fragments. Epilogue: b_h -> state(f32). w/u read straight from global; k
// staged into the decay-scaled Kd tile; v_new kept on-chip (Vbf) for V2.
template <typename TD, int BV>
__global__ void GdnChunkDeltaHRegKernel(float* state, TD* hstate, TD* v_new, const TD* k,
                                        const TD* u, const TD* w, const float* gcum,
                                        const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                        int64_t dk, int64_t hv_n, int64_t dv) {
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  constexpr int BT = kChunk;         // 64
  constexpr int NK = 64 / kWM;       // 4 k-subtiles per K-half
  const int64_t i_v = blockIdx.x, i_nh = blockIdx.y;
  const int64_t n = i_nh / hv_n, hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;

  // Transient smem arena. Region A (Hf32 | Braw | Kd) and Region B (Hbf | Vbf)
  // alias by phase (syncthreads-separated): the persistent state lives in
  // registers, so NONE of this is resident across chunks — that is what frees
  // smem for the STEP-2 ring.
  const int64_t rA0 = BV * dk * (int64_t)sizeof(float);            // Hf32
  const int64_t rA1 = BT * BV * (int64_t)sizeof(float);            // Braw
  const int64_t rA2 = BT * dk * (int64_t)sizeof(TD);               // Kd
  int64_t regA = rA0 > rA1 ? rA0 : rA1;
  regA = regA > rA2 ? regA : rA2;
  extern __shared__ char smem_raw[];
  float* Hf32 = reinterpret_cast<float*>(smem_raw);          // [BV,dk] f32 (region A)
  float* Braw = reinterpret_cast<float*>(smem_raw);          // [BT,BV] f32 (aliases A)
  TD* Kd = reinterpret_cast<TD*>(smem_raw);                  // [BT,dk] TD  (aliases A)
  TD* Hbf = reinterpret_cast<TD*>(smem_raw + regA);          // [BV,dk] TD  (region B)
  TD* Vbf = reinterpret_cast<TD*>(smem_raw + regA);          // [BT,BV] TD  (aliases B)
  __shared__ float decay[BT];

  float* sH = state + (n * hv_n + hv) * sd;
  const int64_t vrow0 = i_v * BV + warp * kWM;  // this warp's first (global) v-row

  // init: b_h += h0  (USE_INITIAL_STATE=1; state carries h0)
  typename Cfg::Acc h1[NK], h2[NK];
  for (int kt = 0; kt < NK; ++kt) {
    wmma::load_matrix_sync(h1[kt], sH + vrow0 * dk + kt * kWM, dk, wmma::mem_row_major);
    wmma::load_matrix_sync(h2[kt], sH + vrow0 * dk + 64 + kt * kWM, dk, wmma::mem_row_major);
  }

  const int64_t nt = (seqlen + BT - 1) / BT;
  for (int64_t it = 0; it < nt; ++it) {
    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * BT;
    const int64_t rem = seqlen - it * BT;
    const int64_t len = rem < BT ? rem : BT;
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += blockDim.x)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);

    // (a) snapshot: store the persistent accumulators to Hf32, then convert to
    // hstate(bf16) AND to the bf16 V1 operand Hbf (one fragment-store serves
    // both). f32 path: Hbf is a plain copy (TD==float).
    for (int kt = 0; kt < NK; ++kt) {
      wmma::store_matrix_sync(Hf32 + (warp * kWM) * dk + kt * kWM, h1[kt], dk, wmma::mem_row_major);
      wmma::store_matrix_sync(Hf32 + (warp * kWM) * dk + 64 + kt * kWM, h2[kt], dk,
                              wmma::mem_row_major);
    }
    __syncthreads();
    // 16-byte-vectorized snapshot: one Hf32 read serves both hstate(bf16) and the
    // V1 operand Hbf. VN = 8 bf16 / 4 f32 per coalesced transfer (V128).
    constexpr int VN = V128<TD>::N;
    TD* hsnap = hstate + (gc * hv_n + hv) * sd;
    for (int64_t e = tid * VN; e < BV * dk; e += (int64_t)blockDim.x * VN) {
      float f[VN];
#pragma unroll
      for (int j = 0; j < VN; j += 4) {
        const float4 v = *reinterpret_cast<const float4*>(Hf32 + e + j);
        f[j] = v.x;
        f[j + 1] = v.y;
        f[j + 2] = v.z;
        f[j + 3] = v.w;
      }
      const int64_t r = e / dk, c = e % dk;
      V128<TD>::Sf(hsnap + (i_v * BV + r) * dk + c, f);
      V128<TD>::Sf(Hbf + e, f);
    }
    __syncthreads();

    // (b) V1: b_v[BT,BV] = W[BT,K] @ b_hᵀ[K,BV]  (warp w owns BT-rows [w*16:+16])
    for (int bvc = 0; bvc < BV / kWM; ++bvc) {
      typename Cfg::Acc acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int64_t kk = 0; kk < dk; kk += WK) {
        typename Cfg::Arow a;   // W[t,k]
        typename Cfg::Bcol b;   // b_hᵀ[k,v] = Hbf[v,k] loaded col-major
        Cfg::load(a, w + (tok0 + warp * kWM) * hv_n * dk + hv * dk + kk, hv_n * dk);
        Cfg::load(b, Hbf + bvc * kWM * dk + kk, dk);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(Braw + (warp * kWM) * BV + bvc * kWM, acc, BV, wmma::mem_row_major);
    }
    __syncthreads();
    // subtract u, write v_new(global)+Vbf(on-chip); zero the partial-tail rows.
    // 16-byte-vectorized: a VN-group shares one row i (BV is a multiple of VN).
    for (int64_t e = tid * VN; e < BT * BV; e += (int64_t)blockDim.x * VN) {
      const int64_t i = e / BV, cvi = e % BV;
      if (i < len) {
        float uf[VN];
        V128<TD>::Uf(u + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, uf);
#pragma unroll
        for (int j = 0; j < VN; ++j) uf[j] -= Braw[i * BV + cvi + j];
        V128<TD>::Sf(v_new + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, uf);
        V128<TD>::Sf(Vbf + e, uf);
      } else {
        V128<TD>::Sf0(Vbf + e);
      }
    }
    __syncthreads();

    // (c) decay folded into K (Kd = k⊙decay, tail rows 0) + b_h *= exp(g_last)
    for (int64_t e = tid * VN; e < BT * dk; e += (int64_t)blockDim.x * VN) {
      const int64_t i = e / dk, ki = e % dk;
      if (i < len) {
        float kf[VN];
        V128<TD>::Uf(k + (tok0 + i) * hk_n * dk + hk * dk + ki, kf);
        const float d = decay[i];
#pragma unroll
        for (int j = 0; j < VN; ++j) kf[j] *= d;
        V128<TD>::Sf(Kd + e, kf);
      } else {
        V128<TD>::Sf0(Kd + e);
      }
    }
    for (int kt = 0; kt < NK; ++kt) {
      for (int x = 0; x < h1[kt].num_elements; ++x) h1[kt].x[x] *= eglast;
      for (int x = 0; x < h2[kt].num_elements; ++x) h2[kt].x[x] *= eglast;
    }
    __syncthreads();

    // (e) V2: b_h[BV,64] += b_vᵀ[BV,BT] @ Kd[BT,64] — accumulate into the
    // persistent fragments directly. matrix_a = Vbf col-major (A[v,t]=Vbf[t,v]).
    for (int64_t tt = 0; tt < BT; tt += WK) {
      typename Cfg::Acol a;
      Cfg::load(a, Vbf + warp * kWM + tt * BV, BV);
      for (int kt = 0; kt < NK; ++kt) {
        typename Cfg::Brow b1, b2;
        Cfg::load(b1, Kd + tt * dk + kt * kWM, dk);
        wmma::mma_sync(h1[kt], a, b1, h1[kt]);
        Cfg::load(b2, Kd + tt * dk + 64 + kt * kWM, dk);
        wmma::mma_sync(h2[kt], a, b2, h2[kt]);
      }
    }
    __syncthreads();
  }
  // epilogue: b_h -> state(f32) directly (fragment f32 -> f32 global, no convert)
  for (int kt = 0; kt < NK; ++kt) {
    wmma::store_matrix_sync(sH + vrow0 * dk + kt * kWM, h1[kt], dk, wmma::mem_row_major);
    wmma::store_matrix_sync(sH + vrow0 * dk + 64 + kt * kWM, h2[kt], dk, wmma::mem_row_major);
  }
}

// ----------------------------------------------------------------------------
// DeltaH (REGISTER-tiled + N-stage cp.async ring, vt::tile Rung-1 STEP 2). Same
// persistent-accumulator FLA blockdim64 port as GdnChunkDeltaHRegKernel, but the
// streamed per-chunk W and K tiles come through an N-stage cp.async software
// pipeline (the ring the freed smem enables): chunk it+STAGES is prefetched while
// chunk it computes, hiding the global-load latency that the 1-block/SM occupancy
// (ring smem => 1 block) cannot hide via warp-switching. bf16 only (the f32 ring
// tiles overflow the 99 KiB opt-in); f32 falls back to the ring-less reg kernel.
//
// Ported FROM: fla/ops/chunk_delta_h.py:43-315 (pin e24d1b24, decay-on-b_v form,
// FLA's actual);  cute/arch/copy_sm80.hpp:40-193 via vt::cuda::tile::cp_async_cg
// (16B ZFILL async copy) + PipelineState. Two smem savings vs the ring-less
// kernel free room for the 2-stage ring: (1) snapshot staged in HALVES (16 KiB
// Hf32 not 32); (2) decay folded into b_v (FLA form) so V2 reads K straight from
// the ring (no separate Kd tile). ZFILL predicate masks the partial-tail rows.
template <typename TD, int BV, int STAGES>
__global__ void GdnChunkDeltaHRegRingKernel(float* state, TD* hstate, TD* v_new, const TD* k,
                                            const TD* u, const TD* w, const float* gcum,
                                            const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                            int64_t dk, int64_t hv_n, int64_t dv) {
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  constexpr int BT = kChunk;
  constexpr int NK = 64 / kWM;
  constexpr int VN = V128<TD>::N;
  constexpr int VW = 16 / sizeof(TD);  // TD per 16B cp.async vector (8 bf16)
  const int64_t i_v = blockIdx.x, i_nh = blockIdx.y;
  const int64_t n = i_nh / hv_n, hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int nthreads = static_cast<int>(blockDim.x);

  const int64_t rA = BV * 64 * (int64_t)sizeof(float);  // Hf32(half) == Braw == 16 KiB
  const int64_t rB = BV * dk * (int64_t)sizeof(TD);     // Hbf full == 16 KiB
  const int64_t tile = BT * dk * (int64_t)sizeof(TD);   // one W or K tile (16 KiB)
  extern __shared__ char smem_raw[];
  float* Hf32 = reinterpret_cast<float*>(smem_raw);  // [BV,64] f32 (region A)
  float* Braw = reinterpret_cast<float*>(smem_raw);  // [BT,BV] f32 (aliases A)
  TD* Hbf = reinterpret_cast<TD*>(smem_raw + rA);    // [BV,dk] TD  (region B)
  TD* Vbf = reinterpret_cast<TD*>(smem_raw + rA);    // [BT,BV] TD  (aliases B)
  char* ring = smem_raw + rA + rB;                   // STAGES * (W tile | K tile)
  __shared__ float decay[BT];

  float* sH = state + (n * hv_n + hv) * sd;
  const int64_t vrow0 = i_v * BV + warp * kWM;
  typename Cfg::Acc h1[NK], h2[NK];
  for (int kt = 0; kt < NK; ++kt) {
    wmma::load_matrix_sync(h1[kt], sH + vrow0 * dk + kt * kWM, dk, wmma::mem_row_major);
    wmma::load_matrix_sync(h2[kt], sH + vrow0 * dk + 64 + kt * kWM, dk, wmma::mem_row_major);
  }
  const int64_t nt = (seqlen + BT - 1) / BT;
  const int nvec = BT * (dk / VW);

  // Issue the cp.async loads of chunk `it`'s W and K into ring stage `s` (ZFILL
  // rows >= len for the partial tail). Caller issues the commit fence.
  auto issue = [&](int64_t it, int s) {
    if (it >= nt) return;
    const int64_t tok0 = begin + it * BT;
    const int64_t rem = seqlen - it * BT;
    const int64_t len = rem < BT ? rem : BT;
    TD* Wd = reinterpret_cast<TD*>(ring + (int64_t)s * 2 * tile);
    TD* Kd = reinterpret_cast<TD*>(ring + (int64_t)s * 2 * tile + tile);
    for (int vi = tid; vi < nvec; vi += nthreads) {
      const int r = vi / (dk / VW), g = vi % (dk / VW);
      const bool pred = r < len;
      vt::cuda::tile::cp_async_cg<16>(Wd + r * dk + g * VW,
                                      w + (tok0 + r) * hv_n * dk + hv * dk + g * VW, pred);
      vt::cuda::tile::cp_async_cg<16>(Kd + r * dk + g * VW,
                                      k + (tok0 + r) * hk_n * dk + hk * dk + g * VW, pred);
    }
  };

  // prologue: prime all STAGES stages
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    issue(s, s);
    vt::cuda::tile::cp_async_fence();
  }

  for (int64_t it = 0; it < nt; ++it) {
    const int stage = static_cast<int>(it % STAGES);
    vt::cuda::tile::cp_async_wait<STAGES - 1>();
    __syncthreads();
    TD* Wcur = reinterpret_cast<TD*>(ring + (int64_t)stage * 2 * tile);
    TD* Kcur = reinterpret_cast<TD*>(ring + (int64_t)stage * 2 * tile + tile);

    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * BT;
    const int64_t rem = seqlen - it * BT;
    const int64_t len = rem < BT ? rem : BT;
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += nthreads)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);

    // (a) snapshot in HALVES (16 KiB Hf32) -> hstate(bf16) + V1 operand Hbf.
    TD* hsnap = hstate + (gc * hv_n + hv) * sd;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int coff = half * 64;
      for (int kt = 0; kt < NK; ++kt)
        wmma::store_matrix_sync(Hf32 + (warp * kWM) * 64 + kt * kWM, half == 0 ? h1[kt] : h2[kt],
                                64, wmma::mem_row_major);
      __syncthreads();
      for (int64_t e = tid * VN; e < BV * 64; e += (int64_t)nthreads * VN) {
        float f[VN];
#pragma unroll
        for (int j = 0; j < VN; j += 4) {
          const float4 v = *reinterpret_cast<const float4*>(Hf32 + e + j);
          f[j] = v.x;
          f[j + 1] = v.y;
          f[j + 2] = v.z;
          f[j + 3] = v.w;
        }
        const int64_t r = e / 64, c = e % 64;
        V128<TD>::Sf(hsnap + (i_v * BV + r) * dk + coff + c, f);
        V128<TD>::Sf(Hbf + r * dk + coff + c, f);
      }
      __syncthreads();
    }

    // (b) V1: b_v[BT,BV] = W @ b_hᵀ  (W straight from the ring)
    for (int bvc = 0; bvc < BV / kWM; ++bvc) {
      typename Cfg::Acc acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int64_t kk = 0; kk < dk; kk += WK) {
        typename Cfg::Arow a;
        typename Cfg::Bcol b;
        Cfg::load(a, Wcur + (warp * kWM) * dk + kk, dk);
        Cfg::load(b, Hbf + bvc * kWM * dk + kk, dk);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(Braw + (warp * kWM) * BV + bvc * kWM, acc, BV, wmma::mem_row_major);
    }
    __syncthreads();
    // subtract u (global): v_new = undecayed b_v; Vbf = decayed b_v (FLA form).
    for (int64_t e = tid * VN; e < BT * BV; e += (int64_t)nthreads * VN) {
      const int64_t i = e / BV, cvi = e % BV;
      if (i < len) {
        float bf[VN];
        V128<TD>::Uf(u + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, bf);
#pragma unroll
        for (int j = 0; j < VN; ++j) bf[j] -= Braw[i * BV + cvi + j];
        V128<TD>::Sf(v_new + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, bf);
        const float d = decay[i];
#pragma unroll
        for (int j = 0; j < VN; ++j) bf[j] *= d;
        V128<TD>::Sf(Vbf + e, bf);
      } else {
        V128<TD>::Sf0(Vbf + e);
      }
    }
    for (int kt = 0; kt < NK; ++kt) {
      for (int x = 0; x < h1[kt].num_elements; ++x) h1[kt].x[x] *= eglast;
      for (int x = 0; x < h2[kt].num_elements; ++x) h2[kt].x[x] *= eglast;
    }
    __syncthreads();

    // (e) V2: b_h += b_vᵀ(decayed) @ K(undecayed, from the ring)
    for (int64_t tt = 0; tt < BT; tt += WK) {
      typename Cfg::Acol a;
      Cfg::load(a, Vbf + warp * kWM + tt * BV, BV);
      for (int kt = 0; kt < NK; ++kt) {
        typename Cfg::Brow b1, b2;
        Cfg::load(b1, Kcur + tt * dk + kt * kWM, dk);
        wmma::mma_sync(h1[kt], a, b1, h1[kt]);
        Cfg::load(b2, Kcur + tt * dk + 64 + kt * kWM, dk);
        wmma::mma_sync(h2[kt], a, b2, h2[kt]);
      }
    }
    __syncthreads();
    // prefetch chunk it+STAGES into the just-consumed stage.
    issue(it + STAGES, stage);
    vt::cuda::tile::cp_async_fence();
  }
  for (int kt = 0; kt < NK; ++kt) {
    wmma::store_matrix_sync(sH + vrow0 * dk + kt * kWM, h1[kt], dk, wmma::mem_row_major);
    wmma::store_matrix_sync(sH + vrow0 * dk + 64 + kt * kWM, h2[kt], dk, wmma::mem_row_major);
  }
}

// DeltaH (REGISTER-tiled + N-stage TMA+mbarrier ring, vt::tile Rung-2). Identical
// persistent-accumulator FLA blockdim64 compute as GdnChunkDeltaHRegRingKernel; the
// ONLY change is the W/K chunk streaming: instead of the sm80 cp.async ring
// (Rung-1), each stage is filled by a TMA bulk-tensor copy (cp.async.bulk.tensor.3d)
// whose completion is signalled on a per-stage mbarrier (transaction-bytes) — the
// Blackwell-native async pipeline Triton emits. This is the DECISIVE A/B: same
// smem layout (plain row-major, swizzle NONE) + same WMMA, so it isolates the
// COPY/PIPELINE mechanism (cp.async vs TMA+mbarrier) from every other variable.
// bf16 only. Behind VT_GDN_TMA (default OFF).
//
// Ported FROM: fla/ops/chunk_delta_h.py:43-315 (compute, decay-on-b_v FLA form) +
// gdn_chunk_cutedsl/kernel_h.py:198-270 (the TMA warp's mbarrier producer loop:
// mbarrier_init(count) / arrive_and_expect_tx(STAGE_SIZE) / simple_tma_copy /
// mbarrier_wait(parity), num_stages ring) mapped to a unified (non-warp-specialized)
// pipeline feeding WMMA (sm_121 has no tcgen05/tmem; the DSL's MMA path is sm_100).
// TMA/mbarrier atoms: vt::cuda::tile::tma_pipeline.cuh (cited to CUTLASS there).
template <typename TD, int BV, int STAGES>
__global__ void GdnChunkDeltaHTmaKernel(float* state, TD* hstate, TD* v_new, const TD* k,
                                        const TD* u, const TD* w, const float* gcum,
                                        const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                                        int64_t dk, int64_t hv_n, int64_t dv,
                                        const __grid_constant__ CUtensorMap descW,
                                        const __grid_constant__ CUtensorMap descK) {
  namespace tp = vt::cuda::tile;
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  constexpr int BT = kChunk;
  constexpr int NK = 64 / kWM;
  constexpr int VN = V128<TD>::N;
  const int64_t i_v = blockIdx.x, i_nh = blockIdx.y;
  const int64_t n = i_nh / hv_n, hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t begin = qsl[n], seqlen = qsl[n + 1] - begin;
  const int64_t boh_n = boh[n];
  const int64_t sd = dv * dk;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int nthreads = static_cast<int>(blockDim.x);

  const int64_t rA = BV * 64 * (int64_t)sizeof(float);  // Hf32(half) == Braw == 16 KiB
  const int64_t rB = BV * dk * (int64_t)sizeof(TD);     // Hbf full == 16 KiB
  const int64_t tile = BT * dk * (int64_t)sizeof(TD);   // one W or K tile (16 KiB)
  extern __shared__ __align__(128) char smem_tma[];     // 128B base for the TMA dst
  float* Hf32 = reinterpret_cast<float*>(smem_tma);  // [BV,64] f32 (region A)
  float* Braw = reinterpret_cast<float*>(smem_tma);  // [BT,BV] f32 (aliases A)
  TD* Hbf = reinterpret_cast<TD*>(smem_tma + rA);    // [BV,dk] TD  (region B)
  TD* Vbf = reinterpret_cast<TD*>(smem_tma + rA);    // [BT,BV] TD  (aliases B)
  char* ring = smem_tma + rA + rB;                   // STAGES * (W tile | K tile), 128B-aligned
  __shared__ float decay[BT];
  __shared__ uint64_t full_mbar[STAGES];             // one completion barrier per stage

  float* sH = state + (n * hv_n + hv) * sd;
  const int64_t vrow0 = i_v * BV + warp * kWM;
  typename Cfg::Acc h1[NK], h2[NK];
  for (int kt = 0; kt < NK; ++kt) {
    wmma::load_matrix_sync(h1[kt], sH + vrow0 * dk + kt * kWM, dk, wmma::mem_row_major);
    wmma::load_matrix_sync(h2[kt], sH + vrow0 * dk + 64 + kt * kWM, dk, wmma::mem_row_major);
  }
  const int64_t nt = (seqlen + BT - 1) / BT;
  const uint32_t txbytes = static_cast<uint32_t>(2 * BT * dk * (int64_t)sizeof(TD));  // W+K

  // arm the per-stage mbarriers (count=1: the single arrive.expect_tx below), then
  // fence so the init is visible to the async (TMA) proxy before any copy.
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) tp::mbarrier_init(&full_mbar[s], 1);
    tp::fence_proxy_async_shared();
  }
  __syncthreads();

  // Producer: one elected thread arms stage `s`'s barrier with the W+K tx bytes and
  // issues the two TMA bulk copies (chunk `it`) into it. TMA hardware zero-fills any
  // OOB tail (partial last chunk) — masked downstream regardless.
  auto issue = [&](int64_t it, int s) {
    if (tid != 0 || it >= nt) return;
    const int64_t tok0 = begin + it * BT;
    TD* Wd = reinterpret_cast<TD*>(ring + (int64_t)s * 2 * tile);
    TD* Kd = reinterpret_cast<TD*>(ring + (int64_t)s * 2 * tile + tile);
    tp::mbarrier_arrive_expect_tx(&full_mbar[s], txbytes);
    tp::tma_load_3d(&descW, &full_mbar[s], Wd, 0, static_cast<int>(hv), static_cast<int>(tok0));
    tp::tma_load_3d(&descK, &full_mbar[s], Kd, 0, static_cast<int>(hk), static_cast<int>(tok0));
  };

  // prologue: prime all STAGES stages
#pragma unroll
  for (int s = 0; s < STAGES; ++s) issue(s, s);

  for (int64_t it = 0; it < nt; ++it) {
    const int stage = static_cast<int>(it % STAGES);
    const uint32_t phase = static_cast<uint32_t>((it / STAGES) & 1);
    tp::mbarrier_wait(&full_mbar[stage], phase);  // all threads block until W+K landed
    __syncthreads();
    TD* Wcur = reinterpret_cast<TD*>(ring + (int64_t)stage * 2 * tile);
    TD* Kcur = reinterpret_cast<TD*>(ring + (int64_t)stage * 2 * tile + tile);

    const int64_t gc = boh_n + it;
    const int64_t tok0 = begin + it * BT;
    const int64_t rem = seqlen - it * BT;
    const int64_t len = rem < BT ? rem : BT;
    const float glast = gcum[(tok0 + len - 1) * hv_n + hv];
    const float eglast = expf(glast);
    for (int64_t i = tid; i < len; i += nthreads)
      decay[i] = expf(glast - gcum[(tok0 + i) * hv_n + hv]);

    // (a) snapshot in HALVES (16 KiB Hf32) -> hstate(bf16) + V1 operand Hbf.
    TD* hsnap = hstate + (gc * hv_n + hv) * sd;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int coff = half * 64;
      for (int kt = 0; kt < NK; ++kt)
        wmma::store_matrix_sync(Hf32 + (warp * kWM) * 64 + kt * kWM, half == 0 ? h1[kt] : h2[kt],
                                64, wmma::mem_row_major);
      __syncthreads();
      for (int64_t e = tid * VN; e < BV * 64; e += (int64_t)nthreads * VN) {
        float f[VN];
#pragma unroll
        for (int j = 0; j < VN; j += 4) {
          const float4 v = *reinterpret_cast<const float4*>(Hf32 + e + j);
          f[j] = v.x;
          f[j + 1] = v.y;
          f[j + 2] = v.z;
          f[j + 3] = v.w;
        }
        const int64_t r = e / 64, c = e % 64;
        V128<TD>::Sf(hsnap + (i_v * BV + r) * dk + coff + c, f);
        V128<TD>::Sf(Hbf + r * dk + coff + c, f);
      }
      __syncthreads();
    }

    // (b) V1: b_v[BT,BV] = W @ b_hᵀ  (W straight from the TMA ring)
    for (int bvc = 0; bvc < BV / kWM; ++bvc) {
      typename Cfg::Acc acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int64_t kk = 0; kk < dk; kk += WK) {
        typename Cfg::Arow a;
        typename Cfg::Bcol b;
        Cfg::load(a, Wcur + (warp * kWM) * dk + kk, dk);
        Cfg::load(b, Hbf + bvc * kWM * dk + kk, dk);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(Braw + (warp * kWM) * BV + bvc * kWM, acc, BV, wmma::mem_row_major);
    }
    __syncthreads();
    // subtract u (global): v_new = undecayed b_v; Vbf = decayed b_v (FLA form).
    for (int64_t e = tid * VN; e < BT * BV; e += (int64_t)nthreads * VN) {
      const int64_t i = e / BV, cvi = e % BV;
      if (i < len) {
        float bf[VN];
        V128<TD>::Uf(u + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, bf);
#pragma unroll
        for (int j = 0; j < VN; ++j) bf[j] -= Braw[i * BV + cvi + j];
        V128<TD>::Sf(v_new + (tok0 + i) * hv_n * dv + hv * dv + i_v * BV + cvi, bf);
        const float d = decay[i];
#pragma unroll
        for (int j = 0; j < VN; ++j) bf[j] *= d;
        V128<TD>::Sf(Vbf + e, bf);
      } else {
        V128<TD>::Sf0(Vbf + e);
      }
    }
    for (int kt = 0; kt < NK; ++kt) {
      for (int x = 0; x < h1[kt].num_elements; ++x) h1[kt].x[x] *= eglast;
      for (int x = 0; x < h2[kt].num_elements; ++x) h2[kt].x[x] *= eglast;
    }
    __syncthreads();

    // (e) V2: b_h += b_vᵀ(decayed) @ K(undecayed, from the TMA ring)
    for (int64_t tt = 0; tt < BT; tt += WK) {
      typename Cfg::Acol a;
      Cfg::load(a, Vbf + warp * kWM + tt * BV, BV);
      for (int kt = 0; kt < NK; ++kt) {
        typename Cfg::Brow b1, b2;
        Cfg::load(b1, Kcur + tt * dk + kt * kWM, dk);
        wmma::mma_sync(h1[kt], a, b1, h1[kt]);
        Cfg::load(b2, Kcur + tt * dk + 64 + kt * kWM, dk);
        wmma::mma_sync(h2[kt], a, b2, h2[kt]);
      }
    }
    __syncthreads();  // WAR: all consumers done reading Wcur/Kcur before the re-issue overwrites
    // prefetch chunk it+STAGES into the just-consumed stage.
    issue(it + STAGES, stage);
  }
  for (int kt = 0; kt < NK; ++kt) {
    wmma::store_matrix_sync(sH + vrow0 * dk + kt * kWM, h1[kt], dk, wmma::mem_row_major);
    wmma::store_matrix_sync(sH + vrow0 * dk + 64 + kt * kWM, h2[kt], dk, wmma::mem_row_major);
  }
}

// ChunkO (WMMA). One block per (chunk, v-head). cross = Q@Hstartᵀ; A = Q@Kᵀ
// (decay-weighted, causal-masked); o = scale*(exp(G)*cross + A@V_new). Buffers
// alias: Ks (used only for qk) is reused as outc after qk.
template <typename TD, typename Tout>
__global__ void GdnChunkOWmmaKernel(Tout* out, const TD* q, const TD* k, const TD* v_new,
                                    const TD* hstate, const float* gcum, const int32_t* tok0a,
                                    const int32_t* lena, int64_t hk_n, int64_t dk, int64_t hv_n,
                                    int64_t dv, float scale, bool vec) {
  using Cfg = WmmaCfg<TD>;
  using V = V128<TD>;
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

  if (vec)
    for (int64_t g = tid; g < (kChunk * dk) / V::N; g += blockDim.x) {
      const int64_t base = g * V::N;
      const int64_t i = base / dk, d = base % dk;
      if (i < len) {
        V::Cpy(Qs + base, q + (tok0 + i) * hk_n * dk + hk * dk + d);
        V::Cpy(Ks + base, k + (tok0 + i) * hk_n * dk + hk * dk + d);
      } else {
        V::Sf0(Qs + base);
        V::Sf0(Ks + base);
      }
    }
  else
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

// Step B (WMMA) — chunk_scaled_dot_kkt Gram on tensor cores. The scalar
// GdnChunkWUKernel spends its bulk in the O(BT²·Dk) Gram K@Kᵀ (prefill's last
// scalar-float GDN chunk kernel, ~22% GPU). This replaces that cooperative
// dot-product with a WMMA 16x16 matmul, f32 accumulate — exactly the qk=Q@Kᵀ
// block of GdnChunkOWmmaKernel but with K in both operands (mirrors FLA
// chunk_scaled_dot_kkt.py:86 b_A += tl.dot(b_kb, tl.trans(b_k))). The
// exp(G_i−G_j) diff is folded into the Gram (chunk_scaled_dot_kkt.py:92
// b_A = b_A*exp(b_g_diff)) so the shared A[i,j] holds exp(G_i−G_j)·(k_i·k_j)
// masked to the strict-lower (causal) triangle. The two WY forward
// substitutions (u from β⊙v, w from β⊙exp(G)⊙k) stay f32 — they carry a
// sequential i→i dependency (solve_tril.py:84 loop) that is not a matmul.
// TD scratch dtype: bf16 native fragments; f32 uses TF32 (WmmaCfg), the path
// the real 35B GDN (f32) hits — same precision regime as DeltaH/O.
template <typename TD>
__global__ void GdnChunkWUWmmaKernel(TD* u, TD* w, const TD* k, const TD* v, const float* beta,
                                     const float* gcum, const int32_t* tok0a, const int32_t* lena,
                                     int64_t hk_n, int64_t dk, int64_t hv_n, int64_t dv) {
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, nwarps = static_cast<int>(blockDim.x) / 32;

  extern __shared__ char smem_raw[];
  TD* Ks = reinterpret_cast<TD*>(smem_raw);                 // [kChunk,Dk] staged K (zero-padded)
  float* kk = reinterpret_cast<float*>(Ks + kChunk * dk);   // [kChunk,kChunk] Gram -> A
  __shared__ float gs[kChunk];                              // G_i (local cumsum)
  __shared__ float bs[kChunk];                              // beta_i
  __shared__ float eg[kChunk];                              // exp(G_i)

  // Stage K into shared, zero past len so tail (partial-chunk) tiles are finite
  // and never read past the k input's t_tot rows (mirrors ChunkO's Ks staging).
  for (int64_t e = tid; e < kChunk * dk; e += blockDim.x) {
    const int64_t i = e / dk, d = e % dk;
    Store(Ks, e, i < len ? Load(k, (tok0 + i) * hk_n * dk + hk * dk + d) : 0.0f);
  }
  for (int64_t i = tid; i < len; i += blockDim.x) {
    bs[i] = beta[(tok0 + i) * hv_n + hv];
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();

  // kk = K @ Kᵀ  [kChunk,kChunk]  (Arow = K[i,:], Bcol = K[j,:] read as Kᵀ)
  const int nkk = (kChunk / kWM) * (kChunk / kWM);
  for (int t = warp; t < nkk; t += nwarps) {
    const int it = t % (kChunk / kWM), jt = t / (kChunk / kWM);
    const int i0 = it * kWM, j0 = jt * kWM;
    typename Cfg::Acc acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t d = 0; d < dk; d += WK) {
      typename Cfg::Arow a;
      typename Cfg::Bcol b;
      Cfg::load(a, Ks + i0 * dk + d, dk);
      Cfg::load(b, Ks + j0 * dk + d, dk);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(kk + i0 * kChunk + j0, acc, kChunk, wmma::mem_row_major);
  }
  __syncthreads();
  // A[i,j] = exp(G_i − G_j) · kk[i,j] for j<i (strict-lower causal), else 0.
  for (int64_t e = tid; e < kChunk * kChunk; e += blockDim.x) {
    const int64_t i = e / kChunk, j = e % kChunk;
    kk[e] = (i < len && j < i) ? expf(gs[i] - gs[j]) * kk[e] : 0.0f;
  }
  __syncthreads();

  // u column vi: u_i = beta_i (v_i − sum_{j<i} A[i,j] u_j)  (f32 forward solve)
  for (int64_t vi = tid; vi < dv; vi += blockDim.x) {
    float ucol[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float sfs = 0.0f;
      for (int64_t j = 0; j < i; ++j) sfs += kk[i * kChunk + j] * ucol[j];
      const float vi_val = Load(v, (tok0 + i) * hv_n * dv + hv * dv + vi);
      ucol[i] = bs[i] * (vi_val - sfs);
    }
    for (int64_t i = 0; i < len; ++i)
      Store(u, (tok0 + i) * hv_n * dv + hv * dv + vi, ucol[i]);
  }
  // w column ki: w_i = beta_i (exp(G_i) k_i − sum_{j<i} A[i,j] w_j)
  for (int64_t ki = tid; ki < dk; ki += blockDim.x) {
    float wcol[kChunk];
    for (int64_t i = 0; i < len; ++i) {
      float sfs = 0.0f;
      for (int64_t j = 0; j < i; ++j) sfs += kk[i * kChunk + j] * wcol[j];
      const float ki_val = Load(k, (tok0 + i) * hk_n * dk + hk * dk + ki);
      wcol[i] = bs[i] * (eg[i] * ki_val - sfs);
    }
    for (int64_t i = 0; i < len; ++i)
      Store(w, (tok0 + i) * hv_n * dk + hv * dk + ki, wcol[i]);
  }
}

// Step B (WMMA, VEC path — VT_GDN_CHUNK_VEC, default). FLA-faithful WY: the
// scalar GdnChunkWUWmmaKernel above spends ~73% of its time (measured GB10
// ablation) in the O(BT²·(Dv+Dk)) per-column forward-substitution — Dv+Dk=256
// independent length-BT serial solves through LOCAL memory (ucol/wcol spill,
// STACK:256), latency/throughput-bound at 256-thread occupancy. This mirrors
// vLLM's FLA instead: fold β into A (chunk_scaled_dot_kkt.py:86-92
// b_kb=b_k*b_beta, b_A*=exp(g_diff), strict-lower mask), compute the [BT,BT]
// triangular inverse T=(I+A)⁻¹ ONCE in shared f32 (solve_tril.py — column
// forward-subst, no local spill, BT² not BT²·(Dv+Dk) work), then apply on the
// TENSOR CORES: u=T@(β⊙v), w=T@(β⊙exp(G)⊙k) (wy_fast.recompute_w_u_fwd). The
// inverse-then-apply is algebraically identical to the fused forward-solve
// ((I+A)u=β⊙v ⇒ u_i=β_i(v_i−Σ_{j<i}exp(G_i−G_j)KK[i,j]u_j)); T is rounded to TD
// (bf16 native / f32→TF32) before the WMMA apply, the same precision regime FLA
// uses (solve_tril output_dtype=k.dtype). Compact aliased shared arena (2 regions,
// reused across the Gram/inverse/apply phases → 2 blocks/SM on GB10, vs a naive
// 1-block/SM layout that leaves the serial inverse + staging occupancy-starved):
//   R0 = max(Ks[BT,Dk], Tf[BT,BT]f32): Gram-K → inverse-T(f32) → apply-operand(TD)
//   R1 = max(Am[BT,BT]f32, Tb[BT,BT]TD + Osh[BT,kOBlk]f32): Gram-out/A → T(TD)+apply-out
// The apply streams u then w in column blocks of kOBlk (per-pass operand build,
// so only ONE [BT,Dcol] operand is resident) — keeps the peak at ~40 KiB (bf16).
constexpr int kOBlk = 64;  // WMMA apply output column block (multiple of kWM)

// Blocked triangular-inverse Schur merge (mirrors FLA solve_tril.py
// merge_16x16_to_64x64_inverse_kernel:356-390, off-diagonal fill). Called by ONE
// warp to compute the [16,16] off-diagonal inverse block
//   T(bi,bj) = -T(bi,bi) @ ( Σ_{k=bj}^{bi-1} A(bi,k) @ T(k,bj) )
// on the tensor cores, where A is the strict-lower [BT,BT] f32 matrix (Am, the
// beta/decay-folded K@Kᵀ) and T is the running [BT,BT] f32 inverse (Tf). The two
// diagonal-inverse operands T(bi,bi)/T(k,bj) and the A(bi,k) blocks must already
// be resident in Tf/Am. `Pw` is a per-warp [16,16] f32 scratch. Result is stored
// (negated) into the (bi,bj) block of Tf. The inverse is carried in f32 shared for
// BOTH kernel dtypes (Am/Tf/Pw are f32; the apply rounds Tf→Tb(TD) afterwards), so
// the merges use the f32/TF32 WMMA config unconditionally — the same precision the
// f32 apply already runs at (solve_tril output_dtype=k.dtype).
__device__ inline void WyMerge(float* Tf, const float* Am, float* Pw, int64_t BT, int bi,
                               int bj) {
  using Cfg = WmmaCfg<float>;
  constexpr int WK = Cfg::WK;
  // P = Σ_{bk=bj}^{bi-1} A(bi,bk) @ T(bk,bj)  (f32 accumulate over the block chain).
  typename Cfg::Acc accP;
  wmma::fill_fragment(accP, 0.0f);
  for (int bk = bj; bk < bi; ++bk) {
    for (int kk = 0; kk < kWM; kk += WK) {
      typename Cfg::Arow a;  // A(bi,bk)[:, kk:kk+WK]
      typename Cfg::Brow b;  // T(bk,bj)[kk:kk+WK, :]
      Cfg::load(a, Am + (bi * kWM) * BT + (bk * kWM) + kk, BT);
      Cfg::load(b, Tf + (bk * kWM + kk) * BT + (bj * kWM), BT);
      wmma::mma_sync(accP, a, b, accP);
    }
  }
  wmma::store_matrix_sync(Pw, accP, kWM, wmma::mem_row_major);
  __syncwarp();
  // T(bi,bj) = -T(bi,bi) @ P.
  typename Cfg::Acc accT;
  wmma::fill_fragment(accT, 0.0f);
  for (int kk = 0; kk < kWM; kk += WK) {
    typename Cfg::Arow a;  // T(bi,bi)[:, kk:kk+WK]
    typename Cfg::Brow b;  // P[kk:kk+WK, :]
    Cfg::load(a, Tf + (bi * kWM) * BT + (bi * kWM) + kk, BT);
    Cfg::load(b, Pw + kk * kWM, kWM);
    wmma::mma_sync(accT, a, b, accT);
  }
  for (int i = 0; i < accT.num_elements; ++i) accT.x[i] = -accT.x[i];
  wmma::store_matrix_sync(Tf + (bi * kWM) * BT + (bj * kWM), accT, BT, wmma::mem_row_major);
}

template <typename TD>
__global__ void GdnChunkWUWmmaVecKernel(TD* u, TD* w, const TD* k, const TD* v,
                                        const float* beta, const float* gcum,
                                        const int32_t* tok0a, const int32_t* lena, int64_t hk_n,
                                        int64_t dk, int64_t hv_n, int64_t dv, bool blocked) {
  using Cfg = WmmaCfg<TD>;
  constexpr int WK = Cfg::WK;
  const int64_t gc = blockIdx.x, hv = blockIdx.y;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t tok0 = tok0a[gc], len = lena[gc];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, nwarps = static_cast<int>(blockDim.x) / 32;

  extern __shared__ char smem_raw[];
  const int64_t BT = kChunk;
  const size_t s0a = static_cast<size_t>(BT * dk) * sizeof(TD);
  const size_t s0b = static_cast<size_t>(BT * BT) * sizeof(float);
  const size_t s0 = s0a > s0b ? s0a : s0b;
  char* R0 = smem_raw;
  char* R1 = smem_raw + s0;
  TD* Ks = reinterpret_cast<TD*>(R0);       // [BT,Dk] TD  (Gram in)
  float* Tf = reinterpret_cast<float*>(R0);  // [BT,BT] f32 (inverse; aliases Ks)
  TD* Bop = reinterpret_cast<TD*>(R0);      // [BT,Dcol] TD (apply operand; aliases R0)
  float* Am = reinterpret_cast<float*>(R1);  // [BT,BT] f32 (Gram out → A)
  TD* Tb = reinterpret_cast<TD*>(R1);       // [BT,BT] TD  (T; aliases Am after inverse)
  float* Osh = reinterpret_cast<float*>(R1 + static_cast<size_t>(BT * BT) * sizeof(TD));  // [BT,kOBlk] f32
  __shared__ float gs[kChunk], bs[kChunk], eg[kChunk];

  // Stage K into shared (zero past len so tail tiles are finite).
  for (int64_t e = tid; e < BT * dk; e += blockDim.x) {
    const int64_t i = e / dk, d = e % dk;
    Store(Ks, e, i < len ? Load(k, (tok0 + i) * hk_n * dk + hk * dk + d) : 0.0f);
  }
  for (int64_t i = tid; i < len; i += blockDim.x) {
    bs[i] = beta[(tok0 + i) * hv_n + hv];
    gs[i] = gcum[(tok0 + i) * hv_n + hv];
    eg[i] = expf(gs[i]);
  }
  __syncthreads();

  // Gram Am = K@Kᵀ  [BT,BT] f32 (Arow=K[i,:], Bcol=K[j,:] read as Kᵀ).
  const int nkk = (BT / kWM) * (BT / kWM);
  for (int t = warp; t < nkk; t += nwarps) {
    const int it = t % (BT / kWM), jt = t / (BT / kWM);
    const int i0 = it * kWM, j0 = jt * kWM;
    typename Cfg::Acc acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int64_t d = 0; d < dk; d += WK) {
      typename Cfg::Arow a;
      typename Cfg::Bcol b;
      Cfg::load(a, Ks + i0 * dk + d, dk);
      Cfg::load(b, Ks + j0 * dk + d, dk);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(Am + i0 * BT + j0, acc, BT, wmma::mem_row_major);
  }
  __syncthreads();

  // A[i,j] = β_i·exp(G_i−G_j)·KK[i,j] for j<i (strict-lower causal), else 0 (in place).
  for (int64_t e = tid; e < BT * BT; e += blockDim.x) {
    const int64_t i = e / BT, j = e % BT;
    Am[e] = (i < len && j < i) ? bs[i] * expf(gs[i] - gs[j]) * Am[e] : 0.0f;
  }
  __syncthreads();  // Ks (R0) free → Tf aliases it.

  // T = (I + A)⁻¹, unit lower-triangular. Two solvers (VT_GDN_WY_BLOCKED):
  for (int64_t e = tid; e < BT * BT; e += blockDim.x) Tf[e] = 0.0f;
  __syncthreads();
  if (!blocked) {
    // Serial (fallback/default): column-wise forward-substitution in shared f32
    // (no local spill). Column c: T[c,c]=1, T[i,c]=−Σ_{m=c}^{i−1} A[i,m]T[m,c].
    // Only BT of the blockDim.x threads work (one dependent chain per column,
    // the longest ~BT·(BT−1)/2 deep) — the phase the blocked path shortens.
    for (int64_t c = tid; c < BT; c += blockDim.x) {
      Tf[c * BT + c] = 1.0f;
      if (c < len)
        for (int64_t i = c + 1; i < len; ++i) {
          float sfs = 0.0f;
          for (int64_t m = c; m < i; ++m) sfs += Am[i * BT + m] * Tf[m * BT + c];
          Tf[i * BT + c] = -sfs;
        }
    }
    __syncthreads();
  } else {
    // Blocked tensor-core inverse (mirrors FLA solve_tril.py
    // merge_16x16_to_64x64_inverse_kernel:238-390). BT=64 → four 16×16 diagonal
    // blocks solved by a short (≤16-deep) per-block forward-substitution, then six
    // off-diagonal blocks filled by tensor-core Schur merges (T(i,j) = −T(i,i)·
    // Σ_{j≤k<i} A(i,k)·T(k,j)), scheduled by merge distance so each phase's inputs
    // are ready. Cuts the serial critical path from ~BT-deep (~2016 dep FMAs on one
    // thread) to ~16-deep-per-block + a handful of 16×16 MMAs. Same [BT,BT] f32 Tf
    // result → identical apply/store below. Partial tails (len<BT) fall out for
    // free: Am is pre-zeroed past len (above) so out-of-range blocks invert to I /
    // merge to 0, and the apply masks rows ≥ len. Requires BT == 4·kWM (== 64).
    __shared__ __align__(16) float Pw[3][kWM * kWM];  // per-warp Schur scratch (≤3 concurrent)
    // Diagonal: 4 blocks × 16 columns = 64 independent within-block column solves.
    for (int p = tid; p < 4 * kWM; p += blockDim.x) {
      const int bd = p / kWM;         // diagonal block 0..3
      const int base = bd * kWM;      // block covers rows/cols [base, base+16)
      const int c = base + (p % kWM);  // global column
      Tf[c * BT + c] = 1.0f;
      if (c < len) {
        const int64_t iend = base + kWM < len ? base + kWM : len;
        for (int64_t i = c + 1; i < iend; ++i) {
          float sfs = 0.0f;
          for (int64_t m = c; m < i; ++m) sfs += Am[i * BT + m] * Tf[m * BT + c];
          Tf[i * BT + c] = -sfs;
        }
      }
    }
    __syncthreads();
    // Off-diagonal Schur merges, phased by distance (deps: T(i,j) needs T(k,j),
    // k<i, same column). Distance 1: (1,0)(2,1)(3,2); 2: (2,0)(3,1); 3: (3,0).
    if (warp == 0) WyMerge(Tf, Am, Pw[0], BT, 1, 0);
    else if (warp == 1) WyMerge(Tf, Am, Pw[1], BT, 2, 1);
    else if (warp == 2) WyMerge(Tf, Am, Pw[2], BT, 3, 2);
    __syncthreads();
    if (warp == 0) WyMerge(Tf, Am, Pw[0], BT, 2, 0);
    else if (warp == 1) WyMerge(Tf, Am, Pw[1], BT, 3, 1);
    __syncthreads();
    if (warp == 0) WyMerge(Tf, Am, Pw[0], BT, 3, 0);
    __syncthreads();
  }
  for (int64_t e = tid; e < BT * BT; e += blockDim.x) Store(Tb, e, Tf[e]);  // T(f32) → Tb(TD), Am free
  __syncthreads();

  // Apply on the tensor cores: u = T @ (β⊙v), w = T @ (β⊙exp(G)⊙k). Each pass
  // rebuilds ONE [BT,Dcol] operand in R0 (Tf now free), streams the output in
  // kOBlk-wide column blocks through Osh (store_matrix_sync f32 → TD global).
  for (int p = 0; p < 2; ++p) {
    TD* dst = (p == 0 ? u : w);
    const int64_t dcol = (p == 0 ? dv : dk);
    const int64_t stride = (p == 0 ? hv_n * dv : hv_n * dk);
    const int64_t hoff = (p == 0 ? hv * dv : hv * dk);
    for (int64_t e = tid; e < BT * dcol; e += blockDim.x) {
      const int64_t i = e / dcol, c = e % dcol;
      float val = 0.0f;
      if (i < len)
        val = p == 0 ? bs[i] * Load(v, (tok0 + i) * hv_n * dv + hv * dv + c)
                     : bs[i] * eg[i] * Load(k, (tok0 + i) * hk_n * dk + hk * dk + c);
      Store(Bop, e, val);
    }
    __syncthreads();
    for (int64_t cb = 0; cb < dcol; cb += kOBlk) {
      const int bw = static_cast<int>(dcol - cb < kOBlk ? dcol - cb : kOBlk);  // block width
      const int ntiles = (static_cast<int>(BT) / kWM) * (bw / kWM);
      for (int t = warp; t < ntiles; t += nwarps) {
        const int it = t % (static_cast<int>(BT) / kWM), ct = t / (static_cast<int>(BT) / kWM);
        const int i0 = it * kWM, c0 = ct * kWM;
        typename Cfg::Acc acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int64_t jj = 0; jj < BT; jj += WK) {
          typename Cfg::Arow a;  // T[i0:,jj:]
          typename Cfg::Brow b;  // Bop[jj:, cb+c0:]
          Cfg::load(a, Tb + i0 * BT + jj, BT);
          Cfg::load(b, Bop + jj * dcol + cb + c0, dcol);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(Osh + i0 * bw + c0, acc, bw, wmma::mem_row_major);
      }
      __syncthreads();
      for (int64_t e = tid; e < BT * bw; e += blockDim.x) {
        const int64_t i = e / bw, c = e % bw;
        if (i < len) Store(dst, (tok0 + i) * stride + hoff + cb + c, Osh[e]);
      }
      __syncthreads();
    }
  }
}

// Builds the per-chunk layout (tok0/len) + per-seq base-offset (boh) arrays
// DIRECTLY on the device from the device query_start_loc — the exact loop the
// host used to run after a D2H copy+sync. Single-thread (n_seq is small, ~batch
// size); writes exactly nt_tot entries into tok0/lenv (nt_tot precomputed on the
// host from the same qsl values, so grid/scratch sizing matches bit-for-bit).
// Lets the chunked-prefill path stay device-resident: no D2H sync, no host-vector
// H2D + launch sync (the prefill host-tax; see GdnArgs::query_start_loc_host).
__global__ void GdnBuildChunkMeta(int32_t* tok0, int32_t* lenv, int32_t* boh,
                                  const int32_t* qsl, int64_t n_seq) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int32_t nt = 0;
  for (int64_t n = 0; n < n_seq; ++n) {
    boh[n] = nt;
    const int32_t begin = qsl[n];
    const int32_t len = qsl[n + 1] - begin;
    const int32_t ntn = (len + kChunk - 1) / kChunk;
    for (int32_t it = 0; it < ntn; ++it) {
      const int32_t rem = len - it * kChunk;
      tok0[nt] = begin + it * kChunk;
      lenv[nt] = rem < kChunk ? rem : kChunk;
      ++nt;
    }
  }
}

#ifdef VLLM_CPP_TRITON
// FLA chunk_indices [NT_total, 2]: per GLOBAL chunk index g (grouped by sequence,
// matching chunk_offsets/boh), stores (i_n, i_t_local). The Triton chunk_o / WU
// kernels use it to recover (sequence, local chunk) from program_id, mirroring
// fla/ops/index.py prepare_chunk_indices. Built on device from qsl + boh (both
// ready in either dev_meta or fallback mode) — one thread per sequence.
__global__ void GdnBuildChunkIndices(int32_t* cidx, const int32_t* qsl, const int32_t* boh,
                                     int64_t n_seq) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= n_seq) return;
  const int32_t begin = qsl[n];
  const int32_t len = qsl[n + 1] - begin;
  const int32_t ntn = (len + kChunk - 1) / kChunk;
  const int32_t base = boh[n];
  for (int32_t it = 0; it < ntn; ++it) {
    cidx[(base + it) * 2] = static_cast<int32_t>(n);
    cidx[(base + it) * 2 + 1] = it;
  }
}
#endif  // VLLM_CPP_TRITON

// Slack-only v_new zeroing (default OFF; VT_GDN_SLACK_MEMSET=1 enables). The
// full-buffer cudaMemsetAsync of v_new below zeros [0,t_pad) rows, but DeltaH V1
// writes EVERY real token row [0,t_tot) before any read that needs it, so zeroing
// those is dead (~97% of the buffer at real prefill lengths). Only the per-seq
// last-partial-chunk slack rows the WMMA tiles over-sweep (DeltaH-V2 @ :1268 /
// ChunkO @ :1397 read a full kChunk-row tile per chunk and multiply the slack
// rows by a 0 mask — Kd/As — so 0*NaN=NaN unless those rows are finite) must be
// zeroed. Mirrors FLA masking with boundary_check instead of a memset
// (chunk_delta_h.py:355 v_new=torch.empty_like(u), stored/loaded boundary_check=(0,1)).
bool GdnSlackMemsetEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GDN_SLACK_MEMSET");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Zeros ONLY v_new's per-sequence last-partial-chunk slack rows, exactly the rows
// the WMMA tiles over-read past each sequence's real tokens:
//   [qsl[n]+seqlen_n, qsl[n]+ceil(seqlen_n/kChunk)*kChunk) x [0,dv)
// (tok0 = qsl[n]+it*kChunk tight packing, cf. GdnBuildChunkMeta). One block per
// (sequence, v-head) — grid (n_seq, hv_n), mirroring grid_seq. Idempotent: when a
// slack range overlaps the next sequence's real rows (middle sequences) DeltaH V1
// overwrites them with real finite values anyway; the invariant is only finiteness
// before the masked V2/ChunkO reads. The LAST sequence's slack range covers the
// over-read prefix of the trailing global pad; rows past it (up to t_pad) are
// never read, so they stay uninitialized (as the real-row reads already assume).
// Runs BEFORE DeltaH, same stream ordering as the full memset it replaces.
template <typename TD>
__global__ void GdnChunkVNewSlackZeroKernel(TD* v_new, const int32_t* qsl, int64_t hv_n,
                                            int64_t dv) {
  const int64_t n = blockIdx.x, hv = blockIdx.y;
  const int64_t begin = qsl[n];
  const int64_t seqlen = qsl[n + 1] - begin;
  const int64_t nt = (seqlen + kChunk - 1) / kChunk;
  const int64_t r0 = begin + seqlen;         // first slack row (== qsl[n+1])
  const int64_t r1 = begin + nt * kChunk;    // one past the last over-read row
  const int64_t nrow = r1 - r0;              // == 0 when seqlen is a kChunk multiple
  if (nrow <= 0) return;
  for (int64_t e = threadIdx.x; e < nrow * dv; e += blockDim.x) {
    const int64_t i = e / dv, d = e % dv;
    Store(v_new, (r0 + i) * hv_n * dv + hv * dv + d, 0.0f);
  }
}

#ifdef VLLM_CPP_TRITON
// DEFAULT-ON (2026-07-10) in the VLLM_CPP_TRITON build: gated by token-exact
// test_ops_gdn 31/31 + 27B AND 35B greedy 16/16 through the Triton path in BOTH
// toggle arms, and the measured every-axis win (35B conc64 +2.64% total, TTFT
// −4.1%, TPOT −2.4%, mem equal → 1.0195× vs graphed vLLM = the MVP gate; 27B
// +13pts at conc32). `VT_GDN_*_TRITON=0` opts back into the hand-C++ kernels.
// (Defined BEFORE the first TryTriton* user — main's default-ON commit placed it
// after TryTritonDeltaH, which fails to compile with VLLM_CPP_TRITON=ON.)
static bool GdnTritonEnvOn(const char* n) {
  const char* e = std::getenv(n);
  return e == nullptr || e[0] != '0';
}

// SANCTIONED Triton AOT fast-path dispatch for GDN delta_h (the state recurrence).
// Returns true iff it launched the Triton kernel (caller then skips the hand
// path). Runtime toggle VT_GDN_DELTAH_TRITON: default ON (GdnTritonEnvOn above;
// =0 restores the hand path). Fires ONLY for the exact bf16 gate-model GDN shapes the two
// AOT specializations were pinned to (K=V=128, Hg=16, H in {48,32}); any other
// shape/dtype returns false so the preserved hand-C++ GdnChunkDeltaHRegRingKernel
// (and the CPU reference) still handle it — the portable contract is intact. The
// buffer layout is a verified 1:1 drop-in (strides checked against FLA), so the
// Triton kernel consumes the SAME device buffers our other chunk kernels produced.
bool TryTritonDeltaH(cudaStream_t s, float* state, __nv_bfloat16* hstate, __nv_bfloat16* v_new,
                     const __nv_bfloat16* k, const __nv_bfloat16* u, const __nv_bfloat16* w,
                     const float* gcum, const int32_t* qsl, const int32_t* boh, int64_t hk_n,
                     int64_t dk, int64_t hv_n, int64_t dv, int64_t n_seq, int64_t t_tot) {
  if (!GdnTritonEnvOn("VT_GDN_DELTAH_TRITON")) return false;  // default ON (see GdnTritonEnvOn); =0 restores hand path
  if (dk != 128 || dv != 128 || hk_n != 16) return false;
  if (hv_n != 48 && hv_n != 32) return false;
  auto D = [](const void* p) { return reinterpret_cast<CUdeviceptr>(p); };
  const int32_t T = static_cast<int32_t>(t_tot);      // overwritten per-seq (IS_VARLEN)
  const int32_t NH = static_cast<int32_t>(n_seq * hv_n);  // grid-y = N*H (grid carrier)
  // state serves as BOTH FLA h0 (loaded before the loop) and ht (stored at the end);
  // the full h0 read precedes any ht write, so aliasing the same pointer is safe.
  const CUresult r =
      hv_n == 48
          ? gdn_deltah_h48_default(s, D(k), D(u), D(w), D(v_new), D(gcum), 0, D(hstate), D(state),
                                   D(state), D(qsl), D(boh), T, NH)
          : gdn_deltah_h32_default(s, D(k), D(u), D(w), D(v_new), D(gcum), 0, D(hstate), D(state),
                                   D(state), D(qsl), D(boh), T, NH);
  VT_CHECK(r == CUDA_SUCCESS, "cuda gdn delta_h(triton): launcher returned non-success");
  return true;
}

// SANCTIONED Triton AOT fast-path for GDN chunk_o (the output kernel). Fires ONLY
// at the exact gate-model shape (K=V=128, Hg=16, H in {48,32}) AND with f32 out
// (the default GDN recurrence-output dtype; the AOT spec pins o=*fp32). Runtime
// toggle VT_GDN_CHUNKO_TRITON (opt-in). Consumes the SAME buffers our hand
// GdnChunkOWmmaKernel does (v_new/hstate from delta_h, gcum, q/k, out) — a verified
// 1:1 drop-in — plus chunk_indices (cidx). Returns true iff it launched (caller
// then skips the hand kernel). The hand path + CPU ref remain the fallback.
bool TryTritonChunkO(cudaStream_t s, float* out, const __nv_bfloat16* q, const __nv_bfloat16* k,
                     const __nv_bfloat16* v_new, const __nv_bfloat16* hstate, const float* gcum,
                     const int32_t* qsl, const int32_t* cidx, float scale, int64_t hk_n,
                     int64_t dk, int64_t hv_n, int64_t dv, int64_t nt_tot, int64_t t_tot) {
  if (!GdnTritonEnvOn("VT_GDN_CHUNKO_TRITON")) return false;
  if (dk != 128 || dv != 128 || hk_n != 16) return false;
  if (hv_n != 48 && hv_n != 32) return false;
  if (cidx == nullptr) return false;
  // scale is PINNED to Dk^-0.5 in the kernel (Triton AOT can't take an fp32 scalar
  // arg — it mis-packs it as a double). The gate model always passes Dk^-0.5; bail
  // to the hand path if a caller ever uses a different scale, keeping it exact.
  if (std::fabs(scale - 1.0f / std::sqrt(static_cast<float>(dk))) > 1e-6f) return false;
  auto D = [](const void* p) { return reinterpret_cast<CUdeviceptr>(p); };
  const int32_t T = static_cast<int32_t>(t_tot);   // overwritten per-seq (IS_VARLEN)
  const int32_t NT = static_cast<int32_t>(nt_tot);  // grid-y carrier (= total chunks)
  const CUresult r =
      hv_n == 48
          ? gdn_chunko_h48_default(s, D(q), D(k), D(v_new), D(hstate), D(gcum), D(out), D(qsl),
                                   D(cidx), T, NT)
          : gdn_chunko_h32_default(s, D(q), D(k), D(v_new), D(hstate), D(gcum), D(out), D(qsl),
                                   D(cidx), T, NT);
  VT_CHECK(r == CUDA_SUCCESS, "cuda gdn chunk_o(triton): launcher returned non-success");
  return true;
}

// SANCTIONED Triton AOT fast-path for the GDN WU (WY-representation) pipeline — the
// 3 FLA kernels (chunk_scaled_dot_kkt -> solve_tril -> recompute_w_u) that our
// single fused hand GdnChunkWUWmmaVecKernel mirrors. Fires ONLY at the gate shape.
// Runtime toggle VT_GDN_WU_TRITON (opt-in). Allocates the two intermediate WY
// buffers (A raw f32, Ai bf16) as stream-ordered scratch, runs the 3 kernels, and
// writes u/w (the SAME buffers the hand path fills, consumed downstream by delta_h)
// — a verified 1:1 drop-in. Returns true iff it launched (caller skips the hand WU).
bool TryTritonWU(cudaStream_t s, __nv_bfloat16* u, __nv_bfloat16* w, const __nv_bfloat16* k,
                 const __nv_bfloat16* v, const float* beta, const float* gcum, const int32_t* qsl,
                 const int32_t* cidx, int64_t hk_n, int64_t dk, int64_t hv_n, int64_t dv,
                 int64_t nt_tot, int64_t t_tot) {
  if (!GdnTritonEnvOn("VT_GDN_WU_TRITON")) return false;
  if (dk != 128 || dv != 128 || hk_n != 16) return false;
  if (hv_n != 48 && hv_n != 32) return false;
  if (cidx == nullptr) return false;
  constexpr int BT = kChunk;  // 64
  // WY intermediates [T,H,BT]: A = beta*K*Kᵀ*exp(gΔ) strictly-lower (kkt out, f32);
  // Ai = (I+A)^-1 (solve_tril out, bf16 = k.dtype). Indexed by packed token, so
  // t_tot rows suffice (block-ptr boundary_check clamps the partial-tail slack).
  float* araw = nullptr;
  __nv_bfloat16* ai = nullptr;
  const size_t na = static_cast<size_t>(t_tot) * static_cast<size_t>(hv_n) * BT;
  Check(cudaMallocAsync(&araw, na * sizeof(float), s), "gdn wu(triton) A alloc");
  Check(cudaMallocAsync(&ai, na * sizeof(__nv_bfloat16), s), "gdn wu(triton) Ai alloc");
  // solve_tril writes ONLY the 10 lower-triangular 16×16 blocks of each [BT,BT]
  // inverse; the 6 upper blocks stay 0 (FLA: `Ai = torch.zeros_like(A)`).
  // recompute_w_u then reads the FULL [BT,BT] Ai block and dots it, so the upper
  // triangle MUST be zeroed — cudaMallocAsync returns dirty pool memory in a busy
  // engine (clean in the op test / under memcheck, which is why it slipped there).
  Check(cudaMemsetAsync(ai, 0, na * sizeof(__nv_bfloat16), s), "gdn wu(triton) Ai zero");
  auto D = [](const void* p) { return reinterpret_cast<CUdeviceptr>(p); };
  const int32_t T = static_cast<int32_t>(t_tot);
  const int32_t NT = static_cast<int32_t>(nt_tot);
  CUresult r;
  if (hv_n == 48) {
    r = gdn_kkt_h48_default(s, D(k), D(beta), D(gcum), D(araw), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.kkt(triton): non-success");
    r = gdn_tril_h48_default(s, D(araw), D(ai), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.solve_tril(triton): non-success");
    r = gdn_wu_h48_default(s, D(k), D(v), D(beta), D(w), D(u), D(ai), D(gcum), D(qsl), D(cidx), T,
                           NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.recompute(triton): non-success");
  } else {
    r = gdn_kkt_h32_default(s, D(k), D(beta), D(gcum), D(araw), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.kkt(triton): non-success");
    r = gdn_tril_h32_default(s, D(araw), D(ai), D(qsl), D(cidx), T, NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.solve_tril(triton): non-success");
    r = gdn_wu_h32_default(s, D(k), D(v), D(beta), D(w), D(u), D(ai), D(gcum), D(qsl), D(cidx), T,
                           NT);
    VT_CHECK(r == CUDA_SUCCESS, "cuda gdn wu.recompute(triton): non-success");
  }
  // The hand WMMA delta_h reads full BT-row w/u tiles into the kChunk-row t_pad
  // over-allocation; the Triton recompute (boundary_check) writes only [0,t_tot),
  // so zero the appended slack rows [t_tot, t_tot+kChunk) to keep them finite
  // (0*NaN=NaN on the tensor core) — mirrors the v_new slack invariant. Disjoint
  // from the written rows, so ordering vs the recompute kernel is irrelevant.
  Check(cudaMemsetAsync(u + static_cast<size_t>(t_tot) * hv_n * dv, 0,
                        static_cast<size_t>(kChunk) * hv_n * dv * sizeof(__nv_bfloat16), s),
        "gdn wu(triton) u slack zero");
  Check(cudaMemsetAsync(w + static_cast<size_t>(t_tot) * hv_n * dk, 0,
                        static_cast<size_t>(kChunk) * hv_n * dk * sizeof(__nv_bfloat16), s),
        "gdn wu(triton) w slack zero");
  Check(cudaFreeAsync(araw, s), "gdn wu(triton) A free");
  Check(cudaFreeAsync(ai, s), "gdn wu(triton) Ai free");
  return true;
}
#endif  // VLLM_CPP_TRITON

template <typename Tin, typename Tout>
void LaunchChunkedPrefill(cudaStream_t s, Tensor& out, const Tensor& q_in, const Tensor& k,
                          const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                          const std::vector<int32_t>& tok0h,
                          const std::vector<int32_t>& lenh, const std::vector<int32_t>& bohh,
                          const Tensor& qsl, int64_t n_seq, int64_t nt_tot, bool dev_meta,
                          const GdnArgs& args) {
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
  if (dev_meta) {
    // Device-resident metadata: fill d_tok0/d_len/d_boh on the device from the
    // device qsl (no host-vector H2D, no launch sync — the caller does not sync).
    // tok0h/lenh/bohh are empty here; nt_tot was precomputed on the host from the
    // same qsl values, so grid_chunk.x + the hstate size match bit-for-bit.
    GdnBuildChunkMeta<<<1, 1, 0, s>>>(d_tok0, d_len, d_boh, qsl.Ptr<int32_t>(), n_seq);
    Check(cudaGetLastError(), "gdn chunked build-meta launch");
  } else {
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
  }

  // FLA chunk_indices [nt_tot,2] for the Triton chunk_o / WU fast-paths. Allocated
  // + built on device ONLY when a Triton chunk path is toggled on at the gate shape
  // (else it stays null and the hand kernels — which don't need it — run): the OFF
  // build never compiles this and an ON build with the toggles off is byte-inert.
#ifdef VLLM_CPP_TRITON
  int32_t* d_cidx = nullptr;
  const bool want_cidx =
      std::is_same<TSc, __nv_bfloat16>::value && dk == 128 && dv == 128 && hk_n == 16 &&
      (hv_n == 48 || hv_n == 32) &&
      (GdnTritonEnvOn("VT_GDN_CHUNKO_TRITON") || GdnTritonEnvOn("VT_GDN_WU_TRITON"));
  if (want_cidx) {
    Check(cudaMallocAsync(&d_cidx, static_cast<size_t>(2 * nt_tot) * sizeof(int32_t), s),
          "gdn chunked cidx alloc");
    const unsigned nb = static_cast<unsigned>((n_seq + 31) / 32);
    GdnBuildChunkIndices<<<nb, 32, 0, s>>>(d_cidx, qsl.Ptr<int32_t>(), d_boh, n_seq);
    Check(cudaGetLastError(), "gdn chunked cidx build launch");
  }
#endif

  const dim3 grid_chunk(static_cast<unsigned>(nt_tot), static_cast<unsigned>(hv_n));
  const dim3 grid_seq(static_cast<unsigned>(n_seq), static_cast<unsigned>(hv_n));

  GdnChunkCumsumKernel<<<grid_chunk, 32, 0, s>>>(gcum, g.Ptr<float>(), d_tok0, d_len, hv_n);
  Check(cudaGetLastError(), "gdn chunked cumsum launch");

  auto opt_in = [](void* kernel, size_t bytes, const char* what) {
    if (bytes > 48 * 1024)
      Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(bytes)),
            what);
  };

  // Step B (WY u/w). WMMA tensor-core Gram when the WMMA path is active (same
  // dim gate as DeltaH/O); A/B: VT_GDN_WU_WMMA=0 forces the scalar-float kernel
  // (also the f32-corner fallback and the unit-test scalar reference path).
  const char* wu_env = std::getenv("VT_GDN_WU_WMMA");
  const bool wu_wmma = wmma && (wu_env == nullptr || wu_env[0] != '0');
  // VEC path (default): replace the per-column WY forward-substitution (measured
  // ~73% of the scalar-solve WU kernel) with a single [BT,BT] triangular inverse
  // + tensor-core apply (GdnChunkWUWmmaVecKernel, mirrors FLA solve_tril+
  // recompute_w_u). VT_GDN_CHUNK_VEC=0 keeps the WMMA-Gram/scalar-solve kernel.
  const char* vec_env = std::getenv("VT_GDN_CHUNK_VEC");
  // VT_GDN_WY_BLOCKED (DEFAULT ON at the gate shape; =0 restores serial-f32): swap ONLY the vec kernel's triangular-inverse
  // phase for the FLA blocked tensor-core inverse (four 16×16 diagonal forward-subs
  // + six 16×16 Schur merges) instead of the serial ~BT-deep column solve. Requires
  // BT == kChunk == 4·kWM == 64 (the merge_16x16_to_64x64 block structure). Greedy 16/16-vs-oracle
  // token-exact on BOTH gate models (35B single+batched, 27B tie-free prefix); the tf32
  // WMMA Schur merges hold vs vLLM's ieee inverse. test_ops_gdn 423/423.
  const char* blk_env = std::getenv("VT_GDN_WY_BLOCKED");
  const bool wy_blocked = kChunk == 4 * kWM && (blk_env == nullptr || blk_env[0] != '0');
  // Compact aliased arena (see kernel): R0=max(Ks[BT,Dk], Tf[BT,BT]f32),
  // R1=max(Am[BT,BT]f32, Tb[BT,BT]TD + Osh[BT,kOBlk]f32). bf16 ~40 KiB (2 blocks/SM),
  // f32/TF32 ~64 KiB (1 block/SM). Both fit the GB10 99 KiB opt-in; the guard keeps
  // the fallback safe on devices/corners with a smaller opt-in budget.
  const size_t r0 = std::max(static_cast<size_t>(kChunk * dk) * sizeof(TSc),
                             static_cast<size_t>(kChunk * kChunk) * sizeof(float));
  const size_t r1 = std::max(static_cast<size_t>(kChunk * kChunk) * sizeof(float),
                             static_cast<size_t>(kChunk * kChunk) * sizeof(TSc) +
                                 static_cast<size_t>(kChunk * kOBlk) * sizeof(float));
  const size_t vec_bytes = r0 + r1;
  int smem_optin = 48 * 1024;
  cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  const bool wu_vec = wu_wmma && (vec_env == nullptr || vec_env[0] != '0') &&
                      vec_bytes <= static_cast<size_t>(smem_optin);
  // SANCTIONED Triton AOT WU fast-path (bf16 gate shape only, opt-in via
  // VT_GDN_WU_TRITON=1). If it fires it produces u/w via the FLA 3-kernel WY
  // pipeline into the SAME buffers and we skip the hand WU kernels; otherwise the
  // preserved hand path below runs unchanged (portable contract intact).
  bool triton_wu = false;
#ifdef VLLM_CPP_TRITON
  if constexpr (std::is_same<TSc, __nv_bfloat16>::value) {
    triton_wu = TryTritonWU(s, u, w, k.Ptr<Tin>(), v.Ptr<Tin>(), beta.Ptr<float>(), gcum,
                            qsl.Ptr<int32_t>(), d_cidx, hk_n, dk, hv_n, dv, nt_tot, t_tot);
  }
#endif
  if (triton_wu) {
    // WU (u,w) already produced by the Triton WY pipeline; skip the hand kernels.
  } else if (wu_vec) {
    Check(cudaFuncSetAttribute(reinterpret_cast<void*>(GdnChunkWUWmmaVecKernel<TSc>),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(vec_bytes)),
          "gdn chunked wu(vec) shared opt-in");
    GdnChunkWUWmmaVecKernel<TSc><<<grid_chunk, 256, vec_bytes, s>>>(
        u, w, k.Ptr<Tin>(), v.Ptr<Tin>(), beta.Ptr<float>(), gcum, d_tok0, d_len, hk_n, dk, hv_n,
        dv, wy_blocked);
    Check(cudaGetLastError(), "gdn chunked wu(vec) launch");
  } else if (wu_wmma) {
    const size_t wu_bytes = static_cast<size_t>(kChunk * dk) * sizeof(TSc) +
                            static_cast<size_t>(kChunk * kChunk) * sizeof(float);
    // Always opt in: the f32/TF32 dynamic request is exactly 48 KiB, which plus
    // the static gs/bs/eg overflows the 48 KiB default (opt_in's ">" guard would
    // miss the boundary). Setting the max-dynamic attribute unconditionally is a
    // no-op when the request already fits.
    Check(cudaFuncSetAttribute(reinterpret_cast<void*>(GdnChunkWUWmmaKernel<TSc>),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(wu_bytes)),
          "gdn chunked wu(wmma) shared opt-in");
    GdnChunkWUWmmaKernel<TSc><<<grid_chunk, 256, wu_bytes, s>>>(
        u, w, k.Ptr<Tin>(), v.Ptr<Tin>(), beta.Ptr<float>(), gcum, d_tok0, d_len, hk_n, dk, hv_n,
        dv);
    Check(cudaGetLastError(), "gdn chunked wu(wmma) launch");
  } else {
    GdnChunkWUKernel<Tin, TSc><<<grid_chunk, 128, 0, s>>>(u, w, k.Ptr<Tin>(), v.Ptr<Tin>(),
                                                          beta.Ptr<float>(), gcum, d_tok0, d_len,
                                                          hk_n, dk, hv_n, dv);
    Check(cudaGetLastError(), "gdn chunked wu launch");
  }

  if (wmma) {
    // WMMA tensor-core path (TF32 for f32 in, native bf16 for bf16 in).
    // Occupancy: DeltaH and O are pinned to 1 resident block per SM by their
    // ~96 KiB dynamic shared (GB10: 100 KiB/SM). At the legacy 256-thread
    // (8-warp) block that single block realizes only 8/48 = 16.7% of the SM's
    // peak warp slots, while their registers (DeltaH 80, O 74 per thread) would
    // admit far more resident warps within that one block. Both kernels' tensor-
    // core tile loops are already warp-strided (t = warp; t < ntiles; t += nwarps)
    // and every elementwise/staging loop strides by blockDim.x, so the block
    // simply scales with width — more warps in the one resident block hide the
    // WMMA + global-load latency the 8-warp block could not (mirrors the
    // GdnDecodeFused 1-warp -> multi-warp occupancy win). Measured GB10 optima
    // (35B f32 dims, T=2048): DeltaH is bottlenecked by its 64-tile V2 rank
    // update and keeps scaling to 768 threads (24 warps, 50% occ, -32% per call:
    // 2.70 -> 1.84 ms); its 80 regs/thread cap the block at 65 536/80 = 819, so
    // 768 is the top warp-multiple that still fits 1 block/SM. O saturates at 512
    // threads (16 warps, 33% occ, -25%: 1.65 -> 1.23 ms) — its smaller 16/32-tile
    // phases leave the extra 768-block warps idle, a hair slower. Both stay
    // 1 block/SM (shared-bound). A/B: VT_GDN_OCC_BLOCK forces BOTH to a fixed
    // thread count (=256 recovers the pre-occupancy 8-warp path).
    unsigned occ_delta = 768u, occ_o = 512u;
    if (const char* e = std::getenv("VT_GDN_OCC_BLOCK")) {
      const int v = std::atoi(e);
      if (v >= 32 && v <= 1024) occ_delta = occ_o = (static_cast<unsigned>(v) / 32u) * 32u;
    }
    // The f32/TF32 DeltaH instantiation is register-heavier (96 vs the bf16 path's
    // 70); regs×threads must fit the 64K/SM register file, so 96×768 would exceed
    // it and fail to launch. Cap the f32 block at 640 (still 20 warps). f32 is the
    // correctness-only corner (the benchmarked 35B GDN is bf16), so the marginal
    // occupancy drop is immaterial; the bf16 hot path keeps its tuned 768.
    if (sizeof(TSc) == 4 && occ_delta > 640u) occ_delta = 640u;
    // 128-bit (int4/float4) staging for the DeltaH/ChunkO gmem+smem copy loops
    // (measured ~32%/13% of those kernels): address-once + one coalesced 16-byte
    // transfer per 8 bf16 / 4 f32, no per-element bf16↔f32 round-trip. Values are
    // bit-identical (same RNE convert), so chunked==sequential is unchanged. A/B:
    // VT_GDN_DELTAH_VEC=0 restores the scalar per-element staging in the same binary.
    const char* dvec_env = std::getenv("VT_GDN_DELTAH_VEC");
    const bool dvec = (dvec_env == nullptr || dvec_env[0] != '0');
    // Zero v_new so the over-allocated tail rows the matmuls may sweep
    // (their A operand is 0 there, but 0*NaN=NaN on the tensor core) are finite.
    // VT_GDN_SLACK_MEMSET zeros ONLY the per-seq partial-chunk slack rows (the
    // ~3% actually over-read); default OFF keeps the byte-identical full memset.
    if (GdnSlackMemsetEnabled()) {
      GdnChunkVNewSlackZeroKernel<TSc><<<grid_seq, 128, 0, s>>>(v_new, qsl.Ptr<int32_t>(), hv_n,
                                                                dv);
      Check(cudaGetLastError(), "gdn chunked v_new slack-zero launch");
    } else {
      Check(cudaMemsetAsync(v_new, 0, static_cast<size_t>(t_pad * hv_n * dv) * sizeof(TSc), s),
            "gdn chunked v_new zero");
    }
    const size_t sz = sizeof(TSc);
    const size_t v1 = static_cast<size_t>(kNB * dk) * sz + static_cast<size_t>(kChunk * kNB) * 4;
    const size_t v2 = static_cast<size_t>(kChunk * dk) * sz;
    const size_t delta_bytes = static_cast<size_t>(dv * dk) * sizeof(float) + (v1 > v2 ? v1 : v2);
    // REGISTER-tiled delta_h (vt::tile Rung-1): the running state H lives in WMMA
    // ACCUMULATOR REGISTERS across the chunk loop (FLA blockdim64's b_h1/b_h2), NOT
    // in 64 KiB of shared memory — the CUTLASS-style persistent-accumulator GEMM
    // that frees smem for the cp.async ring below. DEFAULT ON at the gate shape
    // (dk==128, dv%BV==0, bf16); VT_GDN_TILE_PIPE=0 restores the baseline
    // GdnChunkDeltaHWmmaKernel for A/B. BV=64 (grid.x=dv/BV, 4 warps/block).
    // MEASURED (35B NVFP4, in1024/out8, GB10): delta_h 441us baseline ->
    // 398us reg-no-ring -> 345us reg+ring (0.78x); e2e in1024/out128 conc32 np192
    // +0.85% total & prefill, TTFT/TPOT lower. Token-exact (test_ops_gdn 423/423).
    // SANCTIONED Triton AOT fast-path (bf16 gate shapes only, opt-in via
    // VT_GDN_DELTAH_TRITON=1). If it fires it launches delta_h into the SAME
    // buffers and we skip the hand kernels; otherwise the preserved hand-C++
    // path below runs unchanged (portable contract intact).
    bool triton_deltah = false;
#ifdef VLLM_CPP_TRITON
    if constexpr (std::is_same<TSc, __nv_bfloat16>::value) {
      triton_deltah =
          TryTritonDeltaH(s, state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum,
                          qsl.Ptr<int32_t>(), d_boh, hk_n, dk, hv_n, dv, n_seq, t_tot);
    }
#endif
    const char* tp_env = std::getenv("VT_GDN_TILE_PIPE");
    const bool tile_pipe = (tp_env == nullptr || tp_env[0] != '0') && dk == 128 && (dv % 64 == 0);
    if (triton_deltah) {
      // delta_h already launched by the Triton fast-path; skip the hand kernels.
    } else if (tile_pipe) {
      constexpr int BV = 64;
      const int64_t rA0 = static_cast<int64_t>(BV * dk) * sizeof(float);
      const int64_t rA1 = static_cast<int64_t>(kChunk * BV) * sizeof(float);
      const int64_t rA2 = static_cast<int64_t>(kChunk * dk) * sz;
      int64_t regA = rA0 > rA1 ? rA0 : rA1;
      regA = regA > rA2 ? regA : rA2;
      const int64_t rB0 = static_cast<int64_t>(BV * dk) * sz;
      const int64_t rB1 = static_cast<int64_t>(kChunk * BV) * sz;
      const size_t reg_bytes = static_cast<size_t>(regA + (rB0 > rB1 ? rB0 : rB1));
      const dim3 grid_reg(static_cast<unsigned>(dv / BV),
                          static_cast<unsigned>(n_seq * hv_n));
      const unsigned reg_block = (BV / kWM) * 32;
      // STEP 2 — cp.async ring (VT_GDN_TILE_PIPE_CPASYNC, default ON for bf16):
      // 2-stage software pipeline over the streamed W/K tiles. bf16 only (the f32
      // ring tiles overflow the 99 KiB opt-in); f32 uses the ring-less reg kernel.
      // The ring runs at 1 block/SM (its smem halves occupancy vs reg-no-ring's 2
      // blocks), but the prefetch hides the load-stalls that low occupancy exposes,
      // netting the 398->345us win. VT_GDN_TILE_PIPE_CPASYNC=0 isolates reg-no-ring.
      const char* ca_env = std::getenv("VT_GDN_TILE_PIPE_CPASYNC");
      const bool use_ring = (ca_env == nullptr || ca_env[0] != '0');
      // STEP 3 (Rung-2) — TMA + mbarrier ring (VT_GDN_TMA, default OFF). Same
      // register-tiled WMMA compute + smem layout as the cp.async ring; only the
      // W/K streaming swaps sm80 cp.async for TMA bulk-tensor copies signalled by
      // per-stage mbarriers (the Blackwell-native pipeline). A/B isolates the copy
      // mechanism. bf16 only; falls back to the cp.async ring if TMA desc setup fails.
      const char* tma_env = std::getenv("VT_GDN_TMA");
      const bool use_tma = (tma_env != nullptr && tma_env[0] == '1');
      bool launched = false;
      if constexpr (std::is_same<TSc, __nv_bfloat16>::value) {
        constexpr int STAGES = 2;
        const int64_t rAr = static_cast<int64_t>(BV * 64) * sizeof(float);  // Hf32(half)/Braw f32
        const int64_t rBr = static_cast<int64_t>(BV * dk) * sz;             // Hbf full (TD)
        const int64_t tile = static_cast<int64_t>(kChunk * dk) * sz;
        const size_t ring_bytes = static_cast<size_t>(rAr + rBr + (int64_t)STAGES * 2 * tile);
        if (use_tma) {
          CUtensorMap descW{}, descK{};
          const bool okW = BuildGdnTmaDesc3D(&descW, w, t_pad, hv_n, dk, kChunk);
          const bool okK = BuildGdnTmaDesc3D(&descK, k.Ptr<Tin>(), t_tot, hk_n, dk, kChunk);
          if (okW && okK) {
            Check(cudaFuncSetAttribute(
                      reinterpret_cast<void*>(GdnChunkDeltaHTmaKernel<TSc, BV, STAGES>),
                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(ring_bytes)),
                  "gdn chunked delta_h(tma) shared opt-in");
            GdnChunkDeltaHTmaKernel<TSc, BV, STAGES><<<grid_reg, reg_block, ring_bytes, s>>>(
                state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(),
                d_boh, hk_n, dk, hv_n, dv, descW, descK);
            Check(cudaGetLastError(), "gdn chunked delta_h(tma) launch");
            launched = true;
          }
        }
        if (!launched && use_ring) {
          Check(cudaFuncSetAttribute(
                    reinterpret_cast<void*>(GdnChunkDeltaHRegRingKernel<TSc, BV, STAGES>),
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(ring_bytes)),
                "gdn chunked delta_h(reg-ring) shared opt-in");
          GdnChunkDeltaHRegRingKernel<TSc, BV, STAGES><<<grid_reg, reg_block, ring_bytes, s>>>(
              state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(),
              d_boh, hk_n, dk, hv_n, dv);
          Check(cudaGetLastError(), "gdn chunked delta_h(reg-ring) launch");
          launched = true;
        }
      }
      if (!launched) {
        // Always opt in: the bf16 request is exactly 48 KiB, which the default
        // 48896-byte cap rejects (opt_in's ">" guard would miss the boundary).
        Check(cudaFuncSetAttribute(reinterpret_cast<void*>(GdnChunkDeltaHRegKernel<TSc, BV>),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(reg_bytes)),
              "gdn chunked delta_h(reg) shared opt-in");
        GdnChunkDeltaHRegKernel<TSc, BV><<<grid_reg, reg_block, reg_bytes, s>>>(
            state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(),
            d_boh, hk_n, dk, hv_n, dv);
        Check(cudaGetLastError(), "gdn chunked delta_h(reg) launch");
      }
    } else {
      opt_in(reinterpret_cast<void*>(GdnChunkDeltaHWmmaKernel<TSc>), delta_bytes,
             "gdn chunked delta_h(wmma) shared opt-in");
      GdnChunkDeltaHWmmaKernel<TSc><<<grid_seq, occ_delta, delta_bytes, s>>>(
          state.Ptr<float>(), hstate, v_new, k.Ptr<Tin>(), u, w, gcum, qsl.Ptr<int32_t>(), d_boh,
          hk_n, dk, hv_n, dv, dvec);
      Check(cudaGetLastError(), "gdn chunked delta_h(wmma) launch");
    }

    // SANCTIONED Triton AOT chunk_o fast-path (bf16 gate shape + f32 out only,
    // opt-in via VT_GDN_CHUNKO_TRITON=1). If it fires it writes `out` directly and
    // we skip the hand kernel; otherwise the preserved hand path runs unchanged.
    bool triton_chunko = false;
#ifdef VLLM_CPP_TRITON
    if constexpr (std::is_same<TSc, __nv_bfloat16>::value && std::is_same<Tout, float>::value) {
      triton_chunko = TryTritonChunkO(s, out.Ptr<float>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v_new,
                                      hstate, gcum, qsl.Ptr<int32_t>(), d_cidx, args.scale, hk_n,
                                      dk, hv_n, dv, nt_tot, t_tot);
    }
#endif
    if (!triton_chunko) {
      const size_t r2 = static_cast<size_t>(kChunk * dk) * sz > static_cast<size_t>(kChunk * dv) * 4
                            ? static_cast<size_t>(kChunk * dk) * sz
                            : static_cast<size_t>(kChunk * dv) * 4;
      const size_t chunko_bytes = static_cast<size_t>(kChunk * dk) * sz + r2 +
                                  static_cast<size_t>(kChunk * kChunk) * 4 +
                                  static_cast<size_t>(kChunk * kChunk) * sz;
      opt_in(reinterpret_cast<void*>(GdnChunkOWmmaKernel<TSc, Tout>), chunko_bytes,
             "gdn chunked o(wmma) shared opt-in");
      GdnChunkOWmmaKernel<TSc, Tout><<<grid_chunk, occ_o, chunko_bytes, s>>>(
          out.Ptr<Tout>(), q_in.Ptr<Tin>(), k.Ptr<Tin>(), v_new, hstate, gcum, d_tok0, d_len,
          hk_n, dk, hv_n, dv, args.scale, dvec);
      Check(cudaGetLastError(), "gdn chunked o(wmma) launch");
    }
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
#ifdef VLLM_CPP_TRITON
  if (d_cidx != nullptr) Check(cudaFreeAsync(d_cidx, s), "gdn chunked cidx free");
#endif
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

  // Chunk layout from query_start_loc. FAST PATH (dev_meta): the caller handed us
  // the host-resident qsl (args.query_start_loc_host) — the SAME values already
  // materialized on the host by the GDN attention-metadata build — so we read
  // them directly (no D2H copy, no cudaStreamSynchronize) and let a device kernel
  // fill the per-chunk arrays. This kills the per-GDN-layer host↔GPU lockstep that
  // left the prefill ~67% GPU-idle. FALLBACK (host qsl null: op tests / callers
  // without it): the original D2H copy + sync to read qsl on the host.
  const bool dev_meta = args.query_start_loc_host != nullptr;
  std::vector<int32_t> qslh(static_cast<size_t>(n_seq) + 1);
  if (dev_meta) {
    for (int64_t i = 0; i <= n_seq; ++i) qslh[static_cast<size_t>(i)] = args.query_start_loc_host[i];
  } else {
    Check(cudaMemcpyAsync(qslh.data(), qsl.Ptr<int32_t>(),
                          (static_cast<size_t>(n_seq) + 1) * sizeof(int32_t),
                          cudaMemcpyDeviceToHost, s),
          "gdn chunked qsl download");
    Check(cudaStreamSynchronize(s), "gdn chunked qsl sync");
  }

  // nt_tot (total chunks) is always computed on the host — it sizes the grids and
  // the hstate scratch. In dev_meta mode the GdnBuildChunkMeta device kernel
  // recomputes the identical layout from the same qsl values, so nt_tot matches
  // bit-for-bit; the host tok0h/lenh/bohh vectors are then left empty (unused).
  std::vector<int32_t> tok0h, lenh, bohh;
  int32_t nt_tot = 0;
  for (int64_t n = 0; n < n_seq; ++n) {
    const int32_t begin = qslh[static_cast<size_t>(n)];
    const int32_t len = qslh[static_cast<size_t>(n) + 1] - begin;
    const int32_t nt = (len + kChunk - 1) / kChunk;
    if (!dev_meta) {
      if (bohh.empty()) bohh.resize(static_cast<size_t>(n_seq));
      bohh[static_cast<size_t>(n)] = nt_tot;
      for (int32_t it = 0; it < nt; ++it) {
        const int32_t rem = len - it * kChunk;
        tok0h.push_back(begin + it * kChunk);
        lenh.push_back(rem < kChunk ? rem : kChunk);
      }
    }
    nt_tot += nt;
  }
  if (nt_tot == 0) return;

  if (q_in.dtype == DType::kF32) {
    if (out.dtype == DType::kF32)
      LaunchChunkedPrefill<float, float>(s, out, q_in, k, v, g, beta, state, tok0h, lenh, bohh,
                                         qsl, n_seq, nt_tot, dev_meta, args);
    else
      LaunchChunkedPrefill<float, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, tok0h, lenh,
                                                 bohh, qsl, n_seq, nt_tot, dev_meta, args);
  } else {
    if (out.dtype == DType::kF32)
      LaunchChunkedPrefill<__nv_bfloat16, float>(s, out, q_in, k, v, g, beta, state, tok0h, lenh,
                                                 bohh, qsl, n_seq, nt_tot, dev_meta, args);
    else
      LaunchChunkedPrefill<__nv_bfloat16, __nv_bfloat16>(s, out, q_in, k, v, g, beta, state, tok0h,
                                                         lenh, bohh, qsl, n_seq, nt_tot, dev_meta,
                                                         args);
  }
  // FALLBACK only: keep the host index vectors alive until the async uploads
  // finish. dev_meta builds the arrays on-device (no host vectors) so it does NOT
  // sync here — the whole point (host runs ahead, GPU stays fed across layers).
  if (!dev_meta) Check(cudaStreamSynchronize(s), "gdn chunked launch sync");
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
    RegisterOp(OpId::kGdnPostConv, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernelCuda)));
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
