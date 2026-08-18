// W0 register-pressure probe for row KERNEL-GDN-REPLAYSSM (issue #1171),
// spec .agents/specs/gdn-replayssm.md, work item W0 / risk R1.
//
// THIS IS NOT PRODUCT CODE AND NOTHING LINKS IT. It is the evidence artifact for
// one question the spec asks before any implementation starts: does the buffered
// ReplaySSM decode step, at the 27B [BV=32, BK=128] tile, spill registers as
// hand-written CUDA? src/vt/cuda/gdn_packed_decode_triton.h:9-14 records the
// measured pair the answer used to be read against -- the register-resident
// [BV=32,BK=128] fp32 state tile is REG:205 with zero spill under Triton/ptxas
// and REG:255 + STACK:48 (spilling) as hand CUDA.
//
// THE FIRST VERSION OF THIS PROBE ASKED THE WRONG QUESTION, AND THIS FILE
// RECORDS BOTH. It compiled one buffered kernel that declared `float sh[BK]`
// with BK == 128 and kept that array live across the whole step, including the
// flush branch. vLLM does the opposite on purpose. At the pin
// (vllm 555967922) the decayed checkpoint readout is "streamed over NF dstate
// tiles so the (M, N) state slice is never held whole"
// (selective_state_update_replayssm_output_only.py:309-311), the flush route is
// "streamed over FL dstate tiles" (:361, :395), the two routes use DISTINCT tile
// locals because differing tile widths "would force a shape-mismatched merge at
// the if/else exit" (:391-393), and the tuned Blackwell config at dstate == 128
// is nf_dstate_tile == 32 and fl_dstate_tile == 64, never 128
// (replayssm_config.py:47-52). A 128-float resident array is therefore a
// structure the mirror source deliberately avoids, on the exact axis being
// measured, so its spill said nothing about ReplaySSM.
//
// It also matters WHICH route spills. The non-flush route runs on 15 of every
// 16 steps and carries 100% of the claimed bandwidth saving. In this algebra it
// computes only dot_q += S_0[c]*q[c] and dot_k += S_0[c]*k[c] -- each state
// element is touched twice in one pass, so two scalar accumulators and a
// streaming load suffice and no register array is needed at all. That route was
// never measured on its own.
//
// So this file compiles SIX kernels in ONE translation unit, so that every
// number comes from one ptxas invocation and the two original kernels stay
// comparable to the first run:
//
//   A) ProbeControlRegTileKernel -- a verbatim transcription of the shipped
//      GdnPackedDecodeRegTileKernel (src/vt/cuda/cuda_gdn.cu:2516-2612). This is
//      the kernel the recorded REG:255 + STACK:48 was measured on, so its number
//      under THIS nvcc is the control. Without it the probe numbers float
//      against a figure taken on another toolkit on 2026-07-16.
//
//   B) ProbeReplaySsmRegTileKernel -- KEPT VERBATIM from the first run: the
//      buffered decode loop with the resident `float sh[BK=128]` array reused
//      across both routes. It is retained only so the new numbers are read
//      against the old ones under the same ptxas, not because it is the shape
//      ReplaySSM would be written in.
//
//   C) ProbeReplaySsmNonFlushKernel -- the 15-of-16 route ALONE, with NO
//      register array: the checkpoint is streamed and reduced into two scalars,
//      then the ring loop and the ring append. This is the kernel whose spill
//      or non-spill actually decides the row.
//
//   D) ProbeReplaySsmFlushKernel -- the 1-of-16 route ALONE, streaming the dk
//      dimension in FL_DSTATE_TILE == 64 tiles that match the pin's tuned
//      Blackwell config, re-loading the checkpoint from HBM once per tile.
//      That re-load is semantically identical to the resident-array form: in
//      the resident form `sh[c] * total_decay` reads the pre-decay checkpoint,
//      and the only write to `state` is the final Store after the ring fold, so
//      HBM still holds exactly those values when the tile is re-read. Tiles are
//      disjoint in c, so a tile's store cannot be seen by a later tile's load.
//
//   E) ProbeReplaySsmFlushSeqKernel -- D with the outer dstate-tile loop pinned
//      to `#pragma unroll 1`, so that a spill in D can be attributed to tile
//      liveness rather than to the tiling itself.
//
//   F) ProbeReplaySsmFusedKernel -- both routes in one kernel, transcribing the
//      pin's structure: NF_DSTATE_TILE == 32, FL_DSTATE_TILE == 64, and DISJOINT
//      per-branch tile locals, exactly as :391-393 requires.
//
// The numeric semantics are the shipped kernel's, re-associated exactly as the
// spec's "## Design" derives: with G' = exp(g_t + sum_j g_j) and
// W'_j = exp(g_t + sum_{i>j} g_i),
//
//     (g_t S_{t-1}) k_t = G' (S_0 k_t) + sum_j W'_j d_j (k_j . k_t)
//     y_t              = G' (S_0 q_t) + sum_j W'_j d_j (k_j . q_t)
//                        + d_t (k_t . q_t)
//
// The probe is not gated for numerics -- it is compiled, not run. W0 asks one
// question and this file answers only that one. NOTHING HERE IS A SPEED NUMBER.
//
// Build. build_w0.sh beside this file is the script that was actually run,
// run_on_lease.sh submits it to a leased fleet device, and job1.log / job2.log
// are the raw output of two independent runs of it:
//   nvcc -std=c++20 -O3 -arch=sm_121a -Xptxas -v -c probe.cu -o probe.o
//   cuobjdump -res-usage probe.o
//   cuobjdump -sass probe.o        # LDL/STL placement against the flush branch

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// Minimal stubs, byte-faithful to src/vt/cuda/cuda_gdn.cu:268-297. Copied rather
// than included because the real header pulls the whole vt runtime in, and a
// register count must be read off the kernel, not off a build system.
// ---------------------------------------------------------------------------
enum class DType { kF32, kF16, kBF16 };

__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __half* p, int64_t i) {
  return __half2float(p[i]);
}
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline float LoadFloating(const void* p, DType dtype, int64_t i) {
  if (dtype == DType::kF32) return Load(static_cast<const float*>(p), i);
  if (dtype == DType::kF16) return Load(static_cast<const __half*>(p), i);
  return Load(static_cast<const __nv_bfloat16*>(p), i);
}
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__half* p, int64_t i, float v) {
  p[i] = __float2half_rn(v);
}
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

template <typename T>
__device__ inline float RoundToStorage(float v);
template <>
__device__ inline float RoundToStorage<float>(float v) {
  return v;
}
template <>
__device__ inline float RoundToStorage<__half>(float v) {
  return __half2float(__float2half_rn(v));
}
template <>
__device__ inline float RoundToStorage<__nv_bfloat16>(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

}  // namespace

// ---------------------------------------------------------------------------
// A) CONTROL -- verbatim src/vt/cuda/cuda_gdn.cu:2516-2612
//    (GdnPackedDecodeRegTileKernel). Recorded as REG:255 + STACK:48 on dgx
//    phase1 2026-07-16.
// ---------------------------------------------------------------------------
template <typename T, typename TState, int BK>
__global__ void ProbeControlRegTileKernel(
    T* out, const T* mixed_qkv, const T* a, const T* b, const void* a_log,
    DType a_log_dtype, const void* dt_bias, DType dt_bias_dtype, TState* state,
    const int32_t* state_idx, int64_t state_slots, int64_t mixed_stride,
    int64_t a_stride, int64_t b_stride, int64_t hk_n, int64_t dk, int64_t hv_n,
    int64_t dv, int64_t bv, float scale) {
  const int64_t i_v = blockIdx.x;
  const int64_t i_nh = blockIdx.y;
  const int64_t i_n = i_nh / hv_n;
  const int64_t hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t vbase = i_v * bv;
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t vrow = vbase + lane;

  const int32_t slot = state_idx[i_n];
  if (slot < 0 || slot >= state_slots) {
    if (vrow < dv) Store(out, (i_n * hv_n + hv) * dv + vrow, 0.0f);
    return;
  }

  __shared__ float bq[BK];
  __shared__ float bk[BK];
  const int nthreads = static_cast<int>(blockDim.x);
  const int64_t mixed_row = i_n * mixed_stride;
  const int64_t qbase = mixed_row + hk * dk;
  const int64_t kbase = mixed_row + hk_n * dk + hk * dk;
  float q_sumsq = 0.0f;
  float k_sumsq = 0.0f;
  for (int64_t ki = lane; ki < dk; ki += nthreads) {
    const float qv = Load(mixed_qkv, qbase + ki);
    const float kv = Load(mixed_qkv, kbase + ki);
    bq[ki] = qv;
    bk[ki] = kv;
    q_sumsq += qv * qv;
    k_sumsq += kv * kv;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    q_sumsq += __shfl_xor_sync(0xffffffffu, q_sumsq, off);
    k_sumsq += __shfl_xor_sync(0xffffffffu, k_sumsq, off);
  }
  const float q_scaled_inv = (1.0f / sqrtf(q_sumsq + 1e-6f)) * scale;
  const float k_inv = 1.0f / sqrtf(k_sumsq + 1e-6f);
  for (int64_t ki = lane; ki < dk; ki += nthreads) {
    bq[ki] *= q_scaled_inv;
    bk[ki] *= k_inv;
  }
  __syncwarp();

  const float av = Load(a, i_n * a_stride + hv);
  const float b_raw = Load(b, i_n * b_stride + hv);
  const float x = av + LoadFloating(dt_bias, dt_bias_dtype, hv);
  const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
  const float g = -expf(LoadFloating(a_log, a_log_dtype, hv)) * softplus;
  const float decay = expf(g);
  const float beta = RoundToStorage<T>(1.0f / (1.0f + expf(-b_raw)));

  if (vrow >= dv) return;

  const int64_t s_row =
      ((static_cast<int64_t>(slot) * hv_n + hv) * dv + vrow) * dk;
  float sh[BK];
#pragma unroll
  for (int c = 0; c < BK; ++c) sh[c] = Load(state, s_row + c);
  float dot = 0.0f;
#pragma unroll
  for (int c = 0; c < BK; ++c) {
    sh[c] *= decay;
    dot += sh[c] * bk[c];
  }
  const int64_t v_offset = mixed_row + 2 * hk_n * dk + hv * dv + vrow;
  const float vp = (Load(mixed_qkv, v_offset) - dot) * beta;
  float output = 0.0f;
#pragma unroll
  for (int c = 0; c < BK; ++c) {
    sh[c] += vp * bk[c];
    output += sh[c] * bq[c];
  }
  Store(out, (i_n * hv_n + hv) * dv + vrow, output);
#pragma unroll
  for (int c = 0; c < BK; ++c) Store(state, s_row + c, sh[c]);
}

// ---------------------------------------------------------------------------
// B) FIRST-RUN PROBE, KEPT VERBATIM -- the buffered ReplaySSM decode loop of
//    the spec's "## Design" written with a RESIDENT `float sh[BK=128]` array
//    that both routes share. This is the structure the pin deliberately avoids
//    (replayssm_config.py:47-52 tiles at 32 and 64; the two routes hold
//    DISJOINT tile locals, :391-393). It is retained ONLY so that the numbers
//    below are read against the first run under the same ptxas.
// ---------------------------------------------------------------------------
template <typename T, typename TState, int BK, int L>
__global__ void ProbeReplaySsmRegTileKernel(
    T* out, const T* mixed_qkv, const T* a, const T* b, const void* a_log,
    DType a_log_dtype, const void* dt_bias, DType dt_bias_dtype, TState* state,
    TState* ring_d, TState* ring_k, float* ring_g, const int32_t* state_idx,
    const int32_t* write_pos, int64_t state_slots, int64_t mixed_stride,
    int64_t a_stride, int64_t b_stride, int64_t hk_n, int64_t dk, int64_t hv_n,
    int64_t dv, int64_t bv, float scale) {
  const int64_t i_v = blockIdx.x;
  const int64_t i_nh = blockIdx.y;
  const int64_t i_n = i_nh / hv_n;
  const int64_t hv = i_nh % hv_n;
  const int64_t hk = hv / (hv_n / hk_n);
  const int64_t vbase = i_v * bv;
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t vrow = vbase + lane;

  // Slot ABI identical to GdnPackedDecode (fla ssm_state_indices): a negative or
  // out-of-range slot zeroes the output row and returns.
  const int32_t slot = state_idx[i_n];
  if (slot < 0 || slot >= state_slots) {
    if (vrow < dv) Store(out, (i_n * hv_n + hv) * dv + vrow, 0.0f);
    return;
  }

  // Cursor and flush flag, mirroring vLLM mamba_attn.py:617-618: wp is the slot
  // this step writes, and it is also the count of live records held since the
  // last checkpoint. is_flush on the last slot of the window.
  const int wp = write_pos[i_n];
  const int m = wp;
  const bool is_flush = (wp == L - 1);

  __shared__ float bq[BK];
  __shared__ float bk[BK];
  // vLLM's precompute pass (:22-129): the (k_j . q) and (k_j . k) products are
  // uniform over the 32 value-rows, so they are computed once per block.
  __shared__ float s_kq[L];
  __shared__ float s_kk[L];
  __shared__ float s_w[L];
  __shared__ float s_total_decay;
  __shared__ float s_kq_now;

  const int nthreads = static_cast<int>(blockDim.x);
  const int64_t mixed_row = i_n * mixed_stride;
  const int64_t qbase = mixed_row + hk * dk;
  const int64_t kbase = mixed_row + hk_n * dk + hk * dk;
  float q_sumsq = 0.0f;
  float k_sumsq = 0.0f;
  for (int64_t ki = lane; ki < dk; ki += nthreads) {
    const float qv = Load(mixed_qkv, qbase + ki);
    const float kv = Load(mixed_qkv, kbase + ki);
    bq[ki] = qv;
    bk[ki] = kv;
    q_sumsq += qv * qv;
    k_sumsq += kv * kv;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    q_sumsq += __shfl_xor_sync(0xffffffffu, q_sumsq, off);
    k_sumsq += __shfl_xor_sync(0xffffffffu, k_sumsq, off);
  }
  const float q_scaled_inv = (1.0f / sqrtf(q_sumsq + 1e-6f)) * scale;
  const float k_inv = 1.0f / sqrtf(k_sumsq + 1e-6f);
  for (int64_t ki = lane; ki < dk; ki += nthreads) {
    bq[ki] *= q_scaled_inv;
    bk[ki] *= k_inv;
  }
  __syncwarp();

  const float av = Load(a, i_n * a_stride + hv);
  const float b_raw = Load(b, i_n * b_stride + hv);
  const float x = av + LoadFloating(dt_bias, dt_bias_dtype, hv);
  const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
  const float g_now = -expf(LoadFloating(a_log, a_log_dtype, hv)) * softplus;
  const float beta = RoundToStorage<T>(1.0f / (1.0f + expf(-b_raw)));

  // (k_t . q_t): the current step's record contributes d_t (k_t . q_t) to y.
  {
    float acc = 0.0f;
    for (int64_t ki = lane; ki < dk; ki += nthreads) acc += bk[ki] * bq[ki];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) s_kq_now = acc;
  }

  // Precompute pass over the live ring records. Ring k is grouped over H_k
  // exactly as vLLM's B_cache is grouped over n_groups (mamba_utils.py:216,
  // memory_pool.py:471-475).
  const int64_t k_ring_head =
      ((static_cast<int64_t>(slot) * hk_n + hk) * L) * dk;
#pragma unroll 1
  for (int j = 0; j < L; ++j) {
    if (j >= m) break;
    const int64_t kr = k_ring_head + static_cast<int64_t>(j) * dk;
    float kq = 0.0f;
    float kk = 0.0f;
    for (int64_t c = lane; c < dk; c += nthreads) {
      const float kj = Load(ring_k, kr + c);
      kq += kj * bq[c];
      kk += kj * bk[c];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      kq += __shfl_xor_sync(0xffffffffu, kq, off);
      kk += __shfl_xor_sync(0xffffffffu, kk, off);
    }
    if (lane == 0) {
      s_kq[j] = kq;
      s_kk[j] = kk;
    }
  }

  // Decay refolding: G' = exp(g_t + sum_j g_j) and W'_j = exp(g_t + sum_{i>j}
  // g_i). g is fp32 in both upstreams and stays fp32 here -- it is the one term
  // whose reconstruction error grows linearly in L (spec "## Design").
  const int64_t g_ring_head =
      ((static_cast<int64_t>(slot) * hv_n + hv) * L);
  if (lane == 0) {
    float acc = g_now;
#pragma unroll 1
    for (int j = L - 1; j >= 0; --j) {
      if (j >= m) continue;
      s_w[j] = expf(acc);
      acc += ring_g[g_ring_head + j];
    }
    s_total_decay = expf(acc);
  }
  __syncwarp();

  // Tail lanes helped fill bq/bk and the precompute; they own no state row.
  if (vrow >= dv) return;

  const int64_t s_row =
      ((static_cast<int64_t>(slot) * hv_n + hv) * dv + vrow) * dk;
  const int64_t d_ring_row =
      ((static_cast<int64_t>(slot) * hv_n + hv) * L) * dv + vrow;

  // The checkpoint row, register-resident -- the same [BK] tile the control
  // holds, and the reason R1 is the top risk.
  float sh[BK];
#pragma unroll
  for (int c = 0; c < BK; ++c) sh[c] = Load(state, s_row + c);

  // Non-flush route: y and (g_t S_{t-1}) k_t are read out of the checkpoint and
  // the ring WITHOUT materializing the state
  // (selective_state_update_replayssm_output_only.py:275-279).
  const float total_decay = s_total_decay;
  float dot_q = 0.0f;
  float dot_k = 0.0f;
#pragma unroll
  for (int c = 0; c < BK; ++c) {
    dot_q += sh[c] * bq[c];
    dot_k += sh[c] * bk[c];
  }
  dot_q *= total_decay;
  dot_k *= total_decay;

#pragma unroll 1
  for (int j = 0; j < L; ++j) {
    if (j >= m) break;
    const float dj =
        Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j];
    dot_q += dj * s_kq[j];
    dot_k += dj * s_kk[j];
  }

  const int64_t v_offset = mixed_row + 2 * hk_n * dk + hv * dv + vrow;
  const float d_now = (Load(mixed_qkv, v_offset) - dot_k) * beta;
  const float output = dot_q + d_now * s_kq_now;
  Store(out, (i_n * hv_n + hv) * dv + vrow, output);

  // Append this step's rank-1 factors to the ring.
  Store(ring_d, d_ring_row + static_cast<int64_t>(wp) * dv, d_now);
  if (lane == 0) ring_g[g_ring_head + wp] = g_now;
  const int64_t kr_now = k_ring_head + static_cast<int64_t>(wp) * dk;
  for (int64_t c = lane; c < dk; c += nthreads) Store(ring_k, kr_now + c, bk[c]);

  // Flush route: reconstruct S_t in the same registers, persist it, and only
  // then is the checkpoint advanced (:358-362). This is the branch that keeps
  // sh[] live across the ring loop.
  if (is_flush) {
#pragma unroll
    for (int c = 0; c < BK; ++c) sh[c] = sh[c] * total_decay + d_now * bk[c];
#pragma unroll 1
    for (int j = 0; j < L; ++j) {
      if (j >= m) break;
      const float dj =
          Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j];
      const int64_t kr = k_ring_head + static_cast<int64_t>(j) * dk;
#pragma unroll
      for (int c = 0; c < BK; ++c) sh[c] += dj * Load(ring_k, kr + c);
    }
#pragma unroll
    for (int c = 0; c < BK; ++c) Store(state, s_row + c, sh[c]);
  }
}

// ---------------------------------------------------------------------------
// Shared prologue for the route-isolated kernels C, D, E and F. Identical to the
// first 60 lines of B: q/k RMS-normalisation into shared memory, the per-head
// gate, the (k_t . q_t) product, the ring precompute pass over the live records,
// and the refolded decay weights. It is a macro rather than a __device__
// function so that ptxas sees exactly the same inlined body it saw in B, and so
// that no ABI boundary can hide register pressure from the measurement.
// ---------------------------------------------------------------------------
#define PROBE_REPLAYSSM_PROLOGUE(L_)                                          \
  const int64_t i_v = blockIdx.x;                                             \
  const int64_t i_nh = blockIdx.y;                                            \
  const int64_t i_n = i_nh / hv_n;                                            \
  const int64_t hv = i_nh % hv_n;                                             \
  const int64_t hk = hv / (hv_n / hk_n);                                      \
  const int64_t vbase = i_v * bv;                                             \
  const int lane = static_cast<int>(threadIdx.x);                             \
  const int64_t vrow = vbase + lane;                                          \
  const int32_t slot = state_idx[i_n];                                        \
  if (slot < 0 || slot >= state_slots) {                                      \
    if (vrow < dv) Store(out, (i_n * hv_n + hv) * dv + vrow, 0.0f);           \
    return;                                                                   \
  }                                                                           \
  const int wp = write_pos[i_n];                                              \
  const int m = wp;                                                           \
  __shared__ float bq[BK];                                                    \
  __shared__ float bk[BK];                                                    \
  __shared__ float s_kq[L_];                                                  \
  __shared__ float s_kk[L_];                                                  \
  __shared__ float s_w[L_];                                                   \
  __shared__ float s_total_decay;                                             \
  __shared__ float s_kq_now;                                                  \
  const int nthreads = static_cast<int>(blockDim.x);                          \
  const int64_t mixed_row = i_n * mixed_stride;                               \
  const int64_t qbase = mixed_row + hk * dk;                                  \
  const int64_t kbase = mixed_row + hk_n * dk + hk * dk;                      \
  float q_sumsq = 0.0f;                                                       \
  float k_sumsq = 0.0f;                                                       \
  for (int64_t ki = lane; ki < dk; ki += nthreads) {                          \
    const float qv = Load(mixed_qkv, qbase + ki);                             \
    const float kv = Load(mixed_qkv, kbase + ki);                             \
    bq[ki] = qv;                                                              \
    bk[ki] = kv;                                                              \
    q_sumsq += qv * qv;                                                       \
    k_sumsq += kv * kv;                                                       \
  }                                                                           \
  _Pragma("unroll") for (int off = 16; off > 0; off >>= 1) {                  \
    q_sumsq += __shfl_xor_sync(0xffffffffu, q_sumsq, off);                    \
    k_sumsq += __shfl_xor_sync(0xffffffffu, k_sumsq, off);                    \
  }                                                                           \
  const float q_scaled_inv = (1.0f / sqrtf(q_sumsq + 1e-6f)) * scale;         \
  const float k_inv = 1.0f / sqrtf(k_sumsq + 1e-6f);                          \
  for (int64_t ki = lane; ki < dk; ki += nthreads) {                          \
    bq[ki] *= q_scaled_inv;                                                   \
    bk[ki] *= k_inv;                                                          \
  }                                                                           \
  __syncwarp();                                                               \
  const float av = Load(a, i_n * a_stride + hv);                              \
  const float b_raw = Load(b, i_n * b_stride + hv);                           \
  const float x = av + LoadFloating(dt_bias, dt_bias_dtype, hv);              \
  const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;                    \
  const float g_now = -expf(LoadFloating(a_log, a_log_dtype, hv)) * softplus; \
  const float beta = RoundToStorage<T>(1.0f / (1.0f + expf(-b_raw)));         \
  {                                                                           \
    float acc = 0.0f;                                                         \
    for (int64_t ki = lane; ki < dk; ki += nthreads) acc += bk[ki] * bq[ki];  \
    _Pragma("unroll") for (int off = 16; off > 0; off >>= 1) acc +=           \
        __shfl_xor_sync(0xffffffffu, acc, off);                               \
    if (lane == 0) s_kq_now = acc;                                            \
  }                                                                           \
  const int64_t k_ring_head =                                                 \
      ((static_cast<int64_t>(slot) * hk_n + hk) * L_) * dk;                   \
  _Pragma("unroll 1") for (int j = 0; j < L_; ++j) {                          \
    if (j >= m) break;                                                        \
    const int64_t kr = k_ring_head + static_cast<int64_t>(j) * dk;            \
    float kq = 0.0f;                                                          \
    float kk = 0.0f;                                                          \
    for (int64_t c = lane; c < dk; c += nthreads) {                           \
      const float kj = Load(ring_k, kr + c);                                  \
      kq += kj * bq[c];                                                       \
      kk += kj * bk[c];                                                       \
    }                                                                         \
    _Pragma("unroll") for (int off = 16; off > 0; off >>= 1) {                \
      kq += __shfl_xor_sync(0xffffffffu, kq, off);                            \
      kk += __shfl_xor_sync(0xffffffffu, kk, off);                            \
    }                                                                         \
    if (lane == 0) {                                                          \
      s_kq[j] = kq;                                                           \
      s_kk[j] = kk;                                                           \
    }                                                                         \
  }                                                                           \
  const int64_t g_ring_head = ((static_cast<int64_t>(slot) * hv_n + hv) * L_);\
  if (lane == 0) {                                                            \
    float acc = g_now;                                                        \
    _Pragma("unroll 1") for (int j = L_ - 1; j >= 0; --j) {                   \
      if (j >= m) continue;                                                   \
      s_w[j] = expf(acc);                                                     \
      acc += ring_g[g_ring_head + j];                                         \
    }                                                                         \
    s_total_decay = expf(acc);                                                \
  }                                                                           \
  __syncwarp();                                                               \
  if (vrow >= dv) return;                                                     \
  const int64_t s_row =                                                       \
      ((static_cast<int64_t>(slot) * hv_n + hv) * dv + vrow) * dk;            \
  const int64_t d_ring_row =                                                  \
      ((static_cast<int64_t>(slot) * hv_n + hv) * L_) * dv + vrow;            \
  const float total_decay = s_total_decay;

// The signature every route-isolated kernel takes. Identical to B's.
#define PROBE_REPLAYSSM_PARAMS                                                \
  T *out, const T *mixed_qkv, const T *a, const T *b, const void *a_log,      \
      DType a_log_dtype, const void *dt_bias, DType dt_bias_dtype,            \
      TState *state, TState *ring_d, TState *ring_k, float *ring_g,           \
      const int32_t *state_idx, const int32_t *write_pos, int64_t state_slots,\
      int64_t mixed_stride, int64_t a_stride, int64_t b_stride, int64_t hk_n, \
      int64_t dk, int64_t hv_n, int64_t dv, int64_t bv, float scale

// ---------------------------------------------------------------------------
// C) THE 15-OF-16 ROUTE, ALONE, WITH NO REGISTER ARRAY.
//
//    This is the route that carries the whole bandwidth case: the state is read
//    once and never written. Upstream reads it "streamed over NF dstate tiles so
//    the (M, N) state slice is never held whole"
//    (selective_state_update_replayssm_output_only.py:309-311). In this algebra
//    the per-lane slice is one row of BK values and it is consumed by exactly two
//    reductions, so it needs no array at all: two scalar accumulators and a
//    streaming load. There is no `float sh[BK]` in this kernel, and that is the
//    point of it.
//
//    If this kernel does not spill, the row's stop condition is NOT met, because
//    the structure that spilled in B is not the structure ReplaySSM has to be
//    written in.
// ---------------------------------------------------------------------------
template <typename T, typename TState, int BK, int L>
__global__ void ProbeReplaySsmNonFlushKernel(PROBE_REPLAYSSM_PARAMS) {
  PROBE_REPLAYSSM_PROLOGUE(L)

  // Streamed checkpoint readout. Each element is loaded once and folded into
  // both reductions in the same pass, so nothing stays live behind the loop.
  float dot_q = 0.0f;
  float dot_k = 0.0f;
  for (int c = 0; c < BK; ++c) {
    const float s = Load(state, s_row + c);
    dot_q += s * bq[c];
    dot_k += s * bk[c];
  }
  dot_q *= total_decay;
  dot_k *= total_decay;

#pragma unroll 1
  for (int j = 0; j < L; ++j) {
    if (j >= m) break;
    const float dj =
        Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j];
    dot_q += dj * s_kq[j];
    dot_k += dj * s_kk[j];
  }

  const int64_t v_offset = mixed_row + 2 * hk_n * dk + hv * dv + vrow;
  const float d_now = (Load(mixed_qkv, v_offset) - dot_k) * beta;
  const float output = dot_q + d_now * s_kq_now;
  Store(out, (i_n * hv_n + hv) * dv + vrow, output);

  // Ring append, mirroring the pin's non-flush branch (:352-372).
  Store(ring_d, d_ring_row + static_cast<int64_t>(wp) * dv, d_now);
  if (lane == 0) ring_g[g_ring_head + wp] = g_now;
  const int64_t kr_now = k_ring_head + static_cast<int64_t>(wp) * dk;
  for (int64_t c = lane; c < dk; c += nthreads) Store(ring_k, kr_now + c, bk[c]);
}

// ---------------------------------------------------------------------------
// D) THE 1-OF-16 ROUTE, ALONE, TILED AT THE PIN'S fl_dstate_tile == 64.
//
//    The readout that produces d_now is the same streamed pair of scalars as in
//    C. The reconstruction then walks dk in FL_DSTATE_TILE-wide tiles, re-loading
//    the checkpoint from HBM for each tile.
//
//    That re-load is semantically identical to B's resident array. In B, the
//    flush branch computes `sh[c] * total_decay` from values loaded before any
//    store, and the ONLY write to `state` is the final `Store` after the ring
//    fold. HBM therefore still holds exactly those pre-decay values when tile t
//    is loaded here, and tiles are disjoint in c so an earlier tile's store
//    cannot alias a later tile's load.
// ---------------------------------------------------------------------------
template <typename T, typename TState, int BK, int L, int FL_DSTATE_TILE>
__global__ void ProbeReplaySsmFlushKernel(PROBE_REPLAYSSM_PARAMS) {
  PROBE_REPLAYSSM_PROLOGUE(L)

  float dot_k = 0.0f;
  for (int c = 0; c < BK; ++c) dot_k += Load(state, s_row + c) * bk[c];
  dot_k *= total_decay;
#pragma unroll 1
  for (int j = 0; j < L; ++j) {
    if (j >= m) break;
    dot_k += Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j] *
             s_kk[j];
  }
  const int64_t v_offset = mixed_row + 2 * hk_n * dk + hv * dv + vrow;
  const float d_now = (Load(mixed_qkv, v_offset) - dot_k) * beta;

  // Reconstruct, persist, then read y off the reconstructed tile (:395-434).
  float out_acc = 0.0f;
  for (int t = 0; t < BK; t += FL_DSTATE_TILE) {
    float st_f[FL_DSTATE_TILE];
#pragma unroll
    for (int i = 0; i < FL_DSTATE_TILE; ++i)
      st_f[i] = Load(state, s_row + t + i) * total_decay + d_now * bk[t + i];
#pragma unroll 1
    for (int j = 0; j < L; ++j) {
      if (j >= m) break;
      const float dj =
          Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j];
      const int64_t kr = k_ring_head + static_cast<int64_t>(j) * dk;
#pragma unroll
      for (int i = 0; i < FL_DSTATE_TILE; ++i)
        st_f[i] += dj * Load(ring_k, kr + t + i);
    }
#pragma unroll
    for (int i = 0; i < FL_DSTATE_TILE; ++i) {
      Store(state, s_row + t + i, st_f[i]);
      out_acc += st_f[i] * bq[t + i];
    }
  }
  Store(out, (i_n * hv_n + hv) * dv + vrow, out_acc);
}

// ---------------------------------------------------------------------------
// E) D with the outer dstate-tile loop pinned to one iteration at a time, so
//    that a spill in D can be attributed to two tiles being live at once rather
//    than to the tile width itself. Everything else is byte-identical to D.
// ---------------------------------------------------------------------------
template <typename T, typename TState, int BK, int L, int FL_DSTATE_TILE>
__global__ void ProbeReplaySsmFlushSeqKernel(PROBE_REPLAYSSM_PARAMS) {
  PROBE_REPLAYSSM_PROLOGUE(L)

  float dot_k = 0.0f;
  for (int c = 0; c < BK; ++c) dot_k += Load(state, s_row + c) * bk[c];
  dot_k *= total_decay;
#pragma unroll 1
  for (int j = 0; j < L; ++j) {
    if (j >= m) break;
    dot_k += Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j] *
             s_kk[j];
  }
  const int64_t v_offset = mixed_row + 2 * hk_n * dk + hv * dv + vrow;
  const float d_now = (Load(mixed_qkv, v_offset) - dot_k) * beta;

  float out_acc = 0.0f;
#pragma unroll 1
  for (int t = 0; t < BK; t += FL_DSTATE_TILE) {
    float st_f[FL_DSTATE_TILE];
#pragma unroll
    for (int i = 0; i < FL_DSTATE_TILE; ++i)
      st_f[i] = Load(state, s_row + t + i) * total_decay + d_now * bk[t + i];
#pragma unroll 1
    for (int j = 0; j < L; ++j) {
      if (j >= m) break;
      const float dj =
          Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j];
      const int64_t kr = k_ring_head + static_cast<int64_t>(j) * dk;
#pragma unroll
      for (int i = 0; i < FL_DSTATE_TILE; ++i)
        st_f[i] += dj * Load(ring_k, kr + t + i);
    }
#pragma unroll
    for (int i = 0; i < FL_DSTATE_TILE; ++i) {
      Store(state, s_row + t + i, st_f[i]);
      out_acc += st_f[i] * bq[t + i];
    }
  }
  Store(out, (i_n * hv_n + hv) * dv + vrow, out_acc);
}

// ---------------------------------------------------------------------------
// F) BOTH ROUTES IN ONE KERNEL, TRANSCRIBING THE PIN'S STRUCTURE.
//
//    NF_DSTATE_TILE == 32 and FL_DSTATE_TILE == 64 are the tuned Blackwell
//    values at dstate == 128 (replayssm_config.py:47-52). The two branches hold
//    DISJOINT tile locals -- `st` in the non-flush arm, `st_f` in the flush arm
//    -- which is what the pin does and why (:391-393: differing tile widths
//    "would force a shape-mismatched merge at the if/else exit"). Whether ptxas
//    then merges their live ranges anyway is exactly what this kernel measures.
// ---------------------------------------------------------------------------
template <typename T, typename TState, int BK, int L, int NF_DSTATE_TILE,
          int FL_DSTATE_TILE>
__global__ void ProbeReplaySsmFusedKernel(PROBE_REPLAYSSM_PARAMS) {
  PROBE_REPLAYSSM_PROLOGUE(L)

  const bool is_flush = (wp == L - 1);
  const int64_t v_offset = mixed_row + 2 * hk_n * dk + hv * dv + vrow;

  if (!is_flush) {
    // Non-flush arm: NF tiles, nothing written back to `state`.
    float dot_q = 0.0f;
    float dot_k = 0.0f;
    for (int t = 0; t < BK; t += NF_DSTATE_TILE) {
      float st[NF_DSTATE_TILE];
#pragma unroll
      for (int i = 0; i < NF_DSTATE_TILE; ++i) st[i] = Load(state, s_row + t + i);
#pragma unroll
      for (int i = 0; i < NF_DSTATE_TILE; ++i) {
        dot_q += st[i] * bq[t + i];
        dot_k += st[i] * bk[t + i];
      }
    }
    dot_q *= total_decay;
    dot_k *= total_decay;
#pragma unroll 1
    for (int j = 0; j < L; ++j) {
      if (j >= m) break;
      const float dj =
          Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j];
      dot_q += dj * s_kq[j];
      dot_k += dj * s_kk[j];
    }
    const float d_now = (Load(mixed_qkv, v_offset) - dot_k) * beta;
    Store(out, (i_n * hv_n + hv) * dv + vrow, dot_q + d_now * s_kq_now);
    Store(ring_d, d_ring_row + static_cast<int64_t>(wp) * dv, d_now);
    if (lane == 0) ring_g[g_ring_head + wp] = g_now;
    const int64_t kr_now = k_ring_head + static_cast<int64_t>(wp) * dk;
    for (int64_t c = lane; c < dk; c += nthreads)
      Store(ring_k, kr_now + c, bk[c]);
  } else {
    // Flush arm: FL tiles, checkpoint re-read per tile, state persisted.
    float dot_k = 0.0f;
    for (int c = 0; c < BK; ++c) dot_k += Load(state, s_row + c) * bk[c];
    dot_k *= total_decay;
#pragma unroll 1
    for (int j = 0; j < L; ++j) {
      if (j >= m) break;
      dot_k += Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) *
               s_w[j] * s_kk[j];
    }
    const float d_now = (Load(mixed_qkv, v_offset) - dot_k) * beta;
    float out_acc = 0.0f;
    for (int t = 0; t < BK; t += FL_DSTATE_TILE) {
      float st_f[FL_DSTATE_TILE];
#pragma unroll
      for (int i = 0; i < FL_DSTATE_TILE; ++i)
        st_f[i] = Load(state, s_row + t + i) * total_decay + d_now * bk[t + i];
#pragma unroll 1
      for (int j = 0; j < L; ++j) {
        if (j >= m) break;
        const float dj =
            Load(ring_d, d_ring_row + static_cast<int64_t>(j) * dv) * s_w[j];
        const int64_t kr = k_ring_head + static_cast<int64_t>(j) * dk;
#pragma unroll
        for (int i = 0; i < FL_DSTATE_TILE; ++i)
          st_f[i] += dj * Load(ring_k, kr + t + i);
      }
#pragma unroll
      for (int i = 0; i < FL_DSTATE_TILE; ++i) {
        Store(state, s_row + t + i, st_f[i]);
        out_acc += st_f[i] * bq[t + i];
      }
    }
    Store(out, (i_n * hv_n + hv) * dv + vrow, out_acc);
  }
}


// Explicit instantiation at the 27B decode shape the spec names: activation
// bf16, SSM state fp32, BK == dk == 128, one warp per [BV=32] value tile,
// L == 16 (both upstream defaults, and the spec's derived optimum for this
// shape). Nothing calls these; they exist so ptxas emits them.
template __global__ void ProbeControlRegTileKernel<__nv_bfloat16, float, 128>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const void*, DType, const void*, DType, float*,
    const int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, float);

template __global__ void
ProbeReplaySsmRegTileKernel<__nv_bfloat16, float, 128, 16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const void*, DType, const void*, DType, float*,
    float*, float*, float*, const int32_t*, const int32_t*, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);

// The route-isolated kernels, same shape, at the pin's tuned Blackwell dstate
// tiles: nf_dstate_tile == 32 and fl_dstate_tile == 64 at dstate == 128
// (replayssm_config.py:47-52, via _dstate_tile()).
template __global__ void
ProbeReplaySsmNonFlushKernel<__nv_bfloat16, float, 128, 16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const void*, DType, const void*, DType, float*,
    float*, float*, float*, const int32_t*, const int32_t*, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);

template __global__ void
ProbeReplaySsmFlushKernel<__nv_bfloat16, float, 128, 16, 64>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const void*, DType, const void*, DType, float*,
    float*, float*, float*, const int32_t*, const int32_t*, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);

template __global__ void
ProbeReplaySsmFlushSeqKernel<__nv_bfloat16, float, 128, 16, 64>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const void*, DType, const void*, DType, float*,
    float*, float*, float*, const int32_t*, const int32_t*, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);

template __global__ void
ProbeReplaySsmFusedKernel<__nv_bfloat16, float, 128, 16, 32, 64>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const void*, DType, const void*, DType, float*,
    float*, float*, float*, const int32_t*, const int32_t*, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);
