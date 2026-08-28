// The DSA "Lightning Indexer" selection pair (CUDA) — dots3-note W4b-3c, #699.
//
// The device sibling of src/vt/cpu/cpu_dsa_indexer.cpp, which carries the full
// port map. In short, @ `bc2d63e650`:
//
//   DsaIndexerLogits <- vllm/v1/attention/ops/triton_fp8_mqa_logits.py:120-156
//                       (`tl.dot(..., input_precision="ieee")`, `* kv_scales`,
//                        `tl.maximum(_, 0.0)`, `* w_block`,
//                        `tl.sum(_, axis=0)`, masked store)
//   the FOLD         <- vllm/model_executor/models/deepseek_v2.py:840 and
//                       vllm/model_executor/layers/sparse_attn_indexer.py:203-207
//   DsaTopkSelect    <- sparse_attn_indexer.py:509 `ops.top_k_per_row_prefill`
//                       plus the short-context all-select
//
// ─── DECOMPOSITION, and why it is not upstream's ────────────────────────────
// Upstream tiles the logits kernel over (query block, KV block) and uses
// `tl.dot` on a [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV] tile. Ours is
// ONE BLOCK PER (query token, key tile) with a warp per head and a block-wide
// reduction — a scalar decomposition of the same arithmetic, in the same f32
// precision and the same summation SHAPE (per-head dot, then ReLU, then the
// weighted sum over heads). A tensor-core tiling is a later concern; getting a
// different ANSWER from it would be the defect, and the CPU reference plus the
// host oracle are what pin the answer.
//
// The top-k is one block per query row with a bounded selection pass rather
// than a sort. It reproduces the CPU rule exactly, including the two parts that
// are load-bearing: ties break toward the SMALLER key index, and the emission
// is ASCENDING — which is what makes a full selection reproduce dense attention
// bit for bit inside vt::MlaDecodeAttention.
//
// DETERMINISM. Every reduction here is a fixed-shape shuffle/shared-memory tree
// over a compile-time lane count, run in the same order every launch. There is
// no atomicAdd and no run-to-run-variable reduction, so both ops are
// bit-reproducible run to run for a fixed shape — the house convention
// cuda_mla_attn.cu states for the MLA decode.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

void CheckDsa(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: dsa_indexer: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

__device__ __forceinline__ float LdF(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float LdF(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ float LdF(const __half* p, int64_t i) { return __half2float(p[i]); }

// One WARP per indexer head, `kDsaWarps` warps per block. 64 indexer heads on
// the released config, so a 4-warp block sweeps them in 16 passes with the key
// row resident in registers per lane.
constexpr int kDsaWarps = 4;
constexpr int kDsaThreads = kDsaWarps * 32;

// ─── LOGITS ─────────────────────────────────────────────────────────────────
// Grid (num_tokens, key_tiles). Block computes `kDsaKeyTile` logits of one
// query row. Each warp owns one head at a time; lane `l` accumulates the dot
// over dims l, l+32, ... and the warp reduces it, so the per-head dot is one
// fixed-shape shuffle tree.
constexpr int kDsaKeyTile = 8;

template <typename T>
__global__ __launch_bounds__(kDsaThreads) void DsaIndexerLogitsKernel(
    float* __restrict__ logits, const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ weights, const float* __restrict__ q_scale,
    const int32_t* __restrict__ win_start, const int32_t* __restrict__ win_end, int64_t lg_s0,
    int64_t q_s0, int64_t q_s1, int64_t k_s0, int64_t w_s0, int64_t qs_s0, int num_tokens,
    int num_keys, int n_heads, int head_dim, float gfold) {
  const int t = static_cast<int>(blockIdx.x);
  const int tile = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (t >= num_tokens) return;

  const int s_lo = tile * kDsaKeyTile;
  float* row = logits + static_cast<int64_t>(t) * lg_s0;

  // `tl.store(logits_ptrs, scores, mask=in_window)` (:156) writes only the
  // in-window columns and leaves the rest at the row's pre-filled -inf. Here
  // EACH TILE BLOCK OWNS ITS OWN COLUMNS and writes both cases from the single
  // store below — deliberately, rather than having one block pre-fill the whole
  // row, which would race every other block of the same row.
  const int lo = max(0, win_start[t]);
  const int hi = min(num_keys, win_end[t]);

  __shared__ float partial[kDsaKeyTile][kDsaWarps];
  for (int n = 0; n < kDsaKeyTile; ++n) {
    const int s = s_lo + n;
    float acc = 0.0f;
    if (s >= lo && s < hi) {
      // Sweep the heads `kDsaWarps` at a time; each warp owns head `h`.
      for (int h = warp; h < n_heads; h += kDsaWarps) {
        float dot = 0.0f;
        const T* qp = q + static_cast<int64_t>(t) * q_s0 + static_cast<int64_t>(h) * q_s1;
        const T* kp = k + static_cast<int64_t>(s) * k_s0;
        for (int d = lane; d < head_dim; d += 32) dot += LdF(qp, d) * LdF(kp, d);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }
        // `weights * q_scale * softmax_scale * n_head_scale`
        // (deepseek_v2.py:840). A null q_scale is upstream's unquantized arm,
        // where the per-head factor is exactly 1.
        const float w = LdF(weights, static_cast<int64_t>(t) * w_s0 + h);
        const float per_head =
            q_scale != nullptr ? q_scale[static_cast<int64_t>(t) * qs_s0 + h] : 1.0f;
        // `tl.maximum(scores, 0.0)` (:129/:150) — the ReLU is the load-bearing
        // nuance, and it is what makes an exact 0.0 an ordinary logit value.
        acc += w * per_head * gfold * (dot > 0.0f ? dot : 0.0f);
      }
    }
    if (lane == 0) partial[n][warp] = acc;
  }
  __syncthreads();
  // Fixed-shape block reduction over the `kDsaWarps` per-warp partials, in
  // ascending warp order, by one thread per key — deterministic by
  // construction, no atomics.
  if (static_cast<int>(threadIdx.x) < kDsaKeyTile) {
    const int n = static_cast<int>(threadIdx.x);
    const int s = s_lo + n;
    if (s < num_keys) {
      if (s >= lo && s < hi) {
        float sum = 0.0f;
        for (int w = 0; w < kDsaWarps; ++w) sum += partial[n][w];
        row[s] = sum;
      } else {
        row[s] = -CUDART_INF_F;
      }
    }
  }
}

// ─── TOP-K ──────────────────────────────────────────────────────────────────
// One block per query row. The short-context branch is a straight ascending
// fill. The full branch runs upstream's selection semantics without a sort: for
// each of `topk` output slots, one thread scans and the block reduces to the
// best remaining (logit, index) pair under the rule "larger logit wins; on an
// exact tie the SMALLER index wins". `topk` is 2048 on the released config and
// the scan is over the candidate range, so this is quadratic in the worst case
// and deliberately so — it is a REFERENCE-GRADE device kernel whose job is to
// give the same answer as the CPU arm on any input, not to be the shipping
// selection kernel. A threshold-based selection is a later brick and it will be
// gated against exactly this.
constexpr int kTopkThreads = 256;

__global__ __launch_bounds__(kTopkThreads) void DsaTopkSelectKernel(
    int32_t* __restrict__ indices, int32_t* __restrict__ counts,
    const float* __restrict__ logits, const int32_t* __restrict__ win_start,
    const int32_t* __restrict__ win_end, int64_t idx_s0, int64_t lg_s0, int num_keys,
    int topk) {
  const int t = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  int32_t* dst = indices + static_cast<int64_t>(t) * idx_s0;
  const float* row = logits + static_cast<int64_t>(t) * lg_s0;

  // `-1` is the "no token" sentinel the topk buffer is pre-filled with
  // (sparse_attn_indexer.py:431-432; `:426-430` is the comment above it).
  for (int i = tid; i < topk; i += kTopkThreads) dst[i] = -1;
  __syncthreads();

  const int lo = max(0, win_start[t]);
  const int hi = min(num_keys, win_end[t]);
  const int n = max(0, hi - lo);

  if (n <= topk) {
    // SHORT CONTEXT: every candidate, ascending. This branch is what makes a
    // sparse layer identical to dense attention while the whole context fits in
    // `index_topk` — the regime upstream keeps its dense-MHA prefill for
    // (`use_dense_mha = prefill_max_seq_len <= self.topk_tokens`,
    //  sparse_mla_attention.py:296-299).
    for (int i = tid; i < n; i += kTopkThreads) dst[i] = static_cast<int32_t>(lo + i);
    if (tid == 0) counts[t] = static_cast<int32_t>(n);
    return;
  }

  // FULL TOP-K. `taken_hi` is the exclusive upper bound of what has already been
  // chosen under the ordering "(logit desc, index asc)", so a candidate is still
  // available iff it sorts strictly after the last pick. Comparing against the
  // last pick rather than keeping a visited set is what keeps this O(topk * n)
  // with O(1) state.
  __shared__ float best_val[kTopkThreads];
  __shared__ int best_idx[kTopkThreads];
  __shared__ float pick_val;
  __shared__ int pick_idx;
  if (tid == 0) {
    pick_val = CUDART_INF_F;
    pick_idx = -1;
  }
  __syncthreads();

  for (int slot = 0; slot < topk; ++slot) {
    const float bar_v = pick_val;
    const int bar_i = pick_idx;
    float bv = -CUDART_INF_F;
    int bi = -1;
    for (int s = lo + tid; s < hi; s += kTopkThreads) {
      const float v = row[s];
      // Strictly after the previous pick in (logit desc, index asc) order.
      const bool avail = bar_i < 0 || v < bar_v || (v == bar_v && s > bar_i);
      if (!avail) continue;
      if (bi < 0 || v > bv || (v == bv && s < bi)) {
        bv = v;
        bi = s;
      }
    }
    best_val[tid] = bv;
    best_idx[tid] = bi;
    __syncthreads();
    // Fixed-shape shared-memory tree, ascending stride, same order every launch.
    for (int stride = kTopkThreads >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float ov = best_val[tid + stride];
        const int oi = best_idx[tid + stride];
        const bool better =
            oi >= 0 && (best_idx[tid] < 0 || ov > best_val[tid] ||
                        (ov == best_val[tid] && oi < best_idx[tid]));
        if (better) {
          best_val[tid] = ov;
          best_idx[tid] = oi;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      pick_val = best_val[0];
      pick_idx = best_idx[0];
      // The chosen keys are written in DESCENDING-rank order here and sorted
      // into ascending KEY order below; ascending emission is what makes a full
      // selection reproduce the dense reduction order.
      dst[slot] = static_cast<int32_t>(best_idx[0]);
    }
    __syncthreads();
  }

  // Ascending key order over the `topk` chosen entries. One thread, an
  // insertion pass — `topk` is small against the candidate scan above and this
  // keeps the ordering rule in one readable place.
  if (tid == 0) {
    for (int i = 1; i < topk; ++i) {
      const int32_t v = dst[i];
      int j = i - 1;
      while (j >= 0 && dst[j] > v) {
        dst[j + 1] = dst[j];
        --j;
      }
      dst[j + 1] = v;
    }
    counts[t] = static_cast<int32_t>(topk);
  }
}

template <typename T>
void LaunchDsaIndexerLogits(cudaStream_t s, Tensor& logits, const Tensor& q, const Tensor& k,
                            const Tensor& weights, const Tensor& win_start,
                            const Tensor& win_end, const DsaIndexerLogitsArgs& args) {
  const int T_ = static_cast<int>(q.shape[0]);
  const int H = static_cast<int>(q.shape[1]);
  const int D = static_cast<int>(q.shape[2]);
  const int S = static_cast<int>(k.shape[0]);
  const int tiles = (S + kDsaKeyTile - 1) / kDsaKeyTile;
  const dim3 grid(static_cast<unsigned>(T_), static_cast<unsigned>(tiles));
  DsaIndexerLogitsKernel<T><<<grid, kDsaThreads, 0, s>>>(
      logits.Ptr<float>(), q.Ptr<T>(), k.Ptr<T>(), weights.Ptr<T>(),
      args.q_scale != nullptr ? args.q_scale->Ptr<float>() : nullptr,
      win_start.Ptr<int32_t>(), win_end.Ptr<int32_t>(), logits.stride[0], q.stride[0],
      q.stride[1], k.stride[0], weights.stride[0],
      args.q_scale != nullptr ? args.q_scale->stride[0] : 0, T_, S, H, D,
      args.softmax_scale * args.n_head_scale);
  CheckDsa(cudaGetLastError(), "dsa_indexer_logits launch");
}

void DsaIndexerLogitsCuda(Queue& q, Tensor& logits, const Tensor& q_states, const Tensor& k,
                          const Tensor& weights, const Tensor& win_start,
                          const Tensor& win_end, const DsaIndexerLogitsArgs& args) {
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  switch (q_states.dtype) {
    case DType::kF32:
      LaunchDsaIndexerLogits<float>(s, logits, q_states, k, weights, win_start, win_end, args);
      break;
    case DType::kBF16:
      LaunchDsaIndexerLogits<__nv_bfloat16>(s, logits, q_states, k, weights, win_start,
                                            win_end, args);
      break;
    case DType::kF16:
      LaunchDsaIndexerLogits<__half>(s, logits, q_states, k, weights, win_start, win_end, args);
      break;
    default: VT_CHECK(false, "cuda dsa_indexer_logits: unsupported dtype");
  }
}

void DsaTopkSelectCuda(Queue& q, Tensor& indices, Tensor& counts, const Tensor& logits,
                       const Tensor& win_start, const Tensor& win_end) {
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  const int T_ = static_cast<int>(logits.shape[0]);
  const int S = static_cast<int>(logits.shape[1]);
  const int topk = static_cast<int>(indices.shape[1]);
  DsaTopkSelectKernel<<<static_cast<unsigned>(T_), kTopkThreads, 0, s>>>(
      indices.Ptr<int32_t>(), counts.Ptr<int32_t>(), logits.Ptr<float>(),
      win_start.Ptr<int32_t>(), win_end.Ptr<int32_t>(), indices.stride[0], logits.stride[0], S,
      topk);
  CheckDsa(cudaGetLastError(), "dsa_topk_select launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kDsaIndexerLogits, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<DsaIndexerLogitsFn>(&DsaIndexerLogitsCuda)));
    RegisterOp(OpId::kDsaTopkSelect, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<DsaTopkSelectFn>(&DsaTopkSelectCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
