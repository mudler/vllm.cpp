// ROCm port of GLM-5.3-Flash's k-pool DSA indexer (BACKEND-ROCM; issue #2942).
//
// A mechanical CUDA-to-HIP translation of `src/vt/cuda/cuda_glm5_next.cu`
// (571 lines): the four kernels `KpoolMetaKernel` (:101), `KpoolKeepScanKernel`
// (:147), `KpoolPoolKernel` (:202), `KpoolSelectKernel` (:291), and the two
// launchers `Glm5NextKpoolCompressCuda` (:461) and `Glm5NextKpoolSelectCuda`
// (:522). Read side by side against the donor; the only edits are the spelling
// table in `.agents/specs/rocm-glm5-next-kpool.md` (cuda* -> hip*,
// CUDART_INF_F -> INFINITY, vt::cuda -> vt::rocm).
//
// The host reference `src/vllm/model_executor/models/glm5_next_dsa.cpp`
// (`GetPooledStates`, `SelectIndexerTopkFromPacked`) stays the ORACLE. There is
// no CPU provider for these two ops on purpose: registering the oracle under the
// same OpId would make the seam its own golden (cuda_glm5_next.cu:5-7).
//
// Spec: .agents/specs/rocm-glm5-next-kpool.md. Issue #2942. Staged slice: the
// kernels are unreached by the model forward until the compose (W9c-0, #2415)
// that would call them exists; the op-table entries this file supplies are
// reached directly through `vt::Glm5NextKpoolCompress` / `vt::Glm5NextKpoolSelect`.
//
// Ported from transformers v5.16.1 (`refs/tags/v5.16.1`), the lane pin
// `.agents/oracles/transformers.md` records for `model_type: glm5_next`.
#include <hip/hip_runtime.h>

#include <cfloat>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm kpool: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

inline hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}

constexpr int kMetaThreads = 256;
constexpr int kScanThreads = 1024;
constexpr int kPoolThreads = 128;
constexpr int kSelectThreads = 256;

__device__ __forceinline__ int64_t ClampIdx(int64_t v, int64_t hi) {
  return v < 0 ? 0 : (v > hi ? hi : v);
}

__global__ void FillI32Kernel(int32_t* p, int32_t v, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

// ── (A) `first_key` and per-pool validity ────────────────────────────────────
// `first_key` is `valid_keys.long().argmax(-1)` where any key is valid and
// `seq_len` where none is (`:938-942`). It is what makes a LEFT-PADDED row
// group differently from an unpadded one; pooling from slot 0 instead passes
// every unpadded fixture. A pool is valid iff every member is in range and
// valid (`:955-956`).
__global__ __launch_bounds__(kMetaThreads) void KpoolMetaKernel(
    int32_t* __restrict__ first_key, int32_t* __restrict__ raw_valid,
    const float* __restrict__ packed, int64_t kv_len, int64_t np, int kpool,
    int64_t row_stride, int64_t valid_off) {
  const int64_t b = blockIdx.x;
  const float* prow = packed + b * kv_len * row_stride;
  const int tid = static_cast<int>(threadIdx.x);

  __shared__ int red[kMetaThreads];
  int local = static_cast<int>(kv_len);
  for (int64_t j = tid; j < kv_len; j += kMetaThreads) {
    if (prow[j * row_stride + valid_off] != 0.0f) {
      local = static_cast<int>(j);
      break;
    }
  }
  red[tid] = local;
  __syncthreads();
  for (int stride = kMetaThreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) red[tid] = min(red[tid], red[tid + stride]);
    __syncthreads();
  }
  const int fk = red[0];
  if (tid == 0) first_key[b] = fk;
  __syncthreads();

  for (int64_t p = tid; p < np; p += kMetaThreads) {
    int all_valid = 1;
    for (int j = 0; j < kpool; ++j) {
      const int64_t idx = static_cast<int64_t>(fk) + p * kpool + j;
      // `safe_indices = pool_indices.clamp(0, seq_len - 1)` (`:948`) makes the
      // gather legal; `& (pool_indices < seq_len)` (`:955`) throws the
      // out-of-range members away again.
      const int64_t safe = ClampIdx(idx, kv_len - 1);
      const bool ok = idx < kv_len && prow[safe * row_stride + valid_off] != 0.0f;
      all_valid = all_valid && (ok ? 1 : 0);
    }
    raw_valid[b * np + p] = all_valid;
  }
}

// ── (B) the `keep` compaction ────────────────────────────────────────────────
// `keep = pool_valid.any(0)` (`:968`), then `[:, keep]` (`:970`) renumbers every
// pool after a dropped one. One block, a chunked exclusive scan with a running
// base, so the destination slot of every kept pool and `P` itself are produced
// without a host round trip.
__global__ __launch_bounds__(kScanThreads) void KpoolKeepScanKernel(
    int32_t* __restrict__ dst, int32_t* __restrict__ num_pools,
    const int32_t* __restrict__ raw_valid, int64_t batch, int64_t np) {
  __shared__ int sh[kScanThreads];
  __shared__ int base;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) base = 0;
  __syncthreads();

  for (int64_t chunk = 0; chunk < np; chunk += kScanThreads) {
    const int64_t p = chunk + tid;
    int keep = 0;
    if (p < np) {
      for (int64_t b = 0; b < batch; ++b) {
        if (raw_valid[b * np + p] != 0) {
          keep = 1;
          break;
        }
      }
    }
    sh[tid] = keep;
    __syncthreads();
    // Hillis-Steele inclusive scan, fixed shape, same order every launch.
    for (int off = 1; off < kScanThreads; off <<= 1) {
      int add = 0;
      if (tid >= off) add = sh[tid - off];
      __syncthreads();
      sh[tid] += add;
      __syncthreads();
    }
    if (p < np) dst[p] = keep != 0 ? (base + sh[tid] - 1) : -1;
    __syncthreads();
    if (tid == kScanThreads - 1) base += sh[kScanThreads - 1];
    __syncthreads();
  }
  if (tid == 0) num_pools[0] = base;
}

// ── (C) the LEARNED pool weighting ───────────────────────────────────────────
// `:959-965`: `index_head_dim` INDEPENDENT softmaxes, one per channel, over the
// pool's `index_kpool` members, with the learned intra-pool absolute-position
// embedding added to each member's gate score. It is NOT a mean, and a mean
// passes every shape check this file could carry.
//
// `torch.nan_to_num(logits.softmax(dim=2))` (`:962-964`) is the fully-invalid
// pool: every logit is `-inf`, the softmax is NaN, and the NaN is zeroed so the
// pool contributes nothing instead of poisoning the row. Here that is the
// `isfinite(mx)` guard, and the output buffer's pre-zero supplies the zero.
__device__ __forceinline__ bool MemberValid(const float* prow, int64_t idx, int64_t kv_len,
                                            int64_t row_stride, int64_t valid_off) {
  if (idx >= kv_len) return false;
  const int64_t safe = ClampIdx(idx, kv_len - 1);
  return prow[safe * row_stride + valid_off] != 0.0f;
}

__global__ __launch_bounds__(kPoolThreads) void KpoolPoolKernel(
    float* __restrict__ pool_keys, int32_t* __restrict__ pool_indices,
    int32_t* __restrict__ pool_valid, const float* __restrict__ packed,
    const float* __restrict__ ape, const int32_t* __restrict__ first_key,
    const int32_t* __restrict__ raw_valid, const int32_t* __restrict__ dst,
    int64_t kv_len, int64_t np, int head_dim, int kpool, int64_t row_stride,
    int64_t valid_off) {
  const int64_t p = blockIdx.x;
  const int64_t b = blockIdx.y;
  const int slot = dst[p];
  if (slot < 0) return;  // dropped by `keep`

  const float* prow = packed + b * kv_len * row_stride;
  const int64_t base = static_cast<int64_t>(first_key[b]) + p * kpool;

  if (threadIdx.x == 0) {
    pool_valid[b * np + slot] = raw_valid[b * np + p];
    // `pool_indices.masked_fill(~grouped_valid_keys, -1)` (`:957`). The -1 is
    // written BEFORE `pool_end` is read, so an invalid LAST member makes the
    // pool's visibility probe read slot 0 after the clamp at `:831`.
    for (int j = 0; j < kpool; ++j) {
      const int64_t idx = base + j;
      pool_indices[(b * np + slot) * kpool + j] =
          MemberValid(prow, idx, kv_len, row_stride, valid_off) ? static_cast<int32_t>(idx)
                                                                 : -1;
    }
  }

  // Three passes over the pool's members rather than three register arrays: the
  // member count is a RUNTIME value, so an array indexed by the loop variable
  // spills to local memory, and re-reading `index_kpool <= 16` gate scores is
  // cheaper than that spill. Every pass recomputes the same logit in the same
  // order, so the three agree by construction.
  for (int c = static_cast<int>(threadIdx.x); c < head_dim; c += kPoolThreads) {
    float mx = -INFINITY;
    for (int j = 0; j < kpool; ++j) {
      const int64_t idx = base + j;
      if (!MemberValid(prow, idx, kv_len, row_stride, valid_off)) continue;
      const int64_t safe = ClampIdx(idx, kv_len - 1);
      const float lg =
          __fadd_rn(prow[safe * row_stride + head_dim + c], ape[j * head_dim + c]);
      mx = fmaxf(mx, lg);
    }
    // `torch.nan_to_num(logits.softmax(dim=2))` (`:962-964`): a pool with NO
    // valid member softmaxes to NaN and is then zeroed, so it contributes
    // nothing rather than poisoning the row. The pre-zeroed buffer is the zero.
    if (!isfinite(mx)) continue;

    float denom = 0.0f;
    for (int j = 0; j < kpool; ++j) {
      const int64_t idx = base + j;
      // An invalid member's probability is exactly 0 and adding it changes no
      // f32 sum, which is why the host reference can add it unconditionally.
      if (!MemberValid(prow, idx, kv_len, row_stride, valid_off)) continue;
      const int64_t safe = ClampIdx(idx, kv_len - 1);
      const float lg =
          __fadd_rn(prow[safe * row_stride + head_dim + c], ape[j * head_dim + c]);
      denom = __fadd_rn(denom, expf(__fadd_rn(lg, -mx)));
    }

    float acc = 0.0f;
    for (int j = 0; j < kpool; ++j) {
      const int64_t idx = base + j;
      if (!MemberValid(prow, idx, kv_len, row_stride, valid_off)) continue;
      const int64_t safe = ClampIdx(idx, kv_len - 1);
      const float lg =
          __fadd_rn(prow[safe * row_stride + head_dim + c], ape[j * head_dim + c]);
      const float e = expf(__fadd_rn(lg, -mx));
      if (e == 0.0f) continue;  // underflowed: the host reference skips it too
      acc = __fadd_rn(acc, __fmul_rn(__fdiv_rn(e, denom), prow[safe * row_stride + c]));
    }
    pool_keys[(b * np + slot) * head_dim + c] = acc;
  }
}

// ── (D) score, select, expand, append the tail ───────────────────────────────
// One block per (batch row, query). `:823-828` is the score: a per-head dot,
// then the ReLU, THEN the head mix — the ReLU is before the mix, so a head that
// dislikes a pool contributes zero rather than a negative vote. `:831-837` is
// the candidacy: a pool is selectable only if its LAST member is visible to the
// query and all its members are valid. `:839-842` masks a non-candidate with
// `torch.finfo(dtype).min` and NOT with `-inf`, which is observable rather than
// cosmetic: a query whose candidates are all invalid still gets finite scores
// and a well-defined (then discarded) top-k.
//
// The selection reproduces `torch.topk`'s CPU rule — larger score wins, an exact
// tie breaks toward the SMALLER pool index — and emits in the DESCENDING order
// `topk` returns, because the emission order decides which member indices land
// in which columns and the tail is written at `select_k * index_kpool`.
__global__ __launch_bounds__(kSelectThreads) void KpoolSelectKernel(
    int32_t* __restrict__ topk_indices, float* __restrict__ index_scores,
    const float* __restrict__ q_states, const float* __restrict__ head_weights,
    const float* __restrict__ pool_keys, const int32_t* __restrict__ pool_indices,
    const int32_t* __restrict__ pool_valid, const int32_t* __restrict__ num_pools,
    const int32_t* __restrict__ valid_keys, const int32_t* __restrict__ q_mask,
    int64_t np, int64_t kv_len, int64_t seq_len, int n_heads, int head_dim, int kpool,
    int64_t index_topk, int64_t out_w, int64_t current_length, float softmax_scale,
    float head_scale, int always_tail) {
  const int64_t s = blockIdx.x;
  const int64_t b = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t r = b * seq_len + s;
  int32_t* dst = topk_indices + r * out_w;

  // `F.pad(..., value=-1)` (`:870`) as the initial state, so every column this
  // kernel does not write is already the invalid sentinel.
  for (int64_t i = tid; i < out_w; i += kSelectThreads) dst[i] = -1;
  __syncthreads();

  const int P = num_pools[0];
  const int64_t q_pos = current_length - seq_len + s;

  __shared__ float best_val[kSelectThreads];
  __shared__ int best_idx[kSelectThreads];
  __shared__ int red[kSelectThreads];
  __shared__ float pick_val;
  __shared__ int pick_idx;

  const float* qrow = q_states + r * n_heads * head_dim;
  const float* wrow = head_weights + r * n_heads;

  // The score is computed for EVERY query row, including a padded one. Upstream
  // masks `topk_indices` at `:873` and never masks `index_scores` at `:828`, so
  // returning early on the padding mask here would leave a padded row's scores
  // zero and disagree with the reference on exactly the rows a left-padded
  // fixture adds.
  for (int p = tid; p < P; p += kSelectThreads) {
    float acc = 0.0f;
    const float* pk = pool_keys + (b * np + p) * head_dim;
    for (int h = 0; h < n_heads; ++h) {
      const float* qh = qrow + static_cast<int64_t>(h) * head_dim;
      float dot = 0.0f;
      for (int c = 0; c < head_dim; ++c) dot = __fadd_rn(dot, __fmul_rn(qh[c], pk[c]));
      const float relu = fmaxf(0.0f, __fmul_rn(dot, softmax_scale));
      acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(wrow[h], head_scale), relu));
    }
    index_scores[r * np + p] = acc;
  }
  __syncthreads();

  // `topk_indices.masked_fill(~attention_mask[..., None], -1)` (`:873`): a padded
  // query selects nothing and keeps the -1 prefill. The whole block shares one
  // query, so this return is uniform.
  if (q_mask[r] == 0) return;

  if (tid == 0) {
    pick_val = INFINITY;
    pick_idx = -1;
  }
  __syncthreads();

  const int select_k = min(static_cast<int>(index_topk / kpool), P);
  for (int slot = 0; slot < select_k; ++slot) {
    const float bar_v = pick_val;
    const int bar_i = pick_idx;
    float bv = -INFINITY;
    int bi = -1;
    for (int p = tid; p < P; p += kSelectThreads) {
      // Candidacy is recomputed rather than cached: it is two loads, and a
      // `[B, S, P]` candidate buffer is the allocation this family exists to
      // avoid.
      const int last = pool_indices[(b * np + p) * kpool + kpool - 1];
      const int64_t safe = ClampIdx(last, kv_len - 1);
      const bool vis = safe <= q_pos && valid_keys[b * kv_len + safe] != 0;
      const bool cand = vis && pool_valid[b * np + p] != 0;
      const float v = cand ? index_scores[r * np + p] : -FLT_MAX;
      // Strictly after the previous pick in (score desc, index asc) order.
      const bool avail = bar_i < 0 || v < bar_v || (v == bar_v && p > bar_i);
      if (!avail) continue;
      if (bi < 0 || v > bv || (v == bv && p < bi)) {
        bv = v;
        bi = p;
      }
    }
    best_val[tid] = bv;
    best_idx[tid] = bi;
    __syncthreads();
    for (int stride = kSelectThreads >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float ov = best_val[tid + stride];
        const int oi = best_idx[tid + stride];
        const bool better = oi >= 0 && (best_idx[tid] < 0 || ov > best_val[tid] ||
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
      const int p = pick_idx;
      bool keep = false;
      if (p >= 0) {
        const int last = pool_indices[(b * np + p) * kpool + kpool - 1];
        const int64_t safe = ClampIdx(last, kv_len - 1);
        const bool vis = safe <= q_pos && valid_keys[b * kv_len + safe] != 0;
        keep = vis && pool_valid[b * np + p] != 0;
      }
      // `selected_valid` masks the WHOLE expanded pool to -1 (`:853,:859-862`):
      // a pool the top-k had to pick because nothing better existed does not
      // become a real selection.
      for (int m = 0; m < kpool; ++m) {
        const int64_t col = static_cast<int64_t>(slot) * kpool + m;
        if (col >= out_w) break;
        dst[col] = keep ? pool_indices[(b * np + p) * kpool + m] : -1;
      }
    }
    __syncthreads();
  }

  // `append_visible_tail` (`:972-1022`). `max_tail_width == 0` at
  // `index_kpool == 1` returns unchanged (`:985-986`).
  if (always_tail == 0 || kpool <= 1) return;

  int local_count = 0;
  int local_first = static_cast<int>(kv_len);
  for (int64_t j = tid; j < kv_len; j += kSelectThreads) {
    const bool valid = valid_keys[b * kv_len + j] != 0;
    if (valid && j <= q_pos) ++local_count;
    if (valid && j < local_first) local_first = static_cast<int>(j);
  }
  red[tid] = local_count;
  __syncthreads();
  for (int stride = kSelectThreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  const int visible_count = red[0];
  __syncthreads();
  red[tid] = local_first;
  __syncthreads();
  for (int stride = kSelectThreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) red[tid] = min(red[tid], red[tid + stride]);
    __syncthreads();
  }
  const int first = red[0];

  if (tid == 0) {
    const int tail_count = visible_count % kpool;
    const int64_t tail_start = static_cast<int64_t>(first) + visible_count - tail_count;
    for (int j = 0; j < kpool - 1; ++j) {
      const int64_t col = static_cast<int64_t>(select_k) * kpool + j;
      if (col >= out_w) break;
      const int64_t idx = tail_start + j;
      // `tail_valid` drops the fill positions and anything past the cache
      // (`:1013`); `tail_visible` re-checks the padding mask (`:1016-1017`).
      const bool valid = j < tail_count && idx < kv_len;
      const int64_t safe = ClampIdx(idx, kv_len - 1);
      const bool vis = safe <= q_pos && valid_keys[b * kv_len + safe] != 0;
      dst[col] = (valid && vis) ? static_cast<int32_t>(idx) : -1;
    }
  }
}

// ── the launchers ────────────────────────────────────────────────────────────

}  // namespace

void Glm5NextKpoolCompressRocm(Queue& q, Tensor& pool_keys, Tensor& pool_indices,
                               Tensor& pool_valid, Tensor& num_pools, const Tensor& packed,
                               const Tensor& ape) {
  hipStream_t s = AsStream(q);
  const int64_t batch = packed.shape[0];
  const int64_t kv_len = packed.shape[1];
  const int64_t row_stride = packed.shape[2];
  const int kpool = static_cast<int>(ape.shape[0]);
  const int head_dim = static_cast<int>(ape.shape[1]);
  const int64_t np = pool_keys.shape[1];

  // Deterministic slack. Only `[0, P)` carries meaning, and leaving the rest
  // uninitialised would make a downstream read of it undefined instead of
  // obviously empty.
  Check(hipMemsetAsync(pool_keys.data, 0,
                       static_cast<size_t>(batch * np * head_dim) * sizeof(float), s),
        "memset pool_keys");
  Check(hipMemsetAsync(pool_valid.data, 0,
                       static_cast<size_t>(batch * np) * sizeof(int32_t), s),
        "memset pool_valid");
  {
    const int64_t n = batch * np * kpool;
    if (n > 0) {
      const unsigned grid = static_cast<unsigned>((n + 255) / 256);
      FillI32Kernel<<<grid, 256, 0, s>>>(pool_indices.Ptr<int32_t>(), -1, n);
      Check(hipGetLastError(), "fill pool_indices launch");
    }
  }

  // Per-call scratch on the stream, the shape `cuda_mla_prefill.cu:125` uses. A
  // pooled workspace is a later brick; this family is not on a hot path yet
  // because nothing calls it (spec O36).
  void* scratch = nullptr;
  const size_t scratch_bytes =
      (static_cast<size_t>(batch) + static_cast<size_t>(batch * np) +
       static_cast<size_t>(np)) *
      sizeof(int32_t);
  Check(hipMallocAsync(&scratch, scratch_bytes, s), "hipMallocAsync kpool scratch");
  int32_t* first_key = static_cast<int32_t*>(scratch);
  int32_t* raw_valid = first_key + batch;
  int32_t* dst = raw_valid + batch * np;

  KpoolMetaKernel<<<static_cast<unsigned>(batch), kMetaThreads, 0, s>>>(
      first_key, raw_valid, packed.Ptr<float>(), kv_len, np, kpool, row_stride,
      static_cast<int64_t>(2 * head_dim));
  Check(hipGetLastError(), "kpool meta launch");

  KpoolKeepScanKernel<<<1, kScanThreads, 0, s>>>(dst, num_pools.Ptr<int32_t>(), raw_valid,
                                                 batch, np);
  Check(hipGetLastError(), "kpool keep-scan launch");

  const dim3 pool_grid(static_cast<unsigned>(np), static_cast<unsigned>(batch));
  KpoolPoolKernel<<<pool_grid, kPoolThreads, 0, s>>>(
      pool_keys.Ptr<float>(), pool_indices.Ptr<int32_t>(), pool_valid.Ptr<int32_t>(),
      packed.Ptr<float>(), ape.Ptr<float>(), first_key, raw_valid, dst, kv_len, np,
      head_dim, kpool, row_stride, static_cast<int64_t>(2 * head_dim));
  Check(hipGetLastError(), "kpool pool launch");

  Check(hipFreeAsync(scratch, s), "hipFreeAsync kpool scratch");
}

void Glm5NextKpoolSelectRocm(Queue& q, Tensor& topk_indices, Tensor& index_scores,
                             const Tensor& q_states, const Tensor& head_weights,
                             const Tensor& pool_keys, const Tensor& pool_indices,
                             const Tensor& pool_valid, const Tensor& num_pools,
                             const Tensor& valid_keys, const Tensor& q_mask,
                             const Glm5NextKpoolSelectArgs& args) {
  hipStream_t s = AsStream(q);
  const int64_t batch = q_states.shape[0];
  const int64_t seq_len = q_states.shape[1];
  const int n_heads = static_cast<int>(q_states.shape[2]);
  const int head_dim = static_cast<int>(q_states.shape[3]);
  const int64_t np = pool_keys.shape[1];
  const int kpool = static_cast<int>(pool_indices.shape[2]);
  const int64_t kv_len = valid_keys.shape[1];
  const int64_t out_w = topk_indices.shape[2];

  Check(hipMemsetAsync(index_scores.data, 0,
                       static_cast<size_t>(batch * seq_len * np) * sizeof(float), s),
        "memset index_scores");

  // `weights = self.weights_proj(hidden) * (self.n_heads ** -0.5)` (`:827`).
  // Computed on the HOST in double and narrowed once, which is exactly what the
  // reference does (`glm5_next_dsa.cpp:438`). `rsqrtf` is an approximation and
  // would put a device-only rounding into a constant both arms share.
  const float head_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(n_heads)));
  const dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(batch));
  KpoolSelectKernel<<<grid, kSelectThreads, 0, s>>>(
      topk_indices.Ptr<int32_t>(), index_scores.Ptr<float>(), q_states.Ptr<float>(),
      head_weights.Ptr<float>(), pool_keys.Ptr<float>(), pool_indices.Ptr<int32_t>(),
      pool_valid.Ptr<int32_t>(), num_pools.Ptr<int32_t>(), valid_keys.Ptr<int32_t>(),
      q_mask.Ptr<int32_t>(), np, kv_len, seq_len, n_heads, head_dim, kpool,
      args.index_topk, out_w, args.current_length, args.softmax_scale, head_scale,
      args.always_select_tail ? 1 : 0);
  Check(hipGetLastError(), "kpool select launch");
}

}  // namespace vt::rocm
