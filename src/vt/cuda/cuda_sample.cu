// Ported from: vllm/v1/sample/ops/topk_topp_sampler.py + vllm/v1/sample/sampler.py @ e24d1b24.
//
// Correctness-grade CUDA kernels for the V1 sampling ops (M1.7 Task 2), mirroring
// the CPU reference (src/vt/cpu/cpu_sample.cpp) element for element:
//   - apply_temperature: grid-stride, per-row temp with the eps greedy guard.
//   - greedy_argmax / random_sample: ONE BLOCK per row, single-threaded scan so
//     the lowest-index tie-break matches torch.argmax / the CPU reference exactly.
//   - compute_probs / compute_logprobs: block-per-row max-subtracted softmax.
//   - apply_top_k_top_p: SORT-FREE block-cooperative pivot-bracket THRESHOLD
//     search (one block per row), mirroring flashinfer's TopK/TopPRenormProb
//     (sampling.cuh). It replaces the old per-row full-vocab thrust::stable_sort
//     + <<<n,1>>> single-thread top-p cumsum + blocking cudaStreamSynchronize.
//     The kept set is identical to apply_top_k_top_p_pytorch for distinct logits
//     (validated by the scalar mirror + cross-check in tests/vt/test_ops_sample.cpp).
// NOTE: this file is built + verified on dgx.casa (the CI box is CPU-only); the
// CUDA parity tests are HasCuda-guarded.
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/radix_topk.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;
constexpr float kNegInf = -INFINITY;

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

// Deterministic RNG shared with cpu_sample.cpp (bit-identical integer mixing).
__device__ inline uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

__device__ inline double ExpNoise(uint64_t seed, int64_t row, int64_t col) {
  const uint64_t row_key = SplitMix64(seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(row));
  const uint64_t r = SplitMix64(row_key + static_cast<uint64_t>(col));
  const double u = static_cast<double>((r >> 11) + 1ULL) * (1.0 / 9007199254740993.0);
  return -log(u);
}

// --- apply_temperature ------------------------------------------------------
__global__ void ApplyTemperatureKernel(float* logits, const float* temp, int64_t n, int64_t v,
                                       bool all_random) {
  const int64_t total = n * v;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step) {
    const int64_t row = idx / v;
    float t = temp[row];
    if (!all_random && t < kSamplingEps) t = 1.0f;
    logits[idx] /= t;
  }
}

void ApplyTemperatureCuda(Queue& q, Tensor& logits, const Tensor& temp, bool all_random) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyTemperatureKernel<<<GridFor(n * v), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), temp.Ptr<float>(), n, v, all_random);
  Check(cudaGetLastError(), "apply_temperature launch");
}

// --- greedy_argmax (two-pass multi-block reduction, exact lowest-index tie) ---
// The old kernel was a single block of a single thread scanning the whole vocab
// (~151k) serially -- ~7.5 ms/token on the decode path (a single-SM, latency-
// bound scan). Here we grid-stride the vocab across many blocks (pass 1 -> per-
// block partials) and reduce the partials (pass 2), keeping torch.argmax
// semantics: the maximum value wins; on ties the LOWEST index wins. The reduce
// operator compares the true global index (not thread/block order) so the tie-
// break is order-independent. Unfilled lanes carry (-inf, INT64_MAX) so a real
// index -- even one whose logit is -inf (all-masked row) -- always beats an empty
// lane, yielding index 0 for an all-(-inf) row, exactly like the CPU reference.
__device__ inline void ArgReduce(float& av, int64_t& ai, float bv, int64_t bi) {
  if (bv > av || (bv == av && bi < ai)) {
    av = bv;
    ai = bi;
  }
}

constexpr int64_t kArgSentinel = 0x7fffffffffffffffLL;  // INT64_MAX

__global__ void ArgmaxPartialKernel(float* part_val, int64_t* part_idx, const float* logits,
                                    int64_t v, int blocks_per_row) {
  const int64_t row = blockIdx.y;
  const int blk = blockIdx.x;
  const float* r = logits + row * v;
  __shared__ float sv[kBlock];
  __shared__ int64_t si[kBlock];

  float bv = kNegInf;
  int64_t bi = kArgSentinel;
  const int64_t stride = static_cast<int64_t>(blocks_per_row) * blockDim.x;
  for (int64_t j = static_cast<int64_t>(blk) * blockDim.x + threadIdx.x; j < v; j += stride)
    ArgReduce(bv, bi, r[j], j);

  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s)
      ArgReduce(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + s], si[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    part_val[row * blocks_per_row + blk] = sv[0];
    part_idx[row * blocks_per_row + blk] = si[0];
  }
}

__global__ void ArgmaxFinalKernel(int64_t* out, const float* part_val, const int64_t* part_idx,
                                  int blocks_per_row) {
  const int64_t row = blockIdx.x;
  const float* pv = part_val + row * blocks_per_row;
  const int64_t* pi = part_idx + row * blocks_per_row;
  __shared__ float sv[kBlock];
  __shared__ int64_t si[kBlock];

  float bv = kNegInf;
  int64_t bi = kArgSentinel;
  for (int j = threadIdx.x; j < blocks_per_row; j += blockDim.x) ArgReduce(bv, bi, pv[j], pi[j]);

  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s)
      ArgReduce(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + s], si[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = (si[0] == kArgSentinel) ? 0 : si[0];
}

// Persistent scratch for the argmax partials -- grown on demand and kept alive
// (a few KB), so the decode path never pays a cudaMalloc/cudaFree per token.
float* g_argmax_val = nullptr;
int64_t* g_argmax_idx = nullptr;
size_t g_argmax_cap = 0;  // capacity in elements

void EnsureArgmaxScratch(size_t elems) {
  if (elems <= g_argmax_cap) return;
  if (g_argmax_val) cudaFree(g_argmax_val);
  if (g_argmax_idx) cudaFree(g_argmax_idx);
  Check(cudaMalloc(&g_argmax_val, elems * sizeof(float)), "argmax scratch val");
  Check(cudaMalloc(&g_argmax_idx, elems * sizeof(int64_t)), "argmax scratch idx");
  g_argmax_cap = elems;
}

// Legacy single-block single-thread argmax (bit-exact reference). Retained behind
// VT_FAST_ARGMAX=0 for same-binary A/B against the two-pass kernel above.
__global__ void GreedyArgmaxKernelSlow(int64_t* out, const float* logits, int64_t v) {
  const int64_t row = blockIdx.x;
  if (threadIdx.x != 0) return;
  const float* r = logits + row * v;
  int64_t best = 0;
  float best_v = r[0];
  for (int64_t j = 1; j < v; ++j) {
    if (r[j] > best_v) {
      best_v = r[j];
      best = j;
    }
  }
  out[row] = best;
}

bool FastArgmaxEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FAST_ARGMAX");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

void GreedyArgmaxCuda(Queue& q, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  cudaStream_t s = AsStream(q);

  if (!FastArgmaxEnabled()) {
    GreedyArgmaxKernelSlow<<<static_cast<unsigned>(n), 1, 0, s>>>(token_ids.Ptr<int64_t>(),
                                                                 logits.Ptr<float>(), v);
    Check(cudaGetLastError(), "greedy_argmax launch (slow)");
    return;
  }

  // One block per kBlock vocab elements, capped so pass 2 fits a single block.
  int bpr = static_cast<int>((v + kBlock - 1) / kBlock);
  if (bpr > kBlock) bpr = kBlock;  // pass 2 reduces bpr partials with kBlock threads
  if (bpr < 1) bpr = 1;

  EnsureArgmaxScratch(static_cast<size_t>(n) * bpr);
  dim3 grid1(static_cast<unsigned>(bpr), static_cast<unsigned>(n));
  ArgmaxPartialKernel<<<grid1, kBlock, 0, s>>>(g_argmax_val, g_argmax_idx, logits.Ptr<float>(), v,
                                               bpr);
  ArgmaxFinalKernel<<<static_cast<unsigned>(n), kBlock, 0, s>>>(token_ids.Ptr<int64_t>(),
                                                                g_argmax_val, g_argmax_idx, bpr);
  Check(cudaGetLastError(), "greedy_argmax launch");
}

// --- compute_probs / compute_logprobs (block-per-row softmax) ---------------
__global__ void SoftmaxKernel(float* out, const float* logits, int64_t v, bool log_softmax) {
  const int64_t row = blockIdx.x;
  const float* r = logits + row * v;
  float* o = out + row * v;
  __shared__ float red[kBlock];

  float m = kNegInf;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) m = fmaxf(m, r[j]);
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float mx = red[0];
  __syncthreads();

  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) acc += expf(r[j] - mx);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float sum = red[0];
  const float lse = mx + logf(sum);
  __syncthreads();

  for (int64_t j = threadIdx.x; j < v; j += blockDim.x)
    o[j] = log_softmax ? (r[j] - lse) : expf(r[j] - mx) / sum;
}

void ComputeProbsCuda(Queue& q, Tensor& probs, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  SoftmaxKernel<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(
      probs.Ptr<float>(), logits.Ptr<float>(), v, /*log_softmax=*/false);
  Check(cudaGetLastError(), "compute_probs launch");
}

void ComputeLogprobsCuda(Queue& q, Tensor& logprobs, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  SoftmaxKernel<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(
      logprobs.Ptr<float>(), logits.Ptr<float>(), v, /*log_softmax=*/true);
  Check(cudaGetLastError(), "compute_logprobs launch");
}

// --- random_sample (single-threaded per row: exact tie-break + same RNG) -----
__global__ void RandomSampleKernel(int64_t* out, const float* probs, const int64_t* seeds,
                                   int64_t v) {
  const int64_t row = blockIdx.x;
  if (threadIdx.x != 0) return;
  const float* r = probs + row * v;
  const uint64_t seed = static_cast<uint64_t>(seeds[row]);
  int64_t best = 0;
  float best_v = kNegInf;
  for (int64_t j = 0; j < v; ++j) {
    const float qn = static_cast<float>(ExpNoise(seed, row, j));
    const float score = r[j] / qn;
    if (score > best_v) {
      best_v = score;
      best = j;
    }
  }
  out[row] = best;
}

void RandomSampleCuda(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds) {
  const int64_t n = probs.shape[0], v = probs.shape[1];
  if (n == 0 || v == 0) return;
  RandomSampleKernel<<<static_cast<unsigned>(n), 1, 0, AsStream(q)>>>(
      token_ids.Ptr<int64_t>(), probs.Ptr<float>(), seeds.Ptr<int64_t>(), v);
  Check(cudaGetLastError(), "random_sample launch");
}

// --- apply_top_k_top_p (SORT-FREE block-cooperative threshold search) --------
// Each row is masked by ONE block. The top-k value-threshold (k-th largest logit)
// and the top-p exp-threshold are found with the two-pivot bracket search that
// flashinfer's TopK/TopPRenormProb use (sampling.cuh): each iteration evaluates
// count(>pivot) / sum(>pivot) cooperatively and snaps the bracket to real array
// values via min_gt_low / max_le_high, so it converges to an EXACT array-value
// threshold (empirically <=17 iters over a 152k vocab). No sort, no serial scan,
// no host round-trip, and (unlike the old path) no blocking Synchronize — the op
// is fully async on the stream, ordered before the downstream softmax/sample.
//
// Kept set == apply_top_k_top_p_pytorch for DISTINCT logits (validated by the
// scalar mirror + cross-check in tests/vt/test_ops_sample.cpp). The only
// divergence is the measure-zero exact-tie-straddling-boundary case, where this
// path keeps/drops whole tie groups atomically (like flashinfer) rather than
// splitting them by stable index order; real f32 logits are effectively
// continuous, so this never fires.
constexpr int kThreshMaxIter = 64;  // >3x the observed worst case; converges by pinning

// Block reductions over a kBlock-thread block (kBlock is a power of two). Each
// returns the reduced value broadcast to all threads. `s` is a kBlock-wide shared
// scratch; callers must not hold a live value in it across the call.
__device__ inline float BlockRedMaxF(float v, float* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] = fmaxf(s[t], s[t + o]);
    __syncthreads();
  }
  const float r = s[0];
  __syncthreads();
  return r;
}
__device__ inline float BlockRedMinF(float v, float* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] = fminf(s[t], s[t + o]);
    __syncthreads();
  }
  const float r = s[0];
  __syncthreads();
  return r;
}
__device__ inline float BlockRedSumF(float v, float* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] += s[t + o];
    __syncthreads();
  }
  const float r = s[0];
  __syncthreads();
  return r;
}
__device__ inline int BlockRedSumI(int v, int* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] += s[t + o];
    __syncthreads();
  }
  const int r = s[0];
  __syncthreads();
  return r;
}

__global__ void ApplyTopKTopPRowKernel(float* logits, const int32_t* k_arr, const float* p_arr,
                                       int64_t v) {
  const int64_t row = blockIdx.x;
  float* r = logits + row * v;
  const int t = threadIdx.x;

  __shared__ float red[kBlock];
  __shared__ int redi[kBlock];
  __shared__ float sh_thr_k;
  __shared__ float sh_low;

  const bool has_k = (k_arr != nullptr);
  const bool has_p = (p_arr != nullptr);
  const int32_t k = has_k ? k_arr[row] : 0;
  const float p = has_p ? p_arr[row] : 1.0f;

  // Row max (softmax reference) and min over FINITE logits (the top-k search
  // lower bracket must be finite — upstream masks (min_p / bad_words / allowed
  // ids) can leave -inf logits, and an -inf bracket would stall the pivot
  // arithmetic). mn == global min in the common (no pre-masked) case.
  float lmax = kNegInf, lmin = INFINITY;
  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    lmax = fmaxf(lmax, x);
    if (x != kNegInf) lmin = fminf(lmin, x);
  }
  const float mx = BlockRedMaxF(lmax, red);
  const float mn = BlockRedMinF(lmin, red);  // +inf if every logit is -inf

  // ---- top-k: thr_k = k-th largest logit; keep {r[j] >= thr_k} (ties kept) ----
  // Active only for 1 <= k < v (k>=v is a no-op; k<1 invalid, guarded). Mirrors
  // apply_top_k_only / the torch top_k_mask (logits_sort < sorted[v-k]).
  const bool topk_active = has_k && (k >= 1) && (static_cast<int64_t>(k) < v);
  float thr_k = kNegInf;  // -inf => every token survives
  if (topk_active) {
    int lc = 0;
    for (int64_t j = t; j < v; j += kBlock)
      if (r[j] > mn) ++lc;
    const int cnt_gt_min = BlockRedSumI(lc, redi);
    if (cnt_gt_min < k) {
      thr_k = mn;  // k-th largest == global min => masks nothing
    } else {
      // Find `low` = largest value with count(r>low) >= k; the k-th largest is the
      // smallest value strictly greater than `low` (min_gt_low at convergence).
      float low = mn, high = mx, min_gt_low = mx, max_le_high = mn;
      for (int iter = 0; iter < kThreshMaxIter; ++iter) {
        const float p0 = (2.0f * low + high) / 3.0f;
        const float p1 = (low + 2.0f * high) / 3.0f;
        int l0 = 0, l1 = 0;
        float lmglow = high, lmleh = low;
        for (int64_t j = t; j < v; j += kBlock) {
          const float x = r[j];
          if (x > p0) ++l0;
          if (x > p1) ++l1;
          if (x > low) lmglow = fminf(lmglow, x);
          if (x <= high) lmleh = fmaxf(lmleh, x);
        }
        const int c0 = BlockRedSumI(l0, redi);
        const int c1 = BlockRedSumI(l1, redi);
        min_gt_low = BlockRedMinF(lmglow, red);
        max_le_high = BlockRedMaxF(lmleh, red);
        if (c1 >= k) {
          low = p1;
        } else if (c0 >= k) {
          low = p0;
          high = fminf(p1, max_le_high);
        } else {
          high = fminf(p0, max_le_high);
        }
        if (min_gt_low == max_le_high) break;
      }
      thr_k = min_gt_low;
    }
  }
  if (t == 0) sh_thr_k = thr_k;
  __syncthreads();
  thr_k = sh_thr_k;
  // survivor(j) := r[j] >= thr_k

  // ---- top-p over survivors: keep {survivor && e_j > low}, e_j = exp(r-mx) ----
  // denom / target normalize over the TOP-K SURVIVORS (matching the torch path,
  // whose top-p softmax runs on the top-k-masked logits). p >= 1 is a no-op.
  float low = -1.0f;  // on e in (0,1]; -1 => keep every survivor
  const bool topp_active = has_p && (p < 1.0f);
  if (topp_active) {
    float lden = 0.0f;
    for (int64_t j = t; j < v; j += kBlock)
      if (r[j] >= thr_k) lden += expf(r[j] - mx);
    const float denom = BlockRedSumF(lden, red);
    const float target = p * denom;  // survivor mass to keep (>= p fraction)
    float lo = 0.0f, hi = 1.0f, min_gt_low = 1.0f, max_le_high = 0.0f;
    for (int iter = 0; denom > 0.0f && iter < kThreshMaxIter; ++iter) {
      const float p0 = (2.0f * lo + hi) / 3.0f;
      const float p1 = (lo + 2.0f * hi) / 3.0f;
      float la0 = 0.0f, la1 = 0.0f, lmglow = hi, lmleh = lo;
      for (int64_t j = t; j < v; j += kBlock) {
        if (r[j] < thr_k) continue;  // non-survivor: excluded from the top-p mass
        const float e = expf(r[j] - mx);
        if (e > p0) la0 += e;
        if (e > p1) la1 += e;
        if (e > lo) lmglow = fminf(lmglow, e);
        if (e <= hi) lmleh = fmaxf(lmleh, e);
      }
      const float a0 = BlockRedSumF(la0, red);
      const float a1 = BlockRedSumF(la1, red);
      min_gt_low = BlockRedMinF(lmglow, red);
      max_le_high = BlockRedMaxF(lmleh, red);
      if (a1 >= target) {
        lo = p1;
      } else if (a0 >= target) {
        lo = p0;
        hi = fminf(p1, max_le_high);
      } else {
        hi = fminf(p0, max_le_high);
      }
      if (min_gt_low == max_le_high) break;
    }
    low = lo;
  }
  if (t == 0) sh_low = low;
  __syncthreads();
  low = sh_low;

  // ---- final in-place mask ----
  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    const bool keep = (x >= thr_k) && (low < 0.0f || expf(x - mx) > low);
    if (!keep) r[j] = kNegInf;
  }
}

void ApplyTopKTopPCuda(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  // Both k-only and k+p go through the same threshold kernel; k-only is the torch
  // apply_top_k_only set (k-th largest, ties kept). Fully async — no Synchronize.
  ApplyTopKTopPRowKernel<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), k != nullptr ? k->Ptr<int32_t>() : nullptr,
      p != nullptr ? p->Ptr<float>() : nullptr, v);
  Check(cudaGetLastError(), "top_k_top_p launch");
}

// --- top-k that EMITS the surviving (id, value) pairs -----------------------
// SPEC-DFLASH2 W3 / D2 revisited (#1314,
// [#1867](https://github.com/mudler/vllm.cpp/issues/1867)). The CUDA arm of
// vt::TopKValuesIndices, whose contract, tie-break and upstream anchor live on
// `TopKValuesIndicesArgs` (include/vt/ops.h).
// `src/vt/cpu/cpu_ops.cpp::TopKValuesIndicesKernel` is the authoritative
// reference and is UNCHANGED by #1867 — it stays a `std::partial_sort` under the
// explicit float comparator, so the two arms still answer by different routes
// and an agreement between them is still evidence.
//
// WHAT #1867 CHANGED, and why D2 no longer refuses the port. This kernel used to
// find its threshold by TERNARY BISECTION over float VALUES under a
// `kThreshMaxIter = 64` budget, and every one of those iterations was a full
// pass over a 248320-wide row. #1857's kernel table, measured on `dgx:gpu0` with
// nsys against SGLang on the identical checkpoint and workload, read
// `TopKValuesIndicesRowKernel` at **683 us/step** for 8 rows x 248320, K=16
// where SGLang's `RadixTopKKernel_Unified` did the same work in **40 us** — the
// fourth-largest per-step lever on that table at +0.65 ms/step. D2 named its own
// revisit condition ("Revisit only if W3 measures the top-k as the selector's
// dominant cost here, as it is upstream"), and that is the measurement.
//
// The search is now the RADIX select vLLM itself dispatches to. `include/vt/
// radix_topk.h` carries the ported arithmetic and the upstream anchors; this
// kernel carries the parallelism and no arithmetic. Four rounds of 8-bit
// histogram over a monotone key fix the k-th largest EXACTLY — no bracket, no
// tolerance, no iteration budget.
//
// HOW MANY OF THOSE ROUNDS READ GLOBAL MEMORY IS DATA, and saying otherwise was
// this kernel's first defect. Rounds 0 and 1 always read the row. Whether rounds
// 2 and 3 do depends on how many columns share the k-th largest key's round-0
// digit, and that digit is the sign bit plus the top SEVEN exponent bits, so one
// bucket spans TWO exponents. On the 8 x 248320 shape #1867 targets it holds
// about 93000 columns — 45x the 2048-entry buffer the first version of this file
// sized, and 5.7x `kRadixTopKCandCapMax`. `include/vt/radix_topk.h` carries the
// measurement for every row the tests run. So there are three arms and the
// kernel names all three:
//
//   TWO global passes   the round-0 bucket fits the candidate buffer; rounds 2
//                       and 3 read it out of shared.
//   THREE global passes it does not, but the ROUND-1 bucket does — which the
//                       kernel knows exactly, from the histogram it already
//                       built — so one more pass re-compacts to that. Every
//                       production and tie-dense row takes this arm, narrowing
//                       93000 columns to between 452 and 522.
//   SIX global passes   neither fits. An all-equal or a -inf-saturated row is
//                       the real case. Rounds 2 and 3, the winner collect and
//                       the tie fill each read the row again. Correct, and it
//                       costs the passes back.
//
// The three-arm shape is FlashInfer's: its coarse filter compacts what equals
// the threshold bin and emits what exceeds it
// (`filter_and_add_to_histogram`, `flashinfer/topk.cuh:2530-2551`), each refine
// round re-compacts the same way into the next buffer
// (`collect_with_threshold_non_last_round`, `topk.cuh:2566-2592`), and an
// overflow of either sets `s_refine_overflow` and takes a slow path that
// re-reads the input (`topk.cuh:2631`, `2692-2720`).
//
// WHERE IT DELIBERATELY IS NOT FlashInfer. `RadixTopKKernel_Unified` splits each
// row across several CTAs and joins them with an acquire/release grid barrier
// over a persistent `RadixRowState` workspace. This kernel keeps ONE CTA PER ROW
// and therefore needs no barrier, no workspace and no cooperative launch. That
// is the part of D2 that still stands: the general kernel's multi-CTA machinery
// is 3380 lines for a shape that is 8 rows here. The cost of the simplification
// is occupancy — 8 CTAs on a 48-SM part — and it is named as an owed lever
// rather than hidden, because it is the residual between this kernel and the
// 40 us SGLang measured.
//
// THE TIE IS STILL THE WHOLE DIFFICULTY, and it is now easier rather than
// harder. The radix prefix converges to the k-th largest KEY exactly, so the set
// `{key >= pivot}` is the kept set and holds at least k members: everything
// strictly above the pivot survives unconditionally (fewer than k of them, by
// the definition of the k-th largest) and the remaining slots go to the
// LOWEST-INDEXED elements whose key EQUALS the pivot. That is the (value DESC,
// index ASC) order `include/vt/ops.h` pins. The old kernel additionally had to
// DROP its threshold when a tie group ran out — a -inf-saturated row was the
// real case — and the radix search cannot reach that state at all: `k <= V -
// num_org_vocab_padding` is enforced by `src/vt/ops.cpp::TopKValuesIndices`, so
// `count(key >= pivot) >= k` holds by construction of the prefix.
//
// WHAT IS GATED IN THIS REPOSITORY, AND WHAT IS NOT. There is no `nvcc` on the
// authoring host, so nothing below has ever compiled; `cuda-fat-build` in CI
// compiles this file and runs nothing from it. Two of the three things this
// kernel is made of ARE gated, on the host, by
// `tests/vt/test_ops_radix_topk.cpp`:
//
//   * the ARITHMETIC — `include/vt/radix_topk.h`'s key, digits, prefix test,
//     bucket search, tie predicate and candidate-cap sizing;
//   * the COMPOSITION — `BlockSim` in that file drives all three arms below in
//     the order they run here, against a full stable sort of the same row, over
//     the cap boundary, the production shape, the tie-dense twin, and rows that
//     defeat both compaction stages.
//
// The third is NOT gated and this file will not imply that it is. The PARALLEL
// PLUMBING — `atomicAdd` and `atomicOr` on shared memory, `__syncthreads`, the
// block min-reduce, the `extern __shared__` layout and the launch's shared-memory
// sizing — has no host runner, and `BlockSim` is a hand transcription, which can
// gate what the composition COMPUTES and never that this text computes it. A
// device run is what would close that, and it is owed with the timing: `## Owed`
// O34 of .agents/specs/dflash2-spec-decode.md.
__device__ inline int BlockRedMinI(int v, int* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] = min(s[t], s[t + o]);
    __syncthreads();
  }
  const int r = s[0];
  __syncthreads();
  return r;
}

// The radix top-k. One block per row. Shared state, in the order it is used:
//   s_hist        the round's 256-bucket histogram
//   s_cand_*      the compacted candidate set, valid only when `compacted`
//   s_pairs       the k winners, (key, index)
//
// `s_cand_*` and `s_pairs` both live in the DYNAMIC allocation, whose size the
// launcher takes from the device rather than from a constant in this file. See
// `include/vt/radix_topk.h`'s candidate-buffer section for the measured bucket
// populations that decide the arm, and `TopKValuesIndicesCuda` below for the
// `cudaFuncSetAttribute` opt-in that mirrors FlashInfer's.
__global__ void TopKValuesIndicesRadixRowKernel(float* out_values, int64_t* out_indices,
                                                const float* logits, int64_t v, int64_t usable,
                                                int k, int cand_cap) {
  const int64_t row = blockIdx.x;
  const float* r = logits + row * v;
  const int t = threadIdx.x;

  __shared__ uint32_t s_hist[kRadixTopKRadix];
  __shared__ int redi[kBlock];
  __shared__ uint32_t s_prefix;
  __shared__ uint32_t s_remaining;
  __shared__ int s_cand_count;
  __shared__ int s_overflow;
  __shared__ int s_filled;
  __shared__ int s_top_bucket;
  __shared__ int s_top_bucket1;
  __shared__ int s_restage;

  // The dynamic allocation, in order: k winner pairs, then the candidate buffer.
  // `k` and `cand_cap` are both runtime values, so this is the one layout the
  // launcher and the kernel have to agree on; `RadixTopKDynamicSmemBytes` states
  // its size once, for both sides.
  extern __shared__ uint32_t s_dyn[];
  uint32_t* s_pairs = s_dyn;
  int* s_pair_idx = reinterpret_cast<int*>(s_dyn + k);
  uint32_t* s_cand_key = s_dyn + 2 * k;
  int* s_cand_idx = reinterpret_cast<int*>(s_cand_key + cand_cap);

  if (t == 0) {
    s_prefix = 0u;
    s_remaining = static_cast<uint32_t>(k);
    s_cand_count = 0;
    s_overflow = 0;
    s_filled = 0;
    s_restage = 0;
  }
  for (int i = t; i < kRadixTopKRadix; i += kBlock) s_hist[i] = 0u;
  __syncthreads();

  // ROUND 0 — the only pass that reads every usable column. `usable = V -
  // num_org_vocab_padding`, so an org-vocab padding tail can never contribute a
  // candidate; this is upstream's `logits[..., -num_pad:] = -inf`, done by
  // restricting the search rather than by writing to a read-only input.
  for (int64_t j = t; j < usable; j += kBlock) {
    atomicAdd(&s_hist[RadixTopKBucket(RadixTopKKey(r[j]), 0)], 1u);
  }
  __syncthreads();
  if (t == 0) {
    uint32_t next = 0u;
    const uint32_t bucket = RadixTopKPickBucket(s_hist, s_remaining, &next);
    s_top_bucket = static_cast<int>(bucket);
    s_prefix = bucket << (32 - kRadixTopKBits);
    s_remaining = next;
  }
  // Thread 0 READS the histogram above, so the clear below cannot start until it
  // is done. Two barriers, not one.
  __syncthreads();
  for (int i = t; i < kRadixTopKRadix; i += kBlock) s_hist[i] = 0u;
  __syncthreads();
  const int top_bucket = s_top_bucket;

  // ROUND 1 — the second global pass, and the last one on a row whose round-0
  // bucket fits the candidate buffer. It does three things at once, which is
  // what keeps the pass count at two on such a row:
  //   * a column ABOVE the round-0 bucket is a winner outright. There are fewer
  //     than k of them (that is what `count_gt < remaining_k` means in
  //     `RadixTopKPickBucket`), so they go straight into the pair buffer.
  //   * a column IN the round-0 bucket is a candidate: it is histogrammed for
  //     round 1 and compacted into shared memory for rounds 2 and 3.
  //   * a column BELOW it is finished with.
  for (int64_t j = t; j < usable; j += kBlock) {
    const uint32_t key = RadixTopKKey(r[j]);
    const int bucket = static_cast<int>(RadixTopKBucket(key, 0));
    if (bucket > top_bucket) {
      const int slot = atomicAdd(&s_filled, 1);
      if (slot < k) {
        s_pairs[slot] = key;
        s_pair_idx[slot] = static_cast<int>(j);
      }
    } else if (bucket == top_bucket) {
      atomicAdd(&s_hist[RadixTopKBucket(key, 1)], 1u);
      const int slot = atomicAdd(&s_cand_count, 1);
      if (slot < cand_cap) {
        s_cand_key[slot] = key;
        s_cand_idx[slot] = static_cast<int>(j);
      } else {
        atomicOr(&s_overflow, 1);
      }
    }
  }
  __syncthreads();
  if (t == 0) {
    uint32_t next = 0u;
    const uint32_t bucket = RadixTopKPickBucket(s_hist, s_remaining, &next);
    s_prefix |= bucket << (32 - kRadixTopKBits * 2);
    s_remaining = next;
    s_top_bucket1 = static_cast<int>(bucket);
    // WHETHER TO RE-COMPACT, decided EXACTLY and for free. `s_hist[bucket]` is
    // the population of the round-1 bucket, counted by the pass above over every
    // column of the round-0 bucket — including the ones that failed to compact,
    // because the histogram add happens before the cap test. So the kernel knows
    // the size of the round-1 candidate set before it pays a pass for it, and it
    // re-compacts only when the result is guaranteed to fit. A second stage that
    // could itself overflow would need a second escape; this one cannot.
    //
    // THIS TEST'S SAFETY DOES NOT DEPEND ON THE MEASURED BUCKET SIZE, and the
    // measurement two comments up should not be read as if it did. The bin holds
    // 452 to 522 columns on every production row, which is why the re-staged arm
    // is REACHED; but the comparison is a known count against the real capacity,
    // so it would be exact at 490 or at 490000. If a wider vocabulary or a
    // different `kRadixTopKBits` makes the round-1 bucket stop fitting, this
    // reads false and the row takes the global arm below — slower, still exact.
    // Do not tighten a constant here to "protect" it.
    s_restage = (s_overflow != 0 && s_hist[bucket] <= static_cast<uint32_t>(cand_cap)) ? 1 : 0;
  }
  __syncthreads();
  const int top_bucket1 = s_top_bucket1;
  const bool restage = s_restage != 0;

  // STAGE 2 — the third and last global pass, and only on a row whose round-0
  // bucket was too wide to compact. It is FlashInfer's
  // `collect_with_threshold_non_last_round` (`flashinfer/topk.cuh:2566-2592`):
  // emit the columns strictly ABOVE this round's threshold bin as outright
  // winners, and carry the columns EQUAL to it into the candidate buffer.
  //
  // This is the pass that makes the compaction reach the shape #1867 targets.
  // Round 0's digit is the sign plus the top SEVEN exponent bits, so its bucket
  // spans two exponents and holds about 93000 of the 248320 columns on every
  // production row — 45x what a 2048-entry buffer held and 5.7x
  // `kRadixTopKCandCapMax`. One round later the same rows hold 452 to 522.
  // `include/vt/radix_topk.h` records the measurement.
  if (restage) {
    if (t == 0) s_cand_count = 0;
    __syncthreads();
    for (int64_t j = t; j < usable; j += kBlock) {
      const uint32_t key = RadixTopKKey(r[j]);
      if (static_cast<int>(RadixTopKBucket(key, 0)) != top_bucket) continue;
      const int b1 = static_cast<int>(RadixTopKBucket(key, 1));
      if (b1 > top_bucket1) {
        const int slot = atomicAdd(&s_filled, 1);
        if (slot < k) {
          s_pairs[slot] = key;
          s_pair_idx[slot] = static_cast<int>(j);
        }
      } else if (b1 == top_bucket1) {
        const int slot = atomicAdd(&s_cand_count, 1);
        // UNREACHABLE, and it stays. `s_restage` was set only when
        // `s_hist[top_bucket1] <= cand_cap`, and that histogram counted exactly
        // the columns this branch admits, so `slot < cand_cap` always holds. The
        // bound is here because the alternative to a proof that is right is an
        // out-of-bounds SHARED-MEMORY WRITE on a device, which corrupts another
        // block's state silently and cannot be seen from the answer.
        if (slot < cand_cap) {
          s_cand_key[slot] = key;
          s_cand_idx[slot] = static_cast<int>(j);
        }
      }
    }
    __syncthreads();
  }
  const bool compacted = s_overflow == 0 || restage;
  const int cand_count = s_cand_count < cand_cap ? s_cand_count : cand_cap;

  // ROUNDS 2 and 3 — over the compacted candidates when they fit, and over
  // global memory when they did not. The two arms differ only in WHERE the
  // histogram's inputs come from; the decision they feed is the same
  // `RadixTopKPickBucket` either way.
  for (int round = 2; round < kRadixTopKRounds; ++round) {
    for (int i = t; i < kRadixTopKRadix; i += kBlock) s_hist[i] = 0u;
    __syncthreads();
    const uint32_t prefix = s_prefix;
    if (compacted) {
      for (int i = t; i < cand_count; i += kBlock) {
        const uint32_t key = s_cand_key[i];
        if (RadixTopKPrefixMatches(key, prefix, round))
          atomicAdd(&s_hist[RadixTopKBucket(key, round)], 1u);
      }
    } else {
      for (int64_t j = t; j < usable; j += kBlock) {
        const uint32_t key = RadixTopKKey(r[j]);
        if (RadixTopKPrefixMatches(key, prefix, round))
          atomicAdd(&s_hist[RadixTopKBucket(key, round)], 1u);
      }
    }
    __syncthreads();
    if (t == 0) {
      uint32_t next = 0u;
      const uint32_t bucket = RadixTopKPickBucket(s_hist, s_remaining, &next);
      s_prefix |= bucket << (32 - kRadixTopKBits * (round + 1));
      s_remaining = next;
    }
    __syncthreads();
  }
  const uint32_t pivot = s_prefix;  // all 32 bits fixed: the k-th largest KEY

  // THE WINNERS STRICTLY ABOVE THE PIVOT. Those above the round-0 bucket are
  // already in the pair buffer; these are the ones inside it. Together they
  // number fewer than k, so the buffer cannot overflow. The count is exactly
  // `count(key > pivot)` however the arms below split it up, because each arm
  // emits a disjoint slice of that same set.
  if (compacted) {
    for (int i = t; i < cand_count; i += kBlock) {
      if (s_cand_key[i] > pivot) {
        const int slot = atomicAdd(&s_filled, 1);
        if (slot < k) {
          s_pairs[slot] = s_cand_key[i];
          s_pair_idx[slot] = s_cand_idx[i];
        }
      }
    }
  } else {
    for (int64_t j = t; j < usable; j += kBlock) {
      const uint32_t key = RadixTopKKey(r[j]);
      // The `== top_bucket` guard is not redundant: a column above the round-0
      // bucket is also above the pivot, and it was already emitted.
      if (key > pivot && static_cast<int>(RadixTopKBucket(key, 0)) == top_bucket) {
        const int slot = atomicAdd(&s_filled, 1);
        if (slot < k) {
          s_pairs[slot] = key;
          s_pair_idx[slot] = static_cast<int>(j);
        }
      }
    }
  }
  __syncthreads();
  int filled = s_filled < k ? s_filled : k;

  // THE TIE, and the whole of the index rule. The remaining `k - filled` slots
  // go to the LOWEST-INDEXED columns whose key EQUALS the pivot, one block-wide
  // min-index pass each. `k - filled` is `s_remaining` and is 1 in the ordinary
  // case — exactly one column attains the k-th largest value — so this is one
  // pass and not k of them. It always succeeds: `count(key >= pivot) >= k` holds
  // by construction of the radix prefix, given `k <= usable`, which
  // `src/vt/ops.cpp::TopKValuesIndices` enforces.
  int taken = -1;
  while (filled < k) {
    int local = INT_MAX;
    if (compacted) {
      for (int i = t; i < cand_count; i += kBlock) {
        if (s_cand_key[i] == pivot && s_cand_idx[i] > taken) local = min(local, s_cand_idx[i]);
      }
    } else {
      for (int64_t j = t; j < usable; j += kBlock) {
        if (RadixTopKKey(r[j]) == pivot && static_cast<int>(j) > taken) {
          local = min(local, static_cast<int>(j));
          break;  // ascending j: the first match in this thread's stride is its min
        }
      }
    }
    const int pick = BlockRedMinI(local, redi);
    if (pick == INT_MAX) break;  // unreachable; a defensive stop, not a fallback
    if (t == 0) {
      s_pairs[filled] = pivot;
      s_pair_idx[filled] = pick;
    }
    taken = pick;
    ++filled;
    __syncthreads();
  }

  // Order the k pairs: descending KEY, ties by ascending INDEX, which
  // `RadixTopKOutranks` states once for this kernel and for the host gate. k is
  // 16 on both published checkpoints, so an O(k^2) selection sort in one thread
  // is cheaper than any cooperative alternative.
  if (t == 0) {
    for (int a = 0; a < filled; ++a) {
      int best = a;
      for (int b = a + 1; b < filled; ++b) {
        if (RadixTopKOutranks(s_pairs[b], s_pair_idx[b], s_pairs[best], s_pair_idx[best])) best = b;
      }
      const uint32_t fk = s_pairs[a];
      const int fi = s_pair_idx[a];
      s_pairs[a] = s_pairs[best];
      s_pair_idx[a] = s_pair_idx[best];
      s_pairs[best] = fk;
      s_pair_idx[best] = fi;
    }
  }
  __syncthreads();
  // The VALUE is re-read from the row rather than reconstructed from the key.
  // The key is deliberately lossy — every NaN shares one key and -0.0f shares
  // +0.0f's — so inverting it would hand back a canonical NaN or a sign-flipped
  // zero where the CPU reference hands back the caller's own bits.
  for (int j = t; j < k; j += kBlock) {
    const int idx = j < filled ? s_pair_idx[j] : 0;
    out_values[row * k + j] = j < filled ? r[idx] : kNegInf;
    out_indices[row * k + j] = j < filled ? static_cast<int64_t>(idx) : 0;
  }
}

void TopKValuesIndicesCuda(Queue& q, Tensor& values, Tensor& indices, const Tensor& logits,
                           const TopKValuesIndicesArgs& args) {
  const int64_t rows = logits.shape[0], v = logits.shape[1];
  if (rows == 0) return;
  const int k = static_cast<int>(args.k);

  // THE CANDIDATE BUFFER IS SIZED BY THE DEVICE, not by a constant here. This
  // mirrors what FlashInfer does for the same buffer: `LaunchFilteredTopK-
  // Unified` opts into `FILTERED_TOPK_SMEM_DYNAMIC` (128 KB) with
  // `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
  // smem_size)` (`flashinfer/topk.cuh:3088-3105`), and its multi-CTA sizing
  // reads `cudaDevAttrMaxSharedMemoryPerBlockOptin` and subtracts the fixed
  // static shared before dividing (`topk.cuh:1480-1481`, and
  // `GetRadixTopKAvailableOrderedSmemBytes` at `topk.cuh:42-59`). The buffer
  // cannot be a static `__shared__` array at that size: ptxas caps static shared
  // at 48 KB, and `kRadixTopKCandCapMax` alone is 128 KB.
  //
  // The static half is read from the compiler rather than transcribed.
  // `cudaFuncGetAttributes().sharedSizeBytes` is the exact figure the driver
  // will charge, so it cannot drift as the kernel's `__shared__` declarations
  // change — which is the one way FlashInfer's hand-computed `fixed_smem_size`
  // can go wrong.
  //
  // Occupancy costs nothing here. One CTA per row over 8 rows never reaches two
  // blocks per SM on any part this runs on, so taking the whole per-block
  // shared budget cannot displace a block that would otherwise have resided.
  //
  // The kernel is named at each of the three call sites rather than bound to a
  // variable. `<<<>>>` through a `__global__` function POINTER is not a spelling
  // this repository can check: there is no `nvcc` here, and the only build that
  // compiles this file takes about two hours.
  int device = 0;
  int optin = 0;
  cudaFuncAttributes fa{};
  size_t static_shared = 0;
  if (cudaGetDevice(&device) == cudaSuccess &&
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) ==
          cudaSuccess &&
      cudaFuncGetAttributes(&fa, TopKValuesIndicesRadixRowKernel) == cudaSuccess) {
    static_shared = fa.sharedSizeBytes;
  } else {
    optin = 0;
  }
  const size_t budget =
      static_cast<size_t>(optin) > static_shared ? static_cast<size_t>(optin) - static_shared : 0;
  int cand_cap = RadixTopKCandCap(static_cast<uint32_t>(budget), k);
  size_t shared = RadixTopKDynamicSmemBytes(cand_cap, k);

  // The opt-in can be refused — an older driver, a device whose reported limit
  // the launch configuration cannot actually meet. A refusal is not a failure:
  // `cand_cap = 0` makes every row take the global arm, which answers the same
  // question by the same rounds and only costs the passes back. So retry once
  // without the opt-in, at whatever fits the DEFAULT 48 KB block allowance, and
  // fall through to zero if even that is refused.
  if (cand_cap > 0 &&
      cudaFuncSetAttribute(TopKValuesIndicesRadixRowKernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared)) != cudaSuccess) {
    constexpr size_t kDefaultBlockShared = 48u * 1024u;
    const size_t fallback =
        kDefaultBlockShared > static_shared ? kDefaultBlockShared - static_shared : 0;
    cand_cap = RadixTopKCandCap(static_cast<uint32_t>(fallback), k);
    shared = RadixTopKDynamicSmemBytes(cand_cap, k);
  }
  // `cudaFuncSetAttribute` leaves an error latched on the context when it
  // refuses, and the launch check below would then report a failure that is not
  // one. Clear it here, where the refusal was handled.
  cudaGetLastError();

  TopKValuesIndicesRadixRowKernel<<<static_cast<unsigned>(rows), kBlock, shared, AsStream(q)>>>(
      values.Ptr<float>(), indices.Ptr<int64_t>(), logits.Ptr<float>(), v,
      v - args.num_org_vocab_padding, k, cand_cap);
  Check(cudaGetLastError(), "topk_values_indices launch");
}

// --- apply_penalties (fused repetition + frequency + presence) --------------
__global__ void ApplyPenaltiesKernel(float* logits, const int8_t* prompt_mask,
                                     const int32_t* output_bin_counts, const int8_t* output_mask,
                                     const float* freq, const float* pres, const float* rep,
                                     int64_t n, int64_t v) {
  const int64_t total = n * v;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step) {
    const int64_t row = idx / v;
    float x = logits[idx];
    if (prompt_mask[idx] || output_mask[idx]) {
      const float r = rep[row];
      x = x > 0.0f ? x / r : x * r;
    }
    x -= freq[row] * static_cast<float>(output_bin_counts[idx]);
    x -= pres[row] * static_cast<float>(output_mask[idx]);
    logits[idx] = x;
  }
}

void ApplyPenaltiesCuda(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                        const Tensor& output_bin_counts, const Tensor& output_mask,
                        const Tensor& frequency_penalties, const Tensor& presence_penalties,
                        const Tensor& repetition_penalties) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyPenaltiesKernel<<<GridFor(n * v), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), prompt_mask.Ptr<int8_t>(), output_bin_counts.Ptr<int32_t>(),
      output_mask.Ptr<int8_t>(), frequency_penalties.Ptr<float>(), presence_penalties.Ptr<float>(),
      repetition_penalties.Ptr<float>(), n, v);
  Check(cudaGetLastError(), "apply_penalties launch");
}

// --- apply_min_p (block-per-row softmax, thread 0 threshold + mask) ----------
__global__ void ApplyMinPKernel(float* logits, const float* min_p, int64_t v) {
  const int64_t row = blockIdx.x;
  const float m = min_p[row];
  if (m <= 0.0f) return;
  float* r = logits + row * v;
  __shared__ float red[kBlock];

  float mx = kNegInf;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) mx = fmaxf(mx, r[j]);
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float rowmax = red[0];
  __syncthreads();

  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) acc += expf(r[j] - rowmax);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float sum = red[0];
  __syncthreads();

  // max prob == exp(rowmax - rowmax)/sum == 1/sum, so threshold = m / sum.
  const float thr = m / sum;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x)
    if (expf(r[j] - rowmax) / sum < thr) r[j] = kNegInf;
}

void ApplyMinPCuda(Queue& q, Tensor& logits, const Tensor& min_p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyMinPKernel<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(logits.Ptr<float>(),
                                                                        min_p.Ptr<float>(), v);
  Check(cudaGetLastError(), "apply_min_p launch");
}

// --- sparse scatter ops (logit-bias add / -inf token mask) ------------------
__global__ void ApplyLogitBiasKernel(float* logits, const int32_t* rows, const int32_t* cols,
                                     const float* biases, int64_t v, int64_t m) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < m; k += step)
    logits[static_cast<int64_t>(rows[k]) * v + cols[k]] += biases[k];
}

void ApplyLogitBiasCuda(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                        const Tensor& biases) {
  const int64_t v = logits.shape[1], m = rows.shape[0];
  if (m == 0) return;
  ApplyLogitBiasKernel<<<GridFor(m), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), rows.Ptr<int32_t>(), cols.Ptr<int32_t>(), biases.Ptr<float>(), v, m);
  Check(cudaGetLastError(), "apply_logit_bias launch");
}

__global__ void ApplyTokenMaskKernel(float* logits, const int32_t* rows, const int32_t* cols,
                                     int64_t v, int64_t m) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < m; k += step)
    logits[static_cast<int64_t>(rows[k]) * v + cols[k]] = kNegInf;
}

void ApplyTokenMaskCuda(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols) {
  const int64_t v = logits.shape[1], m = rows.shape[0];
  if (m == 0) return;
  ApplyTokenMaskKernel<<<GridFor(m), kBlock, 0, AsStream(q)>>>(logits.Ptr<float>(),
                                                              rows.Ptr<int32_t>(),
                                                              cols.Ptr<int32_t>(), v, m);
  Check(cudaGetLastError(), "apply_token_mask launch");
}

// --- apply_allowed_token_ids (masked_fill, mask TRUE == exclude) -------------
__global__ void ApplyAllowedTokenIdsKernel(float* logits, const int8_t* mask, int64_t total) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step)
    if (mask[idx]) logits[idx] = kNegInf;
}

void ApplyAllowedTokenIdsCuda(Queue& q, Tensor& logits, const Tensor& mask) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyAllowedTokenIdsKernel<<<GridFor(n * v), kBlock, 0, AsStream(q)>>>(logits.Ptr<float>(),
                                                                         mask.Ptr<int8_t>(), n * v);
  Check(cudaGetLastError(), "apply_allowed_token_ids launch");
}

// --- Greedy spec-decode rejection sampling (SPEC-REJECTION I3) --------------
// Mirrors the CPU reference (cpu_sample.cpp GreedyRejectionSampleKernel) element
// for element, and upstream's TWO-PHASE decomposition: a per-expanded-row argmax
// (upstream `_compute_local_logits_stats_kernel` per-vocab-block partials +
// `_compute_global_target_argmax`, rejection_sampler_utils.py:923-946) followed by
// a one-thread-per-request sequential accept walk (`_rejection_kernel`, launched
// `num_warps=1` at :1032-1067). Bit-exact vs the CPU reference on the accepted
// token ids: the argmax uses the SAME ArgReduce lowest-index tie-break as
// GreedyArgmax, and the accept walk is pure integer equality.

// Phase 1: target_argmax[row] = argmax(logits[row]) for every expanded row.
__global__ void RejectionRowArgmaxKernel(int32_t* out, const float* logits, int64_t v) {
  const int64_t row = blockIdx.x;
  const float* r = logits + row * v;
  __shared__ float sv[kBlock];
  __shared__ int64_t si[kBlock];

  float bv = kNegInf;
  int64_t bi = kArgSentinel;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) ArgReduce(bv, bi, r[j], j);

  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s)
      ArgReduce(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + s], si[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[row] = static_cast<int32_t>((si[0] == kArgSentinel) ? 0 : si[0]);
}

// Phase 2: one thread per request walks its draft positions, accepting while the
// draft equals the target argmax; stores the accepted stream + accepted_length+1.
__global__ void GreedyRejectAcceptKernel(int32_t* sampled, int32_t* num_sampled,
                                         const int32_t* target_argmax,
                                         const int32_t* draft_sampled, const int32_t* cu_num_logits,
                                         int64_t width) {
  const int64_t req = blockIdx.x;
  const int64_t start = cu_num_logits[req];
  const int64_t end = cu_num_logits[req + 1];
  const int64_t num_draft_tokens = end - start - 1;
  int32_t* row = sampled + req * width;
  for (int64_t j = 0; j < width; ++j) row[j] = -1;  // PLACEHOLDER_TOKEN_ID pad

  bool accepted = true;
  int64_t accepted_length = 0;
  for (int64_t i = 0; i < num_draft_tokens; ++i) {
    if (!accepted) break;  // upstream `elif accepted:` guard
    const int32_t ta = target_argmax[start + i];
    // +1: draft token i is the input id at the NEXT expanded row (:534). A -1
    // placeholder can never equal an argmax (>= 0) => rejected, no OOB read.
    const int32_t draft = draft_sampled[start + i + 1];
    accepted = (ta == draft);
    row[i] = accepted ? draft : ta;
    accepted_length += accepted ? 1 : 0;
  }
  if (accepted_length == num_draft_tokens) {
    // Bonus token (greedy resample == argmax of the bonus row).
    row[accepted_length] = target_argmax[start + accepted_length];
  }
  num_sampled[req] = static_cast<int32_t>(accepted_length) + 1;
}

// Persistent grow-only scratch for the per-row argmax (a few KB), so a verify
// step pays no cudaMalloc/cudaFree. Same pattern as the greedy-argmax partials.
int32_t* g_reject_argmax = nullptr;
size_t g_reject_argmax_cap = 0;

void EnsureRejectArgmaxScratch(size_t elems) {
  if (elems <= g_reject_argmax_cap) return;
  if (g_reject_argmax) cudaFree(g_reject_argmax);
  Check(cudaMalloc(&g_reject_argmax, elems * sizeof(int32_t)), "rejection argmax scratch");
  g_reject_argmax_cap = elems;
}

void GreedyRejectionSampleCuda(Queue& q, Tensor& sampled, Tensor& num_sampled, const Tensor& logits,
                               const Tensor& draft_sampled, const Tensor& cu_num_logits) {
  const int64_t num_logits = logits.shape[0], v = logits.shape[1];
  const int64_t num_reqs = cu_num_logits.shape[0] - 1;
  if (num_reqs == 0 || num_logits == 0 || v == 0) return;
  cudaStream_t s = AsStream(q);

  EnsureRejectArgmaxScratch(static_cast<size_t>(num_logits));
  RejectionRowArgmaxKernel<<<static_cast<unsigned>(num_logits), kBlock, 0, s>>>(
      g_reject_argmax, logits.Ptr<float>(), v);
  Check(cudaGetLastError(), "greedy_rejection_sample argmax launch");
  GreedyRejectAcceptKernel<<<static_cast<unsigned>(num_reqs), 1, 0, s>>>(
      sampled.Ptr<int32_t>(), num_sampled.Ptr<int32_t>(), g_reject_argmax,
      draft_sampled.Ptr<int32_t>(), cu_num_logits.Ptr<int32_t>(), sampled.shape[1]);
  Check(cudaGetLastError(), "greedy_rejection_sample accept launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kApplyTemperature, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyTemperatureFn>(&ApplyTemperatureCuda)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxCuda)));
    RegisterOp(OpId::kTopKValuesIndices, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<TopKValuesIndicesFn>(&TopKValuesIndicesCuda)));
    RegisterOp(OpId::kApplyTopKTopP, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyTopKTopPFn>(&ApplyTopKTopPCuda)));
    RegisterOp(OpId::kComputeProbs, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ComputeProbsFn>(&ComputeProbsCuda)));
    RegisterOp(OpId::kComputeLogprobs, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ComputeLogprobsFn>(&ComputeLogprobsCuda)));
    RegisterOp(OpId::kRandomSample, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RandomSampleFn>(&RandomSampleCuda)));
    RegisterOp(OpId::kGreedyRejectionSample, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<GreedyRejectionSampleFn>(&GreedyRejectionSampleCuda)));
    RegisterOp(OpId::kApplyPenalties, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyPenaltiesFn>(&ApplyPenaltiesCuda)));
    RegisterOp(OpId::kApplyMinP, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyMinPFn>(&ApplyMinPCuda)));
    RegisterOp(OpId::kApplyLogitBias, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyLogitBiasFn>(&ApplyLogitBiasCuda)));
    RegisterOp(OpId::kApplyTokenMask, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyTokenMaskFn>(&ApplyTokenMaskCuda)));
    RegisterOp(
        OpId::kApplyAllowedTokenIds, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<ApplyAllowedTokenIdsFn>(&ApplyAllowedTokenIdsCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
