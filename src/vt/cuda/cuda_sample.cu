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

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vt/backend.h"
#include "vt/ops.h"

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

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kApplyTemperature, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyTemperatureFn>(&ApplyTemperatureCuda)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxCuda)));
    RegisterOp(OpId::kApplyTopKTopP, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyTopKTopPFn>(&ApplyTopKTopPCuda)));
    RegisterOp(OpId::kComputeProbs, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ComputeProbsFn>(&ComputeProbsCuda)));
    RegisterOp(OpId::kComputeLogprobs, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ComputeLogprobsFn>(&ComputeLogprobsCuda)));
    RegisterOp(OpId::kRandomSample, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RandomSampleFn>(&RandomSampleCuda)));
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
