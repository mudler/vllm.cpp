// ROCm V1 sampling ops — port of cuda_sample.cu for Hermes / OpenAI paths.
// Correctness-grade: temperature, top-k/top-p, softmax, gumbel-max sample,
// penalties, min-p, logit bias. (Allowed/token mask also live in dense_basic.)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;
constexpr float kNegInf = -INFINITY;
constexpr int kThreshMaxIter = 64;

void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " + hipGetErrorString(err));
  }
}
hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }
unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

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

// --- temperature ------------------------------------------------------------
__global__ void ApplyTemperatureK(float* logits, const float* temp, int64_t n, int64_t v,
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

// --- softmax / log_softmax --------------------------------------------------
__global__ void SoftmaxK(float* out, const float* logits, int64_t v, bool log_softmax) {
  const int64_t row = blockIdx.x;
  const float* r = logits + row * v;
  float* o = out + row * v;
  __shared__ float red[kBlock];

  float m = kNegInf;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) m = fmaxf(m, r[j]);
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
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

// --- random sample (gumbel-max / exp noise) ---------------------------------
__global__ void RandomSampleK(int64_t* out, const float* probs, const int64_t* seeds, int64_t v) {
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

// --- top-k / top-p (sort-free threshold, from cuda_sample) ------------------
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

__global__ void ApplyTopKTopPRowK(float* logits, const int32_t* k_arr, const float* p_arr,
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

  float lmax = kNegInf, lmin = INFINITY;
  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    lmax = fmaxf(lmax, x);
    if (x != kNegInf) lmin = fminf(lmin, x);
  }
  const float mx = BlockRedMaxF(lmax, red);
  const float mn = BlockRedMinF(lmin, red);

  const bool topk_active = has_k && (k >= 1) && (static_cast<int64_t>(k) < v);
  float thr_k = kNegInf;
  if (topk_active) {
    int lc = 0;
    for (int64_t j = t; j < v; j += kBlock)
      if (r[j] > mn) ++lc;
    const int cnt_gt_min = BlockRedSumI(lc, redi);
    if (cnt_gt_min < k) {
      thr_k = mn;
    } else {
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

  float low = -1.0f;
  const bool topp_active = has_p && (p < 1.0f);
  if (topp_active) {
    float lden = 0.0f;
    for (int64_t j = t; j < v; j += kBlock)
      if (r[j] >= thr_k) lden += expf(r[j] - mx);
    const float denom = BlockRedSumF(lden, red);
    const float target = p * denom;
    float lo = 0.0f, hi = 1.0f, min_gt_low = 1.0f, max_le_high = 0.0f;
    for (int iter = 0; denom > 0.0f && iter < kThreshMaxIter; ++iter) {
      const float p0 = (2.0f * lo + hi) / 3.0f;
      const float p1 = (lo + 2.0f * hi) / 3.0f;
      float la0 = 0.0f, la1 = 0.0f, lmglow = hi, lmleh = lo;
      for (int64_t j = t; j < v; j += kBlock) {
        if (r[j] < thr_k) continue;
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

  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    const bool keep = (x >= thr_k) && (low < 0.0f || expf(x - mx) > low);
    if (!keep) r[j] = kNegInf;
  }
}

// --- penalties / min_p / logit bias -----------------------------------------
__global__ void ApplyPenaltiesK(float* logits, const int8_t* prompt_mask,
                                const int32_t* output_bin_counts, const int8_t* output_mask,
                                const float* freq, const float* pres, const float* rep, int64_t n,
                                int64_t v) {
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

__global__ void ApplyMinPK(float* logits, const float* min_p, int64_t v) {
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
    if (static_cast<int>(threadIdx.x) < s)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
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
  const float thr = m / sum;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x)
    if (expf(r[j] - rowmax) / sum < thr) r[j] = kNegInf;
}

__global__ void ApplyLogitBiasK(float* logits, const int32_t* rows, const int32_t* cols,
                                const float* biases, int64_t v, int64_t m) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < m; k += step)
    logits[static_cast<int64_t>(rows[k]) * v + cols[k]] += biases[k];
}

}  // namespace

void ApplyTemperatureKernelRocm(Queue& q, Tensor& logits, const Tensor& temp, bool all_random) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyTemperatureK<<<GridFor(n * v), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), temp.Ptr<float>(), n, v, all_random);
  Check(hipGetLastError(), "apply_temperature");
}

void ApplyTopKTopPKernelRocm(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyTopKTopPRowK<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), k != nullptr ? k->Ptr<int32_t>() : nullptr,
      p != nullptr ? p->Ptr<float>() : nullptr, v);
  Check(hipGetLastError(), "top_k_top_p");
}

void ComputeProbsKernelRocm(Queue& q, Tensor& probs, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  SoftmaxK<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(probs.Ptr<float>(),
                                                                   logits.Ptr<float>(), v, false);
  Check(hipGetLastError(), "compute_probs");
}

void ComputeLogprobsKernelRocm(Queue& q, Tensor& logprobs, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  SoftmaxK<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(
      logprobs.Ptr<float>(), logits.Ptr<float>(), v, true);
  Check(hipGetLastError(), "compute_logprobs");
}

void RandomSampleKernelRocm(Queue& q, Tensor& token_ids, const Tensor& probs,
                            const Tensor& seeds) {
  const int64_t n = probs.shape[0], v = probs.shape[1];
  if (n == 0 || v == 0) return;
  RandomSampleK<<<static_cast<unsigned>(n), 1, 0, AsStream(q)>>>(
      token_ids.Ptr<int64_t>(), probs.Ptr<float>(), seeds.Ptr<int64_t>(), v);
  Check(hipGetLastError(), "random_sample");
}

void ApplyPenaltiesKernelRocm(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                              const Tensor& output_bin_counts, const Tensor& output_mask,
                              const Tensor& frequency_penalties, const Tensor& presence_penalties,
                              const Tensor& repetition_penalties) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyPenaltiesK<<<GridFor(n * v), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), prompt_mask.Ptr<int8_t>(), output_bin_counts.Ptr<int32_t>(),
      output_mask.Ptr<int8_t>(), frequency_penalties.Ptr<float>(), presence_penalties.Ptr<float>(),
      repetition_penalties.Ptr<float>(), n, v);
  Check(hipGetLastError(), "apply_penalties");
}

void ApplyMinPKernelRocm(Queue& q, Tensor& logits, const Tensor& min_p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyMinPK<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(logits.Ptr<float>(),
                                                                     min_p.Ptr<float>(), v);
  Check(hipGetLastError(), "apply_min_p");
}

void ApplyLogitBiasKernelRocm(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                              const Tensor& biases) {
  const int64_t v = logits.shape[1], m = rows.shape[0];
  if (m == 0) return;
  ApplyLogitBiasK<<<GridFor(m), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), rows.Ptr<int32_t>(), cols.Ptr<int32_t>(), biases.Ptr<float>(), v, m);
  Check(hipGetLastError(), "apply_logit_bias");
}

}  // namespace vt::rocm
