// Ported from: vllm/v1/sample/ops/topk_topp_sampler.py + vllm/v1/sample/sampler.py @ e24d1b24.
//
// Correctness-grade CUDA kernels for the V1 sampling ops (M1.7 Task 2), mirroring
// the CPU reference (src/vt/cpu/cpu_sample.cpp) element for element:
//   - apply_temperature: grid-stride, per-row temp with the eps greedy guard.
//   - greedy_argmax / random_sample: ONE BLOCK per row, single-threaded scan so
//     the lowest-index tie-break matches torch.argmax / the CPU reference exactly.
//   - compute_probs / compute_logprobs: block-per-row max-subtracted softmax.
//   - apply_top_k_top_p: per-row ascending sort (thrust::stable_sort_by_key over
//     backend scratch) + the top-k threshold / top-p cumsum masking + scatter,
//     the same sort-based path as apply_top_k_top_p_pytorch.
// The FlashInfer fused sampler is M2.4. NOTE: this file is built + verified on
// dgx.casa (the CI box is CPU-only); the CUDA parity tests are HasCuda-guarded.
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cstdint>
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

// --- greedy_argmax (single-threaded per row: exact lowest-index tie-break) ---
__global__ void GreedyArgmaxKernel(int64_t* out, const float* logits, int64_t v) {
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

void GreedyArgmaxCuda(Queue& q, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  GreedyArgmaxKernel<<<static_cast<unsigned>(n), 1, 0, AsStream(q)>>>(token_ids.Ptr<int64_t>(),
                                                                      logits.Ptr<float>(), v);
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

// --- apply_top_k_top_p (sort-based path) ------------------------------------
// The masking passes run one block per row over the per-row sorted scratch; the
// cumsum for top-p is single-threaded (thread 0) to match the CPU reference's
// ascending prefix sum exactly.
__global__ void TopKMaskKernel(float* sorted, const int32_t* k, int64_t v) {
  const int64_t row = blockIdx.x;
  const int32_t kv = k[row];
  if (kv >= v || kv < 1) return;  // k>=vocab no-op; k<1 invalid — keep s[v-kv] in-bounds
  float* s = sorted + row * v;
  const float threshold = s[v - kv];  // sorted ascending: index v-k is k-th largest
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x)
    if (s[j] < threshold) s[j] = kNegInf;
}

__global__ void TopPMaskKernel(float* sorted, const float* p, int64_t v) {
  const int64_t row = blockIdx.x;
  if (threadIdx.x != 0) return;
  float* s = sorted + row * v;
  float mx = kNegInf;
  for (int64_t j = 0; j < v; ++j) mx = fmaxf(mx, s[j]);
  float denom = 0.0f;
  for (int64_t j = 0; j < v; ++j) denom += (s[j] == kNegInf) ? 0.0f : expf(s[j] - mx);
  const float cutoff = 1.0f - p[row];
  float cum = 0.0f;
  for (int64_t j = 0; j < v; ++j) {
    cum += ((s[j] == kNegInf) ? 0.0f : expf(s[j] - mx)) / denom;
    const bool keep_last = (j == v - 1);
    if (!keep_last && cum <= cutoff) s[j] = kNegInf;
  }
}

__global__ void ScatterKernel(float* logits, const float* sorted, const int64_t* idx, int64_t v) {
  const int64_t row = blockIdx.x;
  const float* s = sorted + row * v;
  const int64_t* ix = idx + row * v;
  float* out = logits + row * v;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) out[ix[j]] = s[j];
}

void ApplyTopKTopPCuda(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  cudaStream_t s = AsStream(q);

  // Both the k-only and the k+p cases take the sort path here: the sort-based
  // top-k threshold masks exactly the set apply_top_k_only would (the CPU fast
  // path is only an optimization), so CUDA/CPU parity holds either way.
  // Backend scratch for the per-row sorted values + permutation indices.
  Backend& backend = GetBackend(DeviceType::kCUDA);
  const size_t nf = static_cast<size_t>(n * v);
  float* sorted = static_cast<float*>(backend.Alloc(nf * sizeof(float)));
  int64_t* perm = static_cast<int64_t*>(backend.Alloc(nf * sizeof(int64_t)));

  // Copy logits into the scratch, then per-row ascending stable_sort_by_key.
  Check(cudaMemcpyAsync(sorted, logits.Ptr<float>(), nf * sizeof(float),
                        cudaMemcpyDeviceToDevice, s),
        "top_k_top_p scratch copy");
  auto policy = thrust::cuda::par.on(s);
  for (int64_t row = 0; row < n; ++row) {
    thrust::device_ptr<int64_t> ix(perm + row * v);
    thrust::sequence(policy, ix, ix + v, static_cast<int64_t>(0));
    thrust::device_ptr<float> keys(sorted + row * v);
    thrust::stable_sort_by_key(policy, keys, keys + v, ix);
  }

  if (k != nullptr)
    TopKMaskKernel<<<static_cast<unsigned>(n), kBlock, 0, s>>>(sorted, k->Ptr<int32_t>(), v);
  if (p != nullptr)
    TopPMaskKernel<<<static_cast<unsigned>(n), 1, 0, s>>>(sorted, p->Ptr<float>(), v);
  Check(cudaGetLastError(), "top_k_top_p mask launch");

  ScatterKernel<<<static_cast<unsigned>(n), kBlock, 0, s>>>(logits.Ptr<float>(), sorted, perm, v);
  Check(cudaGetLastError(), "top_k_top_p scatter launch");
  backend.Synchronize(q);
  backend.Free(sorted);
  backend.Free(perm);
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
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
