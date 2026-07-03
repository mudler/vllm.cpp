// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA MoE ops (M0.8 Task 3): router top-k (softmax + greedy top-k +
// renormalize) and weighted combine. Correctness-grade — plain kernels
// matching the CPU reference math in src/vt/cpu/cpu_ops.cpp element for
// element; formulas from .agents/moe-semantics.md (§3 router, §4/§6 combine).
//
// Upstream counterpart: layers/fused_moe/ (fused_topk / moe_align + grouped
// GEMM Triton/cutlass kernels — M2.2 replaces this correctness-grade path).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;

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

// ---------------------------------------------------------------------------
// moe_router_topk (moe-semantics.md §3): one BLOCK per token. The softmax is a
// block reduction (max-subtracted, f32, over all E experts); the greedy top-k
// runs SINGLE-THREADED so the lowest-index tie-break matches the CPU reference
// (cpu_ops.cpp MoeRouterTopKKernel) bit-for-bit — k linear scans over E masking
// the winner each round (k <= 8, E <= 256: correctness-grade, cheap). The probs
// live in dynamic shared memory [E]; `red[kBlock]` is the reduction scratch.
// Upstream counterpart: layers/fused_moe/ (topk_softmax_kernels.cu topkGating /
// moeSoftmax+moeTopK) — M2.2 replaces this correctness-grade path.

template <typename Tin>
__global__ void MoeRouterTopKKernel(float* weights, int32_t* indices, const Tin* logits,
                                    int64_t e, int k, bool renormalize) {
  const int64_t row = blockIdx.x;
  const Tin* lrow = logits + row * e;
  extern __shared__ float sp[];  // [e] softmax probs
  __shared__ float red[kBlock];

  // Max over E (max-subtraction, topk_softmax_kernels.cu / cpu_ops.cpp §3).
  float m = -INFINITY;
  for (int64_t j = threadIdx.x; j < e; j += blockDim.x) m = fmaxf(m, Load(lrow, j));
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float mx = red[0];
  __syncthreads();

  // exp(logit - max) into shared, block-summed for the denominator.
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < e; j += blockDim.x) {
    const float ex = expf(Load(lrow, j) - mx);
    sp[j] = ex;
    acc += ex;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float sum = red[0];
  __syncthreads();

  // Normalize with the sum>0 guard + NaN/Inf clamp (cpu_ops.cpp §3, .cu:136).
  for (int64_t j = threadIdx.x; j < e; j += blockDim.x) {
    float pj = sum > 0.0f ? sp[j] / sum : 0.0f;
    if (!isfinite(pj)) pj = 0.0f;
    sp[j] = pj;
  }
  __syncthreads();

  // Greedy argmax, k rounds, single-threaded. Strict `>` over ascending idx ->
  // lowest expert index wins ties (matches cpu_ops.cpp exactly). Probs are
  // finite >= 0; masking a winner with -INFINITY excludes it from later rounds.
  if (threadIdx.x == 0) {
    float denom = 0.0f;
    for (int j = 0; j < k; ++j) {
      int64_t best = -1;
      float best_v = -INFINITY;
      for (int64_t idx = 0; idx < e; ++idx) {
        if (sp[idx] > best_v) {
          best_v = sp[idx];
          best = idx;
        }
      }
      sp[best] = -INFINITY;  // exclude from subsequent rounds
      weights[row * k + j] = best_v;
      indices[row * k + j] = static_cast<int32_t>(best);
      denom += best_v;
    }
    if (renormalize) {
      if (!(denom > 0.0f)) denom = 1.0f;  // denom<=0 -> 1 guard (.cu:245-253)
      for (int j = 0; j < k; ++j) weights[row * k + j] /= denom;
    }
  }
}

template <typename Tin>
void LaunchRouter(cudaStream_t s, Tensor& weights, Tensor& indices, const Tensor& logits,
                  int64_t t, int64_t e, int k, bool renorm) {
  const size_t shmem = static_cast<size_t>(e) * sizeof(float);
  MoeRouterTopKKernel<Tin><<<static_cast<unsigned>(t), kBlock, shmem, s>>>(
      weights.Ptr<float>(), indices.Ptr<int32_t>(), logits.Ptr<Tin>(), e, k, renorm);
  Check(cudaGetLastError(), "moe_router_topk launch");
}

void MoeRouterTopKKernelCuda(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                             const MoeRouterTopKArgs& args) {
  VT_CHECK(logits.dtype == DType::kF32 || logits.dtype == DType::kBF16,
           "cuda moe_router_topk: unsupported logits dtype (f32/bf16 only)");
  const int64_t t = logits.shape[0], e = logits.shape[1];
  if (t == 0 || e == 0) return;
  cudaStream_t s = AsStream(q);
  if (logits.dtype == DType::kF32) {
    LaunchRouter<float>(s, weights, indices, logits, t, e, args.top_k, args.renormalize);
  } else {
    LaunchRouter<__nv_bfloat16>(s, weights, indices, logits, t, e, args.top_k, args.renormalize);
  }
}

// ---------------------------------------------------------------------------
// moe_combine (moe-semantics.md §4/§6): thread per (token, output-dim). Sums
// the k expert contributions weighted by the router weights (f32 accumulation),
// adds the optional shared term in f32, single store-rounding — same as the CPU
// reference (cpu_ops.cpp MoeCombineKernel), so CPU and CUDA agree bit-for-bit.
// (No upstream double-round here; that M0.9 decision is separate.)
// Upstream counterpart: layers/fused_moe/ (moe_sum reduction over the topk
// weighted w2 outputs) — M2.2 replaces this correctness-grade path.

template <typename Teo, typename Tsh, typename Tout>
__global__ void MoeCombineKernel(Tout* out, const Teo* expert_out, const float* weights,
                                 const Tsh* shared, int64_t t, int64_t h, int k) {
  const int64_t n = t * h;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t row = idx / h;
    const int64_t col = idx % h;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j)
      acc += weights[row * k + j] * Load(expert_out, (row * k + j) * h + col);
    if (shared != nullptr) acc += Load(shared, idx);
    Store(out, idx, acc);
  }
}

template <typename Teo, typename Tsh, typename Tout>
void LaunchCombine(cudaStream_t s, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                   const Tensor* shared, int64_t t, int64_t h, int k) {
  MoeCombineKernel<Teo, Tsh, Tout><<<GridFor(t * h), kBlock, 0, s>>>(
      out.Ptr<Tout>(), expert_out.Ptr<Teo>(), weights.Ptr<float>(),
      shared != nullptr ? shared->Ptr<Tsh>() : nullptr, t, h, k);
  Check(cudaGetLastError(), "moe_combine launch");
}

// Dispatch shared dtype (or the no-shared path, where Tsh is unused).
template <typename Teo, typename Tout>
void DispatchShared(cudaStream_t s, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                    const Tensor* shared, int64_t t, int64_t h, int k) {
  if (shared == nullptr || shared->dtype == DType::kF32) {
    LaunchCombine<Teo, float, Tout>(s, out, expert_out, weights, shared, t, h, k);
  } else {
    LaunchCombine<Teo, __nv_bfloat16, Tout>(s, out, expert_out, weights, shared, t, h, k);
  }
}

template <typename Teo>
void DispatchOut(cudaStream_t s, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                 const Tensor* shared, int64_t t, int64_t h, int k) {
  if (out.dtype == DType::kF32) {
    DispatchShared<Teo, float>(s, out, expert_out, weights, shared, t, h, k);
  } else {
    DispatchShared<Teo, __nv_bfloat16>(s, out, expert_out, weights, shared, t, h, k);
  }
}

void MoeCombineKernelCuda(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                          const Tensor* shared) {
  VT_CHECK(expert_out.dtype == DType::kF32 || expert_out.dtype == DType::kBF16,
           "cuda moe_combine: unsupported expert_out dtype (f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "cuda moe_combine: unsupported out dtype (f32/bf16 only)");
  VT_CHECK(shared == nullptr || shared->dtype == DType::kF32 || shared->dtype == DType::kBF16,
           "cuda moe_combine: unsupported shared dtype (f32/bf16 only)");
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  if (t == 0 || h == 0) return;
  cudaStream_t s = AsStream(q);
  if (expert_out.dtype == DType::kF32) {
    DispatchOut<float>(s, out, expert_out, weights, shared, t, h, static_cast<int>(k));
  } else {
    DispatchOut<__nv_bfloat16>(s, out, expert_out, weights, shared, t, h, static_cast<int>(k));
  }
}

// Registers the CUDA MoE kernels during static init (pre-main, like the M0.6
// ops in cuda_ops.cu). Filling the op table is harmless on machines without a
// GPU: the kCUDA backend never registers there, so no CUDA queue can dispatch.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMoeRouterTopK, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MoeRouterTopKFn>(&MoeRouterTopKKernelCuda)));
    RegisterOp(OpId::kMoeCombine, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MoeCombineFn>(&MoeCombineKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
