// ROCm GDN indexed state I/O (BACKEND-ROCM-GDN-KERNELS family 1; issue #41).
// Hand-translation of src/vt/cuda/cuda_gdn.cu:248-335 (GdnStateGatherKernel /
// GdnStateScatterKernel + host dispatch), readable side by side against the
// donor. CPU oracle: src/vt/cpu/cpu_ops.cpp:1666/1708. kGdnStateGather/Scatter
// are the first two ops `IndexedGdnOpsNative()` (qwen3_5.cpp) requires for the
// device-resident GDN state path, and the current Qwen3.5-0.8B M2 throw on
// discrete boards (op 77). Semantics: .agents/specs/gdn-semantics.md;
// .agents/specs/rocm-gdn-kernels.md is this family's spec.
//
// Gate: the "GDN state gather/scatter" case in test_backend_cross_device.cpp —
// bit-exact vs the CPU oracle (pure indexed data movement + boundary RNE
// conversion both sides share).

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm gdn: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}
inline unsigned GridFor(int64_t n) {
  if (n <= 0) return 1;
  const int64_t g = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(g > 65535 ? 65535 : g);
}

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}

// ── State I/O (donor cuda_gdn.cu:248-298) ────────────────────────────────────
template <typename TCache>
__global__ void GdnStateGatherK(float* working, const TCache* cache,
                                const int32_t* state_idx, const int8_t* has_i8,
                                const int32_t* has_i32, int64_t row_elems,
                                int64_t n, int64_t work_inner,
                                int64_t cache_inner) {
  const bool widened = cache_inner != work_inner;
  const int64_t cache_row = (row_elems / work_inner) * cache_inner;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    const int64_t row = i / row_elems;
    const int64_t col = i - row * row_elems;
    const bool keep = has_i8 != nullptr ? has_i8[row] != 0
                      : has_i32 != nullptr ? has_i32[row] != 0
                                           : true;
    int64_t cache_off;
    if (!widened) {
      cache_off = static_cast<int64_t>(state_idx[row]) * row_elems + col;
    } else {
      const int64_t m = col / work_inner, wcol = col - m * work_inner;
      cache_off = static_cast<int64_t>(state_idx[row]) * cache_row +
                  m * cache_inner + wcol;
    }
    working[i] = keep ? Ld(cache, cache_off) : 0.0f;
  }
}

template <typename TCache>
__global__ void GdnStateScatterK(TCache* cache, const float* working,
                                 const int32_t* state_idx, int64_t row_elems,
                                 int64_t n, int64_t work_inner,
                                 int64_t cache_inner) {
  const bool widened = cache_inner != work_inner;
  const int64_t cache_row = (row_elems / work_inner) * cache_inner;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    const int64_t row = i / row_elems;
    const int64_t col = i - row * row_elems;
    int64_t cache_off;
    if (!widened) {
      cache_off = static_cast<int64_t>(state_idx[row]) * row_elems + col;
    } else {
      const int64_t m = col / work_inner, wcol = col - m * work_inner;
      cache_off = static_cast<int64_t>(state_idx[row]) * cache_row +
                  m * cache_inner + wcol;
    }
    St(cache, cache_off, working[i]);
  }
}

}  // namespace

void GdnStateGatherKernelRocm(Queue& q, Tensor& working, const Tensor& cache,
                              const Tensor& state_idx, const Tensor* has_initial_state) {
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int64_t n = working.Numel();
  const int64_t row_elems = n / rows;
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  const int8_t* has_i8 =
      has_initial_state != nullptr && has_initial_state->dtype == DType::kI8
          ? has_initial_state->Ptr<int8_t>()
          : nullptr;
  const int32_t* has_i32 =
      has_initial_state != nullptr && has_initial_state->dtype == DType::kI32
          ? has_initial_state->Ptr<int32_t>()
          : nullptr;
  hipStream_t s = AsStream(q);
  if (cache.dtype == DType::kBF16) {
    GdnStateGatherK<<<GridFor(n), kBlock, 0, s>>>(
        working.Ptr<float>(), cache.Ptr<__hip_bfloat16>(), state_idx.Ptr<int32_t>(),
        has_i8, has_i32, row_elems, n, work_inner, cache_inner);
  } else if (cache.dtype == DType::kF16) {
    GdnStateGatherK<<<GridFor(n), kBlock, 0, s>>>(
        working.Ptr<float>(), cache.Ptr<__half>(), state_idx.Ptr<int32_t>(),
        has_i8, has_i32, row_elems, n, work_inner, cache_inner);
  } else {
    GdnStateGatherK<<<GridFor(n), kBlock, 0, s>>>(
        working.Ptr<float>(), cache.Ptr<float>(), state_idx.Ptr<int32_t>(),
        has_i8, has_i32, row_elems, n, work_inner, cache_inner);
  }
  Check(hipGetLastError(), "gdn state gather launch");
}

void GdnStateScatterKernelRocm(Queue& q, Tensor& cache, const Tensor& working,
                               const Tensor& state_idx) {
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int64_t n = working.Numel();
  const int64_t row_elems = n / rows;
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  hipStream_t s = AsStream(q);
  if (cache.dtype == DType::kBF16) {
    GdnStateScatterK<<<GridFor(n), kBlock, 0, s>>>(
        cache.Ptr<__hip_bfloat16>(), working.Ptr<float>(), state_idx.Ptr<int32_t>(),
        row_elems, n, work_inner, cache_inner);
  } else if (cache.dtype == DType::kF16) {
    GdnStateScatterK<<<GridFor(n), kBlock, 0, s>>>(
        cache.Ptr<__half>(), working.Ptr<float>(), state_idx.Ptr<int32_t>(),
        row_elems, n, work_inner, cache_inner);
  } else {
    GdnStateScatterK<<<GridFor(n), kBlock, 0, s>>>(
        cache.Ptr<float>(), working.Ptr<float>(), state_idx.Ptr<int32_t>(),
        row_elems, n, work_inner, cache_inner);
  }
  Check(hipGetLastError(), "gdn state scatter launch");
}

}  // namespace vt::rocm
