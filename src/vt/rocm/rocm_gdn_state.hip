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


// ─────────────────────────────────────────────────────────────────────────────
// ROW GATHER/SCATTER over dim 0 — `vt::IndexSelect` / `vt::IndexCopy`.
//
// W2 of `.agents/specs/qwen4-exp-rocm-ops.md`, row MODEL-MM-QWEN4-EXP, issue
// ISSUE-LOCAL-01M2A1DTCZQVAH7M193XT9PN2V. Hand-translation of
// `src/vt/cuda/cuda_gdn.cu:432-527` (`IndexSelectRowKernel` /
// `IndexCopyRowKernel` + their host dispatch). CPU oracle and contract:
// `src/vt/cpu/cpu_ops.cpp:3048-3085`; the dispatcher's shape contract is
// `CheckIndexRowOp` at `src/vt/ops.cpp:3288-3322`.
//
// ─── THE PORT PROVENANCE DIFFERS FROM W1's, AND THE SPEC'S D3 DOES NOT APPLY ─
// W1's three ops each had a counterpart in vLLM's first-party AMD backend
// (`vllm/models/qwen4_exp/amd/ops/hc.py`), so D3's tie-break — where our CUDA
// arm and vLLM's AMD arm disagree, vLLM wins — had something to arbitrate.
// **These two have no counterpart there.** They are generic gather/scatter
// helpers, not `qwen4_exp` kernels: upstream spells both as
// `torch.index_select` and `Tensor.index_copy_` and ships no Triton kernel for
// either, on `amd/` or on `nvidia/`. So there is no second definition to
// reconcile, this tree's own CUDA arm is the donor, and the CPU arm is the
// contract and the only oracle that can fail this one. That is recorded here
// and in the spec rather than left for a reader to wonder about.
//
// ─── WHY THESE JOIN THIS TU AND SELF-REGISTER ───────────────────────────────
// The spec's port map puts them in an existing ROCm translation unit rather
// than a new file: they are two small kernels, their donor is `cuda_gdn.cu` —
// the same donor this file already carries — and their production callers sit
// beside the GDN state I/O above. The registration is nonetheless the
// SELF-REGISTERING form W1 introduced, not an entry in `rocm_ops.hip` beside
// this file's other two ops, for the reason AGENTS.md "Records" gives: a file
// that every row must edit is a lock, and `rocm_ops.hip` is one.
//
// ─── NOTHING HERE IS ARCHITECTURE-DEPENDENT ─────────────────────────────────
// Pure indexed data movement dispatched on the element WIDTH, so there is no
// arithmetic, no cross-lane primitive, no wavefront-size assumption and no
// gfx-family dependency; and because there is no arithmetic the arms are
// BIT-EXACT against the CPU oracle rather than close to it. The `int64_t`
// offsets are the donor's and are load-bearing: `idx[row] * row_stride + col`
// overflows 32 bits on a real KV cache.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

// `n_total` = rows * inner, over the COMPACT side. `row_stride` is the BASE
// side's outer stride IN ELEMENTS, which the contract allows to exceed `inner`
// (`ops.cpp:3317`), so a padded or row-strided view is addressed directly.
// `inner` is the packed row extent and comes from `Numel() / rows`, NOT from
// `shape[1]`: the compact side may be rank 3 or higher.
template <typename T>
__global__ void IndexSelectRowK(T* out, const T* in, const int32_t* idx,
                                int64_t inner, int64_t in_row_stride,
                                int64_t n_total) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_total; i += step) {
    const int64_t row = i / inner;
    const int64_t col = i - row * inner;
    out[i] = in[static_cast<int64_t>(idx[row]) * in_row_stride + col];
  }
}

template <typename T>
__global__ void IndexCopyRowK(T* out, const T* in, const int32_t* idx,
                              int64_t inner, int64_t out_row_stride,
                              int64_t n_total) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_total; i += step) {
    const int64_t row = i / inner;
    const int64_t col = i - row * inner;
    out[static_cast<int64_t>(idx[row]) * out_row_stride + col] = in[i];
  }
}

// The dtype is erased to an element WIDTH, exactly as the donor does: the op
// moves bytes and never reads a value, so bf16/f16 need no conversion helper
// and every dtype of a given width shares one instantiation.
void IndexSelectKernelRocm(Queue& q, Tensor& out, const Tensor& in,
                           const Tensor& idx) {
  const int64_t rows = idx.shape[0];
  if (rows == 0) return;
  const int64_t n = out.Numel();
  const int64_t inner = n / rows;
  const int64_t in_row_stride = in.stride[0];
  hipStream_t s = AsStream(q);
  switch (SizeOf(out.dtype)) {
    case 1:
      IndexSelectRowK<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint8_t*>(out.data), static_cast<const uint8_t*>(in.data),
          idx.Ptr<int32_t>(), inner, in_row_stride, n);
      break;
    case 2:
      IndexSelectRowK<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint16_t*>(out.data),
          static_cast<const uint16_t*>(in.data), idx.Ptr<int32_t>(), inner,
          in_row_stride, n);
      break;
    case 4:
      IndexSelectRowK<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint32_t*>(out.data),
          static_cast<const uint32_t*>(in.data), idx.Ptr<int32_t>(), inner,
          in_row_stride, n);
      break;
    default:
      IndexSelectRowK<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint64_t*>(out.data),
          static_cast<const uint64_t*>(in.data), idx.Ptr<int32_t>(), inner,
          in_row_stride, n);
      break;
  }
  Check(hipGetLastError(), "index_select row launch");
}

void IndexCopyKernelRocm(Queue& q, Tensor& out, const Tensor& in,
                         const Tensor& idx) {
  const int64_t rows = idx.shape[0];
  if (rows == 0) return;
  const int64_t n = in.Numel();
  const int64_t inner = n / rows;
  const int64_t out_row_stride = out.stride[0];
  hipStream_t s = AsStream(q);
  switch (SizeOf(in.dtype)) {
    case 1:
      IndexCopyRowK<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint8_t*>(out.data), static_cast<const uint8_t*>(in.data),
          idx.Ptr<int32_t>(), inner, out_row_stride, n);
      break;
    case 2:
      IndexCopyRowK<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint16_t*>(out.data),
          static_cast<const uint16_t*>(in.data), idx.Ptr<int32_t>(), inner,
          out_row_stride, n);
      break;
    case 4:
      IndexCopyRowK<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint32_t*>(out.data),
          static_cast<const uint32_t*>(in.data), idx.Ptr<int32_t>(), inner,
          out_row_stride, n);
      break;
    default:
      IndexCopyRowK<<<GridFor(n), kBlock, 0, s>>>(
          static_cast<uint64_t*>(out.data),
          static_cast<const uint64_t*>(in.data), idx.Ptr<int32_t>(), inner,
          out_row_stride, n);
      break;
  }
  Check(hipGetLastError(), "index_copy row launch");
}

// Registers both arms during static init, UNCONDITIONAL and at preprocessor
// depth 0. Neither kernel has an arch literal or a feature dependency, so a
// build gate would only make the ops vanish on some builds; on `gfx1151` there
// is nothing to fall through to, because the portable reference tier is not
// installed there (spec D2), and the call would refuse by name.
struct IndexRowRegistrar {
  IndexRowRegistrar() {
    RegisterOp(OpId::kIndexSelect, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<IndexSelectFn>(&IndexSelectKernelRocm)));
    RegisterOp(OpId::kIndexCopy, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<IndexCopyFn>(&IndexCopyKernelRocm)));
  }
};
const IndexRowRegistrar index_row_registrar;

}  // namespace

}  // namespace vt::rocm
