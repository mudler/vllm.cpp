// ROCm Embedding — second registered kernel (BACKEND-ROCM, M3 start).
//
// PORTED FROM: src/vt/cuda/cuda_ops.cu EmbeddingKernel (grid-stride gather with
// in-kernel id clamp + atomic error flag). Simplified host path: allocate a
// one-shot error flag, launch, synchronize, throw — same correctness contract
// as the original CUDA M0.6 shape (ops.h). The CUDA ring is a later speed
// concern; discrete R9700 needs a working op first.
//
// Wavefront-agnostic: no warp primitives, only grid-stride over elements.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_f16_codec.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

inline unsigned GridFor(int64_t n) {
  if (n <= 0) return 1;
  const int64_t g = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(g > 65535 ? 65535 : g);
}

struct EmbeddingErr {
  int status;  // 0 ok, 1 bad id
  int pad;
  long long id;
};

template <typename T>
__device__ inline T Load(const T* p, int64_t i) {
  return p[i];
}
template <typename T>
__device__ inline void Store(T* p, int64_t i, T v) {
  p[i] = v;
}

// bf16 load/store via float only when casting between dtypes — same-dtype path
// copies the 16-bit word.
template <typename Tin, typename Tout, typename Tid>
__global__ void EmbeddingKernel(Tout* out, const Tin* table, const Tid* ids,
                                int64_t n, int64_t h, int64_t v,
                                EmbeddingErr* err, bool round_bf16) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += step) {
    const int64_t i = idx / h;
    const int64_t j = idx - i * h;
    int64_t id = static_cast<int64_t>(ids[i]);
    if (id < 0 || id >= v) {
      if (atomicCAS(&err->status, 0, 1) == 0) err->id = static_cast<long long>(id);
      id = id < 0 ? 0 : v - 1;
    }
    if constexpr (std::is_same_v<Tin, uint16_t>) {
      float value = DF16ToF32(table[id * h + j]);
      if (round_bf16) value = DBF16ToF32(DF32ToBF16(value));
      if constexpr (std::is_same_v<Tout, float>) out[idx] = value;
      else reinterpret_cast<uint16_t*>(out)[idx] = DF32ToBF16(value);
    } else if constexpr (std::is_same_v<Tin, Tout>) {
      Store(out, idx, Load(table, id * h + j));
    } else if constexpr (std::is_same_v<Tin, float> &&
                         std::is_same_v<Tout, __hip_bfloat16>) {
      Store(out, idx, __float2bfloat16(Load(table, id * h + j)));
    } else if constexpr (std::is_same_v<Tin, __hip_bfloat16> &&
                         std::is_same_v<Tout, float>) {
      Store(out, idx, __bfloat162float(Load(table, id * h + j)));
    } else {
      Store(out, idx, static_cast<Tout>(Load(table, id * h + j)));
    }
  }
}

template <typename Tin, typename Tout>
void LaunchEmbedding(hipStream_t s, Tensor& out, const Tensor& table,
                     const Tensor& ids, EmbeddingErr* err) {
  const int64_t t = ids.shape[0], h = table.shape[1], v = table.shape[0];
  const int64_t n = t * h;
  if (n == 0) return;
  if (ids.dtype == DType::kI32) {
    EmbeddingKernel<Tin, Tout, int32_t><<<GridFor(n), kBlock, 0, s>>>(
        out.Ptr<Tout>(), table.Ptr<Tin>(), ids.Ptr<int32_t>(), n, h, v, err,
        table.weight_value_dtype == DType::kBF16);
  } else {
    EmbeddingKernel<Tin, Tout, int64_t><<<GridFor(n), kBlock, 0, s>>>(
        out.Ptr<Tout>(), table.Ptr<Tin>(), ids.Ptr<int64_t>(), n, h, v, err,
        table.weight_value_dtype == DType::kBF16);
  }
}

template <typename Tin>
void LaunchEmbeddingIn(hipStream_t s, Tensor& out, const Tensor& table,
                       const Tensor& ids, EmbeddingErr* err) {
  if (out.dtype == DType::kF32) {
    LaunchEmbedding<Tin, float>(s, out, table, ids, err);
  } else {
    LaunchEmbedding<Tin, __hip_bfloat16>(s, out, table, ids, err);
  }
}

}  // namespace

void EmbeddingKernelRocm(Queue& q, Tensor& out, const Tensor& table,
                         const Tensor& ids) {
  EnsureQueueDevice(q);
  VT_CHECK(table.dtype == DType::kF32 || table.dtype == DType::kBF16 ||
               table.dtype == DType::kF16,
           "rocm embedding: unsupported table dtype (f16/f32/bf16 only)");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "rocm embedding: unsupported out dtype");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "rocm embedding: ids must be i32 or i64");
  VT_CHECK(table.rank == 2 && out.rank == 2 && ids.rank == 1,
           "rocm embedding: expected table[V,H], out[T,H], ids[T]");
  VT_CHECK(out.shape[0] == ids.shape[0] && out.shape[1] == table.shape[1],
           "rocm embedding: shape mismatch");
  const int64_t n = ids.shape[0] * table.shape[1];
  if (n == 0) return;
  VT_CHECK(table.shape[0] > 0, "rocm embedding: empty table (vocab 0)");

  hipStream_t s = static_cast<hipStream_t>(q.handle);
  EmbeddingErr* derr = nullptr;
  Check(hipMalloc(reinterpret_cast<void**>(&derr), sizeof(EmbeddingErr)),
        "hipMalloc embedding flag");
  Check(hipMemsetAsync(derr, 0, sizeof(EmbeddingErr), s), "hipMemset embedding flag");

  if (table.dtype == DType::kF32) {
    LaunchEmbeddingIn<float>(s, out, table, ids, derr);
  } else if (table.dtype == DType::kF16) {
    LaunchEmbeddingIn<uint16_t>(s, out, table, ids, derr);
  } else {
    LaunchEmbeddingIn<__hip_bfloat16>(s, out, table, ids, derr);
  }
  Check(hipGetLastError(), "embedding launch");
  Check(hipStreamSynchronize(s), "embedding sync");

  EmbeddingErr herr{};
  Check(hipMemcpy(&herr, derr, sizeof(EmbeddingErr), hipMemcpyDeviceToHost),
        "embedding flag D2H");
  Check(hipFree(derr), "hipFree embedding flag");

  if (herr.status != 0) {
    throw std::runtime_error("vt rocm: embedding: id " + std::to_string(herr.id) +
                             " out of range [0, " +
                             std::to_string(table.shape[0]) + ")");
  }
}

}  // namespace vt::rocm
