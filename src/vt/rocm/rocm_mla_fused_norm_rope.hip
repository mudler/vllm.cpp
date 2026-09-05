// ROCm `vt::FusedNormRope` (kFusedNormRope) — the Tier-A2+A5 MLA norm-rope fold
// (ROCM-FUSED-NORM-ROPE, issue #2564).
//
// WHY THIS OP IS NOT OPTIONAL ON THIS BACKEND. Every other missing MLA op falls
// through to the portable reference tier and merely runs on the host. This one
// does not, because `src/vllm/model_executor/layers/attention/mla_attention.cpp`
// BRANCHES on `vt::OpRegistered(kFusedNormRope, device)` before it ever calls
// the op, and `OpRegistered` is a native-only probe by design
// (`src/vt/op_provider.cpp:779-803`). A backend that does not register this
// kernel therefore takes the SPLIT A-projection path, which row-slices
// `kv_a_proj_with_mqa` — and a block-quantized weight has no row slice. So the
// absence of this kernel is a REFUSAL on a keep-quant MLA checkpoint, not a
// slowdown. That is what #2564 measured on `strix:gpu0` with GLM-5.3.
//
// UPSTREAM (pinned oracle `~/_git/vllm` @ `5559679229`). vLLM has no fused
// kernel for this pair; it composes the two steps, and this kernel is that
// composition:
//   vllm/model_executor/layers/mla.py:164-165 — `kv_c, k_pe = kv_lora.split(
//     [kv_lora_rank, qk_rope_head_dim], dim=-1)` then `self.kv_a_layernorm(kv_c)`
//   vllm/model_executor/layers/mla.py:175-177 — `self.rotary_emb(positions, ...,
//     k_pe)` over the trailing slice of the SAME merged row
//   vllm/model_executor/models/deepseek_v2.py:512-518 — the merged weight is
//     [kv_lora_rank + qk_rope_head_dim, hidden_size], which fixes the row shape
// The two halves address DISJOINT dims, so fusing them is arithmetically inert.
//
// PORTED FROM: src/vt/cuda/cuda_ops.cu:1163-1245 (`FusedNormRopeKernel`,
// `LaunchFusedNormRope`, `FusedNormRopeKernelCuda`), which is itself the
// composition of two kernels this backend ALREADY has natively:
//   * the latent RMS reduce + scale — src/vt/rocm/rocm_rmsnorm.hip:67-98
//     (`RmsNormRowKernel`), reproduced element for element. It is a
//     __syncthreads() tree over shared memory with NO warp-level primitive, so
//     it is wavefront-width agnostic; that is the property that makes it
//     portable to AMD's 64-lane wavefront unchanged, and it is the reason
//     `kBlock` stays 256 rather than being raised to a wavefront multiple —
//     256 is already four whole wavefronts AND it keeps the reduction ORDER
//     identical to the CUDA and CPU siblings, which is what makes the NMSE bar
//     mean anything.
//   * the cache read + neox/gpt-j rotation — src/vt/rocm/rocm_dense_basic.hip:
//     607-635 (`RopeFromCacheK`), reproduced element for element, specialized
//     to the single-vector rank-1-positions case the MLA decoupled-pe slice is.
//
// Gate: the FusedNormRope arm of tests/vt/test_backend_cross_device.cpp (NMSE
// <= 5e-4 against the CPU oracle, both rope styles, f32 and bf16), and the
// GLM-5.3 e2e leg on `strix:gpu0` through `vllm-cli`.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"

namespace vt::rocm {
namespace {

// cuda_ops.cu:24 / rocm_rmsnorm.hip:41. See the file header for why it is not
// widened to a wavefront multiple.
constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm fused_norm_rope: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, as the CUDA path and the host do
}

// One block per token. Byte-for-byte the CUDA donor: same reduction tree, same
// scale loop, same cache offsets, same neox/gpt-j pairing, and the same EARLY
// RETURN on an out-of-range position — which happens AFTER the latent half has
// been written, so an out-of-range token still gets its norm and keeps its
// pe_out row untouched, exactly as RopeFromCache's own `continue` does.
template <typename T, typename Tid>
__global__ void FusedNormRopeKernel(T* latent_out, T* pe_out, const T* x, const T* w,
                                    const Tid* positions, const T* cache, int64_t cache_rows,
                                    int64_t off, int rot, int64_t half, int64_t x_row_stride,
                                    int64_t lat_row_stride, int64_t pe_row_stride,
                                    bool is_neox_style, bool gemma, float eps) {
  const int64_t row = blockIdx.x;
  const T* xrow = x + row * x_row_stride;
  T* lrow = latent_out + row * lat_row_stride;
  T* prow = pe_out + row * pe_row_stride;

  // --- latent RMSNorm over [0, off) — identical to RmsNormRowKernel. ----------
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < off; j += kBlock) {
    const float v = Load(xrow, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(off) + eps);
  for (int64_t j = threadIdx.x; j < off; j += kBlock) {
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(lrow, j, Load(xrow, j) * inv * wj);
  }

  // --- decoupled-pe RopeFromCache over [off, off+rot) — identical to
  //     RopeFromCacheK (single vector, base rope, positions rank-1). -----------
  const int64_t position = static_cast<int64_t>(positions[row]);
  if (position < 0 || position >= cache_rows) return;
  const int64_t cache_offset = position * rot;
  for (int64_t pair = threadIdx.x; pair < half; pair += kBlock) {
    const float c = Load(cache, cache_offset + pair);
    const float sn = Load(cache, cache_offset + half + pair);
    const int64_t first = is_neox_style ? pair : pair * 2;
    const int64_t second = is_neox_style ? pair + half : pair * 2 + 1;
    const float xr = Load(xrow, off + first);
    const float yr = Load(xrow, off + second);
    Store(prow, first, xr * c - yr * sn);
    Store(prow, second, xr * sn + yr * c);
  }
}

template <typename T>
void LaunchFusedNormRope(hipStream_t stream, Tensor& latent_out, Tensor& pe_out,
                         const Tensor& x, const Tensor& w, const Tensor& positions,
                         const Tensor& cache, const RmsNormArgs& norm_args,
                         const RopeArgs& rope_args) {
  const int64_t t = x.shape[0];
  const int64_t off = w.shape[0];
  const int rot = rope_args.rotary_dim;
  const int64_t half = rot / 2;
  if (t == 0) return;
  const unsigned rows = static_cast<unsigned>(t);
  if (positions.dtype == DType::kI32) {
    FusedNormRopeKernel<T, int32_t><<<rows, kBlock, 0, stream>>>(
        latent_out.Ptr<T>(), pe_out.Ptr<T>(), x.Ptr<T>(), w.Ptr<T>(),
        positions.Ptr<int32_t>(), cache.Ptr<T>(), cache.shape[0], off, rot, half,
        x.stride[0], latent_out.stride[0], pe_out.stride[0], rope_args.is_neox_style,
        norm_args.gemma, norm_args.eps);
  } else {
    FusedNormRopeKernel<T, int64_t><<<rows, kBlock, 0, stream>>>(
        latent_out.Ptr<T>(), pe_out.Ptr<T>(), x.Ptr<T>(), w.Ptr<T>(),
        positions.Ptr<int64_t>(), cache.Ptr<T>(), cache.shape[0], off, rot, half,
        x.stride[0], latent_out.stride[0], pe_out.stride[0], rope_args.is_neox_style,
        norm_args.gemma, norm_args.eps);
  }
  Check(hipGetLastError(), "launch");
}

}  // namespace

// The registered FusedNormRopeFn (include/vt/ops.h:3388). Signature is the
// shared vt one, unchanged — that is the whole contract a backend has to meet.
void FusedNormRopeKernelRocm(Queue& q, Tensor& latent_out, Tensor& pe_out, const Tensor& x,
                             const Tensor& w, const Tensor& positions, const Tensor& cache,
                             const RmsNormArgs& norm_args, const RopeArgs& rope_args) {
  EnsureQueueDevice(q);
  switch (x.dtype) {
    case DType::kF32:
      LaunchFusedNormRope<float>(AsStream(q), latent_out, pe_out, x, w, positions, cache,
                                 norm_args, rope_args);
      break;
    case DType::kBF16:
      LaunchFusedNormRope<__hip_bfloat16>(AsStream(q), latent_out, pe_out, x, w, positions,
                                          cache, norm_args, rope_args);
      break;
    default:
      VT_CHECK(false, "rocm fused_norm_rope: unsupported dtype (f32/bf16 only)");
  }
}

}  // namespace vt::rocm
