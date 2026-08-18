// vllm.cpp — static per-tensor FP8 (e4m3) activation quant, CUDA arm.
//
// Mirror of vLLM's `static_scaled_fp8_quant`
// (csrc/quantization/w8a8/fp8/common.cuh:58-77 `scaled_fp8_conversion` and
// csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:31/:204-210, pinned
// oracle @ 5559679229bc961848b121ccdeaa8fa5d79bec98). `is_scale_inverted ==
// false` at the call site, so the reciprocal is formed once by the caller and
// the elementwise math is a MULTIPLY.
//
// WHY THIS FILE EXISTS AT ALL — issue #960, and read it before moving anything
// back. This kernel used to live in `cuda_matmul_fp8_cutlass.cu`, whose sole
// build gate is `VT_CUTLASS_FP8_ARCHS` (CMakeLists.txt: the TU is added to
// `_FP8_CUTLASS_SOURCES` only when that variable is non-empty). The kernel has
// NO cutlass dependency of any kind — it is `x * (1/s)` followed by a hardware
// e4m3 convert — but sharing the translation unit made its REGISTRATION
// inherit cutlass's arch set. On every CUDA arch outside that set (sm_110/Thor
// is the measured one; it is not a Thor quirk) `OpId::kQuantFp8Static` was
// therefore not registered for `DeviceType::kCUDA` at all, so a CUDA queue
// asking for it fell through to the portable CPU reference tier, which
// dereferenced device pointers and took the process down with SIGSEGV (#844 is
// the same defect seen from the fallback's end). Its GEMM partner
// `kMatmulFp8CublasLt` is registered unconditionally in `cuda_matmul.cu`, so
// nothing upstream of the quant refused: the build looked complete and crashed
// one call later.
//
// So this TU is listed in the UNCONDITIONAL `if(VLLM_CPP_CUDA)` source list and
// carries no feature-gated include. Keep it that way: a kernel whose
// compilation is governed by a feature it does not use is the defect, and
// co-locating it with either the cutlass GEMM or the general op grab-bag would
// re-create a weaker form of the same coupling.
// `scripts/check-cuda-op-arch-gate.py` pins the invariant structurally;
// `tests/vt/test_ops_fp8_cpu.cpp` G4 pins it at run time.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: quant_fp8: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// ---- Static per-tensor fp8 activation quant (vLLM static_scaled_fp8_quant) ---
// inv = 1/input_scale; out_fp8[i] = fp8_e4m3(clamp(x[i]*inv, -448, 448)).
// A RECIPROCAL MULTIPLY, not a divide, and the reciprocal is hoisted out of the
// loop — that is upstream's shipped form (`x = val * scale` with the inverse
// formed by the caller: csrc/quantization/w8a8/fp8/common.cuh:62 and
// csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:31). The code below is
// RIGHT; do not "fix" it into `x / input_scale` to match a prose formula. The two
// differ by up to one f32 ulp before the fp8 round, and near an e4m3 tie that
// ulp changes the emitted byte on a default-ON 35B path.
// __NV_SATFINITE cvt saturates == clamp-then-cvt; RNE == vLLM's hardware cvt.
// Tin f32/bf16.
//
// The CPU arm (src/vt/cpu/cpu_ops.cpp QuantFp8StaticKernel) is the byte-for-byte
// mirror of this kernel. That equivalence is gate G2 of
// .agents/specs/vt-fp8-w8a8-cpu-arm.md; it is MEASURED on sm_110 and sm_121a
// (see .agents/specs/vt-fp8-quant-arch-gate.md — it could not be measured before
// #960 because this kernel was not registered on a non-cutlass-fp8 CUDA arch).
// The independent evidence on the CPU side is weaker and separate: G1 proves the
// CPU kernel matches an e4m3 reference derived from the format.
//
// The FUSED arm of this same math is `RmsNormQuantFp8` in cuda_ops.cu, whose
// `RmsNormF32ToFp8Dev` is deliberately the identical convert: that op's
// bit-identity claim to `RmsNorm(bf16) + QuantFp8Static` depends on it. Change
// one and you have silently changed the other's contract.
__device__ __forceinline__ uint8_t F32ToFp8Dev(float f) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
}
__device__ inline float LoadIn(const float* p, int64_t i) { return p[i]; }
__device__ inline float LoadIn(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }

template <typename Tin>
__global__ void QuantFp8StaticKernel(uint8_t* out, const Tin* x, float input_scale, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float inv = 1.0f / input_scale;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step)
    out[i] = F32ToFp8Dev(LoadIn(x, i) * inv);
}

void QuantFp8StaticKernelCuda(Queue& q, Tensor& out_fp8, const Tensor& x, float input_scale) {
  const int64_t n = x.shape[0] * x.shape[1];
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  const int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 65535));
  switch (x.dtype) {
    case DType::kF32:
      QuantFp8StaticKernel<float><<<blocks, 256, 0, s>>>(out_fp8.Ptr<uint8_t>(), x.Ptr<float>(),
                                                         input_scale, n);
      break;
    case DType::kBF16:
      QuantFp8StaticKernel<__nv_bfloat16><<<blocks, 256, 0, s>>>(
          out_fp8.Ptr<uint8_t>(), x.Ptr<__nv_bfloat16>(), input_scale, n);
      break;
    default: VT_CHECK(false, "cuda quant_fp8_static: unsupported x dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "quant_fp8_static launch");
}

// ---- Dynamic per-token, per-GROUP fp8 activation quant -----------------------
// VT-QUANT-FP8-GROUP (#1189 M1, .agents/specs/vt-quant-fp8-group.md).
//
// Mirror of the kernel that ACTUALLY EXECUTES upstream on this architecture.
// `per_token_group_quant_fp8` calls `torch.ops._C.per_token_group_fp8_quant`
// and returns whenever the platform is CUDA-alike and the input is contiguous
// (vllm/model_executor/layers/quantization/utils/fp8_utils.py:635-650), so its
// Triton kernel never runs here. The executing kernel is
// csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu:
//   :47  float local_absmax = eps                  eps SEEDS the reduction
//   :53  fmaxf(local_absmax, fabsf((float)src))
//   :68  float y_s = local_absmax / max_8bit                    a DIVIDE
//   :85  fminf(fmaxf((float)src / y_s, min_8bit), max_8bit)     a DIVIDE
//   :86  DST_DTYPE(q)                              hardware e4m3 RNE
//
// THE TWO DIVIDES ARE THE CONTRACT. QuantFp8StaticKernel above multiplies by a
// hoisted reciprocal because that is upstream's shipped form for the static
// per-tensor path; here upstream divides, and the scale changes per group, so
// nothing is loop-invariant. The Triton fallback's `_absmax * (1.0 / fp8_max)`
// (fp8_utils.py:145) differs by up to one f32 ulp and says so in its own
// comment. Near an e4m3 tie that ulp changes the emitted byte. Do not "optimise"
// either divide into a reciprocal: tests/vt/test_ops_quant_fp8_group_cpu.cpp G6
// compares this kernel with the CPU arm BYTE FOR BYTE, and G1 compares the CPU
// arm with a reference that spells the divide out.
//
// STRUCTURE. Upstream splits one group across 16 lanes and reduces with
// `__shfl_xor_sync` (:21-40, :106). This kernel gives ONE THREAD the whole
// group instead. The two agree bit for bit because `fmaxf` is exact and
// associative over finite inputs, so the reduction ORDER cannot change the
// result, unlike a floating-point sum. That keeps the arm a correctness mirror
// with no shared memory and no intra-group synchronisation; a lane-parallel
// rewrite is a performance question for #1189 M5, which needs a GPU to measure.
//
// This TU is in the UNCONDITIONAL CUDA source list and this kernel, like its
// neighbour, has no cutlass dependency of any kind — it is a max, two divides
// and a hardware convert. `scripts/check-cuda-op-arch-gate.py` pins that, for
// the reason #960 records at the top of this file.
template <typename Tin>
__global__ void QuantFp8GroupKernel(uint8_t* out, float* scale, const Tin* x, int group_size,
                                    int64_t num_groups) {
  constexpr float kEps = 1e-10f;      // fp8_utils.py:570
  constexpr float kFp8MaxV = 448.0f;  // quant_utils.py:27-35 finfo(e4m3fn).max
  constexpr float kFp8MinV = -448.0f;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       gid < num_groups; gid += step) {
    const int64_t base = gid * group_size;
    float amax = kEps;  // :47 — eps SEEDS the reduction, so an all-zero group
                        // yields 1e-10/448 rather than dividing by zero
    for (int i = 0; i < group_size; ++i) amax = fmaxf(amax, fabsf(LoadIn(x, base + i)));  // :53
    const float y_s = amax / kFp8MaxV;                                                    // :68
    scale[gid] = y_s;
    for (int i = 0; i < group_size; ++i)
      out[base + i] =
          F32ToFp8Dev(fminf(fmaxf(LoadIn(x, base + i) / y_s, kFp8MinV), kFp8MaxV));  // :85
  }
}

void QuantFp8GroupKernelCuda(Queue& q, Tensor& out_fp8, Tensor& out_scale, const Tensor& x,
                             int group_size) {
  const int64_t m = x.shape[0], k = x.shape[1];
  const int64_t groups_per_row = k / group_size;
  const int64_t num_groups = m * groups_per_row;
  if (num_groups == 0) return;
  cudaStream_t s = AsStream(q);
  const int blocks = static_cast<int>(std::min<int64_t>((num_groups + 255) / 256, 65535));
  switch (x.dtype) {
    case DType::kF32:
      QuantFp8GroupKernel<float><<<blocks, 256, 0, s>>>(out_fp8.Ptr<uint8_t>(),
                                                        out_scale.Ptr<float>(), x.Ptr<float>(),
                                                        group_size, num_groups);
      break;
    case DType::kBF16:
      QuantFp8GroupKernel<__nv_bfloat16><<<blocks, 256, 0, s>>>(
          out_fp8.Ptr<uint8_t>(), out_scale.Ptr<float>(), x.Ptr<__nv_bfloat16>(), group_size,
          num_groups);
      break;
    default: VT_CHECK(false, "cuda quant_fp8_group: unsupported x dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "quant_fp8_group launch");
}

// Table fill only, no CUDA calls (see cuda_ops.cu for the rationale). This
// registration must stay at preprocessor-conditional depth 0 in a TU that is
// unconditionally compiled for CUDA — that IS the fix for #960.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQuantFp8Static, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<QuantFp8StaticFn>(&QuantFp8StaticKernelCuda)));
    RegisterOp(OpId::kQuantFp8Group, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<QuantFp8GroupFn>(&QuantFp8GroupKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda
