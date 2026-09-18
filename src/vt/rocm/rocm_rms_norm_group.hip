// ROCm/HIP arm of the UNGATED PER-GROUP RMS NORM — `vt::RmsNormGroup`.
// Row MODEL-MM-QWEN4-EXP, W1 of `.agents/specs/qwen4-exp-rocm-ops.md`,
// issue ISSUE-LOCAL-01M2A1DTCZQVAH7M193XT9PN2V.
//
// SELF-REGISTERING translation unit in the established additive pattern: no
// existing kernel TU and no shared op array is edited. It is a NEW file rather
// than an addition to `rocm_rmsnorm.hip` for the reason AGENTS.md "Records"
// gives — that TU is a surface several rows write, and a shared file is a lock.
// The name mirrors its CUDA sibling `src/vt/cuda/cuda_rms_norm_group.cu` one
// for one, which the spec's port map requires.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// The donor is `src/vt/cuda/cuda_rms_norm_group.cu` (217 lines), transcribed;
// the only edits are the HIP spelling table (`cuda_runtime.h` ->
// `hip/hip_runtime.h`, `cuda_bf16.h` -> `hip/hip_bf16.h`, `__nv_bfloat16` ->
// `__hip_bfloat16`, `cudaStream_t` -> `hipStream_t`, `cudaGetLastError` ->
// `hipGetLastError`, `vt::cuda` -> `vt::rocm`). Behind the donor stands the CPU
// arm `src/vt/cpu/cpu_ops.cpp::RmsNormGroupKernel`, which is the oracle this
// arm's gate compares against, and behind that the `group_size is not None` arm
// of `Qwen4ExpTextRMSNorm` (transformers v5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py:158-181`), executed per row:
//
//   g       = i / group_size                          (:168-169, the reshape)
//   ms[g]   = mean_{i in g}( x[i]^2 )                 (:170, `pow(2).mean(-1)`)
//   out[i]  = x[i] * rsqrt(ms[g] + eps) * (1 + w[i])  when gemma      (:174-178)
//
// ─── AND THE vLLM AMD KERNEL BESIDE IT ───────────────────────────────────────
// Unlike when the donor was written, vLLM now DOES define this op on AMD:
// `vllm/models/qwen4_exp/amd/ops/hc.py::_grouped_gemma_rmsnorm_kernel` at the
// pin `e126687a9a`. It computes the same function. Two spellings differ, both
// recorded in the spec's D3a rather than resolved silently here:
//
//  * It writes the Gemma affine as `y = x*rrms; y += y*w` and says in its own
//    comment that the form exists "to lower to an FMA". This arm keeps the
//    donor's `(x*rrms) * (1 + w)`, which is what transformers writes at :177
//    and what the CPU oracle this arm is GATED against computes. The two agree
//    in exact arithmetic and differ by one rounding; adopting the Triton
//    spelling would move this arm away from the only oracle that can fail it.
//  * It admits a SHARED `[GROUP_DIM]` affine as well as a full `[DIM]` one
//    (`W_SHARED`). This op's contract requires the full-row weight
//    (`src/vt/ops.cpp:1228`, `weight.shape[0] == x.shape[1]`), so the shared
//    form is not expressible here. It is recorded as owed, not implemented,
//    because no `qwen4_exp` checkpoint this tree loads stores one.
//
// ─── ONE THREAD PER GROUP, AND WHY THAT IS THE CONTRACT AND NOT A SHORTCUT ────
// The CPU arm reduces the group's sum of squares in **f32, ascending,
// sequentially**, and its own header says why that width was chosen: "a wider
// host-reference accumulator would make the two arms answer to different
// numbers". The goldens were dumped in that order. A wavefront or block tree
// reduction is a DIFFERENT order and would put the two arms one re-association
// apart for no stated reason, so this arm walks the group with one thread in
// the host's index order. `vt::Qwen4ExpGatedResidual` reduces the same shape in
// DOUBLE and the two must NOT be unified; a device arm that picked its own
// width would have unified them by accident.
//
// The parallel axis is therefore (row x group), which is `T * H / group_size`
// threads — 97 sites x T tokens x hc groups on this architecture's real caller,
// and ample. Where it is not, the kernel is correct and slow, and the spec
// records the tree reduction as a SPEED item conditional on measuring it to be
// the bottleneck first.
//
// ─── THE `_rn` INTRINSICS, AND WHAT IS NOT ONE ──────────────────────────────
// The host provider is pinned to `-ffp-contract=off` (CMakeLists.txt:41-56), so
// `sumsq += v * v` on the host is a SEPARATE round-to-nearest multiply and add.
// In the donor the `_rn` intrinsics carry that rounding because `nvcc`'s
// `-fmad` defaults to ON and is not pinned. This build ALSO pins
// `-ffp-contract=off` for HIP (CMakeLists.txt:414), so the plain spelling would
// already round the host's way here; the intrinsics are kept anyway so the file
// stays diffable against its donor line for line, and so this arm's numerics do
// not depend on a compile option staying set. Every arithmetic operation below
// is therefore spelled with an `_rn` intrinsic. The
// two that are not are `sqrtf` and the `__frcp_rn` over it: IEEE-754 requires
// both to be correctly rounded, which is the same guarantee glibc gives, so
// they agree by the standard rather than by an intrinsic. There is no
// transcendental anywhere in this op.
//
// ─── NO MFMA, NO DP4A, NO WAVEFRONT ASSUMPTION ──────────────────────────────
// The kernel is a grid-stride loop over independent (row, group) pairs with no
// cross-lane primitive of any kind, so it is wavefront-size agnostic and
// carries no gfx-family dependency.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// <math.h>, not <cmath>: HIP's math headers declare the device functions this
// file uses in the GLOBAL namespace, and spelling the include this way keeps
// the file parseable by a plain host compiler as well.
#include <math.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

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

// Local dtype tag + accessors, the donor's arrangement: a local copy rather
// than a hoist, because hoisting would edit a translation unit several other
// rows are working in. `__float2bfloat16` / `__float2half` are
// round-to-nearest-even, identical to the host `vt::F32ToBF16` / `vt::F32ToF16`.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("rocm rmsnorm_group: unsupported ") + what +
                          " dtype (f32/f16/bf16)");
      return DTag::kF32;
  }
}

__device__ inline float LoadAt(const void* p, DTag tag, int64_t i) {
  switch (tag) {
    case DTag::kF32: return static_cast<const float*>(p)[i];
    case DTag::kF16: return __half2float(static_cast<const __half*>(p)[i]);
    default: return __bfloat162float(static_cast<const __hip_bfloat16*>(p)[i]);
  }
}

__device__ inline void StoreAt(void* p, DTag tag, int64_t i, float v) {
  switch (tag) {
    case DTag::kF32: static_cast<float*>(p)[i] = v; break;
    case DTag::kF16: static_cast<__half*>(p)[i] = __float2half(v); break;
    default: static_cast<__hip_bfloat16*>(p)[i] = __float2bfloat16(v); break;
  }
}

// ---------------------------------------------------------------------------
// ONE THREAD PER (row, group). The two loops below are `RmsNormGroupKernel`'s
// inner two, index for index, with each host `+`/`*` replaced by the intrinsic
// that spells the host's rounding.
__global__ void RmsNormGroupRocmKernel(void* out, DTag out_tag, const void* x, DTag x_tag,
                                       const void* w, DTag w_tag, int64_t rows, int64_t h,
                                       int64_t group_size, int64_t groups, float eps,
                                       bool gemma) {
  const int64_t total = rows * groups;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float gs_f = static_cast<float>(group_size);
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step) {
    const int64_t i = idx / groups;   // the token row
    const int64_t g = idx - i * groups;
    const int64_t rbase = i * h;
    const int64_t base = g * group_size;

    // `x.pow(2).mean(-1)` over the GROUP (:170, after the :168-169 reshape), in
    // f32 and ASCENDING, which is the host's order and the order the goldens
    // were dumped in.
    float sumsq = 0.0f;
    for (int64_t j = 0; j < group_size; ++j) {
      const float v = LoadAt(x, x_tag, rbase + base + j);
      sumsq = __fadd_rn(sumsq, __fmul_rn(v, v));
    }
    // eps is INSIDE the rsqrt and added to the MEAN SQUARE, once per group.
    // `__frcp_rn` over `sqrtf` is `1.0f / std::sqrt(...)`: both are correctly
    // rounded on both sides, so this is an equality and not an approximation.
    const float inv = __frcp_rn(sqrtf(__fadd_rn(__fdiv_rn(sumsq, gs_f), eps)));
    for (int64_t j = 0; j < group_size; ++j) {
      const int64_t off = base + j;
      // The weight index is the FLAT one: upstream multiplies at :177, AFTER
      // `out.flatten(-2)` at :171, so `weight` spans the whole row and is not
      // broadcast per group. vLLM's AMD kernel spells the same layout as its
      // non-`W_SHARED` arm.
      float wj = LoadAt(w, w_tag, off);
      if (gemma) wj = __fadd_rn(wj, 1.0f);  // `1.0 + self.weight.float()` (:177)
      // ONE rounding, on the store (`output.type_as(x)`, :178). The normalized
      // value is NOT narrowed before the weight multiply; upstream's own comment
      // at :175-176 says that is what separates this norm from Llama's. The
      // association is the host's `(x * inv) * wj`, left to right.
      StoreAt(out, out_tag, rbase + off,
              __fmul_rn(__fmul_rn(LoadAt(x, x_tag, rbase + off), inv), wj));
    }
  }
}

void RmsNormGroupKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                            const RmsNormGroupArgs& args) {
  const int64_t rows = x.shape[0];
  const int64_t h = x.shape[1];
  const int64_t groups = h / args.group_size;  // the dispatcher checked divisibility
  const int64_t total = rows * groups;
  if (total == 0) return;  // empty-work early return, before the launch
  const DTag out_tag = TagOf(out.dtype, "out");
  const DTag x_tag = TagOf(x.dtype, "x");
  const DTag w_tag = TagOf(weight.dtype, "weight");
  RmsNormGroupRocmKernel<<<GridFor(total), kBlock, 0, AsStream(q)>>>(
      out.data, out_tag, x.data, x_tag, weight.data, w_tag, rows, h, args.group_size, groups,
      args.eps, args.gemma);
  Check(hipGetLastError(), "rmsnorm_group launch");
}

// Registers the ROCm arm during static init, UNCONDITIONAL and at preprocessor
// depth 0. This kernel has no arch literal and no feature dependency of any
// kind, so a build gate on it would only make the op vanish on some builds; on
// `gfx1151` there is nothing to fall through to, because the portable reference
// tier is not installed there (spec D2), and the call would refuse by name.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kRmsNormGroup, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RmsNormGroupFn>(&RmsNormGroupKernelRocm)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::rocm
