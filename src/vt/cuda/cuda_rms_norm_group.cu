// CUDA kernel for the UNGATED PER-GROUP RMS NORM — `vt::RmsNormGroup`.
// Row MODEL-MM-QWEN4-EXP W6-CUDA-B, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`src/vt/cuda/cuda_layernorm.cu`, `src/vt/cuda/cuda_qwen4_exp_ple.cu`): no
// existing kernel TU and no shared op array is edited. It is a NEW file rather
// than an addition to `cuda_layernorm.cu` for the reason AGENTS.md "Records"
// gives — that TU is a surface several rows write, and a shared file is a lock.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// The CPU arm, `src/vt/cpu/cpu_ops.cpp::RmsNormGroupKernel`, which is itself the
// port of the `group_size is not None` arm of `Qwen4ExpTextRMSNorm`
// (transformers v5.16.0 `models/qwen4_exp/modeling_qwen4_exp.py:158-181`, sha256
// 77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459 — the lane
// pin in `.agents/oracles/transformers.md`), executed per row:
//
//   g       = i / group_size                          (:168-169, the reshape)
//   ms[g]   = mean_{i in g}( x[i]^2 )                 (:170, `pow(2).mean(-1)`)
//   out[i]  = x[i] * rsqrt(ms[g] + eps) * (1 + w[i])  when gemma      (:174-178)
//
// vLLM registers no `qwen4_exp` at its pin `6a5e8f5979`, so there is no vLLM
// kernel to mirror for the ALGORITHM. vLLM does define the op FORM this
// mirrors — `RmsNormArgs`, the f32 interior, the round-on-store — and
// `vt::RmsNorm` beside it is where those come from. The KERNEL STRUCTURE
// (`Check`, `AsStream`, `GridFor`, the grid-stride loop, the unconditional
// `Registrar`) is mirrored from `src/vt/cuda/cuda_qwen4_exp_ple.cu`.
//
// ─── ONE THREAD PER GROUP, AND WHY THAT IS THE CONTRACT AND NOT A SHORTCUT ────
// `vt::RmsNormGroup`'s CPU arm reduces the group's sum of squares in **f32,
// ascending, sequentially**, and its own header says why that width was chosen:
// "a wider host-reference accumulator would make the two arms answer to
// different numbers". The goldens were dumped in that order. A block- or
// warp-level tree reduction is a DIFFERENT order and would put the two arms one
// re-association apart for no stated reason, so this arm walks the group with
// one thread in the host's index order and is BYTE-IDENTICAL rather than close.
//
// The parallel axis is therefore (row x group), which is `T * H / group_size`
// threads. That is 97 sites x T tokens x hc groups on this architecture's real
// caller and is ample; where it is not — a single row with one group — the
// kernel is correct and slow, and the spec's `## Owed` records the tree
// reduction as a SPEED item with the condition that it must first be measured to
// be the bottleneck. Correctness first is not a slogan here: `kQwen4ExpGated
// Residual` reduces the same shape in DOUBLE, and W6-CUDA's split table records
// that the two must NOT be unified. A device arm that picked its own width would
// have unified them by accident.
//
// ─── BYTE IDENTITY, AND THE INTRINSICS IT REQUIRES ───────────────────────────
// The host provider is pinned to `-ffp-contract=off` (CMakeLists.txt:41-56), so
// `sumsq += v * v` on the host is a SEPARATE round-to-nearest multiply and add.
// nvcc's `-fmad` defaults to ON and is NOT pinned, so the same line written
// plainly here would contract into one fma and produce a different — slightly
// more accurate — answer, which is exactly the break
// `src/vt/cuda/cuda_conv1d_general.cu:138-141` records. Every arithmetic
// operation below is therefore spelled with an `_rn` intrinsic. The two that are
// not are `sqrtf` and the `__frcp_rn` over it: IEEE-754 requires both to be
// correctly rounded and nvcc's defaults (`-prec-sqrt=true`, `-prec-div=true`)
// keep them so, which is the same guarantee glibc gives, so they agree by the
// standard rather than by an intrinsic. There is no transcendental anywhere in
// this op, so unlike `vt::Qwen4ExpPleGate` this arm has NO divergence source at
// all and its gate is a `memcmp`.
//
// ─── DTYPE ────────────────────────────────────────────────────────────────────
// A RUNTIME TAG, not a template parameter, for the argument
// `src/vt/cuda/cuda_qwen4_exp.cu` sets out in full: three independently-typed
// float operands, a tag that is a kernel-wide scalar so its switch is a
// warp-UNIFORM branch rather than divergence, and a spelling that reads as the
// CPU arm's `LoadF32`/`StoreF32` line for line. The op contract
// (`src/vt/ops.cpp::RmsNormGroup`) admits f32/f16/bf16 in and f32/bf16 out, and
// this arm implements all of them: a device arm that refused a dtype its CPU
// sibling accepts would be a divergence to record.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// <math.h>, not <cmath>: CUDA's math headers declare the device functions this
// file uses in the GLOBAL namespace, and spelling the include this way keeps the
// file parseable by a plain host compiler as well.
#include <math.h>

#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

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

// Local dtype tag + accessors, the `cuda_qwen4_exp_ple.cu` arrangement: a local
// copy rather than a hoist, because hoisting would edit a translation unit
// several other rows are working in. `__float2bfloat16` / `__float2half` are
// round-to-nearest-even, identical to the host `vt::F32ToBF16` / `vt::F32ToF16`.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("cuda rmsnorm_group: unsupported ") + what +
                          " dtype (f32/f16/bf16)");
      return DTag::kF32;
  }
}

__device__ inline float LoadAt(const void* p, DTag tag, int64_t i) {
  switch (tag) {
    case DTag::kF32: return static_cast<const float*>(p)[i];
    case DTag::kF16: return __half2float(static_cast<const __half*>(p)[i]);
    default: return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
  }
}

__device__ inline void StoreAt(void* p, DTag tag, int64_t i, float v) {
  switch (tag) {
    case DTag::kF32: static_cast<float*>(p)[i] = v; break;
    case DTag::kF16: static_cast<__half*>(p)[i] = __float2half(v); break;
    default: static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(v); break;
  }
}

// ---------------------------------------------------------------------------
// ONE THREAD PER (row, group). The two loops below are `RmsNormGroupKernel`'s
// inner two, index for index, with each host `+`/`*` replaced by the intrinsic
// that spells the host's rounding.
__global__ void RmsNormGroupCudaKernel(void* out, DTag out_tag, const void* x, DTag x_tag,
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
      // broadcast per group.
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

void RmsNormGroupKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                            const RmsNormGroupArgs& args) {
  const int64_t rows = x.shape[0];
  const int64_t h = x.shape[1];
  const int64_t groups = h / args.group_size;  // the dispatcher checked divisibility
  const int64_t total = rows * groups;
  if (total == 0) return;  // empty-work early return, before the launch
  const DTag out_tag = TagOf(out.dtype, "out");
  const DTag x_tag = TagOf(x.dtype, "x");
  const DTag w_tag = TagOf(weight.dtype, "weight");
  RmsNormGroupCudaKernel<<<GridFor(total), kBlock, 0, AsStream(q)>>>(
      out.data, out_tag, x.data, x_tag, weight.data, w_tag, rows, h, args.group_size, groups,
      args.eps, args.gemma);
  Check(cudaGetLastError(), "rmsnorm_group launch");
}

// Registers the CUDA arm during static init (pre-main, like cuda_layernorm.cu's
// Registrar). Filling the op table is harmless on a machine without a GPU: the
// kCUDA backend never registers there, so no CUDA queue can exist to dispatch
// with. UNCONDITIONAL and at preprocessor depth 0, which is what
// `scripts/check-cuda-op-arch-gate.py` requires of any TU that registers an op —
// this kernel has no cutlass, no arch literal and no feature dependency of any
// kind, so a build gate on it would be the #960 defect in a new place.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kRmsNormGroup, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RmsNormGroupFn>(&RmsNormGroupKernelCuda)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cuda
