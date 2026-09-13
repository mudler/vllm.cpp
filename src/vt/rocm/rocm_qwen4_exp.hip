// ROCm/HIP arm of the Qwen4-Exp (`Qwen3.8-Flash-Next`) gated-residual
// hyper-connection stream — `vt::Qwen4ExpGatedResidual` (the MIXER) and
// `vt::Qwen4ExpGatedResidualWriteBack` (the rank-1 write-back).
// Row MODEL-MM-QWEN4-EXP, W1 of `.agents/specs/qwen4-exp-rocm-ops.md`,
// issue ISSUE-LOCAL-01M2A1DTCZQVAH7M193XT9PN2V.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`src/vt/rocm/rocm_glm5_next_kpool.hip`, `src/vt/rocm/rocm_moe_router.hip`):
// no existing kernel TU and no shared op array is edited. The file name mirrors
// its CUDA sibling `src/vt/cuda/cuda_qwen4_exp.cu` one for one, which the spec's
// port map requires.
//
// ─── WHAT THIS IS A PORT OF, AND WHICH ORACLE DECIDED EACH LINE ──────────────
// The donor is `src/vt/cuda/cuda_qwen4_exp.cu` (458 lines), transcribed. Read
// the two side by side: the only edits are the HIP spelling table
// (`cuda_runtime.h` -> `hip/hip_runtime.h`, `cuda_bf16.h` -> `hip/hip_bf16.h`,
// `__nv_bfloat16` -> `__hip_bfloat16`, `cudaStream_t` -> `hipStream_t`,
// `cudaMalloc`/`cudaFree`/`cudaGetLastError` -> their `hip*` names, and
// `vt::cuda` -> `vt::rocm`). Every kernel body, every index decomposition and
// every rounding intrinsic is the donor's, character for character.
//
// THE DONOR IS NOT THE ONLY ORACLE. vLLM at the pin `e126687a9a` ships a
// first-party AMD backend for this architecture, `vllm/models/qwen4_exp/amd/`,
// and AGENTS.md makes it the mirror wherever it defines behaviour. Its
// `amd/ops/hc.py` implements this same stream in Triton. The three places where
// its spelling differs from the donor's are recorded in the spec's D3a section
// rather than silently resolved here; none of the three is an algorithm
// difference, and the spec says why each was resolved the way it was.
//
// ─── WHY THE `_rn` INTRINSICS SURVIVE THE PORT ───────────────────────────────
// The host provider is pinned to `-ffp-contract=off` (CMakeLists.txt:41-56), so
// the CPU arm evaluates `hyper + block_out * w` as a SEPARATE round-to-nearest
// multiply and a round-to-nearest add. In the DONOR that is the whole reason
// the intrinsics are there: `nvcc`'s `-fmad` defaults to ON and is not pinned,
// so `a + b * w` written plainly would fuse into one FMA and answer a
// different — slightly more accurate — number.
//
// **THE HIP LANE IS DIFFERENT AND THE DIFFERENCE IS STATED RATHER THAN
// ASSUMED.** This build pins `-ffp-contract=off` for HIP as well
// (CMakeLists.txt:414), which the CUDA lane does not, so the plain spelling
// would already round the host's way here. The intrinsics are kept anyway, for
// two reasons that outlive the flag: the file must be diffable against its
// donor line for line, which is how a transcription is reviewed at all; and an
// arm whose numerics depend on a compile option staying set is an arm that
// changes answer when somebody adds a flag. `__fmul_rn`/`__fadd_rn` say what is
// meant in the source instead.
//
// ─── THE ORACLE DEFINES BOTH THE SHAPE AND THE WIDTH, AND IT WAS READ ───────
// AGENTS.md makes vLLM the primary reference wherever it implements the
// behaviour, and it implements THIS one. `vllm/models/qwen4_exp/amd/ops/hc.py`
// at the pin `e126687a9a` is vLLM's own first-party ROCm arm of this exact
// norm, and `_grouped_gemma_rmsnorm_kernel` (`:13-51`) settles both questions
// this file used to answer from its own CPU sibling:
//
//   `_grouped_gemma_rmsnorm_kernel[(N * num_groups,)](...)`            `:66`
//   `x = tl.load(x_ptr + row * stride_x + offsets, ...).to(tl.float32)` `:42`
//   `rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)`                  `:45`
//
// The GRID IS `N * num_groups` PROGRAMS -- one BLOCK per (token, hc stream)
// group, never one thread -- with `BLOCK_SIZE = next_power_of_2(GROUP_DIM)`, so
// the whole group is one vector inside one program and `tl.sum` folds it as a
// TREE. The accumulation is `tl.float32`. eps is inside the rsqrt, added to the
// mean square. vLLM's own PyTorch reference for the same norm agrees on the
// width at `common/hyperconnection.py:75`, `hidden_states.float()`.
//
// `amd/ops/hc.py` AND `nvidia/ops/hc.py` ARE THE SAME FILE -- diffed at the pin,
// byte-identical but for one docstring line, 487 lines each -- so this is not an
// AMD-specific shape a ROCm arm inherits by accident. It is what vLLM computes
// on both backends, and what our own CUDA arm has computed since `ee0644eab`.
//
// So the kernel below is a MIRROR and not an invention: one block per group and
// an f32 tree is what the oracle computes, and the one-thread-per-group `double`
// walk this file shipped until now diverged from it on BOTH axes. The remaining
// difference is below the behaviour line -- vLLM lets Triton pick the program
// shape for a 4096-wide vector where this kernel takes a fixed `kBlock` and
// strides, so the two trees have different DEPTHS while having the same width
// and the same order-independence. That is what the gate's derived bound is
// sized from.
//
// ONE vLLM SPELLING IS DELIBERATELY NOT ADOPTED, and it is the one D3a of
// `.agents/specs/qwen4-exp-rocm-ops.md` already argued: `hc.py:47-48` writes the
// Gemma affine as `y = x * rrms; y += y * w` and says in its own comment that
// the form exists "to lower to an FMA". This file keeps `(x*rrms) * (1 + w)`,
// which is what vLLM's OWN reference defines at
// `common/hyperconnection.py:87` (`normalized * (1.0 + self.weight.float())`).
// A lowering hint is not a behaviour. Nothing in this change touches it.
//
// ─── THE REDUCTION WIDTH IS THE ORACLE'S, AND THIS PARAGRAPH REPLACES THE ONE
//     THAT SAID IT WAS INHERITED ────────────────────────────────────────────
// `HcGroupedNormKernel` below accumulates the grouped sum of squares in
// **float**. This paragraph used to say the opposite -- that the `double` was
// "inherited, not chosen" from `cpu_qwen4_exp.cpp` and that unifying it with
// `vt::RmsNormGroup`'s f32 would be an accident -- and that reading was wrong
// about which arm the convention belongs to. The CPU reference's own header
// settles it at `src/vllm/model_executor/models/qwen4_exp_hc.h:99-104`: the
// double is "the `deepseek_v4_mhc.cpp` house convention for a HOST REFERENCE",
// upstream runs the norm in fp32 (`self._norm(x.float())`) and vLLM likewise,
// and "the device arm is the thing that must be fp32-accumulate and gated
// against these numbers". A device arm carrying the reference's width was the
// outlier; this moves it onto the oracle. The CUDA twin made the same move in
// `ee0644eab` and this file mirrors it rather than diverging from it.
//
// THE COST IS REAL AND IS NAMED RATHER THAN HIDDEN. On MAGNITUDE-SEPARATED data
// -- the shape `test_qwen4_exp_hc_device.cpp:504` was built for -- an f32
// accumulator is a worse approximation of the reference than a double one. That
// case measures a serial f32 walk at 6.702e-04 against a double walk's
// 1.173e-06, and it runs the CPU arm ONLY, so it cannot red this kernel. The
// number this kernel actually pays is not that one, because the reduction below
// is a TREE about ten deep at H = 2560 rather than a 2560-deep serial chain,
// and `test_qwen4_exp_rocm_reductions.cpp`'s magnitude-separated case measures
// and prints it against that same 1e-5 bar.
//
// The `1 +` gamma fold stays f32, which is upstream's width, and it is NOT
// covered by any of this: the accumulator width question is about the REDUCTION
// and must not be allowed to move the multiplier.
//
// ─── NO MFMA, NO DP4A, AND NO WAVEFRONT *SIZE* ASSUMPTION ───────────────────
// This paragraph used to say "no cross-lane primitive" as well, and that clause
// is now false: `HcGroupedNormKernel` folds its per-group partials with
// `__shfl_down`, because the alternative to a cross-lane reduction was the
// four-active-threads launch this file shipped until now.
//
// WHAT STILL HOLDS, AND IT IS THE PART THAT MATTERED. No matrix intrinsic and
// no MFMA -- those are CDNA-only and would have made this file refuse to build
// for RDNA. And no wavefront SIZE is assumed anywhere: every lane constant in
// the reduction is derived from `warpSize`, which HIP defines as
// `__builtin_amdgcn_wavefrontsize()` and constant-folds per target, so the tree
// is 5 levels on a wave32 RDNA part and 6 on a wave64 CDNA one from the same
// source. The shared partial array is sized for the MAXIMUM wave count a
// `kBlock` block can have (wave32); a wave64 target uses the first HALF of it
// -- `kBlock / 32` is 8 slots and a wave64 block of 256 threads has `nwave` 4 --
// and the `lane < nwave` guard keeps the fold off the slots no wave wrote.
//
// ONLY WAVE32 IS MEASURED. `strix:gpu0` (gfx1151, RDNA 3.5) is the only AMD part
// in this fleet, so the wave64 path is REASONED and not run. The spec's
// `## Owed` says so rather than letting the silence read as coverage.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

// ONE BLOCK PER ITEM, where `GridFor` gives one THREAD per item. Same 4096-block
// cap, which is why every kernel launched through this one keeps a grid-stride
// loop over its items.
unsigned GridForGroups(int64_t n) {
  return static_cast<unsigned>(n < 4096 ? (n > 0 ? n : 1) : 4096);
}

// The donor's runtime dtype tag. Local to this TU for the reason its own header
// gives: hoisting it would edit a translation unit several other rows are
// working in, which is the shared-file lock AGENTS.md "Records" names.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("rocm qwen4_exp_gated_residual: unsupported ") + what +
                          " dtype (f32/f16/bf16)");
      return DTag::kF32;
  }
}

// `__float2bfloat16` / `__float2half` are round-to-nearest-even on HIP as on
// CUDA, identical to the host `vt::F32ToBF16` / `vt::F32ToF16`.
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
// vt::Qwen4ExpGatedResidualWriteBack — the rank-1 update, IN PLACE on `hyper`.
//
//   hyper[t, j*H + h] += block_out[t, h] * injection[t, j]
//
// One thread per OUTPUT element. `hyper` is `[T, hc*H]` contiguous, so the flat
// grid index IS the `hyper` offset and the decomposition below is exact:
// `idx = t*(hc*H) + j*H + h`.
//
// vLLM's AMD arm spells the identical arithmetic at `amd/ops/hc.py`
// `_hc_combine_kernel`: `out = res + block[None, :] * inj[:, None]`, with `inj`
// broadcast along the hc axis exactly as `t * hc + j` reads it here. Its two
// differences from this kernel — that it writes a fresh `out` rather than
// updating in place, and that it computes `2*sigmoid(logits/hc)` inside the
// same kernel instead of consuming an already-scaled `injection` — are
// FUSION-BOUNDARY differences, not arithmetic ones, and the spec's D3a records
// both. The boundary here is this tree's op contract
// (`include/vt/ops.h::Qwen4ExpGatedResidualWriteBackFn`), which the CPU oracle
// and the CUDA arm already share; moving it for one device would make the ROCm
// arm answer a different op.
__global__ void GatedResidualWriteBackKernel(void* hyper, DTag hyper_tag, const void* block_out,
                                             DTag block_tag, const void* injection, DTag inj_tag,
                                             int64_t hc, int64_t hidden, int64_t n) {
  const int64_t flat = hc * hidden;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t t = idx / flat;
    const int64_t rem = idx - t * flat;
    const int64_t j = rem / hidden;
    const int64_t h = rem - j * hidden;
    const float w = LoadAt(injection, inj_tag, t * hc + j);
    const float base = LoadAt(hyper, hyper_tag, idx);
    const float add = LoadAt(block_out, block_tag, t * hidden + h);
    // TWO roundings, never one fma — see the header.
    StoreAt(hyper, hyper_tag, idx, __fadd_rn(base, __fmul_rn(add, w)));
  }
}

void Qwen4ExpGatedResidualWriteBackKernelRocm(Queue& q, Tensor& hyper, const Tensor& block_out,
                                              const Tensor& injection,
                                              const Qwen4ExpGatedResidualArgs& args) {
  // `args.lowrank` and `args.eps` are DELIBERATELY not read, as in both
  // siblings: the args struct is shared with `vt::Qwen4ExpGatedResidual` so a
  // caller cannot describe the same stream two ways, and reading either field
  // here would be reading a field with no meaning at this op.
  const int64_t hc = args.hc_count;
  const int64_t hidden = args.hidden_size;
  const int64_t T = hyper.shape[0];
  const int64_t n = T * hc * hidden;
  if (n == 0) return;  // empty-work early return, before the launch
  const DTag hyper_tag = TagOf(hyper.dtype, "hyper");
  const DTag block_tag = TagOf(block_out.dtype, "block output");
  const DTag inj_tag = TagOf(injection.dtype, "injection");
  GatedResidualWriteBackKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      hyper.data, hyper_tag, block_out.data, block_tag, injection.data, inj_tag, hc, hidden, n);
  Check(hipGetLastError(), "qwen4_exp_gated_residual_write_back launch");
}

// ---------------------------------------------------------------------------
// vt::Qwen4ExpGatedResidual — THE MIXER, per token:
//
//   normed      = hc_norm(hyper)                 grouped RMS norm, group == H
//   low         = silu( down(normed) / hc )      DIVISION 1, INSIDE the SiLU
//   gate        = sigmoid( up(low) )             NO division here
//   mixed[h]    = mean_j( gate[j,h] * normed[j,h] )
//   injection[j]= 2 * sigmoid( inject(normed)[j] / hc )   DIVISION 2
//
// Four kernels and three `vt::MatmulBT` calls, the donor's decomposition. The
// projections go through the shared seam rather than a private GEMV, which is
// what AGENTS.md "Shared seams" requires and what lets a block-quantized mix
// weight (the released checkpoint stores all 194 of them as Q8_0) reach
// `kMatmulBTQuant` on this device. The CONSEQUENCE is the donor's and is
// stated rather than hidden: a device GEMM re-associates the K reduction, so
// this arm is not bit-identical to its CPU sibling. Its gate is the tolerance
// the cross-device suite measures.
//
// vLLM's AMD arm splits the same stream the same way — `_grouped_gemma_rmsnorm`,
// `_hc_silu`, `_hc_gate_mix` in `amd/ops/hc.py` are stages 1, 2 and 3 below —
// and routes its projections through the standard vLLM linear methods
// (`amd/low_latency_gemm.py` is a 376-byte no-op that keeps them). So the
// decomposition is the mirror's as well as the donor's.

__device__ inline float SigmoidF(float x) {
  // The CPU arm's `Sigmoid`: `1.0f / (1.0f + std::exp(-x))`, in FLOAT. The
  // reciprocal-then-nothing shape is the host's; `x / (1.0f + expf(-x))` would
  // be a different rounding.
  return __frcp_rn(__fadd_rn(1.0f, expf(-x)));
}

// STAGE 1 — the grouped RMS norm, ONE BLOCK PER (token, hc stream) group. Each
// group is `hidden` CONSECUTIVE elements, so no block can see another's group.
//
// THE PER-GROUP WALK IS NO LONGER THE HOST'S ASCENDING SERIAL ONE. It is a
// strided partial per thread folded in a `__shfl_down` tree and then across the
// waves, in f32. Both header sections above say why the width and the shape each
// changed; the short version is that the shape was costing 35.10% of the decode
// kernel budget on `strix:gpu0` at `hc_count = 4` and `T = 1`, where the old
// launch gave this kernel FOUR ACTIVE THREADS and each of them walked H = 2560
// twice.
__global__ void HcGroupedNormKernel(float* normed, const void* hyper, DTag hyper_tag,
                                    const void* hc_norm_w, DTag w_tag, int64_t T, int64_t hc,
                                    int64_t H, float eps) {
  // One slot per WAVE of the block, sized for the largest wave count a `kBlock`
  // block can have -- that is wave32's `kBlock / 32`, 8 slots. A wave64 target
  // has `nwave = 256 / 64 = 4` and so fills the first HALF, and the
  // `lane < nwave` guard below keeps the fold off the rest. This kernel is in an anonymous namespace and has exactly ONE launch
  // site, which passes `kBlock`, so the array is sized from that constant while
  // every index is still computed from `blockDim.x`; a second launch site with a
  // different block size would have to resize it, and that is why there is not
  // one.
  __shared__ float s_part[kBlock / 32];
  __shared__ float s_r;
  const int64_t flat = hc * H;
  const int64_t total = T * hc;
  // `warpSize` is `__builtin_amdgcn_wavefrontsize()` and is a per-target
  // constant, so these three are folded at compile time on both wave widths.
  const unsigned lane = static_cast<unsigned>(threadIdx.x) % static_cast<unsigned>(warpSize);
  const unsigned wave = static_cast<unsigned>(threadIdx.x) / static_cast<unsigned>(warpSize);
  const unsigned nwave = static_cast<unsigned>(blockDim.x) / static_cast<unsigned>(warpSize);
  for (int64_t g = static_cast<int64_t>(blockIdx.x); g < total;
       g += static_cast<int64_t>(gridDim.x)) {
    const int64_t t = g / hc;
    const int64_t j = g - t * hc;
    const int64_t base = t * flat + j * H;
    // `x.reshape(...).pow(2).mean(-1)` over the GROUP, in f32 — upstream's and
    // vLLM's width. Every thread of the block takes a strided slice; a thread
    // whose slice is empty (H < blockDim.x) contributes an exact 0.
    float part = 0.0f;
    for (int64_t h = static_cast<int64_t>(threadIdx.x); h < H;
         h += static_cast<int64_t>(blockDim.x)) {
      const float v = LoadAt(hyper, hyper_tag, base + h);
      part = __fadd_rn(part, __fmul_rn(v, v));
    }
    // `_rn` throughout, for the same reason the rest of this TU spells it out:
    // this build pins `-ffp-contract=off` for HIP, but an arm whose numerics
    // depend on a compile option staying set is an arm that changes answer when
    // somebody adds a flag.
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      part = __fadd_rn(part, __shfl_down(part, static_cast<unsigned>(off)));
    }
    if (lane == 0) s_part[wave] = part;
    __syncthreads();
    if (wave == 0) {
      float v = (lane < nwave) ? s_part[lane] : 0.0f;
      for (int off = warpSize / 2; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down(v, static_cast<unsigned>(off)));
      }
      if (lane == 0) {
        // eps is INSIDE the rsqrt, added to the MEAN SQUARE, never to the norm.
        // The narrowing point is GONE rather than moved: the quotient was
        // already float at this line, and eps is still added to it and nowhere
        // else.
        s_r = __frcp_rn(sqrtf(__fadd_rn(__fdiv_rn(v, static_cast<float>(H)), eps)));
      }
    }
    __syncthreads();
    const float r = s_r;
    // ORDER-INDEPENDENT, so it simply strides across the block: every output
    // element reads operands no other thread writes.
    for (int64_t h = static_cast<int64_t>(threadIdx.x); h < H;
         h += static_cast<int64_t>(blockDim.x)) {
      // THE `1 +` IS THE OP'S. `hc_norm_w` is the RAW HuggingFace gamma, centred
      // on zero (#2218); dropping the fold scales the stream by ~0 and reads as
      // a corrupt checkpoint rather than as a wiring bug. THE FOLD IS f32, which
      // is upstream's width (`1.0 + self.weight.float()`) and not a convenience.
      const float w = LoadAt(hc_norm_w, w_tag, j * H + h);
      normed[base + h] = __fmul_rn(__fmul_rn(LoadAt(hyper, hyper_tag, base + h), r),
                                   __fadd_rn(1.0f, w));
    }
    // NO TRAILING `__syncthreads()` BELONGS HERE, and the argument is the
    // donor's. `s_part` and `s_r` are the only shared state, so the group grid
    // stride has exactly TWO cross-iteration hazards and each already has one of
    // the barriers above between its ends:
    //
    //   `s_part` WAR -- wave 0 READS `s_part[lane]` in the cross-wave fold of
    //   trip g; lane 0 of every wave WRITES `s_part[wave]` at trip g+1. The
    //   SECOND barrier of trip g stands between them.
    //
    //   `s_r` WAR -- every thread READS `s_r` into `r` at trip g; thread 0
    //   WRITES `s_r` at trip g+1 inside the `wave == 0` fold. The FIRST barrier
    //   of trip g+1 stands between them.
    //
    // The two RAW pairs are inside one trip and the same two barriers order
    // them. Every step of this rests on the trip count being BLOCK-UNIFORM, and
    // it is: `g` is seeded from `blockIdx.x` and stepped by `gridDim.x`, neither
    // of which varies within a block. `r` is a REGISTER copy, and
    // `__syncthreads()` is a compiler barrier as well as a hardware one, so the
    // load of `s_r` cannot be sunk past the next trip's first barrier.
    //
    // THE DONOR'S RACECHECK MEASUREMENT IS NOT INHERITED AS EVIDENCE. CUDA's
    // `compute-sanitizer --tool racecheck` reported `0 hazards displayed` on the
    // equivalent bytes; no ROCm equivalent was run for this file, and the spec's
    // `## Owed` records that rather than borrowing the CUDA result. What is
    // gated here is the VALUE, by the grid-cap case -- the only committed case
    // that takes a second trip.
  }
}

// STAGE 2 — DIVISION 1, inside the SiLU, on the [R] low-rank intermediate,
// BEFORE the activation. `F.silu(down(x) / hc_count)`. SiLU is not homogeneous,
// so `silu(a)/hc` is a different function and the placement is load-bearing.
// vLLM's `_hc_silu_kernel` divides in exactly this position (`x = load(...) /
// HC; y = x * sigmoid(x)`), which is the mirror agreeing with the donor.
__global__ void HcSiluScaleKernel(float* low, int64_t n, float hc_f) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    const float a = __fdiv_rn(low[i], hc_f);
    low[i] = __fmul_rn(a, SigmoidF(a));
  }
}

// STAGE 3 — `torch.sigmoid(up(...))` with NO division, then the unflattened
// gate multiplied against the NORMED stream — not the raw one — then
// `.mean(dim=-2)`. A MEAN over the branches, never a sum, and a TRUE division
// by `hc_count`: 1/hc is inexact for any hc that is not a power of two, so a
// reciprocal multiply would put a one-ulp wedge between this arm and the oracle
// at hc_count = 3. vLLM's `_hc_gate_mix_kernel` accumulates over the hc streams
// ascending and divides at the end (`acc /= HC`) in the same order.
__global__ void HcMixKernel(void* mixed, DTag mixed_tag, const float* gate_pre,
                            const float* normed, int64_t T, int64_t hc, int64_t H,
                            float hc_f) {
  const int64_t flat = hc * H;
  const int64_t total = T * H;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step) {
    const int64_t t = idx / H;
    const int64_t h = idx - t * H;
    float acc = 0.0f;
    for (int64_t j = 0; j < hc; ++j) {
      const int64_t p = t * flat + j * H + h;
      acc = __fadd_rn(acc, __fmul_rn(SigmoidF(gate_pre[p]), normed[p]));
    }
    StoreAt(mixed, mixed_tag, idx, __fdiv_rn(acc, hc_f));
  }
}

// STAGE 4 — DIVISION 2, inside the injection sigmoid, whole sigmoid scaled by 2.
// `2 * sigmoid(inject(x) / hc_count)`. Range (0, 2), exactly 1.0 at a zero
// logit, so an untrained branch is the identity rather than a half-scale.
// vLLM writes the identical expression, `2.0 * tl.sigmoid(inj / HC)`, inside
// `_hc_combine_kernel` instead of in its own pass; see D3a.
__global__ void HcInjectKernel(void* injection, DTag inj_tag, const float* inject_pre,
                               int64_t n, float hc_f) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    StoreAt(injection, inj_tag, i,
            __fmul_rn(2.0f, SigmoidF(__fdiv_rn(inject_pre[i], hc_f))));
  }
}

// An f32 scratch view on the device, built the way `Tensor::Contiguous` builds
// one. These four views are FRESH f32 buffers this function allocated, so they
// carry no load-time marker to lose; the three WEIGHTS are passed to
// `vt::MatmulBT` as the caller's own `Tensor` objects, untouched, so whatever
// markers they carry survive by construction. Nothing here rebuilds a weight
// view — a marker dropped at this boundary produces plausible numbers and no
// crash, which is the failure the donor's header records.
Tensor ScratchF32(float* p, Device dev, int64_t rows, int64_t cols) {
  return Tensor::Contiguous(p, DType::kF32, dev, {rows, cols});
}

void Qwen4ExpGatedResidualKernelRocm(Queue& q, Tensor& mixed, Tensor* injection,
                                     const Tensor& hyper, const Tensor& hc_norm_w,
                                     const Tensor& mix_down, const Tensor& mix_up,
                                     const Tensor* block_inject,
                                     const Qwen4ExpGatedResidualArgs& args) {
  const int64_t hc = args.hc_count;
  const int64_t H = args.hidden_size;
  const int64_t R = args.lowrank;
  const int64_t flat = hc * H;
  const int64_t T = hyper.shape[0];
  if (T == 0 || flat == 0) return;  // empty-work early return, before any alloc
  const float hc_f = static_cast<float>(hc);
  const DTag hyper_tag = TagOf(hyper.dtype, "hyper");
  const DTag w_tag = TagOf(hc_norm_w.dtype, "hc_norm weight");
  const DTag mixed_tag = TagOf(mixed.dtype, "mixed");

  // ONE allocation, sliced. `normed` and `gate_pre` are [T, flat]; `low` is
  // [T, R]; `inject_pre` is [T, hc] and is allocated only on the live arm.
  const int64_t n_normed = T * flat;
  const int64_t n_low = T * R;
  const int64_t n_gate = T * flat;
  const int64_t n_inject = (block_inject != nullptr) ? T * hc : 0;
  const size_t bytes =
      static_cast<size_t>(n_normed + n_low + n_gate + n_inject) * sizeof(float);
  float* scratch = nullptr;
  Check(hipMalloc(&scratch, bytes), "qwen4_exp_gated_residual scratch alloc");
  float* d_normed = scratch;
  float* d_low = d_normed + n_normed;
  float* d_gate = d_low + n_low;
  float* d_inject = d_gate + n_gate;

  // A scope guard, because every `vt::` call below can throw (an unregistered
  // op, a refused dtype) and a scratch leak on a device is not recoverable.
  struct FreeGuard {
    float* p;
    ~FreeGuard() { if (p != nullptr) (void)hipFree(p); }
  } guard{scratch};

  // ONE BLOCK PER GROUP, not one thread per group. `GridForGroups` keeps the
  // same 4096-block cap `GridFor` applies, which is why the kernel's group loop
  // is a grid stride.
  HcGroupedNormKernel<<<GridForGroups(T * hc), kBlock, 0, AsStream(q)>>>(
      d_normed, hyper.data, hyper_tag, hc_norm_w.data, w_tag, T, hc, H, args.eps);
  Check(hipGetLastError(), "qwen4_exp_gated_residual norm launch");

  Tensor t_normed = ScratchF32(d_normed, hyper.device, T, flat);
  Tensor t_low = ScratchF32(d_low, hyper.device, T, R);
  Tensor t_gate = ScratchF32(d_gate, hyper.device, T, flat);
  // `out[M,N] = a[M,K] @ b^T`, b in [N,K] row-major — the SAME orientation the
  // CPU arm's `LinearNoBias` walks, which is ggml's src0 layout and GGUF's disk
  // order, so a block-typed weight needs no transpose. `mix_down` is
  // [R, hc*H] and `mix_up` is [hc*H, R], exactly as the dispatcher checks.
  MatmulBT(q, t_low, t_normed, mix_down);
  HcSiluScaleKernel<<<GridFor(n_low), kBlock, 0, AsStream(q)>>>(d_low, n_low, hc_f);
  Check(hipGetLastError(), "qwen4_exp_gated_residual silu launch");

  MatmulBT(q, t_gate, t_low, mix_up);
  HcMixKernel<<<GridFor(T * H), kBlock, 0, AsStream(q)>>>(mixed.data, mixed_tag, d_gate,
                                                          d_normed, T, hc, H, hc_f);
  Check(hipGetLastError(), "qwen4_exp_gated_residual mix launch");

  // `block_inject_weight is None` returns `mixed_input` alone, and that early
  // return IS `Qwen4ExpTextModel`'s terminal `use_combine=False` mixer. The
  // dispatcher has already refused a half-specified pair.
  if (block_inject == nullptr) return;
  Tensor t_inject = ScratchF32(d_inject, hyper.device, T, hc);
  MatmulBT(q, t_inject, t_normed, *block_inject);
  const DTag inj_tag = TagOf(injection->dtype, "injection");
  HcInjectKernel<<<GridFor(n_inject), kBlock, 0, AsStream(q)>>>(injection->data, inj_tag,
                                                                d_inject, n_inject, hc_f);
  Check(hipGetLastError(), "qwen4_exp_gated_residual inject launch");
}

// Registers the ROCm arms during static init, the pattern every `rocm_*.hip`
// registrar in this directory uses. UNCONDITIONAL and at preprocessor depth 0:
// an op registered from a feature-gated TU disappears on a build that does not
// compile the feature, the resolver falls through, and on `gfx1151` there is
// nothing to fall through TO — the portable reference tier is not installed
// there (spec D2), so the call refuses by name instead. That refusal is the
// intended failure and a conditional registration would turn it into a silent
// absence.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpGatedResidual, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpGatedResidualFn>(&Qwen4ExpGatedResidualKernelRocm)));
    RegisterOp(OpId::kQwen4ExpGatedResidualWriteBack, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<Qwen4ExpGatedResidualWriteBackFn>(
                   &Qwen4ExpGatedResidualWriteBackKernelRocm)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::rocm
