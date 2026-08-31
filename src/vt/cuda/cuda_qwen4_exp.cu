// CUDA kernels for the Qwen4-Exp (`Qwen3.8-Flash-Next`) gated-residual
// hyper-connection stream — `vt::Qwen4ExpGatedResidualWriteBack` (W6-CUDA) and
// `vt::Qwen4ExpGatedResidual`, the MIXER (W6-CUDA-B).
// Row MODEL-MM-QWEN4-EXP, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`src/vt/cuda/cuda_layernorm.cu`, `src/vt/cuda/cuda_conv1d_general.cu`): no
// existing kernel TU and no shared op array is edited. The file name mirrors its
// CPU sibling `src/vt/cpu/cpu_qwen4_exp.cpp` one for one, as CMakeLists.txt
// already pairs the three `cpu_qwen4_exp*.cpp` units.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// The CPU arm, `src/vt/cpu/cpu_qwen4_exp.cpp::Qwen4ExpGatedResidualWriteBackKernel`,
// which is itself the port of the two verbatim lines of
// `Qwen4ExpTextDecoderLayer.forward` (transformers v5.16.0,
// `models/qwen4_exp/modeling_qwen4_exp.py`):
//
//     injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
//     hidden_states = hyper_input + injection.flatten(-2)
//
// vLLM has never registered `qwen4_exp` at any revision, so there is no vLLM
// kernel to mirror; the KERNEL STRUCTURE (self-registering TU, `Check`,
// `AsStream`, `GridFor`, grid-stride loop, one `Check(cudaGetLastError(), ...)`
// per launch site) is mirrored from `src/vt/cuda/cuda_layernorm.cu` and
// `src/vt/cuda/cuda_ops.cu`'s `MulScalarKernel`, which is this tree's template
// for a small elementwise op.
//
// ─── WHY THE WRITE-BACK WAS IN THE FIRST CUDA TRANCHE ────────────────────────
// It performs NO reduction across a parallel axis. Every output element is one
// multiply and one add over operands it alone reads, so there is no reduction
// ORDER for a device arm to choose and the CPU arm's numeric contract transfers
// unchanged.
//
// **THE SENTENCE THAT STOOD HERE IS NOW FALSE AND IS REPLACED RATHER THAN LEFT
// TO AGE.** It read "The four ops this row leaves owed (`vt::Qwen4ExpGated
// Residual`, `vt::RmsNormGroup`, and the two QSA ops) each own a reduction whose
// width or whose visit order is a decision no wave has made". W6-CUDA-B made all
// four of those decisions and landed all four arms — the mixer below,
// `src/vt/cuda/cuda_rms_norm_group.cu` and `src/vt/cuda/cuda_qwen4_exp_qsa.cu`.
// What each one decided, and why the mixer alone is held to a tolerance rather
// than to a `memcmp`, is in the W6-CUDA-B section of the spec and in the mixer's
// own header below.
//
// ─── BYTE IDENTITY, AND THE INTRINSICS IT REQUIRES ───────────────────────────
// The host provider is pinned to `-ffp-contract=off` (CMakeLists.txt:41-56), so
// the CPU arm evaluates `hyper + block_out * w` as a SEPARATE round-to-nearest
// multiply and a round-to-nearest add. nvcc's `-fmad` defaults to ON and is not
// pinned, so `a + b * w` written plainly here would contract into a single
// fma and produce a DIFFERENT — and slightly more accurate — answer, which is
// exactly the byte-agreement break `src/vt/cuda/cuda_conv1d_general.cu:138-141`
// records for the same reason. `__fmul_rn`/`__fadd_rn` spell the host's two
// roundings out, so this arm is BYTE-IDENTICAL to the CPU arm rather than merely
// close, and its gate is a `memcmp` and not a tolerance. A tolerance would be
// the wrong instrument here: a transposed `injection` axis or a dropped `hc`
// stride lands well inside any epsilon anyone would write.
//
// ─── DTYPE ────────────────────────────────────────────────────────────────────
// The op contract (`src/vt/ops.cpp::Qwen4ExpGatedResidualWriteBack`) admits
// f32/f16/bf16 inputs and f32/bf16 outputs, and this arm implements ALL of them
// rather than narrowing to the f32+bf16 set the tree's other CUDA arms use: a
// device arm that refused a dtype its CPU sibling accepts would be a divergence
// to record, and the tag design below makes admitting it free.
//
// The dtype is carried as a RUNTIME TAG rather than a template parameter, which
// is a deliberate divergence from the `<Tin, Tout>` house style of
// `cuda_ops.cu`. Three operands with independent dtypes is 8 to 18
// instantiations of a kernel whose body is four lines, and — the load-bearing
// half — the tag is a KERNEL-WIDE SCALAR, identical for every thread, so the
// switch is a warp-UNIFORM branch and costs a prediction rather than divergence.
// The spelling also reads as the CPU arm's own `LoadF32At`/`StoreF32At` line for
// line, which is what makes the two arms diffable by eye. Templated
// specialisations are recorded in the spec's `## Owed` as a SPEED item; no speed
// claim on this row is admissible before the MoE adapter hoist in any case.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

// The runtime dtype tag described in the header. Local to this TU for the same
// reason `cpu_qwen4_exp.cpp` keeps a local copy of `LoadF32At`/`StoreF32At`:
// hoisting it would edit a translation unit several other rows are working in,
// which is the shared-file lock AGENTS.md "Records" names.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("cuda qwen4_exp_gated_residual: unsupported ") + what +
                          " dtype (f32/f16/bf16)");
      return DTag::kF32;
  }
}

// `__float2bfloat16` / `__float2half` are round-to-nearest-even, identical to
// the host `vt::F32ToBF16` / `vt::F32ToF16` — the equality this arm's byte
// gate depends on, and the same one `cuda_ops.cu:48-54` asserts.
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
// vt::Qwen4ExpGatedResidualWriteBack — the rank-1 update, IN PLACE on `hyper`.
//
//   hyper[t, j*H + h] += block_out[t, h] * injection[t, j]
//
// One thread per OUTPUT element. `hyper` is `[T, hc*H]` contiguous, so the flat
// grid index IS the `hyper` offset and the decomposition below is exact:
// `idx = t*(hc*H) + j*H + h`. Both llama.cpp implementations of this
// architecture materialise this as `repeat_4d` + `mul` — a dense `[H, hc, T]`
// broadcast built and thrown away 96 times a step (48 layers x 2 sites); the op
// exists so neither arm has to, and the device arm keeps that property.
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
    // TWO roundings, never one fma — see the header. This is the whole reason
    // the gate can be a memcmp.
    StoreAt(hyper, hyper_tag, idx, __fadd_rn(base, __fmul_rn(add, w)));
  }
}

void Qwen4ExpGatedResidualWriteBackKernelCuda(Queue& q, Tensor& hyper, const Tensor& block_out,
                                              const Tensor& injection,
                                              const Qwen4ExpGatedResidualArgs& args) {
  // `args.lowrank` and `args.eps` are DELIBERATELY not read. The args struct is
  // shared with `vt::Qwen4ExpGatedResidual` so a caller cannot describe the same
  // stream two ways (`include/vt/ops.h`, the Qwen4ExpGatedResidualArgs comment),
  // and the CPU arm reads only these two fields as well. Reading either here
  // would be reading a field with no meaning at this op.
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
  Check(cudaGetLastError(), "qwen4_exp_gated_residual_write_back launch");
}

// ---------------------------------------------------------------------------
// vt::Qwen4ExpGatedResidual — THE MIXER. Row MODEL-MM-QWEN4-EXP W6-CUDA-B.
//
// The port of `src/vt/cpu/cpu_qwen4_exp.cpp::Qwen4ExpGatedResidualKernel`, which
// is itself `Qwen4ExpTextHyperConnection.forward` (transformers v5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py:940-971`), per token:
//
//   normed      = hc_norm(hyper)                 grouped RMS norm, group == H
//   low         = silu( down(normed) / hc )      DIVISION 1, INSIDE the SiLU
//   gate        = sigmoid( up(low) )             NO division here
//   mixed[h]    = mean_j( gate[j,h] * normed[j,h] )
//   injection[j]= 2 * sigmoid( inject(normed)[j] / hc )   DIVISION 2
//
// ─── WHY THIS ARM IS FOUR KERNELS AND THREE `vt::MatmulBT` CALLS, NOT ONE ────
// The CPU arm is one fused loop because it can be: it holds `normed` (hc*H
// floats), `low` (R) and `gate` (hc*H) in `std::vector`s per token. A device
// block cannot. At the released config `hc*H` is thousands of floats, well past
// what shared memory holds, so the intermediates live in a device scratch buffer
// and the stages are separate launches. This is the same shape decision
// `cuda_qwen4_exp_ple.cu` records for `hist`, one size up.
//
// **AND THE THREE PROJECTIONS GO THROUGH THE SHARED SEAM, WHICH COSTS THIS ARM
// ITS BYTE-IDENTITY AND IS STILL THE RIGHT CALL.** The CPU arm's `ProjectRow`
// routes a BLOCK-QUANTIZED weight to `vt::MatmulBTQuant` and a float weight to
// its private `LinearNoBias`, which accumulates `sum_i w[o*K+i]*x[i]` in f32 in
// index order — and `LinearNoBias` exists there precisely so the float arm stays
// bit-identical to the pre-W5p kernel. Writing a device `LinearNoBias` here
// would reproduce that order and hand this arm a `memcmp` gate. It would also be
// a hand-written GEMV beside `vt::MatmulBT`, which is the parallel path
// AGENTS.md "Shared seams" forbids, and it would give the released checkpoint's
// Q8_0 mix weights (W5p; all 194 of them) a second, private route on the one
// device where the quantized GEMM lives. So both float and block weights go to
// `vt::MatmulBT`, which dispatches a block dtype to `kMatmulBTQuant` itself,
// and the CONSEQUENCE is stated rather than hidden: a device GEMM re-associates
// the K reduction, so this arm is NOT bit-identical to its CPU sibling and its
// gate is a tolerance the suite MEASURES. Every OTHER stage below is spelled
// with `_rn` intrinsics and keeps the CPU arm's order exactly, so the
// divergence has two named sources and no third: the GEMM's association and
// `exp`.
//
// ─── THE REDUCTION WIDTH IS INHERITED, NOT CHOSEN ───────────────────────────
// `cpu_qwen4_exp.cpp` accumulates the grouped sum of squares in **double**, and
// its own header measures why: "on magnitude-separated data, a float accumulator
// and this one differ by 742x". W6-CUDA's split table records that "an
// f32-accumulating CUDA kernel does not inherit that". This arm therefore
// accumulates in `double` with `__dadd_rn`/`__dmul_rn` and narrows exactly where
// the host narrows — `static_cast<float>(ss / static_cast<double>(H)) + eps` —
// so no width decision is made here at all. `vt::RmsNormGroup`'s CUDA arm in
// `cuda_rms_norm_group.cu` reduces the same SHAPE in **f32**, deliberately, and
// the two must NOT be unified: each keeps the width its own oracle was dumped
// in.
//
// The scratch is ONE `cudaMalloc` sliced four ways, freed before return. A
// per-call allocation in a decode loop is a cost, and it is recorded in the
// spec's `## Owed` as a SPEED item rather than papered over with a static cache
// that would not be re-entrant.

__device__ inline float SigmoidF(float x) {
  // The CPU arm's `Sigmoid`: `1.0f / (1.0f + std::exp(-x))`, in FLOAT. The
  // reciprocal-then-nothing shape is the host's; `x / (1.0f + expf(-x))` would
  // be a different rounding.
  return __frcp_rn(__fadd_rn(1.0f, expf(-x)));
}

// STAGE 1 — the grouped RMS norm, ONE THREAD PER (token, hc stream). Each group
// is `hidden` CONSECUTIVE elements, so no thread can see another's group, and
// the walk is the host's ascending one.
__global__ void HcGroupedNormKernel(float* normed, const void* hyper, DTag hyper_tag,
                                    const void* hc_norm_w, DTag w_tag, int64_t T, int64_t hc,
                                    int64_t H, float eps) {
  const int64_t flat = hc * H;
  const int64_t total = T * hc;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step) {
    const int64_t t = idx / hc;
    const int64_t j = idx - t * hc;
    const int64_t base = t * flat + j * H;
    // `x.reshape(...).pow(2).mean(-1)` over the GROUP. DOUBLE, inherited from
    // the CPU arm rather than chosen — see the header.
    double ss = 0.0;
    for (int64_t h = 0; h < H; ++h) {
      const double v = static_cast<double>(LoadAt(hyper, hyper_tag, base + h));
      ss = __dadd_rn(ss, __dmul_rn(v, v));
    }
    // eps is INSIDE the rsqrt, added to the MEAN SQUARE, never to the norm. The
    // narrowing point is the host's: the quotient becomes float BEFORE eps is
    // added.
    const float r = __frcp_rn(sqrtf(
        __fadd_rn(static_cast<float>(__ddiv_rn(ss, static_cast<double>(H))), eps)));
    for (int64_t h = 0; h < H; ++h) {
      // THE `1 +` IS THE OP'S. `hc_norm_w` is the RAW HuggingFace gamma, centred
      // on zero (#2218); dropping the fold scales the stream by ~0 and reads as
      // a corrupt checkpoint rather than as a wiring bug. THE FOLD IS f32, which
      // is upstream's width (`1.0 + self.weight.float()`, :177) and not a
      // convenience — the double above isolates the REDUCTION and must not also
      // move the multiplier.
      const float w = LoadAt(hc_norm_w, w_tag, j * H + h);
      normed[base + h] = __fmul_rn(__fmul_rn(LoadAt(hyper, hyper_tag, base + h), r),
                                   __fadd_rn(1.0f, w));
    }
  }
}

// STAGE 2 — DIVISION 1, inside the SiLU, on the [R] low-rank intermediate,
// BEFORE the activation. `F.silu(down(x) / hc_count)`. SiLU is not homogeneous,
// so `silu(a)/hc` is a different function and the placement is load-bearing.
__global__ void HcSiluScaleKernel(float* low, int64_t n, float hc_f) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    const float a = __fdiv_rn(low[i], hc_f);
    low[i] = __fmul_rn(a, SigmoidF(a));
  }
}

// STAGE 3 — `torch.sigmoid(up(...))` with NO division, then
// `.unflatten(-1, (hc, H))` multiplied against the NORMED stream — not the raw
// one — then `.mean(dim=-2)`. A MEAN over the branches, never a sum, and a TRUE
// division by `hc_count`: 1/hc is inexact for any hc that is not a power of two,
// so a reciprocal multiply would put a one-ulp wedge between this arm and the
// oracle at hc_count = 3, which is golden case B.
//
// The sigmoid is FUSED into the mix rather than run as its own pass. It is
// elementwise and each `gate` element is read exactly once here, so the fused
// form computes the identical value; a separate pass would only cost a round
// trip through global memory. ONE THREAD PER (token, hidden), accumulating over
// the hc branches ASCENDING, which is the host's order.
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
// one. EVERY MARKER MATTERS AT THIS BOUNDARY: a marker dropped between the host
// and a kernel produces plausible numbers and no crash, which is what
// `## Owed` records for `repacked` (a NaN at layer 0, all-zero logits, every
// token id 0, nothing thrown). These four views are FRESH f32 buffers this
// function allocated, so they carry no load-time marker to lose; the three
// WEIGHTS are passed to `vt::MatmulBT` as the caller's own `Tensor` objects,
// untouched, so whatever markers they carry survive by construction. Nothing
// here rebuilds a weight view.
Tensor ScratchF32(float* p, Device dev, int64_t rows, int64_t cols) {
  return Tensor::Contiguous(p, DType::kF32, dev, {rows, cols});
}

void Qwen4ExpGatedResidualKernelCuda(Queue& q, Tensor& mixed, Tensor* injection,
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
  Check(cudaMalloc(&scratch, bytes), "qwen4_exp_gated_residual scratch alloc");
  float* d_normed = scratch;
  float* d_low = d_normed + n_normed;
  float* d_gate = d_low + n_low;
  float* d_inject = d_gate + n_gate;

  // A scope guard, because every `vt::` call below can throw (an unregistered
  // op, a refused dtype) and a scratch leak on a device is not recoverable.
  struct FreeGuard {
    float* p;
    ~FreeGuard() { if (p != nullptr) cudaFree(p); }
  } guard{scratch};

  HcGroupedNormKernel<<<GridFor(T * hc), kBlock, 0, AsStream(q)>>>(
      d_normed, hyper.data, hyper_tag, hc_norm_w.data, w_tag, T, hc, H, args.eps);
  Check(cudaGetLastError(), "qwen4_exp_gated_residual norm launch");

  Tensor t_normed = ScratchF32(d_normed, hyper.device, T, flat);
  Tensor t_low = ScratchF32(d_low, hyper.device, T, R);
  Tensor t_gate = ScratchF32(d_gate, hyper.device, T, flat);
  // `out[M,N] = a[M,K] @ b^T`, b in [N,K] row-major — the SAME orientation
  // `LinearNoBias` walks, which is ggml's src0 layout and GGUF's disk order, so
  // a block-typed weight needs no transpose (a block row cannot be transposed
  // without requantizing). `mix_down` is [R, hc*H] and `mix_up` is [hc*H, R],
  // exactly as the dispatcher checks.
  MatmulBT(q, t_low, t_normed, mix_down);
  HcSiluScaleKernel<<<GridFor(n_low), kBlock, 0, AsStream(q)>>>(d_low, n_low, hc_f);
  Check(cudaGetLastError(), "qwen4_exp_gated_residual silu launch");

  MatmulBT(q, t_gate, t_low, mix_up);
  HcMixKernel<<<GridFor(T * H), kBlock, 0, AsStream(q)>>>(mixed.data, mixed_tag, d_gate,
                                                          d_normed, T, hc, H, hc_f);
  Check(cudaGetLastError(), "qwen4_exp_gated_residual mix launch");

  // `block_inject_weight is None` returns `mixed_input` alone (:966-967), and
  // that early return IS `Qwen4ExpTextModel`'s terminal `use_combine=False`
  // mixer. The dispatcher has already refused a half-specified pair.
  if (block_inject == nullptr) return;
  Tensor t_inject = ScratchF32(d_inject, hyper.device, T, hc);
  MatmulBT(q, t_inject, t_normed, *block_inject);
  const DTag inj_tag = TagOf(injection->dtype, "injection");
  HcInjectKernel<<<GridFor(n_inject), kBlock, 0, AsStream(q)>>>(injection->data, inj_tag,
                                                                d_inject, n_inject, hc_f);
  Check(cudaGetLastError(), "qwen4_exp_gated_residual inject launch");
}

// Registers the CUDA Qwen4-Exp gated-residual write-back during static init
// (pre-main, like cuda_layernorm.cu's Registrar). Filling the op table is
// harmless on a machine without a GPU: the kCUDA backend never registers there,
// so no CUDA queue can exist to dispatch with.
//
// UNCONDITIONAL, at `if(VLLM_CPP_CUDA)` depth exactly one in CMakeLists.txt and
// at preprocessor depth 0 here. `scripts/check-cuda-op-arch-gate.py` requires
// both: an op registered from a feature-gated TU disappears on an arch that does
// not build it, the resolver falls to the portable CPU tier, and a CPU kernel
// then dereferences device pointers (#960/#844).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpGatedResidual, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpGatedResidualFn>(&Qwen4ExpGatedResidualKernelCuda)));
    RegisterOp(OpId::kQwen4ExpGatedResidualWriteBack, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Qwen4ExpGatedResidualWriteBackFn>(
                   &Qwen4ExpGatedResidualWriteBackKernelCuda)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cuda
