// CUDA kernels for the Qwen4-Exp (`Qwen3.8-Flash-Next`) Per-Layer-Embedding
// block — `vt::Qwen4ExpPleConv` and `vt::Qwen4ExpPleGate`.
// Row MODEL-MM-QWEN4-EXP W6-CUDA, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`src/vt/cuda/cuda_layernorm.cu`, `src/vt/cuda/cuda_conv1d_general.cu`): no
// existing kernel TU and no shared op array is edited. The file name mirrors its
// CPU sibling `src/vt/cpu/cpu_qwen4_exp_ple.cpp` one for one.
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// The CPU arms in `src/vt/cpu/cpu_qwen4_exp_ple.cpp`, which are themselves the
// port of transformers v5.16.0 `models/qwen4_exp/modeling_qwen4_exp.py`:
//   CONV  `Qwen4ExpTextPLELayer._short_conv` (:1150-1167) with
//         `conv_dilation = config.ngram_size` (:1134) and
//         `short_conv_state_len = (conv_kernel_size - 1) * conv_dilation` (:1135),
//         plus `LinearAttentionLayer.update_conv_state` (cache_utils.py:1036-1075)
//   GATE  `Qwen4ExpTextPLELayer.forward` :1180-1182, flattened at :1184
// vLLM has never registered `qwen4_exp` at any revision, so there is no vLLM
// kernel to mirror; the KERNEL STRUCTURE is mirrored from
// `src/vt/cuda/cuda_layernorm.cu` (`Check`, `AsStream`, `GridFor`, the Registrar)
// and `src/vt/cuda/cuda_conv1d_general.cu` (the round-to-nearest intrinsics, and
// the "byte-identical, not within a tolerance" position its own header argues).
//
// ─── WHY THESE TWO OPS ARE IN THE FIRST CUDA TRANCHE ─────────────────────────
// Neither performs a reduction across a parallel axis. The conv's reduction is
// the FOUR TAPS of one output element, walked by one thread in the kernel's own
// index order, exactly as the host walks them; the gate is elementwise. So there
// is no reduction ORDER for a device arm to choose, and the CPU arms' recorded
// precision contracts transfer unchanged. That is the criterion this wave split
// on, not convenience: the four ops left owed each own a reduction whose WIDTH
// or whose VISIT ORDER is a decision no wave has made (spec `## Owed`).
//
// ─── PRECISION, AND WHAT IS AND IS NOT BYTE-IDENTICAL ────────────────────────
// The host provider is pinned to `-ffp-contract=off` (CMakeLists.txt:41-56) and
// nvcc's `-fmad` defaults to ON and is NOT pinned, so every multiply-add that
// has to match the host is spelled with `__dmul_rn`/`__dadd_rn` here — the same
// measure, for the same reason, that `cuda_conv1d_general.cu:138-141` records.
//
// THE FOUR TAPS ACCUMULATE IN DOUBLE, inheriting the CPU arm's decision rather
// than making a new one. `cpu_qwen4_exp_ple.cpp` accumulates in double to match
// the W2 host reference term for term at the model's 10240-channel width, and
// the spec's `## Owed` states that "an f32-accumulating CUDA kernel does not
// inherit that". A double accumulator over four taps is four `__dadd_rn`s on a
// device that has fp64, so the obligation is met exactly rather than argued
// away, and this arm needs no widening decision of its own.
//
// ONE DIVERGENCE SOURCE REMAINS AND IT IS NAMED RATHER THAN ASSUMED AWAY: the
// SiLU and the sigmoid both evaluate `exp()` in double, and CUDA's libdevice
// `exp` is not required to return the same double as glibc's. Every other
// operation on both paths is IEEE-exact or spelled with an `_rn` intrinsic
// (`sqrt` is correctly rounded by IEEE-754 on both sides, and so is the divide).
// A sub-ulp double difference is below the f32 store's rounding for all but an
// exact tie, so these two arms are expected to be byte-identical in practice and
// are NOT ASSERTED to be: their gate reports the measured `max|diff|` and states
// whether it is exactly zero, and the tolerance it holds is one f32 ulp of the
// value's magnitude rather than a number chosen to pass. `vt::Qwen4Exp
// GatedResidualWriteBack` in the sibling TU has no transcendental and IS gated
// by `memcmp`; the difference between the two gates is this paragraph.
//
// ─── DTYPE ────────────────────────────────────────────────────────────────────
// The dtype is a RUNTIME TAG, not a template parameter — the sibling TU
// `src/vt/cuda/cuda_qwen4_exp.cu` carries the full argument for it. In short:
// the conv has four independently-typed float operands (16 instantiations of a
// short kernel), the tag is a kernel-wide scalar so its switch is a warp-UNIFORM
// branch rather than divergence, and the spelling reads as the CPU arm's
// `LoadF32At`/`StoreF32At` line for line. Templated specialisations are a SPEED
// item in the spec's `## Owed`.
//
// `score` is f32 ONLY, which is the op contract and not this arm's narrowing
// (`src/vt/ops.cpp`: "score must be f32 (it is the sigmoid/sqrt argument)"), so
// it is read through a plain `const float*` exactly as the CPU arm reads it.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// <math.h>, not <cmath>: the device math functions this file uses (`isnan`,
// `fabs`, `sqrt`, `exp`) are declared in the GLOBAL namespace by CUDA's math
// headers, and spelling the include the same way keeps the file parseable by a
// plain host compiler as well — which is how it was checked before it ever
// reached a device.
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

// Local dtype tag + accessors, the `cpu_qwen4_exp_ple.cpp` arrangement: a local
// copy rather than a hoist, because hoisting would edit a translation unit
// several other rows are working in — the shared-file lock AGENTS.md "Records"
// names. `__float2bfloat16` / `__float2half` are round-to-nearest-even,
// identical to the host `vt::F32ToBF16` / `vt::F32ToF16`.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* op, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("cuda ") + op + ": unsupported " + what +
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

// The CPU arm's `Silu`, in double, term for term:
//   `double Silu(double v) { return v * (1.0 / (1.0 + std::exp(-v))); }`
// The reciprocal-then-multiply shape is upstream's and the host's; writing
// `v / (1.0 + exp(-v))` would be a different rounding.
__device__ inline double SiluD(double v) { return v * (1.0 / (1.0 + exp(-v))); }

// ---------------------------------------------------------------------------
// vt::Qwen4ExpPleConv — dilated depthwise causal conv with a persistent ring.
//
// ONE THREAD PER (sequence, channel). The channel axis is depthwise, so the
// channels are independent by construction; the token axis inside a sequence is
// NOT parallelised because the ring write-back at the end reads the same window
// the outputs read, and keeping one thread over the whole segment reproduces the
// host's order exactly and needs no scratch.
//
// `query_start_loc` AND `conv_state_indices` ARE READ ON THE DEVICE, not on the
// host. `n_seqs` is a SHAPE (`query_start_loc.shape[0] - 1`) and so is known
// host-side without a dereference; `t0`, `tokens` and the cache row are read
// inside the kernel from the device pointers. This matters beyond this op: the
// spec's `## Owed` records that the QSA indexer's page translation is a HOST
// read that "refuses a device-resident table by name", and names giving it a
// device-side home as a debt the CUDA wave inherits. This op is the smaller
// instance of the same problem, and it is discharged rather than inherited. The
// validating wrapper in `src/vt/ops.cpp` correspondingly runs its `qsl[0] == 0`
// / non-decreasing / row-in-range checks under `if (q.device.type == kCPU)`
// only, so on a CUDA queue those preconditions are the CALLER's — unchanged by
// this arm, and stated here because a reader of this kernel will ask.
//
// `hist` IS NEVER MATERIALISED. The host builds a `[state_len + tokens]` vector
// per channel; the same values are `conv_state` for the first `state_len`
// columns and `x` after it, so `HistAt` below is that vector as a function. It
// is not an optimisation but a necessity: `tokens` is unbounded on a prefill and
// a device thread has no growable buffer.
__device__ inline double HistAt(int64_t j, const void* conv_state, DTag state_tag, int64_t st,
                                const void* x, DTag x_tag, int64_t t0, int64_t channels,
                                int64_t c, int64_t state_len) {
  // [old state | this chunk] — the `torch.cat` at cache_utils.py:1065 after the
  // pad-and-slice at modeling_qwen4_exp.py:1159-1160. A zeroed cache row IS the
  // first call's left zero pad, which is why this op has no `has_initial_state`.
  if (j < state_len) return static_cast<double>(LoadAt(conv_state, state_tag, st + j));
  return static_cast<double>(LoadAt(x, x_tag, (t0 + j - state_len) * channels + c));
}

__global__ void PleConvKernel(void* out, DTag out_tag, const void* x, DTag x_tag,
                              const void* weight, DTag w_tag, void* conv_state, DTag state_tag,
                              const int32_t* qsl, const int32_t* rows, int64_t channels,
                              int64_t kernel_w, int64_t dilation, int64_t state_len,
                              int64_t n_seqs) {
  const int64_t row_stride = channels * state_len;
  const int64_t total = n_seqs * channels;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step) {
    const int64_t s = idx / channels;
    const int64_t c = idx - s * channels;
    const int64_t t0 = static_cast<int64_t>(qsl[s]);
    const int64_t tokens = static_cast<int64_t>(qsl[s + 1]) - t0;
    // An EMPTY segment is an IDENTITY, and this is an early-out rather than a
    // correctness guard, exactly as the CPU arm's own comment says: with
    // `tokens == 0` the output loop never runs and the write-back would read the
    // very column it is about to overwrite. A padded batch row is the caller
    // that produces one.
    if (tokens <= 0) continue;
    const int64_t row = (rows == nullptr) ? s : static_cast<int64_t>(rows[s]);
    const int64_t st = row * row_stride + c * state_len;

    for (int64_t t = 0; t < tokens; ++t) {
      double acc = 0.0;
      // k = 0..K-1 reads lags {(K-1)*d, ..., 2d, d, 0}: `t + k * dilation`
      // against a window whose current token sits at `t + state_len`, and
      // `(K-1)*dilation == state_len` makes the LAST tap the current token.
      // Causal by that tap, and by nothing else. DILATION IS READ HERE and
      // nowhere else: a kernel that ignored it would compute an ordinary K=4
      // causal conv over the same 9-column state and return a plausible tensor,
      // which is why the gate carries a dilation-separation case.
      for (int64_t k = 0; k < kernel_w; ++k) {
        const double w = static_cast<double>(LoadAt(weight, w_tag, c * kernel_w + k));
        const double h =
            HistAt(t + k * dilation, conv_state, state_tag, st, x, x_tag, t0, channels, c,
                   state_len);
        acc = __dadd_rn(acc, __dmul_rn(w, h));
      }
      StoreAt(out, out_tag, (t0 + t) * channels + c, static_cast<float>(SiluD(acc)));
    }

    // `self.conv_states[state_idx].copy_(full_conv_states[..., -L:])`
    // (cache_utils.py:1068): the last `state_len` columns of [state | chunk],
    // holding the RAW conv input — never the conv output and never the
    // activation. A chunk shorter than the window keeps the tail of the old
    // state ahead of it, unshifted.
    //
    // NO READ-AFTER-WRITE HAZARD, and it is proven rather than hoped: iteration
    // `j` writes ring index `j` and reads ring index `tokens + j`. The indices
    // written before iteration `j2` are `0 .. j2-1`, and `tokens + j2 > j2 - 1`
    // for every `tokens >= 1`, so no already-written column is ever read back.
    // The host arm is safe for the same reason and materialises `hist` only
    // because it can.
    for (int64_t j = 0; j < state_len; ++j) {
      StoreAt(conv_state, state_tag, st + j,
              static_cast<float>(HistAt(tokens + j, conv_state, state_tag, st, x, x_tag, t0,
                                        channels, c, state_len)));
    }
  }
}

void Qwen4ExpPleConvKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                               Tensor& conv_state, const Tensor& query_start_loc,
                               const Tensor* conv_state_indices,
                               const Qwen4ExpPleConvArgs& args) {
  constexpr const char* kOp = "qwen4_exp_ple_conv";
  const int64_t channels = x.shape[1];
  const int64_t kernel_w = weight.shape[1];
  const int64_t state_len = conv_state.shape[2];  // == (kernel - 1) * dilation
  const int64_t n_seqs = query_start_loc.shape[0] - 1;
  const int64_t total = n_seqs * channels;
  if (total == 0) return;  // empty-work early return, before the launch
  const DTag out_tag = TagOf(out.dtype, kOp, "out");
  const DTag x_tag = TagOf(x.dtype, kOp, "x");
  const DTag w_tag = TagOf(weight.dtype, kOp, "weight");
  const DTag state_tag = TagOf(conv_state.dtype, kOp, "conv_state");
  PleConvKernel<<<GridFor(total), kBlock, 0, AsStream(q)>>>(
      out.data, out_tag, x.data, x_tag, weight.data, w_tag, conv_state.data, state_tag,
      query_start_loc.Ptr<int32_t>(),
      conv_state_indices == nullptr ? nullptr : conv_state_indices->Ptr<int32_t>(), channels,
      kernel_w, args.dilation, state_len, n_seqs);
  Check(cudaGetLastError(), "qwen4_exp_ple_conv launch");
}

// ---------------------------------------------------------------------------
// vt::Qwen4ExpPleGate — signed-sqrt gate + broadcast sigmoid scale.
//
//     gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
//     gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
//
// THE CLAMP IS BEFORE THE SQRT and the sign is applied AFTER it, so this is not
// `sqrt(clamp(g))` with a sign carried through: at g == 0 the sign is 0 and the
// 1e-6 floor is cancelled, which is why the origin is handled by the SIGN and
// not by a branch. The clamp's whole effect is the 1e-3 floor it puts on the
// output magnitude, and the wrapper refuses a zero `clamp_min` by name because
// 0 is NOT "no floor". Reordering the clamp past the sqrt passes every shape
// check and fails the goldens.
__device__ inline double SignedSqrtD(double g, double clamp_min) {
  // NaN IS UPSTREAM'S ANSWER HERE, and the zero arm below would swallow it.
  // `torch.sign(NaN) == 0` but `NaN * 0.0 == NaN`, so upstream propagates a NaN
  // gate and a NaN output. Without this line `NaN < clamp_min` is false,
  // `floored` and `root` are NaN, NEITHER sign branch is taken, and the
  // fall-through returns 0.0 — which sigmoids to a perfectly plausible
  // `0.5 * value`, a poison value rendered as a number. `include/vt/ops.h`
  // states that a CUDA arm "inherits the NaN obligation above and owes its own
  // case for it"; this line is that inheritance and the gate carries the case.
  // `+/-inf` and `+/-0.0` need NO guard: they already match upstream term for
  // term, which is why this is spelled for NaN alone and not as a finiteness
  // test.
  if (isnan(g)) return g;
  const double magnitude = fabs(g);
  const double floored = magnitude < clamp_min ? clamp_min : magnitude;
  const double root = sqrt(floored);
  // `torch.sign`: -1, 0 or +1. The zero arm is upstream's and it is reachable —
  // a fully masked row scores exactly zero — so it is spelled out rather than
  // left to a `g > 0 ? +root : -root` that would return -1e-3 for it.
  if (g > 0.0) return root;
  if (g < 0.0) return -root;
  return 0.0;
}

__device__ inline double SigmoidD(double v) { return 1.0 / (1.0 + exp(-v)); }

// ONE BLOCK PER (token, hc stream). The gate weight is one transcendental per
// (t, j) and is broadcast across `hidden`, so it is computed ONCE by thread 0
// into shared memory rather than `hidden` times per block. `value` is read once
// per (t, d) and broadcast across the hc streams; the broadcast is never
// materialised, which is the whole reason this is one op.
__global__ void PleGateKernel(void* out, DTag out_tag, const float* score, const void* value,
                              DTag val_tag, int64_t tokens, int64_t hc, int64_t hidden,
                              double divisor, double clamp_min) {
  __shared__ double s_weight;
  const int64_t pairs = tokens * hc;
  // `pair` depends on blockIdx only, so every thread of a block runs the same
  // iterations and the __syncthreads() below are uniformly reached.
  for (int64_t pair = blockIdx.x; pair < pairs; pair += gridDim.x) {
    __syncthreads();  // every thread is done reading the PREVIOUS s_weight
    if (threadIdx.x == 0) {
      // A TRUE division by the divisor, never a reciprocal multiply: the op
      // holds `gate_divisor` as the divisor and not as its reciprocal precisely
      // so this arm performs the operation upstream performs.
      const double g = static_cast<double>(score[pair]) / divisor;
      s_weight = SigmoidD(SignedSqrtD(g, clamp_min));
    }
    __syncthreads();
    const double w = s_weight;
    const int64_t t = pair / hc;
    const int64_t obase = pair * hidden;  // t*(hc*hidden) + j*hidden == pair*hidden
    for (int64_t d = threadIdx.x; d < hidden; d += blockDim.x) {
      StoreAt(out, out_tag, obase + d,
              static_cast<float>(w * static_cast<double>(LoadAt(value, val_tag,
                                                                t * hidden + d))));
    }
  }
}

void Qwen4ExpPleGateKernelCuda(Queue& q, Tensor& out, const Tensor& score, const Tensor& value,
                               const Qwen4ExpPleGateArgs& args) {
  constexpr const char* kOp = "qwen4_exp_ple_gate";
  const int64_t tokens = score.shape[0];
  const int64_t hc = score.shape[1];
  const int64_t hidden = value.shape[1];
  const int64_t pairs = tokens * hc;
  if (pairs == 0 || hidden == 0) return;  // empty-work early return
  // `score` is f32 by the op contract — it is the argument of a sigmoid AND of
  // a square root — so it is not tagged, it is dereferenced as f32.
  VT_CHECK(score.dtype == DType::kF32,
           "cuda qwen4_exp_ple_gate: score must be f32 (it is the sigmoid/sqrt argument)");
  const DTag out_tag = TagOf(out.dtype, kOp, "out");
  const DTag val_tag = TagOf(value.dtype, kOp, "value");
  const unsigned grid = static_cast<unsigned>(pairs < 4096 ? pairs : 4096);
  PleGateKernel<<<grid, kBlock, 0, AsStream(q)>>>(
      out.data, out_tag, score.Ptr<float>(), value.data, val_tag, tokens, hc, hidden,
      static_cast<double>(args.gate_divisor), static_cast<double>(args.clamp_min));
  Check(cudaGetLastError(), "qwen4_exp_ple_gate launch");
}

// Registers the CUDA Qwen4-Exp PLE kernels during static init (pre-main, like
// cuda_layernorm.cu's Registrar). Filling the op table is harmless on a machine
// without a GPU: the kCUDA backend never registers there, so no CUDA queue can
// exist to dispatch with. UNCONDITIONAL and at preprocessor depth 0 — see the
// sibling TU's Registrar comment for what `scripts/check-cuda-op-arch-gate.py`
// requires and why.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpPleConv, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpPleConvFn>(&Qwen4ExpPleConvKernelCuda)));
    RegisterOp(OpId::kQwen4ExpPleGate, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpPleGateFn>(&Qwen4ExpPleGateKernelCuda)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cuda
