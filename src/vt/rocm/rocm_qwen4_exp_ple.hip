// ROCm/HIP arm of the Qwen4-Exp (`Qwen3.8-Flash-Next`) Per-Layer-Embedding
// block — `vt::Qwen4ExpPleConv` and `vt::Qwen4ExpPleGate`.
// Row MODEL-MM-QWEN4-EXP, W3 of `.agents/specs/qwen4-exp-rocm-ops.md`,
// issue ISSUE-LOCAL-01M2A1DTCZQVAH7M193XT9PN2V.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`src/vt/rocm/rocm_qwen4_exp.hip` from W1, `src/vt/rocm/rocm_glm5_next_kpool.hip`):
// no existing kernel TU and no shared op array is edited, and in particular
// `rocm_ops.hip` is not, because that file is a surface every ROCm row writes
// and AGENTS.md "Records" calls such a file a lock. The file name mirrors its
// CUDA sibling `src/vt/cuda/cuda_qwen4_exp_ple.cu` one for one, which the
// spec's port map requires.
//
// ─── WHAT THIS IS A PORT OF, AND WHICH ORACLE DECIDED EACH LINE ──────────────
// The donor is `src/vt/cuda/cuda_qwen4_exp_ple.cu`, transcribed. Read the two
// side by side: the only edits are the HIP spelling table (`cuda_runtime.h` ->
// `hip/hip_runtime.h`, `cuda_bf16.h` -> `hip/hip_bf16.h`, `__nv_bfloat16` ->
// `__hip_bfloat16`, `cudaStream_t` -> `hipStream_t`, `cudaGetLastError` ->
// `hipGetLastError`, and `vt::cuda` -> `vt::rocm`). Every kernel body, every
// index decomposition and every rounding intrinsic is the donor's.
//
// ─── WHAT vLLM's AMD BACKEND SAYS ABOUT THESE TWO, AND WHAT IT DOES NOT ──────
// Spec D3 makes vLLM's first-party AMD backend the mirror wherever it defines
// behaviour, so `vllm/models/qwen4_exp/amd/ple_layer.py` at the pin
// `e126687a9a` was READ before this file was written rather than after. It
// defines BOTH of these behaviours, and it defines them in PyTorch: unlike its
// sibling `amd/ops/hc.py` (5 `@triton.jit` kernels) and `amd/ops/qsa.py` (6),
// `amd/ple_layer.py` contains ZERO Triton kernels. The spec's D3d section
// records that finding, what the comparison found, and the two differences that
// are not algorithm differences. In one line: the algorithms agree, so D3's
// "where they agree, port ours" applies and this file is a transcription of the
// CUDA arm rather than of the mirror.
//
// ─── PRECISION, AND WHAT IS AND IS NOT BYTE-IDENTICAL ────────────────────────
// THE FOUR TAPS ACCUMULATE IN DOUBLE, inheriting the CPU arm's decision rather
// than making a new one; `cpu_qwen4_exp_ple.cpp` accumulates in double to match
// the W2 host reference term for term at the model's 10240-channel width. The
// donor's header records that "an f32-accumulating CUDA kernel does not inherit
// that", and the same sentence applies to an f32-accumulating HIP one. Four
// `__dadd_rn`s on a board that has fp64 is the obligation met exactly rather
// than argued away. gfx1151 runs fp64 at a reduced rate; this file chooses
// CORRECTNESS-BY-INHERITANCE over that rate, and a same-width f32 variant is a
// SPEED item for a later wave and not a numerics decision this arm may make on
// its own. No throughput number is admissible from this row in any case (spec
// `## Gates`), so there is nothing here to trade the width against yet.
//
// The `_rn` intrinsics survive the port for the reason W1's header gives at
// length: this build pins `-ffp-contract=off` for HIP (CMakeLists.txt:414) so
// the plain spelling would already round the host's way, but the file must stay
// diffable against its donor line for line, and an arm whose numerics depend on
// a compile option staying set changes answer when somebody adds a flag.
//
// ONE DIVERGENCE SOURCE REMAINS AND IT IS NAMED RATHER THAN ASSUMED AWAY: the
// SiLU and the sigmoid both evaluate `exp()` in double, and ROCm's device `exp`
// is not required to return the same double as glibc's — the donor states the
// identical caveat about CUDA's libdevice. Every other operation on both paths
// is IEEE-exact or spelled with an `_rn` intrinsic. So these two arms are
// EXPECTED to be byte-identical in practice and are NOT ASSERTED to be: their
// gate is the cross-device NMSE band, `5e-4`, and it reports the measured value
// beside the verdict.
//
// ─── DTYPE ────────────────────────────────────────────────────────────────────
// The dtype is a RUNTIME TAG, not a template parameter, exactly as in the donor
// and in W1's sibling TU: the conv has four independently-typed float operands
// (16 instantiations of a short kernel), and the tag is a kernel-wide scalar so
// its switch is a wavefront-UNIFORM branch rather than divergence. `score` is
// f32 ONLY, which is the op contract and not this arm's narrowing, so it is
// read through a plain `const float*`.
//
// ─── NO MFMA, NO DP4A, NO WAVEFRONT ASSUMPTION ──────────────────────────────
// Neither kernel uses a cross-lane primitive, a matrix intrinsic or a warp
// width. The conv is a grid-stride loop over independent (sequence, channel)
// pairs; the gate uses ONE shared-memory scalar per block and two
// `__syncthreads()`, which are wavefront-size agnostic. So the file carries no
// gfx-family dependency, and in particular nothing that RDNA lacks: MFMA is
// CDNA-only and nothing here would have reached for it.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// <math.h>, not <cmath>: the device math functions this file uses (`isnan`,
// `fabs`, `sqrt`, `exp`) are declared in the GLOBAL namespace by the HIP math
// headers, and the donor spells the include the same way.
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
// rows are working in — the shared-file lock AGENTS.md "Records" names.
// `__float2bfloat16` / `__float2half` are round-to-nearest-even on HIP as on
// CUDA, identical to the host `vt::F32ToBF16` / `vt::F32ToF16`.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* op, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false, std::string("rocm ") + op + ": unsupported " + what +
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
// the outputs read, and keeping one thread over the whole segment reproduces
// the host's order exactly and needs no scratch.
//
// `query_start_loc` AND `conv_state_indices` ARE READ ON THE DEVICE, not on the
// host: `n_seqs` is a SHAPE and so is known host-side without a dereference,
// and `t0`, `tokens` and the cache row are read inside the kernel from the
// device pointers. The validating wrapper in `src/vt/ops.cpp` correspondingly
// runs its `qsl[0] == 0` / non-decreasing / row-in-range checks under
// `if (q.device.type == kCPU)` only, so on a ROCm queue those preconditions are
// the CALLER's — unchanged by this arm, and stated here because a reader of
// this kernel will ask.
//
// `hist` IS NEVER MATERIALISED. The host builds a `[state_len + tokens]` vector
// per channel; the same values are `conv_state` for the first `state_len`
// columns and `x` after it, so `HistAt` below is that vector as a function. It
// is not an optimisation but a necessity: `tokens` is unbounded on a prefill
// and a device thread has no growable buffer.
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
    // `tokens == 0` the output loop never runs and the write-back would read
    // the very column it is about to overwrite. A padded batch row is the
    // caller that produces one.
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
      // causal conv over the same 9-column state and return a plausible
      // tensor, which is why the gate carries a dilation-separation case.
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

void Qwen4ExpPleConvKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
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
  Check(hipGetLastError(), "qwen4_exp_ple_conv launch");
}

// ---------------------------------------------------------------------------
// vt::Qwen4ExpPleGate — signed-sqrt gate + broadcast sigmoid scale.
//
//     gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
//     gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
//
// vLLM's AMD arm spells the SAME two lines at `amd/ple_layer.py`
// `Qwen4ExpPLELayer.forward`:
//     gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
//     gated_value = gate * value.unsqueeze(-2)
// with the divisor `/ math.sqrt(self.hidden_size)` applied to the dot on the
// line above, which is what `Qwen4ExpPleGateArgs::gate_divisor` holds. Clamp
// before the root, sign after it, sigmoid outside: term for term this kernel.
//
// THE CLAMP IS BEFORE THE SQRT and the sign is applied AFTER it, so this is not
// `sqrt(clamp(g))` with a sign carried through: at g == 0 the sign is 0 and the
// 1e-6 floor is cancelled, which is why the origin is handled by the SIGN and
// not by a branch. Reordering the clamp past the sqrt passes every shape check
// and fails the goldens.
__device__ inline double SignedSqrtD(double g, double clamp_min) {
  // NaN IS UPSTREAM'S ANSWER HERE, and the zero arm below would swallow it.
  // `torch.sign(NaN) == 0` but `NaN * 0.0 == NaN`, so upstream propagates a NaN
  // gate and a NaN output. Without this line `NaN < clamp_min` is false,
  // `floored` and `root` are NaN, NEITHER sign branch is taken, and the
  // fall-through returns 0.0 — which sigmoids to a perfectly plausible
  // `0.5 * value`, a poison value rendered as a number. `include/vt/ops.h`
  // states that a device arm "inherits the NaN obligation above and owes its
  // own case for it"; this line is that inheritance, and the CPU-side golden
  // suite `tests/vllm/models/test_qwen4_exp_ple_gate.cpp` carries the case.
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
  // iterations and the __syncthreads() below are uniformly reached. That
  // property is what makes this kernel wavefront-size agnostic: a barrier that
  // some lanes skipped would be undefined on a 32-lane RDNA wavefront exactly
  // as it is on a 64-lane CDNA one.
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

void Qwen4ExpPleGateKernelRocm(Queue& q, Tensor& out, const Tensor& score, const Tensor& value,
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
           "rocm qwen4_exp_ple_gate: score must be f32 (it is the sigmoid/sqrt argument)");
  const DTag out_tag = TagOf(out.dtype, kOp, "out");
  const DTag val_tag = TagOf(value.dtype, kOp, "value");
  const unsigned grid = static_cast<unsigned>(pairs < 4096 ? pairs : 4096);
  PleGateKernel<<<grid, kBlock, 0, AsStream(q)>>>(
      out.data, out_tag, score.Ptr<float>(), value.data, val_tag, tokens, hc, hidden,
      static_cast<double>(args.gate_divisor), static_cast<double>(args.clamp_min));
  Check(hipGetLastError(), "qwen4_exp_ple_gate launch");
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
    RegisterOp(OpId::kQwen4ExpPleConv, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpPleConvFn>(&Qwen4ExpPleConvKernelRocm)));
    RegisterOp(OpId::kQwen4ExpPleGate, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpPleGateFn>(&Qwen4ExpPleGateKernelRocm)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::rocm
