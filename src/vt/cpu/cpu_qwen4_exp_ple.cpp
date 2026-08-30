// CPU kernels for the Qwen4-Exp (`Qwen3.8-Flash-Next`) PLE layer: the dilated
// depthwise causal convolution — `vt::Qwen4ExpPleConv`, row MODEL-MM-QWEN4-EXP
// W5b-3 (#2156) — and the signed-sqrt GATE that feeds it —
// `vt::Qwen4ExpPleGate`, W5e-1 (#2336). Campaign #1978, spec
// `.agents/specs/qwen4-exp-flash-next.md`
// (`### PLE: a strided-history conv with no vLLM op, confirmed`).
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// This row splits its oracles by developer direction (spec `## Oracles`):
// transformers supplies the ALGORITHM, vLLM supplies the OP FORM. Here vLLM
// supplies NEITHER, and that negative is confirmed rather than unfound: at vLLM
// `origin/main` = `6a5e8f5979`, `git grep -in dilat` returns zero lines in
// `vllm/model_executor/layers/mamba/`, zero in `csrc/` and zero in `tests/`, and
// `layers/conv.py` defines only `Conv2dLayer` and `Conv3dLayer`. Upstream
// reached the same conclusion from the other side and hand-rolled it, saying so
// in a comment: "We cannot use the usual functions/kernels here for the short
// conv as the conv1d has dilation".
//
//   ALGORITHM  transformers v5.16.0 (this row's accepted lane pin),
//              `models/qwen4_exp/modeling_qwen4_exp.py`
//                ::Qwen4ExpTextPLELayer._short_conv      (:1150-1167)
//                ::Qwen4ExpTextPLELayer.__init__, the conv  (:1133-1145)
//              `cache_utils.py`
//                ::LinearAttentionLayer.update_conv_state (:1036-1075)
//   OP FORM    none; see above.
//
// The landed HOST reference for the same arithmetic is
// `src/vllm/model_executor/models/qwen4_exp_ple.cpp::PleShortConv` (W2, #1987),
// gated against goldens dumped by EXECUTING the pinned oracle. This kernel is
// gated against THE SAME GOLDENS rather than against that reference, so the two
// arms answer to one oracle instead of to each other, and they are additionally
// required to agree BIT FOR BIT at the model's real 10240-channel width
// (`tests/vllm/models/test_qwen4_exp_ple_device.cpp`).
//
// ─── THE SHAPE OF THE WORK ────────────────────────────────────────────────────
// Depthwise, so every channel is an independent 4-tap FIR over its own history,
// and the history is `(K - 1) * dilation` = 9 columns read at stride 3. The
// per-channel gather into `hist` is the same arrangement the host reference
// uses and the same one a device arm wants: channel-major, so the four taps of
// one output are contiguous-strided in a single row rather than `C` apart.
//
// A `hist` buffer is materialised per channel rather than indexing `x` and
// `conv_state` directly, because the window STRADDLES them — output t < 9 reads
// some taps from the state and some from this chunk — and the branch that would
// replace it sits in the innermost loop of a 10240-channel kernel. It costs
// `state_len + tokens` floats, reused across channels.
//
// ─── PRECISION ────────────────────────────────────────────────────────────────
// Taps accumulate in DOUBLE and the SiLU is evaluated in double, matching the
// host reference term for term, then ONE f32 (or bf16) store. Four terms is a
// short reduction, so this is not the load-bearing width the gated-residual
// norm needed; it is here so that the two arms are bit-identical rather than
// merely close, which is what lets the device arm replace the host one without
// re-gating the whole layer. A CUDA arm that accumulates in f32 will not
// inherit that identity and must be gated against the oracle directly; the
// spec's `## Owed` records that.
//
// ─── SCOPE ────────────────────────────────────────────────────────────────────
// Nothing here is registered for any device but kCPU, so the dispatcher refuses
// BY NAME on every other one rather than silently falling back. The CUDA arm is
// OWED, not written: this is a CPU-only host and an ungated kernel is worse
// than an absent one.
#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

namespace vt::cpu {
namespace {

// Local dtype accessors, the `cpu_layernorm.cpp` / `cpu_qwen4_exp.cpp`
// arrangement: `cpu_ops.cpp`'s `LoadF32`/`StoreF32` are file-static there and
// hoisting them would edit a 3900-line translation unit several other rows are
// working in, which is the shared-file lock AGENTS.md "Records" names.
float LoadF32At(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "qwen4_exp_ple: unsupported input dtype"); return 0.0f;
  }
}

void StoreF32At(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kF16: t.Ptr<uint16_t>()[i] = F32ToF16(v); break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "qwen4_exp_ple: unsupported output dtype");
  }
}

// The host reference's `Silu`, in double, term for term.
double Silu(double v) { return v * (1.0 / (1.0 + std::exp(-v))); }

void Qwen4ExpPleConvKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                           Tensor& conv_state, const Tensor& query_start_loc,
                           const Tensor* conv_state_indices,
                           const Qwen4ExpPleConvArgs& args) {
  const int64_t channels = x.shape[1];
  const int64_t kernel = weight.shape[1];
  const int64_t dilation = args.dilation;
  const int64_t state_len = conv_state.shape[2];  // == (kernel - 1) * dilation
  const int64_t n_seqs = query_start_loc.shape[0] - 1;
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* rows =
      conv_state_indices == nullptr ? nullptr : conv_state_indices->Ptr<int32_t>();
  float* state_base = conv_state.Ptr<float>();
  const int64_t row_stride = channels * state_len;

  std::vector<double> hist;
  for (int64_t s = 0; s < n_seqs; ++s) {
    const int64_t t0 = qsl[s];
    const int64_t tokens = static_cast<int64_t>(qsl[s + 1]) - t0;
    // An EMPTY segment is an IDENTITY either way, and this line is an early-out
    // rather than a correctness guard — said plainly because the comment that
    // stood here first claimed it stopped the cache being shifted, and it does
    // not: with `tokens == 0` the window loop never runs and the write-back
    // reads `hist[0 + j]`, which is the column it is about to overwrite. The
    // dispatcher already refuses a decreasing `query_start_loc`, so 0 is the
    // only value that reaches here. A padded batch row is the caller that
    // produces one, and `test_qwen4_exp_ple_device.cpp` pins the identity.
    if (tokens <= 0) continue;
    const int64_t row = rows == nullptr ? s : static_cast<int64_t>(rows[s]);
    float* st = state_base + row * row_stride;

    const int64_t span = state_len + tokens;
    hist.assign(static_cast<size_t>(span), 0.0);
    for (int64_t c = 0; c < channels; ++c) {
      // [old state | this chunk], the `torch.cat` at cache_utils.py:1065 after
      // the pad-and-slice at modeling_qwen4_exp.py:1159-1160. The two are
      // arithmetically the same thing: a first call left-zero-pads, and a zeroed
      // cache row IS that padding, which is why this op has no
      // `has_initial_state`.
      for (int64_t j = 0; j < state_len; ++j) {
        hist[static_cast<size_t>(j)] = st[c * state_len + j];
      }
      for (int64_t t = 0; t < tokens; ++t) {
        hist[static_cast<size_t>(state_len + t)] =
            LoadF32At(x, (t0 + t) * channels + c);
      }

      for (int64_t t = 0; t < tokens; ++t) {
        double acc = 0.0;
        // k = 0..K-1 reads lags {(K-1)*d, ..., 2d, d, 0}: `t + k * dilation`
        // against a window whose current token sits at `t + state_len`, and
        // `(K-1)*dilation == state_len` makes the LAST tap the current token.
        // Causal by that tap, and by nothing else.
        for (int64_t k = 0; k < kernel; ++k) {
          acc += static_cast<double>(LoadF32At(weight, c * kernel + k)) *
                 hist[static_cast<size_t>(t + k * dilation)];
        }
        StoreF32At(out, (t0 + t) * channels + c, static_cast<float>(Silu(acc)));
      }

      // `self.conv_states[state_idx].copy_(full_conv_states[..., -L:])`
      // (cache_utils.py:1068): the last `state_len` columns of [state | chunk],
      // holding the RAW conv input — never the conv output and never the
      // activation. `span - state_len == tokens`, so a chunk shorter than the
      // window keeps the tail of the old state ahead of it, unshifted.
      for (int64_t j = 0; j < state_len; ++j) {
        st[c * state_len + j] = static_cast<float>(hist[static_cast<size_t>(tokens + j)]);
      }
    }
  }
}

// ─── THE GATE — `vt::Qwen4ExpPleGate` (W5e-1, #2336) ─────────────────────────
// `Qwen4ExpTextPLELayer.forward` :1181-1182, flattened at :1184:
//
//     gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
//     gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
//
// THE CLAMP IS BEFORE THE SQRT and the sign is applied AFTER it, so this is not
// `sqrt(clamp(g))` with a sign carried through: at g == 0 the sign is 0 and the
// 1e-3 floor is cancelled, which is why the origin is handled by the SIGN and
// not by a branch. Written as one expression whose order matches upstream's
// method chain left to right, so a reader can diff them by eye.
//
// `value` is read once per (t, d) and broadcast across the hc streams; the
// broadcast is never materialised, which is the whole reason this is one op.
double SignedSqrt(double g, double clamp_min) {
  // NaN IS UPSTREAM'S ANSWER HERE, and the zero arm below would swallow it.
  // `torch.sign(NaN) == 0`, but `NaN * 0.0 == NaN`, so `:1181` propagates a NaN
  // gate and `:1182` a NaN output -- measured by running the pinned expression
  // itself under torch, not inferred from the sign rule. Without this line
  // `NaN < clamp_min` is false, `floored` and `root` are NaN, NEITHER sign
  // branch is taken, and the fall-through returns 0.0 -- which sigmoids to a
  // perfectly plausible `0.5 * value`. That is #2272's polarity, a poison value
  // rendered as a number, inside an op whose comparisons route through
  // `max_abs_diff.h` precisely so poison cannot be absorbed silently. `+/-inf`
  // and `+/-0.0` need NO guard: they already match upstream term for term,
  // which is why this is spelled for NaN alone rather than as a finiteness
  // test.
  if (std::isnan(g)) return g;
  const double magnitude = std::abs(g);
  const double floored = magnitude < clamp_min ? clamp_min : magnitude;
  const double root = std::sqrt(floored);
  // `torch.sign`: -1, 0 or +1. The zero arm is upstream's and it is reachable —
  // a fully masked row scores exactly zero — so it is spelled out rather than
  // left to a `g > 0 ? +root : -root` that would return -1e-3 for it.
  if (g > 0.0) return root;
  if (g < 0.0) return -root;
  return 0.0;
}

double Sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

void Qwen4ExpPleGateKernel(Queue&, Tensor& out, const Tensor& score, const Tensor& value,
                           const Qwen4ExpPleGateArgs& args) {
  const int64_t tokens = score.shape[0];
  const int64_t hc = score.shape[1];
  const int64_t hidden = value.shape[1];
  const double divisor = static_cast<double>(args.gate_divisor);
  const double clamp_min = static_cast<double>(args.clamp_min);
  const float* scores = score.Ptr<float>();

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t j = 0; j < hc; ++j) {
      const double g = static_cast<double>(scores[t * hc + j]) / divisor;
      const double weight = Sigmoid(SignedSqrt(g, clamp_min));
      for (int64_t d = 0; d < hidden; ++d) {
        StoreF32At(out, t * (hc * hidden) + j * hidden + d,
                   static_cast<float>(weight * static_cast<double>(
                                                   LoadF32At(value, t * hidden + d))));
      }
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kQwen4ExpPleConv, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpPleConvFn>(&Qwen4ExpPleConvKernel)));
    RegisterOp(OpId::kQwen4ExpPleGate, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<Qwen4ExpPleGateFn>(&Qwen4ExpPleGateKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
