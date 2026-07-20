// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// Tiered Declarative Recipe (TDR) — the portable fusion framework's declaration
// surface. See .agents/specs/portable-fusion-framework.md (§2c/§3b/§3c/§5/§10).
//
// A fusion is declared ONCE, backend-agnostic, as a constexpr `FusedRecipe` POD:
// an OPCODE LIST plus an INDEXED OPERAND TABLE. It lives ABOVE vt:: and is the
// single source of truth realized per backend in tiers:
//   Tier 0 (composite, default): walk the steps, DISPATCHING each opcode to the
//     already-registered standalone vt:: op (kRmsNorm->vt::RmsNorm,
//     kSiluMul->vt::MoeSiluMul, kQuantFp4->vt::ScaledFp4Quant, ...). A recipe's
//     Tier-0 composite is therefore BYTE-EXACT by construction to the exact
//     standalone-op sequence the model hand-calls today, and is the CPU golden.
//   Tier 1 (interpreter): one single-pass kernel per backend for the elementwise
//     + reduce subset {kAdd,kMul,kSilu,kSigmoid,kRmsNorm}; the perf tier.
// This header is the declaration (the WHAT); realizations (the HOW) live in the
// backend op tables (src/vt/{cpu,cuda}) and the device-agnostic composite walker
// (src/vt/ops.cpp). No heap, no vt:: deps — a recipe is a `constexpr` object
// usable from model-forward code.
//
// W1 (KERNEL-FUSION-FRAMEWORK) generalized the Phase-0 skeleton: opcodes grew
// from {kAdd,kMul,kRmsNorm} to the full activation/norm/quant/rope vocabulary,
// and the fixed 4-role operand model (kIn/kResidual/kWeight/kOut) became a small
// INDEXED operand table so multi-input chains (silu·mul, sigmoid-gate,
// qk-norm-rope-gate) are expressible. Infrastructure only — perf-neutral.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace vt {

// Primitive opcodes a recipe step can carry. Two families:
//
//  * GRANULAR elementwise/reduce (Tier-1 interpreter vocabulary; no standalone
//    vt:: op of their own): kAdd, kMul, kSilu, kSigmoid, kRmsNorm. kAdd in a
//    residual-add position (out==residual) folds into the following norm's
//    RmsNorm(residual) call in the composite tier.
//  * FUSED-PRIMITIVE opcodes that map 1:1 to a standalone vt:: op, so the
//    composite realization is byte-exact by construction (the fused activation
//    goldens round through a single bf16 store that a granular kSilu+kMul split
//    would not reproduce — see §5). These are composite-realized:
//      kSiluMul     -> vt::MoeSiluMul       (silu(a)·b)
//      kSigmoidGate -> vt::SigmoidGateBf16  (a·sigmoid(b))
//      kRmsNormGated-> vt::RmsNormGated     (gated rms-normalize)
//      kRope        -> vt::RopeFromCache    (partial NeoX RoPE from a cos/sin cache)
//      kQuantFp8    -> vt::QuantFp8Static   (static per-tensor fp8 terminal; CUDA-only)
//      kQuantFp4    -> vt::ScaledFp4Quant   (dynamic per-group fp4 terminal)
//      kAttnQkNormRopeGate -> vt::AttnQkNormRopeGate (fused full-attention preamble)
//
// New primitives append here and cost +1 composite switch case per backend —
// O(1), not O(fusions). The quant/rope/attn opcodes are composite-only (Tier-1
// leaves them to the composite); the elementwise/kRmsNorm subset is Tier-1-able.
enum class FOp : uint8_t {
  kAdd,       // out = a + b  (residual-add when out==residual folds into next norm)
  kMul,       // out = a * b
  kSilu,      // out = silu(a) = a * sigmoid(a)                       [Tier-1 only]
  kSigmoid,   // out = sigmoid(a)                                     [Tier-1 only]
  kRmsNorm,   // out = a * rsqrt(mean(a^2)+eps) * (gemma ? 1+w : w)   [Tier-1 + composite]
  kSiluMul,   // out = silu(a) * b            -> vt::MoeSiluMul       [composite]
  kSigmoidGate,   // out = a * sigmoid(b)     -> vt::SigmoidGateBf16  [composite]
  kRmsNormGated,  // gated rms-normalize      -> vt::RmsNormGated     [composite]
  kRope,          // partial NeoX RoPE        -> vt::RopeFromCache    [composite]
  kQuantFp8,      // static per-tensor fp8    -> vt::QuantFp8Static   [composite terminal]
  kQuantFp4,      // dynamic per-group fp4    -> vt::ScaledFp4Quant   [composite terminal]
  kAttnQkNormRopeGate,  // fused attn preamble -> vt::AttnQkNormRopeGate [composite macro]
};

// Row-reduction kind for reducing ops (kRmsNorm). kMeanSquare = mean(a^2) over
// the row, the RMSNorm variance (f32 accumulation). kNone marks non-reducing steps.
enum class FReduce : uint8_t {
  kNone,
  kMeanSquare,
};

// Logical kind of an operand SLOT — the shape/role a realization binds/indexes.
// The physical tensor (with its own dtype) is bound positionally at the call.
//   kRow      [.,W]  row-major activation (x, up, gate, attn, out, intermediates)
//   kWeight   [W]    per-column norm weight (indexed by column j)
//   kResidual [.,W]  residual stream, read+written in place
//   kAux             a backend-negotiated / non-row operand the generic Tier-1
//                    interpreter does NOT model: fp4 scale-out, fp8/fp4 packed
//                    output, rope cos/sin cache, positions, per-head 3-D q/k/gate.
//                    Only the composite tier (which hands it to a standalone op)
//                    touches a kAux operand.
enum class FKind : uint8_t {
  kUnused = 0,
  kRow,
  kWeight,
  kResidual,
  kAux,
};

// Sentinel for FStep::out2 (a step with no secondary output).
constexpr uint8_t kNoOperand = 0xFF;

// Max structural sizes. A recipe holds a fixed step list + operand table (POD,
// no heap). The slack is headroom for the finite Class-A pattern set.
constexpr int kMaxFusedSteps = 8;
constexpr int kMaxFusedOperands = 8;
constexpr int kMaxStepIns = 3;

// One operand slot in the recipe's INDEXED operand table. Steps reference
// operands by their index into FusedRecipe::operands (== the index into the
// tensors bound at the call).
struct FOperandSlot {
  FKind kind = FKind::kUnused;
  const char* name = nullptr;  // debug only
};

// One recipe step: an opcode + the operand INDICES it reads/writes + the small
// structural constants the opcode needs. `out` is the primary output operand
// index; `out2` a secondary output (fp4 scale stream) or kNoOperand; `in[nin]`
// the input operand indices. `gemma`/`reduce`/`sigmoid_gate` are structural;
// runtime scalars (eps, quant scale, rope args) travel in the op call.
struct FStep {
  FOp op = FOp::kAdd;
  uint8_t out = 0;                     // primary output operand index
  uint8_t in[kMaxStepIns] = {0, 0, 0};  // input operand indices
  uint8_t nin = 0;                     // number of inputs used
  uint8_t out2 = kNoOperand;           // secondary output (e.g. fp4 scale) or kNoOperand
  FReduce reduce = FReduce::kNone;
  bool gemma = false;         // kRmsNorm/kAttn: weight as (1 + w), GemmaRMSNorm style
  bool sigmoid_gate = false;  // kRmsNormGated: sigmoid gate activation (else silu)
};

// The declaration: a fixed-size step list + an indexed operand table + live
// counts + a debug name. POD, constexpr-constructible, no heap.
struct FusedRecipe {
  FStep steps[kMaxFusedSteps] = {};
  FOperandSlot operands[kMaxFusedOperands] = {};
  int n = 0;           // number of steps
  int n_operands = 0;  // number of bound operand slots
  const char* name = nullptr;
};

// Runtime tier selector. Default 0 = Tier-0 composite (the safe default: reuses
// the already-registered standalone ops, so behavior is identical to the unfused
// path). VT_FUSED_TIER=1 selects the Tier-1 single-pass interpreter WHERE the
// recipe is Tier-1-able (all steps in {kAdd,kMul,kSilu,kSigmoid,kRmsNorm});
// recipes with composite-only opcodes (quant/rope/gated/attn) always run via the
// composite regardless. Read fresh each call (getenv is cheap; keeps the flag
// togglable within one process, which the parity test relies on).
inline int FusedTier() {
  const char* e = std::getenv("VT_FUSED_TIER");
  return (e != nullptr && e[0] == '1') ? 1 : 0;
}

// True iff every step of `r` is in the Tier-1 interpreter vocabulary (so
// VT_FUSED_TIER=1 may dispatch it to the backend interpreter kernel; otherwise
// the composite tier realizes it). Composite-only opcodes force the composite.
inline bool RecipeIsTier1Able(const FusedRecipe& r) {
  for (int s = 0; s < r.n; ++s) {
    switch (r.steps[s].op) {
      case FOp::kAdd:
      case FOp::kMul:
      case FOp::kSilu:
      case FOp::kSigmoid:
      case FOp::kRmsNorm:
        break;
      default:
        return false;
    }
  }
  return true;
}

}  // namespace vt
