// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// Tiered Declarative Recipe (TDR) — Phase 0 skeleton. See
// .agents/specs/fusion-architecture-2026-07-08.md.
//
// A fusion is declared ONCE, backend-agnostic, as a constexpr `FusedRecipe`
// POD (an opcode list + operand bindings) that lives ABOVE vt::. It is the
// single source of truth realized per backend in tiers:
//   Tier 0 (composite, default): walk the steps through the already-registered
//     vt:: primitives — correct on every backend for free, and the CPU golden.
//   Tier 1 (interpreter): one kernel per backend that walks the recipe in a
//     single pass over each row.
// This header is the declaration surface (the WHAT); the realizations (the HOW)
// live in the backend op tables (src/vt/{cpu,cuda}). No heap, no vt:: deps —
// pure POD so a recipe is a `constexpr` object usable from model-forward code.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace vt {

// Primitive opcodes a recipe step can carry. Phase 0 needs exactly kAdd (the
// residual add) and kRmsNorm (the gemma row-normalize); kMul is included as the
// obvious next elementwise primitive so the enum has room to grow without a POD
// layout change. New primitives (activations, quant terminals) append here and
// cost +1 switch case per backend realization — O(1), not O(fusions).
enum class FOp : uint8_t {
  kAdd,      // out = a + b  (elementwise; residual-add when out == b == kResidual)
  kMul,      // out = a * b  (elementwise; room to grow)
  kRmsNorm,  // out = a * rsqrt(reduce(a) + eps) * (gemma ? 1+b : b)  (row reduce)
};

// Row-reduction kind for reducing ops (kRmsNorm). kMeanSquare = mean(a^2) over
// the row, the RMSNorm variance (f32 accumulation). kNone marks elementwise
// steps that carry no reduction.
enum class FReduce : uint8_t {
  kNone,
  kMeanSquare,
};

// Operand slots a step binds to. These are ROLES, resolved to physical tensors
// by each backend realization (kIn->x, kResidual->residual, kWeight->weight,
// kOut->out). A role has a fixed dtype per call (kIn/kWeight follow x; kResidual
// is f32/bf16; kOut is f32/bf16), so a realization reads/writes it type-safely.
enum class FOperand : uint8_t {
  kIn,        // the input activation x [T,H]
  kResidual,  // the residual stream [T,H] (in/out; may be null for non-residual recipes)
  kWeight,    // the norm weight [H]
  kOut,       // the output [T,H]
};

// One recipe step: an opcode + its operand bindings + the small structural
// constants the opcode needs. `gemma` and `reduce` are structural (baked into
// the recipe); runtime scalars (eps) travel in the op call, not the recipe.
struct FStep {
  FOp op = FOp::kAdd;
  FOperand out = FOperand::kOut;  // where the result is written
  FOperand a = FOperand::kIn;     // first input
  FOperand b = FOperand::kIn;     // second input (weight for kRmsNorm)
  FReduce reduce = FReduce::kNone;
  bool gemma = false;  // kRmsNorm: apply the weight as (1 + w), GemmaRMSNorm style
};

// Max steps a Phase-0 recipe holds. fused_add_rms_norm uses 2; the slack is
// headroom for the finite Class-A list (Phase 1) without a layout change.
constexpr int kMaxFusedSteps = 8;

// The declaration: a fixed-size step list + a live count + a debug name. POD,
// constexpr-constructible, no heap — a recipe is a compile-time constant that
// every realization tier reads.
struct FusedRecipe {
  FStep steps[kMaxFusedSteps] = {};
  int n = 0;
  const char* name = nullptr;
};

// Runtime tier selector (Phase 0). Default 0 = Tier-0 composite (the safe
// default: reuses the already-registered primitives, so behavior is identical
// to the unfused path). VT_FUSED_TIER=1 selects the Tier-1 single-pass
// interpreter. Read fresh each call (getenv is cheap; keeps the flag togglable
// within one process, which the parity test relies on).
inline int FusedTier() {
  const char* e = std::getenv("VT_FUSED_TIER");
  return (e != nullptr && e[0] == '1') ? 1 : 0;
}

}  // namespace vt
