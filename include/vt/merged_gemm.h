// vt::MergedGemmGroup — the DECLARATIVE descriptor for a MERGED-GEMM GROUP: N
// contraction-tier GEMMs that SHARE operand A (the activation), each with its own
// weight operand, collapsed with a fused elementwise EPILOGUE into ONE device
// launch. It is the contraction-tier analog of vt::FusedRecipe (fused_recipe.h),
// which is contraction-FREE by design (FusedChain owns only the norm/quant/act/rope
// GLUE and SANDWICHES a first-class GEMM op via `fast_op`). A MergedGemmGroup is the
// mirror image: the GEMMs ARE the payload, and the epilogue is the fused tail.
//
// GROUNDING — the best-gemm-path "[FusedChain prologue -> fast_op = <GEMM OpId> ->
// FusedChain epilogue]" shape (.agents/specs/best-gemm-path-2026-07-30.md §FusedChain
// verdict). MergedGemmGroup names the middle box when the middle is itself several
// merged GEMMs plus a byte-exact epilogue — e.g. the routed-MoE gate+up GEMMs sharing
// the broadcast activation, then silu(gate)·up. The realized single-launch kernel is
// the promoted shared op OpId::kMoeGateUpSwiGLUGrouped (cuda_quant_dot.cu); the
// byte-exact Tier-0 composite (N× vt::MatmulBTQuantGrouped + the elementwise epilogue)
// is the CPU golden, exactly mirroring fused_recipe.h's Tier-0/fast_op tiering.
//
// SELECTION is by (weight dtype, merge-arity, epilogue): a small STRUCTURAL-KERNEL
// library keyed on those three fields picks the fast launch; a backend that has not
// registered the fast op inherits the Tier-0 composite automatically (the same
// additivity property the FusedChain catalog has). Adding a new merged-GEMM shape is
// ONE constexpr descriptor + (optionally) one structural kernel — no model-site edits.
#pragma once

#include <cstdint>

#include "vt/device.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vt {

// The fused elementwise tail applied to the N merged GEMM outputs (per output row).
enum class MergedEpilogue : uint8_t {
  // No epilogue — the N GEMM outputs are written as-is (a plain merged/batched GEMM).
  kNone,
  // arity-2 clamped SwiGLU: out = min(g,limit)·sigmoid(min(g,limit))·clamp(u,±limit).
  // limit=+inf reduces to the standard silu(g)·u SwiGLU MLP. Matches
  // QuantDotGemmGroupedFusedSwiGLUKernel (cuda_quant_dot.cu) / ClampedSwiGLUKernel.
  kSiluMulClamp,
};

// A merged-GEMM group. constexpr-POD like FusedRecipe: pure description, no state.
struct MergedGemmGroup {
  // Number of GEMMs sharing operand A (weight operands). 2 for gate+up.
  int arity;
  // The fused epilogue opcode applied to the `arity` GEMM outputs.
  MergedEpilogue epilogue;
  // The single-launch fast realization (an OpId cast to int), or kNoFastOp when the
  // group has no fused kernel yet and must run the Tier-0 composite everywhere.
  int fast_op;
  // Human-readable tag (records / diagnostics).
  const char* name;
};

// Sentinel: a group with no fused kernel — runs the Tier-0 composite on every backend.
inline constexpr int kNoMergedFastOp = -1;

// ── Realized instance ────────────────────────────────────────────────────────
// The routed-MoE gate+up keep-quant GEMMs sharing the broadcast Q8_K activation,
// with the clamped-SwiGLU epilogue — realized end-to-end by OpId::kMoeGateUpSwiGLUGrouped
// (CUDA fused kernel) with the CPU composite golden. This is the DeepSeek-V4 Brick-6
// fusion, now expressed declaratively so any keep-quant MoE arch reuses it.
inline constexpr MergedGemmGroup kKeepQuantGateUpSwiGLU = {
    /*arity=*/2,
    /*epilogue=*/MergedEpilogue::kSiluMulClamp,
    /*fast_op=*/static_cast<int>(OpId::kMoeGateUpSwiGLUGrouped),
    /*name=*/"keepquant_gate_up_swiglu",
};

// The BF16-native SIBLING arm of this gate+up+SwiGLU family (Tier-A4 fold) is NOT a
// MergedGemmGroup instance: the bf16 grouped-MoE archs (Qwen3-Coder, DeepSeek-V2,
// kimi) marshal their experts as per-expert weight-POINTER ARRAYS [E] i64 with a
// pair->token row_map (not the contiguous [E*N,K] + broadcast-activation convention
// MergedGemm's Tier-0 composite consumes), so it is realized directly as the shared
// op vt::MoeGroupedGemmBf16GateUpSilu / OpId::kMoeGroupedGemmBf16GateUpSilu — the
// bf16 twin of kMoeGateUpSwiGLUGrouped, BIT-IDENTICAL to {2x MoeGroupedGemmBf16 +
// MoeSiluMul}. Same family, distinct weight-marshaling seam.
//
// NON-GATED experts are NOT in this family at all, and deliberately get no
// descriptor. NemotronH's expert (models/nemotron_h.py:126-256 @ 555967922) has
// NO gate half — `ckpt_names=("up_proj", "down_proj", "")` (:220), the empty
// third entry being the absent gate — so the expert is
//     h = up_proj(x); h = relu(h)^2; y = down_proj(h)
// with `activation_without_mul(config.mlp_hidden_act)` (:227). A MergedGemmGroup
// describes N GEMMs SHARING operand A collapsed into one launch; with N == 1
// there is nothing to merge and no launch to save, so an arity-1 descriptor would
// name a fusion that does not exist. The non-gated arm is therefore realized as
// the EXISTING single grouped GEMM plus the activation — kMoeGroupedGemmBf16 (or
// kMoeGroupedGemmNvfp4Marlin for the W4A16 group-16 arm) followed by
// OpId::kMoeRelu2 — which is exactly the shape the gated bf16 archs had before
// their pair was folded. See vt::MoeRelu2 (ops.h) and
// .agents/specs/nemotron-h-model.md §4 W2.

// ── Dispatch ─────────────────────────────────────────────────────────────────
// Run a merged-GEMM group. For an arity-2 kSiluMulClamp group over keep-quant
// towers: out[P,N] f32, act[Pa,K] (Pa==1 broadcast), gate_w/up_w[E*N,K] SAME
// block-quant dtype, expert_ids[P] i32, epilogue_scalar = the SwiGLU clamp `limit`.
//
// Realization, mirroring FusedRecipe tiering:
//   * if desc.fast_op names an OpId registered for this device -> ONE fused launch
//     (the promoted shared op) — the fast structural kernel selected by dtype/arity/epilogue;
//   * else -> the Tier-0 COMPOSITE: `arity` vt::MatmulBTQuantGrouped into f32 temps
//     + the elementwise epilogue, BYTE-EXACT to the standalone-op sequence.
// The `force_composite` knob runs the Tier-0 path even where the fast op exists — the
// A/B oracle the unit test uses to prove composite == fused byte-for-byte.
void MergedGemm(Queue& q, const MergedGemmGroup& desc, Tensor& out, const Tensor& act,
                const Tensor& gate_w, const Tensor& up_w, const Tensor& expert_ids,
                float epilogue_scalar, bool force_composite = false);

}  // namespace vt
