// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// The declared fusion catalog (TDR Phase 0). Each `constexpr FusedRecipe` here
// TRANSCRIBES a vLLM fusion PATTERN as a backend-agnostic opcode list — we lift
// the WHAT (which primitives, in which order, over which operands), never a CUDA
// kernel. See .agents/fusion-architecture-2026-07-08.md §1.
#pragma once

#include "vt/fused_recipe.h"

namespace vt {

// kFusedAddRmsNorm — residual-add + gemma-RMSNorm, the fused_add_rms_norm chain.
//
// Transcribes vLLM's add+RMSNorm fusion pattern:
//   * semantics: vllm/model_executor/layers/layernorm.py::RMSNorm.forward_*
//     with the `residual` argument — `residual = x + residual` (rounded to the
//     model dtype), then `out = rms_norm(residual)` with the weight applied as
//     (1 + w) for the GemmaRMSNorm subclass (Qwen3NextRMSNorm).
//   * fused kernel it collapses to: csrc/layernorm_kernels.cu::
//     fused_add_rms_norm_kernel (f32 variance accumulation over the rounded
//     residual), which vLLM's compilation fusion (vllm/compilation/passes/
//     fusion/) rewrites the add+RMSNorm subgraph into.
//
// The vt:: golden this must equal bit-for-bit is vt::RmsNorm(out, x, weight,
// {eps, gemma=true}, residual) — the call at
// src/vllm/model_executor/models/qwen3_5.cpp:2322.
//
// Step 0: kAdd  out=kResidual  a=kIn  b=kResidual
//   residual[j] = round_to_residual_dtype(x[j] + residual[j])   (new residual
//   stream; the rounded value is what the next step reads — matches the fused
//   kernel, whose f32 variance squares the rounded residual, not raw x+res).
// Step 1: kRmsNorm  out=kOut  a=kResidual  b=kWeight  reduce=kMeanSquare  gemma
//   var = mean(residual[j]^2);  out[j] = residual[j] * rsqrt(var + eps) * (1+w[j]).
constexpr FusedRecipe kFusedAddRmsNorm = {
    {
        {FOp::kAdd, FOperand::kResidual, FOperand::kIn, FOperand::kResidual, FReduce::kNone,
         false},
        {FOp::kRmsNorm, FOperand::kOut, FOperand::kResidual, FOperand::kWeight,
         FReduce::kMeanSquare, true},
    },
    /*n=*/2,
    /*name=*/"fused_add_rms_norm",
};

}  // namespace vt
