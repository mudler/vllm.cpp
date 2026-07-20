// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// The declared fusion catalog (TDR). Each `constexpr FusedRecipe` here
// TRANSCRIBES a vLLM fusion PATTERN as a backend-agnostic opcode list + indexed
// operand table — we lift the WHAT (which primitives, in which order, over which
// operands), never a CUDA kernel. See .agents/specs/portable-fusion-framework.md
// §1b (the upstream pass backlog) and §3b (the declaration layer).
//
// Operand-index convention: `operands[i]` names slot i; a step references slots
// by index (in[]/out/out2). The tensors bound at the FusedChain call are ordered
// to match this table. Intermediate operands (a bf16 norm/activation result the
// next step quantizes) are caller-bound scratch slots — exactly as the unfused
// standalone sequence materializes them — so the composite tier is alloc-free and
// byte-exact to that sequence.
#pragma once

#include "vt/fused_recipe.h"
#include "vt/ops.h"  // OpId — for the per-recipe fast-realization binding (fast_op)

namespace vt {

// kFusedAddRmsNorm — residual-add + gemma-RMSNorm, the fused_add_rms_norm chain.
//
// Transcribes vLLM's add+RMSNorm fusion pattern
// (vllm/model_executor/layers/layernorm.py::RMSNorm.forward_* with `residual`):
//   residual = round_to_residual_dtype(x + residual);  out = rms_norm(residual)
// with the weight applied as (1 + w) for the GemmaRMSNorm subclass
// (Qwen3NextRMSNorm). Collapses to csrc/layernorm_kernels.cu::
// fused_add_rms_norm_kernel (f32 variance over the rounded residual), which the
// compilation fusion (vllm/compilation/passes/fusion/) rewrites the subgraph into.
//
// Golden (byte-exact): vt::RmsNorm(out, x, weight, {eps, gemma=true}, residual)
// — the W0-adopted call at src/vllm/model_executor/models/qwen3_5.cpp.
//
// operands: 0=x[T,H], 1=weight[H], 2=residual[T,H] (in/out), 3=out[T,H]
// step0 kAdd  out=2(residual) in=[0(x),2(residual)]     residual = x + residual
// step1 kRmsNorm out=3(out) in=[2(residual),1(weight)]  out = rms_norm(residual)*(1+w)
constexpr FusedRecipe kFusedAddRmsNorm = {
    {
        {FOp::kAdd, /*out=*/2, /*in=*/{0, 2}, /*nin=*/2, kNoOperand, FReduce::kNone, false, false},
        {FOp::kRmsNorm, /*out=*/3, /*in=*/{2, 1}, /*nin=*/2, kNoOperand, FReduce::kMeanSquare,
         /*gemma=*/true, false},
    },
    {
        {FKind::kRow, "x"},
        {FKind::kWeight, "weight"},
        {FKind::kResidual, "residual"},
        {FKind::kRow, "out"},
    },
    /*n=*/2,
    /*n_operands=*/4,
    /*name=*/"fused_add_rms_norm",
};

// kRmsNormQuantFp8 — (add residual) + gemma-RMSNorm -> static per-tensor fp8.
//
// Transcribes vLLM RMSNormQuantFusionPass static-FP8 + residual
// (vllm/compilation/passes/fusion/rms_quant_fusion.py:226 ->
// _C.fused_add_rms_norm_static_fp8_quant; csrc layernorm_quant_kernels).
//
// Golden (byte-exact): vt::RmsNorm(tmp_bf16, x, weight, {eps,gemma}, residual)
// then vt::QuantFp8Static(out_fp8, tmp_bf16, input_scale) — the bf16-intermediate
// form the split pass rounds through (vt::RmsNormQuantFp8 doc, ops.h). The fp8
// terminal is a CUDA-only vt:: op (no CPU kernel) — this recipe's composite runs
// end-to-end on CUDA; on CPU the fp8 terminal is unregistered (backend-negotiated
// quant tail, §3b/§6). Tier: composite-only.
//
// operands: 0=x[T,H], 1=weight[H], 2=residual[T,H], 3=tmp[T,H] bf16, 4=out_fp8[T,H] i8
constexpr FusedRecipe kRmsNormQuantFp8 = {
    {
        {FOp::kAdd, /*out=*/2, /*in=*/{0, 2}, /*nin=*/2, kNoOperand, FReduce::kNone, false, false},
        {FOp::kRmsNorm, /*out=*/3, /*in=*/{2, 1}, /*nin=*/2, kNoOperand, FReduce::kMeanSquare,
         /*gemma=*/true, false},
        {FOp::kQuantFp8, /*out=*/4, /*in=*/{3}, /*nin=*/1, kNoOperand, FReduce::kNone, false,
         false},
    },
    {
        {FKind::kRow, "x"},
        {FKind::kWeight, "weight"},
        {FKind::kResidual, "residual"},
        {FKind::kRow, "tmp_bf16"},
        {FKind::kAux, "out_fp8"},
    },
    /*n=*/3,
    /*n_operands=*/5,
    /*name=*/"rms_norm_quant_fp8",
    /*fast_op=*/static_cast<int>(OpId::kRmsNormQuantFp8),
};

// kRmsNormGatedQuantFp8 — gated-RMSNorm -> static per-tensor fp8 (the GDN
// out_proj producer). Transcribes the gated-RMSNorm epilogue + FP8 quant
// (vt::RmsNormGatedQuantFp8, ops.h; the fla RMSNormGated + static fp8 quant).
//
// Golden (byte-exact): vt::RmsNormGated(tmp_bf16, x, gate, weight,
// {eps, sigmoid_gate}) then vt::QuantFp8Static(out_fp8, tmp_bf16, input_scale).
// Tier: composite-only; fp8 terminal CUDA-only (as above).
//
// operands: 0=x[.,D], 1=gate[.,D], 2=weight[D], 3=tmp[.,D] bf16, 4=out_fp8[.,D] i8
constexpr FusedRecipe kRmsNormGatedQuantFp8 = {
    {
        {FOp::kRmsNormGated, /*out=*/3, /*in=*/{0, 1, 2}, /*nin=*/3, kNoOperand,
         FReduce::kMeanSquare, /*gemma=*/false, /*sigmoid_gate=*/false},
        {FOp::kQuantFp8, /*out=*/4, /*in=*/{3}, /*nin=*/1, kNoOperand, FReduce::kNone, false,
         false},
    },
    {
        {FKind::kRow, "x"},
        {FKind::kRow, "gate"},
        {FKind::kWeight, "weight"},
        {FKind::kRow, "tmp_bf16"},
        {FKind::kAux, "out_fp8"},
    },
    /*n=*/2,
    /*n_operands=*/5,
    /*name=*/"rms_norm_gated_quant_fp8",
    /*fast_op=*/static_cast<int>(OpId::kRmsNormGatedQuantFp8),
};

// kSiluMulFp4Quant — silu(gate)·up -> NVFP4 activation quant (the MoE gate·up
// epilogue). Transcribes vLLM ActivationQuantFusionPass NVFP4
// (vllm/compilation/passes/fusion/act_quant_fusion.py:128 ->
// _C.silu_and_mul_nvfp4_quant).
//
// Golden (byte-exact): vt::MoeSiluMul(tmp_bf16, gate, up) then
// vt::ScaledFp4Quant(out_packed, out_scale, tmp_bf16, input_global_scale_inv) —
// the silu·up value rounded through bf16 before quant (vt::SiluMulFp4Quant doc,
// ops.h). Tier: composite-only (fp4 terminal is a kAux non-row output).
//
// operands: 0=gate[M,I], 1=up[M,I], 2=tmp[M,I] bf16,
//           3=out_packed[M,I/2] i8, 4=out_scale[M,I/16] i8
constexpr FusedRecipe kSiluMulFp4Quant = {
    {
        {FOp::kSiluMul, /*out=*/2, /*in=*/{0, 1}, /*nin=*/2, kNoOperand, FReduce::kNone, false,
         false},
        {FOp::kQuantFp4, /*out=*/3, /*in=*/{2}, /*nin=*/1, /*out2=*/4, FReduce::kNone, false,
         false},
    },
    {
        {FKind::kRow, "gate"},
        {FKind::kRow, "up"},
        {FKind::kRow, "tmp_bf16"},
        {FKind::kAux, "out_packed"},
        {FKind::kAux, "out_scale"},
    },
    /*n=*/2,
    /*n_operands=*/5,
    /*name=*/"silu_mul_fp4_quant",
    /*fast_op=*/static_cast<int>(OpId::kSiluMulFp4Quant),
};

// kSiluMulQuantFp8 — silu(gate)·up -> static per-tensor fp8 activation quant.
//
// W3 MECHANICAL-UPSTREAM-SYNC PROOF (.agents/specs/portable-fusion-framework.md
// §10 W3): a NEW vLLM fusion-pass variant we did NOT previously have a recipe for,
// ported as ONE declaration + its byte-exact test — touching only recipes.h + the
// test (no kernel/dispatch/model-site edits). This is the static-FP8 sibling of
// kSiluMulFp4Quant, transcribing vLLM ActivationQuantFusionPass's ALWAYS-ON
// static-FP8 activation pattern:
//   SiluMulFp8StaticQuantPattern
//   (vllm/compilation/passes/fusion/act_quant_fusion.py:81 -> _C.silu_and_mul_quant;
//    registered unconditionally at act_quant_fusion.py:296; csrc activation_kernels)
// It matches `_C.silu_and_mul` + static-per-tensor-fp8 quant (kFp8StaticTensorSym).
//
// Its Tier-0 composite is expressible ENTIRELY from EXISTING standalone vt:: ops —
// vt::MoeSiluMul (kSiluMul) then vt::QuantFp8Static (kQuantFp8) — so the port needs
// NO new primitive and NO composite-walker case (both opcodes were already added in
// W1). Realization: composite-only (there is no bespoke silu·mul→static-fp8 fused
// OpId in our tree; every existing silu-mul fused op is NVFP4), so fast_op is
// kNoFastOp — the recipe realizes through the byte-exact Tier-0 composite. A fast
// single-launch kernel is a separate later perf step (§10), not part of this port.
//
// Golden (byte-exact): vt::MoeSiluMul(tmp_bf16, gate, up) then
// vt::QuantFp8Static(out_fp8, tmp_bf16, input_scale) — the silu·up value rounded
// through bf16 before the fp8 quant, exactly as the unfused standalone sequence
// materializes it (mirrors the kSiluMulFp4Quant bf16-intermediate discipline, §5).
// The fp8 terminal is a CUDA-only vt:: op (no CPU kernel), so this recipe's
// composite runs end-to-end on CUDA; the portable silu·mul prefix is the same
// MoeSiluMul the kSiluMulFp4Quant CPU test already pins. Tier: composite-only
// (backend-negotiated quant tail, §3b/§6), exactly like kRmsNormQuantFp8.
//
// operands: 0=gate[M,I], 1=up[M,I], 2=tmp[M,I] bf16, 3=out_fp8[M,I] i8
constexpr FusedRecipe kSiluMulQuantFp8 = {
    {
        {FOp::kSiluMul, /*out=*/2, /*in=*/{0, 1}, /*nin=*/2, kNoOperand, FReduce::kNone, false,
         false},
        {FOp::kQuantFp8, /*out=*/3, /*in=*/{2}, /*nin=*/1, kNoOperand, FReduce::kNone, false,
         false},
    },
    {
        {FKind::kRow, "gate"},
        {FKind::kRow, "up"},
        {FKind::kRow, "tmp_bf16"},
        {FKind::kAux, "out_fp8"},
    },
    /*n=*/2,
    /*n_operands=*/4,
    /*name=*/"silu_mul_quant_fp8",
};

// kSigmoidGateFp4Quant — attn·sigmoid(gate) -> NVFP4 activation quant (the
// full-attention output-gate o_proj epilogue). Transcribes vLLM Inductor
// triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view (glue-fusion-2026-07-19.md;
// vt::SigmoidGateFp4Quant doc, ops.h).
//
// Golden (byte-exact): vt::SigmoidGateBf16(tmp_bf16, attn, gate) then
// vt::ScaledFp4Quant(out_packed, out_scale, tmp_bf16, input_global_scale_inv).
// Tier: composite-only.
//
// operands: 0=attn[M,K], 1=gate[M,K], 2=tmp[M,K] bf16,
//           3=out_packed[M,K/2] i8, 4=out_scale[M,K/16] i8
constexpr FusedRecipe kSigmoidGateFp4Quant = {
    {
        {FOp::kSigmoidGate, /*out=*/2, /*in=*/{0, 1}, /*nin=*/2, kNoOperand, FReduce::kNone, false,
         false},
        {FOp::kQuantFp4, /*out=*/3, /*in=*/{2}, /*nin=*/1, /*out2=*/4, FReduce::kNone, false,
         false},
    },
    {
        {FKind::kRow, "attn"},
        {FKind::kRow, "gate"},
        {FKind::kRow, "tmp_bf16"},
        {FKind::kAux, "out_packed"},
        {FKind::kAux, "out_scale"},
    },
    /*n=*/2,
    /*n_operands=*/5,
    /*name=*/"sigmoid_gate_fp4_quant",
    /*fast_op=*/static_cast<int>(OpId::kSigmoidGateFp4Quant),
};

// kAttnQkNormRopeGate — gemma-RMSNorm(q) + gemma-RMSNorm(k) + partial NeoX RoPE
// (from a precomputed cos/sin cache) + gate passthrough, the fused full-attention
// preamble. Transcribes vLLM QKNormRoPEFusionPass
// (vllm/compilation/passes/fusion/qk_norm_rope_fusion.py:188 ->
// _C.fused_qk_norm_rope).
//
// Golden (byte-exact): vt::AttnQkNormRopeGate(q_out, k_out, gate_out, qgate, kf,
// q_norm, k_norm, cos_sin, {eps,gemma}, rope_args) — the single fused-preamble
// standalone op the model hand-calls today (qwen3_5.cpp), itself bit-for-bit
// equal to composing RmsNorm(q)+RmsNorm(k)+RopeFromCache+gate (ops.h). Tier:
// composite-only MACRO — the per-head 3-D operands are outside the generic 2-D
// row interpreter, so the composite dispatches the whole preamble to the one
// standalone op. Operand order is FIXED for this macro (the composite reads it
// positionally):
//   0=qgate[T,Hq*2*Dh], 1=kf[T,Hkv*Dh], 2=q_norm[Dh], 3=k_norm[Dh],
//   4=cos_sin[T,rot], 5=q_out[T,Hq,Dh], 6=k_out[T,Hkv,Dh], 7=gate_out[T,Hq,Dh]
constexpr FusedRecipe kAttnQkNormRopeGate = {
    {
        {FOp::kAttnQkNormRopeGate, /*out=*/5, /*in=*/{0, 1, 4}, /*nin=*/3, kNoOperand,
         FReduce::kMeanSquare, /*gemma=*/true, false},
    },
    {
        {FKind::kAux, "qgate"},
        {FKind::kAux, "kf"},
        {FKind::kWeight, "q_norm"},
        {FKind::kWeight, "k_norm"},
        {FKind::kAux, "cos_sin"},
        {FKind::kAux, "q_out"},
        {FKind::kAux, "k_out"},
        {FKind::kAux, "gate_out"},
    },
    /*n=*/1,
    /*n_operands=*/8,
    /*name=*/"attn_qk_norm_rope_gate",
};

}  // namespace vt
