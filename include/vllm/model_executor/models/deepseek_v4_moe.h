// DeepSeek-V4-Flash W6 — the sqrtsoftplus + hash-routed MoE, as portable host
// (CPU) reference implementations. This is the MoE brick: V4 keeps the DeepSeek
// grouped-GEMM / shared-expert machinery we already have, but replaces THREE
// pieces with genuinely-new primitives (distinct from V2/V3's sigmoid/softmax
// `noaux_tc` router + plain SwiGLU). W6 owns exactly those three, each ported 1:1
// with `file:line` on BOTH sides (vLLM primary, SGLang v0.5.15 cross-reference):
//
//   (1) SqrtSoftplus         — the V4 router SCORE function `sqrt(softplus(x))`,
//                              distinct from V2/V3's sigmoid/softmax. `softplus`
//                              then `sqrt`, composed; the composition is the
//                              load-bearing nuance (softplus alone gives a
//                              different score, sqrt alone is undefined on the
//                              raw logits).
//   (2) SqrtSoftplusRouteTopk — the full router: score every expert with (1),
//                              add `e_score_correction_bias` for SELECTION ONLY
//                              (`noaux_tc`), pick top-k (OR, for the HASH-routed
//                              first `num_hash_layers` layers, look the experts up
//                              directly in the `tid2eid` token-id→expert table),
//                              GATHER the weights from the UNBIASED scores,
//                              optionally renormalize, and scale by
//                              `routed_scaling_factor`. The bias-affects-selection-
//                              but-NOT-weights split + the hash bypass are the two
//                              load-bearing nuances.
//   (3) ClampedSwiGLU        — the V4 expert / shared-expert MLP activation
//                              `SiluAndMulWithClamp`: `gate` clamped to `max=limit`
//                              (max ONLY), `up` clamped to `[-limit, +limit]` (both
//                              sides), then `gate·sigmoid(alpha·gate)·(up+beta)`.
//                              The ASYMMETRIC clamp (gate max-only vs up both-sided)
//                              is the load-bearing nuance.
//
// REUSE, not re-port: the DeepSeek grouped-GEMM expert forward, the 256-expert
// w13/w2 layout, the shared-expert block, and the NVFP4/FP8 expert GEMMs are the
// EXISTING DeepSeek-V2 MoE + NVFP4 machinery (deepseek_v2.cpp + the
// cuda_matmul_nvfp4 fast path); W6 does NOT re-port them. Only the scoring func,
// the hash route, and the clamp are net-new, so only those three land here. The
// MegaMoE DeepGEMM fast path is SM100-only (major==10) and is NOT the GB10 target
// (nvidia/model.py:307-315) — W6 references the FusedMoE-fallback router that GB10
// actually runs (nvidia/model.py:647-691, _init_fused_moe_experts).
//
// WHY host/CPU reference (honest scope, mirrors W3/W4/W5): the full V4 forward is
// a multi-Spark campaign — the checkpoint is 156.7 GiB (does not fit one GB10, see
// deepseek_v4.h) and the forward still needs the device kernels + assembly (W7),
// none landed. So W6 lands + UNIT-GATES these primitives against hand-derived
// small cases with literal expected numbers verified from the vLLM source AND
// from-first-principles double-precision references on randomized shapes
// (tests/vllm/models/test_deepseek_v4_moe.cpp), rather than a full-model
// dumped-oracle rel-L2 (the fixed-config 167B arch cannot be built at a tiny
// shape). The eventual GPU forward (W7) reuses the existing grouped-GEMM kernels;
// this file pins the NEW router + clamp numerics portably so the port has an
// oracle.
//
// SACRED-inert: additive TU only. It does NOT touch any existing forward — in
// particular the shared DeepSeek-V2 MoE router / grouped GEMM / shared experts
// stay untouched (V4's scoring + hash + clamp land as a V4-specific path; the
// shared-MoE extraction is a NAMED W7 seam). The device kernels + folding these
// into DeepseekV4Model::Forward are NAMED W7 residuals.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ───────
//   OURS                     <-  UPSTREAM (vllm/, @ 0.26.0.dev0)  [+ SGLang v0.5.15]
//   SqrtSoftplus             <-  fused_moe/router/fused_topk_bias_router.py:88
//                                (`torch.sqrt(F.softplus(gating_output.float()))`);
//                                SGLang python/sglang/srt/layers/moe/topk.py:1013-1014
//                                (`F.softplus(gating_output).sqrt()`), :652-653
//   SqrtSoftplusRouteTopk    <-  fused_topk_bias_router.py:75-118
//                                (`_topk_softplus_sqrt_torch`, the pure-PyTorch
//                                fallback = the eager reference the CUDA kernel
//                                matches) + the hash branch :100-106; the
//                                scoring_func="sqrtsoftplus" dispatch
//                                :254-265,:296-320; hash table wiring
//                                nvidia/model.py:562-578,:686,:696-717; SGLang
//                                hash route python/sglang/srt/layers/moe/hash_topk.py
//                                :137-180 (`_forward_torch`, tid2eid gather from
//                                UNBIASED scores + renormalize)
//   ClampedSwiGLU            <-  model_executor/layers/activation.py:162-208
//                                (`SiluAndMulWithClamp.forward_native`: gate max-clamp,
//                                up ±clamp, `gate·sigmoid(alpha·gate)·(up+beta)`);
//                                used by DeepseekV4MLP nvidia/model.py:126-133 with
//                                swiglu_limit (alpha=1, beta=0 defaults)
//   Upstream tests           <-  tests/kernels/moe/test_topk_softplus_sqrt.py
//                                (`_torch_topk_softplus_sqrt`, the topk + hash + bias
//                                cases) + tests/kernels/core/test_activation.py
//                                (SiluAndMulWithClamp)
#pragma once

#include <cstdint>
#include <vector>

namespace vllm::deepseek_v4 {

// ── (1) sqrtsoftplus score function ───────────────────────────────────────────
//
// The V4 router score of a single gating logit (fused_topk_bias_router.py:88):
//
//   score = sqrt(softplus(x)),   softplus(x) = log(1 + exp(x))
//
// Distinct from V2/V3's sigmoid/softmax. Computed in f32 upstream
// (`gating_output.float()`); this reference matches that. `softplus` uses the
// numerically-stable identity `max(x,0) + log1p(exp(-|x|))`, equal to
// `log(1+exp(x))` for all x (torch.nn.functional.softplus caps at a threshold=20
// linear tail, which the stable form already agrees with to < 1e-8). The result
// is >= 0, so the outer sqrt is always defined.
float SqrtSoftplus(float x);

// ── (2) the sqrtsoftplus / hash router ────────────────────────────────────────
//
// The full DeepSeek-V4 MoE router (fused_topk_bias_router.py:75-118, the eager
// `_topk_softplus_sqrt_torch` reference the CUDA `topk_hash_softplus_sqrt` kernel
// matches). For each of the M tokens over E experts:
//
//   scores[e]           = SqrtSoftplus(gating[e])                    (UNBIASED)
//   scores_for_choice[e]= scores[e] + e_score_correction_bias[e]     (SELECTION ONLY)
//   if hash-routed (hash_indices_table + input_tokens given):
//     topk_ids[j]       = hash_indices_table[token, j]               (direct lookup,
//                                                                      NO top-k)
//   else:
//     topk_ids          = argtop-k(scores_for_choice, topk)          (descending
//                                                                      score; ties →
//                                                                      smaller expert
//                                                                      index)
//   topk_weights[j]     = scores[topk_ids[j]]        (GATHER from the UNBIASED scores
//                                                     — NOT scores_for_choice; using
//                                                     biased scores as weights would
//                                                     flatten the distribution, see
//                                                     fused_topk_bias_router.py:90-96)
//   if renormalize:  topk_weights /= max(Σ_j topk_weights[j], 1e-20)
//   topk_weights      *= routed_scaling_factor
//
// `e_score_correction_bias` empty ⇒ no bias (scores_for_choice == scores).
// `hash_indices_table` (row-major [vocab, topk]) + `input_tokens` ([M]) BOTH given
// ⇒ hash route; both empty ⇒ learned top-k route. The hash route ignores the bias
// entirely (matches nvidia/model.py:562-567: a hash layer carries no
// e_score_correction_bias). NOTE (W7 seam): the device kernel supports an
// is_padding guard that zeroes padded rows; this host reference computes every
// real row (padding is a device-batching concern, mirrored not folded here, as
// W3/W4/W5 left their device seams).
struct MoeRouteResult {
  std::vector<int32_t> topk_ids;      // [M*topk] row-major
  std::vector<float> topk_weights;    // [M*topk] row-major
};
//
// ── MODEL-MM-deepseek-v4 W4 (#2411): THE VISION ROUTING BIAS, PER TOKEN ──────
//
// A DeepSeek-V4-Flash-Vision checkpoint carries a SECOND router bias,
// `exp_probs_b_vl` (`layers.N.ffn.gate.bias_vl` in safetensors), on every one of
// its 43 language layers, hash layers included. It is the bias the router adds
// when the token being routed is an IMAGE token, in place of the text bias --
// and on a hash layer it replaces the `tid2eid` routing itself, because an image
// token has no meaningful identifier to hash. A text checkpoint carries none of
// these tensors and `vision_bias` is then empty, which is byte-identical.
//
// PER TOKEN, and the divergence from the oracle is deliberate. At
// `llama-cpp-dsv4vision` the selection is PER UBATCH -- `const bool is_media =
// ubatch.embd != nullptr;` -- and when it is set every layer takes
// `ffn_exp_probs_b_vl` and the hash branch is skipped WHOLESALE. That is
// indistinguishable from the per-token rule on every input llama.cpp can build,
// because a media ubatch carries no text rows. It is NOT indistinguishable here:
// this engine batches continuously, one step mixes an image request's prefill
// rows with other requests' decode rows, and applying a whole-step flag would
// route another request's TEXT tokens on the vision bias, which the oracle never
// does. So the per-token rule reduces to the oracle's on the oracle's own
// inputs and is defined on the inputs the oracle cannot express.
//
// `is_media_token` is `[num_tokens]`, non-zero for an image row. Empty means no
// row is one, which is every text step.
MoeRouteResult SqrtSoftplusRouteTopk(const std::vector<float>& gating, int64_t num_tokens,
                                     int64_t num_experts, int64_t topk,
                                     const std::vector<float>& e_score_correction_bias,
                                     bool renormalize, float routed_scaling_factor,
                                     const std::vector<int64_t>& input_tokens,
                                     const std::vector<int32_t>& hash_indices_table,
                                     int64_t vocab_size,
                                     const std::vector<float>& vision_bias = {},
                                     const std::vector<char>& is_media_token = {});

// ── (3) clamped SwiGLU expert activation ──────────────────────────────────────
//
// The V4 expert / shared-expert MLP activation `SiluAndMulWithClamp`
// (activation.py:197-201). Per token, `gate_up` is [2*d] with gate = first d,
// up = last d:
//
//   gate = clamp(gate_up[:d],  max=limit)          (MAX only — no lower clamp)
//   up   = clamp(gate_up[d:], -limit, +limit)      (BOTH sides)
//   out[i] = gate[i] · sigmoid(alpha·gate[i]) · (up[i] + beta)      ([d])
//
// The ASYMMETRY (gate clamped max-only, up clamped both-sided) is load-bearing.
// DeepseekV4MLP passes swiglu_limit with the defaults alpha=1, beta=0
// (nvidia/model.py:131), which reduces the activation to
// `silu(clamp(gate,max=limit)) · clamp(up,-limit,limit)`. alpha/beta are
// parameters here for gateability (SwiGLU-OAI style uses alpha!=1, beta=1).
//
//   gate_up : [2*d] row-major, gate = [0,d), up = [d,2d)
// Returns [d].
std::vector<float> ClampedSwiGLU(const std::vector<float>& gate_up, int64_t d,
                                 float limit, float alpha, float beta);

}  // namespace vllm::deepseek_v4
