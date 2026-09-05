// Qwen4-Exp (`Qwen3.8-Flash-Next`) W3 — the 4-branch GATED-RESIDUAL
// hyper-connection stream and the grouped RMSNorm it stands on. HOST reference.
// Issue #1988, spec `.agents/specs/qwen4-exp-flash-next.md` ("Gated Residual:
// what our MHC actually gives us").
//
// ─── WHAT THIS IS A PORT OF ───────────────────────────────────────────────────
// This row splits its oracles by design (spec `## Oracles`, developer direction
// 2026-08-26): transformers supplies the ALGORITHM, vLLM supplies the OP FORM.
// SUPERSEDED 2026-08-31: vLLM DOES register `qwen4_exp`, at `e126687a9a`, which
// is 595 commits BEYOND our pin `555967922` and not reachable from it. It is now
// this row's primary oracle; see the spec's reconciliation (#2489).
//
//   OURS                       <-  ALGORITHM (transformers v5.16.0, the lane pin,
//                                  `models/qwen4_exp/modeling_qwen4_exp.py`)
//   GroupedRmsNorm             <-  `Qwen4ExpTextRMSNorm._norm` (:167-171)
//   HcNormWeightFromHf         <-  `Qwen4ExpTextRMSNorm.forward` (:173-178),
//                                  `output * (1.0 + self.weight.float())`
//   GatedResidualForward       <-  `Qwen4ExpTextGatedResidual.forward` (:952-969)
//   GatedResidualWriteBack     <-  `Qwen4ExpTextDecoderLayer.forward`, the two
//                                  lines `injection = hidden_states.unsqueeze(-2)
//                                  * injection_weights.unsqueeze(-1)` /
//                                  `hidden_states = hyper_input +
//                                  injection.flatten(-2)`
//
//   OURS                       <-  OP FORM (vLLM `origin/main` 6a5e8f5979,
//                                  `model_executor/layers/layernorm.py`)
//   GroupedRmsNorm             <-  `RMSNormGated` (:172), `group_size` (:187),
//                                  grouped branch `forward_static` (:258-264):
//                                  x.float() (:243), weight.float() (:244),
//                                  rearrange "... (g d) -> ... g d",
//                                  variance = mean(x_group^2), rsqrt(var + eps),
//                                  out = flatten * weight.
//                                  NOT the plain `RMSNorm` (:37): its only
//                                  related knob is `var_hidden_size`, a PREFIX
//                                  reduction that cannot express per-group norms.
//                                  The gate (`z`) is disabled here; Qwen's
//                                  hc_norm has none.
//
// ─── THE `1 + w` PARAMETERIZATION, RESOLVED EXPLICITLY ────────────────────────
// The two oracles disagree on where the identity lives and this is the single
// most expensive detail in the module (issue #1988, trap 2):
//
//   transformers: weight ZERO-init,  out = normed * (1.0 + w_hf)
//   vLLM:         weight ONES-init,  out = normed * w
//
// They coincide under `w = 1.0 + w_hf`, applied ONCE, at LOAD time. We mirror
// vLLM's op form — the kernel multiplies by `w` and knows nothing about the
// offset — and give the transform one and only one home, `HcNormWeightFromHf`,
// so that "did anyone already add the 1?" is a question about a single call site
// rather than about arithmetic buried in a kernel. It matters twice over:
//   * Skip it and every `hc_norm` scales by ~0. That looks like a broken
//     checkpoint, not like a broken port, and it costs a day.
//   * Apply it twice and the scale is ~2x. The published GGUF has the fold done
//     at CONVERT time, so a GGUF reader must NOT call this function.
//
// ─── THE GGUF FOLD IS A BLANKET RULE, NOT AN `hc_norm` RULE ───────────────────
// Read this before writing a loader. The sentence above is true and, stated on
// its own, it teaches the wrong lesson: a reader who skips the fold for
// `hc_norm` alone will DOUBLE-FOLD every other norm in the same file, which is
// the same silent ~2x defect moved one tensor to the left.
//
// The fold does not live in a qwen4exp code path at all. Every anchor below is
// read at our recorded llama.cpp pin, stock upstream tag `b10451`
// (`10bf611e533d81f739128304991c5e133c6aebd8`, `.agents/oracles/llama-cpp.md`).
// Stock upstream has no `qwen4exp` whatsoever there: `git grep -il qwen4exp`
// returns nothing tree-wide, so a released llama.cpp can neither convert nor
// load this architecture. The converter is ggml-org/llama.cpp PR #27742, head
// `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, still OPEN. Its
// `conversion/qwen4exp.py` declares `class Qwen4ExpTextModel(_Qwen35MRopeMixin,
// _LinearAttentionVReorderBase)`; `_LinearAttentionVReorderBase` is
// `conversion/qwen.py:438`, a subclass of `Qwen3NextModel` (`:365`, whose own
// signature is `class Qwen3NextModel(_QwenMtpMixin, Qwen2MoeModel)`), and there
// is NO `hc_norm` branch in the PR's `modify_tensors` — `hc_norm.weight` falls
// through to `super()`. The `+1` is the INHERITED Qwen3-Next rule, at
// `conversion/qwen.py:387-388`:
//
//     elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
//         data_torch = data_torch + 1
//
// So the rule a loader has to implement is: **every tensor whose name ends in
// `norm.weight` carries the fold, with `linear_attn.norm.weight` (the GDN
// `ssm_norm`) as the one exception.** `hc_norm`, `attn_q_norm` and `attn_k_norm`
// all match it. The PLE and indexer gammas — `.ple.norm_{key,query,conv}` and
// `.indexer.{q,k}_layernorm` — are folded too, by the PR's own branch, which
// returns early so they are never folded twice.
//
// This is a property of ONE IN-FLIGHT CONVERTER and not of "GGUF". #27742 is
// unmerged and can change before it lands, and a publisher using a different
// tool is under no obligation to match it. A loader must therefore treat the
// fold as a checkpoint-provenance question, not a format constant; the check is
// cheap, because an unfolded `hc_norm` is a ZERO-init gamma and a folded one is
// centred on 1.0. Corroborated on published artifacts during fresh review of
// #1988 (`unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ1_S` and `UD-Q4_K_XL`,
// `vumpt/...-Q4_K_M`, read by HTTP range request against the bf16 HF tensors):
// every `*hc_norm.weight` is HF + 1.0 exactly, elementwise, while `ssm_norm` is
// not folded and sits in [0.875, 1.023].
//
// ─── PRECISION ────────────────────────────────────────────────────────────────
// fp32 interior throughout, with the per-group sum of squares accumulated in
// double — the `deepseek_v4_mhc.cpp` house convention for a host reference.
// Upstream runs the norm in fp32 (`self._norm(x.float())`) and vLLM likewise
// (`x = x.float()`), so nothing here is a widening of the model path: this is a
// CPU reference and the device arm is the thing that must be fp32-accumulate and
// gated against these numbers. No `f32` model-path buffer is introduced.
//
// ─── SCOPE ────────────────────────────────────────────────────────────────────
// Host reference only, and NOT YET REACHED from a production entry point: the
// W1 config/registration wave (#1986) is still in review, so there is no
// `qwen4_exp` loader row for this to hang off. Named per AGENTS.md "Nothing
// lands dead": the wiring is owed by row `MODEL-MM-QWEN4-EXP` W5 (assembly)
// under issue #1978, and it is listed in the spec's `## Owed`.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm::qwen4_exp {

// `w_vllm = 1.0 + w_hf`, elementwise. See the parameterization note above. Call
// this exactly once, at load, on a HuggingFace `hc_norm.weight`. Never on a
// weight read out of a GGUF written by ggml-org/llama.cpp#27742, which has the
// fold applied already — and note that the SAME is true of `attn_q_norm`,
// `attn_k_norm` and the PLE and indexer gammas in that file, while `ssm_norm` is
// the one exception. That rule is spelled out above; skipping the fold for
// `hc_norm` alone double-folds everything else.
std::vector<float> HcNormWeightFromHf(const std::vector<float>& w_hf);

// Grouped RMSNorm over one token's `x`, in vLLM's `RMSNormGated(group_size=)`
// form with the gate disabled: `x.size() / group_size` INDEPENDENT reductions,
// eps inside the rsqrt, weight applied per element after the normalize.
//
//   out[g*G + d] = x[g*G + d] * rsqrt(mean_d(x[g*G + .]^2) + eps) * weight[g*G + d]
//
// Refuses an indivisible width the way `Qwen4ExpTextRMSNorm.__init__` (:164-165)
// does, and refuses a weight that is not the full flattened width.
std::vector<float> GroupedRmsNorm(const std::vector<float>& x, const std::vector<float>& weight,
                                  int64_t group_size, float eps);

// The four `Qwen4ExpTextGatedResidual` parameters, all bias-free `nn.Linear`
// weights in PyTorch's `(out_features, in_features)` row-major layout.
struct GatedResidualWeights {
  std::vector<float> hc_norm_weight;  // [hc*H]  — vLLM form, i.e. 1 + w_hf
  std::vector<float> mix_down;        // [R, hc*H]
  std::vector<float> mix_up;          // [hc*H, R]
  // EMPTY is `use_combine=False`, i.e. `block_inject_weight = None`. That arm is
  // the model's FINAL mixer: `Qwen4ExpTextModel` holds one of these and there is
  // NO trailing RMSNorm after it — `hc_norm` is the last normalization before
  // `lm_head` (issue #1988, trap 1).
  std::vector<float> block_inject;  // [hc, hc*H] or empty
};

struct GatedResidualResult {
  std::vector<float> mixed_input;         // [H]     — the block's input
  std::vector<float> hyper_input_normed;  // [hc*H]  — what the mean multiplies
  std::vector<float> injection_weights;   // [hc]    — empty on the use_combine=false arm
};

// One token of `Qwen4ExpTextGatedResidual.forward`. `hyper_input` is [hc*H] and
// is NOT modified: upstream returns it RAW, un-normed, and it is the raw stream
// that the write-back adds to.
//
// The two divisions by `hc` are different and both load-bearing:
//   * inside the SiLU, on the [R] low-rank intermediate, BEFORE the activation —
//     `silu(down(x) / hc)`, not `silu(down(x)) / hc`. SiLU is not homogeneous.
//   * inside the injection sigmoid, with the whole sigmoid scaled by 2 —
//     `2 * sigmoid(inject(x) / hc)`, range (0, 2), exactly 1.0 at a zero logit.
// There is NO division on the up-projection sigmoid, and the reduce over `hc` is
// a MEAN, not a sum.
GatedResidualResult GatedResidualForward(const std::vector<float>& hyper_input,
                                         const GatedResidualWeights& weights, int64_t hc,
                                         int64_t hidden, float eps);

// The rank-1 write-back, in place:
//
//   hyper[j*H + h] += block_out[h] * injection_weights[j]
//
// THIS IS THE FUSION SEAM, and it is deliberately the primitive rather than the
// convenience. Upstream spells it `hidden_states.unsqueeze(-2) *
// injection_weights.unsqueeze(-1)` then `.flatten(-2)`, and both llama.cpp
// implementations of this architecture materialise it as a `repeat_4d` + `mul`,
// i.e. a dense [H, hc, T] broadcast built and thrown away 96 times per forward
// pass at 48 layers x 2 sites. It is a rank-1 update. A device arm reads
// `block_out` once per (j, h) tile and `injection_weights[j]` once per row; it
// never allocates the broadcast. Keeping the in-place pointer form as the
// primitive means the fused kernel replaces exactly this function and nothing
// above it has to change.
void GatedResidualWriteBackInPlace(float* hyper, const float* block_out,
                                   const float* injection_weights, int64_t hc, int64_t hidden);

// Allocating wrapper over `GatedResidualWriteBackInPlace`, for tests and for the
// host path. Bit-identical to the in-place form by construction.
std::vector<float> GatedResidualWriteBack(const std::vector<float>& hyper_input,
                                          const std::vector<float>& block_out,
                                          const std::vector<float>& injection_weights, int64_t hc,
                                          int64_t hidden);

}  // namespace vllm::qwen4_exp
