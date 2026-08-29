// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W5: the 288 routed + 1 shared
// expert MoE block, its grouped `noaux_tc` router, and the clamped-SwiGLU
// epilogue both feed-forward kinds share.
//
// Model-private header, deliberately not under `include/`: nothing outside this
// model needs these types yet, and `include/vllm.h` is the ABI seam a shipped
// capability is exposed through. Same arrangement as `glm5_next.h` (W1),
// `glm5_next_mhc.h` (W4) and `glm5_next_dsa.h` (W3).
//
// ORACLE. vLLM registers no `glm5_next` at our parity pin `555967922` nor at its
// `main`, and neither do vllm-omni, SGLang or llama.cpp. Under AGENTS.md "When
// vLLM has no implementation" the reference for this surface is `transformers`
// **v5.16.1**, the lane pin W0 (#2096) recorded in
// `.agents/oracles/transformers.md`. Every anchor below is
// `src/transformers/models/glm5_next/modeling_glm5_next.py` at that tag — the
// FLATTENED, EXECUTED file, sha256
// 2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b. Reading the
// modular file and assuming its inheritance survives into the generated class is
// the error that cost `MODEL-MM-QWEN4-EXP` a review cycle.
//
// ─── PORT ANCHORS (file:line on BOTH sides) ──────────────────────────────────
//   OURS                        <-  transformers v5.16.1, models/glm5_next/
//   MoeDimsFrom                 <-  configuration_glm5_next.py:104-120 and :142 (the
//                                   published router constants)
//   RouterLogits                <-  modeling_glm5_next.py:159-160
//                                   (`Glm5NextTextTopkRouter.forward`, the
//                                   EXPLICIT float32 upcast of both operands)
//   RouteTopk                   <-  modeling_glm5_next.py:158-183; ours wraps
//                                   the shared seam `vt::MoeRouterTopK`
//                                   (include/vt/ops.h) on its grouped
//                                   `noaux_tc` arm
//   ExpertGate                  <-  modeling_glm5_next.py:137-142
//                                   (`Glm5NextTextExperts._apply_gate`); ours
//                                   wraps `deepseek_v4::ClampedSwiGLU`
//                                   (include/vllm/model_executor/models/
//                                   deepseek_v4_moe.h) at alpha=1, beta=0
//   DenseMlpForward             <-  modeling_glm5_next.py:98-104
//                                   (`Glm5NextTextMLP.forward`)
//   MoeForward                  <-  modeling_glm5_next.py:120-135 (`Experts
//                                   .forward`) + :200-207 (`TextMoE.forward`);
//                                   the weighted scatter-combine runs on the
//                                   shared seam `vt::MoeCombine`
//
// ─── WHY NOTHING HERE IS NEW NUMERICS ────────────────────────────────────────
// Both pieces this model needs already exist in this tree and are gated, so this
// file BINDS them rather than reimplementing them. That polarity is the point:
//
//   * The router is the DeepSeek `noaux_tc` grouped top-k, and `vt::MoeRouterTopK`
//     implements it on its `num_expert_group > 0` arm — sigmoid scores, the bias
//     added for SELECTION only, the weight read from the UNBIASED score,
//     renormalize, then `routed_scaling_factor`. Step for step that is
//     `Glm5NextTextTopkRouter.forward`.
//   * The epilogue is `deepseek_v4::ClampedSwiGLU` at `alpha = 1, beta = 0`,
//     which reduces to `silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`
//     — `_apply_gate`'s "Simple swiglu instead of alpha" comment, exactly.
//
// ─── THE FIVE PLACES THIS PORT GOES WRONG SILENTLY ───────────────────────────
//   1. `n_group` and `topk_group` are BOTH 1, which makes the group stage a
//      no-op — one group holding all 288 experts, and the single group always
//      survives. It is passed through anyway rather than special-cased: the
//      config carries the fields, upstream runs the code, and a port that
//      hardcodes "ungrouped" cannot represent a checkpoint that sets them.
//   2. `routed_scaling_factor` 2.5 is applied AFTER the renormalize divide
//      (`:179-182`), not before and not folded into the shared term. Reversing
//      the order changes every routed weight by the normaliser.
//   3. The shared expert's width is `moe_intermediate_size * n_shared_experts`
//      (`:196-198`), not `intermediate_size`. At `n_shared_experts` 1 the two
//      spellings differ by 6x on the published checkpoint (2048 vs 12288).
//   4. The shared term is added UNSCALED (`:206`) — `routed_scaling_factor`
//      multiplies the router weights and therefore only the routed sum. This is
//      `vt::MoeCombine`'s default `routed_scale = 1.0f` polarity, and the flag
//      exists because getting it wrong is the error a token gate catches late.
//   5. The router GEMM runs in explicit float32 on BOTH operands
//      (`:160`, `hidden_states.type(torch.float32)` and
//      `self.weight.type(torch.float32)`) whatever the model dtype is. This is
//      the annotated `f32` exception AGENTS.md requires a reason for, and the
//      reason is upstream's: a bf16 router logit quantises the sigmoid score to
//      ~3 decimal digits, and the 8th and 9th expert of 288 are routinely closer
//      than that. Top-k error is BIMODAL, so the damage is a different expert
//      set rather than a slightly different value.
//
// HOST REFERENCE. Every buffer here is `float`, as in `glm5_next_kda.cpp`,
// `glm5_next_dsa.cpp` and `glm5_next_mhc.cpp`. The device arm of this block is
// the assembled text forward's and is refused by name rather than half-built.
//
// NOT REACHED YET, and the reason is NOT that `load_weights` refuses. It no
// longer does: W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242))
// landed the weight tower, so the GGUF arm returns a real `Glm5NextLoadedModel`
// and a handle to forward now exists. What refuses is
// `ForwardGlm5NextForConditionalGeneration` ITSELF
// (`glm5_next_registry.cpp:129`), because the decoder layer and the assembled
// `Glm5NextTextModel::Forward` that would call this block do not exist yet; they
// are W5b's ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)). W5 lands
// the MoE block and the KV-cache spec -- and the KV-cache spec IS reached, from
// the production `make_kv_cache` factory hook -- while this block is carried as
// owed debt in `.agents/specs/glm5-next-flash.md` `## Owed` (O23).
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_MOE_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_MOE_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/glm5_next.h"
#include "vt/ops.h"  // vt::Queue, vt::MoeRouterTopK, vt::MoeCombine

namespace vllm::glm5_next {

// The MoE geometry, resolved. Built from a parsed config by `MoeDimsFrom` below
// — never by hand in production code, because every one of these fields has a
// class default that differs from the published checkpoint's value except
// `n_group` and `topk_group`.
struct MoeDims {
  int64_t hidden_size = 0;           // `config.hidden_size`          — 4096
  int64_t n_routed_experts = 0;      // `config.n_routed_experts`     — 288
  int64_t n_shared_experts = 0;      // `config.n_shared_experts`     — 1
  int64_t num_experts_per_tok = 0;   // `config.num_experts_per_tok`  — 8
  int64_t moe_intermediate_size = 0; // `config.moe_intermediate_size`— 2048
  int64_t n_group = 0;               // `config.n_group`              — 1
  int64_t topk_group = 0;            // `config.topk_group`           — 1
  double routed_scaling_factor = 0.0;  // `config.routed_scaling_factor` — 2.5
  bool norm_topk_prob = true;        // `config.norm_topk_prob`       — true
  float swiglu_limit = 0.0f;         // `config.swiglu_limit`         — 10.0

  // `Glm5NextTextMLP(config, intermediate_size=config.moe_intermediate_size *
  // config.n_shared_experts)` (`:196-198`). NOT `config.intermediate_size`,
  // which is the DENSE layers' 12288.
  int64_t shared_intermediate_size() const {
    return moe_intermediate_size * n_shared_experts;
  }

  // Refuses a partial or incoherent group BY NAME rather than serving a wrong
  // routing: every field positive, `top_k <= n_routed_experts`, `n_group`
  // dividing `n_routed_experts`, and `topk_group` in `[1, n_group]` — which is
  // `vt::MoeRouterTopK`'s own admission rule, checked here so the message names
  // this model instead of the op.
  void Validate() const;
};

MoeDims MoeDimsFrom(const Glm5NextParams& p);

// The dense (non-MoE) feed-forward's weights, `Glm5NextTextMLP` (`:86-96`).
// Every projection is `[out, in]` row-major, bias-free (`bias=False` on all
// three).
struct DenseMlpWeights {
  std::vector<float> gate_proj;  // [intermediate, hidden]
  std::vector<float> up_proj;    // [intermediate, hidden]
  std::vector<float> down_proj;  // [hidden, intermediate]
};

// One sparse layer's weights, in the checkpoint's own packing: the routed
// experts arrive STACKED and gate/up arrive FUSED, which is how
// `Glm5NextTextExperts` declares them (`:116-117`) and how
// `scripts/convert-glm5-next-gguf.py` writes them.
struct MoeLayerWeights {
  // `Glm5NextTextTopkRouter.weight`, [n_routed_experts, hidden].
  std::vector<float> router_weight;
  // `e_score_correction_bias`, [n_routed_experts]. A BUFFER upstream, zeroed by
  // the constructor (`:156`); an empty vector here selects the no-bias arm,
  // which changes the group score from "sum of the top 2" to "the max" and is
  // therefore a different router, not a smaller one.
  std::vector<float> e_score_correction_bias;
  // [n_routed_experts, 2 * moe_intermediate_size, hidden], gate first.
  std::vector<float> expert_gate_up;
  // [n_routed_experts, hidden, moe_intermediate_size].
  std::vector<float> expert_down;
  // The shared expert, at `shared_intermediate_size()`.
  DenseMlpWeights shared;
};

// `Glm5NextTextTopkRouter.forward`'s return value (`:183`), plus the logits it
// computed on the way. `router_logits` is returned rather than discarded because
// it is the only place the fp32 router GEMM is observable, and because a gate
// that reads only the selection cannot tell a wrong logit from a wrong top-k.
struct MoeRouting {
  std::vector<float> router_logits;   // [num_tokens, n_routed_experts]
  std::vector<int32_t> topk_ids;      // [num_tokens, num_experts_per_tok]
  std::vector<float> topk_weights;    // [num_tokens, num_experts_per_tok]
};

// `F.linear(hidden.type(float32), weight.type(float32))` (`:160`).
//
//   hidden        : [num_tokens, hidden_size]        row-major
//   router_weight : [n_routed_experts, hidden_size]  row-major
// Returns         : [num_tokens, n_routed_experts]   row-major
std::vector<float> RouterLogits(const MoeDims& d, const std::vector<float>& hidden,
                                const std::vector<float>& router_weight,
                                int64_t num_tokens);

// The whole grouped `noaux_tc` selection, through the shared seam
// `vt::MoeRouterTopK`.
//
// `queue` must be a CPU queue: the op dispatches on the queue's device and
// handing it host pointers on a CUDA queue is a crash rather than a fallback.
//
// ORDER DEVIATION, recorded. Upstream calls `torch.topk(..., sorted=False)`, so
// the order of the `num_experts_per_tok` selected ids is unspecified; the seam
// emits them in DESCENDING selection-score order with the lowest expert index
// winning an exact tie, which is this tree's determinism convention and what
// makes CPU and CUDA agree bit-for-bit. The SET is identical and the per-id
// weight is identical, and the combine is a sum over the set, so no downstream
// value moves. A gate on this must therefore assert SET equality, never
// positional equality.
//
// RENORMALIZE DEVIATION, recorded and bounded. Upstream divides by
// `sum + 1e-20` (`:180`); the seam divides by `sum` with a `sum <= 0 -> 1`
// guard. The eight summands are sigmoid outputs in (0, 1), so the sum is
// positive on every finite input and the two denominators differ by a relative
// 1e-20 — 13 orders of magnitude below float32 epsilon.
//
//   hidden : [num_tokens, hidden_size] row-major
MoeRouting RouteTopk(const MoeDims& d, const MoeLayerWeights& w,
                     const std::vector<float>& hidden, int64_t num_tokens,
                     vt::Queue& queue);

// `Glm5NextTextExperts._apply_gate` (`:137-142`) for ONE row: split `gate_up`
// into its gate and up halves, clamp each (the gate MAX-ONLY, the up on BOTH
// sides), then `silu(gate) * up`.
//
// Delegates to `deepseek_v4::ClampedSwiGLU` at `alpha = 1, beta = 0`. The
// asymmetry is load-bearing and is upstream's, not a choice: clamping the gate
// symmetrically changes every strongly-negative channel, and the result stays
// smooth and plausible.
//
//   gate_up : [2 * intermediate] row-major, gate = [0, I), up = [I, 2I)
// Returns   : [intermediate]
std::vector<float> ExpertGate(const std::vector<float>& gate_up, int64_t intermediate,
                              float limit);

// `Glm5NextTextMLP.forward` (`:98-104`). The dense layers' feed-forward, and
// also the shared expert of every sparse layer — the same class at a different
// width, which is why there is one function.
//
//   hidden : [num_tokens, hidden_size]  row-major
// Returns  : [num_tokens, hidden_size]  row-major
std::vector<float> DenseMlpForward(const DenseMlpWeights& w,
                                   const std::vector<float>& hidden,
                                   int64_t hidden_size, int64_t intermediate,
                                   int64_t num_tokens, float limit);

// The composed sparse block: `Glm5NextTextExperts.forward` (`:120-135`) scattered
// by `Glm5NextTextMoE.forward` (`:200-207`), with the shared expert added
// unscaled.
//
//   out[t] = sum_j topk_weights[t, j] * expert_{ids[t,j]}(hidden[t])
//            + shared(hidden[t])
//
// The weighted sum runs on `vt::MoeCombine` at its default `routed_scale = 1.0f`,
// because `routed_scaling_factor` is already folded into the router weights by
// `MoeRouterTopKArgs::routed_scaling_factor`. Passing it in BOTH places would
// square it.
//
//   hidden : [num_tokens, hidden_size]  row-major
// Returns  : [num_tokens, hidden_size]  row-major
std::vector<float> MoeForward(const MoeDims& d, const MoeLayerWeights& w,
                              const std::vector<float>& hidden, int64_t num_tokens,
                              vt::Queue& queue);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_MOE_H_
