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

// A source of ONE routed expert's host-f32 weights, consulted on demand.
//
// ─── WHY THIS EXISTS, AND IT IS ARITHMETIC AND NOT PREFERENCE ────────────────
//
// `MoeLayerWeights::expert_gate_up` and `expert_down` are the whole bank. At
// the published geometry (288 experts, `moe_intermediate_size` 2048,
// `hidden_size` 4096) that is:
//
//   | what | f32 bytes | GiB |
//   |---|---:|---:|
//   | one expert's `gate_up`, [2 * 2048, 4096] | 67,108,864 | 0.0625 |
//   | one expert's `down`, [4096, 2048] | 33,554,432 | 0.03125 |
//   | **one expert, both** | **100,663,296** | **0.09375** |
//   | one sparse layer's 288 experts | 28,991,029,248 | 27.0 |
//   | the 42 sparse layers' experts, all held | 1,217,623,228,416 | **1,134.0** |
//   | usable on `dgx:gpu0`, the largest device this project reaches | | ~119.63 |
//
// A resident float bank is 1,134 GiB against a 119.63 GiB box — 9.5x over — and
// `glm5_next_bridge.h`'s `kBridgeTensorF32ByteCeiling` already refuses one bank
// BY NAME at 9.0 GiB, which is that gate working rather than an obstacle to
// route around. **`num_experts_per_tok` is 8 of 288**, so what a token actually
// needs is at most 8 experts, and what a step needs is the DISTINCT experts its
// tokens hit. Decoding one at a time and dropping it makes the peak ONE expert:
// 0.09375 GiB, 0.078% of the box, and 12,096x under the whole-layer figure.
//
// **What this rules out, said positively.** There is no cache and no map keyed
// by expert index, for the same reason `glm5_next_bridge.h` has no `BridgeTower`:
// either one turns "one expert" into "every expert this request has ever
// selected", which is the bank again with a slower ramp. `MoeForward` reuses two
// buffers across the experts of one call and holds nothing between calls.
class ExpertSource {
 public:
  virtual ~ExpertSource() = default;

  // Fill `gate_up` with expert `e`'s [2 * moe_intermediate_size, hidden_size]
  // (GATE first, then up — the fused order `Glm5NextTextExperts` declares) and
  // `down` with its [hidden_size, moe_intermediate_size].
  //
  // The implementation OVERWRITES both buffers. `MoeForward` hands it the same
  // two vectors for every expert of a call, which is what makes the peak one
  // expert; an implementation that appended, or that kept its own copy, would
  // defeat the whole point of the interface.
  virtual void Expert(int64_t e, std::vector<float>& gate_up,
                      std::vector<float>& down) = 0;
};

// The routed-expert banks in the checkpoint's OWN block encoding, BORROWED.
//
// ─── WHY A THIRD RESIDENCY, WHEN THE OTHER TWO ALREADY BOUND THE PEAK ────────
//
// `ExpertSource` above bounds the PEAK at one expert and it does that
// correctly. What it does not bound is the WORK: it decodes an expert's
// `[2I, H]` + `[H, I]` out of GGUF blocks into host f32 on EVERY step, for
// every expert that step's tokens hit. The blocks it decodes FROM are 2.20 GiB
// per sparse layer and are already in the file; the f32 it decodes TO is 27.0
// GiB per sparse layer and is thrown away before the next step. Reading the
// blocks where they lie copies nothing and decodes nothing.
//
// ─── AND IT IS NOT NEW NUMERICS, WHICH IS THE POINT ──────────────────────────
//
// `vt::MoeGateUpSwiGLUGrouped` (`include/vt/ops.h`) is the shared keep-quant
// seam that DeepSeek-V4's private fused kernel was PROMOTED into, in its own
// words "so any keep-quant MoE arch inherits it". Its epilogue is specified as
// `gate = min(F.(gate_w.xq), limit)`, `up = clamp(F.(up_w.xq), -limit, limit)`,
// `out = gate.sigmoid(gate).up` — clamped SwiGLU at alpha=1, beta=0, which is
// `ExpertGate` above, which is `deepseek_v4::ClampedSwiGLU` at alpha=1, beta=0,
// which is `_apply_gate` (`:137-142`). Same function, three spellings, and this
// arm uses the one that already has two providers. The down projection is
// `vt::MatmulBTQuantGrouped`. Both dispatch on the queue's device, so this is
// also the ONLY expert arm a CUDA queue could ever run.
//
// ─── THE SHAPES ARE THE CHECKPOINT'S OWN, WITH NO REPACK ─────────────────────
//
// `LoadStackedExperts` (`glm5_next_loader.cpp`) builds each bank as
// `OwnGgufQuantBlocks(t, E * N, K)` and then RESHAPES the record to `[E, N, K]`;
// its own comment says the bytes are identical either way. `[E * N, K]` is
// exactly the stacked tower both ops declare, so these views are a rank change
// over the same pointer and never a copy. Measured on the published artifact:
// every `ffn_{gate,up}_exps` is `ne = [4096, 2048, 288]` and every
// `ffn_down_exps` is `[2048, 4096, 288]`.
//
// ─── WHAT THIS ARM CHANGES NUMERICALLY, AND IT IS NOT NOTHING ────────────────
//
// The seam quantizes the ACTIVATION to Q8_K once per call; the two f32 arms do
// not. So this arm and the f32 arms do NOT agree bit-for-bit, and a gate
// between them is an error band and not an equality. The ROUTER is untouched
// and still runs in f32 (`RouterLogits`), so expert SELECTION is unaffected —
// which matters, because selection error is bimodal and a tolerance cannot see
// it, while this one is an ordinary value perturbation that a tolerance can.
// This is the same activation-quantization the DeepSeek-V4 and Qwen3.5 MoE arms
// already ship.
//
// BORROWED, exactly like `expert_source`: every `data` pointer here aims into
// the loader's mmap or its owned block bytes, and must outlive every
// `MoeForward` call that reads the enclosing struct.
struct MoeQuantBanks {
  vt::Tensor gate;  // [E * moe_intermediate_size, hidden_size], block-quant
  vt::Tensor up;    // [E * moe_intermediate_size, hidden_size], the SAME dtype
  vt::Tensor down;  // [E * hidden_size, moe_intermediate_size], block-quant
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
  // The ON-DEMAND alternative to the two banks above, BORROWED and not owned.
  //
  // Exactly one of the two shapes is admissible per layer and `MoeForward`
  // refuses the other two states BY NAME: both populated is ambiguous, and
  // neither populated would run the router and then read an empty bank as a
  // zero weight — a finite, fluent, wrong block, which is the failure this
  // row's headers keep naming. The pointer must outlive every `MoeForward`
  // call that reads this struct.
  ExpertSource* expert_source = nullptr;
  // The KEEP-QUANT alternative to both shapes above, and the one `MoeForward`
  // prefers when it is present. Valid iff `has_quant_banks`; borrowed, with the
  // lifetime `expert_source` has. A layer may carry this together with an
  // `expert_source` — that is not the ambiguous state the two f32 shapes are in,
  // because the two are the same weights in two encodings and the keep-quant one
  // simply wins. What is refused is BOTH f32 shapes at once, as before.
  MoeQuantBanks quant_banks;
  bool has_quant_banks = false;
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
// THE EXPERTS ARE VISITED ONCE EACH, grouped by expert before any of them is
// evaluated. That is upstream's own order (`:120-135` loops over the HIT
// experts and `index_add_`s each one's tokens, rather than looping over tokens)
// and it is what bounds the residency when `w.expert_source` is set: a second
// visit to an already-seen expert would be a second 96 MiB decode. Each [t, j]
// slot is still computed independently and written to its own row of the
// combine's `[T, K, H]` operand, so the grouping moves no arithmetic and the
// resident path is byte-identical to a token-major visit.
//
//   hidden : [num_tokens, hidden_size]  row-major
// Returns  : [num_tokens, hidden_size]  row-major
std::vector<float> MoeForward(const MoeDims& d, const MoeLayerWeights& w,
                              const std::vector<float>& hidden, int64_t num_tokens,
                              vt::Queue& queue);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_MOE_H_
