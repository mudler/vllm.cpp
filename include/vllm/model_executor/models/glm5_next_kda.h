// GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) — the KDA linear-attention
// arm's numerics, as portable host (CPU) reference implementations.
//
// 34 of this model's 45 layers are KDA, so this arm is most of the model.
//
// ─── WHY THIS FILE EXISTS AND `kimi_kda.h` DOES NOT SERVE ────────────────────
// `Glm5NextTextForgetGate.forward` BRANCHES on `safe_gate_lower_bound`, and the
// published checkpoint (`zai-org/GLM-5.3-Flash`, `linear_attn_config
// .gate_lower_bound: -5.0`) takes the branch our Kimi-Linear KDA does not:
//
//   GLM-5.3-Flash : g = bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
//   Kimi-Linear   : g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
//
// They are DIFFERENT FUNCTIONS of the same inputs, not a clamp of one another.
// Both are smooth, both are negative, both decay, and both produce fluent text.
// Three separate things differ and each one alone is enough to make the model
// wrong while it still reads well:
//
//   (a) the SHAPE — a bounded logistic against an unbounded softplus;
//   (b) the RANGE — the sigmoid branch cannot leave `(bound, 0)`, so the decay
//       per step is floored at `exp(-5)`; the softplus branch is unbounded
//       below and drives the state to zero on a large gate;
//   (c) the SIGN of `decay_rate` — `+exp(A_log)` MULTIPLIES `g` inside the
//       sigmoid here, where the softplus branch negates it outside. Feeding
//       `-exp(A_log)` into the sigmoid mirrors the gate about `g = 0`, which
//       turns "this channel forgets nothing" into "this channel forgets
//       everything" and never once produces a NaN.
//
// A token gate cannot see any of that, which is why the branch is ported as its
// own code with its own goldens rather than parameterised onto `kimi_kda.cpp`.
// `kimi_kda.cpp` is Kimi-Linear's and is deliberately untouched by this file.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream is `huggingface/transformers` at the LANE PIN `v5.16.1` (the first
// release carrying `glm5_next`; `v5.16.0` is 404 for this model), file
// `src/transformers/models/glm5_next/modular_glm5_next.py` unless stated. vLLM
// implements `glm5_next` at NO revision — not at the parity pin `555967922`,
// not at vLLM `main` — so under AGENTS.md "When vLLM has no implementation"
// transformers is the sole admissible reference here. The lane pin itself is
// W0's deliverable ([#2096]); see `.agents/specs/glm5-next-flash.md` §Oracles.
//
//   OURS                          <-  UPSTREAM (transformers @ v5.16.1)
//   Glm5NextForgetGate            <-  :375-408 (`Glm5NextTextForgetGate`),
//                                     the branch at :398-399 and the fallback
//                                     at :401-405
//   Glm5NextRmsNormGated          <-  :408-426 (`Glm5NextTextRMSNormGated`),
//                                     constructed at :635 with
//                                     `eps = config.rms_norm_eps` = 1e-5
//   Glm5NextL2Norm                <-  :429-437 (`l2norm`)
//   Glm5NextMixedQkvConvWeight    <-  :620-628 (the ONE grouped conv) vs the
//                                     checkpoint's THREE `{q,k,v}_conv1d`
//   Glm5NextMixedQkvConv          <-  inkling/modeling_inkling.py:441-457
//                                     (`causal_conv1d_update`) and :461-480
//                                     (`causal_conv1d_fn`), imported at :78
//   Glm5NextKdaLayerForward       <-  :597-746 (`Glm5NextTextLinearAttention`)
//                                     over :441-491 (`recurrent_kimi_delta_
//                                     attention`)
//
// ─── THREE LAYOUT FACTS, EACH MEASURED FROM THE REAL SHARD HEADERS ───────────
//   1. The reference has ONE grouped depthwise `nn.Conv1d` over the
//      concatenated `[q; k; v]` channel axis (`in = out = groups = 3*8192 =
//      24576`, kernel 4, BIAS-FREE). The checkpoint stores THREE separate
//      depthwise convs, `self_attn.{q,k,v}_conv1d.weight`. Concatenate them in
//      **q, k, v** order. Any other order is a silent permutation of channels.
//   2. `g`, `beta` and the output gate are computed from the **PRE-CONV**
//      hidden states (:709, :710, :742 all read `hidden_states`), so they must
//      NOT be fused into the conv path. Fusing them is cheap, plausible, and
//      changes every one of them.
//   3. The cache is a `[B, conv_dim, K]` conv state plus an fp32
//      `[B, H, Dk, Dv]` recurrent state — `[B, 24576, 4]` and
//      `[B, 64, 128, 128]` at this model's geometry, 4 MiB per layer per
//      sequence and ~136 MiB across the 34 KDA layers.
//
// ─── `vt::KdaChunkPrefill` IS NOT REACHABLE FROM THIS MODEL ──────────────────
// The chunked prefill op takes the RAW gate projection and FUSES the gate on
// device (`ops.h`: "kda_gate_cumsum fuses the gate:
// -exp(a_log)*softplus(g_raw+dt_bias)"), and its CPU reference does the same
// (`src/vt/cpu/cpu_ops.cpp:1779-1786`). That is the SOFTPLUS branch, baked into
// the vendored FLA Triton-AOT cubins. GLM-5.3-Flash needs the sigmoid branch,
// and there is no `(a_log, dt_bias, g_raw)` that makes the fused softplus
// reproduce it: inverting it needs `g_raw = log(exp(-target) - 1)`, which
// diverges to `-inf` as the gate approaches 0 — precisely where 34 layers of
// this model spend most of their channels. So BOTH the prefill and the decode
// path here route through `vt::KdaGatedDeltaRule`, which consumes an ALREADY
// COMPUTED per-K-channel log-decay `g` and therefore does not care which branch
// produced it. Recorded as owed debt (O14) rather than worked around.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_KDA_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_KDA_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vt/ops.h"  // vt::Queue, vt::KdaGatedDeltaRule

namespace vllm::glm5_next_kda {

// ── (1) the forget gate ──────────────────────────────────────────────────────

// `Glm5NextTextForgetGate.forward` (:389-405), BOTH branches, because the
// branch is selected by a config field that is `float | None` upstream and this
// port mirrors upstream rather than hardcoding the published checkpoint's
// value. Per (token, head h, channel d), with `g1` the low-rank projection
// `f_b_proj(f_a_proj(x))`:
//
//     g          = g1[t, h, d] + dt_bias[h*D + d]        (:393, always added)
//     decay_rate = exp(A_log[h])                          (POSITIVE, per head)
//     bound present : out = bound * sigmoid(decay_rate * g)
//     bound absent  : out = -decay_rate * softplus20(g)
//
// where `softplus20(g) = g` when `g > 20` and `log(1 + exp(g))` otherwise
// (:403, the overflow linearisation).
//
// `safe_gate_lower_bound` is `config.linear_lower_bound`, which upstream
// installs as -5.0 whenever the `linear_attn_config` dict is present and does
// not explicitly disable `safe_gate` (`configuration_glm5_next.py:195-199`), so
// the published checkpoint takes the FIRST branch. `bound` is NEGATIVE there
// and is used as-is, NOT negated: the return is `self.safe_gate_lower_bound *
// torch.sigmoid(...)`, so a positive bound would produce a POSITIVE log-decay,
// i.e. a state that grows without bound. This refuses a non-negative bound by
// name rather than producing that model.
//
// The result is the per-K-channel LOG-decay the recurrence exponentiates; it is
// `g` in `vt::KdaGatedDeltaRule`'s contract and is passed there unchanged.
//
//   g1      : [num_tokens, num_heads*head_dim]   row-major
//   a_log   : [num_heads]                        row-major
//   dt_bias : [num_heads*head_dim]               row-major, REQUIRED (:384)
// Returns  : [num_tokens, num_heads, head_dim]   row-major
std::vector<float> Glm5NextForgetGate(
    const std::vector<float>& g1, const std::vector<float>& a_log,
    const std::vector<float>& dt_bias, int64_t num_tokens, int64_t num_heads,
    int64_t head_dim, std::optional<double> safe_gate_lower_bound);

// The low-rank bottleneck that FEEDS the gate, `f_b_proj(f_a_proj(x))`
// (:392). `f_a` projects hidden -> head_dim (the rank) and `f_b` head_dim ->
// H*D, with NO activation between them: a pure low-rank linear map, bias-free
// on both legs. Named and ported here rather than reused so that this arm's
// gate chain is readable end to end in one file; the math is a plain
// [out, in] x [in] matvec twice and is gated against a double reference.
//
//   x    : [num_tokens, hidden_size]             row-major
//   f_a  : [head_dim, hidden_size]               row-major
//   f_b  : [num_heads*head_dim, head_dim]        row-major
// Returns: [num_tokens, num_heads*head_dim]      row-major
std::vector<float> Glm5NextLowRankProjection(const std::vector<float>& x,
                                             const std::vector<float>& f_a,
                                             const std::vector<float>& f_b,
                                             int64_t num_tokens,
                                             int64_t hidden_size,
                                             int64_t num_heads,
                                             int64_t head_dim);

// ── (2) the strict-fp32 gated output norm ────────────────────────────────────

// The activation dtype the model runs in. vLLM resolves ONE model dtype and
// every layer inherits it (AGENTS.md "Inherit vLLM defaults"); this enum exists
// only so `Glm5NextRmsNormGated` can reproduce upstream's final
// `.to(input_dtype)` (:426), which is the ONLY place the model dtype is
// observable inside a norm that is otherwise required to compute in fp32.
enum class Glm5NextActivationDType { kFloat32, kBFloat16 };

// `Glm5NextTextRMSNormGated.forward` (:414-426), per (token, head) over
// head_dim:
//
//     var    = mean_d(x^2)
//     normed = x * rsqrt(var + eps) * weight[d]
//     out    = normed * sigmoid(gate)          <- SIGMOID, not silu
//     return out.to(input_dtype)
//
// Three values here are traps and each is upstream's, not a choice:
//   * the activation is SIGMOID (:412). Qwen3.5-GDN and FLA's own KDA both use
//     silu at this position, and silu(z) = z*sigmoid(z) differs from
//     sigmoid(z) by a factor of z that is ~1 near z = 1 and unbounded away
//     from it.
//   * `eps` is `rms_norm_eps` = 1e-5, PASSED IN at :635. The constructor
//     default is 1e-6 (:410), so a port that constructs with the default is
//     wrong by 10x on a term that only matters where it matters most, in a
//     near-zero-variance row.
//   * the norm is STRICT FP32 and the WEIGHT is upcast too (:418, :421). This
//     is the annotated `f32` exception AGENTS.md requires a reason for, and
//     the reason is upstream's own comment: "Strict FP32 norm (do not downcast
//     on the weights)" — the reciprocal square root of a bf16-rounded variance
//     loses the low bits that separate two adjacent RMS scales. The host
//     reference is float-in/float-out, so `out_dtype` carries the ONLY part of
//     that polarity a host vector can express: the result is rounded to the
//     model dtype on the way out, and to nothing else on the way through.
//
//   x, gate : [num_tokens, num_heads, head_dim]  row-major
//   weight  : [head_dim]                         row-major
// Returns   : [num_tokens, num_heads, head_dim]  row-major
std::vector<float> Glm5NextRmsNormGated(
    const std::vector<float>& x, const std::vector<float>& gate,
    const std::vector<float>& weight, int64_t num_tokens, int64_t num_heads,
    int64_t head_dim, double eps,
    Glm5NextActivationDType out_dtype = Glm5NextActivationDType::kFloat32);

// ── (3) l2norm ───────────────────────────────────────────────────────────────

// `l2norm` (:429-437). Divides each row by `sqrt(sum(x*x) + eps)` — the eps is
// INSIDE the square root and is ADDED, which is upstream's deliberate match to
// FLA's Triton kernel and NOT `F.normalize`'s `x / max(norm, eps)`. Its own
// comment says so at :433. The two agree to within eps on any row of ordinary
// magnitude and disagree by orders of magnitude on a near-zero row, which is
// exactly the row a randomized test is least likely to draw.
//
//   x    : [num_rows, dim]  row-major
// Returns: [num_rows, dim]  row-major
std::vector<float> Glm5NextL2Norm(const std::vector<float>& x, int64_t num_rows,
                                  int64_t dim, double eps = 1e-6);

// ── (4) the q/k/v short convs ────────────────────────────────────────────────

// Concatenate the checkpoint's THREE separate depthwise conv kernels into the
// ONE grouped kernel the reference declares (:620-628). Order is q, k, v, which
// is the order `torch.cat` builds `mixed_qkv` in at :655-661. Every conv here
// is bias-free, so there is no bias to concatenate.
//
//   q_conv, k_conv, v_conv : [qkv_dim, kernel_size] each, row-major
// Returns                  : [3*qkv_dim, kernel_size]  row-major
std::vector<float> Glm5NextMixedQkvConvWeight(const std::vector<float>& q_conv,
                                              const std::vector<float>& k_conv,
                                              const std::vector<float>& v_conv,
                                              int64_t qkv_dim,
                                              int64_t kernel_size);

// The depthwise causal short conv over the concatenated `mixed_qkv` stream,
// with the conv-state carry, followed by the `hidden_act` activation.
//
// Upstream reaches this through two entry points that compute the same thing:
// `causal_conv1d_update` on a single-token decode (inkling:441-457) and
// `causal_conv1d_fn` after `update_conv_state` on a prefill (:461-480, called
// at :683-696). Both reduce to a causal convolution over
// `[conv_state ++ x]` with the state zero-filled for a fresh sequence, so this
// is ONE function.
//
// `conv_state` is `[channels, state_len]` IN AND OUT and holds the most recent
// `state_len` positions of the PRE-conv stream, newest last. `state_len` is
// `conv_kernel_size` at this model's cache geometry (`[B, 24576, 4]`), one
// column wider than the K-1 the arithmetic needs; upstream's
// `conv_state.copy_(hidden_states_new[:, :, -state_len:])` (inkling:452) keeps
// that slack column, and mirroring the width is what makes our cache spec agree
// with the reference's. Pass an empty vector for a fresh sequence with no cache
// (equivalent to an all-zero state); pass a sized vector to carry.
//
//   x          : [num_tokens, channels]        row-major, PRE-conv
//   weight     : [channels, kernel_size]       row-major
//   conv_state : [channels, state_len] or empty, IN AND OUT
//   activation : `config.hidden_act`; "silu" on this checkpoint. Any other
//                spelling is refused by name rather than silently ignored.
// Returns      : [num_tokens, channels]        row-major
std::vector<float> Glm5NextMixedQkvConv(const std::vector<float>& x,
                                        const std::vector<float>& weight,
                                        int64_t num_tokens, int64_t channels,
                                        int64_t kernel_size,
                                        std::vector<float>* conv_state,
                                        const std::string& activation);

// ── (5) the assembled KDA layer ──────────────────────────────────────────────

// The geometry, all of it from `Glm5NextTextConfig` and none of it a default
// this port chose. `Glm5NextParams::kda` (src/vllm/model_executor/models/
// glm5_next.h) carries the same values; this struct is separate so that this
// header stays free of the model-private config header.
struct Glm5NextKdaDims {
  int64_t hidden_size = 0;       // 4096
  int64_t num_heads = 0;         // linear_num_heads = 64
  int64_t head_dim = 0;          // linear_head_dim = 128
  int64_t conv_kernel_size = 0;  // linear_conv_kernel_dim = 4
  double rms_norm_eps = 1e-5;    // the o_norm eps, passed in at :635
  // Absent selects the softplus branch. -5.0 on the published checkpoint.
  std::optional<double> gate_lower_bound;
  std::string hidden_act = "silu";
  Glm5NextActivationDType activation_dtype = Glm5NextActivationDType::kFloat32;

  int64_t qkv_dim() const { return num_heads * head_dim; }
  int64_t conv_dim() const { return 3 * qkv_dim(); }
};

// Host-float weights for one KDA layer, in the checkpoint's own packing: the
// three separate convs stay separate here and are concatenated inside.
// Every projection is `[out, in]` row-major and bias-free.
struct Glm5NextKdaLayerWeights {
  std::vector<float> q_proj;    // [qkv_dim, hidden]
  std::vector<float> k_proj;    // [qkv_dim, hidden]
  std::vector<float> v_proj;    // [qkv_dim, hidden]
  std::vector<float> q_conv1d;  // [qkv_dim, K]
  std::vector<float> k_conv1d;  // [qkv_dim, K]
  std::vector<float> v_conv1d;  // [qkv_dim, K]
  std::vector<float> f_a_proj;  // [head_dim, hidden]
  std::vector<float> f_b_proj;  // [qkv_dim, head_dim]
  std::vector<float> dt_bias;   // [qkv_dim]
  std::vector<float> a_log;     // [num_heads]
  std::vector<float> b_proj;    // [num_heads, hidden]
  std::vector<float> g_a_proj;  // [head_dim, hidden]
  std::vector<float> g_b_proj;  // [qkv_dim, head_dim]
  std::vector<float> o_norm;    // [head_dim]
  std::vector<float> o_proj;    // [hidden, qkv_dim]
};

// The per-sequence KDA cache: what `Glm5NextTextLinearAttention` reads at
// :667-668 and writes at :683 and :739.
//
// `recurrent_state` is f32 and stays f32 whatever the model dtype is. That is
// the second annotated `f32` exception in this arm, and upstream annotates it
// twice: `cache_params.update_recurrent_state(last_recurrent_state.to(torch
// .float32), ...)` at :739 casts EXPLICITLY, and :452 says why — "calculations
// happen in float as states are more susceptible to rounding errors". The
// state is a running sum over the whole sequence, so a bf16 store would
// accumulate a rounding error that has no way out.
struct Glm5NextKdaCache {
  // [conv_dim, state_len], newest position last. Empty => fresh sequence.
  std::vector<float> conv_state;
  // [num_heads, head_dim, head_dim] f32, laid out as state[h][v][k] to match
  // vt::KdaGatedDeltaRule's [N, Hv, Dv, Dk]. The reference's own [k][v] layout
  // (:471) is the transpose of this one. Empty => fresh sequence.
  std::vector<float> recurrent_state;
};

// `Glm5NextTextLinearAttention.forward` (:641-746), single sequence, as a host
// reference. The delta recurrence routes through the SHARED SEAM
// `vt::KdaGatedDeltaRule` (include/vt/ops.h) with the already-computed
// per-K-channel log-decay `g`; see the header comment for why
// `vt::KdaChunkPrefill` cannot serve this model.
//
// `queue` must be a CPU queue. The device arm of this layer is the assembled
// text forward's (W5) and is refused by name here rather than half-built.
//
// `cache` may be null (a one-shot forward with no carry). When it is not null
// it is read for the initial conv and recurrent state and written with the
// final ones, so `Forward(x[0:T])` and `Forward(x[0:T-1])` then
// `Forward(x[T-1:T])` on the same cache agree.
//
//   hidden_states : [num_tokens, hidden_size]  row-major, POST input-layernorm
// Returns         : [num_tokens, hidden_size]  row-major, POST o_proj
std::vector<float> Glm5NextKdaLayerForward(
    const Glm5NextKdaLayerWeights& weights,
    const std::vector<float>& hidden_states, const Glm5NextKdaDims& dims,
    int64_t num_tokens, Glm5NextKdaCache* cache, vt::Queue& queue);

}  // namespace vllm::glm5_next_kda

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_KDA_H_
