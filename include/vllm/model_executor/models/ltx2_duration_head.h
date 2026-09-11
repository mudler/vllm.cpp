// LTX-2.5 DURATION HEAD — predicts a shot's natural length from the connector
// outputs, so a request need not state `num_frames`.
//
// Row: MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model. Spec:
// .agents/specs/ltx-2-5.md (phase L5). Issue #435.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/duration_head/
//   OURS                        <-  UPSTREAM
//   Ltx2DurationPredict         <-  duration_head.py:89-118 (DurationHead.forward)
//   Ltx2DurationAttentionPool   <-  duration_head.py:45-49 (AttentionPooler.forward)
//   Ltx2DurationHeadConfig      <-  duration_head.py:63-71 + model_configurator.py:22-35
//   EnumerateLtx2DurationHeadTensors <- the module's own named_parameters()
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * THE OUTPUT IS `exp(mlp_out)`, NOT `mlp_out`. The head is trained in
//    LOG-seconds and exponentiates on the way out (duration_head.py:117-118).
//    Returning the raw regression gives a number that is finite, positive-ish and
//    a completely different duration.
//  * THE TWO STREAMS CONCATENATE ALONG THE TOKEN AXIS (`dim=1`, :113), after each
//    is projected to the SHARED pooler width. Concatenating along the feature axis
//    also type-checks when the widths happen to line up.
//  * THE MODALITY EMBEDDING IS ADDED AFTER THE PROJECTION (:109, :111), which is
//    what lets the pooler tell the streams apart. Adding it before the projection
//    puts it through a different linear map per stream and loses that. The
//    embedding is the ONLY thing that tags a stream — see the invariance below.
//  * THE POOLER IS `torch.nn.MultiheadAttention`, whose PACKED `in_proj_weight` is
//    [3 * E, E] in Q, K, V order. It is CROSS attention here — the queries are the
//    learnable tokens and the keys/values are the token stream — so the three
//    slices are applied to two different inputs, not one.
//
// ─── AN INVARIANCE, so nobody mistakes it for a gate hole ────────────────────
// THE CONCAT ORDER IS NOT OBSERVABLE. A mutation that reversed the two streams'
// concatenation left every golden green, and that is correct rather than a weak
// fixture: `AttentionPooler` is cross-attention with no mask and no positional
// encoding over the token axis, so it is PERMUTATION INVARIANT and upstream
// cannot distinguish the orders either. Measured on upstream — a reversed concat
// and a random permutation each move the pooled output by 2.98e-08 (f32
// reduction-order noise), while giving the audio stream the VIDEO modality
// embedding moves it by 4.80e-03.
//
// The consequence is worth stating plainly: what separates the two streams is the
// modality EMBEDDING, nothing else. Both facts are gated in
// tests/vllm/models/test_ltx2_pipeline.cpp, so if upstream ever gives the pooler
// positional information the invariance assertion fails and this note stops being
// true at the moment it stops being true.
//
// ─── NO MASK, BY CONSTRUCTION ────────────────────────────────────────────────
// The pooler takes no attention mask because the connector has already replaced
// every padded position with a learnable register and zeroed its mask
// (duration_head.py:15-16, embeddings_connector.py:139-152). This port therefore
// requires the connector's output, not a raw padded batch — a caller that hands
// it padded tokens gets a duration computed over the padding, silently.
//
// ─── DTYPE — TWO ARMS, AND THEY CLAIM DIFFERENT THINGS ───────────────────────
//
// Upstream resolves ONE pipeline dtype and it is `torch.bfloat16`
// (distilled.py:109), handed to `DurationPredictor.from_checkpoint` at
// :163-165, so upstream's head IS a bf16 module: all fifteen of its tensors
// narrow under `.to(dtype)` and every store point in its forward rounds. That
// arm is row LTX25-A24-DURATION-HEAD-BF16 (#2955), gap A24's eighth and last
// component, and it is what a production render loads
// (`Ltx2VideoEngine::Load` asks for `kBF16` at both call sites).
//
// f32 remains the PARITY arm of the goldens above, and it is WIDER than
// upstream. Both are served; `Ltx2VaeWeights::dtype` selects, and a third width
// refuses by name.
//
// WHICH CLAIM EACH ARM MAKES, kept apart on purpose — wave 5 of this gap failed
// review for gating an arithmetic width while claiming a storage one:
//
//  * STORAGE is the weight bag. On `kBF16` it holds 16-bit words and
//    `Ltx2VaeWeights::Bytes()` is EXACTLY half the f32 arm's over the same
//    tensor set. That ratio is the observable, so no byte count is quoted.
//  * ARITHMETIC is the activations. They stay in `std::vector<float>` holding
//    NARROWED values, exactly as the connector's arm does, and this port does
//    NOT claim the activation buffers are half. Saying so is the point: the
//    claim that is true is gated, and the claim that is not true is not made.
//    `Ltx2DurationWidthCounts` is what makes the arithmetic claim observable,
//    because nothing else on this path can see a dtype that is too wide.
//
// THE MEASURED RULES ARE IN `.agents/specs/ltx25-a24-duration-head-bf16.md` §3,
// with the rejected hypothesis beside each. The one whose obvious C++ spelling
// is wrong is the Linear: upstream's bf16 `F.linear` is bit-exactly
// `bf16(f32(a) @ f32(w)^T + f32(b))`, ONE rounding with the bias added at the
// accumulator's width. Rounding the GEMM and then adding the bias in bf16
// type-checks identically and disagrees in a QUARTER of all outputs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"  // Ltx2VaeWeights
#include "vt/dtype.h"                                  // vt::DType

namespace vllm {

class SafetensorsFile;

// DurationHead.__init__ defaults (duration_head.py:63-71), which are also
// DurationHeadConfigurator.from_metadata's `config.get` fallbacks
// (model_configurator.py:28-35). The two cross-attention dims mirror the DiT's
// own `cross_attention_dim` / `audio_cross_attention_dim`.
struct Ltx2DurationHeadConfig {
  int64_t video_cross_attention_dim = 4096;
  int64_t audio_cross_attention_dim = 2048;
  int64_t pooler_hidden_dim = 256;
  int64_t num_queries = 1;
  int64_t num_pooler_heads = 4;
  int64_t mlp_hidden = 256;
  std::string prefix;
};

// The parameter contract, in `named_parameters()` order. Note that the two bare
// `nn.Parameter`s (`video_modality_emb`, `audio_modality_emb`) come FIRST, before
// any submodule — that is torch's own ordering, and the parity suite asserts it.
struct Ltx2DurationHeadTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};
std::vector<Ltx2DurationHeadTensorSpec> EnumerateLtx2DurationHeadTensors(
    const Ltx2DurationHeadConfig& config);

// THE ARITHMETIC WIDTH, MADE OBSERVABLE FROM OUTSIDE THE HEAD (A24 wave 6).
//
// How many of the values the head PRODUCED could not have come out of a bf16
// store, out of how many it produced. On the bf16 arm every store point rounds,
// so `not_bf16` is ZERO; on the f32 parity arm it is essentially the whole
// population. `values` is the control that says the counter looked at anything
// at all — "zero wide values" is also what a counter that ran over nothing
// reports, and AGENTS.md names this exact blind spot: a token gate cannot detect
// a dtype that is too wide. The INPUT tokens are deliberately not counted; they
// are the caller's data, not the head's product.
//
// WHAT IT MEASURES IS NARROWER THAN "THE HEAD'S WIDTH", and the difference
// matters to whoever reads the number next. Each count is taken at its store
// point AFTER that store has run, so on the bf16 arm it is zero by idempotence
// whether the store point is there or not. It therefore detects a store point
// handed the WRONG ARM — an f32 head, or one site reverted while the others
// stay — and NOT a store point that is missing altogether: deleting one outright
// leaves this counter at zero. That half is covered by the bit-exact goldens and
// by the assertion that the returned seconds survive a bf16 round trip, both of
// which a deleted store does move.
struct Ltx2DurationWidthCounts {
  int64_t not_bf16 = 0;
  int64_t values = 0;
};

// AttentionPooler.forward (duration_head.py:45-49) on its own, so a pooler defect
// localizes instead of arriving as one wrong scalar. `tokens` is
// [batch, token_count, pooler_hidden_dim]; the result is
// [batch, num_queries, pooler_hidden_dim].
//
// THE ENTRY CONTRACT ON THE BF16 ARM, WHICH IS A CONTRACT AND NOT A GUARD.
// `tokens` are the CALLER'S already-stored activations, and this function does
// not narrow them. Upstream's pooler is a submodule and narrows nothing either:
// it receives whatever the enclosing forward stored, which is why an entry
// narrowing here MASKED a real store-point defect for a whole review round
// (see the note at the head of the definition). `Ltx2DurationPredict`, the only
// caller in this tree, satisfies the contract by storing at the projection and
// again after the modality-embedding add.
//
// No `Require` enforces it, deliberately. Checking it means walking the whole
// token buffer on every render-path call to re-derive something the one
// production caller establishes by construction, and it would refuse the use
// this entry point EXISTS for: a pooler case localizes a defect by choosing its
// own input, and the bf16 pooler case does exactly that. Handing it f32 tokens
// is well defined and is the sharper gate, because a pooler whose own store
// point was deleted then returns values that are visibly not on the bf16 grid.
// What the contract governs is COMPOSITION, not safety: tokens off the grid are
// simply not the chain upstream builds.
std::vector<float> Ltx2DurationAttentionPool(const Ltx2DurationHeadConfig& config,
                                             const Ltx2VaeWeights& weights, const float* tokens,
                                             int64_t batch, int64_t token_count,
                                             Ltx2DurationWidthCounts* widths = nullptr);

// DurationHead.forward (duration_head.py:89-118). Either stream may be null, and
// both being null throws exactly as upstream does (:104-105). Returns the
// predicted duration in SECONDS, [batch].
std::vector<float> Ltx2DurationPredict(const Ltx2DurationHeadConfig& config,
                                       const Ltx2VaeWeights& weights, const float* video_tokens,
                                       int64_t video_token_count, const float* audio_tokens,
                                       int64_t audio_token_count, int64_t batch,
                                       Ltx2DurationWidthCounts* widths = nullptr);


// ─── THE DRIVER (row LTX25-DURATION-HEAD-WIRE, #2900) ────────────────────────
//
// Everything above is the head's ARITHMETIC. What follows is what upstream wraps
// around it to turn a predicted duration into the frame count a render uses.
// Ported from packages/ltx-pipelines/src/ltx_pipelines/utils/, which is a
// different package from the head itself — kept together here because the two
// are useless apart, and because the port has no `ltx-pipelines` layer to mirror.
//
//   OURS                          <-  UPSTREAM
//   Ltx2SnapFramesToGrid          <-  utils/helpers.py:554-562
//   Ltx2SecondsToClampedNumFrames <-  utils/helpers.py:565-585
//   Ltx2DurationPredictFrames     <-  utils/blocks.py:850-889 (DurationPredictor.__call__)
//   Ltx2RequireNumFramesSource    <-  utils/blocks.py:894-905
//   Ltx2LoadDurationHeadWeights   <-  utils/blocks.py:816-848 (from_checkpoint)
//
// ─── THE FOUR RULES THAT FAIL SILENTLY HERE ──────────────────────────────────
//  * THE CLAMP PRECEDES THE SNAP (helpers.py:579-580). Both orders type-check
//    and they disagree whenever the floored grid point leaves the window.
//  * THE UNDERSHOOT REPAIR SNAPS UP AND IS ITSELF CAPPED (:581-584). Snapping
//    FLOORS, so a `min_frames` off the grid lands BELOW the window; upstream
//    ceils back onto the grid, then takes `min` with `max_frames`. That `min` is
//    why a min == max == 5 request returns 5, which is NOT on the 8k+1 grid:
//    upstream honours the window over the grid when both cannot hold.
//  * THE ROUNDING IS PYTHON'S, WHICH IS HALF-TO-EVEN, NOT `std::llround`.
//    0.34 s at 25 fps is exactly 8.5: upstream takes 8 and returns frame 1,
//    `llround` takes 9 and returns frame 9.
//  * A MISSING HEAD IS `None`, NOT AN ERROR, AND SO IS A PARTIAL ONE
//    (blocks.py:838-844). Every checkpoint predating LTX-2.5 / gemma4 has no
//    head, and upstream runs those happily.
//
// All four are gated against upstream's OWN answers, with the rejected rule
// beside each, in tests/vllm/models/ltx2_duration_wire_goldens.inc.

// `snap_frames_to_grid` (utils/helpers.py:554-562). Floors to `k * time_scale + 1`
// and refuses `frames < 1`, which is also what keeps the C++ division agreeing
// with Python's flooring `//`.
int64_t Ltx2SnapFramesToGrid(int64_t frames, int64_t time_scale);

// `seconds_to_clamped_num_frames` (utils/helpers.py:565-585). Clamps BEFORE
// snapping and repairs an undershoot upward; see the header note above.
int64_t Ltx2SecondsToClampedNumFrames(double seconds, double frame_rate, int64_t min_frames,
                                      int64_t max_frames, int64_t time_scale);

// `DurationPredictor.__call__` (utils/blocks.py:850-889). Either token stream may
// be null and both being null throws, exactly as the forward does. Upstream's
// single-item-batch refusal (:857-861) is expressed as a CONTRACT rather than a
// check: there is no `batch` parameter, so the shape it rejects cannot be asked
// for. `predicted_seconds`, when non-null, receives the head's raw prediction so
// a caller can log what upstream logs at `:881-888`.
int64_t Ltx2DurationPredictFrames(const Ltx2DurationHeadConfig& config,
                                  const Ltx2VaeWeights& weights, const float* video_tokens,
                                  int64_t video_token_count, const float* audio_tokens,
                                  int64_t audio_token_count, double frame_rate, double min_seconds,
                                  double max_seconds, int64_t time_scale,
                                  float* predicted_seconds,
                                  Ltx2DurationWidthCounts* widths = nullptr);

// `require_num_frames_source` (utils/blocks.py:894-905). THE POSITION IS PART OF
// THE BEHAVIOUR: upstream calls this at the very top of `__call__`, before prompt
// encoding or any other work, so a checkpoint with no head fails fast instead of
// paying for work whose result is discarded. A caller that moves it later refuses
// the same requests and is still a different pipeline.
void Ltx2RequireNumFramesSource(bool auto_requested, bool has_predictor);

// `DurationPredictor.from_checkpoint` (utils/blocks.py:816-848). Returns FALSE —
// upstream's `None` — when the file carries no head or only part of one, and
// refuses only a tensor that is PRESENT at the wrong shape. `compute_dtype`
// selects the arm — `kBF16` is upstream's own model dtype and what a production
// render asks for (A24 wave 6, #2955), `kF32` is the wider parity arm — and
// anything else refuses by name. A BF16 source is moved WORD FOR WORD onto the
// bf16 arm rather than round-tripped through f32, which is what makes
// `Ltx2VaeWeights::Bytes()` the storage claim rather than a report of it.
bool Ltx2LoadDurationHeadWeights(const SafetensorsFile& file,
                                 const Ltx2DurationHeadConfig& config, vt::DType compute_dtype,
                                 Ltx2VaeWeights* out);

}  // namespace vllm
