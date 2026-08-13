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
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// f32, the parity dtype of this gate. Upstream constructs the head in the
// pipeline's dtype (distilled.py:163-167 passes `self.dtype` = bfloat16), so the
// bf16 arm is owed by phase L6.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"  // Ltx2VaeWeights

namespace vllm {

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

// AttentionPooler.forward (duration_head.py:45-49) on its own, so a pooler defect
// localizes instead of arriving as one wrong scalar. `tokens` is
// [batch, token_count, pooler_hidden_dim]; the result is
// [batch, num_queries, pooler_hidden_dim].
std::vector<float> Ltx2DurationAttentionPool(const Ltx2DurationHeadConfig& config,
                                             const Ltx2VaeWeights& weights, const float* tokens,
                                             int64_t batch, int64_t token_count);

// DurationHead.forward (duration_head.py:89-118). Either stream may be null, and
// both being null throws exactly as upstream does (:104-105). Returns the
// predicted duration in SECONDS, [batch].
std::vector<float> Ltx2DurationPredict(const Ltx2DurationHeadConfig& config,
                                       const Ltx2VaeWeights& weights, const float* video_tokens,
                                       int64_t video_token_count, const float* audio_tokens,
                                       int64_t audio_token_count, int64_t batch);

}  // namespace vllm
