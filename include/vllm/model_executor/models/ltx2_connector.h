// LTX-2.5 EMBEDDINGS CONNECTOR — the 1-D transformer between the Gemma-4 text
// encoder and the DiT's cross-attention.
//
// Row: MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model. Spec:
// .agents/specs/ltx-2-5.md (phase L5). Issue #435.
//
// This brick sits BETWEEN two phases and was orphaned between them: L3 stops at
// the text encoder's output and L2 starts at the DiT's context input, so nothing
// owned the module that turns one into the other. It lands here.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream: Lightricks/LTX-2,
// packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py
//   OURS                          <-  UPSTREAM
//   Ltx2ConnectorForward          <-  :154-191 (Embeddings1DConnector.forward)
//   Ltx2ConnectorBlockForward     <-  :41-71  (_BasicTransformerBlock1D.forward)
//   Ltx2ConnectorReplaceRegisters <-  :139-152
//   Ltx2ConnectorConfig           <-  :95-137 + :194-256 (both configurators)
//
// ─── IT IS BUILT ON THE DiT's OWN PARTS, AND SO IS THIS PORT ─────────────────
// `_BasicTransformerBlock1D` imports the DiT's `Attention`, `FeedForward` and
// RoPE verbatim (:4-11). This TU therefore routes through `vllm::Ltx2Attention`,
// `vllm::Ltx2FeedForward`, `vllm::Ltx2PrecomputeFreqsCis` and
// `vllm::Ltx2ApplyRotaryEmb` from phase L2 rather than re-deriving them — a
// second attention implementation here would be a parallel path that could drift
// from the one the DiT is gated on.
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * `learnable_registers` IS A BFLOAT16 PARAMETER (:135-137 constructs it with
//    `dtype=torch.bfloat16`). Keeping those values in f32 is WIDER than upstream:
//    every register position then carries ~8 extra mantissa bits, the result is
//    still finite and still shaped right, and only the padded positions move.
//    AGENTS.md names this polarity exactly — a value gate cannot catch a dtype
//    that is too wide — so the rounding is explicit here and gated on its own
//    golden.
//  * THE REGISTER TABLE IS TILED, NOT INDEXED: `repeat(seq_len // num_registers, 1)`
//    (:146) walks 0..N-1, 0..N-1, …, so `seq_len` MUST be a multiple of
//    `num_learnable_registers` and upstream asserts it (:144).
//  * THE MASK IS REPLACED BY ZEROS once registers are substituted (:152). Every
//    position is attendable afterwards — including the ones that were padding. A
//    port that kept the original mask attends over fewer tokens and produces a
//    different, plausible conditioning.
//  * THE FINAL `rms_norm` (:189) has NO weight and torch's default eps of 1e-6
//    (utils.py:7-12). It is applied AFTER the last block, on top of the two the
//    block already applies.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// TWO ARMS, AND THE WEIGHTS CHOOSE. `Ltx2VaeWeights::dtype` says which, because
// upstream's module dtype is a property of the parameters and not of the call.
//
//  * `kBF16` is what the PRODUCTION path runs, and it is upstream's answer:
//    `distilled.py:109` resolves ONE pipeline dtype, `torch.bfloat16`, hands it to
//    `PromptEncoder` at `:113`, and `Embeddings1DConnector` is a `PromptEncoder`
//    submodule. Row LTX25-A24-CONNECTOR-BF16, issue #2720, A24 wave 2.
//  * `kF32` is the PARITY arm the five upstream goldens of section 10 cover. It
//    is kept, not deprecated: it is the reference the bf16 arm's own
//    discrimination gate is measured against, and deleting it would delete the
//    only stream on this fixture that is provably wider than bf16.
//
// THREE ROUNDING FACTS THE bf16 ARM CANNOT BE READ OFF THE SOURCE FOR. Each was
// measured by executing the pinned module and each has a rejected hypothesis
// recorded beside it in .agents/specs/ltx25-a24-connector-bf16.md section 4:
//
//  * `rms_norm` widens to f32, accumulates there, adds the f32 `1e-6` — NOT
//    `bf16(1e-6)`, which is A24 wave 1's answer for the TOWER's norm — and rounds
//    ONCE. The two epsilons part on 0 of 49152 values at ordinary magnitude, so
//    the arm goldens gate nothing about it and a separate small-row probe does.
//  * SPLIT rope rounds TWICE per output and INTERLEAVED rope rounds THREE times,
//    six lines apart in upstream's own file, because `addcmul_` fuses one
//    multiply into the add and a plain `a*cos + b*sin` does not.
//  * the attention accumulates in f32 and rounds ONCE, because that is exactly
//    `SDPBackend.MATH`; the kernel torch actually selects is reproducible by no
//    formula at all, so the goldens are emitted against MATH and the distance to
//    the unpatched module is reported rather than absorbed.
//
// `learnable_registers` is bf16 at BOTH widths (:135-137). On the f32 arm that
// costs an explicit `RoundToBf16`; on the bf16 arm it is the stored word and the
// rounding is an identity. Same number by two routes, and the one place the two
// arms provably agree.
//
// The FP8 and NVFP4 arms are A22 — upstream's quantization policies
// (`quantization_factory.py:22-26`), not its default dtype — and are refused by
// name in `Linear`.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vt/dtype.h"

#include "vllm/model_executor/models/ltx2.h"            // Ltx2RopeType and the DiT's parts
#include "vllm/model_executor/models/ltx2_audio_vae.h"  // Ltx2VaeWeights, the shared bag

namespace vllm {

// utils.py:7-12 — `torch.nn.functional.rms_norm`'s eps as `rms_norm` passes it.
// The connector uses the WEIGHTLESS form, so this is the only stabilizer in the
// residual path. NOT a member of the invisible-constant class, however near-zero
// the fixture's rows are: `rms_norm` adds the epsilon to the MEAN SQUARE, not to
// a row minimum, so it perturbs every row it normalizes. At the class's own 100x
// bar (1e-6 -> 1e-4) it REDS 5 of the arms in "ltx2 the Embeddings1DConnector
// reproduces upstream on every arm" — Split 0.0558581, Interleaved 0.104284,
// Float64 0.140343, NoRegisters 0.000542641, GatedNoBias 0.0892045.
// It is pinned as well as gated, in test_ltx2_pipeline.cpp, case "the constants
// the headers call pinned are actually pinned", which compares this against
// upstream's own `rms_norm` signature default rather than a retyped literal —
// the one check a regenerated golden cannot satisfy by moving with it.
inline constexpr double kLtx2ConnectorRmsNormEps = 1e-6;

// Embeddings1DConnector.__init__ defaults (:95-108), which are also both
// configurators' `transformer_config.get` fallbacks (:199-218, :227-255).
struct Ltx2ConnectorConfig {
  int64_t attention_head_dim = 128;
  int64_t num_attention_heads = 30;
  int64_t num_layers = 2;
  double positional_embedding_theta = 10000.0;
  // `[1]` upstream (:114-116). The RoPE grid is 1-D: one position axis over the
  // token index.
  std::vector<int64_t> positional_embedding_max_pos = {1};
  // 0 disables the substitution entirely, which is upstream's `None` (:103, :167).
  int64_t num_learnable_registers = 128;
  Ltx2RopeType rope_type = Ltx2RopeType::kSplit;
  bool double_precision_rope = false;
  bool apply_gated_attention = false;
  bool ff_bias = true;
  std::string prefix;

  int64_t inner_dim() const { return num_attention_heads * attention_head_dim; }
};

// The parameter contract, in `named_parameters()` order: `learnable_registers`
// first (a bare nn.Parameter precedes every submodule in torch's ordering), then
// each block's attention and feed-forward.
struct Ltx2ConnectorTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};
std::vector<Ltx2ConnectorTensorSpec> EnumerateLtx2ConnectorTensors(
    const Ltx2ConnectorConfig& config);

// The result of a connector forward: the conditioning the DiT's cross-attention
// consumes, plus the mask it must be paired with. Upstream returns both (:191)
// and a caller that keeps its ORIGINAL mask instead of this one re-masks
// positions the connector has just filled with registers.
struct Ltx2ConnectorOutput {
  std::vector<float> hidden_states;  // [batch, seq, inner_dim]
  std::vector<float> mask;           // [batch, 1, 1, seq], additive
};

// `rms_norm(x)` with no weight over the last dimension (utils.py:7-12), exposed
// for the same reason `Ltx2ConnectorReplaceRegisters` is: it is where the epsilon
// lives, and at bf16 the epsilon is INVISIBLE in the final tensor on rows of
// ordinary magnitude. `Ltx2ConnectorForward` calls this three times per block plus
// once at the end, so it is the production path and not a test hook.
//
// `compute_dtype` selects the rounding, not the arithmetic: BOTH arms accumulate
// the mean square in f64 and add the f64 epsilon, and the bf16 arm then rounds the
// result once. The f64 accumulator was MEASURED against upstream rather than
// inherited -- at the shipped width 3840 a sequential f32 accumulation mismatches
// 11000 of 15 360 000 values and this one mismatches 24.
void Ltx2ConnectorRmsNormRows(float* x, int64_t rows, int64_t width,
                              vt::DType compute_dtype = vt::DType::kF32);

// _replace_padded_with_learnable_registers (:139-152), exposed on its own because
// it is where the bf16 narrowing lives and a defect there is otherwise absorbed
// by two transformer blocks before anyone sees it.
Ltx2ConnectorOutput Ltx2ConnectorReplaceRegisters(const Ltx2ConnectorConfig& config,
                                                  const Ltx2VaeWeights& weights,
                                                  const float* hidden_states,
                                                  const float* additive_attention_mask,
                                                  int64_t batch, int64_t seq);

// Embeddings1DConnector.forward (:154-191). `additive_attention_mask` is
// [batch, 1, 1, seq] with 0 for a kept token and -finfo(f32).max for a padded
// one — the form `_prepare_attention_mask` produces (transformer_args.py:199-206)
// — and may be null only when the config carries no registers, which mirrors
// upstream's own unconditional dereference at :168-170.
Ltx2ConnectorOutput Ltx2ConnectorForward(const Ltx2ConnectorConfig& config,
                                         const Ltx2VaeWeights& weights,
                                         const float* hidden_states,
                                         const float* additive_attention_mask, int64_t batch,
                                         int64_t seq);

// ─── THE PROCESSOR AROUND THE TWO CONNECTORS ─────────────────────────────────
//
// Upstream: `EmbeddingsProcessor.create_embeddings`
// (text_encoders/gemma/embeddings_processor.py:70-95). It is a SEPARATE module
// from the connector and it does three things the connector does not, each of
// which changes the conditioning silently when it is skipped:
//
//   * IT RIGHT-PAD-SORTS THE FEATURES FIRST (:82-84, `_compute_right_pad_order` /
//     `_apply_right_pad_order`). Upstream's own comment is "Connectors expect
//     right-padded input ([valid, pad])", because the register table is indexed
//     by ABSOLUTE position (`s % num_registers`) rather than by which positions
//     were padded. A LEFT-padded batch handed straight to the connector puts
//     real tokens where registers belong and registers where tokens belong, and
//     the result is finite, correctly shaped and conditioned on nothing.
//   * IT MULTIPLIES THE VIDEO ENCODING BY A BINARY MASK (:86-87) and does NOT do
//     the same to the audio one (:91-93) — an asymmetry that reads like an
//     oversight, is upstream's behaviour, and is mirrored rather than tidied.
//   * IT RETURNS THE BINARY MASK THE DiT CONSUMES (:89), which is the connector's
//     OUTPUT mask, not the caller's input one. That mask is derived with
//     `encoded_mask < 0.000001` (:46-48), and BOTH values an additive mask can
//     hold — 0.0 and -finfo(f32).max — satisfy it, so it is all ones for every
//     input either reference can produce. See the implementation: the direction
//     is surprising, `diffusers` writes the identical comparison, and the
//     multiply is consequently an identity on every reachable path.
//
// THE TWO REFERENCES DISAGREE ABOUT WHERE THE SORT LIVES, and the disagreement is
// recorded rather than resolved by preference. `diffusers`'
// `LTX2ConnectorTransformer1d.forward` folds the sort INTO the connector
// (`src/diffusers/pipelines/ltx2/connectors.py`, the `torch.argsort(1 -
// binary_attn_mask, stable=True)` branch) and its comment claims that matches
// "the original LTX implementation" — which is true only because `ltx_core`
// sorts one level up, in the processor. The two compose to the same function;
// they differ in which module owns it. This port follows `ltx_core`, so
// `Ltx2ConnectorForward` stays a faithful port of `Embeddings1DConnector` and
// the sort lands here, at the processor.
struct Ltx2ConnectorEmbeddings {
  std::vector<float> video;  // [batch, seq, video inner_dim]
  std::vector<float> audio;  // [batch, seq, audio inner_dim]
  // `binary_mask.squeeze(-1)` (:89): [batch, seq], 1.0 for a position the DiT's
  // cross-attention may attend to and 0.0 for one it may not. With registers
  // enabled every position is attendable, which is the whole point of them.
  std::vector<float> mask;
};

// `additive_attention_mask` is [batch, seq] with 0 for a kept token and
// -finfo(f32).max for a padded one, and it is required: it is what decides which
// positions become registers.
//
// NO NUMERIC ORACLE, and that is owed rather than overlooked. `Ltx2ConnectorForward`
// — the connector this wraps — is gated on five arms against EXECUTED upstream
// (`gen-ltx2-pipeline-goldens.py` section 10). This wrapper is not. Its own tests
// are PROPERTY tests: padding-side agnosticism compares two of OUR OWN calls, so
// a defect present in both arms cancels, and the binary-mask case pins a polarity
// rather than a value. A reviewer's mutations at the layer above — video scaled
// x1.5, conditioning rows reversed — passed every assertion in the suite.
// Upstream's counterpart is `EmbeddingsProcessor.create_embeddings`
// (embeddings_processor.py:70-95), reachable from the same generator that already
// executes the connector; see the closure note on `Ltx2ConditioningTrace`
// (multimodal/ltx2_video.h).
Ltx2ConnectorEmbeddings Ltx2ConnectorCreateEmbeddings(
    const Ltx2ConnectorConfig& video_config, const Ltx2VaeWeights& video_weights,
    const float* video_features, const Ltx2ConnectorConfig& audio_config,
    const Ltx2VaeWeights& audio_weights, const float* audio_features,
    const float* additive_attention_mask, int64_t batch, int64_t seq);

}  // namespace vllm
