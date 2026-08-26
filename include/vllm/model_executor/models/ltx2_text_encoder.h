// LTX-2.5 TEXT CONDITIONING — the Gemma-4 multi-layer feature aggregation, the
// two caption projections, and the embedded tokenizer/asset pack.
//
// ─── THE ONE THING THAT MAKES THIS DIFFERENT FROM EVERY OTHER TEXT ENCODER ────
//
// LTX-2.5 does NOT condition on the encoder's last hidden state. It takes EVERY
// Gemma-4 hidden state — the embedding output plus all 48 decoder outputs, 49 in
// total — stacks them on a new LAST axis to [batch, seq, hidden, layers],
// normalizes, concatenates ACROSS THE LAYER AXIS, and projects the flattened
// result twice (feature_extractor.py:114-129).
//
// Measured on the shipped `vonkaiser/LTX-2.5-FP8-NVFP4` text encoder
// (`gemma4-12b-with-proj-nvfp4-torchao.safetensors`, 1688 tensors):
//
//   text_embedding_projection.video_aggregate_embed.weight  U8 [4096, 94080]
//   text_embedding_projection.audio_aggregate_embed.weight  U8 [2048, 94080]
//   model.embed_tokens.weight                               U8 [262144, 1920]
//   model.norm.weight                                     BF16 [3840]
//   model.layers.{0..47}.*                                       48 layers
//
// NVFP4 packs TWO values per byte, so those U8 widths are HALF the real feature
// counts: the projections take 188160 = 3840 x 49 inputs, not 94080, and the
// Gemma hidden size is 3840, not 1920. `model.norm.weight [3840]` is the
// independent confirmation — it is stored BF16 and therefore unpacked.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/
//   OURS                              <-  UPSTREAM
//   Ltx2StackHiddenStates             <-  text_encoders/gemma/feature_extractor.py:120
//   Ltx2NormAndConcatPaddedBatch      <-  text_encoders/gemma/feature_extractor.py:12-45
//   Ltx2NormAndConcatPerTokenRms      <-  text_encoders/gemma/feature_extractor.py:48-64
//   Ltx2RescaleNorm                   <-  text_encoders/gemma/feature_extractor.py:67-69
//   Ltx2TextFeatureExtractorForward   <-  text_encoders/gemma/feature_extractor.py:85-129
//   Ltx2SelectTextFeatureVariant      <-  text_encoders/gemma/encoders/encoder_configurator.py:163-209
//   Ltx2ConvertToAdditiveMask         <-  text_encoders/gemma/embeddings_processor.py:16-20
//   Ltx2ComputeRightPadOrder          <-  text_encoders/gemma/embeddings_processor.py:23-38
//   Ltx2ApplyRightPadOrder            <-  text_encoders/gemma/embeddings_processor.py:41-43
//   Ltx2ToBinaryMask                  <-  text_encoders/gemma/embeddings_processor.py:46-48
//   Ltx2TextEncoderConditioning       <-  text_encoders/gemma/embeddings_processor.py:70-117
//   Ltx2LoadGemmaAssets               <-  text_encoders/gemma/gemma_assets.py:104-159
//   Ltx2GemmaHiddenStateContract      <-  text_encoders/gemma/encoders/base_encoder.py:49-71
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * THE CONCATENATION IS HIDDEN-MAJOR, LAYER-MINOR. `stack(..., dim=-1)` then
//    `.reshape(B, T, D*L)` puts layer `l` of channel `d` at flat index
//    `d * L + l`. A port that concatenates layer-major (`l * D + d`) produces a
//    correctly shaped, finite, PERMUTED conditioning vector.
//  * THERE ARE TWO NORMALIZATION VARIANTS AND THEY ARE NOT INTERCHANGEABLE.
//    V1 is a per-batch, per-layer masked mean/range with an `8 *` scale and
//    eps 1e-6; V2 is a per-token RMS over the HIDDEN axis with eps 1e-6 and no
//    scale. The choice comes from config (`Ltx2SelectTextFeatureVariant`), never
//    from a guess.
//  * THE `+1` IS THE EMBEDDING LAYER. `num_layers = num_hidden_layers + 1`
//    (encoder_configurator.py:182). Dropping it makes the projection 3840 inputs
//    too narrow, which a shape check catches; taking the LAST 48 of the 49
//    instead of the first 48 plus the embedding does NOT change any shape.
//  * V1'S PROJECTION HAS NO BIAS AND V2'S HAVE ONE (encoder_configurator.py:187,
//    206-208). Because the norm zeroes padded positions, a padded position's
//    PROJECTED value is exactly the bias — not zero. A port that force-zeroes the
//    projected pads silently diverges from upstream on every padded row. The
//    DECLARED contract below (`aggregate_bias`, `*_out_features`) is therefore
//    checked against the weights the loader actually supplied — see
//    `Ltx2TextFeatureExtractorForward`.
//
// ─── AND THE EPSILONS, WHICH ARE A CLASS AND NOT ONE INSTANCE ────────────────
// A constant that only changes the answer on a DEGENERATE input is invisible to a
// golden built from random values — but NEITHER of the two below is that
// constant, and the detailed note at kLtx2TextNormV1Eps already says why for the
// V1 half. Both are additive terms on an O(1) denominator against a 1e-5 band, so
// at the 100x bar they move ORDINARY random-value goldens: V1 1e-6 -> 1e-4 REDS
// "`_norm_and_concat_padded_batch`, both padding sides" at 0.000524044 and
// "FeatureExtractorV1" at 7.53999e-05 / 6.61612e-05; V2 1e-6 -> 1e-4 REDS
// "`norm_and_concat_per_token_rms`, both padding sides" at 0.00232971 and
// carries into "FeatureExtractorV2" and the hand-off at 0.000344872 /
// 0.000259042 / 0.00039053. They are held two ways all the same: their VALUE
// against upstream measured by probe, and the degenerate input on which each is
// the only thing between the port and a division by zero — that input is what
// makes a division-by-zero visible, not what makes the constant visible at all.
// When a fourth epsilon arrives, it owes the same pair —
// not a comment saying it matches upstream.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// Everything here is f32, exactly as phase L2 (ltx2.h) records for the DiT: that
// is NOT a widening of a bf16 path, it is the PARITY dtype of this gate, which
// compares the ALGORITHM against upstream run in torch float32. Upstream resolves
// ONE model dtype and every layer inherits it — `LTXGemmaTextEncoder` takes a
// single `dtype` (base_encoder.py:41) and `FeatureExtractorV2.forward` casts the
// normalized tensor straight back to `encoded.dtype` (feature_extractor.py:122),
// so the production bf16 / FP8 / NVFP4 arms are a single stream-dtype choice —
// phase L6 — and are OWED, not shipped. Every entry point that takes a
// `compute_dtype` REFUSES anything but `vt::DType::kF32` with a message naming
// the missing phase rather than silently computing in f32.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/gemma4.h"
#include "vt/dtype.h"

namespace vllm {

class SafetensorsFile;
namespace tok {
class Tokenizer;
}  // namespace tok

// ─────────────────────────── the hidden-state contract ───────────────────────

// Which hidden states LTX consumes, and in which order.
//
// `LTXGemmaTextEncoder.encode` (base_encoder.py:68-71) calls the inner Gemma
// model with `output_hidden_states=True` and passes `outputs.hidden_states`
// straight through. In transformers that tuple has `num_hidden_layers + 1`
// entries in this exact order:
//
//   [0]              the token embeddings, AFTER the sqrt(hidden) embed scale
//   [1 .. L-1]       the output of decoder layers 0 .. L-2
//   [L]              the output of decoder layer L-1 AFTER `model.norm`
//
// The last entry is FINAL-NORMED and the raw output of the last decoder layer
// never appears. A port that appends L raw layer outputs plus the embeddings, or
// that appends the raw last layer and then the normed one, has 49 finite tensors
// of the right shape and the wrong content.
struct Ltx2GemmaHiddenStateContract {
  static constexpr const char* kOrder =
      "hidden_states[0] = embeddings * sqrt(hidden); hidden_states[i] = output of "
      "decoder layer i-1; hidden_states[num_hidden_layers] = model.norm(output of "
      "the LAST decoder layer). Count = num_hidden_layers + 1.";
  // The count the caption projections' in_features must agree with.
  static int64_t Count(int64_t num_hidden_layers) { return num_hidden_layers + 1; }
};

// A batch of Gemma hidden states, one pointer per layer, each [batch, seq, hidden]
// in row-major order. `layers.size()` must equal
// `Ltx2GemmaHiddenStateContract::Count(num_hidden_layers)`.
struct Ltx2TextHiddenStates {
  std::vector<const float*> layers;
  int64_t batch = 0;
  int64_t seq = 0;
  int64_t hidden = 0;
};

// ───────────────────────────── feature aggregation ───────────────────────────

// feature_extractor.py:28 — ONE `eps = 1e-6` bound at the top of
// `_norm_and_concat_padded_batch` and used TWICE: in the mean's denominator
// (`denom` is defined at :34, the `+ eps` is at :35) and in the range's
// (`range_ + eps`, :41). Named here so there is a single thing to pin, and pinned
// against the value MEASURED out of upstream in
// tests/vllm/models/test_ltx2_text_encoder.cpp.
//
// Reachability, stated because it is what makes the pin necessary, and stated
// CORRECTLY because an earlier revision of this comment got it wrong in the
// direction that makes a reader relax:
//
//   * `range_ + eps` (:41) is reachable only when a whole (batch, layer) slice is
//     CONSTANT over its valid positions, so `range_` collapses to 0.
//   * `denom + eps` (:35) is DIVISION-BY-ZERO-reachable only when a batch row has
//     no valid token. But it is OUTPUT-OBSERVABLE far more widely, because its
//     DTYPE is observable: `denom` is an int64 tensor and `eps` a python float, so
//     upstream adds them in FLOAT32, and an f64 add differs by one f32 ulp in the
//     mean. `range_ + eps` then amplifies that by 8/eps = 8e6 as `range_` -> 0. On
//     a constant 0.5 stack under a 3-valid-token mask, upstream reads 0.476837158
//     and an f64 denominator reads 0.238418579 — 23842x the suite's kTol.
//
// So this constant is UNOBSERVABLE at the output only on an all-pad row or an
// all-zero mask, where :44-45 zeroes every position before anything escapes. On a
// realistic Gemma workload `range_` is O(1), so the same one-ulp perturbation
// lands around 2e-12 at the output — small, but not the "any input" the earlier
// comment claimed. Both the value and the arithmetic width are gated on the
// degenerate inputs, where they are visible.
//
// That is the ONE-ULP dtype perturbation. A VALUE move of the size the pin
// exists to catch is far louder and needs no degenerate input at all: 1e-6 ->
// 1e-4 REDS "`_norm_and_concat_padded_batch`, both padding sides" at 0.000524044
// and "FeatureExtractorV1" at 7.53999e-05 / 6.61612e-05, on the ordinary
// random-value goldens.
inline constexpr double kLtx2TextNormV1Eps = 1e-6;

// feature_extractor.py:61 — `torch.rsqrt(variance + 1e-6)`. DIVISION-BY-ZERO-
// reachable only when a token's whole hidden slice is zero, which is the narrow
// claim the earlier "reachable only when" was making and the wrong one to state
// alone: the epsilon is added to an O(1) variance, so it is OUTPUT-observable on
// ordinary random values too. 1e-6 -> 1e-4 REDS
// "`norm_and_concat_per_token_rms`, both padding sides" at 0.00232971, and the
// perturbation carries through the projections into "FeatureExtractorV2" and
// "the encoder -> conditioning hand-off" at 0.000344872 / 0.000259042 /
// 0.00039053 against the suite's 1e-5 kTol. Same shape as the V1 half above.
inline constexpr float kLtx2TextNormV2Eps = 1e-6f;

// feature_extractor.py:12-64. Which of the two normalizations runs.
enum class Ltx2TextNormVariant {
  // `_norm_and_concat_padded_batch` (:12-45) — per-batch, per-layer masked mean
  // and range, `8 * (x - mean) / (range + 1e-6)`. The 19B / V1 checkpoints.
  kPaddedBatchV1,
  // `norm_and_concat_per_token_rms` (:48-64) — per-token RMS over the HIDDEN
  // axis, `x * rsqrt(mean(x^2) + 1e-6)`. Upstream's docstring: "for V2 models".
  kPerTokenRmsV2,
};

// The resolved feature-extractor shape. Produced by `Ltx2SelectTextFeatureVariant`
// from the checkpoint config; never hand-assembled on a model path.
struct Ltx2TextFeatureConfig {
  Ltx2TextNormVariant variant = Ltx2TextNormVariant::kPerTokenRmsV2;
  int64_t embedding_dim = 0;       // gemma_text_config.hidden_size (3840)
  int64_t num_layers = 0;          // num_hidden_layers + 1        (49)
  int64_t video_out_features = 0;  // video_aggregate_embed.out_features (4096)
  int64_t audio_out_features = 0;  // audio_aggregate_embed.out_features (2048); 0 = absent
  bool aggregate_bias = false;     // V1 false (:187), V2 true (:206-208)
  bool is_av = false;              // V1 only (:188): the audio arm IS the video tensor
  int64_t FlatDim() const { return embedding_dim * num_layers; }
};

// encoder_configurator.py:163-209 — the selection, mirrored including both of its
// refusals. `transformer_config` is the diffusion checkpoint's `config.transformer`
// object.
//
//   none of the four V2 marker keys present -> V1 (projection lives in the DiT)
//   all four present with their exact expected values -> V2
//   a partial set, or a drifted value -> throws std::runtime_error naming the keys
//
// Never infers the variant from tensor shapes: 3840 x 49 is the flat width under
// BOTH variants, so shapes cannot distinguish them.
Ltx2TextFeatureConfig Ltx2SelectTextFeatureVariant(
    const nlohmann::json& transformer_config, int64_t gemma_hidden_size,
    int64_t gemma_num_hidden_layers);

// feature_extractor.py:120 — `torch.stack(hidden_states, dim=-1)`.
// Output is [batch, seq, hidden, layers], layer being the LAST (fastest) axis.
std::vector<float> Ltx2StackHiddenStates(const Ltx2TextHiddenStates& states);

// feature_extractor.py:12-45. `stacked` is [B, T, D, L]; `mask` is [B, T] in
// {0, 1}. Returns [B, T, D * L] with PADDED POSITIONS ZEROED. Padding-side
// agnostic — the binary mask alone decides which positions are valid.
std::vector<float> Ltx2NormAndConcatPaddedBatch(const float* stacked,
                                                const int32_t* mask, int64_t batch,
                                                int64_t seq, int64_t hidden,
                                                int64_t layers);

// feature_extractor.py:48-64. Same shapes; per-token RMS over the hidden axis.
// Padded positions ZEROED.
std::vector<float> Ltx2NormAndConcatPerTokenRms(const float* stacked,
                                                const int32_t* mask, int64_t batch,
                                                int64_t seq, int64_t hidden,
                                                int64_t layers);

// feature_extractor.py:67-69 — `x * sqrt(target_dim / source_dim)`, computed with
// the same `math.sqrt` of a double ratio upstream uses. V2 only.
double Ltx2RescaleNorm(int64_t target_dim, int64_t source_dim);

// One caption projection: `torch.nn.Linear(flat_dim, out_features, bias=...)`.
// `weight` is row-major [out_features, in_features] — torch's own layout, so the
// checkpoint tensor is used as stored.
struct Ltx2TextAggregateEmbed {
  std::vector<float> weight;
  std::vector<float> bias;  // empty when the Linear has bias=False
  int64_t out_features = 0;
  int64_t in_features = 0;
};

// The text encoder's projection weights. V1 populates `video` only and reports
// the same tensor for audio (`is_av`); V2 populates both.
struct Ltx2TextEncoderWeights {
  Ltx2TextAggregateEmbed video;  // text_embedding_projection.video_aggregate_embed
                                 // (V1: .aggregate_embed)
  Ltx2TextAggregateEmbed audio;  // text_embedding_projection.audio_aggregate_embed
};

// The extractor output. `audio` is empty when the config has no audio projection;
// under V1's `is_av` it is a COPY of `video`, matching upstream returning the same
// tensor twice (feature_extractor.py:95-96).
struct Ltx2TextFeatures {
  std::vector<float> video;  // [batch, seq, video_out_features]
  std::vector<float> audio;  // [batch, seq, audio_out_features] or empty
};

// feature_extractor.py:85-129 — the whole extractor: stack, normalize by the
// selected variant, (V2) rescale per projection, project. `compute_dtype` must be
// vt::DType::kF32 — see the DTYPE note at the top of this header.
//
// REFUSES, by name, any disagreement between what `config` DECLARES and what
// `weights` actually carries: `aggregate_bias` vs `w.bias.empty()`,
// `*_out_features` vs `w.out_features`, and `FlatDim()` vs `w.in_features`.
// Upstream builds both Linears from the one config object
// (encoder_configurator.py:187, 206-208) and so cannot disagree with itself; a
// port that loads the config and the tensors separately can. The concrete case is
// a loader that reads `video_aggregate_embed.weight` (U8/NVFP4) and misses
// `.bias` (BF16, a different unpack path) while the config still says bias=True:
// every conditioning row is then shifted by the missing bias and every padded row
// projects to 0 rather than to the bias — finite, correctly shaped, wrong prompt.
Ltx2TextFeatures Ltx2TextFeatureExtractorForward(
    const Ltx2TextHiddenStates& states, const int32_t* mask,
    const Ltx2TextEncoderWeights& weights, const Ltx2TextFeatureConfig& config,
    vt::DType compute_dtype = vt::DType::kF32);

// ──────────────────── the encoder -> conditioning hand-off ───────────────────

// embeddings_processor.py:16-20 — `(mask - 1) * finfo(f32).max`, i.e. 0.0 for a
// kept position and -FLT_MAX for a pad. Returns [batch, 1, 1, seq] flattened.
std::vector<float> Ltx2ConvertToAdditiveMask(const int32_t* mask, int64_t batch,
                                             int64_t seq);

// embeddings_processor.py:23-38 — the STABLE descending argsort of the binary
// mask that places valid positions before pads while preserving their relative
// order. Idempotent on an already right-padded input. Fills `sort_index`
// [batch, seq] and `reordered_additive_mask` [batch, 1, 1, seq].
void Ltx2ComputeRightPadOrder(const float* additive_mask, int64_t batch,
                              int64_t seq, std::vector<int32_t>& sort_index,
                              std::vector<float>& reordered_additive_mask);

// embeddings_processor.py:41-43 — gather `features` [batch, seq, dim] along seq
// by `sort_index`.
std::vector<float> Ltx2ApplyRightPadOrder(const float* features,
                                          const int32_t* sort_index, int64_t batch,
                                          int64_t seq, int64_t dim);

// embeddings_processor.py:46-48 — `(encoded_mask < 1e-6)` as {0, 1}, [batch, seq].
//
// MEASURED, and gated as measured: BOTH masks upstream can hand this function
// satisfy the predicate everywhere. With learnable registers on — which LTX-2.5
// has — the connector returns `zeros_like(additive_mask)` (embeddings_connector.py:152)
// and 0.0 < 1e-6; with them off it returns the additive mask and -FLT_MAX < 1e-6
// too. So the mask `EmbeddingsProcessor` hands the DiT is ALL ONES. That is
// upstream's behaviour, not ours to repair.
std::vector<int32_t> Ltx2ToBinaryMask(const float* encoded_mask, int64_t batch,
                                      int64_t seq);

// The conditioning `EmbeddingsProcessor.process_hidden_states` produces, up to
// the connector call. `video`/`audio` are RIGHT-PAD ORDERED features ready for
// `Embeddings1DConnector`; `additive_mask` is the matching reordered mask.
struct Ltx2TextConditioning {
  std::vector<float> video;          // [batch, seq, video_out_features]
  std::vector<float> audio;          // [batch, seq, audio_out_features] or empty
  std::vector<float> additive_mask;  // [batch, 1, 1, seq]
  std::vector<int32_t> sort_index;   // [batch, seq]
};

// embeddings_processor.py:70-117, minus the two connector calls.
//
// WHERE THE CONNECTOR IS, corrected 2026-08-13. This note used to say
// `Embeddings1DConnector` (embeddings_connector.py:74-191) was NOT ported and
// that this function therefore stopped at its INPUT contract. Phase L5 ported it
// (`Ltx2ConnectorForward`, ltx2_connector.h), phase L9c put it on the render
// path with the checkpoint's own weights, and phase L13 runs this function's
// output through it per request. The stopping point is unchanged — this is still
// the processor MINUS the connector calls — but "the connector does not exist"
// was true only until L5, and a stale owed-note is how a later refusal came to
// cite a missing piece that had landed.
//
// ONE OVERLAP A CALLER MUST KNOW ABOUT. `Ltx2ConnectorCreateEmbeddings` is the
// OTHER port of embeddings_processor.py:23-43 and carries the right-pad sort
// too, so feeding this function's output into it sorts an already-sorted stream.
// That composes to the identity — a stable descending argsort of a 0/1 key is
// idempotent — and `Ltx2VideoEngine::Generate` asserts the precondition rather
// than assuming it.
Ltx2TextConditioning Ltx2TextEncoderConditioning(
    const Ltx2TextHiddenStates& states, const int32_t* mask,
    const Ltx2TextEncoderWeights& weights, const Ltx2TextFeatureConfig& config,
    vt::DType compute_dtype = vt::DType::kF32);

// ───────────────────── the embedded tokenizer / asset pack ───────────────────

// gemma_assets.py:58-159. The LTX-2.5 text encoder ships as ONE .safetensors file
// with its HuggingFace assets stored AS TENSORS, which is unusual enough that a
// loader assuming a sibling `tokenizer.json` fails on it:
//
//   tokenizer_json                    U8 [32169626]   ~32 MB, the whole tokenizer
//   hf_asset__tokenizer_config.json   U8 [3736]
//   hf_asset__processor_config.json   U8 [1382]
//   hf_asset__generation_config.json  U8 [255]
//   hf_asset__chat_template.jinja     U8 [18683]
//
// and the HF config itself in the file's `__metadata__` under `gemma_config`.
struct Ltx2GemmaAssets {
  std::vector<uint8_t> tokenizer_json;
  // Sidecar name (the part after `hf_asset__`) -> raw bytes.
  std::map<std::string, std::vector<uint8_t>> sidecars;
  // The parsed `__metadata__["gemma_config"]` JSON. Null when `require_config`
  // was false and the file carried no metadata.
  nlohmann::json config;
  bool has_config = false;

  // gemma_assets.py:144-151.
  const std::vector<uint8_t>& SidecarBytes(const std::string& name) const;
  nlohmann::json SidecarJson(const std::string& name) const;
};

// gemma_assets.py:34-36 — the two names the pack format is keyed on.
inline constexpr const char* kLtx2GemmaTokenizerTensor = "tokenizer_json";
inline constexpr const char* kLtx2GemmaAssetPrefix = "hf_asset__";

// gemma_assets.py:104-142 + `_require_sidecars` (:153-159). Throws
// std::runtime_error when `tokenizer_json` is missing or when either REQUIRED
// sidecar (`tokenizer_config.json`, `processor_config.json`, gemma_assets.py:38-41)
// is absent.
//
// MEASURED FINDING, reported rather than worked around: the shipped
// `vonkaiser/LTX-2.5-FP8-NVFP4` text encoder carries NO `__metadata__` block at
// all, so upstream's `GemmaAssets.from_single_file` raises on it before it reads a
// single tensor (gemma_assets.py:110-114). `require_config` mirrors that refusal
// by default; a caller that has the Gemma config from elsewhere passes false and
// gets the tensors, with `has_config` reporting which happened.
Ltx2GemmaAssets Ltx2LoadGemmaAssets(const SafetensorsFile& file,
                                    bool require_config = true);

// ─────────────────────── the prompt -> tokens hand-off ───────────────────────
//
// `LTXGemmaTokenizer.tokenize_with_weights` (tokenizer.py:31-59), which is the
// ONLY tokenization on the conditioning path. Both references agree that no chat
// template is applied here — the template belongs to the separate prompt
// ENHANCEMENT path (base_encoder.py:100, diffusers pipeline_ltx2.py:624), which
// produces a plain string that then comes back through this function.

// gemma_assets.py:162 — `TOKENIZER_MAX_LENGTH = 1024`, bound at
// base_encoder.py:231-236. Confirmed independently by diffusers, whose
// `_get_gemma_prompt_embeds` defaults `max_sequence_length: int = 1024`
// (pipeline_ltx2.py:304).
inline constexpr int64_t kLtx2GemmaTokenizerMaxLength = 1024;

// base_encoder.py:235 — `PaddingSide.LEFT`. diffusers sets the same thing
// explicitly with the comment "Gemma expects left padding for chat-style
// prompts" (pipeline_ltx2.py:328-329).
//
// This is why `Ltx2GemmaPromptTokens` reports `first_valid` rather than assuming
// position 0: the valid tokens are the TAIL, and their absolute positions start
// at the pad count, not at zero.
enum class Ltx2GemmaPaddingSide { kLeft, kRight };

// One tokenized prompt, in upstream's `[max_length]` shape.
struct Ltx2GemmaPromptTokens {
  // [max_length]. Pad positions carry `pad_id`.
  std::vector<int32_t> input_ids;
  // [max_length] in {0, 1} — the second element of upstream's (token, weight)
  // pairs (tokenizer.py:57-59), used as the attention mask at
  // base_encoder.py:65-67.
  std::vector<int32_t> attention_mask;
  // Index of the first valid token, and how many there are. On the LEFT-padded
  // default `first_valid` is the pad count and the valid run is the tail.
  int64_t first_valid = 0;
  int64_t num_valid = 0;
  // True when the prompt was longer than `max_length` and lost tokens.
  bool truncated = false;
};

// tokenizer.py:31-59, mirrored including the parts that look like details:
//
//   * `text.strip()` first (:33). diffusers strips too (pipeline_ltx2.py:333).
//   * encode, then PREPEND BOS if it is not already first — CONDITIONAL, on
//     upstream's own `if not input_ids or input_ids[0] != bos_id` guard
//     (:44-46). A port that prepends unconditionally doubles the BOS.
//
//     Two things about that, and the first one is a KNOWN DIVERGENCE rather than
//     a mirrored default. Upstream calls `self.tokenizer(text, ...)` — `__call__`
//     with its default `add_special_tokens=True` (tokenizer.py:37-43) — so
//     upstream DOES run the post_processor and we call plain `Encode`, which
//     does not. On THIS checkpoint the two are identical, because the shipped
//     `post_processor` is a TemplateProcessing whose `special_tokens` map is
//     EMPTY and whose template is the bare sequence, so it has nothing to add:
//     measured on the shipped file, not assumed. If a future checkpoint ships a
//     post_processor that DOES add something, upstream would emit it and we
//     would not — so this is the line to change, not a property to rely on.
//
//     What the two references actually disagree about is narrower than "one
//     runs the post-processor": both let it run. `ltx_core` ALSO prepends BOS
//     explicitly and says why — "Gemma 3 already emits it via post_processor;
//     Gemma 4 does not, so we prepend" (tokenizer.py:12-15) — while diffusers
//     relies on the post_processor alone (pipeline_ltx2.py:339), which for this
//     tokenizer.json adds nothing, so following diffusers would drop token 0 of
//     every prompt. `ltx_core` is the model author's own runtime and is explicit
//     about the case, so it is the one followed.
//   * EOS is never appended (:14).
//   * truncation happens BEFORE the BOS prepend and again after (:41, :46), so a
//     maximal prompt loses its LAST token to make room for BOS rather than
//     losing the BOS.
//   * pad to exactly `max_length` on `padding_side` (:48-54).
//
// Throws std::runtime_error when `bos_id` is negative — upstream raises for the
// same reason at tokenizer.py:34-36, because a conditioning path with no BOS is
// a different prompt, not a degraded one.
Ltx2GemmaPromptTokens Ltx2TokenizeGemmaPrompt(
    const tok::Tokenizer& tokenizer, const std::string& prompt, int32_t bos_id,
    int32_t pad_id, int64_t max_length = kLtx2GemmaTokenizerMaxLength,
    Ltx2GemmaPaddingSide padding_side = Ltx2GemmaPaddingSide::kLeft);

// The two ids upstream reads off the tokenizer/config rather than hardcoding.
// MEASURED on the shipped checkpoint's own `hf_asset__generation_config.json`:
// `bos_token_id` 2, `pad_token_id` 0 — and its `tokenizer_json` added_tokens
// agree (`<pad>` 0, `<eos>` 1, `<bos>` 2).
struct Ltx2GemmaSpecialIds {
  int32_t bos_id = -1;
  int32_t pad_id = -1;
};

// Resolve the two ids from the asset pack, or throw naming which one is missing.
// Reads `hf_asset__generation_config.json` first (that is where the checkpoint
// states them) and falls back to the tokenizer's own added-token table, which is
// what `LTXGemmaTokenizer` does through `tokenizer.bos_token_id`
// (tokenizer.py:35) and `tokenizer.pad_token` (:26-27).
Ltx2GemmaSpecialIds Ltx2ResolveGemmaSpecialIds(const Ltx2GemmaAssets& assets,
                                               const tok::Tokenizer& tokenizer);

// ───────────────────────────── the Gemma-4 TOWER ─────────────────────────────
//
// Phase L6 loaded the two caption projections and the asset pack and recorded
// "wiring the tower's torchao arm onto `Gemma4Weights` is owed and named as such
// rather than half-done" (ltx2_loader.h:325-329). This is that wiring.
//
// WHERE THE CONFIG COMES FROM, because this checkpoint cannot answer it.
// The shipped `vonkaiser` NVFP4 build carries NO `__metadata__` block at all, so
// upstream's own `GemmaAssets.from_single_file` raises on it before reading a
// tensor (gemma_assets.py:110-114) and there is nothing in the file to read a
// Gemma config out of. The config is therefore an INPUT, and a caller that has
// none gets a refusal naming the missing piece rather than a plausible default.
//
// The fields that no shape encodes, and that a default would get wrong:
//
//   layer_types                  which layers are full vs sliding. The shipped
//                                tower is (sliding x 5, full) x 8, and the two
//                                have DIFFERENT geometry.
//   global_head_dim              512 on the full layers against head_dim 256.
//   num_global_key_value_heads   ONE, against 8 on the sliding layers.
//   attention_k_eq_v             true — the full layers ship no `v_proj` at all.
//   rope_parameters              two rope types at two thetas, partial rotary
//                                0.25 on the full arm only.
//   rms_norm_eps, sliding_window, final_logit_softcapping
//
// Each moves every hidden state while leaving the tensor set byte-identical, so
// a wrong config resolves a DIFFERENT MODEL out of the same file and nothing
// downstream can tell. That is why this takes the config rather than inferring
// it: shapes can confirm a config, and cannot supply one.

// The tower, materialized. `weights` is bf16 throughout — the NVFP4 modules are
// dequantized on the way in, which is what `Gemma4Model::ForwardHiddenStates`
// consumes and what upstream's own resolved model dtype is
// (base_encoder.py:41). At the shipped 12B that is ~24 GB of host memory, and
// the caller is told so here rather than discovering it.
struct Ltx2GemmaTower {
  HfConfig config;
  Gemma4Weights weights;
  // Every module that arrived NVFP4-packed and was dequantized, in header order.
  // A tower whose modules are all bf16 already leaves this empty.
  std::vector<std::string> dequantized_modules;
};

// Materialize the Gemma-4 tower from an LTX text-encoder safetensors file.
//
// Reads `model.embed_tokens.*`, `model.norm.weight` and `model.layers.{i}.*` and
// IGNORES everything else the file carries — `vision_model.*`,
// `multi_modal_projector`, `audio_projector` and the two caption projections are
// all present in the shipped checkpoint and none of them is on the text path.
// Ignoring is deliberate and not laziness: choking on their presence would make
// the only shipped checkpoint unreadable.
//
// Each module is taken bf16 when it is stored bf16 and dequantized through
// `Ltx2DequantTorchaoNvfp4ToBf16` when it carries a `torchao_nvfp4` marker.
// A module in neither form throws BY NAME.
//
// REFUSES, by name, rather than loading something shaped like a tower:
//   * a `v_proj` present on a layer the config says is `attention_k_eq_v`, or
//     absent on one it does not — the two cases are 16 kv heads apart;
//   * any per-layer tensor whose width disagrees with the geometry the config
//     resolves for THAT layer;
//   * `hidden_size_per_layer_input` > 0, i.e. a PLE tower, since this checkpoint
//     family ships no `embed_tokens_per_layer` and a silently-absent PLE is a
//     different model.
Ltx2GemmaTower Ltx2LoadGemmaTowerFromSafetensors(const SafetensorsFile& file,
                                                 const nlohmann::json& gemma_config);

// ─────────────────────── prompt -> conditioning, end to end ──────────────────
//
// `LTXGemmaTextEncoder.encode` (base_encoder.py:49-71) followed by
// `EmbeddingsProcessor.process_hidden_states` (embeddings_processor.py:70-117).
//
// WHAT THIS DOES NOT DO, and why it is not a shortcut. Upstream pads every
// prompt to 1024 and runs all 1024 rows through the tower. This runs only the
// VALID tokens, at their ORIGINAL absolute positions, and treats the pad rows as
// zero. That is equivalent, not approximate: pads are masked out of attention
// and sit causally before every valid token, and the feature extractor zeroes
// their rows anyway (feature_extractor.py:63-64). Both halves of that are
// MEASURED rather than argued — upstream's own padded-vs-short f32 spread is
// f32 round-off, and our short run reproduces the padded run's valid rows, both
// gated in tests/vllm/models/test_ltx2_text_encoder.cpp. At the shipped 1024 it
// is the difference between a 12B forward over 1024 rows and over the prompt's
// own length.
struct Ltx2PromptConditioning {
  Ltx2TextConditioning conditioning;
  Ltx2GemmaPromptTokens tokens;
  // [max_length], the binary mask the extractor consumed — 1 on a valid token.
  std::vector<int32_t> mask;
  // [num_valid], the ABSOLUTE tower positions the surviving tokens ran at —
  // `first_valid .. first_valid + num_valid - 1`, because upstream counts the
  // pad rows. Exposed because it is the one part of this path no value
  // comparison can hold: renumbering it from zero is EXACTLY a no-op in real
  // arithmetic, so its whole effect on the states is a bf16 rounding difference
  // that the end-to-end conditioning cannot separate from its own noise. The
  // gate on it is therefore an integer one, in
  // `"ltx2 prompt -> conditioning: the VALUES"`, which carries the measurement.
  // #1467.
  std::vector<int32_t> positions;
  int64_t seq = 0;  // == max_length; the DiT sees the full padded width
};

// `queue` runs the tower. `weights`/`config` are the caption projections and the
// resolved V1/V2 shape, exactly as `Ltx2TextFeatureExtractorForward` takes them.
Ltx2PromptConditioning Ltx2EncodePromptToConditioning(
    const Ltx2GemmaTower& tower, const tok::Tokenizer& tokenizer,
    const Ltx2GemmaSpecialIds& ids, const Ltx2TextEncoderWeights& weights,
    const Ltx2TextFeatureConfig& feature_config, const std::string& prompt,
    vt::Queue& queue, int64_t max_length = kLtx2GemmaTokenizerMaxLength);

}  // namespace vllm
