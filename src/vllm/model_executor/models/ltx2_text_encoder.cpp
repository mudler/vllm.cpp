// LTX-2.5 TEXT CONDITIONING — see include/vllm/model_executor/models/ltx2_text_encoder.h
// for the port map, the four silent-failure traps and the DTYPE note.
//
// Ported from Lightricks/LTX-2,
// packages/ltx-core/src/ltx_core/text_encoders/gemma/{feature_extractor,
// embeddings_processor,gemma_assets}.py and encoders/encoder_configurator.py.
//
// The NORMALIZATION reductions accumulate in `double`. That is an accumulator
// width, not a memory format: every buffer this file produces, stores or returns
// is f32, so it does not widen the stream the way AGENTS.md's dtype-polarity rule
// is about. Each of those accumulators carries its own reason where it is
// written, and both reduce over the 3840-wide hidden axis or over one masked
// slice, not over the projection's 188160.
//
// The PROJECTION does not, and the reason is a mirror decision rather than a
// numeric preference (LTX25-TEXT-LINEAR-SEAM, #1208). It is `vt::MatmulBT`, the
// shared f32 GEMM seam, exactly as the LTX-2.5 DiT's own `Linear` is
// (ltx2.cpp:29-48). `F.linear` on f32 inputs accumulates in f32, and the goldens
// beside this file were produced by executing that very module in
// `torch.float32`, so f32 is what upstream does. This file previously argued the
// other way — that at 188160 wide an f64 accumulator lands closer to torch's
// BLOCKED f32 GEMM than a naive f32 one does. That is true and it is not
// decisive: closer to exact is not the same as closer to what upstream computes,
// upstream's own blocked sum carries error of the same order, and the cost of the
// f64 loop was measured at 671.8 s of ONE core per conditioning pass at the
// shipped geometry against the seam's 78.4 s, paid twice on a guided render. See
// .agents/specs/ltx25-text-linear-seam.md for both numbers and for the error each
// accumulator carries at K = 188160.
#include "vllm/model_executor/models/ltx2_text_encoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstring>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/ops.h"

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("ltx2 text encoder: " + message);
}

void RequireF32(vt::DType dtype) {
  if (dtype != vt::DType::kF32) {
    Fail(
        "compute_dtype must be f32. The bf16 / FP8 / NVFP4 arms of the text tower "
        "are phase L6 of .agents/specs/ltx-2-5.md and are NOT implemented; "
        "computing them in f32 would silently return a wider-than-checkpoint "
        "result rather than the requested arm.");
  }
}

// `torch.nn.functional.linear`: out[b, o] = sum_i x[b, i] * W[o, i] + bias[o],
// through `vt::MatmulBT` — `out[M,N] = a[M,K] @ b^T` with `b` `[N,K]` row-major,
// which is torch's own `nn.Linear` weight orientation, so the weight is handed to
// the seam exactly as the checkpoint stores it. The same seam and the same
// bias-add shape as the LTX-2.5 DiT's `Linear` (ltx2.cpp:29-48); see the file
// header for the accumulator decision this carries.
std::vector<float> Linear(vt::Queue& q, const std::vector<float>& x, int64_t rows,
                          const Ltx2TextAggregateEmbed& w) {
  if (static_cast<int64_t>(x.size()) != rows * w.in_features)
    Fail("linear: input size does not match in_features");
  if (static_cast<int64_t>(w.weight.size()) != w.out_features * w.in_features)
    Fail("linear: weight size does not match [out_features, in_features]");
  const bool has_bias = !w.bias.empty();
  if (has_bias && static_cast<int64_t>(w.bias.size()) != w.out_features)
    Fail("linear: bias size does not match out_features");

  std::vector<float> out(static_cast<size_t>(rows * w.out_features));
  // `out` is empty in both of these cases, so there is nothing to write and no
  // zero-shaped tensor to hand vt::MatmulBT.
  if (rows == 0 || w.out_features == 0) return out;

  // A zero-width reduction is not a GEMM either, but it is NOT the same as
  // writing nothing: the loop this replaced seeded its accumulator with the bias
  // and skipped the inner loop, so every output was the bias alone. Skipping only
  // the GEMM and still running the bias add below reproduces that exactly. None
  // of this is reachable through the extractor — RequireDeclaredProjection pins
  // `in_features` to `embedding_dim * num_layers` and both are refused at zero —
  // and it is written this way so the replacement is behaviour-preserving rather
  // than merely equivalent where the tests happen to look.
  if (w.in_features > 0) {
    const vt::Device dev{vt::DeviceType::kCPU, 0};
    vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()), vt::DType::kF32,
                                          dev, {rows, w.in_features});
    vt::Tensor b = vt::Tensor::Contiguous(const_cast<float*>(w.weight.data()),
                                          vt::DType::kF32, dev,
                                          {w.out_features, w.in_features});
    vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, dev,
                                          {rows, w.out_features});
    vt::MatmulBT(q, o, a, b);
  }

  if (has_bias) {
    for (int64_t r = 0; r < rows; ++r) {
      float* dst = out.data() + static_cast<size_t>(r * w.out_features);
      for (int64_t i = 0; i < w.out_features; ++i)
        dst[i] += w.bias[static_cast<size_t>(i)];
    }
  }
  return out;
}

// encoder_configurator.py:163-168 — the EXACT V2 marker set, values included.
struct V2Marker {
  const char* key;
  bool expected;
};
constexpr V2Marker kV2Markers[] = {
    {"caption_proj_before_connector", true},
    {"caption_projection_first_linear", false},
    {"caption_proj_input_norm", false},
    {"caption_projection_second_linear", false},
};

// The DECLARED contract, checked against the SUPPLIED weights.
//
// `config.aggregate_bias` and `config.*_out_features` are what the SELECTOR read
// out of the checkpoint's config (Ltx2SelectTextFeatureVariant, :130 and :166).
// `w.bias.empty()`, `w.out_features` and `w.in_features` are what the LOADER
// actually bound. Upstream builds both `nn.Linear`s from the one config object
// (encoder_configurator.py:187, 206-208) and therefore cannot disagree with
// itself; a port that resolves the config and the tensors on separate paths can,
// and nothing compared them.
//
// The concrete failure this refuses, which is the LTX-2.5 loader's most likely
// mistake: `video_aggregate_embed.weight` is U8/NVFP4 and `.bias` is BF16 — a
// different dtype on a different unpack path — so a loader can bind the weight
// and miss the bias while the config still says bias=True. Every conditioning row
// is then shifted by the missing bias, and every padded row projects to 0 instead
// of to the bias (feature_extractor.py:44-45 zeroes the NORM, not the
// projection). Finite, correctly shaped, wrong prompt.
void RequireDeclaredProjection(const char* which, const Ltx2TextAggregateEmbed& w,
                               int64_t declared_out, bool declared_bias, int64_t flat) {
  const bool has_bias = !w.bias.empty();
  if (has_bias != declared_bias)
    Fail(std::string(which) + " projection: the config declares aggregate_bias=" +
         (declared_bias ? "true" : "false") + " but the supplied weights carry " +
         (has_bias ? "a bias" : "NO bias") +
         ". A missing bias shifts every conditioning row and projects every PADDED "
         "row to 0 instead of to the bias — finite, correctly shaped, wrong prompt. "
         "The .weight is U8/NVFP4 and the .bias is BF16, so a loader can bind one "
         "and miss the other (encoder_configurator.py:187, 206-208).");
  if (w.out_features != declared_out)
    Fail(std::string(which) + " projection: the config declares out_features=" +
         std::to_string(declared_out) + " but the supplied weights are " +
         std::to_string(w.out_features) +
         " wide. The width that RUNS comes from the weights; the config's copy only "
         "feeds _rescale_norm (feature_extractor.py:121-129), so a mismatch rescales "
         "by one width and emits another.");
  if (w.in_features != flat)
    Fail(std::string(which) + " projection: the config declares in_features=" +
         std::to_string(flat) + " = embedding_dim x (num_hidden_layers + 1) but the "
         "supplied weights take " + std::to_string(w.in_features) +
         ". LTX conditions on EVERY Gemma hidden state plus the embedding output "
         "(encoder_configurator.py:182).");
}

int64_t RequireInt(const nlohmann::json& config, const char* key) {
  if (!config.contains(key) || !config.at(key).is_number_integer())
    Fail(std::string("transformer config is missing integer key '") + key + "'");
  return config.at(key).get<int64_t>();
}

}  // namespace

// ───────────────────────────── asset accessors ───────────────────────────────

const std::vector<uint8_t>& Ltx2GemmaAssets::SidecarBytes(const std::string& name) const {
  const auto it = sidecars.find(name);
  if (it == sidecars.end()) Fail("gemma assets are missing sidecar '" + name + "'");
  return it->second;
}

nlohmann::json Ltx2GemmaAssets::SidecarJson(const std::string& name) const {
  const std::vector<uint8_t>& bytes = SidecarBytes(name);
  return nlohmann::json::parse(std::string(bytes.begin(), bytes.end()));
}

// ─────────────────────────── the variant selection ───────────────────────────

Ltx2TextFeatureConfig Ltx2SelectTextFeatureVariant(const nlohmann::json& transformer_config,
                                                   int64_t gemma_hidden_size,
                                                   int64_t gemma_num_hidden_layers) {
  if (gemma_hidden_size <= 0 || gemma_num_hidden_layers <= 0)
    Fail("gemma hidden_size and num_hidden_layers must be positive");

  Ltx2TextFeatureConfig cfg;
  cfg.embedding_dim = gemma_hidden_size;
  // encoder_configurator.py:182 — "+1 for the embedding layer".
  cfg.num_layers = Ltx2GemmaHiddenStateContract::Count(gemma_num_hidden_layers);

  std::vector<std::string> present;
  std::vector<std::string> missing;
  for (const V2Marker& marker : kV2Markers) {
    if (transformer_config.contains(marker.key))
      present.emplace_back(marker.key);
    else
      missing.emplace_back(marker.key);
  }

  // encoder_configurator.py:185-188 — no marker at all means a pre-2.5 checkpoint
  // whose single projection lives in the DiT, reached through the V1 extractor.
  if (present.empty()) {
    cfg.variant = Ltx2TextNormVariant::kPaddedBatchV1;
    cfg.video_out_features = gemma_hidden_size;  // Linear(flat_dim, embedding_dim)
    cfg.audio_out_features = 0;
    cfg.aggregate_bias = false;  // encoder_configurator.py:187 — bias=False
    cfg.is_av = true;            // encoder_configurator.py:188
    return cfg;
  }

  // encoder_configurator.py:190-192 — a PARTIAL marker set is config drift.
  if (!missing.empty()) {
    std::sort(missing.begin(), missing.end());  // upstream sorts too
    std::string names;
    for (size_t i = 0; i < missing.size(); ++i) {
      if (i != 0) names += ", ";
      names += missing[i];
    }
    Fail("Partial V2 config — missing keys: " + names);
  }

  // encoder_configurator.py:194-201 — a marker present with the WRONG value.
  std::string drift;
  for (const V2Marker& marker : kV2Markers) {
    const nlohmann::json& value = transformer_config.at(marker.key);
    if (!value.is_boolean() || value.get<bool>() != marker.expected) {
      if (!drift.empty()) drift += ", ";
      drift += std::string(marker.key) + "=" + value.dump() + " (expected " +
               (marker.expected ? "True" : "False") + ")";
    }
  }
  if (!drift.empty()) Fail("Unknown config: " + drift);

  // encoder_configurator.py:203-209.
  cfg.variant = Ltx2TextNormVariant::kPerTokenRmsV2;
  cfg.video_out_features =
      RequireInt(transformer_config, "num_attention_heads") *
      RequireInt(transformer_config, "attention_head_dim");
  cfg.audio_out_features =
      RequireInt(transformer_config, "audio_num_attention_heads") *
      RequireInt(transformer_config, "audio_attention_head_dim");
  cfg.aggregate_bias = true;  // both Linears are bias=True
  cfg.is_av = false;
  return cfg;
}

// ───────────────────────────── feature aggregation ───────────────────────────

std::vector<float> Ltx2StackHiddenStates(const Ltx2TextHiddenStates& states) {
  const int64_t B = states.batch, T = states.seq, D = states.hidden;
  const int64_t L = static_cast<int64_t>(states.layers.size());
  if (B <= 0 || T <= 0 || D <= 0 || L <= 0) Fail("hidden states have a zero extent");
  // feature_extractor.py:120 — stack on a NEW LAST axis, so layer is the fastest
  // moving index and the later reshape interleaves as `d * L + l`.
  std::vector<float> out(static_cast<size_t>(B * T * D * L));
  for (int64_t l = 0; l < L; ++l) {
    const float* src = states.layers[static_cast<size_t>(l)];
    if (src == nullptr) Fail("hidden state layer pointer is null");
    for (int64_t bt = 0; bt < B * T; ++bt)
      for (int64_t d = 0; d < D; ++d)
        out[static_cast<size_t>((bt * D + d) * L + l)] =
            src[static_cast<size_t>(bt * D + d)];
  }
  return out;
}

std::vector<float> Ltx2NormAndConcatPaddedBatch(const float* stacked, const int32_t* mask,
                                                int64_t batch, int64_t seq,
                                                int64_t hidden, int64_t layers) {
  if (stacked == nullptr || mask == nullptr) Fail("null input to the V1 normalization");
  const int64_t B = batch, T = seq, D = hidden, L = layers;
  // feature_extractor.py:28 — the ONE `eps`, named in the header and pinned there
  // against the value measured out of upstream, because neither of its two uses is
  // reachable from a random fixture.
  constexpr double kEps = kLtx2TextNormV1Eps;
  const size_t count = static_cast<size_t>(B * T * D * L);
  std::vector<float> out(count);

  for (int64_t b = 0; b < B; ++b) {
    // feature_extractor.py:30 — sequence_lengths = attention_mask.sum(-1).
    int64_t seq_len = 0;
    for (int64_t t = 0; t < T; ++t)
      if (mask[static_cast<size_t>(b * T + t)] != 0) ++seq_len;
    // :34 — denom = (sequence_lengths * d), i.e. the number of VALID (t, d) pairs.
    // INT64, exactly as upstream: `attention_mask.sum(-1)` is an int64 tensor and
    // `* d` keeps it there. The width it is added to `eps` in is decided at :35.
    const int64_t denom = seq_len * D;

    for (int64_t l = 0; l < L; ++l) {
      // :33-38 — mean over the masked entries and min/max over them, both reduced
      // over BOTH the token and the hidden axis, per (batch, layer).
      //
      // f64 SUM ACCUMULATOR, and it is the deliberate escape rather than an
      // oversight: upstream's `masked.sum` is a float32 reduction, but a BLOCKED
      // one whose order no straight loop reproduces. Accumulating exactly and
      // rounding once is the closest single-rounding approximation to any order.
      // Measured against upstream on this fixture: f64 accumulate reads 2.38e-07
      // (left) / 4.77e-07 (right), a naive f32 accumulate reads 4.77e-07 / 2.38e-07
      // — it wins on one mask and loses on the other, which is noise, not a mirror.
      // The min/max stay f32 because they ARE f32 values and `hi - lo` is then the
      // same single rounding upstream's f32 subtraction does; taking the
      // difference in f64 and rounding after would double-round.
      double sum = 0.0;
      float lo = std::numeric_limits<float>::infinity();
      float hi = -std::numeric_limits<float>::infinity();
      for (int64_t t = 0; t < T; ++t) {
        if (mask[static_cast<size_t>(b * T + t)] == 0) continue;
        for (int64_t d = 0; d < D; ++d) {
          const float v = stacked[static_cast<size_t>((((b * T) + t) * D + d) * L + l)];
          sum += static_cast<double>(v);
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
      // :35 — `denom + eps` where `denom` is an int64 TENSOR and `eps` a python
      // float. Torch promotes int64-tensor + python-float to the DEFAULT dtype, so
      // that add happens in FLOAT32, not float64: upstream's denominator for
      // seq_len*D == 18 is 18.000001907348633, where an f64 add gives 18.000001.
      // Adding in f64 is numerically finer and therefore WRONG here — it is a
      // dtype that is too wide, which no token gate and no random-input golden can
      // see. It costs one f32 ulp in `mean`, which `range_ + eps` below multiplies
      // by 8/eps = 8e6 as `range_` collapses: on a constant slice it moves the
      // output from 0.476837158 to 0.238418579. Gated on that input in
      // tests/vllm/models/test_ltx2_text_encoder.cpp.
      const float mean = static_cast<float>(sum) /
                         (static_cast<float>(denom) + static_cast<float>(kEps));
      const float range = hi - lo;

      // :41 — 8 * (x - mean) / (range + eps), applied to the UNMASKED tensor.
      for (int64_t t = 0; t < T; ++t) {
        const bool valid = mask[static_cast<size_t>(b * T + t)] != 0;
        for (int64_t d = 0; d < D; ++d) {
          const size_t src = static_cast<size_t>((((b * T) + t) * D + d) * L + l);
          // :42 — reshape [B, T, D, L] -> [B, T, D * L], so `d * L + l`.
          const size_t dst = static_cast<size_t>(((b * T) + t) * D * L + d * L + l);
          // :44-45 — the padded positions are zeroed AFTER the normalization.
          out[dst] = valid ? 8.0f * (stacked[src] - mean) /
                                 (range + static_cast<float>(kEps))
                           : 0.0f;
        }
      }
    }
  }
  return out;
}

std::vector<float> Ltx2NormAndConcatPerTokenRms(const float* stacked, const int32_t* mask,
                                                int64_t batch, int64_t seq,
                                                int64_t hidden, int64_t layers) {
  if (stacked == nullptr || mask == nullptr) Fail("null input to the V2 normalization");
  const int64_t B = batch, T = seq, D = hidden, L = layers;
  constexpr float kEps = kLtx2TextNormV2Eps;  // feature_extractor.py:61
  std::vector<float> out(static_cast<size_t>(B * T * D * L));

  for (int64_t bt = 0; bt < B * T; ++bt) {
    const bool valid = mask[static_cast<size_t>(bt)] != 0;
    for (int64_t l = 0; l < L; ++l) {
      // :60 — the variance reduces over dim=2, the HIDDEN axis, per (b, t, layer).
      // The mask does NOT participate: a padded token's own values set its own
      // scale, and the result is discarded at :64.
      double sum_sq = 0.0;
      for (int64_t d = 0; d < D; ++d) {
        const double v =
            static_cast<double>(stacked[static_cast<size_t>((bt * D + d) * L + l)]);
        sum_sq += v * v;
      }
      const float variance = static_cast<float>(sum_sq / static_cast<double>(D));
      const float inv = 1.0f / std::sqrt(variance + kEps);
      for (int64_t d = 0; d < D; ++d) {
        const size_t src = static_cast<size_t>((bt * D + d) * L + l);
        const size_t dst = static_cast<size_t>(bt * D * L + d * L + l);
        out[dst] = valid ? stacked[src] * inv : 0.0f;
      }
    }
  }
  return out;
}

double Ltx2RescaleNorm(int64_t target_dim, int64_t source_dim) {
  if (source_dim <= 0) Fail("rescale: source_dim must be positive");
  // feature_extractor.py:69 — `x * math.sqrt(target_dim / source_dim)`. torch
  // converts a Python float scalar to the TENSOR's dtype before multiplying, so
  // the factor that actually multiplies an f32 activation is the f32-rounded one.
  // Returning the double here instead would diverge from upstream in the last
  // ulps of every conditioning value.
  const double exact = std::sqrt(static_cast<double>(target_dim) /
                                 static_cast<double>(source_dim));
  return static_cast<double>(static_cast<float>(exact));
}

Ltx2TextFeatures Ltx2TextFeatureExtractorForward(const Ltx2TextHiddenStates& states,
                                                 const int32_t* mask,
                                                 const Ltx2TextEncoderWeights& weights,
                                                 const Ltx2TextFeatureConfig& config,
                                                 vt::DType compute_dtype) {
  RequireF32(compute_dtype);
  if (mask == nullptr) Fail("attention mask is null");
  if (static_cast<int64_t>(states.layers.size()) != config.num_layers)
    Fail("hidden state count " + std::to_string(states.layers.size()) +
         " != num_hidden_layers + 1 = " + std::to_string(config.num_layers) +
         ". LTX conditions on EVERY Gemma hidden state plus the embedding output "
         "(base_encoder.py:68-71), not on the last one.");
  if (states.hidden != config.embedding_dim)
    Fail("hidden width does not match the configured gemma hidden_size");

  const int64_t B = states.batch, T = states.seq;
  const int64_t flat = config.FlatDim();

  // The declared contract, checked BEFORE any work — see RequireDeclaredProjection.
  // Only the projections that actually run are checked: V1 uses `video` alone and
  // returns it twice under `is_av` (feature_extractor.py:95-96), and V2's audio arm
  // exists only when the config gave it a width.
  RequireDeclaredProjection("video", weights.video, config.video_out_features,
                            config.aggregate_bias, flat);
  if (config.variant != Ltx2TextNormVariant::kPaddedBatchV1 &&
      config.audio_out_features > 0)
    RequireDeclaredProjection("audio", weights.audio, config.audio_out_features,
                              config.aggregate_bias, flat);

  const std::vector<float> stacked = Ltx2StackHiddenStates(states);

  std::vector<float> normed =
      config.variant == Ltx2TextNormVariant::kPaddedBatchV1
          ? Ltx2NormAndConcatPaddedBatch(stacked.data(), mask, B, T, states.hidden,
                                         config.num_layers)
          : Ltx2NormAndConcatPerTokenRms(stacked.data(), mask, B, T, states.hidden,
                                         config.num_layers);

  // The text tower is host-only by construction — `Ltx2TextEncoderWeights` holds
  // `std::vector<float>` — so the projection's queue is the CPU one, which is what
  // every caller of `Ltx2EncodePromptToConditioning` already builds for the tower
  // itself (ltx2_video.cpp:2085, :2799, :4479).
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  Ltx2TextFeatures features;
  if (config.variant == Ltx2TextNormVariant::kPaddedBatchV1) {
    // feature_extractor.py:93-97 — ONE projection, no rescale, and `is_av` returns
    // the same tensor twice rather than running a second projection.
    features.video = Linear(q, normed, B * T, weights.video);
    if (config.is_av) features.audio = features.video;
    return features;
  }

  // feature_extractor.py:121-129 — the rescale is applied SEPARATELY per
  // projection, each with that projection's OWN out_features over the GEMMA
  // hidden size (not over the flat width).
  auto project = [&](const Ltx2TextAggregateEmbed& w, int64_t out_features) {
    const float scale = static_cast<float>(Ltx2RescaleNorm(out_features, config.embedding_dim));
    std::vector<float> scaled(normed.size());
    for (size_t i = 0; i < normed.size(); ++i) scaled[i] = normed[i] * scale;
    return Linear(q, scaled, B * T, w);
  };
  features.video = project(weights.video, config.video_out_features);
  if (config.audio_out_features > 0)
    features.audio = project(weights.audio, config.audio_out_features);
  return features;
}

// ──────────────────── the encoder -> conditioning hand-off ───────────────────

std::vector<float> Ltx2ConvertToAdditiveMask(const int32_t* mask, int64_t batch,
                                             int64_t seq) {
  if (mask == nullptr) Fail("attention mask is null");
  // embeddings_processor.py:18-20 — (mask - 1) * finfo(f32).max: 0.0 for a kept
  // position, -FLT_MAX for a pad.
  const float big = std::numeric_limits<float>::max();
  std::vector<float> out(static_cast<size_t>(batch * seq));
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<float>(mask[i] != 0 ? 0 : -1) * big;
  return out;
}

void Ltx2ComputeRightPadOrder(const float* additive_mask, int64_t batch, int64_t seq,
                              std::vector<int32_t>& sort_index,
                              std::vector<float>& reordered_additive_mask) {
  if (additive_mask == nullptr) Fail("additive mask is null");
  const float big = std::numeric_limits<float>::max();
  sort_index.assign(static_cast<size_t>(batch * seq), 0);
  reordered_additive_mask.assign(static_cast<size_t>(batch * seq), 0.0f);

  for (int64_t b = 0; b < batch; ++b) {
    // embeddings_processor.py:34-36 — binary = (additive >= 0), then a STABLE
    // descending argsort, i.e. valid positions first in their original relative
    // order, pads after in theirs. Written as a stable partition, which is the
    // same permutation for a 0/1 key and avoids a comparator that could reorder
    // equal elements.
    int64_t w = 0;
    for (int64_t t = 0; t < seq; ++t)
      if (additive_mask[static_cast<size_t>(b * seq + t)] >= 0.0f)
        sort_index[static_cast<size_t>(b * seq + w++)] = static_cast<int32_t>(t);
    const int64_t valid = w;
    for (int64_t t = 0; t < seq; ++t)
      if (additive_mask[static_cast<size_t>(b * seq + t)] < 0.0f)
        sort_index[static_cast<size_t>(b * seq + w++)] = static_cast<int32_t>(t);
    // :37 — the reordered mask is rebuilt from the reordered BINARY values, so it
    // carries the canonical -FLT_MAX rather than whatever the input held.
    for (int64_t t = 0; t < seq; ++t)
      reordered_additive_mask[static_cast<size_t>(b * seq + t)] =
          t < valid ? 0.0f : -big;
  }
}

std::vector<float> Ltx2ApplyRightPadOrder(const float* features, const int32_t* sort_index,
                                          int64_t batch, int64_t seq, int64_t dim) {
  if (features == nullptr || sort_index == nullptr) Fail("null input to the right-pad gather");
  std::vector<float> out(static_cast<size_t>(batch * seq * dim));
  for (int64_t b = 0; b < batch; ++b)
    for (int64_t t = 0; t < seq; ++t) {
      const int64_t src = sort_index[static_cast<size_t>(b * seq + t)];
      if (src < 0 || src >= seq) Fail("right-pad sort index out of range");
      std::copy_n(features + static_cast<size_t>((b * seq + src) * dim),
                  static_cast<size_t>(dim),
                  out.begin() + static_cast<ptrdiff_t>((b * seq + t) * dim));
    }
  return out;
}

std::vector<int32_t> Ltx2ToBinaryMask(const float* encoded_mask, int64_t batch,
                                      int64_t seq) {
  if (encoded_mask == nullptr) Fail("encoded mask is null");
  // embeddings_processor.py:48 — (encoded_mask < 0.000001). See the header: BOTH
  // masks upstream can pass in satisfy this everywhere.
  std::vector<int32_t> out(static_cast<size_t>(batch * seq));
  for (size_t i = 0; i < out.size(); ++i) out[i] = encoded_mask[i] < 1e-6f ? 1 : 0;
  return out;
}

Ltx2TextConditioning Ltx2TextEncoderConditioning(const Ltx2TextHiddenStates& states,
                                                 const int32_t* mask,
                                                 const Ltx2TextEncoderWeights& weights,
                                                 const Ltx2TextFeatureConfig& config,
                                                 vt::DType compute_dtype) {
  RequireF32(compute_dtype);
  // embeddings_processor.py:114-116 — extract, convert the mask, then normalize
  // the padding layout to right-padded before the connector sees anything.
  const Ltx2TextFeatures features =
      Ltx2TextFeatureExtractorForward(states, mask, weights, config, compute_dtype);
  const std::vector<float> additive =
      Ltx2ConvertToAdditiveMask(mask, states.batch, states.seq);

  Ltx2TextConditioning out;
  // :84 — the sort index depends only on the mask, so it is computed ONCE and
  // reused for the audio arm.
  Ltx2ComputeRightPadOrder(additive.data(), states.batch, states.seq, out.sort_index,
                           out.additive_mask);
  out.video = Ltx2ApplyRightPadOrder(features.video.data(), out.sort_index.data(),
                                     states.batch, states.seq, config.video_out_features);
  if (!features.audio.empty()) {
    const int64_t audio_dim =
        config.audio_out_features > 0 ? config.audio_out_features : config.video_out_features;
    out.audio = Ltx2ApplyRightPadOrder(features.audio.data(), out.sort_index.data(),
                                       states.batch, states.seq, audio_dim);
  }
  return out;
}

// ───────────────────── the embedded tokenizer / asset pack ───────────────────

namespace {

// gemma_assets.py:302-307 — the pack stores asset bytes as a U8 tensor; Comfy may
// emit I8 for the same bytes, which reinterprets identically.
std::vector<uint8_t> AssetBytes(const StTensor& tensor, const std::string& name) {
  if (tensor.dtype != "U8" && tensor.dtype != "I8")
    Fail("asset tensor '" + name + "' has dtype " + tensor.dtype + ", expected U8 or I8");
  return std::vector<uint8_t>(tensor.data, tensor.data + tensor.nbytes);
}

// gemma_assets.py:38-41 — the two sidecars a pack MUST carry.
constexpr const char* kRequiredSidecars[] = {"tokenizer_config.json",
                                             "processor_config.json"};
// gemma_assets.py:43-47 — older / Comfy packs may store these as metadata strings.
constexpr const char* kMetadataFallbacks[] = {"tokenizer_config.json",
                                              "processor_config.json",
                                              "chat_template.jinja",
                                              "generation_config.json"};

}  // namespace

Ltx2GemmaAssets Ltx2LoadGemmaAssets(const SafetensorsFile& file, bool require_config) {
  Ltx2GemmaAssets assets;
  const std::map<std::string, std::string>& metadata = file.Metadata();

  // gemma_assets.py:108-115 — the HF config, JSON-encoded inside __metadata__.
  const auto config_it = metadata.find("gemma_config");
  if (config_it == metadata.end()) {
    if (require_config)
      Fail(
          "safetensors text encoder is missing metadata key 'gemma_config' "
          "(JSON-encoded HuggingFace config). MEASURED: the shipped "
          "vonkaiser/LTX-2.5-FP8-NVFP4 text encoder carries NO __metadata__ block "
          "at all, and upstream's GemmaAssets.from_single_file refuses it too "
          "(gemma_assets.py:110-114). Supply the Gemma config out of band and pass "
          "require_config=false.");
  } else {
    assets.config = nlohmann::json::parse(config_it->second);
    assets.has_config = true;
  }

  // :117-120 — the tokenizer, as one ~32 MB U8 tensor.
  const std::vector<std::string>& names = file.Names();
  if (std::find(names.begin(), names.end(), std::string(kLtx2GemmaTokenizerTensor)) ==
      names.end())
    Fail(std::string("safetensors text encoder is missing tensor '") +
         kLtx2GemmaTokenizerTensor +
         "'. The LTX-2.5 pack embeds the tokenizer AS A TENSOR; a loader that "
         "expects a sibling tokenizer.json file cannot read this checkpoint.");
  assets.tokenizer_json =
      AssetBytes(file.Get(kLtx2GemmaTokenizerTensor), kLtx2GemmaTokenizerTensor);

  // :122-127 — every hf_asset__<name> tensor is a sidecar file.
  const std::string prefix(kLtx2GemmaAssetPrefix);
  for (const std::string& name : names) {
    if (name.compare(0, prefix.size(), prefix) != 0) continue;
    assets.sidecars[name.substr(prefix.size())] = AssetBytes(file.Get(name), name);
  }

  // :129-132 — metadata fallback for the small JSON sidecars.
  for (const char* name : kMetadataFallbacks) {
    if (assets.sidecars.count(name) != 0) continue;
    const auto it = metadata.find(name);
    if (it == metadata.end()) continue;
    assets.sidecars[name] = std::vector<uint8_t>(it->second.begin(), it->second.end());
  }

  // :141 / :153-159 — the required sidecars, named in the failure.
  std::string missing;
  for (const char* name : kRequiredSidecars) {
    if (assets.sidecars.count(name) != 0) continue;
    if (!missing.empty()) missing += ", ";
    missing += name;
  }
  if (!missing.empty())
    Fail("safetensors text encoder is missing required sidecar(s): " + missing +
         " (embed as " + prefix + "<name>, or as metadata for small JSON)");

  return assets;
}

// ─────────────────────── the prompt -> tokens hand-off ───────────────────────

Ltx2GemmaSpecialIds Ltx2ResolveGemmaSpecialIds(const Ltx2GemmaAssets& assets,
                                               const tok::Tokenizer& tokenizer) {
  Ltx2GemmaSpecialIds ids;

  // The checkpoint STATES both, and that is the first place to look: the
  // tokenizer's own added-token table lists `<bos>` and `<pad>` by content, but
  // reading them by content would make this port depend on two strings rather
  // than on the ids the model was trained with.
  const auto gen = assets.sidecars.find("generation_config.json");
  if (gen != assets.sidecars.end()) {
    nlohmann::json doc;
    try {
      doc = nlohmann::json::parse(std::string(gen->second.begin(), gen->second.end()));
    } catch (const nlohmann::json::exception&) {
      doc = nlohmann::json::object();
    }
    if (doc.is_object()) {
      const auto bos = doc.find("bos_token_id");
      if (bos != doc.end() && bos->is_number_integer())
        ids.bos_id = bos->get<int32_t>();
      const auto pad = doc.find("pad_token_id");
      if (pad != doc.end() && pad->is_number_integer())
        ids.pad_id = pad->get<int32_t>();
    }
  }

  // Fall back to the tokenizer, which is what upstream reads
  // (tokenizer.py:35 `tokenizer.bos_token_id`, :26-27 the pad_token default).
  if (ids.bos_id < 0) ids.bos_id = tokenizer.BosId();
  if (ids.bos_id < 0) {
    for (const tok::SpecialToken& t : tokenizer.AddedTokens())
      if (t.text == "<bos>") ids.bos_id = t.id;
  }
  if (ids.pad_id < 0) {
    for (const tok::SpecialToken& t : tokenizer.AddedTokens())
      if (t.text == "<pad>") ids.pad_id = t.id;
  }
  // tokenizer.py:26-27 — when there is no pad token, the EOS one is used.
  if (ids.pad_id < 0) ids.pad_id = tokenizer.EosId();

  if (ids.bos_id < 0) {
    Fail(
        "the text encoder's asset pack declares no bos_token_id, and its tokenizer "
        "carries no <bos> added token. Upstream raises here too "
        "(tokenizer.py:34-36): every LTX conditioning prompt is tokenized with a "
        "LEADING BOS, so a prompt without one is a DIFFERENT prompt rather than a "
        "slightly degraded one.");
  }
  if (ids.pad_id < 0) {
    Fail(
        "the text encoder's asset pack declares no pad_token_id and its tokenizer "
        "has neither a <pad> added token nor an eos to fall back on "
        "(tokenizer.py:26-27), so the prompt cannot be padded to the "
        "fixed-width batch the tower is run on.");
  }
  return ids;
}

Ltx2GemmaPromptTokens Ltx2TokenizeGemmaPrompt(const tok::Tokenizer& tokenizer,
                                              const std::string& prompt,
                                              int32_t bos_id, int32_t pad_id,
                                              int64_t max_length,
                                              Ltx2GemmaPaddingSide padding_side) {
  if (bos_id < 0) {
    Fail(
        "Ltx2TokenizeGemmaPrompt requires a bos_token_id: upstream refuses the same "
        "way (tokenizer.py:34-36) because the conditioning path always tokenizes "
        "with a leading BOS.");
  }
  if (max_length <= 0) Fail("max_length must be positive");

  // tokenizer.py:33 — `text.strip()`. Python's bare `str.strip()` removes
  // whitespace from BOTH ends; mirrored on the ASCII whitespace set, which is
  // what a prompt's leading/trailing run is in practice.
  const char* kSpace = " \t\n\r\f\v";
  const size_t begin = prompt.find_first_not_of(kSpace);
  const std::string stripped =
      begin == std::string::npos
          ? std::string()
          : prompt.substr(begin, prompt.find_last_not_of(kSpace) - begin + 1);

  // NOT EncodeWithSpecialTokens — and that is a KNOWN DIVERGENCE, not a mirror.
  // Upstream calls `self.tokenizer(text, ...)` (tokenizer.py:37-43) — `__call__`
  // with its default `add_special_tokens=True` — so upstream DOES run the
  // post_processor, and plain `Encode` does not. The two are identical HERE only
  // because the shipped tokenizer's post_processor is a TemplateProcessing whose
  // `special_tokens` map is EMPTY, so it has nothing to add: MEASURED on the
  // shipped file, not assumed. If a future checkpoint ships a post_processor that
  // DOES add something, upstream would emit it and we would not — so this is the
  // line to change, not a property to rely on.
  //
  // The one thing that would NOT go wrong is a doubled BOS: the guard below
  // (`ids.front() != bos_id`) is upstream's own guard at :45, so a post_processor
  // that emitted a leading BOS would be absorbed by either call. The exposure is
  // a post_processor that adds anything ELSE. Full note at the declaration.
  std::vector<int32_t> ids = tokenizer.Encode(stripped);

  Ltx2GemmaPromptTokens out;
  // tokenizer.py:39-42 — truncate to max_length FIRST.
  out.truncated = static_cast<int64_t>(ids.size()) > max_length;
  if (out.truncated) ids.resize(static_cast<size_t>(max_length));

  // tokenizer.py:44-46 — then prepend BOS if it is not already leading, and
  // re-truncate. A maximal prompt therefore loses its LAST token, not its BOS.
  if (ids.empty() || ids.front() != bos_id) {
    ids.insert(ids.begin(), bos_id);
    if (static_cast<int64_t>(ids.size()) > max_length) {
      ids.resize(static_cast<size_t>(max_length));
      out.truncated = true;
    }
  }

  out.num_valid = static_cast<int64_t>(ids.size());
  const int64_t pad = max_length - out.num_valid;
  out.input_ids.assign(static_cast<size_t>(max_length), pad_id);
  out.attention_mask.assign(static_cast<size_t>(max_length), 0);
  // tokenizer.py:48-54 — pad to exactly max_length on the configured side.
  out.first_valid = padding_side == Ltx2GemmaPaddingSide::kLeft ? pad : 0;
  for (int64_t i = 0; i < out.num_valid; ++i) {
    out.input_ids[static_cast<size_t>(out.first_valid + i)] =
        ids[static_cast<size_t>(i)];
    out.attention_mask[static_cast<size_t>(out.first_valid + i)] = 1;
  }
  return out;
}

// ───────────────────────────── the Gemma-4 TOWER ─────────────────────────────

namespace {

// Every name this loader reads lives under `model.`. The shipped checkpoint is
// FLAT (`model.layers.0.self_attn.q_proj.weight`), not the multimodal-wrapper
// form gemma4_weights.cpp reads (`model.language_model.layers.0....`), which is
// why the tower cannot simply go through that loader.
constexpr const char* kTowerPrefix = "model.";

bool TowerHas(const SafetensorsFile& file, const std::string& name) {
  try {
    file.Get(name);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

int64_t TowerRawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number_integer()) return fallback;
  return it->get<int64_t>();
}

// One module -> bf16, whichever form it arrived in.
//
// `out`/`in` are the LOGICAL widths the config resolves, and they are passed in
// rather than read off the tensor precisely because an NVFP4 tensor's stored
// width is HALF its logical one. A loader that trusted the stored shape would
// build a tower of exactly half the right width, whose every matmul still has
// conforming dimensions among themselves.
OwnedTensor TowerModule(const SafetensorsFile& file, const std::string& module,
                        int64_t out_features, int64_t in_features, bool nk,
                        std::vector<std::string>* dequantized) {
  const std::string weight = module + ".weight";
  const StTensor& w = file.Get(weight);

  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.nk = nk;
  t.rank = 2;
  t.shape[0] = out_features;
  t.shape[1] = in_features;

  if (w.dtype == "BF16") {
    if (w.shape.size() != 2 || w.shape[0] != out_features || w.shape[1] != in_features) {
      Fail("tower module '" + module + "' is BF16 " + std::to_string(w.shape.size()) +
           "-D but the config resolves [" + std::to_string(out_features) + ", " +
           std::to_string(in_features) + "] for this layer");
    }
    t.bytes.resize(w.nbytes);
    std::memcpy(t.bytes.data(), w.data, w.nbytes);
    return t;
  }

  const std::string marker = module + std::string(kLtx2TorchaoNvfp4MarkerSuffix);
  if (w.dtype != "U8" || !TowerHas(file, marker)) {
    Fail("tower module '" + module + "' is stored as " + w.dtype +
         " with no '" + marker +
         "' sidecar, so it is neither the BF16 form nor the torchao-NVFP4 form "
         "this loader understands. Refusing rather than reading its bytes as "
         "whichever of the two happens to parse.");
  }
  // Parsed, not assumed: the marker is what says block_size 16 and swizzled
  // scales, and it REFUSES a combination the dequant would silently mis-read.
  ParseLtx2TorchaoNvfp4Marker(module, file.Get(marker));

  t.bytes.resize(static_cast<size_t>(out_features) * static_cast<size_t>(in_features) *
                 sizeof(uint16_t));
  // The producer is NOT inferred here: `ParseLtx2TorchaoNvfp4Marker` above has
  // already REFUSED anything whose marker does not say torchao, so by the time
  // we reach this line the low-nibble-first reading is established rather than
  // assumed. That is why the seam takes no default (ltx2_loader.h:323-325).
  Ltx2DequantNvfp4ToBf16(module, w, file.Get(module + ".weight_scale"),
                         file.Get(module + ".weight_scale_2"), out_features, in_features,
                         Ltx2Nvfp4Producer::kTorchao, reinterpret_cast<uint16_t*>(t.bytes.data()));
  if (dequantized != nullptr) dequantized->push_back(module);
  return t;
}

// A plain bf16 vector weight (a norm, or `layer_scalar`). Never quantized in any
// shipped build, so a non-BF16 one is a checkpoint this loader has not seen.
OwnedTensor TowerVector(const SafetensorsFile& file, const std::string& name,
                        int64_t n) {
  const StTensor& w = file.Get(name);
  if (w.dtype != "BF16")
    Fail("tower tensor '" + name + "' is " + w.dtype + ", expected BF16");
  int64_t numel = 1;
  for (int64_t d : w.shape) numel *= d;
  if (numel != n)
    Fail("tower tensor '" + name + "' has " + std::to_string(numel) +
         " elements, expected " + std::to_string(n));
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = 1;
  t.shape[0] = n;
  t.bytes.resize(w.nbytes);
  std::memcpy(t.bytes.data(), w.data, w.nbytes);
  return t;
}

// Concatenate already-materialized bf16 [rows_i, in] blocks into one raw-NK
// tensor, which is the layout `Gemma4Weights` carries q/k/v and gate/up in
// (gemma4_weights.cpp:290-296).
OwnedTensor TowerConcat(std::vector<const OwnedTensor*> parts, int64_t in_features) {
  int64_t rows = 0;
  for (const OwnedTensor* p : parts) rows += p->shape[0];
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.nk = true;
  t.rank = 2;
  t.shape[0] = rows;
  t.shape[1] = in_features;
  t.bytes.resize(static_cast<size_t>(rows) * static_cast<size_t>(in_features) *
                 sizeof(uint16_t));
  size_t at = 0;
  for (const OwnedTensor* p : parts) {
    std::memcpy(t.bytes.data() + at, p->bytes.data(), p->bytes.size());
    at += p->bytes.size();
  }
  return t;
}

}  // namespace

Ltx2GemmaTower Ltx2LoadGemmaTowerFromSafetensors(const SafetensorsFile& file,
                                                 const nlohmann::json& gemma_config) {
  Ltx2GemmaTower tower;
  tower.config = ParseHfConfig(gemma_config, "ltx2 gemma tower config");
  const HfConfig& c = tower.config;

  const nlohmann::json& text = c.raw.contains("text_config") && c.raw["text_config"].is_object()
                                   ? c.raw["text_config"]
                                   : c.raw;

  const int64_t H = c.hidden_size;
  const int64_t L = c.num_hidden_layers;
  const int64_t V = c.vocab_size;
  const int64_t I = c.intermediate_size;
  const int64_t Hq = c.num_attention_heads;
  const int64_t Dh_slide = c.head_dim;
  const int64_t Dh_full = TowerRawInt(text, "global_head_dim", Dh_slide);
  const int64_t Hkv_slide = c.num_key_value_heads;
  const int64_t Hkv_full = TowerRawInt(text, "num_global_key_value_heads", Hkv_slide);
  const int64_t ple = TowerRawInt(text, "hidden_size_per_layer_input", 0);
  const int64_t shared = TowerRawInt(text, "num_kv_shared_layers", 0);

  if (H <= 0 || L <= 0 || V <= 0 || I <= 0 || Hq <= 0)
    Fail("the gemma config resolves a degenerate tower geometry");
  if (ple > 0) {
    Fail(
        "the gemma config declares hidden_size_per_layer_input=" + std::to_string(ple) +
        ", i.e. a Per-Layer-Embeddings tower, but this checkpoint family ships no "
        "`model.embed_tokens_per_layer` and no `per_layer_model_projection`. A PLE "
        "tower whose PLE tensors are silently absent is a DIFFERENT model that "
        "still produces 49 finite hidden states.");
  }
  if (shared != 0) {
    Fail("the gemma config declares num_kv_shared_layers=" + std::to_string(shared) +
         "; the LTX text tower ships none and YOCO KV-sharing is not wired on this "
         "path. Refusing rather than reading a target layer's cache that was never "
         "written.");
  }
  if (static_cast<int64_t>(c.layer_types.size()) != L) {
    Fail("the gemma config declares " + std::to_string(c.layer_types.size()) +
         " layer_types for " + std::to_string(L) +
         " layers. Which layers are full-attention decides their head_dim (" +
         std::to_string(Dh_full) + " vs " + std::to_string(Dh_slide) +
         ") and their kv head count, so it cannot be defaulted.");
  }

  const std::string prefix = kTowerPrefix;
  tower.weights.tie_word_embeddings = true;
  tower.weights.embed_tokens =
      TowerModule(file, prefix + "embed_tokens", V, H, /*nk=*/false,
                  &tower.dequantized_modules);
  tower.weights.final_norm = TowerVector(file, prefix + "norm.weight", H);

  const bool k_eq_v_declared = text.contains("attention_k_eq_v") &&
                               text["attention_k_eq_v"].is_boolean() &&
                               text["attention_k_eq_v"].get<bool>();

  tower.weights.layers.reserve(static_cast<size_t>(L));
  for (int64_t l = 0; l < L; ++l) {
    const std::string base = prefix + "layers." + std::to_string(l) + ".";
    const std::string sa = base + "self_attn.";
    const std::string mlp = base + "mlp.";
    const bool full = c.layer_types[static_cast<size_t>(l)] == "full_attention";
    const int64_t Dh = full ? Dh_full : Dh_slide;
    const int64_t Hkv = full ? Hkv_full : Hkv_slide;

    Gemma4LayerWeights w;
    w.is_full_attention = full;
    w.is_kv_shared = false;
    w.kv_target_layer = -1;
    w.head_dim = Dh;
    w.num_kv_heads = Hkv;

    // `attention_k_eq_v` is a config-level flag but a PER-LAYER fact: the shipped
    // tower's full layers carry no `v_proj` and its sliding layers do. Resolving
    // it from tensor PRESENCE is what gemma4_weights.cpp:275 does; cross-checking
    // it against the declaration is what stops a checkpoint that dropped a
    // `v_proj` by accident from loading as a deliberately K-aliased layer, which
    // is a 16-kv-head difference that every shape check still passes.
    const bool has_v = TowerHas(file, sa + "v_proj.weight");
    w.k_eq_v = !has_v;
    if (!has_v && !k_eq_v_declared) {
      Fail("layer " + std::to_string(l) + " has no `" + sa +
           "v_proj.weight`, but the config does not declare attention_k_eq_v. "
           "Aliasing V onto K is a deliberate architecture, not a repair for a "
           "missing tensor.");
    }

    const OwnedTensor q = TowerModule(file, sa + "q_proj", Hq * Dh, H, true,
                                      &tower.dequantized_modules);
    const OwnedTensor k = TowerModule(file, sa + "k_proj", Hkv * Dh, H, true,
                                      &tower.dequantized_modules);
    if (has_v) {
      const OwnedTensor v = TowerModule(file, sa + "v_proj", Hkv * Dh, H, true,
                                        &tower.dequantized_modules);
      w.attn.qkv_proj = TowerConcat({&q, &k, &v}, H);
    } else {
      w.attn.qkv_proj = TowerConcat({&q, &k, &k}, H);
    }
    w.attn.o_proj = TowerModule(file, sa + "o_proj", H, Hq * Dh, true,
                                &tower.dequantized_modules);
    w.attn.q_norm = TowerVector(file, sa + "q_norm.weight", Dh);
    w.attn.k_norm = TowerVector(file, sa + "k_norm.weight", Dh);

    const OwnedTensor gate =
        TowerModule(file, mlp + "gate_proj", I, H, true, &tower.dequantized_modules);
    const OwnedTensor up =
        TowerModule(file, mlp + "up_proj", I, H, true, &tower.dequantized_modules);
    w.mlp.gate_up_proj = TowerConcat({&gate, &up}, H);
    w.mlp.down_proj =
        TowerModule(file, mlp + "down_proj", H, I, true, &tower.dequantized_modules);

    w.input_layernorm = TowerVector(file, base + "input_layernorm.weight", H);
    w.post_attention_layernorm =
        TowerVector(file, base + "post_attention_layernorm.weight", H);
    w.pre_feedforward_layernorm =
        TowerVector(file, base + "pre_feedforward_layernorm.weight", H);
    w.post_feedforward_layernorm =
        TowerVector(file, base + "post_feedforward_layernorm.weight", H);
    // Optional: a tower without it is upstream's `torch.ones(1)` default
    // (modeling_gemma4_unified.py:501), which is the identity.
    if (TowerHas(file, base + "layer_scalar"))
      w.layer_scalar = TowerVector(file, base + "layer_scalar", 1);

    tower.weights.layers.push_back(std::move(w));
  }
  return tower;
}

// ─────────────────────── prompt -> conditioning, end to end ──────────────────

Ltx2PromptConditioning Ltx2EncodePromptToConditioning(
    const Ltx2GemmaTower& tower, const tok::Tokenizer& tokenizer,
    const Ltx2GemmaSpecialIds& ids, const Ltx2TextEncoderWeights& weights,
    const Ltx2TextFeatureConfig& feature_config, const std::string& prompt,
    vt::Queue& queue, int64_t max_length) {
  const HfConfig& c = tower.config;
  const int64_t H = c.hidden_size;
  const int64_t L = c.num_hidden_layers;

  if (feature_config.embedding_dim != H) {
    Fail("the caption projections were resolved for a " +
         std::to_string(feature_config.embedding_dim) +
         "-wide Gemma hidden state but the tower is " + std::to_string(H) + " wide");
  }
  if (feature_config.num_layers != Ltx2GemmaHiddenStateContract::Count(L)) {
    Fail("the caption projections expect " + std::to_string(feature_config.num_layers) +
         " hidden states but the tower has " + std::to_string(L) +
         " layers, i.e. " + std::to_string(Ltx2GemmaHiddenStateContract::Count(L)) +
         " states (encoder_configurator.py:182)");
  }

  Ltx2PromptConditioning out;
  out.tokens = Ltx2TokenizeGemmaPrompt(tokenizer, prompt, ids.bos_id, ids.pad_id,
                                       max_length);
  out.seq = max_length;
  out.mask = out.tokens.attention_mask;

  const int64_t T = out.tokens.num_valid;
  // Upstream always has at least the BOS, so this cannot be zero; asserted
  // because a zero-token forward would produce empty states that every later
  // shape check would then compare against zero-length goldens.
  if (T <= 0) Fail("the tokenizer produced no valid tokens");

  std::vector<int32_t> token_ids(static_cast<size_t>(T));
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    token_ids[static_cast<size_t>(i)] =
        out.tokens.input_ids[static_cast<size_t>(out.tokens.first_valid + i)];
    // The ORIGINAL absolute position, not a renumbering from zero: transformers
    // derives positions from the cache position, which counts the pad rows, so a
    // left-padded prompt's first real token sits at `first_valid`. This mirrors
    // upstream because upstream does it, and for no stronger reason than that —
    // MEASURED against the running oracle rather than argued: rotary position
    // embedding is RELATIVE and the pads are masked out of attention, so shifting
    // every position by the same amount is a no-op. Upstream's own f32 answers
    // for positions 12..19 and 0..7 on the same 8 tokens differ by 5.11e-05 on
    // values of magnitude 14.35, i.e. 3.6e-06 relative — f32 round-off and
    // nothing else. At bf16 they differ by 1.09, but that is the rounding of
    // different absolute angles and is 0.65-1.70x the per-state dtype floor, so
    // no tolerance separates the two cleanly either. Renumbering here would
    // therefore NOT rotate a query by the wrong angle; it would simply stop
    // matching what transformers passes. Keep it matching.
    positions[static_cast<size_t>(i)] =
        static_cast<int32_t>(out.tokens.first_valid + i);
  }

  // A KV pool sized PER LAYER. The sliding and full layers do not share a
  // geometry, and the runner's single-uniform-head_dim allocation cannot express
  // that (gemma4.h, G1 HONEST STATUS) — so this path builds its own rather than
  // routing through a seam that would have to lie about one of the two.
  //
  // BF16, which is upstream's ONE resolved model dtype for this tower
  // (base_encoder.py:41 `dtype: torch.dtype = torch.bfloat16`) and the dtype the
  // attention itself runs in (gemma4.cpp, `adt = DType::kBF16`). This was f32
  // until the L10 review: nothing was wrong with the numbers — an f32 KV cache
  // is strictly WIDER, so every token gate and every golden still passed — it
  // simply held twice the bytes and made `Gemma4AttnBlock` allocate two extra
  // per-layer cast buffers and run a CastF32 on every K and V it wrote
  // (gemma4.cpp:306-315, the `kv.dtype != adt` arm — the two DBufs at :307-308
  // and the CastF32 pair at :313-314). That is exactly the defect class
  // AGENTS.md names: a token gate cannot see a dtype that is too wide.
  // At the shipped 48 layers x 1024 positions this is ~2x on a cache that peaks
  // near 200 MB for one prompt.
  const int64_t block = 16;
  const int64_t blocks = (T + block - 1) / block;
  std::vector<std::vector<uint16_t>> kv_storage;
  std::vector<PagedKvCache> attn_kv;
  kv_storage.reserve(static_cast<size_t>(L));
  for (int64_t l = 0; l < L; ++l) {
    const Gemma4LayerWeights& lw = tower.weights.layers[static_cast<size_t>(l)];
    kv_storage.emplace_back(
        static_cast<size_t>(blocks * 2 * block * lw.num_kv_heads * lw.head_dim),
        static_cast<uint16_t>(0));
    PagedKvCache kv;
    kv.data = kv_storage.back().data();
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = blocks;
    kv.block_size = block;
    kv.num_kv_heads = lw.num_kv_heads;
    kv.head_size = lw.head_dim;
    attn_kv.push_back(kv);
  }

  v1::CommonAttentionMetadata meta;
  meta.num_reqs = 1;
  meta.num_actual_tokens = static_cast<int>(T);
  meta.query_start_loc = {0, static_cast<int32_t>(T)};
  meta.query_start_loc_cpu = meta.query_start_loc;
  meta.seq_lens = {static_cast<int32_t>(T)};
  meta.seq_lens_cpu = meta.seq_lens;
  meta.max_query_len = static_cast<int>(T);
  meta.max_seq_len = static_cast<int>(T);
  meta.block_table_num_cols = static_cast<int>(blocks);
  meta.block_table_tensor.resize(static_cast<size_t>(blocks));
  for (int64_t b = 0; b < blocks; ++b)
    meta.block_table_tensor[static_cast<size_t>(b)] = static_cast<int32_t>(b);
  meta.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    meta.slot_mapping[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  meta.causal = true;

  const Gemma4HiddenStatesResult run = Gemma4Model::ForwardHiddenStates(
      token_ids, positions, meta, attn_kv, tower.weights, c, queue);
  if (static_cast<int64_t>(run.hidden_states.size()) != feature_config.num_layers) {
    Fail("the tower returned " + std::to_string(run.hidden_states.size()) +
         " hidden states, expected " + std::to_string(feature_config.num_layers));
  }

  // Scatter the valid rows back into the full padded width and leave the pad rows
  // ZERO. Their VALUE is irrelevant — the extractor zeroes every masked position
  // before the projection (feature_extractor.py:63-64) — but the DiT is fed the
  // full 1024-wide conditioning, so the rows have to exist.
  std::vector<std::vector<float>> padded(static_cast<size_t>(feature_config.num_layers));
  std::vector<const float*> layer_ptrs;
  layer_ptrs.reserve(static_cast<size_t>(feature_config.num_layers));
  for (int64_t s = 0; s < feature_config.num_layers; ++s) {
    std::vector<float>& dst = padded[static_cast<size_t>(s)];
    dst.assign(static_cast<size_t>(out.seq * H), 0.0f);
    const std::vector<float>& src = run.hidden_states[static_cast<size_t>(s)];
    std::memcpy(dst.data() + static_cast<size_t>(out.tokens.first_valid * H),
                src.data(), src.size() * sizeof(float));
    layer_ptrs.push_back(dst.data());
  }

  Ltx2TextHiddenStates states;
  states.layers = std::move(layer_ptrs);
  states.batch = 1;
  states.seq = out.seq;
  states.hidden = H;

  out.conditioning =
      Ltx2TextEncoderConditioning(states, out.mask.data(), weights, feature_config);
  return out;
}

}  // namespace vllm
