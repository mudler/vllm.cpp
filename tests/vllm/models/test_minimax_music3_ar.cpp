// MiniMax-Music3 — the AUTOREGRESSIVE half at REDUCED dimensions (#672, W2+W3).
//
// Every golden here was produced by EXECUTING upstream's own classes
// (scripts/gen-minimax-music3-ar-goldens.py against diffusers PR #14456 head
// c6da9936) in float32 at dimensions small enough to check in. No weight byte of
// the 28.5 GB checkpoint is present, so this gate runs in CI with no asset.
//
// The FULL-SCALE companion — the real bf16 checkpoint against the committed
// oracle goldens — is tests/parity/test_minimax_music3_ar_real.cpp. This file
// separates an ALGEBRA defect from bf16 rounding; that one proves the algebra
// survives contact with the real weights.
//
// THE TOLERANCE, and why it is what it is. The goldens are torch float32; this
// port accumulates in double and rounds once. Over the reductions here (<= 12
// terms for the MLP, <= 8 for attention) the two differ only by float32's own
// rounding of the golden, ~6e-8 relative per operation, and a 2-layer stack
// cannot compound that past ~1e-6. kRelTol is 1e-5 with a 1e-6 absolute floor —
// an order above that bound and orders BELOW any algebra defect, every one of
// which (a transposed weight, a missing softmax, an inverted interpolation
// ratio) moves values by O(1). That claim was PROVEN by mutation rather than
// asserted: softmax -> plain normalize reds 3 cases / 3 assertions, an inverted
// interpolation ratio 3 / 11, a dropped codebook offset 1 / 1, a non-causal
// attention 4 / 65, and a frame row that keeps depth position 0 reds 1 / 24.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "minimax_music3_ar_goldens.inc"
#include "vllm/model_executor/models/minimax_music3_ar.h"

namespace {

namespace m3 = vllm::models::music3;

constexpr double kRelTol = 1e-5;
constexpr double kAbsFloor = 1e-6;

// Compare and REPORT the count: a gate that cannot say how many values it
// examined has not reported. Returns the worst absolute deviation seen.
double ExpectClose(const std::vector<float>& got, const float* want, size_t count,
                   const char* what) {
  REQUIRE_MESSAGE(got.size() == count, what);
  double worst = 0.0;
  size_t bad = 0;
  size_t first_bad = 0;
  for (size_t i = 0; i < count; ++i) {
    const double a = got[i];
    const double b = want[i];
    const double diff = std::abs(a - b);
    const double bound = std::max(kAbsFloor, kRelTol * std::max(std::abs(a), std::abs(b)));
    if (!(diff <= bound)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
    worst = std::max(worst, diff);
  }
  INFO(what << ": " << count << " values compared, " << bad << " outside tolerance"
            << (bad != 0 ? ", first at index " + std::to_string(first_bad) + " got " +
                               std::to_string(got[first_bad]) + " want " +
                               std::to_string(want[first_bad])
                         : std::string()));
  CHECK(bad == 0);
  return worst;
}

std::vector<float> ToVector(const float* data, size_t count) {
  return std::vector<float>(data, data + count);
}

m3::ConditionMixConfig CondConfig() {
  m3::ConditionMixConfig config;
  config.condition_hidden_dim = vllm_test::kMusic3CondHidden;
  config.num_condition_layers = vllm_test::kMusic3CondLayers;
  config.out_dim = vllm_test::kMusic3CondOutDim;
  config.input_sampling_rate = vllm_test::kMusic3CondInputSamplingRate;
  config.input_hop_length = vllm_test::kMusic3CondInputHopLength;
  config.output_sampling_rate = vllm_test::kMusic3CondOutputSamplingRate;
  config.output_hop_length = vllm_test::kMusic3CondOutputHopLength;
  return config;
}

m3::ConditionMixWeights CondWeights() {
  m3::ConditionMixWeights weights;
  weights.layer_weight_logits =
      ToVector(vllm_test::kMusic3CondLayerWeightLogits,
               static_cast<size_t>(vllm_test::kMusic3CondLayers));
  weights.layer_scale = ToVector(vllm_test::kMusic3CondLayerScale, 1);
  weights.proj_weight = ToVector(
      vllm_test::kMusic3CondProjWeight,
      static_cast<size_t>(vllm_test::kMusic3CondOutDim * vllm_test::kMusic3CondHidden * 3));
  weights.proj_bias =
      ToVector(vllm_test::kMusic3CondProjBias, static_cast<size_t>(vllm_test::kMusic3CondOutDim));
  return weights;
}

m3::ConditionMixConfig CondDownConfig() {
  m3::ConditionMixConfig config;
  config.condition_hidden_dim = vllm_test::kMusic3CondDownHidden;
  config.num_condition_layers = vllm_test::kMusic3CondDownLayers;
  config.out_dim = vllm_test::kMusic3CondDownOutDim;
  config.input_sampling_rate = vllm_test::kMusic3CondDownInputSamplingRate;
  config.input_hop_length = vllm_test::kMusic3CondDownInputHopLength;
  config.output_sampling_rate = vllm_test::kMusic3CondDownOutputSamplingRate;
  config.output_hop_length = vllm_test::kMusic3CondDownOutputHopLength;
  return config;
}

m3::ConditionMixWeights CondDownWeights() {
  m3::ConditionMixWeights weights;
  weights.layer_weight_logits =
      ToVector(vllm_test::kMusic3CondDownLayerWeightLogits,
               static_cast<size_t>(vllm_test::kMusic3CondDownLayers));
  weights.layer_scale = ToVector(vllm_test::kMusic3CondDownLayerScale, 1);
  weights.proj_weight = ToVector(vllm_test::kMusic3CondDownProjWeight,
                                 static_cast<size_t>(vllm_test::kMusic3CondDownOutDim *
                                                     vllm_test::kMusic3CondDownHidden * 3));
  weights.proj_bias = ToVector(vllm_test::kMusic3CondDownProjBias,
                               static_cast<size_t>(vllm_test::kMusic3CondDownOutDim));
  return weights;
}

m3::DepthDecoderConfig DepthConfig() {
  m3::DepthDecoderConfig config;
  config.hidden_size = vllm_test::kMusic3DepthHidden;
  config.num_layers = vllm_test::kMusic3DepthLayers;
  config.num_attention_heads = vllm_test::kMusic3DepthHeads;
  config.intermediate_size = vllm_test::kMusic3DepthIntermediate;
  config.audio_vocab_size = vllm_test::kMusic3DepthAudioVocab;
  config.num_codebooks = vllm_test::kMusic3DepthCodebooks;
  config.max_position_embeddings = vllm_test::kMusic3DepthMaxPositions;
  return config;
}

m3::DepthDecoderWeights DepthWeights() {
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t inter = static_cast<size_t>(config.intermediate_size);
  m3::DepthDecoderWeights weights;
  weights.audio_embeddings =
      ToVector(vllm_test::kMusic3DepthAudioEmbeddingsWeight,
               static_cast<size_t>(config.audio_vocab_size * config.residual_codebooks()) * hidden);
  weights.projection = ToVector(vllm_test::kMusic3DepthProjectionWeight, hidden * hidden);
  weights.pos_embedding =
      ToVector(vllm_test::kMusic3DepthPosEmbeddingWeight,
               static_cast<size_t>(config.max_position_embeddings) * hidden);
  weights.norm = ToVector(vllm_test::kMusic3DepthNormWeight, hidden);

  const float* const input_norms[] = {vllm_test::kMusic3DepthLayers0InputLayernormWeight,
                                      vllm_test::kMusic3DepthLayers1InputLayernormWeight};
  const float* const post_norms[] = {
      vllm_test::kMusic3DepthLayers0PostAttentionLayernormWeight,
      vllm_test::kMusic3DepthLayers1PostAttentionLayernormWeight};
  const float* const to_q[] = {vllm_test::kMusic3DepthLayers0AttnToQWeight,
                               vllm_test::kMusic3DepthLayers1AttnToQWeight};
  const float* const to_k[] = {vllm_test::kMusic3DepthLayers0AttnToKWeight,
                               vllm_test::kMusic3DepthLayers1AttnToKWeight};
  const float* const to_v[] = {vllm_test::kMusic3DepthLayers0AttnToVWeight,
                               vllm_test::kMusic3DepthLayers1AttnToVWeight};
  const float* const to_out[] = {vllm_test::kMusic3DepthLayers0AttnToOutWeight,
                                 vllm_test::kMusic3DepthLayers1AttnToOutWeight};
  const float* const gate[] = {vllm_test::kMusic3DepthLayers0GateProjWeight,
                               vllm_test::kMusic3DepthLayers1GateProjWeight};
  const float* const up[] = {vllm_test::kMusic3DepthLayers0UpProjWeight,
                             vllm_test::kMusic3DepthLayers1UpProjWeight};
  const float* const down[] = {vllm_test::kMusic3DepthLayers0DownProjWeight,
                               vllm_test::kMusic3DepthLayers1DownProjWeight};
  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    m3::DepthDecoderLayerWeights entry;
    entry.input_layernorm = ToVector(input_norms[layer], hidden);
    entry.post_attention_layernorm = ToVector(post_norms[layer], hidden);
    entry.to_q = ToVector(to_q[layer], hidden * hidden);
    entry.to_k = ToVector(to_k[layer], hidden * hidden);
    entry.to_v = ToVector(to_v[layer], hidden * hidden);
    entry.to_out = ToVector(to_out[layer], hidden * hidden);
    entry.gate_proj = ToVector(gate[layer], inter * hidden);
    entry.up_proj = ToVector(up[layer], inter * hidden);
    entry.down_proj = ToVector(down[layer], hidden * inter);
    weights.layers.push_back(std::move(entry));
  }
  const float* const heads[] = {vllm_test::kMusic3DepthAudioHead0,
                                vllm_test::kMusic3DepthAudioHead1,
                                vllm_test::kMusic3DepthAudioHead2};
  for (int64_t head = 0; head < config.residual_codebooks(); ++head) {
    weights.audio_heads.push_back(
        ToVector(heads[head], static_cast<size_t>(config.audio_vocab_size) * hidden));
  }
  return weights;
}

}  // namespace

// ---------------------------------------------------------------------------
// W2 — the prompt the checkpoint contract fixes
// ---------------------------------------------------------------------------

TEST_CASE("music3 ar: the assembled prompt matches upstream string for string") {
  REQUIRE(vllm_test::kMusic3PromptGoldenCount == 2);
  int cases = 0;
  for (int64_t i = 0; i < vllm_test::kMusic3PromptGoldenCount; ++i) {
    const vllm_test::Music3PromptGolden& golden = vllm_test::kMusic3PromptGoldens[i];
    INFO("prompt golden " << golden.name);
    CHECK(m3::CleanCaption(golden.prompt) == std::string(golden.clean_caption));
    CHECK(m3::NormalizeLyrics(golden.lyrics) == std::string(golden.normalized_lyrics));
    CHECK(m3::AssembleArPrompt(golden.prompt, golden.lyrics) == std::string(golden.assembled));
    ++cases;
  }
  MESSAGE("prompt goldens checked: " << cases);
}

TEST_CASE("music3 ar: the prompt template constants are the checkpoint's, not ours") {
  // Re-emitted from the upstream module by the generator, so a rename upstream
  // reds here rather than silently generating a different song.
  CHECK(m3::kAudioEndTokenId == vllm_test::kMusic3AudioEndTokenId);
  CHECK(m3::kAudioCfgTokenId == vllm_test::kMusic3AudioCfgTokenId);
  CHECK(m3::kAudioCodeOffset == vllm_test::kMusic3AudioCodeOffset);
  CHECK(m3::kSemanticVocabSize == vllm_test::kMusic3SemanticVocabSize);
  CHECK(m3::kMaxPromptTokens == vllm_test::kMusic3MaxPromptTokens);
  CHECK(m3::kMaxAudioFrames == vllm_test::kMusic3MaxAudioFrames);
  CHECK(m3::kArCfgTopK == vllm_test::kMusic3ArCfgTopK);
  CHECK(m3::kArSamplingTopK == vllm_test::kMusic3ArSamplingTopK);
  CHECK(m3::kArCfgScale == doctest::Approx(vllm_test::kMusic3ArCfgScale));
}

TEST_CASE("music3 ar: an empty description or empty lyrics is refused") {
  CHECK_THROWS_AS(m3::AssembleArPrompt("", "[verse]\nx"), std::runtime_error);
  CHECK_THROWS_AS(m3::AssembleArPrompt("   \n\t ", "[verse]\nx"), std::runtime_error);
  CHECK_THROWS_AS(m3::AssembleArPrompt("pop", ""), std::runtime_error);
  CHECK_THROWS_AS(m3::AssembleArPrompt("pop", " \n "), std::runtime_error);
}

TEST_CASE("music3 ar: the unconditional row keeps the first and last two tokens") {
  const std::vector<int32_t> ids(
      vllm_test::kMusic3UncondIdsIn,
      vllm_test::kMusic3UncondIdsIn + vllm_test::kMusic3UncondIdsCount);
  const std::vector<int32_t> want(
      vllm_test::kMusic3UncondIdsOut,
      vllm_test::kMusic3UncondIdsOut + vllm_test::kMusic3UncondIdsCount);
  const std::vector<int32_t> got = m3::UnconditionalPromptIds(ids);
  REQUIRE(got.size() == want.size());
  int matched = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    CHECK(got[i] == want[i]);
    if (got[i] == want[i]) ++matched;
  }
  MESSAGE("unconditional ids matched: " << matched << "/" << want.size());
  // The input must not be mutated in place.
  CHECK(ids[1] == vllm_test::kMusic3UncondIdsIn[1]);
}

TEST_CASE("music3 ar: a prompt too short for the [1:-2] slice is refused") {
  CHECK_THROWS_AS(m3::UnconditionalPromptIds({1, 2, 3}), std::runtime_error);
  CHECK_NOTHROW(m3::UnconditionalPromptIds({1, 2, 3, 4}));
}

TEST_CASE("music3 ar: the frame budget is capped, floored and refused") {
  // encoders.py:287 — 25 Hz is the AR frame rate.
  CHECK(m3::MaxArFrames(1.0, 25.0) == 25);
  CHECK(m3::MaxArFrames(60.0, 25.0) == 1500);
  // The 9000-frame ceiling binds before the duration does.
  CHECK(m3::MaxArFrames(600.0, 25.0) == m3::kMaxAudioFrames);
  CHECK(m3::MaxArFrames(1e9, 25.0) == m3::kMaxAudioFrames);
  CHECK_THROWS_AS(m3::MaxArFrames(0.0, 25.0), std::runtime_error);
  CHECK_THROWS_AS(m3::MaxArFrames(-1.0, 25.0), std::runtime_error);
  // Shorter than one frame: upstream raises rather than emitting silence.
  CHECK_THROWS_AS(m3::MaxArFrames(0.01, 25.0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// W2 — the semantic stage's logit pipeline
// ---------------------------------------------------------------------------

TEST_CASE("music3 ar: the vocabulary mask leaves only the code window and the end token") {
  const std::vector<bool> blocked = m3::SemanticVocabMask(
      vllm_test::kMusic3SemanticVocab, vllm_test::kMusic3SemanticOffset,
      vllm_test::kMusic3SemanticWindow,
      static_cast<int32_t>(vllm_test::kMusic3SemanticEndId));
  REQUIRE(static_cast<int64_t>(blocked.size()) == vllm_test::kMusic3SemanticVocab);
  int allowed = 0;
  int matched = 0;
  for (size_t i = 0; i < blocked.size(); ++i) {
    CHECK(blocked[i] == vllm_test::kMusic3SemanticVocabMask[i]);
    if (blocked[i] == vllm_test::kMusic3SemanticVocabMask[i]) ++matched;
    if (!blocked[i]) ++allowed;
  }
  MESSAGE("vocab mask entries matched: " << matched << "/" << blocked.size()
                                         << ", allowed: " << allowed);
  CHECK(allowed == vllm_test::kMusic3SemanticWindow + 1);
}

TEST_CASE("music3 ar: the vocabulary mask refuses a window that does not fit") {
  CHECK_THROWS_AS(m3::SemanticVocabMask(10, 8, 6, 3), std::runtime_error);
  CHECK_THROWS_AS(m3::SemanticVocabMask(40, 8, 6, 40), std::runtime_error);
  CHECK_THROWS_AS(m3::SemanticVocabMask(0, 0, 0, 0), std::runtime_error);
  // The REAL configuration must fit: 151675 + 16384 <= 200000.
  CHECK_NOTHROW(m3::SemanticVocabMask(200000, m3::kAudioCodeOffset, m3::kSemanticVocabSize,
                                      m3::kAudioEndTokenId));
}

TEST_CASE("music3 ar: guided semantic logits match upstream's inline CFG block") {
  const size_t vocab = static_cast<size_t>(vllm_test::kMusic3SemanticVocab);
  const std::vector<float> conditional = ToVector(vllm_test::kMusic3SemanticLogitsIn, vocab);
  const std::vector<float> unconditional =
      ToVector(vllm_test::kMusic3SemanticLogitsIn + vocab, vocab);
  const std::vector<bool> blocked = m3::SemanticVocabMask(
      vllm_test::kMusic3SemanticVocab, vllm_test::kMusic3SemanticOffset,
      vllm_test::kMusic3SemanticWindow,
      static_cast<int32_t>(vllm_test::kMusic3SemanticEndId));
  const std::vector<float> got = m3::GuidedSemanticLogits(
      conditional, unconditional, blocked, vllm_test::kMusic3SemanticCfgTopK, m3::kArCfgScale);
  REQUIRE(got.size() == vocab);
  size_t finite = 0;
  size_t neg_inf = 0;
  size_t nan_count = 0;
  size_t matched = 0;
  for (size_t i = 0; i < vocab; ++i) {
    const float want = vllm_test::kMusic3SemanticGuided[i];
    if (std::isnan(got[i])) ++nan_count;
    if (std::isinf(want)) {
      ++neg_inf;
      CHECK(std::isinf(got[i]));
      CHECK(got[i] < 0.0f);
      if (std::isinf(got[i]) && got[i] < 0.0f) ++matched;
    } else {
      ++finite;
      const double bound = std::max(kAbsFloor, kRelTol * std::abs(static_cast<double>(want)));
      CHECK(std::abs(static_cast<double>(got[i]) - want) <= bound);
      if (std::abs(static_cast<double>(got[i]) - want) <= bound) ++matched;
    }
  }
  MESSAGE("guided semantic logits: " << vocab << " compared, " << matched << " matched, "
                                     << finite << " finite, " << neg_inf << " -inf, "
                                     << nan_count << " NaN");
  // The re-mask exists precisely so no position is NaN (header note).
  CHECK(nan_count == 0);
}

TEST_CASE("music3 ar: depth CFG is the plain mix with no mask and no pre-restriction") {
  const std::vector<float> cond{1.0f, -2.0f, 0.5f, 4.0f};
  const std::vector<float> uncond{0.0f, 1.0f, 0.5f, -1.0f};
  const std::vector<float> got = m3::GuidedDepthLogits(cond, uncond, m3::kArCfgScale);
  REQUIRE(got.size() == 4);
  for (size_t i = 0; i < got.size(); ++i) {
    const double want = uncond[i] + (cond[i] - uncond[i]) * m3::kArCfgScale;
    CHECK(std::abs(static_cast<double>(got[i]) - want) <= 1e-6);
  }
  MESSAGE("depth CFG values checked: " << got.size());
  CHECK_THROWS_AS(m3::GuidedDepthLogits({1.0f}, {1.0f, 2.0f}, m3::kArCfgScale),
                  std::runtime_error);
}

TEST_CASE("music3 ar: the top-k filter reproduces _sample_top_k up to the draw") {
  const size_t n = static_cast<size_t>(vllm_test::kMusic3TopKProbeN);
  const std::vector<float> logits = ToVector(vllm_test::kMusic3TopKProbeIn, n);
  // The golden INPUT carries a real NaN at index 3 and a real -inf at index 7 —
  // the generator emits them as NAN / -INFINITY rather than substituting them
  // away, because `nan_to_num` (encoders.py:95) exists for exactly those.
  REQUIRE(std::isnan(logits[3]));
  REQUIRE(std::isinf(logits[7]));
  const std::vector<float> got = m3::TopKProbabilities(logits, vllm_test::kMusic3TopKProbeK);
  const double worst = ExpectClose(got, vllm_test::kMusic3TopKProbeProbs, n, "top-k probabilities");
  double sum = 0.0;
  size_t nonzero = 0;
  for (const float p : got) {
    sum += p;
    if (p > 0.0f) ++nonzero;
  }
  MESSAGE("top-k probabilities: " << n << " values, " << nonzero
                                  << " nonzero, sum " << sum << ", worst dev " << worst);
  CHECK(std::abs(sum - 1.0) <= 1e-6);
  CHECK(nonzero == static_cast<size_t>(vllm_test::kMusic3TopKProbeK));
}

// ---------------------------------------------------------------------------
// W3 — the learned condition mix
// ---------------------------------------------------------------------------

TEST_CASE("music3 ar: the latent timeline length is the checkpoint's, at both rate polarities") {
  CHECK(m3::ConditionLatentLength(vllm_test::kMusic3CondFrames, CondConfig()) ==
        vllm_test::kMusic3CondLatentLength);
  CHECK(m3::ConditionLatentLength(vllm_test::kMusic3CondDownFrames, CondDownConfig()) ==
        vllm_test::kMusic3CondDownLatentLength);
  // The REAL configuration: 25 AR frames -> 86 latent frames (86.133 truncated),
  // which is the shape of the committed full-scale golden condition_chunk0.
  m3::ConditionMixConfig real;
  CHECK(m3::ConditionLatentLength(25, real) == 86);
  // 1 AR frame is 3 latent frames (1 * 1.8375 * 1.875 = 3.445, truncated), NOT
  // 1: at the real polarity the latent timeline is always the LONGER one.
  CHECK(m3::ConditionLatentLength(1, real) == 3);
  // max(1, ...) is not decoration: one frame at the DOWN polarity truncates to 0.
  CHECK(m3::ConditionLatentLength(1, CondDownConfig()) == 1);
  CHECK_THROWS_AS(m3::ConditionLatentLength(0, real), std::runtime_error);
}

TEST_CASE("music3 ar: nearest interpolation upsamples and downsamples on input/output scale") {
  // in_len 3 -> out_len 7: floor(t * 3/7) = 0,0,0,1,1,2,2.
  const std::vector<float> in{1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
  const std::vector<float> up = m3::NearestInterpolate1d(in, 2, 3, 7);
  const std::vector<float> want_up{1, 1, 1, 2, 2, 3, 3, 10, 10, 10, 20, 20, 30, 30};
  REQUIRE(up.size() == want_up.size());
  int matched = 0;
  for (size_t i = 0; i < up.size(); ++i) {
    CHECK(up[i] == want_up[i]);
    if (up[i] == want_up[i]) ++matched;
  }
  // in_len 7 -> out_len 2: floor(t * 7/2) = 0, 3.
  const std::vector<float> wide{0, 1, 2, 3, 4, 5, 6};
  const std::vector<float> down = m3::NearestInterpolate1d(wide, 1, 7, 2);
  REQUIRE(down.size() == 2);
  CHECK(down[0] == 0.0f);
  CHECK(down[1] == 3.0f);
  MESSAGE("nearest interpolation values matched: " << matched << "/" << want_up.size()
                                                   << " up, 2/2 down");
  CHECK_THROWS_AS(m3::NearestInterpolate1d(in, 2, 4, 7), std::runtime_error);
}

TEST_CASE("music3 ar: the layer mix weights are a softmax") {
  const std::vector<float> weights = m3::ConditionLayerWeights(
      ToVector(vllm_test::kMusic3CondLayerWeightLogits,
               static_cast<size_t>(vllm_test::kMusic3CondLayers)));
  REQUIRE(static_cast<int64_t>(weights.size()) == vllm_test::kMusic3CondLayers);
  double sum = 0.0;
  for (const float w : weights) {
    CHECK(w > 0.0f);
    sum += w;
  }
  MESSAGE("layer mix weights: " << weights.size() << " entries summing to " << sum);
  CHECK(std::abs(sum - 1.0) <= 1e-6);
  // A softmax is shift-invariant; a plain normalize is not. This separates them.
  std::vector<float> shifted = ToVector(vllm_test::kMusic3CondLayerWeightLogits,
                                        static_cast<size_t>(vllm_test::kMusic3CondLayers));
  for (float& value : shifted) value += 3.5f;
  const std::vector<float> shifted_weights = m3::ConditionLayerWeights(shifted);
  for (size_t i = 0; i < weights.size(); ++i) {
    CHECK(std::abs(static_cast<double>(shifted_weights[i]) - weights[i]) <= 1e-6);
  }
}

TEST_CASE("music3 ar: the condition mix matches upstream (upsampling polarity)") {
  const m3::ConditionMixConfig config = CondConfig();
  const size_t width = static_cast<size_t>(config.num_condition_layers * config.condition_hidden_dim);
  const std::vector<float> hidden =
      ToVector(vllm_test::kMusic3CondHiddenIn,
               static_cast<size_t>(vllm_test::kMusic3CondFrames) * width);
  const std::vector<float> got =
      m3::ConditionMix(hidden, vllm_test::kMusic3CondFrames, config, CondWeights());
  const size_t count =
      static_cast<size_t>(vllm_test::kMusic3CondLatentLength * vllm_test::kMusic3CondOutDim);
  const double worst = ExpectClose(got, vllm_test::kMusic3CondOut, count, "condition mix (up)");
  MESSAGE("condition mix (up): " << count << " values, worst deviation " << worst);
}

TEST_CASE("music3 ar: the condition mix matches upstream (downsampling polarity)") {
  const m3::ConditionMixConfig config = CondDownConfig();
  const size_t width =
      static_cast<size_t>(config.num_condition_layers * config.condition_hidden_dim);
  const std::vector<float> hidden =
      ToVector(vllm_test::kMusic3CondDownHiddenIn,
               static_cast<size_t>(vllm_test::kMusic3CondDownFrames) * width);
  const std::vector<float> got =
      m3::ConditionMix(hidden, vllm_test::kMusic3CondDownFrames, config, CondDownWeights());
  const size_t count = static_cast<size_t>(vllm_test::kMusic3CondDownLatentLength *
                                           vllm_test::kMusic3CondDownOutDim);
  const double worst = ExpectClose(got, vllm_test::kMusic3CondDownOut, count, "condition mix (down)");
  MESSAGE("condition mix (down): " << count << " values, worst deviation " << worst);
}

TEST_CASE("music3 ar: the condition mix refuses every wrong-shaped input by name") {
  const m3::ConditionMixConfig config = CondConfig();
  const size_t width =
      static_cast<size_t>(config.num_condition_layers * config.condition_hidden_dim);
  const std::vector<float> hidden =
      ToVector(vllm_test::kMusic3CondHiddenIn,
               static_cast<size_t>(vllm_test::kMusic3CondFrames) * width);
  CHECK_THROWS_AS(m3::ConditionMix(hidden, vllm_test::kMusic3CondFrames + 1, config, CondWeights()),
                  std::runtime_error);
  m3::ConditionMixWeights broken = CondWeights();
  broken.layer_weight_logits.pop_back();
  CHECK_THROWS_AS(m3::ConditionMix(hidden, vllm_test::kMusic3CondFrames, config, broken),
                  std::runtime_error);
  broken = CondWeights();
  broken.layer_scale.push_back(1.0f);
  CHECK_THROWS_AS(m3::ConditionMix(hidden, vllm_test::kMusic3CondFrames, config, broken),
                  std::runtime_error);
  broken = CondWeights();
  broken.proj_weight.pop_back();
  CHECK_THROWS_AS(m3::ConditionMix(hidden, vllm_test::kMusic3CondFrames, config, broken),
                  std::runtime_error);
  broken = CondWeights();
  broken.proj_bias.pop_back();
  CHECK_THROWS_AS(m3::ConditionMix(hidden, vllm_test::kMusic3CondFrames, config, broken),
                  std::runtime_error);
}

// ---------------------------------------------------------------------------
// W3 — the RVQ depth decoder
// ---------------------------------------------------------------------------

TEST_CASE("music3 ar: the depth sequence is assembled as _generate_depth_codes does") {
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const std::vector<int32_t> codes(
      vllm_test::kMusic3DepthResidualCodes,
      vllm_test::kMusic3DepthResidualCodes + (vllm_test::kMusic3DepthSeqLen - 2));
  const std::vector<float> got = m3::DepthSequenceEmbeds(
      ToVector(vllm_test::kMusic3DepthLastHidden, hidden),
      ToVector(vllm_test::kMusic3DepthSemanticEmbed, hidden), codes, config, DepthWeights());
  const size_t count = static_cast<size_t>(vllm_test::kMusic3DepthSeqLen) * hidden;
  const double worst =
      ExpectClose(got, vllm_test::kMusic3DepthInputsEmbeds, count, "depth sequence embeds");
  MESSAGE("depth sequence: " << vllm_test::kMusic3DepthSeqLen << " positions, " << count
                             << " values, worst deviation " << worst);
}

TEST_CASE("music3 ar: a residual code outside the audio vocabulary is refused") {
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const std::vector<float> last = ToVector(vllm_test::kMusic3DepthLastHidden, hidden);
  const std::vector<float> semantic = ToVector(vllm_test::kMusic3DepthSemanticEmbed, hidden);
  const m3::DepthDecoderWeights weights = DepthWeights();
  CHECK_THROWS_AS(
      m3::DepthSequenceEmbeds(last, semantic,
                              {static_cast<int32_t>(config.audio_vocab_size)}, config, weights),
      std::runtime_error);
  CHECK_THROWS_AS(m3::DepthSequenceEmbeds(last, semantic, {-1}, config, weights),
                  std::runtime_error);
  // The last codebook is predicted, never fed back, so the sequence can carry at
  // most residual_codebooks() - 1 codes.
  CHECK_THROWS_AS(m3::DepthSequenceEmbeds(last, semantic, {0, 0, 0}, config, weights),
                  std::runtime_error);
  CHECK_THROWS_AS(m3::DepthSequenceEmbeds({1.0f}, semantic, {0}, config, weights),
                  std::runtime_error);
}

TEST_CASE("music3 ar: the depth decoder forward matches upstream") {
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t count = static_cast<size_t>(vllm_test::kMusic3DepthSeqLen) * hidden;
  const std::vector<float> got = m3::DepthDecoderForward(
      ToVector(vllm_test::kMusic3DepthInputsEmbeds, count), vllm_test::kMusic3DepthSeqLen, config,
      DepthWeights());
  const double worst = ExpectClose(got, vllm_test::kMusic3DepthOut, count, "depth decoder forward");
  MESSAGE("depth decoder: " << vllm_test::kMusic3DepthLayers << " layers, "
                            << vllm_test::kMusic3DepthSeqLen << " positions, " << count
                            << " values, worst deviation " << worst);
}

TEST_CASE("music3 ar: the depth decoder is CAUSAL, so a truncated prefix is unchanged") {
  // The whole reason ONE forward over the depth sequence may stand in for
  // upstream's incremental schedule. Break causality and the prefix moves.
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t full_count = static_cast<size_t>(vllm_test::kMusic3DepthSeqLen) * hidden;
  const std::vector<float> full_in = ToVector(vllm_test::kMusic3DepthInputsEmbeds, full_count);
  const std::vector<float> full = m3::DepthDecoderForward(
      full_in, vllm_test::kMusic3DepthSeqLen, config, DepthWeights());
  int compared = 0;
  for (int64_t prefix = 1; prefix < vllm_test::kMusic3DepthSeqLen; ++prefix) {
    const std::vector<float> shortened(
        full_in.begin(), full_in.begin() + static_cast<int64_t>(hidden) * prefix);
    const std::vector<float> got =
        m3::DepthDecoderForward(shortened, prefix, config, DepthWeights());
    REQUIRE(got.size() == hidden * static_cast<size_t>(prefix));
    for (size_t i = 0; i < got.size(); ++i) {
      const double bound =
          std::max(kAbsFloor, kRelTol * std::max(std::abs(static_cast<double>(got[i])),
                                                 std::abs(static_cast<double>(full[i]))));
      CHECK(std::abs(static_cast<double>(got[i]) - full[i]) <= bound);
      ++compared;
    }
  }
  MESSAGE("causal prefix values compared: " << compared);
}

TEST_CASE("music3 ar: the position window binds at its boundary and one past it") {
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t count = static_cast<size_t>(config.max_position_embeddings) * hidden;
  const std::vector<float> got =
      m3::DepthDecoderForward(ToVector(vllm_test::kMusic3DepthBoundaryIn, count),
                              config.max_position_embeddings, config, DepthWeights());
  const double worst =
      ExpectClose(got, vllm_test::kMusic3DepthBoundaryOut, count, "depth decoder at boundary");
  MESSAGE("depth decoder boundary: seq_len == max_position_embeddings == "
          << config.max_position_embeddings << ", " << count << " values, worst deviation "
          << worst);
  // One past the window has no pos_embedding row; upstream would index out of
  // bounds, so this port refuses by name.
  std::vector<float> over(count + hidden, 0.0f);
  CHECK_THROWS_AS(
      m3::DepthDecoderForward(over, config.max_position_embeddings + 1, config, DepthWeights()),
      std::runtime_error);
  CHECK_THROWS_AS(m3::DepthDecoderForward({}, 0, config, DepthWeights()), std::runtime_error);
  // The REAL configuration's depth sequence (8) fits inside its window (16).
  m3::DepthDecoderConfig real;
  CHECK(real.num_codebooks <= real.max_position_embeddings);
}

TEST_CASE("music3 ar: the audio heads match upstream, one per residual codebook") {
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t count = static_cast<size_t>(vllm_test::kMusic3DepthSeqLen) * hidden;
  const m3::DepthDecoderWeights weights = DepthWeights();
  const std::vector<float> hidden_states = m3::DepthDecoderForward(
      ToVector(vllm_test::kMusic3DepthInputsEmbeds, count), vllm_test::kMusic3DepthSeqLen, config,
      weights);
  const int64_t heads = vllm_test::kMusic3DepthSeqLen - 1;
  int values = 0;
  for (int64_t head = 0; head < heads; ++head) {
    // encoders.py:131-133 — depth step i reads position i and head i-1.
    const std::vector<float> state(
        hidden_states.begin() + static_cast<int64_t>(hidden) * (head + 1),
        hidden_states.begin() + static_cast<int64_t>(hidden) * (head + 2));
    const std::vector<float> got = m3::AudioHeadLogits(state, head, config, weights);
    REQUIRE(static_cast<int64_t>(got.size()) == config.audio_vocab_size);
    for (int64_t j = 0; j < config.audio_vocab_size; ++j) {
      const float want =
          vllm_test::kMusic3DepthHeadLogits[head * config.audio_vocab_size + j];
      const double bound =
          std::max(kAbsFloor, kRelTol * std::max(std::abs(static_cast<double>(got[j])),
                                                 std::abs(static_cast<double>(want))));
      CHECK(std::abs(static_cast<double>(got[j]) - want) <= bound);
      ++values;
    }
  }
  MESSAGE("audio head logits compared: " << values << " across " << heads << " heads");
  CHECK_THROWS_AS(m3::AudioHeadLogits(std::vector<float>(hidden, 0.0f), heads + 5, config, weights),
                  std::runtime_error);
  CHECK_THROWS_AS(m3::AudioHeadLogits(std::vector<float>(hidden, 0.0f), -1, config, weights),
                  std::runtime_error);
}

TEST_CASE("music3 ar: the frame conditioning row is lm hidden then depth steps 1..n") {
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  std::vector<float> last(hidden);
  for (size_t i = 0; i < hidden; ++i) last[i] = static_cast<float>(i) + 1000.0f;
  std::vector<float> depth(hidden * static_cast<size_t>(config.num_codebooks));
  for (size_t i = 0; i < depth.size(); ++i) depth[i] = static_cast<float>(i);
  const std::vector<float> row = m3::FrameHiddenRow(last, depth, config.num_codebooks, config);
  REQUIRE(row.size() == hidden * static_cast<size_t>(config.num_codebooks));
  int matched = 0;
  for (size_t i = 0; i < hidden; ++i) {
    CHECK(row[i] == last[i]);
    if (row[i] == last[i]) ++matched;
  }
  // Position 0 of the depth block is the language model's own projected state
  // and is DROPPED; the row carries depth steps 1..num_codebooks-1.
  for (size_t i = hidden; i < row.size(); ++i) {
    CHECK(row[i] == depth[i]);
    if (row[i] == depth[i]) ++matched;
  }
  MESSAGE("frame row values matched: " << matched << "/" << row.size());
  CHECK_THROWS_AS(m3::FrameHiddenRow(last, depth, config.num_codebooks - 1, config),
                  std::runtime_error);
}

TEST_CASE("music3 ar: the frame feedback embedding matches _embed_audio_frame") {
  const m3::DepthDecoderConfig config = DepthConfig();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const std::vector<int32_t> codes(
      vllm_test::kMusic3FeedbackCodes + 1,
      vllm_test::kMusic3FeedbackCodes + config.num_codebooks);
  REQUIRE(static_cast<int64_t>(codes.size()) == config.residual_codebooks());
  const std::vector<float> got =
      m3::EmbedAudioFrame(ToVector(vllm_test::kMusic3FeedbackLmRow, hidden), codes, config,
                          DepthWeights());
  const double worst = ExpectClose(got, vllm_test::kMusic3FeedbackOut, hidden, "frame feedback");
  MESSAGE("frame feedback: " << hidden << " values, worst deviation " << worst);
  // The 1/sqrt(num_codebooks) scale is the difference between a plausible-looking
  // embedding and the right one; a partial code set is refused.
  CHECK_THROWS_AS(m3::EmbedAudioFrame(ToVector(vllm_test::kMusic3FeedbackLmRow, hidden), {0},
                                      config, DepthWeights()),
                  std::runtime_error);
}
