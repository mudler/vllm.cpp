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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "minimax_music3_ar_goldens.inc"
#include "vllm/model_executor/models/minimax_music3_ar.h"
#include "vllm/model_executor/models/minimax_music3_llm.h"
#include "vt/dtype.h"

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
  REQUIRE(vllm_test::kMusic3PromptGoldenCount == 4);
  int cases = 0;
  int fields = 0;
  for (int64_t i = 0; i < vllm_test::kMusic3PromptGoldenCount; ++i) {
    const vllm_test::Music3PromptGolden& golden = vllm_test::kMusic3PromptGoldens[i];
    INFO("prompt golden " << golden.name);
    CHECK(m3::CleanCaption(golden.prompt) == std::string(golden.clean_caption));
    CHECK(m3::NormalizeLyrics(golden.lyrics) == std::string(golden.normalized_lyrics));
    CHECK(m3::AssembleArPrompt(golden.prompt, golden.lyrics) == std::string(golden.assembled));
    fields += 3;
    ++cases;
  }
  MESSAGE("prompt goldens checked: " << cases << " cases / " << fields << " string comparisons");
}

// #1083 / #672. `markdown_and_tags` above carries ONE italic span per line, so
// the corpus could not see this class: emulating `(?!\*)` by CAPTURING the
// trailing neighbour consumes it, `regex_replace` resumes scanning past it, and
// a span that opens within one character of the previous close is skipped — the
// surviving asterisks then re-pair ACROSS the intended spans. Row three below is
// the one that matters: `*jazzy keys with a soft brushed*` is not a leftover
// marker, it is a string upstream would never emit, and encoders.py's own header
// says whitespace-level prompt changes change the generated audio.
//
// Every expectation is upstream's own output, not a reading of the regex:
//   git -C <diffusers> worktree add --detach <dir> c6da9936e4bda83107943a16eb8682e9a37d8527
//   PYTHONPATH=<dir>/src python3 -c "from diffusers.modular_pipelines.minimax_music3
//       import encoders as up; print(repr(up._clean_caption(<row>)))"
// (the two lines above are one shell command; they are split for the 100-column
// limit and a backslash continuation is not spellable inside a `//` comment)
TEST_CASE("music3 ar: adjacent italic spans unwrap the way upstream unwraps them") {
  struct Case {
    const char* caption;
    const char* want;
  };
  static const Case kCases[] = {
      {"a *b* *c* d", "a b c d"},
      {"*dreamy* *ambient* pads", "dreamy ambient pads"},
      {"Warm *lo-fi* *jazzy* keys with a *soft* *brushed* snare",
       "Warm lo-fi jazzy keys with a soft brushed snare"},
      {"*a* *b* *c*", "a b c"},
      // The negative side of the same rule, which is why the LEADING guard has
      // to survive the trailing one becoming a true lookahead: a span whose
      // neighbour on either side is an asterisk is NOT a span.
      {"a **b* c", "a **b* c"},
      {"a *b** c", "a *b** c"},
      {"***x*** y", "x y"},
      {"*only* tail", "only tail"},
  };
  int checked = 0;
  for (const Case& entry : kCases) {
    INFO("caption " << entry.caption);
    CHECK(m3::CleanCaption(entry.caption) == std::string(entry.want));
    ++checked;
  }
  MESSAGE("italic captions checked: " << checked);
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

// ---------------------------------------------------------------------------
// The INCREMENTAL depth decode (#672) — the fast schedule, and the proof that
// it is the same arithmetic rather than merely a close one.
// ---------------------------------------------------------------------------

namespace {

// Bit-for-bit, not within a tolerance. Returns how many values were compared so
// a caller can report the count: a gate that cannot say how many things it
// examined has not reported.
size_t ExpectBitIdentical(const std::vector<float>& got, const std::vector<float>& want,
                          const std::string& what) {
  REQUIRE_MESSAGE(got.size() == want.size(), what);
  size_t differing = 0;
  size_t first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
      if (differing == 0) first = i;
      ++differing;
    }
  }
  CHECK_MESSAGE(differing == 0, what << ": " << differing << " of " << got.size()
                                    << " values differ, first at index " << first << " ("
                                    << (differing == 0 ? 0.0f : got[first]) << " vs "
                                    << (differing == 0 ? 0.0f : want[first]) << ")");
  return got.size();
}

}  // namespace

TEST_CASE("music3 ar: the incremental depth decode is BIT-IDENTICAL to the whole sequence") {
  // The claim the generation loop rests on. `DepthDecoderForward` is the
  // MIRRORED reference and stays gated against upstream above; this asserts the
  // incremental schedule that replaced it in `Music3DepthStage` returns the same
  // BYTES for the row that schedule actually reads, at every prefix length and
  // at both compute widths.
  const m3::DepthDecoderConfig config = DepthConfig();
  const m3::DepthDecoderWeights weights = DepthWeights();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t count = static_cast<size_t>(config.max_position_embeddings) * hidden;
  const std::vector<float> all_in = ToVector(vllm_test::kMusic3DepthBoundaryIn, count);

  size_t compared = 0;
  for (const m3::ArCompute compute : {m3::ArCompute::kFloat32, m3::ArCompute::kBFloat16}) {
    m3::DepthDecoderCache cache;
    for (int64_t prefix = 1; prefix <= config.max_position_embeddings; ++prefix) {
      const std::vector<float> whole(
          all_in.begin(), all_in.begin() + static_cast<int64_t>(hidden) * prefix);
      const std::vector<float> reference =
          m3::DepthDecoderForward(whole, prefix, config, weights, compute);
      const std::vector<float> row(
          all_in.begin() + static_cast<int64_t>(hidden) * (prefix - 1),
          all_in.begin() + static_cast<int64_t>(hidden) * prefix);
      const std::vector<float> got =
          m3::DepthDecoderAppend(row, /*batch=*/1, config, weights, compute, &cache);
      CHECK(cache.positions == prefix);
      compared += ExpectBitIdentical(
          got, std::vector<float>(reference.end() - static_cast<int64_t>(hidden), reference.end()),
          std::string("incremental depth position ") + std::to_string(prefix - 1) + " compute=" +
              (compute == m3::ArCompute::kFloat32 ? "f32" : "bf16"));
    }
  }
  MESSAGE("incremental depth decode: " << compared << " values compared BITWISE over "
                                       << config.max_position_embeddings
                                       << " positions x 2 compute widths");
}

TEST_CASE("music3 ar: the incremental depth decode holds at a WIDER geometry") {
  // COVERAGE BREADTH, not a blind spot in the case above. An earlier revision of
  // this comment said the goldens are "8-wide with ONE head", so that a head
  // stride is invisible there. That is FALSE (#1247) and the correction is
  // measured:
  // `minimax_music3_ar_goldens.inc` sets `kMusic3DepthHeads` = 2 over
  // `kMusic3DepthHidden` = 8, so the goldens are 2 heads of 4 and
  // `heads * head_dim` (8) is NOT `head_dim` (4). Dropping the head stride from
  // the cached KEY index — every head reading head 0's keys — reds 4 cases / 32
  // assertions here, and TWO of them are golden-geometry cases: the bit-identity
  // case above and the batched one below.
  //
  // What this case adds is reach, which is worth having on its own. 8 heads of
  // 8, three layers, the real 8-position schedule and pseudo-random weights:
  // 1024 values compared where the goldens compare 96, at a head count and a
  // head_dim that differ from each other AND from the goldens'. It is also one
  // of the two cases that catch a batch-history defect.
  m3::DepthDecoderConfig config;
  config.hidden_size = 64;
  config.num_layers = 3;
  config.num_attention_heads = 8;
  config.intermediate_size = 96;
  config.audio_vocab_size = 4;
  config.num_codebooks = 8;
  config.max_position_embeddings = 8;
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t inter = static_cast<size_t>(config.intermediate_size);

  // A cheap deterministic spread in [-1, 1); the values only have to be generic.
  uint32_t state = 0x9E3779B9u;
  const auto draw = [&state]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<double>(state >> 8) / 8388608.0 - 1.0);
  };
  const auto fill = [&draw](size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = draw();
    return v;
  };

  m3::DepthDecoderWeights weights;
  weights.audio_embeddings = fill(
      static_cast<size_t>(config.audio_vocab_size * config.residual_codebooks()) * hidden);
  weights.projection = fill(hidden * hidden);
  weights.pos_embedding = fill(static_cast<size_t>(config.max_position_embeddings) * hidden);
  weights.norm = fill(hidden);
  for (int64_t l = 0; l < config.num_layers; ++l) {
    m3::DepthDecoderLayerWeights layer;
    layer.input_layernorm = fill(hidden);
    layer.post_attention_layernorm = fill(hidden);
    layer.to_q = fill(hidden * hidden);
    layer.to_k = fill(hidden * hidden);
    layer.to_v = fill(hidden * hidden);
    layer.to_out = fill(hidden * hidden);
    layer.gate_proj = fill(inter * hidden);
    layer.up_proj = fill(inter * hidden);
    layer.down_proj = fill(hidden * inter);
    weights.layers.push_back(std::move(layer));
  }
  for (int64_t h = 0; h < config.residual_codebooks(); ++h) {
    weights.audio_heads.push_back(fill(static_cast<size_t>(config.audio_vocab_size) * hidden));
  }

  const int64_t positions = config.num_codebooks;
  std::vector<float> rows[2];
  rows[0] = fill(static_cast<size_t>(positions) * hidden);
  rows[1] = rows[0];
  for (size_t c = 0; c < hidden; ++c) rows[1][c] = draw();  // position 0 only, as CFG does

  size_t compared = 0;
  m3::DepthDecoderCache cache;
  for (int64_t p = 0; p < positions; ++p) {
    std::vector<float> batched;
    batched.reserve(2 * hidden);
    for (int row = 0; row < 2; ++row) {
      batched.insert(batched.end(), rows[row].begin() + static_cast<int64_t>(hidden) * p,
                     rows[row].begin() + static_cast<int64_t>(hidden) * (p + 1));
    }
    const std::vector<float> got = m3::DepthDecoderAppend(
        batched, /*batch=*/2, config, weights, m3::ArCompute::kBFloat16, &cache);
    for (int row = 0; row < 2; ++row) {
      const std::vector<float> whole(rows[row].begin(),
                                     rows[row].begin() + static_cast<int64_t>(hidden) * (p + 1));
      const std::vector<float> want =
          m3::DepthDecoderForward(whole, p + 1, config, weights, m3::ArCompute::kBFloat16);
      compared += ExpectBitIdentical(
          std::vector<float>(got.begin() + static_cast<int64_t>(hidden) * row,
                             got.begin() + static_cast<int64_t>(hidden) * (row + 1)),
          std::vector<float>(want.end() - static_cast<int64_t>(hidden), want.end()),
          std::string("wide depth row ") + std::to_string(row) + " position " +
              std::to_string(p));
    }
  }
  MESSAGE("wide incremental depth: " << compared << " values compared BITWISE, "
                                     << config.num_attention_heads << " heads of "
                                     << config.head_dim() << " over " << positions
                                     << " positions x 2 rows");
}

TEST_CASE("music3 ar: the incremental depth decode keeps its BATCH rows independent") {
  // The two CFG branches share every appended row and differ ONLY at position 0
  // (encoders.py:125-127), so a cache that mixed them would still return
  // plausible numbers. Each batch row is compared against its OWN whole-sequence
  // forward, which is the only comparison that can see the mixing.
  const m3::DepthDecoderConfig config = DepthConfig();
  const m3::DepthDecoderWeights weights = DepthWeights();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t count = static_cast<size_t>(config.max_position_embeddings) * hidden;
  const std::vector<float> all_in = ToVector(vllm_test::kMusic3DepthBoundaryIn, count);
  const int64_t positions = config.num_codebooks;  // the real schedule's depth

  // Row 1's position 0 is a DIFFERENT vector; every later position is shared.
  std::vector<float> shared[2];
  shared[0].assign(all_in.begin(), all_in.begin() + static_cast<int64_t>(hidden) * positions);
  shared[1] = shared[0];
  for (size_t c = 0; c < hidden; ++c) {
    shared[1][c] = -0.5f * all_in[c] + 0.25f;
  }

  m3::DepthDecoderCache cache;
  std::vector<std::vector<float>> reference(2);
  size_t compared = 0;
  for (int64_t p = 0; p < positions; ++p) {
    std::vector<float> batched;
    batched.reserve(2 * hidden);
    for (int row = 0; row < 2; ++row) {
      batched.insert(batched.end(), shared[row].begin() + static_cast<int64_t>(hidden) * p,
                     shared[row].begin() + static_cast<int64_t>(hidden) * (p + 1));
    }
    const std::vector<float> got = m3::DepthDecoderAppend(
        batched, /*batch=*/2, config, weights, m3::ArCompute::kBFloat16, &cache);
    REQUIRE(got.size() == 2 * hidden);
    for (int row = 0; row < 2; ++row) {
      const std::vector<float> whole(shared[row].begin(),
                                     shared[row].begin() + static_cast<int64_t>(hidden) * (p + 1));
      const std::vector<float> want_all =
          m3::DepthDecoderForward(whole, p + 1, config, weights, m3::ArCompute::kBFloat16);
      compared += ExpectBitIdentical(
          std::vector<float>(got.begin() + static_cast<int64_t>(hidden) * row,
                             got.begin() + static_cast<int64_t>(hidden) * (row + 1)),
          std::vector<float>(want_all.end() - static_cast<int64_t>(hidden), want_all.end()),
          std::string("batched depth row ") + std::to_string(row) + " position " +
              std::to_string(p));
    }
  }
  // The two rows must not have COLLAPSED into each other, or the check above
  // would be satisfied by a cache that served row 0 to both.
  const std::vector<float> last = m3::DepthDecoderAppend(
      std::vector<float>(2 * hidden, 0.125f), /*batch=*/2, config, weights,
      m3::ArCompute::kBFloat16, &cache);
  size_t row_differences = 0;
  for (size_t c = 0; c < hidden; ++c) {
    if (std::memcmp(&last[c], &last[hidden + c], sizeof(float)) != 0) ++row_differences;
  }
  CHECK_MESSAGE(row_differences > 0,
                "the two batch rows are identical, so nothing here could see them mixed");
  MESSAGE("batched incremental depth: " << compared << " values compared BITWISE over "
                                        << positions << " positions x 2 rows; "
                                        << row_differences << " of " << hidden
                                        << " components still separate the rows");
}

TEST_CASE("music3 ar: the incremental depth attention keeps its REDUCTION ORDER") {
  // WHY THIS CASE EXISTS. Every accumulator either side of the comparison above
  // is a `double` stored through a `float`, so a reassociated sum of well-scaled
  // terms differs by ~2^-53 relative while the store rounds at 2^-24 and the
  // narrowing swallows it — the recorded trap, one dtype up. The bit-identity
  // case above therefore CANNOT see the attention's value sweep walk `j` the
  // other way, and a gate that cannot see its own defect is not a gate.
  //
  // So this engineers the cancellation. A one-layer decoder with `to_q` and
  // `to_k` zero makes every softmax weight exactly 1/seq, and an
  // `input_layernorm` whose FIRST component is 2^60 makes positions 0 and 1
  // carry +2^60 and -2^60 into `v` while every later position carries 0 there
  // and O(1) elsewhere. Ascending `j` cancels the pair immediately and keeps the
  // remainder exactly; any order that carries 2^60 through the remainder
  // annihilates it, because the gap is 57 bits and a double has 53.
  m3::DepthDecoderConfig config;
  config.hidden_size = 8;
  config.num_layers = 1;
  config.num_attention_heads = 1;
  config.intermediate_size = 8;
  config.audio_vocab_size = 2;
  config.num_codebooks = 4;
  config.max_position_embeddings = 8;
  const size_t hidden = static_cast<size_t>(config.hidden_size);

  m3::DepthDecoderWeights weights;
  weights.audio_embeddings.assign(
      static_cast<size_t>(config.audio_vocab_size * config.residual_codebooks()) * hidden, 0.0f);
  weights.projection.assign(hidden * hidden, 0.0f);
  weights.pos_embedding.assign(
      static_cast<size_t>(config.max_position_embeddings) * hidden, 0.0f);
  weights.norm.assign(hidden, 1.0f);
  m3::DepthDecoderLayerWeights layer;
  layer.input_layernorm.assign(hidden, 1.0f);
  layer.input_layernorm[0] = std::ldexp(1.0f, 60);
  layer.post_attention_layernorm.assign(hidden, 1.0f);
  layer.to_q.assign(hidden * hidden, 0.0f);
  layer.to_k.assign(hidden * hidden, 0.0f);
  layer.to_v.assign(hidden * hidden, 1.0f);
  layer.to_out.assign(hidden * hidden, 0.0f);
  for (size_t d = 0; d < hidden; ++d) layer.to_out[d * hidden + d] = 1.0f;
  layer.gate_proj.assign(static_cast<size_t>(config.intermediate_size) * hidden, 0.0f);
  layer.up_proj.assign(static_cast<size_t>(config.intermediate_size) * hidden, 0.0f);
  layer.down_proj.assign(hidden * static_cast<size_t>(config.intermediate_size), 0.0f);
  weights.layers.push_back(layer);
  weights.audio_heads.assign(static_cast<size_t>(config.residual_codebooks()),
                             std::vector<float>(static_cast<size_t>(config.audio_vocab_size) *
                                                    hidden, 0.0f));

  // Position 0 carries +1 in the amplified component, position 1 its exact
  // negation, and every later position carries ZERO there so its `v` stays O(1).
  const int64_t positions = 6;
  std::vector<float> embeds(static_cast<size_t>(positions) * hidden, 1.0f);
  for (size_t c = 0; c < hidden; ++c) embeds[hidden + c] = -embeds[c];
  for (int64_t p = 2; p < positions; ++p) {
    embeds[static_cast<size_t>(p) * hidden] = 0.0f;
    for (size_t c = 1; c < hidden; ++c) {
      embeds[static_cast<size_t>(p) * hidden + c] = 1.0f + 0.25f * static_cast<float>(p + c);
    }
  }

  size_t compared = 0;
  size_t nonzero = 0;
  m3::DepthDecoderCache cache;
  for (int64_t p = 0; p < positions; ++p) {
    const std::vector<float> row(embeds.begin() + static_cast<int64_t>(hidden) * p,
                                 embeds.begin() + static_cast<int64_t>(hidden) * (p + 1));
    const std::vector<float> got =
        m3::DepthDecoderAppend(row, /*batch=*/1, config, weights, m3::ArCompute::kFloat32, &cache);
    const std::vector<float> whole(embeds.begin(),
                                   embeds.begin() + static_cast<int64_t>(hidden) * (p + 1));
    const std::vector<float> want =
        m3::DepthDecoderForward(whole, p + 1, config, weights, m3::ArCompute::kFloat32);
    compared += ExpectBitIdentical(
        got, std::vector<float>(want.end() - static_cast<int64_t>(hidden), want.end()),
        std::string("cancellation position ") + std::to_string(p));
    if (p >= 2) {
      for (const float value : got) {
        if (value != 0.0f && std::isfinite(value)) ++nonzero;
      }
    }
  }
  // The teeth check. If the cancelled remainder were zero — or infinite — the
  // case above would be satisfied by any order at all, and the whole apparatus
  // would be vacuous while still printing green.
  CHECK_MESSAGE(nonzero > 0,
                "the cancelled remainder is zero or non-finite, so no order can be distinguished");
  MESSAGE("cancellation case: " << compared << " values compared BITWISE, " << nonzero
                                << " finite non-zero remainder components");
}

TEST_CASE("music3 ar: the incremental depth cache refuses what it cannot hold, by name") {
  const m3::DepthDecoderConfig config = DepthConfig();
  const m3::DepthDecoderWeights weights = DepthWeights();
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const std::vector<float> row(hidden, 0.5f);

  m3::DepthDecoderCache cache;
  CHECK_THROWS_AS(m3::DepthDecoderAppend(row, 1, config, weights, m3::ArCompute::kFloat32,
                                         nullptr),
                  std::runtime_error);
  CHECK_THROWS_AS(m3::DepthDecoderAppend(row, 0, config, weights, m3::ArCompute::kFloat32,
                                         &cache),
                  std::runtime_error);
  CHECK_THROWS_AS(m3::DepthDecoderAppend(std::vector<float>(hidden + 1, 0.5f), 1, config, weights,
                                         m3::ArCompute::kFloat32, &cache),
                  std::runtime_error);
  // A cache opened at batch 1 may not be continued at batch 2: the K/V blocks
  // for the second row do not exist and a silent resize would serve row 0's
  // history to it.
  m3::DepthDecoderAppend(row, 1, config, weights, m3::ArCompute::kFloat32, &cache);
  CHECK(cache.positions == 1);
  CHECK_THROWS_AS(m3::DepthDecoderAppend(std::vector<float>(2 * hidden, 0.5f), 2, config, weights,
                                         m3::ArCompute::kFloat32, &cache),
                  std::runtime_error);
  // The learned position table is the ceiling, exactly as it is for the
  // whole-sequence forward.
  for (int64_t p = 1; p < config.max_position_embeddings; ++p) {
    m3::DepthDecoderAppend(row, 1, config, weights, m3::ArCompute::kFloat32, &cache);
  }
  CHECK(cache.positions == config.max_position_embeddings);
  CHECK_THROWS_AS(m3::DepthDecoderAppend(row, 1, config, weights, m3::ArCompute::kFloat32, &cache),
                  std::runtime_error);
}

TEST_CASE("music3 ar: the COMPOSED depth stage is BIT-IDENTICAL to the schedule it replaced") {
  // WHY THIS CASE EXISTS, and why the four above do not cover it. Those gate
  // `DepthDecoderAppend` — ONE row against ONE whole-sequence forward. They say
  // nothing about the SCHEDULE `Music3DepthStage` composes out of it: the 3-row
  // prefix projection, position 0 fed for its K/V alone, the batch-2 call
  // sequencing, and the fed-back `(index-1) * audio_vocab_size + drawn`
  // projection row. Silently dropping the feedback row changes the generated
  // song and leaves every other case in this file GREEN, and so does deleting
  // the prefix append; both were measured, not supposed (#1246).
  //
  // So this drives the PRODUCTION function — `Music3DepthStage`, the one
  // `Music3GenerateFrameHiddens` calls, which is how the registered speech
  // family reaches this row's change — against a TRANSCRIPTION of the
  // whole-sequence schedule it replaced (`DepthSequenceEmbeds` +
  // `DepthDecoderForward` per CFG branch, encoders.py:117-142), with the same
  // sampler, the same weights and the same order of draws. No checkpoint: this
  // path reads `lm_config.hidden_size` and the embedding rows `EmbedRow`
  // serves, and nothing else.
  //
  // The geometry is WIDER than any committed golden — 8 heads of 8, the full
  // 8-codebook schedule, a 32-entry audio vocabulary — so the head stride, all
  // seven steps and the feedback row index are exercised together.
  m3::DepthDecoderConfig config;
  config.hidden_size = 64;
  config.num_layers = 2;
  config.num_attention_heads = 8;
  config.intermediate_size = 96;
  config.audio_vocab_size = 32;
  config.num_codebooks = 8;
  config.max_position_embeddings = 16;
  const int64_t H = config.hidden_size;
  const size_t hidden = static_cast<size_t>(H);
  const size_t inter = static_cast<size_t>(config.intermediate_size);

  uint32_t state = 0x9E3779B9u;
  const auto draw = [&state]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<double>(state >> 8) / 8388608.0 - 1.0) * 0.5f;
  };
  const auto fill = [&draw](size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = draw();
    return v;
  };

  m3::DepthDecoderWeights depth;
  depth.audio_embeddings =
      fill(static_cast<size_t>(config.audio_vocab_size * config.residual_codebooks()) * hidden);
  depth.projection = fill(hidden * hidden);
  depth.pos_embedding = fill(static_cast<size_t>(config.max_position_embeddings) * hidden);
  depth.norm = fill(hidden);
  for (int64_t l = 0; l < config.num_layers; ++l) {
    m3::DepthDecoderLayerWeights layer;
    layer.input_layernorm = fill(hidden);
    layer.post_attention_layernorm = fill(hidden);
    layer.to_q = fill(hidden * hidden);
    layer.to_k = fill(hidden * hidden);
    layer.to_v = fill(hidden * hidden);
    layer.to_out = fill(hidden * hidden);
    layer.gate_proj = fill(inter * hidden);
    layer.up_proj = fill(inter * hidden);
    layer.down_proj = fill(hidden * inter);
    depth.layers.push_back(std::move(layer));
  }
  for (int64_t h = 0; h < config.residual_codebooks(); ++h) {
    depth.audio_heads.push_back(fill(static_cast<size_t>(config.audio_vocab_size) * hidden));
  }

  m3::Music3ArWeights weights;
  weights.depth_config = config;
  weights.depth = depth;
  weights.lm_config.hidden_size = H;
  const int64_t vocab = m3::kAudioCodeOffset + 64;
  weights.lm_config.vocab_size = vocab;
  {
    // bf16 rows, because `EmbedRow` reads the table as the loader stores it. Only
    // the audio-code window can be reached from here, so only it is filled.
    std::vector<uint8_t> bytes(static_cast<size_t>(vocab) * hidden * sizeof(uint16_t), 0);
    uint16_t* const rows = reinterpret_cast<uint16_t*>(bytes.data());
    for (int64_t t = m3::kAudioCodeOffset; t < vocab; ++t) {
      for (int64_t j = 0; j < H; ++j) rows[t * H + j] = vt::F32ToBF16(draw());
    }
    weights.lm.embed_tokens.bytes = vllm::OwnedBytes(std::move(bytes));
    weights.lm.embed_tokens.dtype = vt::DType::kBF16;
    weights.lm.embed_tokens.rank = 2;
    weights.lm.embed_tokens.shape[0] = vocab;
    weights.lm.embed_tokens.shape[1] = H;
  }

  const int32_t semantic_code = 7;
  const int64_t frame_index = 3;
  // DISTINCT per branch, which is the whole of what the unconditional row
  // contributes: two equal rows would hide a schedule that served one to both.
  const std::vector<float> last_conditional = fill(hidden);
  const std::vector<float> last_unconditional = fill(hidden);

  // A sampler keyed on the LOGITS rather than on the call order: argmax, ties to
  // the lowest index. The two schedules agree on a drawn code only if they ask
  // the same question, in the same order, from the same hidden state.
  int64_t calls_new = 0;
  int64_t calls_old = 0;
  const auto make_sampler = [](int64_t* counter) {
    return m3::Music3CodeSampler(
        [counter](const std::vector<float>& probs, const m3::Music3Draw&) -> int64_t {
          ++*counter;
          size_t best = 0;
          for (size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > probs[best]) best = i;
          }
          return static_cast<int64_t>(best);
        });
  };

  std::vector<int32_t> codes_new{semantic_code};
  const std::vector<float> got =
      m3::Music3DepthStage(last_conditional, last_unconditional, frame_index, weights,
                           make_sampler(&calls_new), &codes_new);

  const std::vector<float> semantic_embed =
      weights.EmbedRow(static_cast<int64_t>(semantic_code) + m3::kAudioCodeOffset);
  std::vector<float> want;
  std::vector<int32_t> codes_old{semantic_code};
  std::vector<int32_t> fed_back;
  const m3::Music3CodeSampler sampler_old = make_sampler(&calls_old);
  for (int64_t index = 1; index < config.num_codebooks; ++index) {
    std::vector<float> hidden_rows[2];
    for (int row = 0; row < 2; ++row) {
      const std::vector<float>& last = row == 0 ? last_conditional : last_unconditional;
      const std::vector<float> embeds = m3::DepthSequenceEmbeds(
          last, semantic_embed, fed_back, config, depth, m3::ArCompute::kBFloat16);
      const std::vector<float> states = m3::DepthDecoderForward(
          embeds, /*seq_len=*/index + 1, config, depth, m3::ArCompute::kBFloat16);
      hidden_rows[row].assign(states.end() - H, states.end());
    }
    // `hidden_parts.append(hidden[:1])` — the CONDITIONAL row alone.
    want.insert(want.end(), hidden_rows[0].begin(), hidden_rows[0].end());
    const std::vector<float> conditional =
        m3::AudioHeadLogits(hidden_rows[0], index - 1, config, depth, m3::ArCompute::kBFloat16);
    const std::vector<float> unconditional =
        m3::AudioHeadLogits(hidden_rows[1], index - 1, config, depth, m3::ArCompute::kBFloat16);
    const std::vector<float> guided =
        m3::GuidedDepthLogits(conditional, unconditional, m3::kArCfgScale);
    const std::vector<float> probs = m3::TopKProbabilities(guided, m3::kArSamplingTopK);
    const int64_t drawn = sampler_old(probs, m3::Music3Draw{frame_index, index});
    codes_old.push_back(static_cast<int32_t>(drawn));
    // c7 is only ever PREDICTED (encoders.py:139).
    if (index < config.num_codebooks - 1) fed_back.push_back(static_cast<int32_t>(drawn));
  }

  const size_t compared =
      ExpectBitIdentical(got, want, "composed depth stage, conditional hidden rows");
  CHECK_MESSAGE(codes_new == codes_old, "the two schedules drew different residual codes");
  CHECK(calls_new == calls_old);
  CHECK(calls_new == config.residual_codebooks());
  CHECK(static_cast<int64_t>(codes_new.size()) == config.num_codebooks);
  CHECK(codes_new[0] == semantic_code);
  // The teeth, twice over. An all-zero reference block would satisfy the
  // comparison while proving nothing, and a stage that drew nothing would
  // satisfy the code check the same way.
  size_t nonzero = 0;
  for (const float value : want) {
    if (value != 0.0f) ++nonzero;
  }
  CHECK_MESSAGE(nonzero > 0, "the reference block is all zeros, so nothing here is comparable");
  CHECK_MESSAGE(calls_new > 0, "no code was drawn, so the draw order is untested");
  MESSAGE("composed depth stage: " << compared << " values compared BITWISE over "
                                   << config.residual_codebooks() << " codebook steps x 2 CFG rows, "
                                   << nonzero << " of " << want.size()
                                   << " reference values non-zero, " << calls_new << " draws");
}

// ---------------------------------------------------------------------------
// The DEVICE arm (#1309, spec §19)
// ---------------------------------------------------------------------------
//
// These run on a `vt::Queue` whose device is kCPU. That is not a compromise, it
// is the point: every op the device forward composes — kMatmulBT, kRmsNorm,
// kAttentionCross, kSiluAndMul, kAdd — carries a CPU provider AND a CUDA one, so
// the composition, the merged weight layouts, the cache indexing and the
// PRODUCTION SELECTION are all gated here with no GPU and no checkpoint.
//
// #1131 is why the selection is gated rather than only the forward. The DiT
// device arm's kernels are gated and its switch is not, so `on_device = false`
// leaves every suite green — the two arms agree numerically BY DESIGN, and no
// gate there ever asked WHICH ONE RAN. `Music3DepthDeviceForwardCount()` is what
// asks it here.
//
// What these cases do NOT reach is the CUDA kernels themselves, and that is said
// rather than implied: the CUDA `AttentionCross` uses an online-softmax
// recurrence where the CPU one uses three passes, and cuBLASLt splits K by an
// algorithm no CPU provider has. Spec §19.6 owes that leg to `thor:gpu0`.

// The spacing of bf16 at `value`'s magnitude. bf16 keeps 8 significand bits
// (1 implicit + 7 stored), so a value in [2^k, 2^(k+1)) has ULP 2^(k-7).
double Bf16Ulp(double value) {
  if (value == 0.0 || !std::isfinite(value)) return std::ldexp(1.0, -133);
  int exponent = 0;
  std::frexp(std::abs(value), &exponent);  // |v| = m * 2^e, m in [0.5, 1)
  return std::ldexp(1.0, exponent - 8);
}

// A CPU queue. `vt::Queue`'s CPU form carries a null handle by construction —
// the speech engine builds exactly this when `--speech-device 0`.
vt::Queue CpuQueue() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// A deterministic reduced geometry, WIDER than the goldens on both axes that a
// merged weight layout could get wrong: 8 heads of 8 (so `heads * head_dim` is
// not `head_dim`), and an intermediate size that is not a multiple of the hidden
// size (so a gate/up half swap cannot be masked by a symmetric shape).
m3::DepthDecoderConfig DeviceArmConfig() {
  m3::DepthDecoderConfig config;
  config.hidden_size = 64;
  config.num_layers = 3;
  config.num_attention_heads = 8;
  config.intermediate_size = 96;
  config.audio_vocab_size = 32;
  config.num_codebooks = 8;
  config.max_position_embeddings = 8;
  return config;
}

// Weights whose values are ALREADY bf16-exact, because that is what the loader
// hands the arms: `AtRuntimeDtype` rounds every AR-half tensor through bf16 into
// its f32 carrier. Drawing f32 noise here instead would make the device arm's
// staging lossy for a reason the shipped path does not have, and would price a
// rounding this row does not perform.
std::vector<float> DeviceArmFill(size_t n, uint32_t* state, float scale) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    *state = *state * 1664525u + 1013904223u;
    const float raw =
        static_cast<float>(static_cast<double>(*state >> 8) / 8388608.0 - 1.0) * scale;
    v[i] = vt::BF16ToF32(vt::F32ToBF16(raw));
  }
  return v;
}

m3::DepthDecoderWeights DeviceArmWeights(const m3::DepthDecoderConfig& config, uint32_t* state) {
  const size_t hidden = static_cast<size_t>(config.hidden_size);
  const size_t inter = static_cast<size_t>(config.intermediate_size);
  m3::DepthDecoderWeights weights;
  weights.audio_embeddings = DeviceArmFill(
      static_cast<size_t>(config.audio_vocab_size * config.residual_codebooks()) * hidden, state,
      0.5f);
  weights.projection = DeviceArmFill(hidden * hidden, state, 0.25f);
  weights.pos_embedding =
      DeviceArmFill(static_cast<size_t>(config.max_position_embeddings) * hidden, state, 0.5f);
  weights.norm = DeviceArmFill(hidden, state, 0.5f);
  for (int64_t l = 0; l < config.num_layers; ++l) {
    m3::DepthDecoderLayerWeights layer;
    layer.input_layernorm = DeviceArmFill(hidden, state, 0.5f);
    layer.post_attention_layernorm = DeviceArmFill(hidden, state, 0.5f);
    layer.to_q = DeviceArmFill(hidden * hidden, state, 0.25f);
    layer.to_k = DeviceArmFill(hidden * hidden, state, 0.25f);
    layer.to_v = DeviceArmFill(hidden * hidden, state, 0.25f);
    layer.to_out = DeviceArmFill(hidden * hidden, state, 0.25f);
    layer.gate_proj = DeviceArmFill(inter * hidden, state, 0.25f);
    layer.up_proj = DeviceArmFill(inter * hidden, state, 0.25f);
    layer.down_proj = DeviceArmFill(hidden * inter, state, 0.25f);
    weights.layers.push_back(std::move(layer));
  }
  for (int64_t h = 0; h < config.residual_codebooks(); ++h) {
    weights.audio_heads.push_back(
        DeviceArmFill(static_cast<size_t>(config.audio_vocab_size) * hidden, state, 0.25f));
  }
  return weights;
}

// The seeds this gate measures over, and WHY there is more than one. The bound
// below was originally fitted to the first of them, and a review falsified it by
// changing NOTHING BUT THE SEED: same distribution, same geometry, no defect, and
// the shipped `mean <= 4.0` red on three of six equally valid draws. A tolerance
// that a redraw of the reference weights can fail is measuring the draw, not the
// arm. §19.4b carries the per-seed table and the arithmetic behind the bounds.
constexpr uint32_t kDeviceArmSeeds[] = {0x9E3779B9u, 0x2468ACE0u, 0x00000001u,
                                        0x13579BDFu, 0xDEADBEEFu, 0x51ED2701u};
constexpr size_t kNumDeviceArmSeeds = sizeof(kDeviceArmSeeds) / sizeof(kDeviceArmSeeds[0]);

// One draw's worth of device-vs-host deviation.
//
// TWO BUCKETS, AND THEY SUM. A reference value of EXACTLY zero has no ULP —
// bf16's spacing at zero is the denormal floor, so `|got| / Bf16Ulp(0)` reads
// ~1e35 for an absolute difference of 1e-5 and the statistic stops meaning
// anything about the arm. Seed `0x51ED2701` draws one such value in 1024 and the
// CORRECT arm reads mean 8.8e32 on it. So a zero reference is measured
// ABSOLUTELY instead, and the two counts are asserted to sum to the compared
// total: a defect that produced zeros would have to grow that bucket to hide in
// it, and growing it is a failure.
struct DeviceArmBand {
  double worst = 0.0;       // bf16 ULP, over the non-zero references only
  double mean = 0.0;        // bf16 ULP, over the non-zero references only
  double median = 0.0;      // bf16 ULP, over the non-zero references only
  double worst_at_zero = 0.0;  // absolute, over the zero references only
  size_t compared = 0;
  size_t nonzero = 0;
  size_t at_zero = 0;
  uint64_t forwards = 0;
};

// Run BOTH arms over the whole position schedule at one seed and measure the gap.
DeviceArmBand MeasureDeviceArmBand(const m3::DepthDecoderConfig& config, uint32_t seed) {
  uint32_t state = seed;
  m3::DepthDecoderWeights host_weights = DeviceArmWeights(config, &state);
  m3::DepthDecoderWeights stage_source = host_weights;

  vt::Queue queue = CpuQueue();
  // `release_host=false`: this gate compares the two arms, so the host copy must
  // survive. A serving path passes true.
  const m3::Music3DepthDeviceWeights staged =
      m3::StageMusic3DepthWeights(queue, config, stage_source, /*release_host=*/false);
  REQUIRE(staged.staged());

  const int64_t batch = 2;
  const int64_t H = config.hidden_size;
  m3::DepthDecoderCache host_cache;
  m3::Music3DepthDeviceCache device_cache;
  DeviceArmBand band;
  std::vector<double> ulps;
  ulps.reserve(static_cast<size_t>(config.max_position_embeddings * batch * H));
  const uint64_t before = m3::Music3DepthDeviceForwardCount();

  for (int64_t position = 0; position < config.max_position_embeddings; ++position) {
    // The two CFG rows differ, which is what makes a schedule that served one to
    // both detectable at all.
    const std::vector<float> embeds = DeviceArmFill(static_cast<size_t>(batch * H), &state, 1.0f);
    const std::vector<float> want = m3::DepthDecoderAppend(
        embeds, batch, config, host_weights, m3::ArCompute::kBFloat16, &host_cache);
    const std::vector<float> got =
        m3::DepthDecoderAppendDevice(queue, config, staged, embeds, batch, &device_cache);
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < want.size(); ++i) {
      const double absolute = std::abs(static_cast<double>(got[i]) - want[i]);
      ++band.compared;
      if (want[i] == 0.0f) {
        band.worst_at_zero = std::max(band.worst_at_zero, absolute);
        ++band.at_zero;
        continue;
      }
      const double ulp = absolute / Bf16Ulp(want[i]);
      band.worst = std::max(band.worst, ulp);
      band.mean += ulp;
      ulps.push_back(ulp);
      ++band.nonzero;
    }
  }
  band.forwards = m3::Music3DepthDeviceForwardCount() - before;
  band.mean /= static_cast<double>(band.nonzero == 0 ? 1 : band.nonzero);
  if (!ulps.empty()) {
    const size_t mid = ulps.size() / 2;
    std::nth_element(ulps.begin(), ulps.begin() + static_cast<long>(mid), ulps.end());
    band.median = ulps[mid];
  }
  return band;
}

// THE BOUNDS, placed from a MULTI-SEED measurement against a mutation battery
// rather than from the deviation one draw happened to produce. §19.4b carries
// the whole table; the two things that decide the numbers are these.
//
// THE CORRECT ARM, over six seeds:
//
//   seed         worst    mean  median  note
//   0x9E3779B9     110   2.095       1  the seed the old bound was fitted to
//   0x2468ACE0     435   3.804       1
//   0x00000001    1110   5.755       1  RED under the old `mean <= 4.0`
//   0x13579BDF    3663   7.154       1  RED under the old `mean <= 4.0`
//   0xDEADBEEF    7340   9.904       1  RED under the old `mean <= 4.0`
//   0x51ED2701     939   3.103       1  one reference is exactly zero
//
// FIVE STRUCTURAL MUTATIONS, each over the same six seeds, worst case first:
//
//   mutation                       mean range   median range
//   wrong attention scale          19.6 -  71.3      4 -   4
//   gate/up half swap              52.5 - 158.8     12 -  15
//   K/V cache row collision       166.4 - 376.1      3 -   4
//   dropped position embedding    357.8 - 1487       76 -  90
//   q|k|v merge order swapped     400.0 - 1731      104 - 131
//
// WHY THE MEDIAN IS THE PRIMARY GATE. It is EXACTLY 1 bf16 ULP at all six seeds
// for the correct arm — a constant, not a distribution — because the deviation
// this arm actually carries is one rounding-polarity tick per element, and the
// seed only changes the tail. Every defect is at least 3. So the median's window
// is (1, 3] and it cannot be moved by a redraw, which is precisely the property
// the old single-seed bound lacked.
//
// WHY THE MEAN IS STILL GATED, and what it costs. The mean is tail-sensitive, so
// it catches a SPARSE defect that leaves the middle of the distribution alone and
// the median would not see. Its window is narrow and asymmetric: the correct
// arm's worst draw is 9.904 and the tightest defect is 19.6 — the wrong attention
// scale at `0xDEADBEEF`, which is the same seed that produces the correct arm's
// own worst mean. 15 sits inside that window, 1.51x above every correct draw
// measured and 1.31x below every defect draw measured. That is a real margin and
// it is not a large one, and a seventh seed drawing a correct mean above 15 is
// possible in a way that one drawing a median above 2 is not.
//
// WHY `worst` IS NOT GATED AT ALL. It cannot discriminate. A CORRECT arm reads
// worst 7340 at `0xDEADBEEF` while the gate/up half swap reads 6641 and the wrong
// attention scale 6865 at the seed each is worst-separated on, so any `worst`
// bound loose enough to admit a correct implementation admits two structural
// defects. It is reported, and it carries only a canary far above any draw here,
// because a value that has gone non-finite must not read as rounding.
constexpr double kDeviceArmMedianUlpBound = 2.0;
constexpr double kDeviceArmMeanUlpBound = 15.0;
constexpr double kDeviceArmWorstUlpCanary = 1.0e6;
// The zero-reference bucket, measured ABSOLUTELY because it has no ULP. See
// `DeviceArmBand`.
constexpr double kDeviceArmZeroAbsBound = 1.0e-2;

TEST_CASE("music3 ar: the DEVICE depth decode tracks the host arm inside a bf16 band") {
  // NOT bitwise, and the spec says so BEFORE this code rather than after a
  // surprise (§19.4). Three reasons, each sufficient alone: the host keeps a
  // sequential `double` per output element where `vt::MatmulBT` accumulates in
  // f32; the reduction re-associates; and `vt::RmsNorm` keeps full f32 precision
  // across the weight multiply where our `RmsNorm` mirrors upstream's TWO
  // roundings. The first of those makes the device arm the CLOSER mirror of
  // torch, which accumulates bf16 matmuls in f32.
  //
  // So the bound is in bf16 ULPs OF THE REFERENCE VALUE, not in absolute units:
  // four layers of activations span orders of magnitude and an absolute bound
  // would be vacuous at the top and impossible at the bottom.
  //
  // And it is asserted over SIX DRAWS, because the single-draw version of this
  // case passed while a correct arm failed on three other seeds (§19.4b).
  const m3::DepthDecoderConfig config = DeviceArmConfig();
  double worst_over_seeds = 0.0;
  double worst_mean = 0.0;
  double worst_median = 0.0;
  size_t total_compared = 0;
  size_t total_nonzero = 0;
  size_t total_at_zero = 0;
  double worst_at_zero = 0.0;
  uint64_t total_forwards = 0;

  for (size_t s = 0; s < kNumDeviceArmSeeds; ++s) {
    const uint32_t seed = kDeviceArmSeeds[s];
    CAPTURE(seed);
    const DeviceArmBand band = MeasureDeviceArmBand(config, seed);
    worst_over_seeds = std::max(worst_over_seeds, band.worst);
    worst_mean = std::max(worst_mean, band.mean);
    worst_median = std::max(worst_median, band.median);
    total_compared += band.compared;
    total_nonzero += band.nonzero;
    total_at_zero += band.at_zero;
    worst_at_zero = std::max(worst_at_zero, band.worst_at_zero);
    total_forwards += band.forwards;

    CHECK_MESSAGE(band.median <= kDeviceArmMedianUlpBound,
                  "seed " << seed << ": device/host MEDIAN deviation " << band.median
                          << " bf16 ULP exceeds " << kDeviceArmMedianUlpBound
                          << "; the correct arm is exactly 1 at every seed measured, so this "
                             "is a STRUCTURAL defect and not a redraw");
    CHECK_MESSAGE(band.mean <= kDeviceArmMeanUlpBound,
                  "seed " << seed << ": device/host MEAN deviation " << band.mean
                          << " bf16 ULP exceeds " << kDeviceArmMeanUlpBound
                          << ", which is a STRUCTURAL defect, not rounding");
    CHECK_MESSAGE(band.worst <= kDeviceArmWorstUlpCanary,
                  "seed " << seed << ": device/host WORST deviation " << band.worst
                          << " bf16 ULP is past the non-finite canary");
    CHECK_MESSAGE(band.worst_at_zero <= kDeviceArmZeroAbsBound,
                  "seed " << seed << ": where the reference is EXACTLY zero the device arm is "
                          << band.worst_at_zero << " away, past " << kDeviceArmZeroAbsBound);
    // Per draw, not only in aggregate, and the buckets SUM. An all-zero
    // reference satisfies any tolerance; so does a forward that never ran; and
    // so does a defect that quietly moved every element into the un-gated
    // zero-reference bucket.
    CHECK_MESSAGE(band.nonzero > 0, "seed " << seed << ": the reference block is all zeros");
    const size_t bucketed = band.nonzero + band.at_zero;
    CHECK_MESSAGE(bucketed == band.compared,
                  "seed " << seed << ": " << band.nonzero << " + " << band.at_zero
                          << " buckets do not sum to " << band.compared << " compared values");
    CHECK_MESSAGE(band.compared ==
                      static_cast<size_t>(config.max_position_embeddings * 2 * config.hidden_size),
                  "seed " << seed << ": the position sweep did not compare every position");
    CHECK_MESSAGE(band.forwards == static_cast<uint64_t>(config.max_position_embeddings),
                  "seed " << seed << ": the DEVICE forward ran " << band.forwards
                          << " times, expected " << config.max_position_embeddings);
    MESSAGE("seed " << seed << ": worst " << band.worst << " bf16 ULP, mean " << band.mean
                    << ", median " << band.median << ", " << band.nonzero << "/" << band.compared
                    << " references non-zero, " << band.at_zero << " at zero (worst absolute "
                    << band.worst_at_zero << ")");
  }

  // THE SWEEP ITSELF HAS TEETH. A loop that silently measured one seed six times,
  // or zero seeds, would satisfy every bound above.
  CHECK_MESSAGE(kNumDeviceArmSeeds >= 6,
                "a multi-seed bound needs more than one seed, got " << kNumDeviceArmSeeds);
  const uint64_t want_forwards =
      static_cast<uint64_t>(config.max_position_embeddings) * kNumDeviceArmSeeds;
  CHECK(total_forwards == want_forwards);
  const size_t total_bucketed = total_nonzero + total_at_zero;
  CHECK(total_bucketed == total_compared);
  // The un-gated bucket must stay NEGLIGIBLE. It is the one place a defect could
  // sit without being measured in ULPs, so its size is a gate of its own.
  CHECK_MESSAGE(total_at_zero * 100 < total_compared,
                total_at_zero << " of " << total_compared
                              << " references are exactly zero, so the ULP statistic is being "
                                 "computed over a shrinking minority");
  MESSAGE("device depth decode over " << kNumDeviceArmSeeds << " seeds: " << total_compared
                                      << " values (" << total_at_zero << " at zero, worst "
                                      << worst_at_zero << " absolute), worst "
                                      << worst_over_seeds
                                      << " bf16 ULP (reported, not gated), worst mean "
                                      << worst_mean << " against " << kDeviceArmMeanUlpBound
                                      << ", worst median " << worst_median << " against "
                                      << kDeviceArmMedianUlpBound);
}

TEST_CASE("music3 ar: the device depth arm is bf16 RESIDENT, on the weights AND the activations") {
  // #1131 offers two instruments — "invocation count OR resident dtype" — and
  // this row took only the first until a review took the second one's absence
  // and proved it mattered: widening a single activation buffer to `kF32` left
  // the ULP band, the drawn codes and every case in this file GREEN while the
  // path moved twice the bytes. That is AGENTS.md's "a token gate cannot detect
  // a dtype that is too WIDE" landing on the very row whose thesis is a dtype,
  // and it is §19.2a's finding about the HOST arm arriving inside the arm that
  // exists to fix it.
  const m3::DepthDecoderConfig config = DeviceArmConfig();
  uint32_t state = 0x5EEDD7EEu;
  m3::DepthDecoderWeights source = DeviceArmWeights(config, &state);
  vt::Queue queue = CpuQueue();
  const m3::Music3DepthDeviceWeights staged =
      m3::StageMusic3DepthWeights(queue, config, source, /*release_host=*/false);
  REQUIRE(staged.staged());

  // THE STAGED WEIGHTS, read off the tensors rather than inferred from the
  // numbers. f32 weights would be MORE precise and would sail through every
  // tolerance in this file.
  CHECK(staged.pos_embedding.dtype == vt::DType::kBF16);
  CHECK(staged.norm.dtype == vt::DType::kBF16);
  REQUIRE(static_cast<int64_t>(staged.layers.size()) == config.num_layers);
  for (const m3::Music3DepthDeviceLayer& layer : staged.layers) {
    CHECK(layer.input_layernorm.dtype == vt::DType::kBF16);
    CHECK(layer.post_attention_layernorm.dtype == vt::DType::kBF16);
    CHECK(layer.qkv.dtype == vt::DType::kBF16);
    CHECK(layer.to_out.dtype == vt::DType::kBF16);
    CHECK(layer.gate_up.dtype == vt::DType::kBF16);
    CHECK(layer.down_proj.dtype == vt::DType::kBF16);
  }

  // THE ACTIVATIONS AND THE K/V CACHE, which no caller can see from outside, so
  // the forward reports what it allocated. One call is enough to populate the
  // mask; the mask is monotone over the process, so a widened buffer anywhere in
  // this binary lands here.
  const int64_t H = config.hidden_size;
  const std::vector<float> embeds = DeviceArmFill(static_cast<size_t>(2 * H), &state, 1.0f);
  m3::Music3DepthDeviceCache cache;
  const std::vector<float> out =
      m3::DepthDecoderAppendDevice(queue, config, staged, embeds, 2, &cache);
  CHECK(out.size() == static_cast<size_t>(2 * H));

  const uint64_t resident = m3::Music3DepthDeviceResidentDtypes();
  const uint64_t bf16_only = uint64_t{1} << static_cast<unsigned>(vt::DType::kBF16);
  CHECK_MESSAGE(resident != 0,
                "the resident-dtype instrument reported NOTHING, so it measured nothing");
  CHECK_MESSAGE(resident == bf16_only,
                "the device depth forward made buffers resident in dtypes other than bf16: "
                "mask 0x"
                    << std::hex << resident << std::dec << ", expected 0x" << std::hex
                    << bf16_only << std::dec
                    << " — a WIDER buffer is invisible to every tolerance in this file");
}

TEST_CASE("music3 ar: the depth arm SELECTION stages on a device queue and never on a CPU one") {
  // THE #1131 REPAIR. The engine used to decide this with an `if` on
  // `queue_.device.type`, and that condition is false on every runner CI owns —
  // so the branch was the one line no gate could enter, which is #1131's shape
  // reproduced by the change that cites #1131 as its reason for existing. The
  // rule now lives in `Music3SelectDepthArm`, which runs on BOTH sides of that
  // condition and is therefore drivable here.
  const m3::DepthDecoderConfig config = DeviceArmConfig();
  uint32_t state = 0x0BADC0DEu;
  const m3::DepthDecoderWeights depth = DeviceArmWeights(config, &state);

  SUBCASE("a CPU queue stages NOTHING and keeps the host reference arm") {
    vt::Queue queue = CpuQueue();
    m3::DepthDecoderWeights source = depth;
    m3::Music3DepthDeviceWeights staged;
    const uint64_t before = m3::Music3DepthDeviceForwardCount();
    const m3::Music3DepthDeviceArm arm =
        m3::Music3SelectDepthArm(queue, config, source, /*release_host=*/true, &staged);
    CHECK_FALSE(arm.engaged());
    CHECK_FALSE(arm.half_set());
    CHECK_FALSE(staged.staged());
    CHECK(m3::Music3DepthDeviceForwardCount() == before);
    // `release_host` was TRUE and NOTHING may have been released, because the
    // host loop is what this queue selected and the host loop reads these.
    REQUIRE(source.layers.size() == depth.layers.size());
    CHECK(source.pos_embedding.size() == depth.pos_embedding.size());
    CHECK(source.layers[0].to_q.size() == depth.layers[0].to_q.size());
    CHECK(source.layers[0].gate_proj.size() == depth.layers[0].gate_proj.size());
  }

  SUBCASE("EVERY non-CPU device stages or refuses, and never silently falls back") {
    // THE THIRD OUTCOME IS THE DEFECT. A device queue must come back with an
    // ENGAGED arm, or the staging must REFUSE by name because this build has no
    // provider for that device. What must not happen is a quiet return of a
    // disengaged arm, because that is a caller that asked for the device, was
    // given the host loop, and was told nothing — #1131 exactly.
    //
    // `kCUDA` is deliberately NOT in this list. On a CUDA build without a device
    // the staging would fail INSIDE the CUDA runtime, and a CUDA call designed
    // to fail latches a sticky error that the next unrelated kernel reports as
    // its own. Every entry below takes the identical branch, so the rule is
    // covered and the latch is not armed.
    constexpr vt::DeviceType kNonCpuDevices[] = {
        vt::DeviceType::kMETAL, vt::DeviceType::kVULKAN, vt::DeviceType::kXPU,
        vt::DeviceType::kROCM, vt::DeviceType::kTENSTORRENT};
    int engaged_count = 0;
    int refused_count = 0;
    for (vt::DeviceType type : kNonCpuDevices) {
      CAPTURE(vt::DeviceTypeName(type));
      vt::Queue queue{vt::Device{type, 0}, nullptr};
      m3::DepthDecoderWeights source = depth;
      m3::Music3DepthDeviceWeights staged;
      bool engaged = false;
      bool refused = false;
      try {
        const m3::Music3DepthDeviceArm arm =
            m3::Music3SelectDepthArm(queue, config, source, /*release_host=*/false, &staged);
        engaged = arm.engaged();
      } catch (const std::runtime_error&) {
        refused = true;
      }
      engaged_count += engaged ? 1 : 0;
      refused_count += refused ? 1 : 0;
      // Named, because doctest cannot decompose a `||` inside a CHECK.
      const bool selection_fired = engaged || refused;
      CHECK_MESSAGE(selection_fired, "device '"
                                            << vt::DeviceTypeName(type)
                                            << "' quietly took the HOST arm: the selection did "
                                               "not fire and nothing said so");
    }
    // The loop must have RUN. A list that emptied, or a body that threw before
    // the first CHECK, would leave every assertion above unexecuted.
    const int outcomes = engaged_count + refused_count;
    CHECK(outcomes == static_cast<int>(sizeof(kNonCpuDevices) / sizeof(kNonCpuDevices[0])));
    MESSAGE("non-CPU selection: " << engaged_count << " staged, " << refused_count
                                  << " refused by name");
  }

  SUBCASE("a null staging slot is refused rather than dereferenced") {
    vt::Queue queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
    m3::DepthDecoderWeights source = depth;
    CHECK_THROWS_AS(
        m3::Music3SelectDepthArm(queue, config, source, /*release_host=*/false, nullptr),
        std::runtime_error);
  }
}

TEST_CASE("music3 ar: the COMPOSED depth stage TAKES the device arm and draws the same codes") {
  // The reachability gate. It drives the PRODUCTION `Music3DepthStage` — the one
  // `Music3GenerateFrameHiddens` calls, which is how the registered speech family
  // reaches this row — with the device arm ENGAGED, and it asks the question
  // #1131 says nobody asked of the DiT arm: which arm ran?
  //
  // A numeric comparison alone cannot answer that, because the two arms agree by
  // design. So the counter is checked, and it is checked for an EXACT value
  // rather than for movement: `num_codebooks` calls a frame, which is §16.2's 8.
  const m3::DepthDecoderConfig config = DeviceArmConfig();
  uint32_t state = 0x51ED2701u;
  m3::DepthDecoderWeights depth = DeviceArmWeights(config, &state);
  const int64_t H = config.hidden_size;
  const size_t hidden = static_cast<size_t>(H);

  m3::Music3ArWeights weights;
  weights.depth_config = config;
  weights.depth = depth;
  weights.lm_config.hidden_size = H;
  const int64_t vocab = m3::kAudioCodeOffset + 64;
  weights.lm_config.vocab_size = vocab;
  {
    std::vector<uint8_t> bytes(static_cast<size_t>(vocab) * hidden * sizeof(uint16_t), 0);
    uint16_t* const rows = reinterpret_cast<uint16_t*>(bytes.data());
    const std::vector<float> table =
        DeviceArmFill(static_cast<size_t>(vocab - m3::kAudioCodeOffset) * hidden, &state, 0.5f);
    for (int64_t t = m3::kAudioCodeOffset; t < vocab; ++t) {
      for (int64_t j = 0; j < H; ++j) {
        rows[t * H + j] =
            vt::F32ToBF16(table[static_cast<size_t>((t - m3::kAudioCodeOffset) * H + j)]);
      }
    }
    weights.lm.embed_tokens.bytes = vllm::OwnedBytes(std::move(bytes));
    weights.lm.embed_tokens.dtype = vt::DType::kBF16;
    weights.lm.embed_tokens.rank = 2;
    weights.lm.embed_tokens.shape[0] = vocab;
    weights.lm.embed_tokens.shape[1] = H;
  }

  const int32_t semantic_code = 11;
  const int64_t frame_index = 2;
  const std::vector<float> last_conditional = DeviceArmFill(hidden, &state, 1.0f);
  const std::vector<float> last_unconditional = DeviceArmFill(hidden, &state, 1.0f);

  int64_t host_draws = 0;
  int64_t device_draws = 0;
  // Each draw records the code it took AND the probability vector it took it
  // from. The code comparison below needs the vector: an argmax whose top two
  // candidates sit closer together than the two arms' own arithmetic resolution
  // is not a property either arm can be held to, and the only way to say that
  // without asserting a coincidence is to ask what the OTHER arm's pick was
  // worth in this arm's own distribution. See #1458.
  struct DrawRecord {
    int64_t code = 0;
    std::vector<float> probs;
    double best() const { return static_cast<double>(probs[static_cast<size_t>(code)]); }
    double runner_up() const {
      double r = -std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < probs.size(); ++i)
        if (static_cast<int64_t>(i) != code) r = std::max(r, static_cast<double>(probs[i]));
      return r;
    }
  };
  std::vector<DrawRecord> host_rec;
  std::vector<DrawRecord> device_rec;
  const auto make_sampler = [](int64_t* counter, std::vector<DrawRecord>* rec) {
    return m3::Music3CodeSampler(
        [counter, rec](const std::vector<float>& probs, const m3::Music3Draw&) -> int64_t {
          ++*counter;
          size_t best = 0;
          for (size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > probs[best]) best = i;
          }
          DrawRecord r;
          r.code = static_cast<int64_t>(best);
          r.probs = probs;
          rec->push_back(std::move(r));
          return static_cast<int64_t>(best);
        });
  };

  // The HOST arm, through the same production entry point, with a
  // default-constructed arm — which is what every existing caller passes.
  std::vector<int32_t> host_codes{semantic_code};
  const std::vector<float> host = m3::Music3DepthStage(
      last_conditional, last_unconditional, frame_index, weights,
      make_sampler(&host_draws, &host_rec), &host_codes);

  vt::Queue queue = CpuQueue();
  m3::DepthDecoderWeights stage_source = depth;
  const m3::Music3DepthDeviceWeights staged =
      m3::StageMusic3DepthWeights(queue, config, stage_source, /*release_host=*/false);
  m3::Music3DepthDeviceArm arm;
  arm.queue = &queue;
  arm.depth = &staged;
  REQUIRE(arm.engaged());
  REQUIRE_FALSE(arm.half_set());

  const uint64_t before = m3::Music3DepthDeviceForwardCount();
  std::vector<int32_t> device_codes{semantic_code};
  const std::vector<float> device =
      m3::Music3DepthStage(last_conditional, last_unconditional, frame_index, weights,
                           make_sampler(&device_draws, &device_rec), &device_codes, arm);
  const uint64_t after = m3::Music3DepthDeviceForwardCount();

  // THE ASSERTION #1131 SAYS IS MISSING. `num_codebooks` appends a frame: one
  // for the batch-2 prefix at position 0, then one per residual codebook step.
  CHECK_MESSAGE(after - before == static_cast<uint64_t>(config.num_codebooks),
                "the composed stage ran the DEVICE forward " << (after - before)
                                                             << " times, expected "
                                                             << config.num_codebooks);
  REQUIRE(device.size() == host.size());
  double worst_ulp = 0.0;
  size_t nonzero = 0;
  for (size_t i = 0; i < host.size(); ++i) {
    worst_ulp = std::max(worst_ulp,
                         std::abs(static_cast<double>(device[i]) - host[i]) / Bf16Ulp(host[i]));
    if (host[i] != 0.0f) ++nonzero;
  }
  CHECK_MESSAGE(worst_ulp <= 512.0, "composed device/host worst deviation "
                                        << worst_ulp << " bf16 ULP exceeds 512, which is a "
                                                        "STRUCTURAL defect, not rounding");
  // A tolerance CANNOT see a dropped stage — the prefix append, the fed-back
  // projection row — because a schedule missing one still produces finite,
  // plausible numbers. The drawn codes can.
  //
  // WHAT THE CODES CANNOT DECIDE, and why this is a per-draw comparison rather
  // than one `==`. The two arms are two implementations of the same block, not
  // two runs of one: the device arm goes through `vt` GEMMs and the host arm
  // through the `ArCompute::kBFloat16` reference, and this suite measures their
  // separation at 174.5 to 6924 bf16 ULP across its device/host comparisons —
  // 308 in this case. Both store at bf16, so their probabilities agree to about
  // one bf16 unit roundoff: 2^-8, the largest relative error a store at that
  // width can carry. An argmax whose top two candidates are separated by LESS
  // than that is decided by rounding, and holding the two arms to the same code
  // there asserts a coincidence, not a guarantee.
  //
  // MEASURED on this fixture at `aeba0de6f`: draws 0..5 have relative top-2
  // margins of 1.74e-02 to 3.53e-01 — 4.5x to 90x the resolution — and both arms
  // agree at every one. Draw 6 is a tie at 1.95e-03 (device) and 2.93e-03
  // (host), BELOW one unit roundoff, between codes 2 and 17, and each arm
  // ranks the other's pick inside it. It flipped when `4712dac40` gave
  // `vt`'s gated activations upstream's rounding polarity, which is the correct
  // polarity and the only one that reproduces the committed
  // `silu_and_mul_bf16_8x256` oracle golden bit-exactly. The blanket `==` was
  // reading a coin. #1458.
  //
  // A dropped stage is NOT let through: it moves the distributions by O(1), so
  // the arms stop agreeing about which candidates are tied at all. MEASURED —
  // deleting the MLP residual add in `minimax_music3_depth_device.cpp` diverges
  // at draw 1 with the DEVICE's pick 1.60e-01 below the host's own best, 41x the
  // unit roundoff this admits, and `shared_tie` is RED.
  // 2^-8 is bf16's UNIT ROUNDOFF -- the largest relative error a correctly
  // rounded bf16 store can carry. It is NOT the format's relative spacing
  // (machine epsilon), which is twice it at 2^-7, and it is not the `Bf16Ulp`
  // helper this file uses for its value metric either. The unit roundoff is what
  // bounds a stored probability's departure, which is what this comparison wants.
  constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
  REQUIRE(host_rec.size() == device_rec.size());
  REQUIRE(host_codes.size() == device_codes.size());
  REQUIRE(host_rec.size() + 1 == host_codes.size());
  int64_t compared = 0;
  bool diverged = false;
  for (size_t i = 0; i < host_rec.size() && !diverged; ++i) {
    ++compared;
    const DrawRecord& h = host_rec[i];
    const DrawRecord& d = device_rec[i];
    if (h.code == d.code) continue;
    // Each arm is about to be indexed with the OTHER arm's argmax, so the two
    // heads must agree on the vocabulary width before that read happens. A
    // defect that changed the device head's output width would otherwise be
    // undefined behaviour here, and NDEBUG is the configuration this runs in.
    REQUIRE(h.probs.size() == d.probs.size());
    REQUIRE(h.code >= 0);
    REQUIRE(d.code >= 0);
    REQUIRE(static_cast<size_t>(h.code) < h.probs.size());
    REQUIRE(static_cast<size_t>(d.code) < d.probs.size());
    // A divergence is admissible ONLY as a shared near-tie: each arm must rank
    // the OTHER arm's pick within one bf16 unit roundoff of its own. That is not a
    // tolerance anybody chose — it is the width the two arms store at, so a gap
    // narrower than it is decided by rounding and not by the model. If either
    // arm ranked the other's pick BELOW that, the two arms disagree about the
    // distribution and not merely about a tie, and this fires.
    const double h_gap = (h.best() - static_cast<double>(h.probs[static_cast<size_t>(d.code)])) /
                         std::max(h.best(), 1e-30);
    const double d_gap = (d.best() - static_cast<double>(d.probs[static_cast<size_t>(h.code)])) /
                         std::max(d.best(), 1e-30);
    const bool shared_tie = (h_gap <= kBf16UnitRoundoff) && (d_gap <= kBf16UnitRoundoff);
    CHECK_MESSAGE(shared_tie,
                  "draw " << i << ": the two arms drew different codes (host " << h.code
                          << ", device " << d.code
                          << ") and it is NOT a shared tie — in the host's own distribution "
                             "the device's pick is "
                          << h_gap << " below its best, and in the device's the host's pick is "
                          << d_gap << " below its best, against one bf16 unit roundoff " << kBf16UnitRoundoff);
    if (shared_tie)
      MESSAGE("draw " << i << ": shared near-tie, host drew " << h.code << " (runner-up gap "
                    << ((h.best() - h.runner_up()) / h.best()) << "), device drew " << d.code
                    << " (runner-up gap " << ((d.best() - d.runner_up()) / d.best())
                    << "), both inside one bf16 unit roundoff " << kBf16UnitRoundoff
                    << ". Every later draw is fed a different code, so the comparison stops "
                       "here.");
    diverged = true;
  }
  CHECK_MESSAGE(compared > 0, "no draw was compared, so the code comparison gated nothing");
  CHECK(device_draws == host_draws);
  CHECK(device_draws == config.residual_codebooks());
  CHECK(static_cast<int64_t>(device_codes.size()) == config.num_codebooks);
  CHECK_MESSAGE(nonzero > 0, "the reference block is all zeros, so nothing is comparable");
  CHECK_MESSAGE(device_draws > 0, "no code was drawn, so the draw order is untested");
  MESSAGE("composed device stage: " << host.size() << " values, worst " << worst_ulp
                                    << " bf16 ULP, " << (after - before) << " device forwards, "
                                    << device_draws << " draws, " << nonzero << " of "
                                    << host.size() << " reference values non-zero");
}

TEST_CASE("music3 ar: the device depth arm refuses what it cannot serve, by name") {
  const m3::DepthDecoderConfig config = DeviceArmConfig();
  uint32_t state = 0x1234567u;
  m3::DepthDecoderWeights depth = DeviceArmWeights(config, &state);
  vt::Queue queue = CpuQueue();
  m3::DepthDecoderWeights stage_source = depth;
  const m3::Music3DepthDeviceWeights staged =
      m3::StageMusic3DepthWeights(queue, config, stage_source, /*release_host=*/false);
  const int64_t H = config.hidden_size;
  const std::vector<float> row = DeviceArmFill(static_cast<size_t>(2 * H), &state, 1.0f);

  SUBCASE("a null cache, a non-positive batch and a mis-sized input") {
    m3::Music3DepthDeviceCache cache;
    CHECK_THROWS_AS(m3::DepthDecoderAppendDevice(queue, config, staged, row, 2, nullptr),
                    std::runtime_error);
    CHECK_THROWS_AS(m3::DepthDecoderAppendDevice(queue, config, staged, row, 0, &cache),
                    std::runtime_error);
    CHECK_THROWS_AS(m3::DepthDecoderAppendDevice(queue, config, staged, row, 3, &cache),
                    std::runtime_error);
  }
  SUBCASE("unstaged weights are refused rather than dereferenced") {
    m3::Music3DepthDeviceCache cache;
    const m3::Music3DepthDeviceWeights empty;
    CHECK_FALSE(empty.staged());
    CHECK_THROWS_AS(m3::DepthDecoderAppendDevice(queue, config, empty, row, 2, &cache),
                    std::runtime_error);
  }
  SUBCASE("the batch cannot change mid-cache, and the position ceiling binds") {
    m3::Music3DepthDeviceCache cache;
    m3::DepthDecoderAppendDevice(queue, config, staged, row, 2, &cache);
    const std::vector<float> one = DeviceArmFill(static_cast<size_t>(H), &state, 1.0f);
    CHECK_THROWS_AS(m3::DepthDecoderAppendDevice(queue, config, staged, one, 1, &cache),
                    std::runtime_error);
    for (int64_t p = 1; p < config.max_position_embeddings; ++p) {
      m3::DepthDecoderAppendDevice(queue, config, staged, row, 2, &cache);
    }
    CHECK(cache.positions == config.max_position_embeddings);
    CHECK_THROWS_AS(m3::DepthDecoderAppendDevice(queue, config, staged, row, 2, &cache),
                    std::runtime_error);
  }
  SUBCASE("a mis-sized weight is refused AT STAGE TIME, naming the tensor") {
    m3::DepthDecoderWeights broken = depth;
    broken.layers[0].gate_proj.pop_back();
    CHECK_THROWS_AS(
        m3::StageMusic3DepthWeights(queue, config, broken, /*release_host=*/false),
        std::runtime_error);
  }
  SUBCASE("HALF an arm is refused at the composed stage rather than silently ignored") {
    // A caller that set one field thinks it asked for the device and did not.
    // Falling back to the host loop would be a 4.4x slowdown wearing a correct
    // answer, so it is a refusal.
    m3::Music3ArWeights weights;
    weights.depth_config = config;
    weights.depth = depth;
    weights.lm_config.hidden_size = H;
    weights.lm_config.vocab_size = m3::kAudioCodeOffset + 8;
    m3::Music3DepthDeviceArm queue_only;
    queue_only.queue = &queue;
    CHECK(queue_only.half_set());
    std::vector<int32_t> codes{1};
    const std::vector<float> last(static_cast<size_t>(H), 0.5f);
    const m3::Music3CodeSampler sampler =
        [](const std::vector<float>&, const m3::Music3Draw&) -> int64_t { return 0; };
    CHECK_THROWS_AS(m3::Music3DepthStage(last, last, 0, weights, sampler, &codes, queue_only),
                    std::runtime_error);
    m3::Music3DepthDeviceArm weights_only;
    weights_only.depth = &staged;
    CHECK(weights_only.half_set());
    std::vector<int32_t> codes2{1};
    CHECK_THROWS_AS(m3::Music3DepthStage(last, last, 0, weights, sampler, &codes2, weights_only),
                    std::runtime_error);
  }
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
