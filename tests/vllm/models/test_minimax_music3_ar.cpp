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
