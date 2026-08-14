// IndexTTS-2.5 pipeline composition (#634).
//
// STRUCTURAL, not numerical. Every stage's arithmetic is gated against upstream
// in its own suite; this proves the stages CONNECT -- that each output shape is
// the next input shape, that a change at the front reaches the back, and that a
// mismatch throws BY STAGE rather than reshaping silently.
//
// It is not a quality result, not a parity result and not a render.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/indextts2_pipeline.h"

namespace {
vllm::models::indextts2::PipelineDims Dims() {
  vllm::models::indextts2::PipelineDims d;
  d.ref_frames = 9;
  d.semantic_dim = 6;
  d.codec_dim = 6;
  d.codebook_dim = 4;
  d.codebook_size = 16;
  d.style_feat_dim = 8;
  d.style_dim = 5;
  d.mel_frames = 20;     // deliberately NOT a multiple of ref_frames
  d.mel_channels = 7;
  d.talker_dim = 6;
  d.talker_vocab = 12;
  return d;
}
std::vector<float> Clip(const vllm::models::indextts2::PipelineDims& d, double bias) {
  std::vector<float> v(static_cast<size_t>(d.ref_frames * d.semantic_dim));
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<float>(std::sin(0.7 * static_cast<double>(i)) + bias);
  }
  return v;
}
}  // namespace

TEST_CASE("indextts2 pipeline composes all six stages end to end") {
  const auto d = Dims();
  const std::vector<int64_t> text{3, 7, 1, 0, 11};
  const auto r = vllm::models::indextts2::RunReduced(d, Clip(d, 0.0), text, 1234);

  CHECK(static_cast<int64_t>(r.semantic_codes.size()) == d.ref_frames);
  for (const int64_t c : r.semantic_codes) {
    CHECK(c >= 0);
    CHECK(c < d.codebook_size);
  }
  CHECK(static_cast<int64_t>(r.style.size()) == d.style_dim);
  CHECK(static_cast<int64_t>(r.prompt_condition.size()) == d.codec_dim * d.mel_frames);
  CHECK(static_cast<int64_t>(r.mel.size()) == d.mel_channels * d.mel_frames);
  CHECK(static_cast<int64_t>(r.waveform.size()) == d.mel_channels * d.mel_frames);
  CHECK(r.sample_rate == 22050);
  for (const float v : r.waveform) CHECK(std::isfinite(v));
}

TEST_CASE("indextts2 pipeline propagates a change in the reference clip to the waveform") {
  // The point of a composition gate: a different reference must produce a
  // different waveform. A stage wired to a constant, or dropped, breaks this
  // while every shape assertion still passes.
  const auto d = Dims();
  const std::vector<int64_t> text{3, 7, 1, 0, 11};
  const auto a = vllm::models::indextts2::RunReduced(d, Clip(d, 0.0), text, 1234);
  const auto b = vllm::models::indextts2::RunReduced(d, Clip(d, 0.9), text, 1234);

  double moved = 0.0;
  for (size_t i = 0; i < a.waveform.size(); ++i) {
    moved = std::max(moved, std::fabs(static_cast<double>(a.waveform[i] - b.waveform[i])));
  }
  CHECK(moved > 1e-4);
}

TEST_CASE("indextts2 pipeline propagates a change in the TEXT to the waveform") {
  // The talker's contribution must reach the output too -- otherwise the lane
  // would render the same audio for every utterance.
  const auto d = Dims();
  const auto a = vllm::models::indextts2::RunReduced(d, Clip(d, 0.0), {3, 7, 1, 0, 11}, 1234);
  const auto b = vllm::models::indextts2::RunReduced(d, Clip(d, 0.0), {2, 2, 9, 4, 5}, 1234);
  double moved = 0.0;
  for (size_t i = 0; i < a.waveform.size(); ++i) {
    moved = std::max(moved, std::fabs(static_cast<double>(a.waveform[i] - b.waveform[i])));
  }
  CHECK(moved > 1e-4);
}

TEST_CASE("indextts2 pipeline is deterministic for one seed") {
  const auto d = Dims();
  const std::vector<int64_t> text{3, 7, 1, 0, 11};
  const auto a = vllm::models::indextts2::RunReduced(d, Clip(d, 0.0), text, 99);
  const auto b = vllm::models::indextts2::RunReduced(d, Clip(d, 0.0), text, 99);
  REQUIRE(a.waveform.size() == b.waveform.size());
  for (size_t i = 0; i < a.waveform.size(); ++i) CHECK(a.waveform[i] == b.waveform[i]);
  CHECK(a.semantic_codes == b.semantic_codes);
}

TEST_CASE("indextts2 pipeline refuses a mismatched reference clip BY STAGE") {
  // A silent reshape between stages is how a pipeline ends up rendering the
  // wrong tensor, so the refusal names the stage.
  const auto d = Dims();
  const std::vector<float> wrong(static_cast<size_t>(d.ref_frames * d.semantic_dim + 3), 0.1F);
  CHECK_THROWS_WITH_AS(vllm::models::indextts2::RunReduced(d, wrong, {1, 2}, 7),
                       doctest::Contains("stage 1"), std::runtime_error);
}

TEST_CASE("indextts2 pipeline refuses empty text BY STAGE") {
  const auto d = Dims();
  CHECK_THROWS_WITH_AS(vllm::models::indextts2::RunReduced(d, Clip(d, 0.0), {}, 7),
                       doctest::Contains("stage 5"), std::runtime_error);
}

TEST_CASE("indextts2 pipeline: the PROMPT CONDITION reaches the waveform") {
  // Found by mutation: wiring the condition to a constant passed every earlier
  // case, because the STYLE path alone kept the waveform moving. This isolates
  // the condition path.
  //
  // A time-PERMUTED clip has identical per-channel mean and std, so StatsPool
  // gives the SAME style vector -- while the quantized sequence, and therefore
  // the prompt condition, is reordered. Any difference in the waveform must
  // therefore have arrived through the condition.
  const auto d = Dims();
  const std::vector<int64_t> text{3, 7, 1, 0, 11};
  const std::vector<float> clip = Clip(d, 0.0);

  // A ROTATION, not a reversal. This fixture's codes happen to be palindromic
  // (8 11 8 8 11 8 8 11 8), so reversing is a genuine no-op and the case would
  // fail for a reason that has nothing to do with the pipeline. A cyclic shift
  // preserves the per-channel statistics just as exactly and does reorder.
  std::vector<float> permuted(clip.size());
  for (int64_t t = 0; t < d.ref_frames; ++t) {
    const int64_t src = (t + 1) % d.ref_frames;
    for (int64_t c = 0; c < d.semantic_dim; ++c) {
      permuted[static_cast<size_t>(t * d.semantic_dim + c)] =
          clip[static_cast<size_t>(src * d.semantic_dim + c)];
    }
  }

  const auto a = vllm::models::indextts2::RunReduced(d, clip, text, 1234);
  const auto b = vllm::models::indextts2::RunReduced(d, permuted, text, 1234);

  // The style must be (near) identical -- that is what makes this an isolation.
  double style_delta = 0.0;
  for (size_t i = 0; i < a.style.size(); ++i) {
    style_delta = std::max(style_delta, std::fabs(static_cast<double>(a.style[i] - b.style[i])));
  }
  CHECK(style_delta < 1e-5);

  // ...and the waveform must still move, which it can only do via the condition.
  double moved = 0.0;
  for (size_t i = 0; i < a.waveform.size(); ++i) {
    moved = std::max(moved, std::fabs(static_cast<double>(a.waveform[i] - b.waveform[i])));
  }
  CHECK(moved > 1e-4);
}

TEST_CASE("indextts2 pipeline: the prompt condition comes from the QUANTIZED features") {
  // Found by mutation: regulating the RAW features instead of the codec output
  // passed everything, since both depend on the clip. Nearest interpolation
  // COPIES samples, so every prompt-condition value must appear verbatim in the
  // quantized tensor -- which raw features would not satisfy.
  const auto d = Dims();
  const auto r = vllm::models::indextts2::RunReduced(d, Clip(d, 0.0), {3, 7, 1, 0, 11}, 1234);
  REQUIRE(static_cast<int64_t>(r.quantized.size()) == d.codec_dim * d.ref_frames);
  for (int64_t c = 0; c < d.codec_dim; ++c) {
    for (int64_t t = 0; t < d.mel_frames; ++t) {
      const float v = r.prompt_condition[static_cast<size_t>(c * d.mel_frames + t)];
      bool found = false;
      for (int64_t s = 0; s < d.ref_frames && !found; ++s) {
        found = (r.quantized[static_cast<size_t>(c * d.ref_frames + s)] == v);
      }
      CHECK(found);
    }
  }
}
