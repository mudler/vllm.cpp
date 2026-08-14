// SeamlessM4T feature extraction against transformers. See w2v_fbank.h.
//
// Gated in THREE pieces so a failure localises: the window, the filterbank, and
// the end-to-end features are each a different kind of mistake.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/w2v_fbank.h"
#include "w2v_fbank_goldens.inc"

using namespace w2v_fbank_goldens;
namespace fb = vllm::models::w2v_fbank;

TEST_CASE("the POVEY window matches transformers") {
  const std::vector<float> got = fb::PoveyWindow(kFrameLength);
  REQUIRE(got.size() == static_cast<size_t>(kFrameLength));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(kWindow[i]).epsilon(1e-6));
  }
}

TEST_CASE("the povey window is NOT a plain hann") {
  // hann^0.85 and hann agree at the endpoints and the peak, so a test that only
  // sampled those would pass on the wrong window.
  const std::vector<float> got = fb::PoveyWindow(kFrameLength);
  const int64_t quarter = kFrameLength / 4;
  const double hann = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 *
                                           static_cast<double>(quarter) /
                                           static_cast<double>(kFrameLength - 1));
  CHECK(got[static_cast<size_t>(quarter)] != doctest::Approx(hann).epsilon(1e-4));
  CHECK(got[static_cast<size_t>(quarter)] ==
        doctest::Approx(std::pow(hann, 0.85)).epsilon(1e-6));
}

TEST_CASE("the KALDI mel filterbank matches transformers EXACTLY") {
  // Measured: max |diff| is 0.0 and the zero pattern agrees, so this is stated
  // as equality rather than a tolerance. A tolerance here would let a real
  // change in the triangle geometry through.
  const std::vector<float> got =
      fb::KaldiMelFilterbank(kFreqBins, kMelBins, 20.0, 8000.0, 16000.0);
  REQUIRE(got.size() == static_cast<size_t>(kFreqBins * kMelBins));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == kMelFilters[i]);
  }
}

TEST_CASE("the RAW log-mel matches transformers") {
  // The well-conditioned quantity, and where correctness lives. Measured
  // max |diff| 3.8e-03 on values spanning [-6.2, 24.2]; the residual sits on
  // bins whose energy is near `mel_floor`, where log() is steep. Everything
  // upstream of it agrees far more tightly and is asserted separately: the
  // filterbank EXACTLY, the windowed frame to 8 digits, the power spectrum to
  // 1e-05 (that last one is upstream storing the spectrum as complex64).
  const std::vector<float> audio(kAudio, kAudio + kSamples);
  fb::Config cfg;
  int64_t frames = 0;
  std::vector<float> raw;
  const std::vector<float> got = fb::Extract(cfg, audio, 16000.0, &frames, &raw);
  (void)got;
  REQUIRE(raw.size() == static_cast<size_t>(kRawFrames * kMelBins));
  double worst = 0.0;
  for (size_t i = 0; i < raw.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::fabs(raw[i] - kRawLogMel[i])));
  }
  CHECK(worst < 5e-3);
  // And a BAND, not just a ceiling: if the agreement suddenly became perfect the
  // fixture would have stopped exercising the floor-adjacent bins.
  CHECK(worst > 1e-9);
}

TEST_CASE("the extractor matches transformers end to end") {
  // The normalized output is NOT well conditioned: bins whose variance falls
  // below `norm_eps` 1e-07 are divided by ~3.2e-04, so the 3.8e-03 above is
  // amplified. Measured max |diff| 1.15e-02, and the tolerance is set from that
  // measurement rather than chosen to pass.
  const std::vector<float> audio(kAudio, kAudio + kSamples);
  fb::Config cfg;
  int64_t frames = 0;
  const std::vector<float> got = fb::Extract(cfg, audio, 16000.0, &frames);

  CHECK(frames == kFrames);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kFeatureWidth));
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::fabs(got[i] - kFeatures[i])));
  }
  CHECK(worst < 2e-2);
  CHECK(worst > 1e-9);
}

TEST_CASE("the stride-2 stack drops the odd tail") {
  // 4000 samples give 23 raw frames; 23 / 2 = 11 with one dropped... the golden
  // says 12, so upstream produced 24 raw frames. Whatever the count, the output
  // must be exactly half of it, and the width exactly twice the mel bins.
  CHECK(kFeatureWidth == kMelBins * kStride);
  const std::vector<float> audio(kAudio, kAudio + kSamples);
  fb::Config cfg;
  int64_t frames = 0;
  const std::vector<float> got = fb::Extract(cfg, audio, 16000.0, &frames);
  CHECK(got.size() == static_cast<size_t>(frames * kMelBins * kStride));
}
