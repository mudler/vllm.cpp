// The render entry point, and the whole pipeline on real weights (#634).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/bigvgan_loader.h"
#include "vllm/model_executor/models/indextts2_config.h"
#include "vllm/model_executor/models/indextts2_render.h"
#include "vllm/model_executor/models/indextts2_s2mel_loader.h"
#include "vllm/model_executor/models/indextts2_talker_loader.h"
#include "vllm/model_executor/models/talker.h"

namespace ix = vllm::models::indextts2;

TEST_CASE("the rotary table is built, not zeroed") {
  // A zeroed table makes every position identical: the model still integrates
  // to a mel and still renders, which is exactly why a stand-in reads as a
  // working pipeline. cos(0) is 1, so position 0 is all (1, 0) pairs and later
  // positions must DIFFER from it.
  const std::vector<float> f = ix::RotaryTable(4, 8, 10000.0);
  REQUIRE(f.size() == 4u * 4u * 2u);
  for (int k = 0; k < 4; ++k) {
    CHECK(f[static_cast<size_t>(k * 2)] == doctest::Approx(1.0));      // cos 0
    CHECK(f[static_cast<size_t>(k * 2 + 1)] == doctest::Approx(0.0));  // sin 0
  }
  bool later_differs = false;
  for (size_t i = 8; i < f.size(); ++i) {
    if (std::fabs(f[i] - f[i % 8]) > 1e-6F) {
      later_differs = true;
    }
  }
  CHECK(later_differs);
  // The lowest frequency band turns fastest: theta = pos / base^(2k/head_dim),
  // so k = 0 has inv 1 and its angle at position 1 is exactly 1 radian.
  CHECK(f[8] == doctest::Approx(std::cos(1.0)).epsilon(1e-6));
  CHECK(f[9] == doctest::Approx(std::sin(1.0)).epsilon(1e-6));
}

TEST_CASE("an odd head_dim is refused rather than half-rotated") {
  CHECK_THROWS(ix::RotaryTable(4, 7, 10000.0));
  CHECK_THROWS(ix::RotaryTable(0, 8, 10000.0));
}

TEST_CASE("the WHOLE pipeline renders from the talker's OWN codes") {
  const char* s2 = std::getenv("VLLM_CPP_INDEXTTS2_S2MEL");
  const char* bv = std::getenv("VLLM_CPP_INDEXTTS2_BIGVGAN");
  const char* gp = std::getenv("VLLM_CPP_INDEXTTS2_GPT");
  if (s2 == nullptr || bv == nullptr || gp == nullptr) {
    MESSAGE("SKIPPED: set VLLM_CPP_INDEXTTS2_S2MEL, _BIGVGAN and _GPT to render; "
            "without all three NOTHING was rendered");
    return;
  }
  const auto file = vllm::SafetensorsFile::Open(std::string(s2));
  auto est = ix::LoadS2MelEstimator(file, 8);
  ix::RenderStages st;
  st.regulator = vllm::models::lenreg::LoadRegulator(file, &st.regulator_config);
  st.front_config = est.front_config;
  st.front = est.front;
  st.stack_config = est.stack_config;
  st.stack = est.stack;
  st.tail_config = est.tail_config;
  st.tail = est.tail;
  const auto vocoder = vllm::models::bigvgan::Load(std::string(bv));
  st.vocoder_config = vocoder.config;
  st.vocoder = vocoder.weights;
  const auto talker = ix::LoadTalker(std::string(gp), ix::kTalkerHeads);

  // 1. The talker emits its OWN codes.
  const int64_t D = talker.params.hidden_size;
  std::vector<float> conds(static_cast<size_t>(3 * D), 0.0F);
  for (int64_t d = 0; d < D; ++d) {
    conds[static_cast<size_t>(d)] = 0.01F * std::sin(0.05F * static_cast<float>(d));
  }
  vllm::models::talker::PromptConfig pc;
  pc.dim = D;
  pc.start_text_token = 0;
  pc.stop_text_token = 1;
  pc.start_mel_token = ix::kStartMelToken;
  pc.text_slots = 8;
  vllm::models::talker::PromptWeights pw;
  pw.text_embedding = talker.text_embedding;
  pw.text_pos_embedding = talker.text_pos_embedding;
  pw.lang_embedding = talker.lang_embedding;
  const auto prompt =
      vllm::models::talker::PrepareInputs(pc, pw, conds, 3, {5, 42, 7, 90}, 0);

  vllm::models::talker::GenerateConfig gc;
  gc.dim = D;
  gc.mel_codes = ix::kNumberMelCodes;
  gc.start_mel_token = ix::kStartMelToken;
  gc.stop_mel_token = ix::kStopMelToken;
  gc.max_mel_tokens = 6;
  vllm::models::talker::GenerateWeights gw;
  gw.mel_embedding = talker.mel_embedding;
  gw.mel_pos_embedding = talker.mel_pos_embedding;
  gw.final_norm_w = talker.final_norm_w;
  gw.final_norm_b = talker.final_norm_b;
  gw.mel_head_w = talker.mel_head_w;
  gw.mel_head_b = talker.mel_head_b;
  const auto codes = vllm::models::talker::GenerateMelCodes(
      gc, gw, talker.params, talker.backbone, prompt.embeds, prompt.target_len);

  REQUIRE(!codes.empty());
  // Every code must be a LEGAL mel code, and they must not all be the same --
  // a talker stuck on one token would still render.
  bool varied = false;
  for (const int64_t c : codes) {
    CHECK(c >= 0);
    CHECK(c < ix::kNumberMelCodes);
    CHECK(c != ix::kStopMelToken);
    if (c != codes[0]) {
      varied = true;
    }
  }
  CHECK(varied);

  // 2. Those codes become the content the regulator consumes.
  const int64_t IN = st.regulator_config.in_channels;
  const int64_t cf = static_cast<int64_t>(codes.size());
  std::vector<float> content(static_cast<size_t>(cf * IN));
  for (int64_t f = 0; f < cf; ++f) {
    for (int64_t i = 0; i < IN; ++i) {
      content[static_cast<size_t>(f * IN + i)] =
          talker.mel_embedding[static_cast<size_t>(codes[f] * D + (i % D))] * 0.1F;
    }
  }

  // 3. Render.
  ix::RenderConfig rc;
  rc.mel_frames = cf * 4;
  rc.steps = 4;
  rc.cfg_rate = 0.7;
  std::vector<float> noise(
      static_cast<size_t>(st.front_config.in_channels * rc.mel_frames));
  for (size_t i = 0; i < noise.size(); ++i) {
    noise[i] = 0.01F * std::sin(0.7F * static_cast<float>(i));
  }
  const std::vector<float> style(static_cast<size_t>(st.front_config.style), 0.02F);
  const auto wave = ix::Render(rc, st, content, cf, style, noise);

  REQUIRE(wave.size() == static_cast<size_t>(rc.mel_frames * 256));
  float lo = wave[0];
  float hi = wave[0];
  double energy = 0.0;
  for (const float v : wave) {
    REQUIRE(std::isfinite(v));
    CHECK(v >= -1.0F);
    CHECK(v <= 1.0F);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    energy += static_cast<double>(v) * static_cast<double>(v);
  }
  const double rms = std::sqrt(energy / static_cast<double>(wave.size()));
  CHECK(rms > 1e-3);  // not silence
  CHECK(hi > lo);     // not a rail
  MESSAGE("RENDERED from " << codes.size() << " talker codes: " << wave.size()
          << " samples (" << static_cast<double>(wave.size()) / 22050.0
          << " s at 22050 Hz), rms " << rms);
}
