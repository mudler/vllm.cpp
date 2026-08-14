// TEXT to AUDIO, end to end, on the real shipped checkpoints (#634).
//
// The one gate that exercises every stage this lane built: tokenize with the
// checkpoint's own tiktoken vocabulary, run the talker to mel codes, resample
// them through the length regulator, integrate the S2Mel estimator under
// classifier-free guidance, and vocode to samples.
//
// It runs only with all four checkpoint paths set, and SKIPS LOUDLY otherwise,
// so "no audio was produced" can never read as "the render was fine".
//
// WHAT IT DOES NOT CLAIM: correctness. vLLM-Omni is unpinned (#633), so there is
// no oracle to compare against and nothing here is a quality statement. Every
// assertion below is about STRUCTURE -- that each stage consumed the previous
// one's real output and that the waveform is bounded, varied and not silence.
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
#include "vllm/model_executor/models/tiktoken_bpe.h"

namespace ix = vllm::models::indextts2;
namespace tk = vllm::models::tiktoken;

TEST_CASE("TEXT becomes AUDIO through every ported stage") {
  const char* s2p = std::getenv("VLLM_CPP_INDEXTTS2_S2MEL");
  const char* bvp = std::getenv("VLLM_CPP_INDEXTTS2_BIGVGAN");
  const char* gpp = std::getenv("VLLM_CPP_INDEXTTS2_GPT");
  const char* tkp = std::getenv("VLLM_CPP_INDEXTTS2_TIKTOKEN");
  if (s2p == nullptr || bvp == nullptr || gpp == nullptr || tkp == nullptr) {
    MESSAGE("SKIPPED: needs VLLM_CPP_INDEXTTS2_S2MEL, _BIGVGAN, _GPT and "
            "_TIKTOKEN. NO audio was produced.");
    return;
  }

  // 1. TOKENIZE with the checkpoint's own vocabulary.
  const auto ranks = tk::LoadRanks(std::string(tkp));
  bool exact = false;
  const auto ids = tk::Encode("hello world", ranks, &exact);
  REQUIRE(!ids.empty());
  CHECK(exact);
  for (const int64_t id : ids) {
    CHECK(id >= 0);
    CHECK(id < ix::kNumberTextTokens + 1);  // the checkpoint's table is one wider
  }

  // 2. TALKER: real text ids in, real mel codes out.
  const auto talker = ix::LoadTalker(std::string(gpp), ix::kTalkerHeads);
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
  pc.text_slots = static_cast<int64_t>(ids.size()) + 4;
  vllm::models::talker::PromptWeights pw;
  pw.text_embedding = talker.text_embedding;
  pw.text_pos_embedding = talker.text_pos_embedding;
  pw.lang_embedding = talker.lang_embedding;
  const auto prompt = vllm::models::talker::PrepareInputs(pc, pw, conds, 3, ids, 0);
  // The prompt must actually CONTAIN the text: conditioning rows plus the
  // delimited ids plus padding.
  CHECK(prompt.target_len == 3 + pc.text_slots + 2);

  vllm::models::talker::GenerateConfig gc;
  gc.dim = D;
  gc.mel_codes = ix::kNumberMelCodes;
  gc.start_mel_token = ix::kStartMelToken;
  gc.stop_mel_token = ix::kStopMelToken;
  gc.max_mel_tokens = 5;
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
  bool codes_vary = false;
  for (const int64_t c : codes) {
    CHECK(c >= 0);
    CHECK(c < ix::kNumberMelCodes);
    CHECK(c != ix::kStopMelToken);
    if (c != codes[0]) {
      codes_vary = true;
    }
  }
  CHECK(codes_vary);  // a talker stuck on one token still renders

  // 3. ACOUSTIC CHAIN.
  const auto s2 = vllm::SafetensorsFile::Open(std::string(s2p));
  auto est = ix::LoadS2MelEstimator(s2, 8);
  ix::RenderStages st;
  st.regulator = vllm::models::lenreg::LoadRegulator(s2, &st.regulator_config);
  st.front_config = est.front_config;
  st.front = est.front;
  st.stack_config = est.stack_config;
  st.stack = est.stack;
  st.tail_config = est.tail_config;
  st.tail = est.tail;
  const auto bv = vllm::models::bigvgan::Load(std::string(bvp));
  st.vocoder_config = bv.config;
  st.vocoder = bv.weights;

  const int64_t IN = st.regulator_config.in_channels;
  const int64_t cf = static_cast<int64_t>(codes.size());
  std::vector<float> content(static_cast<size_t>(cf * IN));
  for (int64_t f = 0; f < cf; ++f) {
    for (int64_t i = 0; i < IN; ++i) {
      content[static_cast<size_t>(f * IN + i)] =
          talker.mel_embedding[static_cast<size_t>(codes[f] * D + (i % D))] * 0.1F;
    }
  }

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

  // The sample count must follow from the CODES, so a stage that quietly
  // dropped its input would change it.
  REQUIRE(wave.size() == static_cast<size_t>(cf * 4 * 256));
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
  MESSAGE("TEXT -> AUDIO: " << ids.size() << " ids -> " << codes.size()
          << " mel codes -> " << wave.size() << " samples ("
          << static_cast<double>(wave.size()) / 22050.0 << " s at 22050 Hz), rms "
          << rms);
}
