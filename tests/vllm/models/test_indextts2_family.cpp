// IndexTTS-2.5 speech-family registration (#634).
//
// The stages are not implemented (W3-W5), so this gate covers what IS true
// today: the family DETECTS its own checkpoint, and loading refuses while naming
// the exact pieces that are missing. An arm that is silently absent is the
// failure this project has already recorded; an arm that refuses by name is
// visible debt.
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/indextts2.h"
#include "vllm/multimodal/speech_engine.h"

namespace {

std::string MakeCheckpoint(const std::string& leaf, const std::string& config_body) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("indextts2_gate_" + leaf);
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "config.yaml") << config_body;
  return dir.string();
}

}  // namespace

TEST_CASE("indextts2 detects its own checkpoint by INSPECTING config.yaml") {
  vllm::multimodal::SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);

  vllm::multimodal::SpeechModelParams params;
  params.path = MakeCheckpoint("real",
                               "gpt:\n  model_dim: 1280\n"
                               "s2mel_checkpoint: s2mel.pth\n"
                               "semantic_codec:\n  codebook_size: 8192\n");
  std::string why;
  // CHANGED DELIBERATELY. This used to assert the LOAD refuses, which was true
  // when only W1/W2 existed. The render path is implemented now, so detection
  // succeeds AND the load succeeds; the refusal moved to `Synthesize`, where
  // the one genuinely missing piece is. Asserting the old behaviour here would
  // pin the lane to a state it has left.
  const auto engine = registry.Load(params, &why);
  REQUIRE(engine != nullptr);
  CHECK(engine->family() == "indextts2");
  CHECK(why.empty());
}

TEST_CASE("indextts2 refusal names the missing piece and its issue") {
  vllm::multimodal::SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);
  vllm::multimodal::SpeechModelParams params;
  params.path = MakeCheckpoint("named",
                               "gpt:\n  model_dim: 1280\n"
                               "s2mel_checkpoint: s2mel.pth\n"
                               "semantic_codec:\n  codebook_size: 8192\n");
  std::string why;
  const auto engine = registry.Load(params, &why);
  REQUIRE(engine != nullptr);

  vllm::multimodal::SpeechGenParams gen;
  gen.text = "hello";
  gen.reference_audio.assign(1600, 0.01F);
  gen.reference_sample_rate = 16000;
  try {
    engine->Synthesize(gen);
    FAIL("synthesis must refuse while the tokenizer is missing");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    // Naming the piece is the whole point: "unsupported" sends the next person
    // reading loader source. Match the EXPLANATION, not the filename -- the
    // shipped vocabulary is called `...char_del.tiktoken`, so searching for
    // "tiktoken" alone passes on the filename even when the reason is gone.
    // That is what a mutation removing the explanation revealed.
    CHECK(msg.find("no tiktoken reader") != std::string::npos);
    CHECK(msg.find("cannot tokenize") != std::string::npos);
    CHECK(msg.find("#634") != std::string::npos);
    // And it must name the ORACLE separately, since that blocks a different
    // thing -- correctness, not capability.
    CHECK(msg.find("#633") != std::string::npos);
  }
}

TEST_CASE("indextts2 does not claim a checkpoint that is not its own") {
  vllm::multimodal::SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);

  vllm::multimodal::SpeechModelParams params;
  params.path = MakeCheckpoint("other", "architectures:\n  - SomethingElse\n");
  std::string why;
  CHECK(registry.Load(params, &why) == nullptr);
  CHECK(why.find("indextts2") != std::string::npos);
}

TEST_CASE("indextts2 detector does not throw on a missing directory") {
  // Detectors run over artifacts chosen by whoever repackaged a checkpoint; an
  // unreadable one is a no-match, never a crash.
  vllm::multimodal::SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);
  vllm::multimodal::SpeechModelParams params;
  params.path = "/definitely/not/here";
  std::string why;
  CHECK_NOTHROW(registry.Load(params, &why));
}

// ---------------------------------------------------------------------------
// The seam is now POPULATED: loading succeeds and the capability is
// describable, and the refusal names the ONE remaining piece (#634).
// ---------------------------------------------------------------------------

TEST_CASE("an IndexTTS-2.5 checkpoint now LOADS through the seam") {
  vllm::multimodal::SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);
  vllm::multimodal::SpeechModelParams params;
  params.path = MakeCheckpoint("engine0", "gpt:\n  model_dim: 1280\n"
      "s2mel_checkpoint: s2mel.pth\n"
      "semantic_codec:\n  codebook_size: 8192\n");
  std::string why;
  const auto engine = registry.Load(params, &why);
  REQUIRE(engine != nullptr);
  CHECK(engine->family() == "indextts2");
  // 22050, the OUTPUT rate. The talker's mel front end runs at 24000 and
  // reporting that would make every caller resample to the wrong rate.
  CHECK(engine->sample_rate() == 22050);
  // Upstream has no text-only synthesis, so a server can reject before staging.
  CHECK(engine->requires_reference_audio());
}

TEST_CASE("synthesis without a reference clip is refused FIRST") {
  vllm::multimodal::SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);
  vllm::multimodal::SpeechModelParams params;
  params.path = MakeCheckpoint("engine1", "gpt:\n  model_dim: 1280\n"
      "s2mel_checkpoint: s2mel.pth\n"
      "semantic_codec:\n  codebook_size: 8192\n");
  std::string why;
  const auto engine = registry.Load(params, &why);
  REQUIRE(engine != nullptr);

  vllm::multimodal::SpeechGenParams gen;
  gen.text = "hello";
  // No reference audio at all.
  try {
    engine->Synthesize(gen);
    FAIL("synthesis without a reference clip must refuse");
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    CHECK(what.find("reference clip is REQUIRED") != std::string::npos);
    // It must refuse for THAT reason, not incidentally for the tokenizer.
    CHECK(what.find("tokenize") == std::string::npos);
  }
}

TEST_CASE("with a clip, the refusal names the TOKENIZER and nothing stale") {
  vllm::multimodal::SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);
  vllm::multimodal::SpeechModelParams params;
  params.path = MakeCheckpoint("engine2", "gpt:\n  model_dim: 1280\n"
      "s2mel_checkpoint: s2mel.pth\n"
      "semantic_codec:\n  codebook_size: 8192\n");
  std::string why;
  const auto engine = registry.Load(params, &why);
  REQUIRE(engine != nullptr);

  vllm::multimodal::SpeechGenParams gen;
  gen.text = "hello";
  gen.reference_audio.assign(16000, 0.01F);
  gen.reference_sample_rate = 16000;
  try {
    engine->Synthesize(gen);
    FAIL("synthesis must refuse while the tokenizer is missing");
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    CHECK(what.find("no tiktoken reader") != std::string::npos);
    CHECK(what.find("cannot tokenize") != std::string::npos);
    CHECK(what.find("#634") != std::string::npos);
    // The refusal must NOT still claim the render path is unimplemented: that
    // was true when this family was registered and is not true now, and a
    // stale refusal sends the next reader to build what already exists.
    CHECK(what.find("S2Mel CFM/DiT decoder") == std::string::npos);
    CHECK(what.find("not implemented yet") == std::string::npos);
  }
}
