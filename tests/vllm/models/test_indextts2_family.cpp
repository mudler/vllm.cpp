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
  // Detection must SUCCEED and the load must then refuse for a reason about the
  // PORT, not about the file.
  CHECK_THROWS_WITH_AS(registry.Load(params, &why),
                       doctest::Contains("not implemented"), std::runtime_error);
}

TEST_CASE("indextts2 refusal names every missing stage and its issue") {
  vllm::multimodal::SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);
  vllm::multimodal::SpeechModelParams params;
  params.path = MakeCheckpoint("named",
                               "gpt:\n  model_dim: 1280\n"
                               "s2mel_checkpoint: s2mel.pth\n"
                               "semantic_codec:\n  codebook_size: 8192\n");
  std::string why;
  try {
    registry.Load(params, &why);
    FAIL("load must refuse while the stages are unimplemented");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    // Naming the pieces is the whole point: "unsupported" sends the next person
    // reading loader source.
    CHECK(msg.find("w2v-bert") != std::string::npos);
    CHECK(msg.find("EnhancedCodec") != std::string::npos);
    CHECK(msg.find("S2Mel") != std::string::npos);
    CHECK(msg.find("#634") != std::string::npos);
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
