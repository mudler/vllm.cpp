// The speech (TTS) engine seam — W6a of #634.
//
// Mirrors `vllm::multimodal::VideoEngine`, which solved the same problem for the
// other generative modality: one abstract seam, per-family self-registration, and
// detection that INSPECTS the artifact rather than trusting a path spelling.
//
// This gate covers the seam's contract, which is testable with no model and no
// oracle: registration, refusal-by-name, and the never-guess guarantees. The
// first real family (IndexTTS-2.5) arrives with W3-W5; until then a load must
// REFUSE and say what is missing, because an arm that is silently absent is the
// failure this project has already recorded.
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/multimodal/speech_engine.h"

namespace {

using vllm::multimodal::SpeechEngine;
using vllm::multimodal::SpeechFamilyRegistration;
using vllm::multimodal::SpeechGenParams;
using vllm::multimodal::SpeechModelParams;
using vllm::multimodal::SpeechRegistry;
using vllm::multimodal::SpeechResult;

class FakeEngine final : public SpeechEngine {
 public:
  explicit FakeEngine(std::string family) : family_(std::move(family)) {}
  std::string family() const override { return family_; }
  int64_t sample_rate() const override { return 22050; }
  bool requires_reference_audio() const override { return true; }
  SpeechResult Synthesize(const SpeechGenParams& params) override {
    if (params.reference_audio.empty()) {
      throw std::runtime_error("fake-tts: reference audio is required");
    }
    SpeechResult r;
    r.sample_rate = 22050;
    r.samples.assign(4, 0.25F);
    return r;
  }

 private:
  std::string family_;
};

SpeechFamilyRegistration MakeFamily(const std::string& name, bool detects) {
  SpeechFamilyRegistration reg;
  reg.name = name;
  reg.detect = [detects](const SpeechModelParams&) { return detects; };
  reg.load = [name](const SpeechModelParams&) -> std::unique_ptr<SpeechEngine> {
    return std::make_unique<FakeEngine>(name);
  };
  return reg;
}

}  // namespace

TEST_CASE("speech registry resolves a family by inspecting the checkpoint") {
  SpeechRegistry registry;
  registry.Register(MakeFamily("no-match", false));
  registry.Register(MakeFamily("match", true));

  SpeechModelParams params;
  params.path = "/some/checkpoint";
  std::string why;
  std::unique_ptr<SpeechEngine> engine = registry.Load(params, &why);
  REQUIRE(engine != nullptr);
  CHECK(engine->family() == "match");
  CHECK(engine->sample_rate() == 22050);
  CHECK(engine->requires_reference_audio());
}

TEST_CASE("speech registry refuses a duplicate family name") {
  // Two families sharing one name is the never-guess guarantee defeated from the
  // inside: the listing shows one family while two claimants collapse into it.
  SpeechRegistry registry;
  registry.Register(MakeFamily("dup", true));
  CHECK_THROWS_AS(registry.Register(MakeFamily("dup", true)), std::runtime_error);
}

TEST_CASE("speech registry refuses an incomplete registration") {
  SpeechRegistry registry;
  SpeechFamilyRegistration empty_name = MakeFamily("", true);
  CHECK_THROWS_AS(registry.Register(empty_name), std::runtime_error);

  SpeechFamilyRegistration no_detect = MakeFamily("x", true);
  no_detect.detect = nullptr;
  CHECK_THROWS_AS(registry.Register(no_detect), std::runtime_error);

  SpeechFamilyRegistration no_load = MakeFamily("y", true);
  no_load.load = nullptr;
  CHECK_THROWS_AS(registry.Register(no_load), std::runtime_error);
}

TEST_CASE("an unresolved checkpoint refuses BY NAME rather than guessing") {
  // The refusal has to name what was tried. "unsupported" with no evidence is
  // what sends the next person reading loader source.
  SpeechRegistry registry;
  registry.Register(MakeFamily("alpha", false));
  registry.Register(MakeFamily("beta", false));

  SpeechModelParams params;
  params.path = "/unknown/checkpoint";
  std::string why;
  CHECK(registry.Load(params, &why) == nullptr);
  CHECK(why.find("alpha") != std::string::npos);
  CHECK(why.find("beta") != std::string::npos);
  CHECK(why.find("/unknown/checkpoint") != std::string::npos);
}

TEST_CASE("an EMPTY registry says the lane is unimplemented, not that the file is bad") {
  // Today this is the real state: no family is registered, because IndexTTS-2.5
  // lands with W3-W5. A caller must be told the LANE is missing rather than be
  // left to conclude their checkpoint is corrupt.
  SpeechRegistry registry;
  SpeechModelParams params;
  params.path = "/anything";
  std::string why;
  CHECK(registry.Load(params, &why) == nullptr);
  CHECK(why.find("no speech") != std::string::npos);
}

TEST_CASE("a detector that throws is treated as no-match, not as a crash") {
  // Detection runs over artifacts chosen by whoever repackaged a checkpoint. One
  // family's bad day must not deny every other family a chance to claim it.
  SpeechRegistry registry;
  SpeechFamilyRegistration thrower = MakeFamily("thrower", true);
  thrower.detect = [](const SpeechModelParams&) -> bool {
    throw std::runtime_error("detector exploded");
  };
  registry.Register(thrower);
  registry.Register(MakeFamily("survivor", true));

  SpeechModelParams params;
  params.path = "/x";
  std::string why;
  std::unique_ptr<SpeechEngine> engine = registry.Load(params, &why);
  REQUIRE(engine != nullptr);
  CHECK(engine->family() == "survivor");
}

TEST_CASE("synthesis refuses when the mandatory reference clip is absent") {
  // IndexTTS-2 has NO text-only synthesis, so a missing reference clip is a
  // refusal rather than a default voice.
  SpeechRegistry registry;
  registry.Register(MakeFamily("match", true));
  SpeechModelParams params;
  params.path = "/ckpt";
  std::string why;
  std::unique_ptr<SpeechEngine> engine = registry.Load(params, &why);
  REQUIRE(engine != nullptr);

  SpeechGenParams gen;
  gen.text = "hello";
  CHECK_THROWS_AS(engine->Synthesize(gen), std::runtime_error);

  gen.reference_audio.assign(16, 0.0F);
  gen.reference_sample_rate = 16000;
  const SpeechResult out = engine->Synthesize(gen);
  CHECK(out.sample_rate == 22050);
  CHECK(out.samples.size() == 4U);
}
