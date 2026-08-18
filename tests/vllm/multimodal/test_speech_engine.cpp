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
#include "vllm/platforms/interface.h"
#include "vt/backend.h"

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

// ── The device selector (#672) ────────────────────────────────────────────────
//
// The seam grew a place to say WHERE a family runs, because MiniMax-Music3's
// queue used to be a compile-time constant and a 28.5 GB music model was
// therefore host-only whatever hardware the box had.
//
// Everything below is checkpoint-free and runs in CI on every build. What it
// CANNOT see is a granted accelerator — on a CPU-only build there is none to
// grant — so the device-1 arm is gated here only for its REFUSAL, and the
// granted arm is gated on hardware (see the row's evidence).

TEST_CASE("speech: the device selector's zero value is the CPU arm") {
  // Load-bearing, not cosmetic. Every Music3 correctness gate was taken on the
  // CPU path, so a zero-initialized `SpeechModelParams` — which is what every
  // pre-#672 caller has — must keep resolving to it. This is also why the field
  // is NOT `vllm_model_params.device`'s 0=auto spelling: `auto` would flip an
  // accelerator build's default under exactly those callers.
  const SpeechModelParams defaults;
  CHECK(defaults.device == 0);
  CHECK(vllm::multimodal::SpeechEngineDeviceType(0, "minimax-music3") ==
        vt::DeviceType::kCPU);
}

TEST_CASE("speech: a family with no device arm reports CPU rather than being edited") {
  // `SpeechEngine::device()` is non-pure with a CPU default precisely so that
  // IndexTTS-2.5 — and every family after it that has no device arm — answers
  // honestly without a line changing. FakeEngine overrides nothing.
  FakeEngine engine("no-device-arm");
  CHECK(engine.device().type == vt::DeviceType::kCPU);
  CHECK(engine.device().index == 0);
}

TEST_CASE("speech: a device selector that is neither 0 nor 1 is refused BY NAME") {
  // NOT clamped and NOT defaulted. `static_cast<vt::DeviceType>(device)` is the
  // mistake this helper exists to prevent (minimax_h3_video.cpp:230-237): it
  // reads the ABI selector AS AN ENUM VALUE, so device 2 would silently bind
  // whichever backend sits at index 2 in vt::DeviceType.
  for (const int32_t bad : {-1, 2, 7}) {
    CHECK_THROWS_AS(vllm::multimodal::SpeechEngineDeviceType(bad, "minimax-music3"),
                    std::runtime_error);
    std::string message;
    try {
      vllm::multimodal::SpeechEngineDeviceType(bad, "minimax-music3");
    } catch (const std::runtime_error& e) {
      message = e.what();
    }
    // The refusal must name BOTH legal values, or it tells the caller it was
    // wrong without telling it what right looks like.
    CHECK(message.find("0 (cpu)") != std::string::npos);
    CHECK(message.find("1 (") != std::string::npos);
    CHECK(message.find(std::to_string(bad)) != std::string::npos);
  }
}

TEST_CASE("speech: device 1 on a build with no accelerator is refused, never substituted") {
  // On a CPU-only build this asserts the refusal and its wording; on an
  // accelerator build the same call GRANTS a device, and asserting a throw
  // there would be asserting that the arm this row added does not work. The
  // case therefore reports which arm it examined rather than assuming one — a
  // gate that cannot say what it looked at has not reported.
  const vt::DeviceType accelerator = vllm::platforms::CurrentPlatform().device_type();
  const bool have_accelerator =
      accelerator != vt::DeviceType::kCPU && vt::TryGetBackend(accelerator) != nullptr;
  MESSAGE("speech device-1 arm examined on a build whose platform resolves to '"
          << vt::DeviceTypeName(accelerator) << "', accelerator backend registered: "
          << (have_accelerator ? "yes" : "no"));
  if (!have_accelerator) {
    std::string message;
    try {
      vllm::multimodal::SpeechEngineDeviceType(1, "minimax-music3");
      FAIL("device 1 was GRANTED on a build with no accelerator backend");
    } catch (const std::runtime_error& e) {
      message = e.what();
    }
    // Naming the piece that is missing is what separates owed debt from a
    // mystery: "no accelerator backend is registered in this build".
    CHECK(message.find("no accelerator backend") != std::string::npos);
    CHECK(message.find("Refusing") != std::string::npos);
  } else {
    // The architecture key reaches the platform: an accelerator that DECLINES
    // this family must refuse by name rather than hand back a queue that dies
    // inside a kernel bind.
    const bool accepted =
        vllm::platforms::CurrentPlatform().supports_model_architecture("minimax-music3");
    if (accepted) {
      CHECK(vllm::multimodal::SpeechEngineDeviceType(1, "minimax-music3") == accelerator);
    } else {
      CHECK_THROWS_AS(vllm::multimodal::SpeechEngineDeviceType(1, "minimax-music3"),
                      std::runtime_error);
    }
  }
}
