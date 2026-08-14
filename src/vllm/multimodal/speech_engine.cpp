// The speech engine seam. See speech_engine.h for why it mirrors VideoEngine.
#include "vllm/multimodal/speech_engine.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace vllm {
namespace multimodal {

SpeechEngine::~SpeechEngine() = default;

void SpeechRegistry::Register(SpeechFamilyRegistration registration) {
  if (registration.name.empty()) {
    throw std::runtime_error("speech registry: a family must have a name");
  }
  if (!registration.detect) {
    throw std::runtime_error("speech registry: family '" + registration.name +
                             "' has no detector");
  }
  if (!registration.load) {
    throw std::runtime_error("speech registry: family '" + registration.name + "' has no loader");
  }
  for (const SpeechFamilyRegistration& existing : families_) {
    if (existing.name == registration.name) {
      // Two claimants collapsing into one listed name is the never-guess
      // guarantee defeated from the inside.
      throw std::runtime_error("speech registry: family '" + registration.name +
                               "' is already registered");
    }
  }
  families_.push_back(std::move(registration));
}

std::vector<std::string> SpeechRegistry::families() const {
  std::vector<std::string> out;
  out.reserve(families_.size());
  for (const SpeechFamilyRegistration& f : families_) out.push_back(f.name);
  return out;
}

std::unique_ptr<SpeechEngine> SpeechRegistry::Load(const SpeechModelParams& params,
                                                   std::string* why) const {
  const auto say = [why](const std::string& message) {
    if (why != nullptr) *why = message;
  };

  if (families_.empty()) {
    // Distinguish "this lane does not exist yet" from "your checkpoint is bad".
    // Reading the second when the first is true sends people to debug a file
    // that was never the problem.
    say("no speech (TTS) family is registered: the IndexTTS-2.5 stages are not "
        "implemented yet (see .agents/specs/indextts-2-5.md, W3-W5)");
    return nullptr;
  }

  std::string tried;
  for (const SpeechFamilyRegistration& family : families_) {
    if (!params.family.empty() && params.family != family.name) continue;
    if (!tried.empty()) tried += ", ";
    tried += family.name;

    bool claimed = false;
    try {
      claimed = family.detect(params);
    } catch (const std::exception&) {
      // A detector that throws is a no-match, never a crash: it runs over
      // artifacts chosen by whoever repackaged the checkpoint.
      claimed = false;
    }
    if (claimed) return family.load(params);
  }

  if (!params.family.empty() && tried.empty()) {
    say("speech: no family named '" + params.family + "' is registered");
    return nullptr;
  }
  say("speech: no registered family claims '" + params.path + "'; tried " + tried);
  return nullptr;
}

SpeechRegistry& GlobalSpeechRegistry() {
  static SpeechRegistry registry;
  return registry;
}

}  // namespace multimodal
}  // namespace vllm
