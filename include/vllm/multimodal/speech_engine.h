// The speech (text-to-speech) engine seam — W6a of #634.
//
// This project's audio surface CONSUMES audio today (Parakeet, Voxtral,
// audio_processor.cpp); nothing synthesizes it. IndexTTS-2.5 is the first
// generating lane, and vLLM-Omni carries roughly ten more TTS architectures
// behind it, so the seam is shaped for a family rather than for one model.
//
// It mirrors `vllm::multimodal::VideoEngine`, which solved the same problem for
// the other generative modality: one abstract engine, per-family
// self-registration so adding a family edits no shared array, and detection that
// INSPECTS the artifact rather than trusting a file extension or a path spelling
// (both chosen by whoever repackaged the checkpoint).
//
// WHAT THIS IS NOT. No family is registered yet: the IndexTTS-2.5 stages land
// with W3-W5 (.agents/specs/indextts-2-5.md). Until one does, `Load` REFUSES and
// names what it tried, because a silently absent arm is a failure this project
// has already recorded, while a refusal that names the missing piece is owed
// debt.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vllm {
namespace multimodal {

// Where the checkpoint set lives, plus the optional family override a caller
// uses to SKIP detection (never to override a detector that disagrees).
struct SpeechModelParams {
  std::string path;
  std::string family;  // empty => detect
};

// One synthesis request.
struct SpeechGenParams {
  std::string text;
  std::string language;  // upstream's `lang`; empty => the family's default

  // The reference clip. IndexTTS-2 has NO text-only synthesis, so for that
  // family an empty clip is a refusal rather than a default voice.
  std::vector<float> reference_audio;
  int64_t reference_sample_rate = 0;

  // Upstream states a seed controls both AR sampling and per-request CFM noise.
  int64_t seed = 0;
};

// A rendered waveform. Mono unless a family says otherwise; `sample_rate` is the
// family's native rate (22050 for IndexTTS-2.5) rather than a resampled one, so
// the caller decides whether to resample.
struct SpeechResult {
  std::vector<float> samples;
  int64_t sample_rate = 0;
  int64_t channels = 1;
};

// A loaded speech checkpoint set, weights staged once, ready to synthesize.
class SpeechEngine {
 public:
  virtual ~SpeechEngine();

  // The stable registry name of the family this engine implements.
  virtual std::string family() const = 0;

  // The native output rate, so a caller never has to infer it from the family.
  virtual int64_t sample_rate() const = 0;

  // True when the family cannot synthesize without a reference clip. Exposed
  // rather than implied, so a server can reject a request before staging.
  virtual bool requires_reference_audio() const = 0;

  // Run one blocking synthesis. Implementations serialize internally (staged
  // weights are shared state); throws std::runtime_error to fail the request.
  virtual SpeechResult Synthesize(const SpeechGenParams& params) = 0;

 protected:
  SpeechEngine() = default;
  SpeechEngine(const SpeechEngine&) = default;
  SpeechEngine& operator=(const SpeechEngine&) = default;
  SpeechEngine(SpeechEngine&&) = default;
  SpeechEngine& operator=(SpeechEngine&&) = default;
};

// Does this checkpoint set belong to the family? A detector must not throw: an
// unreadable or unrecognizable artifact is `false`, and one family's bad day
// must not deny every other family a chance to claim the checkpoint.
using SpeechFamilyDetector = std::function<bool(const SpeechModelParams&)>;

// Load the checkpoint set as this family. Throws std::runtime_error naming the
// problem on any mismatch.
using SpeechFamilyLoader = std::function<std::unique_ptr<SpeechEngine>(const SpeechModelParams&)>;

struct SpeechFamilyRegistration {
  std::string name;  // stable family name, e.g. "indextts2"
  SpeechFamilyDetector detect;
  SpeechFamilyLoader load;
};

class SpeechRegistry {
 public:
  // Throws on an empty name, a missing detector or loader, or A NAME ALREADY
  // REGISTERED — the last because two families sharing one name is the
  // never-guess guarantee defeated from the inside.
  void Register(SpeechFamilyRegistration registration);

  // Resolve and load. Returns nullptr and fills `why` when nothing claims the
  // checkpoint; `why` names every family that was tried and the path, so the
  // refusal is evidence rather than a verdict.
  std::unique_ptr<SpeechEngine> Load(const SpeechModelParams& params, std::string* why) const;

  std::vector<std::string> families() const;

 private:
  std::vector<SpeechFamilyRegistration> families_;
};

// The process-global registry, for families that self-register.
SpeechRegistry& GlobalSpeechRegistry();

}  // namespace multimodal
}  // namespace vllm
