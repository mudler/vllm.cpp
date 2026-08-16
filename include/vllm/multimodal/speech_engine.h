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

#include "vt/device.h"

namespace vllm {
namespace multimodal {

// Where the checkpoint set lives, plus the optional family override a caller
// uses to SKIP detection (never to override a detector that disagrees).
struct SpeechModelParams {
  std::string path;
  std::string family;  // empty => detect

  // WHERE the family runs (the device arm of #672). 0 = CPU, 1 = the
  // accelerator this build resolves; `SpeechEngineDeviceType` below is the
  // mapping and it REFUSES anything else BY NAME.
  //
  // Load-time rather than per-request, because weights are staged once: a
  // family that re-uploaded its 28.5 GB per `Synthesize` would be a device arm
  // on paper and a transfer benchmark in practice.
  //
  // ZERO IS CPU, deliberately, and this is NOT the `vllm_model_params.device`
  // spelling (0=auto / 1=cpu / 2=cuda, mirroring vLLM's DeviceConfig). The text
  // engine's `auto` picks an accelerator when one exists; a speech family's CPU
  // path is what every correctness gate for it was taken on, so a zero-value
  // caller must keep getting exactly the arm it has always had. This mirrors
  // `VideoModelParams::device` (video_engine.h) — the sibling generative-engine
  // seam, which fixed the same polarity for the same reason.
  int32_t device = 0;
};

// Resolve the ABI's device selector for a speech family, asking the three
// questions this seam asks everywhere else (minimax_h3_video.cpp:255, the
// original): is there an accelerator at all, is a backend registered for it,
// and does that platform accept THIS architecture — a PARTIAL backend (Metal
// 15/75 ops, Tenstorrent) must be able to decline by name rather than be handed
// a queue and die inside a kernel bind.
//
// `family` is the architecture key because that is this lane's stable registry
// name (`SpeechModelParams::family`); a speech engine is not reached through
// ModelRegistry's HF `architectures` entry. It lives HERE rather than beside one
// family because the tree already carried two copies of this mapping
// (minimax_h3_video.cpp, ltx2_video.cpp) and a third is where copies start to
// disagree.
//
// THROWS on anything but 0 or 1, and on a device 1 this build cannot serve.
vt::DeviceType SpeechEngineDeviceType(int32_t device, const std::string& family);

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

  // ── MUSIC families (W6 of #672, .agents/specs/minimax-music3.md §4.1) ──────
  //
  // A text-to-MUSIC family takes TWO distinct texts, not one. MiniMax-Music3
  // assembles `<|caption_start|>{description}<|caption_end|>` and
  // `<|lyrics_start|>{lyrics}<|lyrics_end|>` into one prompt
  // (encoders.py:207-210) after running each through a DIFFERENT normalizer —
  // `_clean_caption` on the description, `_normalize_lyrics` on the lyrics — so
  // the two are not interchangeable, and packing both into `text` behind a
  // separator would be a private protocol inside a shared struct. They are
  // named fields instead.
  //
  // A family that synthesizes ONE utterance keeps using `text` and ignores
  // these; a field it ignores costs it nothing, whereas a second params struct
  // would cost every future family a choice. IndexTTS-2.5 is unchanged by their
  // presence — see tests/vllm/multimodal/test_speech_engine.cpp.

  // The sung text, with the `[Verse]` / `[Chorus]` section tags the family's
  // own normalizer consumes. Empty => the family decides (Music3 REFUSES).
  std::string lyrics;
  // The structured music description — genre, BPM, key, instrumentation, mood.
  // NOT a voice or speaker description. Empty => the family decides.
  std::string description;

  // ── Generation controls ───────────────────────────────────────────────────
  // Every one is inert at its default, so a caller written against the
  // pre-extension struct gets exactly the behaviour it got before.

  // Requested output length in seconds. <= 0 => the family's default.
  double audio_duration_s = 0.0;
  // Denoise / flow-matching steps for a family that has such a loop.
  // <= 0 => the family's default.
  int64_t num_inference_steps = 0;
  // Classifier-free guidance scale. NEGATIVE => the family's default, because
  // 0 is a LEGAL guidance scale (it selects the unconditional branch) and a
  // 0-means-default sentinel would make that value unreachable.
  double guidance_scale = -1.0;
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

  // WHERE this engine actually resolved to run — the mirror of
  // `VideoEngine::device()`. Reported rather than inferred from the request,
  // because "I asked for device 1" and "device 1 was granted" are different
  // facts and a benchmark that confuses them measures the wrong arm.
  //
  // NON-pure with a CPU default: every family that has no device arm answers
  // honestly without being edited, so IndexTTS-2.5 is untouched by this seam.
  virtual vt::Device device() const;

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
