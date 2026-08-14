// vllm.cpp original. The `/v1/audio/speech` request contract (W6 of #672).
//
// OpenAI's audio-speech route is the established spelling for "text in, audio
// bytes out" (platform.openai.com/docs/api-reference/audio/createSpeech:
// {model, input, voice, response_format}), and it is what a client already
// knows how to call. So the route keeps that name and that shape rather than
// inventing `/v1/music`, and the MUSIC inputs enter as two ADDITIONAL named
// fields — because upstream MiniMax-Music3 runs a DIFFERENT normalizer over the
// lyrics and over the description (encoders.py:54-91), so packing both into
// `input` behind a separator would be a private protocol.
//
// This header carries the request contract ONLY. Everything it describes is
// pure, which is what makes it gateable without a checkpoint; the synthesis
// itself is a callback the server is given (ApiServer::set_synthesizer), the
// same shape the video runner and the transcriber already use.
//
// WHAT IS REFUSED RATHER THAN IGNORED, and why each:
//   * `voice` — no registered family exposes named voices, and the seam has no
//     voice-enumeration surface. Accepting and dropping it would answer a
//     request for one voice with another.
//   * `response_format` other than "wav" — no mp3/opus/aac/flac encoder is
//     vendored, and re-labelling RIFF bytes as mp3 is worse than refusing.
//   * `stream` / `stream_format` — upstream Music3 has no streaming
//     (spec §7); buffering the whole song and calling it a stream is the lie
//     that rule exists to prevent.
//   * `speed` — no family implements a rate control, and silently ignoring it
//     returns audio at the wrong tempo with a 200.
#ifndef VLLM_ENTRYPOINTS_OPENAI_SPEECH_API_H_
#define VLLM_ENTRYPOINTS_OPENAI_SPEECH_API_H_

#include <cstdint>
#include <string>
#include <vector>

namespace vllm::openai {

// One parsed `/v1/audio/speech` body. Every generation control is inert at its
// default, which means "the family decides".
struct SpeechRequest {
  std::string model;  // echoed, never checked here — only the route knows the names
  // The one-utterance input, OpenAI's `input`.
  std::string text;
  std::string language;
  // The music inputs.
  std::string lyrics;
  std::string description;
  // The reference clip, for a family that requires one. Supplied as a 16-bit
  // PCM WAV `data:` URL under `reference_audio`; decoded to mono f32 here so
  // the seam never sees a container.
  std::vector<float> reference_audio;
  int64_t reference_sample_rate = 0;

  double audio_duration_s = 0.0;
  int64_t num_inference_steps = 0;
  double guidance_scale = 0.0;
  // 0 IS A LEGAL guidance scale, so "the caller specified one" cannot be
  // encoded as a non-zero value and is this flag instead.
  bool has_guidance_scale = false;
  int64_t seed = 0;

  bool has_any_text() const { return !text.empty() || !lyrics.empty() || !description.empty(); }
};

// Parse, or THROW naming the field. A body that cannot be fully read is a 400
// rather than a half-honoured request.
//
// It requires at least one of `input`, `lyrics` or `description` — WHICH of
// them a family needs is the family's decision, made against the whole request
// (Music3 refuses `input` and requires `lyrics`), and duplicating that here
// would be a second contract to keep in step.
SpeechRequest ParseSpeechRequest(const std::string& body);

// What the server knows about the attached family WITHOUT running a synthesis.
// It is what lets a request be refused BEFORE tens of gigabytes stage, which is
// the reason `SpeechEngine::requires_reference_audio()` is exposed at all.
struct SpeechCapabilities {
  std::string family;
  int64_t sample_rate = 0;
  int64_t channels = 1;
  bool requires_reference_audio = false;
};

// One finished synthesis, as the route serves it.
struct SpeechResponse {
  std::string wav;  // RIFF/WAVE 16-bit PCM
  int64_t sample_rate = 0;
  int64_t channels = 1;
  int64_t samples_per_channel = 0;
};

}  // namespace vllm::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SPEECH_API_H_
