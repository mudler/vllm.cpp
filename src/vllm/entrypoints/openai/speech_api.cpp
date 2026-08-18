// See include/vllm/entrypoints/openai/speech_api.h. Request contract only; the
// library performs no synthesis here.
#include "vllm/entrypoints/openai/speech_api.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

// DecodeDataUri: the RFC 2397 decode the chat multimodal parts and the video
// route already use, so a reference clip arrives through the SAME decoder
// rather than a second, subtly different one.
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vt/dtype.h"

namespace vllm::openai {
namespace {

bool Has(const nlohmann::json& body, const char* key) {
  return body.contains(key) && !body.at(key).is_null();
}

std::string ReadString(const nlohmann::json& body, const char* key) {
  VT_CHECK(body.at(key).is_string(),
           std::string("speech request: `") + key + "` must be a string");
  return body.at(key).get<std::string>();
}

double ReadNumber(const nlohmann::json& body, const char* key) {
  VT_CHECK(body.at(key).is_number(),
           std::string("speech request: `") + key + "` must be a number");
  return body.at(key).get<double>();
}

// A 16-bit PCM RIFF/WAVE reference clip -> mono f32 in [-1, 1). Refuses anything
// else BY NAME rather than reinterpreting the bytes: a wrongly parsed clip
// conditions the synthesis on noise and still returns 200.
void DecodeReferenceWav(const std::vector<uint8_t>& bytes, std::vector<float>* out,
                        int64_t* sample_rate) {
  VT_CHECK(bytes.size() >= 44, "speech request: `reference_audio` is too short to be a WAV");
  VT_CHECK(std::memcmp(bytes.data(), "RIFF", 4) == 0 &&
               std::memcmp(bytes.data() + 8, "WAVE", 4) == 0,
           "speech request: `reference_audio` is not a RIFF/WAVE file");
  const auto u16 = [&bytes](size_t offset) {
    return static_cast<uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
  };
  const auto u32 = [&bytes](size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
  };
  VT_CHECK(u16(20) == 1, "speech request: `reference_audio` must be uncompressed PCM");
  const uint16_t channels = u16(22);
  VT_CHECK(channels == 1, "speech request: `reference_audio` must be MONO, got " +
                              std::to_string(channels) + " channels");
  VT_CHECK(u16(34) == 16, "speech request: `reference_audio` must be 16-bit PCM");
  *sample_rate = u32(24);
  VT_CHECK(*sample_rate > 0, "speech request: `reference_audio` declares a 0 Hz sample rate");
  // The canonical 44-byte header; a file carrying extra chunks is refused
  // rather than read at the wrong offset.
  VT_CHECK(std::memcmp(bytes.data() + 36, "data", 4) == 0,
           "speech request: `reference_audio` must carry its `data` chunk at the canonical "
           "44-byte offset (extra RIFF chunks are not parsed)");
  const uint32_t payload = u32(40);
  VT_CHECK(payload + 44u <= bytes.size(),
           "speech request: `reference_audio` declares more PCM than it carries");
  out->resize(payload / 2);
  for (size_t i = 0; i < out->size(); ++i) {
    const int16_t sample =
        static_cast<int16_t>(bytes[44 + 2 * i] | (bytes[44 + 2 * i + 1] << 8));
    (*out)[i] = static_cast<float>(sample) / 32768.0f;
  }
}

}  // namespace

SpeechRequest ParseSpeechRequest(const std::string& body) {
  nlohmann::json json;
  try {
    json = nlohmann::json::parse(body);
  } catch (const std::exception&) {
    VT_CHECK(false, "speech request: body is not valid JSON");
  }
  VT_CHECK(json.is_object(), "speech request: body must be a JSON object");

  SpeechRequest out;
  if (Has(json, "model")) {
    out.model = ReadString(json, "model");
    VT_CHECK(!out.model.empty(), "speech request: `model` must not be empty");
  }
  // OpenAI's field is `input`; `text` is accepted as the native spelling so a
  // caller does not have to know which surface it is talking to.
  if (Has(json, "input")) out.text = ReadString(json, "input");
  if (Has(json, "text")) {
    VT_CHECK(out.text.empty() || out.text == ReadString(json, "text"),
             "speech request: `input` and `text` are the same field and disagree; supply one");
    out.text = ReadString(json, "text");
  }
  if (Has(json, "language")) out.language = ReadString(json, "language");
  if (Has(json, "lyrics")) out.lyrics = ReadString(json, "lyrics");
  if (Has(json, "description")) out.description = ReadString(json, "description");
  // vLLM-Omni and the video route both carry the music description under
  // `prompt`; accept it as the documented alias rather than making a caller
  // learn a second name for the same string.
  if (Has(json, "prompt")) {
    const std::string prompt = ReadString(json, "prompt");
    VT_CHECK(out.description.empty() || out.description == prompt,
             "speech request: `prompt` and `description` are the same field and disagree; "
             "supply one");
    out.description = prompt;
  }

  VT_CHECK(out.has_any_text(),
           "speech request: one of `input` (a spoken utterance), `lyrics` or `description` "
           "is required; which of them applies is the loaded family's decision");

  // ── The named residuals. Each is refused rather than ignored. ─────────────
  VT_CHECK(!Has(json, "voice"),
           "speech request: `voice` is not supported — no registered speech family exposes "
           "named voices, and there is no voice-enumeration endpoint to pick one from. A "
           "family that clones a voice takes `reference_audio` instead");
  VT_CHECK(!Has(json, "speed"),
           "speech request: `speed` is not supported — no registered family implements a rate "
           "control, and honouring it silently would return audio at the wrong tempo");
  VT_CHECK(!Has(json, "stream") && !Has(json, "stream_format"),
           "speech request: streaming is not supported — MiniMax-Music3 generates the whole "
           "song before the first sample exists (.agents/specs/minimax-music3.md §7), and "
           "buffering it to emit chunks would be a stream in name only");
  if (Has(json, "response_format")) {
    const std::string format = ReadString(json, "response_format");
    VT_CHECK(format == "wav",
             "speech request: `response_format` '" + format +
                 "' is not supported; only \"wav\" is, because no mp3/opus/aac/flac encoder "
                 "is vendored and relabelling RIFF bytes would be worse than refusing");
  }

  if (Has(json, "reference_audio")) {
    const std::string value = ReadString(json, "reference_audio");
    VT_CHECK(value.compare(0, 5, "data:") == 0,
             "speech request: `reference_audio` must be a `data:` URL carrying a 16-bit PCM "
             "mono WAV; a filesystem path is not read, because the server and the client "
             "need not share one");
    std::vector<uint8_t> bytes;
    try {
      bytes = entrypoints::openai::DecodeDataUri(value).bytes;
    } catch (const std::exception& e) {
      VT_CHECK(false, std::string("speech request: `reference_audio` is not a valid data: URL: ") +
                          e.what());
    }
    DecodeReferenceWav(bytes, &out.reference_audio, &out.reference_sample_rate);
  }

  // vLLM-Omni nests the generation knobs under `extra_params`; accept them at
  // the top level too, exactly as the video route does.
  const nlohmann::json& extra =
      (json.contains("extra_params") && json.at("extra_params").is_object()) ? json.at("extra_params")
                                                                            : json;
  // `audio_duration_s` is the name of the C++/C-API FIELD this key fills, not a
  // wire key, and it is the one misspelling a caller reaches for by reading the
  // struct instead of the docs. Dropping it silently hands back the family's
  // default duration — 60 s for MiniMax-Music3 — while returning 200, so the
  // caller cannot tell. That is exactly how the Music3 e2e gate spent three
  // multi-hour runs generating a 60 s song it had asked 0.1 s for (#852): 2 AR
  // frames became 1500 and the run was misread as a hung weight load. REFUSED
  // and NAMED, like every other unsupported field above.
  VT_CHECK(!Has(extra, "audio_duration_s"),
           "speech request: `audio_duration_s` is not a request key — it is the name of the "
           "field it fills. Use `audio_duration` (seconds), or omit it for the family's "
           "default; accepting it here would silently return the default duration instead");
  if (Has(extra, "audio_duration")) {
    out.audio_duration_s = ReadNumber(extra, "audio_duration");
  }
  // OpenAI spells a duration `seconds` on the video surface; keep one spelling
  // per meaning here and accept both, with the native one winning.
  if (Has(extra, "duration") && !Has(extra, "audio_duration")) {
    out.audio_duration_s = ReadNumber(extra, "duration");
  }
  VT_CHECK(out.audio_duration_s >= 0.0,
           "speech request: `audio_duration` must be > 0 (omit it for the family's default)");
  if (Has(extra, "num_inference_steps")) {
    out.num_inference_steps = static_cast<int64_t>(ReadNumber(extra, "num_inference_steps"));
    VT_CHECK(out.num_inference_steps > 0,
             "speech request: `num_inference_steps` must be > 0 (omit it for the family's "
             "default)");
  }
  if (Has(extra, "guidance_scale")) {
    out.guidance_scale = ReadNumber(extra, "guidance_scale");
    VT_CHECK(out.guidance_scale >= 0.0, "speech request: `guidance_scale` must be >= 0");
    out.has_guidance_scale = true;
  }
  if (Has(extra, "seed")) out.seed = static_cast<int64_t>(ReadNumber(extra, "seed"));

  // ── The keys UPSTREAM refuses, refused here too (#672) ────────────────────
  //
  // SGLang-Omni's own `/v1/audio/speech` accepts these on the schema
  // (`serve/protocol.py:361-364`) and then REFUSES them for MiniMax-Music3 when
  // they are explicitly set — `models/minimax_music3/request_builders.py:14-19`
  // lists them and `:109-114` raises
  //   "MiniMax Music 3 does not support sampling parameters: …".
  // It refuses them because this model's AR stage has no temperature and no
  // nucleus branch at all: `_sample_top_k` is the ONLY sampler either stage
  // uses, and `_AR_SAMPLING_TOP_K` is a module constant of 50
  // (diffusers `encoders.py:48,94-103`). A knob that cannot be honoured must
  // not return 200.
  //
  // We were SILENT on them, which is the #925 failure class exactly: a caller
  // porting an SGLang or OpenAI recipe sends `temperature`, gets a well-formed
  // WAV, and has no way to learn the knob was dropped. The cost of that
  // silence is already recorded in this file, one refusal above.
  for (const char* key : {"temperature", "top_p", "top_k", "repetition_penalty"}) {
    VT_CHECK(!Has(extra, key),
             std::string("speech request: `") + key +
                 "` is not supported — MiniMax-Music3's autoregressive stage has no "
                 "temperature and no nucleus sampling; its only sampler is a fixed top-50 "
                 "draw (encoders.py:48,94-103), so the knob can be neither honoured nor "
                 "honestly ignored. Upstream refuses it by name too "
                 "(request_builders.py:109-114). Use `seed` to control the draw");
  }
  // SGLang-Omni spells the LENGTH as `max_new_tokens`, counted in 25 Hz audio
  // FRAMES rather than in seconds (`request_builders.py:56-68`,
  // `constants.py:4-5`). Our wire key is diffusers' `audio_duration`, in
  // SECONDS, because diffusers is this row's primary oracle. Two duration
  // spellings on one route is what #925 was, so the second one is REFUSED and
  // converted for the caller rather than accepted as a near-synonym.
  VT_CHECK(!Has(extra, "max_new_tokens"),
           "speech request: `max_new_tokens` is SGLang-Omni's spelling of the length, counted "
           "in 25 Hz audio frames (request_builders.py:56-68). This route takes "
           "`audio_duration` in SECONDS instead — divide by 25 — because accepting both would "
           "be two names for one meaning, and a duration key that is read by nobody is how "
           "this project shipped a 750x job behind a 200 (#925)");
  return out;
}

}  // namespace vllm::openai
