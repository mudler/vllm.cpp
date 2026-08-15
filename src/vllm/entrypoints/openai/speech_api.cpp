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
  return out;
}

}  // namespace vllm::openai
