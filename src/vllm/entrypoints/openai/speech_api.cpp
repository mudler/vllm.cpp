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

// What `extra_params` binds to when a body carries none. It must NOT be the top
// level: binding it there is what made `extra_params` a replacement rather than
// a second place to look. See the `Owner` resolver in ParseSpeechRequest.
const nlohmann::json& EmptyObject() {
  static const nlohmann::json kEmpty = nlohmann::json::object();
  return kEmpty;
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

  // `extra_params` is a SECOND place to look, NEVER a replacement for the top
  // level. vLLM-Omni nests the generation knobs there and OpenAI puts its own
  // at the top level, so a client may reasonably do either -- and one that did
  // both used to lose every key in whichever object this did not bind. An
  // `extra_params` object as empty as `{}` was enough to drop a top-level
  // `audio_duration`, leave the field at its 0.0 sentinel, and have
  // MiniMax-Music3 answer with its 60 s default behind a 200: #925's cost with
  // #925's own guard in the tree and unable to fire (#1315).
  //
  // `extra_params` WINS where both carry a key, which is the precedence the
  // video route already documents (video_api.cpp:216-225), so the two routes
  // resolve a knob the same way rather than two ways.
  //
  // EVERY request key below resolves through `Owner` -- the CONTENT fields as
  // much as the knobs and the refusals -- and that is the repair rather than
  // tidiness.
  //
  // The first version of this change routed only the knobs and the refusals
  // through `Owner` and left `model`, `input`, `text`, `language`, `lyrics`,
  // `description`, `prompt` and `reference_audio` reading `json` directly. That
  // left the SAME hole open for the content keys, and it defeated three
  // refusals by nesting (#1336): a nested `text` was dropped instead of
  // reaching `minimax_music3_speech.cpp:440-446`, whose own last words are
  // "rather than having it silently dropped"; a nested `language` instead of
  // `:456-460`; and a nested `reference_audio` instead of being decoded, which
  // made a family whose `requires_reference_audio()` is true answer
  // "`reference_audio` ... is required" (api_server.cpp:511-516) to a caller
  // who had supplied one.
  //
  // `json` STAYS in scope, because `Owner` needs it as the fallback. So this is
  // a convention that this comment states and that the tests pin, NOT a
  // structural impossibility. An earlier draft claimed the single-placement
  // handle was gone; it was not, and that claim is what let eight direct reads
  // sit under it.
  const nlohmann::json& extra =
      (json.contains("extra_params") && json.at("extra_params").is_object())
          ? json.at("extra_params")
          : EmptyObject();
  const auto Owner = [&json, &extra](const char* key) -> const nlohmann::json* {
    if (Has(extra, key)) return &extra;
    if (Has(json, key)) return &json;
    return nullptr;
  };

  SpeechRequest out;
  if (const nlohmann::json* o = Owner("model")) {
    out.model = ReadString(*o, "model");
    VT_CHECK(!out.model.empty(), "speech request: `model` must not be empty");
  }
  // OpenAI's field is `input`; `text` is accepted as the native spelling so a
  // caller does not have to know which surface it is talking to.
  if (const nlohmann::json* o = Owner("input")) out.text = ReadString(*o, "input");
  if (const nlohmann::json* o = Owner("text")) {
    const std::string text = ReadString(*o, "text");
    VT_CHECK(out.text.empty() || out.text == text,
             "speech request: `input` and `text` are the same field and disagree; supply one");
    out.text = text;
  }
  if (const nlohmann::json* o = Owner("language")) out.language = ReadString(*o, "language");
  if (const nlohmann::json* o = Owner("lyrics")) out.lyrics = ReadString(*o, "lyrics");
  if (const nlohmann::json* o = Owner("description")) {
    out.description = ReadString(*o, "description");
  }
  // vLLM-Omni and the video route both carry the music description under
  // `prompt`; accept it as the documented alias rather than making a caller
  // learn a second name for the same string. This is the one ALIAS this route
  // accepts, and the rule it follows is stated at the `instructions` refusal
  // below: both spellings carry the same value in the same units for every
  // family this route can load.
  if (const nlohmann::json* o = Owner("prompt")) {
    const std::string prompt = ReadString(*o, "prompt");
    VT_CHECK(out.description.empty() || out.description == prompt,
             "speech request: `prompt` and `description` are the same field and disagree; "
             "supply one");
    out.description = prompt;
  }

  VT_CHECK(out.has_any_text(),
           "speech request: one of `input` (a spoken utterance), `lyrics` or `description` "
           "is required; which of them applies is the loaded family's decision");

  // ── The named residuals. Each is refused rather than ignored. ─────────────
  VT_CHECK(Owner("voice") == nullptr,
           "speech request: `voice` is not supported — no registered speech family exposes "
           "named voices, and there is no voice-enumeration endpoint to pick one from. A "
           "family that clones a voice takes `reference_audio` instead");
  VT_CHECK(Owner("speed") == nullptr,
           "speech request: `speed` is not supported — no registered family implements a rate "
           "control, and honouring it silently would return audio at the wrong tempo");
  VT_CHECK(Owner("stream") == nullptr && Owner("stream_format") == nullptr,
           "speech request: streaming is not supported — MiniMax-Music3 generates the whole "
           "song before the first sample exists (.agents/specs/minimax-music3.md §7), and "
           "buffering it to emit chunks would be a stream in name only");
  if (const nlohmann::json* format_owner = Owner("response_format")) {
    const std::string format = ReadString(*format_owner, "response_format");
    VT_CHECK(format == "wav",
             "speech request: `response_format` '" + format +
                 "' is not supported; only \"wav\" is, because no mp3/opus/aac/flac encoder "
                 "is vendored and relabelling RIFF bytes would be worse than refusing");
  }

  if (const nlohmann::json* o = Owner("reference_audio")) {
    const std::string value = ReadString(*o, "reference_audio");
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

  // `audio_duration_s` is the name of the C++/C-API FIELD this key fills, not a
  // wire key, and it is the one misspelling a caller reaches for by reading the
  // struct instead of the docs. Dropping it silently hands back the family's
  // default duration — 60 s for MiniMax-Music3 — while returning 200, so the
  // caller cannot tell. That is exactly how the Music3 e2e gate spent three
  // multi-hour runs generating a 60 s song it had asked 0.1 s for (#852): 2 AR
  // frames became 1500 and the run was misread as a hung weight load. REFUSED
  // and NAMED, like every other unsupported field above.
  VT_CHECK(Owner("audio_duration_s") == nullptr,
           "speech request: `audio_duration_s` is not a request key — it is the name of the "
           "field it fills. Use `audio_duration` (seconds), or omit it for the family's "
           "default; accepting it here would silently return the default duration instead");
  // OpenAI spells a duration `seconds` on the video surface; keep one spelling
  // per meaning here and accept both, with the native one winning.
  const char* duration_key = "audio_duration";
  const nlohmann::json* duration_owner = Owner(duration_key);
  if (duration_owner == nullptr) {
    duration_key = "duration";
    duration_owner = Owner(duration_key);
  }
  if (duration_owner != nullptr) {
    out.audio_duration_s = ReadNumber(*duration_owner, duration_key);
  }
  // The predicate is `>= 0.0` and the message says so. It used to promise
  // "must be > 0" while accepting an explicit `0`, which then resolved to the
  // family's 60 s default and never printed the sentence that would have
  // explained it (#1338). ZERO means "omitted, take the family's default" --
  // the same split `minimax_music3_speech.cpp:465-474` argues at the family
  // layer -- and only a NEGATIVE duration is impossible.
  VT_CHECK(out.audio_duration_s >= 0.0,
           "speech request: `audio_duration` must not be NEGATIVE; omit it, or send 0, to take "
           "the family's default duration");
  if (const nlohmann::json* o = Owner("num_inference_steps")) {
    out.num_inference_steps = static_cast<int64_t>(ReadNumber(*o, "num_inference_steps"));
    VT_CHECK(out.num_inference_steps > 0,
             "speech request: `num_inference_steps` must be > 0 (omit it for the family's "
             "default)");
  }
  if (const nlohmann::json* o = Owner("guidance_scale")) {
    out.guidance_scale = ReadNumber(*o, "guidance_scale");
    VT_CHECK(out.guidance_scale >= 0.0, "speech request: `guidance_scale` must be >= 0");
    out.has_guidance_scale = true;
  }
  if (const nlohmann::json* o = Owner("seed")) {
    out.seed = static_cast<int64_t>(ReadNumber(*o, "seed"));
  }

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
    VT_CHECK(Owner(key) == nullptr,
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
  VT_CHECK(Owner("max_new_tokens") == nullptr,
           "speech request: `max_new_tokens` is SGLang-Omni's spelling of the length, counted "
           "in 25 Hz audio frames (request_builders.py:56-68). This route takes "
           "`audio_duration` in SECONDS instead — divide by 25 — because accepting both would "
           "be two names for one meaning, and a duration key that is read by nobody is how "
           "this project shipped a 750x job behind a 200 (#925)");

  // ── The rest of upstream's named set (#1315) ──────────────────────────────
  //
  // `_build_tts_params` (sglang_omni/serve/speech_service.py:737-779) forwards
  // these wire keys into the model's builder, and `_UNSUPPORTED_TTS_PARAMS`
  // (models/minimax_music3/request_builders.py:20-30) makes
  // `_validate_tts_contract` (:71-81) raise
  //   "MiniMax Music 3 does not support speech parameters: <names>"
  // for each one. We were SILENT on all of them, which is #925 again: a key the
  // server will not honour must not come back behind a 200.
  //
  // `language` is in upstream's set and is deliberately NOT here. It is already
  // refused BY NAME one layer down (minimax_music3_speech.cpp:456-460), which
  // is the layer this tree puts family-specific refusals at and where a family
  // that HAS a language can still take it. Moving it up would break
  // IndexTTS-2.5.
  //
  // The boundary #925 drew is unchanged: only the keys upstream NAMES are
  // refused, so an unknown key still parses and `extra_params` stays
  // forward-compatible.
  VT_CHECK(Owner("speaker") == nullptr,
           "speech request: `speaker` is SGLang-Omni's alias for `voice` "
           "(protocol.py:337-339), and neither is supported — no registered speech family "
           "exposes named voices. Refusing one spelling and dropping the other would return "
           "a 200 for half of one field. A family that clones a voice takes `reference_audio` "
           "instead");
  // `instructions` is REFUSED rather than accepted as an alias, although
  // upstream honours it. TWO reasons hold. A secondary oracle never becomes the
  // mirror source (AGENTS.md), and SGLang-Omni is only the secondary here
  // because vLLM registers no `/v1/audio/speech` at all. And `instructions`
  // means style-and-emotion on OpenAI's own createSpeech and for a TTS family
  // (protocol.py:348), and the music CAPTION for this one, so a global alias on
  // a SHARED route would bake one family's meaning in.
  //
  // A third reason an earlier draft gave, "one meaning keeps one name on this
  // route", does NOT hold. `prompt` four refusals above is the counter-example:
  // it is accepted as a second spelling of `description`, with a comment saying
  // so. The rule the two cases actually follow is narrower. An ALIAS is
  // accepted when both spellings carry the same value in the same UNITS and
  // mean the same thing for EVERY family this route can load.
  // `prompt`/`description` qualify. `instructions` fails on MEANING, and
  // `max_new_tokens` fails on UNITS, 25 Hz frames against seconds, so aliasing
  // it would need a silent conversion.
  //
  // The refusal therefore names BOTH readings. Redirecting every caller to
  // `description` is right only for one who learned the key from SGLang-Omni.
  // For one who learned it from OpenAI it would move a VOICE-STYLE string into
  // the music caption, which is the conflation this refusal exists to prevent.
  VT_CHECK(Owner("instructions") == nullptr,
           "speech request: `instructions` names two different things and this route honours "
           "neither. OpenAI's createSpeech uses it for VOICE STYLE and emotion, and no "
           "registered speech family exposes a style control, so there is nothing to send if "
           "that is what you meant. `instructions` is SGLang-Omni's spelling of the music "
           "CAPTION, the string MiniMax-Music3 assembles into `<|caption_start|>` and which "
           "upstream requires non-empty (request_builders.py:104-106), and this route calls "
           "THAT `description`; send `description` if the caption is what you meant. Dropping "
           "it silently would lose the whole music description of a request that named it");
  VT_CHECK(Owner("ref_audio") == nullptr,
           "speech request: `ref_audio` is SGLang-Omni's spelling of the reference clip "
           "(protocol.py:351), where it is a path or a URL. This route takes "
           "`reference_audio`, a `data:` URL carrying a 16-bit PCM mono WAV, because the "
           "server and the client need not share a filesystem; send that instead");
  VT_CHECK(Owner("ref_text") == nullptr,
           "speech request: `ref_text` (the transcript of a reference clip) is not supported "
           "— no registered family conditions on a reference TRANSCRIPT, and upstream refuses "
           "it by name for this model too (request_builders.py:20-30,71-81). Supply the text "
           "to synthesize in `input`, or the sung text in `lyrics`");
  VT_CHECK(Owner("task_type") == nullptr,
           "speech request: `task_type` (`Base`/`CustomVoice`/`VoiceDesign`) is not supported "
           "— this route has one synthesis mode per loaded family and no task selector, so "
           "the field can select nothing. Upstream refuses it by name too "
           "(request_builders.py:20-30,71-81)");
  VT_CHECK(Owner("x_vector_only_mode") == nullptr,
           "speech request: `x_vector_only_mode` is not supported — no registered family "
           "exposes a speaker-embedding-only path to switch into, and upstream refuses it by "
           "name for this model too (request_builders.py:20-30,71-81)");
  VT_CHECK(Owner("initial_codec_chunk_frames") == nullptr,
           "speech request: `initial_codec_chunk_frames` is not supported — the chunk "
           "schedule is the family's own (MiniMax-Music3 fixes it at 200 frames with a 100 "
           "hop, constants.py:9-10), so the value can be neither honoured nor honestly "
           "ignored. Upstream refuses it by name too (request_builders.py:20-30,71-81)");
  // The two LENGTH spellings. These are the ones that repeat #925's cost
  // exactly: a duration key nobody reads is how a short request becomes a 60 s
  // song.
  for (const char* key : {"token_count", "duration_tokens"}) {
    VT_CHECK(Owner(key) == nullptr,
             std::string("speech request: `") + key +
                 "` is SGLang-Omni's LENGTH, counted in duration tokens "
                 "(protocol.py:355-356). This route takes `audio_duration` in SECONDS "
                 "instead, because two spellings of one meaning is what let a length key go "
                 "unread and turn a short request into the family's 60 s default (#925). "
                 "Upstream refuses it by name for this model too "
                 "(request_builders.py:20-30,71-81)");
  }
  return out;
}

}  // namespace vllm::openai
