// `/v1/audio/speech`: the request contract (W6 of #672).
//
// The route's synthesis is a callback the server is given, so this file is the
// whole of the endpoint's own logic and it is pure — the same split that makes
// tests/vllm/entrypoints/openai/test_video_api.cpp testable without a model.
//
// Every value asserted below DIFFERS from the field's default, so a passing
// check proves the parser was reached rather than that a default happened to
// match.
#include "vllm/entrypoints/openai/speech_api.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

using vllm::openai::ParseSpeechRequest;
using vllm::openai::SpeechRequest;

namespace {

// A 44-byte-header 16-bit PCM mono WAV as a base64 `data:` URL. Built here
// rather than committed so the sample values are visible to the assertions.
std::string ReferenceWavDataUri(const std::vector<int16_t>& samples, uint32_t rate) {
  std::string wav;
  const uint32_t payload = static_cast<uint32_t>(samples.size() * 2);
  const auto put32 = [&wav](uint32_t v) {
    for (int i = 0; i < 4; ++i) wav += static_cast<char>((v >> (8 * i)) & 0xFF);
  };
  const auto put16 = [&wav](uint16_t v) {
    for (int i = 0; i < 2; ++i) wav += static_cast<char>((v >> (8 * i)) & 0xFF);
  };
  wav += "RIFF";
  put32(36u + payload);
  wav += "WAVE";
  wav += "fmt ";
  put32(16u);
  put16(1u);      // PCM
  put16(1u);      // mono
  put32(rate);
  put32(rate * 2);
  put16(2u);
  put16(16u);
  wav += "data";
  put32(payload);
  for (const int16_t sample : samples) put16(static_cast<uint16_t>(sample));

  static const char* kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string b64;
  for (size_t i = 0; i < wav.size(); i += 3) {
    const uint32_t a = static_cast<unsigned char>(wav[i]);
    const uint32_t b = i + 1 < wav.size() ? static_cast<unsigned char>(wav[i + 1]) : 0u;
    const uint32_t c = i + 2 < wav.size() ? static_cast<unsigned char>(wav[i + 2]) : 0u;
    const uint32_t triple = (a << 16) | (b << 8) | c;
    b64 += kAlphabet[(triple >> 18) & 0x3F];
    b64 += kAlphabet[(triple >> 12) & 0x3F];
    b64 += i + 1 < wav.size() ? kAlphabet[(triple >> 6) & 0x3F] : '=';
    b64 += i + 2 < wav.size() ? kAlphabet[triple & 0x3F] : '=';
  }
  return "data:audio/wav;base64," + b64;
}

}  // namespace

TEST_CASE("speech api: the MUSIC inputs are two NAMED fields, not one `input`") {
  const SpeechRequest music = ParseSpeechRequest(R"({
    "model": "minimax-music3",
    "lyrics": "[Verse]\nMorning light\n",
    "description": "Genre: acoustic pop. BPM: 96."
  })");
  CHECK(music.model == "minimax-music3");
  CHECK(music.lyrics == "[Verse]\nMorning light\n");
  CHECK(music.description == "Genre: acoustic pop. BPM: 96.");
  // The one-utterance field stays EMPTY: nothing merged the two into it, which
  // is the whole reason they are separate.
  CHECK(music.text.empty());
  CHECK(music.has_any_text());

  // OpenAI's own spelling for a one-utterance family still works unchanged.
  const SpeechRequest speech = ParseSpeechRequest(R"({"input": "hello there"})");
  CHECK(speech.text == "hello there");
  CHECK(speech.lyrics.empty());
  CHECK(speech.description.empty());

  // `prompt` is the documented alias for `description` (vLLM-Omni and the video
  // route both carry the description there).
  const SpeechRequest aliased =
      ParseSpeechRequest(R"({"prompt": "Genre: lo-fi", "lyrics": "[Chorus]\nx\n"})");
  CHECK(aliased.description == "Genre: lo-fi");
  CHECK(aliased.lyrics == "[Chorus]\nx\n");
  // The alias DISAGREEING with the field it aliases is a 400, never a silent
  // winner — the vllm_video_model_params.partition precedent.
  CHECK_THROWS(ParseSpeechRequest(R"({"prompt":"a","description":"b","lyrics":"x"})"));
  CHECK_THROWS(ParseSpeechRequest(R"({"input":"a","text":"b"})"));
}

TEST_CASE("speech api: the generation controls default to the FAMILY's, and explicit values win") {
  const SpeechRequest minimal = ParseSpeechRequest(R"({"lyrics":"[Verse]\nx\n"})");
  CHECK(minimal.audio_duration_s == doctest::Approx(0.0));  // => family default
  CHECK(minimal.num_inference_steps == 0);                  // => family default
  CHECK_FALSE(minimal.has_guidance_scale);                  // => family default
  CHECK(minimal.seed == 0);

  const SpeechRequest flat = ParseSpeechRequest(R"({
    "lyrics": "[Verse]\nx\n", "audio_duration": 12.5,
    "num_inference_steps": 4, "guidance_scale": 2.75, "seed": 7
  })");
  CHECK(flat.audio_duration_s == doctest::Approx(12.5));
  CHECK(flat.num_inference_steps == 4);
  CHECK(flat.has_guidance_scale);
  CHECK(flat.guidance_scale == doctest::Approx(2.75));
  CHECK(flat.seed == 7);

  // vLLM-Omni nests them under extra_params; both spellings must work, exactly
  // as they do on /v1/videos.
  const SpeechRequest nested = ParseSpeechRequest(R"({
    "lyrics": "[Verse]\nx\n",
    "extra_params": {"audio_duration": 30.0, "num_inference_steps": 8, "seed": 99}
  })");
  CHECK(nested.audio_duration_s == doctest::Approx(30.0));
  CHECK(nested.num_inference_steps == 8);
  CHECK(nested.seed == 99);

  // 0 IS A LEGAL guidance scale — it selects the unconditional branch — so an
  // explicit 0 must be distinguishable from "not supplied".
  const SpeechRequest zero =
      ParseSpeechRequest(R"({"lyrics":"[Verse]\nx\n","guidance_scale":0})");
  CHECK(zero.has_guidance_scale);
  CHECK(zero.guidance_scale == doctest::Approx(0.0));

  CHECK_THROWS(ParseSpeechRequest(R"({"lyrics":"x","num_inference_steps":0})"));
  CHECK_THROWS(ParseSpeechRequest(R"({"lyrics":"x","audio_duration":-1})"));
  CHECK_THROWS(ParseSpeechRequest(R"({"lyrics":"x","guidance_scale":-0.5})"));

  // `audio_duration_s` is the C++/C-API FIELD name, not the wire key, and a
  // request carrying it was accepted with the key silently dropped — so the
  // caller got the family's 60 s default instead of the duration it asked for.
  // That is not a small error: it cost the MiniMax-Music3 e2e gate three
  // multi-hour runs (#852), because 0.1 s became 60 s, 2 AR frames became 1500,
  // and the run was read as a hung weight load. A duration the server will not
  // honour is REFUSED, exactly like every other unsupported field in this file.
  CHECK_THROWS(ParseSpeechRequest(R"({"lyrics":"x","audio_duration_s":0.1})"));
  CHECK_THROWS(
      ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"audio_duration_s":0.1}})"));
  // The honoured spellings are unaffected by the guard.
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","audio_duration":0.1})").audio_duration_s ==
        doctest::Approx(0.1));
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","duration":0.1})").audio_duration_s ==
        doctest::Approx(0.1));
}

TEST_CASE("speech api: every unsupported field is REFUSED, never ignored") {
  // Ignoring any of these returns a 200 carrying audio the caller did not ask
  // for, which is the failure each refusal exists to prevent.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","voice":"alloy"})"),
                    doctest::Contains("`voice` is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","speed":1.5})"),
                    doctest::Contains("`speed` is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","stream":true})"),
                    doctest::Contains("streaming is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","stream_format":"sse"})"),
                    doctest::Contains("streaming is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","response_format":"mp3"})"),
                    doctest::Contains("no mp3/opus/aac/flac encoder"));
  // "wav" is accepted, so the refusal is about the FORMAT and not about the
  // field existing.
  CHECK_NOTHROW(ParseSpeechRequest(R"({"input":"hi","response_format":"wav"})"));
}

// The parity sweep of #672: SGLang-Omni serves this same model on this same
// route and REFUSES four sampling knobs and its own length spelling. We were
// SILENT on all five, which is the #925 failure class one case above — a knob
// the server will not honour must not come back behind a 200.
TEST_CASE("speech api: the sampling knobs UPSTREAM refuses are refused here too") {
  // `request_builders.py:14-19,109-114`. This model's AR stage has ONE sampler,
  // a fixed top-50 draw (`encoders.py:48,94-103`): there is no temperature to
  // set and no nucleus branch to widen, so honouring any of these is impossible
  // and ignoring them is dishonest.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","temperature":0.7})"),
                    doctest::Contains("`temperature` is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","top_p":0.9})"),
                    doctest::Contains("`top_p` is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","top_k":40})"),
                    doctest::Contains("`top_k` is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","repetition_penalty":1.1})"),
                    doctest::Contains("`repetition_penalty` is not supported"));
  // Nested under `extra_params` too, because that is the second place every
  // other generation knob on this route is read from.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"temperature":0.7}})"),
                    doctest::Contains("`temperature` is not supported"));

  // `max_new_tokens` is SGLang-Omni's LENGTH, in 25 Hz audio frames rather than
  // in seconds (`request_builders.py:56-68`). Accepting it as a near-synonym of
  // `audio_duration` would put two spellings of one meaning on one route, and
  // dropping it silently is how a 250-frame (10 s) request becomes the family's
  // 60 s default. The refusal names the key to use AND the conversion.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","max_new_tokens":250})"),
                    doctest::Contains("`max_new_tokens` is SGLang-Omni's spelling"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","max_new_tokens":250})"),
                    doctest::Contains("`audio_duration` in SECONDS"));

  // The NEGATIVE side, because a guard that fired on an ordinary request would
  // refuse every real one: a body carrying none of the five still parses, and
  // the knobs this route DOES honour are untouched by the guards above.
  const SpeechRequest ok = ParseSpeechRequest(
      R"({"lyrics":"[Verse]\nx\n","description":"pop","audio_duration":12.5,
          "num_inference_steps":8,"guidance_scale":1.7,"seed":7})");
  CHECK(ok.audio_duration_s == doctest::Approx(12.5));
  CHECK(ok.num_inference_steps == 8);
  CHECK(ok.guidance_scale == doctest::Approx(1.7));
  CHECK(ok.seed == 7);
}

TEST_CASE("speech api: a malformed body is a 400, never a silent default") {
  CHECK_THROWS(ParseSpeechRequest("not json"));
  CHECK_THROWS(ParseSpeechRequest("[]"));
  CHECK_THROWS(ParseSpeechRequest("{}"));                      // no text of any kind
  CHECK_THROWS(ParseSpeechRequest(R"({"input": 5})"));         // wrong type
  CHECK_THROWS(ParseSpeechRequest(R"({"lyrics": ["a"]})"));    // wrong type
  CHECK_THROWS(ParseSpeechRequest(R"({"input":"hi","model":""})"));
  CHECK_THROWS(ParseSpeechRequest(R"({"input":"hi","seed":"soon"})"));
}

TEST_CASE("speech api: a reference clip decodes from a data: URL to mono f32") {
  // Values chosen so each maps to a DISTINCT float: 0 and +/-32768 would both
  // survive a sign or scale error.
  const std::string uri = ReferenceWavDataUri({16384, -16384, 8192, -8192}, 22050);
  const SpeechRequest r =
      ParseSpeechRequest(R"({"input":"hi","reference_audio":")" + uri + R"("})");
  REQUIRE(r.reference_audio.size() == 4);
  CHECK(r.reference_sample_rate == 22050);  // default is 0
  CHECK(r.reference_audio[0] == doctest::Approx(0.5));
  CHECK(r.reference_audio[1] == doctest::Approx(-0.5));
  CHECK(r.reference_audio[2] == doctest::Approx(0.25));
  CHECK(r.reference_audio[3] == doctest::Approx(-0.25));

  // A filesystem path is refused rather than stat-ed: the server and the client
  // need not share one, and a later "no such file" names the wrong layer.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","reference_audio":"/tmp/voice.wav"})"),
                    doctest::Contains("must be a `data:` URL"));
  // A STEREO or non-PCM clip is refused BY NAME rather than reinterpreted: a
  // mis-parsed clip conditions the synthesis on noise and still returns 200.
  std::string stereo = ReferenceWavDataUri({1, 2, 3, 4}, 16000);
  // Flip the channel-count field (offset 22) by rebuilding through the raw path
  // is fiddly, so assert the two refusals the decoder can be shown directly.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","reference_audio":"data:audio/wav;base64,UklGRg=="})"),
                    doctest::Contains("too short to be a WAV"));
  CHECK_THROWS(ParseSpeechRequest(R"({"input":"hi","reference_audio":"data:audio/wav;base64,####"})"));
  CHECK_FALSE(stereo.empty());
}

// `extra_params` was a REPLACEMENT for the top level, not a second place to
// look: `const json& extra = has_extra_params ? json["extra_params"] : json`
// binds ONE of the two, so every knob and every refusal above stopped seeing
// the top level the moment a body carried an `extra_params` object at all.
//
// That is the #925 mechanism still live, in its worst spelling: a body that
// nests `seed` and puts `audio_duration` where OpenAI puts it got a 200 and the
// family's 60 s default, and neither the `audio_duration_s` guard nor the five
// #953 guards could fire, because none of them was looking at the top level any
// more. The video route already resolves this the right way and says so —
// `extra_params` first, then the top level (`video_api.cpp:216-225`).
TEST_CASE("speech api: `extra_params` is a SECOND place to look, never a replacement") {
  // THE COST CASE. Before the fix this returned `audio_duration_s == 0`, which
  // MiniMax-Music3 reads as "omitted" and answers with 60 s of music.
  const SpeechRequest split = ParseSpeechRequest(
      R"({"lyrics":"[Verse]\nx\n","extra_params":{"seed":7},"audio_duration":0.1})");
  CHECK(split.audio_duration_s == doctest::Approx(0.1));
  CHECK(split.seed == 7);

  // The other direction: a knob nested while the rest stays flat.
  const SpeechRequest other = ParseSpeechRequest(
      R"({"lyrics":"[Verse]\nx\n","extra_params":{"audio_duration":0.25},
          "num_inference_steps":8,"guidance_scale":1.7,"seed":9})");
  CHECK(other.audio_duration_s == doctest::Approx(0.25));
  CHECK(other.num_inference_steps == 8);
  CHECK(other.guidance_scale == doctest::Approx(1.7));
  CHECK(other.seed == 9);

  // PRECEDENCE, stated rather than left to whichever branch runs first:
  // `extra_params` wins, exactly as `video_api.cpp:216-225` documents.
  CHECK(ParseSpeechRequest(
            R"({"lyrics":"x","audio_duration":9.0,"extra_params":{"audio_duration":0.5}})")
            .audio_duration_s == doctest::Approx(0.5));

  // And every refusal sees BOTH placements. A guard that stops firing because
  // the body happens to carry an `extra_params` object is not a guard.
  CHECK_THROWS_WITH(
      ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"seed":7},"audio_duration_s":0.1})"),
      doctest::Contains("`audio_duration_s` is not a request key"));
  CHECK_THROWS_WITH(
      ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"seed":7},"temperature":0.7})"),
      doctest::Contains("`temperature` is not supported"));
  CHECK_THROWS_WITH(
      ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"seed":7},"max_new_tokens":250})"),
      doctest::Contains("`max_new_tokens` is SGLang-Omni's spelling"));
  CHECK_THROWS_WITH(
      ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"seed":7},"voice":"alloy"})"),
      doctest::Contains("`voice` is not supported"));
  // ... and the nested placement of a field that was only ever read at the top
  // level, which is the same hole seen from the other side.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"voice":"alloy"}})"),
                    doctest::Contains("`voice` is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"stream":true}})"),
                    doctest::Contains("streaming is not supported"));
  CHECK_THROWS_WITH(
      ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"response_format":"mp3"}})"),
      doctest::Contains("no mp3/opus/aac/flac encoder"));

  // The NEGATIVE control: an empty or absent `extra_params` changes nothing.
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{},"audio_duration":0.1})")
            .audio_duration_s == doctest::Approx(0.1));
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","audio_duration":0.1})").audio_duration_s ==
        doctest::Approx(0.1));
}

// The rest of the sweep #953 started. `_build_tts_params`
// (`sglang_omni/serve/speech_service.py:737-779`) forwards these keys from the
// wire into the model's builder, and `_UNSUPPORTED_TTS_PARAMS`
// (`request_builders.py:20-30`) makes `_validate_tts_contract` (`:71-81`) raise
// "MiniMax Music 3 does not support speech parameters: <names>" for each one.
// We were SILENT on all of them, which is #925 again: the caller sends a key
// the server will not honour and gets a 200 that cannot say so.
//
// `language` is in upstream's set too and is NOT here, because it is already
// refused BY NAME one layer down, where a family that HAS a language can still
// take it (`minimax_music3_speech.cpp:456-460`). That is the layer this tree
// puts family-specific refusals at, and moving it up would break IndexTTS-2.5.
TEST_CASE("speech api: the SPEECH parameters upstream refuses by name are refused here too") {
  // `speaker` is SGLang-Omni's declared ALIAS for `voice`
  // (`protocol.py:337-339`), so refusing `voice` and dropping `speaker` refused
  // one spelling of one field and honoured nothing for the other.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","speaker":"alloy"})"),
                    doctest::Contains("`speaker` is SGLang-Omni's alias for `voice`"));

  // `instructions` is upstream's spelling of the music CAPTION — the string it
  // assembles into `<|caption_start|>`, and which it REQUIRES to be non-empty
  // (`request_builders.py:104-106`). Dropping it silently loses the whole music
  // description of anyone porting the upstream recipe.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","instructions":"Genre: lo-fi"})"),
                    doctest::Contains("`instructions` is SGLang-Omni's spelling"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","instructions":"Genre: lo-fi"})"),
                    doctest::Contains("`description`"));

  // The reference-clip spellings. This route takes `reference_audio` as a
  // `data:` URL; upstream's `ref_audio` is a path or URL and its `ref_text` the
  // transcript beside it.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","ref_audio":"/tmp/v.wav"})"),
                    doctest::Contains("`ref_audio` is SGLang-Omni's spelling"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","ref_audio":"/tmp/v.wav"})"),
                    doctest::Contains("`reference_audio`"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","ref_text":"hello"})"),
                    doctest::Contains("`ref_text` (the transcript of a reference clip) is not "
                                      "supported"));

  // The two LENGTH spellings, which are the ones that cost the runs: a duration
  // key nobody reads is exactly how a short request becomes a 60 s song.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","token_count":250})"),
                    doctest::Contains("`token_count` is SGLang-Omni's LENGTH"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","token_count":250})"),
                    doctest::Contains("`audio_duration` in SECONDS"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","duration_tokens":250})"),
                    doctest::Contains("`duration_tokens` is SGLang-Omni's LENGTH"));

  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","task_type":"CustomVoice"})"),
                    doctest::Contains("`task_type` (`Base`/`CustomVoice`/`VoiceDesign`) is not "
                                      "supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","x_vector_only_mode":true})"),
                    doctest::Contains("`x_vector_only_mode` is not supported"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","initial_codec_chunk_frames":4})"),
                    doctest::Contains("`initial_codec_chunk_frames` is not supported"));

  // Nested under `extra_params` as well, for the same reason as every other
  // guard on this route.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"instructions":"lo-fi"}})"),
                    doctest::Contains("`instructions` is SGLang-Omni's spelling"));

  // THE NEGATIVE CONTROL, and it is the whole boundary of this guard: an
  // unknown key is still ACCEPTED. #925 chose to refuse the NAMED near-misses
  // rather than every unrecognised key, so that `extra_params` stays
  // forward-compatible; a sweep that turned into "refuse everything" would have
  // changed that decision without saying so.
  CHECK_NOTHROW(ParseSpeechRequest(R"({"lyrics":"x","some_future_knob":1})"));
  CHECK_NOTHROW(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"some_future_knob":1}})"));
  // And the fields this route DOES read keep their meaning.
  const SpeechRequest ok = ParseSpeechRequest(
      R"({"lyrics":"[Verse]\nx\n","description":"pop","language":"en","audio_duration":12.5})");
  CHECK(ok.description == "pop");
  CHECK(ok.language == "en");
  CHECK(ok.audio_duration_s == doctest::Approx(12.5));
}

// The CONTENT keys were the half `#1315` did not route, and leaving them on the
// bare `json` handle left the SAME hole open (#1336). The comment above them
// claimed every read resolved through `Owner`; eight of them did not, and three
// refusals downstream were defeatable by moving the key one level down.
//
// Each case below is a body that returned 200 with the key silently dropped
// before this repair, and the three marked REFUSAL-DEFEAT are the ones that
// bypassed a guard rather than only losing a value.
TEST_CASE("speech api: the CONTENT keys resolve through BOTH placements, not the top level only") {
  // REFUSAL-DEFEAT 1. `text` at the top level reaches Music3's refusal at
  // `minimax_music3_speech.cpp:440-446`, whose last words are "rather than
  // having it silently dropped". Nested, it produced exactly that drop.
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"text":"hello"}})").text == "hello");
  // REFUSAL-DEFEAT 2. `language` nested skipped `:456-460`.
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"language":"en"}})").language == "en");
  // REFUSAL-DEFEAT 3. A nested reference clip was dropped, so a family whose
  // `requires_reference_audio()` is true answered "`reference_audio` is
  // required" to a caller who had supplied one. The decoded samples are
  // asserted, not just the size, so a clip read at the wrong offset still reds.
  const std::string uri = ReferenceWavDataUri({16384, -16384}, 22050);
  const SpeechRequest nested_clip =
      ParseSpeechRequest(R"({"input":"hi","extra_params":{"reference_audio":")" + uri + R"("}})");
  REQUIRE(nested_clip.reference_audio.size() == 2);
  CHECK(nested_clip.reference_sample_rate == 22050);
  CHECK(nested_clip.reference_audio[0] == doctest::Approx(0.5));
  CHECK(nested_clip.reference_audio[1] == doctest::Approx(-0.5));

  // The remaining content keys, each in the nested placement, each asserted
  // against a value that differs from the field's default.
  CHECK(ParseSpeechRequest(R"({"extra_params":{"input":"hi"}})").text == "hi");
  CHECK(ParseSpeechRequest(R"({"extra_params":{"lyrics":"[Verse]\nx\n"}})").lyrics ==
        "[Verse]\nx\n");
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"description":"pop"}})").description ==
        "pop");
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"prompt":"lo-fi"}})").description ==
        "lo-fi");
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"model":"minimax-music3"}})").model ==
        "minimax-music3");

  // PRECEDENCE, for a content key as for a knob: `extra_params` wins, the same
  // way `video_api.cpp:216-225` resolves it.
  CHECK(ParseSpeechRequest(R"({"lyrics":"top","extra_params":{"lyrics":"nested"}})").lyrics ==
        "nested");
  CHECK(ParseSpeechRequest(R"({"input":"top","extra_params":{"input":"nested"}})").text ==
        "nested");

  // The two "same field, two spellings" guards see ACROSS the placements too,
  // which is the property that made routing them through `Owner` the repair
  // rather than eight separate lookups.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"a","extra_params":{"text":"b"}})"),
                    doctest::Contains("`input` and `text` are the same field and disagree"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"description":"a","extra_params":{"prompt":"b"}})"),
                    doctest::Contains("`prompt` and `description` are the same field and "
                                      "disagree"));
  // ... and they still ACCEPT the agreeing body, so the guard is not "refuse
  // whenever both placements carry the key".
  CHECK(ParseSpeechRequest(R"({"input":"a","extra_params":{"text":"a"}})").text == "a");

  // `model` keeps its emptiness check in the nested placement as well.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"input":"hi","extra_params":{"model":""}})"),
                    doctest::Contains("`model` must not be empty"));

  // THE #925 BOUNDARY, restated for the content sweep: an unknown key in EITHER
  // placement still parses. Routing the content keys through `Owner` is a
  // change of WHERE a known key is read, never a change of which keys are
  // known, and a strict whitelist would red this line.
  CHECK_NOTHROW(ParseSpeechRequest(R"({"lyrics":"x","some_future_knob":1})"));
  CHECK_NOTHROW(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"some_future_knob":1}})"));
}

// `instructions` is refused, and the refusal has to name BOTH readings of the
// key. It is OpenAI's own createSpeech field for VOICE STYLE, and this route
// declares itself "OpenAI's createSpeech, extended with the two MUSIC inputs"
// (`api_server.cpp:496-497`), so a caller who sends it may well mean the style
// and not SGLang-Omni's caption. Redirecting that caller to `description` would
// move a voice-style string into the music caption, which is the exact
// conflation the refusal exists to prevent.
TEST_CASE("speech api: the `instructions` refusal names the OpenAI reading as well as upstream's") {
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","instructions":"speak warmly"})"),
                    doctest::Contains("VOICE STYLE"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","instructions":"speak warmly"})"),
                    doctest::Contains("no registered speech family exposes a style control"));
  // The upstream reading and its redirect are still there.
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","instructions":"Genre: lo-fi"})"),
                    doctest::Contains("`instructions` is SGLang-Omni's spelling of the music"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","instructions":"Genre: lo-fi"})"),
                    doctest::Contains("`description`"));
}

// `audio_duration` (#1338). The predicate has always been `>= 0.0`, so an
// explicit 0 parses and resolves to the family's default; the message promised
// "> 0", a rule this route does not have, and was printed by nothing.
TEST_CASE("speech api: the `audio_duration` refusal states the guard the code implements") {
  // ZERO is accepted and means "take the family's default", the same split
  // `minimax_music3_speech.cpp:465-474` argues at the family layer.
  CHECK(ParseSpeechRequest(R"({"lyrics":"x","audio_duration":0})").audio_duration_s ==
        doctest::Approx(0.0));
  // NEGATIVE is refused, and the message says NEGATIVE rather than "> 0".
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","audio_duration":-1.0})"),
                    doctest::Contains("must not be NEGATIVE"));
  CHECK_THROWS_WITH(ParseSpeechRequest(R"({"lyrics":"x","extra_params":{"audio_duration":-1.0}})"),
                    doctest::Contains("must not be NEGATIVE"));
}
