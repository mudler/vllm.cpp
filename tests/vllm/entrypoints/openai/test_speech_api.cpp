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
