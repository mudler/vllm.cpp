// ROAD-V1-MM serving W1 — CPU gate for the OpenAI multimodal content-part
// parse + decode + processor route (the first CPU-reachable brick of wiring
// multimodal into the OpenAI server).
//
// Proves, entirely on CPU with only the processor CONFIG (no model weights):
//   1. INERTNESS (RED line): a bare-string chat `content` parses byte-identically
//      to before — content set, content_parts == nullopt, no mm parts, the
//      DefaultChatPromptFallback prompt unchanged.
//   2. base64 / data-URI DECODE round-trips against known vectors + the committed
//      fixtures.
//   3. PARSE: an array-form `content` ([{text},{image_url},{input_audio}]) parses
//      into typed ChatContentParts with the mirrored schema; the RED contrast is
//      that the pre-wiring parser only handled `content.is_string()`, so an array
//      produced content_parts == nullopt / empty content (asserted inert above).
//   4. ROUTE: an input_audio part (base64 of the committed whisper WAV) runs the
//      EXISTING Whisper processor -> input_features [80,3000] + 1500 placeholder
//      tokens; an image_url part (data: URI of the committed raw-RGB fixture) runs
//      the EXISTING Qwen3-VL processor -> grid [1,28,28] + 196 merged tokens.
//      Concrete numbers vs the M1/A1 processor-parity fixtures.
//
// Fixtures: tests/vllm/multimodal/fixtures/{qwen3vl,whisper_audio} (the same M1 /
// A1 processor-parity goldens). NAMED residual: the container-format image decode
// (PNG/JPEG -> RGB) — no codec is vendored, so the image route consumes the raw
// RGB the processor expects (as the single-sequence e2e path itself does).
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/request.h"

namespace {

using vllm::entrypoints::openai::ChatCompletionRequest;
using vllm::entrypoints::openai::ChatContentPart;
using vllm::entrypoints::openai::ChatMessage;

std::string ImgFixDir() { return std::string(MM_FIXTURE_DIR); }
std::string AudFixDir() { return std::string(MM_AUDIO_FIXTURE_DIR); }

std::vector<uint8_t> ReadBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

std::vector<float> ReadF32(const std::string& path) {
  const std::vector<uint8_t> b = ReadBytes(path);
  REQUIRE(b.size() % sizeof(float) == 0);
  std::vector<float> v(b.size() / sizeof(float));
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", path);
  nlohmann::json j;
  f >> j;
  return j;
}

// Standalone base64 encode (test-side; mirrors what an OpenAI client sends).
std::string EncodeBase64(const std::vector<uint8_t>& in) {
  static const char* kAlpha =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((in.size() + 2) / 3) * 4);
  size_t i = 0;
  for (; i + 3 <= in.size(); i += 3) {
    const uint32_t n = (static_cast<uint32_t>(in[i]) << 16) |
                       (static_cast<uint32_t>(in[i + 1]) << 8) |
                       static_cast<uint32_t>(in[i + 2]);
    out.push_back(kAlpha[(n >> 18) & 63]);
    out.push_back(kAlpha[(n >> 12) & 63]);
    out.push_back(kAlpha[(n >> 6) & 63]);
    out.push_back(kAlpha[n & 63]);
  }
  const size_t rem = in.size() - i;
  if (rem == 1) {
    const uint32_t n = static_cast<uint32_t>(in[i]) << 16;
    out.push_back(kAlpha[(n >> 18) & 63]);
    out.push_back(kAlpha[(n >> 12) & 63]);
    out.push_back('=');
    out.push_back('=');
  } else if (rem == 2) {
    const uint32_t n = (static_cast<uint32_t>(in[i]) << 16) |
                       (static_cast<uint32_t>(in[i + 1]) << 8);
    out.push_back(kAlpha[(n >> 18) & 63]);
    out.push_back(kAlpha[(n >> 12) & 63]);
    out.push_back(kAlpha[(n >> 6) & 63]);
    out.push_back('=');
  }
  return out;
}

vllm::multimodal::Qwen3VLProcessorConfig ImageConfigFromManifest(
    const nlohmann::json& m) {
  vllm::multimodal::Qwen3VLProcessorConfig cfg;
  const auto& c = m.at("config");
  cfg.patch_size = c.at("patch_size").get<int>();
  cfg.temporal_patch_size = c.at("temporal_patch_size").get<int>();
  cfg.merge_size = c.at("merge_size").get<int>();
  cfg.image_mean = c.at("image_mean")[0].get<double>();
  cfg.image_std = c.at("image_std")[0].get<double>();
  cfg.image_token_id = c.at("image_token_id").get<int32_t>();
  cfg.vision_start_token_id = c.at("vision_start_token_id").get<int32_t>();
  cfg.vision_end_token_id = c.at("vision_end_token_id").get<int32_t>();
  cfg.model_id = m.at("model_id").get<std::string>();
  return cfg;
}

vllm::multimodal::AudioProcessorConfig AudioConfigFromManifest(
    const nlohmann::json& m) {
  vllm::multimodal::AudioProcessorConfig cfg;
  const auto& c = m.at("feature_contract");
  cfg.n_fft = c.at("n_fft").get<int>();
  cfg.hop_length = c.at("hop_length").get<int>();
  cfg.n_mels = c.at("n_mels").get<int>();
  cfg.sampling_rate = c.at("sampling_rate").get<int>();
  cfg.chunk_length_s = c.at("chunk_length_s").get<int>();
  cfg.dither = c.at("dither").get<double>();
  cfg.max_source_positions = c.at("max_source_positions").get<int>();
  cfg.audio_placeholder_id =
      m.at("placeholder").at("audio_placeholder_id").get<int32_t>();
  cfg.model_id = m.at("model_id").get<std::string>();
  return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. INERTNESS (the RED line): a bare-string content is byte-identical to today.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm inertness: bare-string content unchanged") {
  const nlohmann::json j = {
      {"messages",
       {{{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Hello there"}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();
  REQUIRE(req.messages.size() == 2);

  // Bare-string content: content set, content_parts stays nullopt, no mm parts.
  for (const ChatMessage& m : req.messages) {
    CHECK(m.content.has_value());
    CHECK_FALSE(m.content_parts.has_value());
    CHECK_FALSE(vllm::entrypoints::openai::HasMultiModalParts(m));
  }
  CHECK(*req.messages[1].content == "Hello there");

  // The rendered fallback prompt is unchanged from the pure-text path.
  const std::string prompt =
      vllm::entrypoints::openai::DefaultChatPromptFallback(
          req.messages, /*add_generation_prompt=*/true, {});
  CHECK(prompt ==
        "system: You are helpful.\nuser: Hello there\nassistant:");
}

// ---------------------------------------------------------------------------
// 2. base64 / data-URI decode round-trips.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm base64 decode: known vectors + round-trip") {
  using vllm::entrypoints::openai::DecodeBase64;
  auto S = [](const char* s) {
    return std::vector<uint8_t>(s, s + std::strlen(s));
  };
  CHECK(DecodeBase64("TWFu") == S("Man"));      // no padding
  CHECK(DecodeBase64("TWE=") == S("Ma"));        // one pad
  CHECK(DecodeBase64("TQ==") == S("M"));         // two pad
  CHECK(DecodeBase64("aGVsbG8gd29ybGQ=") == S("hello world"));
  CHECK(DecodeBase64("aGVsbG8g\nd29ybGQ=") == S("hello world"));  // whitespace ok
  CHECK_THROWS(DecodeBase64("****"));             // invalid chars
  CHECK_THROWS(DecodeBase64("TWF"));              // truncated group

  // Round-trip a byte ramp through the test encoder + the production decoder.
  std::vector<uint8_t> ramp(256);
  for (int i = 0; i < 256; ++i) ramp[i] = static_cast<uint8_t>(i);
  CHECK(DecodeBase64(EncodeBase64(ramp)) == ramp);
}

TEST_CASE("chat-mm data-URI decode: header + base64 payload") {
  using vllm::entrypoints::openai::DecodeDataUri;
  const auto m = DecodeDataUri("data:image/png;base64,TWFu");
  CHECK(m.media_type == "image/png");
  CHECK(m.bytes == std::vector<uint8_t>({'M', 'a', 'n'}));
  CHECK_THROWS(DecodeDataUri("https://example.com/a.png"));  // http residual
  CHECK_THROWS(DecodeDataUri("data:image/png,rawtext"));     // non-base64
}

// ---------------------------------------------------------------------------
// 3+4. PARSE + ROUTE: input_audio content part -> Whisper processor.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm audio: input_audio part -> processor -> features + expansion") {
  const std::string dir = AudFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = AudioConfigFromManifest(manifest);
  const std::vector<float> mel_filters = ReadF32(dir + "/mel_filters_f32.bin");
  const std::vector<uint8_t> wav = ReadBytes(dir + "/audio_tone_16k_mono.wav");

  // Build the OpenAI chat request with an inline base64 WAV input_audio part.
  const std::string wav_b64 = EncodeBase64(wav);
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "text"}, {"text", "Transcribe this:"}},
           {{"type", "input_audio"},
            {"input_audio", {{"data", wav_b64}, {"format", "wav"}}}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();

  // PARSE: the array form populated content_parts (the pre-wiring parser dropped
  // it — the inert bare-string case above is the RED contrast).
  REQUIRE(req.messages.size() == 1);
  const ChatMessage& msg = req.messages[0];
  REQUIRE(msg.content_parts.has_value());
  REQUIRE(msg.content_parts->size() == 2);
  CHECK((*msg.content_parts)[0].type == "text");
  CHECK((*msg.content_parts)[0].text == "Transcribe this:");
  CHECK((*msg.content_parts)[1].type == "input_audio");
  CHECK((*msg.content_parts)[1].audio_format == "wav");
  CHECK(vllm::entrypoints::openai::HasMultiModalParts(msg));
  // The joined text span drives the existing text prompt path.
  CHECK(*msg.content == "Transcribe this:");

  // DECODE: the base64 payload round-trips to the exact committed WAV bytes.
  const auto media =
      vllm::entrypoints::openai::DecodeInputAudioPart((*msg.content_parts)[1]);
  CHECK(media.media_type == "audio/wav");
  REQUIRE(media.bytes == wav);

  // ROUTE: the existing Whisper processor produces the log-mel features + the
  // placeholder-expanded prompt.
  vllm::multimodal::WhisperAudioProcessor proc(cfg, mel_filters);
  const std::vector<int32_t> prompt_ids = {100, 200, cfg.audio_placeholder_id,
                                           300};
  const vllm::multimodal::MultiModalInputs mm =
      vllm::entrypoints::openai::RouteAudioWav(proc, media, prompt_ids);

  const int num_tokens = cfg.max_source_positions;  // 1500
  const int64_t n_frames =
      manifest.at("input_features").at("shape")[1].get<int64_t>();  // 3000
  REQUIRE(mm.mm_features.size() == 1);
  CHECK(mm.mm_features[0].modality == "audio");
  CHECK(mm.mm_features[0].offset == 2);           // placeholder position
  CHECK(mm.mm_features[0].length == num_tokens);  // 1500
  REQUIRE(mm.mm_features[0].audio_data != nullptr);
  CHECK(mm.mm_features[0].audio_data->n_mels == cfg.n_mels);        // 80
  CHECK(mm.mm_features[0].audio_data->n_frames == n_frames);        // 3000
  // Expanded prompt: 3 real tokens + 1500 placeholder copies.
  CHECK(mm.prompt_token_ids.size() ==
        static_cast<size_t>(3 + num_tokens));
  CHECK(mm.mm_features[0].mm_hash == manifest.at("mm_hash").get<std::string>());
}

// ---------------------------------------------------------------------------
// 4. ROUTE: image_url content part (data: URI of the raw-RGB fixture) ->
//    Qwen3-VL processor. Container-format decode (PNG->RGB) is the named residual.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm image: image_url part -> processor -> grid + expansion") {
  const std::string dir = ImgFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = ImageConfigFromManifest(manifest);
  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");
  REQUIRE(rgb.size() == static_cast<size_t>(H * W * 3));

  // A data: URI carrying the raw RGB bytes. `image/x-raw-rgb` documents that the
  // container-format decode (PNG/JPEG -> RGB) is the NAMED residual; the parse +
  // base64 + route seams are exact.
  const std::string uri = "data:image/x-raw-rgb;base64," + EncodeBase64(rgb);
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "text"}, {"text", "What is in this image?"}},
           {{"type", "image_url"}, {"image_url", {{"url", uri}}}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();

  REQUIRE(req.messages.size() == 1);
  const ChatMessage& msg = req.messages[0];
  REQUIRE(msg.content_parts.has_value());
  REQUIRE(msg.content_parts->size() == 2);
  CHECK((*msg.content_parts)[1].type == "image_url");
  CHECK(vllm::entrypoints::openai::HasMultiModalParts(msg));

  // DECODE: the data URI round-trips to the exact raw RGB bytes.
  const auto media =
      vllm::entrypoints::openai::DecodeImageUrlPart((*msg.content_parts)[1]);
  CHECK(media.media_type == "image/x-raw-rgb");
  REQUIRE(media.bytes.size() == rgb.size());
  REQUIRE(media.bytes == rgb);

  // ROUTE: the existing Qwen3-VL processor produces the grid + merged tokens.
  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);
  const std::vector<int32_t> prompt_ids = {5, cfg.image_token_id, 6};
  const vllm::multimodal::MultiModalInputs mm =
      vllm::entrypoints::openai::RouteImageRgb(proc, media.bytes.data(), H, W,
                                               prompt_ids);

  const auto g = manifest.at("image_grid_thw").at("values");
  const int64_t merged =
      (g[0].get<int64_t>() * g[1].get<int64_t>() * g[2].get<int64_t>()) /
      (cfg.merge_size * cfg.merge_size);  // (1*28*28)/4 = 196
  REQUIRE(mm.mm_features.size() == 1);
  CHECK(mm.mm_features[0].modality == "image");
  CHECK(mm.mm_features[0].offset == 1);          // placeholder position
  CHECK(mm.mm_features[0].length == merged);     // 196
  REQUIRE(mm.mm_features[0].data != nullptr);
  CHECK(mm.mm_features[0].data->image_grid_thw[0] == g[0].get<int64_t>());
  CHECK(mm.mm_features[0].data->image_grid_thw[1] == g[1].get<int64_t>());
  CHECK(mm.mm_features[0].data->image_grid_thw[2] == g[2].get<int64_t>());
  // Expanded prompt: 2 real tokens + 196 placeholder copies.
  CHECK(mm.prompt_token_ids.size() == static_cast<size_t>(2 + merged));
}

// ---------------------------------------------------------------------------
// 5. PLACEHOLDER STRINGS (MM-SERVE-ENGINE): the chat-template markers mirror
//    vLLM get_placeholder_str (qwen3_vl.py:1714 / qwen2_audio.py:333).
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm placeholder strings mirror vLLM get_placeholder_str") {
  using vllm::entrypoints::openai::AudioPlaceholderString;
  using vllm::entrypoints::openai::CollectChatPlaceholders;
  using vllm::entrypoints::openai::ImagePlaceholderString;
  using vllm::entrypoints::openai::VideoPlaceholderString;

  CHECK(ImagePlaceholderString() ==
        "<|vision_start|><|image_pad|><|vision_end|>");
  CHECK(VideoPlaceholderString() ==
        "<|vision_start|><|video_pad|><|vision_end|>");
  CHECK(AudioPlaceholderString(1) ==
        "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>");

  // CollectChatPlaceholders: one marker per mm part, audios numbered 1..k,
  // text parts contribute none.
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "text"}, {"text", "look:"}},
           {{"type", "image_url"}, {"image_url", {{"url", "data:x"}}}},
           {{"type", "input_audio"},
            {"input_audio", {{"data", "AA=="}, {"format", "wav"}}}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();
  const std::vector<std::string> markers =
      CollectChatPlaceholders(req.messages[0]);
  REQUIRE(markers.size() == 2);
  CHECK(markers[0] == "<|vision_start|><|image_pad|><|vision_end|>");
  CHECK(markers[1] == "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>");

  // A bare-string message contributes no markers (byte-identical text path).
  ChatMessage bare;
  bare.role = "user";
  bare.content = "hello";
  CHECK(CollectChatPlaceholders(bare).empty());
}

// ---------------------------------------------------------------------------
// 6. FULL CHAIN (MM-SERVE-ENGINE, the RED-first "New" gate): parse an image_url
//    request -> route through the processor -> placeholder-EXPANDED prompt ->
//    the engine mm path (InputProcessor::process_inputs_mm) receives the
//    MultiModalInputs -> the built Request carries the mm handles + the expanded
//    prompt with the CORRECT feature count. Everything UP TO the mm forward (the
//    GPU consumer = MM-SERVE-E2E residual) is asserted on CPU.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm full chain: image request -> engine request carries mm") {
  const std::string dir = ImgFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = ImageConfigFromManifest(manifest);
  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");

  // Parse an OpenAI chat request with an image_url part.
  const std::string uri = "data:image/x-raw-rgb;base64," + EncodeBase64(rgb);
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "text"}, {"text", "What is in this image?"}},
           {{"type", "image_url"}, {"image_url", {{"url", uri}}}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();
  const ChatMessage& msg = req.messages[0];
  REQUIRE(vllm::entrypoints::openai::HasMultiModalParts(msg));

  // ROUTE (brick 1): decode + processor -> placeholder-EXPANDED MultiModalInputs.
  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);
  const auto media =
      vllm::entrypoints::openai::DecodeImageUrlPart((*msg.content_parts)[1]);
  const std::vector<int32_t> base_prompt_ids = {5, cfg.image_token_id, 6};
  const vllm::multimodal::MultiModalInputs mm =
      vllm::entrypoints::openai::RouteImageRgb(proc, media.bytes.data(), H, W,
                                               base_prompt_ids);

  const auto g = manifest.at("image_grid_thw").at("values");
  const int64_t merged =
      (g[0].get<int64_t>() * g[1].get<int64_t>() * g[2].get<int64_t>()) /
      (cfg.merge_size * cfg.merge_size);  // 196
  // The placeholder-inserted prompt has EXACTLY the processor feature count of
  // image_pad slots (the count == the grid/feature count, MM-SERVE item 2).
  int64_t pad_slots = 0;
  for (int32_t id : mm.prompt_token_ids) {
    if (id == cfg.image_token_id) ++pad_slots;
  }
  CHECK(pad_slots == merged);
  CHECK(mm.mm_features.size() == 1);
  CHECK(mm.mm_features[0].length == merged);

  // ENGINE (MM-SERVE-ENGINE): process_inputs_mm is the exact call the engine mm
  // add_request overload makes. Feed the routed MultiModalInputs through it.
  const vllm::HfConfig hf = [] {
    vllm::HfConfig c;
    c.max_position_embeddings = 4096;
    c.raw = nlohmann::json::object();
    return c;
  }();
  static const vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(
      std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36/tokenizer.json");
  vllm::v1::InputProcessor input_proc(tok, hf);

  vllm::SamplingParams params;
  vllm::v1::EngineCoreRequest core_req = input_proc.process_inputs_mm(
      "chatcmpl-0", mm.prompt_token_ids, mm.mm_features, params);

  // The engine request carries BOTH the expanded prompt AND the mm handles.
  CHECK(core_req.prompt_token_ids == mm.prompt_token_ids);
  REQUIRE(core_req.mm_features.size() == 1);
  CHECK(core_req.mm_features[0].modality == "image");
  CHECK(core_req.mm_features[0].length == merged);
  CHECK(core_req.mm_features[0].mm_hash ==
        manifest.at("mm_hash").get<std::string>());
  REQUIRE(core_req.mm_features[0].data != nullptr);

  // ...and the built Request (what the scheduler/encoder-cache consume).
  vllm::v1::Request built = vllm::v1::Request::FromEngineCoreRequest(core_req);
  REQUIRE(built.mm_features.size() == 1);
  CHECK(built.mm_features[0].length == merged);
  CHECK(built.prompt_token_ids == mm.prompt_token_ids);
  // E2E residual (MM-SERVE-E2E): the mm forward on the GPU worker consuming
  // built.mm_features to produce token-correct output on Qwen3-VL-4B.
}

// ---------------------------------------------------------------------------
// 7. SEAM BODY (MM-SERVE-E2E, the RED-first gate): the ACTUAL MultiModalChatFn
//    the server sets — MakeQwen3VLImageChatFn — driving the placeholder->token-id
//    mapping through the REAL tokenizer + chat template + processor. Whereas the
//    full-chain test above hand-builds {5, image_token_id, 6}, this exercises
//    the seam that turns raw chat `messages` into the engine input:
//      messages -> marker-inject -> chat template -> tokenize (single image_pad
//      id) -> RouteImageRgb (EXPAND to N=196 + mm_features).
//    RED line: the text-only path (DefaultChatPromptFallback over the joined-text
//    content, image dropped) yields ZERO image tokens; the seam yields 196.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm seam body: MakeQwen3VLImageChatFn -> expanded engine input") {
  namespace oai = vllm::entrypoints::openai;
  const std::string dir = ImgFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  auto cfg = ImageConfigFromManifest(manifest);
  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");

  // The committed Qwen3.6 tokenizer shares the Qwen vision special-token markers
  // (<|vision_start|>/<|image_pad|>/<|vision_end|>) — at ITS ids. Align the
  // processor's image_token_id to what THIS tokenizer maps <|image_pad|> to so
  // RouteImageRgb expands exactly the id the tokenizer emits (in production both
  // are the model's own 151655).
  static const vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(
      std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36/tokenizer.json");
  const std::vector<int32_t> pad_ids =
      tok.EncodeWithSpecialTokens("<|image_pad|>");
  REQUIRE(pad_ids.size() == 1u);
  cfg.image_token_id = pad_ids[0];

  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);

  const int64_t merged =
      (manifest.at("image_grid_thw").at("values")[0].get<int64_t>() *
       manifest.at("image_grid_thw").at("values")[1].get<int64_t>() *
       manifest.at("image_grid_thw").at("values")[2].get<int64_t>()) /
      (cfg.merge_size * cfg.merge_size);  // 196
  REQUIRE(merged == 196);

  // A raw-RGB codec (the container-format decode is the named residual).
  oai::ImageCodecFn codec =
      [&](const oai::DecodedMedia& media) -> oai::DecodedImageRgb {
    oai::DecodedImageRgb out;
    out.rgb = media.bytes;
    out.height = H;
    out.width = W;
    return out;
  };

  // The seam body, wired exactly as server_main.cpp does (real tokenizer + the
  // chat-prompt renderer + the seam's own processing info, #607 L2). The config
  // is DEFAULT here — 999 per modality — so this pre-existing single-image case
  // is byte-identical to before the limits landed.
  const vllm::MultiModalConfig default_mm_config;
  const vllm::multimodal::BaseProcessingInfo info(
      default_mm_config, oai::Qwen3VLChatSupportedMmLimits());
  auto mm_fn = oai::MakeQwen3VLImageChatFn(
      proc, tok, oai::DefaultChatPromptFallback, std::move(codec), info);

  // An OpenAI image chat request (image then text, matching the M0 golden order).
  const std::string uri = "data:image/x-raw-rgb;base64," + EncodeBase64(rgb);
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "image_url"}, {"image_url", {{"url", uri}}}},
           {{"type", "text"}, {"text", "What is in this image?"}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();

  // RED: the text-only path (what a server WITHOUT the seam renders) drops the
  // image — the tokenized prompt carries ZERO image tokens.
  const std::string text_prompt = oai::DefaultChatPromptFallback(
      req.messages, /*add_generation_prompt=*/true, {});
  const std::vector<int32_t> text_ids = tok.EncodeWithSpecialTokens(text_prompt);
  int64_t red_pad = 0;
  for (int32_t id : text_ids)
    if (id == cfg.image_token_id) ++red_pad;
  CHECK(red_pad == 0);

  // GREEN: the seam turns the SAME request into the placeholder-EXPANDED engine
  // input — 196 image tokens + one image mm_feature.
  const std::optional<vllm::multimodal::MultiModalInputs> mm =
      mm_fn(req.messages);
  REQUIRE(mm.has_value());
  int64_t pad_slots = 0;
  for (int32_t id : mm->prompt_token_ids)
    if (id == cfg.image_token_id) ++pad_slots;
  CHECK(pad_slots == merged);
  REQUIRE(mm->mm_features.size() == 1u);
  CHECK(mm->mm_features[0].modality == "image");
  CHECK(mm->mm_features[0].length == merged);
  CHECK(mm->mm_features[0].mm_hash == manifest.at("mm_hash").get<std::string>());
  REQUIRE(mm->mm_features[0].data != nullptr);

  // A text-only request (no mm parts) leaves the seam a no-op (nullopt) — the
  // server then renders the byte-identical text path.
  const nlohmann::json jt = {
      {"messages", {{{"role", "user"}, {"content", "hello there"}}}}};
  const ChatCompletionRequest text_req = jt.get<ChatCompletionRequest>();
  CHECK_FALSE(mm_fn(text_req.messages).has_value());
}

// ---------------------------------------------------------------------------
// 8. THE LIMIT CHECK ON THE CHAT PATH (#607 wave L2, closing #686).
//
//    Ported from vllm/entrypoints/chat_utils.py:630-662 @ 5559679229bc — the
//    tracker that counts every parsed multimodal content part and validates the
//    RUNNING count before the request reaches the engine — over
//    BaseProcessingInfo::{AllowedMmLimits,ValidateNumItems,ValidateTrackedChatItem}
//    (#607 L1, multimodal/processing/context.py:392-405,409-428).
//
//    RED line, and it is a BEHAVIOURAL one rather than an absence: before this
//    wave the seam located the FIRST image_url part and `break`ed
//    (PRE-L2 chat_mm.cpp:256-268; the same loop is chat_mm.cpp:313-323 today,
//    now preceded by the check at :311), so a three-image request was neither
//    served nor refused, it was quietly reduced to one — no
//    error, no warning, a confident answer about a subset of the input (#686).
//    Every CHECK_THROWS below fails against that code, because it does not
//    throw; it returns the first image's 196 tokens and discards two.
// ---------------------------------------------------------------------------
namespace {

// One `image_url` content part carrying the committed raw-RGB fixture. The
// payload is only decoded AFTER the limit check passes, so the refusal cases
// never reach the codec — which is the point of validating first.
ChatContentPart ImagePart(const std::string& uri) {
  ChatContentPart p;
  p.type = "image_url";
  p.url = uri;
  return p;
}

ChatContentPart TypedPart(const std::string& type) {
  ChatContentPart p;
  p.type = type;
  p.url = "data:application/octet-stream;base64,AAAA";
  return p;
}

// N image parts in ONE user message.
std::vector<ChatMessage> ImageMessages(int n, const std::string& uri) {
  ChatMessage m;
  m.role = "user";
  m.content = std::string("what is in these?");
  std::vector<ChatContentPart> parts;
  for (int i = 0; i < n; ++i) parts.push_back(ImagePart(uri));
  m.content_parts = std::move(parts);
  std::vector<ChatMessage> out;
  out.push_back(std::move(m));
  return out;
}

}  // namespace

TEST_CASE("chat mm limits: ChatPartModality mirrors MM_PARSER_MAP") {
  namespace oai = vllm::entrypoints::openai;
  // chat_utils.py:1478-1483.
  CHECK(oai::ChatPartModality(ImagePart("data:image/x-raw-rgb;base64,AA")) ==
        std::optional<std::string>("image"));
  CHECK(oai::ChatPartModality(TypedPart("video_url")) ==
        std::optional<std::string>("video"));
  CHECK(oai::ChatPartModality(TypedPart("audio_url")) ==
        std::optional<std::string>("audio"));
  CHECK(oai::ChatPartModality(TypedPart("input_audio")) ==
        std::optional<std::string>("audio"));
  // Text is not multimodal input; it never reaches a limit.
  CHECK_FALSE(oai::ChatPartModality(TypedPart("text")).has_value());
  CHECK_FALSE(oai::ChatPartModality(TypedPart("refusal")).has_value());
  // `*_embeds` passes THROUGH verbatim (chat_utils.py:635): the suffix strip and
  // the enable_mm_embeds escape belong to ValidateTrackedChatItem, not here.
  CHECK(oai::ChatPartModality(TypedPart("image_embeds")) ==
        std::optional<std::string>("image_embeds"));
}

TEST_CASE("chat mm limits: THREE images are REFUSED, not truncated (#686)") {
  namespace oai = vllm::entrypoints::openai;
  const std::string dir = ImgFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  auto cfg = ImageConfigFromManifest(manifest);
  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");
  static const vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(
      std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36/tokenizer.json");
  const std::vector<int32_t> pad_ids =
      tok.EncodeWithSpecialTokens("<|image_pad|>");
  REQUIRE(pad_ids.size() == 1u);
  cfg.image_token_id = pad_ids[0];
  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);

  oai::ImageCodecFn codec =
      [&](const oai::DecodedMedia& media) -> oai::DecodedImageRgb {
    oai::DecodedImageRgb out;
    out.rgb = media.bytes;
    out.height = H;
    out.width = W;
    return out;
  };
  const std::string uri = "data:image/x-raw-rgb;base64," + EncodeBase64(rgb);

  // The DEFAULT config: no --limit-mm-per-prompt, no --language-model-only. The
  // refusal below therefore comes from the SEAM's own ceiling
  // (Qwen3VLChatSupportedMmLimits: one image), folded by min() against the
  // user's 999 (context.py:392-405). #686 asked exactly this question — "the
  // model's own supported limit, the min() fold's other operand; today nothing
  // declares one" — and this is the declaration.
  const vllm::MultiModalConfig default_cfg;
  const vllm::multimodal::BaseProcessingInfo info(
      default_cfg, oai::Qwen3VLChatSupportedMmLimits());
  CHECK(info.AllowedMmLimits().at("image") == 1);

  auto mm_fn = oai::MakeQwen3VLImageChatFn(
      proc, tok, oai::DefaultChatPromptFallback, codec, info);

  // ONE image still works, byte-identically: 196 expanded tokens.
  const std::optional<vllm::multimodal::MultiModalInputs> one =
      mm_fn(ImageMessages(1, uri));
  REQUIRE(one.has_value());
  CHECK(one->mm_features.size() == 1u);

  // THREE images are REFUSED. Before L2 this returned the first image's 196
  // tokens and dropped two, which is the defect #686 records.
  CHECK_THROWS_AS(mm_fn(ImageMessages(3, uri)), vllm::v1::InputValidationError);
  try {
    mm_fn(ImageMessages(3, uri));
    FAIL("expected the seam to refuse three images");
  } catch (const vllm::v1::InputValidationError& e) {
    // Upstream's message VERBATIM (context.py:421-423).
    CHECK(std::string(e.what()) ==
          "At most 1 image(s) may be provided in one prompt.");
    // ...and WITHOUT the "--limit-mm-per-prompt" hint (:425-426): raising the
    // user's limit would not help, because it is the seam that caps at one, so
    // a hint would send the user to a flag that cannot fix their request.
    CHECK(std::string(e.what()).find("--limit-mm-per-prompt") ==
          std::string::npos);
  }

  // TWO images across TWO messages are refused too: the tracker's count is
  // cumulative (chat_utils.py:648-652), so splitting the turn does not evade it.
  std::vector<ChatMessage> split = ImageMessages(1, uri);
  split.push_back(ImageMessages(1, uri)[0]);
  CHECK_THROWS_AS(mm_fn(split), vllm::v1::InputValidationError);

  // A VIDEO part through the IMAGE seam is refused rather than dropped: video is
  // absent from the seam's supported limits, which context.py:414-415 reads as
  // "not supported", limit 0.
  {
    ChatMessage m;
    m.role = "user";
    m.content = std::string("describe");
    m.content_parts = std::vector<ChatContentPart>{TypedPart("video_url")};
    std::vector<ChatMessage> msgs;
    msgs.push_back(std::move(m));
    try {
      mm_fn(msgs);
      FAIL("expected the seam to refuse a video part");
    } catch (const vllm::v1::InputValidationError& e) {
      CHECK(std::string(e.what()) ==
            "At most 0 video(s) may be provided in one prompt.");
    }
  }
}

TEST_CASE("chat mm limits: --limit-mm-per-prompt and --language-model-only bite") {
  namespace oai = vllm::entrypoints::openai;
  const std::string dir = ImgFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  auto cfg = ImageConfigFromManifest(manifest);
  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");
  static const vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(
      std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36/tokenizer.json");
  cfg.image_token_id = tok.EncodeWithSpecialTokens("<|image_pad|>")[0];
  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);
  oai::ImageCodecFn codec =
      [&](const oai::DecodedMedia& media) -> oai::DecodedImageRgb {
    oai::DecodedImageRgb out;
    out.rgb = media.bytes;
    out.height = H;
    out.width = W;
    return out;
  };
  const std::string uri = "data:image/x-raw-rgb;base64," + EncodeBase64(rgb);

  // (a) --language-model-only: EVERY modality limit becomes 0
  // (multimodal.py:78-80,326-327), so the ONE image a default server serves is
  // refused. This is the flag's main observable effect and the half a port most
  // easily leaves out, because omitting it breaks nothing a text-only workload
  // would notice.
  {
    vllm::MultiModalConfig lm_only;
    lm_only.language_model_only = true;
    const vllm::multimodal::BaseProcessingInfo info(
        lm_only, oai::Qwen3VLChatSupportedMmLimits());
    auto mm_fn = oai::MakeQwen3VLImageChatFn(
        proc, tok, oai::DefaultChatPromptFallback, codec, info);
    try {
      mm_fn(ImageMessages(1, uri));
      FAIL("expected --language-model-only to refuse an image request");
    } catch (const vllm::v1::InputValidationError& e) {
      // HERE the hint IS appended (context.py:425-426): the seam CAN take this
      // image, the configuration is what refused it, so raising the limit helps.
      CHECK(std::string(e.what()) ==
            "At most 0 image(s) may be provided in one prompt. "
            "Set `--limit-mm-per-prompt` to increase this limit.");
    }
    // A text-only request is untouched by the flag — it limits multimodal INPUT,
    // it does not refuse the server.
    ChatMessage text;
    text.role = "user";
    text.content = std::string("hello there");
    std::vector<ChatMessage> text_msgs;
    text_msgs.push_back(std::move(text));
    CHECK_FALSE(mm_fn(text_msgs).has_value());
  }

  // (b) --limit-mm-per-prompt '{"image": 0}' reaches the same refusal by the
  // other route, which is the whole reason L1 ported the limits before the flag:
  // the flag is sugar, the limits are the mechanism.
  {
    vllm::MultiModalConfig zeroed;
    zeroed.limit_per_prompt = {{"image", 0}};
    const vllm::multimodal::BaseProcessingInfo info(
        zeroed, oai::Qwen3VLChatSupportedMmLimits());
    auto mm_fn = oai::MakeQwen3VLImageChatFn(
        proc, tok, oai::DefaultChatPromptFallback, codec, info);
    CHECK_THROWS_AS(mm_fn(ImageMessages(1, uri)),
                    vllm::v1::InputValidationError);
  }

  // (c) A user limit can only LOWER the seam's ceiling, never raise it
  // (context.py:392-405 folds by min). `image=99` still refuses the second
  // image.
  {
    vllm::MultiModalConfig raised;
    raised.limit_per_prompt = {{"image", 99}};
    const vllm::multimodal::BaseProcessingInfo info(
        raised, oai::Qwen3VLChatSupportedMmLimits());
    CHECK(info.AllowedMmLimits().at("image") == 1);
    auto mm_fn = oai::MakeQwen3VLImageChatFn(
        proc, tok, oai::DefaultChatPromptFallback, codec, info);
    CHECK(mm_fn(ImageMessages(1, uri)).has_value());
    CHECK_THROWS_AS(mm_fn(ImageMessages(2, uri)),
                    vllm::v1::InputValidationError);
  }
}
