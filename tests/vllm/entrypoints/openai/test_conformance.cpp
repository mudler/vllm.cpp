// M3.6 — OpenAI server CONFORMANCE suite. A comprehensive end-to-end validation
// of the OpenAI HTTP server against the API contract, exercised over the REAL
// cpp-httplib server on an ephemeral port (via httplib::Client), NOT the
// socket-free dispatch layer. It reuses the ServerHarness pattern from
// tests/vllm/entrypoints/openai/test_api_server.cpp (a fully-wired synthetic
// LLMEngine — tiny hybrid-MoE Qwen3.6 + the tiny BPE fixture, greedy/deterministic)
// and drives EVERY serving feature built in M3.1–M3.4 through real HTTP:
//
//   1. /v1/completions          — non-stream + stream (SSE), length/stop finish,
//                                  usage consistency, echo (accepted), n=1.
//   2. /v1/chat/completions      — non-stream + stream cadence, multi-turn, the
//                                  chat-template/prompt seam reflecting messages.
//   3. tool calling              — tool_choice auto is RELAXED (a plain reply is
//                                  NOT forced into a tool call); named/required
//                                  requests are accepted + the response SHAPE is
//                                  contract-correct.
//   4. grammars / structured out — response_format json_object / json_schema are
//                                  accepted + shaped; a malformed json_schema → 400.
//   5. errors                    — malformed JSON → 400, unknown model → 404,
//                                  unknown route → 404, each carrying the OpenAI
//                                  ErrorResponse{error:{message,type,code}}.
//   6. endpoints                 — /v1/models, /health, /version.
//   7. UTF-8 / robustness        — a completion over an emoji-capable vocab never
//                                  500s + serializes valid UTF-8 (the SanitizeUtf8
//                                  boundary); concurrent clients all succeed (the
//                                  engine mutex).
//
// WHAT IS ASSERTED vs NOTED-AS-REAL-MODEL-PENDING: the synthetic argmax over
// random weights cannot deterministically emit a `<tool_call>{...}</tool_call>`
// wrapper (the tiny vocab has no `<`/`{`/`"` tokens), so the "auto emits a tool
// call → populated tool_calls + finish_reason=tool_calls" and the streaming
// tool-call delta cadence are NOT drivable over HTTP here. Those response SHAPES
// are proven exhaustively at the serving layer in test_serving.cpp
// (ShapeChatMessage / ShapeChatDelta) and the constrained-decode WIRING end to
// end in tests/vllm/v1/structured_output/test_response_format_e2e.cpp. Here we
// assert the request is ACCEPTED + the contract shape + the auto-not-forced case,
// and mark the forced-tool-output assertion as real-weights-pending (dgx).
#include "vllm/entrypoints/openai/api_server.h"

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5MoeWeights;
using vllm::SchedulerConfig;
using vllm::entrypoints::openai::ApiServer;
using vllm::entrypoints::openai::ChatCompletionToolsParam;
using vllm::entrypoints::openai::ChatMessage;
using vllm::entrypoints::openai::OpenAIServingChat;
using vllm::entrypoints::openai::OpenAIServingCompletion;
using vllm::entrypoints::openai::OpenAIServingModels;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCore;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::GPUModelRunner;
using vllm::v1::init_none_hash;
using vllm::v1::InputProcessor;
using vllm::v1::KVCacheConfig;
using vllm::v1::LLMEngine;
using vllm::v1::MambaSpec;
using vllm::v1::OutputProcessor;
using vllm::v1::Scheduler;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

// ─── Synthetic weights (mirrors test_api_server.cpp) ─────────────────────────
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

constexpr int kBlockSize = 32;
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 32;

// vocab_size is a parameter: 22 = the plain fixture (ids 0..21, every id decodes
// to valid UTF-8 → exact stream/non-stream text parity); 24 = the emoji fixture
// (adds the two half-emoji BYTE tokens 22/23) which CAN emit a split multibyte
// sequence, exercising the SanitizeUtf8 serving boundary over HTTP.
HfConfig MakeConfig(int vocab) {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = vocab;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = kMaxModelLen;
  c.raw = json::object();  // no eos_token_id -> generation runs to max_tokens.
  return c;
}

vllm::MoeBlockWeights MakeMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeOwned(DType::kBF16, {H, I}, s + 100 + e * 7));
    m.expert_up.push_back(MakeOwned(DType::kBF16, {H, I}, s + 200 + e * 7));
    m.expert_down.push_back(MakeOwned(DType::kBF16, {I, H}, s + 300 + e * 7));
  }
  m.shared_gate_proj = MakeOwned(DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(DType::kBF16, {Is, H}, s + 5);
  return m;
}

Qwen3_5MoeWeights MakeWeights(const HfConfig& c) {
  Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.moe = MakeMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

KVCacheConfig MakeKvConfig(const HfConfig& c) {
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);
  const int Hv = static_cast<int>(c.linear_num_value_heads);
  const int Dv = static_cast<int>(c.linear_value_head_dim);
  const int Dk = static_cast<int>(c.linear_key_head_dim);
  const int Kw = static_cast<int>(c.linear_conv_kernel_dim);
  const int key_dim = static_cast<int>(c.linear_num_key_heads) * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa3"},
      std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh, DType::kF32));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn0", "gdn1", "gdn2"},
      std::make_shared<MambaSpec>(
          kBlockSize,
          std::vector<std::vector<int64_t>>{{conv_dim, Kw - 1},
                                            {Hv, Dv, Dk}},
          std::vector<DType>{DType::kF32, DType::kF32}));
  return kv;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The tiny BPE fixture. `with_emoji` adds the two half-emoji byte tokens (ids
// 22/23) so a synthetic argmax can emit a split multibyte run (the UTF-8 case).
Tokenizer BuildFixture(bool with_emoji) {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_conf_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},    {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},    {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11},  {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15},  {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  if (with_emoji) {
    vocab[MapBytesToUnicode("\xF0\x9F")] = 22;  // 2 of a 4-byte emoji lead.
    vocab[MapBytesToUnicode("\x8C\x8D")] = 23;  // 2 continuation bytes.
  }
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture(/*with_emoji=*/false);
  return tok;
}
const Tokenizer& EmojiFixture() {
  static const Tokenizer tok = BuildFixture(/*with_emoji=*/true);
  return tok;
}

// A fully-wired serving stack + ApiServer over the synthetic engine (mirrors
// test_api_server.cpp::ServerHarness). The chat prompt seam CAPTURES the last
// prompt it produced so a test can assert the request's messages actually
// reached the template/prompt builder (the "chat template applied" contract).
struct ServerHarness {
  ServerHarness(const HfConfig& c, const Qwen3_5MoeWeights& w,
                const Tokenizer& tok)
      : scheduler(MakeSchedulerConfig(), MakeKvConfig(c), kBlockSize,
                  /*enable_caching=*/true),
        runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, kMaxModelLen * 8),
        executor(runner),
        engine_core(scheduler, executor),
        input_processor(tok, c),
        output_processor(&tok),
        engine(input_processor, engine_core, output_processor, Hasher()),
        models("test-model"),
        completion(engine, "test-model"),
        chat(engine, "test-model",
             [this](const std::vector<ChatMessage>& messages, bool,
                    const std::vector<ChatCompletionToolsParam>&) {
               // In-vocab seam: concatenate every message's content (the M3.2
               // template renderer replaces this; here it stands in AND records
               // what it received so a test can prove the messages flowed in).
               std::string p;
               for (const ChatMessage& m : messages)
                 if (m.content.has_value()) p += *m.content;
               {
                 std::lock_guard<std::mutex> lk(prompt_mu);
                 last_chat_prompt = p;
               }
               return p;
             }),
        server(completion, chat, models, "9.9.9") {}

  static SchedulerConfig MakeSchedulerConfig() {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 8;
    cfg.max_num_batched_tokens = kMaxModelLen * 8;
    cfg.enable_chunked_prefill = true;
    cfg.max_model_len = kMaxModelLen;
    cfg.watermark = 0.0;
    return cfg;
  }
  static vllm::v1::BlockHasher Hasher() {
    static bool init = false;
    if (!init) {
      init_none_hash(sha256_cbor);
      init = true;
    }
    return get_request_block_hasher(kBlockSize, sha256_cbor);
  }

  std::string CapturedChatPrompt() {
    std::lock_guard<std::mutex> lk(prompt_mu);
    return last_chat_prompt;
  }

  Scheduler scheduler;
  GPUModelRunner runner;
  Executor executor;
  EngineCore engine_core;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  LLMEngine engine;
  std::mutex prompt_mu;
  std::string last_chat_prompt;
  OpenAIServingModels models;
  OpenAIServingCompletion completion;
  OpenAIServingChat chat;
  ApiServer server;
};

// RAII: bind the server to an ephemeral port, run its accept loop on a background
// thread, wait until it is accepting, and stop+join on destruction. Mirrors the
// smoke-test lifecycle in test_api_server.cpp so every case is flake-free.
struct LiveServer {
  explicit LiveServer(ApiServer& s) : server(s) {
    port = server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    thread = std::thread([this]() { server.serve(); });
    for (int i = 0; i < 500 && !server.is_running(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(server.is_running());
  }
  ~LiveServer() {
    server.stop();
    if (thread.joinable()) thread.join();
  }
  LiveServer(const LiveServer&) = delete;
  LiveServer& operator=(const LiveServer&) = delete;

  httplib::Client Client() const {
    httplib::Client c("127.0.0.1", port);
    c.set_connection_timeout(5, 0);
    c.set_read_timeout(30, 0);
    return c;
  }

  ApiServer& server;
  int port = -1;
  std::thread thread;
};

// Split an SSE body (the concatenated `data: {...}\n\n` frames httplib streamed)
// into per-frame strings WITHOUT the trailing blank line (i.e. "data: {json}" /
// "data: [DONE]").
std::vector<std::string> ParseSseFrames(const std::string& body) {
  std::vector<std::string> frames;
  size_t pos = 0;
  while (pos < body.size()) {
    const size_t end = body.find("\n\n", pos);
    if (end == std::string::npos) break;
    frames.push_back(body.substr(pos, end - pos));
    pos = end + 2;
  }
  return frames;
}

// The JSON payload of a `data: {json}` frame.
json SseFramePayload(const std::string& frame) {
  REQUIRE(frame.rfind("data: ", 0) == 0);
  return json::parse(frame.substr(6));
}

// A completion POST body with greedy sampling (deterministic).
std::string CompletionBody(const std::string& prompt, int max_tokens,
                           bool stream) {
  json j = {{"prompt", prompt},
            {"max_tokens", max_tokens},
            {"temperature", 0.0},
            {"stream", stream}};
  return j.dump();
}

}  // namespace

// ═══════════════════════════ 1. Completions ═════════════════════════════════

TEST_CASE("conformance: /v1/completions non-stream is a contract-correct text_completion") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  auto res = client.Post("/v1/completions", CompletionBody("hello", 5, false),
                         "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  CHECK(res->get_header_value("Content-Type").rfind("application/json", 0) == 0);

  json j = json::parse(res->body);
  CHECK(j.at("object") == "text_completion");
  CHECK(j.at("model") == "test-model");
  CHECK(j.at("id").get<std::string>().rfind("cmpl-", 0) == 0);
  REQUIRE(j.at("choices").size() == 1);
  const auto& choice = j.at("choices").at(0);
  CHECK(choice.at("index") == 0);
  CHECK_FALSE(choice.at("text").get<std::string>().empty());
  // No eos in the fixture → max_tokens length stop.
  CHECK(choice.at("finish_reason") == "length");

  // usage: prompt + completion == total; completion == max_tokens.
  const auto& usage = j.at("usage");
  CHECK(usage.at("completion_tokens") == 5);
  CHECK(usage.at("prompt_tokens").get<int>() > 0);
  CHECK(usage.at("total_tokens").get<int>() ==
        usage.at("prompt_tokens").get<int>() +
            usage.at("completion_tokens").get<int>());
}

TEST_CASE("conformance: /v1/completions stream — SSE frames concat to the non-stream text, [DONE] terminal") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  // Oracle: the deterministic greedy text via a non-stream request (independent,
  // greedy → identical output).
  auto oracle = client.Post("/v1/completions", CompletionBody("hello", 5, false),
                            "application/json");
  REQUIRE(oracle);
  const std::string full_text =
      json::parse(oracle->body).at("choices").at(0).at("text").get<std::string>();

  auto res = client.Post("/v1/completions", CompletionBody("hello", 5, true),
                         "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  CHECK(res->get_header_value("Content-Type").rfind("text/event-stream", 0) == 0);

  std::vector<std::string> frames = ParseSseFrames(res->body);
  REQUIRE(frames.size() >= 2);
  CHECK(frames.back() == "data: [DONE]");

  std::string concatenated;
  std::optional<std::string> last_finish;
  for (size_t i = 0; i + 1 < frames.size(); ++i) {
    json j = SseFramePayload(frames[i]);
    CHECK(j.at("object") == "text_completion");
    CHECK(j.at("id").get<std::string>().rfind("cmpl-", 0) == 0);
    const auto& choice = j.at("choices").at(0);
    concatenated += choice.at("text").get<std::string>();
    if (!choice.at("finish_reason").is_null())
      last_finish = choice.at("finish_reason").get<std::string>();
  }
  CHECK(concatenated == full_text);
  REQUIRE(last_finish.has_value());
  CHECK(*last_finish == "length");  // finish_reason on the last content chunk.
}

TEST_CASE("conformance: /v1/completions honors a stop string (finish_reason=stop, output truncated)") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  // Discover the deterministic greedy text, then pick an in-text stop substring.
  auto probe = client.Post("/v1/completions", CompletionBody("hello", 8, false),
                           "application/json");
  REQUIRE(probe);
  const std::string full_text =
      json::parse(probe->body).at("choices").at(0).at("text").get<std::string>();
  REQUIRE(full_text.size() >= 2);
  const std::string stop = full_text.substr(1, 1);

  json body = {{"prompt", "hello"},
               {"max_tokens", 8},
               {"temperature", 0.0},
               {"stop", json::array({stop})}};
  auto res = client.Post("/v1/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("choices").at(0).at("finish_reason") == "stop");
  // include_stop_str_in_output defaults false → the stop string is truncated off.
  CHECK(j.at("choices").at(0).at("text").get<std::string>().find(stop) ==
        std::string::npos);
}

TEST_CASE("conformance: /v1/completions max_tokens bounds the length stop") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  auto res = client.Post("/v1/completions", CompletionBody("hello", 3, false),
                         "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("choices").at(0).at("finish_reason") == "length");
  CHECK(j.at("usage").at("completion_tokens") == 3);
}

TEST_CASE("conformance: /v1/completions echo is accepted (behavior deferred, must not error)") {
  // Upstream `echo` is parsed; the prompt-prepend BEHAVIOR is deferred (see
  // serving_completion.h). The conformance bar here is that an echo request is
  // ACCEPTED (200 + a well-formed text_completion), not that the prompt is echoed
  // (that is a real-behavior TODO, noted in the header).
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json body = {{"prompt", "hello"},
               {"max_tokens", 4},
               {"temperature", 0.0},
               {"echo", true}};
  auto res = client.Post("/v1/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("object") == "text_completion");
  CHECK(j.at("choices").size() == 1);  // n=1 default → a single choice.
}

// ═══════════════════════════ 2. Chat completions ════════════════════════════

TEST_CASE("conformance: /v1/chat/completions non-stream is a contract-correct chat.completion") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
               {"max_completion_tokens", 4},
               {"temperature", 0.0}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("id").get<std::string>().rfind("chatcmpl-", 0) == 0);
  const auto& choice = j.at("choices").at(0);
  CHECK(choice.at("message").at("role") == "assistant");
  REQUIRE(choice.at("message").at("content").is_string());
  CHECK_FALSE(choice.at("message").at("content").get<std::string>().empty());
  CHECK(choice.at("finish_reason") == "length");
  const auto& usage = j.at("usage");
  CHECK(usage.at("total_tokens").get<int>() ==
        usage.at("prompt_tokens").get<int>() +
            usage.at("completion_tokens").get<int>());
}

TEST_CASE("conformance: /v1/chat/completions stream — role delta, content deltas, finish, [DONE]") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json body = {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
               {"max_completion_tokens", 5},
               {"temperature", 0.0},
               {"stream", true}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  CHECK(res->get_header_value("Content-Type").rfind("text/event-stream", 0) == 0);

  std::vector<std::string> frames = ParseSseFrames(res->body);
  REQUIRE(frames.size() >= 3);
  CHECK(frames.back() == "data: [DONE]");

  // First chunk: the role delta.
  json first = SseFramePayload(frames.front());
  CHECK(first.at("object") == "chat.completion.chunk");
  CHECK(first.at("choices").at(0).at("delta").at("role") == "assistant");
  CHECK(first.at("choices").at(0).at("finish_reason").is_null());

  // Middle chunks: content deltas (no role); the last content chunk: finish_reason.
  std::string streamed;
  std::optional<std::string> last_finish;
  for (size_t i = 1; i + 1 < frames.size(); ++i) {
    json j = SseFramePayload(frames[i]);
    CHECK(j.at("object") == "chat.completion.chunk");
    const auto& choice = j.at("choices").at(0);
    const auto& delta = choice.at("delta");
    CHECK_FALSE(delta.contains("role"));  // role only on the first chunk.
    if (delta.contains("content"))
      streamed += delta.at("content").get<std::string>();
    if (!choice.at("finish_reason").is_null())
      last_finish = choice.at("finish_reason").get<std::string>();
  }
  CHECK_FALSE(streamed.empty());
  REQUIRE(last_finish.has_value());
  CHECK(*last_finish == "length");
}

TEST_CASE("conformance: /v1/chat/completions multi-turn conversation reflects every message in the prompt") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  // A [system, user, assistant, user] conversation (in-vocab contents so the
  // fixture tokenizer accepts the rendered prompt).
  json body = {
      {"messages",
       json::array({{{"role", "system"}, {"content", "hello"}},
                    {{"role", "user"}, {"content", "world"}},
                    {{"role", "assistant"}, {"content", "hello"}},
                    {{"role", "user"}, {"content", "world"}}})},
      {"max_completion_tokens", 4},
      {"temperature", 0.0}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");

  // The chat template / prompt seam received ALL FOUR messages: the rendered
  // prompt is the concatenation of every message's content.
  CHECK(h.CapturedChatPrompt() == "helloworldhelloworld");
}

// ═══════════════════════════ 3. Tool calling ════════════════════════════════

TEST_CASE("conformance: tool_choice=auto is RELAXED — a plain reply is NOT forced into a tool call") {
  // The synthetic model emits plain content (its tiny vocab cannot spell a
  // `<tool_call>` wrapper). With tool_choice=auto (RELAXED) the response must be a
  // NORMAL assistant message — no tool_calls, finish_reason is the engine's stop
  // (NOT "tool_calls"). This is the whole point of the relaxed/lazy semantics.
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json body = {
      {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
      {"max_completion_tokens", 5},
      {"temperature", 0.0},
      {"tools",
       json::array({{{"type", "function"},
                     {"function",
                      {{"name", "get_weather"},
                       {"description", "Get the weather for a city."},
                       {"parameters",
                        {{"type", "object"},
                         {"properties", {{"city", {{"type", "string"}}}}},
                         {"required", json::array({"city"})}}}}}}})},
      {"tool_choice", "auto"}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  const auto& msg = j.at("choices").at(0).at("message");
  CHECK(msg.at("role") == "assistant");
  // NOT forced: a normal content message, no tool_calls, engine finish.
  CHECK_FALSE(msg.contains("tool_calls"));
  CHECK(j.at("choices").at(0).at("finish_reason") != "tool_calls");
  CHECK(j.at("choices").at(0).at("finish_reason") == "length");
  // NOTE (real-weights-pending, dgx): the "auto EMITS a <tool_call> →
  // tool_calls populated + finish_reason=tool_calls" case and the streaming
  // tool-call delta cadence require a model whose vocab can spell the wrapper.
  // Those SHAPES are asserted at the serving layer in test_serving.cpp.
}

TEST_CASE("conformance: tool_choice=required is accepted and returns a contract-correct message") {
  // The forced-tool constraint flows onto SamplingParams (structural_tag) and is
  // enforced end to end over the NATIVE backend in test_response_format_e2e.cpp.
  // Over the synthetic HTTP server we assert the request is ACCEPTED + the
  // response SHAPE is contract-correct (the constrained tool OUTPUT needs a real
  // model — noted).
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json tool = {{"type", "function"},
               {"function",
                {{"name", "get_weather"},
                 {"parameters",
                  {{"type", "object"},
                   {"properties", {{"city", {{"type", "string"}}}}},
                   {"required", json::array({"city"})}}}}}};
  json body = {
      {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
      {"max_completion_tokens", 4},
      {"temperature", 0.0},
      {"tools", json::array({tool})},
      {"tool_choice", "required"}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
}

TEST_CASE("conformance: tool_choice=named is accepted and returns a contract-correct message") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json tool = {{"type", "function"},
               {"function",
                {{"name", "get_weather"},
                 {"parameters",
                  {{"type", "object"},
                   {"properties", {{"city", {{"type", "string"}}}}},
                   {"required", json::array({"city"})}}}}}};
  json body = {
      {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
      {"max_completion_tokens", 4},
      {"temperature", 0.0},
      {"tools", json::array({tool})},
      {"tool_choice",
       {{"type", "function"}, {"function", {{"name", "get_weather"}}}}}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
}

// ═══════════════════════ 4. Grammars / structured output ════════════════════

TEST_CASE("conformance: response_format=json_object is accepted + contract-correct") {
  // The constraint flows onto SamplingParams.structured_outputs (proven at the
  // serving layer + end to end over the native backend in
  // test_response_format_e2e.cpp). Over the synthetic HTTP server we assert the
  // request is ACCEPTED + the response shape (the constrained JSON OUTPUT needs a
  // model whose vocab can spell JSON — noted).
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json body = {
      {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
      {"max_completion_tokens", 4},
      {"temperature", 0.0},
      {"response_format", {{"type", "json_object"}}}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
}

TEST_CASE("conformance: response_format=json_schema is accepted + contract-correct") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json body = {
      {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
      {"max_completion_tokens", 4},
      {"temperature", 0.0},
      {"response_format",
       {{"type", "json_schema"},
        {"json_schema",
         {{"name", "obj"},
          {"schema",
           {{"type", "object"},
            {"properties", {{"a", {{"type", "integer"}}}}},
            {"required", json::array({"a"})}}}}}}}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("object") == "chat.completion");
}

TEST_CASE("conformance: a malformed response_format (json_schema w/o the schema) → 400 error shape") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  // json_schema type MUST carry the json_schema field (validate_response_format).
  json body = {
      {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
      {"max_completion_tokens", 4},
      {"response_format", {{"type", "json_schema"}}}};
  auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 400);
  json j = json::parse(res->body);
  CHECK(j.at("error").at("type") == "BadRequestError");
  CHECK(j.at("error").at("code") == 400);
  CHECK_FALSE(j.at("error").at("message").get<std::string>().empty());
}

// ═══════════════════════════ 5. Errors ══════════════════════════════════════

TEST_CASE("conformance: a malformed JSON body → 400 + OpenAI ErrorResponse shape") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  auto res = client.Post("/v1/completions", "{not valid json",
                         "application/json");
  REQUIRE(res);
  CHECK(res->status == 400);
  json j = json::parse(res->body);
  // Clients parse error.message / error.type / error.code.
  REQUIRE(j.contains("error"));
  CHECK(j.at("error").at("type") == "BadRequestError");
  CHECK(j.at("error").at("code") == 400);
  CHECK_FALSE(j.at("error").at("message").get<std::string>().empty());
}

TEST_CASE("conformance: an unknown model → 404 + ErrorResponse shape") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  json body = {{"model", "does-not-exist"},
               {"prompt", "hello"},
               {"max_tokens", 3},
               {"temperature", 0.0}};
  auto res = client.Post("/v1/completions", body.dump(), "application/json");
  REQUIRE(res);
  CHECK(res->status == 404);
  json j = json::parse(res->body);
  CHECK(j.at("error").at("type") == "NotFoundError");
  CHECK(j.at("error").at("code") == 404);
}

TEST_CASE("conformance: an unknown route → 404") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  auto res = client.Get("/v1/nonexistent");
  REQUIRE(res);
  CHECK(res->status == 404);
}

TEST_CASE("conformance: a valid request → 200 (the positive control)") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  auto res = client.Post("/v1/completions", CompletionBody("hello", 2, false),
                         "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);
}

// ═══════════════════════════ 6. Endpoints ═══════════════════════════════════

TEST_CASE("conformance: /v1/models is an OpenAI model list") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  auto res = client.Get("/v1/models");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("object") == "list");
  REQUIRE(j.at("data").size() >= 1);
  CHECK(j.at("data").at(0).at("id") == "test-model");
  CHECK(j.at("data").at(0).at("object") == "model");
}

TEST_CASE("conformance: /health returns 200") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  auto res = client.Get("/health");
  REQUIRE(res);
  CHECK(res->status == 200);
}

TEST_CASE("conformance: /version reports the served version") {
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  auto res = client.Get("/version");
  REQUIRE(res);
  CHECK(res->status == 200);
  json j = json::parse(res->body);
  CHECK(j.at("version") == "9.9.9");
}

// ═══════════════════════ 7. UTF-8 / robustness ══════════════════════════════

TEST_CASE("conformance: a completion over an emoji-capable vocab never 500s + serializes valid UTF-8") {
  // The emoji fixture (ids 0..23) can emit the two half-emoji BYTE tokens, so the
  // synthetic argmax may produce a SPLIT multibyte run. The serving boundary's
  // SanitizeUtf8 must keep nlohmann::json::dump() from throwing (→ HTTP 500): the
  // server MUST return a 200 whose body is parseable, valid-UTF-8 JSON regardless
  // of which raw bytes the model emitted. (Whether a split actually occurs is
  // model-dependent; the invariant tested is no-500 + valid UTF-8 out.)
  const HfConfig c = MakeConfig(24);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, EmojiFixture());
  LiveServer ls(h.server);
  httplib::Client client = ls.Client();

  // A longer generation gives the model room to reach a byte token.
  auto res = client.Post("/v1/completions", CompletionBody("hello", 20, false),
                         "application/json");
  REQUIRE(res);
  CHECK(res->status == 200);  // NOT a 500 from a dump() throw.
  // json::parse succeeds only on valid UTF-8 → the text was sanitized.
  json j = json::parse(res->body);
  CHECK(j.at("object") == "text_completion");
  CHECK(j.at("choices").at(0).at("text").is_string());

  // The streaming path sanitizes each delta too — it must also stay valid.
  auto stream = client.Post("/v1/completions", CompletionBody("hello", 20, true),
                            "application/json");
  REQUIRE(stream);
  CHECK(stream->status == 200);
  for (const std::string& frame : ParseSseFrames(stream->body)) {
    if (frame == "data: [DONE]") continue;
    CHECK_NOTHROW(SseFramePayload(frame));  // valid UTF-8 JSON per frame.
  }
}

TEST_CASE("conformance: concurrent clients are serialized — every request succeeds identically") {
  // httplib services requests on a worker-thread pool; the api_server serializes
  // engine-touching requests with a mutex. Fire N concurrent completions and
  // assert every one is a well-formed 200 with the SAME deterministic greedy text
  // (no crash, no corrupted/empty body, no cross-request state bleed).
  const HfConfig c = MakeConfig(22);
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());
  LiveServer ls(h.server);

  constexpr int kClients = 6;
  std::vector<std::thread> clients;
  std::vector<int> statuses(kClients, -1);
  std::vector<std::string> texts(kClients);
  const int port = ls.port;
  for (int i = 0; i < kClients; ++i) {
    clients.emplace_back([&, i]() {
      httplib::Client client("127.0.0.1", port);
      client.set_read_timeout(30, 0);
      auto res = client.Post("/v1/completions", CompletionBody("hello", 4, false),
                             "application/json");
      if (res) {
        statuses[static_cast<size_t>(i)] = res->status;
        try {
          json j = json::parse(res->body);
          texts[static_cast<size_t>(i)] =
              j.at("choices").at(0).at("text").get<std::string>();
        } catch (...) {
          statuses[static_cast<size_t>(i)] = -2;  // malformed body.
        }
      }
    });
  }
  for (auto& t : clients) t.join();

  for (int i = 0; i < kClients; ++i) {
    CHECK(statuses[static_cast<size_t>(i)] == 200);
    CHECK_FALSE(texts[static_cast<size_t>(i)].empty());
  }
  for (int i = 1; i < kClients; ++i)
    CHECK(texts[static_cast<size_t>(i)] == texts[0]);
}
