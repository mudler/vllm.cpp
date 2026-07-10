// Tests for the OpenAI HTTP api_server (M3.1 Task 4). Two layers:
//   1. HANDLER-DISPATCH (no socket, the primary coverage): drive the ApiServer
//      handle_* methods with request bodies over a small synthetic AsyncLLM and
//      assert the OpenAI response shape, SSE framing, error status codes and
//      that a completion serializes without a 500 from invalid UTF-8.
//   2. SOCKET SMOKE (real HTTP over an ephemeral port via cpp-httplib's client):
//      start the server on a background thread, issue real requests to /health,
//      /v1/models, a non-streaming + streaming /v1/completions and a
//      /v1/chat/completions, asserting the framing end to end.
//
// The synthetic model mirrors tests/vllm/entrypoints/openai/test_serving.cpp
// (tiny hybrid-MoE Qwen3.6 + the BPE fixture, vocab ids 0..21).
#include "vllm/entrypoints/openai/api_server.h"

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
#include "vllm/v1/engine/async_llm.h"
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
using vllm::entrypoints::openai::ChatMessage;
using vllm::entrypoints::openai::OpenAIServingChat;
using vllm::entrypoints::openai::OpenAIServingCompletion;
using vllm::entrypoints::openai::OpenAIServingModels;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCore;
using vllm::v1::AsyncLLM;
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

// ─── Synthetic weights (mirrors test_serving.cpp) ────────────────────────────
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

constexpr int kVocab = 22;
constexpr int kBlockSize = 32;
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 32;

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = kVocab;
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
  c.raw = json::object();
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
          std::vector<std::vector<int64_t>>{{Hv, Dv, Dk}, {conv_dim, Kw - 1}},
          std::vector<DType>{DType::kF32, DType::kF32}));
  return kv;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_apisrv_tok_" + std::to_string(counter++) + ".json"))
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
  static const Tokenizer tok = BuildFixture();
  return tok;
}

// In-vocab chat prompt seam (the fixture vocab is ids 0..21).
std::string InVocabChatPrompt(
    const std::vector<ChatMessage>& messages, bool,
    const std::vector<vllm::entrypoints::openai::ChatCompletionToolsParam>&) {
  std::string p;
  for (const ChatMessage& m : messages)
    if (m.content.has_value()) p += *m.content;
  return p;
}

// A fully-wired serving stack + ApiServer over the synthetic engine.
struct ServerHarness {
  ServerHarness(const HfConfig& c, const Qwen3_5MoeWeights& w,
                const Tokenizer& tok)
      : scheduler(MakeSchedulerConfig(), MakeKvConfig(c), kBlockSize,
                  /*enable_caching=*/true),
        runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, kMaxModelLen * 8),
        executor(runner),
        input_processor(tok, c),
        output_processor(&tok),
        async_engine(input_processor, scheduler, executor, output_processor,
                     Hasher()),
        models("test-model"),
        completion(async_engine, "test-model"),
        chat(async_engine, "test-model", InVocabChatPrompt),
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

  Scheduler scheduler;
  GPUModelRunner runner;
  Executor executor;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  AsyncLLM async_engine;
  OpenAIServingModels models;
  OpenAIServingCompletion completion;
  OpenAIServingChat chat;
  ApiServer server;
};

}  // namespace

// ─── 1. Handler-dispatch (no socket) ─────────────────────────────────────────
TEST_CASE("api_server: non-stream completion dispatch → 200 + OpenAI shape") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"model":"test-model","prompt":"hello","max_tokens":5,"temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_completions(body);

  CHECK(r.status == 200);
  CHECK_FALSE(r.streaming);
  CHECK(r.content_type == "application/json");
  json j = json::parse(r.body);
  CHECK(j.at("object") == "text_completion");
  CHECK(j.at("model") == "test-model");
  CHECK(j.at("id").get<std::string>().rfind("cmpl-", 0) == 0);
  CHECK(j.at("choices").at(0).at("finish_reason") == "length");
  CHECK(j.at("usage").at("completion_tokens") == 5);
}

TEST_CASE("api_server: streaming completion dispatch → SSE chunks ending [DONE]") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"prompt":"hello","max_tokens":5,"temperature":0.0,"stream":true})";
  ApiServer::DispatchResult r = h.server.handle_completions(body);

  CHECK(r.status == 200);
  CHECK(r.streaming);
  CHECK(r.content_type == "text/event-stream");
  REQUIRE(r.sse_stream != nullptr);
  std::vector<std::string> chunks;
  std::string chunk;
  while (r.sse_stream->next(chunk)) chunks.push_back(chunk);
  REQUIRE(chunks.size() >= 2);
  CHECK(chunks.back() == "data: [DONE]\n\n");
  // Each non-terminal chunk is a valid `data: {json}\n\n` frame.
  for (size_t i = 0; i + 1 < chunks.size(); ++i) {
    REQUIRE(chunks[i].rfind("data: ", 0) == 0);
    REQUIRE(chunks[i].substr(chunks[i].size() - 2) == "\n\n");
    json j = json::parse(chunks[i].substr(6, chunks[i].size() - 8));
    CHECK(j.at("object") == "text_completion");
  }
}

// Ported from tests/entrypoints/openai/completion/test_completion.py:259
// (test_completion_streaming), with W2's load-bearing arrival assertion: the
// first content frame is observed before the terminal frame instead of being
// replayed from a precomputed vector after generation completes.
TEST_CASE("api_server: live SSE first chunk arrives before generation completes") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"prompt":"hello","max_tokens":20,"temperature":0.0,"stream":true})";
  const auto started = std::chrono::steady_clock::now();
  ApiServer::DispatchResult result = h.server.handle_completions(body);
  const auto dispatched = std::chrono::steady_clock::now();
  REQUIRE(result.sse_stream != nullptr);
  CHECK(result.sse_chunks.empty());  // no precomputed fake-SSE fallback

  std::string first;
  REQUIRE(result.sse_stream->next(first));
  const auto first_arrival = std::chrono::steady_clock::now();
  REQUIRE(first.rfind("data: ", 0) == 0);
  const json first_payload =
      json::parse(first.substr(6, first.size() - 8));
  CHECK(first_payload.at("choices").at(0).at("finish_reason").is_null());

  size_t chunk_count = 1;
  std::string chunk;
  while (result.sse_stream->next(chunk)) ++chunk_count;
  const auto completed = std::chrono::steady_clock::now();
  CHECK(chunk_count > 2);  // content frames plus [DONE]
  CHECK(dispatched < first_arrival);
  CHECK(first_arrival < completed);
  CHECK(started < completed);
}

TEST_CASE("api_server: chat dispatch → assistant message") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":4,"temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_chat_completions(body);

  CHECK(r.status == 200);
  json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
}

TEST_CASE("api_server: live chat SSE emits role, content, finish, and DONE") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":8,"temperature":0.0,"stream":true})";
  ApiServer::DispatchResult result = h.server.handle_chat_completions(body);
  REQUIRE(result.sse_stream != nullptr);

  std::string chunk;
  REQUIRE(result.sse_stream->next(chunk));
  json role = json::parse(chunk.substr(6, chunk.size() - 8));
  CHECK(role.at("choices").at(0).at("delta").at("role") == "assistant");
  CHECK(role.at("choices").at(0).at("finish_reason").is_null());

  size_t content_frames = 0;
  bool finish_seen = false;
  bool done_seen = false;
  while (result.sse_stream->next(chunk)) {
    if (chunk == "data: [DONE]\n\n") {
      done_seen = true;
      continue;
    }
    json frame = json::parse(chunk.substr(6, chunk.size() - 8));
    const json& choice = frame.at("choices").at(0);
    if (!choice.at("finish_reason").is_null()) finish_seen = true;
    if (choice.at("delta").contains("content")) ++content_frames;
  }
  CHECK(content_frames > 0);
  CHECK(finish_seen);
  CHECK(done_seen);
}

TEST_CASE("api_server: disconnect aborts a live SSE request") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"prompt":"hello","max_tokens":30,"temperature":0.0,"stream":true})";
  ApiServer::DispatchResult result = h.server.handle_completions(body);
  REQUIRE(result.sse_stream != nullptr);
  std::string first;
  REQUIRE(result.sse_stream->next(first));
  result.sse_stream->abort();
  for (int i = 0; i < 500 && h.async_engine.has_unfinished_requests(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CHECK_FALSE(h.async_engine.has_unfinished_requests());
}

TEST_CASE("api_server: malformed JSON → 400 error shape") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  ApiServer::DispatchResult r = h.server.handle_completions("{not json");
  CHECK(r.status == 400);
  json j = json::parse(r.body);
  CHECK(j.at("error").at("type") == "BadRequestError");
  CHECK(j.at("error").at("code") == 400);
}

TEST_CASE("api_server: unknown model → 404 error shape") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const std::string body =
      R"({"model":"nope","prompt":"hello","max_tokens":3,"temperature":0.0})";
  ApiServer::DispatchResult r = h.server.handle_completions(body);
  CHECK(r.status == 404);
  json j = json::parse(r.body);
  CHECK(j.at("error").at("type") == "NotFoundError");
}

TEST_CASE("api_server: /v1/models, /health, /version dispatch") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  json models = json::parse(h.server.handle_models().body);
  CHECK(models.at("object") == "list");
  CHECK(models.at("data").at(0).at("id") == "test-model");
  CHECK(models.at("data").at(0).at("object") == "model");

  ApiServer::DispatchResult health = h.server.handle_health();
  CHECK(health.status == 200);

  json ver = json::parse(h.server.handle_version().body);
  CHECK(ver.at("version") == "9.9.9");
}

// ─── 2. Socket smoke test (real HTTP over an ephemeral port) ─────────────────
TEST_CASE("api_server: socket smoke — real HTTP requests over an ephemeral port") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  std::thread server_thread([&h]() { h.server.serve(); });
  // Wait until the accept loop is up.
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  httplib::Client client("127.0.0.1", port);
  client.set_connection_timeout(5, 0);
  client.set_read_timeout(15, 0);

  // /health
  {
    auto res = client.Get("/health");
    REQUIRE(res);
    CHECK(res->status == 200);
  }
  // /v1/models
  {
    auto res = client.Get("/v1/models");
    REQUIRE(res);
    CHECK(res->status == 200);
    json j = json::parse(res->body);
    CHECK(j.at("data").at(0).at("id") == "test-model");
  }
  // Non-streaming /v1/completions
  {
    auto res = client.Post(
        "/v1/completions", R"({"prompt":"hello","max_tokens":5,"temperature":0.0})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    json j = json::parse(res->body);
    CHECK(j.at("object") == "text_completion");
    CHECK_FALSE(j.at("choices").at(0).at("text").get<std::string>().empty());
  }
  // Streaming /v1/completions — the body is the concatenated SSE frames.
  {
    auto res = client.Post(
        "/v1/completions",
        R"({"prompt":"hello","max_tokens":5,"temperature":0.0,"stream":true})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.rfind("data: ", 0) == 0);
    CHECK(res->body.find("data: [DONE]\n\n") != std::string::npos);
  }
  // /v1/chat/completions
  {
    auto res = client.Post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":4,"temperature":0.0})",
        "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    json j = json::parse(res->body);
    CHECK(j.at("object") == "chat.completion");
    CHECK(j.at("choices").at(0).at("message").at("role") == "assistant");
  }

  h.server.stop();
  server_thread.join();
}

// W2 port of test_async_llm.test_load at the HTTP boundary: concurrent workers
// submit into one AsyncLLM queue and complete as one scheduler batch; there is
// no server-wide engine mutex.
TEST_CASE("api_server: concurrent requests share AsyncLLM without state races") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  ServerHarness h(c, w, Fixture());

  const int port = h.server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  std::thread server_thread([&h]() { h.server.serve(); });
  for (int i = 0; i < 500 && !h.server.is_running(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  REQUIRE(h.server.is_running());

  constexpr int kClients = 6;
  std::vector<std::thread> clients;
  std::vector<int> statuses(kClients, -1);
  std::vector<std::string> texts(kClients);
  for (int i = 0; i < kClients; ++i) {
    clients.emplace_back([&, i]() {
      httplib::Client client("127.0.0.1", port);
      client.set_read_timeout(30, 0);
      auto res = client.Post(
          "/v1/completions",
          R"({"prompt":"hello","max_tokens":4,"temperature":0.0})",
          "application/json");
      if (res) {
        statuses[static_cast<size_t>(i)] = res->status;
        try {
          json j = json::parse(res->body);
          texts[static_cast<size_t>(i)] =
              j.at("choices").at(0).at("text").get<std::string>();
        } catch (...) {
          statuses[static_cast<size_t>(i)] = -2;  // malformed body
        }
      }
    });
  }
  for (auto& t : clients) t.join();

  for (int i = 0; i < kClients; ++i) {
    CHECK(statuses[static_cast<size_t>(i)] == 200);
    CHECK_FALSE(texts[static_cast<size_t>(i)].empty());
  }
  // All greedy on the same prompt → identical deterministic output, which also
  // confirms no cross-request state bleed.
  for (int i = 1; i < kClients; ++i)
    CHECK(texts[static_cast<size_t>(i)] == texts[0]);

  h.server.stop();
  server_thread.join();
}
