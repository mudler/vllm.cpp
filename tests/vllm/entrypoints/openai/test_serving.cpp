// Unit tests for the OpenAI serving handlers (M3.1 Task 2) — the decoupled
// serving logic driven over a small SYNTHETIC LLMEngine, NO socket. Mirrors the
// M1.8 test harness (tests/vllm/v1/test_llm_engine.cpp): a tiny hybrid-MoE
// Qwen3.6 model on CPU with the tiny BPE fixture (vocab ids 0..23) so every
// greedy argmax is decodable. We assert the OpenAI response / SSE-chunk shapes,
// the streaming cadence, the finish_reason surfacing and the usage counts.
//
// Ported serving logic: vllm/entrypoints/openai/{completion,chat_completion}/
// serving.py @ e24d1b24.
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
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
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/engine/llm_engine.h"
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
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::entrypoints::openai::ChatCompletionRequest;
using vllm::entrypoints::openai::ChatCompletionResult;
using vllm::entrypoints::openai::ChatMessage;
using vllm::entrypoints::openai::CompletionRequest;
using vllm::entrypoints::openai::CompletionResult;
using vllm::entrypoints::openai::DefaultChatPromptFallback;
using vllm::entrypoints::openai::OpenAIServingChat;
using vllm::entrypoints::openai::OpenAIServingCompletion;
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

// ─── Synthetic weights (mirrors tests/vllm/v1/test_llm_engine.cpp) ───────────
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

// Vocab ids 0..21 — the tiny BPE fixture MINUS the two half-emoji byte tokens
// (ids 22/23). Every id here decodes to valid UTF-8, so a DELTA text fragment is
// never a split multibyte sequence that json::dump() would reject. (A real
// incremental detokenizer buffers incomplete UTF-8; the synthetic argmax over
// random weights would otherwise split the 4-byte emoji across deltas.)
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
          std::vector<std::vector<int64_t>>{{Hv, Dv, Dk}, {conv_dim, Kw - 1}},
          std::vector<DType>{DType::kF32, DType::kF32}));
  return kv;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_serving_tok_" + std::to_string(counter++) + ".json"))
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

// Owns a fully-wired LLMEngine stack (mirrors test_llm_engine.cpp::Harness).
struct Harness {
  Harness(const HfConfig& c, const Qwen3_5MoeWeights& w, const Tokenizer& tok,
          int max_num_reqs = 8)
      : scheduler(MakeSchedulerConfig(), MakeKvConfig(c), kBlockSize,
                  /*enable_caching=*/true),
        runner(c, w, MakeKvConfig(c), Q(), max_num_reqs, kMaxModelLen,
               /*max_num_batched_tokens=*/kMaxModelLen * max_num_reqs),
        executor(runner),
        engine_core(scheduler, executor),
        input_processor(tok, c),
        output_processor(&tok),
        engine(input_processor, engine_core, output_processor, Hasher()) {}

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
  EngineCore engine_core;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  LLMEngine engine;
};

// Strip the SSE `data: ` frame + trailing `\n\n` off a chunk line.
std::string SsePayload(const std::string& chunk) {
  REQUIRE(chunk.rfind("data: ", 0) == 0);
  REQUIRE(chunk.size() >= 8);
  REQUIRE(chunk.substr(chunk.size() - 2) == "\n\n");
  return chunk.substr(6, chunk.size() - 6 - 2);
}

CompletionRequest MakeCompletionRequest(const std::string& prompt, int max_tokens,
                                        bool stream) {
  CompletionRequest r;
  r.prompt = prompt;
  r.max_tokens = max_tokens;
  r.temperature = 0.0;  // greedy -> deterministic
  r.stream = stream;
  return r;
}

}  // namespace

// ─── (a) Non-streaming completion → text, finish_reason, usage ───────────────
TEST_CASE("serving_completion: non-stream response carries text, length finish, usage") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 6;

  Harness h(c, w, tok);
  OpenAIServingCompletion serving(h.engine, "test-model");

  CompletionResult res = serving.create_completion(
      MakeCompletionRequest("hello", kN, /*stream=*/false));

  REQUIRE_FALSE(res.streaming);
  REQUIRE(res.response.has_value());
  const auto& resp = *res.response;
  CHECK(resp.object == "text_completion");
  CHECK(resp.model == "test-model");
  CHECK(resp.id.rfind("cmpl-", 0) == 0);
  REQUIRE(resp.choices.size() == 1);
  CHECK(resp.choices[0].index == 0);
  CHECK_FALSE(resp.choices[0].text.empty());
  REQUIRE(resp.choices[0].finish_reason.has_value());
  CHECK(*resp.choices[0].finish_reason == "length");  // no eos -> max_tokens

  // (e) usage: prompt "hello" is a single in-vocab token; kN generated.
  CHECK(resp.usage.completion_tokens == kN);
  CHECK(resp.usage.prompt_tokens > 0);
  CHECK(resp.usage.total_tokens ==
        resp.usage.prompt_tokens + resp.usage.completion_tokens);
}

// ─── (b) Streaming completion → deltas concat to full text, [DONE], finish ───
TEST_CASE("serving_completion: streamed deltas concatenate to the non-stream text") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 6;

  // Oracle full text via a fresh non-streaming run.
  std::string full_text;
  {
    Harness h(c, w, tok);
    OpenAIServingCompletion serving(h.engine, "test-model");
    CompletionResult r = serving.create_completion(
        MakeCompletionRequest("hello", kN, /*stream=*/false));
    full_text = r.response->choices[0].text;
  }

  Harness h(c, w, tok);  // fresh KV -> identical greedy stream
  OpenAIServingCompletion serving(h.engine, "test-model");
  CompletionResult res = serving.create_completion(
      MakeCompletionRequest("hello", kN, /*stream=*/true));

  REQUIRE(res.streaming);
  REQUIRE(res.sse_chunks.size() >= 2);
  // Last line is the terminator.
  CHECK(res.sse_chunks.back() == "data: [DONE]\n\n");

  std::string concatenated;
  std::optional<std::string> last_finish;
  for (size_t i = 0; i + 1 < res.sse_chunks.size(); ++i) {
    json j = json::parse(SsePayload(res.sse_chunks[i]));
    CHECK(j.at("object") == "text_completion");
    CHECK(j.at("id").get<std::string>().rfind("cmpl-", 0) == 0);
    const auto& choice = j.at("choices").at(0);
    concatenated += choice.at("text").get<std::string>();
    if (!choice.at("finish_reason").is_null()) {
      last_finish = choice.at("finish_reason").get<std::string>();
    }
  }
  CHECK(concatenated == full_text);
  // (f) finish_reason surfaces on the last content chunk.
  REQUIRE(last_finish.has_value());
  CHECK(*last_finish == "length");
}

// ─── (f) Completion stop-string finish_reason surfaces ───────────────────────
TEST_CASE("serving_completion: a stop string yields finish_reason 'stop'") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();

  // Discover the deterministic greedy text to pick an in-text stop substring.
  std::string full_text;
  {
    Harness h(c, w, tok);
    OpenAIServingCompletion serving(h.engine, "test-model");
    full_text = serving
                    .create_completion(
                        MakeCompletionRequest("hello", 8, /*stream=*/false))
                    .response->choices[0]
                    .text;
  }
  REQUIRE(full_text.size() >= 2);
  const std::string stop = full_text.substr(1, 1);

  Harness h(c, w, tok);
  OpenAIServingCompletion serving(h.engine, "test-model");
  CompletionRequest req = MakeCompletionRequest("hello", 8, /*stream=*/false);
  req.stop = {stop};
  CompletionResult res = serving.create_completion(req);

  REQUIRE(res.response.has_value());
  REQUIRE(res.response->choices[0].finish_reason.has_value());
  CHECK(*res.response->choices[0].finish_reason == "stop");
  // Output truncated before the stop string (include_stop_str_in_output=false).
  CHECK(res.response->choices[0].text.find(stop) == std::string::npos);
}

// ─── DefaultChatPromptFallback (the Task-3 seam) ─────────────────────────────
TEST_CASE("serving_chat: default fallback joins role: content + generation prompt") {
  std::vector<ChatMessage> msgs;
  msgs.push_back(ChatMessage{"system", std::string("be nice")});
  msgs.push_back(ChatMessage{"user", std::string("hello")});
  const std::string prompt =
      DefaultChatPromptFallback(msgs, /*add_generation_prompt=*/true);
  CHECK(prompt == "system: be nice\nuser: hello\nassistant:");
  const std::string no_gen =
      DefaultChatPromptFallback(msgs, /*add_generation_prompt=*/false);
  CHECK(no_gen == "system: be nice\nuser: hello\n");
}

namespace {
// In-vocab prompt seam for the engine-driven chat tests (the fixture vocab is
// ids 0..23; a "role: content" join contains out-of-vocab bytes). Task 3's real
// template renderer replaces this — here it stands in as the injected seam.
std::string InVocabChatPrompt(const std::vector<ChatMessage>& messages, bool) {
  std::string p;
  for (const ChatMessage& m : messages) {
    if (m.content.has_value()) p += *m.content;
  }
  return p;  // e.g. "hello"
}

ChatCompletionRequest MakeChatRequest(std::vector<ChatMessage> messages,
                                      int max_tokens, bool stream) {
  ChatCompletionRequest r;
  r.messages = std::move(messages);
  r.max_completion_tokens = max_tokens;
  r.temperature = 0.0;
  r.stream = stream;
  return r;
}
}  // namespace

// ─── (c) Non-streaming chat → message{role:assistant, content} ───────────────
TEST_CASE("serving_chat: non-stream response carries assistant message + usage") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 5;

  Harness h(c, w, tok);
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);

  ChatCompletionResult res = serving.create_chat_completion(MakeChatRequest(
      {ChatMessage{"user", std::string("hello")}}, kN, /*stream=*/false));

  REQUIRE_FALSE(res.streaming);
  REQUIRE(res.response.has_value());
  const auto& resp = *res.response;
  CHECK(resp.object == "chat.completion");
  CHECK(resp.id.rfind("chatcmpl-", 0) == 0);
  REQUIRE(resp.choices.size() == 1);
  CHECK(resp.choices[0].message.role == "assistant");
  REQUIRE(resp.choices[0].message.content.has_value());
  CHECK_FALSE(resp.choices[0].message.content->empty());
  REQUIRE(resp.choices[0].finish_reason.has_value());
  CHECK(*resp.choices[0].finish_reason == "length");
  CHECK(resp.usage.completion_tokens == kN);
  CHECK(resp.usage.total_tokens ==
        resp.usage.prompt_tokens + resp.usage.completion_tokens);
}

// ─── (d) Streaming chat → role delta, content deltas, finish, [DONE] ─────────
TEST_CASE("serving_chat: stream cadence is role delta, content deltas, finish, DONE") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 5;

  // Oracle content via a non-streaming run over a fresh stack.
  std::string full_content;
  {
    Harness h(c, w, tok);
    OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
    full_content = *serving
                        .create_chat_completion(MakeChatRequest(
                            {ChatMessage{"user", std::string("hello")}}, kN,
                            /*stream=*/false))
                        .response->choices[0]
                        .message.content;
  }

  Harness h(c, w, tok);
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  ChatCompletionResult res = serving.create_chat_completion(MakeChatRequest(
      {ChatMessage{"user", std::string("hello")}}, kN, /*stream=*/true));

  REQUIRE(res.streaming);
  REQUIRE(res.sse_chunks.size() >= 3);
  CHECK(res.sse_chunks.back() == "data: [DONE]\n\n");

  // First chunk: the role delta.
  json first = json::parse(SsePayload(res.sse_chunks.front()));
  CHECK(first.at("object") == "chat.completion.chunk");
  const auto& first_delta = first.at("choices").at(0).at("delta");
  CHECK(first_delta.at("role") == "assistant");
  CHECK(first_delta.at("content") == "");
  CHECK(first.at("choices").at(0).at("finish_reason").is_null());

  // Middle chunks: content deltas (no role). Last content chunk: finish_reason.
  std::string streamed;
  std::optional<std::string> last_finish;
  for (size_t i = 1; i + 1 < res.sse_chunks.size(); ++i) {
    json j = json::parse(SsePayload(res.sse_chunks[i]));
    const auto& choice = j.at("choices").at(0);
    const auto& delta = choice.at("delta");
    CHECK_FALSE(delta.contains("role"));  // role only on the first chunk
    if (delta.contains("content")) {
      streamed += delta.at("content").get<std::string>();
    }
    if (!choice.at("finish_reason").is_null()) {
      last_finish = choice.at("finish_reason").get<std::string>();
    }
  }
  CHECK(streamed == full_content);
  REQUIRE(last_finish.has_value());
  CHECK(*last_finish == "length");
}
