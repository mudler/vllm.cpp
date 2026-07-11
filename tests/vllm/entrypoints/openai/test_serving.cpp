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
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
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
#include "vllm/v1/structured_output/json_schema_to_gbnf.h"
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
using vllm::entrypoints::openai::ChatCompletionResponseChoice;
using vllm::entrypoints::openai::ChatCompletionResponseStreamChoice;
using vllm::entrypoints::openai::ChatCompletionResult;
using vllm::entrypoints::openai::ChatCompletionToolsParam;
using vllm::entrypoints::openai::ChatMessage;
using vllm::entrypoints::openai::CompletionRequest;
using vllm::entrypoints::openai::CompletionResult;
using vllm::entrypoints::openai::DefaultChatPromptFallback;
using vllm::entrypoints::openai::DeltaMessage;
using vllm::entrypoints::openai::get_tool_parser;
using vllm::entrypoints::openai::HermesToolParser;
using vllm::entrypoints::openai::OpenAIServingChat;
using vllm::entrypoints::openai::OpenAIServingCompletion;
using vllm::entrypoints::openai::ShapeChatDelta;
using vllm::entrypoints::openai::ShapeChatMessage;
using vllm::entrypoints::openai::ShapedChatMessage;
using vllm::entrypoints::openai::ApplyToolChoiceStructuredOutput;
using vllm::entrypoints::openai::StreamOptions;
using vllm::entrypoints::openai::ToolChoice;
using vllm::entrypoints::openai::ToolChoiceStructuralTagSpec;
using vllm::entrypoints::openai::ToolParser;
using vllm::entrypoints::openai::ToolsEnabled;
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

// Ported from tests/entrypoints/openai/completion/test_completion.py:
// test_completion_stream_options @ e24d1b24. This exercises the retained
// synchronous serving seam; production AsyncLLM is covered by test_api_server.
TEST_CASE("serving_completion: continuous and final usage use native token IDs") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const int kN = 6;
  Harness h(c, w, Fixture());
  OpenAIServingCompletion serving(h.engine, "test-model");

  CompletionRequest req =
      MakeCompletionRequest("hello", kN, /*stream=*/true);
  req.stream_options = StreamOptions{/*include_usage=*/true,
                                     /*continuous_usage_stats=*/true};
  CompletionResult res = serving.create_completion(req);
  REQUIRE(res.streaming);
  REQUIRE(res.sse_chunks.size() >= 3);
  CHECK(res.sse_chunks.back() == "data: [DONE]\n\n");

  int previous_completion_tokens = 0;
  for (size_t i = 0; i + 1 < res.sse_chunks.size(); ++i) {
    const json frame = json::parse(SsePayload(res.sse_chunks[i]));
    REQUIRE(frame.contains("usage"));
    const int completion_tokens =
        frame.at("usage").at("completion_tokens");
    CHECK(completion_tokens >= previous_completion_tokens);
    CHECK(frame.at("usage").at("total_tokens").get<int>() ==
          frame.at("usage").at("prompt_tokens").get<int>() +
              completion_tokens);
    if (frame.at("choices").empty()) {
      CHECK(i + 2 == res.sse_chunks.size());
      CHECK(completion_tokens == kN);
    } else {
      CHECK(completion_tokens > previous_completion_tokens);
    }
    previous_completion_tokens = completion_tokens;
  }
  CHECK(previous_completion_tokens == kN);
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
std::string InVocabChatPrompt(
    const std::vector<ChatMessage>& messages, bool,
    const std::vector<ChatCompletionToolsParam>&) {
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

// Ported from tests/entrypoints/openai/chat_completion/test_chat.py:
// test_chat_completion_stream_options @ e24d1b24.
TEST_CASE("serving_chat: continuous usage starts at role and ends empty-choice") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const int kN = 5;
  Harness h(c, w, Fixture());
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);

  ChatCompletionRequest req = MakeChatRequest(
      {ChatMessage{"user", std::string("hello")}}, kN, /*stream=*/true);
  req.stream_options = StreamOptions{/*include_usage=*/true,
                                     /*continuous_usage_stats=*/true};
  ChatCompletionResult res = serving.create_chat_completion(req);
  REQUIRE(res.sse_chunks.size() >= 4);
  CHECK(res.sse_chunks.back() == "data: [DONE]\n\n");

  const json role = json::parse(SsePayload(res.sse_chunks.front()));
  CHECK(role.at("choices").at(0).at("delta").at("role") == "assistant");
  CHECK(role.at("usage").at("prompt_tokens").get<int>() > 0);
  CHECK(role.at("usage").at("completion_tokens") == 0);

  int previous_completion_tokens = 0;
  for (size_t i = 1; i + 1 < res.sse_chunks.size(); ++i) {
    const json frame = json::parse(SsePayload(res.sse_chunks[i]));
    REQUIRE(frame.contains("usage"));
    const int completion_tokens =
        frame.at("usage").at("completion_tokens");
    CHECK(completion_tokens >= previous_completion_tokens);
    if (frame.at("choices").empty()) {
      CHECK(i + 2 == res.sse_chunks.size());
      CHECK(completion_tokens == kN);
    }
    previous_completion_tokens = completion_tokens;
  }
  CHECK(previous_completion_tokens == kN);
}

// ─── M3.3 Task 3: tool-call serving wiring ───────────────────────────────────
namespace {
// A request carrying one tool (get_weather) with tool_choice defaulting to auto.
ChatCompletionRequest ToolRequest(bool stream = false) {
  ChatCompletionRequest r;
  r.messages = {ChatMessage{"user", std::string("weather in Paris?")}};
  r.stream = stream;
  ChatCompletionToolsParam tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.description = "Get the weather for a city.";
  tool.function.parameters = nlohmann::json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},)"
      R"("required":["city"]})");
  r.tools = std::vector<ChatCompletionToolsParam>{tool};
  return r;
}
}  // namespace

// ─── (a) NON-STREAM tool call: a forced <tool_call> output → the response
//     message carries tool_calls + finish_reason="tool_calls" ──────────────────
TEST_CASE("serving_chat: non-stream tool call shapes tool_calls + finish_reason") {
  const ChatCompletionRequest req = ToolRequest();
  REQUIRE(ToolsEnabled(req));
  std::unique_ptr<ToolParser> parser = get_tool_parser("hermes");
  REQUIRE(parser != nullptr);

  // Mocked model output (the synthetic engine can't emit a real tool call).
  const std::string model_output =
      "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": "
      "\"Paris\"}}\n</tool_call>";
  ShapedChatMessage shaped =
      ShapeChatMessage("assistant", model_output, std::string("stop"), req,
                       parser.get());

  CHECK(shaped.message.role == "assistant");
  REQUIRE(shaped.message.tool_calls.has_value());
  REQUIRE(shaped.message.tool_calls->size() == 1);
  const auto& tc = (*shaped.message.tool_calls)[0];
  CHECK(tc.type == "function");
  CHECK(tc.function.name == "get_weather");
  CHECK(tc.function.arguments == "{\"city\":\"Paris\"}");
  CHECK(tc.id.rfind("chatcmpl-tool-", 0) == 0);
  // No leading content before the tool call.
  CHECK_FALSE(shaped.message.content.has_value());
  REQUIRE(shaped.finish_reason.has_value());
  CHECK(*shaped.finish_reason == "tool_calls");

  // The serialized OpenAI response omits content(null)/keeps tool_calls.
  ChatCompletionResponseChoice choice;
  choice.message = shaped.message;
  choice.finish_reason = shaped.finish_reason;
  json j = choice;
  CHECK(j.at("finish_reason") == "tool_calls");
  CHECK(j.at("message").at("content").is_null());
  REQUIRE(j.at("message").contains("tool_calls"));
  CHECK(j.at("message").at("tool_calls").at(0).at("function").at("name") ==
        "get_weather");
}

// ─── (a2) Leading content before the tool call is preserved on the message ────
TEST_CASE("serving_chat: non-stream tool call keeps leading content") {
  const ChatCompletionRequest req = ToolRequest();
  std::unique_ptr<ToolParser> parser = get_tool_parser("hermes");
  const std::string model_output =
      "Let me check.\n<tool_call>{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Paris\"}}</tool_call>";
  ShapedChatMessage shaped = ShapeChatMessage(
      "assistant", model_output, std::string("stop"), req, parser.get());
  REQUIRE(shaped.message.content.has_value());
  CHECK(*shaped.message.content == "Let me check.\n");
  REQUIRE(shaped.message.tool_calls.has_value());
  CHECK((*shaped.message.tool_calls)[0].function.name == "get_weather");
  CHECK(*shaped.finish_reason == "tool_calls");
}

// ─── (e) content-only (tools present but no <tool_call>) → normal message ─────
TEST_CASE("serving_chat: non-stream content-only with tools is a normal message") {
  const ChatCompletionRequest req = ToolRequest();
  std::unique_ptr<ToolParser> parser = get_tool_parser("hermes");
  const std::string model_output = "The weather in Paris is sunny.";
  ShapedChatMessage shaped = ShapeChatMessage(
      "assistant", model_output, std::string("length"), req, parser.get());
  REQUIRE(shaped.message.content.has_value());
  CHECK(*shaped.message.content == model_output);
  CHECK_FALSE(shaped.message.tool_calls.has_value());
  CHECK(*shaped.finish_reason == "length");  // passthrough, not "tool_calls"
}

// ─── tool_choice="none" disables extraction (backward compat) ────────────────
TEST_CASE("serving_chat: tool_choice none disables tool extraction") {
  ChatCompletionRequest req = ToolRequest();
  req.tool_choice = ToolChoice{"none", std::nullopt};
  CHECK_FALSE(ToolsEnabled(req));
  std::unique_ptr<ToolParser> parser = get_tool_parser("hermes");
  const std::string model_output =
      "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>";
  ShapedChatMessage shaped = ShapeChatMessage(
      "assistant", model_output, std::string("stop"), req, parser.get());
  // With tools disabled the <tool_call> text is passed through as content.
  CHECK_FALSE(shaped.message.tool_calls.has_value());
  REQUIRE(shaped.message.content.has_value());
  CHECK(*shaped.message.content == model_output);
  CHECK(*shaped.finish_reason == "stop");
}

// ─── M3.3b Task 3: tool_choice -> STRUCTURAL-TAG spec (auto lazy, required/named
//     forced). ApplyToolChoiceStructuredOutput now sets structured_outputs
//     .structural_tag (the native lazy/forced compile), NOT the old forced-json
//     grammar. Mirrors vLLM get_hermes_structural_tag (structural_tag_registry.py
//     :237-269) + _hermes_tool_tags (:213-234). ──────────────────────────────
namespace {
// ToolRequest() + a second tool (set_alarm), for required-with-multiple tests.
ChatCompletionRequest TwoToolRequest() {
  ChatCompletionRequest r = ToolRequest();
  ChatCompletionToolsParam tool2;
  tool2.type = "function";
  tool2.function.name = "set_alarm";
  tool2.function.parameters = nlohmann::json::parse(
      R"({"type":"object","properties":{"time":{"type":"string"}},)"
      R"("required":["time"]})");
  r.tools->push_back(tool2);
  return r;
}

// The distinct tool names covered by a spec's `tags` (each tool -> 2 variants).
std::set<std::string> TagBeginNames(const json& spec) {
  std::set<std::string> names;
  for (const auto& tag : spec.at("tags")) {
    names.insert(tag.at("begin").get<std::string>());
  }
  return names;
}
}  // namespace

// (a) tool_choice=auto (+ tools) -> a LAZY structural tag: lazy:true, triggers
//     ["<tool_call>"], NOT forced (the whole point — a plain reply stays free).
TEST_CASE("serving_chat: tool_choice auto -> LAZY structural tag (not forced)") {
  ChatCompletionRequest req = ToolRequest();  // tool_choice unset == auto default
  const auto spec_unset = ToolChoiceStructuralTagSpec(req);
  REQUIRE(spec_unset.has_value());
  CHECK(spec_unset->at("lazy") == true);
  CHECK(spec_unset->at("triggers") == json::array({"<tool_call>"}));
  CHECK(spec_unset->at("stop_after_first") == false);

  // Explicit "auto" behaves identically.
  req.tool_choice = ToolChoice{"auto", std::nullopt};
  const auto spec = ToolChoiceStructuralTagSpec(req);
  REQUIRE(spec.has_value());
  CHECK(spec->at("lazy") == true);
  CHECK(spec->at("triggers") == json::array({"<tool_call>"}));

  // Two surface variants per tool (structural_tag_registry.py:219-221): begins
  // carry the baked-in name + arguments prefix; content is the params schema.
  REQUIRE(spec->at("tags").size() == 2);
  CHECK(spec->at("tags").at(0).at("begin") ==
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": ");
  CHECK(spec->at("tags").at(0).at("end") == "}\n</tool_call>");
  CHECK(spec->at("tags").at(1).at("begin") ==
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": ");
  CHECK(spec->at("tags").at(1).at("end") == "}</tool_call>");
  CHECK(spec->at("tags").at(0).at("content_schema") ==
        *req.tools->at(0).function.parameters);

  // It flows onto the SamplingParams as structured_outputs.structural_tag (the
  // ONE constraint; json/grammar stay unset), passing Verify().
  SamplingParams sp = req.to_sampling_params(16);
  ApplyToolChoiceStructuredOutput(req, sp);
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->structural_tag.has_value());
  CHECK_FALSE(sp.structured_outputs->json.has_value());
  CHECK_FALSE(sp.structured_outputs->grammar.has_value());
  // The serialized spec parses back to the LAZY form (the user-facing payoff:
  // auto does NOT force a tool call from token 0).
  const json flowed = json::parse(*sp.structured_outputs->structural_tag);
  CHECK(flowed.at("lazy") == true);
  CHECK(flowed.at("triggers") == json::array({"<tool_call>"}));
}

// A tool with NO parameters -> content_schema constrained to any JSON (`true`).
TEST_CASE("serving_chat: tool without parameters -> content_schema=any (true)") {
  ChatCompletionRequest req = ToolRequest();
  req.tools->at(0).function.parameters = std::nullopt;
  req.tool_choice = ToolChoice{"function", std::string("get_weather")};
  const auto spec = ToolChoiceStructuralTagSpec(req);
  REQUIRE(spec.has_value());
  CHECK(spec->at("tags").at(0).at("content_schema") == true);
}

// (d) tool_choice=required -> FORCED (lazy:false), stop_after_first:false, one
//     tag-pair per tool.
TEST_CASE("serving_chat: tool_choice required -> forced structural tag (>=1)") {
  ChatCompletionRequest req = TwoToolRequest();
  req.tool_choice = ToolChoice{"required", std::nullopt};
  const auto spec = ToolChoiceStructuralTagSpec(req);
  REQUIRE(spec.has_value());
  CHECK(spec->at("lazy") == false);
  CHECK(spec->at("stop_after_first") == false);
  CHECK_FALSE(spec->contains("triggers"));  // forced from token 0, no trigger.
  // Both tools appear (each with two surface variants -> 4 tags).
  REQUIRE(spec->at("tags").size() == 4);
  CHECK(TagBeginNames(*spec) ==
        std::set<std::string>{
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": ",
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": ",
            "<tool_call>\n{\"name\": \"set_alarm\", \"arguments\": ",
            "<tool_call>{\"name\": \"set_alarm\", \"arguments\": "});

  SamplingParams sp = req.to_sampling_params(16);
  ApplyToolChoiceStructuredOutput(req, sp);
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->structural_tag.has_value());
  CHECK_FALSE(sp.structured_outputs->grammar.has_value());
  CHECK(json::parse(*sp.structured_outputs->structural_tag).at("lazy") == false);
}

// (e) tool_choice=named ("function") -> FORCED exactly one (lazy:false,
//     stop_after_first:true), ONLY the chosen tool's tags.
TEST_CASE("serving_chat: tool_choice named -> forced one (stop_after_first)") {
  ChatCompletionRequest req = TwoToolRequest();
  req.tool_choice = ToolChoice{"function", std::string("set_alarm")};
  const auto spec = ToolChoiceStructuralTagSpec(req);
  REQUIRE(spec.has_value());
  CHECK(spec->at("lazy") == false);
  CHECK(spec->at("stop_after_first") == true);
  // Only the named tool's two variants (not the other tool).
  REQUIRE(spec->at("tags").size() == 2);
  CHECK(TagBeginNames(*spec) ==
        std::set<std::string>{
            "<tool_call>\n{\"name\": \"set_alarm\", \"arguments\": ",
            "<tool_call>{\"name\": \"set_alarm\", \"arguments\": "});

  SamplingParams sp = req.to_sampling_params(16);
  ApplyToolChoiceStructuredOutput(req, sp);
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->structural_tag.has_value());
  const json flowed = json::parse(*sp.structured_outputs->structural_tag);
  CHECK(flowed.at("stop_after_first") == true);
}

// A named choice matching NO listed tool -> no constraint (upstream 400s first).
TEST_CASE("serving_chat: named choice with no matching tool -> no tag") {
  ChatCompletionRequest req = ToolRequest();
  req.tool_choice = ToolChoice{"function", std::string("does_not_exist")};
  CHECK_FALSE(ToolChoiceStructuralTagSpec(req).has_value());
}

// (f) tool_choice=none / no tools -> NO structural tag (model unconstrained).
TEST_CASE("serving_chat: tool_choice none (and no tools) -> no structural tag") {
  ChatCompletionRequest req = ToolRequest();
  req.tool_choice = ToolChoice{"none", std::nullopt};
  CHECK_FALSE(ToolChoiceStructuralTagSpec(req).has_value());

  ChatCompletionRequest bare;
  bare.messages = {ChatMessage{"user", std::string("hi")}};
  CHECK_FALSE(ToolChoiceStructuralTagSpec(bare).has_value());

  // ApplyToolChoiceStructuredOutput leaves structured_outputs untouched (unset).
  SamplingParams sp = req.to_sampling_params(16);
  ApplyToolChoiceStructuredOutput(req, sp);
  CHECK_FALSE(sp.structured_outputs.has_value());
}

// (b)+(c) auto is RELAXED end-to-end at the serving layer: a plain-text output
//     (no `<tool_call>`) shapes a NORMAL message (finish_reason="stop", no
//     tool_calls) — the model was NOT forced — while a `<tool_call>{...}` output
//     shapes tool_calls + finish_reason="tool_calls".
TEST_CASE("serving_chat: auto -> plain reply is NOT forced into a tool call") {
  ChatCompletionRequest req = ToolRequest();  // auto default
  req.tool_choice = ToolChoice{"auto", std::nullopt};
  std::unique_ptr<ToolParser> parser = get_tool_parser("hermes");
  REQUIRE(parser != nullptr);

  // (b) The model chose to just reply (no trigger) -> plain content message.
  const std::string plain = "It is sunny in Paris today.";
  ShapedChatMessage reply = ShapeChatMessage(
      "assistant", plain, std::string("stop"), req, parser.get());
  CHECK_FALSE(reply.message.tool_calls.has_value());
  REQUIRE(reply.message.content.has_value());
  CHECK(*reply.message.content == plain);
  REQUIRE(reply.finish_reason.has_value());
  CHECK(*reply.finish_reason == "stop");

  // (c) The model chose to call the tool -> tool_calls + "tool_calls".
  std::unique_ptr<ToolParser> parser2 = get_tool_parser("hermes");
  const std::string called =
      "<tool_call>{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Paris\"}}</tool_call>";
  ShapedChatMessage call = ShapeChatMessage(
      "assistant", called, std::string("stop"), req, parser2.get());
  REQUIRE(call.message.tool_calls.has_value());
  REQUIRE(call.message.tool_calls->size() == 1);
  CHECK((*call.message.tool_calls)[0].function.name == "get_weather");
  REQUIRE(call.finish_reason.has_value());
  CHECK(*call.finish_reason == "tool_calls");
}

// ─── (b/d) STREAM tool call: the delta sequence carries tool_calls[index=0,
//     id, function.name] first, then function.arguments deltas ────────────────
TEST_CASE("serving_chat: streaming tool call emits name-first then arg deltas") {
  const ChatCompletionRequest req = ToolRequest(/*stream=*/true);
  std::unique_ptr<ToolParser> parser = get_tool_parser("hermes");

  // The model output arrives fragmented across deltas (what the engine streams).
  const std::vector<std::string> deltas = {
      "<tool_call>", "{\"name\": \"get_weather\", ",
      "\"arguments\": {\"city\": ", "\"Paris\"}}", "</tool_call>"};

  // Drive ShapeChatDelta exactly as serving_chat's streaming loop does, then
  // serialize each emitted DeltaMessage into a chat.completion.chunk.
  std::string previous;
  bool tools_streamed = false;
  std::optional<std::string> streamed_id;
  std::optional<std::string> streamed_name;
  std::string streamed_args;
  int name_chunks = 0;
  for (const std::string& delta : deltas) {
    const std::string current = previous + delta;
    std::optional<DeltaMessage> dm =
        ShapeChatDelta(previous, current, delta, req, parser.get());
    previous = current;
    if (!dm.has_value()) continue;
    if (dm->tool_calls.has_value() && !dm->tool_calls->empty()) {
      tools_streamed = true;
    }
    // Serialize as the stream chunk (the same DeltaMessage the SSE carries).
    ChatCompletionResponseStreamChoice choice;
    choice.delta = *dm;
    json jc = choice;
    if (jc.at("delta").contains("tool_calls")) {
      for (const auto& tcj : jc.at("delta").at("tool_calls")) {
        CHECK(tcj.at("index") == 0);
        const auto& fn = tcj.at("function");
        if (fn.contains("name")) {
          ++name_chunks;
          streamed_name = fn.at("name").get<std::string>();
          REQUIRE(tcj.contains("id"));
          streamed_id = tcj.at("id").get<std::string>();
          CHECK(tcj.at("type") == "function");
        }
        if (fn.contains("arguments")) {
          streamed_args += fn.at("arguments").get<std::string>();
        }
      }
    }
  }
  CHECK(tools_streamed);
  CHECK(name_chunks == 1);  // the name is emitted exactly once
  REQUIRE(streamed_name.has_value());
  CHECK(*streamed_name == "get_weather");
  REQUIRE(streamed_id.has_value());
  CHECK(streamed_id->rfind("chatcmpl-tool-", 0) == 0);
  CHECK(streamed_args == "{\"city\": \"Paris\"}");

  // The finish-reason rule the serving loop applies on the terminal chunk.
  const std::string finish = (tools_streamed) ? "tool_calls" : "stop";
  CHECK(finish == "tool_calls");
}

// ─── (e) STREAM backward compat: tools present, content-only output over the
//     REAL engine → normal role/content/finish cadence (no tool_calls) ─────────
TEST_CASE("serving_chat: streaming with tools but content-only is unchanged") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 5;

  Harness h(c, w, tok);
  OpenAIServingChat serving(h.engine, "test-model", InVocabChatPrompt);
  ChatCompletionRequest req =
      MakeChatRequest({ChatMessage{"user", std::string("hello")}}, kN, true);
  // Attach tools: the synthetic model won't emit <tool_call>, so the stream
  // must still be plain content with finish_reason from the engine.
  ChatCompletionToolsParam tool;
  tool.function.name = "noop";
  req.tools = std::vector<ChatCompletionToolsParam>{tool};

  ChatCompletionResult res = serving.create_chat_completion(req);
  REQUIRE(res.streaming);
  CHECK(res.sse_chunks.back() == "data: [DONE]\n\n");

  std::string streamed;
  std::optional<std::string> last_finish;
  for (size_t i = 1; i + 1 < res.sse_chunks.size(); ++i) {
    json j = json::parse(SsePayload(res.sse_chunks[i]));
    const auto& choice = j.at("choices").at(0);
    const auto& delta = choice.at("delta");
    CHECK_FALSE(delta.contains("tool_calls"));  // no tool call emitted
    if (delta.contains("content")) {
      streamed += delta.at("content").get<std::string>();
    }
    if (!choice.at("finish_reason").is_null()) {
      last_finish = choice.at("finish_reason").get<std::string>();
    }
  }
  CHECK_FALSE(streamed.empty());
  REQUIRE(last_finish.has_value());
  CHECK(*last_finish == "length");  // engine finish, NOT "tool_calls"
}
