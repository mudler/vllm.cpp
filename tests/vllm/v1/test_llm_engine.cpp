// End-to-end tests for the V1 LLMEngine (M1.8 Task 6) — the whole loop wired:
// InputProcessor -> EngineCore(Scheduler + Executor over the real GPUModelRunner)
// -> OutputProcessor -> RequestOutput. Ported from vllm/v1/engine/llm_engine.py @
// e24d1b24 (add_request/step) + the LLM.generate driver.
//
// Drives a SMALL SYNTHETIC hybrid-MoE Qwen3.6 model on CPU (the real 35B greedy
// through the full paged loop on dgx is the milestone acceptance gate — dgx-
// pending). This proves the pieces actually CONNECT and produce a deterministic
// stream. Cases:
//   1. greedy determinism + termination: generate(prompt, greedy, max_tokens=N)
//      yields exactly N tokens, the loop terminates, and two runs of the same
//      prompt+greedy produce the SAME tokens.
//   2. 2-request concurrent batch: two requests added, stepped together, each
//      gets its own correct stream (matches its standalone single-request run).
//   3. streaming (DELTA) vs non-streaming (CUMULATIVE): the concatenated deltas
//      equal the cumulative full text for the same prompt+greedy.
//   4. stop on max_tokens (LENGTH finish) end to end.
//   5. stop on a stop STRING: the OutputProcessor's string-stop -> reqs_to_abort
//      -> EngineCore aborts -> the loop ends (finish_reason "stop").
//
// The synthetic model uses vocab_size == the tiny BPE fixture's assigned id count
// (0..23, no holes) so every greedy argmax is decodable. The KV cache config uses
// a UNIFIED block_size == max_model_len == hash_block_size (the hybrid KV
// coordinator requires every group's block_size == hash_block_size); with prompts
// far shorter than a block, prefix caching stays inert (no block ever completes).
#include "vllm/v1/engine/llm_engine.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
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
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::SchedulerConfig;
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

// ─── Synthetic weights (mirrors tests/vllm/v1/worker/test_runner.cpp) ────────
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

// Vocab size == the tiny BPE fixture's assigned ids (0..23): every greedy argmax
// over the lm_head is decodable by the detokenizer (no vocab holes).
constexpr int kVocab = 24;
constexpr int kBlockSize = 32;      // == max_model_len == hash_block_size (hybrid)
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 32;

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
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

// Hybrid KV config: one full-attn group + one mamba (GDN) group, UNIFIED
// block_size == max_model_len == hash_block_size (hybrid coordinator constraint).
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

// The tiny oracle-verified BPE fixture from tests/vllm/v1/test_output_processor
// (ids 0..23, no holes): "hello"=13, " world"=17, "1"=8, "2"=9, ...
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_llmengine_tok_" + std::to_string(counter++) + ".json"))
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
  vocab[MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 23;
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

SamplingParams Greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax) -> deterministic.
  sp.max_tokens = max_tokens;
  sp.output_kind = RequestOutputKind::kCumulative;
  return sp;
}

// Owns a fully-wired LLMEngine stack over the shared const config/weights/
// tokenizer. Members are declared in dependency order (Scheduler + runner ->
// Executor -> EngineCore; InputProcessor/OutputProcessor -> LLMEngine) so the
// by-reference seams stay valid for the stack's lifetime.
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
    // block_size == hash_block_size; prompts are far shorter than a block, so no
    // block ever completes and this stays inert (prefix caching never fires).
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

}  // namespace

// ─── 1. Greedy determinism + termination ─────────────────────────────────────
TEST_CASE("llm_engine: greedy generate is deterministic and terminates at max_tokens") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";
  const int kN = 6;

  RequestOutput run1;
  RequestOutput run2;
  {
    Harness h(c, w, tok);
    run1 = h.engine.generate(prompt, Greedy(kN), "req");
    CHECK_FALSE(h.engine.has_unfinished_requests());  // loop terminated.
  }
  {
    Harness h(c, w, tok);  // fresh stack -> fresh KV/scheduler state.
    run2 = h.engine.generate(prompt, Greedy(kN), "req");
  }

  REQUIRE(run1.finished);
  REQUIRE(run1.outputs.size() == 1);
  // Exactly N tokens produced (no eos configured -> length finish).
  CHECK(static_cast<int>(run1.outputs[0].token_ids.size()) == kN);
  REQUIRE(run1.outputs[0].finish_reason.has_value());
  CHECK(*run1.outputs[0].finish_reason == "length");

  // Deterministic: same prompt + greedy -> identical token stream + text.
  REQUIRE(run2.outputs.size() == 1);
  CHECK(run1.outputs[0].token_ids == run2.outputs[0].token_ids);
  CHECK(run1.outputs[0].text == run2.outputs[0].text);
}

// ─── 2. Two-request concurrent batch ─────────────────────────────────────────
TEST_CASE("llm_engine: 2-request concurrent batch produces correct per-request streams") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 5;

  // Oracle: each request run on its OWN engine.
  RequestOutput solo_a;
  RequestOutput solo_b;
  {
    Harness h(c, w, tok);
    solo_a = h.engine.generate("hello", Greedy(kN), "A");
  }
  {
    Harness h(c, w, tok);
    solo_b = h.engine.generate("world", Greedy(kN), "B");
  }

  // Both concurrently on one engine, stepped together.
  Harness h(c, w, tok);
  h.engine.add_request("A", "hello", Greedy(kN));
  h.engine.add_request("B", "world", Greedy(kN));

  std::map<std::string, RequestOutput> finished;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) finished[out.request_id] = out;
    }
  }

  REQUIRE(finished.count("A") == 1);
  REQUIRE(finished.count("B") == 1);
  // Each concurrent stream matches its standalone single-request run (the batch
  // path is per-request independent).
  CHECK(finished["A"].outputs[0].token_ids == solo_a.outputs[0].token_ids);
  CHECK(finished["B"].outputs[0].token_ids == solo_b.outputs[0].token_ids);
  CHECK(finished["A"].outputs[0].token_ids != finished["B"].outputs[0].token_ids);
}

// ─── 3. Streaming (DELTA) vs non-streaming (CUMULATIVE) equivalence ───────────
TEST_CASE("llm_engine: streaming deltas concatenate to the non-streaming full text") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";
  const int kN = 6;

  // Non-streaming: the finished CUMULATIVE output carries the full text.
  RequestOutput cumulative;
  {
    Harness h(c, w, tok);
    cumulative = h.engine.generate(prompt, Greedy(kN), "req");
  }

  // Streaming: DELTA output_kind — accumulate the per-step deltas.
  std::string streamed;
  std::vector<int32_t> streamed_ids;
  {
    Harness h(c, w, tok);
    SamplingParams sp = Greedy(kN);
    sp.output_kind = RequestOutputKind::kDelta;
    h.engine.add_request("req", prompt, sp);
    while (h.engine.has_unfinished_requests()) {
      for (const RequestOutput& out : h.engine.step()) {
        if (!out.outputs.empty()) {
          streamed += out.outputs[0].text;
          for (int32_t t : out.outputs[0].token_ids) streamed_ids.push_back(t);
        }
      }
    }
  }

  REQUIRE(cumulative.outputs.size() == 1);
  CHECK(streamed == cumulative.outputs[0].text);
  CHECK(streamed_ids == cumulative.outputs[0].token_ids);
}

// ─── 4. Stop on max_tokens (LENGTH finish), driven a step at a time ───────────
TEST_CASE("llm_engine: max_tokens length-stops the request end to end") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 3;

  Harness h(c, w, tok);
  h.engine.add_request("req", "hello", Greedy(kN));

  int steps = 0;
  RequestOutput result;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) result = std::move(out);
    }
    ++steps;
    REQUIRE(steps < 100);  // must terminate.
  }

  REQUIRE(result.finished);
  REQUIRE(result.outputs.size() == 1);
  CHECK(static_cast<int>(result.outputs[0].token_ids.size()) == kN);
  CHECK(*result.outputs[0].finish_reason == "length");
  CHECK_FALSE(h.engine.has_unfinished_requests());
}

// ─── 5. Stop on a stop STRING (OutputProcessor -> reqs_to_abort -> abort) ──────
TEST_CASE("llm_engine: a stop string ends the request through the full loop") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";

  // First produce the deterministic full greedy text so we can pick a stop
  // substring that actually appears in it.
  std::string full_text;
  {
    Harness h(c, w, tok);
    RequestOutput r = h.engine.generate(prompt, Greedy(8), "probe");
    full_text = r.outputs[0].text;
  }
  REQUIRE(full_text.size() >= 2);
  // A stop string drawn from the middle of the produced text (skip the first
  // char so truncation leaves a non-trivial prefix).
  const std::string stop = full_text.substr(1, 1);

  Harness h(c, w, tok);
  SamplingParams sp = Greedy(8);
  sp.stop = {stop};
  h.engine.add_request("req", prompt, sp);

  int steps = 0;
  RequestOutput result;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) result = std::move(out);
    }
    ++steps;
    REQUIRE(steps < 100);
  }

  REQUIRE(result.finished);
  REQUIRE(result.outputs.size() == 1);
  REQUIRE(result.outputs[0].finish_reason.has_value());
  // The string stop is detected by the OutputProcessor (not the scheduler's
  // token-level check_stop) -> finish_reason "stop", stop_reason == the string.
  CHECK(*result.outputs[0].finish_reason == "stop");
  REQUIRE(result.outputs[0].stop_reason.has_value());
  CHECK(*result.outputs[0].stop_reason == stop);
  // The reqs_to_abort feedback tore the request down in the EngineCore too.
  CHECK_FALSE(h.engine.has_unfinished_requests());
  // Output truncated before the stop string (include_stop_str_in_output=false).
  CHECK(result.outputs[0].text.find(stop) == std::string::npos);
}
