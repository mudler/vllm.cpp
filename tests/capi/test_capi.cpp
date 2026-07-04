// C ABI tests (M3.5 Task 1): drive the PUBLIC C API (include/vllm.h) over a
// SYNTHETIC hybrid-MoE Qwen3.5 engine built in-memory (no disk weights) via the
// internal MakeEngineHandle test hook. Proves vllm_complete produces
// deterministic non-empty text + finish reason + token counts, the error/null
// paths return status codes without throwing, and the string ownership is
// leak-free (the suite runs under ASan in CI).
//
// The synthetic weights/config/tokenizer mirror tests/vllm/v1/test_llm_engine.cpp
// (the M1.8/M3.1 harness): vocab 0..23 with no holes (every greedy argmax is
// decodable), UNIFIED block_size == max_model_len == hash_block_size.
#include "vllm.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/engine_handle.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5MoeWeights;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vt::DType;

namespace {

// ─── Synthetic weights (mirrors test_llm_engine.cpp) ─────────────────────────
constexpr int kVocab = 24;
constexpr int kBlockSize = 32;   // == max_model_len == hash_block_size (hybrid).
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 32;

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
  const int64_t Hv = c.linear_num_value_heads, Dk = c.linear_key_head_dim,
                Dv = c.linear_value_head_dim, Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv,
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

// The tiny oracle-verified BPE fixture (ids 0..23, no holes) from
// test_llm_engine.cpp: "hello"=13, " world"=17, ...
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_capi_tok_" + std::to_string(counter++) + ".json"))
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

EngineParams SyntheticParams() {
  EngineParams p;
  p.block_size = kBlockSize;
  p.num_blocks = kNumBlocks;
  p.max_model_len = kMaxModelLen;
  p.max_num_seqs = 8;
  return p;
}

// Build a fresh synthetic engine handle (owns the full C++ stack), driven by the
// public C API. Caller frees via vllm_engine_free.
vllm_engine* MakeSyntheticEngine() {
  const HfConfig c = MakeConfig();
  auto loaded = std::make_unique<LoadedEngine>(c, MakeWeights(c), BuildFixture(),
                                               SyntheticParams());
  return vllm::capi::MakeEngineHandle(std::move(loaded));
}

vllm_sampling_params GreedyParams(int32_t max_tokens) {
  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = 0.0f;  // greedy (argmax) -> deterministic.
  sp.max_tokens = max_tokens;
  return sp;
}

}  // namespace

// ─── (a) greedy complete: deterministic non-empty text + reason + counts ─────
TEST_CASE("capi: vllm_complete greedy yields non-empty text, length finish, counts") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  vllm_sampling_params sp = GreedyParams(6);
  vllm_completion out;
  const vllm_status st = vllm_complete(eng, "hello", &sp, &out);

  CHECK(st == VLLM_OK);
  REQUIRE(out.text != nullptr);
  CHECK(std::string(out.text).size() > 0);
  REQUIRE(out.finish_reason != nullptr);
  CHECK(std::string(out.finish_reason) == "length");  // no eos -> max_tokens.
  CHECK(out.prompt_tokens == 1);        // "hello" == single token id 13.
  CHECK(out.completion_tokens == 6);    // exactly max_tokens produced.

  vllm_completion_free(&out);
  CHECK(out.text == nullptr);  // freed + zeroed.
  vllm_engine_free(eng);
}

// ─── (b) determinism: two greedy calls, same prompt -> identical text ────────
TEST_CASE("capi: two greedy completions of the same prompt are identical") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  vllm_sampling_params sp = GreedyParams(6);
  vllm_completion a;
  vllm_completion b;
  CHECK(vllm_complete(eng, "hello", &sp, &a) == VLLM_OK);
  CHECK(vllm_complete(eng, "hello", &sp, &b) == VLLM_OK);
  REQUIRE(a.text != nullptr);
  REQUIRE(b.text != nullptr);
  CHECK(std::string(a.text) == std::string(b.text));
  CHECK(a.completion_tokens == b.completion_tokens);

  vllm_completion_free(&a);
  vllm_completion_free(&b);
  vllm_engine_free(eng);
}

// ─── (c) error path: bad model path -> status + last_error, no crash ─────────
TEST_CASE("capi: vllm_engine_load with a bad path returns an error and sets last_error") {
  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = "/nonexistent/vllm-cpp/model/dir";
  vllm_engine* eng = reinterpret_cast<vllm_engine*>(0x1);  // must be nulled.
  const vllm_status st = vllm_engine_load(&mp, &eng);

  CHECK(st != VLLM_OK);
  CHECK(st == VLLM_ERR_MODEL_LOAD);
  CHECK(eng == nullptr);  // out left null on failure.
  CHECK(std::string(vllm_last_error()).size() > 0);
  // No throw / crash reaching here proves the ABI boundary caught the exception.
}

// ─── (d) null-argument path: VLLM_ERR_INVALID_ARGUMENT, no crash ─────────────
TEST_CASE("capi: null arguments return VLLM_ERR_INVALID_ARGUMENT without crashing") {
  // Null out-handle on load.
  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = "/whatever";
  CHECK(vllm_engine_load(&mp, nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  // Null params on load, out gets nulled.
  vllm_engine* eng = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(nullptr, &eng) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(eng == nullptr);

  // Null engine on complete (checked before any handle deref).
  vllm_sampling_params sp = GreedyParams(4);
  vllm_completion out;
  CHECK(vllm_complete(nullptr, "hi", &sp, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(out.text == nullptr);  // out is zeroed even on the error path.

  // Null prompt / params / out with a VALID engine (a C ABI can only null-check
  // pointers, not validate a non-null garbage handle).
  vllm_engine* engine = MakeSyntheticEngine();
  REQUIRE(engine != nullptr);
  CHECK(vllm_complete(engine, nullptr, &sp, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete(engine, "hi", nullptr, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete(engine, "hi", &sp, nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  vllm_engine_free(engine);
}

// ─── (e) string ownership: free helpers are leak-free (ASan) ─────────────────
TEST_CASE("capi: string / completion free helpers are leak-free and null-safe") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);
  vllm_sampling_params sp = GreedyParams(5);

  // vllm_completion_free frees out->text.
  vllm_completion out;
  CHECK(vllm_complete(eng, "hello", &sp, &out) == VLLM_OK);
  vllm_completion_free(&out);

  // vllm_string_free frees the raw text member directly.
  vllm_completion out2;
  CHECK(vllm_complete(eng, "world", &sp, &out2) == VLLM_OK);
  vllm_string_free(out2.text);
  out2.text = nullptr;

  // Null-safety of the free helpers.
  vllm_string_free(nullptr);
  vllm_completion_free(nullptr);
  vllm_engine_free(nullptr);

  vllm_engine_free(eng);
}

// ─── version / abi ───────────────────────────────────────────────────────────
TEST_CASE("capi: version and abi-version are exposed") {
  CHECK(std::string(vllm_version()).size() > 0);
  CHECK(vllm_abi_version() == VLLM_ABI_VERSION);
}
