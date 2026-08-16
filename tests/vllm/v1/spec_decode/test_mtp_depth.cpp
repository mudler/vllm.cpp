// SPEC-MTP-K-GT-1 (#81) — MTP speculation DEPTH, through the production loader.
//
// This is the reachability gate for the row. It builds a real engine stack
// (LoadedEngine -> EngineCore -> Scheduler -> GPUModelRunner) over a small
// synthetic Qwen3.5 DENSE model on CPU, exactly as
// tests/vllm/entrypoints/test_loaded_engine_dense.cpp does, and configures depth
// the way a user does: through EngineParams::speculative_config. Nothing here
// constructs a proposer by hand, because a test that does proves the class works
// and never that the configured depth reaches it.
//
// PHASE 1, the refusal. The MTP propose is k=1 only: this tree ported upstream's
// k=1 early exit (autoregressive/speculator.py:236-238 @ 555967922) and not the
// autoregressive multi-step loop behind it, so
// GPUModelRunner::propose_drafts stashes exactly ONE draft per request
// (runner.cpp:2174) while the KV pool and the verify graph are both sized for k.
// The engine must therefore refuse depth by name instead of billing for it.
//
// RED-first for this file: before the refusal, constructing the engine with
// `{"method":"mtp","num_speculative_tokens":3}` SUCCEEDS and silently drafts one
// token, so the CHECK_THROWS_AS below does not throw at all.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::DenseMlpWeights;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseWeights;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vt::DType;

namespace {

// ─── Synthetic dense weights (mirrors test_loaded_engine_dense.cpp) ──────────
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

constexpr int kVocab = 24;      // == the tiny BPE fixture's ids 0..23, no holes.
constexpr int kMaxModelLen = 32;

HfConfig MakeDenseConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = kVocab;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = kMaxModelLen;
  // mtp_num_hidden_layers == 1, which is what both gate checkpoints ship and
  // what LoadedEngine::ResolveSpecConfig reads to resolve n_predict.
  c.raw = json::object();
  c.raw["mtp_num_hidden_layers"] = 1;
  return c;
}

DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

Qwen3_5DenseWeights MakeDenseWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
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
    Qwen3_5DenseLayerWeights lw;
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
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// The tiny oracle-verified BPE fixture (ids 0..23, no holes).
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_mtp_depth_tok_" + std::to_string(counter++) + ".json"))
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
  json vocab = {{"h", 0},   {"e", 1},   {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},   {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11}, {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15}, {"orld", 16}, {"Ġworld", 17},
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

SamplingParams Greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax) -> deterministic and depth-neutral.
  sp.max_tokens = max_tokens;
  sp.output_kind = RequestOutputKind::kCumulative;
  return sp;
}

EngineParams SpecParams(int k) {
  EngineParams p;  // defaults: block_size 32 == max_model_len 32.
  p.speculative_config = vllm::ParseSpeculativeConfigJson(
      R"({"method":"mtp","num_speculative_tokens":)" + std::to_string(k) + "}");
  return p;
}

}  // namespace

TEST_CASE("mtp depth: a configured depth above 1 is REFUSED by name") {
  const HfConfig c = MakeDenseConfig();

  // The whole defect in one construction: this used to build an engine that
  // reserved KV for 3 and drafted 1, silently.
  CHECK_THROWS_AS(
      LoadedEngine(c, MakeDenseWeights(c), BuildFixture(), SpecParams(3)),
      std::invalid_argument);

  // The message must NAME the missing part, so a reader learns WHAT is absent
  // rather than that "something is unsupported".
  std::string what;
  try {
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(2));
  } catch (const std::invalid_argument& e) {
    what = e.what();
  }
  CHECK(what.find("num_speculative_tokens") != std::string::npos);
  CHECK(what.find("multi-step") != std::string::npos);
  CHECK(what.find("#81") != std::string::npos);
}

TEST_CASE("mtp depth: k=1 still builds and generates, and spec-OFF is unmoved") {
  // The refusal must be a depth gate and nothing wider: the served depth and the
  // no-speculation default both keep working, and greedy MTP is exactness
  // preserving, so the two agree token for token.
  const HfConfig c = MakeDenseConfig();
  const std::string prompt = "hello";
  const int kN = 6;

  RequestOutput off;
  RequestOutput k1;
  {
    EngineParams p;  // no speculative_config: the production default.
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), p);
    off = eng.engine().generate(prompt, Greedy(kN), "req");
  }
  {
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(1));
    k1 = eng.engine().generate(prompt, Greedy(kN), "req");
  }

  REQUIRE(off.finished);
  REQUIRE(off.outputs.size() == 1);
  CHECK(static_cast<int>(off.outputs[0].token_ids.size()) == kN);
  REQUIRE(k1.finished);
  REQUIRE(k1.outputs.size() == 1);
  CHECK(k1.outputs[0].token_ids == off.outputs[0].token_ids);
}

TEST_CASE("mtp depth: the refusal is MTP-only and does not catch ngram") {
  // The n-gram proposer genuinely returns 0..k drafts per request per step
  // (runner.cpp:2273-2282 moves the whole per-request vector), so a depth gate
  // that caught it would be a regression rather than a safety net.
  const HfConfig c = MakeDenseConfig();
  EngineParams p;
  p.speculative_config = vllm::ParseSpeculativeConfigJson(
      R"({"method":"ngram","num_speculative_tokens":3})");
  LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), p);
  const RequestOutput out = eng.engine().generate("hello", Greedy(4), "req");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  CHECK(static_cast<int>(out.outputs[0].token_ids.size()) == 4);
}
