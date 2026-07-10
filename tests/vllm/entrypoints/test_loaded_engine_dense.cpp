// LoadedEngine DENSE-arch dispatch test (M0.8 27B CPU wiring) — proves the last
// CPU-side plumbing: the full LoadedEngine / Executor / EngineCore / LLMEngine
// stack loads and dispatches the DENSE 27B arch (Qwen3_5ForConditionalGeneration,
// num_experts==0) through the dense weights + the dense paged forward
// (Qwen3_5DenseModel::Forward), without regressing the MoE 35B path. CPU-only;
// the real 12GB W4A4 checkpoint + GPU are NOT touched (the greedy acceptance gate
// on the real checkpoint is test_qwen27_paged_engine.cpp, still SKIPPING).
//
// The 35B analogue for the MoE stack is tests/vllm/v1/test_llm_engine.cpp; this is
// its dense sibling driven through the packaging LoadedEngine seam. Cases:
//   1. ModelRegistry dispatch decision: explicit architecture IDs select their
//      factories; num_experts is no longer used as a model-class key.
//   2. Dense stack runs end to end: a LoadedEngine built via the DENSE constructor
//      over synthetic dense weights + a small hybrid dense config generates
//      exactly N greedy tokens, terminates, and is deterministic across two fresh
//      stacks — proving the dense weights thread executor -> engine -> runner and
//      the dense forward actually drives the loop.
#include "vllm/entrypoints/model_loader.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::DenseMlpWeights;
using vllm::HfConfig;
using vllm::ModelRegistry;
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

// ─── Synthetic weights (mirrors test_qwen27_paged_forward.cpp) ───────────────
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

// Vocab == the tiny BPE fixture's assigned ids (0..23), block_size ==
// max_model_len == hash_block_size (hybrid coordinator constraint; prompts far
// shorter than a block keep prefix caching inert), matching test_llm_engine.cpp.
constexpr int kVocab = 24;
constexpr int kMaxModelLen = 32;

// 27B-shaped small DENSE config: layer_types [LA, LA, LA, FA], no experts,
// GQA ratio 3 (Hv/Hk = 6/2), attn_output_gate. num_experts==0 => dense arch.
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
  c.linear_num_value_heads = 6;  // GQA ratio 3
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

// The tiny oracle-verified BPE fixture (ids 0..23, no holes) from test_llm_engine.
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_dense_engine_tok_" + std::to_string(counter++) + ".json"))
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

Tokenizer FreshFixture() { return BuildFixture(); }

SamplingParams Greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax) -> deterministic.
  sp.max_tokens = max_tokens;
  sp.output_kind = RequestOutputKind::kCumulative;
  return sp;
}

}  // namespace

// ─── 1. Arch-select: the FromModelDir dispatch decision ──────────────────────
TEST_CASE("loaded_engine: ModelRegistry routes explicit 27B dense vs 35B MoE IDs") {
  HfConfig dense = MakeDenseConfig();
  // Deliberately contradict the old structural heuristic: the architecture ID
  // remains authoritative.
  dense.num_experts = 4;
  CHECK(ModelRegistry::Resolve(dense).architecture ==
        "Qwen3_5ForConditionalGeneration");

  HfConfig moe = MakeDenseConfig();
  moe.model_type = "qwen3_5_moe_text";
  moe.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  moe.num_experts = 0;
  CHECK(ModelRegistry::Resolve(moe).architecture ==
        "Qwen3_5MoeForConditionalGeneration");
}

// ─── 2. The dense stack drives the full engine loop end to end ───────────────
TEST_CASE("loaded_engine: dense 27B arch generates deterministically through the full stack") {
  const HfConfig c = MakeDenseConfig();
  const std::string prompt = "hello";
  const int kN = 6;
  EngineParams params;  // defaults: block_size 32 == max_model_len 32.

  RequestOutput run1;
  RequestOutput run2;
  {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    run1 = eng.engine().generate(prompt, Greedy(kN), "req");
    CHECK_FALSE(eng.engine().has_unfinished_requests());  // loop terminated.
  }
  {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    run2 = eng.engine().generate(prompt, Greedy(kN), "req");
  }

  REQUIRE(run1.finished);
  REQUIRE(run1.outputs.size() == 1);
  // Exactly N tokens (no eos configured -> length finish): the dense forward
  // actually produced a stream through executor -> engine_core -> runner.
  CHECK(static_cast<int>(run1.outputs[0].token_ids.size()) == kN);
  REQUIRE(run1.outputs[0].finish_reason.has_value());
  CHECK(*run1.outputs[0].finish_reason == "length");

  // Deterministic: two fresh dense stacks over the same prompt agree.
  REQUIRE(run2.outputs.size() == 1);
  CHECK(run1.outputs[0].token_ids == run2.outputs[0].token_ids);
  CHECK(run1.outputs[0].text == run2.outputs[0].text);
}

TEST_CASE(
    "loaded_engine: ResolveMaxNumBatchedTokens per-arch default (dense 2048 "
    "flat, MoE concurrency-aware)") {
  EngineParams p;
  const int kBigLen = 262144;  // large max_model_len => the tiny-model ceiling
                               // (max_model_len*seqs) never binds here.

  // DENSE arch: vLLM's scheduler default 2048, FLAT across concurrency
  // (DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, vllm/config/scheduler.py:42 @
  // e24d1b24).
  for (int seqs : {8, 16, 32, 64}) {
    p.max_num_seqs = seqs;
    CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen,
                                                   /*is_dense_arch=*/true) ==
          2048);
  }

  // MoE arch: GB10-tuned concurrency-aware budget (unchanged behavior).
  p.max_num_seqs = 8;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 4096);
  p.max_num_seqs = 16;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 4096);
  p.max_num_seqs = 32;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 8192);
  p.max_num_seqs = 64;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 8192);

  // Explicit override wins for BOTH arches (the CLI --max-num-batched-tokens).
  p.max_num_seqs = 32;
  p.max_num_batched_tokens = 8192;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, true) == 8192);
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 8192);
  // ... but is still clamped up to the >= max_num_seqs invariant
  // (SchedulerConfig.verify_max_model_len, vllm/config/scheduler.py:87).
  p.max_num_seqs = 64;
  p.max_num_batched_tokens = 4;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, true) == 64);

  // Tiny-model ceiling preservation: whole workload smaller than the default
  // => budget capped at max_model_len*seqs (no behavior change for the small
  // synthetic CPU engines).
  p.max_num_batched_tokens = 0;
  p.max_num_seqs = 4;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, /*max_model_len=*/64,
                                                 true) == 256);
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, /*max_model_len=*/64,
                                                 false) == 256);
}
