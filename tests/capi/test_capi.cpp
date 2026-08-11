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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "capi/engine_handle.h"
#include "vllm/config/device.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/platforms/interface.h"
#include "vllm/entrypoints/openai/serving_utils.h"
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

// Chat-capable synthetic engine: same stack, but the chat serving is built with
// an IN-VOCAB prompt seam (the tiny fixture vocab cannot spell a real chat
// template), mirroring the api-server harness's InVocabChatPrompt.
vllm_engine* MakeSyntheticChatEngine() {
  const HfConfig c = MakeConfig();
  auto loaded = std::make_unique<LoadedEngine>(c, MakeWeights(c), BuildFixture(),
                                               SyntheticParams());
  return vllm::capi::MakeEngineHandle(
      std::move(loaded),
      [](const std::vector<vllm::entrypoints::openai::ChatMessage>& messages,
         bool /*add_generation_prompt*/,
         const std::vector<
             vllm::entrypoints::openai::ChatCompletionToolsParam>& /*tools*/) {
        std::string p;
        for (const auto& m : messages)
          if (m.content.has_value()) p += *m.content;
        return p;
      });
}

vllm_sampling_params GreedyParams(int32_t max_tokens) {
  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = 0.0f;  // greedy (argmax) -> deterministic.
  sp.max_tokens = max_tokens;
  return sp;
}

// A seeded SAMPLED config (temperature > 0 + a fixed seed) -> the M1.7 sampler
// drives generation, but a fixed seed makes two runs deterministic.
vllm_sampling_params SeededSampledParams(int32_t max_tokens, uint64_t seed) {
  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = 0.8f;
  sp.top_k = 0;   // consider all tokens.
  sp.top_p = 1.0f;
  sp.max_tokens = max_tokens;
  sp.has_seed = 1;
  sp.seed = seed;
  return sp;
}

// ─── Streaming callback accumulator (user_data round-trip) ───────────────────
struct StreamAccumulator {
  std::string text;          // concatenation of all delta_text.
  int deltas = 0;            // number of callback invocations.
  bool saw_finished = false;  // a final call with finished == true arrived.
  bool all_valid_utf8 = true;  // every delta_text was well-formed UTF-8.
  int stop_after = -1;       // if >= 0, return false once `deltas` reaches it.
};

// Minimal UTF-8 well-formedness check (rejects stray continuation bytes,
// overlong / truncated sequences) — proves no invalid bytes reach the callback.
bool IsValidUtf8(const std::string& s) {
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 0;
    if (c < 0x80) {
      len = 1;
    } else if ((c & 0xE0) == 0xC0) {
      len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      len = 4;
    } else {
      return false;  // stray continuation / invalid lead byte.
    }
    if (i + len > n) return false;  // truncated.
    for (size_t k = 1; k < len; ++k) {
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    }
    i += len;
  }
  return true;
}

// The C callback: appends the delta, tracks finish + UTF-8 validity, and honors
// the early-stop threshold. Signature matches vllm_token_callback exactly.
bool AccumulateCb(const char* delta_text, bool finished, void* user_data) {
  auto* acc = static_cast<StreamAccumulator*>(user_data);
  acc->deltas += 1;
  if (delta_text != nullptr) {
    const std::string d(delta_text);
    if (!IsValidUtf8(d)) acc->all_valid_utf8 = false;
    acc->text += d;
  }
  if (finished) acc->saw_finished = true;
  if (acc->stop_after >= 0 && acc->deltas >= acc->stop_after) return false;
  return true;
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

// ─── (b1b) ABI v13 pre-tokenized completion ──────────────────────────────────
TEST_CASE("capi: vllm_complete_tokens matches the string-prompt completion (ABI v13)") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  vllm_sampling_params sp = GreedyParams(6);
  // The string leg: "hello" tokenizes to the single id 13 in the synthetic
  // tokenizer (see the vllm_complete greedy case above).
  vllm_completion via_str;
  REQUIRE(vllm_complete(eng, "hello", &sp, &via_str) == VLLM_OK);

  const int32_t prompt[1] = {13};
  int32_t out_tokens[16] = {0};
  int32_t n_out = -1;
  vllm_completion via_tok;
  const vllm_status st =
      vllm_complete_tokens(eng, prompt, 1, &sp, out_tokens, 16, &n_out, &via_tok);
  CHECK(st == VLLM_OK);
  CHECK(n_out == 6);  // greedy max_tokens, all reported
  // Hand-pinned synthetic-model greedy stream.  This is intentionally
  // independent of the string leg: a broken implementation that merely
  // reports six zero-initialized buffer entries must not satisfy ABI v12.
  const int32_t expected_ids[6] = {22, 12, 14, 9, 13, 2};
  for (int i = 0; i < 6; ++i) {
    INFO("generated token index ", i);
    CHECK(out_tokens[i] == expected_ids[i]);
  }
  REQUIRE(via_tok.text != nullptr);
  // Same engine, same greedy params, same (single-token) prompt => the SAME
  // deterministic completion through both entry points.
  CHECK(std::string(via_tok.text) == std::string(via_str.text));
  CHECK(via_tok.prompt_tokens == 1);
  CHECK(via_tok.completion_tokens == 6);
  REQUIRE(via_tok.finish_reason != nullptr);
  CHECK(std::string(via_tok.finish_reason) == "length");

  // A truncating buffer reports fewer ids but never changes the generation.
  int32_t small[2] = {0};
  int32_t n_small = -1;
  CHECK(vllm_complete_tokens(eng, prompt, 1, &sp, small, 2, &n_small, nullptr) ==
        VLLM_OK);
  CHECK(n_small == 2);
  CHECK(small[0] == out_tokens[0]);
  CHECK(small[1] == out_tokens[1]);

  // Null contracts.
  CHECK(vllm_complete_tokens(nullptr, prompt, 1, &sp, out_tokens, 16, &n_out,
                             nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete_tokens(eng, nullptr, 1, &sp, out_tokens, 16, &n_out,
                             nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete_tokens(eng, prompt, 0, &sp, out_tokens, 16, &n_out,
                             nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete_tokens(eng, prompt, 1, &sp, nullptr, 16, &n_out,
                             nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete_tokens(eng, prompt, 1, &sp, out_tokens, 16, nullptr,
                             nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  n_out = -1;
  CHECK(vllm_complete_tokens(eng, prompt, 1, &sp, out_tokens, -1, &n_out,
                             nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(n_out == 0);

  vllm_completion_free(&via_str);
  vllm_completion_free(&via_tok);
  vllm_engine_free(eng);
}

// ─── (b2) ABI v8 custom logits processor: forces a token end-to-end ──────────
namespace {
// Force-a-token processor: state carried through vllm_logits_processor_user_data.
struct CApiForceState {
  int32_t forced = 0;         // the token id to force every decode step.
  int calls = 0;              // number of callback invocations.
  int32_t last_vocab = 0;     // vocab_size seen (all rows share the model vocab).
  int max_n_tokens = -1;      // largest n_token_ids observed.
  bool all_prev_forced = true;  // every observed prior token == forced.
  bool tokens_ptr_ok = true;  // token_ids non-null whenever n_token_ids > 0.
};

// Matches vllm_logits_processor exactly. Forces greedy argmax to `forced` and
// records the arity/contract the sampler handed it.
void CApiForceCb(const int32_t* token_ids, int32_t n_token_ids, float* logits,
                 int32_t vocab_size, void* user_data) {
  auto* s = static_cast<CApiForceState*>(user_data);
  s->calls += 1;
  s->last_vocab = vocab_size;
  if (n_token_ids > s->max_n_tokens) s->max_n_tokens = n_token_ids;
  if (n_token_ids > 0 && token_ids == nullptr) s->tokens_ptr_ok = false;
  for (int32_t i = 0; i < n_token_ids; ++i)
    if (token_ids[i] != s->forced) s->all_prev_forced = false;
  for (int32_t j = 0; j < vocab_size; ++j) logits[j] = -1e30f;
  if (s->forced >= 0 && s->forced < vocab_size) logits[s->forced] = 1e30f;
}
}  // namespace

TEST_CASE("capi: custom logits processor forces the generated token (ABI v8)") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  const int32_t kMax = 6;

  // Baseline greedy (no processor) — the byte-identical default path.
  vllm_sampling_params base = GreedyParams(kMax);
  CHECK(base.logits_processor == nullptr);  // default is unset
  vllm_completion baseline;
  REQUIRE(vllm_complete(eng, "hello", &base, &baseline) == VLLM_OK);
  REQUIRE(baseline.text != nullptr);
  const std::string baseline_text = baseline.text;

  // Force token id 5 every step. vocab is 0..23, all decodable.
  CApiForceState st;
  st.forced = 5;
  vllm_sampling_params sp = GreedyParams(kMax);
  sp.logits_processor = &CApiForceCb;
  sp.logits_processor_user_data = &st;
  vllm_completion forced;
  REQUIRE(vllm_complete(eng, "hello", &sp, &forced) == VLLM_OK);
  REQUIRE(forced.text != nullptr);

  // (a) The callback fired ONCE PER DECODE STEP with the correct arity.
  CHECK(st.calls == forced.completion_tokens);  // one call per generated token
  CHECK(st.calls == kMax);
  CHECK(st.last_vocab == kVocab);               // full model vocab exposed
  CHECK(st.tokens_ptr_ok);                      // ptr non-null whenever len > 0
  CHECK(st.max_n_tokens >= 0);                  // well-formed token_ids view
  // Every prior token the callback observed was the forced one (vacuously true
  // when the async engine has not yet fed back output tokens; see residual note).
  CHECK(st.all_prev_forced);
  // RESIDUAL (async scheduling): under the async scheduler the generated
  // output-token bookkeeping is fed back by the scheduler's update_from_output
  // (exactly like penalties / min_tokens / bad_words on the async path), so the
  // token_ids view a callback sees end-to-end can lag the emitted tokens. The
  // STRICT per-step token_ids contract (n_token_ids == the generated prefix, the
  // ptr valid) is gated deterministically at the sampler level in
  // tests/vllm/v1/sample/test_sampler.cpp ("custom logits processor forces the
  // greedy token exactly", which passes output_token_ids and asserts n == 2).

  // (b) RED-first: forcing changed the output vs the untouched greedy baseline.
  CHECK(std::string(forced.text) != baseline_text);

  // Determinism: forcing a DIFFERENT token yields a DIFFERENT output.
  CApiForceState st2;
  st2.forced = 9;
  vllm_sampling_params sp2 = GreedyParams(kMax);
  sp2.logits_processor = &CApiForceCb;
  sp2.logits_processor_user_data = &st2;
  vllm_completion forced2;
  REQUIRE(vllm_complete(eng, "hello", &sp2, &forced2) == VLLM_OK);
  REQUIRE(forced2.text != nullptr);
  CHECK(std::string(forced2.text) != std::string(forced.text));
  CHECK(st2.all_prev_forced);

  vllm_completion_free(&baseline);
  vllm_completion_free(&forced);
  vllm_completion_free(&forced2);
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

// ─── (c2) ABI v7 tri-state prefix-caching field round-trip ───────────────────
TEST_CASE("capi: enable_prefix_caching tri-state defaults to 0 and validates") {
  // Default is 0 == model default (the byte-identical default). This is the
  // field the server's --enable-radix-attention alias sets (RadixAttention is
  // fused into our APC; see .agents/specs/sglang-radixattention.md §1).
  vllm_model_params mp = vllm_model_params_default();
  CHECK(mp.enable_prefix_caching == 0);

  // Valid states 0/1/2 pass the tri-state gate (they then reach model load,
  // which fails on the fake path with MODEL_LOAD — not INVALID_ARGUMENT).
  for (int v : {0, 1, 2}) {
    vllm_model_params p = vllm_model_params_default();
    p.model_path = "/nonexistent/vllm-cpp/model/dir";
    p.enable_prefix_caching = v;
    vllm_engine* eng = nullptr;
    CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
    CHECK(eng == nullptr);
  }

  // An out-of-range tri-state is rejected BEFORE any load attempt.
  vllm_model_params bad = vllm_model_params_default();
  bad.model_path = "/nonexistent/vllm-cpp/model/dir";
  bad.enable_prefix_caching = 3;
  vllm_engine* eng = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&bad, &eng) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(eng == nullptr);
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

// ─── (f) streaming == blocking: deltas concatenate to the blocking result ────
TEST_CASE("capi: vllm_complete_stream deltas concatenate to the blocking result") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  vllm_sampling_params sp = GreedyParams(6);

  // Blocking reference.
  vllm_completion blocking;
  REQUIRE(vllm_complete(eng, "hello", &sp, &blocking) == VLLM_OK);
  REQUIRE(blocking.text != nullptr);

  // Streaming: accumulate every delta via the user_data pointer.
  StreamAccumulator acc;
  const vllm_status st =
      vllm_complete_stream(eng, "hello", &sp, &AccumulateCb, &acc);

  CHECK(st == VLLM_OK);
  CHECK(acc.deltas > 0);
  CHECK(acc.saw_finished);                    // a final finished=true call.
  CHECK(acc.all_valid_utf8);                  // (d) no invalid bytes reached us.
  // The streaming boundary sanitizes UTF-8 per delta (a raw-bytes detokenizer
  // can emit an invalid multibyte); blocking vllm_complete returns the raw text.
  // So the invariant is: concat(deltas) == SanitizeUtf8(blocking text).
  const std::string sanitized_blocking =
      vllm::entrypoints::openai::SanitizeUtf8(std::string(blocking.text));
  CHECK(acc.text == sanitized_blocking);

  vllm_completion_free(&blocking);
  vllm_engine_free(eng);
}

// ─── (g) early-stop: cb returns false -> generation stops, request torn down ──
TEST_CASE("capi: vllm_complete_stream early-stop tears the request down cleanly") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  vllm_sampling_params sp = GreedyParams(10);

  // Stop after the 2nd delta -> far fewer than max_tokens (10) deltas.
  StreamAccumulator acc;
  acc.stop_after = 2;
  CHECK(vllm_complete_stream(eng, "hello", &sp, &AccumulateCb, &acc) == VLLM_OK);
  CHECK(acc.deltas == 2);  // stopped early, did not run to max_tokens.

  // The engine must be reusable: a subsequent full call works (proves the
  // early-stopped request was aborted, not left lingering as unfinished).
  StreamAccumulator acc2;
  CHECK(vllm_complete_stream(eng, "hello", &sp, &AccumulateCb, &acc2) == VLLM_OK);
  CHECK(acc2.saw_finished);
  CHECK(acc2.deltas > 2);  // this one runs to natural finish.

  // And blocking still works on the same engine after an early-stop.
  vllm_completion out;
  CHECK(vllm_complete(eng, "hello", &sp, &out) == VLLM_OK);
  CHECK(out.text != nullptr);
  vllm_completion_free(&out);

  vllm_engine_free(eng);
}

// A callback that THROWS mid-stream (a C++ FFI consumer's callback can raise).
bool ThrowingCb(const char* /*delta_text*/, bool /*finished*/, void* user_data) {
  auto* n = static_cast<int*>(user_data);
  ++(*n);
  if (*n >= 1) throw std::runtime_error("callback boom");
  return true;
}

// ─── (g2) a throwing callback / mid-stream error must NOT poison the engine ───
// Regression for a heap-use-after-free: the stream path formerly aborted the
// in-flight request ONLY on the callback-returns-false branch, and both entry
// points reused a FIXED request id "0". An exception escaping the loop left "0"
// registered; the NEXT call's add_request("0") freed-and-reinserted the key while
// the scheduler still held the old Request → UAF in Request::NumTokens(). The fix
// = unique per-call ids + a RAII guard that aborts on every exit path. This test
// throws from the callback, then reuses the engine (a plain build corrupts /
// ASan flags UAF without the fix).
TEST_CASE("capi: a throwing stream callback leaves the engine reusable (no UAF)") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);
  vllm_sampling_params sp = GreedyParams(10);

  // The throwing callback unwinds out of vllm_complete_stream; the ABI catches it
  // and returns a runtime error (never throws across the boundary).
  int calls = 0;
  CHECK(vllm_complete_stream(eng, "hello", &sp, &ThrowingCb, &calls) ==
        VLLM_ERR_RUNTIME);
  CHECK(std::string(vllm_last_error()).find("boom") != std::string::npos);

  // The engine must still be fully usable — the aborted request left no dangling
  // state, and the next call uses a fresh id so it cannot collide.
  StreamAccumulator acc;
  CHECK(vllm_complete_stream(eng, "hello", &sp, &AccumulateCb, &acc) == VLLM_OK);
  CHECK(acc.saw_finished);
  vllm_completion out;
  CHECK(vllm_complete(eng, "hello", &sp, &out) == VLLM_OK);
  CHECK(out.text != nullptr);
  vllm_completion_free(&out);

  vllm_engine_free(eng);
}

// ─── (h) seeded SAMPLED run is deterministic across two calls ────────────────
TEST_CASE("capi: seeded sampled streaming is deterministic across two calls") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  vllm_sampling_params sp = SeededSampledParams(6, /*seed=*/1234u);

  StreamAccumulator a;
  StreamAccumulator b;
  CHECK(vllm_complete_stream(eng, "hello", &sp, &AccumulateCb, &a) == VLLM_OK);
  CHECK(vllm_complete_stream(eng, "hello", &sp, &AccumulateCb, &b) == VLLM_OK);

  CHECK(a.saw_finished);
  CHECK(b.saw_finished);
  CHECK(a.text == b.text);      // same seed -> identical sampled output.
  // W2 mirrors RequestOutputCollector's single-slot DELTA coalescing: callback
  // chunk count is scheduling-dependent when the producer outruns the
  // consumer, while concatenated text/tokens remain deterministic.
  CHECK(a.deltas > 0);
  CHECK(b.deltas > 0);

  vllm_engine_free(eng);
}

// ─── (i) null-argument path for the streaming API ────────────────────────────
TEST_CASE("capi: vllm_complete_stream null arguments return INVALID_ARGUMENT") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);
  vllm_sampling_params sp = GreedyParams(4);
  StreamAccumulator acc;

  CHECK(vllm_complete_stream(nullptr, "hi", &sp, &AccumulateCb, &acc) ==
        VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete_stream(eng, nullptr, &sp, &AccumulateCb, &acc) ==
        VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete_stream(eng, "hi", nullptr, &AccumulateCb, &acc) ==
        VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_complete_stream(eng, "hi", &sp, nullptr, &acc) ==
        VLLM_ERR_INVALID_ARGUMENT);

  vllm_engine_free(eng);
}

// ─── (j) W2 non-blocking submit: independent callbacks share AsyncLLM ───────
TEST_CASE("capi: non-blocking request_submit runs concurrent callback streams") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);
  vllm_sampling_params sp = GreedyParams(8);

  StreamAccumulator first;
  StreamAccumulator second;
  vllm_request* first_request = nullptr;
  vllm_request* second_request = nullptr;
  REQUIRE(vllm_request_submit(eng, "hello", &sp, &AccumulateCb, &first,
                              &first_request) == VLLM_OK);
  REQUIRE(vllm_request_submit(eng, "world", &sp, &AccumulateCb, &second,
                              &second_request) == VLLM_OK);
  REQUIRE(first_request != nullptr);
  REQUIRE(second_request != nullptr);

  CHECK(vllm_request_wait(first_request) == VLLM_OK);
  CHECK(vllm_request_wait(second_request) == VLLM_OK);
  CHECK(vllm_request_done(first_request));
  CHECK(vllm_request_done(second_request));
  CHECK(std::string(vllm_request_error(first_request)).empty());
  CHECK(std::string(vllm_request_error(second_request)).empty());
  CHECK(first.saw_finished);
  CHECK(second.saw_finished);
  CHECK(first.deltas > 0);
  CHECK(second.deltas > 0);

  vllm_request_free(first_request);
  vllm_request_free(second_request);
  vllm_engine_free(eng);
}

TEST_CASE("capi: non-blocking request cancellation and null contracts") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);
  vllm_sampling_params sp = GreedyParams(24);
  StreamAccumulator acc;
  vllm_request* request = nullptr;
  REQUIRE(vllm_request_submit(eng, "hello", &sp, &AccumulateCb, &acc,
                              &request) == VLLM_OK);
  REQUIRE(request != nullptr);
  CHECK(vllm_request_cancel(request) == VLLM_OK);
  CHECK(vllm_request_cancel(request) == VLLM_OK);  // idempotent
  CHECK(vllm_request_wait(request) == VLLM_OK);
  CHECK(vllm_request_done(request));
  vllm_request_free(request);

  CHECK(vllm_request_submit(nullptr, "hello", &sp, &AccumulateCb, &acc,
                            &request) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_request_submit(eng, nullptr, &sp, &AccumulateCb, &acc,
                            &request) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_request_submit(eng, "hello", nullptr, &AccumulateCb, &acc,
                            &request) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_request_submit(eng, "hello", &sp, nullptr, &acc, &request) ==
        VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_request_submit(eng, "hello", &sp, &AccumulateCb, &acc, nullptr) ==
        VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_request_cancel(nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_request_wait(nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK_FALSE(vllm_request_done(nullptr));
  CHECK(std::string(vllm_request_error(nullptr)).empty());
  vllm_request_free(nullptr);

  vllm_engine_free(eng);
}

TEST_CASE("capi: non-blocking callback errors propagate and leave the engine reusable") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);
  vllm_sampling_params sp = GreedyParams(10);

  int calls = 0;
  vllm_request* request = nullptr;
  REQUIRE(vllm_request_submit(eng, "hello", &sp, &ThrowingCb, &calls,
                              &request) == VLLM_OK);
  REQUIRE(request != nullptr);
  CHECK(vllm_request_wait(request) == VLLM_ERR_RUNTIME);
  CHECK(vllm_request_done(request));
  CHECK(std::string(vllm_request_error(request)).find("callback boom") !=
        std::string::npos);
  vllm_request_free(request);

  vllm_completion out;
  CHECK(vllm_complete(eng, "hello", &sp, &out) == VLLM_OK);
  CHECK(out.text != nullptr);
  vllm_completion_free(&out);
  vllm_engine_free(eng);
}

// ─── structured output (ABI v2) ──────────────────────────────────────────────
// A `choice` constraint over the synthetic vocab: the single greedy token must
// be one of the allowed strings ("1" == id 8, "2" == id 9), which the
// unconstrained argmax provably is not (asserted first). This proves the C ABI
// fields reach the engine's per-step grammar bitmask, end to end through the
// production LoadedEngine wiring — not just the SamplingParams translation.
TEST_CASE("capi: structured_choice constrains greedy decoding") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  // Unconstrained greedy baseline: argmax of "hello" is NOT "1" or "2".
  vllm_sampling_params base = GreedyParams(1);
  vllm_completion unconstrained;
  REQUIRE(vllm_complete(eng, "hello", &base, &unconstrained) == VLLM_OK);
  REQUIRE(unconstrained.text != nullptr);
  const std::string baseline(unconstrained.text);
  CAPTURE(baseline);
  CHECK(baseline != "1");
  CHECK(baseline != "2");
  vllm_completion_free(&unconstrained);

  const char* choices[] = {"1", "2"};
  vllm_sampling_params sp = GreedyParams(1);
  sp.structured_choice = choices;
  sp.n_structured_choice = 2;
  vllm_completion out;
  REQUIRE(vllm_complete(eng, "hello", &sp, &out) == VLLM_OK);
  REQUIRE(out.text != nullptr);
  const std::string text(out.text);
  CAPTURE(text);
  CHECK((text == "1" || text == "2"));
  vllm_completion_free(&out);
  vllm_engine_free(eng);
}

// The same constraint through the streaming path: every delta is drawn from the
// constrained token set, so the concatenation equals one of the choices for the
// first token.
TEST_CASE("capi: structured_choice constrains vllm_complete_stream") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  const char* choices[] = {"1", "2"};
  vllm_sampling_params sp = GreedyParams(1);
  sp.structured_choice = choices;
  sp.n_structured_choice = 2;

  StreamAccumulator acc;
  REQUIRE(vllm_complete_stream(eng, "hello", &sp, &AccumulateCb, &acc) ==
          VLLM_OK);
  CHECK(acc.saw_finished);
  CAPTURE(acc.text);
  CHECK((acc.text == "1" || acc.text == "2"));
  vllm_engine_free(eng);
}

// Exactly-one rule: two constraints set at once must be rejected with a status
// (no throw across the ABI) and a mentioning error, and the engine must stay
// usable afterwards.
TEST_CASE("capi: more than one structured constraint is rejected cleanly") {
  vllm_engine* eng = MakeSyntheticEngine();
  REQUIRE(eng != nullptr);

  vllm_sampling_params sp = GreedyParams(1);
  sp.structured_grammar = "root ::= \"1\"";
  sp.structured_json_object = 1;
  vllm_completion out;
  CHECK(vllm_complete(eng, "hello", &sp, &out) != VLLM_OK);
  CHECK(out.text == nullptr);
  CHECK(std::string(vllm_last_error()).size() > 0);

  // Engine reusable after the rejected request.
  vllm_sampling_params ok = GreedyParams(1);
  vllm_completion out2;
  CHECK(vllm_complete(eng, "hello", &ok, &out2) == VLLM_OK);
  CHECK(out2.text != nullptr);
  vllm_completion_free(&out2);
  vllm_engine_free(eng);
}

// ─── chat entry points (ABI v3) ──────────────────────────────────────────────
// vllm_chat: one OpenAI chat request in, one ChatCompletionResponse JSON out.
// The engine-side serving stack (template seam -> sampling -> engine ->
// response shaping) runs behind the C ABI; greedy keeps it deterministic.
TEST_CASE("capi: vllm_chat returns a chat.completion response JSON") {
  vllm_engine* eng = MakeSyntheticChatEngine();
  REQUIRE(eng != nullptr);

  const char* request =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
      "\"temperature\":0,\"max_tokens\":6}";
  char* response = nullptr;
  REQUIRE(vllm_chat(eng, request, &response) == VLLM_OK);
  REQUIRE(response != nullptr);
  const json body = json::parse(response);
  CAPTURE(std::string(response));
  CHECK(body.at("object") == "chat.completion");
  CHECK(body.at("choices").size() == 1);
  CHECK(body.at("choices").at(0).at("message").at("role") == "assistant");
  CHECK(!body.at("choices").at(0).at("message").at("content")
             .get<std::string>()
             .empty());
  CHECK(body.at("usage").at("completion_tokens").get<int>() == 6);
  vllm_string_free(response);
  vllm_engine_free(eng);
}

// vllm_chat_stream: role-first chunk cadence, content deltas that concatenate
// to the blocking result, then the terminal finished callback.
TEST_CASE("capi: vllm_chat_stream chunks concatenate to the blocking result") {
  vllm_engine* eng = MakeSyntheticChatEngine();
  REQUIRE(eng != nullptr);

  const char* request =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
      "\"temperature\":0,\"max_tokens\":6}";

  char* blocking = nullptr;
  REQUIRE(vllm_chat(eng, request, &blocking) == VLLM_OK);
  const std::string expected = json::parse(blocking)
                                   .at("choices").at(0)
                                   .at("message").at("content")
                                   .get<std::string>();
  vllm_string_free(blocking);

  struct ChatAccumulator {
    std::string content;
    bool saw_role = false;
    bool saw_finished = false;
    int chunks = 0;
    bool every_chunk_parsed = true;
  } acc;
  auto cb = [](const char* delta_text, bool finished, void* user_data) -> bool {
    auto* a = static_cast<ChatAccumulator*>(user_data);
    if (finished) {
      a->saw_finished = true;
      return true;
    }
    a->chunks++;
    json chunk;
    try {
      chunk = json::parse(delta_text);
    } catch (...) {
      a->every_chunk_parsed = false;
      return true;
    }
    if (chunk.value("object", "") != "chat.completion.chunk") {
      a->every_chunk_parsed = false;
      return true;
    }
    const json& delta = chunk.at("choices").at(0).at("delta");
    if (delta.contains("role")) a->saw_role = true;
    if (delta.contains("content") && delta.at("content").is_string())
      a->content += delta.at("content").get<std::string>();
    return true;
  };
  REQUIRE(vllm_chat_stream(eng, request, cb, &acc) == VLLM_OK);
  CHECK(acc.saw_finished);
  CHECK(acc.saw_role);
  CHECK(acc.every_chunk_parsed);
  CHECK(acc.chunks > 1);
  CAPTURE(acc.content);
  CAPTURE(expected);
  CHECK(acc.content == expected);
  vllm_engine_free(eng);
}

// Malformed request JSON is rejected with INVALID_ARGUMENT (no throw across
// the ABI) and the engine stays usable.
TEST_CASE("capi: vllm_chat rejects malformed request JSON cleanly") {
  vllm_engine* eng = MakeSyntheticChatEngine();
  REQUIRE(eng != nullptr);

  char* response = nullptr;
  CHECK(vllm_chat(eng, "{not json", &response) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(response == nullptr);
  CHECK(std::string(vllm_last_error()).size() > 0);

  const char* request =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
      "\"temperature\":0,\"max_tokens\":2}";
  CHECK(vllm_chat(eng, request, &response) == VLLM_OK);
  CHECK(response != nullptr);
  vllm_string_free(response);
  vllm_engine_free(eng);
}

// ─── tool-parser selection (ABI v4) ──────────────────────────────────────────
// An UNKNOWN explicit tool-parser name must fail the FIRST chat call with
// VLLM_ERR_INVALID_ARGUMENT (not crash, not silently disable parsing) and set
// vllm_last_error; an explicit "hermes" (a registered parser) serves normally.
TEST_CASE("capi: an unknown tool_parser fails the first chat call cleanly") {
  vllm_engine* eng = MakeSyntheticChatEngine();
  REQUIRE(eng != nullptr);
  vllm::capi::SetEngineToolParser(eng, "no-such-parser");

  const char* request =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
      "\"temperature\":0,\"max_tokens\":4}";
  char* response = nullptr;
  CHECK(vllm_chat(eng, request, &response) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(response == nullptr);
  CHECK(std::string(vllm_last_error()).find("no-such-parser") !=
        std::string::npos);
  vllm_engine_free(eng);
}

TEST_CASE("capi: an explicit registered tool_parser (hermes) serves normally") {
  vllm_engine* eng = MakeSyntheticChatEngine();
  REQUIRE(eng != nullptr);
  vllm::capi::SetEngineToolParser(eng, "hermes");

  const char* request =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
      "\"temperature\":0,\"max_tokens\":4}";
  char* response = nullptr;
  REQUIRE(vllm_chat(eng, request, &response) == VLLM_OK);
  REQUIRE(response != nullptr);
  CHECK(json::parse(response).at("object") == "chat.completion");
  vllm_string_free(response);
  vllm_engine_free(eng);
}

// ─── reasoning-parser selection (ABI v5) ─────────────────────────────────────
TEST_CASE("capi: an unknown reasoning parser is rejected on the first chat call") {
  vllm_engine* eng = MakeSyntheticChatEngine();
  REQUIRE(eng != nullptr);
  vllm::capi::SetEngineReasoningParser(eng, "no-such-reasoner");

  const char* request =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
      "\"temperature\":0,\"max_tokens\":2}";
  char* response = nullptr;
  CHECK(vllm_chat(eng, request, &response) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(response == nullptr);
  CHECK(std::string(vllm_last_error()).find("reasoning") != std::string::npos);
  vllm_engine_free(eng);
}

TEST_CASE("capi: explicit deepseek_r1 and the 'none' opt-out both serve chat") {
  for (const char* choice : {"deepseek_r1", "none"}) {
    CAPTURE(choice);
    vllm_engine* eng = MakeSyntheticChatEngine();
    REQUIRE(eng != nullptr);
    vllm::capi::SetEngineReasoningParser(eng, choice);
    const char* request =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
        "\"temperature\":0,\"max_tokens\":2}";
    char* response = nullptr;
    const vllm_status st = vllm_chat(eng, request, &response);
    const std::string last_err = vllm_last_error();
    CAPTURE(last_err);
    REQUIRE(st == VLLM_OK);
    REQUIRE(response != nullptr);
    vllm_string_free(response);
    vllm_engine_free(eng);
  }
}

// ─── (c3) ABI v9 engine-config fields ────────────────────────────────────────
// The v9 additions close the gap between what the bundled server can configure
// (examples/server/main.cpp flags) and what an embedder reaches through the C
// ABI: the chunked-prefill token budget, the scheduling policy, and the external
// KV connector (LMCache). Each is inert at its default, so a v8 caller that
// zero-fills the growth gets the byte-identical pre-v9 engine.
TEST_CASE("capi: v9 engine-config fields default to inert") {
  vllm_model_params mp = vllm_model_params_default();
  CHECK(mp.max_num_batched_tokens == 0);  // 0 => the per-arch default.
  CHECK(mp.scheduling_policy == nullptr);  // NULL => "fcfs".
  CHECK(mp.kv_transfer_config == nullptr);  // NULL => no connector.
  CHECK(mp.tokenizer_config_path == nullptr);  // NULL => <model_dir>/....
}

TEST_CASE("capi: max_num_batched_tokens passes the load gate") {
  // A positive budget is accepted and reaches model load (which fails on the
  // fake path with MODEL_LOAD, not INVALID_ARGUMENT).
  vllm_model_params p = vllm_model_params_default();
  p.model_path = "/nonexistent/vllm-cpp/model/dir";
  p.max_num_batched_tokens = 8192;
  vllm_engine* eng = nullptr;
  CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(eng == nullptr);
}

TEST_CASE("capi: scheduling_policy accepts the wire names and rejects others") {
  for (const char* policy : {"fcfs", "priority", "lpm"}) {
    vllm_model_params p = vllm_model_params_default();
    p.model_path = "/nonexistent/vllm-cpp/model/dir";
    p.scheduling_policy = policy;
    vllm_engine* eng = nullptr;
    CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
    CHECK(eng == nullptr);
  }

  // An unknown policy is rejected BEFORE any load attempt, so the caller gets
  // INVALID_ARGUMENT rather than a misleading "model load failed".
  vllm_model_params bad = vllm_model_params_default();
  bad.model_path = "/nonexistent/vllm-cpp/model/dir";
  bad.scheduling_policy = "round-robin";
  vllm_engine* eng = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&bad, &eng) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(eng == nullptr);
  CHECK(std::string(vllm_last_error()).find("round-robin") != std::string::npos);

  // Empty string is the documented "unset" spelling, same as NULL.
  vllm_model_params empty = vllm_model_params_default();
  empty.model_path = "/nonexistent/vllm-cpp/model/dir";
  empty.scheduling_policy = "";
  vllm_engine* eng2 = nullptr;
  CHECK(vllm_engine_load(&empty, &eng2) == VLLM_ERR_MODEL_LOAD);
}

// ─── (c4) ABI v10 jump-forward toggle ────────────────────────────────────────
// Grounds .agents/specs/sglang-enablement.md: the tri-state enable_jump_forward
// field defaults to 0 (byte-identical to ABI v9), validates its range at the
// load gate, and — set through EngineParams (the SAME field vllm_engine_load
// translates onto) — actually reaches the built engine
// (LoadedEngine::jump_forward_enabled()). LPM/scheduler policy is NOT retested
// here: it is the v9 string field .scheduling_policy, covered above.
TEST_CASE("capi: enable_jump_forward defaults to 0 and validates (ABI v10)") {
  vllm_model_params mp = vllm_model_params_default();
  CHECK(mp.enable_jump_forward == 0);  // env-resolved, default OFF.

  // Valid tri-states pass the gate (then fail at model load on the fake path).
  for (int v : {0, 1, 2}) {
    vllm_model_params p = vllm_model_params_default();
    p.model_path = "/nonexistent/vllm-cpp/model/dir";
    p.enable_jump_forward = v;
    vllm_engine* eng = nullptr;
    CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
    CHECK(eng == nullptr);
  }

  // An out-of-range value is rejected BEFORE any load attempt.
  vllm_model_params bad = vllm_model_params_default();
  bad.model_path = "/nonexistent/vllm-cpp/model/dir";
  bad.enable_jump_forward = 7;
  vllm_engine* eng = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&bad, &eng) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(eng == nullptr);
}

TEST_CASE("capi: enable_jump_forward=on reaches the engine; default is inert (ABI v10)") {
  // Resolution reads VT_ENABLE_JUMP_FORWARD as an override; clear it so this
  // test asserts the FIELD's effect, not an ambient env override.
  ::unsetenv("VT_ENABLE_JUMP_FORWARD");
  const HfConfig c = MakeConfig();

  // Default (nullopt): jump-forward resolves OFF — byte-identical to before v10.
  {
    EngineParams p = SyntheticParams();
    CHECK_FALSE(p.enable_jump_forward.has_value());
    LoadedEngine e(c, MakeWeights(c), BuildFixture(), p);
    CHECK(e.jump_forward_enabled() == false);
  }

  // enable_jump_forward = true reaches the built engine.
  {
    EngineParams p = SyntheticParams();
    p.enable_jump_forward = true;
    LoadedEngine e(c, MakeWeights(c), BuildFixture(), p);
    CHECK(e.jump_forward_enabled() == true);
  }

  // enable_jump_forward = false is explicit OFF.
  {
    EngineParams p = SyntheticParams();
    p.enable_jump_forward = false;
    LoadedEngine e(c, MakeWeights(c), BuildFixture(), p);
    CHECK(e.jump_forward_enabled() == false);
  }
}

TEST_CASE("capi: kv_transfer_config parses and validates the connector name") {
  // A well-formed config naming a REGISTERED connector passes the gate and
  // reaches model load.
  vllm_model_params p = vllm_model_params_default();
  p.model_path = "/nonexistent/vllm-cpp/model/dir";
  p.kv_transfer_config =
      "{\"kv_connector\":\"LMCacheConnector\",\"kv_role\":\"kv_both\","
      "\"kv_connector_extra_config\":{\"host\":\"127.0.0.1\",\"port\":65432}}";
  vllm_engine* eng = nullptr;
  CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(eng == nullptr);

  // Malformed JSON is a caller error, not a model-load error.
  vllm_model_params bad = vllm_model_params_default();
  bad.model_path = "/nonexistent/vllm-cpp/model/dir";
  bad.kv_transfer_config = "{not json";
  vllm_engine* eng2 = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&bad, &eng2) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(eng2 == nullptr);

  // An unregistered connector name is caught HERE (mirroring the server's
  // startup check) rather than surfacing as an opaque load failure.
  vllm_model_params unknown = vllm_model_params_default();
  unknown.model_path = "/nonexistent/vllm-cpp/model/dir";
  unknown.kv_transfer_config =
      "{\"kv_connector\":\"NoSuchConnector\",\"kv_role\":\"kv_both\"}";
  vllm_engine* eng3 = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&unknown, &eng3) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(eng3 == nullptr);
  CHECK(std::string(vllm_last_error()).find("NoSuchConnector") !=
        std::string::npos);
}

// A malformed speculative-config is a CALLER error. vllm.h has documented this
// as VLLM_ERR_INVALID_ARGUMENT since v6; before v9 the throw fell through to the
// generic std::exception catch and was reported as VLLM_ERR_MODEL_LOAD.
TEST_CASE("capi: malformed speculative_config is INVALID_ARGUMENT") {
  vllm_model_params p = vllm_model_params_default();
  p.model_path = "/nonexistent/vllm-cpp/model/dir";
  p.speculative_config = "{\"method\":\"no-such-method\"}";
  vllm_engine* eng = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(eng == nullptr);
}

// ─── version / abi ───────────────────────────────────────────────────────────
TEST_CASE("capi: version and abi-version are exposed") {
  CHECK(std::string(vllm_version()).size() > 0);
  CHECK(vllm_abi_version() == VLLM_ABI_VERSION);
  // The engine-config growth (max_num_batched_tokens / scheduling_policy /
  // kv_transfer_config) is ABI v9; the jump-forward toggle is ABI v10; the
  // transcription slice (vllm_transcribe) is ABI v11; the video-generation
  // slice (vllm_video_*) is ABI v12; the pre-tokenized completion entry
  // point (vllm_complete_tokens) is ABI v13; the device-selection field
  // (vllm_model_params.device) is ABI v14; the embeddings slice (vllm_embed /
  // vllm_embedding_result_free) is ABI v15; the KV-pool sizing knobs
  // (vllm_model_params.gpu_memory_utilization / kv_cache_memory_bytes,
  // ROAD-V1-MEM M1) are ABI v16. The >= pin is the one check that
  // can catch a WRONG bump: the == VLLM_ABI_VERSION assertions here and in
  // test_dlopen compare against the same macro and move with it (the #121
  // lesson: an == floor moves with the macro and proves nothing).
  CHECK(vllm_abi_version() >= 16);
}

// ─── ABI v16: KV-pool sizing knobs (ROAD-V1-MEM M1) ──────────────────────────
TEST_CASE("capi: v16 KV-sizing knobs default and round-trip") {
  vllm_model_params p = vllm_model_params_default();
  // num_blocks now defaults to AUTO (0), not the historical 256; the resolver
  // still falls back to 256, so the zero-struct behaviour is unchanged.
  CHECK(p.num_blocks == 0);
  // gpu_memory_utilization defaults to vLLM's 0.92; kv_cache_memory_bytes unset.
  CHECK(p.gpu_memory_utilization == doctest::Approx(0.92));
  CHECK(p.kv_cache_memory_bytes == 0);
  // The appended fields are writable POD (borrowed for the load call only).
  p.gpu_memory_utilization = 0.85;
  p.kv_cache_memory_bytes = int64_t{4} * 1024 * 1024 * 1024;
  CHECK(p.gpu_memory_utilization == doctest::Approx(0.85));
  CHECK(p.kv_cache_memory_bytes == int64_t{4} * 1024 * 1024 * 1024);
}

// ─── ABI v11: audio transcription (ARCH-ONE-SURFACE ROW 1) ───────────────────
// The FIRST real-checkpoint load gated through the PUBLIC ABI: vllm_engine_load
// on the committed tiny Parakeet fixtures (tests/vllm/models/fixtures/
// parakeet_e2e), then vllm_transcribe reproducing the transcript goldens the
// PRE-refactor example binary printed. Everything before this exercised
// vllm_engine_load's bad-path contract only (the severity note in
// .agents/specs/surface-coverage-2026-08-07.md § C-ABI capability coverage).

namespace {
std::string ParakeetFixture(const char* head) {
  return std::string(PARAKEET_E2E_FIXTURE_DIR) + "/" + head;
}
std::string ParakeetWav() {
  return std::string(PARAKEET_E2E_FIXTURE_DIR) + "/audio.wav";
}
}  // namespace

TEST_CASE("capi v11: vllm_transcribe reproduces the pre-refactor goldens") {
  struct Golden {
    const char* head;
    std::vector<int32_t> ids;
    const char* text;
  };
  const std::vector<Golden> goldens = {
      {"ctc", {3, 4, 3}, "atheat"},
      {"rnnt",
       {5, 5, 5, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6},
       "sss on on onssssss on on on on on on on on"},
  };
  for (const Golden& g : goldens) {
    CAPTURE(g.head);
    vllm_model_params mp = vllm_model_params_default();
    const std::string dir = ParakeetFixture(g.head);
    mp.model_path = dir.c_str();
    vllm_engine* eng = nullptr;
    REQUIRE(vllm_engine_load(&mp, &eng) == VLLM_OK);
    REQUIRE(eng != nullptr);

    vllm_transcription_params tp = vllm_transcription_params_default();
    const std::string wav = ParakeetWav();
    tp.audio_path = wav.c_str();
    vllm_transcription out;
    REQUIRE(vllm_transcribe(eng, &tp, &out) == VLLM_OK);
    REQUIRE(out.token_ids != nullptr);
    const std::vector<int32_t> ids(out.token_ids,
                                   out.token_ids + out.n_token_ids);
    CHECK(ids == g.ids);
    CHECK(out.has_text == 1);
    REQUIRE(out.text != nullptr);
    CHECK(std::string(out.text) == g.text);
    vllm_transcription_free(&out);
    CHECK(out.text == nullptr);      // zeroed after free
    CHECK(out.token_ids == nullptr);
    vllm_transcription_free(&out);   // double-free is a safe no-op
    vllm_engine_free(eng);
  }
}

TEST_CASE("capi v11: refuse-by-task in both directions") {
  // Transcription handle: every text entry point refuses with the actionable
  // message instead of crashing on the absent text stack.
  vllm_model_params mp = vllm_model_params_default();
  const std::string dir = ParakeetFixture("ctc");
  mp.model_path = dir.c_str();
  vllm_engine* asr = nullptr;
  REQUIRE(vllm_engine_load(&mp, &asr) == VLLM_OK);

  vllm_sampling_params sp = vllm_sampling_params_default();
  vllm_completion comp;
  CHECK(vllm_complete(asr, "hello", &sp, &comp) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("transcription-only") !=
        std::string::npos);
  char* chat_out = nullptr;
  CHECK(vllm_chat(asr, "{\"messages\":[]}", &chat_out) ==
        VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("vllm_transcribe") !=
        std::string::npos);
  vllm_engine_free(asr);

  // Text handle: vllm_transcribe refuses symmetrically.
  vllm_engine* text = MakeSyntheticEngine();
  REQUIRE(text != nullptr);
  vllm_transcription_params tp = vllm_transcription_params_default();
  const std::string wav = ParakeetWav();
  tp.audio_path = wav.c_str();
  vllm_transcription out;
  CHECK(vllm_transcribe(text, &tp, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("text-generation engine") !=
        std::string::npos);
  vllm_engine_free(text);
}

TEST_CASE("capi v11: vllm_transcribe argument contract") {
  vllm_model_params mp = vllm_model_params_default();
  const std::string dir = ParakeetFixture("ctc");
  mp.model_path = dir.c_str();
  vllm_engine* eng = nullptr;
  REQUIRE(vllm_engine_load(&mp, &eng) == VLLM_OK);

  vllm_transcription out;
  // Neither input selected.
  vllm_transcription_params none = vllm_transcription_params_default();
  CHECK(vllm_transcribe(eng, &none, &out) == VLLM_ERR_INVALID_ARGUMENT);
  // Both inputs selected.
  vllm_transcription_params both = vllm_transcription_params_default();
  const std::string wav = ParakeetWav();
  const float pcm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  both.audio_path = wav.c_str();
  both.pcm = pcm;
  both.n_samples = 4;
  both.sample_rate = 16000;
  CHECK(vllm_transcribe(eng, &both, &out) == VLLM_ERR_INVALID_ARGUMENT);
  // pcm without a sample rate.
  vllm_transcription_params bad_pcm = vllm_transcription_params_default();
  bad_pcm.pcm = pcm;
  bad_pcm.n_samples = 4;
  CHECK(vllm_transcribe(eng, &bad_pcm, &out) == VLLM_ERR_INVALID_ARGUMENT);
  // Null engine / params / out.
  CHECK(vllm_transcribe(nullptr, &none, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_transcribe(eng, nullptr, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_transcribe(eng, &none, nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  // Unreadable audio file -> runtime error naming the path problem.
  vllm_transcription_params missing = vllm_transcription_params_default();
  missing.audio_path = "/nonexistent/vllm-cpp/audio.wav";
  CHECK(vllm_transcribe(eng, &missing, &out) == VLLM_ERR_RUNTIME);
  CHECK(std::string(vllm_last_error()).size() > 0);
  // A raw-PCM arm through the ABI marshals and runs end to end.
  std::vector<float> silence(4000, 0.0f);
  vllm_transcription_params pcm_ok = vllm_transcription_params_default();
  pcm_ok.pcm = silence.data();
  pcm_ok.n_samples = static_cast<int64_t>(silence.size());
  pcm_ok.sample_rate = 16000;
  REQUIRE(vllm_transcribe(eng, &pcm_ok, &out) == VLLM_OK);
  CHECK(out.has_text == 1);
  vllm_transcription_free(&out);
  // NULL result free is a no-op.
  vllm_transcription_free(nullptr);
  vllm_engine_free(eng);
}

// ─── ABI v12: video+audio generation (ARCH-ONE-SURFACE ROW 2) ────────────────
// The video slice gated THROUGH the public ABI on the fold fixture: the same
// tiny checkpoint set + goldens the PRE-fold minimax-h3-gen binary rendered at
// the branch base (tests/vllm/models/fixtures/minimax_h3_video_fold). What the
// library-seam gate (test_minimax_h3_video_fold) proves for the C++ entry
// point, this proves for the C marshalling on top of it.

#include "../vllm/models/minimax_h3_video_fold_fixture.h"

namespace {

std::string ReadAllBytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open ", path);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// A pid-unique fixture + output workspace, torn down with the test.
struct VideoFoldWorkspace {
  std::string root, fixture;
  VideoFoldWorkspace() {
    static int counter = 0;
    root = "/tmp/vllm_capi_video_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++);
    std::filesystem::create_directories(root);
    fixture = root + "/fixture";
    minimax_h3_fold::WriteFoldFixture(fixture);
  }
  ~VideoFoldWorkspace() {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }
};

// Owned strings so the borrowed const char* fields stay alive per call.
struct VideoFixtureParams {
  std::string dit, vvae, vcfg, avae, acfg, embeds;
  vllm_video_model_params mp;
  explicit VideoFixtureParams(const std::string& dir)
      : dit(dir + "/dit.gguf"),
        vvae(dir + "/video_vae.safetensors"),
        vcfg(dir + "/video_vae_config.json"),
        avae(dir + "/audio_vae.safetensors"),
        acfg(dir + "/audio_vae_config.json"),
        embeds(dir + "/prompt_embeds.f32") {
    mp = vllm_video_model_params_default();
    mp.dit_path = dit.c_str();
    mp.video_vae_path = vvae.c_str();
    mp.video_vae_config_path = vcfg.c_str();
    mp.audio_vae_path = avae.c_str();
    mp.audio_vae_config_path = acfg.c_str();
    mp.prompt_embeds_path = embeds.c_str();
    mp.partition = "fl2va";
  }
};

}  // namespace

TEST_CASE("capi v12: the zero-value/default contract") {
  // Zero values must preserve behaviour: the _default() constructors return
  // fully zeroed structs (cpu, keep-quant, no paths, unseeded, and — via the
  // documented <=0 mapping — noise_aug 1.0). The golden e2e case below runs
  // on exactly these defaults (t2va), which pins every zero-value resolution
  // EXCEPT noise_aug: that mapping only fires on a keyframe render, which the
  // tiny fixture cannot express (the fl2va VAE-encoder half is the real
  // 128-channel geometry). It is pinned at the seam level instead
  // (MiniMaxH3VideoGenParams.noise_aug defaults to 1.0; the capi maps <=0
  // onto it) — an honest, disclosed edge of this contract test.
  const vllm_video_model_params mp = vllm_video_model_params_default();
  CHECK(mp.dit_path == nullptr);
  CHECK(mp.encoder_path == nullptr);
  CHECK(mp.tokenizer_path == nullptr);
  CHECK(mp.video_vae_path == nullptr);
  CHECK(mp.video_vae_config_path == nullptr);
  CHECK(mp.audio_vae_path == nullptr);
  CHECK(mp.audio_vae_config_path == nullptr);
  CHECK(mp.prompt_embeds_path == nullptr);
  CHECK(mp.partition == nullptr);
  CHECK(mp.device == 0);
  CHECK(mp.dequant_bf16 == 0);
  CHECK(mp.fp4_resident == 0);

  const vllm_video_params vp = vllm_video_params_default();
  CHECK(vp.prompt == nullptr);
  CHECK(vp.width == 0);
  CHECK(vp.height == 0);
  CHECK(vp.num_frames == 0);
  CHECK(vp.steps == 0);
  CHECK(vp.seed == 0);
  CHECK(vp.has_seed == 0);
  CHECK(vp.first_frame == nullptr);
  CHECK(vp.last_frame == nullptr);
  CHECK(vp.ref_image == nullptr);
  CHECK(vp.ref_video == nullptr);
  CHECK(vp.ref_audio == nullptr);
  CHECK(vp.noise_aug == 0.0f);  // <= 0 resolves engine-side to the 1.0 pin
  CHECK(vp.output_dir == nullptr);

  const vllm_video_mux_params mx = vllm_video_mux_params_default();
  CHECK(mx.frames == nullptr);
  CHECK(mx.audio_path == nullptr);
  CHECK(mx.output_path == nullptr);
  CHECK(mx.fps == 0);  // <= 0 resolves to the H3 default 24
  CHECK(mx.crf == 0);  // <= 0 resolves to the library default 18
}

TEST_CASE("capi v12: vllm_video_generate reproduces the pre-fold goldens") {
  VideoFoldWorkspace ws;
  VideoFixtureParams fp(ws.fixture);
  vllm_video_engine* eng = nullptr;
  REQUIRE_MESSAGE(vllm_video_engine_load(&fp.mp, &eng) == VLLM_OK,
                  vllm_last_error());
  REQUIRE(eng != nullptr);

  const std::string out_dir = ws.root + "/out";
  vllm_video_params vp = vllm_video_params_default();
  vp.num_frames = 5;
  vp.height = 32;
  vp.width = 32;
  vp.steps = 3;
  vp.output_dir = out_dir.c_str();

  vllm_video_result out;
  REQUIRE_MESSAGE(vllm_video_generate(eng, &vp, &out) == VLLM_OK,
                  vllm_last_error());
  CHECK(std::string(out.frame_dir) == out_dir);
  CHECK(std::string(out.audio_path) == out_dir + "/audio.wav");
  CHECK(out.frame_count == 8);
  CHECK(out.width == 32);
  CHECK(out.height == 32);
  CHECK(out.fps == 24);
  CHECK(out.sample_rate == 32000);

  const std::string golden_dir = MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR;
  for (int f = 0; f < 8; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06d.ppm", f);
    INFO("frame ", f);
    CHECK(ReadAllBytes(out_dir + name) == ReadAllBytes(golden_dir + name));
  }
  CHECK(ReadAllBytes(out_dir + "/audio.wav") ==
        ReadAllBytes(golden_dir + "/audio.wav"));

  // The mux argv is execvp-ready (NULL-terminated) and byte-matches the
  // pre-fold `minimax-h3-mux --print-only` capture.
  REQUIRE(out.mux_argv != nullptr);
  REQUIRE(out.mux_argc > 0);
  CHECK(out.mux_argv[out.mux_argc] == nullptr);
  std::string joined;
  for (int32_t i = 0; i < out.mux_argc; ++i) {
    joined += (i == 0 ? "" : " ") + std::string(out.mux_argv[i]);
  }
  std::string golden_argv = ReadAllBytes(golden_dir + "/golden_mux_argv.txt");
  while (!golden_argv.empty() &&
         (golden_argv.back() == '\n' || golden_argv.back() == '\r')) {
    golden_argv.pop_back();
  }
  size_t pos = 0;
  while ((pos = golden_argv.find("W/", pos)) != std::string::npos) {
    golden_argv.replace(pos, 1, out_dir);
    pos += out_dir.size() + 1;
  }
  CHECK(joined == golden_argv);

  vllm_video_result_free(&out);
  CHECK(out.frame_dir == nullptr);
  CHECK(out.mux_argv == nullptr);
  vllm_video_result_free(&out);  // double-free is a safe no-op
  vllm_video_result_free(nullptr);
  vllm_video_engine_free(eng);
}

TEST_CASE("capi v12: text and video engines refuse each other's checkpoints") {
  // Direction 1: vllm_video_engine_load on a TEXT/transcription checkpoint
  // directory fails LOUDLY, naming vllm_engine_load as the right entry point.
  VideoFoldWorkspace ws;
  const std::string text_dir = ParakeetFixture("ctc");
  vllm_video_model_params mp = vllm_video_model_params_default();
  mp.dit_path = text_dir.c_str();
  vllm_video_engine* eng = reinterpret_cast<vllm_video_engine*>(0x1);
  CHECK(vllm_video_engine_load(&mp, &eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(eng == nullptr);
  CHECK(std::string(vllm_last_error()).find("vllm_engine_load") !=
        std::string::npos);

  // Direction 2: vllm_engine_load on the H3 checkpoint directory keeps
  // failing EXACTLY as it did at v11 (captured at the branch base): status 2
  // with the missing-config.json cause — the fold changed nothing here.
  vllm_model_params tmp = vllm_model_params_default();
  tmp.model_path = ws.fixture.c_str();
  vllm_engine* text_eng = nullptr;
  CHECK(vllm_engine_load(&tmp, &text_eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(text_eng == nullptr);
  CHECK(std::string(vllm_last_error()).find("config.json") != std::string::npos);
}

TEST_CASE("capi v12: video argument contract") {
  VideoFoldWorkspace ws;

  // Null/missing load arguments.
  vllm_video_engine* eng = nullptr;
  CHECK(vllm_video_engine_load(nullptr, &eng) == VLLM_ERR_INVALID_ARGUMENT);
  vllm_video_model_params empty = vllm_video_model_params_default();
  CHECK(vllm_video_engine_load(&empty, &eng) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_video_engine_load(&empty, nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  // A missing VAE is a load error with the cause named.
  VideoFixtureParams no_vae(ws.fixture);
  no_vae.mp.video_vae_path = nullptr;
  CHECK(vllm_video_engine_load(&no_vae.mp, &eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(std::string(vllm_last_error()).find("video_vae") != std::string::npos);

  // Generate-side contract on a good engine.
  VideoFixtureParams fp(ws.fixture);
  REQUIRE(vllm_video_engine_load(&fp.mp, &eng) == VLLM_OK);
  vllm_video_result out;
  vllm_video_params vp = vllm_video_params_default();
  CHECK(vllm_video_generate(nullptr, &vp, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_video_generate(eng, nullptr, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_video_generate(eng, &vp, nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  // output_dir is required.
  CHECK(vllm_video_generate(eng, &vp, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("output_dir") != std::string::npos);
  // An illegal reference combination surfaces as a runtime refusal.
  const std::string out_dir = ws.root + "/out";
  vp.output_dir = out_dir.c_str();
  vp.first_frame = "/nonexistent.ppm";
  vp.ref_video = ws.fixture.c_str();
  CHECK(vllm_video_generate(eng, &vp, &out) == VLLM_ERR_RUNTIME);
  CHECK(std::string(vllm_last_error()).find("exclusive") != std::string::npos);
  vllm_video_engine_free(eng);
  vllm_video_engine_free(nullptr);  // no-op

  // The standalone mux composer: contract + golden byte-match.
  char** argv = nullptr;
  int32_t argc = 0;
  CHECK(vllm_video_mux_argv(nullptr, &argv, &argc) == VLLM_ERR_INVALID_ARGUMENT);
  vllm_video_mux_params mx = vllm_video_mux_params_default();
  CHECK(vllm_video_mux_argv(&mx, &argv, &argc) == VLLM_ERR_INVALID_ARGUMENT);
  mx.frames = "frames_%06d.ppm";
  mx.output_path = "silent.mp4";
  REQUIRE(vllm_video_mux_argv(&mx, &argv, &argc) == VLLM_OK);
  REQUIRE(argv != nullptr);
  REQUIRE(argc > 0);
  CHECK(argv[argc] == nullptr);  // execvp-ready
  std::string joined;
  for (int32_t i = 0; i < argc; ++i) joined += (i == 0 ? "" : " ") + std::string(argv[i]);
  std::string golden = ReadAllBytes(std::string(MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR) +
                                    "/golden_mux_argv_silent.txt");
  while (!golden.empty() && (golden.back() == '\n' || golden.back() == '\r')) {
    golden.pop_back();
  }
  CHECK(joined == golden);
  vllm_video_mux_argv_free(argv, argc);
  vllm_video_mux_argv_free(nullptr, 3);  // no-op
}


// ─── ABI v14: explicit device selection (ARCH-ONE-SURFACE ROW 8) ─────────────
// The device knob: 0=auto (the byte-identical accelerator-first probe), 1=cpu,
// 2=cuda — the vLLM DeviceConfig.device names (vllm/config/device.py:13). The
// pure policy matrix (explicit cpu beats a registered accelerator; explicit
// cuda never falls back) is gated in test_loaded_engine_dense.cpp; here the
// C-ABI wire contract and the params->EngineParams->SelectQueue plumb.

namespace {
// A synthetic engine over an explicit EngineParams device selection, sharing
// MakeSyntheticEngine's stack.
vllm_engine* MakeSyntheticEngineWithDevice(vllm::Device device) {
  const HfConfig c = MakeConfig();
  EngineParams params = SyntheticParams();
  params.device = device;
  auto loaded = std::make_unique<LoadedEngine>(c, MakeWeights(c), BuildFixture(),
                                               params);
  return vllm::capi::MakeEngineHandle(std::move(loaded));
}
}  // namespace

TEST_CASE("capi v14: the device zero-value/default contract") {
  // The default is 0 == auto — a zero-initialized struct and
  // vllm_model_params_default() agree, so a pre-v14 caller's zero-filled
  // growth keeps the accelerator-first probe engine byte-identical.
  const vllm_model_params def = vllm_model_params_default();
  CHECK(def.device == 0);
  vllm_model_params zeroed;
  std::memset(&zeroed, 0, sizeof(zeroed));
  CHECK(zeroed.device == def.device);

  // The zero value maps to AUTO, not to an explicit device: with device left
  // at the default and a bogus path, the load must report the PATH (the auto
  // arm defers to the probe and never resolves an explicit device up front).
  // A mutation that maps 0 to an explicit cuda would surface the device error
  // here instead — on the CPU tier that is the distinguishable half; 0-as-
  // explicit-cpu is behaviorally identical on this tier and is pinned by the
  // pure policy matrix in test_loaded_engine_dense.cpp plus the CUDA-build
  // residual named in the spec.
  vllm_model_params bogus = vllm_model_params_default();
  bogus.model_path = "/nonexistent/vllm-cpp/model/dir";
  vllm_engine* probe_eng = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&bogus, &probe_eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(probe_eng == nullptr);
  CHECK(std::string(vllm_last_error()).find("not a directory") !=
        std::string::npos);

  // Behavior: on a CPU-only process the auto probe resolves CPU, so an
  // explicit-cpu engine must generate EXACTLY what the default (auto) engine
  // generates. (Guarded: on an accelerator build auto legitimately selects the
  // accelerator, and byte-equality with a CPU run is not the contract.)
  if (!vllm::platforms::HasPlatform(vt::DeviceType::kCUDA)) {
    vllm_engine* auto_eng = MakeSyntheticEngine();  // device unset == kAuto
    vllm_engine* cpu_eng = MakeSyntheticEngineWithDevice(vllm::Device::kCPU);
    REQUIRE(auto_eng != nullptr);
    REQUIRE(cpu_eng != nullptr);
    vllm_sampling_params sp = GreedyParams(6);
    vllm_completion a{};
    vllm_completion b{};
    REQUIRE(vllm_complete(auto_eng, "hello world", &sp, &a) == VLLM_OK);
    REQUIRE(vllm_complete(cpu_eng, "hello world", &sp, &b) == VLLM_OK);
    CHECK(std::string(a.text) == std::string(b.text));
    CHECK(a.completion_tokens == b.completion_tokens);
    vllm_completion_free(&a);
    vllm_completion_free(&b);
    vllm_engine_free(auto_eng);
    vllm_engine_free(cpu_eng);
  }
}

TEST_CASE("capi v14: device range validates before any load work") {
  // An out-of-range device is a CALLER error caught before FromModelDir — the
  // bogus path must NOT be what fails here.
  for (const int32_t bad : {static_cast<int32_t>(3), static_cast<int32_t>(-1),
                            static_cast<int32_t>(7)}) {
    vllm_model_params p = vllm_model_params_default();
    p.model_path = "/nonexistent/vllm-cpp/model/dir";
    p.device = bad;
    vllm_engine* eng = reinterpret_cast<vllm_engine*>(0x1);
    CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_INVALID_ARGUMENT);
    CHECK(eng == nullptr);
    CHECK(std::string(vllm_last_error())
              .find("device must be 0 (auto), 1 (cpu), or 2 (cuda)") !=
          std::string::npos);
  }
}

TEST_CASE("capi v14: explicit cuda on a CUDA-less process fails LOUD (plumb pin)") {
  if (vllm::platforms::HasPlatform(vt::DeviceType::kCUDA)) {
    return;  // CUDA build/box: the explicit-cuda arm resolves; nothing to pin.
  }
  // device=2 with a bogus path: the DEVICE error must surface, not the path
  // error — FromModelDir resolves an explicit device BEFORE any I/O, so this
  // pins the whole capi -> EngineParams -> FromModelDir plumb (an implementation
  // that drops params->device would report the path instead and go RED here).
  vllm_model_params p = vllm_model_params_default();
  p.model_path = "/nonexistent/vllm-cpp/model/dir";
  p.device = 2;  // cuda
  vllm_engine* eng = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(eng == nullptr);
  CHECK(std::string(vllm_last_error())
            .find("device 'cuda' was requested but no CUDA platform") !=
        std::string::npos);

  // device=1 (cpu) on the same bogus path proceeds to the path and reports IT —
  // the plumb forwards the field's VALUE, not a constant.
  p.device = 1;  // cpu
  eng = reinterpret_cast<vllm_engine*>(0x1);
  CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(eng == nullptr);
  CHECK(std::string(vllm_last_error()).find("not a directory") !=
        std::string::npos);
}

TEST_CASE("capi v14: explicit cpu forces the CPU queue at the EngineParams seam") {
  // The observable queue seam: an engine constructed with device=kCPU runs its
  // runner on the CPU device. On the CPU tier this is trivially true of auto as
  // well; on a CUDA build it is the force-CPU pin (the runner would otherwise
  // sit on the CUDA queue). The CPU-tier statement of "cpu beats a registered
  // accelerator" is the pure matrix in test_loaded_engine_dense.cpp.
  const HfConfig c = MakeConfig();
  EngineParams params = SyntheticParams();
  params.device = vllm::Device::kCPU;
  LoadedEngine loaded(c, MakeWeights(c), BuildFixture(), params);
  CHECK(loaded.runner().device().type == vt::DeviceType::kCPU);

  // And the ctor-path plumb (LoadedEngine's own SelectQueue call, the arm the
  // GGUF/MoE branches take): explicit cuda on a CUDA-less process must throw
  // the pinned message out of construction — never silently build on CPU.
  if (!vllm::platforms::HasPlatform(vt::DeviceType::kCUDA)) {
    EngineParams cuda_params = SyntheticParams();
    cuda_params.device = vllm::Device::kNamedPlatform;
    CHECK_THROWS_WITH_AS(
        LoadedEngine(MakeConfig(), MakeWeights(c), BuildFixture(), cuda_params),
        doctest::Contains("device 'cuda' was requested but no CUDA platform"),
        std::runtime_error);
  }
}

// ─── ABI v15: embeddings (ARCH-ONE-SURFACE ROW 6) ────────────────────────────
// The embeddings slice gated THROUGH the public ABI on the committed tiny
// LlamaModel fixture (tests/vllm/models/fixtures/llama_embed_e2e): a REAL
// checkpoint-directory load through vllm_engine_load, then vllm_embed through
// the SAME registry forward + PoolingRunner engine step the fold gate
// (test_llama_embedding_fold) anchors. Plus the argument contract and the
// refuse-by-task pins in BOTH directions (the v11 precedent applied to the
// pooling task).

namespace {
std::string LlamaEmbedFixture() { return std::string(LLAMA_EMBED_FIXTURE_DIR); }
}  // namespace

TEST_CASE("capi v15: vllm_embed embeds through the public ABI (fixture load)") {
  vllm_model_params mp = vllm_model_params_default();
  const std::string dir = LlamaEmbedFixture();
  mp.model_path = dir.c_str();
  vllm_engine* eng = nullptr;
  REQUIRE(vllm_engine_load(&mp, &eng) == VLLM_OK);
  REQUIRE(eng != nullptr);

  const char* texts[2] = {"the quick brown fox", "the lazy dog"};
  vllm_embedding_result out;
  REQUIRE(vllm_embed(eng, texts, 2, &out) == VLLM_OK);
  REQUIRE(out.values != nullptr);
  CHECK(out.n_embeddings == 2);
  CHECK(out.dim == 64);  // the fixture's hidden_size
  CHECK(out.prompt_tokens > 0);
  // Each embedding is unit-L2 (the pooling normalize ran) and the two DIFFER
  // (different prompts pool different last-token hiddens).
  double delta = 0.0;
  for (int32_t r = 0; r < out.n_embeddings; ++r) {
    double l2 = 0.0;
    for (int32_t c = 0; c < out.dim; ++c) {
      const double v = out.values[r * out.dim + c];
      l2 += v * v;
    }
    CHECK(std::sqrt(l2) == doctest::Approx(1.0).epsilon(1e-5));
  }
  for (int32_t c = 0; c < out.dim; ++c) {
    delta += std::abs(static_cast<double>(out.values[c]) -
                      static_cast<double>(out.values[out.dim + c]));
  }
  CHECK(delta > 1e-3);

  vllm_embedding_result_free(&out);
  CHECK(out.values == nullptr);  // zeroed after free
  CHECK(out.n_embeddings == 0);
  vllm_embedding_result_free(&out);  // double-free is a safe no-op
  vllm_embedding_result_free(nullptr);
  vllm_engine_free(eng);
}

TEST_CASE("capi v15: refuse-by-task in both directions (pooling vs text)") {
  // Pooling handle: every text entry point refuses with the actionable message
  // instead of driving generation over hidden states.
  vllm_model_params mp = vllm_model_params_default();
  const std::string dir = LlamaEmbedFixture();
  mp.model_path = dir.c_str();
  vllm_engine* emb = nullptr;
  REQUIRE(vllm_engine_load(&mp, &emb) == VLLM_OK);

  vllm_sampling_params sp = vllm_sampling_params_default();
  vllm_completion comp;
  CHECK(vllm_complete(emb, "hello", &sp, &comp) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("pooling (embedding)") !=
        std::string::npos);
  CHECK(std::string(vllm_last_error()).find("vllm_embed") != std::string::npos);
  char* chat_out = nullptr;
  CHECK(vllm_chat(emb, "{\"messages\":[]}", &chat_out) ==
        VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("vllm_embed") != std::string::npos);
  {
    const int32_t prompt_ids[1] = {0};
    int32_t out_ids[4];
    int32_t n_out = 0;
    CHECK(vllm_complete_tokens(emb, prompt_ids, 1, &sp, out_ids, 4, &n_out,
                               nullptr) == VLLM_ERR_INVALID_ARGUMENT);
    CHECK(std::string(vllm_last_error()).find("vllm_embed") !=
          std::string::npos);
  }
  vllm_engine_free(emb);

  // Text handle: vllm_embed refuses symmetrically, naming the text entry
  // points.
  vllm_engine* text = MakeSyntheticEngine();
  REQUIRE(text != nullptr);
  const char* texts[1] = {"hello"};
  vllm_embedding_result out;
  CHECK(vllm_embed(text, texts, 1, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("text-generation") !=
        std::string::npos);
  CHECK(std::string(vllm_last_error()).find("vllm_complete") !=
        std::string::npos);
  vllm_engine_free(text);
}

TEST_CASE("capi v15: vllm_embed argument contract") {
  vllm_model_params mp = vllm_model_params_default();
  const std::string dir = LlamaEmbedFixture();
  mp.model_path = dir.c_str();
  vllm_engine* eng = nullptr;
  REQUIRE(vllm_engine_load(&mp, &eng) == VLLM_OK);

  const char* texts[2] = {"the fox", nullptr};
  vllm_embedding_result out;
  // Null engine / texts / out; non-positive n_texts; a NULL texts entry.
  CHECK(vllm_embed(nullptr, texts, 1, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_embed(eng, nullptr, 1, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_embed(eng, texts, 1, nullptr) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_embed(eng, texts, 0, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_embed(eng, texts, -3, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(vllm_embed(eng, texts, 2, &out) == VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("texts[1]") != std::string::npos);
  // On every refused call *out stays zeroed.
  CHECK(out.values == nullptr);
  CHECK(out.n_embeddings == 0);
  vllm_engine_free(eng);
}
