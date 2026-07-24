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
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/engine_handle.h"
#include "vllm/entrypoints/model_loader.h"
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

// ─── version / abi ───────────────────────────────────────────────────────────
TEST_CASE("capi: version and abi-version are exposed") {
  CHECK(std::string(vllm_version()).size() > 0);
  CHECK(vllm_abi_version() == VLLM_ABI_VERSION);
}
