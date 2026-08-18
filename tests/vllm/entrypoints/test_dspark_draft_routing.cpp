// `SPEC-DSPARK-QWEN3-ROUTING` (issue #1193) — REACHABILITY of the DSpark draft
// classification, not the predicate that performs it.
//
// WHY THIS FILE EXISTS SEPARATELY. `tests/vllm/config/test_speculative_dspark.cpp`
// proves `SpeculativeConfig::IsDsparkDraft` and
// `SpeculativeConfig::ResolveDsparkArchitecture` answer correctly. Before this row
// every reference to `IsDsparkDraft` outside its own header was in that file:
// `ResolveSpecConfig` branched on the CLI method string alone, so nothing shipped
// classified a draft by its own config, and a `DSparkDraftModel` checkpoint loaded
// as a Qwen3 DSpark draft by OMISSION rather than by decision
// (.agents/specs/dspark-qwen3-routing.md §0). AGENTS.md `## Nothing lands dead`
// requires the smallest failing test to enter through a production entry point, and
// a unit test on the predicate is exactly the unit test that rule excludes.
//
// THE PRODUCTION ENTRY POINT is the `LoadedEngine` CONSTRUCTOR, which resolves the
// speculative config the caller passed in `EngineParams`
// (`resolved_spec_config_(ResolveSpecConfig(params, config_))`,
// src/vllm/entrypoints/model_loader.cpp). `ResolveSpecConfig` is private, and a
// test that called it would prove the function works rather than that anything
// reaches it — the same argument `test_loaded_engine_dense.cpp` makes for
// `ResolveNumBlocks`. The chain under test is therefore
// `EngineParams::speculative_config` -> LoadedEngine ctor -> ResolveSpecConfig ->
// IsDsparkDraft / ResolveDsparkArchitecture, entered with a real draft DIRECTORY on
// disk, which is what the classification reads. The synthetic dense model is the
// TARGET and is irrelevant to the decision: the dspark branch reads the draft's
// config.json and never the target's.
//
// THE REACHABILITY MUTATION for this row deletes the classification block in
// `ResolveSpecConfig` and requires this suite RED.
#include "vllm/entrypoints/model_loader.h"

#include <doctest/doctest.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "support/process_id.h"
#include "vllm/config/speculative.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
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
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vt::DType;

namespace fs = std::filesystem;

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

// ─── The draft checkpoints the classification reads ─────────────────────────

// A draft checkpoint directory carrying just `config.json`. The classification
// reads nothing else, and no draft weight is ever opened on this path.
class DraftDir {
 public:
  DraftDir(const std::string& leaf, const std::string& config_json) {
    // The shared portable process id, so two concurrent runs cannot collide and
    // the file does not reintroduce the ::getpid() include class this tree just
    // fixed (tests/support/process_id.h).
    dir_ = fs::temp_directory_path() /
           ("vllm-cpp-" + leaf + "-" + std::to_string(vllm_test::ProcessId()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
    std::ofstream(dir_ / "config.json") << config_json;
  }
  ~DraftDir() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  DraftDir(const DraftDir&) = delete;
  DraftDir& operator=(const DraftDir&) = delete;
  std::string path() const { return dir_.string(); }

 private:
  fs::path dir_;
};

// The instrument's own precondition. `IsDsparkDraft` answers TRUE for any model
// id whose lowercase form contains "dspark", so a temporary directory that
// happened to carry the substring would make the ARCHITECTURE arm unobservable —
// the mute-switch shape .agents/specs/dspark-qwen3-routing.md §5 names as this
// test set's one trap.
bool CarriesDsparkSubstring(const std::string& path) {
  std::string lowered = path;
  for (char& c : lowered) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return lowered.find("dspark") != std::string::npos;
}

// RadixArk/Qwen3.8-27B-DSpark @ 85ef153be924f17ce4bf62726954eeaa4a73e854, the
// checkpoint .agents/specs/dspark-qwen3-routing.md §7 R1 pins, reduced to the two
// keys the classification reads. `architectures` is the post-vllm#52197 pair.
constexpr const char* kQwen3DraftModelConfig =
    R"({"architectures":["DSparkDraftModel"],"model_type":"qwen3",)"
    R"("block_size":7,"num_hidden_layers":5})";

// The same architecture string with the DeepSeek-V4 model_type. Upstream's
// fallback (vllm/config/speculative.py:934-944 @ 555967922) rewrites exactly this
// config into the DeepSeek-V4 DSpark lane, which this engine does not implement.
constexpr const char* kDeepseekV4DraftConfig =
    R"({"architectures":["DSparkDraftModel"],"model_type":"deepseek_v4",)"
    R"("num_hidden_layers":5})";

// A draft that DECLARES no architecture at all. Upstream reads the key off a
// HuggingFace ModelConfig, where an absent key is `[]`, and its catch-all would
// send that to DeepSeek-V4. This engine does not classify on the ABSENCE of
// evidence, because the native `deepseek-ai/dspark_qwen3_*_block7` layouts have
// not been read here to confirm they declare it, and refusing them would break a
// lane that loads today.
constexpr const char* kNoArchitecturesDraftConfig =
    R"({"model_type":"qwen3","block_size":7,"num_hidden_layers":5})";

// The native name, routed since SPEC-DSPARK W5. The regression guard: the
// classification must not refuse the lane that already loads.
constexpr const char* kNativeQwen3DSparkConfig =
    R"({"architectures":["Qwen3DSparkModel"],"model_type":"qwen3",)"
    R"("block_size":7,"num_hidden_layers":5})";

// The GEMMA4 name, with its own model_type. This is BRANCH 3 of
// `SpeculativeConfig::ResolveDsparkArchitecture`: upstream leaves
// `Gemma4DSparkModel` in place and normalizes only its keys, because upstream has
// a Gemma4 DSpark class to dispatch to; this engine has one DSpark draft lane, so
// the branch COLLAPSES onto "Qwen3DSparkModel". That collapse is the SECOND
// tracked divergence recorded in .agents/specs/dspark-qwen3-routing.md
// `## Outcome`, and until this case it was exercised only by the hand-called unit
// case in tests/vllm/config/test_speculative_dspark.cpp — the shape AGENTS.md
// `## Nothing lands dead` excludes, because it proves the function works rather
// than that anything reaches it.
constexpr const char* kGemma4DSparkConfig =
    R"({"architectures":["Gemma4DSparkModel"],"model_type":"gemma4",)"
    R"("block_size":7,"num_hidden_layers":5})";

// The SPECULATORS layout, the second published shape `LoadDsparkDraft` accepts.
// It declares NO top-level `architectures` at all — `speculators_model_type`
// identifies it, and `Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig` writes
// `["Qwen3DSparkModel"]` while translating (algos.py update_dspark). So it is the
// one layout for which the no-architecture narrowing above and the classification
// disagree, and which of the two answers it gets depends on the translation
// running FIRST. Keys reduced to what the translation asserts on.
constexpr const char* kSpeculatorsDsparkConfig =
    R"({"speculators_model_type":"dspark","aux_hidden_state_layer_ids":[5,17,29],)"
    R"("speculators_config":{"proposal_methods":[{"speculative_tokens":7}]},)"
    R"("block_size":7,"mask_token_id":151665,)"
    R"("transformer_layer_config":{"model_type":"qwen3","num_hidden_layers":5}})";

EngineParams DsparkParams(const std::string& draft_dir, int k) {
  EngineParams params;
  vllm::SpeculativeConfig cli;
  cli.method = "dspark";
  cli.draft_model_path = draft_dir;
  cli.num_speculative_tokens = k;
  params.speculative_config = cli;
  return params;
}

// Build one engine through the public constructor and return the refusal text it
// produced, or "" when construction got PAST the classification. Returning the
// empty string rather than asserting keeps the RED legible: before this row the
// loader classified nothing, so the refusal is ABSENT, not wrong.
std::string RefusalForDraft(const std::string& draft_dir) {
  const HfConfig c = MakeDenseConfig();
  try {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(),
                     DsparkParams(draft_dir, 7));
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST_CASE("loader refuses a DeepSeek-V4 DSpark draft by name") {
  const DraftDir draft("draft-v4", kDeepseekV4DraftConfig);
  REQUIRE_FALSE(CarriesDsparkSubstring(draft.path()));

  const std::string message = RefusalForDraft(draft.path());
  REQUIRE_FALSE(message.empty());  // RED before this row: nothing refuses.
  // AGENTS.md: an unimplemented arm refuses with a message that NAMES the missing
  // part. "DeepSeek-V4" is that name; without it the user is sent hunting.
  CHECK(message.find("DeepSeek-V4") != std::string::npos);
  CHECK(message.find("not implemented") != std::string::npos);
  // The ARM, not just the lane. Both refusals on this path name DeepSeek-V4 and
  // say "not implemented", so those two substrings alone cannot tell them apart —
  // and a test that cannot tell them apart cannot see the identity check
  // disappear. This case must reach `SpeculativeConfig::IsDsparkDraft`'s refusal,
  // whose wording is its own.
  CHECK(message.find("does not identify as") != std::string::npos);
}

TEST_CASE("loader refuses a DeepSeek-V4 draft whose id DOES carry \"dspark\"") {
  // The other arm of the same refusal. Here the model id alone makes
  // `IsDsparkDraft` answer true (the "dspark" substring,
  // vllm/config/speculative.py:882), so the DeepSeek-V4 lane has to be caught by
  // the ARCHITECTURE normalization rather than by the identity check. Both arms
  // must name the same missing part.
  const DraftDir draft("dspark-v4", kDeepseekV4DraftConfig);
  REQUIRE(CarriesDsparkSubstring(draft.path()));

  const std::string message = RefusalForDraft(draft.path());
  REQUIRE_FALSE(message.empty());  // RED before this row: nothing refuses.
  CHECK(message.find("DeepSeek-V4") != std::string::npos);
  CHECK(message.find("not implemented") != std::string::npos);
  // The other arm's own wording, for the reason given in the case above: this
  // draft must be refused by `SpeculativeConfig::ResolveDsparkArchitecture`,
  // having PASSED the identity check on its model id.
  CHECK(message.find("routes to the DeepSeek-V4 DSpark lane") != std::string::npos);
}

TEST_CASE("loader admits a DSparkDraftModel + qwen3 draft") {
  // The routing this row exists for, and the case that goes RED if the
  // `DSparkDraftModel` + "qwen3" pair is reverted out of `IsDsparkDraft` (gate
  // G3). The model id deliberately carries no "dspark" substring, so ONLY the
  // architecture arm can answer here.
  const DraftDir draft("draft-qwen3", kQwen3DraftModelConfig);
  REQUIRE_FALSE(CarriesDsparkSubstring(draft.path()));

  CHECK(RefusalForDraft(draft.path()).empty());
}

TEST_CASE("loader still admits the native Qwen3DSparkModel draft") {
  const DraftDir draft("draft-native", kNativeQwen3DSparkConfig);
  CHECK(RefusalForDraft(draft.path()).empty());
}

TEST_CASE("loader admits a Gemma4DSparkModel draft through the collapsed lane") {
  // Branch 3 of `ResolveDsparkArchitecture`, reached from the production entry
  // point rather than by hand. Two things are asserted at once: the Gemma4 name
  // is ADMITTED (it must not fall into the DeepSeek-V4 refusal, which is what a
  // `!has_qwen3 && !has_gemma4` guard reverted to `!has_qwen3` would do), and it
  // is admitted onto the ONE lane this engine implements. The directory name
  // carries no "dspark", so `IsDsparkDraft`'s model-id arm cannot answer for it
  // and only the architecture arm can — the mute switch this file's
  // `CarriesDsparkSubstring` precondition exists to keep shut.
  const DraftDir draft("draft-gemma4", kGemma4DSparkConfig);
  REQUIRE_FALSE(CarriesDsparkSubstring(draft.path()));

  CHECK(RefusalForDraft(draft.path()).empty());
}

TEST_CASE("loader does not classify a draft that declares no architecture") {
  // The narrowing recorded beside `ReadDsparkDraftIdentity`. The directory name
  // carries "dspark", exactly as the published native drafts' repo ids do, so
  // `IsDsparkDraft` answers true and the ONLY thing standing between this draft
  // and the DeepSeek-V4 refusal is the empty-list guard.
  const DraftDir draft("dspark-noarch", kNoArchitecturesDraftConfig);
  REQUIRE(CarriesDsparkSubstring(draft.path()));

  CHECK(RefusalForDraft(draft.path()).empty());
}

TEST_CASE("loader classifies a Speculators-layout DSpark draft as the Qwen3 lane") {
  // The gap this case closes: nothing drove the second published layout through
  // the new classification. It matters because that layout reaches
  // `ReadDsparkDraftIdentity` with no `architectures` key of its own, so the
  // empty-list narrowing would skip it — unless the speculators translation runs
  // before the key is read, which is what
  // `src/vllm/entrypoints/model_loader.cpp::ReadDsparkDraftIdentity` does. The
  // directory name deliberately carries no "dspark", so the ONLY thing that can
  // admit this draft is the translated architecture.
  const DraftDir draft("draft-spec-layout", kSpeculatorsDsparkConfig);
  REQUIRE_FALSE(CarriesDsparkSubstring(draft.path()));

  CHECK(RefusalForDraft(draft.path()).empty());
}
