// Qwen4-Exp W1 scaffold (MODEL-MM-QWEN4-EXP, #1981).
//
// Everything here drives the PRODUCTION entry point:
// `LoadHfConfig` -> `ModelRegistry::Resolve` -> `factory->parse_config`, and
// the refusals through `factory->load_weights` / `->forward` / `->make_kv_cache`.
// A case that built `Qwen4ExpParams` by hand would prove the struct parses and
// NOT that anything reaches it, which AGENTS.md "Nothing lands dead" refuses to
// accept as evidence.
//
// ORACLE: transformers **5.16.0**, the lane pin accepted for this row. vLLM
// implements `qwen4_exp` at no revision, so there is nothing to mirror on this
// surface. Values come from the committed fixture, which is the published
// `Qwen/Qwen3.8-Flash-Next` `config.json` verbatim.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "doctest/doctest.h"
#include "nlohmann/json.hpp"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/transformers_utils/hf_config.h"

using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::ModelRegistry;
using vllm::ParseQwen4ExpParams;
using vllm::Qwen4ExpLayerKind;
using vllm::Qwen4ExpParams;

namespace {

const char* FixtureDir() {
#ifdef QWEN4_EXP_CKPT_FIXTURE_DIR
  return QWEN4_EXP_CKPT_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/qwen4_exp";
#endif
}

// Unique to THIS PROCESS, not merely to this object. A bare `static int
// counter` makes two concurrent runs of this binary share a path and delete
// each other's directory (#1860); the failure reads as NO RESULT rather than
// as a failure, so it is worth the six lines. No `getpid()`, which MSVC spells
// differently.
std::filesystem::path UniqueTempDir(const std::string& stem) {
  static const std::string kToken = [] {
    std::random_device rd;
    std::ostringstream os;
    os << std::hex << rd() << "_"
       << std::chrono::steady_clock::now().time_since_epoch().count();
    return os.str();
  }();
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         (stem + kToken + "_" + std::to_string(counter++));
}

class TempConfig {
 public:
  explicit TempConfig(const nlohmann::json& doc) {
    dir_ = UniqueTempDir("qwen4_exp_cfg_");
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "config.json") << doc.dump();
  }
  ~TempConfig() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string path() const { return (dir_ / "config.json").string(); }

 private:
  std::filesystem::path dir_;
};

nlohmann::json FixtureDoc() {
  std::ifstream in(std::string(FixtureDir()) + "/config.json");
  REQUIRE_MESSAGE(in.good(), "fixture config.json missing under " << FixtureDir());
  nlohmann::json doc;
  in >> doc;
  return doc;
}

// Resolve through the registry and run the model's own config hook, which is
// exactly what `ModelRegistry::Load` does before it touches a weight.
Qwen4ExpParams ParseThroughRegistry(const nlohmann::json& doc) {
  TempConfig cfg(doc);
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.factory != nullptr);
  reg.factory->parse_config(config);  // the production hook
  return ParseQwen4ExpParams(config);
}

std::string ThrowText(const nlohmann::json& doc) {
  try {
    ParseThroughRegistry(doc);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST_CASE("qwen4_exp: the published config resolves through the registry") {
  const nlohmann::json doc = FixtureDoc();
  REQUIRE(doc["architectures"][0] == "Qwen4ExpForConditionalGeneration");
  REQUIRE(doc["model_type"] == "qwen4_exp");

  const Qwen4ExpParams p = ParseThroughRegistry(doc);

  CHECK(p.hidden_size == 2560);
  CHECK(p.num_hidden_layers == 48);
  CHECK(p.vocab_size == 248320);
  CHECK(p.hc_count == 4);
  CHECK(p.hc_lowrank == 320);
  CHECK(p.num_experts == 512);
  CHECK(p.num_experts_per_tok == 10);
  CHECK(p.moe_intermediate_size == 640);
  CHECK(p.shared_expert_intermediate_size == 640);
  CHECK(p.num_attention_heads == 24);
  CHECK(p.num_key_value_heads == 2);
  CHECK(p.head_dim == 256);

  // QSA. block_topk is DERIVED, never read: 2048 / 4 = 512.
  CHECK(p.qsa.n_heads == 4);
  CHECK(p.qsa.kv_heads == 1);
  CHECK(p.qsa.head_dim == 128);
  CHECK(p.qsa.budget == 2048);
  CHECK(p.qsa.compress_ratio == 4);
  CHECK(p.qsa.block_topk() == 512);

  // PLE geometry, and the derived values a port gets wrong silently.
  CHECK(p.ple.ngram_size == 3);
  CHECK(p.ple.heads_per_ngram == 8);
  CHECK(p.ple.ngram_heads() == 16);
  CHECK(p.ple.embed_dim == 2560);
  CHECK(p.ple.head_dim_per_ngram() == 160);
  // (4 - 1) * 3 = 9, NOT kernel-1. The conv is dilated, so its state is three
  // times deeper than an undilated one.
  CHECK(p.ple.short_conv_state_len() == 9);
  CHECK(p.ple.split_ngram_parts == 128);
  // Absent from the published config; the dataclass default. This value is
  // load-bearing: it seeds the splitmix64 chain that produces the n-gram hash
  // multipliers, and 1234 is what reproduces the `layer_multipliers` buffer
  // stored in the released checkpoint.
  CHECK(p.ple.seed == 1234);

  CHECK(p.mtp_num_hidden_layers == 1);
  // Three conv states: GDN conv, PLE conv, and the n-gram token history.
  CHECK(p.number_of_conv_states() == 3);
}

TEST_CASE("qwen4_exp: full_attention is rewritten, and the rewrite equals the interval synthesis") {
  nlohmann::json doc = FixtureDoc();
  // The published checkpoint says `full_attention` for layers that actually run
  // the QSA indexer. A reader that takes it at face value wires DENSE attention
  // on 12 of 48 layers and is wrong without saying so.
  const auto& published = doc["text_config"]["layer_types"];
  REQUIRE(published.size() == 48);
  bool saw_full = false;
  for (const auto& e : published) {
    if (e == "full_attention") saw_full = true;
    CHECK_MESSAGE((e == "full_attention" || e == "linear_attention"),
                  "unexpected published layer type " << e);
  }
  REQUIRE_MESSAGE(saw_full,
                  "the fixture must still contain `full_attention`, or this "
                  "case is asserting nothing");

  const Qwen4ExpParams from_list = ParseThroughRegistry(doc);

  // Same config with `layer_types` DELETED, so the interval path runs instead.
  nlohmann::json synth = doc;
  synth["text_config"].erase("layer_types");
  REQUIRE(synth["text_config"].contains("full_attention_interval"));
  const Qwen4ExpParams from_interval = ParseThroughRegistry(synth);

  REQUIRE(from_list.layer_types.size() == 48);
  REQUIRE(from_interval.layer_types.size() == 48);
  CHECK_MESSAGE(from_list.layer_types == from_interval.layer_types,
                "the rewritten published list and the interval synthesis must "
                "agree; if they diverge one of the two paths is wrong and the "
                "checkpoint will not say which");

  std::vector<int64_t> sparse;
  for (size_t i = 0; i < from_list.layer_types.size(); ++i) {
    if (from_list.layer_types[i] == Qwen4ExpLayerKind::kQwenSparseAttention) {
      sparse.push_back(static_cast<int64_t>(i));
    }
  }
  const std::vector<int64_t> expected{3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47};
  CHECK(sparse == expected);
  CHECK(sparse.size() == 12);
}

TEST_CASE("qwen4_exp: ple_layer_ids is ONE-indexed and lands on layer 1") {
  const nlohmann::json doc = FixtureDoc();
  REQUIRE(doc["text_config"]["ple_layer_ids"] == nlohmann::json::array({2}));

  const Qwen4ExpParams p = ParseThroughRegistry(doc);
  // `[2]` one-indexed selects 0-based layer 1. Upstream documents the field as
  // one-indexed, its validator resolves `layer_types[layer_id - 1]`, and every
  // PLE tensor in the released checkpoint sits under `...layers.1.ple.`.
  REQUIRE(p.ple.layer_ids_zero_based.size() == 1);
  CHECK(p.ple.layer_ids_zero_based[0] == 1);
  // And that layer must be a linear-attention one, which is what upstream's
  // own PLE validation requires.
  CHECK(p.layer_types[1] == Qwen4ExpLayerKind::kLinearAttention);
}

TEST_CASE("qwen4_exp: an omitted partial_rotary_factor keeps upstream's inherited 0.25") {
  // REGRESSION GUARD, not a nicety. `IsQwen35Family` in the shared HfConfig
  // reader does not list `qwen4_exp`, so an absent `partial_rotary_factor`
  // defaults THERE to 1.0 (full rotary), while upstream `Qwen4ExpTextConfig`
  // subclasses `Qwen3_5MoeTextConfig` and inherits 0.25. Taking the shared
  // reader's value would give rotary_dim 256, and because upstream's own guard
  // is `rotary_dim > indexer_head_dim` (128), we would REFUSE a config upstream
  // ACCEPTS. Mirroring the inheritance is what keeps the refusal sets identical.
  nlohmann::json doc = FixtureDoc();
  doc["text_config"].erase("partial_rotary_factor");
  if (doc["text_config"].contains("rope_parameters")) {
    doc["text_config"]["rope_parameters"].erase("partial_rotary_factor");
  }

  const Qwen4ExpParams p = ParseThroughRegistry(doc);
  CHECK(p.partial_rotary_factor == doctest::Approx(0.25));
  CHECK(p.rotary_dim == 64);
  CHECK(p.rotary_dim <= p.qsa.head_dim);
}

TEST_CASE("qwen4_exp: the config refuses every unrepresentable combination BY NAME") {
  SUBCASE("an unsupported layer type") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["layer_types"][0] = "sliding_attention";
    CHECK(ThrowText(doc).find("sliding_attention") != std::string::npos);
  }
  SUBCASE("hc_count must exceed 1") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["hc_count"] = 1;
    CHECK(ThrowText(doc).find("hc_count") != std::string::npos);
  }
  SUBCASE("num_experts_per_tok above num_experts") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["num_experts_per_tok"] = 513;
    CHECK(ThrowText(doc).find("num_experts_per_tok") != std::string::npos);
  }
  SUBCASE("a partial QSA group names what is missing") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"].erase("indexer_budget");
    const std::string msg = ThrowText(doc);
    CHECK(msg.find("QSA") != std::string::npos);
    CHECK(msg.find("indexer_budget") != std::string::npos);
  }
  SUBCASE("QSA requires exactly one indexer kv head") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["indexer_kv_heads"] = 2;
    CHECK(ThrowText(doc).find("indexer_kv_heads") != std::string::npos);
  }
  SUBCASE("the indexer budget must divide by the compress ratio") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["indexer_budget"] = 2049;
    CHECK(ThrowText(doc).find("indexer_budget") != std::string::npos);
  }
  SUBCASE("a rotary dim wider than the indexer head") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["partial_rotary_factor"] = 1.0;
    if (doc["text_config"].contains("rope_parameters")) {
      doc["text_config"]["rope_parameters"]["partial_rotary_factor"] = 1.0;
    }
    CHECK(ThrowText(doc).find("indexer_head_dim") != std::string::npos);
  }
  SUBCASE("a PLE id outside the one-indexed range") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["ple_layer_ids"] = nlohmann::json::array({0});
    CHECK(ThrowText(doc).find("one-indexed") != std::string::npos);
  }
  SUBCASE("a PLE id on a sparse-attention layer") {
    nlohmann::json doc = FixtureDoc();
    // One-indexed 4 is 0-based 3, which the rewrite makes sparse.
    doc["text_config"]["ple_layer_ids"] = nlohmann::json::array({4});
    CHECK(ThrowText(doc).find("linear_attention") != std::string::npos);
  }
  SUBCASE("a layer_types list whose length disagrees with num_hidden_layers") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["layer_types"].erase(0);
    CHECK(ThrowText(doc).find("layer_types") != std::string::npos);
  }
}

TEST_CASE("qwen4_exp: load, forward and the KV spec refuse BY NAME, naming the owing wave") {
  const nlohmann::json doc = FixtureDoc();
  TempConfig cfg(doc);
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.factory != nullptr);

  // The config hook must NOT throw: W1 claims exactly this much works.
  CHECK_NOTHROW(reg.factory->parse_config(config));

  SUBCASE("the safetensors loader") {
    const vllm::ModelSource source{};
    std::string msg;
    try {
      (void)reg.factory->load_weights(reg, config, source);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    // Name the architecture, name what is missing, point at the record. A bare
    // "not implemented" sends the reader to the wrong layer.
    CHECK(msg.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(msg.find("weight loader") != std::string::npos);
    CHECK(msg.find("#1978") != std::string::npos);
    // And it must NOT degrade into a lower-layer shape or dtype complaint.
    CHECK(msg.find("tensor not found") == std::string::npos);
  }

  SUBCASE("the KV-cache spec") {
    std::string msg;
    try {
      (void)reg.factory->make_kv_cache(config, 16, 4);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(msg.find("KV-cache spec") != std::string::npos);
  }
}

TEST_CASE("qwen4_exp: the registry reports it as multimodal and hybrid") {
  const nlohmann::json doc = FixtureDoc();
  TempConfig cfg(doc);
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.info.is_text_generation_model);
  CHECK(reg.info.supports_multimodal);
  // 36 of 48 layers are Gated DeltaNet carrying recurrent state.
  CHECK(reg.info.is_hybrid);
  // FALSE by the house convention: the ModelInfo subset's only reader
  // short-circuits on is_hybrid, so every GDN-hybrid wrapper leaves this
  // false even though upstream's class carries HasInnerState.
  CHECK_FALSE(reg.info.has_inner_state);
  CHECK_FALSE(reg.info.is_pooling_model);
}
