// Qwen4-Exp W1 scaffold (MODEL-MM-QWEN4-EXP, #1981).
//
// Everything here drives the PRODUCTION entry point:
// `LoadHfConfig` -> `ModelRegistry::Resolve` -> `factory->parse_config`, and
// the refusals through `factory->load_weights` / `->forward` / `->make_kv_cache`.
// A case that built `Qwen4ExpParams` by hand would prove the struct parses and
// NOT that anything reaches it, which AGENTS.md "Nothing lands dead" refuses to
// accept as evidence.
//
// EVERY refusal below is observed through `reg.factory->parse_config` and
// NOTHING ELSE. The earlier shape of this file called the hook and then
// returned `ParseQwen4ExpParams(config)`, so every assertion in it observed the
// FREE FUNCTION: gutting the registered hook to `(void)config;` left all 151
// assertions green (review finding F2). `ThrowText` now enters through the hook
// alone, so that mutation reds. The value cases still need the returned struct
// -- the hook is `void` and cannot carry one -- so the hook's identity with the
// free function is pinned separately, by the equivalence case below.
//
// ORACLE: transformers **5.16.0**, the lane pin accepted for this row. vLLM
// implements `qwen4_exp` at no revision, so there is nothing to mirror on this
// surface. Values come from the committed fixture, which is the published
// `Qwen/Qwen3.8-Flash-Next` `config.json` verbatim.
#include <chrono>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "doctest/doctest.h"
#include "nlohmann/json.hpp"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, *KvCache
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vt/device.h"
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::ModelRegistry;
using vllm::ParseQwen4ExpParams;
using vllm::Qwen4ExpLayerKind;
using vllm::Qwen4ExpParams;
using vllm::LoadedModel;
using vllm::ModelForwardInput;
using vllm::ModelRegistration;

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
// exactly what `ModelRegistry::Load` does before it touches a weight. The hook
// runs FIRST and is what refuses; the free function then supplies the resolved
// struct the value assertions read, which a `void` hook cannot return.
Qwen4ExpParams ParseThroughRegistry(const nlohmann::json& doc) {
  TempConfig cfg(doc);
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.factory != nullptr);
  reg.factory->parse_config(config);  // the production hook
  return ParseQwen4ExpParams(config);
}

// THE HOOK AND NOTHING ELSE. Every refusal case goes through this, so a hook
// that stopped validating reds the whole refusal suite.
std::string ThrowText(const nlohmann::json& doc) {
  try {
    TempConfig cfg(doc);
    const HfConfig config = LoadHfConfig(cfg.path());
    const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
    REQUIRE(reg.factory != nullptr);
    reg.factory->parse_config(config);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// The same doc put through the FREE FUNCTION alone, for the equivalence case.
std::string ThrowTextDirect(const nlohmann::json& doc) {
  try {
    TempConfig cfg(doc);
    const HfConfig config = LoadHfConfig(cfg.path());
    (void)ParseQwen4ExpParams(config);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// A FOREIGN `LoadedModel`: well-formed, and simply not this model's type. Same
// shape as test_registry_downcast_refusal.cpp, and the only shape available
// here, because nothing can produce a loaded Qwen4-Exp while the loader
// refuses. The forward must therefore refuse BEFORE it opens the handle, or the
// refusal it advertises is unreachable.
class ForeignLoadedModel final : public LoadedModel {
 public:
  explicit ForeignLoadedModel(const ModelRegistration& registration)
      : LoadedModel(registration) {}
};

struct EmptyForwardInput {
  std::vector<int32_t> token_ids{0};
  std::vector<int32_t> positions{0};
  std::vector<int32_t> logits_indices{0};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  HfConfig config{};
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  ModelForwardInput Get() {
    return ModelForwardInput{.token_ids = token_ids,
                             .positions = positions,
                             .attn_meta = attn_meta,
                             .gdn_meta = gdn_meta,
                             .attn_kv = attn_kv,
                             .gdn_state = gdn_state,
                             .config = config,
                             .queue = queue,
                             .logits_indices = logits_indices,
                             .num_reqs = 1};
  }
};

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

TEST_CASE("qwen4_exp: partial_rotary_factor comes from rope_parameters and defaults to 1.0") {
  // THIS CASE IS THE INVERSION OF ITS PREDECESSOR, and the inversion is the
  // point. The earlier case pinned an "inherited 0.25" that does not exist at
  // the pin. `Qwen4ExpTextConfig` is generated as
  // `class Qwen4ExpTextConfig(PreTrainedConfig)`
  // (configuration_qwen4_exp.py:29) -- it does NOT subclass
  // `Qwen3_5MoeTextConfig`, `partial_rotary_factor` is not among its declared
  // fields (:109-164), and the string `0.25` does not appear in the file. Its
  // only two occurrences of the name are the validator's own read,
  // `partial_rotary_factor = (self.rope_parameters or {}).get(
  //      "partial_rotary_factor", 1.0)` (:225) and the `rotary_dim` it feeds
  // (:226). The modular source confirms it deliberately:
  // `Qwen4ExpTextConfig.__post_init__` calls
  // `PreTrainedConfig.__post_init__(self, **kwargs)` DIRECTLY
  // (modular_qwen4_exp.py:194), bypassing the `kwargs.setdefault(
  //      "partial_rotary_factor", 0.25)  # assign default for BC` that is the
  // sole source of 0.25 (configuration_qwen3_5_moe.py:124).
  //
  // So the default is 1.0, and the value lives in `rope_parameters`. The shared
  // reader already implements exactly that: `ParseRopeParameters` takes the top
  // level first and lets `rope_parameters` override, which is what
  // `convert_rope_params_to_dict`'s
  // `self.rope_parameters.setdefault("partial_rotary_factor", kwargs[...])`
  // does (modeling_rope_utils.py:755-757) -- and it runs BEFORE the generic
  // `setattr` loop that would otherwise let `standardize_rope_params`:788
  // overwrite the dict (configuration_utils.py:314 vs :339), so `setdefault`
  // is the whole precedence. `IsQwen35Family` correctly excludes `qwen4_exp`,
  // so `config.rotary_dim` IS upstream's `rotary_dim`.
  SUBCASE("the published checkpoint, where both spellings say 0.25") {
    const Qwen4ExpParams p = ParseThroughRegistry(FixtureDoc());
    CHECK(p.partial_rotary_factor == doctest::Approx(0.25));
    CHECK(p.rotary_dim == 64);
    CHECK(p.rotary_dim <= p.qsa.head_dim);
  }

  SUBCASE("absent everywhere: 1.0, rotary_dim 256, and upstream REFUSES") {
    // Upstream: `(None or {}).get(..., 1.0)` -> 1.0 -> rotary_dim 256 >
    // indexer_head_dim 128 -> ValueError. The old code answered 0.25/64 and
    // ACCEPTED, handing W4 a 64-of-256 slice on a checkpoint that wants 256 --
    // a silent numerics error on a row with no reachable token gate.
    nlohmann::json doc = FixtureDoc();
    doc["text_config"].erase("partial_rotary_factor");
    doc["text_config"]["rope_parameters"].erase("partial_rotary_factor");
    const std::string msg = ThrowText(doc);
    CHECK(msg.find("rotary dim 256") != std::string::npos);
    CHECK(msg.find("indexer_head_dim") != std::string::npos);
  }

  SUBCASE("rope_parameters WINS over the top level") {
    // Upstream reads the rope dict and never the top-level key, so top-level
    // 1.0 with rope 0.25 is ACCEPTED at rotary_dim 64. The old code read the
    // top level, got 1.0, and refused -- the exact false refusal its comment
    // claimed to prevent.
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["partial_rotary_factor"] = 1.0;
    doc["text_config"]["rope_parameters"]["partial_rotary_factor"] = 0.25;
    const Qwen4ExpParams p = ParseThroughRegistry(doc);
    CHECK(p.partial_rotary_factor == doctest::Approx(0.25));
    CHECK(p.rotary_dim == 64);
  }

  SUBCASE("the top level fills in when the rope dict omits the key") {
    // `setdefault` semantics: the top-level key is folded into
    // `rope_parameters` only where the dict has none.
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["partial_rotary_factor"] = 0.25;
    doc["text_config"]["rope_parameters"].erase("partial_rotary_factor");
    const Qwen4ExpParams p = ParseThroughRegistry(doc);
    CHECK(p.rotary_dim == 64);
  }

  SUBCASE("top-level 0.25 does NOT rescue a rope dict that says 1.0") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["partial_rotary_factor"] = 0.25;
    doc["text_config"]["rope_parameters"]["partial_rotary_factor"] = 1.0;
    CHECK(ThrowText(doc).find("rotary dim 256") != std::string::npos);
  }
}

TEST_CASE("qwen4_exp: the REGISTERED hook is the validator, not a bystander") {
  // The hook is `void`, so no value can flow through it and no value case can
  // be made unreachable without it. What CAN be pinned is its identity with the
  // free function: for the same document the two must give the same answer,
  // refusal text included. A hook gutted to `(void)config;` reds every row.
  const nlohmann::json fixture = FixtureDoc();

  nlohmann::json bad_gate = fixture;
  bad_gate["text_config"].erase("output_gate_type");
  bad_gate["text_config"]["hidden_act"] = "gelu";

  nlohmann::json bad_eos = fixture;
  bad_eos["text_config"]["eos_token_id"] = nullptr;

  nlohmann::json bad_hc = fixture;
  bad_hc["text_config"]["hc_count"] = 1;

  nlohmann::json bad_rope = fixture;
  bad_rope["text_config"].erase("partial_rotary_factor");
  bad_rope["text_config"]["rope_parameters"].erase("partial_rotary_factor");

  for (const nlohmann::json* doc : {&bad_gate, &bad_eos, &bad_hc, &bad_rope}) {
    const std::string through_hook = ThrowText(*doc);
    const std::string direct = ThrowTextDirect(*doc);
    CHECK_FALSE(through_hook.empty());
    CHECK(through_hook == direct);
  }
  // ...and the fixture passes through BOTH.
  CHECK(ThrowText(fixture).empty());
  CHECK(ThrowTextDirect(fixture).empty());
}

TEST_CASE("qwen4_exp: the PLE defaults are upstream's, not zero") {
  // Every one of these is a declared field with a default
  // (configuration_qwen4_exp.py:149-157), so a config that omits them is legal
  // upstream. Defaulting them to 0 made us REFUSE such a config with
  // "`ngram_size` must be >= 2 ... got 0" (review probe F), and carried a zero
  // n-gram vocabulary silently into W2 for the two fields that have no guard.
  nlohmann::json doc = FixtureDoc();
  for (const char* key : {"ngram_size", "heads_per_ngram",
                          "ngram_vocab_size_base",
                          "make_ngram_vocab_size_divisible_by",
                          "split_ngram_parts", "ple_conv_kernel_size",
                          "ple_embed_dim", "seed"}) {
    doc["text_config"].erase(key);
  }
  const Qwen4ExpParams p = ParseThroughRegistry(doc);
  CHECK(p.ple.ngram_size == 3);
  CHECK(p.ple.heads_per_ngram == 8);
  CHECK(p.ple.ngram_vocab_size_base == 20000000);
  CHECK(p.ple.make_ngram_vocab_size_divisible_by == 128);
  CHECK(p.ple.split_ngram_parts == 512);
  CHECK(p.ple.conv_kernel_size == 4);
  CHECK(p.ple.seed == 1234);
  // `ple_embed_dim` defaults to `hidden_size` in upstream's `__post_init__`.
  CHECK(p.ple.embed_dim == 2560);
  CHECK(p.ple.ngram_heads() == 16);
}

TEST_CASE("qwen4_exp: the n-gram fields resolve even when no layer uses PLE") {
  // Upstream carries them as dataclass fields, so they hold their defaults
  // whether or not `ple_layer_ids` is set. Reading them only inside the PLE
  // branch left `ngram_size == 0`, which made `ngram_heads()` zero and
  // `head_dim_per_ngram()` a division by zero on a legally-parsed config.
  nlohmann::json doc = FixtureDoc();
  doc["text_config"].erase("ple_layer_ids");
  const Qwen4ExpParams p = ParseThroughRegistry(doc);
  CHECK(p.ple.layer_ids_zero_based.empty());
  CHECK(p.number_of_conv_states() == 1);
  CHECK(p.ple.ngram_size == 3);
  CHECK(p.ple.ngram_heads() == 16);
  CHECK(p.ple.head_dim_per_ngram() == 160);
}

TEST_CASE("qwen4_exp: the derived helpers refuse instead of dividing by zero") {
  // `block_topk()` and `head_dim_per_ngram()` are advertised by the header as
  // the derived values a port should use, and W2/W4 will call them. Both
  // divide, and both divisors can legally be zero: QSA is optional as a group,
  // and `ngram_size == 1` makes the head count zero. `budget / 0` is SIGFPE on
  // x86 -- a crash, not a refusal, and one no gate downstream would attribute.
  SUBCASE("QSA absent: block_topk refuses by name") {
    nlohmann::json doc = FixtureDoc();
    for (const char* f : {"indexer_n_heads", "indexer_kv_heads",
                          "indexer_head_dim", "indexer_budget",
                          "indexer_compress_ratio"}) {
      doc["text_config"].erase(f);
    }
    const Qwen4ExpParams p = ParseThroughRegistry(doc);
    REQUIRE(p.qsa.compress_ratio == 0);
    CHECK_THROWS_WITH_AS(p.qsa.block_topk(),
                         doctest::Contains("indexer_compress_ratio"),
                         std::runtime_error);
  }
  SUBCASE("a zero n-gram head count: head_dim_per_ngram refuses by name") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"].erase("ple_layer_ids");  // no PLE => not validated
    doc["text_config"]["ngram_size"] = 1;       // (1 - 1) * 8 == 0
    const Qwen4ExpParams p = ParseThroughRegistry(doc);
    REQUIRE(p.ple.ngram_heads() == 0);
    CHECK_THROWS_WITH_AS(p.ple.head_dim_per_ngram(),
                         doctest::Contains("n-gram head count"),
                         std::runtime_error);
  }
}

TEST_CASE("qwen4_exp: a text config nested under llm_config resolves the SAME way") {
  // `HfConfig`'s own `ResolveTextConfig` handles `text_config`, `llm_config`
  // and `thinker_config.text_config`; the model's local `TextOf` handled only
  // the first, so on an `llm_config` wrapper the shared reader found
  // `hidden_size`/`layer_types` while the model found NO `hc_*`, QSA, PLE or
  // MTP key and silently produced a half-parsed config.
  nlohmann::json doc = FixtureDoc();
  nlohmann::json nested = doc;
  nested["llm_config"] = doc["text_config"];
  nested.erase("text_config");

  const Qwen4ExpParams a = ParseThroughRegistry(doc);
  const Qwen4ExpParams b = ParseThroughRegistry(nested);
  CHECK(b.hc_count == a.hc_count);
  CHECK(b.qsa.budget == a.qsa.budget);
  CHECK(b.ple.ngram_size == a.ple.ngram_size);
  CHECK(b.ple.layer_ids_zero_based == a.ple.layer_ids_zero_based);
  CHECK(b.mtp_num_hidden_layers == a.mtp_num_hidden_layers);
  CHECK(b.number_of_conv_states() == a.number_of_conv_states());
}

// EVERY refusal in `ParseQwen4ExpParams` has a subcase here, and that is a GATE
// OBLIGATION rather than thoroughness: the row's `## Gates` G0 item 6 is "every
// rejection in `validate_architecture`", and this row has no reachable token
// gate, so the config layer is the last place the refusal boundary is checkable
// at all. A single mutation deleting 13 of the 22 refusals used to leave the
// suite green (review finding F10).
//
// The `[UP]` rows mirror one upstream raise; the `[LOCAL]` rows are refusals
// this port imposes that `validate_architecture` does not. Both directions are
// tabulated against their upstream line in
// `.agents/specs/qwen4-exp-flash-next.md` `## The refusal boundary`.
TEST_CASE("qwen4_exp: the config refuses every unrepresentable combination BY NAME") {
  SUBCASE("[LOCAL] num_hidden_layers must be positive") {
    // The DISTINGUISHING text, not the bare field name. The next refusal down
    // ("`layer_types` has 48 entries but `num_hidden_layers` is 0") also names
    // the field, so deleting this guard left the case green -- caught by
    // mutating this refusal alone, and it is the shape a substring assertion
    // takes whenever two refusals share a word.
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["num_hidden_layers"] = 0;
    CHECK(ThrowText(doc).find("`num_hidden_layers` must be > 0") !=
          std::string::npos);
  }
  SUBCASE("[LOCAL] full_attention_interval must be positive") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"].erase("layer_types");
    doc["text_config"]["full_attention_interval"] = 0;
    CHECK(ThrowText(doc).find("full_attention_interval") != std::string::npos);
  }
  SUBCASE("[LOCAL] hc_lowrank must be positive") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["hc_lowrank"] = 0;
    CHECK(ThrowText(doc).find("hc_lowrank") != std::string::npos);
  }
  SUBCASE("[UP] num_experts must be positive") {
    // Same trap: the `num_experts_per_tok` range refusal below also prints
    // `num_experts`, so the bare field name passed with this guard deleted.
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["num_experts"] = 0;
    CHECK(ThrowText(doc).find("`num_experts` must be > 0") != std::string::npos);
  }
  SUBCASE("[UP] the MoE intermediate sizes must be positive") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["moe_intermediate_size"] = 0;
    CHECK(ThrowText(doc).find("moe_intermediate_size") != std::string::npos);
  }
  SUBCASE("[UP] an absent output_gate_type falls back to hidden_act") {
    // `output_gate_type = self.output_gate_type or self.hidden_act`
    // (configuration_qwen4_exp.py:193). The shared reader defaults an ABSENT
    // key to "silu" unconditionally, which made us accept a checkpoint whose
    // gate is whatever `hidden_act` says (review probe G) and left the local
    // check a constant false.
    nlohmann::json doc = FixtureDoc();
    doc["text_config"].erase("output_gate_type");
    doc["text_config"]["hidden_act"] = "gelu";
    const std::string msg = ThrowText(doc);
    CHECK(msg.find("output gate") != std::string::npos);
    CHECK(msg.find("gelu") != std::string::npos);
  }
  SUBCASE("[UP] an explicit output_gate_type outside {sigmoid, silu}") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["output_gate_type"] = "swish";
    // Upstream compares the RAW string against {"sigmoid", "silu"}, so `swish`
    // raises there. The shared reader canonicalizes `swish` to `silu` for the
    // GDN family, so the refusal has to be taken on the raw value here.
    CHECK(ThrowText(doc).find("swish") != std::string::npos);
  }
  SUBCASE("[SHARED, tighter than upstream] a factor outside (0, 1]") {
    // Named for what it IS: this refusal comes from the SHARED reader
    // (`hf_config: partial_rotary_factor must be in (0, 1]`), before this
    // model's parse runs, and upstream validates the factor not at all. A local
    // `> 0` guard here would be unreachable, which is why there is not one --
    // the same constant-false shape the output-gate check used to have. The
    // assertion names the shared message so that a later widening of that bound
    // shows up here rather than passing on a substring both messages share.
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["rope_parameters"]["partial_rotary_factor"] = -0.25;
    CHECK(ThrowText(doc).find("must be in (0, 1]") != std::string::npos);
  }
  SUBCASE("[UP] QSA values must be positive") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["indexer_n_heads"] = 0;
    CHECK(ThrowText(doc).find("positive") != std::string::npos);
  }
  SUBCASE("[LOCAL] ngram_size below 2") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["ngram_size"] = 1;
    CHECK(ThrowText(doc).find("ngram_size") != std::string::npos);
  }
  SUBCASE("[LOCAL] heads_per_ngram must be positive") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["heads_per_ngram"] = 0;
    CHECK(ThrowText(doc).find("heads_per_ngram") != std::string::npos);
  }
  SUBCASE("[LOCAL] ple_conv_kernel_size must be positive") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["ple_conv_kernel_size"] = 0;
    CHECK(ThrowText(doc).find("ple_conv_kernel_size") != std::string::npos);
  }
  SUBCASE("[UP] a NEGATIVE ple_embed_dim, which -2560 % 16 == 0 lets through") {
    // Upstream's condition is `ngram_heads <= 0 or self.ple_embed_dim <= 0 or
    // self.ple_embed_dim % ngram_heads != 0` (:235). Dropping the middle term
    // accepted -2560 in C++, where the remainder is 0, and
    // `head_dim_per_ngram()` then returned -160 (review probe H).
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["ple_embed_dim"] = -2560;
    CHECK(ThrowText(doc).find("ple_embed_dim") != std::string::npos);
  }
  SUBCASE("[UP] a ple_embed_dim that does not divide by the head count") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["ple_embed_dim"] = 2561;
    CHECK(ThrowText(doc).find("ple_embed_dim") != std::string::npos);
  }
  SUBCASE("[UP] eos_token_id must be set when PLE is enabled") {
    // `configuration_qwen4_exp.py:256-257`. Not cosmetic: the n-gram history
    // uses `_shift_right_ignore_eos`, so EOS is a SEGMENT BOUNDARY in the
    // hashed n-gram construction, and the published GGUF stores it as
    // `qwen4exp.ple.eos_token_id`. A null there hands W2 a config whose n-gram
    // ids cannot be built, against a gate that is integer equality.
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["eos_token_id"] = nullptr;
    CHECK(ThrowText(doc).find("eos_token_id") != std::string::npos);
  }
  SUBCASE("[UP] an EMPTY eos_token_id list is refused too") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["eos_token_id"] = nlohmann::json::array();
    CHECK(ThrowText(doc).find("eos_token_id") != std::string::npos);
  }
  SUBCASE("[LOCAL] mtp_num_hidden_layers must not be negative") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["mtp_num_hidden_layers"] = -1;
    CHECK(ThrowText(doc).find("mtp_num_hidden_layers") != std::string::npos);
  }
  SUBCASE("[LOCAL] a non-integer where an integer belongs") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["hc_lowrank"] = "three hundred and twenty";
    CHECK(ThrowText(doc).find("must be an integer") != std::string::npos);
  }
  SUBCASE("[LOCAL] ple_layer_ids that is not an array") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["ple_layer_ids"] = 2;
    CHECK(ThrowText(doc).find("must be an array") != std::string::npos);
  }
  SUBCASE("[LOCAL] ple_layer_ids holding a non-integer") {
    nlohmann::json doc = FixtureDoc();
    doc["text_config"]["ple_layer_ids"] = nlohmann::json::array({"two"});
    CHECK(ThrowText(doc).find("only integers") != std::string::npos);
  }
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

TEST_CASE("qwen4_exp: the safetensors load, the forward and the KV spec refuse BY NAME") {
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
    // W5a (#2031) sharpened this one. It used to say the loader "is not ported
    // yet", which reads as scheduling; the real reason is that every published
    // safetensors artifact of this model — bf16 ~360 GB, FP8 ~180 GB, NVFP4
    // ~128 GB — exceeds every device this project owns, so the arm would be
    // code nothing could run. The message has to say WHICH arm is the supported
    // one, or the reader is left thinking no arm exists.
    CHECK(msg.find("safetensors") != std::string::npos);
    CHECK(msg.find("GGUF") != std::string::npos);
    // And it must NOT degrade into a lower-layer shape or dtype complaint.
    CHECK(msg.find("tensor not found") == std::string::npos);
  }

  SUBCASE("a GGUF source with no file refuses instead of dereferencing null") {
    // THIS SUBCASE CHANGED MEANING IN W5a (#2031) AND THE OLD ONE IS RECORDED
    // HERE SO THE CHANGE IS NOT READ AS A WEAKENING. It used to assert that the
    // GGUF arm refused, naming the IQ4_NL reader arm and the quantized gather
    // that W6 owed. Both landed in W6a (#1989), and W5a loads the arm — so the
    // refusal that assertion pinned no longer exists, and the case that gates
    // the real GGUF load is `test_qwen4_exp_gguf_weights.cpp`, which drives a
    // synthetic `qwen4exp` file through this same hook.
    //
    // What is left to gate HERE is the shape this scaffold can still reach: a
    // caller that sets the KIND without the FILE. Before W5a that combination
    // hit an unconditional throw; now it reaches a loader, and a loader that
    // dereferenced the null pointer would segfault inside a code path a reader
    // is entitled to read as "GGUF is unsupported".
    vllm::ModelSource source{};
    source.kind = vllm::ModelSource::Kind::kGguf;
    std::string msg;
    try {
      (void)reg.factory->load_weights(reg, config, source);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(msg.find("GGUF") != std::string::npos);
    CHECK(msg.find("carries no file") != std::string::npos);
    // ...and it must NOT be the safetensors message, which would send the
    // reader to an arm that is refused for an entirely different reason.
    CHECK(msg.find("safetensors") == std::string::npos);
  }

  SUBCASE("the forward") {
    // The case was TITLED "load, forward and the KV spec refuse" and had no
    // forward subcase: deleting the `VT_CHECK` left the forward returning an
    // empty `ForwardLogits{}` and the gate green (review mutation M6).
    //
    // The refusal has to come BEFORE the `ModelAs` downcast, or it is
    // unreachable rather than merely untested: nothing can produce a loaded
    // Qwen4-Exp while the loader refuses, so the only handle any caller can
    // present is a foreign one, and a downcast placed first turns every reach
    // into a type-mismatch report instead.
    REQUIRE(reg.factory->forward != nullptr);
    ForeignLoadedModel foreign(reg);
    EmptyForwardInput in;
    const ModelForwardInput input = in.Get();
    std::string msg;
    try {
      (void)reg.factory->forward(foreign, input);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(msg.find("forward is not ported") != std::string::npos);
    // Each wave that owes a piece of it is named, so the reader is not sent to
    // the loader for work W2/W3/W4 owe.
    CHECK(msg.find("W2") != std::string::npos);
    CHECK(msg.find("W4") != std::string::npos);
    CHECK(msg.find("#1978") != std::string::npos);
    // And it is NOT the type-mismatch report, which would mean the refusal this
    // model advertises is unreachable behind a downcast.
    CHECK(msg.find("was not produced by") == std::string::npos);
  }

  SUBCASE("the KV-cache spec no longer refuses") {
    // CHANGED AT W5c (#2031), AND THE OLD ASSERTION IS RECORDED HERE SO THE
    // CHANGE IS NOT READ AS A WEAKENING. This subcase used to assert that
    // `make_kv_cache` threw naming "Qwen4ExpForConditionalGeneration" and
    // "KV-cache spec"; W5c makes it RETURN a three-group config, so the
    // refusal that assertion pinned no longer exists. What gates the spec's
    // CONTENT is `test_qwen4_exp_kv_cache.cpp`, which drives this same hook.
    // What is left to gate HERE is the polarity: the forward above still
    // refuses while this one does not, and a reader of this case is entitled
    // to see which of the two moved.
    CHECK_NOTHROW((void)reg.factory->make_kv_cache(config, 16, 4));
    const vllm::v1::KVCacheConfig kv =
        reg.factory->make_kv_cache(config, 16, 4);
    CHECK(kv.kv_cache_groups.size() == 3);
    CHECK(kv.num_blocks == 4);
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
