// dots3-note (`Dots3NoteForCausalLM`) W1 STRUCTURAL gate — issue #699, spec
// `.agents/specs/dots3-note.md` §4 / §7 W1.
//
// WHY THIS FILE IS THE WHOLE GATE FOR W1. Spec §6.4 records the decision that
// this row has NO oracle: the checkpoint is ~576 GB bf16 / ~290 GB fp8 and the
// biggest host this project owns is 122 GiB, so vLLM cannot run this model
// anywhere we can reach. There is therefore no token gate downstream of these
// assertions. A config field read wrong here does not crash and does not change
// a shape — the model emits plausible text with the wrong experts selected or
// the wrong RoPE coordinates rotated. These unit assertions are the only
// instrument this row has, which is why §4 requires one per trap, RED-first,
// before the layer that consumes it exists.
//
// What it proves, all on CPU with no GPU and without the 576 GB checkpoint:
//   (1) the arch RESOLVES through the registry (the additive TU registered it),
//       and `Dots3NoteMTPModel` deliberately does NOT (INVENTORIED, W10);
//   (2) the REAL released config.json parses (committed byte-for-byte as a
//       fixture) — the 46-entry 13-full/33-sliding schedule, BOTH MLA
//       geometries, the MoE dims, and ALL SIX §4 TRAPS;
//   (3) the on-disk NAME MAP is faithful: every one of the 1614 tensors in the
//       committed slice of the released `model.safetensors.index.json` is
//       CLAIMED by a named consumer, and nothing is enumerated that the
//       checkpoint does not ship;
//   (4) the geometry the params imply is the geometry the released safetensors
//       HEADERS carry (shapes, and the one F32 tensor in a BF16 tower);
//   (5) the padded MLA KV row is the SLIDING row, not the full one;
//   (6) load succeeds only with 100% accounting, the forward REFUSES BY NAME
//       through the REAL model the factory returns, and GGUF refuses by name.
//
// The fixtures are the released `config.json` verbatim and a HEADERS-ONLY
// projection of the shard index at revision
// `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b`; see `index.json`'s `_provenance`.
// No tensor byte of the checkpoint was ever read.
//
// Upstream anchors are at vLLM `origin/main` =
// `c205726108df54bb6fbf15b19e725a4a3add2b18`. `dots3_note` does NOT exist at
// our parity pin `555967922`.
#include "vllm/model_executor/models/dots3_note.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
// The forward-refusal case CALLS the type-erased forward, so it needs the
// concrete definitions of the seam types `model_registry.h` forward-declares.
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, PagedKvCache
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

using vllm::AccountDots3NoteTensors;
using vllm::Dots3NoteAccounting;
using vllm::Dots3NoteLayerKind;
using vllm::Dots3NoteParams;
using vllm::Dots3NoteTensor;
using vllm::Dots3NoteWeights;
using vllm::EnumerateDots3NoteTensors;
using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::MakeDots3NoteKVCache;
using vllm::ModelRegistry;
using vllm::ParseDots3NoteParams;

namespace {

std::string FixtureDir() {
#ifdef DOTS3_NOTE_CKPT_FIXTURE_DIR
  return DOTS3_NOTE_CKPT_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/dots3_note_prev";
#endif
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "cannot open " << path);
  nlohmann::json j;
  in >> j;
  return j;
}

// A throwaway config.json on disk, so every case drives the SAME LoadHfConfig
// the engine uses rather than hand-building an HfConfig.
class TempConfig {
 public:
  explicit TempConfig(const nlohmann::json& doc) {
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
           ("dots3_note_cfg_" + std::to_string(counter++));
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

nlohmann::json FixtureConfigDoc() { return ReadJson(FixtureDir() + "/config.json"); }

Dots3NoteParams FixtureParams() {
  TempConfig cfg(FixtureConfigDoc());
  return ParseDots3NoteParams(LoadHfConfig(cfg.path()));
}

// Expand the committed index fixture back into one entry per real tensor name.
std::map<std::string, std::pair<std::string, std::vector<int64_t>>>
ExpandIndexFixture(const nlohmann::json& index) {
  std::map<std::string, std::pair<std::string, std::vector<int64_t>>> out;
  for (const auto& [pattern, meta] : index.at("tensors").items()) {
    const auto dtype = meta.at("dtype").get<std::string>();
    const auto shape = meta.at("shape").get<std::vector<int64_t>>();
    const auto count = meta.at("count").get<int64_t>();
    const std::string marker = "{E}";
    const size_t at = pattern.find(marker);
    if (at == std::string::npos) {
      REQUIRE_MESSAGE(count == 1, "non-expert family with count != 1: " << pattern);
      out.emplace(pattern, std::make_pair(dtype, shape));
      continue;
    }
    for (int64_t e = 0; e < count; ++e) {
      std::string name = pattern;
      name.replace(at, marker.size(), std::to_string(e));
      out.emplace(std::move(name), std::make_pair(dtype, shape));
    }
  }
  return out;
}

// The backbone layers the committed slice carries: one of every class.
const std::vector<int64_t>& SliceLayers() {
  static const std::vector<int64_t> layers{0, 1, 2};
  return layers;
}

}  // namespace

TEST_CASE("dots3-note: the architecture resolves through the model registry") {
  const std::vector<std::string> archs{"Dots3NoteForCausalLM"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(archs);
  CHECK(reg.architecture == "Dots3NoteForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  // registry.py:381 puts it in _MULTIMODAL_MODELS: image, video AND audio
  // (multimodal.py:65-72).
  CHECK(reg.info.is_text_generation_model);
  CHECK(reg.info.supports_multimodal);
  // Both attention classes page the same MLA cache; the sliding half is a
  // window on it, not a recurrent state.
  CHECK_FALSE(reg.info.is_hybrid);

  // The speculative head is INVENTORIED, not registered: registering a
  // speculator that cannot propose makes the engine accept a config it then
  // dies on mid-run. W10 owns it.
  const std::vector<std::string> mtp{"Dots3NoteMTPModel"};
  CHECK_THROWS_AS((void)ModelRegistry::Resolve(mtp), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 — THE SIX CONFIG TRAPS. Four of them are values the released config.json
// does NOT carry, and every one of the six is numerically SILENT when wrong.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("dots3-note §4 TRAP 1+2: the router is UNGROUPED, and the JSON never says so") {
  // `Dots3NoteConfig.__init__` (transformers_utils/configs/dots3_note.py:18-19)
  //     kwargs.setdefault("n_group", 1)
  //     kwargs.setdefault("topk_group", 1)
  // runs BEFORE `super().__init__`, so DeepseekV3Config's own defaults of 8 and
  // 4 (transformers configuration_deepseek_v3.py:168-169) NEVER apply. Upstream
  // says why in the comment right above them: "Do not inherit DeepSeek-V3's
  // 8-group/4-group router defaults: Note was trained with an ungrouped (1/1)
  // noaux_tc router ... A different grouping changes the selected experts at
  // every MoE layer."
  //
  // This is the trap that costs the most and shows the least: our noaux_tc
  // router is gated at V3's GROUPED dims, so inheriting 8/4 here would route
  // every one of the 45 MoE layers through a group mask the model was never
  // trained with. Same shapes, same dtypes, no error, plausible text.
  const nlohmann::json doc = FixtureConfigDoc();
  REQUIRE_MESSAGE(!doc.contains("n_group"),
                  "the released config.json grew an n_group key — re-read "
                  "configs/dots3_note.py before trusting this default");
  REQUIRE_MESSAGE(!doc.contains("topk_group"),
                  "the released config.json grew a topk_group key");

  const Dots3NoteParams p = FixtureParams();
  CHECK_MESSAGE(p.n_group == 1,
                "n_group resolved to " << p.n_group
                                       << ", not 1 — DeepSeek-V3's default of 8 "
                                          "regroups the router at every MoE layer "
                                          "(configs/dots3_note.py:18)");
  CHECK_MESSAGE(p.topk_group == 1,
                "topk_group resolved to "
                    << p.topk_group
                    << ", not 1 — DeepSeek-V3's default of 4 selects experts "
                       "from a subset of groups (configs/dots3_note.py:19)");
  // ...and the ungrouped router is the only arm this port has: a config that
  // really did carry a grouping is REFUSED by name rather than approximated.
  nlohmann::json grouped = doc;
  grouped["n_group"] = 8;
  grouped["topk_group"] = 4;
  TempConfig cfg(grouped);
  CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                       doctest::Contains("UNGROUPED noaux_tc router"),
                       std::runtime_error);
}

TEST_CASE("dots3-note §4 TRAP 3: the indexer RoPE is GPT-J interleaved, and the JSON never says so") {
  // `Dots3NoteConfig.__init__` (configs/dots3_note.py:23) sets
  // `indexer_rope_interleave = True`. It is consumed exactly once, as
  //     is_neox_style = not getattr(config, "indexer_rope_interleave", False)
  // in `deepseek_v2.py`::DeepseekV2MLAAttention (:1148) — note that getattr's
  // own default is False, i.e. DeepSeek-V3.2's split-half NeoX, which is what
  // our DSA indexer was ported against. Wrong value => the indexer rotates a
  // different set of learned coordinates, with no shape change and no error.
  const nlohmann::json doc = FixtureConfigDoc();
  REQUIRE_MESSAGE(!doc.contains("indexer_rope_interleave"),
                  "the released config.json grew an indexer_rope_interleave key");

  const Dots3NoteParams p = FixtureParams();
  CHECK_MESSAGE(p.indexer_rope_interleave,
                "indexer_rope_interleave resolved FALSE — that is DeepSeek-V3.2's "
                "absent-key default (deepseek_v2.py:1148), not dots3-note's "
                "(configs/dots3_note.py:23)");
  // The form the layer actually consumes.
  CHECK_MESSAGE(!p.indexer_rope_is_neox_style(),
                "the indexer RoPE resolved to NeoX (split-half); dots3-note "
                "projects adjacent GPT-J pairs");
  // An explicit false in the JSON is honoured — this is a setdefault, not a
  // hard-code, and a future checkpoint may say so.
  nlohmann::json neox = doc;
  neox["indexer_rope_interleave"] = false;
  TempConfig cfg(neox);
  const Dots3NoteParams q = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
  CHECK_FALSE(q.indexer_rope_interleave);
  CHECK(q.indexer_rope_is_neox_style());
}

TEST_CASE("dots3-note §4 TRAP 4: there is exactly ONE nextn layer, and the JSON never says so") {
  // `Dots3NoteConfig.__init__` (configs/dots3_note.py:24) sets
  // `num_nextn_predict_layers = 1`. DeepseekV3Config has no such field at all,
  // so an absent key resolves to 0 in any reader that does not know the
  // dots3 default — and 0 leaves the ENTIRE nextn tail unclaimed by the loader.
  const nlohmann::json doc = FixtureConfigDoc();
  REQUIRE_MESSAGE(!doc.contains("num_nextn_predict_layers"),
                  "the released config.json grew a num_nextn_predict_layers key");

  const Dots3NoteParams p = FixtureParams();
  CHECK_MESSAGE(p.num_nextn_predict_layers == 1,
                "num_nextn_predict_layers resolved to "
                    << p.num_nextn_predict_layers
                    << ", not 1 (configs/dots3_note.py:24)");

  // ...and the CHECKPOINT agrees, which is the part §1.4 of the spec left
  // open. The released shard index carries backbone layers 0..45 plus EXACTLY
  // ONE more, `model.layers.46.*` (18 tensors), and one
  // `model.mtp.embed_tokens.weight`. So the model card's "three-token
  // speculative decoding" is the depth ONE head is driven at, not three heads.
  const auto released = ExpandIndexFixture(ReadJson(FixtureDir() + "/index.json"));
  std::set<int64_t> nextn_layers;
  for (const auto& [name, meta] : released) {
    (void)meta;
    int64_t layer = -1;
    if (std::sscanf(name.c_str(), "model.layers.%ld.", &layer) == 1 &&
        layer >= p.num_hidden_layers) {
      nextn_layers.insert(layer);
    }
  }
  CHECK(nextn_layers == std::set<int64_t>{46});
  CHECK(released.count("model.mtp.embed_tokens.weight") == 1);
  // `has_own_lm_head = False` (mtp.py:142): the nextn block shares the target's
  // lm_head, so there is no `shared_head.head.weight` on disk.
  CHECK(released.count("model.layers.46.shared_head.head.weight") == 0);
  CHECK(released.count("model.layers.46.shared_head.norm.weight") == 1);
}

TEST_CASE("dots3-note §4 TRAP 5: apply_mla_qkv_lora_rescale scales BOTH geometries") {
  // model.py:305-307 (full) and :438-443 (sliding):
  //   q_lora_scale  = (hidden_size / q_lora_rank)  ** 0.5
  //   kv_lora_scale = (hidden_size / kv_lora_rank) ** 0.5
  // applied AFTER q_a_layernorm / kv_a_layernorm (model.py:155, :159). Our
  // DeepSeek MLA has no such scalar, so an unported rescale is a uniform gain
  // error on every q and every kv latent — invisible to a shape check.
  const Dots3NoteParams p = FixtureParams();
  REQUIRE(p.apply_mla_qkv_lora_rescale);
  const double h = static_cast<double>(p.hidden_size);
  CHECK(p.full.q_lora_scale == doctest::Approx(std::sqrt(h / 1024.0)));   // 5120/1024
  CHECK(p.full.kv_lora_scale == doctest::Approx(std::sqrt(h / 512.0)));   // 5120/512
  CHECK(p.swa.q_lora_scale == doctest::Approx(std::sqrt(h / 1024.0)));
  CHECK(p.swa.kv_lora_scale == doctest::Approx(std::sqrt(h / 1024.0)));
  // The two geometries genuinely DISAGREE on kv: full is sqrt(10) and sliding
  // is sqrt(5). A port that resolved one scalar for the whole model would be
  // wrong on 33 of the 46 layers.
  CHECK(p.full.kv_lora_scale != doctest::Approx(p.swa.kv_lora_scale));

  // ...and the flag is honoured, not assumed: false means 1.0 everywhere.
  nlohmann::json off = FixtureConfigDoc();
  off["apply_mla_qkv_lora_rescale"] = false;
  TempConfig cfg(off);
  const Dots3NoteParams q = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
  CHECK(q.full.q_lora_scale == doctest::Approx(1.0));
  CHECK(q.full.kv_lora_scale == doctest::Approx(1.0));
  CHECK(q.swa.q_lora_scale == doctest::Approx(1.0));
  CHECK(q.swa.kv_lora_scale == doctest::Approx(1.0));
}

TEST_CASE("dots3-note §4 TRAP 6: the sliding RoPE has its OWN theta, and BOTH ropes are GPT-J") {
  // 6a — the sliding layers pass `rope_theta: config.swa_rope_theta`
  // (model.py:406), 5e4, against the full layers' model-level 8e7
  // (model.py:246-250 hands `config.rope_parameters["rope_theta"]` down). Three
  // orders of magnitude apart on 33 of the 46 layers.
  const Dots3NoteParams p = FixtureParams();
  CHECK(p.full.rope_theta == doctest::Approx(8e7));
  CHECK(p.swa.rope_theta == doctest::Approx(5e4));
  CHECK_MESSAGE(p.full.rope_theta != doctest::Approx(p.swa.rope_theta),
                "the two geometries resolved the SAME rope theta");

  // 6b — the LAYOUT, and this CORRECTS the spec's W0 reading. §4 item 6 said
  // `is_neox_style=False` was "on the sliding rope only". It is not: the
  // sliding class says so literally (model.py:408) and the full class inherits
  // the same value from `deepseek_v2.py`::DeepseekV2MLAAttention:1093-1097,
  // which hard-codes `is_neox_style=False`. BOTH MLA ropes are GPT-J.
  CHECK_FALSE(p.full.rope_is_neox_style);
  CHECK_FALSE(p.swa.rope_is_neox_style);
  // The polarity that DOES differ is the INDEXER's, which is trap 3's point:
  // at DeepSeek-V3.2's absent-key default the indexer rope would be NeoX while
  // the MLA rope beside it is GPT-J. dots3-note makes them agree.
  CHECK(p.indexer_rope_is_neox_style() == p.full.rope_is_neox_style);
}

// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// §4 ITEM 4 — "check each field we read rather than assuming the JSON is
// complete". This is the trap with no shape and no error at all: the other five
// are wrong VALUES, this one is a wrong SOURCE.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Every key `ParseDots3NoteParams` REQUIRES, paired with a same-typed but
// wrong-typed replacement. Derived from the reader calls, not transcribed from
// the fixture: a key that stops being required drops out of the parse and this
// list goes red.
struct RequiredKey {
  const char* key;
  nlohmann::json wrong_type;
  // Whether a WRONG-TYPED value reaches dots3's own reader at all. Eleven of
  // these keys are ALSO typed on the shared `HfConfig`, so `LoadHfConfig`
  // refuses them first — a real refusal, but one that names the config PATH
  // and the JSON type rather than the key. Recording which layer refuses is
  // the honest form: papering over it would let a dots3 reader be deleted
  // while the case stayed green on somebody else's throw.
  bool dots3_names_it = true;
};

std::vector<RequiredKey> RequiredKeys() {
  return {
      {"hidden_size", "5120", /*dots3_names_it=*/false},
      {"num_hidden_layers", "46", /*dots3_names_it=*/false},
      {"vocab_size", "152064", /*dots3_names_it=*/false},
      {"intermediate_size", "13824", /*dots3_names_it=*/false},
      {"rms_norm_eps", "1e-5", /*dots3_names_it=*/false},
      {"max_position_embeddings", "524288", /*dots3_names_it=*/false},
      {"layer_types", "full_attention", /*dots3_names_it=*/false},
      {"n_routed_experts", "256"},
      {"num_experts_per_tok", "8", /*dots3_names_it=*/false},
      {"moe_intermediate_size", "1536", /*dots3_names_it=*/false},
      {"n_shared_experts", "1"},
      {"first_k_dense_replace", "1"},
      {"norm_topk_prob", "true"},
      {"scoring_func", true},
      {"topk_method", true},
      {"index_n_heads", "64"},
      {"index_head_dim", "128"},
      {"index_topk", "2048"},
      {"apply_mla_qkv_lora_rescale", "true"},
      {"num_attention_heads", "128", /*dots3_names_it=*/false},
      {"q_lora_rank", "1024"},
      {"kv_lora_rank", "512"},
      {"qk_nope_head_dim", "128"},
      {"qk_rope_head_dim", "64"},
      {"v_head_dim", "128"},
      {"rope_theta", "80000000.0", /*dots3_names_it=*/false},
      {"attention_gate_type", true},
      {"swa_num_attention_heads", "64"},
      {"swa_q_lora_rank", "1024"},
      {"swa_kv_lora_rank", "1024"},
      {"swa_qk_nope_head_dim", "192"},
      {"swa_qk_rope_head_dim", "64"},
      {"swa_v_head_dim", "128"},
      {"swa_rope_theta", "50000.0"},
      {"sliding_window_size", "513"},
      {"swa_attention_gate_type", true},
  };
}

}  // namespace

TEST_CASE("dots3-note §4 TRAP 4: an ABSENT field REFUSES BY NAME, never a default") {
  // W1 shipped this the other way round and the review's probe P1 caught it:
  // deleting `apply_mla_qkv_lora_rescale` and `swa_rope_theta` from the fixture
  // left `ParseDots3NoteParams` SUCCEEDING, with all four LoRA scales at 1.0
  // and 33 of the 46 layers rotating at 1e4 instead of 5e4. Nothing changed
  // shape, nothing threw, and §6.4 says no oracle for this model runs anywhere
  // we can reach — so no gate downstream of this one could have seen it.
  //
  // The `-fp8` sibling, a re-release, or a community re-quant that drops one
  // key is not hypothetical: the whole point of §4 is that this checkpoint's
  // `config.json` is already known to be incomplete relative to what the code
  // reads.
  const nlohmann::json fixture = FixtureConfigDoc();
  const std::vector<RequiredKey> required = RequiredKeys();
  // A guard against this list quietly emptying out.
  REQUIRE(required.size() == 36);

  for (const RequiredKey& r : required) {
    const std::string key(r.key);
    CAPTURE(key);
    REQUIRE_MESSAGE(fixture.contains(r.key),
                    "the released config.json no longer carries " << r.key
                        << " — re-derive what upstream does with it absent "
                           "before changing this list");
    nlohmann::json doc = fixture;
    doc.erase(r.key);
    TempConfig cfg(doc);
    // It must throw, and it must NAME THE KEY: a refusal that only says
    // "invalid config" sends the next reader to the wrong file.
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains(r.key), std::runtime_error);
  }
}

TEST_CASE("dots3-note §4 TRAP 4: a WRONG-TYPED field REFUSES BY NAME too") {
  // The same hole through a different door. W1's `RawBool`/`RawDouble`
  // swallowed a wrong JSON type into the identical silent fallback, so a
  // publisher writing `"apply_mla_qkv_lora_rescale": "true"` — a string, which
  // is truthy in several languages — got `false`.
  const nlohmann::json fixture = FixtureConfigDoc();
  for (const RequiredKey& r : RequiredKeys()) {
    const std::string key(r.key);
    CAPTURE(key);
    nlohmann::json doc = fixture;
    doc[r.key] = r.wrong_type;
    TempConfig cfg(doc);
    // Every one must REFUSE. What differs is which layer gets there first.
    CHECK_THROWS_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                    std::runtime_error);
    if (r.dots3_names_it) {
      CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                           doctest::Contains(r.key), std::runtime_error);
    } else {
      // The shared container reader owns this key too and refuses on type
      // before dots3 sees it. Its message names the config path and the JSON
      // type, not the key, so assert THAT rather than pretending otherwise.
      CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                           doctest::Contains("hf_config: bad field type"),
                           std::runtime_error);
    }
  }
  // ...and the OPTIONAL fields are type-strict as well, even though an absent
  // one legitimately takes an upstream default. Upstream would carry a string
  // into arithmetic, not substitute its `setdefault`.
  for (const std::string key : {"n_group", "topk_group",
                                "num_nextn_predict_layers", "moe_layer_freq"}) {
    CAPTURE(key);
    nlohmann::json doc = fixture;
    doc[key] = "1";
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains(key), std::runtime_error);
  }
  for (const std::string key : {"indexer_rope_interleave"}) {
    CAPTURE(key);
    nlohmann::json doc = fixture;
    doc[key] = "true";
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains(key), std::runtime_error);
  }
  for (const std::string key : {"routed_scaling_factor"}) {
    CAPTURE(key);
    nlohmann::json doc = fixture;
    doc[key] = "1.0";
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains(key), std::runtime_error);
  }
}

TEST_CASE("dots3-note §4 TRAP 4: the OPTIONAL four keep their upstream defaults") {
  // The other half of the obligation, and the reason this is not "require
  // everything". Exactly four keys are `Dots3NoteConfig.__init__` setdefaults
  // and two more are `model.py` getattrs; those six must still parse when
  // absent, taking the value upstream would. Requiring them would refuse the
  // real released checkpoint, which carries none of the four.
  const nlohmann::json fixture = FixtureConfigDoc();
  for (const std::string key : {"n_group", "topk_group",
                                "indexer_rope_interleave",
                                "num_nextn_predict_layers", "moe_layer_freq",
                                "routed_scaling_factor", "tie_word_embeddings"}) {
    CAPTURE(key);
    nlohmann::json doc = fixture;
    doc.erase(key);  // four of the seven are absent already
    TempConfig cfg(doc);
    CHECK_NOTHROW((void)ParseDots3NoteParams(LoadHfConfig(cfg.path())));
  }
  // And the defaults they take are upstream's, not zero.
  nlohmann::json doc = fixture;
  doc.erase("moe_layer_freq");
  doc.erase("routed_scaling_factor");
  TempConfig cfg(doc);
  const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
  CHECK(p.moe_layer_freq == 1);                                // model.py:513
  CHECK(p.routed_scaling_factor == doctest::Approx(1.0));      // model.py:546
  CHECK(p.n_group == 1);
  CHECK(p.topk_group == 1);
  CHECK(p.indexer_rope_interleave);
  CHECK(p.num_nextn_predict_layers == 1);
}

TEST_CASE("dots3-note §4 TRAP 4: a WRAPPED config layout refuses instead of reading defaults") {
  // A layout that silently deserializes to all-defaults is a wrong-shaped model
  // with no error (porting-a-model.md §1). dots3-note's released config is
  // flat; a `text_config` wrapper would put every field one level down, where
  // the top-level read finds nothing.
  nlohmann::json doc;
  doc["model_type"] = "dots3_note";
  doc["architectures"] = nlohmann::json::array({"Dots3NoteForCausalLM"});
  doc["hidden_size"] = 5120;
  doc["num_hidden_layers"] = 46;
  doc["text_config"] = FixtureConfigDoc();
  TempConfig cfg(doc);
  CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                       doctest::Contains("wrapper key"), std::runtime_error);
}

TEST_CASE("dots3-note config: the REAL released config.json parses") {
  const Dots3NoteParams p = FixtureParams();

  CHECK(p.hidden_size == 5120);
  CHECK(p.num_hidden_layers == 46);
  CHECK(p.vocab_size == 152064);
  CHECK(p.intermediate_size == 13824);
  CHECK(p.max_position_embeddings == 524288);
  CHECK(p.rms_norm_eps == doctest::Approx(1e-5));
  CHECK_FALSE(p.tie_word_embeddings);

  // The hybrid schedule: 13 full at 0,1,5,9,...,45 and 33 sliding.
  REQUIRE(p.layer_types.size() == 46);
  std::vector<int64_t> full_at;
  for (int64_t l = 0; l < 46; ++l) {
    if (p.kind_of(l) == Dots3NoteLayerKind::kFullAttention) full_at.push_back(l);
  }
  CHECK(full_at == std::vector<int64_t>{0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45});
  CHECK(46 - static_cast<int64_t>(full_at.size()) == 33);

  // Two geometries, and they differ in more than head count.
  CHECK(p.full.num_attention_heads == 128);
  CHECK(p.full.q_lora_rank == 1024);
  CHECK(p.full.kv_lora_rank == 512);
  CHECK(p.full.qk_nope_head_dim == 128);
  CHECK(p.full.qk_rope_head_dim == 64);
  CHECK(p.full.v_head_dim == 128);
  CHECK(p.full.sliding_window == 0);
  CHECK(p.full.has_indexer);
  CHECK(p.full.attention_gate_type == "headwise");
  CHECK(p.full.latent_row() == 576);

  CHECK(p.swa.num_attention_heads == 64);
  CHECK(p.swa.q_lora_rank == 1024);
  CHECK(p.swa.kv_lora_rank == 1024);
  CHECK(p.swa.qk_nope_head_dim == 192);
  CHECK(p.swa.qk_rope_head_dim == 64);
  CHECK(p.swa.v_head_dim == 128);
  CHECK(p.swa.sliding_window == 513);
  CHECK_FALSE(p.swa.has_indexer);
  CHECK(p.swa.attention_gate_type == "headwise");
  CHECK(p.swa.latent_row() == 1088);

  // MoE: 256 routed + 1 shared, top-8, ungrouped sigmoid noaux_tc.
  CHECK(p.n_routed_experts == 256);
  CHECK(p.num_experts_per_tok == 8);
  CHECK(p.moe_intermediate_size == 1536);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.first_k_dense_replace == 1);
  CHECK(p.moe_layer_freq == 1);
  CHECK(p.norm_topk_prob);
  CHECK(p.routed_scaling_factor == doctest::Approx(1.0));
  CHECK(p.scoring_func == "sigmoid");
  CHECK(p.topk_method == "noaux_tc");
  // `first_k_dense_replace = 1` — layer 0 is dense, every later layer is MoE.
  CHECK_FALSE(p.is_moe_layer(0));
  CHECK(p.is_moe_layer(1));
  CHECK(p.is_moe_layer(45));

  // DSA lightning indexer.
  CHECK(p.index_n_heads == 64);
  CHECK(p.index_head_dim == 128);
  CHECK(p.index_topk == 2048);
}

TEST_CASE("dots3-note config: unrepresentable configs REFUSE BY NAME") {
  SUBCASE("a third attention class") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["layer_types"][3] = "linear_attention";
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains("linear_attention"),
                         std::runtime_error);
  }
  SUBCASE("a schedule that does not cover the layers") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["layer_types"].erase(doc["layer_types"].begin());
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains("layer_types has 45 entries"),
                         std::runtime_error);
  }
  SUBCASE("a softmax router") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["scoring_func"] = "softmax";
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains("scoring_func"), std::runtime_error);
  }
  SUBCASE("the non-headwise attention gate") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["swa_attention_gate_type"] = "elementwise";
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains("headwise attention gate"),
                         std::runtime_error);
  }
  SUBCASE("no sliding window") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["sliding_window_size"] = 0;
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains("sliding_window_size"),
                         std::runtime_error);
  }
  SUBCASE("a padded row narrower than the full layers' logical row") {
    nlohmann::json doc = FixtureConfigDoc();
    doc["swa_kv_lora_rank"] = 256;  // 256+64 = 320 < the full layers' 576
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS(ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains("padded physical MLA row"),
                         std::runtime_error);
  }
}

TEST_CASE("dots3-note KV: the physical MLA row is the SLIDING row, not the full one") {
  // `model.py`::Dots3NoteFullAttention:283 passes
  // `physical_head_size = swa_kv_lora_rank + swa_qk_rope_head_dim` into
  // `Dots3NotePaddedMLAAttention`, whose `get_kv_cache_spec` (:213-217) reports
  // THAT rather than its own 576. Reporting 576 here under-allocates every one
  // of the 33 sliding layers by 512 rows per token.
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const Dots3NoteParams p = ParseDots3NoteParams(config);
  CHECK(p.physical_latent_row() == 1088);
  CHECK(p.full.latent_row() == 576);

  const vllm::v1::KVCacheConfig kv =
      MakeDots3NoteKVCache(config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  const auto* mla = dynamic_cast<const vllm::v1::MLAAttentionSpec*>(
      kv.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE_MESSAGE(mla != nullptr, "the dots3-note KV group is not an MLA spec");
  CHECK_MESSAGE(mla->head_size == 1088,
                "the MLA page is " << mla->head_size
                                   << " wide; the padded row is 1088 "
                                      "(model.py:283)");
}

TEST_CASE("dots3-note enumeration: all 1614 tensors of the released slice are claimed") {
  const Dots3NoteParams p = FixtureParams();
  const nlohmann::json index = ReadJson(FixtureDir() + "/index.json");
  CHECK(index.at("total_tensors").get<int64_t>() == 1614);
  // The slice is one layer of every class the model has, out of the 38006 the
  // checkpoint ships. W2 owns the whole index and the two tower files.
  CHECK(index.at("checkpoint_total_tensors").get<int64_t>() == 38006);
  CHECK(index.at("slice_layers").get<std::vector<int64_t>>() ==
        std::vector<int64_t>{0, 1, 2, 46});

  const auto released = ExpandIndexFixture(index);
  REQUIRE(released.size() == 1614);

  const std::vector<Dots3NoteTensor> enumerated = EnumerateDots3NoteTensors(
      p, SliceLayers(), /*include_root=*/true, /*include_nextn=*/true);

  std::set<std::string> seen;
  std::vector<std::string> invented;
  for (const Dots3NoteTensor& t : enumerated) {
    CHECK_MESSAGE(!t.consumer.empty(), "no named consumer for " << t.name);
    CHECK_MESSAGE(seen.insert(t.name).second, "enumerated twice: " << t.name);
    if (released.count(t.name) == 0) invented.push_back(t.name);
  }
  CHECK_MESSAGE(invented.empty(),
                "enumerated tensors the checkpoint does not ship, first: "
                    << (invented.empty() ? std::string("-") : invented.front())
                    << " (" << invented.size() << " total)");

  std::vector<std::string> unaccounted;
  for (const auto& [name, meta] : released) {
    (void)meta;
    if (seen.count(name) == 0) unaccounted.push_back(name);
  }
  CHECK_MESSAGE(unaccounted.empty(),
                "UNCLAIMED checkpoint tensors, first: "
                    << (unaccounted.empty() ? std::string("-")
                                            : unaccounted.front())
                    << " (" << unaccounted.size() << " total)");
  CHECK(enumerated.size() == 1614);

  // The classifier the loader runs agrees, and reports zero of everything bad.
  std::vector<std::string> names;
  for (const auto& [name, meta] : released) {
    (void)meta;
    names.push_back(name);
  }
  const Dots3NoteAccounting acc =
      AccountDots3NoteTensors(p, names, SliceLayers());
  CHECK(acc.language == 1614);
  CHECK(acc.vision == 0);
  CHECK(acc.audio == 0);
  CHECK(acc.unaccounted.empty());
  CHECK(acc.missing.empty());
  CHECK(acc.duplicated.empty());
}

TEST_CASE("dots3-note enumeration: the two attention classes claim DIFFERENT tensor sets") {
  // The DSA indexer exists only on the full layers (model.py:432-434), so a
  // producer that emitted one tensor set for all 46 layers would invent five
  // tensors per sliding layer and go unnoticed by a count-only check.
  const Dots3NoteParams p = FixtureParams();
  const auto names_for = [&](int64_t layer) {
    std::set<std::string> out;
    for (const Dots3NoteTensor& t : EnumerateDots3NoteTensors(
             p, {layer}, /*include_root=*/false, /*include_nextn=*/false)) {
      out.insert(t.name.substr(std::string("model.layers.X.").size() +
                               (layer >= 10 ? 1 : 0)));
    }
    return out;
  };
  const std::set<std::string> full = names_for(1);   // full + MoE
  const std::set<std::string> sliding = names_for(2);  // sliding + MoE
  CHECK(full.size() == sliding.size() + 5);
  for (const char* t : {"self_attn.indexer.wq_b.weight",
                        "self_attn.indexer.wk.weight",
                        "self_attn.indexer.k_norm.weight",
                        "self_attn.indexer.k_norm.bias",
                        "self_attn.indexer.weights_proj.weight"}) {
    CHECK_MESSAGE(full.count(t) == 1, "the full layer does not claim " << t);
    CHECK_MESSAGE(sliding.count(t) == 0, "the sliding layer claims " << t);
  }
  // ...and both claim the two dots3-only attention tensors DeepSeek has not.
  for (const std::set<std::string>* s : {&full, &sliding}) {
    CHECK(s->count("self_attn.g_proj.weight") == 1);
    CHECK(s->count("self_attn.k_rope_only_layernorm.weight") == 1);
  }
}

TEST_CASE("dots3-note enumeration: the shapes it implies are the ones on disk") {
  // The geometry facts, checked against the RELEASED safetensors headers rather
  // than against our own arithmetic.
  const Dots3NoteParams p = FixtureParams();
  const auto released = ExpandIndexFixture(ReadJson(FixtureDir() + "/index.json"));
  const auto meta_of = [&](const std::string& name) {
    const auto it = released.find(name);
    REQUIRE_MESSAGE(it != released.end(), "absent from the index: " << name);
    return it->second;
  };
  const auto shape_of = [&](const std::string& name) {
    return meta_of(name).second;
  };

  const int64_t h = p.hidden_size;
  // FULL layer 1: 128 heads, kv_lora 512 => latent row 576.
  CHECK(shape_of("model.layers.1.self_attn.q_a_proj.weight") ==
        std::vector<int64_t>{p.full.q_lora_rank, h});
  CHECK(shape_of("model.layers.1.self_attn.q_b_proj.weight") ==
        std::vector<int64_t>{p.full.num_attention_heads * p.full.qk_head_dim(),
                             p.full.q_lora_rank});
  CHECK(shape_of("model.layers.1.self_attn.kv_a_proj_with_mqa.weight") ==
        std::vector<int64_t>{p.full.latent_row(), h});
  CHECK(shape_of("model.layers.1.self_attn.kv_b_proj.weight") ==
        std::vector<int64_t>{p.full.num_attention_heads *
                                 (p.full.qk_nope_head_dim + p.full.v_head_dim),
                             p.full.kv_lora_rank});
  CHECK(shape_of("model.layers.1.self_attn.o_proj.weight") ==
        std::vector<int64_t>{h, p.full.num_attention_heads * p.full.v_head_dim});
  // SLIDING layer 2: 64 heads, qk_nope 192, kv_lora 1024 => latent row 1088.
  CHECK(shape_of("model.layers.2.self_attn.q_b_proj.weight") ==
        std::vector<int64_t>{p.swa.num_attention_heads * p.swa.qk_head_dim(),
                             p.swa.q_lora_rank});
  CHECK(shape_of("model.layers.2.self_attn.kv_a_proj_with_mqa.weight") ==
        std::vector<int64_t>{p.swa.latent_row(), h});
  CHECK(shape_of("model.layers.2.self_attn.kv_b_proj.weight") ==
        std::vector<int64_t>{p.swa.num_attention_heads *
                                 (p.swa.qk_nope_head_dim + p.swa.v_head_dim),
                             p.swa.kv_lora_rank});
  CHECK(shape_of("model.layers.2.self_attn.o_proj.weight") ==
        std::vector<int64_t>{h, p.swa.num_attention_heads * p.swa.v_head_dim});

  // The HEADWISE gate is one scalar per head, so its width is the head count —
  // which is how the shapes tell the two classes apart (model.py:295-300).
  CHECK(shape_of("model.layers.1.self_attn.g_proj.weight") ==
        std::vector<int64_t>{p.full.num_attention_heads, h});
  CHECK(shape_of("model.layers.2.self_attn.g_proj.weight") ==
        std::vector<int64_t>{p.swa.num_attention_heads, h});
  // The extra RMSNorm over the 64-wide rope-only k slice (model.py:299-301).
  CHECK(shape_of("model.layers.1.self_attn.k_rope_only_layernorm.weight") ==
        std::vector<int64_t>{p.full.qk_rope_head_dim});

  // The DSA indexer: 64 heads x 128 dim.
  CHECK(shape_of("model.layers.1.self_attn.indexer.wq_b.weight") ==
        std::vector<int64_t>{p.index_n_heads * p.index_head_dim,
                             p.full.q_lora_rank});
  CHECK(shape_of("model.layers.1.self_attn.indexer.wk.weight") ==
        std::vector<int64_t>{p.index_head_dim, h});
  CHECK(shape_of("model.layers.1.self_attn.indexer.weights_proj.weight") ==
        std::vector<int64_t>{p.index_n_heads, h});

  // MoE: 256 routed experts at 1536, 1 shared at the same width.
  CHECK(shape_of("model.layers.1.mlp.gate.weight") ==
        std::vector<int64_t>{p.n_routed_experts, h});
  CHECK(shape_of("model.layers.1.mlp.experts.0.gate_proj.weight") ==
        std::vector<int64_t>{p.moe_intermediate_size, h});
  CHECK(shape_of("model.layers.1.mlp.experts.255.down_proj.weight") ==
        std::vector<int64_t>{h, p.moe_intermediate_size});
  CHECK(shape_of("model.layers.1.mlp.shared_experts.gate_proj.weight") ==
        std::vector<int64_t>{p.moe_intermediate_size * p.n_shared_experts, h});
  // Layer 0 is the one dense layer (first_k_dense_replace = 1).
  CHECK(shape_of("model.layers.0.mlp.gate_proj.weight") ==
        std::vector<int64_t>{p.intermediate_size, h});

  // ★ MEMORY FORMAT. The whole language tower is BF16 except ONE family: the
  // noaux_tc per-expert bias ships F32. porting.md requires this to be checked
  // deliberately, because a loader that assumed one dtype for the checkpoint
  // would misread it and a token gate could not see the difference.
  CHECK(meta_of("model.layers.1.mlp.gate.e_score_correction_bias").first == "F32");
  CHECK(meta_of("model.layers.1.mlp.gate.weight").first == "BF16");
  CHECK(meta_of("model.embed_tokens.weight").first == "BF16");
  int64_t f32 = 0;
  for (const auto& [name, meta] : released) {
    if (meta.first != "BF16") {
      CHECK_MESSAGE(meta.first == "F32", "unexpected dtype " << meta.first
                                                             << " on " << name);
      CHECK_MESSAGE(name.find(".mlp.gate.e_score_correction_bias") !=
                        std::string::npos,
                    "a SECOND non-BF16 family appeared: " << name);
      ++f32;
    }
  }
  CHECK(f32 == 2);  // one per MoE layer in the slice (layers 1 and 2)

  // The nextn block carries the SLIDING geometry and a DENSE MLP — a fact the
  // checkpoint answers and upstream does not, because `layer_types` has no
  // entry at index 46 (model.py:503).
  CHECK(shape_of("model.layers.46.self_attn.q_b_proj.weight") ==
        std::vector<int64_t>{p.swa.num_attention_heads * p.swa.qk_head_dim(),
                             p.swa.q_lora_rank});
  CHECK(shape_of("model.layers.46.mlp.gate_proj.weight") ==
        std::vector<int64_t>{p.intermediate_size, h});
  // eh_proj folds [embedding ; hidden] back to hidden (mtp.py:40-45).
  CHECK(shape_of("model.layers.46.eh_proj.weight") ==
        std::vector<int64_t>{h, 2 * h});
}

// ─────────────────────────────────────────────────────────────────────────────
// The unported arms, and the refusal driven through the REAL loaded model.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// A synthetic safetensors checkpoint carrying the LANGUAGE tower's names at
// scalar shapes. It exists so the registry can produce a REAL
// `Dots3NoteLoadedModel` — the refusal below must be reached through the object
// the factory actually returns, never through a fabricated `LoadedModel`
// subclass. Downcasting a look-alike is undefined behaviour and UBSan's vptr
// check reports it; that is the defect #730/#784 removed from the NemotronH
// gate, and it must not be reintroduced here.
struct StEntry {
  std::string name;
  std::vector<uint8_t> bytes{0, 0};  // one BF16 element
};

void WriteSafetensors(const std::vector<StEntry>& entries,
                      const std::string& path) {
  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  for (const StEntry& e : entries) {
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"BF16\",\"shape\":[1],";
    header += "\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  FILE* fh = std::fopen(path.c_str(), "wb");
  REQUIRE(fh != nullptr);
  const uint64_t n = header.size();
  std::fwrite(&n, sizeof(n), 1, fh);
  std::fwrite(header.data(), 1, header.size(), fh);
  for (const StEntry& e : entries) {
    std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
  }
  std::fclose(fh);
}

class TempCheckpoint {
 public:
  explicit TempCheckpoint(const std::vector<std::string>& names) {
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
           ("dots3_note_ckpt_" + std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    std::vector<StEntry> entries;
    entries.reserve(names.size());
    for (const std::string& n : names) entries.push_back(StEntry{n, {0, 0}});
    WriteSafetensors(entries, (dir_ / "model.safetensors").string());
  }
  ~TempCheckpoint() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string file() const { return (dir_ / "model.safetensors").string(); }

 private:
  std::filesystem::path dir_;
};

// Every language-tower name the released checkpoint ships, derived from the
// params rather than transcribed.
std::vector<std::string> AllLanguageNames(const Dots3NoteParams& p) {
  std::vector<std::string> names;
  for (const Dots3NoteTensor& t : EnumerateDots3NoteTensors(p)) {
    names.push_back(t.name);
  }
  return names;
}

}  // namespace

TEST_CASE("dots3-note: the unported arms REFUSE BY NAME") {
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);

  SUBCASE("a safetensors source with NO SHARDS refuses at LOAD, by name") {
    const std::vector<vllm::SafetensorsFile> none;
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(none);
    CHECK_THROWS_WITH_AS(reg.factory->load_weights(reg, config, source),
                         doctest::Contains("no shards"), std::runtime_error);
  }

  SUBCASE("an UNCLAIMED tensor refuses at LOAD, naming the tensor") {
    // A weight nobody loads reads as zeros and renders. The accounting pass is
    // what stops that, so it has to fail on a name it does not know.
    const Dots3NoteParams p = ParseDots3NoteParams(config);
    std::vector<std::string> names = AllLanguageNames(p);
    names.push_back("model.layers.3.self_attn.mystery_proj.weight");
    TempCheckpoint ckpt(names);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    CHECK_THROWS_WITH_AS(reg.factory->load_weights(reg, config, source),
                         doctest::Contains("mystery_proj"), std::runtime_error);
  }

  SUBCASE("a MISSING enumerated tensor refuses at LOAD, naming the tensor") {
    const Dots3NoteParams p = ParseDots3NoteParams(config);
    std::vector<std::string> names = AllLanguageNames(p);
    const std::string dropped = "model.layers.5.self_attn.g_proj.weight";
    names.erase(std::remove(names.begin(), names.end(), dropped), names.end());
    TempCheckpoint ckpt(names);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    CHECK_THROWS_WITH_AS(reg.factory->load_weights(reg, config, source),
                         doctest::Contains(dropped), std::runtime_error);
  }

  SUBCASE("the two towers are NAMED deferrals, not unaccounted tensors") {
    // WHAT "NAMED DEFERRAL" HAS TO MEAN, and what this case used to prove
    // instead. It asserted only that the load did not throw — so a classifier
    // that counted all 2625 tower tensors as LANGUAGE passed it unchanged
    // (#1805 review, mutation M15). "Not unaccounted" and "claimed by the
    // language tower" are opposite answers, and only one of them is true.
    //
    // The damage lands at W2, which extends this exact classifier over the
    // whole index: the tower tensors would fold into the language count, 35381
    // would silently become 38006, and "100% accounted" would start covering
    // weights nobody loads. So the buckets are asserted here, one by one.
    const Dots3NoteParams p = ParseDots3NoteParams(config);
    const std::vector<std::string> language = AllLanguageNames(p);
    std::vector<std::string> names = language;
    const std::vector<std::string> vision{
        "vision_encoder.blocks.0.attn.qkv.weight",
        "vision_encoder.blocks.0.mlp.fc1.weight",
        "vision_encoder.patch_embed.proj.weight",
    };
    const std::vector<std::string> audio{
        "audio_encoder.dots_encoder.speech_encoder.conv2d1.weight",
        "audio_encoder.audio_adapter.proj.0.weight",
    };
    names.insert(names.end(), vision.begin(), vision.end());
    names.insert(names.end(), audio.begin(), audio.end());

    // Every backbone layer, because the loader accounts over the whole
    // backbone rather than over W1's committed slice.
    std::vector<int64_t> backbone;
    for (int64_t l = 0; l < p.num_hidden_layers; ++l) backbone.push_back(l);
    const Dots3NoteAccounting acc =
        AccountDots3NoteTensors(p, names, backbone);

    // Each bucket by count, not by "nothing was left over": the three counts
    // have to add up the ONE way that says the towers are deferred rather than
    // claimed.
    CHECK(acc.language == static_cast<int64_t>(language.size()));
    CHECK(acc.language == 35381);
    CHECK_MESSAGE(acc.vision == static_cast<int64_t>(vision.size()),
                  "vision tensors are not landing in the vision bucket — "
                  "acc.vision=" << acc.vision << ", acc.language="
                                << acc.language);
    CHECK_MESSAGE(acc.audio == static_cast<int64_t>(audio.size()),
                  "audio tensors are not landing in the audio bucket — "
                  "acc.audio=" << acc.audio << ", acc.language="
                               << acc.language);
    CHECK(acc.total() == static_cast<int64_t>(names.size()));
    CHECK(acc.unaccounted.empty());
    CHECK(acc.missing.empty());
    CHECK(acc.duplicated.empty());
    // ...and a tower tensor is NEVER claimed by a language-tower consumer, so
    // no enumerated name may collide with one.
    std::set<std::string> claimed;
    for (const Dots3NoteTensor& t : EnumerateDots3NoteTensors(p)) {
      claimed.insert(t.name);
    }
    for (const std::string& n : vision) CHECK(claimed.count(n) == 0);
    for (const std::string& n : audio) CHECK(claimed.count(n) == 0);

    // The production entry point still accepts the same checkpoint.
    TempCheckpoint ckpt(names);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    std::unique_ptr<vllm::LoadedModel> model;
    REQUIRE_NOTHROW(model = reg.factory->load_weights(reg, config, source));
    REQUIRE(model != nullptr);
  }

  SUBCASE("GGUF k-quants are OWED (W9), never silently dequantized") {
    // llama.cpp has no `dots3_note` architecture, so the converter is ours to
    // write and there is no quant-matched llama.cpp bar for this row.
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors({});
    vllm::ModelSource gguf = source;
    gguf.kind = vllm::ModelSource::Kind::kGguf;
    gguf.safetensors = nullptr;
    CHECK_THROWS_WITH_AS(reg.factory->load_weights(reg, config, gguf),
                         doctest::Contains("GGUF k-quants are not ported"),
                         std::runtime_error);
  }
}

TEST_CASE("dots3-note: the forward REFUSES BY NAME through the REAL loaded model") {
  // THE OBJECT UNDER TEST IS THE ONE THE FACTORY RETURNS. Reaching this refusal
  // by handing the entry point a hand-rolled `LoadedModel` subclass would be
  // undefined behaviour the moment the forward opened it, whether the open is a
  // `static_cast` (it is not — see `ForwardDots3NoteForCausalLM`) or a checked
  // `ModelAs`. #730/#784 is exactly that mistake, made once already.
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  const Dots3NoteParams p = ParseDots3NoteParams(config);

  TempCheckpoint ckpt(AllLanguageNames(p));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model =
      reg.factory->load_weights(reg, config, source);
  REQUIRE(model != nullptr);
  // 100% accounted: 35381 language-tower tensors, zero unaccounted. (The real
  // checkpoint adds 2625 tower tensors on top, for 38006.)
  CHECK(EnumerateDots3NoteTensors(p).size() == 35381);

  const std::vector<int32_t> token_ids{0};
  const std::vector<int32_t> positions{0};
  const std::vector<int32_t> logits_indices{0};
  const vllm::v1::CommonAttentionMetadata attn_meta{};
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::ModelForwardInput input{.token_ids = token_ids,
                                      .positions = positions,
                                      .attn_meta = attn_meta,
                                      .gdn_meta = gdn_meta,
                                      .attn_kv = attn_kv,
                                      .gdn_state = gdn_state,
                                      .config = config,
                                      .queue = queue,
                                      .logits_indices = logits_indices,
                                      .num_reqs = 1};

  // It must THROW rather than return zero logits: a zero-logit step samples a
  // plausible-looking garbage token, and §6.4 says there is no oracle anywhere
  // that could catch it.
  CHECK_THROWS_AS(reg.factory->forward(*model, input), std::runtime_error);
  // It must say so in dots3-note's OWN name, so a refusal from a shared helper
  // cannot be mistaken for this one...
  CHECK_THROWS_WITH_AS(reg.factory->forward(*model, input),
                       doctest::Contains("Dots3NoteForCausalLM forward"),
                       std::runtime_error);
  // ...name the missing piece rather than only failing...
  CHECK_THROWS_WITH_AS(reg.factory->forward(*model, input),
                       doctest::Contains("sliding-window MLA"),
                       std::runtime_error);
  // ...and point at the record that owns the brick.
  CHECK_THROWS_WITH_AS(reg.factory->forward(*model, input),
                       doctest::Contains("dots3-note.md"), std::runtime_error);
}

// A FOREIGN `LoadedModel` handed to the dots3-note forward entry point: a
// complete, well-formed object of a type that simply is not
// `Dots3NoteLoadedModel` — the shape a caller produces by pairing one
// architecture's `ModelRegistration` with another architecture's model, which
// is what `ModelRegistry::Forward` cannot check for its callers.
//
// This is NOT the stub #784 removed, and it must never be used to reach the
// refusal above. That stub was wrong because the forward DEREFERENCED it. This
// one asserts the opposite guarantee: that the entry point establishes the
// dynamic type BEFORE any member call and refuses a mismatch by name.
namespace {
class ForeignLoadedModel final : public vllm::LoadedModel {
 public:
  explicit ForeignLoadedModel(const vllm::ModelRegistration& registration)
      : vllm::LoadedModel(registration) {}
};
}  // namespace

TEST_CASE("dots3-note: the forward entry point REFUSES a foreign LoadedModel by name") {
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);

  ForeignLoadedModel foreign(reg);
  const std::vector<int32_t> token_ids{0};
  const std::vector<int32_t> positions{0};
  const std::vector<int32_t> logits_indices{0};
  const vllm::v1::CommonAttentionMetadata attn_meta{};
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::ModelForwardInput input{.token_ids = token_ids,
                                      .positions = positions,
                                      .attn_meta = attn_meta,
                                      .gdn_meta = gdn_meta,
                                      .attn_kv = attn_kv,
                                      .gdn_state = gdn_state,
                                      .config = config,
                                      .queue = queue,
                                      .logits_indices = logits_indices,
                                      .num_reqs = 1};

  CHECK_THROWS_WITH_AS(reg.factory->forward(foreign, input),
                       doctest::Contains("Dots3NoteForCausalLM"),
                       std::runtime_error);
  // ...and it must name what actually went wrong, rather than blaming the
  // unported forward the real object would have failed on.
  CHECK_THROWS_WITH_AS(reg.factory->forward(foreign, input),
                       doctest::Contains("was not produced by"),
                       std::runtime_error);
}
