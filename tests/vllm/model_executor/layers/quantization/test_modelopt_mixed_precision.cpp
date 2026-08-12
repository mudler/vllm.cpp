// CPU unit tests for the ModelOpt MIXED_PRECISION per-module quant-algo
// resolver (modelopt_mixed_precision.h). No weights, no GPU, no oracle
// process: the whole capability is config parsing plus string resolution, so
// the always-on gate runs anywhere.
//
// Row MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm W1, issue #517, spec
// .agents/specs/nemotron-h-model.md.
//
// UPSTREAM (ported FROM) @ 5559679229bc961848b121ccdeaa8fa5d79bec98:
//   vllm/model_executor/layers/quantization/modelopt.py:2279-2410
//     ModelOptMixedPrecisionConfig + _from_config
//   vllm/model_executor/layers/quantization/modelopt.py:2412-2487
//     _resolve_quant_algo — the FIVE strategies, in order
//   vllm/model_executor/layers/quantization/modelopt.py:2491-2505
//     _quantized_layer_prefix_candidates
//   vllm/model_executor/layers/quantization/modelopt.py:145-181
//     ModelOptQuantConfigBase.is_layer_excluded
//   vllm/model_executor/layers/quantization/utils/quant_utils.py:510-572
//     is_layer_skipped
//
// Every expectation below was cross-checked against a line-by-line Python
// transcription of those upstream functions, executed on the REAL 1.3 MB
// config.json of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 @29f2d174.
#include <doctest/doctest.h>

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/quantization/modelopt_mixed_precision.h"

using vllm::layers::modelopt::MixedPrecisionConfig;
using vllm::layers::modelopt::ModuleQuant;
using vllm::layers::modelopt::PackedModulesMapping;
using vllm::layers::modelopt::QuantAlgo;
using vllm::layers::modelopt::Resolution;

namespace {

std::string FixturePath() {
  return std::string(MODELOPT_MIXED_FIXTURE_DIR) + "/curated_config.json";
}

// ordered_json, not json: upstream iterates a Python dict, whose order is
// INSERTION order. nlohmann::json sorts object keys lexicographically, which
// would silently change which entry "the first NVFP4-family entry" (group_size
// seeding, modelopt.py:2360-2372) and the prefix scans (:2450, :2458) return.
nlohmann::ordered_json LoadFixture() {
  std::ifstream f(FixturePath());
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", FixturePath());
  nlohmann::ordered_json doc = nlohmann::ordered_json::parse(f);
  return doc.at("quantization_config");
}

MixedPrecisionConfig Curated() {
  return MixedPrecisionConfig::Parse(LoadFixture());
}

// vLLM registers packed_modules_mapping per MODEL. Strategy 2 is dead without
// one, so the packed cases install the standard attention/MLP mapping.
PackedModulesMapping StandardPacked() {
  return PackedModulesMapping{
      {"qkv_proj", {"q_proj", "k_proj", "v_proj"}},
      {"gate_up_proj", {"gate_proj", "up_proj"}},
  };
}

}  // namespace

// --- detection + parse (modelopt.py:2330-2410) -----------------------------

TEST_CASE("modelopt-mixed: MIXED_PRECISION is detected, other algos are not") {
  const nlohmann::ordered_json qc = LoadFixture();
  CHECK(MixedPrecisionConfig::IsMixedPrecision(qc));

  nlohmann::ordered_json plain = qc;
  plain["quant_algo"] = "NVFP4";
  CHECK_FALSE(MixedPrecisionConfig::IsMixedPrecision(plain));

  nlohmann::ordered_json other_vendor = qc;
  other_vendor["quant_method"] = "compressed-tensors";
  CHECK_FALSE(MixedPrecisionConfig::IsMixedPrecision(other_vendor));

  // modelopt.py:256 lower-cases and uses startswith("modelopt").
  nlohmann::ordered_json cased = qc;
  cased["quant_method"] = "ModelOpt";
  CHECK(MixedPrecisionConfig::IsMixedPrecision(cased));

  // modelopt.py:2340-2347 also accepts the legacy nested shape. Built with the
  // REAL top-level key set of an hf_quant_config.json — {producer,
  // quantization} — which names NO quant_method anywhere in the file.
  nlohmann::ordered_json nested;
  nested["producer"] = nlohmann::ordered_json::object();
  nested["producer"]["name"] = "modelopt";
  nested["quantization"] = nlohmann::ordered_json::object();
  nested["quantization"]["quant_algo"] = "MIXED_PRECISION";
  nested["quantization"]["quantized_layers"] = qc.at("quantized_layers");

  // PARSING mirrors from_config (modelopt.py:282-367), which dispatches on the
  // SHAPE and never reads quant_method. Gating Parse on quant_method — the
  // precondition that belongs to the SELECTION hook :245-263 — refused the
  // driver checkpoint's own hf_quant_config.json outright.
  CHECK(MixedPrecisionConfig::Parse(nested).num_quantized_layers() ==
        qc.at("quantized_layers").size());
  // DETECTION does keep it: _extract_modelopt_quant_algo returns None for a
  // config that does not name modelopt, so the override hook must not claim it.
  CHECK_FALSE(MixedPrecisionConfig::IsMixedPrecision(nested));
  nested["quant_method"] = "modelopt";
  CHECK(MixedPrecisionConfig::IsMixedPrecision(nested));
  CHECK(MixedPrecisionConfig::Parse(nested).num_quantized_layers() ==
        qc.at("quantized_layers").size());
}

TEST_CASE("modelopt-mixed: parse reads the fixture's shape") {
  const MixedPrecisionConfig c = Curated();
  CHECK(c.num_quantized_layers() == 32);
  CHECK(c.exclude_modules().size() == 14);
  // modelopt.py:2360-2372: no top-level group_size, so it is SEEDED from the
  // first NVFP4-family entry.
  CHECK(c.group_size() == 16);
  // kv_cache_scheme {type:"float", num_bits:8} -> "FP8" (modelopt.py:2306-2314)
  CHECK(c.kv_cache_quant_algo() == "FP8");
}

TEST_CASE("modelopt-mixed: an empty quantized_layers map is refused") {
  nlohmann::ordered_json qc = LoadFixture();
  qc["quantized_layers"] = nlohmann::ordered_json::object();
  CHECK_THROWS_AS(MixedPrecisionConfig::Parse(qc), std::invalid_argument);
  qc.erase("quantized_layers");
  CHECK_THROWS_AS(MixedPrecisionConfig::Parse(qc), std::invalid_argument);
}

// --- strategy 1: direct lookup (modelopt.py:2424-2427) ---------------------

TEST_CASE("modelopt-mixed: strategy 1 direct lookup, both real algos") {
  const MixedPrecisionConfig c = Curated();

  const ModuleQuant in_proj = c.Resolve("backbone.layers.0.mixer.in_proj");
  CHECK(in_proj.algo == QuantAlgo::kFp8);
  CHECK(in_proj.how == Resolution::kDirect);
  CHECK(in_proj.group_size == 0);  // FP8 is not group-quantized

  const ModuleQuant expert = c.Resolve("backbone.layers.1.mixer.experts.0.up_proj");
  CHECK(expert.algo == QuantAlgo::kW4A16Nvfp4);
  CHECK(expert.how == Resolution::kDirect);
  CHECK(expert.group_size == 16);

  // Not listed anywhere and not ignored -> unquantized, NOT an error.
  const ModuleQuant absent = c.Resolve("backbone.layers.3.mixer.nonexistent");
  CHECK(absent.algo == QuantAlgo::kUnquantized);
  CHECK(absent.how == Resolution::kUnlisted);
  CHECK_FALSE(absent.Quantized());
}

TEST_CASE("modelopt-mixed: prefix candidates — bare lm_head and the lm swap") {
  const MixedPrecisionConfig c = Curated();

  // The real map stores a BARE "lm_head"; our loader's prefix is "model.lm_head".
  CHECK(c.Resolve("lm_head").algo == QuantAlgo::kW4A16Nvfp4);
  const ModuleQuant head = c.Resolve("model.lm_head");
  CHECK(head.algo == QuantAlgo::kW4A16Nvfp4);
  CHECK(head.how == Resolution::kDirect);
  CHECK(head.group_size == 16);
  // The suffix rule is ".lm_head", so a name merely CONTAINING it must miss.
  CHECK(c.Resolve("model.not_lm_head").algo == QuantAlgo::kUnquantized);

  // modelopt.py:2496-2503, both directions of the language_model swap. Each
  // direction needs its OWN map entry: with only one, whichever branch is not
  // exercised can be deleted with the suite still green (my mutation run
  // proved exactly that, so the second entry is here because of it).
  CHECK(c.Resolve("language_model.model.layers.0.mlp.down_proj").algo ==
        QuantAlgo::kFp8);  // direct hit, no swap needed
  CHECK(c.Resolve("model.language_model.layers.0.mlp.down_proj").how ==
        Resolution::kDirect);  // model.language_model -> language_model.model
  CHECK(c.Resolve("model.language_model.layers.9.mlp.up_proj").algo ==
        QuantAlgo::kFp8);  // direct hit the other way round
  CHECK(c.Resolve("language_model.model.layers.9.mlp.up_proj").how ==
        Resolution::kDirect);  // language_model.model -> model.language_model
  CHECK(c.Resolve("language_model.model.layers.9.mlp.up_proj").algo ==
        QuantAlgo::kFp8);
}

// --- strategy 2: packed / fused lookup (modelopt.py:2429-2447) -------------

TEST_CASE("modelopt-mixed: strategy 2 packed lookup unfuses qkv_proj") {
  MixedPrecisionConfig c = Curated();
  c.SetPackedModulesMapping(StandardPacked());

  const ModuleQuant qkv = c.Resolve("synthetic.layers.1.self_attn.qkv_proj");
  CHECK(qkv.algo == QuantAlgo::kFp8);
  CHECK(qkv.how == Resolution::kPacked);

  // Without the mapping registered, strategy 2 cannot fire; strategies 3 and 4
  // do not apply to qkv_proj either, so strategy 5 catches it instead.
  const MixedPrecisionConfig unmapped = Curated();
  const ModuleQuant fallback =
      unmapped.Resolve("synthetic.layers.1.self_attn.qkv_proj");
  CHECK(fallback.algo == QuantAlgo::kFp8);
  CHECK(fallback.how == Resolution::kFusedShards);
}

TEST_CASE("modelopt-mixed: strategy 2 RAISES when fused shards disagree") {
  MixedPrecisionConfig c = Curated();
  c.SetPackedModulesMapping(StandardPacked());

  // q,v are FP8 and k is W4A16_NVFP4. Upstream raises ValueError rather than
  // picking one (modelopt.py:2444-2447). Returning ANY value here is the
  // silent-wrong-bytes failure a token gate cannot see.
  CHECK_THROWS_AS(c.Resolve("synthetic.layers.2.self_attn.qkv_proj"),
                  std::invalid_argument);
  CHECK_THROWS_WITH_AS(c.Resolve("synthetic.layers.2.self_attn.qkv_proj"),
                       doctest::Contains("synthetic.layers.2.self_attn.qkv_proj"),
                       std::invalid_argument);
}

TEST_CASE("modelopt-mixed: strategy 2's algo set spans ALL base candidates") {
  MixedPrecisionConfig c = Curated();
  c.SetPackedModulesMapping(StandardPacked());

  // Strategy 2 (modelopt.py:2429-2447) builds ONE algo set over every base
  // prefix candidate and decides once. Strategy 5 (:2463-2486) rebuilds the set
  // per candidate and returns on the first candidate that yields exactly one.
  // The code carries a comment saying so, and every other fused entry in this
  // fixture has a SINGLE prefix candidate — where the two are indistinguishable,
  // so flattening strategy 2 into strategy 5's shape survives them all.
  //
  // Here the shards are split across the two `language_model` spellings
  // (:2496-2503): q_proj=FP8 under "language_model.model.", k_proj=W4A16_NVFP4
  // under "model.language_model.". The union is {FP8, W4A16_NVFP4} and must
  // RAISE. Rebuilding per candidate would see {FP8} on the first spelling,
  // return it, and never look at the second — a fused layer half loaded as 4-bit
  // and half as 8-bit, silently.
  CHECK_THROWS_AS(c.Resolve("language_model.model.layers.5.self_attn.qkv_proj"),
                  std::invalid_argument);
  CHECK_THROWS_WITH_AS(
      c.Resolve("language_model.model.layers.5.self_attn.qkv_proj"),
      doctest::Contains("W4A16_NVFP4"), std::invalid_argument);
  // ...and from the other spelling, where the candidate ORDER is reversed.
  CHECK_THROWS_AS(c.Resolve("model.language_model.layers.5.self_attn.qkv_proj"),
                  std::invalid_argument);
}

// --- strategy 3: prefix lookup (modelopt.py:2449-2453) ---------------------

TEST_CASE("modelopt-mixed: strategy 3 resolves a parent module by prefix") {
  const MixedPrecisionConfig c = Curated();

  // The routed-expert container: children are "...experts.<i>.<proj>".
  const ModuleQuant experts = c.Resolve("backbone.layers.1.mixer.experts");
  CHECK(experts.algo == QuantAlgo::kW4A16Nvfp4);
  CHECK(experts.how == Resolution::kPrefix);
  CHECK(experts.group_size == 16);

  const ModuleQuant shared = c.Resolve("backbone.layers.1.mixer.shared_experts");
  CHECK(shared.algo == QuantAlgo::kW4A16Nvfp4);
  CHECK(shared.how == Resolution::kPrefix);

  // The scan is on `prefix + "."`, so a bare string prefix must NOT match.
  CHECK(c.Resolve("backbone.layers.1.mixer.experts.").algo ==
        QuantAlgo::kUnquantized);
  CHECK(c.Resolve("backbone.layers.1.mixer.expert").algo ==
        QuantAlgo::kUnquantized);
}

TEST_CASE("modelopt-mixed: strategy 3 returns the FIRST child, not the last") {
  const MixedPrecisionConfig c = Curated();

  // modelopt.py:2449-2453 iterates `quantized_layers` and returns on the first
  // key that startswith(candidate + "."). Every OTHER parent in this fixture
  // has children that agree, or whose first and last child agree
  // (synthetic.layers.2.self_attn is q/k/v = FP8/W4A16/FP8), so "return the
  // LAST match" survives them. synthetic.layers.8.moe is a_proj=FP8 then
  // b_proj=W4A16_NVFP4: first and last differ.
  const ModuleQuant moe = c.Resolve("synthetic.layers.8.moe");
  CHECK(moe.algo == QuantAlgo::kFp8);
  CHECK(moe.how == Resolution::kPrefix);
  CHECK(moe.group_size == 0);

  // Strategy 4 (:2455-2461) scans the same way and owes the same guarantee.
  const ModuleQuant experts = c.Resolve("synthetic.layers.8.moe.experts");
  CHECK(experts.algo == QuantAlgo::kFp8);
  CHECK(experts.how == Resolution::kExpertsParent);
  CHECK(experts.group_size == 0);
}

TEST_CASE("modelopt-mixed: INSERTION order, not lexicographic, decides the scans") {
  // Divergence 3 in the header: `Parse` is templated so callers can hand it
  // `ordered_json`. Upstream iterates a Python dict, i.e. INSERTION order;
  // plain `nlohmann::json` sorts object keys lexicographically. That is not a
  // stylistic preference — it changes the ANSWER, and this case is what pins
  // it. synthetic.layers.2.self_attn is inserted q_proj(FP8), k_proj(W4A16),
  // v_proj(FP8); sorted, k_proj comes FIRST. So a plain-`json` load flips both
  // the strategy-3 and the strategy-4 result from FP8 to W4A16_NVFP4 — a wrong
  // scheme on a real module, which is exactly the invisible-to-a-token-gate
  // failure this row exists for.
  const MixedPrecisionConfig c = Curated();

  const ModuleQuant parent = c.Resolve("synthetic.layers.2.self_attn");
  CHECK(parent.algo == QuantAlgo::kFp8);
  CHECK(parent.how == Resolution::kPrefix);
  CHECK(parent.group_size == 0);

  const ModuleQuant experts = c.Resolve("synthetic.layers.2.self_attn.experts");
  CHECK(experts.algo == QuantAlgo::kFp8);
  CHECK(experts.how == Resolution::kExpertsParent);
  CHECK(experts.group_size == 0);
}

// --- strategy 4: the ".experts" special case (modelopt.py:2455-2461) -------

TEST_CASE("modelopt-mixed: strategy 4 maps a FusedMoE .experts prefix onto its parent") {
  const MixedPrecisionConfig c = Curated();

  // ModelOpt lists "synthetic.layers.0.moe.up_proj"; a FusedMoE layer's prefix
  // is "synthetic.layers.0.moe.experts", which strategies 1-3 all miss.
  const ModuleQuant moe = c.Resolve("synthetic.layers.0.moe.experts");
  CHECK(moe.algo == QuantAlgo::kW4A16Nvfp4);
  CHECK(moe.how == Resolution::kExpertsParent);
  CHECK(moe.group_size == 16);

  // The special case is keyed on the ".experts" SUFFIX only.
  CHECK(c.Resolve("synthetic.layers.0.moe.expertsx").algo ==
        QuantAlgo::kUnquantized);
}

// --- strategy 5: fused_projection_shards fallback (modelopt.py:2463-2486) --

TEST_CASE("modelopt-mixed: strategy 5 falls back to gate/up shard names") {
  const MixedPrecisionConfig c = Curated();  // no packed mapping registered

  const ModuleQuant gate_up = c.Resolve("synthetic.layers.3.mlp.gate_up_proj");
  CHECK(gate_up.algo == QuantAlgo::kFp8);
  CHECK(gate_up.how == Resolution::kFusedShards);

  // Only qkv_proj and gate_up_proj are in the fallback table.
  CHECK(c.Resolve("synthetic.layers.3.mlp.fc_proj").algo ==
        QuantAlgo::kUnquantized);
}

TEST_CASE("modelopt-mixed: strategy 5 RAISES when gate/up disagree") {
  const MixedPrecisionConfig c = Curated();
  CHECK_THROWS_AS(c.Resolve("synthetic.layers.4.mlp.gate_up_proj"),
                  std::invalid_argument);
}

// --- the ignore list (modelopt.py:145-181, quant_utils.py:510-572) ---------

TEST_CASE("modelopt-mixed: ignore-list entries resolve to unquantized") {
  const MixedPrecisionConfig c = Curated();

  for (const char* p : {"backbone.embeddings", "backbone.layers.0.mixer.conv1d",
                        "backbone.layers.1.mixer.gate",
                        "backbone.layers.12.mixer.q_proj",
                        "backbone.layers.12.mixer.o_proj"}) {
    CAPTURE(p);
    const ModuleQuant m = c.Resolve(p);
    CHECK(m.algo == QuantAlgo::kUnquantized);
    CHECK(m.how == Resolution::kExcluded);
    CHECK(c.IsLayerExcluded(p));
  }

  // "mtp*" is a real wildcard entry — the whole MTP head is unquantized.
  CHECK(c.IsLayerExcluded("mtp*"));
  CHECK(c.IsLayerExcluded("mtp.layers.0.mixer.up_proj"));
  CHECK(c.Resolve("mtp.layers.0.mixer.up_proj").how == Resolution::kExcluded);
  CHECK_FALSE(c.IsLayerExcluded("backbone.layers.1.mixer.experts.0.up_proj"));
}

TEST_CASE("modelopt-mixed: an ignored expert CHILD excludes its CONTAINER") {
  nlohmann::ordered_json qc = LoadFixture();
  qc["ignore"] = nlohmann::ordered_json::array(
      {"backbone.layers.1.mixer.experts.0.up_proj"});
  const MixedPrecisionConfig c = MixedPrecisionConfig::Parse(qc);

  // quant_utils.py:559-565 gives a prefix containing "experts" its own rule,
  // and the direction is the surprising one: `prefix in layer_name` — the
  // IGNORE ENTRY must contain the PREFIX, not the other way round. ModelOpt
  // lists experts per index, while a FusedMoE layer is ONE module covering all
  // of them, so naming a single expert child unquantizes the whole container.
  //
  // Nothing else can reach this branch: the exact pass wants equality, the
  // legacy substring pass tests `prefix.find(entry)` (the OPPOSITE direction,
  // which is npos here), and the wildcard pass has no wildcard to match. So
  // without this case both inverting the test to `prefix.find(entry)` and
  // deleting the branch outright leave the suite green.
  CHECK(c.IsLayerExcluded("backbone.layers.1.mixer.experts"));
  CHECK(c.Resolve("backbone.layers.1.mixer.experts").how == Resolution::kExcluded);
  CHECK(c.Resolve("backbone.layers.1.mixer.experts").algo ==
        QuantAlgo::kUnquantized);

  // The named child itself, and every deeper path the entry contains.
  CHECK(c.IsLayerExcluded("backbone.layers.1.mixer.experts.0.up_proj"));
  CHECK(c.IsLayerExcluded("backbone.layers.1.mixer.experts.0"));

  // ...but NOT a sibling the entry does not contain: expert 1 stays quantized,
  // and so does a differently named container in the same layer.
  CHECK_FALSE(c.IsLayerExcluded("backbone.layers.1.mixer.experts.1.up_proj"));
  CHECK(c.Resolve("backbone.layers.1.mixer.experts.1.up_proj").algo ==
        QuantAlgo::kW4A16Nvfp4);
  CHECK_FALSE(c.IsLayerExcluded("backbone.layers.1.mixer.shared_experts"));

  // A prefix WITHOUT "experts" never enters the branch, so the same entry does
  // not exclude it even though it is a strict prefix of that entry.
  CHECK_FALSE(c.IsLayerExcluded("backbone.layers.1.mixer"));
}

TEST_CASE("modelopt-mixed: the legacy SUBSTRING exclusion rule still applies") {
  const MixedPrecisionConfig c = Curated();
  // modelopt.py:165-174 keeps a substring rule for pre-0.39 ModelOpt exports,
  // where "ignore" carried a bare module name rather than a full path. Neither
  // the exact pass nor the wildcard pass can reach it, so without a case here
  // the whole rule deletes clean.
  CHECK(c.IsLayerExcluded("deep.legacy_substr.mixer.q_proj"));
  CHECK(c.Resolve("deep.legacy_substr.mixer.q_proj").how == Resolution::kExcluded);
  CHECK(c.IsLayerExcluded("language_model.deep.legacy_substr.mixer.o_proj"));
  // It is a SUBSTRING rule, not a segment rule: "legacy_substr.mixerX" still
  // contains "legacy_substr.mixer" and so is excluded, whereas breaking the
  // match earlier is not. Upstream's bluntness here is deliberate and mirrored.
  CHECK(c.IsLayerExcluded("deep.legacy_substr.mixerX.q_proj"));
  CHECK_FALSE(c.IsLayerExcluded("deep.legacy_substrX.mixer.q_proj"));
}

TEST_CASE("modelopt-mixed: exclusion WINS over a quantized_layers entry") {
  const MixedPrecisionConfig c = Curated();
  // synthetic.layers.7.self_attn.q_proj is in BOTH maps. get_quant_method
  // (modelopt.py:2515-2522) tests exclusion BEFORE _resolve_quant_algo, so the
  // ignore list wins. Reversing that order would quantize an excluded layer.
  const ModuleQuant m = c.Resolve("synthetic.layers.7.self_attn.q_proj");
  CHECK(m.algo == QuantAlgo::kUnquantized);
  CHECK(m.how == Resolution::kExcluded);
  // ...while its unexcluded siblings still resolve.
  CHECK(c.Resolve("synthetic.layers.7.self_attn.k_proj").algo == QuantAlgo::kFp8);
}

TEST_CASE("modelopt-mixed: a PARTIALLY excluded fused layer RAISES") {
  MixedPrecisionConfig c = Curated();
  c.SetPackedModulesMapping(StandardPacked());
  // q_proj is ignored, k/v are not (quant_utils.py:549-556).
  CHECK_THROWS_AS(c.Resolve("synthetic.layers.7.self_attn.qkv_proj"),
                  std::invalid_argument);

  // A FULLY excluded fused layer is excluded, not an error.
  const ModuleQuant all_ignored = c.Resolve("backbone.layers.12.mixer.qkv_proj");
  CHECK(all_ignored.how == Resolution::kExcluded);
}

TEST_CASE("modelopt-mixed: the ignore matcher is fnmatch, not startswith") {
  nlohmann::ordered_json qc = LoadFixture();
  qc["ignore"] = nlohmann::ordered_json::array(
      {"a.mid*.tail", "b.?.gate", "c.[0-9].gate", "d.[!0-9].gate", "e.literal["});
  const MixedPrecisionConfig c = MixedPrecisionConfig::Parse(qc);

  CHECK(c.IsLayerExcluded("a.mid.tail"));       // '*' matches empty
  CHECK(c.IsLayerExcluded("a.midXY.z.tail"));   // ...and spans separators
  CHECK_FALSE(c.IsLayerExcluded("a.mid.tail.x"));  // anchored at both ends
  CHECK(c.IsLayerExcluded("b.7.gate"));
  CHECK_FALSE(c.IsLayerExcluded("b.77.gate"));  // '?' is exactly one char
  CHECK(c.IsLayerExcluded("c.4.gate"));
  CHECK_FALSE(c.IsLayerExcluded("c.x.gate"));
  CHECK(c.IsLayerExcluded("d.x.gate"));         // negated class
  CHECK_FALSE(c.IsLayerExcluded("d.4.gate"));
  CHECK(c.IsLayerExcluded("e.literal["));       // unterminated class is literal
}

TEST_CASE("modelopt-mixed: an empty ignore list excludes nothing") {
  nlohmann::ordered_json qc = LoadFixture();
  qc["ignore"] = nlohmann::ordered_json::array();
  const MixedPrecisionConfig c = MixedPrecisionConfig::Parse(qc);
  CHECK_FALSE(c.IsLayerExcluded("backbone.embeddings"));
  CHECK(c.Resolve("synthetic.layers.7.self_attn.q_proj").algo == QuantAlgo::kFp8);
}

// --- refusal by name (spec stop condition) ---------------------------------

TEST_CASE("modelopt-mixed: an UNKNOWN quant_algo is refused BY NAME") {
  const MixedPrecisionConfig c = Curated();
  CHECK_THROWS_WITH_AS(c.Resolve("synthetic.layers.5.mixer.in_proj"),
                       doctest::Contains("AWQ_LITE"), std::runtime_error);
  CHECK_THROWS_WITH_AS(c.Resolve("synthetic.layers.5.mixer.in_proj"),
                       doctest::Contains("synthetic.layers.5.mixer.in_proj"),
                       std::runtime_error);
}

TEST_CASE("modelopt-mixed: a KNOWN but unimplemented algo is refused BY NAME") {
  const MixedPrecisionConfig c = Curated();
  // FP8_PB_WO is a real ModelOpt algo (modelopt.py:105-120) that the
  // MIXED_PRECISION consumer has no branch for. Upstream silently hands it an
  // UnquantizedLinearMethod; we refuse, because dequantizing a quantized layer
  // is numerically fine and therefore INVISIBLE to a token gate.
  CHECK_THROWS_WITH_AS(c.Resolve("synthetic.layers.6.mixer.in_proj"),
                       doctest::Contains("FP8_PB_WO"), std::runtime_error);
}

TEST_CASE("modelopt-mixed: refusal survives every resolution strategy") {
  nlohmann::ordered_json qc = LoadFixture();
  auto& ql = qc["quantized_layers"];
  ql["refuse.layers.0.moe.up_proj"] = {{"quant_algo", "AWQ_LITE"}};
  ql["refuse.layers.0.moe.down_proj"] = {{"quant_algo", "AWQ_LITE"}};
  ql["refuse.layers.1.mlp.gate_proj"] = {{"quant_algo", "AWQ_LITE"}};
  ql["refuse.layers.1.mlp.up_proj"] = {{"quant_algo", "AWQ_LITE"}};
  ql["refuse.layers.2.self_attn.q_proj"] = {{"quant_algo", "AWQ_LITE"}};
  ql["refuse.layers.2.self_attn.k_proj"] = {{"quant_algo", "AWQ_LITE"}};
  ql["refuse.layers.2.self_attn.v_proj"] = {{"quant_algo", "AWQ_LITE"}};
  MixedPrecisionConfig c = MixedPrecisionConfig::Parse(qc);
  c.SetPackedModulesMapping(StandardPacked());

  // strategy 4 (.experts), strategy 5 (gate_up), strategy 2 (packed),
  // strategy 3 (prefix). None of them may launder an unknown algo into a
  // supported one or into "unquantized".
  CHECK_THROWS_AS(c.Resolve("refuse.layers.0.moe.experts"), std::runtime_error);
  CHECK_THROWS_AS(c.Resolve("refuse.layers.1.mlp.gate_up_proj"), std::runtime_error);
  CHECK_THROWS_AS(c.Resolve("refuse.layers.2.self_attn.qkv_proj"), std::runtime_error);
  CHECK_THROWS_AS(c.Resolve("refuse.layers.0.moe"), std::runtime_error);
}

// --- algo mapping is exhaustive and case-normalised ------------------------

TEST_CASE("modelopt-mixed: every implemented algo maps, lower-case included") {
  nlohmann::ordered_json qc = LoadFixture();
  auto& ql = qc["quantized_layers"];
  ql["algo.fp8"] = {{"quant_algo", "FP8"}};
  ql["algo.nvfp4"] = {{"quant_algo", "NVFP4"}};
  ql["algo.w4a16"] = {{"quant_algo", "W4A16_NVFP4"}};
  ql["algo.mxfp8"] = {{"quant_algo", "MXFP8"}};
  ql["algo.lowercase"] = {{"quant_algo", "w4a16_nvfp4"}};  // modelopt.py:2427 .upper()
  const MixedPrecisionConfig c = MixedPrecisionConfig::Parse(qc);

  CHECK(c.Resolve("algo.fp8").algo == QuantAlgo::kFp8);
  CHECK(c.Resolve("algo.nvfp4").algo == QuantAlgo::kNvfp4);
  CHECK(c.Resolve("algo.w4a16").algo == QuantAlgo::kW4A16Nvfp4);
  CHECK(c.Resolve("algo.mxfp8").algo == QuantAlgo::kMxfp8);
  CHECK(c.Resolve("algo.lowercase").algo == QuantAlgo::kW4A16Nvfp4);

  // The NVFP4 family carries the group size; FP8/MXFP8 do not.
  CHECK(c.Resolve("algo.nvfp4").group_size == 16);
  CHECK(c.Resolve("algo.w4a16").group_size == 16);
  CHECK(c.Resolve("algo.fp8").group_size == 0);
  CHECK(c.Resolve("algo.mxfp8").group_size == 0);

  CHECK(std::string(QuantAlgoName(QuantAlgo::kW4A16Nvfp4)) == "W4A16_NVFP4");
  CHECK(std::string(QuantAlgoName(QuantAlgo::kFp8)) == "FP8");
  CHECK(std::string(QuantAlgoName(QuantAlgo::kUnquantized)) == "UNQUANTIZED");
}

TEST_CASE("modelopt-mixed: an explicit top-level group_size overrides seeding") {
  nlohmann::ordered_json qc = LoadFixture();
  qc["group_size"] = 32;
  const MixedPrecisionConfig c = MixedPrecisionConfig::Parse(qc);
  CHECK(c.group_size() == 32);
  // modelopt.py builds ONE nvfp4 config from the config-level group_size, so
  // that is the value every NVFP4 module gets — the per-entry field only SEEDS
  // it when the top level is absent. Mirroring that polarity matters: picking
  // the per-entry value instead would be an invention.
  CHECK(c.Resolve("backbone.layers.1.mixer.experts.0.up_proj").group_size == 32);
}
