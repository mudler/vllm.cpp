// Muse Glimmer (`MuseGlimmerForConditionalGeneration`) W0 SCAFFOLDING gate.
// Proves what this lane can prove WITHOUT a checkpoint or a GPU:
//   (1) BOTH architecture strings RESOLVE through the registry (the additive TU),
//   (2) the config DESCENDS from the canonical NESTED layout AND from the older
//       FLAT layout, and the two agree — the normalization upstream added because
//       a flat config otherwise deserializes to an ALL-DEFAULT text config,
//   (3) the four named CORRECTNESS TRAPS behave, each asserted against the value
//       the trap would produce if we got it wrong:
//         (a) the query pre-scale resolves to the SAME ~3.87 from the native raw
//             (~43.784) and the modular pre-folded (~3.87) schemas,
//         (b) `use_qk_norm` / `use_attn_output_gate` default TRUE when ABSENT,
//         (c) the iRoPE mask is NoPE-every-4th counted BACKWARD from the last,
//         (d) the legacy `guac` sandwich-norm remap does not swap the two norms,
//             and `.self_attn.gate_proj` never collides with the MLP gate,
//   (4) the structural name map is faithful, including the WEIGHTLESS modules
//       that deliberately contribute no tensor.
// The forward REFUSES-by-name (asserted). Nothing here claims the 30B model runs:
// Muse Glimmer is BEYOND the pinned oracle (555967922) and is anchored to the OPEN
// vllm#51655. See .agents/specs/muse-glimmer.md §0.
//
// ─── UPSTREAM ANCHOR, AND WHICH MODULES THIS PORTS ───────────────────────────
// REVISION ANCHOR: vllm#51655 head `075d645af`. Deliberately NOT the parity pin
// `555967922`, which carries no muse_glimmer at all (porting-inventory §9
// deviation 16); every `muse_glimmer.py:NNNN` and `configs/muse_glimmer.py:NNNN`
// cited below is that head.
//
// This file is the local form of the TWO upstream config-test modules that
// specs/muse-glimmer.md §4 names, both at `075d645af`:
//
//   tests/transformers_utils/test_muse_glimmer_config.py
//       -> nested-and-flat config descent, the dimension reads, the token ids
//   tests/transformers_utils/test_muse_glimmer_config_schema_norm.py
//       -> schema normalization: the dual `qk_scale_factor` magnitude rule, the
//          `None`-not-False defaults for use_qk_norm / use_attn_output_gate, the
//          BACKWARD-counted iRoPE mask, and the legacy `guac` norm renames
//
// HARNESS ADAPTATION (the only kind here): upstream parametrizes with pytest over
// dict fixtures and asserts on a constructed `MuseGlimmerConfig`; this asserts on
// `ParseMuseGlimmerParams(HfConfig)` over the equivalent JSON, case by case. The
// parameters, schemas, expected values and failure cases are carried across
// unchanged. The five upstream tests/tool_use/test_muse_glimmer_*.py modules are
// ported separately, under tests/vllm/entrypoints/openai/.
#include "vllm/model_executor/models/muse_glimmer.h"
#include "vllm/model_executor/models/model_registry.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

using vllm::DefaultMuseGlimmerNoRopeLayers;
using vllm::EnumerateMuseGlimmerTensors;
using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::MuseGlimmerCheckpointConvention;
using vllm::MuseGlimmerConventionOf;
using vllm::MuseGlimmerParams;
using vllm::NormalizeMuseGlimmerWeightName;
using vllm::ParseMuseGlimmerParams;
using vllm::ResolveMuseGlimmerQueryPreScale;

namespace {

// The released meta-models/Muse-Glimmer-30B geometry, scaled DOWN in layer count
// (the enumeration formula is scale-invariant) but keeping head_dim 128 so the
// query-pre-scale magnitude test exercises the real sqrt(128) threshold.
nlohmann::json TextConfigJson() {
  return nlohmann::json{
      {"model_type", "muse_glimmer_text"},
      {"vocab_size", 4096},
      {"hidden_size", 512},
      {"intermediate_size", 1024},
      {"num_hidden_layers", 8},
      {"num_attention_heads", 4},
      {"num_key_value_heads", 2},
      {"head_dim", 128},
      {"max_position_embeddings", 131072},
      {"sliding_window", 1024},
      {"rms_norm_eps", 1e-6},
      {"post_norm_eps", 1e-8},
      {"hidden_activation", "silu"},
      // The MODULAR schema: pre-folded ~3.87, and use_qk_norm /
      // use_attn_output_gate deliberately ABSENT (they must read as TRUE).
      {"qk_scale_factor", 3.87},
      {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 500000.0}}},
  };
}

nlohmann::json VisionConfigJson() {
  return nlohmann::json{
      {"model_type", "muse_glimmer_vision"},
      {"patch_size", 14},   {"pos_emb_height", 32}, {"pos_emb_width", 32},
      {"num_attention_heads", 16}, {"num_hidden_layers", 4},
      {"hidden_size", 1536},       {"intermediate_size", 8960},
      {"merge_kernel_size", 2},    {"output_dim", 1536 * 2 * 2},
      {"patch_temporal", 2},       {"adapter_dim", 4096},
      {"layer_norm_eps", 1e-5},
  };
}

HfConfig NestedConfig() {
  HfConfig c;
  c.architectures = {"MuseGlimmerForConditionalGeneration"};
  c.hidden_size = 512;
  c.num_hidden_layers = 8;
  c.vocab_size = 4096;
  c.num_attention_heads = 4;
  c.raw = nlohmann::json{{"model_type", "muse_glimmer"},
                         {"text_config", TextConfigJson()},
                         {"vision_config", VisionConfigJson()},
                         {"image_token_id", 200092},
                         {"video_token_id", 200091}};
  return c;
}

// The older FLAT converter layout: every text field at the TOP level, with the
// legacy names, and NO text_config nesting.
HfConfig FlatConfig() {
  HfConfig c;
  c.architectures = {"MuseGlimmerForConditionalGeneration"};
  c.hidden_size = 512;
  c.num_hidden_layers = 8;
  c.vocab_size = 4096;
  c.num_attention_heads = 4;
  nlohmann::json raw = TextConfigJson();
  raw.erase("model_type");
  raw["model_type"] = "muse_glimmer";
  // Legacy names for the two renamed text fields.
  raw["hidden_act"] = "silu";
  raw.erase("hidden_activation");
  // Flat vision fields.
  raw["vision_latent_dim"] = 1536;
  raw["vision_heads"] = 16;
  raw["vision_layers"] = 4;
  raw["vision_output_dim"] = 1536 * 2 * 2;
  raw["vision_patch_size"] = 14;
  raw["vision_patch_temporal"] = 2;
  raw["vision_adapter_dim"] = 4096;
  raw["vision_pos_emb_grid_h"] = 32;
  raw["vision_pos_emb_grid_w"] = 32;
  raw["vision_downsample_factor"] = 2;
  raw["image_token_id"] = 200092;
  raw["video_token_id"] = 200091;
  c.raw = raw;
  return c;
}

bool Has(const std::vector<std::string>& v, const std::string& s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

std::string Normalized(const std::string& in) {
  std::string out;
  const bool kept = NormalizeMuseGlimmerWeightName(in, &out);
  return kept ? out : std::string("<dropped>");
}

}  // namespace

TEST_CASE("MuseGlimmer: both architecture strings resolve through the registry") {
  // Upstream maps BOTH names onto the same class (registry.py @ vllm#51655), so a
  // text-only and a multimodal checkpoint must each resolve.
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  CHECK(std::find(supported.begin(), supported.end(),
                  std::string_view("MuseGlimmerForCausalLM")) != supported.end());
  CHECK(std::find(supported.begin(), supported.end(),
                  std::string_view("MuseGlimmerForConditionalGeneration")) !=
        supported.end());

  // Bind the config to a named local: Resolve() returns a reference, and passing a
  // temporary trips -Werror=dangling-reference.
  const HfConfig mm_config = NestedConfig();
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(mm_config);
  CHECK(reg.info.is_text_generation_model);
  CHECK(reg.info.supports_multimodal);
  // The iRoPE split is sliding-vs-full ATTENTION, not a recurrent lane: Muse
  // Glimmer is NOT a hybrid model and must not claim inner state.
  CHECK_FALSE(reg.info.is_hybrid);
  CHECK_FALSE(reg.info.has_inner_state);

  // The text-only architecture string resolves to the SAME factory.
  HfConfig text_only = NestedConfig();
  text_only.architectures = {"MuseGlimmerForCausalLM"};
  CHECK(ModelRegistry::Resolve(text_only).info.supports_multimodal);
}

TEST_CASE("MuseGlimmer: nested and flat configs resolve to the SAME params") {
  const MuseGlimmerParams nested = ParseMuseGlimmerParams(NestedConfig());
  const MuseGlimmerParams flat = ParseMuseGlimmerParams(FlatConfig());

  // The trap: without flat normalization the flat config silently deserializes to
  // an ALL-DEFAULT text config. If that regressed, hidden_size would come back as
  // the upstream default rather than 512 and these would disagree.
  CHECK(flat.text.hidden_size == nested.text.hidden_size);
  CHECK(flat.text.num_hidden_layers == nested.text.num_hidden_layers);
  CHECK(flat.text.head_dim == nested.text.head_dim);
  CHECK(flat.text.num_key_value_heads == nested.text.num_key_value_heads);
  CHECK(flat.text.hidden_activation == nested.text.hidden_activation);
  CHECK(flat.text.scale_query_by == doctest::Approx(nested.text.scale_query_by));

  CHECK(flat.vision.present);
  CHECK(flat.vision.hidden_size == nested.vision.hidden_size);
  CHECK(flat.vision.num_hidden_layers == nested.vision.num_hidden_layers);
  CHECK(flat.vision.merge_kernel_size == nested.vision.merge_kernel_size);
  CHECK(flat.vision.output_dim == nested.vision.output_dim);
}

TEST_CASE("MuseGlimmer: the query pre-scale agrees across BOTH config schemas") {
  // muse_glimmer.py:472-517. head_dim 128 => sqrt = 11.3137; the native raw value
  // is ~43.784 and the modular pre-folded one ~3.87. Both must land on ~3.87.
  const int64_t head_dim = 128;
  const double folded =
      ResolveMuseGlimmerQueryPreScale(3.87, true, 0.0, false, head_dim);
  const double native =
      ResolveMuseGlimmerQueryPreScale(43.784, true, 0.0, false, head_dim);

  CHECK(folded == doctest::Approx(3.87).epsilon(1e-6));
  CHECK(native == doctest::Approx(3.87).epsilon(1e-3));
  // The failure mode this guards: treating the native value as already folded
  // scales every query by sqrt(128) = 11.3x.
  CHECK(native < 11.0);
  CHECK(std::abs(native - 43.784) > 1.0);

  // An explicit scale_query_by is already final and wins outright.
  CHECK(ResolveMuseGlimmerQueryPreScale(43.784, true, 2.5, true, head_dim) ==
        doctest::Approx(2.5));
  // No scale info at all degrades to the identity, not to 0.
  CHECK(ResolveMuseGlimmerQueryPreScale(0.0, false, 0.0, false, head_dim) ==
        doctest::Approx(1.0));
}

TEST_CASE("MuseGlimmer: qk-norm and the output gate default ON when ABSENT") {
  // muse_glimmer.py:456-469 — the modular schema OMITS both flags, so a naive
  // getattr(..., False) silently drops both mechanisms while still emitting
  // plausible text. Absent MUST mean true; only an explicit false disables.
  const MuseGlimmerParams absent = ParseMuseGlimmerParams(NestedConfig());
  CHECK(absent.text.use_qk_norm);
  CHECK(absent.text.use_attn_output_gate);

  HfConfig off = NestedConfig();
  off.raw["text_config"]["use_qk_norm"] = false;
  off.raw["text_config"]["use_attn_output_gate"] = false;
  const MuseGlimmerParams disabled = ParseMuseGlimmerParams(off);
  CHECK_FALSE(disabled.text.use_qk_norm);
  CHECK_FALSE(disabled.text.use_attn_output_gate);

  // THE SAME TRAP, third field (#405). `normalize_tok_embeddings` is ABSENT
  // from the released config too, and upstream defaults it TRUE
  // (configs/muse_glimmer.py:66; SGLang srt/configs/muse_glimmer.py:100 agrees
  // independently). We defaulted it false, which made `perception_emb_norm` a
  // silent no-op on the vision path.
  //
  // The wiring gate drives that norm by setting the flag EXPLICITLY, so it
  // exercised the mechanism and never the DEFAULT — and the default is the only
  // case a released checkpoint actually hits.
  CHECK(absent.text.normalize_tok_embeddings);
  off.raw["text_config"]["normalize_tok_embeddings"] = false;
  CHECK_FALSE(ParseMuseGlimmerParams(off).text.normalize_tok_embeddings);
}

TEST_CASE("MuseGlimmer: the iRoPE mask counts BACKWARD from the last layer") {
  // configs/muse_glimmer.py:20-26. NoPE (0) every 4th layer counted backward, so
  // the LAST layer is always NoPE/full-attention. Counting FORWARD instead would
  // put the NoPE layers at 0,4,8,... and leave the last layer on RoPE.
  const std::vector<int64_t> mask = DefaultMuseGlimmerNoRopeLayers(8);
  REQUIRE(mask.size() == 8u);
  CHECK(mask == std::vector<int64_t>{1, 1, 1, 0, 1, 1, 1, 0});
  CHECK(mask.back() == 0);

  // 52 layers is the released depth; the last must still be NoPE.
  const std::vector<int64_t> real = DefaultMuseGlimmerNoRopeLayers(52);
  REQUIRE(real.size() == 52u);
  CHECK(real.back() == 0);   // layer 51
  CHECK(real[47] == 0);      // 51 - 47 == 4
  CHECK(real[48] == 1);
  CHECK(real[50] == 1);

  // A checkpoint-supplied mask wins over the default.
  HfConfig c = NestedConfig();
  c.raw["text_config"]["no_rope_layers"] =
      nlohmann::json::array({0, 1, 1, 1, 0, 1, 1, 1});
  const MuseGlimmerParams p = ParseMuseGlimmerParams(c);
  CHECK(p.text.no_rope_layers == std::vector<int64_t>{0, 1, 1, 1, 0, 1, 1, 1});
}

TEST_CASE("MuseGlimmer: the legacy guac sandwich norms do not swap") {
  // muse_glimmer.py:1364-1388. The PREFIX is the discriminator.
  CHECK(MuseGlimmerConventionOf("model.layers.0.input_layernorm.weight") ==
        MuseGlimmerCheckpointConvention::kLegacyGuac);
  CHECK(MuseGlimmerConventionOf(
            "model.language_model.layers.0.input_layernorm.weight") ==
        MuseGlimmerCheckpointConvention::kCanonical);

  // LEGACY: post_attention_layernorm is really the PRE-feedforward norm, and
  // post_attn_norm is the true post-attention one. Renaming in the wrong order
  // makes these two swap, which is silent and wrong.
  CHECK(Normalized("model.layers.3.post_attention_layernorm.weight") ==
        "model.layers.3.pre_feedforward_layernorm.weight");
  CHECK(Normalized("model.layers.3.post_attn_norm.weight") ==
        "model.layers.3.post_attention_layernorm.weight");
  CHECK(Normalized("model.layers.3.post_ffn_norm.weight") ==
        "model.layers.3.post_feedforward_layernorm.weight");

  // CANONICAL: the norms are already correct and must pass through untouched
  // (only the language_model prefix is stripped).
  CHECK(Normalized("model.language_model.layers.3.post_attention_layernorm.weight") ==
        "model.layers.3.post_attention_layernorm.weight");
  CHECK(Normalized("model.language_model.layers.3.pre_feedforward_layernorm.weight") ==
        "model.layers.3.pre_feedforward_layernorm.weight");
}

TEST_CASE("MuseGlimmer: the attention output gate never collides with the MLP gate") {
  // muse_glimmer.py:1400. `.self_attn.gate_proj` is the ATTENTION OUTPUT GATE and
  // must become `output_gate_proj`; the MLP's own `.mlp.gate_proj` is a different
  // tensor and must survive unchanged. Conflating them corrupts both.
  CHECK(Normalized("model.language_model.layers.2.self_attn.gate_proj.weight") ==
        "model.layers.2.self_attn.output_gate_proj.weight");
  CHECK(Normalized("model.language_model.layers.2.mlp.gate_proj.weight") ==
        "model.layers.2.mlp.gate_proj.weight");
}

TEST_CASE("MuseGlimmer: vision weight names normalize to the tower layout") {
  CHECK(Normalized("model.vision_tower.layers.1.norm1.weight") ==
        "vision_encoder.transformer.1.ln_1.weight");
  CHECK(Normalized("model.vision_tower.layers.1.attn.proj.weight") ==
        "vision_encoder.transformer.1.attn.o_proj.weight");
  CHECK(Normalized("model.vision_tower.layers.1.mlp.fc1.weight") ==
        "vision_encoder.transformer.1.mlp.c_fc.weight");
  CHECK(Normalized("model.vision_tower.layers.1.mlp.fc2.bias") ==
        "vision_encoder.transformer.1.mlp.c_proj.bias");
  CHECK(Normalized("model.vision_tower.patch_embedder.patch_embedding.weight") ==
        "vision_encoder.conv1_linear.weight");
  CHECK(Normalized("model.vision_adapter.fc1.weight") == "vision_adapter.c_fc.weight");
  CHECK(Normalized("model.vision_projection.weight") == "vision_projection.weight");
  // Upstream drops the rotary buffer outright.
  CHECK(Normalized("model.rotary_emb.inv_freq") == "<dropped>");
}

TEST_CASE("MuseGlimmer: the structural name map is faithful") {
  const MuseGlimmerParams p = ParseMuseGlimmerParams(NestedConfig());
  const std::vector<std::string> names = EnumerateMuseGlimmerTensors(p);

  // All four sandwich norms are real tensors.
  CHECK(Has(names, "model.layers.0.input_layernorm.weight"));
  CHECK(Has(names, "model.layers.0.post_attention_layernorm.weight"));
  CHECK(Has(names, "model.layers.0.pre_feedforward_layernorm.weight"));
  CHECK(Has(names, "model.layers.0.post_feedforward_layernorm.weight"));
  // The output gate is enumerated because the flag defaults ON.
  CHECK(Has(names, "model.layers.0.self_attn.output_gate_proj.weight"));

  // The WEIGHTLESS modules must contribute NO tensor: embed_norm (:1286), the
  // per-head qk_norm (:1121) and perception_emb_norm (:1470). Enumerating them
  // would make the loader demand tensors no checkpoint ships.
  CHECK_FALSE(Has(names, "model.embed_norm.weight"));
  CHECK_FALSE(Has(names, "model.layers.0.self_attn.qk_norm.weight"));
  CHECK_FALSE(Has(names, "perception_emb_norm.weight"));

  // Vision tower + adapter + projector.
  CHECK(Has(names, "vision_encoder.conv1_linear.weight"));
  CHECK(Has(names, "vision_encoder.positional_embedding_vlm"));
  // W4 CORRECTION: the vision attention ships SEPARATE q/k/v shards, each WITH a
  // bias, and `attn.proj` (-> `attn.o_proj`) also has one. W0 pinned upstream's
  // merged `qkv_proj` MODULE name instead, which no checkpoint contains — the merge
  // is a LOAD-time fold (packed_modules_mapping, muse_glimmer.py:1427-1430). Gated
  // against the real 1436-tensor index in test_muse_glimmer_wiring.cpp.
  CHECK_FALSE(Has(names, "vision_encoder.transformer.0.attn.qkv_proj.weight"));
  for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    CHECK(Has(names, std::string("vision_encoder.transformer.0.attn.") + proj +
                         ".weight"));
    CHECK(Has(names, std::string("vision_encoder.transformer.0.attn.") + proj +
                         ".bias"));
  }
  CHECK(Has(names, "vision_encoder.ln_post.bias"));
  CHECK(Has(names, "vision_adapter.c_proj.weight"));
  CHECK(Has(names, "vision_projection.weight"));

  // lm_head is present only when embeddings are untied.
  CHECK(Has(names, "lm_head.weight"));
  HfConfig tied = NestedConfig();
  tied.raw["text_config"]["tie_word_embeddings"] = true;
  CHECK_FALSE(Has(EnumerateMuseGlimmerTensors(ParseMuseGlimmerParams(tied)),
                  "lm_head.weight"));

  // Disabling the gate removes exactly one tensor per layer.
  HfConfig no_gate = NestedConfig();
  no_gate.raw["text_config"]["use_attn_output_gate"] = false;
  const auto without = EnumerateMuseGlimmerTensors(ParseMuseGlimmerParams(no_gate));
  CHECK(static_cast<int64_t>(names.size() - without.size()) ==
        p.text.num_hidden_layers);
}

TEST_CASE("MuseGlimmer: a mismatched vision output_dim is rejected") {
  // muse_glimmer.py:734-739 — output_dim MUST equal hidden_size * merge^2.
  HfConfig bad = NestedConfig();
  bad.raw["vision_config"]["output_dim"] = 4096;  // != 1536 * 2 * 2
  CHECK_THROWS(ParseMuseGlimmerParams(bad));
}

TEST_CASE("MuseGlimmer: a non-silu hidden activation is rejected") {
  HfConfig bad = NestedConfig();
  bad.raw["text_config"]["hidden_activation"] = "gelu";
  CHECK_THROWS(vllm::ParseMuseGlimmerConfig(bad));
}
