// DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`) W1 scaffold gate.
//
// WHAT THIS GATE IS FOR. W1 of
// `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`
// (ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP) makes the architecture RESOLVE and
// makes its config PARSE and VALIDATE, and nothing else. This file proves those
// two things and proves that every arm W1 does NOT implement refuses BY NAME.
// It deliberately proves NO load and NO forward, because W1 claims neither.
//
// REACHABILITY (AGENTS.md, "Nothing lands dead"). Every case that can enter
// through a production entry point does:
//   * the RESOLVE cases call `ModelRegistry::Resolve`, which is what
//     `LoadedEngine` calls to pick a factory; nothing here builds a
//     `ModelRegistration` or a `ModelFactory` by hand;
//   * the LOADER cases call `LoadedEngine::FromModelDir`, the same call
//     `vllm_c.cpp` and `server_main.cpp` make;
//   * the config-descent and KV-cache cases go through the resolved
//     registration's own hooks rather than calling the parser directly, so a
//     registration that pointed at the wrong parser would fail here.
// The `forward` hook is the one thing that CANNOT be entered from a production
// entry point in this wave, and the reason is the wave's own boundary: reaching
// a forward requires a `LoadedModel`, and the only constructor of one is the
// loader, which refuses. That is asserted below as a property rather than
// papered over with a hand-built model.
//
// THE ORACLE. vLLM `e77daef89e`, which is AHEAD OF our parity pin; no pin was
// advanced by this wave and none is needed, because registration and config
// validation claim no gate against the oracle. `DeepseekV41Config` is a
// FLATTENING SHIM that raises nothing, so every validation case below cites the
// MODEL-code line its rejection mirrors, or says the rejection is ours.
#include "vllm/model_executor/models/deepseek_v4_1.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, *KvCache
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vt/device.h"

using vllm::DeepseekV41Params;
using vllm::HfConfig;
using vllm::ModelRegistry;

namespace {

const std::vector<std::string>& Archs() {
  static const std::vector<std::string> a = {"DeepseekV41ForCausalLM"};
  return a;
}

// ── the REAL published artifact, checked in byte-for-byte ───────────────────
// `deepseek-ai/DeepSeek-V4.1-Flash` @ revision
// `dba1be0a40aa45a94ad051997016db3960a90277`, fetched 2026-09-13:
//
//   curl -sL https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/config.json
//
//   sha256  8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879
//   bytes   3311
//
// The BYTES are the pin and git is what holds them: a re-quantization upstream
// under an unchanged repo id cannot move what this gate ran against, because
// this gate reads the checked-in copy and never the network. The size assertion
// below is the cheap tripwire for an accidental in-tree edit (a reformat, a
// stray newline); the sha256 above is what a reader recomputes to confirm the
// checked-in copy is still the published one.
std::string FixtureText() {
  const std::string path =
      std::string(DEEPSEEK_V4_1_CKPT_FIXTURE_DIR) + "/config.json";
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open fixture: " << path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

const nlohmann::json& PublishedConfigJson() {
  static const nlohmann::json doc = nlohmann::json::parse(FixtureText());
  return doc;
}

HfConfig ConfigFrom(const nlohmann::json& doc) {
  return vllm::ParseHfConfig(doc, "deepseek_v4_1_fixture");
}

HfConfig PublishedConfig() { return ConfigFrom(PublishedConfigJson()); }

// Parse through the RESOLVED REGISTRATION rather than by calling
// `ParseDeepseekV41Params` directly, wherever the case is about validation
// reaching a user. `parse_config` returns void, so the params cases below call
// the parser for its VALUE and the refusal cases drive the registration hook.
void ParseThroughRegistration(const HfConfig& cfg) {
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(Archs());
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->parse_config != nullptr);
  reg.factory->parse_config(cfg);
}

// Mutate ONE key inside `text_config` and return the message the registration's
// own config hook refuses it with. Returns "" when it did not refuse, which is
// a failure the caller asserts rather than a silent pass.
std::string RefusalForTextKey(const char* key, const nlohmann::json& value) {
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"][key] = value;
  try {
    ParseThroughRegistration(ConfigFrom(doc));
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// A `LoadedModel` of a DIFFERENT registration. This is the ONLY handle any
// caller can present while every load arm of this architecture refuses, and it
// is admissible here for one reason: `ForwardDeepseekV41ForCausalLM` refuses
// BEFORE any downcast, so it never reads the handle. If a later wave puts a
// `ModelAs<...>` first, the case below turns red on the type-mismatch message
// instead of silently measuring the wrong refusal.
class ForeignLoadedModel final : public vllm::LoadedModel {
 public:
  explicit ForeignLoadedModel(const vllm::ModelRegistration& registration)
      : vllm::LoadedModel(registration) {}
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

  vllm::ModelForwardInput Get() {
    return vllm::ModelForwardInput{.token_ids = token_ids,
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

// ═══════════════════════════════════════════════════════════════════════════
// 1. THE ARCHITECTURE RESOLVES, through the production entry point.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("deepseek-v4.1: the published architecture resolves through the registry") {
  // `registry.py:379` maps "DeepseekV41ForCausalLM" -> vllm.models.deepseek_v4_1.
  // This is the string the published `config.json` declares in `architectures`,
  // and it is the ONLY string this row registers.
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(Archs());
  CHECK(reg.architecture == std::string("DeepseekV41ForCausalLM"));
  REQUIRE(reg.factory != nullptr);

  // It resolves from the REAL config too, not only from a hand-written list:
  // this is the overload `LoadedEngine` actually uses. The config is NAMED
  // rather than passed as a temporary: `Resolve` returns a reference, and a
  // temporary argument would leave it dangling (-Werror=dangling-reference).
  const HfConfig cfg = PublishedConfig();
  const vllm::ModelRegistration& from_cfg = ModelRegistry::Resolve(cfg);
  CHECK(from_cfg.architecture == std::string("DeepseekV41ForCausalLM"));
  CHECK(&from_cfg == &reg);
}

TEST_CASE("deepseek-v4.1: it resolves as a DISTINCT registration from V4") {
  // The two must not collapse onto one factory. Upstream FORKED the package
  // rather than parameterizing it, and V4's parser rejects this checkpoint
  // outright (its compress_ratios are {0,4,128}); a shared registration would
  // make that rejection look like a V4.1 defect.
  const vllm::ModelRegistration& v41 = ModelRegistry::Resolve(Archs());
  const std::vector<std::string> v4_archs = {"DeepseekV4ForCausalLM"};
  const vllm::ModelRegistration& v4 = ModelRegistry::Resolve(v4_archs);
  CHECK(&v41 != &v4);
  CHECK(v41.factory != v4.factory);
}

TEST_CASE("deepseek-v4.1: the internal text-only class is NOT registered") {
  // `DeepseekV41LLMForCausalLM` (`nvidia/model.py:979`) is upstream-internal:
  // it is declared by no published artifact and sits in no registry table.
  // Registering it would claim support for a string nothing produces.
  const std::vector<std::string> internal = {"DeepseekV41LLMForCausalLM"};
  CHECK_THROWS_AS((void)ModelRegistry::Resolve(internal), std::exception);
  // Same for the speculator, which is W3e's and is a separate architecture.
  const std::vector<std::string> draft = {"DSparkV41DraftModel"};
  CHECK_THROWS_AS((void)ModelRegistry::Resolve(draft), std::exception);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. THE REAL config.json DESCENDS AND VALIDATES, field by field.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("deepseek-v4.1: the checked-in fixture is the published artifact") {
  // The tripwire for an accidental in-tree edit. If this fires, recompute the
  // sha256 in this file's header against the published revision before touching
  // anything else: the fixture, not the parser, is what moved.
  CHECK(FixtureText().size() == 3311U);
  const nlohmann::json& doc = PublishedConfigJson();
  REQUIRE(doc.contains("text_config"));
  REQUIRE(doc.contains("vision_config"));
  CHECK(doc.at("model_type") == "deepseek_v41");
  CHECK(doc.at("architectures").at(0) == "DeepseekV41ForCausalLM");
}

TEST_CASE("deepseek-v4.1: the real config descends through the registration") {
  CHECK_NOTHROW(ParseThroughRegistration(PublishedConfig()));
}

TEST_CASE("deepseek-v4.1: the nested text_config lifts to flat scalars") {
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(PublishedConfig());

  // Every one of these sits under `text_config` in the file and is read flat
  // here. That descent IS the nested->flat lift `DeepseekV41Config` performs.
  CHECK(p.hidden_size == 5120);
  CHECK(p.num_hidden_layers == 40);
  CHECK(p.vocab_size == 129280);
  CHECK(p.num_attention_heads == 64);
  CHECK(p.num_key_value_heads == 1);
  CHECK(p.max_position_embeddings == 1048576);
  CHECK(p.tie_word_embeddings == false);
  CHECK(p.num_nextn_predict_layers == 3);

  // 1e-20 is the PUBLISHED value and not a typo. `float` would REPRESENT it --
  // `float(1e-20)` is `0x1e3ce508`, a NORMAL float, eighteen orders of
  // magnitude above f32's smallest normal 1.175e-38 -- so nothing here is
  // about denormals. What a float carrier would lose is the JSON literal
  // itself: `float(1e-20)` is 9.99999968e-21, off by 3.2e-8 relative, and this
  // exact compare against the published literal would go RED. That is the
  // point of the EXACT compare rather than `Approx`: the property under test is
  // that the literal ROUND-TRIPPED the descent undamaged, and at this magnitude
  // any relative epsilon would be an arbitrary choice that admits the damage.
  CHECK(p.rms_norm_eps == 1e-20);
  CHECK(p.rms_norm_eps > 0.0);

  // MLA geometry. `q_lora_rank` and `index_n_heads` are the two that MOVED
  // from V4-Flash (1024 -> 1280, 64 -> 32), which is why this row cannot reuse
  // V4's parsed params even where the key spellings agree.
  CHECK(p.head_dim == 512);
  CHECK(p.qk_rope_head_dim == 64);
  CHECK(p.q_lora_rank == 1280);
  CHECK(p.o_lora_rank == 1024);
  CHECK(p.o_groups == 8);
  CHECK(p.sliding_window == 128);
  CHECK(p.rope_theta == doctest::Approx(10000.0));
  CHECK(p.compress_rope_theta == doctest::Approx(160000.0));

  // MoE.
  CHECK(p.n_routed_experts == 384);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.num_experts_per_tok == 6);
  CHECK(p.moe_intermediate_size == 2304);
  CHECK(p.scoring_func == "sqrtsoftplus");
  CHECK(p.topk_method == "noaux_tc");
  CHECK(p.norm_topk_prob == true);
  CHECK(p.routed_scaling_factor == doctest::Approx(1.5));
  CHECK(p.swiglu_limit == doctest::Approx(10.0));

  // MHC.
  CHECK(p.hc_mult == 4);
  CHECK(p.hc_sinkhorn_iters == 20);
  CHECK(p.hc_eps == doctest::Approx(1e-6));

  // DSA.
  CHECK(p.index_n_heads == 32);
  CHECK(p.index_head_dim == 128);
  CHECK(p.index_topk == 512);
  CHECK(p.kv_source_layer_ids == std::vector<int64_t>{2, 8, 14, 20});
  CHECK(p.index_source_layer_ids ==
        std::vector<int64_t>{2, 8, 14, 20, 24, 28, 32, 36});
  CHECK(p.candidate_source_layer_id == 20);
  CHECK(p.candidate_topk_blocks == 2048);
  CHECK(p.candidate_block_size == 8);
}

TEST_CASE("deepseek-v4.1: rope_scaling descends as YaRN") {
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(PublishedConfig());
  // Upstream POPS `rope_scaling` before `super().__init__` and restores it onto
  // `rope_parameters` afterwards, because `rope_scaling` is a property in
  // transformers v5 that the base constructor re-standardizes
  // (deepseek_v41.py:29-44). We have no such property to dodge, so we read the
  // same five values straight off the sub-object; this case pins that they
  // arrive rather than silently defaulting.
  CHECK(p.rope_type == "yarn");
  CHECK(p.rope_scale_factor == doctest::Approx(16.0));
  CHECK(p.rope_orig_ctx == 65536);
  CHECK(p.rope_beta_fast == doctest::Approx(32.0));
  CHECK(p.rope_beta_slow == doctest::Approx(1.0));
}

TEST_CASE("deepseek-v4.1: compress_ratios is LONGER than num_hidden_layers") {
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(PublishedConfig());

  // THE CASE THAT WOULD CATCH A LENGTH-EQUALITY CHECK. 43 entries against 40
  // layers, and the difference is exactly `num_nextn_predict_layers`. A parser
  // that asserted `len == num_hidden_layers` would REJECT the real published
  // artifact, so this gate pins the inequality deliberately.
  REQUIRE(p.compress_ratios.size() == 43U);
  CHECK(p.num_hidden_layers == 40);
  CHECK(static_cast<int64_t>(p.compress_ratios.size()) ==
        p.num_hidden_layers + p.num_nextn_predict_layers);

  // The values are drawn from {0,1,2} -- NOT V4's {0,4,128}. This is the single
  // clearest reason this architecture needed its own parser.
  for (const int64_t r : p.compress_ratios) {
    CHECK((r == 0 || r == 1 || r == 2));
  }
  CHECK(p.compress_ratio(0) == 0);
  CHECK(p.compress_ratio(2) == 2);
  CHECK(p.compress_ratio(20) == 1);

  // The bounds guard, mirroring attention.py:251-256 ("MTP layers past the
  // configured list are pure sliding-window"). Past the END of the list it is 0
  // rather than out-of-range.
  CHECK(p.compress_ratio(43) == 0);
  CHECK(p.compress_ratio(9999) == 0);
  CHECK(p.compress_ratio(-1) == 0);

  // `is_backbone = layer_id < config.num_hidden_layers`
  // (`deepseek_v4_1/attention.py:273`): the three trailing entries are NOT
  // backbone layers.
  CHECK(p.is_backbone_layer(39));
  CHECK_FALSE(p.is_backbone_layer(40));
  CHECK_FALSE(p.is_backbone_layer(42));
}

TEST_CASE("deepseek-v4.1: engram and dspark descend") {
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(PublishedConfig());

  // Engram has NO V4 counterpart at all; W3a owes the host reference. Parsing
  // it in W1 is what lets that wave start from a validated config rather than
  // from the file.
  CHECK(p.engram_layer_ids == std::vector<int64_t>{1, 14});
  CHECK(p.engram_num_embeddings ==
        std::vector<int64_t>{384006168, 384016682});
  CHECK(p.engram_max_ngram_size == 4);
  CHECK(p.engram_vocab_size == 16000000);
  CHECK(p.engram_n_heads == 8);
  CHECK(p.engram_head_dim == 256);
  CHECK(p.engram_pad_token_id == 2);
  CHECK(p.engram_compressed_vocab_size == 99092);

  // DSpark is IN this config, where V4 shipped it as a separate draft model.
  CHECK(p.dspark_block_size == 5);
  CHECK(p.dspark_noise_token_id == 128799);
  CHECK(p.dspark_target_layer_ids == std::vector<int64_t>{37, 38, 39});
  CHECK(p.dspark_markov_rank == 256);
  CHECK(p.dspark_n_routed_experts == 128);
  CHECK(p.dspark_num_experts_per_tok == 3);
}

TEST_CASE("deepseek-v4.1: the vision_* lift and its derived mm-prefix flags") {
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(PublishedConfig());

  // deepseek_v41.py:52-62 -- `vision_config` re-exposed under `vision_*`.
  CHECK(p.vision_n_layers == 32);
  CHECK(p.vision_dim == 1024);
  CHECK(p.vision_n_heads == 16);
  CHECK(p.vision_inter_dim == 2816);
  CHECK(p.vision_patch_size == 14);
  CHECK(p.vision_rope_theta == doctest::Approx(10000.0));
  CHECK(p.vision_downsample_ratio == 3);
  CHECK(p.vision_max_n_token == 1024);
  CHECK(p.vision_min_pixels == 295936);

  // `max_wh_ratio` is literally `null` in the published file and upstream's
  // `.get("max_wh_ratio")` yields None with NO fallback. Absent is not zero: a
  // ratio of zero is a value, and this distinction is the reason the field is
  // an optional rather than a double.
  CHECK_FALSE(p.vision_max_wh_ratio.has_value());

  // deepseek_v41.py:66-76 -- all three are DERIVED from `vision_n_layers > 0`
  // and none is read from the file.
  CHECK(p.is_mm_prefix_lm);
  CHECK(p.mm_prefix_clamp_sliding_window);
  // 2 for V4.1 against V4's 4 (deepseek_v4.py:51), because V4.1's compressors
  // are ratio-2 where V4's were ratio-4. Verified on both upstream files.
  CHECK(p.mm_prefix_span_leading_pad_modulus == 2);
}

TEST_CASE("deepseek-v4.1: a config with no vision tower derives the flags OFF") {
  // The text-only variant. All three mm-prefix flags are functions of
  // `vision_n_layers > 0`, so removing the tower must turn all three off and
  // must take the MODULUS to 0 rather than leaving it at 2 -- a stale modulus
  // would pad image spans in a model that has none.
  nlohmann::json doc = PublishedConfigJson();
  doc.erase("vision_config");
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(ConfigFrom(doc));
  CHECK(p.vision_n_layers == 0);
  CHECK_FALSE(p.is_mm_prefix_lm);
  CHECK_FALSE(p.mm_prefix_clamp_sliding_window);
  CHECK(p.mm_prefix_span_leading_pad_modulus == 0);
  // Upstream's `.get(...)` fallbacks still apply with no tower present, and
  // they are NOT zero. This pins that we mirror the fallback rather than
  // zero-filling, which is a difference a later vision wave would trip on.
  CHECK(p.vision_dim == 1024);
  CHECK(p.vision_n_heads == 16);
  CHECK(p.vision_inter_dim == 2816);
}

TEST_CASE("deepseek-v4.1: quantization_config descends and expert_dtype LIFTS") {
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(PublishedConfig());
  CHECK(p.quant_method == "fp8");
  CHECK(p.quant_activation_scheme == "dynamic");
  CHECK(p.quant_scale_fmt == "ue8m0");
  // [32,32], where V4 blocks at [128,128]. W3b owes the MXFP8 32x32 arm.
  CHECK(p.weight_block_size == std::vector<int64_t>{32, 32});

  // deepseek_v41.py:48-50: the model code reads `expert_dtype` off the TOP
  // level, and the published file carries it only inside
  // `quantization_config`. This lift is the whole reason that block is parsed.
  CHECK(p.expert_dtype == "fp4");
}

TEST_CASE("deepseek-v4.1: an explicit top-level expert_dtype WINS over the lift") {
  // Upstream lifts only `if not hasattr(self, "expert_dtype")`, so a top-level
  // value takes precedence. Pinning the ORDER matters: reversing it would
  // silently ignore an explicit override.
  nlohmann::json doc = PublishedConfigJson();
  doc["expert_dtype"] = "fp8";
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(ConfigFrom(doc));
  CHECK(p.expert_dtype == "fp8");
}

TEST_CASE("deepseek-v4.1: a text_config.expert_dtype ALSO wins over the lift") {
  // THE MIDDLE TIER, which an earlier revision of the parser did not have. The
  // shim `setattr`s every `text_config` key onto `self` at
  // `deepseek_v41.py:34-40`, BEFORE the `if not hasattr(self, "expert_dtype")`
  // guard at `:48-50` — so `text_config.expert_dtype` makes `hasattr` true and
  // the `quantization_config` lift never fires. Reading the top level and
  // `quantization_config` only would answer `fp4` here where upstream answers
  // `fp8`.
  nlohmann::json doc = PublishedConfigJson();
  REQUIRE(doc["quantization_config"]["expert_dtype"] == "fp4");
  doc["text_config"]["expert_dtype"] = "fp8";
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(ConfigFrom(doc));
  CHECK(p.expert_dtype == "fp8");
}

TEST_CASE("deepseek-v4.1: the top level OUTRANKS text_config for expert_dtype") {
  // The third tier's order against the second. `super().__init__(**kwargs)` at
  // `deepseek_v41.py:41` runs AFTER the `text_config` `setattr` loop, so a
  // top-level key overwrites what `text_config` put there. All three tiers
  // disagree here, so this case fails on any permutation of the order.
  nlohmann::json doc = PublishedConfigJson();
  REQUIRE(doc["quantization_config"]["expert_dtype"] == "fp4");
  doc["text_config"]["expert_dtype"] = "fp4";
  doc["expert_dtype"] = "fp8";
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(ConfigFrom(doc));
  CHECK(p.expert_dtype == "fp8");
}

TEST_CASE("deepseek-v4.1: quantization_config is found under text_config too") {
  // Same mechanism, same two-tier order, on the block rather than the one key:
  // `text_config.quantization_config` becomes `self.quantization_config`
  // through the `setattr` loop, and `:48` reads it by `getattr`. Moving the
  // published block down one level must change nothing.
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["quantization_config"] = doc["quantization_config"];
  doc.erase("quantization_config");
  const DeepseekV41Params p = vllm::ParseDeepseekV41Params(ConfigFrom(doc));
  CHECK(p.quant_method == "fp8");
  CHECK(p.quant_scale_fmt == "ue8m0");
  CHECK(p.weight_block_size == std::vector<int64_t>{32, 32});
  CHECK(p.expert_dtype == "fp4");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. EVERY VALIDATION ERROR, and each names its field.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("deepseek-v4.1: a missing text_config is refused BY NAME") {
  nlohmann::json doc = PublishedConfigJson();
  // Keep the keys `ParseHfConfig` itself requires at the top level, so the
  // refusal under test is OURS and not the generic HF-config one.
  doc["hidden_size"] = 5120;
  doc["num_hidden_layers"] = 40;
  doc.erase("text_config");
  std::string msg;
  try {
    ParseThroughRegistration(ConfigFrom(doc));
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("text_config") != std::string::npos);
  CHECK(msg.find("deepseek-v4.1") != std::string::npos);
}

TEST_CASE("deepseek-v4.1: a V4 compress_ratio is refused, naming the layer") {
  // THE DISCRIMINATING CASE between the two architectures. 4 and 128 are V4's
  // ratios and are valid there; here they are not representable, and upstream
  // says so explicitly (attention.py:257-262).
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["compress_ratios"][3] = 4;
  std::string msg;
  try {
    ParseThroughRegistration(ConfigFrom(doc));
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("compress_ratio") != std::string::npos);
  CHECK(msg.find("layer 3") != std::string::npos);  // the offending layer
  // The offending VALUE, asserted with its field so the assertion is not
  // vacuous: a bare `find("4")` is satisfied by the "deepseek-v4.1" that every
  // message in this file carries, and would stay green on a message that never
  // named the value at all.
  CHECK(msg.find("compress_ratio=4") != std::string::npos);
  // 128 must NOT be reported here; it is the other V4 ratio and belongs to the
  // second half of this case.
  CHECK(msg.find("compress_ratio=128") == std::string::npos);

  // 128 is refused for the same reason and names its own layer.
  nlohmann::json doc128 = PublishedConfigJson();
  doc128["text_config"]["compress_ratios"][7] = 128;
  std::string msg128;
  try {
    ParseThroughRegistration(ConfigFrom(doc128));
  } catch (const std::exception& e) {
    msg128 = e.what();
  }
  REQUIRE_FALSE(msg128.empty());
  CHECK(msg128.find("layer 7") != std::string::npos);
  CHECK(msg128.find("compress_ratio=128") != std::string::npos);
}

TEST_CASE("deepseek-v4.1: a compressed layer with no source layers is refused") {
  // attention.py:277-283. The published config HAS compressed layers, so
  // emptying either source list must refuse.
  const std::string no_kv =
      RefusalForTextKey("kv_source_layer_ids", nlohmann::json::array());
  REQUIRE_FALSE(no_kv.empty());
  CHECK(no_kv.find("kv_source_layer_ids") != std::string::npos);
  CHECK(no_kv.find("index_source_layer_ids") != std::string::npos);

  const std::string no_index =
      RefusalForTextKey("index_source_layer_ids", nlohmann::json::array());
  REQUIRE_FALSE(no_index.empty());
  CHECK(no_index.find("index_source_layer_ids") != std::string::npos);
}

TEST_CASE("deepseek-v4.1: an all-zero compress_ratios needs NO source layers") {
  // The complement of the case above, and the reason the check is conditional
  // rather than unconditional: with no compressed layer there is nothing to
  // source, and upstream only raises inside `if self.compress_ratio > 0`.
  // Without this case an unconditional check would pass the test above and
  // still be wrong.
  nlohmann::json doc = PublishedConfigJson();
  for (auto& r : doc["text_config"]["compress_ratios"]) r = 0;
  doc["text_config"]["kv_source_layer_ids"] = nlohmann::json::array();
  doc["text_config"]["index_source_layer_ids"] = nlohmann::json::array();
  CHECK_NOTHROW(ParseThroughRegistration(ConfigFrom(doc)));
}

TEST_CASE("deepseek-v4.1: a compressed layer BELOW every source is refused") {
  // THE SECOND WAY `attention.py:277-289` raises, and the one a non-empty check
  // does not cover:
  //
  //     self.kv_source_layer_id = max(s for s in self.kv_source_layers
  //                                   if s <= layer_id)
  //
  // With a NON-EMPTY source list whose every entry sits above the compressed
  // layer, the generator is empty and `max()` raises. The published config is
  // ONE entry from that boundary — its first compressed layer is 2 and its
  // first kv source is also 2 — so this is a reachable config class and not a
  // hypothetical. Upstream raises a bare `ValueError: max() arg is an empty
  // sequence`, which names no field; what is mirrored is WHEN it raises, and
  // the message here additionally names the field, the layer and the list.
  nlohmann::json doc = PublishedConfigJson();
  REQUIRE(doc["text_config"]["compress_ratios"][2] == 2);
  REQUIRE(doc["text_config"]["kv_source_layer_ids"][0] == 2);
  doc["text_config"]["kv_source_layer_ids"][0] = 8;  // now [8, 8, 14, 20]
  std::string msg;
  try {
    ParseThroughRegistration(ConfigFrom(doc));
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("kv_source_layer_ids") != std::string::npos);
  CHECK(msg.find("layer 2") != std::string::npos);
  // NOT the empty-list refusal: the list here is non-empty, and sending the
  // reader to "add source layers" would be the wrong next step.
  CHECK(msg.find("requires kv_source_layer_ids") == std::string::npos);

  // The index list is guarded the same way and names ITSELF, so a guard written
  // once and applied to the wrong list would fail here.
  nlohmann::json doc_ix = PublishedConfigJson();
  doc_ix["text_config"]["index_source_layer_ids"][0] = 8;
  std::string msg_ix;
  try {
    ParseThroughRegistration(ConfigFrom(doc_ix));
  } catch (const std::exception& e) {
    msg_ix = e.what();
  }
  REQUIRE_FALSE(msg_ix.empty());
  CHECK(msg_ix.find("index_source_layer_ids") != std::string::npos);
  CHECK(msg_ix.find("layer 2") != std::string::npos);
}

TEST_CASE("deepseek-v4.1: a source exactly AT the compressed layer is accepted") {
  // The boundary itself, in the direction that must NOT refuse. Upstream's
  // predicate is `s <= layer_id` and not `s < layer_id`, so a source published
  // BY the compressed layer counts — which is exactly what the published config
  // does at layer 2. Without this case the guard above could be written with a
  // strict `<` and stay green.
  CHECK_NOTHROW(ParseThroughRegistration(PublishedConfig()));

  // And one entry in the other direction is still fine: a source BELOW the
  // first compressed layer.
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["kv_source_layer_ids"][0] = 1;
  doc["text_config"]["index_source_layer_ids"][0] = 1;
  CHECK_NOTHROW(ParseThroughRegistration(ConfigFrom(doc)));
}

TEST_CASE("deepseek-v4.1: the 512/64 MLA geometry pair is refused together") {
  // compressor.py:210 asserts `head_dim == 512 and rope_head_dim == 64`. The
  // NoPE width is their DIFFERENCE, so a half-satisfied pair is not
  // representable and each half must refuse.
  const std::string bad_head = RefusalForTextKey("head_dim", 256);
  REQUIRE_FALSE(bad_head.empty());
  CHECK(bad_head.find("head_dim") != std::string::npos);
  CHECK(bad_head.find("512") != std::string::npos);

  const std::string bad_rope = RefusalForTextKey("qk_rope_head_dim", 32);
  REQUIRE_FALSE(bad_rope.empty());
  CHECK(bad_rope.find("qk_rope_head_dim") != std::string::npos);
}

TEST_CASE("deepseek-v4.1: the required positive scalars are refused BY NAME") {
  struct Case {
    const char* key;
    const char* needle;
  };
  const Case cases[] = {
      {"hidden_size", "hidden_size"},
      {"num_hidden_layers", "num_hidden_layers"},
      {"n_routed_experts", "n_routed_experts"},
      {"hc_mult", "hc_mult"},
  };
  for (const Case& c : cases) {
    CAPTURE(c.key);
    const std::string msg = RefusalForTextKey(c.key, 0);
    REQUIRE_FALSE(msg.empty());
    // The message names the FIELD, so a reader does not have to bisect a
    // config to find which scalar was missing.
    CHECK(msg.find(c.needle) != std::string::npos);
  }
}

TEST_CASE("deepseek-v4.1: an unscoped scorer or expert dtype is refused BY NAME") {
  const std::string bad_score = RefusalForTextKey("scoring_func", "sigmoid");
  REQUIRE_FALSE(bad_score.empty());
  CHECK(bad_score.find("scoring_func") != std::string::npos);
  CHECK(bad_score.find("sqrtsoftplus") != std::string::npos);
  CHECK(bad_score.find("sigmoid") != std::string::npos);

  // `expert_dtype` is the LIFTED one, so this also proves the lift is what the
  // validator reads.
  nlohmann::json doc = PublishedConfigJson();
  doc["quantization_config"]["expert_dtype"] = "int4";
  std::string msg;
  try {
    ParseThroughRegistration(ConfigFrom(doc));
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("expert_dtype") != std::string::npos);
  CHECK(msg.find("int4") != std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. EVERY UNIMPLEMENTED ARM REFUSES BY NAME.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// The loader refusal reached through the entry point a USER reaches it by --
// the same call `vllm_c.cpp` and `server_main.cpp` make. Not a call to
// `factory->load_weights`: that would measure the function, where what matters
// is the capability.
std::string LoadRefusalFor(const std::string& model_path) {
  vllm::entrypoints::EngineParams params;
  // PINNED rather than left `kAuto`: the resolution `kAuto` takes depends on
  // what this build registered, and these cases are about the loader.
  params.device = vllm::Device::kCPU;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(model_path, params);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// A self-deleting model directory carrying the three files `FromModelDir`'s
// safetensors branch reads before it reaches the weight loader: the REAL
// published `config.json`, a tokenizer, and one shard. The shard is empty on
// purpose -- the loader refuses before any tensor is looked up, and building 40
// layers of fake weights to show that would be proving something else.
class TempSafetensorsDir {
 public:
  // Defaults to the REAL published config. A caller may pass a variant when the
  // case needs to reach a wall that the published config stops short of.
  explicit TempSafetensorsDir(nlohmann::json config = PublishedConfigJson()) {
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
           ("dsv41_st_dir_" + std::to_string(counter++));
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
    Write("config.json", config.dump(1));
    Write("tokenizer.json", TinyTokenizerJson());
    Write("model.safetensors", EmptySafetensors());
    path_ = gguf_test::Utf8Path(dir_);
  }
  ~TempSafetensorsDir() {
    std::error_code ignored;
    std::filesystem::remove_all(dir_, ignored);
  }
  TempSafetensorsDir(const TempSafetensorsDir&) = delete;
  TempSafetensorsDir& operator=(const TempSafetensorsDir&) = delete;
  const std::string& path() const { return path_; }

 private:
  static std::string TinyTokenizerJson() {
    nlohmann::json vocab = nlohmann::json::object();
    vocab["▁"] = 0;
    vocab["a"] = 1;
    vocab["b"] = 2;
    vocab["c"] = 3;
    const nlohmann::json meta{{"type", "Metaspace"},
                              {"replacement", "▁"},
                              {"prepend_scheme", "always"},
                              {"split", true}};
    const nlohmann::json j{
        {"version", "1.0"},
        {"pre_tokenizer", meta},
        {"decoder", meta},
        {"model",
         {{"type", "BPE"},
          {"unk_token", nullptr},
          {"vocab", vocab},
          {"merges", nlohmann::json::array()}}},
        {"added_tokens", nlohmann::json::array()}};
    return j.dump(1);
  }

  static std::string EmptySafetensors() {
    const std::string header = R"({"__metadata__":{"format":"pt"}})";
    std::string out(8, '\0');
    const uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
      out[static_cast<size_t>(i)] = static_cast<char>((n >> (8 * i)) & 0xffU);
    }
    return out + header;
  }

  void Write(const std::string& name, const std::string& bytes) const {
    std::ofstream out(dir_ / name, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  std::filesystem::path dir_;
  std::string path_;
};

}  // namespace

namespace {

// The published config with its `quantization_config` removed. NOT a real
// checkpoint shape and never presented as one -- it exists only to step past
// the FP8 block-shape wall documented in the case below, so the case after it
// can prove the ARCHITECTURE's own loader refusal is reachable and correct.
nlohmann::json UnquantizedVariant() {
  nlohmann::json doc = PublishedConfigJson();
  doc.erase("quantization_config");
  // `expert_dtype` is lifted out of that block, so it has to be restated at the
  // top level or the parser refuses before the loader is reached at all -- the
  // very precedence this file pins elsewhere.
  doc["expert_dtype"] = "fp4";
  return doc;
}

}  // namespace

TEST_CASE("deepseek-v4.1: the published safetensors config hits the FP8 32x32 wall") {
  // WHAT THIS CASE RECORDS, and it is a finding rather than a formality. The
  // REAL published config never reaches this architecture's own loader hook: it
  // is refused first, and by a guard that belongs to nobody in this row. The
  // checkpoint is FP8 with `weight_block_size [32, 32]`, and this build
  // implements only [128, 128] (issue #1189).
  //
  // That guard is correct and its message already names the missing part, so
  // W1 does NOT move it. It is worth pinning because the missing arm it names
  // IS this campaign's W3b -- "MXFP8 32x32 UE8M0 host reference: block dequant
  // and the linear arm, against V4's 128x128 fp8". The first wall a user hits
  // on this model is therefore already labelled with the right work, and a
  // later wave that lands W3b must revisit this case rather than discover the
  // interaction.
  const TempSafetensorsDir dir;
  const std::string msg = LoadRefusalFor(dir.path());
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("weight_block_size") != std::string::npos);
  CHECK(msg.find("[32, 32]") != std::string::npos);
  CHECK(msg.find("[128, 128]") != std::string::npos);
  // It names the missing part as OURS, not as a bad checkpoint.
  CHECK(msg.find("missing arm in vllm.cpp") != std::string::npos);
  CHECK(msg.find("1189") != std::string::npos);
}

TEST_CASE("deepseek-v4.1: the safetensors loader refuses BY NAME behind that wall") {
  // Past the FP8 guard, the refusal a user reaches is the ARCHITECTURE's own,
  // and this is the case that proves it is wired and says the right thing.
  const TempSafetensorsDir dir{UnquantizedVariant()};
  const std::string msg = LoadRefusalFor(dir.path());
  REQUIRE_FALSE(msg.empty());
  // It names the architecture, the missing part, and where the work is owed.
  CHECK(msg.find("DeepseekV41ForCausalLM") != std::string::npos);
  CHECK(msg.find("safetensors") != std::string::npos);
  CHECK(msg.find("not ported") != std::string::npos);
  CHECK(msg.find("MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm") !=
        std::string::npos);
  CHECK(msg.find("ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP") != std::string::npos);
  // It must NOT be the GGUF refusal: the two are different texts for different
  // next steps, and a bare "something threw" would pass on either.
  CHECK(msg.find("GGUF weights") == std::string::npos);
}

TEST_CASE("deepseek-v4.1: a `deepseek41` GGUF refuses BY NAME, not as unknown") {
  // The container nothing writes yet. Before this wave a `deepseek41` file fell
  // through to the generic "architecture is not supported by this build", which
  // UNDERSTATES it: this build knows the architecture -- it resolves it and
  // validates its config -- and only the weight arm is missing.
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "deepseek41"));
  const gguf_test::TempFile file(b.Build());

  const std::string msg = LoadRefusalFor(gguf_test::Utf8Path(file.path()));
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("DeepseekV41ForCausalLM") != std::string::npos);
  CHECK(msg.find("GGUF") != std::string::npos);
  CHECK(msg.find("W8") != std::string::npos);
  CHECK(msg.find("MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm") !=
        std::string::npos);
  // The refusal that this case exists to prevent a regression to.
  CHECK(msg.find("is not supported by this build") == std::string::npos);
}

TEST_CASE("deepseek-v4.1: the KV-cache spec refuses BY NAME") {
  // Reached through the RESOLVED registration's own hook, so a registration
  // wired to the wrong builder would fail here.
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(Archs());
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->make_kv_cache != nullptr);
  std::string msg;
  try {
    (void)reg.factory->make_kv_cache(PublishedConfig(), /*block_size=*/64,
                                     /*num_blocks=*/8);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("DeepseekV41ForCausalLM") != std::string::npos);
  CHECK(msg.find("KV-cache") != std::string::npos);
  CHECK(msg.find("not ported") != std::string::npos);
}

TEST_CASE("deepseek-v4.1: the KV-cache hook validates the config FIRST") {
  // A wrong config must be reported as a wrong FIELD, not as a missing
  // KV-cache spec. Otherwise the refusal sends the reader to the wrong wave.
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(Archs());
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["hc_mult"] = 0;
  std::string msg;
  try {
    (void)reg.factory->make_kv_cache(ConfigFrom(doc), 64, 8);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("hc_mult") != std::string::npos);
  CHECK(msg.find("KV-cache") == std::string::npos);
}

TEST_CASE("deepseek-v4.1: the forward hook is registered and UNREACHABLE") {
  // W1 registers a forward that refuses. It cannot be entered from a production
  // entry point in this wave, and that is the wave's boundary rather than a
  // gap: reaching a forward needs a `LoadedModel`, and the only constructor of
  // one for this architecture is the loader, which refuses. This case pins BOTH
  // halves -- the hook exists, and the only door to it is shut -- so a later
  // wave that opens the loader without porting the forward turns this red.
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(Archs());
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->parse_config != nullptr);

  // The door: every load arm refuses, so no `LoadedModel` of this architecture
  // can exist and `forward` cannot be called.
  const TempSafetensorsDir dir;
  CHECK_FALSE(LoadRefusalFor(dir.path()).empty());
}

TEST_CASE("deepseek-v4.1: the forward REFUSES BY NAME, and says which waves") {
  // WHY THIS CASE EXISTS. `tests/CMakeLists.txt` and `docs/FEATURES.md` both
  // say every unimplemented arm -- load, FORWARD, KV-cache spec and the GGUF
  // container -- refuses BY NAME, and three of those four had a message
  // assertion while the forward had only `forward != nullptr`. That left the
  // claim about the forward gated by nothing: a forward rewritten to `return
  // {};` left this whole suite GREEN, caught only by
  // `tests/scripts/test_check_runner_routing_consistency.py` (and
  // `check-runner-routing-consistency.py` itself still exits 0 while moving
  // this model from REFUSE to the silently-exempt NONE bucket); a forward
  // gutted to `VT_CHECK(false, "boom")` was caught by NOTHING at all. The claim
  // is now gated where it is made.
  //
  // The hook is driven through the RESOLVED registration, the same way the
  // KV-cache case above is, so a registration wired to another architecture's
  // forward fails here.
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(Archs());
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->forward != nullptr);

  ForeignLoadedModel foreign(reg);
  EmptyForwardInput in;
  const vllm::ModelForwardInput input = in.Get();
  std::string msg;
  try {
    (void)reg.factory->forward(foreign, input);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE_MESSAGE(!msg.empty(), "the forward RETURNED instead of refusing");

  // BY NAME: this model, and not some generic "not implemented".
  CHECK(msg.find("DeepseekV41ForCausalLM") != std::string::npos);
  CHECK(msg.find("the forward is not ported") != std::string::npos);
  // The owed WORK, so the refusal is a next step and not a dead end. Same
  // shape as the KV-cache case's "not ported" + wave assertions.
  CHECK(msg.find("W3a") != std::string::npos);
  CHECK(msg.find("W4") != std::string::npos);
  CHECK(msg.find("MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm") !=
        std::string::npos);
  CHECK(msg.find("ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP") !=
        std::string::npos);
  // NOT one of the sibling refusals. Each of the four arms sends the reader to
  // a different wave, so a hook wired to the wrong one must not read as a pass.
  CHECK(msg.find("KV-cache") == std::string::npos);
  CHECK(msg.find("GGUF") == std::string::npos);
}

TEST_CASE("deepseek-v4.1: W1 claims NO load and NO forward") {
  // The scope statement, executable. If a later wave makes this architecture
  // load, this case is the one that must be deliberately rewritten -- which is
  // the point: the claim changes in the same change as the capability.
  const TempSafetensorsDir dir{UnquantizedVariant()};
  const std::string msg = LoadRefusalFor(dir.path());
  CHECK_FALSE(msg.empty());
  CHECK(msg.find("loads no weights in any format") != std::string::npos);
}
