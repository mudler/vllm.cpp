// GLM-5.3 (`GlmMoeDsaForCausalLM`) — the W2 gate.
//
// Spec `.agents/specs/glm-dsa-latest-deepseek.md` §3.7 W2, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214). vLLM parity pin
// `5559679229bc961848b121ccdeaa8fa5d79bec98`; llama.cpp secondary pin `b10451`.
//
// ─── WHAT THIS SUITE IS FOR ──────────────────────────────────────────────────
// The centrepiece is the first case. §3.5.1 of the spec claims that three
// independent sources agree bit for bit on which of GLM-5.3's 78 layers build a
// lightning indexer: the checkpoint's own `indexer_types` list, vLLM's DERIVED
// rule at `deepseek_v2.py:1097-1101`, and llama.cpp's hardcoded
// `GLM_5_2_DEFAULT_INDEXER_TYPES` (`b10451:src/models/glm-dsa.cpp:6-27`). A
// claim in a spec is prose; this file makes it executable against the
// checkpoint's config.json as shipped.
//
// It matters because the port DERIVES the schedule. A checkpoint that ships the
// list can be read; the published GGUF ships neither the list nor the freq and
// offset keys, so a reader that synthesizes the wrong schedule builds 78
// indexers where the reference builds 21, produces plausible tokens, and
// reports nothing. There is no token gate on this fleet to catch that
// (spec O1), which is why the agreement is gated at the config layer instead.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "glm_moe_dsa_config_glm53.inc"

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/glm_moe_dsa.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"

#include "../gguf_builder.h"

namespace {

using vllm::GlmMoeDsaIndexerKind;
using vllm::GlmMoeDsaMlpKind;

// llama.cpp's own answer to the same question, transcribed from
// `b10451:src/models/glm-dsa.cpp:6-27`. It is admissible as a THIRD source
// precisely because it is not derived from anything in this repository: it is a
// constant another project wrote from the published `config.json` of GLM-5.2,
// and the spec's §3.5.1 claim is that GLM-5.3's list is bit-identical to it over
// all 78 backbone entries. `GLM_5_2_DEFAULT_INDEXER_TYPES` is declared over
// `LLAMA_MAX_LAYERS` and zero-fills past its 78th initializer; only the first
// 78 are compared, because that is the extent the claim covers.
constexpr const char* kLlamaCppDefaultIndexerTypes =
    "111000100010001000100010001000100010001000100010001000100010001000100010"
    "001000";

nlohmann::json Glm53Doc() {
  return nlohmann::json::parse(glm_moe_dsa_fixture::kGlm53ConfigJson);
}

vllm::HfConfig Glm53Config() {
  return vllm::ParseHfConfig(Glm53Doc(), "zai-org/GLM-5.3 config.json");
}

vllm::HfConfig ConfigFrom(const nlohmann::json& doc) {
  return vllm::ParseHfConfig(doc, "test config");
}

std::string Bits(const std::vector<GlmMoeDsaIndexerKind>& s) {
  std::string out;
  for (GlmMoeDsaIndexerKind k : s) {
    out.push_back(k == GlmMoeDsaIndexerKind::kFull ? '1' : '0');
  }
  return out;
}

// Runs `fn`, requires it to throw, and returns the message. FAILS the case when
// nothing is thrown, so a refusal that silently stopped refusing cannot read as
// a pass.
template <typename Fn>
std::string RefusalOf(Fn fn, const char* what) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  FAIL_CHECK("expected a refusal from " << what << " and none was thrown");
  return {};
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// 1. THE CENTREPIECE
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE(
    "glm_moe_dsa: the DERIVED indexer schedule reproduces the checkpoint's own "
    "`indexer_types`, all 78 entries") {
  const nlohmann::json doc = Glm53Doc();

  // The checkpoint's own list, read off the committed config.json.
  const std::vector<std::string> shipped =
      doc.at("indexer_types").get<std::vector<std::string>>();
  const int64_t layers = doc.at("num_hidden_layers").get<int64_t>();
  REQUIRE(layers == 78);
  REQUIRE(shipped.size() == 78u);

  std::string shipped_bits;
  for (const std::string& s : shipped) {
    REQUIRE((s == "full" || s == "shared"));
    shipped_bits.push_back(s == "full" ? '1' : '0');
  }

  // The list is NOT degenerate. A schedule of all-`full` or all-`shared` would
  // be reproduced by a broken derivation too, so the discriminating power of
  // this case rests on the list carrying both kinds — and on how many of each.
  const size_t full_count =
      static_cast<size_t>(std::count(shipped_bits.begin(), shipped_bits.end(), '1'));
  CHECK(full_count == 21u);
  CHECK(full_count < shipped_bits.size());
  CHECK(full_count > 0u);

  // vLLM's DERIVED rule, `deepseek_v2.py:1097-1101`, evaluated over the
  // checkpoint's own freq and offset.
  const int64_t freq = doc.at("index_topk_freq").get<int64_t>();
  const int64_t offset = doc.at("index_skip_topk_offset").get<int64_t>();
  CHECK(freq == 4);
  CHECK(offset == 3);
  const std::vector<GlmMoeDsaIndexerKind> derived =
      vllm::DeriveGlmMoeDsaIndexerSchedule(layers, freq, offset);
  REQUIRE(derived.size() == 78u);

  // Entry by entry, so a failure names the layer rather than the string.
  for (size_t i = 0; i < shipped.size(); ++i) {
    CAPTURE(i);
    CHECK(Bits(derived)[i] == shipped_bits[i]);
  }
  CHECK(Bits(derived) == shipped_bits);

  // And llama.cpp's independent constant, over the same 78 entries.
  CHECK(std::string(kLlamaCppDefaultIndexerTypes).size() == 78u);
  CHECK(shipped_bits == std::string(kLlamaCppDefaultIndexerTypes));

  // The derivation is not a constant: a different period gives a different
  // schedule. Without this a `DeriveGlmMoeDsaIndexerSchedule` that ignored its
  // arguments and returned the published bitstring would pass everything above.
  CHECK(Bits(vllm::DeriveGlmMoeDsaIndexerSchedule(layers, 2, offset)) !=
        shipped_bits);
  CHECK(Bits(vllm::DeriveGlmMoeDsaIndexerSchedule(layers, freq, 0)) !=
        shipped_bits);
  CHECK(Bits(vllm::DeriveGlmMoeDsaIndexerSchedule(layers, 1, offset)) ==
        std::string(78, '1'));
}

TEST_CASE(
    "glm_moe_dsa: the DERIVED dense/MoE layout reproduces the checkpoint's own "
    "`mlp_layer_types`") {
  const nlohmann::json doc = Glm53Doc();
  const std::vector<std::string> shipped =
      doc.at("mlp_layer_types").get<std::vector<std::string>>();
  REQUIRE(shipped.size() == 78u);
  const std::vector<GlmMoeDsaMlpKind> derived = vllm::DeriveGlmMoeDsaMlpSchedule(
      78, doc.at("first_k_dense_replace").get<int64_t>(),
      doc.at("moe_layer_freq").get<int64_t>(),
      doc.at("n_routed_experts").get<int64_t>());
  REQUIRE(derived.size() == 78u);
  size_t dense = 0;
  for (size_t i = 0; i < shipped.size(); ++i) {
    CAPTURE(i);
    const bool want_dense = shipped[i] == "dense";
    CHECK(want_dense == (derived[i] == GlmMoeDsaMlpKind::kDense));
    if (want_dense) ++dense;
  }
  CHECK(dense == 3u);  // first_k_dense_replace
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. THE FULL RESOLVE
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("glm_moe_dsa: the real config.json RESOLVES to the published geometry") {
  const vllm::GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(Glm53Config());

  CHECK(p.num_hidden_layers == 78);
  CHECK(p.hidden_size == 6144);
  CHECK(p.vocab_size == 154880);
  CHECK(p.num_attention_heads == 64);
  CHECK(p.intermediate_size == 12288);

  CHECK(p.q_lora_rank == 2048);
  CHECK(p.kv_lora_rank == 512);
  CHECK(p.qk_nope_head_dim == 192);
  CHECK(p.qk_rope_head_dim == 64);
  CHECK(p.v_head_dim == 256);
  // The MLA geometry `mla::MlaBlockDims::Validate` has to accept:
  // `v_head_dim <= qk_nope_head_dim + qk_rope_head_dim` (mla_attention.cpp).
  CHECK(p.v_head_dim <= p.qk_nope_head_dim + p.qk_rope_head_dim);
  CHECK(p.mla_kv_head_size() == 576);

  CHECK(p.n_routed_experts == 256);
  CHECK(p.num_experts_per_tok == 8);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.moe_intermediate_size == 2048);
  CHECK(p.first_k_dense_replace == 3);
  CHECK(p.n_group == 1);
  CHECK(p.topk_group == 1);
  CHECK(p.norm_topk_prob);
  CHECK(p.has_e_score_correction_bias);
  CHECK(p.routed_scaling_factor == doctest::Approx(2.5));
  // `_get_moe_router_dtype` forces f32 on `model_type == "glm_moe_dsa"`
  // (deepseek_v2.py:127), ahead of the generic `moe_router_dtype` branch.
  CHECK(p.router_dtype_is_f32);

  CHECK(p.index_topk == 2048);
  CHECK(p.index_n_heads == 32);
  CHECK(p.index_head_dim == 128);
  CHECK(p.index_topk_freq == 4);
  CHECK(p.index_skip_topk_offset == 3);

  CHECK(p.num_nextn_predict_layers == 1);
  CHECK(p.index_share_for_mtp_iteration);

  REQUIRE(p.indexer_types.size() == 78u);
  CHECK(std::count(p.indexer_types.begin(), p.indexer_types.end(),
                   GlmMoeDsaIndexerKind::kFull) == 21);
  REQUIRE(p.mlp_layer_types.size() == 78u);
  CHECK(std::count(p.mlp_layer_types.begin(), p.mlp_layer_types.end(),
                   GlmMoeDsaMlpKind::kDense) == 3);
  CHECK(!p.is_moe_layer(2));
  CHECK(p.is_moe_layer(3));
}

TEST_CASE(
    "glm_moe_dsa: `is_neox_style` is ASSERTED false, and no config key moves it") {
  // Upstream passes `is_neox_style=False` unconditionally at
  // `deepseek_v2.py:1073`. Our `mla::MlaBlockDims::is_neox_style` ALSO defaults
  // false, so the value is right today by two defaults agreeing. This case is
  // what turns it into a read.
  CHECK(vllm::ParseGlmMoeDsaParams(Glm53Config()).is_neox_style == false);

  nlohmann::json doc = Glm53Doc();
  doc["rope_interleave"] = false;
  doc["is_neox_style"] = true;
  doc["rope_parameters"]["rope_type"] = "default";
  CHECK(vllm::ParseGlmMoeDsaParams(ConfigFrom(doc)).is_neox_style == false);

  // The INDEXER's rope is a different field and IS config-driven:
  // `is_neox_style=not getattr(config, "indexer_rope_interleave", False)`
  // (deepseek_v2.py:1120). GLM-5.3 ships `indexer_rope_interleave: true`.
  CHECK(vllm::ParseGlmMoeDsaParams(Glm53Config()).indexer_rope_is_neox_style ==
        false);
  nlohmann::json without = Glm53Doc();
  without.erase("indexer_rope_interleave");
  CHECK(vllm::ParseGlmMoeDsaParams(ConfigFrom(without))
            .indexer_rope_is_neox_style == true);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. THE REFUSALS
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE(
    "glm_moe_dsa: a `mlp_layer_types` that disagrees with "
    "`first_k_dense_replace` is REFUSED") {
  // Upstream reads `first_k_dense_replace` and never reads `mlp_layer_types`
  // (`grep -c mlp_layer_types` over deepseek_v2.py at the pin is 0). A
  // checkpoint whose two descriptions disagree therefore has one wrong
  // description, and the one this port would follow is not the one a reader of
  // the config.json would expect. It refuses instead.
  nlohmann::json doc = Glm53Doc();
  doc["mlp_layer_types"][7] = "dense";  // derived layout says `sparse`
  const std::string msg = RefusalOf([&] { vllm::ParseGlmMoeDsaParams(ConfigFrom(doc)); },
                                    "ParseGlmMoeDsaParams");
  CHECK(msg.find("layer 7") != std::string::npos);
  CHECK(msg.find("mlp_layer_types") != std::string::npos);
  CHECK(msg.find("first_k_dense_replace") != std::string::npos);
  CHECK(msg.find("dense") != std::string::npos);
  CHECK(msg.find("sparse") != std::string::npos);

  // The other direction, so a one-sided comparison cannot pass.
  nlohmann::json other = Glm53Doc();
  other["mlp_layer_types"][1] = "sparse";  // derived layout says `dense`
  const std::string msg2 = RefusalOf(
      [&] { vllm::ParseGlmMoeDsaParams(ConfigFrom(other)); }, "ParseGlmMoeDsaParams");
  CHECK(msg2.find("layer 1") != std::string::npos);

  // A wrong LENGTH is refused too, and by a different message.
  nlohmann::json shortened = Glm53Doc();
  shortened["mlp_layer_types"].erase(0);
  const std::string msg3 = RefusalOf(
      [&] { vllm::ParseGlmMoeDsaParams(ConfigFrom(shortened)); },
      "ParseGlmMoeDsaParams");
  CHECK(msg3.find("77 entries") != std::string::npos);

  // And the UNMODIFIED checkpoint passes, so the case above is measuring the
  // edit and not a parser that refuses everything.
  CHECK_NOTHROW(vllm::ParseGlmMoeDsaParams(Glm53Config()));
}

TEST_CASE("glm_moe_dsa: the explicit `indexer_types` list OVERRIDES the derivation") {
  // Spec §3.7 W2: derived rule, with the list as an override. It is the
  // spelling the GGUF arm must use, because llama.cpp's converter writes the
  // list and writes no freq/offset key at all.
  nlohmann::json doc = Glm53Doc();
  for (size_t i = 3; i < 78; ++i) doc["indexer_types"][i] = "full";
  const vllm::GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(ConfigFrom(doc));
  CHECK(std::count(p.indexer_types.begin(), p.indexer_types.end(),
                   GlmMoeDsaIndexerKind::kFull) == 78);

  // A first entry of `shared` reuses a selection that does not exist.
  nlohmann::json bad = Glm53Doc();
  bad["indexer_types"][0] = "shared";
  const std::string msg =
      RefusalOf([&] { vllm::ParseGlmMoeDsaParams(ConfigFrom(bad)); },
                "ParseGlmMoeDsaParams");
  CHECK(msg.find("first layer") != std::string::npos);

  // A wrong length, and an unknown code.
  nlohmann::json shortened = Glm53Doc();
  shortened["indexer_types"].erase(0);
  CHECK(RefusalOf([&] { vllm::ParseGlmMoeDsaParams(ConfigFrom(shortened)); },
                  "ParseGlmMoeDsaParams")
            .find("77 entries") != std::string::npos);
  nlohmann::json unknown = Glm53Doc();
  unknown["indexer_types"][5] = "sparse";
  CHECK(RefusalOf([&] { vllm::ParseGlmMoeDsaParams(ConfigFrom(unknown)); },
                  "ParseGlmMoeDsaParams")
            .find("neither `full` nor `shared`") != std::string::npos);
}

TEST_CASE(
    "glm_moe_dsa: a config stating NEITHER the list nor freq/offset is REFUSED, "
    "not defaulted") {
  // Spec D3. Upstream's `getattr(config, "index_topk_freq", 1)` would make
  // every layer `full`; llama.cpp falls back to a hardcoded 78-entry table that
  // is right for exactly today's checkpoints. Neither is acceptable here.
  nlohmann::json doc = Glm53Doc();
  doc.erase("indexer_types");
  doc.erase("index_topk_freq");
  doc.erase("index_skip_topk_offset");
  const std::string msg = RefusalOf(
      [&] { vllm::ParseGlmMoeDsaParams(ConfigFrom(doc)); }, "ParseGlmMoeDsaParams");
  CHECK(msg.find("indexer_types") != std::string::npos);
  CHECK(msg.find("index_topk_freq") != std::string::npos);
  CHECK(msg.find("glm-dsa.cpp:6-27") != std::string::npos);
  CHECK(msg.find("D3") != std::string::npos);

  // Keeping EITHER source is enough, and the two give the same 78-entry answer.
  nlohmann::json list_only = Glm53Doc();
  list_only.erase("index_topk_freq");
  list_only.erase("index_skip_topk_offset");
  nlohmann::json derive_only = Glm53Doc();
  derive_only.erase("indexer_types");
  CHECK(Bits(vllm::ParseGlmMoeDsaParams(ConfigFrom(list_only)).indexer_types) ==
        Bits(vllm::ParseGlmMoeDsaParams(ConfigFrom(derive_only)).indexer_types));
}

TEST_CASE("glm_moe_dsa: a non-`glm_moe_dsa` model_type is refused by name") {
  nlohmann::json doc = Glm53Doc();
  doc["model_type"] = "deepseek_v3";
  const std::string msg = RefusalOf(
      [&] { vllm::ParseGlmMoeDsaParams(ConfigFrom(doc)); }, "ParseGlmMoeDsaParams");
  CHECK(msg.find("deepseek_v3") != std::string::npos);
  CHECK(msg.find("glm_moe_dsa") != std::string::npos);
}

TEST_CASE("glm_moe_dsa: the forward refusal NAMES each missing primitive") {
  const std::string msg = vllm::GlmMoeDsaForwardRefusal();
  // What is STILL missing, and only that. Two primitives, each with the record
  // that owns it.
  for (const char* needle :
       {"indexer KV side cache", "DeepseekV32IndexerCache", "sparse prefill",
        "MlaPrefillAttentionArgs", "MlaPrefillAttention", "#1925", "#2323",
        "W6", "W7", "§3.7", ".agents/specs/glm-dsa-latest-deepseek.md"}) {
    CAPTURE(needle);
    CHECK(msg.find(needle) != std::string::npos);
  }
  // The predicate rather than a line range (spec O20): the range moved twice
  // while the message was being written.
  CHECK(msg.find("!elig.prunes || elig.Active()") != std::string::npos);

  // AND WHAT IT MUST NO LONGER NAME. This half is the point of the case: W2
  // wrote a seven-item list and five items have since landed, so a refusal that
  // still named them would send its reader looking for work that is done. Each
  // needle below is a thing this build HAS, and the assertion is that the
  // message stopped claiming otherwise.
  //
  //   `IQ4_XS`      -> `VecDotIQ4_XSQ8_K`, `2e9f4d88d` (spec O2, DISCHARGED)
  //   `qwen3_5.cpp` -> the seam is `expert_stream_seam.{h,cpp}` (spec O8)
  //   `selection-reuse` / `mla.py:180` -> `GlmMoeDsaMlaSchedule` (W4)
  //   `fp32 router` -> the forward sizes `dlog` from `router_dtype_is_f32`
  //   `1147-1180`   -> the wrong `dots3_note_device.cpp` range (spec O20)
  //   THE FORWARD ITSELF -> W9, `glm_moe_dsa_forward.cpp`
  for (const char* stale :
       {"IQ4_XS", "welded", "selection-reuse", "fp32 router GEMM", "1147-1180",
        "QUANT-GGUF-IQ4_XS", "forward is not implemented"}) {
    CAPTURE(stale);
    CHECK(msg.find(stale) == std::string::npos);
  }

  // The loader is no longer owed, and neither is the forward. W9 turned this
  // message from "the forward does not exist" into "this STEP cannot be
  // served", which is a different and much narrower claim.
  CHECK(msg.find("forward IS implemented") != std::string::npos);
  CHECK(msg.find("A FIRST token on a fresh prompt is reachable") !=
        std::string::npos);
}

TEST_CASE("glm_moe_dsa: the forward refuses a model that never went through the loader") {
  // W9 (#2214). This case used to assert that the forward refuses EVERY call.
  // It no longer does — `tests/vllm/models/test_glm_moe_dsa_forward.cpp` drives
  // a token through it — so what is gated here is the precondition a
  // hand-constructed `GlmMoeDsaWeights` violates: only
  // `LoadGlmMoeDsaFromGguf` runs the post-load absorption that produces
  // `kv_b_proj`, `w_uk_t` and `w_uv`, and reading those empty would fail one
  // frame deeper with a message about a single missing tensor rather than about
  // a stage that did not run.
  vllm::GlmMoeDsaWeights weights;
  vllm::v1::CommonAttentionMetadata meta;
  std::vector<vllm::PagedKvCache> kv;
  vt::Queue q;  // CPU-default; the refusal fires before it is touched
  const std::vector<int32_t> ids = {1};
  const std::vector<int32_t> pos = {0};
  const std::string thrown = RefusalOf(
      [&] { vllm::GlmMoeDsaModel::Forward(ids, pos, meta, kv, weights, q, {}); },
      "GlmMoeDsaModel::Forward");
  CHECK(thrown.find("post-load absorption") != std::string::npos);
  const std::string thrown_dev = RefusalOf(
      [&] {
        (void)vllm::GlmMoeDsaModel::ForwardDevice(ids, pos, meta, kv, weights, q, {});
      },
      "GlmMoeDsaModel::ForwardDevice");
  CHECK(thrown_dev.find("post-load absorption") != std::string::npos);
}

TEST_CASE("glm_moe_dsa: the safetensors arm refuses permanently, and says why") {
  const vllm::HfConfig config = Glm53Config();
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  vllm::ModelSource source;
  source.kind = vllm::ModelSource::Kind::kSafetensors;
  const std::string msg = RefusalOf(
      [&] { (void)reg.factory->load_weights(reg, config, source); },
      "the safetensors load_weights arm");
  CHECK(msg.find("703.74 GiB") != std::string::npos);
  CHECK(msg.find("block-fp8") != std::string::npos);
  CHECK(msg.find("D1") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. REACHABILITY — the registry and the GGUF dispatch table
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("glm_moe_dsa: the architecture RESOLVES through the registry") {
  const std::vector<std::string_view> archs = vllm::ModelRegistry::SupportedArchs();
  CHECK(std::find(archs.begin(), archs.end(), "GlmMoeDsaForCausalLM") !=
        archs.end());
  const vllm::HfConfig config = Glm53Config();
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "GlmMoeDsaForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(!reg.factory->is_dense_model);
  CHECK(reg.info.is_text_generation_model);
  CHECK(!reg.info.is_hybrid);
  CHECK(!reg.info.supports_multimodal);

  // The config hook is the resolve, so it refuses what the resolve refuses.
  CHECK_NOTHROW(reg.factory->parse_config(Glm53Config()));

  // One MLA group, `kv_lora_rank + qk_rope_head_dim` wide. The indexer's own
  // side cache is a SECOND group upstream and does not exist here (spec O4).
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(Glm53Config(), 16, 8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.num_blocks == 8);
}

namespace {

// A byte-exact miniature of a `b10451`-converted `glm-dsa` file: five blocks of
// which one is the multi-token-prediction block, so the backbone is four layers
// deep and `block_count` is NOT `num_hidden_layers`.
//
// The indexer schedule is `full, shared, full, shared`, which upstream's
// freq/offset rule cannot produce at any (freq, offset) that also puts layer 0
// full and layer 1 shared with period 2 starting at 0 — it is READ, not
// synthesized, and no freq/offset key is written into the file because
// llama.cpp's converter writes none (`b10451:conversion/glm.py:333-339`).
std::string BuildGlmDsaFixture(bool with_indexer_types = true,
                               int indexer_entries = 4) {
  gguf_test::GgufModelBuilder b;
  const std::string p = "glm-dsa.";
  b.AddKv(gguf_test::StrKv("general.architecture", "glm-dsa"));
  b.AddKv(gguf_test::U32Kv(p + "block_count", 5));
  b.AddKv(gguf_test::U32Kv(p + "nextn_predict_layers", 1));
  b.AddKv(gguf_test::U32Kv(p + "embedding_length", 64));
  b.AddKv(gguf_test::U32Kv(p + "context_length", 1048576));
  b.AddKv(gguf_test::U32Kv(p + "attention.head_count", 4));
  b.AddKv(gguf_test::U32Kv(p + "attention.head_count_kv", 1));
  b.AddKv(gguf_test::U32Kv(p + "feed_forward_length", 128));
  b.AddKv(gguf_test::F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-5f));
  b.AddKv(gguf_test::F32Kv(p + "rope.freq_base", 8e6f));
  // The MLA latent, stated three times the way llama.cpp states it.
  b.AddKv(gguf_test::U32Kv(p + "attention.kv_lora_rank", 32));
  b.AddKv(gguf_test::U32Kv(p + "rope.dimension_count", 8));
  b.AddKv(gguf_test::U32Kv(p + "attention.key_length", 40));   // 32 + 8
  b.AddKv(gguf_test::U32Kv(p + "attention.key_length_mla", 24));   // 16 + 8
  b.AddKv(gguf_test::U32Kv(p + "attention.value_length_mla", 24));
  b.AddKv(gguf_test::U32Kv(p + "attention.q_lora_rank", 48));
  // MoE.
  b.AddKv(gguf_test::U32Kv(p + "expert_count", 8));
  b.AddKv(gguf_test::U32Kv(p + "expert_used_count", 2));
  b.AddKv(gguf_test::U32Kv(p + "expert_feed_forward_length", 32));
  b.AddKv(gguf_test::U32Kv(p + "expert_shared_count", 1));
  b.AddKv(gguf_test::U32Kv(p + "leading_dense_block_count", 1));
  b.AddKv(gguf_test::F32Kv(p + "expert_weights_scale", 2.5f));
  b.AddKv(gguf_test::BoolKv(p + "expert_weights_norm", true));
  // The indexer.
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.head_count", 4));
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.key_length", 16));
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.top_k", 64));
  if (with_indexer_types) {
    std::vector<bool> types;
    for (int i = 0; i < indexer_entries; ++i) types.push_back(i % 2 == 0);
    b.AddKv(gguf_test::BoolArrayKv(p + "attention.indexer.types", types));
  }
  // A miniature byte-level BPE vocabulary, in the shape `Tokenizer::FromGguf`
  // requires. `pre = "glm4"` is the value every GLM GGUF in the ecosystem
  // carries, including `unsloth/GLM-5.3-Flash-GGUF` (#2277). It is here because
  // `FromModelDir` builds the tokenizer BEFORE it loads weights, so a file
  // without one never reaches this architecture's loader and the reachability
  // case below would be measuring the tokenizer instead.
  {
    b.AddKv(gguf_test::StrKv("tokenizer.ggml.model", "gpt2"));
    b.AddKv(gguf_test::StrKv("tokenizer.ggml.pre", "glm4"));
    std::vector<std::string> toks;
    std::vector<int32_t> types;
    for (int i = 0; i < 32; ++i) {
      toks.push_back("t" + std::to_string(i));
      types.push_back(1);
    }
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.tokens", toks));
    b.AddKv(gguf_test::I32ArrayKv("tokenizer.ggml.token_type", types));
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.merges",
                                  std::vector<std::string>{}));
  }
  b.AddTensor("token_embd.weight", {64, 32}, 0, std::string(64 * 32 * 4, '\0'));
  return b.Build();
}

}  // namespace

TEST_CASE(
    "glm_moe_dsa GGUF: a `glm-dsa` header reaches the config builder through "
    "LoadedEngine::FromModelDir") {
  // THE REACHABILITY CASE. It enters through the production entry point, so it
  // fails if either the `kGgufArchArms` row or the `REGISTER_VLLM_MODEL` line is
  // removed — the two mutations spec §3.7 W2 requires. A unit test over
  // `GlmMoeDsaHfConfigFromGguf` would prove the builder works and nothing about
  // whether anything reaches it.
  gguf_test::TempFile f(BuildGlmDsaFixture());
  vllm::entrypoints::EngineParams params;
  const std::string msg = RefusalOf(
      [&] { (void)vllm::entrypoints::LoadedEngine::FromModelDir(f.path(), params); },
      "LoadedEngine::FromModelDir");

  MESSAGE("FromModelDir said: " << msg);
  // NOT the dispatch table's default arm: the `glm-dsa` row was found.
  CHECK(msg.find("is not supported by this build") == std::string::npos);
  // NOT the registry's unsupported-architecture message: `GlmMoeDsaForCausalLM`
  // resolved from the config the builder synthesized.
  CHECK(msg.find("are not supported for now") == std::string::npos);
  // IT REACHED THE WEIGHT LOADER. W2 asserted the loader's refusal text here,
  // because at W2 the GGUF arm's whole body was a `throw` naming the wave. W7
  // replaced that throw with `LoadGlmMoeDsaFromGguf`, so what this header-only
  // fixture now produces is the loader asking for the second tensor it needs
  // and not finding it — which is a STRICTLY DEEPER reach than W2 could prove,
  // and still fails if either the `kGgufArchArms` row or the
  // `REGISTER_VLLM_MODEL` line is removed.
  //
  // The complete-model load, with every tensor present and the accounting
  // asserted, is `test_glm_moe_dsa_gguf_load.cpp`. This case keeps the narrow
  // question — does a `glm-dsa` header reach this architecture at all — because
  // that is the one a fixture without weights can answer.
  CHECK(msg.find("no tensor named") != std::string::npos);
  CHECK(msg.find("output_norm.weight") != std::string::npos);
}

TEST_CASE("glm_moe_dsa GGUF: the header builds the config the config.json does") {
  gguf_test::TempFile f(BuildGlmDsaFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  CHECK(vllm::IsGlmMoeDsaGguf(g));
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);
  CHECK(c.model_type == "glm_moe_dsa");
  REQUIRE(c.architectures.size() == 1);
  CHECK(c.architectures[0] == "GlmMoeDsaForCausalLM");
  // BLOCKS ARE NOT LAYERS: `block_count` 5 minus one MTP block.
  CHECK(c.num_hidden_layers == 4);
  CHECK(c.vocab_size == 32);  // from tokenizer.ggml.tokens

  const vllm::GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(c);
  CHECK(p.kv_lora_rank == 32);
  CHECK(p.qk_rope_head_dim == 8);
  CHECK(p.qk_nope_head_dim == 16);  // key_length_mla - rope.dimension_count
  CHECK(p.v_head_dim == 24);
  CHECK(p.q_lora_rank == 48);
  CHECK(p.n_routed_experts == 8);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.first_k_dense_replace == 1);
  CHECK(p.num_nextn_predict_layers == 1);
  CHECK(p.index_topk == 64);
  CHECK(p.index_n_heads == 4);
  CHECK(p.index_head_dim == 16);
  // The schedule is the FILE's, which no freq/offset would derive.
  CHECK(Bits(p.indexer_types) == "1010");
  CHECK(p.mlp_layer_types[0] == GlmMoeDsaMlpKind::kDense);
  CHECK(p.mlp_layer_types[1] == GlmMoeDsaMlpKind::kSparse);

  // The registry resolves the synthesized config, so both sources meet one
  // validator and one registration.
  CHECK(vllm::ModelRegistry::Resolve(c).architecture == "GlmMoeDsaForCausalLM");
}

TEST_CASE(
    "glm_moe_dsa GGUF: a file that states NO indexer schedule is refused, and "
    "llama.cpp's hardcoded table is not substituted") {
  // This is the shape of the PUBLISHED artifact: spec D3 records that
  // `unsloth/GLM-5.3-GGUF` declares indexer weights on all 79 blocks and writes
  // no `glm-dsa.attention.indexer.types`. llama.cpp survives it with
  // `GLM_5_2_DEFAULT_INDEXER_TYPES`; we refuse it by name instead.
  gguf_test::TempFile f(BuildGlmDsaFixture(/*with_indexer_types=*/false));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const std::string msg = RefusalOf(
      [&] { (void)vllm::GlmMoeDsaHfConfigFromGguf(g); },
      "GlmMoeDsaHfConfigFromGguf");
  CHECK(msg.find("indexer_types") != std::string::npos);
  CHECK(msg.find("index_topk_freq") != std::string::npos);
  CHECK(msg.find("D3") != std::string::npos);
}

TEST_CASE("glm_moe_dsa GGUF: a wrong-length indexer schedule is refused") {
  // The converter writes the TRUNK list, so a five-entry schedule against a
  // four-layer backbone is the mistake a reader who takes `block_count` for the
  // depth makes. Checked against the backbone, not against `block_count`.
  gguf_test::TempFile f(BuildGlmDsaFixture(true, /*indexer_entries=*/5));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const std::string msg = RefusalOf(
      [&] { (void)vllm::GlmMoeDsaHfConfigFromGguf(g); },
      "GlmMoeDsaHfConfigFromGguf");
  CHECK(msg.find("5 entries") != std::string::npos);
  CHECK(msg.find("4 layers deep") != std::string::npos);
}
