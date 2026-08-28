// GLM-5.3-Flash W1 scaffold gate (#2067, row
// MODEL-MM-glm5-next-glm5-next-for-conditional-generation).
//
// WHAT THIS GATES, and what it deliberately does not. It gates the CONFIG
// LAYER and the two production entry points that reach it: `ModelRegistry`
// resolution and the `general.architecture` GGUF dispatch. It gates NO
// numerics, because there are none in W1, and it cannot gate the MODEL at all
// — no oracle can execute `Glm5NextForConditionalGeneration` on any device this
// project reaches (the smallest published artifact is 181.32 GiB against
// ~119.63 GiB on GB10), which is O1 in `.agents/specs/glm5-next-flash.md`.
//
// THE ORACLE FOR THIS SURFACE is transformers **v5.16.1**
// (`models/glm5_next/configuration_glm5_next.py`, implementing commit
// `eb4d9e2a64`; `v5.16.0` is 404 for this model). vLLM implements `glm5_next`
// at no revision. Every expectation below is either read from the REAL
// published `config.json` (checked in verbatim as this test's fixture) or is a
// clause of that class's `__post_init__` / `validate_architecture`.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "gguf_builder.h"
#include "nlohmann/json.hpp"
#include "support/process_id.h"  // vllm_test::ProcessId, for a unique temp dir
#include "vllm/entrypoints/model_loader.h"  // LoadedEngine::FromModelDir
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_weights.h"
#include "vllm/model_executor/models/mla_attention.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, *KvCache
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vt/device.h"

using vllm::Glm5NextExpectedGgufTensors;
using vllm::Glm5NextIndexerKind;
using vllm::Glm5NextLayerKind;
using vllm::Glm5NextMlpKind;
using vllm::Glm5NextParams;
using vllm::HfConfig;
using vllm::LoadedModel;
using vllm::ModelForwardInput;
using vllm::ModelRegistration;
using vllm::ModelRegistry;
using vllm::ParseGlm5NextParams;

namespace {

// A model of a DIFFERENT registration, which is the only handle any caller can
// present while `load_weights` refuses unconditionally. Passing one proves the
// refusal fires BEFORE any downcast: a `ModelAs<...>` placed first would turn
// this into a type-mismatch report and leave the advertised refusal dead.
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

// The REAL published config.json, byte-for-byte. A hand-written fixture would
// gate this port against whatever the port's author believed the checkpoint
// says; this gates it against what the checkpoint says.
nlohmann::json PublishedConfigJson() {
  const std::string path =
      std::string(GLM5_NEXT_CKPT_FIXTURE_DIR) + "/config.json";
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "missing fixture: " << path);
  return nlohmann::json::parse(in);
}

HfConfig ConfigFrom(const nlohmann::json& doc) {
  return vllm::ParseHfConfig(doc, std::string(GLM5_NEXT_CKPT_FIXTURE_DIR) +
                                      "/config.json");
}

HfConfig PublishedConfig() { return ConfigFrom(PublishedConfigJson()); }

}  // namespace

TEST_CASE("glm5_next: the published config resolves to the published geometry") {
  const Glm5NextParams p = ParseGlm5NextParams(PublishedConfig());

  CHECK(p.hidden_size == 4096);
  CHECK(p.num_hidden_layers == 45);
  CHECK(p.intermediate_size == 12288);
  CHECK(p.vocab_size == 154880);
  CHECK(p.num_attention_heads == 64);
  CHECK(p.num_key_value_heads == 64);
  CHECK(p.max_position_embeddings == 1048576);
  // 1e-5, and NOT the 1e-6 the `GlmMoeDsa` parent constructs its MLA LoRA norms
  // with: `Glm5NextTextAttention.__init__` passes `eps` explicitly.
  // EXACT, not `Approx`. doctest's `Approx` carries a scale term that makes
  // 1e-6 compare equal to 1e-5, so an Approx assertion on either of this
  // model's two epsilons is a mute switch rather than a gate. Both values are
  // parsed from JSON doubles and are exactly representable decisions.
  CHECK(p.rms_norm_eps == 1e-5);
  CHECK(p.swiglu_limit == doctest::Approx(10.0));
  CHECK_FALSE(p.tie_word_embeddings);

  // The interleave: 34 KDA and 11 DSA at 3, 7, ..., 43.
  CHECK(p.num_kda_layers() == 34);
  CHECK(p.num_dsa_layers() == 11);
  for (int64_t i = 0; i < 45; ++i) {
    CAPTURE(i);
    const bool dsa = (i % 4) == 3;
    CHECK(p.layer_types[static_cast<size_t>(i)] ==
          (dsa ? Glm5NextLayerKind::kDeepseekSparseAttention
               : Glm5NextLayerKind::kLinearAttention));
  }
  // `mlp_layer_types`: dense on 0-2, sparse thereafter.
  for (int64_t i = 0; i < 45; ++i) {
    CAPTURE(i);
    CHECK(p.mlp_layer_types[static_cast<size_t>(i)] ==
          (i < 3 ? Glm5NextMlpKind::kDense : Glm5NextMlpKind::kSparse));
  }
  for (const Glm5NextIndexerKind k : p.indexer_types) {
    CHECK(k == Glm5NextIndexerKind::kFull);
  }

  // The residual manifold is 4 * 4096 wide through the WHOLE stack.
  CHECK(p.mhc.mult == 4);
  CHECK(p.mhc.sinkhorn_iters == 20);
  // hc_eps is a DIFFERENT constant from rms_norm_eps. If a port ever collapses
  // the two this case is what says so.
  CHECK(p.mhc.eps == 1e-6);
  CHECK(p.mhc.eps != p.rms_norm_eps);
  CHECK(p.residual_stream_width() == 16384);

  CHECK(p.moe.n_routed_experts == 288);
  CHECK(p.moe.n_shared_experts == 1);
  CHECK(p.moe.num_experts_per_tok == 8);
  CHECK(p.moe.moe_intermediate_size == 2048);
  // n_group == topk_group == 1 makes the group stage a NO-OP. Recorded so a
  // later wave does not implement masked group selection that cannot change
  // the result.
  CHECK(p.moe.n_group == 1);
  CHECK(p.moe.topk_group == 1);
  CHECK(p.moe.routed_scaling_factor == doctest::Approx(2.5));
  CHECK(p.moe.norm_topk_prob);

  CHECK(p.has_vision);
  CHECK(p.vision.depth == 24);
  CHECK(p.vision.hidden_size == 1024);
  CHECK(p.vision.num_heads == 16);
  CHECK(p.vision.patch_size == 14);
  // 448 and 4096 against class defaults of 336 and 1536: a reader that
  // defaults instead of reading is wrong on both.
  CHECK(p.vision.image_size == 448);
  CHECK(p.vision.out_hidden_size == 4096);
  // The merger's context dim is `projection_intermediate_size` 10240, NOT the
  // GlmOcr parent's `out_hidden_size * in_channels` = 12288.
  CHECK(p.vision.projection_intermediate_size == 10240);
  CHECK(p.vision.projection_intermediate_size !=
        p.vision.out_hidden_size * p.vision.in_channels);
  CHECK(p.vision.attention_bias);

  // Image and video share one id in the emitted sequence; all six ids travel
  // together or a reader classifies every video frame as an image.
  CHECK(p.mm_tokens.image == 154854);
  CHECK(p.mm_tokens.video == 154855);
  CHECK(p.mm_tokens.image_start == 154830);
  CHECK(p.mm_tokens.image_end == 154831);
  CHECK(p.mm_tokens.video_start == 154832);
  CHECK(p.mm_tokens.video_end == 154833);

  // The published checkpoint is FP8 e4m3 block-quantized at [128, 128], and
  // `hyper_connection` is in `modules_to_not_convert`, so the mHC parameters
  // ship unquantised.
  CHECK(p.quant_method == "fp8");
  CHECK(p.quant_fmt == "e4m3");
  REQUIRE(p.weight_block_size.size() == 2);
  CHECK(p.weight_block_size[0] == 128);
  CHECK(p.weight_block_size[1] == 128);
  bool hc_unconverted = false;
  for (const std::string& m : p.modules_to_not_convert) {
    if (m == "hyper_connection") hc_unconverted = true;
  }
  CHECK(hc_unconverted);
}

TEST_CASE("glm5_next: the text stack is fully NoPE and upstream REQUIRES it") {
  const Glm5NextParams p = ParseGlm5NextParams(PublishedConfig());

  CHECK(p.mla.q_lora_rank == 1536);
  CHECK(p.mla.kv_lora_rank == 512);
  CHECK(p.mla.qk_nope_head_dim == 256);
  CHECK(p.mla.v_head_dim == 256);
  // ZERO, and the two derived fields follow upstream's forced overrides:
  // `head_dim = qk_rope_head_dim` and `qk_head_dim = rope + nope`.
  CHECK(p.mla.qk_rope_head_dim == 0);
  CHECK(p.mla.head_dim == 0);
  CHECK(p.mla.qk_head_dim == 256);

  // There is no rotary anywhere in the text stack. `text_config` carries no
  // `rope_theta` and no `rope_scaling`; the reference DELETES the inherited
  // `rope_parameters` field and passes `position_embeddings=None` to every
  // layer. `indexer_rope_interleave: true` is a VESTIGIAL flag the indexer
  // override ignores -- it is present in the fixture and must reach no field.
  const nlohmann::json text = PublishedConfigJson()["text_config"];
  CHECK(text.find("rope_theta") == text.end());
  CHECK(text.find("rope_scaling") == text.end());
  CHECK(text.contains("indexer_rope_interleave"));
  CHECK(text["indexer_rope_interleave"].get<bool>());

  // A positive rope dim is REFUSED, in upstream's own words. This is the
  // clause that makes the NoPE geometry mandatory rather than incidental.
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["qk_rope_head_dim"] = 64;
  CHECK_THROWS_WITH_AS(
      ParseGlm5NextParams(ConfigFrom(doc)),
      doctest::Contains("Expecting NoPE for the DSA attention layers"),
      std::runtime_error);
}

TEST_CASE("glm5_next: MlaBlockDims ACCEPTS this geometry, which is what W3 bought") {
  // O11, DISCHARGED by W3 (#2213). This case was the executable pin on the
  // blocker: `MlaBlockDims::Validate` required every dimension `> 0` while
  // upstream's `validate_architecture` requires `qk_rope_head_dim` to be ZERO,
  // so the two validators were exact complements over that one field and no
  // value satisfied both. W3 made 0 the ABSENT state of the rotary rather than
  // an invalid width, and the pin moved WITH the change rather than being
  // deleted by it.
  //
  // The refusal half now lives beside the relaxation, in
  // `tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp`
  // ("the NoPE geometry is REFUSED when it cannot describe a layer"), because
  // that is where the geometry's own gate is.
  const Glm5NextParams p = ParseGlm5NextParams(PublishedConfig());
  vllm::mla::MlaBlockDims dims;
  dims.hidden_size = p.hidden_size;
  dims.num_heads = p.num_attention_heads;
  dims.qk_nope_head_dim = p.mla.qk_nope_head_dim;
  dims.qk_rope_head_dim = p.mla.qk_rope_head_dim;  // 0 — the NoPE condition
  dims.v_head_dim = p.mla.v_head_dim;
  dims.kv_lora_rank = p.mla.kv_lora_rank;
  dims.q_lora_rank = p.mla.q_lora_rank;
  // `self.scaling = self.qk_head_dim ** (-0.5)` (modular_glm5_next.py:1028):
  // a plain scale, with no YaRN mscale correction, because there is no rotary.
  dims.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dims.qk_head_dim())));
  CHECK_NOTHROW(dims.Validate());
  // The cache row is the LATENT and nothing else: 512, not 576. This is the
  // consequence the KV arithmetic in the spec's Hardware section assumes.
  CHECK(dims.head_size() == p.mla.kv_lora_rank);
  CHECK(dims.head_size() == 512);
  CHECK(dims.qk_head_dim() == 256);
  // A positive rope dim is still refused UPSTREAM (see the config case above),
  // and our block still refuses an ODD one, so 0 is accepted because it is the
  // absent state and not because the check was deleted.
  vllm::mla::MlaBlockDims odd = dims;
  odd.qk_rope_head_dim = 1;
  CHECK_THROWS_AS(odd.Validate(), std::invalid_argument);
}

TEST_CASE("glm5_next: the KDA forget gate takes the SIGMOID branch") {
  const Glm5NextParams p = ParseGlm5NextParams(PublishedConfig());

  // The `linear_attn_config` sub-object's four keys, under their FOUR
  // DIFFERENT spellings, remapped by `__post_init__`.
  CHECK(p.kda.num_heads == 64);
  CHECK(p.kda.head_dim == 128);
  CHECK(p.kda.conv_kernel_dim == 4);

  // -5.0 is not None, which selects
  //   -bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
  // where our Kimi-Linear KDA (`kimi_kda.cpp`) implements
  //   -exp(A_log) * softplus(g + dt_bias).
  // Different functions of the same inputs, and the sign of `decay_rate`
  // differs. This is the single value the KDA port hinges on.
  REQUIRE(p.kda.lower_bound.has_value());
  CHECK(*p.kda.lower_bound == doctest::Approx(-5.0));
  CHECK(p.kda.takes_sigmoid_branch());

  // The softplus branch exists and is reachable ONLY through an explicit
  // `safe_gate: false`. Upstream defaults `safe_gate` to TRUE and a true
  // `safe_gate` over an absent bound installs -5.0, so a bare null does NOT
  // reach softplus -- getting that backwards would silently move every
  // published checkpoint onto the wrong branch.
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["linear_attn_config"]["gate_lower_bound"] = nullptr;
  const Glm5NextParams safe = ParseGlm5NextParams(ConfigFrom(doc));
  CHECK(safe.kda.takes_sigmoid_branch());
  CHECK(*safe.kda.lower_bound == doctest::Approx(-5.0));

  doc["text_config"]["linear_attn_config"]["safe_gate"] = false;
  const Glm5NextParams unsafe = ParseGlm5NextParams(ConfigFrom(doc));
  CHECK_FALSE(unsafe.kda.takes_sigmoid_branch());
  CHECK_FALSE(unsafe.kda.lower_bound.has_value());

  // A non-negative bound turns decay into growth and is refused rather than
  // parsed into a model that still generates text.
  doc = PublishedConfigJson();
  doc["text_config"]["linear_attn_config"]["gate_lower_bound"] = 5.0;
  CHECK_THROWS_WITH_AS(ParseGlm5NextParams(ConfigFrom(doc)),
                       doctest::Contains("turns decay into growth"),
                       std::runtime_error);
}

TEST_CASE("glm5_next: layer_types is the authority and full_attention is rewritten") {
  // Upstream rewrites every `full_attention` entry to
  // `deepseek_sparse_attention` in `__post_init__`, so a checkpoint that spells
  // it the old way must land on the SAME kinds. `Glm5NextLayerKind` has no
  // `kFullAttention` enumerator at all, which makes the pre-rewrite state
  // unrepresentable rather than merely unused.
  nlohmann::json doc = PublishedConfigJson();
  const Glm5NextParams published = ParseGlm5NextParams(ConfigFrom(doc));
  for (auto& e : doc["text_config"]["layer_types"]) {
    if (e.get<std::string>() == "deepseek_sparse_attention") e = "full_attention";
  }
  const Glm5NextParams rewritten = ParseGlm5NextParams(ConfigFrom(doc));
  CHECK(rewritten.layer_types == published.layer_types);

  // The `linear_attn_config` index lists are IGNORED by the reference; the
  // top-level list is the authority. This port checks them against it rather
  // than reading them, so a checkpoint whose two descriptions of one schedule
  // disagree is a loud failure instead of a first-wins.
  doc = PublishedConfigJson();
  doc["text_config"]["linear_attn_config"]["kda_layers"].push_back(3);
  CHECK_THROWS_WITH_AS(
      ParseGlm5NextParams(ConfigFrom(doc)),
      doctest::Contains("`layer_types` is the authority upstream reads"),
      std::runtime_error);

  // `first_k_dense_replace` is not a field of the runtime config class and
  // `__post_init__` never reads it -- the name does not occur in
  // `configuration_glm5_next.py` at all; the inherited attribute is deleted one
  // level up in `modular_glm5_next.py:169`. So the checkpoint's copy is an
  // inert kwarg. Changing it must move NOTHING.
  //
  // BOTH directions, because only the second one can fail. With an explicit
  // `mlp_layer_types` present the key is unreachable by construction, so a port
  // that reads it still passes -- a mutation that made the default read
  // `first_k_dense_replace` survived a first draft of this case that only
  // checked that direction. The load-bearing case is the one where
  // `mlp_layer_types` is ABSENT and the default schedule is synthesized, which
  // is the only place a port could consult the key at all.
  doc = PublishedConfigJson();
  CHECK(doc["text_config"].contains("first_k_dense_replace"));
  doc["text_config"]["first_k_dense_replace"] = 17;
  const Glm5NextParams inert = ParseGlm5NextParams(ConfigFrom(doc));
  CHECK(inert.mlp_layer_types == published.mlp_layer_types);

  doc["text_config"].erase("mlp_layer_types");
  const Glm5NextParams synthesized = ParseGlm5NextParams(ConfigFrom(doc));
  // Upstream's literal `min(3, num_hidden_layers)`, not the 17 the key asks
  // for: layer 3 is SPARSE, and layers 0-2 are the only dense ones.
  CHECK(synthesized.mlp_layer_types[2] == Glm5NextMlpKind::kDense);
  CHECK(synthesized.mlp_layer_types[3] == Glm5NextMlpKind::kSparse);
  CHECK(synthesized.mlp_layer_types[16] == Glm5NextMlpKind::kSparse);
  CHECK(synthesized.mlp_layer_types == published.mlp_layer_types);
}

TEST_CASE("glm5_next: kda_layers is ZERO-indexed, and the schedule ignores it") {
  // #2070. The shared reader synthesizes `layer_types` from
  // `linear_attn_config.kda_layers` under KIMI-LINEAR's ONE-INDEXED rule when
  // the key is absent. This model's list is ZERO-indexed -- it contains `0`,
  // which a one-indexed list of 45 layers cannot, and its maximum is 44 on 45
  // layers -- so reading through that rule shifts a third of the stack onto
  // the wrong attention kind, silently. `ParseGlm5NextParams` therefore takes
  // the schedule from this model's own `text_config` and never from the shared
  // reader's synthesized field.
  const nlohmann::json text = PublishedConfigJson()["text_config"];
  const nlohmann::json& lac = text["linear_attn_config"];
  int64_t lo = 1 << 30, hi = -1;
  for (const auto& e : lac["kda_layers"]) {
    lo = std::min<int64_t>(lo, e.get<int64_t>());
    hi = std::max<int64_t>(hi, e.get<int64_t>());
  }
  CHECK(lo == 0);   // a one-indexed list cannot contain 0
  CHECK(hi == 44);  // ...and a one-indexed list of 45 layers would reach 45
  CHECK(lac["kda_layers"].size() == 34u);

  // With `layer_types` erased the resolve must land on UPSTREAM's
  // `idx % 4 != 3` default, not on the one-indexed reading of that list. Read
  // one-indexed, layer 2 comes out `deepseek_sparse_attention`; upstream says
  // `linear_attention`, and so must we.
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"].erase("layer_types");
  const Glm5NextParams p = ParseGlm5NextParams(ConfigFrom(doc));
  CHECK(p.layer_types[2] == Glm5NextLayerKind::kLinearAttention);
  CHECK(p.layer_types == ParseGlm5NextParams(PublishedConfig()).layer_types);
}

TEST_CASE("glm5_next: the defaults are upstream's, not this port's guesses") {
  // With every optional schedule and geometry key removed, the resolve must
  // reproduce upstream's `__post_init__` defaults exactly. `index_kpool` is the
  // sharpest of these: the class default is 16 and the checkpoint says 4, so a
  // port that defaults where it should read is wrong by a factor of four.
  nlohmann::json doc = PublishedConfigJson();
  nlohmann::json& text = doc["text_config"];
  for (const char* key :
       {"layer_types", "mlp_layer_types", "indexer_types", "index_kpool"}) {
    text.erase(key);
  }
  const Glm5NextParams p = ParseGlm5NextParams(ConfigFrom(doc));

  // `idx % 4 != 3` is KDA.
  for (int64_t i = 0; i < 45; ++i) {
    CAPTURE(i);
    CHECK(p.layer_types[static_cast<size_t>(i)] ==
          ((i % 4 != 3) ? Glm5NextLayerKind::kLinearAttention
                        : Glm5NextLayerKind::kDeepseekSparseAttention));
  }
  // `["dense"] * min(3, L) + ["sparse"] * (L - 3)`.
  CHECK(p.mlp_layer_types[0] == Glm5NextMlpKind::kDense);
  CHECK(p.mlp_layer_types[2] == Glm5NextMlpKind::kDense);
  CHECK(p.mlp_layer_types[3] == Glm5NextMlpKind::kSparse);
  // The freq/offset schedule at freq 1 makes every layer `full`.
  for (const Glm5NextIndexerKind k : p.indexer_types) {
    CHECK(k == Glm5NextIndexerKind::kFull);
  }
  CHECK(p.indexer.kpool == 16);
  CHECK(p.indexer.kpool != 4);

  // And the published value is READ, not defaulted past.
  const Glm5NextParams real = ParseGlm5NextParams(PublishedConfig());
  CHECK(real.indexer.kpool == 4);
  CHECK(real.indexer.topk == 2048);
  CHECK(real.indexer.head_dim == 128);
  CHECK(real.indexer.n_heads == 32);
  CHECK(real.indexer.kpool_always_select_tail);
  // select_k = index_topk / index_kpool = 512.
  CHECK(real.indexer.max_select_pools() == 512);
}

TEST_CASE("glm5_next: index_topk_pattern is upstream's FIRST indexer fallback") {
  // `__post_init__` reads `index_topk_pattern` before it reaches the
  // freq/offset schedule (configuration_glm5_next.py, the
  // `if self.indexer_types is None` block), so a port that implements only the
  // schedule resolves a pattern config to a different stack. There is no token
  // gate anywhere on this fleet that would see that, which is why it is
  // asserted here rather than left to a downstream run.

  // The published checkpoint carries an explicit 45-entry `indexer_types`, and
  // upstream reads the pattern ONLY in the `is None` branch. A pattern beside
  // an explicit list must therefore move nothing. This is the direction that
  // cannot fail by accident, and it is asserted first so the ones below are
  // read as the load-bearing half.
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["index_topk_pattern"] = std::string(45, 'S');
  const Glm5NextParams ignored = ParseGlm5NextParams(ConfigFrom(doc));
  for (const Glm5NextIndexerKind k : ignored.indexer_types) {
    CHECK(k == Glm5NextIndexerKind::kFull);
  }

  // The STRING spelling. `F` runs the indexer, `S` reuses the previous full
  // layer's selection. The schedule below is full on the DSA layers at 3, 11,
  // 19, ... and shared on the rest, so it disagrees with the freq/offset
  // default on 39 of 45 layers -- a pattern of all-`F` would pass against a
  // port that ignored the key entirely.
  std::string pattern;
  for (int64_t i = 0; i < 45; ++i) pattern += (i % 8 == 3) ? 'F' : 'S';
  doc = PublishedConfigJson();
  doc["text_config"].erase("indexer_types");
  doc["text_config"]["index_topk_pattern"] = pattern;
  const Glm5NextParams from_string = ParseGlm5NextParams(ConfigFrom(doc));
  REQUIRE(from_string.indexer_types.size() == 45u);
  for (int64_t i = 0; i < 45; ++i) {
    CAPTURE(i);
    CHECK(from_string.indexer_types[static_cast<size_t>(i)] ==
          ((i % 8 == 3) ? Glm5NextIndexerKind::kFull
                        : Glm5NextIndexerKind::kShared));
  }

  // The SEQUENCE spelling: upstream's `list(pattern)` for a non-string, whose
  // entries are already the `full`/`shared` names. The two spellings must
  // resolve to the same schedule.
  nlohmann::json names = nlohmann::json::array();
  for (const char c : pattern) names.push_back(c == 'F' ? "full" : "shared");
  doc = PublishedConfigJson();
  doc["text_config"].erase("indexer_types");
  doc["text_config"]["index_topk_pattern"] = names;
  const Glm5NextParams from_list = ParseGlm5NextParams(ConfigFrom(doc));
  CHECK(from_list.indexer_types == from_string.indexer_types);

  // An unknown code is REFUSED, not defaulted: upstream's own
  // `{"F": "full", "S": "shared"}[c]` raises `KeyError` on one.
  doc = PublishedConfigJson();
  doc["text_config"].erase("indexer_types");
  doc["text_config"]["index_topk_pattern"] = std::string(45, 'X');
  CHECK_THROWS_WITH_AS(ParseGlm5NextParams(ConfigFrom(doc)),
                       doctest::Contains("`index_topk_pattern` is a string of"),
                       std::runtime_error);

  // And a pattern of the wrong length meets the same length check an explicit
  // `indexer_types` does, because upstream assigns the pattern INTO
  // `indexer_types` and everything after it is the same code path.
  doc = PublishedConfigJson();
  doc["text_config"].erase("indexer_types");
  doc["text_config"]["index_topk_pattern"] = std::string(44, 'F');
  CHECK_THROWS_WITH_AS(
      ParseGlm5NextParams(ConfigFrom(doc)),
      doctest::Contains("`indexer_types` has 44 entries"), std::runtime_error);

  // With the key ABSENT the freq/offset schedule is still what runs, so the
  // branch above is an addition rather than a replacement.
  doc = PublishedConfigJson();
  doc["text_config"].erase("indexer_types");
  const Glm5NextParams no_pattern = ParseGlm5NextParams(ConfigFrom(doc));
  for (const Glm5NextIndexerKind k : no_pattern.indexer_types) {
    CHECK(k == Glm5NextIndexerKind::kFull);
  }
}

TEST_CASE("glm5_next: every upstream validate_architecture rejection is implemented") {
  // All five, each against the clause it mirrors.
  struct Case {
    const char* key;
    nlohmann::json value;
    const char* expect;
  };
  const std::vector<Case> cases = {
      {"num_key_value_heads", 8,
       "must be the same as num_key_value_heads"},
      {"index_kpool", 0, "index_kpool must be positive"},
      {"index_kpool", 3, "must be divisible by index_kpool"},
      {"qk_rope_head_dim", 64, "Expecting NoPE for the DSA attention layers"},
  };
  for (const Case& c : cases) {
    CAPTURE(c.key);
    nlohmann::json doc = PublishedConfigJson();
    doc["text_config"][c.key] = c.value;
    CHECK_THROWS_WITH_AS(ParseGlm5NextParams(ConfigFrom(doc)),
                         doctest::Contains(c.expect), std::runtime_error);
  }
  // The fifth: `q_lora_rank is None`. Upstream's message is quoted verbatim,
  // exclamation mark included, because a reader who searches for it should
  // land on both implementations. An explicit NULL is what reaches it --
  // upstream's `is None` -- and it is a different config from an absent key.
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["q_lora_rank"] = nullptr;
  CHECK_THROWS_WITH_AS(
      ParseGlm5NextParams(ConfigFrom(doc)),
      doctest::Contains(
          "For DSA usage in the attention layers, the `q_lora_rank` is "
          "strictly required!"),
      std::runtime_error);

  // An ABSENT key takes the class default 1536 and is a config upstream
  // ACCEPTS, so refusing it would be a local guard LOOSER nowhere and stricter
  // in the one direction that matters -- a false refusal of a legal
  // checkpoint. The two cases are distinguished, not collapsed.
  doc = PublishedConfigJson();
  doc["text_config"].erase("q_lora_rank");
  const Glm5NextParams defaulted = ParseGlm5NextParams(ConfigFrom(doc));
  CHECK(defaulted.mla.q_lora_rank == 1536);

  // Every refusal names the model, the spec and the issue, so a reader lands on
  // the file that owes the work rather than on a bare message.
  doc = PublishedConfigJson();
  doc["text_config"]["index_kpool"] = 0;
  try {
    ParseGlm5NextParams(ConfigFrom(doc));
    FAIL("expected a refusal");
  } catch (const std::runtime_error& e) {
    const std::string m = e.what();
    CHECK(m.rfind("glm5_next: ", 0) == 0);
    CHECK(m.find(".agents/specs/glm5-next-flash.md") != std::string::npos);
    CHECK(m.find("#2067") != std::string::npos);
  }
}

TEST_CASE("glm5_next: the architecture RESOLVES through the production registry") {
  // The reachability case for the registration. `ModelRegistry::Resolve` is the
  // entry point `LoadedEngine` uses; nothing here constructs a factory by hand.
  const std::vector<std::string> archs = {"Glm5NextForConditionalGeneration"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(archs);
  CHECK(reg.architecture == "Glm5NextForConditionalGeneration");
  CHECK(reg.info.is_text_generation_model);
  CHECK(reg.info.supports_multimodal);
  // HYBRID: 34 of 45 layers are KDA carrying recurrent state.
  CHECK(reg.info.is_hybrid);
  CHECK_FALSE(reg.info.is_pooling_model);
  CHECK_FALSE(reg.factory->is_dense_model);

  // `parse_config` is the hook the loader calls, and it must be the SAME
  // validator: a malformed config is refused at load, not at first forward.
  REQUIRE(reg.factory->parse_config != nullptr);
  reg.factory->parse_config(PublishedConfig());
  nlohmann::json bad = PublishedConfigJson();
  bad["text_config"]["index_kpool"] = 0;
  CHECK_THROWS_WITH_AS(reg.factory->parse_config(ConfigFrom(bad)),
                       doctest::Contains("index_kpool must be positive"),
                       std::runtime_error);
}

TEST_CASE("glm5_next: the forward and the KV spec REFUSE BY NAME") {
  const std::vector<std::string> archs = {"Glm5NextForConditionalGeneration"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(archs);

  // The forward. WHAT THE REFUSAL BUYS is precision about a wrong MODEL, not a
  // wrong number: two of the primitives it names LOOK implemented in this tree
  // and are the wrong function for this model, so the message has to say which
  // and why.
  REQUIRE(reg.factory->forward != nullptr);
  ForeignLoadedModel foreign(reg);
  EmptyForwardInput in;
  const ModelForwardInput input = in.Get();
  std::string forward_msg;
  try {
    (void)reg.factory->forward(foreign, input);
    FAIL("expected a refusal");
  } catch (const std::exception& e) {
    forward_msg = e.what();
  }
  CHECK(forward_msg.find("Glm5NextForConditionalGeneration") !=
        std::string::npos);
  // Each missing primitive, named, with the wave that owes it.
  CHECK(forward_msg.find("W2") != std::string::npos);
  CHECK(forward_msg.find("W3") != std::string::npos);
  CHECK(forward_msg.find("W4") != std::string::npos);
  CHECK(forward_msg.find("W5") != std::string::npos);
  CHECK(forward_msg.find("W6") != std::string::npos);
  CHECK(forward_msg.find("SIGMOID branch") != std::string::npos);
  CHECK(forward_msg.find("kimi_kda.cpp") != std::string::npos);
  CHECK(forward_msg.find("UNWEIGHTED mHC head collapse") != std::string::npos);
  CHECK(forward_msg.find("HcHeadCollapse") != std::string::npos);
  CHECK(forward_msg.find("MlaBlockDims::Validate") != std::string::npos);
  CHECK(forward_msg.find("k-pool indexer") != std::string::npos);
  CHECK(forward_msg.find("glm5-next-flash.md") != std::string::npos);

  // The KV-cache spec refuses rather than returning an empty config, and names
  // all three cache shapes this model needs.
  std::string kv_msg;
  try {
    reg.factory->make_kv_cache(PublishedConfig(), 16, 8);
    FAIL("expected a refusal");
  } catch (const std::exception& e) {
    kv_msg = e.what();
  }
  CHECK(kv_msg.find("KV-cache spec is not ported") != std::string::npos);
  CHECK(kv_msg.find("NoPE MLA latent") != std::string::npos);
  CHECK(kv_msg.find("k-pool indexer side cache") != std::string::npos);
  CHECK(kv_msg.find("KDA recurrent") != std::string::npos);
}

TEST_CASE("glm5_next: the tensor inventory is generated from the topology") {
  const Glm5NextParams p = ParseGlm5NextParams(PublishedConfig());
  const std::vector<std::string> names = Glm5NextExpectedGgufTensors(p);

  // Counted from the maps and the schedule rather than transcribed: 3
  // model-level (`tie_word_embeddings` is false, so `output.weight` is its own
  // tensor), then per layer 8 common + (15 KDA | 14 DSA) + (3 dense | 5+3
  // sparse), then 11 vision + 24 * 14 vision-block.
  const int64_t expected =
      3 + (34 * (8 + 15)) + (11 * (8 + 14)) + (3 * 3) + (42 * (5 + 3)) +
      11 + (24 * 14);
  CHECK(static_cast<int64_t>(names.size()) == expected);

  auto has = [&names](const std::string& n) {
    return std::find(names.begin(), names.end(), n) != names.end();
  };
  // A KDA layer carries the three SEPARATE depthwise convs and no MLA tensor.
  CHECK(has("blk.0.ssm_conv1d_q.weight"));
  CHECK(has("blk.0.ssm_conv1d_k.weight"));
  CHECK(has("blk.0.ssm_conv1d_v.weight"));
  CHECK(has("blk.0.ssm_a"));
  CHECK_FALSE(has("blk.0.attn_kv_a_mqa.weight"));
  // A DSA layer carries the MLA tower, the indexer, its LayerNorm BIAS (which
  // is what settles LayerNorm over RMSNorm) and the k-pool compressor.
  CHECK(has("blk.3.attn_kv_a_mqa.weight"));
  CHECK(has("blk.3.indexer.k_norm.bias"));
  CHECK(has("blk.3.indexer_compressor_ape.weight"));
  CHECK(has("blk.3.indexer_compressor_gate.weight"));
  CHECK_FALSE(has("blk.3.ssm_a"));
  // mHC is FLAT on every layer, and there is NO `hc_head.*` anywhere -- which
  // is what independently settles the unweighted-mean head collapse.
  CHECK(has("blk.0.hc_attn_fn.weight"));
  CHECK(has("blk.0.hc_ffn_scale.weight"));
  for (const std::string& n : names) {
    CHECK(n.find("hc_head") == std::string::npos);
  }
  // Experts are STACKED, not per-expert, and only on sparse layers.
  CHECK(has("blk.3.ffn_gate_exps.weight"));
  CHECK(has("blk.3.exp_probs_b.bias"));
  CHECK_FALSE(has("blk.0.ffn_gate_exps.weight"));
  CHECK(has("blk.0.ffn_gate.weight"));
  // The MTP block is layer 45 and the reference discards it; nothing is
  // enumerated past `num_hidden_layers`.
  CHECK_FALSE(has("blk.45.attn_norm.weight"));

  // A hand-built `Glm5NextParams` whose schedules do not match its layer count
  // is refused rather than indexed past the end.
  Glm5NextParams bad = p;
  bad.num_hidden_layers = 46;
  CHECK_THROWS_WITH_AS(Glm5NextExpectedGgufTensors(bad),
                       doctest::Contains("needs both per-layer schedules"),
                       std::runtime_error);
}

// ---------------------------------------------------------------------------
// The GGUF half: one parser, two sources.

namespace {

// A `glm5next` GGUF carrying exactly the metadata
// `scripts/convert-glm5-next-gguf.py` writes for the PUBLISHED checkpoint, at a
// small `block_count` so the fixture stays a fixture. Every key spelling here
// is the converter's; if the two ever disagree, this is where it shows.
// `with_tokenizer` adds the four kvs `tok::Tokenizer::FromGguf` requires. It is
// OFF by default and on only for the cases that route the file through
// `LoadedEngine::FromModelDir`: that entry point reads the tokenizer out of the
// SAME container BEFORE it reaches the weight loader, so a file without one
// dies at the tokenizer and never reaches the refusal those cases are about.
// The config-layer cases below must not also be asserting a vocabulary, which
// is why this is a switch rather than an unconditional block.
std::string PublishedShapeGguf(int64_t n_layers,
                               const std::vector<std::string>& layer_types,
                               uint32_t head_count_kv = 64,
                               bool with_tokenizer = false) {
  gguf_test::GgufModelBuilder b;
  const std::string k = "glm5next.";
  b.AddKv(gguf_test::StrKv("general.architecture", "glm5next"));
  b.AddKv(gguf_test::U32Kv(k + "vocab_size", 154880));
  b.AddKv(gguf_test::U32Kv(k + "context_length", 1048576));
  b.AddKv(gguf_test::U32Kv(k + "embedding_length", 4096));
  b.AddKv(gguf_test::U32Kv(k + "block_count",
                           static_cast<uint32_t>(n_layers)));
  b.AddKv(gguf_test::U32Kv(k + "feed_forward_length", 12288));
  b.AddKv(gguf_test::U32Kv(k + "expert_feed_forward_length", 2048));
  b.AddKv(gguf_test::U32Kv(k + "expert_shared_feed_forward_length", 2048));
  b.AddKv(gguf_test::U32Kv(k + "expert_count", 288));
  b.AddKv(gguf_test::U32Kv(k + "expert_used_count", 8));
  b.AddKv(gguf_test::U32Kv(k + "expert_shared_count", 1));
  b.AddKv(gguf_test::U32Kv(k + "expert_group_count", 1));
  b.AddKv(gguf_test::U32Kv(k + "expert_group_used_count", 1));
  b.AddKv(gguf_test::F32Kv(k + "expert_weights_scale", 2.5f));
  b.AddKv(gguf_test::BoolKv(k + "expert_weights_norm", true));
  b.AddKv(gguf_test::U32Kv(k + "attention.head_count", 64));
  b.AddKv(gguf_test::U32Kv(k + "attention.head_count_kv", head_count_kv));
  b.AddKv(gguf_test::F32Kv(k + "attention.layer_norm_rms_epsilon", 1e-5f));
  b.AddKv(gguf_test::U32Kv(k + "attention.q_lora_rank", 1536));
  b.AddKv(gguf_test::U32Kv(k + "attention.kv_lora_rank", 512));
  b.AddKv(gguf_test::U32Kv(k + "attention.key_length_mla", 256));
  b.AddKv(gguf_test::U32Kv(k + "attention.value_length_mla", 256));
  b.AddKv(gguf_test::U32Kv(k + "attention.key_length", 256));
  b.AddKv(gguf_test::F32Kv(k + "swiglu_clamp_exp", 10.0f));
  b.AddKv(gguf_test::U32Kv(k + "rope.dimension_count", 0));
  b.AddKv(gguf_test::U32Kv(k + "attention.indexer.head_count", 32));
  b.AddKv(gguf_test::U32Kv(k + "attention.indexer.key_length", 128));
  b.AddKv(gguf_test::U32Kv(k + "attention.indexer.top_k", 2048));
  b.AddKv(gguf_test::U32Kv(k + "attention.indexer.kpool", 4));
  b.AddKv(gguf_test::BoolKv(k + "attention.indexer.kpool_always_select_tail",
                            true));
  b.AddKv(gguf_test::U32Kv(k + "kda.head_dim", 128));
  b.AddKv(gguf_test::F32Kv(k + "kda.gate_lower_bound", -5.0f));
  b.AddKv(gguf_test::U32Kv(k + "attention.linear_head_count", 64));
  b.AddKv(gguf_test::U32Kv(k + "ssm.conv_kernel", 4));
  b.AddKv(gguf_test::U32Kv(k + "hyper_connection.count", 4));
  b.AddKv(gguf_test::U32Kv(k + "hyper_connection.sinkhorn_iterations", 20));
  b.AddKv(gguf_test::F32Kv(k + "hyper_connection.epsilon", 1e-6f));
  b.AddKv(gguf_test::StrArrayKv(k + "layer_types", layer_types));
  std::vector<std::string> mlp;
  std::vector<std::string> indexer;
  for (int64_t i = 0; i < n_layers; ++i) {
    mlp.push_back(i < 3 ? "dense" : "sparse");
    indexer.emplace_back("full");
  }
  b.AddKv(gguf_test::StrArrayKv(k + "mlp_layer_types", mlp));
  b.AddKv(gguf_test::StrArrayKv(k + "attention.indexer.types", indexer));
  b.AddKv(gguf_test::U32Kv(k + "image_token_id", 154854));
  b.AddKv(gguf_test::U32Kv(k + "video_token_id", 154855));
  b.AddKv(gguf_test::U32Kv(k + "image_start_token_id", 154830));
  b.AddKv(gguf_test::U32Kv(k + "image_end_token_id", 154831));
  b.AddKv(gguf_test::U32Kv(k + "video_start_token_id", 154832));
  b.AddKv(gguf_test::U32Kv(k + "video_end_token_id", 154833));
  b.AddKv(gguf_test::U32Kv(k + "vision.block_count", 24));
  b.AddKv(gguf_test::U32Kv(k + "vision.embedding_length", 1024));
  b.AddKv(gguf_test::U32Kv(k + "vision.feed_forward_length", 4096));
  b.AddKv(gguf_test::U32Kv(k + "vision.head_count", 16));
  b.AddKv(gguf_test::U32Kv(k + "vision.patch_size", 14));
  b.AddKv(gguf_test::U32Kv(k + "vision.image_size", 448));
  b.AddKv(gguf_test::U32Kv(k + "vision.spatial_merge_size", 2));
  b.AddKv(gguf_test::U32Kv(k + "vision.temporal_patch_size", 2));
  b.AddKv(gguf_test::U32Kv(k + "vision.out_embedding_length", 4096));
  b.AddKv(gguf_test::U32Kv(k + "vision.projection_intermediate_size", 10240));
  b.AddKv(gguf_test::F32Kv(k + "vision.attention.layer_norm_rms_epsilon",
                           1e-5f));
  b.AddKv(gguf_test::F32Kv(k + "vision.swiglu_clamp", 10.0f));
  if (with_tokenizer) {
    // "gpt2" is llama.cpp's name for byte-level BPE, and "qwen35" is the only
    // pre name this tree maps without an approximation. A four-token vocabulary
    // with no merges is enough: nothing below tokenizes anything, and the load
    // refuses two steps later.
    b.AddKv(gguf_test::StrKv("tokenizer.ggml.model", "gpt2"));
    b.AddKv(gguf_test::StrKv("tokenizer.ggml.pre", "qwen35"));
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.tokens",
                                  {"a", "b", "c", "d"}));
    b.AddKv(gguf_test::I32ArrayKv("tokenizer.ggml.token_type", {1, 1, 1, 1}));
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.merges", {}));
  }
  return b.Build();
}

std::vector<std::string> PublishedLayerTypes(int64_t n) {
  std::vector<std::string> out;
  for (int64_t i = 0; i < n; ++i) {
    out.emplace_back(i % 4 == 3 ? "deepseek_sparse_attention"
                                : "linear_attention");
  }
  return out;
}

HfConfig ConfigFromGgufBytes(const std::string& bytes) {
  gguf_test::TempFile file(bytes);
  vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  return vllm::Glm5NextHfConfigFromGguf(gguf);
}

// The minimum metadata the inventory check is reached through: everything the
// builder reads before it, and `layer_types` itself. Kept separate from
// `PublishedShapeGguf` so a case about the INVENTORY is not also asserting a
// hundred metadata keys.
std::string SchedulePlusTensor(const std::vector<std::string>& types,
                               const std::string& tensor_name) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "glm5next"));
  b.AddKv(gguf_test::U32Kv("glm5next.block_count",
                           static_cast<uint32_t>(types.size())));
  b.AddKv(gguf_test::U32Kv("glm5next.embedding_length", 4096));
  b.AddKv(gguf_test::U32Kv("glm5next.context_length", 1048576));
  b.AddKv(gguf_test::U32Kv("glm5next.attention.head_count", 64));
  b.AddKv(gguf_test::F32Kv("glm5next.attention.layer_norm_rms_epsilon", 1e-5f));
  b.AddKv(gguf_test::U32Kv("glm5next.attention.key_length_mla", 256));
  b.AddKv(gguf_test::U32Kv("glm5next.attention.key_length", 256));
  b.AddKv(gguf_test::U32Kv("glm5next.rope.dimension_count", 0));
  b.AddKv(gguf_test::U32Kv("glm5next.vocab_size", 154880));
  b.AddKv(gguf_test::StrArrayKv("glm5next.layer_types", types));
  if (!tensor_name.empty()) {
    b.AddTensor(tensor_name, {4}, /*GGML_TYPE_F32=*/0, std::string(16, '\0'));
  }
  return b.Build();
}

std::string RefusalForGguf(const std::string& bytes) {
  try {
    (void)ConfigFromGgufBytes(bytes);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST_CASE("glm5_next: a GGUF and a config.json meet the SAME parser") {
  // The whole design claim of W1's GGUF half. If these two ever diverge, one of
  // the two sources is being validated by rules the other is not, on a model
  // that has no token gate anywhere on this fleet to notice.
  const HfConfig from_gguf =
      ConfigFromGgufBytes(PublishedShapeGguf(8, PublishedLayerTypes(8)));
  const Glm5NextParams g = ParseGlm5NextParams(from_gguf);
  nlohmann::json doc = PublishedConfigJson();
  doc["text_config"]["num_hidden_layers"] = 8;
  doc["text_config"]["layer_types"] = PublishedLayerTypes(8);
  doc["text_config"].erase("mlp_layer_types");
  doc["text_config"].erase("indexer_types");
  doc["text_config"]["linear_attn_config"].erase("kda_layers");
  doc["text_config"]["linear_attn_config"].erase("full_attn_layers");
  const Glm5NextParams j = ParseGlm5NextParams(ConfigFrom(doc));

  CHECK(from_gguf.model_type == "glm5_next");
  REQUIRE(from_gguf.architectures.size() == 1u);
  CHECK(from_gguf.architectures[0] == "Glm5NextForConditionalGeneration");
  CHECK(g.hidden_size == j.hidden_size);
  CHECK(g.vocab_size == j.vocab_size);
  CHECK(g.num_hidden_layers == j.num_hidden_layers);
  CHECK(g.layer_types == j.layer_types);
  CHECK(g.mlp_layer_types == j.mlp_layer_types);
  CHECK(g.indexer_types == j.indexer_types);
  CHECK(g.mla.qk_rope_head_dim == j.mla.qk_rope_head_dim);
  CHECK(g.mla.qk_head_dim == j.mla.qk_head_dim);
  CHECK(g.mla.q_lora_rank == j.mla.q_lora_rank);
  CHECK(g.mla.kv_lora_rank == j.mla.kv_lora_rank);
  CHECK(g.indexer.kpool == j.indexer.kpool);
  CHECK(g.indexer.topk == j.indexer.topk);
  CHECK(g.kda.num_heads == j.kda.num_heads);
  CHECK(g.kda.head_dim == j.kda.head_dim);
  CHECK(g.kda.conv_kernel_dim == j.kda.conv_kernel_dim);
  CHECK(g.kda.takes_sigmoid_branch() == j.kda.takes_sigmoid_branch());
  CHECK(*g.kda.lower_bound == *j.kda.lower_bound);
  CHECK(g.mhc.mult == j.mhc.mult);
  // The GGUF carries `hc_eps` as an F32 and the config.json as a decimal
  // double, so these agree to f32 precision and not bit-for-bit. The bound is
  // seven orders of magnitude below the value, which is a bound and not a
  // tolerance wide enough to hide a different constant (1e-5 is the one that
  // would).
  CHECK(std::fabs(g.mhc.eps - j.mhc.eps) < 1e-13);
  CHECK(g.moe.n_routed_experts == j.moe.n_routed_experts);
  CHECK(g.has_vision);
  CHECK(g.vision.image_size == j.vision.image_size);
  CHECK(g.vision.out_hidden_size == j.vision.out_hidden_size);
  CHECK(g.mm_tokens.image == j.mm_tokens.image);
  CHECK(g.mm_tokens.video_end == j.mm_tokens.video_end);

  // ...and the SAME validator. A GGUF whose metadata describes something
  // unrepresentable is refused by the message a config.json would get, from
  // `ParseGlm5NextParams`, and not by a private rule of the GGUF builder's.
  const std::string message = RefusalForGguf(
      PublishedShapeGguf(8, PublishedLayerTypes(8), /*head_count_kv=*/8));
  CHECK(message.rfind("glm5_next: ", 0) == 0);
  CHECK(message.find("must be the same as num_key_value_heads") !=
        std::string::npos);
}

TEST_CASE("glm5_next: an absent gate_lower_bound is NOT promoted to -5.0") {
  // The sharpest single value in the port. A file that does not state the bound
  // means the SOFTPLUS branch, and quietly defaulting it to -5.0 would move
  // every such file onto the other formula -- fluently, and undetectably. The
  // builder says so explicitly, by writing `safe_gate: false` alongside a null
  // bound rather than leaving a default that means the opposite.
  const HfConfig with =
      ConfigFromGgufBytes(PublishedShapeGguf(8, PublishedLayerTypes(8)));
  CHECK(ParseGlm5NextParams(with).kda.takes_sigmoid_branch());
  CHECK(with.raw["text_config"]["linear_attn_config"]["gate_lower_bound"]
            .get<double>() == -5.0);

  nlohmann::json doc = with.raw;
  doc["text_config"]["linear_attn_config"]["gate_lower_bound"] = nullptr;
  doc["text_config"]["linear_attn_config"]["safe_gate"] = false;
  CHECK_FALSE(ParseGlm5NextParams(ConfigFrom(doc)).kda.takes_sigmoid_branch());
}

TEST_CASE("glm5_next: a tensor inventory that CONTRADICTS layer_types is refused") {
  // Absence proves nothing on a sharded or partial file, so this is a
  // CONTRADICTION check and not a completeness check. A `blk.N` that carries
  // `ssm_a` while the metadata calls layer N `deepseek_sparse_attention` is a
  // file whose two descriptions of one layer disagree, and believing the
  // metadata loads a wrong model quietly.
  const std::vector<std::string> types = PublishedLayerTypes(8);
  // `SchedulePlusTensor` writes only the metadata the inventory check is
  // reached THROUGH, so every file below still refuses further on for a later
  // missing key. That is deliberate: the assertions are about which refusal
  // fires, not about the file being complete, and a case that asserted
  // "no refusal" would be asserting something this fixture cannot be.
  const char* kSsm = "carries `ssm_a`";
  const char* kMla = "carries `attn_kv_a_mqa.weight`";
  const std::string none = RefusalForGguf(SchedulePlusTensor(types, ""));
  CHECK(none.find(kSsm) == std::string::npos);
  CHECK(none.find(kMla) == std::string::npos);

  // Layer 3 is `deepseek_sparse_attention`; `ssm_a` exists on no such layer.
  const std::string m1 = RefusalForGguf(SchedulePlusTensor(types, "blk.3.ssm_a"));
  CHECK(m1.find(kSsm) != std::string::npos);
  CHECK(m1.find("blk.3") != std::string::npos);

  // The converse: a KDA layer carrying the MLA compression projection.
  const std::string m2 =
      RefusalForGguf(SchedulePlusTensor(types, "blk.0.attn_kv_a_mqa.weight"));
  CHECK(m2.find(kMla) != std::string::npos);
  CHECK(m2.find("blk.0") != std::string::npos);

  // And a tensor that AGREES with the schedule passes, so the check is a
  // discriminator and not a blanket refusal of any file with tensors in it.
  CHECK(RefusalForGguf(SchedulePlusTensor(types, "blk.0.ssm_a")).find(kSsm) ==
        std::string::npos);
  CHECK(RefusalForGguf(SchedulePlusTensor(types, "blk.3.attn_kv_a_mqa.weight"))
            .find(kMla) == std::string::npos);
}

TEST_CASE("glm5_next: the file states the rotary width twice and both must agree") {
  // `attention.key_length_mla - attention.key_length` and
  // `rope.dimension_count` are two independent statements of the same number,
  // written by the converter from two different config fields. On a NoPE model
  // a disagreement between them is exactly the defect a token gate could not
  // see, so it is a hard refusal rather than a first-wins.
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "glm5next"));
  b.AddKv(gguf_test::U32Kv("glm5next.block_count", 4));
  b.AddKv(gguf_test::U32Kv("glm5next.embedding_length", 4096));
  b.AddKv(gguf_test::U32Kv("glm5next.context_length", 1024));
  b.AddKv(gguf_test::U32Kv("glm5next.attention.head_count", 64));
  b.AddKv(gguf_test::F32Kv("glm5next.attention.layer_norm_rms_epsilon", 1e-5f));
  b.AddKv(gguf_test::U32Kv("glm5next.attention.key_length_mla", 320));
  b.AddKv(gguf_test::U32Kv("glm5next.attention.key_length", 256));
  b.AddKv(gguf_test::U32Kv("glm5next.rope.dimension_count", 0));
  b.AddKv(gguf_test::U32Kv("glm5next.vocab_size", 154880));
  CHECK(RefusalForGguf(b.Build())
            .find("states this model's rotary width twice") !=
        std::string::npos);
}

// ---------------------------------------------------------------------------
// The LOADER refusal, reached through the entry point a user reaches it by.
//
// WHY THIS IS NOT A CALL TO `factory->load_weights`. The refusal that matters
// here is a CAPABILITY -- "a user who hands this build a `glm5_next` checkpoint
// is told which wave owes the weight tower" -- and a case that invokes the hook
// directly measures the function instead. Both refusals below therefore enter
// through `LoadedEngine::FromModelDir`, the same call `vllm_c.cpp` and
// `server_main.cpp` make, and neither builds a `ModelSource` or a
// `ModelRegistration` by hand.
//
// The two arms are asserted SEPARATELY and each case checks it got the OTHER
// arm's message nowhere, because the two refusals are deliberately different
// texts for different next steps and a single "something threw" assertion would
// pass on either.

namespace {

std::string LoadRefusalFor(const std::string& model_path) {
  vllm::entrypoints::EngineParams params;
  // PINNED rather than left `kAuto`: the resolution `kAuto` takes depends on
  // what this build registered, and this case is about the loader, not about
  // which device the box has.
  params.device = vllm::Device::kCPU;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(model_path, params);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// A self-deleting safetensors MODEL DIRECTORY carrying the three files
// `FromModelDir`'s non-GGUF branch reads before it reaches the weight loader:
// the REAL published `config.json`, a tokenizer, and one shard (`LoadShards`
// refuses a directory with none). The shard is deliberately EMPTY: the loader
// refuses before any tensor is looked up, and a case that had to build 45
// layers of fake weights to prove that would be proving something else.
class TempSafetensorsDir {
 public:
  TempSafetensorsDir() {
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
           ("glm5_next_st_dir_" + std::to_string(vllm_test::ProcessId()) + "_" +
            std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    Write("config.json", PublishedConfigJson().dump(1));
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
  // A byte-level BPE over four single characters. Same shape as
  // `muse_glimmer_tiny_fixture.h`'s, which is what makes a synthetic directory
  // loadable by the production entry point rather than only by a weight loader.
  static std::string TinyTokenizerJson() {
    nlohmann::json vocab = nlohmann::json::object();
    vocab["\u2581"] = 0;
    vocab["a"] = 1;
    vocab["b"] = 2;
    vocab["c"] = 3;
    // `FromHfJson` REQUIRES a pre_tokenizer -- an absent one is
    // `tokenizer: missing pre_tokenizer`, measured, and it fires before the
    // loader. Metaspace is the shape `muse_glimmer_tiny_fixture.h` uses for the
    // same purpose.
    const nlohmann::json meta{{"type", "Metaspace"},
                              {"replacement", "\u2581"},
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

  // u64 LE header length + a header naming no tensor + no data section.
  static std::string EmptySafetensors() {
    const std::string header = R"({"__metadata__":{"format":"pt"}})";
    std::string out(8, '\0');
    const uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
      out[static_cast<size_t>(i)] =
          static_cast<char>((n >> (8 * i)) & 0xffU);
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

TEST_CASE("glm5_next: the GGUF LOADER refuses by name through FromModelDir") {
  const gguf_test::TempFile file(PublishedShapeGguf(
      8, PublishedLayerTypes(8), /*head_count_kv=*/64, /*with_tokenizer=*/true));
  const std::string msg = LoadRefusalFor(file.path());
  REQUIRE_FALSE(msg.empty());
  CAPTURE(msg);

  // It is THE LOADER's refusal and not an earlier failure. Naming the two steps
  // the load passed through is what makes that positive rather than inferred:
  // a config that did not parse dies inside `Glm5NextHfConfigFromGguf` with
  // "missing metadata key", and a file without the tokenizer kvs above dies
  // with `GGUF missing kv "tokenizer.ggml.model"`.
  CHECK(msg.find("missing metadata key") == std::string::npos);
  CHECK(msg.find("tokenizer") == std::string::npos);

  CHECK(msg.find("Glm5NextForConditionalGeneration") != std::string::npos);
  CHECK(msg.find("the weight loader is not ported") != std::string::npos);
  // The wave that owes the tower, and each primitive it owes.
  CHECK(msg.find("W5") != std::string::npos);
  CHECK(msg.find("KDA") != std::string::npos);
  CHECK(msg.find("NoPE MLA") != std::string::npos);
  CHECK(msg.find("mHC") != std::string::npos);
  CHECK(msg.find("stacked-expert") != std::string::npos);
  // And O7: no `.gguf` of this model exists anywhere, so the reader's next step
  // is the converter and not a download.
  CHECK(msg.find("O7") != std::string::npos);
  CHECK(msg.find("scripts/convert-glm5-next-gguf.py") != std::string::npos);
  CHECK(msg.find(".agents/specs/glm5-next-flash.md") != std::string::npos);
  CHECK(msg.find("#1998") != std::string::npos);
  // This arm and NOT the safetensors one: the two are different next steps.
  CHECK(msg.find("the GGUF config is read and validated") != std::string::npos);
  CHECK(msg.find("598.53 GiB") == std::string::npos);
}

TEST_CASE("glm5_next: the safetensors LOADER refuses by name through FromModelDir") {
  const TempSafetensorsDir dir;
  const std::string msg = LoadRefusalFor(dir.path());
  REQUIRE_FALSE(msg.empty());
  CAPTURE(msg);

  // Past the config, past the tokenizer and past the shard scan: each of those
  // has its own message and none of them is this one.
  CHECK(msg.find("no *.safetensors shards found") == std::string::npos);
  CHECK(msg.find("tokenizer") == std::string::npos);

  CHECK(msg.find("Glm5NextForConditionalGeneration") != std::string::npos);
  CHECK(msg.find("the weight loader is not ported yet") != std::string::npos);
  CHECK(msg.find("W5 owes it") != std::string::npos);
  // The published arms and the device they do not fit, so the reader is not
  // sent looking for a checkpoint that would work.
  CHECK(msg.find("305.78 GiB") != std::string::npos);
  CHECK(msg.find("598.53 GiB") != std::string::npos);
  CHECK(msg.find(".agents/specs/glm5-next-flash.md") != std::string::npos);
  CHECK(msg.find("#1998") != std::string::npos);
  // This arm and NOT the GGUF one.
  CHECK(msg.find("the GGUF config is read and validated") == std::string::npos);
}
