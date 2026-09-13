// DeepSeek-V4.1-Flash config descent + validation (W1). See
// `include/vllm/model_executor/models/deepseek_v4_1.h` for the scope statement,
// the upstream anchors and the two upstream facts this wave records rather than
// smooths over.
#include "vllm/model_executor/models/deepseek_v4_1.h"

#include <nlohmann/json.hpp>

#include <algorithm>  // std::min, for the source-layer floor below
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

// The nested-object accessors. V4 read its DeepSeek keys straight off
// `config.raw`; V4.1's live one level down, so every read here is explicit
// about WHICH sub-object it came from. That is the whole nested->flat lift, and
// keeping it visible is the point: a field silently read from the wrong level
// would default rather than fail, and a default is a silent wrong answer.
const nlohmann::json* Sub(const nlohmann::json& doc, const char* key) {
  if (!doc.is_object()) return nullptr;
  const auto it = doc.find(key);
  if (it == doc.end() || !it->is_object()) return nullptr;
  return &*it;
}

const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  if (!doc.is_object()) return nullptr;
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &*it;
}

int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number_integer()) ? f->get<int64_t>() : fallback;
}

double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<double>() : fallback;
}

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_boolean()) ? f->get<bool>() : fallback;
}

std::string RawString(const nlohmann::json& doc, const char* key,
                      const std::string& fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_string()) ? f->get<std::string>() : fallback;
}

std::vector<int64_t> RawIntArray(const nlohmann::json& doc, const char* key) {
  std::vector<int64_t> out;
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr || !f->is_array()) return out;
  for (const auto& v : *f) out.push_back(v.is_number() ? v.get<int64_t>() : 0);
  return out;
}

std::string JoinInts(const std::vector<int64_t>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) s += ",";
    s += std::to_string(v[i]);
  }
  return s;
}

}  // namespace

DeepseekV41Params ParseDeepseekV41Params(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  DeepseekV41Params p;

  // `text_config` is REQUIRED and its absence is refused by name rather than
  // defaulted through. Upstream tolerates a missing `text_config` because its
  // shim is also the base-config constructor for synthesized configs
  // (`text_config or {}`); here the only producer is a real config.json, and a
  // V4.1 config.json without a `text_config` is malformed. Defaulting would
  // produce a params struct full of zeros that then fails the geometry checks
  // below with a message about `hidden_size`, sending the reader to the wrong
  // field entirely.
  const nlohmann::json* text = Sub(raw, "text_config");
  VT_CHECK(text != nullptr,
           "deepseek-v4.1: config.json carries no `text_config` object. V4.1 "
           "nests every text-model field under `text_config` (V4 kept them at "
           "the top level); a config without it is not a DeepSeek-V4.1 config.");
  const nlohmann::json& t = *text;

  // The vision tower and the quantization block are OPTIONAL: upstream reads
  // both through `.get(...)` with fallbacks, and a text-only V4.1 variant is a
  // config with no `vision_config` (whereupon `vision_n_layers` is 0 and every
  // mm-prefix flag is false). Both are therefore read off an empty object when
  // absent rather than refused.
  static const nlohmann::json kEmpty = nlohmann::json::object();
  const nlohmann::json* vision = Sub(raw, "vision_config");
  const nlohmann::json& v = vision != nullptr ? *vision : kEmpty;
  // `quantization_config` takes the SAME two-tier precedence upstream's shim
  // gives it, and for the same mechanical reason as `expert_dtype` below:
  // `deepseek_v41.py:34-40` `setattr`s every `text_config` key onto `self`, so
  // a `text_config.quantization_config` becomes `self.quantization_config`;
  // `super().__init__(**kwargs)` at `:41` then OVERWRITES it from a top-level
  // key if one is present. Top level wins, `text_config` is the fallback. The
  // published file carries it at the top level only, so this tier is unreached
  // there and exists so the two configs cannot disagree.
  const nlohmann::json* quant = Sub(raw, "quantization_config");
  if (quant == nullptr) quant = Sub(t, "quantization_config");
  const nlohmann::json& q = quant != nullptr ? *quant : kEmpty;

  // --- shared geometry ---
  // `ParseHfConfig` already resolved these off `text_config` for us, mirroring
  // upstream `PretrainedConfig.get_text_config()`; the raw read is the fallback
  // for a config that reached us another way.
  p.hidden_size =
      config.hidden_size > 0 ? config.hidden_size : RawInt(t, "hidden_size", 0);
  p.num_hidden_layers = config.num_hidden_layers > 0
                            ? config.num_hidden_layers
                            : RawInt(t, "num_hidden_layers", 0);
  p.vocab_size =
      config.vocab_size > 0 ? config.vocab_size : RawInt(t, "vocab_size", 0);
  p.num_attention_heads = config.num_attention_heads > 0
                              ? config.num_attention_heads
                              : RawInt(t, "num_attention_heads", 0);
  p.num_key_value_heads = RawInt(t, "num_key_value_heads", 1);
  p.rms_norm_eps = RawDouble(t, "rms_norm_eps", 1e-6);
  p.tie_word_embeddings = RawBool(t, "tie_word_embeddings", false);
  p.max_position_embeddings = RawInt(t, "max_position_embeddings", 0);
  p.num_nextn_predict_layers = RawInt(t, "num_nextn_predict_layers", 0);

  // --- 512-wide MLA ---
  p.head_dim = config.head_dim > 0 ? config.head_dim : RawInt(t, "head_dim", 0);
  p.qk_rope_head_dim = RawInt(t, "qk_rope_head_dim", 64);
  p.q_lora_rank = RawInt(t, "q_lora_rank", 0);
  p.o_lora_rank = RawInt(t, "o_lora_rank", 0);
  p.o_groups = RawInt(t, "o_groups", 0);
  p.sliding_window = RawInt(t, "sliding_window", 0);
  p.rope_theta = RawDouble(t, "rope_theta", 10000.0);
  p.compress_rope_theta = RawDouble(t, "compress_rope_theta", 160000.0);
  if (const nlohmann::json* rope = Sub(t, "rope_scaling"); rope != nullptr) {
    p.rope_type = RawString(*rope, "rope_type", "yarn");
    p.rope_scale_factor = RawDouble(*rope, "factor", 1.0);
    p.rope_orig_ctx = RawInt(*rope, "original_max_position_embeddings", 0);
    p.rope_beta_fast = RawDouble(*rope, "beta_fast", 32.0);
    p.rope_beta_slow = RawDouble(*rope, "beta_slow", 1.0);
  }

  // --- MoE ---
  p.n_routed_experts = RawInt(t, "n_routed_experts", 0);
  p.num_experts_per_tok = RawInt(t, "num_experts_per_tok", 0);
  p.moe_intermediate_size = RawInt(t, "moe_intermediate_size", 0);
  p.n_shared_experts = RawInt(t, "n_shared_experts", 0);
  p.norm_topk_prob = RawBool(t, "norm_topk_prob", true);
  p.routed_scaling_factor = RawDouble(t, "routed_scaling_factor", 1.0);
  p.swiglu_limit = RawDouble(t, "swiglu_limit", 0.0);
  p.scoring_func = RawString(t, "scoring_func", "sqrtsoftplus");
  p.topk_method = RawString(t, "topk_method", "noaux_tc");

  // --- MHC ---
  p.hc_mult = RawInt(t, "hc_mult", 0);
  p.hc_sinkhorn_iters = RawInt(t, "hc_sinkhorn_iters", 0);
  p.hc_eps = RawDouble(t, "hc_eps", 1e-6);

  // --- DSA ---
  p.index_head_dim = RawInt(t, "index_head_dim", 0);
  p.index_n_heads = RawInt(t, "index_n_heads", 0);
  p.index_topk = RawInt(t, "index_topk", 0);
  p.compress_ratios = RawIntArray(t, "compress_ratios");
  p.kv_source_layer_ids = RawIntArray(t, "kv_source_layer_ids");
  p.index_source_layer_ids = RawIntArray(t, "index_source_layer_ids");
  p.candidate_source_layer_id = RawInt(t, "candidate_source_layer_id", -1);
  p.candidate_topk_blocks = RawInt(t, "candidate_topk_blocks", 0);
  p.candidate_block_size = RawInt(t, "candidate_block_size", 0);

  // --- Engram ---
  p.engram_layer_ids = RawIntArray(t, "engram_layer_ids");
  p.engram_num_embeddings = RawIntArray(t, "engram_num_embeddings");
  p.engram_max_ngram_size = RawInt(t, "engram_max_ngram_size", 0);
  p.engram_vocab_size = RawInt(t, "engram_vocab_size", 0);
  p.engram_n_heads = RawInt(t, "engram_n_heads", 0);
  p.engram_head_dim = RawInt(t, "engram_head_dim", 0);
  p.engram_pad_token_id = RawInt(t, "engram_pad_token_id", 0);
  p.engram_compressed_vocab_size = RawInt(t, "engram_compressed_vocab_size", 0);

  // --- DSpark ---
  p.dspark_block_size = RawInt(t, "dspark_block_size", 0);
  p.dspark_noise_token_id = RawInt(t, "dspark_noise_token_id", 0);
  p.dspark_target_layer_ids = RawIntArray(t, "dspark_target_layer_ids");
  p.dspark_markov_rank = RawInt(t, "dspark_markov_rank", 0);
  p.dspark_n_routed_experts = RawInt(t, "dspark_n_routed_experts", 0);
  p.dspark_num_experts_per_tok = RawInt(t, "dspark_num_experts_per_tok", 0);

  // --- the vision_* lift (deepseek_v41.py:52-62), upstream's own fallbacks ---
  p.vision_n_layers = RawInt(v, "num_hidden_layers", 0);
  p.vision_dim = RawInt(v, "hidden_size", 1024);
  p.vision_n_heads = RawInt(v, "num_attention_heads", 16);
  p.vision_inter_dim = RawInt(v, "intermediate_size", 2816);
  p.vision_patch_size = RawInt(v, "patch_size", 14);
  p.vision_rope_theta = RawDouble(v, "rope_theta", 10000.0);
  p.vision_downsample_ratio = RawInt(v, "downsample_ratio", 3);
  p.vision_max_n_token = RawInt(v, "max_image_tokens", 1024);
  p.vision_min_pixels = RawInt(v, "min_pixels", 295936);
  // `Field` treats an explicit JSON `null` as absent, which is exactly
  // upstream's `.get("max_wh_ratio")` -> None for the published file.
  if (const nlohmann::json* wh = Field(v, "max_wh_ratio");
      wh != nullptr && wh->is_number()) {
    p.vision_max_wh_ratio = wh->get<double>();
  }

  // --- the three derived mm-prefix flags (deepseek_v41.py:66-76) ---
  const bool has_tower = p.vision_n_layers > 0;
  p.is_mm_prefix_lm = has_tower;
  p.mm_prefix_clamp_sliding_window = has_tower;
  p.mm_prefix_span_leading_pad_modulus = has_tower ? 2 : 0;

  // --- quantization_config, plus the expert_dtype lift ---
  p.quant_method = RawString(q, "quant_method", "");
  p.quant_activation_scheme = RawString(q, "activation_scheme", "");
  p.quant_scale_fmt = RawString(q, "scale_fmt", "");
  p.weight_block_size = RawIntArray(q, "weight_block_size");
  // deepseek_v41.py:34-50, and the precedence has THREE tiers, not two. The
  // shim `setattr`s every `text_config` key onto `self` at `:34-40`, then
  // `super().__init__(**kwargs)` at `:41` sets any top-level key over it, and
  // only THEN does `:48-50` consult `quantization_config` -- behind
  // `if not hasattr(self, "expert_dtype")`, so it fires only when neither of
  // the first two supplied one. Top level, then `text_config`, then
  // `quantization_config`. An earlier revision of this line read the top level
  // and `quantization_config` only, which would have answered `fp4` where
  // upstream answers `fp8` for a config carrying `text_config.expert_dtype`.
  // The published file supplies it under `quantization_config` alone, so only
  // the last tier is reached there; the middle one is gated by a case.
  p.expert_dtype =
      RawString(raw, "expert_dtype",
                RawString(t, "expert_dtype", RawString(q, "expert_dtype", "")));

  // ─── validation ──────────────────────────────────────────────────────────
  // Upstream's config class raises NOTHING (it is a flattening shim), so every
  // check below is either a mirror of a raise in the MODEL code — cited — or a
  // limit of this bring-up, marked as ours.
  VT_CHECK(p.hidden_size > 0,
           "deepseek-v4.1: text_config.hidden_size must be positive");
  VT_CHECK(p.num_hidden_layers > 0,
           "deepseek-v4.1: text_config.num_hidden_layers must be positive");
  VT_CHECK(p.n_routed_experts > 0,
           "deepseek-v4.1: text_config.n_routed_experts must be positive (this "
           "is a MoE architecture)");
  VT_CHECK(p.hc_mult > 0,
           "deepseek-v4.1: text_config.hc_mult must be positive — Manifold "
           "Hyper-Connections are structural to this architecture");

  // compressor.py:210 — `assert self.head_dim == 512 and self.rope_head_dim ==
  // 64`. Upstream asserts the pair together and so do we, because the NoPE
  // width is their difference and a half-satisfied pair is not representable.
  // STRICTER THAN UPSTREAM ON ONE CONFIG CLASS, deliberately: upstream's assert
  // lives in `Compressor.__init__`, so it fires only once a compressor is
  // CONSTRUCTED, and a V4.1 config with every `compress_ratio` 0 would never
  // construct one and would never reach it. This check is unconditional, so
  // that config is refused here and accepted there. The refusal is still the
  // right answer for THIS tree — every scoped W3c/W3d op assumes 448+64, so an
  // all-sliding-window V4.1 at another width has no path past W1 either — but
  // it is ours and not a mirror, and the message says "is scoped" rather than
  // citing the assert as its authority.
  VT_CHECK(p.head_dim == 512 && p.qk_rope_head_dim == 64,
           "deepseek-v4.1: only the 512-wide MLA geometry (448 NoPE + 64 RoPE) "
           "is scoped; got head_dim=" +
               std::to_string(p.head_dim) + " qk_rope_head_dim=" +
               std::to_string(p.qk_rope_head_dim) +
               " (upstream compressor.py:210 asserts the same pair)");

  // attention.py:257-262, mirrored including the set {0,1,2} and the layer id
  // in the message. This is the check that a V4 parser would fail on this
  // config and vice versa: V4's ratios are {0,4,128}.
  for (size_t i = 0; i < p.compress_ratios.size(); ++i) {
    const int64_t r = p.compress_ratios[i];
    VT_CHECK(r == 0 || r == 1 || r == 2,
             "deepseek-v4.1: layer " + std::to_string(i) +
                 " has compress_ratio=" + std::to_string(r) +
                 "; only 0 (sliding window), 1 and 2 are supported. The "
                 "ratio-4/128 kernels are DeepSeek-V4.0-specific and are not "
                 "this architecture (upstream attention.py:257-262)");
  }

  // attention.py:277-283. Upstream raises per compressed LAYER; the condition
  // it tests is config-global, so we test it once and name the first layer that
  // triggers it, which is the same information without 40 identical throws.
  bool any_compressed = false;
  int64_t first_compressed = -1;
  for (size_t i = 0; i < p.compress_ratios.size(); ++i) {
    if (p.compress_ratios[i] > 0) {
      any_compressed = true;
      first_compressed = static_cast<int64_t>(i);
      break;
    }
  }
  if (any_compressed) {
    VT_CHECK(!p.kv_source_layer_ids.empty() &&
                 !p.index_source_layer_ids.empty(),
             "deepseek-v4.1: requires kv_source_layer_ids / "
             "index_source_layer_ids in text_config for compressed layers "
             "(layer " +
                 std::to_string(first_compressed) + " has compress_ratio=" +
                 std::to_string(p.compress_ratio(first_compressed)) +
                 "); got kv_source_layer_ids=[" +
                 JoinInts(p.kv_source_layer_ids) +
                 "] index_source_layer_ids=[" +
                 JoinInts(p.index_source_layer_ids) +
                 "] (upstream attention.py:277-283)");

    // attention.py:284-289, the SECOND way that block raises and the one a
    // non-empty check does not cover:
    //
    //     self.kv_source_layer_id = max(s for s in self.kv_source_layers
    //                                   if s <= layer_id)
    //
    // `max()` over an empty sequence raises `ValueError: max() arg is an empty
    // sequence` when a compressed layer sits BELOW every source layer. Upstream
    // raises that bare Python error rather than a named one, so what is
    // mirrored is WHEN it raises, not its text; the message here names the
    // field, the layer and the list, which is what a bare `ValueError` does not.
    //
    // Testing the FIRST compressed layer is necessary and sufficient. The
    // predicate fails for layer L exactly when `min(sources) > L`, and the
    // first compressed layer is the smallest such L, so if it passes, every
    // later compressed layer passes too.
    //
    // This is not a hypothetical class. The published config is ONE entry from
    // the boundary: its first compressed layer is 2 and its first kv source is
    // also 2, so moving either by one in the wrong direction reaches it.
    int64_t min_kv = p.kv_source_layer_ids[0];
    for (const int64_t s : p.kv_source_layer_ids) min_kv = std::min(min_kv, s);
    int64_t min_index = p.index_source_layer_ids[0];
    for (const int64_t s : p.index_source_layer_ids) {
      min_index = std::min(min_index, s);
    }
    VT_CHECK(min_kv <= first_compressed,
             "deepseek-v4.1: compressed layer " +
                 std::to_string(first_compressed) +
                 " has no kv_source_layer_ids entry at or below it; the lowest "
                 "is " +
                 std::to_string(min_kv) + " in kv_source_layer_ids=[" +
                 JoinInts(p.kv_source_layer_ids) +
                 "]. A compressed layer consumes the most recently published "
                 "source BELOW it, so there must be one (upstream "
                 "attention.py:284-286 takes `max()` over an empty sequence "
                 "here)");
    VT_CHECK(min_index <= first_compressed,
             "deepseek-v4.1: compressed layer " +
                 std::to_string(first_compressed) +
                 " has no index_source_layer_ids entry at or below it; the "
                 "lowest is " +
                 std::to_string(min_index) + " in index_source_layer_ids=[" +
                 JoinInts(p.index_source_layer_ids) +
                 "]. A compressed layer consumes the most recently published "
                 "source BELOW it, so there must be one (upstream "
                 "attention.py:287-289 takes `max()` over an empty sequence "
                 "here)");
  }

  // OURS, not upstream's: the routing this bring-up scopes. V4 carries the same
  // pair of checks for the same reason — a different scorer or expert dtype is
  // a different MoE and would be a silent wrong answer rather than a refusal.
  VT_CHECK(p.scoring_func == "sqrtsoftplus",
           "deepseek-v4.1: only scoring_func='sqrtsoftplus' is scoped; got '" +
               p.scoring_func + "'");
  VT_CHECK(p.expert_dtype == "fp4" || p.expert_dtype == "fp8",
           "deepseek-v4.1: expert_dtype must be 'fp4' or 'fp8'; got '" +
               p.expert_dtype +
               "' (it is lifted from quantization_config.expert_dtype when the "
               "top level carries none)");
  return p;
}

void ParseDeepseekV41Config(const HfConfig& config) {
  // The resolve itself IS the validation: it throws on every field this wave
  // cannot represent.
  (void)ParseDeepseekV41Params(config);
}

bool IsDeepseekV41Gguf(const GgufFile& gguf) {
  const GgufValue* arch = gguf.FindKv("general.architecture");
  if (arch == nullptr || arch->TypeId() != kGgufString) return false;
  return std::get<std::string>(arch->v) == kDeepseekV41GgufArch;
}

std::string DeepseekV41GgufRefusal() {
  return
      "Model architecture DeepseekV41ForCausalLM does not support GGUF weights "
      "yet: the GGUF k-quant arm is OWED to W8 -- both a loader arm beside the "
      "existing `deepseek4` one and, since no tool anywhere writes such a "
      "container today, a converter. This build RESOLVES the architecture and "
      "VALIDATES its config.json (W1); it cannot read weights in any format "
      "yet, GGUF or safetensors. Row "
      "MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm, spec "
      ".agents/specs/deepseek-v4-1-flash.md section `## Work breakdown`, issue "
      "ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP. The architecture string "
      "`deepseek41` this refusal recognizes is DERIVED from this project's own "
      "`deepseek_v4` -> `deepseek4` mapping and is provisional until W8 settles "
      "it; no released llama.cpp defines this architecture.";
}

}  // namespace vllm
