// DeepSeek-V2 config resolution + BF16 safetensors weight loader — MLA campaign
// W7. Header (include/vllm/model_executor/models/deepseek_v2.h) carries the full
// `file:line`-on-both-sides port map; this TU implements the config parse, the
// checkpoint name map, and the LOAD-TIME `kv_b_proj -> W_UK/W_UV` absorption.
//
// Name map (DeepSeek-V2-Lite config.json, plain `model.layers.` prefix — the
// checkpoint has NO multimodal wrapper), verified against the shipped
// model.safetensors.index.json (5291 tensors, 4 shards):
//   model.embed_tokens.weight                            -> embed_tokens [V,H]
//   model.norm.weight                                    -> final_norm [H]
//   lm_head.weight                                       -> lm_head [H,V] (UNTIED)
//   model.layers.N.input_layernorm.weight                -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight       -> post_attention_ln [H]
//   -- MLA attention (deepseek_v2.py:1003-1049) --
//   model.layers.N.self_attn.q_proj.weight               -> q_proj (raw-NK)
//        [q_lora_rank IS NULL branch, :1028-1034 — the V2-Lite branch]
//   model.layers.N.self_attn.q_a_proj.weight    ]        -> fused_qkv_a_proj
//   model.layers.N.self_attn.kv_a_proj_with_mqa.weight ] (raw-NK, MERGED)
//        [q_lora_rank NOT null branch — packed_modules_mapping :1812-1820]
//   model.layers.N.self_attn.q_a_layernorm.weight        -> q_a_layernorm [ql]
//   model.layers.N.self_attn.q_b_proj.weight             -> q_b_proj (raw-NK)
//   model.layers.N.self_attn.kv_a_proj_with_mqa.weight   -> kv_a_proj_with_mqa
//        [q_lora_rank IS NULL branch, :1010-1016]
//   model.layers.N.self_attn.kv_a_layernorm.weight       -> kv_a_layernorm [L]
//   model.layers.N.self_attn.kv_b_proj.weight            -> kv_b_proj (raw-NK)
//                                                        -> ALSO w_uk_t / w_uv
//   model.layers.N.self_attn.o_proj.weight               -> o_proj (raw-NK)
//   -- dense layers, N < first_k_dense_replace (deepseek_v2.py:1251-1258) --
//   model.layers.N.mlp.{gate,up}_proj.weight             -> dense.gate_up (merged)
//   model.layers.N.mlp.down_proj.weight                  -> dense.down (raw-NK)
//   -- MoE layers (deepseek_v2.py:276-393) --
//   model.layers.N.mlp.gate.weight                       -> moe.router_gate [H,E]
//   model.layers.N.mlp.gate.e_score_correction_bias      -> (noaux_tc ONLY)
//   model.layers.N.mlp.experts.E.{gate,up}_proj.weight   -> expert_gate/up [H,I]
//   model.layers.N.mlp.experts.E.down_proj.weight        -> expert_down [I,H]
//   model.layers.N.mlp.shared_experts.{gate,up}_proj.weight -> moe.shared.gate_up
//   model.layers.N.mlp.shared_experts.down_proj.weight   -> moe.shared.down
//
// ─── THE ABSORPTION IS DONE HERE, AT LOAD TIME ──────────────────────────────
// Upstream performs it in `MLAAttention.process_weights_after_loading`
// (mla_attention.py:875-962), i.e. after the checkpoint is read and before the
// first forward. We do the identical transform (mla::AbsorbKvBProjBf16, W6) at
// the same lifecycle point — while the `kv_b_proj` bytes are already in hand —
// so the forward never sees an unabsorbed decode path. Both forms are retained:
// `kv_b_proj` itself feeds the PREFILL (materialized-MHA) path and the
// chunked-context up-projection (mla_attention.py:2371-2375, :2141-2170), and
// `w_uk_t`/`w_uv` feed the ABSORBED decode (`:789`, `:1034`). Upstream keeps
// these bf16 rather than quantizing them and says why at `:876-878`.
#include "vllm/model_executor/models/deepseek_v2.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::MakeOwned;

// --- raw config.json readers (DeepSeek keys are not all typed on HfConfig) ---
const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &(*it);
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<int64_t>() : fallback;
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
// `q_lora_rank: null` is MEANINGFUL (deepseek_v2.py:1003 branches on `is not
// None`), so an absent/null value must map to 0 (our "no q_lora" encoding) and
// NOT to a default rank.
int64_t RawOptionalRank(const nlohmann::json& doc, const char* key) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<int64_t>() : 0;
}

// The `rope_scaling` / `rope_parameters` sub-dict. HfConfig types most of it
// (RopeParameters), but `mscale` / `mscale_all_dim` are DeepSeek-only YaRN
// controls (deepseek_scaling_rope.py:31-74) that no other family uses, so they
// are read from the raw doc here rather than widened onto the shared struct.
const nlohmann::json* RopeDict(const nlohmann::json& doc) {
  const nlohmann::json* f = Field(doc, "rope_parameters");
  if (f == nullptr) f = Field(doc, "rope_scaling");
  return (f != nullptr && f->is_object()) ? f : nullptr;
}

OwnedTensor LoadF32Vector(const TensorResolver& get,
                          const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "F32", "deepseek-v2: expected F32 for " + name);
  VT_CHECK(t.shape.size() == 1, "deepseek-v2: expected 1-D tensor for " + name);
  OwnedTensor o = MakeOwned(vt::DType::kF32, t.shape);
  VT_CHECK(t.nbytes == o.bytes.size(),
           "deepseek-v2: byte-size mismatch for " + name);
  std::memcpy(o.bytes.data(), t.data, t.nbytes);
  return o;
}

// `DeepseekV2MLP` (deepseek_v2.py:229-274): MergedColumnParallelLinear gate_up
// (gate rows THEN up rows — upstream's `[intermediate_size] * 2` split order at
// :246) + RowParallelLinear down. Both raw-NK for vt::MatmulBT.
DeepseekV2DenseMlp LoadDenseMlp(const TensorResolver& get,
                                const std::string& prefix) {
  DeepseekV2DenseMlp m;
  m.gate_up_proj =
      LoadMergedBf16RawNK(get, {prefix + "gate_proj.weight", prefix + "up_proj.weight"});
  m.down_proj = LoadMergedBf16RawNK(get, {prefix + "down_proj.weight"});
  return m;
}

// The load-time absorption (mla_attention.py:875-962) — see the TU header.
void AbsorbInto(DeepseekV2MlaWeights& w, const mla::MlaBlockDims& dims) {
  const int64_t N = dims.num_heads, P = dims.qk_nope_head_dim;
  const int64_t V = dims.v_head_dim, L = dims.kv_lora_rank;
  VT_CHECK(w.kv_b_proj.rank == 2 && w.kv_b_proj.shape[0] == N * (P + V) &&
               w.kv_b_proj.shape[1] == L,
           "deepseek-v2: kv_b_proj must be [num_heads*(qk_nope+v), kv_lora_rank] "
           "(deepseek_v2.py:518-519)");
  const mla::AbsorbedKvBProj absorbed = mla::AbsorbKvBProjBf16(
      reinterpret_cast<const uint16_t*>(w.kv_b_proj.bytes.data()), dims);
  w.w_uk_t = MakeOwned(vt::DType::kBF16, {N, P, L});
  std::memcpy(w.w_uk_t.bytes.data(), absorbed.w_uk_t.data(),
              absorbed.w_uk_t.size() * sizeof(uint16_t));
  w.w_uv = MakeOwned(vt::DType::kBF16, {N, L, V});
  std::memcpy(w.w_uv.bytes.data(), absorbed.w_uv.data(),
              absorbed.w_uv.size() * sizeof(uint16_t));
}

DeepseekV2MlaWeights LoadMlaLayer(const TensorResolver& get,
                                  const std::string& sa,
                                  const DeepseekV2Params& p) {
  const mla::MlaBlockDims& d = p.mla;
  DeepseekV2MlaWeights w;
  if (d.has_q_lora()) {
    // packed_modules_mapping["fused_qkv_a_proj"] = ["q_a_proj",
    // "kv_a_proj_with_mqa"] (deepseek_v2.py:1812-1820) — ONE merged owner whose
    // row blocks are [q_lora_rank | kv_lora_rank + qk_rope_head_dim], exactly the
    // layout MlaBlockWeights::fused_qkv_a_proj documents.
    w.fused_qkv_a_proj = LoadMergedBf16RawNK(
        get, {sa + "q_a_proj.weight", sa + "kv_a_proj_with_mqa.weight"});
    w.q_a_layernorm = LoadBf16Direct(get, sa + "q_a_layernorm.weight");
    w.q_b_proj = LoadMergedBf16RawNK(get, {sa + "q_b_proj.weight"});
  } else {
    w.kv_a_proj_with_mqa =
        LoadMergedBf16RawNK(get, {sa + "kv_a_proj_with_mqa.weight"});
    w.q_proj = LoadMergedBf16RawNK(get, {sa + "q_proj.weight"});
  }
  w.kv_a_layernorm = LoadBf16Direct(get, sa + "kv_a_layernorm.weight");
  w.kv_b_proj = LoadMergedBf16RawNK(get, {sa + "kv_b_proj.weight"});
  w.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  AbsorbInto(w, d);
  return w;
}

DeepseekV2MoeWeights LoadMoeLayer(const TensorResolver& get,
                                  const std::string& mlp,
                                  const DeepseekV2Params& p) {
  DeepseekV2MoeWeights m;
  m.router_gate = LoadBf16Transposed(get, mlp + "gate.weight");  // [E,H] -> [H,E]
  if (p.has_e_score_correction_bias) {
    m.e_score_correction_bias =
        LoadF32Vector(get, mlp + "gate.e_score_correction_bias");
  }
  const int64_t E = p.n_routed_experts;
  m.expert_gate.reserve(static_cast<size_t>(E));
  m.expert_up.reserve(static_cast<size_t>(E));
  m.expert_down.reserve(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    const std::string ex = mlp + "experts." + std::to_string(e) + ".";
    m.expert_gate.push_back(LoadBf16Transposed(get, ex + "gate_proj.weight"));
    m.expert_up.push_back(LoadBf16Transposed(get, ex + "up_proj.weight"));
    m.expert_down.push_back(LoadBf16Transposed(get, ex + "down_proj.weight"));
  }
  // Shared experts (deepseek_v2.py:344-357). DeepSeek-V2-Lite has TWO, fused into
  // one MLP of width `moe_intermediate_size * n_shared_experts` — this is the
  // path Qwen3-Coder never exercised.
  if (p.n_shared_experts > 0) m.shared = LoadDenseMlp(get, mlp + "shared_experts.");
  return m;
}

}  // namespace

DeepseekV2Params ParseDeepseekV2Params(const HfConfig& config) {
  const nlohmann::json& doc = config.raw;
  DeepseekV2Params p;
  p.hidden_size = config.hidden_size;
  p.num_hidden_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.intermediate_size = config.intermediate_size;
  p.rms_norm_eps = static_cast<float>(config.rms_norm_eps);
  p.tie_word_embeddings = RawBool(doc, "tie_word_embeddings", false);
  p.max_position_embeddings =
      config.max_position_embeddings > 0 ? config.max_position_embeddings : 8192;

  // --- MLA geometry (deepseek_v2.py:960-1002) ---
  mla::MlaBlockDims& d = p.mla;
  d.hidden_size = config.hidden_size;
  d.num_heads = config.num_attention_heads;
  d.qk_nope_head_dim = RawInt(doc, "qk_nope_head_dim", 0);
  d.qk_rope_head_dim = RawInt(doc, "qk_rope_head_dim", 0);
  d.v_head_dim = RawInt(doc, "v_head_dim", 0);
  d.kv_lora_rank = RawInt(doc, "kv_lora_rank", 0);
  d.q_lora_rank = RawOptionalRank(doc, "q_lora_rank");
  d.rms_norm_eps = p.rms_norm_eps;

  // The `use_mha` guard, verbatim (deepseek_v2.py:1201-1211): `model_type ==
  // "deepseek"` OR both qk dims zero selects PLAIN MHA — a model this TU does
  // not implement. Refuse rather than silently run MLA math on an MHA checkpoint.
  if (config.model_type == "deepseek" ||
      (d.qk_nope_head_dim == 0 && d.qk_rope_head_dim == 0)) {
    throw std::runtime_error(
        "DeepseekV2ForCausalLM: this checkpoint takes upstream's use_mha branch "
        "(deepseek_v2.py:1201-1211 -> DeepseekAttention, plain MHA). That is a "
        "DIFFERENT architecture and is not implemented by the MLA model TU.");
  }
  if (d.v_head_dim <= 0 || d.kv_lora_rank <= 0) {
    throw std::runtime_error(
        "DeepseekV2ForCausalLM config: v_head_dim and kv_lora_rank must be > 0");
  }

  // --- YaRN rope (deepseek_v2.py:1053-1075 + deepseek_scaling_rope.py:31-74) ---
  mla::DeepseekYarnRopeParams& r = p.rope;
  r.base = config.rope_theta;
  r.rotary_dim = d.qk_rope_head_dim;
  const RopeParameters& rp = config.rope_parameters;
  // `rope_parameters["rope_type"] != "default"` -> deepseek_yarn (when
  // apply_yarn_scaling) / deepseek_llama_scaling (:1053-1058). Only the YaRN form
  // contributes the mscale^2 softmax correction (:1067-1075).
  r.yarn = rp.rope_type != "default" && rp.apply_yarn_scaling;
  if (rp.rope_type != "default" && !rp.apply_yarn_scaling) {
    throw std::runtime_error(
        "DeepseekV2ForCausalLM config: rope_type '" + rp.rope_type +
        "' with apply_yarn_scaling=false selects upstream's "
        "`deepseek_llama_scaling` rope (deepseek_v2.py:1053-1058), which is not "
        "implemented");
  }
  if (r.yarn) {
    if (rp.rope_type != "yarn") {
      throw std::runtime_error(
          "DeepseekV2ForCausalLM config: unsupported scaled rope_type '" +
          rp.rope_type + "' (only 'yarn' is implemented)");
    }
    r.scaling_factor = rp.factor.value_or(1.0);
    r.extrapolation_factor = rp.extrapolation_factor;
    r.attn_factor = rp.attn_factor;
    r.beta_fast = static_cast<double>(rp.beta_fast);
    r.beta_slow = static_cast<double>(rp.beta_slow);
    r.original_max_position_embeddings =
        rp.original_max_position_embeddings.value_or(4096);
    // `mscale` / `mscale_all_dim` default to 1 / 0 exactly as
    // DeepseekScalingRotaryEmbedding's signature does; upstream reads
    // `mscale_all_dim` with `.get(..., False)` at :1071, i.e. 0.0 when absent.
    const nlohmann::json* rd = RopeDict(doc);
    r.mscale = rd != nullptr ? RawDouble(*rd, "mscale", 1.0) : 1.0;
    r.mscale_all_dim = rd != nullptr ? RawDouble(*rd, "mscale_all_dim", 0.0) : 0.0;
  }
  // The softmax scale INCLUDING the mscale^2 correction (:995 then :1067-1075).
  d.scale = mla::MlaAttentionScale(d, r);
  d.Validate();

  // --- MoE (deepseek_v2.py:286-393) ---
  p.n_routed_experts = RawInt(doc, "n_routed_experts", 0);
  p.num_experts_per_tok = RawInt(doc, "num_experts_per_tok", 0);
  p.moe_intermediate_size = RawInt(doc, "moe_intermediate_size", 0);
  p.n_shared_experts = RawInt(doc, "n_shared_experts", 0);
  p.first_k_dense_replace = RawInt(doc, "first_k_dense_replace", 0);
  p.moe_layer_freq = RawInt(doc, "moe_layer_freq", 1);
  p.n_group = RawInt(doc, "n_group", 1);
  p.topk_group = RawInt(doc, "topk_group", 1);
  p.norm_topk_prob = RawBool(doc, "norm_topk_prob", false);
  p.routed_scaling_factor =
      static_cast<float>(RawDouble(doc, "routed_scaling_factor", 1.0));
  const std::string scoring = RawString(doc, "scoring_func", "softmax");
  if (scoring == "softmax") {
    p.scoring_func = vt::MoeScoringFunc::kSoftmax;
  } else if (scoring == "sigmoid") {
    p.scoring_func = vt::MoeScoringFunc::kSigmoid;
  } else {
    throw std::runtime_error(
        "DeepseekV2ForCausalLM config: unsupported scoring_func '" + scoring +
        "' (grouped_topk_router.py:110-117 defines only softmax/sigmoid)");
  }
  // The learned gate bias exists ONLY for topk_method "noaux_tc" (:313-318).
  p.has_e_score_correction_bias = RawString(doc, "topk_method", "greedy") == "noaux_tc";

  if (p.n_routed_experts > 0) {
    if (p.num_experts_per_tok <= 0 || p.num_experts_per_tok > p.n_routed_experts) {
      throw std::runtime_error(
          "DeepseekV2ForCausalLM config: num_experts_per_tok must be in "
          "[1, n_routed_experts]");
    }
    if (p.moe_intermediate_size <= 0) {
      throw std::runtime_error(
          "DeepseekV2ForCausalLM config: moe_intermediate_size must be > 0");
    }
    if (p.n_group <= 0 || p.n_routed_experts % p.n_group != 0) {
      throw std::runtime_error(
          "DeepseekV2ForCausalLM config: n_group must divide n_routed_experts");
    }
    if (p.topk_group <= 0 || p.topk_group > p.n_group) {
      throw std::runtime_error(
          "DeepseekV2ForCausalLM config: topk_group must be in [1, n_group]");
    }
  }
  if (p.first_k_dense_replace > 0 && p.intermediate_size <= 0) {
    throw std::runtime_error(
        "DeepseekV2ForCausalLM config: first_k_dense_replace > 0 requires "
        "intermediate_size (the dense MLP width, deepseek_v2.py:1251-1258)");
  }
  // MTP draft layers ship as EXTRA `model.layers.{num_hidden_layers + i}.*`
  // blocks (deepseek_v2.py:1917-1933 get_spec_layer_idx_from_weight_name). We
  // load none of them, so refuse rather than silently ignore.
  if (RawInt(doc, "num_nextn_predict_layers", 0) > 0) {
    throw std::runtime_error(
        "DeepseekV2ForCausalLM: num_nextn_predict_layers > 0 (MTP draft layers) "
        "is not implemented");
  }
  // V3.2's sparse DSA indexer is selected by the presence of `index_topk`
  // (deepseek_v2.py:1068 `self.is_v32 = hasattr(config, "index_topk")`).
  if (Field(doc, "index_topk") != nullptr) {
    throw std::runtime_error(
        "DeepseekV2ForCausalLM: this is a V3.2 (DSA sparse-indexer) checkpoint; "
        "the indexer (deepseek_v2.py:613-642) is not implemented");
  }
  if (Field(doc, "quantization_config") != nullptr) {
    throw std::runtime_error(
        "DeepseekV2ForCausalLM: quantized checkpoints are not supported by the "
        "BF16 MLA loader");
  }
  return p;
}

DeepseekV2Weights LoadDeepseekV2ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "deepseek-v2: tensor not found: " + name);
    return it->second->Get(name);
  };

  DeepseekV2Weights w;
  w.params = ParseDeepseekV2Params(config);
  const DeepseekV2Params& p = w.params;
  VT_CHECK(p.num_hidden_layers > 0,
           "deepseek-v2: num_hidden_layers must be positive");

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  // DeepSeek-V2-Lite is UNTIED: the separate lm_head.weight [V,H] is loaded and
  // transposed to the Matmul-B [H,V] layout (ParallelLMHead, :1825-1833).
  if (!p.tie_word_embeddings) w.lm_head = LoadBf16Transposed(get, "lm_head.weight");

  // ONE shared YaRN [cos|sin] cache for every layer (upstream shares the single
  // `get_rope` module instance across layers via its cache — rotary_embedding/
  // __init__.py memoizes on the parameter tuple). Built in the forward dtype
  // (bf16) so vt::RopeFromCache reads it directly.
  {
    const int64_t rows = p.max_position_embeddings;
    const int64_t rot = p.mla.qk_rope_head_dim;
    const std::vector<float> cache = mla::BuildDeepseekRopeCosSinCache(p.rope, rows);
    w.rope_cos_sin_cache = MakeOwned(vt::DType::kBF16, {rows, rot});
    auto* dst = reinterpret_cast<uint16_t*>(w.rope_cos_sin_cache.bytes.data());
    for (size_t i = 0; i < cache.size(); ++i) dst[i] = vt::F32ToBF16(cache[i]);
  }

  w.layers.reserve(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".";
    DeepseekV2LayerWeights lw;
    lw.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
    lw.post_attention_layernorm =
        LoadBf16Direct(get, base + "post_attention_layernorm.weight");
    lw.attn = LoadMlaLayer(get, base + "self_attn.", p);
    lw.is_moe = p.is_moe_layer(l);
    if (lw.is_moe) {
      lw.moe = LoadMoeLayer(get, base + "mlp.", p);
    } else {
      lw.dense = LoadDenseMlp(get, base + "mlp.");
    }
    w.layers.push_back(std::move(lw));
  }
  return w;
}

}  // namespace vllm
