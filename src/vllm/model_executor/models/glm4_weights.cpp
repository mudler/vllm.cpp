// Weight loader for GLM-4-0414 dense (`Glm4ForCausalLM`, GLM-4-9B-0414, BF16) —
// task G2/W2. Loads the checkpoint safetensors into Glm4Weights (glm4.h) via the
// SHARED dense_weight_loaders.h helpers.
//
// Grounding: vllm/model_executor/models/glm4.py @ e24d1b24 —
//   - Glm4Attention (:55-140): qkv_proj (QKVParallelLinear, bias=attention_bias)
//     split [q_size,kv_size,kv_size]; NO q/k norm; o_proj (RowParallelLinear, no
//     bias).
//   - Glm4MLP = LlamaMLP: merged gate_up_proj -> SiluAndMul -> down_proj.
//   - Glm4DecoderLayer (:167-187): FOUR RMSNorms (input_layernorm,
//     post_attention_layernorm, post_self_attn_layernorm, post_mlp_layernorm).
//   - packed_modules_mapping (:229-232): qkv_proj<-[q,k,v]_proj,
//     gate_up_proj<-[gate,up]_proj.
//   - load_weights (:295-305,308): skip lm_head only when tied (GLM-4-9B-0414 is
//     UNTIED → lm_head loaded); skip the speculative (MTP) tail layers
//     model.layers.{num_hidden_layers+i}.
//
// Name map (GLM-4-9B-0414/config.json, flat — NO multimodal prefix):
//   model.embed_tokens.weight                          -> embed_tokens [V,H]
//   model.norm.weight                                  -> final_norm [H]
//   lm_head.weight                                     -> lm_head (untied)
//   model.layers.N.input_layernorm.weight              -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight     -> post_attention_ln [H]
//   model.layers.N.post_self_attn_layernorm.weight     -> post_self_attn_ln [H]
//   model.layers.N.post_mlp_layernorm.weight           -> post_mlp_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight       -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.{q,k,v}_proj.bias         -> merged qkv_bias
//   model.layers.N.self_attn.o_proj.weight             -> o_proj (raw-NK, no bias)
//   model.layers.N.mlp.gate_up_proj.weight             -> gate_up_proj (pre-merged raw-NK)
//   model.layers.N.mlp.down_proj.weight                -> down_proj (raw-NK)
#include "vllm/model_executor/models/glm4.h"

#include <functional>
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
using dense_loaders::LoadMergedBf16Vector;

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

Glm4LayerWeights LoadGlm4Layer(const TensorResolver& get, int64_t layer,
                               bool attention_bias) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Glm4LayerWeights w;
  // FOUR sandwich RMSNorms (glm4.py:180-187). Always BF16.
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  w.post_self_attn_layernorm =
      LoadBf16Direct(get, base + "post_self_attn_layernorm.weight");
  w.post_mlp_layernorm = LoadBf16Direct(get, base + "post_mlp_layernorm.weight");

  // QKVParallelLinear (bias=attention_bias): one merged owner in exact [q,k,v]
  // output-row order (packed_modules_mapping), kept raw-NK for MatmulBT.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  if (attention_bias) {
    // qkv bias is a 1-D [out] vector per shard — concat with the 1-D merger
    // (LoadMergedBf16Vector), NOT the 2-D LoadMergedBf16RawNK (OPT precedent).
    w.attn.qkv_bias = LoadMergedBf16Vector(
        get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"});
  }
  // RowParallelLinear o_proj — single raw-NK owner, NO bias.
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});

  // GLM-4-9B-0414 ships the MLP gate_up ALREADY MERGED as a single
  // mlp.gate_up_proj.weight [2I, H] tensor (rows gate|up), not separate
  // gate_proj/up_proj — load it directly (raw-NK), then down_proj.
  w.mlp.gate_up_proj = LoadMergedBf16RawNK(get, {mlp + "gate_up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

Glm4Weights LoadGlm4ForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                       const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "glm4: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "glm4: num_hidden_layers must be positive");

  Glm4Weights w;
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);
  const bool attention_bias = RawBool(config.raw, "attention_bias", false);
  VT_CHECK(attention_bias, "glm4: GLM-4-9B-0414 has attention_bias=true");

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  // GLM-4-9B-0414 is UNTIED: load the standalone lm_head (Matmul-B [H, vocab]).
  // The tied path (skip lm_head, alias embed_tokens) is retained for parity with
  // the qwen3 template but is not exercised by this checkpoint.
  if (!w.tie_word_embeddings) w.lm_head = LoadBf16Transposed(get, "lm_head.weight");

  // Only [0, num_hidden_layers) are materialized — the speculative (MTP) tail
  // layers model.layers.{num_hidden_layers+i} are SKIPPED by construction
  // (glm4.py:295-305,308).
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadGlm4Layer(get, l, attention_bias));
  return w;
}

}  // namespace vllm
