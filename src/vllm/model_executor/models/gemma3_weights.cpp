// Weight loader for `Gemma3ForCausalLM` (gemma-3-1b-it, BF16) — sweep W2. Loads
// the checkpoint safetensors into Gemma3Weights (gemma3.h) via the shared
// dense_weight_loaders.h helpers.
//
// Grounding: vllm/model_executor/models/gemma3.py @ e24d1b24 —
//   packed_modules_mapping (:382-392): qkv_proj<-[q,k,v]_proj,
//   gate_up_proj<-[gate,up]_proj. tie_word_embeddings (:411-412,447): lm_head
//   aliases embed_tokens, the checkpoint's lm_head.weight is skipped.
//
// Name map (gemma-3-1b-it, flat `model.` prefix, no multimodal wrapper):
//   model.embed_tokens.weight                            -> embed_tokens [V,H]
//   model.norm.weight                                    -> final_norm [H]
//   lm_head.weight                                       -> SKIPPED (tied)
//   model.layers.N.input_layernorm.weight                -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight       -> post_attention_ln [H]
//   model.layers.N.pre_feedforward_layernorm.weight      -> pre_feedforward_ln [H]
//   model.layers.N.post_feedforward_layernorm.weight     -> post_feedforward_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight         -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.o_proj.weight               -> o_proj (raw-NK)
//   model.layers.N.self_attn.{q,k}_norm.weight           -> q_norm/k_norm [Dh]
//   model.layers.N.mlp.{gate,up}_proj.weight             -> merged gate_up_proj (raw)
//   model.layers.N.mlp.down_proj.weight                  -> down_proj (raw-NK)
#include "vllm/model_executor/models/gemma3.h"

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

// Read a top-level boolean from the raw config.json doc, defaulting when
// absent/null/non-boolean. Gemma3 configs are flat (no text_config nesting for
// the standalone Gemma3ForCausalLM checkpoint).
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

Gemma3LayerWeights LoadGemma3Layer(const TensorResolver& get, int64_t layer) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Gemma3LayerWeights w;
  // Four sandwich GemmaRMSNorm weights (gemma3.py:254-263).
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  w.pre_feedforward_layernorm =
      LoadBf16Direct(get, base + "pre_feedforward_layernorm.weight");
  w.post_feedforward_layernorm =
      LoadBf16Direct(get, base + "post_feedforward_layernorm.weight");

  // QKVParallelLinear: one merged owner in [q,k,v] output-row order (raw-NK),
  // no bias (attention_bias=false). RowParallelLinear o_proj (raw-NK).
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // Per-head Gemma q/k RMSNorm (GemmaRMSNorm(head_dim), 1+w) before RoPE.
  w.attn.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  w.attn.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");

  // MergedColumnParallelLinear gate_up in [gate,up] order, then down_proj.
  w.mlp.gate_up_proj = LoadMergedBf16RawNK(
      get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

Gemma3Weights LoadGemma3ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "gemma3: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "gemma3: num_hidden_layers must be positive");

  Gemma3Weights w;
  // Gemma ties embeddings by default (Gemma3TextConfig.tie_word_embeddings=True).
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", true);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadGemma3Layer(get, l));
  return w;
}

}  // namespace vllm
