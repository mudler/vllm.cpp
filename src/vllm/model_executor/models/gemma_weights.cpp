// Weight loader for `GemmaForCausalLM` (gemma-2b, BF16) — sweep W5. Loads the
// checkpoint safetensors into GemmaWeights (gemma.h) via the shared
// dense_weight_loaders.h helpers.
//
// Grounding: vllm/model_executor/models/gemma.py @ e24d1b24 —
//   packed_modules_mapping (:325-328): qkv_proj<-[q,k,v]_proj,
//   gate_up_proj<-[gate,up]_proj. tie_word_embeddings (:335,338-339): lm_head
//   aliases embed_tokens, the checkpoint's lm_head.weight skipped.
//
// Name map (gemma-2b, flat `model.` prefix). TWO norms per layer only (no
// sandwich pre/post-ff, no q/k norm):
//   model.embed_tokens.weight                       -> embed_tokens [V,H]
//   model.norm.weight                               -> final_norm [H]
//   lm_head.weight                                  -> SKIPPED (tied)
//   model.layers.N.input_layernorm.weight           -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight  -> post_attention_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight    -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.o_proj.weight          -> o_proj (raw-NK)
//   model.layers.N.mlp.{gate,up}_proj.weight        -> merged gate_up_proj (raw)
//   model.layers.N.mlp.down_proj.weight             -> down_proj (raw-NK)
#include "vllm/model_executor/models/gemma.h"

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

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

GemmaLayerWeights LoadGemmaLayer(const TensorResolver& get, int64_t layer) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  GemmaLayerWeights w;
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");

  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});

  w.mlp.gate_up_proj = LoadMergedBf16RawNK(
      get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

GemmaWeights LoadGemmaForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "gemma: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "gemma: num_hidden_layers must be positive");

  GemmaWeights w;
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", true);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadGemmaLayer(get, l));
  return w;
}

}  // namespace vllm
