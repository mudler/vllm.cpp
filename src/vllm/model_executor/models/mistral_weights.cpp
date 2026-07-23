// Weight loader for the DENSE Mistral text arch (`MistralForCausalLM`,
// Mistral-7B-v0.3, BF16) — the fifth-family additive bring-up W2. Loads the
// checkpoint safetensors into the SHARED dense container (Qwen3DenseWeights,
// qwen3.h) via the SHARED dense_weight_loaders.h helpers. The name map is
// BYTE-IDENTICAL to Llama's (Mistral uses the same HF tensor names); the only
// config-driven differences are UNTIED embeddings (a standalone lm_head is loaded)
// and the absence of qk-norm (q_norm/k_norm stay empty).
//
// Grounding: vllm/model_executor/models/mistral.py @ e24d1b24 — MistralForCausalLM
// subclasses LlamaForCausalLM; MistralAttention subclasses LlamaAttention (qkv_proj
// bias=attention_bias default false; NO q_norm/k_norm; o_proj RowParallelLinear);
// MistralMLP is merged gate_up_proj -> SiluAndMul -> down_proj (bias false).
//   - packed_modules_mapping (inherited from Llama): qkv_proj<-[q,k,v]_proj,
//     gate_up_proj<-[gate,up]_proj.
//   - tie_word_embeddings=false (Mistral-7B) -> a real lm_head.weight is loaded.
//
// Name map (Mistral-7B-v0.3/config.json, flat — no multimodal prefix):
//   model.embed_tokens.weight                         -> embed_tokens [V,H]
//   model.norm.weight                                 -> final_norm [H]
//   lm_head.weight                                    -> lm_head (untied, Matmul-B)
//   model.layers.N.input_layernorm.weight             -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight    -> post_attention_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight      -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.o_proj.weight            -> o_proj (raw-NK)
//   model.layers.N.mlp.{gate,up}_proj.weight          -> merged gate_up_proj (raw)
//   model.layers.N.mlp.down_proj.weight               -> down_proj (raw-NK)
#include "vllm/model_executor/models/mistral.h"

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

// Read a top-level boolean from the raw config.json doc (Mistral configs are flat),
// defaulting when absent/null/non-boolean.
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

Qwen3DenseLayerWeights LoadMistralLayer(const TensorResolver& get, int64_t layer,
                                        bool attention_bias) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Qwen3DenseLayerWeights w;
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");

  // QKVParallelLinear: one merged owner in exact [q,k,v] output-row order
  // (packed_modules_mapping qkv_proj<-[q,k,v]_proj), kept raw-NK for MatmulBT.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  // RowParallelLinear o_proj — single raw-NK owner.
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // NO per-head q/k RMSNorm: Mistral has none, so q_norm/k_norm stay EMPTY and the
  // shared AttnBlock skips the norm step (has_qk_norm == false).
  if (attention_bias) {
    w.attn.qkv_bias = LoadMergedBf16RawNK(
        get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"});
  }

  // MergedColumnParallelLinear gate_up in exact [gate,up] order, then down_proj.
  w.mlp.gate_up_proj = LoadMergedBf16RawNK(
      get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

MistralWeights LoadMistralForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "mistral dense: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "mistral dense: num_hidden_layers must be positive");

  MistralWeights w;
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);
  w.attention_bias = RawBool(config.raw, "attention_bias", false);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  // Mistral-7B is UNTIED (tie_word_embeddings=false): a standalone lm_head.weight is
  // loaded as Matmul-B [H, vocab]. The tied branch (some Mistral fine-tunes tie) is
  // supported for completeness and mirrors the Llama loader (skip lm_head, alias
  // embed_tokens at forward time).
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadMistralLayer(get, l, w.attention_bias));
  return w;
}

}  // namespace vllm
