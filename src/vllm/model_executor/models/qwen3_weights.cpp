// Weight loader for the DENSE Qwen3 text gate (`Qwen3ForCausalLM`, Qwen3-0.6B,
// BF16) — the first ADDITIVE-MODEL bring-up W2. Loads the checkpoint safetensors
// into Qwen3DenseWeights (qwen3.h) via the SHARED dense_weight_loaders.h helpers
// (SEAM GAP #3 extraction) rather than re-deriving them.
//
// Grounding: vllm/model_executor/models/qwen3.py @ e24d1b24 —
//   - Qwen3Attention (:65-168): qkv_proj (QKVParallelLinear) split
//     [q_size,kv_size,kv_size]; per-head q_norm/k_norm = RMSNorm(head_dim);
//     o_proj (RowParallelLinear). bias = attention_bias (default false).
//   - Qwen3MLP = Qwen2MLP (:58): merged gate_up_proj -> SiluAndMul -> down_proj.
//   - packed_modules_mapping (:271-274): qkv_proj<-[q,k,v]_proj,
//     gate_up_proj<-[gate,up]_proj.
//   - tie_word_embeddings (:294-295): lm_head aliases embed_tokens; the loader
//     skips the checkpoint's lm_head.weight via skip_prefixes (:339).
//
// Name map (Qwen3-0.6B/config.json, NO multimodal `language_model` prefix):
//   model.embed_tokens.weight                         -> embed_tokens [V,H]
//   model.norm.weight                                 -> final_norm [H]
//   lm_head.weight                                    -> SKIPPED when tied
//   model.layers.N.input_layernorm.weight             -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight    -> post_attention_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight      -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.o_proj.weight            -> o_proj (raw-NK)
//   model.layers.N.self_attn.{q,k}_norm.weight        -> q_norm/k_norm [Dh]
//   model.layers.N.mlp.{gate,up}_proj.weight          -> merged gate_up_proj (raw)
//   model.layers.N.mlp.down_proj.weight               -> down_proj (raw-NK)
#include "vllm/model_executor/models/qwen3.h"

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

// Read a top-level boolean from the raw config.json doc (Qwen3 dense configs are
// flat — no text_config nesting), defaulting when absent/null/non-boolean.
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

Qwen3DenseLayerWeights LoadQwen3Layer(const TensorResolver& get, int64_t layer,
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
  // Per-head q/k RMSNorm (RMSNorm(head_dim)), applied before RoPE.
  w.attn.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  w.attn.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");
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

Qwen3DenseWeights LoadQwen3ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3 dense: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "qwen3 dense: num_hidden_layers must be positive");

  Qwen3DenseWeights w;
  // Deferred from W0: tie_word_embeddings + attention_bias (qwen3.h notes).
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);
  w.attention_bias = RawBool(config.raw, "attention_bias", false);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  // tie_word_embeddings: lm_head aliases embed_tokens; the checkpoint's redundant
  // lm_head.weight is SKIPPED (mirrors vLLM skip_prefixes=["lm_head."]). Only the
  // untied case loads a standalone lm_head (Matmul-B [H, vocab]).
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadQwen3Layer(get, l, w.attention_bias));
  return w;
}

}  // namespace vllm
