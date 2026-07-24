// Weight loader for Granite-3 (`GraniteForCausalLM`, granite-3.3-2b-instruct,
// BF16). Loads the checkpoint safetensors into the SHARED dense container
// (Qwen3DenseWeights, qwen3.h) via the SHARED dense_weight_loaders.h helpers. The
// name map is IDENTICAL to Llama (Granite is Llama + scalar multipliers, which
// live in config, not the weights): separate q/k/v merged to one qkv_proj, gate/up
// merged to one gate_up_proj, tied lm_head, NO qk-norm, NO biases.
//
// Grounding: vllm/model_executor/models/granite.py @ e24d1b24 —
//   - GraniteAttention: qkv_proj (QKVParallelLinear, bias=attention_bias, default
//     false) split [q,k,v]; NO q_norm/k_norm; o_proj (RowParallelLinear).
//   - GraniteMLP: merged gate_up_proj -> SiluAndMul -> down_proj (mlp_bias false).
//   - hf_to_vllm_mapper: q/k/v_proj -> qkv_proj, gate/up_proj -> gate_up_proj.
//   - tie_word_embeddings: lm_head aliases embed_tokens; the loader skips the
//     checkpoint lm_head.weight (granite-3.3-2b ties).
#include "vllm/model_executor/models/granite.h"

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

Qwen3DenseLayerWeights LoadGraniteLayer(const TensorResolver& get, int64_t layer,
                                        bool attention_bias) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Qwen3DenseLayerWeights w;
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");

  // QKVParallelLinear: one merged owner in exact [q,k,v] output-row order, raw-NK.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // Granite has NO per-head q/k RMSNorm (q_norm/k_norm stay EMPTY).
  if (attention_bias) {
    w.attn.qkv_bias = LoadMergedBf16RawNK(
        get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"});
  }

  w.mlp.gate_up_proj = LoadMergedBf16RawNK(
      get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

GraniteWeights LoadGraniteForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "granite: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "granite: num_hidden_layers must be positive");

  GraniteWeights w;
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", true);
  w.attention_bias = RawBool(config.raw, "attention_bias", false);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadGraniteLayer(get, l, w.attention_bias));
  return w;
}

}  // namespace vllm
