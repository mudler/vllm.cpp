// Weight loader for Phi-3 / Phi-4 dense (`Phi3ForCausalLM`, Phi-4-mini-instruct,
// BF16). Loads the checkpoint safetensors into the shared dense container and
// precomputes the LongRoPE cos/sin cache.
//
// Phi-3 checkpoints ship PRE-FUSED qkv_proj.weight ([q|k|v] rows) and gate_up_proj
// .weight ([gate|up] rows) (phi3.py packed_modules_mapping), so those are loaded
// DIRECTLY as single raw-NK owners (LoadMergedBf16RawNK with one shard). No qk-norm,
// no biases. tie_word_embeddings=true (Phi-4-mini) skips the checkpoint lm_head.
//
// The rope cache is built from the checkpoint's rope_scaling: for Phi-4-mini
// (longrope, max_position 131072 > original 4096 -> use_long_rope), the pinned
// Phi3LongRoPEScaledRotaryEmbedding builds a [original+max, rotary_dim] cache; we
// keep the LONG half ([original, original+max)), indexed by REAL position. A plain
// Phi-3 (rope_type default) builds a plain RotaryEmbedding cache.
#include "vllm/model_executor/models/phi3.h"

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::MakeOwned;

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

Qwen3DenseLayerWeights LoadPhi3Layer(const TensorResolver& get, int64_t layer) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Qwen3DenseLayerWeights w;
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");

  // PRE-FUSED qkv_proj / gate_up_proj: loaded DIRECTLY as single raw-NK owners.
  w.attn.qkv_proj = LoadMergedBf16RawNK(get, {sa + "qkv_proj.weight"});
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // No qk-norm, no biases.
  w.mlp.gate_up_proj = LoadMergedBf16RawNK(get, {mlp + "gate_up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

// Build the Phi-3 rope cos/sin cache (bf16 [P, rotary_dim], [cos|sin] halves,
// indexed by REAL position) from the checkpoint's rope_parameters.
OwnedTensor BuildPhi3RopeCache(const HfConfig& config) {
  const RopeParameters& rp = config.rope_parameters;
  const int64_t head_dim = config.head_dim;
  const int64_t rotary_dim = config.rotary_dim;
  const int64_t max_pos = config.max_position_embeddings;
  VT_CHECK(rotary_dim > 0 && rotary_dim <= head_dim,
           "phi3: rotary_dim must be in (0, head_dim]");
  VT_CHECK(max_pos > 0, "phi3: max_position_embeddings must be positive");

  std::unique_ptr<RotaryEmbeddingBase> rope;
  int64_t offset = 0;
  int64_t rows = max_pos;
  if (rp.rope_type == "longrope") {
    const int64_t orig = rp.original_max_position_embeddings.value_or(max_pos);
    // Force use_long_rope by passing max_model_len = max_pos (the oracle default:
    // model_config.max_model_len == max_position_embeddings > original).
    auto lr = std::make_unique<Phi3LongRoPEScaledRotaryEmbedding>(
        head_dim, rotary_dim, max_pos, orig, rp.rope_theta,
        /*is_neox_style=*/true, vt::DType::kBF16, rp.short_factor, rp.long_factor,
        rp.short_mscale, rp.long_mscale, /*max_model_len=*/max_pos);
    // The cache concatenates [short(orig rows) | long(max_pos rows)]. use_long_rope
    // selects the LONG half (rows [orig, orig+max_pos)), indexed by real position.
    if (lr->use_long_rope()) {
      offset = orig;
      rows = max_pos;
    } else {
      offset = 0;
      rows = orig;
    }
    rope = std::move(lr);
  } else {
    VT_CHECK(rp.rope_type == "default",
             "phi3: unsupported rope_type '" + rp.rope_type + "'");
    rope = std::make_unique<RotaryEmbedding>(head_dim, rotary_dim, max_pos,
                                             rp.rope_theta, /*is_neox_style=*/true,
                                             vt::DType::kBF16);
    offset = 0;
    rows = max_pos;
  }

  const vt::Tensor cache = rope->cos_sin_cache();  // bf16 [total_rows, rotary_dim]
  VT_CHECK(cache.rank == 2 && cache.shape[1] == rotary_dim,
           "phi3: rope cache shape mismatch");
  VT_CHECK(offset + rows <= cache.shape[0], "phi3: rope cache slice out of range");

  OwnedTensor out = MakeOwned(vt::DType::kBF16, {rows, rotary_dim});
  const size_t row_bytes = static_cast<size_t>(rotary_dim) * sizeof(uint16_t);
  const auto* src = static_cast<const uint8_t*>(cache.data) +
                    static_cast<size_t>(offset) * row_bytes;
  std::memcpy(out.bytes.data(), src, static_cast<size_t>(rows) * row_bytes);
  return out;
}

}  // namespace

Phi3Weights LoadPhi3ForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                       const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "phi3: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "phi3: num_hidden_layers must be positive");

  Phi3Weights w;
  w.dense.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", true);
  w.dense.attention_bias = false;

  w.dense.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.dense.final_norm = LoadBf16Direct(get, "model.norm.weight");
  if (!w.dense.tie_word_embeddings) {
    w.dense.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.dense.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.dense.layers.push_back(LoadPhi3Layer(get, l));

  w.rope_cos_sin = BuildPhi3RopeCache(config);
  return w;
}

}  // namespace vllm
