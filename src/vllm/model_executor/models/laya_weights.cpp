// Production weight loader for Laya (MODEL-LAYA Phase 4).
//
// Reads all F32 tensors from the safetensors checkpoint, infers the ModernBERT
// encoder config from weight shapes (ModernBERT-large defaults for fields the
// checkpoint does not carry), reads the Laya head config from config.raw
// (rl_agent_config.json fields), and calls the existing host reference loaders
// modernbert::Load + laya::Load.
//
// The checkpoint is all F32 tensors. There is no BF16 arm — the
// dense_weight_loaders.h BF16 helpers do not apply, so this loader copies raw
// F32 bytes straight from the safetensors mmap into CheckpointTensors.
#include "vllm/model_executor/models/laya.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/modernbert.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// Copy one F32 StTensor into a CheckpointTensors map. Throws if the tensor
// is not F32 — the checkpoint carries only F32, so any other dtype is a
// format mismatch the loader should surface. Templated because
// modernbert::CheckpointTensors and laya::CheckpointTensors are distinct
// types with the same Set/Get/Shape/Has interface.
template <typename CheckpointT>
void LoadF32Tensor(const StTensor& t, const std::string& name,
                   CheckpointT& out) {
  VT_CHECK(t.dtype == "F32",
           "laya: tensor '" + name + "' has dtype " + t.dtype +
               ", expected F32");
  VT_CHECK(t.data != nullptr,
           "laya: tensor '" + name + "' has null data");
  VT_CHECK(t.nbytes % sizeof(float) == 0,
           "laya: tensor '" + name + "' nbytes not a multiple of 4");
  size_t numel = t.nbytes / sizeof(float);
  const float* ptr = reinterpret_cast<const float*>(t.data);
  std::vector<float> data(ptr, ptr + numel);
  std::vector<int64_t> shape(t.shape.begin(), t.shape.end());
  out.Set(name, std::move(shape), std::move(data));
}

// Infer ModernBERT encoder params from weight shapes, with ModernBERT-large
// defaults for fields the checkpoint does not carry.
modernbert::Params InferEncoderParams(
    const modernbert::CheckpointTensors& tensors, const HfConfig& config) {
  modernbert::Params p;

  // hidden_size and vocab_size from tok_embeddings [vocab, H].
  const auto& te_shape =
      tensors.Shape("encoder.embeddings.tok_embeddings.weight");
  p.vocab_size = te_shape[0];
  p.hidden_size = te_shape[1];

  // num_hidden_layers from counting encoder.layers.N.attn.Wqkv.weight keys.
  int64_t num_layers = 0;
  while (tensors.Has("encoder.layers." + std::to_string(num_layers) +
                     ".attn.Wqkv.weight")) {
    ++num_layers;
  }
  p.num_hidden_layers = num_layers;

  // intermediate_size from layer 0 mlp.Wi [2*I, H] → I = shape[0] / 2.
  const auto& wi_shape =
      tensors.Shape("encoder.layers.0.mlp.Wi.weight");
  p.intermediate_size = wi_shape[0] / 2;

  // ModernBERT-large defaults for fields the checkpoint does not carry.
  // num_attention_heads: ModernBERT-large uses 16 heads, head_dim=64.
  p.num_attention_heads = 16;
  p.head_dim = 64;
  p.local_attention = 128;
  p.global_attn_every_n_layers = 3;
  p.local_rope_theta = 10000.0;
  p.global_rope_theta = 160000.0;
  p.layer_norm_eps = 1e-5;
  p.norm_bias = false;
  p.mlp_bias = false;
  p.attention_bias = false;

  // Override from config.raw if present (rl_agent_config.json may carry
  // encoder config under an "encoder_config" key, or the model dir may have
  // a standard ModernBERT config.json merged in).
  const auto& raw = config.raw;
  if (raw.contains("num_attention_heads") &&
      raw["num_attention_heads"].is_number()) {
    p.num_attention_heads = raw["num_attention_heads"].get<int64_t>();
  }
  if (raw.contains("local_attention") &&
      raw["local_attention"].is_number()) {
    p.local_attention = raw["local_attention"].get<int64_t>();
  }
  if (raw.contains("global_attn_every_n_layers") &&
      raw["global_attn_every_n_layers"].is_number()) {
    p.global_attn_every_n_layers =
        raw["global_attn_every_n_layers"].get<int64_t>();
  }
  if (raw.contains("local_rope_theta") &&
      raw["local_rope_theta"].is_number()) {
    p.local_rope_theta = raw["local_rope_theta"].get<double>();
  }
  if (raw.contains("global_rope_theta") &&
      raw["global_rope_theta"].is_number()) {
    p.global_rope_theta = raw["global_rope_theta"].get<double>();
  }

  return p;
}

// Parse Laya head params from config.raw (rl_agent_config.json), inferring
// any missing fields from weight shapes.
laya::Params ParseHeadParams(
    const laya::CheckpointTensors& tensors,
    const modernbert::Params& enc_params, const HfConfig& config) {
  laya::Params p;
  p.hidden_size = enc_params.hidden_size;
  p.num_heads = enc_params.num_attention_heads;
  p.dim_ff = 4 * enc_params.hidden_size;

  // head_layers from counting head.layers.N keys.
  int64_t head_layers = 0;
  while (tensors.Has("head.layers." + std::to_string(head_layers) +
                     ".self_attn.in_proj_weight")) {
    ++head_layers;
  }
  p.head_layers = head_layers > 0 ? head_layers : 2;

  // n_act from act_head.2.weight [n_act, act_hidden].
  if (tensors.Has("act_head.2.weight")) {
    const auto& ah_shape = tensors.Shape("act_head.2.weight");
    p.n_act = ah_shape[0];
    // act_hidden from act_head.0.weight [act_hidden, d+4].
    if (tensors.Has("act_head.0.weight")) {
      const auto& a0_shape = tensors.Shape("act_head.0.weight");
      p.act_hidden = a0_shape[0];
    }
  }

  // dim_ff from head.layers.0.linear1.weight [dim_ff, H].
  if (tensors.Has("head.layers.0.linear1.weight")) {
    const auto& l1_shape = tensors.Shape("head.layers.0.linear1.weight");
    p.dim_ff = l1_shape[0];
  }

  // Override from config.raw (rl_agent_config.json fields).
  const auto& raw = config.raw;
  if (raw.contains("head_layers") && raw["head_layers"].is_number()) {
    p.head_layers = raw["head_layers"].get<int64_t>();
  }
  if (raw.contains("n_act") && raw["n_act"].is_number()) {
    p.n_act = raw["n_act"].get<int64_t>();
  }

  return p;
}

}  // namespace

LayaModelWeights LoadLayaWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  // Build the name → shard index so each Get(name) finds the right file.
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;

  // Read all F32 tensors into both CheckpointTensors types. The encoder
  // tensors go into modernbert::CheckpointTensors; the head tensors go into
  // laya::CheckpointTensors. Both types share the same data because the
  // safetensors reader gives us the raw F32 bytes and we copy them.
  modernbert::CheckpointTensors enc_tensors;
  laya::CheckpointTensors head_tensors;
  for (const auto& [name, shard_ptr] : where) {
    const StTensor& t = shard_ptr->Get(name);
    // Every tensor goes into both — the Load functions look up by name, so
    // tensors the other side does not need are simply ignored.
    LoadF32Tensor(t, name, enc_tensors);
    LoadF32Tensor(t, name, head_tensors);
  }

  // Infer encoder params from weight shapes + config.
  modernbert::Params enc_params = InferEncoderParams(enc_tensors, config);

  // Parse head params from weight shapes + config.
  laya::Params head_params = ParseHeadParams(head_tensors, enc_params, config);

  // Load encoder + head weights via the existing host reference loaders,
  // which look up tensors by name and validate shapes.
  modernbert::Weights enc_weights = modernbert::Load(enc_params, enc_tensors);
  laya::Weights head_weights = laya::Load(head_params, head_tensors);

  return LayaModelWeights{
      std::move(enc_params), std::move(enc_weights),
      std::move(head_params), std::move(head_weights)};
}

}  // namespace vllm
