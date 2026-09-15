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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/gemma3.h"
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

Gemma3Weights LoadGemma3ForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                           const HfConfig& config, vt::Queue* load_queue) {
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
  const bool compiled_device = config.rope_parameters.rope_type == "linear" &&
                               load_queue != nullptr &&
                               vt::GetBackend(load_queue->device.type).GetResidualNormPolicy() ==
                                   vt::ResidualNormPolicy::kCompiledExpression;
  auto enabled = [](const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] != '0';
  };
  const bool stage_weights =
      compiled_device && enabled("VT_DIRECT_DEVICE_LOAD") && enabled("VT_RELEASE_HOST_WEIGHTS");
  auto stage = [&](OwnedTensor& tensor) {
    if (!stage_weights || tensor.Empty()) return;
    dense_attn::Dev device{vt::GetBackend(load_queue->device.type), *load_queue};
    dense_attn::ResidentWeight(device, tensor);
    device.b.Synchronize(device.q);
    // The compiled forward reads the authoritative device copy. Release each
    // completed tensor here so loading never retains the whole host model.
    tensor.ReleaseHost();
  };
  if (config.rope_parameters.rope_type == "linear") {
    auto build_cache = [&](const RopeParameters& params) {
      if (compiled_device) {
        // The primary constructs its cache on-device. Avoid building and
        // retaining a second, unused CPU cache before the weights are loaded.
        const double factor = params.rope_type == "linear" ? params.factor.value_or(1.) : 1.;
        const double rows = std::ceil(static_cast<double>(config.max_position_embeddings) * factor);
        VT_CHECK(std::isfinite(factor) && factor > 0. && std::isfinite(rows) && rows > 0. &&
                     config.head_dim > 0 && config.head_dim % 2 == 0 &&
                     static_cast<long double>(rows) <=
                         static_cast<long double>(std::numeric_limits<int64_t>::max() /
                                                  sizeof(float) / config.head_dim),
                 "gemma3: invalid device RoPE cache dimensions");
        OwnedTensor out;
        out.dtype = vt::DType::kBF16;
        out.rank = 2;
        out.shape[0] = static_cast<int64_t>(rows);
        out.shape[1] = config.head_dim;
        dense_attn::Dev device{vt::GetBackend(load_queue->device.type), *load_queue};
        dense_attn::ResidentWeight(device, out, {}, [&](vt::Tensor& target) {
          std::vector<int64_t> positions(static_cast<size_t>(out.shape[0]));
          std::iota(positions.begin(), positions.end(), int64_t{0});
          dense_attn::DBuf indices(device, vt::DType::kI64, {out.shape[0]}, positions.data());
          // FP32 construction followed by BF16 storage mirrors rotary_embedding/base.py.
          dense_attn::DBuf generated(device, vt::DType::kF32, {out.shape[0], config.head_dim});
          vt::RopeArgs args;
          args.base = static_cast<float>(params.rope_theta);
          args.rotary_dim = static_cast<int>(config.head_dim);
          args.linear_scaling_factor = static_cast<float>(factor);
          vt::RopeCosSinCache(device.q, generated.t(), indices.t(), args);
          vt::CastBf16(device.q, target, generated.t());
          device.b.Synchronize(device.q);  // Keep the host positions alive through their upload.
        });
        return out;
      }
      auto rope = get_rope(config.head_dim, config.max_position_embeddings,
          /*is_neox_style=*/true, params, vt::DType::kBF16);
      const auto cache = rope->cos_sin_cache();
      OwnedTensor out;
      out.dtype = vt::DType::kBF16;
      out.rank = 2;
      out.shape[0] = cache.shape[0];
      out.shape[1] = cache.shape[1];
      out.bytes = OwnedBytes::Borrow(static_cast<const uint8_t*>(cache.data), cache.Bytes(), rope);
      return out;
    };
    w.rope_global = build_cache(config.rope_parameters);
    // gemma3.py:170-178: sliding layers override the global scaling and theta.
    RopeParameters local;
    const auto local_base = config.raw.find("rope_local_base_freq");
    local.rope_theta = local_base != config.raw.end() && local_base->is_number()
                           ? local_base->get<double>() : 10000.;
    local.rope_dim = config.head_dim;
    w.rope_local = build_cache(local);
  }
  // Gemma ties embeddings by default (Gemma3TextConfig.tie_word_embeddings=True).
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", true);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  stage(w.embed_tokens);
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  stage(w.final_norm);
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
    stage(w.lm_head);
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    auto layer = LoadGemma3Layer(get, l);
    for (OwnedTensor* tensor :
         {&layer.input_layernorm, &layer.post_attention_layernorm, &layer.pre_feedforward_layernorm,
          &layer.post_feedforward_layernorm, &layer.attn.qkv_proj, &layer.attn.o_proj,
          &layer.attn.q_norm, &layer.attn.k_norm, &layer.mlp.gate_up_proj, &layer.mlp.down_proj})
      stage(*tensor);
    w.layers.push_back(std::move(layer));
  }
  return w;
}

}  // namespace vllm
