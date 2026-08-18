// StableLM (`StableLmForCausalLM`, stabilityai/stablelm-2-1_6b) registry TU — the
// ZERO-NEW-KERNEL dense bring-up (nn.LayerNorm block + partial NeoX RoPE +
// optional qkv bias + SwiGLU, all REUSE). Self-registers "StableLmForCausalLM"
// via REGISTER_VLLM_MODEL and owns the arch entry points + config helpers. Mirrors
// the phi3_registry.cpp seam (new TU + one in-TU REGISTER line -> ZERO shared-array
// edit). See .agents/specs/sweep-recent-dense-batch.md §0.2 row 3.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/model_executor/models/stablelm.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kStablelmInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class StablelmLoadedModel final : public LoadedModel {
 public:
  StablelmLoadedModel(const ModelRegistration& registration, StablelmWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const StablelmWeights& weights() const { return weights_; }

 private:
  StablelmWeights weights_;
};

std::unique_ptr<LoadedModel> LoadStableLmForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture StableLmForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<StablelmLoadedModel>(
      registration, LoadStableLmForCausalLMWeights(*source.safetensors, config));
}

void PrepareStableLmForCausalLM(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardStableLmForCausalLM(LoadedModel& model,
                                         const ModelForwardInput& input) {
  const auto& sm = ModelAs<StablelmLoadedModel>(model, "StableLmForCausalLM");
  const StablelmWeights& weights = sm.weights();
  if (input.gather_logits) {
    return StablelmModel::ForwardDevice(input.token_ids, input.positions,
                                        input.attn_meta, input.attn_kv, weights,
                                        input.config, input.queue,
                                        input.logits_indices);
  }
  return HostLogits(
      StablelmModel::Forward(input.token_ids, input.positions, input.attn_meta,
                             input.attn_kv, weights, input.config, input.queue,
                             input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kStablelmFactory{
    .parse_config = &ParseStableLmForCausalLMConfig,
    .load_weights = &LoadStableLmForCausalLM,
    .prepare = &PrepareStableLmForCausalLM,
    .forward = &ForwardStableLmForCausalLM,
    .make_kv_cache = &MakeStableLmForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

// nn.LayerNorm eps: getattr(config, "norm_eps", getattr(config, "layer_norm_eps",
// 1e-05)) (stablelm.py:187,240). stablelm-2-1_6b sets layer_norm_eps 1e-5.
float StablelmLayerNormEps(const HfConfig& config) {
  const nlohmann::json& doc = config.raw;
  auto read = [&doc](const char* key, double& out) -> bool {
    const auto it = doc.find(key);
    if (it == doc.end() || it->is_null() || !it->is_number()) return false;
    out = it->get<double>();
    return true;
  };
  double eps = 1e-5;
  if (!read("norm_eps", eps)) read("layer_norm_eps", eps);
  return static_cast<float>(eps);
}

// use_qkv_bias (stablelm.py:117). stablelm-2-1_6b sets it True; older
// stablelm-3b-4e1t leaves it False.
bool StablelmUseQkvBias(const HfConfig& config) {
  const nlohmann::json& doc = config.raw;
  const auto it = doc.find("use_qkv_bias");
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return false;
  return it->get<bool>();
}

void ParseStableLmForCausalLMConfig(const HfConfig& config) {
  VT_CHECK(config.rotary_dim > 0 && config.rotary_dim <= config.head_dim,
           "stablelm: rotary_dim must be in (0, head_dim]");
  VT_CHECK(config.rope_parameters.rope_type == "default",
           "stablelm: expected default rope");
  VT_CHECK(config.num_attention_heads > 0 && config.num_key_value_heads > 0,
           "stablelm: head counts must be positive");
  VT_CHECK(config.num_attention_heads % config.num_key_value_heads == 0,
           "stablelm: num_attention_heads must be divisible by num_key_value_heads");
}

v1::KVCacheConfig MakeStableLmForCausalLMKVCache(const HfConfig& config,
                                                 int block_size, int num_blocks) {
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(stablelm, "StableLmForCausalLM", kStablelmFactory, kStablelmInfo)

}  // namespace vllm
