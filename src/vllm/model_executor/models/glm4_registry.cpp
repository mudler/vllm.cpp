// GLM-4-0414 dense (`Glm4ForCausalLM`, GLM-4-9B-0414) registry TU — task G2, the
// first GLM-family model. Self-registers "Glm4ForCausalLM" via REGISTER_VLLM_MODEL
// and owns the arch entry points: config hook, full-attention-ONLY KV-cache spec
// (one FA group, NO MambaSpec/GDN group), the LoadedModel subclass, and the
// factory. Mirrors the qwen3_dense.cpp seam (new TU + one in-TU REGISTER line →
// ZERO shared-array edit). See .agents/specs/glm-dsa-latest-deepseek.md §Port map.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/glm4.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for GLM-4 dense: text generation, NOT hybrid (pure
// full-attention), NOT multimodal.
inline constexpr ModelInfo kGlm4Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class Glm4LoadedModel final : public LoadedModel {
 public:
  Glm4LoadedModel(const ModelRegistration& registration, Glm4Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const Glm4Weights& weights() const { return weights_; }

 private:
  Glm4Weights weights_;
};

std::unique_ptr<LoadedModel> LoadGlm4ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Glm4ForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Glm4LoadedModel>(
      registration, LoadGlm4ForCausalLMWeights(*source.safetensors, config));
}

void PrepareGlm4ForCausalLM(LoadedModel& model, const HfConfig& config,
                            vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGlm4ForCausalLM(LoadedModel& model,
                                     const ModelForwardInput& input) {
  const auto& glm = static_cast<Glm4LoadedModel&>(model);
  const Glm4Weights& weights = glm.weights();
  if (input.gather_logits) {
    return Glm4Model::ForwardDevice(input.token_ids, input.positions,
                                    input.attn_meta, input.attn_kv, weights,
                                    input.config, input.queue,
                                    input.logits_indices);
  }
  return HostLogits(
      Glm4Model::Forward(input.token_ids, input.positions, input.attn_meta,
                         input.attn_kv, weights, input.config, input.queue,
                         input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kGlm4Factory{
    .parse_config = &ParseGlm4ForCausalLMConfig,
    .load_weights = &LoadGlm4ForCausalLM,
    .prepare = &PrepareGlm4ForCausalLM,
    .forward = &ForwardGlm4ForCausalLM,
    .make_kv_cache = &MakeGlm4ForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseGlm4ForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig already materializes the consumed GLM-4 fields (num_key_value_
  // heads, head_dim, rope_theta, rotary_dim from partial_rotary_factor,
  // intermediate_size, rms_norm_eps). Validate the GLM invariant: partial rotary
  // must be a proper leading slice (rotary_dim < head_dim for GLM-4-9B-0414).
  VT_CHECK(config.rotary_dim > 0 && config.rotary_dim <= config.head_dim,
           "glm4: rotary_dim must be in (0, head_dim] (partial_rotary_factor)");
}

v1::KVCacheConfig MakeGlm4ForCausalLMKVCache(const HfConfig& config, int block_size,
                                             int num_blocks) {
  // Pure dense: exactly ONE full-attention KV group, NO MambaSpec/GDN group.
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, num_kv_heads, head_dim,
                                              v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(glm4, "Glm4ForCausalLM", kGlm4Factory, kGlm4Info)

}  // namespace vllm
