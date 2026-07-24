// Granite-3 (`GraniteForCausalLM`) registry TU — the ZERO-NEW-KERNEL dense
// bring-up (Llama + 4 scalar multipliers). Self-registers "GraniteForCausalLM" via
// REGISTER_VLLM_MODEL and owns the arch entry points: config hook, full-attention-
// ONLY KV-cache spec, the LoadedModel subclass, and the factory. Mirrors the
// mistral_registry.cpp seam (new TU + one in-TU REGISTER line -> ZERO shared-array
// edit). See .agents/specs/sweep-recent-dense-batch.md §0.2 row 2.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/granite.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kGraniteInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class GraniteLoadedModel final : public LoadedModel {
 public:
  GraniteLoadedModel(const ModelRegistration& registration, GraniteWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const GraniteWeights& weights() const { return weights_; }

 private:
  GraniteWeights weights_;
};

std::unique_ptr<LoadedModel> LoadGraniteForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture GraniteForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<GraniteLoadedModel>(
      registration, LoadGraniteForCausalLMWeights(*source.safetensors, config));
}

void PrepareGraniteForCausalLM(LoadedModel& model, const HfConfig& config,
                               vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGraniteForCausalLM(LoadedModel& model,
                                        const ModelForwardInput& input) {
  const auto& granite = static_cast<GraniteLoadedModel&>(model);
  const GraniteWeights& weights = granite.weights();
  if (input.gather_logits) {
    return GraniteModel::ForwardDevice(input.token_ids, input.positions,
                                       input.attn_meta, input.attn_kv, weights,
                                       input.config, input.queue,
                                       input.logits_indices);
  }
  return HostLogits(
      GraniteModel::Forward(input.token_ids, input.positions, input.attn_meta,
                            input.attn_kv, weights, input.config, input.queue,
                            input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kGraniteFactory{
    .parse_config = &ParseGraniteForCausalLMConfig,
    .load_weights = &LoadGraniteForCausalLM,
    .prepare = &PrepareGraniteForCausalLM,
    .forward = &ForwardGraniteForCausalLM,
    .make_kv_cache = &MakeGraniteForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseGraniteForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig materializes the typed fields; the four Granite scalars are read
  // from config.raw by the forward. Validate plain (non-partial) NeoX rope.
  VT_CHECK(config.rotary_dim == config.head_dim,
           "granite: expected full NeoX rope (rotary_dim == head_dim)");
}

v1::KVCacheConfig MakeGraniteForCausalLMKVCache(const HfConfig& config,
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

REGISTER_VLLM_MODEL(granite, "GraniteForCausalLM", kGraniteFactory, kGraniteInfo)

}  // namespace vllm
