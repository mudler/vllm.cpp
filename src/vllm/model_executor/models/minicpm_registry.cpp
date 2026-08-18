// MiniCPM (`MiniCPMForCausalLM`) registry TU — the ZERO-NEW-KERNEL dense bring-up
// (Llama + 3 scalar deltas). Self-registers "MiniCPMForCausalLM" via
// REGISTER_VLLM_MODEL and owns the arch entry points: config hook, full-attention-
// ONLY KV-cache spec, the LoadedModel subclass, and the factory. Mirrors the
// granite_registry.cpp seam (new TU + one in-TU REGISTER line -> ZERO shared-array
// edit). See .agents/specs/sweep-recent-dense-batch.md §0.2 row 4.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/minicpm.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kMiniCPMInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class MiniCPMLoadedModel final : public LoadedModel {
 public:
  MiniCPMLoadedModel(const ModelRegistration& registration, MiniCPMWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const MiniCPMWeights& weights() const { return weights_; }

 private:
  MiniCPMWeights weights_;
};

std::unique_ptr<LoadedModel> LoadMiniCPMForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture MiniCPMForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<MiniCPMLoadedModel>(
      registration, LoadMiniCPMForCausalLMWeights(*source.safetensors, config));
}

void PrepareMiniCPMForCausalLM(LoadedModel& model, const HfConfig& config,
                               vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardMiniCPMForCausalLM(LoadedModel& model,
                                        const ModelForwardInput& input) {
  const auto& minicpm = ModelAs<MiniCPMLoadedModel>(model, "MiniCPMForCausalLM");
  const MiniCPMWeights& weights = minicpm.weights();
  if (input.gather_logits) {
    return MiniCPMModel::ForwardDevice(input.token_ids, input.positions,
                                       input.attn_meta, input.attn_kv, weights,
                                       input.config, input.queue,
                                       input.logits_indices);
  }
  return HostLogits(
      MiniCPMModel::Forward(input.token_ids, input.positions, input.attn_meta,
                            input.attn_kv, weights, input.config, input.queue,
                            input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kMiniCPMFactory{
    .parse_config = &ParseMiniCPMForCausalLMConfig,
    .load_weights = &LoadMiniCPMForCausalLM,
    .prepare = &PrepareMiniCPMForCausalLM,
    .forward = &ForwardMiniCPMForCausalLM,
    .make_kv_cache = &MakeMiniCPMForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseMiniCPMForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig materializes the typed fields; the three MiniCPM scalars are read
  // from config.raw by the forward. Validate plain (non-partial) NeoX rope and
  // reject the config-gated MoE variant (dense checkpoint only for this row).
  VT_CHECK(config.rotary_dim == config.head_dim,
           "minicpm: expected full NeoX rope (rotary_dim == head_dim)");
  const auto it = config.raw.find("num_experts");
  const bool is_moe =
      it != config.raw.end() && it->is_number() && it->get<int>() > 0;
  VT_CHECK(!is_moe,
           "minicpm: MoE variant (num_experts>0) not supported by this row "
           "(dense MiniCPMForCausalLM only)");
}

v1::KVCacheConfig MakeMiniCPMForCausalLMKVCache(const HfConfig& config,
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

REGISTER_VLLM_MODEL(minicpm, "MiniCPMForCausalLM", kMiniCPMFactory, kMiniCPMInfo)

}  // namespace vllm
