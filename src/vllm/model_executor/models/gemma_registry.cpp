// Gemma-1 text (`GemmaForCausalLM`) registry TU — sweep W5. Self-registers
// "GemmaForCausalLM" via REGISTER_VLLM_MODEL and owns the arch entry points: the
// config hook, the full-attention-only KV-cache spec, the LoadedModel subclass
// and the factory. Mirrors the gemma3_registry.cpp seam (new TU + one in-TU
// REGISTER line -> ZERO shared-array edit). See .agents/specs/sweep-gemma.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/gemma.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Gemma-1 text: text generation, NOT hybrid (dense
// full-attention only), NOT multimodal (gemma-2b pure text).
inline constexpr ModelInfo kGemmaInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class GemmaLoadedModel final : public LoadedModel {
 public:
  GemmaLoadedModel(const ModelRegistration& registration, GemmaWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const GemmaWeights& weights() const { return weights_; }

 private:
  GemmaWeights weights_;
};

std::unique_ptr<LoadedModel> LoadGemmaForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture GemmaForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<GemmaLoadedModel>(
      registration, LoadGemmaForCausalLMWeights(*source.safetensors, config));
}

void PrepareGemmaForCausalLM(LoadedModel& model, const HfConfig& config,
                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGemmaForCausalLM(LoadedModel& model,
                                      const ModelForwardInput& input) {
  const auto& gemma = static_cast<GemmaLoadedModel&>(model);
  const GemmaWeights& weights = gemma.weights();
  if (input.gather_logits) {
    return GemmaModel::ForwardDevice(input.token_ids, input.positions,
                                     input.attn_meta, input.attn_kv, weights,
                                     input.config, input.queue,
                                     input.logits_indices);
  }
  return HostLogits(
      GemmaModel::Forward(input.token_ids, input.positions, input.attn_meta,
                          input.attn_kv, weights, input.config, input.queue,
                          input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kGemmaFactory{
    .parse_config = &ParseGemmaForCausalLMConfig,
    .load_weights = &LoadGemmaForCausalLM,
    .prepare = &PrepareGemmaForCausalLM,
    .forward = &ForwardGemmaForCausalLM,
    .make_kv_cache = &MakeGemmaForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseGemmaForCausalLMConfig(const HfConfig& config) {
  (void)config;
}

v1::KVCacheConfig MakeGemmaForCausalLMKVCache(const HfConfig& config,
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

REGISTER_VLLM_MODEL(gemma, "GemmaForCausalLM", kGemmaFactory, kGemmaInfo)

}  // namespace vllm
