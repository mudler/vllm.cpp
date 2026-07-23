// Mistral (`MistralForCausalLM`) registry TU — the fifth-family near-additive dense
// bring-up. Self-registers "MistralForCausalLM" via REGISTER_VLLM_MODEL and owns the
// arch-specific entry points: the config hook, the full-attention-ONLY KV-cache
// spec, the LoadedModel subclass, and the factory. Mirrors the llama_registry.cpp
// seam (new TU + one in-TU REGISTER line -> ZERO shared-array edit).
//
// The forward + KV topology are REUSED VERBATIM from the Qwen3-dense path
// (MistralModel == Qwen3DenseModel): Mistral is that forward with qk-norm skipped
// and PLAIN rope (theta 1e6, no scaling — simpler than Llama's llama3 rescale) and
// an untied lm_head, all handled by the shared dense machinery. So this TU only
// wires the loader + forward hooks. See .agents/specs/sweep-mistral.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/mistral.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Mistral: text generation, NOT hybrid (pure
// full-attention), NOT multimodal (text-only checkpoint).
inline constexpr ModelInfo kMistralInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: holds the loaded Mistral dense weights (shared dense
// container); the forward reuses the Qwen3-dense forward machinery.
class MistralLoadedModel final : public LoadedModel {
 public:
  MistralLoadedModel(const ModelRegistration& registration, MistralWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const MistralWeights& weights() const { return weights_; }

 private:
  MistralWeights weights_;
};

std::unique_ptr<LoadedModel> LoadMistralForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  // Mistral dense is text-only BF16 safetensors (no GGUF path yet).
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture MistralForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<MistralLoadedModel>(
      registration, LoadMistralForCausalLMWeights(*source.safetensors, config));
}

void PrepareMistralForCausalLM(LoadedModel& model, const HfConfig& config,
                               vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardMistralForCausalLM(LoadedModel& model,
                                        const ModelForwardInput& input) {
  const auto& mistral = static_cast<MistralLoadedModel&>(model);
  const MistralWeights& weights = mistral.weights();
  // DEVICE-resident logits (sampler-on-device) on the gather path; HOST logits on
  // the opt-out. Mistral is pure full-attention (input.gdn_* unused).
  if (input.gather_logits) {
    return MistralModel::ForwardDevice(input.token_ids, input.positions,
                                       input.attn_meta, input.attn_kv, weights,
                                       input.config, input.queue,
                                       input.logits_indices);
  }
  return HostLogits(
      MistralModel::Forward(input.token_ids, input.positions, input.attn_meta,
                            input.attn_kv, weights, input.config, input.queue,
                            input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kMistralFactory{
    .parse_config = &ParseMistralForCausalLMConfig,
    .load_weights = &LoadMistralForCausalLM,
    .prepare = &PrepareMistralForCausalLM,
    .forward = &ForwardMistralForCausalLM,
    .make_kv_cache = &MakeMistralForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseMistralForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig already materializes + validates every consumed Mistral field
  // (rope_theta 1e6, the null sliding_window, GQA, derived head_dim). No-op hook
  // (mirrors ParseLlamaForCausalLMConfig) — the family normalization/validation seam.
  (void)config;
}

v1::KVCacheConfig MakeMistralForCausalLMKVCache(const HfConfig& config,
                                                int block_size, int num_blocks) {
  // Pure dense: exactly ONE full-attention KV group, NO MambaSpec/GDN group.
  // sliding_window is null in Mistral-7B-v0.3, so no SlidingWindowSpec group.
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

REGISTER_VLLM_MODEL(mistral_dense, "MistralForCausalLM", kMistralFactory, kMistralInfo)

}  // namespace vllm
