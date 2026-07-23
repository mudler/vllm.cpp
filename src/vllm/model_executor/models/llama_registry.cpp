// Llama-3.x (`LlamaForCausalLM`) registry TU — the cross-family dense additive
// bring-up. Self-registers "LlamaForCausalLM" via REGISTER_VLLM_MODEL and owns the
// arch-specific entry points: the config hook, the full-attention-ONLY KV-cache
// spec, the LoadedModel subclass, and the factory. Mirrors the qwen3_dense.cpp
// seam (new TU + one in-TU REGISTER line -> ZERO shared-array edit).
//
// The forward + KV topology are REUSED VERBATIM from the Qwen3-dense path
// (LlamaModel == Qwen3DenseModel): Llama is that forward with qk-norm skipped and
// llama3 rope-scaling applied, both handled additively inside the shared
// dense_attn_block.h AttnBlock. So this TU only wires the loader + forward hooks.
// See .agents/specs/sweep-llama-3.2.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/llama.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Llama: text generation, NOT hybrid (pure
// full-attention), NOT multimodal (text-only checkpoint).
inline constexpr ModelInfo kLlamaInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: holds the loaded Llama dense weights (shared dense
// container); the forward reuses the Qwen3-dense forward machinery.
class LlamaLoadedModel final : public LoadedModel {
 public:
  LlamaLoadedModel(const ModelRegistration& registration, LlamaWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const LlamaWeights& weights() const { return weights_; }

 private:
  LlamaWeights weights_;
};

std::unique_ptr<LoadedModel> LoadLlamaForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  // Llama dense is text-only BF16 safetensors (no GGUF path yet).
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture LlamaForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<LlamaLoadedModel>(
      registration, LoadLlamaForCausalLMWeights(*source.safetensors, config));
}

void PrepareLlamaForCausalLM(LoadedModel& model, const HfConfig& config,
                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardLlamaForCausalLM(LoadedModel& model,
                                      const ModelForwardInput& input) {
  const auto& llama = static_cast<LlamaLoadedModel&>(model);
  const LlamaWeights& weights = llama.weights();
  // DEVICE-resident logits (sampler-on-device) on the gather path; HOST logits on
  // the opt-out. Llama is pure full-attention (input.gdn_* unused).
  if (input.gather_logits) {
    return LlamaModel::ForwardDevice(input.token_ids, input.positions,
                                     input.attn_meta, input.attn_kv, weights,
                                     input.config, input.queue,
                                     input.logits_indices);
  }
  return HostLogits(
      LlamaModel::Forward(input.token_ids, input.positions, input.attn_meta,
                          input.attn_kv, weights, input.config, input.queue,
                          input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kLlamaFactory{
    .parse_config = &ParseLlamaForCausalLMConfig,
    .load_weights = &LoadLlamaForCausalLM,
    .prepare = &PrepareLlamaForCausalLM,
    .forward = &ForwardLlamaForCausalLM,
    .make_kv_cache = &MakeLlamaForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseLlamaForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig already materializes + validates every consumed Llama field
  // (including the llama3 rope_scaling dictionary). No-op hook (mirrors
  // ParseQwen3ForCausalLMConfig) — the family normalization/validation seam.
  (void)config;
}

v1::KVCacheConfig MakeLlamaForCausalLMKVCache(const HfConfig& config,
                                              int block_size, int num_blocks) {
  // Pure dense: exactly ONE full-attention KV group, NO MambaSpec/GDN group.
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

REGISTER_VLLM_MODEL(llama_dense, "LlamaForCausalLM", kLlamaFactory, kLlamaInfo)

}  // namespace vllm
