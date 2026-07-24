// Gemma-2 text (`Gemma2ForCausalLM`) registry TU — sweep W4. Self-registers
// "Gemma2ForCausalLM" via REGISTER_VLLM_MODEL and owns the arch entry points: the
// config hook, the full-attention-only KV-cache spec (sliding layers masked at
// the kernel), the LoadedModel subclass and the factory. Mirrors the
// gemma3_registry.cpp seam (new TU + one in-TU REGISTER line -> ZERO shared-array
// edit). See .agents/specs/sweep-gemma.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/gemma2.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Gemma-2 text: text generation, NOT hybrid (dense
// full-attention + interleaved sliding), NOT multimodal (gemma-2-2b-it pure text).
inline constexpr ModelInfo kGemma2Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class Gemma2LoadedModel final : public LoadedModel {
 public:
  Gemma2LoadedModel(const ModelRegistration& registration, Gemma2Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Gemma2Weights& weights() const { return weights_; }

 private:
  Gemma2Weights weights_;
};

std::unique_ptr<LoadedModel> LoadGemma2ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Gemma2ForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Gemma2LoadedModel>(
      registration, LoadGemma2ForCausalLMWeights(*source.safetensors, config));
}

void PrepareGemma2ForCausalLM(LoadedModel& model, const HfConfig& config,
                              vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGemma2ForCausalLM(LoadedModel& model,
                                       const ModelForwardInput& input) {
  const auto& gemma = static_cast<Gemma2LoadedModel&>(model);
  const Gemma2Weights& weights = gemma.weights();
  if (input.gather_logits) {
    return Gemma2Model::ForwardDevice(input.token_ids, input.positions,
                                      input.attn_meta, input.attn_kv, weights,
                                      input.config, input.queue,
                                      input.logits_indices);
  }
  return HostLogits(
      Gemma2Model::Forward(input.token_ids, input.positions, input.attn_meta,
                           input.attn_kv, weights, input.config, input.queue,
                           input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kGemma2Factory{
    .parse_config = &ParseGemma2ForCausalLMConfig,
    .load_weights = &LoadGemma2ForCausalLM,
    .prepare = &PrepareGemma2ForCausalLM,
    .forward = &ForwardGemma2ForCausalLM,
    .make_kv_cache = &MakeGemma2ForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseGemma2ForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig materializes the typed fields; the Gemma-2 scalars
  // (query_pre_attn_scalar, attn_logit_softcapping, final_logit_softcapping,
  // layer_types) are read from config.raw by the forward. No-op hook.
  (void)config;
}

v1::KVCacheConfig MakeGemma2ForCausalLMKVCache(const HfConfig& config,
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

REGISTER_VLLM_MODEL(gemma2, "Gemma2ForCausalLM", kGemma2Factory, kGemma2Info)

}  // namespace vllm
