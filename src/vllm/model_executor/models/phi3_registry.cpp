// Phi-3 / Phi-4 dense (`Phi3ForCausalLM`) registry TU — the ZERO-NEW-KERNEL dense
// bring-up (Llama forward + pre-fused loader + LongRoPE cache). Self-registers
// "Phi3ForCausalLM" via REGISTER_VLLM_MODEL and owns the arch entry points. Mirrors
// the mistral_registry.cpp seam (new TU + one in-TU REGISTER line -> ZERO
// shared-array edit). See .agents/specs/sweep-recent-dense-batch.md §0.2 row 1.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/phi3.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kPhi3Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class Phi3LoadedModel final : public LoadedModel {
 public:
  Phi3LoadedModel(const ModelRegistration& registration, Phi3Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Phi3Weights& weights() const { return weights_; }

 private:
  Phi3Weights weights_;
};

std::unique_ptr<LoadedModel> LoadPhi3ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Phi3ForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Phi3LoadedModel>(
      registration, LoadPhi3ForCausalLMWeights(*source.safetensors, config));
}

void PreparePhi3ForCausalLM(LoadedModel& model, const HfConfig& config,
                            vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardPhi3ForCausalLM(LoadedModel& model,
                                     const ModelForwardInput& input) {
  const auto& phi3 = static_cast<Phi3LoadedModel&>(model);
  const Phi3Weights& weights = phi3.weights();
  if (input.gather_logits) {
    return Phi3Model::ForwardDevice(input.token_ids, input.positions,
                                    input.attn_meta, input.attn_kv, weights,
                                    input.config, input.queue,
                                    input.logits_indices);
  }
  return HostLogits(
      Phi3Model::Forward(input.token_ids, input.positions, input.attn_meta,
                         input.attn_kv, weights, input.config, input.queue,
                         input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kPhi3Factory{
    .parse_config = &ParsePhi3ForCausalLMConfig,
    .load_weights = &LoadPhi3ForCausalLM,
    .prepare = &PreparePhi3ForCausalLM,
    .forward = &ForwardPhi3ForCausalLM,
    .make_kv_cache = &MakePhi3ForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParsePhi3ForCausalLMConfig(const HfConfig& config) {
  VT_CHECK(config.rotary_dim > 0 && config.rotary_dim <= config.head_dim,
           "phi3: rotary_dim must be in (0, head_dim]");
  VT_CHECK(config.rope_parameters.rope_type == "default" ||
               config.rope_parameters.rope_type == "longrope",
           "phi3: expected default or longrope rope");
}

v1::KVCacheConfig MakePhi3ForCausalLMKVCache(const HfConfig& config, int block_size,
                                             int num_blocks) {
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

REGISTER_VLLM_MODEL(phi3, "Phi3ForCausalLM", kPhi3Factory, kPhi3Info)

}  // namespace vllm
