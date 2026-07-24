// OLMo-2 dense (`Olmo2ForCausalLM`; alias `Olmo3ForCausalLM`) registry TU — the
// cleanest dense bring-up yet (ZERO new compute kernels). Self-registers BOTH arch
// strings via REGISTER_VLLM_MODEL and owns the arch entry points: config hook,
// full-attention-ONLY KV-cache spec (one FA group, NO MambaSpec/GDN group), the
// LoadedModel subclass, and the factory. Mirrors the glm4_registry.cpp seam (new
// TU + in-TU REGISTER lines -> ZERO shared-array edit). Olmo3ForCausalLM maps to
// the SAME class in vLLM (registry.py:172-173, config-gated Olmo2Config vs
// Olmo3Config); the dense forward covers it structurally — the Olmo-3 interleaved
// sliding-window variant is a config-driven W5 follow-on, out of scope here.
// See .agents/specs/sweep-olmo2.md §Port map.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/olmo2.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for OLMo-2/3 dense: text generation, NOT hybrid (pure
// full-attention), NOT multimodal.
inline constexpr ModelInfo kOlmo2Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class Olmo2LoadedModel final : public LoadedModel {
 public:
  Olmo2LoadedModel(const ModelRegistration& registration, Olmo2Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const Olmo2Weights& weights() const { return weights_; }

 private:
  Olmo2Weights weights_;
};

std::unique_ptr<LoadedModel> LoadOlmo2ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Olmo2ForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Olmo2LoadedModel>(
      registration, LoadOlmo2ForCausalLMWeights(*source.safetensors, config));
}

void PrepareOlmo2ForCausalLM(LoadedModel& model, const HfConfig& config,
                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardOlmo2ForCausalLM(LoadedModel& model,
                                      const ModelForwardInput& input) {
  const auto& olmo = static_cast<Olmo2LoadedModel&>(model);
  const Olmo2Weights& weights = olmo.weights();
  if (input.gather_logits) {
    return Olmo2Model::ForwardDevice(input.token_ids, input.positions,
                                     input.attn_meta, input.attn_kv, weights,
                                     input.config, input.queue,
                                     input.logits_indices);
  }
  return HostLogits(
      Olmo2Model::Forward(input.token_ids, input.positions, input.attn_meta,
                          input.attn_kv, weights, input.config, input.queue,
                          input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kOlmo2Factory{
    .parse_config = &ParseOlmo2ForCausalLMConfig,
    .load_weights = &LoadOlmo2ForCausalLM,
    .prepare = &PrepareOlmo2ForCausalLM,
    .forward = &ForwardOlmo2ForCausalLM,
    .make_kv_cache = &MakeOlmo2ForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseOlmo2ForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig already materializes the consumed OLMo-2 fields (num_key_value_
  // heads, head_dim, rope_theta, rotary_dim, intermediate_size, rms_norm_eps).
  // Validate the OLMo-2 invariant: FULL (non-partial) NeoX rope over the whole
  // head_dim — OLMo-2/3 do not use partial rotary.
  VT_CHECK(config.rotary_dim == config.head_dim,
           "olmo2: expected full NeoX rope (rotary_dim == head_dim)");
}

v1::KVCacheConfig MakeOlmo2ForCausalLMKVCache(const HfConfig& config, int block_size,
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

// Both arch strings map to the SAME factory (registry.py:172-173). Two REGISTER
// lines (distinct unique_tags) — still ZERO shared-array edit.
REGISTER_VLLM_MODEL(olmo2, "Olmo2ForCausalLM", kOlmo2Factory, kOlmo2Info)
REGISTER_VLLM_MODEL(olmo3, "Olmo3ForCausalLM", kOlmo2Factory, kOlmo2Info)

}  // namespace vllm
