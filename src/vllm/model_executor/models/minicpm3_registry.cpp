// MiniCPM3 (`MiniCPM3ForCausalLM`) registry TU. Self-registers "MiniCPM3ForCausalLM"
// via REGISTER_VLLM_MODEL and owns the arch entry points: the config hook, the
// **MLA** KV-cache spec (one MLAAttentionSpec group — 1 head,
// kv_lora_rank + qk_rope_head_dim wide, NO factor 2 and NO separate V), the
// LoadedModel subclass, and the factory. Mirrors the deepseek_v2_registry.cpp seam
// (new TU + one in-TU REGISTER line -> ZERO shared-array edit).
//
// SCOPE HONESTY: this makes the model LOAD and FORWARD (eager). The SACRED
// token-exact gate against the vLLM 0.25.0 oracle is the paged-engine test; a
// decode CUDA-graph driver (the DeepSeek-V2 W9 sibling) is SPEED-PENDING follow-up.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/minicpm3.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kMiniCPM3Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class MiniCPM3LoadedModel final : public LoadedModel {
 public:
  MiniCPM3LoadedModel(const ModelRegistration& registration, MiniCPM3Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const MiniCPM3Weights& weights() const { return weights_; }

 private:
  MiniCPM3Weights weights_;
};

std::unique_ptr<LoadedModel> LoadMiniCPM3ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture MiniCPM3ForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<MiniCPM3LoadedModel>(
      registration, LoadMiniCPM3ForCausalLMWeights(*source.safetensors, config));
}

void PrepareMiniCPM3ForCausalLM(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardMiniCPM3ForCausalLM(LoadedModel& model,
                                         const ModelForwardInput& input) {
  const auto& m3 = ModelAs<MiniCPM3LoadedModel>(model, "MiniCPM3ForCausalLM");
  const MiniCPM3Weights& weights = m3.weights();
  if (input.gather_logits) {
    return MiniCPM3Model::ForwardDevice(input.token_ids, input.positions,
                                        input.attn_meta, input.attn_kv, weights,
                                        input.queue, input.logits_indices);
  }
  return HostLogits(
      MiniCPM3Model::Forward(input.token_ids, input.positions, input.attn_meta,
                             input.attn_kv, weights, input.queue,
                             input.logits_indices),
      weights.params.vocab_size);
}

const ModelFactory kMiniCPM3Factory{
    .parse_config = &ParseMiniCPM3ForCausalLMConfig,
    .load_weights = &LoadMiniCPM3ForCausalLM,
    .prepare = &PrepareMiniCPM3ForCausalLM,
    .forward = &ForwardMiniCPM3ForCausalLM,
    .make_kv_cache = &MakeMiniCPM3ForCausalLMKVCache,
    // MLA (like DeepSeek-V2) — NOT the direct-device dense-loader / dense-MTP
    // path (which assumes the shared dense weight container). The "dense"
    // skeleton here is the SwiGLU/scalar MLP, but the attention + KV cache are
    // MLA, so this mirrors deepseek_v2's is_dense_model=false.
    .is_dense_model = false,
};

}  // namespace

void ParseMiniCPM3ForCausalLMConfig(const HfConfig& config) {
  // The resolve itself IS the validation: ParseMiniCPM3Params throws precisely on
  // a missing q_lora_rank, a non-longrope rope, or the LongRoPE long-cache regime.
  (void)ParseMiniCPM3Params(config);
}

v1::KVCacheConfig MakeMiniCPM3ForCausalLMKVCache(const HfConfig& config,
                                                 int block_size, int num_blocks) {
  // MLA: exactly ONE attention group carrying an MLAAttentionSpec — 1 head,
  // head_size == kv_lora_rank + qk_rope_head_dim, NO separate V.
  const MiniCPM3Params p = ParseMiniCPM3Params(config);
  const int head_size = static_cast<int>(p.mla.head_size());

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<v1::MLAAttentionSpec>(block_size, head_size,
                                             v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(minicpm3, "MiniCPM3ForCausalLM", kMiniCPM3Factory, kMiniCPM3Info)

}  // namespace vllm
