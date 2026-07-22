// DeepSeek-V2 (`DeepseekV2ForCausalLM`) registry TU — MLA campaign W7. Follows
// the qwen3_moe_registry.cpp seam exactly: a NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array. It owns the
// arch-specific entry points: the config hook, the **MLA** KV-cache spec (one
// MLAAttentionSpec group — 1 head, `kv_lora_rank + qk_rope_head_dim` wide, NO
// factor 2 and NO separate V), the LoadedModel subclass, and the factory.
//
// Registry routing upstream (`registry.py:90-93`) sends FOUR architecture
// strings into `deepseek_v2`; we register exactly ONE and say why in
// deepseek_v2.h ("WHAT IS DELIBERATELY NOT REGISTERED") — in particular
// `DeepseekForCausalLM` is plain MHA (`deepseek_v2.py:1201-1211`), not MLA, so
// registering it here would be a false support claim.
//
// SCOPE HONESTY: W7 makes this model LOAD and FORWARD. The SACRED token-exact
// gate against the vLLM 0.25.0 oracle is W8; until it passes, the model-matrix
// row does NOT move to a supported state.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for DeepSeek-V2: text generation, NOT hybrid (there is
// no recurrent/linear-attention group — MLA is still full attention over a
// paged cache), NOT multimodal.
inline constexpr ModelInfo kDeepseekV2Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class DeepseekV2LoadedModel final : public LoadedModel {
 public:
  DeepseekV2LoadedModel(const ModelRegistration& registration,
                        DeepseekV2Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const DeepseekV2Weights& weights() const { return weights_; }

 private:
  DeepseekV2Weights weights_;
};

std::unique_ptr<LoadedModel> LoadDeepseekV2ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture DeepseekV2ForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<DeepseekV2LoadedModel>(
      registration, LoadDeepseekV2ForCausalLMWeights(*source.safetensors, config));
}

void PrepareDeepseekV2ForCausalLM(LoadedModel& model, const HfConfig& config,
                                  vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardDeepseekV2ForCausalLM(LoadedModel& model,
                                           const ModelForwardInput& input) {
  auto& ds = static_cast<DeepseekV2LoadedModel&>(model);
  const DeepseekV2Weights& weights = ds.weights();
  // No decode CUDA-graph sibling yet (W9); MLA is a pure full-attention arch, so
  // input.gdn_* are unused.
  if (input.gather_logits) {
    return DeepseekV2Model::ForwardDevice(input.token_ids, input.positions,
                                          input.attn_meta, input.attn_kv, weights,
                                          input.queue, input.logits_indices);
  }
  return HostLogits(
      DeepseekV2Model::Forward(input.token_ids, input.positions, input.attn_meta,
                               input.attn_kv, weights, input.queue,
                               input.logits_indices),
      weights.params.vocab_size);
}

const ModelFactory kDeepseekV2Factory{
    .parse_config = &ParseDeepseekV2Config,
    .load_weights = &LoadDeepseekV2ForCausalLM,
    .prepare = &PrepareDeepseekV2ForCausalLM,
    .forward = &ForwardDeepseekV2ForCausalLM,
    .make_kv_cache = &MakeDeepseekV2KVCache,
    .is_dense_model = false,
};

}  // namespace

void ParseDeepseekV2Config(const HfConfig& config) {
  // The resolve itself IS the validation: ParseDeepseekV2Params throws with a
  // precise message on every field this port cannot serve (the use_mha branch,
  // a non-YaRN scaled rope, an unsupported scoring_func, MTP draft layers, the
  // V3.2 indexer, a quantized checkpoint).
  (void)ParseDeepseekV2Params(config);
}

v1::KVCacheConfig MakeDeepseekV2KVCache(const HfConfig& config, int block_size,
                                        int num_blocks) {
  // MLA: exactly ONE attention group carrying an `MLAAttentionSpec`. The cache
  // stores ONE latent vector plus the decoupled rope part per token —
  // `num_kv_heads == 1`, `head_size == kv_lora_rank + qk_rope_head_dim`, and NO
  // separate V (mla_attention.py:387 + config/model.py:1270-1274). The runner's
  // spec-driven allocation (MLA campaign W1) reads the page cost straight off
  // this spec, so the factor 2 simply never appears.
  const DeepseekV2Params p = ParseDeepseekV2Params(config);
  const int head_size = static_cast<int>(p.mla.head_size());

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<v1::MLAAttentionSpec>(block_size, head_size,
                                             v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(deepseek_v2, "DeepseekV2ForCausalLM", kDeepseekV2Factory,
                    kDeepseekV2Info)

}  // namespace vllm
