// Laguna-S-2.1 (`LagunaForCausalLM` / `model_type=laguna`) registry TU — the
// ADDITIVE self-registration seam (mirrors deepseek_v4_registry.cpp /
// olmo2_registry.cpp exactly: a NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array). It owns the arch
// entry points: the config hook (ParseLagunaConfig), the KV-cache spec, the
// LoadedModel subclass, and the factory.
//
// Registry routing upstream sends `LagunaForCausalLM` into
// `vllm/model_executor/models/laguna.py`. We register exactly that ONE string.
//
// SCOPE HONESTY (ds4 precedent): registering this arch makes it RESOLVE + parse
// config + build the KV-cache spec. It does NOT make it forward — LagunaModel is
// a W3/W4 stub whose forward VT_CHECK(false, ...)s, and the weight loaders
// VT_CHECK(false, ...) on device materialization (name-map + quant-mix scaffolded
// this increment). A load/forward LOUDLY reports the pending brick — never a
// silent wrong answer. The model-matrix row stays SPIKE/ACTIVE until the strict
// dual-oracle gate (W4) passes on a fetched checkpoint. See
// `.agents/specs/laguna-s21-w1w2-2026-07-30.md`.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/laguna.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Laguna: text generation, NOT hybrid (interleaved
// full + sliding-window attention over a paged full-attention cache), NOT
// multimodal.
inline constexpr ModelInfo kLagunaInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class LagunaLoadedModel final : public LoadedModel {
 public:
  LagunaLoadedModel(const ModelRegistration& registration, LagunaWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const LagunaWeights& weights() const { return weights_; }

 private:
  LagunaWeights weights_;
};

std::unique_ptr<LoadedModel> LoadLagunaForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // The single-GB10 vehicle: unsloth/Laguna-S-2.1-GGUF UD-Q4_K_XL (~73.4 GiB).
    if (source.gguf == nullptr)
      throw std::runtime_error("laguna GGUF model source is empty");
    return std::make_unique<LagunaLoadedModel>(
        registration, LoadLagunaFromGguf(*source.gguf, config));
  }
  if (source.safetensors == nullptr)
    throw std::runtime_error("safetensors model source is empty");
  return std::make_unique<LagunaLoadedModel>(
      registration, LoadLagunaForCausalLMWeights(*source.safetensors, config));
}

void PrepareLagunaForCausalLM(LoadedModel& model, const HfConfig& config,
                              vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardLagunaForCausalLM(LoadedModel& model,
                                       const ModelForwardInput& input) {
  auto& laguna = ModelAs<LagunaLoadedModel>(model, "LagunaForCausalLM");
  const LagunaWeights& weights = laguna.weights();
  if (input.gather_logits) {
    return LagunaModel::ForwardDevice(input.token_ids, input.positions,
                                      input.attn_meta, input.attn_kv, weights,
                                      input.config, input.queue,
                                      input.logits_indices);
  }
  return HostLogits(
      LagunaModel::Forward(input.token_ids, input.positions, input.attn_meta,
                           input.attn_kv, weights, input.config, input.queue,
                           input.logits_indices),
      weights.params.vocab_size);
}

const ModelFactory kLagunaFactory{
    .parse_config = &ParseLagunaConfig,
    .load_weights = &LoadLagunaForCausalLM,
    .prepare = &PrepareLagunaForCausalLM,
    .forward = &ForwardLagunaForCausalLM,
    .make_kv_cache = &MakeLagunaKVCache,
    // MoE model: NOT the dense per-arch scheduler default.
    .is_dense_model = false,
};

}  // namespace

v1::KVCacheConfig MakeLagunaKVCache(const HfConfig& config, int block_size,
                                    int num_blocks) {
  // One FULL-ATTENTION KV group over all layers. The interleaved sliding-window
  // (512) layers are masked at the attention kernel (per-layer window), NOT by a
  // smaller SlidingWindowSpec cache — the gemma3 topology the shape-agnostic
  // runner already handles (gemma3_registry.cpp:103-121). KV heads 8, head_dim
  // 128 are uniform across layers (only the Q-head COUNT varies per layer, which
  // does not affect the KV cache spec).
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

REGISTER_VLLM_MODEL(laguna, "LagunaForCausalLM", kLagunaFactory, kLagunaInfo)

}  // namespace vllm
