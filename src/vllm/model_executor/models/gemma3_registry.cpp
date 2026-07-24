// Gemma-3 text (`Gemma3ForCausalLM`) registry TU — sweep W2. Self-registers
// "Gemma3ForCausalLM" via REGISTER_VLLM_MODEL and owns the arch entry points: the
// config hook, the full-attention-only KV-cache spec (one FA group; sliding
// layers are masked at the kernel, not by a smaller cache), the LoadedModel
// subclass and the factory. Mirrors the qwen3_dense.cpp seam (new TU + one in-TU
// REGISTER line -> ZERO shared-array edit). See .agents/specs/sweep-gemma.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/gemma3.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Gemma-3 text: text generation, NOT hybrid (dense
// full-attention + interleaved sliding, no linear-attention group), NOT
// multimodal (gemma-3-1b-it is pure text).
inline constexpr ModelInfo kGemma3Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class Gemma3LoadedModel final : public LoadedModel {
 public:
  Gemma3LoadedModel(const ModelRegistration& registration, Gemma3Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Gemma3Weights& weights() const { return weights_; }

 private:
  Gemma3Weights weights_;
};

std::unique_ptr<LoadedModel> LoadGemma3ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Gemma3ForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Gemma3LoadedModel>(
      registration, LoadGemma3ForCausalLMWeights(*source.safetensors, config));
}

void PrepareGemma3ForCausalLM(LoadedModel& model, const HfConfig& config,
                              vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGemma3ForCausalLM(LoadedModel& model,
                                       const ModelForwardInput& input) {
  const auto& gemma = static_cast<Gemma3LoadedModel&>(model);
  const Gemma3Weights& weights = gemma.weights();
  if (input.gather_logits) {
    return Gemma3Model::ForwardDevice(input.token_ids, input.positions,
                                      input.attn_meta, input.attn_kv, weights,
                                      input.config, input.queue,
                                      input.logits_indices);
  }
  return HostLogits(
      Gemma3Model::Forward(input.token_ids, input.positions, input.attn_meta,
                           input.attn_kv, weights, input.config, input.queue,
                           input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kGemma3Factory{
    .parse_config = &ParseGemma3ForCausalLMConfig,
    .load_weights = &LoadGemma3ForCausalLM,
    .prepare = &PrepareGemma3ForCausalLM,
    .forward = &ForwardGemma3ForCausalLM,
    .make_kv_cache = &MakeGemma3ForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseGemma3ForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig materializes the typed fields; the Gemma-specific scalars
  // (query_pre_attn_scalar, rope_local_base_freq, sliding_window_pattern) are
  // read from config.raw by the forward. No-op hook (mirrors ParseQwen3).
  (void)config;
}

v1::KVCacheConfig MakeGemma3ForCausalLMKVCache(const HfConfig& config,
                                               int block_size, int num_blocks) {
  // One FULL-ATTENTION KV group over all layers. The interleaved sliding-window
  // layers are masked at the attention kernel (per-layer window_size), NOT by a
  // smaller SlidingWindowSpec cache — a memory-only vLLM optimization not needed
  // for correctness. This is the pure-dense full-attention topology the
  // shape-agnostic runner (ENG-RUNNER-MODELSHAPE) already handles.
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

REGISTER_VLLM_MODEL(gemma3, "Gemma3ForCausalLM", kGemma3Factory, kGemma3Info)

}  // namespace vllm
