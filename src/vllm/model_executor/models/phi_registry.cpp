// Phi-1/Phi-2 (`PhiForCausalLM`, microsoft/phi-2) registry TU — the ZERO-NEW-KERNEL
// dense bring-up (parallel-residual nn.LayerNorm block + biased q/k/v/dense +
// partial NeoX RoPE + NON-gated NewGELU MLP + untied biased lm_head, all REUSE).
// Self-registers "PhiForCausalLM" via REGISTER_VLLM_MODEL and owns the arch entry
// points + config helpers. Mirrors the stablelm_registry.cpp seam (new TU + one
// in-TU REGISTER line -> ZERO shared-array edit). This is the OLDER Phi arch,
// DISTINCT from the already-registered "Phi3ForCausalLM" (phi3_registry.cpp). See
// .agents/specs/sweep-recent-dense-batch.md §0.2 row 7.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/phi.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kPhiInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class PhiLoadedModel final : public LoadedModel {
 public:
  PhiLoadedModel(const ModelRegistration& registration, PhiWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const PhiWeights& weights() const { return weights_; }

 private:
  PhiWeights weights_;
};

std::unique_ptr<LoadedModel> LoadPhiForCausalLM(const ModelRegistration& registration,
                                                const HfConfig& config,
                                                const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture PhiForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<PhiLoadedModel>(
      registration, LoadPhiForCausalLMWeights(*source.safetensors, config));
}

void PreparePhiForCausalLM(LoadedModel& model, const HfConfig& config,
                           vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardPhiForCausalLM(LoadedModel& model,
                                    const ModelForwardInput& input) {
  const auto& pm = ModelAs<PhiLoadedModel>(model, "PhiForCausalLM");
  const PhiWeights& weights = pm.weights();
  if (input.gather_logits) {
    return PhiModel::ForwardDevice(input.token_ids, input.positions, input.attn_meta,
                                   input.attn_kv, weights, input.config, input.queue,
                                   input.logits_indices);
  }
  return HostLogits(
      PhiModel::Forward(input.token_ids, input.positions, input.attn_meta,
                        input.attn_kv, weights, input.config, input.queue,
                        input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kPhiFactory{
    .parse_config = &ParsePhiForCausalLMConfig,
    .load_weights = &LoadPhiForCausalLM,
    .prepare = &PreparePhiForCausalLM,
    .forward = &ForwardPhiForCausalLM,
    .make_kv_cache = &MakePhiForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

// nn.LayerNorm eps: config.layer_norm_eps (phi.py:182,225). microsoft/phi-2 sets
// layer_norm_eps 1e-5.
float PhiLayerNormEps(const HfConfig& config) {
  const nlohmann::json& doc = config.raw;
  const auto it = doc.find("layer_norm_eps");
  if (it != doc.end() && !it->is_null() && it->is_number())
    return static_cast<float>(it->get<double>());
  return 1e-5F;
}

// The Phi MLP inner width: `n_inner` when present and non-null, else
// 4 * hidden_size (phi.py:148-149). microsoft/phi-2 leaves n_inner null -> 10240.
int64_t PhiFfnDim(const HfConfig& config) {
  const nlohmann::json& doc = config.raw;
  const auto it = doc.find("n_inner");
  if (it != doc.end() && !it->is_null() && it->is_number_integer())
    return it->get<int64_t>();
  return 4 * config.hidden_size;
}

void ParsePhiForCausalLMConfig(const HfConfig& config) {
  VT_CHECK(config.rotary_dim > 0 && config.rotary_dim <= config.head_dim,
           "phi: rotary_dim must be in (0, head_dim]");
  VT_CHECK(config.rope_parameters.rope_type == "default",
           "phi: expected default rope");
  VT_CHECK(config.num_attention_heads > 0 && config.num_key_value_heads > 0,
           "phi: head counts must be positive");
  VT_CHECK(config.num_attention_heads % config.num_key_value_heads == 0,
           "phi: num_attention_heads must be divisible by num_key_value_heads");
  const auto tie = config.raw.find("tie_word_embeddings");
  const bool tied =
      tie != config.raw.end() && tie->is_boolean() && tie->get<bool>();
  VT_CHECK(!tied,
           "phi: PhiForCausalLM requires untied embeddings (lm_head has a bias)");
}

v1::KVCacheConfig MakePhiForCausalLMKVCache(const HfConfig& config, int block_size,
                                            int num_blocks) {
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

REGISTER_VLLM_MODEL(phi, "PhiForCausalLM", kPhiFactory, kPhiInfo)

}  // namespace vllm
