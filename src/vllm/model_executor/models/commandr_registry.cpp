// Command-R / Cohere (`CohereForCausalLM`) registry TU — the ZERO-NEW-KERNEL dense
// bring-up (weight-only Cohere LayerNorm + GPT-J full-width RoPE + PARALLEL residual
// + logit_scale + tied embeddings, all REUSE). Self-registers "CohereForCausalLM"
// via REGISTER_VLLM_MODEL and owns the arch entry points + config helpers. Mirrors
// the stablelm_registry.cpp seam (new TU + one in-TU REGISTER line -> ZERO shared-
// array edit). See .agents/specs/sweep-recent-dense-batch.md §0.2 row 6.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/commandr.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kCommandrInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class CommandrLoadedModel final : public LoadedModel {
 public:
  CommandrLoadedModel(const ModelRegistration& registration, CommandrWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const CommandrWeights& weights() const { return weights_; }

 private:
  CommandrWeights weights_;
};

std::unique_ptr<LoadedModel> LoadCohereForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture CohereForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<CommandrLoadedModel>(
      registration, LoadCohereForCausalLMWeights(*source.safetensors, config));
}

void PrepareCohereForCausalLM(LoadedModel& model, const HfConfig& config,
                              vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardCohereForCausalLM(LoadedModel& model,
                                       const ModelForwardInput& input) {
  const auto& cm = ModelAs<CommandrLoadedModel>(model, "CohereForCausalLM");
  const CommandrWeights& weights = cm.weights();
  if (input.gather_logits) {
    return CommandrModel::ForwardDevice(input.token_ids, input.positions,
                                        input.attn_meta, input.attn_kv, weights,
                                        input.config, input.queue,
                                        input.logits_indices);
  }
  return HostLogits(
      CommandrModel::Forward(input.token_ids, input.positions, input.attn_meta,
                             input.attn_kv, weights, input.config, input.queue,
                             input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kCommandrFactory{
    .parse_config = &ParseCohereForCausalLMConfig,
    .load_weights = &LoadCohereForCausalLM,
    .prepare = &PrepareCohereForCausalLM,
    .forward = &ForwardCohereForCausalLM,
    .make_kv_cache = &MakeCohereForCausalLMKVCache,
    .is_dense_model = true,
};

// Read a scalar double from the raw config; returns fallback if absent/non-number.
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return fallback;
  return it->get<double>();
}

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

}  // namespace

// Cohere LayerNorm eps: config.layer_norm_eps (commandr.py:301, default 1e-5).
float CommandrLayerNormEps(const HfConfig& config) {
  return static_cast<float>(RawDouble(config.raw, "layer_norm_eps", 1e-5));
}

// logit_scale scalar (commandr.py:376). Default 1.0 when absent.
double CommandrLogitScale(const HfConfig& config) {
  return RawDouble(config.raw, "logit_scale", 1.0);
}

void ParseCohereForCausalLMConfig(const HfConfig& config) {
  VT_CHECK(config.head_dim > 0, "commandr: head_dim must be positive");
  VT_CHECK(config.num_attention_heads > 0 && config.num_key_value_heads > 0,
           "commandr: head counts must be positive");
  VT_CHECK(config.num_attention_heads % config.num_key_value_heads == 0,
           "commandr: num_attention_heads must be divisible by num_key_value_heads");
  VT_CHECK(config.rope_parameters.rope_type == "default",
           "commandr: expected default rope");
  // Command-R (CohereForCausalLM) is always tied (commandr.py:372 asserts it).
  VT_CHECK(RawBool(config.raw, "tie_word_embeddings", true),
           "commandr: CohereForCausalLM requires tie_word_embeddings=True");
  // qk-norm + interleaved sliding window are the Cohere2ForCausalLM arch (a
  // separate, newer model) — reject rather than silently mis-run.
  VT_CHECK(!RawBool(config.raw, "use_qk_norm", false),
           "commandr: use_qk_norm (Cohere2ForCausalLM) is not supported by the "
           "CohereForCausalLM path");
  const auto sw = config.raw.find("sliding_window");
  VT_CHECK(sw == config.raw.end() || sw->is_null(),
           "commandr: sliding_window (Cohere2ForCausalLM) is not supported by the "
           "CohereForCausalLM path");
}

v1::KVCacheConfig MakeCohereForCausalLMKVCache(const HfConfig& config,
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

REGISTER_VLLM_MODEL(commandr, "CohereForCausalLM", kCommandrFactory, kCommandrInfo)

}  // namespace vllm
