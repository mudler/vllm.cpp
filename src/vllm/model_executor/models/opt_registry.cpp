// OPT (`OPTForCausalLM`) registry TU — the CROSS-FAMILY additivity canary, W0.
// Self-registers "OPTForCausalLM" via REGISTER_VLLM_MODEL and owns the
// arch-specific entry points: the config hook (which, unlike the Qwen no-op
// hooks, actually parses OPT's family-specific fields out of `HfConfig::raw`),
// the full-attention-ONLY KV-cache spec, the LoadedModel subclass, and the
// factory. Mirrors the qwen3_dense.cpp seam: a new TU + one in-TU REGISTER line
// => ZERO shared-array edit.
//
// Upstream: vllm/model_executor/models/opt.py::OPTForCausalLM @ e24d1b24,
// registered at registry.py:176 as ("opt", "OPTForCausalLM").
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/opt.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for OPT: text generation, NOT hybrid (pure full
// attention), NOT multimodal. (Upstream also declares SupportsLoRA/SupportsPP,
// opt.py:327 — neither is part of our consumed ModelInfo subset.)
inline constexpr ModelInfo kOPTInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// Raw-config readers. OPT keeps its family fields at the TOP level of
// config.json (no text_config nesting), so these read `config.raw` directly —
// the seam that lets a new family add config surface without widening the
// shared HfConfig POD.
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number_integer()) return fallback;
  return it->get<int64_t>();
}

std::string RawStr(const nlohmann::json& doc, const char* key, const char* fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_string()) return fallback;
  return it->get<std::string>();
}

class OPTLoadedModel final : public LoadedModel {
 public:
  OPTLoadedModel(const ModelRegistration& registration, OPTWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const OPTWeights& weights() const { return weights_; }

 private:
  OPTWeights weights_;
};

std::unique_ptr<LoadedModel> LoadOPT(const ModelRegistration& registration,
                                     const HfConfig& config, const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture OPTForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<OPTLoadedModel>(
      registration, LoadOPTForCausalLMWeights(*source.safetensors, config));
}

void PrepareOPT(LoadedModel& model, const HfConfig& config, vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardOPT(LoadedModel& model, const ModelForwardInput& input) {
  const auto& opt = static_cast<OPTLoadedModel&>(model);
  const OPTWeights& weights = opt.weights();
  // DEVICE-resident logits (sampler-on-device) on the gather path; HOST logits
  // on the opt-out. OPT is pure full-attention (input.gdn_* unused).
  if (input.gather_logits) {
    return OPTModel::ForwardDevice(input.token_ids, input.positions, input.attn_meta,
                                   input.attn_kv, weights, input.config, input.queue,
                                   input.logits_indices);
  }
  return HostLogits(OPTModel::Forward(input.token_ids, input.positions, input.attn_meta,
                                      input.attn_kv, weights, input.config, input.queue,
                                      input.logits_indices),
                    input.config.vocab_size);
}

const ModelFactory kOPTFactory{
    .parse_config = &ParseOPTConfig,
    .load_weights = &LoadOPT,
    .prepare = &PrepareOPT,
    .forward = &ForwardOPT,
    .make_kv_cache = &MakeOPTKVCache,
    .is_dense_model = true,
};

}  // namespace

OPTConfigExtras GetOPTConfigExtras(const HfConfig& config) {
  const nlohmann::json& doc = config.raw;
  OPTConfigExtras e;
  // OPT names its FFN width `ffn_dim`, NOT `intermediate_size` — our typed
  // HfConfig therefore leaves `intermediate_size` at 0 for an OPT checkpoint,
  // which is exactly why this hook exists.
  e.ffn_dim = RawInt(doc, "ffn_dim", 0);
  e.word_embed_proj_dim = RawInt(doc, "word_embed_proj_dim", config.hidden_size);
  // Defaults mirror transformers `OPTConfig` for keys the released checkpoints
  // omit (facebook/opt-125m omits all four).
  e.do_layer_norm_before = RawBool(doc, "do_layer_norm_before", true);
  e.enable_bias = RawBool(doc, "enable_bias", true);
  e.layer_norm_elementwise_affine = RawBool(doc, "layer_norm_elementwise_affine", true);
  e.remove_final_layer_norm = RawBool(doc, "_remove_final_layer_norm", false);
  e.tie_word_embeddings = RawBool(doc, "tie_word_embeddings", true);
  e.activation_function = RawStr(doc, "activation_function", "relu");
  return e;
}

void ParseOPTConfig(const HfConfig& config) {
  const OPTConfigExtras e = GetOPTConfigExtras(config);
  VT_CHECK(e.ffn_dim > 0, "opt config: ffn_dim must be present and positive");
  VT_CHECK(config.num_attention_heads > 0 && config.hidden_size > 0,
           "opt config: num_attention_heads / hidden_size must be positive");
  VT_CHECK(config.hidden_size % config.num_attention_heads == 0,
           "opt config: hidden_size must be divisible by num_attention_heads");
  VT_CHECK(config.max_position_embeddings > 0,
           "opt config: max_position_embeddings must be positive "
           "(OPT uses LEARNED absolute position embeddings)");
  // Reject the two variants we have no checkpoint on the box to gate, rather
  // than shipping untested paths (spec R2). Both are implemented upstream and
  // both are cheap to add once a checkpoint exists.
  VT_CHECK(e.word_embed_proj_dim == config.hidden_size,
           "opt config: word_embed_proj_dim != hidden_size (project_in/project_out, "
           "opt.py:220-241) is not supported yet — no checkpoint available to gate it");
  VT_CHECK(e.activation_function == "relu",
           "opt config: only activation_function=relu is supported "
           "(every released OPT checkpoint uses relu; opt.py:156)");
}

v1::KVCacheConfig MakeOPTKVCache(const HfConfig& config, int block_size, int num_blocks) {
  // Pure dense full attention: exactly ONE KV group, NO MambaSpec/GDN group.
  // OPT predates GQA, so num_key_value_heads == num_attention_heads (HfConfig
  // already defaults it that way when the key is absent, which it is for OPT).
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, num_kv_heads, head_dim,
                                              vt::DType::kF32));
  return kv;
}

REGISTER_VLLM_MODEL(opt, "OPTForCausalLM", kOPTFactory, kOPTInfo)

}  // namespace vllm
