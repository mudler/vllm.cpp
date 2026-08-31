// Maple (`MapleForCausalLM`) registry TU — mirrors qwen3_moe_registry.cpp's
// shape: ModelInfo, LoadedModel subclass, factory fn-pointers, one
// REGISTER_VLLM_MODEL line, ZERO shared-array edits.
//
// Grounding: HF deepgrove/maple-preview config.json (`architectures[0]` =
// "MapleForCausalLM"); deepgrove-ai/llama.cpp src/models/maple.cpp @ 7e30f3a
// for the arch semantics the config validator pins.
#include "vllm/model_executor/models/maple.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"    // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kMapleInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,          // pure FA MoE — no GDN
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class MapleLoadedModel final : public LoadedModel {
 public:
  MapleLoadedModel(const ModelRegistration& registration, MapleWeights w)
      : LoadedModel(registration), weights_(std::move(w)) {}

  const MapleWeights& weights() const { return weights_; }

 private:
  MapleWeights weights_;
};

std::unique_ptr<LoadedModel> LoadMapleForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  // GGUF-only family: maple-preview ships no safetensors checkpoint; the
  // ternary expert towers are a GGUF block-quant story end to end.
  if (source.kind != ModelSource::Kind::kGguf) {
    throw std::runtime_error(
        "Model architecture MapleForCausalLM requires GGUF weights "
        "(no safetensors release exists)");
  }
  if (source.gguf == nullptr) {
    throw std::runtime_error("gguf model source is empty");
  }
  return std::make_unique<MapleLoadedModel>(
      registration, LoadMapleFromGguf(*source.gguf, config, nullptr));
}

void PrepareMaple(LoadedModel& model, const HfConfig& config, vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardMaple(LoadedModel& model, const ModelForwardInput& input) {
  auto& m = ModelAs<MapleLoadedModel>(model, "MapleForCausalLM");
  // Eager-only (no decode graph): the forward returns host logits through the
  // same HostLogits carrier qwen3_moe's non-gather arm uses; the gather arm
  // passes logits_indices straight through to the model forward.
  return HostLogits(MapleForward(input.token_ids, input.positions,
                                 input.attn_meta, input.attn_kv, m.weights(),
                                 input.config, input.queue,
                                 input.logits_indices),
                    input.config.vocab_size);
}

const ModelFactory kMapleFactory{
    .parse_config = &ParseMapleConfig,
    .load_weights = &LoadMapleForCausalLM,
    .prepare = &PrepareMaple,
    .forward = &ForwardMaple,
    .make_kv_cache = &MakeMapleKVCache,
    .is_dense_model = false,
    .stage_on_load = false,
    .supports_weight_offload = false,
    // REQUIRED: the forward composes RunMoeBlock -> KqExpertSlice slot seam;
    // gates the *_exps.weight device-fit bound (#1124).
    .streams_routed_experts = true,
};

}  // namespace

void ParseMapleConfig(const HfConfig& config) {
  // The MoE fields the loader/forward consume (mirrors ParseQwen3MoeConfig).
  if (config.num_experts <= 0) {
    throw std::runtime_error("MapleForCausalLM config: num_experts must be > 0");
  }
  if (config.num_experts_per_tok <= 0 ||
      config.num_experts_per_tok > config.num_experts) {
    throw std::runtime_error(
        "MapleForCausalLM config: num_experts_per_tok must be in [1, num_experts]");
  }
  if (config.moe_intermediate_size <= 0) {
    throw std::runtime_error(
        "MapleForCausalLM config: moe_intermediate_size must be > 0");
  }
  (void)config.shared_expert_intermediate_size;  // 0 is valid (no shared expert)
  // Layer pattern sanity: layer_types drives BOTH the SWA window and the
  // rope-skip inversion (nope_on_global_attention), so a wrong list would be
  // silently wrong twice.
  if (static_cast<int64_t>(config.layer_types.size()) !=
      config.num_hidden_layers) {
    throw std::runtime_error(
        "MapleForCausalLM config: layer_types size must equal num_hidden_layers");
  }
  const bool has_window = config.sliding_window.has_value() &&
                          *config.sliding_window > 0;
  for (const std::string& lt : config.layer_types) {
    if (lt != "sliding_attention" && lt != "full_attention") {
      throw std::runtime_error("MapleForCausalLM config: unknown layer_type " +
                               lt);
    }
    if (lt == "sliding_attention" && !has_window) {
      throw std::runtime_error(
          "MapleForCausalLM config: sliding_attention layers require sliding_window > 0");
    }
  }
}

v1::KVCacheConfig MakeMapleKVCache(const HfConfig& config, int block_size,
                                   int num_blocks) {
  // Full-attention MoE topology: ONE "fa" KV group, NO MambaSpec/GDN group.
  // The SWA/global split is a WINDOWING concern inside paged attention (the
  // per-layer PagedAttentionArgs.window_size) plus the rope-skip — NOT a
  // second cache group; every layer still owns slots in the one group.
  // Byte-for-byte the MakeQwen3MoeKVCache clone otherwise.
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

REGISTER_VLLM_MODEL(maple, "MapleForCausalLM", kMapleFactory, kMapleInfo)

}  // namespace vllm
