// Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`) registry TU — the first
// full-attention MoE bring-up (W0). Self-registers "Qwen3MoeForCausalLM" via
// REGISTER_VLLM_MODEL and owns the arch-specific entry points: the config hook,
// the full-attention-ONLY KV-cache spec (one FA group, NO MambaSpec/GDN group —
// cloned from qwen3_dense.cpp's MakeQwen3ForCausalLMKVCache), the LoadedModel
// subclass, and the factory. Mirrors the qwen3_dense.cpp seam (new TU + one in-TU
// REGISTER line -> ZERO shared-array edit).
//
// W0/W1 scope: the weight loader (W2) and the forward (W3) are NOT implemented —
// the registered load_weights + forward hooks are clear-throwing stubs so the
// registry resolves and the KV-cache/runner generalization can be gated on CPU,
// while any attempt to actually load/run the model fails loudly with a "W2"/"W3"
// message rather than silently. The reusable dense `AttnBlock`
// (dense_attn_block.h) + the exposed `RunMoeBlock` (qwen3_5_moe_block.h) + the
// no-shared-expert guard land in W1 as behavior-preserving refactors that W3 then
// composes. See .agents/specs/sweep-qwen3-coder-30b.md §3, §7.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"    // HostLogits (W3)
#include "vllm/model_executor/models/qwen3_moe.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Qwen3-Coder: text generation, NOT hybrid (pure
// full-attention MoE — no GDN), NOT multimodal (text-only, no vision tower).
inline constexpr ModelInfo kQwen3MoeInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// Opaque owned model. W2: holds the loaded Qwen3-Coder MoE weights; the forward
// is not yet wired (throws until W3).
class Qwen3MoeLoadedModel final : public LoadedModel {
 public:
  Qwen3MoeLoadedModel(const ModelRegistration& registration,
                      Qwen3MoeWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const Qwen3MoeWeights& weights() const { return weights_; }

 private:
  Qwen3MoeWeights weights_;
};

std::unique_ptr<LoadedModel> LoadQwen3MoeForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  (void)registration;
  (void)config;
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Qwen3MoeForCausalLM does not support GGUF weights");
  }
  // W2 stub: the BF16 safetensors expert loader (per-expert
  // mlp.experts.E.{gate,up,down}_proj + router gate + untied lm_head) is not yet
  // implemented — fail loudly rather than construct an empty model.
  throw std::runtime_error(
      "Qwen3MoeForCausalLM weight loading is not implemented yet (W2): the BF16 "
      "per-expert safetensors loader lands in a follow-up change");
}

void PrepareQwen3MoeForCausalLM(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardQwen3MoeForCausalLM(LoadedModel& model,
                                         const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // W3 stub: the forward (dense AttnBlock + RunMoeBlock per layer, no GDN, no
  // shared expert, untied lm_head) is not yet wired.
  throw std::runtime_error(
      "Qwen3MoeForCausalLM forward is not implemented yet (W3): composes the "
      "shared dense AttnBlock + the exposed RunMoeBlock per layer");
}

const ModelFactory kQwen3MoeFactory{
    .parse_config = &ParseQwen3MoeConfig,
    .load_weights = &LoadQwen3MoeForCausalLM,
    .prepare = &PrepareQwen3MoeForCausalLM,
    .forward = &ForwardQwen3MoeForCausalLM,
    .make_kv_cache = &MakeQwen3MoeKVCache,
    .is_dense_model = false,
};

}  // namespace

void ParseQwen3MoeConfig(const HfConfig& config) {
  // Verify the MoE fields the W2 loader / W3 forward consume are materialized by
  // LoadHfConfig. shared_expert_intermediate_size may legitimately be 0
  // (Qwen3-Coder has no shared expert — the no-shared guard, SEAM GAP #3).
  if (config.num_experts <= 0) {
    throw std::runtime_error(
        "Qwen3MoeForCausalLM config: num_experts must be > 0");
  }
  if (config.num_experts_per_tok <= 0 ||
      config.num_experts_per_tok > config.num_experts) {
    throw std::runtime_error(
        "Qwen3MoeForCausalLM config: num_experts_per_tok must be in "
        "[1, num_experts]");
  }
  if (config.moe_intermediate_size <= 0) {
    throw std::runtime_error(
        "Qwen3MoeForCausalLM config: moe_intermediate_size must be > 0");
  }
  (void)config.shared_expert_intermediate_size;  // 0 is valid (no shared expert)
}

v1::KVCacheConfig MakeQwen3MoeKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  // Full-attention MoE: exactly ONE full-attention KV group, NO MambaSpec/GDN
  // group. Byte-for-byte the pure-dense topology (clone of
  // MakeQwen3ForCausalLMKVCache) — the runner's full-attention-only path
  // (gdn_group_id_ < 0) already handles it with zero runner change.
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

REGISTER_VLLM_MODEL(qwen3_moe, "Qwen3MoeForCausalLM", kQwen3MoeFactory,
                    kQwen3MoeInfo)

}  // namespace vllm
