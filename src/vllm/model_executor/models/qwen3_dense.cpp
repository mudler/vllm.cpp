// Qwen3 DENSE (`Qwen3ForCausalLM`) registry TU — the first ADDITIVE-MODEL
// bring-up (W0/W1). Self-registers "Qwen3ForCausalLM" via REGISTER_VLLM_MODEL
// and owns the arch-specific entry points: the config hook, the
// full-attention-ONLY KV-cache spec (one FA group, NO MambaSpec/GDN group), the
// LoadedModel subclass, and the factory. Mirrors the qwen3_5_dense.cpp seam
// (new TU + one in-TU REGISTER line → ZERO shared-array edit).
//
// W0/W1 scope: the forward is NOT implemented (it lands in W3 over vt:: ops +
// the fusion catalog). The registered load_weights + forward hooks are
// clear-throwing stubs so the registry resolves and the KV-cache/runner
// generalization can be gated on CPU, while any attempt to actually run the
// model fails loudly with a "W2"/"W3" message rather than silently.
// See .agents/specs/first-additive-model-qwen3-dense.md §3, §6.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits (shared carrier)
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Qwen3 dense: text generation, NOT hybrid (pure
// full-attention), NOT multimodal (no vision tower — Qwen3-0.6B is text-only).
inline constexpr ModelInfo kQwen3Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// Opaque owned model. W0/W1: holds the placeholder weights only; the forward is
// not yet wired (throws until W3).
class Qwen3DenseLoadedModel final : public LoadedModel {
 public:
  Qwen3DenseLoadedModel(const ModelRegistration& registration,
                        Qwen3DenseWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

 private:
  Qwen3DenseWeights weights_;
};

std::unique_ptr<LoadedModel> LoadQwen3ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  (void)registration;
  (void)config;
  (void)source;
  // W2 lands the safetensors name map + tied lm_head. Fail loudly until then.
  throw std::runtime_error(
      "Qwen3ForCausalLM weight loading is not implemented yet "
      "(additive-model bring-up W2)");
}

void PrepareQwen3ForCausalLM(LoadedModel& model, const HfConfig& config,
                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardQwen3ForCausalLM(LoadedModel& model,
                                      const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // W3 lands the dense decode graph (vt:: ops + the 2 new catalog recipes).
  throw std::runtime_error(
      "Qwen3ForCausalLM forward is not implemented yet "
      "(additive-model bring-up W3)");
}

const ModelFactory kQwen3DenseFactory{
    .parse_config = &ParseQwen3ForCausalLMConfig,
    .load_weights = &LoadQwen3ForCausalLM,
    .prepare = &PrepareQwen3ForCausalLM,
    .forward = &ForwardQwen3ForCausalLM,
    .make_kv_cache = &MakeQwen3ForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseQwen3ForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig/HfConfigFromGguf already materialize the consumed Qwen3 fields.
  // No-op hook (mirrors ParseQwen3_5Config) — the seam where the family would
  // add normalization/validation without changing the registry/runner contract.
  (void)config;
}

v1::KVCacheConfig MakeQwen3ForCausalLMKVCache(const HfConfig& config,
                                              int block_size, int num_blocks) {
  // Pure dense: exactly ONE full-attention KV group, NO MambaSpec/GDN group.
  // This is the full-attention-only KV topology the runner (W1) must handle.
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

REGISTER_VLLM_MODEL(qwen3_dense, "Qwen3ForCausalLM", kQwen3DenseFactory,
                    kQwen3Info)

}  // namespace vllm
