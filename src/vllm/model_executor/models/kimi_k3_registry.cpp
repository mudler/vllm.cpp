// Kimi K3 (`KimiK3ForConditionalGeneration`) registry TU — the ADDITIVE self-
// registration seam for the Kimi-K3 structural bring-up (`CLAIM-KIMI-K3-W2-W5`,
// W2/W5). Follows the deepseek_v4_registry.cpp / qwen3_5_moe.cpp seam exactly: a
// NEW translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array. It owns the arch entry points: the config hook (config-descent
// validation), the KV-cache spec (STUB), the LoadedModel subclass + factory.
//
// K3 is beyond the pinned oracle (registry.py registers KimiLinearForCausalLM:140,
// KimiK25ForConditionalGeneration:460, MoonshotKimiaForCausalLM:461 — but NO
// KimiK3*). We register the ONE released string `KimiK3ForConditionalGeneration`
// so it RESOLVES + parses config + accounts the text-backbone structure. It does
// NOT forward: KimiK3Model REFUSE-by-name (VT_CHECK(false)), and the loader refuses
// MXFP4 (the real checkpoint dtype) — both by design, per DERIVE-AND-SHIP. The
// model-matrix row stays SPIKE. See .agents/specs/kimi-k3.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/kimi_k3.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Kimi K3: text generation, HYBRID (69 KDA linear-attn
// layers), MULTIMODAL (MoonViT-V2 vision). Like the Qwen3.5 sibling wrappers, the
// OUTER KimiK3ForConditionalGeneration registration inherits IsHybrid but NOT
// HasInnerState: the recurrent/conv inner state belongs to the inner
// KimiLinearForCausalLM language-model class, not this multimodal wrapper.
inline constexpr ModelInfo kKimiK3Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

class KimiK3LoadedModel final : public LoadedModel {
 public:
  KimiK3LoadedModel(const ModelRegistration& registration, KimiK3Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const KimiK3Weights& weights() const { return weights_; }

 private:
  KimiK3Weights weights_;
};

std::unique_ptr<LoadedModel> LoadKimiK3ForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture KimiK3ForConditionalGeneration does not support GGUF "
        "weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<KimiK3LoadedModel>(
      registration,
      LoadKimiK3ForConditionalGenerationWeights(*source.safetensors, config));
}

void PrepareKimiK3ForConditionalGeneration(LoadedModel& model,
                                           const HfConfig& config,
                                           vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardKimiK3ForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  auto& k3 = ModelAs<KimiK3LoadedModel>(model, "KimiK3ForConditionalGeneration");
  const KimiK3Weights& weights = k3.weights();
  if (input.gather_logits) {
    return KimiK3Model::ForwardDevice(input.token_ids, input.positions,
                                      input.attn_meta, input.attn_kv, weights,
                                      input.queue, input.logits_indices);
  }
  return HostLogits(
      KimiK3Model::Forward(input.token_ids, input.positions, input.attn_meta,
                           input.attn_kv, weights, input.queue,
                           input.logits_indices),
      weights.params.text.vocab_size);
}

const ModelFactory kKimiK3Factory{
    .parse_config = &ParseKimiK3Config,
    .load_weights = &LoadKimiK3ForConditionalGeneration,
    .prepare = &PrepareKimiK3ForConditionalGeneration,
    .forward = &ForwardKimiK3ForConditionalGeneration,
    .make_kv_cache = &MakeKimiK3KVCache,
    .is_dense_model = false,
};

}  // namespace

v1::KVCacheConfig MakeKimiK3KVCache(const HfConfig& config, int block_size,
                                    int num_blocks) {
  // STUB: K3's TRUE KV topology is a MULTI-group hybrid — MLA latent-KV for the 24
  // full-attn layers + KDA conv/recurrent state for the 69 linear-attn layers
  // (kimi_linear.py:600-633). That hybrid geometry is a NOT-YET-BUILDABLE residual
  // (Kimi-Linear KDA row). We emit ONE placeholder MLA group sized to the
  // compressed latent + rope so the arch RESOLVES. Never exercised — the forward
  // REFUSES-by-name.
  const KimiK3Params p = ParseKimiK3Params(config);
  const int head_size = static_cast<int>(p.text.kv_lora_rank +
                                         p.text.qk_rope_head_dim);  // 512 + 64
  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<v1::MLAAttentionSpec>(block_size, head_size,
                                             v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(kimi_k3, "KimiK3ForConditionalGeneration", kKimiK3Factory,
                    kKimiK3Info)

}  // namespace vllm
