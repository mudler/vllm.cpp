// Kimi-Linear (`KimiLinearForCausalLM`) registry TU — the ADDITIVE self-
// registration seam for the Kimi-Linear W1 structural bring-up
// (`CLAIM-KIMI-LINEAR-W1`). Follows the deepseek_v2_registry.cpp / kimi_k3_registry
// .cpp seam exactly: a NEW translation unit with ONE REGISTER_VLLM_MODEL line and
// ZERO edit to any shared array. It owns the arch entry points: the config hook
// (config-descent validation), the HETEROGENEOUS KV-cache spec (MLA latent group +
// KDA/GDN recurrent-state group), the LoadedModel subclass + factory.
//
// Kimi-Linear IS registered in the pinned oracle (registry.py:139 -> kimi_linear,
// KimiLinearForCausalLM) and it FITS one GB10, so — unlike its 2.8T K3 sibling —
// it earns a REAL e2e SACRED token gate (spike §4/§8). W1 registers the arch so it
// RESOLVES + parses config + loads (name-map coverage). It does NOT forward yet:
// KimiLinearModel REFUSES-by-name (VT_CHECK(false)); the model-matrix row stays
// SPIKE until the W3-W6 forward + e2e gate land. See .agents/specs/kimi-linear.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/kimi_linear.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Kimi-Linear: text generation, HYBRID (20 KDA linear-
// attention layers ⇒ a GDN recurrent-state KV group), NOT multimodal. Mirrors the
// GDN-hybrid twin kQwen3_5Info (is_hybrid=true, has_inner_state=false — the
// upstream KimiLinearForCausalLM carries HasInnerState + IsHybrid, but our
// ModelInfo is a consumed subset whose only has_inner_state reader short-circuits
// on is_hybrid, so we follow the established hybrid-recurrent registration
// convention). supports_multimodal is false — Kimi-Linear is text-only (its K3
// wrapper is the multimodal one).
inline constexpr ModelInfo kKimiLinearInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class KimiLinearLoadedModel final : public LoadedModel {
 public:
  KimiLinearLoadedModel(const ModelRegistration& registration,
                        KimiLinearWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const KimiLinearWeights& weights() const { return weights_; }

 private:
  KimiLinearWeights weights_;
};

std::unique_ptr<LoadedModel> LoadKimiLinearForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture KimiLinearForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<KimiLinearLoadedModel>(
      registration,
      LoadKimiLinearForCausalLMWeights(*source.safetensors, config));
}

void PrepareKimiLinearForCausalLM(LoadedModel& model, const HfConfig& config,
                                  vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardKimiLinearForCausalLM(LoadedModel& model,
                                           const ModelForwardInput& input) {
  auto& kl = static_cast<KimiLinearLoadedModel&>(model);
  const KimiLinearWeights& weights = kl.weights();
  if (input.gather_logits) {
    return KimiLinearModel::ForwardDevice(input.token_ids, input.positions,
                                          input.attn_meta, input.attn_kv, weights,
                                          input.queue, input.logits_indices);
  }
  return HostLogits(
      KimiLinearModel::Forward(input.token_ids, input.positions, input.attn_meta,
                               input.attn_kv, weights, input.queue,
                               input.logits_indices),
      weights.params.vocab_size);
}

const ModelFactory kKimiLinearFactory{
    .parse_config = &ParseKimiLinearConfig,
    .load_weights = &LoadKimiLinearForCausalLM,
    .prepare = &PrepareKimiLinearForCausalLM,
    .forward = &ForwardKimiLinearForCausalLM,
    .make_kv_cache = &MakeKimiLinearKVCache,
    .is_dense_model = false,
};

}  // namespace

v1::KVCacheConfig MakeKimiLinearKVCache(const HfConfig& config, int block_size,
                                        int num_blocks) {
  // The HETEROGENEOUS per-layer KV topology (spike §3): TWO groups.
  //  (1) an MLA latent-KV group for the 7 full-attn layers — ONE latent row per
  //      token, kv_lora_rank + qk_rope_head_dim wide, num_kv_heads==1, NO separate
  //      V (MLAAttentionSpec, kv_cache_interface.h:189-238), exactly as DeepSeek-V2.
  //  (2) a KDA/GDN recurrent-state MambaSpec group for the 20 KDA layers — a
  //      conv-state row (3*num_heads*head_dim wide, conv_kernel-1 taps) + a square
  //      recurrent-state (num_heads, head_dim, head_dim), per mamba_utils.py
  //      kda_state_shape (:274-294) / kda_state_dtype (:130-137: conv=cache dtype,
  //      recurrent=float32). Mirrors the qwen3_5 GDN-hybrid two-group builder
  //      (qwen3_5_common.cpp:65-105) with MLA in place of the full-attention group.
  // W1 DECLARES the shapes/routing; the runner wiring is W6.
  const KimiLinearParams p = ParseKimiLinearParams(config);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<v1::MLAAttentionSpec>(
          block_size, static_cast<int>(p.mla_head_size()),
          v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"kda"},
      std::make_shared<v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{
              {p.kda_conv_dim(), p.kda_short_conv_kernel_size - 1},
              {p.kda_num_heads, p.kda_head_dim, p.kda_head_dim}},
          std::vector<vt::DType>{vt::DType::kBF16, vt::DType::kF32}));
  return kv;
}

REGISTER_VLLM_MODEL(kimi_linear, "KimiLinearForCausalLM", kKimiLinearFactory,
                    kKimiLinearInfo)

}  // namespace vllm
