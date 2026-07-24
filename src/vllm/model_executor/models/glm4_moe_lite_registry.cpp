// GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`) registry TU — GLM/DSA spike G1.
//
// GLM-4.7-Flash IS DeepSeek-V2 with GLM's MoE block bolted in: upstream
// `glm4_moe_lite.py:94-95,98-99` are LITERAL zero-override subclasses of
// `DeepseekV2Attention` / `DeepseekV2MLAAttention`, and its decoder layer, model
// and `load_weights` (incl. the `fused_qkv_a_proj` merge at `:330-335,544-551`)
// are structural copies of deepseek_v2. The only GLM-specific piece upstream is
// `Glm4MoeLite = Glm4MoE` (`:86-87`), and `Glm4MoE` is itself a near-verbatim
// `DeepseekV2MoE` port (sigmoid `noaux_tc` grouped router with
// `e_score_correction_bias` + `routed_scaling_factor`, one shared expert —
// glm4_moe.py:139-224). Every one of those pieces already exists in our
// DeepSeek-V2 TU (MLA campaign W6-W9): the MLA attention block with BOTH q_lora
// branches, the `noaux_tc` grouped router (`MoeRouterTopKArgs` sigmoid + bias +
// group masking + `routed_scaling_factor`, cuda_moe.cu), the bf16 grouped MoE
// GEMM with shared expert, the loader with the `fused_qkv_a_proj` merge, and the
// decode CUDA-graph driver. So this TU registers `Glm4MoeLiteForCausalLM` by
// COMPOSING deepseek_v2's forward/loader over the SAME `DeepseekV2Weights`,
// changing exactly three things vs `deepseek_v2_registry.cpp`:
//
//   1. the registered arch string / factory identity;
//   2. `allow_mtp_tail = true` on the parse+load (GLM-4.7-Flash ships
//      `num_nextn_predict_layers: 1`; the loader requests only the main
//      `[0, num_hidden_layers)` layers, so the MTP tail is never pulled — exactly
//      upstream's `get_spec_layer_idx_from_weight_name` skip,
//      glm4_moe_lite.py:358-360,633-643);
//   3. nothing else — the head dims differ (qk_nope 192 / qk_rope 64 / v 256 vs
//      DeepSeek-V2-Lite's 128/64/128, q_lora_rank 768 vs null, sigmoid+bias+
//      `routed_scaling_factor: 1.8` vs softmax/greedy/1.0), but every one of
//      those is a CONFIG VALUE the DeepSeek-V2 code already reads and dispatches
//      on. GLM is the first checkpoint that EXERCISES the q_lora query branch and
//      the whole `noaux_tc` router end-to-end — the two gaps the MLA campaign
//      named as unit-gated-only on DeepSeek-V2-Lite (glm-dsa spike §0.1 C2).
//
// The MLA prefill head_dim-256 (GLM qk_nope 192 + qk_rope 64) is an ADDITIVE
// launcher-dispatch entry in cuda_flash_attn_fa2.cu (the 256 split-KV kernel is
// already compiled for the 27B/35B gate models); DeepSeek's 192 path is
// byte-identical.
//
// SCOPE HONESTY: this makes the model LOAD and FORWARD. The SACRED token-exact
// gate against the vLLM 0.25.0 oracle is what moves the matrix row off `SPIKE`.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/platforms/interface.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for GLM-4.7-Flash: text generation, NOT hybrid (MLA is
// still full attention over a paged cache), NOT multimodal.
inline constexpr ModelInfo kGlm4MoeLiteInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// GLM-4.7-Flash loads into the SAME weight struct as DeepSeek-V2 — the config
// values differ, the layout does not — and reuses the SAME decode CUDA-graph
// driver (the graph outlives a single forward).
class Glm4MoeLiteLoadedModel final : public LoadedModel {
 public:
  Glm4MoeLiteLoadedModel(const ModelRegistration& registration,
                         DeepseekV2Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const DeepseekV2Weights& weights() const { return weights_; }
  std::unique_ptr<DeepseekV2DecodeGraph>& decode_graph() { return decode_graph_; }

 private:
  DeepseekV2Weights weights_;
  std::unique_ptr<DeepseekV2DecodeGraph> decode_graph_;
};

std::unique_ptr<LoadedModel> LoadGlm4MoeLiteForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Glm4MoeLiteForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  // allow_mtp_tail = true: GLM-4.7-Flash ships `num_nextn_predict_layers: 1`; the
  // loader pulls only the main layers, so the MTP tail is skipped, not refused.
  return std::make_unique<Glm4MoeLiteLoadedModel>(
      registration, LoadDeepseekV2ForCausalLMWeights(*source.safetensors, config,
                                                     /*allow_mtp_tail=*/true));
}

void PrepareGlm4MoeLiteForCausalLM(LoadedModel& model, const HfConfig& config,
                                   vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGlm4MoeLiteForCausalLM(LoadedModel& model,
                                            const ModelForwardInput& input) {
  auto& glm = static_cast<Glm4MoeLiteLoadedModel&>(model);
  const DeepseekV2Weights& weights = glm.weights();

  // Identical dispatch to deepseek_v2_registry.cpp: a PURE-DECODE CUDA step goes
  // through the model's graph driver (pad-to-nearest capture size, replay);
  // prefill/mixed/over-cap/CPU fall back to eager INSIDE the driver. Real-row
  // output is bit-identical to eager. `gdn_state_slots` carries max_num_reqs for
  // every arch, so this pure-MLA model reads its capture-size cap from it.
  if (input.pure_decode &&
      platforms::GetPlatform(input.queue.device.type).support_static_graph_mode()) {
    if (!glm.decode_graph()) {
      glm.decode_graph() = std::make_unique<DeepseekV2DecodeGraph>(
          weights, input.queue, input.gdn_state_slots);
    }
    return glm.decode_graph()->Step(input.token_ids, input.positions,
                                    input.attn_meta, input.attn_kv);
  }

  if (input.gather_logits) {
    return DeepseekV2Model::ForwardDevice(input.token_ids, input.positions,
                                          input.attn_meta, input.attn_kv, weights,
                                          input.queue, input.logits_indices);
  }
  return HostLogits(
      DeepseekV2Model::Forward(input.token_ids, input.positions, input.attn_meta,
                               input.attn_kv, weights, input.queue,
                               input.logits_indices),
      weights.params.vocab_size);
}

// Registry config hook: the resolve IS the validation. allow_mtp_tail = true so
// GLM-4.7-Flash's `num_nextn_predict_layers: 1` is accepted (tail skipped); every
// other unsupported field still throws with a precise message.
void ParseGlm4MoeLiteConfig(const HfConfig& config) {
  (void)ParseDeepseekV2Params(config, /*allow_mtp_tail=*/true);
}

// MLA KV-cache spec: exactly ONE `MLAAttentionSpec` group, 1 head,
// `kv_lora_rank + qk_rope_head_dim` (= 512 + 64 = 576, the SAME latent width as
// DeepSeek-V2) wide, NO factor-2 and NO separate V.
v1::KVCacheConfig MakeGlm4MoeLiteKVCache(const HfConfig& config, int block_size,
                                         int num_blocks) {
  const DeepseekV2Params p = ParseDeepseekV2Params(config, /*allow_mtp_tail=*/true);
  const int head_size = static_cast<int>(p.mla.head_size());

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<v1::MLAAttentionSpec>(block_size, head_size,
                                             v1::ResolveKvCacheDType()));
  return kv;
}

const ModelFactory kGlm4MoeLiteFactory{
    .parse_config = &ParseGlm4MoeLiteConfig,
    .load_weights = &LoadGlm4MoeLiteForCausalLM,
    .prepare = &PrepareGlm4MoeLiteForCausalLM,
    .forward = &ForwardGlm4MoeLiteForCausalLM,
    .make_kv_cache = &MakeGlm4MoeLiteKVCache,
    .is_dense_model = false,
};

}  // namespace

REGISTER_VLLM_MODEL(glm4_moe_lite, "Glm4MoeLiteForCausalLM", kGlm4MoeLiteFactory,
                    kGlm4MoeLiteInfo)

}  // namespace vllm
