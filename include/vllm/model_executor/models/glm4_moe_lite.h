// GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`, HF `model_type: glm4_moe_lite`) — the
// one place this port says what GLM's MoE block is, as opposed to DeepSeek-V2's.
// Row `MODEL-TEXT-GLM4-MOE-LITE-ROUTER-F32`, issue
// [#2928](https://github.com/mudler/vllm.cpp/issues/2928).
//
// ─── WHY THIS MODEL SHARES DeepseekV2Params AND STILL NEEDS ITS OWN PARSE ────
// Upstream at the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`,
// `Glm4MoeLiteAttention` / `Glm4MoeLiteMLAAttention` are literal zero-override
// subclasses of `DeepseekV2Attention` / `DeepseekV2MLAAttention`
// (glm4_moe_lite.py:94-95, :98-99), and the decoder layer, model and
// `load_weights` are structural copies of deepseek_v2. So the LAYOUT is
// DeepSeek-V2's and one weight struct serves both.
//
// The MoE BLOCK is not. `Glm4MoeLite` is `Glm4MoE` (glm4_moe_lite.py:86-87,
// instantiated at :161-165), NOT `DeepseekV2MoE`, and the two resolve their
// router logit dtype differently:
//
//   * `Glm4MoE`'s gate is `nn.Linear(hidden_size, n_routed_experts, bias=False,
//     dtype=torch.float32)` (glm4_moe.py:141-146), fed
//     `hidden_states.to(dtype=torch.float32)` (:218), declared
//     `router_logits_dtype=torch.float32` (:205). No config key participates:
//     fp32 is a property of the CLASS.
//   * `DeepseekV2MoE` reads it from the config —
//     `GateLinear(..., out_dtype=_get_moe_router_dtype(config))`
//     (deepseek_v2.py:308-314, :123-133) — which is fp32 only for
//     `model_type == "glm_moe_dsa"` or an explicit `moe_router_dtype:
//     "float32"`.
//
// The published `zai-org/GLM-4.7-Flash` config.json declares no
// `moe_router_dtype`, so composing DeepSeek-V2's parser for GLM resolved the
// wrong dtype and rounded the router logits to bf16 in front of a top-4-of-64
// `noaux_tc` selection. `ParseDeepseekV2Params` is NOT the place to fix that: it
// serves `DeepseekV2ForCausalLM` and deliberately does not read `model_type`
// (deepseek_v2_weights.cpp:321-330). GLM-5.3 already carries its own answer the
// same way (glm_moe_dsa.cpp:353).
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM4_MOE_LITE_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM4_MOE_LITE_H_

#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Resolve `Glm4MoeLiteForCausalLM` params. Composes `ParseDeepseekV2Params` with
// `allow_mtp_tail = true` (GLM-4.7-Flash ships `num_nextn_predict_layers: 1` and
// the loader never requests the tail — glm4_moe_lite.py:358-360, :633-643) and
// then applies what `Glm4MoeLite` being `Glm4MoE` rather than `DeepseekV2MoE`
// means. Every GLM registry hook — `parse_config`, `load_weights`,
// `make_kv_cache` — resolves through this one function, so the params the
// forward reads and the params the config hook validates cannot drift.
//
// Pure/host; unit-testable without a checkpoint
// (tests/vllm/models/test_glm4_moe_lite_router_dtype.cpp).
DeepseekV2Params ParseGlm4MoeLiteParams(const HfConfig& config);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM4_MOE_LITE_H_
