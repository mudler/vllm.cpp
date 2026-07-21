// Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`) — the first full-attention MoE
// bring-up. It COMBINES two already-DONE paths: the Qwen3-dense standard
// attention (qwen3.cpp `Qwen3DenseModel`) and the Qwen3.6-35B sparse-MoE block
// (qwen3_5.cpp `MoeBlock`, exposed via qwen3_5_moe_block.h). Structurally it is
// Qwen3 dense with the per-layer SwiGLU MLP replaced by the MoE block:
//   - full-attention KV group per layer (NO GDN, NO hybrid split, one FA group);
//   - standard (non-gemma) RMSNorm at input/post/final norms;
//   - per-head q_norm/k_norm before RoPE (reuses the dense `AttnBlock`);
//   - a router + top-k experts, NO shared expert (guarded on
//     shared_expert_intermediate_size==0), untied lm_head, unquantized BF16.
//
// Grounding: vllm/model_executor/models/qwen3_moe.py @ e24d1b24
//   Qwen3MoeAttention (:254-354, == qwen3.py Qwen3Attention), Qwen3MoeSparseMoeBlock
//   (:130-251), Qwen3MoeDecoderLayer (:357-429), Qwen3MoeForCausalLM (:541-657).
// See .agents/specs/sweep-qwen3-coder-30b.md §2/§3/§4/§7.
//
// W0/W1 scope (this change): the registry stub + the reusable extracted/exposed
// pieces. This header carries the whole-model weight struct + the registry-facing
// config/KV-cache hooks + the Forward decls. The BF16 loader (W2) and the forward
// (W3, composing the dense `AttnBlock` + the exposed `RunMoeBlock`) are NOT yet
// implemented; the registered load/forward hooks are clear-throwing stubs.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"             // Qwen3DenseAttnWeights, PagedKvCache
#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor, MoeBlockWeights
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// One Qwen3-Coder decoder layer (qwen3_moe.py::Qwen3MoeDecoderLayer): input/post
// standard (non-gemma) RMSNorm + full-attention (REUSES the dense
// `Qwen3DenseAttnWeights`: merged qkv, per-head q/k RMSNorm, no attn gate) + the
// sparse-MoE block (REUSES `MoeBlockWeights`: router gate + per-expert bf16
// gate/up/down). Qwen3-Coder has NO shared expert, so the `shared_*` fields of
// `MoeBlockWeights` stay EMPTY (guarded on shared_expert_intermediate_size==0).
struct Qwen3MoeLayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  Qwen3DenseAttnWeights attn;            // reused dense full-attention weights
  MoeBlockWeights moe;                   // reused 35B MoE block weights (bf16, no shared)
};

// Whole Qwen3-Coder text-model weights. `tie_word_embeddings` is FALSE for
// Qwen3-Coder (untied `lm_head`, Matmul-B [H, vocab] layout), unlike Qwen3-0.6B.
struct Qwen3MoeWeights {
  bool tie_word_embeddings = false;
  bool attention_bias = false;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (NOT transposed; embed lookup)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY only when tied
  std::vector<Qwen3MoeLayerWeights> layers;
};

// The full-attention MoE forward (W3 capstone — NOT implemented in W0/W1). Built
// by COMPOSING the shared dense `AttnBlock` (dense_attn_block.h) + the exposed
// `RunMoeBlock` (qwen3_5_moe_block.h) per layer, with NO GDN and NO shared
// expert. Same batched-paged contract + return shape as `Qwen3DenseModel`.
class Qwen3MoeModel {
 public:
  // Batched PAGED forward. token_ids/positions are the flattened
  // length-num_actual_tokens step inputs; attn_meta the full-attention KV group's
  // CommonAttentionMetadata; attn_kv one PagedKvCache per layer (all layers are
  // full-attention). Returns [n_out, vocab] f32 logits.
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Qwen3MoeWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // DEVICE-resident variant (sampler-on-device hot path): same contract as
  // Forward but returns the lm_head output as a device buffer with no full-logits
  // D2H (ForwardLogits::device_*).
  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Qwen3MoeWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// Load `Qwen3MoeForCausalLM` (Qwen3-Coder-30B-A3B) safetensors into
// Qwen3MoeWeights (W2 — NOT implemented in W0/W1). Name map: model.embed_tokens,
// model.norm, model.layers.N.{input_layernorm,post_attention_layernorm},
// .self_attn.{q,k,v,o}_proj (+ per-head q/k norm), .mlp.gate (router),
// .mlp.experts.E.{gate,up,down}_proj (per-expert bf16), untied lm_head.
Qwen3MoeWeights LoadQwen3MoeForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// Per-family config hook. Verifies the MoE fields the loader/forward consume are
// materialized from the HfConfig (num_experts / num_experts_per_tok /
// moe_intermediate_size; shared_expert_intermediate_size may legitimately be 0 —
// Qwen3-Coder has no shared expert). is_dense_model=false is set on the factory.
void ParseQwen3MoeConfig(const HfConfig& config);

// KV-cache spec builder for the full-attention MoE arch: exactly ONE
// full-attention KV group, NO MambaSpec/GDN group (Qwen3-Coder has no
// linear-attention layers). Clone of MakeQwen3ForCausalLMKVCache — the runner's
// full-attention-only generalization (ENG-RUNNER-MODELSHAPE) already covers it
// (gdn_group_id_ < 0), so there is ZERO runner change for this arch.
v1::KVCacheConfig MakeQwen3MoeKVCache(const HfConfig& config, int block_size,
                                      int num_blocks);

}  // namespace vllm
