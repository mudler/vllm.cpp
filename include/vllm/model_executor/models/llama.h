// Llama-3.x (`LlamaForCausalLM`) — the cross-family dense additive bring-up.
// (Upstream: vllm/model_executor/models/llama.py @ e24d1b24; config
// unsloth/Llama-3.2-1B/config.json — a standard dense transformer: NO GDN, NO MoE,
// standard (non-gemma) RMSNorm, GQA, SwiGLU, tied lm_head, sliding_window absent
// -> one full-attention KV group.)
//
// Llama IS the Qwen3-dense forward with exactly two deltas, both handled
// ADDITIVELY by the shared dense attention block (dense_attn_block.h):
//   1. NO per-head q/k RMSNorm — Qwen3 has qk-norm, Llama does not. The loader
//      leaves Qwen3DenseAttnWeights::q_norm/k_norm EMPTY and AttnBlock skips the
//      norm step (byte-preserving for Qwen3, which loads them).
//   2. llama3 RoPE frequency scaling — Llama-3.2 declares rope_theta=500000 AND
//      rope_scaling of type "llama3" (a piecewise low/high-frequency wavelength
//      rescale of the base inv_freq). MakeRopeArgs(cfg) fills the RopeArgs llama3
//      fields when rope_type=="llama3"; the RoPE kernels apply the rescale
//      (byte-identical to plain RoPE for every non-llama3 model). Mirrors vLLM
//      Llama3RotaryEmbedding (rotary_embedding/llama3_rope.py:33-54).
// Everything else — merged QKV, GQA, SwiGLU MLP, standard RMSNorm, tied lm_head,
// the full-attention paged forward — is REUSED VERBATIM from the Qwen3-dense path.
// So the ONLY Llama-specific code is the weight NAME MAP (no q_norm/k_norm) and the
// registry wiring; the weight container and forward machinery are the shared
// dense ones. See .agents/specs/sweep-llama-3.2.md.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"  // shared dense weights + forward
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

class SafetensorsFile;

// Llama reuses the shared dense weight container and forward. q_norm/k_norm stay
// EMPTY (Llama has no qk-norm); tie_word_embeddings/attention_bias come from the
// checkpoint config. Named aliases keep the Llama TUs self-documenting.
using LlamaWeights = Qwen3DenseWeights;
using LlamaModel = Qwen3DenseModel;

// Load `LlamaForCausalLM` (Llama-3.2-1B, BF16) safetensors into the shared dense
// container. Name map (identical to Qwen3 MINUS the per-head q/k norms):
//   model.embed_tokens.weight, model.norm.weight, lm_head.weight (SKIPPED when
//   tie_word_embeddings), and per layer model.layers.N.{input_layernorm,
//   post_attention_layernorm}.weight, .self_attn.{q,k,v,o}_proj.weight,
//   .mlp.{gate,up,down}_proj.weight. q/k/v merged into one qkv_proj and gate/up
//   into one gate_up_proj (vLLM packed_modules_mapping). Reuses the shared
//   dense_weight_loaders.h helpers. Text path only.
LlamaWeights LoadLlamaForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                         const HfConfig& config);

// Per-family config hook (mirrors ParseQwen3ForCausalLMConfig). LoadHfConfig
// already materializes + validates every consumed Llama field, including the
// llama3 rope_scaling dictionary; this explicit no-op hook is the family's
// normalization/validation seam.
void ParseLlamaForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group, NO MambaSpec/GDN
// (Llama is pure full-attention). Identical topology to the Qwen3-dense builder.
v1::KVCacheConfig MakeLlamaForCausalLMKVCache(const HfConfig& config, int block_size,
                                              int num_blocks);

}  // namespace vllm
