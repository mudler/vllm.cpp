// Mistral (`MistralForCausalLM`) — the fifth model family, the near-additive
// dense bring-up that sits even closer to plain Llama than Llama-3.2 does.
// (Upstream: vllm/model_executor/models/mistral.py @ e24d1b24 — literally "Mistral
// adaptation of the LLaMA architecture": MistralForCausalLM(LlamaForCausalLM),
// MistralAttention(LlamaAttention), MistralModel(LlamaModel). Config
// mistralai/Mistral-7B-v0.3/config.json — a standard dense transformer: NO GDN,
// NO MoE, standard (non-gemma) RMSNorm, GQA 32/8, head_dim 128, SwiGLU, sliding_window
// null -> one full-attention KV group.)
//
// Mistral IS the Qwen3-dense forward, and simpler than Llama — the two Llama deltas
// both resolve to REUSE with NO new primitive:
//   1. NO per-head q/k RMSNorm — like Llama, Mistral has none. The loader leaves
//      Qwen3DenseAttnWeights::q_norm/k_norm EMPTY and the shared AttnBlock skips the
//      norm step (has_qk_norm == false). Reuses the qk-norm-optional seam landed for
//      Llama.
//   2. PLAIN RoPE, theta=1000000, NO rope_scaling — Mistral-7B declares no
//      rope_scaling dictionary, so rope_type=="default" and MakeRopeArgs(cfg) leaves
//      the llama3 fields zero: the RoPE kernels take their no-op branch. This is
//      SIMPLER than Llama-3.2 (which has the piecewise llama3 rescale); Mistral rides
//      the exact same plain-rope path as Qwen3-dense with base 1e6. ZERO new kernel.
// Two config-shape differences vs Llama-3.2-1B, both already handled generically:
//   * UNTIED embeddings (tie_word_embeddings=false) — Mistral-7B ships a standalone
//     lm_head; the shared loader loads it (Matmul-B [H, vocab]) when untied, exactly
//     as the Llama loader's untied branch does.
//   * sliding_window field present but null (SWA DISABLED in v0.3) — parsed by
//     HfConfig into config.sliding_window (None), so Mistral rides the FULL-attention
//     path with ONE FA KV group. A SWA-enabled Mistral variant (v0.1, window 4096) is
//     OUT OF SCOPE here: SlidingWindowSpec has no production construction site, so
//     such a variant would still use full attention (documented, separate row).
// Everything else — merged QKV, GQA, SwiGLU MLP, standard RMSNorm, untied lm_head,
// the full-attention paged forward — is REUSED VERBATIM from the Qwen3-dense path
// (MistralModel == Qwen3DenseModel). So the ONLY Mistral-specific code is the weight
// NAME MAP (identical to Llama's) and the registry wiring; the weight container and
// forward machinery are the shared dense ones. See .agents/specs/sweep-mistral.md.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"  // shared dense weights + forward
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

class SafetensorsFile;

// Mistral reuses the shared dense weight container and forward. q_norm/k_norm stay
// EMPTY (Mistral has no qk-norm); tie_word_embeddings is FALSE (untied lm_head) and
// attention_bias FALSE — both read from the checkpoint config. Named aliases keep
// the Mistral TUs self-documenting.
using MistralWeights = Qwen3DenseWeights;
using MistralModel = Qwen3DenseModel;

// Load `MistralForCausalLM` (Mistral-7B-v0.3, BF16) safetensors into the shared
// dense container. Name map is IDENTICAL to Llama (flat, no multimodal prefix):
//   model.embed_tokens.weight, model.norm.weight, lm_head.weight (LOADED — Mistral
//   is untied), and per layer model.layers.N.{input_layernorm,
//   post_attention_layernorm}.weight, .self_attn.{q,k,v,o}_proj.weight,
//   .mlp.{gate,up,down}_proj.weight. q/k/v merged into one qkv_proj and gate/up into
//   one gate_up_proj (vLLM packed_modules_mapping). Reuses the shared
//   dense_weight_loaders.h helpers. Text path only.
MistralWeights LoadMistralForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// Per-family config hook (mirrors ParseLlamaForCausalLMConfig). LoadHfConfig already
// materializes + validates every consumed Mistral field (rope_theta, the null
// sliding_window, GQA, head_dim); this explicit no-op hook is the family's
// normalization/validation seam.
void ParseMistralForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group, NO MambaSpec/GDN
// (Mistral-7B-v0.3 is pure full-attention, sliding_window null). Identical topology
// to the Llama/Qwen3-dense builder.
v1::KVCacheConfig MakeMistralForCausalLMKVCache(const HfConfig& config,
                                                int block_size, int num_blocks);

}  // namespace vllm
