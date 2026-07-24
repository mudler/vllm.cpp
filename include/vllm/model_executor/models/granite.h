// Granite-3 (`GraniteForCausalLM`, IBM Granite-3.x) — a ZERO-NEW-KERNEL dense
// bring-up: the Llama dense forward plus FOUR default-1 scalar multipliers.
// (Upstream: vllm/model_executor/models/granite.py @ e24d1b24; config
// ibm-granite/granite-3.3-2b-instruct/config.json — a standard dense transformer:
// NO GDN, NO MoE, standard (non-gemma) RMSNorm, GQA 32/8, head_dim 64, SwiGLU,
// tied lm_head, rope_scaling null -> plain NeoX rope, one full-attention KV group.)
//
// Granite IS the Llama forward with exactly four scalar multipliers, all read
// from config and all default-1 (so a Llama checkpoint would be byte-identical):
//   1. embedding_multiplier (granite.py:313): hidden = embed(ids) * embedding_multiplier.
//   2. attention_multiplier (granite.py:137,165): the attention softmax scale is
//      config.attention_multiplier (NOT head_dim**-0.5). For granite-3.3-2b this
//      is 0.015625 (= 1/64), distinct from 1/sqrt(64)=0.125 — a silent-corruption
//      hazard if wired as the default scale (fluent-WRONG tokens, the OPT mode).
//   3. residual_multiplier (granite.py:240,245): each sublayer output is scaled by
//      residual_multiplier BEFORE the residual add: h = residual + sublayer(h)*mult.
//      So the fused add+RMSNorm form (residual += hidden) is NOT usable — the norm
//      is a STANDALONE RMSNorm and the scaled add is a separate MulScalar+Add.
//   4. logits_scaling (granite.py:371-372): logits = lm_head(h) / logits_scaling.
//      (A positive divisor is a no-op for greedy argmax, but applied for fidelity.)
//
// Everything else — merged QKV (loaded from separate q/k/v like Llama), GQA,
// SwiGLU MLP, standard RMSNorm, plain NeoX RoPE, tied lm_head, the full-attention
// paged path — is REUSED from the shared dense substrate (dense_attn_block.h glue,
// dense_weight_loaders.h). The self-attention block + decoder layer are written
// FRESH (like OLMo-2/OPT) because dense_attn::AttnBlock hard-codes the 1/sqrt(Dh)
// scale and the fused residual add — two Granite violations. See
// .agents/specs/sweep-recent-dense-batch.md §0.2 row 2.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"  // shared dense weights container
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// Granite reuses the shared dense weight container (Qwen3DenseWeights): input/post
// RMSNorm + merged qkv/gate_up + tied-or-untied lm_head, with q_norm/k_norm EMPTY
// (Granite has no qk-norm) and qkv_bias EMPTY (attention_bias=false). The four
// scalar multipliers live in HfConfig, not the weights.
using GraniteWeights = Qwen3DenseWeights;

// Load `GraniteForCausalLM` (granite-3.3-2b-instruct, BF16) safetensors into the
// shared dense container. Name map is IDENTICAL to Llama (flat, no multimodal
// prefix): model.embed_tokens.weight, model.norm.weight, lm_head.weight (SKIPPED
// when tie_word_embeddings), per layer model.layers.N.{input_layernorm,
// post_attention_layernorm}.weight, .self_attn.{q,k,v,o}_proj.weight,
// .mlp.{gate,up,down}_proj.weight. q/k/v merged into one qkv_proj and gate/up into
// one gate_up_proj (vLLM packed_modules_mapping). Reuses the shared
// dense_weight_loaders.h helpers. Text path only.
GraniteWeights LoadGraniteForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The Granite dense forward. Per decoder layer (granite.py::GraniteDecoderLayer):
//   residual = h
//   h = input_layernorm(h)                    # STANDALONE (non-gemma) RMSNorm
//   h = self_attn(h)                          # scale = attention_multiplier
//   h = residual + h * residual_multiplier    # scaled residual add
//   residual = h
//   h = post_attention_layernorm(h)           # STANDALONE RMSNorm
//   h = mlp(h)                                # SwiGLU
//   h = residual + h * residual_multiplier
// Model: h = embed(ids) * embedding_multiplier -> N layers -> final RMSNorm ->
// lm_head(h) / logits_scaling. bf16 residual stream (mirrors vLLM's per-op bf16
// stores); qkv/rope/KV/attn/mlp flow bf16; the standalone norms + scaled adds
// compute in f32 and round to bf16. Returns [n_out, vocab] f32 logits.
class GraniteModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const GraniteWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const GraniteWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// Per-family config hook (mirrors ParseLlamaForCausalLMConfig). LoadHfConfig
// materializes the typed fields; the four Granite scalars (embedding_multiplier,
// residual_multiplier, attention_multiplier, logits_scaling) are read from
// config.raw by the forward. Validates plain (non-partial) NeoX rope.
void ParseGraniteForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group, NO MambaSpec/GDN
// (Granite-3 dense is pure full-attention). Identical topology to Llama's.
v1::KVCacheConfig MakeGraniteForCausalLMKVCache(const HfConfig& config,
                                                int block_size, int num_blocks);

}  // namespace vllm
