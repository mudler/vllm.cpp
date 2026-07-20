// Qwen3 DENSE (`Qwen3ForCausalLM`) — the first ADDITIVE-MODEL bring-up.
// (Upstream: vllm/model_executor/models/qwen3.py @ e24d1b24; config
// Qwen3-0.6B/config.json — a pure standard-dense transformer: NO GDN, NO MoE,
// standard (non-gemma) RMSNorm, per-head q/k norm, tied lm_head, sliding_window
// null → one full-attention KV group only.)
//
// This header carries the registry-facing declarations shared by the Qwen3
// registry TU (qwen3_dense.cpp): the per-family config hook and the
// full-attention-ONLY KV-cache spec builder. The heavy dense forward machinery
// (Qwen3DenseModel::Forward/ForwardDevice) and the on-disk weight name map land
// in W2/W3 (qwen3.cpp / qwen3_weights.cpp) — see
// .agents/specs/first-additive-model-qwen3-dense.md §6. W0/W1 deliberately do
// NOT implement the forward; the registered forward hook is a clear-throwing
// stub until W3.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

class SafetensorsFile;

// One Qwen3 dense self-attention block. Mirrors vLLM `Qwen3Attention`
// (qwen3.py:65-168): merged QKV (packed_modules_mapping qkv_proj<-[q,k,v]) and
// per-head q/k RMSNorm (RMSNorm(head_dim)) applied before RoPE. Qwen3 has NO
// attention output gate (unlike the Qwen3.6 gated full-attention), so the merged
// QKV width is exactly Hq*Dh + 2*Hkv*Dh (no doubling).
//
// Projections are kept RAW in the on-disk torch Linear [N=out, K=in] orientation
// (nk=true), consumed by vt::MatmulBT — the cuBLASLt TN fast path vLLM's
// F.linear hits (mirrors the qwen3_5-dense in_proj_qkvz choice, notes §3.6).
struct Qwen3DenseAttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh], nk
  OwnedTensor q_norm;    // bf16 [head_dim]  (per-head RMSNorm, non-gemma)
  OwnedTensor k_norm;    // bf16 [head_dim]
  // Merged QKV bias [Hq*Dh + 2*Hkv*Dh], present only when config attention_bias
  // is true. EMPTY for Qwen3-0.6B (attention_bias=false).
  OwnedTensor qkv_bias;
};

// Dense SwiGLU MLP. Mirrors vLLM `Qwen3MLP` = `Qwen2MLP` (qwen3.py:58): merged
// gate_up (packed_modules_mapping gate_up_proj<-[gate,up]) -> SiluAndMul ->
// down_proj. Raw-NK like the attention projections.
struct Qwen3DenseMlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up), nk
  OwnedTensor down_proj;     // bf16 raw-NK [H, I], nk
};

// One Qwen3 dense decoder layer: input/post standard (non-gemma) RMSNorm +
// attention + dense SwiGLU MLP (qwen3.py:171-242).
struct Qwen3DenseLayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  Qwen3DenseAttnWeights attn;
  Qwen3DenseMlpWeights mlp;
};

// Whole Qwen3 dense text-model weights. When `tie_word_embeddings` is true
// (Qwen3-0.6B), `lm_head` is EMPTY and the output projection aliases
// `embed_tokens` — mirroring vLLM `Qwen3ForCausalLM.__init__` (`self.lm_head =
// self.model.embed_tokens`, qwen3.py:294-295) and its loader skip_prefixes
// (`["lm_head."]`, qwen3.py:339) which drops the checkpoint's redundant
// lm_head.weight. When false, `lm_head` holds the ParallelLMHead weight in
// Matmul-B [H, vocab] layout.
struct Qwen3DenseWeights {
  bool tie_word_embeddings = true;
  bool attention_bias = false;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (NOT transposed; embed lookup)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY when tied
  std::vector<Qwen3DenseLayerWeights> layers;
};

// Load `Qwen3ForCausalLM` (Qwen3-0.6B) safetensors into Qwen3DenseWeights. Name
// map: model.embed_tokens.weight, model.norm.weight, and per layer
// model.layers.N.{input_layernorm,post_attention_layernorm}.weight,
// .self_attn.{q,k,v,o}_proj.weight (+ per-head .self_attn.{q,k}_norm.weight),
// .mlp.{gate,up,down}_proj.weight. q/k/v merged into one qkv_proj and gate/up
// into one gate_up_proj (vLLM packed_modules_mapping). The checkpoint's
// lm_head.weight is intentionally SKIPPED when tie_word_embeddings (aliased to
// embed_tokens). Reuses the shared dense_weight_loaders.h helpers. Text path
// only; no vision tower (Qwen3-0.6B is text-only).
Qwen3DenseWeights LoadQwen3ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// Per-family config hook (mirrors ParseQwen3_5Config). LoadHfConfig already
// materializes the consumed Qwen3 fields (num_key_value_heads, head_dim,
// rope_theta, intermediate_size, rms_norm_eps, ...); this explicit no-op hook is
// where the family would add normalization/validation without touching the
// registry/runner contract. (tie_word_embeddings / attention_bias are consumed
// by the W2/W3 loader+forward, not by W0/W1.)
void ParseQwen3ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder for the pure-dense arch: exactly ONE full-attention KV
// group, NO MambaSpec/GDN group (Qwen3 dense has no linear-attention layers).
// This is what forces — and validates — the runner's full-attention-only
// generalization (W1): a KVCacheConfig with no mamba group and an empty
// layer_types must allocate + build metadata without the hybrid GDN path.
v1::KVCacheConfig MakeQwen3ForCausalLMKVCache(const HfConfig& config,
                                              int block_size, int num_blocks);

}  // namespace vllm
