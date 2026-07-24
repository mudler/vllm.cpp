// OLMo-2 dense (`Olmo2ForCausalLM`; alias `Olmo3ForCausalLM`) — the cleanest
// dense bring-up yet: ZERO new compute kernels. A plain dense GQA decoder (SiLU
// SwiGLU MLP, NeoX RoPE, no biases, tied-or-untied embeddings) with exactly TWO
// wiring facts that make it look distinctive, both realized over LANDED ops:
//
//   1. PURE POST-NORM placement (`norm_after`, olmo2.py:261-277). The attention
//      and MLP run on the RAW residual stream (NO pre/input norm); a standalone
//      RMSNorm hits each sublayer OUTPUT; a PLAIN residual add re-joins. This is a
//      strict SUBSET of the GLM-4/Gemma sandwich (glm4.cpp:174-178,186-188 keep
//      BOTH pre- and post-norm) — OLMo-2 keeps ONLY the standalone-output-norm op
//      and DROPS the pre-norms, re-joining with vt::Add instead of the fused
//      residual-add form. Reuses the standalone vt::RmsNorm + vt::Add; adds no kernel.
//   2. FULL-WIDTH QK-norm (olmo2.py:113-117,160-172). q_norm=RMSNorm(hidden_size)
//      and k_norm=RMSNorm(kv_size) normalize the WHOLE q/k projection (all heads
//      folded into one statistic) before RoPE — NOT the per-head head_dim norm the
//      fused kAttnQkNormRope(Gate) recipe expresses. Realized as two standalone
//      vt::RmsNorm over the [T,q_size]/[T,kv_size] views before a standard
//      RopeNeox. Reuses vt::RmsNorm at a new shape; adds no kernel.
//
// Everything else REUSES landed infrastructure: plain (non-gemma) RMSNorm, the
// SiLU SwiGLU merged gate_up MLP (kSiluAndMul), NeoX RoPE (RopeFromCache/RopeNeox),
// the shared dense device glue + GQA paged path (dense_attn_block.h), the merged
// qkv/gate_up loader (dense_weight_loaders.h), and tied-or-untied embeddings. The
// self-attention block is written FRESH (like OPT/GLM/Gemma) because dense_attn::
// AttnBlock hard-codes PRE-norm placement and per-head qk-norm — it reuses only
// the shared glue, never AttnBlock's body (spike .agents/specs/sweep-olmo2.md D2).
//
// Grounding: vllm/model_executor/models/olmo2.py @ e24d1b24 — Olmo2Attention
// (:67-185), Olmo2MLP (:188-230), Olmo2DecoderLayer.forward (:261-277), Olmo2Model
// final norm (:297-343), Olmo2ForCausalLM tie (:374-382,413-420). Config
// allenai/OLMo-2-0425-1B: dense pure-text, plain RMSNorm, tie_word_embeddings,
// NeoX rope (rotary_dim == head_dim), no biases. bf16-only.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// One OLMo-2 self-attention block (olmo2.py::Olmo2Attention). Merged QKV
// (packed_modules_mapping qkv_proj<-[q,k,v]) with NO bias, FULL-WIDTH q/k RMSNorm
// (q_norm[q_size], k_norm[kv_size] — the WHOLE projection, not per-head) applied
// before a standard NeoX RoPE, and an unbiased o_proj. Raw torch-Linear [N=out,
// K=in] orientation for vt::MatmulBT.
struct Olmo2AttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh], nk (no bias)
  OwnedTensor q_norm;    // bf16 [Hq*Dh == hidden_size]  (FULL-WIDTH RMSNorm, non-gemma)
  OwnedTensor k_norm;    // bf16 [Hkv*Dh == kv_size]     (FULL-WIDTH RMSNorm, non-gemma)
};

// OLMo-2 SwiGLU MLP (olmo2.py::Olmo2MLP): merged gate_up (gate_up_proj<-[gate,up])
// -> SiluAndMul -> down_proj. Raw-NK. No bias. Identical to Qwen3/Llama's.
struct Olmo2MlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up), nk
  OwnedTensor down_proj;     // bf16 raw-NK [H, I], nk
};

// One OLMo-2 decoder layer (olmo2.py::Olmo2DecoderLayer). The PURE POST-NORM
// layout: TWO standalone (non-gemma) RMSNorms applied to the attn / mlp sublayer
// OUTPUT before a PLAIN residual add — and NO input/pre norm at all.
struct Olmo2LayerWeights {
  OwnedTensor post_attention_layernorm;    // bf16 [H]  (standalone, attn output)
  OwnedTensor post_feedforward_layernorm;  // bf16 [H]  (standalone, mlp output)
  Olmo2AttnWeights attn;
  Olmo2MlpWeights mlp;
};

// Whole OLMo-2 dense text-model weights. When `tie_word_embeddings` (OLMo-2-1B),
// `lm_head` is EMPTY and the output projection aliases `embed_tokens` — mirroring
// vLLM `Olmo2ForCausalLM.__init__` (`self.lm_head = self.model.embed_tokens`,
// olmo2.py:374-375) and the loader skip_prefixes `["lm_head.weight"]`
// (olmo2.py:416-418). When false, `lm_head` holds the ParallelLMHead weight in
// Matmul-B [H, vocab] layout.
struct Olmo2Weights {
  bool tie_word_embeddings = true;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (embed lookup, NOT transposed)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY when tied
  std::vector<Olmo2LayerWeights> layers;
  // OLMo-3 ONLY (`Olmo3ForCausalLM`): the precomputed YaRN cos/sin cache
  // (bf16 [P, rotary_dim], [cos|sin] halves, indexed by REAL position) consumed by
  // the FULL-attention layers. Rope scaling is applied on full-attention layers
  // ONLY (olmo2.py:139-149); sliding-window layers use the plain-rope path (base
  // rope_theta). EMPTY for OLMo-2 (no rope_scaling) — so the OLMo-2 forward is
  // byte-identical (this field is never read when config.layer_types is empty).
  OwnedTensor rope_cos_sin_yarn;
};

// Load `Olmo2ForCausalLM` (OLMo-2-0425-1B) safetensors into Olmo2Weights. Name map
// (flat config, NO multimodal prefix):
//   model.embed_tokens.weight                          -> embed_tokens [V,H]
//   model.norm.weight                                  -> final_norm [H]
//   lm_head.weight                                     -> lm_head (untied only)
//   model.layers.N.post_attention_layernorm.weight     -> post_attention_ln [H]
//   model.layers.N.post_feedforward_layernorm.weight   -> post_feedforward_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight       -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.{q,k}_norm.weight         -> q_norm[q_size]/k_norm[kv_size]
//   model.layers.N.self_attn.o_proj.weight             -> o_proj (raw-NK, no bias)
//   model.layers.N.mlp.{gate,up}_proj.weight           -> merged gate_up_proj (raw-NK)
//   model.layers.N.mlp.down_proj.weight                -> down_proj (raw-NK)
// The checkpoint's lm_head.weight is intentionally SKIPPED when tie_word_embeddings.
// Reuses the shared dense_weight_loaders.h helpers. Text path only.
Olmo2Weights LoadOlmo2ForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                         const HfConfig& config);

// The DENSE OLMo-2 (`Olmo2ForCausalLM`) forward. Per decoder layer (olmo2.py
// Olmo2DecoderLayer.forward @ e24d1b24), PURE POST-NORM:
//   residual = h
//   h = self_attn(h)                     # qkv on RAW h (no input norm); full-width
//                                        #   q/k RMSNorm -> NeoX RoPE -> FA2 paged -> o_proj
//   h = post_attention_layernorm(h)      # STANDALONE norm on the attn output
//   h = h + residual                     # PLAIN residual add
//   residual = h
//   h = mlp(h)                           # SwiGLU on RAW h (no pre-FF norm)
//   h = post_feedforward_layernorm(h)    # STANDALONE norm on the mlp output
//   h = residual + h                     # PLAIN residual add
// Then final_norm (standalone RMSNorm) -> lm_head (tied to embed_tokens, or
// untied). The residual stream, qkv, rope, KV cache, attention and MLP all flow
// bf16 (mirroring vLLM's per-op bf16 stores; head_dim -> FLASH_ATTN/FA2, matching
// the Qwen3-dense/GLM-4 token-exact numeric contract). The standalone RMSNorms
// compute variance in f32 and round to bf16; the plain adds compute in f32 and
// round to bf16. Returns [n_out, vocab] f32 logits.
class Olmo2Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Olmo2Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Olmo2Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// Per-family config hook (mirrors ParseGlm4ForCausalLMConfig). LoadHfConfig already
// materializes every consumed OLMo-2 field (num_key_value_heads, head_dim,
// rope_theta, rotary_dim, intermediate_size, rms_norm_eps). Validates the OLMo-2
// invariant: FULL (non-partial) NeoX rope — rotary_dim == head_dim.
void ParseOlmo2ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group (OLMo-2 dense has no
// linear-attention / MambaSpec group), mirroring MakeGlm4ForCausalLMKVCache. (The
// Olmo-3 interleaved sliding-window variant reuses the Gemma-3 per-layer routing —
// out of scope for this dense bring-up, W5.)
v1::KVCacheConfig MakeOlmo2ForCausalLMKVCache(const HfConfig& config, int block_size,
                                              int num_blocks);

}  // namespace vllm
