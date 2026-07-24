// GLM-4-0414 dense (`Glm4ForCausalLM`, GLM-4-9B-0414, BF16) — the first
// GLM-family model (spike .agents/specs/glm-dsa-latest-deepseek.md, task G2/W2).
//
// Upstream: vllm/model_executor/models/glm4.py @ e24d1b24; config
// zai-org/GLM-4-9B-0414/config.json (hidden 4096, 40 layers, 32 q-heads,
// 2 kv-heads, head_dim 128, attention_bias TRUE, partial_rotary_factor 0.5,
// is_neox_style FALSE, rope_theta 1e4, rms_norm_eps 1e-5, tie_word_embeddings
// FALSE, NO qk-norm). Structurally a standard dense transformer with exactly two
// GLM deltas over the Qwen3-dense template (qwen3.h):
//   1. PARTIAL + INTERLEAVED RoPE: rope only the leading rotary_dim = 0.5*head_dim
//      = 64 slice, in the adjacent-pair (GPT-J, is_neox_style=false) layout,
//      passing the trailing 64 dims through bit-exactly. This reuses the EXISTING
//      RopeFromCache primitive (cuda_ops.cu:656-703 / cpu_ops.cpp:713-746), which
//      already honors both partial rotary_dim and is_neox_style — the same
//      interleaved path DeepSeek-V2's decoupled RoPE is gated on. NEW: nothing but
//      the is_neox_style=false selection at the call site.
//   2. SANDWICH NORMS (Gemma2 pattern, glm4.py:180-187,206,211): the decoder layer
//      has FOUR RMSNorms — input_layernorm, post_attention_layernorm PLUS
//      post_self_attn_layernorm and post_mlp_layernorm applied to the sublayer
//      OUTPUT before the residual add. These are STANDALONE vt::RmsNorm calls (no
//      residual accumulation) on the attn/mlp output; the existing op covers them.
// Plus GQA with BIASED qkv (attention_bias=true → qkv_proj carries a bias, o_proj
// does not) and an UNTIED lm_head. No NVFP4 (GLM-4-9B-0414 is bf16-only).
//
// The self-attention block is GLM-specific (biased qkv + partial non-NeoX rope +
// NO qk-norm violate dense_attn_block.h's asserts), so it is written fresh in
// glm4.cpp reusing ONLY the shared glue (Dev/DBuf/ResidentWeight/KvSlice/
// StepInputs/BuildStepInputs), per the OPT precedent (spike §"Our baseline", D4).
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

// One GLM-4 self-attention block (glm4.py::Glm4Attention). Merged QKV
// (packed_modules_mapping qkv_proj<-[q,k,v]) with a BIAS (attention_bias=true),
// NO per-head q/k norm, partial + interleaved RoPE, and an unbiased o_proj. Raw-NK
// torch Linear [N=out, K=in] orientation for vt::MatmulBT.
struct Glm4AttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  OwnedTensor qkv_bias;  // bf16 [Hq*Dh + 2*Hkv*Dh] (attention_bias=true → present)
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh], nk (no bias)
};

// GLM-4 SwiGLU MLP (glm4.py Glm4MLP = LlamaMLP): merged gate_up (gate_up_proj<-
// [gate,up]) -> SiluAndMul -> down_proj. Raw-NK. No bias.
struct Glm4MlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up), nk
  OwnedTensor down_proj;     // bf16 raw-NK [H, I], nk
};

// One GLM-4 decoder layer: FOUR standard (non-gemma) RMSNorms — the sandwich
// pattern (glm4.py:180-187). input_layernorm / post_attention_layernorm are the
// fused residual-add norms; post_self_attn_layernorm / post_mlp_layernorm are the
// standalone output norms applied to the attn / mlp sublayer output before the
// residual add.
struct Glm4LayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  OwnedTensor post_self_attn_layernorm;  // bf16 [H]  (sandwich, attn output)
  OwnedTensor post_mlp_layernorm;        // bf16 [H]  (sandwich, mlp output)
  Glm4AttnWeights attn;
  Glm4MlpWeights mlp;
};

// Whole GLM-4 dense text-model weights. GLM-4-9B-0414 has
// tie_word_embeddings=false, so `lm_head` holds the ParallelLMHead weight in
// Matmul-B [H, vocab] layout (a real checkpoint tensor, not aliased).
struct Glm4Weights {
  bool tie_word_embeddings = false;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (embed lookup, NOT transposed)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY only if tied
  std::vector<Glm4LayerWeights> layers;
};

// Load `Glm4ForCausalLM` (GLM-4-9B-0414) safetensors into Glm4Weights. Name map
// (flat config, NO multimodal prefix):
//   model.embed_tokens.weight, model.norm.weight, lm_head.weight (untied),
//   per layer model.layers.N.{input_layernorm,post_attention_layernorm,
//     post_self_attn_layernorm,post_mlp_layernorm}.weight,
//   .self_attn.{q,k,v}_proj.{weight,bias} (merged qkv + bias),
//   .self_attn.o_proj.weight, .mlp.{gate,up}_proj.weight (merged), .mlp.down_proj.weight.
// The speculative (MTP) tail layers model.layers.{num_hidden_layers+i} are
// SKIPPED (glm4.py:295-305,308) — the loader only materializes [0,num_hidden_layers).
Glm4Weights LoadGlm4ForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                       const HfConfig& config);

// The DENSE GLM-4 (`Glm4ForCausalLM`) forward. Per decoder layer (glm4.py
// Glm4DecoderLayer.forward @ e24d1b24):
//   input_layernorm (std add+RMSNorm) -> BIASED qkv (MatmulBT + bias) -> split
//   q/k/v -> partial INTERLEAVED RoPE (rotary_dim 64, is_neox_style=false, no
//   qk-norm) -> FA2 causal paged attention -> o_proj (no bias)
//   -> post_self_attn_layernorm (STANDALONE norm on attn output)
//   -> post_attention_layernorm (std add+RMSNorm) -> gate_up -> SiluAndMul ->
//   down_proj -> post_mlp_layernorm (STANDALONE norm on mlp output).
// Then final_norm (std add+RMSNorm) -> lm_head (untied Matmul). The residual
// stream, qkv, rope, KV cache, attention and MLP all flow bf16 (mirroring vLLM's
// per-op bf16 stores; head_dim 128 → FLASH_ATTN/FA2, matching Qwen3-dense's
// token-exact numeric contract). Returns [n_out, vocab] f32 logits.
class Glm4Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Glm4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Glm4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// Per-family config hook (mirrors ParseQwen3ForCausalLMConfig). LoadHfConfig
// already materializes every consumed GLM-4 field (num_key_value_heads, head_dim,
// rope_theta, rotary_dim from partial_rotary_factor, intermediate_size,
// rms_norm_eps); this hook validates the GLM invariants (partial rotary present).
void ParseGlm4ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group (GLM-4 dense has no
// linear-attention / MambaSpec group), mirroring MakeQwen3ForCausalLMKVCache.
v1::KVCacheConfig MakeGlm4ForCausalLMKVCache(const HfConfig& config, int block_size,
                                             int num_blocks);

}  // namespace vllm
