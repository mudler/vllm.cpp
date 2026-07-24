// Gemma-1 text (`GemmaForCausalLM`) — sweep W5, the original Gemma. The SIMPLEST
// Gemma: a plain Llama-style pre-norm decoder (two fused add+RMSNorm per layer,
// NO sandwich norms) with the Gemma vocabulary — GeGLU (gelu_pytorch_tanh) MLP,
// sqrt(hidden) embed-scale, tied lm_head, GemmaRMSNorm (1+w) — but NO QK-norm, NO
// soft-cap, NO sliding window, a SINGLE RoPE theta, and `head_dim**-0.5`
// attention scaling (not query_pre_attn_scalar).
//
// Upstream: vllm/model_executor/models/gemma.py @ e24d1b24 (GemmaMLP :86-115
// [act _get_gemma_act_fn :58-84], GemmaAttention :118-198 [scaling :158, no
// soft-cap/qk-norm/sliding], GemmaDecoderLayer :201-256 [two fused norms
// :246-253], GemmaModel embed-scale :288-295, GemmaForCausalLM tie :338-339 +
// plain LogitsProcessor :341); layernorm.py::GemmaRMSNorm (1+w, fp32);
// activation.py::GeluAndMul("tanh"). Config gemma-2b/config.json (hidden 2048,
// 18L, MQA 8/1, head_dim 256, intermediate 16384, rope_theta 1e4).
//
// Numeric contract mirrors the bf16 dense path (gemma2.cpp / dense_attn_block.h).
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

// One Gemma-1 self-attention block (gemma.py::GemmaAttention). Merged QKV, NO q/k
// norm, NeoX RoPE (single theta), scaling head_dim**-0.5, no soft-cap, no bias.
struct GemmaAttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v)
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh]
};

// Gemma-1 GeGLU MLP (gemma.py::GemmaMLP): merged gate_up -> GeluAndMul(tanh) ->
// down_proj. Raw-NK.
struct GemmaMlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up)
  OwnedTensor down_proj;     // bf16 raw-NK [H, I]
};

// One Gemma-1 decoder layer: TWO GemmaRMSNorms, BOTH fused add+RMSNorm (the plain
// pre-norm pattern; NO standalone sandwich norms).
struct GemmaLayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]  (fused add)
  OwnedTensor post_attention_layernorm;  // bf16 [H]  (fused add)
  GemmaAttnWeights attn;
  GemmaMlpWeights mlp;
};

// Whole Gemma-1 text-model weights. tie_word_embeddings asserted TRUE
// (gemma.py:335); the checkpoint has no lm_head.weight.
struct GemmaWeights {
  bool tie_word_embeddings = true;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (embed lookup; NOT scaled)
  OwnedTensor final_norm;    // bf16 [H]  (model.norm, GemmaRMSNorm)
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY when tied
  std::vector<GemmaLayerWeights> layers;
};

// Load `GemmaForCausalLM` (gemma-2b) safetensors into GemmaWeights. Name map
// (model.layers.N.*): input_layernorm, post_attention_layernorm,
// self_attn.{q,k,v,o}_proj (q/k/v merged), mlp.{gate,up,down}_proj (gate/up
// merged); model.embed_tokens, model.norm; lm_head.weight skipped when tied.
GemmaWeights LoadGemmaForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The Gemma-1 dense forward. Per decoder layer (gemma.py::GemmaDecoderLayer):
//   input_layernorm (fused add, Gemma) -> attn(qkv, NeoX RoPE, head_dim^-0.5
//   paged attention, o_proj) -> post_attention_layernorm (fused add, Gemma) ->
//   GeGLU MLP. Then model.norm (fused add, Gemma) -> lm_head (tied). Token
//   embeddings are scaled by sqrt(hidden) (bf16) before the first layer. Returns
//   [n_out, vocab] f32 logits (no soft-cap).
class GemmaModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const GemmaWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const GemmaWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

void ParseGemmaForCausalLMConfig(const HfConfig& config);

v1::KVCacheConfig MakeGemmaForCausalLMKVCache(const HfConfig& config,
                                              int block_size, int num_blocks);

}  // namespace vllm
