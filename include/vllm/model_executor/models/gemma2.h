// Gemma-2 text (`Gemma2ForCausalLM`) — sweep W4. A dense decoder with the Gemma
// vocabulary that is the INVERSE of Gemma-3 on the attention delta: Gemma-2 has
// BOTH an attention logit soft-cap AND a final logit soft-cap and NO QK-norm,
// where Gemma-3 removed the soft-caps and added QK-norm. It shares Gemma-3's
// sandwich-norm layout, GeGLU MLP, sqrt(hidden) embed-scale, tied lm_head,
// interleaved sliding window, and `query_pre_attn_scalar**-0.5` attention
// scaling, but uses a SINGLE per-model RoPE theta (no dual local/global cache).
//
// Upstream: vllm/model_executor/models/gemma2.py @ e24d1b24 (Gemma2MLP :57-90,
// Gemma2Attention :93-176 [scaling :129, attn soft-cap :106,165,202, sliding
// :154-166], Gemma2DecoderLayer sandwich :180-245, Gemma2Model embed-scale
// :276-283, Gemma2ForCausalLM tie :338-339 + final soft-cap :344-345);
// layernorm.py::GemmaRMSNorm (1+w, fp32); activation.py::GeluAndMul("tanh").
// Config gemma-2-2b-it/config.json (hidden 2304, 26L, GQA 8/4, head_dim 256,
// intermediate 9216, sliding_window 4096, query_pre_attn_scalar 256,
// attn_logit_softcapping 50.0, final_logit_softcapping 30.0, rope_theta 1e4).
//
// Numeric contract mirrors the bf16 dense path (gemma3.cpp / dense_attn_block.h):
// the residual stream is bf16; the qkv GEMM, RoPE, flash attention (with the
// attention logit soft-cap applied on the scaled pre-softmax score) and the whole
// MLP flow bf16; the paged KV cache is bf16; the final f32 logits are soft-capped
// (cap * tanh(logits/cap)) for faithfulness (greedy argmax is soft-cap-invariant).
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

// One Gemma-2 self-attention block (gemma2.py::Gemma2Attention). Merged QKV
// (packed_modules_mapping qkv_proj<-[q,k,v]_proj), NO q/k norm (the inverse of
// Gemma-3), NeoX RoPE (single per-model theta), an ATTENTION logit soft-cap
// (attn_logit_softcapping), no bias (attention_bias=false).
struct Gemma2AttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v)
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh]
};

// Gemma-2 GeGLU MLP (gemma2.py::Gemma2MLP): merged gate_up (packed
// gate_up_proj<-[gate,up]) -> GeluAndMul(tanh) -> down_proj. Raw-NK.
struct Gemma2MlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up)
  OwnedTensor down_proj;     // bf16 raw-NK [H, I]
};

// One Gemma-2 decoder layer. FOUR GemmaRMSNorms in the sandwich layout
// (gemma2.py:213-245), identical to Gemma-3: input + pre_feedforward are fused
// add+RMSNorm; post_attention + post_feedforward are standalone norms on the
// sublayer output before it re-enters the residual.
struct Gemma2LayerWeights {
  OwnedTensor input_layernorm;            // bf16 [H]  (fused add)
  OwnedTensor post_attention_layernorm;   // bf16 [H]  (standalone, sandwich)
  OwnedTensor pre_feedforward_layernorm;  // bf16 [H]  (fused add)
  OwnedTensor post_feedforward_layernorm; // bf16 [H]  (standalone, sandwich)
  Gemma2AttnWeights attn;
  Gemma2MlpWeights mlp;
};

// Whole Gemma-2 text-model weights. tie_word_embeddings is asserted TRUE for
// every Gemma model (gemma2.py:335); the checkpoint has no lm_head.weight.
struct Gemma2Weights {
  bool tie_word_embeddings = true;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (embed lookup; NOT scaled)
  OwnedTensor final_norm;    // bf16 [H]  (model.norm, GemmaRMSNorm)
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY when tied
  std::vector<Gemma2LayerWeights> layers;
};

// Load `Gemma2ForCausalLM` (gemma-2-2b-it) safetensors into Gemma2Weights. Name
// map (model.layers.N.*): input_layernorm, post_attention_layernorm,
// pre_feedforward_layernorm, post_feedforward_layernorm, self_attn.{q,k,v,o}_proj
// (q/k/v merged); model.embed_tokens, model.norm; the checkpoint's lm_head.weight
// is absent/skipped when tied (vLLM skip_prefixes=["lm_head."]). BF16 text only.
Gemma2Weights LoadGemma2ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The Gemma-2 dense forward. Per decoder layer (gemma2.py::Gemma2DecoderLayer):
//   input_layernorm (fused add, Gemma) -> attn(qkv, NeoX RoPE, qpas-scaled paged
//   attention with the attention logit soft-cap + per-layer sliding window,
//   o_proj) -> post_attention_layernorm (standalone Gemma) ->
//   pre_feedforward_layernorm (fused add, Gemma) -> GeGLU MLP ->
//   post_feedforward_layernorm (standalone Gemma). Then model.norm (fused add,
//   Gemma) -> lm_head (tied) -> final logit soft-cap. Token embeddings are scaled
//   by sqrt(hidden) (bf16) before the first layer. Returns [n_out, vocab] f32.
class Gemma2Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma2Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma2Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// Per-family config hook (mirrors ParseQwen3ForCausalLMConfig). No-op: the
// consumed Gemma scalars (query_pre_attn_scalar, attn_logit_softcapping,
// final_logit_softcapping, sliding_window, layer_types) are read from config.raw.
void ParseGemma2ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: one FULL-ATTENTION KV group covering all layers. The
// interleaved sliding-window layers are masked at the attention-kernel level
// (per-layer window_size), not by a smaller cache — the pure-dense full-attention
// topology the shape-agnostic runner already handles.
v1::KVCacheConfig MakeGemma2ForCausalLMKVCache(const HfConfig& config,
                                               int block_size, int num_blocks);

}  // namespace vllm
