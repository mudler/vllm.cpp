// Gemma-3 text (`Gemma3ForCausalLM`) — the first Gemma-family bring-up (sweep
// W2). A dense decoder with the Gemma vocabulary: GemmaRMSNorm (1+w) everywhere,
// the Gemma-2/3 SANDWICH norm layout (a standalone norm on each sublayer output
// before it re-enters the residual, exactly the GLM-4 pattern), per-head Gemma
// q/k RMSNorm before RoPE, GeGLU (gelu_pytorch_tanh) MLP, a sqrt(hidden) embed
// normalizer, `query_pre_attn_scalar**-0.5` attention scaling, DUAL per-layer
// RoPE theta (global rope_theta on full-attn layers, rope_local_base_freq on
// sliding layers) + interleaved sliding-window routing, and tied lm_head.
//
// Upstream: vllm/model_executor/models/gemma3.py @ e24d1b24 (Gemma3MLP :63-93,
// Gemma3Attention :96-221 [scaling :130, q/k-norm :149-150,205-213, dual rope
// :152-176], Gemma3DecoderLayer sandwich :265-289, Gemma3Model embed-scale
// :328-341, Gemma3ForCausalLM tie+soft-cap :411-416); layernorm.py::GemmaRMSNorm
// (1+w, fp32); activation.py::GeluAndMul(approximate="tanh"). Config
// gemma-3-1b-it/config.json (hidden 1152, 26L, GQA 4/1, head_dim 256,
// intermediate 6912, sliding_window 512, sliding_window_pattern 6,
// query_pre_attn_scalar 256, rope_theta 1e6, rope_local_base_freq 1e4,
// final_logit_softcapping null). See .agents/specs/sweep-gemma.md.
//
// Numeric contract mirrors the bf16 dense path (qwen3.cpp / dense_attn_block.h):
// the residual stream is bf16 (matching vLLM's fused_add_rms_norm residual); the
// qkv GEMM, per-head Gemma q/k RMSNorm, RoPE, flash attention and the whole MLP
// flow bf16 (matching vLLM's per-op bf16 stores); the paged KV cache is bf16.
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

// One Gemma-3 self-attention block (gemma3.py::Gemma3Attention). Merged QKV
// (packed_modules_mapping qkv_proj<-[q,k,v]_proj), per-head q/k Gemma-RMSNorm
// (GemmaRMSNorm(head_dim), 1+w) applied BEFORE RoPE, no attention output gate,
// no attention logit soft-cap, no bias (attention_bias=false). Projections are
// kept RAW in the on-disk [N=out, K=in] orientation (nk), consumed by MatmulBT.
struct Gemma3AttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v)
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh]
  OwnedTensor q_norm;    // bf16 [head_dim]  (per-head Gemma-RMSNorm, 1+w)
  OwnedTensor k_norm;    // bf16 [head_dim]
};

// Gemma-3 GeGLU MLP (gemma3.py::Gemma3MLP): merged gate_up (packed
// gate_up_proj<-[gate,up]) -> GeluAndMul(tanh) -> down_proj. Raw-NK.
struct Gemma3MlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up)
  OwnedTensor down_proj;     // bf16 raw-NK [H, I]
};

// One Gemma-3 decoder layer. FOUR GemmaRMSNorms in the sandwich layout
// (gemma3.py:254-289): input + pre_feedforward are fused add+RMSNorm (they touch
// the residual stream); post_attention + post_feedforward are standalone norms
// on the sublayer output before it re-enters the residual.
struct Gemma3LayerWeights {
  OwnedTensor input_layernorm;            // bf16 [H]  (fused add)
  OwnedTensor post_attention_layernorm;   // bf16 [H]  (standalone, sandwich)
  OwnedTensor pre_feedforward_layernorm;  // bf16 [H]  (fused add)
  OwnedTensor post_feedforward_layernorm; // bf16 [H]  (standalone, sandwich)
  Gemma3AttnWeights attn;
  Gemma3MlpWeights mlp;
};

// Whole Gemma-3 text-model weights. tie_word_embeddings defaults TRUE for Gemma
// (lm_head aliases embed_tokens; the checkpoint has no lm_head.weight).
struct Gemma3Weights {
  bool tie_word_embeddings = true;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (embed lookup; NOT scaled)
  OwnedTensor final_norm;    // bf16 [H]  (model.norm, GemmaRMSNorm)
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY when tied
  std::vector<Gemma3LayerWeights> layers;
};

// Load `Gemma3ForCausalLM` (gemma-3-1b-it) safetensors into Gemma3Weights. Name
// map (model.layers.N.*): input_layernorm, post_attention_layernorm,
// pre_feedforward_layernorm, post_feedforward_layernorm, self_attn.{q,k,v,o}_proj
// (q/k/v merged), self_attn.{q,k}_norm, mlp.{gate,up,down}_proj (gate/up merged);
// model.embed_tokens, model.norm; the checkpoint's lm_head.weight is absent/
// skipped when tied (vLLM skip_prefixes=["lm_head."]). BF16 text path only.
Gemma3Weights LoadGemma3ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The Gemma-3 dense forward. Per decoder layer (gemma3.py::Gemma3DecoderLayer):
//   input_layernorm (fused add, Gemma) -> attn(qkv, per-head Gemma q/k norm,
//   DUAL-theta NeoX RoPE, qpas-scaled paged attention w/ per-layer sliding
//   window, o_proj) -> post_attention_layernorm (standalone Gemma) ->
//   pre_feedforward_layernorm (fused add, Gemma) -> GeGLU MLP ->
//   post_feedforward_layernorm (standalone Gemma). Then model.norm (fused add,
//   Gemma) -> lm_head (tied). The token embeddings are scaled by sqrt(hidden)
//   (bf16) before the first layer. Returns [n_out, vocab] f32 logits.
class Gemma3Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma3Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma3Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// Per-family config hook (mirrors ParseQwen3ForCausalLMConfig). No-op: the
// consumed Gemma scalars (query_pre_attn_scalar, rope_local_base_freq,
// sliding_window_pattern, final_logit_softcapping) are read from config.raw by
// the forward/loader.
void ParseGemma3ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: one FULL-ATTENTION KV group covering all layers. The
// sliding-window layers are masked at the attention-kernel level (window_size),
// not by a smaller cache — a memory-only vLLM optimization not needed for
// correctness. This is the pure-dense full-attention KV topology the
// shape-agnostic runner already handles.
v1::KVCacheConfig MakeGemma3ForCausalLMKVCache(const HfConfig& config,
                                               int block_size, int num_blocks);

}  // namespace vllm
