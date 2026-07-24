// Phi-3 / Phi-4 dense (`Phi3ForCausalLM`, microsoft/Phi-4-mini-instruct) — a
// ZERO-NEW-KERNEL dense bring-up: the Llama dense forward VERBATIM with two loader/
// rope deltas. (Upstream: vllm/model_executor/models/phi3.py @ e24d1b24 — a bare
// LlamaForCausalLM subclass whose ONLY override is packed_modules_mapping; config
// microsoft/Phi-4-mini-instruct/config.json.)
//
// Phi-3 IS Llama, with exactly two deltas, both handled here:
//   1. PRE-FUSED checkpoints (phi3.py packed_modules_mapping): the checkpoint ships
//      ALREADY-MERGED self_attn.qkv_proj.weight ([q|k|v] rows) and mlp.gate_up_proj
//      .weight ([gate|up] rows) — so the loader loads them DIRECTLY as single raw-NK
//      owners instead of merging separate q/k/v and gate/up tensors.
//   2. LongRoPE + PARTIAL rotary (Phi-4-mini): rope_scaling type "longrope",
//      partial_rotary_factor 0.75 (rotary_dim = head_dim*0.75 = 96 of 128). vLLM
//      builds a per-position cos/sin cache with the LONG rescale factors + an mscale
//      (max_position 131072 > original 4096 -> use_long_rope). We precompute that
//      exact cache once at load (Phi3LongRoPEScaledRotaryEmbedding, the pinned CPU
//      class) and feed it to RopeFromCache indexed by the REAL positions — the
//      dense-path MakeRopeArgs only knows default/llama3, so LongRoPE cannot ride it.
//
// Everything else — pre-norm fused add+RMSNorm, GQA, standard 1/sqrt(Dh) attention
// scale, SwiGLU MLP, standard (non-gemma) RMSNorm, tied lm_head, NO qk-norm, NO
// biases, the full-attention paged path (sliding_window 262144 is inert for the
// gate battery) — is REUSED from the shared dense substrate. The forward is written
// FRESH (like OLMo-2) only because the LongRoPE cache must replace the built-in one;
// it reuses all the shared device glue. See .agents/specs/sweep-recent-dense-batch.md
// §0.2 row 1.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"  // shared dense weights container
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// Phi-3 weights: the shared dense container (tied lm_head, no qk-norm, PRE-FUSED
// qkv/gate_up) plus the precomputed LongRoPE cos/sin cache (bf16 [P, rotary_dim],
// [cos|sin] halves) built at load from the checkpoint's rope_scaling. When the
// checkpoint uses plain rope, rope_cos_sin is the plain per-position cache.
struct Phi3Weights {
  Qwen3DenseWeights dense;
  OwnedTensor rope_cos_sin;  // bf16 [P, rotary_dim]; row p = angle for position p
};

// Load `Phi3ForCausalLM` (Phi-4-mini-instruct, BF16) safetensors + build the rope
// cache. Name map (flat): model.embed_tokens.weight, model.norm.weight,
// lm_head.weight (SKIPPED when tie_word_embeddings), per layer
// model.layers.N.{input_layernorm,post_attention_layernorm}.weight,
// .self_attn.qkv_proj.weight (PRE-FUSED), .self_attn.o_proj.weight,
// .mlp.gate_up_proj.weight (PRE-FUSED), .mlp.down_proj.weight. Text path only.
Phi3Weights LoadPhi3ForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                       const HfConfig& config);

// The Phi-3 dense forward (pre-norm Llama with the LongRoPE cache). Per layer:
//   input_layernorm (std fused add+RMSNorm) -> qkv_proj (BF16 MatmulBT) -> split
//   -> RoPE from the precomputed LongRoPE cache (real positions, partial rotary_dim)
//   -> FA2 causal paged attention (scale 1/sqrt(Dh)) -> o_proj ->
//   post_attention_layernorm (std fused add+RMSNorm) -> gate_up_proj -> SiluAndMul
//   -> down_proj. Then final_norm -> lm_head (tied). bf16 residual/qkv/rope/KV/attn/
//   mlp; norms compute variance in f32 and round to bf16. Returns [n_out, vocab] f32.
class Phi3Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Phi3Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Phi3Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// Per-family config hook. Validates the Phi-3 rope invariant (rope_type default or
// longrope; rotary_dim in (0, head_dim]).
void ParsePhi3ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group (Phi-3's very large
// sliding_window is inert for the gate; masked at the kernel, no separate spec).
v1::KVCacheConfig MakePhi3ForCausalLMKVCache(const HfConfig& config, int block_size,
                                             int num_blocks);

}  // namespace vllm
