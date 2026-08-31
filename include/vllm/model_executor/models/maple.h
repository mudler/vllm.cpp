// Maple (`MapleForCausalLM`) — deepgrove's 20B-A1B ternary MoE. The port
// mirrors the qwen3_moe bring-up shape: registry TU + weights + forward, all
// composing the shared dense attention block and the exposed MoE block.
//
// Grounding (the fork is the ONLY reference implementation):
//   deepgrove-ai/llama.cpp src/models/maple.cpp @ 7e30f3a (graph build),
//   HF deepgrove/maple-preview config.json (hyperparameters).
//
// Architecture deltas vs Qwen3-MoE, each cited to its source:
//   1. LAYER PATTERN — layer_types = [sliding, sliding, sliding, full] repeating:
//      full attention at layer indices {3,7,11,15,19,23} of 24; the other 18 are
//      SWA with window 512 (config.json `layer_types` + `sliding_window`).
//   2. ROPE INVERSION — nope_on_global_attention=true: RoPE applies ONLY on the
//      SWA layers; the 6 global layers take NO RoPE at all (maple.cpp:96-106,
//      `if (hparams.is_swa(il)) { rope(Q); rope(K); }`). This is the inverse of
//      every standard arch (which ropes exactly the global layers).
//   3. PARTIAL ROPE — partial_rotary_factor=0.5: rotary_dim = 64 of head_dim 128.
//   4. CLAMPED SWIGLU — swiglu_clamp_exp = 7.0 per layer: the expert SwiGLU
//      activation is clamped to ±7 BEFORE the up multiply (maple.cpp:21-22
//      `swiglu_clamp_exp.fill(7.0f)` + build_moe_ffn clamp plumbing). Standard
//      RunMoeBlock does not clamp, so maple runs its own expert MLP body.
//   5. TERNARY EXPERTS — ffn_gate/up/down_exps towers are TQ1_0 or TQ2_0
//      block-quantized; they route keep-quant (Q8_K activations) like any
//      K-quant super-block.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"             // Qwen3DenseAttnWeights, PagedKvCache
#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor, MoeBlockWeights
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class GgufFile;
struct GgufLoadPolicy;

// One maple decoder layer. Attention REUSES Qwen3DenseAttnWeights (merged qkv +
// per-head q/k RMSNorm + o_proj — maple has exactly those pieces, no biases);
// the MoE block REUSES MoeBlockWeights (router + stacked experts, no shared).
struct MapleLayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  Qwen3DenseAttnWeights attn;
  MoeBlockWeights moe;

  bool is_swa = false;  // true for the 18 sliding-window layers (window 512);
                        // false for the 6 global layers {3,7,11,15,19,23},
                        // which additionally get NO RoPE.
};

struct MapleWeights {
  bool tie_word_embeddings = false;
  bool attention_bias = false;
  OwnedTensor embed_tokens;  // bf16 [vocab, H] gather table
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B, untied
  std::vector<MapleLayerWeights> layers;
};

// Batched PAGED forward. token_ids/positions are the flattened length-T step
// inputs; attn_meta the common attention metadata; attn_kv one PagedKvCache per
// layer (all layers allocate a cache slot; the SWA/global distinction is a
// windowing + rope concern inside the attention call, not a cache-topology one).
// Returns [T, vocab] f32 logits.
std::vector<float> MapleForward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const MapleWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices = {});

// Per-family config hook (ModelFactory::parse_config): validates the MoE fields
// the loader/forward consume.
void ParseMapleConfig(const HfConfig& config);

// KV-cache spec builder: ONE full-attention group, no Mamba/GDN (clone of
// MakeQwen3MoeKVCache — the runner's full-attention-only path covers it).
v1::KVCacheConfig MakeMapleKVCache(const HfConfig& config, int block_size,
                                   int num_blocks);

// GGUF path. `IsMapleGguf` gates on general.architecture == "maple";
// `MapleHfConfigFromGguf` maps the maple.* keys onto HfConfig;
// `LoadMapleFromGguf` builds the whole-model weights (experts stay
// block-quantized via keep-quant when available).
bool IsMapleGguf(const GgufFile& gguf);
HfConfig MapleHfConfigFromGguf(const GgufFile& gguf);
MapleWeights LoadMapleFromGguf(const GgufFile& gguf, const HfConfig& config,
                               const GgufLoadPolicy* policy);

}  // namespace vllm
