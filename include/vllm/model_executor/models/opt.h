// OPT (`OPTForCausalLM`) — the CROSS-FAMILY additivity canary of the breadth
// sweep (.agents/specs/breadth-sweep-plan.md §B.3 rank 4;
// .agents/specs/sweep-opt-125m.md).
//
// Upstream: vllm/model_executor/models/opt.py @ e24d1b24 (registered at
// registry.py:176); config facebook/opt-125m/config.json.
//
// OPT is deliberately the first model we port that shares almost NOTHING with
// the Qwen families already in the tree. Every one of these is a first:
//   * NO RoPE — LEARNED absolute position embeddings, with the fairseq
//     padding-idx OFFSET OF 2 (`OPTLearnedPositionalEmbedding`, opt.py:59-68;
//     the checkpoint's embed_positions.weight is [max_pos + 2, H]).
//   * ATTENTION BIAS — q/k/v/out_proj all carry bias (`config.enable_bias`,
//     opt.py:90-104); every Qwen projection we had was bias-free.
//   * LAYERNORM, not RMSNorm — mean-subtracting, with a bias term
//     (opt.py:146-148,164-166,248-251), and NO per-head q/k norm.
//   * RELU + a plain fc1/fc2 MLP (opt.py:149-163), not SwiGLU gate/up/down.
//   * `do_layer_norm_before` — a config switch between PRE-LN (125m/1.7B/…/175B)
//     and POST-LN (350m) placement (opt.py:175-194).
//   * `word_embed_proj_dim != hidden_size` optionally inserts project_in /
//     project_out (opt.py:220-241).
//   * Tied embeddings by DEFAULT (opt.py:352-359).
//
// Structure mirrors the Qwen3-dense bring-up (qwen3.h): this header carries the
// weight PODs + the registry-facing config/KV-cache hooks; the loader lands in
// opt_weights.cpp, the forward in opt.cpp, the registry TU in opt_registry.cpp.
#pragma once

#include <cstdint>
#include <string>
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

// The OPT fields that live OUTSIDE our typed HfConfig. OPT names its FFN width
// `ffn_dim` (not `intermediate_size`) and adds five family-specific switches, so
// they are read from `HfConfig::raw` by ParseOPTConfig rather than by widening
// the shared HfConfig POD — the `raw` escape hatch is exactly the seam that lets
// a new family land without touching shared config code
// (.agents/specs/sweep-opt-125m.md "seams that held" S2).
//
// Defaults mirror transformers `OPTConfig` for keys absent from the checkpoint
// (facebook/opt-125m omits enable_bias / layer_norm_elementwise_affine /
// _remove_final_layer_norm / tie_word_embeddings, all of which default true,
// true, false, true).
struct OPTConfigExtras {
  int64_t ffn_dim = 0;
  int64_t word_embed_proj_dim = 0;
  bool do_layer_norm_before = true;
  bool enable_bias = true;
  bool layer_norm_elementwise_affine = true;
  bool remove_final_layer_norm = false;
  bool tie_word_embeddings = true;
  std::string activation_function = "relu";
  // torch `nn.LayerNorm` default eps; OPT never overrides it in config.json, and
  // upstream constructs nn.LayerNorm without an eps argument
  // (opt.py:146-148,164-166,248-251), so this constant IS the upstream value.
  float layer_norm_eps = 1e-5F;
};

// Reads OPTConfigExtras out of `config.raw`. Also used by the loader/forward, so
// the "what does this checkpoint say" logic exists exactly once.
OPTConfigExtras GetOPTConfigExtras(const HfConfig& config);

// One OPT self-attention block (opt.py::OPTAttention, :71-122). Merged q|k|v in
// the on-disk torch Linear [N=out, K=in] orientation (nk=true) consumed by
// vt::MatmulBT — the same physical ownership rule as vLLM's QKVParallelLinear
// (one merged param) and the same layout choice the Qwen3-dense loader makes.
// UNLIKE every Qwen block: bias on BOTH projections, and NO q/k norm, NO RoPE.
struct OPTAttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [3*H, H] (rows q|k|v), nk
  OwnedTensor qkv_bias;  // bf16 [3*H]  (merged q|k|v bias; enable_bias)
  OwnedTensor out_proj;  // bf16 raw-NK [H, H], nk
  OwnedTensor out_bias;  // bf16 [H]
};

// OPT's plain two-layer MLP (opt.py:149-163): fc1 -> ReLU -> fc2, both biased.
// No gate/up merge exists to do — there is no gating at all.
struct OPTMlpWeights {
  OwnedTensor fc1;       // bf16 raw-NK [ffn_dim, H], nk
  OwnedTensor fc1_bias;  // bf16 [ffn_dim]
  OwnedTensor fc2;       // bf16 raw-NK [H, ffn_dim], nk
  OwnedTensor fc2_bias;  // bf16 [H]
};

// One OPT decoder layer (opt.py::OPTDecoderLayer, :125-195). Both LayerNorms
// carry weight AND bias (elementwise_affine); their PLACEMENT (before vs after
// the sub-block) is the `do_layer_norm_before` switch.
struct OPTLayerWeights {
  OwnedTensor self_attn_layer_norm;       // bf16 [H]
  OwnedTensor self_attn_layer_norm_bias;  // bf16 [H]
  OwnedTensor final_layer_norm;           // bf16 [H]
  OwnedTensor final_layer_norm_bias;      // bf16 [H]
  OPTAttnWeights attn;
  OPTMlpWeights mlp;
};

// Whole OPT text-model weights (opt.py::OPTDecoder, :198-293). `embed_positions`
// is the LEARNED table of [max_position_embeddings + 2, H] — the +2 is
// OPTLearnedPositionalEmbedding.offset (opt.py:64-65), and the forward adds 2 to
// every position before the lookup. When `tie_word_embeddings` (the default, and
// what facebook/opt-125m sets) `lm_head` is EMPTY and the output projection
// aliases `embed_tokens`, mirroring opt.py:352-353 + the loader's
// `skip_prefixes=["lm_head.weight"]` (opt.py:390-392).
struct OPTWeights {
  bool tie_word_embeddings = true;
  bool do_layer_norm_before = true;
  OwnedTensor embed_tokens;     // bf16 [vocab, word_embed_proj_dim]
  OwnedTensor embed_positions;  // bf16 [max_pos + 2, H]
  // The decoder-level final LayerNorm exists only when `do_layer_norm_before &&
  // !_remove_final_layer_norm` (opt.py:243-253); EMPTY otherwise.
  OwnedTensor final_layer_norm;
  OwnedTensor final_layer_norm_bias;
  OwnedTensor lm_head;  // bf16 [word_embed_proj_dim, vocab]; EMPTY when tied
  std::vector<OPTLayerWeights> layers;
};

// Load `OPTForCausalLM` safetensors into OPTWeights.
//
// On-disk names (verbatim from the HF checkpoint, which is what vLLM's
// WeightsMapper consumes after its `decoder.` -> `model.decoder.` prefix rename,
// opt.py:328-338): model.decoder.{embed_tokens,embed_positions}.weight,
// model.decoder.final_layer_norm.{weight,bias}, and per layer
// model.decoder.layers.N.{self_attn_layer_norm,final_layer_norm}.{weight,bias},
// .self_attn.{q,k,v,out}_proj.{weight,bias}, .{fc1,fc2}.{weight,bias}.
// q/k/v are merged into one qkv_proj (and their biases into one qkv_bias) per
// vLLM's packed_modules_mapping (opt.py:339-341). The checkpoint's redundant
// lm_head.weight is SKIPPED when tie_word_embeddings (opt.py:390-392).
// Reuses the shared dense_weight_loaders.h helpers.
OPTWeights LoadOPTForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config);

// The OPT forward (opt.py::OPTDecoder.forward :266-293 + OPTDecoderLayer.forward
// :168-195), composed from the public vt:: ops:
//
//   hidden = embed_tokens[ids] + embed_positions[positions + 2]
//   per layer (do_layer_norm_before == true, the 125m/1.7B/…/175B placement):
//     res = hidden; hidden = LayerNorm(hidden)
//     -> qkv MatmulBT + bias -> split q/k/v -> paged causal attention
//     -> out_proj + bias ; hidden = res + hidden
//     res = hidden; hidden = LayerNorm(hidden)
//     -> fc1 + bias -> ReLU -> fc2 + bias ; hidden = res + hidden
//   (do_layer_norm_before == false, the 350m placement: each LayerNorm moves to
//    AFTER its residual join — opt.py:175-194.)
//   final_layer_norm (when present) -> lm_head (TIED to embed_tokens by default)
//
// Numeric contract: the whole stream is bf16 (matching vLLM's per-op bf16 stores
// under `--dtype bfloat16`, the arm the SACRED gate runs on — see the spec's
// decision D1 on why bf16 rather than the checkpoint's fp16); LayerNorm
// accumulates its mean/variance in f32 and rounds once on store, exactly as
// torch's `acc_type<bfloat16> == float` LayerNorm does. The paged KV cache is
// written bf16. Returns [n_out, vocab] f32 logits.
class OPTModel {
 public:
  // Batched PAGED forward. Same contract as Qwen3DenseModel::Forward:
  // token_ids/positions are the flattened length-num_actual_tokens step inputs;
  // attn_kv is one PagedKvCache per layer (all layers are full attention);
  // `logits_indices`, when a proper subset of the T rows, gathers the final
  // hidden rows on-device BEFORE lm_head so the return is [num_reqs, vocab].
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const OPTWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // DEVICE-resident variant (sampler-on-device hot path): same contract as
  // Forward but returns the lm_head output as a device buffer with no
  // full-logits D2H (ForwardLogits::device_*).
  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const OPTWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// Per-family config hook (mirrors ParseQwen3ForCausalLMConfig). Unlike the Qwen
// hooks this one is NOT a no-op: it validates the OPT-only fields read out of
// `HfConfig::raw` and rejects the variants we have no checkpoint to gate
// (word_embed_proj_dim != hidden_size, non-relu activation).
void ParseOPTConfig(const HfConfig& config);

// KV-cache spec builder: OPT is pure full attention — exactly ONE full-attention
// KV group, NO MambaSpec/GDN group. Multi-head (num_key_value_heads ==
// num_attention_heads); OPT predates GQA.
v1::KVCacheConfig MakeOPTKVCache(const HfConfig& config, int block_size, int num_blocks);

}  // namespace vllm
