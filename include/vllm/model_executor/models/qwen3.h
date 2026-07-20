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

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

// Placeholder owned-weights carrier for the Qwen3 dense LoadedModel. Fleshed out
// in W2 (safetensors name map + tied lm_head). W0/W1 only need a type for the
// stub LoadedModel subclass; the forward is not yet implemented.
struct Qwen3DenseWeights {};

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
