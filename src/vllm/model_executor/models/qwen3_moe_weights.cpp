// Weight loader for the full-attention MoE Qwen3-Coder text model
// (`Qwen3MoeForCausalLM`, Qwen3-Coder-30B-A3B-Instruct, BF16) — the first
// full-attention MoE bring-up W2. Loads the checkpoint safetensors into
// Qwen3MoeWeights (qwen3_moe.h) by COMBINING the dense-attention name map
// (qwen3_weights.cpp, via the SHARED dense_weight_loaders.h helpers) with a NEW
// bf16 per-expert loader (the bf16 analog of qwen3_5_weights.cpp
// LoadMoeExpertsInto, which loads NVFP4). See .agents/specs/
// sweep-qwen3-coder-30b.md §3b SEAM GAP #4, §7 W2.
//
// Grounding: vllm/model_executor/models/qwen3_moe.py @ e24d1b24 —
//   - Qwen3MoeAttention (:254-354): qkv_proj (QKVParallelLinear) split
//     [q_size,kv_size,kv_size]; per-head q_norm/k_norm = RMSNorm(head_dim);
//     o_proj (RowParallelLinear). bias = attention_bias (default false). Byte-for-
//     byte the same shape as qwen3.py::Qwen3Attention (the dense path).
//   - Qwen3MoeSparseMoeBlock (:130-251): router gate = ReplicatedLinear(hidden,
//     num_experts, bias=False); FusedMoE per-expert gate/up/down; shared expert
//     ONLY when shared_expert_intermediate_size>0 (Coder has none — the shared_*
//     fields stay EMPTY).
//   - packed_modules_mapping (:571): qkv_proj<-[q,k,v]_proj (the per-expert
//     gate/up/down are NOT packed — each expert owns separate gate/up/down files).
//   - lm_head = ParallelLMHead, tied only if tie_word_embeddings (:578-585) —
//     Qwen3-Coder is UNTIED (tie_word_embeddings=false) → the checkpoint's
//     separate lm_head.weight is loaded (NOT aliased to embed_tokens).
//
// Name map (Qwen3-Coder config.json, plain `model.layers.` prefix — NOT the 35B
// multimodal `model.language_model` wrapper):
//   model.embed_tokens.weight                          -> embed_tokens [V,H]
//   model.norm.weight                                  -> final_norm [H]
//   lm_head.weight                                     -> lm_head [H,V] (UNTIED)
//   model.layers.N.input_layernorm.weight              -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight     -> post_attention_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight       -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.o_proj.weight             -> o_proj (raw-NK)
//   model.layers.N.self_attn.{q,k}_norm.weight         -> q_norm/k_norm [Dh]
//   model.layers.N.mlp.gate.weight                     -> moe.router_gate [H,E]
//   model.layers.N.mlp.experts.E.gate_proj.weight      -> moe.expert_gate[E] [H,I]
//   model.layers.N.mlp.experts.E.up_proj.weight        -> moe.expert_up[E]   [H,I]
//   model.layers.N.mlp.experts.E.down_proj.weight      -> moe.expert_down[E] [I,H]
#include "vllm/model_executor/models/qwen3_moe.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;

// Read a top-level boolean from the raw config.json doc (Qwen3-Coder configs are
// flat — no text_config nesting), defaulting when absent/null/non-boolean.
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

// NEW bf16 per-expert loader (SEAM GAP #4 — the bf16 analog of the NVFP4
// LoadMoeExpertsInto). Each expert owns separate torch-Linear gate/up/down files;
// they are transposed to Matmul-B layout (gate/up -> [H,I]; down -> [I,H]) exactly
// like the GGUF bf16 expert loader (qwen3_5_gguf_weights.cpp LoadExpertsT), which
// the reference-path ExpertMlp (qwen3_5.cpp) consumes. The router gate is a plain
// ReplicatedLinear [E,H] transposed to [H,E]. Qwen3-Coder has NO shared expert
// (shared_expert_intermediate_size==0), so the shared_* fields stay EMPTY.
MoeBlockWeights LoadQwen3MoeBlock(const TensorResolver& get, const std::string& base,
                                  int64_t num_experts) {
  const std::string mlp = base + "mlp.";
  MoeBlockWeights m;
  m.router_gate = LoadBf16Transposed(get, mlp + "gate.weight");  // [E,H] -> [H,E]
  m.expert_gate.reserve(static_cast<size_t>(num_experts));
  m.expert_up.reserve(static_cast<size_t>(num_experts));
  m.expert_down.reserve(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    const std::string ex = mlp + "experts." + std::to_string(e) + ".";
    m.expert_gate.push_back(LoadBf16Transposed(get, ex + "gate_proj.weight"));  // [I,H]->[H,I]
    m.expert_up.push_back(LoadBf16Transposed(get, ex + "up_proj.weight"));      // [I,H]->[H,I]
    m.expert_down.push_back(LoadBf16Transposed(get, ex + "down_proj.weight"));  // [H,I]->[I,H]
  }
  // shared_gate / shared_*_proj and all fp4 fields intentionally left EMPTY.
  return m;
}

Qwen3MoeLayerWeights LoadQwen3MoeLayer(const TensorResolver& get, int64_t layer,
                                       int64_t num_experts, bool attention_bias) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";

  Qwen3MoeLayerWeights w;
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");

  // QKVParallelLinear: one merged owner in exact [q,k,v] output-row order
  // (packed_modules_mapping qkv_proj<-[q,k,v]_proj), kept raw-NK for MatmulBT.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  // RowParallelLinear o_proj — single raw-NK owner.
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // Per-head q/k RMSNorm (RMSNorm(head_dim)), applied before RoPE.
  w.attn.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  w.attn.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");
  if (attention_bias) {
    w.attn.qkv_bias = LoadMergedBf16RawNK(
        get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"});
  }

  w.moe = LoadQwen3MoeBlock(get, base, num_experts);
  return w;
}

}  // namespace

Qwen3MoeWeights LoadQwen3MoeForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3 moe: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "qwen3 moe: num_hidden_layers must be positive");
  VT_CHECK(config.num_experts > 0, "qwen3 moe: num_experts must be positive");

  Qwen3MoeWeights w;
  // Qwen3-Coder is UNTIED (tie_word_embeddings=false); attention_bias=false.
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);
  w.attention_bias = RawBool(config.raw, "attention_bias", false);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  // UNTIED lm_head: load the SEPARATE checkpoint lm_head.weight [V,H] transposed
  // to the Matmul-B [H,V] layout (do NOT alias embed_tokens). Mirrors vLLM's
  // ParallelLMHead when tie_word_embeddings is false (qwen3_moe.py:578-585). The
  // tied branch is kept for completeness but Qwen3-Coder never takes it.
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(
        LoadQwen3MoeLayer(get, l, config.num_experts, w.attention_bias));
  return w;
}

}  // namespace vllm
