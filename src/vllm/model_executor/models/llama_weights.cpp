// Weight loader for the DENSE Llama text arch (`LlamaForCausalLM`, Llama-3.2-1B,
// BF16) — the cross-family additive bring-up W2. Loads the checkpoint safetensors
// into the SHARED dense container (Qwen3DenseWeights, qwen3.h) via the SHARED
// dense_weight_loaders.h helpers. It is the Qwen3-dense loader MINUS the per-head
// q/k norms (Llama has no qk-norm), which is the ONLY structural weight-map
// difference between the two dense arches.
//
// Grounding: vllm/model_executor/models/llama.py @ e24d1b24 —
//   - LlamaAttention: qkv_proj (QKVParallelLinear, bias=attention_bias, default
//     false) split [q,k,v]; NO q_norm/k_norm; o_proj (RowParallelLinear).
//   - LlamaMLP: merged gate_up_proj -> SiluAndMul -> down_proj (mlp_bias false).
//   - packed_modules_mapping: qkv_proj<-[q,k,v]_proj, gate_up_proj<-[gate,up]_proj.
//   - tie_word_embeddings: lm_head aliases embed_tokens; the loader skips the
//     checkpoint lm_head.weight via skip_prefixes=(["lm_head."]) (llama.py:538).
//
// Name map (Llama-3.2-1B/config.json, flat — no multimodal prefix):
//   model.embed_tokens.weight                         -> embed_tokens [V,H]
//   model.norm.weight                                 -> final_norm [H]
//   lm_head.weight                                    -> SKIPPED when tied
//   model.layers.N.input_layernorm.weight             -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight    -> post_attention_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight      -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.o_proj.weight            -> o_proj (raw-NK)
//   model.layers.N.mlp.{gate,up}_proj.weight          -> merged gate_up_proj (raw)
//   model.layers.N.mlp.down_proj.weight               -> down_proj (raw-NK)
#include "vllm/model_executor/models/llama.h"

#include <functional>
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
using dense_loaders::LoadExl3;
using dense_loaders::LoadF16AsBf16Direct;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::IsExl3Projection;

// Read a top-level boolean from the raw config.json doc (Llama configs are flat),
// defaulting when absent/null/non-boolean.
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

Qwen3DenseLayerWeights LoadLlamaLayer(const TensorResolver& get,
                                      const std::function<bool(const std::string&)>& has,
                                      int64_t layer, bool attention_bias) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Qwen3DenseLayerWeights w;

  // QUANT-EXL3 W1b (#2181). The scheme is decided per projection by upstream's
  // own storage predicate, not by a config flag, so a checkpoint that quantizes
  // only some layers loads each one the way it is actually stored.
  const bool exl3 = IsExl3Projection(has, sa + "q_proj");
  if (exl3) {
    // The norms and the layernorms are NOT quantized in an EXL3 checkpoint;
    // they ship F16 beside the trellis.
    w.input_layernorm = LoadF16AsBf16Direct(get, base + "input_layernorm.weight");
    w.post_attention_layernorm =
        LoadF16AsBf16Direct(get, base + "post_attention_layernorm.weight");
    // q/k/v stay SEPARATE where the bf16 and NVFP4 arms hold one merged owner.
    // Merging trellis operands is a real transform (joining on the output dim
    // interleaves per input tile), valid for this family but owed its own gate.
    w.attn.q_proj_exl3 = LoadExl3(get, has, sa + "q_proj");
    w.attn.k_proj_exl3 = LoadExl3(get, has, sa + "k_proj");
    w.attn.v_proj_exl3 = LoadExl3(get, has, sa + "v_proj");
    w.attn.o_proj_exl3 = LoadExl3(get, has, sa + "o_proj");
    w.mlp.gate_proj_exl3 = LoadExl3(get, has, mlp + "gate_proj");
    w.mlp.up_proj_exl3 = LoadExl3(get, has, mlp + "up_proj");
    w.mlp.down_proj_exl3 = LoadExl3(get, has, mlp + "down_proj");
    VT_CHECK(!attention_bias,
             "llama exl3: attention_bias is not implemented on the EXL3 arm "
             "(QUANT-EXL3, #2181); the bf16 arm's bias path does not apply to a "
             "trellis projection");
    return w;
  }

  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");

  // QKVParallelLinear: one merged owner in exact [q,k,v] output-row order
  // (packed_modules_mapping qkv_proj<-[q,k,v]_proj), kept raw-NK for MatmulBT.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  // RowParallelLinear o_proj — single raw-NK owner.
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // NO per-head q/k RMSNorm: Llama has none, so q_norm/k_norm stay EMPTY and the
  // shared AttnBlock skips the norm step (has_qk_norm == false).
  if (attention_bias) {
    w.attn.qkv_bias = LoadMergedBf16RawNK(
        get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"});
  }

  // MergedColumnParallelLinear gate_up in exact [gate,up] order, then down_proj.
  w.mlp.gate_up_proj = LoadMergedBf16RawNK(
      get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

LlamaWeights LoadLlamaForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const std::function<bool(const std::string&)> has =
      [&where](const std::string& name) { return where.find(name) != where.end(); };
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "llama dense: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "llama dense: num_hidden_layers must be positive");

  LlamaWeights w;
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);
  w.attention_bias = RawBool(config.raw, "attention_bias", false);

  // QUANT-EXL3 (#2181): an EXL3 checkpoint's unquantized remainder is F16.
  const bool exl3 = IsExl3Projection(has, "model.layers.0.self_attn.q_proj");
  if (exl3) {
    w.embed_tokens = LoadF16AsBf16Direct(get, "model.embed_tokens.weight");
    w.final_norm = LoadF16AsBf16Direct(get, "model.norm.weight");
  } else {
    w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
    w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  }
  // tie_word_embeddings: lm_head aliases embed_tokens; the checkpoint's redundant
  // lm_head.weight is SKIPPED (mirrors vLLM skip_prefixes=["lm_head."]). Only the
  // untied case loads a standalone lm_head (Matmul-B [H, vocab]).
  if (exl3 && IsExl3Projection(has, "lm_head")) {
    // The EXL3 head is a REAL quantized tensor, and it is preferred over the
    // tied embedding table EVEN THOUGH the artifact declares
    // `tie_word_embeddings: true`. That is a deliberate divergence from the
    // bf16 arm's reading of the same flag, argued at the field's declaration in
    // `qwen3.h`: the publisher quantized a separate head at its own width, so
    // the head is what it intends to be used. Its width is its own -- 6-bit
    // against the body's 3 in the published 3.0bpw quant -- which the reader
    // takes from the tensor and never from a config scalar.
    w.lm_head_exl3 = LoadExl3(get, has, "lm_head");
  } else if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadLlamaLayer(get, has, l, w.attention_bias));
  return w;
}

LlamaWeights LoadLlamaModelEmbeddingWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  // ARCH-ONE-SURFACE ROW 6: the `LlamaModel` EMBEDDING checkpoint loader —
  // the SAME name map as LoadLlamaForCausalLMWeights (LoadLlamaLayer above is
  // the single source of it), with the two as_embedding_model deltas:
  //   1. BOTH name layouts load (adapters.py:178-181 candidate_prefixes
  //      ["", "model."]): a bare `LlamaModel` checkpoint names its tensors
  //      `embed_tokens.weight` / `layers.N...` / `norm.weight` (no "model."
  //      prefix); a `*ForCausalLM`-layout export keeps the prefix. The
  //      resolver maps the canonical "model."-prefixed ask onto whichever
  //      layout the shards actually carry.
  //   2. NO lm_head, ever (adapters.py:135-151 replaces the output layer with
  //      a missing-layer stage): tie_word_embeddings is forced true so the
  //      pooling forward — which never multiplies by an output layer — has a
  //      well-formed container, and a checkpoint lm_head.weight is ignored.
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  // The same two-layout probe the resolver below performs, so an EXL3 embedding
  // checkpoint is DETECTED the same way it would be RESOLVED. A probe that
  // checked only the prefixed name would answer "not EXL3" for a bare `*Model`
  // layout and silently take the bf16 arm.
  const std::function<bool(const std::string&)> has =
      [&where](const std::string& name) {
        if (where.find(name) != where.end()) return true;
        return name.rfind("model.", 0) == 0 && where.find(name.substr(6)) != where.end();
      };
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    std::string key = name;
    auto it = where.find(key);
    if (it == where.end() && key.rfind("model.", 0) == 0) {
      key = key.substr(6);  // the bare `*Model` layout
      it = where.find(key);
    }
    VT_CHECK(it != where.end(), "llama embedding: tensor not found: " + name);
    return it->second->Get(key);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "llama embedding: num_hidden_layers must be positive");

  LlamaWeights w;
  w.tie_word_embeddings = true;  // no output layer on the pooling forward
  w.attention_bias = RawBool(config.raw, "attention_bias", false);

  // QUANT-EXL3 (#2181): this arm has NOT been given the EXL3 reader that
  // `LoadLlamaForCausalLMWeights` has, so an EXL3 embedding checkpoint would
  // otherwise refuse further down on an F16 dtype and name a tensor rather than
  // the cause. Refuse it here, where the cause is still in hand. Wiring the arm
  // is a wave of its own: an embedding model has no lm_head and no forward to
  // gate against, so it needs a different equivalence than the causal-LM arm.
  VT_CHECK(!IsExl3Projection(has, "model.layers.0.self_attn.q_proj"),
           "llama embedding: this checkpoint is EXL3-quantized, and the embedding "
           "loader implements the bf16 arm only (QUANT-EXL3, #2181). The causal-LM "
           "loader reads EXL3; this one does not.");

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadLlamaLayer(get, has, l, w.attention_bias));
  return w;
}

}  // namespace vllm
