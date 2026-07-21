// Weight loader for OPT (`OPTForCausalLM`, facebook/opt-125m, BF16) — the
// CROSS-FAMILY additivity canary, W2. Loads the checkpoint safetensors into
// OPTWeights (opt.h) via the SHARED dense_weight_loaders.h helpers.
//
// Grounding: vllm/model_executor/models/opt.py @ e24d1b24 —
//   - OPTAttention (:71-122): qkv_proj (QKVParallelLinear, bias=config.
//     enable_bias) chunked into q,k,v; out_proj (RowParallelLinear, same bias).
//   - OPTDecoderLayer (:125-195): self_attn_layer_norm / final_layer_norm are
//     `nn.LayerNorm` (weight AND bias); fc1/fc2 are Column/RowParallelLinear
//     with `config.enable_bias`.
//   - OPTDecoder (:198-261): embed_tokens [vocab, word_embed_proj_dim];
//     embed_positions = OPTLearnedPositionalEmbedding(max_position_embeddings,
//     hidden_size) whose nn.Embedding is allocated with `num_embeddings +
//     offset` where offset == 2 (:59-68) — so the on-disk table is [max_pos+2,H];
//     project_in/project_out only when word_embed_proj_dim != hidden_size
//     (:220-241); decoder final_layer_norm only when `do_layer_norm_before and
//     not _remove_final_layer_norm` (:243-253).
//   - OPTForCausalLM (:327-394): hf_to_vllm_mapper renames the checkpoint's
//     `decoder.` -> `model.decoder.` and stacks q/k/v into qkv_proj (:328-338);
//     packed_modules_mapping (:339-341); lm_head aliases embed_tokens when
//     tie_word_embeddings (:352-353) and the loader then skips the checkpoint's
//     `lm_head.weight` (:390-392).
//
// Name map (facebook/opt-125m, verified on the real checkpoint — 197 tensors =
// 5 top-level + 12 layers x 16):
//   model.decoder.embed_tokens.weight                   -> embed_tokens [V,H]
//   model.decoder.embed_positions.weight                -> embed_positions [P+2,H]
//   model.decoder.final_layer_norm.{weight,bias}        -> final LN (+bias)
//   lm_head.weight                                      -> SKIPPED when tied
//   ...layers.N.self_attn_layer_norm.{weight,bias}      -> attn LN (+bias)
//   ...layers.N.final_layer_norm.{weight,bias}          -> mlp LN (+bias)
//   ...layers.N.self_attn.{q,k,v}_proj.weight           -> merged qkv_proj (raw-NK)
//   ...layers.N.self_attn.{q,k,v}_proj.bias             -> merged qkv_bias [3H]
//   ...layers.N.self_attn.out_proj.{weight,bias}        -> out_proj (raw-NK) + bias
//   ...layers.N.{fc1,fc2}.{weight,bias}                 -> fc1/fc2 (raw-NK) + bias
#include "vllm/model_executor/models/opt.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::LoadMergedBf16Vector;

OPTLayerWeights LoadOPTLayer(const TensorResolver& get, int64_t layer,
                             const OPTConfigExtras& extras) {
  const std::string base = "model.decoder.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";

  OPTLayerWeights w;
  // Both LayerNorms carry weight AND bias when elementwise_affine (the default,
  // and what every released OPT checkpoint sets).
  if (extras.layer_norm_elementwise_affine) {
    w.self_attn_layer_norm = LoadBf16Direct(get, base + "self_attn_layer_norm.weight");
    w.self_attn_layer_norm_bias = LoadBf16Direct(get, base + "self_attn_layer_norm.bias");
    w.final_layer_norm = LoadBf16Direct(get, base + "final_layer_norm.weight");
    w.final_layer_norm_bias = LoadBf16Direct(get, base + "final_layer_norm.bias");
  }

  // QKVParallelLinear: one merged owner in exact [q,k,v] output-row order
  // (packed_modules_mapping qkv_proj<-[q,k,v]_proj), kept raw-NK for MatmulBT.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  w.attn.out_proj = LoadMergedBf16RawNK(get, {sa + "out_proj.weight"});
  // OPT's first-in-tree feature: BIASED projections. The merged qkv bias must be
  // concatenated in the SAME [q,k,v] order as the merged weight.
  if (extras.enable_bias) {
    w.attn.qkv_bias = LoadMergedBf16Vector(
        get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"});
    w.attn.out_bias = LoadBf16Direct(get, sa + "out_proj.bias");
  }

  // Plain fc1 -> ReLU -> fc2 MLP (no gate/up merge exists; there is no gating).
  w.mlp.fc1 = LoadMergedBf16RawNK(get, {base + "fc1.weight"});
  w.mlp.fc2 = LoadMergedBf16RawNK(get, {base + "fc2.weight"});
  if (extras.enable_bias) {
    w.mlp.fc1_bias = LoadBf16Direct(get, base + "fc1.bias");
    w.mlp.fc2_bias = LoadBf16Direct(get, base + "fc2.bias");
  }
  return w;
}

}  // namespace

OPTWeights LoadOPTForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get = [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "opt: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0, "opt: num_hidden_layers must be positive");
  const OPTConfigExtras extras = GetOPTConfigExtras(config);

  OPTWeights w;
  w.tie_word_embeddings = extras.tie_word_embeddings;
  w.do_layer_norm_before = extras.do_layer_norm_before;

  w.embed_tokens = LoadBf16Direct(get, "model.decoder.embed_tokens.weight");
  // LEARNED absolute positions. The table is [max_position_embeddings + 2, H] —
  // OPTLearnedPositionalEmbedding allocates `num_embeddings + self.offset` with
  // offset == 2 (opt.py:59-68). Assert that, because silently loading a table
  // sized max_pos would make every position lookup off by two rows.
  w.embed_positions = LoadBf16Direct(get, "model.decoder.embed_positions.weight");
  VT_CHECK(w.embed_positions.rank == 2 &&
               w.embed_positions.shape[0] == config.max_position_embeddings + 2,
           "opt: embed_positions must be [max_position_embeddings + 2, H] "
           "(OPTLearnedPositionalEmbedding.offset == 2)");

  // Decoder-level final LayerNorm exists only for the pre-LN placement without
  // the legacy `_remove_final_layer_norm` flag (opt.py:243-253).
  if (extras.do_layer_norm_before && !extras.remove_final_layer_norm) {
    w.final_layer_norm = LoadBf16Direct(get, "model.decoder.final_layer_norm.weight");
    w.final_layer_norm_bias = LoadBf16Direct(get, "model.decoder.final_layer_norm.bias");
  }

  // tie_word_embeddings (the OPT default): lm_head aliases embed_tokens and the
  // checkpoint's redundant lm_head.weight is SKIPPED, mirroring vLLM's
  // skip_prefixes=["lm_head.weight"] (opt.py:390-392).
  if (!w.tie_word_embeddings) w.lm_head = LoadBf16Transposed(get, "lm_head.weight");

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadOPTLayer(get, l, extras));
  return w;
}

}  // namespace vllm
