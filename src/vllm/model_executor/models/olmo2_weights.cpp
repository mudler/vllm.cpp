// Weight loader for OLMo-2 dense (`Olmo2ForCausalLM`, OLMo-2-0425-1B) — task W2.
// Loads the checkpoint safetensors into Olmo2Weights (olmo2.h).
//
// NOTE — the OLMo-2 checkpoint is stored FLOAT32 on disk (config torch_dtype
// "float32"; all 179 tensors F32), but vLLM serves it in bfloat16 (the oracle
// capture forces dtype="bfloat16"), downcasting f32->bf16 with round-to-nearest-
// even on load. The shared dense_weight_loaders.h helpers require BF16 on-disk, so
// this loader carries its OWN f32->bf16 downcast helpers (round-to-nearest-even via
// vt::F32ToBF16, bit-identical to torch's .to(bfloat16) and to every on-device
// F32ToBF16), producing exactly the bf16 weights vLLM computes with. Same [N=out,
// K=in] raw orientation + merged-shard ownership rule as the BF16 helpers.
//
// Grounding: vllm/model_executor/models/olmo2.py @ e24d1b24 —
//   - Olmo2Attention (:67-185): qkv_proj (QKVParallelLinear, bias=False) split
//     [q_size,kv_size,kv_size]; FULL-WIDTH q_norm[hidden]/k_norm[kv_size]; o_proj
//     (RowParallelLinear, no bias).
//   - Olmo2MLP (:188-230): gate_proj/up_proj (merged gate_up) -> SiluAndMul -> down_proj.
//   - Olmo2DecoderLayer (:233-259): TWO RMSNorms — post_attention_layernorm,
//     post_feedforward_layernorm (NO input/pre norm).
//   - packed_modules_mapping (:361-364): qkv_proj<-[q,k,v]_proj, gate_up_proj<-[gate,up]_proj.
//   - load_weights (:413-420): skip lm_head.weight only when tied (OLMo-2-1B is
//     UNTIED -> lm_head loaded).
//
// Name map (OLMo-2-0425-1B, flat — NO multimodal prefix):
//   model.embed_tokens.weight                          -> embed_tokens [V,H]
//   model.norm.weight                                  -> final_norm [H]
//   lm_head.weight                                     -> lm_head (untied, Matmul-B)
//   model.layers.N.post_attention_layernorm.weight     -> post_attention_ln [H]
//   model.layers.N.post_feedforward_layernorm.weight   -> post_feedforward_ln [H]
//   model.layers.N.self_attn.{q,k,v}_proj.weight       -> merged qkv_proj (raw-NK)
//   model.layers.N.self_attn.q_norm.weight             -> q_norm [q_size]
//   model.layers.N.self_attn.k_norm.weight             -> k_norm [kv_size]
//   model.layers.N.self_attn.o_proj.weight             -> o_proj (raw-NK, no bias)
//   model.layers.N.mlp.{gate,up}_proj.weight           -> merged gate_up_proj (raw-NK)
//   model.layers.N.mlp.down_proj.weight                -> down_proj (raw-NK)
#include "vllm/model_executor/models/olmo2.h"

#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"  // MakeOwned
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::MakeOwned;

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

// Downcast a contiguous F32 buffer to bf16 (round-to-nearest-even), the same
// rounding torch's .to(bfloat16) and vt::F32ToBF16 use.
void F32ToBf16Into(const float* src, int64_t n, uint16_t* dst) {
  for (int64_t i = 0; i < n; ++i) dst[i] = vt::F32ToBF16(src[i]);
}

// F32 tensor -> owned bf16, copied verbatim (optionally reshaped). Mirrors
// LoadBf16Direct but downcasts f32->bf16.
OwnedTensor LoadF32ToBf16Direct(const TensorResolver& get, const std::string& name,
                                const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "F32", "olmo2: expected F32 for " + name);
  std::vector<int64_t> shape = shape_override.empty() ? t.shape : shape_override;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  const int64_t n = o.Numel();
  VT_CHECK(t.nbytes == static_cast<size_t>(n) * sizeof(float),
           "olmo2: byte-size mismatch for " + name);
  F32ToBf16Into(reinterpret_cast<const float*>(t.data), n,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// F32 [out, in] -> owned bf16 [in, out] (Matmul-B layout for the untied lm_head).
// Mirrors LoadBf16Transposed but downcasts f32->bf16.
OwnedTensor LoadF32ToBf16Transposed(const TensorResolver& get,
                                    const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "F32", "olmo2: expected F32 for " + name);
  VT_CHECK(t.shape.size() == 2, "olmo2: expected 2-D weight for " + name);
  const int64_t out_dim = t.shape[0];
  const int64_t in_dim = t.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  const auto* src = reinterpret_cast<const float*>(t.data);
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t r = 0; r < out_dim; ++r)
    for (int64_t c = 0; c < in_dim; ++c)
      dst[c * out_dim + r] = vt::F32ToBF16(src[r * in_dim + c]);
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// Concatenate F32 torch-Linear shards [N_i,K] along output rows, downcast to bf16,
// kept RAW [N,K] with nk=true for vt::MatmulBT. Mirrors LoadMergedBf16RawNK.
OwnedTensor LoadMergedF32ToBf16RawNK(const TensorResolver& get,
                                     const std::vector<std::string>& names) {
  VT_CHECK(!names.empty(), "olmo2: merged projection requires at least one shard");
  int64_t in_dim = -1;
  int64_t out_dim = 0;
  std::vector<const StTensor*> shards;
  shards.reserve(names.size());
  for (const std::string& name : names) {
    const StTensor& tensor = get(name);
    VT_CHECK(tensor.dtype == "F32", "olmo2: expected F32 for " + name);
    VT_CHECK(tensor.shape.size() == 2, "olmo2: expected 2-D weight for " + name);
    VT_CHECK(tensor.shape[0] > 0 && tensor.shape[1] > 0,
             "olmo2: merged shard has an empty dimension: " + name);
    if (in_dim < 0) in_dim = tensor.shape[1];
    VT_CHECK(tensor.shape[1] == in_dim,
             "olmo2: merged shards must share input width");
    VT_CHECK(out_dim <= std::numeric_limits<int64_t>::max() - tensor.shape[0],
             "olmo2: merged output width overflow");
    out_dim += tensor.shape[0];
    shards.push_back(&tensor);
  }
  OwnedTensor merged = MakeOwned(vt::DType::kBF16, {out_dim, in_dim});
  auto* dst = reinterpret_cast<uint16_t*>(merged.bytes.data());
  size_t off = 0;  // element offset into dst
  for (size_t i = 0; i < shards.size(); ++i) {
    const StTensor& shard = *shards[i];
    const int64_t n = shard.shape[0] * in_dim;
    VT_CHECK(shard.nbytes == static_cast<size_t>(n) * sizeof(float),
             "olmo2: byte-size mismatch for " + names[i]);
    F32ToBf16Into(reinterpret_cast<const float*>(shard.data), n, dst + off);
    off += static_cast<size_t>(n);
    MaybeReleaseSourcePages(shard.data, shard.nbytes);
  }
  VT_CHECK(off == static_cast<size_t>(out_dim * in_dim),
           "olmo2: merged byte accounting mismatch");
  merged.nk = true;
  return merged;
}

Olmo2LayerWeights LoadOlmo2Layer(const TensorResolver& get, int64_t layer) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Olmo2LayerWeights w;
  // TWO standalone post-norms (olmo2.py:253-259). NO input/pre norm.
  w.post_attention_layernorm =
      LoadF32ToBf16Direct(get, base + "post_attention_layernorm.weight");
  w.post_feedforward_layernorm =
      LoadF32ToBf16Direct(get, base + "post_feedforward_layernorm.weight");

  // QKVParallelLinear (bias=False): one merged owner in exact [q,k,v] output-row
  // order (packed_modules_mapping), kept raw-NK for MatmulBT.
  w.attn.qkv_proj = LoadMergedF32ToBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  // FULL-WIDTH q/k RMSNorm weights (olmo2.py:113-117): q_norm[hidden_size],
  // k_norm[kv_size] — over the WHOLE projection, NOT per-head head_dim.
  w.attn.q_norm = LoadF32ToBf16Direct(get, sa + "q_norm.weight");
  w.attn.k_norm = LoadF32ToBf16Direct(get, sa + "k_norm.weight");
  // RowParallelLinear o_proj — single raw-NK owner, NO bias.
  w.attn.o_proj = LoadMergedF32ToBf16RawNK(get, {sa + "o_proj.weight"});

  // OLMo-2 ships gate_proj/up_proj as SEPARATE tensors -> merge to gate_up (raw-NK).
  w.mlp.gate_up_proj = LoadMergedF32ToBf16RawNK(
      get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedF32ToBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

Olmo2Weights LoadOlmo2ForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                         const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "olmo2: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "olmo2: num_hidden_layers must be positive");

  Olmo2Weights w;
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);

  w.embed_tokens = LoadF32ToBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadF32ToBf16Direct(get, "model.norm.weight");
  // OLMo-2-1B is UNTIED: load the standalone lm_head (Matmul-B [H, vocab]). The
  // tied path (skip lm_head, alias embed_tokens via MatmulBT) is retained for a
  // tied OLMo-2 checkpoint but is not exercised by OLMo-2-0425-1B.
  if (!w.tie_word_embeddings)
    w.lm_head = LoadF32ToBf16Transposed(get, "lm_head.weight");

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadOlmo2Layer(get, l));
  return w;
}

}  // namespace vllm
