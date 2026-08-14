// Weight loader for Phi-1/Phi-2 (`PhiForCausalLM`, microsoft/phi-2). Merges the
// checkpoint's separate q/k/v -> qkv_proj (weight AND bias — WeightsMapper
// orig_to_new_stacked, phi.py:262-267), loads the biased `dense` (o_proj), the
// non-gated fc1/fc2 (both biased), the per-layer + final nn.LayerNorm weight+bias
// pairs, the UNTIED lm_head weight+bias, and precomputes the plain partial-rope
// cos/sin cache.
//
// DTYPE: microsoft/phi-2 ships FLOAT16 on disk (all 453 tensors F16), but vLLM
// serves it in bfloat16 (the oracle forces dtype="bfloat16"; the load log:
// "Casting torch.float16 to torch.bfloat16"). So the loader is DTYPE-AWARE: a BF16
// checkpoint reuses the shared dense_weight_loaders.h helpers verbatim; an F16
// checkpoint is downcast f16->bf16 via vt::F16ToF32 then vt::F32ToBF16 (round-to-
// nearest-even) — bit-identical to torch's `.to(bfloat16)` and to the on-device
// F32ToBF16 path, producing exactly the bf16 weights vLLM computes with. Same
// [N=out, K=in] raw orientation + merged-shard ownership rule as the BF16 helpers.
// This mirrors the OLMo-2 dtype-aware loader (F32->BF16) and keeps the shared
// header untouched (new-files-only ⇒ every other model byte-identical).
//
// On-disk names (verified microsoft/phi-2, safetensors, 453 tensors): SEPARATE
// q_proj/k_proj/v_proj (NOT a fused Wqkv); self_attn.dense; mlp.fc1/fc2;
// input_layernorm; model.final_layernorm; lm_head — all carrying .bias.
#include "vllm/model_executor/models/phi.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"
#include "vt/unaligned.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::LoadMergedBf16Vector;
using dense_loaders::MakeOwned;

// Downcast one F16 element to bf16 via the exact f16->f32->bf16(RNE) path torch's
// `.to(bfloat16)` uses (F16 is a subset of F32, so F16ToF32 is exact).
inline uint16_t F16ToBf16(uint16_t h) { return vt::F32ToBF16(vt::F16ToF32(h)); }

// `src` is `const void*`: every caller below hands it a pointer INTO the mmap'd
// safetensors payload, whose per-tensor byte offset is the running total of
// everything ahead of it and so need not be even. Forming or loading through a
// `const uint16_t*` there is undefined (issue #627); vt::LoadUnaligned is the
// project's seam for it.
void F16ToBf16Into(const void* src, int64_t n, uint16_t* dst) {
  const auto* bytes = static_cast<const uint8_t*>(src);
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = F16ToBf16(vt::LoadUnaligned<uint16_t>(bytes + i * 2));
  }
}

// F16 tensor -> owned bf16, copied verbatim (optionally reshaped). Mirrors
// LoadBf16Direct but downcasts f16->bf16.
OwnedTensor LoadF16ToBf16Direct(const TensorResolver& get, const std::string& name,
                                const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "F16", "phi: expected F16 for " + name);
  std::vector<int64_t> shape = shape_override.empty() ? t.shape : shape_override;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  const int64_t n = o.Numel();
  VT_CHECK(t.nbytes == static_cast<size_t>(n) * sizeof(uint16_t),
           "phi: byte-size mismatch for " + name);
  F16ToBf16Into(t.data, n, reinterpret_cast<uint16_t*>(o.bytes.data()));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// F16 [out, in] -> owned bf16 [in, out] (Matmul-B layout for the untied lm_head).
OwnedTensor LoadF16ToBf16Transposed(const TensorResolver& get,
                                    const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "F16", "phi: expected F16 for " + name);
  VT_CHECK(t.shape.size() == 2, "phi: expected 2-D weight for " + name);
  const int64_t out_dim = t.shape[0];
  const int64_t in_dim = t.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  // Unaligned: `t.data` is an arbitrary byte offset into the mmap (#627).
  const uint8_t* src = t.data;
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t r = 0; r < out_dim; ++r)
    for (int64_t c = 0; c < in_dim; ++c)
      dst[c * out_dim + r] =
          F16ToBf16(vt::LoadUnaligned<uint16_t>(src + (r * in_dim + c) * 2));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// Concatenate F16 torch-Linear shards [N_i,K] along output rows, downcast to bf16,
// kept RAW [N,K] with nk=true for vt::MatmulBT. Mirrors LoadMergedBf16RawNK.
OwnedTensor LoadMergedF16ToBf16RawNK(const TensorResolver& get,
                                     const std::vector<std::string>& names) {
  VT_CHECK(!names.empty(), "phi: merged projection requires at least one shard");
  int64_t in_dim = -1;
  int64_t out_dim = 0;
  std::vector<const StTensor*> shards;
  shards.reserve(names.size());
  for (const std::string& name : names) {
    const StTensor& tensor = get(name);
    VT_CHECK(tensor.dtype == "F16", "phi: expected F16 for " + name);
    VT_CHECK(tensor.shape.size() == 2, "phi: expected 2-D weight for " + name);
    VT_CHECK(tensor.shape[0] > 0 && tensor.shape[1] > 0,
             "phi: merged shard has an empty dimension: " + name);
    if (in_dim < 0) in_dim = tensor.shape[1];
    VT_CHECK(tensor.shape[1] == in_dim, "phi: merged shards must share input width");
    VT_CHECK(out_dim <= std::numeric_limits<int64_t>::max() - tensor.shape[0],
             "phi: merged output width overflow");
    out_dim += tensor.shape[0];
    shards.push_back(&tensor);
  }
  OwnedTensor merged = MakeOwned(vt::DType::kBF16, {out_dim, in_dim});
  auto* dst = reinterpret_cast<uint16_t*>(merged.bytes.data());
  size_t off = 0;  // element offset into dst
  for (size_t i = 0; i < shards.size(); ++i) {
    const StTensor& shard = *shards[i];
    const int64_t n = shard.shape[0] * in_dim;
    VT_CHECK(shard.nbytes == static_cast<size_t>(n) * sizeof(uint16_t),
             "phi: byte-size mismatch for " + names[i]);
    F16ToBf16Into(shard.data, n, dst + off);
    off += static_cast<size_t>(n);
    MaybeReleaseSourcePages(shard.data, shard.nbytes);
  }
  VT_CHECK(off == static_cast<size_t>(out_dim * in_dim),
           "phi: merged byte accounting mismatch");
  merged.nk = true;
  return merged;
}

// Concatenate 1-D F16 shards (the per-shard qkv biases) -> owned bf16 vector.
OwnedTensor LoadMergedF16ToBf16Vector(const TensorResolver& get,
                                      const std::vector<std::string>& names) {
  VT_CHECK(!names.empty(), "phi: merged vector requires at least one shard");
  int64_t total = 0;
  std::vector<const StTensor*> shards;
  shards.reserve(names.size());
  for (const std::string& name : names) {
    const StTensor& tensor = get(name);
    VT_CHECK(tensor.dtype == "F16", "phi: expected F16 for " + name);
    VT_CHECK(tensor.shape.size() == 1, "phi: expected 1-D vector for " + name);
    VT_CHECK(tensor.shape[0] > 0, "phi: merged vector shard is empty: " + name);
    total += tensor.shape[0];
    shards.push_back(&tensor);
  }
  OwnedTensor merged = MakeOwned(vt::DType::kBF16, {total});
  auto* dst = reinterpret_cast<uint16_t*>(merged.bytes.data());
  size_t off = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    const StTensor& shard = *shards[i];
    const int64_t n = shard.shape[0];
    VT_CHECK(shard.nbytes == static_cast<size_t>(n) * sizeof(uint16_t),
             "phi: byte-size mismatch for " + names[i]);
    F16ToBf16Into(shard.data, n, dst + off);
    off += static_cast<size_t>(n);
    MaybeReleaseSourcePages(shard.data, shard.nbytes);
  }
  return merged;
}

// ---- dtype-aware dispatch (BF16 verbatim reuse / F16 downcast) ---------------
OwnedTensor LoadDirect(const TensorResolver& get, const std::string& name,
                       bool is_bf16, const std::vector<int64_t>& shape = {}) {
  return is_bf16 ? LoadBf16Direct(get, name, shape)
                 : LoadF16ToBf16Direct(get, name, shape);
}
OwnedTensor LoadTransposed(const TensorResolver& get, const std::string& name,
                           bool is_bf16) {
  return is_bf16 ? LoadBf16Transposed(get, name)
                 : LoadF16ToBf16Transposed(get, name);
}
OwnedTensor LoadMergedRawNK(const TensorResolver& get,
                            const std::vector<std::string>& names, bool is_bf16) {
  return is_bf16 ? LoadMergedBf16RawNK(get, names)
                 : LoadMergedF16ToBf16RawNK(get, names);
}
OwnedTensor LoadMergedVector(const TensorResolver& get,
                             const std::vector<std::string>& names, bool is_bf16) {
  return is_bf16 ? LoadMergedBf16Vector(get, names)
                 : LoadMergedF16ToBf16Vector(get, names);
}

PhiLayerWeights LoadPhiLayer(const TensorResolver& get, int64_t layer, bool is_bf16) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  PhiLayerWeights w;
  w.input_layernorm = LoadDirect(get, base + "input_layernorm.weight", is_bf16);
  w.input_layernorm_bias = LoadDirect(get, base + "input_layernorm.bias", is_bf16);

  // q/k/v ship SEPARATE (WeightsMapper merges them): concat into one raw-NK qkv,
  // and likewise merge the three per-shard biases (q/k/v ALWAYS biased, phi.py:98).
  w.attn.qkv_proj = LoadMergedRawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"},
      is_bf16);
  w.attn.qkv_bias = LoadMergedVector(
      get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"}, is_bf16);
  // dense (o_proj) carries a bias (RowParallelLinear default bias=True).
  w.attn.dense = LoadMergedRawNK(get, {sa + "dense.weight"}, is_bf16);
  w.attn.dense_bias = LoadDirect(get, sa + "dense.bias", is_bf16);

  // Non-gated MLP: fc1 (+bias) -> gelu -> fc2 (+bias). No gate, no merge.
  w.mlp.fc1 = LoadMergedRawNK(get, {mlp + "fc1.weight"}, is_bf16);
  w.mlp.fc1_bias = LoadDirect(get, mlp + "fc1.bias", is_bf16);
  w.mlp.fc2 = LoadMergedRawNK(get, {mlp + "fc2.weight"}, is_bf16);
  w.mlp.fc2_bias = LoadDirect(get, mlp + "fc2.bias", is_bf16);
  return w;
}

// Build the plain partial-rope cos/sin cache (bf16 [max_pos, rotary_dim], [cos|sin]
// halves, indexed by REAL position). phi-2 uses rope_type "default" with
// partial_rotary_factor 0.4 -> rotary_dim = int(head_dim*0.4) = 32. Mirrors the
// StableLM/phi3 default branch (RotaryEmbedding, is_neox_style=true).
OwnedTensor BuildPhiRopeCache(const HfConfig& config) {
  const int64_t head_dim = config.head_dim;
  const int64_t rotary_dim = config.rotary_dim;
  const int64_t max_pos = config.max_position_embeddings;
  const RopeParameters& rp = config.rope_parameters;
  VT_CHECK(rotary_dim > 0 && rotary_dim <= head_dim,
           "phi: rotary_dim must be in (0, head_dim]");
  VT_CHECK(max_pos > 0, "phi: max_position_embeddings must be positive");
  VT_CHECK(rp.rope_type == "default",
           "phi: only default rope is supported (got '" + rp.rope_type + "')");

  RotaryEmbedding rope(head_dim, rotary_dim, max_pos, rp.rope_theta,
                       /*is_neox_style=*/true, vt::DType::kBF16);
  const vt::Tensor cache = rope.cos_sin_cache();  // bf16 [max_pos, rotary_dim]
  VT_CHECK(cache.rank == 2 && cache.shape[1] == rotary_dim,
           "phi: rope cache shape mismatch");
  VT_CHECK(cache.shape[0] >= max_pos, "phi: rope cache too short");

  OwnedTensor out = MakeOwned(vt::DType::kBF16, {max_pos, rotary_dim});
  std::memcpy(out.bytes.data(), cache.data, out.bytes.size());
  return out;
}

}  // namespace

PhiWeights LoadPhiForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "phi: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0, "phi: num_hidden_layers must be positive");

  // Probe the on-disk dtype once: microsoft/phi-2 is F16; a BF16 Phi checkpoint
  // reuses the shared verbatim helpers.
  const std::string& disk_dtype = get("model.embed_tokens.weight").dtype;
  VT_CHECK(disk_dtype == "BF16" || disk_dtype == "F16",
           "phi: unsupported checkpoint dtype '" + disk_dtype +
               "' (expected BF16 or F16)");
  const bool is_bf16 = disk_dtype == "BF16";

  PhiWeights w;
  w.embed_tokens = LoadDirect(get, "model.embed_tokens.weight", is_bf16);
  w.final_norm = LoadDirect(get, "model.final_layernorm.weight", is_bf16);
  w.final_norm_bias = LoadDirect(get, "model.final_layernorm.bias", is_bf16);
  // UNTIED lm_head with a per-vocab bias (phi.py asserts NOT tie_word_embeddings).
  w.lm_head = LoadTransposed(get, "lm_head.weight", is_bf16);  // [vocab,H]->[H,vocab]
  w.lm_head_bias = LoadDirect(get, "lm_head.bias", is_bf16);

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadPhiLayer(get, l, is_bf16));

  w.rope_cos_sin = BuildPhiRopeCache(config);
  return w;
}

}  // namespace vllm
