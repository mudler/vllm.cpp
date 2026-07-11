// vllm.cpp original; see qwen3_5_dense.h. Loader for the DENSE Qwen3.6-27B text
// gate (compressed-tensors NVFP4 W4A4). Mirrors the 35B loader structure
// (qwen3_5_weights.cpp) but routes each Linear bf16 vs W4A4-materialized-to-bf16
// by name (notes §3.6) and swaps the MoE block for the dense SwiGLU MLP.
#include "vllm/model_executor/models/qwen3_5_dense.h"

#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vt/dtype.h"

namespace vllm {

namespace {

using TensorExists = std::function<bool(const std::string&)>;

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "qwen3_5 dense: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

float ReadF32Scalar(const StTensor& t) {
  VT_CHECK(t.data != nullptr && t.nbytes >= sizeof(float),
           "qwen3_5 dense: scalar tensor too small for f32");
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  return v;
}

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
void TransposeBf16(const uint16_t* src, int64_t rows, int64_t cols,
                   uint16_t* dst) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* src_row = src + r * cols;
    for (int64_t c = 0; c < cols; ++c) dst[c * rows + r] = src_row[c];
  }
}

// BF16 tensor copied verbatim (optionally reshaped).
OwnedTensor LoadBf16Direct(const TensorResolver& get, const std::string& name,
                           const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 dense: expected BF16 for " + name);
  std::vector<int64_t> shape = shape_override.empty() ? t.shape : shape_override;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  VT_CHECK(t.nbytes == o.bytes.size(),
           "qwen3_5 dense: byte-size mismatch for " + name);
  std::memcpy(o.bytes.data(), t.data, t.nbytes);
  return o;
}

OwnedTensor LoadBf16OrF32Direct(const TensorResolver& get,
                                const std::string& name,
                                const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  const bool is_bf16 = t.dtype == "BF16";
  const bool is_f32 = t.dtype == "F32";
  VT_CHECK(is_bf16 || is_f32,
           "qwen3_5 dense: expected BF16 or F32 for " + name);
  std::vector<int64_t> shape = shape_override.empty() ? t.shape : shape_override;
  OwnedTensor o = MakeOwned(is_bf16 ? vt::DType::kBF16 : vt::DType::kF32, shape);
  VT_CHECK(t.nbytes == o.bytes.size(),
           "qwen3_5 dense: byte-size mismatch for " + name);
  std::memcpy(o.bytes.data(), t.data, t.nbytes);
  return o;
}

// Preserve the explicit 27B lm_head layout qualified by its GB10 gate. The
// tied 4B head follows the raw-NK path below instead.
OwnedTensor LoadBf16Transposed(const TensorResolver& get,
                               const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 dense: expected BF16 for " + name);
  VT_CHECK(t.shape.size() == 2,
           "qwen3_5 dense: expected 2-D weight for " + name);
  const int64_t out_dim = t.shape[0];
  const int64_t in_dim = t.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(reinterpret_cast<const uint16_t*>(t.data), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// BF16 [out, in] kept RAW in the on-disk torch Linear orientation (nk=true),
// consumed via vt::MatmulBT — the cuBLASLt TN fast path (K contiguous in both
// operands, the layout vLLM's F.linear hits; nvjet TNNN on GB10). Measured
// 2026-07-10: the transposed [K,N] Matmul-B orientation forces cuBLASLt onto
// NNNN/sm80-cutlass kernels — 27B GDN in_proj site 2.29 vs vLLM 1.80 us/tok.
// Also skips the load-time host transpose.
OwnedTensor LoadBf16RawNK(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.shape.size() == 2, "qwen3_5 dense: expected 2-D weight for " + name);
  OwnedTensor o = LoadBf16Direct(get, name);
  o.nk = true;
  return o;
}

OwnedTensor PackBf16RawNK(const std::vector<const OwnedTensor*>& parts,
                          const std::string& label) {
  VT_CHECK(!parts.empty(), "qwen3_5 dense: no tensors to pack for " + label);
  const OwnedTensor* first = parts.front();
  VT_CHECK(first != nullptr && first->dtype == vt::DType::kBF16 &&
               first->rank == 2 && first->nk,
           "qwen3_5 dense: invalid first tensor in " + label);
  const int64_t k = first->shape[1];
  int64_t n = 0;
  for (const OwnedTensor* part : parts) {
    VT_CHECK(part != nullptr && part->dtype == vt::DType::kBF16 &&
                 part->rank == 2 && part->nk && part->shape[1] == k,
             "qwen3_5 dense: incompatible raw-NK tensor in " + label);
    n += part->shape[0];
  }
  OwnedTensor packed = MakeOwned(vt::DType::kBF16, {n, k});
  packed.nk = true;
  size_t offset = 0;
  for (const OwnedTensor* part : parts) {
    std::memcpy(packed.bytes.data() + offset, part->bytes.data(),
                part->bytes.size());
    offset += part->bytes.size();
  }
  VT_CHECK(offset == packed.bytes.size(),
           "qwen3_5 dense: packed byte-size mismatch for " + label);
  return packed;
}

// BF16/F32 [n] -> owned f32 [n] (A_log / dt_bias; bf16 upcast is lossless).
OwnedTensor LoadToF32(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  const bool is_bf16 = t.dtype == "BF16";
  const bool is_f32 = t.dtype == "F32";
  VT_CHECK(is_bf16 || is_f32,
           "qwen3_5 dense: expected BF16 or F32 for " + name);
  VT_CHECK(t.shape.size() == 1,
           "qwen3_5 dense: expected 1-D tensor for " + name);
  const int64_t n = t.shape[0];
  OwnedTensor o = MakeOwned(vt::DType::kF32, {n});
  auto* dst = reinterpret_cast<float*>(o.bytes.data());
  if (is_f32) {
    std::memcpy(dst, t.data, t.nbytes);
  } else {
    const auto* src = reinterpret_cast<const uint16_t*>(t.data);
    for (int64_t i = 0; i < n; ++i) dst[i] = vt::BF16ToF32(src[i]);
  }
  return o;
}

// One compressed-tensors NVFP4 W4A4 Linear -> RAW fp4-resident Nvfp4Weight kept
// in the on-disk [N=out, K=in] orientation vt::MatmulNvfp4 reads directly (notes
// §5 step-6a — the throughput path; NO bf16 materialization). Reads the CT
// tensor names `<proj>.weight_packed` (U8 [out,in/2]), `<proj>.weight_scale`
// (F8_E4M3 [out,in/16], LINEAR non-swizzled) and `<proj>.weight_global_scale`
// (F32 scalar). The CT global scale is stored as a DIVISOR, so scale2 is its
// RECIPROCAL (notes §3.3) — the ONLY math delta vs the modelopt LoadNvfp4Raw;
// the byte encoding (E2M1 nibbles + fp8-e4m3 group-16 scale) is identical, so the
// existing M2.7 tensor-core GEMM carries these bit-for-bit as the CPU dequant
// reference (DequantCtNvfp4WeightToF32). Activation-quant is dropped (bf16
// activations, W4A16-style) — the on-disk `input_global_scale` is not read.
Nvfp4Weight LoadCtNvfp4Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "qwen3_5 dense: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "qwen3_5 dense: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "qwen3_5 dense: expected F8_E4M3 weight_scale for " + proj);
  const float wgs_disk = ReadF32Scalar(get(proj + ".weight_global_scale"));
  VT_CHECK(wgs_disk != 0.0F,
           "qwen3_5 dense: zero weight_global_scale (divisor) for " + proj);

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.weight_global_scale_inv = wgs_disk;  // retain exact divisor for fused linears
  r.scale2 = 1.0F / wgs_disk;  // CT stores 1/scale (divisor) -> reciprocate
  // TRUE W4A4 (notes §7): the on-disk activation divisor drives vt::ScaledFp4Quant
  // (used DIRECTLY), and alpha folds both reciprocated globals for the fp4xfp4
  // GEMM: alpha = (1/input_divisor)·(1/weight_divisor). input_global_scale is a
  // per-tensor F32 scalar present on every 27B quantized Linear (§3.2).
  const float igs_disk = ReadF32Scalar(get(proj + ".input_global_scale"));
  VT_CHECK(igs_disk != 0.0F,
           "qwen3_5 dense: zero input_global_scale (divisor) for " + proj);
  r.input_global_scale_inv = igs_disk;   // on-disk divisor, used directly
  r.alpha = r.scale2 * (1.0F / igs_disk);
  r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  VT_CHECK(packed.nbytes == r.packed.bytes.size(),
           "qwen3_5 dense: packed byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
  r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(),
           "qwen3_5 dense: scale byte-size mismatch for " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  return r;
}

GdnLayerWeights LoadGdnDense(const TensorResolver& get, const TensorExists& has,
                             const std::string& base) {
  const std::string la = base + "linear_attn.";
  GdnLayerWeights g;
  // in_proj_{qkv,z,a,b}: bf16 (ignore list, notes §3.6). Kept raw [N,K]
  // (nk=true -> vt::MatmulBT TN fast path; see LoadBf16RawNK).
  g.in_proj_qkv = LoadBf16RawNK(get, la + "in_proj_qkv.weight");
  g.in_proj_z = LoadBf16RawNK(get, la + "in_proj_z.weight");
  g.in_proj_b = LoadBf16RawNK(get, la + "in_proj_b.weight");
  g.in_proj_a = LoadBf16RawNK(get, la + "in_proj_a.weight");
  g.in_proj_ba =
      PackBf16RawNK({&g.in_proj_b, &g.in_proj_a}, la + "in_proj_ba");
  g.in_proj_b = {};
  g.in_proj_a = {};
  // 27B NVFP4 checkpoints store this as compressed tensors; smaller BF16
  // Qwen3.5 dense checkpoints store an ordinary `.weight`.
  if (has(la + "out_proj.weight_packed")) {
    g.out_proj_fp4 = LoadCtNvfp4Raw(get, la + "out_proj");
  } else {
    g.out_proj = LoadBf16RawNK(get, la + "out_proj.weight");
  }
  // conv1d.weight ships [conv_dim,1,K]; collapse the singleton to [conv_dim,K].
  const StTensor& conv = get(la + "conv1d.weight");
  VT_CHECK(conv.shape.size() == 3 && conv.shape[1] == 1,
           "qwen3_5 dense: unexpected conv1d shape");
  g.conv1d_weight =
      LoadBf16Direct(get, la + "conv1d.weight", {conv.shape[0], conv.shape[2]});
  g.a_log = LoadToF32(get, la + "A_log");
  g.dt_bias = LoadToF32(get, la + "dt_bias");
  g.norm_weight = LoadBf16OrF32Direct(get, la + "norm.weight");
  return g;
}

FullAttnLayerWeights LoadAttnDense(const TensorResolver& get,
                                   const TensorExists& has,
                                   const std::string& base) {
  const std::string sa = base + "self_attn.";
  FullAttnLayerWeights a;
  // q/k/v/o_proj: 27B NVFP4 checkpoints use compressed tensors; smaller BF16
  // Qwen3.5 dense checkpoints use ordinary `.weight` tensors.
  if (has(sa + "q_proj.weight_packed")) {
    a.q_proj_fp4 = LoadCtNvfp4Raw(get, sa + "q_proj");
  } else {
    a.q_proj = LoadBf16RawNK(get, sa + "q_proj.weight");
  }
  if (has(sa + "k_proj.weight_packed")) {
    a.k_proj_fp4 = LoadCtNvfp4Raw(get, sa + "k_proj");
  } else {
    a.k_proj = LoadBf16RawNK(get, sa + "k_proj.weight");
  }
  if (has(sa + "v_proj.weight_packed")) {
    a.v_proj_fp4 = LoadCtNvfp4Raw(get, sa + "v_proj");
  } else {
    a.v_proj = LoadBf16RawNK(get, sa + "v_proj.weight");
  }
  if (has(sa + "o_proj.weight_packed")) {
    a.o_proj_fp4 = LoadCtNvfp4Raw(get, sa + "o_proj");
  } else {
    a.o_proj = LoadBf16RawNK(get, sa + "o_proj.weight");
  }
  a.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  a.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");
  return a;
}

// Dense SwiGLU MLP: gate/up/down all W4A4-quantized -> fp4-resident (§5 6a).
DenseMlpWeights LoadDenseMlp(const TensorResolver& get, const TensorExists& has,
                             const std::string& base) {
  const std::string mlp = base + "mlp.";
  DenseMlpWeights m;
  if (has(mlp + "gate_proj.weight_packed")) {
    m.gate_proj_fp4 = LoadCtNvfp4Raw(get, mlp + "gate_proj");
  } else {
    m.gate_proj = LoadBf16RawNK(get, mlp + "gate_proj.weight");
  }
  if (has(mlp + "up_proj.weight_packed")) {
    m.up_proj_fp4 = LoadCtNvfp4Raw(get, mlp + "up_proj");
  } else {
    m.up_proj = LoadBf16RawNK(get, mlp + "up_proj.weight");
  }
  if (has(mlp + "down_proj.weight_packed")) {
    m.down_proj_fp4 = LoadCtNvfp4Raw(get, mlp + "down_proj");
  } else {
    m.down_proj = LoadBf16RawNK(get, mlp + "down_proj.weight");
  }
  if (!m.gate_proj.Empty() && !m.up_proj.Empty()) {
    m.gate_up_proj =
        PackBf16RawNK({&m.gate_proj, &m.up_proj}, mlp + "gate_up_proj");
    m.gate_proj = {};
    m.up_proj = {};
  }
  return m;
}

}  // namespace

bool IsQwen27QuantizedLinear(const std::string& name) {
  // Never quantized regardless of suffix (notes §3.6 `ignore`).
  if (name.rfind("mtp.", 0) == 0) return false;
  if (name.find("model.visual.") != std::string::npos) return false;
  if (name.find(".linear_attn.in_proj_") != std::string::npos) return false;
  if (name.find("lm_head") != std::string::npos) return false;
  // Quantized set: dense-MLP gate/up/down, self_attn q/k/v/o, GDN out_proj.
  auto ends_with = [&name](const char* suf) {
    const std::string s(suf);
    return name.size() >= s.size() &&
           name.compare(name.size() - s.size(), s.size(), s) == 0;
  };
  if (name.find(".mlp.") != std::string::npos &&
      (ends_with(".gate_proj") || ends_with(".up_proj") ||
       ends_with(".down_proj")))
    return true;
  if (name.find(".self_attn.") != std::string::npos &&
      (ends_with(".q_proj") || ends_with(".k_proj") || ends_with(".v_proj") ||
       ends_with(".o_proj")))
    return true;
  if (ends_with(".linear_attn.out_proj")) return true;
  return false;
}

OwnedTensor MaterializeCtNvfp4Bf16Transposed(const TensorResolver& get,
                                             const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "qwen3_5 dense: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "qwen3_5 dense: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& wscale = get(proj + ".weight_scale");
  VT_CHECK(wscale.dtype == "F8_E4M3",
           "qwen3_5 dense: expected F8_E4M3 weight_scale for " + proj);
  const float wgs_disk = ReadF32Scalar(get(proj + ".weight_global_scale"));

  // Dequant to f32 [out, in] (the divisor is reciprocated inside), then round to
  // bf16 while transposing to Matmul-B layout [in, out].
  std::vector<float> f32(static_cast<size_t>(out_dim) * in_dim);
  DequantCtNvfp4WeightToF32(reinterpret_cast<const uint8_t*>(packed.data),
                            reinterpret_cast<const uint8_t*>(wscale.data),
                            wgs_disk, out_dim, in_dim, f32.data());
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t r = 0; r < out_dim; ++r)
    for (int64_t c = 0; c < in_dim; ++c)
      dst[c * out_dim + r] =
          vt::F32ToBF16(f32[static_cast<size_t>(r) * in_dim + c]);
  return o;
}

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(const TensorResolver& get,
                                               const TensorExists& has,
                                               const std::string& layer_type,
                                               int64_t layer_idx) {
  const std::string base =
      "model.language_model.layers." + std::to_string(layer_idx) + ".";
  Qwen3_5DenseLayerWeights layer;
  layer.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  layer.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  if (layer_type == "linear_attention") {
    layer.is_linear_attention = true;
    layer.gdn = LoadGdnDense(get, has, base);
  } else if (layer_type == "full_attention") {
    layer.is_linear_attention = false;
    layer.attn = LoadAttnDense(get, has, base);
  } else {
    VT_CHECK(false, "qwen3_5 dense: unknown layer_type " + layer_type);
  }
  layer.mlp = LoadDenseMlp(get, has, base);
  return layer;
}

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(const TensorResolver& get,
                                               const std::string& layer_type,
                                               int64_t layer_idx) {
  const TensorExists has = [](const std::string&) { return false; };
  return LoadQwen3_5DenseLayer(get, has, layer_type, layer_idx);
}

Qwen3_5DenseWeights LoadQwen3_5Dense(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3_5 dense: tensor not found: " + name);
    return it->second->Get(name);
  };
  const TensorExists has = [&where](const std::string& name) -> bool {
    return where.find(name) != where.end();
  };

  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 dense: layer_types size must equal num_hidden_layers");

  Qwen3_5DenseWeights w;
  w.embed_tokens =
      LoadBf16Direct(get, "model.language_model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.language_model.norm.weight");
  // The 27B has an explicit unquantized lm_head; smaller Qwen3.5 dense
  // checkpoints tie lm_head to embeddings and omit `lm_head.weight`.
  if (has("lm_head.weight")) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  } else {
    w.tied_lm_head = true;
    w.embed_tokens.nk = true;
  }
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadQwen3_5DenseLayer(
        get, has, config.layer_types[static_cast<size_t>(l)], l));
  return w;
}

bool IsPlainBf16Qwen3_5Dense(const Qwen3_5DenseWeights& weights) {
  for (const Qwen3_5DenseLayerWeights& layer : weights.layers) {
    if (!layer.mlp.gate_proj_fp4.Empty() || !layer.mlp.up_proj_fp4.Empty() ||
        !layer.mlp.down_proj_fp4.Empty())
      return false;
    if (layer.is_linear_attention) {
      if (!layer.gdn.out_proj_fp4.Empty() ||
          !layer.gdn.in_proj_qkv_fp8.Empty() ||
          !layer.gdn.in_proj_z_fp8.Empty() ||
          !layer.gdn.out_proj_fp8.Empty())
        return false;
    } else if (!layer.attn.q_proj_fp4.Empty() ||
               !layer.attn.k_proj_fp4.Empty() ||
               !layer.attn.v_proj_fp4.Empty() ||
               !layer.attn.o_proj_fp4.Empty() ||
               !layer.attn.q_proj_fp8.Empty() ||
               !layer.attn.k_proj_fp8.Empty() ||
               !layer.attn.v_proj_fp8.Empty() ||
               !layer.attn.o_proj_fp8.Empty()) {
      return false;
    }
  }
  return true;
}

size_t ReleaseResidentQwen3_5DenseHostWeights(
    Qwen3_5DenseWeights& weights) {
  size_t released = 0;
  const auto release = [&released](OwnedTensor& tensor) {
    if (tensor.HasHostBytes() && (tensor.d_dev || tensor.d_dev_f32))
      released += tensor.ReleaseHost();
  };
  const auto release_gdn = [&release](GdnLayerWeights& gdn) {
    release(gdn.in_proj_qkv);
    release(gdn.in_proj_z);
    release(gdn.in_proj_b);
    release(gdn.in_proj_a);
    release(gdn.in_proj_ba);
    release(gdn.conv1d_weight);
    release(gdn.a_log);
    release(gdn.dt_bias);
    release(gdn.norm_weight);
    release(gdn.out_proj);
  };
  const auto release_attn = [&release](FullAttnLayerWeights& attn) {
    release(attn.q_proj);
    release(attn.k_proj);
    release(attn.v_proj);
    release(attn.o_proj);
    release(attn.q_norm);
    release(attn.k_norm);
  };

  release(weights.embed_tokens);
  release(weights.final_norm);
  release(weights.lm_head);
  for (Qwen3_5DenseLayerWeights& layer : weights.layers) {
    release(layer.input_layernorm);
    release(layer.post_attention_layernorm);
    if (layer.is_linear_attention)
      release_gdn(layer.gdn);
    else
      release_attn(layer.attn);
    release(layer.mlp.gate_proj);
    release(layer.mlp.up_proj);
    release(layer.mlp.gate_up_proj);
    release(layer.mlp.down_proj);
  }
  return released;
}

}  // namespace vllm
