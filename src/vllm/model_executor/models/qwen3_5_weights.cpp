// vllm.cpp original; see qwen3_5_weights.h. Weight naming/quant verified
// against nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot 491c2f1e
// (.agents/specs/qwen36-forward-notes.md §6).
#include "vllm/model_executor/models/qwen3_5_weights.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/dtype.h"

namespace vllm {

int64_t OwnedTensor::Numel() const {
  if (rank == 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < rank; ++i) n *= shape[i];
  return n;
}

vt::Tensor OwnedTensor::View() const {
  vt::Tensor t;
  t.data = const_cast<uint8_t*>(bytes.data());
  t.dtype = dtype;
  t.device = vt::Device{};  // default = CPU host
  t.rank = rank;
  int64_t stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    t.shape[i] = shape[i];
    t.stride[i] = stride;
    stride *= shape[i];
  }
  return t;
}

namespace {

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "qwen3_5 weights: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

float ReadF32Scalar(const StTensor& t) {
  VT_CHECK(t.data != nullptr && t.nbytes >= sizeof(float),
           "qwen3_5 weights: scalar tensor too small for f32");
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  return v;
}

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
void TransposeBf16(const uint16_t* src, int64_t rows, int64_t cols,
                   uint16_t* dst) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* src_row = src + r * cols;
    for (int64_t c = 0; c < cols; ++c) {
      dst[c * rows + r] = src_row[c];
    }
  }
}

// BF16 tensor copied verbatim (optionally reshaped): shape override lets the
// [conv_dim,1,K] conv weight collapse to [conv_dim,K] with no data change.
OwnedTensor LoadBf16Direct(const TensorResolver& get, const std::string& name,
                           const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  std::vector<int64_t> shape =
      shape_override.empty() ? t.shape : shape_override;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  VT_CHECK(t.nbytes == o.bytes.size(),
           "qwen3_5 weights: byte-size mismatch for " + name);
  std::memcpy(o.bytes.data(), t.data, t.nbytes);
  return o;
}

// BF16 [out, in] -> owned bf16 [in, out] (Matmul-B layout).
OwnedTensor LoadBf16Transposed(const TensorResolver& get,
                               const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  VT_CHECK(t.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + name);
  const int64_t out_dim = t.shape[0];
  const int64_t in_dim = t.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(reinterpret_cast<const uint16_t*>(t.data), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// BF16 [n] -> owned f32 [n] (A_log / dt_bias; upcast is lossless).
OwnedTensor LoadBf16ToF32(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  VT_CHECK(t.shape.size() == 1,
           "qwen3_5 weights: expected 1-D tensor for " + name);
  const int64_t n = t.shape[0];
  OwnedTensor o = MakeOwned(vt::DType::kF32, {n});
  const auto* src = reinterpret_cast<const uint16_t*>(t.data);
  auto* dst = reinterpret_cast<float*>(o.bytes.data());
  for (int64_t i = 0; i < n; ++i) dst[i] = vt::BF16ToF32(src[i]);
  return o;
}

// Native-precision dense projections — keep the 35B's per-tensor FP8 W8A8 attn
// (q/k/v/o) + GDN (in_proj_qkv/z, out_proj) weights RESIDENT in fp8 (LoadFp8Raw)
// + load the static input_scale, so the forward runs a native fp8 W8A8 GEMM
// (static per-tensor activation quant + fp8 tensor cores — either the cutlass
// sm120 kernel or, by default, cuBLASLt fp8; see MatmulFp8CutlassD /
// DenseCublasLtFp8Enabled). Mirrors vLLM's actual scheme (the checkpoint IS
// W8A8) and HALVES those projections' weight bytes vs bf16.
//   DEFAULT ON (VT_DENSE_NATIVE) on a CUDA+cutlass build. VT_DENSE_NATIVE=0 (or
// the legacy VT_FP8_CUTLASS=0) restores the dequant-fp8->bf16-AT-LOAD path
// (LoadFp8Transposed into the bf16 fields + cublas bf16 W8A16) — the previous
// default, kept for the parent's A/B (and the only path on a build WITHOUT the
// fp8 cutlass kernel, guarded by VT_CUTLASS_FP8 so we never route to an
// uncompiled GEMM). The dense NVFP4 sinks (shared-expert + lm_head) are already
// native (fp4-resident + Marlin W4A16) and not gated here.
bool DenseNativeEnabled() {
#ifdef VT_CUTLASS_FP8
  const char* dn = std::getenv("VT_DENSE_NATIVE");
  if (dn != nullptr && dn[0] == '0') return false;
  const char* legacy = std::getenv("VT_FP8_CUTLASS");  // back-compat opt-out
  if (legacy != nullptr && legacy[0] == '0') return false;
  return true;
#else
  return false;
#endif
}

// Per-tensor FP8 projection `<proj>.weight` F8_E4M3 [out,in] + `.weight_scale`
// scalar + `.input_scale` scalar -> RAW fp8-resident W8A8 Fp8Weight (no dequant,
// no transpose: the bytes stay in the on-disk [N=out, K=in] orientation the
// cutlass W8A8 GEMM reads directly). `alpha = input_scale * weight_scale` is
// precomputed (per-tensor scalars) — the folded scalar vt::MatmulFp8Cutlass
// multiplies the fp8 accumulator by (mirror vLLM ScaledEpilogue for per-tensor).
Fp8Weight LoadFp8Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + proj);
  Fp8Weight r;
  r.n = w.shape[0];
  r.k = w.shape[1];
  r.weight_scale = ReadF32Scalar(get(proj + ".weight_scale"));
  r.input_scale = ReadF32Scalar(get(proj + ".input_scale"));
  r.alpha = r.input_scale * r.weight_scale;
  r.packed = MakeOwned(vt::DType::kI8, {r.n, r.k});
  VT_CHECK(w.nbytes == r.packed.bytes.size(),
           "qwen3_5 weights: fp8 byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), w.data, w.nbytes);
  return r;
}

// Per-tensor FP8 projection `<proj>.weight` [out,in] + `<proj>.weight_scale`
// scalar -> owned bf16 [in, out] (dequant then transpose).
OwnedTensor LoadFp8Transposed(const TensorResolver& get,
                              const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + proj);
  const int64_t out_dim = w.shape[0];
  const int64_t in_dim = w.shape[1];
  const float scale = ReadF32Scalar(get(proj + ".weight_scale"));

  std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
  DequantFp8ToBf16(w.data, scale, out_dim * in_dim, dq.data());

  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(dq.data(), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// NVFP4 W4A16 projection `<proj>.weight` U8 [out,in/2] + `.weight_scale` F8
// [out,in/16] + `.weight_scale_2` f32 scalar -> RAW fp4-resident Nvfp4Weight
// (M2.2b). No dequant, no transpose: the bytes are kept in the on-disk
// [N=out, K=in] orientation vt::MatmulNvfp4 reads directly. This is the
// storage refactor that removes the ~40-min CPU dequant + ~70GB bf16 tensors.
Nvfp4Weight LoadNvfp4Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "U8",
           "qwen3_5 weights: expected U8 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D packed weight for " + proj);
  const int64_t out_dim = w.shape[0];
  const int64_t in_dim = w.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 weights: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight_scale");
  const float ws2 = ReadF32Scalar(get(proj + ".weight_scale_2"));

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.scale2 = ws2;
  r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  VT_CHECK(w.nbytes == r.packed.bytes.size(),
           "qwen3_5 weights: packed byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), w.data, w.nbytes);
  r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(),
           "qwen3_5 weights: scale byte-size mismatch for " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  return r;
}

GdnLayerWeights LoadGdn(const TensorResolver& get, const std::string& base) {
  const std::string la = base + "linear_attn.";
  GdnLayerWeights g;
  // W8A8 fp8 (35B), DEFAULT: keep raw fp8 + input_scale, run the native fp8 GEMM.
  // VT_DENSE_NATIVE=0 restores dequant-at-load into the bf16 fields (parent A/B).
  if (DenseNativeEnabled()) {
    g.in_proj_qkv_fp8 = LoadFp8Raw(get, la + "in_proj_qkv");
    g.in_proj_z_fp8 = LoadFp8Raw(get, la + "in_proj_z");
    g.out_proj_fp8 = LoadFp8Raw(get, la + "out_proj");
  } else {
    g.in_proj_qkv = LoadFp8Transposed(get, la + "in_proj_qkv");
    g.in_proj_z = LoadFp8Transposed(get, la + "in_proj_z");
    g.out_proj = LoadFp8Transposed(get, la + "out_proj");
  }
  g.in_proj_b = LoadBf16Transposed(get, la + "in_proj_b.weight");
  g.in_proj_a = LoadBf16Transposed(get, la + "in_proj_a.weight");
  // conv1d.weight ships [conv_dim,1,K]; collapse the singleton to [conv_dim,K].
  const StTensor& conv = get(la + "conv1d.weight");
  VT_CHECK(conv.shape.size() == 3 && conv.shape[1] == 1,
           "qwen3_5 weights: unexpected conv1d shape");
  g.conv1d_weight =
      LoadBf16Direct(get, la + "conv1d.weight", {conv.shape[0], conv.shape[2]});
  g.a_log = LoadBf16ToF32(get, la + "A_log");
  g.dt_bias = LoadBf16ToF32(get, la + "dt_bias");
  g.norm_weight = LoadBf16Direct(get, la + "norm.weight");
  return g;
}

FullAttnLayerWeights LoadAttn(const TensorResolver& get,
                              const std::string& base) {
  const std::string sa = base + "self_attn.";
  FullAttnLayerWeights a;
  // W8A8 fp8 (35B), DEFAULT: keep raw fp8 + input_scale, run the native fp8 GEMM.
  // VT_DENSE_NATIVE=0 restores dequant-at-load into the bf16 fields (parent A/B).
  if (DenseNativeEnabled()) {
    a.q_proj_fp8 = LoadFp8Raw(get, sa + "q_proj");
    a.k_proj_fp8 = LoadFp8Raw(get, sa + "k_proj");
    a.v_proj_fp8 = LoadFp8Raw(get, sa + "v_proj");
    a.o_proj_fp8 = LoadFp8Raw(get, sa + "o_proj");
  } else {
    a.q_proj = LoadFp8Transposed(get, sa + "q_proj");
    a.k_proj = LoadFp8Transposed(get, sa + "k_proj");
    a.v_proj = LoadFp8Transposed(get, sa + "v_proj");
    a.o_proj = LoadFp8Transposed(get, sa + "o_proj");
  }
  a.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  a.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");
  return a;
}

MoeBlockWeights LoadMoe(const TensorResolver& get, const std::string& base,
                        int64_t num_experts) {
  const std::string mlp = base + "mlp.";
  MoeBlockWeights m;
  m.router_gate = LoadBf16Transposed(get, mlp + "gate.weight");
  m.shared_gate = LoadBf16Transposed(get, mlp + "shared_expert_gate.weight");
  // M2.2b: the NVFP4 expert + shared projections are kept fp4-resident (raw
  // packed + scales, no dequant/transpose); the bf16 expert_*/shared_*_proj
  // fields stay EMPTY. The forward calls vt::MatmulNvfp4 on the fp4 fields.
  m.expert_gate_fp4.reserve(static_cast<size_t>(num_experts));
  m.expert_up_fp4.reserve(static_cast<size_t>(num_experts));
  m.expert_down_fp4.reserve(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    const std::string ex = mlp + "experts." + std::to_string(e) + ".";
    m.expert_gate_fp4.push_back(LoadNvfp4Raw(get, ex + "gate_proj"));
    m.expert_up_fp4.push_back(LoadNvfp4Raw(get, ex + "up_proj"));
    m.expert_down_fp4.push_back(LoadNvfp4Raw(get, ex + "down_proj"));
  }
  const std::string se = mlp + "shared_expert.";
  m.shared_gate_proj_fp4 = LoadNvfp4Raw(get, se + "gate_proj");
  m.shared_up_proj_fp4 = LoadNvfp4Raw(get, se + "up_proj");
  m.shared_down_proj_fp4 = LoadNvfp4Raw(get, se + "down_proj");
  return m;
}

}  // namespace

Qwen3_5MoeLayerWeights LoadQwen3_5MoeLayer(const TensorResolver& get,
                                           const std::string& layer_type,
                                           int64_t layer_idx,
                                           int64_t num_experts) {
  const std::string base =
      "model.language_model.layers." + std::to_string(layer_idx) + ".";
  Qwen3_5MoeLayerWeights layer;
  layer.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  layer.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  if (layer_type == "linear_attention") {
    layer.is_linear_attention = true;
    layer.gdn = LoadGdn(get, base);
  } else if (layer_type == "full_attention") {
    layer.is_linear_attention = false;
    layer.attn = LoadAttn(get, base);
  } else {
    VT_CHECK(false, "qwen3_5 weights: unknown layer_type " + layer_type);
  }
  layer.moe = LoadMoe(get, base, num_experts);
  return layer;
}

Qwen3_5MoeWeights LoadQwen3_5Moe(const std::vector<SafetensorsFile>& shards,
                                 const HfConfig& config) {
  // Build a name -> shard index from each shard's own header. mtp.* tensors are
  // ignored (not requested); the resolver throws on any missing name.
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) where[name] = &shard;
  }
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3_5 weights: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 weights: layer_types size must equal num_hidden_layers");
  VT_CHECK(config.num_experts > 0,
           "qwen3_5 weights: num_experts must be > 0 for the MoE model");

  Qwen3_5MoeWeights w;
  w.embed_tokens =
      LoadBf16Direct(get, "model.language_model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.language_model.norm.weight");
  w.lm_head_fp4 = LoadNvfp4Raw(get, "lm_head");  // M2.2b fp4-resident
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    w.layers.push_back(LoadQwen3_5MoeLayer(
        get, config.layer_types[static_cast<size_t>(l)], l, config.num_experts));
  }
  return w;
}

}  // namespace vllm
