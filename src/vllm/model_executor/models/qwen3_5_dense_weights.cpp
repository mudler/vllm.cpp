// vllm.cpp original; see qwen3_5_dense.h. Loader for the DENSE Qwen3.6-27B text
// gate (compressed-tensors NVFP4 W4A4). Mirrors the 35B loader structure
// (qwen3_5_weights.cpp) but routes each Linear bf16 vs W4A4-materialized-to-bf16
// by name (notes §3.6) and swaps the MoE block for the dense SwiGLU MLP.
#include "vllm/model_executor/models/qwen3_5_dense.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm {

// The generic BF16 copy/transpose/merge routines now live in the shared
// dense_weight_loaders.h (SEAM GAP #3 extraction) so qwen3_weights.cpp reuses
// them. Re-expose them unqualified for the 27B call sites below — behavior and
// loaded bytes are byte-identical to the former anon-namespace copies.
using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::MakeOwned;

namespace {

using TensorExists = std::function<bool(const std::string&)>;

float ReadF32Scalar(const StTensor& t) {
  VT_CHECK(t.data != nullptr && t.nbytes >= sizeof(float),
           "qwen3_5 dense: scalar tensor too small for f32");
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  return v;
}

// The GDN in-projections stay RAW in the on-disk torch Linear [out, in]
// orientation (nk=true, via LoadMergedBf16RawNK), consumed by vt::MatmulBT —
// the cuBLASLt TN fast path (K contiguous in both operands, the layout vLLM's
// F.linear hits; nvjet TNNN on GB10). Measured 2026-07-10: the transposed
// [K,N] Matmul-B orientation forces cuBLASLt onto NNNN/sm80-cutlass kernels —
// 27B GDN in_proj site 2.29 vs vLLM 1.80 us/tok. Raw NK also skips the
// load-time host transpose.

// Model-dtype vectors are BF16 in the NVFP4 gate checkpoint but F32 in ordinary
// Qwen3.5 dense checkpoints. Normalize them to the representation consumed by
// the existing forward without changing the quantized path.
OwnedTensor LoadModelBf16Direct(
    const TensorResolver& get, const std::string& name,
    const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16" || t.dtype == "F32",
           "qwen3_5 dense: expected BF16 or F32 for " + name);
  const std::vector<int64_t> shape =
      shape_override.empty() ? t.shape : shape_override;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  if (t.dtype == "BF16") {
    VT_CHECK(t.nbytes == o.bytes.size(),
             "qwen3_5 dense: byte-size mismatch for " + name);
    std::memcpy(o.bytes.data(), t.data, t.nbytes);
  } else {
    VT_CHECK(t.nbytes == static_cast<size_t>(o.Numel()) * sizeof(float),
             "qwen3_5 dense: byte-size mismatch for " + name);
    auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
    const auto* src = static_cast<const uint8_t*>(t.data);
    for (int64_t i = 0; i < o.Numel(); ++i) {
      float value = 0.0F;
      std::memcpy(&value, src + static_cast<size_t>(i) * sizeof(value),
                  sizeof(value));
      dst[i] = vt::F32ToBF16(value);
    }
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

OwnedTensor LoadBf16RawNK(const TensorResolver& get,
                          const std::string& name) {
  OwnedTensor out = LoadBf16Direct(get, name);
  VT_CHECK(out.rank == 2,
           "qwen3_5 dense: expected 2-D weight for " + name);
  out.nk = true;
  return out;
}

// BF16/F32 [n] -> owned f32 [n] (A_log / dt_bias).
OwnedTensor LoadToF32(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16" || t.dtype == "F32",
           "qwen3_5 dense: expected BF16 or F32 for " + name);
  VT_CHECK(t.shape.size() == 1,
           "qwen3_5 dense: expected 1-D tensor for " + name);
  const int64_t n = t.shape[0];
  OwnedTensor o = MakeOwned(vt::DType::kF32, {n});
  auto* dst = reinterpret_cast<float*>(o.bytes.data());
  if (t.dtype == "F32") {
    VT_CHECK(t.nbytes == o.bytes.size(),
             "qwen3_5 dense: byte-size mismatch for " + name);
    std::memcpy(dst, t.data, t.nbytes);
  } else {
    const auto* src = reinterpret_cast<const uint16_t*>(t.data);
    for (int64_t i = 0; i < n; ++i) dst[i] = vt::BF16ToF32(src[i]);
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

bool DirectDeviceLoadEligible(vt::Queue* queue) {
  if (queue == nullptr ||
      !platforms::GetPlatform(queue->device.type).needs_weight_staging()) {
    return false;
  }
  const char* release = std::getenv("VT_RELEASE_HOST_WEIGHTS");
  if (release != nullptr && release[0] == '0') return false;
  const char* direct = std::getenv("VT_DIRECT_DEVICE_LOAD");
  if (direct != nullptr && direct[0] == '0') return false;
  const auto& platform = platforms::GetPlatform(queue->device.type);
  return !platform.is_unified_memory() &&
         platform.residency_policy().release_host_weights_after_upload;
}

void StageAndReleaseLoadedDense(Qwen3_5DenseWeights& weights,
                                vt::Queue& queue) {
  Qwen3_5DenseModel::PrepareBf16Resident(weights, queue);
  vt::GetBackend(queue.device.type).Synchronize(queue);
  (void)ReleaseResidentQwen3_5DenseHostWeights(weights);
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
  MaybeReleaseSourcePages(packed.data, packed.nbytes);
  r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(),
           "qwen3_5 dense: scale byte-size mismatch for " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  MaybeReleaseSourcePages(ws.data, ws.nbytes);
  return r;
}

GdnLayerWeights LoadGdnDense(const TensorResolver& get, const TensorExists& has,
                             const std::string& base) {
  const std::string la = base + "linear_attn.";
  GdnLayerWeights g;
  // in_proj_{qkv,z,a,b}: bf16 (ignore list, notes §3.6). Kept raw [N,K]
  // (nk=true -> vt::MatmulBT TN fast path). Mirror vLLM's TWO physical
  // MergedColumnParallelLinear owners (qwen3_5.py:203-210 stacked mapping +
  // qwen_gdn_linear_attn.py:481-496): in_proj_qkvz in exact [q,k,v,z] row
  // order (the checkpoint's in_proj_qkv already stacks q|k|v; z appended) and
  // in_proj_ba in exact [b,a] row order. The rollback paths take non-owning
  // row slices of these owners, so the split fields deliberately stay empty.
  g.in_proj_qkvz = LoadMergedBf16RawNK(
      get, {la + "in_proj_qkv.weight", la + "in_proj_z.weight"});
  g.in_proj_ba = LoadMergedBf16RawNK(
      get, {la + "in_proj_b.weight", la + "in_proj_a.weight"});
  // NVFP4 checkpoints use compressed tensors; ordinary checkpoints use raw
  // torch-Linear BF16 [N,K].
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
  g.norm_weight = LoadModelBf16Direct(get, la + "norm.weight");
  return g;
}

FullAttnLayerWeights LoadAttnDense(const TensorResolver& get,
                                   const TensorExists& has,
                                   const std::string& base) {
  const std::string sa = base + "self_attn.";
  FullAttnLayerWeights a;
  const auto load_projection = [&](const std::string& name, Nvfp4Weight& fp4,
                                   OwnedTensor& plain) {
    if (has(name + ".weight_packed"))
      fp4 = LoadCtNvfp4Raw(get, name);
    else
      plain = LoadBf16RawNK(get, name + ".weight");
  };
  load_projection(sa + "q_proj", a.q_proj_fp4, a.q_proj);
  load_projection(sa + "k_proj", a.k_proj_fp4, a.k_proj);
  load_projection(sa + "v_proj", a.v_proj_fp4, a.v_proj);
  load_projection(sa + "o_proj", a.o_proj_fp4, a.o_proj);
  a.q_norm = LoadModelBf16Direct(get, sa + "q_norm.weight");
  a.k_norm = LoadModelBf16Direct(get, sa + "k_norm.weight");
  return a;
}

// Dense SwiGLU MLP: gate/up/down all W4A4-quantized -> fp4-resident (§5 6a).
DenseMlpWeights LoadDenseMlp(const TensorResolver& get, const TensorExists& has,
                             const std::string& base) {
  const std::string mlp = base + "mlp.";
  DenseMlpWeights m;
  if (has(mlp + "gate_proj.weight_packed")) {
    m.gate_proj_fp4 = LoadCtNvfp4Raw(get, mlp + "gate_proj");
    m.up_proj_fp4 = LoadCtNvfp4Raw(get, mlp + "up_proj");
    m.down_proj_fp4 = LoadCtNvfp4Raw(get, mlp + "down_proj");
  } else {
    m.gate_up_proj = dense_loaders::LoadMergedBf16RawNK(
        get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
    m.down_proj = LoadBf16RawNK(get, mlp + "down_proj.weight");
  }
  return m;
}

}  // namespace

OwnedTensor LoadMergedBf16RawNK(const TensorResolver& get,
                                const std::vector<std::string>& names) {
  // Extracted to the shared dense_weight_loaders.h (SEAM GAP #3); this retains
  // the public vllm::LoadMergedBf16RawNK API (used by the 27B GDN loader below
  // and test_qwen27_dense_forward) as a byte-identical thin forward.
  return dense_loaders::LoadMergedBf16RawNK(get, names);
}

GdnLayerWeights LoadQwen3_5DenseGdn(const TensorResolver& get,
                                    const std::string& layer_base) {
  // Public focused-loader seam historically describes the 27B checkpoint.
  const TensorExists has = [](const std::string&) { return true; };
  return LoadGdnDense(get, has, layer_base);
}

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
  MaybeReleaseSourcePages(packed.data, packed.nbytes);
  MaybeReleaseSourcePages(wscale.data, wscale.nbytes);
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
  layer.input_layernorm =
      LoadModelBf16Direct(get, base + "input_layernorm.weight");
  layer.post_attention_layernorm =
      LoadModelBf16Direct(get, base + "post_attention_layernorm.weight");
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
  // The public resolver-only seam is used by the compressed-tensors parity
  // fixture, where every routed projection is NVFP4.
  const TensorExists has = [](const std::string&) { return true; };
  return LoadQwen3_5DenseLayer(get, has, layer_type, layer_idx);
}

Qwen3_5DenseWeights LoadQwen3_5Dense(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config,
                                     vt::Queue* load_queue) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3_5 dense: tensor not found: " + name);
    return it->second->Get(name);
  };
  const TensorExists has = [&where](const std::string& name) {
    return where.find(name) != where.end();
  };

  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 dense: layer_types size must equal num_hidden_layers");

  Qwen3_5DenseWeights w;
  w.embed_tokens =
      LoadBf16Direct(get, "model.language_model.embed_tokens.weight");
  w.final_norm =
      LoadModelBf16Direct(get, "model.language_model.norm.weight");
  // The 27B owns an explicit head; smaller Qwen3.5 checkpoints tie logits to
  // the embedding table and omit lm_head.weight.
  if (has("lm_head.weight")) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  } else {
    w.tied_lm_head = true;
    w.embed_tokens.nk = true;
  }
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  bool direct_device = DirectDeviceLoadEligible(load_queue);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    w.layers.push_back(LoadQwen3_5DenseLayer(
        get, has, config.layer_types[static_cast<size_t>(l)], l));
    if (direct_device) {
      direct_device = IsPlainBf16Qwen3_5Dense(w);
      if (direct_device) StageAndReleaseLoadedDense(w, *load_queue);
    }
  }
  return w;
}

bool IsPlainBf16Qwen3_5Dense(const Qwen3_5DenseWeights& weights) {
  for (const Qwen3_5DenseLayerWeights& layer : weights.layers) {
    if (!layer.mlp.gate_proj_fp4.Empty() || !layer.mlp.up_proj_fp4.Empty() ||
        !layer.mlp.down_proj_fp4.Empty()) {
      return false;
    }
    if (layer.is_linear_attention) {
      if (!layer.gdn.out_proj_fp4.Empty() ||
          !layer.gdn.in_proj_qkv_fp8.Empty() ||
          !layer.gdn.in_proj_z_fp8.Empty() ||
          !layer.gdn.out_proj_fp8.Empty()) {
        return false;
      }
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
    if (tensor.HasHostBytes() && (tensor.d_dev || tensor.d_dev_f32)) {
      released += tensor.bytes.size();
      tensor.ReleaseHost();
    }
  };
  const auto release_gdn = [&release](GdnLayerWeights& gdn) {
    release(gdn.in_proj_qkv);
    release(gdn.in_proj_z);
    release(gdn.in_proj_qkvz);
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
