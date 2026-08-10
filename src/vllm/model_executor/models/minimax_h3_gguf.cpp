// MiniMax-H3 GGUF arm — the quantized checkpoints that actually FIT one GB10.
//
// ─── WHY THIS EXISTS ─────────────────────────────────────────────────────────
// The bf16 MiniMax-H3 release is ~354 GB and needs 4x B300, which is why the
// safetensors arm is hardware-blocked here. The community ComfyUI-format GGUFs
// change that verdict: the DiT at Q3_K_M is ~15.6 GB and the Qwen3-VL encoder at
// Q4_K_M is ~14.6 GB, so a whole working set (plus the two fp16/fp32 VAEs) lands
// around ~41 GB — comfortably inside the 119 GiB unified pool. See
// .agents/specs/minimax-h3.md section 0.
//
// ─── THE NAME MAP IS THE IDENTITY (verified against a real checkpoint) ───────
// ComfyUI's H3 GGUF keeps the checkpoint's own parameter names, so every one of
// the 535 tensors in `MiniMax-H3-FL2VA-Q3_K_M.gguf` matches the contract
// `EnumerateMiniMaxH3DitTensors` derives from upstream source — no rename table.
// That equivalence is GATED on the real manifest
// (tests/vllm/models/minimax_h3_gguf_manifest.inc, produced by
// scripts/gen-minimax-h3-gguf-manifest.py from the file's header alone).
//
// Two shape rules, and they are the whole subtlety:
//   1. GGUF `ne` is REVERSED relative to torch: a `[out, in]` weight is stored
//      `[in, out]` (qkv_proj is ne=[5376, 21504] for logical [21504, 5376]).
//   2. When ComfyUI had to RESHAPE a tensor so its fastest axis is a whole
//      number of quant blocks, it records the true torch shape in the metadata
//      key `comfy.gguf.orig_shape.<name>`, which is already in `[out, in]`
//      order and therefore must NOT be reversed. The 50 AdaLN projections are
//      exactly this case: logical [96768, 2688], but 2688 is not a multiple of
//      the 256-element Q3_K block, so they ship as ne=[256, 1016064].
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"

namespace vllm {

namespace {

// Parse the trailing integer of a `prefix<N>.` path segment; -1 when absent.
int64_t IndexAfter(const std::string& name, const std::string& prefix) {
  if (name.compare(0, prefix.size(), prefix) != 0) return -1;
  size_t pos = prefix.size();
  if (pos >= name.size() || name[pos] < '0' || name[pos] > '9') return -1;
  int64_t value = 0;
  while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
    value = value * 10 + (name[pos] - '0');
    ++pos;
  }
  return value;
}

}  // namespace

// The logical (torch) shape of a tensor from its RAW GGUF `ne` dims: reverse the
// ne order, unless ComfyUI recorded `orig_shape`, which is already torch order.
// NOTE the asymmetry with `EnumerateMiniMaxH3GgufTensors` below: `GgufTensorInfo::shape`
// has ALREADY been reversed by the reader, so that path must not reverse again.
// This overload is the one that speaks the on-disk order, and it is what the
// real-manifest gate exercises.
std::vector<int64_t> MiniMaxH3GgufLogicalShape(const std::vector<int64_t>& gguf_dims,
                                               const std::vector<int64_t>& orig_shape) {
  if (!orig_shape.empty()) return orig_shape;
  std::vector<int64_t> shape(gguf_dims.rbegin(), gguf_dims.rend());
  // GGUF pads trailing dims with 1; drop the leading 1s the reversal produces so
  // a 1-D tensor stays 1-D.
  while (shape.size() > 1 && shape.front() == 1) shape.erase(shape.begin());
  return shape;
}

// Derive the H3 geometry from the tensor manifest alone. A ComfyUI GGUF carries
// no transformer config.json, so the shapes ARE the config: this is what lets a
// GGUF checkpoint load without the original repo.
MiniMaxH3DitParams ParseMiniMaxH3DitParamsFromGgufManifest(
    const std::vector<MiniMaxH3TensorSpec>& manifest) {
  MiniMaxH3DitParams p;
  int64_t max_block = -1, max_refiner = -1;
  bool saw_hidden = false, saw_head_dim = false;
  // PRUNED (curve) vs unpruned: the two forms are mutually exclusive, and BOTH
  // want to set time_embed_dim, so which one the file carries is recorded and
  // checked rather than left to iteration order.
  bool saw_curve = false, saw_time_embedder = false;

  for (const MiniMaxH3TensorSpec& spec : manifest) {
    const std::string& name = spec.name;
    max_block = std::max(max_block, IndexAfter(name, "blocks."));
    max_refiner = std::max(max_refiner, IndexAfter(name, "token_refiner.blocks."));

    if (name == "final_layer.norm.weight" && spec.shape.size() == 1) {
      p.hidden_size = spec.shape[0];
      saw_hidden = true;
    } else if (name == "blocks.0.attn.q_norm.weight" && spec.shape.size() == 1) {
      p.attention_head_dim = spec.shape[0];
      saw_head_dim = true;
    } else if (name == "condition_proj.weight" && spec.shape.size() == 2) {
      p.text_dim = spec.shape[1];
    } else if (name == "video_patch_proj.weight" && spec.shape.size() == 2) {
      // [hidden, latents_dim * patch volume]; the patch is fixed at (1,2,2).
      p.latents_dim = spec.shape[1] / (p.patch_size_t * p.patch_size_h * p.patch_size_w);
    } else if (name == "audio_patch_proj.weight" && spec.shape.size() == 2) {
      p.audio_latents_dim = spec.shape[1];
    } else if (name == "time_embedder.proj_in.weight" && spec.shape.size() == 2) {
      p.time_embed_hidden_size = spec.shape[0];
      p.timestep_input_dim = spec.shape[1];
      saw_time_embedder = true;
    } else if (name == "time_embedder.proj_out.weight" && spec.shape.size() == 2) {
      p.time_embed_dim = spec.shape[0];
      saw_time_embedder = true;
    } else if (name == "adaln_t_table" && spec.shape.size() == 2) {
      // [grid, curve width]; the width IS the AdaLN linear's in_features
      // (comfy/ldm/minimax/model.py:429), so it replaces time_embed_dim.
      p.adaln_curve_grid = spec.shape[0];
      p.time_embed_dim = spec.shape[1];
      saw_curve = true;
    } else if (name == "blocks.0.mlp.fc2.weight" && spec.shape.size() == 2) {
      p.ffn_hidden_size = spec.shape[1];
    } else if (name == "blocks.0.adaln_proj.linear.weight" && spec.shape.size() == 2) {
      p.adaln_out_features = spec.shape[0];
    } else if (name == "final_layer.adaln_proj.linear.weight" && spec.shape.size() == 2) {
      p.final_adaln_out_features = spec.shape[0];
    } else if (name == "rope.inv_freq" && spec.shape.size() == 1) {
      p.rope_inv_freq_len = spec.shape[0];
    }
  }

  VT_CHECK(saw_hidden, "minimax_h3 gguf: final_layer.norm.weight is required to size the model");
  VT_CHECK(saw_head_dim, "minimax_h3 gguf: blocks.0.attn.q_norm.weight is required");
  VT_CHECK(max_block >= 0, "minimax_h3 gguf: no blocks.<N> tensors found");
  p.num_layers = max_block + 1;
  p.token_refiner_num_layers = max_refiner + 1;

  // MHA: qkv is 3 * heads * head_dim wide, so the head count follows.
  for (const MiniMaxH3TensorSpec& spec : manifest) {
    if (spec.name == "blocks.0.attn.qkv_proj.weight" && spec.shape.size() == 2) {
      VT_CHECK(spec.shape[0] % (3 * p.attention_head_dim) == 0,
               "minimax_h3 gguf: qkv width is not 3 * heads * head_dim");
      p.num_attention_heads = spec.shape[0] / (3 * p.attention_head_dim);
    }
  }
  VT_CHECK(p.num_attention_heads > 0, "minimax_h3 gguf: could not derive the head count");
  VT_CHECK(p.adaln_out_features == 6 * p.hidden_size * kMiniMaxH3AdalnModalityNum,
           "minimax_h3 gguf: adaln width does not match 6 * hidden * 3");
  // Spec 8.20 stop condition: a file carrying BOTH forms is a third checkpoint
  // shape, not this one, and would load with half the modulation path unbound.
  VT_CHECK(!(saw_curve && saw_time_embedder),
           "minimax_h3 gguf: checkpoint carries BOTH adaln_t_table and time_embedder.* "
           "-- the pruned and unpruned forms are mutually exclusive");
  VT_CHECK(saw_curve || saw_time_embedder,
           "minimax_h3 gguf: checkpoint carries NEITHER adaln_t_table nor time_embedder.*");
  VT_CHECK(p.adaln_curve_grid == 0 || p.adaln_curve_grid >= 2,
           "minimax_h3 gguf: adaln_t_table needs at least two rows to interpolate");
  VT_CHECK(p.rope_rot_dim() <= p.attention_head_dim,
           "minimax_h3 gguf: 6 * rope_inv_freq_len exceeds attention_head_dim");
  return p;
}

// Read the manifest (names + logical shapes + ggml types) out of a GGUF.
std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3GgufTensors(const GgufFile& file) {
  std::vector<MiniMaxH3TensorSpec> out;
  out.reserve(file.Tensors().size());
  for (const GgufTensorInfo& info : file.Tensors()) {
    MiniMaxH3TensorSpec spec;
    spec.name = info.name;
    // `info.shape` is already torch row-major (the reader reverses ne), so it is
    // used as-is; `orig_shape` overrides it when ComfyUI reshaped the tensor for
    // quant-block alignment.
    spec.shape = info.shape;
    if (const GgufValue* kv = file.FindKv("comfy.gguf.orig_shape." + info.name)) {
      if (const GgufArray* array = std::get_if<GgufArray>(&kv->v)) {
        std::vector<int64_t> orig;
        orig.reserve(array->elems.size());
        for (const GgufValue& elem : array->elems) {
          switch (elem.TypeId()) {
            case kGgufU32: orig.push_back(std::get<uint32_t>(elem.v)); break;
            case kGgufI32: orig.push_back(std::get<int32_t>(elem.v)); break;
            case kGgufU64: orig.push_back(static_cast<int64_t>(std::get<uint64_t>(elem.v))); break;
            case kGgufI64: orig.push_back(std::get<int64_t>(elem.v)); break;
            default:
              VT_CHECK(false, "minimax_h3 gguf: comfy.gguf.orig_shape must hold integers");
          }
        }
        if (!orig.empty()) spec.shape = orig;
      }
    }
    // The fp32 islands stay unquantized in the GGUF too (the ComfyUI quantizer
    // leaves norms, biases, and the small projections alone).
    spec.fp32 = info.ggml_type == 0;
    out.push_back(std::move(spec));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Loading: manifest -> dequantized DiT weights
// ---------------------------------------------------------------------------

// Materialize every DiT tensor from a ComfyUI-format GGUF into owned f32 buffers
// and hand back the non-owning `MiniMaxH3DitWeights` views the forward consumes.
//
// Each tensor is dequantized to f32 by the shared `DequantGgufRowToF32`, so the
// K-quant families the H3 GGUFs actually use (Q2_K / Q3_K / Q4_K, plus the F32 and
// F16 islands) are handled by the same code path every other GGUF model uses.
// Shapes are resolved by the rules gated in the manifest test: reversed `ne`,
// unless `comfy.gguf.orig_shape.<name>` says otherwise.
// Bind the forward's non-owning views onto the owned buffers. Shared by the GGUF
// and NVFP4 arms — both land on the SAME weight contract, so both bind identically.
void BindMiniMaxH3DitViews(MiniMaxH3GgufDit* out) {
  // Bind the views. Any missing name throws by NAME rather than yielding a null
  // tensor the forward would read as zeros.
  const MiniMaxH3DitParams& p = out->params;
  auto view = [out](const std::string& name) -> vt::Tensor {
    const auto quant = out->quant_storage.find(name);
    const auto bf16 = out->bf16_storage.find(name);
    const auto it = out->storage.find(name);
    VT_CHECK(quant != out->quant_storage.end() || bf16 != out->bf16_storage.end() ||
                 it != out->storage.end(),
             "minimax_h3 gguf: checkpoint is missing a required tensor");
    const std::vector<int64_t>& shape = out->shapes.at(name);
    vt::Tensor t;
    // A keep-quant tensor keeps its ggml bytes and its BLOCK dtype; the logical
    // [N,K] shape is unchanged, which is what lets the forward stay identical.
    if (quant != out->quant_storage.end()) {
      t.data = quant->second.data();
      t.dtype = out->quant_dtype.at(name);
    } else if (bf16 != out->bf16_storage.end()) {
      t.data = bf16->second.data();
      t.dtype = vt::DType::kBF16;
    } else {
      t.data = it->second.data();
      t.dtype = vt::DType::kF32;
    }
    t.device = vt::Device{};
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = t.rank - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    return t;
  };

  out->weights.video_patch_proj_w = view("video_patch_proj.weight");
  out->weights.video_patch_proj_b = view("video_patch_proj.bias");
  out->weights.audio_patch_proj_w = view("audio_patch_proj.weight");
  out->weights.audio_patch_proj_b = view("audio_patch_proj.bias");
  out->weights.condition_proj_w = view("condition_proj.weight");
  out->weights.condition_proj_b = view("condition_proj.bias");
  // The PRUNED form has no time embedder at all; it binds the curve table
  // instead (comfy/ldm/minimax/model.py:428-432). Binding the wrong one throws
  // by NAME, which is what makes a mismatched --dit fail loudly.
  if (p.use_adaln_curves()) {
    out->weights.adaln_t_table = view("adaln_t_table");
  } else {
    out->weights.time_proj_in_w = view("time_embedder.proj_in.weight");
    out->weights.time_proj_in_b = view("time_embedder.proj_in.bias");
    out->weights.time_proj_out_w = view("time_embedder.proj_out.weight");
    out->weights.time_proj_out_b = view("time_embedder.proj_out.bias");
  }
  out->weights.rope_inv_freq = view("rope.inv_freq");

  auto bind_block = [&](const std::string& prefix, bool with_adaln) {
    MiniMaxH3DitBlockWeights block;
    block.norm1 = view(prefix + ".norm1.weight");
    block.norm2 = view(prefix + ".norm2.weight");
    block.qkv_proj = view(prefix + ".attn.qkv_proj.weight");
    block.q_norm = view(prefix + ".attn.q_norm.weight");
    block.k_norm = view(prefix + ".attn.k_norm.weight");
    block.out_proj = view(prefix + ".attn.out_proj.weight");
    block.fc1 = view(prefix + ".mlp.fc1.weight");
    block.fc2 = view(prefix + ".mlp.fc2.weight");
    if (with_adaln) {
      block.adaln_w = view(prefix + ".adaln_proj.linear.weight");
      block.adaln_b = view(prefix + ".adaln_proj.linear.bias");
      out->weights.blocks.push_back(block);
    } else {
      out->weights.refiner.push_back(block);
    }
  };
  for (int64_t i = 0; i < p.token_refiner_num_layers; ++i) {
    bind_block("token_refiner.blocks." + std::to_string(i), /*with_adaln=*/false);
  }
  out->weights.refiner_final_norm = view("token_refiner.final_norm.weight");
  for (int64_t i = 0; i < p.num_layers; ++i) {
    bind_block("blocks." + std::to_string(i), /*with_adaln=*/true);
  }
  out->weights.final_norm = view("final_layer.norm.weight");
  out->weights.final_adaln_w = view("final_layer.adaln_proj.linear.weight");
  out->weights.final_adaln_b = view("final_layer.adaln_proj.linear.bias");
  out->weights.video_out_w = view("final_layer.video_out.weight");
  out->weights.video_out_b = view("final_layer.video_out.bias");
  out->weights.audio_out_w = view("final_layer.audio_out.weight");
  out->weights.audio_out_b = view("final_layer.audio_out.bias");
}

MiniMaxH3GgufDit LoadMiniMaxH3DitFromGguf(const GgufFile& file, bool keep_quant) {
  MiniMaxH3GgufDit out;
  const std::vector<MiniMaxH3TensorSpec> manifest = EnumerateMiniMaxH3GgufTensors(file);
  out.params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);

  for (const MiniMaxH3TensorSpec& spec : manifest) {
    const GgufTensorInfo& info = file.Get(spec.name);
    int64_t numel = 1;
    for (int64_t d : spec.shape) numel *= d;
    VT_CHECK(numel > 0, "minimax_h3 gguf: tensor has an empty logical shape");
    out.shapes[spec.name] = spec.shape;

    // KEEP-QUANT eligibility, the SHARED rule: a supported block encoding, rank 2
    // (a projection, not a norm or bias), and K a whole number of blocks. Anything
    // else dequantizes, so the decision is total and the forward is unaffected.
    vt::DType block = vt::DType::kF32;
    const bool eligible = keep_quant && spec.shape.size() == 2 &&
                          KeepQuantDType(info.ggml_type, &block) &&
                          spec.shape[1] % vt::BlockElems(block) == 0;
    if (eligible) {
      const size_t bytes =
          static_cast<size_t>(spec.shape[0]) * vt::RowSizeBytes(block, spec.shape[1]);
      const uint8_t* src = static_cast<const uint8_t*>(info.data);
      out.quant_storage[spec.name].assign(src, src + bytes);
      out.quant_dtype[spec.name] = block;
      out.quant_ggml_type[spec.name] = info.ggml_type;
      continue;
    }

    // Dequantize once per tensor, keyed by checkpoint name.
    out.storage[spec.name] = DequantGgufRowToF32(info.ggml_type, info.data, numel);
    VT_CHECK(static_cast<int64_t>(out.storage[spec.name].size()) == numel,
             "minimax_h3 gguf: dequant produced the wrong element count");
  }

  BindMiniMaxH3DitViews(&out);
  return out;
}


MiniMaxH3GgufDit LoadMiniMaxH3DitFromGgufBf16(const GgufFile& file) {
  MiniMaxH3GgufDit out;
  const std::vector<MiniMaxH3TensorSpec> manifest = EnumerateMiniMaxH3GgufTensors(file);
  out.params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  for (const MiniMaxH3TensorSpec& spec : manifest) {
    const GgufTensorInfo& info = file.Get(spec.name);
    int64_t numel = 1;
    for (int64_t d : spec.shape) numel *= d;
    VT_CHECK(numel > 0, "minimax_h3 gguf bf16: tensor has an empty logical shape");
    // The fp32 ISLANDS stay f32 here, consulted from the SINGLE source rather
    // than re-listed. Two distinct hazards live behind this one predicate:
    //
    //   * `rope.inv_freq` and `adaln_t_table` are read on the HOST through
    //     `Ptr<float>()`, an unchecked cast — a bf16 buffer is reinterpreted as
    //     garbage floats, not converted (#244).
    //   * the patch projections, the time embedder and the output heads are
    //     upstream's fp32 island (minimax_h3_transformer.py:85-101). Those are
    //     correctly TYPED when bound, so they are not garbage, but rounding them
    //     here silently makes this path lower precision than every other loader.
    //
    // Re-listing names is what let those two drift apart: this loader named two
    // of the seven and no gate noticed, because the rule was documented as
    // binding on "all four staging paths" and this host loader is a fifth.
    if (MiniMaxH3IsFp32IslandTensor(spec.name)) {
      out.storage[spec.name] = DequantGgufRowToF32(info.ggml_type, info.data, numel);
      out.shapes[spec.name] = spec.shape;
      continue;
    }
    // Straight to bf16: going via f32 would double the peak for no benefit, and the
    // f32 intermediate is exactly what does not fit.
    out.bf16_storage[spec.name] = DequantGgufRowToBf16(info.ggml_type, info.data, numel);
    VT_CHECK(static_cast<int64_t>(out.bf16_storage[spec.name].size()) == numel,
             "minimax_h3 gguf bf16: dequant produced the wrong element count");
    out.shapes[spec.name] = spec.shape;
  }
  BindMiniMaxH3DitViews(&out);
  return out;
}

}  // namespace vllm
