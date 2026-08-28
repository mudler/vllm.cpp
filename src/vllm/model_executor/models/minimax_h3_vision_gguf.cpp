// MiniMax-H3 encoder VISION tower — GGUF `visual.*` loader (image/video conditioning).
//
// The H3 encoder is a fine-tuned Qwen3-VL, so its ViT is the architecture the project
// already ports as `multimodal::Qwen3VLVisionForward`. This file is the ONE piece the
// record reconciliation (spec §8.8) found missing: a loader that fills the shared
// `Qwen3VLVisionWeights` from the encoder GGUF's `visual.*` tensors, so the tower can
// run on REAL weights instead of the synthetic reduced-dim fixtures it was gated on.
//
// It mirrors the safetensors `LoadQwen3VLVisionWeights` (qwen3_vl.cpp) with two GGUF
// deltas:
//   * NAME: the GGUF drops the `model.` level — the ViT is `visual.*`, not
//     `model.visual.*` — so the prefix is just `visual.`.
//   * ENCODING: every projection is a ggml BLOCK type (Q4_K / Q5_K) and the patch/pos
//     tensors are F16, so each is DEQUANTIZED to f32 via DequantGgufRowToF32. The
//     ComfyUI export reshapes a non-256-aligned row (hidden 1152 -> 4.5 Q_K blocks) to
//     ne0=256; dequantizing the WHOLE flat buffer is reshape-agnostic and preserves the
//     row-major [out,in] order the tower reads (it indexes every weight as a flat buffer
//     with dims from the config), so no `comfy.gguf.orig_shape` metadata is needed.
#include "vllm/model_executor/models/minimax_h3.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"

namespace vllm {

multimodal::Qwen3VLVisionConfig MiniMaxH3EncoderVisionConfig() {
  multimodal::Qwen3VLVisionConfig cfg;
  // Measured from ~/h3fp4/ckpt/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf (visual.* shapes) +
  // the shared Qwen3.6-27B vision config (state.md 2026-07-25). See the header.
  cfg.hidden_size = 1152;
  cfg.num_heads = 16;                 // head_dim 72
  cfg.depth = 27;
  cfg.intermediate_size = 4304;
  cfg.out_hidden_size = 5120;         // == encoder text dim
  cfg.patch_size = 16;
  cfg.temporal_patch_size = 2;
  cfg.spatial_merge_size = 2;
  cfg.num_position_embeddings = 2304;  // 48^2
  cfg.in_channels = 3;
  // H3's encoder carries 3 REAL DeepStack mergers (visual.deepstack_merger_list.{0,1,2}),
  // UNLIKE the Qwen3.6-27B (empty). The WHICH-layers taps are not in the weights-only
  // GGUF but are now CONFIRMED against the release config: MiniMax-H3's text_encoder is
  // Qwen3-VL-32B-Instruct, whose `vision_config.deepstack_visual_indexes = [8, 16, 24]`
  // (depth 27) — identical to the Qwen3-VL-MoE vision tower (vllm-omni
  // `Qwen3VLMoeVisionConfig` default, and the public Qwen/Qwen3-VL-30B-A3B config.json).
  // The #86 inference was correct (spec §8.8).
  cfg.deepstack_visual_indexes = {8, 16, 24};
  cfg.norm_eps = 1e-6f;
  return cfg;
}

namespace {

// Dequantize one `visual.*` tensor to a flat f32 buffer, whatever its ggml encoding
// (F32 / F16 / Q4_K / Q5_K / Q6_K). numel = prod(shape); the tower reads the result as
// a flat [out, in] row-major buffer with dims taken from the config.
std::vector<float> LoadVisionGgufF32(const GgufFile& file, const std::string& name) {
  const GgufTensorInfo& info = file.Get(name);
  int64_t numel = 1;
  for (int64_t d : info.shape) numel *= d;
  return DequantGgufRowToF32(info.ggml_type, static_cast<const uint8_t*>(info.data), numel);
}

// The tower's host store is bf16 bits (#1359), so the dequantized f32 narrows
// once HERE instead of once per upload inside `MakeDevBf16`. Same function, same
// input, same output — the single narrowing simply moves from upload time to
// load time, which is why this path's device bytes do not change either. The
// narrowing MUST stay `vt::F32ToBF16`: a second, truncating rounding function
// would change the GGUF towers' numbers and no token gate would see it.
std::vector<uint16_t> ToBf16Bits(const std::vector<float>& f) {
  std::vector<uint16_t> out(f.size());
  for (size_t i = 0; i < f.size(); ++i) out[i] = vt::F32ToBF16(f[i]);
  return out;
}

}  // namespace

multimodal::Qwen3VLVisionWeights LoadQwen3VLVisionFromGguf(
    const GgufFile& file, const multimodal::Qwen3VLVisionConfig& cfg) {
  // Membership set: GgufFile has no has() query, and a missing tensor should name
  // itself rather than surface as a generic "not found".
  std::set<std::string> present;
  for (const GgufTensorInfo& info : file.Tensors()) present.insert(info.name);
  auto load_f32 = [&](const std::string& name) -> std::vector<float> {
    VT_CHECK(present.count(name) != 0,
             "minimax_h3 vision gguf: missing tensor " + name +
                 " (is this the encoder GGUF, and does it carry the visual.* tower?)");
    return LoadVisionGgufF32(file, name);
  };
  auto load = [&](const std::string& name) { return ToBf16Bits(load_f32(name)); };

  multimodal::Qwen3VLVisionWeights vw;
  const std::string V = "visual.";
  vw.patch_proj_w = load(V + "patch_embed.proj.weight");
  vw.patch_proj_b = load(V + "patch_embed.proj.bias");
  // The pos-embed table stays host f32: `VisionPosEmbedInterpolate` gathers and
  // sums it before anything narrows, and on THIS path the dequantized value is a
  // genuine f32 that narrowing would truncate (qwen3_vl_vision.h, `pos_embed_w`).
  vw.pos_embed_w = load_f32(V + "pos_embed.weight");

  vw.blocks.resize(static_cast<size_t>(cfg.depth));
  for (int64_t l = 0; l < cfg.depth; ++l) {
    const std::string p = V + "blocks." + std::to_string(l);
    multimodal::VisionBlockWeights& b = vw.blocks[static_cast<size_t>(l)];
    b.norm1_w = load(p + ".norm1.weight");
    b.norm1_b = load(p + ".norm1.bias");
    b.norm2_w = load(p + ".norm2.weight");
    b.norm2_b = load(p + ".norm2.bias");
    b.qkv_w = load(p + ".attn.qkv.weight");
    b.qkv_b = load(p + ".attn.qkv.bias");
    b.proj_w = load(p + ".attn.proj.weight");
    b.proj_b = load(p + ".attn.proj.bias");
    b.fc1_w = load(p + ".mlp.linear_fc1.weight");
    b.fc1_b = load(p + ".mlp.linear_fc1.bias");
    b.fc2_w = load(p + ".mlp.linear_fc2.weight");
    b.fc2_b = load(p + ".mlp.linear_fc2.bias");
  }

  // The main merger norms the PRE-shuffle width (dim); the DeepStack mergers norm the
  // POST-shuffle width (merge^2 * dim) — same split the safetensors loader encodes.
  auto load_merger = [&](const std::string& prefix, bool postshuffle) {
    multimodal::VisionMergerWeights m;
    m.use_postshuffle_norm = postshuffle;
    m.norm_w = load(prefix + ".norm.weight");
    m.norm_b = load(prefix + ".norm.bias");
    m.fc1_w = load(prefix + ".linear_fc1.weight");
    m.fc1_b = load(prefix + ".linear_fc1.bias");
    m.fc2_w = load(prefix + ".linear_fc2.weight");
    m.fc2_b = load(prefix + ".linear_fc2.bias");
    return m;
  };
  vw.merger = load_merger(V + "merger", /*postshuffle=*/false);
  for (size_t i = 0; i < cfg.deepstack_visual_indexes.size(); ++i) {
    vw.deepstack_mergers.push_back(
        load_merger(V + "deepstack_merger_list." + std::to_string(i), /*postshuffle=*/true));
  }
  return vw;
}

}  // namespace vllm
