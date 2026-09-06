// The llama.cpp `clip` mmproj reader. Contract, upstream anchors and the reason
// each refusal exists: `include/vllm/model_executor/models/clip_mmproj_gguf.h`.
#include "vllm/model_executor/models/clip_mmproj_gguf.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/models/muse_glimmer_gguf_weights.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// The `clip.*` metadata keys, spelled as llama.cpp writes them
// (clip-impl.h KEY_* with the "%s" modality slot filled with "vision").
constexpr const char* kKvArch = "general.architecture";
constexpr const char* kKvType = "general.type";
constexpr const char* kKvProjType = "clip.projector_type";
constexpr const char* kKvEmbd = "clip.vision.embedding_length";
constexpr const char* kKvFf = "clip.vision.feed_forward_length";
constexpr const char* kKvBlocks = "clip.vision.block_count";
constexpr const char* kKvProjDim = "clip.vision.projection_dim";
constexpr const char* kKvHeads = "clip.vision.attention.head_count";
constexpr const char* kKvEps = "clip.vision.attention.layer_norm_epsilon";
constexpr const char* kKvPatch = "clip.vision.patch_size";
constexpr const char* kKvMerge = "clip.vision.spatial_merge_size";

// The `v.*` / `mm.*` tensor names (clip-impl.h TN_*, with the "%s" modality
// prefix filled with "v" as clip.cpp does for a vision projector).
constexpr const char* kTnPatchEmbd = "v.patch_embd.weight";
constexpr const char* kTnPatchEmbd1 = "v.patch_embd.weight.1";
constexpr const char* kTnPatchBias = "v.patch_embd.bias";
constexpr const char* kTnPosEmbd = "v.position_embd.weight";

std::string KvString(const GgufFile& gguf, const char* key) {
  const GgufValue* v = gguf.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufString) return std::string();
  return std::get<std::string>(v->v);
}

// Widen any GGUF integer spelling to int64. llama.cpp writes these keys as u32,
// but a converter is free to use any width, and a silently-rejected key would
// become a wrong-shaped tower rather than an error.
int64_t KvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    default:
      throw std::runtime_error("clip mmproj gguf: key " + key +
                               " is not an integer");
  }
}

int64_t ReqInt(const GgufFile& gguf, const char* key) {
  const GgufValue* v = gguf.FindKv(key);
  VT_CHECK(v != nullptr,
           std::string("clip mmproj gguf: missing metadata key ") + key);
  return KvInt(*v, key);
}

int64_t OptInt(const GgufFile& gguf, const char* key, int64_t dflt) {
  const GgufValue* v = gguf.FindKv(key);
  return v == nullptr ? dflt : KvInt(*v, key);
}

double ReqFloat(const GgufFile& gguf, const char* key) {
  const GgufValue* v = gguf.FindKv(key);
  VT_CHECK(v != nullptr,
           std::string("clip mmproj gguf: missing metadata key ") + key);
  if (v->TypeId() == kGgufF32) return std::get<float>(v->v);
  if (v->TypeId() == kGgufF64) return std::get<double>(v->v);
  return static_cast<double>(KvInt(*v, key));
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

int64_t Numel(const GgufTensorInfo& info) {
  int64_t n = 1;
  for (int64_t d : info.shape) n *= d;
  return n;
}

}  // namespace

bool IsClipMmprojGguf(const GgufFile& gguf) {
  return KvString(gguf, kKvArch) == kClipGgufArch;
}

std::string ClipProjectorType(const GgufFile& gguf) {
  return KvString(gguf, kKvProjType);
}

void RefuseUnsupportedClipMmproj(const GgufFile& gguf,
                                 const std::string& path) {
  const std::string arch = KvString(gguf, kKvArch);
  VT_CHECK(arch == kClipGgufArch,
           "--mmproj: '" + path + "' is not a multimodal projector: its "
           "general.architecture is '" +
               (arch.empty() ? std::string("<absent>") : arch) +
               "', and a projector file carries '" + kClipGgufArch +
               "'. Pass the language GGUF as the model and the mmproj-*.gguf "
               "here, not the other way round");
  // `general.type` is advisory: llama.cpp writes "mmproj", but the load-bearing
  // discriminator is the architecture above plus the projector type below, so a
  // file missing the key is accepted rather than refused on metadata alone.
  const std::string type = KvString(gguf, kKvType);
  VT_CHECK(type.empty() || type == kClipGgufTypeMmproj,
           "--mmproj: '" + path + "' declares general.type '" + type +
               "', not '" + kClipGgufTypeMmproj + "'");

  const std::string proj = ClipProjectorType(gguf);
  // The already-recorded refusal, now reached from production rather than only
  // from `tests/vllm/models/test_muse_glimmer_gguf.cpp`. Its message names the
  // exact missing axis and the workaround; restating it here would let the two
  // drift.
  if (proj == kClipProjectorMuseGlimmer) MuseGlimmerRefuseMmproj();
  VT_CHECK(proj == kClipProjectorQwen3VL,
           "--mmproj: '" + path + "' has clip.projector_type '" +
               (proj.empty() ? std::string("<absent>") : proj) +
               "'; this build loads '" + kClipProjectorQwen3VL +
               "' projectors only");
}

multimodal::Qwen3VLVisionConfig ClipMmprojVisionConfig(const GgufFile& gguf) {
  multimodal::Qwen3VLVisionConfig cfg;
  cfg.hidden_size = ReqInt(gguf, kKvEmbd);
  cfg.num_heads = ReqInt(gguf, kKvHeads);
  cfg.depth = ReqInt(gguf, kKvBlocks);
  cfg.intermediate_size = ReqInt(gguf, kKvFf);
  cfg.out_hidden_size = ReqInt(gguf, kKvProjDim);
  cfg.patch_size = ReqInt(gguf, kKvPatch);
  // No `clip.*` key states it, and llama.cpp's temporal merge is exactly two
  // frames (qwen2vl.cpp::build_inp_with_temporal_merge). The join below
  // ENFORCES the pair, so this constant cannot silently disagree with the file.
  cfg.temporal_patch_size = 2;
  cfg.spatial_merge_size = OptInt(gguf, kKvMerge, 2);
  cfg.norm_eps = static_cast<float>(ReqFloat(gguf, kKvEps));

  // Read from the tensors, because the file states them nowhere else.
  // `GgufTensorInfo::shape` is the on-disk ggml dims REVERSED into torch
  // row-major order, so `v.patch_embd.weight` is torch [hidden, C, p, p] and
  // `v.position_embd.weight` is torch [num_position_embeddings, hidden].
  const GgufTensorInfo& patch = gguf.Get(kTnPatchEmbd);
  VT_CHECK(patch.shape.size() == 4,
           std::string("clip mmproj gguf: ") + kTnPatchEmbd +
               " must be 4-D [out, C, p, p], got " + ShapeText(patch.shape));
  cfg.in_channels = patch.shape[1];
  const GgufTensorInfo& pos = gguf.Get(kTnPosEmbd);
  VT_CHECK(pos.shape.size() == 2,
           std::string("clip mmproj gguf: ") + kTnPosEmbd +
               " must be 2-D [num_position_embeddings, hidden], got " +
               ShapeText(pos.shape));
  cfg.num_position_embeddings = pos.shape[0];

  // DeepStack taps are named by the LAYER they tap
  // (clip.cpp::load_tensors reads TN_DEEPSTACK_* with `il`), so the present
  // names ARE the indexes. An mmproj without them (Qwen3.8-27B's, whose
  // `deepstack_visual_indexes` is `[]`) yields an empty list, which compiles
  // that leg out exactly as upstream does — not-applicable, not owed.
  cfg.deepstack_visual_indexes.clear();
  for (int64_t l = 0; l < cfg.depth; ++l) {
    const std::string probe =
        "v.deepstack." + std::to_string(l) + ".fc1.weight";
    for (const GgufTensorInfo& info : gguf.Tensors()) {
      if (info.name == probe) {
        cfg.deepstack_visual_indexes.push_back(static_cast<int>(l));
        break;
      }
    }
  }
  return cfg;
}

multimodal::Qwen3VLVisionWeights LoadQwen3VLVisionFromClipMmproj(
    const GgufFile& gguf, const multimodal::Qwen3VLVisionConfig& cfg) {
  // Membership set: GgufFile has no `has()` query, and a missing tensor must
  // name itself rather than surface as a generic "not found". Same shape as
  // minimax_h3_vision_gguf.cpp.
  std::set<std::string> present;
  for (const GgufTensorInfo& info : gguf.Tensors()) present.insert(info.name);
  auto has = [&](const std::string& name) { return present.count(name) != 0; };
  auto load_f32 = [&](const std::string& name) -> std::vector<float> {
    VT_CHECK(has(name), "clip mmproj gguf: missing tensor " + name +
                            " (is this a " + kClipProjectorQwen3VL +
                            " projector?)");
    const GgufTensorInfo& info = gguf.Get(name);
    return DequantGgufRowToF32(info.ggml_type, info.data, Numel(info));
  };
  // The tower's host store is bf16 bits (#1359). The dequantized f32 therefore
  // narrows ONCE here rather than once per upload inside `MakeDevBf16`: same
  // `vt::F32ToBF16`, same input, same output, so the device bytes do not move.
  // Writing a second, truncating narrow here instead would change this tower's
  // numbers and no token gate would see it.
  auto load = [&](const std::string& name) {
    const std::vector<float> f = load_f32(name);
    std::vector<uint16_t> out(f.size());
    for (size_t i = 0; i < f.size(); ++i) out[i] = vt::F32ToBF16(f[i]);
    return out;
  };

  multimodal::Qwen3VLVisionWeights vw;

  // ── The patch embedding: TWO conv2d halves joined into ONE conv3d operand ──
  //
  // llama.cpp stores `conv3d(in=C, out=hidden, k=(2, p, p))` as two conv2d
  // weights and SUMS their outputs over the two temporal frames
  // (qwen2vl.cpp::build_inp_with_temporal_merge). Each half is torch
  // [hidden, C, p, p]. Our `patch_proj_w` is the conv3d weight flattened in
  // torch order [hidden, C, T, p, p] = [hidden, C * T * p * p], because that is
  // the layout `Qwen3VLVisionForward` reads and the layout the safetensors
  // `patch_embed.proj.weight` already has (vLLM Qwen3_VisionPatchEmbed reshapes
  // pixel_values to (-1, C, T, p, p) before the conv).
  //
  // So the join INTERLEAVES per channel: half 0 supplies t = 0 and half 1
  // supplies t = 1, inside each channel's p*p block. A concatenation instead of
  // an interleave produces a tower that runs and is wrong.
  VT_CHECK(has(kTnPatchEmbd),
           std::string("clip mmproj gguf: missing tensor ") + kTnPatchEmbd);
  const GgufTensorInfo& half0 = gguf.Get(kTnPatchEmbd);
  VT_CHECK(
      has(kTnPatchEmbd1),
      std::string("clip mmproj gguf: this projector carries '") + kTnPatchEmbd +
          "' " + ShapeText(half0.shape) + " but NOT '" + kTnPatchEmbd1 +
          "', so it holds only " +
          std::to_string(cfg.in_channels * cfg.patch_size * cfg.patch_size) +
          " of the " +
          std::to_string(cfg.temporal_patch_size * cfg.in_channels *
                         cfg.patch_size * cfg.patch_size) +
          " input features the temporal patch embedding needs "
          "(temporal_patch_size * C * patch_size^2). The temporal half of the "
          "weight is ABSENT from the file, so loading it would mean inventing "
          "it. Use a safetensors checkpoint for image and video until the "
          "converter emits both halves");
  const GgufTensorInfo& half1 = gguf.Get(kTnPatchEmbd1);
  VT_CHECK(half0.shape == half1.shape,
           std::string("clip mmproj gguf: ") + kTnPatchEmbd + " " +
               ShapeText(half0.shape) + " and " + kTnPatchEmbd1 + " " +
               ShapeText(half1.shape) + " must have the same shape");
  const int64_t out = half0.shape[0];
  const int64_t spatial = cfg.in_channels * cfg.patch_size * cfg.patch_size;
  VT_CHECK(half0.shape[1] == cfg.in_channels &&
               half0.shape[2] == cfg.patch_size &&
               half0.shape[3] == cfg.patch_size,
           std::string("clip mmproj gguf: ") + kTnPatchEmbd + " is " +
               ShapeText(half0.shape) + ", expected [out, " +
               std::to_string(cfg.in_channels) + ", " +
               std::to_string(cfg.patch_size) + ", " +
               std::to_string(cfg.patch_size) + "]");
  VT_CHECK(out == cfg.hidden_size,
           std::string("clip mmproj gguf: ") + kTnPatchEmbd + " has out=" +
               std::to_string(out) + " but clip.vision.embedding_length is " +
               std::to_string(cfg.hidden_size));
  const std::vector<uint16_t> w0 = load(kTnPatchEmbd);
  const std::vector<uint16_t> w1 = load(kTnPatchEmbd1);
  const int64_t plane = cfg.patch_size * cfg.patch_size;
  const int64_t tp = cfg.temporal_patch_size;
  vw.patch_proj_w.assign(static_cast<size_t>(out * spatial * tp), uint16_t{0});
  for (int64_t o = 0; o < out; ++o) {
    for (int64_t c = 0; c < cfg.in_channels; ++c) {
      const int64_t src = (o * cfg.in_channels + c) * plane;
      const int64_t dst = (o * cfg.in_channels + c) * tp * plane;
      for (int64_t i = 0; i < plane; ++i) {
        vw.patch_proj_w[static_cast<size_t>(dst + i)] =
            w0[static_cast<size_t>(src + i)];
        vw.patch_proj_w[static_cast<size_t>(dst + plane + i)] =
            w1[static_cast<size_t>(src + i)];
      }
    }
  }
  vw.patch_proj_b = load(kTnPatchBias);
  // The pos-embed table stays host f32: `VisionPosEmbedInterpolate` gathers and
  // sums it before anything narrows, and on THIS path the dequantized value is a
  // genuine f32 that narrowing would truncate (qwen3_vl_vision.h, `pos_embed_w`).
  vw.pos_embed_w = load_f32(kTnPosEmbd);

  // ── The blocks ────────────────────────────────────────────────────────────
  // qwen3vl reads a MERGED qkv (clip.cpp reads TN_ATTN_QKV and
  // qwen3vl.cpp::build slices it), which is our `qkv_w` [3*hidden, hidden]
  // directly. `ffn_up` is our fc1 and `ffn_down` is our fc2; the legacy
  // up/down swap in clip.cpp::load_tensors explicitly EXCLUDES
  // PROJECTOR_TYPE_QWEN3VL, so no swap here.
  vw.blocks.resize(static_cast<size_t>(cfg.depth));
  for (int64_t l = 0; l < cfg.depth; ++l) {
    const std::string p = "v.blk." + std::to_string(l) + ".";
    multimodal::VisionBlockWeights& b = vw.blocks[static_cast<size_t>(l)];
    b.norm1_w = load(p + "ln1.weight");
    b.norm1_b = load(p + "ln1.bias");
    b.norm2_w = load(p + "ln2.weight");
    b.norm2_b = load(p + "ln2.bias");
    b.qkv_w = load(p + "attn_qkv.weight");
    b.qkv_b = load(p + "attn_qkv.bias");
    b.proj_w = load(p + "attn_out.weight");
    b.proj_b = load(p + "attn_out.bias");
    b.fc1_w = load(p + "ffn_up.weight");
    b.fc1_b = load(p + "ffn_up.bias");
    b.fc2_w = load(p + "ffn_down.weight");
    b.fc2_b = load(p + "ffn_down.bias");
  }

  // ── The merger ────────────────────────────────────────────────────────────
  // `v.post_ln` is applied to the PRE-shuffle width, before the reshape to
  // hidden * merge^2 (qwen3vl.cpp::build applies post_ln, THEN reshapes), so it
  // is the merger's norm with `use_postshuffle_norm = false` — the same split
  // the safetensors loader encodes. `mm.0` / `mm.2` are fc1 / fc2
  // (clip.cpp PROJECTOR_TYPE_QWEN3VL reads TN_LLAVA_PROJ 0 and 2).
  vw.merger.use_postshuffle_norm = false;
  vw.merger.norm_w = load("v.post_ln.weight");
  vw.merger.norm_b = load("v.post_ln.bias");
  vw.merger.fc1_w = load("mm.0.weight");
  vw.merger.fc1_b = load("mm.0.bias");
  vw.merger.fc2_w = load("mm.2.weight");
  vw.merger.fc2_b = load("mm.2.bias");

  // DeepStack mergers norm the POST-shuffle width (qwen3vl.cpp reshapes to
  // n_embd * merge_factor and THEN norms), which is `use_postshuffle_norm`.
  for (int idx : cfg.deepstack_visual_indexes) {
    const std::string p = "v.deepstack." + std::to_string(idx) + ".";
    multimodal::VisionMergerWeights m;
    m.use_postshuffle_norm = true;
    m.norm_w = load(p + "norm.weight");
    m.norm_b = load(p + "norm.bias");
    m.fc1_w = load(p + "fc1.weight");
    m.fc1_b = load(p + "fc1.bias");
    m.fc2_w = load(p + "fc2.weight");
    m.fc2_b = load(p + "fc2.bias");
    vw.deepstack_mergers.push_back(std::move(m));
  }
  return vw;
}

// ─── Tensor accounting (QUANT-QWEN38-27B-GGUF-ARM, issue #821) ──────────────

std::vector<std::string> Qwen3VLClipMmprojExpectedTensors(
    const multimodal::Qwen3VLVisionConfig& cfg) {
  std::vector<std::string> out;
  // The patch embedding, BOTH temporal halves, and the position table.
  out.emplace_back(kTnPatchEmbd);
  out.emplace_back(kTnPatchEmbd1);
  out.emplace_back(kTnPatchBias);
  out.emplace_back(kTnPosEmbd);
  for (int64_t l = 0; l < cfg.depth; ++l) {
    const std::string p = "v.blk." + std::to_string(l) + ".";
    for (const char* stem :
         {"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
          "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight",
          "attn_out.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight",
          "ffn_down.bias"}) {
      out.push_back(p + stem);
    }
  }
  // The merger: `v.post_ln` is the PRE-shuffle norm and `mm.0` / `mm.2` are
  // fc1 / fc2. There is no `mm.1` in a qwen3vl_merger export.
  for (const char* name : {"v.post_ln.weight", "v.post_ln.bias", "mm.0.weight",
                           "mm.0.bias", "mm.2.weight", "mm.2.bias"}) {
    out.emplace_back(name);
  }
  for (int idx : cfg.deepstack_visual_indexes) {
    const std::string p = "v.deepstack." + std::to_string(idx) + ".";
    for (const char* stem : {"norm.weight", "norm.bias", "fc1.weight",
                             "fc1.bias", "fc2.weight", "fc2.bias"}) {
      out.push_back(p + stem);
    }
  }
  return out;
}

void RefuseUnaccountedClipMmproj(const GgufFile& gguf,
                                 const multimodal::Qwen3VLVisionConfig& cfg,
                                 const std::string& path) {
  const std::vector<std::string> want = Qwen3VLClipMmprojExpectedTensors(cfg);
  const std::set<std::string> wanted(want.begin(), want.end());
  std::vector<std::string> extra;
  for (const GgufTensorInfo& t : gguf.Tensors()) {
    if (wanted.count(t.name) == 0) extra.push_back(t.name);
  }
  if (extra.empty()) return;
  constexpr size_t kMaxNamed = 12;
  std::string names;
  for (size_t i = 0; i < extra.size() && i < kMaxNamed; ++i) {
    names += (i == 0 ? "" : ", ") + extra[i];
  }
  if (extra.size() > kMaxNamed) {
    names += ", ... (" + std::to_string(extra.size() - kMaxNamed) + " more)";
  }
  VT_CHECK(false,
           "--mmproj: '" + path + "' carries " + std::to_string(extra.size()) +
               " tensor(s) that this build's " + kClipProjectorQwen3VL +
               " reader NEVER reads, out of " +
               std::to_string(gguf.Tensors().size()) + " present against " +
               std::to_string(wanted.size()) + " enumerated for depth " +
               std::to_string(cfg.depth) + " and " +
               std::to_string(cfg.deepstack_visual_indexes.size()) +
               " deepstack tap(s): " + names +
               ". Loading it would drop them SILENTLY and build a tower that "
               "runs and is wrong");
}

// ─── DeepSeek-V4 Flash Vision (`deepseek4v`) ────────────────────────────────
// Contract, upstream anchors and the reason each refusal exists:
// `include/vllm/model_executor/models/clip_mmproj_gguf.h`.
namespace {

// The two `clip.*` keys the `qwen3vl_merger` arm above does not read.
constexpr const char* kKvScaleFactor = "clip.vision.projector.scale_factor";
constexpr const char* kKvUseSilu = "clip.use_silu";

// The `v.*` / `mm.*` names a `deepseek4v` export carries and the arm above does
// not (clip-impl.h TN_LN_POST, TN_LLAVA_PROJ, TN_TOK_IMG_START/_END/_PAD,
// TN_IMAGE_NEWLINE). `mm.1` and `mm.2` are the aligner's two projections: there
// is no `mm.0` in a deepseek4v export, which is the mirror image of
// `qwen3vl_merger` having no `mm.1`.
constexpr const char* kTnPostLn = "v.post_ln.weight";
constexpr const char* kTnMm1Weight = "mm.1.weight";
constexpr const char* kTnMm1Bias = "mm.1.bias";
constexpr const char* kTnMm2Weight = "mm.2.weight";
constexpr const char* kTnMm2Bias = "mm.2.bias";
constexpr const char* kTnImgStart = "v.token_embd.img_start";
constexpr const char* kTnImgEnd = "v.token_embd.img_end";
constexpr const char* kTnImgPad = "v.token_embd.img_pad";
constexpr const char* kTnImageNewline = "v.image_newline";

// The thirteen tensors ONE `deepseek4v` block carries IN THE SHIPPED VEHICLE.
// Attention arrives as three SEPARATE projections and the MLP as three separate
// matrices, so this list is what makes the block count 13 rather than 12.
//
// The split is a property of the FILE, not of the family: the pinned converter
// emits the FUSED `attn_qkv` (see the header). `kTnFusedQkvProbe` below is how
// a fused file is recognised and refused by name instead of being blamed for
// carrying tensors this build never reads.
constexpr const char* kDeepSeekV4BlockTensors[] = {
    "attn_q.weight",   "attn_q.bias",   "attn_k.weight",  "attn_k.bias",
    "attn_v.weight",   "attn_v.bias",   "attn_out.weight", "attn_out.bias",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ln1.weight",
    "ln2.weight",
};

// Layer 0 always exists in a projector this reader would otherwise accept, so
// its fused attention weight is a sufficient probe for the whole file.
constexpr const char* kTnFusedQkvProbe = "v.blk.0.attn_qkv.weight";

// The largest `clip.vision.block_count` this reader will honour. It is not a
// capability limit; it is the boundary between a geometry a projector can
// plausibly declare and one that only reaches `std::vector::resize`. The
// shipped artifact is depth 32 and no published vision tower is near this, so
// a file above it is corrupt or hostile rather than new.
constexpr int64_t kMaxDeepSeekV4Depth = 1024;

// The same boundary for every WIDTH. These do not reach a `resize` on their
// own, but they multiply into one (`aligner_input_size()` is
// hidden * ratio^2), so an unbounded pair is the same defect one step removed.
constexpr int64_t kMaxGeometry = 1 << 20;

// The largest element count any ONE tensor this reader materializes may have.
//
// BOUNDING THE FACTORS IS NOT BOUNDING THE PRODUCT, and the comment above named
// that defect and then chose a bound that does not contain it. `kMaxGeometry`
// admits an `embedding_length` of 65536, a sixteenth of what it allows, and the
// fused qkv buffer was reserved at `3 * hidden * hidden` BEFORE the first
// file-shaped read, so nothing about the file bounded it: 12,884,901,888
// elements at 65536, and about 6.6 TB at the permitted maximum. A 1.6 MB
// projector declaring 65536 at `patch_size` 1, carrying only the four tensors
// read before that point, passed every refusal in this file and threw a bare
// `std::bad_alloc`.
//
// The shipped artifact's largest tensor is `mm.1` at 4096 * 1024 * 9 =
// 37,748,736 elements, so this ceiling is about seven times the real thing. A
// projector above it is corrupt or hostile rather than new, and it is refused
// with the keys whose product produced it.
constexpr int64_t kMaxTensorElements = 1 << 28;

// The product of `factors`, SATURATED at one past the ceiling rather than
// wrapped. Four factors at `kMaxGeometry` is 2^80, which overflows int64 and
// silently becomes a small positive number -- the same defect this bound exists
// to refuse, one step further removed again.
int64_t SaturatingElements(std::initializer_list<int64_t> factors) {
  int64_t product = 1;
  for (int64_t f : factors) {
    if (f <= 0 || f > kMaxTensorElements / product) return kMaxTensorElements + 1;
    product *= f;
  }
  return product;
}

std::string DeepSeekV4BlockPrefix(int64_t layer) {
  return "v.blk." + std::to_string(layer) + ".";
}

bool HasTensor(const GgufFile& gguf, const std::string& name) {
  for (const GgufTensorInfo& info : gguf.Tensors()) {
    if (info.name == name) return true;
  }
  return false;
}

// One geometry field, refused BY NAME rather than by the allocation that would
// follow. `KvInt` widens every integer spelling, so a signed `block_count` of
// -1 and an unsigned one of four billion both arrive here, and both become a
// `size_t` at the `resize` below. A `length_error` or a `bad_alloc` names
// neither the file nor the key, and this path runs on a user-supplied
// `--mmproj`.
void RequireGeometry(int64_t value, const char* key, int64_t max,
                     const std::string& what) {
  VT_CHECK(value >= 1 && value <= max,
           "clip mmproj gguf: " + std::string(key) + " is " +
               std::to_string(value) + ", and this reader accepts 1 to " +
               std::to_string(max) + " (" + what +
               "). A projector declaring that is corrupt, not new");
}

// One tensor's element count, refused on the PARSED VALUES before anything is
// reserved or resized from them. `elements` is saturated, so it is a lower
// bound on the real product rather than the product itself, which is why the
// message says "or more".
void RequireTensorElements(int64_t elements, const std::string& keys,
                           const std::string& what) {
  VT_CHECK(elements <= kMaxTensorElements,
           "clip mmproj gguf: the geometry from " + keys + " sizes " + what +
               " at " + std::to_string(elements) +
               " elements or more, and this reader accepts up to " +
               std::to_string(kMaxTensorElements) +
               " per tensor. A projector declaring that is corrupt, not new");
}

// A contiguous HOST view over `data`. W4 owns the upload, so this wave keeps
// every weight on the default device rather than inventing a device policy.
vt::Tensor HostView(void* data, vt::DType dtype,
                    const std::vector<int64_t>& shape) {
  vt::Tensor view;
  view.data = data;
  view.dtype = dtype;
  view.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = view.rank - 1; i >= 0; --i) {
    view.shape[i] = shape[static_cast<size_t>(i)];
    view.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return view;
}

// Reads one `deepseek4v` projector, keeping the host storage inside the result
// so every `vt::Tensor` the caller receives points at a buffer the result owns.
class DeepSeekV4MmprojReader {
 public:
  DeepSeekV4MmprojReader(const GgufFile& gguf,
                         const multimodal::DeepSeekV4VisionConfig& config,
                         DeepSeekV4ClipMmproj* out)
      : gguf_(gguf), config_(config), out_(out) {
    for (const GgufTensorInfo& info : gguf.Tensors()) present_.insert(info.name);
  }

  // A missing tensor names itself and a wrong-shaped one names both shapes.
  // `GgufTensorInfo::shape` is the on-disk ggml dims REVERSED into torch
  // row-major order, so every `want` below is written in torch order.
  const GgufTensorInfo& Require(const std::string& name,
                                const std::vector<int64_t>& want) {
    VT_CHECK(present_.count(name) != 0,
             "clip mmproj gguf: missing tensor " + name + " (is this a " +
                 kClipProjectorDeepSeekV4 + " projector?)");
    const GgufTensorInfo& info = gguf_.Get(name);
    VT_CHECK(info.shape == want,
             "clip mmproj gguf: " + name + " is " + ShapeText(info.shape) +
                 ", expected " + ShapeText(want));
    return info;
  }

  std::vector<float> F32(const std::string& name,
                         const std::vector<int64_t>& want) {
    const GgufTensorInfo& info = Require(name, want);
    return DequantGgufRowToF32(info.ggml_type, info.data, Numel(info));
  }

  std::vector<uint16_t> Bf16(const std::string& name,
                             const std::vector<int64_t>& want) {
    const GgufTensorInfo& info = Require(name, want);
    return DequantGgufRowToBf16(info.ggml_type, info.data, Numel(info));
  }

  // A linear weight or bias. W2's contract says both take the model dtype, and
  // this file stores every bias and the patch embedding as f32 because that is
  // llama.cpp's convention for a small tensor, not because the checkpoint holds
  // f32 there. Passing the file's dtype through instead would move twice the
  // bytes on the model path and leave every token identical, which is the one
  // defect a token gate cannot see.
  vt::Tensor Model(std::vector<uint16_t> words,
                   const std::vector<int64_t>& shape) {
    out_->bf16_storage.push_back(std::move(words));
    return HostView(out_->bf16_storage.back().data(), config_.compute_dtype,
                    shape);
  }

  // An RMSNorm weight. It stays f32: the pinned module declares it f32 and
  // widens x before the variance and the affine (deepseek_v4_vision.h), so
  // narrowing it here would change the tower's numbers.
  vt::Tensor Norm(const std::string& name) {
    const std::vector<int64_t> shape = {config_.hidden_size};
    out_->f32_storage.push_back(F32(name, shape));
    return HostView(out_->f32_storage.back().data(), vt::DType::kF32, shape);
  }

 private:
  const GgufFile& gguf_;
  const multimodal::DeepSeekV4VisionConfig& config_;
  DeepSeekV4ClipMmproj* out_;
  std::set<std::string> present_;
};

}  // namespace

void RefuseUnsupportedDeepSeekV4ClipMmproj(const GgufFile& gguf,
                                           const std::string& path) {
  const std::string arch = KvString(gguf, kKvArch);
  VT_CHECK(arch == kClipGgufArch,
           "--mmproj: '" + path + "' is not a multimodal projector: its "
           "general.architecture is '" +
               (arch.empty() ? std::string("<absent>") : arch) +
               "', and a projector file carries '" + kClipGgufArch +
               "'. Pass the language GGUF as the model and the mmproj-*.gguf "
               "here, not the other way round");
  const std::string type = KvString(gguf, kKvType);
  VT_CHECK(type.empty() || type == kClipGgufTypeMmproj,
           "--mmproj: '" + path + "' declares general.type '" + type +
               "', not '" + kClipGgufTypeMmproj + "'");
  const std::string proj = ClipProjectorType(gguf);
  VT_CHECK(proj == kClipProjectorDeepSeekV4,
           "--mmproj: '" + path + "' has clip.projector_type '" +
               (proj.empty() ? std::string("<absent>") : proj) +
               "'; the DeepSeek-V4 vision arm loads '" +
               kClipProjectorDeepSeekV4 + "' projectors only");
  // The FUSED attention arm, refused BY NAME. `gguf-py/gguf/constants.py` at
  // the pin spells V_ENC_ATTN_QKV `v.blk.{bid}.attn_qkv`, and nothing splits it
  // for this family, so `convert_hf_to_gguf.py` emits the fused form while the
  // shipped `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` mmproj carries the
  // split one. This refusal has to come BEFORE
  // `RefuseUnaccountedDeepSeekV4ClipMmproj`, which would otherwise report that
  // the FILE carries tensors this reader never reads and send the user to
  // re-convert an artifact that is already correct.
  VT_CHECK(!HasTensor(gguf, kTnFusedQkvProbe),
           "--mmproj: '" + path + "' stores its vision attention FUSED as '" +
               kTnFusedQkvProbe +
               "', and this build's deepseek4v reader reads the SPLIT "
               "'attn_q' / 'attn_k' / 'attn_v' form only. The fused arm is NOT "
               "IMPLEMENTED here: row "
               "MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm owns it and "
               "issue #2411 tracks it. Your file is not at fault -- "
               "llama.cpp's own convert_hf_to_gguf.py writes this layout. The "
               "shipped 'unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF' "
               "mmproj-BF16.gguf carries the split form and this build loads "
               "it");
  const GgufValue* silu = gguf.FindKv(kKvUseSilu);
  VT_CHECK(silu != nullptr && silu->TypeId() == kGgufBool &&
               std::get<bool>(silu->v),
           "--mmproj: '" + path +
               "' does not declare clip.use_silu = true. The DeepSeek-V4 "
               "vision MLP is SwiGLU by construction, the pinned converter "
               "writes that key for exactly that reason, and a projector with "
               "another activation loaded through this reader would run and be "
               "wrong");
}

multimodal::DeepSeekV4VisionConfig DeepSeekV4ClipMmprojVisionConfig(
    const GgufFile& gguf) {
  multimodal::DeepSeekV4VisionConfig config;
  config.hidden_size = ReqInt(gguf, kKvEmbd);
  config.num_heads = ReqInt(gguf, kKvHeads);
  config.depth = ReqInt(gguf, kKvBlocks);
  config.intermediate_size = ReqInt(gguf, kKvFf);
  // Every one of these is a `Require` shape, a loop bound or a `resize`
  // argument further down, so each is bounded HERE, where the key that carried
  // it can still be named. `depth` carries the tight bound because it is the
  // only one that reaches `std::vector::resize` directly; the rest are refused
  // for being absent-in-effect, which would otherwise build a tower of empty
  // matrices that runs and is wrong.
  RequireGeometry(config.depth, kKvBlocks, kMaxDeepSeekV4Depth,
                  "it becomes the block vector's size");
  RequireGeometry(config.hidden_size, kKvEmbd, kMaxGeometry,
                  "it is every block tensor's shape");
  RequireGeometry(config.num_heads, kKvHeads, kMaxGeometry,
                  "it divides the hidden size into heads");
  RequireGeometry(config.intermediate_size, kKvFf, kMaxGeometry,
                  "it is the MLP's inner width");
  // `projection_dim` is the aligner's output width and `projector.scale_factor`
  // is the 3x3 downsample ratio: clip.cpp's PROJECTOR_TYPE_DEEPSEEK4V case
  // reads KEY_PROJ_SCALE_FACTOR into `hparams.n_merge`, and deepseek4v.cpp
  // unfolds the patch grid with it.
  config.output_size = ReqInt(gguf, kKvProjDim);
  config.downsample_ratio = ReqInt(gguf, kKvScaleFactor);
  config.patch_size = ReqInt(gguf, kKvPatch);
  RequireGeometry(config.output_size, kKvProjDim, kMaxGeometry,
                  "it is the aligner's output width");
  RequireGeometry(config.downsample_ratio, kKvScaleFactor, kMaxGeometry,
                  "it squares into the aligner's input width");
  RequireGeometry(config.patch_size, kKvPatch, kMaxGeometry,
                  "it squares into the patch embedding's input width");
  // EVERY PRODUCT THE LOADER FORMS, bounded here rather than at the allocation
  // it becomes. Each of these is the element count of one tensor the loader
  // materializes; the first is the one that reached `reserve` with nothing
  // file-shaped in front of it. `patch_dim()` is 3 * patch^2 and
  // `aligner_input_size()` is hidden * ratio^2, both spelled out so the factors
  // this refusal names are the keys that carried them.
  constexpr int64_t kRgbChannels = 3;
  RequireTensorElements(
      SaturatingElements({3, config.hidden_size, config.hidden_size}), kKvEmbd,
      "the fused qkv weight");
  RequireTensorElements(
      SaturatingElements({config.hidden_size, kRgbChannels, config.patch_size,
                          config.patch_size}),
      std::string(kKvEmbd) + " and " + kKvPatch, "the patch embedding weight");
  RequireTensorElements(
      SaturatingElements({2, config.intermediate_size, config.hidden_size}),
      std::string(kKvFf) + " and " + kKvEmbd, "the merged gate/up weight");
  RequireTensorElements(
      SaturatingElements({config.output_size, config.hidden_size,
                          config.downsample_ratio, config.downsample_ratio}),
      std::string(kKvProjDim) + ", " + kKvEmbd + " and " + kKvScaleFactor,
      "the aligner's first projection");
  RequireTensorElements(
      SaturatingElements({config.output_size, config.output_size}), kKvProjDim,
      "the aligner's second projection");
  // READ, never assumed: this projector's eps is the vision RMSNorm's torch
  // default rather than the language model's, and a reader that kept the W2
  // default would agree with this artifact by luck.
  config.norm_epsilon = static_cast<float>(ReqFloat(gguf, kKvEps));
  // `rope_theta` and `compute_dtype` keep their W2 defaults. No `clip.*` key
  // carries the theta: llama.cpp hardcodes 10000.0 for this projector in the
  // same `clip_model_loader` case that reads the keys above, and the pinned
  // converter's `get_vision_config` defaults `vision_rope_theta` to the same
  // value without writing it.
  return config;
}

DeepSeekV4ClipMmproj LoadDeepSeekV4VisionFromClipMmproj(
    const GgufFile& gguf, const multimodal::DeepSeekV4VisionConfig& config) {
  VT_CHECK(config.compute_dtype == vt::DType::kBF16,
           "clip mmproj gguf: the deepseek4v reader stores bf16 words, and the "
           "DeepSeek-V4 vision tower refuses any other compute dtype");
  DeepSeekV4ClipMmproj out;
  DeepSeekV4MmprojReader read(gguf, config, &out);

  const int64_t hidden = config.hidden_size;
  const int64_t intermediate = config.intermediate_size;
  const int64_t output = config.output_size;
  const int64_t patch = config.patch_size;
  const int64_t patch_dim = config.patch_dim();
  const int64_t aligner_in = config.aligner_input_size();
  // `patch_dim()` is `3 * patch^2`, so the channel count is W2's own contract
  // rather than a number read here; the shape check below refuses a file that
  // disagrees instead of reshaping around it.
  constexpr int64_t kChannels = 3;

  // ── The patch embedding: a conv2d weight flattened back to torch Linear ──
  //
  // The pinned `vision.patch_embed.proj` is an `nn.Linear` over patches
  // flattened by `F.unfold`, whose element order is [channel, dy, dx], so its
  // weight is torch [hidden, C * p * p] in exactly that column order. The
  // pinned converter turns it into a conv2d weight with a pure VIEW --
  // `data_torch.reshape(data_torch.shape[0], 3, p, p)`, which moves no byte --
  // and llama.cpp stores that in ggml dim order {p, p, C, out}.
  // `GgufTensorInfo::shape` reverses it back to torch [out, C, p, p], whose
  // row-major flattening is [channel, dy, dx] again. The map is therefore the
  // IDENTITY, and it is written as an explicit index walk rather than a bulk
  // copy because the [channel, dy, dx] claim is the load-bearing part: the
  // [dy, dx, channel] order a naive conv2d reading produces is a fluent, wrong
  // tower rather than an error.
  const std::vector<uint16_t> patch_source =
      read.Bf16(kTnPatchEmbd, {hidden, kChannels, patch, patch});
  std::vector<uint16_t> patch_weight(static_cast<size_t>(hidden * patch_dim));
  for (int64_t o = 0; o < hidden; ++o) {
    for (int64_t c = 0; c < kChannels; ++c) {
      for (int64_t dy = 0; dy < patch; ++dy) {
        for (int64_t dx = 0; dx < patch; ++dx) {
          const int64_t source = ((o * kChannels + c) * patch + dy) * patch + dx;
          const int64_t target = o * patch_dim + (c * patch + dy) * patch + dx;
          patch_weight[static_cast<size_t>(target)] =
              patch_source[static_cast<size_t>(source)];
        }
      }
    }
  }
  out.weights.patch_weight =
      read.Model(std::move(patch_weight), {hidden, patch_dim});
  out.weights.patch_bias =
      read.Model(read.Bf16(kTnPatchBias, {hidden}), {hidden});

  // ── The blocks ────────────────────────────────────────────────────────────
  out.weights.blocks.resize(static_cast<size_t>(config.depth));
  for (int64_t layer = 0; layer < config.depth; ++layer) {
    const std::string p = DeepSeekV4BlockPrefix(layer);
    multimodal::DeepSeekV4VisionBlockWeights& block =
        out.weights.blocks[static_cast<size_t>(layer)];
    block.norm1_weight = read.Norm(p + "ln1.weight");
    block.norm2_weight = read.Norm(p + "ln2.weight");

    // q, k, v FUSE in that row order, and the order is the consumer's rather
    // than a convention chosen here: `deepseek_v4_vision.cpp` takes Q back out
    // with `RowSlice(layer.qkv_weight, 0, hidden)`, K at `hidden`, V at
    // `2 * hidden`, and the matching `VectorSlice`s for the bias. Permuting the
    // three swaps which projection each head attends with and stays fluent.
    // NOT RESERVED FROM THE DECLARED GEOMETRY. A `reserve` here ran ahead of
    // every file-shaped read, so it sized an allocation from a number no file
    // had yet had to justify; the product bound above refuses the absurd case
    // by name, and growing on `insert` keeps this site bounded by the bytes
    // `read.Bf16` actually returns even if that bound is ever widened.
    std::vector<uint16_t> qkv_weight;
    std::vector<uint16_t> qkv_bias;
    for (const char* part : {"attn_q", "attn_k", "attn_v"}) {
      const std::vector<uint16_t> weight =
          read.Bf16(p + part + ".weight", {hidden, hidden});
      qkv_weight.insert(qkv_weight.end(), weight.begin(), weight.end());
      const std::vector<uint16_t> bias =
          read.Bf16(p + part + ".bias", {hidden});
      qkv_bias.insert(qkv_bias.end(), bias.begin(), bias.end());
    }
    block.qkv_weight = read.Model(std::move(qkv_weight), {3 * hidden, hidden});
    block.qkv_bias = read.Model(std::move(qkv_bias), {3 * hidden});
    block.out_weight = read.Model(
        read.Bf16(p + "attn_out.weight", {hidden, hidden}), {hidden, hidden});
    block.out_bias =
        read.Model(read.Bf16(p + "attn_out.bias", {hidden}), {hidden});

    // GATE FIRST, UP SECOND. W2 hands `mlp_w1_weight` to
    // `layers::UnquantizedMlpGateUpMethod`, which runs one `MatmulBT` over the
    // merged [2I, H] weight and then `vt::SiluAndMul`; that kernel reads the
    // gate at column `j` and the up at column `d + j`, so the FIRST
    // `intermediate` rows are the gate. It is also where the pinned converter
    // took them from -- it split the checkpoint's fused `mlp.w1` with
    // `gate, up = data_torch.chunk(2, dim=0)` -- so this puts each half back
    // where it came from. Swapping them applies SiLU to the wrong projection
    // and stays fluent.
    std::vector<uint16_t> gate_up =
        read.Bf16(p + "ffn_gate.weight", {intermediate, hidden});
    const std::vector<uint16_t> up =
        read.Bf16(p + "ffn_up.weight", {intermediate, hidden});
    gate_up.insert(gate_up.end(), up.begin(), up.end());
    block.mlp_w1_weight =
        read.Model(std::move(gate_up), {2 * intermediate, hidden});
    block.mlp_w2_weight = read.Model(
        read.Bf16(p + "ffn_down.weight", {hidden, intermediate}),
        {hidden, intermediate});
  }

  // ── The final norm and the aligner ────────────────────────────────────────
  // `v.post_ln` is the tower's final RMSNorm, applied before the 3x3 unfold
  // (deepseek4v.cpp runs `build_vit` and only then reshapes and unfolds), and
  // `mm.1` is that unfold's consumer: its input width is hidden * ratio^2.
  out.weights.final_norm_weight = read.Norm(kTnPostLn);
  out.weights.aligner_w1_weight = read.Model(
      read.Bf16(kTnMm1Weight, {output, aligner_in}), {output, aligner_in});
  out.weights.aligner_w1_bias =
      read.Model(read.Bf16(kTnMm1Bias, {output}), {output});
  out.weights.aligner_w2_weight = read.Model(
      read.Bf16(kTnMm2Weight, {output, output}), {output, output});
  out.weights.aligner_w2_bias =
      read.Model(read.Bf16(kTnMm2Bias, {output}), {output});

  // The four learned sentinel vectors. They stay f32 for the reason the header
  // states: W2 declares no dtype for them because it has no field for them.
  out.image_start = read.F32(kTnImgStart, {output});
  out.image_end = read.F32(kTnImgEnd, {output});
  out.image_pad = read.F32(kTnImgPad, {output});
  out.image_newline = read.F32(kTnImageNewline, {output});
  return out;
}

std::vector<std::string> DeepSeekV4ClipMmprojExpectedTensors(
    const multimodal::DeepSeekV4VisionConfig& config) {
  std::vector<std::string> out;
  out.emplace_back(kTnPatchEmbd);
  out.emplace_back(kTnPatchBias);
  for (int64_t layer = 0; layer < config.depth; ++layer) {
    const std::string p = DeepSeekV4BlockPrefix(layer);
    for (const char* stem : kDeepSeekV4BlockTensors) out.push_back(p + stem);
  }
  out.emplace_back(kTnPostLn);
  for (const char* name :
       {kTnMm1Weight, kTnMm1Bias, kTnMm2Weight, kTnMm2Bias}) {
    out.emplace_back(name);
  }
  for (const char* name :
       {kTnImgStart, kTnImgEnd, kTnImgPad, kTnImageNewline}) {
    out.emplace_back(name);
  }
  return out;
}

void RefuseUnaccountedDeepSeekV4ClipMmproj(
    const GgufFile& gguf, const multimodal::DeepSeekV4VisionConfig& config,
    const std::string& path) {
  const std::vector<std::string> want =
      DeepSeekV4ClipMmprojExpectedTensors(config);
  const std::set<std::string> wanted(want.begin(), want.end());
  std::vector<std::string> extra;
  for (const GgufTensorInfo& info : gguf.Tensors()) {
    if (wanted.count(info.name) == 0) extra.push_back(info.name);
  }
  if (extra.empty()) return;
  constexpr size_t kMaxNamed = 12;
  std::string names;
  for (size_t i = 0; i < extra.size() && i < kMaxNamed; ++i) {
    names += (i == 0 ? "" : ", ") + extra[i];
  }
  if (extra.size() > kMaxNamed) {
    names += ", ... (" + std::to_string(extra.size() - kMaxNamed) + " more)";
  }
  VT_CHECK(false,
           "--mmproj: '" + path + "' carries " + std::to_string(extra.size()) +
               " tensor(s) that this build's " + kClipProjectorDeepSeekV4 +
               " reader NEVER reads, out of " +
               std::to_string(gguf.Tensors().size()) + " present against " +
               std::to_string(wanted.size()) + " enumerated for depth " +
               std::to_string(config.depth) + ": " + names +
               ". Loading it would drop them SILENTLY and build a tower that "
               "runs and is wrong");
}

}  // namespace vllm
