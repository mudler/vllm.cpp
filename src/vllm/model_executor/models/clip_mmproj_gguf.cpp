// The llama.cpp `clip` mmproj reader. Contract, upstream anchors and the reason
// each refusal exists: `include/vllm/model_executor/models/clip_mmproj_gguf.h`.
#include "vllm/model_executor/models/clip_mmproj_gguf.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
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
  auto load = [&](const std::string& name) -> std::vector<float> {
    VT_CHECK(has(name), "clip mmproj gguf: missing tensor " + name +
                            " (is this a " + kClipProjectorQwen3VL +
                            " projector?)");
    const GgufTensorInfo& info = gguf.Get(name);
    return DequantGgufRowToF32(info.ggml_type, info.data, Numel(info));
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
  const std::vector<float> w0 = load(kTnPatchEmbd);
  const std::vector<float> w1 = load(kTnPatchEmbd1);
  const int64_t plane = cfg.patch_size * cfg.patch_size;
  const int64_t tp = cfg.temporal_patch_size;
  vw.patch_proj_w.assign(static_cast<size_t>(out * spatial * tp), 0.0F);
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
  vw.pos_embed_w = load(kTnPosEmbd);

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

}  // namespace vllm
