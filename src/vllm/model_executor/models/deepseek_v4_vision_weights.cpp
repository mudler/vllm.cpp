// DeepSeek-V4-Flash-Vision — the OFFICIAL safetensors vision arm (row
// `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue
// [#2411](https://github.com/mudler/vllm.cpp/issues/2411)).
//
// A PORT, and this comment is the provenance. The file is ported from a
// parallel implementation of this row preserved at
// `row/MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm-CODEX-LINE` (`3f3860851`,
// "load the released vision tower"), whose own header named the same pinned
// source: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
// `86f746b36186f0e567729a5c06a8c918caba82a9`, shard 1 safetensors header. That
// line forked before this row's spec amendment and built against DIFFERENT
// types, so this is a re-expression onto the canonical ones rather than a copy.
// The differences are listed at the bottom of this comment, because each of
// them is a decision somebody may want to revisit with a reason in hand.
//
// WHAT GAP IT CLOSES. Until this file, `LoadDeepseekV4ForCausalLM`'s
// safetensors branch said so itself: "THE SAFETENSORS ARM STAYS TOWER-FREE ...
// The official arm carries `vision.*` and `aligner.*` in its own shards;
// ACCOUNTING for them landed with W3, MATERIALISING them is owed". Only the
// GGUF vehicle could carry a tower, through `--mmproj`. The 267 vision tensors
// of the released checkpoint sat in shard 1 and NOTHING read them: the dense
// name-map pass has no leftover refusal, so they were not even counted, and an
// image request on this arm refused in `encode_mm`. This is the second arm
// beside the `deepseek4v` mmproj reader, not a replacement for it.
//
// WHERE THE NAMES AND SHAPES COME FROM. The pinned `config.json` and the shard-1
// safetensors header, both committed under
// `tests/parity/goldens/deepseek_v4_vision/` and checked against the live
// artifact by `scripts/check-deepseek-v4-vision-manifests.py --refresh`. The
// released group is 267 tensors and every one of them is BF16 on disk:
// the patch embedding and its bias, `depth` blocks of eight, the final norm,
// the aligner's two weight/bias pairs, and the four learned sentinel vectors.
//
// ─── HOW THIS DIFFERS FROM THE LINE IT WAS PORTED FROM, AND WHY ──────────────
//
//   1. IT COPIES INSTEAD OF BORROWING, and that is a correctness repair rather
//      than only a house-style choice. The ported line handed `vt::Tensor::data`
//      a pointer straight INTO the safetensors mmap and kept the mapping alive
//      through a `backing_owner` handle it added to the shared W2 weight struct.
//      A safetensors payload offset carries NO alignment guarantee -- this
//      tree's own fixture writer deliberately forces an ODD payload base
//      (`dsv4_exl3_fixture.h`, `kMisalignedPayloadBase`) for exactly that
//      reason -- so a borrowed bf16 weight can begin at an address no bf16
//      access is allowed to assume. Copying through `std::memcpy` lands every
//      tensor in an owned, aligned buffer, and it is what the `deepseek4v`
//      mmproj reader beside this file already does, what the DeepSeek-V4
//      language loader does for every tensor it reads, and what lets
//      `MaybeReleaseSourcePages` drop the source pages behind it.
//      The cost is 0.870 GiB of host copy on the released artifact.
//
//   2. IT ADDS NO FIELD TO THE SHARED W2 TYPES. The ported line grew
//      `DeepSeekV4VisionWeights` by a `backing_owner` handle and four
//      `image_*` tensors. Neither is needed here: ownership lives in the result
//      struct (below), and the four sentinels already have canonical homes as
//      f32 vectors beside the weights, which is the dtype
//      `deepseek_v4_mm.cpp`'s merge reads them at. Keeping the W2 struct fixed
//      also keeps the storage-layout accessor the canonical line carries
//      (`mlp_gate_up_markers`), which that line had deleted.
//
//   3. IT RESOLVES GEOMETRY FROM `config.json`, NOT FROM A WIDENED
//      `DeepseekV4Params`. The ported line added eleven `vision_*` fields to the
//      language model's parameter struct. The canonical tree resolves a vision
//      geometry per VEHICLE instead -- `DeepSeekV4ClipMmprojVisionConfig` reads
//      the mmproj's `clip.*` keys -- so this is that function's safetensors
//      sibling, and the language parameters stay about the language model.
//
//   4. THERE IS NO COMBINED-GGUF ENTRY POINT. The ported line also carried a
//      `LoadDeepseekV4VisionFromGguf` reading `vision.*` out of ONE combined
//      `deepseek4` GGUF. This row's spec pins the two-file llama.cpp vehicle
//      (`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`: language shards plus
//      `mmproj-BF16.gguf`), which `clip_mmproj_gguf.cpp` already reads, so that
//      entry point would describe an artifact nothing ships and no oracle runs.
//
// WHAT IS NOT PROVEN HERE. THE REAL SAFETENSORS PAYLOAD HAS NEVER BEEN READ.
// The released checkpoint is 156.287 GiB across 48 shards and is not staged on
// any gate device, so every case is a synthetic fixture built to the MEASURED
// shard-1 header plus the committed manifests. See `## Owed` in
// `.agents/specs/deepseek-v4-flash-vision.md`.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/clip_mmproj_gguf.h"
#include "vllm/model_executor/models/deepseek_v4_mm.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"  // VT_CHECK
#include "vt/tensor.h"

namespace vllm {
namespace {

// The released group's on-disk dtype. Every one of the 267 tensors is BF16 in
// the pinned shard-1 header, the tower's compute dtype is bf16, and the spec's
// weight contract keeps the vision arm BF16 rather than quantizing it. A file
// that stores another dtype here is REFUSED BY NAME rather than widened,
// because a silently widened model path moves twice the bytes while every token
// stays identical (AGENTS.md, "Inherit vLLM defaults").
constexpr const char* kOfficialVisionDType = "BF16";

constexpr const char* kPatchWeight = "vision.patch_embed.proj.weight";
constexpr const char* kPatchBias = "vision.patch_embed.proj.bias";
constexpr const char* kFinalNorm = "vision.norm.weight";
constexpr const char* kAlignerW1Weight = "aligner.w1.weight";
constexpr const char* kAlignerW1Bias = "aligner.w1.bias";
constexpr const char* kAlignerW2Weight = "aligner.w2.weight";
constexpr const char* kAlignerW2Bias = "aligner.w2.bias";
constexpr const char* kImageStart = "image_start";
constexpr const char* kImageEnd = "image_end";
constexpr const char* kImageNewline = "image_newline";
constexpr const char* kImagePad = "image_pad";

// The eight per-block tensors, in the order the released checkpoint spells
// them. `attn.wqkv` is ALREADY FUSED on this vehicle, which is the one place
// the official arm is simpler than the GGUF one: the mmproj reader has to fuse
// `attn_q` / `attn_k` / `attn_v` itself because llama.cpp's converter splits
// them, while here the checkpoint stores the [3*hidden, hidden] tensor the
// tower slices Q, K and V back out of.
const char* const kBlockTensors[] = {
    "norm1.weight", "attn.wqkv.weight", "attn.wqkv.bias", "attn.wo.weight",
    "attn.wo.bias", "norm2.weight",     "mlp.w1.weight",  "mlp.w2.weight",
};

std::string BlockPrefix(int64_t layer) {
  return "vision.blocks." + std::to_string(layer) + ".";
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += ",";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || !it->is_number()) return fallback;
  return it->get<int64_t>();
}

double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || !it->is_number()) return fallback;
  return it->get<double>();
}

// Refuse an absent-in-effect or absurd config value BY THE KEY THAT CARRIED IT.
// Every one of these becomes a `Require` shape, a loop bound or a `resize`
// argument below, so each is bounded here while the key is still nameable —
// the same polarity, and the same reason, as `RequireGeometry` in the
// `deepseek4v` mmproj reader beside this file.
void RequireGeometry(int64_t value, const char* key, int64_t limit,
                     const char* because) {
  VT_CHECK(value > 0,
           std::string("deepseek-v4 vision config: ") + key + " is " +
               std::to_string(value) +
               ", and it must be positive because " + because);
  VT_CHECK(value <= limit,
           std::string("deepseek-v4 vision config: ") + key + " is " +
               std::to_string(value) + ", above the " + std::to_string(limit) +
               " this reader accepts because " + because);
}

// A contiguous HOST view over `data`. W4 owns the upload; this arm keeps every
// weight on the default device rather than inventing a device policy, exactly
// as the mmproj reader does.
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

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// Reads the official vision group out of the checkpoint shards, keeping the
// host storage inside the result so every `vt::Tensor` the caller receives
// points at a buffer the result owns and outlives the shards.
class OfficialVisionReader {
 public:
  OfficialVisionReader(const std::vector<SafetensorsFile>& shards,
                       const multimodal::DeepSeekV4VisionConfig& config,
                       DeepSeekV4ClipMmproj* out)
      : config_(config), out_(out) {
    for (const SafetensorsFile& shard : shards) {
      for (const std::string& name : shard.Names()) {
        // A name present twice across shards is REFUSED rather than resolved by
        // shard order: the two copies can differ, and picking one silently
        // builds a tower that runs and is wrong.
        const auto inserted = tensors_.emplace(name, &shard.Get(name));
        VT_CHECK(inserted.second,
                 "deepseek-v4 vision loader: duplicate checkpoint tensor " +
                     name + " across shards");
      }
    }
  }

  // A missing tensor names itself, a wrong-shaped one names both shapes, and a
  // wrong-dtype one names both dtypes.
  const StTensor& Require(const std::string& name,
                          const std::vector<int64_t>& want) {
    const auto it = tensors_.find(name);
    VT_CHECK(it != tensors_.end(),
             "deepseek-v4 vision loader: expected checkpoint tensor missing: " +
                 name +
                 " (the official DeepSeek-V4-Flash-Vision group is 267 tensors "
                 "and this checkpoint carries some of them)");
    const StTensor& tensor = *it->second;
    VT_CHECK(tensor.dtype == kOfficialVisionDType,
             "deepseek-v4 vision loader: " + name + " has dtype " +
                 tensor.dtype + ", expected " + kOfficialVisionDType +
                 ". The released vision group is BF16 on disk and this reader "
                 "refuses another storage variant rather than widening it "
                 "silently");
    VT_CHECK(tensor.shape == want,
             "deepseek-v4 vision loader: " + name + " is " +
                 ShapeText(tensor.shape) + ", expected " + ShapeText(want));
    const int64_t count = Numel(want);
    VT_CHECK(tensor.nbytes == static_cast<size_t>(count) * 2,
             "deepseek-v4 vision loader: " + name + " holds " +
                 std::to_string(tensor.nbytes) + " bytes for " +
                 std::to_string(count) +
                 " BF16 elements; it is not one contiguous tensor");
    VT_CHECK(tensor.data != nullptr,
             "deepseek-v4 vision loader: " + name + " has no backing storage");
    consumed_.insert(name);
    return tensor;
  }

  // The bf16 words, COPIED OUT. `std::memcpy` rather than a cast through
  // `const uint16_t*`: a safetensors data offset carries no alignment, so the
  // source may begin at an odd address (see this file's header).
  std::vector<uint16_t> Bf16(const std::string& name,
                             const std::vector<int64_t>& want) {
    const StTensor& tensor = Require(name, want);
    std::vector<uint16_t> words(static_cast<size_t>(Numel(want)));
    std::memcpy(words.data(), tensor.data, tensor.nbytes);
    MaybeReleaseSourcePages(tensor.data, tensor.nbytes);
    return words;
  }

  // A linear weight, bias or learned vector at the MODEL dtype.
  vt::Tensor Model(std::vector<uint16_t> words,
                   const std::vector<int64_t>& shape) {
    out_->bf16_storage.push_back(std::move(words));
    return HostView(out_->bf16_storage.back().data(), config_.compute_dtype,
                    shape);
  }

  // An RMSNorm weight, WIDENED ONCE to f32. The checkpoint stores it bf16, and
  // the pinned module declares it f32 and widens x before the variance and the
  // affine (`deepseek_v4_vision.h`), so the widening happens here rather than
  // per token in the tower. This is the ONE dtype this arm changes on the way
  // in, and `ValidateWeights` requires f32 for exactly these three names.
  vt::Tensor Norm(const std::string& name) {
    const std::vector<int64_t> shape = {config_.hidden_size};
    const std::vector<uint16_t> words = Bf16(name, shape);
    std::vector<float> widened(words.size());
    for (size_t i = 0; i < words.size(); ++i)
      widened[i] = vt::BF16ToF32(words[i]);
    out_->f32_storage.push_back(std::move(widened));
    return HostView(out_->f32_storage.back().data(), vt::DType::kF32, shape);
  }

  // A learned sentinel vector, widened to f32 for the reason the canonical
  // `DeepSeekV4ClipMmproj` header gives: the merge in `deepseek_v4_mm.cpp`
  // reads these four as `std::vector<float>`, and the mmproj vehicle stores
  // them f32 too, so the two arms hand the merge the same thing.
  std::vector<float> Sentinel(const std::string& name) {
    const std::vector<int64_t> shape = {config_.output_size};
    const std::vector<uint16_t> words = Bf16(name, shape);
    std::vector<float> widened(words.size());
    for (size_t i = 0; i < words.size(); ++i)
      widened[i] = vt::BF16ToF32(words[i]);
    return widened;
  }

  size_t consumed() const { return consumed_.size(); }

 private:
  const multimodal::DeepSeekV4VisionConfig& config_;
  DeepSeekV4ClipMmproj* out_;
  std::unordered_map<std::string, const StTensor*> tensors_;
  std::set<std::string> consumed_;
};

}  // namespace

multimodal::DeepSeekV4VisionConfig DeepSeekV4OfficialVisionConfig(
    const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  multimodal::DeepSeekV4VisionConfig out;
  out.hidden_size = RawInt(raw, "vision_dim", 0);
  out.num_heads = RawInt(raw, "vision_n_heads", 0);
  out.depth = RawInt(raw, "vision_n_layers", 0);
  out.intermediate_size = RawInt(raw, "vision_inter_dim", 0);
  out.patch_size = RawInt(raw, "vision_patch_size", 0);
  out.downsample_ratio = RawInt(raw, "vision_downsample_ratio", 0);
  // THE ALIGNER LANDS IN THE TEXT HIDDEN SPACE, so the output width is the
  // language model's own rather than a vision key. The mmproj arm reads
  // `clip.projection_dim` and `LoadDeepseekV4VisionRuntime` then refuses a
  // projector whose width is not the language model's; on this vehicle the two
  // come from ONE config.json and cannot disagree.
  out.output_size = config.hidden_size > 0 ? config.hidden_size
                                           : RawInt(raw, "hidden_size", 0);
  out.rope_theta = RawDouble(raw, "vision_rope_theta", 10000.0);
  // The released config carries no vision normalization epsilon. 1e-6 is the
  // pinned module's own RMSNorm default and the W2 tower's default, so it is
  // inherited rather than invented here.
  out.norm_epsilon = 1.0e-6f;
  out.compute_dtype = vt::DType::kBF16;

  // Bounded here, where the config key that carried each value can still be
  // named. `kMaxDepth` is the tight one because `depth` is what reaches
  // `std::vector::resize` directly.
  constexpr int64_t kMaxDepth = 512;
  constexpr int64_t kMaxGeometry = 1 << 20;
  RequireGeometry(out.depth, "vision_n_layers", kMaxDepth,
                  "it becomes the block vector's size");
  RequireGeometry(out.hidden_size, "vision_dim", kMaxGeometry,
                  "it is every block tensor's shape");
  RequireGeometry(out.num_heads, "vision_n_heads", kMaxGeometry,
                  "it divides the hidden size into heads");
  RequireGeometry(out.intermediate_size, "vision_inter_dim", kMaxGeometry,
                  "it is the MLP's inner width");
  RequireGeometry(out.patch_size, "vision_patch_size", kMaxGeometry,
                  "it is the patch embedding's input width");
  RequireGeometry(out.downsample_ratio, "vision_downsample_ratio", kMaxGeometry,
                  "it squares into the aligner's input width");
  RequireGeometry(out.output_size, "hidden_size", kMaxGeometry,
                  "it is the width the aligner projects image rows into");
  VT_CHECK(out.hidden_size % out.num_heads == 0,
           "deepseek-v4 vision config: vision_n_heads (" +
               std::to_string(out.num_heads) + ") must divide vision_dim (" +
               std::to_string(out.hidden_size) + ")");
  VT_CHECK((out.hidden_size / out.num_heads) % 4 == 0,
           "deepseek-v4 vision config: the vision head dimension (" +
               std::to_string(out.hidden_size / out.num_heads) +
               ") must be divisible by four, because the 2-D RoPE pairs it into "
               "a height half and a width half");
  VT_CHECK(out.rope_theta > 0.0 && std::isfinite(out.rope_theta),
           "deepseek-v4 vision config: vision_rope_theta must be finite and "
           "positive");
  return out;
}

std::vector<std::string> DeepSeekV4OfficialVisionExpectedTensors(
    const multimodal::DeepSeekV4VisionConfig& config) {
  std::vector<std::string> out;
  out.emplace_back(kPatchWeight);
  out.emplace_back(kPatchBias);
  for (int64_t layer = 0; layer < config.depth; ++layer) {
    const std::string prefix = BlockPrefix(layer);
    for (const char* stem : kBlockTensors) out.push_back(prefix + stem);
  }
  out.emplace_back(kFinalNorm);
  for (const char* name :
       {kAlignerW1Weight, kAlignerW1Bias, kAlignerW2Weight, kAlignerW2Bias}) {
    out.emplace_back(name);
  }
  for (const char* name :
       {kImageStart, kImageEnd, kImageNewline, kImagePad}) {
    out.emplace_back(name);
  }
  return out;
}

bool DeepSeekV4ShardsCarryVision(
    const std::vector<SafetensorsFile>& shards) {
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      if (name == kPatchWeight) return true;
    }
  }
  return false;
}

DeepSeekV4ClipMmproj LoadDeepSeekV4VisionFromSafetensors(
    const std::vector<SafetensorsFile>& shards,
    const multimodal::DeepSeekV4VisionConfig& config) {
  VT_CHECK(config.compute_dtype == vt::DType::kBF16,
           "deepseek-v4 vision loader: the official arm stores bf16 words, and "
           "the DeepSeek-V4 vision tower refuses any other compute dtype");
  DeepSeekV4ClipMmproj out;
  OfficialVisionReader read(shards, config, &out);

  const int64_t hidden = config.hidden_size;
  const int64_t intermediate = config.intermediate_size;
  const int64_t output = config.output_size;
  const int64_t patch_dim = config.patch_dim();
  const int64_t aligner_in = config.aligner_input_size();

  // ── The patch embedding ───────────────────────────────────────────────────
  // NO RESHAPE HERE, and that is the difference from the GGUF vehicle. The
  // pinned `vision.patch_embed.proj` is an `nn.Linear` over patches flattened
  // by `F.unfold`, so the checkpoint already stores the 2-D torch weight
  // [hidden, 3 * patch^2] in [channel, dy, dx] column order — the order the
  // tower reads. llama.cpp's converter is what turns it into a 4-D conv2d
  // weight, which is why `clip_mmproj_gguf.cpp` has an index walk to undo and
  // this arm does not.
  out.weights.patch_weight =
      read.Model(read.Bf16(kPatchWeight, {hidden, patch_dim}),
                 {hidden, patch_dim});
  out.weights.patch_bias =
      read.Model(read.Bf16(kPatchBias, {hidden}), {hidden});

  // ── The blocks ────────────────────────────────────────────────────────────
  out.weights.blocks.resize(static_cast<size_t>(config.depth));
  for (int64_t layer = 0; layer < config.depth; ++layer) {
    const std::string p = BlockPrefix(layer);
    multimodal::DeepSeekV4VisionBlockWeights& block =
        out.weights.blocks[static_cast<size_t>(layer)];
    block.norm1_weight = read.Norm(p + "norm1.weight");
    // FUSED ON DISK, in q, k, v row order — the order
    // `deepseek_v4_vision.cpp` takes Q back out of with
    // `RowSlice(layer.qkv_weight, 0, hidden)`, K at `hidden` and V at
    // `2 * hidden`. The shape check is what holds the checkpoint to it.
    block.qkv_weight =
        read.Model(read.Bf16(p + "attn.wqkv.weight", {3 * hidden, hidden}),
                   {3 * hidden, hidden});
    block.qkv_bias = read.Model(read.Bf16(p + "attn.wqkv.bias", {3 * hidden}),
                                {3 * hidden});
    block.out_weight = read.Model(
        read.Bf16(p + "attn.wo.weight", {hidden, hidden}), {hidden, hidden});
    block.out_bias =
        read.Model(read.Bf16(p + "attn.wo.bias", {hidden}), {hidden});
    block.norm2_weight = read.Norm(p + "norm2.weight");
    // ALSO FUSED ON DISK, GATE FIRST. The checkpoint stores one
    // [2*intermediate, hidden] `mlp.w1`; llama.cpp's converter is what splits it
    // with `gate, up = data_torch.chunk(2, dim=0)`, so the first `intermediate`
    // rows are the gate on both vehicles. W2 hands this to
    // `layers::UnquantizedMlpGateUpMethod`, whose `vt::SiluAndMul` reads the
    // gate at column `j` and the up at column `d + j`. A file that stored them
    // the other way round would apply SiLU to the wrong projection and stay
    // fluent, which is why the merged width is shape-checked rather than
    // inferred.
    block.mlp_w1_weight =
        read.Model(read.Bf16(p + "mlp.w1.weight", {2 * intermediate, hidden}),
                   {2 * intermediate, hidden});
    block.mlp_w2_weight =
        read.Model(read.Bf16(p + "mlp.w2.weight", {hidden, intermediate}),
                   {hidden, intermediate});
  }

  // ── The final norm and the aligner ────────────────────────────────────────
  // The final RMSNorm is applied before the 3x3 unfold, and `aligner.w1` is
  // that unfold's consumer: its input width is hidden * ratio^2.
  out.weights.final_norm_weight = read.Norm(kFinalNorm);
  out.weights.aligner_w1_weight = read.Model(
      read.Bf16(kAlignerW1Weight, {output, aligner_in}), {output, aligner_in});
  out.weights.aligner_w1_bias =
      read.Model(read.Bf16(kAlignerW1Bias, {output}), {output});
  out.weights.aligner_w2_weight = read.Model(
      read.Bf16(kAlignerW2Weight, {output, output}), {output, output});
  out.weights.aligner_w2_bias =
      read.Model(read.Bf16(kAlignerW2Bias, {output}), {output});

  // ── The four learned sentinel vectors ─────────────────────────────────────
  out.image_start = read.Sentinel(kImageStart);
  out.image_end = read.Sentinel(kImageEnd);
  out.image_newline = read.Sentinel(kImageNewline);
  out.image_pad = read.Sentinel(kImagePad);

  // Totality, in the direction the per-tensor refusals above cannot see. Every
  // enumerated name was read exactly once; if this ever disagrees, a name was
  // read twice and another never at all.
  const size_t expected = DeepSeekV4OfficialVisionExpectedTensors(config).size();
  VT_CHECK(read.consumed() == expected,
           "deepseek-v4 vision loader: consumed " +
               std::to_string(read.consumed()) + " of " +
               std::to_string(expected) +
               " enumerated vision tensors; the reader and its name map "
               "disagree");
  return out;
}

}  // namespace vllm
