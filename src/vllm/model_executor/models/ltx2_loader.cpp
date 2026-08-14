// LTX-2.5 phase L6 — the quantized loaders. See
// include/vllm/model_executor/models/ltx2_loader.h for the port map, the one
// delta, the dtype polarity and the five unported families.
//
// Ported from / grounded in:
//   the swizzle inverted here      <- vllm/model_executor/layers/quantization/
//                                    qutlass_utils.py:165-180 (`to_blocked`) and
//                                    utils/nvfp4_utils.py:44-49
//                                    (`swizzle_blockscale`) — the same
//                                    permutation written twice. qutlass_utils.py
//                                    records its own provenance as a copy of
//                                    torchao/prototype/mx_formats, which is the
//                                    module that quantized this checkpoint.
//   the fp4 decode + group scale   <- REUSED VERBATIM: DequantNvfp4ToBf16
//                                    (nvfp4_dequant.h:59). No new quant scheme.
//   the per-tensor fp8 decode      <- REUSED VERBATIM: DequantFp8ToBf16 (:76).
//   the tensor-at-a-time staging   <- the shape MiniMaxH3 arrived at for the
//                                    same reason (minimax_h3.h:1598-1618,
//                                    minimax_h3_device.cpp:1259-1360).
//   the asset pack                 <- REUSED: Ltx2LoadGemmaAssets
//                                    (ltx2_text_encoder.h:372), phase L3.
#include "vllm/model_executor/models/ltx2_loader.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/unaligned.h"  // LoadUnaligned — safetensors offsets carry no alignment

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("ltx2 loader: " + message);
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

float ReadScalarF32(const std::string& name, const StTensor& t) {
  if (t.dtype != "F32") {
    Fail("'" + name + "' must be F32 (it is the per-tensor scale), not " + t.dtype);
  }
  if (t.nbytes < sizeof(float)) Fail("'" + name + "' is too small to hold an f32 scale");
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(v));
  return v;
}

float Bf16ToF32(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

int64_t RoundUp(int64_t v, int64_t m) { return ((v + m - 1) / m) * m; }

// A contiguous non-owning view of arbitrary rank. `vt::Tensor::Contiguous` takes
// an initializer_list, which a runtime-length shape cannot supply.
vt::Tensor MakeView(void* data, vt::DType dtype, vt::Device device,
                    const std::vector<int64_t>& shape) {
  vt::Tensor t;
  t.data = data;
  t.dtype = dtype;
  t.device = device;
  t.rank = static_cast<int>(shape.size());
  int64_t acc = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = acc;
    acc *= t.shape[i];
  }
  return t;
}

// The module prefix a tensor belongs to, for reporting an unported FAMILY once
// rather than its 129 tensors individually.
std::string FamilyOf(const std::string& name) {
  const size_t dot = name.find('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

// `patchify_proj.weight` -> `patchify_proj`. The torchao helpers take the MODULE
// and append `.weight` / `.weight_scale` / `.weight_scale_2` themselves, so
// handing them a full tensor name yields `patchify_proj.weight.weight_scale_2`,
// which a user grepping the checkpoint will never find. A rank-2 quantized DiT
// tensor is always `<module>.weight` (ltx2.cpp PushLinear), but the fallback
// keeps a future name from silently losing its last component.
std::string ModulePrefixOfWeight(const std::string& tensor_name) {
  static const std::string kSuffix = ".weight";
  if (EndsWith(tensor_name, kSuffix)) {
    return tensor_name.substr(0, tensor_name.size() - kSuffix.size());
  }
  return tensor_name;
}

}  // namespace

// ---------------------------------------------------------------------------
// torchao NVFP4: the marker, and the one delta
// ---------------------------------------------------------------------------

Ltx2TorchaoNvfp4Marker ParseLtx2TorchaoNvfp4Marker(const std::string& module,
                                                   const StTensor& marker) {
  if (marker.data == nullptr || marker.nbytes == 0) {
    Fail("'" + module + "': the torchao_nvfp4 marker is empty");
  }
  const std::string text(reinterpret_cast<const char*>(marker.data), marker.nbytes);
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(text);
  } catch (const std::exception& e) {
    Fail("'" + module + "': the torchao_nvfp4 marker is not JSON (" + e.what() +
         "). Its payload was: " + text.substr(0, 120));
  }
  if (!parsed.is_object()) Fail("'" + module + "': the torchao_nvfp4 marker is not an object");

  Ltx2TorchaoNvfp4Marker m;
  m.format = parsed.value("format", std::string());
  m.block_size = parsed.value("block_size", static_cast<int64_t>(0));
  m.scope = parsed.value("scope", std::string());
  m.config = parsed.value("config", std::string());
  m.is_swizzled_scales = parsed.value("is_swizzled_scales", false);
  m.use_triton_kernel = parsed.value("use_triton_kernel", false);
  m.use_dynamic_activation = parsed.value("use_dynamic_activation", false);
  m.use_dynamic_per_tensor_scale = parsed.value("use_dynamic_per_tensor_scale", false);

  if (m.format != "torchao_nvfp4") {
    Fail("'" + module + "': quantization format is '" + m.format +
         "', not 'torchao_nvfp4'. This port implements torchao NVFP4 only; the "
         "compressed-tensors layout stores its global scales as DIVISORS "
         "(nvfp4_emulation.h:18-23) and reading one as a multiplier is finite and wrong.");
  }
  if (m.block_size != kNvfp4GroupSize) {
    Fail("'" + module + "': torchao block_size is " + std::to_string(m.block_size) +
         ", and this port implements " + std::to_string(kNvfp4GroupSize) +
         " only (nvfp4_dequant.h:32). A different group size regroups every scale.");
  }
  if (!m.is_swizzled_scales) {
    Fail("'" + module +
         "': torchao marker says is_swizzled_scales=false, so its group scales are "
         "LINEAR. Applying the unswizzle to them would permute every scale within a "
         "128x4 tile — finite, correctly shaped and wrong. Read it linearly instead.");
  }
  return m;
}

void Ltx2UnswizzleNvfp4BlockScale(const uint8_t* swizzled, size_t swizzled_bytes,
                                  int64_t rows, int64_t cols, uint8_t* linear) {
  if (swizzled == nullptr || linear == nullptr) Fail("unswizzle: null buffer");
  if (rows <= 0 || cols <= 0) Fail("unswizzle: non-positive block-scale dims");
  const int64_t padded_rows = RoundUp(rows, 128);
  const int64_t padded_cols = RoundUp(cols, 4);
  const size_t want = static_cast<size_t>(padded_rows) * static_cast<size_t>(padded_cols);
  if (swizzled_bytes != want) {
    Fail("unswizzle: the block-scale buffer is " + std::to_string(swizzled_bytes) +
         " bytes but the layout for " + ShapeText({rows, cols}) + " stores " +
         std::to_string(want) + " (padded to " + ShapeText({padded_rows, padded_cols}) +
         "). A short buffer here means the stored shape was read as if it were linear.");
  }
  const int64_t col_tiles = padded_cols / 4;
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t row_tile = r / 128;
    const int64_t within = r % 128;
    const int64_t quarter = within / 32;  // `a`: the 4-way split of the 128-row tile
    const int64_t lane = within % 32;     // `s`
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t col_tile = c / 4;
      const int64_t q = c % 4;
      const int64_t src =
          ((((row_tile * col_tiles) + col_tile) * 32 + lane) * 4 + quarter) * 4 + q;
      linear[static_cast<size_t>(r * cols + c)] = swizzled[static_cast<size_t>(src)];
    }
  }
}

std::vector<int64_t> Ltx2Nvfp4ToBlockedScaleShape(int64_t out_features,
                                                  int64_t in_features) {
  const int64_t groups = in_features / kNvfp4GroupSize;
  return {RoundUp(out_features, 128) / 4, RoundUp(groups, 4) * 4};
}

std::vector<int64_t> Ltx2Nvfp4PaddedScaleShape(int64_t out_features,
                                               int64_t in_features) {
  const int64_t groups = in_features / kNvfp4GroupSize;
  return {RoundUp(out_features, 128), RoundUp(groups, 4)};
}

Nvfp4NibbleOrder Ltx2Nvfp4NibbleOrderFor(Ltx2Nvfp4Producer producer) {
  return producer == Ltx2Nvfp4Producer::kTorchao ? Nvfp4NibbleOrder::kLowFirst
                                                 : Nvfp4NibbleOrder::kHighFirst;
}

Ltx2Nvfp4Producer Ltx2ResolveNvfp4Producer(const std::string& module,
                                           const Ltx2TorchaoNvfp4Marker* marker,
                                           const std::vector<int64_t>& scale_shape,
                                           int64_t out_features, int64_t in_features) {
  if (in_features % kNvfp4GroupSize != 0) {
    Fail("'" + module + "': in_features " + std::to_string(in_features) +
         " is not a multiple of the group size " + std::to_string(kNvfp4GroupSize));
  }
  const std::vector<int64_t> blocked = Ltx2Nvfp4ToBlockedScaleShape(out_features, in_features);
  const std::vector<int64_t> padded = Ltx2Nvfp4PaddedScaleShape(out_features, in_features);
  const bool is_blocked = scale_shape == blocked;
  const bool is_padded = scale_shape == padded;

  // The MARKER decides; the shape only corroborates. A disagreement is refused
  // rather than resolved in either direction, because both readings type-check
  // and one of them permutes every scale in a 128x4 tile.
  if (marker != nullptr) {
    if (is_blocked) return Ltx2Nvfp4Producer::kTorchao;
    Fail("'" + module + ".weight_scale' is " + ShapeText(scale_shape) +
         " but the module carries a torchao_nvfp4 marker, and a torchao swizzled scale "
         "for [" + std::to_string(out_features) + ", " + std::to_string(in_features) +
         "] is stored " + ShapeText(blocked) +
         (is_padded ? ". It is instead the cuBLAS-PADDED framing " + ShapeText(padded) +
                          ", which is what the Lightricks nvfp4-prequant tool writes -- and that "
                          "producer also packs the OPPOSITE nibble order. The marker says "
                          "torchao and the shape says nvfp4-prequant; refusing rather "
                          "than picking one, because each reading is finite and wrong "
                          "under the assumption of the other."
                    : ". The declaration and the stored shape disagree.") +
         " See .agents/specs/nvfp4-nibble-order.md section 3.2.");
  }

  // No marker anywhere for this module. torchao ALWAYS writes one, so its absence
  // excludes torchao — the inference ltx2_loader.h records, with its evidence.
  if (is_padded) return Ltx2Nvfp4Producer::kNvfp4Prequant;
  Fail("'" + module + ".weight_scale' is " + ShapeText(scale_shape) +
       " and the module carries NO torchao_nvfp4 marker. Without a marker this port "
       "reads the Lightricks nvfp4-prequant layout, whose swizzled scale for [" +
       std::to_string(out_features) + ", " + std::to_string(in_features) +
       "] is stored " + ShapeText(padded) +
       (is_blocked ? " -- but this is the torchao to_blocked framing " + ShapeText(blocked) +
                         ", and a torchao file without its marker is a combination this "
                         "port does not recognise. The two producers pack DIFFERENT "
                         "nibble orders, so guessing picks a byte encoding as well as a "
                         "layout."
                   : ", which this is not.") +
       " Refusing by name. See .agents/specs/nvfp4-nibble-order.md section 3.2.");
}

void Ltx2DequantNvfp4ToBf16(const std::string& module, const StTensor& packed,
                            const StTensor& scale, const StTensor& scale_2,
                            int64_t out_features, int64_t in_features,
                            Ltx2Nvfp4Producer producer, uint16_t* out_bf16) {
  if (packed.dtype != "U8") {
    Fail("'" + module + ".weight' must be U8 (two E2M1 values per byte), not " +
         packed.dtype);
  }
  if (packed.shape.size() != 2) {
    Fail("'" + module + ".weight' is rank " + std::to_string(packed.shape.size()) + " (" +
         ShapeText(packed.shape) + "); an NVFP4-packed weight is rank 2.");
  }
  if (packed.shape[0] != out_features || packed.shape[1] * 2 != in_features) {
    // Print the LOGICAL width the stored shape implies, never the stored one
    // twice. NVFP4 packs two values per byte, so "is [128, 8] but the module is
    // [128, 8]" — which is what naming the stored shape on both sides produced —
    // reads as a contradiction and tells a reader nothing.
    Fail("'" + module + ".weight' is stored " + ShapeText(packed.shape) +
         ", i.e. a LOGICAL " + ShapeText({packed.shape[0], packed.shape[1] * 2}) +
         ", but the module is " + ShapeText({out_features, in_features}) +
         ". NVFP4 packs TWO values per byte along the last dimension, so the stored "
         "width must be exactly half the logical one.");
  }
  if (in_features % kNvfp4GroupSize != 0) {
    Fail("'" + module + "': in_features " + std::to_string(in_features) +
         " is not a multiple of the group size " + std::to_string(kNvfp4GroupSize));
  }
  if (scale.dtype != "F8_E4M3") {
    Fail("'" + module + ".weight_scale' must be F8_E4M3, not " + scale.dtype);
  }
  const int64_t groups = in_features / kNvfp4GroupSize;
  // The shape the RESOLVED producer stores. Both framings dress the same buffer,
  // so this asserts the caller's resolution against the file rather than
  // re-deciding it: a caller that resolved torchao and hands a padded-framing
  // scale is refused here even though the bytes would unswizzle identically,
  // because it would then decode them with the wrong NIBBLE ORDER.
  const std::vector<int64_t> want_shape =
      producer == Ltx2Nvfp4Producer::kTorchao
          ? Ltx2Nvfp4ToBlockedScaleShape(out_features, in_features)
          : Ltx2Nvfp4PaddedScaleShape(out_features, in_features);
  if (scale.shape != want_shape) {
    const std::vector<int64_t> other =
        producer == Ltx2Nvfp4Producer::kTorchao
            ? Ltx2Nvfp4PaddedScaleShape(out_features, in_features)
            : Ltx2Nvfp4ToBlockedScaleShape(out_features, in_features);
    Fail("'" + module + ".weight_scale' is " + scale.dtype + " " + ShapeText(scale.shape) +
         ", but this module resolved to the " +
         (producer == Ltx2Nvfp4Producer::kTorchao ? "torchao" : "nvfp4-prequant") +
         " producer, whose swizzled group scale for [" + std::to_string(out_features) +
         ", " + std::to_string(in_features) + "] is stored " + ShapeText(want_shape) +
         ". The other producer framing would be " + ShapeText(other) +
         "; both have the same element count, so reading one as the other type-checks "
         "and — because the two producers also pack OPPOSITE nibble orders — decodes "
         "every weight pair transposed. See .agents/specs/nvfp4-nibble-order.md.");
  }
  const float global = ReadScalarF32(module + ".weight_scale_2", scale_2);

  // ONE permutation, both framings. `Ltx2UnswizzleNvfp4BlockScale` keys off the
  // LOGICAL rows/cols and the byte count, which are identical either way, so
  // nothing about the swizzle inverse is producer-specific.
  std::vector<uint8_t> linear(static_cast<size_t>(out_features) *
                              static_cast<size_t>(groups));
  Ltx2UnswizzleNvfp4BlockScale(static_cast<const uint8_t*>(scale.data), scale.nbytes,
                               out_features, groups, linear.data());
  // From here the path is the EXISTING modelopt one, byte for byte, except for the
  // producer's nibble order.
  DequantNvfp4ToBf16(static_cast<const uint8_t*>(packed.data), linear.data(), global,
                     out_features, in_features, out_bf16,
                     Ltx2Nvfp4NibbleOrderFor(producer));
}

// ---------------------------------------------------------------------------
// The DiT
// ---------------------------------------------------------------------------

namespace {

// Everything a DiT checkpoint's header tells us, before any payload is touched.
struct DitPlan {
  Ltx2DitQuant quant = Ltx2DitQuant::kFp8;
  std::string prefix;
  // contract name -> file name
  std::map<std::string, std::string> file_name;
  // contract name -> LOGICAL shape (U8 widths already doubled)
  std::map<std::string, std::vector<int64_t>> logical;
  std::vector<Ltx2TensorSpec> manifest;
  std::vector<std::string> unported;  // module families, header order, deduped
  // module (bare) -> the `<module>.torchao_nvfp4` FILE name, for every module that
  // declares one. Absent from this map means the file carries NO marker for that
  // module, which is what `Ltx2ResolveNvfp4Producer` reads as "not torchao".
  std::map<std::string, std::string> marker_name;
};

bool IsScaleSidecar(const std::string& bare) {
  return EndsWith(bare, ".weight_scale") || EndsWith(bare, ".weight_scale_2") ||
         EndsWith(bare, "_scale") || EndsWith(bare, kLtx2TorchaoNvfp4MarkerSuffix);
}

DitPlan PlanDit(const SafetensorsFile& file) {
  DitPlan plan;
  const std::string prefix = kLtx2DitCheckpointPrefix;
  const std::vector<std::string>& names = file.Names();
  if (names.empty()) Fail("the DiT checkpoint has no tensors");

  int64_t prefixed = 0;
  for (const std::string& n : names) {
    if (StartsWith(n, prefix)) ++prefixed;
  }
  if (prefixed != 0 && prefixed != static_cast<int64_t>(names.size())) {
    Fail("the DiT checkpoint mixes prefixed and unprefixed names (" +
         std::to_string(prefixed) + " of " + std::to_string(names.size()) +
         " carry '" + prefix +
         "'). Stripping one prefix from some names and not others would bind two "
         "different models' tensors into one contract.");
  }
  plan.prefix = prefixed != 0 ? prefix : std::string();

  bool saw_u8 = false, saw_f8 = false;
  const std::string marker_suffix = kLtx2TorchaoNvfp4MarkerSuffix;
  for (const std::string& n : names) {
    const std::string bare = n.substr(plan.prefix.size());
    if (EndsWith(bare, marker_suffix)) {
      plan.marker_name[bare.substr(0, bare.size() - marker_suffix.size())] = n;
    }
    if (IsScaleSidecar(bare)) continue;
    const StTensor& t = file.Get(n);
    std::vector<int64_t> shape = t.shape;
    if (t.dtype == "U8") {
      saw_u8 = true;
      if (shape.size() != 2) {
        Fail("'" + bare + "' is U8 (NVFP4-packed) but rank " +
             std::to_string(shape.size()) + "; a packed weight is rank 2");
      }
      shape[1] *= 2;  // TWO values per byte, always
    } else if (t.dtype == "F8_E4M3") {
      saw_f8 = true;
    }
    plan.file_name[bare] = n;
    plan.logical[bare] = shape;
    plan.manifest.push_back({bare, shape});
  }
  if (saw_u8 && saw_f8) {
    Fail(
        "the DiT checkpoint carries BOTH U8-packed (NVFP4) and F8_E4M3 (FP8) weights. "
        "The two arms use different scale sidecars, so a mixed file would be loaded "
        "half one way and half the other.");
  }
  if (!saw_u8 && !saw_f8) {
    Fail(
        "the DiT checkpoint carries no quantized weights at all (no U8 and no "
        "F8_E4M3). A bf16 DiT is not what phase L6 loads; use the L2 path.");
  }
  plan.quant = saw_u8 ? Ltx2DitQuant::kNvfp4 : Ltx2DitQuant::kFp8;
  return plan;
}

// Materialize one contract tensor into `buffer`, returning its dtype.
vt::DType MaterializeDitTensor(const SafetensorsFile& file, const DitPlan& plan,
                               const Ltx2TensorSpec& spec, std::vector<uint8_t>& buffer) {
  const auto it = plan.file_name.find(spec.name);
  if (it == plan.file_name.end()) {
    Fail("the checkpoint is missing '" + spec.name + "' " + ShapeText(spec.shape) +
         ", which the LTX-2.5 DiT contract requires. Refusing rather than binding a "
         "zero-filled tensor.");
  }
  const std::string& fname = it->second;
  const std::vector<int64_t>& got = plan.logical.at(spec.name);
  if (got != spec.shape) {
    Fail("'" + spec.name + "' is " + ShapeText(got) + " in the checkpoint but the "
         "contract requires " + ShapeText(spec.shape));
  }
  const StTensor& t = file.Get(fname);
  int64_t numel = 1;
  for (int64_t d : spec.shape) numel *= d;

  if (t.dtype == "F32") {
    // Kept F32 because the CHECKPOINT stores it F32 (the scale_shift tables).
    // Narrowing a tensor the file itself widened would be the dtype rule
    // applied backwards.
    buffer.resize(static_cast<size_t>(numel) * sizeof(float));
    if (t.nbytes != buffer.size()) {
      Fail("'" + spec.name + "' declares " + std::to_string(t.nbytes) +
           " F32 bytes but its shape needs " + std::to_string(buffer.size()));
    }
    std::memcpy(buffer.data(), t.data, buffer.size());
    return vt::DType::kF32;
  }
  if (t.dtype == "BF16") {
    buffer.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
    if (t.nbytes != buffer.size()) {
      Fail("'" + spec.name + "' declares " + std::to_string(t.nbytes) +
           " BF16 bytes but its shape needs " + std::to_string(buffer.size()));
    }
    std::memcpy(buffer.data(), t.data, buffer.size());
    return vt::DType::kBF16;
  }
  if (t.dtype == "F8_E4M3") {
    const std::string sname = fname + "_scale";
    const float scale = ReadScalarF32(spec.name + "_scale", file.Get(sname));
    buffer.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
    DequantFp8ToBf16(static_cast<const uint8_t*>(t.data), scale, numel,
                     reinterpret_cast<uint16_t*>(buffer.data()));
    return vt::DType::kBF16;
  }
  if (t.dtype == "U8") {
    if (spec.shape.size() != 2) {
      Fail("'" + spec.name + "' is NVFP4-packed but rank " +
           std::to_string(spec.shape.size()));
    }
    const StTensor& scale = file.Get(fname + "_scale");
    const StTensor& global = file.Get(fname + "_scale_2");
    buffer.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
    // Ltx2DequantNvfp4ToBf16 takes a MODULE prefix and appends `.weight`,
    // `.weight_scale` and `.weight_scale_2` itself. `spec.name` is already the
    // full tensor name, so passing it produced 'patchify_proj.weight.weight' —
    // a name that appears nowhere in the checkpoint, which defeats the whole
    // point of refusing BY NAME. Strip the suffix the callee will re-add.
    const std::string module = ModulePrefixOfWeight(spec.name);
    // Which producer wrote this file decides BOTH the scale framing and the nibble
    // order, and it is resolved from the marker the file does or does not carry.
    const auto marker_it = plan.marker_name.find(module);
    Ltx2TorchaoNvfp4Marker marker;
    const bool has_marker = marker_it != plan.marker_name.end();
    if (has_marker) {
      marker = ParseLtx2TorchaoNvfp4Marker(module, file.Get(marker_it->second));
    }
    const Ltx2Nvfp4Producer producer = Ltx2ResolveNvfp4Producer(
        module, has_marker ? &marker : nullptr, scale.shape, spec.shape[0], spec.shape[1]);
    Ltx2DequantNvfp4ToBf16(module, t, scale, global, spec.shape[0], spec.shape[1],
                           producer, reinterpret_cast<uint16_t*>(buffer.data()));
    return vt::DType::kBF16;
  }
  Fail("'" + spec.name + "' has dtype " + t.dtype + ", which this loader does not read");
}

// The families the file carries and NOTHING IN THIS PORT reads.
//
// The two `*_embeddings_connector` families are outside the DiT contract and are
// NOT unported: phase L9c materializes them through `Ltx2LoadConnectorWeights`
// and the video engine runs them, which is where upstream puts them too — its
// key ops rewrite `model.diffusion_model.video_embeddings_connector.` into the
// TEXT ENCODER's `EmbeddingsProcessor` (encoder_configurator.py:331-346). Naming
// them here would make the refusal say something untrue about the tree, and
// would demand `allow_unported_modules` from a caller whose checkpoint this port
// reads completely. Their contract is checked where it can be: against a parsed
// connector configuration, in `Ltx2LoadConnectorWeights`, which refuses a
// missing tensor, a wrong shape, and a leftover tensor by name.
bool LoadedElsewhere(const std::string& family) {
  return family == "video_embeddings_connector" || family == "audio_embeddings_connector";
}

std::vector<std::string> UnportedFamilies(const DitPlan& plan,
                                          const std::vector<Ltx2TensorSpec>& contract) {
  std::set<std::string> known;
  for (const Ltx2TensorSpec& spec : contract) known.insert(spec.name);
  std::vector<std::string> families;
  std::set<std::string> seen;
  for (const Ltx2TensorSpec& spec : plan.manifest) {
    if (known.count(spec.name) != 0) continue;
    const std::string family = FamilyOf(spec.name);
    if (LoadedElsewhere(family)) continue;
    if (seen.insert(family).second) families.push_back(family);
  }
  return families;
}

std::vector<Ltx2TensorSpec> ContractOf(const Ltx2DitParams& params) {
  return EnumerateLtx2DitTensors(params);
}

[[noreturn]] void RefuseUnported(const std::vector<std::string>& families) {
  std::string list;
  for (size_t i = 0; i < families.size(); ++i) {
    if (i != 0) list += ", ";
    list += families[i];
  }
  Fail(
      "the checkpoint carries modules this port does NOT carry: " + list +
      ". They are not dropped silently: keyframes_abs_pos_embedding means "
      "use_keyframes_abs_pos_embedding is TRUE, and nothing here applies it. "
      "prompt_adaln_single / audio_prompt_adaln_single are NO LONGER in this list "
      "— they were ported by row LTX25-PROMPT-ADALN "
      "(.agents/specs/ltx25-prompt-adaln.md, issue #644) and are now part of the "
      "contract whenever the checkpoint carries them. The two "
      "*_embeddings_connector families are not in this list either and never will "
      "be — they are outside the DiT contract by design and are loaded by "
      "Ltx2LoadConnectorWeights, which is what the video engine calls. Pass "
      "Ltx2DitLoadOptions::allow_unported_modules to load the ported SUBSET, which "
      "still reports every one of them.");
}

// THE GUARD THAT REPLACED THREE `use_prompt_adaln_single = false` ASSIGNMENTS.
//
// Those assignments existed only so `EnumerateLtx2DitTensors` would not throw on
// a module this port did not carry. The module is carried now, so the contract
// simply includes it — and the assignments would have become a silent DROP of the
// module's 12 parameters (18 entries in the shipped FP8 manifest, which carries a
// `weight_scale` per quantized weight), reachable through
// `allow_unported_modules=1`, which is exactly the defect issue #644 row 0 fixes.
//
// So the invariant they violated is asserted instead: the resolved flag must say
// what the FILE says. Deliberately an EQUALITY. Clearing it with the tensors
// present is the old defect; setting it with them absent would bind weights that
// are not there. Either way this refuses by name rather than rendering.
void CheckPromptAdalnAgreesWithFile(const DitPlan& plan, const Ltx2DitParams& params,
                                    const char* where) {
  bool file_has = false;
  for (const Ltx2TensorSpec& spec : plan.manifest) {
    if (spec.name == "prompt_adaln_single.linear.weight") {
      file_has = true;
      break;
    }
  }
  if (file_has == params.use_prompt_adaln_single) return;
  Fail(std::string(where) + ": use_prompt_adaln_single resolved to " +
       (params.use_prompt_adaln_single ? "TRUE" : "FALSE") + " while the file " +
       (file_has ? "DOES" : "does NOT") +
       " carry prompt_adaln_single. Upstream builds that module exactly when the flag is set "
       "(model.py:222-226), so the two cannot disagree. A FALSE flag over a file that carries "
       "the module would drop the timestep term from every cross-attention K/V modulation "
       "(transformer.py:441-443) and render with only the static table — finite, same-shaped, "
       "and invisible to every gate, which is why this is checked rather than assumed.");
}

}  // namespace

Ltx2DitParams Ltx2ParseDitParamsFromCheckpoint(const SafetensorsFile& file,
                                               Ltx2DitQuant* out_quant) {
  const DitPlan plan = PlanDit(file);
  if (out_quant != nullptr) *out_quant = plan.quant;
  return ParseLtx2DitParamsFromManifest(plan.manifest);
}

Ltx2DitCheckpoint Ltx2LoadDitFromSafetensors(const SafetensorsFile& file,
                                             const Ltx2DitLoadOptions& options) {
  const DitPlan plan = PlanDit(file);
  Ltx2DitCheckpoint out;
  out.quant = plan.quant;
  out.checkpoint_params = ParseLtx2DitParamsFromManifest(plan.manifest);
  out.params = out.checkpoint_params;
  CheckPromptAdalnAgreesWithFile(plan, out.params, "Ltx2LoadDitFromSafetensors");

  const std::vector<Ltx2TensorSpec> contract = ContractOf(out.params);
  out.unported = UnportedFamilies(plan, contract);
  if (!out.unported.empty() && !options.allow_unported_modules) {
    RefuseUnported(out.unported);
  }

  for (const Ltx2TensorSpec& spec : contract) {
    auto buffer = std::make_shared<Ltx2HostBuffer>();
    buffer->dtype = MaterializeDitTensor(file, plan, spec, buffer->bytes);
    out.views[spec.name] =
        MakeView(buffer->bytes.data(), buffer->dtype, vt::Device{}, spec.shape);
    out.storage.push_back(std::move(buffer));
  }

  if (options.widen_to_f32) Ltx2WidenDitToF32(out);
  out.weights = BindLtx2DitWeights(out.params, out.views);
  return out;
}

void Ltx2WidenDitToF32(Ltx2DitCheckpoint& checkpoint) {
  for (auto& kv : checkpoint.views) {
    vt::Tensor& view = kv.second;
    if (view.dtype != vt::DType::kBF16) continue;
    const int64_t numel = view.Numel();
    auto widened = std::make_shared<Ltx2HostBuffer>();
    widened->dtype = vt::DType::kF32;
    widened->bytes.resize(static_cast<size_t>(numel) * sizeof(float));
    const uint16_t* src = view.Ptr<uint16_t>();
    float* dst = reinterpret_cast<float*>(widened->bytes.data());
    for (int64_t i = 0; i < numel; ++i) dst[i] = Bf16ToF32(src[static_cast<size_t>(i)]);
    view.data = widened->bytes.data();
    view.dtype = vt::DType::kF32;
    checkpoint.storage.push_back(std::move(widened));
  }
  checkpoint.weights = BindLtx2DitWeights(checkpoint.params, checkpoint.views);
}

Ltx2DitCheckpoint Ltx2StreamDitToDevice(vt::Queue& queue, const SafetensorsFile& file,
                                        const Ltx2DitLoadOptions& options) {
  if (options.widen_to_f32) {
    Fail(
        "Ltx2StreamDitToDevice refuses widen_to_f32. Staging at load exists because "
        "GB10 runs host/ATS-retagged decode weights 20-30% slower, and widening while "
        "staging would move twice the bytes to save nothing. Widen a HOST load "
        "instead, for the f32 parity forward.");
  }
  const DitPlan plan = PlanDit(file);
  Ltx2DitCheckpoint out;
  out.quant = plan.quant;
  out.checkpoint_params = ParseLtx2DitParamsFromManifest(plan.manifest);
  out.params = out.checkpoint_params;
  CheckPromptAdalnAgreesWithFile(plan, out.params, "Ltx2StreamDitToDevice");

  const std::vector<Ltx2TensorSpec> contract = ContractOf(out.params);
  out.unported = UnportedFamilies(plan, contract);
  if (!out.unported.empty() && !options.allow_unported_modules) {
    RefuseUnported(out.unported);
  }

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  for (const Ltx2TensorSpec& spec : contract) {
    // ONE tensor's host buffer is live at a time: it is dequantized, uploaded,
    // and dropped before the next is read. That is what keeps peak residency at
    // the device copy plus one tensor rather than two whole models.
    std::vector<uint8_t> host;
    const vt::DType dtype = MaterializeDitTensor(file, plan, spec, host);
    void* device = backend.Alloc(host.size());
    backend.Copy(queue, device, host.data(), host.size());
    backend.Synchronize(queue);  // `host` dies at the end of this iteration
    out.views[spec.name] = MakeView(device, dtype, queue.device, spec.shape);
    // The device allocation's lifetime rides on `device_storage` — the host
    // load's `storage` stays empty on this path — so a staged checkpoint frees
    // exactly like a host one, by dropping the checkpoint.
    out.device_storage.emplace_back(device, [&backend](void* p) { backend.Free(p); });
    load_stats::AddDeviceUpload(host.size());
  }
  out.weights = BindLtx2DitWeights(out.params, out.views);
  return out;
}

// ---------------------------------------------------------------------------
// The text encoder
// ---------------------------------------------------------------------------

namespace {

const StTensor* Find(const SafetensorsFile& file, const std::string& name) {
  const std::vector<std::string>& names = file.Names();
  if (std::find(names.begin(), names.end(), name) == names.end()) return nullptr;
  return &file.Get(name);
}

// Load one caption projection: the U8/NVFP4 weight AND the BF16 bias, which sit
// on different dtype paths — the split ltx2_text_encoder.h:264-269 names as the
// one a loader silently half-does.
Ltx2TextProjection LoadProjection(const SafetensorsFile& file, const std::string& module,
                                  int64_t in_features) {
  const StTensor* w = Find(file, module + ".weight");
  if (w == nullptr) Fail("the text encoder is missing '" + module + ".weight'");
  if (w->shape.size() != 2) {
    Fail("'" + module + ".weight' is rank " + std::to_string(w->shape.size()) +
         "; a caption projection is rank 2");
  }
  Ltx2TextProjection proj;
  proj.out_features = w->shape[0];
  proj.in_features = w->shape[1] * 2;  // NVFP4 packs TWO values per byte
  if (proj.in_features != in_features) {
    Fail("'" + module + ".weight' unpacks to in_features " +
         std::to_string(proj.in_features) + " but the Gemma geometry gives " +
         std::to_string(in_features) +
         " (hidden_size * (num_hidden_layers + 1), feature_extractor.py:120). Reading "
         "the STORED U8 width as logical is what halves it.");
  }
  const StTensor* s = Find(file, module + ".weight_scale");
  const StTensor* g = Find(file, module + ".weight_scale_2");
  if (s == nullptr) Fail("the text encoder is missing '" + module + ".weight_scale'");
  if (g == nullptr) Fail("the text encoder is missing '" + module + ".weight_scale_2'");
  // Resolved, not assumed — even though this file is torchao and its marker is
  // present. Hard-coding kTorchao here would make the projections the one NVFP4
  // path in the loader that cannot notice a producer change.
  const StTensor* m = Find(file, module + kLtx2TorchaoNvfp4MarkerSuffix);
  Ltx2TorchaoNvfp4Marker marker;
  if (m != nullptr) marker = ParseLtx2TorchaoNvfp4Marker(module, *m);
  const Ltx2Nvfp4Producer producer = Ltx2ResolveNvfp4Producer(
      module, m != nullptr ? &marker : nullptr, s->shape, proj.out_features,
      proj.in_features);
  proj.weight_bf16.resize(static_cast<size_t>(proj.out_features) *
                          static_cast<size_t>(proj.in_features));
  Ltx2DequantNvfp4ToBf16(module, *w, *s, *g, proj.out_features, proj.in_features,
                         producer, proj.weight_bf16.data());

  const StTensor* b = Find(file, module + ".bias");
  if (b != nullptr) {
    if (b->dtype != "BF16") {
      Fail("'" + module + ".bias' must be BF16 (it is not quantized), not " + b->dtype);
    }
    if (b->shape.size() != 1 || b->shape[0] != proj.out_features) {
      Fail("'" + module + ".bias' is " + ShapeText(b->shape) + " but the projection has " +
           std::to_string(proj.out_features) + " outputs");
    }
    proj.bias_bf16.resize(static_cast<size_t>(proj.out_features));
    std::memcpy(proj.bias_bf16.data(), b->data, b->nbytes);
  }
  return proj;
}

}  // namespace

Ltx2TextEncoderCheckpoint Ltx2LoadTextEncoderFromSafetensors(
    const SafetensorsFile& file, const Ltx2TextEncoderLoadOptions& options) {
  Ltx2TextEncoderCheckpoint out;

  // The width authority is the UNPACKED tensor. `model.norm.weight` is BF16, so
  // it is stored one value per element and cannot be misread by a factor of two.
  const StTensor* norm = Find(file, "model.norm.weight");
  if (norm == nullptr) {
    Fail(
        "the text encoder is missing 'model.norm.weight'. It is the only UNPACKED "
        "tensor that fixes the Gemma hidden size; every quantized width in this file "
        "is half its logical value, so without it the geometry is a guess.");
  }
  if (norm->dtype != "BF16" || norm->shape.size() != 1) {
    Fail("'model.norm.weight' must be BF16 rank 1, not " + norm->dtype + " " +
         ShapeText(norm->shape));
  }
  out.gemma_hidden_size = norm->shape[0];

  int64_t layers = 0;
  const std::string layer_prefix = "model.layers.";
  for (const std::string& name : file.Names()) {
    if (!StartsWith(name, layer_prefix)) continue;
    const size_t dot = name.find('.', layer_prefix.size());
    if (dot == std::string::npos) continue;
    const std::string index = name.substr(layer_prefix.size(), dot - layer_prefix.size());
    if (index.empty() || index.find_first_not_of("0123456789") != std::string::npos) continue;
    layers = std::max<int64_t>(layers, std::stoll(index) + 1);
  }
  if (layers <= 0) Fail("the text encoder has no 'model.layers.<i>.*' tensors");
  out.gemma_num_hidden_layers = layers;

  // Every quantized module in the file — the multimodal tower included — is
  // VALIDATED, so an unreadable one is a load-time refusal by name rather than a
  // phase-L7 surprise. Nothing but the two caption projections is materialized.
  const std::string marker_suffix = kLtx2TorchaoNvfp4MarkerSuffix;
  for (const std::string& name : file.Names()) {
    if (!EndsWith(name, marker_suffix)) continue;
    const std::string module = name.substr(0, name.size() - marker_suffix.size());
    out.quantized_modules.push_back(module);
    const Ltx2TorchaoNvfp4Marker parsed = ParseLtx2TorchaoNvfp4Marker(module, file.Get(name));

    const StTensor* w = Find(file, module + ".weight");
    const StTensor* s = Find(file, module + ".weight_scale");
    const StTensor* g = Find(file, module + ".weight_scale_2");
    if (w == nullptr || s == nullptr || g == nullptr) {
      Fail("'" + module +
           "' carries a torchao_nvfp4 marker but is missing " +
           std::string(w == nullptr ? "weight " : "") +
           std::string(s == nullptr ? "weight_scale " : "") +
           std::string(g == nullptr ? "weight_scale_2 " : "") +
           "- an incomplete quantized module cannot be dequantized, and skipping it "
           "would read as zeros.");
    }
    if (w->dtype != "U8" || w->shape.size() != 2) {
      Fail("'" + module + ".weight' must be U8 rank 2, not " + w->dtype + " " +
           ShapeText(w->shape));
    }
    const int64_t out_features = w->shape[0];
    const int64_t in_features = w->shape[1] * 2;
    if (in_features % kNvfp4GroupSize != 0) {
      Fail("'" + module + "' unpacks to in_features " + std::to_string(in_features) +
           ", not a multiple of " + std::to_string(kNvfp4GroupSize));
    }
    if (s->dtype != "F8_E4M3") {
      Fail("'" + module + ".weight_scale' is " + s->dtype +
           ", and an NVFP4 group scale is F8_E4M3");
    }
    // Every module in this loop carries a marker (the loop is keyed on them), so
    // this validates the torchao framing — through the SAME resolver the
    // materializing paths use, so a validation pass cannot accept a shape the
    // load would then refuse.
    (void)Ltx2ResolveNvfp4Producer(module, &parsed, s->shape, out_features, in_features);
    if (g->dtype != "F32") {
      Fail("'" + module + ".weight_scale_2' must be F32, not " + g->dtype);
    }
  }

  const int64_t in_features = out.gemma_hidden_size * (out.gemma_num_hidden_layers + 1);
  if (!options.skip_projections) {
    out.video = LoadProjection(file, "text_embedding_projection.video_aggregate_embed",
                               in_features);
    if (Find(file, "text_embedding_projection.audio_aggregate_embed.weight") != nullptr) {
      out.audio = LoadProjection(file, "text_embedding_projection.audio_aggregate_embed",
                                 in_features);
    }
  }

  // Phase L3's own asset reader, unchanged: the tokenizer ships AS A TENSOR.
  out.assets = Ltx2LoadGemmaAssets(file, options.require_config);
  return out;
}

Ltx2TextEncoderWeights Ltx2WidenTextProjectionsToF32(
    const Ltx2TextEncoderCheckpoint& checkpoint) {
  auto widen = [](const Ltx2TextProjection& src, Ltx2TextAggregateEmbed& dst) {
    dst.out_features = src.out_features;
    dst.in_features = src.in_features;
    dst.weight.resize(src.weight_bf16.size());
    for (size_t i = 0; i < src.weight_bf16.size(); ++i) {
      dst.weight[i] = Bf16ToF32(src.weight_bf16[i]);
    }
    dst.bias.resize(src.bias_bf16.size());
    for (size_t i = 0; i < src.bias_bf16.size(); ++i) {
      dst.bias[i] = Bf16ToF32(src.bias_bf16[i]);
    }
  };
  Ltx2TextEncoderWeights out;
  widen(checkpoint.video, out.video);
  widen(checkpoint.audio, out.audio);
  return out;
}

// ---------------------------------------------------------------------------
// The VAEs, the upsampler and the duration head (phase L7)
// ---------------------------------------------------------------------------

namespace {

// `config.get(key, fallback)`, refusing a present-but-wrong-typed value rather
// than falling back to the default. A checkpoint that says
// `"patch_size": "4"` means something; treating it as absent silently builds a
// decoder for a different latent grid.
template <typename T>
T ConfigGet(const nlohmann::json& config, const std::string& key, T fallback,
            const std::string& where) {
  const auto it = config.find(key);
  if (it == config.end() || it->is_null()) return fallback;
  try {
    return it->get<T>();
  } catch (const std::exception&) {
    Fail("'" + where + "." + key + "' is present but not the expected type (" + it->dump() + ")");
  }
}

// `check_config_value` (ltx_core/utils.py:15-18): the exact assertion upstream
// makes, with the same expectation, so a checkpoint upstream would refuse is
// refused here too instead of being built into a plausible wrong model.
void CheckConfigValue(const nlohmann::json& config, const std::string& key,
                      const nlohmann::json& expected, const std::string& where) {
  const auto it = config.find(key);
  const nlohmann::json actual = it == config.end() ? nlohmann::json() : *it;
  if (actual != expected) {
    Fail("config value " + where + "." + key + " is " + actual.dump() + ", expected " +
         expected.dump() + " (check_config_value, audio_vae/model_configurator.py:61-66)");
  }
}

const nlohmann::json& ConfigObject(const nlohmann::json& parent, const std::string& key,
                                   const std::string& where) {
  static const nlohmann::json kEmpty = nlohmann::json::object();
  const auto it = parent.find(key);
  if (it == parent.end() || it->is_null()) return kEmpty;
  if (!it->is_object()) Fail("'" + where + "." + key + "' is not a JSON object");
  return *it;
}

Ltx2NormLayer ParseNormLayer(const std::string& name) {
  // NormLayerType (video_vae/enums.py). `group_norm` and `pixel_norm` are the
  // two the ConvVideoDecoder builds; anything else is a checkpoint this port has
  // never seen, and mapping it onto the nearest one would normalize with the
  // wrong statistics everywhere.
  if (name == "pixel_norm") return Ltx2NormLayer::kPixelNorm;
  if (name == "group_norm") return Ltx2NormLayer::kGroupNorm;
  Fail("vae.norm_layer is '" + name + "'; only 'pixel_norm' and 'group_norm' are ported");
}

Ltx2PaddingMode ParsePaddingMode(const std::string& name) {
  if (name == "zeros") return Ltx2PaddingMode::kZeros;
  if (name == "reflect") return Ltx2PaddingMode::kReflect;
  if (name == "replicate") return Ltx2PaddingMode::kReplicate;
  Fail("vae.spatial_padding_mode is '" + name +
       "'; only 'zeros', 'reflect' and 'replicate' are ported");
}

Ltx2NormType ParseAudioNormType(const std::string& name) {
  if (name == "pixel") return Ltx2NormType::kPixel;
  if (name == "group") return Ltx2NormType::kGroup;
  Fail("audio_vae ddconfig.norm_type is '" + name + "'; only 'pixel' and 'group' are ported");
}

Ltx2CausalityAxis ParseCausalityAxis(const std::string& name) {
  // causality_axis.py:4-10.
  if (name == "none") return Ltx2CausalityAxis::kNone;
  if (name == "width") return Ltx2CausalityAxis::kWidth;
  if (name == "height") return Ltx2CausalityAxis::kHeight;
  if (name == "width_compatibility") return Ltx2CausalityAxis::kWidthCompatibility;
  Fail("audio_vae ddconfig.causality_axis is '" + name + "'; it is not one of the four");
}

// `[["res_x", {"num_layers": 4}], ["compress_space", {"multiplier": 2}], ...]`
// — the pair form every shipped LTX-2 VAE config uses
// (video_vae/video_vae.py `_make_decoder_block` reads exactly these keys).
std::vector<Ltx2VideoDecoderBlock> ParseDecoderBlocks(const nlohmann::json& blocks) {
  if (!blocks.is_array()) Fail("vae.decoder_blocks is not an array");
  std::vector<Ltx2VideoDecoderBlock> out;
  for (const nlohmann::json& entry : blocks) {
    if (!entry.is_array() || entry.size() != 2 || !entry[0].is_string() || !entry[1].is_object()) {
      Fail("a vae.decoder_blocks entry is not a [name, {params}] pair: " + entry.dump());
    }
    Ltx2VideoDecoderBlock block;
    block.name = entry[0].get<std::string>();
    const nlohmann::json& params = entry[1];
    block.num_layers = ConfigGet<int64_t>(params, "num_layers", 1, "decoder_block");
    // 0 is the sentinel `Ltx2VideoDecoderBlock` documents for "the upstream
    // default for this block kind", so an ABSENT multiplier must stay 0 rather
    // than becoming 1 and quietly halving res_x_y's width.
    block.multiplier = ConfigGet<int64_t>(params, "multiplier", 0, "decoder_block");
    block.inject_noise = ConfigGet<bool>(params, "inject_noise", false, "decoder_block");
    block.residual = ConfigGet<bool>(params, "residual", false, "decoder_block");
    out.push_back(std::move(block));
  }
  if (out.empty()) Fail("vae.decoder_blocks is empty; there is no decoder to build");
  return out;
}

Ltx2VocoderConfig ParseVocoderArm(const nlohmann::json& cfg, const std::string& where,
                                  int64_t output_sampling_rate, bool apply_final_activation,
                                  const std::string& prefix) {
  // `_vocoder_from_config` (audio_vae/model_configurator.py:13-39).
  Ltx2VocoderConfig out;
  out.resblock_kernel_sizes =
      ConfigGet<std::vector<int64_t>>(cfg, "resblock_kernel_sizes", {3, 7, 11}, where);
  out.upsample_rates = ConfigGet<std::vector<int64_t>>(cfg, "upsample_rates", {6, 5, 2, 2, 2}, where);
  out.upsample_kernel_sizes =
      ConfigGet<std::vector<int64_t>>(cfg, "upsample_kernel_sizes", {16, 15, 8, 4, 4}, where);
  out.resblock_dilation_sizes = ConfigGet<std::vector<std::vector<int64_t>>>(
      cfg, "resblock_dilation_sizes", {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}}, where);
  out.upsample_initial_channel =
      ConfigGet<int64_t>(cfg, "upsample_initial_channel", 1024, where);
  // The two ARM selectors. The BWE branch has already asserted both are
  // "AMP1"/"snakebeta" through check_config_value, so these reads cannot
  // disagree with that assertion — they are here so the config, not a default,
  // is what sets them.
  out.amp = ConfigGet<std::string>(cfg, "resblock", "1", where) == "AMP1";
  out.snakebeta = ConfigGet<std::string>(cfg, "activation", "snake", where) == "snakebeta";
  out.use_tanh_at_final = ConfigGet<bool>(cfg, "use_tanh_at_final", true, where);
  out.apply_final_activation = apply_final_activation;
  out.use_bias_at_final = ConfigGet<bool>(cfg, "use_bias_at_final", true, where);
  out.output_sampling_rate = output_sampling_rate;
  out.prefix = prefix;
  return out;
}

}  // namespace

Ltx2DitParams Ltx2AdoptDeclaredDitParams(const nlohmann::json& config,
                                         const Ltx2DitParams& from_shapes,
                                         bool allow_unported_modules,
                                         const std::string& source) {
  nlohmann::json copy = config;
  // THE UNPORTED FLAG IS CLEARED IN THE COPY, NOT ARGUED WITH. `ParseLtx2DitParams`
  // refuses `use_keyframes_abs_pos_embedding` by name (ltx2.cpp:191-196) and the
  // first-party LTX-2.5 DiT declares it, so reading the declared config verbatim
  // would refuse a real checkpoint the loader has just accepted under
  // `allow_unported_modules`.
  //
  // EXACTLY ONE FLAG, and that is now structural rather than a comment. This block
  // also cleared `use_prompt_adaln_single`, whose module IS ported
  // (.agents/specs/ltx25-prompt-adaln.md, issue #644) — so `allow_unported=1`,
  // which a real render needs, silently turned off a correctness setting. Whatever
  // is cleared here must be a module nothing below applies; a ported one belongs
  // in the contract, where the equality check further down can see it.
  if (allow_unported_modules && copy.contains("transformer") &&
      copy["transformer"].is_object()) {
    copy["transformer"]["use_keyframes_abs_pos_embedding"] = false;
  }
  nlohmann::json wrapper;
  wrapper["config"] = copy;
  const Ltx2DitParams declared = ParseLtx2DitParams(wrapper);

  const std::vector<Ltx2TensorSpec> a = EnumerateLtx2DitTensors(from_shapes);
  const std::vector<Ltx2TensorSpec> b = EnumerateLtx2DitTensors(declared);
  bool same = a.size() == b.size();
  for (size_t i = 0; same && i < a.size(); ++i) {
    same = a[i].name == b[i].name && a[i].shape == b[i].shape;
  }
  if (!same) {
    Fail(source +
         " describes a DIFFERENT weight contract from the one its tensor shapes describe. "
         "Refusing rather than preferring either: the config decides values no shape can see "
         "(frequencies_precision, av_ca_timestep_scale_multiplier, the positional-embedding "
         "bounds), so taking the shapes would render with the wrong RoPE and taking the "
         "config would bind the wrong tensors.");
  }
  return declared;
}

nlohmann::json Ltx2ReadCheckpointConfig(const SafetensorsFile& file) {
  const std::map<std::string, std::string>& meta = file.Metadata();
  const auto it = meta.find("config");
  if (it == meta.end()) {
    Fail(
        "the checkpoint carries no __metadata__[\"config\"]. Every shipped LTX-2.5 VAE, "
        "upsampler and duration head embeds its own config there, and upstream reads it "
        "from the same place (video_vae/model_configurator.py:21-24). Refusing rather than "
        "assuming a geometry: a VAE built to the wrong config decodes a finite, "
        "correctly shaped, WRONG picture.");
  }
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(it->second);
  } catch (const std::exception& e) {
    Fail(std::string("__metadata__[\"config\"] is not readable JSON: ") + e.what());
  }
  if (!parsed.is_object()) Fail("__metadata__[\"config\"] is not a JSON object");
  return parsed;
}

// ---------------------------------------------------------------------------
// The embeddings connector (phase L9c) — see ltx2_loader.h for why it loads here
// ---------------------------------------------------------------------------

const char* Ltx2ConnectorCheckpointPrefix(Ltx2ConnectorStream stream) {
  return stream == Ltx2ConnectorStream::kVideo ? "video_embeddings_connector."
                                               : "audio_embeddings_connector.";
}

bool Ltx2CheckpointHasConnector(const SafetensorsFile& file, Ltx2ConnectorStream stream) {
  const std::string bare =
      std::string(Ltx2ConnectorCheckpointPrefix(stream)) + "learnable_registers";
  const std::string prefixed = std::string(kLtx2DitCheckpointPrefix) + bare;
  for (const std::string& n : file.Names()) {
    if (n == bare || n == prefixed) return true;
  }
  return false;
}

namespace {

// `transformer_config.get(key, fallback)` for the four scalar shapes the two
// connector configurators read. Kept local because a config value that is
// present but of the WRONG JSON type must refuse rather than fall back — a
// silent fallback is exactly how a declared 8-layer connector becomes a 2-layer
// one.
int64_t ConnectorInt(const nlohmann::json& t, const std::string& key, int64_t fallback) {
  const auto it = t.find(key);
  if (it == t.end() || it->is_null()) return fallback;
  if (!it->is_number_integer() && !it->is_number_unsigned()) {
    Fail("the connector config key '" + key + "' is " + it->dump() + ", not an integer");
  }
  return it->get<int64_t>();
}

bool ConnectorBool(const nlohmann::json& t, const std::string& key, bool fallback) {
  const auto it = t.find(key);
  if (it == t.end() || it->is_null()) return fallback;
  if (!it->is_boolean()) {
    Fail("the connector config key '" + key + "' is " + it->dump() + ", not a boolean");
  }
  return it->get<bool>();
}

}  // namespace

Ltx2ConnectorConfig Ltx2ParseConnectorConfig(const nlohmann::json& config,
                                             Ltx2ConnectorStream stream) {
  const auto tit = config.find("transformer");
  if (tit == config.end() || !tit->is_object()) {
    Fail("the connector configuration needs a `{\"transformer\": {...}}` object; the two "
         "configurators read the DiT's own transformer config "
         "(embeddings_connector.py:196, :227)");
  }
  const nlohmann::json& t = *tit;

  Ltx2ConnectorConfig out;
  out.prefix = Ltx2ConnectorCheckpointPrefix(stream);

  // :198, :226 — `LTXRopeType(transformer_config.get("rope_type", "split"))`.
  const auto rope = t.find("rope_type");
  std::string rope_type = "split";
  if (rope != t.end() && !rope->is_null()) {
    if (!rope->is_string()) Fail("the connector config key 'rope_type' is not a string");
    rope_type = rope->get<std::string>();
  }
  if (rope_type == "split") {
    out.rope_type = Ltx2RopeType::kSplit;
  } else if (rope_type == "interleaved") {
    out.rope_type = Ltx2RopeType::kInterleaved;
  } else {
    Fail("the connector config declares rope_type '" + rope_type +
         "', which is neither 'split' nor 'interleaved' (rope.py:11-13)");
  }

  // :199, :228 — `transformer_config.get("frequencies_precision", False) == "float64"`.
  // Absent or false means SINGLE precision, so a non-string value is not an
  // error upstream; it simply is not "float64".
  const auto freq = t.find("frequencies_precision");
  out.double_precision_rope =
      freq != t.end() && freq->is_string() && freq->get<std::string>() == "float64";

  // :200, :229 — `connector_positional_embedding_max_pos`, default [1]. The one
  // value the shipped config moves off its default, and it scales every position.
  const auto max_pos = t.find("connector_positional_embedding_max_pos");
  if (max_pos == t.end() || max_pos->is_null()) {
    out.positional_embedding_max_pos = {1};
  } else {
    if (!max_pos->is_array()) {
      Fail("the connector config key 'connector_positional_embedding_max_pos' is " +
           max_pos->dump() + ", not an array");
    }
    out.positional_embedding_max_pos.clear();
    for (const nlohmann::json& e : *max_pos) {
      if (!e.is_number()) {
        Fail("'connector_positional_embedding_max_pos' holds a non-numeric entry " + e.dump());
      }
      out.positional_embedding_max_pos.push_back(e.get<int64_t>());
    }
    if (out.positional_embedding_max_pos.size() != 1) {
      // `get_fractional_positions` asserts the grid's position-dimension count
      // equals `len(max_pos)` (rope.py:130-134), and the connector's grid is 1-D.
      Fail("'connector_positional_embedding_max_pos' has " +
           std::to_string(out.positional_embedding_max_pos.size()) +
           " entries; the connector's position grid is 1-D (embeddings_connector.py:172-174) "
           "so exactly one is required");
    }
  }

  // :203-206 for video; :232-241 for audio, which falls back to the VIDEO keys
  // "for backwards compatibility" rather than to the class defaults.
  if (stream == Ltx2ConnectorStream::kVideo) {
    out.num_attention_heads = ConnectorInt(t, "connector_num_attention_heads", 30);
    out.attention_head_dim = ConnectorInt(t, "connector_attention_head_dim", 128);
    out.num_layers = ConnectorInt(t, "connector_num_layers", 2);
  } else {
    out.num_attention_heads = ConnectorInt(
        t, "audio_connector_num_attention_heads", ConnectorInt(t, "connector_num_attention_heads", 30));
    out.attention_head_dim = ConnectorInt(
        t, "audio_connector_attention_head_dim", ConnectorInt(t, "connector_attention_head_dim", 128));
    out.num_layers =
        ConnectorInt(t, "audio_connector_num_layers", ConnectorInt(t, "connector_num_layers", 2));
  }
  // :212-213, :253-254 — BOTH configurators read the VIDEO spellings of these two.
  out.apply_gated_attention = ConnectorBool(t, "connector_apply_gated_attention", false);
  out.ff_bias = ConnectorBool(t, "connector_ff_bias", true);

  // NOT read by either configurator: `Embeddings1DConnector`'s own default of 128
  // is what upstream always runs (:102, :131). The shipped config declares
  // `connector_num_learnable_registers` and declares 128, so the two agree; it is
  // read here so a checkpoint that declares something else is not silently run at
  // 128, and `Ltx2LoadConnectorWeights` checks it against the stored table.
  out.num_learnable_registers = ConnectorInt(t, "connector_num_learnable_registers", 128);

  if (out.num_attention_heads < 1 || out.attention_head_dim < 1 || out.num_layers < 1) {
    Fail("the connector config resolves " + std::to_string(out.num_layers) + " layers of " +
         std::to_string(out.num_attention_heads) + " x " +
         std::to_string(out.attention_head_dim) + " heads, which is not a module");
  }
  return out;
}

Ltx2VaeWeights Ltx2LoadConnectorWeights(const SafetensorsFile& file,
                                        const Ltx2ConnectorConfig& config) {
  const DitPlan plan = PlanDit(file);
  const std::vector<Ltx2ConnectorTensorSpec> specs = EnumerateLtx2ConnectorTensors(config);

  Ltx2VaeWeights out;
  for (const Ltx2ConnectorTensorSpec& spec : specs) {
    const auto shape_it = plan.logical.find(spec.name);
    if (shape_it == plan.logical.end()) {
      Fail("the checkpoint is missing '" + spec.name + "' " + ShapeText(spec.shape) +
           ", which the connector configuration resolved from " +
           std::to_string(config.num_layers) + " layers of " +
           std::to_string(config.num_attention_heads) + " x " +
           std::to_string(config.attention_head_dim) +
           " heads. The config and the file describe different modules; refusing rather "
           "than binding the subset that happens to be present.");
    }
    if (shape_it->second != spec.shape) {
      Fail("'" + spec.name + "' is " + ShapeText(shape_it->second) +
           " in the checkpoint but the connector configuration needs " + ShapeText(spec.shape));
    }
    std::vector<uint8_t> bytes;
    const vt::DType dtype = MaterializeDitTensor(file, plan, {spec.name, spec.shape}, bytes);
    int64_t numel = 1;
    for (const int64_t d : spec.shape) numel *= d;
    std::vector<float> widened(static_cast<size_t>(numel));
    if (dtype == vt::DType::kBF16) {
      const auto* src = reinterpret_cast<const uint16_t*>(bytes.data());
      for (int64_t i = 0; i < numel; ++i) widened[static_cast<size_t>(i)] = Bf16ToF32(src[static_cast<size_t>(i)]);
    } else if (dtype == vt::DType::kF32) {
      std::memcpy(widened.data(), bytes.data(), static_cast<size_t>(numel) * sizeof(float));
    } else {
      Fail("'" + spec.name + "' materialized as a dtype the connector bag cannot hold");
    }
    out.tensors[spec.name] = std::move(widened);
  }

  // The other half of the contract check: a tensor of this family that the
  // configuration did NOT enumerate. Without it a config declaring FEWER layers
  // than the file carries binds a valid prefix and runs — the exact shape of the
  // defect the shape check above cannot see.
  int64_t extra = 0;
  std::string first_extra;
  for (const auto& kv : plan.logical) {
    if (!StartsWith(kv.first, config.prefix)) continue;
    if (out.tensors.count(kv.first) != 0) continue;
    if (extra == 0) first_extra = kv.first;
    ++extra;
  }
  if (extra != 0) {
    Fail("the checkpoint carries " + std::to_string(extra) + " '" + config.prefix +
         "' tensors the connector configuration does not name (first: '" + first_extra +
         "'). The configuration resolved " + std::to_string(config.num_layers) +
         " layers, gated_attention=" + (config.apply_gated_attention ? "true" : "false") +
         ", ff_bias=" + (config.ff_bias ? "true" : "false") +
         ". Refusing rather than binding the prefix that matches and leaving the rest.");
  }
  return out;
}

std::string Ltx2ReadCheckpointModelVersion(const SafetensorsFile& file) {
  const std::map<std::string, std::string>& meta = file.Metadata();
  const auto it = meta.find("model_version");
  return it == meta.end() ? std::string() : it->second;
}

std::vector<Ltx2VaeKeyRule> Ltx2VideoVaeDecoderKeyRules() {
  return {
      {"vae.decoder.", ""},
      {"vae.per_channel_statistics.", "per_channel_statistics."},
      {"decoder.", ""},
      {"per_channel_statistics.", "per_channel_statistics."},
  };
}

std::vector<Ltx2VaeKeyRule> Ltx2AudioVaeDecoderKeyRules() {
  return {
      {"audio_vae.decoder.", ""},
      {"audio_vae.per_channel_statistics.", "per_channel_statistics."},
  };
}

std::vector<Ltx2VaeKeyRule> Ltx2VocoderKeyRules() { return {{"vocoder.", ""}}; }

Ltx2VaeWeights Ltx2LoadVaeWeights(const SafetensorsFile& file,
                                  const std::vector<Ltx2VaeKeyRule>& rules) {
  Ltx2VaeWeights out;
  for (const std::string& name : file.Names()) {
    std::string key = name;
    if (!rules.empty()) {
      const Ltx2VaeKeyRule* matched = nullptr;
      for (const Ltx2VaeKeyRule& rule : rules) {
        if (StartsWith(name, rule.match_prefix)) {
          matched = &rule;
          break;
        }
      }
      if (matched == nullptr) continue;  // upstream's filters drop it too
      key = matched->replacement + name.substr(matched->match_prefix.size());
    }
    // A COLLISION IS NOT A LAST-WRITE-WINS. Two file names rewriting onto one
    // module name means the rule set is wrong for this checkpoint, and taking
    // either tensor binds half a model to the other half's weights.
    if (out.tensors.count(key) != 0) {
      Fail("'" + name + "' rewrites onto '" + key +
           "', which another tensor already claimed; the key rules do not fit this checkpoint");
    }
    const StTensor& t = file.Get(name);
    int64_t numel = 1;
    for (const int64_t d : t.shape) numel *= d;
    std::vector<float> values(static_cast<size_t>(numel));
    if (t.dtype == "BF16") {
      if (t.nbytes != static_cast<size_t>(numel) * sizeof(uint16_t)) {
        Fail("'" + name + "' declares " + std::to_string(t.nbytes) +
             " BF16 bytes but its shape needs " +
             std::to_string(static_cast<size_t>(numel) * sizeof(uint16_t)));
      }
      // Byte-wise load through the shared seam, NOT
      // `reinterpret_cast<const uint16_t*>(t.data)[i]`. `t.data` points into the
      // safetensors mmap at `8 + <JSON header length> + <sum of the preceding
      // tensors' sizes>`, and NONE of those three terms is required to be even,
      // so this tensor's first byte is 2-byte aligned only when the writer
      // happened to pad. The cast was UB on every such file, UBSan caught it
      // here (issue #674, "load of misaligned address ... requires 2 byte
      // alignment"), and it is a genuine fault on the strict-alignment targets
      // this builds for (build-test-cpu-arm64, Jetson/Orin sm_110).
      // `vt::LoadUnaligned` is a memcpy with no alignment precondition and
      // compiles to the same single load where the address does happen to be
      // aligned; it is the seam #301 left behind, and
      // minimax_h3_vae_loader.cpp:87-101 took the same repair.
      for (int64_t i = 0; i < numel; ++i) {
        values[static_cast<size_t>(i)] = Bf16ToF32(
            vt::LoadUnaligned<uint16_t>(t.data + static_cast<size_t>(i) * sizeof(uint16_t)));
      }
    } else if (t.dtype == "F32") {
      if (t.nbytes != static_cast<size_t>(numel) * sizeof(float)) {
        Fail("'" + name + "' declares " + std::to_string(t.nbytes) +
             " F32 bytes but its shape needs " +
             std::to_string(static_cast<size_t>(numel) * sizeof(float)));
      }
      std::memcpy(values.data(), t.data, t.nbytes);
    } else {
      Fail("'" + name + "' is " + t.dtype +
           "; the shipped VAE, upsampler and duration-head checkpoints are BF16/F32 and a "
           "quantized one would need its scale sidecars read, not its bytes reinterpreted");
    }
    out.tensors.emplace(std::move(key), std::move(values));
  }
  if (out.tensors.empty()) Fail("no tensor in the checkpoint matched the key rules");
  return out;
}

Ltx2ConvVideoDecoderConfig Ltx2ParseConvVideoDecoderConfig(const nlohmann::json& config,
                                                            Ltx2VideoDecoderKind* out_kind) {
  const nlohmann::json& vae = ConfigObject(config, "vae", "config");
  if (vae.empty()) Fail("the video VAE config carries no 'vae' object");
  // `_vae_class_name_from_metadata` (video_vae/model_configurator.py:21-24): the
  // DEFAULT is the conv class, so a file that omits the key reads as conv —
  // upstream's own rule, kept rather than tightened.
  const std::string class_name =
      ConfigGet<std::string>(vae, "_class_name", "CausalVideoAutoencoder", "vae");
  if (out_kind != nullptr) *out_kind = Ltx2ParseVideoDecoderKind(class_name);

  Ltx2ConvVideoDecoderConfig out;
  // `_build_conv_video_decoder` (video_vae/model_configurator.py:81-94), key for
  // key. `dims` is asserted rather than stored: this port is 3-D only, and a 2-D
  // checkpoint would need a different convolution everywhere.
  const int64_t dims = ConfigGet<int64_t>(vae, "dims", 3, "vae");
  if (dims != 3) Fail("vae.dims is " + std::to_string(dims) + "; only the 3-D decoder is ported");
  out.in_channels = ConfigGet<int64_t>(vae, "latent_channels", 128, "vae");
  out.out_channels = ConfigGet<int64_t>(vae, "out_channels", 3, "vae");
  out.decoder_blocks = ParseDecoderBlocks(vae.contains("decoder_blocks")
                                              ? vae.at("decoder_blocks")
                                              : nlohmann::json::array());
  out.patch_size = ConfigGet<int64_t>(vae, "patch_size", 4, "vae");
  out.norm_layer = ParseNormLayer(ConfigGet<std::string>(vae, "norm_layer", "pixel_norm", "vae"));
  out.causal = ConfigGet<bool>(vae, "causal_decoder", false, "vae");
  out.timestep_conditioning = ConfigGet<bool>(vae, "timestep_conditioning", true, "vae");
  out.spatial_padding_mode =
      ParsePaddingMode(ConfigGet<std::string>(vae, "spatial_padding_mode", "reflect", "vae"));
  out.base_channels = ConfigGet<int64_t>(vae, "decoder_base_channels", 128, "vae");
  // norm_num_groups, norm_eps, pixel_norm_eps and the two decode-noise knobs are
  // NOT config keys upstream — they are constructor defaults on ResnetBlock3D /
  // PixelNorm — so they keep the values ltx2_video_vae.h pins to their upstream
  // lines. Reading them from config would let a file move a constant no golden
  // can see (ltx2_video_vae.h:102-122).
  return out;
}

Ltx2AudioDecoderConfig Ltx2ParseAudioDecoderConfig(const nlohmann::json& config) {
  // `AudioDecoderConfigurator.from_metadata` (audio_vae/model_configurator.py:108-141).
  const nlohmann::json& audio_vae = ConfigObject(config, "audio_vae", "config");
  if (audio_vae.empty()) Fail("the audio VAE config carries no 'audio_vae' object");
  const nlohmann::json& model = ConfigObject(audio_vae, "model", "audio_vae");
  const nlohmann::json& params = ConfigObject(model, "params", "audio_vae.model");
  const nlohmann::json& dd = ConfigObject(params, "ddconfig", "audio_vae.model.params");
  const nlohmann::json& pre = ConfigObject(audio_vae, "preprocessing", "audio_vae");
  const nlohmann::json& mel = ConfigObject(pre, "mel", "audio_vae.preprocessing");
  const nlohmann::json& variables = ConfigObject(audio_vae, "variables", "audio_vae");

  Ltx2AudioDecoderConfig out;
  out.ch = ConfigGet<int64_t>(dd, "ch", 128, "ddconfig");
  out.out_ch = ConfigGet<int64_t>(dd, "out_ch", 2, "ddconfig");
  out.ch_mult = ConfigGet<std::vector<int64_t>>(dd, "ch_mult", {1, 2, 4}, "ddconfig");
  out.num_res_blocks = ConfigGet<int64_t>(dd, "num_res_blocks", 2, "ddconfig");
  out.attn_resolutions =
      ConfigGet<std::vector<int64_t>>(dd, "attn_resolutions", {8, 16, 32}, "ddconfig");
  out.resolution = ConfigGet<int64_t>(dd, "resolution", 256, "ddconfig");
  out.z_channels = ConfigGet<int64_t>(dd, "z_channels", 8, "ddconfig");
  out.norm_type = ParseAudioNormType(ConfigGet<std::string>(dd, "norm_type", "pixel", "ddconfig"));
  out.causality_axis =
      ParseCausalityAxis(ConfigGet<std::string>(dd, "causality_axis", "height", "ddconfig"));
  out.mid_block_add_attention =
      ConfigGet<bool>(dd, "mid_block_add_attention", true, "ddconfig");
  // `ddconfig.mel_bins or mel.n_mel_channels or variables.mel_bins` (:120) —
  // a PYTHON `or` chain, so a 0 falls through exactly as an absent key does.
  int64_t mel_bins = ConfigGet<int64_t>(dd, "mel_bins", 0, "ddconfig");
  if (mel_bins == 0) mel_bins = ConfigGet<int64_t>(mel, "n_mel_channels", 0, "mel");
  if (mel_bins == 0) mel_bins = ConfigGet<int64_t>(variables, "mel_bins", 0, "variables");
  out.mel_bins = mel_bins;
  return out;
}

Ltx2UpsamplerConfig Ltx2ParseUpsamplerConfig(const nlohmann::json& config) {
  // `LatentUpsamplerConfigurator.from_metadata` (upsampler/model_configurator.py:12-30),
  // key for key, off the FLAT config object.
  Ltx2UpsamplerConfig out;
  out.in_channels = ConfigGet<int64_t>(config, "in_channels", 128, "config");
  out.mid_channels = ConfigGet<int64_t>(config, "mid_channels", 512, "config");
  out.num_blocks_per_stage = ConfigGet<int64_t>(config, "num_blocks_per_stage", 4, "config");
  out.dims = ConfigGet<int64_t>(config, "dims", 3, "config");
  out.spatial_upsample = ConfigGet<bool>(config, "spatial_upsample", true, "config");
  out.temporal_upsample = ConfigGet<bool>(config, "temporal_upsample", false, "config");
  out.spatial_scale = ConfigGet<double>(config, "spatial_scale", 2.0, "config");
  out.rational_resampler = ConfigGet<bool>(config, "rational_resampler", false, "config");
  return out;
}

Ltx2VocoderBweConfig Ltx2ParseVocoderBweConfig(const nlohmann::json& config) {
  // `VocoderConfigurator.from_metadata` (audio_vae/model_configurator.py:49-88).
  const nlohmann::json& cfg = ConfigObject(config, "vocoder", "config");
  if (cfg.empty()) Fail("the audio VAE config carries no 'vocoder' object");
  if (!cfg.contains("bwe")) {
    Fail(
        "the vocoder config has no 'bwe' block, so it is the PRE-2.3 flat vocoder "
        "(audio_vae/model_configurator.py:53-56). That arm is not ported: LTX-2.5 ships the "
        "BWE chain, and building the legacy generator instead would emit audio from the "
        "wrong model rather than failing.");
  }
  const nlohmann::json& vocoder_cfg = ConfigObject(cfg, "vocoder", "vocoder");
  const nlohmann::json& bwe_cfg = ConfigObject(cfg, "bwe", "vocoder");

  CheckConfigValue(vocoder_cfg, "resblock", "AMP1", "vocoder.vocoder");
  CheckConfigValue(vocoder_cfg, "stereo", true, "vocoder.vocoder");
  CheckConfigValue(vocoder_cfg, "activation", "snakebeta", "vocoder.vocoder");
  CheckConfigValue(bwe_cfg, "resblock", "AMP1", "vocoder.bwe");
  CheckConfigValue(bwe_cfg, "stereo", true, "vocoder.bwe");
  CheckConfigValue(bwe_cfg, "activation", "snakebeta", "vocoder.bwe");

  const auto input_rate = bwe_cfg.find("input_sampling_rate");
  const auto output_rate = bwe_cfg.find("output_sampling_rate");
  const auto hop = bwe_cfg.find("hop_length");
  const auto n_fft = bwe_cfg.find("n_fft");
  const auto num_mels = bwe_cfg.find("num_mels");
  if (input_rate == bwe_cfg.end() || output_rate == bwe_cfg.end() || hop == bwe_cfg.end() ||
      n_fft == bwe_cfg.end() || num_mels == bwe_cfg.end()) {
    // Upstream subscripts these five (`bwe_cfg["input_sampling_rate"]`, :69-82)
    // rather than `.get`-ing them, so a missing one is a KeyError there and is a
    // refusal here. They set the resample ratio and the mel analysis; a default
    // for any of them retunes the whole BWE stage.
    Fail(
        "the vocoder 'bwe' block is missing one of input_sampling_rate / output_sampling_rate "
        "/ hop_length / n_fft / num_mels, which upstream subscripts directly "
        "(audio_vae/model_configurator.py:69-82)");
  }
  Ltx2VocoderBweConfig out;
  out.input_sampling_rate = input_rate->get<int64_t>();
  out.output_sampling_rate = output_rate->get<int64_t>();
  out.hop_length = hop->get<int64_t>();
  out.filter_length = n_fft->get<int64_t>();
  out.win_length = out.filter_length;  // MelSTFT(win_length=n_fft), :77-82
  out.n_mel_channels = num_mels->get<int64_t>();
  out.prefix = std::string();
  // The inner vocoder's output rate is the BWE's INPUT rate (:69-71), and the
  // BWE generator runs with `apply_final_activation=False` (:72-76).
  out.vocoder = ParseVocoderArm(vocoder_cfg, "vocoder.vocoder", out.input_sampling_rate, true,
                                out.prefix + "vocoder.");
  out.bwe_generator = ParseVocoderArm(bwe_cfg, "vocoder.bwe", out.output_sampling_rate, false,
                                      out.prefix + "bwe_generator.");
  return out;
}

}  // namespace vllm
