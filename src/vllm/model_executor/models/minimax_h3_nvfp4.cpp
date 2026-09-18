// MiniMax-H3 NVFP4 arm — the quantized safetensors checkpoint that is also the
// SPEED path on sm_121 (native FP4 tensor cores).
//
// The real 1051-tensor manifest of `lilcheaty/MiniMax-H3-NVFP4` is gated in
// tests/vllm/models/minimax_h3_nvfp4_manifest.inc and shows the checkpoint is the
// textbook compressed-tensors triple this project ALREADY consumes:
//
//   <name>.weight          U8       FP4 packed 2-per-byte, [out, in/2]
//   <name>.weight_scale    F8_E4M3  one per group of 16 along K, [out, in/16]
//   <name>.weight_scale_2  F32      one global scalar
//
// 258 quantized projections carry all three; the fp32/bf16 ISLANDS (both patch
// projections, the time embedder, both output heads, the norms, rope.inv_freq)
// are left unquantized so the DiT's dtype policy survives. The names are
// IDENTICAL to the contract derived from upstream source, so — as with the GGUF
// arm — there is no rename table, only a dequantization decision per tensor.
//
// This loader materializes to f32 for the reference forward. The DEVICE path
// (brick W2b/W10b) will instead keep the packed FP4 resident and route the
// projections through the existing cutlass FP4 GEMM, which is where the speed
// actually comes from; nothing here claims a speed result.
#include "vllm/model_executor/models/minimax_h3.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// vt bf16 bit pattern -> f32.
float Bf16ToF32(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &widened, sizeof(out));
  return out;
}

}  // namespace

bool MiniMaxH3Nvfp4HighNibbleFirst() {
  // Read once. DEFAULT high-first (the shipped community checkpoint's packing);
  // VT_H3_NVFP4_LOWNIBBLE=1 restores the pre-fix modelopt low-first read for A/B.
  static const bool high = [] {
    const char* e = std::getenv("VT_H3_NVFP4_LOWNIBBLE");
    return !(e != nullptr && e[0] == '1');
  }();
  return high;
}

void MiniMaxH3Nvfp4SwapNibbles(const uint8_t* src, size_t n, uint8_t* dst) {
  for (size_t i = 0; i < n; ++i) {
    const uint8_t b = src[i];
    dst[i] = static_cast<uint8_t>((b >> 4) | (b << 4));
  }
}

MiniMaxH3GgufDit LoadMiniMaxH3DitFromNvfp4(const SafetensorsFile& file,
                                          const MiniMaxH3DitLoadOptions& options) {
  MiniMaxH3GgufDit out;

  // --- pass 1: materialize every tensor to f32 ---
  for (const std::string& name : file.Names()) {
    // Quantization sidecars are consumed with their parent, never on their own.
    if (name.size() > 12 &&
        (name.compare(name.size() - 12, 12, "weight_scale") == 0 ||
         (name.size() > 14 && name.compare(name.size() - 14, 14, "weight_scale_2") == 0))) {
      continue;
    }
    // `comfy_quant` is a rank-1 U8 tensor holding the layer's quant config as
    // JSON. Only the PRUNED checkpoints carry it (one per quantized layer); it
    // is metadata, not a weight, and must not reach the packed-FP4 branch below.
    if (name.size() > 12 && name.compare(name.size() - 12, 12, ".comfy_quant") == 0) {
      continue;
    }
    const StTensor& t = file.Get(name);
    if (t.dtype != "U8") {
      out.storage[name] = MiniMaxH3ReadSafetensorF32(t);
      out.shapes[name] = t.shape;
      continue;
    }

    // NVFP4: [out, in/2] packed + [out, in/16] E4M3 group scales + f32 global.
    VT_CHECK(t.shape.size() == 2, "minimax_h3 nvfp4: a packed weight must be rank 2");
    const int64_t out_dim = t.shape[0];
    const int64_t in_dim = t.shape[1] * 2;
    const StTensor& scale = file.Get(name + "_scale");
    const StTensor& global = file.Get(name + "_scale_2");
    VT_CHECK(scale.dtype == "F8_E4M3", "minimax_h3 nvfp4: weight_scale must be F8_E4M3");
    VT_CHECK(global.dtype == "F32", "minimax_h3 nvfp4: weight_scale_2 must be F32");
    VT_CHECK(scale.shape.size() == 2 && scale.shape[0] == out_dim &&
                 scale.shape[1] * 16 == in_dim,
             "minimax_h3 nvfp4: weight_scale must be [out, in/16] (NVFP4 group size 16)");
    float global_scale = 0.0f;
    VT_CHECK(global.nbytes >= sizeof(float), "minimax_h3 nvfp4: weight_scale_2 is too small");
    std::memcpy(&global_scale, global.data, sizeof(float));

    std::vector<uint16_t> bf16(static_cast<size_t>(out_dim * in_dim));
    // Correct the community checkpoint's high-nibble-first fp4 packing before the
    // standard (low-first) dequant. See MiniMaxH3Nvfp4HighNibbleFirst.
    const uint8_t* packed = t.data;
    std::vector<uint8_t> swapped;
    if (MiniMaxH3Nvfp4HighNibbleFirst()) {
      swapped.resize(t.nbytes);
      MiniMaxH3Nvfp4SwapNibbles(t.data, t.nbytes, swapped.data());
      packed = swapped.data();
    }
    DequantNvfp4ToBf16(packed, scale.data, global_scale, out_dim, in_dim, bf16.data());
    std::vector<float> values(bf16.size());
    for (size_t i = 0; i < bf16.size(); ++i) values[i] = Bf16ToF32(bf16[i]);
    out.storage[name] = std::move(values);
    out.shapes[name] = {out_dim, in_dim};
  }

  // --- pass 1b: fuse LoRA deltas into the materialized f32 buffers ---
  if (!options.loras.empty()) {
    std::vector<std::string> contract_names;
    contract_names.reserve(out.shapes.size());
    for (const auto& kv : out.shapes) contract_names.push_back(kv.first);
    const std::vector<DitLoraAdapter> loras = DitOpenLoras(
        options.loras, contract_names, {"model.diffusion_model.", "diffusion_model."});
    int64_t fused = 0;
    for (const auto& kv : out.shapes) {
      const std::string& name = kv.first;
      auto it = out.storage.find(name);
      if (it == out.storage.end()) continue;
      if (DitFuseLorasIntoBuffer(loras, name, kv.second, vt::DType::kF32,
          reinterpret_cast<uint8_t*>(it->second.data()),
          it->second.size() * sizeof(float))) {
        ++fused;
      }
    }
    DitCheckLorasWereApplied(loras, fused);
  }

  // --- pass 2: geometry from the materialized shapes, then bind the views ---
  std::vector<MiniMaxH3TensorSpec> manifest;
  manifest.reserve(out.shapes.size());
  for (const auto& entry : out.shapes) {
    MiniMaxH3TensorSpec spec;
    spec.name = entry.first;
    spec.shape = entry.second;
    manifest.push_back(std::move(spec));
  }
  out.params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  BindMiniMaxH3DitViews(&out);
  return out;
}

}  // namespace vllm
