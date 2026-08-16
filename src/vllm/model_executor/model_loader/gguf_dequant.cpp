// vllm.cpp original GGUF-format dequant loader (porting-inventory.md §9
// deviation). Ported byte-for-byte from llama.cpp @
// 237ad9b961f009ae19ac29dbce4cd0c1251f94b3:
//   ggml/src/ggml-common.h  (block_q2_K/q4_0/q8_0/q3_K/q4_K/q5_K/q6_K +
//                            block_iq2_xxs layouts)
//   ggml/src/ggml-quants.c  (dequantize_row_* + get_scale_min_k4). Q2_K (10)
//                            and IQ2_XXS (16) are the ~2-bit DeepSeek-V4-Flash
//                            GGUF vehicles (UD-Q2_K_XL / UD-IQ2_XXS).
// The BLOCK decoders themselves now live one layer down in vt
// (src/vt/cpu/cpu_quant_dequant.cpp) and are shared with the compute-in-quant
// GEMM; this file keeps the GGUF-facing validation, the unquantized types, and
// the ggml-type -> vt::DType routing. See gguf_dequant.h for the format
// citation.
// NVFP4 (type 40) decodes HERE rather than in vt, because it reuses the
// already-gated safetensors NVFP4 numerics (nvfp4_dequant.h) and has no vt
// block dtype / vec_dot: only its CONTAINER differs from the safetensors form.
#include "vllm/model_executor/model_loader/gguf_dequant.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/dtype.h"
#include "vt/quant.h"

namespace vllm {
namespace {

// Read a little-endian ggml_half (f16) at byte pointer `p` and widen to f32.
// (Aligned load is not guaranteed for mmap'd block bytes, so memcpy.)
float ReadF16(const uint8_t* p) {
  uint16_t h = 0;
  std::memcpy(&h, p, sizeof(h));
  return vt::F16ToF32(h);
}

// One 4-bit e2m1 nibble -> f32, with the SAME LUT and sign rule the
// safetensors NVFP4 path uses (nvfp4_dequant.h `kE2M1Lut`). Shared on purpose:
// the two containers hold the same nibbles, so their decoders must not drift.
inline float E2M1ToF32(uint8_t nibble) {
  return kE2M1Lut[nibble & 0x7U] * ((nibble & 0x8U) != 0 ? -1.0F : 1.0F);
}

}  // namespace

bool GgmlTypeNeedsGlobalScale(uint32_t ggml_type) { return ggml_type == 40; }

std::vector<float> DequantGgufRowToF32(uint32_t ggml_type, const uint8_t* data,
                                       int64_t numel, float global_scale) {
  VT_CHECK(data != nullptr, "gguf dequant: data is null");
  VT_CHECK(numel >= 0, "gguf dequant: negative numel");
  if (GgmlTypeNeedsGlobalScale(ggml_type)) {
    VT_CHECK(std::isfinite(global_scale) && global_scale > 0.0F,
             "gguf dequant: NVFP4 needs the finite, positive per-tensor "
             "<stem>.scale sidecar value");
  } else {
    // Refuse to DROP a scale the caller passed. Every other encoding here is
    // self-contained, so a non-1.0 scale means the caller has confused two
    // tensors and would otherwise be handed silently wrong weights.
    VT_CHECK(global_scale == 1.0F,
             std::string("gguf dequant: ggml type ")
                 .append(std::to_string(ggml_type))
                 .append(" has self-contained blocks and cannot apply a global "
                         "scale")
                 .c_str());
  }

  const GgmlTypeTraits& traits = GgmlTraits(ggml_type);
  const int64_t block_elems = traits.block_elems;
  VT_CHECK(numel % block_elems == 0,
           std::string("gguf dequant: numel not a multiple of block_elems for ")
               .append(traits.name).c_str());

  std::vector<float> out(static_cast<size_t>(numel));
  float* y = out.data();

  // Guard: our hardcoded per-type block byte stride must equal the reader's
  // traits. A mismatch means a port bug in one place or the other.
  auto check_bytes = [&](int64_t bytes) {
    VT_CHECK(traits.block_bytes == bytes,
             std::string("gguf dequant: block_bytes mismatch vs traits for ")
                 .append(traits.name).c_str());
  };

  switch (ggml_type) {
    case 0:  // F32
      check_bytes(4);
      std::memcpy(y, data, static_cast<size_t>(numel) * sizeof(float));
      break;
    case 1:  // F16 (ggml fp16 = IEEE half; unquantized tensors in mixed files)
      check_bytes(2);
      for (int64_t i = 0; i < numel; ++i) y[i] = ReadF16(data + i * 2);
      break;
    case 30: {  // BF16 (upper 16 bits of f32)
      check_bytes(2);
      for (int64_t i = 0; i < numel; ++i) {
        uint16_t b;
        std::memcpy(&b, data + i * 2, sizeof(b));
        uint32_t u = static_cast<uint32_t>(b) << 16;
        std::memcpy(&y[i], &u, sizeof(u));
      }
      break;
    }
    case 2:    // Q4_0
    case 8:    // Q8_0
    case 10:   // Q2_K
    case 11:   // Q3_K
    case 12:   // Q4_K
    case 13:   // Q5_K
    case 14:   // Q6_K
    case 22:   // IQ2_S  (~2.5-bit codebook; UD-IQ2_M ffn_gate/up experts)
    case 39:   // MXFP4  (OCP micro-scaling fp4; UD-IQ2_M ffn_down experts)
    case 18:   // IQ3_XXS (~3-bit codebook; UD-IQ2_XXS ffn_down)
    case 19:   // IQ1_S    (1.5625 bpw; Qwen3.8-2.4T-A95B UD-IQ1_S experts)
    case 66:   // IQ1_XXXS (1.1875 bpw; UD-Q1_0 experts, fork-anchored)
    case 16: {  // IQ2_XXS (~2-bit codebook; UD-IQ2_XXS DeepSeek-V4 vehicle)
      // The block decoders moved to vt (src/vt/cpu/cpu_quant_dequant.cpp) so
      // the loader oracle and the compute-in-quant GEMM's generic fallback
      // share ONE implementation. The code is byte-identical to what lived
      // here, so numerics are unchanged; this test suite gates that.
      // This list is the loader's EXPANSION path, and it has drifted behind
      // `BlockDTypeFromGgmlTypeId` twice: 18, 19 and 66 were all decodable in
      // vt while this switch still threw "unsupported ggml type" for them. The
      // drift is not a corner case, because `RouteGgufTensor` routes a tensor
      // here whenever the VT_CPU_REF oracle switch is on, keep-quant is off, K
      // is ragged, or the role is not a verbatim one. A missing case is
      // therefore a refusal to load the checkpoint on the reference lane.
      // Keep this list in step with vt/dtype.cpp `kBlockDTypes`.
      // NOTE: MXFP4 (39) decodes here through the vt kMXFP4 block dtype — the
      // GGML micro-scaling form (kvalues_mxfp4 == 2*e2m1, scale 2^(byte-128)) —
      // NOT through the compressed-tensors E8M0ToF32 path used for NVFP4 (40)
      // below; the two containers use different scale conventions.
      vt::DType dtype = vt::DType::kF32;
      VT_CHECK(vt::BlockDTypeFromGgmlTypeId(ggml_type, &dtype),
               "gguf dequant: no vt block dtype for this ggml type");
      check_bytes(vt::BlockBytes(dtype));
      VT_CHECK(vt::BlockElems(dtype) == block_elems,
               "gguf dequant: vt block_elems disagrees with reader traits");
      vt::cpu::BlockToFloat(dtype)(data, y, numel);
      break;
    }
    case 26: {  // I32 (raw int32 table tensor — DeepSeek-V4 hash `ffn_gate_tid2eid`
                // stores per-token expert ids as I32 in the real GGUF; the small
                // ids (< n_routed_experts) round-trip exactly through f32).
      check_bytes(4);
      for (int64_t i = 0; i < numel; ++i) {
        int32_t v;
        std::memcpy(&v, data + i * 4, sizeof(v));
        y[i] = static_cast<float>(v);
      }
      break;
    }
    case 40: {  // NVFP4
      // block_nvfp4 = uint8 d[4] then uint8 qs[32], 64 elements in 36 bytes.
      // d[s] is an IEEE fp8-e4m3fn scale for 16-element sub-block s. Sub-block
      // s owns qs[s*8 .. s*8+8); within it, byte j holds element j in the LOW
      // nibble and element j+8 in the HIGH nibble (the ggml split-half packing,
      // NOT the torch pairwise packing the safetensors container uses).
      //
      // Layout established by BYTE-COMPARING the real Qwen3.6-27B-NVFP4 GGUF
      // against the compressed-tensors export of the same quantization run:
      // the scale bytes are identical and the nibbles are exactly this
      // permutation. .agents/specs/gguf-nvfp4-notes.md Sec 5.
      //
      // The arithmetic and its ORDER mirror DequantNvfp4ToBf16: the group scale
      // is formed first (fp8 x global), then multiplied into the e2m1 value, so
      // the two containers agree bit for bit.
      check_bytes(36);
      VT_CHECK(block_elems == 64, "gguf dequant: NVFP4 block_elems must be 64");
      const int64_t nblocks = numel / 64;
      for (int64_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = data + b * 36;
        const uint8_t* qs = blk + 4;
        float* yb = y + b * 64;
        for (int s = 0; s < 4; ++s) {
          const float group_scale = F8E4M3ToF32(blk[s]) * global_scale;
          const uint8_t* sub = qs + s * 8;
          float* ys = yb + s * kNvfp4GroupSize;
          for (int j = 0; j < 8; ++j) {
            const uint8_t byte = sub[j];
            ys[j] = E2M1ToF32(byte & 0x0FU) * group_scale;
            ys[j + 8] = E2M1ToF32(byte >> 4) * group_scale;
          }
        }
      }
      break;
    }
    default:
      throw std::runtime_error("gguf dequant: unsupported ggml type " +
                               std::to_string(ggml_type) +
                               " (" + traits.name + ") (Task 2/i-quant)");
  }
  return out;
}

std::vector<float> DequantGgufRowToF32(uint32_t ggml_type, const uint8_t* data,
                                       int64_t numel) {
  VT_CHECK(!GgmlTypeNeedsGlobalScale(ggml_type),
           std::string("gguf dequant: ggml type ")
               .append(std::to_string(ggml_type))
               .append(" blocks are not self-contained; dequant it through the "
                       "4-argument overload with its <stem>.scale sidecar")
               .c_str());
  return DequantGgufRowToF32(ggml_type, data, numel, 1.0F);
}

std::vector<uint16_t> DequantGgufRowToBf16(uint32_t ggml_type,
                                           const uint8_t* data, int64_t numel,
                                           float global_scale) {
  const std::vector<float> f32 =
      DequantGgufRowToF32(ggml_type, data, numel, global_scale);
  std::vector<uint16_t> out(f32.size());
  for (size_t i = 0; i < f32.size(); ++i) out[i] = vt::F32ToBF16(f32[i]);
  return out;
}

std::vector<uint16_t> DequantGgufRowToBf16(uint32_t ggml_type,
                                           const uint8_t* data, int64_t numel) {
  VT_CHECK(!GgmlTypeNeedsGlobalScale(ggml_type),
           std::string("gguf dequant: ggml type ")
               .append(std::to_string(ggml_type))
               .append(" blocks are not self-contained; dequant it through the "
                       "4-argument overload with its <stem>.scale sidecar")
               .c_str());
  return DequantGgufRowToBf16(ggml_type, data, numel, 1.0F);
}

void RepackGgufNvfp4Rows(const uint8_t* src, int64_t rows, int64_t k,
                         uint8_t* out_packed, uint8_t* out_scale) {
  VT_CHECK(src != nullptr && out_packed != nullptr && out_scale != nullptr,
           "gguf nvfp4 repack: null buffer");
  VT_CHECK(rows >= 0 && k >= 0, "gguf nvfp4 repack: negative extent");
  VT_CHECK(k % 64 == 0,
           "gguf nvfp4 repack: K must be a whole number of 64-element NVFP4 "
           "blocks");
  const int64_t nblocks = k / 64;
  for (int64_t r = 0; r < rows; ++r) {
    const uint8_t* srow = src + r * nblocks * 36;
    uint8_t* prow = out_packed + r * (k / 2);
    uint8_t* scrow = out_scale + r * (k / 16);
    for (int64_t b = 0; b < nblocks; ++b) {
      const uint8_t* blk = srow + b * 36;
      // Scales: 4 fp8-e4m3 bytes per 64-element block, one per 16-element
      // sub-block, already in ascending sub-block order — so the linear
      // [rows, k/16] stream is a straight copy, not a permutation.
      std::memcpy(scrow + b * 4, blk, 4);
      const uint8_t* qs = blk + 4;
      for (int s = 0; s < 4; ++s) {
        const uint8_t* sub = qs + s * 8;
        // ggml: sub[j] holds element j in the LOW nibble and element j+8 in the
        // HIGH nibble. torch: byte i holds element 2i low, 2i+1 high. Nothing
        // crosses a 16-element sub-block, so this is a local 8-byte shuffle.
        uint8_t* dst = prow + (b * 64 + s * 16) / 2;
        for (int i = 0; i < 8; ++i) {
          const int e0 = 2 * i;
          const int e1 = 2 * i + 1;
          const uint8_t n0 = e0 < 8 ? static_cast<uint8_t>(sub[e0] & 0x0FU)
                                    : static_cast<uint8_t>(sub[e0 - 8] >> 4);
          const uint8_t n1 = e1 < 8 ? static_cast<uint8_t>(sub[e1] & 0x0FU)
                                    : static_cast<uint8_t>(sub[e1 - 8] >> 4);
          dst[i] = static_cast<uint8_t>(n0 | static_cast<uint8_t>(n1 << 4));
        }
      }
    }
  }
}

}  // namespace vllm
