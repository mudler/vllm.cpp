// Ported from:
//   compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py:20-97 +
//   utils/mxfp8_utils.py:61-65,222 (E8M0 descale) +
//   tests/quantization/reference_mxfp4.py:28-117 (dq_mxfp4_torch golden)
//   @ pin 555967922 (vLLM 0.26.0.dev0)
#include "vllm/model_executor/model_loader/mxfp4_dequant.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "vt/dtype.h"  // VT_CHECK, F32ToBF16

namespace vllm {

float E8M0ToF32(uint8_t byte) {
  // E8M0 (UE8M0): byte is the biased exponent (bias 127), 0 mantissa, 0 sign.
  // reference_mxfp4.py:31-34  scale_half = 2 ** (scale.to(int16) - 127).
  // 0xFF is the OCP E8M0 NaN encoding.
  if (byte == 0xFFU) return std::numeric_limits<float>::quiet_NaN();
  // ldexp(1, e) == 2^e exactly for e in [-127, 127] (finite normal f32 range).
  return std::ldexp(1.0F, static_cast<int>(byte) - 127);
}

float E8M0BitsToF32(uint8_t byte) {
  // The BITCAST decode, `engram.py:613-614`:
  //   scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
  // "ue8m0 is a power of two, so its byte *is* the fp32 exponent field."
  // Byte 0 therefore yields +0.0 and byte 255 yields +inf, where the arithmetic
  // `E8M0ToF32` above yields 2^-127 and NaN. Both are in use upstream, on
  // different arms; see the header for which arm takes which.
  float value = 0.0F;
  const uint32_t bits = static_cast<uint32_t>(byte) << 23U;
  static_assert(sizeof(float) == sizeof(uint32_t),
                "E8M0BitsToF32 assumes a 32-bit float");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

namespace {

// One row, shared by the bf16 and f32 emitters. `store` maps an f32 dequant
// result to the caller's output element. The E2M1 unpack + per-32-group E8M0
// scale is identical to reference_mxfp4.py's upcast + group multiply; the ONLY
// difference from DequantNvfp4ToBf16 is group size 32, the E8M0 scale, and the
// absence of a global scale.
template <typename Store>
void DequantMxfp4Row(const uint8_t* packed, const uint8_t* scale_e8m0,
                     int64_t out_dim, int64_t in_dim, Store&& store) {
  VT_CHECK(packed != nullptr, "mxfp4 dequant: packed weight is null");
  VT_CHECK(scale_e8m0 != nullptr, "mxfp4 dequant: weight_scale is null");
  VT_CHECK(out_dim >= 0 && in_dim >= 0, "mxfp4 dequant: negative dimension");
  VT_CHECK(in_dim % kMxfp4GroupSize == 0,
           "mxfp4 dequant: in_dim must be a multiple of 32");

  const int64_t packed_cols = in_dim / 2;
  const int64_t groups = in_dim / kMxfp4GroupSize;

  for (int64_t o = 0; o < out_dim; ++o) {
    const uint8_t* packed_row = packed + o * packed_cols;
    const uint8_t* scale_row = scale_e8m0 + o * groups;
    for (int64_t g = 0; g < groups; ++g) {
      const float group_scale = E8M0ToF32(scale_row[g]);
      const int64_t base_elem = g * kMxfp4GroupSize;
      // 32 elements per group = 16 packed bytes.
      for (int64_t j = 0; j < kMxfp4GroupSize / 2; ++j) {
        const uint8_t b = packed_row[base_elem / 2 + j];
        const uint8_t low = b & 0x0FU;   // element 2i
        const uint8_t high = b >> 4;     // element 2i+1
        const float lo_val =
            kE2M1Lut[low & 0x7U] * ((low & 0x8U) ? -1.0F : 1.0F);
        const float hi_val =
            kE2M1Lut[high & 0x7U] * ((high & 0x8U) ? -1.0F : 1.0F);
        store(o * in_dim + base_elem + 2 * j, lo_val * group_scale);
        store(o * in_dim + base_elem + 2 * j + 1, hi_val * group_scale);
      }
    }
  }
}

}  // namespace

void DequantMxfp4ToBf16(const uint8_t* packed, const uint8_t* weight_scale_e8m0,
                        int64_t out_dim, int64_t in_dim, uint16_t* out_bf16) {
  VT_CHECK(out_bf16 != nullptr, "mxfp4 dequant: output buffer is null");
  DequantMxfp4Row(packed, weight_scale_e8m0, out_dim, in_dim,
                  [out_bf16](int64_t idx, float v) {
                    out_bf16[idx] = vt::F32ToBF16(v);
                  });
}

void DequantMxfp4ToF32(const uint8_t* packed, const uint8_t* weight_scale_e8m0,
                       int64_t out_dim, int64_t in_dim, float* out_f32) {
  VT_CHECK(out_f32 != nullptr, "mxfp4 dequant: output buffer is null");
  DequantMxfp4Row(packed, weight_scale_e8m0, out_dim, in_dim,
                  [out_f32](int64_t idx, float v) { out_f32[idx] = v; });
}

}  // namespace vllm
