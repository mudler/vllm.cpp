// Ported from: vllm/model_executor/layers/quantization/modelopt.py (NVFP4 W4A16 dequant) @ e24d1b24
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"

#include <cmath>
#include <limits>

#include "vt/dtype.h"

namespace vllm {

float F8E4M3ToF32(uint8_t byte) {
  // IEEE fp8-e4m3fn: 1 sign | 4 exp | 3 mantissa, bias 7, finite (no inf),
  // NaN only at 0x7F / 0xFF (S.1111.111). Mirrors
  // torch .view(float8_e4m3fn).to(float32).
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1U;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFU;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7U;
  const float sign_mul = sign ? -1.0F : 1.0F;

  if (exp == 0xFU && mant == 0x7U) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (exp == 0U) {
    // Subnormal: value = mant/8 * 2^(1-7) = mant * 2^-9.
    return sign_mul * (static_cast<float>(mant) * (1.0F / 512.0F));
  }
  // Normal: value = 2^(exp-7) * (1 + mant/8).
  const float mantissa = 1.0F + static_cast<float>(mant) * (1.0F / 8.0F);
  const int e = static_cast<int>(exp) - 7;
  return sign_mul * std::ldexp(mantissa, e);
}

void DequantNvfp4ToBf16(const uint8_t* packed, const uint8_t* weight_scale_fp8,
                        float weight_scale_2, int64_t out_dim, int64_t in_dim,
                        uint16_t* out_bf16) {
  VT_CHECK(packed != nullptr, "nvfp4 dequant: packed weight is null");
  VT_CHECK(weight_scale_fp8 != nullptr, "nvfp4 dequant: weight_scale is null");
  VT_CHECK(out_bf16 != nullptr, "nvfp4 dequant: output buffer is null");
  VT_CHECK(out_dim >= 0 && in_dim >= 0, "nvfp4 dequant: negative dimension");
  VT_CHECK(in_dim % kNvfp4GroupSize == 0,
           "nvfp4 dequant: in_dim must be a multiple of 16");

  const int64_t packed_cols = in_dim / 2;
  const int64_t groups = in_dim / kNvfp4GroupSize;

  for (int64_t o = 0; o < out_dim; ++o) {
    const uint8_t* packed_row = packed + o * packed_cols;
    const uint8_t* scale_row = weight_scale_fp8 + o * groups;
    uint16_t* out_row = out_bf16 + o * in_dim;

    for (int64_t g = 0; g < groups; ++g) {
      // Group scale: f32(weight_scale) * weight_scale_2 (fp8xws2 computed
      // first), the same-order f32 arithmetic as torch's
      // tensor_sf.to(f32) * global_scale. Only weight_scale_2 carries >4
      // significant bits, so C++ matches torch by construction and the
      // subsequent bf16 store-round is bit-exact.
      const float group_scale = F8E4M3ToF32(scale_row[g]) * weight_scale_2;

      const int64_t base_elem = g * kNvfp4GroupSize;
      // 16 elements per group = 8 packed bytes.
      for (int64_t j = 0; j < kNvfp4GroupSize / 2; ++j) {
        const uint8_t b = packed_row[base_elem / 2 + j];
        const uint8_t low = b & 0x0FU;   // element 2i
        const uint8_t high = b >> 4;     // element 2i+1

        const float lo_val =
            kE2M1Lut[low & 0x7U] * ((low & 0x8U) ? -1.0F : 1.0F);
        const float hi_val =
            kE2M1Lut[high & 0x7U] * ((high & 0x8U) ? -1.0F : 1.0F);

        out_row[base_elem + 2 * j] = vt::F32ToBF16(lo_val * group_scale);
        out_row[base_elem + 2 * j + 1] = vt::F32ToBF16(hi_val * group_scale);
      }
    }
  }
}

}  // namespace vllm
