// Ported from: vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
//   + compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py
//   + kernels/linear/nvfp4/emulation.py @ e24d1b24
#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"

#include <cfenv>
#include <cmath>
#include <vector>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm {

uint8_t F32ToF8E4M3(float f) {
  // torch `.to(torch.float8_e4m3fn)`: round-to-nearest-even, SATURATING to the
  // max finite (+/-448); "FN" has no inf and NaN only at 0x7F/0xFF.
  if (std::isnan(f)) return 0x7FU;
  const uint8_t sign = std::signbit(f) ? 0x80U : 0x00U;
  const float a = std::fabs(f);
  if (!std::isfinite(a) || a >= kFloat8E4M3Max) {
    // +/-inf and out-of-range magnitudes saturate to 448 (byte 0x7E | sign).
    // The exact-448 case also lands here and encodes as 0x7E.
    return static_cast<uint8_t>(sign | 0x7EU);
  }
  if (a == 0.0F) return sign;  // +0 / -0

  int e2 = 0;
  const float frac = std::frexp(a, &e2);  // a = frac * 2^e2, frac in [0.5, 1)
  const int expo = e2 - 1;                // unbiased exponent of the 1.m form
  int exp_field = expo + 7;               // e4m3 bias 7

  if (exp_field <= 0) {
    // Subnormal / underflow: value = q * 2^-9, q in [1, 7]; q == 8 promotes to
    // the smallest normal (exp_field 1, mantissa 0). q rounded to nearest even.
    const double q = static_cast<double>(a) * 512.0;  // / 2^-9
    const int qi = static_cast<int>(std::nearbyint(q));
    if (qi <= 0) return sign;                    // rounds to zero
    if (qi < 8) return static_cast<uint8_t>(sign | static_cast<uint8_t>(qi));
    return static_cast<uint8_t>(sign | (1U << 3));  // smallest normal
  }

  // Normal: significand in [1, 2); mantissa = round(sig * 8) - 8, RNE.
  const double sig = static_cast<double>(frac) * 2.0;  // in [1, 2)
  int mi = static_cast<int>(std::nearbyint(sig * 8.0));  // in [8, 16]
  if (mi == 16) {                                        // mantissa carry
    mi = 8;
    exp_field += 1;
  }
  const int mant = mi - 8;  // 0..7
  if (exp_field > 15 || (exp_field == 15 && mant >= 7)) {
    // exp_field 15 / mant 7 is NaN; anything above 448 saturates to 448.
    return static_cast<uint8_t>(sign | 0x7EU);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint8_t>(exp_field) << 3) |
                              static_cast<uint8_t>(mant));
}

float CastToFp4(float x) {
  // cast_to_fp4 (nvfp4_emulation_utils.py:413-424). torch.sign keeps 0 at 0.
  const float sign = (x > 0.0F) ? 1.0F : (x < 0.0F ? -1.0F : 0.0F);
  const float a = std::fabs(x);
  float m = 0.0F;
  if (a <= 0.25F) {
    m = 0.0F;
  } else if (a < 0.75F) {
    m = 0.5F;
  } else if (a <= 1.25F) {
    m = 1.0F;
  } else if (a < 1.75F) {
    m = 1.5F;
  } else if (a <= 2.5F) {
    m = 2.0F;
  } else if (a < 3.5F) {
    m = 3.0F;
  } else if (a <= 5.0F) {
    m = 4.0F;
  } else {
    m = 6.0F;
  }
  return m * sign;
}

namespace {

// get_reciprocal (nvfp4_emulation_utils.py:403-410) for scalars: 0 stays 0.
inline float Reciprocal(float x) { return x == 0.0F ? 0.0F : 1.0F / x; }

inline float ClampF(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

}  // namespace

void RefNvfp4QuantDequant(const float* x, int64_t m, int64_t n,
                          float input_global_scale, float* x_dq,
                          int block_size) {
  VT_CHECK(x != nullptr, "nvfp4 emulation: x is null");
  VT_CHECK(x_dq != nullptr, "nvfp4 emulation: x_dq is null");
  VT_CHECK(block_size > 0 && n % block_size == 0,
           "nvfp4 emulation: n must be a multiple of block_size");

  const int64_t blocks = n / block_size;
  const float gs_recip = Reciprocal(input_global_scale);
  const float fp4_max_recip = 1.0F / kFloat4E2M1Max;  // FLOAT4_E2M1_MAX_RECIPROCAL

  for (int64_t i = 0; i < m; ++i) {
    const float* x_row = x + i * n;
    float* dq_row = x_dq + i * n;
    for (int64_t b = 0; b < blocks; ++b) {
      const int64_t base = b * block_size;
      // vec_max over the block (f32).
      float vec_max = 0.0F;
      for (int j = 0; j < block_size; ++j) {
        vec_max = std::fmax(vec_max, std::fabs(x_row[base + j]));
      }
      // scale = global_scale * (vec_max / 6); clamp; round to fp8; back to f32.
      float scale = input_global_scale * (vec_max * fp4_max_recip);
      scale = ClampF(scale, -kFloat8E4M3Max, kFloat8E4M3Max);
      const float scale_f8 = F8E4M3ToF32(F32ToF8E4M3(scale));
      // output_scale = get_reciprocal(scale_f8 * get_reciprocal(global_scale)).
      const float block_scale = scale_f8 * gs_recip;  // == x_blockscale (dequant)
      const float output_scale = Reciprocal(block_scale);
      for (int j = 0; j < block_size; ++j) {
        const float scaled = x_row[base + j] * output_scale;
        const float clipped = ClampF(scaled, -kFloat4E2M1Max, kFloat4E2M1Max);
        const float fp4 = CastToFp4(clipped);
        dq_row[base + j] = fp4 * block_scale;
      }
    }
  }
}

void DequantCtNvfp4WeightToF32(const uint8_t* packed,
                               const uint8_t* weight_scale_fp8,
                               float weight_global_scale_disk, int64_t out_dim,
                               int64_t in_dim, float* out_f32) {
  VT_CHECK(packed != nullptr, "ct nvfp4 dequant: packed weight is null");
  VT_CHECK(weight_scale_fp8 != nullptr, "ct nvfp4 dequant: weight_scale is null");
  VT_CHECK(out_f32 != nullptr, "ct nvfp4 dequant: output buffer is null");
  VT_CHECK(out_dim >= 0 && in_dim >= 0, "ct nvfp4 dequant: negative dimension");
  VT_CHECK(in_dim % kNvfp4GroupSize == 0,
           "ct nvfp4 dequant: in_dim must be a multiple of 16");

  // compressed-tensors stores the global scale as a divisor: dequant multiplies
  // by its reciprocal (compressed_tensors_w4a4_nvfp4.py:111-113).
  const float global_scale = Reciprocal(weight_global_scale_disk);
  const int64_t packed_cols = in_dim / 2;
  const int64_t groups = in_dim / kNvfp4GroupSize;

  for (int64_t o = 0; o < out_dim; ++o) {
    const uint8_t* packed_row = packed + o * packed_cols;
    const uint8_t* scale_row = weight_scale_fp8 + o * groups;
    float* out_row = out_f32 + o * in_dim;
    for (int64_t g = 0; g < groups; ++g) {
      const float group_scale = F8E4M3ToF32(scale_row[g]) * global_scale;
      const int64_t base_elem = g * kNvfp4GroupSize;
      for (int64_t j = 0; j < kNvfp4GroupSize / 2; ++j) {
        const uint8_t byte = packed_row[base_elem / 2 + j];
        const uint8_t low = byte & 0x0FU;
        const uint8_t high = byte >> 4;
        const float lo_val =
            kE2M1Lut[low & 0x7U] * ((low & 0x8U) ? -1.0F : 1.0F);
        const float hi_val =
            kE2M1Lut[high & 0x7U] * ((high & 0x8U) ? -1.0F : 1.0F);
        out_row[base_elem + 2 * j] = lo_val * group_scale;
        out_row[base_elem + 2 * j + 1] = hi_val * group_scale;
      }
    }
  }
}

void RunNvfp4Emulation(const float* x, int64_t m, int64_t in_dim,
                       const uint8_t* packed, const uint8_t* weight_scale_fp8,
                       float weight_global_scale_disk,
                       float input_global_scale_disk, int64_t out_dim,
                       float* out, int block_size) {
  VT_CHECK(out != nullptr, "nvfp4 emulation: out is null");
  // Activation round-trip (x_dq) into a scratch buffer.
  std::vector<float> x_dq(static_cast<size_t>(m) * static_cast<size_t>(in_dim));
  RefNvfp4QuantDequant(x, m, in_dim, input_global_scale_disk, x_dq.data(),
                       block_size);
  // Weight dequant (w_dq [out_dim, in_dim]).
  std::vector<float> w_dq(static_cast<size_t>(out_dim) *
                          static_cast<size_t>(in_dim));
  DequantCtNvfp4WeightToF32(packed, weight_scale_fp8, weight_global_scale_disk,
                            out_dim, in_dim, w_dq.data());
  // out = x_dq @ w_dq^T (f32 accumulation).
  for (int64_t i = 0; i < m; ++i) {
    const float* x_row = x_dq.data() + i * in_dim;
    float* out_row = out + i * out_dim;
    for (int64_t o = 0; o < out_dim; ++o) {
      const float* w_row = w_dq.data() + o * in_dim;
      float acc = 0.0F;
      for (int64_t k = 0; k < in_dim; ++k) acc += x_row[k] * w_row[k];
      out_row[o] = acc;
    }
  }
}

}  // namespace vllm
