#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/dtype.h"

using vllm::DequantNvfp4ToBf16;
using vllm::F8E4M3ToF32;

// --- F8E4M3ToF32: IEEE fp8-e4m3fn (1 sign | 4 exp | 3 mant, bias 7, no inf,
// NaN=0x7F/0xFF). Bit patterns hand-decoded below; matches
// torch.Tensor.view(torch.float8_e4m3fn).to(torch.float32). ---
TEST_CASE("F8E4M3ToF32 known bit patterns") {
  // 0x00 = 0 0000 000 -> +0
  CHECK(F8E4M3ToF32(0x00) == 0.0F);
  // 0x38 = 0 0111 000 -> exp 7 (bias 7 => 2^0), mant 0 => 1.0
  CHECK(F8E4M3ToF32(0x38) == doctest::Approx(1.0F));
  // 0x40 = 0 1000 000 -> exp 8 => 2^1, mant 0 => 2.0
  CHECK(F8E4M3ToF32(0x40) == doctest::Approx(2.0F));
  // 0x3A = 0 0111 010 -> 2^0 * (1 + 2/8) = 1.25
  CHECK(F8E4M3ToF32(0x3A) == doctest::Approx(1.25F));
  // 0x3C = 0 0111 100 -> 2^0 * (1 + 4/8) = 1.5
  CHECK(F8E4M3ToF32(0x3C) == doctest::Approx(1.5F));
  // 0xB8 = 1 0111 000 -> -1.0 (sign bit set)
  CHECK(F8E4M3ToF32(0xB8) == doctest::Approx(-1.0F));
  // 0x7E = 0 1111 110 -> 2^8 * (1 + 6/8) = 256 * 1.75 = 448 (E4M3FN max)
  CHECK(F8E4M3ToF32(0x7E) == doctest::Approx(448.0F));
  // 0x04 = 0 0000 100 -> subnormal: 4 * 2^-9 = 0.0078125
  CHECK(F8E4M3ToF32(0x04) == doctest::Approx(0.0078125F));
  // 0x01 = 0 0000 001 -> subnormal: 1 * 2^-9 = 0.001953125
  CHECK(F8E4M3ToF32(0x01) == doctest::Approx(0.001953125F));
  // 0x7F and 0xFF are the only NaN encodings.
  CHECK(std::isnan(F8E4M3ToF32(0x7F)));
  CHECK(std::isnan(F8E4M3ToF32(0xFF)));
}

// --- DequantNvfp4ToBf16: one row, one 16-element group (8 packed bytes, one
// fp8 scale byte). Hand-computed per moe-semantics.md Sec 8.
//
// E2M1 nibble = sign(bit3) | magnitude(bits0-2); LUT[0..7]=
//   {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Packing: element 2i = low nibble of byte i,
// element 2i+1 = high nibble.
//
// group_scale = F8E4M3ToF32(scale) * weight_scale_2
//             = 1.25 (0x3A) * 2.0 = 2.5
// out[i] = bf16( e2m1(nibble_i) * 2.5 )    (all products bf16-exact here)
TEST_CASE("DequantNvfp4ToBf16 hand-computed block") {
  // elem: 0=+0.5(0x1) 1=+6.0(0x7) 2=-6.0(0xF) 3=0.0(0x0)
  //       4=+1.0(0x2) 5=-1.0(0xA) 6..15=0.0
  std::vector<uint8_t> packed = {
      0x71,  // byte0: high=0x7(+6.0) low=0x1(+0.5)
      0x0F,  // byte1: high=0x0(0.0)  low=0xF(-6.0)
      0xA2,  // byte2: high=0xA(-1.0) low=0x2(+1.0)
      0x00, 0x00, 0x00, 0x00, 0x00,
  };
  std::vector<uint8_t> scale = {0x3A};  // fp8-e4m3 = 1.25
  const float ws2 = 2.0F;               // weight_scale_2

  std::vector<uint16_t> out(16, 0xFFFF);
  DequantNvfp4ToBf16(packed.data(), scale.data(), ws2, /*out_dim=*/1,
                     /*in_dim=*/16, out.data());

  const float expected[16] = {
      1.25F,   // 0.5 * 2.5
      15.0F,   // 6.0 * 2.5
      -15.0F,  // -6.0 * 2.5
      0.0F,    // 0
      2.5F,    // 1.0 * 2.5
      -2.5F,   // -1.0 * 2.5
      0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
  };
  for (int i = 0; i < 16; ++i) {
    CHECK(vt::BF16ToF32(out[i]) == doctest::Approx(expected[i]));
  }
}

// --- Rounding + multiply-order gate: the ONLY source of bf16 rounding in this
// dequant is weight_scale_2 (an fp4*fp8 product never exceeds 8 significant
// bits, so it is always bf16-exact). Force a rounding with ws2 = 1.1f.
//
// fp4 = 1.0 (nibble 0x2), scale = 0x38 (=1.0), ws2 = 1.1f = 0x3F8CCCCD.
// f32 product = 1.0 * (1.0 * 1.1f) = 0x3F8CCCCD.
// bf16 round-to-nearest-even: 0x3F8CCCCD + 0x7FFF = 0x3F8D4CCC, >>16 = 0x3F8D.
// BF16ToF32(0x3F8D) = 1.1015625.
TEST_CASE("DequantNvfp4ToBf16 weight_scale_2 rounding") {
  std::vector<uint8_t> packed(8, 0x00);
  packed[0] = 0x02;  // element 0 = +1.0, rest 0
  std::vector<uint8_t> scale = {0x38};  // 1.0
  const float ws2 = 1.1F;

  std::vector<uint16_t> out(16, 0xFFFF);
  DequantNvfp4ToBf16(packed.data(), scale.data(), ws2, 1, 16, out.data());

  CHECK(out[0] == 0x3F8D);
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(1.1015625F));
  for (int i = 1; i < 16; ++i) {
    CHECK(out[i] == 0x0000);  // bf16 +0
  }
}

// --- Multi-row, multi-group: exercises row/group offset arithmetic. Row 1
// uses a distinct scale so a row-stride bug would surface. ---
TEST_CASE("DequantNvfp4ToBf16 two rows two groups") {
  // in_dim = 32 -> 16 packed bytes/row, 2 fp8 scales/row.
  const int64_t in_dim = 32;
  const int64_t packed_cols = in_dim / 2;  // 16
  const int64_t groups = in_dim / 16;      // 2

  std::vector<uint8_t> packed(2 * packed_cols, 0x00);
  // Row 0, group 0, elem 0 = +2.0 (nibble 0x4).
  packed[0] = 0x04;
  // Row 0, group 1, elem 16 = +1.0 (nibble 0x2) -> byte index 8.
  packed[8] = 0x02;
  // Row 1, group 0, elem 0 = +4.0 (nibble 0x6) -> byte index packed_cols.
  packed[packed_cols + 0] = 0x06;

  std::vector<uint8_t> scale(2 * groups, 0x00);
  scale[0] = 0x38;  // row0 g0 = 1.0
  scale[1] = 0x40;  // row0 g1 = 2.0
  scale[2] = 0x3C;  // row1 g0 = 1.5
  scale[3] = 0x38;  // row1 g1 = 1.0
  const float ws2 = 2.0F;

  std::vector<uint16_t> out(2 * in_dim, 0xFFFF);
  DequantNvfp4ToBf16(packed.data(), scale.data(), ws2, 2, in_dim, out.data());

  // Row 0 elem 0: 2.0 * (1.0 * 2.0) = 4.0
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(4.0F));
  // Row 0 elem 16: 1.0 * (2.0 * 2.0) = 4.0
  CHECK(vt::BF16ToF32(out[16]) == doctest::Approx(4.0F));
  // Row 1 elem 0: 4.0 * (1.5 * 2.0) = 12.0
  CHECK(vt::BF16ToF32(out[in_dim + 0]) == doctest::Approx(12.0F));
}
