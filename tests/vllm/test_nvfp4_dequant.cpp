#include <doctest/doctest.h>

#include <algorithm>
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

// --- Rounding + multiply-order gate: C++ uses the same-order f32 arithmetic as
// torch (fp8*ws2 computed first, then fp4*group_scale); only weight_scale_2
// carries >4 significant bits, so C++ matches torch by construction and the
// weight_scale_2 round is the ONLY source of bf16 rounding here. Force a
// rounding with ws2 = 1.1f.
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

// --- The NIBBLE ORDER, which is a PRODUCER convention and not a fact about
// NVFP4. .agents/specs/nvfp4-nibble-order.md.
//
// The default is LOW-first: element 2j in the low nibble. That is torchao's
// `pack_uint4` (torchao/prototype/mx_formats/kernels.py:160,
// `uint8_data[::2] | uint8_data[1::2] << 4`) and what vLLM's own reader assumes
// (`break_fp4_bytes`, nvfp4_emulation_utils.py:321-324). Lightricks'
// `nvfp4-prequant` packs the other way (ltx-kernels/docs/NVFP4.md:27-29), and the
// H3 community converter does too (minimax_h3.h:1500-1503).
//
// WHY THIS NEEDS A GATE AND NOT A COMMENT: reading a file with the wrong order
// transposes every adjacent pair. Magnitudes, per-group extrema and every
// summary statistic survive intact — only the POSITIONS move — so nothing
// downstream can notice. The final CHECK states that blindness outright.
TEST_CASE("DequantNvfp4ToBf16 nibble order is selectable and defaults to low-first") {
  // One group. Element 0 = +0.5 (0x1) low, element 1 = +6.0 (0x7) high.
  std::vector<uint8_t> packed(8, 0x00);
  packed[0] = 0x71;
  std::vector<uint8_t> scale = {0x38};  // fp8-e4m3 = 1.0
  const float ws2 = 1.0F;

  std::vector<uint16_t> lo(16, 0xFFFF), hi(16, 0xFFFF), dflt(16, 0xFFFF);
  DequantNvfp4ToBf16(packed.data(), scale.data(), ws2, 1, 16, dflt.data());
  DequantNvfp4ToBf16(packed.data(), scale.data(), ws2, 1, 16, lo.data(),
                     vllm::Nvfp4NibbleOrder::kLowFirst);
  DequantNvfp4ToBf16(packed.data(), scale.data(), ws2, 1, 16, hi.data(),
                     vllm::Nvfp4NibbleOrder::kHighFirst);

  // The DEFAULT must be low-first, bit for bit. This is what makes "every caller
  // predating the parameter is unchanged" true by construction rather than by
  // inspection of call sites.
  CHECK(dflt == lo);

  // Low-first: element 0 is the LOW nibble (0x1 = +0.5).
  CHECK(vt::BF16ToF32(lo[0]) == doctest::Approx(0.5F));
  CHECK(vt::BF16ToF32(lo[1]) == doctest::Approx(6.0F));
  // High-first: element 0 is the HIGH nibble (0x7 = +6.0). Exactly transposed.
  CHECK(vt::BF16ToF32(hi[0]) == doctest::Approx(6.0F));
  CHECK(vt::BF16ToF32(hi[1]) == doctest::Approx(0.5F));

  // The two differ ONLY by a pairwise swap, and every element is still present.
  for (int j = 0; j < 8; ++j) {
    CHECK(lo[2 * j] == hi[2 * j + 1]);
    CHECK(lo[2 * j + 1] == hi[2 * j]);
  }

  // THE BLINDNESS, stated as a gated fact: the two readings are the same multiset,
  // so no magnitude summary can tell them apart. Anything checking "is the output
  // finite / correctly scaled / the right absmax" passes under BOTH.
  std::vector<uint16_t> lo_sorted = lo, hi_sorted = hi;
  std::sort(lo_sorted.begin(), lo_sorted.end());
  std::sort(hi_sorted.begin(), hi_sorted.end());
  CHECK(lo_sorted == hi_sorted);
}
