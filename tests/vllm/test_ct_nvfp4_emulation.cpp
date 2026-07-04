// CPU unit tests for the compressed-tensors NVFP4 W4A4 emulation reference
// (nvfp4_emulation.h). All cases are hand-computed against the pinned upstream
// nvfp4_emulation_utils.py math @ e24d1b24 — no GPU, no oracle.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"

using vllm::CastToFp4;
using vllm::DequantCtNvfp4WeightToF32;
using vllm::F32ToF8E4M3;
using vllm::F8E4M3ToF32;
using vllm::RefNvfp4QuantDequant;
using vllm::RunNvfp4Emulation;

// --- F32ToF8E4M3: inverse of F8E4M3ToF32, round-to-nearest-even, saturating. ---
TEST_CASE("F32ToF8E4M3 known values and round-trip") {
  CHECK(F32ToF8E4M3(0.0F) == 0x00);
  CHECK(F32ToF8E4M3(1.0F) == 0x38);      // 2^0 * 1.0
  CHECK(F32ToF8E4M3(2.0F) == 0x40);      // 2^1 * 1.0
  CHECK(F32ToF8E4M3(1.25F) == 0x3A);     // 2^0 * 1.25
  CHECK(F32ToF8E4M3(1.5F) == 0x3C);      // 2^0 * 1.5
  CHECK(F32ToF8E4M3(0.5F) == 0x30);      // 2^-1
  CHECK(F32ToF8E4M3(-1.0F) == 0xB8);     // sign bit
  CHECK(F32ToF8E4M3(448.0F) == 0x7E);    // E4M3FN max
  CHECK(F32ToF8E4M3(500.0F) == 0x7E);    // saturates (no inf)
  CHECK(F32ToF8E4M3(0.001953125F) == 0x01);   // subnormal 1 * 2^-9
  CHECK(F32ToF8E4M3(0.0078125F) == 0x04);      // subnormal 4 * 2^-9
  CHECK(F32ToF8E4M3(-0.0078125F) == 0x84);
  CHECK(F32ToF8E4M3(std::nanf("")) == 0x7F);   // only NaN encoding (positive)

  // Round-trip for every exactly-representable value used above.
  for (uint8_t byte : {0x00, 0x38, 0x40, 0x3A, 0x3C, 0x30, 0xB8, 0x7E, 0x01,
                       0x04, 0x84}) {
    const float v = F8E4M3ToF32(byte);
    CHECK(F32ToF8E4M3(v) == byte);
  }
}

// --- CastToFp4: the fixed half-open bucket boundaries (lines 413-424). ---
TEST_CASE("CastToFp4 buckets") {
  CHECK(CastToFp4(0.0F) == 0.0F);
  CHECK(CastToFp4(0.1F) == 0.0F);     // [0, 0.25]
  CHECK(CastToFp4(0.25F) == 0.0F);    // boundary -> 0
  CHECK(CastToFp4(0.3F) == 0.5F);     // (0.25, 0.75)
  CHECK(CastToFp4(0.75F) == 1.0F);    // boundary -> 1.0
  CHECK(CastToFp4(1.25F) == 1.0F);    // boundary -> 1.0
  CHECK(CastToFp4(1.5F) == 1.5F);
  CHECK(CastToFp4(1.75F) == 2.0F);    // boundary -> 2.0 (not 1.5)
  CHECK(CastToFp4(2.5F) == 2.0F);     // boundary -> 2.0
  CHECK(CastToFp4(2.7F) == 3.0F);
  CHECK(CastToFp4(3.5F) == 4.0F);     // boundary -> 4.0
  CHECK(CastToFp4(5.0F) == 4.0F);     // boundary -> 4.0
  CHECK(CastToFp4(5.5F) == 6.0F);
  CHECK(CastToFp4(-2.7F) == -3.0F);   // sign preserved
  CHECK(CastToFp4(-6.0F) == -6.0F);
}

// --- CT weight dequant: same E2M1 x fp8-group hand case as the modelopt test,
// but the CT global scale is a DIVISOR (0.5 -> effective multiplier 1/0.5=2.0),
// and the output is f32 (no bf16 round). ---
TEST_CASE("DequantCtNvfp4WeightToF32 hand-computed block") {
  // elem: 0=+0.5(0x1) 1=+6.0(0x7) 2=-6.0(0xF) 3=0.0(0x0)
  //       4=+1.0(0x2) 5=-1.0(0xA) 6..15=0.0
  std::vector<uint8_t> packed = {0x71, 0x0F, 0xA2, 0x00,
                                 0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> scale = {0x3A};              // fp8-e4m3 = 1.25
  const float wgs_disk = 0.5F;                      // on-disk divisor
  // group_scale = 1.25 * (1/0.5) = 2.5
  std::vector<float> out(16, -999.0F);
  DequantCtNvfp4WeightToF32(packed.data(), scale.data(), wgs_disk, 1, 16,
                            out.data());
  const float expected[16] = {1.25F, 15.0F, -15.0F, 0.0F, 2.5F, -2.5F,
                              0.0F,  0.0F,  0.0F,    0.0F, 0.0F, 0.0F,
                              0.0F,  0.0F,  0.0F,    0.0F};
  for (int i = 0; i < 16; ++i) CHECK(out[i] == doctest::Approx(expected[i]));
}

// --- Activation round-trip: all-exact-fp4 block with global_scale = 1.0 and
// block max = 6.0 makes the per-block scale exactly 1.0, so x_dq == x. ---
TEST_CASE("RefNvfp4QuantDequant identity on exact-fp4 block") {
  const float x[16] = {6.0F,  -6.0F, 0.0F,  0.5F, 1.0F,  1.5F,  2.0F,  3.0F,
                       4.0F,  -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, 0.0F};
  std::vector<float> dq(16, -999.0F);
  RefNvfp4QuantDequant(x, /*m=*/1, /*n=*/16, /*input_global_scale=*/1.0F,
                       dq.data());
  for (int i = 0; i < 16; ++i) CHECK(dq[i] == doctest::Approx(x[i]));
}

// --- Activation round-trip with a non-unit block scale + rounding. Block max
// 3.0, global_scale 1.0 -> scale = 0.5 (fp8-exact), block_scale = 0.5,
// output_scale = 2.0. Exact-fp4/2 inputs survive; others snap. ---
TEST_CASE("RefNvfp4QuantDequant scaled block with rounding") {
  const float x[16] = {3.0F, 1.5F, 0.75F, 0.4F, 0.0F, -3.0F, -1.5F, -0.75F,
                       0.0F, 0.0F, 0.0F,  0.0F, 0.0F, 0.0F,  0.0F,  0.0F};
  std::vector<float> dq(16, -999.0F);
  RefNvfp4QuantDequant(x, 1, 16, 1.0F, dq.data());
  // scaled = x*2: 3->6 (fp4 6) ->3.0; 1.5->3 ->1.5; 0.75->1.5 ->0.75;
  // 0.4->0.8 -> CastToFp4=1.0 -> *0.5 = 0.5 (rounding up).
  CHECK(dq[0] == doctest::Approx(3.0F));
  CHECK(dq[1] == doctest::Approx(1.5F));
  CHECK(dq[2] == doctest::Approx(0.75F));
  CHECK(dq[3] == doctest::Approx(0.5F));
  CHECK(dq[4] == doctest::Approx(0.0F));
  CHECK(dq[5] == doctest::Approx(-3.0F));
  CHECK(dq[6] == doctest::Approx(-1.5F));
  CHECK(dq[7] == doctest::Approx(-0.75F));
}

// --- Full emulated W4A4 linear: m=1, in=16, out=2. Weight rows dequant to all
// 1.0 (row0) and all 2.0 (row1); x_dq is the exact-fp4 identity block above.
// out[0] = sum(x)*1, out[1] = sum(x)*2. ---
TEST_CASE("RunNvfp4Emulation small matmul") {
  const float x[16] = {6.0F,  -6.0F, 0.0F,  0.5F, 1.0F,  1.5F,  2.0F,  3.0F,
                       4.0F,  -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, 0.0F};
  float sum = 0.0F;
  for (float v : x) sum += v;  // = 2.5

  // in_dim=16 -> 8 packed bytes/row, 1 fp8 scale/row. nibble 0x2 = +1.0.
  std::vector<uint8_t> packed(2 * 8, 0x22);  // every element = +1.0
  std::vector<uint8_t> wscale = {0x38, 0x38};  // both group scales = 1.0
  // Row0 global divisor 1.0 -> mult 1.0 -> w=1.0. Row1 divisor 0.5 -> mult 2.0.
  // (One weight_global_scale per output partition; emulate per-row by two calls
  // would differ — instead keep a single shared divisor and scale via wscale.)
  // Use wscale row1 = 0x40 (2.0) to get row1 weights = 2.0 with divisor 1.0.
  wscale[1] = 0x40;
  const float wgs_disk = 1.0F;   // effective weight multiplier 1.0
  const float igs_disk = 1.0F;   // activation global scale 1.0 (identity block)

  std::vector<float> out(2, -999.0F);
  RunNvfp4Emulation(x, 1, 16, packed.data(), wscale.data(), wgs_disk, igs_disk,
                    2, out.data());
  CHECK(out[0] == doctest::Approx(sum));         // row0 weights all 1.0
  CHECK(out[1] == doctest::Approx(sum * 2.0F));  // row1 weights all 2.0
}
