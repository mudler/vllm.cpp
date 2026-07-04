#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vt/dtype.h"

using vllm::DequantGgufRowToBf16;
using vllm::DequantGgufRowToF32;

namespace {

// Append a little-endian f16 (from an f16-exact f32) to a byte vector.
void PushF16(std::vector<uint8_t>& b, float f) {
  const uint16_t h = vt::F32ToF16(f);
  b.push_back(static_cast<uint8_t>(h & 0xFF));
  b.push_back(static_cast<uint8_t>(h >> 8));
}

}  // namespace

// --- F32 (0): raw memcpy passthrough. ---
TEST_CASE("DequantGgufRowToF32 F32 passthrough") {
  const float src[4] = {1.5F, -2.25F, 0.0F, 100.0F};
  std::vector<uint8_t> bytes(sizeof(src));
  std::memcpy(bytes.data(), src, sizeof(src));
  const std::vector<float> out = DequantGgufRowToF32(0, bytes.data(), 4);
  REQUIRE(out.size() == 4);
  for (int i = 0; i < 4; ++i) CHECK(out[i] == src[i]);
}

// --- Q8_0 (8): block = { f16 d; i8 qs[32]; } (34 bytes). y = d * qs[i]. ---
TEST_CASE("DequantGgufRowToF32 Q8_0 synthetic block") {
  std::vector<uint8_t> b;
  PushF16(b, 0.5F);  // d
  const int8_t qs[32] = {10, -4, 0, 127, -128, 2, 3, 4, 5, 6, 7,  8,
                         9,  10, 11, 12, 13,  14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25,  26, 27, 28};
  for (int8_t q : qs) b.push_back(static_cast<uint8_t>(q));
  REQUIRE(b.size() == 34);

  const std::vector<float> out = DequantGgufRowToF32(8, b.data(), 32);
  REQUIRE(out.size() == 32);
  for (int i = 0; i < 32; ++i) CHECK(out[i] == doctest::Approx(0.5F * qs[i]));
}

// --- Q4_0 (2): block = { f16 d; u8 qs[16]; } (18 bytes).
// y[j]=(nibble_lo-8)*d, y[j+16]=(nibble_hi-8)*d. byte 0x6A -> lo=10, hi=6. ---
TEST_CASE("DequantGgufRowToF32 Q4_0 synthetic block") {
  std::vector<uint8_t> b;
  PushF16(b, 0.5F);  // d
  b.push_back(0x6A);  // element 0: lo nibble 10, element 16: hi nibble 6
  for (int i = 1; i < 16; ++i) b.push_back(0x88);  // lo=8,hi=8 -> zero
  REQUIRE(b.size() == 18);

  const std::vector<float> out = DequantGgufRowToF32(2, b.data(), 32);
  REQUIRE(out.size() == 32);
  CHECK(out[0] == doctest::Approx((10 - 8) * 0.5F));   // 1.0
  CHECK(out[16] == doctest::Approx((6 - 8) * 0.5F));   // -1.0
  for (int i = 1; i < 16; ++i) {
    CHECK(out[i] == doctest::Approx(0.0F));
    CHECK(out[i + 16] == doctest::Approx(0.0F));
  }
}

// --- Q4_K (12): block = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; }
// (144 bytes). 8 sub-blocks of 32; y = d*sc*(nibble) - dmin*m.
//
// Scale packing (inverse of get_scale_min_k4) chosen so:
//   sub-block scales sc = {2,3,4,5, 18,3,4,5}, all mins m = 1.
//   scales[0..7] hold sc/m for sb0..3 in low 6 bits; scales[0] carries a top-2
//   bit (0x42) that lifts sb4's sc to 18 (exercises the j>=4 packing branch).
//   scales[8..11] = 0x1{2,3,4,5}: low nibble = sb4..7 sc, high nibble = m=1.
// Every qs byte = 0x21 -> low nibble 1, high nibble 2. d=0.5, dmin=0.25. ---
TEST_CASE("DequantGgufRowToF32 Q4_K synthetic block (6-bit scale unpack)") {
  std::vector<uint8_t> b;
  PushF16(b, 0.5F);   // d
  PushF16(b, 0.25F);  // dmin
  const uint8_t scales[12] = {0x42, 0x03, 0x04, 0x05, 0x01, 0x01,
                              0x01, 0x01, 0x12, 0x13, 0x14, 0x15};
  for (uint8_t s : scales) b.push_back(s);
  for (int i = 0; i < 128; ++i) b.push_back(0x21);
  REQUIRE(b.size() == 144);

  const std::vector<float> out = DequantGgufRowToF32(12, b.data(), 256);
  REQUIRE(out.size() == 256);
  // sc per sub-block {2,3,4,5,18,3,4,5}, m=1; low nibble=1, high nibble=2.
  CHECK(out[0] == doctest::Approx(0.5F * 2 * 1 - 0.25F));    // sb0 lo -> 0.75
  CHECK(out[32] == doctest::Approx(0.5F * 3 * 2 - 0.25F));   // sb1 hi -> 2.75
  CHECK(out[64] == doctest::Approx(0.5F * 4 * 1 - 0.25F));   // sb2 lo -> 1.75
  CHECK(out[96] == doctest::Approx(0.5F * 5 * 2 - 0.25F));   // sb3 hi -> 4.75
  CHECK(out[128] == doctest::Approx(0.5F * 18 * 1 - 0.25F)); // sb4 lo -> 8.75
  CHECK(out[160] == doctest::Approx(0.5F * 3 * 2 - 0.25F));  // sb5 hi -> 2.75
  CHECK(out[192] == doctest::Approx(0.5F * 4 * 1 - 0.25F));  // sb6 lo -> 1.75
  CHECK(out[224] == doctest::Approx(0.5F * 5 * 2 - 0.25F));  // sb7 hi -> 4.75
}

// --- Q5_K (13): block = { f16 d; f16 dmin; u8 scales[12]; u8 qh[32];
// u8 qs[128]; } (176 bytes). Like Q4_K plus a 5th (high) bit from qh:
// bit u1 (=1<<2g) for the low nibbles, u2 (=2<<2g) for the high nibbles.
// Reuse the Q4_K scale packing (sc {2,3,4,5,18,3,4,5}, m=1). qs byte 0x21,
// qh byte 0x01 -> only group-0 low nibbles (u1=1) get +16; all others don't. ---
TEST_CASE("DequantGgufRowToF32 Q5_K synthetic block (high-bit plane)") {
  std::vector<uint8_t> b;
  PushF16(b, 0.5F);   // d
  PushF16(b, 0.25F);  // dmin
  const uint8_t scales[12] = {0x42, 0x03, 0x04, 0x05, 0x01, 0x01,
                              0x01, 0x01, 0x12, 0x13, 0x14, 0x15};
  for (uint8_t s : scales) b.push_back(s);
  for (int i = 0; i < 32; ++i) b.push_back(0x01);   // qh: bit0 set only
  for (int i = 0; i < 128; ++i) b.push_back(0x21);  // qs: lo=1, hi=2
  REQUIRE(b.size() == 176);

  const std::vector<float> out = DequantGgufRowToF32(13, b.data(), 256);
  REQUIRE(out.size() == 256);
  // group0 (u1=1): qh bit0 set -> low nibbles +16.
  CHECK(out[0] == doctest::Approx(0.5F * 2 * (1 + 16) - 0.25F));  // sb0 lo 16.75
  // group0 high (u2=2): qh bit1 clear -> no +16.
  CHECK(out[32] == doctest::Approx(0.5F * 3 * 2 - 0.25F));        // sb1 hi 2.75
  // group1 (u1=4): qh bit2 clear -> no +16.
  CHECK(out[64] == doctest::Approx(0.5F * 4 * 1 - 0.25F));        // sb2 lo 1.75
  CHECK(out[96] == doctest::Approx(0.5F * 5 * 2 - 0.25F));        // sb3 hi 4.75
  // group2 (u1=16): clear -> no +16, sc=18.
  CHECK(out[128] == doctest::Approx(0.5F * 18 * 1 - 0.25F));      // sb4 lo 8.75
  CHECK(out[224] == doctest::Approx(0.5F * 5 * 2 - 0.25F));       // sb7 hi 4.75
}

// --- Q6_K (14): block = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; }
// (210 bytes). 6-bit quant = 4 low bits (ql) + 2 high bits (qh), -32 biased,
// times an int8 scale. All scales=4, d=0.5. ql[0]=0x35 (lo=5,hi=3), ql[32]=0,
// qh[0]=0xE4 -> the four 2-bit high fields are {0,1,2,3}. ---
TEST_CASE("DequantGgufRowToF32 Q6_K synthetic block") {
  std::vector<uint8_t> b(210, 0);
  b[0] = 0x35;               // ql[0]: lo nibble 5, hi nibble 3
  // ql[32] stays 0
  b[128] = 0xE4;             // qh[0]: >>0&3=0, >>2&3=1, >>4&3=2, >>6&3=3
  for (int i = 0; i < 16; ++i) b[192 + i] = 4;  // scales i8 = 4
  const uint16_t d = vt::F32ToF16(0.5F);
  b[208] = static_cast<uint8_t>(d & 0xFF);
  b[209] = static_cast<uint8_t>(d >> 8);

  const std::vector<float> out = DequantGgufRowToF32(14, b.data(), 256);
  REQUIRE(out.size() == 256);
  // q1 = (5 | (0<<4)) - 32 = -27 ; y[0]  = 0.5*4*-27 = -54
  CHECK(out[0] == doctest::Approx(0.5F * 4 * -27));
  // q2 = (0 | (1<<4)) - 32 = -16 ; y[32] = 0.5*4*-16 = -32
  CHECK(out[32] == doctest::Approx(0.5F * 4 * -16));
  // q3 = (3 | (2<<4)) - 32 = 3   ; y[64] = 0.5*4*3   = 6
  CHECK(out[64] == doctest::Approx(0.5F * 4 * 3));
  // q4 = (0 | (3<<4)) - 32 = 16  ; y[96] = 0.5*4*16  = 32
  CHECK(out[96] == doctest::Approx(0.5F * 4 * 16));
}

// --- Q3_K (11): block = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d; }
// (110 bytes). 3-bit quant = 2 low bits (qs, shift 0/2/4/6) + 1 high bit
// (hmask; ABSENT -> -4) times the 6-bit scale (-32 biased).
//
// scales[0..7]=0x22, scales[8..11]=0xAA packs all 16 unpacked scales to 34
// (=> scales-32 = 2). d=0.5 -> dl = 1.0.
//   qs[0]=0x0B -> shift0 field=3, shift2 field=2; hmask[0]=0xFF (high bit set).
//   qs[1]=0x02 -> shift0 field=2; hmask[1]=0x00 (high bit absent -> -4). ---
TEST_CASE("DequantGgufRowToF32 Q3_K synthetic block (scale shuffle + hi bit)") {
  std::vector<uint8_t> b(110, 0);
  b[0] = 0xFF;  // hmask[0]: high bit present for all sub-block masks
  // hmask[1] stays 0 -> high bit absent
  b[32] = 0x0B;  // qs[0]: bits0-1=3, bits2-3=2
  b[33] = 0x02;  // qs[1]: bits0-1=2
  for (int i = 0; i < 8; ++i) b[96 + i] = 0x22;    // scales[0..7]
  for (int i = 8; i < 12; ++i) b[96 + i] = 0xAA;   // scales[8..11]
  const uint16_t d = vt::F32ToF16(0.5F);
  b[108] = static_cast<uint8_t>(d & 0xFF);
  b[109] = static_cast<uint8_t>(d >> 8);

  const std::vector<float> out = DequantGgufRowToF32(11, b.data(), 256);
  REQUIRE(out.size() == 256);
  // dl = 0.5*(34-32) = 1.0.
  // y[0] : j=0 shift0, q=3, hmask set -> -0 ; 1.0*(3-0) = 3.0
  CHECK(out[0] == doctest::Approx(3.0F));
  // y[1] : j=0 shift0, q=2, hmask absent -> -4 ; 1.0*(2-4) = -2.0
  CHECK(out[1] == doctest::Approx(-2.0F));
  // y[32]: j=1 shift2 on qs[0]=3->(0x0B>>2)&3=2, m=2 & hmask[0]=set -> -0 ; 2.0
  CHECK(out[32] == doctest::Approx(2.0F));
}

// --- Multi-block row: two Q8_0 blocks (64 elems) with distinct scales. ---
TEST_CASE("DequantGgufRowToF32 Q8_0 multi-block row") {
  std::vector<uint8_t> b;
  PushF16(b, 0.5F);
  for (int i = 0; i < 32; ++i) b.push_back(static_cast<uint8_t>(i));  // 0..31
  PushF16(b, 2.0F);
  for (int i = 0; i < 32; ++i)
    b.push_back(static_cast<uint8_t>(static_cast<int8_t>(-i)));  // 0..-31
  REQUIRE(b.size() == 68);

  const std::vector<float> out = DequantGgufRowToF32(8, b.data(), 64);
  REQUIRE(out.size() == 64);
  for (int i = 0; i < 32; ++i) CHECK(out[i] == doctest::Approx(0.5F * i));
  for (int i = 0; i < 32; ++i) CHECK(out[32 + i] == doctest::Approx(2.0F * -i));
}

// --- bf16 variant: dequant to f32 then round-to-nearest-even bf16. ---
TEST_CASE("DequantGgufRowToBf16 matches f32 path rounded") {
  std::vector<uint8_t> b;
  PushF16(b, 0.5F);
  const int8_t qs[32] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                         23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  for (int8_t q : qs) b.push_back(static_cast<uint8_t>(q));
  const std::vector<uint16_t> out = DequantGgufRowToBf16(8, b.data(), 32);
  REQUIRE(out.size() == 32);
  for (int i = 0; i < 32; ++i)
    CHECK(out[i] == vt::F32ToBF16(0.5F * qs[i]));
}

// --- Guards: non-block-multiple numel throws; unsupported i-quant throws. ---
TEST_CASE("DequantGgufRowToF32 rejects non-block-multiple numel") {
  std::vector<uint8_t> b(34, 0);
  CHECK_THROWS_AS(DequantGgufRowToF32(8, b.data(), 31), std::runtime_error);
}

TEST_CASE("DequantGgufRowToF32 rejects unsupported i-quant type") {
  std::vector<uint8_t> b(82, 0);
  // IQ2_S (22): tabulated in the reader (256 elems) but not dequant-able yet.
  CHECK_THROWS_AS(DequantGgufRowToF32(22, b.data(), 256), std::runtime_error);
  // IQ4_XS (23) likewise.
  std::vector<uint8_t> b2(136, 0);
  CHECK_THROWS_AS(DequantGgufRowToF32(23, b2.data(), 256), std::runtime_error);
}
