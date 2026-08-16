#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <iterator>
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

// --- F16 (1) / BF16 (30): unquantized rows in a MIXED file. Not an academic
// case: `Qwen3.5-2B-UD-Q8_K_XL.gguf` stores 56 of its 335 tensors as f16,
// 1.615 GiB of them including the tied token_embd/lm_head, and those are the
// tensors no block encoding can cover — see the CIQ G4 result. ---
TEST_CASE("DequantGgufRowToF32 F16 row") {
  const float src[6] = {1.5F, -2.25F, 0.0F, 100.0F, 0.00006103515625F, -1.0F};
  std::vector<uint8_t> b;
  for (float f : src) PushF16(b, f);
  const std::vector<float> out = DequantGgufRowToF32(1, b.data(), 6);
  REQUIRE(out.size() == 6);
  // Every value above is exactly representable in IEEE half (the fifth is the
  // smallest normal), so the decode must be EXACT, not approximate.
  for (int i = 0; i < 6; ++i) CHECK(out[i] == src[i]);
  // Unquantized rows have block_elems == 1, so unlike the block encodings there
  // is no "whole number of blocks" precondition to violate: ANY length decodes.
  CHECK(DequantGgufRowToF32(1, b.data(), 5).size() == 5);
}

TEST_CASE("DequantGgufRowToF32 BF16 row") {
  const float src[4] = {1.5F, -2.25F, 0.0F, 100.0F};
  std::vector<uint8_t> b;
  for (float f : src) {
    const uint16_t h = vt::F32ToBF16(f);
    b.push_back(static_cast<uint8_t>(h & 0xff));
    b.push_back(static_cast<uint8_t>(h >> 8));
  }
  const std::vector<float> out = DequantGgufRowToF32(30, b.data(), 4);
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

// --- Q2_K (10): block = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin; }
// (84 bytes). 2-bit quant (qs, shift 0/2/4/6) times a per-16 4-bit sub-scale
// (low nibble of scales[]) minus a 4-bit sub-min (high nibble), both scaled by
// f16 d / dmin. Output order: is 0..15, each covering 16 elements; is=0,1 at
// shift0 (grpA q[0..15], grpB q[16..31]), is=2,3 at shift2, ... then q += 32
// for the second half (n=128, is 8..15). d=0.5, dmin=0.25.
//   scales[0]=0x14 -> sc=4,min=1 (dl=2.0, ml=0.25)
//   scales[1]=0x23 -> sc=3,min=2 (dl=1.5, ml=0.5)
//   scales[2]=0x31 -> sc=1,min=3 (dl=0.5, ml=0.75)
//   qs[0]=0x1B -> shift0 field=3, shift2 field=2 ; qs[16]=0x02 -> shift0 field=2
TEST_CASE("DequantGgufRowToF32 Q2_K synthetic block (nibble sub-scale/min)") {
  std::vector<uint8_t> b(84, 0);
  b[0] = 0x14;   // scales[0]: sc=4, min=1
  b[1] = 0x23;   // scales[1]: sc=3, min=2
  b[2] = 0x31;   // scales[2]: sc=1, min=3
  b[16] = 0x1B;  // qs[0]
  b[32] = 0x02;  // qs[16]
  const uint16_t d = vt::F32ToF16(0.5F);
  const uint16_t dmin = vt::F32ToF16(0.25F);
  b[80] = static_cast<uint8_t>(d & 0xFF);
  b[81] = static_cast<uint8_t>(d >> 8);
  b[82] = static_cast<uint8_t>(dmin & 0xFF);
  b[83] = static_cast<uint8_t>(dmin >> 8);

  const std::vector<float> out = DequantGgufRowToF32(10, b.data(), 256);
  REQUIRE(out.size() == 256);
  // y[0]  : is=0, q[0]=0x1B shift0&3=3 ; 2.0*3 - 0.25 = 5.75
  CHECK(out[0] == doctest::Approx(5.75F));
  // y[1]  : is=0, q[1]=0 ; 2.0*0 - 0.25 = -0.25
  CHECK(out[1] == doctest::Approx(-0.25F));
  // y[16] : is=1, q[16]=0x02 shift0&3=2 ; 1.5*2 - 0.5 = 2.5
  CHECK(out[16] == doctest::Approx(2.5F));
  // y[32] : is=2, q[0]=0x1B shift2 -> (0x1B>>2)&3=2 ; 0.5*2 - 0.75 = 0.25
  CHECK(out[32] == doctest::Approx(0.25F));
}

// --- IQ2_XXS (16): block = { f16 d; u16 qs[32]; } (66 bytes). Codebook decode:
// each 32-elem sub-block reads two u32 from qs -- aux32[0] = four 8-bit grid
// indices, aux32[1] = four 7-bit sign selectors + a 4-bit scale in the top
// nibble (db = d*(0.5 + (aux32[1]>>28))*0.25). Each grid index picks 8 bytes of
// iq2xxs_grid; ksigns_iq2xs[selector] & kmask_iq2xs[j] flips the sign.
//
// Hand-verified against ggml-common.h iq2xxs_grid @ 237ad9b96:
//   grid[0] = 0x0808080808080808 -> all 8 grid bytes.
//   grid[1] = 0x080808080808082b -> byte0 = 0x2b (43), bytes1..7 = 8.
//   ksigns_iq2xs[0] = 0 (no flips) ; ksigns_iq2xs[1] = 129 = 0b10000001
//     (flips lane j=0 via kmask 1 and j=7 via kmask 128).
TEST_CASE("DequantGgufRowToF32 IQ2_XXS codebook block (grid + signs + scale)") {
  std::vector<uint8_t> b(66, 0);
  const uint16_t d = vt::F32ToF16(1.0F);
  b[0] = static_cast<uint8_t>(d & 0xFF);
  b[1] = static_cast<uint8_t>(d >> 8);
  // qs bytes start at offset 2. ib32=0: aux32[0]=0x00000001 (grid idx 1 for
  // lane l=0, idx 0 for l=1..3), aux32[1]=0 (scale sel 0 -> db=0.125, no signs).
  b[2] = 0x01;  // aux32[0] low byte = 1
  // ib32=1: aux32[0]=0 (all grid idx 0), aux32[1]=0x10000001
  //   (>>28 = 1 -> db=0.375 ; bits0-6 = 1 -> l=0 sign selector = 1).
  // aux32[1] for ib32=1 sits at qs offset 8+4 = 12 -> block offset 2+12 = 14.
  b[14] = 0x01;  // aux32[1] byte0
  b[17] = 0x10;  // aux32[1] byte3 (top nibble = 1)

  const std::vector<float> out = DequantGgufRowToF32(16, b.data(), 256);
  REQUIRE(out.size() == 256);
  // ib32=0, l=0 (grid idx 1): db=0.125. y[0]=0.125*43=5.375, y[1..7]=0.125*8=1.0
  CHECK(out[0] == doctest::Approx(5.375F));
  for (int j = 1; j < 8; ++j) CHECK(out[j] == doctest::Approx(1.0F));
  // ib32=0, l=1..3 (grid idx 0, no signs): all 1.0
  for (int j = 8; j < 32; ++j) CHECK(out[j] == doctest::Approx(1.0F));
  // ib32=1, l=0 (grid idx 0, db=0.375, signs=129 flips j=0 and j=7):
  //   y[32]=-3.0, y[33..38]=+3.0, y[39]=-3.0
  CHECK(out[32] == doctest::Approx(-3.0F));
  for (int j = 33; j < 39; ++j) CHECK(out[j] == doctest::Approx(3.0F));
  CHECK(out[39] == doctest::Approx(-3.0F));
  // ib32=1, l=1..3 (no signs): all +3.0
  for (int j = 40; j < 64; ++j) CHECK(out[j] == doctest::Approx(3.0F));
  // ib32=2..7 (all-zero -> grid idx 0, db=0.125, no signs): all 1.0
  for (int j = 64; j < 256; ++j) CHECK(out[j] == doctest::Approx(1.0F));

  // bf16 variant round-trips through the same f32 decode.
  const std::vector<uint16_t> bf = DequantGgufRowToBf16(16, b.data(), 256);
  REQUIRE(bf.size() == 256);
  CHECK(bf[0] == vt::F32ToBF16(5.375F));
  CHECK(bf[32] == vt::F32ToBF16(-3.0F));
}

// --- IQ2_S (22): block = { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8]; }
// (82 bytes). Codebook decode over 8 sub-blocks of 32. Each 32-lane sub-block
// has 4 lanes; lane l reads a 10-bit grid index = qs[l] | ((qh[ib32]<<(8-2l))
// & 0x300), looks up iq2s_grid (1024 x u64 = 8 grid bytes), and applies the
// DIRECT sign byte signs[l] (= qs+32) per kmask_iq2xs — NO ksigns lookup.
// scales[ib32] packs two 4-bit ls: low nibble -> db[0] (l=0,1), high -> db[1]
// (l=2,3); db = d*(0.5 + ls)*0.25.
//
// Hand-verified against ggml-common.h iq2s_grid @ 237ad9b96:
//   grid[0]   = 0x0808080808080808 -> all 8 bytes = 8.
//   grid[1]   = 0x080808080808082b -> byte0 = 0x2b (43), bytes1..7 = 8.
//   grid[2]   = 0x0808080808081919 -> bytes0,1 = 0x19 (25), bytes2..7 = 8.
//   grid[256] = 0x0819081919191919 -> bytes = {25,25,25,25,25,8,25,8}.
// kmask_iq2xs[j] = 1<<j; sign byte 129 = 0b10000001 flips lanes j=0 and j=7.
TEST_CASE("DequantGgufRowToF32 IQ2_S codebook block (grid + qh + signs + scale)") {
  std::vector<uint8_t> b(82, 0);
  const uint16_t d = vt::F32ToF16(1.0F);
  b[0] = static_cast<uint8_t>(d & 0xFF);
  b[1] = static_cast<uint8_t>(d >> 8);
  // qs field @2: grid-low bytes qs[0..31] then sign bytes qs[32..63].
  b[2] = 1;      // ib32=0 lane0 grid idx low = 1  -> grid[1]
  b[4] = 2;      // ib32=0 lane2 grid idx low = 2  -> grid[2]
  b[6] = 1;      // ib32=1 lane0 grid idx low = 1  -> grid[1]
  b[38] = 0x81;  // ib32=1 lane0 sign byte = 129 (flips j=0, j=7)
  // qh field @66: 2 high index bits per lane.
  b[68] = 0x01;  // qh[2] bit0 -> ib32=2 lane0 idx high = 0x100 -> grid[256]
  // scales field @74. scales[0]: low nib 0 -> db[0]=0.125, high nib 1 -> db[1]=0.375.
  b[74] = 0x10;  // scales[0]; scales[1..7] stay 0 -> db[0]=db[1]=0.125.

  const std::vector<float> out = DequantGgufRowToF32(22, b.data(), 256);
  REQUIRE(out.size() == 256);
  // ib32=0 l0 (grid[1], db[0]=0.125): y[0]=0.125*43=5.375, y[1..7]=1.0.
  CHECK(out[0] == doctest::Approx(5.375F));
  CHECK(out[1] == doctest::Approx(1.0F));
  CHECK(out[7] == doctest::Approx(1.0F));
  // ib32=0 l1 (grid[0]): y[8..15]=1.0.
  CHECK(out[8] == doctest::Approx(1.0F));
  // ib32=0 l2 (grid[2], db[1]=0.375): y[16]=y[17]=0.375*25=9.375, y[18..23]=3.0.
  CHECK(out[16] == doctest::Approx(9.375F));
  CHECK(out[17] == doctest::Approx(9.375F));
  CHECK(out[18] == doctest::Approx(3.0F));
  // ib32=0 l3 (grid[0], db[1]): y[24..31]=0.375*8=3.0.
  CHECK(out[24] == doctest::Approx(3.0F));
  // ib32=1 l0 (grid[1], db[0]=0.125, sign 129 flips j=0 and j=7):
  //   y[32]=-5.375, y[33..38]=1.0, y[39]=-1.0.
  CHECK(out[32] == doctest::Approx(-5.375F));
  CHECK(out[33] == doctest::Approx(1.0F));
  CHECK(out[39] == doctest::Approx(-1.0F));
  // ib32=1 l1..3 (grid[0], no signs): all 1.0.
  CHECK(out[40] == doctest::Approx(1.0F));
  CHECK(out[63] == doctest::Approx(1.0F));
  // ib32=2 l0 (qh high bit -> grid[256], db[0]=0.125): y[64]=0.125*25=3.125,
  //   y[69]=0.125*8=1.0 (grid[256] byte5 = 8, distinguishes idx 256 from idx 0).
  CHECK(out[64] == doctest::Approx(3.125F));
  CHECK(out[69] == doctest::Approx(1.0F));
  // ib32=2 l1..3 (qh bits for other lanes are 0 -> grid[0]): 1.0.
  CHECK(out[72] == doctest::Approx(1.0F));
  // ib32=3..7 (all-zero -> grid[0], db=0.125): all 1.0.
  CHECK(out[96] == doctest::Approx(1.0F));
  CHECK(out[255] == doctest::Approx(1.0F));

  // bf16 variant round-trips through the same f32 decode.
  const std::vector<uint16_t> bf = DequantGgufRowToBf16(22, b.data(), 256);
  REQUIRE(bf.size() == 256);
  CHECK(bf[0] == vt::F32ToBF16(5.375F));
  CHECK(bf[32] == vt::F32ToBF16(-5.375F));
}

// --- MXFP4 (39): block = { u8 e; u8 qs[16]; } (17 bytes). OCP micro-scaling
// fp4: d = E8M0ToF32Half(e) = 2^(e-128); each of the 32 e2m1 nibbles is looked
// up in kvalues_mxfp4 = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}. Split-half
// packing: element j in the low nibble of qs[j], j+16 in the high nibble.
//
// Hand cases: e=0x80 -> 2^0 = 1.0 ; e=0x81 -> 2^1 = 2.0.
TEST_CASE("DequantGgufRowToF32 MXFP4 micro-scaling block (e8m0 + kvalues)") {
  std::vector<uint8_t> b(34, 0);  // two 17-byte blocks
  // Block A: e = 0x80 (scale 1.0).
  b[0] = 0x80;
  b[1] = 0x17;  // qs[0]: low nib 7 -> kvalues[7]=12 (y[0]); high nib 1 -> 1 (y[16])
  b[2] = 0x9F;  // qs[1]: low nib 15 -> kvalues[15]=-12 (y[1]); high nib 9 -> -1 (y[17])
  // Block B (offset 17): e = 0x81 (scale 2.0).
  b[17] = 0x81;
  b[18] = 0x06;  // qs[0]: low nib 6 -> kvalues[6]=8 (y[32]); high nib 0 -> 0 (y[48])

  const std::vector<float> out = DequantGgufRowToF32(39, b.data(), 64);
  REQUIRE(out.size() == 64);
  // Block A, scale 1.0.
  CHECK(out[0] == doctest::Approx(12.0F));
  CHECK(out[16] == doctest::Approx(1.0F));
  CHECK(out[1] == doctest::Approx(-12.0F));
  CHECK(out[17] == doctest::Approx(-1.0F));
  for (int j = 2; j < 16; ++j) {
    CHECK(out[j] == doctest::Approx(0.0F));
    CHECK(out[j + 16] == doctest::Approx(0.0F));
  }
  // Block B, scale 2.0: y[32] = 8*2 = 16, y[48] = 0.
  CHECK(out[32] == doctest::Approx(16.0F));
  CHECK(out[48] == doctest::Approx(0.0F));

  const std::vector<uint16_t> bf = DequantGgufRowToBf16(39, b.data(), 64);
  REQUIRE(bf.size() == 64);
  CHECK(bf[0] == vt::F32ToBF16(12.0F));
  CHECK(bf[32] == vt::F32ToBF16(16.0F));
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
  // IQ2_S (22) and MXFP4 (39) are now decodable (UD-IQ2_M vehicle); their golden
  // cases are above. IQ4_XS (23) remains tabulated in the reader but has no
  // decoder yet, so it must still fail loudly rather than silently mis-decode.
  std::vector<uint8_t> b2(136, 0);
  CHECK_THROWS_AS(DequantGgufRowToF32(23, b2.data(), 256), std::runtime_error);
}

// --- IQ1_S (19) / IQ1_XXXS (66): the two encodings the Qwen3.8-2.4T-A95B
// checkpoints are 96.92 % made of. `vt::cpu::BlockToFloat` has decoded both
// since #946, but this expand-to-bf16 path did not list them, so it threw
// "unsupported ggml type". That is not a corner: `RouteGgufTensor` sends a
// tensor here whenever the VT_CPU_REF oracle switch is on, keep-quant is off,
// K is ragged, or the role is not a verbatim one, so the reference lane could
// not load the target checkpoint at all.
//
// Gated on the ORACLE-produced goldens rather than on "does not throw": the
// inputs are real checkpoint bytes and the expected values come from the pinned
// upstream and fork ggml themselves. Provenance in
// tests/vt/iq1_golden_vectors.h.
#include "../vt/iq1_golden_vectors.h"

namespace {

void CheckGgufDequantAgainstOracle(uint32_t ggml_type, const uint8_t* blocks,
                                   const uint32_t* golden_bits, size_t n) {
  const std::vector<float> out =
      DequantGgufRowToF32(ggml_type, blocks, static_cast<int64_t>(n));
  REQUIRE(out.size() == n);
  for (size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    uint32_t bits = 0;
    std::memcpy(&bits, &out[i], sizeof(bits));
    CHECK(bits == golden_bits[i]);
  }
}

}  // namespace

TEST_CASE("DequantGgufRowToF32 IQ1_S row matches the pinned oracle") {
  CheckGgufDequantAgainstOracle(19, vllm_test::kIq1sGoldenBlocks,
                                vllm_test::kIq1sGoldenBits,
                                std::size(vllm_test::kIq1sGoldenBits));
}

TEST_CASE("DequantGgufRowToF32 IQ1_XXXS row matches the pinned FORK oracle") {
  CheckGgufDequantAgainstOracle(66, vllm_test::kIq1xxxsGoldenBlocks,
                                vllm_test::kIq1xxxsGoldenBits,
                                std::size(vllm_test::kIq1xxxsGoldenBits));
}

TEST_CASE("DequantGgufRowToF32 IQ3_XXS row matches the pinned oracle") {
  // Same omission, found by the same review, in the same shared branch, so it
  // is fixed in the same flow rather than filed and deferred (issue #1023).
  // IQ3_XXS is the `ffn_down` encoding of the DeepSeek-V4 UD-IQ2_XXS vehicle,
  // and `vt::cpu::BlockToFloat` has decoded it since that port; only this
  // expansion path never listed it.
  CheckGgufDequantAgainstOracle(18, vllm_test::kIq3xxsGoldenBlocks,
                                vllm_test::kIq3xxsGoldenBits,
                                std::size(vllm_test::kIq3xxsGoldenBits));
}
