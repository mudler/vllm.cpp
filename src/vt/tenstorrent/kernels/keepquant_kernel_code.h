// SPDX-License-Identifier: Apache-2.0
//
// KEEPQUANT W4b (#3031): the quantized-domain keep-quant dot and the Q8
// activation quantization for the TT int8-dot device kernel, as ONE source
// shared by the device kernel (compiled by tt-metal's RISC toolchain, where
// `float` is soft-float IEEE with the same `-ffp-contract=off` semantics the
// CPU ports compile with) and by the host-side review against the CPU ports
// it must mirror bit-exactly:
//
//   quantize_row_q8_K_ref  cpu_quant_act.cpp:88  (ggml-quants.c:2696)
//   quantize_row_q8_0_ref  cpu_quant_act.cpp:47  (ggml-quants.c:~478)
//   vec_dot_q4_K_q8_K      cpu_quant_dot.cpp:285 (quants.c:645)
//   vec_dot_q5_K_q8_K      cpu_quant_dot.cpp:367 (quants.c:720)
//   vec_dot_q6_K_q8_K      cpu_quant_dot.cpp:457 (quants.c:800)
//   vec_dot_q8_0_q8_0      cpu_quant_dot.cpp:~170 (quants.c:400)
//
// The accumulation ORDER inside every routine is the ported order, because
// the whole point of the lever is bit-exactness against it: per block 8
// int32 lanes (each <= 2^24 by upstream design, so exactly f32-
// representable), a per-lane f32 mul-then-add across blocks, the min
// correction interleaved per block, and the lane sum LAST in ascending lane
// order. The arithmetic is written in that order; the code claims no more
// than the build actually delivers.
//
// The bit-exactness is carried by ONE device build hazard: tt-metal compiles
// ALL device kernels with -ffast-math (tt_metal/hw/CMakeLists.txt:311), and
// its -freciprocal-math rewrites the q8_K quantizer's two divisions —
// `iscale = -127.0f / max` and the `1.0f / iscale` stored as the block's d —
// into algebraic reciprocal-multiply against a compile-time-rounded
// constant, which changes the y payload and with it every dot that consumes
// the row (legs A7/A14, ~1 ulp). The defense is provenance hiding: `iscale`
// is a `volatile float`, so its defining division is invisible to the
// algebraic passes and both divisions execute as written. The cross-check
// that pinned the mechanism: strict-y + ANY dot build = oracle bits, fast-y
// + ANY dot build = the 1-ulp-off bits — the dot's own build flags were
// irrelevant, and legA21 proved the dot-side volatile accumulators changed
// nothing bit-for-bit, so the dots carry none. The q8_0 divisions
// (amax/127.0f, 1.0f/d) stay as written: the q8_0 arm matches the oracle on
// device today; revisit only on evidence.
//
// All multi-byte loads/stores are explicit little-endian: the packed GGUF
// blocks, the staged i32 words, and the device DRAM/L1 are little-endian.

#pragma once

#include <cstdint>

// ---- little-endian loads --------------------------------------------------

static inline uint16_t kq_load16(const uint8_t* p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8));
}

static inline uint32_t kq_load32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

static inline void kq_store32(uint8_t* p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16);
  p[3] = static_cast<uint8_t>(v >> 24);
}

static inline void kq_store16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
}

// ---- exact f32 <-> f16 bit conversions -------------------------------------
// Exact IEEE conversions in integer arithmetic: the device soft-float has no
// conversion instructions to diverge from, and the host reviewer can compare
// against the hardware's cvt directly.

static inline float kq_f16_bits_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;  // +/-0
    } else {
      // Subnormal f16: value = man * 2^-24. Shift the MSB to bit 10, then
      // value = 1.m x 2^(-14-shifts), so the f32 exponent field is
      // (-14 - shifts) + 127 = 113 - shifts.
      uint32_t m = man;
      int32_t shifts = 0;
      while (!(m & 0x400u)) {
        m <<= 1;
        shifts++;
      }
      m &= 0x3FFu;
      bits = sign | (static_cast<uint32_t>(113 - shifts) << 23) | (m << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);  // inf / nan
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

// Round-to-nearest-even f32 -> f16. All input classes handled (the q8_0
// activation scale d = amax/127 can be tiny or huge).
static inline uint16_t kq_f32_to_f16_bits(float f) {
  uint32_t x;
  __builtin_memcpy(&x, &f, 4);
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  const uint32_t exp = (x >> 23) & 0xFFu;
  const uint32_t man = x & 0x7FFFFFu;
  if (exp == 0xFF) {  // inf / nan
    return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
  }
  int32_t e = static_cast<int32_t>(exp) - 127 + 15;  // unbiased+15
  if (e >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
  }
  if (e <= 0) {
    if (e < -10) {
      return sign;  // underflow -> +/-0
    }
    // Subnormal: keep the implicit bit, round to nearest even on the 13
    // dropped bits.
    const uint32_t m = man | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e);
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1u))) r++;
    return static_cast<uint16_t>(sign | r);
  }
  uint32_t r = man >> 13;
  const uint32_t rem = man & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;
  if (r == 0x400u) {  // mantissa carry
    r = 0;
    e++;
    if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | r);
}

// ---- NearestInt (cpu_quant_act.cpp:42) --------------------------------------
// The magic-constant round-half-to-even. One f32 add (RNE by the add's own
// rounding) then a bit extract; valid while |fval| <= 4194303 — the callers
// keep the values inside that window.
static inline int kq_nearest_int(float fval) {
  float val = fval + 12582912.0f;
  int32_t i;
  __builtin_memcpy(&i, &val, 4);
  return static_cast<int>((i & 0x007fffff) - 0x00400000);
}

// roundf (round-half-away) for the q8_0 quantizer. The quant values satisfy
// |x| <= ~128, where x +/- 0.5f is exact, so truncation of x+/-0.5f IS
// roundf; kept as the named helper so the domain assumption lives with the
// code.
static inline float kq_roundf(float x) {
  return (x >= 0.0f) ? static_cast<float>(static_cast<int32_t>(x + 0.5f))
                     : static_cast<float>(static_cast<int32_t>(x - 0.5f));
}

// ---- row byte geometry -----------------------------------------------------
// q8_K row: nb * 292B blocks {f32 d; i8 qs[256]; i16 bsums[16]}.
// q8_0 row: n32 * 34B blocks {f16 d; i8 qs[32]}.
static inline uint32_t kq_q8_k_row_bytes(uint32_t K) { return (K / 256) * 292u; }
static inline uint32_t kq_q8_0_row_bytes(uint32_t K) { return (K / 32) * 34u; }

// ---- activation quantizers --------------------------------------------------

// cpu_quant_act.cpp:88 quantize_row_q8_K. x is K little-endian f32 values;
// y receives kq_q8_k_row_bytes(K) bytes.
static inline void kq_quantize_row_q8_K(const uint8_t* x, uint8_t* y, uint32_t K) {
  const uint32_t nb = K / 256;
  for (uint32_t i = 0; i < nb; ++i, x += 256 * 4, y += 292) {
    const float* xf = reinterpret_cast<const float*>(x);
    float amax = 0.0f;
    float max = 0.0f;
    for (uint32_t j = 0; j < 256; ++j) {
      // Upstream's fabs via integer sign clear (bit-exact, no fabsf call).
      uint32_t xb;
      __builtin_memcpy(&xb, &xf[j], 4);
      const float av = __builtin_bit_cast(float, xb & 0x7FFFFFFFu);
      if (av > amax) {
        amax = av;
        max = xf[j];
      }
    }
    kq_store32(y + 0, 0);  // d
    if (amax == 0.0f) {
      for (uint32_t j = 0; j < 256; ++j) y[4 + j] = 0;
      for (uint32_t j = 0; j < 32; ++j) y[260 + j] = 0;
      continue;
    }
    // fast-math: volatile hides the division's provenance — without it
    // -freciprocal-math rewrote `1.0f / iscale` into `max * (1/-127.0f)`
    // (constant reciprocal, rounded at compile time) and folded iscale's
    // own rounding away in `iscale * xf[j]`; the payload drifted by ~1 ulp
    // and the dot followed (legs A7/A14/A21).
    volatile float iscale = -127.0f / max;
    for (uint32_t j = 0; j < 256; ++j) {
      volatile float xq = iscale * xf[j];  // fast-math: keep mul+round un-fused
      const int v = kq_nearest_int(xq);
      y[4 + j] = static_cast<uint8_t>(v < 127 ? v : 127);
    }
    for (uint32_t j = 0; j < 16; ++j) {
      int32_t sum = 0;
      for (uint32_t l = 0; l < 16; ++l) sum += static_cast<int8_t>(y[4 + j * 16 + l]);
      kq_store16(y + 260 + j * 2, static_cast<uint16_t>(static_cast<int16_t>(sum)));
    }
    kq_store32(y + 0, __builtin_bit_cast(uint32_t, 1.0f / iscale));
  }
}

// cpu_quant_act.cpp:47 quantize_row_q8_0. y receives kq_q8_0_row_bytes(K)
// bytes; d is stored f16 (the f16 rounding is part of the encoding — the
// vec_dot widens it back).
static inline void kq_quantize_row_q8_0(const uint8_t* x, uint8_t* y, uint32_t K) {
  const uint32_t nb = K / 32;
  for (uint32_t i = 0; i < nb; ++i, x += 32 * 4, y += 34) {
    const float* xf = reinterpret_cast<const float*>(x);
    float amax = 0.0f;
    for (uint32_t j = 0; j < 32; ++j) {
      uint32_t xb;
      __builtin_memcpy(&xb, &xf[j], 4);
      const float av = __builtin_bit_cast(float, xb & 0x7FFFFFFFu);
      amax = amax > av ? amax : av;
    }
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    kq_store16(y + 0, kq_f32_to_f16_bits(d));
    for (uint32_t j = 0; j < 32; ++j) {
      volatile float x0 = xf[j] * id;  // fast-math: keep mul+round un-fused
      y[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(kq_roundf(x0)));
    }
  }
}

// ---- the dots ---------------------------------------------------------------
// xblock points at ONE packed weight block in the staged i32-words byte form
// (the block's bytes are the little-endian byte lanes of its wpb words — see
// EnsureKeepQuantWords — with the same layout BlockQ4_K/BlockQ5_K/BlockQ6_K/
// BlockQ8_0 have host-side, padded to wpb*4); yrow points at the activation
// row's FIRST block; nb blocks are consumed from both, the x stride being
// block_word_bytes (wpb*4) and the y stride the activation block size.

// cpu_quant_dot.cpp:285 (quants.c:645). Q4_K block: {f16 d; f16 dmin;
// u8 scales[12]; u8 qs[128]} = 144B.
static inline float kq_vec_dot_q4_K_q8_K(const uint8_t* xblock,
                                         uint32_t block_word_bytes,
                                         const uint8_t* yrow, uint32_t nb) {
  float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float sumf = 0.0f;
  int8_t aux8[256];
  for (uint32_t i = 0; i < nb; ++i, xblock += block_word_bytes, yrow += 292) {
    const uint8_t* q4 = xblock + 16;
    const int8_t* q8 = reinterpret_cast<const int8_t*>(yrow + 4);
    const float yd = __builtin_bit_cast(float, kq_load32(yrow));
    for (uint32_t j = 0; j < 4; ++j) {
      for (uint32_t l = 0; l < 32; ++l)
        aux8[j * 64 + l] = static_cast<int8_t>(q4[j * 32 + l] & 0xF);
      for (uint32_t l = 0; l < 32; ++l)
        aux8[j * 64 + 32 + l] = static_cast<int8_t>(q4[j * 32 + l] >> 4);
    }
    uint32_t utmp[4];
    utmp[0] = kq_load32(xblock + 4);
    utmp[1] = kq_load32(xblock + 8);
    utmp[2] = kq_load32(xblock + 12);
    utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0fu) | (((utmp[1] >> 6) & 0x03030303u) << 4);
    const uint32_t uaux = utmp[1] & 0x3f3f3f3fu;
    utmp[1] = (utmp[2] & 0x0f0f0f0fu) | (((utmp[0] >> 6) & 0x03030303u) << 4);
    utmp[2] = uaux;
    utmp[0] &= 0x3f3f3f3fu;
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
    const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
    int32_t sumi = 0;
    for (uint32_t j = 0; j < 16; ++j) {
      int16_t bs;
      __builtin_memcpy(&bs, yrow + 260 + j * 2, 2);
      sumi += bs * mins[j / 2];
    }
    int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t* a = aux8;
    const int8_t* q8p = q8;
    uint32_t is = 0;
    for (uint32_t j = 0; j < 8; ++j) {
      const int32_t scale = static_cast<int32_t>(scales[is++]);
      for (uint32_t rep = 0; rep < 4; ++rep) {
        for (uint32_t l = 0; l < 8; ++l) {
          const int16_t p = static_cast<int16_t>(q8p[l] * a[l]);
          aux32[l] += scale * p;
        }
        q8p += 8;
        a += 8;
      }
    }
    const float d = kq_f16_bits_to_f32(kq_load16(xblock)) * yd;
    for (uint32_t l = 0; l < 8; ++l) {
      // mul-then-add as written (the ported order; see the file header).
      const float prod = d * static_cast<float>(aux32[l]);
      sums[l] = sums[l] + prod;
    }
    const float dmin = kq_f16_bits_to_f32(kq_load16(xblock + 2)) * yd;
    const float corr = dmin * static_cast<float>(sumi);
    sumf = sumf - corr;
  }
  float s = sumf;
  for (uint32_t l = 0; l < 8; ++l) s = s + sums[l];
  return s;
}

// cpu_quant_dot.cpp:367 (quants.c:720). Q5_K block: {f16 d; f16 dmin;
// u8 scales[12]; u8 qh[32]; u8 qs[128]} = 176B.
static inline float kq_vec_dot_q5_K_q8_K(const uint8_t* xblock,
                                         uint32_t block_word_bytes,
                                         const uint8_t* yrow, uint32_t nb) {
  float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float sumf = 0.0f;
  int8_t aux8[256];
  for (uint32_t i = 0; i < nb; ++i, xblock += block_word_bytes, yrow += 292) {
    const uint8_t* q4 = xblock + 48;
    const uint8_t* hm = xblock + 16;
    const int8_t* q8 = reinterpret_cast<const int8_t*>(yrow + 4);
    const float yd = __builtin_bit_cast(float, kq_load32(yrow));
    uint8_t m = 1;
    for (uint32_t j = 0; j < 4; ++j) {
      for (uint32_t l = 0; l < 32; ++l)
        aux8[j * 64 + l] = static_cast<int8_t>(q4[l] & 0xF);
      for (uint32_t l = 0; l < 32; ++l)
        aux8[j * 64 + l] =
            static_cast<int8_t>(aux8[j * 64 + l] + ((hm[l] & m) ? 16 : 0));
      m = static_cast<uint8_t>(m << 1);
      for (uint32_t l = 0; l < 32; ++l)
        aux8[j * 64 + 32 + l] = static_cast<int8_t>(q4[l] >> 4);
      for (uint32_t l = 0; l < 32; ++l)
        aux8[j * 64 + 32 + l] =
            static_cast<int8_t>(aux8[j * 64 + 32 + l] + ((hm[l] & m) ? 16 : 0));
      m = static_cast<uint8_t>(m << 1);
      q4 += 32;
    }
    uint32_t utmp[4];
    utmp[0] = kq_load32(xblock + 4);
    utmp[1] = kq_load32(xblock + 8);
    utmp[2] = kq_load32(xblock + 12);
    utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0fu) | (((utmp[1] >> 6) & 0x03030303u) << 4);
    const uint32_t uaux = utmp[1] & 0x3f3f3f3fu;
    utmp[1] = (utmp[2] & 0x0f0f0f0fu) | (((utmp[0] >> 6) & 0x03030303u) << 4);
    utmp[2] = uaux;
    utmp[0] &= 0x3f3f3f3fu;
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
    const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
    int32_t sumi = 0;
    for (uint32_t j = 0; j < 16; ++j) {
      int16_t bs;
      __builtin_memcpy(&bs, yrow + 260 + j * 2, 2);
      sumi += bs * mins[j / 2];
    }
    int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t* a = aux8;
    const int8_t* q8p = q8;
    uint32_t is = 0;
    for (uint32_t j = 0; j < 8; ++j) {
      const int32_t scale = static_cast<int32_t>(scales[is++]);
      for (uint32_t rep = 0; rep < 4; ++rep) {
        for (uint32_t l = 0; l < 8; ++l) {
          const int16_t p = static_cast<int16_t>(q8p[l] * a[l]);
          aux32[l] += scale * p;
        }
        q8p += 8;
        a += 8;
      }
    }
    const float d = kq_f16_bits_to_f32(kq_load16(xblock)) * yd;
    for (uint32_t l = 0; l < 8; ++l) {
      // mul-then-add as written (the ported order; see the file header).
      const float prod = d * static_cast<float>(aux32[l]);
      sums[l] = sums[l] + prod;
    }
    const float dmin = kq_f16_bits_to_f32(kq_load16(xblock + 2)) * yd;
    const float corr = dmin * static_cast<float>(sumi);
    sumf = sumf - corr;
  }
  float s = sumf;
  for (uint32_t l = 0; l < 8; ++l) s = s + sums[l];
  return s;
}

// cpu_quant_dot.cpp:457 (quants.c:800). Q6_K block: {u8 ql[128]; u8 qh[64];
// i8 scales[16]; f16 d} = 210B. No min term; the -32 bias is baked in.
static inline float kq_vec_dot_q6_K_q8_K(const uint8_t* xblock,
                                         uint32_t block_word_bytes,
                                         const uint8_t* yrow, uint32_t nb) {
  float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float sumf = 0.0f;
  int8_t aux8[256];
  for (uint32_t i = 0; i < nb; ++i, xblock += block_word_bytes, yrow += 292) {
    const uint8_t* q4 = xblock;
    const uint8_t* qh = xblock + 128;
    const int8_t* q8 = reinterpret_cast<const int8_t*>(yrow + 4);
    const float yd = __builtin_bit_cast(float, kq_load32(yrow));
    for (uint32_t j = 0; j < 256; j += 128) {
      for (uint32_t l = 0; l < 32; ++l) {
        aux8[j + l + 0] = static_cast<int8_t>(
            static_cast<int8_t>((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
        aux8[j + l + 32] = static_cast<int8_t>(
            static_cast<int8_t>((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
        aux8[j + l + 64] = static_cast<int8_t>(
            static_cast<int8_t>((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
        aux8[j + l + 96] = static_cast<int8_t>(
            static_cast<int8_t>((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
      }
      q4 += 64;
      qh += 32;
    }
    int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t* a = aux8;
    const int8_t* q8p = q8;
    for (uint32_t j = 0; j < 16; ++j) {
      const int32_t scale = static_cast<int32_t>(static_cast<int8_t>(xblock[192 + j]));
      for (uint32_t rep = 0; rep < 2; ++rep) {
        for (uint32_t l = 0; l < 8; ++l) {
          const int16_t p = static_cast<int16_t>(q8p[l] * a[l]);
          aux32[l] += scale * p;
        }
        q8p += 8;
        a += 8;
      }
    }
    const float d = kq_f16_bits_to_f32(kq_load16(xblock + 208)) * yd;
    for (uint32_t l = 0; l < 8; ++l) {
      const float prod = d * static_cast<float>(aux32[l]);
      sums[l] = sums[l] + prod;
    }
  }
  float s = sumf;
  for (uint32_t l = 0; l < 8; ++l) s = s + sums[l];
  return s;
}

// cpu_quant_dot.cpp ~170 (quants.c:400). Q8_0 x Q8_0: {f16 d; i8 qs[32]}
// = 34B activation blocks; sumi per block, then ONE f32 mul-add per block —
// note the CPU's parenthesization: sumi * (dx * dy), the d product FIRST.
static inline float kq_vec_dot_q8_0_q8_0(const uint8_t* xblock,
                                         uint32_t block_word_bytes,
                                         const uint8_t* yrow, uint32_t nb) {
  float sumf = 0.0f;
  for (uint32_t i = 0; i < nb; ++i, xblock += block_word_bytes, yrow += 34) {
    int32_t sumi = 0;
    for (uint32_t j = 0; j < 32; ++j) {
      sumi += static_cast<int8_t>(xblock[2 + j]) * static_cast<int8_t>(yrow[2 + j]);
    }
    const float dx = kq_f16_bits_to_f32(kq_load16(xblock));
    const float dy = kq_f16_bits_to_f32(kq_load16(yrow));
    const float prod = static_cast<float>(sumi) * (dx * dy);
    sumf = sumf + prod;
  }
  return sumf;
}
