// Block-quant `to_float` decoders — the `dequantize_row_*` kernels ported
// byte-for-byte from llama.cpp @ 237ad9b96:
//   ggml/src/ggml-common.h  (block_q2_K/q4_0/q8_0/q3_K/q4_K/q5_K/q6_K +
//                            block_iq2_xxs layouts; iq2xxs_grid/ksigns tables)
//   ggml/src/ggml-quants.c  (dequantize_row_q4_0:401, _q8_0:495, _q2_K:903,
//                            _q3_K:1247, _q4_K:1471, _q5_K:1673, _q6_K:1881,
//                            _iq2_xxs:2416, get_scale_min_k4:822)
//
// These moved here VERBATIM from the GGUF loader
// (src/vllm/model_executor/model_loader/gguf_dequant.cpp), which now delegates:
// vt:: is the lower layer and the compute-in-quant GEMM's generic fallback
// needs the same decode, so a single implementation serves both the loader
// oracle and `vt::MatmulBTQuant`. Numerics are unchanged by construction (same
// code, same order, same `-ffp-contract=off` pinning) and
// tests/vllm/test_gguf_dequant.cpp gates that.
#include <cstring>

#include "cpu_quant_iq_tables.h"  // kIq2xxsGrid/kIq3xxsGrid/kKsignsIq2xs/kKmaskIq2xs
#include "vt/quant.h"
#include "vt/dtype.h"

namespace vt::cpu {
namespace {

// Read a little-endian ggml_half (f16) at byte pointer `p` and widen to f32.
// (Aligned load is not guaranteed for mmap'd block bytes, so memcpy.)
float ReadF16(const uint8_t* p) {
  uint16_t h = 0;
  std::memcpy(&h, p, sizeof(h));
  return vt::F16ToF32(h);
}

// get_scale_min_k4 (ggml-quants.c:822): unpack the j-th 6-bit scale `d` and
// 6-bit min `m` from a Q4_K/Q5_K block's packed scales[12]. j in 0..7.
void GetScaleMinK4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

// --- Per-type dequant (one full row = nb blocks). Each mirrors the matching
// dequantize_row_* in ggml-quants.c; byte offsets follow the ggml-common.h
// struct layouts. `y` is written in order (numel outputs). ---

// block_q4_0 = { f16 d; u8 qs[16]; }  (18 bytes)   dequantize_row_q4_0:401
void DequantQ4_0(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 32;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 18;
    const float d = ReadF16(blk);
    const uint8_t* qs = blk + 2;
    for (int j = 0; j < qk / 2; ++j) {
      const int x0 = (qs[j] & 0x0F) - 8;
      const int x1 = (qs[j] >> 4) - 8;
      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

// block_q8_0 = { f16 d; i8 qs[32]; }  (34 bytes)   dequantize_row_q8_0:495
void DequantQ8_0(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 32;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 34;
    const float d = ReadF16(blk);
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = qs[j] * d;
    }
  }
}

// block_q3_K = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d; } (110 bytes)
// dequantize_row_q3_K:1247. The 3-bit quant = 2 low bits (qs) + 1 high bit
// (hmask, inverted: absent bit -> -4) times the 6-bit scale (-32 biased).
void DequantQ3_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 110;
    const uint8_t* hm = blk;         // hmask[32]
    const uint8_t* q = blk + 32;     // qs[64]
    const uint8_t* sc_raw = blk + 96;  // scales[12]
    const float d_all = ReadF16(blk + 108);

    // Scale unpack: 12 packed bytes -> 16 6-bit scales in int8 view of aux.
    uint32_t aux[4];
    std::memcpy(aux, sc_raw, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < qk; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * (static_cast<int8_t>((q[l + 0] >> shift) & 3) -
                       ((hm[l + 0] & m) ? 0 : 4));
        }
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * (static_cast<int8_t>((q[l + 16] >> shift) & 3) -
                       ((hm[l + 16] & m) ? 0 : 4));
        }
        shift += 2;
        m = static_cast<uint8_t>(m << 1);
      }
      q += 32;
    }
  }
}

// block_q4_K = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; } (144 bytes)
// dequantize_row_q4_K:1471. y = d*sc*(nibble) - dmin*m over 8 sub-blocks of 32.
void DequantQ4_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 144;
    const float d = ReadF16(blk);
    const float min = ReadF16(blk + 2);
    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;  // qs[128]

    int is = 0;
    uint8_t sc = 0;
    uint8_t mm = 0;
    for (int j = 0; j < qk; j += 64) {
      GetScaleMinK4(is + 0, scales, &sc, &mm);
      const float d1 = d * sc;
      const float m1 = min * mm;
      GetScaleMinK4(is + 1, scales, &sc, &mm);
      const float d2 = d * sc;
      const float m2 = min * mm;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

// block_q5_K = { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }
// (176 bytes) dequantize_row_q5_K:1673. Like Q4_K plus the 5th (high) bit from
// qh: bit u1 for the low nibbles, u2 for the high nibbles (both <<=2 per pair).
void DequantQ5_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 176;
    const float d = ReadF16(blk);
    const float min = ReadF16(blk + 2);
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;   // qh[32]
    const uint8_t* ql = blk + 48;   // qs[128]

    int is = 0;
    uint8_t sc = 0;
    uint8_t mm = 0;
    uint8_t u1 = 1;
    uint8_t u2 = 2;
    for (int j = 0; j < qk; j += 64) {
      GetScaleMinK4(is + 0, scales, &sc, &mm);
      const float d1 = d * sc;
      const float m1 = min * mm;
      GetScaleMinK4(is + 1, scales, &sc, &mm);
      const float d2 = d * sc;
      const float m2 = min * mm;
      for (int l = 0; l < 32; ++l)
        *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l)
        *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 = static_cast<uint8_t>(u1 << 2);
      u2 = static_cast<uint8_t>(u2 << 2);
    }
  }
}

// block_q6_K = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; } (210 bytes)
// dequantize_row_q6_K:1881. 6-bit quant = 4 low bits (ql) + 2 high bits (qh),
// -32 biased, times an 8-bit (int8) scale. 16 blocks of 16.
void DequantQ6_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 210;
    const uint8_t* ql = blk;         // ql[128]
    const uint8_t* qh = blk + 128;   // qh[64]
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);  // scales[16]
    const float d = ReadF16(blk + 208);

    for (int n = 0; n < qk; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int8_t q1 = static_cast<int8_t>(
            (ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = static_cast<int8_t>(
            (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = static_cast<int8_t>(
            (ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = static_cast<int8_t>(
            (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

// block_q8_K = { f32 d; i8 qs[256]; i16 bsums[16]; } (292 bytes)
// dequantize_row_q8_K (ggml-quants.c). Q8_K is the K-quant ACTIVATION type; it
// never appears in a GGUF file, but the decoder completes the table and lets
// the activation-quant round trip be unit-tested in G2.
void DequantQ8_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 292;
    float d;
    std::memcpy(&d, blk, sizeof(d));
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
    for (int j = 0; j < qk; ++j) *y++ = d * qs[j];
  }
}

// block_q2_K = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin; } (84 bytes)
// dequantize_row_q2_K:903. 2-bit quant (qs, shift 0/2/4/6) times a 4-bit
// per-16 sub-scale (low nibble of scales[]) minus a 4-bit sub-min (high
// nibble), both scaled by the f16 super-block d / dmin.
void DequantQ2_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 84;
    const uint8_t* scales = blk;       // scales[16]
    const uint8_t* q = blk + 16;       // qs[64]
    const float d = ReadF16(blk + 80);
    const float min = ReadF16(blk + 82);

    int is = 0;
    for (int n = 0; n < qk; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = scales[is++];
        float dl = d * (sc & 0xF);
        float ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * static_cast<int8_t>((q[l] >> shift) & 3) - ml;
        sc = scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * static_cast<int8_t>((q[l + 16] >> shift) & 3) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}

// block_iq2_xxs = { f16 d; u16 qs[32]; } (66 bytes) dequantize_row_iq2_xxs:2416.
// Codebook decode: each 32-element sub-block reads two u32 from qs -- aux32[0]
// holds four 8-bit grid indices, aux32[1] holds four 7-bit sign selectors plus a
// 4-bit block scale in its top nibble (db = d*(0.5 + (aux32[1]>>28))*0.25). The
// eight grid bytes are looked up in kIq2xxsGrid and sign-flipped per
// kKsignsIq2xs & kKmaskIq2xs.
void DequantIQ2_XXS(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 66;
    const float d = ReadF16(blk);
    const uint8_t* qs = blk + 2;       // u16 qs[32], little-endian bytes
    for (int ib32 = 0; ib32 < qk / 32; ++ib32) {
      uint32_t aux32[2];
      std::memcpy(aux32, qs + 8 * ib32, 2 * sizeof(uint32_t));
      const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
      const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t* grid =
            reinterpret_cast<const uint8_t*>(kIq2xxsGrid + aux8[l]);
        const uint8_t signs = kKsignsIq2xs[(aux32[1] >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j)
          y[j] = db * grid[j] * ((signs & kKmaskIq2xs[j]) ? -1.f : 1.f);
        y += 8;
      }
    }
  }
}

// block_iq3_xxs = { f16 d; u8 qs[96]; } (98 bytes) dequantize_row_iq3_xxs:2503.
// Codebook decode: qs[0..63] are grid-index bytes (2 per 8-lane), qs[64..95] the
// per-32 scale+sign u32s. `db = d*(0.5 + (aux32>>28))*0.5`; each lane reads TWO
// 4-byte grid entries from kIq3xxsGrid and sign-flips per kKsignsIq2xs & kKmask.
void DequantIQ3_XXS(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 98;
    const float d = ReadF16(blk);
    const uint8_t* qs = blk + 2;              // grid indices, QK_K/4 = 64 bytes
    const uint8_t* scales_and_signs = qs + qk / 4;  // QK_K/8 = 32 bytes
    for (int ib32 = 0; ib32 < qk / 32; ++ib32) {
      uint32_t aux32;
      std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
      const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t signs = kKsignsIq2xs[(aux32 >> (7 * l)) & 127];
        const uint8_t* grid1 =
            reinterpret_cast<const uint8_t*>(kIq3xxsGrid + qs[2 * l + 0]);
        const uint8_t* grid2 =
            reinterpret_cast<const uint8_t*>(kIq3xxsGrid + qs[2 * l + 1]);
        for (int j = 0; j < 4; ++j) {
          y[j + 0] = db * grid1[j] * ((signs & kKmaskIq2xs[j + 0]) ? -1.f : 1.f);
          y[j + 4] = db * grid2[j] * ((signs & kKmaskIq2xs[j + 4]) ? -1.f : 1.f);
        }
        y += 8;
      }
      qs += 8;
    }
  }
}

// block_iq2_s = { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8]; } (82 bytes)
// dequantize_row_iq2_s:2471. Codebook decode: 8 sub-blocks of 32. Each of the 4
// lanes reads a grid-index low byte (qs[l]) OR'd with 2 high bits from qh[ib32]
// (10-bit index into the 1024-entry kIq2sGrid), then applies the DIRECT sign
// byte signs[l] (signs = qs + QK_K/8; NO ksigns lookup, unlike IQ2_XXS).
// scales[ib32] packs two 4-bit ls: low nibble -> db[0] (l=0,1), high nibble ->
// db[1] (l=2,3); db = d*(0.5 + ls)*0.25.
void DequantIQ2_S(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 82;
    const float d = ReadF16(blk);
    const uint8_t* qs = blk + 2;         // grid-index low bytes
    const uint8_t* qh = blk + 66;        // 2 high index bits per lane
    const uint8_t* scales = blk + 74;    // per-ib32 packed ls
    const uint8_t* signs = qs + qk / 8;  // qs + 32 (direct sign bytes)
    float db[2];
    for (int ib32 = 0; ib32 < qk / 32; ++ib32) {
      db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
      db[1] = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const float dl = db[l / 2];
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(
            kIq2sGrid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
        for (int j = 0; j < 8; ++j)
          y[j] = dl * grid[j] * ((signs[l] & kKmaskIq2xs[j]) ? -1.f : 1.f);
        y += 8;
      }
      qs += 4;
      signs += 4;
    }
  }
}

// block_iq1_s = { f16 d; u8 qs[32]; u16 qh[8]; } (50 bytes)
// dequantize_row_iq1_s:2578. Codebook decode: 8 sub-blocks of 32, each 4 lane
// groups of 8. The grid index is 11 bits, where qs[l] supplies the low 8 and
// (qh[ib] >> 3*l) & 7 the high 3, addressing the 2048-entry kIq1sGrid, whose
// entries are packed TERNARY (-1/0/+1) bytes. qh[ib] also carries the scale
// in bits 12-14 (dl = d * (2*ls + 1)) and the delta sign in bit 15, so a lane
// reconstructs as dl * (grid[j] + delta) with delta = +/-kIq1sDelta. There is no
// sign array: unlike IQ2_XXS/IQ2_S the sign lives in the codebook entry itself.
void DequantIQ1_S(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 50;
    const float d = ReadF16(blk);
    const uint8_t* qs = blk + 2;
    for (int ib = 0; ib < qk / 32; ++ib) {
      uint16_t qh = 0;
      std::memcpy(&qh, blk + 34 + 2 * ib, sizeof(qh));
      const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
      const float delta = (qh & 0x8000) ? -kIq1sDelta : kIq1sDelta;
      for (int l = 0; l < 4; ++l) {
        const int8_t* grid = reinterpret_cast<const int8_t*>(
            kIq1sGrid + (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
        for (int j = 0; j < 8; ++j)
          y[j] = dl * (static_cast<float>(grid[j]) + delta);
        y += 8;
      }
      qs += 4;
    }
  }
}

// block_iq1_xxxs = { f16 d; u8 qs[32]; u8 sc[4]; } (38 bytes)
// dequantize_row_iq1_xxxs:2727 of the PINNED FORK oracle `llama-cpp-unsloth`
// @ 36fe8e1cc. Like IQ1_S but wound tighter: the 256-entry grid means qs[l] is
// a WHOLE index (no high bits to splice), and one nibble of sc carries both the
// sub-block scale (bits 0-2, dl = d*(2*ls+1)) and the delta sign (bit 3). The
// delta magnitude is upstream's own IQ1S_DELTA, reused unchanged by the fork.
void DequantIQ1_XXXS(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 38;
    const float d = ReadF16(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* sc = blk + 34;
    for (int ib = 0; ib < qk / 32; ++ib) {
      const int nib = (sc[ib / 2] >> (4 * (ib & 1))) & 0xf;
      const float dl = d * static_cast<float>(2 * (nib & 7) + 1);
      const float delta = (nib & 8) ? -kIq1sDelta : kIq1sDelta;
      for (int l = 0; l < 4; ++l) {
        const int8_t* grid =
            reinterpret_cast<const int8_t*>(kIq1xxxsGrid + qs[l]);
        for (int j = 0; j < 8; ++j)
          y[j] = dl * (static_cast<float>(grid[j]) + delta);
        y += 8;
      }
      qs += 4;
    }
  }
}

// block_mxfp4 = { u8 e; u8 qs[16]; } (17 bytes) dequantize_row_mxfp4:511.
// OCP micro-scaling fp4: d = E8M0ToF32Half(e) is one power-of-two block scale;
// each of the 32 elements is an e2m1 nibble looked up in kValuesMxfp4. The
// nibbles use the split-half packing (element j in the low nibble of qs[j],
// j+16 in the high nibble), matching q4_0 and the ggml reference.
void DequantMXFP4(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 32;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 17;
    const float d = E8M0ToF32Half(blk[0]);
    const uint8_t* qs = blk + 1;
    for (int j = 0; j < qk / 2; ++j) {
      const int8_t x0 = kValuesMxfp4[qs[j] & 0x0F];
      const int8_t x1 = kValuesMxfp4[qs[j] >> 4];
      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

// Adapt a whole-row `(data, nb, y)` decoder to upstream's
// `ggml_to_float_t(x, y, k)` shape.
template <void (*Kernel)(const uint8_t*, int64_t, float*), int64_t kBlockElems>
void ToFloatAdapter(const void* x, float* y, int64_t k) {
  VT_CHECK(k % kBlockElems == 0,
           "block to_float: element count is not a whole number of blocks");
  Kernel(static_cast<const uint8_t*>(x), k / kBlockElems, y);
}

}  // namespace

ToFloatFn BlockToFloat(DType dtype) {
  switch (dtype) {
    case DType::kQ4_0: return &ToFloatAdapter<&DequantQ4_0, 32>;
    case DType::kQ8_0: return &ToFloatAdapter<&DequantQ8_0, 32>;
    case DType::kQ2_K: return &ToFloatAdapter<&DequantQ2_K, 256>;
    case DType::kQ3_K: return &ToFloatAdapter<&DequantQ3_K, 256>;
    case DType::kQ4_K: return &ToFloatAdapter<&DequantQ4_K, 256>;
    case DType::kQ5_K: return &ToFloatAdapter<&DequantQ5_K, 256>;
    case DType::kQ6_K: return &ToFloatAdapter<&DequantQ6_K, 256>;
    case DType::kQ8_K: return &ToFloatAdapter<&DequantQ8_K, 256>;
    case DType::kIQ2_XXS: return &ToFloatAdapter<&DequantIQ2_XXS, 256>;
    case DType::kIQ3_XXS: return &ToFloatAdapter<&DequantIQ3_XXS, 256>;
    case DType::kIQ2_S: return &ToFloatAdapter<&DequantIQ2_S, 256>;
    case DType::kIQ1_S: return &ToFloatAdapter<&DequantIQ1_S, 256>;
    case DType::kIQ1_XXXS: return &ToFloatAdapter<&DequantIQ1_XXXS, 256>;
    case DType::kMXFP4: return &ToFloatAdapter<&DequantMXFP4, 32>;
    default: return nullptr;
  }
}

}  // namespace vt::cpu
