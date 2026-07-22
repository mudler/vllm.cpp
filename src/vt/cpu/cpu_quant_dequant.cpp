// Block-quant `to_float` decoders — the `dequantize_row_*` kernels ported
// byte-for-byte from llama.cpp @ 237ad9b96:
//   ggml/src/ggml-common.h  (block_q4_0/q8_0/q3_K/q4_K/q5_K/q6_K layouts)
//   ggml/src/ggml-quants.c  (dequantize_row_q4_0:401, _q8_0:495, _q3_K:1247,
//                            _q4_K:1471, _q5_K:1673, _q6_K:1881,
//                            get_scale_min_k4:822)
//
// These moved here VERBATIM from the GGUF loader
// (src/vllm/model_executor/model_loader/gguf_dequant.cpp), which now delegates:
// vt:: is the lower layer and the compute-in-quant GEMM's generic fallback
// needs the same decode, so a single implementation serves both the loader
// oracle and `vt::MatmulBTQuant`. Numerics are unchanged by construction (same
// code, same order, same `-ffp-contract=off` pinning) and
// tests/vllm/test_gguf_dequant.cpp gates that.
#include <cstring>

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
    case DType::kQ3_K: return &ToFloatAdapter<&DequantQ3_K, 256>;
    case DType::kQ4_K: return &ToFloatAdapter<&DequantQ4_K, 256>;
    case DType::kQ5_K: return &ToFloatAdapter<&DequantQ5_K, 256>;
    case DType::kQ6_K: return &ToFloatAdapter<&DequantQ6_K, 256>;
    case DType::kQ8_K: return &ToFloatAdapter<&DequantQ8_K, 256>;
    default: return nullptr;
  }
}

}  // namespace vt::cpu
