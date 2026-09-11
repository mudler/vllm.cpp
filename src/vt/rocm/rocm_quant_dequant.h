// Selected-block decoders for BACKEND-ROCM-QUANT-GATHER (#3093).
// Independently owned port of src/vt/cuda/cuda_quant_dequant.cuh at 6db4bef90685.
// Primary behavior: plugin d4c1f0d vocal_embeds.py:79-98 selects and decodes rows.
// This fused HIP implementation writes selected blocks directly to caller output.
// Scalar expressions preserve the pinned ggml-quants.c dequantize_row_* order.
// Stock pin: 10bf611e533d81f739128304991c5e133c6aebd8. Fork: 36fe8e1cc.
// Byte assembly permits every packed-table offset, including odd block strides.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#include "vt/rocm/rocm_quant_iq_tables.h"

namespace vt::rocm {
namespace {

// Numeric conversions mirror src/vt/dtype.cpp, including bf16 ties and NaNs.
__device__ inline float DF16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0x1F) return __int_as_float(sign | 0x7F800000 | (mant << 13));
  if (exp == 0) {
    if (mant == 0) return __int_as_float(sign);
    int shift = 0;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      ++shift;
    }
    mant &= 0x3FF;
    return __int_as_float(sign | ((113 - shift) << 23) | (mant << 13));
  }
  return __int_as_float(sign | ((exp + 112) << 23) | (mant << 13));
}

__device__ inline uint16_t DF32ToBF16(float f) {
  uint32_t u = __float_as_int(f);
  if ((u & 0x7F800000) == 0x7F800000 && (u & 0x7FFFFF)) {
    return static_cast<uint16_t>((u >> 16) | 0x0040);
  }
  uint32_t rounding = 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

__device__ inline float DE8M0ToF32Half(uint8_t x) {
  const uint32_t bits =
      x < 2 ? (0x00200000u << x) : (static_cast<uint32_t>(x - 1) << 23);
  return __int_as_float(static_cast<int>(bits));
}

// --- unaligned little-endian loads -----------------------------------------
__device__ inline uint16_t DqU16(const uint8_t* p) {
  return static_cast<uint16_t>(static_cast<uint32_t>(p[0]) |
                               (static_cast<uint32_t>(p[1]) << 8));
}

__device__ inline uint32_t DqU32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

// `ReadF16` of cpu_quant_dequant.cpp: little-endian ggml_half widened to f32.
__device__ inline float DqF16(const uint8_t* p) { return DF16ToF32(DqU16(p)); }

// The f32 super-block delta Q8_K stores raw (cpu_quant_dequant.cpp DequantQ8_K).
__device__ inline float DqF32(const uint8_t* p) {
  return __int_as_float(static_cast<int>(DqU32(p)));
}

// --- output store ------------------------------------------------------------
// `StoreF32(out, i, v)` of the CPU arm, specialised on the out dtype. bf16 uses
// the SAME round-to-nearest-even + NaN-quieting `DF32ToBF16` the rest of this
// TU uses, which is the bit-exact port of `vt::F32ToBF16` (src/vt/dtype.cpp).
template <typename Tout>
struct DqStore;
template <>
struct DqStore<float> {
  static __device__ inline void Set(float* y, int64_t i, float v) { y[i] = v; }
};
template <>
struct DqStore<uint16_t> {
  static __device__ inline void Set(uint16_t* y, int64_t i, float v) {
    y[i] = DF32ToBF16(v);
  }
};

// HIP and the scalar control use -ffp-contract=off to preserve two operations.
__device__ inline float DqMulSub(float a, float b, float c) {
  return a * b - c;
}

// get_scale_min_k4 (ggml-quants.c:880); cpu_quant_dequant.cpp GetScaleMinK4.
__device__ inline void DqGetScaleMinK4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

// --- one CODEC per block dtype ----------------------------------------------
// Each `Decode` writes exactly `kElems` outputs starting at `y[0]`, in the same
// ORDER and with the same float expression as its `Dequant*` twin in
// src/vt/cpu/cpu_quant_dequant.cpp. `kBytes` is the block stride and is
// cross-checked against `vt::BlockBytes` by the host launcher below.

// block_q4_0 = { f16 d; u8 qs[16] }.
// Stock ggml/src/ggml-quants.c:459.
struct DqQ4_0 {
  static constexpr int kBytes = 18;
  static constexpr int kElems = 32;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
      const int x0 = (qs[j] & 0x0F) - 8;
      const int x1 = (qs[j] >> 4) - 8;
      DqStore<Tout>::Set(y, j, x0 * d);
      DqStore<Tout>::Set(y, j + 16, x1 * d);
    }
  }
};

// block_q5_0 = { f16 d; u8 qh[4]; u8 qs[16] }.
// Stock ggml/src/ggml-quants.c:500.
struct DqQ5_0 {
  static constexpr int kBytes = 22;
  static constexpr int kElems = 32;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint32_t qh = DqU32(blk + 2);
    const uint8_t* qs = blk + 6;
    for (int j = 0; j < 16; ++j) {
      const uint8_t xh_0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
      const int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = ((qs[j] >> 4) | xh_1) - 16;
      DqStore<Tout>::Set(y, j, x0 * d);
      DqStore<Tout>::Set(y, j + 16, x1 * d);
    }
  }
};

// block_iq4_nl = { f16 d; u8 qs[16] }.
// Stock ggml/src/ggml-quants.c:2725.
struct DqIQ4_NL {
  static constexpr int kBytes = 18;
  static constexpr int kElems = 32;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
      DqStore<Tout>::Set(y, j, d * d_kvalues_iq4nl[qs[j] & 0x0F]);
      DqStore<Tout>::Set(y, j + 16, d * d_kvalues_iq4nl[qs[j] >> 4]);
    }
  }
};

// block_q8_0 = { f16 d; i8 qs[32] }.
// Stock ggml/src/ggml-quants.c:553.
struct DqQ8_0 {
  static constexpr int kBytes = 34;
  static constexpr int kElems = 32;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    for (int j = 0; j < 32; ++j) DqStore<Tout>::Set(y, j, qs[j] * d);
  }
};

// block_mxfp4 = { u8 e; u8 qs[16] }.
// Stock ggml/src/ggml-quants.c:569.
struct DqMXFP4 {
  static constexpr int kBytes = 17;
  static constexpr int kElems = 32;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DE8M0ToF32Half(blk[0]);
    const uint8_t* qs = blk + 1;
    for (int j = 0; j < 16; ++j) {
      const int8_t x0 = d_kvalues_mxfp4[qs[j] & 0x0F];
      const int8_t x1 = d_kvalues_mxfp4[qs[j] >> 4];
      DqStore<Tout>::Set(y, j, x0 * d);
      DqStore<Tout>::Set(y, j + 16, x1 * d);
    }
  }
};

// block_q2_K = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }.
// Stock ggml/src/ggml-quants.c:961.
struct DqQ2_K {
  static constexpr int kBytes = 84;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const uint8_t* scales = blk;
    const uint8_t* q = blk + 16;
    const float d = DqF16(blk + 80);
    const float min = DqF16(blk + 82);
    int o = 0;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = scales[is++];
        float dl = d * (sc & 0xF);
        float ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          DqStore<Tout>::Set(
              y, o++, DqMulSub(dl, static_cast<int8_t>((q[l] >> shift) & 3), ml));
        sc = scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          DqStore<Tout>::Set(
              y, o++, DqMulSub(dl, static_cast<int8_t>((q[l + 16] >> shift) & 3), ml));
        shift += 2;
      }
      q += 32;
    }
  }
};

// block_q3_K = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }.
// Stock ggml/src/ggml-quants.c:1305.
struct DqQ3_K {
  static constexpr int kBytes = 110;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint8_t* hm = blk;
    const uint8_t* q = blk + 32;
    const uint8_t* sc_raw = blk + 96;
    const float d_all = DqF16(blk + 108);

    uint32_t aux[4];
    aux[0] = DqU32(sc_raw + 0);
    aux[1] = DqU32(sc_raw + 4);
    aux[2] = DqU32(sc_raw + 8);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    int o = 0;
    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l)
          DqStore<Tout>::Set(y, o++,
                             dl * (static_cast<int8_t>((q[l + 0] >> shift) & 3) -
                                   ((hm[l + 0] & m) ? 0 : 4)));
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l)
          DqStore<Tout>::Set(y, o++,
                             dl * (static_cast<int8_t>((q[l + 16] >> shift) & 3) -
                                   ((hm[l + 16] & m) ? 0 : 4)));
        shift += 2;
        m = static_cast<uint8_t>(m << 1);
      }
      q += 32;
    }
  }
};

// block_q4_K = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }.
// Stock ggml/src/ggml-quants.c:1529.
struct DqQ4_K {
  static constexpr int kBytes = 144;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const float min = DqF16(blk + 2);
    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;
    int o = 0;
    int is = 0;
    uint8_t sc = 0;
    uint8_t mm = 0;
    for (int j = 0; j < 256; j += 64) {
      DqGetScaleMinK4(is + 0, scales, &sc, &mm);
      const float d1 = d * sc;
      const float m1 = min * mm;
      DqGetScaleMinK4(is + 1, scales, &sc, &mm);
      const float d2 = d * sc;
      const float m2 = min * mm;
      for (int l = 0; l < 32; ++l)
        DqStore<Tout>::Set(y, o++, DqMulSub(d1, q[l] & 0xF, m1));
      for (int l = 0; l < 32; ++l)
        DqStore<Tout>::Set(y, o++, DqMulSub(d2, q[l] >> 4, m2));
      q += 32;
      is += 2;
    }
  }
};

// block_q5_K = { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }
// Stock ggml/src/ggml-quants.c:1731.
struct DqQ5_K {
  static constexpr int kBytes = 176;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const float min = DqF16(blk + 2);
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* ql = blk + 48;
    int o = 0;
    int is = 0;
    uint8_t sc = 0;
    uint8_t mm = 0;
    uint8_t u1 = 1;
    uint8_t u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      DqGetScaleMinK4(is + 0, scales, &sc, &mm);
      const float d1 = d * sc;
      const float m1 = min * mm;
      DqGetScaleMinK4(is + 1, scales, &sc, &mm);
      const float d2 = d * sc;
      const float m2 = min * mm;
      for (int l = 0; l < 32; ++l)
        DqStore<Tout>::Set(
            y, o++, DqMulSub(d1, (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0), m1));
      for (int l = 0; l < 32; ++l)
        DqStore<Tout>::Set(
            y, o++, DqMulSub(d2, (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0), m2));
      ql += 32;
      is += 2;
      u1 = static_cast<uint8_t>(u1 << 2);
      u2 = static_cast<uint8_t>(u2 << 2);
    }
  }
};

// block_q6_K = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }.
// Stock ggml/src/ggml-quants.c:1939.
struct DqQ6_K {
  static constexpr int kBytes = 210;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
    const float d = DqF16(blk + 208);
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) |
                                              (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) |
                                              (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) |
                                              (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) |
                                              (((qh[l] >> 6) & 3) << 4)) - 32;
        DqStore<Tout>::Set(y, o + l + 0, d * sc[is + 0] * q1);
        DqStore<Tout>::Set(y, o + l + 32, d * sc[is + 2] * q2);
        DqStore<Tout>::Set(y, o + l + 64, d * sc[is + 4] * q3);
        DqStore<Tout>::Set(y, o + l + 96, d * sc[is + 6] * q4);
      }
      o += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
};

// block_q8_K = { f32 d; i8 qs[256]; i16 bsums[16] }.
// The GGUF reader admits type 15 for embedding tables. Matrix weights still
// require a dot kernel; the embedding role keeps these original packed bytes.
// Stock ggml/src/ggml-quants.c:2807.
struct DqQ8_K {
  static constexpr int kBytes = 292;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF32(blk);
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
    for (int j = 0; j < 256; ++j) DqStore<Tout>::Set(y, j, d * qs[j]);
  }
};

// block_iq2_xxs = { f16 d; u16 qs[32] }.
// `aux32` is a memcpy of the eight bytes at `qs + 8*ib32` on the CPU side, so
// `aux8` IS those bytes; the device reads them directly and takes only aux32[1]
// through the byte-assembly load.
// Stock ggml/src/ggml-quants.c:2488.
struct DqIQ2_XXS {
  static constexpr int kBytes = 66;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    int o = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const uint8_t* aux8 = qs + 8 * ib32;
      const uint32_t aux32_1 = DqU32(qs + 8 * ib32 + 4);
      const float db = d * (0.5f + (aux32_1 >> 28)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t* grid =
            reinterpret_cast<const uint8_t*>(d_iq2xxs_grid + aux8[l]);
        const uint8_t signs = d_ksigns_iq2xs[(aux32_1 >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j)
          DqStore<Tout>::Set(y, o + j,
                             db * grid[j] * ((signs & d_kmask_iq2xs[j]) ? -1.f : 1.f));
        o += 8;
      }
    }
  }
};

// block_iq3_xxs = { f16 d; u8 qs[96] }.
// Stock ggml/src/ggml-quants.c:2575.
struct DqIQ3_XXS {
  static constexpr int kBytes = 98;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* scales_and_signs = qs + 64;
    int o = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const uint32_t aux32 = DqU32(scales_and_signs + 4 * ib32);
      const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t signs = d_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        // d_iq3xxs_grid is the one codebook held in `__constant__`; it is read
        // Read the codebook word by value, then extract its bytes,
        // rather than through a pointer into the constant window.
        const uint32_t g1 = d_iq3xxs_grid[qs[2 * l + 0]];
        const uint32_t g2 = d_iq3xxs_grid[qs[2 * l + 1]];
        for (int j = 0; j < 4; ++j) {
          const uint8_t b1 = static_cast<uint8_t>((g1 >> (8 * j)) & 0xFFu);
          const uint8_t b2 = static_cast<uint8_t>((g2 >> (8 * j)) & 0xFFu);
          DqStore<Tout>::Set(y, o + j + 0,
                             db * b1 * ((signs & d_kmask_iq2xs[j + 0]) ? -1.f : 1.f));
          DqStore<Tout>::Set(y, o + j + 4,
                             db * b2 * ((signs & d_kmask_iq2xs[j + 4]) ? -1.f : 1.f));
        }
        o += 8;
      }
      qs += 8;
    }
  }
};

// block_iq3_s = { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] }
// The 512-entry codebook uses a ninth index bit from qh and direct sign bytes.
// Each odd-integer scale is shared by two sub-blocks, matching the upstream loop.
// This decoder serves HIP gather only. IQ3_S matrix compute is outside this row;
// its owning record is `.agents/specs/gguf-iq3s.md`.
// Stock ggml/src/ggml-quants.c:2607.
struct DqIQ3_S {
  static constexpr int kBytes = 110;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 66;
    const uint8_t* signs = blk + 74;
    const uint8_t* scales = blk + 106;
    int o = 0;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
      const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
      const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
      for (int l = 0; l < 4; ++l) {
        const uint32_t g1 = d_iq3s_grid[qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)];
        const uint32_t g2 = d_iq3s_grid[qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)];
        for (int j = 0; j < 4; ++j) {
          const uint8_t b1 = static_cast<uint8_t>((g1 >> (8 * j)) & 0xFFu);
          const uint8_t b2 = static_cast<uint8_t>((g2 >> (8 * j)) & 0xFFu);
          DqStore<Tout>::Set(y, o + j + 0,
                             db1 * b1 * ((signs[l] & d_kmask_iq2xs[j + 0]) ? -1.f : 1.f));
          DqStore<Tout>::Set(y, o + j + 4,
                             db1 * b2 * ((signs[l] & d_kmask_iq2xs[j + 4]) ? -1.f : 1.f));
        }
        o += 8;
      }
      qs += 8;
      signs += 4;
      for (int l = 0; l < 4; ++l) {
        const uint32_t g1 = d_iq3s_grid[qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)];
        const uint32_t g2 = d_iq3s_grid[qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)];
        for (int j = 0; j < 4; ++j) {
          const uint8_t b1 = static_cast<uint8_t>((g1 >> (8 * j)) & 0xFFu);
          const uint8_t b2 = static_cast<uint8_t>((g2 >> (8 * j)) & 0xFFu);
          DqStore<Tout>::Set(y, o + j + 0,
                             db2 * b1 * ((signs[l] & d_kmask_iq2xs[j + 0]) ? -1.f : 1.f));
          DqStore<Tout>::Set(y, o + j + 4,
                             db2 * b2 * ((signs[l] & d_kmask_iq2xs[j + 4]) ? -1.f : 1.f));
        }
        o += 8;
      }
      qh += 2;
      qs += 8;
      signs += 4;
    }
  }
};

// block_iq2_xs = { f16 d; u16 qs[32]; u8 scales[8] }.
// Stock ggml/src/ggml-quants.c:2516.
struct DqIQ2_XS {
  static constexpr int kBytes = 74;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* scales = blk + 66;
    int o = 0;
    float db[2];
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
      db[1] = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const uint16_t q = DqU16(qs + 2 * (4 * ib32 + l));
        const uint8_t* grid =
            reinterpret_cast<const uint8_t*>(d_iq2xs_grid + (q & 511));
        const uint8_t signs = d_ksigns_iq2xs[q >> 9];
        for (int j = 0; j < 8; ++j)
          DqStore<Tout>::Set(y, o + j,
                             db[l / 2] * grid[j] *
                                 ((signs & d_kmask_iq2xs[j]) ? -1.f : 1.f));
        o += 8;
      }
    }
  }
};

// block_iq4_xs = { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }
// Stock ggml/src/ggml-quants.c:2743.
struct DqIQ4_XS {
  static constexpr int kBytes = 136;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint16_t scales_h = DqU16(blk + 2);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    int o = 0;
    for (int ib = 0; ib < 8; ++ib) {
      const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                     (((scales_h >> (2 * ib)) & 3) << 4);
      const float dl = d * (ls - 32);
      for (int j = 0; j < 16; ++j) {
        DqStore<Tout>::Set(y, o + j + 0, dl * d_kvalues_iq4nl[qs[j] & 0xf]);
        DqStore<Tout>::Set(y, o + j + 16, dl * d_kvalues_iq4nl[qs[j] >> 4]);
      }
      o += 32;
      qs += 16;
    }
  }
};

// block_iq2_s = { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] }.
// Stock ggml/src/ggml-quants.c:2543.
struct DqIQ2_S {
  static constexpr int kBytes = 82;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 66;
    const uint8_t* scales = blk + 74;
    const uint8_t* signs = qs + 32;
    int o = 0;
    float db[2];
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
      db[1] = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const float dl = db[l / 2];
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(
            d_iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
        for (int j = 0; j < 8; ++j)
          DqStore<Tout>::Set(y, o + j,
                             dl * grid[j] * ((signs[l] & d_kmask_iq2xs[j]) ? -1.f : 1.f));
        o += 8;
      }
      qs += 4;
      signs += 4;
    }
  }
};

// block_iq1_s = { f16 d; u8 qs[32]; u16 qh[8] }.
// kIq1sDelta (0.125f, ggml-common.h:1132 IQ1S_DELTA) is stated here rather than
// included, because this header sees no host constants.
// Stock ggml/src/ggml-quants.c:2650.
struct DqIQ1_S {
  static constexpr int kBytes = 50;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    constexpr float kDelta = 0.125f;
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    int o = 0;
    for (int ib = 0; ib < 8; ++ib) {
      const uint16_t qh = DqU16(blk + 34 + 2 * ib);
      const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
      const float delta = (qh & 0x8000) ? -kDelta : kDelta;
      for (int l = 0; l < 4; ++l) {
        const int8_t* grid = reinterpret_cast<const int8_t*>(
            d_iq1s_grid + (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
        for (int j = 0; j < 8; ++j)
          DqStore<Tout>::Set(y, o + j, dl * (static_cast<float>(grid[j]) + delta));
        o += 8;
      }
      qs += 4;
    }
  }
};

// block_iq1_xxxs = { f16 d; u8 qs[32]; u8 sc[4] }.
// Encoding from the registered llama-cpp-unsloth pin above.
// Fork ggml/src/ggml-quants.c:2727.
struct DqIQ1_XXXS {
  static constexpr int kBytes = 38;
  static constexpr int kElems = 256;
  template <typename Tout>
  static __device__ void Decode(const uint8_t* blk, Tout* y) {
    constexpr float kDelta = 0.125f;
    const float d = DqF16(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* sc = blk + 34;
    int o = 0;
    for (int ib = 0; ib < 8; ++ib) {
      const int nib = (sc[ib / 2] >> (4 * (ib & 1))) & 0xf;
      const float dl = d * static_cast<float>(2 * (nib & 7) + 1);
      const float delta = (nib & 8) ? -kDelta : kDelta;
      for (int l = 0; l < 4; ++l) {
        const int8_t* grid =
            reinterpret_cast<const int8_t*>(d_iq1xxxs_grid + qs[l]);
        for (int j = 0; j < 8; ++j)
          DqStore<Tout>::Set(y, o + j, dl * (static_cast<float>(grid[j]) + delta));
        o += 8;
      }
      qs += 4;
    }
  }
};


}  // namespace
}  // namespace vt::rocm
