// Device BLOCK-ROW DECODERS for the dequantizing gather (KGATHER).
//
// WHAT THIS IS. `vt::Embedding` admits a block-quantized table and decodes ONE
// ROW per gathered id (`vt/ops.cpp`), a port of `ggml_compute_forward_get_rows_q`
// (llama.cpp @ b10451 `ggml/src/ggml-cpu/ops.cpp:4850`). Until this file that
// existed only on the CPU: `EmbeddingKernelCuda` asserted f32/bf16, so
// `DeviceQuantGatherSupported` refused every non-CPU device and a kept table
// expanded at load. For `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` that
// expansion is 26.822 GiB of IQ4_NL -> 95.368 GiB of bf16, which fits nothing in
// this fleet, so the device-resident quantized table is what makes a GPU arm
// possible at all.
//
// WHERE THE CODE COMES FROM. vLLM has NO implementation: its embedding is
// `torch.nn.functional.embedding` over a dense tensor and it has no
// block-quantized gather at any revision. The secondary oracle
// (`AGENTS.md`, "When vLLM has no implementation") is llama.cpp, whose CUDA
// backend decodes blocks in `ggml/src/ggml-cuda/convert.cu`
// (`dequantize_block_q4_K:169`, `_q5_K:209`, `_q6_K:257`, `_q2_K:130`,
// `_q3_K:88`, `_iq4_nl:462`, `_iq2_xxs:296`, `_iq1_s:406`) and gathers in
// `ggml/src/ggml-cuda/getrows.cu:6` (`k_get_rows`). Two deliberate deviations
// from that shape are recorded here rather than discovered by a reader:
//
//   1. llama.cpp's CUDA `get_rows` handles the LEGACY quants only -- its
//      `get_rows_cuda` switch (getrows.cu:172) lists F16/BF16/F32/Q4_0/Q4_1/
//      Q5_0/Q5_1/Q8_0 and aborts on every K-quant and every IQ type. The
//      shipped n-gram table is IQ4_NL and the shipped experts are IQ1_S/IQ2_XXS,
//      so a 1:1 port of `get_rows_cuda` would not run this model. This file
//      therefore takes llama.cpp's DECODE (convert.cu) and this project's own
//      gather shape, which is the CPU arm's.
//   2. The decoders below are transliterations of `src/vt/cpu/cpu_quant_dequant.cpp`
//      -- itself the byte-for-byte port of `ggml-quants.c`'s `dequantize_row_*`
//      -- rather than of convert.cu's warp-cooperative variants. The CPU arm is
//      the BEHAVIOURAL ORACLE this kernel is gated against, and upstream's own
//      CUDA decoders are not bit-identical to its scalar ones (they reassociate
//      the scale products). Matching the scalar order is what makes an exact
//      CPU-vs-CUDA comparison the gate rather than a tolerance.
//
// ALIGNMENT. A block base is NOT 4-byte aligned in general (66-byte IQ2_XXS,
// 110-byte Q3_K, 210-byte Q6_K ...), and a misaligned 32-bit device load is a
// fault, not a slow load. Every multi-byte read below therefore goes through
// the byte-assembly helpers `DqU16`/`DqU32`/`DqF16`/`DqF32`. This is the one
// systematic difference from the CPU code, which uses `std::memcpy`.
//
// This header is included from INSIDE `namespace vt::cuda { namespace {` in
// cuda_quant_dot.cu -- the only translation unit that defines the device
// codebooks (`cuda_quant_iq_tables.cuh` defines, not declares, them) -- and it
// opens no namespace of its own.
#pragma once

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

// --- contraction discipline --------------------------------------------------
// The CPU arm is compiled `-ffp-contract=off` (CMakeLists.txt:55), nvcc defaults
// to `--fmad=true`, and three of the decoders below compute `d*q - m`, which is
// exactly the shape a contracting compiler folds into one FMA. An FMA and a
// separate multiply-then-subtract differ in the LAST BIT, so leaving this to the
// default would turn an exact CPU-vs-CUDA gate into a tolerance, and a tolerance
// cannot see a wrong scale unpack that lands within it. `DqMulSub` states the
// two operations explicitly; nothing else in this file pairs a multiply with an
// add or a subtract (verified case by case: Q4_0/Q5_0/Q8_0/IQ4_NL/IQ4_XS/MXFP4
// are a single product, Q3_K/Q6_K subtract in INTEGER before the product, and
// the IQ1 family adds the delta before the product, which is not an FMA shape).
__device__ inline float DqMulSub(float a, float b, float c) {
  return __fsub_rn(__fmul_rn(a, b), c);
}

// get_scale_min_k4 (ggml-quants.c:822); cpu_quant_dequant.cpp GetScaleMinK4.
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

// block_q4_0 = { f16 d; u8 qs[16] } — DequantQ4_0 / dequantize_row_q4_0:401
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

// block_q5_0 = { f16 d; u8 qh[4]; u8 qs[16] } — DequantQ5_0 / :500
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

// block_iq4_nl = { f16 d; u8 qs[16] } — DequantIQ4_NL / :2725
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

// block_q8_0 = { f16 d; i8 qs[32] } — DequantQ8_0 / :495
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

// block_mxfp4 = { u8 e; u8 qs[16] } — DequantMXFP4 / :511
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

// block_q2_K = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin } — DequantQ2_K / :903
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

// block_q3_K = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d } — DequantQ3_K / :1247
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

// block_q4_K = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] } — DequantQ4_K / :1471
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
// DequantQ5_K / :1673
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

// block_q6_K = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d } — DequantQ6_K / :1881
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

// block_q8_K = { f32 d; i8 qs[256]; i16 bsums[16] } — DequantQ8_K.
// Activation-only encoding; it never appears as a FILE weight, so it can never
// reach the gather from the loader. It is here because the CPU table carries it
// and an asymmetric capability list is a defect waiting to be found later.
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

// block_iq2_xxs = { f16 d; u16 qs[32] } — DequantIQ2_XXS / :2416.
// `aux32` is a memcpy of the eight bytes at `qs + 8*ib32` on the CPU side, so
// `aux8` IS those bytes; the device reads them directly and takes only aux32[1]
// through the byte-assembly load.
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

// block_iq3_xxs = { f16 d; u8 qs[96] } — DequantIQ3_XXS / :2503
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
        // BY VALUE and byte-extracted, exactly as DotIQ3XXS above reads it,
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
// DequantIQ3_S / :2607. NOT an IQ3_XXS variant: 512-entry table, a ninth index
// bit spliced out of `qh` with an ASYMMETRIC shift pair, direct sign bytes, and
// an ODD-integer sub-block scale `db = d * (1 + 2*ls)` shared by TWO sub-blocks.
// Upstream walks the sub-blocks in PAIRS for that last reason and this codec
// keeps that loop shape, so the two diff by eye.
//
// GATHER ONLY. There is deliberately no `WType::kIQ3_S` beside it: the GEMM arm
// needs a CPU `vec_dot` first (`KeepQuantDType` gates on `HasQuantDotKernel`),
// and the two halves have to land together or a CUDA IQ3_S GEMM takes
// `IsCudaKeepQuantSupported`'s host fallback. See `.agents/specs/gguf-iq3s.md`.
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

// block_iq2_xs = { f16 d; u16 qs[32]; u8 scales[8] } — DequantIQ2_XS / :2516
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
// DequantIQ4_XS / :2743
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

// block_iq2_s = { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] } — DequantIQ2_S / :2471
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

// block_iq1_s = { f16 d; u8 qs[32]; u16 qh[8] } — DequantIQ1_S / :2578.
// kIq1sDelta (0.125f, ggml-common.h:1121 IQ1S_DELTA) is stated here rather than
// included, because this header sees no host constants.
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

// block_iq1_xxxs = { f16 d; u8 qs[32]; u8 sc[4] } — DequantIQ1_XXXS.
// Fork encoding (`llama-cpp-unsloth` @ 36fe8e1cc, dequantize_row_iq1_xxxs:2727).
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

// --- the gather -------------------------------------------------------------
// One thread decodes ONE BLOCK of one gathered row. The CPU arm parallelises
// over gathered ROWS; a row of the shipped table is five IQ4_NL blocks, so
// per-row would leave four fifths of a warp idle on the very table this exists
// for. Out-of-range ids are handled EXACTLY as EmbeddingKernelCuda already does
// them: clamped in-kernel so the read stays in bounds, and the first offender
// recorded through the same device flag (cuda_ops.cu, "Out-of-range reporting
// WITHOUT a per-call barrier"). The CPU arm throws instead; that difference is
// pre-existing and belongs to the float path too.
template <typename Tout, typename Tid, class Codec>
__global__ void EmbeddingQuantGatherKernel(Tout* out, const uint8_t* table,
                                           const Tid* ids, int64_t t, int64_t nb,
                                           int64_t h, int64_t v, size_t row_bytes,
                                           EmbeddingQuantErr* err) {
  const int64_t total = t * nb;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += step) {
    const int64_t i = idx / nb;
    const int64_t b = idx - i * nb;
    int64_t id = static_cast<int64_t>(ids[i]);
    if (id < 0 || id >= v) {
      if (atomicCAS(&err->status, 0, 1) == 0) err->id = static_cast<long long>(id);
      id = id < 0 ? 0 : v - 1;
    }
    const uint8_t* blk =
        table + static_cast<size_t>(id) * row_bytes + static_cast<size_t>(b) * Codec::kBytes;
    Codec::template Decode<Tout>(blk, out + i * h + b * Codec::kElems);
  }
}
