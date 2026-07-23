// Block layout structs — a 1:1 mirror of llama.cpp @ 237ad9b96
// `ggml/src/ggml-common.h`:
//   :213-218 block_q4_0 · :242-245 block_q8_0 · :305-310 block_q3_K
//   :317-327 block_q4_K · :334-345 block_q5_K · :352-357 block_q6_K
//   :361-365 block_q8_K
//
// Upstream's Q4_K/Q5_K use a `GGML_EXTENSION union { struct { ggml_half d,
// dmin; }; ggml_half2 dm; }` purely so SIMD kernels can load both halves in one
// go; the union has no effect on the byte layout, so the portable tier states
// the two members directly (recorded deviation, layout-identical).
//
// `ggml_half` is stated as `uint16_t` rather than a float16 type because vt::
// has no native half: every read goes through `vt::F16ToF32`, which is the same
// IEEE binary16 decode `GGML_CPU_FP16_TO_FP32` performs on the generic tier.
//
// The `static_assert`s below are the load-bearing part of this file: they tie
// each struct to the INDEPENDENT block-geometry table in `src/vt/dtype.cpp`, so
// a padding surprise or a mistyped field is a compile error rather than a
// silently mis-strided weight buffer. (`vt::BlockBytes` is not constexpr, so the
// sizes are restated here from the same ggml-common.h arithmetic and
// cross-checked at RUNTIME by tests/vt/test_ops_quant_dot.cpp.)
#pragma once

#include <cstdint>

namespace vt::cpu {

// ggml-common.h:89-90
inline constexpr int kQK_K = 256;
inline constexpr int kKScaleSize = 12;
// ggml-common.h:184, :241
inline constexpr int kQK4_0 = 32;
inline constexpr int kQK8_0 = 32;

// ggml-common.h:213-218
struct BlockQ4_0 {
  uint16_t d;                  // delta (ggml_half)
  uint8_t qs[kQK4_0 / 2];      // nibbles / quants
};
static_assert(sizeof(BlockQ4_0) == 18, "wrong q4_0 block size/padding");

// ggml-common.h:242-245
struct BlockQ8_0 {
  uint16_t d;              // delta (ggml_half)
  int8_t qs[kQK8_0];       // quants
};
static_assert(sizeof(BlockQ8_0) == 34, "wrong q8_0 block size/padding");

// llama.cpp repack.h:23-40 — block<8,4> == block_q8_0x4: FOUR q8_0 rows
// interleaved for the i8mm repack GEMM (nrows_interleaved = 4, interleave_block
// = 8). d[i] is row i's fp16 delta; qs is the 4 rows' 32 int8 quants laid out
// in 8-byte chunks round-robin across rows: [r0[0:8] r1[0:8] r2[0:8] r3[0:8]
// r0[8:16] ...]. Same total bytes as 4 plain BlockQ8_0 (4*34 == 136), so a
// repacked weight buffer is byte-for-byte the same size as the plain one.
struct BlockQ8_0x4 {
  uint16_t d[4];
  int8_t qs[kQK8_0 * 4];  // 128
};
static_assert(sizeof(BlockQ8_0x4) == 4 * 2 + 128, "wrong q8_0x4 block size/padding");
inline constexpr int kQ8_0xNrowsInterleaved = 4;
inline constexpr int kQ8_0xInterleaveBlock = 8;

// ggml-common.h:305-310
struct BlockQ3_K {
  uint8_t hmask[kQK_K / 8];      // quants - high bit
  uint8_t qs[kQK_K / 4];         // quants - low 2 bits
  uint8_t scales[12];            // scales, quantized with 6 bits
  uint16_t d;                    // super-block scale (ggml_half)
};
static_assert(sizeof(BlockQ3_K) == 110, "wrong q3_K block size/padding");

// ggml-common.h:317-327
struct BlockQ4_K {
  uint16_t d;                       // super-block scale for quantized scales
  uint16_t dmin;                    // super-block scale for quantized mins
  uint8_t scales[kKScaleSize];      // scales and mins, quantized with 6 bits
  uint8_t qs[kQK_K / 2];            // 4-bit quants
};
static_assert(sizeof(BlockQ4_K) == 144, "wrong q4_K block size/padding");

// ggml-common.h:334-345
struct BlockQ5_K {
  uint16_t d;                       // super-block scale for quantized scales
  uint16_t dmin;                    // super-block scale for quantized mins
  uint8_t scales[kKScaleSize];      // scales and mins, quantized with 6 bits
  uint8_t qh[kQK_K / 8];            // quants, high bit
  uint8_t qs[kQK_K / 2];            // quants, low 4 bits
};
static_assert(sizeof(BlockQ5_K) == 176, "wrong q5_K block size/padding");

// ggml-common.h:352-357
struct BlockQ6_K {
  uint8_t ql[kQK_K / 2];        // quants, lower 4 bits
  uint8_t qh[kQK_K / 4];        // quants, upper 2 bits
  int8_t scales[kQK_K / 16];    // scales, quantized with 8 bits
  uint16_t d;                   // super-block scale (ggml_half)
};
static_assert(sizeof(BlockQ6_K) == 210, "wrong q6_K block size/padding");

// ggml-common.h:361-365 — the K-quant ACTIVATION encoding. `d` is a full f32
// (not ggml_half) and `bsums` caches the per-16-element quant sums that the
// Q4_K/Q5_K vec_dots use to apply the block minimum in one pass.
struct BlockQ8_K {
  float d;                          // delta
  int8_t qs[kQK_K];                 // quants
  int16_t bsums[kQK_K / 16];        // sum of quants in groups of 16
};
static_assert(sizeof(BlockQ8_K) == 292, "wrong q8_K block size/padding");

}  // namespace vt::cpu
