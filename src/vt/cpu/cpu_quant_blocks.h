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
// ggml-common.h:204
inline constexpr int kQK_MXFP4 = 32;

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

// ggml-common.h:288-299 block_q2_K. Upstream stores d/dmin in a
// `GGML_EXTENSION union { struct { ggml_half d, dmin; }; ggml_half2 dm; }` after
// scales+qs; the union is layout-inert (recorded deviation, layout-identical),
// so the two halves are stated directly. Field ORDER matters: scales and qs LEAD
// the block, the deltas TRAIL it (unlike q4_K/q5_K which lead with the deltas).
struct BlockQ2_K {
  uint8_t scales[kQK_K / 16];   // 16 — 4-bit sub-scale (low) + 4-bit sub-min (high)
  uint8_t qs[kQK_K / 4];        // 64 — 2-bit quants
  uint16_t d;                   // super-block scale (ggml_half)
  uint16_t dmin;                // super-block min scale (ggml_half)
};
static_assert(sizeof(BlockQ2_K) == 84, "wrong q2_K block size/padding");

// ggml-common.h:371-374 block_iq2_xxs. 2.0625 bpw codebook quant: `qs` is 32
// u16 = 8 u32 per block (four 8-bit grid indices in aux[0], four 7-bit sign
// selectors + a 4-bit scale in aux[1]) per 32-element sub-block. Decodes via
// kIq2xxsGrid + kKsignsIq2xs (cpu_quant_iq_tables.h).
struct BlockIQ2_XXS {
  uint16_t d;                   // super-block scale (ggml_half)
  uint16_t qs[kQK_K / 8];       // 32
};
static_assert(sizeof(BlockIQ2_XXS) == 66, "wrong iq2_xxs block size/padding");

// ggml-common.h:385-400 block_iq3_xxs. 3.0625 bpw codebook quant: `qs` holds
// QK_K/4 grid-index bytes (2 per lane) followed by QK_K/8 scale+sign bytes
// (one u32 per 32-element sub-block: 4-bit scale in the top nibble + four 7-bit
// sign selectors). Decodes via kIq3xxsGrid + kKsignsIq2xs.
struct BlockIQ3_XXS {
  uint16_t d;                   // super-block scale (ggml_half)
  uint8_t qs[3 * kQK_K / 8];    // 96 — QK_K/4 grid indices + QK_K/8 scale+signs
};
static_assert(sizeof(BlockIQ3_XXS) == 98, "wrong iq3_xxs block size/padding");

// ggml-common.h:386-392 block_iq2_s. 2.5625 bpw codebook quant: `qs` holds the
// 8-bit grid-index low bytes (first QK_K/8 = 32) followed by the per-lane sign
// bytes (last QK_K/8 = 32, applied DIRECTLY — no ksigns lookup, unlike IQ2_XXS);
// `qh` supplies 2 high index bits per lane (10-bit index into the 1024-entry
// iq2s_grid); `scales` packs two 4-bit sub-scales per 32-element sub-block.
struct BlockIQ2_S {
  uint16_t d;                   // super-block scale (ggml_half)
  uint8_t qs[kQK_K / 4];        // 64 — 32 grid-index low bytes + 32 sign bytes
  uint8_t qh[kQK_K / 32];       // 8 — 2 high index bits per lane
  uint8_t scales[kQK_K / 32];   // 8 — 4-bit ls (low) + 4-bit ls (high) per ib32
};
static_assert(sizeof(BlockIQ2_S) == 82, "wrong iq2_s block size/padding");

// ggml-common.h:414-419 block_iq1_s. 1.5625 bpw codebook quant, and the
// encoding that carries 96.92 % of `Qwen3.8-2.4T-A95B UD-IQ1_S` (see the target
// checkpoint census in .agents/specs/expert-streaming.md).
//
// Unlike every other codebook here, the grid entry is not an index into
// magnitudes: each kIq1sGrid u64 packs 8 TERNARY signed bytes (-1, 0 or +1),
// and the reconstructed value is `dl * (grid[j] + delta)`, so the codebook
// supplies the sign pattern and `delta` (+/- 0.125) shifts the whole lane group
// off the ternary lattice. That is why there is no sign-byte array and no
// ksigns lookup: the sign IS the codebook entry.
//
// `qh` does three jobs at once per 32-element sub-block: bits 0-8 give the
// three high index bits for each of the 4 lane groups (`(qh >> 3*l) & 7`,
// widening qs's 8-bit index to the 11 bits that address 2048 entries), bits
// 12-14 give the sub-block scale (`2*((qh >> 12) & 7) + 1`), and bit 15 selects
// the sign of `delta`.
struct BlockIQ1_S {
  uint16_t d;                   // super-block scale (ggml_half)
  uint8_t qs[kQK_K / 8];        // 32: one 8-bit grid-index low byte per group
  uint16_t qh[kQK_K / 32];      // 8: high index bits, scale, delta sign
};
static_assert(sizeof(BlockIQ1_S) == 50, "wrong iq1_s block size/padding");

// PINNED FORK oracle `llama-cpp-unsloth` @ 36fe8e1cc, ggml-common.h:478-483
// block_iq1_xxxs. 1.1875 bpw codebook quant, ggml type 66, which no upstream
// llama.cpp defines. It carries 96.92 % of `Qwen3.8-2.4T-A95B UD-Q1_0`. See
// .agents/oracles/llama-cpp-unsloth.md for why a fork is admitted for it.
//
// The same ternary codebook idea as IQ1_S, wound tighter in two ways. The grid
// has 256 entries rather than 2048, so `qs` is a WHOLE 8-bit index and there
// are no high bits to splice in from elsewhere. And the per-32 scale plus delta
// sign live in one NIBBLE of `sc` (bits 0-2 scale, bit 3 delta sign) rather
// than in a u16 per sub-block, which is where most of the 0.375 bpw saving
// against IQ1_S comes from.
struct BlockIQ1_XXXS {
  uint16_t d;                   // super-block scale (ggml_half)
  uint8_t qs[kQK_K / 8];        // 32: one whole 8-bit grid index per lane group
  uint8_t sc[kQK_K / 64];       // 4: two sub-block nibbles per byte
};
static_assert(sizeof(BlockIQ1_XXXS) == 38, "wrong iq1_xxxs block size/padding");

// ggml-common.h:204-209 block_mxfp4. OCP micro-scaling fp4: `e` is one E8M0
// (power-of-two) shared exponent for the whole 32-element block, `qs` packs the
// 32 e2m1 4-bit elements two-per-byte (element j in the low nibble of qs[j],
// element j+16 in the high nibble — the same split-half packing q4_0 uses).
struct BlockMXFP4 {
  uint8_t e;                     // E8M0 shared exponent
  uint8_t qs[kQK_MXFP4 / 2];     // 16 — packed e2m1 nibbles
};
static_assert(sizeof(BlockMXFP4) == 17, "wrong mxfp4 block size/padding");

}  // namespace vt::cpu
