// Activation quantization (`from_float`) — QUANT-GGUF-CIQ-GEMM work row G2.
//
// Ported byte-for-byte from llama.cpp @ 237ad9b96:
//   `ggml/src/ggml-quants.c:238`  quantize_row_q8_0_ref
//   `ggml/src/ggml-quants.c:2696` quantize_row_q8_K_ref
//   `ggml/src/ggml-quants.c:563`  nearest_int (the magic-constant round)
// The CPU trait-table entry points that wrap those references are
//   `ggml/src/ggml-cpu/quants.c:45`  quantize_row_q8_0_generic
//   `ggml/src/ggml-cpu/quants.c:117` quantize_row_q8_K_generic
// — both are a bare call to the `_ref` form on the generic tier, so this file
// implements the reference bodies directly and registers them as the traits
// table's `from_float`.
//
// These two encodings are the only `vec_dot_type`s the six executable GGUF
// weight types dispatch to (`ggml-cpu.c:230-326`): Q8_0 for the legacy
// 32-element types (Q4_0, Q8_0) and Q8_K for the K-quants (Q3_K, Q4_K, Q5_K,
// Q6_K). Upstream quantizes src1 ONCE per GEMM into a scratch buffer
// (`ggml-cpu.c:1313-1349`) and then dots every weight row against it; the
// scratch sizing at the bottom of this file mirrors the `ggml_graph_plan`
// wdata computation (`ggml-cpu.c:2752-2980`).
//
// FIDELITY NOTE (deliberate, matches upstream exactly): Q8_0 derives the
// quants from the UNROUNDED delta `d` while STORING the f16-rounded `d`, so a
// round-trip is not `q*d_stored` exactly. Q8_K uses `iscale = -127/max` keyed
// on the SIGNED extremum (not the absolute max), which is why its quants can
// reach -127 but its positive side is clamped by `MIN(127, v)`. Both are
// reproduced verbatim; "fixing" either would silently diverge from every
// llama.cpp vec_dot that consumes them.
#include <cmath>
#include <cstring>

#include "cpu_quant_blocks.h"
#include "vt/quant.h"

namespace vt::cpu {
namespace {

// ggml-quants.c:563 — round-to-nearest via the 2^23+2^22 magic constant. Kept
// bit-for-bit rather than replaced with `roundf`/`lrintf`: it rounds halfway
// cases to EVEN (the FPU's current mode), and Q8_K's quants depend on that
// exact tie behaviour to reproduce llama.cpp's blocks.
inline int NearestInt(float fval) {
  // Upstream asserts fabsf(fval) <= 4194303.f; the caller's iscale scaling
  // keeps |fval| <= 127 here, well inside the representable window.
  float val = fval + 12582912.0f;
  int i = 0;
  std::memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

// ggml-quants.c:238 quantize_row_q8_0_ref (via ggml-cpu/quants.c:45).
void QuantizeRowQ8_0(const float* x, void* vy, int64_t k) {
  VT_CHECK(k % kQK8_0 == 0,
           "quantize_row_q8_0: k must be a whole number of 32-element blocks");
  const int64_t nb = k / kQK8_0;
  BlockQ8_0* y = static_cast<BlockQ8_0*>(vy);

  for (int64_t i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max

    for (int j = 0; j < kQK8_0; j++) {
      const float v = x[i * kQK8_0 + j];
      // Upstream's MAX() is a ternary, NOT fmaxf: they differ on NaN input
      // (ternary propagates the incumbent, fmaxf picks the non-NaN operand).
      // Kept as the ternary so a NaN activation degrades identically.
      const float av = std::fabs(v);
      amax = amax > av ? amax : av;
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = F32ToF16(d);

    for (int j = 0; j < kQK8_0; ++j) {
      const float x0 = x[i * kQK8_0 + j] * id;

      y[i].qs[j] = static_cast<int8_t>(std::roundf(x0));
    }
  }
}

// ggml-quants.c:2696 quantize_row_q8_K_ref (via ggml-cpu/quants.c:117).
// `bsums[j]` (the sum of each group of 16 quants) is not an optimization
// detail: the Q4_K/Q5_K vec_dots consume it to subtract the block minimum in
// one pass, so an incorrect bsum corrupts those dots while leaving `qs`
// looking perfectly fine.
void QuantizeRowQ8_K(const float* x, void* vy, int64_t k) {
  VT_CHECK(k % kQK_K == 0,
           "quantize_row_q8_K: k must be a whole number of 256-element blocks");
  const int64_t nb = k / kQK_K;
  BlockQ8_K* y = static_cast<BlockQ8_K*>(vy);

  for (int64_t i = 0; i < nb; i++) {
    float max = 0;
    float amax = 0;
    for (int j = 0; j < kQK_K; ++j) {
      float ax = std::fabs(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (!amax) {
      y[i].d = 0;
      std::memset(y[i].qs, 0, kQK_K);
      // Upstream leaves `bsums` untouched on the all-zero path because its
      // scratch is calloc'd; ours is raw, so zero it explicitly — the values
      // are all-zero either way, this only removes the uninitialized read.
      std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
      x += kQK_K;
      continue;
    }
    // const float iscale = -128.f/max;
    // We need this change for IQ2_XXS, else the AVX implementation becomes
    // very awkward
    const float iscale = -127.0f / max;
    for (int j = 0; j < kQK_K; ++j) {
      int v = NearestInt(iscale * x[j]);
      y[i].qs[j] = static_cast<int8_t>(v < 127 ? v : 127);
    }
    for (int j = 0; j < kQK_K / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) {
        sum += y[i].qs[j * 16 + ii];
      }
      y[i].bsums[j] = static_cast<int16_t>(sum);
    }
    y[i].d = 1 / iscale;
    x += kQK_K;
  }
}

}  // namespace

FromFloatFn BlockFromFloat(DType dtype) {
  switch (dtype) {
    case DType::kQ8_0: return &QuantizeRowQ8_0;
    case DType::kQ8_K: return &QuantizeRowQ8_K;
    default:
      // Every other block dtype is WEIGHT-side only in this project: GGUF
      // files carry them, but nothing quantizes an activation into them, so
      // upstream's `from_float` for them (the k-quant encoders in
      // ggml-quants.c) is deliberately not ported. See spec § Scope.
      return nullptr;
  }
}

size_t QuantActRowBytes(DType weight_dtype, int64_t k) {
  const DType act = QuantTraits(weight_dtype).vec_dot_type;
  VT_CHECK(IsBlockQuant(act),
           "QuantActRowBytes: weight dtype has no block activation encoding");
  // ggml_row_size(vec_dot_type, k) — the per-row stride upstream uses when it
  // lays src1 out in `wdata` (ggml-cpu.c:1313-1349). This THROWS when k is not
  // a whole number of activation blocks, which is the fail-loud guard for the
  // K-quants' 256-element granularity.
  return RowSizeBytes(act, k);
}

size_t QuantActScratchBytes(DType weight_dtype, int64_t rows, int64_t k) {
  VT_CHECK(rows >= 0, "QuantActScratchBytes: negative row count");
  // Mirrors ggml_graph_plan's mul_mat wdata sizing (ggml-cpu.c:2752-2980):
  // one quantized row per src1 row, contiguous, no padding between rows.
  return static_cast<size_t>(rows) * QuantActRowBytes(weight_dtype, k);
}

}  // namespace vt::cpu
