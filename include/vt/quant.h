// vt:: block-quant type traits — 1:1 mirror of llama.cpp's `type_traits_cpu[]`
// table, ported from the local fork @ 237ad9b96
// `ggml/src/ggml-cpu/ggml-cpu.c:211-406` (the table itself; Q4_0 :230-239,
// Q8_0 :262-271, Q3_K :295-300, Q4_K :301-310, Q5_K :311-316, Q6_K :317-326).
//
// The table shape is kept identical to upstream's so future llama.cpp diffs
// port mechanically: one row per block dtype carrying `{from_float, vec_dot,
// vec_dot_type, nrows}`. `to_float` is upstream's `ggml_type_traits.to_float`
// (`ggml/src/ggml.c`), folded into this one CPU-side table because vt:: has no
// separate device-neutral trait registry — a recorded deviation, nothing else
// changes.
//
// Population status (G1 skeleton + G2 activation quant + G3 tier-0 vec_dot):
//   to_float      — POPULATED for all six executable weight types (G1).
//   vec_dot       — POPULATED for all six by the portable tier-0 kernels in
//                   cpu_quant_dot.cpp (G3). The x86 AVX2 (G5) and Arm
//                   NEON/i8mm (G6) tiers replace these entries per-ISA later;
//                   the generic tier is always built and is what CI runs.
//   from_float    — POPULATED for the two ACTIVATION encodings Q8_0 and Q8_K
//                   in cpu_quant_act.cpp (G2). It stays nullptr for the
//                   weight-only encodings: nothing in this project quantizes
//                   an activation INTO a k-quant, so upstream's k-quant
//                   encoders are deliberately not ported.
//   vec_dot_type  — POPULATED (the dispatch fact G1 owns).
//   nrows         — POPULATED (1 everywhere on the generic tier; the
//                   `__ARM_FEATURE_MATMUL_INT8` nrows==2 rows arrive with the
//                   i8mm mmla kernels in G6, together with the boundary guards
//                   at ggml-cpu.c:1426-1433).
#pragma once

#include <cstddef>
#include <cstdint>

#include "vt/dtype.h"

namespace vt::cpu {

// Upstream `ggml_from_float_t` (ggml-cpu-traits / ggml-impl.h): quantize `k`
// f32 activations into the block encoding at `y`.
using FromFloatFn = void (*)(const float* x, void* y, int64_t k);

// Upstream `ggml_to_float_t`: decode `k` elements of packed blocks into f32.
using ToFloatFn = void (*)(const void* x, float* y, int64_t k);

// Upstream `ggml_vec_dot_t` (ggml-impl.h): `nrc` row-dots of length `n` between
// weight blocks `x` (row stride `bx` BYTES) and activation blocks `y` (row
// stride `by` BYTES), written to `s` with row stride `bs` BYTES.
using VecDotFn = void (*)(int n, float* s, size_t bs, const void* x, size_t bx,
                          const void* y, size_t by, int nrc);

struct QuantTypeTraits {
  FromFloatFn from_float = nullptr;
  ToFloatFn to_float = nullptr;
  VecDotFn vec_dot = nullptr;
  // The activation encoding this weight type is dotted against: Q8_0 for the
  // legacy 32-element types, Q8_K for the K-quants (ggml-cpu.c:230-326).
  DType vec_dot_type = DType::kF32;
  // Rows the vec_dot kernel consumes per call (2 only on the i8mm mmla tier).
  int nrows = 1;
};

// The `to_float` decoder for a block dtype (nullptr for elementwise dtypes).
// Implemented in cpu_quant_dequant.cpp.
ToFloatFn BlockToFloat(DType dtype);

// The `from_float` activation quantizer for a block dtype — non-null only for
// the two `vec_dot_type` encodings Q8_0 and Q8_K. Implemented in
// cpu_quant_act.cpp (ports of quantize_row_q8_0_ref / quantize_row_q8_K_ref).
FromFloatFn BlockFromFloat(DType dtype);

// The tier-0 generic `vec_dot` for a block WEIGHT dtype (nullptr for Q8_K,
// which is activation-only, and for elementwise dtypes). Implemented in
// cpu_quant_dot.cpp.
VecDotFn BlockVecDot(DType dtype);

// Bytes one quantized ACTIVATION row occupies for a given weight dtype, i.e.
// `ggml_row_size(vec_dot_type(weight_dtype), k)`. Throws when `k` is not a
// whole number of activation blocks (256 for the K-quants, 32 otherwise) —
// the fail-loud guard against a ragged-K GEMM silently mis-striding scratch.
size_t QuantActRowBytes(DType weight_dtype, int64_t k);

// Total scratch for quantizing `rows` activation rows of length `k` ahead of a
// `kMatmulBTQuant` against `weight_dtype`. Mirrors the mul_mat `wdata` sizing
// in `ggml_graph_plan` (ggml-cpu.c:2752-2980): contiguous rows, no padding.
size_t QuantActScratchBytes(DType weight_dtype, int64_t rows, int64_t k);

// Traits for a block-quantized dtype; throws for any other dtype and for a
// block dtype this table does not (yet) carry.
const QuantTypeTraits& QuantTraits(DType dtype);

// True when `dtype` has a traits row AND that row can currently execute a
// quantized dot: it needs BOTH its own `vec_dot` and a `from_float` on the
// activation encoding its `vec_dot_type` names. True for the six executable
// weight types since G2+G3; false for Q8_K (activation-only, no vec_dot),
// which is what keeps it on the generic dequant-composite fallback.
bool HasQuantDotKernel(DType dtype);

}  // namespace vt::cpu
