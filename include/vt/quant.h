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
// Population status (this is CIQ work row G1, the skeleton):
//   to_float      — POPULATED for all six executable weight types (G1).
//   vec_dot       — nullptr until G3 ports the six generic `ggml_vec_dot_*`.
//   from_float    — nullptr until G2 ports quantize_row_q8_0/q8_K.
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

// Traits for a block-quantized dtype; throws for any other dtype and for a
// block dtype this table does not (yet) carry.
const QuantTypeTraits& QuantTraits(DType dtype);

// True when `dtype` has a traits row AND that row can currently execute a
// quantized dot (vec_dot + from_float both present). Until G2/G3 land this is
// false for every type, which is exactly what routes `kMatmulBTQuant` to the
// generic dequant-composite fallback.
bool HasQuantDotKernel(DType dtype);

}  // namespace vt::cpu
