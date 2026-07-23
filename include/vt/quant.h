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

// The Arm i8mm (mmla) `nrc == 2` `vec_dot` for a block WEIGHT dtype — QUANT-
// GGUF-CIQ-GEMM work row G6 (cpu_quant_dot_arm.cpp). Non-null ONLY when the
// process runs on i8mm-capable aarch64 (compile-time `__ARM_FEATURE_MATMUL_INT8`
// AND runtime `HWCAP2_I8MM`) AND the dtype is one of the four encodings upstream
// gives an mmla path (Q8_0, Q4_0, Q4_K, Q6_K). Returns nullptr everywhere else —
// on any other CPU, when `VT_CPU_QUANT_MMLA=0`, and for q3_K/q5_K (no upstream
// mmla) — so the caller falls back to the portable nrc==1 tier. A returned
// kernel produces a 2x2 output tile: it MUST be called with nrc==2, two
// consecutive weight rows (stride bx) and two consecutive activation rows
// (stride by), writing s[0]=(w0,a0), s[1]=(w1,a0), s[bs]=(w0,a1), s[bs+1]=(w1,a1).
VecDotFn QuantMmlaVecDot(DType dtype);

// True when the Arm i8mm mmla tier is live in this process (i8mm probed present
// and not defeated by VT_CPU_QUANT_MMLA). Always false off i8mm-capable aarch64.
bool QuantMmlaActive();

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

// --- CIQ G7: repack-at-load for the q8_0 quant GEMM -----------------------
//
// llama.cpp repacks a quantized WEIGHT once at load into a SIMD/cache-friendly
// interleave so the GEMM inner loop reads contiguous, pre-shuffled blocks with
// no in-register row shuffles (repack.cpp @ 237ad9b96). On GB10 (aarch64 NEON +
// i8mm) `ggml_repack_get_optimal_repack_type` selects `q8_0_4x8_q8_0`
// (block_q8_0x4, nrows_interleaved=4, interleave_block=8) for a q8_0 weight
// with ne[1] % 4 == 0 (repack.cpp:4683-4695). We mirror exactly that tier.
//
// The transform is a pure BYTE PERMUTATION (the quant values and fp16 deltas
// are untouched) and the gemm/gemv fold the per-block scale in the SAME order
// as the tier-0 / mmla path with a NON-FUSED multiply-add, so the repacked GEMM
// is BIT-IDENTICAL to `kMatmulBTQuant`'s non-repacked output (proven by the
// memcmp round-trip test). Only q8_0 is repacked in G7 (the profile's
// kMatmulBTQuant is q8_0); the k-quants keep the mmla tier.

// True when the i8mm repack tier is LIVE in this process: compiled for aarch64
// with i8mm, `HWCAP2_I8MM` probed present, and not disabled by
// `VT_CPU_QUANT_REPACK=0|off|false`. Always false off i8mm-capable aarch64, so
// the loader never repacks and every consumer keeps the portable/mmla path.
bool QuantRepackActive();

// True when a weight of (dtype, N=out, K=in) is repack-eligible on THIS process:
// QuantRepackActive() AND dtype == kQ8_0 AND N % 4 == 0 AND K % 32 == 0 (the
// exact `ne[1] % 4 == 0`, `ne[0] % 8 == 0` guard of repack_q8_0_to_q8_0_4_bl;
// K % 32 subsumes the K % 8 one). A weight that fails it stays plain and takes
// the normal path — correct, just unrepacked.
bool QuantRepackEligible(DType weight_dtype, int64_t n, int64_t k);

// Repack a [N,K] q8_0 weight buffer IN PLACE into the block_q8_0x4 interleave.
// `blocks` holds N*(K/32) plain BlockQ8_0 on entry and N/4 groups of (K/32)
// BlockQ8_0x4 on return (same total bytes). Requires QuantRepackEligible.
void QuantRepackWeight(DType weight_dtype, uint8_t* blocks, int64_t n,
                       int64_t k);

}  // namespace vt::cpu

namespace vt {
struct Tensor;
namespace cpu {
// The repacked-weight GEMM dispatched by `kMatmulBTQuant` when `b.repacked`.
// out[M,N] = a[M,K] @ b[N,K]^T with `b` a q8_0 weight already repacked by
// QuantRepackWeight. Bit-identical to the non-repacked quant path: the i8mm
// gemm runs 4-row activation groups, the i8mm gemv the M=1 / leftover rows.
void QuantRepackMatmul(vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b);
}  // namespace cpu
}  // namespace vt
