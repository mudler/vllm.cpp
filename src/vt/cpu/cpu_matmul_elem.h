// Elementwise (non-block-quantized) CPU GEMM micro-kernels.
//
// Upstream anchors (llama.cpp local fork @ 237ad9b96 / b9892):
//   ggml/src/ggml-cpu/vec.cpp:139  ggml_vec_dot_bf16  — bf16 widen-by-shift +
//                                  multi-accumulator mul/add SIMD dot
//   ggml/src/ggml-cpu/vec.cpp:264  ggml_vec_dot_f16   — f16 widen + SIMD dot
//   ggml/src/ggml-cpu/vec.h:72-73  the declarations mul_mat dispatches to
//   ggml/src/ggml-cpu/ggml-cpu.c:1155-1243 mul_mat chunk worker (our caller)
//
// RECORDED DEVIATION FROM UPSTREAM (deliberate, and the whole point of this
// file): ggml vectorizes each dot product ALONG K and finishes with a
// horizontal reduce, which REASSOCIATES the sum. We instead vectorize ACROSS
// OUTPUT COLUMNS (16 weight rows / output lanes at a time), so every output
// element keeps the SAME strictly sequential f32 accumulation over p that the
// scalar kernel had. That makes the SIMD path BIT-IDENTICAL to the scalar
// reference (asserted in tests/vt/test_ops_matmul_elem.cpp), which the CUDA-
// captured goldens and the run-to-run/thread-count determinism contract both
// require. The SIMD primitives themselves are ggml's: bf16 widening is the
// shift-left-16 of vec.cpp:172 (`_mm256_slli_epi32(_mm256_cvtepu16_epi32(..),16)`)
// and f16 widening is the hardware convert ggml uses via GGML_F16_VEC_LOAD.
// Products use separate mul+add (never FMA), matching ggml's own
// `_mm256_add_ps(_mm256_mul_ps(..))` shape and our global -ffp-contract=off.
#pragma once

#include <cstdint>

#include "vt/dtype.h"

namespace vt::cpu {

// Number of output columns a micro-kernel produces per call. 16 matches the
// mul_mat chunk worker's blck_0 tile (ggml-cpu.c:1192-1194) and gives 16
// independent f32 accumulator chains, which is what removes the serial
// FP-add latency bottleneck of the one-accumulator scalar loop.
inline constexpr int kElemLanes = 16;

// The three elementwise dtypes a GEMM operand may have.
enum class ElemKind : int { kF32 = 0, kF16 = 1, kBF16 = 2, kCount = 3 };

// Returns false for dtypes with no elementwise micro-kernel (i8/i32/i64 and
// every block-quantized encoding).
bool ElemKindOf(DType dt, ElemKind* out);

// acc[l] = sum_p af[p] * B[l * k + p]     (weight rows contiguous, [N,K])
using ElemBt16Fn = void (*)(const float* af, const void* b, int64_t k, float* acc);
// acc[l] = sum_p af[p] * B[p * n + l]     (weight rows strided by n, [K,N])
using ElemNk16Fn = void (*)(const float* af, const void* b, int64_t k, int64_t n, float* acc);
// The same [N,K] kernel over `mr` ACTIVATION ROWS at once (acc is mr*16, row
// major). The weight loads and — on the SIMD tiers — the 4x4 transpose that
// turns them into per-p across-column vectors are the dominant cost at
// prefill shapes, and they are IDENTICAL for every activation row, so hoisting
// them out of an M loop is pure amortization. It does not touch any output's
// accumulation order, so it is bit-exact like everything else here.
using ElemBtMFn = void (*)(const float* af, int64_t a_stride, const void* b, int64_t k,
                           float* acc);

// The same [K,N] kernel over `mr` ACTIVATION ROWS at once (acc is mr*16, row
// major). The [K,N] orientation needs no transpose, so what M blocking
// amortizes here is the WEIGHT LOAD itself: one 16-wide load per p is reused by
// every activation row instead of being re-read once per row. Like ElemBtMFn it
// touches no output's accumulation order (lane l still accumulates output l
// over p in strict increasing order), so it is bit-exact.
//
// Before this existed, cpu_ops.cpp forced mr=1 on the [K,N] path, so a
// 131-row activation re-read the whole weight 131 times.
using ElemNkMFn = void (*)(const float* af, int64_t a_stride, const void* b, int64_t k,
                           int64_t n, float* acc);

struct ElemGemmTierTable {
  ElemBt16Fn bt[static_cast<int>(ElemKind::kCount)];
  ElemNk16Fn nk[static_cast<int>(ElemKind::kCount)];
  ElemBtMFn btm[static_cast<int>(ElemKind::kCount)];
  ElemNkMFn nkm[static_cast<int>(ElemKind::kCount)];
  int mr;  // activation rows per btm/nkm call (1 = no M blocking)
  const char* name;
};

// Wide-x86 tiers (KERNEL-GEMM-CPU-ELEM-X86WIDE). Each lives in its own TU built
// with that ISA's flags, mirroring llama.cpp's per-ISA CPU build; they are
// always COMPILED on x86-64 and only ever ENTERED when the runtime probe in
// BuildTier() says the CPU supports them. Defined in cpu_matmul_elem_avx2.cpp.
#if defined(__x86_64__) || defined(_M_X64)
void FillAvx2Tier(ElemGemmTierTable* t);
void FillAvx512Tier(ElemGemmTierTable* t);  // cpu_matmul_elem_avx512.cpp
#endif

// The [N,K] -> [K,N] load-time repack entry points are declared in the PUBLIC
// vt/quant.h (the loader needs them and this header is private).

// Selected ONCE per process: compile-time ISA plus a runtime feature probe
// (mirrors ggml's `ggml_cpu_has_*` discipline, ggml-cpu.c). The portable
// tier is always built and is what CI exercises.
const ElemGemmTierTable& ElemGemmTier();

// VT_CPU_MATMUL_TIER selects the tier for a SAME-BINARY A/B:
//   "ref"      — the historical one-accumulator scalar chunk kernel
//   "portable" — tier 0 only (no arch SIMD)
//   "sse2", "sse2+f16c", "avx2", "avx512" — exact x86 tier; an unavailable
//                  tier fails closed after CPUID + XCR0 validation
//   unset      — the best tier this CPU probes into (production default)
const char* ElemGemmTierName();

// True when VT_CPU_MATMUL_TIER=ref: the caller must run the historical scalar
// chunk kernel. Exists so the benchmark A/B and the bit-exactness test can
// reach the pre-change path without rebuilding.
bool ElemGemmUseRef();

// Widen `n` contiguous elements of `src` (dtype `dt`) into `dst` as f32.
// Bit-identical to a per-element LoadF32.
void WidenRowToF32(DType dt, const void* src, int64_t n, float* dst);

// The output counterpart: narrow `n` f32 values into `dst`'s storage dtype with
// ONE branch for the row instead of one per element. Same supported set and the
// same round-to-nearest-even (`F32ToBF16`) / `F32ToF16` as a per-element
// `StoreF32`, so it is bit-identical to one.
//
// Row VT-CPU-ELEM-SURVEY, .agents/specs/vt-cpu-elem-survey.md. It lives here,
// beside `WidenRowToF32`, because that is where a caller already looks for the
// pair; `cpu_paged_attn.cpp` carries a file-private `StoreRowF32` that predates
// it and differs only in its refusal MESSAGE, which is why folding the two is a
// separate row rather than a silent behaviour change (see that spec's `## Owed`).
void NarrowRowFromF32(DType dt, void* dst, int64_t n, const float* src);

}  // namespace vt::cpu
