# Unaligned scalar-tail loads in the elementwise CPU GEMM

## Now

Fixing the address/undefined-sanitizer finding in the elementwise CPU GEMM
scalar tails. The typed `b[...]` dereference in every tier's ragged-K tail
forms a misaligned lvalue when the weight buffer starts at an odd byte
address, which is UB under UBSan and crashes the sanitize-cpu lane.

## Issue

#2908 — AVX2 elementwise GEMM performs a misaligned typed BF16 load under
UBSan.

## Scope

Thirteen scalar-tail load sites across four files:

- `src/vt/cpu/cpu_matmul_elem.cpp` lines 63, 79, 102, 186, 232, 331, 377 —
  portable, Neon, and SSE2 tiers. Includes both `b[expr]` and `row[l]`
  dereferences (the `nk`/`nkm` paths).
- `src/vt/cpu/cpu_matmul_elem_avx2.cpp` lines 157, 231 — AVX2 tier.
- `src/vt/cpu/cpu_matmul_elem_avx512.cpp` lines 121, 177 — AVX-512 tier.
- `src/vt/cpu/cpu_matmul_elem_f16c.cpp` lines 52, 84 — F16C tier.

Each site reads a typed `T*` lvalue (`uint16_t` for f16/bf16, `float` for
f32) through `b[expr]` or `row[l]`. When the base pointer `bv` is at an odd
byte address, constructing the typed lvalue is UB before the converter widens
it.

## Design

Replace each typed `b[expr]` or `row[l]` dereference with
`LoadUnaligned<T>(b + expr)` or `LoadUnaligned<T>(row + l)` from
`include/vt/unaligned.h`, which uses `memcpy` and is always safe.
`cpu_matmul_elem.cpp` already includes `vt/unaligned.h`; add the include to
the AVX2, AVX-512, and F16C files.

The vectorized loops are already safe: they use `_mm_loadu_ps`,
`_mm_loadu_si128`, `vld1q_f32`, and similar unaligned-tolerant intrinsics.
Only the scalar tails need the fix.

The fix is numerically invisible: `LoadUnaligned<T>` copies the same bytes
the typed dereference would have read, so the byte-identity contract is
preserved.

## Test

Add a test case to `tests/vt/test_ops_matmul_elem.cpp` that calls the kernel
function pointers from `ElemGemmTier()` directly with a one-byte-front-padded
weight buffer (odd address) and a ragged K so the scalar tail executes. The
high-level `MatmulBT` path cannot reach the bug because `Tensor::Contiguous`
copies the weight into an aligned buffer before the kernel sees it. Calling
the tier-table function pointers directly is the only way to deliver an odd
pointer to the kernel without a loader that produces one. The test covers
both `bt` and `nk` kernels for f16 and bf16. Under UBSan, the old code
crashes; the new code passes.

## Gates

- The existing byte-identity suite must stay green (memcmp, not Approx).
- The new odd-offset test must pass under a normal build and under UBSan.
- The full sanitize-cpu (address,undefined) lane must pass.

## Stop conditions

If `LoadUnaligned` changes the generated code in a way that breaks
byte-identity on any tier, stop and use `memcpy` inline instead.
