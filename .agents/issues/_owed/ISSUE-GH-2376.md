ID: ISSUE-GH-2376
Title: vt CPU attention: the per-element dtype dispatch is 64.6% of the kernel, and 219 call sites share the defect
Row: -
State: OPEN
Kind: UNKNOWN
GitHub: 2376
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `VT-CPU-ELEM-DISPATCH`
>
> `vt::AttentionCross` and `vt::Attention` spend most of their CPU time resolving
> an element type and a byte offset, not doing arithmetic.
>
> Both CPU kernels (`src/vt/cpu/cpu_ops.cpp`) read every operand element through
> `LoadF32`, which switches on `t.dtype` and multiplies the offset by
> `vt::SizeOf(t.dtype)` **once per element**. `SizeOf` was defined out of line in
> `src/vt/dtype.cpp` and the build enables no LTO — `CMakeLists.txt` sets no
> `INTERPROCEDURAL_OPTIMIZATION` and passes no `-flto` — so it was a cross-
> translation-unit call per element. `AttentionCrossKernel`'s score loop calls
> `LoadF32` twice per element, around a body that is one multiply and one add.
>
> A `perf` profile of `vt::AttentionCross` alone, at the LTX-2.5 connector's
> video shape (Tq=S=1024, H=32, D=128, f32), on this x86-64 AVX-512 devbox:
>
> | symbol | self |
> |---|---:|
> | `LoadF32(Tensor const&, long)` | 36.14% |
> | `vt::SizeOf(vt::DType)` | 28.41% |
> | `AttentionCrossKernel(...)::{lambda(long, long)#1}` | 15.60% |
>
> 64.6% of the kernel's own CPU time resolves an element type and an address; the
> arithmetic and the softmax are the 15.60% line. The kernel runs at ~3.6 GFLOP/s
> on 20 threads.
>
> This is not an LTX-2.5 problem. `LoadF32`/`StoreF32` appear at **219 call sites
> across 64 kernels** in `src/vt/cpu/cpu_ops.cpp` and `src/vt/cpu/cpu_paged_attn.cpp`;
> attention is only where it was measured. Every CPU model path pays it.
>
> The repair is the transformation `MatmulOneChunk` already applies against
> `MatmulOneChunkRef` in the same file: resolve the element type once, outside the
> loops, and walk typed pointers. It touches no output's accumulation order, so it
> is bit-exact rather than close.
>
> Measured on `dgx`-class hardware by the predecessor row
> (`.agents/specs/ltx25-connector-gemm.md`, `## Owed`), and reproduced here.
>

## Resolution

-
