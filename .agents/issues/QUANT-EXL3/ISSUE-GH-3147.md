ID: ISSUE-GH-3147
Title: EXL3 GEMM shape discrimination test falsely fails for shapes 2 and 3 on GPU
Row: QUANT-EXL3
State: CLOSED
Kind: UNKNOWN
GitHub: 3147
Mirror: SYNCED
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `QUANT-EXL3`
>
> The "every shape in the kernel table is FORCED" test case in `test_exl3_gemm.cpp` has a discrimination check (line 891) that asserts every pair of shapes produces at least one differing output byte. This check fails on GPU for shapes 2 and 3, which produce byte-identical output.
>
> This is correct behavior, not a bug. Shapes 2 and 3 share the same `(tile_k=32, sh_stages=4, frag_stages=3)` computation signature — they differ only in `tile_n` (128 vs 256). `tile_n` determines how many output columns each thread block computes (parallelization), but does not affect the per-element K-accumulation order, which is determined by `tile_k`. Two shapes with the same `tile_k`, `sh_stages`, and `frag_stages` produce identical per-element results regardless of `tile_n`.
>
> The upstream exllamav3 shape table (`exl3_kernel_map.cuh:53-56`) confirms this: SHAPE_2 and SHAPE_3 have the same tile_k=32 and pipeline stages, differing only in tile_n.
>
> The test was committed on 2026-09-03 (commit 39f29b808) but was never run on a GPU before, so the false failure was not caught until the EXL3 reconstruct work brought the test binary to dgx:gpu0.
>
> The refusal check (lines 897-933) already proves that `force_shape_idx` is respected: forcing shape 4 at n=768 is correctly rejected because shape 4's tile_n=512 does not divide 768. If `force_shape_idx` were ignored, the launcher would select a compatible shape and return numbers instead of throwing.
>
> Fix: skip the discrimination check for shape pairs that share the same `(tile_k, sh_stages, frag_stages)` computation signature, since identical output is expected for those pairs.

## Resolution

Fixed by PR #3150: shape discrimination relaxed to rel_rms tolerance
