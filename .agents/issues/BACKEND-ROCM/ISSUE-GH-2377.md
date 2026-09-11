ID: ISSUE-GH-2377
Title: ROCm: DFlash2 reaches host fallbacks and hangs gfx1151
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 2377
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> Qwen3.8-27B with the published DFlash2 safetensors draft does not complete on `strix:gpu0` (`gfx1151`, ROCm 7.2.4).
>
> The plain target completes 32 greedy tokens. The DFlash2 arm reaches three CPU reference-tier ops and then the GPU reports a hang:
>
> ```text
> [vt reference-tier] op=IndexSelect device=rocm has NO native kernel
> [vt reference-tier] op=IndexCopy device=rocm has NO native kernel
> [vt reference-tier] op=DFlashGroupedConv device=rocm has NO native kernel
> HW Exception by GPU node-1 ... reason :GPU Hang
> ```
>
> The DFlash2 process exits 134 before its first completion. `test_dflash2_runner_reach` is also red on the same HIP build. Its D9 case reports zero differing blocks after changing the selector scalars. One repeated run also failed the W11 call-count and output-size checks.
>
> `GetOp` calls `Backend::FlushPending()` before a CPU reference-tier kernel reads managed device memory. `RocmBackend` does not override that method, so it inherits the no-op. Metal and Vulkan both override it. This is the first root-cause candidate for the hang and the unstable runner output.
>
> A safe run is not enough for a performance result. The production DFlash2 chain also needs native ROCm implementations for every op that the full provider census reports. The known static set includes row gather/scatter, grouped convolution, top-k pairs, selector edges, path walk, and the block-attention variants.
>
> Acceptance:
>
> - A device test proves that `RocmBackend::FlushPending()` drains queued HIP work before a host fallback.
> - The real Qwen3.8-27B plus DFlash2 arm completes greedy generation without a GPU hang.
> - The focused DFlash2 suites pass on `gfx1151`.
> - `VT_OP_PROVIDER_STATS=1` reports zero CPU reference-tier hits for the measured DFlash2 workload.
> - A repeated local-checkpoint benchmark records clock state and warm-leg throughput.

## Resolution

-
