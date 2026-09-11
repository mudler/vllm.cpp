ID: ISSUE-GH-2107
Title: TT eager decode spends the wall on host staging, not device kernels: cache resolved handles, bulk the EnsureDevice2D element loop, batch per-layer staging
Row: BACKEND-TENSTORRENT-QWEN35
State: OPEN
Kind: perf
GitHub: 2107
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-27
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT-QWEN35`
>
> Owning row: `BACKEND-TENSTORRENT-QWEN35`. Follow-up to the #1715 profile lever, which measured the wall before naming a fix.
>
> ## Measured (2026-08-27, P150, tree `a0db99b31`, `docs/bench-evidence/tt-qwen35-eager-profile-20260827.log`)
>
> One eager Qwen3.5-0.8B decode step runs at **0.104 tok/s** and the profile names **no device kernel** in any ranked frame. The wall is host dispatch around the TT GEMM:
>
> - `vt::Tensor::Numel()` 27.09% of samples — per-element shape math inside the staging loop
> - `EnsureDevice2D`→`MatmulBTKernel` = 24.10% of the call graph, under `MatmulBf16D` 19.33%, fed by `DenseMlpBlock` 9.44% and `DenseLogitsF32D` 9.32%
> - repeated TT-Metal context/UMD discovery ~12% (`MetalContext::instance` 5.81%, `Cluster::get_chip` 3.27%, `get_closest_mmio_capable_chip` 2.95%, `DeviceManager::get_active_device` 2.49%)
> - CPU threadpool spin 11.4%, `memcpy` 7.04%, `bfloat16::from_float` 2.62%
> - Control: the same leg without the TT backend runs at 7.521 tok/s (~73x faster)
>
> ## Root cause in code
>
> `EnsureDevice2D` (`src/vt/tenstorrent/tenstorrent_ops.cpp:434`) stages **element-by-element** through an f32 intermediate:
>
> ```cpp
> std::vector<float> host(rows * cols);
> for (int64_t i = 0; i < t.Numel(); ++i)
>   host[i] = LoadElemF32(t, i);   // per-element stride/dtype dispatch
> ttnn::Tensor dev = UploadRows(host.data(), rows, cols, device);  // f32->bf16 again
> ```
>
> plus up to four `FindSlot` mutex acquisitions per call, and every upload path re-resolves TT-Metal context/device/chip handles.
>
> ## The three levers (in the recorded next gate)
>
> 1. Cache resolved context/device/chip handles across calls.
> 2. Hoist `Numel()`/shape math out of per-call staging; bulk the element loop.
> 3. Batch per-layer staging.
>
> ## Constraints
>
> - Numerics may not move: the sacred golden pair must stay 16/16 (`tests/parity/goldens/qwen35_greedy_0_8b`), full TT suite green.
> - bf16 host masters must not round-trip through f32 where a bulk copy suffices; f32 stays where the model path declares it (logits GEMM f32 output is the annotated exception).
> - Captured tracing stays blocked behind #1625; this is eager-side only.

## Resolution

-
