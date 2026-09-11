ID: ISSUE-GH-2442
Title: The EXL3 routed experts cannot run on a CUDA queue, so the 44-47 tok/s target is blocked on the device-resident tower, not the drafter
Row: MODEL-DSV4-EXL3
State: OPEN
Kind: UNKNOWN
GitHub: 2442
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-DSV4-EXL3`
>
> The 44-47 tok/s DeepSeek-V4-Flash target cannot be reached on the EXL3 arm as it
> stands, and the reason is not the drafter.
>
> `Exl3Linear` (`src/vllm/model_executor/models/deepseek_v4.cpp:1295-1302`) refuses
> a non-CPU queue unless `Backend::DeviceMemoryIsHostAddressable()` is true. W1b
> copies each TP1-coalesced linear into HOST owner buffers, so a device kernel would
> have to dereference host pointers. `CudaBackend` answers **false** to that
> predicate and does so deliberately even on GB10: `src/vt/cuda/cuda_backend.cu:354-391`
> pins it with a `static_assert` and records why -- a `cudaMalloc` pointer is not
> host-dereferenceable even where `UnifiedMemory()` is true, and reading the wider
> predicate is what SIGSEGV'd the reference tier (#844, #1435).
>
> So on this arm all 216 routed experts execute on a CPU queue. Speculation cannot
> recover that: it multiplies the step rate by accepted tokens per step, and 2.64x
> a CPU-bound MoE step is still CPU-bound.
>
> This makes the already-owed "Real-checkpoint residency for the coalesced tower"
> (`.agents/specs/model-dsv4-exl3.md`, W2) the blocking item for the throughput
> goal, ahead of `DSV4-DSPARK-DRAFTER` W-6. Filing it under its own number because
> it now gates a target rather than a nicety, and because a throughput measurement
> taken before it lands would measure the wrong thing -- and would read as "the
> drafter did not help".
>
> Sizing, from the load measured 2026-08-31: the tower is 81.952 GiB coalesced at
> TP1 plus a 15.726 GiB carried host tower, 97.68 GiB total at a 111 GiB peak RSS.
> A device-resident destination has to fit the first of those in device allocations
> on a ~119 GiB GB10 alongside KV and activations.
>
> Derived from source, not measured. What is measured is the load and its residency.
>

## Resolution

-
