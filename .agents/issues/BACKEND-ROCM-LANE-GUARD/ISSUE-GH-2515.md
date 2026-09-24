ID: ISSUE-GH-2515
Title: gfx1151 reports pageableMemoryAccess=0, so the host-slot expert lane can never serve on Strix Halo -- #2507's second condition is false, not satisfied
Row: BACKEND-ROCM-LANE-GUARD
State: OPEN
Kind: UNKNOWN
GitHub: 2515
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-01
Updated: 2026-09-01
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-LANE-GUARD`
>
> [#2507](https://github.com/mudler/vllm.cpp/issues/2507) states that the streamed-expert
> lane guard's other five conditions "are satisfied on this board", and specifically that
>
> > `host_memory_is_device_addressable()` reads a real probe
> > (`src/vllm/platforms/rocm.cpp:80-82`) and `gfx1151` is integrated with
> > `managedMemory=1 concurrentManagedAccess=1`
>
> **Those are the attributes of a different predicate.** `managedMemory` and
> `concurrentManagedAccess` are the inputs to `UseManagedAlloc`
> (`src/vt/rocm/rocm_backend.hip:142-144`). The predicate the lane guard reads is
>
> ```cpp
> bool HostMemoryIsDeviceAddressable(int index) noexcept {
>   const DeviceCaps caps = ProbeDevice(index);
>   return caps.valid && caps.integrated && caps.pageable_memory_access;
> }
> ```
> (`src/vt/rocm/rocm_backend.hip:474-481`, whose own comment says it is
> "deliberately NOT `unified_memory_`, which the registrar widens with
> `managed_alloc`".)
>
> `pageable_memory_access` on this board is **0**, and it was already measured
> twice before #2507 was filed:
>
> - `.agents/specs/rocm-glm53-dsa.md:29` — `| pageableMemoryAccess | 0 |`
> - `/mnt/nas_share/rc/glm53-rocm/out2/probe2.log:55` —
>   `managedMemory=1 pageableMemoryAccess=0 concurrentManagedAccess=1 integrated=1 warpSize=32`
>
> So `host_memory_is_device_addressable()` is **false** on `strix:gpu0`, and the
> lane guard's SECOND condition fails independently of its first.
>
> ## Why this is not merely a bookkeeping correction
>
> The same predicate gates the lane at RUNTIME. `ExpertSlice`
> (`src/vllm/model_executor/expert_stream_seam.cpp:431`) admits the lane on
> `cpu || p.host_memory_is_device_addressable()`, and falls through to the caller's
> resident slice otherwise — which stages the whole tower. The
> `HostExpertSlotStore` arena is an ordinary `std::vector<uint8_t>`
> (`expert_stream_seam.cpp:343`), i.e. exactly the pageable host pointer a device
> kernel here may not dereference.
>
> Forcing the lane on at load time would therefore not make the model stream. It
> would delete a correct refusal and restore the load-then-die shape #1123 exists
> to prevent. **The refusal is correct on this board for the streaming route**, and
> the host-slot lane is not the mechanism that can fit GLM-5.3 on Strix Halo.
>
> ## What this does not overturn
>
> The predicate defect #2507 identifies is real and is fixed under
> `BACKEND-ROCM-LANE-GUARD`: the lane guard was reading `needs_weight_staging()`
> while the refusal read `allocates_bounded_device_memory()`, so the two halves of
> one sum asked different questions. That repair is necessary and correct. It is
> simply not sufficient on `gfx1151`, and this issue records why, so the next
> reader does not conclude the fix failed.
>
> ## What would be needed
>
> A device-side expert slot store (`ENG-EXPERT-STREAM-DEVICE` W2) — a slot arena in
> device memory rather than a host `std::vector` — which is the one mechanism that
> does not depend on `pageable_memory_access`. That is a separate row.

## Resolution

-
