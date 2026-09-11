ID: ISSUE-GH-2402
Title: Op provider: route runtime selection by full device
Row: BACKEND-ACCEL-PROVIDER
State: OPEN
Kind: UNKNOWN
GitHub: 2402
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ACCEL-PROVIDER`
>
> The backend registry is keyed by `Device{type,index}`, but the operation provider resolves runtime state by `DeviceType`. Every queue-bearing wrapper discards the index before `GetOp`.
>
> This is memory-unsafe for the portable CPU reference tier. If index 0 is host-addressable, it can install and cache `vt-cpu-ref` for the whole type. A discrete index can then select that host function over non-host-addressable memory. Fallback draining also calls backend index 0 rather than the queue's backend.
>
> The same runtime path has a publication race. `MaybeInstallReferenceTier` appends a provider after `main()` starts without synchronization. Two concurrent first misses can race `slot.count` and `slot.providers`.
>
> Fresh review of #2377 found the device-index defect. Independent adversarial analysis confirmed that synchronizing every GPU does not fix host-addressability. A shared selected pointer with separate relaxed `ref_selected` state is also insufficient.
>
> Acceptance:
>
> - Provider definitions remain type-keyed and immutable after static registration.
> - Runtime selection state, capability state, reference identity, and negative caches are keyed by the full `Device`.
> - Queue-bearing execution APIs carry `Queue::device` through normal and fallback resolution.
> - A CPU reference provider is a synthetic per-device candidate. Runtime code does not append it to the shared provider list.
> - Every reference invocation checks the exact backend's host-addressability and drains that exact backend.
> - A hardware-free two-index test covers both host-addressability polarities, resolution order, exact drain routing, and an already-warm reference selection.
> - A concurrent first-miss test passes under ThreadSanitizer and exposes one coherent selection.
> - Native provider ordering, disable behavior, statistics, and the cached native hot path retain their existing contracts.
> - Cached decline paths cannot bypass per-invocation reference eligibility or draining.

## Resolution

-
