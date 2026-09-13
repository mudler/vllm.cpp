ID: ISSUE-GH-3062
Title: fix(BACKEND-ROCM): isolate split-sampling scratch by queue and device
Row: BACKEND-ROCM
State: CLOSED
Kind: UNKNOWN
GitHub: 3062
Mirror: SYNCED
Availability: FULL
Created: 2026-09-08
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> PR #3010 stores split-sampling partial buffers and capacity in process-wide statics. Concurrent queues can overwrite another queue's partials between the two kernels. Calls on another device can reuse allocations from the first device. Host pointer and capacity updates also race.
>
> The repair belongs in #3010. Use the existing device-and-stream scratch ownership mechanism and preserve captured graph lifetimes. Add regression coverage for overlapping queues and allocation ownership. Retaining allocations for captured graphs must follow the existing backend lifetime contract.
>
> Acceptance requires focused tests, an independent mutation review, and the operator's ROCm gate. CPU-only skips do not satisfy the device gate.
>

## Resolution

Fixed by PR #3010: sampling block widen and split-phase implementation.
