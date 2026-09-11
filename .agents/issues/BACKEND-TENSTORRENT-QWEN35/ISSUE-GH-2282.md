ID: ISSUE-GH-2282
Title: W7: eliminate staging restages — the residency state manufactures 7-8 full-tensor CQ writes per decode step
Row: BACKEND-TENSTORRENT-QWEN35
State: CLOSED
Kind: perf
GitHub: 2282
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-29
Updated: 2026-08-30
Closed: 2026-08-30

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> **Row**: `BACKEND-TENSTORRENT-QWEN35` (spec `.agents/specs/tenstorrent-qwen35.md`, successor to W6 — see its `## Evidence` W6 verdict).
>
> ## The measured premise
>
> W6's probe (#2273, closed as inexpressible for write BATCHING) counted the staging writes a real 3-token eager leg pays: **30 persistent-route restages ≈ 7-8 per step** — `[11,6144]`×17 (one stable activation-hidden slot) + `[176,128]`×13 (three rotating pool bases). Each restage is a full-tensor mesh-CQ write, and the per-op CQ tax (W5's trace: `Threadpool::PollForWork` 14.29%, `MetalContext::instance` 11.14%, `memcpy` 6.23%, `Cluster::get_chip` 5.90%, `read_cq_host_ptr` 5.27%+sub-slices) is charged on every one.
>
> The restages are manufactured by our own residency state, not by tt-metal:
>
> - `MarkHostWritten` (`src/vt/tenstorrent/tenstorrent_ops.cpp:5654`, callers `tenstorrent_backend.cpp:56,66,70`) marks a slot host-current/device-stale — including `OnScratchBlockAcquired`, where a retained DevicePool block is handed to a NEW tensor whose device bytes are garbage.
> - `CommitHost` (`tenstorrent_ops.cpp:1231`) drops the device shadow entirely whenever the host writes a tracked tensor in place.
> - The next device use then re-uploads the FULL tensor through the persistent arm (`ttnn::copy_to_device`, `tenstorrent_ops.cpp:561`).
>
> The probe's rotating-pool-base writes smell like avoidable restages: a slot reset via `OnScratchBlockAcquired` forces an upload even when the pending device op fully overwrites the buffer, and a host in-place write drops a shadow whose subsequent consumer might have been reachable without a full-tensor restage. Every restage that re-uploads bytes the device was about to overwrite — or that could be satisfied by a narrower transfer — is a wasted CQ op at the measured wall.
>
> ## The lever
>
> Eliminate staging writes, don't amortize them: make the residency state precise enough that a decode step stages each slot's bytes once — or zero times when the consumer overwrites the full buffer on device. Candidate directions (the implementer derives the actual mechanism from the code): a device-will-overwrite reservation for slots consumed by a full-overwrite op; narrowing what `CommitHost` drops; upload-on-write for host-written device-resident slots. Record the chosen mechanism and its restage semantics explicitly (W5 review aliasing awareness: same-geometry restage aliases the persistent buffer).
>
> ## Invariants
>
> - **Bit-identical staging** — the sacred golden pair stays 16/16 STRICT token-exact; the full TT suite stays green; this wave changes SPEED, never tokens.
> - `StagingStats` counters must already let a test observe the write count per step; if a new counter is needed it lands red-first.
> - Capture-unsafe host-write refusals keep their semantics; the f32-conversion arms keep their declared dtypes; the never-destroy slot rule (#1486) holds.
> - Production reachability: the change must move the real decode path's write count, observable from the production entry point, not only a hand-built test.
>
> ## Evidence owed on landing
>
> Same-method before/after on the P150 (identical 3-token leg, JIT-discard per arm, one lock hold): the write count per step (from the counters) AND the wall time, plus a fresh benchmark-record entry. A count that does not drop, or a wall that does not move, is a reported result — the attribution shifts, or the lever is named unreachable **with the trace that proves it**.
>
> Following-Agents-Protocol: true
>

## Resolution

GitHub records closing pull request #2341 (https://github.com/mudler/vllm.cpp/pull/2341) merged on 2026-08-30 as commit `7d53ae3b405761202b13065ee32ea1d0dd13cd29`. GitHub closed issue #2282 on 2026-08-30.
