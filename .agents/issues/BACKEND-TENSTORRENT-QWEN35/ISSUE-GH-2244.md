ID: ISSUE-GH-2244
Title: Stage TT uploads through a per-slot persistent device buffer written via the mesh command queue
Row: BACKEND-TENSTORRENT-QWEN35
State: CLOSED
Kind: feature
GitHub: 2244
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-29
Updated: 2026-08-29
Closed: 2026-08-29

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> W4 of the TT Qwen3.5 row (#2107) landed levers 1+2 of the staging breakdown — bulk bf16 staging and single-slot resolution — and moved the P150 decode leg from 0.104 to 0.177 tok/s (+70%), with the staging `Numel()` share down from 27.09% to 1.76%. Lever 3 (batch per-layer staging) was not taken: the re-attribution put the residual wall inside tt-metal itself.
>
> **The attribution.** Of the remaining staging chain, ~23% is tt-metal per-upload internal work and ~19.2% is CPU threadpool spin. Every staging upload goes through `UploadRowsBf16` (`src/vt/tenstorrent/tenstorrent_ops.cpp:469`), which builds a fresh `ttnn::Tensor` via `from_span` — so every step pays tt-metal's full creation path: a fresh `MeshBuffer` allocation, cluster/chip discovery, and CQ completion handling, even though the geometry is identical step over step.
>
> **The lever.** Make the upload allocation-free: allocate the device buffer once per staging slot, and on each step write the host bytes into the persistent buffer through the mesh command queue (`MeshCommandQueue::enqueue_write` / `enqueue_write_shard`) instead of creating a new tensor per upload. The W4 record names exactly this as the next hypothesis: a per-slot persistent device buffer via mesh CQ.
>
> **The tt-metal-internal half.** The claim "per-upload work drops" must be proven against the pinned tt-metal's `MeshCommandQueue` write path (the `enqueue_write_shard` → per-device buffer write + completion chain), not assumed. Read the source, trace the executed path, and dump it before declaring any part of the lever unreachable.
>
> **Scope.**
> - Persistent per-slot device buffers for the staging slots; lifecycle tied to the existing slot structures (and the #1486 never-destroy rule for static caches).
> - Bulk path (`UploadRowsBf16`) routed through the mesh CQ write; geometry pure as today.
> - The f32-conversion arms keep their declared dtypes; the capture-unsafe host-write guards (the `[TT-UP] ... during capture` refusals) keep their semantics.
> - `StagingStats` gains route counters for the new path so the before/after profile attributes the shift.
>
> **Invariant.** Staging stays bit-identical: the sacred golden pair stays 16/16 byte-identical and the full TT suite stays green (44/44 · 4340 at filing). This wave changes speed, never tokens.
>
> **Evidence owed on landing.** Same-method before/after profile on the P150 (identical leg, lock discipline) plus a fresh benchmark-record entry. A wall that does not move is a reported result, not a failure — the attribution either shifts or the lever is named unreachable with the trace that proves it.
>
> Owned by `BACKEND-TENSTORRENT-QWEN35`; this is the row's next wave after W3 (#2201, landed via #2217).
>
> Following-Agents-Protocol: true
> AI-Assisted: true
> Assisted-by: AGENT:zai-glm-5.3-flash [maki]
>

## Resolution

GitHub records closing pull request #2258 (https://github.com/mudler/vllm.cpp/pull/2258) merged on 2026-08-29 as commit `017c3277f527a65ece29e87259206a7f3f739127`. GitHub closed issue #2244 on 2026-08-29.
