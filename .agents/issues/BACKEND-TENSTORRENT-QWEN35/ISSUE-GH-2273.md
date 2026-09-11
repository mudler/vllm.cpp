ID: ISSUE-GH-2273
Title: W6: batch per-layer staging — one mesh-CQ write per step, so the measured per-op CQ tax divides by fan-in
Row: BACKEND-TENSTORRENT-QWEN35
State: CLOSED
Kind: perf
GitHub: 2273
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-29
Updated: 2026-08-29
Closed: 2026-08-29

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> **Row**: `BACKEND-TENSTORRENT-QWEN35` (spec `.agents/specs/tenstorrent-qwen35.md`, W4 lever 3 — deferred there as optional, re-derived as owed by the W5 trace).
>
> ## The measured premise
>
> W5 (#2244) removed the per-upload device allocation and the wall did **not** move: same-method interleaved A/B on one lock hold, 19.154 s vs 19.181 s mean (−0.14%, noise) on the 3-token greedy leg. The `perf` traces split W4's hypothesis: `allocate_mesh_tensor_on_device_with_topology` is **0.02%** of the AFTER profile (the route is active — staging really is allocation-free now), and BEFORE shows no allocation stack either, so allocation was never the wall. What both arms share is the per-CQ-operation tt-metal stack, charged **once per staging write**:
>
> - `Threadpool::PollForWork` 14.29%
> - `MetalContext::instance` 11.14%
> - `memcpy` 6.23%
> - `Cluster::get_chip` 5.90%
> - `read_cq_host_ptr` 5.27% (+ sub-slices 4.08 / 2.33 / 2.24 / 1.35)
>
> A decode step issues one mesh-CQ write per staged tensor (the layer fan-in), so this fixed cost multiplies by the number of writes, not by the bytes.
>
> ## The lever
>
> Batch per-layer staging: pack a step's staged host rows into one contiguous host block and issue **one** mesh-CQ write per step (or per layer group), instead of one per tensor. The per-op tax then divides by the fan-in. Full log: `docs/bench-evidence/tt-qwen35-eager-profile-w5-20260829.log`.
>
> ## Invariants and constraints
>
> - **Bit-identical staging** — the sacred golden pair stays 16/16 STRICT token-exact and the full TT suite stays green; this wave changes SPEED, never tokens.
> - `StagingStats` gains route counters for the new path (red-first: the counters land with a test that fails on zero).
> - The capture-unsafe host-write refusals keep their semantics; the f32-conversion arms keep their declared dtypes.
> - **Aliasing awareness from the W5 review**: staged device tensors already alias the persistent buffer under same-geometry restage (safe today because each stage fully overwrites before return and weights stage once). A batched/arena layout must state its restage semantics explicitly, not inherit W4's fresh-snapshot reasoning.
> - Reachability: the new route must be reachable from the production decode path (`ModelRegistry::Forward` → staging), not only from a hand-constructed test.
>
> ## Evidence owed on landing
>
> Same-method before/after profile on the P150 (identical leg, JIT-discard per arm, one lock hold, `perf record -F 199 -g` per measured leg), plus a fresh benchmark-record entry. A wall that does not move is a reported result, not a failure — the attribution either shifts, or the lever is named unreachable **with the trace that proves it**. The tt-metal-side residual (cached context handles, amortized CQ polling) stays recorded as the upstream-shaped alternative.
>
> Following-Agents-Protocol: true
>

## Resolution

GitHub records closing pull request #2280 (https://github.com/mudler/vllm.cpp/pull/2280) merged on 2026-08-29 as commit `785d4304f52881622f46edbab6a4c9da616bddfa`. GitHub closed issue #2273 on 2026-08-29.
