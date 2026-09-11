ID: ISSUE-GH-623
Title: perf(cpu): add packed Cortex-A76 Q8 multi-output kernel
Row: KERNEL-CPU-A76-Q8-DOT
State: OPEN
Kind: UNKNOWN
GitHub: 623
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-13
Updated: 2026-08-13
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Track KERNEL-CPU-A76-Q8-DOT follow-up work to close the measured Raspberry Pi 5 gap against llama.cpp by increasing data reuse rather than rescheduling the existing nrc=1 leaf.
>
> Scope:
> - profile and speed-limit model the current A76 Q8_0 x Q8_0 path;
> - add one explicit, non-default packed 1x4 or 4x4 candidate through the shared MatmulBTQuant dispatch;
> - preserve strict block-order floating-point accumulation and exact output/token gates;
> - prove selector, packing/layout, tails, alignment, ABI and reporting with RED-first tests and mutations;
> - require pinned GCC13 AArch64 disassembly plus QEMU correctness before any Pi measurement;
> - compare same-binary current A76 assembly vs candidate on M=1 and reached prefill shapes with wall time, cycles, instructions, frontend/backend stalls and cache traffic;
> - do not change the automatic default unless physical Pi and recursive model gates pass every throughput/latency/memory axis.
>
> The current one-output assembly measures about 14.94 cycles per two Q8 blocks versus an optimistic ~10-cycle SIMD-port bound, while the full prefill profile attributes ~30% of cycles to Q8 and shows backend/cache rather than frontend pressure. The candidate hypothesis is activation reuse across four output rows, following the project-pinned llama.cpp 1x4/4x4 structure but adapting it to this project strict reduction-order contract.

## Resolution

-
