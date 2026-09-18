ID: ISSUE-LOCAL-01M2TZ9AE416FW4GH23V09783Q
Title: QuantDotGemmGrouped32Kernel runs one warp per output element with a 2.5-iteration inner loop, and it is the slowest row in the decode bandwidth table at 15.2% of peak
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

DERIVED from the committed bandwidth attribution plus the kernel source. No GPU
and no lease: the shapes and the launch geometry are in the tree, and the
per-kernel rates are already measured in
`ISSUE-LOCAL-01M2EK69SESGH6ST1ESMFZC808`.

## The measurement this starts from

That attribution ranked the decode kernels by achieved bandwidth ascending. **The
slowest row is the IQ1_S expert gate/up grouped kernel at 41.6 GB/s, 15.2% of
GB10's ~273 GB/s**, against cuBLAS `gemvx` at 162.3 GB/s (59.5%) on the same box
in the same capture. Our own `QuantDotGemm*` family averages 93.4 GB/s (34.2%)
and is 34.1% of decode kernel time -- 26.22 ms per step.

## The structural cause, read off the source

`src/vt/cuda/cuda_quant_dot.cu:1986` `QuantDotGemmGrouped32Kernel` assigns **ONE
WARP PER OUTPUT ELEMENT** `(p, j)`, and `LaunchGrouped32` (`:2092-2098`) uses
`kWarpsPerBlock = 4`, `dim3 block(32, 4)`.

At the released shape -- `hidden_size` 2560, so `nb = 80` blocks of 32 -- the
inner loop is:

```
for (int64_t b = lane; b < nb; b += 32)     // nb = 80, 32 lanes
```

**2.50 iterations per lane.** Each thread issues two or three loads, then the
warp pays a five-step `__shfl_down_sync` tree, then 31 of its 32 lanes go idle
while lane 0 writes a single float. Five reduction steps to amortise two and a
half loads.

That shape is latency-bound by construction: there is almost no memory-level
parallelism WITHIN a thread, so the kernel can only hide latency through warp
occupancy. 15.2% of peak is what that looks like.

## The fix has a precedent 120 lines below it in the same file

`QuantDotGemmGroupedFusedSwiGLU32Kernel` (`:2019`) already does the amortisation
at width 2: one warp computes BOTH the gate and up outputs for `(p, j)` against
the same broadcast Q8_0 activation, doubling the work per warp and paying the
reduction once for two results.

**Generalise that to N output COLUMNS per warp.** A warp loads the activation
blocks it needs once and dots them against N consecutive weight rows, keeping N
accumulators, then reduces N of them. The loop becomes `2.5 x N` iterations of
useful work per reduction, and the activation -- which is broadcast, `bcast=true`
-- is read once per warp instead of once per output element.

## What this is worth, stated as a bound and not a promise

If the family reached cuBLAS's measured 162.3 GB/s it would be **1.74x** on
26.22 ms/step, i.e. ~11.1 ms off a 76.9 ms kernel step. Reaching the full 273
would be 2.92x, ~17.2 ms. **NEITHER IS PREDICTED HERE.** The kernel may be bound
by something this reading does not see -- register pressure at higher N, the
Q8_0 activation's own traffic, or an occupancy cliff -- and 15.2% of peak has
exactly one measurement behind it.

## Owed BEFORE any width is chosen

1. **An N sweep**, 1/2/4/8, on the released shape, measuring achieved bandwidth
   per N rather than assuming monotonicity. Register pressure turns this over at
   some N and the turning point is the answer.
2. **The same sweep for the non-grouped `QuantDotGemm32Kernel`** and the Q8_K
   family, which share the one-warp-per-output shape and were not separately
   measured.
3. Whether `kWarpsPerBlock = 4` (128 threads) is itself the occupancy limit at
   the released grid of 1,600 blocks.

## The bar

**Bit identity.** This is a re-association of nothing -- each output element's
dot product keeps its own accumulator and its own ascending block order; only the
assignment of work to warps changes. If any float moves, the change is not this
change. The existing `test_quant_dot*` suites are the red-first surface.

This row has twice written an unachievable bit-identity bar
(`ISSUE-LOCAL-01M2ENTH6YA5FWEDY6CFHF4NAM` section 2, and the W8 spec before it).
This one is achievable precisely because the accumulator order per output is
untouched -- state that in the scope so the distinction is not lost again.


## Resolution

-
