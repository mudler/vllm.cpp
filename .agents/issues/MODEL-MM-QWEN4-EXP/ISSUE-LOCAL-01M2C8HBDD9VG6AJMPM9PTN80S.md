ID: ISSUE-LOCAL-01M2C8HBDD9VG6AJMPM9PTN80S
Title: HcGroupedNormKernel runs four threads and accumulates in double, against its own header's contract, and it is 40.7% of decode kernel time
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

`src/vt/cuda/cuda_qwen4_exp.cu:258` launches `HcGroupedNormKernel` with ONE THREAD PER (token, hc stream) -- `GridFor(T * hc)` at `:402`. The released Qwen3.8-Flash-Next has `hc_count = 4` (`qwen4_exp.h:179`) and decode runs at T = 1, so this kernel does its work with FOUR ACTIVE THREADS on a 48-SM GB10. Each thread then walks `H = 2560` elements TWICE in a serial loop and accumulates the sum of squares in `double` via `__dadd_rn` / `__dmul_rn` (`:271-275`).

MEASURED, `nsys` on `dgx:gpu0`, 60 s window over ~15.2 steady-state decode steps (2026-09-12, GB10 sm_121a, released UD-IQ1_S): `HcGroupedNormKernel` is **40.7% of all GPU kernel time** -- 626.94 ms across 1,439 instances, mean 435.7 us, min 410.7, max 470.8. That is 96 instances and ~42 ms per decode step against 101 ms for every kernel combined, and it is the largest kernel by 2.6x over the next one.

NOW THE TOP ITEM, because the allocator defect above it landed. After `3b3ed716f` the step is 116.9 ms and 8.57 tok/s, so this kernel is ~36% of a decode token rather than ~1% of one.

THE DOUBLE CONTRADICTS THE CONTRACT ITS OWN HEADER STATES. `qwen4_exp_hc.h:99-104` says the sum of squares is accumulated in double as "the `deepseek_v4_mhc.cpp` house convention for a HOST REFERENCE", and states the rule for the other arm in the same paragraph: "Upstream runs the norm in fp32 (`self._norm(x.float())`) and vLLM likewise (`x = x.float()`) ... this is a CPU reference and THE DEVICE ARM IS THE THING THAT MUST BE FP32-ACCUMULATE and gated against these numbers." The device arm is not fp32-accumulate; it inherited the host reference's double. Upstream and vLLM are both fp32 here and the kernel is the outlier, which is also the failure AGENTS.md names: "A token gate cannot detect a dtype that is too wide."

TWO DEFECTS, SEPARABLE. (1) PRECISION: double-rate throughput is a fraction of fp32 and the accumulation is 2 x 2560 serial double FMAs per thread. (2) PARALLELISM: four threads is the larger scandal. One block per (token, hc) group with a block-wide reduction is the ordinary shape, and the elementwise normalize-and-write loop after the reduction is order-independent and parallelises with no numerical question at all.

THE GATES ALREADY ADMIT THIS, WHICH IS WHY IT IS NOT A PARITY DECISION. The device arm is NOT held bit-exact against the CPU arm. `tests/vllm/models/test_qwen4_exp_hc_device.cpp:76` gates the golden widths at `kTol = 1e-5` absolute, and its model-width case (`:379`) gates a RELATIVE `4e-5`, documented there as "6.6x the sqrt(K)*u random-walk bound for K = 10240" -- a bound derived for FP32 unit roundoff, with the comment saying so directly: "the oracle itself sits at the same ORDER as 1e-5 against an exact double evaluation of its own algorithm, because torch runs this in fp32 too". A serial fp32 walk sits at sqrt(K)*u ~= 6e-6 relative; a block tree reduction is BETTER, at ~sqrt(log K)*u. Both land inside a bound the tree derived before this work existed.

The goldens' discriminating power is also already measured and does not depend on the accumulator width: `test_qwen4_exp_hc.cpp` separates the narrowest single-character defect at 6.63e-3 against the 1e-5 tolerance, a 663x band.

OWED BY THE FIX: a red-first case that fails for the intended reason, the two existing gates green, and an nsys re-measurement on `dgx:gpu0` against the NEW 116.9 ms step -- never against the retired 3.95 s one.

## Resolution

-
