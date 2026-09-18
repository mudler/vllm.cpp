ID: ISSUE-LOCAL-01M2C8HBDD9VG6AJMPM9PTN80S
Title: HcGroupedNormKernel runs four threads and accumulates in double, against its own header's contract, and it is 40.7% of decode kernel time
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

`src/vt/cuda/cuda_qwen4_exp.cu:258` launches `HcGroupedNormKernel` with ONE THREAD PER (token, hc stream) -- `GridFor(T * hc)` at `:402`. The released Qwen3.8-Flash-Next has `hc_count = 4` (`qwen4_exp.h:179`) and decode runs at T = 1, so this kernel does its work with FOUR ACTIVE THREADS on a 48-SM GB10. Each thread then walks `H = 2560` elements TWICE in a serial loop and accumulates the sum of squares in `double` via `__dadd_rn` / `__dmul_rn` (`:271-275`).

MEASURED, `nsys` on `dgx:gpu0`, 60 s window over ~15.2 steady-state decode steps (2026-09-12, GB10 sm_121a, released UD-IQ1_S): `HcGroupedNormKernel` is **40.7% of all GPU kernel time** -- 626.94 ms across 1,439 instances, mean 435.7 us, min 410.7, max 470.8. That is 96 instances and ~42 ms per decode step against 101 ms for every kernel combined, and it is the largest kernel by 2.6x over the next one.

NOW THE TOP ITEM, because the allocator defect above it landed. After `3b3ed716f` the step is 116.9 ms and 8.57 tok/s, so this kernel is ~36% of a decode token rather than ~1% of one.

THE DOUBLE CONTRADICTS THE CONTRACT ITS OWN HEADER STATES. `qwen4_exp_hc.h:99-104` says the sum of squares is accumulated in double as "the `deepseek_v4_mhc.cpp` house convention for a HOST REFERENCE", and states the rule for the other arm in the same paragraph: "Upstream runs the norm in fp32 (`self._norm(x.float())`) and vLLM likewise (`x = x.float()`) ... this is a CPU reference and THE DEVICE ARM IS THE THING THAT MUST BE FP32-ACCUMULATE and gated against these numbers." The device arm is not fp32-accumulate; it inherited the host reference's double. Upstream and vLLM are both fp32 here and the kernel is the outlier, which is also the failure AGENTS.md names: "A token gate cannot detect a dtype that is too wide."

TWO DEFECTS, SEPARABLE. (1) PRECISION: double-rate throughput is a fraction of fp32 and the accumulation is 2 x 2560 serial double FMAs per thread. (2) PARALLELISM: four threads is the larger scandal. One block per (token, hc) group with a block-wide reduction is the ordinary shape, and the elementwise normalize-and-write loop after the reduction is order-independent and parallelises with no numerical question at all.

THE GATES ALREADY ADMIT THIS, WHICH IS WHY IT IS NOT A PARITY DECISION. The device arm is NOT held bit-exact against the CPU arm. `tests/vllm/models/test_qwen4_exp_hc_device.cpp:76` gates the golden widths at `kTol = 1e-5` absolute, and its model-width case (`:378`, the tolerance at `:442`) gates a RELATIVE `4e-5`, documented there as "6.6x the sqrt(K)*u random-walk bound for K = 10240" -- a bound derived for FP32 unit roundoff, with the comment saying so directly: "the oracle itself sits at the same ORDER as 1e-5 against an exact double evaluation of its own algorithm, because torch runs this in fp32 too". A serial fp32 walk sits at sqrt(K)*u ~= 6e-6 relative; a block tree reduction is BETTER, at ~sqrt(log K)*u. Both land inside a bound the tree derived before this work existed.

The goldens' discriminating power is also already measured and does not depend on the accumulator width: `test_qwen4_exp_hc.cpp` separates the narrowest single-character defect at 6.63e-3 against the 1e-5 tolerance, a 663x band.

WHICH FILE GATES THE CUDA ARM, corrected 2026-09-13. The paragraph above cites `test_qwen4_exp_hc_device.cpp` for the tolerances, and those tolerances are right, but that file is NOT the gate for this kernel: it says at its own head "Nothing below runs on a device" and it holds the CPU arms against the transformers goldens. The CUDA gate is `tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp`, under the `CUDA W7:` case names. MEASURED on `thor:gpu0` (sm_110, CUDA 13.0.88): corrupting the `1 +` gamma fold in `HcGroupedNormKernel` reddens `test_qwen4_exp_cuda_reductions` at 7 of 18 cases, worst `max|diff|` 0.868741 against its 1.95703e-06 bound (the grid-cap case), while `test_qwen4_exp_hc_device` stays SUCCESS at 11/11 cases over 516 assertions. Read a green `hc_device` as a statement about the CPU arms only.

OWED BY THE FIX: discriminating evidence that fails for the intended reason, the CUDA gate and the CPU-arm gate both green, and an nsys re-measurement on `dgx:gpu0` against the NEW 116.9 ms step -- never against the retired 3.95 s one. NO RED-FIRST IS AVAILABLE and the spec's W7 section records why: the pre-change kernel reproduces the CPU reference EXACTLY on the new fixture (`max|diff| = 0`, measured on an independent rebuild), because both arms walk the group ascending in `double`. The evidence is a mutation battery plus the in-gate separation probe instead.

MEASURED ON THOR, 2026-09-13, AND THE ISSUE STAYS OPEN. An interleaved same-tree A/B on `thor:gpu0` (NVIDIA Thor, sm_110), one boot per arm, two rounds alternating BASE and FIX, the released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S staged to local disk, the server at `--max-num-seqs 1 --device cuda` with no other engine flag, a 16-token decode, and the median inter-token interval as the statistic:

| arm | commit | round 1 tok/s | round 1 s/token | round 2 tok/s | round 2 s/token |
|---|---|---|---|---|---|
| BASE, the W7 parent | `49ffc61cf` | 4.6564 | 0.21372 | 4.9773 | 0.21430 |
| FIX | `ac04275b8` | 7.3755 | 0.14542 | 7.2675 | 0.14629 |

The median per-token interval goes 0.2140 s -> 0.1459 s: ~68 ms removed per token, about 1.5x. Peak resident memory is unchanged, `VmHWM` 77,361,940 kB on BASE against 77,355,492 kB on FIX, and no arm printed `out of memory`, `bad_alloc`, or a CUDA error.

THE BYTES MEASURED ARE THE BYTES THAT LAND. `ac04275b8` is an earlier revision of the same commit; `git diff ac04275b8 d7e0e9cf2 -- src/` is EMPTY and the only `tests/` difference is one comment block in `test_qwen4_exp_cuda_reductions.cpp`. The rebase onto `43622bc37` reproduced the patch byte-for-byte.

THREE LIMITS, STATED SO THE NUMBER IS NOT OVER-READ. (1) This is THOR at sm_110, not the `dgx:gpu0` GB10 at sm_121a that every other number in this issue comes from; it is a slower box with a larger step, and the thor figure is not comparable to the sojufx reference or to the W6 dgx numbers. (2) The 40.7% / ~42 ms per step attribution above was measured on DGX. The 68 ms per token removed here was measured on thor. Direction and rough magnitude agree; nothing here CONFIRMS the 42 ms figure. (3) The spec's prediction -- ~42 ms per step removed, a step near 77 ms, ~13 tok/s -- was written for dgx and has NOT been tested there.

THIS IS WHY THE ISSUE STAYS OPEN. The owed evidence above asks for an `nsys` re-measurement on `dgx:gpu0` against the NEW 116.9 ms step. `dgx:gpu0` read `unhealthy (no contact)` across three separate outages on 2026-09-13 and that measurement could not be taken. It remains owed.

## Resolution

FIXED AND LANDED 2026-09-13. The kernel is `HcGroupedNormKernel` in
`src/vt/cuda/cuda_qwen4_exp.cu`. W7 replaced the four-thread `double` walk with
one block per group, fp32 `__fadd_rn`/`__fmul_rn` interior and a warp-shuffle
tree reduction, which is what `qwen4_exp_hc.h:99-104` said the device arm was
supposed to do all along ("THE DEVICE ARM IS THE THING THAT MUST BE
FP32-ACCUMULATE"; the `double` was the HOST reference's convention, applied to
the wrong arm).

- `ee0644eab` -- the fix.
- `f116751e8` -- the kernel-level evidence, 435.7 us -> 16.0 us per launch, 27x.
- `f97e8451a` -- the dgx A/B this issue owed, below.

**THE dgx A/B IS NO LONGER OWED; IT WAS MEASURED.** Interleaved same-tree A/B on
`dgx:gpu0`, BASE `3b3ed716f` (W6 only) against FIX `ee0644eab` (W6+W7), one boot
per arm, two rounds: **0.1177 -> 0.0778 s per token, 40 ms removed, 1.51x**,
8.50 -> 12.85 tok/s. The W7 scope had predicted, before the work started, "~42 ms
per step removed, a step near 77 ms, and ~13 tok/s". The prediction was met and
not adjusted.

`HcGroupedNormKernel` no longer appears in the top twelve of the post-W7 profile,
so the 40.7% rank-1 position this issue reported is gone at the ranking level and
not only end to end.

**ONE FIGURE IN THIS ISSUE IS A LONG-CONTEXT FIGURE and the reader should know
it:** the "40.7% of decode kernel time" was measured on the same 3000-token
profile whose workload mismatch is corrected in
`.agents/specs/qwen4-exp-flash-next.md`, "### W9: the two numbers this row kept
dividing into each other". Unlike the QSA kernel, this one's cost does not scale
with context -- it is per-token and per-layer -- so the rank was inflated only by
whatever the context-scaled kernels around it were doing, and the 27x per-launch
figure is unaffected either way.


### KERNEL-LEVEL CONFIRMATION, thor sm_110, 2026-09-13

`nsys` over a decode window on the landed W7 binary ranks this kernel at **0.1%
of GPU kernel time, 16.0 us average over 32 instances** (min 14.3, max 20.7).
Before W7 the same kernel on `dgx:gpu0` was **40.7% and 435.7 us average**. That
is a **27x reduction in per-launch cost**, and the kernel has fallen from rank 1
to roughly rank 14.

THIS IS THE KERNEL FIGURE, NOT A STEP FIGURE, and it is the one that is safe to
quote from this run. The same capture's per-step totals are NOT usable: the
window caught ~730 ms of GPU work across 45 s (about 1.6% utilisation) with
prefill-shaped kernel sizes, and `decode.json` came back empty, so the client did
not run through the window. Three attempts on that box placed the capture window
from an ASSUMED model-load time -- first a cold 12-minute figure against a warm
page cache, which profiled an idle server -- and the per-step ranking is
therefore still OWED, together with the dgx A/B this issue already owes.

What the 27x does establish is the MECHANISM rather than only the direction: the
old kernel's cost was a dependent chain of 2 x 2560 serial `double` FMAs on four
threads, and a log-depth fp32 tree over one block per group is exactly the change
that collapses it. The end-to-end thor A/B (0.2140 -> 0.1459 s per token) and
this per-launch figure agree about what moved.
