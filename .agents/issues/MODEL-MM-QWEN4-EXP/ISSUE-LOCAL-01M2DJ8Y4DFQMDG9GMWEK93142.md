ID: ISSUE-LOCAL-01M2DJ8Y4DFQMDG9GMWEK93142
Title: QsaGatherAttentionKernel is 34.5% of decode GPU time at 2.545 ms a launch, and at decode it runs 24 blocks on 48 SMs
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

MEASURED on `dgx:gpu0` (GB10 sm_121a) 2026-09-13, `nsys` over a 60 s window inside a 3000-token decode at `ee0644eab` (W6+W7 landed, step 77.8 ms, 12.85 tok/s): `QsaGatherAttentionKernel` is **34.5% of all GPU kernel time** -- 16.079 s across 6,319 instances, mean **2.545 ms**, median 2.555, min 1.268, max 4.987. That is ~8.2 instances and **~20.4 ms per decode step**, and it is the largest kernel by a factor of **ten** over the next one's mean (cuBLAS `gemvx` at 242 us).

THIS IS A NEW #1 AND IT WAS INVISIBLE BEFORE. In the pre-W6 trace the same kernel was 2.7% at 231 us, buried under 2.87 s/step of allocator churn. W6 removed the allocator and W7 removed `HcGroupedNormKernel` (which was 40.7%), and this is what the profile shows underneath. The next row was going to be the dense GEMV path on the strength of the OLD ranking; scoping from a stale profile would have aimed the work at the wrong kernel. Re-rank before scoping, and re-rank again after this lands.

THE STRUCTURAL OBSERVATION, which is the same shape W7 had. `cuda_qwen4_exp_qsa.cu:609-612` launches `grid = pairs < 4096 ? pairs : 4096` with `width = BlockWidthFor(DH)`. At decode `T = 1`, `pairs = T * HQ`, and the released Qwen3.8-Flash-Next has `num_attention_heads = 24` (`qwen4_exp.h:216`) with `head_dim = 128` (`:45`). So the decode launch is **24 blocks of 128 threads on a 48-SM GB10** -- half the SMs idle before the kernel does any work. W7's predecessor was four threads on 48 SMs; this is 24 blocks, less extreme but the same family, and the W7 result (435.7 us -> 16.0 us per launch once the work was spread) is the reason to look here first.

WHAT IS NOT YET KNOWN, and must not be assumed. Whether the 2.545 ms is dominated by (a) the grid being narrower than the machine, (b) the per-block walk over selected KV blocks, (c) the gather's memory pattern, or (d) something in the paged path that the fixture never exercises. The instances-per-step figure (~8.2) also implies only about 8 of the 48 layers are `kQwenSparseAttention` on this checkpoint; confirm that against `layer_types` before sizing any fix, because a per-layer cost is a different target from a per-head one.

OWED BEFORE A FIX IS SCOPED: an `nsys`/Nsight-Compute attribution of where the 2.545 ms goes inside the kernel, on `dgx:gpu0`, against the CURRENT 77.8 ms step. Do not scope from this issue's paragraph alone -- that is the mistake this row already made once, when `~1,400 kernel launches` was carried as a cause for months without anyone counting them.

## Resolution

**STAYS OPEN, and this is what for.** W9 discharged the ATTRIBUTION this issue
owed and removed the term it found, and the W9 repair gates the fold order that
change rests on. Two things this issue asked for are NOT done, and they are the
only reasons it is still open:

1. **The RE-RANK this issue's own problem statement demands.** "Re-rank before
   scoping, and re-rank again after this lands." The 34.5% figure is a
   `dgx:gpu0` profile taken BEFORE W9; after a 6.52x on the kernel it cannot
   still be the ranking, and the next row must not be scoped from it. Nothing
   has re-profiled since `7d0d74c2c`.
2. **The step-level number, which this issue is careful NOT to derive.** The
   6.52x is a kernel result at `|sel| = 2048`. What QSA costs per decode step
   depends on the context length, and the 16-token A/B harness runs the kernel
   at `|sel|` of about 36, so no end-to-end figure follows. A long-context
   end-to-end measurement is owed before any step-budget claim.

Also unresolved and recorded above rather than fixed: `ncu` returns
`ERR_NVGPUCTRPERM` on this worker, so the arithmetic-latency and load-latency
halves of the remaining cost are unseparated, and the 86-89% fraction is an
sm_110 reading that GB10 has never reproduced.

### ATTRIBUTED 2026-09-13 on `thor:gpu0` (sm_110): 86-89% is a dot product on ONE THREAD

Measured at `ce51eeea6`, CUDA 13.0.88, through the production op
`vt::Qwen4ExpQsaGatherAttention` linked against the real `libvllm.a`, 15 timed
reps after 3 warm-ups, median. Released decode shape: `T=1, HQ=24, HKV=2,
DH=256, CR=4, block_topk=512`, so `|sel| = 2048`.

**The dominant term is the pass-2 dot at `cuda_qwen4_exp_qsa.cu:517-521`, which
`if (threadIdx.x == 0)` runs on a SINGLE LANE** -- `|sel| x head_dim = 524,288`
dependent `__fadd_rn`/`__fmul_rn` pairs, each fed by its own scalar bf16 global
load, while the block's other 255 threads wait at `__syncthreads()`.

The identification is FORCED rather than argued. `blockDim = BlockWidthFor(DH) =
ceil32(DH)` scales with `DH`, so every other term's per-thread serial length is
independent of `head_dim` and only this one is proportional to it. A head_dim
sweep at fixed `|sel|` therefore isolates it: slope **0.04767 ms per unit
head_dim**, R2 = 0.99998 and 0.99970 across two independent runs, giving a
DH-proportional share of **86.4%** and **88.8%** at the released `DH=256`. That
the term is per selected row and not a fixed per-call cost is confirmed
independently: `slope(|sel|=2048) / slope(|sel|=512) = 4.03` against a selection
ratio of 4.00.

**CANDIDATE (a), THE NARROW GRID, IS REFUTED AS THE COST.** Holding per-block
work fixed and varying the grid: 2 blocks 13.481 ms, 24 blocks 14.156, 48 blocks
14.670, 96 blocks 30.322. **Two blocks take the same wall time as twenty-four.**
The machine saturates between 48 and 96 blocks, so at the released `grid=24`
there is no parallelism to recover across (token, head) pairs. The idle SMs are a
symptom. The same defect family is one level down: for ~86% of the wall time,
255 of 256 threads are idle INSIDE the block.

Candidate (c), the gather's address pattern, is **~5%**: spreading the selected
blocks over a 16x larger cache moves 13.784 -> 14.442 ms, and `keys_visited`
reads 98,304 in every case, so the gather is honest. Candidate (d), the paged
path, is **+9.4%** (14.127 -> 15.455 ms). Candidate (b) is the whole cost and is
localised by the above.

### WHAT NONE OF THE FOUR CANDIDATES NAMED: every selected key is dotted TWICE

`keys_visited = 98,304 = 2048 rows x 24 heads x 2 passes`. Pass 1 computes each
selected key's dot SPREAD ACROSS 256 THREADS to find the softmax max; pass 2
recomputes the identical value on thread 0. They are bit-identical by
construction -- same `s_q`, same ascending order over `d`, same `__fmul_rn` /
`__fadd_rn` -- and pass 1's copy costs about 1/256th of pass 2's, which is the
intercept in the fit. The expensive half is the RECOMPUTATION.

This matters for what a fix may do. The kernel's own header names a tree
reduction over `d` as the lever and declines it because it would break the
CPU-vs-CUDA bit relation. **Reusing pass 1's already-computed dot would not
require that reassociation**, so it is a different lever than the one the header
rejected. Stated as a measured property of the code; NOTHING WAS IMPLEMENTED and
no speedup is claimed.

### TWO CORRECTIONS TO THIS ISSUE'S OWN TEXT

1. The decode launch is **24 blocks x 256 threads**, not 128. `DH = query.shape[2]`
   is the MODEL's `head_dim` (`qwen4_exp.h:216-218`, and `qsa_block.cpp:797` says
   so in words); the 128 at `qwen4_exp.h:45` is the INDEXER's head_dim. The
   original text read the wrong field.
2. "~8 of the 48 layers are QSA" is wrong. `qwen4_exp.h:29-33` has the
   `__post_init__` rewrite covering **12 of 48**. At 12 the per-step QSA total is
   12 x 2.545 = **30.5 ms of the 77.8 ms step**, not the 20.4 ms this issue
   derived.

### AND THAT FALSIFIES A STEP COUNT THIS ROW PUBLISHED

The re-rank record (`f97e8451a`) derived "~771 steps" in the nsys window from
60 s / 77.8 ms. The QSA instance count implies **6,319 / 12 = 527**. The two
disagree by **1.46x**, and every per-step figure derived from 771 inherits that
uncertainty -- notably `cudaFree`, quoted there as ~68 calls and ~43 ms per step,
which at 527 steps would be ~99 calls and ~63 ms. NEITHER derivation is retracted
here, because neither has been checked against a step counter; what is retracted
is the confidence. Anyone scoping the `cudaFree` question must resolve the step
count first, by instrumenting it rather than dividing.

### RESOLVED 2026-09-13, BY ARITHMETIC: 527, AND 771 IS IMPOSSIBLE

The paragraph above asked for a step counter. It is not needed, and the reason the
two derivations disagreed by 1.46x is that they describe two different CONTEXT
LENGTHS rather than one being miscounted.

QSA is 34.5% of GPU kernel time and 16.079 s of the 60 s window, so total GPU
kernel time in the window is `16.079 / 0.345 = 46.6 s`. At 527 steps that is
**88.5 ms of GPU kernel per step**, inside a 113.9 ms step: 77.7% busy, coherent.
The 771 figure requires the window's step to have BEEN 77.8 ms, and **88.5 ms of
measured kernel cannot fit inside 77.8 ms of wall.** So 527 is right and the
window's step was 113.9 ms.

What made 77.8 ms look like the window's step is that it was measured on a
DIFFERENT WORKLOAD: `bench_decode.py` generates **16 tokens** off a ~20-token
prompt, so `kv_len` is about 36, whereas `bench_nsys3000.py` generates 3,000 and
captures `[150 s, 210 s]`, i.e. roughly 1,300-1,900 tokens of context. This
kernel's cost is proportional to `|sel| = min(kv_len, block_topk * CR)`, so it
does ~44x less work in the A/B than in the profile.

**THE CONSEQUENCE FOR THIS ISSUE'S OWN HEADLINE.** "34.5% of decode GPU time at
2.545 ms a launch" is a ~1600-token-context statement. Per decode step this kernel
is 30.5 ms at `|sel|=1600`, **4.2 ms** at the 400-token reference workload, and
**0.69 ms** at the 16-token A/B. It is the top of the budget at long context and a
few percent of it at the workload the sojufx gap is quoted on. The fix (W9) is
worth doing either way -- 86-89% of the kernel is 86-89% of it at any context --
but this issue must not be cited as "the top of the decode budget" without the
context length attached.

The `cudaFree` population inherits the correction in the direction that makes it
LARGER per step: ~99 calls and ~63 ms per step rather than ~68 and ~43 ms, and
those are host costs that do not scale with context. That is where the next
profile should be aimed. It is still not a claim that it IS the cost -- host API
time can overlap device work and the per-step attribution is still owed.

### WHAT THIS DOES NOT ESTABLISH

Not measured on GB10: `dgx:gpu0` was unhealthy throughout, and sm_110 runs this
shape at 13.78 ms against GB10's 2.545 ms, a 5.4x difference. The IDENTIFICATION
of the dominant term is architecture-independent (it rests on which code path
scales with `head_dim`); the 86-89% FRACTION is an sm_110 number.

`ncu` produced nothing: every capture died with `ERR_NVGPUCTRPERM`, the driver
refusing performance counters to the container. There is no occupancy, stall or
memory-throughput reading here, and the empty capture is recorded as a FAILURE
and not as a result. So within the 86% the arithmetic-latency and load-latency
halves are not separated -- 6.914 us per selected row over 256 elements is 27 ns
per element, far above a bare dependent FADD, so a load component is certainly
present but is unsized.

Synthetic call, not the engine: random bf16 K/V, one query token, contiguous
ascending block ids, no ragged tail. Nothing here speaks to per-step launch
counts or the step budget, and no correctness comparison against goldens was run.

Artifacts: `/workspace/qsa-attrib/20260913T145638Z/`, `...T145839Z/`,
`...T150040Z/` on the shared NAS; `rc` jobs `3ea9b1c5`, `3ef2fe85`, `130642f7`.

### W9 LANDED 2026-09-13 on `thor:gpu0` (sm_110): 17.69 ms -> 2.71 ms, **6.52x**

The pass-2 dot now runs one whole dot per thread. Measured through the
production op `vt::Qwen4ExpQsaGatherAttention` against the real `libvllm.a`,
CUDA 13.0.88, sm_110, at the SAME released shape the attribution used --
`T=1, HQ=24, HKV=2, DH=256, CR=4, block_topk=512`, `|sel| = 2048` -- 15 timed
reps after 3 warm-ups, median, both arms in ONE job off ONE base tree:

| arm | median | min | max |
|---|---|---|---|
| base `6a47d4370` | **17.6921 ms** | 17.6123 | 17.8523 |
| fix | **2.7118 ms** | 2.5574 | 4.9118 |

**6.52x against the spec's `~6.4x` prediction, which is NOT adjusted.** An
earlier, aborted run of the same job measured the base arm at 17.8571 ms, 0.9%
from the number above, so the denominator is reproduced rather than taken once.

**THE STEP-LEVEL NUMBER IS NOT DERIVED HERE, and the reason is the operator's
correction to the spec.** The 2.545 ms per launch and the 77.8 ms step come from
two different context lengths, so they must not be divided into each other: QSA
cost tracks `|sel| = min(kv_len, ~2048)`, which is ~1600 in the profile window
and ~36 in the 16-token A/B harness. The 6.52x above is a KERNEL result at
`|sel| = 2048`. What it is worth per step depends on the context the step runs
at, and no end-to-end number is claimed from a harness whose `|sel|` is 36.

**Bit identity is DEMONSTRATED, not argued.** The suite's arm-vs-arm gate holds
the gather to a DERIVED BOUND, not to equality
(`test_qwen4_exp_cuda_reductions.cpp:37-46`), so a green suite is not evidence of
bit identity. The gather's raw output bytes were therefore captured on the device
at base and at fix over 11 fixtures -- the two transformers goldens, seven
synthetics spanning `DH` 1 to 256 with multi-tile and ragged-tail selections, the
released decode shape, and a malformed selection whose row is poisoned -- 11,836
floats, 47,344 bytes. Both streams are
`b5bdad52c1a692626fdfc06cf6c87796d462cedc78f3437054aaadeaf3960f61` and `cmp`
exits 0. The paged arm rides on this: the suite gates it BITWISE against the
contiguous arm, which the dump covers.

**`racecheck` is clean WITH a positive control.** `compute-sanitizer --tool
racecheck` over the two multi-tile QSA cases reports `0 hazards displayed
(0 errors, 0 warnings)`. Deleting the `__syncthreads()` between the tile write
and the denominator fold, rebuilt (md5 `7a31c372e515` vs the fix's
`5e4ab5a98147`), reports hazards in `QsaGatherAttentionKernel` in the tens of
thousands (19,200 at one site). The clean run therefore measures the barrier
rather than the tool being silent.

**Four mutations, each rebuilt and each DETECTED** (per-arm binary md5 beside the
fix's `5e4ab5a98147`, EIGHT gather cases selected and counted in every arm):
bounding the tile map by `head_dim` instead of `blockDim.x` (`bac53105fc8a`,
1 case red), dropping the last entry of every tile (`a63dcb30cb32`, abort),
deleting the denominator fold (`d51ffb73ee3e`, 7 cases red), and not counting the
pass-2 read (`dcbc383fd18d`, 2 cases red). Restoring the source returns 8/8 green
at source sha256 `55d34c66a72a495db5db79045f72b616e2171f723c1557e3ddbbdb48fd657507`.

**The count in the two lines above read `nine` and `9/9` and it is WRONG. The
file carries EIGHT** `TEST_CASE("vt::Qwen4ExpQsaGatherAttention...` at
`7d0d74c2c`, counted in the tree. The corrected figure changes no verdict: every
arm was red where it is recorded red and green where it is recorded green. It is
corrected because a denominator nobody counted is how a selector that matches
nothing reads as a pass. The W9 REPAIR adds a ninth, the softmax-fold gate below,
so `nine` becomes true after the repair lands and was not true when it was
written.

Artifacts: `/workspace/qsa-w9/20260913T160154Z/` on the shared NAS; `rc` jobs
`5a24eeef` (base+fix+racecheck+mutations) and `c3666e8f` (an earlier run of the
same job, aborted by a defect in the job script's own binary comparison, whose
base timing is quoted above).
