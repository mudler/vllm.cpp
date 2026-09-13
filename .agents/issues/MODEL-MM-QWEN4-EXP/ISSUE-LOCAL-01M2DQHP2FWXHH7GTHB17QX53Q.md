ID: ISSUE-LOCAL-01M2DQHP2FWXHH7GTHB17QX53Q
Title: the decode step makes ~99 cudaFree calls at 634 us each, and DevicePool means it should be making almost none
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

MEASURED (host API side) on `dgx:gpu0` 2026-09-13, `nsys` over a 60 s window
inside a 3000-token decode at `ee0644eab`: `cudaFree` is **63.0% of all CUDA API
time** -- 52,130 calls at **634 us average**. Over the window's **527** steps
(see the step-count resolution in `ISSUE-LOCAL-01M2DJ8Y4DFQMDG9GMWEK93142`) that
is **~99 calls and ~63 ms per step**. A 634 us free is itself anomalous and is
the reason to look: `cudaFree` synchronises the device, so each one drains the
stream before the next launch can be enqueued.

**THE STATIC FINDING, and it is the opposite of what the first revision of this
issue said.** The `vt::Free` seam itself does have no pooling: `vt::Free`
(`src/vt/backend.cpp:132-144`) forwards to the CUDA backend's `FreeOnDevice`
(`src/vt/cuda/cuda_dropin.cu:209-212`), a bare `Check(cudaFree(pointer), ...)`.
**But almost nothing in a decode step is supposed to reach it**, because
`DevicePool` (`include/vllm/model_executor/models/device_pool.h`) sits above that
seam and exists precisely to stop it. Its own `Get` says so
(`device_pool.h:111-125`): blocks "are never returned to the driver", and the
`VT_POOL_BYPASS=1` lane that turns every `Get`/`Put` back into a raw driver
`Alloc`/`Free` "reinstates the per-op `cudaMalloc`/`cudaFree` sync storm this pool
exists to remove, so it is never a timing configuration". A block only goes back
to the driver through `DevicePool::Drain` (`device_pool.h:322-338`), "one
`cudaFree` per retained block, once".

`qwen4_exp` routes its temporaries through that pool -- `qwen4_exp_forward.h:140`,
`qwen4_exp_qsa_block.h:200` and `qwen4_exp_ple_block.h:171` each describe "the
shared_ptr that returns its pool block to the `DevicePool` when the last reference
drops".

**SO THE MEASUREMENT IS MORE ANOMALOUS, NOT LESS.** A steady-state decode step
whose temporaries are pooled should free almost nothing, and a pool MISS costs an
`Alloc`, never a `Free`. ~99 `cudaFree` per step at 634 us is therefore not "the
allocator this tree uses"; it is ~99 frees that the pool's design says should not
be happening at all. **THE FIRST REVISION OF THIS ISSUE GOT THIS BACKWARDS** and
proposed `WorkspacePool` and the FA2 scratch as precedents for a caching allocator
to add. The cache already exists. Do not scope one.

**ONE CANDIDATE IS ALREADY REFUTED, STATICALLY.** A per-step `Drain` would
explain the count almost exactly -- "one `cudaFree` per retained block" against
~99 frees a step. It is not the cause: `Drain` has exactly two call sites in the
tree, `minimax_h3_pipeline.cpp:559` and `ltx2_video.cpp:5800`, both at a
diffusion/video phase boundary, and neither is reachable from a `qwen4_exp` text
decode. Whoever takes this should not spend a lease re-testing it.

### THE ESCAPE ROUTE IS NAMED, AND THE COUNT MATCHES TO 1.9% (static, 2026-09-13)

`vt::Qwen4ExpGatedResidual`'s CUDA arm takes a **raw `cudaMalloc` per call** for
its four intermediates and frees it before return -- `cuda_qwen4_exp.cu:496` for
the alloc, the `FreeGuard` destructor at `:504-506` for the free. Neither goes
through `DevicePool`. The kernel's own header says so at `:283-286`: "The scratch
is ONE `cudaMalloc` sliced four ways, freed before return. A per-call allocation
in a decode loop is a cost, and it is recorded in the spec's `## Owed` as a SPEED
item rather than papered over with a static cache that would not be re-entrant."

**It is called twice per layer, unconditionally, plus once after the loop.**
`qwen4_exp_forward.cpp` has three call sites: `:565` and `:638` both sit inside
the layer loop opened at `:491`, each in a bare scope block and NEITHER behind a
conditional (verified by brace depth, not by reading), and `:747` runs once after
it. At `L = 48`:

```
2 * 48 + 1  = 97 cudaMalloc + 97 cudaFree per decode step
measured      98.92 per step (52,130 calls / 527 steps)
unexplained    1.92 per step, 1.9%
```

At the measured 634 us per free, 97 frees is **61.5 ms of the ~63 ms per step**
this issue opened on. The remaining ~2 per step are unattributed; candidates are
cuBLAS/cuBLASLt internals and any path building a device `Tensor` without a
`DBuf`. (The `keys_visited` counter in `cuda_qwen4_exp_qsa.cu:600-628` is NOT one
of them: it is guarded by `args.keys_visited != nullptr`, which the header names
as the instrument-only path that production does not take.)

### THE SAME TRACE CORROBORATES IT FROM THE ALLOC SIDE, AND NAMES A CHEAPER FIX

The API table this issue quoted only its `cudaFree` row from carries the paired
row too:

| API | % | calls | avg |
|---|---|---|---|
| `cudaFree` | 63.0 | **52,130** | **633.69 us** |
| `cudaMalloc` | 0.4 | **52,143** | **4.04 us** |
| `cudaMallocAsync` | 0.2 | 50,553 | 2.26 us |
| `cudaFreeAsync` | 0.1 | 50,553 | **1.30 us** |

**The synchronous pair differs by 13 calls across the whole 60 s window** --
52,143 allocs against 52,130 frees, 98.94 and 98.92 per step. That is a strictly
paired population, which is what a per-call scratch allocated and freed inside one
op looks like, and it is independent corroboration of the call-site count above.

**AND THE COST IS ENTIRELY ON THE FREE SIDE: 633.69 us against 4.04 us, 157x.**
The allocation is nearly free; `cudaFree` is expensive because it synchronises.

**THAT POINTS AT A MUCH SMALLER FIX THAN THE ONE `## Owed` DESCRIBES.** The same
trace shows `cudaFreeAsync` at **1.30 us**, 487x cheaper than `cudaFree`, over a
population of 50,553 -- so the stream-ordered allocator is already carrying a
comparable number of allocations in this very run, at negligible cost. Moving the
mixer's scratch from `cudaMalloc`/`cudaFree` to
`cudaMallocAsync`/`cudaFreeAsync` on the op's own stream would need no signature
change, no caller-supplied workspace and no re-entrancy argument, and the tree
already uses that pair in `cuda_matmul_nvfp4_cutlass.cu`, `cuda_ops.cu`,
`cuda_marlin_repack.cu` and `cuda_mamba2_ssd.cuh`. **IT IS NOT FREE OF
CONDITIONS**: `cuda_mla_attn.cu:454` records that under graph capture a
`cudaMallocAsync` becomes a graph-owned memory node, which is a live concern for
this row precisely because W8 exists to make capture possible. Whoever scopes
this owes that interaction an answer; it is named here so the scope starts from
it rather than discovering it.

### THE RECOVERABLE TIME IS BOUNDED BY IDLE GPU, NOT BY THE 61.5 ms

**DO NOT READ "61.5 ms of a ~77.8 ms step" AS A 4x SPEEDUP WAITING TO BE
COLLECTED.** `cudaFree` is expensive here *because it synchronises*, and what it
synchronises on is the work already queued on the stream. Most of those 633.69 us
are therefore spent WAITING FOR KERNELS THAT HAD TO RUN ANYWAY, not performing
work that disappears when the call does. Removing the synchronisation removes the
SERIALISATION -- it lets the host run ahead and keep the device fed -- so the
recoverable time is bounded above by how long the GPU is currently IDLE, and not
by the host-side total.

The one measurement of that ratio this row has is long-context: 46.6 s of GPU
kernel time in a 60 s window, **77.7% busy**, leaving ~22% idle. If the short-
context step has a similar shape, the ceiling on this fix is roughly the idle
fraction of the step, not 61.5 ms of it. **THAT RATIO HAS NOT BEEN MEASURED AT
THE REFERENCE WORKLOAD** and it is the number that sizes this work, so measure it
in the same run that instruments the call sites: GPU busy percentage at 400
tokens, with and without the change.

This is a bound on the CLAIM, not a ceiling on the work -- an apparent limit is an
unresolved implementation difference until it is traced, and a host that runs
ahead can also expose the next bottleneck rather than merely hiding behind this
one.

**THIS IS AN ARITHMETIC MATCH, NOT A MEASUREMENT, and the difference matters.** A
count that lands within 2% of the observed population is a strong hypothesis and
nothing more. It does not establish that removing these frees removes 61.5 ms of
STEP time, because `cudaFree` is host API time that can overlap device work --
the second owed item below is still owed, unchanged. What the match does do is
make the per-call-site attribution cheap: instrument these three sites and the
question is settled in one run.

**THE FIX IS ALREADY SCOPED IN PRINCIPLE and is not a new allocator.** The spec's
`## Owed` (the second of "TWO SPEED ITEMS THIS WAVE DECLINED") states it: "A
static cache would not be re-entrant; the fix is a caller-supplied workspace,
which changes the op signature and owes its own spec." That is the same shape as
the FA2 decode scratch, and it keeps the op re-entrant where a file-static would
not.

Candidates that remain for the residual ~2: frees inside cuBLAS or cuBLASLt that
the profile attributes to our process, and any path that constructs a device
`Tensor` without going through a `DBuf` at all.

**WHAT IS NOT ESTABLISHED, AND NO ROW MAY SCOPE A FIX UNTIL IT IS.** The spec's
`## Owed` already says this and it still holds:

1. **Which allocations, BY CALL SITE.** Largely answered above by arithmetic --
   97 of ~99 are `Qwen4ExpGatedResidual`'s per-call scratch -- but answered
   STATICALLY. Confirm it by instrumenting the three call sites and counting, in
   one run. W6 already showed that guessing the population wrong sends a row at
   the wrong object, and a 1.9% residual is exactly the size of a second, smaller
   population hiding behind a good-looking match.
2. **How much of the 63 ms is recoverable WALL time.** `cudaFree` is host API
   time and it can overlap device work. It also synchronises, so it can serialise
   work that would otherwise overlap -- which is the opposite sign. Only a
   measurement separates "63 ms of the step" from "63 ms of host time hidden
   behind device work".
3. **The workload.** These counts come from the same 3000-token profile whose
   mismatch is corrected in the spec's "### W9: the two numbers this row kept
   dividing into each other". Unlike the QSA kernel, `cudaFree` count per step
   should NOT scale with context -- which, if confirmed, is exactly what makes it
   the leading candidate at the 400-token reference workload while QSA is not.
   **CONFIRM THAT RATHER THAN ASSUMING IT**, by counting per step at two context
   lengths in the same run.

**Owed before a scope:** a per-step attribution of the `cudaFree` population by
call site, at the reference workload (400 tokens), with the per-step count taken
at two context lengths. `dgx:gpu0` has been unhealthy repeatedly; `thor:gpu0`
can carry the attribution, and the fleet-comparable numbers are owed on dgx.


## Resolution

**MEASURED AND CLOSED 2026-09-13.** `dgx:gpu0`, `rc` job `4e36bbae`, post-W9
`main` (`3eabd4dbd`), 600-token decode at the reference workload, `nsys` window
opened and closed by the client and SELF-VALIDATED at 1,594,621
`cudaLaunchKernel` calls.

**The derivation was right.** 97 per step was read off three call sites in
`Qwen4ExpGatedResidual`; the device says **99.0** (59,400 frees / 600 steps),
paired with 99.0 `cudaMalloc` to within 28 calls across the whole window.
`cudaFree` is **80.0% of CUDA API time** at **652.5 us**, i.e. **64.6 ms of an
87.6 ms step**.

**AND THE LEVER IS NOT WHAT THE COUNT SUGGESTS, which is why this issue insisted
on the second owed item.** The GPU is **88% BUSY** -- 76.9 ms of kernel time in
that 87.6 ms step -- leaving **10.6 ms idle**. The 64.6 ms of host time overlaps
GPU work almost entirely, so removing it recovers at most the idle:
**87.6 -> 76.9 ms, 11.4 -> 13.0 tok/s, about +14%.** Not the 4x that "74% of the
step" invites.

The issue's own words were "the recoverable time is bounded above by how long the
GPU is currently IDLE, and not by the host-side total... that ratio has not been
measured at the reference workload and it is the number that sizes this work."
It is measured now, and it sizes the work DOWN.

**What to do with it.** Still worth doing and still cheap -- `cudaFreeAsync` is in
the same trace at **1,340 ns** against 652,550 ns -- but it is a 14% item, and the
same trace shows `QsaGatherAttentionKernel` at 18.16 ms/step and cuBLAS `gemvx` at
17.53 ms/step. **This row is GPU-bound at 88%**; the next work is less GPU work.
Anyone scoping the async move must still answer the graph-capture interaction
(`cuda_mla_attn.cu:454`) and the retention finding
(`ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ`), both unchanged.

Closed because both owed items are answered: the population is attributed by call
site to 2.1%, and the recoverable fraction is bounded by measurement. The full
record, including the kernel table and a new observation about QSA's
context-independent per-launch cost, is in
`.agents/specs/qwen4-exp-flash-next.md`, "### THE ALLOCATOR, MEASURED AT THE
REFERENCE WORKLOAD".

**THE OTHER SIDE OF THIS POOL IS NOW MEASURED, AND IT CONSTRAINS THE FIX.**
`ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ` (row ENG-POOL-BEST-FIT, landed
`abaa79c43`) measured `DevicePool`'s UNCAPPED RETENTION on `dgx:gpu0`: a c=32
EXL3 DFlash2 leg with CUDA graphs on drove `MemAvailable` from 58.2 to 13.8 GiB
in two minutes before a watchdog stopped the server, while the same leg with the
pool bypassed and graphs off held at 34.3 GiB for 27 minutes.

That is the same allocator from the opposite direction -- retention cost there,
escape rate here -- and the two findings pull in OPPOSITE directions, which is
the thing to notice before scoping either. This issue says ~97 allocations a step
should be IN the pool and are not. That one says the pool already retains more
than a GB10 can afford. **A fix that simply routes the mixer's scratch into a
retaining allocator therefore adds to a population that is already the subject of
a memory-exhaustion issue**, and that includes the `cudaMallocAsync` route
proposed above: the driver's stream-ordered pool has a release threshold of its
own and retains by default. Whoever scopes this reads both issues first and says
which allocator the scratch lands in and under what cap, rather than treating
"put it in a pool" as the answer.

### CORRECTION 2026-09-13

Filed earlier the same day claiming "there is no pooling at that seam" and
proposing a caching allocator. `DevicePool` is that allocator and it has been in
the tree throughout, including for every measurement quoted here. The seam-level
reading was right and the inference drawn from it was wrong: reading one seam and
generalising to the path above it is the failure this record now carries.
