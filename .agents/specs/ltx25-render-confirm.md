# SPEC — `LTX25-RENDER-CONFIRM`: turn the connector repair's projected render into a wall clock, by advancing the pin rather than removing it

Issue: [#2457](https://github.com/mudler/vllm.cpp/issues/2457).
Owner row: `LTX25-RENDER-CONFIRM`.
Predecessors: [#2296](https://github.com/mudler/vllm.cpp/issues/2296)
(`.agents/specs/ltx25-render-speed-parity.md`, the 516.751 s baseline and its
method), [#2434](https://github.com/mudler/vllm.cpp/issues/2434)
(`.agents/specs/ltx25-connector-repair.md`, the component reading and the
projection), [#1854](https://github.com/mudler/vllm.cpp/issues/1854)
(`.agents/specs/ltx25-oracle-absolute.md`, the correctness gate this row
re-takes), and [#1864](https://github.com/mudler/vllm.cpp/issues/1864) (the
reference render the verdict is taken against).

## The gap

`LTX25-CONNECTOR-REPAIR` measured **one `Ltx2ConnectorForward` call** on GB10 at
**128.808 s -> 50.34 s**, a **2.56x**, with `vt::AttentionCross` **85.704 ->
8.47 s** (10.12x) and the GEMM unmoved at **31.906 -> 32.863 s**. From that it
projected the render: connector compute **224.882 -> ~87.9 s**, the render
**516.751 -> ~380 s**, the oracle gap **5.51x -> ~4.05x**.

**Its own `## Owed` says no render was run**, and the reason is a guard rather
than an oversight. `scripts/ltx25-render-speed-repeat.sh` asserts the binary's
`sha256` against `rc` job `4b0666ee`'s -- the run that returned `VERDICT PASS`
against #1864's reference -- and exits **51** on any other binary. That guard is
what stops a speed number being quoted for a binary whose correctness nobody
established. **A component speedup is not a system speedup**, and the only thing
that makes the distinction checkable is that the wall is timed on a binary with
a verdict behind it.

So the gap is not "run the harness". It is: **produce a binary at the current
head that has itself taken #1864's verdict, and time THAT.**

## Scope

IN scope:

- **W1** — this spec.
- **W2** — one committed harness, `scripts/ltx25-render-confirm.sh`, that in ONE
  lease and on ONE binary: builds the head, runs the CUDA unit gate, renders,
  takes #1864's blockiness verdict against both reference forms, and only on a
  PASS goes on to time `N >= 3` renders and fold their phase tables.
- **W3** — the measurement, the phase table, the verdict, and the pin in
  `scripts/ltx25-render-speed-repeat.sh` advanced to the binary that earned it.

OUT of scope, declared rather than approximated:

- **Any repair.** This row measures. If the wall does not reach ~380 s, the
  result is the number and the next traceable hypothesis, not a fix.
- **Weakening the pin by any route other than re-taking the verdict.** The three
  `WANT_*_SHA` values in `ltx25-render-speed-repeat.sh` move only after a PASS
  on the binary they will name, and they move to that binary's digests. A
  `WANT_*_SHA` made overridable-by-default, a comparison skipped, or a guard
  deleted is `NEEDS_DECISION`, not a workaround.
- **A published benchmark ID.** `docs/BENCHMARKS.md` gains nothing: this is one
  request at one geometry, and `ltx25-render-speed-parity.md` already owns the
  public statement of the gap.
- **The oracle side.** The denominator is the pinned oracle's own recorded
  `render_seconds = 93.8` from
  `tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json`, `n = 1`, and its
  missing spread is stated as the limit it is rather than filled in.

## Why one harness and not two

`ltx25-oracle-absolute-render.sh` builds and judges the PICTURE, renders once,
and exits on the comparison's verdict. `ltx25-render-speed-repeat.sh` times the
WALL `N` times and refuses to build. Running them back to back in two leases
would time a binary built twice, and this repository has the standing note that
an A/B reusing one build directory measures one binary twice -- the mirror of
the same defect.

**The two obligations have to close over the same bytes**, so this row's harness
takes the artefacts once, hashes them once, and both the verdict and every
timing quote that one digest. That is the whole reason the file exists, and it
is why it is committed rather than typed into a lease: `oracle-ltx-2-pin.md`
records four earlier attempts whose manifest described a script that no longer
existed.

**It reuses the neighbours' hard-won parts verbatim** rather than re-deriving
them: the `soname_ok` CUDA-toolkit check that #2220 paid 21 minutes for, the
`MemAvailable` start floor #1709 paid four leases for, the `exec` in the clock
sampler subshell that #2305 paid three orphaned windows for, the all-or-nothing
build cache, the `.part`-rename staging, the `steps_observed` denominator read
back from the render's own lines, and the `< 1%` phase-coverage refusal.

## The order is correctness, then timing, and the harness enforces it

1. CUDA unit gate (`test_ltx2_device`) before any render.
2. Render 1.
3. #1864's comparison, against the 25 PPM frames upstream's own decode wrote AND
   against the committed `upstream-render.mp4`. **A non-zero verdict exits the
   job**, before renders 2..N exist. The blockiness gate failing is the headline
   and the speed number is void; a harness that collected the wall anyway would
   invite quoting it.
4. Renders 2..N, then the fold.

Render 1 is timed like every other render and is not discarded for having been
the one compared. The comparison runs after its wall is recorded.

## What the gate cannot see, and what is therefore printed beside it

`ltx25-oracle-absolute.md` records that our render is already **less blocky,
less sharp and less clipped** than upstream's, and that `blockiness_grid8` /
`blockiness_grid32` are **one-sided ceilings**: they are blind to further
smoothing. `sharpness_mean`, `clipped_fraction` and the audio RMS stay
`reported_statistics` in the tool's own JSON for exactly that reason.

**So this row prints the reported-only panel beside the verdict**, both arms'
values, and reports the direction each moved. A PASS on a one-sided ceiling
plus an unremarked collapse in sharpness would be a gate reporting green on a
render that got worse.

## Reachability

Nothing new lands in `src/` or `include/`. The subject is
`ModelRegistry::Forward`'s own LTX-2.5 path exercised through the shipped
`ltx2-gen` binary on its default configuration -- the production entry point --
at the manifest's request. `scripts/ltx25-render-confirm.sh` is a measurement
harness and is reached by being run in the lease this spec records.

## Tests to port

There is no upstream test. Upstream emits no phase table and has no blockiness
gate; #1864's reference render IS the ported artefact, and
`scripts/ltx25-render-compare.py` and its committed suite already hold the
criterion. This row adds no assertion to that tool and changes none of its
thresholds.

## Gates

| ID | Gate | Refusal |
|---|---|---|
| G1 | all four checkpoint `sha256` equal the manifest's, recomputed inside the lease | 23 |
| G2 | a CUDA toolkit whose `libcudart` / `libcublasLt` SONAME links resolve | 38 |
| G3 | `test_ltx2_device` green, case and assertion counts recorded | 44 / 45 |
| G4 | every render writes exactly 25 frames, a non-empty wav and a `phase-log.json` | 48 |
| G5 | every render observes `steps_observed={8}` | 53 |
| G6 | every phase table covers `>= 99%` of its own wall | 52 |
| G7 | **#1864's comparison against the PPM frames exits 0**, on the binary that produced the timing | the comparison's own status |
| G8 | the same comparison against the committed mp4, cross-checked, disagreement reported | reported |
| G9 | `N >= 3` renders completed | 49 |
| G10 | `scripts/check-tree-compiles.py` and `scripts/agent-preflight.sh` | — |

## Risks/decisions

- **A component ratio does not transfer to a system.** The projection is
  explicitly a ratio applied to a different quantity: the probe's connector
  total uses synthetic weights, a different valid-token count, and times two
  streams in one process. This row's deliverable is the wall, and the phase
  table is what says whether the wall moved for the predicted reason.
- **THE WALL CAN MOVE FOR THE WRONG REASON.** `load.dit` had a **49.70%**
  spread in the baseline because its cost is page cache. A render that comes in
  at 380 s because the DiT was warm is not this repair. The claim is checked at
  the LEAF: `conditioning.connector` (0.44% spread) and the connector half of
  `generate.guiders` (1.12%) must fall, while `denoise`, `decode.video` and
  `decode.audio` must not. If the wall moves and those leaves do not, the
  projection was right by accident and this spec says so.
- **`n = 1` is an anecdote.** `N >= 3` and the spread is quoted per phase.
- **Contention reweights rather than adding noise.** The render is 87-88%
  GPU-idle and host-bound, so `loadavg` and `MemAvailable` are the axes that
  transfer. Both are sampled before and after every render, and the SM clock is
  folded through `tools.bench.gpu_clock_state` anyway because a number is
  quotable only with the clock it was taken at.
- **The oracle denominator is `n = 1`.** 93.8 s has no spread of its own. Every
  ratio here inherits that limit and says so.
- **ENOSPC.** A full CMake build writes 9.4 GiB and named targets are used for
  that reason; the checkpoints are 70 GB and are staged to `/root`, which is
  local, with the free-space check before the copy.

## Stop conditions

Stop and report, do not work around:

- **the blockiness gate failing** -- the speed number is not reportable and the
  failure is the finding;
- a checkpoint digest that is not the manifest's;
- an unreachable or unhealthy fleet device, or one held by somebody else;
- a wall that does not reach ~380 s -- that is a RESULT, reported with the phase
  table and the next traceable hypothesis, never a ceiling;
- any pressure to weaken the sha256 pin other than by re-taking the verdict.

## Work breakdown

- **W1** — this spec.
- **W2** — `scripts/ltx25-render-confirm.sh`.
- **W3** — the lease, the measurement, `## Outcome`, and the advanced pin.

## Now

`DONE`. W1, W2 and W3 are in this change. The wall is measured, the verdict is
re-taken on the binary that produced it, and the pin is advanced to that binary.

## Outcome

### The headline

**302.954 s, n = 3, spread 8.03%, against the 518.398 s baseline — 1.711x — and
the oracle gap is 3.230x rather than 5.53x.** The projection said ~380 s and
~4.05x. **The measured render BEATS it by 20%,** and the reason is not that the
connector did better than projected. It is that a second landed change the
projection never carried moved a leaf of its own.

| | seconds | ours/oracle |
|---|---:|---:|
| ours, `phase-log.json` `wall_seconds`, n = 3 | 289.002 / 313.335 / 306.525, mean **302.954** | **3.230x** |
| ours, whole `ltx2-gen` subprocess, n = 3 | 294 / 318 / 312, mean **308.0** | 3.284x |
| #2296 baseline, same harness form, n = 3 | 518.398 | 5.527x |
| `LTX25-CONNECTOR-REPAIR`'s projection | ~380 | ~4.05x |
| oracle `render_seconds`, `ltx2_oracle_manifest.json`, **n = 1** | **93.8** | 1.000 |

**The oracle denominator is still n = 1 and still has no spread of its own.**
Our 8.03% is measured; theirs is unknown. Every ratio above inherits that.

### The projection, checked leaf by leaf, which is the only way to check it

The prediction was specific: `conditioning.connector` and
`guiders.connector.compute` collapse while `load`, `denoise` and `decode.video`
do not. **Three of those four hold, and the fourth is where the extra 77 s came
from.**

| leaf | #2296 baseline | this run | change | predicted |
|---|---:|---:|---:|---|
| `conditioning.connector` | 122.388 | **49.960** | **2.45x** | collapse ✓ |
| `generate.guiders` | 190.016 | **95.506** | **1.99x** | collapse ✓ |
| `denoise` | 15.129 | **15.122** | **1.000x** | unmoved ✓ |
| `load` (span) | 94.483 | **94.550** | **0.999x** | unmoved ✓ |
| `load.dit` | 35.043 | 36.110 | 0.97x | unmoved ✓ |
| `load.text_encoder` | 56.156 | 55.160 | 1.02x | unmoved ✓ |
| `conditioning.tower` | 28.343 | 27.014 | 1.05x | unmoved ✓ |
| **`decode.audio`** | 50.745 | **8.773** | **5.78x** | **NOT predicted** |
| ~ `decode.audio.mel` | 47.175 | **4.926** | **9.58x** | **NOT predicted** |
| **`decode.video`** | 15.970 | **10.927** | **1.46x** | predicted UNMOVED ✗ |

**`denoise` is unchanged to five significant figures — 15.129 s against
15.122 s.** The 21B transformer did not get faster and nothing here claims it
did. That is the control that makes the rest of the table readable.

**THE CONNECTOR'S OWN PROJECTION WAS RIGHT, and it is now directly measured
rather than inferred.** The two `.compute` sub-phases the projection is about
are leaves in this binary:

| | #2354 | projected | **measured** |
|---|---:|---:|---:|
| `conditioning.connector.compute` + `guiders.connector.compute` | **224.882** | ~87.9 | **81.338** (41.234 + 40.104) |
| the ratio | — | 2.56x | **2.765x** |

81.338 s against a projected 87.9 s is **7.5% better than projected**, on the
leaf the projection actually named. A prediction recorded before the render
existed and then met is worth more than either number alone.

**THE EXTRA 77 s IS `decode.audio.mel`, AND IT IS ATTRIBUTED RATHER THAN
GUESSED.** `4fef1f413 perf(LTX25-AUDIO-DECODE-COST): the audio VAE had 20 cores
and used one, for 47 s of a 518 s render` landed between the verdict binary's
tree and this head. Its own commit subject names the 47 s; the leaf reads
47.175 -> **4.926 s**, a **9.58x** and **-42.2 s**. The connector projection
never carried it, which is why the render beat a projection whose own leaf it
met.

**The residue closes, and the itemisation is over THIS TABLE'S OWN LEAVES.**
The wall fell **215.444 s**. Summed over every leaf both tables share:

| leaf | delta |
|---|---:|
| `generate.guiders` | **-94.510** |
| `conditioning.connector` | **-72.428** |
| `decode.audio` | -41.972 |
| `decode.video` | -5.043 |
| `conditioning.tower` | -1.329 |
| `load.text_encoder` | -0.996 |
| `artifacts.frames` | -0.100 |
| `load.video_vae` | +0.068 |
| `load.dit` | **+1.067** |
| everything else | < 0.05 each |
| **sum** | **-215.285** |

against the wall's **-215.444**: a residual of **0.159 s**, 0.05% of the fall.

**THE 143.5 s OF CONNECTOR COMPUTE IS NOT A TERM IN THAT SUM, and an earlier
draft of this section wrongly made it one.** 224.882 -> 81.338 is measured
against #2354's connector figure, which was taken in a DIFFERENT run and is not
a leaf of the #2296 baseline table. The baseline binary predates the `.compute`
split, so its connector cost is spread across `generate.guiders` and
`conditioning.connector` together with the two towers, and those two leaves are
what the residue sum may use. Adding the cross-run figure to the same total
double-counts it and does not even add up: it gives 191.7 s against a 215.4 s
fall. The two comparisons are both valid and they are kept apart.

**`decode.video` MOVED AND THE PROJECTION SAID IT WOULD NOT.** 15.970 ->
10.927 s, 1.46x, and it is outside its own 1.71% spread, so it is a real move
and not noise. **No commit in the range names it**, so this row does not
attribute it. The leading hypothesis is a shared-seam side effect — the video
VAE's GEMMs run through the same `vt::MatmulBT` that `LTX25-CONNECTOR-REPAIR`
re-blocked and `VT-CPU-ELEM-DISPATCH` de-dispatched — and it is recorded under
`## Owed` as unexplained rather than assigned.

### The correctness verdict, on the binary that produced the timing

**`VERDICT PASS`, taken inside this same job on render 1, before renders 2 and 3
existed.** Both reference forms agree: the 25 PPM frames upstream's own decode
wrote (`exit 0`) and the committed `upstream-render.mp4` (`exit 0`).

| gated statistic | ours | reference mean | reference per-frame max | margin |
|---|---:|---:|---:|---:|
| `blockiness_grid8` | 1.030110 | 1.042812 | 1.143393 | **+0.113283** |
| `blockiness_grid32` | 1.024809 | 1.037230 | 1.148672 | **+0.123862** |

`blockiness_grid8_defined` and `blockiness_grid32_defined` both pass at **0 of
1600 collapsed bands**, so the ceiling was not cleared by a flat block grid
reading as the smallest possible value.

**THE REPORTED-ONLY PANEL, BECAUSE THE GATE IS A ONE-SIDED CEILING.**
`ltx25-oracle-absolute.md` records that our render is already less blocky, less
sharp and less clipped than upstream's, and a ceiling is blind to further
smoothing. So the panel is printed beside the verdict, and it says the render
moved **toward** the reference on every one of them:

| statistic | verdict binary `0002ddfba` | **this binary** | upstream |
|---|---:|---:|---:|
| `blockiness_grid8` | 1.0221 (-1.98%) | **1.0301 (-1.22%)** | 1.042812 |
| `blockiness_grid32` | 1.0254 (-1.14%) | **1.0248 (-1.20%)** | 1.037230 |
| `sharpness_mean` | 10.5176 (-6.71%) | **10.6374 (-5.65%)** | 11.274039 |
| `clipped_fraction` | 0.000758 (-54.06%) | **0.001076 (-34.79%)** | 0.001650 |

Three of the four moved toward upstream and the fourth moved 0.06% away. **There
is no smoothing regression hiding under the passing ceiling**, and that sentence
is a measurement here rather than an assumption.

`test_ltx2_device` ran before any render at **23 cases / 806 assertions / 0
failed** — the same counts `fa9903b86` and the #2296 baseline recorded.

### The pin, earned rather than asserted

`scripts/ltx25-render-speed-repeat.sh` now names this binary, and it names it
because this binary took the verdict above in the same job that produced the
timing:

```text
WANT_BIN_SHA=600cf798c48ebabebc1fa25fb4891fe0b550f31f995501105aea856cced4c54d
WANT_LIB_SHA=c4692db9025899463122b64b26cc40486dd445d37f9b7835e06cbf26daf83492
WANT_SRC_SHA=790c582bbba45ab0f7b74aafee361e4557a84bf2
```

The guard is unchanged in force: the harness still refuses any other binary with
exit 51, and it still never builds. What moved is which verified binary it
points at, and `W`/`CACHE` move with it so the digests and the artefacts they
name stay in the same place.

### The lease, the box, and the load

`rc` job **`93a60151-7d4d-4718-842c-ef724208be0e`** on **`dgx:gpu0`**, the GB10
the oracle and the 516.751 s baseline were both measured on. Queued at position
2, ran 2026-09-01T07:58:37Z to 08:39:30Z, **40m53s**. Evidence at
`/mnt/nas_share/rc/ltx25-render-confirm/run/20260901T075837Z`, which a leased
worker sees as `/workspace/ltx25-render-confirm/run/20260901T075837Z`.

aarch64, 20 cores, 119 GiB total. **`MemAvailable` 113.7 GiB at the start and
114.3-115.0 GiB across every render.** `loadavg` **1.97** before render 1, then
10.10 and 12.51 before renders 2 and 3 — that rise is the previous render's own
decay, not another tenant: `nvidia-smi --query-compute-apps` listed **no other
GPU process for the whole job**, and the #2296 baseline has the identical shape
(1.44, then 7.6 to 8.1). Render 1 is the fastest leg in both runs.

**The clock, because a number is quotable only with it.** GB10, persistence
`Enabled`, `clocks.max.sm` 3003 MHz, applications graphics 2418 MHz. Busy-sample
SM medians **2434 / 2411 / 2411 MHz** at spreads **3.74% / 4.56% / 4.31%**, all
inside `.agents/benchmarking.md`'s 5% ceiling and within 1.0% of each other. The
baseline read 2418 / 2411 / 2437 at 4.30 / 4.02 / 3.98%. **The clock is the
same on both sides, so the 1.711x is not a clock artefact.**

**The independent instrument still says host-bound, and says it less loudly.**
The clock sampler excluded **117 of 141, 122 of 153 and 122 of 150** samples as
idle: the GPU is idle for **79.7 to 83.0%** of this render, against the
baseline's 87 to 88%. Removing 215 s of host-side f32 work is exactly what
predicts that shift. The sample counts also check the instrument: 141/153/150
samples at 2 s over 289/313/307 s renders is the window spanning its own render,
so #2305's orphaned-sampler defect is not present here.

Every render read `steps_observed={8}` and `dit_forwards=32`, wrote 25 frames
and a non-empty wav, and covered its own wall to better than **0.16%**
(unaccounted 0.315 / 0.408 / 0.473 s).

**All four checkpoint sha256 recomputed inside the lease and all four match
`ltx2_oracle_manifest.json`:** transformer `792a2bad...`, text encoder
`ef724361...`, video VAE `685b06ee...`, audio VAE `c52733d3...`.

### The build, and the ccache instruction that did nothing

The binary was built in the lease in **1404 s (23.4 min)**, 850 ninja targets,
named targets only, `-j 4`.

**`ccache` was mandatory, was configured exactly as `rc describe dgx:gpu0`'s
usage sheet requires, and hit nothing.** `ccache 4.9.1` was installed,
`CCACHE_DIR=/workspace/ccache` was exported, and all three of
`-DCMAKE_C_COMPILER_LAUNCHER`, `-DCMAKE_CXX_COMPILER_LAUNCHER` and
`-DCMAKE_CUDA_COMPILER_LAUNCHER` were passed. `ccache -s` reads
`Cache size (GB): 0.0 / 20.0` **before AND after** the build, the NAS
`CCACHE_DIR` holds **one file with zero written during the build**, and
`build.log` contains **zero** ccache lines of any kind, error included. A cache
that records no hits, no misses and no stores was not consulted. The leading
hypothesis is the CIFS `CCACHE_DIR` the sheet mandates; it is under `## Owed`
with the probe that would settle it, and it is reported rather than rounded up
into "ccache was used".

## Owed

- **`decode.video` FELL 1.46x AND NOTHING IN THE RANGE EXPLAINS IT.** 15.970 ->
  10.927 s, outside its own 1.71% spread, so it is a move and not noise. No
  commit between `0002ddfba` and `790c582bb` names the video VAE. The leading
  hypothesis is a shared-seam side effect, because that VAE's projections run
  through the same `vt::MatmulBT` that `LTX25-CONNECTOR-REPAIR` re-blocked and
  `VT-CPU-ELEM-DISPATCH` de-dispatched, and the way to settle it is one
  `ltx25-text-cond-ab.sh`-shaped A/B of those two commits with `decode.video`
  read off the phase table. **It is not attributed here**, because a leaf that
  moves for a reason nobody named is exactly the shape of a wall that moved by
  accident. Owner: unowned; sizing is one lease.
- **`ccache` IS MANDATORY ON THIS HOST AND DID NOTHING. ANSWERED, and it was
  CIFS.** The probe this bullet asked for ran as `rc` jobs `9b287a1f` and
  `e4793984`: `/workspace` is mounted `nounix`, `symlink(2)` on it returns
  `EOPNOTSUPP`, and `ccache` 4.9.1 takes every cache **and stats** lock by
  creating a symlink. Every store and every counter update was refused, which is
  why the counters read zero rather than reading misses. The CMake launcher was
  never the problem: it reached the compile line and produced
  `Cacheable calls: 9 / 9`. The fix and its measurement are
  [`eng-rc-ccache.md`](eng-rc-ccache.md), and this script now puts `CCACHE_DIR`
  on local disk and persists it through ccache's remote storage.
  Owner: `ENG-RC-CCACHE`,
  [#2473](https://github.com/mudler/vllm.cpp/issues/2473).
- **The oracle side has no spread.** `render_seconds = 93.8` is `n = 1` at the
  pin. Re-running upstream `n >= 3` at the same request would give the
  denominator its own error bar. Our own 8.03% is measured; theirs is unknown,
  and every ratio in `## Outcome` inherits that. Owner: unowned; sizing is one
  lease.
- **THE REMAINING GEMM IS STILL THE LEAF AND THIS ROW DECLARES NO CEILING.**
  Connector compute is 81.338 s of a 302.954 s render, **26.8%**, and the two
  tower phases are another 82.3 s. `LTX25-CONNECTOR-REPAIR` already sized the
  next traceable hypothesis — `KERNEL-GEMM-CPU-TILED`, a K-blocked macro-kernel
  that keeps its accumulators live across K panels — and this row does not open
  it. Owner: unowned.
