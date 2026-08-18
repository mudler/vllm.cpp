# LTX25-DECODE-SPEED — where an LTX-2.5 render's hours go, and the ranked levers

Row: `LTX25-DECODE-SPEED`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)).
Issue: [#1006](https://github.com/mudler/vllm.cpp/issues/1006).

This row produces **a finding and a ranked lever list. It ships no product code.**
Each lever it names becomes its own row, with its own issue, spec and fresh
review. The reason for that split is in §8.

## Now

`SPIKE`. The source half is complete. The runtime half ran both rungs and is
recorded in §1.3: rung 1 settled the memory *mechanism* and rung 2 mapped four
post-load compute regimes, none of which uses the GPU. Rung 2 then exited 1 and
the gate host went unreachable before its log could be read, so its failure
reason is **`REMOTE_UNVERIFIED`** and stays that way. Where an axis is
unfinished it says so rather than borrowing another axis's confidence.

**One lever this row filed has already landed.** Lever 2's dtype half, issue
[#1008](https://github.com/mudler/vllm.cpp/issues/1008), was taken by
`LTX25-DECODE-DTYPE` and merged as `d1b0ea3a8`, which is this branch's base. §3
is therefore written in the past tense for the f64 accumulation, §6 records the
landing rather than ranking it as available work, and `## Owed` does not list it.
The memory-format half is still owed, to that row.

**None of this row's measurements has a retrievable evidence artifact.** They are
on `dgx.casa`, which does not answer. §1.3 states it, `## Owed` carries it, and
[#1040](https://github.com/mudler/vllm.cpp/issues/1040) owns it. The
`REMOTE_UNVERIFIED` mark this row already applies to rung 2's *exit reason*
applies to the *passing* numbers too, and saying so is cheaper than a reader
discovering it.

## 0. What was asked, and what was actually wrong with the question

The dispatch asked why LTX-2.5 rendering is slow here, and named the video VAE
decode as the suspect on the strength of two records:

> *"Expect minutes, not seconds: most of a 320x192/25f render is spent
> single-threaded in the host VAE decode at 0% GPU."* — `docs/USAGE.md:873-874`
> at `332aed738`

> `Ltx2ConvVideoDecode` at 448x256/25f took **2681.02 s**, and the decode's own
> exact heap peak at that size is **361.72 MiB** —
> [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` items 2 and 3.

Both survive checking as *records*, and §1.1 records where each number came from
and what conditions each carries — the 2681.02 s in particular is one sample on
a contended box, and it measures an arm that has since changed (§3). The
**framing** did not survive: this row was dispatched believing that the
un-ported `memory_efficient_decode.py` was the leading candidate for a missing
~59 GiB. It is not, and it cannot be — §4 shows the arithmetic that closes that
hypothesis rather than leaving it open. The 59 GiB and the 2681 s are **two
unrelated defects** that happen to be adjacent in the log, and treating them as
one problem is what kept both open.

## 1. Provenance of every number this row rests on

AGENTS.md's rule that a number quoted often becomes treated as measured applies
to this row's own inputs. Each is traced to its origin before it ranks anything.

### 1.1 Numbers that survive

| Number | Origin | Status |
|---|---|---|
| decode wall 2681.02 s @ 448x256/25f | [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 2, exact-heap-accounted probe on the shipped conv VAE | MEASURED — **one sample, on a contended box** |
| decode heap peak 361.72 MiB, largest single allocation 99.20 MiB | same table, exact `operator new` accounting; the analytic ladder predicted the largest allocation to 3 s.f. at two scales | MEASURED |
| auto tiling resolves to 1 tile / 1 chunk at 448x256/25f | same spec, `kLtx2AutoCases` golden, executed at the pin and asserted in `test_ltx2_tiling` | MEASURED |
| ~7.25 TFLOP of dense 3x3x3 convolution in one 448x256/25f decode | derived this row from the LTX-2.5 conv VAE config read out of the checkpoint header, over 42 convs | COMPUTED |
| ~3.5 TMAC for the same decode | [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md):383-384, stated without a derivation | **UNDETERMINED — see below** |

**2681.02 s carries its conditions, because it is the denominator of this row's
headline 2.7 GFLOP/s.** It is a **single sample**, and its own source records
what else was running: *"a full 405-target build and a 416-test `ctest -j 6` ran
on the same box during the window"*
([`ltx25-tiled-decode.md`](ltx25-tiled-decode.md):377-380), with `uptime` load
average **30.2** at build end (`:454`). That contention is exactly why that spec
quotes the exact heap counter and refuses its own `MemAvailable` delta. The
heap number is immune to it; a **wall** is not. So 2681.02 s bounds the decode
from below on a loaded 20-core box, and no idle-host repeat exists. AGENTS.md
requires an idle-host same-binary A/B before a figure is accepted, and this one
has not had it.

**The 3.5 TMAC figure does NOT corroborate the 7.25 TFLOP, and this row
withdraws the claim that it does.** An earlier draft called them "two
independent derivations agreeing to 3.6%" (3.5 TMAC = 7.0 TFLOP against 7.25
TFLOP). Checked: `3.5 TMAC` appears **once**, in
[`ltx25-tiled-decode.md`](ltx25-tiled-decode.md):383-384, as *"about 3.5 TMAC at
the 1.9 GMAC/s the 64x64/9f run measured"*, with **no derivation of either
factor** — and it does not reconcile with its own sentence in either direction:
2681 s x 1.9 GMAC/s is **5.09 TMAC** (+45% on 3.5), and 3.5 TMAC at 1.9 GMAC/s
is **1842 s** (-31% on 2681). A number that fails its own stated arithmetic
cannot corroborate anything. **7.25 TFLOP therefore stands alone**, as one
COMPUTED figure from the checkpoint header, and the levers it ranks are ranked
on that plus the oracle source, not on a corroboration that does not exist.

### 1.2 A number that does NOT survive as stated

`docs/USAGE.md:868` at `332aed738` says a 448x256/25f render *"loses about 59 GB
in 24 seconds inside the decode"*. The sentence is **half supported, and the
half that fails is the one that has been driving work.**

**Two different sentences are in play and this row keeps them apart, because an
earlier draft did not.** `:868` is the *"inside the decode"* claim, and it is
the one corrected below. `:873-874` is a *different* claim — *"most of a
320x192/25f render is spent single-threaded in the host VAE decode at 0%
GPU"* — quoted in §0 and corrected in §5 for a different reason. A correction
aimed at `:873-874` would rewrite the wrong sentence.

*Supported:* the 59 GiB is real, and the fall is on the decode **side** of the
denoise-to-decode boundary — `benchmark-record.md:21116-21129` shows both
denoise phases finishing and draining first (§4.2).

*Not supported:* that the decode **allocates** it. It cannot —
`Ltx2ConvVideoDecode`'s exact heap peak at that size is 361.72 MiB, a factor of
170 (§4.1), and [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome`
item 3 had already noted the 24-second fall *"cannot have been a completed
decode"*, which takes 2681 s.

"On the decode side of a boundary" and "inside the decode" are different claims,
and the doc collapses them. §4 keeps the first and replaces the second.

### 1.3 What this row measured itself

A probe was queued on `dgx.casa` behind `$HOME/gpu.lock`. It never jumped the
queue: it waited out a `llama-imatrix` holder and two other waiters, plus its
own 85 GiB sustained-headroom guard, and acquired the lock at 10:59:57Z.

Host: `kairos-17dd`, GB10, driver `580.173.02`, `clocks.max.sm` 3003 MHz,
persistence mode **Disabled**, boot id `03717c9d-63c8-4652-a8fe-a63d012c5718`,
20 cores. Build `0e1bee42f`, CUDA on, arch `121a`, `vllmcpp-build:gb10`, the
recipe of `~/work/ltx25-e2e` with `--device cuda` and the full `--encoder`.

**Per the benchmarking guide, nothing here is a throughput ratio.** There is no
denominator (§7), so these are one-sided phase attributions of our own engine.
They say where our bytes and time go; they do not say what the gap to a
reference is, and this row claims none.

**The instrument had a defect, and it is recorded because it is the exact
failure this project keeps paying for.** The sampler found its target with
`pgrep -f ltx2-gen`, which matches the **`sudo`/`docker` wrapper** — whose RSS is
meaningless — rather than the workload inside the container. Its per-PID columns
came back empty, and an empty `Anonymous` column reads as *"the process
allocated nothing"*. A side-car sampler using `pgrep -x` (comm name, so it finds
the real `ltx2-gen`) was started alongside, and every per-PID figure below comes
from it. **A broken instrument fails toward a verdict about the code.**

**No evidence artifact for any of this is retrievable, and that is stated here
rather than left for a reader to discover.** `.agents/verification.md` requires a
gate report to carry the immutable SHA, the exact command, the environment, the
exit status **and the evidence path**. The first four are above. The fifth is a
path on one host: rung 1's and rung 2's sampler CSVs and rung 2's `run.log` are
on `dgx.casa`, as is §1.4's `render8-console.log`, and **`dgx.casa` does not
answer** — `ping -c 2 -W 3 dgx.casa` exits 1 with 100% packet loss and
`Destination Host Unreachable`, checked 2026-08-16, which is this box's
documented unified-memory OOM-reboot mode and needs a physical power cycle.
Nothing in this tree reproduces these numbers.

This row already applies `REMOTE_UNVERIFIED` to rung 2's **exit reason**. The
same honesty is owed to the **passing** measurements, and this is it: they are
recorded, they are not reproducible from anything committed, and every figure
below inherits that. It is filed as
[#1040](https://github.com/mudler/vllm.cpp/issues/1040), which owes the capture
and the cadence reconciliation, and is listed under `## Owed`.

#### Rung 1 — 448x256/25f, the size that fails

It failed, and **not where anyone has been looking**. The watchdog fired at
t=701 s with `MemAvailable` at 38.36 GiB, **during the text-tower load, before a
single denoise step**. 0 frames, `EXIT=137`. 248 per-PID samples at a nominal
2 s cadence.

**The cadence does not close, and no rate for it was recorded.** 248 samples at
2 s cover **496 s of a 701 s run**, and the per-phase split is short the same
way: 192 samples against a 450 s phase-A window (384 s of nominal coverage, 85%)
and 56 against a 164 s phase-B window (112 s, 68%). 192 + 56 = 248, so the
counts are internally consistent with each other and **not** with the wall. Some
combination of dropped samples and a cadence that slipped under load is the only
reading, and **the sampler recorded no drop count**, so this row cannot say
which or in what proportion. Every figure below is therefore stated as a sample
count or a fraction of samples; where a wall-clock window is named it comes from
the run's own timestamps, not from multiplying samples by 2 s. The raw CSV
settles it in one pass and is on the box —
[#1040](https://github.com/mudler/vllm.cpp/issues/1040).

| | phase A: DiT staging to device | phase B: host text tower |
|---|---|---|
| window | t 0 -> 450 s | t 453 -> 617 s |
| CUDA compute-app footprint | 4.22 -> **35.20 GiB** (+30.98) | 35.48 -> 35.54 (+0.06) |
| process `VmRSS` | 2.11 -> 12.94 (+10.83) | 13.04 -> **44.77** (+31.73) |
| process `Anonymous` | 0.26 -> 0.27 (**+0.01**) | 0.14 -> **27.57** (+27.43) |
| `MemAvailable` | 110.25 -> 66.75 (**-43.50**) | 66.69 -> 39.91 (-26.78) |
| GPU utilization | mean 0.2%, **zero in 164/192 samples** | mean 0.0%, **zero in 56/56** |
| CPU busy | **0.15 cores** of 20 | **0.39 cores** of 20 |
| threads | 3 | 3 |

**Three findings, each measured rather than argued.**

1. **§4.3's mechanism is confirmed, and it is stronger than stated there.** Over
   the staging phase the anonymous heap grew by **0.01 GiB against a 43.50 GiB
   fall — 0.0% of it.** The fall is the device counter (+30.98) plus file-backed
   RSS (+10.83), which together account for **96%**. Any sampler reading `VmRSS`
   or `Anonymous` is structurally blind to this, which is why every previous
   attempt came back empty. `--query-compute-apps=used_memory` reports it
   per-PID on this box and was available the whole time.
2. **#1016 is no longer merely "not excluded" — it is measured.** The +10.83 GiB
   of file-backed RSS during staging is the mmap source pages that no `ltx2_*`
   loader ever releases, growing in lockstep with the device copy **on the same
   unified pool**. It is ~25% of the load-phase fall.
3. **The staged total lands within 1% of the computed figure.** The plateau is
   36396 MiB = **35.54 GiB**, against 35.32 GiB derived from the loader's own
   contract. The device-side model is validated.

**One discontinuity inside that table is not noise and is not explained.**
`Anonymous` ends phase A at **0.27 GiB** and begins phase B at **0.14** — a
**0.13 GiB decrease across a 3-second boundary** — while `VmRSS` (12.94 ->
13.04) and `MemAvailable` (66.75 -> 66.69) are both continuous across the same
gap. So one anonymous buffer of about that size was freed exactly at the end of
the staging loop and nothing else moved. The plausible candidate is the staging
loop's own transient: `Ltx2StreamDitToDevice` holds one
`std::vector<uint8_t> host` per tensor and drops it at the end of each iteration
(`ltx2_loader.cpp:739-749`), so the last one dies when the loop exits. **That is
a hypothesis, not a measurement** — nothing in the sampler attributes a free to
a call site. It does not touch finding 1, which is about the **+0.01 GiB of
growth within** phase A rather than about the level at either end.

**A separate lever falls out of phase A that no record names:** the DiT takes
**450 s — 7.5 minutes — to stage**, with the GPU idle in 85% of samples and
0.15 of one core busy. It is neither GPU-bound nor CPU-bound. That is the serial
per-tensor `Alloc` + copy + `Synchronize` loop at `ltx2_loader.cpp:738-756`
(`backend.Alloc` at `:747`, `backend.Synchronize` at `:749`), ~3,504 round
trips.

**The staging RATE, recomputed from the counters, and it is not one number.** An
earlier draft of this row said *"at ~52 MiB/s"*, which contradicted the same
sentence's own inputs. 52.0 MiB/s is the **plateau divided by the whole run** —
35.54 GiB = 36393 MiB over ~700 s — and the whole run is not the staging phase.
Over the staging window itself:

| window | bytes staged | wall | rate |
|---|---|---|---|
| rung 1 phase A | 4.22 -> 35.20 GiB = **31723 MiB** | 450 s | **70.5 MiB/s** |
| rung 2 staging | `capp_mib` 4322 -> 36396 = **32074 MiB** | 251 s | **127.8 MiB/s** |

**Same host, same boot id, same build, and a 1.81x spread** for the same ~32 GiB
of tensors. The 52.0 MiB/s figure is not merely low, it is inconsistent with its
own arithmetic in the other direction too: 450 s at 52 MiB/s is 22.9 GiB against
a recorded 35.54 GiB plateau. **This row records the spread rather than the
slower number**, and does not explain it: two staging runs differing 1.81x on
one box is itself unattributed, and the obvious candidate — contention from
whatever else held the box — is not measured, because the sampler recorded no
system-wide load column. Any row that takes the staging lever must measure the
rate itself rather than inherit either figure.

The lever is filed as [#1021](https://github.com/mudler/vllm.cpp/issues/1021),
alongside [#1016](https://github.com/mudler/vllm.cpp/issues/1016), which is the
same loop.

**And the shape of the 448x256/25f failure is not what the record says.** With
the text tower on the path, the render is out of headroom **at load**: 35.54 GiB
device plus 44.77 GiB host RSS is **~80 GiB before any compute at all**. The
`benchmark-record.md` trace §4.2 quotes was a *prompt-embeds* run with no tower,
which is why it had 75.2 GiB free to plateau at. **These are two different
failures at the same resolution**, and conflating them is available to anyone
reading either trace alone.

#### Rung 2 — 320x192/25f, the size that was expected to complete

It did not. `LOCK_ACQUIRED 11:14:41Z avail=104 GiB`, then **`EXIT=1`, 0 frames,
elapsed 2407 s, 1082 sampler rows.** No watchdog was armed on this rung, so the
non-zero exit is the engine's own.

**The failure reason is `REMOTE_UNVERIFIED`.** `dgx.casa` became unreachable
minutes after the probe printed `PROBE END` and has not answered since — `ssh`
returns *"No route to host"* and `ping` loses 100% of packets across repeated
attempts. `run.log` was never read. Per AGENTS.md, unknown is not absence and is
not success: **this row does not know why rung 2 exited 1**, and it does not
guess. The CSV and `run.log` are on the box and are recoverable when it returns.
Whether the box rebooted, and whether this render or a peer's job caused it, are
both unverified. Note that `MemAvailable` had risen to 66.51 GiB shortly before
the exit, which does not fit a memory exhaustion at that moment.

**What the 1082 rows did establish before it exited**, all per-PID at 2 s from
the side-car, and all independent of the reason for the exit:

| t | regime | evidence |
|---|---|---|
| 0-251 s | DiT staging to device | `capp_mib` 4322 -> 36396 (35.54 GiB); GPU 0-2%; 0.15 cores |
| ~251-460 s | host tower load | `Anonymous` -> ~15 GiB; GPU 0 |
| ~460-620 s | **host text encode, multi-threaded** | 22 threads, `utime` +327 s over 113 s = **2.9 cores**; GPU 0 |
| ~700-2250 s | **one thread**, across a ~1550 s span | `utime` +116.7/117 s, then +142.2/142 s, then +648.9/649 s = **1.000 core** in each window; RSS, `Anonymous` and `capp_mib` all flat; GPU 0 |
| ~2260-2390 s | **a 15-core burst** | `utime` +1991.9 s over 132 s = **15.1 cores**; RSS 37.8 -> 43.7; GPU 0 |
| ~2390-2490 s | **~30 GiB released** | RSS 43.7 -> 13.9, `Anonymous` 39.9 -> 13.5, `MemAvailable` -> 66.51 |

**The 1.000-core row is INTERPOLATED across its span, and an earlier draft read
it as continuous.** The three `utime` windows total **908 s** of the ~1550 s
span, so roughly **642 s inside that regime is unsampled** by the `utime`
columns. What is measured is that in each of three windows totalling 908 s the
process accumulated CPU time within 0.3 s of wall — one runnable thread — and
that RSS, `Anonymous` and `capp_mib` did not move. What is *inferred* is that
nothing else happened in the gaps. That inference is the ordinary one and it is
still an inference; the row that names this window must not inherit it as a
measurement.

Two things are worth carrying forward from that table even without the exit
reason. **The engine does release the text tower** — the ~30 GiB drop at the end
is it — so lever 5's "held across the decode" framing is wrong for this
configuration too, and #1014 should be read against this. And **the render has
at least four distinct compute regimes after load** (2.9 cores, 1.0 core, 15.1
cores, then a free), **none of which uses the GPU**, which is the disjunction
#1024 records: nothing in the tree timestamps which phase any of them is.

**The 1.000-core window is the single largest block of the run** — a ~1550 s
span of a 2407 s run, of which 908 s carries a `utime` measurement — and it is
the one this row was dispatched to explain. It cannot yet be *named* as the VAE
decode, because no frame was written and no phase boundary is emitted (#1010).
Naming it is the first thing the follow-up row should do, and it needs #1010
rather than another probe.

**Rung 2's cadence does not close either.** 1082 sampler rows over 2407 s at a
nominal 2 s cadence would be 1204, and the side-car's per-PID set is smaller
still: §5 works from **347** per-PID samples. 347 at 2 s spans 694 s, which does
not reach the regimes this table places past t=2250 s. The two sets are not the
same instrument — the 1082 rows are the whole sampler, the 347 are the side-car
that §1.3's instrument note describes — but **no record says when the side-car
started, when it stopped, or how many samples either dropped**, so the wall each
covers is not derivable from what is written down. §5 therefore states sample
counts and fractions and states no minute figure. The CSVs settle it;
[#1040](https://github.com/mudler/vllm.cpp/issues/1040) owes them.

### 1.4 Evidence that already existed and had not been read as evidence

The completed 320x192/**49f** render of 2026-08-15
(`~/work/ltx25-e2e/render8-console.log` on `dgx.casa`) sampled `MemAvailable`
and 1-minute load average every 2 minutes for its whole 8598 s. Nobody had used
it to attribute a phase. It does:

* `LOCK_ACQUIRED 18:12:52Z avail=115 GiB`; frames written 20:29; audio at 20:36;
  `EXIT=127 ELAPSED=8598s` (127 is `ffmpeg` absent from the image — the render
  itself completed, 49 frames on disk).
* `MemAvailable` falls 115 -> 43 GiB over the first ~10 minutes and **never
  returns**. That is ~72 GiB acquired during load and held for the whole run.
* From 18:22 to 20:34 — **~2 h 07 m of a 2 h 23 m render, about 89% of its
  wall** — `MemAvailable` is flat at 43-46 GiB and the **1-minute load average
  sits at 1.0-1.3 on a 20-core box**, with two brief excursions (8.66, 12.36) at
  what are evidently phase boundaries.

A sustained load of ~1.1 for two hours is one runnable thread. That is a
measured, pre-existing observation of the single-threading, independent of this
row's probe and of `docs/USAGE.md`.

**What load alone cannot decide** is whether that one thread is computing on the
host or blocking on a GPU. Both look like load 1.0. Separating them is exactly
what rung 2's GPU-utilization column is for, and it is why this row queued for
the GPU rather than quoting the doc.

## 2. What the oracles actually do

Six checkouts, each verified at the SHA the dispatch named. `git status
--porcelain` was clean on five; vLLM had 6 deleted files (`build_rust.sh`,
`requirements/build/*.txt`) and its `vllm/` source tree is clean.

| Oracle | SHA | Implements LTX-2.5? |
|---|---|---|
| Lightricks LTX-2 | `fd4ded7f2` | **yes — the reference implementation** |
| `diffusers` | `c6da9936e` — the recorded pin | **yes — both LTX-2.5 decode arms** |
| SGLang | `f63458b5b` | no — LTX-2 and LTX-2.3 only |
| vLLM-Omni | `a4ea67a21` | no — recipes stop at 2.3 |
| vLLM | `555967922` | no — no LTX, no diffusion VAE decode at all |
| SGLang-Omni | `748a0b437` | no — no LTX, no video VAE, no video model |

Two of those cells contradict what this campaign has been recording, and both
contradictions matter. They are §2.2 and §2.6.

### 2.1 Lightricks LTX-2 — the primary reference

Paths are relative to `packages/ltx-core/src/ltx_core/` unless stated.

**Device: GPU, decided at build time, not at call time.** The decoder is
constructed directly onto a device —
`packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1139`
(`decoder = self._decoder_builder.build(device=self._device, dtype=build_dtype).eval()`),
`loader/single_gpu_model_builder.py:267-288` with `:273` defaulting the device
to CUDA, and `devices.py:29-39` resolving CUDA -> MPS -> CPU. There is no
`.cuda()` on the decode path because placement already happened.
*Positive control for that negative:* `grep -rn "\.cuda()" packages/*/src`
returns 4 hits, all in quantization weight prep
(`quantization/blockwise/_impl.py:129,297`,
`ltx_kernels/blockwise/linear.py:193,263`), and `grep -rn "to(device="` returns
90 — the search reaches the files.

**Ops: plain `torch.nn.Conv3d`.** `CausalConv3d` at
`model/video_vae/convolution.py:266`, its `nn.Conv3d` built at `:292-302` with
`padding=(0, H//2, W//2)`, and the **single conv call site for the entire
decoder** at `convolution.py:312`. Temporal padding is done by hand at
`:306-307` (causal) / `:309-311` (non-causal). No Triton, no CUDA extension —
`packages/ltx-kernels/src/ltx_kernels/vae/` is entirely fused neighborhood
attention for the *diffusion* decoder.

**Tiled or whole-tensor: both exist, and at our size upstream runs whole-tensor.**
`decode_video` (`model/video_vae/conv_video_decoder.py:486-506`) selects
`tiled_decode` (`:383`) when a tiling config is present, else one whole-tensor
`self(latent)` at `:504-506`. Pipelines default to `AUTO_TILING`
(`ltx_pipelines/distilled.py:197`, `:352`; `tiling.py:859`), which resolves
through `ltx_pipelines/utils/helpers.py:119-146` -> `:75-116` to the conv-VAE
defaults at `helpers.py:62-63`: long-side tile 768 / overlap 64, frames tile 80
/ overlap 24. The scale factors are `(time=8, height=32, width=32)`, bound as
`VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()` at
`packages/ltx-core/src/ltx_core/types.py:70` — there is no literal tuple
anywhere; `.default()` at `types.py:32-33` is what returns those three numbers,
and `ltx_pipelines/utils/constants.py:118` binds the same name a second time.
With those factors, a
448x256/25f request is 4 latent frames and a 768/448-px spatial envelope — under
every threshold, so `split_by_size` returns the untiled interval
(`tiling.py:199-200`) and `split_temporal_causal` short-circuits identically
(`tiling.py:239-240`). **One tile, one chunk.**

That is the same resolution our own `kLtx2AutoCases` golden records
([`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 1). Upstream
and this port agree that tiling is inert at the failing size, so **tiling is not
a lever for it** on either side.

**Dtype: bf16 end to end, no autocast, no f32 promotion.** `distilled.py:109`
sets `self.dtype = torch.bfloat16`; `blocks.py:1136-1139` casts the latent and
builds at that dtype; the decoder follows its own weights at
`conv_video_decoder.py:282-284` (`output_dtype = sample.dtype`;
`weights_dtype = next(self.parameters()).dtype`; `sample = sample.to(weights_dtype)`).
The only f32 override in the family is HDR
(`ltx_pipelines/utils/media_io/color_config.py:64-66`). `PixelNorm.forward`
(`model/common/normalization.py:32-40`) computes `mean(x**2)` **in the
activation dtype** — it does not promote.
*Positive control:* `grep -rn "autocast" packages/` returns **12** hits — seven
in `audio_vae/vocoder.py` (`:22`, `:525`, `:589`, `:593`, `:596`, `:597`,
`:608`), two comments in `blocks.py:1184-1185`, and three comments in the
DiffVAE transformer (`video_vae/transformer/swiglu.py:335,336`,
`video_vae/transformer/attention.py:77`). **Zero in any conv-decoder file**, and
`grep -rnc` confirms grep opened all three of them
(`conv_video_decoder.py:0`, `convolution.py:0`, `memory_efficient_decode.py:0`,
against 9 / 11 / 19 for `def `), so the zero is absence and not a bad path.

**Memory format: `channels_last_3d` (NDHWC), for weights AND activations.**
`memory_efficient_decode.py:617-627` (`CHANNELS_LAST_3D_WEIGHTS`) stores every
5-D conv weight as `channels_last_3d` at load; `:655-656` and every workspace
allocation (`:167`, `:226`, `:389`, `:442`, `:492`) pass `memory_format=`. The
plain `forward` is NCDHW. This selects a different cuDNN 3-D convolution kernel
family, and it is precisely the class of difference AGENTS.md says a token gate
cannot see.

**`memory_efficient_decode.py`, read in full (683 lines).** Its docstring
enumerates four optimizations at `:1-20`, and the module is **on by default** —
`memory_efficient: bool = True` at `blocks.py:1059`, its sole caller
`blocks.py:1090-1095` gated only on `memory_efficient and not diffusion_vae`,
and no pipeline passes `False`.

1. **Workspace buffers** (`:4-7`): one `[B, C, T+2, H, W]` tensor per block
   holds real data in `[1:-1]` with replicate padding written into `[0]` and
   `[-1]` (`_pad_workspace_temporal`, `:108-114`). This replaces
   `CausalConv3d`'s per-call `repeat` + `torch.concatenate`
   (`convolution.py:306-311`), which allocates a fresh `T+2` tensor **per conv**.
2. **In-place temporally-chunked Conv3d**, non-causal only (`:122-204`), writing
   back over `workspace[:, :C_out, 1:-1]`; aliasing is made safe by chunking
   with 1-frame boundary save/restore (`:159-169`, `:190-204`). Chunk size is
   searched `16 -> 3` by `_find_temporal_split_size` (`:91-105`) and engages only
   when `total_frames > 16` (`:151-156`).
3. **In-place norm / affine / SiLU** (`_pixel_norm_inplace` `:256-259`,
   `F.silu(..., inplace=True)` at `:318`, `:339`, `:519`).
4. **Free-before-conv** (`_causal_pad_free_and_conv` `:234-248`, with `del x` at
   `:245`), so input and output never coexist — `:15-16`.

Dispatch is `_memory_efficient_forward` (`:541-609`), per-block at `:592-607`.

**Streaming: a generator, at temporal-tile granularity, with no host offload.**
`tiled_decode` yields per temporal group and keeps at most the current buffer
plus the previous chunk alive (`conv_video_decoder.py:466-471`, rotate at
`:474-476`, drain at `:479-484`); the pipeline hands the iterator straight to
the encoder (`blocks.py:1140-1144`, `distilled.py:315`,
`ltx_pipelines/utils/media_io/encode.py:120-121`, `:175-178`). At 25 frames
there is one group and one yield, so streaming buys nothing at our size.
*Positive control for "no VAE offload":* `grep -rn "offload"` over
`packages/ltx-core/src packages/ltx-pipelines/src` returns **82** hits — the term
is live, `--offload {none,cpu,disk}` is documented at
`ltx_pipelines/docs/installation.md:90` — and filtering those 82 for `vae|decod`
returns **zero**. Offload is transformer-weights-only upstream.

**The LTX-2.5 conv VAE config, read from the shipped checkpoint header** (range
read of `ltx-2.5-video-vae-conv-bf16.safetensors`; the repo ships no copy — a
`grep -rn "decoder_blocks"` returns **9** hits, every one a code reference:
`types.py:64`, `model_configurator.py:87` and
`conv_video_decoder.py:43,52,166,181,192,207,222`):
`dims 3`, `latent_channels 128`, `patch_size 4`, `norm_layer pixel_norm`,
`decoder_base_channels 128`, **`causal_decoder FALSE`**,
**`timestep_conditioning FALSE`**, `spatial_padding_mode "zeros"`, blocks
`res_x(4), compress_space(m2), res_x(6), compress_time(m2), res_x(4),
compress_all(m1), res_x(2), compress_all(m2), res_x(2)`.

Two consequences follow from `model_configurator.py:90-91`: the **non-causal**
symmetric-pad branch (`convolution.py:309-311`) is the one that runs, and there
is **no decode noise injection, no `decode_timestep`, no AdaLN** in the shipped
LTX-2.5 conv arm at all.

**LTX-2.5 ships two video VAEs, selected from checkpoint metadata, and the
diffusion one is the recommended download** — `README.md:80-81`,
`model_configurator.py:18`, `:26-34`, `:242-250`. This port implements the conv
arm and refuses the diffusion arm by name. That refusal is correct and is not
this row's business, but every comparison below is about the **conv** arm and
says so, because the two have different ops, different tiling defaults and
different memory profiles.

### 2.2 `diffusers` — implements LTX-2.5, and no LTX-2.5 record in this tree treats it as the oracle

This is the first of the two contradictions, and it is the one that changes the
denominator answer.

**Anchored at the recorded pin, not at the checkout.** Every diffusers anchor in
this section is cited at `c6da9936e4bda83107943a16eb8682e9a37d8527`, the
revision [`oracles/diffusers.md`](../oracles/diffusers.md) records. An earlier
draft cited `3a2f35d4e`, which is the SHA the local checkout happens to sit on
and is **16 commits behind the pin** — `git rev-list --count
3a2f35d4e..c6da9936e` is 16 with and without `--no-merges`. An earlier count of
17 came from `git rev-list --count c6da9936e` in a **shallow** clone whose
`.git/shallow` holds exactly `3a2f35d4e` — it was measured across the shallow
boundary, and so counted the boundary commit itself. `3a2f35d4e` is an ancestor
of the pin, and the pin is on no branch in that checkout, so a `git pull` there
would not reach it. Re-checked at the pin, every line below holds
byte-identically, so nothing in the finding changes; the citation was simply
off-pin, and AGENTS.md's reason for pinning is that an unpinned upstream is not
reproducible.

`diffusers` at the pin implements **both** LTX-2.5 video decode arms:

* the convolutional arm, `AutoencoderKLLTX2Video`
  (`src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py:1025`, registered
  at `models/autoencoders/__init__.py:16`), and
* **the LTX-2.5 diffusion decoder**, `LTX2VideoDiffusionDecoderModel`
  (`ltx2_diffusion_decoder.py:700`, registered at
  `models/autoencoders/__init__.py:31` — *not* `models/__init__.py:31`, which is
  `AsymmetricAutoencoderKL`; in `models/__init__.py` the decoder is at `:61` and
  `:205`, and `:205` is the one anchor in this section the pin moves, from
  `:201`), whose
  docstring at `ltx2_diffusion_decoder.py:702` reads *"The LTX-2 diffusion video
  decoder, introduced in LTX-2.5."*, with its own pipeline
  `LTX2VideoDiffusionDecodePipeline` (`pipeline_ltx2_diffusion_decode.py:27`,
  docstring at `:29`).

Further 2.5-specific text: `ltx2_diffusion_decoder.py:477`, `transformer_ltx2.py:1118`
(`(LTX-2.5.1+)`), and `pipelines/ltx2/utils.py:272` naming LTX-2.5's Gemma-4
text encoder.

The two arms share a latent space, stated at `ltx2_diffusion_decoder.py:704-706`:
*"latents are interchangeable between the convolutional decoder and this one."*

**Conv-arm behaviour, for comparison with ours:** tiling flags default OFF —
`use_tiling = False` at `autoencoder_kl_ltx2.py:1169`,
`use_framewise_decoding = False` at `:1174`; tile minima 512/512/16 and strides
448/448/8 at `:1183-1190`. Decode dispatch is `_decode` at `:1266-1289`: framewise
at `:1278`, spatial tiling at `:1281` (requires width or height **above** the
512-px minimum), otherwise **one whole-tensor `self.decoder(z, temb, causal=causal)`
at `:1284`**. At 448x256 neither branch is taken even with tiling on. Dtype is the
single inherited pipeline dtype, bf16 in the shipped example
(`pipeline_ltx2.py:74`), with the cast running latents **to** the VAE
(`pipeline_ltx2.py:1642-1643`) and **no f32 promotion anywhere in the decoder** —
`PerChannelRMSNorm.forward` computes in the activation dtype
(`autoencoder_kl_ltx2.py:50-59`, mean at `:56`, sqrt at `:58`).
*Positive control:* grepping `\.float()|autocast|float32|torch\.float` across both
LTX autoencoder files returns exactly two hits, both the same `timestep_scale_multiplier`
scalar parameter (`autoencoder_kl_ltx2.py:967`, `autoencoder_kl_ltx.py:984`), while
the same grep hits `.float()` at `image_processor.py:203`.

The temporal chunking loop exists (`_temporal_tiled_decode` at
`autoencoder_kl_ltx2.py:1497`, loop at `:1510`) but **is dead for LTX by
default**: `enable_tiling` (`:1192`, setting `use_tiling` at `:1218`) never sets
`use_framewise_decoding`. *Positive control:* `use_framewise_decoding` has **15
hits under `src/diffusers` and 25 repo-wide** — the other 10 are LTX pipeline and
LoRA tests setting it to `False` — and three **other** VAEs do set it True
(`autoencoder_kl_hunyuan_video.py:712`, `autoencoder_kl_mochi.py:834`,
`autoencoder_kl_magvit.py:798`) — neither LTX VAE ever does.
**`autoencoder_kl_magvit.py:799` is a different symbol** —
`self.use_framewise_encoding = True`, *encoding* rather than *decoding* — and an
earlier draft cited it. An off-by-one that lands on a line reading
`use_framewise_?coding = True` is invisible to a reader checking the shape
rather than the string, which is why the string is quoted here.

**The one quantified memory statement in any oracle** is diffusers' own, and it
is about the *diffusion* arm at a size an order of magnitude past ours —
`ltx2_diffusion_decoder.py:319-323`: *"at 121 frames and 512x768 that is 3 x
5.67 GiB, which by itself dominates decode memory"*. Also
`ltx2_diffusion_decoder.py:297-300`, on why a neighborhood-attention mask must
never be materialized (*"a 69x64x96 stage needs 167 GiB"*), and `:768-770` /
`:797-799` on which stages tiling covers.

*Not found in diffusers:* any `memory_efficient_decode` analogue, any CPU or
threaded decode path, and **no `def enable_vae_tiling` anywhere in
`src/diffusers`** — 0 hits.

*Positive controls, respectively.* `grep -rn 'memory_efficient' src/diffusers`
returns **61** hits and **every one** is `memory_efficient_attention`
(`attention.py:208` and `attention_processor.py:361` are both
`def set_use_memory_efficient_attention_xformers(`, `:1967` is
`xformers.ops.memory_efficient_attention(`). And
`grep -n 'def enable_' src/diffusers/pipelines/pipeline_utils.py` returns
**six** entries:

```
1195:    def enable_model_cpu_offload(
1313:    def enable_sequential_cpu_offload(
1380:    def enable_group_offload(
2021:    def enable_xformers_memory_efficient_attention(
2076:    def enable_attention_slicing(
2296:    def enable_freeu(
```

An earlier draft listed **five**, silently omitting `:2021` — output grep cannot
produce, because grep does not skip a matching middle line, presented as
reproduced. The omitted entry is the one that bears on the control immediately
above it.

**The tiling negative needed narrowing too.** `grep -rn 'enable_vae_tiling'`
returns **59** hits repo-wide (56 under `examples/`) and **3** under
`src/diffusers`, all in one file: `enable_vae_tiling` is a **bool parameter** on
`pipelines/ltx/pipeline_ltx_i2v_long_multi_prompt.py:752`, documented at `:778`,
which delegates at `:829` to `self.vae.enable_tiling()`. So the true statement is
that diffusers exposes no *method* by that name, not that no pipeline offers the
knob — and the VAE-level `enable_tiling` it delegates to is defined 17 times
across the autoencoders. The claim as first written was false repo-wide.

diffusers assumes the decoder runs on the accelerator; **it has no host decode
to compare our 2681 s against.**

### 2.3 SGLang — a full LTX-2/2.3 lane, but no 2.5

SGLang ships a first-class LTX video-generation lane that runs in its own GPU
CI. Registry `python/sglang/multimodal_gen/registry.py:631-640` (LTX-2, HF path
`"Lightricks/LTX-2"`) and `:641-648` (LTX-2.3); pipelines
`runtime/pipelines/ltx_2_pipeline.py:324`, `:526`, `:916`, exported at `:926`;
GPU CI cases at `test/server/gpu_cases.py:408-412`, `:714-718`, `:741-746`.

**LTX-2.5 is absent.**
`grep -rnE 'ltx-2\.5|ltx_2_5|ltx2\.5|LTX25|LTX 2\.5' .` returns **0** hits
repo-wide, and **0** with `-i` as well. *Positive control, identical tool and
identical alternatives, only the version changed:*
`grep -rnE 'ltx-2\.3|ltx_2_3|ltx2\.3|LTX23|LTX 2\.3' python/sglang/multimodal_gen/`
returns **78** lines. An earlier draft of this row put that control at 113
without naming its command; 113 is reproducible, but only from a **different**
pattern — `grep -rniE 'ltx-2\.3|ltx_2_3|ltx2\.3'`, three alternatives and
case-insensitive — which is not the identical tool the sentence claimed. The
identical-pattern figure is 78 case-sensitive and 294 with `-i`. Either supports
the negative; the point of quoting a control is that a reader can rerun the
exact command, so the exact command is now written down.

Its decode is worth recording as a design reference even so. Two paths below are
written in full, because the abbreviated forms an earlier draft used do not
resolve: the decode stage is
`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py`
and the sharded conv is
`python/sglang/multimodal_gen/runtime/layers/parallel_conv.py`. GPU-resident —
`decoding_av.py:71` moves latents to
`get_local_torch_device()`, all under `torch.autocast` at `:80-84`. bf16 —
`configs/pipeline_configs/ltx_2.py:189` sets `vae_precision = "bf16"`,
overriding the base default `fp32` at `configs/pipeline_configs/base.py:206`.
Tiling **on** by default (`base.py:207`) and enabled at `decoding_av.py:86-87`,
but the spatial gate at `runtime/models/vaes/ltx_2_vae.py:1908-1910` needs a
latent extent above `512 // 32 = 16` and a 448x256 latent grid is 14 x 8 — so
**SGLang also takes the single whole-tensor pass at our size**. It carries two
levers it does not use for LTX: `_temporal_tiled_decode`
(`ltx_2_vae.py:2206`), reachable at 25 frames if `use_framewise_decoding`
(`:1759`) were flipped, and a streaming `decode_chunk` with per-conv cache
(`AutoencoderKLCausalLTX2Video`, `:2287`, `:2304`) wired only into the SANA-WM
stages. It also shards the decode spatially across ranks
(`ltx_2_vae.py:1742-1745`, `:1913-1925`, `layers/parallel_conv.py:667`).

A caution for any future row: `.agents/oracles/sglang.md:33-40` records
`gateable = yes` on the strength of an **LLM serving** run. That says nothing
about whether `sglang.multimodal_gen` builds and runs a video model here. A row
that reuses the mark for the diffusion lane would be asserting a state it has
not checked.

### 2.4 vLLM-Omni — 2.3 verified as the ceiling, and the adapter disqualified

The dispatch's two claims were checked exactly and **both hold**, with one line
number to correct.

`_PIPELINE_RECIPES` is declared at `ltx2_recipes.py:161` with its entries at
**`:162-166`** and closing brace at `:167`; the version axis takes only `"2"`
and `"2.3"`, and `:170-175` raises for anything else. The dispatch's
`:162-166` is right for the entries; cite `:161-167` for the whole dict. The
component table matches — `ltx2_components.py:105-111`, raising at `:114-119`.

**No LTX-2.5 anywhere in the repo.** *Positive controls:* `LTX-2` hits abound
(`ltx2_transformer.py:171,360,539,556,585,709,790,961-962`) and
`grep -rn "2\.3" vllm_omni/diffusion/models/ltx2/ recipes/LTX/` returns **41**
matching lines — 26 under the module and 15 in the recipe — over exactly those
two paths, which is the scope the figure is only true on. (Counting
*occurrences* rather than lines gives 42; one line carries two.)

Worth recording because it is a silent-wrong-answer shape: version detection
`detect_ltx_model_version` (`ltx2_components.py:141-169`) is binary — `"2.3"`
(`:150,160,164,166`) or `"2"` (`:169`) — and the fallback only logs
(`:168`). **An LTX-2.5 checkpoint on the native path is silently treated as
LTX-2**, failing later at weight load rather than being refused by name.

`DiffusersAdapterPipeline` (`pipeline_diffusers_adapter.py:54`) carries
`supports_request_batch = False` at **`:68`** and
`supports_step_execution: bool = False` at **`:69`** — both confirmed. All four
step-execution hooks raise `NotImplementedError` at `:153-175`, and with request
batching off `_resolve_execution_mode` **raises unless `max_num_seqs <= 1`**
(`diffusion_engine.py:204-210` — the guard is `_max_num_seqs(od_config) > 1`,
so `<= 1` and not strictly `== 1`). `forward()` at `:181-189` is a single
black-box `self._pipeline(**kwargs)` over a stock
`diffusers.DiffusionPipeline` (`:116`), with CFG parallel, sequence parallel,
non-`none` caching, `enforce_eager` and DDUF-with-omni-quantization all refused
up front (`:195-232`) — step execution is **not** among those refusals.

**How step mode is actually refused, because an earlier draft named the wrong
mechanism and the real one is worse.** That draft said
`DiffusionEngine._resolve_execution_mode` (`diffusion_engine.py:193-211`)
*"cannot pick step mode"*. It can, and it does:
`:200-202` reads `if self.step_execution: self.supports_request_batch = False;
return DiffusionExecutionMode.STEP_BATCH`, resolved from `od_config` alone
without ever touching a pipeline object. The only production gate on step support
is `DiffusionModelRunner.__init__`, which **raises** at
`worker/diffusion_model_runner.py:253` via `_supports_step_mode()` (`:600-602`)
-> `supports_step_execution()` (`models/interface.py:103-106`).

And that gate does not fire for this adapter. `interface.py:106` is
`isinstance(pipeline, SupportsStepExecution)` against a `@runtime_checkable`
`Protocol` (`interface.py:47-48`), and a runtime-checkable protocol `isinstance`
tests **attribute presence only — it never reads the boolean's value**.
`DiffusersAdapterPipeline` declares the attribute *and* defines all four hooks,
so it satisfies the protocol structurally and **passes**. Executing the repo's
own `supports_step_execution` against a stub carrying the adapter's exact
attribute surface returns `True`; a negative control with the flag `True` and no
hooks returns `False`. **No production code anywhere reads the value of
`supports_step_execution`** — the flag is decorative, and step mode for this
adapter is refused only at call time, by the `NotImplementedError` inside each
hook.

The disqualification is unchanged and the reason is now the right one: this is a
reference-degraded serial eager wrapper, the exact analogue of benchmarking vLLM
with `--enforce-eager`. What changed is that the refusal is a runtime exception
rather than a configuration gate, which is a weaker thing to be standing on and
is recorded as such.

For the 2.0-2.3 generations vLLM-Omni's decode is GPU-resident
(`interface.py:92` states *"VAE(s) (always on GPU)"*; the offload backends pull
them back at `offloader/sequential_backend.py:234-239` and
`layerwise_backend.py:301-306`), bf16 at one pipeline dtype
(`ltx2_components.py:325`, `data.py:970-975`), whole-tensor by default
(`vae_use_tiling` False at `data.py:697-698`), and its tiling is **spatial
only** — every tile carries the full frame range
(`distributed/autoencoders/autoencoder_kl_ltx2.py:102`, `:137`), unlike its WAN
VAE which does chunk over time (`autoencoder_kl_wan.py:121`). The decoder math
is diffusers' (`autoencoder_kl_ltx2.py:7`, `:76`, `:152`); vLLM-Omni's own
contribution is the distributed tile executor.

**The shipped engine never imports Lightricks `ltx_core`, and the earlier form
of that negative was wrong in both directions.** An earlier draft said
`ltx_core|ltx_video` *"returns zero"*. Repo-wide it returns **4**:
`tests/e2e/accuracy/ltx/run_ltx_reference.py:73,74,115` are three **real
imports** of `ltx_core`, and
`vllm_omni/diffusion/models/schedulers/scheduling_flow_match_euler_discrete.py:275`
is an upstream provenance URL inside a docstring. Scoped to `vllm_omni/` it
returns **1** — that URL — so "`vllm_omni/` has 0 hits" is also wrong. The
accurate statement is that **`vllm_omni/` contains no import of and no call into
`ltx_core` or `ltx_video`; its one textual match is a docstring URL, and the only
real imports live in the accuracy test that runs the reference side by side.**
*Positive control:* `grep -rn 'Lightricks'` returns 37 hits, including both
`ltx2_transformer.py:1` and `scheduling_flow_match_euler_discrete.py:275`, so the
pattern reaches the files.

**Its maintainers' own memory note, quoted with the hedges it carries.**
`recipes/LTX/LTX-2.md:315-316` reads: *"LTX-2 one-stage **previously** loaded and
peaked at about 73.5 GiB on one H200 141GB; **remeasure on the deployed commit
and hardware**."* Both emphasised clauses are the source's own, and an earlier
draft of this row dropped both — turning a self-flagged historical figure into a
current one, which is precisely the failure this spec exists to prevent and which
§1.1 applies to this row's own inputs. `:317-318` gives LTX-2.3 no measured
footprint at all, only a sizing recommendation: *"Start on a 96GB-class GPU or
use CPU/layerwise offload on smaller devices."* Neither line is a measurement
this row may compare against.

### 2.5 vLLM — implements nothing here, so the primary-reference rule does not bind

`grep -ril "ltx"` over the whole vLLM tree returns **2 files, and both are
false positives**: `vllm/entrypoints/serve/instrumentator/static/swagger-ui-bundle.js`,
where the match is the substring `ltX` inside `tiltX:0` in two lines of minified
vendored JavaScript, and
`docs/assets/features/speculative_decoding/speculators-user-flow-light.svg`,
where it is `LTx`/`lTx` inside a base64 image payload. `grep -ril "lightricks"`
returns **0**. So the conclusion survives exactly — **no LTX implementation
exists in vLLM at this revision** — and it is now stated in a form a reader can
reproduce, rather than as a zero that is not one. One tool caveat, because it
changes the answer: `git grep -il "ltx"` returns **8**, matching raw bytes inside
tracked binary PNGs under `docs/assets/`; the claim holds for `grep -ril` only.

*Positive control:* the same machinery finds `Qwen3ForCausalLM` at
`vllm/model_executor/models/registry.py:196`, with 48 `Qwen` hit **lines** in
that one file (72 occurrences). The registry has no image- or video-generation
section at all. The newer `vllm/models/` package holds only `deepseek_v32`,
`deepseek_v4`, `inkling`, `minimax_m3` and its `__init__.py`.

vLLM contains VAE code but no diffusion decode: `CheersVAEDecoder`
(`models/cheers.py:223-281`) is 2-D `nn.Conv2d` only and feeds SigLIP in the
*understanding* path (`cheers.py:677-696`); `CheersVAEModel` is encoder-only
(`:284-285`); `bagel.py:546-560` and `cosmos3.py:59` only route VAE weights at
load. The one "diffusion" registry entry,
`"DiffusionGemmaForBlockDiffusion"` (`registry.py:400-402`), is a text
block-diffusion LLM.

So under AGENTS.md's *"where it implements nothing"* branch, LTX-2.5 legitimately
falls to a secondary oracle.

### 2.6 SGLang-Omni — nothing, and the campaign should stop expecting otherwise

This is the second contradiction: SGLang-Omni was carried as a live candidate
for this model class, and it is not one.

`grep -rniE 'ltx|lightricks' .` over the whole tree returns **4 lines, all false
positives** — `lTx`/`LTx` inside base64 audio fixtures at
`benchmarks/tts_serving/voice_upload_fixtures.py:85`, `:105`, `:124`, `:148`,
one file. **The `-i` is load-bearing**: case-sensitive the same command returns
0, which would make "4 false positives" unreproducible, and an earlier draft
wrote the pattern without it. *Positive control:* the identical command over
SGLang proper returns **1885 lines across 129 files**, and within sglang-omni
`grep -rn 'Conv1d' .` returns **57** hits, so the tree is greppable and the
pattern is right. (`git grep -niE 'ltx'` over SGLang gives 1891, a different tool
and a different pattern; the figure above is the `grep -rniE` one.)

Its registry (`sglang_omni/models/registry.py:98`, `:136`) scans
`sglang_omni/models/`, whose complete contents are ASR, TTS, music and omni-chat
architectures. Its only VAEs are 1-D audio (`minimax_music3/dav.py:2`, `:114`);
its only `Conv3d` is a VLM patch embedding on the *input* side, which both
models replace with a Linear for speed
(`ming_omni/components/vision_encoder.py:104,122,171,342`,
`qwen3_omni/components/image_encoder.py:26,31`).

No LTX, no video VAE, no diffusion video pipeline, at any generation.

## 3. What our decode was at `332aed738`, and what has since changed

Anchors are in this tree at `332aed738`, the SHA this row's source half was read
at. **One finding in this section has since been repaired and this section is
written in the past tense for it**: the f64 accumulation below landed as
[#1008](https://github.com/mudler/vllm.cpp/issues/1008) /
`LTX25-DECODE-DTYPE` and is **gone from `main`** as of `d1b0ea3a8`. What that
row took and what it deliberately did not are set out where the f64 paragraph
used to make its claim. Everything else in this section still describes the tree
today.

**It is the CPU reference arm, and it says so.** The header block at
`include/vllm/model_executor/models/ltx2_video_vae.h:47-54` and the file block
at `src/vllm/model_executor/models/ltx2_video_vae.cpp:25-49` both record that
every buffer is f32 because this is a reference arm, that upstream instead runs
the checkpoint dtype, and — `ltx2_video_vae.cpp:46-49` — that **"PHASE L6 OWES
THE PRODUCTION ARM... this file is a correctness reference, not the shipping
path, and no memory or throughput number should be taken from it."**

That annotation is honest and predates this row. What it does not say, and what
this row establishes, is that **the correctness reference is what production
runs today**: `src/vllm/multimodal/ltx2_video.cpp:3258` calls
`Ltx2VideoDecodeStreaming` on the render path, which reaches
`Ltx2ConvVideoDecode` through `ltx2_video_vae_tiled.cpp:113` and `:369`. There
is no second arm to fall back to. A file that disclaims its own throughput
number is nonetheless the file every render executes.

**The convolution WAS a seven-deep scalar loop nest accumulating in `double`.**
At `332aed738`, `ltx2_video_vae.cpp:161-185`: the output loops at `:161-164`, the
accumulator declared `double acc` at **`:165`**, and the multiply-accumulate at
**`:170-176`** casting **both** operands with `static_cast<double>` before the
FMA, stored back through `static_cast<float>` at `:181`.

It was not confined to the conv. `double acc` occurred at **8 sites** in that one
file (`:165`, `:201`, `:303`, `:312`, `:546`, `:570`, `:579`, `:916`), and
`static_cast<double>` on **29 lines / 30 occurrences** — `:287` carries two, so
a line count and an occurrence count differ here and this row states both. The
1x1x1 convolution `Linear3d` — a plain GEMM, `[out_ch, in_ch] x [in_ch, N]` —
was a scalar f64 loop at `:190-209`, accumulator at `:201`. The attention block
was three more (`:546`, `:570`, `:579`).

No oracle does this. Lightricks computes in bf16 with no autocast and no
promotion (§2.1); diffusers the same, including inside its norms (§2.2); SGLang
sets bf16 explicitly over an fp32 base default (§2.3). **f64 accumulation on the
model path appears in no reference at all.**

**It has landed, and this row is written after it.** `LTX25-DECODE-DTYPE` took
the dtype half of lever 2 and merged as `d1b0ea3a8`
([PR #1036](https://github.com/mudler/vllm.cpp/pull/1036), *"Closes #1008"*),
which is this branch's base. At that commit
`grep -c "double acc" src/vllm/model_executor/models/ltx2_video_vae.cpp` returns
**0** (was 8) and `static_cast<double>` **6 lines / 7 occurrences** (was 29 / 30)
— the remainder being the annotated scalar-constant exceptions that spec names
one by one. Reading its
[`## Outcome`](ltx25-decode-dtype.md) rather than re-deriving it:

* **The numerics had to be defended, and the width alone was not enough.**
  Narrowing to f32 while keeping the naive serial summation order pushed
  `test_ltx2_tiling`'s non-causal untiled control to **5.00679e-06 against a
  5e-06 tolerance** — a genuine RED. The repair was the summation **order**:
  `CausalConv3d` now keeps one partial per input channel, which is what torch's
  blocked f32 convolution does. **No tolerance was widened.** The shipped arm
  sits at 1.07x-1.31x of the f64 arm's `max|diff|` on every video arm, worst case
  56% of tolerance.
* **What it did NOT take, and is still owed.** Storage stays f32, and
  **NDHWC / `channels_last_3d` was filed rather than half-built** — that spec's
  §5 gives the reason: a shared helper (`minimax_h3.h:756`
  `MiniMaxH3GroupNorm3d`) hard-codes NCDHW across three models, ten sites bypass
  `Volume::At()`, and on a host arm with no vectorization the layout buys nothing
  by itself. So **the memory-format half of lever 2 below is unchanged**, and it
  is owed to that row's own `## Owed` rather than re-filed here.
* **Nine narrowed sites still have no width gate**, `Linear3d` among them: it was
  widened back to `double` and 40/40 cases passed. That is recorded there as an
  honest gap.

**It routes through no shared op and no threadpool.** `ParallelForRows`
(`src/vt/cpu/cpu_threadpool.cpp:413`) is synchronous, and 10+ CPU kernels use it
— `cpu_conv2d.cpp:78`, `cpu_conv1d_depthwise.cpp:72`, `cpu_layernorm.cpp:53`,
`cpu_paged_attn.cpp:185`, `cpu_quant_gemm.cpp:191`, `cpu_attn_relpos.cpp:89`,
`cpu_ops.cpp:28` among them. **Zero of them are in the video VAE decode.**
Grepping `ParallelForRows|Threadpool|std::thread|omp|vt::` across
`ltx2_video_vae.cpp`, `ltx2_video_vae_tiled.cpp` and `ltx2_tiling.cpp` matches
only the substrings `complementary` and `compress`. That grep is the positive
control for its own negative: it returns >0 rows, and every row is a false
match, which is a different and stronger statement than a grep returning
nothing.

The CUDA table that does exist for LTX-2.5 —
`include/vllm/model_executor/models/ltx2_kernels.h` and
`src/vt/cuda/cuda_ltx2.cu` — is the **DiT** device-forward glue
(`vt::OpId::kLtx2`), seven ops covering AdaLN, modulate, add-gated, gate-heads,
RoPE, output-modulate and SiLU. It contains no convolution and nothing the VAE
decode reaches. **The video VAE decode has no device arm at all.**

**Memory format is NCDHW f32** (`Volume::At` at `ltx2_video_vae.cpp:73-75`
indexes `((c * t + ti) * h + hi) * w + wi`), against upstream's NDHWC bf16 fast
path (§2.1). That is 2x the bytes per element and a different kernel family, and
it is exactly the difference AGENTS.md says a token gate structurally cannot
report.

## 4. The 60 GiB — excluded from the decode, and NOT the residency either

### 4.1 The decode is excluded, twice, by two independent methods

1. *Measured.* Exact `operator new` accounting over a real 448x256/25f decode
   run to completion gives a heap peak of **361.72 MiB**
   ([`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 2) — a
   factor of **170** below 60 GiB.
2. *Computed, as an upper bound.* Summing **every** intermediate the LTX-2.5
   conv decoder ever produces at 448x256/25f and assuming **nothing is ever
   freed**, in f32:

   | stage | GiB |
   |---|---|
   | `res_x(4)` @ 512, 13x32x56 | 2.010 |
   | `res_x(6)` @ 256, 25x32x56 | 2.861 |
   | `res_x(4)` @ 128, 25x64x112 | 3.814 |
   | everything else | 0.964 |
   | **total** | **9.649** |

   The pathological ceiling for the entire conv decode with no frees at all, and
   still **6x short of 59 GiB**. The realistic upstream peak is ~95 MiB bf16 on
   the workspace path and ~200-300 MiB on the plain path, which makes our
   measured 361.72 MiB the right order for a correct f32 NCDHW port rather than
   evidence of a leak.

**So `memory_efficient_decode.py` cannot be the missing 59 GiB**, and this row
closes that hypothesis (#1011) rather than carrying it. The dispatch named it as
the obvious candidate; the arithmetic above is what a guess would have skipped.

### 4.2 The shape of the fall, which rules out the answer this row first reached

The richest record of it is **not** in either LTX spec. It is
`.agents/benchmark-record.md:21116-21129` at `332aed738`, and it is more specific
than the summaries that quote it. (An earlier draft cited `:21144-21160`, which
is the *next* subsection — the instrument caveat and the ladder conclusion, not
the trace.) At 448x256/25f, on the prompt-embeds path with no text tower on the
machine at all:

* both denoise phases finished and drained normally;
* `MemAvailable` was **flat at 75.2 GiB through all of it**;
* then, **after the last drain**:

```
03:47:11 avail_kB=73014000 rss_kB=4972520
03:47:21 avail_kB=57800944 rss_kB=4972520
03:47:31 avail_kB=27711644 rss_kB=4899272
03:47:35 WATCHDOG_KILL avail_kB=13774472 floor=18000000
```

**~59 GB in 24 seconds with the process's own RSS flat at 4.9 GB.** The fall is
on the decode side of the denoise-to-decode boundary — which is the one part of
`docs/USAGE.md:868` that *is* supported — but no frame was ever written, and
`Ltx2ConvVideoDecode` at that size cannot allocate it (§4.1).

**A plateau followed by a cliff refutes the hypothesis this row reached first.**
The render's documented residency is large and real — `docs/USAGE.md:862-864`
gives ~44 GB for the staged transformer and ~24 GB for the text tower — and it
is tempting to sum the resident objects and land on "about 60 GiB". **That sum
is a load-time total and cannot explain a fall that starts after a flat
plateau.** The residency was already paid for at 75.2 GiB. This row wrote the
residency hypothesis down before reading `benchmark-record.md`, and records the
correction here rather than quietly deleting it, because the sum is exactly the
attractive wrong answer the next reader will also reach.

The 320x192/49f trace of §1.4 is a **different shape and not a counter-example**:
it ran *with* `--encoder`, so its fall is during load and it then plateaus and
completes. Only the plateau-then-cliff shape fails.

### 4.3 The mechanism that fits, and why nobody has seen it

**`Backend::Alloc` on CUDA is a raw `cudaMalloc`** —
`src/vt/cuda/cuda_backend.cu:77-81`. On a GB10's unified pool that consumes the
same bytes as host RAM and **does not appear in `VmRSS`** the way a
`std::vector` does. So *"flat process RSS at 4.9 GB while MemAvailable fell
60 GiB"* is not evidence that nothing allocated. It is the **signature** of a
device-class allocation on this box, and it is why every RSS sampler pointed at
this has come back empty.

The instrument that would have caught it does not exist here, and
`benchmark-record.md:21146-21150` says so: **`nvidia-smi --query-gpu=memory.used`
returns `[N/A]` on GB10.** There is no per-process device-memory reading, so
device usage has only ever been observable as `MemAvailable` — *"which moves for
anything on the machine"*.

**Two things follow, and the second is uncomfortable.** First, the fall has never
been attributed to *our process* at all; a system-wide counter on a shared box
cannot do that, and the same record notes the box was saturated with other
coordinators' `ctest` work during that session. Second,
`--query-compute-apps=used_memory` **does** return a real figure on this box —
verified this row at 50419 MiB for an unrelated `llama-imatrix` — so a
per-process device reading was available the whole time and was not used.

**This is no longer a hypothesis about the mechanism.** Rung 1 (§1.3) measured
it directly: over the DiT staging phase the process's `Anonymous` grew by
**0.01 GiB against a 43.50 GiB `MemAvailable` fall — 0.0% of it** — while the
device counter grew 30.98 GiB and file-backed RSS grew 10.83 GiB, together 96%
of the fall. An RSS- or `Anonymous`-based sampler cannot see any of it. What
remains open is not *how* bytes can vanish from `MemAvailable` without touching
the heap, but *which* allocation does it in the post-denoise window §4.2
describes.

### 4.4 Next hypotheses, ranked, and the instrument that separates them

Stated as hypotheses. This row does not close the axis, and per AGENTS.md the
448x256 result is a measurement and not a ceiling.

1. **A device/unified allocation between the last drain and the first frame.**
   Fits the RSS-flat signature exactly (§4.3). Against it: the decode is host
   C++ that touches no device, and the pool drain runs *before* the fall and
   returns only 0.14 GiB (`benchmark-record.md:21111`).
2. **Another process on the box.** The counter is system-wide, the box is
   shared, and this has already voided one figure in this campaign — the same
   spec's own probe lost 11.1 GiB of its `MemAvailable` delta to a concurrent
   build ([`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 2).
3. **File-backed pages faulted and never released.** §4.5 defect 2 is real and
   unfixed; whether `MemAvailable` (which discounts reclaimable pages) can fall
   this far from it is exactly what the `Anonymous`-vs-`Rss_File` split decides.

**The instrument, which is queued (§1.3) and is the one all three of the records
above lacked:** per-PID at 2 s, `smaps_rollup`'s `Anonymous` separated from
`VmRSS` (which conflates anonymous with file-backed),
`--query-compute-apps=used_memory` for the device side, `MemAvailable` for the
pool, and the written frame count — all on **one clock**, under the GPU lock, on
a box whose other load is recorded rather than assumed.

### 4.5 Two residency defects found while doing this, real regardless of the 59 GiB

Neither is the fall. Both are genuine and are filed.

* **`Ltx2WidenDitToF32` holds the bf16 originals and the f32 copies at the same
  time, permanently.** `src/vllm/model_executor/models/ltx2_loader.cpp:694-710`:
  it allocates a widened buffer, repoints `view.data` at it, and **appends** it
  with `checkpoint.storage.push_back(std::move(widened))` at `:707`. Nothing
  drops the original. At the **21.00B** geometry measured from the checkpoint
  header ([`ltx-2.5.md`](ltx-2-5.md):55; `ltx2_video.h:109` at `332aed738`, the
  line this defect is cited against, states the same figure) a fully-bf16 DiT is
  2 + 4 bytes per parameter — **~117.3 GiB COMPUTED on a 119 GiB box**, of which
  the ~39.1 GiB of bf16 originals is dead the instant widening finishes. Reached
  on every host-arm load via `ltx2_video.cpp:786`
  (`widen_to_f32 = !on_device`).

  **Two corrections to that figure, stated rather than buried.** An earlier draft
  used **18.95B**, which is a measured parameter count nowhere in this tree: it
  is `ltx2_video.h:109`'s own *"~76 GB"* f32 figure divided by 4 bytes. That
  header line states `21.00B` and `~76 GB` in the same sentence and the two do
  not reconcile — 21.00B in f32 is ~84 GB — which is a small record defect of its
  own, unowned and noted here rather than fixed. Second, the loop widens only
  views whose dtype is `kBF16` (`ltx2_loader.cpp:697`), so on an FP8 checkpoint
  it touches only the bf16 biases and norms. **117.3 GiB is the bf16-arm bound**,
  not a cost every host-arm load pays, and no host-arm load has been measured
  against it (§7's `## Owed` carries the peak-RSS measurement).
* **The LTX loaders never release their mmap source pages.** `grep -c
  MaybeReleaseSourcePages` returns **0** for both `ltx2_loader.cpp` and
  `ltx2_video.cpp`. *Positive control:* the same symbol appears in **15** other
  files under `src/vllm` (`muse_glimmer_weights.cpp`, `phi_weights.cpp`,
  `gemma4_weights.cpp`, `nemotron_h_weights.cpp`, `kimi_linear_weights.cpp`,
  `qwen3_5_dense_weights.cpp`, `olmo2_weights.cpp` among them) — the symbol and
  the path set are right, and the calls genuinely are not there. The whole DiT
  file stays faulted resident alongside the growing device copy.

## 5. Why the decode is single-threaded and on the host

Three separate answers, because they are three separate defects and only the
first is about threads.

**Could it use `ParallelForRows`? Yes, trivially.** The op is embarrassingly
parallel over `oc` and over `ti`: `ltx2_video_vae.cpp:161-164` is a perfectly
nested loop whose body writes one output element and reads only `padded` and
`weight`. `ParallelForRows(CurrentThreadpool(), out_channels, ...)` is the same
shape `cpu_conv2d.cpp:78` already uses for 2-D convolution in this tree. Nothing
structural prevents it.

**Does it? No, and nothing in the file was ever wired to a threadpool** (§3,
with the false-match positive control). This is not a tuning gap. The decode was
written as a scalar reference and shipped as production.

**Is there a device path that is simply not wired? No — there is no device path
to wire.** This is the sharper finding, and it is why "add threads" is the wrong
instruction. `vt::OpId::kLtx2` covers the DiT only (§3). There is no `vt::Conv3d`
op reaching this decoder, no CUDA kernel for it, and no CPU `vt::` op either.
Wiring is not the missing step; the arm does not exist.

**How much of the render this accounts for.** ~89% of a completed 320x192/49f
render's wall is a flat-memory phase at load ~1.1 on 20 cores (§1.4). Whether
that phase is host compute or GPU-blocking is what rung 2 decides.

**Rung 2's answer is broader than the question, and it is measured.** Across
**347 per-PID samples** of the 320x192/25f render — stated as a sample count,
because §1.3 shows the sampler's wall coverage does not close and no drop rate
was recorded:

* GPU utilization **never exceeded 2%**, and was **exactly 0 in 321 of 347
  samples**;
* every one of the 26 non-zero samples falls **inside the DiT staging window**
  (t <= 251 s), where it is incidental copy traffic;
* therefore **every sample after t = 251 s reads 0**, with `capp_mib` flat at
  36396 MiB throughout;
* across that same stretch the process holds **exactly 1.00 core** of 20 in the
  two `utime` windows this figure covers — `+116.7 s over 117 s` and
  `+142.2 s over 142 s`, **259 s of measurement in total**; §1.3's table records
  a third window in the same regime (`+648.9 s over 649 s`), so 259 s is what
  these two cover and not the sampler's total;
* no frame had been written.

**The two halves of that are not equally strong, and an earlier draft did not
separate them.** The **GPU-zero half is over every sample** — it is a property of
the whole 347-sample set and needs no interpolation. The **core-count half rests
on 259 s of `utime`**, not on the whole window. Earlier drafts attached a minute
figure to this — *"over 15 minutes and counting"* here, *"17+ minutes"* in
[#1024](https://github.com/mudler/vllm.cpp/issues/1024), *"the first 1192 s"*
above — and **no consistent reading satisfies all three**: 347 samples at a 2 s
cadence span 694 s, so neither 1192 s nor a 17-minute sub-window past t=251 s
fits inside them. Rather than pick one, this row states the sample count and the
zero fraction, which is what the record supports, and states no minute figure at
all. The raw CSV resolves it and is on the unreachable box
([#1040](https://github.com/mudler/vllm.cpp/issues/1040)).

So this is not merely a host VAE decode at 0% GPU. **Nothing after the weight
staging has used the GPU for compute at all**, on a run that staged 35.54 GiB of
transformer weights onto that GPU and was invoked with `--device cuda`. The
weights are resident on a device that then does no work.

**The instrument caveat, stated before the conclusion leans on it.** This row has
**no positive control that `utilization.gpu` reports a high value for a real
compute kernel on GB10.** It is demonstrably live — it moved between 0% and 3%
during staging — but "live" is not "correctly scaled", and this box already has
form here: `nvidia-smi --query-gpu=memory.used` returns `[N/A]` on it (§4.3).
A 0% reading from an uncalibrated counter is exactly the shape of a broken
instrument producing a verdict about the code. **That control is owed**, and it
is one command under the lock: run any known GPU-saturating job and confirm the
counter reads high.

**The CPU column does not depend on that counter, and it carries the claim on its
own.** The process holds **exactly 1.00 core of 20** across 259 s of sampled
`utime` with `capp_mib` flat. A host thread that had handed the work to the device would not
accumulate a full core of time *and* leave the device counter unchanged *and*
produce no output; the three together are only consistent with the host doing
the arithmetic. The GPU-utilization column corroborates that reading rather than
establishing it.

**What that does and does not establish.** It establishes, for this
configuration and this build (`0e1bee42f`), that **no sampled moment after the
weight staging used the GPU**, and that in the 259 s of `utime` measured across
that stretch the process ran exactly one thread. Read together with §1.3's table
that is a render whose post-load wall is single-threaded host execution; the
"end to end" reading is an interpolation over the unsampled gaps, and §1.3 says
which seconds are measured and which are not. It does **not** by itself identify
which phase the sampled window is in — the run had not reached a frame, and the
engine emits nothing that timestamps a phase boundary (#1010), which is exactly
why that issue is ranked as the precondition it is. Attributing the window
between the connector and the first frame needs either #1010's instrumentation
or a completed run whose frame mtimes bound the decode. Rung 2's completion is
recorded in §1.3 when it returns.

This is filed as [#1024](https://github.com/mudler/vllm.cpp/issues/1024),
including the owed positive control.

**One thing follows immediately and does not wait for that.** `docs/USAGE.md`'s
*"most of a 320x192/25f render is spent single-threaded in the host VAE decode
at 0% GPU"* understates the problem. The 0% GPU is not a property of the decode
phase. It is a property of the whole render after load.

**Why 2681 s is not mysterious once the arithmetic is done.** One 448x256/25f
decode is **~7.25 TFLOP** of dense 3x3x3 convolution across 42 conv calls
(§1.1). 7.25 TFLOP in 2681 s is ~2.7 GFLOP/s sustained — an entirely ordinary
figure for a naive direct 3-D convolution on one core, and one made worse at the
time of that measurement by f64 operands, which halve the achievable NEON lane
width against f32 before any blocking or SIMD is considered. **The decode is not
slow because of an algorithmic difference from upstream. It is slow because 7.25
TFLOP is being executed one scalar FMA at a time on one of twenty cores.**

Two caveats on that ratio, and neither dissolves it. The denominator is a
**single sample on a contended box** (§1.1). And the f64 half of the numerator's
explanation has since been removed — `LTX25-DECODE-DTYPE` landed as `d1b0ea3a8`
(§3) — so **2681.02 s no longer describes the shipped decode**, and its speed
magnitude was booked by that row as owed and **unmeasured**, because
`dgx.casa` was unreachable for its whole duration. Nobody has timed the f32
decode. The scalar-on-one-core term is untouched and is what levers 1 and 3
address.

## 6. The ranked levers

Ranked by expected magnitude over effort. Every estimate names its reasoning and
says plainly when it is speculative. **No estimate here is a measured speedup**,
because measuring one requires an arm that does not exist yet; they are
magnitude arguments from arithmetic and from what the oracles run.

| # | Lever | Expected magnitude | Reasoning | Upstream anchor | Size | What would prove it |
|---|---|---|---|---|---|---|
| 0 | [#1024](https://github.com/mudler/vllm.cpp/issues/1024) **Find out why the GPU is idle for the whole post-load render.** 35.54 GiB staged onto it; 0% utilization in 321/347 samples, all 26 non-zero ones inside staging. | Bounds every other lever: if the denoise is also on the host, #1007 alone does not close the render | MEASURED rung 2 (§5); disjunction stated, not guessed | `ltx2_video.cpp:2946` vs `:2948`; `:786` | **small to diagnose** | #1010's phase timings, plus the owed `utilization.gpu` positive control |
| 1 | [#1007](https://github.com/mudler/vllm.cpp/issues/1007) **Give the video VAE decode a device arm.** It has none; production runs the CPU reference. | The dominant term. 7.25 TFLOP that upstream runs on an accelerator in a fraction of a second | §5 arithmetic; every oracle is GPU-resident | `blocks.py:1139` + `single_gpu_model_builder.py:273`; `decoding_av.py:71`; `interface.py:92` | **large** — a new `vt::` conv3d op plus CUDA/CPU arms, mirroring `cuda_ltx2.cu`'s seam | end-to-end wall at 320x192/25f, same seed, same frames, byte-compared pixels against the f32 reference |
| 2 | [#1008](https://github.com/mudler/vllm.cpp/issues/1008) **Drop f64 accumulation to the checkpoint dtype, and NCDHW to NDHWC.** Was 8 `double acc` sites, 29 lines / 30 occurrences of `static_cast<double>`, f32 buffers. | Was ranked large on the host arm; on a device arm the layout decides which cuDNN family runs | f64 appears in no oracle; NDHWC is upstream's default-on fast path | `conv_video_decoder.py:282-284`; `normalization.py:32-40`; `memory_efficient_decode.py:617-627`, `:655-656` | **dtype half LANDED** as `LTX25-DECODE-DTYPE` / `d1b0ea3a8` (PR [#1036](https://github.com/mudler/vllm.cpp/pull/1036)); memory-format half filed and owed to that row's `## Owed` | **the dtype half shipped its own instrument** — a separable-reduction width case entering through `Ltx2VideoDecodeStreaming` (`tests/vllm/models/test_ltx2_vae.cpp`), red before and green after. Its **speed magnitude is UNMEASURED**: `dgx.casa` was unreachable for that row's whole duration |
| 3 | [#1009](https://github.com/mudler/vllm.cpp/issues/1009) **Route the decode through `ParallelForRows`.** Exists, synchronous, used by 10+ CPU kernels, unused here. | Bounded by core count; 20 on GB10 | §5; `cpu_conv2d.cpp:78` is the same shape | none — this is a local seam, not an upstream mirror | **small** | wall at fixed thread counts, plus `max\|diff\| == 0` against the serial arm |
| 4 | [#1011](https://github.com/mudler/vllm.cpp/issues/1011) **Port `memory_efficient_decode.py`.** Workspace reuse, in-place norm/SiLU, free-before-conv, temporal conv chunking, NDHWC. On by DEFAULT upstream. | Byte traffic, not the 60 GiB — §4 closes that | `blocks.py:1059` default True; the four optimizations at `:1-20` | `memory_efficient_decode.py:1-20`, `:91-105`, `:122-204`, `:234-248`, `:541-609`, `:617-627` | **medium** — it is a second independent rewrite of `CausalConv3d` | peak-heap counter at a fixed size, against the current 361.72 MiB |
| 5 | [#1014](https://github.com/mudler/vllm.cpp/issues/1014) **Attribute the 59 GiB, then release what holds it.** Decode excluded twice (§4.1); residency excluded by the plateau (§4.2); mechanism candidate is a device-class allocation invisible to `VmRSS` (§4.3). | Decides whether 448x256/25f completes at all | `benchmark-record.md:21116-21129`; `src/vt/cuda/cuda_backend.cu:77-81` | upstream offloads transformer weights (`installation.md:90`) and never the VAE | **medium** | rung 1 of §1.3. **The lever is not yet known** — §4.4 names three hypotheses and the one instrument that separates them |
| 5a | [#1015](https://github.com/mudler/vllm.cpp/issues/1015) **`Ltx2WidenDitToF32` holds bf16 and f32 at once, permanently** — a bf16-arm bound of ~117.3 GiB on a 119 GiB box at the measured 21.00B geometry, ~39.1 GiB of it dead. | Not the 59 GiB; a real residency defect on the host arm | `ltx2_loader.cpp:694-710`, append at `:707` | — | **small** | peak RSS across a host-arm load |
| 5b | [#1016](https://github.com/mudler/vllm.cpp/issues/1016) **The LTX loaders never call `MaybeReleaseSourcePages`** — 0 hits against 15 other files under `src/vllm`. | Not the 59 GiB; the DiT file stays faulted resident beside its device copy | `ltx2_loader.cpp:738-756` | — | **small** | `Rss_File` across a load |
| 5c | [#1021](https://github.com/mudler/vllm.cpp/issues/1021) **Overlap and batch the DiT device staging.** ~3,504 serial `Alloc`+copy+`Synchronize` round trips. | 7.5 min off the front of every render and every gate run | MEASURED rung 1: 450 s for 31723 MiB = 70.5 MiB/s, GPU idle in 164/192 samples, 0.15 cores. Rung 2 stages the same ~32 GiB in 251 s = 127.8 MiB/s on the same host and boot — a **1.81x spread**, unattributed (§1.3) | `ltx2_loader.cpp:738-756`, `:747`, `:749`; `cuda_backend.cu:77-81` | **medium** | staging wall and `capp_mib` slope, plus a byte-compare of the staged weights |
| 6 | [#1010](https://github.com/mudler/vllm.cpp/issues/1010) **Emit phase timings and peak memory from the render path.** A 2.5-hour render wrote **one** line to `run.log`. | No speedup. It is the precondition for measuring any of 1-5 | — | — | **small** | its own output |
| 7 | [#655](https://github.com/mudler/vllm.cpp/issues/655) + [#1012](https://github.com/mudler/vllm.cpp/issues/1012) **Register `ltx_core` as an oracle and install it on the gate host.** | No speedup. It is the precondition for any *ratio* (§7) | AGENTS.md oracle table; issue #655 | — | **small-to-medium** | a recorded pin plus a gateability measurement |

**Honest about lever 1's size.** It is ranked first on magnitude and is by far
the largest change. Levers [#1008](https://github.com/mudler/vllm.cpp/issues/1008) and [#1009](https://github.com/mudler/vllm.cpp/issues/1009) are cheap, compose with each other, and pay
on the host arm that exists today, so a sensible order is 6, 2, 3, then 1, with
5 gated on the probe. That is a recommendation, not a finding, and the row that
takes lever 1 should re-derive the order against whatever the probe returns.

**Lever 2's dtype half has since been taken, and the order stands.** It landed as
`LTX25-DECODE-DTYPE` / `d1b0ea3a8` **without** the phase timings lever 6 owes —
so its speed magnitude is unmeasured, which is exactly the outcome lever 6 exists
to prevent and is a small vindication of ranking 6 first. The remaining order is
6, 3, then 1, with lever 2's memory-format half owed to that row and correctly
placed after or with the arm that consumes it.

**Speculative, and labelled so.** The *magnitude* of lever 3 is arithmetic
(core count) and is as solid as arithmetic gets. Lever 2's dtype magnitude was
argued the same way (dtype width) and **has still not been measured**, which is a
standing reminder that an arithmetic magnitude argument is not a measurement. The
*composition* of 2 with 3 is more speculative still: a threaded f32 SIMD arm may
become memory-bound where the scalar arm was ALU-bound, and the combined figure
could fall well short of the product. Nothing here predicts it, and the row that
implements them must measure the composition rather than multiply the parts.

**No ceiling is declared anywhere in this table.** Where a magnitude is unknown
it says so.

## 7. The denominator

**Today there is none, and this row will not manufacture one.**

Ruled out, each for a reason established in §2:

* **vLLM** implements no LTX and no diffusion VAE decode (§2.5). Under AGENTS.md
  this is the *"implements nothing"* branch, not a failure.
* **vLLM-Omni** stops at 2.3 (§2.4), and its only route to a 2.5 checkpoint is
  `DiffusersAdapterPipeline`, which declares `supports_step_execution = False`
  (`:69`) and `supports_request_batch = False` (`:68`), is forced to
  `max_num_seqs <= 1`, and refuses CFG parallel, sequence parallel and caching.
  That is a reference-degraded configuration and is disqualified by the same rule
  that forbids `--enforce-eager`. §2.4 records that the step-execution refusal is
  a call-time `NotImplementedError` rather than a configuration gate, which
  changes the mechanism and not the verdict. Worse for a *correctness* gate, its
  native path would silently mislabel a 2.5 checkpoint as LTX-2
  (`ltx2_components.py:168-169`).
* **SGLang** implements 2.0 and 2.3 but not 2.5 (§2.3, 0 hits against a 78-hit
  identical-pattern control).
* **SGLang-Omni** implements nothing in this class (§2.6).

That leaves two candidates, and the choice between them is a real decision this
row does not have the authority to make alone:

* **Lightricks `ltx_core`** is the reference implementation and *is already the
  oracle every LTX-2.5 gate in this repository executes against*. It is in **no**
  `.agents/oracles/` file and **not** in the AGENTS.md table. That is issue
  **#655**, and it means a stack of existing LTX-2.5 correctness gates run
  against an oracle the policy does not admit — a protocol violation no checker
  can see, because no checker knows the oracle exists.
* **`diffusers`** is **already in the AGENTS.md table, already has
  `.agents/oracles/diffusers.md` with a pin, and — §2.2 — actually implements
  LTX-2.5, both decode arms.** No record in this tree says "no admitted oracle
  carries 2.5" in those words, and this row does not claim one does. What the
  records say is narrower and adds up to the same working assumption:
  `roadmap_v1.md:92` at `332aed738` scopes its negative to one oracle —
  *"vLLM-Omni does NOT yet carry 2.5 (recipes stop at 2.3 …), so the binding
  oracle is vLLM-Omni's GENERIC diffusers adapter and the immediate cross-check
  is Lightricks `ltx-pipelines`"* — so that line **does** name `diffusers`,
  twice, but both times as vLLM-Omni's *adapter* rather than as a repo at its own
  pin (the second: *"the diffusers adapter is a black box
  (`supports_step_execution=False`)"*), and the cross-check it names, Lightricks
  `ltx-pipelines`, is **not** admitted;
  `docs/BENCHMARKS.md:465` leaves the LTX-2.5 speed axis *and* the binding oracle
  both `PENDING`, and does not name `diffusers` at all. **No record treats the
  `diffusers` repo at its own pin as the LTX-2.5 oracle**, and the campaign has
  been proceeding as though no admitted oracle covered 2.5. That is the record
  defect [#1012](https://github.com/mudler/vllm.cpp/issues/1012) files: not a
  false sentence to strike, but an admitted oracle that no LTX-2.5 record has
  ever reached for.

**Does #655 block measurement here? Split the question, because the answer
differs by axis.**

* For a **throughput ratio** — yes, it blocks, and it is not the only blocker.
  A ratio needs a reference running the same workload in a production
  configuration. Beyond registration, `ltx_core` **is not installed on the gate
  host at all**: `dgx.casa` carries one venv,
  `~/venvs/vllm-oracle-pin-555967922`, and a search for `ltx_core` under `$HOME`
  returns nothing. A same-tool both-sides profile of LTX-2.5 is therefore not
  possible today on any oracle. That is a hard, citable blocker on every ratio
  this campaign has left `PENDING`.
* For **levers 1-4** — no, it does not block, and waiting on it would be an
  error. Those levers are justified by *upstream source* (every reference runs
  this decode on an accelerator, in checkpoint dtype, with no f64 anywhere) and
  by *our own* one-sided phase attribution. An 89%-of-wall single-threaded host
  phase is a defect against our own engine's structure; it does not need a
  denominator to be worth removing. AGENTS.md requires the denominator before a
  **parity claim**, and this row makes none.

**The recommendation, offered as a decision to be taken and not as a finding:**
register `ltx_core` per #655 *and* record that `diffusers` covers LTX-2.5, then
measure gateability of each on `dgx.casa` before any ratio is quoted. `diffusers`
is the lower-friction path — already admitted, already pinned — and `ltx_core`
is the higher-fidelity one and the one the existing correctness gates already
depend on. They are not alternatives; #655 has to be closed either way, because
the gates that already ran against `ltx_core` do not become admissible by
choosing a different oracle for a future one.

## 8. Gates, scope and stop conditions

**Scope.** This row ships a spec and issues. It writes no product code, changes
no checker, and makes no lifecycle transition, so it owes no `docs/STATUS.md`,
`docs/BENCHMARKS.md` or `docs/FEATURES.md` edit under AGENTS.md's projection
table. The `docs/USAGE.md:868` correction that §1.2 identifies — the *"inside the
decode"* half of that sentence — is **owed and deliberately not taken here**: it
belongs with the row whose probe settles §4, so the doc changes once, to
something measured, rather than twice. The separate `:873-874` sentence
(*"0% GPU"*, understated by §5) is owed to
[#1024](https://github.com/mudler/vllm.cpp/issues/1024). They are two edits to
two sentences and this row keeps them apart.

**Why the levers are separate rows.** Each of 1-5 needs a red-first test, an
independent fresh review, and its own upstream-anchored spec. Bundling them
would put a new CUDA op, a dtype change no golden can see, a threading change,
and a memory-lifetime change behind one review — and the second of those is
precisely the class AGENTS.md says a correctness gate cannot report on.

**Gate for any lever that follows.** Correctness first: pixels byte-compared
against the current f32 reference decode at a fixed seed and size, `max|diff|`
recorded, before any wall-clock number is accepted. A decode that is faster and
different is not a win, and the goldens as they stand **cannot** catch a dtype
that is merely too wide — the generator casts every upstream parameter to f32
(`gen-ltx2-vae-goldens.py:223`), so the oracle itself runs f32 and the comparison
is vacuous by construction (`ltx2_video_vae.cpp:41-44`).

**Lever 2 did ship its own instrument, and what it cost is worth carrying into
the levers that follow.** `LTX25-DECODE-DTYPE` added a separable-reduction width
case entering through the production `Ltx2VideoDecodeStreaming` — 27 taps of
`[+1e8, 0.1 x 25, -1e8]`, where an f32 accumulator lands on exactly 0 in any
summation order and an f64 one on 2.5 — red before and green after, and it caught
its own width mutation while three of the other narrowed sites stayed green under
theirs. Two things it learned belong to **every** lever here, not just that one:
its **first draft of that case expected zero**, and a reachability mutation that
replaced the decode with a zero-filled buffer **passed**, because a decode that
never ran produces zeros too; the repair was a non-zero bias so the reached, the
unreached and the wrong arms are three distinguishable values. And its first
reachability mutation **failed to build** while a stale binary printed a
plausible verdict. Any lever below that mutates must report `git diff --stat`,
whether it built, and the exit code — and must not let an expectation coincide
with the zero value of an absent computation.

**Stop conditions.**

* Stop and report `NEEDS_DECISION` if the probe shows the 59 GiB is **not**
  model residency. Lever 5 dissolves and §4's next hypothesis takes over.
* Stop and report if the GPU lock never frees. The source half of this row
  stands on its own and is marked as such; an unmeasured axis that says so beats
  a guess.
* Do not implement any lever from this row. It has no implementation authority
  and no fresh review.
* Do not quote a ratio against any oracle until #655 is closed and gateability
  is measured on the gate host.

## Owed

Every issue this row filed is owned here, **except one, which has since been
fixed by its own row and is therefore not owed** — see the note under the table.
The rest are not fixed in this flow, because this row has no implementation
authority and no fresh review (§8).

| Issue | Lever | State |
|---|---|---|
| [#1006](https://github.com/mudler/vllm.cpp/issues/1006) | this row: the investigation and this spec | closed by this row landing |
| [#1007](https://github.com/mudler/vllm.cpp/issues/1007) | 1 — the video VAE decode has no device arm | owed |
| [#1009](https://github.com/mudler/vllm.cpp/issues/1009) | 3 — `ParallelForRows` unused by the decode | owed |
| [#1010](https://github.com/mudler/vllm.cpp/issues/1010) | 6 — one log line per 2.5-hour render | owed, and it should land first |
| [#1011](https://github.com/mudler/vllm.cpp/issues/1011) | 4 — `memory_efficient_decode.py`, re-ranked | owed |
| [#1012](https://github.com/mudler/vllm.cpp/issues/1012) | record: `diffusers` implements LTX-2.5 and no LTX-2.5 record considers it | owed |
| [#1014](https://github.com/mudler/vllm.cpp/issues/1014) | 5 — the 59 GiB: decode excluded, residency excluded, three hypotheses ranked | owed |
| [#1015](https://github.com/mudler/vllm.cpp/issues/1015) | 5a — `Ltx2WidenDitToF32` holds bf16 and f32 at once | owed |
| [#1016](https://github.com/mudler/vllm.cpp/issues/1016) | 5b — LTX loaders never release mmap source pages | owed, MEASURED at +10.83 GiB in rung 1 |
| [#1021](https://github.com/mudler/vllm.cpp/issues/1021) | 5c — DiT staging is 7.5 min at 70.5 MiB/s, against 127.8 on rung 2; GPU idle, 0.15 cores | owed |
| [#1024](https://github.com/mudler/vllm.cpp/issues/1024) | 0 — the GPU is idle for the WHOLE post-load render, not only the decode | owed; carries the owed `utilization.gpu` positive control |
| [#1040](https://github.com/mudler/vllm.cpp/issues/1040) | none — the evidence for rungs 1 and 2 and for §1.4 is on an unreachable host, and neither rung's sampler cadence closes | owed |
| [#1202](https://github.com/mudler/vllm.cpp/issues/1202) | 7 — `Ltx2FuseLoraIntoTensor` is a scalar single-threaded loop, ~0.53 GFLOP/s | owed, MEASURED on the full model: 2.3% of one pass in 10.4 min |
| [#1208](https://github.com/mudler/vllm.cpp/issues/1208) | 8 — the text tower's `Linear` is scalar, single-threaded and `double`-accumulating | owed, MEASURED at 1073 s of byte-identical RSS; also a dtype-polarity divergence from `F.linear` |
| [#1210](https://github.com/mudler/vllm.cpp/issues/1210) | 9 — the two-stage rebind fuses, un-fuses and re-fuses; the load-time pass is provably wasted | owed; supplies the number `ltx2_video.cpp:2849` records as UNMEASURED |

**[#1008](https://github.com/mudler/vllm.cpp/issues/1008) is NOT owed here. It
landed.** It was filed by this row as lever 2 and taken by `LTX25-DECODE-DTYPE`,
which merged as `d1b0ea3a8` (*"Closes #1008"*) — this branch's own base. It is
therefore listed in [`issue-index.md`](../issue-index.md) with
`LTX25-DECODE-DTYPE` as its owning row rather than with a dash, which is what
that column is for, and it is deliberately absent from this table. Listing a
closed issue as owed by a row that did not close it would misattribute the work
and would leave a fixed defect reading as open debt. The memory-format half of
lever 2 stays owed — to
[`ltx25-decode-dtype.md`](ltx25-decode-dtype.md)'s own `## Owed`, where §5
argues why it must land after or with the arm that consumes it — and this row
does not re-file it.

* **The 60 GiB attribution.** Carried forward from
  [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome`, now with the
  decode excluded by two independent methods (§4.1), the residency answer
  excluded by the plateau (§4.2), the RSS-flat signature explained (§4.3), and
  three ranked hypotheses with the one instrument that separates them (§4.4).
* **`docs/USAGE.md:868`'s "inside the decode".** Half supported: the fall is
  on the decode side of the boundary, and the decode does not allocate it
  (§1.2). Owed to the row that closes §4. The phrase is on `:868`, not on the
  `:873-874` sentence about 0% GPU, which is a different claim owed to #1024.
* **Whether the 59 GiB is even our process.** The counter is system-wide,
  `nvidia-smi --query-gpu=memory.used` returns `[N/A]` on GB10, and this
  campaign has already lost 11.1 GiB of one such delta to a concurrent build.
  A per-process device reading was available the whole time
  (`--query-compute-apps=used_memory`) and was never used (§4.3).
* **A throughput number for LTX-2.5 on any axis.** `docs/BENCHMARKS.md` carries
  one LTX line, under `## Open gaps`. It stays there, and §7 says why.
* **`memory_efficient_decode.py`.** Still unported; re-ranked by §4 as byte
  traffic and memory format rather than as the 60 GiB.
* **Gateability of `ltx_core` and of `diffusers` for LTX-2.5 on `dgx.casa`.**
  Neither is installed there today.
* **Rung 2's exit reason.** `EXIT=1`, 0 frames, 2407 s. `REMOTE_UNVERIFIED` —
  the host went unreachable before `run.log` was read. The file is on the box.
* **A positive control that `utilization.gpu` reads high for a real kernel on
  GB10.** One command under the lock; it gates how much weight §5's 0% carries.
  Filed with [#1024](https://github.com/mudler/vllm.cpp/issues/1024).
* **Naming the 1.000-core window.** The single largest block of rung 2, and the
  one this row was dispatched to explain. It needs
  [#1010](https://github.com/mudler/vllm.cpp/issues/1010)'s phase timestamps,
  not another probe.
* **Every measurement in §1.3, §1.4 and §5, as a retrievable artifact.** The
  sampler CSVs, rung 2's `run.log` and `render8-console.log` exist only on
  `dgx.casa`, which does not answer. `.agents/verification.md` requires the
  evidence path and there is none. Filed as
  [#1040](https://github.com/mudler/vllm.cpp/issues/1040), together with the
  sampler-cadence reconciliation that neither rung closes.
* **An idle-host repeat of the 2681.02 s decode wall.** It is one sample taken
  while a 405-target build and a 416-test `ctest -j 6` ran on the same box at
  load average 30.2 (§1.1), and it is the denominator of this row's headline
  2.7 GFLOP/s. AGENTS.md requires an idle-host same-binary A/B before a figure is
  accepted. It has not had one, and the arm it measured has since changed (§3).
* **The `~76 GB` / `21.00B` mismatch inside `ltx2_video.h:109`.** One sentence
  states both, and 21.00B in f32 is ~84 GB. Noticed while correcting §4.5;
  unowned, and small enough that the next row touching that header should take
  it.
