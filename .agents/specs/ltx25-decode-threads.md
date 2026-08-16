# LTX25-DECODE-THREADS — the decode runs on one core of twenty, and the seam it needs already exists

Row: `LTX25-DECODE-THREADS`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)).
Issue: [#1009](https://github.com/mudler/vllm.cpp/issues/1009).
Parent: lever 3 of the `LTX25-DECODE-SPEED` investigation
([#1006](https://github.com/mudler/vllm.cpp/issues/1006)), which filed this
issue and lists it under `## Owed`. That spec is
`.agents/specs/ltx25-decode-speed.md` on [PR
#1038](https://github.com/mudler/vllm.cpp/pull/1038), branch
`row/LTX25-DECODE-SPEED-R2`, and is **not yet on `main`**, so it is cited by pull
request rather than by relative link, exactly as the sibling dtype row
([`ltx25-decode-dtype.md`](ltx25-decode-dtype.md)) does. It was PR #1018 while
this row was implemented; that pull request is now **closed** and #1038
supersedes it, so every citation here names the open one.

Sibling, and the reason this row is riskier than it looks:
[`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) (#1008) landed at `d1b0ea3a8`
and changed the convolution's **summation order** to a blocked one. This row
adds parallelism on top of that, and parallelism is the second thing that can
change a summation order.

## Now

`DONE`, pending review. The three convolution sites dispatch, the numerics are
byte-identical to the serial arm, and the CPU A/B is in `## Outcome`: **~9x at 16
to 20 workers**, medians 9.15x and 9.14x, on a box that was not idle and where
those two counts spread 21-23% run to run. A second channel width corroborates at
9.67x on n=3. Read the band, not the decimals; §8.3 carries both with the load
they were taken at. No end-to-end render number, and none is claimed.

## 0. Scope

**In scope.** Route the LTX-2.5 conv video VAE's convolution loops through
`vt::cpu::ParallelForRows`, the synchronous row-chunked parallel-for that 10+
CPU kernels in this tree already use and that no line of the video VAE uses
today.

**Not in scope, and deliberately so.**

* The device arm ([#1007](https://github.com/mudler/vllm.cpp/issues/1007)).
  There is no `vt::` conv3d op on any backend; that is a much larger change and
  needs NDHWC first.
* NDHWC / memory format ([#1008](https://github.com/mudler/vllm.cpp/issues/1008)
  §5 records the verdict and the blocker, `MiniMaxH3GroupNorm3d`'s signature).
* `memory_efficient_decode.py` ([#1011](https://github.com/mudler/vllm.cpp/issues/1011)).
* SIMD. A vectorised inner tap loop is a separate change with a separate
  summation-order question, and mixing the two would make an order regression
  unattributable.
* The **audio** VAE and the video **encoder**'s non-convolution paths. The
  encoder shares `CausalConv3d`, so it inherits the change; nothing else in
  either file is touched.

**No end-to-end render number.** `dgx.casa` is unreachable and this box has no
GPU, so this row claims no render speedup and no ratio against any oracle. What
it can measure, and does, is a same-binary wall-clock A/B of the decode itself
at fixed thread counts on 20 local cores (§6).

## 1. Why there is no upstream to mirror here

Every oracle runs this decoder on an accelerator and none of them has a
host-parallel arm to port. The four anchors below are the parent investigation's
([`ltx25-decode-speed.md` on PR #1038](https://github.com/mudler/vllm.cpp/pull/1038) §2), read there at the
revisions named; none of these repositories is checked out on this box, so this
row cites them rather than re-deriving them. `ltx_core` has no
`.agents/oracles/` file at all — that pin is owed by
[#655](https://github.com/mudler/vllm.cpp/issues/655) and
[#1012](https://github.com/mudler/vllm.cpp/issues/1012), and the parent spec
lists both.

* Lightricks LTX-2 @ `fd4ded7f2` builds the decoder onto a device
  (`packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1139`,
  `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:267-288`);
  the whole decoder's convolution work is one `nn.Conv3d` call at
  `model/video_vae/convolution.py:312`.
* SGLang @ `f63458b5b` moves the latents to the local torch device
  (`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py:71`).
* vLLM-Omni @ `a4ea67a21` states the contract outright — *"VAE(s) (always on
  GPU)"*, `vllm_omni/diffusion/models/interface.py:92`.
* `diffusers` @ `3a2f35d4e` ships no CPU decode path at all
  (`ltx2_diffusion_decoder.py:208-209`, *"No CPU path"*).

So this row is a **local seam**, not an upstream mirror, and
[`ltx25-decode-speed.md` on PR #1038](https://github.com/mudler/vllm.cpp/pull/1038) §6 lever 3 records it as such.
What it does mirror is *this tree's own* CPU convolution:
`src/vt/cpu/cpu_conv2d.cpp:75-78 @ d1b0ea3a8` partitions a 2-D convolution over
`n * cout * hout` output lines through the same call, with the same comment this
row's change carries — *"independent outputs, so the partition can never change a
reduction"*.

## 2. The axis, and why it is reduction-safe

`CausalConv3d`'s output loop nest is `oc / ti / hi / wi`
(`src/vllm/model_executor/models/ltx2_video_vae.cpp:178-217 @ d1b0ea3a8`). The
`ci * kernel^3` reduction lives **entirely inside one `(oc, ti, hi, wi)` body**,
in the blocked order #1008 shipped: one `float tap` partial per input channel,
added into `float acc`.

**The parallel axis is the output line `(oc, ti, hi)`, and each unit is
`out.w` contiguous output elements.** `Volume::At(oc, ti, hi, wi)` is
`((oc*t + ti)*h + hi)*w + wi`, so output line `r` is exactly the contiguous span
`[r*out.w, (r+1)*out.w)` of `out.data`.

Three properties follow, and together they are the determinism argument:

1. **No output element is written by more than one worker.** The partition is a
   partition of `r`, and lines do not overlap.
2. **No reduction crosses a worker boundary.** Every accumulation — over `ic`,
   over `a`, `b`, `d` — is inside one `wi` iteration of one line. A worker
   executes exactly the instruction sequence the serial arm executes for that
   element, in the same order, on the same values.
3. **The result therefore does not depend on the worker count**, and it does not
   depend on which worker took which chunk either. That matters, because
   `ParallelForRows` (`src/vt/cpu/cpu_threadpool.cpp:413-458`) **steals work**
   through an atomic cursor, so the row-to-thread assignment is genuinely
   non-deterministic run to run. Bit-identity has to survive that, and it does,
   for reason 2.

This is not a new contract. `src/vt/cpu/cpu_threadpool.h:39-43` already states
it for the whole CPU backend: *"parallelism partitions OUTPUT elements only ...
No atomic accumulation into shared outputs, no reduction-order changes — results
are bit-identical to n_threads==1 by construction."* This row's job is to stay
inside that contract, not to invent one.

**What was rejected, and why.** Parallelising over the *reduction* axis `ic`
with per-thread partials and a final combine would also be a legal
convolution — and it would change the summation order as a function of the
thread count, which is exactly the defect #1008 spent its budget removing. It is
not taken, and no tolerance is widened anywhere in this row.

**`ParallelForRows` is synchronous**, so the `[&]` capture of the local `padded`,
`weight`, `out` and the loop bounds is safe: `Run` returns only after every
worker has passed the closing `Barrier()` inside `ComputeThread`
(`cpu_threadpool.cpp:234`), whose exit is a seq-cst fence (`:206-212`).

## 3. The sites

| site | what it is | parallel unit | rows |
|---|---|---|---|
| `ltx2_video_vae.cpp` `CausalConv3d`, output nest | 42 convs, ~all of the decode's FLOPs (`ltx25-decode-speed.md` §1.1) | one output line `(oc, ti, hi)` | `out_channels * out.t * out.h` |
| `ltx2_video_vae.cpp` `CausalConv3d`, pad gather | the replicate/reflect pad materialisation | one padded line `(c, ti, hi)` | `ci * pt * ph` |
| `ltx2_video_vae.cpp` `Linear3d` | the 1x1x1 conv used as `conv_shortcut` | a contiguous span of `(oc, i)` | `out_channels * in.spatial()` |

The pad gather has no reduction at all — it is a pure gather, one source element
per destination element — so it is trivially order-independent. It is included
because it is `O(ci * pt * ph * pw)` inside the same function and would otherwise
become a serial section that bounds the speedup by Amdahl's law.

Sites **not** taken, each for a stated reason:

* `PixelNorm`, `Silu`, `ApplyAdaLn`, `expand`, `drop_first_frame` — memory-bound
  elementwise passes. They are candidates, but they are not where the 7.25 TFLOP
  is, and each one added is another surface for a reviewer to check. Owed
  (§7) rather than done silently.
* `FeedSpatialNoise` — **must not** be parallelised. It consumes
  `Ltx2NoiseStream` in call order
  (`include/vllm/model_executor/models/ltx2_video_vae.h:201-210`), and that call
  order is the reproducibility contract with upstream's `torch.Generator`. The
  draw itself is already outside the loop; the loop that applies the plane could
  be partitioned, but the win is nil and the risk is a later edit moving the draw
  inside. Left alone deliberately.
* `AttnBlock3d` — the shipped decoder cannot construct an attention block at all:
  `attn_res_x` is refused by name (`ltx2_video_vae.cpp:10-14`), because upstream
  at the pinned revision cannot construct it either.

## 4. Risks

* **A partition that changes the summation order.** The one risk that can change
  the design, and the one that bound on the sibling row. Mitigation: §2's axis,
  plus the golden margins measured before and after and required to be
  **exactly equal**, not merely within tolerance. Any movement at all in a
  recorded `max|diff|` means the order moved and the design is wrong. No
  tolerance is widened; that is the stop condition (§8).
* **A result that depends on the thread count.** Mitigation: the determinism
  case in §5, which decodes the same input at five different worker counts and
  requires `memcmp == 0`.
* **A data race.** New concurrency in a file that had none. Mitigation: the
  ThreadSanitizer lane over the LTX suites (§6), because a race in a parallel
  reduction is precisely the defect this row could introduce and CI's sanitize
  lane is unreliable — it was cancelled in 4 of the last 12 `main` runs.
* **Nested dispatch.** `Threadpool::Run` throws on a dispatch from inside a
  parallel region (`cpu_threadpool.cpp:355`). The decode is called from
  `Ltx2VideoDecodeStreaming`, which is called from
  `src/vllm/multimodal/ltx2_video.cpp:3258` on the render path, and no caller in
  that chain is inside a parallel region. Checked by reading the chain, and the
  full gate would throw loudly if it were wrong.
* **A determinism test that measures nothing.** Two runs of a *serial*
  implementation are also bit-identical, so the determinism case alone is green
  before this row's change. That is why §5 ships a **second** case that observes
  the dispatch itself.

## 5. The gate

Two new cases in `tests/vllm/models/test_ltx2_vae.cpp`, both entering through
the production entry point `Ltx2VideoDecodeStreaming` — the one
`src/vllm/multimodal/ltx2_video.cpp:3258` calls on the render path, reaching
`Ltx2ConvVideoDecode` through `ltx2_video_vae_tiled.cpp:113`.

**Case A — the decode dispatches partitioned work to the CPU threadpool.** This
is the case that is RED before the change. A fresh `vt::cpu::Threadpool` is
installed with `SwapForTesting`, and its work-stealing cursor is read through
the public `ChunkAdd(0)`, which returns the current value and adds nothing. The
cursor is `0` on a fresh pool; `ParallelForRows` seeds it with `ChunkSet(nth)`
and every steal advances it (`cpu_threadpool.cpp:437-455`). So a non-zero cursor
after a decode is a direct observation that a multi-chunk partitioned dispatch
ran on that pool, and a zero cursor is the observation that none did. Before this
row the decode never touches a pool, so the case reads `0` and fails.

The same case asserts the decoded output against an analytically derived value
rather than a recorded one, so a decode that never ran cannot pass it. This is
the trap the sibling row hit and recorded: a zero-filled stub satisfies an
expectation of zero. The fixture therefore offsets `conv_out.conv.bias` off
zero, exactly as the width case does.

**Case B — the decode is bit-identical across thread counts.** The same latent
is decoded at worker counts 1, 2, 3, 5 and 8 and every result is `memcmp`-equal
to the 1-thread arm. Worker count 1 short-circuits `ParallelForRows` to
`body(0, nr)` on the caller (`cpu_threadpool.cpp:423-426`), so the 1-thread arm
*is* the pre-change code path, byte for byte. The counts are deliberately not
all powers of two: 3 and 5 do not divide the row counts, so the chunk boundaries
land in different places on every arm.

Case B also asserts that the decoded volume is not degenerate — that it holds
more than one distinct value — because an all-equal buffer, which is what a
stubbed decode returns, would satisfy a pure A-equals-B comparison.

**And the existing goldens become a threading gate for free.** The full suite
runs on the global pool, which is `hardware_concurrency` wide (20 here), so every
LTX-2.5 video golden already executes the threaded path. Their recorded
`max|diff|` values from #1008 are the before-picture, and §6 requires them to
come back **identical**.

## 6. Gates and evidence

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Reported: `CONFIGURE_EXIT`, `BUILD_EXIT`, the `: error:` count, `ctest -N`,
`CTEST_EXIT`, the full pass/fail line, `No space left` and `BFD assertion` with
positive controls, load average and free disk.

**ThreadSanitizer**, because this row adds concurrency:

```sh
cmake -S . -B build-san -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SANITIZE=thread
```

with the LTX suites run under it.

**The wall-clock A/B.** Same binary, one decode driven through
`Ltx2VideoDecodeStreaming` at a fixed synthetic decoder configuration, at
`VLLM_CPP_CPU_THREADS` 1, 2, 4, 8, 16 and 20, repeated enough times to show the
spread rather than one number, with the host load average recorded beside it.
The configuration and the harness source are recorded in `## Outcome` so the
measurement is reproducible; the shape is synthetic and is stated as such,
because the shipped checkpoint's `decoder_blocks` list comes out of a checkpoint
header this box does not have.

**What this row may not claim:** an end-to-end render speedup, any ratio against
any oracle, or a composition figure with #1008. There is no GPU here, no
large-render host, and no installed `ltx_core`.

## 7. Owed

| Item | Why it is not done here |
|---|---|
| The elementwise passes — `PixelNorm`, `Silu`, `ApplyAdaLn`, `expand`, `drop_first_frame` | Memory-bound and not where the FLOPs are. Each is order-independent and could be partitioned the same way; measuring whether it pays needs the A/B this row establishes first. |
| `AttnBlock3d` | Unreachable in the shipped decoder (`attn_res_x` is refused by name). Parallelising a path nothing can construct is dead code. |
| A SIMD inner tap loop | Separate summation-order question; see §0. |
| The composition of this row with #1008 | the parent spec §6 warns that a threaded arm may become memory-bound where the scalar arm was ALU-bound. This row measures its own axis only. |
| An end-to-end render number | `dgx.casa` unreachable; no GPU here. |
| A per-SITE dispatch gate — [#1044](https://github.com/mudler/vllm.cpp/issues/1044) | §8.6's T1/T2/T3 measured it: reverting any ONE of the three sites is detected by nothing, because Case A reads one cursor the whole pool shares. Correctness stays gated; what is ungated is a site silently going serial again. Closing it needs a per-dispatch `Threadpool::RunCount()` and an EXACT expected count, plus a `res_x_y` fixture for `Linear3d`. A new gate needs its own red-before evidence and its own review, so it is a row rather than an in-flow repair. |

No `.agents/issue-index.md` row is appended for #1009. That row already exists at
`.agents/issue-index.md:279` on PR
[#1038](https://github.com/mudler/vllm.cpp/pull/1038), branch
`row/LTX25-DECODE-SPEED-R2`, which is the open successor to the closed PR #1018
and is unmerged. `.gitattributes` sets `merge=union` on that file and
`scripts/check-agent-record.py` refuses a duplicate issue number, so appending a
second copy here would turn `main` red for every branch the moment #1038 merges —
which is exactly what a duplicate #995 row did on 2026-08-16. The sibling dtype
row made the same call for #1008 and recorded it in its pull request body.

**#1044 is different and IS appended.** It is a new issue this change filed, it
is not one of the ids #1038 appends (#1006-#1012, #1014-#1016, #1021, #1024,
#1040), and its index row names `LTX25-DECODE-THREADS` as the owning row. The
owner has to be named in the row rather than left to this section, because
`owed_issues()` in `scripts/check-agent-record.py` splits on a bare `\n## Owed`
and this spec's heading is numbered, so nothing listed here is visible to that
ratchet.

## 8. Outcome — what was measured

Everything below was measured on the shared 20-core development box on
2026-08-16, at **`d653f7319`**, the implementation commit on this branch. No GPU
was involved and none was available; `dgx.casa` was unreachable for this row's
whole duration.

**The commit the measurement ran on was `dac85969c`, and that SHA is deliberately
not the citation.** It was rewritten out of the branch and is not an ancestor of
the head (`git merge-base --is-ancestor dac85969c HEAD` exits **1**), so it does
not resolve in a fresh clone and citing it would name evidence nobody can reach.
The measurement transfers because the two commits are **byte-identical in
`src/` and `tests/`**: `git rev-parse dac85969c:src d653f7319:src` both give
`7444ffa171b0c2868c505b5b9ea1113fa39c5477` and both `:tests` give
`f0e5eac268119e9fe94da478c50e2d668a2e64b3`, and
`git diff --stat dac85969c d653f7319` touches only
`.agents/specs/ltx25-decode-threads.md`, `docs/BENCHMARKS.md`,
`docs/FEATURES.md` and `docs/USAGE.md`. Every number below came out of a binary
built from the code `d653f7319` carries.

### 8.1 The numerics did not move at all, and that is the point

The one risk that could have changed the design did not bind. Both suites were
built with `kLtx2GoldenTol` temporarily set to `0.0` so every golden reports its
`max|diff|` rather than its verdict, **before** the change and **again after**,
and both tolerances were restored.

**All 34 recorded margins — 23 in `test_ltx2_vae`, 11 in `test_ltx2_tiling` —
came back byte-for-byte identical.** The comparison was a `diff` of the two
sorted value lists, not an eyeball: `VAE_MARGINS_IDENTICAL (23 values)` and
`TILING_MARGINS_IDENTICAL (11 values)`. Nothing was within tolerance; nothing
moved.

| golden arm | before (serial) | after (20-thread global pool) | tol |
|---|---|---|---|
| Conv video decoder | 1.72853e-06 | 1.72853e-06 | 5e-06 |
| non-causal Conv video decoder | 2.08616e-06 | 2.08616e-06 | 5e-06 |
| norm_eps-binding video decoder | 1.54972e-06 | 1.54972e-06 | 5e-06 |
| tiled decode, untiled control A | 2.74181e-06 | 2.74181e-06 | 5e-06 |
| tiled decode, untiled control B | 2.80142e-06 | 2.80142e-06 | 5e-06 |
| video encoder (`*_res`) | 4.76837e-07 | 4.76837e-07 | 5e-06 |
| video encoder (strided convs) | 8.34465e-07 | 8.34465e-07 | 5e-06 |
| cropped video encoder | 4.76837e-07 | 4.76837e-07 | 5e-06 |
| causal-arm video encoder | 4.17233e-07 | 4.17233e-07 | 5e-06 |
| every other arm in both suites | unchanged | unchanged | — |

Those "before" values are also the ones
[`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) §8.1 recorded on its own host,
which is an independent check that this box reproduces the sibling row's
measurement rather than a local artefact. **No tolerance was touched.**

That table is a threading gate in its own right and worth naming as one: the
full suite runs on the global pool, `hardware_concurrency` wide, so every LTX-2.5
video golden after this change executes on 20 workers. The "Conv video decoder"
fixture carries a `res_x_y` block, so `Linear3d` and its `conv_shortcut` are on
that path too.

### 8.2 Determinism, proven twice and at two scales

* **Case B**, `test_ltx2_vae` "the decode is BIT-IDENTICAL across thread
  counts": the same latent decoded at 1, 2, 3, 5 and 8 workers, every arm
  `memcmp`-equal to the 1-worker arm, which short-circuits to the pre-change
  serial path.
* **The A/B harness**, independently: across **84 decodes** spanning worker
  counts 1, 2, 4, 8, 16 and 20, two sweep directions and two tensor shapes, the
  output checksum was **bit-identical every time** — `763841.709997177` at
  `c=64` and `973177.818164825` at `c=128`, on pseudo-random weights and a
  pseudo-random latent rather than the engineered fixture.

### 8.3 The wall-clock A/B

Same binary throughout; `VLLM_CPP_CPU_THREADS` selects the pool width and
nothing else changes. One decode driven through `Ltx2VideoDecodeStreaming` at a
**synthetic** decoder configuration — stated as synthetic because the shipped
checkpoint's `decoder_blocks` list lives in a checkpoint header this box does not
have. 14 runs per thread count: 7 on an ascending sweep and 7 on a descending
one, so an ordering drift would show as a spread rather than hide in a mean.

Configuration: `in_channels = base_channels = 64`, `out_channels = 3`,
`patch_size = 1`, one `res_x` block of 2 layers, non-causal, no timestep
conditioning, `PixelNorm`, replicate padding, latent `64 x 5 x 40 x 40`. Six
convolutions, 8.930 GFLOP.

| threads | runs | min s | median s | max s | spread | speedup | parallel efficiency |
|---|---|---|---|---|---|---|---|
| 1 | 14 | 1.9859 | **2.0418** | 2.1172 | 6.4% | 1.00x | 100% |
| 2 | 14 | 1.0266 | **1.0552** | 1.0843 | 5.5% | **1.93x** | 96.7% |
| 4 | 14 | 0.5464 | **0.5555** | 0.5674 | 3.8% | **3.68x** | 91.9% |
| 8 | 14 | 0.2920 | **0.3013** | 0.3072 | 5.0% | **6.78x** | 84.7% |
| 16 | 14 | 0.2129 | **0.2232** | 0.2597 | 21.0% | **9.15x** | 57.2% |
| 20 | 14 | 0.2024 | **0.2234** | 0.2534 | 22.8% | **9.14x** | 45.7% |

A second shape at the checkpoint's real `base_channels`, 3 runs each:
`c = 128`, latent `128 x 5 x 32 x 32`, 22.755 GFLOP — 5.1015 s at one thread
against 0.5276 s at twenty, **9.67x**.

**That 9.67x is the weakest number this row produced, and it is labelled as
such wherever it is projected.** `n = 3` against the table's 14, no min/median/max
recorded, and the same contended box — so it corroborates the table's shape at a
second channel width and is not independently a three-significant-figure result.

**The load this was taken at.** One-minute load average 4.03 to 6.77 across both
sweeps, on a box whose one-minute average had been between 2 and 52 earlier the
same day. It was not idle and no measurement here claims it was: one non-agent
process (`minimax-music3-`, PID 2291593, running for 9h57m) held ~1.07 cores for
the entire measurement. That process alone accounts for part of the gap at 16 and
20 threads, and it is also why the 16- and 20-thread spreads are 21-23% where
every count at or below 8 is under 7%.

**No ceiling is declared.** The curve flattens at ~9.15x from 16 threads, and the
next traceable hypothesis is named rather than the flattening being called a
limit: Amdahl on the passes this row deliberately did **not** parallelise. From
the measured 9.14x at 20 workers the implied serial fraction is 6.3%
(`1/(s + (1-s)/20) = 9.14` gives `s = 0.063`), which is the right order for
`PixelNorm`, `Silu`, `ApplyAdaLn`, the residual add and `expand` — every one of
them still serial, every one of them listed under §7. Memory bandwidth is the
second candidate and is not separated here. Whether the flattening is Amdahl,
bandwidth, or the ~1.07 cores another process was holding is **not resolved by
this measurement**, and §7 owns the follow-up.

### 8.4 What this row does NOT claim

* **No end-to-end render speedup.** No GPU here, no large-render host, and this
  row never ran a render. The 2681 s figure for a 448x256/25f decode is
  [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md)'s measurement on GB10 and
  nothing here divides into it.
* **No ratio against any oracle.** `ltx_core` is not installed on this box and
  has no pin; §1 and the parent spec's §7 both say why there is no denominator.
* **No composition figure with #1008.** The dtype row landed unmeasured for
  speed, and separating the two contributions needs the f64 arm rebuilt and
  re-timed. Not done.
* **Nothing about the shipped checkpoint's shape.** The harness configuration is
  synthetic. What generalises from it is the *scaling*, not the absolute wall.

### 8.5 ThreadSanitizer, with the instrument positive-controlled first

`cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_CUDA=OFF
-DVLLM_CPP_SANITIZE=thread`, three suites.

| suite | exit | cases | `WARNING: ThreadSanitizer` |
|---|---|---|---|
| `test_ltx2_vae` | 0 | 42/42 | 0 |
| `test_ltx2_tiling` | 0 | 10/10 | 0 |
| `test_ltx2_video` | 0 | 57/57 | 0 |

**Two things had to be settled before that table meant anything.**

First, the binaries would not start: `FATAL: ThreadSanitizer: unexpected memory
mapping`, exit **66**, before a single case ran. That is the kernel's ASLR
entropy against TSan's fixed shadow layout, not a defect in this change, and it
is a verdict-shaped instrument failure — a `&&` chain would have read that
non-zero exit as a race. `setarch x86_64 -R` fixes it and every run above uses
it.

Second, a sanitizer that reports nothing is indistinguishable from one that is
not instrumenting. A deliberate race — one unsynchronised `static int64_t`
incremented from inside `CausalConv3d`'s parallel body — was compiled into the
same lane, and TSan reported **87** `WARNING: ThreadSanitizer: data race` with
`EXIT=66`, naming the `CausalConv3d` lambda by its full signature and the line
the race sat on in the mutated tree. No anchor is cited for that line, because
that tree no longer exists. The mutation was then reverted, rebuilt, and rerun:
back to 0 warnings and `EXIT=0`. The clean table is a measured clean, not a
silent one.

### 8.6 Mutations, each with three facts

`git diff --numstat`, whether it BUILT with its `: error:` count, and the exit
code captured directly rather than through a pipe.

| mutation | numstat | built | exit | detected by |
|---|---|---|---|---|
| **T0** — all three dispatches reverted to serial | 13/8 | yes, 0 errors | **1** | Case A, `CHECK( 0 > 0 )` on the cursor |
| T1 — `CausalConv3d`'s OUTPUT loop alone reverted | 3/2 | yes, 0 errors | **0** | **nothing. 42/42 and 10/10 pass** |
| T2 — the padding gather alone reverted | 5/3 | yes, 0 errors | **0** | **nothing. 42/42 and 10/10 pass** |
| T3 — `Linear3d` alone reverted | 5/3 | yes, 0 errors | **0** | **nothing. 42/42 and 10/10 pass** |
| D1 — chunk-boundary-dependent value, visible at 1 worker too | 1/0 | yes, 0 errors | **1** | 10 cases in `test_ltx2_vae` + 2 in `test_ltx2_tiling`, including Case B |
| **D2** — the same defect made INVISIBLE to the 1-worker arm | 1/0 | yes, 0 errors | **1** | Case B's `memcmp`, on all four of the 2/3/5/8-worker arms |
| **R** — the production `Ltx2ConvVideoDecode` call site deleted | 17/2 | yes, 0 errors | **1** | Case A on the cursor AND on the value; Case B's non-degeneracy `REQUIRE` |
| T1, first attempt | 4/2 | **NO, 45 errors** | — | **nothing — a mutation that does not build establishes nothing** |

**T1's first attempt is in the table on purpose.** One unbalanced brace closed
the anonymous namespace early and produced 45 `-Werror` errors that read as
unrelated `unused-function` complaints hundreds of lines away. The runner
refused to draw a verdict, printed the errors and restored the tree. Had it run
the stale binary instead, it would have printed a plausible 42/42.

**T1, T2 and T3 are an honest gap, and it is owed as
[#1044](https://github.com/mudler/vllm.cpp/issues/1044)** (§7). Case A observes one
work-stealing cursor, and the cursor is shared: reverting any single site leaves
the other two dispatching, so the case reads non-zero and passes. It gates *"at
least one of the three sites dispatches partitioned work"*, not each site
individually, and T0 is what holds the conjunction. T3 additionally cannot be
seen by this fixture at all, because `decoder_blocks` is empty and `Linear3d` is
only reached through a `res_x_y` block. What does bound each site is §8.1's
golden table — the "Conv video decoder" arm reaches all three at 20 workers and
its margin did not move — and §8.3's wall-clock, which is what a serial
convolution would actually cost. Closing the gap properly needs one dispatch
observation per site, which needs an instrument the pool does not have today.

**D1 is in the table beside D2 because it is the weaker of the two.** D1
perturbs the first row of every chunk including the first, so the 1-worker arm
moves as well and the case fails on its value assertion before reaching the
`memcmp`. D2 perturbs only chunks that do not start at row 0, which is invisible
at one worker — `ParallelForRows` short-circuits to `body(0, nr)` there — so the
`memcmp` across worker counts is the only thing that can report it. It does, on
every one of the four non-base arms. That is the determinism guarantee mutated
rather than read.

### 8.7 The gate

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Run twice: once on the implementation commit, and once again at the branch head
so that a green gate is chained directly to the push.

| | first run | head run |
|---|---|---|
| `CONFIGURE_EXIT` | 0 | 0 |
| `BUILD_EXIT` | 0 | 0 |
| `: error:` count | **0** | **0** |
| `ctest -N` | **492** | **492** |
| `CTEST_EXIT` | **0** | **0** |
| result | **100% passed, 0 failed of 492** | **100% passed, 0 failed of 492** |
| total test time | 308.99 s | 316.87 s |
| one-minute load | 32 to 52 | **82 to 94** |

Two skips in both runs, both pre-existing and unrelated:
`test_modelopt_mixed_precision_checkpoint` and `test_voxtral_e2e`.

`No space left` **0** and `BFD` internal-error/assertion **0** across every build
and ctest log, both greps positive-controlled against a synthetic file carrying
the real message forms — 1 and 2 hits respectively there, 0 in the real logs.

None of the load-dependent suites flaked in either run, and the head run passed
at a one-minute load of 82-94 on a 20-core box, which is four times
oversubscribed. Free disk 21-30 GiB of 447 GB throughout; it dipped to 19 GiB
mid-run under other agents' builds. The sanitizer tree was 834 MiB and was
removed after §8.5.

### 8.8 The harness, recorded so the measurement is reproducible

Not shipped — a scratch developer tool, and the row deliberately does not add a
benchmark surface to the tree for it. Built against the gate's own `libvllm.a`:

```sh
g++ -O3 -DNDEBUG -std=c++20 -I include -I third_party ltx2_decode_bench.cpp \
    build/libvllm.a build/libblake3_vendored.a -lpthread -o bench
VLLM_CPP_CPU_THREADS=<n> ./bench <repeats> <channels> <lt> <lh> <lw>
```

It builds `Ltx2VaeWeights` from a fixed LCG, decodes through
`Ltx2VideoDecodeStreaming` with an untiled `Ltx2TileSizeConfig`, and prints wall
seconds, the derived GFLOP/s and a full-output checksum per run. The checksum is
what makes it a determinism instrument as well as a timer: it is printed at every
thread count and must not move.

## 9. Stop conditions

* Report `NEEDS_DECISION` rather than widening `kLtx2GoldenTol`, or any other
  tolerance, if a partition moves a golden. The answer to a moved golden is a
  partition that does not move it.
* Report `NEEDS_DECISION` rather than shipping a decode whose output depends on
  the worker count. A decode that gives different pixels at 1 thread and at 16 is
  a defect even with every golden green.
* Report the ThreadSanitizer result as it comes back. `test_ltx2_video` already
  carries a pre-existing LeakSanitizer leak under the `address,undefined` lane
  ([#1037](https://github.com/mudler/vllm.cpp/issues/1037), in the Gemma-4 rope
  cache via `DevicePool`); that one is not this row's and must not be allowed to
  mask a new report.
* Claim no number that was not measured on this box, in this session, with the
  load recorded.
