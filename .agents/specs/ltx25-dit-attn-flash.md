# LTX25-DIT-ATTN-FLASH — the LTX-2.5 DiT never opted into a fast attention op, and paid 47.84 s a forward for it

Row: `LTX25-DIT-ATTN-FLASH`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)), against the
model-matrix row `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`.
Issue: [#1549](https://github.com/mudler/vllm.cpp/issues/1549).

## Now

`ACTIVE`. The routing change landed as `90e8c3c85` (#1557), and **§10's pixel
A/B is now TAKEN** ([#1612](https://github.com/mudler/vllm.cpp/issues/1612)).
The acceptance criterion in §10.4 was committed before the renders, and the
result is in §10.7.

**Two results came out of the one lease, and they point opposite ways.** The
speed A/B that §8 carried as `PENDING` is settled: **7.112x**, same binary, same
lease, n=119 each. The pixel A/B **fails every registered check**, the control
is bit-identical so the delta is entirely the kernel's, and §10.5 reads that as
**visibly different** — a finding about a change already on `main`, filed as
[#1743](https://github.com/mudler/vllm.cpp/issues/1743). The op-level kernel is
correct to within its committed tolerance; what moves the picture is the render's
30 sampler steps, 4 DiT forwards each, amplifying that over 120 forwards.
**Whether a 7.112x arm that renders a different video should stay the default is
a product decision this row does not take**, and #1743 is where it is argued.

**AND THE FLASH ARM IS NO LONGER THE DEFAULT.**
[#1551](https://github.com/mudler/vllm.cpp/issues/1551) landed as `a4f2a9585`
while this row was in review: `vt::AttentionDenseFa2` is instantiated at
head_dim 128 and the DiT now routes onto its tensor cores, so
`VLLM_LTX2_DIT_FLASH_ATTN` is a three-way knob whose unset default is FA-2 and
whose `=flash` value is what §10.7 measured. **Nothing in §10.7 is invalidated
by that** -- it is a same-binary naive-against-flash pair at a recorded binary
sha256, and it remains the denominator #1551's own ratio is taken against. What
changes is the reach of the QUESTION: §10.4 wrote V4 as self-calibrating
precisely "why the FA-2 arm (#1551) can take this same criterion", and #1743 now
asks it of a rung this row never rendered. A pixel A/B of the FA-2 arm is owed
and is not this row's.

**AND #1743's CRITERION IS RELOCATED, NOT WIDENED.** §11 replaces what decides
the pixel verdict. Every threshold in §10.4 keeps its value, keeps its
computation and keeps its printed line; what it loses is the exit status. The
verdict now rests on **correspondence** (the frames and audio samples still line
up) and **incoherence** (the difference has no direction), which discriminate a
degraded render from a separated trajectory where an identity bound reads the
same on both. The retired bounds are printed under an `IDENTITY` verdict that
keeps reading `DIFFERENT` on §10.7's frames. Two gaps are declared rather than
proxied: [#1853](https://github.com/mudler/vllm.cpp/issues/1853), the
perturbation reference render that a lease owes, and
[#1854](https://github.com/mudler/vllm.cpp/issues/1854), absolute render quality
which nothing in this tree can gate. **And the criterion found something on the
frames that were already taken**: the video difference has no direction on any
of three statistics, the AUDIO does, and
[#1855](https://github.com/mudler/vllm.cpp/issues/1855) is that finding. §11.8
has the measurement, and it also has the ordering that shows this is not a
widened tolerance: the LARGER cross-build divergence passes and the SMALLER swap
fails.

The diagnosis is confirmed against the tree and the change is scoped
to **one production call site**. It was once scoped to that call site plus a
shared-memory cap repair in `LaunchAttentionDenseFlash`; that repair is
**reverted and out of scope** — see §4.3 for why, and for who owns the bound
instead. The A/B is the confirming evidence for the whole diagnosis and is not
optional; it needs `dgx:gpu0`, which is the box the denominator was measured on,
and no other box is a valid denominator for it.

Every bare `file:line` anchor below is read at this row's **base, `6b48edb2c`**,
which is `origin/main` at the claim, and every one of them has been re-read
against that SHA rather than carried forward. Two were wrong at the base as well
as at HEAD and are corrected: the reachability entry point, and the CPU
registration block. The change itself moves some of the rest —
`ltx2_device.cpp:421` most obviously — and a reader who wants the line rather
than the symbol should check out that SHA. Anchors written `file::Symbol`
survive the move and are what `scripts/check-symbol-anchors.py` gates.

## 0. The one-sentence finding

`src/vllm/model_executor/models/ltx2_device.cpp:421` calls `vt::Attention` for
every DiT self-attention, that op is frozen on a kernel whose own header calls
itself "Correctness-grade (M0.9)", and nothing anywhere tells a caller that a
faster op exists.

## 1. The measurement this row starts from

One LTX-2.5 DiT forward at `768x448 / 49f` (2352 video tokens), full checkpoint,
bf16, on `dgx:gpu0` (GB10):

| quantity | value |
|---|---|
| mean per forward | 47.84 s |
| median | 47.91 s |
| n | 119 |
| spread | 5.8% |

The samples are the engine's own `last=` lines, emitted by
`src/vllm/multimodal/render_phase_log.cpp:388` from the tick at
`src/vllm/multimodal/ltx2_video.cpp:4238-4244`. That tick fires immediately
before the DiT call, so `last=` is the interval between the start of forward
N-1 and the start of forward N: one fused two-stream (video + audio) DiT
forward plus the small host glue between them. It is **not** a per-denoise-step
number — one guided denoiser evaluation is one to four forwards — and this row
never treats it as one.

A governor/estimator in the same harness has reported 1.00 s, 69.1 s, 162 s and
396.9 s for this one quantity. Only `last=` is used here, and every number below
is reduced from those lines.

## 2. Attribution, and why it is not a guess

`vt::Attention` dispatches to `AttentionKernel` at
`src/vt/cuda/cuda_ops.cu:1463`. The kernel's own header comment at `:1456-1459`
says what it is. Its shape:

- one block of `kBlock = 256` threads (`cuda_ops.cu:33`) per (query, head);
- for every key, a 256-wide shared-memory tree reduction, about 12
  `__syncthreads()` per key;
- no K/V tiling, so K and V are re-read from global memory once per
  (query, head).

For the LTX video stream at 2352 tokens and 32 heads: grid = 75,264 blocks, each
looping 2352 keys = **1.77e8 block-key iterations per call**, times **48
layers**.

The per-iteration cost is not assumed. `.agents/specs/multimodal-speed.md:24-26`
records nsys attributing **98.9%** of the Qwen3-VL tower forward to this same
kernel, on this same box, at **5.70 ns per block-key iteration**. Scaling that
to LTX's geometry:

    1.77e8 x 5.70 ns x 48 = 48.4 s   vs the measured 47.84 s

A 1% match. The kernel is the forward. Everything else in the DiT — 48 blocks of
GEMMs, norms, RoPE, gating, six attentions per block — fits in the remaining 1%.

This is a scaling argument across two models, not a direct nsys run on LTX. It
is offered as such. The A/B in §7 is what actually settles it: if the diagnosis
is right, replacing this one kernel moves the whole forward, and if it is wrong,
it will not.

## 3. Why this was missed, which is the part worth keeping

`kAttention` was deliberately frozen on the naive kernel so that text decode
stays byte-identical (`src/vt/cuda/cuda_ops.cu:3120-3122`). That decision is
correct and this row does not touch it.

The consequence is the problem. The fast kernels are **separate ops that each
caller must opt into by name**: `kAttentionDenseFast`, `kAttentionDenseFlash`,
`kAttentionDenseFa2`. There is no automatic selection, no shape-based routing,
and no fallback notice. A model that never opts in silently gets the
correctness-grade kernel: **correct output, roughly 500x the cost, no warning
anywhere**.

Nothing in this tree can detect that. The output is right, so every golden
passes. The op is registered, so no refusal fires. `GetOpProviderStats` counts a
selection of `kAttention` and reports it as a success, because it is one. The
only symptom is the wall clock, and a diffusion render has no reference wall
clock to be measured against.

The lesson generalises past LTX: **every remaining caller of `vt::Attention` on
a non-decode path is a candidate for the same defect**, and nobody will be told.
This row does not sweep them — that is a separate row with its own issue — but
it states the shape so that the next reader recognises it.

## 4. What changes

### 4.1 The production call site

`src/vllm/model_executor/models/ltx2_device.cpp:421` at the base, the
self-attention branch of `AttentionDev`, moves from `vt::Attention` to
`vt::AttentionDenseFlash`. After this change the branch condition is `:418` and
the swapped call `:478`, with the A/B rung's `vt::Attention` at `:476`.

`vt::AttentionDenseFlash` is registered at `src/vt/cuda/cuda_ops.cu:3839-3840`
and is head_dim-generic up to 256 (`cuda_ops.cu:3335`), so LTX's video head_dim
128 and audio head_dim 64 are both served. Its contract requires a square problem
(`key.shape[0] == query.shape[0]`, `src/vt/ops.cpp:3089-3090`), which the
self-attention branch already satisfies: that branch is entered only when
`context == nullptr`, and `AttentionDev` sets `s = tq` in exactly that case
(`ltx2_device.cpp:356`).

The dispatch **rule** is unchanged. The branch is still selected by
`context == nullptr && a.bias == nullptr`, which is upstream's own
self-attention marker, and not by what the numbers happen to be. Only the op the
branch calls changes.

### 4.2 The host call site is deliberately NOT changed

`src/vllm/model_executor/models/ltx2.cpp:959` at the base (`:966` after this
change, which adds the marker comment above it) stays on `vt::Attention`. That
arm
computes into `std::vector<float>` and hands host pointers to
`Tensor::Contiguous`, so it is CPU-only by construction. On CPU
`kAttentionDenseFlash` is registered to the **same** `AttentionKernel` as
`kAttention` (`src/vt/cpu/cpu_ops.cpp:3750-3761`), so the swap would be a
byte-identical no-op that buys nothing and moves the L2 parity reference off the
reference op. The reference arm stays on the reference kernel.

It carries a `// VT-ATTN-NAIVE:` marker recording that reason beside the call, in
the form [#1578](https://github.com/mudler/vllm.cpp/pull/1578)'s
`scripts/check-attention-rung-consistency.py` reads: a `//` comment on the call
line or within the 20 lines above it, with a substantive reason after the colon.
Same for the A/B rung in `ltx2_device.cpp` (§4.4). That checker carries `ltx2`
and `ltx2_device` on `scripts/attention-rung-allowlist.txt` because this row was
in flight when it was written; once #1578 is on `main` both stems will report
`STALE (not a failure)` — the marked-and-therefore-no-longer-excused state its
allowlist header describes — and whoever runs preflight next may delete them,
which also edits that checker's own pinned-set test.

### 4.3 The shared-memory bound: attempted here, REVERTED, and owned by #1578

**This row does not touch `LaunchAttentionDenseFlash`.** An earlier version of it
did, and the reason it no longer does is the useful part of this section.

`LaunchAttentionDenseFlash` (`src/vt/cuda/cuda_ops.cu:3330-3348`) sizes its K/V
tile at

    shmem = 2 * kFlashBc(64) * head_dim * sizeof(Tin)

and never calls `cudaFuncSetAttribute`, so it is capped at the 48 KiB of dynamic
shared memory every CUDA architecture guarantees without an opt-in. This row
originally read that as a prerequisite: raise the cap, then swap the call site.

**The swap does not need it.** LTX's DiT runs at the stream dtype, which is
**bf16 in production** (`ltx2_device.cpp:1187` after this change, `:1166` at the
base: "the device stream dtype is bf16 (production) or f32 (the L2 parity arm)"),
and the video stream's head_dim is 128:

    2 * 64 * 128 * 2 = 32,768 B

which is inside the 49,152 B no-opt-in cap with room to spare. The audio stream
at head_dim 64 asks 16,384 B. **Neither shape ever needed the opt-in**, and the
7.680 s in §7.1 was produced by a bf16 render that would have launched
identically without it.

**The cap-raise was serving a shape that does not fit at all.** What actually
exceeds 48 KiB on this path is f32 at head_dim 128 (65,536 B) and f32 at head_dim
256 (131,072 B). GB10's queried
`cudaDevAttrMaxSharedMemoryPerBlockOptin` is **101,376 B**
(`cuda_device_caps.h:46`), so f32 256 does not fit **even with the opt-in** — it
was falling back to the bit-identical `AttentionDenseFast` and reporting a launch
it never made. Raising the cap therefore bought this row one shape it does not
run (f32 128), at the price of moving a shared helper across two files and
colliding with another row on the same lines.

**So it is reverted in full**, and the four files it touched —
`src/vt/cuda/cuda_ops.cu`, `cuda_device_caps.h`, `cuda_arch_tactics.cu`,
`cuda_paged_attn.cu` — plus `tests/vt/test_ops_attention.cpp` are byte-identical
to `main`. `vt::cuda::SetDynamicSmemOptIn` stays the file-local helper of
`cuda_paged_attn.cu` that it always was.

**[#1578](https://github.com/mudler/vllm.cpp/pull/1578) owns the bound, and it
takes the opposite and correct approach.** Instead of raising the cap it makes
the ADVERTISED domain honest: `AttentionDenseFlash` declares `head_dim <= 256`
while it can only launch bf16 192 / f32 96, and #1578 narrows the declaration to
what the code can actually do and `VT_CHECK`s above it. That is the
`supports_head_size()` polarity vLLM already has
(`vllm/v1/attention/backend.py:155-163`), and it is a better answer than an
opt-in because it is a property of the code rather than of whichever device
happens to be under it. #1578 merges first; after it, bf16 head_dim 128 is inside
the declared bound and this row's swap is unaffected.

**One consequence is disclosed rather than left to be found.** With the cap-raise
gone, the f32 L2 parity arm at production geometry (head_dim 128, 65,536 B) now
reaches `AttentionDenseFlash` and cannot launch: before #1578 that is a
`cudaGetLastError` throw at `cuda_ops.cu:3352`, and after #1578 it is a
`VT_CHECK` naming the head_dim. **It fails loud in both worlds and never
silently**, which is why it is a disclosure and not a blocker. It is not reached
by anything gated: the f32 arm is a parity reference exercised at the fixture's
reduced dimensions (§5), production is bf16, and an f32 render at 2352 tokens on
the naive kernel would have taken hours anyway. It is filed under `## Owed`.

**The original diagnosis is still on the record and still true**, because it was
not a reading of the code. `src/vt/cuda/cuda_attention_cross.cu:88-96` — a kernel
written for this very model, and itself a structural port of
`AttentionDenseFlashKernel` — says:

> LTX-2.5's video stream is head_dim 128, and at f32 a fixed 64-column tile would
> ask for `2 * 64 * 128 * 4` = 64 KiB of dynamic shared memory — over the 48 KiB
> a launch gets without opting in — so the kernel would fail to launch on exactly
> the real geometry while every reduced-dimension gate (head_dim 8 and 4) passed.

Same number, same arithmetic, same reason the existing gates cannot see it. That
author halved the tile (`ChooseTileCols`). This row now does neither: it runs the
bf16 shape, which fits, and leaves the f32 shape to the row that owns the
contract.

Two things follow that are worth keeping. The first is that "every
reduced-dimension gate passed" is why this row's numeric evidence is stated as
bounding the ARITHMETIC change and nothing more (§5, §7.1, `## Owed`). The second
is that `vt::AttentionCross` on CUDA is already flash-tiled, so **the LTX DiT's
cross-attentions were never on the naive kernel** — the four cross-attentions per
block and the two self-attentions were on different kernels the whole time, and
only the self-attentions were slow. That makes §2's attribution tighter, not
looser.

### 4.4 The A/B knob

`VLLM_LTX2_DIT_FLASH_ATTN=0` restores `vt::Attention` at the swapped call site,
so both arms of §7 run from **one binary**. Default on, read fresh per call on a
path taken about 96 times per forward. This mirrors `VT_FA2_DENSE`
(`cuda_ops.cu:3383-3389`), which exists for exactly this reason and is
documented as "Same-binary A/B + RED knob".

The knob is documented in `docs/ENVIRONMENT.md`, which `scripts/check-env-doc.py`
requires for any `VT_*` / `VLLM_*` string literal under `src/`.

## 5. Numerics: what is bit-identical and what is not

State plainly, because the two halves have different answers.

**On CPU: byte-identical.** `kAttentionDenseFlash` and `kAttention` are the same
registered function pointer (`src/vt/cpu/cpu_ops.cpp:3750-3761`). Every existing
CPU golden case in `tests/vllm/models/test_ltx2_device.cpp` must stay
byte-unchanged, and if one moves, the change is wrong.

**On CUDA: NOT bit-identical, and this row does not pretend otherwise.**
`AttentionDenseFast` differs from `Attention` in how the head_dim partial sums
are grouped (`include/vt/ops.h:3315-3316`): the naive kernel reduces across a
256-thread block, the warp kernel across 32 lanes with `__shfl_xor`. The
arithmetic is the same f32 online softmax; the association order is not.
`AttentionDenseFlash` is then bit-identical to `AttentionDenseFast`
(`include/vt/ops.h:3307-3317`), so the whole difference is naive-vs-warp
regrouping.

The magnitude is therefore a floating-point reassociation of a length-`head_dim`
sum, not an algorithmic change. The binding gate is the existing CUDA
host-vs-device parity case `tests/vllm/models/test_ltx2_device.cpp:753`, which
holds the f32 device forward to the CPU f32 host forward at
`kDeviceRoundOff = 2e-5` and the bf16 arm at `kBf16RoundOff = 5e-3`. That case
already exercises exactly the swapped path and does not need to be invented.

The measured deviation is reported in §7 as a number, not as a claim. If it
exceeds the committed tolerance, this row reports that and stops rather than
widening the tolerance. Widening a gate to admit a change is the failure this
protocol exists to prevent.

## 6. Reachability

AGENTS.md "Nothing lands dead". The swap must be proved to be *entered* from a
production entry point, not merely to compile.

Instrument: `vt::EnableOpProviderCallStats(true)` plus
`vt::GetOpProviderStats(OpId, DeviceType)` (`include/vt/op_provider.h:166-186`),
which counts dispatches through `GetOp`. This is a positive signal and it works
on the **CPU backend**, so the proof does not need a GPU.

Entry point: `vllm::Ltx2DitForwardDevice`, the production device forward called
from the denoise loop at `src/vllm/multimodal/ltx2_video.cpp:4246`. The test
drives that function, not a hand-constructed `vt::Attention` call.

**That anchor was wrong and it is the one the whole reachability argument rests
on.** This spec and the test both cited `:4055-4059`, which points at prose about
the res_2s step counter — at this row's base `6b48edb2c` and at HEAD alike, so
re-reading it at either revision would have caught it and neither did. `:4246` is
verified at both: it is the `im.on_device ? Ltx2DitForwardDevice(...)` ternary
inside the denoise loop, with the progress tick that emits §1's `last=` samples
seven lines above it at `:4238-4244`. `file::Symbol` form is what survives a
move, and `scripts/check-symbol-anchors.py` gates that form.

The assertion is **two-sided**, because a one-sided count cannot tell a routed
call from an added one:

1. `kAttentionDenseFlash` selections > 0 after one forward, and
2. `kAttention` selections == 0 after that same forward.

Half (2) is what makes it a routing proof: LTX's cross-attentions use
`kAttentionCross`, so after the swap there is no remaining `kAttention` caller
anywhere in the DiT forward. If somebody adds a second, unswapped self-attention
path later, that half goes red.

Mutation: delete the `vt::AttentionDenseFlash` call at
`ltx2_device.cpp:478` in a scratch copy, restoring `vt::Attention`, and rerun the
focused gate. Both halves must go red. The result is recorded in `## Outcome`
with the exact diff applied and the exact failure text, and the tree is restored
byte-for-byte afterwards.

The case's own process state is scope-guarded, and that is not tidiness. It
enables `vt::EnableOpProviderCallStats` and sets `VLLM_LTX2_DIT_FLASH_ATTN`, both
of which are per-process, and it contains `REQUIRE`s that unwind. Measured with a
scratch `REQUIRE(false)` after the `SetEnv` plus an observer case appended after
it: **without the guards the observer reads the knob still set and counts 8
leaked `kAttention` selections; with them it reads neither.** An unrelated case
failing because of which case failed before it is the shape this removes.

## 7. The A/B

Required, not optional: it is the confirming evidence for §2's whole attribution
chain.

**Method.** One binary. Two arms selected by `VLLM_LTX2_DIT_FLASH_ATTN`
(unset = flash, `0` = the naive kernel). Identical geometry
(`--width 768 --height 448 --frames 49`, 2352 tokens), identical seed, identical
prompt, identical checkpoint, one `rc` lease, one build.

**Denominator.** `dgx:gpu0` (GB10), which is the box the 47.84 s baseline was
measured on. A different box is not a valid denominator for this comparison and
this row will not substitute one.

**Statistic.** Per-forward **median** and **n**, reduced from the engine's own
`last=` lines exactly as `job/profile_forward.sh` already reduces them. No
governor output, no estimator output, no mean-of-a-skewed-tail.

**Prediction.** The precedent for the size of the win is
`.agents/specs/multimodal-speed.md` §7 and §16: the Qwen3-VL tower went from the
naive kernel at 56 ms/block to the warp kernel at 4.66 ms/block, **12.0x**, and
warp-to-flash then added 1.04x at 784 tokens and 1.82x at 1500 tokens (§14),
because the flash tiling pays off with context length and LTX runs at 2352. So
~48 s should become single-digit seconds.

**Stop condition.** If the measured speedup is far from that range, this row
reports the measurement and stops. It does not tune until the number looks
right.

### 7.1 What has been measured, 2026-08-21

Lease `6c724dfd-a5e8-4832-b4fb-d0fd7d6eb458` on `dgx:gpu0` (GB10, sm_121a),
source `30dce3a1d`, one binary built in-lease at
`-DVLLM_CPP_CUDA=ON` with cutlass-nvfp4, cutlass-fp8 and FlashAttention-2 all
`ENABLED for [121a]` — the same production feature set the recorded reference
build used. Artifacts under `/workspace/ltx25-attnflash/out/20260821T092516Z/`.

**Correctness first, and it passed before any speed number was read.**

| gate | result |
|---|---|
| `test_ltx2_device` on CUDA | **22/22 cases, 749/749 assertions, SUCCESS** |
| CUDA-vs-host f32 parity | video `8.9407e-08`, audio `4.47035e-08`, against the committed `2e-5` |
| CUDA-vs-CPU-backend bf16 | `0` on both streams |

**Which of this survives the §4.3 revert, stated rather than assumed.** The
lease also ran `test_ops_attention` at 10/10 and 88,439 assertions, and that
result is **withdrawn from this row's evidence**: it measured the cap-raise's
head_dim 64/128/256 dense-flash case, which is no longer in this branch. Two of
its arms never proved what they were read as proving in any case — f32 256 could
not fit GB10's 101,376 B ceiling and was silently answered by the bit-identical
`AttentionDenseFast`, so a numeric comparison could not tell a fallback from a
launch. The row claims nothing from that binary's `test_ops_attention` run.
[#1578](https://github.com/mudler/vllm.cpp/pull/1578) owns that file's head_dim
coverage.

**The `test_ltx2_device` rows and the render below DO survive it**, and that is
arithmetic rather than assertion. The measured binary carried the opt-in call,
but LTX renders bf16 at head_dim 128, whose tile is `2 * 64 * 128 * 2` =
**32,768 B**, and `SetDynamicSmemOptIn` returns immediately below 49,152 B
without touching the kernel. The opt-in was therefore a **no-op on every launch
these numbers came from**, so removing it cannot move them. The f32 parity case
runs the fixture's reduced dimensions and is below the cap by a wider margin
still.

The **numerics answer is now a number**: the swap is **not bit-identical on
CUDA**, and the measured deviation is **8.94e-08 video / 4.47e-08 audio**, which
is f32 round-off scale and sits **224x inside** the gate it was held to. The gate
was not widened. Caveat stated rather than left implicit: that case runs the
fixture's reduced dimensions, so it bounds the *arithmetic* change, not the
change at head_dim 128.

**Production reachability, on the real model, from the render's own log.**
`VT_OP_PROVIDER_STATS=1` makes each op announce itself once when it resolves.
In the full 21.00B render at `768x448/49f`:

- `op=21 device=1` (`kAttentionDenseFlash` on CUDA): **1** announce — present.
- `op=18 device=1` (`kAttention` on CUDA): **0** announces — the naive op is
  never resolved at all.
- `op=19 device=1` (`kAttentionCross` on CUDA): 1 announce.

That is the same two-sided claim §6's unit case makes, taken at full scale
through `vllm_video_engine_load` on `--device cuda` rather than on a fixture.

**The flash arm, per forward.** `768x448/49f` = 2352 video tokens, seed 20260820,
full checkpoint, bf16, reduced from the engine's own `last=` lines:

| n | median | mean | min | max | spread |
|---|---|---|---|---|---|
| 19 | **7.680 s** | 7.633 s | 7.109 s | 8.196 s | 14.2% |

Against the recorded **47.84 s** denominator that is 6.23x, and **6.23x is not
the number to quote.** The denominator carries two confounds this arm does not,
and both inflate the ratio. **The headline is ~6.0x:**

**Instrumentation.** The denominator ran under `job/runguard.py --stack-period 12`
(`render.log:1`), which samples `eu-stack -p` and therefore `ptrace`-stops every
thread in the process. Its own `stacks.txt` says what that cost: **523 samples,
median inter-sample delta 12.40 s against a 12.0 s period**, so ~0.40 s median
and 1.50 s maximum of stopped process per sample. At that cadence ~3.9 samples
land inside each 47.84 s forward, which is ~1.54 s, or **~3.2% of the
denominator**. The flash arm ran with no such sampler. Correcting only the
denominator gives **46.3 s / 7.680 s = 6.03x**.

**A different prompt.** The denominator's `render.log:1` carries a ~70-word
prompt; `scripts/ltx25-dit-attn-flash-ab.sh` uses one short sentence.
`ltx2_video.cpp:2253` sets `context_tokens = encoded.seq` **unpadded**, so the
DiT's four cross-attentions per block see a different number of keys in the two
arms. It is corroborated rather than inferred: `conditioning.tower` runs
**45.013 s** in the denominator against **28.426 s** in the flash arm. The sign
is the same as the sampler's — the longer prompt makes the denominator's forward
more expensive for a reason that is not the self-attention kernel — so it inflates
the ratio too, and it is not quantified here.

**So the honest statement is a range: 6.03x to 6.23x, and ~6.0x is the value to
quote**, with the sampler correction named and the prompt confound uncorrected
and pushing the same way. `6.23x` appears in this row's records only as the
uncorrected upper end of that range, never on its own. None of this is an A/B;
see below.

**The naive arm did not run, so there is NO same-binary A/B yet.** At forward 20
of the flash arm the `rc` worker was **lost** — `rc devices` then read
`dgx:gpu0 unhealthy (no contact)`, and it still did 43 minutes later.

**The cause is UNPROVEN and this row does not name one.** No memory trace was
taken, the worker's own log ends mid-forward, and the box did not come back to be
asked. Host-RAM exhaustion is the leading hypothesis only because GB10 shares
host RAM with the GPU and an unconstrained job has OOM-rebooted this box before;
that is a prior, not evidence. What IS established is that `job/ab.sh` as first
written carried **no memory guard, no sample cap and no memory trace**, unlike the
sibling campaign's `runguard.py` — so the run could neither avoid the failure nor
say afterwards what it was. That is a defect in this row's harness, and it is the
reason the next attempt can answer the question this one cannot.

Until the naive arm is taken **on the same binary in the same lease**, the
honest statement is:

- the flash arm is **measured**: 7.680 s median, n=19;
- the **~6.0x is a cross-run comparison** against a number produced by a
  different binary, in a different lease, under a stack sampler, on a different
  prompt — which is exactly the weaker form the same-binary rule exists to
  replace;
- so the A/B result is **PENDING**, not satisfied.

**SUPERSEDED 2026-08-22 by §10.7, and kept because it is the reasoning that
made the second attempt worth spending.** The naive arm was taken on the same
binary in the same lease. The pair is `naive` 45.547 s against `flash` 6.404 s,
n=119 each, no stack sampler: **7.112x**. Every bullet above is now history —
the 7.680 s median and the ~6.0x range are both retired, and neither should be
quoted from this section.

**The flash arm's artifacts do not say what was run, and that is a defect of this
row's harness rather than a caveat about it.** `arm-flash.log` opens at
`[render] + load` with no command line, `wd-flash/` is empty, and no
`phase-log.json` was written; the only description of the run was
`/mnt/nas_share/rc/ltx25-attnflash/job/ab.sh`, a mutable path on a share, whose
mtime is 25 minutes AFTER the run finished. The denominator has its full command
line as line 1 of its own `render.log` and this arm has nothing. So the geometry,
prompt, seed and sample cap behind 7.680 s are not verifiable from its own
evidence, which is the reason the two confounds above had to be established from
a `conditioning.tower` duration rather than read off a recipe.

Both halves are repaired. The harness is now
**`scripts/ltx25-dit-attn-flash-ab.sh`**, committed and therefore immutable per
revision, and every arm writes its own invocation — harness sha256, binary
sha256, source SHA, geometry, seed, prompt and the resolved command line — to
line 1 of its own log, plus a `harness_sha256` line into `PROVENANCE`. It also
carries a sample cap (13 per arm), a `MemAvailable` floor (12 GiB, the sibling
campaign's), a per-arm memory trace, a build cache keyed on the source SHA so a
resumed run does not re-spend 18 minutes compiling, and it runs the **naive arm
first** — the previous order took the cheap arm first and lost the box before the
expensive one, which is how a two-arm measurement became a one-arm one.

## 8. Gates

Exactly one result per gate. `PENDING` names the resource it waits on; it is
never a synonym for "probably fine".

| gate | where | result |
|---|---|---|
| CPU byte-identity | `test_ltx2_device`, `test_ltx2` | **PASS** — 22/22 and 652/652; 43/43 and 4581/4581; goldens unmoved |
| reachability, unit | `test_ltx2_device`, new case | **PASS** — flash 8 of 8, naive 0 |
| reachability mutation | in-place, restored and re-gated | **PASS** — both halves red, `CHECK( 0 == 8 )` and `CHECK( 8 == 0 )`, exit 1, compile rc 0 |
| reachability, production | the real render's own log, GB10 | **PASS** — `op=21 device=1` present, `op=18 device=1` absent (§7.1) |
| instrument scope guard, mutation | `test_ltx2_device`, in-place, restored | **PASS** — without the guards an appended observer reads the knob still set and 8 leaked `kAttention` selections; with them, neither (§6) |
| CUDA host-vs-device parity | `test_ltx2_device` on `dgx:gpu0` | **PASS** — f32 8.94e-08 / 4.47e-08 vs 2e-5; bf16 0 |
| bf16 head_dim 128 tile fits without an opt-in | arithmetic, `cuda_ops.cu:3338` | **PASS** — `2 * kFlashBc(64) * 128 * sizeof(bf16)` = 32,768 B against the 49,152 B every architecture gives without an opt-in. This is the whole of what the swapped shape needs from §4.3, and it is a property of the code, so no device is owed for it |
| the flash op's advertised head_dim bound | not this row | **NOT A GATE HERE** — §4.3's cap-raise is reverted, so `LaunchAttentionDenseFlash` and `tests/vt/test_ops_attention.cpp` are byte-identical to `main`. Owned by [#1578](https://github.com/mudler/vllm.cpp/pull/1578), which merges first |
| A/B, same binary, both arms | `dgx:gpu0` under an `rc` lease | **PASS** — `naive` 45.547 s median against `flash` 6.404 s, n=119 each, one binary, one lease, no stack sampler: **7.112x** (§10.7). This replaces §7.1's cross-run 6.03-6.23x range and the single-arm 7.680 s figure that used to sit here |
| pixel A/B at production geometry | `dgx:gpu0` under an `rc` lease, `scripts/ltx25-dit-attn-flash-pixel-ab.sh` | **FAIL, and the failure is the finding** — all four V and both A checks fail: mean \|delta\| 6.414 against `<= 1.0`, worst PSNR 22.269 dB against `>= 40`, worst SSIM 0.880694 against `>= 0.99`, V4 0.709 against `<= 0.10`, audio 29.368 dB and r 0.932682. Criterion registered in §10.4 before the run; result and reading in §10.7. §10.5 selects **visibly different**, filed as [#1743](https://github.com/mudler/vllm.cpp/issues/1743). No threshold moved (§9) |
| run-to-run control (`flash` twice) | the same lease | **PASS** — `flash-ctl` is **bit-identical** to `flash`, 49/49 frames, max \|delta\| 0, PSNR inf, SSIM 1.000000, and it passes its own C0 content checks. `R = 0.000000`, so the noise floor is nil and the whole treatment delta is the kernel's (§10.3's strongest branch) |
| C0 content, all three renders | the same lease | **PASS** — 9 checks: each render has 49 distinct frames, no near-uniform frame (`min_var` 3683.8-3739.0) and no zero-motion pair. The control's three were executed by re-running the committed tool, because the staged tool predated them (§10.7) |
| the numbers reproduce under the COMMITTED tool | this checkout, no GPU | **PASS** — phase [I] ran the tool from a tarball staged at `source_sha 3e2961ef0`, two commits behind head. Re-run at `7597cd741` over the same frames: **every check result and the verdict unchanged**, control C0 now executed and green, still exit 1. Recorded as `recheck.txt` / `recheck.json` / `recheck-cross.txt` in the evidence directory, so this row is re-derivable rather than asserted. Exactly one printed FIGURE moves, the audio `pearson_r` at its **15th** significant figure, and the check it feeds still reads `[FAIL]`; the JSON differs in 37 numeric leaves at `<= 2e-15` relative (the bound is that same `pearson_r` at `1.90457e-15`, 16 ULP) plus exactly 27 structural leaves that ARE the repair — the `judges` field on every check, the three `content.flash-ctl.*` checks and the three verdict keys. §10.7 enumerates it against a measured diff, because this row has twice claimed less than the truth |
| the comparison tool discriminates | `tests/scripts/test_ltx25_render_compare.py` | **PASS** — **45 tests, `OK`, re-run at this head on 2026-08-22** (the row read "37" before the control-C0 and `*)` tripwires were added). It needs no GPU, no lease and no NAS, so `PENDING until §10.7` was misreporting a gate that was already green: a dither passes, a one-pixel shift fails all four V checks, two all-black renders fail C0 while reading as a perfect match on every V, an unreadable input exits 2 while a threshold failure exits 1, A1 and A2 disagree on a time-shifted waveform, and the SSIM is pinned by its taps, its impulse response and three fixture values (§10.4). The count dates the run; it is not a floor to defend |
| the comparison tool runs on a lane | `scripts/agent-preflight.sh`, `.github/workflows/ci.yml` | **PASS** — it ran on NO lane when it landed: absent from preflight's `SUITES`, from the enumerated python block in CI and from `tests/CMakeLists.txt`, while the row above registered it as a gate. Both are registered now. Preflight SKIPs it when numpy is absent, which is the third state and never an `ok`; the CI lane installs `python3-numpy` so the lane that must not be silent cannot be |
| the harness's own preconditions | `tests/scripts/test_ltx25_pixel_ab_harness.py` | **PASS** — 27 tests, `OK`, at `2026-08-22`. The memory precondition and the arm-completeness check are extracted verbatim from the harness and run against a fabricated `/proc/meminfo`. The call sites that only a lease can execute are text tripwires and are labelled as such; §10.8 counts them and holds its own count. The count dates the run; it is not a floor to defend, and it said "19 tests" and "four call sites" after both had moved |
| full preflight | `scripts/agent-preflight.sh` | **PASS at HEAD** — and it was NOT before: `documentation-checkpoint` was red on two of this branch's own commits (see below) |
| `documentation-checkpoint` | CI, and locally over the branch range | **PASS at HEAD, RED before it, and the red was THIS BRANCH's** — `2aa78c69b` and `2f39a9426` each recorded a measurement in `.agents/benchmark-record.md` without writing the public projection that then existed. The control on the main-only range `4c193bd55..5d548d003` is rc 0, so it was not inherited. Both commits were replaced by one that writes the surfaces together when the branch was rebuilt, and the checker is re-run at each head rather than trusted to have stayed fixed — a job that has stopped appearing in a failing set is not the same fact as a job that passes. **THE COUPLING THAT PRODUCED THAT RED NO LONGER EXISTS, and the row is corrected rather than left to mislead:** `1db7e59cf` deleted `docs/STATUS.md` from the tree, deleted `scripts/check-doc-checkpoint.py` and `scripts/check-public-doc-tables.py`, and reduced this job to `check-now-current.py` plus `check-role-discipline.py`. A measurement now owes the row spec's `## Now` and nothing under `docs/` unless it adds a benchmark ID, which this one does not. An earlier revision of this row named `docs/STATUS.md` as the repair, and a reader who followed it would have recreated a file that `scripts/check-site.py` reds on for want of a `nav.yaml` entry |
| `build-newest-gcc` | CI | **PASS, and now green on `main` too** — it was red on `main` on `::getpid` in `test_qwen3_dflash2_gguf.cpp:547`, a file this change does not touch; [#1581](https://github.com/mudler/vllm.cpp/pull/1581) fixed it and this branch carries that fix through the merge. A red here after the merge is therefore this row's, not inherited |
| `build-test-cpu`, `sanitize-cpu` (both) | CI | **PASS, and they were INHERITED-RED until this branch's last merge.** All three used to fail on one case, `test_runner.cpp:1557` from #1273, owned by [#1602](https://github.com/mudler/vllm.cpp/issues/1602) and [#1608](https://github.com/mudler/vllm.cpp/issues/1608). [PR #1700](https://github.com/mudler/vllm.cpp/pull/1700) fixed #1608 on `main` by declaring `ROCM_ATTN`'s `MultipleOf(16)` block sizes, and merging `main` carried the fix here: on head `3f31bf0d4` all three report `pass`, and `main` itself now reports `success` on `build-test-cpu` and `sanitize-cpu (thread)`. **The row is corrected rather than left standing, because an inherited-red claim that has gone green is a record contradicting its own tree** — it would have excused a future red on three lanes that are now clean |
| `windows-msvc-cpu` / `-vulkan` | CI | **INHERITED, baseline-less lane, and the attribution is READ FROM THIS BRANCH'S OWN LOG rather than carried forward.** Job `97122086205` on head `36e596f22` prints `Windows portability contract OK` and then dies in `test_openai_api_server.exe` with `exited with status -1073740791` (`STATUS_STACK_BUFFER_OVERRUN`), immediately after `The decoder prompt (length 40) is longer than the maximum model length of 32`. **That is [#584](https://github.com/mudler/vllm.cpp/issues/584), NOT [#1649](https://github.com/mudler/vllm.cpp/issues/1649)**: #1649 is the `/W4 /WX` negated-by-`/w` refusal that fires BEFORE compilation, and [#1701](https://github.com/mudler/vllm.cpp/pull/1701) rescoped that gate, so the lane now gets past it and fails somewhere else. A post-#1701 red citing #1649 is a false attribution to an issue whose message the log no longer contains. A markdown-only control PR (#1295) fails the identical step; #584/#965 own it. **`windows-msvc-vulkan` on that head reads `cancelled`, which `gh pr checks` renders as `fail`** and which is not a verdict at all (#274) |
| `agent-record` | CI | **INHERITED, and it is NOT the checker its name suggests** — `scripts/check-agent-record.py` and `scripts/audit-live-rows.py` both return rc 0 locally on `main` and on this head, so the job's red is elsewhere inside it. Read by failure text rather than by job name: the job also runs `tests/scripts/test_check_site.py`, whose `test_rendered_benchmark_index_links_resolve_to_emitted_pages` shells out to `hugo` and **ERRORs with `FileNotFoundError: [Errno 2] No such file or directory: 'hugo'`** on a runner that has no `hugo`. That is the #1661 shape exactly — a guard that probes a `returncode` which a missing binary never produces — one restructure later. Newly inherited because `1db7e59cf` added the test. **The local-against-CI asymmetry is the whole reason this was misreadable**: `hugo` is on this developer's `PATH`, so `test_check_site.py` runs 7/7 `OK` here and every checker in the job returns rc 0, while the same job reds on a runner that lacks the binary. A job name is not a diagnosis, and "the checkers pass locally" was true and did not explain it. Already owned and in flight: [#1722](https://github.com/mudler/vllm.cpp/issues/1722), with fix [PR #1726](https://github.com/mudler/vllm.cpp/pull/1726). Not repaired here, because it is another row's open pull request and duplicating it is the failure that check exists to avoid. **[#1754](https://github.com/mudler/vllm.cpp/issues/1754) is a DUPLICATE of #1722** — same defect, same file, same job, filed later and while #1726 was already open — and both are open as of 2026-08-23. Cite #1722, which is the one carrying the fix; the duplicate is recorded here rather than silently preferred, because two open issues for one red is how a fix gets written twice |
| complete CI rollup | CI, exact SHA `3f31bf0d4` | **TAKEN, 18 pass / 6 skip / 3 fail**, and every failure characterised BY TEXT rather than by job name: `windows-msvc-cpu` and `-vulkan` both print `Windows portability contract OK` and then die in `test_openai_api_server.exe` at `STATUS_STACK_BUFFER_OVERRUN` after the length-40 refusal ([#584](https://github.com/mudler/vllm.cpp/issues/584)), and `agent-record` prints `agent record OK: ENGINE=170 ...` and then `FileNotFoundError: 'hugo'` ([#1722](https://github.com/mudler/vllm.cpp/issues/1722)). Green on this head and NOT inherited: `build-newest-gcc`, `cuda-fat-build`, `device-leakage`, `vulkan-spirv-freshness`, `documentation-checkpoint`, `commit-protocol-tag`, `pr-size`, `last-gated-commit`, both `verify` lanes, `build-test-vulkan`, `build-test-cpu-arm64`, `cuda-arch-features` |
| merge authority | `.agents/developer-preferences.md` | **GRANTED by the developer for this row, 2026-08-23**, in their own words in session. It was `PENDING` before that and the merge was refused on it: an agent relaying "merge it" is not developer consent, and this row waited rather than treating a relayed instruction as the value. The grant is scoped to `LTX25-DIT-ATTN-FLASH` and is not blanket authority |

**A side effect of that red is worth recording, because it was invisible.** The
`documentation-checkpoint` job runs `set -eu` and this checker is the FIRST of
three commands in the step, so `check-now-current.py` and
`check-role-discipline.py` **never ran in CI on this branch at all**. Nothing
hides behind it — both were run locally at the reviewed head and both returned
rc 0 — but a gate that stops two other gates from running is a wider failure than
its own message says.

**The A/B used to be the gate below the bar, and it no longer is.** Everything
this row claimed about SPEED rested on one arm until 2026-08-22; the pair now
exists, same binary and same lease, and §10.7 carries it. The head_dim refusal
that used to sit beside it here is gone from the list because the change it
gated is reverted, not because it passed.

**The gate that is below the bar is now the pixel A/B, and it fails.** It is
listed as `FAIL` rather than softened, because that is what the registered
criterion returned and §10.5 already wrote down how to read it: the two arms
render visibly differently, the control proves the difference is the kernel's,
and the finding is about `90e8c3c85` rather than about this measurement. The one
thing this row will not do with it is widen a threshold (§9).

## 9. Stop conditions

- `NEEDS_DECISION` if the CUDA deviation exceeds the committed tolerances in
  `test_ltx2_device.cpp`. The tolerance is not widened to admit the change.
- `NEEDS_DECISION` if `AttentionDenseFlash` refuses the bf16 head_dim 128 or 64
  shape LTX actually runs. §4.3 shows both fit without an opt-in, so a refusal
  there falsifies this row's premise rather than needing a workaround.
- `NEEDS_DECISION` if the measured speedup is far from §7's prediction.
- The A/B is reported as **pending an external resource** if `dgx:gpu0` is not
  free. It is never taken on another box, and never replaced by an estimate.
- **The pixel thresholds in §10.4 are never widened to admit the change.** A
  failing check is a finding about a change already on `main`, filed as its own
  issue with what diverged and by how much (§10.5), and it is never repaired by
  moving a number in `scripts/ltx25-render-compare.py`.

## 10. The pixel A/B — what the model RENDERS, designed before it is read

Issue: [#1612](https://github.com/mudler/vllm.cpp/issues/1612). This section is
written and committed **before the renders are taken**, because a criterion read
off the numbers it is meant to judge is not a criterion.

### 10.1 Why this section has to exist at all

Every other model in this tree leans on a token gate. Greedy decode gives a
discrete output, and either the tokens match the oracle or they do not. **A
diffusion render has no such output.** The DiT emits latents, the VAE emits
pixels, and nothing in that chain is a symbol that can be compared for equality
against a reference by construction.

So when §5 says the swap is **not bit-identical on CUDA**, the usual net is
absent. What remains is the reduced-dimension host-vs-device parity case
(`8.94e-08` video, `4.47e-08` audio against `2e-5`), and §7.1 already states its
limit in the same breath as its result: it bounds the *arithmetic* change — a
length-`head_dim` sum reassociated — at the fixture's dimensions. Production is
bf16, head_dim 128, 2352 keys, 32 heads, 48 layers, 120 forwards. Nothing has
measured that.

### 10.2 What bf16 predicts, which is that the frames WILL differ

This is a prediction registered in advance, not a result.

`vt::Attention` and `vt::AttentionDenseFlash` run the same f32 online softmax
and differ only in association (`include/vt/ops.h:3315-3316`). Two summation
orders of an `n`-term f32 sum differ by roughly `sqrt(n) * u` in the
random-walk regime, with `u = 2^-24 = 5.96e-08` the f32 unit roundoff:

| axis reassociated | `n` | relative deviation |
|---|---|---|
| the head_dim dot product, which §5 names | 128 | `~6.7e-07` |
| the key-axis online accumulation | 2352 | `~2.9e-06` |

The DiT then **stores that result to bf16**, whose spacing is `2^-7 = 7.81e-03`
relative — four orders of magnitude coarser. A perturbation of relative size
`d` moves the rounded bf16 value only when the exact value sits within `d` of a
rounding boundary, so the per-element flip probability is `d / 7.81e-03`:
between `8.6e-05` and `3.7e-04`.

The video stream carries `2352 * 32 * 128 = 9.63e6` attention output elements
per layer and `4.62e8` over 48 layers, so **one forward injects between 4.0e4
and 1.7e5 single-ULP bf16 flips**, and a render is 120 forwards inside a
nonlinear sampler that feeds each step's output into the next.

**Three things follow, and the third is the one that shapes the gate.**

1. **Bit-identical frames are not the expected outcome.** Predicting them and
   then finding a difference would make any threshold read as a retrofit.
2. **The floor is not zero.** "Within bf16 noise" has to mean something other
   than equality, because equality is not what the arithmetic predicts.
3. **A tight a-priori pixel bound is NOT derivable.** The sampler is nonlinear
   and iterative; whether ~1e5 ULP flips per step damp or amplify over 120 steps
   is an empirical property of this model at this geometry, not something the
   error analysis above can be pushed to answer. Anyone who claims to derive one
   is deriving it from an assumption of contraction that nothing here has
   measured.

So the bound is not derived from the arithmetic. It is derived from two things
that exist independently of this experiment: a convention the video-coding field
already agreed on, and the render's own scale.

### 10.3 The design: three renders, and the third is the whole argument

One binary, built once, in one `rc` lease on `dgx:gpu0`, on one staged
checkpoint set, at `768x448/49f` (2352 tokens), seed `20260820`, with the exact
70-word prompt of the recorded 20260820 baseline.

| # | render | knob | what it is |
|---|---|---|---|
| 1 | `flash` | `VLLM_LTX2_DIT_FLASH_ATTN=1` | the arm #1549 shipped |
| 2 | `naive` | `VLLM_LTX2_DIT_FLASH_ATTN=0` | the arm it replaced |
| 3 | `flash-ctl` | `VLLM_LTX2_DIT_FLASH_ATTN=1` | **flash again**, same binary, same seed |

**The `=1` in this table is the #1549 knob, not the one `main` builds today.**
When this run was designed and taken, `VLLM_LTX2_DIT_FLASH_ATTN` was BINARY:
`=0` selected `vt::Attention` and every other value, `1` included, selected
`vt::AttentionDenseFlash`. [#1551](https://github.com/mudler/vllm.cpp/issues/1551)
made the knob three-way and gave the flash rung the exact spelling `flash`, and
[#1751](https://github.com/mudler/vllm.cpp/issues/1751) made an unrecognised value
REFUSE at the first device DiT forward, so `=1` now aborts the render instead of
selecting flash. §10.4's routing table is what makes the row above readable: the
`knob=1` arm announced `op21_flash=1` and `ROUTING_OK=flash` in its own log, so
the binary that took these numbers is one where `=1` still WAS the flash arm.
The RUNNABLE path is where the repair went: `scripts/ltx25-dit-attn-flash-pixel-ab.sh`
exports `flash` for both flash arms since
[#1794](https://github.com/mudler/vllm.cpp/issues/1794), so a rerun selects the rung
this table names. The table itself is kept AS WRITTEN, because it records what that
run actually ran, and a literal corrected here would describe a binary that never
produced these frames.

**(3) is not a spare. It is the control, and without it the experiment does not
answer its own question.** A difference between (1) and (2) is only attributable
to the kernel if the box produces the same render twice when nothing changes.
cuBLAS reduction splits, allocator-dependent kernel selection and any
nondeterminism anywhere in 120 forwards plus a VAE decode would otherwise sit
inside the measured delta with no way to separate them. So:

- **control == 0** (bit-identical): the noise floor is exactly zero, and every
  bit of the flash-vs-naive delta is the swapped op. The strongest attribution
  available.
- **control ~= treatment**: the swap changed nothing the machine does not change
  by itself. The strongest possible *null* result, and it can only be stated
  because the control was taken.
- **control > 0 and < treatment**: the delta is partly kernel, and the control
  is the floor the thresholds must be read against.

Every one of those is an answer. None is available from two renders, which is
what #1612 asked for and what this section deliberately exceeds.

**Order: flash, naive, flash-ctl.** The naive arm is ~6x the wall clock and it
is the one whose loss leaves no A/B at all, so it is taken while the box is
known good rather than last. The control goes last because it is the only one
recoverable cheaply: the harness caches the binary keyed on the source sha, so a
follow-up lease reaches a render in minutes rather than re-spending the build.

**Both arms are instrumented identically and neither is stack-sampled.** §7.1
established that `runguard.py --stack-period 12` cost the recorded 47.84 s
denominator ~3.2% by `ptrace`-stopping every thread. This harness runs a
`MemAvailable` watchdog and nothing else, so the speed pair it produces needs no
sampler correction and finally replaces the cross-run 6.03-6.23x range with a
same-binary ratio.

**Routing is proved per arm, two-sided, from that arm's own log, and a failure
STOPS THE RUN.** `VT_OP_PROVIDER_STATS=1` makes each op announce itself once when
it resolves. The flash arm must show `op=21 device=1` **and no** `op=18
device=1`; the naive arm the reverse. A one-sided count cannot tell a routed call
from an added one, and "the knob was exported" is not evidence that the branch
was taken. The verdict was originally echoed inside a `case ... | tee` pipeline
and nothing acted on it, which is the shape that matters most here: if the knob
is not read, both arms are flash, the renders come out bit-identical, and this
design publishes PASS with all four thresholds vacuous — the strongest positive
verdict it can produce, from an experiment that had one arm. It is now computed
into a variable, tested outside any pipeline, and exits 46.

**The lease is not assumed to survive, and the box is not assumed to be empty.**
Two failures of this harness are recorded rather than smoothed over. It printed
`available 5` at +0s on 2026-08-22 and built into a lost worker anyway, so phase
[0b] now gates on `MemAvailable` against a 60 GiB start floor — derived from this
geometry's own recorded 79.503 GiB peak and 40.13 GiB low-water — waits for a
previous tenant's memory to be reclaimed, and refuses at exit 39 rather than
proceeding. And three lost workers each discarded every arm already rendered, so
`RUN_ID` is overridable and a resumed lease reuses any arm that is complete in
`$OUT`, states in that arm's record that its timings came from an earlier lease,
and still proves that arm's routing from the log it already has. A reused arm
makes the speed pair a cross-lease pair, and phase [H] says so rather than
letting the ratio imply otherwise. **That declaration is not bookkeeping.** The
speedup #1549 shipped on was qualified when this was written — its control arm
had never run, and §8 carried the A/B as `PENDING` because of it — so a resumed
run that quietly produced a cross-lease ratio presented as a within-lease one
would have replaced a stated gap with a false closure. The arm's `ARM` file, the
run's `PROVENANCE` and phase [H]'s own line all carry `timing_source`, and a
reader who takes the ratio without them has to have ignored three places that
said so. **In the event no arm was reused:** all three arms of `1612-r3` record
`timing_source=this-lease` and `PROVENANCE` records `speed_pair_same_lease=yes`,
so the 7.112x in §10.7 is a within-lease pair and this guard was never load-
bearing for it. It stays because the next resumed run is the one it protects.

### 10.4 The registered acceptance criterion

Committed as the defaults of `scripts/ltx25-render-compare.py`, so the numbers
are read by a tool that already holds the thresholds rather than compared to
them by hand afterwards.

| # | check | threshold | where it comes from |
|---|---|---|---|
| C0 | each RENDER, ON ITS OWN — arm A, arm B and the control: no near-uniform frame, every frame hash distinct, no zero-motion pair | all three, per render | **the hole every difference-only comparison has** |
| V1 | mean \|delta\|, 8-bit RGB | `<= 1.0` level | one level is the quantisation step of the artefact itself; a mean below it says the average pixel is within the PPM's own resolution |
| V2 | worst-frame PSNR | `>= 40 dB` | the video-coding "visually lossless" convention. This experiment did not choose it |
| V3 | worst-frame SSIM | `>= 0.99` | Wang et al. 2004, 11x11 Gaussian sigma=1.5 on luma. 0.98 is the usual transparency line; this is stricter, and it is the WORST frame rather than the mean |
| V4 | mean \|delta\| on luma / arm A's mean adjacent-frame MAD | `<= 0.10` | **the self-calibrating one** |
| A1 | audio PSNR vs full scale | `>= 40 dB` | same convention as V2 |
| A2 | audio Pearson r | `>= 0.999` | a waveform that has drifted in time fails this while PSNR can still look tolerable |

**C0 is not a formality, and this criterion did not have it at first.** Every
other line in the table is a DIFFERENCE, and a difference cannot tell two good
renders from two identically broken ones. Two all-black renders differ by zero,
score infinite PSNR and SSIM `1.000000`, and would read as the strongest
possible pass this table can produce. A run that exited 0 having written frames
that were all one colour has happened in this repository, so that is a recorded
failure mode rather than a hypothesis. C0 is computed per render before anything
is subtracted, and it is checked first.

**C0 covers the CONTROL, and it did not at first.** The control is the third
render and it is the one every reading in §10.5 is built on, and its content was
computed, printed and never checked. Its three checks are registered like the
arms', and they drive a DIFFERENT outcome: a failing arm check is a treatment
FAIL at exit 1, and a failing control check is exit 3, `CONTROL_DEGENERATE`,
because a control with no picture in it is a broken experiment rather than a
visible difference between two renders. §10.5 records the reading that status
selects and the repair's mid-flight timing.

**"Frames written" used to be a fourth C0 check and it is now a REFUSAL.** An
arm directory holding no `frame_*.ppm` leaves the tool at exit 2 with no JSON,
before any check is built, so the check could never report `False` and one of the
four sub-checks this table registered could never fire. Exit 1 is the status that
says *the two renders differ*, which is a reading of an experiment that happened;
an arm that rendered nothing is not that, and a broken render that reported it
would be indistinguishable from the finding this whole section exists to make. An
absent `audio.wav` stays a failed CHECK rather than a refusal, deliberately: the
video comparison is fully defined without it, so there is a result to report.

`verify_render.py` makes the same checks,
and they are recomputed inside `scripts/ltx25-render-compare.py` rather than
shelled out to, because that file lives on a mutable path on a share and this
one is committed per revision — the same reason §7.1 gives for the harness.

**V4 is the bound that is derived rather than borrowed, so it carries the
argument.** The denominator is the render's own frame-to-frame step: how much
one frame differs from the next, in the same 8-bit luma units as the numerator.
`0.10` therefore says *the two arms differ by less than a tenth of one frame of
this video's own motion*. It needs no re-argument at another geometry or another
prompt, because both terms move together — which is exactly what a constant
cannot do, and why the FA-2 arm (#1551) can take this same criterion.

**The scale is executable, and it comes from TWO populations that must not be
read as one.** An earlier version of this section quoted the first table and said
it was "pinned in `tests/...` so it is a gate rather than a claim". That sentence
was false of those numbers: what the suite pins is the second table, on synthetic
frames, and the two differ by 3.2x on mean `|d|` for the same perturbation.

**Population 1, MEASURED, not gated here.** The recorded 20260820 baseline's own
`768x448/49f` frames, on the NAS. Nothing in this repository can reach that share,
so no test asserts these:

| perturbation of a REAL frame | mean \|d\| | PSNR | SSIM | V4 ratio | verdict |
|---|---|---|---|---|---|
| none | 0 | inf | 1.000000 | 0.000 | bit-identical |
| +/-1 LSB on 3% of samples | 0.0199 | 65.1 dB | 0.99992 | 0.0026 | **PASS**, all four |
| one pixel of global horizontal shift | 5.183 | 28.1 dB | 0.8705 | 0.624 | **FAIL**, all four |

**Population 2, GATED.** The synthetic 96x64/6f fixtures of
`tests/scripts/test_ltx25_render_compare.py`, seed `20260822`, which every lane
can compute. These are the numbers that are a gate:

| perturbation of a FIXTURE frame | mean \|d\| | worst PSNR | worst SSIM | V4 ratio | verdict |
|---|---|---|---|---|---|
| none | 0 | inf | 1.000000 | 0.000 | bit-identical |
| +/-1 LSB on 3% of samples | 0.01935944733796296 | 65.04621554883309 dB | 0.99998566099925 | 0.0007008817611029302 | **PASS**, all four |
| one pixel of global horizontal shift | 16.611979166666668 | 21.789085745489977 dB | 0.7703600993461216 | 0.44871294988741245 | **FAIL**, all four |

**"GATED" was a label and not a fact until this revision, and a fresh review
found it.** The eight cells of the two lower rows were reproducible and
accurate — recomputed independently, all eight — and no test asserted any of
them. `test_dither_passes_every_video_check` asserted `mean < 0.1`, `psnr > 50`
and `ssim > 0.99`, bounds every one of those cells clears by two orders of
magnitude, and `test_one_pixel_shift_fails_all_four_video_checks` asserted four
booleans. A change to the fixture generator therefore moved all eight numbers
with the suite green: raising `make_render`'s `motion` from 3 to 4 left 43 of 43
tests passing while this table went silently wrong. A fixture that failed V4 by a
hair and one that fails it by 4.5x are the same test to a boolean, and the
argument two paragraphs below rests on which of those it is.

The eight cells are now asserted to the precision printed, by
`Discrimination.test_the_dither_row_of_the_gated_table_is_pinned` and
`Discrimination.test_the_one_pixel_shift_row_of_the_gated_table_is_pinned`, at
the precision this file already pins its three SSIMs to. Three independent
mutations of the generator — the motion step, the dither seed, the texture
period — each move at least one cell and each red the suite. The `none` row
needs no separate assertion: a bit-identical pair registers `video.bit_identical`
and no threshold at all, which
`test_identical_renders_read_as_bit_identical` already pins.

**Why they differ, and why both belong.** The fixture is a sine grating with a
4-pixel period plus uniform noise, so one pixel of shift moves each row by a
quarter of a cycle. The render's real frames are smoother, and the same shift
moves them less. The fixture is therefore a HARSHER shift and a slightly gentler
dither, which is the direction a discrimination proof should err in: the pass row
passes with less headroom than the real one, and the fail row fails harder.
Neither table calibrates the other, and the sentence that used to imply one
pinned the other is the defect being repaired here.

The SSIM those tables are computed with is itself pinned by property and by
value: the eleven taps of the `sigma=1.5` window, the impulse response as
`outer(k, k)`, its equality along both axes, the Rec.601 luma triple, and three
fixture SSIMs to eight decimals. Ten separate mutations of that metric -- sigma,
window, C1, C2, the second separable pass, the luma weights -- left the earlier
suite green, so "Wang et al. 2004" was a name and not a criterion.

So the thresholds sit between a dither and a single pixel of motion, nearer the
dither: V4 at `0.10` refuses anything above a quarter of a one-pixel global shift
on the fixtures, and a sixth of one on the real frames. A criterion that admitted
either shift row would not be a criterion.

### 10.5 Reading the result, stated before there is one

**The selector is a number, `R`, and the tool computes it.** Two of the three
readings below are opposite published verdicts on the same passing result, and
until this revision nothing computed the comparison they turn on: the tool
printed the control's figures for a reader to eyeball, and "comparable to the
delta" was left to a judgement made after the numbers were in view. So:

```
R = control mean |delta| on luma / treatment mean |delta| on luma
```

Both terms are the same statistic in the same 8-bit luma units, one from the
control-vs-its-own-arm comparison and one from the flash-vs-naive comparison.
The tool reports `R` as `control_ratio.ratio_mean_abs_luma`, alongside the RGB
ratio and both numerators. **It is reported and never checked**, because it
selects between two readings of a pass rather than between pass and fail, and a
check cannot express a choice between two answers that are both answers. `R` is
`null`, with a stated reason, when the treatment is bit-identical: that is a
division by zero the arithmetic expects, and §10.2 predicts it will not occur.

The boundary is `0.5`, and it is written here before any `R` exists. It is the
point at which the control accounts for half of the measured delta, so that
attributing the delta to the kernel and attributing it to the box are equally
defensible; a selector without a stated boundary is a judgement call wearing a
number.

- **All checks pass and `R = 0`** (the control is bit-identical) — the swap
  changes the render by less than a tenth of its own motion step, the difference
  is entirely the kernel's, and the verdict is **within bf16 noise**. The
  measured values, with their headroom, are then what belongs in the record.
- **All checks pass and `R >= 0.5`** — the verdict is **indistinguishable from
  run-to-run nondeterminism**, which is stronger, and the row additionally owes
  an issue for the nondeterminism itself, because a render that is not
  reproducible is its own defect.
- **All checks pass and `0 < R < 0.5`** — the delta is partly the kernel's, and
  the control is the floor every threshold above must be read against. The record
  states `R` rather than choosing one of the two verdicts.
- **`R` is `null`** — the two arms are bit-identical, which §10.2 says they will
  not be. Read that as a finding about the experiment (one arm rendered twice,
  a knob not read, a cached artefact) before reading it as a result.
- **`R` is marked UNUSABLE and the tool exits 3** — the CONTROL failed its own
  C0 content checks. None of the four readings above is available, because every
  one of them assumes the control is a repeat of a render and this one is a
  repeat of no picture. The treatment result stands and is reported; the
  *reading* of it does not exist, and the control arm has to be re-taken before
  one does. This is not "visibly different" and it is not a pass: it is a broken
  experiment, and it has its own status so that it can be neither.
- **Any check fails** — the verdict is **visibly different**, and that is a
  finding about a change already on `main`, not a failure of this work. It owes
  an issue naming what diverged and by how much, and it does not owe a widened
  threshold. Widening a gate to admit a change is the failure this protocol
  exists to prevent, and §9's stop conditions already say so for the numeric
  gate. This branch outranks the one above it: a failing check and a degenerate
  control together exit 1, because "the two renders differ" is established
  without the control at all and a broken control must not swallow it.

**THE CONTROL'S CONTENT WAS UNJUDGED, AND THAT IS REPAIRED HERE, MID-FLIGHT.**
`arm_content(control)` was computed, printed, and never registered as a check.
Arms A and B were judged on their own content and the control was not, so a
control of six one-colour frames left the tool at exit 0, verdict `PASS`, with
`R = 112.77` — the second branch above, **indistinguishable from run-to-run
nondeterminism**, which is the *stronger* of the two null readings. A control
that rendered no picture at all upgraded the published conclusion. That is
§10.6's class one step removed, and §10.4's own C0 rationale — a difference
cannot tell two good renders from two identically broken ones — applies to the
control-vs-arm comparison with identical force. "Reported, never checked" was
argued for the ratio `R`, never for the control's content, and §10.8 does not
list the control as exempt.

**The timing has to be said plainly, because this section's whole standing is
that it was committed before the numbers.** A fresh review raised this while the
renders were running in an `rc` lease and **before any number had been read**.
The criterion is therefore edited after the run started and before the run
reported, and the edit only ever TIGHTENS: it adds a way for a run to fail that
did not exist, removes none, and widens no threshold. Every V and A bound in
§10.4 is byte-for-byte what it was. A reader entitled to be suspicious of a
criterion edited mid-flight should check exactly that, and `git log -p` on this
file is where.

**WHICH ARM the control repeats is now an argument, not a convention.**
`--control-of {a,b}` names it, the JSON records it, and the control block prints
the sentence. The harness passed `--a naive --b flash --control flash-ctl` while
the tool compared the control against arm A, so the "run-to-run noise floor" was
a SECOND naive-vs-flash comparison: it necessarily read about the same size as
the treatment, and the second branch above would have been published whatever the
kernel did. A silent convention a caller can invert is not a convention.

### 10.6 The failure this design nearly shipped: a well-formed answer about the wrong thing

**A control compared against the wrong arm is not a weak control. It is a
fabricated result.** The harness passed `--a naive --b flash --control
flash-ctl` while the tool compared the control against arm A. `flash-ctl` is a
repeat of FLASH, so what got labelled "the run-to-run noise floor" was a second
naive-vs-flash comparison. It would have printed a number of the right
magnitude, in the right units, in the right place in the report — and it would
have been the treatment measured twice. §10.5's second branch,
**indistinguishable from run-to-run nondeterminism**, would then have been the
published verdict *whatever the kernel did*, and it would have read as the
strongest null result this design can produce.

**No threshold in §10.4 could have caught it.** V1 to V4 and A1 to A2 judge the
treatment, and the treatment was wired correctly; C0 judges each arm's content,
and all three arms had content. The defect was in the WIRING, one argument
deep, and every number downstream of it was arithmetically correct. A criterion
committed in advance protects against a threshold moved after the fact. It does
not protect against an instrument pointed at the wrong thing.

**This campaign keeps meeting that class, and the instances are worth naming
together, because the shape is what makes them recognisable:**

- the governor that reported `1.00 s`, `69.1 s`, `162 s` and `396.9 s` for one
  quantity — four well-formed answers, at most one of them about the thing that
  was asked (§7.1 reads the engine's own `last=` lines instead, for this
  reason);
- the append-only checker that read the WORKING TREE instead of the commits and
  returned `rc=0` three times over a violation that was there;
- the `static_assert` that compared a literal against itself and read
  `256 == 256`, staying green when the constant it was meant to pin changed.

Each returned a confident, structurally valid answer to a question nobody had
verified it was asking. **The repair for the class is the same each time: make
the instrument state what it is measuring, in its own output, in words.** So
`--control-of` is now a required-by-default argument, the JSON records
`control_of`, and the control block prints the sentence naming the arm it was
read against. A convention a caller can invert silently is not a convention, and
a reader who cannot see the wiring in the report cannot audit it.

### 10.7 The measurement

**TAKEN, 2026-08-22, `RUN_ID=1612-r3`.** All three renders completed in one
lease on `dgx:gpu0`, `rc` job `acff8e89-d704-4f17-a9f2-d354aba53b0d`, on the
binary `834cec557c16cf77eef9a2804cccd2189248c9c64973932670c7e92649320fb1` built
in the preceding lease (`b4d45dc7-3a74-48b0-94f3-eb9c907c1403`) from
`source_sha 3e2961ef0`, against `768x448/49f` (2352 video tokens), seed
`20260820`, prompt `sha256 451a8860...`, checkpoints staged at `/root/ckpt`.
Low-water `MemAvailable` 40.1 GiB on every arm. **The harness returned exit 1,
the pixel verdict, and `PROVENANCE` records `pixel_compare_rc=1`.**

**THE VERDICT IS `visibly different`, and §10.5 selects it without a
judgement call: any check fails.** All six V and A checks fail, the control is
bit-identical so `R = 0.000000`, and C0 passes on all three renders. Under
§10.5's last branch this is *a finding about a change already on `main`*, filed
as [#1743](https://github.com/mudler/vllm.cpp/issues/1743) naming what diverged
and by how much. §9 forbids repairing it by moving a number, and no number in
§10.4 moved.

**Routing, two-sided, from each arm's own log.** Every arm proved its own kernel
and none of them rests on "the knob was exported":

| render | knob | `op18_naive` | `op21_flash` | verdict line |
|---|---|---|---|---|
| `flash` | `1` | 0 | 1 | `ROUTING_OK=flash` |
| `naive` | `0` | 1 | 0 | `ROUTING_OK=naive` |
| `flash-ctl` | `1` | 0 | 1 | `ROUTING_OK=flash-ctl` |

**The control is exactly zero, which is the strongest branch §10.3 offered.**
`flash-ctl` is `flash` rendered again on the same binary and seed, and it came
back **bit-identical: 49/49 frames, max `|delta|` 0, PSNR `inf`, SSIM
`1.000000`** — and it passes its own C0 checks (49 distinct frames, no
near-uniform frame, no zero-motion pair). So the run-to-run noise floor is not
merely small, it is nil, and **every bit of the delta below is the swapped op.**

**The treatment, `flash` vs `naive`, same binary, knob only.**

| # | check | threshold | measured | result |
|---|---|---|---|---|
| C0 | content, all three renders | 9 checks | 49 distinct frames, `min_var` 3683.8-3739.0, 0 zero-motion pairs each | **PASS** |
| V1 | mean `\|delta\|` RGB | `<= 1.0` | **6.414156** | **FAIL**, 6.4x |
| V2 | worst-frame PSNR | `>= 40 dB` | **22.269 dB** (aggregate 25.822) | **FAIL**, by 17.7 dB |
| V3 | worst-frame SSIM | `>= 0.99` | **0.880694** (mean 0.901395) | **FAIL** |
| V4 | luma `\|delta\|` / adjacent MAD | `<= 0.10` | **0.709189** | **FAIL**, 7.1x |
| A1 | audio PSNR vs full scale | `>= 40 dB` | **29.368 dB** | **FAIL**, by 10.6 dB |
| A2 | audio Pearson r | `>= 0.999` | **0.932682** | **FAIL** |

Beside them: 0 of 49 frames bit-identical, max `|delta|` **253** of 255, mean
`|delta|` luma 6.049980, RMSE 13.045514, audio max `|delta|` 0.5557 full-scale
and RMS diff 0.0340 FS. **Between 98.9% and 99.7% of the pixels in every frame
differ**, and the `|delta|` histogram is broad and unimodal — so this is a
whole-image shift and not §10.4's "small mean hiding a bimodal tail".

**A2 exists to separate a drifted waveform from a different one, and here it
says DIFFERENT.** §10.4 registered the Pearson term because "a waveform that has
drifted in time fails this while PSNR can still look tolerable". Both tracks are
96,480 frames of 48 kHz stereo and a lag sweep over ±2000 samples finds its
maximum at **lag 0**, `r = 0.932682` — no shift improves the correlation. The
flash track also carries **3.3% less RMS energy** (ratio 0.967435), and the
peak difference is 18,209 LSB of a 16-bit scale. So the audio is not
mis-synchronised, it is a different rendering of the same prompt, which is the
same conclusion the frames reach.

**V4 carries the argument, because it is the bound derived from the render's own
scale.** The arms differ by **71% of this video's own frame-to-frame motion**
where the criterion admits 10%. §10.4's Population 1 then makes that concrete
against perturbations of a real frame, and the measured delta is **comparable to
one pixel of global image shift — worse on three of the four axes**:

| | mean `\|d\|` | PSNR | SSIM | V4 |
|---|---|---|---|---|
| ±1 LSB dither on 3% of samples — the bf16 floor | 0.0199 | 65.1 dB | 0.99992 | 0.0026 |
| one pixel of global horizontal shift | 5.183 | 28.1 dB | 0.8705 | 0.624 |
| **measured `flash` vs `naive`** | **6.414** | **25.8 dB** | **0.881 worst** | **0.709** |

**Read the SSIM column in the right direction, because an earlier revision of
this section did not.** SSIM is a similarity, so higher is less degraded. The
measured worst frame is `0.880694` against the shift's `0.8705`, and the measured
MEAN is `0.901395` — both ABOVE the shift, so on that one axis the swap is
slightly LESS degrading than a one-pixel shift. Saying "worse on every axis"
was wrong against the very table printed beneath it. Mean `|delta|`, PSNR and V4
are each worse, which is what the comparison is entitled to claim.

**322x the dither row's mean, and 39 dB below its PSNR.**

**§10.2 registered one question it refused to derive, and this answers it in the
amplifying direction.** The op-level difference is ~1e-7 relative and the
`test_ltx2_device` run in this same lease bounds it: 22/22 cases, 749/749
assertions, `SUCCESS!`, with the **device-vs-host maximum at `8.94e-08`** against
a committed tolerance of `2e-5` — the same figure §8 already carried. **Quote the
maximum of a NAMED family, not the minimum, and not a summary of the whole log.**
The lines labelled `device-vs-host` run from `5.96e-08` to `8.94e-08`, and an
earlier revision quoted the smallest of them, which is selective; the conclusion
does not need it, because the largest still clears `2e-5` by two orders of
magnitude. That revision also reached for a `1.49e-07` "device-vs-golden
maximum", and both halves of that were wrong: the reading sits inside a **CPU
backend** case, and the log carries several comparison families on different
tolerances — the bf16 keyframe arm reads `3.31e-03` legitimately — so no single
number summarises it. One family, named, with its own tolerance beside it.

**What that test does and does NOT establish, stated because the distinction is
load-bearing.** It bounds the ARITHMETIC at the fixture's reduced dimensions. The
`kAttentionDenseFlash selections = 8 (want 8), kAttention selections = 0` case
beside it reads `vt::GetOpProviderStats(..., vt::DeviceType::kCPU)` on
`ReducedParams`, so it proves the KNOB ROUTES on the CPU backend at fixture
size — it is not a CUDA measurement at head_dim 128 with 2352 keys over 48
layers, and `## Owed` already says no gate reaches that shape. **The CUDA routing
proof for this run is the render's own `op=21`/`op=18` log**, which is a separate
instrument and is the one §10.7 relies on.

So: **the kernel agrees with its reference wherever this tree can measure it, and
the render still moves.** The 4.0e4-1.7e5 single-ULP bf16 flips §10.2 predicts
per forward **amplify rather than damp** across the render's **30 sampler steps
at 4 DiT forwards each — 120 forwards** — by about two and a half orders of
magnitude on mean `|delta|`. §10.2 already counted 120 forwards; an earlier
revision of this section called them "120 sampler steps", which overstates the
number of sampler stages by 4x. The divergence also grows along the frame axis:
Pearson `r = +0.753` between frame index and mean `|delta|`, `r = -0.828`
against SSIM, rising from 5.03 over the first 8 frames to 7.05 over the last 8.
That is a property of this sampler at this geometry, and it is the reason §10.2
said no tight a-priori pixel bound was derivable from the arithmetic. **It is
also an inference about MECHANISM rather than a measurement of one**: what is
measured is a ~1e-7 op-level bound and a 6.414 pixel delta, and amplification is
the deduction that connects them.

**The speed pair, which §8 carried as `PENDING` for want of a second arm, now
exists.** Same binary, same lease, `n = 119` timed forwards each, no stack
sampler on either side:

| arm | median | mean | min | max |
|---|---|---|---|---|
| `naive` | **45.547 s** | 45.245 | 44.638 | 46.160 |
| `flash` | **6.404 s** | 6.329 | 5.871 | 6.576 |
| `flash-ctl` | 6.393 s | 6.321 | 5.882 | 6.660 |

**`naive / flash` = 7.112x**, and `flash-ctl` reproduces `flash` to 0.17% on the
median, so the control bounds the timing as well as the pixels. This replaces
the cross-run 6.03-6.23x range of §7.1 with a same-binary, same-lease ratio, and
`PROVENANCE` records `speed_pair_same_lease=yes` with `timing_source=this-lease`
on all three arms.

**THE DENOMINATOR MOVED TOO, and it is reconciled rather than quietly dropped.**
This row's title and §0 quote the naive path at **47.84 s** a forward. The
same-lease naive arm is **45.547 s**, 4.79% below it. §7.1 records the
`runguard.py --stack-period 12` sampler cost as an **absolute ~1.54 s** a
forward, "or ~3.2% of the denominator", so the unsampled figure is
`47.84 - 1.54 = 46.30 s` and the residual is `0.753 s`. **State the base, because
the two available ones differ and this row has already mislabelled one:**
`0.753 / 46.30 = ` **1.63%** of the unsampled figure, and `0.753 / 45.547 = `
**1.65%** of the measured arm. Both are quoted rather than one being passed off
as the other. That residual is recorded as open rather than attributed: a
different lease, a different prompt and a different binary are each candidates
and none is measured. **Take the 1.54 s and not the 3.2%**, because the
percentage is derived from it and compounding a derived percentage back through
a different base is how this paragraph went wrong once already — an earlier
revision read the 3.2% as a multiplier, got 46.36 s and quoted a 1.7-1.8%
residual on the 45.547 base. On that same base the absolute gives 1.65%, so the
correction is worth about 0.1 points and not the 0.15 a base swap would
suggest. None of
this moves the ratio, because both arms of the 7.112x were taken in one lease
with no sampler on either side. What it means is that **47.84 s is a superseded
number and 45.547 s is this row's naive denominator.**

**The cross-check against the 20260820 baseline, which is context and never the
control.** That render came from `a50c57d69`, an ancestor of the swap, on the
**naive** path. Against today's **naive** arm it reads mean `|delta|`
**9.452407**, PSNR **22.841 dB**, worst SSIM **0.803977**, V4 **1.026**, audio
PSNR 31.458 dB, `r` 0.959152. **Two naive renders across builds diverge more
than `flash` and `naive` do at one build.** Everything else that landed on
`main` in that window moved the render further than this swap did. The binary
lineage differs, so §10.8's reading holds — it bounds the class rather than
closing it — but it says the trajectory is unstable under any arithmetic
perturbation, and this swap is one instance of a general sensitivity rather than
a uniquely bad one.

**THE TOOL THAT PRODUCED THESE NUMBERS WAS NOT THE COMMITTED ONE, AND THAT IS
CHECKED RATHER THAN ASSUMED.** Phase [I] runs `$SRC/scripts/ltx25-render-compare.py`
out of a tarball staged on the share, and `PROVENANCE` records that tarball as
`source_sha 3e2961ef0` — two commits behind this branch's head. The two missing
commits are `12c880a52` and `7597cd741`, both of which only TIGHTEN: they add
the control's own C0 checks and the phase [L] `*)` arm. So the run was made by a
tool that could not return exit 3, and a degenerate control would have read as a
plain 0 there. **The whole comparison was therefore re-run at head `7597cd741`
over the same frames** — it needs no GPU, no lease and only the frames on the
share — and **every check result and the verdict are unchanged**, with the
control's three C0 checks now executed and all three `PASS`, still exit 1. **The
re-run is an artefact rather than an assertion:** `recheck.txt`, `recheck.json`
and `recheck-cross.txt` sit beside the originals in the evidence directory, so a
reader re-derives this row instead of taking it.

**What differs between the two runs. This paragraph has been wrong three times,
so it is enumerated against a measured diff rather than described.** First the
printed report, `diff pixel-compare.txt recheck.txt`, whose **three** hunks
(`32c32`, `33a34`, `45,47c46,52`) cover 13 changed lines:

- **one printed FIGURE moves**: the audio `pearson_r`, `0.932682102497646`
  against `0.9326821024976478` — they diverge at the **15th** significant
  figure, which is what a `1.9e-15` relative change is — and the same value
  again in its check-detail line. Nothing else numeric changes, and the check
  still reads `[FAIL]`. An earlier revision said "no printed figure changes",
  which this diff refutes;
- the rest is not a figure: the two section headers naming which checks decide
  the verdict, the three `content.flash-ctl.*` lines, `VERDICT FAIL` gaining
  `(exit 1)`, and the trailing `wrote <path>` line, which names a different
  output file and belongs to the invocation rather than to either tool.

Then the JSON, compared leaf by leaf with the `checks` array keyed by **`name`**
and not by index:

- **37 numeric values differ, every one by at most `2e-15` relative.** The bound
  is set by that same `pearson_r`, at `1.90457e-15`, which is **16 ULP** rather
  than one; every other leaf is 1 to 4 ULP. An earlier revision wrote
  `<= 1.9e-15`, which the maximum exceeds, and called the outlier "last-ULP";
- the **input paths**, a different mount;
- the `checks` array grows from **12 entries to 15**, nothing removed, the
  additions being exactly `content.flash-ctl.not_uniform`, `.distinct_frames`
  and `.motion`. Each new object carries FOUR keys — `name`, `pass`, `detail`
  and `judges` — and contributes **9** leaves here, because the fourth is
  counted in the next bullet;
- **every check gains a `judges` field**, `0 of 12` before and `15 of 15` after;
- three further keys appear: `treatment_verdict`, `control_verdict` and
  `control_ratio.unusable`.

Those last three bullets are **exactly 27 new leaves and 0 removed** (9 + 15 +
3), measured by flattening both documents. An earlier revision enumerated only
the final group of three and called the list exhaustive, which is the same
defect it was written to repair, at nine times the scale.

The `content.flash-ctl.*` checks and the three verdict keys ARE the exit-3
machinery `12c880a52` introduced, which is the direct evidence that the staged
tool could not have returned a 3. **Do not diff the `checks` array by index**:
the three insertions shift the tail, so an index-wise comparison over the
zipped 12 and 15 entries reports **6 differing check objects and 15 differing
`name`/`pass`/`detail` leaves, every one of them the same check at a moved
position**. None of those 15 is a real difference, and the `37` above is not
reproducible without keying by `name`.

**AND THE EXIT-3 PATH IS PROVED BY MUTATION RATHER THAN BY READING IT.** A
degenerate control was synthesised — 49 frames of one flat colour at this
geometry, with a real `audio.wav` — and the committed tool run against it with
both arms set to `flash`, so the TREATMENT is bit-identical and passes every
check it has. Summarised — this is a paraphrase and not a transcript, and
`degen.txt` holds the tool's actual output:

```
[PASS] video.bit_identical / audio.bit_identical, and all six arm C0 checks
[FAIL] content.degen.not_uniform      near-uniform frames 49 == 0 (min var 0.000)
[FAIL] content.degen.distinct_frames  1 distinct of 49
[FAIL] content.degen.motion           zero-motion pairs 48, mean adjacent MAD 0.0000
VERDICT CONTROL_DEGENERATE (exit 3)
```

That is exactly the case §10.5 records as having previously exited **0** with
verdict `PASS` and `R = 112.77` — a control of one-colour frames upgrading the
published conclusion to the STRONGER null. The repair holds: the treatment's own
verdict is untouched, the control's failure is reported separately, and the run
exits 3 rather than 0. Artefact: `degen.txt` beside the other evidence.

#### The four leases this measurement cost

Kept because §10.3's start gate and `RUN_ID` resume were built out of them, and
because [#1709](https://github.com/mudler/vllm.cpp/issues/1709) is still open on
the half this row does not own.

A run was earlier submitted on `f407019e3` as `rc` job
`2ccd1acf-dfa6-40b4-b095-1928caebe2c1`, `RUN_ID=1612-r2`. The attempt before it
(job `5fb9399f-4f4e-417c-adbd-4d741a2e18e4`, 2026-08-22) lost its worker during
the build with the box already at 114 of 119 GiB used before it allocated
anything, which is what §10.3's start-floor gate now refuses.

**AND THE GATE IS ALREADY REFUSING, WHICH FALSIFIES THE READING THIS ROW STARTED
FROM.** Job `2ccd1acf` reports, from its own poll log:

```
[pixab +0s] === [0b] the MemAvailable PRECONDITION, gated rather than printed ===
[pixab +0s]   MemAvailable 5.1 GiB < 60.0 GiB, waited 0s of 1200s
[pixab +360s]  MemAvailable 5.1 GiB < 60.0 GiB, waited 360s of 1200s
```

`5.0` GiB in the lost lease an hour earlier, `5.1` GiB now, **flat for six
minutes across two leases and at least two intervening jobs.** So this is NOT a
previous tenant's memory awaiting reclamation, which is what
[#1709](https://github.com/mudler/vllm.cpp/issues/1709) currently records and
what the wait in §10.3 was designed for: something on that box holds ~114 GiB
persistently. The condition is PERSISTENT and the transient reading is
falsified. That distinction is only available because the gate logs EVERY poll
rather than only its refusal — a single refusal line would have said "5.1 GiB"
and left the cause open. A read-only diagnostic to identify the resident
allocation is with the operator, and #1709 owes the correction. The renders
could not be taken on that box while it held, and no arm had run at that point.
**The box did later come back**, and the run recorded at the top of this section
took all three arms on it at a 40.1 GiB low-water mark. That resolves the
schedule, not the defect: #1709 stays open, because nothing here explains where
the 110.41 GiB went or stops `rc` handing out a lease against it next time.

**THE DIAGNOSTIC IS TAKEN, AND THE MEMORY BELONGS TO NOBODY.** `rc` job
`ab12aac1-b862-4ac6-8292-9f2c641e6a8d`, read-only, 2026-08-22T16:55:46Z, run
from inside a lease because the fleet rule forbids `ssh`. Every ordinary owner
is excluded by measurement rather than by argument:

| field | value |
|---|---|
| `MemTotal` | 119.63 GiB |
| `MemFree` | 5.23 |
| `AnonPages` | 0.93 |
| `Cached` | 0.91 |
| `Buffers` | 0.05 |
| `Shmem` | 0.04, and it sits inside `Cached` rather than beside it |
| `Slab` | 1.10 |
| `VmallocUsed` | 1.00 |
| **accounted** | **9.22 GiB** |
| **UNACCOUNTED** | **110.41 GiB, 92.3% of the box** |

No process: `ps aux --sort=-rss` lists the `rc` worker at 11 MB and two zombies,
and the sum of `VmRSS` over every visible `/proc/*/status` is **0.0 GiB**. No
tmpfs: `/dev/shm` is a 64 M mount containing nothing, `du` 0. Not the GPU being
busy: `nvidia-smi` reads `0%`, `11W`, `No running processes found`. Not a
container view artefact: the cgroup reports `memory.max=max` and
`memory.current` 113 MiB. Not transient: 5.0 GiB at 15:49Z, 5.1 GiB flat through
16:39Z, 4.98 GiB at 16:55Z, across four leases by three submitters, on a box
`up 2:33` at load average 0.25.

On a unified-memory part the driver takes host RAM for the GPU, and an
allocation that outlives its process is attributed to nothing userspace can see
— which fits every observation, including `Memory-Usage: Not Supported` being
the one meter that would have named it. **That is a hypothesis and this row does
not assert it.** What is established is the 110.41 GiB gap and the exclusions.

The operational consequence does not depend on the mechanism. **Nothing a lease
can do repairs it** — there is no process to kill and no file to delete — so
this row's measurement was `PENDING` on a named external resource for as long as
it held, which is a result under AGENTS.md and never a synonym for "probably
fine". `dgx:gpu0` continued to report `ready` and hand out leases throughout,
which is the controller half of #1709 and is unrepaired.

**Five leases spent and no wrong number among them.** `5fb9399f` lost its
worker to an OOM during a build started against 5 GiB; `2ccd1acf` waited its
full 1200 s at a flat 5.0 GiB and refused with exit 39; `ab12aac1` measured the
box; `b4d45dc7` built the binary once the box was free; `acff8e89` rendered all
three arms from that cached binary. The one thing this row did not do is take
the renders somewhere else: §7's denominator argument binds, and a ratio against
a different GPU would not have been this measurement taken late, it would have
been a different measurement. **Waiting for `dgx:gpu0` is why the 7.112x ratio
is a same-lease pair rather than a cross-run range.**

### 10.8 What this deliberately does not measure

- **WHETHER THE CONTENT IS THE RIGHT CONTENT. C0 is necessary and it is not
  sufficient.** C0 asks three questions — is there variance in each frame, is
  every frame hash distinct, does anything move between them — and **two
  identical sequences of pure noise answer all three and pass every C and V
  check with the verdict `PASS`**. That is demonstrated, not feared. The same
  hole admits a pair of renders that are perfectly consistent with each other
  and wrong together: the wrong prompt, the wrong seed, the wrong checkpoint
  class, a text encoder that produced nothing, a VAE decoding a latent the
  sampler never refined. Each of those is a *difference from an intended
  render*, and every measurement in this section is a difference between two
  arms, so none of them can see it. Nothing here should be extended to try:
  a check for "is this a golden retriever shaking off water" is a model, not a
  threshold. **The only absolute reference present is the cross-check against
  the recorded 20260820 baseline in phase [J] of the harness** — a real render
  of this prompt at this geometry, from an ancestor build — and it is a
  cross-check precisely because its binary lineage differs, so it bounds this
  class rather than closing it. When that baseline is unreachable, the run says
  so and this class is unmeasured for that run.
- **MOST OF THE HARNESS ITSELF.** `tests/scripts/test_ltx25_pixel_ab_harness.py`
  extracts the `MemAvailable` reader, the start gate and the arm-completeness
  check verbatim from `scripts/ltx25-dit-attn-flash-pixel-ab.sh` and runs them
  against a fabricated `/proc/meminfo`, so those three execute here. **The render
  loop, the routing assertion, the phase [I] call site, the phase [L] exit, the
  phase [F] unit-gate refusal and the signal traps do not.** All six of those
  surfaces are now pinned as TEXT at least in part, by eleven tripwire tests — the
  suite asserts the exact call site that shipped inverted, the rule that only the
  PRIMARY comparison becomes the run's exit status, the three-sided op proof that
  #1794's lying label needs, the branch that spells the FA-2 arm's knob as
  `unset` rather than as an empty export, the exact `exit` lines, both arms of
  phase [L]'s `case` that this row wrote (the exit-3 verdict and the `*)`
  fallback that a status nobody defined would otherwise fall through silently),
  the two unit-gate statuses and the four `trap` lines. **Those two counts are
  about different sets**: six is how many things never execute, ten is how many
  tests pin them, because phase [L] carries three tests by itself and phase [I]
  and the routing assertion carry two each. They both read "six" for one commit,
  which looked like two records agreeing. A text assertion is a tripwire, not a
  proof: it catches the inversion that happened and would not catch a rewrite
  that reintroduced it in different words. **The render loop is pinned at exactly
  one line** — the knob-selection branch — **and its watchdog poll, memory floor
  and timeout are pinned by nothing at all.** `dgx:gpu0` under a lease is the only place those lines run, which is why every one of them
  was wrong at once: they had never executed anywhere a test could watch. **That
  is a structural explanation of a cluster rather than a run of coincidences**,
  and it tells the next reader which claims in this harness are load-bearing and
  which are decorative — what a test can execute is now checked, and what only a
  lease can execute is a comment to be verified against the run's own log.

  **This list and the tripwire count had both drifted, and a fresh review caught
  it.** The prose named four unexecuted things while the `Wiring` class defined
  five tests, and the two the prose omitted — the phase [F] unit-gate refusal and
  the signal traps — were text-only and absent from `## Owed` as well. The error
  was in the SAFE direction: nothing claimed as executed was in fact only
  text-pinned. It was still a number in a document that no longer described the
  file beside it, so
  `TheDisclosureCountsWhatIsThere.test_the_wiring_docstring_names_as_many_tripwires_as_it_defines`
  now holds the count against the class and a further tripwire cannot be added
  silently. It fired on exactly that when the exit-3 tripwire was added, and
  again when the `*)` tripwire was.
  `TheDisclosureCountsWhatIsThere.test_the_spec_counts_tripwire_TESTS_and_not_unexecuted_SURFACES`
  holds the same number against THIS SECTION, which the docstring gate never
  reached: the review that raised it found the two sixes counting different sets
  and neither list naming the `case` arm.

  **Two mutations of the memory gate are deliberately a TIMEOUT rather than a
  failed assertion, and both are named because one of them was not.** Inverting
  the floor comparison (M27) makes the gate wait out its whole budget instead of
  proceeding, and neutering the wait-budget comparison
  `[ "$waited" -ge "$budget" ]` so that it never fires (M29) makes the loop run
  forever on a box that never recovers. Either way the suite never returns. The
  mutation runner records that as RED with the reason named, because a suite that
  did not report `OK` is not a suite that passed. A timeout read as success is
  the exact silent hole a wait-for-quiet loop already cost this campaign once, so
  that path is exercised on purpose rather than left for whoever next touches
  either line. **The hang is not engineered away**, and that is a choice: a
  budget that gives up early to keep a test suite responsive is a weaker gate on
  a four-hour lease than one that waits, and the guard's whole purpose is to
  wait. What is owed is the disclosure, and M29 had not carried one.

- **A GUARD THAT NO MUTATION COULD REACH, now removed rather than kept.**
  `arm_is_complete` opened with `[ -d "$d" ] || return 1`, and deleting it left
  the harness suite green: the glob does not expand for a directory that is not
  there, so `ls "$d"/frame_*.ppm | wc -l` reports 0 and the frame-count check
  refuses the arm on its own. Observable behaviour was identical with the line
  and without it, so it was a redundant guard rather than a defect — and it is
  the same argument §10.4 already made against C0's fourth check. It is gone, the
  two remaining guards are each proved observable by their own mutation, and
  `test_an_absent_directory_is_not_complete` now says in its docstring which line
  does the refusing, so the next reader adds it back deliberately or not at all.
- **PPM is 8-bit.** The comparison is on the artefact the pipeline writes, which
  is already quantised from the VAE's float output. A difference below `1/255`
  relative is invisible to it. That is the right resolution for the question
  "does it render the same video" and the wrong one for "how large is the
  latent-space deviation"; the second is the host-vs-device case's job and it is
  answered at the fixture's dimensions in §7.1.
- **One prompt, one seed, one geometry.** A sampler is chaotic and one trajectory
  is one trajectory. This bounds the swap on the trajectory production actually
  ran and recorded, and it does not claim a bound over the prompt distribution.
- **The 20260820 NAS baseline is a cross-check, never the control.** It was built
  from `a50c57d69`, an ancestor of the swap, so it is a different binary lineage:
  everything else that landed on `main` in between sits inside any delta measured
  against it. It is compared anyway, because how far two naive renders drift
  across builds bounds how much of the A/B delta could be something other than
  the kernel — but the same-binary pair is the evidence and this is context.

## 11. The criterion asked the wrong question, and it is RELOCATED rather than widened (#1743)

Issue: [#1743](https://github.com/mudler/vllm.cpp/issues/1743). This section is
written and committed **before any number computed by the new criterion is
read**, for the same reason §10 gives: a criterion read off the numbers it is
meant to judge is not a criterion. §11.6 states how every outcome will be read,
in advance, so that no branch is chosen after the fact.

### 11.1 What §10 established, and the one thing it cannot distinguish

§10.7 is not in dispute and nothing here revises a digit of it. Re-verified for
this section from `/mnt/nas_share/rc/ltx25-attnflash/pixel-ab/1612-r3/`, which
this box can read: `pixel-compare.txt` and `pixel-compare.json` carry
`verdict FAIL`, the six failing checks at exactly the values §10.7 tabulates,
and `control_ratio.ratio_mean_abs_luma = 0.0`.

Three facts sit beside each other and they point in opposite directions:

| pair | binary | path | mean \|delta\| RGB | worst SSIM |
|---|---|---|---|---|
| `flash` vs `flash-ctl` | one | flash both sides | **0** (49/49 bit-identical) | 1.000000 |
| `flash` vs `naive` | one | the swap | **6.414156** | 0.880694 |
| `baseline-20260820` vs `naive` | **two** | naive both sides | **9.452407** | 0.803977 |

The first says the box is deterministic: repeating a render changes nothing, so
the noise floor is exactly zero and the second row is entirely the kernel's.
The third says that between `a50c57d69` and `3e2961ef0` something **on the naive
path** moved the render **further than the swap did**. §10.8 is right that the
third row is CONTEXT and not a control, because the binary lineage differs and
every other commit in that window sits inside it. It is quoted here for what it
bounds and never as a control, and no threshold below rests on it.

**So §10.4's criterion answers a question, correctly, and it is not the question
#1743 has to answer.** V1, V2, V3, A1, A2 and V4 all measure IDENTITY: how close
are these two renders to being the same picture. The answer is "not close", it is
measured, and it stands. What nobody has measured is whether that distance is
**unusual for this pipeline** or **ordinary for it**, and identity bounds cannot
tell those apart, because they read the same on both.

### 11.2 The trap this section exists to avoid

§9's stop conditions and [#1668](https://github.com/mudler/vllm.cpp/issues/1668)
forbid the obvious repair, and they are right to. Moving `max_mean_abs` from
`1.0` to `7.0` would turn the gate green while asserting less than it asserted
before, and the next change that moved the render by 6.9 would land unseen. **No
threshold in §10.4 is widened by this section. Every one of them keeps its
value.** `git diff` on `scripts/ltx25-render-compare.py` is where a reader checks
that, and the six `DEFAULT_*` constants are byte-for-byte what they were.

What changes is **which checks decide the exit status**, and the new ones are
built to a different shape.

### 11.3 The shape: correspondence and incoherence

The model is [#1711](https://github.com/mudler/vllm.cpp/issues/1711)'s
`head < 0.5 * serialize`. That check is structural rather than tolerated because
under the correct behaviour one side holds ZERO of the quantity and under any
wrong behaviour it holds at least one in FULL, so any constant strictly between
0 and 1 separates the two populations and the constant carries no argument.

The same split exists here, and it is not in the SIZE of the difference. It is
in the difference's **direction** and in its **correspondence**.

**A reassociated sum makes two renders exchangeable. A defect makes one of them
worse.** `vt::Attention` and `vt::AttentionDenseFlash` compute the same f32
online softmax in a different association order (`include/vt/ops.h:3315-3316`).
Neither order is the reference and neither is a degradation of the other: §10.2's
1e-7 perturbation enters a chaotic sampler and the two trajectories separate.
Under that null, arm A and arm B are two draws from one distribution, so for any
one-sided QUALITY statistic — sharpness, blockiness, motion energy, audio energy
— the sign of the per-tile difference is a fair coin.

A real degradation is not a coin. A blur removes high-frequency energy from
**every** tile of **every** frame. Block artefacts add a step at the block grid
in every tile. A silenced track loses its energy in every window. Each of those
moves every term the same way.

So define, for a statistic `s` measured on `N` matched terms (one per tile per
frame, or per window):

```
K(s) = | SUM_k (s_k^A - s_k^B) |  /  SUM_k | s_k^A - s_k^B |
```

`K` lies in `[0, 1]` by construction, and the two populations sit at its two
ends:

- **coherent difference** (any degradation that acts in one direction):
  every term has the same sign, the numerator equals the denominator, and
  `K = 1` EXACTLY. Not approximately, and not "large": the identity is
  algebraic.
- **incoherent difference** (a chaotic trajectory separation): the signs are a
  fair coin, the numerator is a random walk of `N` steps against a denominator
  that is their sum, and `K` concentrates near `N^(-1/2)`.

The threshold is `K <= 0.5`, and `0.5` is chosen for the only reason a constant
in an open interval can be chosen: it is the point at which the coherent part of
the difference accounts for exactly half of the total variation, so attributing
the difference to a direction and attributing it to chance are equally
defensible. It is the same half §10.5 already uses for `R` and the same half
#1711 uses. **Any constant above the null's own scatter and below 1 gives the same verdict
on both populations**, and that is what makes this structural rather than tuned.
`N^(-1/2)` is where the incoherent population CONCENTRATES and not a bound on
it: the null fluctuates around that value, and in the 96x64/6f fixtures the
statistic with the fewest terms realises `K = 0.31` against a floor of `0.09`,
about three times it. So the usable interval opens a few multiples above the
largest floor among the four statistics rather than at it. At the production
geometry the floors are `0.0039` for sharpness and motion, `0.0116` for
blockiness and `0.0516` for the audio, and the three video statistics measure
between `0.0079` and `0.0325` -- an order of magnitude above their own floors and
one and a half below `0.5`. `TheConstantCarriesNoArgument` runs both populations
at `0.4`, `0.5`, `0.7` and `0.9` and asserts the verdicts do not move.

**The Hoeffding number is CONTEXT and is not the argument.** Under a
sign-symmetric null with independent terms,
`P(K > 0.5) <= 2 exp(-0.5^2 N_eff / 2)`, which the tool reports using the
observed magnitudes. Tiles within a frame are not independent, so `N_eff` is
smaller than `N` and the printed probability is optimistic. The gate does not
rest on it. It rests on `K = 1` under a direction and `K` near zero without one.

**The second half is correspondence.** A perturbation of the arithmetic moves
the picture. It does not move the picture in TIME. So:

- **frame correspondence.** For every frame index `k`, arm B's frame `k` must be
  the closest of arm B's frames to arm A's frame `k`, over a window of
  neighbours. The margin is
  `m_k = min_{j != k, |j-k| <= W} d(A_k, B_j) / d(A_k, B_k)` on luma, and the
  check is `m_k > 1` for every `k`. **The constant is 1 and it is not chosen**:
  it is the exact point at which arm B's frame `k` stops being the nearest thing
  in arm B to arm A's frame `k`, which is what "the same moment of the same
  video" means. A dropped, duplicated, frozen or reordered frame moves the argmin
  off the diagonal by a whole index and the margin goes below 1 in full.
- **audio correspondence.** The lag that maximises the cross-correlation of the
  two tracks must be exactly 0. Any desync moves it by the full shift. §10.7
  already computed this sweep by hand and read lag 0; it was never a check.
- **spatial correspondence.** For every frame, the two-dimensional offset that
  minimises `d(A_k, shift(B_k, dx, dy))` over a small window must be exactly
  `(0, 0)`. A perturbation of the arithmetic does not TRANSLATE the picture
  either, and a translation moves the argmin by the full offset. **This is where
  §10.4's calibration lands.** One pixel of global horizontal shift is the
  perturbation §10.4 says "a criterion that admitted it would not be a
  criterion", and it is admitted by every coherence check, because a rigid
  translation changes no quality statistic at all. It fails HERE, structurally,
  at an argmin that is `(0, 1)` instead of `(0, 0)`, rather than by a borrowed
  decibel. The retired V1 to V3 refused it too, and they refused the swap for
  the same reason; this check refuses the shift and does not refuse the swap,
  which is the discrimination the whole section is for.

**Frame correspondence is what V4 was reaching for, done without a chosen
constant.** V4 divides the arm-to-arm delta by the render's own adjacent-frame
step and admits `0.10`. The derived part of that is the denominator; the `0.10`
is a chosen tenth and it is the part that fails. At `m_k > 1` the same
denominator is used at the only value that carries a meaning, per frame rather
than in aggregate, and against the nearest neighbour rather than the next one.

**A PREDICTION MADE HERE WAS WRONG, AND IT IS CORRECTED RATHER THAN QUIETLY
DROPPED.** The first draft of this section predicted that §10.4's one-sided
dither fixture — `B = A + dither`, one arm literally the other plus noise —
would read `K = 1` and therefore `DIRECTIONAL`, on the argument that every tile
of arm B is noisier than the same tile of arm A. **Measured, it reads
`K = 0.137` on sharpness and passes every coherence check.** The argument was
wrong about the perturbation: a `+/-1` dither is SYMMETRIC, so it raises the
gradient in some tiles and lowers it in others, and the per-tile direction is a
coin even though the two arms are not exchangeable. The prediction, the
measurement that refuted it and the reason are all kept, because a section whose
standing is that it was written before the numbers has to show what the numbers
did to it.

Two things follow that are worth having. §10.4's dither row keeps its role
unchanged and every cell of its gated table is untouched. And the coherence
checks still take a **symmetric** null fixture as well — `A = base + dither_1`,
`B = base + dither_2`, independent draws — because that is the shape the
criterion's null actually names, and a null fixture chosen for the property
being tested is worth having whether or not the one-sided fixture happens to
pass too.

**A `K` BETWEEN THE FLOOR AND 1 IS A PARTIAL DIRECTION, AND THERE THE CONSTANT
IS LOAD-BEARING.** The structural claim is about the two populations the
criterion was built from: a one-directional degradation sits at `K = 1` by
algebra, and an incoherent separation concentrates at `N^(-1/2)`. A measured `K`
that sits in neither place is a real state and it is neither defined away nor
described as if it were at one of the ends. There the verdict does depend on the
constant, and a reader is entitled to see that, so the tool prints `K` beside
its own `null_floor` and beside the fraction of terms in the majority direction,
and the record states where a measured value sits between the two. §11.8 has one
of these, and it is the interesting result of this whole section.

**`K` is magnitude-weighted, so a negligible consistent bias does not fire
it.** The denominator is the sum of the per-term absolute differences, not a
count, so `K` is not a sign test. A statistic that is consistently 0.05% higher
in one arm while varying by 2% per tile gives `K` near `0.03`, and a blur that
takes 30% off every tile gives `K` near 1. The check therefore fires on a
direction that DOMINATES the per-tile variation, which is what a degradation
does and what a bias in the last digit does not.

### 11.4 What is retired, what it is replaced by, and the reason for each

**Nothing is deleted.** Every retired bound keeps its value, keeps its
computation, keeps its line in the printed report and keeps its entry in the
JSON. What it loses is the exit status. A third `judges` class, `identity`, joins
`treatment` and `control`, and the report prints an `IDENTITY` verdict line of
its own so that a reader can never mistake a relocated bound for an absent one.
**On the §10.7 frames that line will keep reading `DIFFERENT` forever.**

| bound | value | disposition | reason |
|---|---|---|---|
| C0 (9 checks) | — | **KEPT, still decides the verdict** | it is the only thing here that is not a difference, it passes, and §10.8's hole is the reason it exists |
| V1 mean \|delta\| | `<= 1.0` | **RELOCATED to `identity`, still FAILS, still printed** | it asks whether the two renders are the same picture to within the artefact's own quantisation step. §10.2 predicted in advance that they would not be, and a bound the correct behaviour is predicted to fail is a measurement rather than a criterion |
| V2 worst PSNR | `>= 40 dB` | **RELOCATED, still FAILS, still printed** | §10.4 records in its own words that this is "the video-coding 'visually lossless' convention. This experiment did not choose it". A convention imported from lossy CODEC transparency judges an encoder against its own source; there is no source here, only two peers |
| V3 worst SSIM | `>= 0.99` | **RELOCATED, still FAILS, still printed** | same class as V2, cited to Wang et al. 2004, and measured between two renders neither of which is the reference the metric assumes |
| V4 temporal ratio | `<= 0.10` | **RELOCATED, still FAILS, still printed; SUPERSEDED by frame correspondence** | the denominator is derived and the numerator `0.10` is not. Its structural version is `m_k > 1` |
| A1 audio PSNR | `>= 40 dB` | **RELOCATED, still FAILS, still printed** | V2's reason in the audio axis |
| A2 audio Pearson r | `>= 0.999` | **RELOCATED, still FAILS, still printed; PARTLY SUPERSEDED by audio correspondence** | §10.4 registered it to catch "a waveform that has drifted in time". The drift half becomes `lag == 0`, which is exact. The `0.999` half is a chosen constant and is retired with the rest |

**No bound is retired because it failed.** Each is retired because of what it
asks. The evidence for that is that the SAME relocation applies unchanged to a
pair that PASSES all six: `flash` against `flash-ctl` is bit-identical, it
passes every one of them, and it passes them in the `identity` class exactly as
it passed them in `treatment`.

### 11.5 What this still does not measure, and what it needs to

Two gaps are named here rather than papered over with a proxy.

**GAP 1: the relative-magnitude criterion the developer asked for is PENDING on
a GPU lease** ([#1853](https://github.com/mudler/vllm.cpp/issues/1853))**.**
The criterion "the swap's divergence is no worse than this
pipeline's own divergence under an arithmetic perturbation of comparable size"
needs a reference perturbation render: one arm rendered on the naive path with a
deliberate, bounded, SMALLER perturbation injected — a `+/-1` bf16 ULP dither on
the attention output is the obvious one, and its size is the quantity §10.2
already derived. Then `D(flash, naive) <= D(dither, naive)` is a relative bound
with no chosen constant at all. **That render needs `dgx:gpu0` and no lease is
authorised for this work, so it is `PENDING`, not skipped, and not substituted.**
The cross-build `9.452407` is NOT that control and is not used as one:
§10.8 already records why, and a gate resting on it would inherit a lineage
difference as if it were a perturbation size.

**GAP 2: absolute quality is NOT GATEABLE here, and no proxy is invented for
it** ([#1854](https://github.com/mudler/vllm.cpp/issues/1854))**.**
Everything in §10 and everything above asks "is it the SAME?". "Is it as
GOOD?" has two forms:

- The RELATIVE form — is either arm systematically worse than the other — IS
  answered, and `K` is the answer. That is what the coherence checks measure.
- The ABSOLUTE form — is this a good render of this prompt — is not answered and
  cannot be by a threshold over these frames. §10.8 already states the reason
  for the neighbouring case: "a check for 'is this a golden retriever shaking
  off water' is a model, not a threshold". Prompt adherence needs a
  vision-language model, and artefact-freedom needs an absolute reference render
  from an oracle that runs this pipeline. Neither exists in this tree.

The tool therefore computes an **absolute quality panel per arm** — the 8-grid
and 32-grid blockiness ratios, the clipped-pixel fraction, and the mean
sharpness — prints it, records it in the JSON, and **does not check any of it**.
It is instrumentation for the next reader, declared as such in the tool's own
output, in the same way `R` is declared. Making it a gate is owed and is filed as
[#1854](https://github.com/mudler/vllm.cpp/issues/1854).

### 11.6 How each outcome will be read, stated before there is one

- **Every correspondence and coherence check passes.** The verdict is
  **`SEPARATED, NOT DEGRADED`**: the two arms differ, the difference has no
  direction and preserves the frame and sample correspondence, and it is
  consistent with a chaotic separation rather than with a defect. This does NOT
  say the arms are the same, and the `IDENTITY` block printed beside it says so
  in the failing numbers. It also does not close GAP 1: without the reference
  perturbation render, "no worse than the pipeline's own sensitivity" remains
  unmeasured, and the verdict claims only what it names.
- **Any coherence check fails.** The verdict is **`DIRECTIONAL`**, and it names
  the statistic and the sign. That is a finding that the flash arm is
  systematically sharper, softer, blockier or quieter than the naive arm, which
  would be a real defect in a shipped default and would owe its own issue. It
  would also mean this section's null is wrong about this pair, and the finding
  outranks the convenience.
- **Any correspondence check fails.** The verdict is **`MISALIGNED`**: the two
  renders no longer depict the same moments. That is a stronger failure than
  `DIRECTIONAL` and it is reported as such.
- **A C0 check fails.** Unchanged from §10.4 and §10.5.

The last two branches exist so that a passing result is a result. A criterion
that passes the case it was written after, and cannot fail anything, is worse
than the one it replaced. §11.7 is the discrimination proof and it is not
optional.

### 11.7 The discrimination proof

Every claimed guarantee is mutated, and the mutation is a DEGRADATION of a real
render rather than an argument about one. The frames of §10.7's arms are on a
share this box can read, and the tool needs no GPU, so each mutation runs on a
scratch copy of the actual production frames as well as on the committed
fixtures:

| mutation | what it injects | the check that must RED |
|---|---|---|
| blur | a 3x3 box blur of arm B's luma | `coherence.sharpness`, `K = 1` |
| block | 8x8 blocks flattened toward their mean | `coherence.blockiness` |
| drop | frame 24 removed and the tail renumbered | `align.frames` |
| desync | the audio of arm B advanced by 480 samples | `align.audio_lag` |
| silence | arm B's audio zeroed | `coherence.audio_rms` |
| shift | one pixel of global horizontal translation | `align.spatial` |

And the control must survive: `flash-ctl` against `flash` is bit-identical and
must PASS every new check, because a gate that cannot pass a repeat of the same
render is measuring the machine.

**THE GATE'S OWN COVERAGE IS MUTATED TOO, AND ONE MUTATION CAME BACK GREEN.**
Proving that the criterion fires on a degraded render is one thing; proving that
a TEST fires when the criterion stops working is another, and the second is what
stops this from rotting. Ten mutations of `scripts/ltx25-render-compare.py`,
each applied to the committed file, each followed by the whole suite, each
restored byte-for-byte and the tree verified clean:

| # | mutation | result |
|---|---|---|
| M1 | `coherence()` returns `k = 0.0` always | RED, 6 of 59 |
| M2 | `K` divided by a further 10, a silent widening | RED, 6 of 59 |
| M3 | `off_diagonal_frames` always 0 | RED, 1 of 59 |
| M4 | `frames_off_origin` always 0 | RED, 4 of 59 |
| M5 | the audio argmax always returns lag 0 | RED, 2 of 59 |
| M6 | `DEFAULT_MAX_COHERENCE` 0.5 to 0.99 | **GREEN, 59 of 59** |
| M7 | `DEFAULT_MAX_MEAN_ABS` 1.0 to 7.0 | RED, 1 of 59 |
| M8 | the `align.spatial` registration deleted | RED, 5 of 59 |
| M9 | `reading` always `SEPARATED, NOT DEGRADED` | RED, 5 of 59 |
| M10 | `blockiness_bands` returns a constant | RED, 1 of 59 |

**M6 is the finding, and the reason it hid is worth naming.** The one constant
this section introduces was the one constant nothing pinned.
`TheConstantCarriesNoArgument` looked like the test that would catch it and could
not, because it passes an explicit `--max-coherence` on every run: it proves the
VERDICTS do not move across the interval and leaves the DEFAULT free. The six
relocated identity thresholds were pinned by
`test_no_identity_threshold_moved` and the new one was not, so the exact
widening §11.2 promises not to do was available on the only threshold this
change adds.

`TheRelocationIsVisible.test_the_coherence_constant_cannot_be_moved_SILENTLY`
now pins it. Red-before: the mutated default fails that test and only that test.
Green-after: 60 of 60, then 62 of 62 after the two repairs below. And the reason
it must be pinned rather than left free is already in §11.3 and is not new here:
where a measured `K` is a PARTIAL direction the constant IS load-bearing,
§11.8's audio result is one at `0.674002`, and a default approaching 1 admits
every partial direction and leaves only the algebraic one.

**AND THE PIN ITSELF WAS WIDENABLE, WHICH A FRESH REVIEW FOUND BY MUTATING IT.**
`test_no_identity_threshold_moved` matched its six values with `assertIn`, a
SUBSTRING match, so `DEFAULT_MAX_MEAN_ABS = 1.09` and
`DEFAULT_MAX_TEMPORAL_RATIO = 0.109` both left all 59 tests green. Both are
MAXIMUM bounds, so appending a digit widens them: the anti-widening pin admitted
the widening it is named after. The values are now parsed out of the module with
`ast` and compared as numbers by `module_float_constants`, and both mutations
red it.

**A FRESH REVIEW FOUND M6 INDEPENDENTLY, AND RAISED SEVERAL MORE.** The reviewer ran
the same ten mutations without seeing this table and returned the same nine REDs
and the same one GREEN, and it went further than the author had: it confirmed on
the real frames that `--max-coherence 0.99` turns the live §11.8 finding into
`[PASS] coherence.audio_rms: K 0.674002 <= 0.99`, `VERDICT PASS (exit 0)`. So the
one-character edit was not a theoretical hole. Its two further findings are
repaired here rather than argued with:

- **`TheConstantCarriesNoArgument` copied one `audio.wav` into both arms**, so
  `coherence.audio_rms` was identically `K = 0` throughout the class — and audio
  energy is the ONLY statistic that fires on the real data. The class that
  defends the constant never exercised the statistic the constant decides. The
  degraded arm's track is now attenuated 20%, the separated arms get independent
  noise, and `test_the_audio_statistic_is_LIVE_in_this_class` refuses a return to
  one shared wav.
- **The printed report gave no way to see that a `K` rests on one event.** The
  tool now prints `top10%`, the share of the net carried by the largest tenth of
  the terms, under a header saying what it means. §11.8's audio reads
  **`+0.989`** — 98.9% of the net comes from a tenth of the windows, which is the
  single loud passage, in the report itself rather than only in this document. A
  uniform attenuation of a constant-amplitude track reads about `+0.1`, and
  that contrast is asserted. A share outside `[0, 1]` means the net is
  cancellation rather than a direction, which is what an incoherent `K` looks
  like from this angle: the swap's blockiness reads `-1.568` beside a `K` of
  `0.0079`.
- **The HARNESS still told the reader that exit 0 meant every threshold held.**
  `scripts/ltx25-dit-attn-flash-pixel-ab.sh` said so in two places, and after
  this change six thresholds can all FAIL at exit 0. Both are reworded, and both
  now say that a 0 does not mean the renders match and that the IDENTITY line is
  the one to read. This is the same drift
  `ThePhaseICommentDescribesTheToolItCalls` exists for, and that tripwire could
  not catch it: it checks that 0, 1, 2 and 3 are NAMED, not what 0 and 1 MEAN.
- **`frame_correspondence` is not symmetric in A and B**, so a recorded margin
  has to name its arm order. `--a flash --b naive` gives `1.4230` at frame 25
  and the reverse order gives `1.3796` at frame 28. Both clear 1 and the verdict
  does not move; the table in §11.8 is `--a flash`, which is what the harness
  runs, and the docstring now says so.
- **`align.audio_lag` did not say WHICH track was constant.** Two legitimately
  silent tracks and one silenced arm reached the same message. It now names the
  side and records `a_is_constant` and `b_is_constant`.

### 11.8 The measurement, and what the new criterion says that the old one could not

**TAKEN 2026-08-24, on a workstation, with no GPU and no lease.** The tool needs
only the frames, and this box can read the share, so every figure below is
reproducible by anyone who can. The inputs are §10.7's own arms at
`/mnt/nas_share/rc/ltx25-attnflash/pixel-ab/1612-r3/` and the 20260820 baseline
at `/mnt/nas_share/rc/ltx25-fullmodel/out/20260820T223701Z/768x448-49f`. Not one
render was re-taken and not one identity figure moved.

**Three pairs, and the ORDERING is the result.**

| pair | mean \|delta\| RGB | identity bounds | correspondence | video coherence | audio coherence | reading |
|---|---|---|---|---|---|---|
| `flash` vs `flash-ctl` (the control) | 0 | all pass | margin `inf`, offset `(0,0)`, lag 0 | `K = 0` on all three | `K = 0` | **`BIT_IDENTICAL`**, exit 0 |
| `baseline-20260820` vs `naive` (two builds, ONE path) | **9.452407** | **6 of 6 FAIL** | margin 1.1445, offset `(0,0)`, lag 0 | 0.041394 / 0.012910 / 0.050994 | 0.312163 | **`SEPARATED, NOT DEGRADED`**, exit 0 |
| `flash` vs `naive` (ONE build, the swap) | **6.414156** | **6 of 6 FAIL** | margin 1.4230, offset `(0,0)`, lag 0 | 0.032512 / 0.007914 / 0.031635 | **0.674002** | **`DIRECTIONAL`**, exit 1 |

**Read the second and third rows against each other, because that is the whole
argument.** The cross-build pair is the LARGER divergence on every identity
axis, and the new criterion passes it. The swap is the smaller one, and the new
criterion fails it. **A widened tolerance cannot produce that ordering**, because
a tolerance is monotone in the size of the difference and these two rows are
ordered the other way. The old criterion returned `FAIL` on both, which is one
answer to two different questions.

**AND THE INVERSION HAS ITS OWN INTERVAL, which a fresh review computed and
which belongs here rather than in a reviewer's report.** The two pairs differ on
exactly one axis, audio energy, at `0.312163` and `0.674002`. So at any constant
at or below `0.312163` BOTH pairs fail, at any constant above `0.674002` BOTH
pass, and **the inversion exists only for a constant in `(0.312163, 0.674002]`**.
`0.5` sits near the middle of it. This does not weaken the argument that no
monotone tolerance can order these two pairs this way, because no constant makes
the `9.452407` pair fail while the `6.414156` pair passes. It does mean that
"the constant carries no argument" holds for the two POPULATIONS the criterion
was built from and NOT for the axis that decides this particular pair, which is
the same partial-direction caveat §11.3 already states, applied to the headline
result. It is written out here so that nobody has to rediscover it.

**The control is the strongest case the design can produce, and it holds.**
`flash-ctl` against `flash` is bit-identical, every structural check passes,
`K = 0` on all four statistics because every term is equal, and the frame margin
is infinite because the diagonal difference is zero. A criterion that could not
pass a repeat of the same render would be measuring the machine.

**The video is directionless, and that is a positive finding rather than an
absence.** Each video statistic sits an order of magnitude above its own
incoherent floor and two to three orders below a direction, and the majority
sign is a coin in every one:

| statistic | terms | floor `N^-1/2` | measured `K` | majority sign |
|---|---|---|---|---|
| sharpness | 65856 | 0.0039 | 0.032512 | 0.502 |
| blockiness | 7448 | 0.0116 | 0.007914 | 0.501 |
| motion energy | 64512 | 0.0039 | 0.031635 | 0.495 |

Neither arm is sharper, blockier or more mobile than the other. The picture is a
separated trajectory, which is what §10.2's error analysis predicted and what no
identity bound could confirm.

**The audio is not, and that is [#1855](https://github.com/mudler/vllm.cpp/issues/1855).**
`K = 0.674002` over 376 windows against a floor of 0.0516. The RMS ratio
`flash / naive` is **0.962289**, which is the 3.3% §10.7 already printed and
never checked, and it is concentrated: the track is 2.01 s and near-silent
outside one passage, and the whole effect is a **4.0% amplitude loss in windows
125 to 249** (2471.7 against 2573.2) with the two near-silent thirds at 1.0116
and 0.9935. The sign is near even, 182 windows quieter and 194 louder, and the
losses are **5.1x the gains in magnitude**. On this axis the swap is twice as
coherent and nearly three times as large as the cross-build pair's 0.312163.

**Two limits on that audio result, stated here rather than in the issue alone,
and now VISIBLE IN THE TOOL'S OWN OUTPUT.** `N = 376` windows is not 376
independent observations, because the track has ONE loud event, so the audio
verdict rests on a single acoustic passage while each video verdict rests on
tens of thousands of tiles across 49 frames. The report says so without the
reader opening this file: `top10%` on the audio row reads **`+0.989`**, so 98.9%
of the net comes from a tenth of the windows. And `K = 0.674` is a PARTIAL
direction: it sits between the floor and 1, so the `0.5` is load-bearing there
and any constant above 0.674 would not fire, which is why `0.5` is pinned to the
byte in §11.7. Both facts are in the record because the verdict is weaker than
the video verdicts and a reader has to be able to see that.

**THE ARMS ARE NOT THE SHIPPED DEFAULT.** These are the `flash` rung of #1549.
[#1551](https://github.com/mudler/vllm.cpp/issues/1551) made the knob three-way
and its unset default is the **FA-2** arm, which has never been rendered at
production geometry. §10.7's own `## Now` already says a pixel A/B of the FA-2
arm is owed. Nothing in this section is a verdict on what `main` builds today.

**The discrimination proof ran on these same frames, not only on fixtures.**
Each mutation is a scratch copy of the real arms, and each reds the check §11.7
names. Arm A is the untouched `flash` render; arm B is a degraded copy of
`naive`, so the swap's own `coherence.audio_rms` failure rides along in every
row and is not what is being demonstrated:

| mutation | reading | the check it was written to RED |
|---|---|---|
| 3x3 box blur | `DIRECTIONAL` | `coherence.sharpness` **K = 0.934838**, and motion 0.580630 |
| 8x8 blocks flattened halfway | `DIRECTIONAL` | `coherence.blockiness` **K = 1.000000**, exactly |
| a one-frame offset, invisible to C0 | `MISALIGNED` | `align.frames`, 47 of 48 frames off the diagonal, worst margin **0.4824** |
| audio advanced 480 samples | `MISALIGNED` | `align.audio_lag`, argmax at **-480**, `r` 0.9348 there against 0.1398 at zero |
| audio zeroed | `MISALIGNED` | `coherence.audio_rms` **K = 1.000000**, exactly |
| one pixel of global horizontal shift | `MISALIGNED` | `align.spatial`, **49 of 49** frames match better at `(0, 1)` |

The frame-offset row is built so that C0 is blind to it: every frame stays
distinct and every pair keeps moving, so `align.frames` is what refuses it and
not the content checks. Two of the six sit at `K = 1.000000` exactly, which is
the algebraic identity §11.3 rests on, observed rather than argued.

**And the constant is shown to carry no argument.**
`TheConstantCarriesNoArgument` runs the degraded and the separated populations at
`0.4`, `0.5`, `0.7` and `0.9` and asserts that both verdicts are unchanged at
every one. The interval is open at the bottom because the null fluctuates around
`N^(-1/2)` and the statistic with the fewest terms sets the usable floor.

### 11.9 The audio direction is carried by ONE channel, and the cross-build pair moves the same one

**Measured on the 1612-r3 frames, with no GPU and no new render**, after #1855
was filed. `scripts/ltx25-render-compare.py` reduces the audio to the MONO MEAN
of the two channels before it windows the track (`audio_rms_terms`), so §11.8
reports one number where the render has two. Splitting it changes what the
finding says.

**The instrument is verified against §11.8 before anything new is read off it.**
On the mono term, at the gate's own 256-sample window, this probe reproduces
`N = 376`, `K = 0.674002`, RMS ratio `0.962289`, `top10% = +0.989` and the three
thirds `1.0116 / 0.9605 / 0.9935` exactly. A probe that could not reproduce the
published numbers could not be trusted with a new one.

| pair | term | `N` | `K` | RMS ratio | `top10%` |
|---|---|---:|---:|---:|---:|
| `flash` vs `naive` | mono mean | 376 | **0.674002** | 0.962289 | +0.989 |
| `flash` vs `naive` | **channel 0** | 376 | **0.756589** | **0.945986** | +1.025 |
| `flash` vs `naive` | **channel 1** | 376 | **0.426780** | **0.978776** | +0.930 |
| `baseline-20260820` vs `naive` | mono mean | 376 | 0.312163 | 0.986061 | +0.923 |
| `baseline-20260820` vs `naive` | **channel 0** | 376 | **0.382499** | **0.980294** | +0.945 |
| `baseline-20260820` vs `naive` | **channel 1** | 376 | **0.047679** | **0.997936** | +0.095 |

The incoherent floor is `N^-1/2 = 0.0516` on every row.

**Two facts follow, and the second one is the load-bearing one.**

**Channel 1 alone does not fire the criterion.** At `K = 0.426780` it sits below
§11.3's `0.5` and reads incoherent. The mono `0.674002` that #1855 reports is
carried by channel 0, which reads `0.756589` and loses 5.4% of its amplitude
against channel 1's 2.1%. So the swap does not take 4% off "the audio". It takes
5.4% off one channel and 2.1% off the other, and the criterion fires on one of
them.

**The cross-build pair moves the SAME channel, in the same direction, and
channel 1 sits BELOW its own floor.** `baseline-20260820` is an ancestor build on
the naive path, and its channel 1 reads `K = 0.047679` against a floor of
`0.0516`. Its channel 0 reads `0.382499`, eight times higher.

A first draft of that sentence called `0.047679` "the most incoherent number this
lane has produced on a pair that is not bit-identical", and a fresh review
falsified it from §11.8's own table: `blockiness` on `flash` against `naive`
reads `K = 0.007914` against a floor of `0.0116`, which is lower in absolute `K`
and lower against its own floor (0.68 of it, against 0.92 here). The superlative
was wrong on either reading and the argument never needed it. It is corrected
rather than deleted, because a claim this section made and could not support is
the thing the next reader has to be able to see.

**A THIRD PAIR, WITH NO `naive` ARM IN IT, KEEPS THE ORDERING.** The two rows
above share `naive` as their reference, so a channel-0 property of that ONE
render would produce the ordering without any kernel doing anything to channel 0.
`flash` against `baseline-20260820` removes that arm from the comparison
entirely, and it reads the same way:

| pair | term | `K` | RMS ratio |
|---|---|---:|---:|
| `flash` vs `baseline-20260820` | mono mean | **0.453425** | 0.975892 |
| `flash` vs `baseline-20260820` | channel 0 | **0.570215** | 0.965002 |
| `flash` vs `baseline-20260820` | channel 1 | **0.354155** | 0.980800 |

**The RMS column of that table was WRONG in its first draft, and a fresh review
caught it.** It read `0.975901 / 0.964996 / 0.980792`. Those three values were
never measured: the three `K` values beside them came from the committed tool and
the ratios did not, and nothing in the session that wrote them produced them. The
measured values are the ones above, computed from the tool's own 376 window terms
at full precision, where `sum/sum` and `mean/mean` agree to nine places. The
error is recorded rather than quietly corrected, because a number that appears in
a table is read as measured and these three were not.

Channel 0 exceeds channel 1 on all three pairs, and on this one the two sit on
OPPOSITE sides of the criterion's `0.5` while the mono term reads `0.453425` and
would not fire at all. So the dilution is not a curiosity of the arithmetic: on
real frames it is the difference between a statistic that fires and one that
does not.

**What this still does NOT establish.** Three pairs share the `flash` arm or the
`naive` arm, and every one of them is a difference between two renders of one
prompt. Whether channel 0 of THIS render is simply the more sensitive channel,
in any arm and under any perturbation, needs a render this lane has not taken.
What is established is that the audio direction #1855 found is not a property of
the track, it is a property of one channel of it.

**The FA-2 render of §12 is the third point this needs.** If `fa2` against
`naive` also loses channel 0 and leaves channel 1 near the floor, the asymmetry
belongs to the pipeline rather than to the swap of #1549, and #1855's
attribution changes. If `fa2` moves neither channel, the swap owns it.


## 12. The pixel A/B has never run on the arm that ships (#1853, #1855)

### 12.1 The hole, in one sentence

Every pixel figure in §10.7 and §11.8 is about `vt::AttentionDenseFlash`. The
engine does not call it. [#1551](https://github.com/mudler/vllm.cpp/issues/1551)
made `VLLM_LTX2_DIT_FLASH_ATTN` three-way and moved the unset default to
`vt::AttentionDenseFa2`, so `ltx2_device.cpp:536-538` selects FA-2 for every
render that does not set the variable. `include/vllm.h`, the loader and the
`ltx2-gen` command line set nothing. The rung that serves has therefore never
been rendered at production geometry, and #1855 states that in its own words:
"NOT measured on the SHIPPED default".

This section owns the render that closes the hole. It is written before the
lease runs, for the reason §10.5 and §11.6 give: a reading rule invented after
the numbers arrive is not a rule.

### 12.2 The ladder, and why the control repeats FA-2

`scripts/ltx25-dit-attn-flash-pixel-ab.sh` renders four arms from one binary,
in this order:

| # | arm | knob | op | why it is in the ladder |
|---|---|---|---|---|
| 1 | `fa2` | unset | 22 | the shipped default, and arm A of the verdict |
| 2 | `naive` | `0` | 18 | `vt::Attention`, the rung #1549 replaced |
| 3 | `fa2-ctl` | unset | 22 | `fa2` again, same binary and seed: the noise floor |
| 4 | `flash` | `flash` | 21 | the #1549 rung, so §11.8's pair is re-taken here |

**The control repeats FA-2 and not `flash`.** §10.6 records what an inverted
control does: a control that repeats the arm the verdict is not about measures
the treatment a second time and reads about the treatment's size, whichever
kernel ran. `--control-of a` is passed explicitly for the same reason.

**The order is chosen against the failure mode of this box, not for tidiness.**
`fa2` is first because it is the cheapest arm and the one the lease was taken
for. `naive` is second, while the box is known good, because it is about six
times the wall clock of a fast arm and its loss leaves no pair at all.
`fa2-ctl` is third and completes the primary triple. `flash` is last because it
is the only render this ladder can lose without losing a verdict.

**Each arm proves its own op from its own log.** The routing check now counts
`op=18`, `op=21` and `op=22`, and each arm must resolve its own op and neither
of the other two.
[#1794](https://github.com/mudler/vllm.cpp/issues/1794) was two harnesses whose
arm labelled `flash` ran FA-2 for months, and a two-sided proof over 18 and 21
cannot see that: an `fa2` arm that fell through to flash shows `op18=0` and
`op21=1`, which the previous naive-or-other `case` read as `ROUTING_OK`.

**The empty knob means unset and is spelled `unset`.**
[#1751](https://github.com/mudler/vllm.cpp/issues/1751) made
`VLLM_LTX2_DIT_FLASH_ATTN=""` a refusal by name, so an `export` of the empty
string aborts the FA-2 arm at its first DiT forward, one hour into a four-hour
lease.

### 12.3 Three comparisons, and exactly one verdict

| pair | control | what it answers | status |
|---|---|---|---|
| `fa2` vs `naive` | `fa2-ctl` | what the shipped default renders against the correctness rung | the run's exit status |
| `fa2` vs `flash` | `fa2-ctl` | what #1551 changed, in pixels | recorded, not the verdict |
| `flash` vs `naive` | none | §11.8's pair, re-taken on a new binary | recorded, not the verdict |

`PIXEL_RC` is assigned once. Each secondary pair records its own status under
its own name. Three statuses reported as one number would let a reader quote
whichever number agreed with them.

**The `flash` vs `naive` pair has no control on this ladder, and the harness
gives it none.** `fa2-ctl` repeats `fa2`, so offering it to that pair is the
inversion §10.6 exists to prevent. The `fa2-ctl` block measures a noise floor
for the FA-2 arm on this binary. Extending that reading to the `flash` arm is
an inference, and this row does not make it. A `flash-ctl` render would close
it and is the fifth arm the ladder does not spend the lease on.

### 12.4 How each outcome will be read, before there is one

The criterion is §11.3's, unchanged and with no constant moved. The FA-2 pair
gets the reading rules of §11.6, and two further readings are available only
because three pairs run on one binary:

- **`fa2` vs `naive` corresponds and is incoherent.** The shipped default
  separates the trajectory and does not degrade it, which is what §11.8 already
  concluded for `flash`. The lane's open questions then belong to `flash`
  alone.
- **`fa2` vs `naive` shows a direction on a statistic.** Name the statistic and
  the arm the direction favours. A direction on the shipped default is a
  finding about `main` under §11.6, and §9 forbids repairing it by moving a
  number.
- **The `flash` vs `naive` audio direction reproduces.** #1855's `K = 0.674002`
  and RMS ratio 0.962289 come from one binary and one lease. A second binary,
  built from a later `main`, that reads a comparable audio `K` raises the
  finding from one observation to two.
- **The `flash` vs `naive` audio direction does not reproduce.** Then #1855
  measured that binary rather than that kernel, and the issue is corrected
  rather than closed.
- **`fa2` and `flash` differ from `naive` in the same direction on audio.** The
  direction belongs to the reassociation class and not to one kernel.
- **The control is not bit-identical.** Then the noise floor is not zero on
  this binary, every delta is read against it, and no pair is attributed to a
  kernel until it exceeds it.

### 12.5b The run's identity was a 75 KB launcher, and the ladder found it live

**[#1881](https://github.com/mudler/vllm.cpp/issues/1881), observed while this
ladder was building and repaired in the same flow.** The harness recorded
`binary_sha256` as the sha256 of `ltx2-gen`, which is 75,344 bytes of `main()`.
Every op this A/B measures is in `libvllm.so.0.0.3`, which was hashed nowhere.

| run | `source_sha` | recorded `binary_sha256` | `libvllm.so.0.0.3` | size |
|---|---|---|---|---|
| `1612-r3` (§10.7) | `3e2961ef0` | `834cec557c16cf77…` | not recorded | 85,703,328 |
| `1853-fa2-r1` (§12.6) | `62cbae10d` | **`834cec557c16cf77…`** | `f046e75dcede2586…` | **92,075,952** |

Both built in a lease with `BUILD_RC=0`, from source trees a whole release window
apart, and the recorded identity is byte-identical. The launcher's own translation
unit did not change, so its output is reproducible **by construction**, and the
one artefact whose hash was stable was the one artefact containing none of the
code under measurement. The libraries differ by 6,372,624 bytes.

So §10.7's sentence "the binary `834cec55…`" does not pin the code that produced
its figures, and a later reader comparing against it read the same string and
concluded the same code ran. The harness now hashes and records both, in
`PROVENANCE` and in every arm's own `render.log` header, and
`test_ltx25_pixel_ab_harness.py` holds all three sites. **The 1612-r3
measurement itself is unaffected**: its four arms ran from one build in one lease
and each proved its own op from its own log. What its record could not do is tell
its build from a later one.

`ltx25-dit-attn-flash-ab.sh` and `ltx25-dit-attn-fa2-hd128-ab.sh` carry the same
idiom and are owed the same repair. They are named in the issue and not touched
here.

### 12.5 What this still does not measure

- **#1853's arithmetic-perturbation control is NOT taken here.** That issue
  asks for a `dither` arm: the naive path with a bounded `+/-1` bf16 ULP
  perturbation at the DiT attention output, at the `8.6e-05` to `3.7e-04`
  per-element flip rate §10.2 derives. It needs a debug knob in
  `ltx2_device.cpp`, a CUDA kernel and a second `naive`-class render, and no
  part of it is in this ladder. It stays `PENDING` under `## Owed`.
- **Absolute quality.** §10.8 and #1854 stand. Nothing here asks whether either
  render is good.
- **Head_dim 128 tensor cores.** #1578's rung is not in this ladder.

### 12.6 The measurement: the shipped arm renders, and it reads DIRECTIONAL on the audio

**The run.** `rc` job `4dcdd916-750d-4e6b-8fd8-186e45199c23` on `dgx:gpu0`,
`RUN_ID=1853-fa2-r1`, lease opened 2026-08-24T19:30:50Z. Source `62cbae10d`,
harness sha256 `9fcf1c85925ec464…`, checkpoints staged to `/root/ckpt`,
`768x448/49f`, 2352 video tokens, seed `20260820`, prompt sha256
`451a8860796760278bd7c08e15108d6639fe6c83b523dbc8573d252165789ce5`. Four arms,
one build, one lease.

**THE RUN'S OWN RECORD CANNOT NAME ITS LIBRARY, AND THAT IS #1881 ARRIVING ONE
COMMIT TOO LATE.** This lease was submitted from `62cbae10d`, which is TWO commits before
`f8420a89c` (`6e7bcb3f2` intervenes and touches only
`tests/scripts/test_ltx25_pixel_ab_harness.py`), so the harness that rendered
these frames is the one WITHOUT the `LIBSHA` repair. Its `PROVENANCE` records `binary_sha256=834cec557c16cf77…` —
the 75,344-byte launcher, byte-identical to the value `1612-r3` recorded from a
tree a release window earlier — and it records no `library_sha256` at all. The
`libvllm.so.0.0.3` that carries every op measured here is
`f046e75dcede2586…`, 92,075,952 bytes, and that value was read BY HAND in the
lease rather than written by the run. A later reader cannot recover it from
these artefacts. The repair is committed and the NEXT run records it; this one
is evidence of the hole it closes. The diff from `62cbae10d` to `f8420a89c`
touches only provenance recording — no arm, no route, no threshold, no
criterion — so the measurement below stands on its own bytes.

**Every arm proved its own op, and the counts were re-derived from each arm's
raw `render.log` rather than read off its `ARM` summary.** This matters here
more than usual: the fresh review of this pull request established that
`arm_report`'s routing predicate was NOT executed by any test at the time this
run was submitted (repaired in `363651bce`, after the lease opened). The
announced verdict is therefore reported beside an independent recount, and the
two agree on every arm.

| arm | knob | announced verdict | announced counts | independent recount | median/forward |
|---|---|---|---|---|---|
| `fa2` | unset | `ROUTING_OK=fa2` | op18=0 op21=0 **op22=1** | op18=0 op21=0 **op22=1** | **2.223 s** (n=119) |
| `naive` | `0` | `ROUTING_OK=naive` | **op18=1** op21=0 op22=0 | **op18=1** op21=0 op22=0 | **45.512 s** (n=119) |
| `fa2-ctl` | unset | `ROUTING_OK=fa2-ctl` | op18=0 op21=0 **op22=1** | op18=0 op21=0 **op22=1** | **2.256 s** (n=119) |
| `flash` | `flash` | `ROUTING_OK=flash` | op18=0 **op21=1** op22=0 | op18=0 **op21=1** op22=0 | **6.360 s** (n=119) |

`op=19` (`kAttentionCross`, `vt-cross-blocked`) is selected once in every arm and
is asserted on in none, which is what §12.2 says it is.

**THE CONTROL IS A ZERO, EXACTLY.** `fa2-ctl` repeats `fa2`'s knob and the two
renders are **bit-identical: 49 of 49 frames byte-equal, `audio.wav` byte-equal,
max |delta| 0, PSNR inf, SSIM 1.000000**, and `control/treatment = 0.000000` on
luma against a treatment effect of `8.952578`, `0.000000` on RGB. So the noise
floor of the FA-2 arm on this binary is not small, it is absent, and no number
below can be attributed to run-to-run variation. §12.4's "the control is not
bit-identical" branch does not fire.

**The verdict pair: `fa2` vs `naive`, controlled by `fa2-ctl`.**

Correspondence passes in full:

| check | value | bound |
|---|---|---|
| `align.frames` | worst margin **1.1928** at frame 29 (nearest other 28), 0 frames off-diagonal | `> 1` |
| `align.spatial` | **0 of 49** frames match better than `(0, 0)`; worst `(0, 0)` | argmin at origin |
| `align.audio_lag` | best lag **0** samples, `r = 0.919182` there and at 0 | `== 0` |

Coherence does not:

| statistic | `K` | `N` | floor `N^-1/2` | direction | verdict |
|---|---:|---:|---:|---|---|
| sharpness | 0.020738 | 65856 | 0.0039 | b>a | incoherent |
| blockiness | 0.040532 | 7448 | 0.0116 | a>b | incoherent |
| motion | 0.061146 | 64512 | 0.0039 | b>a | incoherent |
| **audio_rms** | **0.511574** | 376 | 0.0516 | **b>a** | **DIRECTIONAL** |

`READING DIRECTIONAL (section 11.6)`, `VERDICT FAIL (exit 1)`. The picture is
directionless on all three video statistics, an order of magnitude above their
own floors and more than an order below `0.5`, and the frames correspond. The
audio means are `865.774` for `fa2` against `892.84` for `naive`: **the shipped
arm is 3.03% quieter**, and `top10% = +0.952` says the loss is concentrated in
the one loud passage exactly as §11.8 found for `flash`.

**THE 0.5 CONSTANT IS CARRYING THIS VERDICT BY 2.3%, AND THAT IS STATED RATHER
THAN ROUNDED AWAY.** `0.511574` against `0.5` is the narrowest margin this lane
has produced.

Be exact about what this does and does not say about §11.3. That section's
structural argument is that a coherent difference sits at `K = 1` ALGEBRAICALLY
and an incoherent one concentrates at `N^-1/2`, so any constant between them
separates THOSE TWO POPULATIONS and carries no argument. That reasoning is
untouched. What it does not cover is a PARTIAL direction, which is a third case
sitting between the two, and `0.511574` against a floor of `0.0516` and a
ceiling of 1 is squarely one: 9.9 times its null and half a full direction.
#1855 already recorded this for `K = 0.674002`; at `0.511574` the pair sits so
close to the constant that any constant in `[0.512, 1)` would read it incoherent
and any constant in `(0.0516, 0.511]` would read it directional. For THIS pair,
on the checked mono term, the number `0.5` is therefore a chosen bound and not a
structural one, and §9 forbids repairing that by moving it.

The per-channel reading below is what removes the ambiguity, because it is not
close to any constant.

**§11.9's PRE-REGISTERED PREDICTION FIRES, AND #1855's ATTRIBUTION CHANGES.**
§11.9 wrote, before this render existed: *"If `fa2` against `naive` also loses
channel 0 and leaves channel 1 near the floor, the asymmetry belongs to the
pipeline rather than to the swap of #1549, and #1855's attribution changes."*

| pair | term | `K` | RMS ratio | against floor 0.0516 |
|---|---|---:|---:|---|
| `fa2` vs `naive` | mono mean (CHECKED) | **0.511574** | 0.969686 | 9.9x |
| `fa2` vs `naive` | **channel 0** | **0.705886** | **0.936032** | 13.7x |
| `fa2` vs `naive` | **channel 1** | **0.060690** | **0.997125** | **1.18x** |

Channel 1 reads `0.060690` against a floor of `0.0516`. That is 1.18 times its
own incoherent null and is not distinguishable from it: **the FA-2 arm does not
touch channel 1 at all.** Channel 0 reads `0.705886` and loses 6.4% of its
amplitude. The same channel, in the same direction, as `flash` (§11.9:
`0.756589` / `0.426780`).

So the audio direction is **not a property of the `vt::AttentionDenseFlash` swap
of #1549**. Two different attention kernels — one of which is the shipped
default and was never in #1855's evidence — lose the same channel of the same
passage against the naive path. #1855's attribution moves from "the swap takes
4% off the audio" to "the reassociation class takes 6.4% off channel 0", and the
issue is CORRECTED rather than closed.

**What they share is a CLASS and not an order, and a first draft of this
paragraph got that wrong.** It said the two arms share "a reassociated f32
online-softmax accumulation order". `include/vt/ops.h` says they do not.
`AttentionDenseFast` is "NOT bit-identical to Attention (different head_dim
partial-sum grouping)" (`ops.h:3315-3316`); `AttentionDenseFlash` keeps that
arithmetic and its order UNCHANGED and is bit-identical to `Fast`
(`ops.h:3328-3329`); and `AttentionDenseFa2` is "NOT bit-identical to
AttentionDenseFast/Flash" because `mma.sync` reassociates the QK^T and PV
reductions (`ops.h:3381-3382`). So `flash` and `fa2` reach `naive` by TWO
DIFFERENT reassociations, and what the measurement shows is that both
departures from `naive`'s ordering move the same channel the same way. That is
a weaker and truer statement than one shared order, and it is what the evidence
supports. Why either reassociation costs one channel of one acoustic passage
while leaving the other at its floor is unexplained and is owed under #1886.

**The mono statistic hides this, and here it nearly hid the verdict too.** The
checked term is `K` of the channel-AVERAGED signal, not the average of the two
channels' `K` — `audio_rms_terms` means the channels before it windows the
track, so the cancellation happens in the samples and not in the statistic.
Where the two channels read `0.705886` and `0.060690`, it reported `0.511574`,
which is neither of them and is not their mean either, and which cleared the
criterion by 2.3%. Had channel 0 lost 5% instead of 6.4%, the mono term would
have read below `0.5` and this run would have published `SEPARATED, NOT
DEGRADED` over a one-channel direction of `K > 0.7`. That is the concrete cost
of averaging before windowing, on real frames, and it is why the panel is
printed. Widening the CHECKED set to the per-channel terms is a criterion change
and still owes its own row.

**THE THREE PAIRS, AND EXACTLY ONE OF THEM IS THE VERDICT (§12.3).** `PIXEL_RC`
is assigned once, from the first row. Every number below was produced twice: by
the run inside the lease, and by the operator rerunning
`scripts/ltx25-render-compare.py` on the collected frames afterwards. The two
agree to six decimals on every statistic, so the figures are the tool's and not
one invocation's.

| pair | control | reading | exit | audio `K` | worst frame margin |
|---|---|---|---|---:|---:|
| **`fa2` vs `naive`** | `fa2-ctl` | **DIRECTIONAL** | **1** | **0.511574** | 1.1928 @ 29 |
| `fa2` vs `flash` | `fa2-ctl` | SEPARATED, NOT DEGRADED | 0 | 0.177718 | 1.2344 @ 30 |
| `flash` vs `naive` | none (§12.3) | DIRECTIONAL | 1 | 0.674002 | 1.4230 @ 25 |

**`fa2` vs `flash` shows no direction anywhere.** sharpness `K` **0.001954**,
blockiness **0.037599**, motion **0.094879**, audio **0.177718**, every
correspondence check passing. The two fast kernels agree with each other. So the
direction found in rows 1 and 3 is not between `fa2` and `flash`; it is between
both of them and `naive`.

**THE THIRD ROW IS NOT A SECOND OBSERVATION, AND SAYING SO WAS ONE `cmp` AWAY
FROM BEING WRONG.** §12.4 registered the branch *"a second binary, built from a
later `main`, that reads a comparable audio `K` raises the finding from one
observation to two."* This run reads not a comparable `K` but the **identical**
one: `0.674002`, `ch0 0.756589`, `ch1 0.426780`, sharpness `0.032512`,
blockiness `0.007914`, motion `0.031635`, margin `1.4230` at frame 25 — every
figure of §11.8 to six decimals. That is because the renders are **byte-identical
to `1612-r3`**: 49/49 frames and `audio.wav` equal for BOTH the `flash` and the
`naive` arm, across `3e2961ef0` -> `62cbae10d` and a `libvllm.so.0.0.3` that
differs by 6,372,624 bytes. So #1855's audio direction is **reproduced, not
replicated**: the count of independent observations is still ONE, and what this
run adds is that the LTX-2.5 render path is deterministic across that library
delta. §12.4's branch is answered in the letter and not in the spirit, and the
spirit is what a second observation was for.

That determinism is also the reason #1881's launcher hash did no damage HERE and
is still not evidence: the outputs happened to be identical, so the stable
`binary_sha256` was accidentally honest. A library delta that HAD moved the
picture would have been equally invisible to it.

**THE CHECKED STATISTIC DILUTES THE TWO ARMS BY DIFFERENT AMOUNTS, AND IT
DILUTES THE SHIPPED ONE MOST.** This needs all three pairs to see.

| arm vs `naive` | ch0 RMS ratio | ch1 RMS ratio | ch0 `K` | ch1 `K` | **mono `K` (CHECKED)** | mono / ch0 |
|---|---:|---:|---:|---:|---:|---:|
| `fa2` (SHIPPED) | **0.936032** (-6.40%) | 0.997125 (-0.29%) | **0.705886** | 0.060690 | **0.511574** | **0.725** |
| `flash` | 0.945986 (-5.40%) | 0.978776 (-2.12%) | 0.756589 | 0.426780 | **0.674002** | 0.891 |

**The mono term retains 89.1% of `flash`'s channel-0 direction and only 72.5% of
`fa2`'s.** The cause is arithmetic, not noise: `flash` moves both channels the
same way, so averaging largely preserves its direction, while `fa2` leaves
channel 1 at 0.29% and the average dilutes its channel-0 loss against an
effectively unmoved channel. The dilution is therefore ARM-DEPENDENT, and it
falls hardest on the arm that ships — which is what pushed `fa2` to within 2.3%
of the constant while `flash` sits 35% above it.

**A first draft of this paragraph claimed more than that and was wrong.** It
said the checked statistic "orders the two arms the wrong way round on the one
quantity the direction is carried by", on the grounds that `fa2` loses more
channel-0 amplitude than `flash` (-6.40% against -5.40%) while reading the lower
mono `K`. A fresh review falsified it, and the falsification is simple: on `K`
against `K` there is NO inversion. Channel-0 `K` ranks `fa2` below `flash`
(0.705886 against 0.756589) and the mono term ranks them the same way (0.511574
against 0.674002). The apparent reversal only appears when an AMPLITUDE measure
is set against a COHERENCE measure, and §11.3 says in its own words that the
split "is not in the SIZE of the difference" but in its direction. Comparing the
two is the category error §11.3 exists to prevent, and the claim is withdrawn
rather than quietly softened, because a superlative this section could not
support is what the next reader has to be able to see.

This is what §11.9 could only assert from one pair and can now show from three,
and it is the concrete argument that the CHECKED set is wrong. Changing it is
still a criterion change that owes its own row, its own red-before evidence and
its own mutation (§9, #1668). It is not made here.


## Owed

- **[#1853](https://github.com/mudler/vllm.cpp/issues/1853): the
  arithmetic-perturbation reference render, `PENDING` on a `dgx:gpu0` lease.**
  §11.5 GAP 1 designs it and §11 deliberately ships without it. One further arm
  on the naive path with a bounded `+/-1` bf16 ULP dither injected at the
  attention output turns "no worse than this pipeline's own sensitivity" into a
  bound with no chosen constant. No lease was authorised for #1743, the
  cross-build `9.452407` is not a substitute for it (§10.8), and none is used.
- **[#1855](https://github.com/mudler/vllm.cpp/issues/1855): the swap takes 4%
  off the audio in the only passage that has any.** Found by §11's own criterion
  on §10.7's existing frames, with no GPU. The video is directionless on all
  three statistics and the audio reads `K = 0.674002`. NOT FIXED IN FLOW: it is
  a finding about a change already on `main`, the arms measured are the `flash`
  rung of #1549 rather than today's FA-2 default, and attributing a 4% amplitude
  loss to a reassociated attention sum is its own investigation.
- **[#1855](https://github.com/mudler/vllm.cpp/issues/1855): the FA-2 pixel
  render, §12, TAKEN AND READ.** The four-arm ladder ran on `dgx:gpu0` as
  `1853-fa2-r1` and the result is §12.6. The shipped default corresponds and is
  directionless in the picture, and it reads `DIRECTIONAL` on the audio at
  `K = 0.511574` with a bit-identical control. #1855's ATTRIBUTION is corrected
  rather than closed: `fa2` and `flash` lose the same audio channel against
  `naive`, so the direction belongs to the reassociation class and not to the
  #1549 swap. The remaining debt is carried by #1886 below.
- **[#1886](https://github.com/mudler/vllm.cpp/issues/1886): the shipped FA-2
  arm's audio direction is UNATTRIBUTED, and the checked statistic ranks it
  backwards.** §12.6 measures it: ch0 loses 6.40% at `K = 0.705886` while ch1
  sits at 1.18x its own floor, and the mono term that IS checked reports `fa2`
  as milder than `flash` (0.511574 against 0.674002) although `fa2` damages ch0
  MORE (-6.40% against -5.40%). Two things are owed and neither is done here.
  **The mechanism**: why a reassociated f32 online-softmax order costs 6.4% of
  one audio channel in one acoustic passage while leaving the other channel at
  its floor and the picture directionless on all three video statistics.
  **The criterion**: moving the CHECKED set from the mono mean to the per-channel
  terms is a criterion change under §9 and #1668 and owes its own row, its own
  red-before evidence and its own mutation. NOT FIXED IN FLOW.
- **[#1886](https://github.com/mudler/vllm.cpp/issues/1886) (second half): the
  `flash` vs `naive` pair is still ONE observation.** §12.6 establishes that its
  renders are byte-identical to `1612-r3` across a `libvllm` differing by
  6,372,624 bytes, so §12.4's "a second binary raises the finding from one
  observation to two" is answered in the letter and not in the spirit. A genuine
  replication needs a render that is not the same bytes -- a different seed, a
  different prompt, or a geometry this ladder has not run.
- **[#1881](https://github.com/mudler/vllm.cpp/issues/1881): the two SIBLING
  harnesses still record a launcher as their binary identity.** §12.5b repairs
  `ltx25-dit-attn-flash-pixel-ab.sh` in flow. `ltx25-dit-attn-flash-ab.sh` and
  `ltx25-dit-attn-fa2-hd128-ab.sh` compute the same `BINSHA` over `ltx2-gen` and
  hash `libvllm.so.0.0.3` nowhere, so every speed number they have recorded
  carries an identity that two different builds can share. Not repaired here,
  because each is its own lease-only harness with its own extracted-block tests.
- **[#1872](https://github.com/mudler/vllm.cpp/issues/1872): `align.audio_lag`
  is a bare argmax and carries no margin.** Found under #1855 on the frames
  #1612 rendered, with no GPU. `flash` against `baseline-20260820` reads
  `best lag -1` on `r = 0.926353` against `0.926308` at lag 0 -- a correlation
  difference of `4.5e-05` at one sample of 96,480 -- and the run reads
  `MISALIGNED` on it. `align.frames` was written with a `margin > 1` and prints
  it; this check compares an integer argmax over 4001 float candidates and has
  no tie handling. NOT REPAIRED IN FLOW, because adding the margin changes
  checker semantics and owes its own row, spec section, red-before test and
  fresh review. No published verdict moves: the same-binary `flash`/`naive` pair
  reads `best lag 0` and passes.
- **[#1854](https://github.com/mudler/vllm.cpp/issues/1854): absolute render
  quality is not gateable in this tree.** §11.5 GAP 2. The RELATIVE form of "is
  it as good" is answered by the coherence checks and is gated. The ABSOLUTE
  form needs either an oracle that renders LTX-2.5 or a pinned scoring model,
  and this tree has neither. An absolute quality panel is computed, printed and
  explicitly NOT checked, rather than a proxy being invented for it.

- **DISCHARGED 2026-08-23: this row's two harnesses carried
  [#1734](https://github.com/mudler/vllm.cpp/issues/1734)'s memory-watchdog
  idiom, and one of them carried it wrong.** The issue was filed against
  `ltx25-dit-attn-fa2-hd128-ab.sh`, which owns it. Triage found
  `ltx25-dit-attn-flash-ab.sh` writing the identical two-line watch record and
  printing the identical empty `memavail low-water:`, while
  `ltx25-dit-attn-flash-pixel-ab.sh` had the working writer beside the same
  positional `$4` reducer. All three now share one byte-identical
  `# BEGIN memwatch-helpers` block, and
  `tests/scripts/test_ltx25_ab_memwatch.py` holds the two guarantees and the
  sameness. §8.11 of
  [`ltx25-dit-attn-fa2-hd128.md`](ltx25-dit-attn-fa2-hd128.md) is the diagnosis
  and the red-before. It touches no number in §7 or §10: every affected arm
  reports `stopped_by=sample-cap` or a completed render, which is the direct
  evidence that none was stopped by memory pressure.
- **`dgx:gpu0` holds ~110 GiB that belongs to no `/proc/meminfo` category, and
  the controller keeps handing out leases against it.**
  [#1709](https://github.com/mudler/vllm.cpp/issues/1709), measured in §10.7.
  The half this row owns is fixed in flow: the harness's `MemAvailable` start
  gate waits, logs every poll and refuses by name. Two halves are NOT this
  row's and are named rather than folded in — a device-readiness condition in
  `rc`, which would have parked three jobs instead of spending them, and a line
  in `.agents/environment.md`'s DGX profile saying a granted lease does not
  imply a reclaimed box. Owner: this row until the controller half has one.
- **DISCHARGED 2026-08-22: the pixel comparison at production geometry now
  exists, and it FAILED.** §10 is the design, committed before the renders; §10.7
  is the result. All four V and both A checks fail, the control is bit-identical
  so the delta is entirely the kernel's, and §10.5 reads it as **visibly
  different**. What this closes is the *absence* of a comparison, which was the
  gap #1612 named. What it opens is
  [#1743](https://github.com/mudler/vllm.cpp/issues/1743), which owns the
  divergence itself and the question of whether the arm stays the default. The
  reduced-dimension host-vs-device case still bounds only the ARITHMETIC change
  — `8.94e-08` / `4.47e-08` against `2e-5` — and it is now known to be a weak
  predictor of the render: the kernel agrees with its reference to within that
  bound, two orders of magnitude inside the tolerance, and the video still moves
  by 71% of its own motion step. Owner: this row for the record,
  #1743 for the finding.
- **NEITHER ARM IS ESTABLISHED AS THE CORRECT RENDER, and nothing here can
  establish one.** Every figure in §10.7 is a difference between two renders, so
  none of them sees a defect the two share (§10.8). `naive` is the older
  behaviour and not a proven reference, and the 20260820 cross-check is a
  different binary lineage rather than a golden. Closing this needs an absolute
  reference — an upstream LTX-2.5 render of the same prompt, seed and geometry —
  which is a different row and a different oracle. Owner: this row until that row
  exists. Issue: [#1743](https://github.com/mudler/vllm.cpp/issues/1743).
- **The f32 L2 parity arm cannot run at production geometry any more.** With
  §4.3's cap-raise reverted, that arm reaches `AttentionDenseFlash` at head_dim
  128, whose f32 tile is 65,536 B and does not fit the 49,152 B a launch gets
  without an opt-in. It fails LOUD either way — a `cudaGetLastError` throw at
  `cuda_ops.cu:3352` today, a `VT_CHECK` naming the head_dim once
  [#1578](https://github.com/mudler/vllm.cpp/pull/1578) lands — and never
  silently, which is why it is disclosed rather than blocking. Nothing gated
  reaches it: production is bf16, and the f32 arm is a parity reference exercised
  at the fixture's reduced dimensions (§5). Owner: this row, under
  [#1612](https://github.com/mudler/vllm.cpp/issues/1612).
- **`scripts/attention-rung-allowlist.txt` will carry two STALE stems.** Once
  #1578 is on `main`, its checker sees the `// VT-ATTN-NAIVE:` markers this row
  adds and reports `ltx2` and `ltx2_device` as `STALE (not a failure)`. Deleting
  the two stems also edits that checker's pinned-set test, which is #1578's file
  and not this row's, so it is left to whoever runs preflight next — the handoff
  that allowlist's own header describes. Owner: this row until it is deleted.
- **Most of the harness still runs nowhere but a lease.**
  `tests/scripts/test_ltx25_pixel_ab_harness.py` exercises the memory
  precondition and the arm-completeness check, which are extracted verbatim from
  `scripts/ltx25-dit-attn-flash-pixel-ab.sh`. Six things it does not execute:
  the render loop, the routing assertion, the phase [I] call site, the phase [L]
  exit, the phase [F] unit-gate refusal and the signal traps. All six are now
  pinned as TEXT at least in part, by eleven tripwire tests — a count of tests, not
  of surfaces: phase [L] carries three, for its exit wiring, its exit-3 verdict
  and the `*)` arm that catches a status nobody defined, while phase [I] and the
  routing assertion carry two each. Nothing executes any of them, so a rewrite
  that reintroduced any of those defects in different words would pass; the
  render loop is pinned at its knob-selection branch alone, and its watchdog
  poll, memory floor and timeout are pinned by nothing. That is a limit of where the file runs,
  not a gap that another local test can close. This list said "four" and omitted the unit-gate refusal and
  the traps until a fresh review counted them. Owner: this row. Issue:
  [#1612](https://github.com/mudler/vllm.cpp/issues/1612).
- **`scripts/ltx25-render-compare.py` writes `Infinity` into its JSON.** A
  bit-identical pair has zero MSE and infinite PSNR, and `json.dump` spells that
  `Infinity`, which `json.load` reads back and a strict JSON parser refuses. Every
  consumer here is Python, so it is disclosed rather than repaired: replacing it
  with `null` or a sentinel would change what the report means for the one case
  the whole comparison hopes to see. Owner: this row. Issue:
  [#1612](https://github.com/mudler/vllm.cpp/issues/1612).
- **True tensor cores at head_dim 128. HANDED OFF, not still owed here.**
  `vt::AttentionDenseFa2` refused anything but head_dim 64
  (`src/vt/cuda/cuda_flash_attn_fa2.cu:557-560` at this row's base). Reaching
  the vendored FA-2 `mma.sync` path for LTX's head_dim 128 needs an extra
  `run_mha_fwd_<bfloat16_t, 128, false>` instantiation. That was explicitly out
  of scope for this row, and it is the difference between this fix and a
  materially larger one: everything this row shipped is still a scalar
  warp-per-query recurrence. Owner: row `LTX25-DIT-ATTN-FA2-HD128`, spec
  [`ltx25-dit-attn-fa2-hd128.md`](ltx25-dit-attn-fa2-hd128.md). Issue:
  [#1551](https://github.com/mudler/vllm.cpp/issues/1551).
- **The other `vt::Attention` callers.** §3's defect shape is not LTX-specific.
  A sweep of every remaining non-decode `vt::Attention` call site belongs to its
  own row with its own issue. Owner: this row until that row exists. Issue:
  [#1552](https://github.com/mudler/vllm.cpp/issues/1552).

## Outcome

Recorded when the row reaches `DONE`.
