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
are grouped (`include/vt/ops.h:3304-3306`): the naive kernel reduces across a
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
and differ only in association (`include/vt/ops.h:3304-3306`). Two summation
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
  phase [F] unit-gate refusal and the signal traps do not.** Five of those six
  surfaces are pinned as TEXT, by seven tripwire tests — the suite asserts the
  exact call site that shipped inverted, the exact `exit` lines, both arms of
  phase [L]'s `case` that this row wrote (the exit-3 verdict and the `*)`
  fallback that a status nobody defined would otherwise fall through silently),
  the two unit-gate statuses and the four `trap` lines. **Those two counts are
  about different sets**: six is how many things never execute, seven is how
  many tests pin the five of them that are pinned at all, because phase [L]
  carries three tests by itself. They both read "six" for one commit, which
  looked like two records agreeing. A text assertion is a tripwire, not a proof:
  it catches the inversion that happened and would not catch a rewrite that
  reintroduced it in different words. **The render loop is pinned by nothing at
  all.** `dgx:gpu0` under a lease is the only place those lines run, which is why every one of them
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

## Owed

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
  exit, the phase [F] unit-gate refusal and the signal traps. Five of the six are
  pinned as TEXT, by seven tripwire tests — a count of tests, not of surfaces:
  phase [L] carries three, for its exit wiring, its exit-3 verdict and the `*)`
  arm that catches a status nobody defined. Nothing executes any of them, so a
  rewrite that reintroduced any of those defects in different words would pass;
  the render loop is pinned by nothing. That is a limit of where the file runs,
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
