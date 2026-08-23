# VT-CONV1D-MODEL-BLOCK — every audio model's own suite now enters the blocked path

Row `VT-CONV1D-MODEL-BLOCK`. Issue
[#1684](https://github.com/mudler/vllm.cpp/issues/1684), which
[`vt-conv1d-time-block.md`](vt-conv1d-time-block.md) `## Owed` opened and which
this row closes for the four models it names. Parent lane
[#1664](https://github.com/mudler/vllm.cpp/issues/1664) / [#672](https://github.com/mudler/vllm.cpp/issues/672).

**Where the index link lives.** `.agents/issue-index.md` already carries #1684,
owned by `VT-CONV1D-TIME-BLOCK`. The file is append-only and
`scripts/check-agent-record.py` refuses a second row for an issue that already
has one, so this row deliberately appends nothing and the link stays where it
is. The three places that must agree are that existing index row, this spec, and
the pull request body.

## Now

`DONE`. The four per-model suites named in #1684 now red under M7b; the
before/after contrast is §4 and the residue is §6.

## 0. Scope

**In scope.** Test fixtures only. Four suites gain one case each, entering
through the model's own conv-carrying entry point at a shape whose convolutions
cross a `vt::Conv1d` time-block boundary:

| Suite | Entry point | Fixture |
|---|---|---|
| `test_minimax_music3_acoustic` | `m3::VocoderResidualUnit` | 8 channels x 16 384 positions, closed form |
| `test_bigvgan` | `bigvgan::Forward` | 256 mels x 704 frames, two-window equivalence |
| `test_ltx2_vae` | `Ltx2VocoderForward` | 2 x 1 472 x 64 mel, two-window equivalence |
| `test_minimax_h3` | `MiniMaxH3AudioVaeDecode` | 256 mels x 704 frames, two-window equivalence |

`tests/CMakeLists.txt` gains `-I src` on `test_bigvgan` and `test_minimax_h3`,
which the other two already had, so every case can read
`vt::cpu::Conv1dTimeBlock` and assert the geometry it claims.

**Out of scope, deliberately.**

- **The kernel.** `src/vt/cpu/cpu_conv1d_general.cpp` is not touched. Its
  bit-identity is the load-bearing guarantee of #1664 and four models' goldens
  rest on it; a fixture that moved it would be a different row.
- **Every committed golden.** No `.inc` changes. Each case is additive and the
  existing arms keep their inputs, their tolerances and their counts.
- **`test_indextts2_pipeline`.** §6, with the measurement that puts it there.
- **`test_indextts2_family`.** It does not reach `vt::Conv1d` at all — the M5b
  provider-refusal mutation of #1678 left it green — so there is nothing here to
  cover. It stays green under M7b for the same reason `test_ops_conv1d_depthwise`
  does, and both are the control that says a red is a signal rather than a broken
  build.

## 1. The gap

`vt::Conv1d`'s CPU provider cuts its work into (time block, output row) pairs
(#1664, `src/vt/cpu/cpu_conv1d_block.h`). #1684 measured what saw that second
axis: mutation **M7b** sign-flips every output cell where `blocks > 1`, exactly
and only the axis #1664 added, and at that head EIGHT of ten suites stayed
green. All four audio consumers reach the op — the M5b provider refusal reds
eight suites — but every one of them at SINGLE-BLOCK shapes only.

#1678 closed half of it in flow: `tests/vllm/models/test_vocoder1d.cpp` gained
an arithmetic case crossing a boundary through `vllm::vocoder1d::Conv1d`, the
body all four models call. Re-measured on `origin/main` at `21abaf169` (§4), M7b
now reds three suites and leaves seven green, and the four MODEL suites are all
in the green list. That is what this row closes.

## 2. Design

### 2a. Why every fixture is at least 512 KiB of activation, and how the cost is kept flat

`Conv1dTimeBlock` returns `length` — one block — unless
`out_channels * kernel <= in_len`, and otherwise the largest multiple of
`kConv1dPosTile` whose activation slice fits `kConv1dSliceBytes` (512 KiB). So
`blocks > 1` is equivalent to *the convolution's activation exceeds the slice
budget*, and no fixture can be cheaper than that. What a fixture CAN choose is
how the 512 KiB is spent: the convolution costs
`kernel * out_channels * (in_per_group * length)`, and only the bracket is fixed
by the budget. Every fixture here therefore puts the budget into the LONG axis
and keeps the channel counts small.

### 2b. MiniMax-Music3: a closed form, because the activation can be switched off exactly

`m3::VocoderResidualUnit` is snake -> pad -> conv1(k=7) -> snake -> conv2(k=1) ->
residual add. `vocoder1d::SnakeActivation` computes
`x + (b + 1e-9)^-1 * sin^2(a * x)` with `b = a` for a null beta and no
exponentiation on Music3's `logscale = false` arm, so **alpha = 0 makes it the
identity exactly**: `sin(0)` is `0`, `1e9 * 0` is `0`, and `x + 0` is `x` for
every finite float. The unit then reduces to two convolutions and an add, which
is what the case is about, and the activation keeps its own gate two cases
above.

With channel 0 carrying `x[0][t] = t`, conv1 tapping only channel 0 with weight
1 and conv2 copying channel 0 into every output row, every partial sum is an
integer below 2^24 and f32 holds all of them exactly. The expectation carries no
tolerance, so a one-bit scheduling defect shows as a hard inequality — and it
covers the boundary cell, the cell after it, and the short LAST block.

8 channels x 16 384 positions is the smallest channel count that reaches the
budget, so both convolutions block at 8.3 M multiply-accumulates.

### 2c. BigVGAN, LTX-2.5 and MiniMax-H3: two windows, because no closed form survives an anti-aliased Snake

All three end every stage in `AliasFreeActivation1d` — upsample 2x with a
kaiser-sinc filter, Snake, downsample — which no choice of alpha turns into the
identity. A closed form is therefore not available, and a golden would need the
upstream generator (or, for H3, the checkpoint's remote code) re-run at a 512
KiB activation.

It does not need one. Every stage of these decoders is a **local, shift
-equivariant** operator: zero-padded convolutions, a strided transpose, a
pointwise activation, residual adds and a mean. Decoding a WINDOW of the input
therefore reproduces the long decode sample for sample, except within the
window's own edge — and BIT FOR BIT, because a cell's reduction is `seed`, then
`ic` ascending, then `k` ascending, and that sequence does not mention the block
(`cpu_conv1d_general.cpp`, the DETERMINISM CONTRACT). Each window is short
enough that its convolutions take ONE block, so the comparison is the blocked
arithmetic against the unblocked arithmetic, with the weights, the input values
and the reduction order all held fixed.

**Two windows, not one, and this is the load-bearing part of the design.** The
block boundary always falls at the block length, which is by definition the
longest a single-block reference can be, so a PREFIX window can never reach it:
a defect confined to the LAST block would leave a prefix-only case green. The
second window ends where the long decode ends and starts late enough that the
boundary lands in its interior. Each case asserts the coverage rather than
assuming it — that the two windows meet, that the first starts at sample 0, that
the second ends at the last sample, and that the boundary is strictly inside the
second — and §4's M2 leg is the measurement that says it was necessary.

Every window is trimmed by 192 output samples, against a receptive field of
about 44 (conv_pre 3 frames, the transpose 2, three dilated resblock pairs 12,
seven 12-tap 2x resamplers 21, conv_post 3).

### 2d. The three things each case asserts about itself

Because a case that silently became single-block would be a gate that reports on
a state it was not given:

- `REQUIRE(block < long_length)` — the long run really blocks.
- `REQUIRE(block == ref_length)` (or `block >= ref_length`) — the reference
  really does not.
- `REQUIRE(block % kConv1dPosTile == 0)` — the tile alignment #1664 rests on.

And two more that keep the comparison from being vacuous: the compared waveform
is not saturated (`peak < 0.999`, so a flattened tanh cannot agree for free) and
not constant (`hi - lo > 0.1`).

## 3. Risks

- **A fixture that changed the kernel's arithmetic.** Refused by construction:
  no file under `src/` is touched, and §4's restore leg proves the ten baseline
  binaries hash back byte for byte.
- **A window comparison that is not actually bit-exact.** Measured rather than
  argued: all three report `differing=0 worst|diff|=0` over 1 536, 7 168 and
  1 536 samples, at exact float equality rather than a tolerance.
- **A saturated or silent reference.** Both bounded, and the H3 case's
  weight-norm gain was CHOSEN by that measurement: 0.15 gives a 0.011
  peak-to-peak waveform (two near-silent signals agreeing for the wrong reason)
  and 1.0 saturates the tanh at peak 1.0 (two flat signals agreeing for a
  different wrong reason). 0.4 measures 0.112 peak and 0.220 span.
- **Suite wall time.** The four cases add 8.3 M, ~25 M, ~32 M and ~25 M
  multiply-accumulates to suites that already carry 386 to 57 395 assertions.
  No suite's run time moved perceptibly at `-O0`.

## 4. Gates and evidence

Authoring host: x86-64, 20 cores, `uptime` load 13-16 throughout. Build is the
CI configuration exactly — `cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, no `CMAKE_BUILD_TYPE`, so `NDEBUG` is not
defined and asserts are live. Base `origin/main` at `21abaf169`.

### 4a. The ten suites, before and after — cases, assertions and `Status:` in full

`assertions: 0` is a skip wearing a pass, so every line carries all three.
`test_ops_conv1d_general` prints **4 `[SKIP]` lines** on a CPU-only build (its
CUDA arms), so its counts are not full coverage of that file.

| Suite | baseline | with the four cases |
|---|---|---|
| `test_ops_conv1d_general` | 14 / 19 696 / `SUCCESS!` rc 0 (4 SKIP) | 14 / 19 696 / `SUCCESS!` rc 0 (4 SKIP) |
| `test_host_parallel` | 11 / 968 / `SUCCESS!` rc 0 | 11 / 968 / `SUCCESS!` rc 0 |
| `test_vocoder1d` | 11 / 66 / `SUCCESS!` rc 0 | 11 / 66 / `SUCCESS!` rc 0 |
| `test_bigvgan` | 6 / 65 / `SUCCESS!` rc 0 | **7 / 85** / `SUCCESS!` rc 0 |
| `test_minimax_music3_acoustic` | 39 / 386 / `SUCCESS!` rc 0 | **40 / 395** / `SUCCESS!` rc 0 |
| `test_ltx2_vae` | 44 / 3 131 / `SUCCESS!` rc 0 | **45 / 3 152** / `SUCCESS!` rc 0 |
| `test_minimax_h3` | 79 / 57 395 / `SUCCESS!` rc 0 | **80 / 57 416** / `SUCCESS!` rc 0 |
| `test_indextts2_pipeline` | 8 / 433 / `SUCCESS!` rc 0 | 8 / 433 / `SUCCESS!` rc 0 |
| `test_indextts2_family` | 7 / 22 / `SUCCESS!` rc 0 | 7 / 22 / `SUCCESS!` rc 0 |
| `test_ops_conv1d_depthwise` | 5 / 1 184 / `SUCCESS!` rc 0 | 5 / 1 184 / `SUCCESS!` rc 0 |

### 4b. THE ROW: M7b before and after

**M7b** replaces the kernel's store with
`on[t0 + i] = blocks > 1 ? -acc[i] : acc[i]`, sign-flipping every output cell of
`Conv1dKernel` where and only where the row's second axis is taken. One hunk,
`1 file changed, 1 insertion(+), 1 deletion(-)`, `compile_rc = 0`, 0 `error:`
lines.

| Suite | M7b BEFORE (pristine `21abaf169`) | M7b AFTER (this row) |
|---|---|---|
| `test_ops_conv1d_general` | rc 1, 14 / 12 passed / 2 failed, 19 696 / 2 failed, `FAILURE!` | rc 1, same |
| `test_host_parallel` | rc 1, 11 / 1 failed, 968 / 5 failed, `FAILURE!` | rc 1, same |
| `test_vocoder1d` | rc 1, 11 / 1 failed, 66 / 4 failed, `FAILURE!` | rc 1, same |
| `test_bigvgan` | rc 0, 6 / 65 / `SUCCESS!` | **rc 1, 7 / 1 failed, 85 / 1 failed, `FAILURE!`** |
| `test_minimax_music3_acoustic` | rc 0, 39 / 386 / `SUCCESS!` | **rc 1, 40 / 1 failed, 395 / 4 failed, `FAILURE!`** |
| `test_ltx2_vae` | rc 0, 44 / 3 131 / `SUCCESS!` | **rc 1, 45 / 1 failed, 3 152 / 1 failed, `FAILURE!`** |
| `test_minimax_h3` | rc 0, 79 / 57 395 / `SUCCESS!` | **rc 1, 80 / 1 failed, 57 416 / 1 failed, `FAILURE!`** |
| `test_indextts2_pipeline` | rc 0, 8 / 433 / `SUCCESS!` | rc 0, 8 / 433 / `SUCCESS!` (§6) |
| `test_indextts2_family` | rc 0, 7 / 22 / `SUCCESS!` | rc 0, 7 / 22 / `SUCCESS!` (does not reach the op) |
| `test_ops_conv1d_depthwise` | rc 0, 5 / 1 184 / `SUCCESS!` | rc 0, 5 / 1 184 / `SUCCESS!` (a different op) |

Three red before, **seven red after**, and every suite that gained a case is one
of the four that moved.

### 4c. M2 — the trailing block, which is why there are two windows

**M2** replaces `units = blocks * rows` with
`(blocks > 1 ? blocks - 1 : blocks) * rows`, so the LAST block is never
computed. One hunk, `compile_rc = 0`. It reds exactly the same seven suites:
`test_ops_conv1d_general` 2 failed cases, `test_host_parallel` 1, `test_vocoder1d`
1 (2 assertions), `test_bigvgan` 1, `test_minimax_music3_acoustic` 1 (3
assertions), `test_ltx2_vae` 1, `test_minimax_h3` 1; the three controls stay
green. This is the leg a prefix-only window would have failed, because the
boundary is the reference's own maximum length (§2c).

### 4d. Restore, verified by hash rather than by intention

Every mutation was applied to a clean tree (`git status --porcelain` empty
before each) and reverted with `git checkout -- src/`. After the revert the ten
suites were rebuilt and their binary sha256 prefixes compared: all ten returned
to their pre-mutation values —
`ebc958f65d20bd53`, `33976d362ec6dfdb`, `f75309691ca757f7`, `0990b95a204b52c9`,
`1a234708d8a1f918`, `201cd3a1af795575`, `519ac0625b1d54cb`, `2bfe20be3f5de0e3`,
`9dcd2ff375024fbe`, `c4737a79cd9eae72` for the baseline leg, and the four
changed binaries to `96d58c7ad2f52f86`, `6b3856f88e3ff437`, `53c7e4088cc3038d`,
`3a1564e87e6fc82c` for the green leg.

### 4e. One trap this row hit, recorded because it is silent

Restoring the edited test files with `cp -a` PRESERVED their mtimes, which were
older than the objects `make` had just rebuilt from the pristine sources. `make`
skipped one translation unit and the suite reported the baseline's own 39 cases
/ 386 assertions — a restore that looks exactly like a build. The count was what
caught it. Every later restore was followed by `touch`.

## 5. Reachability

Each case enters through the model's own public entry point rather than through
`vt::Conv1d` — `m3::VocoderResidualUnit`, `bigvgan::Forward`,
`Ltx2VocoderForward`, `MiniMaxH3AudioVaeDecode` — which is the whole point: the
op's own suite already covered the op. §4b is the reachability measurement, and
it is the reachability mutation of
[`reachability.md`](../reachability.md) run from the other end: rather than
deleting a production call site and asking whether the gate notices, it corrupts
exactly the axis under test and asks which suites notice. Before this row the
four models' suites did not.

## 6. Owed

- **`test_indextts2_pipeline` still reaches `vt::Conv1d` at a single-block shape
  only**, and this row declines to change that. Owned by this row and tracked by
  [#1684](https://github.com/mudler/vllm.cpp/issues/1684), which therefore stays
  OPEN. The reason is measured, not assumed: that suite's only reach is stage 6
  of `RunReduced` (`src/vllm/model_executor/models/indextts2_pipeline.cpp:164`),
  a `groups = mel_channels` DEPTHWISE convolution, so `in_per_group == 1` and
  `Conv1dTimeBlock(1, 7, 3, 1, 1, ...)` returns **131 040 positions** — the
  largest block length the budget can produce. Reaching `blocks > 1` needs a
  131 072-frame mel through all six stages, 6 553x the suite's own 20 frames,
  and the stage-6 kernel is `Synth(...)` in that file's anonymous namespace, so
  the expectation would have to be identified by least squares from a second
  run: a new instrument, for a convolution the file itself calls a structural
  seam demo rather than a vocoder. IndexTTS-2.5's REAL last stage is BigVGAN
  (`include/vllm/model_executor/models/bigvgan.h`), and that is now gated by
  `test_bigvgan` §4b. What would close it: either a proportionate fixture for
  the seam demo, or moving stage 6 onto `vt::DepthwiseConv1d`, where a
  `groups == channels` convolution belongs and where the block geometry is a
  different op's question.
- **The CUDA provider is not re-measured.** Unchanged from
  [`vt-conv1d-time-block.md`](vt-conv1d-time-block.md) `## Owed`: this host has
  no CUDA toolkit, `test_ops_conv1d_general` printed 4 `[SKIP]` lines, and
  nothing here touches `cuda_conv1d_general.cu`.
- **The two-window equivalence is a CONSISTENCY gate, not a correctness one.**
  It proves the blocked arithmetic equals the unblocked arithmetic bit for bit;
  it does not prove either is right. The absolute correctness of these three
  decoders is what their committed goldens are for, and this row moves none of
  them. Music3's case is the exception — its expectation is closed form.

## 7. Stop conditions

Stop and report rather than widen scope if: a fixture cannot reach `blocks > 1`
without moving a committed golden; a case would require a checkpoint; the change
reaches any file under `src/` or `include/`; or the two-window equivalence turns
out not to hold bit-exactly for a model, which would be a finding about that
model's locality rather than a reason to loosen the comparison.

## 8. Outcome

**What was measured.** M7b's split went from 3 red / 7 green to 7 red / 3 green,
and every suite that moved is one of the four #1684 names. M2 gives the same
split, which is the evidence that the second window earns its cost.

**What was rejected, and why.**

- **A golden per model.** Rejected: it needs the upstream generator, or the H3
  checkpoint's remote code, re-run at a 512 KiB activation, and it would gate
  nothing the two-window equivalence does not. The equivalence is EXACT where a
  golden would carry a tolerance.
- **A prefix window alone.** Rejected by measurement: the boundary falls at the
  block length, which is the reference's own maximum, so M2 would have left every
  case green. §4c is that measurement.
- **A closed form for the three BigVGAN-lineage decoders.** Rejected: their
  `AliasFreeActivation1d` is an upsample/filter/downsample pair that no alpha
  reduces to the identity, unlike Music3's plain Snake at `alpha = 0`.
- **A `test_indextts2_pipeline` fixture.** Rejected as disproportionate, with
  the block length that makes it so recorded in §6 rather than left to be
  rediscovered.
