# LTX-2.5 — the latent upsampler's `dims == 2` arm

Row: `LTX25-UPSAMPLER-ARMS`. Campaign: [`ltx-2-5.md`](ltx-2-5.md)
(operator-owned; **not edited by this row**). Issue:
[#2577](https://github.com/mudler/vllm.cpp/issues/2577). Scope source:
[`ltx25-completion-scope.md`](ltx25-completion-scope.md) §8 order 6, which pairs
**A8** and **A9** as "the cheapest real gaps", size S each, no dependencies.

Base: `origin/main` at `6974557b0`.

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |
| vLLM-Omni | `a4ea67a` |

vLLM-Omni registers `ltx2` and is the primary oracle for this architecture. It
carries no `LatentUpsampler`, so the operator's definition is read from the
model author's own runtime, which is the row `ltx-2` occupies in AGENTS.md's
secondary-oracle table.

## 0. Honesty statement — what this row does and does not claim

This row lands **A9 only**. A8 was investigated to the same depth and is
returned undecided under `## Owed`, with the measurement that makes it a
decision rather than a port. The scope document sized the two together and
called A8 "the single cheapest real gap"; that sizing is correct about the
*operator* and does not survive contact with the *consumer*, which is the
finding §5 records.

Nothing below is quoted from another document. Every local anchor was read from
the tree at `6974557b0` and every upstream anchor with
`git show fd4ded7f:<path>`, never from a working tree, because a fork's working
tree is not the pin.

## 1. The gap, on this tree

`Ltx2LatentUpsample` refuses every non-3-D config. The anchor below reads
`6974557b0`, the base this spec was written against, because the row's own
implementation DELETES those lines -- a §1 anchor is the only kind here that a
successful row is supposed to falsify:

```
src/vllm/model_executor/models/ltx2_upsampler.cpp:467-472 @ 6974557b0
  Require(config.dims == 3, "ltx2 upsampler: dims=... is not ported. ...");
```

`config.dims` is not a constant this port chose. It is read from the
checkpoint's own metadata at `src/vllm/model_executor/models/ltx2_loader.cpp`
(`Ltx2ParseUpsamplerConfig`), mirroring
`model/upsampler/model_configurator.py:17` (`dims = config.get("dims", 3)`).
A `dims=2` checkpoint is therefore an ordinary input, not a hypothetical.

## 2. What upstream builds

`model/upsampler/model.py:47`:

```python
conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
```

That one line reaches four parameter groups — `initial_conv` (`:49`), both
`ResBlock` stacks (`:53`, `:76-78`, whose own `conv` is chosen the same way at
`res_block.py:21`) and `final_conv` (`:80`). The `upsampler` module itself is
**not** chosen by `dims`: its branch (`:55-72`) tests the two flags only.

`model.py:85-100` is the forward:

```python
if self.dims == 2:
    x = rearrange(latent, "b c f h w -> (b f) c h w")   # :86
    x = self.initial_conv(x); x = self.initial_norm(x); x = self.initial_activation(x)
    for block in self.res_blocks: x = block(x)
    x = self.upsampler(x)                                # :94
    for block in self.post_upsample_res_blocks: x = block(x)
    x = self.final_conv(x)
    x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)  # :100
```

### 2.1 The two consequences of the fold, which ARE the port

**Kernel rank.** Every convolution above is a 4-D `Conv2d` weight, where the
`dims == 3` arm's identically-named tensor is 5-D. This is the same rank trap
the temporal arm already documents at `ltx2_upsampler.cpp:453-456` for
`upsampler.0.weight`, now applying to four more groups.

**GroupNorm statistics.** `(b f)` makes every frame its own sample, so
`GroupNorm(32, ...)` reduces over `(channels_per_group, H, W)`. Our `GroupNorm`
reduces over `frames * height * width` (`ltx2_upsampler.cpp:178`), which is the
`dims == 3` statistic. A port that reused it would produce a correctly shaped,
finite, plausible latent that is wrong everywhere — the failure mode this file's
header already names twice.

There is no third consequence. `Silu` is pointwise, `PixelShuffle2d` and the
rational resampler are already per-frame operators, and the `upsampler` branch
does not read `dims`.

## 3. Design

Mirror the fold rather than re-deriving its effect. Inside the existing
per-batch loop, the `dims == 2` arm iterates frames and runs each as its own
one-frame `Volume`. That reproduces `(b f)` exactly and, because it does, the
existing `GroupNorm` gives the per-frame statistic with no change to it: a
one-frame volume's `frames * height * width` **is** `height * width`.

The convolutions become `Conv2dPad1PerFrame`, which already exists
(`ltx2_upsampler.cpp:120`) and already takes a 4-D weight and already asserts
its element count at `:130-131`. `ResBlockForward` gains a `dims` parameter and
selects between the two conv helpers, mirroring `res_block.py:21` rather than
duplicating the block.

`EnumerateLtx2UpsamplerTensors` emits 4-D shapes for the four groups when
`dims == 2`. Nothing else in that function moves.

**Why not a separate 2-D code path.** AGENTS.md forbids a parallel hand-written
path where a seam can carry the behaviour, and upstream itself expresses the
difference as one conv-constructor choice plus a reshape. Two functions here
would be two things to keep in agreement for no gain.

## 4. Reachability

The dims=2 arm is reached from a production entry point on its default
configuration, and its output is consumable rather than merely computed.

Entry point: the `upsampler_path` load extra
(`include/vllm/multimodal/ltx2_video.h`), read by `LoadVideoEngine` and applied
by the second phase's input transform at
`src/vllm/multimodal/ltx2_video.cpp:3521-3524`.

Consumability is the part worth stating, because it is what separates A9 from
A8. With the default flags a `dims=2` upsampler returns `[c, f, 2h, 2w]`:
frames preserved by the fold, `h` and `w` doubled by `PixelShuffleND(2)`.

`Ltx2UpsampleVideoLatent` has **three** product call sites, all in
`ltx2_video.cpp`, and each pins the frame axis:

| # | Call site | What it requires of the frame axis |
|---|---|---|
| 1 | `:3521`, the video latent | `up.frames == vshape.frames`, checked at `:3525-3531` |
| 2 | `:3548`, the generated keyframe slots (`dfr_pipeline.py:348`) | `up_slots.frames == slot_positions.size()`, checked at `:3552-3563` |
| 3 | `:5058`, DFR's temporal rounds | the **temporal** arm, not this one |

Sites 1 and 2 are the spatial arm's, and both require the frame count to come
back unchanged. A `dims=2` upsampler does exactly that, so the shapes agree at
**every** frame count and a caller who supplies a `dims=2` checkpoint renders at
the full requested size.

The smallest failing test therefore enters through `LoadVideoEngine` and
`Generate`, not through `Ltx2LatentUpsample` directly, and asserts a completed
render rather than a changed error message.

## 5. Why A8 is NOT in this row

The operator is genuinely small: `model.py:55-59` is `Conv3d(mid, 8*mid)` +
`PixelShuffleND(3)`, and `:109-113` routes it through the same first-frame drop
the temporal arm already uses. The generator already constructs the module
(`scripts/gen-ltx2-pipeline-goldens.py`, the `spatiotemporal` block), so an
upstream golden is available for the asking.

The consumers are the problem, and there are two of them rather than one. The
arm returns `[c, 2f-1, 2h, 2w]`, while both spatial call sites in §4's table
require the frame count back unchanged — `vshape.frames` at `:3525-3531` and
`slot_positions.size()` at `:3552-3563`. Those agree only when `f == 1`, that is
when the temporal doubling is exactly undone by the mandatory drop.

`vshape.frames` is `(frames - 1) / factors.time + 1` (`:3411`) and
`factors.time` is the default-constructed 8
(`ltx2_video.cpp:2871`, `ltx2_pipeline.h:460-462`), so `f == 1` needs
`frames <= 8`. That is reachable — `num_frames` in 2..8 passes through verbatim
at `:2852-2853`, and the only lower bound anywhere on the path is `frames < 1` at
`:2869` — but it is the one configuration in which the capability's whole point,
the doubled frame axis, is cancelled before anything can observe it.

For every other clip, porting the operator would replace the named refusal at
`ltx2_upsampler.cpp:497` with the generic shape complaint at `:3527`. The guard
immediately above it (`ltx2_video.cpp:3473-3502`) exists to prevent precisely
that substitution for the temporal-only arm, and its comment says so.

So the choice is between mirroring upstream's compute and keeping a diagnosis
this tree deliberately built. That is a product decision. AGENTS.md's stop
condition covers it: return `NEEDS_DECISION` rather than invent behaviour, and
rather than land an arm no production path consumes. Recorded under `## Owed`.

## 6. Tests, ported in this change

| Upstream | Ported as |
|---|---|
| `model.py:47,85-100` executed at reduced dimensions | a `Dims2` arm in `scripts/gen-ltx2-pipeline-goldens.py` §8, emitting out-shape, parameter manifest and value golden from the real module |
| the parameter contract implied by `conv = Conv2d` | `CheckManifest` over the `Dims2` manifest, which fails on a 5-D shape before any value is compared |
| — | the end-to-end reachability case in `tests/vllm/multimodal/test_ltx2_video.cpp`, which renders through `LoadVideoEngine`/`Generate` with a `dims=2` checkpoint |

Both sides rebuild every weight from the shared name-keyed stream, so no weight
byte is checked in and a shape disagreement changes the values rather than
hiding.

The refusal case in `test_ltx2_pipeline.cpp` ("the upsampler refuses the arms it
does not implement") loses its two `dims` arms and keeps the rest.

It also gains the EIGHTH CELL, where both 2-D contradictions are set at once.
Only one of them is upstream's: `__init__` is an if/elif chain, so with
`spatial_upsample` false `elif temporal_upsample` (`model.py:68`) builds the
Conv3d and never reads `rational_resampler`. The generator drives that cell
through the real module as a third contradiction and asserts the CHANNEL-count
exception, so which refusal the port fires there is measured rather than
reasoned. The C++ case asserts the MESSAGE and asserts the other flag is not
named, because the polarity cannot see the defect: the cell is refused whichever
`Require` runs first.

## 7. Risks

- **The fold is mirrored but the golden is the only thing that proves it.**
  A per-frame loop and a folded batch differ in no shape, only in the GroupNorm
  statistic. The `Dims2` golden is the gate; `mid_channels = 32` with
  `GroupNorm(32, ...)` gives one channel per group, so the per-frame and
  per-clip statistics differ and the golden separates them.
- **A 2-D checkpoint is not one this campaign has in hand.** The arm is gated
  against the executed upstream module, not against a shipped file, and §8 says
  so rather than implying a checkpoint gate it does not have.

## 8. Gates

```sh
python3 scripts/gen-ltx2-pipeline-goldens.py --ltx2 ~/_git/LTX-2 \
    --vllm-omni ~/_git/vllm-omni --out tests/vllm/models/ltx2_pipeline_goldens.inc
ctest --test-dir build -R 'ltx2_pipeline|ltx2_video' --output-on-failure
scripts/agent-preflight.sh --staged
```

## Now

`DONE` — A9 landed against issue #2577. A8 stays open under `## Owed`.

## Outcome

### What was built

`dims == 2` is no longer refused. `EnumerateLtx2UpsamplerTensors` emits 4-D
kernels for `initial_conv`, both ResBlock stacks and `final_conv` when the
checkpoint sets it; `ResBlockForward` takes the rank as a parameter, mirroring
`res_block.py:21`; and the forward runs one frame at a time, which is how
`model.py:86`'s fold is reproduced rather than approximated.

The stack itself became one lambda used by both ranks. Upstream's two forward
branches run the identical module sequence and differ only in the fold around
them, so a second copy here would have been the parallel path AGENTS.md forbids.

`model.py:47`'s `else` is now mirrored as written: any `dims` that is not 2
builds Conv3d. The old `Require(config.dims == 3, ...)` refused checkpoints
upstream runs, which was a refusal this port invented.

### Red before green

The dims=2 golden was written first and failed for both intended reasons on the
unmodified tree: six parameter-count mismatches (`5184 == 1728` for
`initial_conv`, `27648 == 9216` for each ResBlock conv), then the forward threw
`ltx2 upsampler: dims=2 is not ported`. Reproduce it by reverting
`src/vllm/model_executor/models/ltx2_upsampler.cpp` to its parent and rerunning
`test_ltx2_pipeline -tc='*dims=2 arm*'`; the goldens are already in the tree, so
the red is a checkout away and is not carried here as a log file.

After the change, on the restored tree: `test_ltx2_pipeline` 62/62 cases and
3625 assertions green, `test_ltx2_video` 109/109 cases and 4866 assertions green
including the new end-to-end render. The generator regenerates
`ltx2_pipeline_goldens.inc` byte-identically with its third contradiction case
added, because a contradiction case emits nothing.

### Mutations — four RED, each restored byte-for-byte

| Mutation | Golden | End-to-end render | What it proves |
|---|---|---|---|
| **A**: `const bool two_d = false` | RED, `conv3d weight has the wrong element count` | **RED**, same throw at `test_ltx2_video.cpp:690` | The production path — `LoadVideoEngine` + `Generate` through the `upsampler_path` extra — really enters the new branch. This is the reachability proof, and it is a render that fails, not a unit test |
| **B**: keep the 2-D convolutions, drop the per-frame fold | **RED**, `max|diff| = 0.679842`, and the per-frame independence property RED at `0.439442` | **GREEN** | The fold's only consequence is numerical. Every shape is identical, so no shape check anywhere can see it and the value golden is the sole gate. This is the claim the code comment makes, now measured |
| **C**: the eighth cell's expected substring, `channels, but got` -> `not enough values to unpack` | n/a | n/a | **RED**: the generator raised and quoted what upstream really said -- `expected input[1, 3, 32, 4, 6] to have 32 channels, but got 3 channels instead`. The eighth cell fails by the TEMPORAL mechanism, measured against the module rather than argued from the if/elif chain |
| **D**: swap the two `dims=2` `Require`s back, rational first | n/a | n/a | **RED at exactly two assertions**, `test_ltx2_pipeline.cpp:2282-2283`, 61/62 cases and 3623/3625. Nothing else moves, because the polarity is identical either way -- which is the whole reason the case reads the message. Restored byte-for-byte, sha256 verified |

Mutation B is the one worth keeping in mind: it is the shape of defect this file's
header warns about twice, and it passes an end-to-end render.

Mutations C and D are the pair that gates the ORDER of the two refusals. C says
what upstream does; D says the port would stop saying it if the order changed.

### A sibling defect this row did NOT fix, and why

`src/vllm/multimodal/ltx2_video.cpp:3499-3501` tells a caller who supplies the
temporal-only checkpoint that "no phase of any recipe this engine serves
consumes it, because its only upstream consumer is DFRPipeline's rounds loop,
which is not ported". DFR **is** ported and drives that arm at `:5058`. It is
the same false statement this row corrected in `ltx2_upsampler.h` under
[#2580](https://github.com/mudler/vllm.cpp/issues/2580), one file over, and it
is user-facing where the header's was not.

It is already tracked as **D11b** in
`.agents/specs/ltx25-completion-scope.md` §6.1, so it is recorded here rather
than re-filed — intake without an exit is what that document says produced 701
open issues. Left to its owner because repairing a user-facing refusal is its
own red-first change, not a comment fix ridden along on this one.

### The anchors this row cited went stale inside this row

The change moves lines on THREE axes, and the first pass at this section saw
only one of them. The load-time guard adds 16 lines near the top of
`ltx2_video.cpp`, so every anchor below it moved by 16. The port also takes
`ltx2_upsampler.cpp` from 582 lines to 714 across nine separate hunks -- 723
once the repair below swaps the two refusals -- so
anchors into that file moved by an amount that depends on where they point --
`:465` became `:497` and `:166` became `:178`, and no single offset produces
both. That second axis is the one the first pass at this section missed
entirely, and "moved by exactly 16" is why: an offset that holds for one file
reads like a rule.
`test_ltx2_video`'s recorded-reader-anchor gate caught its own list and nothing
catches prose.

Every anchor into those three files that appears in a file this change edits was
re-derived by grepping for the statement, and each is listed with what was
grepped for in the pull request body. That is the claim this section makes, and
it is deliberately narrower than "every anchor in this branch's prose": the
first pass asserted the wider claim, swept the spec, the header and the issues,
and missed both test files and one half-updated range. The narrow claim is what
was measured.

What the narrow claim leaves out, stated rather than implied: 4 anchors to
`ltx2_upsampler.cpp:465` and `:467` survive in `.agents/specs/`
(`ltx25-completion-scope.md:75, :341, :342` and `ltx25-retire-dead-arms.md:433`),
and dozens of anchors into `ltx2_video.cpp` past line 1525 survive across other
rows' specs and `.agents/completed/`. Those are other rows' records of their own
trees, and `ltx25-completion-scope.md` is an operator-owned shared file this row
does not edit for the reason §"The scope document contradicts §5" already gives.
Editing them from here would make this branch write a file every concurrent pull
request writes, which AGENTS.md §Records names as the failure to avoid.

Worth stating because the numbers were correct when written and wrong when
pushed, with no edit in between: a line anchor is a claim about a file that a
later hunk in the SAME change -- or a merge that carries no hunk of its own --
can falsify. The repair commit demonstrated it twice more: swapping the two
`dims=2` refusals moved `BlurDownsample` from `:590` to `:599` inside this same
section's edit, and that anchor had already been stale since #641.

### What was NOT done

No `dims=2` checkpoint exists on the NAS, so this arm is gated against the
executed upstream module at reduced dimensions and has no real-weight result.
That is stated rather than implied, and it is why `docs/USAGE.md` gains no
checkpoint row: there is no artifact to pin.

## Owed

- **A8, the spatiotemporal upsampler arm** (`model/upsampler/model.py:55-59`,
  refused through `Ltx2RefuseUnportedPipelineFeature(kSpatiotemporalUpsampler)`).
  Returned as a decision, not a port: §5 has the measurement. Tracked by
  [#2584](https://github.com/mudler/vllm.cpp/issues/2584), which carries the
  derivation and the three answers a decision has to choose between.

  **It has its own issue for a reason.** It was first recorded against
  [#2577](https://github.com/mudler/vllm.cpp/issues/2577), the issue this row
  CLOSES, so the owed item would have pointed at a closed issue the moment the
  work landed. `scripts/check-agent-record.py` cannot catch that, because this
  `## Owed` entry satisfies the gate on its own -- which is precisely why the
  entry has to name a live issue rather than lean on the gate.

### The scope document contradicts §5, and this is the record of it

`.agents/specs/ltx25-completion-scope.md:371` calls A8 "the single cheapest real
gap and the only reachable unported-feature refusal", and its §8 sizes it S
beside A9. §5 above falsifies the sizing: the operator is cheap and the landing
is not, because the second consumer at `ltx2_video.cpp:3548` was not counted.

That document is operator-owned and a shared-file lock under AGENTS.md
§Records, so it is **not edited from this row**. The contradiction is recorded
here and in [#2584](https://github.com/mudler/vllm.cpp/issues/2584) instead, for
the operator to reconcile at their own cadence. Anyone reading the sequence
should treat order 6's "S each" as holding for A9 alone.
