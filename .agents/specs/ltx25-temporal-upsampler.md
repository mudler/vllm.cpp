# LTX-2.5 — the temporal x2 latent upsampler (`temporal_upsample=True`)

Row: `LTX25-TEMPORAL-UPSAMPLER`. Campaign: [`ltx-2-5.md`](ltx-2-5.md)
(operator-owned; **not edited by this row**). Issue:
[#644](https://github.com/mudler/vllm.cpp/issues/644), row **E** — *"Temporal x2
upsampler (the spatial one ships)"*.

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`, `packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Read from a local checkout at that revision. The golden generator IMPORTS and
EXECUTES the upstream module; nothing below is restated from memory.

---

## 0. Honesty statement — what this row does and does not claim

This row ports **one operator and one slice**, and gates them against executed
upstream. It does **not** deliver a temporal-upsampling capability a user can
drive, because the pipeline that drives it upstream is not ported and the
checkpoint that feeds it is not on the NAS. Both are stated as facts here before
any work starts, so neither can be discovered later as a claim this row did not
support:

1. **Reachability.** Upstream's ONLY consumer of a temporally-configured
   `LatentUpsampler` is `DFRPipeline` (`dfr_pipeline.py:235-245, 402-407`), whose
   `temporal_upsample_rounds ∈ {0,1,2}` loop also doubles the fps, rescales the
   carried keyframe positions and re-tiles the canvas. We have no DFR pipeline.
   Our engine's one upsampler call site is the `kSpatialUpsample` phase input
   transform (`src/vllm/multimodal/ltx2_video.cpp:1408-1466`), which shape-checks
   the result against a **spatially** doubled latent shape and fails otherwise.
   So after this row: **ported, loader-parsed and gated; not reachable from any
   pipeline this project ships.** §7 states this in the exact terms the records
   will carry.
2. **Weights.** `/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/latent_upscale_models/`
   holds `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` and **nothing
   else**. The temporal checkpoint named at `ltx-pipelines/docs/pipelines.md:176`
   (`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`) is **absent**.
   The gate is therefore the reduced-dimension executed-upstream fixture alone;
   no real-checkpoint claim is made and none is implied.

## 1. The grounding claim, verified on this tree

The task brief asserted this is *"the same model class with a flag, not a
separate architecture"*. **Verified, and it holds.** `LatentUpsampler.__init__`
builds one of four `self.upsampler` branches off the two booleans
(`model/upsampler/model.py:55-74`); everything around it — `initial_conv`,
`initial_norm`, `initial_activation`, `res_blocks`, `post_upsample_res_blocks`,
`final_conv` — is constructed identically for every branch:

```python
if spatial_upsample and temporal_upsample:            # :55
    self.upsampler = torch.nn.Sequential(
        torch.nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
        PixelShuffleND(3),
    )
elif spatial_upsample:                                # :60
    ...
elif temporal_upsample:                               # :68
    self.upsampler = torch.nn.Sequential(
        torch.nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
        PixelShuffleND(1),
    )
else:
    raise ValueError("Either spatial_upsample or temporal_upsample must be True")
```

and in `forward` (`model.py:109-113`):

```python
if self.temporal_upsample:
    x = self.upsampler(x)
    # remove the first frame after upsampling.
    # This is done because the first frame encodes one pixel frame.
    x = x[:, :, 1:, :, :]
```

`PixelShuffleND(1)` (`pixel_shuffle.py:47-52`) is a pure `einops.rearrange`:

```python
elif self.dims == 1:
    return rearrange(x, "b (c p1) f h w -> b c (f p1) h w", p1=self.upscale_factors[0])
```

with `upscale_factors` defaulting to `(2, 2, 2)` (`pixel_shuffle.py:25`), which no
construction site overrides — so `p1 = 2` is the shipped factor, not a config key.

**So the only NEW code this row owes is `PixelShuffleND(1)` and the slice.** We
already ship, gated against this same upstream at this same pin:
`Conv3dPad1` (`src/vllm/model_executor/models/ltx2_upsampler.cpp:69`), `GroupNorm`
(`:163`), `Silu` (`:202`), `ResBlockForward` (`:213`), `PixelShuffle2d` (`:229`)
and `SpatialRationalResampler`'s pieces.

### 1.1 Shapes, derived not assumed

Input `[b, C, F, H, W]` → `Conv3d(mid, 2·mid, k=3, p=1)` → `[b, 2·mid, F, H, W]` →
`PixelShuffleND(1)` → `[b, mid, 2F, H, W]` → drop frame 0 → `[b, mid, 2F-1, H, W]`.
`H` and `W` are untouched.

Cross-checked against the consumer's own bookkeeping:
`dfr_pipeline.py:408` computes `num_frames = 2 * (num_frames - 1) + 1` per round,
i.e. `2F - 1`. Executed on this tree (`.scratch/probe_temporal.py`, upstream at
`fd4ded7f`): `in (1, 6, 3, 4, 6)` → `out (1, 6, 5, 4, 6)`. Both agree.

### 1.2 The parameter contract

`named_parameters()` of `LatentUpsampler(dims=3, spatial_upsample=False,
temporal_upsample=True)`, executed at the fixture dims, is the spatial arm's list
with exactly one substitution: `upsampler.0.weight [2·mid, mid, 3, 3, 3]` and
`upsampler.0.bias [2·mid]`. Note the **rank**: this is a `Conv3d`, so the weight
is 5-D, where the non-rational spatial arm's same-named `upsampler.0.weight` is a
4-D `Conv2d` kernel (`model.py:64-66`). A port that reused the spatial arm's
enumeration would build a tensor of the wrong element count and the manifest
check would catch it — which is why the manifest is part of the gate and not a
comment.

## 2. Scope

**In.**

- `PixelShuffleND(1)` — the temporal reshape/permute (`pixel_shuffle.py:47-52`).
- The temporal-only branch of `LatentUpsampler.__init__` (`model.py:68-71`) and of
  `forward`, including the first-frame drop (`model.py:109-113`).
- `EnumerateLtx2UpsamplerTensors` for that branch — the loader-side contract.
- An executed-upstream golden at reduced dims, plus the shape and manifest.

**Out, refused by name rather than approximated.**

- `spatial_upsample AND temporal_upsample` (`model.py:55-59`) — a *different*
  operator: `Conv3d(mid, 8·mid)` + `PixelShuffleND(3)`, which also takes the
  first-frame drop. It is not "the temporal arm with spatial on", so the existing
  refusal is **retargeted** at it rather than deleted. §4.
- `dims == 2` (`model.py:85-100`) — unchanged, still refused.
- `DFRPipeline` and its rounds loop — §0.1.
- Any bf16 arm; this file is the CPU f32 reference arm, exactly as the spatial one
  is (`include/vllm/model_executor/models/ltx2_upsampler.h`, DTYPE note).

## 3. Design

`src/vllm/model_executor/models/ltx2_upsampler.cpp` gains one file-local helper
beside `PixelShuffle2d`:

```
PixelShuffle1d(in, up_f):   out.channels = in.channels / up_f
                            out.frames   = in.frames * up_f
                            out(c, f*up_f + j, y, x) = in(c*up_f + j, f, y, x)
```

which is `rearrange("b (c p1) f h w -> b c (f p1) h w")` written out: the source
channel index is `c * p1 + j` because `(c p1)` puts `p1` **fastest**, and the
destination frame index is `f * p1 + j` because `(f p1)` does the same. Getting
either factor order backwards produces a correctly-shaped, finite, plausible
latent — which is precisely why §5 mutates it.

`Ltx2LatentUpsample` gains the branch, ordered so the refused combination is
rejected before any weight is touched, and `EnumerateLtx2UpsamplerTensors` gains
the matching `upsampler.0.*` pair at `{2·mid, mid, 3, 3, 3}`.

No new public entry point: `Ltx2LatentUpsample` and `Ltx2UpsampleVideoLatent`
already take the config, and `Ltx2ParseUpsamplerConfig`
(`src/vllm/model_executor/models/ltx2_loader.cpp:1431-1444`) already reads
`temporal_upsample` off the checkpoint config, mirroring
`model_configurator.py:19`. The loader arm therefore needs no change — it needed
none because it was ported key-for-key rather than for the one arm in scope.
That is a claim this row VERIFIES rather than assumes: §5's manifest arm builds
the tensor set from a config produced the same way.

## 4. The refusal that moves

`Ltx2UnportedPipelineFeature::kTemporalUpsampler`
(`include/vllm/model_executor/models/ltx2_pipeline.h:581`) currently means "any
`temporal_upsample=True`". After this row that is false for the temporal-ONLY
arm and still true for the spatial+temporal one. It is therefore **renamed** to
`kSpatiotemporalUpsampler` and its message retargeted at `model.py:55-59`,
naming `PixelShuffleND(3)` and `8 * mid_channels`.

Renaming rather than deleting is deliberate: deleting the enumerator would leave
the spatial+temporal arm with no refusal at all, and `Ltx2LatentUpsample` would
fall through to the temporal branch and read `upsampler.0.weight` at the wrong
element count — an error, but the wrong one, reported as a corrupt checkpoint
instead of an unported feature.

## 5. Tests and evidence

Generator: `scripts/gen-ltx2-pipeline-goldens.py`, section 8, which already
imports `LatentUpsampler` and runs it. A fourth arm `Temporal` is added, plus a
temporal latent fixture with `F = 3` (distinct from `H = 4`, `W = 6`, so an axis
mix-up is a shape failure, and `2F-1 = 5 ≠ 2F = 6`, so a missing slice is one
too). Both sides rebuild weights from the shared FNV-1a/splitmix64 stream keyed
by parameter name; no weight byte is checked in.

Reproducibility of the generator was established **before** anything was added:
regenerating `tests/vllm/models/ltx2_pipeline_goldens.inc` from the committed
tree yields md5 `53e2a6aba8885d7d58302ad0b7b09eb4`, byte-identical to the
committed file. So any md5 change after this row is this row's content and not
environment drift.

Gate: `tests/vllm/models/test_ltx2_pipeline.cpp`.

- The `Temporal` arm gets **its own `TEST_CASE`** — "ltx2 the latent temporal
  upsampler reproduces upstream" (`test_ltx2_pipeline.cpp:1441`) — and does
  **not** join `LTX2_UPS_ARM` (`:1413-1422`). The macro cannot carry it, for
  three reasons that are properties of the macro and not preferences:
  1. It expands to `kLtx2Ups<TAG>Rational` and `kLtx2Ups<TAG>Scale`, which the
     temporal arm has no analogue of and the generator therefore does not emit
     — `LTX2_UPS_ARM(Temporal)` would not compile.
  2. Its `run` lambda captures ONE latent, `ReducedUpsamplerLatent()` at
     `F = 2`, shared by every arm; the temporal arm needs the `F = 3` fixture
     named above, because at `F = 2` a kept first frame would not be a shape
     failure.
  3. Its config comes from `ReducedUpsamplerConfig(rational, scale, prefix)`,
     which has no temporal flag, and it lives inside the case named "the latent
     **spatial** upsampler reproduces upstream".

  The separate case does everything the macro does — parameter manifest (names
  AND element counts), all five out-shape scalars, and `max|diff|` against the
  golden, a **MAX** and not a fraction of elements — and adds the two checks the
  macro has no slot for: `2F - 1` asserted as a RELATION against the input
  (`got.frames == kLtx2UpsTemporalFactor * latent.frames - 1`) and not only as a
  golden scalar, and `kLtx2UpsamplerTemporalFactor` pinned against the factor the
  generator read off `PixelShuffleND.__init__`'s own signature.
- The refusal case gains the spatial+temporal arm and keeps `dims=2` and
  "neither". `dims=2` is checked a second time through the TEMPORAL config,
  because the two arms take different branches and a `dims` guard placed inside
  the spatial branch would let a 2-D temporal config through.

**Mutations owed** (RED, restore, GREEN, with real doctest output and both the
assertion and the case counts):

| # | Mutation | Predicted signature |
|---|---|---|
| M1 | `PixelShuffle1d` reads source channel `j * out.channels + c` instead of `c * up_f + j` | golden RED, shape unchanged |
| M2 | drop the first frame → keep it (`1:` → `0:`) | shape RED (frames 6 vs 5) and golden RED |
| M3 | `up_f` 2 → the spatial arm's enumeration (4-D `upsampler.0.weight`) | manifest RED |

If any mutation stays green the magnitude is escalated before anything is
concluded, per the standing trap: only a probe that FAILS to reach proves
unreachability.

**Band.** `kRoundOff` is the suite's existing constant and this row does not
choose a new one; the observed `max|diff|` for the new arm is recorded in §8 as
measured, next to the three shipped arms' values, so the number is derived from
the measurement rather than picked to fit.

## 6. Risks

- **A shared generated file.** `ltx2_pipeline_goldens.inc` is regenerated whole.
  It is deterministic (md5 above), so a concurrent row's regeneration merges as
  a content conflict rather than silently interleaving. Resolution is
  regeneration on the merged tree, never a hand-edit of the `.inc`.
- **Sibling rows.** `row/LTX25-IMAGE-COND-FIX`, `row/LTX25-TILED-DECODE` and
  `row/LTX25-ADALN-CLAIMS` are live. This row touches none of
  `ltx2_conditioning.*`, `ltx2_tiling.*`, `ltx2_video_vae*`, `ltx2_dit.*`, and
  does not touch `src/vllm/multimodal/ltx2_video.cpp` at all.
- **The enum rename** is a compile-time break by construction: every use site is
  named in §4 and the build fails if one is missed.

## 7. Reachability — the sentence the records must carry

> The LTX-2.5 temporal x2 latent upsampler is **ported and gated against executed
> upstream at reduced dimensions**. Its config is parsed from a checkpoint by
> `Ltx2ParseUpsamplerConfig`. It is **not reachable from any shipped pipeline**:
> the engine's only upsampler call site is the spatial-upsample phase input
> transform, which requires a spatially doubled latent, and upstream's only
> consumer, `DFRPipeline`'s multi-round loop, is not ported. The checkpoint
> `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` is not present on the
> NAS, so no real-weight result exists either.

Nothing in `docs/` may say more than that.

**SUPERSEDED 2026-08-20 on both of its claims, by `LTX25-DFR-ROUNDS`
([#986](https://github.com/mudler/vllm.cpp/issues/986)).** The paragraph above is
kept as written, because it was true of this row and of the tree it measured.
What replaced it:

> The temporal x2 latent upsampler is **DRIVEN** by DFR's temporal rounds loop,
> reachable from `vllm_video_generate` with `pipeline_kind=dfr` and
> `temporal_upsample_rounds > 0`, and from `ltx2-gen --temporal-upsample-rounds`.
> The gate is still at REDUCED DIMENSIONS against a fixture checkpoint.

And the second claim went stale on its own, without anybody editing code:
`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` **IS on the NAS**,
mtime 2026-08-17, 261944000 bytes, 72 BF16 tensors, config
`spatial_upsample: false, temporal_upsample: true`, verified by header arithmetic
rather than by name. It arrived one day after the "re-verified 2026-08-16" that
several records in this campaign carry. A real-weight result still does not
exist, but the reason is now the missing `keyframe_slot_sft` BASE and not the
upsampler — see [`ltx25-dfr-rounds.md`](ltx25-dfr-rounds.md) §0.3 and §0.4.

## 8. Outcome

**Ported and gated. Not reachable from any shipped pipeline.** §7's sentence
stands unchanged after implementation — nothing measured here moved it.

### 8.1 What was built

`PixelShuffle1d` and `DropFirstFrame` in
`src/vllm/model_executor/models/ltx2_upsampler.cpp`, one branch in
`Ltx2LatentUpsample`, one branch in `EnumerateLtx2UpsamplerTensors`, and the
`kTemporalUpsampler` → `kSpatiotemporalUpsampler` retarget from §4. The loader
needed no change, as §3 predicted: `Ltx2ParseUpsamplerConfig` already read
`temporal_upsample`, and the engine test in 8.4 proves it end to end on a real
safetensors file.

### 8.2 The band, derived from measurement

All four arms measured on the same tree by temporarily bounding at `0.0`,
restoring immediately (test file md5 back to `214418c4b2c124eb37c64d3ce96ea3de`,
binary md5 back to the GREEN one):

| arm | `max|diff|` vs executed upstream |
|---|---|
| PixelShuffle (spatial x2) | 1.19209e-06 |
| Rational2 | 1.13249e-06 |
| Rational1p5 | 9.68575e-07 |
| **Temporal x2** | **2.38419e-06** |

The suite's existing `kRoundOff = 5e-6` holds without widening: the new arm is
2.0x the largest shipped arm and 0.48x the bound. It was **not** chosen to fit —
the three shipped arms were re-measured next to it precisely so the new number
could be read against a scale that predates this row. The bound is a MAX over
every element, never a fraction of them.

Two extra frames of accumulated convolution is the whole difference in scale:
the temporal arm's post-upsample res-blocks and `final_conv` run over `2F - 1 = 5`
frames where the spatial arms run over `F = 2`.

### 8.3 Mutations — every one RED, restored, GREEN

Run against `tests/vllm/models/test_ltx2_pipeline.cpp` (38 cases / 2443
assertions when green, `RUN_EXIT=0`). Each mutation was applied to the source,
`touch`ed, rebuilt with `BUILD_EXIT=0` and a CHANGED binary md5, run, then
restored to source md5 `32729b8895561d885f4bea9d4c02b0dd` and rebuilt back to
binary md5 `b662957cd1ce693727725cbce74649ae`.

| # | Mutation | Result |
|---|---|---|
| M1 | `PixelShuffle1d` source channel `c*up_f + j` → `j*out.channels + c` | RED, `max|diff| = 2.97727` vs bound `5e-06`; 37 passed / 1 failed, `RUN_EXIT=1` |
| M2 | first frame kept instead of dropped | RED on SHAPE first (`6 == 5`, twice) then `REQUIRE(864 == 720)`; 3 failed, `RUN_EXIT=1` |
| M3 | temporal `upsampler.0.weight` enumerated at the spatial arm's Conv2d rank | RED in the manifest (`18432 == 55296`) and the case then threw `conv3d weight has the wrong element count`; `RUN_EXIT=1` |

None stayed green, so no escalation was needed.

### 8.4 The regression this row would otherwise have shipped, and its gate

Retargeting the refusal made the *engine* worse before it made it better. A
temporal checkpoint handed to `--upsampler` shares the class name and the
`upsampler.0.*` tensor names with the spatial one, so after this row it LOADS
and RUNS, and the `kSpatialUpsample` phase then failed with a bare shape
mismatch instead of the named refusal it used to give.

`src/vllm/multimodal/ltx2_video.cpp` therefore gains a guard that names the swap,
and `tests/vllm/multimodal/test_ltx2_video.cpp` gains a subcase that writes a
`temporal_upsample: true` fixture checkpoint and asserts the message names it
AND does **not** contain the shape complaint. Removing the guard (`if (...)` →
`if (false)`) REDs it with the real prior message:

```
ltx-2.5 video: the upsampled latent is 4x3x1x1 but phase 'refine' needs 4x2x2x2
```

That RED is also the row's only end-to-end evidence, and it is worth more than
the guard: `4x3x1x1` is the temporal arm's output, from a real safetensors file,
through `Ltx2ParseUpsamplerConfig` and `Ltx2LoadVaeWeights`, at `2F - 1 = 3`
frames and unchanged `1x1` spatial. The loader path is not assumed.

### 8.5 The checkpoint

`/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/latent_upscale_models/`
contains `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` and nothing
else. The temporal one is **absent**, so `test_ltx2_video.cpp`'s
checkpoint-gated "loads and configures" subcase covers the spatial arm only and
this row adds no real-weight arm to it. Nothing here claims one.

### 8.6 Counts, against the denominator

Re-measured on the merged tree, merge commit `288bcba8d`, base
`6e6bba63d7c1a198b292207f53727aff79eb73c0`. The first row of figures below was
taken against a base about ninety commits older, so every denominator had moved
under it. A changed count is a red result until it is attributed, so each delta
names what it belongs to.

| Suite | at this base | on this branch | this row's delta |
|---|---|---|---|
| `test_ltx2_pipeline` | 37 cases / 2382 assertions (older base) | 38 / 2443 | +1 case, +61 assertions |
| `test_ltx2_video` | **33 / 576, measured** | 33 / 579 | +0 cases, +3 assertions |

Both `RUN_EXIT=0`. The base figure was measured at
`51e0cb5b1` and still applies here, because the two commits that advanced the
base to `6e6bba63d` touch neither focused suite: the diff of
`test_ltx2_video.cpp` and `test_ltx2_pipeline.cpp` across them is empty, while
the same command over `tests/` names eight other files, so the empty result is a
measurement and not a mistyped path.

`test_ltx2_video` grew from the 30 / 502 recorded above to 33 / 576 on `main`
while this branch was open, and none of that growth is this row's. The base
figure is measured rather than derived: the file was replaced by the base
version, the target rebuilt (`BUILD_EXIT=0`, zero compile errors, and a CHANGED
binary sha256 — so the replacement really ran rather than reading as a pass), the
suite run, and the tree restored to source sha256
`8c676f3e2b158326e9060949cc7ffae953c2e8038fcc608d875e2e5fe652bfe3` and binary
sha256 `dcc0192fb1cefeabbb9c2482296495c2952444da1e7cfd379d6632735f03f4eb`. This
row's only edit to that file is one additive `SUBCASE` carrying three `CHECK`s,
and the measurement returns exactly three. A `SUBCASE` adds no case, which is why
the case count is flat on both sides of the row.

`ctest -N` = **473**, and none of it is this row's. The 423 recorded earlier was
`-N` at the older base, and 472 was `-N` one base ago. This branch changes no
build file at all — the diff of `CMakeLists.txt`, `tests/CMakeLists.txt` and
`examples/CMakeLists.txt` against the base is empty — so it registers no binary
and its `-N` is whatever the base registers. The 472 to 473 step is the base's
own: the two commits that advanced it edited both `CMakeLists.txt` and
`tests/CMakeLists.txt`, which is the positive control for the empty diff above.

Full gate on the merged tree: `ctest -j 4` = **473/473 passed, 0 failed**,
`CTEST_EXIT=0`, 765.03 s, with `test_modelopt_mixed_precision_checkpoint` and
`test_voxtral_e2e` skipped by their own guards and nothing else skipped. Nothing
in the standing known-red set fired: `test_op_parity` passed in 1.40 s because
its fix arrived with the merge, and the parallel-starvation set needed no serial
re-run because nothing failed. The `-j 8` flake recorded above therefore did not
have to be re-tested. The run spanned a one-minute load average from 10 to 92 on
a host shared with another session, and still returned zero failures, which makes
this green harder to obtain than a quiet-box green rather than softer.

The build was a clean rebuild from a wiped directory, not an incremental one,
because the base advanced across `qwen3.h`, two Music3 headers and
`speech_api.h`, and an incremental green over a changed header is not a clean
green. `BUILD_EXIT=0` with zero errors and zero warnings; the 1879-line build log
contains no `No space left` and no `BFD assertion`, against 477 `Built target`
lines from the same grep. Disk 87 GB free, load 11.73 / 4.21 / 2.50 at build
time.

### 8.7 What was NOT done, and why

- **No DFR pipeline.** Out of scope by §2 and the reason §7 says what it says.
- **No bf16 arm.** This file is the CPU f32 reference arm; the bf16 arm is owed
  by phase L6 for the whole upsampler, spatial included, and this row does not
  change that debt either way.
- **The spatial+temporal arm stays refused.** It is a different operator, and
  implementing it because it is adjacent would have shipped an ungated third
  branch.
