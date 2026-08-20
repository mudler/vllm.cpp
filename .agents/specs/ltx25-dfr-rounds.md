# LTX25-DFR-ROUNDS — DFR's temporal refinement rounds, and the upsampler they drive

Issue: [#986](https://github.com/mudler/vllm.cpp/issues/986).
Campaign: [#644](https://github.com/mudler/vllm.cpp/issues/644).
Upstream pin: `Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
verified at `/home/mudler/_git/LTX-2` before any anchor below was read
(`git rev-parse HEAD` matched; `dfr_pipeline.py` is 630 lines and
`dfr_layout.py` is 213, both as recorded).

## 0. Honesty statement — what this row claims and what it does not

**0.1 This row lands one thing: the rounds loop.** Everything the loop needs
already exists. `dfr_layout.py` is ported in full and gated
(`ltx2_dfr.h`, `test_ltx2_dfr` 11/11), the generated keyframe slots are served,
the temporal x2 upsampler operator is ported and gated, and the `dfr` recipe is
reachable from `--pipeline-kind dfr`. What has no local counterpart is the
DENOISE PASS AS A CALLABLE, which is what the shipped refusal
(`ltx2_video.cpp:2018-2021`) names as the blocker. That is a refactor of the
render path, and this row does it.

**0.2 The row's purpose is a reachability answer, not a feature.**
`.agents/specs/ltx25-temporal-upsampler.md` §7 and `docs/FEATURES.md:172`
record the temporal x2 latent upsampler as gated and **UNDRIVEN**. Its only
upstream consumer is this loop. If this row lands, that cell moves; if it
cannot, the cell stays and the reason is recorded here.

**0.3 One recorded blocker turned out to be FALSE, and it was measured, not
assumed.** The shipped refusal at `ltx2_video.cpp:2022-2027`, issue #986 itself,
and `.agents/specs/ltx25-dfr-pipeline.md` §0.2 all state that
`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` is **not on the NAS**,
"re-verified 2026-08-16". **It is on the NAS.** Measured 2026-08-20:

```
$CHECKPOINT_ROOT/ltx-2.5/lightricks-ltx-2.5/latent_upscale_models/
  ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors    995778752  2026-08-12
  ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors   261944000  2026-08-17
```

**THE CANONICAL PREFIX IS `/usr/local/nas_share/checkpoints`**, which is what
`CHECKPOINT_ROOT` resolves to in the shared checkout's `.env`. The measurement
above was taken through `/mnt/nas_share/...`, where the same files are also
visible right now, and that path is the WRONG one to record: `.env` says in terms
that the NAS mounts at `/usr/local/nas_share` and NOT `/mnt`, because `/mnt` lives
on the ephemeral root overlay of the immutable OS and dies on every reboot. That
comment is dated 2026-08-16 and was written after a reboot left `CHECKPOINT_ROOT`
dangling and every checkpoint path in this campaign broken. Recording the
ephemeral path is how that happened; both are named here, and the durable one is
canonical.

Verified SEMANTICALLY rather than by name, because an arch name is not its
contents: the safetensors header is 7608 bytes, it declares 72 tensors, all
`BF16`, and `8 + header + data_end == 261944000` exactly — the file is complete
and self-consistent. Its `__metadata__.config` reads
`{"_class_name": "LatentUpsampler", "in_channels": 128, "mid_channels": 512,
"num_blocks_per_stage": 4, "dims": 3, "spatial_upsample": false,
"temporal_upsample": true, "spatial_scale": 1.0, "rational_resampler": true}` —
`spatial_upsample: false`, `temporal_upsample: true`, so it is the temporal arm
and not a renamed spatial one.

It landed on **2026-08-17**, one day AFTER the "re-verified 2026-08-16" the
refusal carries. Both records were true when written and are false now. This is
the "refusals go stale" risk `.agents/specs/ltx25-dfr-pipeline.md` §8 names,
firing on that row's own text. The refusal is corrected in this change rather
than inherited.

**0.4 What 0.3 does NOT unblock.** A real-weights DFR render is still
impossible, and for a DIFFERENT reason than the one the records give. DFR's
base needs a `keyframe_slot_sft` transformer (`dfr_pipeline.py:157`,
`DfrRecipe` at `ltx2_pipeline.cpp:1320`), and #1137 established by
authenticated reads at the pinned revision that no such checkpoint is
published: `Lightricks/LTX-2.5` ships five transformers, every one `dev` or
`distilled`. So the temporal upsampler now HAS real weights, and the pipeline
that would drive it still cannot load a real base. This row therefore claims a
REDUCED-DIMENSION fixture result and no real-weight result of any kind.

**0.5 No GPU.** A guided render holds `dgx:gpu0` and the fleet is contended, so
no lease is taken. Everything here is CPU. Nothing in this row needs a device.

## 1. The decision on the `keyframe_slot_sft` blocker, stated rather than worked around

Three options were considered.

1. **Loosen #1137's refusal.** REJECTED. It is a correctness gate that landed
   for a reason, and the row that landed it did the authenticated reads.
   Weakening a gate to let a test through is exactly the "never make a red gate
   green by deleting an assertion" case in `AGENTS.md`.
2. **Declare `keyframe_slot_sft` on a `dev`/`distilled` file.** REJECTED.
   `docs/USAGE.md:1528` already names this as the false declaration rather than
   the workaround, and the declaration is never checked against the header
   (`docs/FEATURES.md:175`), so the lie would be undetectable — which makes it
   worse, not safer.
3. **Gate on reduced-dimension synthetic fixtures through the production entry
   point.** TAKEN. This is how `test_ltx2_dfr` and every DFR case in
   `test_ltx2_video` already work (`test_ltx2_video.cpp:112` returns
   `keyframe_slot_sft` for the fixture it builds), and a fixture that declares
   the class it genuinely implements is not a false declaration.

Option 3 satisfies reachability on its own terms: `.agents/reachability.md`
asks whether a **production entry point** reaches the change, and
`LoadVideoEngine` + `VideoEngine::Generate` is that entry point regardless of
whose weights are loaded. The fixture is the checkpoint, not the caller.

**What stays owed: a real-weights DFR render**, blocked on a published
`keyframe_slot_sft` base. Recorded under `## Owed` against #986.

## 2. Upstream, with anchors — every one re-derived at `fd4ded7f`

The loop is `dfr_pipeline.py:402-529` and the frame contract is `:531-540`.
Step by step, with the line that decides each:

| # | Step | Anchor |
|---|---|---|
| 1 | round index runs `1..rounds`, `rounds in {0,1,2}` | `:402`, refused at `:284-285` |
| 2 | temporally x2-upsample the video latent | `:407` |
| 3 | `num_frames = 2 * (num_frames - 1) + 1` | `:408` |
| 4 | `current_fps = 2 * current_fps` | `:409` |
| 5 | seam positions scale x2; carried latents do NOT | `:410-412` |
| 6 | conditioning fps capped at 60.0 | `:414`, rationale `:74-78` |
| 7 | `tile_ranges(seams, num_frames, 2**round_idx)` | `:415` |
| 8 | per tile: `local_frames = (lat_end - lat_start - 1) * 8 + 1` | `:422` |
| 9 | per tile: slice `video_latent[:, :, lat_start:lat_end]` | `:423` |
| 10 | images re-attached TILE-LOCALLY, only those inside the window | `:428-437` |
| 11 | every seam in the window is a hard keyframe at strength 0.95 | `:453-468`, `:72` |
| 12 | mid-segment slots per tile, seeded from the tile's own latent | `:470-478` |
| 13 | ancestral Euler at eta 0.5, PER-TILE noise seed | `:480-499`, `:73` |
| 14 | stitch, and check `T == (num_frames-1)//8 + 1` | `:508-511` |
| 15 | dedupe repeated slots — the EARLIER tile wins | `:516-525` |
| 16 | merge carried anchors + this round's slots into the next bag | `:527-529` |
| 17 | trim to `(requested - 1) * 2**rounds + 1` | `:531-540` |
| 18 | `playback_fps = frame_rate * 2**rounds` | `:542` |
| 19 | audio cut to `num_frames / playback_fps` | `:552-560` |

Steps 7, 8, 12, 14, 15, 16 and 17 are **already ported and gated** in
`ltx2_dfr.h` as `Ltx2DfrTileRanges`, `Ltx2DfrStitchTileLatents`,
`Ltx2DfrSlotInitialsFromVideo`, `Ltx2DfrMergeCarryForwardKeyframes` and
`Ltx2DfrTargetFrames`. This row supplies their first production callers.

The three constants that exist today with **no reader** are steps 6, 11 and 13:
`kLtx2DfrMaxConditioningFps` (60.0), `kLtx2DfrAnchorKeyframeStrength` (0.95),
`kLtx2DfrTemporalAncestralEta` (0.5) — `ltx2_dfr.h:100-115`.

## 3. The seam, which is the actual work

Upstream's rounds invoke `self.stage(...)` — the same `DiffusionStage.__call__`
the two base stages use (`dfr_pipeline.py:480-499` against `:332` and `:375`),
with its own sigmas, stepper, seed, conditionings and initial latent, and with
`audio=None`.

This engine's equivalent is the body of the phase loop at
`ltx2_video.cpp:2959-4422`, which is inline and has no seam a tile can enter
through. **This row turns that body into one callable and calls it from three
places** — base stage, detail stage, and per tile — exactly as upstream does.
Writing a second denoise loop for tiles is the parallel path
`AGENTS.md` `## Shared seams` forbids, and it is not done here.

The invocation record mirrors `DiffusionStage.__call__`'s parameter list:
sigmas, stepper and eta, frames, fps, spatial downscale, initial latent, noise
scale, per-invocation noise seed, the extra conditionings the caller supplies,
whether audio runs, and where the result and the read-back slots go.

**This is a refactor of the shipped render path**, so every existing pipeline
kind runs through the new seam. The floor is that the ten recipes produce
byte-identical output before and after — proved by the existing suites rather
than asserted.

## 4. Scope

**In.**

1. The callable stage seam in `ltx2_video.cpp`, and the base and detail stages
   rewritten as its first two callers.
2. The rounds loop: steps 1-19 of §2, driving `Ltx2UpsampleVideoLatent` on the
   temporal arm.
3. A second upsampler slot: `temporal_upsampler_path`
   (`dfr_pipeline.py:177`, `:578-583`), since the engine holds one upsampler
   today and DFR needs both.
4. `temporal_upsample_rounds` served rather than refused, and
   `--temporal-upsample-rounds` / `--temporal-upsampler-path` on `ltx2-gen`.
5. Retire the two refusals this makes false — the rounds refusal
   (`ltx2_video.cpp:1995-2029`) and the "no phase consumes the temporal arm"
   clause (`:3077-3086`) — retired WITH their tests, per the #920 precedent
   recorded at `test_ltx2_video.cpp:1783-1793`, never widened.
6. Correct §0.3's stale NAS claim everywhere it is carried.
7. `docs/FEATURES.md` (the UNDRIVEN cell and the DFR row), `docs/USAGE.md`,
   `.agents/issue-index.md`, and the owning row.

**Out, refused by name or recorded as owed.**

- A real-weights DFR render — §1, no published `keyframe_slot_sft` base. Owed.
- The stage-2 detailing IC-LoRA — [#975](https://github.com/mudler/vllm.cpp/issues/975)
  owns the reference-latent arm. Already refused by name; left refused.
- The standalone single-frame slot decode — owed under #986 by the base row,
  and this loop does not reach it: rounds keep slots in latent space.
- A device arm for the rounds loop. The upsampler and the DiT keep whatever
  arm the load resolved; the loop adds no kernel.

## 5. Tests and evidence

Every engine case enters at `LoadVideoEngine` + `VideoEngine::Generate` — the
chain `vllm_video_generate` takes — never by constructing a type.

| # | Test | What it falsifies |
|---|---|---|
| T1 | `rounds=1` renders and returns `(requested-1)*2+1` frames | the loop does not run at all |
| T2 | `rounds=2` returns `(requested-1)*4+1` frames | the round count is ignored |
| T3 | `rounds>0` without `temporal_upsampler_path` is refused by name | a silent no-op round |
| T4 | the temporal upsampler is INVOKED, counted on the trace | the loop runs without upsampling |
| T5 | tile count per round is `2**round` | the canvas is denoised whole |
| T6 | the per-tile ancestral seed DIFFERS per tile | byte-identical noise per tile (`:496-498`) |
| T7 | conditioning fps is capped at 60 while playback fps is not | the RoPE time base defect at `:74-78` |
| T8 | `rounds=0` output is byte-identical to today's | the refactor moved the base path |
| T9 | the ten non-DFR recipes are unchanged through the new seam | the refactor moved every pipeline |

**T8 and T9 are the refactor's floor and matter more than T1-T7.** A rounds
loop that works while quietly changing every other pipeline's output is a worse
outcome than no rounds loop.

**The reachability mutation is the headline evidence.** Delete the production
call site where the rounds loop invokes `Ltx2UpsampleVideoLatent`, rerun the
focused gate, record the RED. A green gate there means the loop is a test-only
driver and the UNDRIVEN cell must not move.

**Mutations owed**, each with four facts printed or it does not count:
`BUILT=YES/NO`, the `': error:'` count, `git diff --stat` proving the mutation
applied, and the failing assertion by name. A mutation that fails to build reads
exactly like a passing test, and a mutation that never applied reads the same.
Restore with an explicit `touch`, because `cp -p` restores the mtime and ninja
then skips the rebuild while the harness prints `BUILT=YES`.

| # | Mutation | Predicted signature |
|---|---|---|
| M1 | the temporal upsampler call site deleted | THE reachability proof |
| M2 | `num_frames = 2*num_frames` instead of `2*(num_frames-1)+1` | frame count off by one per round |
| M3 | seam positions not scaled x2 between rounds | anchors land at half their time |
| M4 | the 60 fps conditioning cap removed | motion spike per latent border |
| M5 | the per-tile seed made shared | identical noise in every tile |
| M6 | anchor strength 1.0 instead of 0.95 | the seam frame cannot settle |
| M7 | ancestral eta 0.0 instead of 0.5 | plain Euler; densification collapses |
| M8 | slot dedupe keeps the LATER tile | lead-in slots overwrite settled ones |
| M9 | `2**round` tiles replaced by a fixed 2 | round 2 denoises the wrong windows |

## 6. Risks

- **The refactor is the risk, not the loop.** The phase body is ~1460 lines and
  every pipeline runs through it. T8/T9 are the control and are run before any
  rounds work is believed.
- **`docs/FEATURES.md` is a keyed record with `MAX_CELL_CHARS = 220`** and the
  LTX-2.5 cells sit at it. Reapply by KEY, prove unrelated keys byte-identical,
  and trim only this row's own wording.
- **Two tests go red BY DESIGN** when the refusals retire
  (`test_ltx2_video.cpp:2262` and the `ltx2_dfr.h` symbol re-derivation at
  `:2379-2400`). They are retired with the refusal they describe, and each
  assertion is replaced by one about what replaced it.
- **The four symbols `Ltx2DfrResolveCanvas`, `Ltx2DfrTileRanges`,
  `Ltx2DfrStitchTileLatents`, `Ltx2DfrMergeCarryForwardKeyframes` are
  load-bearing for a refusal message** (`test_ltx2_video.cpp:2390-2398`).
  Renaming any of them reds that case for an unrelated reason.
- **`origin/main` moves constantly.** Merge often and diff the merge base.
- **Anchors decay.** Issue #986 cites `docs/FEATURES.md:166` and
  `.agents/specs/ltx25-resolution-envelope.md:436`; both had already moved to
  `:172` and `:496` when this row read them. Every anchor here was re-derived
  at this tree, and they are re-derived again at the final tree.

## 7. Reachability — the sentence the records must carry

Written before implementation as a target; `## Outcome` states what was reached.

> The LTX-2.5 temporal x2 latent upsampler is **driven** by the DFR pipeline's
> rounds loop, reachable from `vllm_video_generate` with `pipeline_kind=dfr`
> and `temporal_upsample_rounds > 0`, and from `ltx2-gen` on the same flags. It
> is gated at REDUCED DIMENSIONS against a fixture checkpoint. The real
> `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` IS on the NAS as of
> 2026-08-17, but no `keyframe_slot_sft` base is published, so DFR cannot load
> real weights and no real-weight temporal result exists or is implied.

Nothing in `docs/` may say more than that.

## 8. Gates

CPU only; no GPU lease is taken (§0.5).

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

`test_ltx2_video` and `test_capi` run SERIALLY — they contend over `/tmp`
fixture directories. Known-environmental and proved pre-existing rather than
asserted: `windows-msvc-*`, `test_engine_core_proc`/`test_async_llm` under
`ctest -j`, `test_cpu_x86_llamacpp_floor` (#618), `test_nemotron_h_paged_forward`
(#1371). Disk is checked before any failure is attributed to the diff, because
ENOSPC surfaces here as a false POLICY REFUSAL.

## 9. Arms

NVFP4, FP8 and bf16 all reach the loop, because the loop adds no kernel and
inherits whatever arm the load resolved. GGUF is not applicable to this family.
Any arm that cannot run refuses by name rather than falling back.

## Outcome

**The rounds loop landed and the temporal x2 latent upsampler is DRIVEN.** §7's
sentence holds as written, with §0.3's correction folded in.

### What was built

The stage seam is one lambda, `RunStage`, holding the body that was inline in the
per-phase loop. It takes an invocation record mirroring
`DiffusionStage.__call__`'s parameter list — phase, phase index, frames,
conditioning fps, an optional initial latent, whether audio runs, this
invocation's slot positions, and its seam anchors — and the recipe's phases became
its first caller. The rounds loop is its second, once per tile.

The four values the body varies on are bound as SHADOWING locals of the same
names rather than renamed at every use. That was a deliberate choice about the
diff: renaming would have touched a thousand lines of a file whose reader anchors
are gated, and shadowing keeps the base-stage path textually identical to what it
was. It is recorded here because it reads as an accident otherwise.

Three helpers were added, and only one of them was in the plan.
`Ltx2DfrSliceLatentFrames` and `Ltx2DfrConcatLatentFrames` exist because the
first draft of the slot merge did a FLAT `data.insert`, which is wrong for a
`[B, C, T, H, W]` volume with more than one channel: it produces exactly the
right number of floats with the channels interleaved wrong.
`Ltx2DfrStitchTileLatents` already carried a comment about that exact trap, and
this row walked into it anyway three functions later — which is the argument for
named helpers rather than a fourth inline loop.

### The video-only path, which the plan did not anticipate

Upstream's tile call passes `audio=None` (`dfr_pipeline.py:494`), and this engine
had no way to express that: every phase built and denoised both streams. Four
places needed the guard — the audio latent shape, the state build, the Gaussian
noiser, and the stepper — and a fifth is the one that matters:
`audio_latent_volume` must not be OVERWRITTEN by a tile, or the render keeps a
correctly shaped video beside no soundtrack and every check on the video passes.

`Ltx2DitForward` already resolved `audio_tokens = 0` for a null stream, so the
DiT needed no change. A ZERO-TOKEN stream is not a substitute for a null one:
`vt::Tensor` refuses a non-positive dimension by name, which is the right
polarity everywhere else and cost two build-and-run cycles to find here.

### Measured, and it changed the row

**The temporal upsampler checkpoint is on the NAS** (§0.3), which falsifies one
of the four causes the shipped refusal named, and one claim in
`ltx25-temporal-upsampler.md` §7. It arrived 2026-08-17, one day after the
"re-verified 2026-08-16" those records carry. Both were true when written. What
made this findable is the DATE beside the measurement, and that is the
transferable lesson rather than the checkpoint.

**The tile count is CLAMPED, and this row's first test expectation was wrong
about it.** `tile_ranges` takes `min(num_tiles, n_segments)`
(`dfr_layout.py:171`). The 9-frame fixture pads to a 25-frame canvas with ONE
segment, so round 1 asks for 2 windows and gets 1, and round 2 asks for 4 and
gets 2. The assertion was written from the issue's `2**round` without reading the
clamp beside it, and the RED that resulted was the code being right. Corrected to
the derived values, with the bound stated in the test: **this fixture cannot
exercise the unclamped tiling and nothing here gates it.**

### The reachability proof

`M1`, the mutation this row exists for: delete the `Ltx2UpsampleVideoLatent` call
in the rounds loop and replace it with a resize to the shape the round expects.
`BUILT=YES`, `compile_err=0`, `git diff --stat` one line, binary sha changed.

**RED — and only on the reachability counter.** 33 of 35 assertions still passed,
including `frame_count == 17` and `frame_count == 33`. The clip comes out the
right length, at the right size, exit 0. `temporal_upsample_calls` is the only
observable that separates "the rounds ran" from "the rounds drove the upsampler",
which is the whole argument for a dedicated counter over an inference.

### Mutations

Each with four facts printed, each restored byte-exact (sha256 compared against
the pre-mutation copy, and `touch`ed so ninja could not skip the rebuild).

| # | Mutation | Result |
|---|---|---|
| M1 | temporal upsampler call site deleted | RED — `temporal_upsample_calls`; 33/35 still green |
| M2 | `frames = 2 * frames` for `2 * (frames - 1) + 1` | RED — the layout's own seam guard, `dfr_layout.py:153-154` |
| M4 | the 60 fps conditioning cap removed | RED — `round_conditioning_fps[1]`; 34/35 still green |
| M5 | `+ tile_index` dropped from the tile seed | RED — `round_tile_seeds[2] == 2008u` and the `!=` beside it |
| M6 | anchor strength 0.95 to 1.0 | RED — `round_anchor_strengths[i] == Approx(0.95)` |
| M7 | ancestral eta 0.5 to 0.0 | RED — `round_stepper_eta[i] == Approx(0.5)` |
| M8 | slot dedupe first-wins to last-wins | RED — `round_merged_slot_tiles[1] == 0` |

**M5 through M8 were NOT run when this row first landed, and that was a gap
rather than a pass.** It is now closed. The original text said the fixture had
"no assertion that could detect them", and running a mutation with no instrument
on it would have produced a green that reads like coverage — which is exactly
what the fresh review of [#1481](https://github.com/mudler/vllm.cpp/pull/1481)
then measured: it mutated all four and the full suite stayed green at 102 cases /
4141 assertions. That is unasserted-live code rather than dead code, and the same
review proved both paths execute, by throwing from inside the anchor loop and
from a repeated slot position across tiles.

The four are gated by the subcase *the four PER-TILE guarantees, none of which
the rendered clip can show*, which records each value AT THE POINT THAT CONSUMES
IT — the seed off the `SplitMixGaussian` the loop generator is constructed with,
the strength off the `Ltx2ConditionVideoByKeyframe` argument, the eta inside the
ancestral branch on the step it takes. A tile that fell onto the deterministic
Euler arm therefore records no eta at all, and the size check is red before any
value is compared.

**M8 needed a different observable than the finding named.**
`round_merged_slot_positions` cannot detect the dedupe order: first-wins and
last-wins each keep exactly one entry per position and both sort the bag, so the
position list, its length, the carry-forward count and every downstream shape are
byte-for-byte identical. The winning TILE is the only thing the rule moves. The
M8 run is the proof — with last-wins in the tree, the three position CHECKs stay
GREEN and only `round_merged_slot_tiles[1]` reds, 63 of 64 assertions passing.
`round_slots_emitted` carries the other half: it says how many slots the tiles
produced BEFORE the merge, so a reader can see that a duplicate genuinely
occurred rather than take a winning-tile sequence on faith. This fixture emits 4
and keeps 3.

Observed rather than derived, and named as such: the seeds are 1007, 2007, 2008,
so `round_idx` is ONE-BASED here (`seed 7 + 1000 * round + tile`). The merged bag
is positions 24, 24, 72 with winning tiles 0, 0, 1 — round 1's single slot, then
round 2's tile 0 slot at 24 that its tile 1 re-emits as a lead-in, plus tile 1's
own at 72. The anchor-strength count is a FLOOR (`>= tiles`) rather than an
equality, because every tile pins at least its own local frame 0
(`dfr_pipeline.py:453-468`) and the exact total is a property of this fixture's
tile ranges, not of upstream.

### Gates

CPU only, no GPU lease (§0.5). `Release`, `VLLM_CPP_CUDA=OFF`, `compile_err=0`.

- `ctest -j4 -E "test_ltx2_video|test_capi"` — **572/572 passed, 0 failed**, 3
  skipped (`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`,
  `test_qwen35_paged_engine`), 206.83 s.
- `ctest -R "test_ltx2_video|test_capi"` serially — **2/2 passed**, 59.12 s.
- `test_ltx2_video` alone: **102 cases / 4141 assertions / 0 failed**, exit 0.
- `scripts/agent-preflight.sh` — exit 0.

**Re-gated 2026-08-20 after the review repair, on the merge of `origin/main`
`aee6c48d6`** (`Release`, `VLLM_CPP_CUDA=OFF`, full `ninja` with 0 `FAILED:`
lines and `compile_err=0`):

- `ctest -j4 -E "test_ltx2_video|test_capi"` — **572/572 passed, 0 failed**, the
  same 3 skipped, 48.21 s.
- `ctest -R "test_ltx2_video|test_capi"` serially — **2/2 passed**, 30.91 s, at
  loadavg 9.9.
- `test_ltx2_video` alone: **102 cases / 4170 assertions / 0 failed**, exit 0.
  The 29 new assertions are the four per-tile guarantees; the case count is
  unchanged at 102 because the new subcase joins an existing `TEST_CASE`.
- `scripts/agent-preflight.sh` and `--staged` — exit 0.

**One test in this file is load-dependent and it is not this row's**, filed as
[#1494](https://github.com/mudler/vllm.cpp/issues/1494): `ltx2 video: the three
carrying phases contain their work and the load keeps its order` compares a
95% wall-clock coverage ratio at `test_ltx2_video.cpp:3696`, sibling to the one
[#1439](https://github.com/mudler/vllm.cpp/issues/1439) already tracks at
`:3259`. Five consecutive runs of ONE binary with no source change, at loadavg 17
to 19, read 99.28%, 95.46%, **94.90% RED**, 98.77% and 98.35% — four green and
one red. The polarity is the instrument, not the code: the three runs that passed
comfortably are the runs where `denoise` took 0.037 s, 0.025 s and 0.016 s, and
the run that reded is the one where it took 0.0049 s. The un-named residue is
about 0.00025 s in every one of them, so the RATIO moves with the leaf while the
residue does not, and at fixture scale the floor sits inside the scatter.

**The refactor's floor is the 101 pre-existing cases in `test_ltx2_video`, not
the 6 new ones.** Every pipeline kind now runs through the new seam, and all of
them are unchanged in what they assert. Two record gates went red and were
repaired rather than widened: the derived READER ANCHORS in `ltx2_video.cpp` (the
insertions moved all fourteen) and the served-load-extras list (the new
`temporal_upsampler_path`).

### What was rejected

- **Loosening #1137's `keyframe_slot_sft` refusal**, and **declaring the class on
  a `dev` file** — §1, both rejected in writing with the reason beside each.
- **A second denoise loop for tiles.** It would have been a smaller diff, and it
  is the parallel path `AGENTS.md` `## Shared seams` forbids. The seam is the
  reason the base stages and the tiles cannot diverge.
- **Reusing `upsampler_path` for both arms.** DFR holds two upsamplers at once
  upstream, so one slot would have made the arms mutually exclusive.

## Owed

- [#986](https://github.com/mudler/vllm.cpp/issues/986) — a **real-weights DFR
  render**, blocked on a published `keyframe_slot_sft` base (§0.4, §1). Not a
  code gap: nothing in this tree can supply it.
- [#986](https://github.com/mudler/vllm.cpp/issues/986) — the standalone
  single-frame decode of a generated keyframe slot, carried forward from the
  base row. The rounds loop keeps its slots in latent space and does not reach it.
- [#975](https://github.com/mudler/vllm.cpp/issues/975) — DFR's stage-2 x2
  spatial detailing IC-LoRA. Refused by name, pointing at #975.
- [#1493](https://github.com/mudler/vllm.cpp/issues/1493) — the **unclamped
  `2**round_idx` tiling**, which nothing in this tree exercises because every
  fixture canvas here has one keyframe segment and the tiles clamp to 1 and 2
  (`dfr_layout.py:171`). Disclosed in the test body and in the public LTX-2.5
  guide since this row landed — `docs/USAGE.md` until #1491 moved the model
  recipes to `docs/models/ltx-2-5.md`; owned here since the fresh review of #1481. Left open on
  purpose: reaching 4 segments needs a materially longer canvas and round 2 would
  then denoise 4 tiles on a canvas already doubled twice, which is a new fixture
  and a substantially longer CPU run rather than an assertion added to the
  existing render. §5 states what would close it.

The four per-tile guarantees that this section used to owe — the seed, the anchor
strength, the eta and the dedupe order — are **no longer owed**. They are gated
by the subcase named under `### Mutations`, and M5 through M8 each red by name.
Honesty in prose was not ownership, so they are listed here as closed rather than
silently dropped.

## Stop conditions

- If the base and detail stages cannot be expressed through one callable seam
  without changing their output, STOP and report. T8/T9 are the control and a
  failure there is a finding about the refactor, not a test to adjust.
- If the rounds loop cannot reach the temporal upsampler from a production entry
  point, STOP and report. Landing it unreached repeats the exact defect this row
  exists to retire.
- If a refusal this row lifts turns out to have another live cause, keep the
  refusal and retarget it rather than deleting it.

## Now

`ACTIVE`. Spec committed; implementation follows on `row/LTX25-DFR`.
