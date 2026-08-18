# LTX25-DFR-PIPELINE — detail-fidelity rendering, and the upsampler it still does not drive

Row: `LTX25-DFR-PIPELINE`. Issue:
[#986](https://github.com/mudler/vllm.cpp/issues/986). Campaign:
[#644](https://github.com/mudler/vllm.cpp/issues/644).

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`, `packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Verified at the local checkout `/home/mudler/_git/LTX-2` — `git rev-parse HEAD`
returns that SHA — **before any anchor below was read**.

---

## 0. Honesty statement — what this row claims and what it does not

Written before implementation, so neither half can be discovered later as a claim
the row did not support.

1. **The temporal upsampler was INTENDED to become driven, and it did not.**
   This bullet is left standing with its correction attached rather than
   rewritten, because the whole purpose of a section 0 is that a later reader can
   see what the row set out to claim.
   *As written before implementation:* "`DFRPipeline`'s rounds loop is upstream's
   only consumer of a temporally-configured `LatentUpsampler`
   (`dfr_pipeline.py:235-245, 402-407`), and this row ports that loop and reaches
   it from a production entry point. Section 7 states the sentence the records
   will carry."
   *What happened:* the rounds loop is **not** ported. It needs the per-tile
   denoise pass as a callable, which this engine does not have — its denoise is
   inline inside one 680-line per-phase loop — and extracting it is a refactor of
   the render path rather than an addition to it. The loop is refused by name and
   owed by #986, the temporal upsampler stays UNDRIVEN, and `docs/FEATURES.md`
   is UNCHANGED on that point. §12.1 carries the sentence the records actually
   take, and §12.2 states the scope decision rather than leaving it silent.
   What DID land is the DFR base: the padded keyframe canvas, the generated
   keyframe slots and their readback, and the trim back to the caller's count.
2. **There is no real-weight temporal result and there will not be one here.**
   `/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/latent_upscale_models/`
   holds `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` and nothing
   else. Re-verified 2026-08-16 rather than inherited from
   [`ltx25-temporal-upsampler.md`](ltx25-temporal-upsampler.md) §8.5, because that
   measurement is four days old and the NAS is a shared mount. The temporal
   checkpoint named at `ltx-pipelines/docs/pipelines.md:176` is **absent**, so
   every temporal claim here is a reduced-dimension fixture claim.
3. **The GPU is not used.** A render ladder holds `dgx.casa` and the box
   OOM-reboots when the 119 GiB unified pool is exhausted. CPU gate only.
4. **Upstream ships no tests.** `find /home/mudler/_git/LTX-2 -name 'test_*.py'`
   returns **0**. `AGENTS.md` `## vLLM is the reference` requires porting
   upstream's tests in the same change; there are none to port, so the tests here
   are written against upstream ANCHORS, and §6 says what that costs and what
   guards it.

## 1. What is wrong today — measured at `0e1bee42f`

`git grep -n -i dfr -- src include tests examples docs` returns **zero product
hits**. The name appears only in records that say it is absent:
`docs/USAGE.md:791` ("`DFRPipeline`'s multi-round loop, which is not ported"),
`include/vllm/model_executor/models/ltx2_upsampler.h:30`, and
[`ltx25-temporal-upsampler.md`](ltx25-temporal-upsampler.md) §0.1, §7, §8.7. The
positive control for that grep is `git grep -c -i ltx2 -- src`, which returns 87
hits in `ltx2.cpp` alone — so the empty result is a measurement and not a
mistyped path ([#604](https://github.com/mudler/vllm.cpp/issues/604)).

`docs/FEATURES.md:166` carries `Temporal x2 ups gated, UNDRIVEN`. That cell is
**load-bearing**: `.agents/specs/ltx25-resolution-envelope.md:436` asserts it is
untouched byte for byte. This row contradicts it deliberately, by key, and §9
records which row it is contradicting.

## 2. The dependency that decides the row's shape

DFR is not an independent pipeline. Its stage 1 appends
`VideoGeneratedKeyframeSlots` (`dfr_pipeline.py:330`) and then reads
`video_state.generated_keyframes` back (`:346-348`), raising
`RuntimeError("Stage 1 did not return generated_keyframes despite requesting
slots")` if it is absent. That readback is what
[#920](https://github.com/mudler/vllm.cpp/issues/920) refused by name.

So this row cannot be done without lifting that refusal, and lifting it is not a
side effect to be performed quietly. Three things follow.

**(a) The refusal's blocker was correctly identified and this row pays it.**
[`ltx25-generated-keyframes.md`](ltx25-generated-keyframes.md) `## 4` names one
blocker — "readback with a standalone decode" — and `## Owed` names three
pieces against #920. #920 is **CLOSED**, so the debt currently has a spec bullet
and no open issue; #986 picks it up.

**(b) DFR needs TWO of those three pieces and not the third, and the difference
is a fact about DFR rather than a convenience.** The refusal says each slot frame
"must then be decoded as a STANDALONE one-frame clip — a K-frame causal decode
would blend slots that were never temporally adjacent". True, and it is a
constraint on a caller who wants the slot PIXELS. **DFR never decodes its slots.**
It hands them to the spatial latent upsampler (`dfr_pipeline.py:348`) and feeds
them straight back as `initial_keyframes` (`:364`), and in the temporal rounds it
carries them as latents through `_merge_carry_forward_keyframes` (`:527-529`).
They stay in latent space for the whole pipeline. So DFR needs the LAYOUT and the
EXTRACTION; the standalone decode belongs to a slot-output surface DFR does not
have, and it stays owed under §11 rather than being written unreached.

**(c) The #920 refusal carries a tripwire aimed at this row, and it must be
allowed to fire.** Its message declares `ABSENT HERE: GeneratedKeyframe,
generated_keyframe`, and `test_ltx2_video` parses those out of the thrown message
and re-derives them against `ltx2_conditioning.h`'s declarations with comment
lines stripped. §4a of that spec is explicit about what happens next: *"If the
readback lands, `GeneratedKeyframe` appears in the header, ABSENT goes red, and
whoever landed it is told the refusal is now false."* This row lands exactly
that. The RED is the instrument working, and the repair is to retire the refusal,
**never** to widen or delete the assertion — `AGENTS.md` `## Changing the rules or
a checker` forbids making a red gate green by deleting an assertion.

## 3. Upstream, with anchors

Every `file:line` re-derived at the pin. The needle is derived from the CLAIM
rather than read back out of the cited span, because that check is circular.

### 3.1 `dfr_layout.py` — the canvas

| Upstream | What it decides |
|---|---|
| `SEGMENT_CANDIDATES = (24, 32)` (`:12`) | the two keyframe segment lengths |
| `TILE_LEAD_SEGMENTS = 1` (`:18`) | the lead-in every non-first tile denoises through |
| `TileRange` (`:21-38`) | pixel/latent bounds, anchor and slot bags, `drop_latent_prefix` |
| `choose_segment_length` (`:40-57`) | least pad; **ties keep the LARGER** |
| `resolve_canvas` (`:60-81`) | pads `num_frames - 1` to a multiple of S; positions `[S, 2S, ..., N'-1]`, frame 0 EXCLUDED |
| `pixel_to_latent_index` (`:84-90`) | 0 is legal; anything else must sit on the x8 border |
| `_owned_segment_counts` (`:93-100`) | contiguous owned runs, **largest first** |
| `_build_tile` (`:103-134`) | the window, the two keyframe bags, the prefix drop |
| `tile_ranges` (`:137-182`) | gapless partition; `num_tiles` CLAMPED to the segment count |
| `stitch_tile_latents` (`:185-208`) | each tile contributes `latent[drop_latent_prefix:]` |
| `remap_positions_to_local` (`:211-213`) | global to tile-local |

### 3.2 `dfr_pipeline.py` — the driver

| Upstream | What it is |
|---|---|
| `_ANCHOR_KEYFRAME_STRENGTH = 0.95` (`:72`) | carried anchors pinned just short of clean |
| `_TEMPORAL_ANCESTRAL_ETA = 0.5` (`:73`) | **not** the distilled 1.0 |
| `_MAX_CONDITIONING_FPS = 60.0` (`:78`) | RoPE time base cap; playback fps is separate |
| `_keyframe_conditionings_from_latents` (`:81-98`) | anchors as `VideoConditionByKeyframeIndex` |
| `_slot_initials_from_video` (`:101-111`) | nearest latent frame per slot; `round`, not floor |
| `_merge_carry_forward_keyframes` (`:114-139`) | anchors then slots, keyed by position, slot wins |
| `_detailing_downscale_factor` (`:142-152`) | LoRA metadata, default **2** |
| stage 1 (`:317-342`) | half res, `DISTILLED_SIGMAS`, slots at `positions` |
| stage 2 (`:351-394`) | full res, `STAGE_2_DISTILLED_SIGMAS`, upsampled slot seeds, reference latent when a detailing LoRA is present |
| rounds (`:402-529`) | temporal x2, tiles, ancestral Euler, stitch, carry forward |
| the trim (`:531-540`) | `(requested - 1) * 2**rounds + 1` |
| audio (`:552-560`) | shipped from STAGE 1, cut to the video's duration |

Three of those are worth stating as behaviour rather than as a row in a table,
because each renders something plausible when it is got wrong:

- **fps doubles per round and the CONDITIONING fps is capped separately**
  (`:409`, `:414`). Upstream's reason is a statement about RoPE, not a safety
  margin, and it is quoted in `ltx2_dfr.h` beside the constant.
- **each tile gets its own ancestral noise seed**, `seed + 1000 * round + tile`
  (`:498`), with upstream's own reason: *"Tiles are positionally identical, so a
  shared ancestral seed would inject byte-identical noise into every one of
  them."*
- **image conditioning is TILE-LOCAL** (`:428-437`). `frame_idx=0` means the
  tile's first frame, so re-applying the opening image on a non-first tile would
  pin the wrong frame onto the seam. Only images that fall inside the window are
  re-attached, with their indices shifted.

### 3.3 `keyframe_slots.py` and the readback

`VideoGeneratedKeyframeSlots.__init__` (`:47-69`) and `apply_to` (`:71-150`);
`GeneratedKeyframeLayout` (`ltx_core/types.py:220-247`);
`LatentTools.clear_conditioning` (`ltx_core/tools.py:88-117`) and
`extract_generated_keyframes` (`:203-230`).

The four things `apply_to` does that a shape check cannot see:

1. `denoise_mask = 1` on the new tokens (`:118-119`), so the noiser lerps from
   the slot `latent` and **ignores** `clean_latent`. A port that seeded
   `clean_latent` instead would produce a finite, correctly shaped, unconditioned
   slot.
2. `extend_keyframes_mask(..., marked=True)` (`:121`) — upstream's ONLY marked
   call site.
3. the slot's temporal span is exactly `[t, t+1)` with `causal_fix=False`
   (`:152-174`), because the span is set explicitly. Applying the causal fix as
   well would shift every slot.
4. `GeneratedKeyframeLayout` records `first_token` (`:143-147`) so the slots are
   located exactly rather than assumed to trail — items are applied in list order
   and each appends.

## 4. Scope

**In.**

1. `ltx2_dfr.{h,cpp}` — `dfr_layout.py` in full, plus the three `dfr_pipeline.py`
   helpers that are pure index arithmetic (`_slot_initials_from_video`,
   `_merge_carry_forward_keyframes`, the target-frames trim).
2. Generated keyframe slots in `ltx2_conditioning.{h,cpp}`:
   `Ltx2GeneratedKeyframeLayout`, `Ltx2ConditionVideoByGeneratedKeyframeSlots`,
   `Ltx2ExtractGeneratedKeyframes`, and `Ltx2ClearConditioning` extracting BEFORE
   it trims.
3. A `dfr` pipeline kind in `ResolveLtx2PipelineRecipe`, and the DFR driver in
   `ltx2_video.cpp`: canvas resolution, slot conditioning per stage, the slot
   carry between stages, and the temporal rounds loop.
4. Retire the #920 refusal, which has become false, and serve the key.
5. `docs/FEATURES.md` (the UNDRIVEN cell), `docs/USAGE.md`, `.agents/issue-index.md`.

**Out, refused by name or recorded as owed rather than approximated.**

- The standalone single-frame decode of a slot — §2(b). Owed, §11.
- `_detailing_downscale_factor`'s LoRA-metadata read: #923 landed the metadata
  reader, and the arm that consumes it is the reference-latent conditioning that
  [#975](https://github.com/mudler/vllm.cpp/issues/975) still owes for two
  independent reasons. DFR's detailing IC-LoRA is therefore **refused by name**
  rather than half-built, and the refusal points at #975.
- Multi-GPU DFR (`dfr_pipeline.py` has no `_mgpu` sibling; the marker
  `kMultiGpuParallelism` already covers the family).
- Any bf16 device arm of the layout: it is integer index arithmetic and has no
  dtype.

## 5. Design

**One new translation unit for the layout.** `ltx2_dfr.cpp` holds no weights,
runs no kernel and reads no checkpoint. It is separate because every one of its
failure modes produces a correctly shaped, finite, plausible latent, so the only
instrument that can catch them is an exact gate on the indices — which is a
different kind of test from everything in `ltx2_pipeline.cpp`.

**The rounds loop reuses the phase body rather than duplicating it.** Upstream's
tile denoise is `self.stage(...)` — the same `DiffusionStage.__call__` the two
stages use, with a different stepper, a different sigma set and a tile-local
conditioning list. Mirroring that means the engine's per-phase denoise becomes a
callable the rounds loop can invoke per tile. Writing a second denoise loop by
hand is the parallel path `AGENTS.md` `## Shared seams` forbids.

**The DFR recipe is `distilled_two_stage`'s two phases with DFR's names.** Both
use `DISTILLED_SIGMAS` then `STAGE_2_DISTILLED_SIGMAS`
(`dfr_pipeline.py:281-282`), stage 1 at `width // 2` (`:319`), stage 2 with the
spatial upsample and `noise_scale = stage_2_sigmas[0]` (`:386`). That is the
shipped `DistilledTwoStageRecipe` exactly, which is a finding rather than a
shortcut: DFR differs from the distilled two-stage pipeline in its CONDITIONING
and its rounds, not in its schedule. Recorded so a reader does not go looking for
a DFR-specific sigma set that upstream does not have.

## 6. Tests and evidence

Upstream ships no tests (§0.4), so every case here is written against an anchor
and the standing trap applies: **a test asserting only upstream symbol names
cannot detect local staleness** — measured on this campaign at
[`ltx25-generated-keyframes.md`](ltx25-generated-keyframes.md) §4a, where mutation
M7 left a suite at 18/18 exit 0 while the message it checked had become false. So
the layout suite pins LOCAL facts: exact index vectors, exact `drop_latent_prefix`
values, and a stitch whose output is compared element-wise against a
hand-constructed expectation, not against a property.

Every engine-level case enters through the **production entry point** —
`vllm::multimodal::LoadVideoEngine` then `VideoEngine::Generate`, the chain
`vllm_video_generate` takes — never by constructing a type
([`reachability.md`](../reachability.md)).

**The reachability mutation is the row's headline evidence.** Delete the
production call site where the DFR driver invokes the temporal upsampler, rerun
the focused gate, and record the RED. A green gate there would mean the rounds
loop is a test-only driver and the UNDRIVEN cell should not move.

**Mutations owed**, each recorded with three facts — `git diff --stat` after
applying, whether it BUILT with the compile-error count beside it, and the exit
code. A non-building mutation establishes nothing.

| # | Mutation | Predicted signature |
|---|---|---|
| M1 | `choose_segment_length` tie takes the SMALLER candidate | wrong positions on a tied frame count |
| M2 | `_build_tile` keeps boundary 0 in `anchor_kf_global` | an anchor with no latent in the carry bag |
| M3 | `drop_latent_prefix` omits the `+1` for `own_lo > 0` | stitched T off by one per seam |
| M4 | the first tile is given a lead-in | head of the clip truncated |
| M5 | slots seed `clean_latent` instead of `latent` | slot content unconditioned, shape unchanged |
| M6 | `extend_keyframes_mask` called with `marked=false` | the trained marker misses every slot |
| M7 | the slot span uses `causal_fix=true` | every slot shifted in RoPE time |
| M8 | `extract_generated_keyframes` assumes the slots TRAIL | wrong tokens read when a keyframe also appended |
| M9 | **the temporal upsampler call site deleted** | the reachability proof |
| M10 | the per-tile ancestral seed made shared | byte-identical noise per tile |

## 7. Reachability — the sentence the records must carry

Written before implementation as a target; §10 states what was actually reached.

> The LTX-2.5 temporal x2 latent upsampler is **driven** by the DFR pipeline's
> rounds loop, reachable from `vllm_video_generate` with `pipeline_kind=dfr` and
> a positive rounds count, and from `ltx2-gen` on the same two flags. It is gated
> at REDUCED DIMENSIONS against a fixture checkpoint. The real
> `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` is not on the NAS,
> so no real-weight temporal result exists and none is implied.

Nothing in `docs/` may say more than that.

## 8. Risks

- **`docs/FEATURES.md` is a keyed record and `MAX_CELL_CHARS = 220` binds**, with
  the LTX-2.5 cells at or near it. Reapply the scoped edit BY KEY, prove unrelated
  keys byte-identical, and trim only this row's own wording — never a measurement
  or a host qualifier.
- **`ltx2_video.cpp` carries derived READER ANCHORS** gated by `test_ltx2_video`.
  A clean merge will not warn when they go stale. Re-derive at the final tree.
- **`origin/main` moves.** [#983](https://github.com/mudler/vllm.cpp/pull/983) is
  open against `ltx2_video.cpp` (the MSVC C4244 cast for #968). Merge often and
  diff the merge base, never the moving ref.
- **The #920 tripwire fires by design** (§2c). The repair is retirement, not a
  widened assertion.
- **Refusals go stale.** `ltx2_video.cpp:1379-1380` keeps a tally of refusals
  whose stated reason turned out false. Every cause this row writes or lifts is
  re-derived at the merge tree, and ruled-out causes take the
  `WHAT IS *NOT* THE REASON` shape.

## 9. The record this row expected to contradict, and did not

*Written before implementation:* `.agents/specs/ltx25-resolution-envelope.md:436`
records, as evidence for that row, that *"The 'Temporal x2 ups gated, UNDRIVEN'
cell is untouched, byte for byte."* This row expected to make that false.

**It did not, and the prediction is left here rather than deleted.** The rounds
loop is out of scope (§12.2), so the UNDRIVEN cell is still accurate and this row
does not touch it. `LTX25-RESOLUTION-ENVELOPE`
([#919](https://github.com/mudler/vllm.cpp/issues/919), merged as `e5351776c`)
keeps its evidence intact.

What this row DOES change in `docs/FEATURES.md` is a different, adjacent claim in
the same row: the LTX-2.5 cell said `GENkf/DiffVAE/ref refused`, and generated
keyframes are now served, so `GENkf/` is removed and a new per-feature row
carries the DFR base. The edit is scoped by key, six characters long, and leaves
every other cell byte-identical — including the UNDRIVEN one.

## 10. Gates

CPU-only; the GPU is not used (§0.3).

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Known-red and to be proved pre-existing rather than asserted: `windows-msvc-*`
([#584](https://github.com/mudler/vllm.cpp/issues/584)), and #968 on any branch
based on `c7cb59fbb`. A red in `agent-record` or `sanitize-cpu` is NEW
information (#873 and #904 are fixed). Load-dependent and to be re-run alone
before being charged: `test_openai_conformance`, `test_cpu_threadpool`,
`test_engine_core_proc`, `test_async_llm`
([#294](https://github.com/mudler/vllm.cpp/issues/294)),
`test_cpu_x86_llamacpp_floor` (exit 4 = `NO_QUIET_WINDOW`).

## 11. Arms

| Arm | State |
|---|---|
| bf16 / f32 reference | in scope |
| FP8 | the DFR path is resolved on the request, ahead of arm selection, so the FP8 DiT reaches it unchanged |
| NVFP4 | same |
| GGUF k-quants | **not applicable to this family.** LTX-2.5 ships no GGUF: the quantization kinds upstream defines are fp8-cast, fp8-scaled-mm, nvfp4-cast and nvfp4-prequant (`quantization_factory.py:23-26`), and no published LTX-2.5 checkpoint is a GGUF. Owed for LTX-2.5 as a whole under #644 only if such a checkpoint appears |
| detailing IC-LoRA | **refused by name**, pointing at #975 |

## 12. Outcome

**The DFR base landed and the temporal rounds did not.** §7's target sentence is
therefore NOT the sentence this row can carry, and §12.1 states what replaces it.
That is the row's most important result and it is stated first.

### 12.1 Reachability — the sentence the records actually carry

> The LTX-2.5 **DFR base pipeline** is reachable from `vllm_video_generate` with
> the `pipeline_kind` extra set to `dfr`, and from `ltx2-gen --pipeline-kind
> dfr`. It resolves upstream's padded keyframe canvas, places generated keyframe
> slots on the segment grid, marks them, reads them back before the trim, and
> trims the canvas to the caller's own frame count. **Generated keyframe slots
> are SERVED** on every pipeline, which retires the #920 refusal.
>
> The **temporal x2 latent upsampler is still UNDRIVEN**. DFR's rounds loop is
> refused by name and owed by #986. `docs/FEATURES.md`'s
> `Temporal x2 ups gated, UNDRIVEN` cell is therefore **unchanged**, and
> `.agents/specs/ltx25-resolution-envelope.md:436`, which asserts that cell is
> untouched, remains true. §9 of this spec predicted a contradiction that did
> not happen, and is left standing rather than deleted so the prediction and its
> outcome sit together.

### 12.2 What was NOT done, and why — the scope decision, stated rather than silent

The rounds loop is out. `AGENTS.md` asks for `NEEDS_DECISION` on a material
disagreement rather than a silent scope change, so the reasoning is here rather
than absent:

1. **The loop needs a seam this engine does not have.** Upstream's rounds
   invoke the same `DiffusionStage.__call__` the two stages use, per tile, with
   their own sigmas, stepper and seed (`dfr_pipeline.py:480-499`). This engine's
   denoise is written inline inside one per-phase loop of roughly 680 lines with
   no callable a tile can enter through. Extracting it is the correct change and
   it is a refactor of the render path, not an addition to it.
2. **There is no checkpoint.** §0.2 — re-verified again at the end of this row,
   not only at its start.
3. **A refused arm beats a half-built one.** The refusal names the loop, names
   three ruled-out causes with what ruled each one out, and is gated by
   `test_ltx2_video` including a case that re-derives its LOCAL claims from
   `ltx2_dfr.h`. That is visible debt with an instrument on it.

### 12.3 The two upstream comments that overstate themselves

Both were found by mutations that stayed GREEN, and in both cases the green is a
fact about the code rather than a hole in the test. Recorded because a later
reader who removes either construct would otherwise have no way to tell.

- **The first tile's lead-in** (`dfr_layout.py:177`). `lead_segments if index > 0
  else 0` reads as load-bearing and is redundant with the `max(0, own_lo - lead)`
  clamp beside it: for tile 0 the window and the drop prefix are identical for
  every lead. Measured against executed upstream — `_build_tile` with lead 0, 1
  and 5 returns three `TileRange`s that compare EQUAL. `ltx2_dfr.h` carried the
  opposite claim until this mutation refuted it.
- **The slot span's `causal_fix=False`** (`keyframe_slots.py:161-163`), whose
  comment says the fix "must not also be applied". Measured on a one-latent-frame
  shape against executed upstream: computing with `causal_fix=True` and then
  applying the explicit `[t, t+1)` span gives a BYTE-IDENTICAL tensor, because
  the fix touches only the temporal axis and the span overwrites it in full.

Both are mirrored anyway, to follow upstream's text rather than its algebra, and
both now carry a gated invariant so the corrected claim cannot drift back.

### 12.3a Two defects this row shipped and then found in itself

Both were found after the first push, by re-deriving this row's own upstream
anchors at the final tree rather than trusting the ones written during
implementation. All 44 resolve into their cited spans, 0 stale — and twice the
ANCHOR was right while the code was not.

**The refusal order was reversed.** The DFR canvas was resolved before
`Ltx2AssertResolution`, where upstream calls `assert_resolution` at
`dfr_pipeline.py:291` and `resolve_canvas` at `:314`. A caller wrong on both the
resolution and the frame count heard the wrong one of two correct refusals.
Invisible to everything else, because on that input neither port renders. Gated
by M14.

**The soundtrack was not cut to the picture.** `dfr_pipeline.py:552-560` cuts the
decoded audio to `num_frames / playback_fps`, and upstream states the consequence
rather than the mechanism: *"Audio was generated for the padded canvas, so cut it
to the video's duration or the muxed container outlasts the picture."* This row's
video trim moves `frames`; `ashape.frames` was derived from the PADDED count
inside the phase loop, `audio_lf` carries it out, and the vocoder runs over all
of it. **A comment in this row asserted the cut was "already implied by the
trimmed `frames`", and that was false.** A 9-frame DFR request emitted 0.375 s of
picture beside 1.01 s of sound. Nothing about the render's shape, its frame count
or its exit status could see it — it shows up only in a muxed container this
library does not produce. Fixed, and gated by M15, whose RED prints both numbers.

The second one is the more useful record. It is not that the port missed a line;
it is that the port missed a line AND wrote a comment explaining why the line was
unnecessary. A reader who checked upstream would have found `:552-560` accounted
for and stopped. The gate that catches it reads the WAV header rather than
trusting the engine's own report of what it wrote.

### 12.4 Mutations — every one with three facts

Layout suite, `test_ltx2_dfr` (green: 11 cases / 652 assertions, `RUN_EXIT=0`).
Every mutation BUILT with `': error:' count = 0` unless stated, and the binary
sha256 CHANGED on each rebuild, so none of these is a stale binary reporting the
previous result. The tree restored byte-identical (`git diff --stat` empty) and
the binary returned to its baseline sha.

| # | mutation | result | exit |
|---|---|---|---|
| M1 | `choose_segment_length` tie takes the SMALLER candidate | RED | 1 |
| M2 | `_build_tile` keeps boundary 0 as an anchor | RED | 1 |
| M3 | `drop_latent_prefix` omits the seam handover `+1` | RED, 18 failed | 1 |
| M4 | NEGATIVE CONTROL: the first tile is given a lead-in | GREEN | 0 |
| M4b | the lead-in is dropped from EVERY tile | RED | 1 |
| M11 | the slot seed rounds half-away-from-zero, not half-to-even | RED | 1 |
| M12 | the carry-forward merge lets the ANCHOR win a shared position | RED | 1 |
| M13 | CONTROL: a no-op edit | GREEN | 0 |

Conditioning suite, `test_ltx2_dfr`:

| # | mutation | result | exit |
|---|---|---|---|
| M5 | the slot seed never reaches `latent` | RED | 1 |
| M5b | the slot seed ALSO goes into `clean` | RED | 1 |
| M6 | the slot append passes `marked=false` | RED | 1 |
| M7 | NEGATIVE CONTROL: the slot span applies the causal fix | GREEN | 0 |
| M7b | the slot span is a full latent frame, not one pixel frame | RED | 1 |
| M8 | the extraction ASSUMES the slots trail | RED | 1 |
| M8b | the layout records the wrong `first_token` | RED | 1 |

Engine suite, `test_ltx2_video` (green: 52 cases / 1243 assertions, `RUN_EXIT=0`):

| # | mutation | result | exit |
|---|---|---|---|
| **R1** | **REACHABILITY: the production `Ltx2DfrResolveCanvas` call site is deleted** | **RED** | 1 |
| **R2** | **REACHABILITY: the production slot-conditioning call site is deleted** | **RED** | 1 |
| M6 | the slot append passes `marked=false` | RED | 1 |
| M9 | the slots are extracted AFTER the trim instead of before | RED | 1 |
| M10 | the DFR trim is dropped | did NOT BUILD, 1 compile error — establishes nothing |
| M14 | the canvas is resolved BEFORE `assert_resolution` (upstream's order reversed) | RED | 1 |
| M15 | the DFR audio cut is disabled | RED, `picture 0.375s, sound 1.01s` | 1 |

R1 and R2 are the reachability evidence `.agents/reachability.md` asks for: the
production CALL SITE is deleted, not the implementation, and the focused gate
goes red. A green there would have meant the DFR path is a test-only driver.

**Two mutations stayed GREEN against the engine suite, and the first explanation
for one of them was wrong.** Both are recorded with what was actually measured,
because a green mutation with a confident story attached is worse than one with
none.

**M8** needs an appending item AFTER the slot item, and upstream has exactly one:
`VideoConditionByReferenceLatent` (`dfr_pipeline.py:366-373`), whose arm #975
owes. Until that lands the slots always trail and the two readings agree. Gated
in `test_ltx2_dfr`, which builds that configuration by hand and where M8 goes
RED.

**M5** was first explained as "a no-op on phase 0, which is the only phase the
engine's DFR case runs". That explanation predicted that a TWO-PHASE engine case
would make it RED. A two-phase case was written — it uses the fixture spatial
upsampler, renders at full resolution and asserts stage 2 places and reads back
its own slots — **and M5 stayed GREEN with it present, 52/52, exit 0.**

So the real reason is different and it is a property of the pipeline. Stage 2
re-noises to `stage_2_sigmas[0]`, about 0.909, so the seed is almost entirely
replaced by noise before the first step and the denoise loop generates the rest.
The assertions an engine test can make about a slot are structural — counts,
positions, resolutions — and the seed moves none of them. There is no A/B
available either, because the seed is internal rather than a request field.

The two-phase case is kept, and its comment now says exactly this: it REACHES
the seeded path, which no other engine test does, and it does not DETECT the
seed. M5 is gated in `test_ltx2_dfr`, on the property that actually belongs to
the port — the seed lands in `latent` and not in `clean` — and it goes RED
there.

## Owed

- [#986](https://github.com/mudler/vllm.cpp/issues/986) — **DFR's temporal
  refinement rounds** (`dfr_pipeline.py:402-529`), which is what would DRIVE the
  temporal x2 latent upsampler. Refused by name at the request surface. What is
  unreached is the rounds loop itself; the operator, the canvas layout and the
  slots it needs are all ported and gated here. §12.2 says why it is not in this
  row, and `docs/FEATURES.md`'s UNDRIVEN cell is correct until it lands.
- [#986](https://github.com/mudler/vllm.cpp/issues/986) — the standalone
  single-frame decode of a generated keyframe slot, and the surface that would
  return slot PIXELS to a caller. DFR keeps its slots in latent space (§2b), so
  nothing here reaches it and nothing here needs it.
- [#975](https://github.com/mudler/vllm.cpp/issues/975) — DFR's stage-2 x2
  spatial detailing IC-LoRA, which needs the reference-latent arm that issue
  owns. Refused by name, pointing at #975.

## Stop conditions

- If the slots turn out to be reachable without a layout — if they can be assumed
  to trail — stop: §3.3's reading of `keyframe_slots.py:143-147` is wrong and the
  #920 refusal named the wrong blocker.
- If the rounds loop cannot reach the temporal upsampler from a production entry
  point, stop and report. Landing the loop unreached would repeat exactly the
  defect this row exists to retire, and `AGENTS.md` `## Nothing lands dead`
  permits it only with the naming §7 would then have to give.
- If the READER ANCHORS move, re-derive them in this row rather than editing the
  gate.

## Now

`ACTIVE`. The DFR base and the generated keyframe slots are implemented and
gated on `row/LTX25-DFR-PIPELINE`, in review. The temporal rounds are refused
by name and owed above, so the temporal upsampler remains UNDRIVEN and
`docs/FEATURES.md` still says so.
