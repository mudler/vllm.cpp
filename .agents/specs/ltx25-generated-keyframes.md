# LTX25-GENERATED-KEYFRAMES — refuse generated keyframe slots by what is missing

Issue: [#920](https://github.com/mudler/vllm.cpp/issues/920). Campaign:
[#644](https://github.com/mudler/vllm.cpp/issues/644). Sibling that landed the
marker itself: [#658](https://github.com/mudler/vllm.cpp/issues/658),
[`ltx25-keyframes-abs-pos.md`](ltx25-keyframes-abs-pos.md), which put "keyframe
*conditioning* as a user-facing feature (supplying keyframe slots)" explicitly
**out** of its scope. This row picks up that half.

Oracle: Lightricks `LTX-2` at pin `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
checked out at `/home/mudler/_git/LTX-2` and verified at that SHA with a clean
tree before any anchor below was read.

## 0. The two features that both say "keyframe"

This row exists because one word covers two upstream features, and conflating
them has already cost this campaign one falsely-pinned test assertion
(`test_ltx2_video.cpp:1162-1170` records the previous one).

| | Supplied keyframes | **Generated keyframe slots** |
|---|---|---|
| Upstream item | `VideoConditionByKeyframeIndex` | `VideoGeneratedKeyframeSlots` |
| Anchor | `conditioning/types/keyframe_cond.py:36-90` | `conditioning/types/keyframe_slots.py:27-174` |
| Who supplies content | the caller, an image per frame index | nobody — the model generates it |
| `keyframes_mask` | `marked=False` (`keyframe_cond.py:84-86`) | `marked=True` (`keyframe_slots.py:121`) |
| Touches the trained bias | **no** | **yes** |
| State here | **SERVED** as of `c7cb59fbb` (row LTX25-TOKEN-APPEND, #930) | no request surface |

The class spans `27-174`; `apply_to` is `71-150` and `_slot_positions` `153-174`.
The row's first draft cited `27-150` for the class, which stops at the end of
`apply_to` and excludes the very span the same draft cited separately as class
content.

`marked` is the whole difference in one argument. `extend_keyframes_mask`
(`conditioning/mask_utils.py:76-107`) documents the polarity: *"True only for
generated keyframe slots; given-content conditioning tokens (image guidance,
reference latents) are never marked, matching the reference implementation."*

So `VideoGeneratedKeyframeSlots` is the **only** conditioning item upstream that
marks anything, and therefore the only user-facing feature that puts the trained
`keyframes_abs_pos_embedding` on a token other than the target's own first
latent frame. `KeyframeInterpolationPipeline`
(`ltx-pipelines/keyframe_interpolation.py`) does **not** do this: it builds only
`VideoConditionByKeyframeIndex` items through
`image_conditionings_by_adding_guiding_latent` (`utils/helpers.py:343-367`),
contains no reference to `generated_keyframe` or `keyframes_abs_pos` anywhere in
its 362 lines, and is absent from the feature's own "where it applies" list
(`ltx-pipelines/docs/conditioning.md:47-51`, which names
`TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`,
`DistilledPipeline` and the multi-GPU runners). Recorded here because the
opposite was the working assumption when this row was dispatched.

## 1. What is wrong today — measured, not inferred

**Re-measured at the merge with main at `e5351776c`, because the first
measurement went stale under this row.** At the branch's merge base
(`5a0ffe9e3`) `git grep -i generatedkeyframe` returned **0** lines, and the spec
said so. At `e5351776c` it returns **7 lines in 4 files** — three of them outside
`.agents/`: `include/vllm/model_executor/models/ltx2_conditioning.h`,
`src/vllm/model_executor/models/ltx2_conditioning.cpp` and
`tests/vllm/models/test_ltx2_vae.cpp`. Row LTX25-TOKEN-APPEND (#930) put them
there. So the concept is no longer absent from the tree; what is absent is the
**request surface** — no key, no refusal, no test — and the **readback**.

A count that was true when written and false at review is exactly the failure
this row's own refusal is about, which is why it is recorded here rather than
quietly corrected.

The per-generation extras check in `Ltx2VideoEngine::Generate` accepts exactly
one key and rejects every other by the same generic message:

```
unknown per-generation extra 'num_generated_keyframes'. This family defines: image_crf
```

That is the wrong message, and it is wrong in a way this campaign has already
paid for once. It asserts the family does not define the key, which sends the
reader looking for a typo. `CheckUnservedExtras` exists on the load side
precisely because of that distinction (#611): *"Deliberately NOT the 'unknown
load extra' path above. That message says the family does not define the key,
which is false here and would send the reader looking for a typo instead of for
the unported head."*

The obligation is stated twice in policy. [`AGENTS.md`](../../AGENTS.md)
`## Shared seams`: *"Refuse an unimplemented arm with a message that names the
missing part. Record the arm as owed. Never leave the missing path to be
discovered later."* [`porting-a-model.md:81`](../porting-a-model.md): an arm is
*"explicitly refused with a message naming the missing piece."*

## 2. What upstream does, with anchors

Every anchor re-derived at the pin and asserted unique before this spec was
committed (`## 8`).

**The item.** `VideoGeneratedKeyframeSlots.__init__`
(`keyframe_slots.py:47-69`) takes `pixel_frame_indices` — non-empty, strictly
increasing, non-negative — and an optional `initial_keyframes` of shape
`(B, C, K, H, W)` with `K == len(indices)`. `apply_to` (`:71-150`):

- refuses an index at or beyond the target's pixel-frame count (`:77-81`);
- sizes each slot at one latent frame of tokens, `patchifier.get_token_count`
  over `target_shape._replace(frames=1)` (`:83-84`);
- builds slot positions whose temporal span is exactly `[t, t+1)` with
  `causal_fix=False`, because the span is set explicitly (`:152-174`);
- appends to `latent`, `denoise_mask`, `positions`, `clean_latent` (`:136-140`);
- sets `denoise_mask = 1` on the new tokens so the noiser lerps from the slot
  `latent` and ignores `clean_latent` (`:118-119`);
- marks them: `extend_keyframes_mask(..., marked=True)` (`:121`);
- rebuilds the attention mask via `update_attention_mask` (`:123-131`);
- refuses a second application (`:133-134`);
- records `GeneratedKeyframeLayout(pixel_frame_indices, tokens_per_keyframe,
  first_token)` (`:143-147`), so the slots are located exactly rather than
  assumed to be trailing (`types.py:220-247`).

**Readback.** `clear_conditioning` (`ltx_core/tools.py:88-117`) extracts the
denoised slot content into `generated_keyframes` *before* trimming the extra
tokens (`:97`, `:115`), validating the layout against the live token count and
the target resolution (`extract_generated_keyframes`, `tools.py:203-230` — the
first draft wrote `203-241`, which runs 11 lines past the method and into an
unrelated `AudioLatentTools` dataclass at `233`). Each frame must then be decoded as a
standalone one-frame clip — `types.py:269-272` and `conditioning.md:59-61` both
warn that a K-frame causal decode blends slots that were never adjacent.

**The request surface.** `--num-generated-keyframes`, `type=int`, `default=0`
(`ltx-pipelines/utils/args.py:833-844`). Opt-in per CLI *"only pipelines that
actually forward the value to their first diffusion stage should advertise the
flag, otherwise it would parse and be silently ignored"* (`:828-831`) — the same
rule this row is applying. `resolve_generated_keyframes`
(`utils/helpers.py:394-411`): an `int` requests that many evenly spaced interior
positions, a sequence gives explicit indices, `0`/empty means off.
`evenly_spaced_keyframe_positions` (`:370-381`) refuses a negative count
(*"num_keyframes must be non-negative"*) and a target shorter than
`num_keyframes + 2`. `has_generated_keyframes` (`:384-391`) exists so callers do
not test truthiness of a value that may be a tensor.

**The admission gate.** `DiffusionStage.supports_generated_keyframes`
(`ltx-pipelines/utils/blocks.py:395-403`) reads the **declared** config flag
only, *"answered before any weights are built"*.
`assert_generated_keyframes_supported` (`:405-419`) raises naming
`use_keyframes_abs_pos_embedding`; `_assert_supports_conditionings` (`:421-425`)
re-checks as a backstop for callers that build items directly. The reason it
refuses rather than degrades is stated at `keyframe_slots.py:9-12`: on a
checkpoint without the marker *"the slots would be denoised as unmarked tokens
and the extra compute would be wasted"* — and each slot costs one latent frame
of tokens to buy one pixel frame (`docs/conditioning.md:43-46`: about +16%
tokens at 512x768/241 frames, +31% at 1088x1920/121).

## 3. Scope

**In.**

1. Define `num_generated_keyframes` as a key this family **knows**, spelled as
   upstream spells it, on the per-generation surface (`vllm_video_params`
   extras), which is where upstream takes it — a `__call__` argument, not a load
   option.
2. Mirror upstream's `0 = off` default exactly: an explicit `0` is upstream's
   own default and must **not** refuse. This is the half most likely to be
   ported as "any mention refuses", which would break a caller that passes the
   default through.
3. Mirror `evenly_spaced_keyframe_positions`' negative-count `ValueError`
   (`helpers.py:372-373`) for a negative value.
4. For a positive count, **refuse by name**, naming the one missing piece from
   `## 4`, the upstream symbols a later reader can go and check, and the two
   reasons that are ruled out together with what ruled each one out.
5. State the refusal's claims about **this** tree in a form a gate can
   re-derive, and gate them. See `## 4a`.
6. Record the arm as owed under `## Owed` with issue #920.

**Out.**

- Building the token-append machinery. Row LTX25-TOKEN-APPEND (#930) built it
  and it landed at `c7cb59fbb`; nothing is owed here. The one piece of that seam
  still owed is a production caller for `Ltx2ExtendKeyframesMask(..., marked=true)`,
  and that belongs to **this** row — see `## Owed`.
- `initial_keyframes` seeding, `GeneratedKeyframeLayout` readback, and the
  standalone single-frame decode. All owed to this row, none of them landed here.
- Any change to the marker itself. #658 landed it and it is applied on every
  render; `_first_frame_keyframes_mask` (`ltx_core/tools.py:184-196`) marks the
  target's first latent frame **unconditionally**, so the embedding is consulted
  on the default text-to-video path and must not be made conditional on this
  key.
- `Ltx2AdoptDeclaredDitParams`. See `## 5` — it is already correct.

## 4. What the refusal must name

**One blocker: readback with a standalone decode.** Unique to this arm; the
supplied arm needs none of it. `apply_to` records a `GeneratedKeyframeLayout`
that locates the slots exactly, `clear_conditioning` extracts them into
`generated_keyframes` before it trims, and each frame is then decoded on its own.
Without all three the slots would be generated and then thrown away, which is
worse than refusing.

**Two reasons are RULED OUT, and each must name what ruled it out**, in the shape
this file's own `keyframes_abs_pos_embedding` clause already uses. A refusal that
merely omits a plausible reason leaves the next reader to re-derive the
refutation; the point of naming it is that they re-CHECK it instead.

1. **NOT the token-append machinery.** The first draft of this row named it as
   blocker 1, and by the time the row reached review that was false in every
   particular: `c7cb59fbb` (row LTX25-TOKEN-APPEND, #930) landed
   `Ltx2ExtendKeyframesMask` and `Ltx2ClearConditioning`, bound a `target_tokens`
   in the phase loop, made the clear step a real trim, and **served** the
   LAST-frame supplied-keyframe arm through all of it. Two small pieces of that
   seam are still owed, and they are owed to THIS row rather than to #930: the
   `marked=true` branch has no production caller, and `update_attention_mask` has
   no local counterpart, because `Ltx2LatentState` deliberately carries no
   attention-mask field (`include/vllm/model_executor/models/ltx2_conditioning.h:58-70`
   @ `e5351776c` argues that omission from upstream's own polarity, and #930 kept
   it deliberately). Neither is the blocker.
2. **NOT `keyframes_abs_pos_embedding`.** Ported under #658 and applied on every
   render, because `_first_frame_keyframes_mask` marks the target's first latent
   frame unconditionally.

## 4a. And the refusal's LOCAL claims must be gated

The stale half of the first draft was never the upstream half. Upstream symbol
names do not move when this tree changes, so a suite that asserts only on
`update_attention_mask` and `clear_conditioning` stays green through the exact
event that falsifies the message. Measured, not argued: the reviewer's mutation
M7 replaced the local-cause sentence with a self-declared falsehood, left the
upstream names alone, and the suite reported 18/18, exit 0.

So the message carries its claims about **this** tree as two explicit lists —
`DECLARED HERE:` and `ABSENT HERE:` — and `test_ltx2_video` parses them **out of
the thrown message** and re-derives each against
`include/vllm/model_executor/models/ltx2_conditioning.h`. Three properties make
that check real rather than decorative:

- **A different file.** The claim lives in `ltx2_video.cpp`; the truth is read
  from `ltx2_conditioning.h`. Reading a claim out of the file that makes it is
  the tautology [#911](https://github.com/mudler/vllm.cpp/issues/911) records.
- **Comment lines stripped.** `ltx2_conditioning.h:170` names upstream's
  `VideoGeneratedKeyframeSlots` in prose, so an unstripped search for
  `GeneratedKeyframe` finds a hit and the ABSENT half silently inverts. The
  stripping is itself controlled: the case requires
  `struct Ltx2LatentState` to survive it.
- **A count floor.** An empty parsed list satisfies every `for each` and reports
  a pass over nothing. Both lists have a minimum size, so a message that dropped
  its clause cannot report a pass over an empty set.

The check fires in both directions. If the readback lands, `GeneratedKeyframe`
appears in the header, ABSENT goes red, and whoever landed it is told the refusal
is now false. If the append seam is renamed or removed, DECLARED goes red.

## 5. #902 — answered, and it is not a code hole

[#902](https://github.com/mudler/vllm.cpp/issues/902) asks which way
`Ltx2AdoptDeclaredDitParams` resolves a checkpoint that declares
`use_keyframes_abs_pos_embedding=true` and carries no keyframe tensor, and
whether that is upstream's behaviour. Answered from upstream, not from our code,
as the issue asks:

Upstream resolves the same contradiction **two ways at two layers, by design**:

- `LTXModel.supports_keyframes_abs_pos_embedding`
  (`model/transformer/model.py:166-173`) reads the **materialized tensor**:
  `embedding is not None and not embedding.is_meta`. `_init_video:216-218`
  builds the parameter whenever the flag is declared, models are built on `meta`
  and loaded `strict=False, assign=True`, so a declare-true / carry-nothing
  checkpoint leaves it on `meta` and the property is **False**. Its own
  docstring names our exact case.
- `DiffusionStage.supports_generated_keyframes`
  (`ltx-pipelines/utils/blocks.py:395-403`) reads the **declared flag only**, so
  on the same checkpoint it returns True and the admission gate would let a
  request through.

Ours resolves to shapes (`ltx2_loader.cpp`, `Ltx2AdoptDeclaredDitParams`),
matching the first and asserted at `tests/vllm/multimodal/test_ltx2_video.cpp`.
The hypothesised live hole — that the `ltx2_dit.cpp` guard should have refused
the #902 render and did not — **does not exist**, because the flag resolves
false and the guard never arms on that checkpoint.

If anything ours is the safer of the two. `apply_keyframes_absolute_embedding`
(`transformer_args.py:23-43`) skips only on a `None` provider, and
`_keyframes_embedding` (`model.py:158-164`) returns the meta parameter rather
than `None`, so upstream on that checkpoint would reach `embedding.to(dtype=...)`
on a meta tensor. `enable_keyframes_abs_pos_embedding` (`model.py:175-200`)
exists to materialize real zeros for exactly this, and it has **one hit
repo-wide at the pin: its own definition, no caller.** Its docstring is explicit
that this only makes the marker *harmless*, not meaningful.

Residual on #902 is checkpoint availability, not code. Reported on the issue; no
change owed here.

## 6. Tests

RED-first, and the RED must be the intended failure. Every case enters through
the **production entry point** — `vllm::multimodal::LoadVideoEngine` then
`VideoEngine::Generate`, the chain `vllm_video_generate` takes — never by
constructing a type ([`reachability.md`](../reachability.md)).

1. **A positive count is refused, and the message names the READBACK and both
   ruled-out reasons.** Assert on the upstream symbols a reader can go and check
   (`VideoGeneratedKeyframeSlots`, `keyframe_slots.py`, `GeneratedKeyframeLayout`,
   `extract_generated_keyframes`, `generated_keyframes`, `update_attention_mask`,
   `clear_conditioning`) and on `#930`, not on prose.
1a. **Its claims about THIS tree are re-derived from this tree.** Parse the
   `DECLARED HERE:` and `ABSENT HERE:` lists out of the thrown message and check
   each against `ltx2_conditioning.h`'s declarations with comment lines stripped.
   This is the case `## 4a` exists for: every assertion in case 1 is on an
   upstream name, and no change to this tree can move one.
2. **The refusal is NOT the generic one.** Assert the message does not contain
   `unknown per-generation extra`. This is the assertion the row exists for, and
   without it case 1 passes against the message we already have.
3. **`0` does not refuse**, mirroring upstream's default (`args.py:836`). The
   render must complete. This is the half a naive port breaks.
4. **A negative value refuses with upstream's own reason**, mirroring
   `evenly_spaced_keyframe_positions` (`helpers.py:372-373`), and NOT with the
   unported-machinery message — a malformed request and an unported arm are
   different answers.
5. **The refuted reason may be named but never as the blocker.** If the message
   mentions `keyframes_abs_pos_embedding`, it must also say it is not what is
   missing and cite #658, mirroring the guard the LAST-frame case already
   carries at `test_ltx2_video.cpp:1180-1186`.

**Mutations that must be run and recorded**, each with three facts —
`git diff --stat` after applying, whether it BUILT with any compile error beside
it, and the exit code:

- delete the production refusal call site (the reachability mutation) — cases
  1, 1a, 2, 4 must go RED;
- make the refusal fire on `0` — case 3 must go RED;
- make it fire on any presence of the key regardless of value — case 3 RED;
- accept a positive count silently — cases 1, 1a, 2 RED;
- give a negative value the unported-machinery message — case 4 RED;
- **M7, the one this repair adds**: replace the message's LOCAL-cause clause with
  a self-declared falsehood and leave every upstream symbol in place — case 1a
  RED. Against the first draft this mutation left the suite GREEN at 18/18,
  exit 0, which is the whole reason `## 4a` exists.

## 7. Risks

- **`ltx2_video.cpp` carries derived READER ANCHORS** (the comment above
  `kKnownLoadExtras`, gated by `test_ltx2_video`) whose values are line numbers
  in that file. A clean `git merge` will not warn when they go stale. This row
  keeps its `.cpp` edit **below** every anchored line and puts the new key
  constant in the header, so no anchor should move — but that is a prediction,
  and the gate is the check. Re-derive at the final tree.
- **Three other agents are editing `ltx2_video.cpp`, `ltx2.h`,
  `docs/FEATURES.md` and `.agents/issue-index.md` concurrently**, and
  `origin/main` moves several times an hour. Keep the footprint minimal and
  merge often. `docs/FEATURES.md` is a keyed record: reapply the scoped edit by
  key and prove unrelated keys byte-identical rather than accepting a three-way
  merge.
- **Refusing too broadly.** An explicit `0` is upstream's default and a caller
  that plumbs defaults through would hit a refusal that upstream does not raise.
  Test 3 is the guard.
- **The refusal's REASON going stale under the row.** This happened: the row was
  written against a merge base where the token-append seam did not exist, and
  `c7cb59fbb` landed it while the branch sat in review, and by the merge the
  branch was fifteen commits behind. The
  refusal, the index row, `docs/USAGE.md`, the PR body and this spec all named it
  as blocker 1, and nothing went red, because every assertion was on an upstream
  symbol name. `## 4a` is the guard, and it is the only one of the three risks
  here with a test behind it.

## 8. Anchor discipline

Every `file:line` in this spec was re-derived at the pin, and the needle was
derived from the **claim** rather than read out of the cited span — that check
is circular and has reported 27/27 FRESH while five anchors pointed at unrelated
code. Uniqueness is asserted, not existence.

## 9. Gates

CPU-only. The GPU is not used: `dgx.casa` is running a long render under `flock`
and OOM-reboots when the 119 GiB unified pool is exhausted.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Known-red on main and to be proved pre-existing rather than asserted: the #873
family of checkers and `windows-msvc-*` (#584). Load-dependent and to be re-run
alone before being charged: `test_openai_conformance`, `test_cpu_threadpool`,
`test_engine_core_proc`, `test_async_llm` (#294), `test_cpu_x86_llamacpp_floor`.

## 10. Arms

| Arm | State |
|---|---|
| bf16 / f32 reference | refusal reached and gated; the arm itself is owed |
| FP8 | same refusal, same path — the key is checked before any DiT arm is selected |
| NVFP4 | same |
| GGUF k-quants | not applicable to this key; owed for LTX-2.5 as a whole under #644 |

The refusal is resolved on the request, ahead of arm selection, so no arm can
reach the unported readback by a different route. That is the property test 1
pins.

## Owed

- [#920](https://github.com/mudler/vllm.cpp/issues/920) — the generated keyframe
  slots arm itself: `GeneratedKeyframeLayout` readback, extraction into
  `generated_keyframes` before the trim, and the standalone single-frame decode
  of each slot. This row lands the refusal and the request surface only. Campaign
  [#644](https://github.com/mudler/vllm.cpp/issues/644).
- [#920](https://github.com/mudler/vllm.cpp/issues/920) — **a production caller
  for `Ltx2ExtendKeyframesMask`'s `marked=true` branch.** Row LTX25-TOKEN-APPEND
  landed that branch with a unit driver and no production caller, and assigned it
  here by ID and issue
  ([`ltx25-token-append.md` `## 8`](ltx25-token-append.md)). Its only driver in
  the tree is `tests/vllm/models/test_ltx2_vae.cpp:2494` @ `e5351776c`, which is
  a unit case rather than reach. `AGENTS.md` `## Nothing lands dead` permits a
  staged unreached slice only while the owning row's spec lists it under
  `## Owed`, and until this repair it was named in #930's spec and in no other,
  so the permission rested on a bullet this row never wrote. The only upstream
  construct that passes `true` is `VideoGeneratedKeyframeSlots`
  (`keyframe_slots.py:121`), so no other row can claim it.

## Stop conditions

- If the readback turns out to be reachable without a layout — if the slots can
  be assumed to trail — stop: the reading of `keyframe_slots.py:143-147` and
  `ltx_core/types.py:220-247` in `## 2` is wrong and the refusal names the wrong
  blocker.
- If the refusal cannot be placed before arm selection, stop and report — a
  refusal reachable on only some arms is worse than none.
- If the READER ANCHORS move, re-derive them in this row rather than editing the
  gate.

## Now

`ACTIVE`. Refusal and request surface in review on `row/LTX25-GENERATED-KEYFRAMES`,
merged onto `e5351776c` and repaired: the refusal now names the READBACK as its
one blocker, the token-append seam is a ruled-out clause rather than a blocker,
and `## 4a` gates the message's local claims. The arm itself is owed above.
