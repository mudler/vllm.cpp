# `LTX25-IC-LORA-REF-VIDEO` — the IC-LoRA reference clip and its attention mask

Issue: [#3020](https://github.com/mudler/vllm.cpp/issues/3020).
Base: `fafb58ef6`. Oracle: `Lightricks/LTX-2 @ fd4ded7f`
(`.agents/oracles/ltx-2.md`), executed, never read alone.

Gaps **A15** and **A16** of `.agents/specs/ltx25-completion-scope.md`, landed as
one row. §1 establishes why they are one row rather than two.

## 0. What is claimed, and what is not

CLAIMED. An IC-LoRA render can be conditioned on a reference clip through the
public video ABI, and that clip reaches the DiT as appended tokens carrying the
reference item's own RoPE geometry. An optional per-region attention mask
reaches the DiT as an additive log-space self-attention bias.

NOT CLAIMED. No render quality claim, no speed claim, and no real-weights
measurement: this row has no GPU authority and no LTX checkpoint. The gates are
value gates against the executed pinned module plus an end-to-end reachability
gate on the reduced-dimension checkpoint fixture in
`tests/vllm/multimodal/ltx2_video_fixture.h`.

NOT CLAIMED EITHER. The dtype. This engine's LTX host path materialises f32
where upstream materialises bf16, which is the standing divergence the A24 wave
campaign owns; this row mirrors the tree's f32 convention at every new site and
records the width owed rather than pretending the bytes match. See §4 R4.

## 1. Why A15 and A16 are one row

Upstream applies `ConditioningItemAttentionStrengthWrapper` at exactly one site,
`iclora_utils.py:168-169`, and its argument is always the
`VideoConditionByReferenceLatent` built two lines above it.
`combined_image_conditionings` (`utils/helpers.py`) never wraps, and neither
appending VIDEO item passes a mask — `keyframe_cond.py:68-76` and
`reference_video_cond.py:88-96` both pass a literal `attention_mask=None`.

In this tree `Ltx2ConditionVideoByReference` has zero production call sites at
`fafb58ef6`: its definition at `src/vllm/model_executor/models/ltx2_conditioning.cpp:565`,
prose in `ltx2_conditioning.h`, and two test call sites. The reference arm is
refused in `src/vllm/multimodal/ltx2_video.cpp`. So A16 alone has nowhere to
attach that upstream also attaches to: putting the mask on the keyframe or image
items that ARE reached would invent behaviour upstream does not have.

`.agents/specs/ltx25-completion-scope.md` §8 order 8 already says this —
"A15, A16, A17 — the IC-LoRA family, one owed table", M-L total. A16's
standalone "M" sizing was wrong. A17 landed earlier in the campaign.

## 2. The dead seam this row retires

A16's consumption half is **already built, and nothing reaches it**. Verified on
`fafb58ef6` with `grep -rn 'attention_mask' src/ include/ tests/` and a second
pass for assignments:

```
$ grep -rn '\.attention_mask *=\|attention_mask_rows *=' src/ include/ tests/
src/.../ltx2_device.cpp:1097:    VT_CHECK(m.attention_mask_rows == 1 || ...   <- a comparison
src/.../ltx2_dit.cpp:599:       VT_CHECK(m.attention_mask_rows == 1 || ...   <- a comparison
tests/vllm/models/test_ltx2.cpp:282-285          <- BuildModalities, by hand
tests/vllm/models/test_ltx2_device.cpp:282-285   <- BuildModalities, by hand
include/.../ltx2.h:558                            <- the declaration's default
```

Four assignments, all inside a test helper that constructs `Ltx2ModalityInput`
by hand. Nothing in `src/` writes either field. The chain below those
assignments is complete and correct — `Ltx2PrepareSelfAttentionMask`
(`ltx2.cpp:846`) into `self_bias` on host (`ltx2_dit.cpp:598-604`) and device
(`ltx2_device.cpp:1096-1104`), into `vt::AttentionCross`'s additive bias — which
is exactly the failure `.agents/reachability.md` names: the class works, and no
capability reaches it.

Landing this row is the production assignment. The reachability mutation in §5
is deleting it.

## 3. Upstream anchors

Root `Lightricks/LTX-2 @ fd4ded7f`.

| ours | upstream |
|---|---|
| `Ltx2IcLoraReferenceGeometry` | `ltx-pipelines/iclora_utils.py:111-117` |
| `Ltx2TemporalSubsample` | `ltx-pipelines/iclora_utils.py:87-90` (called `:143-144`) |
| the reference item's encode | `ltx-pipelines/iclora_utils.py:141-148` |
| `Ltx2ConditionVideoByReference` (already ported) | `ltx-core/conditioning/types/reference_video_cond.py:46-108`, built at `iclora_utils.py:162-167` |
| `Ltx2DownsampleMaskVideoToLatent` | `ltx-pipelines/iclora_utils.py:52-84` |
| the strength multiply | `ltx-pipelines/iclora_utils.py:151-156` |
| `Ltx2ResolveCrossMask` | `ltx-core/conditioning/mask_utils.py:13-73` |
| `Ltx2BuildAttentionMask` | `ltx-core/conditioning/mask_utils.py:170-243` |
| `Ltx2UpdateAttentionMask` | `ltx-core/conditioning/mask_utils.py:110-167` |
| the wrapper's apply order | `ltx-core/conditioning/types/attention_strength_wrapper.py:43-71` |
| the mask read (`Ltx2ReadFrameDirectory` + `Ltx2MaskVideoFromPixels`, inlined at the call site) | `ltx-pipelines/ic_lora.py:511-537` (`_load_mask_video`) |
| the stage split | `ltx-pipelines/ic_lora.py:269-281` (stage 1) vs `:314-321` (stage 2) |
| the CLI shape | `ltx-pipelines/ic_lora.py:415-441`, `:452-463`, `:481-498` |
| the consumption (already ported) | `ltx-core/model/transformer/transformer_args.py:208-237`, `:289` |

## 4. Design

### 4.1 A15 — the reference clip

Upstream's pixel path is `decode_video_by_frame` into `video_preprocess`
(`iclora_utils.py:141-142`). `video_preprocess` (`media_io/decode.py:82-103`) is
per frame `resize_and_center_crop(f.float(), H, W)` then `normalize_images`,
concatenated on the frame axis. **That function is already ported and gated in
this tree** as `Ltx2ReadFrameDirectory`
(`src/vllm/model_executor/models/ltx2_retake.cpp:211`), which runs each
`frame_%06d.ppm` through `Ltx2LoadImageAndPreprocess` — the same
resize-and-centre-crop plus `/127.5 - 1` chain — and returns `[C, T, H, W]`, the
layout `Ltx2ConvVideoEncode` takes. The container-versus-frame-directory
substitution is the harness adaptation row `LTX25-RETAKE` already recorded and
`video_api.cpp` already documents; no demuxer is vendored here.

So A15's genuinely new code is small and each piece is separately gateable:

1. `Ltx2IcLoraReferenceGeometry(height, width, scale)` — refuse when
   `scale != 1` and either axis is not divisible (`:112-115`), else return
   `height / scale`, `width / scale` (`:116-117`). Integer division, and the
   refusal carries upstream's own wording.
2. `Ltx2TemporalSubsample(clip, channels, frames, plane, factor)` — keep index
   0, then `range(1, frames, factor)` (`:89`). Note the shape: index 1 is
   ALWAYS kept when it exists, so at `factor == 2` and 5 frames the kept set is
   `{0, 1, 3}`, not `{0, 2, 4}`. Guarded on `factor > 1` exactly as `:143` is.
3. Encode the whole clip with the existing `Ltx2ConvVideoEncode`, which the
   retake arm already drives multi-frame at `ltx2_video.cpp:3703`.
4. Apply `Ltx2ConditionVideoByReference` with `downscale_factor = scale` and
   `temporal_scale_factor` from the adapter metadata (`im.dit.lora_reference`,
   already read by row `LTX25-IC-LORA`), and `strength` from the request.

**Stage 1 only.** `ICLoraPipeline` gives stage 1 `_create_conditionings`, which
appends the reference item (`ic_lora.py:269-281`), and stage 2 plain
`combined_image_conditionings` with no reference item (`:314-321`). The phase
loop therefore applies this block on `phase_index == 0` and on no other phase.
The predicate that routes and the predicate that refuses are the same
expression, bound once in a named local, so the two cannot drift.

### 4.2 A16 — the attention mask

The mask read mirrors `_load_mask_video`, and it is NOT a function of its own:
`ltx2_video.cpp` composes it at the call site out of the SAME
`Ltx2ReadFrameDirectory` the reference clip uses plus `Ltx2MaskVideoFromPixels`,
which means over the three channels, remaps `(x + 1) / 2` and clamps to
`[0, 1]`. It is read at the STAGE-1
resolution, which upstream spells `args.height // 2` at `ic_lora.py:460-461`
and this engine reads from the phase's own grid — the same number, derived from
the phase rather than assumed from the CLI.

`Ltx2DownsampleMaskVideoToLatent(mask, f_pix, h_pix, w_pix, latent_shape)`
mirrors `:52-84`: area-interpolate each pixel frame to `(h_lat, w_lat)`; keep
the first latent frame as the area-downsampled PIXEL frame 0 alone; mean-pool
the remaining `f_pix - 1` frames in groups of `t = (f_pix - 1) / (f_lat - 1)`;
flatten to `(f_lat * h_lat * w_lat)`. It refuses a non-divisible pair with
upstream's assertion text (`:74-77`), and it degenerates to the first frame
alone when `f_pix == 1` or `f_lat == 1` (`:81-82`).

The **causal carve-out is the whole content of the temporal half**: latent frame
0 is pixel frame 0 alone, not a pooled group. A uniform pooling separates from
it by 0.168 on the fixture in §5, so the golden can see it.

Then `attn_mask = latent_mask * conditioning_attention_strength` (`:156`).

`Ltx2ResolveCrossMask` and `Ltx2BuildAttentionMask` port `mask_utils.py` at
batch 1. The block structure is the load-bearing part:

```
                noisy(Nn)  prev_ref(N-Nn)  new_ref(M)
     noisy      existing   existing        cross
     prev_ref   existing   existing        0
     new_ref    cross      0               1
```

`Ltx2UpdateAttentionMask` reproduces `:141-156`: a null mask on a state that
already carries one pads the new tokens with ones rather than returning null,
which is what keeps a second reference item from shrinking the mask below the
sequence.

The wrapper's ORDER is ported as written (`attention_strength_wrapper.py:49-71`):
snapshot the pre-item state, apply the inner item, take `num_new_tokens` as the
difference, and build the mask against the ORIGINAL token count. Building it
against the post-append count would put the new block in the wrong place while
every shape check still passed.

`Ltx2LatentState` grows an `attention_mask` field and an
`attention_mask_rows` count. The header comment at `ltx2_conditioning.h:57-74`
that says the field is deliberately absent is rewritten rather than deleted: it
was true, this row is what makes it false, and a reader needs to know which.

### 4.3 The production assignment

`StreamState` carries the mask through the phase loop; `ToLatentState` and
`FromLatentState` carry it in both directions. At the DiT call site
(`ltx2_video.cpp`, where `Ltx2ModalityInput vin` is built) the mask is handed
over with its row count, guarded exactly as the `keyframes_mask` handover beside
it is: the size is CHECKED against `video.tokens` before `data()` is taken,
because an empty vector's `data()` is a null pointer and therefore upstream's
legal "no mask" — a silent drop dressed as a supported path.

### 4.4 Request surface

Three per-request extras, the established family-knob mechanism in this ABI
(`kLtx2RetakeStartTimeExtra` and twenty others):

| extra | upstream |
|---|---|
| `ref_video_strength` | the STRENGTH of `--video-conditioning PATH STRENGTH` (`ic_lora.py:416-425`), default 1.0 |
| `conditioning_attention_mask_dir` | the MASK_PATH of `--conditioning-attention-mask` (`:427-441`), a `frame_%06d.ppm` directory |
| `conditioning_attention_strength` | its STRENGTH (`:452-455`), default 1.0, refused outside `[0, 1]` exactly as `ic_lora.py:230-233` |

`ref_video_dir` already exists on `VideoGenParams`.

## 4a. Risks

* **R1 — a degenerate golden.** A `build_attention_mask` fixture without a PRIOR
  conditioning item has `num_existing == num_noisy`, and then the true block
  structure and the plausible wrong one "cross on ALL existing rows" are
  ELEMENTWISE EQUAL: measured 0 separating elements. Such a golden is a mute
  switch. Mitigation: the generator asserts `separating > 0` for every emitted
  case and REFUSES to write the file otherwise, and the fixture carries a prior
  item so the count is 4.
* **R2 — the mask is too wide to notice being wrong.** A `[T, T]` mask of ones
  is the identity, and a bug that produces all-ones renders correctly. Mitigation:
  the reachability gate compares a masked render against an unmasked one through
  the ABI and requires the pixels to MOVE, and a separate assertion requires the
  built mask to contain a value strictly between 0 and 1.
* **R3 — the routing predicate and the refusal predicate drifting apart.** The
  campaign has shipped one silently wrong answer this way. Mitigation: one named
  local, used by both, and a mutation that changes only the refusal.
* **R4 — dtype.** Upstream runs this whole path at bf16 (`self.dtype`,
  `ic_lora.py:271-281`; `torch.bfloat16` at `ic_lora.py:530`). This tree's LTX
  host path is f32 throughout, which is the A24 campaign's standing divergence.
  This row mirrors the tree, names the width beside each new buffer, and records
  it owed. A token gate cannot see this and neither can these goldens.
* **R5 — anchor rot.** `ltx2_video.cpp`'s READER ANCHORS comment has rotted ten
  times. This row's edit shifts lines in that file. The anchors are re-derived
  with the test case's own rule after the edit, and re-derived AGAIN after any
  later edit in the same file.

## 5. Tests and gates

Every gate below is RUNNABLE from the repository root and all of them were run.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=OFF
cmake --build build -j 2 --target \
  test_diffusion_device_seam test_ltx2 test_ltx2_device test_ltx2_dfr \
  test_ltx2_iclora_reference test_ltx2_image_cond test_ltx2_loader test_ltx2_lora \
  test_ltx2_pipeline test_ltx2_retake test_ltx2_text_encoder test_ltx2_tiling \
  test_ltx2_vae test_ltx2_video test_ltx2_video_device_forward ltx2-gen
ctest --test-dir build -j 2 --output-on-failure -R 'ltx2|diffusion'
python3 scripts/check-pr-size.py --base origin/main --head HEAD --branch row/LTX25-IC-LORA-REF-VIDEO
python3 scripts/agent-issue-index.py --refresh && python3 scripts/check-agent-record.py
scripts/agent-preflight.sh
```

**THE WHOLE-TREE GATE IS NO LONGER PENDING.** This spec recorded it as a
PENDING result on an external resource, because a `kill -9` during the row's
scheduling had corrupted `build/.ninja_deps`, ninja discarded its dependency
graph and scheduled all 1338 edges, and under `-j 2` on a box carrying four
other compiling agents at load 20-33 that did not finish — it reached 76 of
1338. **It has now been run twice on an idle box** and it is a pass, so the
PENDING is discharged locally and not only by CI:

* Fresh review, on head `975bc0906`: full build rc 0, then
  `ctest -j 2` — **100% tests passed, 0 tests failed out of 714**, 7 skipped.
* This repair round, on the merged head: full build rc 0 over all 1337 edges,
  then `ctest --test-dir build -j 2 --output-on-failure` — **100% tests passed,
  0 tests failed out of 714**, rc 0, the same 7 skipped. Every skip is
  GPU-gated or checkpoint-gated and none is in this row's reach.

Per-suite on the merged head: `test_ltx2_iclora_reference` 102/102 over 14
cases, `test_ltx2_video` 5237/5237 over 131, `test_ltx2` 4581/4581,
`test_ltx2_pipeline` 4726/4726, `test_ltx2_retake` 69/69.
`scripts/agent-preflight.sh` rc 0 with 155 `ok` and 5 `SKIP`, every skip being a
checker preflight cannot supply arguments for; `check-pr-size.py` was therefore
run explicitly, rc 0. `check-agent-record.py` rc 0 with the snapshot PRESENT,
refreshed by `agent-issue-index.py --refresh` (316 open issues).

The fifteen suites the recipe above names are still the right focused set — they
are every target that links a translation unit this change touches, plus
`ltx2-gen`, the shipped CLI that reaches the new knobs — but they are no longer
the whole of what was run.

**The goldens are generated by executing the pinned module**, never by reading
it: `scripts/gen-ltx2-iclora-reference-goldens.py` imports
`ltx_pipelines.iclora_utils` and `ltx_core.conditioning.mask_utils` from
`/home/mudler/_git/LTX-2` at `fd4ded7f` and writes
`tests/vllm/models/ltx2_iclora_reference_goldens.inc`. It emits, for every case,
upstream's answer AND the rejected hypothesis beside it, and asserts they
differ.

Measured separations, which are what makes each case able to fail:

| case | rejected hypothesis | separating |
|---|---|---|
| `build_attention_mask` with a prior ref | cross on ALL existing rows | 4 elements |
| the same with `num_existing == num_noisy` | the same | **0 — REFUSED by the generator** |
| `downsample_mask_video_to_latent` | bilinear spatial | 0.20678424835205078 |
| the same | uniform temporal pooling, no causal carve-out | 0.049749284982681274 |
| the same at 9 -> 2 | an integer-stride box filter | 0.08158010244369507 |
| `_prepare_self_attention_mask` on `[1, 0.5, 0, 1e-30]` | — | `[0, -0.6931471824645996, -3.4028234663852886e+38, -69.07755279541016]` |

**Reachability.** `test_ltx2_video` renders through the public `Generate` on the
reduced-dimension checkpoint fixture, once without a reference clip and once
with one, and requires the pixels to differ; then once more with a mask and
requires all three to differ pairwise. The mutation that must red it is deleting
the production call site, not the helper.

## 6. Mutations

Twelve, each applied by `scripts`-external `mutate.py`, which REFUSES when its
`old` text is not found and asserts the edit took; each then COMPILED, both
suites run, and the tree restored with `git checkout --` plus a `git status
--porcelain` emptiness assertion. `unit` is `test_ltx2_iclora_reference`
(97 assertions green), `reach` is `test_ltx2_video --test-case='ltx2 ic-lora*'`
(169 assertions green).

**The counts in this table are AS MEASURED, before fresh review.** The review
round that followed added the non-dividing pooling golden and the no-mask range
refusal, so `unit` is now 102 assertions over 14 cases and the full
`test_ltx2_video` is 5237 over 131. The table is not rewritten to those numbers,
because a mutation result is a measurement and restating it against a tree it
was not run on is exactly the drift these tables exist to prevent. Only M1's
count is corrected below, and it is corrected because it was RE-MEASURED.

| # | mutation | unit | reach | verdict |
|---|---|---|---|---|
| M1 | delete the production `vin.attention_mask` assignment | 97/97 pass | **3 failed** | DETECTED — the reachability property |
| M2 | never apply `Ltx2ConditionVideoByReference` | 97/97 pass | **threw** | DETECTED |
| M3 | `temporal_subsample` steps from 0 | **21 failed** | 169/169 pass | DETECTED |
| M4 | drop the causal first-frame carve-out | **1 failed** | 169/169 pass | DETECTED |
| M5 | cross block on ALL existing rows | **3 failed** | 169/169 pass | DETECTED |
| M6 | drop `scale != 1` from the divisibility guard | 97/97 pass | 169/169 pass | **SURVIVED — IDENTITY** |
| M7 | read the reference at the phase's own grid | **4 failed** | **1 failed** | DETECTED |
| M8 | apply the reference on every phase | 97/97 pass | **1 failed** | DETECTED |
| M9 | never multiply the mask by the strength | 97/97 pass | **1 failed** | DETECTED |
| M10 | an append no longer extends the mask | **1 failed** | 169/169 pass | DETECTED |
| M11 | the refusal takes a second predicate | 97/97 pass | 169/169 pass | **SURVIVED — IDENTITY** |
| M12 | broadcast a short 1-D mask instead of refusing | **1 failed** | 169/169 pass | DETECTED |

**Ten of twelve detected. Neither survivor is a blind spot, and both are
identities provable rather than argued.**

**M1 reds THREE assertions, not four.** This spec recorded four; fresh review
re-ran it and measured three, at `test_ltx2_video.cpp:14097`, `:14109` and
`:14124`. The fourth is worth naming because of WHICH one it is:
`trace.ic_lora_attention_mask_rows == 16` still PASSED under the mutant. The
trace records that the mask was built and how many rows it has, and the deleted
line is the one that hands it to the model — so the counter is still correct
about a mask nothing consumes. **Only the pixel compares can see the drop.** A
future reader tempted to gate this seam on the trace alone, because a trace
assertion is cheaper than a render, would be gating the construction and not the
consumption.

**M6.** The mutant drops `downscale_factor != 1` from
`downscale_factor != 1 && (h % downscale_factor || w % downscale_factor)`. For
`downscale_factor == 1` the right conjunct is `h % 1 || w % 1`, which is `0`
for every `h` and `w`. The two predicates therefore agree on every input, and no
fixture can separate them. Upstream writes the guard (`iclora_utils.py:112`) and
this port mirrors it; it is documentation of intent, not behaviour.

**M11.** The mutant replaces `serves_reference` in the refusal with
`im.pipeline_kind != "ic_lora"`. `serves_reference` is
`im.recipe.ic_lora_reference`, and that field is set by `IcLoraRecipe` alone,
which `ResolveLtx2PipelineRecipe` reaches only for `pipeline_kind == "ic_lora"`.
So over the recipe table AS IT STANDS the two expressions are the same
predicate. **It stops being an identity the moment a second recipe sets the
flag**, and this row therefore adds the tripwire for that day:
`test_ltx2_pipeline`'s recipe-table case walks every kind against all four
versions and requires exactly ONE resolved pair to carry `ic_lora_reference`,
naming `ltx2_video.cpp`'s refusals as what must be re-read. **The kind list is
DERIVED**, from every `pipeline_kind == "..."` branch in
`ltx2_pipeline.cpp` — the file a new kind has to be written into to exist. It was
a hand-copied eleven-entry array until fresh review, which is a list that is
complete only until the next kind lands: a twelfth kind would have gone unwalked
while the case went on reporting "exactly one". Measured rather than argued —
adding a twelfth branch to the resolver now reds the case with
`declares 12 kinds, not the 11 this tripwire was written against`, and the
assertion count rises by two, which is the new kind actually being walked. That converts an
unmeasured residual into a guarded one; it does not make M11 detectable today,
because today it changes nothing.

## Owed

| owed | issue |
|---|---|
| the EXR / HDR reference arm (`iclora_utils.py:120-139`), which needs an OpenEXR reader and `--hdr` colour-space resolution this tree does not have | [#3020](https://github.com/mudler/vllm.cpp/issues/3020) |
| the scalar-only `elif conditioning_attention_strength < 1.0` branch (`iclora_utils.py:157-158`). It is unreachable from upstream's CLI, because `conditioning_attention_strength` is assigned only inside `if args.conditioning_attention_mask is not None` (`ic_lora.py:454-455`) and is otherwise 1.0, so a strength below 1 always arrives with a mask. Python-API-only; refused by name | [#3020](https://github.com/mudler/vllm.cpp/issues/3020) |
| `tiled_encode` for the reference clip (`iclora_utils.py:145-146`), which this engine reaches for the target but not for a reference | [#3020](https://github.com/mudler/vllm.cpp/issues/3020) |
| the reference-AUDIO arm, unchanged by this row and still refused by name | [#3020](https://github.com/mudler/vllm.cpp/issues/3020) |
| the bf16 storage width of every buffer this row adds (R4) | the A24 dtype campaign |
| a real-weights IC-LoRA reference render; this row has no GPU authority and no checkpoint | [#3020](https://github.com/mudler/vllm.cpp/issues/3020) |
| `ref_video_strength` with no `ref_video_dir` is ACCEPTED and then never read. It is the same shape as the `conditioning_attention_strength` defect this round fixed, and it is left open rather than fixed because upstream cannot express it: `--video-conditioning` is `nargs=2` (`ic_lora.py:416-425`), so upstream has no reading of a strength without a clip to be mirrored, and inventing a refusal would be a product decision rather than a port | [#3020](https://github.com/mudler/vllm.cpp/issues/3020) |

## 7. Stop conditions

* Stop and report `NEEDS_DECISION` if the reference pixel path turns out to need
  a media decoder, a checkpoint, or a GPU lease. **This was settled first and it
  did not fire**: `Ltx2ReadFrameDirectory` is already `video_preprocess`,
  `Ltx2ConvVideoEncode` already runs multi-frame, and
  `tests/vllm/multimodal/ltx2_video_fixture.h` is a real reduced checkpoint that
  `Generate` runs end to end.
* Stop if a golden's generator cannot produce a case with a non-zero separation
  from its rejected hypothesis. An unfalsifiable golden is not a gate.
* Stop if the reachability mutation — deleting the production call site — leaves
  the suite green. That measures a class, not a capability.
* No GPU, no lease, no downloads. One compiling agent at `-j 2`.

## Now

`DONE` — landed on `row/LTX25-IC-LORA-REF-VIDEO`. A15 and A16 are served and
reached from `Generate`; the dead `Ltx2ModalityInput::attention_mask` seam has a
production writer. What stays owed is in `## Owed` and is refused by name.
