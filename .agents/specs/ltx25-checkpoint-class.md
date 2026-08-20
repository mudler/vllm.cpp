# LTX-2.5 — refuse a checkpoint whose CLASS the pipeline cannot run

Row: `LTX25-CHECKPOINT-CLASS`. Owning matrix row:
[`MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`](../model-matrix.md).
Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned; **not edited by this
row**). Issue: [#1137](https://github.com/mudler/vllm.cpp/issues/1137).

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |
| vllm-project/vllm-omni (`vllm_omni/diffusion/models/ltx2`) | `a4ea67a21b20054dacc6e83952f9bd407e8ee4e7` |

Both read from local checkouts at those revisions. `git rev-parse HEAD` was run
on each before any anchor below was taken. The Lightricks pin is the one every
`ltx25-*` spec in this tree already carries; it has no
[`.agents/oracles/`](../oracles/) file, which is
[#1433](https://github.com/mudler/vllm.cpp/issues/1433) and not this row.

---

## 0. Honesty statement — what this row does and does not claim

This row makes the engine **refuse** a load whose checkpoint class the selected
pipeline cannot run. It does **not** detect the class. Section 2 is the
measurement that says why detection is not available, and the refusal message
says the same thing to the operator.

**No render on real weights is claimed.** No GPU lease was taken. The gate is
the reduced fixture plus the six real safetensors headers listed in section 2,
read by parsing their JSON prologue and no payload.

**The distilled bf16 transformer WAS read, and its header does not differ.** It
is the one file that settles section 2 by direct comparison. This row first
recorded it as unreadable, because `Lightricks/LTX-2.5` is gated and an
unauthenticated range request for its first 8 bytes answered `401` on
2026-08-20; the same request carrying this box's HuggingFace token answers `206`,
and the whole header is 677,624 bytes off a 42 GB file. Section 2.4 is that
measurement. It **strengthens** the conclusion rather than narrowing it: the two
bf16 headers are byte-identical apart from the key order of the `__metadata__`
sub-map, so there is nothing left for a header detector to key on.

---

## 1. The defect

`ResolveLtx2PipelineRecipe` keys on `(pipeline_kind, model_version)` alone
(`ltx2_pipeline.cpp::ResolveLtx2PipelineRecipe`). Nothing in the load path asks
which **class** of transformer the caller handed over. Upstream's own pipeline
table says that most pipelines cannot run on a distilled one:

`packages/ltx-pipelines/CLAUDE.md:17-30 @ fd4ded7f`, the `Model` column, read in
full:

| Pipeline | Model |
|---|---|
| `TI2VidOneStagePipeline` | `Full` |
| `T2AOneStagePipeline` | `Full` |
| `TI2VidTwoStagesPipeline` | `Full + distilled LoRA` |
| `TI2VidTwoStagesHQPipeline` | `Full + distilled LoRA (both stages)` |
| `A2VidPipelineTwoStage` | `Full + distilled LoRA` |
| `KeyframeInterpolationPipeline` | `Full + distilled LoRA` |
| `DFRPipeline` | `Keyframe-slot SFT + distilled LoRA (+ detailing IC-LoRA stage 2)` |
| `DistilledPipeline` | `Distilled only` |
| `ICLoraPipeline` | `Distilled only` |
| `DubItPipeline` | `Distilled only` |
| `RetakePipeline` | `Full or distilled` |

**Eleven rows, not the ten #1137 quotes.** The issue's copy of this table omits
`DFRPipeline`, and this tree ships a `dfr` kind. Section 3 keys off the eleven.

`t2a_one_stage.py:50 @ fd4ded7f` states the same thing for one pipeline in
prose: *"Assumes full non distilled model is provided in the checkpoint_path."*

**The consequence is a wrong render, not a failure.** The model loads, every
shape matches, and the sampler runs a schedule the weights were not trained for.
The clip has the requested size, the requested frame count and the requested
sample rate. Every output check this tree owns — pixel variance, waveform
energy, RMS, spectral — passes on it. `ltx25-ti2vid-recipe.md:395` and
`ltx25-keyframe-interp.md:438` both already record the same sentence about
#1137: *"picture, no diagnostic"*.

**One flag adjacent to this exists and is not it.**
`Ltx2PipelineRecipe::requires_distilled_lora` says the load must carry an
**adapter file**. It says nothing about the transformer. The two are different
halves of the same `Model` column cell: `Full + distilled LoRA` is a class
(`Full`) and an adapter (`distilled LoRA`), and only the adapter half is gated
today (`ltx2_video.cpp`, the `requires_distilled_lora` refusal).

---

## 2. What separates the two checkpoints — measured, and the answer is NOTHING

The question this row had to answer before it could design anything: does the
safetensors header of a distilled LTX-2.5 transformer differ from a full one, in
any field a loader can read?

Six real files were read on 2026-08-20 by parsing the 8-byte length prefix and
the JSON header, and no payload byte. Four are on local disk. Two — the
distilled bf16 transformer and the PUBLISHED NVFP4 artifact — were read over the
network by authenticated HTTP range request, which is section 2.4's method.

| Short name | Path or source | Bytes | Header bytes | Tensors |
|---|---|---:|---:|---:|
| FULL-BF16 | `.../lightricks-ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | 677,616 | 4349 |
| DIST-BF16 | `Lightricks/LTX-2.5` @ `6c7e5e57`, `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`, RANGE-read | 42,018,190,584 | 677,616 | 4349 |
| DIST-NVFP4 | `.../lightricks-ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`, the LOCAL copy | 18,721,432,024 | 1,179,408 | 7876 |
| PUB-NVFP4 | `Lightricks/LTX-2.5` @ `6c7e5e57`, the same name PUBLISHED, RANGE-read | 18,721,548,408 | 1,287,600 | 7877 |
| DIST-FP8 | `.../vonkaiser-fp8-nvfp4/transformer/ltx-2.5-22b-distilled-fp8.safetensors` | 21,025,119,068 | 881,048 | 6124 |
| DIST-LORA | `.../lightricks-ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8,899,889,568 | 526,744 | 3320 |

Every transformer here is semantically complete: `8 + header + max(data_offsets[1])`
equals the file size exactly, on each one, including both range-read files, whose
size comes from their own `Content-Range` denominator. So every absence recorded
below is a real absence and not a truncated download.

### 2.1 `__metadata__` carries nothing

FULL-BF16 and DIST-NVFP4 carry the same four metadata keys and **every one of
the four values is byte-identical**. Compared as raw strings, with the sha256 of
each string taken on both files:

| key | bytes | sha256 (first 16) on each file | equal |
|---|---:|---|---|
| `config` | 2199 | `13be9edf16635af9` | yes |
| `gemma_source_checkpoint` | 62 | `440f5e30dea8891e` | yes |
| `license` | 34562 | `4912471e6a71cc46` | yes |
| `model_version` | 5 | `f0ea2fc99f3b79d3` | yes |

`model_version` is `2.5.0` and `gemma_source_checkpoint` is
`{"ltx_version": "2.5.0", "gemma_version": "gemma4-12b-ltx-v1"}` on both. So the
whole `__metadata__` map is one and the same map on a full checkpoint and on a
distilled one, and it survives re-quantization unchanged.

**DIST-BF16 makes the same statement without the re-quantization step**, which
is the stronger form of it: all four values compare equal to FULL-BF16's, byte
for byte, on two files of the same dtype and the same size that differ only in
class. Section 2.4 is that comparison in full.

DIST-FP8 carries **no `__metadata__` at all**, which is why the FP8 recipes in
`docs/USAGE.md` need `--dit-config`.

**One file carries a fifth key, and it is not a class signal.** PUB-NVFP4 holds
`_quantization_metadata` beside the four, and the local DIST-NVFP4 copy of the
same name does not. It says the file was quantized, which its own dtypes already
say, and both files are distilled builds, so it separates two BUILDS of one
class rather than the classes. Recorded because a reader who greps for a fifth
metadata key will find it and should not have to work out what it is.

So the only metadata a loader could key on is identical across the class
boundary, or absent, or a quantization marker.

### 2.2 The one structural difference is NOT a class marker, and four facts falsify it

Comparing tensor NAME sets after stripping quantization sidecar suffixes, the
only name present in FULL-BF16 and absent from the LOCAL DIST-NVFP4 copy is
`model.diffusion_model.keyframes_abs_pos_embedding` (`BF16 [1, 4096]`). #1137
reports this and correctly declines to use it. Four measurements taken here say
why that decline was right, and the fourth is decisive on its own:

1. **DIST-NVFP4's own `config` still declares it.** The byte-identical config in
   2.1 contains `"use_keyframes_abs_pos_embedding": true`, while the tensor is
   absent from that file. The config is therefore a verbatim copy that describes
   the architecture, not the build, and the tensor's absence is a
   build-time drop.
2. **A third file named `distilled` CARRIES it.** DIST-FP8 holds
   `model.diffusion_model.keyframes_abs_pos_embedding` as `F8_E4M3 [1, 4096]`
   with an `F32` `..._scale` beside it, and its base tensor-name set is exactly
   FULL-BF16's 4349 names plus 1775 `_scale` sidecars — zero names in FULL-BF16
   are missing from it.
3. **The PUBLISHED file of the same name as DIST-NVFP4 carries it too.**
   PUB-NVFP4 holds it as `BF16 [1, 4096]` among 7877 tensors, against the local
   copy's 7876. So the absence in point 1 is a property of ONE build sitting on
   this project's NAS under a published name, and not of the NVFP4 arm, of the
   distilled class, or of anything a loader could generalise. That is section
   2.3's local-copy divergence again, now with the differing tensor named
   instead of only a byte-count delta.
4. **Both bf16 transformers carry it, at the same offsets.** FULL-BF16 and
   DIST-BF16 each hold `model.diffusion_model.keyframes_abs_pos_embedding` as
   `BF16 [1, 4096]` with identical `data_offsets`. On the one pair where a class
   marker would have decided anything, the tensor is present and identically
   DECLARED on both sides. Its payload was not read and is not claimed equal —
   the declaration is the whole of what a header detector could ever see, and
   the two declarations are the same.

This agrees with what this tree already says about that tensor: it is the
architecture-support flag for keyframe slots
(`ltx2.cpp`, the `use_keyframes_abs_pos_embedding` adoption), not a distillation
marker.

### 2.3 Size does not separate them either

`https://huggingface.co/api/models/Lightricks/LTX-2.5/tree/main/diffusion_models`,
read unauthenticated on 2026-08-20:

- `ltx-2.5-22b-dev-transformer-bf16.safetensors` — 42,018,190,584
- `ltx-2.5-22b-distilled-transformer-bf16.safetensors` — 42,018,190,584

Same byte count. The `lfs.oid` and `xetHash` fields are both returned as 64
literal `*` characters for every file in the repository, so the API gives no
content hash to an unauthenticated caller. The git `oid` of the two pointer
blobs does differ (`b0b8c2de...` against `3ba48d13...`), which proves the two
files are not the same bytes; it is not a fact any loader can read from the file
it was handed.

**One local copy already disagrees with the published artifact, and the
difference is now NAMED.** The tree listing gives the NVFP4 transformer as
18,721,548,408 bytes. The local DIST-NVFP4 above is 18,721,432,024, a difference
of 116,384 bytes, and it is internally complete. Both headers were then parsed —
the local one from disk, PUB-NVFP4 by range request — and they are 7876 tensors
against 7877, with `model.diffusion_model.keyframes_abs_pos_embedding` present
in the published build and absent from the local one, plus a
`_quantization_metadata` key the local copy does not carry. So a file under the
published name on this project's own NAS is not the published build, and the
difference is a whole tensor rather than padding. That is the same hazard
`AGENTS.md` names for re-quantized checkpoints, observed rather than argued, and
it is why section 3 reads no filename.

### 2.4 The direct comparison, made: the two bf16 headers are the same header

FULL-BF16 against DIST-BF16 is the pair that matters most, because they are the
same size and the same dtype and differ only in class. The comparison **was
made**, and the answer is that their headers are identical apart from one
serializer artifact.

**Method, recorded so the next reader does not repeat this row's mistake.**
`Lightricks/LTX-2.5` is gated, and an UNAUTHENTICATED range request for the
file's first 8 bytes answers HTTP `401`. This row first read that `401` as "no
cheap path exists" and deferred the measurement. It is not what a `401` means.
The box carries a HuggingFace token at `~/.cache/huggingface/token`; the same
request with `Authorization: Bearer <token>` answers `206`, and a safetensors
header is reachable in two of them:

```sh
# 1. the 8-byte little-endian header length, then 2. the header itself
Range: bytes=0-7          ->  206, 8 bytes,       Content-Range .../42018190584
Range: bytes=8-677623     ->  206, 677,616 bytes, Content-Range .../42018190584
```

677,624 bytes off a 42 GB file, per file, no payload byte and no `hf download`.
The file's own size arrives free in the `Content-Range` denominator, which is
what makes the completeness check above possible without a tree listing.

**What the comparison says**, with FULL-BF16 read the same way on the same day:

| Property | FULL-BF16 | DIST-BF16 | Equal |
|---|---|---|---|
| header bytes | 677,616 | 677,616 | yes |
| tensor names | 4349 | 4349 | yes, as SETS |
| per-tensor entry (`dtype`, `shape`, `data_offsets`) | — | — | yes, on all 4349 |
| `__metadata__` keys | 4 | 4 | yes |
| `__metadata__` values | — | — | yes, all four byte-identical |
| `keyframes_abs_pos_embedding` | `BF16 [1, 4096]` | `BF16 [1, 4096]` | yes, same offsets |
| `8 + header + max(data_offsets[1])` | 42,018,190,584 | 42,018,190,584 | yes, = file size |
| raw header bytes | — | — | **no** |

**The single difference is key ORDER inside `__metadata__`**: `model_version`
first on FULL-BF16, `gemma_source_checkpoint` first on DIST-BF16. The raw headers
first differ at byte 18 and last differ at byte 37,708 — the whole divergence is
inside that ~37 KB sub-map, and every byte after it is identical. Re-serialize
the sub-map with its keys sorted and the two headers compare EQUAL.

That order is not a field and cannot become one. Nothing in the safetensors
format fixes it, it is whatever the writing process's dictionary iteration
produced, and any re-save reorders it. A detector on it would fail on the first
file anybody round-trips, silently, in the direction that admits the wrong
checkpoint.

The conclusion this row acts on is therefore stated at its real strength, and it
is stronger than the version this section replaced: **nothing in either header
separates the classes, on the exact pair where a separation would have decided
the question, and the one structural candidate is falsified four ways** (2.2). A
detector built on any of these fields would be a guess, and #1137 is explicit
that a wrong detector is worse than none.

`docs/USAGE.md` previously asserted of this file, without a measurement, that it
is "4349 tensors" with "the same four `__metadata__` keys". That sentence was
withdrawn for the right reason — nothing had measured it — and the measurement
now agrees with it. Both facts are recorded, because "it turned out to be true"
is not a reason to have asserted it, and the page now says which of its
statements are measured and how.

---

## 3. Design

### 3.1 The two enumerations mirror the `Model` column

`include/vllm/model_executor/models/ltx2_pipeline.h`:

- `Ltx2CheckpointClass` — what a checkpoint IS, as the caller declares it:
  `kFull`, `kDistilled`, `kKeyframeSlotSft`.
- `Ltx2RequiredCheckpointClass` — what a recipe NEEDS, one enumerator per
  distinct checkpoint half of the `Model` column: `kFull`, `kDistilled`,
  `kFullOrDistilled`, `kKeyframeSlotSft`, plus `kUnstated`.

`kUnstated` exists for `dmd2` alone. `dmd2` comes from vLLM-Omni's
`_PIPELINE_RECIPES` (`ltx2_recipes.py:160-167 @ a4ea67a2`), whose table has no
`Model` column, and it has no row in Lightricks' table. Neither reference states
a class for it, so this row records that instead of inventing one. A silent
`kFullOrDistilled` would have read as a decision nobody made.

### 3.2 The recipe table

Set in each recipe factory in `ltx2_pipeline.cpp`, from the table in section 1:

| kind | upstream pipeline | `Model` column | requirement |
|---|---|---|---|
| `one_stage` | `TI2VidOneStagePipeline` | `Full` | `kFull` |
| `t2a_one_stage` | `T2AOneStagePipeline` | `Full` | `kFull` |
| `ti2vid_two_stage` | `TI2VidTwoStagesPipeline` | `Full + distilled LoRA` | `kFull` |
| `res2s_two_stage` | `TI2VidTwoStagesHQPipeline` | `Full + distilled LoRA (both stages)` | `kFull` |
| `a2vid_two_stage` | `A2VidPipelineTwoStage` | `Full + distilled LoRA` | `kFull` |
| `keyframe_interpolation` | `KeyframeInterpolationPipeline` | `Full + distilled LoRA` | `kFull` |
| `dfr` | `DFRPipeline` | `Keyframe-slot SFT + distilled LoRA` | `kKeyframeSlotSft` |
| `distilled_two_stage` | `DistilledPipeline` | `Distilled only` | `kDistilled` |
| `retake` | `RetakePipeline` | `Full or distilled` | `kFullOrDistilled` |
| `dmd2` | none | none | `kUnstated` |

The field defaults to `kFull`, the most demanding value any row takes, so a
recipe added later that forgets the field refuses rather than admits. The
default is not the gate: section 4's first case walks every `(kind, version)`
pair the table resolves and pins each requirement.

### 3.3 The caller declares; the engine never guesses

New load extra `checkpoint_class`
(`include/vllm/multimodal/ltx2_video.h`, `kLtx2CheckpointClassExtra`), spelled
`full`, `distilled` or `keyframe_slot_sft`.

`Ltx2CheckpointClassRefusal(recipe, pipeline_kind, declared, has_lora)` in
`ltx2_pipeline.cpp` returns the empty string when the load is acceptable and the
refusal text otherwise. Four refusals:

1. **Value not one of the three spellings** — names the three.
2. **Extra absent, and the recipe needs a specific class** — names the class the
   pipeline needs, states that no header field separates them, and names the
   extra to supply. This is the case #1137 calls "an honest refusal that does
   not pretend to a detector".
3. **Declared class does not satisfy the requirement** — names supplied, needed,
   and the upstream row.
4. **`retake` declared `full` with no adapter** — see 3.4.

`kUnstated` accepts everything, including an absent extra, and says so in the
comment beside it.

### 3.4 `Full or distilled` is a condition, not a hole

`RetakePipeline`'s docstring at `retake.py:71-73 @ fd4ded7f` gives the exact
condition: *"Set to `True` if using distilled model or passing distillation lora
with full model."* This tree's `RetakeRecipe` mirrors upstream's CLI, which
hard-codes `distilled=True` (`retake.py:336`, `:359`), and `:287` then takes
`DISTILLED_SIGMAS`.

So a `retake` load that declares `full` and carries no adapter runs the
distilled schedule on undistilled weights — the same defect one enumerator up.
`kFullOrDistilled` therefore requires a `lora_path` when the declared class is
`full`, and accepts `distilled` unconditionally. Without this condition the
permissive enumerator would be an escape hatch that reproduces #1137 on one arm.

### 3.5 Placement, and why it is AFTER the adapter refusal

The refusal sits in `Ltx2VideoEngine::Load`, immediately after the existing
`requires_distilled_lora` block. That order is deliberate: a two-stage load with
no adapter keeps getting the adapter message, which names a missing file, rather
than a class message that would send the reader to the wrong question.

The engine reads the extra and calls one function. The decision lives in
`ltx2_pipeline.cpp` beside the table it reads, so the recipe table and the rule
that consumes it cannot drift into two files.

---

## 4. Tests

`tests/vllm/models/test_ltx2_pipeline.cpp` — the table and the pure decision:

- Every `(kind, version)` pair `ResolveLtx2PipelineRecipe` resolves carries the
  requirement section 3.2 names. The pair list is built from the kinds and
  versions the table accepts, so a new row with no requirement fails here.
- `Ltx2ParseCheckpointClass` accepts the three spellings and rejects others.
- `Ltx2CheckpointClassRefusal` returns empty exactly on the accepting
  combinations and non-empty otherwise, including the `retake` adapter
  condition and `kUnstated`.

`tests/vllm/multimodal/test_ltx2_video.cpp` — the production entry point:

- **The red-first case.** `vllm_video_engine_load` with
  `pipeline_kind = one_stage` and `checkpoint_class = distilled` must fail with
  a message naming both classes. Before this row the same load **succeeds** and
  renders.
- A load that omits `checkpoint_class` on a `kFull` arm must fail and name the
  extra.
- A load that omits it on `distilled_two_stage` must fail, because the default
  kind is not exempt.
- `retake` with `checkpoint_class = full` and no `lora_path` must fail; with
  `distilled` it must load.
- `dmd2`'s exemption is pinned at the pure-function level rather than through
  the engine, because `dmd2` resolves only at model versions `2` and `2.3` and
  this fixture declares `2.5`.

The reachability mutation for this row is the deletion of the
`Ltx2CheckpointClassRefusal` call from `Ltx2VideoEngine::Load`. `## Outcome`
records it beside the guard mutations.

---

## 5. Risks

- **Every LTX-2.5 load now needs one more extra.** That is the point of the row
  and it is a breaking change for any caller of `vllm_video_engine_load` on this
  family. `docs/USAGE.md` gains the flag on every recipe it prints, and
  `ltx2-gen` gains `--checkpoint-class`.
- **A caller can declare the wrong class.** The engine cannot check the claim,
  and says so in the refusal text. What the row buys is that the wrong regime
  now needs a deliberate false statement instead of silence.
- **`dmd2` stays ungated.** Recorded above and under `## Owed`.

---

## 6. Stop conditions

Return `NEEDS_DECISION` rather than widening if the row would have to change
`requires_distilled_lora` on any recipe, or change which sigmas a recipe runs.
Both are behaviour changes on shipped, gated arms and belong to their own rows.

---

## Owed

- [#1137](https://github.com/mudler/vllm.cpp/issues/1137) — the row's own issue,
  closed by this change for the refusal half. The **detector** half stays owed,
  and it is worth naming the three signals that WOULD close it, so that a later
  reader has somewhere to start rather than re-deriving section 2:

  1. **An upstream flag.** The cheapest fix is not ours: one
     `__metadata__` key, or one field in the embedded `config`, that says which
     class the file is. Today the config is copied verbatim across builds
     (section 2.1), so nothing distinguishes them. This is a request to
     Lightricks, not a change here.
  2. **A pinned content digest.** A sha256 per published artifact turns "which
     class is this" into "which build is this", which is strictly stronger.
     `Lightricks/LTX-2.5` is gated and its unauthenticated tree API returns 64
     literal `*` characters for every `lfs.oid` and `xetHash`, so this needs an
     authenticated fetch — and, unlike the header in 2.4, a digest needs the
     whole 42 GB, so an authenticated RANGE request does not reach it. Owned by
     [#1048](https://github.com/mudler/vllm.cpp/issues/1048).
  3. **A payload-level signature.** Unknown, and now known to be the ONLY place
     left to look. Section 2.4 made the header half of the diff the earlier
     draft of this bullet said was unavailable, and it came back empty: the two
     bf16 headers are the same header. So a cheap discriminator, if one exists,
     is in the weights. Whoever tries it should say up front how many bytes it
     costs to read, because that is the property that decides whether a loader
     can afford it at all.

  Until one of those lands, the declaration is **load-bearing rather than
  belt-and-braces**: a wrong declaration is the only remaining path to the
  silent wrong-regime render, and nothing downstream can catch it.
- `dmd2` has no stated checkpoint class in either reference, so it is the one
  kind this row does not gate (`kUnstated`).
- **`dfr` has no obtainable base checkpoint, and no issue here can fix that.**
  Its class is `keyframe_slot_sft` and nobody publishes one: `CLAUDE.md:24` and
  `dfr_pipeline.py:157` name the class and never a file, and the authenticated
  `/api/models` listing of `Lightricks/LTX-2.5` @ `6c7e5e57` (2026-08-20) has 17
  files whose five transformers are all `dev` or `distilled`. This is external
  debt, so it is recorded rather than assigned: it clears when Lightricks
  publishes such a base, and the real-weight LTX-2.5 gate that would then use it
  is [#1048](https://github.com/mudler/vllm.cpp/issues/1048)'s. Until then
  `docs/USAGE.md` and `docs/FEATURES.md` say the arm cannot be fed, and this row
  does NOT loosen the refusal to make it reachable.
- `dfr` and `res2s_two_stage` do not set `requires_distilled_lora`, although
  upstream's table gives both a `+ distilled LoRA` half
  (`CLAUDE.md:24`, `:26`; `dfr_pipeline.py:595` passes `args.distilled_lora`
  from `default_2_stage_arg_parser`). This row does not change that flag — see
  section 6 — and `ltx25-res2s-loop.md` already records the res2s half.
  Tracked by [#1445](https://github.com/mudler/vllm.cpp/issues/1445).

## Now

`ACTIVE`. The spec is committed before the implementation on branch
`row/LTX25-CHECKPOINT-CLASS`, which is the commit order that proves spec-first.
The row reaches `DONE` when a fresh reviewer returns `PASS` on the immutable
head.

## Outcome

Landed as designed: a refusal, not a detector. What follows is what was
measured, what was rejected, and why each default has the value it has.

### What the row shipped

`Ltx2RequiredCheckpointClass` on the recipe, `Ltx2CheckpointClass` as the
caller's declaration, the `checkpoint_class` load extra, one call to
`Ltx2CheckpointClassRefusal` in `Ltx2VideoEngine::Load`, and
`ltx2-gen --checkpoint-class`. Ten recipe factories set the requirement
explicitly; none relies on the field's default.

### Red first, and the red is the defect

Captured at the pre-implementation tree (`include/`, `src/` and both test files
at the spec commit), with the five shipped engine cases appended and ONE textual
substitution: `vllm::multimodal::kLtx2CheckpointClassExtra`, which does not
exist yet, for its literal spelling `"checkpoint_class"`. No assertion was
changed.

**The site count in this paragraph read "12" and the shipped file has 39.** The
correction matters more than the number does, because a red captured against a
text that is not what ships is not evidence of anything. `grep -c` on
`tests/vllm/multimodal/test_ltx2_video.cpp` at the head returns **40** matches:
39 uses of the symbol, plus `test_ltx2_video.cpp:1298`, where the name appears
as the string literal `"kLtx2CheckpointClassExtra"` inside the `served_tokens`
anchor list and must NOT be substituted. So the substitution is 39 sites, and
the 40th is the reason a bare `grep -c` disagrees with it.

**How the red was re-established.** The fresh review of this head did not take
the capture on trust. It rebuilt the pre-implementation tree, took the HEAD's own
`test_ltx2_video.cpp`, applied the substitution at its 39 symbol sites, built,
ran, and got the counts this section records — `5 cases | 0 passed | 5 failed`,
`20 assertions | 6 passed | 14 failed`, the FATAL at the undeclared case with
`msg` empty. So the recorded red is now evidence about the text that ships,
which is what the wrong site count had put in doubt. Anyone re-running it should
check the count with BOTH greps rather than one, because the difference between
40 and 39 is the whole trap.

- `BUILT=YES`, `compile_err=0`, `git diff --stat` against the spec commit
  `1 file changed, 234 insertions(+)` — the test file alone.
- `5 cases | 0 passed | 5 failed | 96 skipped`, `20 assertions | 6 passed |
  14 failed`, `Status: FAILURE!`, exit 1. The case count moved off zero, so this
  is a run and not a filter that matched nothing.
- The defect itself is the FATAL in
  `ltx2 checkpoint class: an UNDECLARED load refuses instead of rendering` at
  `kind = one_stage`, `test_ltx2_video.cpp:10019` at this head — the capture's
  own `:9917` was taken before the file grew above it, which is #911 again:
  `REQUIRE_FALSE( msg.empty() )`
  with `msg` empty — the load SUCCEEDED. A `Full`-arm load that declared nothing
  rendered.
- This capture was taken TWICE: once on the first draft of the cases, and again
  on the FINAL shipped text after the undeclared loop changed, so the recorded
  red matches what lands byte-for-byte modulo the one substitution.
- The mismatch case failed on the message rather than on the throw:
  pre-change the engine refuses `checkpoint_class` as an *unknown load extra*,
  so `msg.find("FULL")`, `msg.find("CLAUDE.md:17-30")` and
  `msg.find("'distilled'")` all miss. Red for the intended reason and not for a
  bare "something threw".

### Green

| Gate | Result |
|---|---|
| `test_ltx2_pipeline` (whole binary) | 59 cases, 3413 assertions, 0 failed, `SUCCESS!`, exit 0 |
| `test_ltx2_pipeline -tc='ltx2 checkpoint class*'` | 3 cases, 97 assertions, 0 failed |
| `test_ltx2_video -tc='ltx2 checkpoint class*'` | 5 cases, 90 assertions, 0 failed |
| `test_ltx2_video` (whole binary) | 101 cases, 3599 assertions, **0 assertions failed**, 100/101 cases passed; the one case that did not is a transient `cannot write .../audio.wav` at 94% disk with three concurrent `test_ltx2_video` runs on the box, and it passes in isolation (1 case, 103 assertions, 0 failed, `SUCCESS!`, exit 0) |
| `test_capi` (run serially; it contends with `test_ltx2_video` over `/tmp`) | 65 cases, 654 assertions, 0 failed, `SUCCESS!`, exit 0 — run strictly after `test_ltx2_video` exited |

**Re-run after the review repairs and the FIFTH `origin/main` merge**, at
`2539a98c5`'s merge with `fdefb4529`, from a clean configure and a cold build
followed by an incremental one that recompiled 7 targets (`BUILT=YES`,
`compile_err=0` on both). The fifth merge is re-gated rather than inherited
because `fdefb4529` (LTX25-RESIDENCY-W0, #1441) edits `ltx2_video.cpp`,
`ltx2_video.h`, `test_ltx2_video.cpp` and `docs/USAGE.md`, four files this row
also touches:

| Gate | Result |
|---|---|
| `test_ltx2_pipeline` (whole binary) | 59 cases, **3415** assertions, 0 failed, `SUCCESS!`, exit 0 |
| `test_ltx2_pipeline -tc='ltx2 checkpoint class*'` | 3 cases, **99** assertions, 0 failed, exit 0 |
| `test_ltx2_video -tc='ltx2 checkpoint class*'` | 5 cases, 90 assertions, 0 failed, exit 0 |
| `test_ltx2_video` (whole binary) | **101 cases, 101 passed, 0 failed**, 4109 assertions, `SUCCESS!`, exit 0 |
| `test_capi` | 65 cases, 654 assertions, 0 failed, `SUCCESS!`, exit 0, started after `test_ltx2_video` had exited |
| `scripts/agent-preflight.sh` | 1 gate failed, **0 SKIPPED**: `audit-live-rows`, which is `main`'s own [#1468](https://github.com/mudler/vllm.cpp/issues/1468) (`KERNEL-DFLASH2-GROUPED-CONV` is an abandoned `ACTIVE` row, arriving with `028438e68`). This branch touches no matrix file. `commit-trailers` and `commit-style` both ran and both passed, which is what the merge existed to achieve |

Three numbers moved and each one has a reason rather than a shrug. The two
pipeline counts gained **+2 assertions**: the refusal message now carries the
header measurement, and `677,616` and `4349` are pinned so a string nobody
gates cannot drift. `test_ltx2_video` gained 48 assertions and one case against
the earlier run because the earlier run's I/O flake aborted a case partway; it
did not recur here, and 101 of 101 cases passed.

### Mutations

Each one was applied to the working tree, built, run, then restored from a byte
snapshot taken before the mutation and rebuilt. `git diff --stat` is against
that snapshot rather than against `HEAD`, so it shows the mutation alone.

| # | Mutation | Applied | Built | Result |
|---|---|---|---|---|
| M2 | `kFull` accepts any class | 1 ins / 1 del | YES, `compile_err=0` | RED: 1/3 cases, `REQUIRE_FALSE( msg.empty() )` at the mismatch case |
| M3 | an absent declaration returns "" | 1 ins | YES, `compile_err=0` | RED: 1/3 cases, `REQUIRE_FALSE( msg.empty() )` at the absent-declaration case |
| M4 | `retake` full-with-no-adapter condition disabled | 1 ins / 1 del | YES, `compile_err=0` | RED: 1/3 cases, `REQUIRE_FALSE( msg.empty() )` at the `kFullOrDistilled` case |
| M5 | `dfr`'s row set to `kDistilled` | 1 ins / 1 del | YES, `compile_err=0` | RED: 2/3 cases, 4 assertions including `recipe.checkpoint_class == row.required` |
| M7 | `dmd2`'s `kUnstated` exemption removed | 1 ins / 1 del | YES, `compile_err=0` | RED: 1/3 cases, the `dmd2` accept assertion |
| M6 | **reachability**: the `Ltx2CheckpointClassRefusal` call deleted from `Ltx2VideoEngine::Load` | 1 ins / 1 del | YES, `compile_err=0` | RED on `test_ltx2_video`: 4 of 5 cases, `REQUIRE_FALSE( msg.empty() )` at four sites. GREEN on `test_ltx2_pipeline`: 3 cases, 97 assertions, `SUCCESS!` |

M6 is the reachability evidence `AGENTS.md` "Nothing lands dead" asks for. Every
engine case enters at `LoadVideoEngine` or `vllm_video_engine_load`; the
pure-function cases in `test_ltx2_pipeline` stay green under M6, which is
exactly the point — they measure the function, not the capability.

### What was rejected, and why

**A detector on `keyframes_abs_pos_embedding`.** It is the only structural
difference between the full bf16 file and the distilled NVFP4 one, and it is
falsified twice (section 2.2). Shipping it would refuse a correct load of the
FP8 distilled build and admit a wrong one, with a green check beside it.

**A size check.** The two bf16 transformers are the same byte count
(section 2.3).

**A filename check.** A local copy of the NVFP4 transformer on this project's
NAS is 116,384 bytes smaller than the published artifact of the same name and is
internally complete, so a different build already exists here under an unchanged
name.

**Making `checkpoint_class` optional, or a warning.** Silence is the defect. A
warning is silence with a log line: the render still completes and the artifacts
still pass every check this tree owns.

**A `kFullOrDistilled` that accepts anything.** `retake` would then run
`DISTILLED_SIGMAS` on undistilled weights, which is #1137 on one arm. The
condition mirrors `retake.py:71-73`.

**Changing `requires_distilled_lora` on `res2s_two_stage` and `dfr`.** Found
while reading the table; it went to
[#1445](https://github.com/mudler/vllm.cpp/issues/1445) and section 6's stop
condition rather than into this diff.

The reason first recorded here — *"it refuses loads that succeed today on two
shipped arms"* — is a weak one, and the correction matters because a weak reason
invites the next reader to overturn a decision that is right. **This pull
request already refuses every load that succeeds today**: any load omitting
`checkpoint_class` is refused on nine of ten kinds. "It would break working
loads" therefore separates nothing. The real reason is **blast radius**. Setting
that flag changes an EXISTING field on arms that are already gated, whose tests
assert an adapter-less load succeeds, and whose behaviour change is a sampling
decision rather than a declaration; this row adds a new field and touches no
existing one. Section 6 named that boundary as a stop condition **before** the
implementation started, in the committed spec, which is what makes the deferral
a plan rather than a retreat from work that turned out to be inconvenient.

**#1445's premise holds in this port, and it is worth writing down where the
next reader will look for it.** `Res2sTwoStageRecipe` gives stage 2
`Stage2DistilledSigmas()` (`ltx2_pipeline.cpp:1457`, `:1460`) and leaves
`requires_distilled_lora` false, which the comment beside its
`checkpoint_class` line states outright. `DfrRecipe` (`:1303`) inherits
`DistilledTwoStageRecipe`'s (`:1229`) schedule on both stages and
leaves it false as well. So the arm this row ships **accepts** a full
transformer running a distilled stage-2 schedule with no adapter on
`res2s_two_stage` — structurally the same hazard that `retake`'s
`kFullOrDistilled` condition refuses. That is #1445's whole content, and it is
now grounded in this tree rather than only in upstream's table.

### Why each default has its value

- `Ltx2PipelineRecipe::checkpoint_class` defaults to `kFull`, the most demanding
  value any row takes, so a recipe added later that forgets the field refuses
  rather than admits. The gate is the cross-product case, not the default.
- `kUnstated` is set explicitly on `dmd2` and nowhere else, so the one ungated
  kind is greppable rather than implied by an absent line.
- `FixtureCheckpointClass` in the test resolves the class FROM the recipe table
  instead of listing it per case, because a per-case list would be a second copy
  of the table that could disagree without anything saying so.
- The refusal sits AFTER the `requires_distilled_lora` one. A two-stage load
  with no adapter fails both, and the adapter message names a missing file the
  caller can fetch.

### Limits

- No GPU lease was taken and no render on real weights is claimed. The gate is
  the reduced fixture plus six real safetensors headers.
- The engine cannot check the declaration. A false declaration still renders in
  the wrong regime.
- The distilled bf16 transformer WAS read (section 2.4), and its header does not
  differ from the full one. An earlier draft of this bullet left open the
  question *"if it ever becomes readable and its header does differ, a detector
  becomes possible"*; it is readable, it does not differ, and no header detector
  is possible on that pair. What stays unmeasured is its PAYLOAD, and the
  `## Owed` bullet 3 above says that is now the only place a discriminator could
  be.
- **`dfr` cannot be fed here at all**, and that is an artifact gap rather than a
  gate gap. Its class is `keyframe_slot_sft`; upstream names the class and never
  an artifact (`CLAUDE.md:24`, `dfr_pipeline.py:157`); and the authenticated
  `/api/models` listing of `Lightricks/LTX-2.5` @ `6c7e5e57` on 2026-08-20 gives
  17 files, of which the five transformers are all `dev` or `distilled`. So the
  refusal this row ships is, for that one kind, the whole of what a user can
  reach. `docs/USAGE.md` and `docs/FEATURES.md` both say so rather than
  advertising a runnable arm, and the row deliberately did NOT loosen the
  refusal to compensate: a declaration a user cannot honestly make is a missing
  checkpoint, and pretending otherwise would put #1137 back on the one arm whose
  weights nobody has.
