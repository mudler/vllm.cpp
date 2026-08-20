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
the reduced fixture plus the four real safetensors headers listed in section 2,
read by parsing their JSON prologue and no payload.

**The distilled bf16 transformer was not read.** It is the one file that would
settle section 2 by direct comparison, `Lightricks/LTX-2.5` is gated, and an
unauthenticated range request for its first 8 bytes answered `401` on
2026-08-20. Section 2.4 states exactly what that leaves unproven.

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

Four real files were read on 2026-08-20 by parsing the 8-byte length prefix and
the JSON header, and no payload byte.

| Short name | Path | Bytes | Header bytes | Tensors |
|---|---|---:|---:|---:|
| FULL-BF16 | `.../lightricks-ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | 677,616 | 4349 |
| DIST-NVFP4 | `.../lightricks-ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,432,024 | 1,179,408 | 7876 |
| DIST-FP8 | `.../vonkaiser-fp8-nvfp4/transformer/ltx-2.5-22b-distilled-fp8.safetensors` | 21,025,119,068 | 881,048 | 6124 |
| DIST-LORA | `.../lightricks-ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8,899,889,568 | 526,744 | 3320 |

All three transformers are semantically complete: `8 + header + max(data_offsets[1])`
equals the file size exactly, on each one. So every absence recorded below is a
real absence and not a truncated download.

### 2.1 `__metadata__` carries nothing

FULL-BF16 and DIST-NVFP4 both carry exactly four metadata keys: `config`,
`gemma_source_checkpoint`, `license`, `model_version`.

- `config` is **2199 bytes on both and byte-identical**: sha256 of the string
  begins `13be9edf16635af9` on each, and the parsed JSON compares equal.
- `model_version` is `2.5.0` on both.
- `gemma_source_checkpoint` is `{"ltx_version": "2.5.0", "gemma_version": "gemma4-12b-ltx-v1"}` on both.
- `license` is the same LTX-2.x Community License text on both.

DIST-FP8 carries **no `__metadata__` at all**, which is why the FP8 recipes in
`docs/USAGE.md` need `--dit-config`.

So the only metadata a loader could key on is identical across the class
boundary, or absent.

### 2.2 The one structural difference is NOT a class marker, and two facts falsify it

Comparing tensor NAME sets after stripping quantization sidecar suffixes, the
only name present in FULL-BF16 and absent from DIST-NVFP4 is
`model.diffusion_model.keyframes_abs_pos_embedding` (`BF16 [1, 4096]`). #1137
reports this and correctly declines to use it. Two measurements taken here say
why that decline was right:

1. **DIST-NVFP4's own `config` still declares it.** The byte-identical config in
   2.1 contains `"use_keyframes_abs_pos_embedding": true`, while the tensor is
   absent from the file. The config is therefore a verbatim copy that describes
   the architecture, not the build, and the tensor's absence is a
   quantization-time drop.
2. **A third file named `distilled` CARRIES it.** DIST-FP8 holds
   `model.diffusion_model.keyframes_abs_pos_embedding` as `F8_E4M3 [1, 4096]`
   with an `F32` `..._scale` beside it, and its base tensor-name set is exactly
   FULL-BF16's 4349 names plus 1775 `_scale` sidecars — zero names in FULL-BF16
   are missing from it.

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

**One local copy already disagrees with the published artifact.** The tree
listing gives the NVFP4 transformer as 18,721,548,408 bytes. The local
DIST-NVFP4 above is 18,721,432,024, a difference of 116,384 bytes, and it is
internally complete. So a file under the published name on this project's own
NAS is not the published build. That is the same hazard `AGENTS.md` names for
re-quantized checkpoints, observed rather than argued, and it is why section 3
reads no filename.

### 2.4 What this leaves unproven

The direct comparison — FULL-BF16 against the distilled bf16, the pair that
matters most because they are the same size and the same dtype — was **not
made**. `Lightricks/LTX-2.5` is gated: an unauthenticated range request for the
first 8 bytes of `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`
returned HTTP `401` on 2026-08-20, and the file is 42 GB, so no cheap
authenticated path was available inside this row.

`docs/USAGE.md` currently asserts of that unread file that it is "4349 tensors"
with "the same four `__metadata__` keys". That claim has no measurement behind
it. This row corrects the sentence rather than inheriting it.

The conclusion this row acts on is therefore stated at its real strength: **no
field measured here separates the classes, and the one candidate is falsified
twice**. A detector built on any of these fields would be a guess, and #1137 is
explicit that a wrong detector is worse than none.

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
`Ltx2CheckpointClassRefusal` call from `Ltx2VideoEngine::Load`. Section 7
records it.

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
  closed by this change for the refusal half. The **detector** half stays owed
  and is not owed to a later revision of this design: section 2.4 names the one
  measurement that could reopen it, a header diff of the two bf16 transformers,
  which needs an authenticated fetch of a gated 42 GB file.
- `dmd2` has no stated checkpoint class in either reference, so it is the one
  kind this row does not gate (`kUnstated`).
- `dfr` and `res2s_two_stage` do not set `requires_distilled_lora`, although
  upstream's table gives both a `+ distilled LoRA` half
  (`CLAUDE.md:24`, `:26`; `dfr_pipeline.py:595` passes `args.distilled_lora`
  from `default_2_stage_arg_parser`). This row does not change that flag — see
  section 6 — and `ltx25-res2s-loop.md` already records the res2s half.
  Tracked by [#1445](https://github.com/mudler/vllm.cpp/issues/1445).

## Now

`READY`. Spec committed before the implementation on branch
`row/LTX25-CHECKPOINT-CLASS`.

## Outcome

Filled at `DONE`.
