# LTX-2.5 — retire the arms that do not exist, and refuse the extra that is ignored

Row: `LTX25-RETIRE-DEAD-ARMS`. Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned;
not edited by this row). Issues:
[#644](https://github.com/mudler/vllm.cpp/issues/644) items D–I and N,
[#611](https://github.com/mudler/vllm.cpp/issues/611).
Pattern this row is an instance of: [#604](https://github.com/mudler/vllm.cpp/issues/604).

Upstream pins:

| Reference | Revision | Local checkout verified at |
|---|---|---|
| Lightricks/LTX-2 | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` | `git rev-parse HEAD`, 2026-08-13; re-verified 2026-08-14 |
| huggingface/diffusers | `3a2f35d4efa4c059c8bfb3bc0d6c906264895c81` | `git rev-parse HEAD`, 2026-08-13; re-verified 2026-08-14 |

Anchors here were derived from those two checkouts by this row. **Four of them
were wrong and were found by review, not by the row** — see §7, which records
what was not re-derived rather than claiming everything was. Where an ABSENCE is
asserted, §1.1 carries the command and the control hits that prove the command
ran; treat any absence in this file without that pairing as unverified.

## 0. What is wrong today

`Ltx2UnportedPipelineFeature` (`include/vllm/model_executor/models/ltx2_pipeline.h:580-587`)
is a ledger of seven arms this port refuses by name. Read as a whole it claims
more than is true, in four separate ways:

1. **One arm is fabricated.** `kMultishot` refuses "multishot generation" and its
   enum comment cites *"ltx-pipelines multishot entry points"*. No such entry
   point exists. Neither does the symbol, the string, or the concept.
2. **One arm's anchor names something upstream does not do.** `kCfgParallelism`
   is anchored to `ltx-pipelines/multigpu`, which contains no CFG batching at
   all — the four real parallelisms are orthogonal to guidance, and the recipe
   this port runs uses no guidance in the first place.
3. **One arm is stale.** `kVideoEngineWiring` says the end-to-end wiring "is
   phase L7, not L5". L7 landed in `cefacd2d0`.
4. **Five of the seven have no product call site.** Recording them as "refused by
   name" overstates what exists: nothing a caller can send reaches them. They are
   declared-out-of-scope markers, and the code did not say so.

Separately, and user-visible: `duration_head_path` is accepted by
`kKnownLoadExtras` (`src/vllm/multimodal/ltx2_video.cpp:257`) and **read by
nothing**. A caller who points at a duration head gets the recipe default and is
told nothing.

## 1. What the references actually contain

### 1.1 `multishot` — FABRICATED

Searched as a **subject**, not as our own phrasing (the self-confirming-grep trap,
#604) — and **every absence below was re-derived on 2026-08-14 with a POSITIVE
CONTROL in the same command**, after two of the claims in this section shipped
false and a third was offered as exhaustive when it was not. A
grep that returns nothing proves nothing on its own — the same command has to
return the hits you know are there. The two commands, verbatim:

```sh
# LTX-2 @ fd4ded7f — `multishot` ABSENCE, controls `scene cuts` /
# `single continuous take` / `DurationPredictor` in the SAME command.
git grep -n -i -E 'multi[_ -]?shot|scene cuts|single continuous take|DurationPredictor'
#   -> 0 lines matching multi[_ -]?shot, 24 control lines.
#   RE-MEASURED 2026-08-15: this said 22. The verbatim command returns 24 (0 either
#   way), and the count is the positive control, so a wrong control is a defect in
#   the evidence even when the conclusion holds. Split: `git grep -c` on the
#   control alternation alone is also 24, so all 24 are controls.
# LTX-2 @ fd4ded7f — every `shot` FILE, control `DurationPredictor`.
git grep -l -E '\bshots?\b'          # -> 7 files (see below)
# LTX-2 @ fd4ded7f — `storyboard` ABSENCE, control `shot list` same command.
git grep -n -i -E 'storyboard|shot list'   # -> 1 line, README.md:136, the control.
# LTX-2 @ fd4ded7f — what the `multi` prefix actually attaches to.
git grep -h -o -i -E 'multi[_ -]?(gpu|scale|stage|shot|clip|prompt|segment)' \
  | sort | uniq -c | sort -rn
#   -> 94 multigpu, 44 multi-GPU, 25 Multi-GPU, 4 MULTI_GPU, 2 multi-prompt.
# diffusers @ 3a2f35d4 — same absence in the ltx2 pipelines package.
git grep -n -i -E 'multi[_ -]?shot|scene cuts|hard cuts|single continuous take' \
  -- src/diffusers/pipelines/ltx2/
#   -> 0 lines matching multi[_ -]?shot, 4 control lines.
```

| Query | LTX-2 `fd4ded7f` | diffusers `3a2f35d4` |
|---|---|---|
| `multi[_ -]?shot` (case-insensitive) | 0 hits, alongside 24 control hits | 0 hits, alongside 4 control hits |
| `\bshots?\b`, whole repository | **7 files**, none a generation mode | — |
| `multi[_ -]?(gpu\|scale\|stage\|shot\|clip\|prompt\|segment)` | `multi[-_ ]?gpu` **167**, `multi-prompt` **2** — and nothing else | — |
| `scene`, whole repository | **three senses, none a generation mode** — see §1.1a | only the prompt-guidance sense inside `pipelines/ltx2/` (`scene cuts` ×3, `scene opens` ×4) |
| `storyboard` | 0 hits, alongside its control | 0 hits |

An earlier revision of this table said the `multi` prefix attaches to
`multi-GPU`/`multi_gpu` **only**. It does not: `multi-prompt` appears twice
(`ltx-pipelines/utils/args.py:618`, "Helps multi-prompt enhance"; and
`utils/denoisers.py:174`, "multi-prompt benchmark panels"). Both are *batching
several captions through one run* — the enhancer's KV cache and a benchmark
panel — not composing several takes into one output, so the conclusion is
unaffected and the "only" was still wrong. Recorded rather than quietly
corrected, for the same reason as everything else in this section.

The **seven** `shot` files, with what the word means in each. An earlier revision
of this list said five and presented that as exhaustive (review finding M1). The
cause is recorded because it is reusable: that query was
`\bshots?\b` **restricted to `*.py *.md *.json *.yaml`**, and upstream's prompt
files are `.txt`. The filter, not the pattern, hid the answer — and it hid
exactly the two files that matter most here, because they are the ones that ship
at inference. Re-run without a path filter it is seven:

- `ltx-core/duration_head/duration_head.py:1,5` — "predicts **shot** duration",
  i.e. the natural length of ONE camera take.
- `ltx-core/duration_head/__init__.py:1` — same.
- `ltx-core/text_encoders/gemma/encoders/prompts/gemma4_i2v_system_prompt.txt:13,14,17`
  — camera **shot type** in the prompt-enhancer's instructions.
- `ltx-core/text_encoders/gemma/encoders/prompts/gemma4_t2v_system_prompt.txt:11,12,15`
  — same.
- `ltx-pipelines/utils/blocks.py:804` — `DurationPredictor`, "Predicts **shot**
  duration (in frames)".
- `ltx-trainer/src/ltx_trainer/captioning.py:41,42,54,57` — captioning prose, for
  describing an EXISTING video as training data.
- `LTX-2/README.md:59,136` — an example prompt ("a medium close-up **shot**") and
  the prompting guide ("think like a cinematographer describing a **shot** list").

Upstream's "shot" is a single continuous take. There is no multi-shot generation
mode, and nothing that composes several takes into one output.

The nearest thing in either tree runs the other direction:
`ltx-trainer/captioning.py:54` tells the CAPTIONER "the video contains multiple
shots, describe each one in turn" — an instruction for describing an input video
during dataset preparation, not for generating one. Named here so the next porter
who greps `shots` does not have to re-derive that it is the trainer.

**The `shot` claim, corrected on the diffusers side too.** This section used to
call `utils.py:217` — "Shot type (exactly one: extreme wide shot / wide shot /
medium shot / medium close-up / close-up / extreme close-up)" — the "stronger
check" because it came from a second, independent reference. It is not
independent: that sentence is **verbatim** LTX-2's own
`gemma4_t2v_system_prompt.txt:12`, and `utils.py:251` is verbatim
`gemma4_i2v_system_prompt.txt:14`. diffusers vendored Lightricks' gemma4 system
prompts wholesale. It corroborates that upstream's `shot` is a camera shot type,
and it does so at a second pin, but it is one source quoted twice, not two
sources agreeing. diffusers' own contribution is the duration head's docstring,
`duration_head.py:83`, "Predicts the natural duration of the **shot** implied by
a caption", and its module list: `pipeline_ltx2`, `_condition`,
`_diffusion_decode`, `_hdr_lora`, `_ic_lora`, `_image2video`, `_latent_upsample`,
plus the components. None is multi-shot.

`ltx-pipelines`' actual entry points, from `docs/pipelines.md` ("Full reference
for all 11 pipelines") and the module list:
`ti2vid_one_stage`, `ti2vid_two_stages`, `ti2vid_two_stages_hq`,
`ti2vid_two_stages_mgpu`, `ti2vid_two_stages_hq_mgpu`, `distilled`,
`distilled_mgpu`, `ic_lora`, `hdr_ic_lora`, `a2vid_two_stage`, `t2a_one_stage`,
`dubit`, `retake`, `dfr_pipeline`, `keyframe_interpolation`. None is multi-shot.

### 1.1a `scene` — three senses, and the one that was missed twice

**This is the third #604 instance this row shipped, and the one that took three
review rounds to find** (review finding B1). The header, this table and the PR
body all said "the only `scene` hit is PySceneDetect in the TRAINER". That is
false at `fd4ded7f`, and it was false in a SHIPPED header — the failure scenario
being a porter who greps `scene`, finds "scene cuts" prohibited in an inference
prompt, and concludes we missed a multi-shot path our own header denied.

`scene` has three senses upstream, established by reading **every** hit of
`git grep -n -i scene` — no path filter, since a path filter is what hid the
`shot` answer above — rather than by grepping our own vocabulary. The anchors
below are representative of each sense, not a transcript of every line; the
command above is the transcript:

| Sense | Where | Is it a generation mode? |
|---|---|---|
| `scene-linear` HDR colour (scene-referred light) | `ltx-core/color/hlg.py:1,5,44,74,93,256`, `ltx-core/hdr.py:29,132,146,162,171`, `ltx-pipelines/docs/hdr.md:15,16,34`, `utils/media_io/{color_config,encode,exr}.py`, `utils/args.py:813,861`, `ic_lora.py:422`, `retake.py:178`, `hdr_ic_lora.py:18`, `CHANGELOG.md:24` | no — colour science |
| PySceneDetect splitting | `ltx-trainer/scripts/split_scenes.py`, `ltx-trainer/docs/*`, `ltx-trainer/pyproject.toml:28` (`scenedetect>=0.6.5.2`), `ltx-trainer/AGENTS.md:85`, `.claude/skills/train-model/**` | no — a **training-data** preprocessor, and the only CODE sense |
| narrative prose, i.e. prompt-writing guidance | **`ltx-core`, which ships at INFERENCE**: `text_encoders/gemma/encoders/prompts/gemma3_i2v_system_prompt.txt:5,6,18,20`, `gemma3_t2v_system_prompt.txt:1,5,24,25`, `gemma4_i2v_system_prompt.txt:7,15,23,27`, `gemma4_t2v_system_prompt.txt:5,13,21,25`; plus `README.md:136` and `ltx-trainer/captioning.py:39` | no — instructions to the prompt enhancer |

The third sense is the one the old sentence denied, and `README.md:136` is
**the same line** the header cites two clauses earlier for "a cinematographer
describing a shot list" — it reads "chronological descriptions of actions and
scenes".

**The disposition does not move; it is STRENGTHENED.** Read as a subject rather
than as a keyword, the shipped enhancer prompts instruct the model *against*
multi-shot output:

- `gemma3_i2v_system_prompt.txt:6` — "Inaccurate descriptions may cause scene cuts."
- `gemma3_i2v_system_prompt.txt:18`, `gemma3_t2v_system_prompt.txt:24` — "No
  timestamps or cuts: DO NOT use timestamps or describe scene cuts unless
  explicitly requested."
- `gemma4_i2v_system_prompt.txt:3` — "**Single continuous take — no hard cuts.**"

diffusers carries the same four lines (`utils.py:121,161,182,240`), because it
vendored the same prompts. An upstream whose own prompt enhancer is told to
produce one continuous take is affirmative evidence that no multi-shot generation
mode exists — which is what `kMultishot` claimed to refuse.

**Disposition: RETIRE.** This is a defect in our record, not a gap in our port.
There is nothing to owe, so recording it as owed is the error. The enumerator is
removed and the retirement recorded in the header, in this spec, and in the
commit message — which is where an exception's reason lives (AGENTS.md, "there is
no waiver registry").

### 1.2 `int8-convrot` — real absence, deliberately out of scope

The inference quantization kinds LTX-2 defines, exhaustively
(`ltx-pipelines/utils/quantization_factory.py:22-26`, a `str`-valued enum with an
`assert_never` at `:50`, the file's last line — re-derived 2026-08-14, review
finding L1. The block quoted below starts at the `class` statement on `:22` and
its four members are `:23-26`, which is the range the header comment and the
shipped refusal message cite; `:23-27` was wrong at both ends and this spec was
the last place still carrying it):

```python
class QuantizationKind(str, Enum):
    FP8_CAST = "fp8-cast"
    FP8_SCALED_MM = "fp8-scaled-mm"
    NVFP4_CAST = "nvfp4-cast"
    NVFP4_PREQUANT = "nvfp4-prequant"
```

No int8 arm. `convrot` / `conv_rot` / `quarot` / `spinquant` are 0 hits across the
repository, re-run at these pins.

**Correction, 2026-08-13 (review finding F2).** An earlier revision of this
section put `hadamard` in that 0-hit list and concluded "no rotation of any kind".
That was itself the #604 pattern this row exists to close — an absence asserted
from a grep of our own vocabulary — and it is false. LTX-2 vendors Tri Dao's
fast-Hadamard-transform family in `ltx-kernels`
(`csrc/ops/include/fast_hadamard_transform.h`, `_common.h`, `_special.h`), and it
is explicitly **quantization-coupled**: `DequantHadamardParamsBase:54`,
`QuantHadamardParamsBase:71`, and `QuantMax<int8_t> = 127.0`
(`fast_hadamard_transform_common.h:14`). A fast Hadamard transform *is* the "rot"
of rotation-based quantization, so the word we should have searched for was there
all along.

What the conclusion rests on instead, re-derived here:

- **Nothing calls it.** `hadamard_mult_thread` / `hadamard_mult_warp` are defined
  at `fast_hadamard_transform_common.h:102,123` and `_special.h:12,31,58,93` and
  invoked at **no site in `csrc/`**. The `fast_hadamard_transform_cuda`
  instantiations are commented out (`rms_norm_rope_cuda.cu:225-228`), and
  `DequantHadamardParamsBase` / `QuantHadamardParamsBase` / `QuantMax<int8_t>` are
  declared and referenced nowhere.
- **The live kernel that carries the Hadamard NAME does not rotate.** The only
  params struct in use is `NormRopeHadamardParamsBase`
  (`rms_norm_rope.cpp:12,19,95-98`, `rms_norm_rope_cuda.cu:188`), and
  `norm_rope_cvt_kernel` (`rms_norm_rope_cuda.cu:51-170`) is RMS-norm → RoPE →
  store, with no Hadamard step. The name is vestigial, inherited from the fork's
  ancestry.
- **And where it is instantiated it is fp8 or bf16, never int8**:
  `<at::BFloat16, at::Float8_e4m3fn>` and `<at::BFloat16, at::BFloat16>`
  (`rms_norm_rope_cuda.cu:219-223`), matching `rms_norm_split_rope_cuda.cu:196,200`.

Corrected statement, same conclusion: **upstream vendors a
quantization-coupled Hadamard family, wires none of it, and never exposes a
rotation as a `QuantizationKind`.** So `int8-convrot` is still unreachable
upstream — but by dead code, not by absence.

`int8` upstream is **unreachable, not absent**. Everything *wired* is trainer-only:

- `ltx-trainer/src/ltx_trainer/gemma_8bit.py:33-36` — bitsandbytes `LLM.int8()`
  for the Gemma backbone during LoRA training.
- `ltx-trainer/src/ltx_trainer/quantization.py:11-15` — optimum-quanto precisions
  (`int8-quanto`, `int4-quanto`, …) for the trainer.

**But one int8 kernel lives in an INFERENCE package** (review finding F1, and the
same false-absence pattern shipped inside the user-visible refusal message):
`ltx-kernels/src/ltx_kernels/blockwise/triton_ops.py` defines `_kernel` at
`:25-37` and `run_quantize_kernel` at `:40-50` — a per-row int8 quantize with fp32
scales, output dtype `torch.int8` at `:43`, not a `uint8` buffer — aliased
`rowwise_int_quantize_triton` at `:436`. It is **dead**: that alias is its only
reference anywhere in the repository, and `blockwise/functional.py:12-18`
re-exports five other names and not this one. Context for why it is there at all:
the package is a fork of Lightricks' int8 kernel library retargeted to
fp8/fp6/nvfp4, and its custom-op namespace is still literally `q8_kernels_ops`
(`functional.py:25`).

Every other `int8` match in the repository is `uint8`: pixel buffers, packed NVFP4
nibbles, block-streaming staging.

**Disposition: KEEP, re-anchored.** Unchanged by the two corrections above —
nothing wired upstream reaches int8, and no rotation is exposed as a quantization
kind. What changes is the sentence: the refusal now says int8 is UNREACHABLE
(trainer-only for anything wired, plus one dead kernel in `ltx-kernels`) instead
of claiming it appears "only in the trainer". Its *kind* also changes: it is a
declared-out-of-scope marker, not a reachable refusal.

### 1.3 CFG parallelism — the name describes something upstream does not do

**THE ORIGINAL VERSION OF THIS SECTION SAID `cfg` IS 0 HITS IN BOTH MULTI-GPU
TREES, AND IT WAS WRONG (#892, review finding F3).** The claim was scoped to
`ltx-pipelines/src/ltx_pipelines/multigpu/` and `ltx-core/src/ltx_core/multigpu/`,
which are the two SOURCE trees — and that path filter excludes
`ltx-pipelines/docs/multigpu/`, where the answer is written out in prose. Fourth
instance of #604 in this row, and the one that proves the row's own lesson twice
over: a path filter is an absence claim too, and this section is where that
sentence was written.

Re-derived without the filter, with the file list as the positive control in the
same pass:

```sh
# LTX-2 @ fd4ded7f — every `multigpu` path, no source/docs split.
git ls-files -- '*multigpu*'          # -> 33 files, the CONTROL
git grep -n -i cfg -- '*multigpu*'    # -> 5 lines, NOT 0
```

Three of the five are the incidental `model_cfg` local in a documentation code
block (`docs/multigpu/sequence-parallel.md:81,84,85`). **Two are substantive**,
and they are the ones that matter:

> `docs/multigpu/gemma.md:103-104` — "Batch-parallel is beneficial only when there
> is **more than one prompt to encode** — the typical **CFG** case, positive +
> negative (B=2 on 2 ranks = one prompt per rank, both forwards concurrent). The
> **distilled** pipeline runs **without CFG**: its `__call__` accepts a single
> `prompt` and no `negative_prompt`, so there is only one prompt to encode and no
> work to partition."

That paragraph is better evidence than the absence ever was. It says in upstream's
own words that the recipe this port runs has no CFG pair, which is the reason CFG
batching is inapplicable here — a reason, rather than a claim that a string does
not appear.

**And there are FOUR forms, not three.** The fourth sits in the very directory the
refusal cites:

| Form | Anchor | What it splits |
|---|---|---|
| Sequence parallel | `multigpu/sp_builder.py:25` (`SequenceParallelBuilder`), `ltx-core multigpu/transformer/sequence_parallel.py`, all-to-all attention | the token axis of one denoise step |
| Tiled data parallel | `multigpu/tdp_builder.py:25` (`TiledDataParallelBuilder`) | spatial tiles, **upscale stage only** |
| Distributed VAE decode | `ltx-core multigpu/vae/distributed_decoder.py:204-256` (`DistributedVideoDecoder.decode_video`) | latent tiles across ranks, driver blends |
| **Batch-parallel Gemma encoding** | `multigpu/bp_gemma_builder.py:42` (`BatchParallelGemmaBuilder`), wrapping `ltx-core multigpu/gemma/batch_parallel_wrapper.py` | a **prompt list** across ranks, one replica per rank |

The four `BuilderProtocol`/`DelegatingBuilder` implementors under
`ltx-pipelines/src/ltx_pipelines/multigpu/` are exactly these, verified by
`git grep -n 'BuilderProtocol' -- packages/ltx-pipelines/src/ltx_pipelines/multigpu/`.
The fourth is the closest thing upstream has to CFG batching, which is precisely
why omitting it made the "none of them is CFG batching" sentence weaker rather
than safer: the honest statement is that the form which WOULD carry a CFG pair is
the one upstream tells you not to use for the distilled pipeline.

Upstream states the purpose in its own words
(`ltx-pipelines/docs/multigpu/README.md:7`, inside the ⚠️ block at `:5-16`):

> **Multi-GPU (MGPU) is a latency tool, not a memory tool.**

and adds that the mutable transformer is a **full replica on every GPU**, so it
cannot make a checkpoint fit.

And CFG is not in the path this port runs at all. The distilled recipe denoises
with `SimpleDenoiser` at both stages (`ltx-pipelines/distilled.py:266`, `:295`),
documented as "**single transformer call, no guidance**"
(`utils/denoisers.py:3`); the guider it degenerates to is
`MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0)`
(`utils/denoisers.py:25-26`), "only runs the conditioned pass and returns cond
unchanged". A `cfg_scale` of 1.0 is one pass, so there is no second pass to place
on a second GPU.

Stronger still, and true even for the recipes that *do* guide: upstream's guided
path is not distributed either. `utils/denoisers.py:7-8` records that
`GuidedDenoiser` and `FactoryGuidedDenoiser` share `_guided_denoise`, "which
batches all guidance passes into a single transformer call". So upstream's CFG is
a batch dimension, never a second device — which is why no amount of `multigpu`
code was ever going to be CFG parallelism, and why the old enumerator name could
not have been right for any recipe.

**Disposition: RENAME + re-anchor.** `kCfgParallelism` → `kMultiGpuParallelism`,
anchored to the four real forms, with the reason it is out of scope stated as
what it is: a single-node multi-GPU **latency** feature, on a port whose target is
one GB10. The disposition never moved across the F3 repair; only the evidence
under it was wrong, and the corrected evidence supports it more strongly.

Gated by `test_ltx2_pipeline`'s "the multi-GPU marker note states the CFG evidence
correctly" (the header) and by the multi-GPU block in "ltx2 every out-of-scope
feature is refused BY NAME" (the shipped message), on the same shape as the int8
marker: the retired sentence must be absent and the replacement evidence present.
Neither can verify the upstream fact — there is no upstream checkout in this tree
— so the commands above, with their control, remain the derivation.

### 1.4 `kLoraFusion` — real upstream, correctly refused

Verified present: `ltx-core/loader/primitives.py:160`
(`class LoraPathStrengthAndSDOps(NamedTuple)`), exported at
`loader/__init__.py:14,46`, consumed by `loader/single_gpu_model_builder.py:21`
and `block_streaming/builder.py:34,90`, fused by `loader/fuse_loras.py`. The
anchor stands; only its *kind* is corrected (marker, not reachable refusal).

### 1.5 `kVideoEngineWiring` — LANDED

`cefacd2d0` ("feat(ltx-2.5): LTX-2.5 joint video+audio DiT, and a video seam that
is no longer MiniMax-only (#435) (#641)", 2026-08-13) shipped exactly this: the
composition through `vllm::multimodal::VideoEngine`, reachable through the C ABI
as video family `ltx-2.5`. `include/vllm/multimodal/ltx2_video.h:12` already
speaks of the refusal in the past tense.

**Disposition: RETIRE.** A refusal whose subject shipped is a false statement, not
a record of debt.

### 1.6 The five with no product call site

`grep` over `src/`, `include/`, `examples/`, `tests/` for every enumerator:

| Enumerator | Product call site |
|---|---|
| `kTemporalUpsampler` | `src/vllm/model_executor/models/ltx2_upsampler.cpp:395` |
| `kBetaScheduler` | `src/vllm/model_executor/models/ltx2_pipeline.cpp:199` |
| `kLoraFusion` | **none** |
| `kMultishot` | **none** |
| `kInt8ConvRot` | **none** |
| `kCfgParallelism` | **none** |
| `kVideoEngineWiring` | **none** |

**Re-derived at integration, 2026-08-15.** The table above was measured at this
row's base. `2e9d95e74` then landed the temporal-only x2 upsampler on the same
issue (#644) and renamed the enumerator `kTemporalUpsampler` →
`kSpatiotemporalUpsampler`, because only the BOTH-flags arm is still unported.
Re-derived against `origin/main` `9a8615672`, the reachable call site is
`src/vllm/model_executor/models/ltx2_upsampler.cpp:465` and the enumerator is the
spatiotemporal one; `kBetaScheduler` is unmoved at `ltx2_pipeline.cpp:199`. The
ported temporal arm is reachable from no shipped pipeline, which is why
`docs/FEATURES.md` still records it as gated and UNDRIVEN.

**THE SPLIT WAS ONE REACHABLE AND FOUR MARKERS, NOT TWO AND THREE (#889, review
finding F1).** The table above answers "does this enumerator appear in `src/`?",
and that is the wrong question. A refusal is reachable only if something CALLS
the function that holds it, and `kBetaScheduler`'s call site
`ltx2_pipeline.cpp:199` sits inside `Ltx2Schedule`, which has **zero** call sites
in `src/`, `include/` or `examples/`. The engine bypasses the dispatcher and calls
`Ltx2SigmaSchedule` directly, in `ltx2_video.cpp`'s phase driver; no ABI field, load extra or
CLI flag carries a scheduler kind. The only thing that ever trips the arm is
`test_ltx2_pipeline`'s "ltx2 the Beta scheduler is refused by name, never substituted", constructing the enumerator by
hand — the **test-only driver** shape that AGENTS.md `## Nothing lands dead` and
[`.agents/reachability.md`](../reachability.md) name.

The corrected table, with the question stated the way it has to be asked:

| Enumerator | Site of the refusal | Function holding it | Product callers of that function | Kind |
|---|---|---|---|---|
| `kSpatiotemporalUpsampler` | `ltx2_upsampler.cpp:465` | `Ltx2LatentUpsample` ← `Ltx2UpsampleVideoLatent` | `ltx2_video.cpp`, the refine phase | **reachable** |
| `kBetaScheduler` | `ltx2_pipeline.cpp:199` | `Ltx2Schedule` | **none** | marker |
| `kLoraFusion` | — | — | — | marker |
| `kInt8ConvRot` | — | — | — | marker |
| `kMultiGpuParallelism` | — | — | — | marker |

**UPSTREAM SETTLES IT, WHICH IS WHY THE REPAIR IS A RECLASSIFICATION AND NOT A
WIRING.** The alternative repair — route the engine through `Ltx2Schedule` so the
claim becomes true — would have to invent a scheduler-selection surface that
upstream does not have. At Lightricks/LTX-2 `fd4ded7f`, `BetaScheduler` is defined
at `ltx-core components/schedulers.py:91` and **constructed nowhere in the
repository**; all seven `ltx-pipelines` entry points hard-code `LTX2Scheduler()`
(`ti2vid_one_stage.py:81`, `ti2vid_two_stages.py:87`, `ti2vid_two_stages_hq.py:90`,
`a2vid_two_stage.py:78`, `t2a_one_stage.py:67`, `keyframe_interpolation.py:82`,
`retake.py:96`), verified by
`git grep -n 'LTX2Scheduler\|LinearQuadraticScheduler\|BetaScheduler'` with the
definition file excluded — a command whose own output is its control, since it
returns the seven `LTX2Scheduler()` constructions it must. vLLM-Omni `a4ea67a21`
has **zero** hits for the name, against 81 files containing `ltx` as the control.
Mirroring upstream therefore means this port has no scheduler-kind field either,
and `kBetaScheduler` records an unported upstream component rather than a refusal
a request can trip.

`Ltx2Schedule` itself is then an unreached dispatcher, disclosed under `## Owed`
and tracked by #893 as required by `## Nothing lands dead`. No wiring wave is
coming for it, because there is nothing upstream to wire it to; the open decision
is retire-or-keep.

**The classification is now derived rather than written.**
`test_ltx2_pipeline`'s "the reachable/marker split matches the source" walks
`src/`, `include/` and `examples/`, counts callers of each arm's entry function,
and requires the message to agree with what it finds — in both directions, so a
caller appearing later fails the marker wording just as its absence failed the
reachable wording. Two positive controls ride in the same walk, because this is an
absence claim about our own tree and this row has shipped four of those already.
It is the anti-tautological shape #691 asks for, applied to the beta arm; #691
stays open for the other three markers, which have no call site at all to derive.

The five are enumerated only by the `reachable`/`markers` vectors in
`test_ltx2_pipeline`'s "ltx2 every out-of-scope feature is refused BY NAME". The
line numbers that used to be written here are deliberately gone: they went stale
twice inside this pull request, and a case name is stable where a line range is
not. There is no request field, load extra, CLI flag or scheduler kind that asks
for a LoRA, an int8-convrot checkpoint, a second GPU or a Beta schedule, so no
caller can trip any of the four.

That is not a defect on its own — a declared boundary is worth having. The defect
is calling it a refusal. The header and the messages now distinguish:

- **reachable refusal** — a product path constructs the condition and throws, and
  something a user arrives through calls the function that holds it;
- **declared-out-of-scope marker** — a record of what upstream has and this port
  does not, reached only by the ledger test.

## 2. `duration_head_path` (#611)

### 2.1 The full `kKnownLoadExtras` audit

Every key the family accepts, and whether any code reads it. Reader anchors are in
`src/vllm/multimodal/ltx2_video.cpp` unless noted, **as of `3c6706cd0`** — this
table is a dated record, not a live one; see the note below it.

| Key | Constant | Reader | Status |
|---|---|---|---|
| `audio_prompt_embeds_path` | `kLtx2AudioPromptEmbedsExtra` | `:969`, `:982` | READ |
| `pipeline_kind` | `kLtx2PipelineKindExtra` | `:805` | READ |
| `model_version` | `kLtx2ModelVersionExtra` | `:789` | READ |
| `allow_unported_modules` | `kLtx2AllowUnportedExtra` | `:638` | READ |
| `max_phase` | `kLtx2MaxPhaseExtra` | `:807` | READ |
| `dit_config_path` | `kLtx2DitConfigPathExtra` | `:693` | READ |
| `prompt_embeds_valid_rows` | `kLtx2PromptValidRowsExtra` | `:1010` | READ |
| `encoder_config_path` | `kLtx2EncoderConfigPathExtra` | `:864` | READ |
| `upsampler_path` | (literal) | `:839` | READ |
| `duration_head_path` | `kLtx2DurationHeadPathExtra` | **none** | **was ACCEPTED AND IGNORED; now refused by name** |

Nine of ten are wired. `duration_head_path` is the only defect, so the sweep this
row owes is complete and closes the "the sweep that found this one did not cover
them all" clause of #611.

**These numbers rotted twice inside this one PR (review finding F3).** The first
set shipped off by 37 lines and named nine lines that read nothing — cited, worst
of all, *in the file they were wrong about*. Correcting them, a merge of
`origin/main` moved the real ones by another 25. A hand-written `file:line` is
stale by the next commit, so the anchors are no longer maintained by hand: the
`READER ANCHORS` comment in `ltx2_video.cpp` carries the one live copy, and
`test_ltx2_video`'s "the recorded reader anchors are the ones in the source" case
derives them from the source on every run and fails with the replacement list
printed. The table above stays because a spec is a dated record and the SHA is
named; the code comment is the live one.

One documentation gap found by the same sweep and fixed here: the paragraph in
`docs/USAGE.md` beginning "Two families are registered." lists the LTX-2.5 extras
and omitted `encoder_config_path` entirely, though it is defined and read.

That paragraph was cited as `docs/USAGE.md:1650-1654`, and it is not there — at
this row's head it starts at `:1949` and the extras list runs to `:1955`. A third
stale anchor in the same spec (review finding, nit). Named by its OPENING WORDS
here rather than re-pinned to a number, for the reason §2.1 already gives about
the reader anchors: this file moved twice inside one pull request, and a phrase
survives a merge that a line number does not.

### 2.2 Why it is inert

`ltx2_duration_head.h` / `ltx2_duration_head.cpp` port `DurationHead` and
`AttentionPooler` and gate them as bricks. Nothing in `ltx2_video.cpp` constructs
one. The AUTO-duration path (`resolve_num_frames`, `ltx-pipelines/utils/blocks.py`)
therefore cannot run, and `Generate` computes `frames` from
`duration_seconds * fps` directly (`ltx2_video.cpp:1152`).

The stated reason for this moved twice already (#604's pattern, and the finding
that produced #611). The reason recorded here is the current one: **no head is
constructed**, so a supplied path is a file the engine never opens.

### 2.3 Fix

Refuse `duration_head_path` **by name** when it is supplied and non-empty, with a
message that says the head is unported, that the recipe default would otherwise be
substituted silently, and what to use instead (`num_frames`, or `duration`, which
is exact arithmetic against the recipe frame rate).

Rejected alternative: dropping the key from `kKnownLoadExtras`. That produces
"unknown load extra", which is *wrong* — the key is defined by this family and its
meaning is understood; what is missing is the implementation. AGENTS.md requires
"a message naming the missing piece", and "unknown key" does not name it.

Rejected alternative: reading it and constructing the head. That is the real fix
and it stays owed — it needs the connector-output plumbing the head consumes
(`duration_head.py:89-118` takes audio and/or video connector token states), which
is a different row. Refusing is the cheap correct answer until then.

## 3. Scope

**In.**

1. Retire `kMultishot` and `kVideoEngineWiring`; record both retirements in the
   header, this spec, and the commit message.
2. Rename `kCfgParallelism` → `kMultiGpuParallelism` and re-anchor its message to
   the three real parallelisms plus the reason CFG is not in our path.
3. Re-anchor `kInt8ConvRot`'s comment and message to record the verified absence
   at these pins.
4. Split the ledger into reachable refusals and declared-out-of-scope markers, in
   the header comment, in the messages, and in the test.
5. Refuse `duration_head_path` by name (#611).
6. `docs/USAGE.md`: the extras paragraph, corrected on both counts.
7. Added by the review repair: correct the int8 sentence in the shipped refusal
   (§1.2), correct this spec's own false `hadamard` absence (§1.2), derive the
   reader anchors instead of writing them by hand (§2.1), and split the
   `docs/FEATURES.md` row so the public surface stops calling the two REACHABLE
   arms unrequestable (§1.6). No disposition changes.
8. Added by the THIRD review round: correct the false `scene` absence in the
   shipped header, in §1.1's table and in the PR body (§1.1a); correct the `shot`
   enumeration from five files to seven and record the path filter that caused it
   (§1.1); correct `quantization_factory.py:23-27` to `:22-26` (§1.2); replace
   §7's "every claim in §1 was re-derived" with what actually happened; and gate
   the header's retirement note with a RED-first case so the sentence cannot come
   back. No disposition changes — the `scene` evidence strengthens the
   `kMultishot` retirement rather than weakening it.

**Out.**

- `.agents/specs/ltx-2-5.md` — operator-owned. This row does not edit it. Its §2
  "Out" list still names `multishot`; correcting that is the operator's edit, and
  this spec is the record it would cite. Filed as
  [#692](https://github.com/mudler/vllm.cpp/issues/692) by the review repair so
  `main` stops carrying the claim unowned.
- Making the reachable/marker split gate REACHABILITY rather than message text.
  The ledger test asserts what each message says; nothing asserts that a marker
  still has no product call site, so giving `kLoraFusion` one would leave the test
  green while the shipped refusal told a caller it is not requestable immediately
  after they requested it. Filed as
  [#691](https://github.com/mudler/vllm.cpp/issues/691), with the extras inventory
  gate named as the anti-tautological shape to copy.
- Constructing a duration head, and the AUTO-duration path. Stays owed (#611
  remains open after this row, retitled by the fix rather than closed by it —
  see §7).
- `kTemporalUpsampler` (row `LTX25-TEMPORAL-UPSAMPLER`), image conditioning (row
  `LTX25-IMAGE-COND-FIX`), tiled decode (row `LTX25-TILED-DECODE`), AdaLN claims
  (row `LTX25-ADALN-CLAIMS`). No file in this diff is theirs.

## 4. Tests

RED first for the behavioural change; the records changes are gated by the ledger
test's own assertions.

1. **`ltx2 duration_head_path is REFUSED by name` (new,
   `tests/vllm/multimodal/test_ltx2_video.cpp`).** Builds a valid load, adds
   `extras["duration_head_path"]`, and requires a throw whose message names the
   key, names the duration head as the missing piece, and names the alternative.
   **RED before the fix**: the load succeeds, because nothing reads the key. That
   is the defect stated as a test.
2. **`ltx2 an accepted load extra is READ by something` (new).** Asserts the
   inventory of §2.1 does not silently grow: every key in `kKnownLoadExtras`
   either round-trips through a reader or is refused by name. Implemented as the
   two known-inert keys being refused and the rest being accepted, so adding a
   tenth decorative key fails.
3. **`ltx2 every out-of-scope feature is refused BY NAME`** (existing, updated).
   The list drops from 7 to 5 entries — a CHANGED COUNT, reported as such — and
   splits into `reachable` and `markers`, with the marker messages required to say
   they are not requestable. Adds a guard that no refusal message mentions
   `multishot` again.

Three more added by the review repair, each RED before its fix:

4. **The int8 marker's own evidence** (an addition to case 3). The message shipped
   the sentence "int8 appears upstream only in the trainer", which §1.2 shows is
   false. The case now requires the message NOT to contain it and to name
   `ltx-kernels`. **RED before**: both assertions failed against the shipped text.
5. **`ltx2 video: the recorded reader anchors are the ones in the source`** (new,
   `tests/vllm/multimodal/test_ltx2_video.cpp`). Derives each served key's reader
   line from `ltx2_video.cpp` itself and compares it with the `READER ANCHORS`
   comment, and separately requires every `kLtx2DurationHeadPathExtra` reference
   after the array to sit inside `CheckUnservedExtras`. It hard-codes no line
   number of its own, so it cannot go stale; the obligation it creates falls on
   whoever moves a reader, in the file they are already editing. **RED before**:
   `Recorded: []. Actual: [638 693 789 805 807 839 864 969 1010]`.
6. **`ltx2 docs/FEATURES.md never calls a REACHABLE refusal unrequestable`** (new,
   `tests/vllm/models/test_ltx2_pipeline.cpp`). The public surface had re-merged
   the reachable/marker split into one "Declared, not requestable" row naming the
   temporal upsampler and `BetaScheduler`. Anti-vacuous: it also requires exactly
   one such LTX-2.5 row to exist, so renaming the row away fails it. **RED
   before**: both reachable arms found on `FEATURES.md:328`.

One more added by the third review round:

7. **`ltx2 the kMultishot retirement note states the scene evidence correctly`**
   (new, `tests/vllm/models/test_ltx2_pipeline.cpp`). Reads
   `include/vllm/model_executor/models/ltx2_pipeline.h` through a new
   `LTX2_PIPELINE_HEADER_PATH` compile definition, flattens the comment block so a
   reflow cannot make it vacuous, and requires the retired false sentence to be
   absent and the replacement evidence to be present. Anti-vacuous: it first
   requires **exactly one** ``kMultishot` — FABRICATED` note in the file, so
   deleting or renaming the paragraph fails rather than passes.

   The three positive assertions are SCOPED TO THE NOTE, not run over the whole
   flattened header, and the description above used to say otherwise. A file-wide
   `find` could not fail for `ltx-core`, which occurs six times in this header, so
   that assertion survived deleting the clause it exists to hold — proven by
   mutation, which left the case green at 8/8. The shipped case slices the
   paragraph from its marker to the next enumerator heading and asserts
   `scene cuts`, the CLAIM `ships at INFERENCE inside \`ltx-core\`` rather than
   the bare package name, and `system_prompt` inside that slice, with a
   `note.size() > 400` guard so a bad slice fails loudly instead of quietly
   passing. **RED before**: 4 of its 8 assertions
   failed on `b718f580f` — `flat.find("the only \`scene\` hit is PySceneDetect")`
   returned 29407 where npos was required, and `scene cuts` / `system_prompt`
   both returned npos. **Anti-vacuity proven by mutation**: renaming the note's
   marker in a staged copy gives `REQUIRE( 0 == 1 )`; the header was restored and
   verified by sha256.

   What it can and cannot prove is stated in the case's own comment: it holds the
   header's TEXT, so the false sentence cannot return, but it cannot verify the
   upstream claim — no upstream checkout exists in this tree. That verification is
   §1.1's commands with their positive controls, run against the two pins, and it
   is not automatable here. Saying so is better than a test that appears to gate
   it and does not.

Three more added by the fourth review round (F1 #889, F2 #890, F3 #892), each RED
before its fix:

8. **`ltx2 the reachable/marker split matches the source`** (new,
   `tests/vllm/models/test_ltx2_pipeline.cpp`). The gate F1 was missing. It walks
   `src/`, `include/` and `examples/` under a new `VLLM_CPP_SOURCE_ROOT` compile
   definition, counts product callers of each arm's ENTRY FUNCTION — not
   occurrences of the enumerator, which is the question that gave the wrong answer
   — and requires the refusal message to agree, **in both directions**: no caller
   demands `DECLARED, NOT REQUESTABLE`, and a caller appearing later forbids it.
   Two positive controls ride in the same walk, plus a `files.size() > 100` floor,
   because a walk that opened nothing reports every symbol as unreachable.
   **RED before**: the walk printed `Ltx2UpsampleVideoLatent callers =
   ltx2_video.cpp:1515`, `Ltx2SigmaSchedule callers = ltx2_video.cpp:1689`,
   `Ltx2Schedule callers = <none>` — both controls live, the claim zero — and the
   `DECLARED, NOT REQUESTABLE` assertion failed against the shipped message.
9. **`ltx2 video: a SPATIOTEMPORAL upsampler checkpoint is refused as
   SPATIOTEMPORAL, not as temporal`** (new subcase in
   `tests/vllm/multimodal/test_ltx2_video.cpp`). Closes the hole F2's reviewer
   named in their own evidence: no fixture drove a BOTH-FLAGS checkpoint through
   the product path, so a mutation could not separate "unreachable" from
   "untested". It writes a both-flags upsampler with `WriteReducedUpsampler` and
   drives it through `LoadVideoEngine` + `Generate` — the production entry point,
   per `## Nothing lands dead`, not a direct call to `Ltx2LatentUpsample`.
   Asserts the message names `SPATIOTEMPORAL` and is neither the temporal-only
   diagnosis nor a shape complaint. **RED before**: `it is the TEMPORAL x2
   upsampler` and `Supply the spatial upsampler` both matched, and `SPATIOTEMPORAL`
   did not.
10. **`ltx2 the multi-GPU marker note states the CFG evidence correctly`** (new,
    `tests/vllm/models/test_ltx2_pipeline.cpp`) plus a multi-GPU block in the
    ledger case. Same shape as the int8 marker's evidence gate, for the same class
    of defect: the retired sentence must be absent from the header and the shipped
    message must carry `four forms`, `BatchParallelGemmaBuilder` and `gemma.md`.
    Anti-vacuous twice — exactly one `kMultiGpuParallelism` enumerator must exist,
    and the positive checks are SCOPED to the sliced note, because `kInt8ConvRot`'s
    own comment already contains the word `four`. **RED before**: 4 assertions in
    the ledger case (`three forms` present, `four forms` / `BatchParallelGemmaBuilder`
    / `gemma.md` absent) and `flat.find("zero \`cfg\` hits")` returned 32030 where
    npos was required.

    One consequence worth recording, because it cost a build: the header may not
    QUOTE the retired sentence while explaining it. A text gate matches text and
    cannot tell a quotation from a claim, so the corrected note describes the old
    sentence instead of reproducing it, and says why in the comment.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Removing an enumerator breaks an out-of-tree caller | `Ltx2UnportedPipelineFeature` is internal (`include/vllm/model_executor/`), not part of `include/vllm.h`. Grep shows 3 call sites, all in-tree. |
| Refusing `duration_head_path` breaks a working caller | It cannot: no code reads it, so no caller was getting anything from it. A caller who passes it today is being silently ignored, which is the bug. |
| The ledger test's case count changes and reads as a regression | Stated up front in the gate report: 7 → 5 entries in one case, plus 2 new cases. The exit code is the authority. |
| A concurrent LTX row edits the same header | Four live rows named; none owns `ltx2_pipeline.h:575-590`, `ltx2_video.cpp:250-300`, or the ledger test case. `LTX25-TEMPORAL-UPSAMPLER` owns `kTemporalUpsampler`, which this row does not touch. |

## 6. Stop conditions

- Return `NEEDS_DECISION` if `multishot` turns out to exist under vocabulary not
  searched here — the retirement is then wrong and the arm must be re-anchored
  instead.
- Return `NEEDS_DECISION` on a collision with a live LTX row.
- Do not close #611 if the refusal cannot be made to fail RED first.

## 7. Outcome

**Four of the claims in §1 were NOT re-derived, and four rounds of review found
them one at a time.** The sentence that stood here said "every claim in §1 was
re-derived"; it was written before the later rounds, and it was not
true when it was written either. What actually happened:

- Round 1 found the `hadamard` absence in §1.2 false, and the "int8 appears only
  in the trainer" sentence false **in the shipped refusal message**.
- Round 2 found nine reader anchors in §2.1 stale, cited in the file they were
  wrong about.
- Round 3 found `scene` asserted trainer-only **in the shipped header**, the
  `shot` file list presented as exhaustive at five when it is seven, and a
  `quantization_factory.py` range wrong at both ends. Re-deriving the rest of
  §1.1's table in the same pass turned up a fourth: the `multi` prefix was
  recorded as `multi-GPU`/`multi_gpu` **only**, and `multi-prompt` is there twice.
  Nobody had asked about that row; it was wrong because the row had never been
  re-run either.
- Round 4 found the `cfg` absence in §1.3 false and a fourth multigpu form
  omitted, from a grep PATH-FILTERED to the two source trees — the same mechanism
  as the `shot` list in round 3, in a section that had already written the lesson
  down. It also found the reachable/marker split itself wrong about
  `kBetaScheduler`, and the guard at `ltx2_video.cpp` shadowing the one refusal
  the split was still right about.

Every one of the four is the same defect: an absence asserted from our own
vocabulary, with no positive control, and in three cases behind a path filter that
excluded the files carrying the answer. **This row's stated purpose is retiring
instances of [#604](https://github.com/mudler/vllm.cpp/issues/604), and it
shipped four of them** — three of those into files a user or a porter reads. That
self-record is worth more than the claim it replaces, because the claim was the
mechanism: writing "every claim was re-derived" is what made the next reader stop
checking. §1.1 now carries the commands with their positive controls inline, so
the claim is reproducible instead of asserted, and
`test_ltx2_pipeline`'s "the kMultishot retirement note states the scene evidence
correctly" holds the header's text so the false sentence cannot come back.

Results, as they now stand:

- `multishot`: **fabricated, confirmed** — and the confirmation is now stronger
  than the original. 0 hits for the term in either reference against 24 and 4
  control hits in the same commands; the only upstream sense of "shot" is one
  camera take across all seven files; and the shipped prompt-enhancer prompts
  instruct the model to keep a "Single continuous take — no hard cuts"
  (§1.1, §1.1a). Retired.
- `int8-convrot`: **unreachable upstream, and the first wording of that was wrong.**
  Four inference quantization kinds exist and none is int8, and no rotation is
  exposed as one — but int8 is not trainer-only (a dead per-row int8 kernel lives
  in the `ltx-kernels` inference package) and `hadamard` is not 0 hits (a whole
  quantization-coupled Hadamard family is vendored, and wired to nothing). Both
  corrections are in §1.2; neither moves the disposition. Kept, re-anchored.
- CFG parallelism: **the name was wrong, the exclusion is right, and the evidence
  under it was wrong twice.** Not 0 `cfg` hits but 5, against 33 files as the
  control, because the grep was path-filtered past `docs/multigpu/`; and four
  forms, not three, the fourth being `BatchParallelGemmaBuilder`. Upstream's own
  README calls MGPU a latency tool, the distilled recipe runs `SimpleDenoiser`
  with no guidance, and `gemma.md:104` states outright that the distilled pipeline
  "runs without CFG" — which is a reason where the old sentence was an absence
  (§1.3, #892). Renamed and re-anchored.
- `kVideoEngineWiring`: **stale, confirmed.** L7 landed in `cefacd2d0`. Retired.
- The reachable/marker split: **one reachable and four markers, and the row
  shipped it as two and three.** The ledger now says which kind each is, because
  "refused by name" overstated a marker — and the fourth review round found that
  the corrected ledger still overstated one, because §1.6 asked whether an
  enumerator appears in `src/` when the question is whether anything CALLS the
  function holding it. `kBetaScheduler` sits in `Ltx2Schedule`, which nothing
  calls (§1.6, #889). Upstream constructs `BetaScheduler` nowhere either, so the
  repair is the reclassification and not a wiring; `Ltx2Schedule` is disclosed as
  unreached under `## Owed` (#893). The split is now derived from the tree by a
  test rather than maintained by hand in two vectors.
- `duration_head_path` was the **only** unread key of ten (§2.1). Now refused by
  name. The full inventory is the durable half of this row: it means the next
  person asking "which extras are decorative?" reads a table instead of grepping.

What this row deliberately did **not** do: construct a duration head. #611 stays
open for that, with its user-visible half — silent substitution of the recipe
default — closed.

**The review's own findings, recorded because they are the point of the row.**
Across four rounds all but one finding was evidence accuracy, not logic: a false
absence inside the shipped refusal (§1.2), a false absence in this spec's §1.2 —
the one search term that mattered — nine stale reader anchors cited in the file
they were wrong about (§2.1), a public-doc row that re-merged the split the ledger
had just made, a false absence inside the shipped HEADER (§1.1a), a five-item list
offered as exhaustive when it is seven (§1.1), a line range wrong at both ends
(§1.2), and a `cfg` absence plus a missing fourth form in §1.3 and the shipped
header (#892). A row whose subject is #604 committed four instances of #604, and
each one survived at least one review that had already been told to look for
exactly this. That is not irony to note and move past; it is why the anchors are
now derived rather than written, why the int8 absence is stated as UNREACHABLE
with its dead kernel named, why the `hadamard` search this spec got wrong is
spelled out in full rather than quietly corrected, and why every absence in §1.1
now ships with the command that proves it AND the control hits that prove the
command ran.

**The exception is round 4's other two findings, and they are a different
defect.** F1 (#889) and F2 (#890) were not inaccurate evidence — they were
CLASSIFICATION and CONTROL FLOW. `kBetaScheduler` was published as reachable
because §1.6 asked whether an enumerator appears in `src/`, which is a question
whose true answer is the wrong answer. And the one arm that genuinely was
reachable had its refusal shadowed by a guard on `temporal_upsample` that every
both-flags config also satisfies, so between them the ledger's reachable column
was empty in practice while both surfaces published two entries. A record can be
false without a single false citation in it, and this row had been correcting
citations for three rounds while the thing the citations described was wrong.

The transferable lessons, now two:

**A path filter is an absence claim too.** The `shot` enumeration is the case
where this is provable rather than inferred — the pattern `\bshots?\b` was right
and the filter `*.py *.md *.json *.yaml` was wrong, over a tree whose prompt files
are `.txt`, and it returns exactly the five files that were recorded. §1.3's `cfg`
claim is the second provable case: the pattern was right and the filter to the two
SOURCE trees excluded the `docs/` tree that answers it. (The `scene` claim's
original command was never written down, so why *it* missed `ltx-core` is not
recoverable; that it was never reproduced is the point.) A grep that returns
nothing has to be re-run without its narrowing, and with a control, before it is
written down as an absence.

**Reachability is a claim about CALLERS, and `git grep` for the symbol does not
answer it.** `Ltx2Schedule` holds the beta refusal, appears in the header and the
source, is covered by a passing unit test, and is called by nothing. AGENTS.md
`## Nothing lands dead` and [`.agents/reachability.md`](../reachability.md) landed
in `8f49ac3be` naming this exact shape — the test-only driver — while this repair
was in flight. The gate this row now ships derives the split by counting callers
of each arm's entry function, with positive controls in the same walk, and fails
in both directions.

## Owed

Disclosed under AGENTS.md `## Nothing lands dead`, which requires the unreached
thing, its owning row and its issue to be named rather than left for the next
reader:

- **`Ltx2Schedule` / `Ltx2SchedulerKind` are unreached** —
  [#893](https://github.com/mudler/vllm.cpp/issues/893), owned by this row. The
  dispatcher at `src/vllm/model_executor/models/ltx2_pipeline.cpp:191` has zero
  call sites in `src/`, `include/` or `examples/`; the engine calls
  `Ltx2SigmaSchedule` directly and its only driver is its own unit test. This is
  not a staged slice awaiting a wiring wave: upstream selects no scheduler
  anywhere (§1.6), so there is nothing to wire it to, and the open decision is
  whether to retire it the way `kMultishot` was or keep it as a deliberately
  unreached mirror of upstream's `SchedulerProtocol`. Deciding that is a behavior
  change with its own red-first evidence, which is why the F1 repair labelled the
  arm honestly and did not also delete the dispatcher.
- **`kLoraFusion`, `kInt8ConvRot` and `kMultiGpuParallelism` have no call site at
  all** — [#691](https://github.com/mudler/vllm.cpp/issues/691), owned by this
  row. The derived split gates the beta arm because there is a function to count
  callers of; the other three markers have nothing to derive from, so their kind
  is still asserted rather than proven. #691 stays open for exactly that gap.
- **A duration head is still not constructed** —
  [#611](https://github.com/mudler/vllm.cpp/issues/611), owned by this row. Its
  user-visible half, silent substitution of the recipe default, is closed.
- **`.agents/specs/ltx-2-5.md` §2 "Out" still lists `multishot`** —
  [#692](https://github.com/mudler/vllm.cpp/issues/692). Operator-owned file.

## Now

Row `LTX25-RETIRE-DEAD-ARMS` is `DONE`. The ledger carries five entries split by
kind — **one reachable, four markers** — and the split is derived from the tree by
`test_ltx2_pipeline`'s "the reachable/marker split matches the source" rather than
maintained by hand. The one reachable arm is no longer shadowed: a both-flags
upsampler checkpoint driven through `LoadVideoEngine` is refused as
SPATIOTEMPORAL. `duration_head_path` is refused by name with a RED-first test; the
`kKnownLoadExtras` inventory is recorded in §2.1; the header's `kMultishot` and
`kMultiGpuParallelism` notes are both gated by `test_ltx2_pipeline` against the
false absences they shipped (§1.1a, §1.3). Owed items are listed above.
`.agents/specs/ltx-2-5.md` §2 "Out" still lists `multishot` and is the operator's
to correct, citing §1.1 here.
