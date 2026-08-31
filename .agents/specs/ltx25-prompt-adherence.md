# SPEC — `LTX25-PROMPT-ADHERENCE`: what would score "a good render OF THIS PROMPT"

Issue: [#2295](https://github.com/mudler/vllm.cpp/issues/2295), which owns
[#1854](https://github.com/mudler/vllm.cpp/issues/1854)'s first sub-question.
Owner row: `LTX25-PROMPT-ADHERENCE`. Until this spec landed, that sub-question was
owed by `LTX25-ORACLE-ABSOLUTE` under `## Owed` in
[`ltx25-oracle-absolute.md`](ltx25-oracle-absolute.md).

**W0 landed a spec and a record and no code, and ended in a decision the
developer takes.** §9 states that decision and it is now ANSWERED: the scoring
model is an INSTRUMENT. W1 and W2 followed on that answer and are in this row's
second change -- the measurement, the gate, and its red-before suite. `## Outcome`
carries what was measured. #1854's first sub-question is answered for the request
the reference render was taken at, and `## Owed` states, in the same breath, the
one it cannot answer.

## Scope

#1854 filed two sub-questions and refused to close either with a proxy. Its
second, artefact freedom, is gated as of `fa9903b86`: `--reference` holds our
blockiness against a bound recomputed from the #1864 oracle render's own frames.
Its FIRST is untouched, and `scripts/ltx25-render-compare.py:1481-1482` still
prints, in the tool's own output, that prompt adherence "is not measured
anywhere in this tree".

IN scope:

- Establishing what the pinned upstreams do about prompt adherence, by reading
  them at their revisions rather than by recall.
- Naming the candidate scorers, each with what it scores, its size, its licence,
  whether a revision is obtainable, and whether anything on this fleet can run it.
- Deriving the gate SHAPE, and arguing why it is a comparison rather than a
  convention, to the standard `fa9903b86` had to meet.
- Stating the gateability of each candidate honestly, on the side of the line
  AGENTS.md draws: an oracle is gateable only once it demonstrably runs.
- The `NEEDS_DECISION` of §9, and the two readings between which it chooses.

OUT of scope, each declared rather than approximated:

- **Any scoring implementation.** No flag, no checker, no statistic. The
  decision of §9 gates whether one may be written at all.
- **Any hand-rolled statistic.** #1854 names the failure by name: a proxy "that
  correlates with nothing would be the
  `a-shape-valid-gate-passes-a-wrong-artefact` failure". Nothing here proposes
  one, and §4 rejects the two that suggest themselves.
- **Any change to AGENTS.md's oracle table.** §9 is why.
- **Any download.** §5 prices the artefacts. None was fetched.
- **The four statistics of the absolute panel.** `fa9903b86` gates two and
  reports three; not one value moves.

## 1. What upstream does, read at the pins

### vLLM at `5559679229` evaluates no image or video generation, and DOES ship the scorer

vLLM is the primary reference wherever it implements the behaviour, so it was
searched first. It evaluates text generation with `lm-eval-harness` and it
implements no adherence evaluation and no image or video generation evaluation
at all. Searched at that revision with
`git grep -iIl` for `clip_score`, `CLIPScore`, `ImageReward`, `PickScore`,
`HPSv2`, `VQAScore`, `aesthetic`, `prompt adherence` and `T2I-CompBench`: **zero
files each**. A word-boundary `FID` search matches only vendored minified
JavaScript and documentation assets. That is a stated set of searches with their
results, not an assertion of absence from one failed grep.

**What it does ship is the scoring MODEL, as a first-class registered runner.**

| Anchor | Value |
|---|---|
| `vllm/model_executor/models/registry.py:251` | `"CLIPModel": ("clip", "CLIPEmbeddingModel")` |
| `vllm/model_executor/models/clip.py:774` | `class CLIPEmbeddingModel(nn.Module, SupportsMultiModal, SupportsQuant)`, `is_pooling_model = True` at `:775` |
| `clip.py:808`, `:820` | `self.text_projection`, `self.visual_projection` — the joint-space projections |
| `clip.py:847`, `:867` | `text_features = self.text_projection(pooled_output)`, `image_features = self.visual_projection(pooled_output)` |
| `tests/models/multimodal/pooling/test_clip.py` | vLLM's own gate, `openai/clip-vit-base-patch32`, `runner="pooling"`, `max_model_len=77` |

The anchor is unique: the literal `CLIPModel` occurs exactly **once** in
`registry.py` at that revision, at line 251. `test_clip.py` compares
`vllm_model.embed(...)` against HuggingFace `CLIPModel.get_image_features` and
`get_text_features` through `check_embeddings_close`, which fixes what vLLM's
`.embed()` returns: the PROJECTED features, in the joint space a CLIP score is a
cosine of. Nothing further has to be built to get the two vectors.

**The consequence is the whole of §9's first reading.** A CLIP-based instrument
runs on vLLM, which is `role = primary`, pinned in
[`../upstream-sync.md`](../upstream-sync.md) and `gateable = yes`. It needs no
new row in AGENTS.md's table. #1854 anticipated a scorer "pinned like any other
oracle"; the scorer turns out to already be inside one.

### vLLM-Omni scores adherence with CLIP, and for VIDEO

Read at `a4ea67a21b20054dacc6e83952f9bd407e8ee4e7`. **Its oracle record is
`pin = UNPINNED`, `gateable = no` ([`../oracles/vllm-omni.md`](../oracles/vllm-omni.md),
[#633](https://github.com/mudler/vllm.cpp/issues/633)), so every line below is a
source reading at a stated revision and none of it is a pinned citation.** That
distinction is the point of the pin rule and is why the shape is taken from here
while the RUNTIME is taken from vLLM.

- `tests/e2e/accuracy/helpers.py:496`, `score()` at `:504` — `class CLIPScorer`, default
  `openai/clip-vit-base-patch16`. `score(image, text)` normalizes both
  projected embeddings and returns their cosine times 100.
- `tests/e2e/accuracy/test_videogen_modelopt_quant.py` — the video case. Its
  docstring states the design: "scores prompt-faithfulness with CLIP on the
  middle frame, then asserts both `CLIP >= absolute_floor` and
  `CLIP_drop <= drop_threshold`". `:238-240` scores the BF16 and FP8 renders of the
  same seed and `:268-272` asserts both halves. `CLIP_ABSOLUTE_FLOOR` is `20.0`
  and `CLIP_DROP_FP8` is `7.0`, both env-overridable (`:54-55`).
- `tests/e2e/accuracy/test_hunyuan_image3.py:621`, `:654`, `:658` — the same
  two-part shape for images.

Its own video prompt (`:50`) is
`"A red fox running through a snowy forest at sunrise, cinematic."` The #1864
reference render's prompt (`tools/oracle/ltx2_oracle.py:85`) is
`"A red fox walks slowly through a snowy pine forest at sunrise, cinematic."`
The two are near-identical, which is convenient and is not evidence of anything.

### For LTX specifically, upstream gates SIMILARITY and not adherence

`tests/e2e/accuracy/ltx/test_ltx_official_similarity.py` at the same revision is
vLLM-Omni's LTX accuracy guard, and it is a pixel comparison against the
Lightricks runtime: `VIDEO_SSIM_MEAN_THRESHOLD = 0.95`,
`VIDEO_SSIM_MIN_THRESHOLD = 0.90`, `VIDEO_PSNR_MEAN_THRESHOLD = 30.0`,
`AUDIO_RELATIVE_L2_THRESHOLD = 0.2`, `AUDIO_COSINE_THRESHOLD = 0.95` (`:56-60`),
asserted at `:436-440`. No CLIP, no VLM. **So the CLIP precedent is upstream's,
but it is not upstream's LTX precedent**, and that is recorded rather than
smoothed over.

### Lightricks `LTX-2` at `fd4ded7f` evaluates nothing

Searched for `clip_score`, `CLIPScore`, `ImageReward`, `PickScore`, `HPSv2` and
`VQAScore`: **zero files each**. The four `adherence` matches are all
CFG-scale documentation — `packages/ltx-core/README.md:46`,
`components/guiders.py:18`, `packages/ltx-pipelines/docs/multimodal-guidance.md:3,40,50`
and `utils/args.py:954,1014` — and describe a KNOB that increases adherence, not
a measurement of it. Its `eval` matches are the `.claude/skills/train-model/`
LoRA-training workflow, whose "post-train eval" renders samples for a human to
look at (`phases/post-train-validate.md:105-126`) and computes no metric.
**The model's author ships no way to score its own output.**

## 2. What a prompt-adherence oracle would have to be

Three properties, each forced by something already in this tree:

- **It maps (frames, prompt) to a number that is comparable ACROSS TWO
  DIFFERENT PICTURES.** Our render and the reference are not the same picture:
  same prompt, same seed integer, different engine, different sampler draw.
  `ltx25-oracle-absolute.md` §5 already excluded `sharpness_mean` for exactly
  this, because its value is set by content. An adherence score must be
  content-driven and still comparable, which is only true if the thing it is
  driven by is the PROMPT both renders share.
- **Its weights are part of its identity.** SSIM is a closed-form function; a
  neural scorer is a file. A swapped checkpoint moves every reading silently,
  which is [#1723](https://github.com/mudler/vllm.cpp/issues/1723) exactly. So
  the scorer needs a revision AND a sha256 asserted before it reads a pixel,
  the way `--reference` asserts the render's digest today.
- **Its own competence must be checkable on the reference.** A scorer that
  cannot tell the reference render's frames from a decoy prompt is a broken
  instrument, and a broken instrument that keeps running reports a CODE verdict.
  `reference_bounds` already refuses a degenerate reference at
  `EXIT_UNREADABLE`; the scorer needs the same precondition, checked on the
  ORACLE's frames, where the answer is known.

## 3. The candidates

| Candidate | Scores | Size, revision | Licence | Runnable here | Gateable today |
|---|---|---|---|---|---|
| `openai/clip-vit-base-patch16` | text-image cosine in the CLIP joint space | `pytorch_model.bin` **598,641,023 B**, rev `57c216476eefef5ab752ec549e440a49ae4ae5f3` | **NONE DECLARED** | yes, CPU; no lease needed | **no** — never run in this tree |
| `openai/clip-vit-base-patch32` | same | `pytorch_model.bin` **605,247,071 B**, rev `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | **NONE DECLARED** | yes, CPU | **no**, but it is the checkpoint vLLM's OWN gate exercises |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | same, larger tower | `model.safetensors` **3,944,552,236 B**, rev `1c2b8495b28150b8a4922ee1c8edee224c284c0c` | `mit` | `dgx:gpu0` comfortably; whether `CLIPEmbeddingModel` accepts it is UNVERIFIED | **no**, and it is not upstream's choice |
| VQAScore, `zhiqiulin/clip-flant5-xxl` | a generative VQA likelihood, better on compositional prompts | ~11 B parameters, unsized here | `apache-2.0` | `dgx:gpu0` only; not `thor`/`orin` | **no**, and vLLM registers no such runner |
| A general VLM as judge | free-form; a rating | 4 B upward | varies | `dgx:gpu0` | **no**, and see §4 |

Sizes and revisions are the `x-linked-size`, `x-linked-etag` and `sha` the
HuggingFace API returned to an unauthenticated request on 2026-08-29. **They are
the repository's CLAIM about those files. Nothing here has been downloaded and
nothing has been hashed**, which is the same distinction
[`../oracles/ltx-2.md`](../oracles/ltx-2.md) draws between the advertised and the
measured digest of the Gemma tower.

**The recommendation is `openai/clip-vit-base-patch16`,** because it is the
checkpoint the pinned diffusion oracle scores adherence with, and because
`CLIPEmbeddingModel` runs it. `patch32` is the fallback with a stated reason:
it is what vLLM's own `test_clip.py` exercises, so if `patch16` fails to load
the runnability evidence already exists one patch size away, and the swap must
then be recorded because it changes the reading.

**Three costs a reader should not rediscover:**

1. **Neither `openai/clip-*` repo declares a licence.** `cardData.license` is
   null for both. The `openai/CLIP` GitHub repository is MIT; the WEIGHTS
   repository asserts nothing, and a repository that asserts nothing has not
   granted anything. `laion/CLIP-ViT-H-14` does declare `mit`. This is a
   developer question, not an agent's to resolve.
2. **Neither ships safetensors.** The only weight file in each is a pickle
   `pytorch_model.bin`.
3. **CLIP's text context is 77 positions** (`config.json`,
   `text_config.max_position_embeddings = 77`), so about 75 usable tokens after
   the boundary tokens. The #1864 reference prompt is **13 words / 73
   characters** and fits. **#1854's own motivating example does not**: the
   golden-retriever prompt at `scripts/ltx25-dit-attn-flash-pixel-ab.sh:697` is
   **70 words / 413 characters** and would be truncated. A CLIP scorer answers
   the request the reference render was taken at, and cannot answer the one
   #1854 quotes. Its vision tower is also 224x224, so a 320x192 frame reaches it
   as a resized centre crop and off-centre content is discarded.

Point 3 is the sharpest limit here and it is stated up front rather than
discovered later, because a gate that silently truncates its own question is
worse than no gate.

## 4. The gate shape, and why it is a comparison

#1854 admits exactly one shape: "worse than the oracle on this statistic,
because that is a comparison and not a convention". Upstream's pair is half
admissible and half not, and the split is worth being explicit about.

- `clip >= 20.0` — **INADMISSIBLE.** `20.0` is env-overridable and has no
  derivation anywhere in vLLM-Omni. Importing it would put a chosen constant at
  the centre of the verdict, which is what #1854 exists to prevent.
- `bf16_clip - quant_clip <= 7.0` — **the right axis, still with a chosen
  tolerance.** The numerator is a comparison against a reference render; the
  `7.0` is not derived either.

### The proposed form, S1: ours-mean against the reference's own per-frame minimum

Score **every** frame of the #1864 reference against the prompt, and hold our
render's mean against the reference's per-frame minimum:

    ours_mean_clip  >=  ref_frame_min_clip        recomputed on every run

This is the exact mirror of the bound `fa9903b86` landed. There, higher is worse
and the form is `ours_mean <= ref_frame_max`. Here higher is better, so the
inequality and the order statistic both flip. Every digit is measured off the
reference render; **no constant is chosen and nothing is transcribed.**

The looseness is stateable in the same terms, and must be measured rather than
assumed: the reference's per-frame minimum sits some number of standard
deviations below its own mean, and the standard error of our mean over 25
frames is one fifth of the per-frame standard deviation, so the check fires
when our adherence falls below the reference's by roughly that many of the
reference's own per-frame standard deviations. **The multiplier is UNMEASURED** — it needs the reference's
per-frame CLIP distribution, which W1 produces — and it is not guessed here. On
blockiness the corresponding figure was about two standard deviations, and there
is no reason to expect the same number for a different statistic.

Two rejected alternatives, recorded because they are what a reader proposes
first:

- **`ours >= ref` on the means.** No margin at all. Two renders of one prompt
  differ by sampler noise, so this fires on the difference in content.
- **`ours_max >= ref_max`, or min against min.** Two single order statistics,
  25 frames each side: with no real difference the verdict is a coin toss. This
  is the same argument `ltx25-oracle-absolute.md` §4 makes against max-against-max,
  and it did not stop being true for a different statistic.

### The complementary form, S2: discrimination against committed decoys

S1 asks "is our number as big as theirs". S2 asks the question #1854 actually
poses. Score our render against the true prompt and against a committed, fixed
set of decoy prompts, and require the true prompt to rank **first**:

    argmax over {true, decoy_1 .. decoy_N} == true, for our render AND for the reference

This is a SET assertion with a printed margin and not a tolerance, which is what
a discrete selection gate needs: its error is bimodal, so a tolerance bounds
nothing. Its null is derivable and needs no
convention: with N decoys, chance is 1/(N+1).

**Why S2 is legitimate where an absolute CLIP threshold is not.** CLIP's
absolute cosine is uncalibrated — that is why `20.0` means nothing — but
contrastive RANKING is the objective the model was trained on. Using it as a
ranker uses it where its correlation is established, and using its raw value as
a quality score uses it where it is not. That argument is the whole difference,
and it is why S2 is proposed rather than a second threshold.

**S2 also covers the blind spot the landed gate declares.**
`ltx25-oracle-absolute.md` records that a pure-noise render PASSES blockiness
and passes C0, with a test that says so. Noise fails S2.

The decoy set must be committed, fixed, and contain both a NEAR decoy (a fox in
a summer forest) and a FAR one (a city street at night), and the margin between
the true prompt and the best decoy must be printed on every run, passing or
failing. A gate that prints only its verdict cannot be seen to be degrading.

### The precondition, S0

Before either check publishes a number, the scorer must rank the true prompt
first **on the REFERENCE's frames**. If it cannot do that on upstream's own
good render of this prompt, the instrument is broken and the run exits
`EXIT_UNREADABLE` rather than failing our render. This mirrors
`reference_bounds` refusing a degenerate reference, and it is the guard against
`broken-instruments-fail-toward-a-code-verdict`.

## 5. Gateability, stated on the side of the line each candidate is on

AGENTS.md: an oracle is gateable "only once it demonstrably builds and runs the
model. Constructing a config proves nothing."

- **The vLLM oracle is `gateable = yes`** and that is a property of the
  repository, established by other models. **Its CLIP path is UNMEASURED.**
  Nothing in this tree has loaded `CLIPEmbeddingModel`, produced an embedding
  from it, or compared one against HuggingFace. Until that happens, no
  adherence number from this route may be quoted, and the row records the debt
  rather than the capability.
- **Every candidate checkpoint in §3 is at `gateable = no` for this use**, and
  for the same reason: none has been downloaded and none has been run.
- **The measurement is cheap and needs no lease and no GPU**, which is the one
  piece of good news here. It needs the already-committed
  `tests/parity/goldens/ltx2_oracle/upstream-render.mp4`, `ffmpeg`, and the
  598 MB checkpoint. Decoding 25 frames and scoring them against one prompt and
  N decoys is seconds of CPU. **That is W1, and it produces the reference's own
  per-frame CLIP distribution — the thing S1's bound IS.**
- **Our side of the comparison EXISTS, and its frames do not.**
  [#2140](https://github.com/mudler/vllm.cpp/issues/2140) is CLOSED, and `rc`
  job `4b0666ee-248c-45fc-9de6-372b6d0c1fab` on `dgx:gpu0` rendered the
  manifest's exact request: 25 of 25 distinct frames, `steps_observed={8}`, and
  `PASS` / `NO_WORSE_THAN_ORACLE_ON_BLOCKINESS`
  (`ltx25-oracle-absolute.md` §Outcome, W3 fourth attempt). What that leaves is
  narrower than a blocker and is stated rather than assumed: **the run's own
  frames are not committed and this spec does not establish that they were
  retained.** Whether W3 needs a fresh lease or only a fetch is UNMEASURED and
  is the first thing W3 should check. **W1 does not depend on either**, because
  it reads the committed reference mp4 and nothing of ours, and a bound measured
  before our frames are scored cannot have been tuned to them.

## 6. Port map

W0, the spec and the record:

| File | Change |
|---|---|
| `.agents/specs/ltx25-prompt-adherence.md` | this file |
| `.agents/issue-index.md` | one appended row for #2295 |
| `.agents/specs/ltx25-oracle-absolute.md` | its `## Owed` bullet for #1854 sub-question 1 names this row as the owner |

No product code, no script, no test, no oracle record, and no change to
AGENTS.md. §9 is why that list stopped there.

W1 and W2, the measurement and the gate, after §9 was answered:

| File | Change |
|---|---|
| `scripts/ltx25-render-compare.py` | `--adherence-model`, and the S0/S1/S2 block behind it. `--reference` is required with it, and the module docstring states the 77-position bound in both states |
| `tests/parity/goldens/ltx25_adherence/scorer-pin.json` | new; the checkpoint's revision, its eight measured sha256 digests, its three costs, and `gateable` |
| `tests/parity/goldens/ltx25_adherence/decoys.json` | new; the six committed decoys, three near and three far, and the derivation of the null |
| `tests/scripts/test_ltx25_prompt_adherence.py` | new; the red-before suite, 47 cases |
| `tests/scripts/test_ltx25_absolute_reference.py` | one assertion, because this change made its declaration line false |
| `scripts/agent-preflight.sh`, `.github/workflows/ci.yml` | the new suite runs on both lanes, registered in the change that adds it |
| `docs/USAGE.md` | the command and the checkpoint it is gated against |

**No `.agents/oracles/` file and no change to AGENTS.md's oracle table.** §9 is
still why.

## 7. Tests to port

**W0 owed none**: there is no upstream test for a decision, and it implemented no
behaviour to test. W2 owes five, and all five are in
`tests/scripts/test_ltx25_prompt_adherence.py`. Each was RED on `origin/main`
before the change: 47 cases, 3 failures and 34 errors, run against `main`'s copy
of the tool with only the new goldens and the new suite added.

| ID | Assertion | State |
|---|---|---|
| P1 | The scorer's checkpoint is refused unless its sha256 matches the pinned digest | LANDED, six cases: an UNMEASURED (`null`) digest, a wrong digest, a missing file, an empty pin, the committed pin against a directory that is not the checkpoint, and the matching case |
| P2 | S0: a scorer that ranks a decoy first on the REFERENCE exits `EXIT_UNREADABLE` and publishes no bound | LANDED, and the sharper case beside it: a CONSTANT scorer, which HAS an argmax (numpy returns the first maximal index, which is the true prompt) and would clear a bare set assertion while measuring nothing. The margin must be strictly positive |
| P3 | The S1 bound equals the reference's own recomputed per-frame minimum, not a literal | LANDED as exactly that case: ONE render, TWO references, OPPOSITE verdicts, plus the at-the-bound pair |
| P4 | S2 on a pure-noise render FAILS, beside the blockiness case that PASSES it | LANDED in the suite, and MEASURED end to end on the real checkpoint; see `## Outcome` |
| P5 | A prompt longer than CLIP's 77 positions is REFUSED rather than silently truncated | LANDED, and the boundary is checked at the limit and one past it |

**The upstream test is ported in the same change**, in
`PinnedCheckpoint.test_ported_vllm_test_clip_the_two_feature_routes_agree`, from
`vllm/tests/models/multimodal/pooling/test_clip.py` at `5559679229`. It preserves
upstream's tolerance (`check_embeddings_close`, `tol = 1e-3`, cosine), upstream's
dtype (`"float"`, `:75`), upstream's context (`max_model_len=77`, `:41`) and
upstream's reference entry points (`get_image_features(pixel_values=...)` and
`get_text_features(input_ids=..., attention_mask=...)`, `:52-57`). **One harness
adaptation, and it is unavoidable**: upstream's other side is `VllmRunner`, which
needs an installed vLLM and a GPU lease, and this tree's vLLM is a source
checkout. So the second side here is the route vLLM-Omni's `CLIPScorer` takes
(`helpers.py:507-508`, `outputs.image_embeds` and `.text_embeds`), and the case
requires it to agree with vLLM's reference route at upstream's own tolerance.
That is not a tautology: they are different code paths in `transformers`, and the
scorer deliberately takes vLLM's, because vLLM is the primary reference. The
`VllmRunner` half stays OWED.

`transformers` 5.x returns a `BaseModelOutputWithPooling` from those two calls
with its `pooler_output` already REPLACED by the projected features, which is
`clip.py:867`'s `self.visual_projection(pooled_output)`. The scorer ports
upstream's own unwrap for it (`test_clip.py:59-61`) rather than assuming a
tensor, so it reads the PROJECTED feature and not the pre-projection pooled one.

## 8. Gates

W0 was a records change and its gates were the record gates. W2 adds executable
ones, and they run on both lanes in the change that adds them:

1. `python3 tests/scripts/test_ltx25_prompt_adherence.py` — the new suite. 47
   cases; 42 need numpy only and no checkpoint, and the 5 that need the pinned
   weights SKIP LOUDLY on `VT_LTX25_ADHERENCE_MODEL` being unset. A skip is never
   an `ok`. Registered in `scripts/agent-preflight.sh` and in `ci.yml`'s
   numpy lane beside the two suites that already exercise this tool.
2. `python3 tests/scripts/test_ltx25_absolute_reference.py` — unchanged in every
   value it checks. One assertion moved, because this change made the tool's
   "prompt adherence is not measured anywhere in this tree" line false; the
   replacement holds the same obligation, that a run with no scorer says it
   measured no adherence and carries no adherence check into the table.
3. `python3 tests/scripts/test_ltx25_render_compare.py` — the older suite over
   the same tool, unchanged and green.
4. `scripts/agent-preflight.sh` — the record gates.

There is no C++ in this change, so there is nothing to compile. That is stated
rather than left implicit, because a record gate builds nothing and a green one
has already let a tree that did not compile onto `main`.

## 9. `NEEDS_DECISION` — ANSWERED

**Is a scoring model an ORACLE or an INSTRUMENT?**

**ANSWERED: INSTRUMENT** (developer, 2026-08-31, recorded in
`.agents/developer-preferences.md`), on this row's own recommendation and
argument. Reading A below. The consequences, all of them discharged in this
change:

- `ltx-2` remains the oracle. **AGENTS.md's oracle table is unchanged** and there
  is no `.agents/oracles/` file for the scorer. A case in the suite asserts the
  second half of that, so a later edit that promotes the scorer in place is red.
- The checkpoint is pinned by **revision AND sha256**, like any other artifact
  this project loads: `tests/parity/goldens/ltx25_adherence/scorer-pin.json`.
- This is an ordinary row. W1 and W2 ran.
- **Reversible.** If it turns out to need oracle status, that returns as a
  decision rather than being promoted in place.

The two readings are kept below, because the argument is what makes the answer
checkable and a later reader needs the one that was rejected.

**Reading A — instrument.** The scorer is a measuring device, like
`torchmetrics`' SSIM in #1743's coherence checks or `ffmpeg` in the reference
decode. The ORACLE remains `ltx-2`, whose render is the other side of every
comparison, running on `vllm`, which supplies the runner. Nothing is added to
AGENTS.md's table. The checkpoint is pinned by revision and sha256 beside every
other weight this project loads.

**Reading B — oracle.** The scorer produces the verdict, so it is the authority
the gate defers to. It would then need its own `.agents/oracles/<id>.md`, its own
`gateable` measurement, and a row in AGENTS.md's table with a `Reach for it when`
line — a policy change under AGENTS.md §"Changing the rules or a checker", with
its own spec and a red-before test of the checker semantics.

**The argument for A is that the scorer never answers a question on its own.**
Every number it produces is consumed as a comparison against the #1864 render,
which is already the pinned oracle's output. S1 holds our mean against the
reference's own frames; S0 requires the reference to pass the same discrimination
our render must pass, and refuses to publish anything if it does not. Delete the
reference and neither check has a bound. That is the test for what the oracle IS,
and the answer is `ltx-2`, unchanged. What the CLIP checkpoint contributes is a
metric space, and a metric space with pinned bytes is an instrument.

Two things made the recommendation less than obvious, and they are why it was the
developer's call. A neural instrument has weights, so it is not a closed form the
way SSIM is, and AGENTS.md's gateability rule was written for exactly that
difference — which is why `gateable` is a MEASURED field in the pin and not an
assertion. And S2's verdict does not reference the oracle's frames in its argmax;
only S0's precondition does, so under S2 the scorer comes closer to speaking on
its own than under S1.

The three smaller decisions that rode along, and their answers:

- **The unlicensed weights of §3.** ACCEPTED (developer, 2026-08-31), with the
  fact recorded beside the pin rather than smoothed over: `openai/clip-vit-*`
  declares no licence, and the pin's `licence` field is `null` rather than a
  guess. A case asserts it stays `null`.
- **The 598 MB download.** AUTHORISED (developer, 2026-08-31), priced first: 598
  MB, no declared licence, a pickle rather than safetensors. All three are in the
  pin.
- **Whether S2 ships at all.** It ships. It is the only one of the two that
  answers #1854's question as #1854 phrases it, and `## Outcome` records that it
  is also the only one of the two that catches the noise render.

## 10. Dependencies

- The #1864 reference render and its `SHA256SUMS`. Landed and committed.
- `ffmpeg`, already required by `ltx2-gen` and by `--reference`. RESOLVED.
- Download authority for one 598 MB checkpoint. **GRANTED**, 2026-08-31.
- `transformers` and `torch` on the host that runs the scorer. Present on the
  devbox; the four cases that need them skip loudly elsewhere. No GPU and no
  lease: the whole measurement is seconds of CPU.
- Our render's frames. **STILL THE ONE OPEN DEPENDENCY.** The render itself is
  DONE ([#2140](https://github.com/mudler/vllm.cpp/issues/2140) is closed and
  `rc` job `4b0666ee-248c-45fc-9de6-372b6d0c1fab` produced it), and whether its
  25 frames survived the lease is UNMEASURED. W1 and W2 needed neither. W3 does.
- The decision of §9. **ANSWERED.**

## 11. Work breakdown

- **W0** — the spec and the record. No code. COMPLETE.
- **W1** — the instrument measurement. COMPLETE, and `## Outcome` is its result:
  the checkpoint downloaded and hashed, the scorer loaded, the committed
  reference mp4 decoded and scored, and the reference's own per-frame CLIP
  distribution recorded. CPU, no lease, no GPU.
- **W2** — the tool change and its red-before suite, S0/S1/S2 and P1..P5.
  COMPLETE.
- **W3** — scoring OUR render at the reference's request, and the verdict.
  **NOT STARTED, and it is what `## Owed` names.** Its first step is to establish
  whether the frames of `rc` job `4b0666ee-248c-45fc-9de6-372b6d0c1fab` were
  retained, because a fetch and a fresh lease are not the same cost.

## 12. Risks and decisions

- **CLIP is a weak instrument and this spec does not pretend otherwise.** It
  reads a 224x224 centre crop, it truncates at 77 positions, and its absolute
  value is uncalibrated. What it is good at is ranking, which is what S2 uses it
  for. S1 uses its value, and S1's defence is only that both sides of the
  comparison are scored by the same model on the same prompt.
- **The 77-token limit means this cannot gate the prompt #1854 quotes.** A
  70-word prompt needs a scorer with a longer text context, and none of the
  candidates in §3 that has one is upstream's choice or has ever run here.
- **The landed blockiness reading gives this row a second motivation, measured
  rather than assumed.** `ltx25-oracle-absolute.md` §Outcome records that our
  render is less blocky, less sharp (10.5176 against a reference per-frame
  minimum of 10.8391) and less clipped (0.000758 against 0.001226) than the
  reference, calls that one coherent picture — "our render is somewhat SMOOTHER
  than upstream's" — and states that a one-sided blockiness ceiling is blind to
  smoothness by construction. Whether a smoother render still depicts the prompt
  is exactly the question this row is about, and nothing in the tree can answer
  it.
- **Scoring frames independently is not scoring a VIDEO.** Every candidate is an
  image-text model. Temporal adherence — "walks SLOWLY" — is invisible to all of
  them. vLLM-Omni scores one middle frame and claims no more; this spec scores
  all 25 and claims no more either. A video-native scorer would be a different
  oracle question and is not proposed.
- **A future re-render of the #1864 reference moves S1's bound.** That is
  correct behaviour, and it is why the bound is recomputed rather than recorded.
  `test_ltx2_oracle_goldens.py` already reds on a silently swapped reference.
- **The recommendation of §9 could be wrong** in the direction that matters
  most: if the developer reads a scorer as an oracle and W1 has already run, W1
  is still not wasted, because a measurement of the reference render is not a
  gate and adds nothing to any verdict.

## 13. Evidence

- §1's four upstream readings, each at a stated revision, with the search terms
  and their counts rather than a claim of absence.
- The uniqueness of `registry.py:251`: one occurrence of `CLIPModel` at
  `5559679229`.
- §3's sizes, revisions and licence fields, from the HuggingFace API on
  2026-08-29, recorded as the repository's claim and not as a measurement.
- §3's prompt lengths: 13 words / 73 characters for the reference request, 70
  words / 413 characters for `ltx25-dit-attn-flash-pixel-ab.sh:697`, against
  `text_config.max_position_embeddings = 77`.
- W1 ADDED, and `## Outcome` carries them: the checkpoint's eight measured
  sha256 digests, the reference's own per-frame CLIP distribution, the S0
  verdict on the reference and the S0 refusal on a noise render, and one
  end-to-end run in which the gate fires.
- §3's prompt-length reading is now MEASURED against the scorer's own tokenizer
  rather than argued: the reference request is 17 CLIP tokens, every committed
  decoy is 20 or fewer, and #1854's 70-word example needs more than 77. The
  vocabulary-free lower bound in the suite reads 83 for it and agrees.

## 14. Stop conditions

Stop and report, do not work around:

- §9 unanswered, for anything past W1;
- a scorer that fails S0 on the reference — that is a dead candidate and a
  finding, not a thing to loosen;
- a prompt that does not fit the scorer's context — refuse, never truncate;
- any state in which the only way to produce an adherence number is to choose a
  constant. #1854 filed this gap rather than proxy it, and a proxy landed under
  this row would be the same failure with a spec attached.

## Outcome

What was measured, on 2026-08-31, on the devbox: no GPU, no lease, CPU only, and
the whole run is seconds once the checkpoint is on disk.

### The instrument is GATEABLE, and that is measured rather than asserted

`openai/clip-vit-base-patch16` at revision
`57c216476eefef5ab752ec549e440a49ae4ae5f3` was fetched and hashed file by file.
Every measured size equals the size the HuggingFace API advertises, and the eight
digests are in `scorer-pin.json`. `pytorch_model.bin` is 598,641,023 bytes and
its sha256 is `ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0`.
`transformers` then loaded the model and the processor from the local directory
and produced embeddings. So `gateable = true` in the pin, and it says what was
run rather than that a config was constructed.

**The 77 positions are measured too.** `config.json` at that revision, sha256
`eaf1c9089a8553c913d27ea66407f8bfc2be9989c80c9f331ddb3d63d4c5e8ad`, has
`text_config.max_position_embeddings = 77` and `vision_config.image_size = 224`.

### S0 PASSED on the reference, and the same scorer says NO on noise

Scored the 25 frames of the committed `upstream-render.mp4` against the prompt
the manifest records and the six committed decoys:

| Prompt | mean | per-frame |
|---|---|---|
| **true** | **38.1278** | [36.0087, 39.8198] |
| `near:1` grey wolf, same forest | 36.3039 | [35.0705, 37.6487] |
| `near:2` fox asleep at night | 31.9409 | [29.8596, 33.6220] |
| `near:0` fox, summer forest | 31.9369 | [30.2435, 33.6573] |
| `far:4` bowl of soup | 20.2937 | [19.6407, 21.2946] |
| `far:5` empty beach | 18.9246 | [18.3531, 19.5103] |
| `far:3` city street at night | 16.2770 | [15.6993, 16.8785] |

S0 PASSED: the true prompt ranks first, margin **+1.8240** to the best decoy,
per-frame wins **25 of 25**, against a null of 1/7 = 0.1429. The near decoys are
genuinely near — the grey wolf sits 1.82 below the true prompt, which is what
makes the set discriminating rather than decorative.

**The instrument then failed on something it should fail on**, which is the whole
of S0's claim. The same scorer on 25 frames of uniform pseudo-random noise ranks
the true prompt **LAST of seven**, margin **−5.6586**, per-frame wins **0 of 25**,
and `scorer_precondition` refuses at `EXIT_UNREADABLE`. An instrument that has
never said no is not known to be able to; this one has.

### S1's bound IS derivable, and here is the number

    ours_mean_clip  >=  36.0087

recomputed on every run from the reference's own 25 frames. The reference's mean
is 38.1278 with a per-frame sd of 0.9518, so the bound sits **2.23 of the
reference's own per-frame standard deviations** below its mean. §4 left that
multiplier UNMEASURED and refused to guess it; this is the measurement. It is
close to the roughly two standard deviations the blockiness bound worked out at,
and that is a coincidence of two different statistics on one render rather than a
property of the construction — it is recorded because §4 said it would be, not
because it means anything.

**No statistic in this row stays REPORTED for want of a derivation.** Both checks
gate. The four statistics of the absolute panel are untouched: two still gate and
three still report, for the reasons `ltx25-oracle-absolute.md` §5 measures.

### The gate FIRES end to end, on exactly the blind spot the landed row declares

One run of the tool, a 25-frame noise render against the committed reference and
the pinned scorer:

    [PASS] content.a.not_uniform / distinct_frames / motion
    [PASS] absolute.a.blockiness_grid8   1.001814 <= 1.143697
    [PASS] absolute.a.blockiness_grid32  0.999606 <= 1.147804
    [FAIL] absolute.a.adherence_clip     15.7422 >= 36.0087 ... margin -20.2665
    [FAIL] absolute.a.adherence_argmax   argmax is 'far:4' ... per-frame wins 0/25
    READING WORSE_THAN_ORACLE
    VERDICT FAIL (exit 1)

`ltx25-oracle-absolute.md` records, with a test, that a pure-noise render passes
C0 and passes both blockiness ratios. That is not an argument here; it is the
first five lines of this run's own output. The two adherence checks are what
catches it.

### What the two checks are worth, separately

S1 and S2 are not redundant and the noise run shows why: S1 fails it by 20 CLIP
points, S2 fails it by ranking. But S1 alone would pass a render that scored
highly against *every* prompt, and S2 alone would pass a render that ranked
correctly at a uniformly terrible score. Each covers the other's failure, and
both print their margin on every run so a passing gate can be seen degrading.

## Owed

- **[#1854](https://github.com/mudler/vllm.cpp/issues/1854) sub-question 1 is
  ANSWERED FOR THE REFERENCE'S REQUEST AND STAYS OPEN.** There is now a scorer,
  a shape, a bound and a gate that fires, and OUR render has not been through it.
  What closes #1854 is W3: our engine's 25 frames at the reference's exact
  request, scored, with the verdict recorded whatever it says. Owner: this row,
  through [#2295](https://github.com/mudler/vllm.cpp/issues/2295).
- **Our render's frames are not established to exist.** `rc` job
  `4b0666ee-248c-45fc-9de6-372b6d0c1fab` produced them on `dgx:gpu0` and they
  were never committed. Whether W3 needs a fetch or a fresh lease is UNMEASURED
  and is its first step. Owner: this row.
- **The `VllmRunner` half of the ported upstream test is NOT run.** The ported
  case holds vLLM-Omni's `outputs.image_embeds` route against vLLM's
  `get_image_features` route at upstream's own tolerance, which is the half that
  can run here; upstream's own other side needs an installed vLLM and a GPU
  lease, and this tree's vLLM is a source checkout. So **`CLIPEmbeddingModel`
  itself has still never been loaded in this tree**, and no claim in this row
  rests on it. Owner: this row.
- **The `20.0` and `7.0` constants in vLLM-Omni's own gate remain undefended**,
  and this row still declines to import them. Nothing here uses an absolute
  floor. If a later reader wants one, it needs a derivation this row does not
  have. Owner: this row.
- **NO CANDIDATE CAN ANSWER #1854'S OWN 70-WORD EXAMPLE.** CLIP's context is 77
  positions and that prompt needs at least 83; the tool refuses it rather than
  truncating. A scorer with a longer text context is not upstream's choice and
  has never run here. This is the sharpest limit on what landed and it is stated
  in the tool's own output, in the pin, and in the suite. Owner: this row.
- **Adherence is scored per FRAME, so temporal adherence is invisible.**
  "walks SLOWLY" is not measured by anything here, exactly as it is not by
  vLLM-Omni's own middle-frame scorer. A video-native scorer would be a different
  question. Owner: this row.

## Now

`ACTIVE`. W0, W1 and W2 are COMPLETE and in this row's two changes. §9 is
ANSWERED — instrument, not oracle — so nothing in this row is blocked on a
decision, and nothing in it is blocked on hardware either: the whole measurement
is CPU.

W3 is the only wave left and it is the one that closes #1854's first
sub-question. It needs our engine's 25 frames at the reference's request, and
its first step is to find out whether the frames `rc` job
`4b0666ee-248c-45fc-9de6-372b6d0c1fab` wrote were retained.
