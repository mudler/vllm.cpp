# SPEC — `LTX25-ADHERENCE-BISECT`: is the adherence gap in the LATENT or in the VAE DECODE?

Issue: [#2514](https://github.com/mudler/vllm.cpp/issues/2514).
Owner row: `LTX25-ADHERENCE-BISECT`.

`LTX25-PROMPT-ADHERENCE` measured the gap and
[`ltx25-prompt-adherence.md`](ltx25-prompt-adherence.md) `## Owed` records that
**no row owns closing it**, and that naming an owner "needs the attribution
first". This row is one step of that attribution and it is deliberately a small
one: it halves the search space and it stops.

## Scope

IN scope:

- Establishing whether our final video latent or our video VAE decode carries
  the divergence, by a measurement that does not depend on matching the
  sampler's noise draw.
- Giving the S1 verdict the error bar it does not have, by retaining and scoring
  the frames of every render taken under this row.
- Recording the memory format of the latent on both sides, because
  AGENTS.md requires the dtype comparison and a token gate cannot see it.

OUT of scope, and named so that nobody reads a silence as a claim:

- **The repair.** Localizing is this row. Fixing is the next one, and this row
  does not open it.
- **The pixel-statistics diagnosis.** [#2513](https://github.com/mudler/vllm.cpp/issues/2513)
  (`LTX25-ADHERENCE-DETAIL-LOSS`) owns the frequency-domain analysis of the two
  existing frame sets. This row must not duplicate it.
- **Tuning anything to raise the score.** Fitting an output to a gate is what
  AGENTS.md forbids, and this gate is one sample deep.
- **Audio.** The audio latent is dumped where it is free to dump, and nothing
  in this row's verdict rests on it.

## The request, which is already pinned

Byte for byte the manifest's, and identical to every neighbouring row's:

    prompt   "A red fox walks slowly through a snowy pine forest at sunrise, cinematic."
    320x192, 25 frames, 8 inference steps, seed 42
    the four BF16 checkpoints of tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json

Latent tokens = `(320/32)*(192/32)*ceil(25/8)` = **240**. The oracle's own render
of this request took **93.8 s** (`ltx2_oracle_manifest.json`,
`result.render_seconds`), so this is minutes of compute behind an hour of
staging.

## Upstream anchors

Read at the pinned oracle revision `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
(`.agents/oracles/ltx-2.md`), in the local clone `/home/mudler/_git/LTX-2`,
verified clean at that SHA.

| What | Anchor |
|---|---|
| Pipeline entry | `packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py:220-241` |
| The denoise loop | `packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py:39-81` |
| Unpatchify to the 5D latent | `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:575-577` |
| **The decode seam** | `ti2vid_one_stage.py:243` — `self.video_decoder(video_state.latent, ...)` |
| `LatentState.latent` | `packages/ltx-core/src/ltx_core/types.py:279` |
| **The supported diagnostics hook** | `blocks.py:585-621` — `RecordingDiffusionStage` |

Our side, on `main` at `6bf3abb58`:

| What | Anchor |
|---|---|
| ABI entry | `include/vllm.h:1140` — `vllm_video_generate` |
| The render | `src/vllm/multimodal/ltx2_video.cpp:2130` — `Ltx2VideoEngine::Generate` |
| The denoise loop | `src/vllm/multimodal/ltx2_video.cpp:4642-4828` (phase `denoise`) |
| Unpatchify to the volume | `src/vllm/multimodal/ltx2_video.cpp:4893` |
| The latent volume | `src/vllm/multimodal/ltx2_video.cpp:2970` — `std::vector<float> video_latent_volume`, `[C,F,H,W]` |
| **The decode seam** | `src/vllm/multimodal/ltx2_video.cpp:5457` — `Ltx2VideoDecodeStreaming(...)` |
| Decoder signature | `include/vllm/model_executor/models/ltx2_tiling.h:313` |

## Design

### The obvious experiment is invalid, and that is this row's first finding

Running both engines at seed 42 and differencing the two final latents cannot
answer the question. [`ltx25-oracle-absolute.md`](ltx25-oracle-absolute.md)
`### What this change lands, and why it is two jobs and not one

| File | What it is |
|---|---|
| `tools/oracle/ltx2_latent_dump.py` | runs upstream's own `main()` in-process with `RecordingDiffusionStage` installed, dumps the final video latent, and proves the dump changed nothing |
| `tools/oracle/jitprobe.py` | the triton JIT gate the LTX-2 setup recipe already depended on, which existed only on an untracked NAS share; committed byte-for-byte at the digest that ran |
| `scripts/ltx25-adherence-bisect.sh` | phase A: pin, venv, stage-and-verify, dump, copy out |
| `scripts/ltx25-render-confirm.sh` | gains `KEEP_FRAMES=1`, which is phase B |

**Phase B is a separate `rc` job, deliberately.** The n = 1 question needs three
of OUR renders, and `ltx25-render-confirm.sh` already takes them with a verified
binary, a staged checkpoint set and a phase table. It deleted renders 2 and 3's
frames at its `:570`, which is the single line that made the reading n = 1
-- `[ "$i" = 1 ] || rm -f "$D"/frame_*.ppm`, the only hit for that pattern in
the file. It sat at `:551` at this row's base and `326f4970b` moved it 19 lines
by inserting the comment above it, so the first version of this sentence cited a
line its own commit had invalidated.
`KEEP_FRAMES=1` retains them instead. **The `if [ "$i" = 1 ]` compare block is
untouched**, so the gate that harness enforces is byte-for-byte the gate it
enforced before, and the retained frames are evidence for a later CPU scoring
pass rather than a second verdict. `.agents/specs/ltx25-prompt-adherence.md`
`## Owed` named both ways to close n = 1 and left the choice open; this is the
retain-past-the-loop one, chosen because it cannot move the verdict.

Two jobs of under an hour survive an hourly crash where one job of two hours
does not, and `dgx` has crashed roughly hourly under long sequences. Phase A is
queued first because it answers the row's primary question and is the cheaper
half: the oracle's own render is 93.8 s, and its output is 61,440 bytes.

**Phase B reuses the CACHED binary at `SRC_SHA 790c582bbba45ab0f7b74aafee361e4557a84bf2`,
which is the binary that produced the reading in the record.** That is the right
subject for an error bar: rebuilding at today's `main` would measure the spread
of a DIFFERENT engine and confound run-to-run noise with 112 commits of drift.

## Risks/decisions` already records the reason: **"Same prompt, same seed
integer, different engine, so the sampler's noise is not the same draw."** Two
different draws through two CORRECT denoisers land on two different latents, so
a large elementwise difference is exactly what a healthy pair would also
produce. The instrument could not separate the case it exists to separate. That
is `an-instrument-whose-failure-looks-like-a-result` in its purest form, and it
is written here rather than discovered after a lease.

### The cross-decode is noise-independent, and it is the measurement

Dump the ORACLE's own final video latent once, then decode that ONE latent
through both VAEs:

| Arm | Decoder | Pixels |
|---|---|---|
| `O/O` | the oracle's VAE | the committed reference render, already digest-verified in the tree |
| `O/V` | **our** VAE | the new measurement |

Both are scored on the pinned CLIP instrument that `LTX25-PROMPT-ADHERENCE`
landed. The input is byte-identical between the arms, so the reading is a paired
test and no noise matching is needed.

The verdict, in the units of the gap itself (reference 38.1278, ours 35.2719,
gap 2.8559):

- `O/V` near **38.1278** → our VAE decode is faithful, and the divergence lives
  in the LATENT: scheduler, sigma schedule, guidance/STG, RoPE, or the DiT
  forward. The VAE is exonerated.
- `O/V` near **35.2719** → the VAE decode accounts for the gap.
- `O/V` between → the split is quantified rather than argued, and the row
  reports the fraction each half carries.

The one-sided reading is the strong one: a VAE that reproduces upstream's own
frames from upstream's own latent cannot be what degrades a good latent.

### The oracle dump changes no upstream byte

`RecordingDiffusionStage` (`blocks.py:585-621`) is upstream's OWN supported seam
for reaching a stage's outputs from outside a pipeline; its docstring names the
substitution `pipeline.stage = RecordingDiffusionStage(pipeline.stage)`. The
driver substitutes it and reads `stage.video_states[0].latent`. **No file under
the pinned checkout is edited**, so the pin still describes the bytes that ran.

`ti2vid_one_stage.main()` builds its pipeline in a local, so the substitution
needs a driver that mirrors `main()` rather than a monkeypatch into it. That
driver is therefore a second thing to validate, and §"Tests" says how.

### The dtypes differ, and the row records it either way

Upstream's final latent is **bf16**, shape `(1, 128, 4, 6, 10)`
(`ti2vid_one_stage.py:79`, `types.py:73-88`). Ours is **f32**, shape
`[C,F,H,W]` with no batch axis (`ltx2_video.cpp:2970`). AGENTS.md requires the
memory-format comparison beside the value one, and an f32 model-path buffer
where upstream carries bf16 is the polarity that section names. This row
REPORTS that difference. It does not change it: narrowing a dtype to move a
score is the tuning this row forbids itself.

## Risks

- **The dump could perturb the run.** A probe that synchronizes can cure the
  symptom it was added to observe. Controlled below.
- **The driver is not upstream's `main()`.** Controlled below.
- **The CPU VAE arm may be too slow or may not reach the same kernels as the
  CUDA arm.** If our cross-decode runs on CPU and the render ran on CUDA, the
  arms differ in more than the latent. Recorded as a limit, and the CUDA arm is
  preferred where a lease allows it.
- **n = 1 may swallow the verdict.** If the S1 reading moves by more than 0.7368
  between renders of one unchanged binary, the -0.7368 margin is not a finding
  and neither is any bisection built on it. This row reports that FIRST if it
  happens.
- **Every `AGREE` row rests on checkpoint metadata that neither side re-reads
  here.** Upstream does not hold these defaults as constants: it resolves them
  from the checkpoint, through `resolve_cli_params` (`args.py:485-495`) into
  `detect_checkpoint_path` (`:460-477`) and `detect_params`
  (`constants.py:166-179`), and that reader falls back to `LTX_2_PARAMS`
  silently for "older, unset, or unreadable versions". `LTX_2_PARAMS` carries
  `stg_blocks=[29]` and 40 steps (`constants.py:47`, `:56`, `:66`, `:80`), not
  `[28]` and 30. So `stg_blocks [28] AGREE` is conditional on the BF16
  transformer's `model_version` parsing as at or above `(2,4)` on upstream's own
  reader, and a checkpoint whose metadata does not parse would put upstream on a
  different STG block and a different step count while this table still read
  AGREE. Phase A's own run confirms it: `detect_model_version` logs the version
  it read (`constants.py:162`), so the run's log settles which branch upstream
  took, and no separate experiment is owed.
- **`dgx` has crashed roughly hourly under long sequences.** The lease job is
  phase-structured and checkpoints each phase's artefacts to `/workspace` before
  the next begins, so a crash costs one phase and not the run.

## Tests, and the controls that make each instrument admissible

1. **The dump did not perturb the oracle.** The recording run's 25 decoded
   frames must match the committed
   `tests/parity/goldens/ltx2_oracle/SHA256SUMS` digests. This single check
   proves three things at once: the oracle is reproducible at this request, the
   `RecordingDiffusionStage` substitution changed nothing, and the hand-written
   driver is equivalent to upstream's `main()`. A mismatch invalidates the
   latent and the row says so rather than scoring it.
2. **The dump actually ran.** The driver asserts `len(stage.video_states) > 0`
   and that the written file's byte length equals the product of the recorded
   shape and itemsize, and it exits non-zero otherwise. A zero counter that
   could not be written must not read as a pass.
3. **The cross-decode consumed the oracle's latent and not ours.** The tool
   prints the sha256 of the latent file it read and the shape it parsed, and
   refuses a shape that is not the oracle's.
4. **n, with its spread.** Every render taken under this row retains its frames
   and is scored. `loadavg` and GPU SM clock are recorded beside each, because
   the existing 8.03% wall spread is what a three-times-busier box looks like
   and not evidence about pixels.
5. **Each control's verdict reaches the EXIT CODE.** Tests 1 and 2 are only worth
   their words if the driver acts on them, and a static check that the functions
   are CALLED cannot see that. `tests/scripts/test_ltx2_latent_dump.py` drives
   `main()` end to end with the torch, upstream, GPU and checkpoint seams stubbed
   and asserts rc 64 on a decode missing a committed frame, rc 62 on a short
   write, and rc 61 when the recording hook never fired. Deleting any one guard
   reds it; deleting none is visible to an AST walk.

   The rc 61 leg was added last and is the one worth naming, because it is the
   half of test 2 -- `len(stage.video_states) > 0` -- that had no executing
   check. Every other leg's fake stage supplied a video state, so the branch was
   never entered and `return 61` was documented and unexecuted; deleting it left
   the suite fully green. The fake now takes its states as an argument, and the
   empty case is driven. With the guard removed, `states[-1]` raises rather than
   returning, so that leg catches and reports instead of letting a traceback out
   of the suite: a red that names a Python line is a worse red than one that
   names the guarantee that moved.
6. **The committed Phase A artefacts are digest-checked, not merely stored.**
   `tests/parity/goldens/ltx2_oracle_latent/SHA256SUMS` had no consumer when it
   landed, which made it a comment. `tests/scripts/test_ltx2_oracle_goldens.py`,
   which already does this for the sibling `ltx2_oracle/` directory, now covers
   this one too rather than a parallel suite existing beside it. Every artefact
   must have a row, every row an artefact, and the bytes must agree with both --
   and, because a digest file beside its own artefacts can always be edited to
   match, each `.raw` is cross-checked against the sha256 the manifest records
   the RUN computing on the worker before anything was copied out.

## Gates

- `scripts/agent-preflight.sh` at rc 0 before the commit.
- The oracle-frame digest check of test 1, which is the run's own admissibility.
- `python3 tests/scripts/test_ltx2_oracle_goldens.py`, which is where test 6
  runs and which the preflight SUITES list already carries.

## Evidence

In the repository, which is where a claim has to be readable after a worker's
share is reaped:

- `tests/parity/goldens/ltx2_oracle_latent/` -- the two latents, the run's own
  manifest, and `phase-a-rc-job-log.txt`, the `rc` log of job
  `a24f9336-897f-4810-9851-13dade5acb52`. Every Phase A number in this spec is
  read off that log rather than typed from memory, including the
  `model_version=2.5.0` line the AGREE-row condition rests on.

On the worker, under `/workspace/ltx25-adherence-bisect/run/<RUN_ID>/`, copied
off its local disk as each phase completes: `PROVENANCE`, the oracle latent and
its sha256, the recording run's frames and their digest comparison, the
cross-decode frames, every retained render's frames, and one `compare-*.json`
per scored arm. That share is CIFS and outside the repository, so it is a
convenience and never the record.

## Owed

- **Attributing the audio divergence.** Phase A's `audio.wav` differs from the
  committed reference on 94.2% of samples and the audio decoder takes no
  generator, so the benign explanation is unavailable. It is a nondeterministic
  audio kernel, an environment difference, or a different audio latent -- and the
  third would be a finding about this row's probe. It needs a second Phase A run
  under a recorded environment, which the driver now writes. No claim in this row
  rests on the audio half in the meantime.

## Stop conditions

- `NEEDS_DECISION` rather than starting a repair. Localizing is this row.
- `REMOTE_UNVERIFIED` rather than a guess if the controller cannot be reached,
  and never `ssh` plus a file mutex on a fleet device.
- If the S1 gap sits inside run-to-run noise, that result outranks the
  bisection and the row leads with it.

## Findings so far

### A LATENT-side divergence is PROVEN statically, and it needed no GPU

The cross-decode is still queued, but a static read of the two denoise loops at
their pinned revisions already answers half the question, and it answers it in
the direction the cross-decode was built to test.

**Of the seven denoise-loop parameters compared below, exactly one differs:
the sigma schedule's shift anchor.** Guidance, the negative prompt, the sampler
and STG all AGREE, each checked to a `file:line` on both sides. The claim is
bounded by the table and does not reach past it: `noise_scale`, the step count,
RoPE, the DiT forward, text encoding and the initial noise draw are NOT compared
here, and `## Design` lists RoPE and the DiT forward among the latent-side
candidates this row leaves open.

| Parameter | Upstream | Ours | |
|---|---|---|---|
| video `cfg_scale` / `stg_scale` / `rescale_scale` / `modality_scale` / `skip_step` | 3.0 / 1.0 / 0.7 / 3.0 / 0 (`constants.py:51-55`) | same (`ltx2_pipeline.cpp:950-954`) | AGREE |
| `stg_blocks` (both modalities) | `[28]`, the 2.3 override the 2.5 key inherits (`constants.py:86`, `:124`, `:130-133`) | `{28}` (`ltx2_pipeline.cpp:969-970`, `:974-981`, `:1005-1014`) | AGREE |
| audio guidance | 7.0 / 1.0 / 0.7 / 3.0 / 0 (`constants.py:61-65`) | same (`ltx2_pipeline.cpp:956-960`) | AGREE |
| negative prompt | `DEFAULT_NEGATIVE_PROMPT` (`constants.py:186-199`) | `LightricksNegativePrompt()` (`ltx2_pipeline.cpp:1101-1107`) | AGREE, byte for byte, 1171 chars |
| sampler | `euler_denoising_loop` + `EulerDiffusionStep`, deterministic, no per-step noise (`samplers.py:39-81`, `diffusion_steps.py:25-40`) | the first-order arm with `kEuler` (`ltx2_video.cpp:4731-4827`, `ltx2_pipeline.cpp:241-258`) | AGREE |
| STG application | `SKIP_VIDEO_SELF_ATTN` + `SKIP_AUDIO_SELF_ATTN` on block 28 (`denoisers.py:111-119`) | `kSkipVideoSelfAttn` + `kSkipAudioSelfAttn` (`ltx2_denoisers.cpp:144-158`) | AGREE |
| **sigma shift anchor** | **4096** | **240** | **DIFFER** |

Two of those cites are not their file's only occurrence, which the AGREE claims
survive but a reader re-resolving them should know: `video_guider.cfg_scale =
3.0` appears at `ltx2_pipeline.cpp:950` and again at `:990`, and
`audio_guider.cfg_scale = 7.0` at `:956` and `:996`. The second of each pair is
inside `Ltx2Params23Hq()`, a lineage `one_stage` never reaches. The cites at
`:969-970`, `:950-954` and `:956-960` are the ones on this render's path, and
`kSchedulerDefault` really is assigned exactly twice in that file (`:1776`,
`:1961`), which are also the only `schedule_tokens` assignments in it.

Upstream's `LTX2Scheduler.execute` takes an OPTIONAL latent and
`packages/ltx-core/src/ltx_core/components/schedulers.py:32` reads
`tokens = math.prod(latent.shape[2:]) if latent is not None else
default_number_of_tokens`. `ti2vid_one_stage.py:207` calls
`self._scheduler.execute(steps=num_inference_steps)` and passes **no latent**,
so upstream takes `MAX_SHIFT_ANCHOR = 4096` (`schedulers.py:11`, `:29`).

Our `Ltx2SigmaSchedule` (`src/vllm/model_executor/models/ltx2_pipeline.cpp:107-147`)
mirrors that formula exactly -- same `max_shift` 2.05, `base_shift` 0.95,
`stretch`, `terminal` 0.1, `power` 1. The formula is not the divergence. The
ARGUMENT is. `src/vllm/multimodal/ltx2_video.cpp:4227-4231` passes
`target_tokens` unless the phase asks for the scheduler default, and
`OneStagePhase` (`ltx2_pipeline.cpp:1124-1147`) never sets `schedule_tokens`, so
it keeps the struct default `kTargetLatent`
(`include/vllm/model_executor/models/ltx2_pipeline.h:725`). The only two
assignments of `kSchedulerDefault` in that file are at `:1776` and `:1961`, and
neither is `one_stage`.

At this render's geometry `target_tokens = ceil(25/8) * (192/32) * (320/32)
= 4 * 6 * 10 = 240`, so `sigma_shift = 240*mm + b = 0.669271` against upstream's
`2.050000`, and `exp` is 1.952813 against 7.767901. Recomputed here from both
sources rather than transcribed:

| step | upstream (4096) | ours (240) | delta |
|---|---|---|---|
| 0 | 1.000000 | 1.000000 | 0 |
| 1 | 0.965712 | 0.921534 | -0.044178 |
| 2 | 0.921875 | 0.832166 | -0.089708 |
| 3 | 0.863856 | 0.729457 | -0.134399 |
| 4 | 0.783445 | 0.610176 | -0.173269 |
| 5 | 0.664579 | 0.469962 | **-0.194617** |
| 6 | 0.471003 | 0.302774 | -0.168228 |
| 7 | 0.100000 | 0.100000 | 0 (pinned by `stretch`/`terminal`) |
| 8 | 0.000000 | 0.000000 | 0 |

Every intermediate sigma differs, by up to 0.1946. Our schedule leaves the
high-noise regime EARLY -- at step 4 we are at 0.610 where upstream is at 0.783 --
and the high-noise steps are where classifier-free guidance sets global
composition and prompt semantics, which is what a CLIP prompt-adherence score
measures. That is a mechanism, and it points the right way.

**This is not a new discovery about upstream; it is a new attribution.** The
tree already carries the reading, in this row's own words, at
`include/vllm/model_executor/models/ltx2_pipeline.h:659-706`: it enumerates
upstream's seven `.execute()` call sites, records that six pass no latent, names
`ti2vid_one_stage.py:207 -> our one_stage x4` in that list, calls the divergence
"REAL rather than a rounding", and states that the default was left at today's
behaviour deliberately because flipping it re-samples six shipped arms and
rewrites their goldens. What was NOT known is that this arm is the one carrying
a measured 2.8559-point adherence FAIL. The comment's own worked example is at
the recipe default geometry, 6144 tokens and shift 2.78, where the error points
the OTHER way; at 320x192x25 it points down and is larger.

### What this does and does not establish

- **ESTABLISHED: our final latent cannot be upstream's**, and for a reason that
  is not the noise draw. Two engines walking different sigma trajectories are
  solving different problems at every intermediate step.
- **NOT ESTABLISHED: that our VAE decode is faithful.** Both halves can be
  wrong at once. Only the cross-decode exonerates the VAE, and it is queued.
- **NOT ESTABLISHED: that this schedule difference CAUSES the 2.8559 points.**
  It is a mechanism with the right sign and the right regime, which is a
  hypothesis and not a measurement. The ablation that would test it is one line
  -- setting `OneStagePhase`'s `schedule_tokens` to `kSchedulerDefault` -- and
  this row does not run it, because changing a value until a score improves is
  the tuning the scope forbids and because that flip belongs to whoever owns the
  six arms it re-samples.

### The owner reference in that comment does not resolve, and that is REMOTE_UNVERIFIED

`ltx2_pipeline.h` names `https://github.com/mudler/vllm.cpp/issues/1150` as the
owner of the flip. `gh api repos/mudler/vllm.cpp/issues/1150` returns 404 --
**and so do 1148, 1149, 1151 and 1152**, while 2513 and 2514 resolve normally in
the same session. Five consecutive numbers failing together is a range effect in
the client, not five deletions, and this repository has already written the
"they were all deleted" conclusion into a policy file once and had to retract
it. So the state of #1150 is **REMOTE_UNVERIFIED**, not absent, and nothing here
concludes that the flip is unowned. It needs a check from a client that can read
that range.

## Phase A RAN, and its control PASSED

`rc` job `a24f9336-897f-4810-9851-13dade5acb52`, `dgx:gpu0` (GB10),
2026-09-01T23:45Z, 1954 s wall of which the render was **75.9 s**.

**All 25 decoded video frames reproduce
`tests/parity/goldens/ltx2_oracle/SHA256SUMS` byte for byte** -- `matched = 25`
of `committed_frames = 25`, no mismatch, no missing frame, no unlisted extra.
Recomputed independently from the retained frames rather than read off the
driver's own manifest. That control establishes all three things it was built to
establish **on the video path**: the oracle is reproducible on it at this
request, installing upstream's own `RecordingDiffusionStage` perturbed nothing on
it, and the in-process driver is equivalent to upstream's `main()` on it. It
gates `frame_*.ppm` and nothing else, so it says nothing about the audio path,
where the same run diverged -- see below, where that divergence is left OPEN and
unattributed rather than folded into this control's verdict.

The job's own log is committed beside the latents as
`tests/parity/goldens/ltx2_oracle_latent/phase-a-rc-job-log.txt`. Every number this
section takes from the run can be read off it.

The video latent is therefore the reference render's own, and both latents are
committed at `tests/parity/goldens/ltx2_oracle_latent/`: video `(1,128,4,6,10)`
bf16, 61,440 bytes; audio `(1,8,26,16)` bf16, 6,656 bytes. **The cross-decode
now needs no lease at any point**, which is why 68 KB is worth committing -- and
that claim needed the decode configuration recovered before it could be made.

### The decode configuration was NOT recorded, and it is now recovered

The committed manifest records the SAMPLING request -- prompt, geometry, steps,
seed, offload, device. Upstream's decode seam takes three more inputs, at
`ti2vid_one_stage.py:243`:

    decoded_video = self.video_decoder(video_state.latent, tiling_config, generator=generator, dtype=vae_dtype)

None of the three appeared in the manifest, this file, or the golden's preamble.
That matters exactly as much as this row's own first finding: a cross-decode run
against an unrecorded decode configuration cannot separate *our VAE is
unfaithful* from *we decoded under different tiling, noise or dtype*, which is
the instrument failure the naive latent diff was rejected for.

All three are recovered **statically**, from the pinned checkout
(`fd4ded7f`, a clean local clone at that revision) plus the video VAE
checkpoint's own 57,464-byte safetensors metadata header. No GPU, no lease.

| Decode input | What it is for THIS request | How it resolves |
|---|---|---|
| `vae_dtype` | `torch.bfloat16` | `main()` computes `vae_dtype_for_hdr(hdr, torch.bfloat16)` (`ti2vid_one_stage.py:264`). `--hdr` defaults to `None` (`args.py:41-45`), there is no EXR input, so `resolve_hdr_color_space` returns `None` (`color_config.py:82-92`) and `vae_dtype_for_hdr` returns its default (`:64-66`). `VideoDecoder.__call__` then does `latent.to(dtype=build_dtype)` (`blocks.py:1136-1137`), a no-op on a bf16 latent. |
| `tiling_config` | `TileSizeConfig(frames=80/24, height=448/64, width=768/64)`, which is **one tile** on this latent grid | `AUTO_TILING` (`ti2vid_one_stage.py:148`, `:296`) resolves through `ensure_tiling_config` (`helpers.py:119-146`). The checkpoint's `config.vae._class_name` is `CausalVideoAutoencoder`, which IS `_CONV_VAE_CLASS_NAME` (`model_configurator.py:18`, `:26-33`), so `is_diffusion_video_vae` is false and `tiling_config_for_vae` takes the aspect-only conv branch (`helpers.py:89-97`) -- which never reads free VRAM, unlike the DiffVAE branch. Resolved by RUNNING those upstream functions against the real checkpoint, not by transcribing them; their splitters on the `(4, 6, 10)` latent grid return one interval on each of T, H and W. |
| `generator` | the seed-42 `torch.Generator` from `:160`, **never consumed by this decode** | It is the same object `GaussianNoiser` uses, so at `:243` it is in whatever state the denoise loop left it -- not its seeded state, and not recoverable from the latent. It does not matter here, because every `torch.randn(..., generator=generator)` on the conv path is behind one of two flags that are both off. `timestep_conditioning` gates the decode-noise draw (`conv_video_decoder.py:287-298`, `memory_efficient_decode.py:562-567`) and this checkpoint sets it `false`. `inject_noise` gates the per-block draws (`resnet.py:115`, `:153`, `:171`; `memory_efficient_decode.py:327`, `:349`) and is built as `block_config.get("inject_noise", False)` (`conv_video_decoder.py:81`, `:92`, `:106`) from a nine-entry `decoder_blocks` list that carries no such key. |

The tiling row is the only one that needed execution rather than reading, and
this is the whole of it -- upstream's own functions, the real checkpoint, CPU,
seconds:

```python
sys.path.insert(0, "<LTX2>/packages/ltx-core/src")
sys.path.insert(0, "<LTX2>/packages/ltx-pipelines/src")   # stub `av`,
                                                          # `OpenImageIO`,
                                                          # `soundfile`: media
                                                          # libs, no tiling math
ensure_tiling_config(
    AUTO_TILING,
    scale_factors=tiling_scale_factors_for_vae(VAE),
    vae_checkpoint_path=VAE,
    video_shape=VideoPixelShape(batch=1, frames=25, height=192, width=320, fps=24.0),
    device=torch.device("cpu"))
# -> TileSizeConfig(frames=DimensionSizeConfig(tile_size=80, overlap=24),
#                   height=DimensionSizeConfig(tile_size=448, overlap=64),
#                   width=DimensionSizeConfig(tile_size=768, overlap=64))
# .video_chunks_number(25) -> 1; .to_splitters(sf) on (4, 6, 10) -> one
# DimensionInterval(start=0, end=N, left_ramp=0, right_ramp=0) per axis.
```

`VAE` is the `ltx-2.5-video-vae-conv-bf16.safetensors` this run staged, whose
sha256 `685b06ee..97dfce8d` the oracle file already pins. `device` is CPU because
the conv branch never reads it; only the DiffVAE branch would.

The generator is the one that decided this. Had it been consumed, upstream's
decode noise would depend on the state 8 denoise steps left the generator in,
and 68 KB of latent could not reproduce it at any price short of replaying the
whole loop. It is not, so the cross-decode's inputs are fully determined by the
pin plus the checkpoint, and there is nothing run-varying left to have recorded.

**This covers the VIDEO decoder only.** `AudioDecoder.__call__(self, latent)`
(`blocks.py:1180`) takes no generator and no tiling config at all, so none of
this explains the audio divergence below.

### The condition on every `AGREE` row is now settled, by this run

`## Risks` records that upstream does not hold its defaults as constants: it
resolves them from checkpoint metadata and `detect_params` falls back SILENTLY
to `LTX_2_PARAMS`, which carries `stg_blocks=[29]` and 40 steps rather than
`[28]` and 30. Phase A's log answers which branch it took, in upstream's own
words:

    INFO:ltx_pipelines.utils.constants:Checkpoint declares model_version=2.5.0 (parsed as (2, 5, 0))

So upstream took the 2.5 branch and the silent fallback did NOT fire, and the
`stg_blocks [28]` row is unconditional for this render. There is no step-count
row in the AGREE table and this sentence used to close one: the 30 is
`Ltx2Params23()`'s default at `ltx2_pipeline.cpp:968`, and BOTH sides override it
with `--num-inference-steps 8` for this render, so it is not a parameter the
comparison rests on either way. Within the seven rows compared, the sigma anchor
stands as the one divergence. Our side reaches `[28]` the same
way: `Ltx2Params20()` sets `{29}` at `ltx2_pipeline.cpp:955` and `:961`, and
`Ltx2Params23()` overrides both to `{28}` at `:969-970`, mirroring upstream's
own base-plus-override shape.

### The video half is bit-exact, and the audio half is an OPEN observation

Recorded because it bounds what this control covers, and left open because
nothing here attributes it. The re-run's `audio.wav` has the same header and
sample count as the committed one and differs on **94.2%** of its samples, by at
most **215 LSB** of int16 against a peak amplitude of 13010 (rms 36.4 LSB). The
container moved too: 225,174 bytes against the committed 225,151.

**"A different noise draw" is not available as the benign explanation here.**
`AudioDecoder.__call__(self, latent) -> Audio` (`blocks.py:1180`) takes no
`torch.Generator` and no tiling config, unlike `VideoDecoder.__call__`
(`blocks.py:1124-1143`). What is left is a nondeterministic audio VAE or vocoder
kernel, a different torch, or **a different audio latent** -- and the audio
latent comes from the very stage `RecordingDiffusionStage` wraps
(`ltx2_latent_dump.py:410` reads `stage.audio_states`, the file's only hit for it). The third case would be a
finding about the probe, so this is not the same statement as "the audio half is
simply not bit-exact".

Phase A recorded no environment, which is why the attribution cannot be made from
what was captured: the reference render's own
`tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json` records
`torch 2.13.0+cu130` on an `NVIDIA GB10`, and this run's own job log records the
same pair across two lines rather than one -- `phase-a-rc-job-log.txt:51` the
torch, `:29` the GB10 -- so a torch mismatch is not the answer, but the driver
and the CUDA driver version were never written down and the comparison stops
there. `tools/oracle/ltx2_latent_dump.py` now writes an `environment` block into
its manifest so a future run does not repeat the omission.

So the DiT plus video-VAE path reproduced exactly while the audio path did not.
The frame control gates only `frame_*.ppm`, which is the half this row needs and
the half it claims. `audio_latent_bfloat16.raw` is recorded WITHOUT a
reproducibility claim, its row in the golden's `SHA256SUMS` is marked
`NOT REPRODUCIBLE` so a reader who greps the digest line never misses it, and
nothing in this row's verdict rests on it. **Owed:** attributing the audio
divergence, which needs a second run under a recorded environment.

### The instrument shipped three defects, and review caught them, not I

Recorded because the spec's `## Tests` section claims these controls work, and
for one commit they did not. `326f4970b` carried an incomplete rename -- a `sed`
that matched `"raw_bf16"` but not `rec['raw_bf16']` inside f-strings -- so the
reporting loop raised `KeyError` on the FIRST latent record and killed the byte
assertion, the frame control and the manifest before any of them ran, exiting rc
1, a code this module documents nowhere. The frame check also iterated the
OBSERVATION set, so a decode producing zero frames yielded no mismatches,
`matched = 0`, and printed `CONTROL PASSED`. An `av` failure read as a passing
control.

Both were catchable with no GPU, no torch and no weights, and neither was
caught: the local validation exercised `write_latent` in isolation and never
drove `main()`. `962db2598` extracts both controls into callable functions,
derives the expected frame set from `SHA256SUMS` instead of a literal 25, and
fails on a missing frame, a wrong digest, an unlisted extra or an empty
expectation. **The job above ran the REPAIRED driver**, which is why its control
means anything.

**A third defect, in that repair's own evidence.** It proved reachability by
walking `main()`'s AST for a `Call` node, and the pull request body claimed that
as "asserts `main()` reaches both controls". A `Call` node proves the call is
written and cannot see whether the verdict it returns reaches the exit code.
Deleting the `if not frame_check["ok"]: return 64` guard leaves the driver
computing a correct `ok: False`, printing `CONTROL PASSED` and returning 0;
deleting `if rc: return rc` after `report_latents` lets a short-written latent
exit 0. Both mutations left the AST walk green -- defect 2 restored one layer up.
`tests/scripts/test_ltx2_latent_dump.py` now drives `main()` end to end with the
torch, upstream, GPU and checkpoint seams stubbed and asserts the EXIT CODES:
64 on a decode missing a committed frame, 62 on a short write. Both mutations red
against it, and so does dropping `bool(expected)` from `check_frames`'s `ok`.

The red-before claim needs the same correction. The suite is red against
`326f4970b`, but for a weaker reason than "it detects the defects": against that
commit the two controls are not yet functions, so the suite exits at its own
precondition and the D1/D2 legs never run. The evidence that it detects the
defects is the mutation pair applied to the REPAIRED file, which is what the
paragraph above records.

## Now

`ACTIVE`. A **LATENT-side divergence is proven statically** -- our final latent
cannot be upstream's, and a mechanism is named -- and Phase A has RUN and PASSED
its video-frame control, so the oracle's own latent is committed and
digest-verified in the tree. That is not the same as answering the row's
question: the VAE is not yet exonerated, and the schedule difference is not yet
shown to CAUSE the 2.8559 points. Both remain as
`### What this does and does not establish` states them.

What remains needs no oracle lease at all:

- **The cross-decode**, which is what EXONERATES or CHARGES the video VAE. It
  decodes `tests/parity/goldens/ltx2_oracle_latent/video_latent_bfloat16.raw`
  through our VAE and scores it on the pinned CLIP instrument against the
  reference's 38.1278 and our 35.2719. It needs our engine to accept a latent
  from disk, which is a code change this row has not made.
- **Phase B**, `KEEP_FRAMES=1 N=3 scripts/ltx25-render-confirm.sh`, which gives
  the -0.7368 S1 margin the error bar it has never had. `KEEP_FRAMES` is landed;
  the renders are not taken. **n is still 1.**

Neither is a blocker on the finding, which was taken from source at both pins,
recomputed rather than transcribed, and whose one open condition Phase A closed.
