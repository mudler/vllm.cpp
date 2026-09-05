# SPEC — `LTX25-TEXT-COND-DEVICE`: split `conditioning.connector`, then repair what the split blames

Issue: [#2354](https://github.com/mudler/vllm.cpp/issues/2354).
[#2296](https://github.com/mudler/vllm.cpp/issues/2296) is the measurement this
row acts on; [#1269](https://github.com/mudler/vllm.cpp/issues/1269) owns the
CPU-pinned text tower and does not carry the connector.
Owner row: `LTX25-TEXT-COND-DEVICE`.

## Scope

`LTX25-RENDER-SPEED-PARITY` measured, at `n = 3` on `dgx:gpu0`
(`rc` job `0baa109c-43ff-475e-abf1-7a50152ffd5d`, evidence
`/mnt/nas_share/rc/ltx25-render-speed/run/20260829T195851Z`), that an LTX-2.5
render of the oracle's own request is **5.53x** the pinned oracle and that the
21B DiT is **2.92%** of it. `generate.guiders` + `conditioning.connector` are
**312.4 s, 60.3% of the wall, and 3.33x the oracle's entire process.** An
independent instrument — `nvidia-smi` sampling, sharing no code with the phase
log — found the GPU idle for **87 to 88%** of the render at median utilization 0%.

That row measures and repairs nothing, and its `## Owed` says so by name:

> `conditioning.connector` and `generate.guiders` are measured and unowned by any
> row that could move them.

IN scope:

- **W1, the split.** `conditioning.connector` decomposed into WEIGHT
  MATERIALIZATION and CONNECTOR ARITHMETIC, and `generate.guiders` decomposed
  into its tower half and its connector half. Both are sub-leaves of leaves that
  already exist, in the instrument that already ships.
- **W2, the repair the split blames**, together with the red-first test and the
  mutation that says the repair is reached.
- **W3, the same-shape re-measurement**: the same harness, the same request, the
  same geometry, `n >= 3` per arm, two separate build directories.
- **W4, correctness**: the committed absolute-quality gate re-run on the changed
  arm, plus a same-arm pixel comparison against a pre-change render.

OUT of scope, declared rather than approximated:

- **A device arm for the Gemma-4 tower.** #1269 records that the tower is host
  resident BY TYPE — `Gemma4Weights` holds `OwnedTensor` over `OwnedBytes`, which
  carries no device field — so moving it is a weight-arm port and not a queue
  swap. This row does not attempt it and does not pretend the queue at
  `ltx2_video.cpp:2344` is what is in the way.
- **A published benchmark ID.** One request, one geometry, one seed. #2296's own
  reason applies unchanged.
- **The oracle side.** It is `n = 1` at 93.8 s and this row does not re-run it.
  Every ratio quoted here inherits that limit.

## Why the split has to come first, stated as two predictions

#2296's `## Owed` names a hypothesis and is careful to call it one:

> `conditioning.connector` is dominated by loading and widening the connector out
> of the 42 GB DiT file rather than by the connector's arithmetic, and the render
> pays for it four times.

Its corroboration is a MEMORY step, not a TIME one: host peak rises 8.59 GiB
across the tower->connector boundary against 8.06 GB predicted for widening
2.016 B parameters to f32. That is evidence the widen HAPPENS. It is not evidence
the widen is what the 122.388 s went to, and this repository has a name for
promoting the first into the second.

**The arithmetic points the other way, and stating it here is what makes this
row's own result falsifiable rather than confirmatory.** The connector is a
transformer over `rows = 1024` (#1269 records the constant) at
`inner_dim = 4096` video / 2048 audio, 8 layers, 12 `dim^2` parameters a layer:
about 4.2 TFLOP of f32 GEMM per `RunConnector` call. At 122.388 s that is
**34 GFLOP/s**, which is a believable-but-poor rate for `vt::MatmulBT`'s threaded
CPU arm on 20 cores — and it is also within an order of magnitude of what an
8 GB bf16->f32 widen plus 4 GB of first-touch page faults costs. **Neither
estimate excludes the other, and an order of magnitude is not a decision.**

So this row measures the boundary rather than reasoning about it, and it does so
BEFORE choosing between two repairs that are not the same repair: caching weights
and moving arithmetic. The prior row said one sub-phase split settles it. This row
takes that split.

## W1 — the instrument

`RunConnector` (`src/vllm/multimodal/ltx2_video.cpp`) grows a `phase_prefix`
argument and emits two NESTED leaves, `<prefix>.weights` around the two
`Ltx2LoadConnectorWeights` calls and `<prefix>.compute` around
`Ltx2ConnectorCreateEmbeddings`. The negative pass's tower gets `guiders.tower`.

**Nested is load bearing.** `render_phase_log.h` marks a leaf opened inside
another leaf `nested` and EXCLUDES it from `sum_leaf_seconds`, so these five new
records cannot move `unaccounted_seconds` and cannot change what the harness's
own coverage gate (H5, `unaccounted / wall < 1%`) reads. They decompose two
leaves; they do not join the table. That property is what makes the before/after
comparable to #2296's table rather than a different table.

**An empty prefix declines the leaves, and the load-time callers pass one.**
`RunConnector` has five call sites. Two are the load-time `prompt_embeds_path`
path, which the measured render does not take; at load time no leaf is
necessarily open, so a sub-leaf there would be a TOP-LEVEL leaf that joins the
sum and shifts the residue. Declining is therefore correctness, not tidiness.

## W2 — the repair

Named after W1 runs, not before it. The two candidates, and what each would cost:

- **If the split blames WEIGHTS:** materialize `Ltx2VaeWeights` once per render
  and share it between the positive and the negative conditioning pass, instead
  of twice. **Bit-exact by construction** — identical weights and identical
  inputs produce identical outputs, so the pixel comparison should be
  byte-identical rather than merely within tolerance, and that is a checkable
  claim rather than a tolerance to argue about. Peak host bytes do not rise: the
  8 GB is HELD across the two passes instead of being allocated, freed and
  allocated again, so the maximum is the same allocation it always was, now
  live for longer. `RunConnector`'s own header states the opposite policy
  ("THE WEIGHTS LIVE AND DIE INSIDE THIS CALL") and that policy is about the
  ENGINE's lifetime on a 119 GB box; a window inside one render is not the case
  it argues against, and this spec says so rather than quietly contradicting it.
- **If the split blames COMPUTE:** the repair is a device arm for the connector,
  which is not a queue swap either — `Ltx2Attention` interleaves host
  `RmsNormRows` and `Ltx2ApplyRotaryEmb` on raw `float*` between its GEMMs, and
  `Ltx2ConnectorForward` reads weights as host `std::vector<float>`. That is a
  port with its own numerics gate and it is a row of its own. **This row would
  then land the split, report the measured negative, and name it.**

## Tests to port

There is no upstream test for either half: upstream renders, and its connector
is a `torch.nn.Module` whose weights `from_pretrained` already holds. The tests
are therefore this tree's own, and each is an executable observable rather than a
comment:

| ID | Assertion | Red before |
|---|---|---|
| T1 | a prompted one_stage render with CFG on writes `conditioning.connector.weights`, `conditioning.connector.compute`, `guiders.tower`, `guiders.connector.weights` and `guiders.connector.compute`, and every one of them is `nested` | the records do not exist |
| T2 | the same render's phase table holds exactly ONE `*.connector.weights` record | it holds two |

T1's `nested` clause is the part that matters: a sub-leaf that landed
NON-nested would silently enter `sum_leaf_seconds` and change every residue this
campaign has published, and asserting only that the record exists would not see
it.

T2 is W2's guarantee written as a number the shipped instrument already emits, so
it is not a test-only hook. The fixture is
`tests/vllm/multimodal/ltx2_video_fixture.h`'s reduced checkpoint set, reached
through the `one_stage` recipe at `cfg_scale = 3.0` with a negative prompt — the
one shape in this suite that runs `ltx2_video.cpp:3073-3094`.

## Gates

1. `ninja -C <build> test_ltx2_video test_render_phase_log` and both suites green,
   with case and assertion counts recorded rather than the exit status alone.
2. The mutation: delete the production call site's prefix argument in a scratch
   copy and rerun T1/T2; a gate that stays green measured a class, not a
   capability.
3. **The measurement**, under one `rc` lease on `dgx:gpu0`: two CLEAN build
   directories, `n >= 3` renders each, the manifest's exact request.
4. **Correctness, before any speed result is accepted**: `ltx25-render-compare.py`
   against `tests/parity/goldens/ltx2_oracle/`'s committed #1864 reference on the
   changed arm, PLUS a same-arm pixel comparison of arm B's frames against arm
   A's. The blockiness gate is one-sided, so a PASS on it is necessary and not
   sufficient; the same-arm comparison is the sharper instrument and for a
   bit-exact repair it should be byte equality. (This item used to justify that
   with "our render is already smoother than upstream's". **That premise is
   FALSE** -- `LTX25-ADHERENCE-DETAIL-LOSS` (#2513) measured the spectrum and our
   render carries 1.4031x the reference's absolute high-band power at 1.0373x its
   mid-band power, so there is no rolloff to call smoothness. The gate's
   one-sidedness is reason enough on its own, and it is the reason that
   survives.)
5. `scripts/agent-preflight.sh`.

## Dependencies

- `dgx:gpu0` through `rc`. Never `ssh`.
- The four BF16 checkpoints, digests in `ltx2_oracle_manifest.json`.
- CUTLASS and a CUDA 13 toolkit inside the lease, staged the way
  `scripts/ltx25-oracle-absolute-render.sh` stages them (#2220's SONAME check).

## Risks/decisions

- **Two arms means two builds, and they are two DIRECTORIES.** An A/B that reuses
  one build directory measures one binary twice. Both binaries' sha256 are
  recorded and asserted to DIFFER, which is the executable form of that rule.
- **`n = 3` per arm bounds the spread; it does not establish a distribution.**
  #2296 measured `conditioning.connector` at 0.44% spread and `generate.guiders`
  at 1.12% — the two most stable rows in its table — so the phases this row acts
  on are the ones where a small `n` is defensible. `load.dit` at 49.70% is not,
  and nothing here is attributed to it.
- **The SM clock cannot be pinned inside a lease** (`LGC_RC=4`). It is sampled.
  It also matters less here than usual, for the measured reason that 60% of this
  render is host-side: HOST load is the axis that transfers, so `loadavg` and
  `MemAvailable` are recorded per render.
- **The lease is the mutex and nothing else is.** No `flock` beside it.

## Evidence

- The `rc` job id, lease wall-clock, queue depth and contention state.
- Both arms' binary and library sha256, and the two source SHAs.
- The four checkpoint sha256 against the manifest.
- The `weights` / `compute` split, with the sentence that says where the boundary
  is in the source.
- Before/after phase tables, `n >= 3` each, spread per phase.
- The blockiness verdict on the changed arm and the same-arm pixel comparison.

## Stop conditions

Stop and report, do not work around:

- a checkpoint sha256 that does not match the manifest;
- two arms whose binaries hash the same, which means one build was measured twice;
- correctness that cannot be preserved — report rather than trade it;
- an unhealthy or unreachable fleet device.

## Work breakdown

- **W1** — this spec, the instrument, T1.
- **W2** — the repair the split blames, T2.
- **W3** — the lease, both arms, the tables.
- **W4** — correctness, and `## Outcome`.

## Owed

- **The AUDIO-ONLY arm still materializes the connector weights twice.**
  `GenerateAudioOnly` is a private static declared in
  `include/vllm/multimodal/ltx2_video.h`, so it cannot take a type defined in
  `ltx2_video.cpp`'s anonymous namespace, and reaching the render's weight set
  from there would mean putting an internal class into a shipped header for an
  arm this row does not measure. It keeps `RunConnectorFromFile`, which is the
  same code path with the materialization inlined rather than a second copy of
  it. Owner: this row, through #2354.
- **The oracle's 93.8 s is still undecomposed.** Inherited from #2296 unchanged.
- **~~`decode.audio.mel` at 47 s is still unattributed.~~ RESOLVED 2026-08-31 by
  `LTX25-AUDIO-DECODE-COST`** (`79ee71b71`), which measured it as 28 convolutions
  totalling 31.46 GMAC at 1.30 GMAC/s on one core -- one multiply-accumulate per
  roughly four cycles, the latency of a dependent scalar `double` chain -- and
  parallelised `Conv2d` over output lines, byte-identical at seven worker counts.
  Struck rather than deleted, because the bullet is the record of what this row
  did not own. Owner: `LTX25-AUDIO-DECODE-COST`.
- **The 206.0 s that remains after `guiders` and `connector` is still open.**
  Owner: `LTX25-RENDER-SPEED-PARITY`, whose `## Owed` carries it with the split
  (`load` 94.5 s, `decode.audio` 50.7 s, `decode.video` 16.0 s, `denoise` 15.1 s).
  Named here without an owner until now, which is the one gap a sweep of the LTX
  specs' owed items found.
  Removing both entirely leaves 2.20x. Naming it here is what stops this row's
  result being read as the whole answer.

## The arms, pinned

| arm | commit | what it is |
|---|---|---|
| A | `c221c43559cbc09f6176b33d00412194762e2613` | the split instrument, no behaviour change |
| B | `055707217c4a9820490488f596675d200996b03e` | A, plus one materialization per render |

Both share the base `85f65b0e8` and differ by this row's commits ALONE. That is
the point rather than a detail: an A/B whose arms also differ by `main`'s traffic
measures `main`'s traffic. The merge commit this branch later takes is OUTSIDE
the measured pair, and no claim here covers it.

Archived to `/mnt/nas_share/rc/ltx25-text-cond/src{A,B}.tar.gz` with their SHAs
beside them, which the harness reads back and records in `PROVENANCE`.

## Now

`DONE` on what it set out to measure. W1-W4 are complete and the reading is
taken. The repair it carries is a **measured near-negative** and the split is the
result.

## Outcome

**The lease.** `rc` job `ab8a0831-4ae4-4698-8068-40ded708df5c` on `dgx:gpu0`,
2026-08-30 09:51:40Z to 11:26Z, **~1h35m**, `renders_completed_per_arm=3`.
Submitted at queue position 4 and it waited for three other jobs; no device was
cleared and no holder was displaced. Evidence at
`/mnt/nas_share/rc/ltx25-text-cond/run/20260830T095140Z`.

**Every precondition held and each was checked rather than assumed.**
`MemAvailable` 115.0 GiB at start against a 78.0 GiB floor. All four checkpoint
sha256 recomputed inside the lease and all four match
`ltx2_oracle_manifest.json`. The two source tarballs unpacked to
`c221c43559cbc09f6176b33d00412194762e2613` (arm A) and
`055707217c4a9820490488f596675d200996b03e` (arm B), which is the pinned pair.
**The two libraries hash DIFFERENTLY** -- `2af944ef...` and `571871a3...` -- so
this is two binaries and not one measured twice, which is what exit 54 exists to
refuse. Both CUDA unit gates read **23 cases / 806 assertions / 0 failed**, the
same counts `fa9903b86` recorded. Every render ran 8 steps and 32 DiT forwards,
and every phase table covered its own wall to better than 0.13%.

**The baseline reproduces #2296.** Arm A's wall is **516.751 s** at 1.74% spread
against #2296's 518.398 s, and the ratio is **5.51x** against its 5.53x. The five
nested sub-leaves cost nothing measurable, which is what `nested` predicted:
`unaccounted_seconds` is 0.599 s, 0.12% of wall.

### The split, which is this row's result

| | seconds (n=3) | spread | % of wall |
|---|---:|---:|---:|
| `conditioning.connector` | 122.167 | 1.68% | 23.64 |
| ~ `.compute` | **112.768** | **1.70%** | **21.82** |
| ~ `.weights` | 8.674 | 6.78% | 1.68 |
| `generate.guiders` | 189.496 | 1.07% | 36.67 |
| ~ `guiders.tower` | 71.548 | 0.81% | 13.85 |
| ~ `guiders.connector.compute` | **112.114** | **2.07%** | **21.70** |
| ~ `guiders.connector.weights` | 5.033 | 7.02% | 0.97 |

Both leaves decompose to a residue under 0.9 s, so this is a decomposition and
not a set of intervals.

**IT IS THE ARITHMETIC, NOT THE WEIGHTS.** Connector compute is **224.882 s,
43.52% of the render**. Materializing the connector -- all four times, both
passes -- is **13.707 s, 2.65%**. #2296's hypothesis, that the 122 s is loading
and widening 2.016 B parameters out of the 42 GB DiT file, is **FALSE as a time
claim**, and the way it went wrong is worth keeping: its 8.59 GiB memory step was
real and correctly observed. A memory step says the widen HAPPENS. It says
nothing about what the widen COSTS, and the widen costs 8.7 s because the DiT is
page-cached by then. The second materialization is cheaper than the first --
5.033 s against 8.674 s -- for the same reason.

**The row's own prediction, committed before the number existed, holds.**
`## Why the split has to come first` put the connector at about 4.2 TFLOP of f32
GEMM per call, "which at 122.388 s is 34 GFLOP/s". Measured: 4.2 TFLOP over
112.768 s = **37.2 GFLOP/s**. The estimate was recorded in advance precisely so
this sentence could be checked rather than asserted.

**The attribution rests on the quiet rows.** The two compute leaves are at 1.70%
and 2.07% spread. The rows that move -- `load.dit` at 61.07%,
`load.text_encoder` at 22.52%, `artifacts.frames.ppm` at 253.05% -- are
attributed to nothing.

### `generate.guiders` is no longer read by subtraction

#2296 measured the positive halves at 150.731 s against a `generate.guiders` of
190.016 s and wrote that the remaining 39.3 s was "a reading rather than a
measurement, because `generate.guiders` is ONE leaf". It is now two:
**`guiders.tower` is 71.548 s against `conditioning.tower`'s 28.287 s, 2.53x**,
both at sub-1% spread. The negative prompt's larger valid-token count costs
43.3 s more than the positive prompt's, and that is measured rather than
inferred.

### The repair: a measured near-negative, reported as one

| | arm A | arm B | delta |
|---|---:|---:|---:|
| `~guiders.connector.weights` | 5.033 s | **0.000 s** | -5.033 |
| `generate.guiders` | 189.496 s | 183.520 s | -5.976 |
| `~conditioning.connector.weights` | 8.674 s | 9.923 s | +1.250 |
| `~conditioning.connector.compute` | 112.768 s | 113.803 s | +1.035 |
| `~guiders.connector.compute` | 112.114 s | 112.160 s | +0.046 |
| **wall** | **516.751 s** | **510.956 s** | **-5.795 (-1.12%)** |

**The structural claim is established and the wall claim is NOT.** The second
materialization is gone -- the leaf reads exactly 0.000 s on all three renders,
and `generate.guiders` falls by 5.976 s, which matches the 5.033 s removed. That
is a directed, predicted, structural change and it is what T2 asserts.

**But -1.12% is INSIDE the same-arm spread of either arm** (A 1.74%, B 2.12%,
which are 9.0 s and 10.8 s). A wall delta smaller than the noise it sits in does
not establish a speedup, and this repository has already paid for calling one
that did: `1997-put-us-on-the-slow-topk-kernel`. So the honest statement is that
the wall is CONSISTENT with removing 5 s and does not on its own demonstrate it.
The two compute leaves are unchanged to within 1%, which is the correct control:
the repair was never supposed to touch them.

**The ceiling was known before the arm ran and is the point.** Once the split
read `weights` at 2.65% of wall, the maximum this repair could return was the
second materialization alone -- 5.033 s, 0.97%. It was still run, because a
predicted negative that is not measured is an opinion, and because it is the
control that proves the compute leaves do not move.

### Correctness, established before the speed result was accepted

- **Byte equality.** All 25 frames and the wav are byte-identical between the
  arms: `pixel_files_differing=0`, re-checked independently at 26/26, and the
  concatenated sha256 of each arm's frame set is `fb5bc236...` on both. The
  repair reuses an identical weight bag, so this is the prediction the spec made
  rather than a tolerance that was met.
- **The #1864 absolute-quality gate PASSES on both arms**, `compare_exit_A=0`
  and `compare_exit_B=0`, `VERDICT PASS`,
  `READING NO_WORSE_THAN_ORACLE_ON_BLOCKINESS`. `blockiness_grid8` 1.030110
  against the reference's per-frame maximum 1.143393 (margin +0.113283) and
  `blockiness_grid32` 1.024809 against 1.148672 (margin +0.123862), with 0 of
  1600 bands collapsing to the off-grid denominator on each grid.
- **The two arms' reference JSONs differ only in the arm LABEL.** Every metric
  value is identical, which is the cross-check that the blockiness gate and the
  byte comparison agree instead of one covering for the other.
- Both CUDA unit gates: 23 / 806 / 0.

**And the blockiness gate remains one-sided, which is why byte equality was run
beside it.** #1864 records our render as already less blocky, less sharp and less
clipped than upstream's, so a change that smoothed it further would PASS that
gate while moving every pixel. On this change the sharper instrument returned
equality, so the question does not arise -- but it would have.

### A claim this row made at n=1 and withdrew at n=2

`artifacts.frames.ppm` read 16.357 s on arm A render 1 -- larger than
`decode.video` and larger than the whole denoise loop -- and was reported as a
finding. Render 2 read **0.944 s**. Over six renders the leaf spreads 253.05% on
arm A and 114.04% on arm B. It is also not an engine cost: the harness renders
into `--workdir` under `/workspace`, which is CIFS, so the leaf measures network
write-back. **Withdrawn**, and recorded here rather than deleted, because
promoting an n=1 number one paragraph after writing "this is n=1" is the failure
this repository names and the correction belongs with its origin.

## Owed

- **THE COMPUTE LEVER IS UNOWNED, and it is 43.52% of the render.** 224.882 s of
  host f32 GEMM at ~37 GFLOP/s, in `Ltx2ConnectorForward` through
  `vt::MatmulBT`'s CPU arm. This row measured it and does not move it. Two
  traceable next steps, neither a ceiling: (1) determine whether `MatmulBT`
  reaches `MatmulOneChunk`'s specialized elementwise kernel on these shapes or
  falls back to `MatmulOneChunkRef`, which is a `VT_CPU_ELEM_GEMM`-class question
  answerable without a lease; (2) the two connector passes run the SAME weights
  over different inputs and could share one batched GEMM at `batch = 2`, which
  doubles arithmetic intensity and halves weight streaming, and is bit-identical
  because each output's K reduction stays sequential over K. The device arm is a
  third and much larger step: `Ltx2Attention` interleaves host `RmsNormRows` and
  `Ltx2ApplyRotaryEmb` on raw `float*` between its GEMMs, and
  `Ltx2ConnectorForward` reads weights as host `std::vector<float>`, so it is a
  weight-arm port and not a queue swap. **No issue was filed for this: the
  GitHub account was suspended mid-row (see below). Owner: this row, until an
  issue can be opened.**
- **`decode.audio.mel` is still unattributed** at 47.171 s, 9.13% of wall, at the
  tightest spread in the whole table (0.16%). It is 3.1x the entire denoise loop
  for 1.02 s of audio. This row confirmed the number and explains it no better
  than #2296 did. Owner: this row.
- **The AUDIO-ONLY arm still materializes twice.** `GenerateAudioOnly` is a
  private static declared in `include/vllm/multimodal/ltx2_video.h`, so it cannot
  take a type defined in `ltx2_video.cpp`'s anonymous namespace. It keeps
  `RunConnectorFromFile`. Owner: this row, through #2354.
- **The oracle side is still n = 1.** Its 93.8 s has no spread and this row did
  not re-run it, so both ratios above (A 5.51x, B 5.45x) are stated against a
  denominator whose own variance is unknown. Inherited from #2296 unchanged.
- **The harness writes renders onto CIFS.** That put network write-back inside
  `artifacts.frames.ppm` and is why the leaf is unusable. A later row should
  render into `/root` and copy out, which is what
  `stage-checkpoints-to-local-disk` already says for inputs. Owner: this row.

## The remote

**`REMOTE_UNVERIFIED` from 2026-08-30 onward.** The `mudler-agent` GitHub account
returned `Your account is suspended` (HTTP 403) on `git push` and on
`gh api user`, roughly forty minutes after `gh issue create` succeeded for #2354.
The branch is committed locally and unpushed and there is no pull request. The
PR body is written and staged at `/tmp/ltx25-text-cond-pr/body.md`. Nothing in
this outcome depends on the remote: the measurement ran on `dgx:gpu0` through
`rc` and its evidence is on the NAS.
