# SPEC — `LTX25-RENDER-SPEED-PARITY`: where an LTX-2.5 render's wall actually goes

Issue: [#2296](https://github.com/mudler/vllm.cpp/issues/2296).
Owner row: `LTX25-RENDER-SPEED-PARITY`.

## Scope

`LTX25-ORACLE-ABSOLUTE` took the correctness reading at `fa9903b86` — `rc` job
`4b0666ee-248c-45fc-9de6-372b6d0c1fab`, `VERDICT PASS` on both reference forms.
`AGENTS.md` `## Gates` opens the speed axis on exactly that condition:
"Correctness always comes first. Establish the declared token-exact gate before
you accept a performance result." This row is the first speed reading of an
LTX-2.5 render against the pinned oracle on an identical request.

IN scope:

- The like-for-like comparison of our render against
  `ltx2_oracle_manifest.json`'s `render_seconds` and `total_seconds`, with each
  quoted figure labelled by what it brackets.
- A phase decomposition of our render, with `n > 1` at the manifest's request
  and the spread stated per phase.
- Attribution of the gap, and the next traceable hypothesis.

OUT of scope, declared rather than approximated:

- **Any repair.** This row measures. The levers it names are owed to other rows
  and to [#1269](https://github.com/mudler/vllm.cpp/issues/1269), which already
  owns the CPU-pinned text tower.
- **A published benchmark ID.** `docs/BENCHMARKS.md`'s surface takes a benchmark
  when one is added, removed, or changes disposition. One request, one geometry,
  one seed and a handful of runs is a reading, not a benchmark, and publishing an
  ID on it would be the `a-number-quoted-often-becomes-treated-as-measured`
  failure with this row as the origin.
- **The oracle's own decomposition.** Upstream emits no phase table and this row
  does not instrument it. Every statement about where the oracle's 93.8 s goes
  would be a guess, so none is made.

## The oracle's number, and what it actually brackets

This is the row's first finding, because the comparison was called invalid on a
misreading and the misreading is the easy one to repeat.

`tools/oracle/ltx2_oracle.py:242-244`:

```python
t0 = time.time()
rc = subprocess.call(argv)
render_s = time.time() - t0
```

`argv` (`:227-241`) is the complete `python -m ltx_pipelines.ti2vid_one_stage`
CLI with all four checkpoint paths on it. That subprocess imports torch, opens
the four checkpoints, runs the text tower, denoises, decodes both VAEs and muxes
the mp4. **`render_seconds = 93.8` therefore already includes the oracle's model
load.** It is not "the render alone".

`started` (`:207`) sits above `assert_identity`, so `total_seconds = 243.7` adds
the sha256 of 70,099,185,228 bytes and the ffmpeg decode to 25 PPM frames.
`.agents/oracles/ltx-2.md` records the same split independently — "Render 93.8 s;
whole run 243.7 s, of which 149 s was hashing the four checkpoints" — and the
arithmetic closes: 149 + 93.8 = 242.8 against 243.7.

**So the comparable pair is `[G]`-wall against `render_seconds`,** both sides one
process, both including load, both reading checkpoints off local disk on the same
GB10. `total_seconds` has no counterpart on our side at all: our harness's
staging and sha256 are separate phases and are not inside the render.

**There is no load-excluded form of this comparison, and that is a property of
the oracle rather than a gap in the instrument.** Upstream ran `--offload cpu`,
which streams the tower and the DiT layer by layer, so its load is interleaved
with its compute and has no boundary to subtract. Quoting our render-minus-load
against its 93.8 s would compare our excluded load against its included one.
Where this row states a load-excluded figure it is stated as a **bound**: our
`generate` span against the oracle's whole process, which is an under-statement
of the ratio because the oracle's load is on its side of the fraction.

## Our baseline

`phase-log.json`, shipped on the default path since ABI v23
([#1010](https://github.com/mudler/vllm.cpp/issues/1010)), is the instrument.
`ltx2-gen` writes it beside the frames and
`examples/ltx2_gen/main.cpp:615-619` prints its path. The `4b0666ee` render's
table is on the NAS at
`/mnt/nas_share/rc/ltx25-oracle-absolute/run/20260828T234052Z/ours/phase-log.json`
and reads, at 320x192/25f/8 steps:

| phase | seconds | share |
|---|---:|---:|
| `generate.guiders` | 187.73 | 37.7% |
| `conditioning.connector` | 120.25 | 24.1% |
| `load` | 80.25 | 16.1% |
| `decode.audio` | 50.63 | 10.2% |
| `conditioning.tower` | 27.58 | 5.5% |
| `decode.video` | 15.98 | 3.2% |
| `denoise` | 14.90 | 3.0% |
| `wall_seconds` | 498.60 | |

`unaccounted_seconds` is 0.432 s, so the named phases cover 99.91% of the wall.
That coverage is what makes the table a decomposition rather than a set of
intervals: nothing large is in the residue.

**That is n = 1 at this geometry**, and the file's own `notice` field refuses to
be read as more: "On a contended box the same binary at the same geometry has
moved by more than an order of magnitude in wall, and the RANK of its two largest
phases has reversed between two such runs." This row's measurement exists to
turn n = 1 into n > 1 before anything is attributed.

## Why the answer is not the DiT

`denoise` is 14.90 s of 498.60 — **3.0%**. Every LTX-2.5 speed row in this tree
so far has worked on the transformer. At the oracle's own request the transformer
is not the term.

The two dominant phases are one defect counted twice.
`src/vllm/multimodal/ltx2_video.cpp:2343-2377` runs the positive prompt through
the Gemma-4 tower and then `RunConnector`; `:3073-3094` runs the negative prompt
through both again, because classifier-free guidance needs the unconditional
conditioning (`guiders.py:275-277`, mirrored at `:3027-3031`). Two towers and two
connectors per request. The positive half is `conditioning.tower` +
`conditioning.connector` = 147.83 s; the negative half is the whole of
`generate.guiders` = 187.73 s, the difference being the negative prompt's larger
valid-token count.

All four passes run on the host. `ltx2_video.cpp:2344` and `:3074` each build
`vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}`, and the
comment at `:2323-2331` states it as a limit rather than an oversight: the tower
is f32 by declaration and "the text path has no device arm to run on".
[#1269](https://github.com/mudler/vllm.cpp/issues/1269) owns that for the tower.
**It does not carry the connector**, which is the larger half at 120.25 s against
the tower's 27.58 s, nor that either runs twice.

## The fixed serial term, measured across fifteen further renders

The NAS carries fifteen other `phase-log.json` files from this campaign, at
320x192/25f through 768x448/49f, across four attention arms and five dates. Read
for the two phases above:

| phase | range over the 12 comparable runs | moves with geometry? |
|---|---|---|
| `generate.guiders` | 185.53 - 194.04 s | no |
| `conditioning.connector` | 116.76 - 127.04 s | no |

(The four `ltx25-ad/20260821` runs sit at 229-241 s and 159-169 s on a different
prompt set and are excluded from the range rather than averaged into it, because
a different prompt is a different valid-token count and that is the term.)

Neither phase moves with the video geometry, because neither depends on it: they
are text-side work whose size is set by the prompt. They are a **fixed serial
cost**, and at the oracle's small request they are 61.8% of the wall.
`component-speedup-is-not-system-speedup-fixed-serial-term` is the exact shape,
and it is why the decomposition matters more than the total: a lever on the DiT
cannot reach 97% of this render.

## The next traceable hypothesis, named rather than a ceiling

`AGENTS.md`: "Never declare a ceiling. An apparent same-architecture performance
limit is an unresolved implementation difference." This row does not repair
anything, so what it owes instead is the next step somebody can take.

`RunConnector` (`src/vllm/multimodal/ltx2_video.cpp:524-566`) opens with

```cpp
const Ltx2VaeWeights video_weights = Ltx2LoadConnectorWeights(dit_file, video_cfg);
const Ltx2VaeWeights audio_weights = Ltx2LoadConnectorWeights(dit_file, audio_cfg);
```

and `Ltx2LoadConnectorWeights`
(`src/vllm/model_executor/models/ltx2_loader.cpp:1515-1573`) parses the whole
DiT plan with `PlanDit(file)` and then materializes every connector tensor and
**widens it from bf16 to f32 into a `std::vector<float>`**, per call. Nothing is
cached. `RunConnector` runs twice per render — `:2372` positive, `:3091`
negative — so one render performs **four `PlanDit` parses of a 4091-tensor
header and four full f32 materializations of the embeddings connector**, whose
audio half alone is 2.016 B parameters (`.agents/roadmap_v1.md`, geometry read
from the checkpoint header).

**The phase log's memory column corroborates this from an independent
direction.** `conditioning.tower` peaks at 33.46 GiB host and
`conditioning.connector` at 42.05 GiB — a step of 8.59 GiB across the phase
boundary, against 2.016 B parameters widened to f32 = 8.06 GB. A load-and-widen
hypothesis predicts that step; a pure-compute one does not.

So the hypothesis is: **`conditioning.connector` is dominated by loading and
widening the connector out of the 42 GB DiT file rather than by the connector's
arithmetic, and the render pays for it four times.** What settles it is one
sub-phase split — `PlanDit`, `MaterializeDitTensor` and the widen loop timed
separately inside `conditioning.connector` — which is the same instrument this
row already reads and needs no new one. If it holds, caching one
`Ltx2VaeWeights` across the two conditioning passes removes half of it without
touching a kernel. That is a row of its own and it is not this one.

**`generate.guiders` is mostly the same load, a second time.** It is the negative
prompt through both stages, so it should read as one tower pass plus one
connector pass. Measured at n = 3: the positive halves are 28.343 + 122.388 =
150.731 s and `generate.guiders` is 190.016 s, leaving **39.3 s unexplained**.
The plausible reading is the negative prompt's larger valid-token count in the
tower — upstream's `DEFAULT_NEGATIVE_PROMPT` is longer than this request's
prompt — and it is a reading rather than a measurement, because
`generate.guiders` is ONE leaf and its tower and connector halves are not
separately scoped. Splitting it is the same one-line instrument change the
paragraph above asks for, and it should be done in the same row.

**`decode.audio.mel` at 47 s has no hypothesis, and saying so is better than
inventing one.** It is larger than the whole denoise loop. It is under
`## Owed`.

## The record this falsifies

`.agents/specs/ltx25-device-residency.md` lists `generate.guiders` beside
`generate.setup`, `generate.geometry` and `phase.finish` as "unanchored, and each
under 0.001% of the leaf sum on the fixture", owned by "nobody, today". The
measurement is honest and names its own scope in the same sentence — a two-block
fixture — and on the 21B model with a real negative prompt the same phase is the
**largest in the render**. A later reader following that row would skip the
biggest phase. The correction is recorded here rather than by editing that row,
and #2296 carries it.

## Port map

| File | Change |
|---|---|
| `.agents/specs/ltx25-render-speed-parity.md` | this file |
| `scripts/ltx25-render-speed-repeat.sh` | new; the repeat-render timing harness `## Gates` item 1 runs |
| `.agents/issue-index.md` | two appended rows: #2296, and #2305 which this harness's own defect owed |

**No product code.** This row measures an engine it does not change. That is
deliberate and it is the reason the harness asserts the binary's sha256 against
`4b0666ee`'s: the reading has to be about the tree that took the correctness
verdict, not about a rebuild of it.

## Tests to port

There is no upstream test for this. Upstream renders; it does not decompose. The
harness's own preconditions are its tests, and each is an executable refusal
rather than a comment:

| ID | Assertion | Exit |
|---|---|---|
| H1 | the cached binary's sha256 equals `4b0666ee`'s `7b1f4367...`, and the library's equals `9e3dc6f4...` | 51 |
| H2 | all four checkpoint sha256 equal the manifest's | 23 |
| H3 | the CUDA unit gate passes before any timing | 44 / 45 |
| H4 | every render writes exactly 25 frames, a non-empty wav and a `phase-log.json` | 48 |
| H5 | every run's `unaccounted_seconds / wall_seconds` is under 1%, so a table that stopped covering the render is refused rather than summed | 52 |
| H6 | `steps_observed` is `{8}` and `dit_forwards` is 32 on every run, so a run that silently changed the schedule is not averaged in | 53 |

H1 is the identity assertion `oracle-identity-must-be-asserted` asks for, applied
to our own side: a harness that rebuilds and then reports a decomposition has
measured a binary nobody verified.

H5 exists because `sum_leaf_seconds / wall_seconds` is the one ratio the phase
log's own `notice` says the table supports. A render whose residue grew is a
render with a phase nobody named, and averaging it into a decomposition would put
that unnamed time into whichever phase happened to be adjacent.

## Gates

1. **The measurement.** `scripts/ltx25-render-speed-repeat.sh` under an `rc`
   lease on `dgx:gpu0`: `N = 3` renders at `ltx2_oracle_manifest.json`'s exact
   request, on the `fa9903b86` binary, phase tables collected, spread reported
   per phase.
2. `scripts/agent-preflight.sh`.

Gate 1 is the only one that needs a device. There is no unit gate here because
this row lands no product code and no checker.

## Dependencies

- `dgx:gpu0` through `rc`. Never `ssh`; `AGENTS.md` `## Work on a GPU happens
  inside a lease`.
- `$W/absref-bin` at `SRC_SHA = 0002ddfba26b59279732aeb4e3c99e092b436f28`, the
  build cache `LTX25-ORACLE-ABSOLUTE` left on the NAS. If it is gone, this row
  refuses rather than rebuilding, because a rebuilt binary is a different
  measurement subject and H1 is what says so.
- The four BF16 checkpoints already on the NAS, digests in the manifest.

## Work breakdown

- **W1** — this spec, committed before the harness.
- **W2** — the harness.
- **W3** — the lease, the reading, and `## Outcome`.

## Risks/decisions

- **`n = 3` is small, and it is stated rather than dressed up.** Three renders
  bound the spread; they do not establish a distribution. What makes the
  attribution safe is not n = 3 on its own but that the two phases being
  attributed are also visible in fifteen independent runs at other geometries,
  which is a much wider base than this row's own lease.
- **The SM clock cannot be pinned inside a lease** (`nvidia-smi -lgc` returns
  `LGC_RC=4`, measured 2026-08-19, `.agents/benchmarking.md`). It is sampled and
  recorded. It also matters less here than usual and the reason is measured
  rather than assumed: 61.8% of this render is host-side f32 work on a CPU
  queue, so **host** load is the contention axis that transfers, and the harness
  records `loadavg`, `MemAvailable` and `nvidia-smi` per run for that reason.
- **The lease is the mutex and nothing else is.** No `flock` beside it. #777 and
  the 2026-08-17 double-mutex incident are what that rule is for.
- **One request, one geometry, one seed, bf16, `n = 3`.** The row claims nothing
  outside that, and `docs/BENCHMARKS.md` gains no ID.

## Evidence

- The `rc` job id, lease wall-clock, and the queue and contention state around it.
- Binary and library sha256 against `4b0666ee`'s.
- The four checkpoint sha256 against the manifest.
- Three `phase-log.json` files, the per-phase spread, and the coverage ratio per
  run.
- Our `[G]` wall and our `generate` span, each labelled, beside the oracle's
  93.8 s and 243.7 s and beside what each of those brackets.

## Stop conditions

Stop and report, do not work around:

- a checkpoint sha256 that does not match the manifest;
- a binary or library sha256 that does not match `4b0666ee`'s;
- an unhealthy or unreachable fleet device — clearing one is a human's call;
- any spread that makes the attribution unsupportable. Saying so is this row's
  deliverable; narrowing the phases until it holds is not.

## Owed

- **The repair is not here.** `conditioning.connector` and `generate.guiders` are
  measured and unowned by any row that could move them.
  [#1269](https://github.com/mudler/vllm.cpp/issues/1269) owns the tower's CPU
  pinning and is the closest existing owner; it does not carry the connector, and
  it does not carry that all four passes are two duplicated halves. Owner of the
  gap record: this row, through #2296.
- **The oracle's own 93.8 s is undecomposed.** Upstream emits no phase table and
  this row did not instrument it, so which part of the 5.4x is load and which is
  compute is known on our side only. Owner: this row.
- **`decode.audio.mel` is unattributed.** Measured at n = 3 it is inside
  `decode.audio`'s 50.745 s, which is 9.79% of the wall and **3.35x the whole
  denoise loop**, for 1.02 s of audio. Nothing in this tree explains it and this
  row does not guess. Owner: this row.
- **The 206.0 s that remains after `guiders` and `connector` is a second
  question and it is open.** Removing both entirely still leaves 2.20x. `load`
  is 94.5 s of it, `decode.audio` 50.7 s, `decode.video` 16.0 s and `denoise`
  15.1 s. Naming it here is what stops the connector finding from being read as
  the whole answer. Owner: this row.
- **The oracle side is n = 1.** Its 93.8 s has no spread and this row did not
  re-run it, because doing so needs the pinned `ltx-2` stack in a lease and that
  is a second measurement rather than a footnote to this one. Every ratio above
  is therefore stated against a denominator whose own variance is unknown.
  Owner: this row.

## Now

`ACTIVE`. W1, W2 and W3 are in this change and complete. The reading is taken.

## Outcome

**The lease.** `rc` job `0baa109c-43ff-475e-abf1-7a50152ffd5d` on `dgx:gpu0`,
2026-08-29 19:58:51Z to 20:26:45Z, **27m54s**, `renders_completed=3`. The device
was `ready` and unleased at submission and no other job held it; `rc ps` showed
an empty queue. Evidence at
`/mnt/nas_share/rc/ltx25-render-speed/run/20260829T195851Z`, which a leased
worker sees as `/workspace/ltx25-render-speed/run/20260829T195851Z`.

**Every precondition held, and each was checked rather than assumed.** The
binary is byte-for-byte `4b0666ee`'s — `ltx2-gen` `7b1f4367...`, `libvllm.so.0.0.3`
`9e3dc6f4...`, source `0002ddfba` — so this is a timing of the tree that took the
correctness verdict and not of a rebuild of it. All four checkpoint sha256
recomputed inside the lease and all four match the manifest. The CUDA unit gate
ran first at **23 cases / 806 assertions / 0 failed**, the same counts
`fa9903b86` recorded. Every render read `steps_observed={8}` and
`dit_forwards=32`, and every phase table covered its own wall to better than
0.12%.

**The like-for-like comparison. Both sides are one process, both include model
load, both read local disk on the same GB10.**

| | brackets | seconds |
|---|---|---:|
| ours, `[G]` wall, n = 3 | the whole `ltx2-gen` invocation | 509 / 531 / 528, mean **522.7** |
| ours, `phase-log.json` `wall_seconds`, n = 3 | first phase open to last phase close | 504.865 / 526.778 / 523.551, mean **518.398** |
| oracle, `render_seconds`, n = 1 | the whole `ti2vid_one_stage` subprocess, load included | **93.8** |
| oracle, `total_seconds`, n = 1 | the above plus 149 s of sha256 and the ffmpeg decode | 243.7 |

**Ratio 5.53x** on the in-process walls (5.38x to 5.62x across our three runs),
**5.57x** on the `[G]` walls. The oracle side is n = 1 and has no spread of its
own, and that is a limit of this comparison rather than a footnote: our
4.23% spread is measured and theirs is unknown.

`total_seconds` has no counterpart on our side and is not compared. Its 149 s of
hashing is a separate phase of our harness, outside the render.

**The load-excluded figure is a BOUND, not a ratio.** Our `generate` span means
423.913 s against the oracle's whole 93.8 s = **4.52x**, and that is a LOWER
bound on the render-only ratio because the oracle's 93.8 s still has its own load
inside it. There is no better form available: `--offload cpu` streams upstream's
weights layer by layer, so its load has no boundary to subtract.

**The decomposition, n = 3, spread stated per phase.** Every boundary here is a
`phase::Scope` in `src/vllm/multimodal/ltx2_video.cpp` writing
`phase-log.json` on the shipped default path — an instrument, not an interval
read off a log.

| phase | r1 | r2 | r3 | mean | spread | % of wall |
|---|---:|---:|---:|---:|---:|---:|
| `wall_seconds` | 504.865 | 526.778 | 523.551 | 518.398 | 4.23% | 100.00 |
| `generate` (span) | 421.733 | 424.300 | 425.707 | 423.913 | 0.94% | 81.77 |
| **`generate.guiders`** | 189.041 | 189.836 | 191.172 | **190.016** | **1.12%** | **36.65** |
| `generate.conditioning` (span) | 149.985 | 151.456 | 151.127 | 150.856 | 0.97% | 29.10 |
| **`conditioning.connector`** | 122.173 | 122.717 | 122.275 | **122.388** | **0.44%** | **23.61** |
| `load` (span) | 83.129 | 102.477 | 97.843 | 94.483 | 20.48% | 18.23 |
| `load.text_encoder` | 55.082 | 55.687 | 57.699 | 56.156 | 4.66% | 10.83 |
| `decode.audio` | 50.703 | 50.709 | 50.822 | 50.745 | 0.23% | 9.79 |
| `load.dit` | 25.792 | 43.208 | 36.128 | 35.043 | 49.70% | 6.76 |
| `conditioning.tower` | 27.697 | 28.603 | 28.729 | 28.343 | 3.64% | 5.47 |
| `decode.video` | 15.818 | 15.958 | 16.133 | 15.970 | 1.98% | 3.08 |
| **`denoise`** | 14.902 | 15.255 | 15.229 | **15.129** | **2.33%** | **2.92** |
| `unaccounted_seconds` | 0.481 | 0.593 | 0.600 | 0.558 | — | 0.11 |

**The spread is what makes the attribution safe, and it is the right way round.**
The two phases this row attributes the gap to are the two most stable numbers in
the table — 1.12% and 0.44% — while the phase that moves, `load.dit` at 49.70%,
is the one whose cost is page cache and is attributed to nothing. A finding
resting on the noisy row would be the
`1997-put-us-on-the-slow-topk-kernel` failure; this one rests on the quiet ones.

**Where the render's time goes.**

- `generate.guiders` + `conditioning.connector` = **312.4 s, 60.3% of our wall**,
  and **3.33x the oracle's entire process**. Both would still be there if the
  denoise were free.
- Text conditioning end to end — tower, connector, and the negative pass through
  both — is **340.7 s, 65.7%**.
- **`denoise` is 2.92%.** The 21B transformer, at the oracle's own request, is
  16.1% of the oracle's whole render on our side. It is not the term.
- Even with `guiders` and `connector` removed entirely, 206.0 s remains and the
  ratio would still be **2.20x**. This row names no ceiling and this is not one:
  it is the statement that the remainder is a second, separate question.

**An independent instrument agrees, and it is not the phase log.** The clock
sampler recorded `utilization.gpu` beside the SM clock. Folded through the
helper's own `build_clock_record`, the three windows excluded **663 of 757, 446
of 512 and 227 of 257** samples as idle: **the GPU is idle for 87 to 88% of the
render**, median utilization 0%. `nvidia-smi` and `phase-log.json` share no code
and agree that this render is host-bound.

**The clock, recorded because a number is quotable only with it.** GB10, driver
`580.173.02`, persistence `Enabled`, `throttle_reasons_active = 0x0`,
`clocks.max.sm` 3003 MHz, applications graphics 2418 MHz. Busy-sample SM clock
medians 2418 / 2411 / 2437 MHz at spreads of 4.30% / 4.02% / 3.98%, inside
`.agents/benchmarking.md`'s 5% ceiling and with the three medians within 1.1% of
each other. Host contention: `MemAvailable` 115.0 GiB throughout, `loadavg` 1.44
before the first render and 7.6 to 8.1 across the rest, no other GPU compute app
for the whole job.

**A defect this row's own harness shipped with, found by execution and fixed in
the same flow: [#2305](https://github.com/mudler/vllm.cpp/issues/2305).** The
clock window was opened as `( ... ) &` and stopped with `kill -TERM "$!"`, but
`$!` is the SUBSHELL's pid, so the sampler was orphaned and the helper never
wrote its `--summary` — it writes one only when it stops. The observable is
arithmetic: three renders at `--interval 2` should give about 255 samples each,
and the files hold **757, 512 and 257**, so render 1's window spans all three
renders and only render 3's describes the render it names. This is
`.agents/benchmarking.md`'s #1657 from the other end — a window that outlives its
arm. Fixed with `exec`, so the subshell is replaced by the sampler and the signal
reaches it, plus an explicit `WARNING` when a summary is absent, because the
failure mode was silence. **No number above depends on it**: the raw sample
streams were written correctly by every sampler and are what the fold above
reads. The one field the fold cannot supply is `boot_id`, and no cross-boot
comparison was made.

**What this row does not claim.** One request, one geometry, one seed, bf16,
n = 3 on our side and n = 1 on the oracle's. No public benchmark ID. No repair.
The oracle's 93.8 s is undecomposed, so which part of the 5.53x is its load and
which is its compute is known on our side only.
