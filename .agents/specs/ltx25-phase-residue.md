# LTX25-PHASE-RESIDUE — the phase table's un-named time

Issues: [#1536](https://github.com/mudler/vllm.cpp/issues/1536),
[#1470](https://github.com/mudler/vllm.cpp/issues/1470),
[#1668](https://github.com/mudler/vllm.cpp/issues/1668).
Related and deliberately NOT closed by this row:
[#1439](https://github.com/mudler/vllm.cpp/issues/1439).
Owning row: LTX25-PHASE-RESIDUE.

Note on spelling: the row id is written WITHOUT backticks above, for the reason
`attention-rung-visibility.md` records — `check-agent-record.py::check_spec`
selects a row's governing spec by searching for the backticked token, and a
second file carrying it can change which document is held to the structured
contract.

**Read `## Now` first.** This spec is a RECORD of a body of work that was
measured, gate-run and reviewed three times, and then overtaken on `main` before
it landed. The measurements are why the file exists. The implementation is not
here, and `## Owed` says who owns each remaining piece.

## Scope

The `test_ltx2_video` phase table carries two wall-clock ratios:

```
CHECK( leaves >= 0.95 * wall )                       // the table sums to wall
CHECK( covered >= c.min_coverage * leaf_seconds )    // a leaf is covered by its parts
```

Both were red, and four issues argued about whether the tolerances were right:
#1439, #1470, #1494 and #1536, filed between 2026-08-20T05:23Z and
2026-08-21T04:40Z. That is under 24 hours, not the "three months" #1556's
spec recorded -- this repository's first commit is `accfae2de`, 2026-07-02, so
no dispute in it can be three months old.

This row's question was different: **where is the un-named time actually going**,
measured rather than argued, so that a phase gets a NAME instead of a threshold
getting a wider number.

IN SCOPE: naming the un-named regions of the LTX-2.5 render, and giving the
instrument a way to report its own cost, so a reader can subtract it before
calling a residue a phase nobody named.

OUT OF SCOPE, and stated because both were tempting: widening either floor, and
re-arguing the tolerance. A floor wide enough to absorb the residue would hide
the phase underneath it, and a floor below the real value is a mute switch.

## Our baseline

### Where the residue actually is, which nobody had measured

Splitting `unaccounted_seconds` into the gaps between consecutive leaves took one
pass over the table the render already writes. On the 64x64x9 fixture, against a
19.178 ms residue:

| gap | ms | share of the residue |
|---|---:|---:|
| `<origin>` -> `load.dit` | 17.661 | **92.09%** |
| `load.dit` -> `load.video_vae` | 0.950 | 4.95% |
| `load.prompt_embeds` -> `generate.setup` | 0.249 | 1.30% |
| `artifacts.audio` -> `WriteJson` | 0.210 | 1.09% |
| the other 16 gaps, together | 0.108 | 0.56% |

**92% of the un-named time is one region**: `Ltx2VideoEngine::Load` from the
timeline's origin to `Open("load.dit")` — the platform probe, the device
resolution, the recipe and checkpoint-class resolution, and
`SafetensorsFile::Open(params.dit_path)`, which on a 1775-tensor 21 B manifest is
not a free call. The sixteen gaps between adjacent named phases hold 6.8 us each,
which is the instrument and nothing else.

### What is NOT the cause, measured rather than assumed

#1536 asked for `d995c52f0`'s temporal x2 upsampler to be tested first. It runs
inside `phase.upsample_latent`, a named leaf, and **does not appear in the
residue at all**. That hypothesis is refuted rather than deprioritised.

### Where the coverage miss actually is

The same defect one level down. `denoise` misses by **49 us per step at nine
frames and 343 us per step at 81** — a 7x move with the latent, inside one run of
one binary, so it is WORK and not instrument overhead. It is the sampler's
post-process and Euler step, which the case's own comment already named and left
un-anchored while a threshold was tuned around it.

## Design, as measured

### 1. Name the regions

| anchor | what it covers | share of the residue |
|---|---|---:|
| the load prologue | timeline origin to `load.dit` | 92% |
| `load.dit_config` | the DiT config resolution, `load.dit` to `load.video_vae` | 5% |
| `artifacts.mux` | result assembly and mux argv build, after `artifacts.audio` | 1% |
| `denoise.update` | the sampler's post-process and step, nested inside `denoise` | the whole coverage miss |

Each leaf must CLOSE before the next opens. A leaf opened while another leaf is
live is marked `nested` and dropped from `sum_leaf_seconds` (`PhaseLog::Open`,
`src/vllm/multimodal/render_phase_log.cpp`), so leaving the prologue open across
the DiT load would take the largest phase of the whole render OUT of the sum
instead of adding the prologue to it.

`denoise.update` is the deliberate exception: it is nested on purpose, so the sum
does not move, and it is counted by a render-side counter rather than by the
phase table. `artifacts.mux` is named `mux` and not `finish` because
`phase.finish` already names something else — what a RECIPE PHASE does after its
sampler — and two names one letter apart for two different things is how a reader
mis-ranks a lever.

### 2. The instrument measures its own out-of-record cost

The residue contains a term the instrument creates and never reported: the wall
it spends inside its own entry points while no record — or no CHILD record — is
open. `PhaseLog::Open` stamps `o.start` AFTER taking the process-wide mutex, so
the mutex wait precedes the record. `PhaseLog::Close` stamps `r.end` BEFORE it
emits its progress line and erases the entry, so that tail follows the record.
Both land outside every record.

The rule is one sentence: **every interval of the instrument's own wall is
charged to the innermost live non-span record at the moment it is spent, and to
the table when none is live.** Spans are excluded because `Sum` excludes spans,
so time inside a span but outside a leaf is exactly the residue.

`WriteJson` should also take `Elapsed()` BEFORE it copies and sorts the record
vector rather than after, so the table's own serialization stops being charged to
the render's wall. It measures the render, not the writer.

### 3. The bound this row proposed, and why it is REJECTED

This is the row's most useful output and a large part of why this file is kept.

The plan was to replace both wall-clock ratios with `residue <= 2 * instrument`,
derived from the instrument's own measured cost so the gate would say the same
thing at 64x64x9 and at 3840x2160x241. **Three fresh reviews measured that and it
is wrong.** The un-instrumented remainder of a boundary dilates FASTER than the
instrumented part under contention, so the comparison has a heavy right tail:

| site | runs | red | median | max |
|---|---:|---:|---:|---:|
| the table bound, load 88 | 45 | **4 (8.9%)** | 1.132 | **4.115** |
| the conservation case, load 80 | 200 | **3** | 1.525 | 2.934 |
| the `unit.parent` case, load 85 | 200 | **2** | 1.70 | — |
| a standalone probe of that shape, load 125 | 160 | **28 (17.5%)** | — | 5.55 |

A 20-run distribution reading 1.021 to 1.464 was the body of the first of those
and saw NONE of its tail. That is itself the finding to carry forward: a 20-run
sample of this quantity does not see the tail that decides the gate.

`wall` in the denominator turns out to be the better-conditioned denominator,
because wall grows with contention exactly when a preemption inflates the
residue. **Both ratios stay. The proposed bound is withdrawn and its constant
deleted rather than raised.** Anyone reaching for an instrument-relative residue
bound on this table should read this section before measuring it again.

### 4. What that costs, stated rather than glossed

With the bound withdrawn, a future un-named region under 5% of wall is
invisible, and mutation D — the `denoise.update` anchor moved off the
post-process, 5 of 5 red against the withdrawn bound — is not detected. That is
[#1570](https://github.com/mudler/vllm.cpp/issues/1570).

## Superseded on `main`, and by what

**The load prologue landed under another row's name.** `519303d15`
([#1622](https://github.com/mudler/vllm.cpp/pull/1622), issue
[#1439](https://github.com/mudler/vllm.cpp/issues/1439), row
`LTX25-DEVICE-RESIDENCY`) opens `phase::Scope open_phase("load.open")`
immediately after the `load` span and calls `open_phase.Close()` immediately
before the `load.dit` block. This row's branch called the same region
`load.setup` and opened and closed it at the same two statements, with the same
`Scope::Close` shape and the same measured justification. **They are the same
repair, and `main`'s is the one that stands.** Compared by behaviour and by open
and close point rather than by name: `src/vllm/multimodal/ltx2_video.cpp:794` and
`:956` on `db648fb88`.

`6b48edb2c` separately repaired the `denoise` red on `main` by moving the share
floor to 0.75 and adding assertion (1c). Its own comment records that as a
holding action -- in substance rather than in those words, which the tree does not
use -- and the comment is still in the tree at
`tests/vllm/multimodal/test_ltx2_video.cpp:4324-4331`:

> NAMING THE UN-NAMED TIME WOULD SETTLE IT PROPERLY, which is what #1439 asks for
> first. A `denoise.update` scope over the sampler's per-step update would put the
> interior residue under a name and make a tight share floor honest again. ... It
> stays owed rather than being folded into the repair of a standing red.

So the coverage floor on `main` is a holding action by its own account, and
`denoise.update` is owed there in writing.

## Dependencies

None on other rows. `LTX25-DEVICE-RESIDENCY` already landed the load prologue, so
anything further in `ltx2_video.cpp`'s load path reads
[`ltx25-device-residency.md`](ltx25-device-residency.md) first, and the two rows
do not name the same region twice.

## Risks and decisions

**D1 — a wider floor is not a repair.** A slack big enough to stop either
assertion flapping exceeds the phase it would then hide. Naming measures the
phase; tolerating it does not. This is why every entry in `## Owed` is an anchor
or a bound on a scheduler-independent quantity, and none is a constant.

**D2 — the instrument's own cost is charged, never subtracted globally.** A
single global subtraction is a number nobody can attribute. Charging each
interval to the innermost live non-span record keeps the attribution local and
makes the conservation invariant testable.

**D3 — `denoise.update`'s denominator comes from the render, not the table.**
`Ltx2ConditioningTrace::sampler_updates` is a render-side counter for the reason
`video_decode_chunks` is: the containment gate's record-count assertion is the
only one there that is not a ratio, and a denominator derived from the phase
table could not falsify a phase table.

**D4 — this row does not close #1439.** Its RED is fixed on `main` by
`519303d15`. Its filed COMPLAINT is that the budget is a share of `wall`, so it
decides by box load and permits minutes of un-named time at 21 B. Nothing here
changes that, and `## Design` 3 is the measured evidence that the obvious
replacement does not work. What would close it is a bound on a quantity the
scheduler cannot move, which is
[#1570](https://github.com/mudler/vllm.cpp/issues/1570).

## Owed

Nothing in this table is implemented on `main`. Verified on `db648fb88` by a
tree-wide grep over `src/`, `include/`, `tests/` and `docs/`: `sampler_updates`,
`artifacts.mux` and `load.dit_config` return nothing, and `denoise.update` occurs
once, at `tests/vllm/multimodal/test_ltx2_video.cpp:4325`, asking for it.

| Issue | Owed |
|---|---|
| [#1668](https://github.com/mudler/vllm.cpp/issues/1668) | **the three anchors and the instrument self-cost, as one implementable unit.** `load.dit_config`, `artifacts.mux`, `denoise.update` plus `Ltx2ConditioningTrace::sampler_updates`, and `Record::instrument_seconds` with its conservation invariant. The reference implementation, the gate report and the mutation table stay readable at `refs/pull/1556/head` = `b45ea3bbb` |
| [#1567](https://github.com/mudler/vllm.cpp/issues/1567) | the res_2s arm's `denoise.update` anchor. `Ltx2Res2sDenoisingLoop` runs its own post-process and step behind `Ltx2Res2sHooks`, so the anchor needs a hook rather than a statement. It lives in `ltx2_samplers.cpp`, is declared in `ltx2_samplers.h` beside the hooks struct, and is called from `ltx2_video.cpp`. **NOT `ltx2_res2s.cpp`**: #1556's spec named that file and it has never existed here, which `git log --all --diff-filter=A` confirms; #1567's forge text names no file at all, so the wrong anchor came from the spec rather than from the issue. No gate in this tree renders on that arm, so landing it beside the first-order arm would land dead code |
| [#1568](https://github.com/mudler/vllm.cpp/issues/1568) | the `denoise.step` / `denoise.update` seconds transfer. (1b') compares `start_seconds` only, so leaving `denoise.step` open across the post-process and emitting `denoise.update` empty after it preserves the alternation, both counters, containment, non-overlap, exclusivity, (1c) and (2), while moving 100% of the decomposed seconds onto one name. No (2b) floor separates it: the honest share of `denoise.update` runs 0.45% to 11.15% across four boxes and a transfer puts it at ~0%. Closing it needs an anchor INSIDE the callee |
| [#1569](https://github.com/mudler/vllm.cpp/issues/1569) | a gate on `WriteJson`'s clock ORDERING, **measured green under its own mutation**. Restoring the old order left the conservation case GREEN 10 of 10, at `wall 0.0608987s, unaccounted 0.000534223s, table charge 0.000301655s`, because the copy and sort of a three-record table are nanoseconds. Gating it needs a table with enough records for the sort to be measurable |
| [#1570](https://github.com/mudler/vllm.cpp/issues/1570) | an upper bound on the instrument's own share of a leaf. `uncovered <= 2 * leaf_instrument` is stricter than the floor it replaces only while `leaf_instrument` stays small, and nothing bounds it. Moving the DiT `Tick` out of `Evaluate` would charge ~110 flushed writes to `denoise` and widen the gate while printing a small number |
| [#1571](https://github.com/mudler/vllm.cpp/issues/1571) | a per-gap decomposition IN the emitted table. The 92% region above was found with a scratch script; a reader of `phase-log.json` still cannot see it without one, and the same investigation will be re-derived the next time the residue moves |
| [#1572](https://github.com/mudler/vllm.cpp/issues/1572) | assertion (1c)'s span slack reds intermittently on `main` — `decode.video` at `0.00256913` against a `0.00075` bound, 3.4x. Pre-existing from `6b48edb2c` and not this row's |
| [#1619](https://github.com/mudler/vllm.cpp/issues/1619) | **the `merge=union` driver duplicates a row, MEASURED on this row's own merges.** Both sides appended before the same trailing anchor rather than at the true end, so the driver concatenated two regions that each carried `#1546` and the resolved index held it TWICE, byte-identical, at 538 lines where the correct union is 537. `git merge-tree` called that merge clean and `check-issue-index-append-only.py` passed it, because a duplicate is an ADDITION and that checker only collects removals. `check-agent-record.py` did NOT pass it -- a claim #1556's spec made and this row REFUTED by reproduction: regenerating the raw driver output and running that same tree's checker returns rc=1 with `issue #1546 listed twice`, and the refusal has existed since `8dd6508da` (2026-08-09), before the merge. So the blind gate is exactly one checker, not two, and the gap is narrower than #1556 recorded. The de-duplication half is CONDITIONAL, and the condition is what #1556's spec omitted: the checker reds a repair only when the DUPLICATE IS ALREADY IN THE BASE. Measured at three pairings -- `--base e2a9e035d` against the real canonical 537-line file rc=0, against a synthetic 537 rc=0, and `--base <committed 538> --head <537 de-dup>` rc=1. It diffs `merge-base..HEAD`, so when the base predates the duplicate the addition and the removal CANCEL and it passes. Since `origin/main` is preflight's base, and is the shape this branch used, the gate does NOT red someone who repairs driver output before committing it -- only someone repairing a corruption that already landed. The same range property is why relocating a base-reachable row DOES red it: moving row `#168` to the end gives rc=1 and a `removed:` line naming it. So "de-duplicating in place FAILS the checker", as #1556's spec put it, is false unqualified and true once the duplicate is base-reachable. #1556's spec added that the same driver dropped `#838` on a later re-merge, making this a recurring class; that is WITHDRAWN as unreproducible. Re-running `git merge-file --union` at every later merge where `#838` was on a side leaves it present in all of them, and `git log -S` finds it absent from no committed state -- mechanically a union driver cannot drop a line that is an addition on one side. If it ever went missing, that points at a wholesale take-ours resolution rather than at the driver |
| [#1439](https://github.com/mudler/vllm.cpp/issues/1439) | **NOT closed by this row, and it must not be.** See `## Risks and decisions` D4 |
| [#1470](https://github.com/mudler/vllm.cpp/issues/1470) | `test_ltx2_video` false-redded once on `main` under load and the failing case's identity was never captured. Untouched here |

## Stop conditions

- Do not widen either ratio to close a red. Name the phase, or leave the red and
  file the gap. D1.
- Do not re-propose `residue <= 2 * instrument` without reading `## Design` 3
  first, and never accept a 20-run distribution as evidence about it.
- Do not close #1439 from this row. D4.

## Now

**This row's implementation is NOT on `main`, and #1556 is closed rather than
merged.** The pull request was measured, gate-run and through three fresh
reviews, and while it was in flight `519303d15` (#1622) landed the same load
prologue repair under a different name. The pull request body — which
`squash_merge_commit_message = PR_BODY` makes the permanent commit message —
argued at length for `load.setup` as new work, so merging it would have written a
materially false narrative onto `main` irreversibly. The branch also carried its
own withdrawn bound, a `build-newest-gcc` repair the tree had already received
twice, and an `## Outcome` section for an outcome that did not happen. On that
repair, one detail #1556's body got wrong and this record does not repeat:
`13548db8f` (#1581) is the `build-newest-gcc` fix, while `d27639e71` is
`BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (#1476/#1595) and merely added the same
`<unistd.h>` in passing. The duplicate include was real; only one of the two
commits was about the lane.

Closed on 2026-08-22 with the record above. The reference implementation, its
full gate report, its mutation table and the three review threads remain readable
at `refs/pull/1556/head` = `b45ea3bbb`, cited from
[#1668](https://github.com/mudler/vllm.cpp/issues/1668).

Nothing is blocked. Both reds this row was filed against are green on `main`: the
sum floor by `519303d15`, which is the right repair, and the coverage floor by
`6b48edb2c`'s 0.75, which is a holding action by its own account and is why
`denoise.update` stays owed under #1668. Stated narrowly on purpose: that is the
two floors this row was filed against, and it is NOT a claim that
`test_ltx2_video` is quiet -- `## Owed` #1572 records assertion (1c)'s span slack
redding intermittently on `main`, which this row neither causes nor repairs.

**LTX25-PHASE-RESIDUE has no matrix row, so it has no lifecycle state**, and
`scripts/now.py` and `audit-live-rows` will never surface it. That is deliberate
rather than an omission: with the implementation unlanded there is nothing to
give a state to, and creating a row would put an empty one in the runnable
population. The forward owner is [#1668](https://github.com/mudler/vllm.cpp/issues/1668),
and whoever picks it up creates the row with the implementation.
