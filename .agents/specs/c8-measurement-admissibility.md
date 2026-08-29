# BENCH-C8-ADMISSIBILITY — a c=8 reading is admissible only from the committed harness, and only with instance variance known

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding) — measurement
methodology, not product code.
**Issue:** [#2152](https://github.com/mudler/vllm.cpp/issues/2152).
**Related:** [#2154](https://github.com/mudler/vllm.cpp/issues/2154) (the defect
whose severity varies 0.0%-79.6% across runs of one binary),
[#545](https://github.com/mudler/vllm.cpp/issues/545) (this host's MTBF is
shorter than one measurement), [#2108](https://github.com/mudler/vllm.cpp/issues/2108)
(nothing in CI decodes on a GPU).
**Kind:** spec first, implementation to follow in a separate pull request,
because the implementation needs a working GPU and this does not.

## Now

`ACTIVE` — spec only. No implementation lands with it.

## CORRECTION: this spec named the WRONG committed harness

As written below, this spec routes the c=8 ladder through
`tools/bench/dflash2_speed_harness.py`. **That harness measures a different
axis.** It is a thin client of `examples/cli` (`vllm-cli`) and drives ONE
PROCESS PER PROMPT — single-stream draft speed, the axis recorded at 0.8017x by
`bae0392dd`. It has no notion of concurrency at all.

The c=8 comparison against vLLM and SGLang is a CONCURRENT SERVING axis, and the
committed instrument for it is **`scripts/dgx-online-serving.sh`**:

- `:627` — `concurrency_points="1 2 4 8 16 32"`, which is the ladder itself.
- `:1209` — `--execute` is "a PURE TIMED production grid: model gate +
  INTERLEAVED timed" runs.
- `:5-7` — "Timed requests are issued only by pinned vLLM `bench serve`; this
  script owns server lifecycle, interleaving, the one-model/one-lock boundary,
  memory return, and artifact capture."

So the ad-hoc `bisect2.sh` was not merely bypassing a harness; it was
reimplementing the grid, the interleaving, the lease boundary and the artifact
capture that this script already owns.

### And the serving driver cannot express THIS workload

Naming the right axis is not the same as having an instrument for it.
`scripts/dgx-online-serving.sh` and `tools/bench/online_gate.py` contain **zero**
occurrences of `speculative`, `dflash` or `draft`, and the driver's `--model` is
closed to `27 | 27n | 35 | q3mxfp4` (`:117-118`). The server it launches
(`:336-346`) passes `--num-blocks`, `--max-num-seqs` and
`--max-num-batched-tokens` and no draft path or speculative config at all.

So the committed concurrency instrument **cannot drive the DFlash2 workload**,
and that — not carelessness alone — is why `bisect2.sh` was written.

**The actionable ask is therefore neither of the first two.** It is:

> Extend `dgx-online-serving.sh` / `online_gate.py` with a speculative arm — a
> draft path, a `--speculative-config`, and a model id for the
> Qwen3.8-27B-NVFP4 + DFlash2 pairing — so the c=8 ladder can run on the
> committed instrument instead of beside it. Then retire `bisect2.sh` by making
> it unnecessary rather than by deleting it.

That is a wave, not a config change, and it is the real content of this row.

**Everything below stands except the harness name.** The refusals
`dflash2_speed_harness.py` carries — `--repeat 1` is "an anecdote", equal repeat
counts across arms, the warm-leg discard, oracle identity, clock state — are the
right rules and they are why the ad-hoc readings are inadmissible. They are also
mostly present in the serving driver, which takes its clock windows through the
same `tools/bench/gpu_clock_state.py`. The correction is WHICH committed tool a
c=8 reading must go through, not whether it must go through one.

This is the fourth instance in one session of the repository already holding a
discipline that was reimplemented beside it: `repeat_reasons` refusing n=1, the
cross-boot refusal in `gpu_clock_state`, the interleaving contract, and now the
concurrency grid itself. The failure is not missing discipline; it is not
looking for it first.

## The defect

**Every c=8 number this repository has quoted was taken outside the committed
harness, and the committed harness would have refused it.**

`tools/bench/dflash2_speed_harness.py` refuses `--repeat 1`:

```python
from tools.bench.dflash2_speed_harness import repeat_reasons
repeat_reasons(1, label="ours")
# ['repeat: the ours arm would run --repeat 1. Run 1 carries the first graph
#   capture and is discarded on both arms, so this leaves no warm leg;
#   a single leg is an anecdote']
```

It also enforces both arms repeating the SAME number of times ("two medians
folded by two different rules is not a ratio"), folds a median over warm legs
through one shared `fold_legs`, discards run 1, reads the resolved attention
backend back off the built engine, asserts oracle identity via the `+g<sha>`
local version segment, and records clock state through
`tools/bench/gpu_clock_state.py`. `scripts/dgx-online-serving.sh` names
"interleaving" in its own contract.

The measurements were taken instead by an ad-hoc `bisect2.sh` living only on
`dgx.casa` under `/usr/local/vcpp`: not in the tree, n=1 per arm, no clock
record, no warm-leg discard, no interleave. That is a parallel path around a
shared seam, which AGENTS.md forbids for product code and which nothing forbids
for measurement — and an out-of-tree harness cannot be reviewed, mutated, or
re-run by anyone else.

## What today established, and what it cost

On 2026-08-28, roughly twenty A/B legs were run on this rung. Two of four
sequences self-invalidated on their own terminal control. Seven mechanisms were
proposed and refuted. One correct product change was reverted on an
uncontrolled comparison and had to be restored (`037ca63eb`).

Zero-draft-block rates across thirteen runs of essentially one binary:

```
0.0  0.0  0.0  8.6  8.9  41.6  48.6  48.7  49.5  67.9  78.5  79.0  79.6
```

**Three of three first-legs-after-a-rebuild read 0.0%**, and every later leg in
those sequences degraded. Every leg started a FRESH server.

## The question this spec exists to answer first

**Is the defect's severity fixed at server startup, or does it vary per
benchmark pass?**

If it is fixed at startup, each A/B leg is a Bernoulli draw on instance health
rather than a measurement of the change under test, and **every throughput A/B
in #2154 is void** — including the one whose terminal control matched. No
further arm-vs-arm comparison on this rung means anything until this is known.

The test needs no code change: one server, three benchmark passes against it,
repeated across at least three instances, with the zero-draft-block rate
computed per pass from the log delta. Within-instance spread small and
between-instance spread large settles it.

It was written and running on 2026-08-28 when `dgx.casa` lost contact with the
controller mid-run, and returned nothing (#545).

## Scope

IN, in this order:

1. **Instance-vs-pass variance**, measured as above. Everything else is
   conditional on the answer. Run it through `scripts/dgx-online-serving.sh`,
   not beside it — see the correction above — and persist each leg with
   `tools/bench/resumable_legs.py`, because three attempts at this measurement
   were killed mid-run by host crashes (#545).
2. **A repeat count DERIVED from the measured spread** rather than assumed. The
   5.9% figure quoted across this repository came from a four-run study that
   sampled one stable window; it bounds that window, not the rung.
3. **The interleave and the terminal control inside the harness**, so a drifting
   box invalidates its own run instead of returning a confident number. Two of
   today's four sequences would have been caught automatically.
4. **An acceptance floor.** No test here asserts a lower bound on acceptance,
   which is why a defect worth 15-35% of c=8 throughput sits under a green
   board — a zero-acceptance step still emits the correct token, so token gates
   cannot see it. The refusal logic must run in CPU CI with no GPU, the polarity
   `gpu_clock_state.py` and the rest of the harness already chose.

OUT: any fix to #2154 or #2155. This spec is about what makes a reading
admissible, not about what the readings say.

## What this does NOT change

It does not retire `bisect2.sh` by deleting it — that script holds the recipe
the harness has to absorb. It is retired by the harness being able to do what it
does, and by readings from it becoming inadmissible.

## Gates

- `repeat_reasons`, the acceptance floor and the terminal-control check each
  have a red-before test in `tests/tools/test_dflash2_speed_harness.py`, run in
  CPU CI.
- A reading produced outside the committed harness is not evidence. This is a
  review obligation, not a checker: no checker can see where a number came from.

## Owed

- **The instrument cannot gate itself on a box that dies mid-run.** #545 is
  unresolved and bounded this work twice today. A harness that survives a host
  reboot mid-sequence, or that refuses a sequence spanning one, is owed and is
  not scoped here.
- **Nothing in CI decodes on a GPU** (#2108), so none of this runs anywhere but
  a lease.

## Stop conditions

Return `NEEDS_DECISION` if the instance-variance test shows BOTH spreads large.
That would mean the rung has a per-pass component this spec's design does not
address, and the repeat count in item 2 cannot be derived until it is
characterised.

## Outcome

Filled in when the row reaches `DONE`.
