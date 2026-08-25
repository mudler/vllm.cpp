# `LTX25-TEST-DETERMINISM` — what decides whether an assertion in `test_ltx2_video` runs

Issues: [#1885](https://github.com/mudler/vllm.cpp/issues/1885) (the assertion
count), [#1439](https://github.com/mudler/vllm.cpp/issues/1439) (the sum floor),
[#1536](https://github.com/mudler/vllm.cpp/issues/1536) (the coverage floors).

Records this row reads rather than re-derives:
[`ltx25-phase-residue.md`](ltx25-phase-residue.md),
[`ltx25-phase-instrument.md`](ltx25-phase-instrument.md),
[`ltx25-device-residency.md`](ltx25-device-residency.md).

This row has **no matrix row and therefore no lifecycle state**, for the reason
`ltx25-phase-residue.md` records for itself: the phase log is an instrument
inside the LTX-2.5 driver and no matrix in this tree keys instruments. Its state
is the issues it closes and the tests that hold it.

## Scope

**No row is appended to `.agents/issue-index.md` for #1885, and the obligation is
already met.** The index carries a row appended by the row that FILED the issue,
naming `LTX25-DISTILLED-LORA-REQUIRED`, and
[`ltx25-distilled-lora-required.md`](ltx25-distilled-lora-required.md) is on
`main` (`813e9ff97`) with #1885 listed under its `## Owed`. So that row satisfies
`AGENTS.md`'s requirement in both of the ways it allows — it names an owning row
AND the spec lists the issue — and nothing about it is broken.

A second row would not add to that, and `scripts/check-agent-record.py` refuses
one outright, because under `merge=union` a duplicate issue number is exactly
what two branches appending the same issue look like. An earlier draft of this
row appended one, was refused by that gate, and the index was restored to its
`ced0ab639` bytes. The work moved to this spec while the index row stayed as
written, which is what an append-only file means.

IN SCOPE: one property of `tests/vllm/multimodal/test_ltx2_video.cpp` — **which
assertions execute must be decided by the tree, never by the clock.**

#1885 reports that the suite "reports a different assertion count on every run of
the same binary", measured as 4337-4343 over seven runs of two binaries at
`5c789015e`. The count is quoted as evidence in this repository's records, so a
count that moves on its own makes every count comparison across a diff
meaningless and hides a case that silently stopped asserting.

#1439 and #1536 are in scope as VERDICTS only. Both are about the two wall-clock
ratios this suite carries, both were reported red, and both have had repairs land
since. This row runs them and records what they do now. It does not move either
constant.

OUT OF SCOPE, and each is named because each was tempting:

- **Widening or narrowing either ratio.** `leaves >= 0.95 * wall` and
  `covered >= c.min_coverage * leaf_seconds` are untouched here, in both their
  form and their constants. `ltx25-phase-residue.md` `## Design` 3 records three
  fresh reviews that measured a replacement bound and refuted it, and #1668's
  direction is that both ratios STAY.
- **The other LTX suites' published counts.** #1885 asks whether
  `docs/FEATURES.md`'s `test_ltx2_dfr` 652 and `test_ltx2_tiling` 915 are
  reproducible. This row measures `test_ltx2_video` and says so; the other two
  are listed under `## Owed`.

## Upstream chain

None, and this is the rare row where that is a finding rather than a gap. vLLM
has no LTX-2.5 phase instrument, no phase table and no equivalent test, so
nothing upstream defines how many assertions a case should run.
`.agents/oracles/vllm.md` therefore contributes no anchor here. What replaces an
upstream anchor is the repository's own rule, which this file quotes because it
is the whole of the argument: a case that returns early on an unmet precondition
is a skip wearing a pass.

## Our baseline

Measured on this row's own binary — one build, one host, back-to-back runs. The
host is shared and its loadavg is recorded beside every number, because this lane
has published load-contaminated numbers before.

Tree `ced0ab639`, Release, `-DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_BUILD_EXAMPLES=ON -DVLLM_CPP_SERVER=ON`, gcc, x86-64, 20 cores. One
binary, `md5 02616ab2e36dc0b67d3903cd54976ffd`, for every run. A second agent was
running its own copy of this suite on the same box throughout, and the loadavg
column says so.

### 1. The count moves, and this is the run it moved on

Seven back-to-back runs of that one binary. Six completed. Five of the six
reported **4719** assertions with byte-identical per-case totals; the sixth
reported **4721**. The seventh aborted (`### 4`) and reported 4637, which is 4719
minus the 82 assertions the aborted case never reached.

| run | assertions | loadavg at start |
|---|---:|---|
| 1 | 4637 (aborted) | 35.75 21.21 10.25 |
| 2 | 4719 | 69.78 66.19 45.47 |
| 3 | 4719 | 44.39 56.42 44.96 |
| 4 | 4719 | 41.30 50.98 44.87 |
| 5 | 4719 (1 assertion failed, `### 3`) | 44.75 50.15 45.26 |
| 6 | 4719 | 42.34 49.10 45.56 |
| 7 | **4721** | 66.23 61.37 51.79 |

The per-case diff of run 7 against every other completed run is ONE line:

```
103c103
< 796  ltx2 video: the three carrying phases contain their work and the load keeps its order
---
> 798  ltx2 video: the three carrying phases contain their work and the load keeps its order
```

So the moving quantity is not spread over the suite. It is one case, and #1885's
"different count on every run" is that case moving by two.

### 2. One assertion in this suite is executed or skipped by the CLOCK

`CheckCarryingPhase`'s span-slack loop runs one `CHECK` per leaf record, and it
runs it only where `min(kSpanSlackPerRecord, 0.5 * record_seconds)` equals
`kSpanSlackPerRecord` — that is, only where the record is at least 60 ms long.
A record shorter than that is skipped with a `MESSAGE` and no assertion. The
comparison is between a constant and a WALL-CLOCK DURATION, so which branch a
record takes is decided by the box.

The margin is not comfortable, and it is measured on both sides:

| quantity | value |
|---|---|
| longest SKIPPED record over the runs here | 26.3 ms (`artifacts.frames`) |
| the resolution threshold | 60 ms |
| `artifacts.frames` over both renders, `ltx25-device-residency.md` | 0.90 ms to **61.0 ms** |

The record that is 26.3 ms on a quiet run has been measured at 61.0 ms on this
project's own hosts, which is on the other side of the threshold. The case says
so itself, and this is the whole of the diagnosis: run 2 reports
`0 of 2 leaf record(s) checked` on two of its leaves and run 7 reports `1 of 2`
on the same two. Two records crossed 60 ms, two more `CHECK`s ran, and 796 became
798.

Isolated on the one case, it takes three runs of ONE binary to see, and it is
verbatim:

```
probe 1 rc=0 asserts=798 fails=0 split=9/17 load=92.63 88.86 76.80
probe 2 rc=0 asserts=798 fails=0 split=9/17 load=103.05 91.62 78.04
probe 3 rc=0 asserts=796 fails=0 split=7/17 load=106.12 92.65 78.52
```

`split` is resolvable records over total records, and `798 - 796 = 9 - 7`. That
is the red this row repairs. After the repair the same probe on the same case
reads 813 six times over, while `split` still reads 7/17 and 9/17 — the box still
decides which records the bound can RESOLVE, and that no longer decides what gets
asserted.

### 3. What the two verdict issues do now

**#1439 is still red, 1 run in 7, and the decomposition names the region.**

| run | wall (s) | leaves (s) | residue (s) | of which instrument | largest gap | ratio |
|---|---:|---:|---:|---:|---|---:|
| 1 | 0.444732 | 0.444240 | 0.000491 | 0.000370 | `load.prompt_embeds -> generate.setup` | 99.889% |
| 2 | 0.491515 | 0.491214 | 0.000301 | 0.000218 | same | 99.939% |
| 3 | 1.154000 | 1.153400 | 0.000598 | 0.000448 | same | 99.948% |
| 4 | 1.226080 | 1.225650 | 0.000437 | 0.000315 | same | 99.965% |
| 5 | 0.375373 | 0.355197 | **0.020177** | **0.020091** | **`artifacts.mux -> <end>` 0.019832** | **94.625% RED** |
| 6 | 0.174765 | 0.174150 | 0.000615 | — | same | 99.648% |
| 7 | 1.414330 | 1.413550 | 0.000776 | — | same | 99.945% |

Run 5 is inside the 0.22-0.58 s fixture regime #1439 identified, and it fails at
`0.355197 >= 0.356605`. Six runs clear the floor by 4.6 to 5.0 points.

What is new is not the red. It is that the table now says what the red IS. The
whole of run 5's residue lies in one gap — the render's tail, after the last named
leaf — and **the instrument charges 99.6% of it to itself**. That is
`PhaseLog::Close`'s teardown, which notifies and joins the sampler thread and is
measured at about 117 us when the box is quiet. It is not a phase nobody named.
It is the instrument, already named, already reported, inside a wall the floor
divides by.

**And the quiet-box run this issue was owed.**
`ltx25-device-residency.md` records the outstanding obligation on #1439 as one
quiet-box run at fixture scale, in the 0.22-0.58 s regime, because the earlier
repair was measured where the wall is 7-10 s and everything sits near 99.96%.
Six runs of the case alone at **loadavg 10.23 to 14.14**, the quietest this host
was all day:

| run | wall (s) | leaves (s) | residue (ms) | ratio |
|---|---:|---:|---:|---:|
| 1 | 1.887960 | 1.887080 | 0.880 | 99.953% |
| 2 | 0.835112 | 0.834288 | 0.824 | 99.901% |
| 3 | 0.510394 | 0.509935 | 0.459 | 99.910% |
| 4 | 0.464973 | 0.464484 | 0.489 | 99.895% |
| 5 | 0.351199 | 0.350755 | 0.444 | 99.874% |
| 6 | 0.551330 | 0.550489 | 0.841 | 99.847% |

Four are inside the regime and all six clear the floor by 4.85 to 5.0 points. The
largest gap is `load.prompt_embeds -> generate.setup` in every one, at 0.17 to
0.51 ms. So the BIAS this issue was filed on — 4.80% to 6.32% of wall in all
thirteen runs — is gone, and what remains is a contention TAIL on one interval.
The obligation is discharged; the issue is not, because the tail still reds.

**This row does not repair it, and the reason is recorded rather than assumed.**
Every repair that suggests itself here is a change to the ratio — dividing by
`wall - instrument_seconds`, or bounding the residue against the instrument's own
charge. `ltx25-phase-residue.md` `## Design` 3 measured the second of those and
three fresh reviews refuted it, and #1668's recorded direction is that both
ratios STAY. So this is a measurement filed against #1439 and not a fix.

**#1536's coverage floors do not reproduce, in 7 runs of 7.**
`covered >= c.min_coverage * leaf_seconds` passed on every leaf of every run.
Worst coverage in each run: 96.5%, 96.7%, 96.5%, 96.8%, 96.9%, 97.4%, 96.2%,
against floors of 0.50 and 0.75 — a margin of 21 points at the worst observation.

#1536's own numbers were 0.18% to 0.8% SHORT, of a floor that was 0.95 when it
was filed. That floor is 0.75 now (`6b48edb2c`) and the phases under it are
named, so the quantity the issue measured is 21 points clear of the number it is
compared against. What #1536 also names is the sum floor, and that one is still
red — but it is #1439's, and #1536 says so itself.

### 4. A workspace under `/tmp` that another process deletes

Run 1 aborted mid-case with `ltx-2.5 video: cannot write
/tmp/vllm_ltx2_video_970765_3/multichunk/audio.wav` — an exception, not an
assertion failure, so doctest recorded 89 assertions for a case that runs 171 and
the run's total fell by 82 with `failures="0"`.

The fixture root is `"/tmp/vllm_ltx2_video_" + getpid() + "_" + counter`
(`test_ltx2_video.cpp:67`) — world-readable, prefix-predictable, and outside any
per-run private directory. A sentinel directory planted at
`/tmp/vllm_ltx2_video_999999_0` was **gone within two minutes**, and a second
sentinel planted four minutes later was still there twenty minutes on. So the
deletion is episodic rather than periodic: `systemd-tmpfiles-clean` last ran 19
hours before and next runs in four, and nothing in this repository globs that
path, while a second agent was running its own copy of this suite from
`/home/mudler/_git/vllm.cpp/.wt/1426` throughout.

What that means for the suite is the same either way: any tidy-up that matches
the prefix removes a LIVE run's output directory, and the render then fails at
its next write. It is not the assertion-count defect and it is not fixed here,
because the right repair is a decision about where this fixture writes rather
than a guess — a random suffix does not help against a prefix glob. It is filed as
[#1906](https://github.com/mudler/vllm.cpp/issues/1906), and listed under
`## Owed`.

## Port map

| Concern | Where it lives |
|---|---|
| the timing-decided assertion | `tests/vllm/multimodal/test_ltx2_video.cpp`, `CheckCarryingPhase`, the span-slack loop |
| the resolution constant | the same function, `kSpanSlackPerRecord` |
| the two ratios under verdict | the same file, the `SUMS to wall` case and `CheckCarryingPhase` (2) |

Nothing under `src/` or `include/` changes.

## Tests to port

None. There is no upstream test for this behaviour, and the change is inside a
test rather than under it.

## Gates

```sh
cmake --build build --target test_ltx2_video -j 10
./build/tests/test_ltx2_video -r=xml            # N consecutive runs, one binary
```

The gate is the ONE property this row adds: N consecutive runs of one binary
report one assertion total, with N and the loadavg stated. A count that moves is
a failure of this row whatever the pass/fail says.

## Dependencies

None. The change is confined to one test translation unit.

## Work breakdown

One unit: name the site, make the count structural, prove it by mutation, and
record the two verdicts.

## Risks/decisions

**The aggregate is not a weaker assertion, and this is the one claim a reviewer
should mutate.** Replacing one `CHECK` per resolvable record with one `CHECK`
over the worst of them is an identity, not a relaxation: every resolvable record
carries the same bound `kSpanSlackPerRecord`, so `max(slack) <= bound` and
`for each: slack <= bound` are the same statement. The identity holds only
because the cap `min(kSpanSlackPerRecord, 0.5 * record_seconds)` is what SELECTS
the resolvable set rather than what bounds it, and the mutation table proves a
violating record still reds.

**A partition assertion is added, not removed, and it is narrower than it first
looks.** `span_checked + span_unresolvable == leaves.size()` holds that no record
leaves the loop by a THIRD path, with no clock in the comparison. It does NOT
distinguish a leaf that resolved everything from one that resolved nothing, since
`0 + N == N` — the first draft of this row claimed otherwise in a code comment and
a fresh review refuted it by mutation. That escape is unchanged from the
per-record form and is listed under `## Owed`.

## Design

One change, in `CheckCarryingPhase`.

The per-record `CHECK(span_slack <= span_bound)` moves out of the loop and
becomes one `CHECK` per leaf over the worst slack among the records the bound can
RESOLVE. Every resolvable record carries the same bound — `span_bound ==
kSpanSlackPerRecord` is what "resolvable" MEANS — so `max(slack) <= bound` and
"each slack <= bound" are the same statement, and the assertion count per leaf
stops being a function of how many records happened to exceed 60 ms.

Beside it, one more `CHECK`: the resolvable and unresolvable counts must add up
to the leaf's record count. The skip used to leave no trace in any assertion,
which is the shape this repository calls a skip wearing a pass. It is now
arithmetic with no clock in it, and it fails if an edit ever makes the loop drop
a record on some third path.

Neither the constant nor the cap moves. The `REQUIRE` on the formula stays where
it is, per record, because its count is the RECORD count and that is structural.

## Evidence

Green, the varying case alone, fixed binary `md5
3d387c9787d8b044758cb568f83e34ca`, six runs back to back:

```
probe 1 rc=0 asserts=813 fails=0 split=7/17 load=111.30 97.14 81.11
probe 2 rc=0 asserts=813 fails=0 split=7/17 load=111.30 97.14 81.11
probe 3 rc=0 asserts=813 fails=0 split=7/17 load=112.32 97.58 81.35
probe 4 rc=0 asserts=813 fails=0 split=7/17 load=114.13 98.21 81.63
probe 5 rc=0 asserts=813 fails=0 split=9/17 load=113.72 98.39 81.78
probe 6 rc=0 asserts=813 fails=0 split=9/17 load=113.02 98.50 81.91
```

The split still moves. The count does not. Full suite on the same binary, four
consecutive runs at loadavg 100 down to 36: 4736, 4736, 4736, 4736. On the merged
head after the review repairs, three more: 4736, 4736, 4736, and `All gates
green.` from `scripts/agent-preflight.sh`.

### Mutations

| # | mutation | anchor unique | built | expected | observed |
|---|---|---|---|---|---|
| M1 | `ltx2_video.cpp` — 40 ms sleep after `Scope denoise_phase("denoise")`, so the leaf opens before its work | 1 | yes | the aggregate CHECK reds | RED, 3 failures, `record 1 of 'denoise' spends 0.0401004s ... over a bound of 0.03s`; total still 813 |
| M2 | `test_ltx2_video.cpp` — `++span_unresolvable;` deleted | 1 | yes | the partition CHECK reds | RED, 6 failures, `the span-slack loop accounted for 0 of the 'decode.video' leaf's 2 records` |
| M3a | the fresh review's: `<` to `<=` on the resolution test, forcing every record unresolvable | 1 | yes | expose vacuity | GREEN at 813, every split `0 of N` — the escape now under `## Owed` |
| M3b | the fresh review's: M1 and M3a together, a real 40 ms swallow with nothing resolvable | both | yes | does it escape | GREEN — it escapes, as it did before this change |

The tree was restored byte-for-byte after each, and the rebuilt binary returns to
its pre-mutation md5 on both sides.

### What the fresh review added that re-running would not have

It reconciled BOTH pre-repair totals to one post-repair number rather than
re-measuring them: the change removes 7-9 clock-gated checks and adds 24 fixed
ones over 12 `CheckCarryingPhase` calls, so `4719 + 17 = 4736` and
`4721 + 15 = 4736`. Two different starting counts landing on the same number is a
stronger check on the baseline than a repeat run.

## Now

Landed as PR #1913 against `main`. #1885 is what this row repairs. #1439 stays
OPEN with the measurement in `### 3`. #1536 is CLOSED on the evidence in `### 3`.
#1906 is filed and owed.

## Owed

- [#1906](https://github.com/mudler/vllm.cpp/issues/1906) — the fixture's `/tmp`
  workspace is deleted by another process mid-run, which aborts a case where it
  stands and costs 82 assertions with `failures="0"`. Measured here (`### 4`),
  filed rather than fixed, because where this fixture should write is a decision
  and a random suffix under the same prefix does not survive a prefix glob.
- **A leaf that resolves NO record is not checked at all, and the aggregate
  passes on nothing.** A fresh review of this row forced every record below the
  60 ms resolution and drove a genuine 40 ms swallow on `denoise` straight
  through — green, at the same 813 assertions. This is not a regression: the
  per-record form ran zero assertions on such a leaf too, and 3 to 5 of the 12
  aggregate checks passed on nothing in every run measured here. It cannot be
  closed by asserting that a record resolved, because that is a statement about a
  wall-clock duration and would red on a fast box for being fast. What closes it
  is a bound that does not need 60 ms of record to resolve — `### Owed out of W0`
  in [`ltx25-device-residency.md`](ltx25-device-residency.md) already carries the
  same escape as a ceiling, and names an anchor inside the short records as the
  fix rather than a wider bound.
- Whether `test_ltx2_dfr` (652) and `test_ltx2_tiling` (915) reproduce their
  published counts. #1885 asks and this row does not answer it.
- #1439 stays open. `### 3` measures it and names the region; the repair is a
  change to a ratio that #1668 recorded as OUT of bounds.
