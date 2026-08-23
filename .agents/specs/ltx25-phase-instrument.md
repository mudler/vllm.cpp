# `LTX25-PHASE-INSTRUMENT` — the phase table measures its own cost, and says where its residue is

Issues: [#1668](https://github.com/mudler/vllm.cpp/issues/1668) (item 4 of four),
[#1569](https://github.com/mudler/vllm.cpp/issues/1569),
[#1571](https://github.com/mudler/vllm.cpp/issues/1571).
Record this row implements from:
[`ltx25-phase-residue.md`](ltx25-phase-residue.md).
Related and deliberately NOT closed here:
[#1439](https://github.com/mudler/vllm.cpp/issues/1439),
[#1567](https://github.com/mudler/vllm.cpp/issues/1567),
[#1568](https://github.com/mudler/vllm.cpp/issues/1568),
[#1570](https://github.com/mudler/vllm.cpp/issues/1570).

This row has **no matrix row and therefore no lifecycle state**, for the reason
`ltx25-phase-residue.md` records for itself: the phase log is an instrument
inside the LTX-2.5 driver, and no matrix in this tree keys instruments. Its state
is the issues it closes and the tests that hold it.

## Scope

[`ltx25-phase-residue.md`](ltx25-phase-residue.md) is a RECORD of work that was
measured, gate-run and reviewed three times, and then closed unmerged. This row
lands the part of it that is about the INSTRUMENT rather than about the LTX-2.5
driver, and it lands the two gaps that record filed against itself.

IN SCOPE:

1. **`Record::instrument_seconds` and `PhaseLog::Instrument()`** — the
   instrument charging its own out-of-record wall to the innermost live non-span
   record, and to the table when none is live. #1668 item 4.
2. **`WriteJson` reads its clock before it serialises**, plus the gate that
   holds it. #1569, and #1668 item 4's second half.
3. **The residue decomposed into the gaps between adjacent leaves**, in the
   emitted file. #1571.
4. **`RenderText` reads its clock before it serialises, and the console block
   is emitted before this writer does any work at all**, plus the gate that
   holds both only in the two thirds `### 10` measures, with
   [#1760](https://github.com/mudler/vllm.cpp/issues/1760) owning the third it
   does not. [#1755](https://github.com/mudler/vllm.cpp/issues/1755), found
   by the fresh review of this row and fixed in the same flow. `### 10`.

OUT OF SCOPE, and each is named because each was tempting:

- **The three driver anchors** — `load.dit_config`, `artifacts.mux`,
  `denoise.update` and `Ltx2ConditioningTrace::sampler_updates`. They are #1668
  items 1 to 3, they touch `ltx2_video.cpp` and the `Carrying` table in
  `test_ltx2_video.cpp`, and #1568 and #1570 both need `denoise.update` to exist
  before they can be closed. They are one unit and this is not it. They land in
  the follow-on row, which is why #1668 stays open here.
- **Widening either floor.** `leaves >= 0.95 * wall` and
  `covered >= min_coverage * leaf_seconds` are untouched by this row, in both
  their form and their constants.
- **Any bound with `instrument_seconds` in a denominator.** See `## Design` 5.

## Our baseline

Everything measured here is already measured, in
[`ltx25-phase-residue.md`](ltx25-phase-residue.md) `## Our baseline`, and is not
re-derived. What that record establishes and this row builds on:

- 92% of a 19.178 ms residue on the 64x64x9 fixture is ONE gap, the load's
  prologue. The sixteen gaps between adjacent named phases hold 6.8 us each.
- The residue does NOT scale with wall — about 1 ms across walls of 0.8 s to
  4.6 s, and 0.82 to 86 ms across 10 s to 120 s. A share-based bound is therefore
  worst at the SMALLEST wall.
- `residue <= 2 * instrument` was measured across three fresh reviews and
  WITHDRAWN. It is not re-proposed. `## Design` 4 states what replaced it.

What is NEW here is one measurement, and it is the one #1569 asks for: what the
copy-and-sort inside `WriteJson` costs, and how far the two clock orderings are
apart once the table is large enough for that copy and that sort to exist. It is
in `## Evidence`.

## Design

### 1. The instrument charges its own wall

`PhaseLog::Open` stamps `o.start` AFTER taking the process-wide mutex, so the
mutex wait precedes the record. `PhaseLog::Close` stamps `r.end` BEFORE it emits
its progress line and erases the entry, so that tail follows the record. Both
land outside every record, and until now nothing could tell them from a phase
nobody named.

The rule is one sentence: **every interval of the instrument's own wall is
charged to the innermost live non-span record at the moment it is spent, and to
the table when none is live.** Spans are excluded because `Sum` excludes spans,
so time inside a span but outside a leaf is exactly the residue; charging it to
the enclosing `load` or `generate` span would hide it in a number nothing adds
up.

Ported from `refs/pull/1556/head` = `b45ea3bbb`, which measured, gate-ran and
three-times-reviewed this mechanism. The port is behaviour-identical. What is
NOT ported is that branch's `load.setup` anchor, which `519303d15` already landed
on `main` as `load.open`, and its withdrawn bound, whose constant is deleted
rather than raised.

### 2. `WriteJson` reads its clock first

`Sum(records, Elapsed())` after `ByStart(Records())` charges the WRITER's copy
and sort to the RENDER's wall, and therefore to `unaccounted_seconds`. The clock
read moves to the first statement of the function.

### 3. The residue is decomposed in the file

The leaves `Sum` adds are non-overlapping — `Open` marks a leaf `nested`
whenever another leaf is live — and `ByStart` orders them. So the complement of
their union inside `[0, wall]` is exactly `wall - sum_leaf_seconds`. The emitter
writes that complement as `gaps`: one interval before each leaf, one after the
last, each carrying the two names it lies between.

**This is the row's best gate, and the reason is that it is not a measurement.**
The gaps add to `unaccounted_seconds` by construction. A gate over that sum is
arithmetic over numbers already in the file, so no box load can move its verdict.
Every other assertion this table has ever carried was a ratio of two wall-clock
quantities, and two of them spent three months being argued about.

### 4. The charges to one record are DISJOINT, so the conservation invariant is one

`instrument_seconds <= duration_seconds` was asserted from the first version of
this row and it was **not an invariant**. `Open` and `Tick` read their clock
BEFORE taking the process-wide mutex, so the interval they charge to a record
spans a window in which another thread, holding that mutex, charges the SAME
record. Both charges are individually correct and they OVERLAP, and the sum of
overlapping intervals is not bounded by the interval that contains them. A fresh
review drove the ratio to **1.914** with 24 threads inside one live leaf, red in
3 runs of 5.

`ChargeLocked` now clamps each charge to the end of the last one that reached
the same target, and seeds that mark with the record's own `start`. Every
charged interval then lies inside `[start, end]` and no two of them overlap, so
the sum is at most the duration **by construction**. Reproduced and gated: the
same shape now measures 0.09 to 0.996 over 45 runs and the mutation that removes
the clamp measures 4.0, 20.3 and 22.0.

It can under-count, and that direction is deliberate: two charges arriving out
of order lose the earlier one. Under-counting is safe here only because this
quantity is in no denominator anywhere, which `## Design` 5 below is the reason
for. If it ever enters one, this clamp becomes a defect.

### 5. What replaces the withdrawn bound, and what does not

**Nothing in this row puts `instrument_seconds` in a denominator.** That is the
single most important sentence here, and [`ltx25-phase-residue.md`](ltx25-phase-residue.md)
`## Design` 3 is the evidence: the un-instrumented remainder of a boundary
dilates FASTER than the instrumented part under contention, so a residue measured
against the instrument's own charge has a heavy right tail — 4 red in 45 runs at
load 88 with a maximum of 4.115, and 28 in 160 at load 125 reaching 5.55.

`instrument_seconds` is therefore emitted and REPORTED, never asserted against.
A reader subtracts it before calling a residue a phase nobody named. The two
floors keep `wall` and `leaf_seconds` in their denominators, which is the better
conditioning: those grow with contention exactly when a preemption inflates the
numerator.

### 6. The one new bound, and how it is derived

#1569 needs a gate, and a gate needs a comparison. The comparison is
`head < 0.5 * serialize`, where both quantities are measured in the same run:

- `head` is `wall_seconds` as the writer recorded it, minus the elapsed clock the
  test read immediately before calling the writer. Under the correct ordering it
  contains one function call and one uncontended mutex — the instrument's own
  resolution. Under the mutated ordering it contains one whole copy of the record
  vector and one whole `stable_sort` of it.
- `serialize` is that same copy and that same sort, performed by the test through
  the same public `Records()`, on the same data, on this box, in this run.

So the constant is not a tolerance. Under the correct order the head holds ZERO
copies and ZERO sorts; under any wrong order it holds at least ONE of the two in
full, so it is at least `1.0 * min(copy, sort)` **by the definition of the
quantities**. Any constant strictly inside `(0, 1)` separates them. 0.5 is the
midpoint.

**THE FIRST VERSION OF THIS PARAGRAPH WAS FALSE AND A FRESH REVIEW MEASURED IT.**
It bounded the head against `copy + sort` TOGETHER, on the argument that the
mutated head contains one of each. That holds only when both move. The reviewer
hoisted `Records()` above the clock read and left `ByStart` below it -- the
natural shape of a partial regression, and the exact edit somebody makes while
moving one line -- and the bound stayed GREEN at a ratio of 0.0588, because the
copy is about 6% of copy-plus-sort. Against `min(copy, sort)` that same mutation
is red -- **1.035** on the review's own run and **1.024** on the full re-run
against the repaired tree. The constant did not move; the quantity under it got
smaller.

**The estimator is a MINIMUM over K probes, and that is what makes this not the
withdrawn bound wearing a new name.** Contention is one-sided: it can only make
a measured interval longer. The honest head is a floor near the clock's
resolution plus a preemption that sometimes lands in it; the mutated head has a
HARD floor of one serialization, present in every iteration. A minimum over K
strips the sporadic term from the honest side and cannot strip the deterministic
term from the defective side. The withdrawn bound compared two single
measurements of comparable magnitude and the tail decided it. This compares the
minima of two populations that differ by five orders of magnitude.

### 7. The console copy answers the same question as the file copy

A fresh review found the two had diverged. `phase-log.json` carried
`instrument_seconds` and `PhaseLog::RenderText` printed `sum(leaf)`,
`unaccounted` and `WALL` without it, so a reader watching a terminal saw the
residue with no way to subtract the cost of naming the phases from it -- which
is the whole reason that number exists. `RenderText` now prints it.

The gate over it is not a measurement. `RenderText` formats every total with
`%10.3f`, so the printed value differs from the number it was given by strictly
less than half of the last digit -- 5e-4 seconds -- **by the definition of the
conversion**. The case reads that line back and compares it with `Instrument()`
through the public entry point. Nothing in it is timed, both sides read the same
number through two different code paths, and no box load can move the verdict.

**AND THE QUANTITY HAS TO BE ABOVE THAT RESOLUTION, WHICH A SECOND FRESH REVIEW
MEASURED THAT IT WAS NOT.** The first version of this gate lived inside the
conservation case, on its three-scope timeline, where `Instrument()` is
**2.80e-4 to 2.95e-4 s -- SMALLER than the tolerance**. `%10.3f` therefore
prints `0.000` for the honest value and the comparison is satisfied by anything
in `(-5e-4, +5e-4)`, the literal `0.0` included. The reviewer replaced
`Instrument()` with `0.0` in the production line and the whole file stayed GREEN
at `5 | 5 passed` and `93 | 93 passed`.

That is #1569's own failure reproduced inside #1569's own repair: a
discriminator below the instrument's resolution cannot discriminate. The repair
is the one #1569 itself took -- make the quantity an EVENT and REFUSE to assert
when it is not. The gate now has its own case with 4000 sequential scopes, where
the table's own share measures **0.40 to 0.44 s, about 800x the format's last
digit**, and a `REQUIRE` above it says so out loud rather than passing quietly
on a table too cheap to print. `NTEXTZERO` reds there; so does `NTEXT`, which
deletes the line.

### 8. The table's own charge is disjoint too, and that is a SECOND arm

`ChargeLocked` has two targets and the clamp landed on both, but the second
review found only the RECORD arm was gated: removing the TABLE arm's high-water
mark left the whole file green, because every other case charges the table from
ONE thread, where the intervals are already disjoint.

The shape that breaks it is the hammer with NOTHING LIVE. `Tick` reads its clock
before taking the process-wide mutex, so N threads blocked on the same
acquisition each charge their own full wait, and with no leaf open they all land
on `instrument_gap`. The bound is the timeline itself and it is arithmetic:
`Instrument()` is a sum of intervals inside `[0, wall]`, so if they are disjoint
their sum is at most `wall`. Measured 0.993 honest, and `NTABLECLAMP` reds it.

**That mutation's FIRST run is worth recording, because it is the trap this
project keeps hitting from a new direction.** The new case's name contained a
COMMA -- `... is disjoint, so it cannot exceed the timeline` -- and doctest's
`-tc` filter SPLITS ON COMMAS, so the filter matched nothing and the run printed
`test cases: 0 | 0 passed | 0 failed | 7 skipped` and `Status: SUCCESS!` at
`rc=0`. Only the harness printing the case count separated that from a mutation
the suite genuinely does not catch. The comma is gone from the name.

`serialize > 1e-5` guards the comparison from the other side. A table too cheap
to serialise cannot separate the two orderings at all, which is precisely why
#1569's three-record case stayed green 10 of 10 under its own mutation. A
precondition that fails loudly is the difference between a gate and a mute
switch.

### 10. The CONSOLE emitter had the same defect, and the call site made it worse

`### 6` repaired `WriteJson`'s clock ordering and left its sibling alone.
`RenderText` read `Elapsed()` AFTER `ByStart(Records())`, so the console copy of
a table charged its own copy and its own sort to the `WALL` it printed and to
the `unaccounted` row above it. A fresh review of this row measured it and filed
[#1755](https://github.com/mudler/vllm.cpp/issues/1755).

The call site was the larger half. The console block stood at the END of
`WriteJson`, after the whole `nlohmann` object was assembled, so the console copy
also absorbed the JSON build while the file copy did not. Over five `WriteJson`
calls on the 8001-record unit timeline the console's `sum(leaf)` held at 0.189 s
while its `unaccounted` climbed 0.065 -> 0.134 -> 0.200 -> 0.265 -> 0.329 s.
About 66 ms of writer work per call, on the copy of the table a reader watching a
terminal actually gets.

**The repair is two statements.** `RenderText` reads its clock first, mirroring
`### 6`. And the console block moves to immediately after `WriteJson`'s own clock
read, above the copy, the sort and the build, so the two emitters' clock reads
are one `getenv` apart and the two copies describe one instant.

**The bound is the FORMAT'S resolution and there is no ratio in it.**
`RenderText` prints every total with `%10.3f`, so a printed total differs from
the number it was given by strictly less than 5e-4 s by the definition of the
conversion. The gate compares the printed `WALL` against the clock the case read
immediately before the call, against ONE FULL step of that format, `1e-3` s. The
honest side is held under that by ARITHMETIC rather than by margin: the two clock
reads are one function call apart, so the whole difference between them IS the
rounding. This is the same constant `### 7` reads off the same format string, and
it is deliberately not the withdrawn ratio that `ltx25-phase-residue.md`
`## Design` 3 measured and #1668 forbids re-proposing.

**And the table has to be big enough for the defect to cross that last digit,
which is the whole reason this case is not three records.** This is `### 6`'s
mute-switch argument on a coarser instrument: a copy and a sort of 8000 records
is 0.12 ms, a QUARTER of one step of `%10.3f`, so applying the one-line repair to
`RenderText` on the old test suite left it at `7 | 7` and `100 | 100`. The two
arms therefore carry two preconditions, each measured in the same run through the
same public entry points, each a MINIMUM over probes, and each demanding TWO full
steps of the format so that a defect sitting exactly on the floor still exceeds
the one-step bound by 1.5x:

- arm (B), the ordering: the CHEAPER of `Records()` and `stable_sort` over
  250000 records. Smaller because a partial regression moves only one of them,
  which is the shape a fresh review of `### 6` actually produced.
- arm (A), the call site: the writer's own per-record `phases` array, assembled
  by the case through the same library over 16000 records.

**WHAT EACH ARM SEPARATES, AND WHAT IT DOES NOT.** The repair above is three
things — the console block moves above the copy, above the sort AND above the
JSON build — and the two arms between them hold two of the three. This is a
statement about the GATE and not about the code, and it is measured rather than
argued. Issue
[#1760](https://github.com/mudler/vllm.cpp/issues/1760) owns the gap.

- **arm (B) holds `RenderText`'s own internal ordering, including a partial
  regression of it.** Its discriminator is `min(copy, sort)` over 250000
  records, and the table is sized to that quantity. `M-RT` and `M-RT-PARTIAL`
  are each RED 10/10, and the partial one moves only ONE of the two steps, so
  the arm is held against half a revert as well as a whole one.
- **arm (A) holds only that the console block stands above the JSON BUILD.** Its
  discriminator is the writer's own per-record `phases` array over 16000
  records, and the table is sized to THAT quantity: measured in the same runs on
  this box, that build is 6.5388e-3 to 7.2559e-3 s, i.e. **6.54 to 7.26 steps**
  of `%10.3f`. The copy and the sort at the SAME 16000 records are roughly an
  order of magnitude cheaper, and the arm cannot separate them. **`M-SITE-MID`
  measures exactly that**: the console block slides BELOW `ByStart(Records())`
  and `Sum(...)` and stays ABOVE the `nlohmann` build, which is #1755's own
  class reintroduced in the two-thirds arm (A) does not cover, and it survived
  **19 of 20 runs**. Its lags ran 3.8975e-4 to 1.0344e-3 s against the 1e-3 s
  bound — **0.39 to 1.03 steps**, straddling the bound rather than clearing it,
  which is why one run in twenty came back red and nineteen did not.

So the composite case holds *"the console block sits above the JSON build"* and
*"`RenderText` reads its clock before its own copy and sort"*. It does **not**
hold *"the console block sits above the copy and the sort"*. A reader who takes
the case's own comment — "it sits above the copy, the sort and the whole JSON
build" — as a statement of what the gate covers is reading the intent rather
than the coverage.

**ENLARGING ARM (A)'s TABLE IS MEASURED SHUT, so this is a recorded limit and
not a to-do.** `WriteJson` holds one `nlohmann` object per record while it dumps,
about 3.3 KB per record; a table big enough for the copy and the sort to cross
the last printed digit costs about a gigabyte of resident set through arm (A),
and arm (A) already costs 141 MB and 2.2 s in CI. A new wall-clock tolerance is
the other obvious move and #1668 forbids re-proposing it. What would settle
#1760 is a bound that does not go through `%10.3f` at all — a structural
assertion over the block's position, or a `RenderText` that is HANDED the wall it
prints instead of reading a clock, which deletes the ordering question. Each is a
production change to a shared emitter and owes its own red-first evidence.

**Two tables rather than one, and the reason is memory.** `WriteJson` holds one
`nlohmann` object per record while it dumps, about 3.3 KB per record, so a single
table big enough for arm (B) costs a gigabyte of resident set through arm (A).
Arm (B) never serialises to JSON and costs 140 MB at 250000 records.

**The build's progress lines go to a sink.** `PhaseLog::Close` flushes one line
per scope, and on this box 120000 scopes cost 3.81 s through a pipe against
0.65 s to a file. `ProgressEnabled()` latches its environment variable once per
process, so a `setenv` inside a case cannot turn it off; redirecting fd 2 can,
and the same mechanism is what reads the console block back.

## Dependencies

None. `LTX25-DEVICE-RESIDENCY` owns `render_phase_log.{h,cpp}`'s existence and
the `load.open` anchor; this row extends the instrument and renames nothing.

## Risks and decisions

**D1 — the instrument's cost is CHARGED, never subtracted globally.** A single
global subtraction is a number nobody can attribute. Charging each interval to
the innermost live non-span record keeps the attribution local and makes the
conservation invariant testable, which is what the unit cases assert.

**D2 — a new test executable rather than a block in `test_ltx2_video`.** Two of
the four cases need a table of thousands of records, which no render produces,
and `test_ltx2_video` costs a fixture build and has been measured at 30-36 GB of
anonymous resident set. Three other issues are editing that file concurrently.
The instrument's own cases go in `tests/vllm/multimodal/test_render_phase_log.cpp`.

**D3 — reachability stays in `test_ltx2_video`.** Every case in the new file
calls `PhaseLog` directly, which proves the class works and never that a render
reaches it — the exact failure `AGENTS.md` "Nothing lands dead" names. The
assertion that `vllm_video_generate`'s own table carries `instrument_seconds` and
a reconciling `gaps` is added to `a render through the ABI emits a phase table
that SUMS to wall`, and it is the only thing this row adds to that file.

**D4 — a negative gap is emitted rather than clamped.** It cannot arise while
the non-overlap invariant holds, so clamping would hide a broken instrument
inside a number that still adds up. The unit case asserts it is never negative.

**D5 — `#1668` is NOT closed by this row.** It owns four items and this row
lands one. Closing it on the strength of item 4 would lose items 1 to 3, which
is the failure #1668 was filed to prevent.

## Tests

`tests/vllm/multimodal/test_render_phase_log.cpp`, EIGHT cases. It was four when
this section was first written; `### 7` and `### 8` each added one, the
many-threads reproduction added a third and `### 10` added a fourth, and the
count is written out here because a `-tc` filter that matches nothing prints
`Status: SUCCESS!` at `rc=0`. Every run recorded under `### The two bounds` reads
`test cases: 7 | 7 passed` and `assertions: 100 | 100 passed`; every run recorded
under `### The console emitter` reads `test cases: 8 | 8 passed` and
`assertions: 120 | 120 passed`. A run that reads neither is not evidence.

| Case | What it holds | Shape |
|---|---|---|
| the instrument charges its own cost to the innermost LEAF | attribution: a child's boundary is the parent's cost, a boundary under a bare span is the table's, a span is not a leaf | "it moved", "it did not move at all", "it is positive" — no duration compared |
| the instrument's own cost is CONSERVED across the table and its records | every charge non-negative, no record charged past its own duration, the table's share no larger than the residue it is part of | inequalities between two numbers in the same file |
| the TABLE's own charge is disjoint so it cannot exceed the timeline | the second arm of the clamp, `### 8`. N threads blocked on one acquisition with NOTHING live all land on `instrument_gap`, and disjoint intervals inside `[0, wall]` cannot sum past `wall` | arithmetic over the timeline, not a tolerance. `NTABLECLAMP` reds it |
| the CONSOLE copy carries the same instrument charge as the FILE copy | `### 7`. 4000 sequential scopes make the table's share ~0.4 s against the `%10.3f` format's 5e-4 last digit, and a `REQUIRE` above that resolution refuses to assert when the quantity is too small to discriminate | reads the printed line back and compares it with `Instrument()` through the public entry point. Nothing is timed. `NTEXT` and `NTEXTZERO` red it |
| a record charged from MANY THREADS is still charged less than it lasted | the record arm of the clamp. 24 threads ticking inside one live leaf drove `instrument / duration` to 1.914 before it, red 3 runs in 5 | `instrument_seconds <= duration_seconds`, held by the disjointness construction rather than by margin. `NCLAMP` reds it |
| the emitted table DECOMPOSES its residue into the gaps between leaves | N leaves give N+1 gaps, each names the two leaves it lies between, none is negative, and they SUM to `unaccounted_seconds` | an accounting identity, plus one lower bound on a `sleep` |
| the emitter reads its CLOCK before it serialises the table | #1569 | `### 6` |
| the console copy reports the wall this emitter was ENTERED at | `### 10`. #1755. Two arms: `RenderText` against the clock read immediately before it over 250000 records, and the `VLLM_RENDER_PHASE_LOG_STDERR` block across `WriteJson` over 16000 | the printed `WALL` against ONE step of that line's own `%10.3f`, with each arm's discriminator measured in the same run and each lag a minimum over probes. `M-RT`, `M-RT-PARTIAL`, `M-SITE` and `M-BOTH` red it, and `M-SITE-MID` does NOT — [#1760](https://github.com/mudler/vllm.cpp/issues/1760) owns the third of the repair this case leaves unheld |

Plus, in `tests/vllm/multimodal/test_ltx2_video.cpp`, inside the existing ABI
render case: the emitted table carries `instrument_seconds`, carries `gaps`, and
those gaps reconcile to that render's own `unaccounted_seconds`. That is D3.

## Gates

`ctest --test-dir build -R 'test_render_phase_log|test_ltx2_video'`, plus
`scripts/agent-preflight.sh`.

Inheritance is established by FAILURE TEXT rather than by job name, and the
inherited set MOVED under this branch while it was open. At `019f66c1a` `main`
was red on `build-test-cpu`, both `sanitize-cpu` arms and both `windows-msvc-*`.
By `6354755ba` three of those five are GREEN again on this branch:
`build-test-cpu` and both sanitizers passed once `main` landed the
`test_runner.cpp:1557` repair ([#1602](https://github.com/mudler/vllm.cpp/issues/1602),
[#1608](https://github.com/mudler/vllm.cpp/issues/1608)), which this branch
inherits by merge and not by any edit of its own.

What is left on this branch is two jobs, and NEITHER is this row's:

- `windows-msvc-cpu` and `windows-msvc-vulkan` — the
  [#1649](https://github.com/mudler/vllm.cpp/issues/1649) `/W4 /WX` refusal,
  which fires BEFORE any translation unit is read and so carries no
  `error C####`. Nothing this row touches is compiled when it fails.
- `agent-record` — read the TEXT, because the job name alone would have been
  waved through here and it is a records gate on a records-heavy branch. Its
  failure was `ERROR: .agents/issue-index.md: issue #1649 listed twice`, a
  `merge=union` duplicate that `main` repaired in `6354755ba` and that this
  branch inherits by merge. The `record anchors ... -> rot 37` line printed
  beside it is the rot budget being MET, not exceeded: `check-agent-record.py`
  prints `ANCHOR-ROT=37` and exits 0 on `main` and on this branch alike.
  Verified locally on the merged tree rather than inferred from the job turning
  green.

## Stop conditions

- Do not widen either floor to close a red. Name the phase, or leave the red and
  file the gap.
- Do not put `instrument_seconds` in a denominator without reading
  [`ltx25-phase-residue.md`](ltx25-phase-residue.md) `## Design` 3, and never
  accept a 20-run distribution as evidence about a quantity of that shape.
- Do not close #1668 from this row. D5.
- Do not close #1439. It asks for a bound on a quantity the scheduler cannot
  move; the gap decomposition gives a reader that quantity and asserts an
  identity over it, and neither is the budget #1439 asks for.

## Owed

| Issue | Owed |
|---|---|
| [#1668](https://github.com/mudler/vllm.cpp/issues/1668) | items 1 to 3, the three driver anchors and `sampler_updates`. Item 4 lands here |
| [#1570](https://github.com/mudler/vllm.cpp/issues/1570) | the bound on `instrument_seconds / duration_seconds`. It needs `instrument_seconds`, which this row lands. **Set it on a CARRYING LEAF, not per record.** See `### The instrument share, measured before it was proposed` below: on a leaf holding seconds of work the honest share is ~1e-4 and the margin is enormous, which is where #1570's actual concern lives -- an instrument that got ten times more expensive would be visible there. On a sub-scope of tens of microseconds it is not a gate at all |
| [#1568](https://github.com/mudler/vllm.cpp/issues/1568) | the `denoise.step` / `denoise.update` seconds transfer. `denoise.update` does not exist yet -- **and the obvious closure is now MEASURED SHUT.** See below |
| [#1567](https://github.com/mudler/vllm.cpp/issues/1567) | the res_2s arm's anchor. No gate in this tree renders on that arm |
| [#1439](https://github.com/mudler/vllm.cpp/issues/1439) | NOT closed. See `## Stop conditions` |
| [#1718](https://github.com/mudler/vllm.cpp/issues/1718) | the charge sites are gated only in AGGREGATE. Three of this row's own mutations -- N4, N6 and NNOSORT -- are GREEN, and they are printed in the mutation table rather than left out. It does NOT cover [#1760](https://github.com/mudler/vllm.cpp/issues/1760), which is a call-site ORDERING survivor rather than a charge site |
| [#1719](https://github.com/mudler/vllm.cpp/issues/1719) | `PhaseLog::Close`'s pre-lock wait is charged to nobody, so it inflates the closing record's UNCOVERED time -- which is the quantity the coverage floor reads. It has to land before #1718 can, because it is where staged contention actually goes |
| [#1720](https://github.com/mudler/vllm.cpp/issues/1720) | `WriteJson` now takes the process-wide mutex TWICE, so `wall_seconds` and the record set are two snapshots rather than one. Unreachable on the shipped path and argued in the function's own comment; the repair is a single locked snapshot, which is a public API change |
| [#1760](https://github.com/mudler/vllm.cpp/issues/1760) | `### 10`'s arm (A) holds only that the console block stands above the JSON BUILD, not that it stands above the copy and the sort. `M-SITE-MID` slides the block below `ByStart` and `Sum` and survives **19 of 20 runs**; it is a GREEN row in the console mutation table with its twenty measured lags beside it. The cause is arm (A)'s table SIZE -- at 16000 records the JSON build is 6.54 to 7.26 steps of `%10.3f` and the copy plus the sort is 0.39 to 1.03 -- and enlarging that table is measured shut at about a gigabyte of resident set. Settling it needs a bound that does not go through the printed format at all: a structural assertion over the block's position, or a `RenderText` HANDED the wall it prints instead of reading a clock. Both are production changes to a shared emitter and each owes its own red-first evidence |

### Owed out of the fresh review

Filed by the fresh review of this row and left open ON PURPOSE, with what each
one would take. None of them is a defect in what landed; each is a gate this row
could not make honest.

| what | why it is not here |
|---|---|
| [#1718](https://github.com/mudler/vllm.cpp/issues/1718) -- **a per-site charge gate.** Deleting any ONE charge site -- `Open`'s pre-lock mutex wait, `Open`'s tail, `Close`'s tail, the sampler join, `SampleLocked`'s self-charge -- leaves the suite green; only deleting every site reddens it. The worst is the pre-lock wait, which `## Design` 1 names as the reason the mechanism exists | A case for it was written and run THREE times and it does not measure the site. `PhaseLog::Records()` is the only public entry point that holds the mutex without charging itself, and contention through it lands mostly in `PhaseLog::Close`'s lock wait -- which is charged to nobody and lies inside the CHILD's duration, not the parent's charge. The case passed in isolation at 615x and failed 5 of 5 inside the suite. The full account is in the comment where it would have been |
| [#1719](https://github.com/mudler/vllm.cpp/issues/1719) -- **`Close`'s pre-lock wait is charged to nobody.** Found by the attempt above. `Open` reads a clock before its lock and `Close` does not, so the wait before a record's `end` is stamped is instrument wall that no record and no table absorbs | It is a production change to a shared instrument and it owes its own red-first evidence. It inflates the closing record's UNCOVERED time, which is the quantity the coverage floor reads |
| [#1718](https://github.com/mudler/vllm.cpp/issues/1718) -- **`ChargeLocked`'s `if (from < 0.0) return;` is ungated.** Changing it to a clamp reddens nothing, although the comment beside it argues at length that clamping is the defect that makes a gate pass | Staging it needs a `Begin()` on another thread between `Open`'s clock read and its lock acquisition, which is a race this row could not make deterministic |
| [#1718](https://github.com/mudler/vllm.cpp/issues/1718) -- **`ByStart`'s REMOVAL is held by nothing, although its INVERSION now is.** `NNOSORT` leaves the suite green and `NREVSORT` reds it | It cannot matter for the decomposition -- non-nested leaves are strictly sequential and already close in start order -- but `## Design` 3 leans on it and the emitted table's monotone-start assertion lives in `test_ltx2_video` |
| **The new suite writes about 8000 flushed `[render]` lines per run.** The live lane is on by the shipped default (#1413) and the #1569 case builds 8000 leaves | Silencing it needs `VLLM_RENDER_PROGRESS=0` at process start, and that would take the flushed progress line -- the most expensive statement in `Open`, and one of the charge sites -- out of everything this file measures |
| **A gap's `after` and `before` are ambiguous when a name repeats.** `decode.video` opens more than once per render, so two gaps can carry the same pair of names | The pairing assertion accepts it, and a reader wanting the exact region has `start_seconds` and `end_seconds`, which this row's repair made gated |

### The instrument share, measured before it was proposed

The natural way to close #1568 and #1570 at once, once `instrument_seconds`
exists, is a ceiling on `instrument_seconds / duration_seconds` per record: an
empty scope is nothing but its own boundaries, so its ratio approaches 1, while
a scope carrying work sits far below. #1568's R1b transfer leaves
`denoise.update` empty, so one ceiling would catch it.

**That was measured on this tree before it was proposed, and at
`denoise.update`'s real scale it does not work.** A standalone probe opens two
nested scopes inside one parent -- one empty, one holding a sleep -- and reports
both ratios from one process. `VLLM_RENDER_PROGRESS=0`, this branch's binary,
`build/libvllm.a`:

| work scope | n | EMPTY min | EMPTY median | WORK median | WORK max | separated? |
|---|---:|---:|---:|---:|---:|---|
| 5 ms | 60 | 0.3656 | 0.8520 | 0.0045 | 0.0097 | yes, 38x |
| ~1.1 ms | 60 | -- | -- | 0.0195 | 0.0528 | yes |
| ~134 us | 80 | 0.3914 | 0.8418 | 0.1248 | 0.2500 | barely, 1.57x |
| ~134 us | **200** | **0.0804** | 0.8487 | 0.1255 | **0.8930** | **NO -- they overlap completely** |

The 80-sample row says the populations are separated by 1.57x. The 200-sample
row, same probe, same box, says the honest maximum (0.8930) is ELEVEN TIMES the
defective minimum (0.0804). **A small sample of this quantity does not see the
tail that decides the gate**, which is the same finding
[`ltx25-phase-residue.md`](ltx25-phase-residue.md) `## Design` 3 records for the
withdrawn bound, reached independently by a different route.

The mechanism is the same one too, and its direction is worth stating. The part
of a boundary this instrument CANNOT measure -- the `lock_guard` release, the
`Close` return, the `Scope` destructor and constructor, the call into `Open` up
to its clock read -- dilates faster under contention than the part it can. It
sits inside the record and outside the charge, so a preemption there drives an
EMPTY scope's ratio DOWN toward an honest one's, and a preemption inside a tiny
honest scope's own boundaries drives it UP toward an empty one's. Both
populations move toward each other, and at 49 to 343 us -- which is what
`denoise.update` measures -- they meet.

So #1568 stays open, and its own text was right for a reason it did not name:
"the honest distribution's bottom touches the defective value". It does, in this
formulation as well as in a coverage floor.

#1570 survives, on a LEAF rather than a record. `denoise` is seconds of work
against an instrument charge measured in microseconds, so a ceiling there has
four orders of magnitude of margin and would still catch the ten-times-more-
expensive instrument #1570 is about. The row that lands `denoise.update` owes
that measurement on the real leaf.

## Evidence

Measured on this branch, on an x86_64 box at load average 103 to 131 -- which is
5 to 6 times its core count and is the regime the withdrawn bound failed in.
Build: `cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON`, no `CMAKE_BUILD_TYPE`,
which is what `build-test-cpu` configures.

### The two bounds, over 355 runs across six load regimes

Every run's `test cases:` line was recorded, because a `-tc` filter that matches
nothing prints `0 cases ran` and `Status: SUCCESS!` and is indistinguishable
from a green run in a log or an `&&` chain. Every run of the whole suite also
read `assertions: N | N passed | 0 failed`.

**Before the fresh review**, on the one-number `copy + sort` budget:

| population | n | load | min | median | p90 | max |
|---|---:|---|---:|---:|---:|---:|
| the #1569 case alone | 50 | 103-131 | 0.000716 | 0.002308 | 0.003115 | **0.004228** |
| the whole suite | 60 | 20-28 | 0.000320 | 0.000759 | 0.001145 | 0.001894 |
| the #1569 case alone | 50 | 57-83 | 0.001650 | 0.002600 | 0.003648 | 0.004197 |
| the whole suite | 60 | 35-48 | 0.000320 | 0.000759 | 0.001145 | 0.001894 |

**After it**, on the repaired `min(copy, sort)` budget, which is about ten times
smaller and therefore about ten times tighter:

| population | n | load | min | median | p90 | max | margin |
|---|---:|---|---:|---:|---:|---:|---:|
| `head / min(copy, sort)` | 45 | 58-85 | 0.00198 | 0.00444 | 0.00654 | **0.00958** | **52x** |
| `instrument / duration`, 24 threads | 45 | 58-85 | 0.0904 | 0.9920 | 0.9945 | **0.9961** | bound 1.0 |

**And re-measured over 45 further runs of the WHOLE suite against the tree that
actually lands**, which is not the tree the two rows above were taken on: the
`RenderText` gate is new, and it is the only quantity a re-run could have moved.

| population | n | load | min | median | p90 | max | margin |
|---|---:|---|---:|---:|---:|---:|---:|
| `head / min(copy, sort)` | 45 | 56-113 | 0.003744 | 0.006467 | 0.010502 | **0.018123** | **27.6x** |
| `instrument / duration`, 24 threads | 45 | 56-113 | 0.0723 | 0.9928 | 0.9962 | **0.99756** | bound 1.0 |

All 45 read `test cases: 5 | 5 passed | 0 failed` and
`assertions: 93 | 93 passed | 0 failed` -- the SAME string on every run, which is
what rules out a filter that matched nothing. The margin is 27.6x rather than
52x because this population is a load regime higher, and that is the direction a
reader should expect: the honest head grows with contention and the budget under
it does not.

**And once more against the tree that finally lands**, which is neither of the
trees above: the console gate moved into its own case, the table arm gained one,
and two gap tautologies were replaced.

| population | n | load | min | median | p90 | max | margin |
|---|---:|---|---:|---:|---:|---:|---:|
| `head / min(copy, sort)` | 45 | 13-24 | 0.003355 | 0.006841 | 0.008413 | **0.010226** | **48.9x** |
| `instrument / duration`, 24 threads | 45 | 13-24 | 0.9898 | 0.9958 | 0.9967 | **0.99686** | bound 1.0 |

All 45 read `test cases: 7 | 7 passed | 0 failed` and
`assertions: 100 | 100 passed | 0 failed` -- the SAME string on every run, which
is what rules out a filter that matched nothing.

**0 red in 355.** And the defective values are not near either bound:

```
#1569        honest [0.0037 .. 0.0181]   bound 0.5    main's order 17.2,  partial regression 1.024
conservation honest [0.072 .. 0.99756]   bound 1.0    no clamp     21.97
```

The conservation ratio sits just under its bound BY CONSTRUCTION and a reader
should not read that as a bound about to flap. That leaf is 24 threads ticking
and almost nothing else, so almost all of it IS instrument. It cannot cross 1.0
on a slow box, because the charges to one record are disjoint intervals inside
it: the arithmetic holds it, not the margin.

Contrast the withdrawn bound: its honest population had a median of 1.132 and a
maximum of 4.115 against a bound of 2, so the bound sat INSIDE its own scatter
and 4 runs in 45 crossed it.

### The operator's own re-run, on the tree that pre-dates the `### 10` repair

`## How work gets done` 4: an implementer or reviewer report is an input, never
a gate result. Everything below was re-run by the merging session on the tree
that carries the `origin/main` merge, after `git status` confirmed `src/`,
`include/` and `tests/` clean and `render_phase_log.cpp` matched sha256
`e490ccc390b845aa0795bb60b70fbc9567575d23532e7a1f5a1faf64f0834224`.

**The whole mutation set, 23 mutations.** Each printed its own
`git diff --numstat`, its `compile_status`, both doctest count lines, and a
`restore_sha256_ok` against a pristine byte copy. 23 of 23 restored true, 23 of
23 compiled at `compile_status=0`, and no numstat was empty. **19 RED, 4 GREEN**,
and the four green are exactly the four this row already files under
[#1718](https://github.com/mudler/vllm.cpp/issues/1718) — `NSEED`, `N4`, `N6`,
`NNOSORT`. No mutation the table calls red came back green.

**That `19 + 4 = 23` counts THIS population and no other, and the row's total is
five green rather than four.** These 23 were measured on the tree whose
`render_phase_log.cpp` reads sha256 `e490ccc3…`, which pre-dates the `### 10`
repair, so the call site `M-SITE-MID` mutates did not exist on it. The console
emitter carries its own separate table below —
`### The console emitter, measured by the session that repaired it`, **seven
rows, six RED and one GREEN** — and its green row is `M-SITE-MID`, filed under
[#1760](https://github.com/mudler/vllm.cpp/issues/1760) rather than under #1718,
because it is the #1755 call-site ORDERING class and not the
`instrument_seconds` charge-site class. The two tables together are **30
mutations, 25 RED and 5 GREEN**, and every green one is named beside the issue
that owns it.

**`M3`'s anchor was NOT unique, and the harness refused rather than mutating the
wrong function.** `if (r.span || r.nested) continue;` occurs TWICE — once in
`Sum` and once in `GapsBetweenLeaves` — so a text replace would have silently hit
whichever came first. Re-run against each site separately: the decomposition site
reds on a `-1.232 ms` gap and `REQUIRE(gaps.size() == leaf_names.size() + 1)` at
`4 == 3`, and the `Sum` site reds on the identity at `0.00114164 < 1e-9`. Both
arms are held; only the harness was ambiguous. This is why an anchor is asserted
UNIQUE rather than merely present.

**#1569's mutation can now fail, which is the whole point of the case.** The
defect this row exists to close is that #1569's mutation stayed GREEN 10 of 10.
Both orderings were therefore re-run 10 consecutive times each, under load, with
the doctest case count asserted on every single run:

| tree | n | verdict | ratio min | ratio max | bound |
|---|---:|---|---:|---:|---:|
| honest | 45 | GREEN 45/45 | 0.009467 | **0.027616** | 0.5 |
| `M1`, `main`'s clock order | 10 | **RED 10/10** | **2.080** | 3.418 | 0.5 |
| `N11`, the PARTIAL regression | 10 | **RED 10/10** | **0.892** | 1.145 | 0.5 |

All 65 runs printed the identical string `test cases:  1 |  1 passed | 0 failed`,
which is what rules out a `-tc` filter that matched nothing and reported
`SUCCESS!`.

**The two populations do not touch.** The honest maximum over 45 runs is 0.0276
and the WORST defective run of the harder mutation is 0.892 — a factor of **32**
between them, with the bound at 0.5 sitting between the two with 18.1x of
headroom below it and 1.78x above. That separation is the property `## Design` 6
argues for from the definition of the quantities, measured rather than asserted.

**The honest distribution, full, not a median.** n=45, loadavg 19.40-26.42,
min 0.009467, median 0.015184, p90 0.021052, p95 0.025348, **max 0.027616**. The
margin is quoted at the MAX and not at the median, because the failure this
cluster exists to stop is a heavy right tail that a median never sees. This run
is a lower load regime than the 13-24 population below and a tighter margin than
its 48.9x, which is the direction a reader should expect and the reason both are
kept rather than the better one.

### The console emitter, measured by the session that repaired it

`### 10`, issue [#1755](https://github.com/mudler/vllm.cpp/issues/1755). A
SEPARATE population from the 23 above, on a later tree, so the counts in
`### The operator's own re-run` still reconcile against
`### The mutation table`. Same x86_64 box, loadavg 11 to 21, Release build with
`-DVLLM_CPP_BUILD_TESTS=ON`. Every run below printed its own doctest count lines
and they are quoted rather than summarised, because a `-tc` filter that matches
nothing prints `Status: SUCCESS!` at `rc=0`.

The honest tree reads `test cases: 8 | 8 passed | 0 failed` and
`assertions: 120 | 120 passed | 0 failed`. `main`'s code — both halves, `M-BOTH`
— reads `test cases: 8 | 7 passed | 1 failed` and
`assertions: 120 | 118 passed | 2 failed`, measured on the WHOLE suite 3 runs of
3. The other seven cases stay green under it, which is the finding restated: the
suite this row already carries cannot see this defect.

**0 RED IN 45 HONEST RUNS.** Ten of the whole suite at `8 | 8 passed` and
`120 | 120 passed`, and thirty-five of the case alone at
`test cases: 1 | 1 passed | 0 failed | 7 skipped` and
`assertions: 20 | 20 passed | 0 failed`. Every one printed `Status: SUCCESS!` and
the identical count strings, which is what rules out a `-tc` filter that matched
nothing. The measured lags on the honest tree ran 7.1e-5 to 4.7e-4 s against the
1e-3 s bound, and their ceiling is the rounding rather than the scatter: the two
clock reads each comparison spans are one function call apart.

The discriminators on this box, at 250000 and 16000 records: `min(copy, sort)`
4.22e-3 to 5.11e-3 s, i.e. 4.2 to 5.1 steps of the printed format against a
precondition of 2; the writer's per-record `phases` build 6.7e-3 to 9.3e-3 s,
i.e. 6.7 to 9.3 steps against the same precondition.

| id | mutation | verdict |
|---|---|---|
| M-RT | `RenderText` reads its clock AFTER the copy and the sort, i.e. `main`'s code | RED 10/10 on arm (B) |
| M-RT-PARTIAL | the clock read moves BELOW the copy and stays ABOVE the sort — a PARTIAL regression, the shape that broke `### 6`'s one-number budget | RED 10/10 on arm (B) |
| M-SITE | the console block returns to the END of `WriteJson`, after the whole JSON object, i.e. `main`'s call site | RED 10/10 on arm (A) |
| M-BOTH | both halves reverted, i.e. `c7ca0142a` | RED 10/10 on both arms |
| N-BOUND-ON-BOTH | M-BOTH plus the two `CHECK` bounds widened a thousandfold and NOTHING else — the control asking whether M-BOTH's red comes from the bounds or from the preconditions | **GREEN 3/3** at `20 \| 20 passed`, so the bounds are what red. Widening `kStep` instead reds at the `REQUIRE`s, because that constant feeds the preconditions too |
| N-PRECOND | arm (B)'s table shrinks to three records, the shape #1569 could not gate | RED at the `serialize` `REQUIRE`, loudly, rather than passing quietly |
| M-SITE-MID | the console block slides BELOW `ByStart(Records())` and `Sum(...)` and stays ABOVE the `nlohmann` build, so the console `WALL` is charged with the writer's own copy and sort but not its serialization -- #1755's class in the two-thirds arm (A) does not cover | **GREEN 19/20** -- [#1760](https://github.com/mudler/vllm.cpp/issues/1760) |

**SEVEN ROWS, six RED and one GREEN, and the green one is printed here rather
than left out.** A mutation table listing only its successes is an argument and
not a measurement. `M-SITE-MID` was staged by the repair session that landed
`### 10`'s wording, on the same Release build and box, at loadavg 8.5 to 10.3,
with `anchor_occurrences=1`, `git diff --numstat` `3 3`, `compile_status=0` and
`restore_sha256_ok=True` against a pristine byte copy, and the numstat EMPTY
after the restore. Nineteen runs read `test cases: 8 | 8 passed | 0 failed` and
`assertions: 120 | 120 passed | 0 failed`; one read `8 | 7 passed | 1 failed`
and `120 | 119 passed | 1 failed`. **A mutation a gate catches once in twenty
is a mutation the gate does not catch**, and it is worse than a clean survivor
because the single red is a flake a later session will reasonably discount. The
twenty lags, each a minimum over 3 probes, against the 1e-3 s bound:

```
7.0954e-4  8.0859e-4  6.1208e-4  5.5063e-4  7.0809e-4
4.4587e-4  1.0344e-3  4.8383e-4  6.9707e-4  5.2762e-4
3.8975e-4  8.3156e-4  4.0191e-4  7.2963e-4  7.7509e-4
7.8492e-4  4.7455e-4  9.6428e-4  6.6708e-4  7.8302e-4
```

The honest tree on the same box, same session, read 7.94e-5, 1.206e-4, 2.876e-4
and 4.492e-4 s. Why the arm cannot separate it is in `### 10` and it is the
table SIZE rather than the bound: at 16000 records the JSON build is 6.54 to
7.26 steps of `%10.3f` and the copy plus the sort is 0.39 to 1.03 steps.

The numbers each arm ran against are in the run log the repair session returned.
Nothing in this block is a ratio, and the two discriminators — the cheaper of
`Records()` and `stable_sort` over 250000 records, and the writer's own
per-record `phases` array over 16000 — are measured inside every run rather than
quoted from another box.

### The mutation table

Every mutation prints its own `compile_status` and a sha256 pair, because a
mutation that fails to build and a mutation that never applied both read as a
passing test. Every one restored the tree byte-for-byte, verified by sha256, and
the first attempt at a text-reverse restore FAILED that check -- its anchor was
no longer unique once applied, because `RenderText` carries the same two lines
`WriteJson` does -- so the harness restores from a pristine byte copy instead.

THE WHOLE SET WAS RE-RUN AGAINST THE REPAIRED TREE, not carried over from the
run that preceded the fresh review, because three of the repairs change what the
suite can see. Every entry below is from that re-run.

TWENTY-THREE ROWS, and `M3` is two of them. Its anchor occurs at both the
`GapsBetweenLeaves` site and the `Sum` site, the harness refused the ambiguous
replace, and each site was mutated separately with a different red. Carrying it
as one row is what made this table read 22 against the 23 counted above it.

| id | mutation | verdict |
|---|---|---|
| M1 | `WriteJson` reads its clock AFTER the copy and sort, i.e. `main`'s code | RED, head 8.331 ms against a budget of 0.484 ms -- 17.2x |
| N11 | the clock read moves BELOW the copy and stays ABOVE the sort -- a PARTIAL regression, which the one-number budget missed at 0.0588 | RED, 1.024 |
| NCLAMP | the per-target high-water mark is removed, so overlapping charges are counted twice again | RED, charged 9.491 s of a 0.432 s record -- 21.97 |
| M2 | the decomposition drops the FIRST gap -- the prologue, 92% of a real residue | RED on 4: the first gap's origin, the identity, the count, and the prologue's own floor |
| M3 | the decomposition counts NESTED records as leaves, at the `GapsBetweenLeaves` site | RED on 3: a negative gap at -1.232 ms, the identity, and the count at `4 == 3` |
| M3 | the same edit at the `Sum` site, run separately because the anchor `if (r.span \|\| r.nested) continue;` occurs at BOTH | RED on the identity at `0.00114164 < 1e-9` |
| M7 | the tail gap reported as zero, count and names untouched | RED on 2: the endpoint agreement and the identity |
| M8 | each gap measured to the leaf's END rather than its START | RED on the identity alone |
| M4 | every instrument interval charged to the TABLE, never to a leaf | RED on 4 assertions across 2 cases |
| M5 | a SPAN absorbs the charge, so the residue's explanation vanishes into a number `Sum` skips | RED on the span assertion |
| M6 | the per-record charge is not emitted | RED at `REQUIRE(e.contains("instrument_seconds"))` |
| NEND | every gap's `start_seconds` and `end_seconds` zeroed, `seconds` untouched | RED on 4 -- and GREEN before this row's repair, which is why the endpoints are now read |
| NCURSOR | each gap starts at the PREVIOUS LEAF'S START, so every gap swallows the leaf before it | RED -- and GREEN against the first repair for it, see `### 9` |
| NREVSORT | `ByStart` orders the table by DESCENDING start | RED at `CHECK(seconds >= 0.0)` on -16.996 ms |
| NTEXT | `RenderText` stops printing the instrument charge, so the console and the file copies answer different questions | RED at `REQUIRE(at != std::string::npos)` |
| NTEXTZERO | `RenderText` prints a hardcoded `0.0` instead of `Instrument()` | RED -- and **GREEN against the first version of that gate**, see `### 7` |
| NTABLECLAMP | the TABLE arm's high-water mark is removed | RED -- and **GREEN before `### 8`'s case existed**; its own first run printed `0 cases ran` and `SUCCESS!` |
| NSEED | the per-record mark is seeded with `0.0` instead of the record's own `start` | **GREEN** -- [#1718](https://github.com/mudler/vllm.cpp/issues/1718) |
| R1 | the production emitter stops writing `gaps` -- run against the RENDER case | RED at `REQUIRE(table.contains("gaps"))` |
| R2 | `ChargeLocked` charges nothing anywhere -- run against the RENDER case | RED at `REQUIRE(instrument > 0.0)` |
| N4 | `ChargeLocked`'s negative-`from` refusal becomes a clamp | **GREEN** -- [#1718](https://github.com/mudler/vllm.cpp/issues/1718) |
| N6 | `Open`'s pre-lock wait is computed and charged to nothing | **GREEN** -- [#1718](https://github.com/mudler/vllm.cpp/issues/1718) |
| NNOSORT | `ByStart` stable-sorts an empty range, i.e. does not sort at all | **GREEN** -- [#1718](https://github.com/mudler/vllm.cpp/issues/1718) |

### 9. Two of the first four gap assertions were TAUTOLOGIES

The first answer to the endpoints finding added four assertions and a second
fresh review measured that two of them have no power. Inside this emitter
`gaps[i].start == records[i-1].end` and `gaps[i-1].end == records[i-1].start`,
so `gaps[i].start >= gaps[i-1].end` reduces to `records[i-1].end >=
records[i-1].start` -- a non-negative leaf duration, which a monotone clock
guarantees unconditionally. And `gaps.front().start == 0.0` is constant-true
unless the first gap is dropped, because `cursor` is initialised to the literal
`0.0`. The reviewer staged `cursor = r.start` instead of `r.end`, so every gap
swallows the leaf before it, and BOTH passed; only the pre-existing identity
fired.

Both are replaced by one comparison that has power: each gap's endpoints are
checked against **the leaf records it lies between, read out of the same emitted
table**. `NCURSOR` cannot survive that, and it needs no tolerance beyond double
rounding because both numbers come out of the same file.

THE FOUR GREENS ARE THE POINT OF PRINTING THEM. A mutation table that lists
only its successes is an argument and not a measurement, and these three are
what [#1718](https://github.com/mudler/vllm.cpp/issues/1718) is. `NREVSORT`
beside `NNOSORT` says exactly how much of `ByStart` is held: an inversion is
caught and a removal is not, because every timeline this suite builds is already
start-ordered. `NSEED` is the fourth, found by the second review: the per-record
mark's SEED -- the half that stops a charge reaching back before the record
began -- is held by nothing, because staging it needs a `Tick` whose clock read
predates a record and whose lock acquisition follows it, which this row could
not make deterministic.

N6's FIRST STAGING DID NOT COMPILE, and that is worth a line because it is the
trap this project keeps walking into. Written as `if (false) { ... }` it left
`entered` unused and failed `-Werror` with `compile_status=1`. A mutation that
fails to build reads exactly like a passing test; only the printed compile
status separated the two. Re-staged as a `(void)` cast of the same expression it
compiles, and it is green.

R1 and R2 are the reachability half. Both were run against
`ltx2 video: a render through the ABI emits a phase table that SUMS to wall`,
which enters through `vllm_video_engine_load` and `vllm_video_generate`, so what
they prove is that a RENDER reaches this code and not that the class works.

M3's verdict is itself a repair. On its first run the case aborted at a fatal
count `REQUIRE` placed above the loop, so the negative gap that same mutation
produces was never observed and two of the case's three assertions were unproven
while the case reddened. The count moved below the loop and became a `CHECK`.

### What the decomposition found on its first real render

On the 64x64x9 ABI render, at wall 21.89 s on a loaded box: a residue of
2.013 ms over 21 gaps, of which **0.944 ms -- 47% -- is charged to the
instrument itself**. That is the quantity #1439 asked to have beside the ratio,
and this is the first table that carries it.

The largest remaining gap is **`load.dit` -> `load.video_vae`, 0.627 ms**. That
is exactly the region `load.dit_config` names, which is item 1 of #1668. The
decomposition pointed at the next un-named region on its first run, from the
emitted file, with no script -- which is the whole of what #1571 asked for.

The load's prologue, which held 92% of this residue when #1556 measured it, no
longer appears: `519303d15` named it `load.open`.

## Outcome

**Closed: #1569, #1571 and
[#1755](https://github.com/mudler/vllm.cpp/issues/1755).** #1668 keeps items 1 to
3 and stays open. #1570, #1568 and #1567 stay open and are recorded under
`## Owed` with what each still needs.
[#1760](https://github.com/mudler/vllm.cpp/issues/1760) is NEW and open: #1755's
repair landed in full, and the gate over it holds two of its three halves. See
the survivor bullet below.

#1755 is the fresh review's own finding on this row: `### 6` repaired the file
emitter's clock ordering and its SIBLING kept the defect, at a scale the console
format could not print. It is filed and fixed in the same flow, per
`## Every change starts from an issue`, because the fix is two statements and
the argument for them is `### 6`'s argument.

What was measured, and what was rejected:

- **`residue <= 2 * instrument` was not re-proposed.** It was not re-measured
  either. `ltx25-phase-residue.md` `## Design` 3 already measured it across four
  sites and hundreds of runs, and re-deriving a settled negative result is the
  cost that record exists to remove.
- **The one new constant is 0.5 and it is not a tolerance.** Its derivation is
  in `### 6`: the two orderings differ by exactly one copy and one sort, so the
  defective value is at least 1.0 by definition and any constant inside `(0, 1)`
  separates them. The measurement's job was to confirm the separation, not to
  choose the number, and `### The operator's own re-run` measures that
  separation at **32x** — an honest maximum of 0.027616 over 45 runs against
  0.892, the worst run of the harder mutation, over 10. The bound at 0.5 sits
  18.1x above the honest side and 1.78x below the defective one. The three
  `head / min(copy, sort)` populations in `## Evidence` quote their own margins
  at their own maxima, 52x, 27.6x and 48.9x, and none of them is 237x.
- **The estimator carries the argument, not the constant.** Contention is
  one-sided, so a minimum over K probes strips the sporadic term from the honest
  side and cannot strip the deterministic term from the defective side. A gate
  whose noise is one-sided AWAY from red does not need a tail budget.
- **`serialize > 1e-5` is a precondition and not decoration.** #1569 exists
  because a three-record table made the two orderings indistinguishable, and a
  gate that silently loses its discriminator is a mute switch. Measured headroom
  is **11.6x to 21.0x**, not the 255x this bullet used to claim — and that range
  spans MORE THAN ONE load regime rather than pretending to be one number,
  because `serialize` is a wall-clock quantity and a single regime's spread
  understates it. Three populations, all of the case alone on the tree carrying
  the `### 10` repair, all on x86_64:

  | population | loadavg | `serialize`, min to max | headroom over `1e-5` |
  |---|---|---|---|
  | 6 consecutive runs | 11 to 21 | 1.2008e-4 to 1.3535e-4 | 12.0x to 13.5x |
  | a second box and load regime | -- | 1.3094e-4 to 2.1012e-4 | 13.1x to 21.0x |
  | 8 consecutive runs, the session that recorded this | 8.35 to 8.41 | 1.15782e-4 to 1.29198e-4 | 11.6x to 12.9x |

  Every run in the third population printed
  `test cases:  1 |  1 passed | 0 failed | 7 skipped` and
  `assertions: 14 | 14 passed | 0 failed`, which is what rules out a `-tc`
  filter that matched nothing and reported `SUCCESS!`. A fresh review measured
  the same quantity at 1.22e-4 to 1.41e-4 independently, inside the union above.
  **The claim the bullet makes is unchanged by any of it**: the headroom is an
  order of magnitude and it is not two.
- **The `### 10` gate carries a KNOWN SURVIVOR and it is recorded rather than
  closed.** `M-SITE-MID` — the console block slid below `ByStart` and `Sum` and
  left above the JSON build — is GREEN in 19 of 20 runs, so #1755's own class
  survives in the two-thirds of the repair arm (A) does not cover. The reason is
  structural and is in `### 10`: arm (A)'s 16000-record table is sized to the
  JSON-build discriminator at 6.54 to 7.26 steps of `%10.3f`, while the copy and
  the sort at that same size are 0.39 to 1.03 steps and straddle the bound. The
  fix a reader would reach for first — enlarging arm (A)'s table — is measured
  shut at about a gigabyte of resident set, and a new wall-clock tolerance is
  forbidden. **Recording the limit is the correct outcome here**, and
  [#1760](https://github.com/mudler/vllm.cpp/issues/1760) owns what would settle
  it.
- **The gap decomposition is arithmetic, not a measurement.** That is why it is
  the strongest thing in this row. **Eight** of the twenty-three rows in
  `### The mutation table` are caught by its assertions, and none of those
  assertions contains a clock: `M2`, both `M3` arms, `M7`, `M8`, `NEND`,
  `NCURSOR` and `NREVSORT`. `R1` is a ninth against the render case.

What a reader should NOT conclude: that either floor is now honest at 21 B
scale. `leaves >= 0.95 * wall` still decides by box load at small wall and still
permits minutes of un-named time at large wall. This row gives a reader the
quantity that would settle it -- the residue, split by region, with the
instrument's own share subtracted -- and asserts an identity over it. #1439
stays open because a quantity a reader can see is not yet a budget a gate holds.

## Now

Landed. #1569, #1571 and #1755 closed; #1668 keeps items 1 to 3.
