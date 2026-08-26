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
use -- and the comment is still in the tree, above the `denoise` entry of the `carrying`
table in `CheckRenderPhases` (`tests/vllm/multimodal/test_ltx2_video.cpp`; it
sat at :4324-4331 when this was written and moves with the file, which is why
the range is no longer quoted):

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

## What landed, and what it is gated by

**The three anchors and the counter land here.** #1668, on base `1724be38e`.
Written in the present tense of the merge that carries this file: while the pull
request is open they are on `row/LTX25-PHASE-RESIDUE-1668` and not on `main`.
The four items this row measured are now in two states rather than one, so the
earlier "nothing in this table is implemented" line above `## Owed` is retired
rather than edited: it was true when it was written and it is not
true now.

| item | state |
|---|---|
| the load prologue | landed as `load.open` by `519303d15` (#1622), under `LTX25-DEVICE-RESIDENCY` |
| `Record::instrument_seconds` | landed by `be432e8e3` (#1711), under `LTX25-PHASE-INSTRUMENT`, with its docs |
| `load.dit_config` | **landed here** |
| `artifacts.mux` | **landed here** |
| `denoise.update` + `Ltx2ConditioningTrace::sampler_updates` | **landed here**, first-order arm only |

### The gate is STRUCTURAL, and that is the whole point

`## Design` 3 withdrew `residue <= 2 * instrument` because a seam's honest share
is a property of the box, and three fresh reviews measured its tail. Nothing
here re-proposes it, and **neither existing wall-clock ratio moved**. The two
seam anchors are held by `CheckSeamAnchor`, which carries **no constant at all**.
Its clauses are positions:

- the anchor is emitted exactly once, is not a span, and is not `nested`;
- it opens at or after its predecessor's end, and closes at or before its
  successor's start;
- no other leaf lies inside the window it claims.

`artifacts.mux` has no successor — it closes one statement before
`WritePhaseLog` reads the clock — so its far-side clause is replaced by the
stronger one that nothing follows it: **it is the last leaf of its render**.

Under a correct placement every clause holds by construction. Under a MISPLACED
one at least one fails outright, whatever the clock did — an anchor left open
across the next leaf marks that leaf `nested` and removes it from
`sum_leaf_seconds` altogether, and an anchor moved onto a neighbour is found
inside that neighbour's window.

**This does NOT extend to an anchor that names NOTHING**, and an earlier draft of
this section said it did. A zero-width window satisfies every clause above,
because a window with no interior contains no leaf and a collapsed anchor still
opens after its predecessor and still precedes its successor. That is
[#1884](https://github.com/mudler/vllm.cpp/issues/1884), measured twice and
below.

`denoise.update` is held by assertion (0), the record count, against
`Ltx2ConditioningTrace::sampler_updates`. That counter is maintained by the
RENDER, inside the scope it counts, for the reason `video_decode_chunks` is: a
denominator derived from the phase table cannot falsify a phase table.

### `sampler_updates` is not a second name for `dit_evaluations`

The two agree on the first-order arm — one update per evaluation — so that arm
cannot tell a real counter from an alias. The **res_2s arm can**, and it is in
the gate for that: `res2s_two_stage` runs its post-process and step inside
`Ltx2Res2sDenoisingLoop` behind `Ltx2Res2sHooks`, which the first-order
statement never reaches, so it counts 7 and 11 evaluations against **zero**
updates. Mutation M4 — the counter incremented beside `dit_evaluations` in the
shared `Evaluate` lambda, which is what an alias actually looks like — is green
on the first-order arm and red on that one.

### Mutations, all six detected

Run on the immutable head, each with its anchor asserted UNIQUE before it was
applied, its application verified by hash, its build's exit code checked, and
the tree restored and re-hashed byte-for-byte afterwards. The harness reported
its own defects first, which is why it exists: an early `M2` was rejected as
`ANCHOR-NOT-UNIQUE` (2 occurrences), an early `M4` as a semantic no-op that read
green, and an early `M5` as `BUILD-FAILED`, and a build failure would have left
the previous binary printing the previous green.

| id | guarantee attacked | verdict |
|---|---|---|
| M1 | `load.dit_config` closes before `load.video_vae` opens | red, 4 assertions |
| M2 | `artifacts.mux` is the render's LAST leaf, not a window around the audio write | red, 4 assertions |
| M3 | `denoise.update` is emitted once per sampler update | red, 2 assertions |
| M4 | `sampler_updates` counts the sampler, not evaluations | red, 3 assertions |
| M5 | reachability: the `load.dit_config` production call site | red |
| M6 | reachability: the `artifacts.mux` production call site | red |
| **M7** | **the anchor still WRAPS the work: collapsed to zero-width. Filed on `artifacts.mux`; a fresh review then reproduced it on BOTH anchors at once** | **GREEN -- NOT detected, #1884** |

### It is not load-flaky, which is the property it was built for

The reason `## Design` 3 withdrew the instrument-relative bound is that its
value moves with the box, and this row's whole history is gates that red under
load. A structural gate should not have that failure mode, and this is the
measurement rather than the argument: **8 runs of 8 green at loadavg 25.98 to
37.92**, on one binary, while a second build was saturating the machine. Small
counts differ between runs (794 to 798 assertions) because the "seam holds
nothing else" clause iterates over the leaves the render actually emitted.

That is a lower bound on stability and not a distribution. `## Design` 3 also
records why it cannot be more than that: a 20-run sample of a scheduler-
dependent quantity on this table did not see the tail that decided the gate. The
claim here is narrower and does not need one — these assertions read ORDER and
COUNT, and neither has a tail.

M5 and M6 delete the production call site and leave a tree that COMPILES, which
is what `.agents/reachability.md` asks for: the gate has to be what notices.
Both anchors sit inside `Ltx2VideoEngine::Load` and `::Generate`, rather than in
a type a test constructs. Stated precisely, because an earlier draft was loose
about it: the case that reds on M5 and M6 enters through
`vllm::multimodal::LoadVideoEngine` and `VideoEngine::Generate` — the loader and
the production `Generate` it hands back. `.agents/reachability.md` names the
loader in those words; the `Generate` is the call the loader exists to reach. It is
NOT a C-ABI render, and a reader should not go looking for one. The C ABI
entry point `vllm_video_generate` is `src/capi/vllm_c.cpp:1623`, and it reaches
the same `Ltx2VideoEngine::Generate` at `:1668`; the server reaches it at
`src/vllm/entrypoints/openai/server_main.cpp:1616`. So the anchors are on the
shipped path by construction, and the separate `SUMS to wall` case is the one
that renders through `vllm_video_generate`. Both anchors were read out of the
file rather than recalled: an earlier draft of THIS sentence cited
`vllm_c.cpp:1667`, which is a blank line, and nothing in this tree gates a
citation.

### M7 is GREEN, and it is the honest limit of a constant-free gate

**A position is not a magnitude.** Close `artifacts.mux` immediately after
opening it and drop the late `Close()`, and all 796 assertions pass: the count
is 1, it is neither span nor nested, it still opens after `artifacts.audio`, it
is still the last leaf, and a zero-width window contains no other leaf
VACUOUSLY. The anchor reports ~0 s and the tail is back in
`unaccounted_seconds`, with the table green and a name on nothing. The same
argument applies to `load.dit_config`.

That is [#1884](https://github.com/mudler/vllm.cpp/issues/1884), and it is
disclosed rather than closed because the obvious closure is the thing this row
already measured shut twice.

**A fresh review found it independently and measured what it costs**, which is
the number this row should carry rather than the argument. Collapsing BOTH
anchors onto their own `Close()` — the exact shape of a refactor that hoists the
work out from under an anchor — is green on the focused case AND on the full
suite, 102 of 102 cases (4668 assertions on that run; the total moves a few
either way between runs, for the reason the counts above do), on a
line-count-preserving variant
so the reader-anchor ledger is undisturbed:

| quantity, `attribution_multichunk/phase-log.json` | head | collapsed |
|---|---:|---:|
| `load.dit_config` duration | 0.000109239 s | **0.000016543 s** |
| gap `load.dit` -> `load.dit_config` | 0.000003296 s | **0.000141781 s** |
| gap `load.dit_config` -> `load.video_vae` | 0.000001042 s | **0.000087407 s** |
| gap `artifacts.audio` -> `artifacts.mux`, r1 / r2 | 0.000000983 / 0.000003287 s | **0.000036901 / 0.000066776 s** |

So the collapse hands about **229 us back to the residue on a fixture whose
whole residue is about 1.2 ms** — roughly a fifth of the thing this row exists
to name — while the anchor that was supposed to name it measures 16.5 us, which
is instrument cost. The `(4b)` gap gate cannot see it either: `gap_count ==
leaf_records + 1` and `gap_total == unaccounted` both still hold, because the
collapse only SPLITS one gap into two.

**The next traceable hypothesis, named rather than left open.** The same review
proposes a discriminator whose denominator is NOT the instrument, which is what
separates it from the bound `## Design` 3 withdrew: the anchor's own extent
against the seam its two neighbours define, both measured in the same run —
`(end - start)` against `(next_start - previous_end)`. Honest, that ratio is
about 1 because a correctly placed anchor covers the seam except for two
boundaries; collapsed, it is about 0. That is a wide separation of the kind the
`decode.video.vae` floor already exploits, and it does not ask how many seconds
a seam SHOULD hold, which is the question that made the instrument-relative
bound a property of the box. It still needs a constant, so it is a candidate and
not a conclusion, and it belongs to #1884 with its own red-first measurement and
its own fresh review. Recorded here so the next attempt starts from it instead
of re-deriving it.

**That measurement has since been made, and the candidate FAILED it.** The
section below carries both arms. This paragraph is kept because its reasoning is
what the measurement tested; read it as the hypothesis and not as the
recommendation it was.

What stays banned here is the shape `## Design` 3 withdrew: a floor whose
denominator is the INSTRUMENT, the wall, or a written-down number of seconds.
Each of those asks how many seconds a seam SHOULD hold, and the honest answer is
a property of the box. The seam-extent ratio above asks a different QUESTION --
what fraction of THIS seam, in THIS run, did the anchor cover -- so the
prohibition never reached it by its own words. An earlier draft of this
paragraph banned "a share floor here" without that qualification, which forbade
the candidate the paragraph had just proposed, and `## Stop conditions` points a
reader at this section as its authority.

**The measurement below shows why escaping the prohibition by its wording was
not enough.** The ratio is conditioned on the same QUANTITY the prohibition is
about -- the un-instrumented remainder of an instrument boundary -- and that
conditioning, not the question asked, is what decided `residue <= 2 *
instrument`. Ask of any future candidate what its denominator DILATES WITH under
contention, not only what it means.

### The seam-extent ratio was MEASURED, and it does not separate

The candidate above was measured before it was adopted, and **it does not work
as proposed.** Its stated premise is false and its distributions overlap, so it
is withdrawn rather than left standing as "where to start". Row
`LTX25-PHASE-RESIDUE`, issue #1884, 2026-08-25, one box, one binary pair.

Method: `test_ltx2_video -tc="ltx2 video: the three carrying phases contain their
work and the load keeps its order"` under `VLLM_KEEP_TEST_ARTIFACTS=1`, reading
`attribution_multichunk/phase-log.json` from each run. The HONEST arm is the
unmutated head; the COLLAPSED arm is `M7` applied to BOTH anchors, line-count
preserving, built and run from the same tree. `(end - start)` over
`(next_start - previous_end)`, exactly as proposed.

Two things about the method a reader has to have. **`artifacts.mux` has no
successor**, so its far side is the enclosing `generate` span's end. That is NOT
what the position clause uses -- `CheckSeamAnchor` is called with `next=""` for
this anchor and its far-side branch never executes -- it is the tightest far
boundary available, chosen because tightening the denominator raises BOTH arms
and raises the honest one more, so it is the reading most favourable to the
candidate. **And the honest arm is two sub-populations**, 22 runs at a wall
median of 1.12 s and 45 at 3.38 s, against the collapsed arm's 1.20 s. They are
not contention-matched, so both are reported and every conclusion below is
checked in each separately.

| arm | anchor | n | min | p05 | median | max |
|---|---|---:|---:|---:|---:|---:|
| honest, 67 runs | `load.dit_config` | 67 | **0.0626** | 0.7366 | 0.8057 | 0.9646 |
| collapsed, 25 runs | `load.dit_config` | 25 | 0.0428 | 0.0473 | 0.1089 | **0.2122** |
| honest, 67 runs | `artifacts.mux` | 134 | **0.0156** | 0.3440 | 0.5977 | 0.8779 |
| collapsed, 25 runs | `artifacts.mux` | 50 | 0.1414 | 0.1474 | 0.2989 | **0.4766** |

**For `artifacts.mux` no constant exists at all, and that is not a contention
artefact.** The collapsed range [0.1414, 0.4766] lies ENTIRELY INSIDE the honest
range [0.0156, 0.8779]. Any `k` at or below 0.4766 passes a collapsed render;
any `k` above 0.0156 reds an honest one. Held to the collapsed maximum, the
honest arm reds **11 of 44 in the quieter sub-population and 15 of 90 in the
busier one** -- both regimes, so the overlap is a property of the anchor and not
of the box's load.

**For `load.dit_config` the overlap IS contention, and it is the withdrawn
bound's contention.** A `k` in (0.2122, 0.7187] catches every collapse in this
population and reds 2 of 67 honest renders, 3.0% -- the same order as the
4-in-45 that withdrew `residue <= 2 * instrument`. Split by regime, that is
**0 of 22 quieter and 2 of 45 busier**: clean where the box is calm, and a tail
where it is not, which is exactly the shape that is not gateable.

**The premise "honest is about 1" is false for `artifacts.mux`, and it is false
without any tail.** Its honest median extent is 0.5977 because the anchor is
tiny: median duration **14.5 us** against a median `instrument_seconds` of
**7.0 us** on the same records, and per record the ratio
`instrument_seconds / duration` runs **min 0.394, median 0.495, max 0.997**.
Half of what this anchor measures is this instrument's own charge inside it.
`### 1`'s note beside the anchor already said it was "close to naming its own
cost"; this is that statement as a distribution. A seam whose non-anchor part is
two instrument boundaries cannot have a near-1 extent ratio when the anchor is
the size of one boundary.

**And the tail's mechanism is the withdrawn bound's mechanism, seen directly in
the table.** The two honest `load.dit_config` outliers are not small anchors;
they are ordinary anchors beside a preempted boundary:

| run | anchor | its duration | the flanking GAP before it | ratio |
|---|---|---:|---:|---:|
| `h2-5` | `load.dit_config` | 234 us | `load.dit` -> it, **3473 us** | 0.0626 |
| `h2-1` | `artifacts.mux` r2 | 39 us | `artifacts.audio` -> it, **2446 us** | 0.0156 |

Both gaps are ONE instrument boundary -- a `Close` stamp followed by an `Open`
stamp -- and both are ~3 us on a quiet run. A descheduling landed in them and
they grew a thousandfold while the anchor did not move. `## Design` 3 states
exactly this: "the un-instrumented remainder of a boundary dilates FASTER than
the instrumented part under contention". The seam-extent ratio puts that
remainder in its denominator, so `anchor / (anchor + gaps)` is a monotone
function of `gaps / anchor` and inherits the same tail. It asks a different
QUESTION than `residue <= 2 * instrument` and it is conditioned on the same
QUANTITY, and the conditioning is what decided the earlier bound.

So the paragraph above is superseded on its conclusion and kept for its
reasoning.

### What the same data says to try NEXT, measured once

A fresh review of the measurement above mutated its CONCLUSION rather than only
checking its arithmetic, and found that the negative result is a property of the
proposed DEFINITION and not of the family. It does not reach a clause that
subtracts the instrument's own charge from the numerator before dividing. **This
is a candidate with one measurement behind it, not a conclusion**, and #1884
still requires it to bring its own constant, its own cross-box distribution, its
own red-first mutation and its own fresh review.

With `net = duration - instrument_seconds`, on the same 67 honest and 25
collapsed runs:

| discriminator | `load.dit_config` honest / collapsed | `artifacts.mux` honest / collapsed | honest reds at the collapsed max |
|---|---|---|---:|
| `extent / seam`, as proposed | [0.0626, 0.9646] / [0.0428, 0.2122] | [0.0156, 0.8779] / [0.1414, 0.4766] | 2/67 and **26/134** |
| `net / seam` | [0.0460, 0.9187] / [0.0024, 0.0041] | [0.0030, 0.4637] / [0.0039, 0.0298] | 0/67 and 2/134 |
| `net / (net + gap_after)` | [0.7578, 0.9922] / [0.0028, 0.0060] | [0.0252, 0.7355] / [0.0183, 0.0677] | **0/67** and **1/134** |

The last row separates `load.dit_config` by more than two orders of magnitude
with no honest red in 67, and leaves `artifacts.mux` with ONE honest red in 134.
That red is named rather than averaged away: it is `h2-38` render 2, where the
anchor's whole 2968.0 us extent was its own `instrument_seconds` of 2957.8 us --
the boundary itself was descheduled, so `net` is 10 us of a 2968 us window. That
is the same mechanism again, one level in, and it is what the warning above
predicts. Whether 0.75% is acceptable is a question for the row that adopts it,
with a population large enough to see the tail; this row measured it once and
records it so the next attempt starts from the definition that worked rather
than from the one that did not.

**What is NOT resolved by any of this**: `CheckSeamAnchor` as it stands proves a
position, and a position clause cannot bound a magnitude. Closing #1884 needs a
magnitude clause -- the row above, or an anchor INSIDE the callee whose window
cannot be narrower than the work because the work is its body -- and #1439's
question sits above both.

The hole itself is the same shape as the `decode.audio.mel` note in `test_ltx2_video`
("a partial transfer ... passes 0.50 and is not detected here. Closing that
needs a scope INSIDE the callee") and as #1568 one level down. Three
appearances, one repair, and the repair is an anchor inside the callee or a
bound on a quantity the scheduler cannot move -- which is #1570's open question
and, above it, #1439's.

## Owed

| Issue | Owed |
|---|---|
| [#1668](https://github.com/mudler/vllm.cpp/issues/1668) | **CLOSED 2026-08-25, after each of its four items was re-verified on `origin/main` at `ced0ab639`.** `load.dit_config` at `src/vllm/multimodal/ltx2_video.cpp:979`, `artifacts.mux` at `:5440`, `denoise.update` at `:4576` and `Ltx2ConditioningTrace::sampler_updates` at `:4577` and `include/vllm/multimodal/ltx2_video.h:920` land with this file. The fourth item, `Record::instrument_seconds`, is at `include/vllm/multimodal/render_phase_log.h:70` and landed separately as `be432e8e3` (#1711) under `LTX25-PHASE-INSTRUMENT`, so the issue's OWN TEXT is stale on that item and closing it does not redo the work. See `## What landed, and what it is gated by`. **The scope note's last obligation was checked rather than assumed**: it asked that `denoise.update` reach "the phase names published in `docs/models/ltx-2-5.md`", and that document publishes NO phase names -- a grep for a leaf name over `docs/` returns MiniMax-Music3 and MiniMax-H3 rows only -- so nothing is outstanding there and nothing was silently skipped. What this issue does NOT close, and did not claim to: the res_2s arm (#1567), the seconds-transfer gate (#1568) and the magnitude hole its own gate left (#1884). It did not close #1570 either, although #1570 has since been CLOSED on the forge by `be432e8e3` (#1711) and the rows below that still call it open are stale. The earlier reference implementation stays readable at `refs/pull/1556/head` = `b45ea3bbb`; it was not reused, and the anchors here were written and gated fresh |
| [#1567](https://github.com/mudler/vllm.cpp/issues/1567) | **OPEN, and its recorded blocker is FALSE.** The name `denoise.update` now exists on the first-order arm, so this issue is no longer waiting on a name that nothing defines -- it is the SECOND arm of an anchor that ships. `Ltx2ConditioningTrace::sampler_updates` reads ZERO on res_2s, and `test_ltx2_video` asserts that zero, so the arm's absence is measured rather than assumed and any hook that lands has a counter to be checked against. **THE BLOCKER BOTH THIS ROW AND THE FORGE ISSUE RECORD -- that no gate in this tree renders the res_2s arm, so an anchor beside the first-order one would land dead -- IS FALSE, and one command falsifies it.** `test_ltx2_video -tc="ltx2 video: the HQ pipeline evaluates the DiT twice per step"` renders `res2s_two_stage` TWICE -- in a case that makes four renders, two per arm -- through `LoadVideoEngine` and `VideoEngine::Generate`, which is a production entry point, and under `VLLM_KEEP_TEST_ARTIFACTS=1` it leaves `hq3/phase-log.json` and `hq5/phase-log.json` on disk. Both tables carry a `denoise` leaf and **7 and 11 `denoise.step` records** -- the res_2s loop's own `2n + 1` evaluation counts -- and no `denoise.update`. So the phase instrument is ALREADY live inside that loop, the arm ALREADY writes a table a gate can read, and the update anchor is the only thing missing. That case has existed since `4d7748646` (#1125), which predates this issue, so the claim was never true of any tree. What is still owed: the res_2s arm's `denoise.update` anchor and its count assertion, and nothing else. `Ltx2Res2sDenoisingLoop` runs its own post-process and step behind `Ltx2Res2sHooks`, so the anchor needs a hook rather than a statement. It lives in `ltx2_samplers.cpp`, is declared in `ltx2_samplers.h` beside the hooks struct, and is called from `ltx2_video.cpp`. **NOT `ltx2_res2s.cpp`**: #1556's spec named that file and it has never existed here, which `git log --all --diff-filter=A` confirms; #1567's forge text names no file at all, so the wrong anchor came from the spec rather than from the issue |
| [#1568](https://github.com/mudler/vllm.cpp/issues/1568) | **UNBLOCKED by #1668 and still open, and its measured reason is now IN THE TREE.** The transfer this issue describes was hypothetical while `denoise.update` did not exist; the name ships now and `denoise` is a multi-part leaf, so (1b) is no longer vacuous on it. **(2b) still is**, and an earlier draft of this row said otherwise: both of this leaf's floors are 0.0 and the (2b) loop skips a 0.0 floor, so it executes zero CHECKs here. `decode.audio` remains the only leaf (2b) asserts anything about. `part_min_coverage` for `denoise.update` is **0.0**, and the note beside it in `test_ltx2_video` carries the reason rather than a constant: the honest share runs 0.45% to 11.15% across four boxes and the transfer puts it at ~0%, so the two distributions overlap and any floor that reds the transfer also reds an honest render this row has produced. Not closed, and deliberately not closed by a number. What is still owed: the `denoise.step` / `denoise.update` seconds transfer. (1b') compares `start_seconds` only, so leaving `denoise.step` open across the post-process and emitting `denoise.update` empty after it preserves the alternation, both counters, containment, non-overlap, exclusivity, (1c) and (2), while moving 100% of the decomposed seconds onto one name. No (2b) floor separates it: the honest share of `denoise.update` runs 0.45% to 11.15% across four boxes and a transfer puts it at ~0%. Closing it needs an anchor INSIDE the callee. **RE-VERIFIED BY MUTATION on `ced0ab639`, 2026-08-25, rather than by reading the assertions.** Mutation `T1`, the empty-sibling form of this issue's `R1b`: `denoise.update` collapsed onto its own `Close()` at the same statement, line-count preserving, so the record is still emitted once per step and still sits between two `denoise.step` records while naming ~0 s. Focused case GREEN 5 runs of 5, and a fresh review reproduced it independently at 3 of 3 after verifying the mutated BINARY differed by hash from the unmutated one. The assertion total is not quoted, because it moves by a few between runs for the reason `### M7` already records. The alternation, both record counts, containment, non-overlap, exclusivity, (1c) and (2) all survive it, and (2b) executes zero CHECKs on this leaf because both its floors are 0.0. The issue is open on measured evidence |
| [#1569](https://github.com/mudler/vllm.cpp/issues/1569) | **CLOSED by `LTX25-PHASE-INSTRUMENT`** ([`ltx25-phase-instrument.md`](ltx25-phase-instrument.md)), which gates it over an 8000-record table where the copy and the sort are a measurable event, against a discriminator measured in the same run rather than written down as a constant. On the tree that lands, restoring `main`'s clock order (`M1`) reds it 10 runs of 10 at ratios of 2.080 to 3.418, and the partial regression (`N11`) reds it 10 of 10 at 0.892 to 1.145, against a bound of 0.5 and an honest 45-run maximum of 0.027616 at loadavg 19-26 — 0.018123 in a higher 56-113 regime. The earlier "1.004 against 0.0042" belonged to the WITHDRAWN one-number `copy + sort` budget, which a fresh review broke and `### 6` replaced with `min(copy, sort)`. What it originally owed: a gate on `WriteJson`'s clock ORDERING, **measured green under its own mutation**. Restoring the old order left the conservation case GREEN 10 of 10, at `wall 0.0608987s, unaccounted 0.000534223s, table charge 0.000301655s`, because the copy and sort of a three-record table are nanoseconds. Gating it needs a table with enough records for the sort to be measurable |
| [#1570](https://github.com/mudler/vllm.cpp/issues/1570) | an upper bound on the instrument's own share of a leaf. `uncovered <= 2 * leaf_instrument` is stricter than the floor it replaces only while `leaf_instrument` stays small, and nothing bounds it. Moving the DiT `Tick` out of `Evaluate` would charge ~110 flushed writes to `denoise` and widen the gate while printing a small number |
| [#1571](https://github.com/mudler/vllm.cpp/issues/1571) | **CLOSED by `LTX25-PHASE-INSTRUMENT`** ([`ltx25-phase-instrument.md`](ltx25-phase-instrument.md)). `phase-log.json` carries `gaps`, and the gate over it is an accounting identity rather than a tolerance: the gaps add to `unaccounted_seconds` by construction. On the fixture render it immediately named the NEXT region, `load.dit` -> `load.video_vae` at 0.627 ms, which is the `load.dit_config` anchor #1668 owes. What it originally owed: a per-gap decomposition IN the emitted table. The 92% region above was found with a scratch script; a reader of `phase-log.json` still cannot see it without one, and the same investigation will be re-derived the next time the residue moves |
| [#1572](https://github.com/mudler/vllm.cpp/issues/1572) | assertion (1c)'s span slack reds intermittently on `main` — `decode.video` at `0.00256913` against a `0.00075` bound, 3.4x. Pre-existing from `6b48edb2c` and not this row's. **STILL OPEN, and two further shapes of that bound are now MEASURED SHUT.** `LTX25-DEVICE-RESIDENCY` built a fifth shape (`4 x` the worst boundary a 1 kHz sampler saw across the whole case) and a sixth (`4 x` the worst inside the record's OWN window) and withdrew both: the fifth lets one 200 ms descheduling of the sampler thread turn a real 20 ms un-named phase from red 9 of 9 into a GREEN case, and the sixth reds an unmutated tree 10 times in 45 consecutive runs at loadavg 21.8-61.5 -- of which **5 in 45 is the defensible figure**, because a second mechanism was identified and repaired while that population was still running, so its reds were measured on a binary that predates its own repair. See `.agents/specs/ltx25-device-residency.md` `### The span-slack bound, FIFTH and SIXTH shapes` for both distributions and for the one hypothesis that has not been tried |
| [#1619](https://github.com/mudler/vllm.cpp/issues/1619) | **the `merge=union` driver duplicates a row, MEASURED on this row's own merges.** Both sides appended before the same trailing anchor rather than at the true end, so the driver concatenated two regions that each carried `#1546` and the resolved index held it TWICE, byte-identical, at 538 lines where the correct union is 537. `git merge-tree` called that merge clean and `check-issue-index-append-only.py` passed it, because a duplicate is an ADDITION and that checker only collects removals. `check-agent-record.py` did NOT pass it -- a claim #1556's spec made and this row REFUTED by reproduction: regenerating the raw driver output and running that same tree's checker returns rc=1 with `issue #1546 listed twice`, and the refusal has existed since `8dd6508da` (2026-08-09), before the merge. So the blind gate is exactly one checker, not two, and the gap is narrower than #1556 recorded. The de-duplication half is CONDITIONAL, and the condition is what #1556's spec omitted: the checker reds a repair only when the DUPLICATE IS ALREADY IN THE BASE. Measured at three pairings -- `--base e2a9e035d` against the real canonical 537-line file rc=0, against a synthetic 537 rc=0, and `--base <committed 538> --head <537 de-dup>` rc=1. It diffs `merge-base..HEAD`, so when the base predates the duplicate the addition and the removal CANCEL and it passes. Since `origin/main` is preflight's base, and is the shape this branch used, the gate does NOT red someone who repairs driver output before committing it -- only someone repairing a corruption that already landed. The same range property is why relocating a base-reachable row DOES red it: moving row `#168` to the end gives rc=1 and a `removed:` line naming it. So "de-duplicating in place FAILS the checker", as #1556's spec put it, is false unqualified and true once the duplicate is base-reachable. #1556's spec added that the same driver dropped `#838` on a later re-merge, making this a recurring class; that is WITHDRAWN as unreproducible. Re-running `git merge-file --union` at every later merge where `#838` was on a side leaves it present in all of them, and `git log -S` finds it absent from no committed state -- mechanically a union driver cannot drop a line that is an addition on one side. If it ever went missing, that points at a wholesale take-ours resolution rather than at the driver |
| [#1884](https://github.com/mudler/vllm.cpp/issues/1884) | **`CheckSeamAnchor` proves POSITION, not MAGNITUDE.** Filed by the change that wrote the gate, against its own work, and measured: mutation M7 -- `artifacts.mux` closed immediately after it opens, the whole tail un-named again -- passes 796 of 796 assertions. Every clause survives a zero-width window, the containment clause vacuously. Third appearance of one shape, after the `decode.audio.mel` partial transfer and #1568. **Must NOT be closed by a share floor against the INSTRUMENT or the wall**: that is `residue <= 2 * instrument` with a different name, and `## Design` 3 is the measured record of why it does not work. **The seam-extent ratio that this row named as where to START is now MEASURED AS PROPOSED, and it is SHUT.** 67 honest runs against 25 collapsed runs, one box, one binary pair: for `artifacts.mux` the collapsed range [0.1414, 0.4766] lies entirely inside the honest range [0.0156, 0.8779], so NO constant separates them -- and in BOTH load regimes separately, 11 honest reds in 44 quieter runs and 15 in 90 busier ones, so that overlap is a property of the anchor rather than of the box. For `load.dit_config` a constant in (0.2122, 0.7187] reds 2 of 67 honest renders, 0 of 22 quieter and 2 of 45 busier, which is contention and is the withdrawn bound's contention. Its premise fails too -- `artifacts.mux`'s honest median extent is 0.5977, not "about 1", because that anchor has a median duration of 14.5 us against a median `instrument_seconds` of 7.0 us, and per record `instrument_seconds / duration` runs 0.394 to 0.997 with a median of 0.495. The tail's mechanism is the withdrawn bound's mechanism: the honest outliers are ordinary anchors beside a preempted instrument boundary (234 us beside a 3473 us gap; 39 us beside a 2446 us gap), and the ratio's denominator holds that un-instrumented remainder. See `### The seam-extent ratio was MEASURED, and it does not separate`. **A fresh review then mutated that CONCLUSION and narrowed it**: the negative result belongs to the DEFINITION, not to the family. Subtracting the record's own `instrument_seconds` from the numerator gives `net / (net + gap_after)`, which on the same runs separates `load.dit_config` by two orders of magnitude with 0 honest reds in 67 and leaves `artifacts.mux` with 1 in 134 -- and that one red is a record whose whole 2968 us extent was its own 2957.8 us instrument charge. One measurement, not a conclusion; it still owes its own constant, cross-box distribution, red-first mutation and fresh review. See `### What the same data says to try NEXT, measured once`. What is NOT resolved: `CheckSeamAnchor` proves a POSITION, and a position clause cannot bound a magnitude. The alternatives remain that row, an anchor INSIDE the callee, and #1439's question above both |
| [#1439](https://github.com/mudler/vllm.cpp/issues/1439) | **NOT closed by this row, and it must not be.** See `## Risks and decisions` D4 |
| [#1470](https://github.com/mudler/vllm.cpp/issues/1470) | `test_ltx2_video` false-redded once on `main` under load and the failing case's identity was never captured. Untouched by THIS row, and **the identity and the rate are now measured** by `LTX25-DEVICE-RESIDENCY`: 1 red in 120 runs of the containment case at loadavg 40-155, on `artifacts.frames` render 2, where a 67.55 ms descheduling between `ppm_phase.Close()` and the leaf's destructor left 67.55 ms of a 74.87 ms leaf uncovered. It reds `covered >= 0.50 * leaf_seconds`. The same row also measured, and WITHDREW, the obvious repair: an instrument-relative second arm on that floor makes mutation `B-empty-ppm` — the writer's anchor opened after the write loop instead of around it, coverage 98% to 1.1% — pass. `artifacts.frames` is 0.2-7 ms on this fixture and one boundary on that host is 0.3-1.2 ms pinned to two idle cores, so no allowance built from the boundary is smaller than the leaf. The repair is an anchor, not a threshold |

## Stop conditions

- Do not widen either ratio to close a red. Name the phase, or leave the red and
  file the gap. D1.
- Do not re-propose `residue <= 2 * instrument` without reading `## Design` 3
  first, and never accept a 20-run distribution as evidence about it.
- Do not close #1439 from this row. D4.
- Do not close #1884 with a share floor whose denominator is the INSTRUMENT,
  the wall, or a written-down number of seconds. That is the withdrawn bound
  wearing a different name. `## Design` 3, and D1. **The seam-extent ratio escaped that
  prohibition by its wording and was MEASURED SHUT anyway, AS PROPOSED**, 67
  honest runs against 25 collapsed: for `artifacts.mux` the collapsed range sits
  entirely inside the honest one in BOTH load regimes, so no constant exists.
  Do not re-propose `extent / seam`. Read `### The seam-extent ratio was
  MEASURED, and it does not separate` first, and ask of any successor what its
  denominator DILATES WITH under contention rather than what it means. **That
  result does NOT reach the whole family**, which an earlier draft of this line
  claimed and a fresh review falsified from the same data: subtracting the
  record's own `instrument_seconds` from the numerator changes it, and
  `### What the same data says to try NEXT, measured once` carries the numbers.
- Do not close #1884 by adding a POSITION clause to `CheckSeamAnchor`. A position
  clause cannot bound a magnitude, whatever else it proves. A magnitude clause
  there is not forbidden -- it owes the measurement the row above began.

## Now

**#1668 IS CLOSED, and #1567, #1568 and #1884 are open with what each still
owes named.** The four items #1668 listed were re-verified on `origin/main` at
`ced0ab639` before it was closed, and `## Owed` carries the `file:line` for each.
The two siblings were re-checked rather than closed alongside it: #1568 by
running its own transfer mutation, which is still green; #1567 by rendering the
res_2s arm, which falsified the "no gate renders that arm" blocker this spec,
the forge issue and a comment in `test_ltx2_video` all carried. That comment is
corrected here, in the same flow, because a record correction that leaves the
false sentence in the product tree has repaired nothing. #1884's named candidate
discriminator was measured across 92 runs and does not separate as proposed;
what the same data says to try instead is recorded beside it. **No assertion
changed**: every constant, floor and tolerance in `test_ltx2_video` is
byte-identical to `ced0ab639`, and the only edit outside `.agents/` is a comment.

**#1570 IS CLOSED**, by `be432e8e3` (#1711) on 2026-08-23. The paragraph below
counts it among five open issues and that count is stale; the open ones under
this spec are #1567, #1568, #1572 and #1884, and #1439 above them.

**The three anchors #1668 owed land with this file.** `load.dit_config`,
`artifacts.mux`, `denoise.update` and `Ltx2ConditioningTrace::sampler_updates`,
from base `1724be38e`, held by a structural gate that carries no
constant, with six mutations detected and both wall-clock ratios untouched.
`Record::instrument_seconds`, the fourth item, had already landed as `be432e8e3`
(#1711). #1567 and #1568 are UNBLOCKED by that and stay open; neither is closed
here. The history below is kept because it is why the anchors have the shape
they have.

**THIS ROW STILL HAS NO MATRIX ROW, and the implementation landing does not
create one.** The paragraph at the end of this section said that whoever picked
up #1668 would create the row with the implementation. That is declined, with a
reason: a matrix row carries a lifecycle state, and the state this work would
give it is not `DONE` — #1567, #1568, #1570, #1572 and #1884 are open under this spec
and three of them are gaps this row measured and could not close. A row created
now would enter the runnable population announcing a completion that four of its
own entries contradict. The spec is the record, `## Owed` names every owner, and
`git log --grep LTX25-PHASE-RESIDUE` is the history. Creating the row is owed to
whichever change closes the last of those five.

### How it got here

**This row's original implementation was NOT on `main`, and #1556 is closed
rather than merged.** The pull request was measured, gate-run and through three fresh
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
`scripts/now.py` and `audit-live-rows` will never surface it. That was deliberate
while the implementation was unlanded: there was nothing to give a state to, and
creating a row would have put an empty one in the runnable population. It said
here that whoever picked up #1668 would create the row with the implementation.
**#1668 lands and the row is still not created**, for the reason the top of
this section gives: the state it would carry is not `DONE`, and a row that
announces a completion five of its own `## Owed` entries contradict is worse
than no row.
