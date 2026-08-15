# BENCH-ASSERT-CLOCK-STATE — a ratio may not be quoted without the clock it was measured at

Issue: [#543](https://github.com/mudler/vllm.cpp/issues/543) (the defect —
per-call attribution is not reproducible across box states),
[#545](https://github.com/mudler/vllm.cpp/issues/545) (the reboot blocker that
makes cross-boot comparison the *normal* case rather than the exception)
Row: `BENCH-ASSERT-CLOCK-STATE`
Prior art: [#375](https://github.com/mudler/vllm.cpp/issues/375) and
[#520](https://github.com/mudler/vllm.cpp/issues/520) — the same class. An
environment variable nobody recorded silently repriced every number, and the
harness could not tell because it never wrote the variable down.

## The defect

On `dgx.casa` (GB10, driver `580.159.03`) the SM clock differs **between boots**
and is not throttling — `clocks_throttle_reasons.active = 0x0`, persistence
`Enabled`:

| boot | SM clock over the captured window | our ms/step |
|---|---|---|
| `f6bbbfc6` | n=61, min 2398 / **med 2470** / max 2489 | **82.1664** |
| `2fca2b02` | n=50, **flat 2190** (`clocks.max.sm` 3003, applications 2418) | **88.1000** |

A **12.79%** median-clock delta produced **+7.22%** step time. The control is
what settles it: `marlin::Marlin`, 129 calls/step, byte-identical invocation,
**no source change** between `a170c81c` and `4064558d0`, moved
**45.2845 → 49.6544 ms/step = +9.65%**.

That control drift is **larger than either deficit it was used to rank** —
`in_proj` +2.97%, `out_proj`/`o_proj` +6.28%. Those two are therefore **NOT
ESTABLISHED**: they were never taken against a clock control. The same effect
explains a same-binary same-arm swing of 382.60 → 357.59 us/call (−6.5%) across
a reboot, and two probes disagreeing ~6% uniformly eight minutes apart *within
one boot* (a 2398 MHz entry against a 1781 MHz one).

Nothing in the tree records any of this. `grep -rn 'clocks\.' tools/ scripts/`
returns one hit, in prose, inside `.agents/benchmark-record.md`. The three
harness scripts that touch `nvidia-smi` capture `-q -d
PERFORMANCE,TEMPERATURE,POWER` into an **unparsed text blob** that no summary
reads, or query `--query-compute-apps` for idleness. No manifest carries a clock
value, a boot id, or a throttle state, so no ratio in this repository can be
attributed to the clock it was measured at.

## Scope

**In scope.**

1. **One helper**, `tools/bench/gpu_clock_state.py`: sample `nvidia-smi`, build
   the per-leg record, validate it fail-closed, and compare two arms. Other
   harnesses import it; it duplicates nothing and defines no framework. It is
   standard-library-only, like `serve_low_common.py`, so its logic runs in CPU
   CI with no GPU and no `nvidia-smi`.
2. **Recording** in the leg-producing harness, `scripts/dgx-online-serving.sh`,
   as a background sampler across the timed bench loop, written to
   `clocks/<model>/<engine>/r<N>.{samples.jsonl,summary.json}` beside
   `memory/…`.

   It carries **both** of the memory sampler's lifetime properties, which is
   what "the same shape" has to mean. `mpid` is a **global** the EXIT trap waits
   on, and `sample_process_memory.py` is **bounded** by `--pid "${spid}"`. The
   clock sampler's `clock_pid` was `local` to `run_leg` and therefore invisible
   to the trap, and it was launched with neither `--pid` nor `--max-duration`,
   so its only stop condition was a `SIGTERM` that `set -euo pipefail` skips
   whenever `online_gate.py bench` fails inside the loop: the sampler was
   orphaned on **every** abort path and went on polling `nvidia-smi` once a
   second forever, on a shared box another session may hold `$HOME/gpu.lock`
   on. `clock_pid` is now a global reaped by `cleanup_server`, and the sampler
   is bounded by `--max-duration ${clock_sampler_max_seconds}` (7200 s, 4× the
   driver's own `ready_timeout_seconds`, so it is a safety net that cannot
   truncate a leg the rest of the driver still considers live).
3. **Asserting** in the ratio-producing surface, `tools/bench/
   online_gate_summary.py`, through the existing `reasons` seam: a violated
   clock contract makes a leg *not binding-eligible*, which is this harness's
   spelling of NOT ESTABLISHED. Every ratio additionally carries a `clock`
   block naming both arms' medians, their offset, and the estimated timing
   effect, so a reader can size the clock against the effect without leaving
   the row.
4. **The operational fix** in `.agents/benchmarking.md`, including the
   shared-host hazard.
5. **A note where existing numbers are cited** that they predate clock
   assertion.

**Out of scope, deliberately.**

- **Editing any recorded number.** AGENTS.md: never delete evidence. The past
  figures stay exactly as they are and gain a note; nothing is restated.
- **Re-measuring.** The GPU is held by another session, `$HOME/gpu.lock` is
  taken, and #545 means the box does not survive a four-leg chain. This row
  makes the *next* measurement attributable; it takes none.
- **Changing the clock.** `nvidia-smi -lgc` is documented here and executed by
  nobody in this row — it is a shared-host mutation (§Operational fix).
- **The trace/per-kernel harnesses** (`finalize_*_trace.py`,
  `summarize_torch_kernels.py`, `gdn_packed_component.py`). They import the same
  helper and the same sampler CLI is what they would call, but wiring each one
  is a separate change with its own fixtures. Recorded as **owed** below rather
  than quietly skipped — and it is the trace path, not the online gate, that
  produced the two retracted findings.

## Design

### What is recorded

Per leg, `clocks/<model>/<engine>/r<N>.summary.json`:

| field | source |
|---|---|
| `boot_id` | `/proc/sys/kernel/random/boot_id` |
| `sm_clock_mhz` | `{n, min, median, max, spread_pct}` over the window |
| `clocks_max_sm_mhz` | `clocks.max.sm` |
| `clocks_applications_graphics_mhz` | `clocks.applications.graphics` |
| `throttle_reasons_active` | sorted union of `clocks_throttle_reasons.active` |
| `persistence_mode` | `persistence_mode` |
| `driver_version`, `gpu_name` | `driver_version`, `name` |
| `idle_samples_excluded` | count of samples with `utilization.gpu == 0` |

`spread_pct` is `(max − min) / median × 100` over the retained samples. The raw
per-sample rows stay in `r<N>.samples.jsonl`, because a summary that cannot be
recomputed from its own evidence is a claim, not a record.

**Idle samples are excluded from the statistics and counted, not dropped.** A
clock read while the GPU is doing nothing did not price any work, and the timed
window necessarily contains the harness's own gaps between concurrency points.
Excluding them silently would be a lie; the count is in the record, and a leg
that is *entirely* idle has `n == 0` and fails validation.

### What is asserted

| assertion | value | how justified |
|---|---|---|
| both arms of a ratio share `boot_id` | exact | cross-boot comparison is what produced the retracted findings; there is no threshold that makes it safe |
| both arms share every `STATIC_FIELDS` value | exact | see **the override**, below |
| within-run spread | `≤ 5.0%` | see below |
| cross-arm median offset | `≤ 1.0%` | see below |
| retained busy samples | `≥ 30` | see below |
| busy fraction of the window | `≥ 50%` | see below |
| `throttle_reasons_active` carries no non-benign bit | mask | a throttled window is not the window the number claims |
| `persistence_mode == Enabled` | exact | already true on the box; its absence changes idle clock behavior |

**Within-run spread, 5.0%.** The admissible band is bounded on both sides by the
data above. It must *accept* the only clean window we have —
`(2489 − 2398) / 2470 = 3.68%` — because a threshold that voids our one good
measurement is useless. It must *reject* the within-boot disagreement that a
2398 MHz entry and a 1781 MHz one represent, which is ~26% however it is
normalized. 5.0 sits just above the clean observation, with ~1.3 points of
headroom so a marginally noisier but still healthy window is not spuriously
voided, and roughly five times below the failure it exists to catch.

It is deliberately **not** held to the forward criterion the offset is held to,
and the reason is recorded rather than left to be rediscovered.
`5.0 × 0.7548 = 3.77%` exceeds the `2.97%` smallest deficit ranked, so a leg
sitting *at* the spread ceiling can carry an artifact larger than that deficit.
The two rules defend different things. The **offset** bounds a *systematic*
difference between the arms — one arm ran at one clock and the other at another,
and the whole of it transfers into the ratio — so it must sit under the smallest
effect anyone ranks. **Spread** bounds *dispersion inside one arm's window*,
which does not transfer that way: both arms sweep the same six concurrency
points on the same box, so most of the dispersion is common, and what survives
into the ratio is the difference of two medians, which the offset rule already
bounds at 1.0%. The spread rule exists to detect that a window was not one state
*at all* — the ~26% within-boot disagreement — not to bound a transferable bias.
Tightening it to satisfy the forward criterion would mean `≤ 3.93%`, which sits
0.25 points above our only clean capture and would void it on any
noisier-but-healthy day, the exact failure the both-sides bound was chosen to
avoid. The residual is therefore stated: **passing spread establishes that each
arm was one state; it does not on its own establish a sub-4% deficit.** The
offset rule is what qualifies the ratio.

**Cross-arm median offset, 1.0%.** For any kernel whose time scales with clock
the transfer is **bounded above by 1.0** — a 12.79% clock deficit can cost at
most 12.79% of time — so a 1.0% offset implies **at most a 1.0% effect on
physics**, with no appeal to any measurement. That is already under the ~2.97%
smallest deficit this harness has been used to rank, so a pair inside the
threshold cannot have had its ranking inverted by clocks.

The one cross-boot event we have repriced a byte-identical kernel by **+9.65%**
for a **12.79%** median-clock offset: a transfer of **0.7548** percentage points
of kernel time per point of clock (step-level transfer was lower, 0.565; the
larger is used). That coefficient is `n = 1`, and it sits *below* the 1.0
ceiling exactly as a partly memory-bound kernel should — which is corroboration,
not the argument. The distinction matters and is recorded: the coefficient is
**not a gate term** (no gate expression evaluates it, which the mutation set
proves) but it *was* a gate **premise**, because the threshold was originally
chosen by multiplying through it. The physics bound retires it from that role —
the threshold holds at the transfer's theoretical ceiling — leaving the
coefficient to *report* an estimated effect and nothing else.

**Retained busy samples, 30, and busy fraction, 50%.** Without a floor the
incentive is **inverted**: `spread_pct` over `n == 1` is definitionally
**0.00%**, the best score the gate can award, so a window the sampler barely
observed outscores one it actually watched — and `idle_samples_excluded`, the
field that would betray it, was written to `r<N>.summary.json` and then never
asserted, never folded into the ratio's `clock` block, and never printed. Six
legs each holding one busy sample and 300 idle scored `gate_pass: true`, `clock
reasons: []`, and a report line reading `+0.00%`.

The count floor is bounded on both sides like the spread rule. It must *accept*
the real windows: #543's two captures are `n = 61` and `n = 50`, and 30 sits 40%
below the smaller. It must *reject* the degenerate window, and 30 is 30× above
`n == 1`. And it reads off the sampler and the grid rather than off what happens
to pass: at the driver's `--interval 1` a busy sample is **a second of observed
busy GPU**, while the smallest configured leg (`POINTS_BY_MODEL`'s four-point
set: 6+6+12+24 prompts at 128 output tokens = 1920 sequential decode steps)
would have to average under ~16 ms/step *including* its 48 prefills of 1024
tokens to finish inside 30 busy seconds. This box measures 82–88 ms/step. A leg
under the floor is not a fast leg; it is an unobserved one.

The count alone does not catch dilution — 30 busy among 3000 idle clears it — so
the busy **fraction** is floored too. The sampler covers the bench loop only: it
starts after the preflight stream and stops before the after-thermal snapshot,
so the non-busy time inside the window is client startup between six
`online_gate.py bench` invocations, not model load, not cache drops, not server
start, while the GPU serves 336 requests of 1024-in/128-out across the span.
Requiring the *majority* of the window to be busy is the weakest form of the
claim the record makes. It is also the field that betrays a sampler which
outlived its leg: an orphan accrues idle samples without bound and nothing else
in the record notices.

Both counts are now surfaced in the ratio's `clock` block
(`<arm>_busy_samples`, `<arm>_idle_samples_excluded`) and printed on the
report's `observed:` line beside the offset, because a number nobody prints is a
number nobody checks.

**The override.** `--allow-cross-boot` exists because #545 makes same-boot
capture of a four-leg chain unreliable, and a gate nobody can satisfy is a gate
everybody routes around. It does not make the comparison clean: it converts the
refusal into a recorded caveat (`cross_boot_override: true` plus the boot ids
and the offset in the ratio's `clock` block and in the report). Every other
assertion still applies — **the override waives `boot_id` and nothing else.**

That has to be enforced explicitly, not implied. `STATIC_FIELDS` was checked
*within* a window (`build_clock_record`) and *between one arm's legs*
(`merge_clock_records`), and same-boot equality was doing duty as the implicit
proxy for "same machine" on the third edge. Removing the boot check removed the
proxy with it: an **NVIDIA GB10 / driver 580.159.03 / max 3003** arm compared
**clean** against an **NVIDIA H100 80GB HBM3 / driver 550.54.15 / max 1980**
one, with a caveat that said only "different boots". `compare_clock_records` now
asserts every `STATIC_FIELDS` value across the arms **unconditionally**. #545
makes the override the normal path, so this was reachable, not theoretical.

### Why this seam

`online_gate_summary._memory_for_leg` already reads an optional per-leg evidence
artifact and turns every defect into a `reason` that clears
`binding_eligible`. `_clock_for_leg` is the same function shape against the same
seam. Nothing new is invented, no gate is weakened, and a missing clock record
voids a leg exactly the way a missing memory summary already does.

That has a consequence worth stating rather than discovering: **re-summarizing
an existing evidence tree now yields NOT ESTABLISHED**, because no tree on disk
carries a clock record. That is the correct answer — those trees genuinely
cannot be attributed to a clock — and it is why the note in §Records exists
instead of a retroactive edit.

## Risks

- **The gate is unpassable if the spread rule is too tight.** Mitigated by
  excluding idle samples and by choosing the threshold from the clean window
  rather than from theory. If a real leg still exceeds 5%, that is a
  *measurement finding to record*, not a threshold to widen quietly.
- **`nvidia-smi` field names drift.** Driver 580 accepts
  `clocks_throttle_reasons.active`; newer drivers prefer
  `clocks_event_reasons.active`. The sampler queries the throttle spelling, and
  a query failure is a refusal to sample, never a default.
- **The sampler perturbs the measurement.** One `nvidia-smi` per second against
  a 128-token × six-concurrency leg, launched identically on both arms, so it
  cancels in the ratio — the same argument `start_server` already makes for the
  memory sampler, and it is recorded in the same place.
- **The coverage floors are unpassable if a real leg falls under them.** Same
  posture as the spread rule: a real leg voided by the 30-sample or 50%-busy
  floor is a *measurement finding to record*, not a threshold to widen quietly.
  The floors were chosen so the failure direction is loud — a voided leg — never
  a silent pass, which is what the unfloored version did.
- **`--max-duration` truncates a genuinely long leg.** At 7200 s the sampler
  would stop while the leg continued and the record would under-cover the
  window. That is 4× the driver's readiness budget and far above any leg the
  grid can produce; it is a bound on an orphan, not a budget on a leg. If a leg
  ever reaches it, the record is partial and the leg is not established.

## Tests

`tests/tools/test_gpu_clock_state.py`, synthetic manifests only, no GPU:

- parsing a real `--query-gpu` CSV line, including the `nounits` form
- summary statistics, idle exclusion, and the `n == 0` refusal
- fail-closed validation: missing field, non-finite value, wrong type,
  negative clock, empty throttle list
- **the cross-boot refusal**, and that `--allow-cross-boot` records a caveat
  rather than silently passing
- **the over-spread refusal**, at the threshold, either side of it
- the cross-arm offset refusal and the reported estimated effect
- the throttle mask: benign bits accepted, each non-benign bit refused
- **the coverage floors**: the one-busy-sample-of-301 record refused on the
  COUNT rather than on the spread it has no right to report; a diluted window
  that clears the count and fails the fraction; both floors at and either side
  of the threshold; a non-integer idle count refused rather than crashing the
  division
- **the hardware identity across arms**: each `STATIC_FIELDS` value, under the
  override *and* without it, plus the full GB10-against-H100 shape
- **the staircase arm**: three individually-flat legs at 2470 / 2300 / 2190 on
  ONE boot, with the vLLM arm pinned at the merged median so the cross-arm
  offset stays 0.00% and cannot mask the removal
- wiring: `online_gate_summary` voids a leg with a missing, cross-boot, or
  over-spread clock record, matched against **that call site's own** message —
  #520 established that an unanchored `assertRaises` stays green on a gutted
  check

Fixture legs now carry a realistic window. A three-sample leg was never a leg
anyone could have captured — the driver samples at 1 Hz across a bench loop of
minutes — so `_window`/`_clock_window` repeat each pattern to clear the floor.
Whole-list repetition preserves min, median, max and therefore `spread_pct`
*exactly*, so every threshold case still asserts what it asserted; only the
count changes, which is the floor's subject.

Regression surface: `tests/tools` in full — **233** tests on the base SHA
`8b00f79f2`, **295** at `3e0f2b2d3`, **310** here; 0 removed at either step,
proven by a sorted test-name diff. A changed count is RED even when it prints
`OK`.

**Twenty-four mutations**, applied one at a time with `count == 1` anchors and
restored byte-for-byte by sha256 — the original twelve (`M1`–`M12`) plus twelve
against this round's surfaces (`R13`–`R24`). **24 / 24 RED.**

| mutation | what it breaks |
|---|---|
| `M1` | the cross-boot refusal |
| `M2` | within-run spread widened to 1000% |
| `M3` | cross-arm offset widened to 1000% |
| `M4` | throttle mask opened to every bit |
| `M5` | idle samples counted as busy |
| `M6` | the straddled-boot fold |
| `M7` | leg clock reasons dropped from the aggregate |
| `M8` | arm clock reasons dropped from the aggregate |
| `M9` | the THROUGHPUT ratio's `clock_established` term |
| `M10` | the MEMORY ratio's `clock_established` term |
| `M11` | the missing-record reason |
| `M12` | the stream/summary reconciliation |
| `R13` | the retained-busy-sample floor |
| `R14` | the busy-fraction floor |
| `R15` | **the override waives spread, throttle AND persistence** |
| `R16` | the cross-arm `STATIC_FIELDS` assertion |
| `R17` | `idle_samples_excluded` type validation |
| `R18` | the busy-sample floor lowered to 1 |
| `R19` | the busy-fraction floor lowered to 0 |
| `R20` | the override waives hardware identity after all |
| `R21` | the merged spread spans only the first leg |
| `R22` | busy-sample surfacing |
| `R23` | idle-sample surfacing |
| `R24` | the report's `observed:` line |

**Three of `M1`–`M12` survived the author's first round** — both ratio-level
terms and the missing-record reason. That is #520's lesson repeating: the
leg-level reason already voided every gate assertion, so each ratio-level site
was dominated and a `gate_pass` assertion let the two ratio families mask each
other's removal. Two cases assert the two families' `binding_eligible`
SEPARATELY on a pair whose arms are individually clean, and a third pins the
reason text that names the offending arm.

**`R15` survived review** and is the reason the staircase case exists. Both
tests that named "the override does not waive state" missed it: one asserts an
*offset* reason, appended OUTSIDE the mutated expression, and the other writes
an over-spread LEG that `_clock_for_leg` has already voided at leg level.
Exactly one check lives only at the compare site — the **merged-arm spread**,
an arm whose three legs each sat flat at a different clock —
because `merge_clock_records` raises only on a straddled boot and on a static
field, and `_clock_for_arm` surfaces only what it raises.

**`R17` survived this round's first pass** and was a genuine gap in the new
code, not a dominated site: nothing constructed a record with a non-integer
`idle_samples_excluded`, so the guard that keeps `busy + idle` out of the
arithmetic was unasserted. Two cases now pin it — the refusal and the
"reason, not a crash" path.

One methodological note worth keeping: `M11` first read SURVIVED because the
mutation itself was inert — `"reasons": [] or [...]` evaluates to the non-empty
list. A byte-level sha256 check proves the file changed, not that the *behavior*
did. `[] and [...]` is the encoding that actually removes the reason, and under
it `M11` is RED.

## Gates

- `python3 -m unittest discover -s tests/tools -t .` — full, serial. `pytest`
  mis-collects this tree (16 false failures on clean `main`); do not use it.
- `scripts/agent-preflight.sh --staged`, then again on committed HEAD.
- `bash -n scripts/dgx-online-serving.sh`.
- **No benchmark gate.** The GPU is held, the box is at 99% disk, and #545 is
  open. Recorded `PENDING` with the exact handoff rather than waived.

## Stop conditions

- Stop before running any GPU work or changing any clock: `$HOME/gpu.lock` is
  held by another session and `-lgc` would corrupt their in-flight measurement.
- Stop before editing a recorded number to agree with the new contract. A past
  figure that cannot be attributed gains a note, never an edit.
- Stop if satisfying the contract requires weakening an existing eligibility
  reason. The clock rule is additive or it is wrong.
- Stop if the helper starts to need a plugin surface, a registry, or a second
  file. One module, imported.

## Evidence

- Live read-only `ssh dgx.casa` sample, 2026-08-12, used to fix the fixture
  format verbatim: `0, NVIDIA GB10, 580.159.03, 2190 MHz, 3003 MHz, 2418 MHz,
  0x0000000000000000, Enabled, P0`, boot id
  `13dc5579-455c-45c8-8e4d-d09c457fa826`. That is a **third** boot id, and the
  clock is again the degraded 2190 — the defect is live, not historical.
- The two-boot table and the `marlin::Marlin` control are #543's, already
  commented there.

## Owed

- The trace/per-kernel harnesses do not yet call the helper (§Scope). Until they
  do, a per-call `us/call` figure carries no clock attribution — which is
  precisely where #543's retracted findings came from.
  [`.agents/benchmarking.md`](../benchmarking.md) now says so in the same words
  rather than describing the wiring in the present tense.
- The `0.7548` transfer coefficient is `n = 1`. A second cross-boot pair, once
  #545 allows one, either confirms it or replaces it. It is no longer load
  bearing for the offset threshold, which holds at the transfer's physical
  ceiling of 1.0.
- **`all-runs.json` carries no clock block and no `allow_cross_boot` flag.**
  `report.md` prints `(OVERRIDDEN)` and `CAVEAT:`, so it is not buried for a
  human reader, but the JSON a consumer parses is silent about both. Recorded
  rather than fixed: the ratio document is the surface that qualifies a ratio,
  and widening the runs document is a change with its own fixtures.
- **`compare_clock_records` builds its keys as `f"{ours_label}_boot_id"`.** Two
  identical labels would collapse one arm's block into the other's. Unreachable
  from any current call site — `online_gate_summary` passes the defaults — and
  recorded so a future caller that takes labels from data knows the shape.
- **The busy floors are asserted, but never yet measured against a real leg.**
  They are derived from the sampling rate and the grid definition (§What is
  asserted); the first attributable grid is what confirms the derivation. If a
  real leg lands under either floor, that is a finding to record, not a
  threshold to widen.

## Now

`ACTIVE` — recorded, asserted, and documented; no measurement is taken and none
is restated. The first attributable grid is `PENDING` on `$HOME/gpu.lock` and
on #545.
