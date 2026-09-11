# CPU floor counter sampling

Row: `BACKEND-GATE-CPU-LLAMACPP`
Issue: [#2448](https://github.com/mudler/vllm.cpp/issues/2448)
Base: `f98b638673b4d2edc0250eec56d229357ea38ab1`

## Scope

Repair counter sampling in `scripts/cpu-x86-llamacpp-floor.sh` and its
production-entry tests. The operator approved this bounded repair on
7 September 2026. Use one integration change with the spec committed first.
Do not change model code, contention thresholds, wait policy, historical
measurements, or the owning row's performance disposition. No GPU work.

## Source anchors and cause

The owning measurement spec is
[the x86 floor](cpu-llamacpp-floor-x86-2026-08-11.md).
`git log -S'stat_busy' -- scripts/cpu-x86-llamacpp-floor.sh` identifies
`0305b909f` as the introduction of the current sampler.
At the base, `busy_pct` and `run_leg` each obtain busy and total counters
from separate reads. Their intervals can differ. This permits percentages
above 100 even when each individual counter advances correctly.
The total includes guest fields twice. Arithmetic through awk also depends
on its numeric precision and output formatting.

[Linux proc documentation](https://docs.kernel.org/filesystems/proc.html)
defines the first eight fields and warns that iowait can decrease.
[Linux v6.12 account_guest_time](https://github.com/torvalds/linux/blob/v6.12/kernel/sched/cputime.c#L132)
adds guest time to user or nice as well as the guest counter. Therefore the
denominator must exclude guest and guest_nice.

## Design

Read one aggregate CPU line per endpoint. Derive busy and total deltas from
the same two lines at both production call sites. This removes mismatched
reads. It does not claim that the kernel freezes all CPUs during a read.
Busy includes user, nice, system, irq, softirq, and steal. Total adds idle
and iowait. Guest fields are already represented in user and nice.

Parse decimal integers without awk floating-point conversion. Accept at most
16 decimal digits per relevant counter. Eight such counters, including a
percentage multiplication by 100, fit signed 64-bit Bash arithmetic.
Validate before arithmetic and interpret leading zeros as decimal.
Reject malformed, missing, oversized, or decreasing relevant counters.
Reject a nonpositive total delta. Report an invalid-sample diagnostic and
the conservative sentinel 100 with failure status, not a measured utilization
of 100 percent. Both gates must reject that status even at a ceiling of 100.
Do not clamp a broken ratio into a valid measurement.

Preserve subtraction of the leg's own CPU time and the existing nonnegative
foreign-time floor. Preserve compiler detection, own-process exclusions,
quiet and foreign ceilings, and timeout behavior. A genuine busy sample
must still refuse a quiet window or discard a contended leg.

## Tests and gates

Run the full production shell with stub engines and deterministic CPU lines
in a scratch copy. Replace only the `/proc/stat` read boundary, not the
sampler or either call site. Independently test mismatched-read regression,
large exact counter differences, guest denominator, idle and iowait semantics,
steal, invalid samples, own-time subtraction, and foreign builder refusal.
Keep a real `/proc/stat` smoke case without retries until green.
Two concurrent foreign-process fixtures reproduced an additional collision:
both saw `builders=2` because their process names were identical. Give each
fixture a distinct process name within Linux's 15-character comm limit.
Test two live fixture processes together and retain the real builder filter.
This isolates synthetic test neighbors. It does not exclude production
neighbors or relax a contention assertion.

Capture red before implementation. Run
`python3 tests/scripts/test_cpu_x86_llamacpp_floor.py` for focused green.
Mutate each new guarantee in scratch and require test failure. Verify
byte-identical restoration. Run `scripts/agent-preflight.sh` once on the
final immutable head and report failures and skips individually.
Fresh review and an operator rerun are required before landing.

## Risks and stop conditions

The bounded integer domain refuses implausibly large counters rather than
overflowing. Counter resets, hotplug inconsistencies, and decreasing iowait
remain conservative samples. No synthetic fixture establishes actual host
utilization or a performance result. Missing authority, a needed threshold
change, or an edit outside this scope requires operator direction.

## Evidence

Initial preflight: `/tmp/cpu2448-initial-preflight.log`.
Focused red, green, mutation, and final gate evidence are recorded with the
implementation result. Historical benchmark numbers remain unchanged.

## Outcome

The sampler uses one CPU line per endpoint and integer deltas. Invalid
samples carry failure status through both gates, including at a ceiling of
100. Test fixtures use distinct process names and synthetic counter inputs.
A separate real-counter smoke accepts only bounded valid results or an
explicit invalid-sample refusal. It never retries until success.

The focused gate passed 18 tests. The mutation checks detected 25 sampler
mutations and the fixture-name collision. The sampler scratch was restored
byte-for-byte after each mutation. Red evidence is in
`/tmp/cpu2448-red.log`, `/tmp/cpu2448-red-leg.log`, and
`/tmp/cpu2448-red-probes.log`. Green evidence is in
`/tmp/cpu2448-green.log`. Mutation evidence is in
`/tmp/cpu2448-mutations.log` and `/tmp/cpu2448-probe-mutation.log`.
The immutable-head preflight result is reported in the implementation
handoff at `/tmp/cpu2448-final-preflight.log`.

No contention default changed. The quiet and foreign ceilings remain 10
percent because this repair changes sampling, not the accepted contention.
The five-second sample window and one-hour wait remain unchanged. Skipping
the gate and clamping broken ratios were rejected because neither repairs
the sampled counters. No CPU speed measurement or historical number changed.

## Review repair: observable fixtures

The fresh review of `d9de7fec8` found two test gaps for #2448. Removing the
own-tree builder exclusion did not fail its test because Bash replaced the
copied-name ancestor with the final command. Removing the aggregate CPU label
guard also passed because invalid fixtures lacked valid numeric `cpu0` rows.

Change only the production-entry tests. Record real process ancestry at each
synthetic CPU-read boundary. First assert that the named ancestor remains
present, then prevent the fixture's tail-exec optimization. Retain the child
status and the real builder filter. Verify own and foreign fixtures together.
Add before- and after-endpoint cases with valid counters and nonaggregate
labels. Require invalid-sample diagnostics and refusal at the entry point.

Capture the missing-ancestor assertion before the fixture repair. For label
cases, capture failure against a scratch copy without the production label
guard. Neither change modifies the production shell. Mutations must detect
removed own-tree exclusion, restored tail-exec, and removed label validation.
Restore scratch bytes after every mutation. Run the focused suite and one
full preflight on the final immutable head, then obtain fresh scoped review.
