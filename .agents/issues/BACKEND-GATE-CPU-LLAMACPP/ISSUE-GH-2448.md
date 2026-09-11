ID: ISSUE-GH-2448
Title: test_cpu_x86_llamacpp_floor cannot obtain a quiet window on a shared box, so preflight returns 1 on correct trees
Row: BACKEND-GATE-CPU-LLAMACPP
State: OPEN
Kind: UNKNOWN
GitHub: 2448
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `-`
>
> `test_cpu_x86_llamacpp_floor` cannot complete on a shared machine. It has failed on at
> least four separate preflight runs on the development box, across three unrelated
> branches and on untouched `main`, always naming its own cause:
>
> ```
> NO_QUIET_WINDOW ... busy=109% load=96.87
> NO_QUIET_WINDOW ... busy=120% load=30.64
> NO_QUIET_WINDOW ... busy=108% load=100.05
> ```
>
> It is the harness, not a tree defect. Three pieces of evidence, strongest first:
>
> 1. **Different runs fail different subtests.** One run failed
>    `test_a_contended_leg_is_discarded_and_never_summarised`, another
>    `test_the_published_figures_are_computed_not_transcribed`. A deterministic defect fails
>    the same subtest every time; a harness that cannot find a quiet window fails whichever
>    subtest happens to need one.
> 2. **It passes when the box is quiet.** Measured green at load 4.5 with 49 GB free on a
>    merge commit that failed the same gate at load 43.
> 3. It has failed on branches whose diff contains **no code at all** (one `.md` file), with
>    the harness byte-identical to `main`.
>
> Why this matters beyond the annoyance: `scripts/agent-preflight.sh --fail-on-skip` is the
> recorded pre-push gate, so this failure makes preflight return 1 on correct trees. Every
> operator then has to attribute it by hand before merging, and the standing risk is that a
> real failure gets waved through as "the known flake" - which is exactly the reasoning a
> mute switch relies on.
>
> This box routinely runs ten or more concurrent sessions; the quiet window the harness
> waits for does not occur. Options, in the order I would consider them: have the harness
> exit 77 (Skipped) rather than fail when it cannot obtain a quiet window, so it reports
> "not measured" instead of "failed"; or gate it behind an explicit opt-in for dedicated
> hosts.
>
> Owed: a decision on which of those, and the change.

## Resolution

-
