# Superseded runs may be cancelled once no gate loses coverage by being cancelled

Issues: [#822](https://github.com/mudler/vllm.cpp/issues/822),
[#863](https://github.com/mudler/vllm.cpp/issues/863)
Row: `GATE-CI-CONCURRENCY`

#822 ratified that superseded main push runs may be cancelled. #863 shows that
is unsafe against the current job layout. This row makes it safe first and then
does it.

## Why

### A diff-scoped gate is in a cancellable job

`ci.yml` states the invariant in its own header: a cancelled main run leaves its
commit range permanently unvalidated, because the next run's `before` is this
run's `sha`. `documentation-checkpoint` and `commit-protocol-tag` therefore carry
no concurrency group.

The strict trailer walk, `check-commit-trailers.py --range`, is diff-scoped and
lives in `agent-record`, which does carry one keyed on `github.ref`. For a push
that key is the constant `refs/heads/main`, so consecutive pushes cancel it.

Measured on run `31851003245`, the main push of `51e0cb5b1`:

| Job | Conclusion |
|---|---|
| `commit-protocol-tag` | success |
| `documentation-checkpoint` | success |
| `agent-record` | cancelled |

`51e0cb5b1` fails the strict walk locally. CI never said so. This is why trailer
defects have been reaching `main` and being found by hand.

`commit-protocol-tag` passed because it greps for the marker's presence. The
marker was present three times. Presence is not parseability, and only the
strict walk knows the difference.

### Cancellation and diff-scoping are incompatible as written

Workflow-level `cancel-in-progress` cancels every job in the run, including jobs
that carry no group of their own. So satisfying #822 by making the main push
lane latest-only would cancel the diff-scoped gates too, and widen #863 rather
than close it.

The two requirements are only compatible if a cancelled range is recoverable.
`github.event.before` is not recoverable: it is the previous push's `sha`
whether or not that push was gated.

## Design

**The diff-scoped base becomes the last SUCCESSFULLY gated commit.** On the push
lane the gates resolve their base by asking the forge for the most recent
successful run of this workflow on this branch and taking its head. A cancelled
or failed run therefore does not advance the base, and the next run walks the
commits the cancelled one skipped. `github.event.before` remains the fallback
when no successful run is found, which preserves today's behaviour on a fresh
branch.

This is what makes cancellation lossless. It is the enabling change, not a
side change.

A network call is acceptable here and only here. The prohibition in this
protocol is on a network call inside a *checker*, because that makes a gate fail
on connectivity. This is workflow plumbing resolving an input, it runs only on
the forge, and it falls back to the current behaviour when the query fails.

**The strict walk moves to a group-free job.** `commit-protocol-tag` already
owns the commit protocol and already carries no group, so the walk joins it
rather than gaining a new job. `agent-record` keeps only tree-scoped work, which
is safely cancellable because re-running it on a newer tree is equivalent.

**The push lane becomes latest-only.** The workflow group keys on
`github.ref` rather than `github.sha` for a push, and `cancel-in-progress`
becomes true for `push` as well as `pull_request`. `schedule` and
`workflow_dispatch` keep their own partition and stay non-cancellable, so the
baseline lane is untouched.

**A closed pull request cancels its run.** `pull_request` gains the `closed`
type, and every job's `if` excludes it, so the run enters the group, supersedes
the open run, and finishes immediately without executing gates.

**`containers.yml` gains a policy.** #822 names it as the largest source of
queued duplicates and it has none at all. Pull request and main-push runs become
latest-only. Tag publication keeps an independent non-cancellable group, because
a cancelled publish can leave a partially pushed manifest.

## Scope

In scope: `.github/workflows/ci.yml`, `.github/workflows/containers.yml`,
`tests/scripts/test_main_baseline.py`, the issue index rows.

Out of scope and stated as owed:

- `release.yml` and `gh-pages.yml`. Both already carry a workflow-level group,
  and neither is named in #822's diagnosis.
- Re-gating the four historical `main` commits that fail the strict walk. They
  predate `PR_BODY` and carry genuinely repeated blocks. History is not
  rewritten, and the diff-scoped walk never revisits them once the base has
  advanced past them.

## Gates

| Gate | Red before | Green after |
|---|---|---|
| Strict walk is in a group-free job | Assert the owning job has no `concurrency:` | Passes |
| `agent-record` keeps no diff-scoped step | Assert no `--range` under it | Passes |
| Push lane is latest-only | Assert group keys on `ref`, cancel true for push | Passes |
| Schedule stays non-cancellable | Assert cancel false for `schedule` | Passes |
| Closed PR runs no gate job | Assert every job `if` excludes `closed` | Passes |
| Containers has a policy | Assert a workflow group exists | Passes |
| Base falls back when no successful run | Simulate an empty query | Falls back to `before` |

The step bodies stay executable under `run_shimmed`, which is what caught the
`set -eu` defect in the previous row.

## Risks

**The forge query can fail or rate-limit.** It falls back to
`github.event.before`, which is exactly today's behaviour, so the worst case is
the status quo rather than a skipped gate. The fallback is asserted.

**A first push to a new branch has no successful run.** Same fallback.

**Latest-only on main hides a red that a superseded run would have shown.** That
is the point of the self-healing base: the newer run's range covers the older
one's commits, so the red still surfaces, attributed to a wider range.

## Outcome

**What the design got wrong first.** The row began by moving the strict walk to
a group-free job, on the theory that group-freeness was the invariant. It is
not. A workflow-level `cancel-in-progress` cancels every job in the run,
including jobs carrying no group, so the move protects nothing once the push
lane is latest-only. The self-healing base is the whole repair; the move is
tidiness. `test_every_diff_scoped_step_bases_on_the_last_gated_commit` asserts
the real invariant and says so in its docstring.

**What the tests found.** Writing the group-free assertion surfaced a second
diff-scoped gate nobody had noticed: `check-role-discipline.py --base/--head`
also ran inside `agent-record`. It now takes the same self-healing base.

Widening `AgentRecordDiffRangeTests` across all three diff-scoped jobs then
failed for the wrong reason, and the failure was informative: there are two
valid ways to keep a diff-scoped checker off the baseline lane. `agent-record`
runs on that lane and guards in the step; the other two opt out at the job
level. The class is scoped back to the first pattern, with the distinction
written down.

**A counting instrument lied.** The mutation sweep reported the closed-trigger
case as a mute switch. It was not: removing the trigger made the case ERROR
rather than FAIL, and the sweep grepped only `^FAIL:`. The case now asserts
cleanly so the message names the defect instead of raising.

**Four pre-existing pins changed meaning and were updated rather than deleted.**
The workflow-level cancellation pin asserted pull-request-only, which #822
ratified away. `agent-record` was pinned as unconditional, and now carries
exactly one permitted condition. Each keeps its original intent in its
docstring.

**GitHub refused the file and no local test could see it.** A blanket insert put
`LAST_GREEN` into one env block twice. `yaml.safe_load` keeps the last duplicate
and reports nothing, so all 54 cases passed while GitHub rejected `ci.yml`
outright: the run carried the workflow's PATH instead of its name, ran zero
jobs, and reported failure on a branch whose pushes the workflow does not even
subscribe to. `test_no_workflow_has_a_duplicate_mapping_key` closes the class
with a strict loader, because no `safe_load`-based assertion ever can.

## Owed

- [#869](https://github.com/mudler/vllm.cpp/issues/869). `github.event.pull_request.head.ref` is interpolated into an
  inline script (`ci.yml:584`). Pre-existing, confirmed by linting the base
  revision, and not introduced here. It is untrusted input on a fork pull request and should move
  to an environment variable.

## Now

Implemented and gated. Awaiting review on the pull request.
