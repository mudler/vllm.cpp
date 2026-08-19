# The baseline lane discards a queued run, so `main` loses the verdict

Issue: [#274](https://github.com/mudler/vllm.cpp/issues/274)
Row: `—` (no matrix row owns CI infrastructure; see `main-verifiability.md`)
Branch: `fix/baseline-lane-eviction`
Base SHA: `250db75a2b4be4d3be89636af0f11248c6385a1f`

Follow-on to [`main-verifiability.md`](main-verifiability.md). That spec built
the baseline lane and named this exact failure as risk 2, "the baseline lane
itself gets cancelled -- the failure that would silently void the whole row". It
recorded the handling as "own group (event discriminator) plus
`cancel-in-progress: false`" and marked the claim `PENDING`, because only a
post-merge observation could settle it. The observation is now made, and the
handling is incomplete.

## Scope

**In.** Make a scheduled baseline run reach a completed verdict, whatever the
suite's wall-clock time. One property, stated as GitHub resolves it: a run or a
job that waits in a concurrency group must never be discarded before it starts.

**Out.** The reds the lane reports. `windows-msvc-cpu` and
`windows-msvc-vulkan` are red on
[#584](https://github.com/mudler/vllm.cpp/issues/584), and that is the lane
working. No product code, no test code outside the workflow suite, and no change
to what any gate asserts.

## Our baseline

Six facts, each measured against `origin/main` on 2026-08-19, not inferred.

### 1. `main` does get verdicts, and the 200-run window hides them

`gh run list --workflow ci.yml --limit 200` reports 175 `cancelled`, 17
`failure` and 0 `success`. That window spans `2026-08-18T12:38:27Z` to
`2026-08-19T06:52:18Z`, which is **18 hours 14 minutes**, because the push and
pull request lanes fill it. By event it is 135 cancelled and 15 failed pull
request runs, 38 cancelled push runs, and **5 scheduled runs**.

Read the lane instead of the window. Over the last 39 completed scheduled runs,
`2026-08-12T17:10Z` to `2026-08-19T04:49Z`:

| Conclusion | Count | Median wall | Max wall |
|---|---|---|---|
| `success` | 20 | 118 min | 397 min |
| `failure` | 17 | 156 min | 568 min |
| `cancelled` | 2 | 200 min | 221 min |

So the lane published a verdict for 37 of 39 triggers. The last fully green
baseline is run `32067210005` at `76f2a6d84e41`, `2026-08-17T20:42:13Z`.

### 2. Both cancelled scheduled runs executed ZERO jobs

This is the defect, and it is not `cancel-in-progress`.

| Run | Created | Jobs started | Cancelled at | Successor created |
|---|---|---|---|---|
| `32140419182` | `13:05:56` | **none** | `16:46:51` | `32162114781` at `16:46:50` |
| `32206456661` | `01:51:48` | **none** | `04:49:54` | `32217173498` at `04:49:53` |

`gh run view <id> --json jobs` returns `startedAt: null` for every job of both
runs. The cancel instant equals the successor's creation instant, 2 out of 2.

`cancel-in-progress` resolves to `false` for `schedule`, so it cancelled
nothing. The mechanism is the other half of GitHub's concurrency contract: a
group holds **one** pending run. When a third run joins a group that already has
one run in progress and one pending, GitHub cancels the pending one. The
baseline lane keys every scheduled run into the single group
`ci-schedule-refs/heads/main-mudler/vllm.cpp`, so a queue depth of two is
reachable and the middle run is discarded before it starts.

### 3. The queue is real, and it is head-to-tail

| Run | Created | First job started | Predecessor completed |
|---|---|---|---|
| `32162114781` | `16:46:50` | `18:20:21` | `32118587477` at `18:20:20` |
| `32183400023` | `20:38:43` | `23:11:38` | `32162114781` at `23:11:37` |

2 out of 2, a successor starts the second its predecessor ends. Waits of 1 h 34
min and 2 h 33 min.

### 4. The overrun is runner contention, not job cost

Run `32118587477` had no predecessor to wait for and still took **9 h 28 min**
(`08:52:09` to `18:20:21`). Its own jobs sum to **345 job-minutes**, of which
`cuda-fat-build` is 98 and the two `windows-2022` proofs are 52. The rest is
queueing for a runner:

* `agent-record` is a 3.8-minute `ubuntu-latest` job. Its only dependency
  finished at `11:26:07`. It started at `15:24:59`, **3 h 59 min** later.
* `baseline-summary` is a 0.2-minute job. Its last dependency finished at
  `13:47:22`. It started at `18:20:10`, **4 h 33 min** later.
* The run itself waited from `08:52:09` to `11:12:56` for its first real job.

`main-verifiability.md` chose the four-hour cron on the premise that "successive
runs still cannot overlap a 99-minute suite". A 99-minute suite is no longer
what runs. The premise is falsified, and a four-hour period against a 9 h 28 min
wall reaches queue depth two by arithmetic.

### 5. The 17 completed failures are not one systemic red

`windows-msvc-cpu` and `windows-msvc-vulkan` fail in 5 of the 5 scheduled runs
since [#503](https://github.com/mudler/vllm.cpp/issues/503) put them on the
lane on 2026-08-17. Both die the same way:
`test_openai_api_server.exe exited with status -1073740791`, which is
`0xC0000409`, `STATUS_STACK_BUFFER_OVERRUN`. That is
[#584](https://github.com/mudler/vllm.cpp/issues/584), open, pre-existing, and
named in `ci.yml`'s own comment before the lane could see it.

The others are separate and shorter-lived: `build-newest-gcc` failed 2 of 5 on
`'::getpid' has not been declared` under gcc 16, which `eb770f595` fixed
([#1296](https://github.com/mudler/vllm.cpp/issues/1296)) and current `main`
does not carry; `sanitize-cpu` both lanes and `build-test-cpu` failed 1 of 5;
`agent-record` failed 2 of the 39. The pull request lane's
`documentation-checkpoint` and `commit-protocol-tag` failures are per-commit
gates that the baseline lane does not run at all.

**So question (b) has no single answer, and that matters for question (a).** One
pre-existing red holds the lane at RED while every other job is green. Fixing
the lane does not turn it green, and it must not.

### 6. `cancel-in-progress` cannot be flipped for `push`

Not attempted, and the arithmetic is unchanged from `main-verifiability.md`. At
55 pushes per day and 345 job-minutes per suite, a non-cancellable push lane is
about **316 hours of job time per day** on a pool that already makes a
3.8-minute job wait 3 h 59 min. It also reverses #822 with no new evidence.

## Upstream chain

**None.** This is GitHub Actions configuration for this repository. vLLM is the
reference for engine behavior and has nothing to mirror here. Nothing is ported.

## Design

### Chosen: every non-cancellable concurrency group varies per run

One rule, applied to all 11 concurrency blocks in `ci.yml`:

> A concurrency group whose `cancel-in-progress` resolves to `false` for an
> event must resolve to a key unique to the run for that event.

A group that never cancels has one remaining behavior, which is to queue. A
queue of depth one that discards its own contents is worse than no queue,
because the discard costs a verdict and saves nothing: neither cancelled run
executed a job, so neither had anything to reclaim.

The key becomes, at the workflow level:

```yaml
group: ci-${{ github.event_name }}-${{ (github.event_name == 'schedule' || github.event_name == 'workflow_dispatch') && github.run_id || github.event.pull_request.number || github.ref }}-${{ github.repository }}
```

and at each job level, with `github.ref` in place of the pull request number:

```yaml
group: ci-<job>-${{ github.event_name }}-${{ (github.event_name == 'schedule' || github.event_name == 'workflow_dispatch') && github.run_id || github.ref }}-${{ github.repository }}
```

`github.run_id` enters **only** on the two baseline events. For `push` the
conditional resolves to `github.ref` and for `pull_request` to the pull request
number, so both keys are byte-identical to the keys they are today. Latest-only
push dedupe (#822) and pull request dedupe are untouched, and this is asserted
as an equality rather than described.

### The cost, stated

The lane's own cost rises by the runs that are no longer discarded: **2 in 39,
about 5 percent**, or roughly 18 job-minutes per day against 345 per run at 6
runs per day. The push and pull request lanes cost exactly what they cost today.

The lane's peak concurrency rises from 1 to `ceil(wall / 4 h)`, which is 3 at
the worst wall observed. Total demand does not rise with it. Serialization
deferred the work; it did not remove it, except in the 2 cases where it deleted
the work and the verdict together.

### Rejected: widen the cron instead

A 12-hour period exceeds the worst wall observed (9 h 28 min) and would make the
queue depth two unreachable today. It is rejected as the primary fix and left
for the operator as a separate decision, for three reasons.

1. It is probabilistic. The wall is set by pool contention, which this workflow
   does not control. A period that clears today's worst wall does not clear
   tomorrow's, and the failure mode when it does not clear is silent.
2. It spends the property the four-hour cron was chosen for. At 55 pushes per
   day a 12-hour cadence makes a newly red baseline name about 27 commits
   instead of about 9.
3. The contention it responds to is one day old. Scheduled runs from
   `2026-08-12` to `2026-08-17` took 98 to 130 minutes. The blowup starts on
   `2026-08-18`. Re-tuning a cadence on one day of data is premature, and the
   per-run key removes the harm rather than the symptom.

The cron therefore stays at `17 */4 * * *`. If the operator wants the cost cut,
`17 */8 * * *` halves the lane's job-minutes and keeps the red range at about 18
commits; the reason belongs in this spec's `## Outcome` when it is taken.

### Rejected: make `push` non-cancellable

See *Our baseline* item 6.

### Rejected: a separate post-merge baseline workflow

`main-verifiability.md` rejected `workflow_call` on drift and permission
grounds, and nothing here changes that. `workflow_dispatch` already pins a
baseline on a chosen SHA.

## Tests

`tests/scripts/test_main_baseline.py` already evaluates this workflow rather
than grepping it, and this change extends that half. Three additions and two
strengthenings, all offline.

**The resolver becomes value-valued.** `resolve_boolean` handles the boolean
subset. `resolve_group` is added beside it for the same expression grammar
extended with contexts that carry a value, and with GitHub's truthiness for
`&&` and `||`. An unrecognised token still raises rather than resolving to a
guess. Every group assertion below reads a fully resolved key, so a conditional
in the key is evaluated, not matched.

1. `test_no_non_cancellable_group_can_be_evicted` -- **the new invariant**, over
   all 11 blocks including the workflow-level one. For each event where
   `cancel-in-progress` resolves to `false`, the resolved group must contain the
   run identity. This is the test that is red before the change.
2. `test_the_workflow_level_group_is_unique_per_baseline_run` -- the same
   statement for the workflow level alone, in the terms of the two runs that
   were discarded.
3. `test_a_baseline_group_is_unique_per_run_and_a_contributor_group_is_not` --
   both directions in one assertion, so a change that makes every key unique
   per run and kills push dedupe is red, and so is a change that makes no key
   unique.
4. `test_each_resolved_group_is_the_base_key_plus_the_event_constant`
   **strengthened**: it resolved only `${{ github.event_name }}` and compared
   the half-resolved string. It now resolves the whole key. The push and pull
   request direction keeps its equality against the base revision's key, so the
   `${{ github.sha }}` mutation it was built for stays red, and the baseline
   direction gains the uniqueness assertion.
5. `test_no_group_key_carries_a_run_varying_token` **strengthened**: it blocked
   the token by substring, for every event. It now blocks the token in the key
   **as resolved for `push` and `pull_request`**, which is the dedupe it was
   protecting, and requires it in the key as resolved for the baseline events,
   which is the dedupe that must not exist. The old form and the new form
   disagree only where the old form was wrong.

Neither strengthening removes an assertion. Each keeps its own direction and
adds the opposite one, and *Gates* records the mutation that proves the kept
direction still fires.

## Gates

| Gate | Command | Result |
|---|---|---|
| Red before | `python3 tests/scripts/test_main_baseline.py` at base | required RED on the new invariant |
| Green after | `python3 tests/scripts/test_main_baseline.py` | required GREEN |
| Mutation: `${{ github.sha }}` in a job key | test 4 | required RED |
| Mutation: run identity in every key, unconditionally | tests 3, 4, 5 | required RED |
| Mutation: drop the conditional from one key | tests 1, 2, 3 | required RED |
| YAML shape | `python3 tests/scripts/test_main_baseline.py::WorkflowShapeTests`, `scripts/check-release-workflow.py`, `scripts/check-container-workflow.py`, `scripts/check-container-matrix.py` | required GREEN |
| Record | `python3 scripts/check-agent-record.py`, `python3 tests/scripts/test_agent_record.py` | required GREEN |
| Preflight | `scripts/agent-preflight.sh --staged` | required GREEN |
| Post-merge | the first scheduled run after merge reaches a completed verdict | **PENDING -- cannot be established before merge** |

**What the pre-merge evidence cannot establish.** The pull request's own run is a
`pull_request` event, so it exercises neither the `schedule` trigger nor the
main-branch grouping. GitHub fires `schedule` only for the workflow committed on
the default branch. Everything green before merge is static, and
`main-verifiability.md` already paid for treating a read workflow as a run one.

The claim only the post-merge observation settles: a scheduled run that starts
while another scheduled run is in progress reaches a completed verdict rather
than `cancelled` with zero jobs. Two consecutive scheduled runs whose walls
overlap are needed to see it, and that condition occurred 8 times in the 39-run
window.

## Risks/decisions

| # | Risk | Handling |
|---|---|---|
| 1 | The per-run key kills push or pull request dedupe | The conditional admits `github.run_id` on `schedule` and `workflow_dispatch` only. Tests 3, 4 and 5 assert the push and pull request keys resolve to today's keys exactly, and the `${{ github.sha }}` mutation is rerun to prove the equality still fires. |
| 2 | Overlapping baselines raise peak runner demand and make the contention worse | Total demand rises about 5 percent, which is the discarded runs. Peak rises to 3 at the worst observed wall. The cron is the lever if the operator wants the cut, and *Design* states its price. |
| 3 | Three concurrent baselines reach queue depth two at JOB level and evict a job | Needs a wall above 8 hours at a 4-hour cron. Worst observed is 9 h 28 min, so it is reachable rather than hypothetical. Job-level keys carry the same conditional, so the depth is one per run and the eviction cannot happen. |
| 4 | The lane stays RED and the change looks ineffective | It does stay red, on [#584](https://github.com/mudler/vllm.cpp/issues/584). A verdict of RED on a named job is the lane working. `main-verifiability.md` made the same statement on the day it landed. |
| 5 | A future job joins with a non-cancellable group and no run identity | Test 1 iterates every concurrency block found in the file, not a fixed list, so the new job is red in the pull request that adds it. |
| 6 | The change is invisible in the pull request's own CI | Stated in *Gates*, not implied. The post-merge observation is owed. |

## Owed

* [#1316](https://github.com/mudler/vllm.cpp/issues/1316) -- `main-baseline.py`
  renders a run that executed zero jobs as `RED` with all 11 jobs `missing`,
  which is a verdict about a tree the run never checked out. Found while reading
  the two evicted runs; it is fail-closed, so it is filed rather than folded in,
  and this change removes its only observed producer.
* The post-merge observation in *Gates*.

## Stop conditions

Stop and report rather than pressing on if:

* the change cannot leave the `push` and `pull_request` keys byte-identical;
* making the lane finish needs any assertion weakened, any test muted, or
  `continue-on-error` relaxed for the push or pull request lanes;
* fixing [#584](https://github.com/mudler/vllm.cpp/issues/584) starts to look
  necessary. It is not. The lane reporting it is the lane working.
