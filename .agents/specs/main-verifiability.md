# `main` is not verified by its own CI — give it a known-good baseline

Issue: [#274](https://github.com/mudler/vllm.cpp/issues/274)
Row: `—` (no matrix row owns CI infrastructure; see *Dependencies*)
Branch: `row/CI-MAIN-VERIFIABILITY`
Base SHA: `0eb049f7e3fe522a1e8763c59be5bcfbbab53139`

## Scope

**In.** Make it possible to answer, without running the suite yourself:

> *What is the last SHA where `main` was fully green, and what failed?*

That means (1) a lane in which the full suite actually **runs to completion** on
`main`, and (2) a way to **read** that lane's verdict.

**Out.** Fixing the six sanitizer failures (`test_load_direct_upload`,
`test_llama_embedding_fold`, `test_laguna_nvfp4_loader`, `test_llm_engine`,
`test_openai_api_server`, `test_capi`) and the TSan reds. Those are
[#274](https://github.com/mudler/vllm.cpp/issues/274) finding #1 and
[#301](https://github.com/mudler/vllm.cpp/issues/301), triaged in parallel. No
product code, no test code, no change to what any gate asserts.

**This change makes the baseline lane RED on day one, and that is the point.**
Muting the six to get a green first run would produce a baseline that certifies
nothing. A baseline whose first verdict is "these six are broken, at this SHA"
is worth more than a green one that cannot see them.

## Upstream chain

**None. This is project infrastructure with no vLLM counterpart.** vLLM is the
reference for engine behavior, not for this repository's GitHub Actions
configuration or its record protocol. There is nothing upstream to mirror, no
`file:line` to port from, and no upstream test to preserve. vLLM's own CI is a
Buildkite/GHA hybrid gating a different repository layout with different runners;
copying its shape would be cargo cult, not mirroring.

## Our baseline — what actually happens on `main` today

Three facts, each measured at `0eb049f7`, not inferred.

### 1. The full suite essentially never completes on `main`

Forty consecutive `ci` runs on `main` (`gh run list --branch main --workflow
ci.yml --limit 40`, all `event=push`): 26 `cancelled`, 12 `failure`, **1
`success`**, 1 still in progress. Those 40 pushes span **17.43 hours — a rate of
55 pushes to `main` per day.**

### 2. The cause is the JOB-level concurrency groups, not the workflow-level one

[#274's third comment](https://github.com/mudler/vllm.cpp/issues/274) correctly
observed that the workflow-level block at `.github/workflows/ci.yml:26-28` is
already safe for `push` — the group falls back to `github.sha` (unique per
commit) and `cancel-in-progress` evaluates `false`. It then concluded the
cancellation must be external ("a repository/org-level runner concurrency or
spending cap … an automation cancelling older runs") and that the next person
should **not** touch the concurrency config.

**That conclusion is wrong, and the correction is the load-bearing finding of
this spec.** Nine *job-level* groups are keyed on `github.ref` with
`cancel-in-progress: true`:

```
ci.yml:34   agent-record            ci-agent-record-${{ github.ref }}-…
ci.yml:322  cuda-arch-features      ci-cuda-arch-features-${{ github.ref }}-…
ci.yml:348  cuda-fat-build          ci-cuda-fat-build-${{ github.ref }}-…
ci.yml:420  vulkan-spirv-freshness  ci-vulkan-spirv-freshness-${{ github.ref }}-…
ci.yml:454  build-test-vulkan       ci-build-test-vulkan-${{ github.ref }}-…
ci.yml:496  device-leakage          ci-device-leakage-${{ github.ref }}-…
ci.yml:508  build-test-cpu          ci-build-test-cpu-${{ github.ref }}-…
ci.yml:559  build-test-cpu-arm64    ci-build-test-cpu-arm64-${{ github.ref }}-…
ci.yml:643  sanitize-cpu (matrix)   ci-sanitize-cpu-${{ matrix.lane }}-${{ github.ref }}-…
```

For every push to `main`, `github.ref` is the constant `refs/heads/main`, so
consecutive pushes **do** share these groups and **do** cancel each other. This
is deliberate and documented (`ci.yml:9-14`): it collapses superseded main
pushes to save runner time. It is not a bug. It is simply incompatible with ever
finishing a run.

The timestamps prove the mechanism rather than assuming it. Every grouped job of
run `31485402200` was cancelled at `11:46:33`; run `31488132224` was created at
`11:46:32`. Every grouped job of run `31482845117` was cancelled at `11:05:12`;
run `31485054749` was created at `11:05:11`. The cancel instant equals the next
push's start instant, three times out of three.

It also explains the "partial cancellation" the comment found puzzling. Job-level
groups cancel *individual jobs*, not the run, so jobs that already finished keep
their result and only the in-flight ones die. In run `31488132224` the survivors
are exactly the fast jobs plus the two DIFF-scoped jobs that carry no group at
all (`documentation-checkpoint`, `commit-protocol-tag`), and the casualties are
exactly the slow grouped ones (`cuda-fat-build`, `build-test-cpu`, both
`sanitize-cpu` lanes). Short-vs-long *is* the split the comment identified; the
cause is runtime crossing the next-push interval, not an external canceller.

No external canceller, no spending cap, and no job-level timeout is involved.
The one relevant timeout in the file is `cuda-fat-build`'s `timeout-minutes:
180`, which nothing has come near.

### 3. `sanitize-cpu` is `continue-on-error`, so a run's conclusion can lie

`ci.yml:640` sets `continue-on-error: true` on the sanitizer matrix, so its
failure does not fail the run. The single `success` run in the window,
`31448896841` at `5812b8b6`:

```
run conclusion: success
  success  agent-record, build-test-cpu, build-test-cpu-arm64, build-test-vulkan,
           commit-protocol-tag, cuda-arch-features, cuda-fat-build,
           device-leakage, documentation-checkpoint, vulkan-spirv-freshness
  failure  sanitize-cpu (address,undefined)
  failure  sanitize-cpu (thread)
```

So the *only* completed `main` run in the last 40 reports `success` while both
sanitizer lanes are red. Any baseline that reads the run-level conclusion would
publish `5812b8b6` as green. It is not green. **The baseline must read per-job
conclusions.** This is not hypothetical — it is the exact SHA a reader would
land on today.

That same run also gives the cost figure: wall-clock `5944000 ms` (99 min),
dominated by `cuda-fat-build` at 99 min; `sanitize-cpu (address,undefined)` 33
min. The repository is public (`"visibility": "public"`), so billable runner
minutes are `0`.

## Port map

**None — nothing is ported.** There is no upstream artifact behind any file in
this change. Written from scratch, and recorded as such here rather than in
`.agents/porting-inventory.md`, which inventories engine/kernel ports from vLLM
and has no category for CI configuration.

| Local artifact | Origin |
|---|---|
| `.github/workflows/ci.yml` (edits) | from scratch |
| `scripts/main-baseline.py` | from scratch; **shape** mirrors the in-repo precedent `scripts/now.py` (derive at read time, degrade to `REMOTE_UNVERIFIED`, never fail on a missing network) |
| `tests/scripts/test_main_baseline.py` | from scratch |

## Design

### Chosen: a scheduled + on-demand baseline lane inside `ci.yml` (option b, hybrid)

Four parts.

**1. `ci.yml` gains `schedule` and `workflow_dispatch` triggers.** Cron
`17 */4 * * *` — six runs a day, off the top of the hour to miss the global
stampede. `workflow_dispatch` is the hybrid half: an operator who has just
merged something they care about can pin a baseline on that SHA immediately,
which recovers option (a)'s "right after this merge" property without paying for
it on every merge.

**2. The job-level groups gain an event discriminator.**

```yaml
group: ci-<job>-${{ github.event_name }}-${{ github.ref }}-${{ github.repository }}
cancel-in-progress: ${{ github.event_name != 'schedule' && github.event_name != 'workflow_dispatch' }}
```

Without this the whole design is dead on arrival: a scheduled run also has
`github.ref == refs/heads/main`, so the very next push to `main` would cancel the
baseline's expensive jobs exactly as it cancels the previous push's. The
discriminator puts the baseline in its own group, and `cancel-in-progress: false`
makes a second baseline queue behind the first rather than kill it.

For `push` and `pull_request` the added token is a constant, so **grouping and
cancellation behavior for ordinary runs is unchanged**. That is the property to
verify in review, and `tests/scripts/test_main_baseline.py` asserts it against
the workflow text. The workflow-level group takes the same discriminator, for
the same reason at run scope: a `workflow_dispatch` on a SHA that was just pushed
would otherwise share group `ci-<sha>-<repo>` and queue behind the push run.

**3. The two DIFF-scoped jobs are skipped on the baseline lane.**
`documentation-checkpoint` and `commit-protocol-tag` derive their range from
`github.event.before..github.sha`. `schedule` and `workflow_dispatch` payloads
have no `before`, so the range is meaningless; they would run against an empty
base. They are per-push gates and the push already ran them (they carry no
concurrency group precisely so they always complete — `ci.yml:216-218`). The
baseline reports on the **tree** at a SHA, and says so.

**4. A `baseline-summary` job publishes the verdict.** Runs only on
`schedule`/`workflow_dispatch`, `needs` every tree-scoped job with `if:
always()`, and calls `scripts/main-baseline.py --run-id ${{ github.run_id }}
--emit-summary`.

It reads the per-job conclusions **from the Actions API for its own run**, not
from the `needs` context. Two reasons: `continue-on-error: true` on
`sanitize-cpu` means the `needs.<job>.result` it exposes is not the job's real
conclusion, which is the trap that finding 3 above is made of; and the API path
is the same code the offline reader runs, so there is one implementation of
"what does green mean" instead of two that can drift.

**The baseline job fails if any job it covers is not `success`, sanitizers
included.** This does not make the sanitizer binding for contributors —
`baseline-summary` never runs on a `pull_request` or a `push`, so no PR can be
blocked by it and `continue-on-error` on the sanitizer job is untouched. It makes
the *baseline* honest: a lane that reported green while `sanitize-cpu` was red
would reproduce, in a new place, precisely the defect #274 was filed about.

### Reading it back: `scripts/main-baseline.py`

```sh
scripts/main-baseline.py              # last-green SHA + newest run's failures
scripts/main-baseline.py --json
scripts/main-baseline.py --limit 20
```

Prints, per baseline run: SHA, timestamp, verdict, and the failing job names;
then the newest verdict and the most recent fully-green SHA — or an explicit
"no fully green baseline in the last N runs", which is the honest answer on day
one and must not be dressed up as anything else.

**Derived at read time. It writes no file and nothing is committed.** AGENTS.md's
"no surface that every PR must write" admits three shapes — one file per row,
append-only, or derived — and a committed `LAST_GREEN.md` would be the worst
possible instance of the shape it forbids: a single line that every merge wants
to rewrite. `scripts/now.py` is the in-repo precedent and this mirrors it,
including degrading to `REMOTE_UNVERIFIED` on an unreachable remote rather than
rendering an absence as "no failures".

### Rejected: option (a), a non-cancellable post-merge run per merge

Rejected on measured cost, not on taste.

`main` took **40 pushes in 17.43 hours — 55 per day**, including two 83 seconds
apart (`31478659162` at `09:39:09` / `31478575965` at `09:38:01`) and three
inside 41 minutes (`11:05`, `11:09`, `11:46`). A full run is 99 minutes
wall-clock and roughly 3.5 hours of summed job time. Making every main push
non-cancellable means ~55 overlapping 99-minute runs a day — on the order of
**190 hours of daily job time** — on a pool already visibly contended: run
`31485402200` sat entirely queued for 37 minutes and was cancelled without a
single job ever starting. Adding 55 concurrent full suites to that pool makes
*PR* feedback slower, taxing every contributor to answer a question asked a few
times a day. At that push rate a per-merge baseline is also self-defeating:
runs would still be superseded faster than they finish, just without being
cancelled, so the newest completed baseline would routinely be several commits
stale anyway — the exact property (a) is supposed to buy.

It also directly reverses a deliberate, documented decision (`ci.yml:9-14`) with
no new evidence against it. The job-level groups are correct for the push lane;
they are only wrong when applied to a lane whose entire purpose is to finish.

The property (a) has and (b) lacks is per-merge attribution: at 55 pushes/day a
4-hour cadence means a newly-red baseline names a *range* of roughly 9 commits,
not one. `workflow_dispatch` recovers the important half of that at a fraction of
the cost, and `git bisect` over nine commits with a known-good end is a far
better position than today's, which is no known-good end at all.

### Also rejected

**A separate `main-baseline.yml` calling `ci.yml` via `workflow_call`.** In a
called workflow `github.ref` is still the caller's — `refs/heads/main` — so it
needs the identical event discriminator to avoid being cancelled by a push, and
then adds `workflow_call` permission/secret inheritance semantics for no gain.
Duplicating the job definitions instead would create two suites that drift.

**Cadences other than 4h.** At 55 pushes/day, daily leaves a ~55-commit range —
barely better than nothing. Hourly is 24 × 99 min against an already contended
pool, for a question asked a few times a day, and consecutive runs would overlap.
Four hours is where the range (~9 commits, three bisect steps) stays workable at
6 runs/day and successive runs still do not overlap a 99-minute suite. It is one
cron line; if the post-merge observation shows contention, it widens and the
reason goes in `## Outcome`.

## Tests to port

**None to port** — see *Upstream chain*. Written from scratch, in
`tests/scripts/test_main_baseline.py`, all offline over fixture payloads:

1. **Verdict is per-job, not run-level.** The exact `31448896841` shape (run
   conclusion `success`, both sanitizer lanes `failure`) must be reported RED.
   This is the mutation that matters: a reader of `run.conclusion` passes every
   other test in the file and fails this one.
2. **`--emit-summary` exit code** is non-zero for that shape and zero for
   all-green.
3. **Last-green selection** picks the newest all-green run, and reports the
   absence explicitly when there is none.
4. **`cancelled` and `skipped` are not green.** A cancelled job must never be
   folded into a pass.
5. **Workflow shape**, asserted against `.github/workflows/ci.yml`: every
   job-level concurrency group carries the event discriminator;
   `baseline-summary` is gated to schedule/dispatch; and `schedule` /
   `workflow_dispatch` triggers exist.
6. **Offline degradation** returns `REMOTE_UNVERIFIED` rather than an empty
   green, and so does absence — "no completed baseline run found" exits
   NON-ZERO, because `main-baseline.py && echo ok` must not print `ok` when
   nothing has ever run.
7. **Expected-job coverage.** A payload missing eight of the nine covered jobs
   is RED and names them, rather than printing GREEN with `jobs covered: 1`.
   `EXPECTED_JOBS` is cross-checked against `baseline-summary`'s `needs:` list,
   so renaming a job reds this suite in the PR that renames it.
8. **In-progress is `pending`, not `failed`.** Both are non-green; only one is
   a true statement about the job.

**Tests 5, 9-12 EVALUATE the workflow; they do not grep it.** The first version
of this suite checked substrings and four planted defects walked through it, so
each is now an assertion about a *resolved value*:

9.  **Cancellation policy**, resolved to a boolean per event by a small
    GitHub-expression evaluator: `push`/`pull_request` → `true`,
    `schedule`/`workflow_dispatch` → `false`, on all nine job groups and on the
    workflow-level key. Inverting the polarity to
    `== 'schedule' || == 'workflow_dispatch'` preserves every substring and
    reverses the meaning of all nine.
10. **Group keys pinned as an equality** against the base revision's key
    (`git show 0eb049f7:.github/workflows/ci.yml`) with the constant `-<event>`
    inserted — which is the entire content of the "`push`/`pull_request`
    grouping is unchanged" claim, and the only form of it that catches an added
    `${{ github.sha }}`. Plus a by-name blocklist of run-varying tokens.
11. **The verdict job cannot swallow its own failure**: no `|| true`, `; true`
    or `set +e` in the `--emit-summary` step body, and no `continue-on-error` on
    the job. Either one rebuilds the `sanitize-cpu` defect this row exists to
    expose, one level up.
12. **`agent-record`'s two `github.event.before` consumers are EXECUTED** under
    a `python3` shim on all four events — see *Gates*.
13. **This suite is registered** in `scripts/agent-preflight.sh`'s `SUITES` and
    in the `agent-record` job, and that job is unconditional. Guarding a claim
    with a test nothing runs is not guarding it.

## Gates

| Gate | Command | Result |
|---|---|---|
| New suite | `python3 tests/scripts/test_main_baseline.py` | required GREEN |
| Record | `python3 scripts/check-agent-record.py && python3 tests/scripts/test_agent_record.py` | required GREEN |
| CI-shape suites | `python3 tests/scripts/test_agent_gates.py`, `python3 scripts/check-test-registration.py`, `python3 scripts/check-release-workflow.py`, `python3 scripts/check-container-workflow.py`, `python3 scripts/check-container-matrix.py` | required GREEN |
| Preflight | `scripts/agent-preflight.sh --staged` | required GREEN |
| YAML parse | `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` | required GREEN |
| Post-merge observation | first `schedule` run after merge completes without cancellation | **PENDING — cannot be established before merge** |

**What the pre-merge evidence cannot establish, stated plainly.** A CI-config
change cannot be proven by the PR's own CI run: the PR run is a `pull_request`
event on a PR ref, so it exercises neither the `schedule` trigger (GitHub only
fires `schedule` on the default branch, and only for the version of the workflow
committed there) nor the main-branch grouping. Everything green before merge is
static: the YAML parses and the suites pass.

**"The workflow text says what it should" was itself unearned, and this row
proved it.** That sentence stood here through the first implementation, and the
first fresh review falsified it: `agent-record` is in `baseline-summary`'s
`needs:`, carried the same diff-scoped `github.event.before` logic that was
correctly guarded out of `documentation-checkpoint` and `commit-protocol-tag`,
and carried no event guard of its own. On a `schedule` payload `PUSH_BASE`
renders empty and, under `set -eu`, `check-commit-trailers.py --range ..<sha>`
exits 2 — so the lane could never have published anything but RED, and every
static gate above was green while that was true. Reading a workflow is not
running it. The repair guards the two range-scoped calls in place (the rest of
that job is tree-scoped and the baseline needs it) and
`tests/scripts/test_main_baseline.py::AgentRecordDiffRangeTests` **executes**
both step bodies under a `python3` shim, on all four events, asserting both
directions: no range-scoped call where there is no range, and the range-scoped
calls still happen on `push` and `pull_request`.

The claims that only a post-merge observation can settle:

* a scheduled run actually fires on the chosen cron;
* it survives concurrent pushes to `main` instead of being cancelled;
* `baseline-summary` obtains the Actions API result for its own run with the
  default `GITHUB_TOKEN` and `actions: read`;
* `scripts/main-baseline.py` finds and parses real baseline runs;
* **every job in `baseline-summary`'s `needs:` can reach a green conclusion on a
  `schedule` payload.** `agent-record` demonstrably could not, and it was the
  only one anybody had reason to doubt only because it is the one carrying
  diff-scoped logic — but no static gate here distinguishes "runs green on the
  push lane" from "runs green on the baseline lane", so the first scheduled run
  is where each of the nine is observed for the first time. Any job that comes
  back RED for a *lane* reason rather than a *tree* reason is this same defect
  in a different job, and is repaired the same way;
* the verdict's `missing` bucket stays empty against a real payload — i.e. the
  job names the Actions API reports still match `EXPECTED_JOBS` (matrix lanes
  arrive as `sanitize-cpu (thread)`, which is why the match is by id prefix).

`## Outcome` is not written until those six are observed. The first scheduled
run is expected **RED** on the six sanitizer failures — that is the correct
first verdict, not a regression introduced here.

## Dependencies

* [#274](https://github.com/mudler/vllm.cpp/issues/274) — this work; finding #2.
* [#301](https://github.com/mudler/vllm.cpp/issues/301) / #274 finding #1 — the
  six sanitizer failures. Independent, in flight in parallel, deliberately
  untouched. They are what the first baseline will report.
* [#408](https://github.com/mudler/vllm.cpp/issues/408) — filed while repairing
  this row's review findings: 12 of 54 `tests/scripts` suites are executed by
  nothing, this row's own suite having been one of them. The instance is fixed
  here; the class needs a checker-semantics change with its own spec and
  mutation evidence, which is why it is filed rather than folded in.
* **No matrix row owns CI infrastructure.** Every `ENG-*` row in
  `.agents/engine-matrix.md` is an engine capability; no row, in any matrix,
  covers `.github/workflows/`. The issue table admits `—` in its Row column
  (`ISSUE_ROW` in `scripts/check-agent-record.py:1070`, and eight existing rows
  use it), so #274 is filed with `—` rather than inventing a row or
  misattributing CI to an unrelated engine row. If CI later earns a row, this
  spec and that table move together. The session claimed `ENG-CI` as its role
  row for traceability; `scripts/agent-role.py` does not validate row IDs
  against the matrices, so that claim string asserts nothing about the record —
  the `—` in the issue table is the honest statement.

## Work breakdown

1. **Spec, committed alone, first.** (this file)
2. **Record.** `#274` into the `## Open issues` table of
   `.agents/roadmap_v1.md`, Row `—`, kind `bug`, in sorted position.
3. **`scripts/main-baseline.py`** + `tests/scripts/test_main_baseline.py`,
   test written first and observed RED against a stub.
4. **`.github/workflows/ci.yml`**: triggers, nine job groups + the workflow
   group, the two diff-job skips, `baseline-summary`.
5. **`.agents/verification.md`**: how to read the baseline (a task guide, which
   is where "how do I do this job" belongs; not a new record surface).
6. **Gates, push, PR.**

## Risks/decisions

| # | Risk | Handling |
|---|---|---|
| 1 | Touching the job groups changes push/PR cancellation | The added token is constant for `push` and for `pull_request`, so grouping is bit-identical for both. Asserted by test 5, and reviewable by eye. |
| 2 | The baseline lane itself gets cancelled — the failure that would silently void the whole row | Own group (event discriminator) plus `cancel-in-progress: false`. Only observable post-merge; named in *Gates* as PENDING. |
| 3 | Reading `needs.<job>.result` would report `success` for a `continue-on-error` job | Not used. The verdict comes from the Actions API for the run. Test 1 is exactly this mutation, on the real `31448896841` shape. |
| 4 | `baseline-summary` failing blocks contributors | It cannot run on `pull_request` or `push`. Gated by `if:` and asserted by test 5. |
| 5 | Six full suites/day worsen queue contention | 99 min wall-clock, 0 billable minutes (public repo). Cadence is one cron line; if contention is observed post-merge, widen it and record the reason in `## Outcome`. |
| 6 | The scheduled trigger goes dormant — GitHub disables cron on repos with 60 days of no activity | Not a live risk at 20 pushes/day, and `scripts/main-baseline.py` shows the newest baseline's **date**, so a stopped lane reads as stale rather than as green. |
| 7 | Cron drift under load; `schedule` is best-effort | Staleness is visible for the same reason as #6. |
| 8 | A baseline pins a SHA nobody can reproduce | It pins the SHA the run checked out, and the run's own logs are the evidence. |
| 9 | A covered job cannot be green on the baseline lane at all, so the verdict is permanently RED for a reason that is not about the tree | Found in review, in `agent-record`; see *Gates*. Guarded, and `AgentRecordDiffRangeTests` executes the step bodies rather than reading them. The residual is the other eight jobs, which are observed for the first time by the first scheduled run — named in the owed-observation list. |
| 10 | **Recorded, not fixed.** `scripts/check-role-discipline.py` accepts `--base ""` and silently degrades: `commits_in_range` (`scripts/check-role-discipline.py:328-333`) cannot resolve the empty base, falls back to `[head]`, and prints `OK: every change on main arrived on a task branch` — a PASS covering ONE commit while looking like it covered a range | On the baseline lane the call is now skipped loudly instead, so this row does not rely on the degradation either way; the two lanes that do have a range (`push`, `pull_request`) are unaffected. The underlying checker behaviour is untouched here: it is a checker-semantics change, which under AGENTS.md needs its own spec and red-before evidence, and it is out of scope for a CI-lane row. Anyone giving that checker an empty base elsewhere gets a vacuous pass. |
| 11 | A suite that guards this row runs on no machine | Exactly what happened: `tests/scripts/test_main_baseline.py` shipped in neither `scripts/agent-preflight.sh`'s `SUITES` nor the `agent-record` job, so all of its tests ran nowhere. Registered in both, and `SuiteRegistrationTests` now asserts both registrations. The CLASS — 12 of 54 `tests/scripts` suites are executed by nothing, and `check-test-registration.py`'s fixed `REQUIRED_TESTS` cannot see it — is [#408](https://github.com/mudler/vllm.cpp/issues/408). |

**Decisions.** (i) Baseline in `ci.yml`, not a second workflow — one definition
of the suite, no drift. (ii) Sanitizers binding **on the baseline lane only**.
(iii) The verdict is derived, never committed. (iv) The two DIFF-scoped jobs are
out of the baseline's coverage, and the tool says which jobs it covered rather
than implying it covered everything.

## Evidence

Collected at base `0eb049f7`; every command is re-runnable.

* `gh run list --branch main --workflow ci.yml --limit 40 --json …` → 27
  cancelled / 12 failure / 1 success, all `event=push`.
* `gh run view 31488132224 --json jobs` / `31485402200` / `31482845117` → the
  cancel-instant-equals-next-push-start correlation, 3/3.
* `gh api …/runs/31448896841/jobs` → run `success`, both sanitizer lanes
  `failure`.
* `gh api …/runs/31448896841/timing` → `run_duration_ms: 5944000`; billable
  `total_ms: 0`.
* `gh api repos/mudler/vllm.cpp` → `"visibility": "public"`.
* `.github/workflows/ci.yml:26-28, 34, 322, 348, 420, 454, 496, 508, 559, 640,
  643` → the workflow-level group, the nine job groups, `continue-on-error`.
* Post-merge, owed: the first `schedule` run's id, its per-job conclusions, and
  `scripts/main-baseline.py` output against it.

## Stop conditions

Stop and report rather than pressing on if:

* making the baseline lane finish requires weakening any assertion, muting any
  test, or relaxing `continue-on-error` semantics for the push/PR lanes;
* the change cannot leave `push`/`pull_request` grouping byte-equivalent;
* fixing any of the six sanitizer failures starts to look necessary — it is not,
  and it belongs to the parallel triage;
* the first post-merge scheduled run is cancelled anyway, which would falsify the
  mechanism in *Our baseline* §2 and means the external-canceller hypothesis in
  #274's third comment deserves another look;
* a green verdict is ever produced while a covered job is red.

## Outcome

Written 2026-08-19 from 39 completed scheduled runs, `2026-08-12T17:10Z` to
`2026-08-19T04:49Z`. The follow-on repair is
[`baseline-lane-eviction.md`](baseline-lane-eviction.md).

**The lane works, and the 200-run window hides it.** `gh run list --workflow
ci.yml --limit 200` reads 175 cancelled, 17 failure, 0 success, and that window
spans 18 hours 14 minutes because the push and pull request lanes fill it. Read
per event, the lane published a verdict for **37 of 39 triggers**: 20 `success`,
17 `failure`, 2 `cancelled`. Median wall 118 minutes for a green run.

The six owed observations, each now made:

1. **A scheduled run fires on the cron.** 39 runs in 7 days at `17 */4 * * *`.
2. **It survives concurrent pushes.** No scheduled run was cancelled by a push
   in the window. The event discriminator holds.
3. **`baseline-summary` reads the Actions API for its own run** with the default
   `GITHUB_TOKEN` and `actions: read`. It published on every run that started.
4. **`scripts/main-baseline.py` finds and parses real baseline runs.** It
   renders SHA, verdict, failing jobs and the last fully green SHA.
5. **Every job in `needs:` reaches green on a `schedule` payload.** Run
   `32067210005` at `76f2a6d84e41` is all-green across the nine jobs of the
   original coverage. No job was RED for a lane reason rather than a tree one.
6. **The `missing` bucket stays empty against a real payload**, for every run
   that executed jobs. It fills only for a run that executed none, which is
   finding 8 below.

**Risk 2 was NOT handled, and that is this spec's error.** It named "the
baseline lane itself gets cancelled" and recorded the handling as "own group
plus `cancel-in-progress: false`". That covers one half of GitHub's concurrency
contract. The other half is the queue, which holds one pending run and cancels
it when a third arrives. Runs `32140419182` and `32206456661` were discarded
that way, with `startedAt: null` for every job, at the seconds their successors
were created. `cancel-in-progress` cancelled nothing. The stop condition "the
first post-merge scheduled run is cancelled anyway" fired, later than expected
and by a mechanism this spec did not consider.

**The cadence argument's premise expired.** "Successive runs still cannot
overlap a 99-minute suite" was true when measured. Run `32118587477` took 9 h 28
min with no predecessor to wait for, on 345 job-minutes of work, because a
3.8-minute `ubuntu-latest` job waited 3 h 59 min for a runner. 8 of 39 runs ran
over four hours. The cron is unchanged at `17 */4 * * *` and the reason is in
`baseline-lane-eviction.md`: the per-run key removes the harm, and re-tuning a
cadence on one day of contention is premature.

**What the lane reports, and why it is not a defect of the lane.** No fully green
baseline exists since [#503](https://github.com/mudler/vllm.cpp/issues/503) put
`windows-msvc-cpu` and `windows-msvc-vulkan` on the lane on 2026-08-17. Both
fail in 5 of 5 scheduled runs with
`test_openai_api_server.exe exited with status -1073740791` (`0xC0000409`,
`STATUS_STACK_BUFFER_OVERRUN`), which is
[#584](https://github.com/mudler/vllm.cpp/issues/584), open and pre-existing.
This is the same statement this spec made on the day it landed, one issue later.

**Rejected, and why.** Making `push` non-cancellable stays rejected on the same
arithmetic, restated with the current suite cost: 55 pushes/day at 345
job-minutes is about 316 hours of job time per day, on a pool that already makes
a 3.8-minute job wait 3 h 59 min.

## Owed

* Finding 8: `scripts/main-baseline.py` renders a run that executed zero jobs as
  `RED` with all 11 jobs `missing`, so `NEWEST BASELINE: RED at <sha>` names a
  tree the run never checked out. Filed as
  [#1316](https://github.com/mudler/vllm.cpp/issues/1316) and owned by
  [`baseline-lane-eviction.md`](baseline-lane-eviction.md), which removes its
  only observed producer.
