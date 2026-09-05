# Should a cheap record gate run on every push to `main`? — an analysis for the `main-verifiability` family

**This document is an analysis, not a new row.** It was drafted as
`ENG-GATE-ON-MAIN`, and the verification pass in `## Relation to
main-verifiability.md` below found that its diagnosis is substantially the one
[`main-verifiability.md`](main-verifiability.md) and `.github/workflows/ci.yml`'s
own header already carry. What is new is one design option, not a new problem.
Read it as a follow-on in that family, in the same place as
[`baseline-lane-eviction.md`](baseline-lane-eviction.md), and do not open a
competing row for it.

Issue: [**#2950**](https://github.com/mudler/vllm.cpp/issues/2950), filed against `GATE-CI-CONCURRENCY` rather than a row of its own, and it is not `#274`. See `## Owed`; AGENTS.md §"Every
change starts from an issue" is not satisfied by this document, and an issue must
exist before any implementation.

**Most of the issue numbers cited below no longer resolve.** Verified
2026-09-05 with unauthenticated `GET /repos/mudler/vllm.cpp/issues/<n>`, which
separates a deleted issue from one this token cannot see: #274, #301, #503,
#584, #822, #863, #873, #874, #1316, #1809, #1844, #2101, #2103, #2157, #2274,
#2304, #2307, #2313, #2322, #2329, #2332, #2342, #2349, #2350, #2357 and #2359
all return **404 to an anonymous reader** — 26 numbers, each probed
individually — while #2389, #2401, #2404, #2429 and #2473 return 200. The repository holds 532 issues over a number space reaching
2937, so roughly four numbers in five are gone. The links are kept because they
still identify the commits, pull requests and specs that cite them — `ci.yml`
itself cites #274 and #822 — but **a reader cannot open them, and this document
must not be read as resting on their contents.** Nearest prior art that does
resolve: [#2389](https://github.com/mudler/vllm.cpp/issues/2389) (closed —
`check-env-doc` was one-directional),
[#2401](https://github.com/mudler/vllm.cpp/issues/2401) (nothing compiles before
a push), [#2404](https://github.com/mudler/vllm.cpp/issues/2404) (MSVC red on
every PR), [#2429](https://github.com/mudler/vllm.cpp/issues/2429)
(`test_check_gate_commands` red on `main`). The gone ones, by subject: #274
(`main` has no baseline, owned by `main-verifiability.md`), #584 (the Windows
runtime crash), #822 / #863 (the latest-only push lane, owned by
`ci-concurrency.md`), #1809 (the walk base and its floor, owned by
`ci-enforcement-floor.md`).

Row: `—`. No matrix row owns CI infrastructure; this follows the precedent set by
[`main-verifiability.md`](main-verifiability.md).

Branch: `row/ENG-GATE-ON-MAIN`. Base SHA: `d023e3357b907927fb6d459f83d21b4729b78d84`.
Verified against the tree and the GitHub API on 2026-09-05; `## Evidence`
carries a `verified` column recording what reproduced, what was corrected, and
what could not be re-measured.

**Verdict up front.** The framing in the request — "preflight gates a branch,
nothing gates `main`" — is right about the outcome and wrong about the
mechanism, and the correction changes what to build.

- The record gates are **already wired onto the push lane**. `agent-record`
  deliberately carries no job-level `if:` ([`ci.yml:133-155`](../../.github/workflows/ci.yml)),
  so `check-env-doc.py` and `check-gate-commands.py` run on every push to `main`
  by construction. They are **cancelled, not absent**: over the last 40 pushes to
  `main`, 33 runs executed *zero jobs* and `agent-record` reached a verdict
  **twice** (re-measured 2026-09-05 over the identical window).
- `main` carries **zero required status checks**
  (`required_status_checks = {checks: [], contexts: [], enforcement_level: "off"}`),
  so no lane can block anything. Pull request
  [#2412](https://github.com/mudler/vllm.cpp/pull/2412) was merged on
  2026-08-31T15:27:19Z with `agent-record` reading `CANCELLED`, and it is the
  pull request that landed instance 4.
- **52 of the last 100 first-parent commits on `main` belong to no pull request
  at all.** For those, the pull-request lane — which carries every gate this
  repository trusts — was never a gate in the first place.
- The hybrid half of #274's own design has **never produced a baseline**: the
  API reports exactly **one** `workflow_dispatch` run of `ci.yml` ever
  (`33618253298`, 2026-09-02T10:12:29Z), it was fired on the row branch
  `row/DEBTFIX-GLUE-RANK-COLLISION` rather than on `main`, and it was
  **cancelled**. So the mechanism has run once and pinned nothing. An earlier
  draft of this document said "zero, ever"; that was wrong, and the corrected
  count does not change the argument.

So the recommendation is narrow, and three of the five named instances are inside
it. Two are not, and this document says so rather than rounding up.

## Scope

| Field | Content |
|---|---|
| In | An answer to: which `scripts/agent-preflight.sh` gates should run on every push to `main`, at what cost, and what the counter-argument is. Design options, a recommendation, and an explicit split between what moves and what deliberately does not |
| Out | Any workflow edit, any checker change, any product code. This document is the analysis and the decision record that AGENTS.md §"Spec before code" requires *before* the change exists. It also does not repair the five heavy jobs that are red on `main` today; that is the prerequisite named under `## The case against`, and no row here owns it |
| Not this row | #584 (the Windows runtime crash), #2401's compile gate (landed), #274's baseline lane (landed), the `sanitize-cpu` reds (#301) |

## Upstream chain

**None. Nothing is ported.** vLLM is the reference for engine behaviour, not for
this repository's GitHub Actions configuration or its record protocol. There is
no upstream `file:line` behind any claim below and no upstream test to preserve.
Recorded here rather than in [`../porting-inventory.md`](../porting-inventory.md),
which has no category for CI configuration —
[`main-verifiability.md`](main-verifiability.md) made the same declaration for
the same reason.

## Relation to `main-verifiability.md` (#274) — most of this is not new

The verification pass was asked one question directly: is this a genuinely new
row, or a duplicate of the one that already owns `main`'s verifiability? **The
answer is split, and the half that matters for process says: do not open a
competing row.**

**The diagnosis is a duplicate.** `.github/workflows/ci.yml`'s own header, lines
17-33, already states this document's Q1 and Q2-B in full: *"The push lane below
cannot answer 'is main green?', by construction. Every expensive job carries a
job-level group keyed on `github.ref` … Measured over 40 consecutive main runs at
`0eb049f7`: 26 cancelled, 12 failure, 1 success."*
[`main-verifiability.md`](main-verifiability.md) §"Our baseline" carries the same
finding with the same mechanism, the same cancel-instant timestamp proof, and a
push rate of 55/day against this document's 46.8. Q2-C — that the baseline has
lost its discriminating power — is stated in that spec's own `## Outcome`: *"No
fully green baseline exists since #503 put `windows-msvc-cpu` and
`windows-msvc-vulkan` on the lane on 2026-08-17."* Three of this document's five
analytical sections re-derive work that is already recorded, and AGENTS.md
§"History is git" says not to.

**Four specs already own the pieces**, and none of them is this one:

| Owner | Spec | Row |
|---|---|---|
| Does `main` have a verdict at all | [`main-verifiability.md`](main-verifiability.md) | `—` (#274, **deleted**) |
| Whether a superseded push may be cancelled | [`ci-concurrency.md`](ci-concurrency.md) | `GATE-CI-CONCURRENCY` (#822, #863) |
| The walk base and its floor | [`ci-enforcement-floor.md`](ci-enforcement-floor.md) | `GATE-CI-ENFORCEMENT-FLOOR` (#1809) |
| The baseline lane's queue eviction | [`baseline-lane-eviction.md`](baseline-lane-eviction.md) | `—` (#274) |

**One thing is new, and it is the recommendation.** #274 rejected a
non-cancellable per-push lane on measured cost — the *full suite*, ~99 min wall
and ~3.5 job-hours per run, ~190 job-hours/day, restated in its `## Outcome` as
~316. Option 2 below is a different object: a **record-only** job at ≈ 4 min and
≈ 3.1 job-hours/day, 1.6% of what was rejected. #274 did not weigh that, because
that shape was not on the table. Asking whether a cheap non-cancellable push job
is worth 3.1 job-hours/day is a live question the family has not answered.

**So the disposition is:** keep this as an analysis in the
`main-verifiability` family, exactly as `baseline-lane-eviction.md` is, and
**do not create an `ENG-GATE-ON-MAIN` row.** The implementing change, if the
recommendation survives, belongs to `ci-concurrency.md`'s row — it is a change to
what the push lane cancels — with this document cited as the analysis behind it.
A new row would give the same subject two owners, which is the outcome AGENTS.md
§Records exists to prevent.

**One practical consequence.** "Fold this into #274" is not literally available:
**#274 is deleted** and returns 404 to an anonymous reader, as do #822, #863 and
#1809. The specs survive in the tree; the issues do not. Whatever issue AGENTS.md
§"Every change starts from an issue" requires has to be a **new** one, and it
should name `main-verifiability.md` in its `Row:` context rather than resurrect a
number nobody can open.

## Q1 — What actually gates `main` today

`.github/workflows/` holds six workflows. Only `ci.yml` carries gates; the other
five are publication or drift machinery.

| Workflow | Triggers | Bears on `main`? |
|---|---|---|
| `ci.yml` | `push:[main]`, `pull_request:[opened, synchronize, reopened, closed]`, `schedule: 17 */4 * * *`, `workflow_dispatch` | Yes — the whole subject of this document |
| `containers.yml` | `push` tags `v*` and `main` **restricted to container paths**, `schedule 0 4 * * *`, `pull_request` | Publication. A product change does not trigger the `main` lane |
| `gh-pages.yml` | `push:[main]` and `pull_request`, both **restricted to `docs/`, `website/`, `assets/`, `scripts/check-site.py`, itself** | Site build/deploy only |
| `release.yml` | `workflow_dispatch`, `push` tags `v*` | Never runs on `main` |
| `release-postpublish-audit.yml` | `workflow_run` after `release` | Never runs on `main` |
| `triton-aot-sync.yml` | `push:[main]` and `pull_request`, both **path-restricted** to the Triton kernels and their cmake | Narrow drift check |

### The three lanes of `ci.yml`, by job

Every anchor below is `.github/workflows/ci.yml`.

| Job | Line | Its own `if:` | push→main | pull_request | schedule + dispatch |
|---|---|---|---|---|---|
| `last-gated-commit` | 870 | `github.event.action != 'closed'` (895) | yes | yes | yes |
| `agent-record` | 127 | **none, deliberately** (133-155); skipped on a closed PR only through `needs:` | yes | yes | yes |
| `documentation-checkpoint` | 816 | not schedule/dispatch (831) | yes | yes | **no** |
| `commit-protocol-tag` | 925 | not schedule/dispatch (936); both steps are further gated `event_name == 'pull_request'` | job yes, **steps no** | yes | **no** |
| `pr-size` | 1068 | `event_name == 'pull_request'` (1075) | **no** | yes | **no** |
| `cuda-arch-features` | 1088 | not-closed (1101) | yes | yes | yes |
| `cuda-fat-build` | 1119 | not-closed (1130) | yes | yes | yes |
| `vulkan-spirv-freshness` | 1184 | not-closed (1205) | yes | yes | yes |
| `build-test-vulkan` | 1220 | not-closed (1242) | yes | yes | yes |
| `device-leakage` | 1296 | not-closed (1319) | yes | yes | yes |
| `windows-msvc-cpu` | 1327 | `pull_request \|\| schedule \|\| workflow_dispatch` (1347) | **no** | yes | yes |
| `windows-msvc-vulkan` | 1367 | same (1372) | **no** | yes | yes |
| `build-test-cpu` | 1392 | not-closed (1399) | yes | yes | yes |
| `build-newest-gcc` | 1443 | not-closed (1458) | yes | yes | yes |
| `build-test-cpu-arm64` | 1483 | not-closed (1493) | yes | yes | yes |
| `build-test-cpu-arm64-full` | 1581 | `schedule \|\| workflow_dispatch` (1613); `continue-on-error: true` (1607) | **no** | **no** | yes |
| `sanitize-cpu` (2 legs) | 1683 | not-closed (1711); `continue-on-error: true` (1702) | yes | yes | yes |
| `macos-metal-mlx` | 1734 | `push \|\| schedule \|\| workflow_dispatch` (1808) | yes | **no** | yes |
| `baseline-summary` | 2005 | `schedule \|\| workflow_dispatch` (2031) | **no** | **no** | yes |

**`windows-msvc` is NOT pull-request-only, and the claim needs correcting.** Both
MSVC jobs run on `pull_request`, `schedule` and `workflow_dispatch`
(1347, 1372). They joined the baseline lane on 2026-08-18 in `4fe0f2f39`,
*"fix(ENG-RELEASE-WINDOWS): the main baseline runs the two MSVC gates it grades
(#503)"* — thirteen days before `b6305b5bc` repaired instance 3 while asserting
`main` "never runs it". What is true is narrower and still decisive: **they do
not run on the push lane**, so no landing produces an MSVC verdict, and the
verdict that does exist arrives up to four hours later against a range of
commits.

**`macos-metal-mlx` is the mirror image** (1808): it runs on `push` and the
baseline but **not** on `pull_request`. A macOS break therefore cannot be seen
before it lands, by construction. It is red on `main` today.

### The enforcement layer: there isn't one

```
GET /repos/mudler/vllm.cpp/branches/main
  {"protected": true,
   "protection": {"required_status_checks":
     {"checks": [], "contexts": [], "enforcement_level": "off"}}}
GET /repos/mudler/vllm.cpp/rulesets  ->  []
```

Zero required contexts, enforcement off, no rulesets — all three reproduced
verbatim on 2026-09-05, and the repository is `public`. Every job above is
advisory. The direct evidence is #2412: merged 2026-08-31T15:27:19Z with
`agent-record=CANCELLED`, `build-test-cpu=CANCELLED`,
`windows-msvc-cpu=CANCELLED` and every other heavy job cancelled, and the only
two `SUCCESS` check-runs from `ci.yml` are `documentation-checkpoint` and
`commit-protocol-tag`. (A third `SUCCESS`, `plan`, is not a `ci.yml` job; the
first draft said "the only two in the whole rollup", which was one too narrow a
reading of the rollup.) That pull request landed instance 4.

`ci.yml:1069-1074` states that the `pr-size` job name "is kept because it is a
required check". Measured against the API above, that is **false today**. The
comment is stale; correcting it belongs to whoever next touches that job.

### And roughly half of `main` never sees the pull-request lane

For each of the last 100 first-parent commits reachable from `d023e3357`,
`GET /repos/mudler/vllm.cpp/commits/<sha>/pulls` filtered to `merged_at != null`.
Re-measured twice on 2026-09-05, byte-identical both passes:

| Sample | With a merged pull request | Without |
|---|---|---|
| newest 30 | 25 | 5 |
| next 70 | 23 | 47 |
| **last 100** | **48** | **52** |

**The first draft read 43 / 57, and its sub-table read 9 / 21 on the newest 30.
That sub-table was an artefact of measuring the newest commits within hours of
their landing:** GitHub's commit→pull-request index is not populated
synchronously, so a freshly pushed commit reports no pull request until the index
catches up. A day later the same 30 commits report 25. **Never measure this
population on commits younger than the index.** The headline survives the
correction — 52 of 100, not 57 — because the older 70 are stable.

Two shapes make up the 52, and `git rev-list --parents` separates them: **37 are
merge commits** (two parents) that GitHub associates with no pull request, the
local-merge path, and **15 are single-parent commits** pushed directly. Of the 48
that do carry a merged pull request, 42 are merge commits and 6 are squashes.

The obvious way to measure this is wrong and worth recording: `git log --oneline
-200 --no-merges | wc -l` returns 200 regardless, because `-<n>` limits **after**
the `--no-merges` filter. Fix the revision range first
(`git rev-list --first-parent -n 100 <tip>`), then classify inside it.

AGENTS.md §"Landing work" permits this: *"An operator with recorded merge
authority can merge the branch locally with a commit that names it."* The
consequence is that for 52% of what lands, the pull-request lane is not a gate
that was cancelled — it is a gate that was never invoked.

## Q2 — The "self-healing ratchet", precisely

Two separate mechanisms get called this, and they fail differently. Keeping them
apart is what decides the design.

### Mechanism A — the walk base for the DIFF-scoped gates

`scripts/ci-walk-base.py` resolves one base for `documentation-checkpoint` and
`agent-record`'s role-discipline step (`ci.yml:855-869, 1041-1052`):

```
base = last_green or push_base            # last_green = head of the last SUCCESSFUL push run
if floor is an ancestor of head and not an ancestor of base:
    base = floor                          # scripts/ci-enforcement-floor.txt
```

**What it protects.** A cancelled push run is *lossless*. The base is the last
successfully gated commit rather than `github.event.before`, so a cancelled run's
commits are not skipped — the next run simply walks a wider range and reports the
same red. That property is exactly what made #822's latest-only push lane safe,
and #863 is the record of what happens when it is reverted.

**What it cannot see, measured at `d023e3357`.**

1. *It has no green to heal toward.* The last successful `push` run of `ci.yml`
   on `main` is `375a471e2`, **2026-08-09T15:29:19Z** — 26 days ago. `last_green`
   has been frozen at that commit ever since.
2. *The floor is doing all the work.* `scripts/ci-enforcement-floor.txt` holds
   `e1b5df1a6` (2026-08-25, confirmed). It is newer than `last_green`, so it wins
   every time, and the walk is `e1b5df1a6..origin/main` = **513 first-parent
   commits, 1753 total** (2026-09-05; 511 / 1748 a day earlier — the range grows
   with every push). Measured instead against this document's base SHA
   `d023e3357` the counts are 554 / 1744, because the two tips do not share a
   first-parent chain. Either way it is not a diff; it is a re-audit of a month
   of history on every push.
3. *Advancing the floor forgives, unexamined.* The floor's own header says so:
   *"a floor set further forward forgives whatever lands in between, unexamined,
   and nothing in this mechanism can detect that."* The self-healing is real and
   it heals by amnesty.
4. **It governs only two steps.** Everything the five instances are about is
   TREE-scoped, and a tree-scoped gate has no range to widen. `check-env-doc.py`
   answers a question about the tree at one SHA (`check-env-doc.py:125-129`,
   `scanned - documented - allowlisted`). If its job is cancelled, nothing
   re-covers that SHA — the next push judges a *different* tree, and if that tree
   still violates, the red is charged to whoever is standing there.

### Mechanism B — the push lane is latest-only, and it mostly runs nothing

This is the mechanism the five instances actually rest on, and it is not a walk
base at all.

`ci.yml:122-124` sets, at workflow scope:

```yaml
group: ci-${{ github.event_name }}-${{ … || github.event.pull_request.number || github.ref }}-…
cancel-in-progress: ${{ github.event_name == 'pull_request' || github.event_name == 'push' }}
```

For a push, `github.ref` is the constant `refs/heads/main`, so **every push to
`main` shares one group and cancels its predecessor at run scope**. Eleven more
jobs carry the same shape at job scope: `cancel-in-progress` at 1098, 1127, 1202,
1239, 1316, 1396, 1457, 1490, 1610, 1708 and 1803, beside `agent-record`'s own at
130-132. An earlier draft counted nine and omitted `build-test-cpu-arm64-full`
(1610) and `macos-metal-mlx` (1803); both carry the group, and neither runs on
the push lane for other reasons.

Re-measured 2026-09-05 over the last 40 `event=push` runs of `ci.yml` on `main`.
No push run had been created since, so this is the same window the first draft
read, run for run:

| Outcome | Runs |
|---|---|
| Cancelled **before any job started** (`total_count == 0`) | **33** |
| Ran, `agent-record` cancelled with it | 5 |
| Ran to completion (`33884168834`, `agent-record` **success**) | **1** |
| Still in flight, `agent-record` **failure** (`33925018417`) | 1 |

`agent-record` — the job that carries `check-env-doc.py`, `check-gate-commands.py`
and thirty other preflight checkers — reached a verdict in **2 of 40 pushes
(5%)**, one `success` and one `failure`. The first draft said one; a `failure`
is a verdict, and the in-flight run had produced one. Over the last 100 push
runs: 97 `cancelled`, 2 `failure`, 1 in flight, **0 `success`**.

The rate that produces this: 100 push runs spanning 2026-09-02T19:05:07Z to
2026-09-04T22:19:30Z = 51.24 h = **46.9 pushes/day** (re-measured: the same two
timestamps, 46.84/day), against first-parent commit days on `main` of 45, 60,
121, 39, 56 over 2026-08-29..09-02.

### Mechanism C — the baseline lane exists, completes, and has lost its meaning

The one lane that does finish is #274's 4-hourly baseline, and it is not a
hypothetical: it runs `agent-record` and both MSVC jobs. But over the last 100
`event=schedule` runs (re-measured 2026-09-05): 86 `failure`, 11 `success`,
2 `cancelled`, 1 in flight — and the last success was **2026-08-17T20:42:13Z at
`76f2a6d84`, followed by 85 runs of which not one succeeded** (83 `failure`,
2 `cancelled`). The first draft said 87/11/2 and "84 consecutive failures"; one
more run has since landed and the two `cancelled` interruptions mean "no success
in 85" is the exact form.

Job detail for the three newest baselines (2026-09-04):

| Job | 33908719686 | 33874132447 | 33836956738 |
|---|---|---|---|
| `agent-record` | success (15 m) | success (15 m) | success (15 m) |
| `build-test-cpu` | **failure** (71 m) | **failure** (46 m) | **failure** (71 m) |
| `sanitize-cpu (address,undefined)` | **failure** (99 m) | **failure** (130 m) | **failure** (165 m) |
| `sanitize-cpu (thread)` | **failure** (133 m) | **failure** (133 m) | **failure** (137 m) |
| `windows-msvc-cpu` | **failure** (35 m) | **failure** (29 m) | **failure** (25 m) |
| `windows-msvc-vulkan` | **failure** (26 m) | **failure** (32 m) | **failure** (30 m) |
| `build-test-cpu-arm64-full` | **failure** (62 m) | **failure** (61 m) | **failure** (62 m) |
| `macos-metal-mlx` | **failure** (2 m) | **failure** (1 m) | success |
| `cuda-fat-build` | success (95 m) | success (78 m) | **failure** (19 m) |
| `baseline-summary` | **failure** | **failure** | **failure** |

Every cell of that table was reproduced job-for-job on 2026-09-05, including the
minute counts.

A verdict that has read `failure` for 85 runs without one success cannot signal a
new red. It is a smoke alarm that has been sounding for eighteen days.

**Summing up.** The "ratchet" is three things stacked: a walk base with no green
to advance to, forgiven forward by hand; a push lane that discards 82.5% of its
own runs before a job starts; and a baseline whose aggregate has been red long
enough to carry no information. None of them is a walk-base bug, and none of them
is repaired by changing `ci-walk-base.py`.

## The five instances, re-measured

Each was traced to the commit that introduced it and to the lane that did or did
not see it.

| # | Defect | Introduced by | In a merged PR? | Documented in the same commit? | Repaired by |
|---|---|---|---|---|---|
| 1 | `VT_DFLASH_BOUNDS_DEVICE` undocumented | `21ef6f053` (#2274/#2304, 2026-08-29) | no | no (0 hunks in `docs/`) | `0fcab5dba` (#2307) **and again** in `42f2342be`/`866075b2f` on the unrelated `ENGINE-HYBRID-PLACEMENT` branch (#2313) |
| 2 | `VT_QWEN35_STAGE_RESERVE_BYTES` undocumented | `85f65b0e8` (#2342/#2349, 2026-08-30) | **no — no pull request exists for this commit** | no (0 hunks in `docs/ENVIRONMENT.md`) | four independent commits: `a81d6e52a`, `78edc68a9`, `95f282d2f` (#2357), `72278b146`/`c31b2496e` (#2359) |
| 2b | `VT_QWEN35_STAGE_MIN_FREE_FRAC` undocumented | (same class, one row over) | — | — | `ebc154341`/`f72e62a2f` — *"so the base gate stops failing every branch"* (#2329, #2332) |
| 3 | `KdaHeadCount` shadows `v`; MSVC C4456 → `/WX` → C2220 | `a36add6a8`/`c3522bc7d` (#2278, 2026-08-29) in `glm5_next_weights.cpp` | — | — | `b6305b5bc` (#2404) from an unrelated branch, two days later |
| 3b | the same class, four days earlier | seven `n` loop variables | — | — | `157636cf1` (#2101, #2103) — *"broke every Windows build"* |
| 4 | `RUNNABLE_BASELINE` not re-pinned when `ENG-PREFLIGHT-COMPILES` grew a `## Gates` section ([#2429](https://github.com/mudler/vllm.cpp/issues/2429): *"fails 12 of 59 on a clean `origin/main`"*) | `ff50af7c4` via **PR #2412, merged with `agent-record=CANCELLED`** | yes | — | `274dc7a06`, two commits later on `main` |
| 5 | `api_server` `STATUS_STACK_BUFFER_OVERRUN` (`-1073740791`) | pre-existing, #584 | — | — | open; it was invisible while instance 3 stopped the MSVC lane from linking |

Both surviving issues confirm their rows verbatim.
[#2404](https://github.com/mudler/vllm.cpp/issues/2404) quotes
`glm5_next_weights.cpp(223,31): warning C4456: declaration of 'v' hides previous
local declaration` and `error C2220`, and names `KdaHeadCount`'s `else if` rungs
as the cause. **It attributes the defect to `b326ea003` (#2242 / #2292); this
table attributes it to `a36add6a8` (#2278), and the tree supports the table** —
`b326ea003` created the file, and `git log -S'KdaHeadCount' -- '*glm5_next_weights.cpp'`
returns `a36add6a8` as the commit that introduced the function the shadowing
lives in. The distinction does not change the argument: both landed on
2026-08-29 and neither was seen by a Windows lane before it landed.

Two properties recur and both are load-bearing:

- **The repair lands from a branch that did not cause the defect.** Instances 1
  and 2 were each repaired two to four times, in parallel, by different sessions
  that each hit the same red. Every repair commit named in the table above was
  confirmed to exist with the subject quoted. The aggregate is measurable:
  `git log --since=2026-08-01 -i -E --grep=` over the four phrases *"red on
  main"*, *"reds every branch"*, *"stops failing every branch"*, *"landed on main
  undocumented"* returns **26 commits out of 3410** on 2026-09-05. The first
  draft said 34 of 3369; the denominator moved with a day of pushes, and the
  numerator does not reproduce at 34 by the method the draft states, so **26 is
  the number this document stands behind** and a wider phrase set would find
  more.
- **Instance 2 met no gate at any point.** `85f65b0e8` belongs to no pull
  request, so there was no PR lane; its push run produced no verdict; and the
  baseline's `agent-record` would have caught it up to four hours later, inside a
  run whose aggregate had already been red for a fortnight.

## Q3 — Which preflight gates are cheap enough to run on `main`

`scripts/agent-preflight.sh` runs, in order: a role check; 32 named `CHECKERS`;
`ready-for-helper`, `upstream-inventory`, `audit-live-rows`; 69 named `SUITES`;
five numpy-conditional suites; the Windows-portability suite behind
`cmake`+`ninja`; the trailer and commit-style suites; `tests/tools/` by
discovery; **every `scripts/check-*.py` by discovery**; two range blocks; and
`scripts/check-tree-compiles.py`.

Two independent measurements, because neither alone is honest. The CI column is
from the `agent-record` job of baseline run `33908719686` (`ubuntu-latest`,
job `101145267044`) — the runner class that would pay. The local column is this
worktree at `d023e3357` on a contended 20-core workstation, and it is here for
the one thing CI cannot show: which gates move with host load.

### The whole checker sweep costs about half a minute

`scripts/check-*.py` is **44 files**. 43 of them run bare, serially, on
`origin/main`: **30.0 s total, 36 `ok`, 7 needs-arguments, 0 failures.**
`check-tree-compiles.py` was not run — it compiles, and this analysis was
conducted under an explicit no-build constraint.

**Re-run 2026-09-05 under the same constraint, with one more exclusion: 42 of
the 44 gave 35 `ok`, 7 needs-arguments, 0 failures, in 26 s wall.**
`check-windows-portability.py` was also skipped, because it shells out to
`cmake -S … -B … -G Ninja` to drive the file-api and derive the shipped-server
source set (`check-windows-portability.py:207-222`) — a configure, not a
compile, but still a `cmake` invocation. Add its 12.8 s back and both the count
(43 run, 36 `ok`) and the total (≈ 39 s here) reconcile with the line above.
**The load-bearing part of this measurement — 0 failures — reproduces.** The
seven that need arguments are `check-arm-isa-build`, `check-commit-style`,
`check-commit-trailers`, `check-cpu-isa-build`, `check-cuda-fat-gencode`,
`check-pr-size` and `check-triton-aot-multiarch`.

**Every second in the table below moves with host load, and should be read as a
ranking rather than a cost.** Re-timed 2026-09-05 at one-minute load average
~120 (an unrelated build on the same box), the same checkers ran 1.4× to 2.2×
these values in an unchanged order: `check-env-doc` 3.1 s, `check-agent-record`
3.4 s, `check-snapshot-pins` 4.1 s, `check-test-registration` 5.9 s,
`check-device-leakage` 6.5 s, `check-symbol-anchors` 1.1 s,
`check-cuda-op-arch-gate` 0.8 s, `check-conflict-markers` 0.5 s,
`check-gate-commands` 0.3 s, `check-role-discipline` 0.1 s. The CI column, not
this one, is the denominator for `## Q4`.

| Checker | local s | | Checker | local s |
|---|---|---|---|---|
| `check-windows-portability` | 12.8 | | `check-symbol-anchors` | 0.8 |
| `check-device-leakage` | 3.3 | | `check-conflict-markers` | 0.6 |
| `check-test-registration` | 3.0 | | `check-release-binary-contract` | 0.2 |
| `check-agent-record` | 2.4 | | `check-gate-commands` | 0.2 |
| **`check-env-doc`** | **2.2** | | `check-cuda-op-arch-gate` | 0.2 |
| `check-snapshot-pins` | 1.9 | | `check-role-discipline` | 0.1 |
| `check-runner-routing-consistency` | 1.2 | | 22 others | ≤ 0.1 each |
| `check-attention-rung-consistency` | 1.2 | | | |

The three flagged `--check` entry points: `claim-view` 0.1 s,
`check-gate-commands` 0.3 s, `ready-for-helper` 0.2 s, `upstream-inventory`
0.1 s, **`audit-live-rows` 37.2 s** (85 s on CI). The 85 s is confirmed from job
`101145267044`; the local 37.2 s was not re-measured.

### The 69 `SUITES` cost 765 s locally, and one of them is not deterministic

`scripts/agent-preflight.sh` names **32 `CHECKERS` (line 97) and 69 `SUITES`
(line 140)**; both counts confirmed 2026-09-05. 68 suites ran
(`test_check_tree_compiles` skipped, it builds), **765 s total, zero failures**.

**NOT RE-MEASURED.** Re-running the suites needs roughly a quarter-hour of the
box, and the verification pass ran under a no-build constraint on a host already
at load ~120, where the number would be neither reproducible nor comparable. The
per-suite CI column below *is* confirmed, step by step, against job
`101145267044`. The tail:

| Suite | local s | CI s | Deterministic on a clean checkout? |
|---|---|---|---|
| `test_cpu_x86_llamacpp_floor` | **191.8** | **3** | **NO — reads host CPU load** |
| `test_check_release_binary_contract` | 176.6 | 84 | yes |
| `test_agent_record` | 45.6 | 39 (bundled) | yes |
| `test_check_env_doc` | 36.6 | 44 (bundled) | yes |
| `test_audit_live_rows` | 27.1 | 85 (bundled) | yes, but reads `origin/main` |
| `test_check_attention_rung_consistency` | 25.1 | 34 | yes |
| `test_check_runner_routing_consistency` | 16.0 | 26 | yes |
| `test_agent_role` | 12.8 | 10 | yes |
| `test_check_test_registration` | 10.7 | 17 | yes |
| `test_check_gate_commands` | 6.3 | 11 | yes |
| 59 others | ≤ 5.1 each | | yes |

**The 64× spread on `test_cpu_x86_llamacpp_floor` is the whole point of running
it twice.** It executes the real `scripts/cpu-x86-llamacpp-floor.sh`, whose
`wait_quiet()` (`cpu-x86-llamacpp-floor.sh:116-134`) measures CPU busy percentage
over a window and refuses to proceed unless it is `<= QUIET_BUSY` (default 10) —
returning `NO_QUIET_WINDOW` and exit 4, which
`test_the_quiet_gate_does_not_see_the_harnesss_own_process_tree`
(`test_cpu_x86_llamacpp_floor.py:182-203`) asserts is 0. On an idle GitHub runner
it costs 3 s. On a box where somebody else is compiling, it costs 192 s and can
fail. **A gate whose verdict depends on what else is running must never name
`main`.**

### The classification rule

Rather than hand-pick a list — a list is a shared surface every new gate has to
edit, which AGENTS.md §Records forbids — a step qualifies for the `main` lane
when **all five** hold:

1. it reads only tracked files and git history;
2. it needs no network beyond the checkout;
3. it binds no socket and reads no host load, CPU counter or wall clock;
4. it installs no package and downloads no tool;
5. it costs under 45 s on `ubuntu-latest`.

That rule is mechanical, it can be stated in a checker later, and it excludes
every gate that has cost this repository a false red.

## Q4 — What it would cost

Step timings below are from the same `agent-record` job (`33908719686`), which is
the correct denominator: the runner class that pays.

**Qualifying steps (the proposed job), CI seconds:** checkout `fetch-depth: 0`
16; roadmap tables + `test_agent_record` + `test_agent_issue_index` 39;
**env-doc 44**; attention rung 34; runner routing 26; test registration 17;
snapshot pins 11; **gate-commands 11**; role machinery 10; symbol anchors 7;
protocol prose 7; preflight skip report 5; `checker_text` 3; conflict markers 2;
CUDA op arch gate 2; fp4 residency 1; surface coverage 1; oracle denominator
flags 1; walk-base suite 1; main-baseline suite 1; claim view 1; README,
quickstart, benchmark index, model checklist, supported models, fusion catalog,
oracle pins, GEMV invocation, GPU mutex, NOW.md, pre-push hook names 0 each.

**Total ≈ 240 s ≈ 4.0 minutes per push.** Every step second above was read back
off job `101145267044` on 2026-09-05 and the listed values sum to exactly 240;
the job's 61 steps total **943 s**, also exact. This is the most thoroughly
confirmed number in the document.

| Shape | per run | × 46.9 pushes/day |
|---|---|---|
| Proposed `record-gates-main` | 4.0 min | **≈ 3.1 job-hours/day** |
| Making `agent-record` itself non-cancellable on push | 15.7 min (measured: 943 s of steps, 15 min wall) | ≈ 12.3 job-hours/day |
| A full non-cancellable suite per merge — **what #274 measured and rejected** | ~99 min wall, ~3.5 h job time | **≈ 190 job-hours/day** |

The proposal is **1.6% of the option #274 rejected on cost**, and a quarter of the
one-line alternative.

The repository is public (`"visibility": "public"`), so runner minutes are
unbilled. **The cost is pool contention, and the pool — not the job — is the
binding constraint.** Queue waits measured for `agent-record` on 2026-09-04:

All four rows below were reproduced exactly on 2026-09-05. `created` is the
**job's** `created_at`, not the run's — for a job behind `needs:` the two differ
by hours, and the run-level figure would read 640 / 378 / 280 / 37 minutes
instead.

| Run | created | started | **queued** | ran |
|---|---|---|---|---|
| 33836956738 | 08:23:36Z | 15:09:07Z | **405 min** | 15 min |
| 33874132447 | 17:49:18Z | 18:59:58Z | 70 min | 15 min |
| 33884168834 | 18:05:42Z | 19:11:15Z | 65 min | 16 min |
| 33908719686 | 19:17:46Z | 19:35:07Z | 17 min | 15 min |

A four-minute job that waits between seventeen minutes and six and a half hours
is a delayed notification, not a landing gate. That is the single most important
number in this document and it is argued both ways below.

## Design options

### Option 1 — Do nothing here; repair the baseline instead

Fix `build-test-cpu`, both `sanitize-cpu` legs, both `windows-msvc-*`,
`build-test-cpu-arm64-full` and `macos-metal-mlx` so `baseline-summary` can be
green again, and treat the 4-hourly baseline as `main`'s verdict.

*For.* It adds no lane, no cost and no new red. It restores discriminating power
to the one lane that completes and already runs every gate discussed here. It is
the prerequisite for any of the others to mean anything.

*Against.* It attributes badly by construction — at 47 pushes/day a red baseline
names ~9 commits — and #274 accepted that trade knowingly. It is also a large
amount of work spread across #584, #301 and several unowned reds, none of which
this row can claim.

### Option 2 — A new per-push, non-cancellable `record-gates-main` job (RECOMMENDED)

```yaml
record-gates-main:
  if: github.event_name == 'push'
  concurrency:
    group: ci-record-gates-main-${{ github.sha }}-${{ github.repository }}
    cancel-in-progress: false
  runs-on: ubuntu-latest
```

Keyed on `github.sha`, so two pushes never share a group and never cancel each
other — the property #822 deliberately removed for the expensive jobs, restored
only for a four-minute one. It runs the qualifying steps and nothing else. It is
**not** added to `baseline-summary`'s `needs:`; the baseline already runs the same
steps inside `agent-record`, and a push-lane job in a baseline aggregate would be
a category error.

*For.* One push, one verdict, one SHA — attribution the baseline cannot give.
3.1 job-hours/day. It starts **green**: every one of its steps passes on
`origin/main` at `d023e3357` right now (43/43 checkers, 68/68 suites, zero
failures), which is the opposite of how #274's baseline started and is what keeps
it from becoming another permanent red.

*Against.* The queue. And two copies of one step list will drift.

*Mitigation for the drift, and it is not optional.* Extract the qualifying steps
into one script — `scripts/ci-record-gates.sh` — called by **both**
`record-gates-main` and `agent-record`, with a suite asserting `agent-record`
calls it, in the neighbourhood of `tests/scripts/test_main_baseline.py`, which
already holds invariants over this workflow's concurrency blocks. Duplicating the
step list into a second job and trusting two lists to stay equal is the shape
AGENTS.md §Records names as a lock.

### Option 3 — Make `agent-record` non-cancellable on push

One word in one expression (`ci.yml:132`).

*For.* Minimal diff. No new job, no drift, no script.

*Against.* 12.3 job-hours/day, four times the proposal, on the pool that is
already the constraint. Worse, it puts `test_cpu_x86_llamacpp_floor` (host load),
`test_tower_skip_rss_arm` (binds a TCP socket with `SO_REUSEPORT`) and the
`sudo apt-get install ninja-build` inside the Windows-portability step onto a
47-runs-a-day lane whose reds name `main`. This repository has already paid for
each of those flake classes once. **Rejected.**

### Option 4 — Required status checks

Set `required_status_checks.enforcement_level` and require `agent-record` on the
pull-request lane.

*For.* It is the **only** option that prevents a landing rather than reporting
one afterwards. Everything else in this document is a notification.

*Against, and it is decisive today.* (a) It is a repository setting, not a file
in this tree; no checker here can hold it, for exactly the reason AGENTS.md gives
about `squash_merge_commit_message` — reading it needs a network call. (b) It
cannot cover the 52% of first-parent commits that land with no pull request. (c)
With `agent-record` routinely cancelled on the PR lane (#2412) and queue waits up
to 405 minutes, requiring it would block every merge for hours starting the day
it is turned on. **Named as the follow-on, after Option 2 shows the job completes
reliably — not as this row's change.**

### Option 5 — The pre-push hook

`.githooks/pre-push` runs three checkers today
(`check-prompt-contract.py`, `check-now-current.py`,
`check-readme-structure.py`). Adding the qualifying checker sweep costs ~30 s
locally and fires **before** the push, which is the only place the 52%
no-pull-request population can be caught at all.

*Against.* AGENTS.md: *"Hooks are bypassable convenience, not evidence."* A hook
is a companion to Option 2, never a substitute, and this document does not
propose it as the gate.

### Rejected without a section

**A separate `main-gates.yml` workflow.** `github.ref` is unchanged in a called
workflow, so it needs the same event discriminator and gains nothing;
duplicating the job definitions creates two suites that drift.
[`main-verifiability.md`](main-verifiability.md) already rejected this shape for
the same reason.

**Moving any build onto the push lane.** `cuda-fat-build` is 78-95 min,
`sanitize-cpu` 99-165 min, `windows-msvc-*` 25-35 min. Nothing here proposes it.

## Q5 — The case against this change

Stated at full strength, because the request asked for a decision and not
advocacy. Points 1 and 4 are the ones that could reverse the recommendation.

**1. `main` has no green lane, and adding a lane does not create one.** 85
baseline runs without a success since 2026-08-17; zero successful push runs since
2026-08-09 (`375a471e2`, 2026-08-09T15:29:19Z — both confirmed). The repository has been unable to answer *"is `main` green?"* for
eighteen days, and the answer is *"no, on seven heavy jobs"*. A thirty-fourth
signal saying *the records are fine* while `build-test-cpu`, both sanitizers,
both MSVC lanes, `build-test-cpu-arm64-full` and `macos-metal-mlx` are red is
precision on the wrong axis. Option 1 first, and the question this row asks may
dissolve.

**2. The gates already run on `main`. Nothing is missing; runs are being
discarded.** `agent-record` carries no `if:` deliberately (`ci.yml:133-155`,
because #873), and it already runs `check-env-doc.py` and
`check-gate-commands.py` on every push. The defect is #822's latest-only push
lane — a measured, argued decision with 40 runs of evidence behind it. The
cheapest reading of the same evidence says the fix is one word in
`ci.yml:132` (Option 3), and this document rejects that on cost. That tension is
real and a reviewer is entitled to weigh it the other way.

**3. AGENTS.md's own rule cuts against this.** *"A gate that fires on ordinary
work is the defect, not the discipline"* — the argument that retired the
per-class line budgets, where 9 of the previous 22 merged pull requests exceeded
the product limit. A per-push record gate on a pool with a 405-minute queue will
fire, late, against a SHA that is by then ten commits behind, and its repair will
land as one more `docs(ENV-DOC): document X so the base gate stops failing every
branch`. The gate would not have prevented instances 1 and 2. It would have
**renamed** them — from *a red charged to a stranger's branch* to *a red charged
to a stale SHA*, which is better, but is a smaller improvement than it sounds.

**4. The mechanism built for exactly this has produced nothing.**
`workflow_dispatch` on `ci.yml` is #274's "hybrid half": *"pin a baseline on a
SHA you care about right after merging it."* Measured 2026-09-05:
`GET /actions/workflows/ci.yml/runs?event=workflow_dispatch` returns
`total_count: 1` — run `33618253298`, 2026-09-02T10:12:29Z, on the row branch
`row/DEBTFIX-GLUE-RANK-COLLISION`, **cancelled**. So it has been fired once, not
on `main`, and it pinned nothing. **Zero baselines, ever.** An earlier draft of
this document said the API returns `[]`; that was false and is corrected here,
and the counter-argument survives it intact. Before building a new lane, use the
idle one that was designed for this question — an operator who has just merged
something they care about can have a full verdict on that exact SHA for the cost
of one command. If a week of actually using it closes the gap, Option 2 is
unnecessary.

**5. The real hole is landing discipline, not CI.** 52 of the last 100
first-parent commits on `main` have no pull request. `main` has
`enforcement_level: "off"` and zero required contexts. A change that lands by
direct push has bypassed every lane by construction, and no amount of post-push
CI turns a notification into a gate. The honest fix is Option 4 plus landing
through pull requests — policy, not YAML — and Option 2 risks being the thing
that lets everyone stop worrying about the policy.

**6. It would have caught three of the five instances, not five.** See below. The
two it misses are the two that were most expensive.

## Recommendation

**Adopt Option 2, sequenced behind Option 4's cheap half and Option 1's
diagnosis, with Option 5 as a companion.** Concretely, in this order:

1. **Use `workflow_dispatch` for two weeks** (counter-argument 4). It costs
   nothing, it exists, and it has never been tried. If a dispatched baseline
   after each significant merge closes the gap, stop here.
2. **Land Option 2** — `record-gates-main`, `if: github.event_name == 'push'`,
   group keyed on `github.sha`, `cancel-in-progress: false`, running the
   qualifying steps through one shared `scripts/ci-record-gates.sh` that
   `agent-record` also calls. ≈ 4 min/run, ≈ 3.1 job-hours/day, green on
   `d023e3357` today.
3. **Add the qualifying checker sweep to `.githooks/pre-push`** (Option 5), as
   convenience for the 52% that land without a pull request. Never cited as
   evidence.
4. **Revisit Option 4** once `record-gates-main` has a week of completion data.
   If its own queue wait exceeds ~30 minutes at p90, the row converts into Option
   4 and this job is deleted rather than kept as decoration.

Option 1 is not this row's work, but it is named as the thing that decides
whether `main` has a verdict at all, and no result here should be read as a
substitute for it.

### What this would and would not have caught

| Instance | Caught? | Why |
|---|---|---|
| 1 — `VT_DFLASH_BOUNDS_DEVICE` | **yes** | `check-env-doc.py` + suite, 44 s, qualifies |
| 2 — `VT_QWEN35_STAGE_RESERVE_BYTES` | **yes** | same; and it is the case with no pull request, so this is the only lane that could |
| 3 — MSVC C4456 in `KdaHeadCount` | **no** | needs a 25-35 min `windows-2022` compile. `scripts/check-windows-portability.py` has no shadowing rule — `grep -niE 'C4456\|shadow\|hides'` over its 93,973 bytes returns nothing, re-confirmed 2026-09-05 — so no cheap static proxy exists today |
| 4 — `RUNNABLE_BASELINE` drift | **yes** | `test_check_gate_commands`, 11 s, qualifies |
| 5 — `api_server` `STATUS_STACK_BUFFER_OVERRUN` | **no** | a Windows *runtime* crash (#584), reachable only once the MSVC lane links |

**Three of five, at 1.6% of the cost #274 rejected.** The other two are the same
Windows lane and need a different answer, named under `## Owed`.

## What moves, and what deliberately does not

**Moves** — every step meeting the five-part rule. Named by the two that the
instances turn on, and by rule for the rest: `check-env-doc.py` +
`test_check_env_doc.py`; `check-gate-commands.py --check` +
`test_check_gate_commands.py`; `check-agent-record.py --report` and its suites;
`check-symbol-anchors`, `check-snapshot-pins`, `check-test-registration`,
`check-surface-coverage`, `check-runner-routing-consistency`,
`check-attention-rung-consistency`, `check-fusion-consistency`,
`check-fp4-resident-consistency`, `check-cuda-op-arch-gate`,
`check-oracle-pins`, `check-oracle-denominator-flags`, `check-conflict-markers`,
`check-now-current`, `check-role-discipline`, `check-readme-structure`,
`check-quickstart-recipes`, `check-benchmark-index`, `check-model-checklist`,
`check-supported-models`, `check-prompt-contract`, `claim-view --check`, and
their suites.

**Does not move, and why each.**

| Excluded | Rule broken | Measured reason |
|---|---|---|
| `test_cpu_x86_llamacpp_floor` | 3 | `wait_quiet()` gates on CPU busy % ≤ 10 (`cpu-x86-llamacpp-floor.sh:116-134`); 3 s idle, 191.8 s under load, and it can exit 4 where the suite asserts 0. **A flaky gate on `main` is worse than none** |
| `test_tower_skip_rss_arm` | 3 | binds a TCP socket on a scratch port; #1844's own repair is about a readiness poll answering from the previous leg |
| Windows source contract step | 4 | runs `sudo apt-get update && apt-get install -y ninja-build` inside the step — a package-mirror dependency, and 69 s |
| `test_qwen4exp_llamacpp_ladder` | 5 | 197 s |
| The release-contract block (`check-release-binary-contract` + 16 suites) | 5 | 84 s on CI, 176.6 s locally. Release-contract drift is not a landing hazard on the same clock |
| `audit-live-rows.py --check` | 5 | 85 s on CI, 37.2 s locally; and it resolves `origin/main`, whose meaning is different on the push lane, where `origin/main` **is** the pushed commit |
| The docs-site step (`check-site.py`, `test_check_site`, `test_ci_site_lane`) | 4 | needs `peaceiris/actions-hugo@v3` to download a Hugo binary |
| The LTX-2.5 numpy suites | 4 | need `python3-numpy` installed |
| `tests/tools/` discovery, IndexTTS-2.5 block | 5 | 39 s and 8 s; low landing-hazard for the price |
| `check-tree-compiles.py` | 5 | it compiles. #2401 landed it as a **pre-push** gate deliberately, and its own spec records why a full build (~12 min, 9.4 GiB) "is a gate that fires on ordinary work". Nothing here moves it |
| Every build job, both sanitizers, both MSVC jobs, `macos-metal-mlx` | 5 | 25-165 minutes each |
| `commit-protocol-tag`'s two steps | — | already pull-request-only as of #2322, deliberately: they walked landed history on `main` where the only repair is a rewrite AGENTS.md forbids. **Restoring them to the push lane would recreate #2157's forgiveness work.** Explicitly not proposed |

## Risks

1. **The pool, and it is the largest.** A 405-minute queue wait was measured on
   2026-09-04. A gate that reports seven hours late is a notification. Recorded,
   not solved; it is the trigger in `## Stop conditions` for converting this row
   into Option 4.
2. **A new permanent red on `main`.** If `record-gates-main` goes red and nobody
   repairs it, the repository has bought a 47-a-day red. Mitigated by starting
   green — measured at `d023e3357`: 43/43 checkers, 68/68 suites, 0 failures —
   which is precisely the property #274's baseline deliberately did not have.
3. **Two copies of one step list drift.** Only the shared-script mitigation makes
   this safe. A reviewer who sees the step list duplicated into a second job
   should reject the change.
4. **Nothing new is demanded of a contributor.** The qualifying set is exactly
   what preflight already runs on a branch. That is the answer to
   counter-argument 3 for the *branch* case; it is not an answer for the *stale
   SHA* case, which stands.
5. **Selecting the steps by hand instead of by rule** re-creates the shared-file
   lock. The five-part rule is the mitigation, and the implementing change should
   state it in the job's own comment, since a checker's message is what defines
   what it enforces.

## Gates

Everything below is re-measurable read-only and needs no build and no GPU.

- `python3 scripts/check-env-doc.py && python3 tests/scripts/test_check_env_doc.py`
- `python3 scripts/check-gate-commands.py --check && python3 tests/scripts/test_check_gate_commands.py`
- `python3 scripts/check-agent-record.py --report`
- `python3 scripts/check-symbol-anchors.py`
- `for f in scripts/check-*.py; do python3 "$f"; done` — the 30-second sweep,
  excluding `check-tree-compiles.py`
- `gh api repos/mudler/vllm.cpp/branches/main --jq .protection.required_status_checks`
  — must report `enforcement_level: "off"` with empty `checks`/`contexts` for the
  Q1 finding to hold; anything else means the setting moved and this document is
  stale
- `gh api "repos/mudler/vllm.cpp/actions/workflows/ci.yml/runs?branch=main&event=push&per_page=40"`
  then per run `.../jobs` — reproduces the 33-zero-job / 1-verdict census
- `gh api "repos/mudler/vllm.cpp/actions/workflows/ci.yml/runs?event=workflow_dispatch"`
  — must report no run that both targets `main` and reached a conclusion, for
  counter-argument 4 to hold. It is **not** `[]`: one cancelled run on a row
  branch exists. Note the query has no `branch=main` filter; adding one is what
  made an earlier reading of this look empty
- `git rev-list --first-parent -n 100 <tip>`, then
  `gh api repos/mudler/vllm.cpp/commits/<sha>/pulls` per sha — reproduces the
  48/52 split. **Do not run it against commits younger than a day**: GitHub's
  commit→pull-request index lags a fresh push and reports no pull request for a
  commit that has one
- `curl -s -o /dev/null -w '%{http_code}' https://api.github.com/repos/mudler/vllm.cpp/issues/<n>`
  — unauthenticated, so a 404 separates a deleted issue from one the caller
  cannot see. Every issue this document cites should be run through it before
  the citation is relied on

## Evidence

First measured 2026-09-04 against `origin/main` at `d023e3357`, on a linked
worktree, with no build run at any point. **Re-measured 2026-09-05 under the same
constraint.** The `verified` column carries the second reading: `=` means it
reproduced exactly, `~` means it drifted with the clock in the expected
direction, `CORRECTED` means the first reading was wrong and the value shown is
the second one, and `NOT RE-MEASURED` means the constraint forbade it.

| Claim | Value | Source | verified |
|---|---|---|---|
| Required status checks on `main` | `checks: []`, `contexts: []`, `enforcement_level: "off"`; rulesets `[]`; repo `public` | GitHub REST | **=** |
| Last 40 push runs on `main` | 33 zero-job, 5 all-cancelled, 1 complete, 1 in flight | `/actions/runs/<id>/jobs` per run | **CORRECTED** (was 33 / 6 / 1) |
| `agent-record` verdicts on push | **2 of 40** (1 success, 1 failure) | same | **CORRECTED** (was 1 of 40) |
| Last 100 push runs | 97 cancelled, 2 failure, 1 in flight, 0 success | `/actions/workflows/ci.yml/runs` | **~** (was 96 / 2 / 2 / 0) |
| Last successful push run | `375a471e2`, 2026-08-09T15:29:19Z | `…&status=success&per_page=1` | **=** |
| Enforcement floor | `e1b5df1a6` (2026-08-25); `floor..origin/main` = 513 first-parent / 1753 total; `floor..d023e3357` = 554 / 1744 | `scripts/ci-enforcement-floor.txt`, `git rev-list --count` | **~** (was 511 / 1748) |
| Baseline lane, last 100 | 86 failure / 11 success / 2 cancelled / 1 in flight; **85 runs with no success** since 2026-08-17T20:42:13Z at `76f2a6d84` | `…&event=schedule` | **~** (was 87 / 11 / 2, "84 consecutive") |
| `workflow_dispatch` runs, ever | **1**, on a row branch, cancelled (`33618253298`) | `…&event=workflow_dispatch` | **CORRECTED** (was `[]`) |
| Push rate | 100 runs in 51.24 h = **46.8/day** | run `created_at` span | **=** |
| First-parent commits with a merged PR | **48 of 100**; of the 52 without, 37 are merge commits and 15 single-parent | `/commits/<sha>/pulls`, `git rev-list --parents` | **CORRECTED** (was 43 of 100; the index lags a fresh push) |
| `agent-record` cost | 943 s of 61 steps, 15 min 46 s wall | job `101145267044` | **=** |
| `agent-record` queue waits | 17 / 65 / 70 / **405** min, from the JOB's `created_at` | four runs, 2026-09-04 | **=** |
| Qualifying-step total on CI | **240 s**, summed from the same job's steps | job `101145267044` | **=** |
| Three newest baselines, job for job | every conclusion and minute count as tabulated | `/actions/runs/<id>/jobs` | **=** |
| `check-windows-portability.py` shadowing rule | none; `grep -niE 'C4456\|shadow\|hides'` returns nothing | the file, 93,973 bytes | **=** |
| `.githooks/pre-push` checkers | exactly three: `check-prompt-contract.py`, `check-now-current.py`, `check-readme-structure.py` | `.githooks/pre-push:30` | **=** |
| Preflight array sizes | 32 `CHECKERS`, 69 `SUITES` | `scripts/agent-preflight.sh:97,140` | **=** |
| Every per-job `if:` and `concurrency:` in the Q1 table | every line number and expression as tabulated | `.github/workflows/ci.yml` | **=** |
| Every commit SHA in the five-instance table | exists, with the subject quoted; `21ef6f053` and `85f65b0e8` both return `[]` from `/pulls` and touch 0 doc hunks | `git log -1`, `/commits/<sha>/pulls` | **=** |
| #2412 | merged 2026-08-31T15:27:19Z, `agent-record` CANCELLED | `/pulls/2412`, `/commits/<sha>/check-runs` | **=** |
| All `scripts/check-*.py` bare | 44 files; 42 run under the tighter constraint = **26 s**, 35 ok, 7 needs-args, **0 fail** | this worktree | **~** (was 43 run / 30.0 s / 36 ok; `check-windows-portability` also skipped, it runs `cmake`) |
| All 69 preflight `SUITES` | **765 s**, 68 ok, 1 skipped (builds), 0 fail | this worktree | **NOT RE-MEASURED** |
| `test_cpu_x86_llamacpp_floor` | 3 s on CI, **191.8 s** here | both | CI **=**; local **NOT RE-MEASURED** |
| `audit-live-rows.py --check` | 85 s on CI, 37.2 s here | both | CI **=**; local **NOT RE-MEASURED** |
| `check-windows-portability.py` | 12.8 s here | this worktree | **NOT RE-MEASURED** (it runs `cmake`) |
| Repair-language commits on `main` since 2026-08-01 | 26 of 3410 | `git log --since -i -E --grep` | **CORRECTED** (was 34 of 3369; does not reproduce by the stated method) |
| Cited issue numbers that still resolve | 5 of 31; the rest 404 to an anonymous reader | unauthenticated `/issues/<n>` | **new finding** |

Not measured, and stated as such: the proposed job's own queue wait, which cannot
be known until it exists, and which the stop condition below is written around.

## Owed

- **An issue.** AGENTS.md §"Every change starts from an issue" is not satisfied.
  #2389 is closed and is about `check-env-doc`'s reverse direction, not the lane
  question. One must be opened, carrying `Row: -` with this spec listed here,
  before any workflow edit. It must be a **new** issue: #274 is deleted, so
  folding into it is not available, and per `## Relation to
  main-verifiability.md` the implementing change belongs to
  `GATE-CI-CONCURRENCY` rather than to a new row.
- **The deleted-issue problem is bigger than this document.** 26 of the 31 issue
  numbers cited here 404 to an anonymous reader, and `ci.yml` cites two of them
  (#274, #822) in comments a reader is invited to follow. Every spec in this
  family has the same defect. Named, not claimed; the fix is not a workflow
  change and does not belong to this analysis.
- **Instances 3 and 5 are not addressed.** The MSVC C4456 class has no cheap
  static proxy: `scripts/check-windows-portability.py` carries no shadowing rule
  (re-confirmed 2026-09-05).
  The candidate is `-Wshadow` on the Linux build lanes, which is a compiler-flag
  change belonging to a build row, not here. #584 is open and owned.
- **The seven heavy reds on `main`** — `build-test-cpu`, both `sanitize-cpu`
  legs, both `windows-msvc-*`, `build-test-cpu-arm64-full`, `macos-metal-mlx` —
  are what make the baseline uninformative. Named, not claimed.
- **`ci.yml:1069-1074` is stale.** It says `pr-size` "is kept because it is a
  required check"; there are no required checks. Correcting it belongs to whoever
  next touches that job, in that change.
- **The five-part qualification rule is prose here and enforced by nothing.** If
  it survives review, a later row should make it executable — a checker that
  refuses a `record-gates-main` step which installs a package or reads
  `/proc/loadavg` — rather than leaving the rule to be re-derived.

## Stop conditions

- Return `NEEDS_DECISION` rather than implementing Option 2 before
  `workflow_dispatch` has actually been tried. Counter-argument 4 is not
  rhetorical: the mechanism has never run once, and if it closes the gap this
  row should be closed rather than built.
- Stop rather than moving any step onto the push lane that reads host load or a
  clock, binds a socket, installs a package, or downloads a tool — whatever its
  measured cost.
- Stop rather than adding `record-gates-main` to `baseline-summary`'s `needs:`,
  or giving it a group keyed on `github.ref`. Either one silently restores the
  cancellation this row exists to avoid.
- Stop rather than duplicating the qualifying step list into a second job without
  the shared script; two lists that must stay equal is the lock shape AGENTS.md
  forbids.
- If `record-gates-main`'s own queue wait exceeds ~30 minutes at p90 over its
  first week, stop treating it as a gate: convert the row into Option 4 and
  delete the job rather than keeping a decorative one.

## Now

Analysis only. Nothing is implemented and no workflow is edited. The issue is
[#2950](https://github.com/mudler/vllm.cpp/issues/2950), which owns the gap under
`GATE-CI-CONCURRENCY`. **No row is claimed, and none should be**: see `## Relation to
main-verifiability.md`. The next action is the decision between Option 1 first,
`workflow_dispatch` first, and Option 2 — and whichever wins needs a new issue,
and lands under `GATE-CI-CONCURRENCY`, before it starts.

Verified against the tree and the GitHub API on 2026-09-05 under a no-build,
no-GPU constraint. Four claims were corrected (the 40-run `agent-record` verdict
count, the `workflow_dispatch` census, the pull-request-less share of `main`, and
the repair-language count) and four were left unverified and marked as such (the
765 s suite total and three local checker timings). Nothing found in that pass
moves the recommendation; the finding that does bear on process is the duplicate
verdict above.
