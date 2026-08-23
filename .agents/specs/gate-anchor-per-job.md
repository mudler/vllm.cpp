# GATE-ANCHOR-PER-JOB — a diff gate anchors on its OWN last verdict, not on a run-level `success`

**Row:** `GATE-ANCHOR-PER-JOB`
**Issue:** [#1773](https://github.com/mudler/vllm.cpp/issues/1773)
**Refs:** [#1764](https://github.com/mudler/vllm.cpp/issues/1764) (the live reds), [#274](https://github.com/mudler/vllm.cpp/issues/274) / [`main-verifiability.md`](main-verifiability.md) (the tool that already knew), [#822](https://github.com/mudler/vllm.cpp/issues/822) and [#863](https://github.com/mudler/vllm.cpp/issues/863) (why the anchor exists at all)
**Base:** `origin/main` `21abaf169f1ce0bcaf2598056c6a0278e8bf0241`
**Status:** ACTIVE, 2026-08-23

## 1. Scope

**In.** How the three diff-scoped jobs in `.github/workflows/ci.yml` resolve the
`base` of the range they walk on the push lane, and the reader that resolves it:

- `documentation-checkpoint` — `check-now-current.py`, `check-role-discipline.py`
- `commit-protocol-tag` — the `FOLLOWING_AGENTS_PROTOCOL` walk, `check-commit-trailers.py`
- `agent-record`'s role-discipline step — `check-role-discipline.py`
- `scripts/main-baseline.py` — gains the anchor query, in the module that
  already owns the per-job doctrine
- `last-gated-commit` — stops resolving a shared string; keeps its closed-PR
  guard role, which is the only thing `agent-record` may inherit an `if:` from
  (#873)

**Out of scope, deliberately.** Every rule the three jobs enforce once the range
exists. `check-role-discipline.py` is **not** modified: not its path
classification, not its cutovers, not `arrives_via_row_pr`. §5 records why, and
that decision is the one a reviewer should attack first. Also out: the
`SiteGuard` error and the `commit-protocol-tag` cause in #1764 §1 and §3 — this
row owns #1764 §2 only.

**This alters which commits a gate examines, never what is demanded of them.**

## 2. Anchors

Local CI plumbing. No vLLM counterpart: vLLM's CI gates a different repository
layout and has no record protocol to diff-scope. Nothing is ported.

| What | Where |
|---|---|
| The defective query | `.github/workflows/ci.yml`, job `last-gated-commit`, step `resolve` |
| Its three consumers | `ci.yml` jobs `agent-record`, `documentation-checkpoint`, `commit-protocol-tag` |
| The doctrine it contradicts | `scripts/main-baseline.py` module docstring, "this is why nothing here reads `run.conclusion`" |
| The per-job verdict machinery being reused | `scripts/main-baseline.py` `job_matches`, `jobs_for`, `gh_api` |
| What actually reddens today | `scripts/check-role-discipline.py` `arrives_via_row_pr` |

## 3. What is measured, at the base revision

Five facts. Each was read from the forge or from this tree on 2026-08-23, and
each is reproducible with the command beside it.

### 3.1 The anchor is eleven days stale, and the range is 484 commits

```
gh api "repos/mudler/vllm.cpp/actions/workflows/ci.yml/runs?branch=main&event=push&status=success&per_page=1" \
  --jq '.workflow_runs[0].head_sha'
  -> fafa16f0f32acc8255e113a2cbc35f8b99cf2072   (2026-08-12T23:53:24Z)

git rev-list --count fafa16f0f..origin/main   ->  484
```

That is what `documentation-checkpoint` walks on the push lane today.

### 3.2 The gate's verdict on `main` is decided by a race, not by the tree

Run `32625264281`, push of `1fdd3e26d` (2026-08-23T07:19:38Z):

```
last-gated-commit:        completed/cancelled
documentation-checkpoint: completed/success
commit-protocol-tag:      completed/success
run conclusion:           cancelled
```

`documentation-checkpoint` carries `if: always()`, so it runs even when the job
that resolves its base was cancelled. `LAST_GREEN` then renders empty, the step
falls back to `PUSH_BASE`, and the gate passes over one push. When
`last-gated-commit` survives instead, the same job walks 484 commits and fails.

**So `main`'s diff gates are green when the resolver loses the cancellation race
and red when it wins.** Neither reading is about the tree. This is sharper than
the cycle #1773 describes and it is the same root: a shared string, resolved in
a separate cancellable job, at the run level.

### 3.3 A run-level query cannot see a job that concluded

The six newest completed push runs on `main`, run conclusion against
`documentation-checkpoint`'s own conclusion:

| run | head | run conclusion | `documentation-checkpoint` |
|---|---|---|---|
| 32627787237 | `15298f033` | cancelled | absent (never started) |
| 32626139993 | `aa67130cc` | cancelled | absent |
| 32626052337 | `6991b78d2` | cancelled | absent |
| 32625688839 | `38e6ac0a3` | cancelled | absent |
| 32625264281 | `1fdd3e26d` | cancelled | **success** |
| 32625120219 | `175733000` | cancelled | absent |

A per-job reader anchors on `1fdd3e26d`, eleven days newer than what the
run-level query returns. `main-baseline.py` has said "the verdict is computed
from PER-JOB conclusions, always" since #274; the job that asks the question
does not call it.

### 3.4 The commits being re-flagged are five, not three, and every one arrived through a merged pull request

`python3 scripts/check-role-discipline.py --base fafa16f0f --head HEAD` reports
`dd8a3b0e1`, `8daf58e77`, `38ec0da4a`, `5073df622`, `65d6cdaed`. #1764 saw three
because it measured at an older head: the widening range **accretes** offenders,
which is the cycle's signature.

| commit | pull request | head branch | fork? |
|---|---|---|---|
| `dd8a3b0e1` | [#640](https://github.com/mudler/vllm.cpp/pull/640) | `ElderOrb/vllm.cpp:fix/windows-msvc-vulkan-build` | yes |
| `8daf58e77` | [#1159](https://github.com/mudler/vllm.cpp/pull/1159) | `mudler/vllm.cpp:row/ENG-RELEASE-WINDOWS-test-thread-raii` | **no** |
| `38ec0da4a` | [#1056](https://github.com/mudler/vllm.cpp/pull/1056) | `tbrasser/vllm.cpp:row/BACKEND-ROCM-ATTN-REGISTER` | yes |
| `5073df622` | [#1065](https://github.com/mudler/vllm.cpp/pull/1065) | `tbrasser/vllm.cpp:row/BACKEND-ROCM-ATTN-RUNNER` | yes |
| `65d6cdaed` | [#945](https://github.com/mudler/vllm.cpp/pull/945) | `jimmykarily/vllm.cpp:row/BUILD-GCC16` | yes |

All five are `merged`, all five have `merge_commit_sha` equal to the flagged
commit, and all five landed on 2026-08-18.

### 3.5 The fork hypothesis is REFUTED, and the real discriminator is the squash subject

#1764 and #1773 both carry the hypothesis that an external contributor's branch
lives on a fork, "so a checker that verifies the change arrived on a task branch
by looking at `origin`'s refs cannot find it and never will."

**`check-role-discipline.py` reads no ref.** For a single-parent commit
`arrives_via_row_pr` is `ROW_BRANCH.search(subject + body) or
PR_REFERENCE.search(subject)` — commit message text and nothing else. `git
ls-remote` is never called, `origin/*` is never resolved. The hypothesis
predicts that the one non-fork commit passes; `8daf58e77` came from
`mudler/vllm.cpp:row/…` and fails identically. The hypothesis is refuted by its
own table.

What the five share is their **subject**:

```
windows: fix native MSVC/Vulkan build portability
fix(ENG-RELEASE-WINDOWS): the api-server gate can report its own failure again
feat(BACKEND-ROCM): register a ROCm attention backend for kROCM
feat(BACKEND-ROCM): select the attention backend in the runner
build: make the tree compile on gcc 16, and add a CI lane so it stays that way
```

No `(#N)`. Every other squash on `main` has one, because GitHub appends the
number even under this repository's `squash_merge_commit_title = PR_TITLE`:
[PR #1752](https://github.com/mudler/vllm.cpp/pull/1752)'s title contains no `#`
at all and it landed as `fix(V1-LOGITSPROC-HOST-ADDRESSABLE): gate the
logits-processor bounce on host addressability, not unified memory (#1752)`.

**The mechanism is INFERRED, not measured, and this row cannot measure it.** An
explicit `commit_title` is the documented way to suppress the append, and it is
the only one this row could find. What the forge does not record is whether the
merge call actually carried that field: the pulls endpoint keeps no merge-input
payload, so the hypothesis is consistent with the evidence rather than
demonstrated by it. Two facts weaken it further, and both are stated here rather
than left out. Four of the five pull requests carry a "Maintainer change on top"
section in their bodies, not three as this spec first said, and #1159 carries
none. And all five were merged by the SAME account, `localai-bot`, that appends
`(#N)` correctly on every other squash it lands, so whatever differed on
2026-08-18 differed within one actor's own behaviour.

The design does not rest on this. §4 is a coverage fix and is correct whatever
suppressed the append; the landing rule in AGENTS.md is worth writing because
the default title is right regardless of which path produced the exception.

| pull request | merged by | "Maintainer change on top" |
|---|---|---|
| [#640](https://github.com/mudler/vllm.cpp/pull/640) | `localai-bot` | yes |
| [#1159](https://github.com/mudler/vllm.cpp/pull/1159) | `localai-bot` | no |
| [#1056](https://github.com/mudler/vllm.cpp/pull/1056) | `localai-bot` | yes |
| [#1065](https://github.com/mudler/vllm.cpp/pull/1065) | `localai-bot` | yes |
| [#945](https://github.com/mudler/vllm.cpp/pull/945) | `localai-bot` | yes, singular |

**So the rule is not unsatisfiable by an external contributor.** A fork pull
request merged with the default squash title passes today. There is nothing to
teach the checker and nothing to waive. §5 is that decision, argued.

## 4. Design

### 4.1 The anchor is per JOB, resolved BY that job

The base a diff-scoped gate walks from is defined by one question:

> From which commit onward has **this job** not yet returned a verdict?

Three properties follow, and each fixes one measured defect.

**PER JOB, not per run.** `sanitize-cpu` is `continue-on-error`, so a run's
conclusion can read `success` over a red job (#274); and a run's conclusion reads
`cancelled` over a job that ran to completion (§3.3). The run conclusion answers
a different question and must never be read. This is `main-baseline.py`'s
doctrine, applied where the question is asked.

**CONCLUDED, not GREEN.** The anchor advances past a commit at which the job
concluded `failure` as readily as past one where it concluded `success`. This is
the change that breaks the cycle, and it is the one to argue for:

- The gate is **per commit**. Every commit in range is inspected on its own; the
  range decides only which commits are inspected.
- A commit on `main` is immutable. Once a violation lands, no later push can
  repair it. Anchoring on `success` therefore converts one violation into a
  permanent block on every future push — and the gate stops being able to say
  anything about new commits, because it is drowned by an old one it cannot fix.
- Anchoring on "concluded" gives every commit **exactly one** verdict, from the
  first run of that job whose range contains it. One alarm per violation is a
  complete alarm. A blocked branch is not enforcement; it is a broken alarm.
- The rule is still **blocking** where blocking works: on the pull request lane
  the same checkers run over `base..head` and refuse the merge. The push lane is
  the detector for whatever bypassed that, and its verdict is a report.

**RESOLVED BY THE CONSUMING JOB.** §3.2 is not a variant of the cycle, it is a
second defect: a shared string resolved in a separate job that `if: always()`
consumers outlive. Each job resolves its own anchor in its own first step, so
the anchor exists exactly when the gate does. `last-gated-commit` stops
resolving anything and keeps only the closed-PR guard that `agent-record`
inherits through `needs:` (#873).

### 4.2 The floor

`scripts/main-baseline.py --gate-anchor <job> --gate-step <name>...` walks the
newest 20 push runs on the branch, newest first, and returns the `head_sha` of
the first run in which every payload entry matching that job id carries a
conclusion in `{success, failure}` **and every named step inside it does too**.

**The window bounds the RANGE. It does not bound the loss, and the first draft
of this section claimed otherwise.** When no run in the window qualifies the
anchor falls back to a floor, so a range never widens past the window however
long `main` has been red. The sentence "the degradation is toward more coverage
rather than less" was false of the code it described. Measured on the
implementation as first written: 21 non-qualifying pushes put the anchor at run
2 and left run 1 outside every future range, and 25 left runs 1 to 6 outside.
Commits roll off the back permanently, because a range that starts inside the
window can never reach behind it again.

Two things are done about that, and neither is a denial.

**The off-by-one is fixed.** The query now reads `window + 1` runs. The first
`window` are anchor CANDIDATES and the extra one is the floor, so the oldest
candidate's own head sits INSIDE the floor range instead of on its base. It used
to sit on the base and its commit was excluded from the very range the floor
exists to guarantee. This costs nothing: it is the same single runs call with
`per_page` one higher.

**The residual loss is stated with its bound.** A commit is lost only when its
own gate step has failed to return a verdict on `window` consecutive pushes, and
only after the pull request lane already gated it at merge. Walking further is
the alternative, and it costs an unbounded number of API calls per push across
three jobs. The trade is deliberate.
`test_past_the_window_commits_roll_off_PERMANENTLY` asserts it as a property of
the code, so it cannot quietly stop being true.

**A degraded query SKIPS the gate. It does not narrow it.** `--gate-anchor`
exits 3 on `REMOTE_UNVERIFIED` and 1 on a clean absence, the split AGENTS.md
already documents for `scripts/agent-pr-body.py`. The two may not be collapsed.
On a clean absence the branch has no gated history and `PUSH_BASE` is the honest
base. On a degraded read the base is UNKNOWN, and falling back to `PUSH_BASE`
would run a narrowed pass whose success then advances the step's own anchor past
everything the narrowing dropped. The anchor step exports
`GATE_ANCHOR_DEGRADED=true` and the gate steps' `if:` turns that into a skip. A
skipped step is not a verdict, so the anchor cannot advance and the next
readable run walks the span whole.

The current run never anchors itself: its own job is `in_progress`, so its
conclusion is `null` and it does not qualify.

### 4.3 What "no commit is skipped" means, exactly, and why the JOB is the wrong unit

`github.event.before` chains: each push's `before` is the previous push's `sha`,
so the union of the naive ranges covers every commit **provided every push's job
runs**. #863 was the hole a cancelled job leaves in that chain.

**The first draft of this row re-created that hole one level down, and the claim
"unbroken by construction" was false.** GitHub concludes a job `failure` the
moment any step fails, and marks every REMAINING step `skipped`. A job-level
question therefore reads `failure` over a gate that refused the range and over a
gate that never executed, and cannot tell them apart. Both `agent-record` and
`commit-protocol-tag` place their diff-scoped gate AFTER other steps, so under a
job-level anchor the base advances past commits the gate never looked at.

Measured live on `commit-protocol-tag`, where the strict trailer walk runs
nowhere else on the push lane:

| run | head | job | strict-trailer step |
|---|---|---|---|
| 32599040638 | `038ff61e5` | success | success |
| 32601353990 | `1a1d17e53` | failure | **skipped** |
| 32608320394 | `6354755ba` | failure | **skipped** |
| 32613454280 | `b508cbce6` | failure | **skipped** |
| 32616777372 | `66d1b0a90` | failure | **skipped** |
| 32623377380 | `a4f2a9585` | failure | **skipped** |
| 32625264281 | `1fdd3e26d` | success | success |

A job-level anchor walks `038ff61e5` to `1a1d17e53` to `1fdd3e26d`, and
`038ff61e5..a4f2a9585`, six commits, never receives a trailer verdict from any
run. `agent-record` has the same shape: the last push run in which its
role-discipline step actually executed is 32580850008 at `8540a2755`, and
`8540a2755..66d1b0a90` is twelve commits. Its coverage survived only because
`documentation-checkpoint` ran the same checker, which is luck rather than
construction. The old run-level rule could not do this, because a failed
`agent-record` forces the RUN conclusion to `failure` and that run never
anchored at all.

**The unit is therefore the STEP, and three things enforce it.**

1. `--gate-step` is REQUIRED. `steps_concluded` reads `steps[].conclusion` from
   the jobs payload and qualifies a run only when every named step of every
   matching entry concluded. A named step that is absent does not qualify: it is
   missing because it was never reached, because it was renamed, or because the
   payload is truncated, and none of the three is evidence that the gate ran.
   The flag-less form exits 2 rather than defaulting to the job.
2. **Every diff-scoped step runs on every push.** Each gate step carries
   `if: !cancelled() && steps.checkout.outcome == 'success' && steps.anchor.outcome == 'success' && env.GATE_ANCHOR_DEGRADED != 'true'`.
   The first condition is the one that closes the hole: without it an earlier
   step's failure marks the gate `skipped`, and the anchor waits for a verdict
   that never comes. The other three are exactly the cases where a conclusion
   would be a lie, and in each the step skips rather than reporting one.
3. **One gate per step.** `documentation-checkpoint` ran `check-now-current.py`
   and `check-role-discipline.py` in ONE step under `set -eu`, so a
   `check-now-current` failure aborted before the arrival gate ran while the
   step still concluded. That is the same hazard inside a single step, and the
   step is now split in two. `test_one_diff_scoped_checker_per_gate_step` holds
   the shape, including that nothing fallible precedes the checker in its body.

§6's `test_no_commit_is_ever_skipped` asserts the property at the granularity
that can actually fail. The union of the ranges is NOT that property: a
job-level anchor keeps the union whole while the gate is skipped on every push,
because `failure` advances the base whether or not the gate executed. The
property is that for every commit there is at least one push whose range
contains it AND whose gate step returned a verdict.

**The transition costs one wide range, in the safe direction.** Measured on
`origin/main` at `11ccdcf76`: `commit-protocol-tag` resolves a verdict anchor at
`ff8f728071` because both its step names already exist, while `agent-record` and
`documentation-checkpoint` fall back to the window floor `08c81a892`, a bounded
23 commits. `documentation-checkpoint` floors because its second step is new in
this change and has no history, and `agent-record` floors because its
role-discipline step was skipped on all 20 pushes in the window, which is the
defect being fixed reporting itself. 23 is the floor working. 484 was the bug.

## 5. The decision on the five commits, and what was rejected

**Decision: `check-role-discipline.py` is not touched, and no exception is
recorded.** With §4 in place the five commits are older than the newest run in
which `documentation-checkpoint` concluded, so they leave the range on the first
push after this lands. They were already reported — `documentation-checkpoint`
concluded `failure` at `dd8a3b0e1`'s own push (run `32080067480`) and at
`8daf58e77`'s (run `32108685135`), and every red run since has named all five.
The alarm rang. It cannot ring them into a state they can no longer reach.

Three alternatives were evaluated and rejected.

| Option | Rejected because |
|---|---|
| Teach the checker about fork-origin branches | It would fix nothing: §3.5 shows the checker reads no ref, and the one **non**-fork commit of the five fails identically. It repairs a hypothesis, not a defect. |
| Widen `PR_REFERENCE` from the subject to the whole message | It deletes the obligation. `dd8a3b0e1`'s body says `Issue: #503`, and *every* commit in this repository names an issue in its body, because AGENTS.md requires one. The gate would pass every direct-to-main push ever made. §6's `test_a_body_only_issue_reference_does_not_satisfy_arrival` pins this shut. |
| A bounded exception list of the five SHAs | AGENTS.md: "The project has no waiver registry. An exception registry is a state log, and this protocol has no state log." A five-line allowlist in a checker is that registry, and it would outlive its reason. |

**What the recurrence needs instead is a landing rule, not a checker change.**
AGENTS.md gains one sentence under `## Landing work`: land a squash with the
default title so GitHub appends `(#N)`, because the arrival gate reads the
commit message and a custom `commit_title` suppresses the only evidence it has.
That is where the defect was introduced and where it can be prevented.

## 6. Tests

All offline. `tests/scripts/test_main_baseline.py` gains four classes; nothing
existing is relaxed.

1. **`GateAnchorTests`** — `gate_anchor()` against synthetic payloads.
   - a run whose *conclusion* is `cancelled` but whose named job concluded
     `success` **is** the anchor. RED before: no such function.
   - a run whose named job concluded `failure` **is** the anchor.
   - a run whose named job is `cancelled`, `skipped`, absent, or still `null` is
     **not** the anchor and the walk continues.
   - a matrix job anchors only when **every** lane concluded.
   - no qualifying run in the window returns the window's oldest head — the
     floor — and says so.
2. **`AnchorCycleConstructionTests`** — the feedback loop, built rather than
   read. A synthetic sequence of pushes P1..P6 in which P2 is a violating commit
   and P3..P6 are ordinary, run through **both** anchor rules:
   - `test_run_level_anchor_widens_across_pushes` — the run-level rule holds the
     anchor at P1 and the range grows 1, 2, 3, 4, 5, re-including P2 every time.
     This is the cycle, asserted as a sequence of range sizes.
   - `test_per_job_anchor_reports_the_violation_once` — the per-job rule reports
     P2 exactly once and the range never exceeds the gap since the last verdict.
   - `test_no_commit_is_ever_skipped` — the union of every range equals every
     commit, on both rules, including across a cancelled run. The per-job rule
     may not buy its exit from the cycle with a hole.
3. **`AnchorStepTests`** — executes the three real step bodies out of `ci.yml`
   under the existing `run_shimmed` argv recorder, with `gh`/`python3` shimmed:
   - each of the three jobs resolves an anchor naming **its own** job id;
   - a push whose resolver returns nothing falls back to `PUSH_BASE`;
   - the `pull_request` lane still uses `PR_BASE`/`PR_HEAD` and never queries
     the forge.
5. **The step-granularity cases added by the review repair** (#1776). All in
   `tests/scripts/test_main_baseline.py`:
   - `test_no_commit_is_ever_skipped`, RE-EXPRESSED at step granularity. The
     job-level version modelled a job as one atomic verdict and could not fail
     for this defect, which made it a correct test of the wrong thing.
   - `test_a_SKIPPED_gate_step_in_a_CONCLUDED_job_does_not_anchor` states F1 as
     one assertion, on the live shape of runs 32601353990 to 32623377380.
   - `test_a_gate_step_ABSENT_from_the_payload_does_not_anchor` and
     `test_every_named_step_must_conclude_not_just_one`.
   - `test_an_anchor_with_no_named_step_is_REFUSED`, where the flag-less form raises
     rather than silently meaning the job.
   - `test_the_floor_covers_the_oldest_CANDIDATES_own_head` and
     `test_past_the_window_commits_roll_off_PERMANENTLY` cover F2, fixed where it
     could be and asserted where it could not.
   - `test_every_named_gate_step_exists_in_its_own_job`,
     `test_every_step_that_READS_the_anchor_is_NAMED_by_it`,
     `test_every_gate_step_SKIPS_rather_than_narrows` and
     `test_one_diff_scoped_checker_per_gate_step` hold the workflow shape the fix
     depends on, selected as a POPULATION by "reads `GATE_ANCHOR`" so a newly
     added diff-scoped gate fails here rather than landing uncovered.
   - `test_a_DEGRADED_query_skips_the_gate_instead_of_narrowing_it` and
     `test_a_CLEAN_absence_still_falls_back_to_push_base` cover F3, executed
     through the real step body with the `python3` shim exiting 3 and then 1.
6. **`ArrivalDiscriminatorTests`** — pins §5's decision in
   `tests/scripts/test_check_role_discipline.py`: a subject carrying `(#N)`
   satisfies arrival, and a body-only `#N` with a bare subject does **not**.
   This is the test that must red if anyone widens the match later.

## 7. Gates

- `python3 tests/scripts/test_main_baseline.py` and
  `python3 tests/scripts/test_check_role_discipline.py`, both with the new cases
  shown RED against the unmodified tree.
- `python3 -c "import yaml"` round-trip of `ci.yml` **plus an explicit
  duplicate-key scan**: PyYAML accepts duplicate mapping keys that GitHub
  rejects, so a parse is not a validation.
- `scripts/agent-preflight.sh` and `python3 scripts/agent-ready.py`.
- `python3 scripts/check-role-discipline.py` over this branch's own range.
- `python3 scripts/main-baseline.py --gate-anchor documentation-checkpoint`
  against the live forge, reported with the SHA it returns.

## 8. Evidence

Measured on `row/GATE-ANCHOR-PER-JOB`, base `21abaf169`, merged onto
`11ccdcf76`. Every mutation below was applied to a tree whose five files were
hashed first, printed with `git diff --stat`, parsed to prove it was not a
build failure wearing a pass, and restored against the hash — never against a
harness's own cleanup. `git status --porcelain` is empty and `sha256sum -c`
reports `OK` on all five afterwards.

### RED before, GREEN after, first round (head `140745e64`)

`python3 tests/scripts/test_main_baseline.py` on the unmodified tree:
`Ran 81 tests`, `FAILED (failures=7, errors=11)`. After: `Ran 82 tests`, `OK`.
`test_run_level_anchor_widens_across_pushes` and the `PUSH_BASE` fallback case
pass on both sides by design — they characterise the defect and the degrade.

`python3 tests/scripts/test_check_role_discipline.py`: `Ran 22 tests`, `OK`.
Its four new cases are characterisation pins, so mutation is the only thing that
shows they bite. M4 below is that.

### The live gate, before and after

| anchor | source | SHA | `fafa16f0f..` or `SHA..origin/main` | `check-role-discipline.py` |
|---|---|---|---|---|
| run-level `status=success` (removed) | run conclusion | `fafa16f0f` (2026-08-12) | **484** commits | `rc=1`, five ERRORs |
| `--gate-anchor documentation-checkpoint` | verdict, run `32626481436` | `ff8f72807` | **2** commits | `rc=0`, `OK` |
| `--gate-anchor commit-protocol-tag` | verdict, run `32626481436` | `ff8f72807` | 2 commits | `check-commit-trailers.py` `OK` |
| `--gate-anchor agent-record` | verdict, run `32616777372` | `66d1b0a90` | **13** commits | `rc=0`, `OK` |

The last row is the argument for a per-job anchor stated as a number: two jobs
in the same workflow have anchors eleven commits apart, because they have
different cancellation profiles. One shared string could not have been right for
both.

### Mutations

| # | Mutation | `git diff --stat` | Result |
|---|---|---|---|
| M1 | `CONCLUDED` admits `cancelled` | `1 insertion(+), 1 deletion(-)` | 3 FAIL: the cancelled/skipped/absent case, the matrix case, the floor case |
| M2 | `resolve_gate_anchor` reads `run["conclusion"] == "success"` again | `1 +, 1 -` | 3 FAIL, including `test_per_job_anchor_reports_the_violation_once` with `AssertionError: 2 != 1 : range at p3: ['p2', 'p3']` — **the cycle, printed by the test that constructs it** |
| M3 | delete `documentation-checkpoint`'s anchor step from `ci.yml` (the production call site) | `34 deletions(-)` | 2 FAIL + 2 ERROR, including the re-pinned `ConcurrencySemanticsTests` case |
| M4 | `PR_REFERENCE.search(subject)` becomes `…search(message)` — §5's rejected option | `1 +, 1 -` | 2 FAIL, and `check-role-discipline.py` over the 484-commit range turns **`OK`**. The widening "fixes" #1764 §2 by deleting the obligation, and the two new cases are what stop it |
| M5 | a genuine direct push inside the NARROWED range: one commit on `probe/direct-push` at `ff8f72807` touching `src/vllm/version.cpp`, subject with no `(#N)`, issue in the body only | `1 file changed, 1 insertion(+)` | `ERROR: 03fd91554: repository change (src/vllm/version.cpp) reached main without arriving on a task branch`, `rc=1`. **The narrowed range still catches what the gate exists for.** Branch deleted, tree restored |

M3 and M5 each parse and apply — M3's YAML loads, M5's commit exists and is
reported by SHA — so neither reading is an unapplied edit wearing a pass.

### The review repair (#1776), RED before and GREEN after

`test_no_commit_is_ever_skipped`, re-expressed at step granularity and run
against the UNMODIFIED implementation. The adapter it calls through forwards
everything except the step names, so the red and the green assert byte-identical
properties and the only difference is whether the anchor may see a step:

```
FAIL: test_no_commit_is_ever_skipped
AssertionError: Items in the second set but not the first:
'p2'
'p3' : commits with no verdict from any run that ran the gate: ['p2', 'p3']
```

After: `python3 tests/scripts/test_main_baseline.py` reports `Ran 99 tests`,
`OK`, up from 82. `python3 tests/scripts/test_check_role_discipline.py` reports
`Ran 22 tests`, `OK`, unchanged.

The F1 defect is also confirmed against the live forge rather than only in the
model. The table in §4.3 is `gh api .../jobs?filter=latest` on the seven runs,
and `git rev-list --count 038ff61e5..a4f2a9585` is `6` while
`8540a2755..66d1b0a90` is `12`.

### The live gate after the repair

Measured against `origin/main` at `11ccdcf76`:

| anchor | source | SHA | range |
|---|---|---|---|
| `commit-protocol-tag`, both steps named | verdict, run `32626481436` | `ff8f728071` | 2 commits |
| `agent-record` | **floor**, run `32594040335` | `08c81a892` | 23 commits |
| `documentation-checkpoint` | **floor**, run `32594040335` | `08c81a892` | 23 commits |

The two floors are the transition, and they are the design working rather than
failing. `documentation-checkpoint` floors because its second gate step is new
in this change and has no history on `main` yet. `agent-record` floors because
its role-discipline step was `skipped` on all 20 pushes in the window, which is
the defect this row fixes reporting its own extent. 23 commits is bounded by the
window. 484 was the bug.

### Mutations, repair round

Every mutation below was applied to a tree hashed first, printed with
`git diff --stat`, parsed to prove it was not a syntax error wearing a pass, and
restored against the hash rather than against the harness's cleanup. The harness
asserts its anchor matches before writing, which caught one mis-written mutation
that would otherwise have read as a passing test.

| # | Mutation | `git diff --stat` | Result |
|---|---|---|---|
| M6 | `steps_concluded` ignores `steps[]`, restoring job granularity: **the finding itself** | `121 +, 27 -` | 4 FAIL, led by `test_no_commit_is_ever_skipped` |
| M7 | drop `!cancelled()` from all five gate-step guards | `248 +, 24 -` | 5 FAIL, one per gate step |
| M8 | drop `env.GATE_ANCHOR_DEGRADED != 'true'` from all five guards | `243 +, 24 -` | 5 FAIL |
| M9 | the anchor CLI exits 1 on a degraded read instead of 3 | `123 +, 26 -` | 1 FAIL, `test_the_anchor_CLI_exits_3_on_a_degraded_read` |
| M10 | `gate_anchor` fetches `window` runs, not `window + 1`: F2's off-by-one restored | `123 +, 26 -` | 1 FAIL, `test_gate_anchor_reads_one_run_PAST_the_window` |
| M11 | a `--gate-step` flag is dropped from `ci.yml`, leaving a gate nothing waits for | `247 +, 24 -` | 2 FAIL, including the population test |
| M12 | the arrival gate goes back to sharing a step with `check-now-current.py` | `249 +, 24 -` | 1 FAIL, `test_one_diff_scoped_checker_per_gate_step` |
| M13 | a gate step is renamed in `ci.yml` and the `--gate-step` value is left behind | `249 +, 25 -` | 2 FAIL + 2 ERROR |

M9 and M10 initially came back GREEN against a first draft of these tests. Both
were real gaps rather than mutation errors: the F3 test drove the shim's exit
code and never the script's own, and the floor test used exactly `window + 1`
runs, where `runs[window]` and `runs[-1]` are the same entry and the assertion
could not discriminate. `test_the_anchor_CLI_exits_3_on_a_degraded_read` and
`test_gate_anchor_reads_one_run_PAST_the_window` were added for that reason, and
the table above is the re-run.

### The five original mutations, re-run

| # | Result now |
|---|---|
| M1 | 5 FAIL, up from 3: the two new floor tests also detect it |
| M2 | 3 FAIL, including `test_per_job_anchor_reports_the_violation_once` |
| M3 | 3 FAIL + 8 ERROR, up from 2 + 2: deleting the production anchor step now also breaks the step-shape population |
| M4 | 2 FAIL, unchanged, on a checker this change does not touch |
| M5 | reproduces: `ERROR: 7c16436d8: repository change (src/vllm/version.cpp) reached main without arriving on a task branch`, `rc=1` |

**M5 needs one note.** Run from THIS worktree it reports `REPORT` and `rc=0`, not
`ERROR` and `rc=1`. That is `has_reached_main`
(`scripts/check-role-discipline.py:317-325`) reading the checkout's own branch
name: on a `row/*` branch an unmerged commit is pending disposition rather than
landed history. Re-run from a detached checkout, which is CI's shape, it gives
`ERROR` and `rc=1` exactly as recorded. `scripts/check-role-discipline.py` is
byte-identical to `origin/main`, so the difference is the checkout and not this
change. The probe commit was built with `git commit-tree` against a temporary
`GIT_INDEX_FILE`, so no ref, index or working tree was touched.

### Records and shape

`.github/workflows/ci.yml` parses under PyYAML **and** an explicit duplicate-key
scan reports `duplicate keys: 0`, because PyYAML accepts duplicates GitHub
rejects. `scripts/agent-preflight.sh` rc 0. `check-commit-style.py` and
`check-commit-trailers.py` `OK` over the branch's own range.
`git diff --numstat origin/main -- .agents/issue-index.md` is `1 0`, and the row
count is 636 on `origin/main` and 637 here, counted again after the merge.

## 9. The fresh review on PR #1776, and what it changed

The review confirmed the fork refutation, the `if: always()` race, the live
anchors, the YAML cleanliness and all five mutations, and FAILED the pull
request on four findings. All four are repaired here.

| # | Finding | Repair |
|---|---|---|
| F1 | **Critical.** The fix re-created #863's hole at STEP granularity. `job_concluded` asked only whether the JOB concluded, and GitHub marks every remaining step `skipped` when an earlier one fails, so the anchor advanced past commits the gate never ran on. Six commits on `commit-protocol-tag`, twelve on `agent-record`. | The unit is the step: `--gate-step` is required, `steps_concluded` reads `steps[].conclusion`, each gate step carries `if: !cancelled() && …` so it runs on every push, and `documentation-checkpoint`'s two-checker step is split in two. §4.3. |
| F2 | **Medium.** Past the 20-push window the floor degraded toward LESS coverage, and §4.2 claimed the opposite. | The off-by-one is fixed by reading `window + 1` runs. The residual loss is stated with its bound and asserted by a test instead of denied. §4.2. |
| F3 | **Low.** `\|\| true` swallowed `REMOTE_UNVERIFIED`, so a degraded query silently narrowed the gate to `PUSH_BASE`. With a step-level anchor that is not only dishonest, it is a coverage hole: the narrowed pass advances the anchor past what the narrowing dropped. | rc 3 means degraded and rc 1 means clean absence, the split AGENTS.md already documents for `agent-pr-body.py`. A degraded query SKIPS the gate. §4.2. |
| F4 | **Low.** The new AGENTS.md rule added the file's only em dash, named no command, and addressed the contributor rather than the automation that performed all five merges. | Rewritten with no em dash, naming `gh pr merge --squash --subject` and the merge endpoint's `commit_title`, binding the MERGING account, and saying plainly that no gate can catch it. |

**F5 is noted and NOT repaired here.** It is informational and pre-existing:
`test_every_diff_scoped_step_bases_on_its_own_jobs_anchor` selects its steps by
`github.event.before` appearing in a step `env:` block, so
`commit-protocol-tag`'s first gate step, which interpolates
`${{ github.event.before }}` inline instead, is outside that test's population.
The step is nonetheless covered by the population test added here, which selects
on `GATE_ANCHOR` and does include it. Normalising the two steps onto one
env-based form is a tidy-up that belongs to whichever row touches that job next,
not to a review repair.

**One claim was softened rather than defended.** §3.5 said the offending squashes
happened because "the merger supplied an explicit `commit_title`". The review
found that plausible but not measured, and it is right: the forge records no
merge-input payload. §3.5 now labels the mechanism inferred, corrects the
"Maintainer change on top" count from three to four, and records that all five
merges were performed by the same account that appends `(#N)` correctly today.

## 10. Stop conditions

- Stop if `test_no_commit_is_ever_skipped` cannot be made to hold. Escaping the
  cycle by skipping commits is #863 again and is worse than the cycle.
- Stop if any rule in `check-role-discipline.py` has to move to make the range
  work. §1 says the range changes and the demands do not.
- Stop if the anchor query costs more than a bounded number of API calls per
  job; an unbounded walk is a new failure mode, not a fix.
- Stop and return `NEEDS_DECISION` if the "concluded, not green" argument in
  §4.1 is rejected, because every other part of the fix depends on it.
