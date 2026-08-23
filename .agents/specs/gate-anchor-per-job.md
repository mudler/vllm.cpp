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
logits-processor bounce on host addressability, not unified memory (#1752)`. The
append is suppressed only when the merger supplies an explicit `commit_title`,
which is what happened to this batch of five on 2026-08-18 — three of them say
so in their own bodies, under "Maintainer changes on top".

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

`scripts/main-baseline.py --gate-anchor <job>` walks the newest `--limit`
(default 20) push runs on the branch, newest first, and returns the `head_sha`
of the first run in which every payload entry matching that job id carries a
conclusion in `{success, failure}`.

**The window is the floor.** When no run in the window qualifies, the anchor is
the `head_sha` of the **oldest** run in the window. A range can therefore never
widen past 20 pushes however long `main` has been red, and the degradation is
toward *more* coverage rather than less — the failure this floor must not have
is silently skipping commits.

When the query degrades or the branch has no push run at all, the anchor is
empty and the consumer keeps today's `PUSH_BASE` fallback. `REMOTE_UNVERIFIED`
stays what it is: not a pass, and not a claim of absence.

The current run never anchors itself: its own job is `in_progress`, so its
conclusion is `null` and it does not qualify.

### 4.3 What "no commit is skipped" means, exactly

`github.event.before` chains: each push's `before` is the previous push's `sha`,
so the union of the naive ranges covers every commit **provided every push's job
runs**. #863 was the hole a cancelled job leaves in that chain. The per-job
anchor closes exactly that hole and nothing else: from the last run in which the
job concluded, the chain is unbroken by construction, and the span since then is
walked whole. §6's `test_no_commit_is_ever_skipped` asserts the union property
over a synthetic push sequence rather than arguing it.

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
4. **`ArrivalDiscriminatorTests`** — pins §5's decision in
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

Recorded in `## Outcome` on completion: the red output of each new case before
the change, green after, each mutation's `git diff --stat` and interpreter
output, and the byte-for-byte restore proof.

## 9. Stop conditions

- Stop if `test_no_commit_is_ever_skipped` cannot be made to hold. Escaping the
  cycle by skipping commits is #863 again and is worse than the cycle.
- Stop if any rule in `check-role-discipline.py` has to move to make the range
  work. §1 says the range changes and the demands do not.
- Stop if the anchor query costs more than a bounded number of API calls per
  job; an unbounded walk is a new failure mode, not a fix.
- Stop and return `NEEDS_DECISION` if the "concluded, not green" argument in
  §4.1 is rejected, because every other part of the fix depends on it.
