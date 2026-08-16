# GATE-PREFLIGHT-SKIP-REPORT: a skipped block reports SKIP, and a run with a skip is not green

**Row:** `GATE-PREFLIGHT-SKIP-REPORT`
**Issue:** [#998](https://github.com/mudler/vllm.cpp/issues/998)
**Base:** `origin/main` `3ce1cf7c7`
**Status:** ACTIVE, 2026-08-16

## 1. Scope

`scripts/agent-preflight.sh` gains a third per-gate state and a pinned base
revision:

- A block that cannot run prints `SKIP` with the reason that stopped it.
- The summary prints `All gates green.` only when nothing was skipped and
  nothing failed.
- The two blocks that compare against `origin/main` resolve that ref once, at
  the start of the run, and the run names the resolved SHA.

**In scope.** The two range blocks in `scripts/agent-preflight.sh`, the new
`skip` reporting path, the summary, and a new executable suite
`tests/scripts/test_agent_preflight_skip_report.py` that runs the script and
reads its report.

**Out of scope.** Every checker the script calls. This change alters *what the
script reports about itself*, never *what any checker demands*. No checker
gains or loses an assertion.

Also out of scope, and recorded under `## Owed`: `scripts/check-commit-style.py`
never got the merge base repair that `GATE-FORK-ANCESTRY` (#773) applied to
`scripts/check-commit-trailers.py`. See §4.

## 2. Anchors

Local governance script. There is no vLLM counterpart.

| What | Where |
|---|---|
| The block that skips silently | `scripts/agent-preflight.sh`, `Commit trailers vs origin/main` |
| The block that shares the shape | `scripts/agent-preflight.sh`, `Committed range vs origin/main` |
| The banner a skip survives today | `scripts/agent-preflight.sh`, `echo "All gates green."` |
| The principle already settled in tree | `scripts/agent-preflight.sh`, the comment above `run "audit-live-rows"` |
| The in-tree precedent for an honest partial run | `scripts/agent-preflight.sh`, the `ROLE_ONLY` branch |
| The checker that already merge-bases | `scripts/check-commit-trailers.py` `validate_range` |
| The checker that does not | `scripts/check-commit-style.py` `validate_range` |

## 3. Design

### 3.1 The defect

The trailer block is guarded by
`git merge-base --is-ancestor origin/main HEAD`. When that guard is false the
block produces no output at all, and control reaches `echo "All gates green."`
with exit 0. Two gates, `commit-trailers` and `commit-style`, leave the report
without a word.

The file already states the rule this violates, in the comment above
`run "audit-live-rows"`:

> A skip would have to survive the "All gates green." banner below, and a green
> preflight that never verified the record is the one unacceptable outcome

That comment argues the point for one block and then leaves the next two blocks
carrying the defect it names. The principle is settled. The reporting is not.

### 3.2 Why the guard is false so often

`origin/main` is a remote-tracking ref. Every linked worktree of one checkout
shares it. A fetch in any other worktree moves it, so the guard can be true when
the run starts and false when line 226 evaluates it. Three occurrences in one
session:

| Occurrence | Branch | What the run printed | Why the guard was false |
|---|---|---|---|
| 1 | `row/FIX-SERVER-CONCURRENCY-931` | `74 ok, 0 FAIL, All gates green.` | the branch was five commits behind `main` |
| 2 | same, after merging `origin/main` | `All gates green.` again | `origin/main` advanced during the run |
| 3 | `row/BENCH-QWEN38-27B-FOUR-WAY` | `All gates green.` with 74 ok, not 76 | `main` moved during the run |

Occurrence 3 is the proof that this is not a one-off. The count falls by exactly
two and the banner does not change, so nothing in the output distinguishes a run
that checked the trailers from a run that did not.

### 3.3 The three parts of the fix

**Pin the base once.** The script resolves `origin/main` to a SHA before the
first gate runs, stores it in `BASE_SHA`, and uses that SHA everywhere it used
the ref. A concurrent fetch can no longer change the answer mid-run. The run
prints the pinned SHA, so a reader knows what the range blocks compared against.
This alone removes occurrences 2 and 3, because the pinned SHA was an ancestor
at the moment it was captured.

**Report a skip as a skip.** A `skip` function mirrors `run`. It prints a yellow
`SKIP` line, prints the reason on the following line, and appends the label to a
`skipped` array. `SKIP` is a third state. It is not `ok`, because nothing was
verified. It is not `FAIL`, because nothing was found wrong.

**Never call a run with a skip green.** The summary prints the skipped labels
and the line `NOT a green preflight` whenever `skipped` is non-empty. The banner
`All gates green.` is reachable only when both arrays are empty.

### 3.4 Exit code: 0, and never the banner

The run exits 0 when a block was skipped and nothing failed. Four things in the
tree settle this.

1. `--role-only` is the existing precedent for an honest partial run. It runs
   one block, prints `ROLE CHECK ONLY -- this is NOT a full preflight; no record
   gate ran.`, refuses the banner, and exits 0. Its own comment states the
   reason: it "never prints the 'All gates green.' banner, because it skips
   every record gate and mutation suite". The defect was never the exit code.
   The defect was the banner.
2. `AGENTS.md` on gate design: "A gate that fires on ordinary work is the
   defect, not the discipline." A branch behind `main` is ordinary. Every helper
   worktree is behind `main` within minutes of being cut. A preflight that exits
   1 there would be red on most first runs of most sessions, and an agent would
   read that red as a broken record rather than as a missing merge.
3. The `audit-live-rows` comment objects to a skip "surviving" the banner, not
   to a skip existing. After this change no skip survives the banner. The
   objection is answered by the report, not by the exit status.
4. `AGENTS.md` on unknown state: "Unknown is not absence or success". A `SKIP`
   line with a reason records the unknown. Exit 0 with a banner claims success.
   Exit 0 without a banner, with the unknown named, is the accurate report.

Exit 1 keeps its meaning: a gate ran and failed. Widening exit 1 to cover "a
gate did not run" would merge two different facts into one signal, which is the
conflation this row exists to remove.

### 3.5 Audit of every other conditional block

Nothing enumerated these before. This is the full list, in file order.

| Block | Guard | Verdict |
|---|---|---|
| argument loop | `case "$arg"` | Not a gate. An unknown argument exits 2. |
| role resolution | `if role_line=$(agent-role.py show)` | Reported. The failure arm prints `--` plus two lines of explanation, and appends `role-undeclared` when `REQUIRE_ROLE` is 1. Under `--no-require-role` the state is still printed, and the flag is a documented opt-out the caller chose. Unchanged. |
| read-only plus `--staged` | `[ "$STAGED" -eq 1 ] && grep role=read-only` | Adds a `FAIL`. Unchanged. |
| `--role-only` early exit | `[ "$ROLE_ONLY" -eq 1 ]` | Already honest, and the model for this fix. It names what did not run and refuses the banner. Unchanged. |
| per-checker dispatch | `case "$checker"` in the `CHECKERS` loop | Selects `--check` for two checkers. Every checker still runs. Not a skip. |
| `Committed range vs origin/main` | `git rev-parse --verify -q origin/main` and `rev-list --count -gt 0` | **Same shape, fixed here.** The unresolvable-ref arm drops three gates silently. |
| `Commit trailers vs origin/main` | the same, plus `--is-ancestor` | **The reported defect, fixed here.** |
| `Staged change` | `[ "$STAGED" -eq 1 ]` | A mode the caller selects, documented in the usage text at the top of the file. Running without `--staged` is not a skipped gate, it is a different run. Unchanged. |
| digest print | `[ "$QUIET" -eq 0 ]` | Not a gate. Unchanged. |

So two blocks share the shape, and both are repaired in this change.

### 3.6 An empty range is not a skip

`rev-list --count "$BASE_SHA..HEAD" -gt 0` guards both range blocks. Two
different situations hide behind one guard today, and they get different
verdicts:

| Situation | Verdict | Why |
|---|---|---|
| `origin/main` does not resolve | `SKIP` | The input is *unknown*. There are commits to check and the script cannot tell which. |
| `HEAD` adds no commits over the pinned base | reported as an empty range, not a skip | The input is *empty*. A gate over zero commits has nothing to report, and calling that a skip would print `SKIP` on the ordinary session-start run of a freshly cut branch. |

The empty-range case prints one line naming the pinned SHA, so the reader still
learns why no gate line followed the heading. It does not suppress the banner.

## 4. Risks and decisions

| Risk | Assessment |
|---|---|
| Weakening a checker to make something pass | Nothing is relaxed. No checker changes. The script reports strictly more than it did, and one previously reachable banner becomes unreachable. |
| `SKIP` becomes the new silence, ignored like a warning | It is loud by construction. It prints at the skip site with its reason, it prints again in the summary with a count, and it deletes the banner every reader greps for. A reader who greps `All gates green.` gets nothing. |
| The green banner never appears again on ordinary work | Measured against the three occurrences. After pinning, occurrences 2 and 3 print the banner, because the pinned SHA was an ancestor when captured. Only occurrence 1, a genuinely behind branch, reports `SKIP`, and that report is the correct one. |
| Exit 0 lets a behind branch land unchecked trailers | It does not, because CI runs `check-commit-trailers.py` on the pull request range independently of this script. Preflight is developer convenience, and `AGENTS.md` already says hooks and local runs are "bypassable convenience, not evidence". What was missing was an honest local report, not a second enforcement point. |
| Pinning hides a base that moved for a good reason | The pinned SHA is printed. A reader who wants the newer base fetches, merges, and reruns, which is the same repair the skip reason names. |
| The audit missed a block | §3.5 lists every conditional in the file, in file order, with a verdict for each. The suite asserts that both repaired blocks report, and the mutation cases in §5 fail if either regresses. |
| `check-commit-style.py` still refuses a non-ancestor base | Real, found during this work, and deliberately **not** fixed here. Removing the preflight guard would send `check-commit-style.py --range` a non-ancestor base, which raises at `validate_range`, so `commit-style` would go red on every branch behind `main`. That is the "gate that fires on ordinary work" defect, traded for this one. Repairing it means changing that checker's range semantics, which `AGENTS.md` says needs its own spec and its own red-before test. Listed under `## Owed`. |

## 5. Tests

`tests/scripts/test_agent_preflight_skip_report.py`, registered in the `SUITES`
array of `scripts/agent-preflight.sh` and in the record lane of
`.github/workflows/ci.yml`.

The suite executes the real script. It cannot run a nested full preflight,
because preflight runs this very suite and would recurse without bound. The
existing dodge for that problem is `--role-only`, which is too narrow here,
because the blocks under test are the last two in the file.

**The harness.** Copy `scripts/agent-preflight.sh` into a scratch git repository
and put a stub `python3` on `PATH` that exits 0. Every `run` then reports `ok`
without executing a checker, no suite re-enters this one, and the git-shaped
control flow under test runs for real against refs the test controls.

RED-first cases, each failing on the unmodified script:

1. `test_a_non_ancestor_base_reports_skip_and_no_green_banner` sets
   `refs/remotes/origin/main` to a divergent commit. **RED before:** the run
   prints `All gates green.`, exit 0, with two fewer `ok` lines than the
   ancestor run and no mention of the trailer block. **After:** the run prints
   `SKIP commit-trailers`, `SKIP commit-style`, a reason naming the pinned SHA,
   a `2 gate(s) SKIPPED` summary, and no banner.
2. `test_the_ok_count_falls_only_with_a_reported_skip` runs the same script
   twice, once with an ancestor base and once with a divergent base, and asserts
   that the difference in `ok` count equals the number of `SKIP` lines.
   **RED before:** the counts differ by two and there are zero `SKIP` lines,
   which is occurrence 3 exactly. This is the case that makes a *silent* drop
   impossible rather than merely unlikely.
3. `test_a_base_that_moves_mid_run_does_not_change_the_verdict` installs a stub
   `python3` that advances `refs/remotes/origin/main` to a divergent commit on
   its first call, which is what another worktree's fetch does. **RED before:**
   the trailer block vanishes and the run still says green. **After:** the run
   gates against the SHA pinned at the start, both trailer gates run, and the
   banner is earned.
4. `test_an_unresolvable_base_reports_skip_for_both_range_blocks` deletes
   `refs/remotes/origin/main`. **RED before:** five gates vanish and the run says
   green. **After:** five `SKIP` lines and no banner.
5. `test_an_empty_range_is_not_a_skip` puts `HEAD` exactly at the pinned base.
   Green both before and after, and it must stay green: it pins §3.6, so a fix
   that reports `SKIP` for an empty range fails here. Without this case the
   obvious over-correction passes every other case.
6. `test_an_ancestor_base_still_earns_the_banner` is the control. It must print
   `All gates green.` and exit 0, so a fix that suppresses the banner
   unconditionally cannot pass.
7. `test_the_run_names_the_base_it_gated_against` asserts the pinned SHA appears
   in the output.
8. `test_the_suite_is_registered` asserts this file's name appears in the
   `SUITES` array and in `.github/workflows/ci.yml`, because a suite wired into
   neither runs on no machine.

Mutation cases, each restoring the tree byte-for-byte afterwards:

- Delete the `skipped` check from the summary. Cases 1 and 4 must go red.
- Replace `"$BASE_SHA"` with `origin/main` in the trailer guard. Case 3 must go
  red.
- Make `skip` print `ok`. Cases 1, 2 and 4 must go red.
- Report `SKIP` for the empty range. Case 5 must go red.

## 6. Gates

- `python3 tests/scripts/test_agent_preflight_skip_report.py`, with cases 1
  through 4 shown RED on the unmodified script.
- `bash scripts/agent-preflight.sh`, run on this branch, with the real per-block
  counts recorded and the pinned SHA named. The run must not itself skip a
  block, which is the same claim this row is about.
- `python3 scripts/check-test-registration.py` and
  `python3 tests/scripts/test_check_gate_commands.py`, because the change edits
  the script both of them read.
- No CUDA, GPU, oracle, checkpoint or SACRED gate is implicated. The change
  touches one shell script and one test file, reaches no forward pass, and loads
  no weights.

## 7. Evidence

Recorded on completion: the red-before capture reproduced in a scratch
repository, the green-after capture with the skip reported, the full preflight
run with per-block counts and the pinned SHA, and the final SHA with a clean
`git status --porcelain`.

## 8. Stop conditions

- Stop if `All gates green.` can still print after any block was skipped. That
  is the whole row.
- Stop if the `ok` count can fall without a matching `SKIP` line. A silent drop
  is occurrence 3, and case 2 exists to make it impossible.
- Stop if an ordinary run on a branch level with `origin/main` stops printing
  the banner. Making the banner unreachable is not the fix.
- Stop if repairing the report requires changing what any checker demands.
- Stop if the empty-range case starts reporting `SKIP`, because that turns the
  most common session-start run into a false alarm.

## Owed

- [#999](https://github.com/mudler/vllm.cpp/issues/999): `scripts/check-commit-style.py`
  `validate_range` still refuses a base that is not an ancestor of head, so it
  never received the merge base repair that `GATE-FORK-ANCESTRY` (#773) applied
  to `scripts/check-commit-trailers.py`. Found while auditing this script's
  guards. Not fixed here, because it changes that checker's range semantics and
  needs its own spec and red-before test.
