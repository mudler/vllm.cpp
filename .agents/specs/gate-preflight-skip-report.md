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
- An opt-in `--fail-on-skip` carries the third state into the exit status, for
  the callers that can read only the exit status. The default is unchanged.

**In scope.** The two range blocks in `scripts/agent-preflight.sh`, the new
`skip` reporting path, the summary, the `--fail-on-skip` flag and its one
production caller in `scripts/agent-ready.py`, and a new executable suite
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
| The one consumer that read the exit status as a verdict | `scripts/agent-ready.py` `run_local_preflight` |
| The consumer that inherits that verdict | `scripts/agent-integration.py` `run_ready` |
| The consumer that requires exit 0 over five skips | `scripts/check-test-registration.py` `_preflight_execution_errors` |
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

### 3.4 Exit code: 0 by default, 1 under `--fail-on-skip`, never the banner

The run exits 0 when a block was skipped and nothing failed. Three things in the
tree settle this.

1. `AGENTS.md` on gate design: "A gate that fires on ordinary work is the
   defect, not the discipline." A branch behind `main` is ordinary. Every helper
   worktree is behind `main` within minutes of being cut. A preflight that exits
   1 there would be red on most first runs of most sessions, and an agent would
   read that red as a broken record rather than as a missing merge.
2. `AGENTS.md` on unknown state: "Unknown is not absence or success". A `SKIP`
   line with a reason records the unknown. Exit 0 with a banner claims success.
   Exit 0 without a banner, with the unknown named, is the accurate report. The
   `audit-live-rows` comment objects to a skip *surviving the banner*, not to a
   skip existing, and after this change no skip survives it.
3. Nothing enforces anything on this exit status by itself. `AGENTS.md`: hooks
   and local runs are "bypassable convenience, not evidence". A hand-chained
   `scripts/agent-preflight.sh && git push` is exactly that convenience, and CI
   runs `check-commit-trailers.py` over the pull request range whatever this
   script returned.

Exit 1 keeps its meaning: a gate ran and failed. Widening exit 1 to cover "a
gate did not run" would merge two different facts into one signal, which is the
conflation this row exists to remove.

**`--role-only` is not a precedent here, and the first version of this section
said it was.** The reviewer of this row rejected that argument and was right.
`--role-only` is a narrowing the caller *asked for*, so its caller knows exactly
what will not run before the run starts. A `SKIP` is imposed on a caller who
asked for a full run and finds out afterwards, and only if it reads the report.
The two cases share what an honest partial run should *print*, which is where
`--role-only` really is the model. They share nothing about what an unrequested
partial run should *return*. The three arguments above stand without it.

### 3.4.1 Who reads the exit status

§3.5 enumerated every conditional in the script and never asked the other
question, which is who consumes the number the script returns. There are five
consumers, and exactly one of them treated that number as a verdict about the
tree.

The fifth was missing from the first version of this table, which listed four.
A census that stops when it has found the defect is not a census, and this row's
whole subject is a report that omits what it did not examine.

| Consumer | How it reads preflight | Verdict |
|---|---|---|
| `scripts/agent-ready.py` `run_local_preflight` | `returncode == 0`, nothing else | **The defect.** `AGENTS.md` names it the gate to run "before remote handoff". The `SKIP` lines, the `N gate(s) SKIPPED` summary and the missing banner are all invisible to it, so a behind branch reached `READY: local and live PR/CI evidence are green` with two gates never run, and an unresolvable base reached it with five. The word "green" printed over a trailer check that had not executed. Fixed by passing `--fail-on-skip`. |
| `scripts/agent-integration.py` `run_ready` | the exit status of `agent-ready.py` | Inherits the repair. It never calls preflight itself. |
| `scripts/check-test-registration.py` `_preflight_execution_errors` | requires `rc == 0` from an instrumented run | Reads the status as a fact about the *script's own execution*, never as a verdict about a tree. Its `git` shim fails every call, so `BASE_SHA` is empty and five gates skip on every run of it. This consumer is why the default cannot flip: making a skip exit 1 unconditionally turns `check-test-registration` red, and that checker is itself in `CHECKERS`. |
| `tests/scripts/test_agent_onboard.py` `:485` and `:500` | executes `scripts/agent-preflight.sh --role-only` and asserts on the exit status and the output | Unaffected, and named here to complete the census rather than because it is at risk. `--role-only` returns from the role block, above every record gate and both range blocks, so no `skip` can have been recorded by the time either call reads the status. Verified by reading the two call sites, not inferred from the flag's name. |
| a human at a shell, plus the `scripts/agent-preflight.sh passes` checkbox in `.github/pull_request_template.md` | reads the report | Unchanged, and the reason the default stays 0. |

So the flag is opt-in because the population splits cleanly: one machine
consumer that must not read a skip as success, one instrumented consumer that
must keep reading exit 0, and a human who reads the report either way. A default
that satisfied the first would break the second.

`--fail-on-skip` changes nothing else. It does not change what any gate demands,
what any block prints, or which blocks run. It fires only after the failure
summary, so a run that both failed and skipped still exits 1 with the failure
message, and it names itself in its own refusal so a reader is never left
guessing which flag turned an ordinary run red.

### 3.5 Audit of every other conditional block

Nothing enumerated these before. This is the full list, in file order.

| Block | Guard | Verdict |
|---|---|---|
| argument loop | `case "$arg"` | Not a gate. An unknown argument exits 2. |
| role resolution | `if role_line=$(agent-role.py show)` | Reported. The failure arm prints `--` plus two lines of explanation, and appends `role-undeclared` when `REQUIRE_ROLE` is 1. Under `--no-require-role` the state is still printed, and the flag is a documented opt-out the caller chose. Unchanged. Named here because `--` is now the only state that is neither `ok`, `FAIL` nor `SKIP`, and the only one `All gates green.` can print over: the check ran and reported, so it is not a skip, and the caller asked for the exemption, so it is not a failure. That verdict is accepted rather than overlooked, and a reader who greps for the three states should not conclude this fourth mark is a gap. |
| read-only plus `--staged` | `[ "$STAGED" -eq 1 ] && grep role=read-only` | Adds a `FAIL`. Unchanged. |
| `--role-only` early exit | `[ "$ROLE_ONLY" -eq 1 ]` | Already honest, and the model for this fix. It names what did not run and refuses the banner. Unchanged. |
| per-checker dispatch | `case "$checker"` in the `CHECKERS` loop | Selects `--check` for two checkers. Every checker still runs. Not a skip. |
| `Committed range vs origin/main` | `git rev-parse --verify -q origin/main` and `rev-list --count -gt 0` | **Same shape, fixed here.** The unresolvable-ref arm drops three gates silently here, and two more in the trailer block below, so an unresolvable base costs **five** gates in one run and not the three this row first recorded. Measured on the reviewer's independent harness and on case 4 of §5. |
| `Commit trailers vs origin/main` | the same, plus `--is-ancestor` | **The reported defect, fixed here.** |
| `Staged change` | `[ "$STAGED" -eq 1 ]` | A mode the caller selects, documented in the usage text at the top of the file. Running without `--staged` is not a skipped gate, it is a different run. Unchanged. |
| digest print | `[ "$QUIET" -eq 0 ]` | Not a gate. Unchanged. |
| the `--fail-on-skip` exit | `[ "${#skipped[@]}" -ne 0 ] && [ "$FAIL_ON_SKIP" -eq 1 ]` | Added here. Not a gate and it runs nothing. It converts a report the caller cannot read into an exit status the caller can, after the failure summary, so a failing run keeps its own message. See §3.4.1. |

So two blocks share the shape, and both are repaired in this change.

### 3.6 An empty range is not a skip

`rev-list --count "$BASE_SHA..HEAD" -gt 0` guards both range blocks. Two
different situations hide behind one guard today, and they get different
verdicts:

| Situation | Verdict | Why |
|---|---|---|
| `origin/main` does not resolve | `SKIP` | The input is *unknown*. There are commits to check and the script cannot tell which. |
| `git rev-list --count` fails with the base resolved | `SKIP` | Also *unknown*, and the first version of this change did not say so. See below. |
| `git rev-list --count` succeeds and prints something that is not a count | `SKIP` | Also *unknown*. The exit status is not evidence about stdout. See below. |
| `git merge-base --is-ancestor` exits above 1 | `SKIP`, with the ancestry reason and not the behind-the-base reason | Also *unknown*. The query failed, so the run has no verdict on ancestry to report. |
| `HEAD` adds no commits over the pinned base | reported as an empty range, not a skip | The input is *empty*. A gate over zero commits has nothing to report, and calling that a skip would print `SKIP` on the ordinary session-start run of a freshly cut branch. |

The empty-range case prints one line naming the pinned SHA, so the reader still
learns why no gate line followed the heading. It does not suppress the banner.

**Two guards filed an unknown under the empty-range exemption.** Both were found
in the fresh review of this row and both are repaired here.

`[ "$(git rev-list --count "${BASE_SHA}..HEAD" 2>/dev/null || echo 0)" -gt 0 ]`
maps a *failed* count onto the count `0`, which is the arm this section
deliberately exempts from `SKIP`. The two facts are opposites. An empty range
withholds nothing, and a count that could not be taken withholds everything.
`git checkout --orphan` reaches it with `origin/main` perfectly resolvable: the
base resolves, `HEAD` is unborn, `rev-list` exits 128, and three gates took the
exemption in silence while the banner printed. The count and its exit status are
now kept separately, and a non-zero status is its own `SKIP` arm.

`git merge-base --is-ancestor` answers 1 for "the base is not an ancestor" and
128 for "that question cannot be asked here". Both took the arm whose reason
reads "is not an ancestor of HEAD, so this branch is behind it", which names a
cause that is not the cause and sends the reader to `git merge` for a tree that
has no commit to merge into. The status above 1 now reports the failed query.

Both of those bugs failed in the honest direction. Neither produced a wrong
verdict about a tree. One filed an unknown as an exemption and the other filed
an unknown under the wrong name.

### 3.6.1 The third bug, which this row itself introduced

**An earlier version of this section claimed "Both bugs failed in the honest
direction", and by the time it was written that was already untrue.** The repair
for `|| echo 0` was spelled

```sh
RANGE_COUNT="$(git rev-list --count "${BASE_SHA}..HEAD" 2>&1)"
RANGE_STATUS=$?
```

and folding stderr into the *value* reintroduced the exact defect this row
exists to remove, through a narrower door and in the **dishonest** direction. It
arrived in `05f6dc04e` and the fresh review of this row found it.

`RANGE_STATUS` then catches only the case where git **fails**. Git can write to
stderr and still exit 0, and the value is the error text with the count after
it. `[ "$RANGE_COUNT" -gt 0 ]` does not evaluate false there. It **errors** with
status 2, bash prints `integer expression expected` on stderr, a `[` that errors
reads as false, and both range blocks fall through to their empty-range arm. Five
gates leave the report, nothing says so, and `All gates green.` prints with exit
0. That is occurrence 3 again, at five gates instead of two.

A `.git/objects/info/alternates` naming a path that does not exist reaches it
with no shim at all: git prints `error: unable to normalize alternate object
path: ...` on nearly every object-reading command, writes its ordinary answer to
stdout, and exits 0. Measured on the shipped script at `23d28243f`:

```
Committed range vs origin/main 5adc7de9e: empty, HEAD adds no commits.
Commit trailers vs origin/main 5adc7de9e: empty, HEAD adds no commits.

All gates green.        rc=0
```

72 `ok`, 0 `FAIL`, 0 `SKIP`, banner, exit 0. The pre-repair script at `ad8dcad8b`
reports 77 `ok` and runs all five gates on the identical scratch repository,
because it discarded stderr. So this is a regression and not an inherited
defect, and §7.9 records both runs.

**The repair is two halves, because they cover different failures.** Neither
alone is enough, and the mutations in §7.9 show each half failing on its own.

1. **Keep stderr out of the value**, with `2>/dev/null`. This is what makes the
   value correct in the reproduced case, where the count was perfectly readable
   on stdout all along. It is also what stops a git that merely *warns* from
   costing five gates a `SKIP` they do not deserve, which validation alone would
   do. The correct report for a noisy git with a readable count is the ordinary
   run, not a skip.
2. **Validate the value** against `^[0-9]+$`, in one `case` predicate that both
   range blocks read, so they cannot drift apart on what counts as unknown. This
   is what makes an arm *exist* for a count that is not a count. Without it any
   non-numeric value still reaches `-gt`, and its status-2 error is
   indistinguishable from "zero commits". Inferring from exit 0 that stdout is a
   decimal integer is the assumption that produced this defect, and the repair
   should not keep making it.

**`ANCESTRY_ERROR` merges stderr the same way and is correct as it stands.**
Checked rather than assumed. Under the same broken-alternates repository
`git merge-base --is-ancestor` also writes to stderr and exits 0. The difference
is that `ANCESTRY_ERROR` is a **message and never a value**: nothing compares it,
and every ancestry arm is selected by `ANCESTRY_STATUS` alone. The only read of
it is inside the `ANCESTRY_UNKNOWN` string, which a run prints only when the
query failed, and where the stderr text is the useful part. It is left merged
deliberately, and the script now says so beside the call.

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
`scripts/agent-ready.py` is copied in beside it, so the cases that exercise the
handoff gate run the real script against the same scratch refs.

**The stub records its argv**, appending each invocation to
`$VLLM_TEST_ARGV_LOG`. Without that the suite reads only the script's *report*,
and the report is silent about the SHA the five range checkers are handed.
Putting `--base origin/main` back on the three range gates, or
`--range origin/main..HEAD` back on the two trailer gates, then leaves every
other case in the file green while the heading names pinned SHA X and the
checkers judge whatever the moving ref points at, which is the exact defect this
row was filed for wearing the fix as a disguise. The script states the guarantee
in the comment above `BASE_REF` ("Every range block below compares against this
SHA and never against the ref"), and nothing detected its loss.
`scripts/check-test-registration.py` traces preflight the same way for the same
reason.

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
7. The pinned SHA has to appear in the output, asserted inside case 3 rather
   than in a case of its own.
8. `test_the_suite_runs_in_preflight_and_in_ci` asserts this file's name appears
   in the `SUITES` array and in `.github/workflows/ci.yml`, because a suite
   wired into neither runs on no machine.

Cases added by the fresh review of this row:

9. `test_every_range_gate_is_handed_the_pinned_sha_and_never_the_ref` reads the
   argv log after an ordinary run and asserts that each of the five gates that
   take a base was invoked exactly once with one, that the invocation carries
   the pinned SHA, and that it names no ref. It counts all five first, because
   a log that recorded nothing satisfies every `assertNotIn` in the case. Green
   before and after, like case 5: it is a regression guard, and its red-before
   is the pair of mutations below.
10. `test_an_unborn_head_reports_skip_rather_than_an_empty_range` runs
    `git checkout --orphan` with `origin/main` still resolvable. **RED before:**
    the three range gates take the empty-range exemption and print
    `empty, HEAD adds no commits`, and the trailer block skips with the reason
    `is not an ancestor of HEAD`, which is not the reason. **After:** five
    `SKIP` lines, no banner, and each reason names the git query that failed.
11. `test_the_flag_makes_a_skip_exit_1_and_the_default_still_exits_0` runs the
    same behind-base tree twice, once plain and once with `--fail-on-skip`, and
    asserts the reports are identical while the exit statuses differ. Both facts
    live in one case because each is the other's justification.
12. `test_the_flag_does_not_fire_on_a_run_that_skipped_nothing` is the control
    for the flag. A flag that reds an ordinary run gates nothing.
13. `test_a_skipped_preflight_stops_the_handoff_gate` runs the real
    `scripts/agent-ready.py` against a scratch repo whose base is divergent.
    **RED before:** preflight exits 0 over two skipped gates, `agent-ready`
    reads that as success and walks straight past it. **After:** it stops at the
    local preflight, relays the `SKIPPED` report to its own caller, and never
    reaches the remote question.
14. `test_an_unskipped_preflight_lets_the_handoff_gate_continue` is the control
    for case 13. The scratch repo has no `origin` remote, so a run that gets
    past the local preflight refuses with `REMOTE_UNVERIFIED` instead. That
    second refusal is the proof the first one did not fire.

Cases added by the second fresh review of this row, both in
`StderrIsNotTheValueTests` and both red on `23d28243f`:

15. `test_a_git_that_warns_and_exits_zero_still_runs_every_range_gate` breaks
    `.git/objects/info/alternates` so git writes to stderr, prints the count on
    stdout, and exits 0. It asserts a precondition first, and asserts all three
    parts of it: exit 0, stderr non-empty, stdout exactly `1`. Without that, a
    git that simply failed would satisfy the case through the arm
    `test_an_unborn_head_...` already covers. **RED before:** the five range
    gates report neither `ok` nor `SKIP`, `empty, HEAD adds no commits` prints
    over both blocks, and the banner prints. **After:** all five run, nothing is
    skipped, and the run keeps the banner it earned. It also compares the `ok`
    count against a control run of the same repository, which is the same
    invariant as case 2: a gate may leave the report only by saying that it did.
16. `test_a_count_that_is_not_a_number_reports_skip_rather_than_empty` puts a
    `git` shim on `PATH` that answers `rev-list` with a non-numeric line and
    exit 0, and forwards every other subcommand to the real program. Its
    precondition asserts both halves of that, so a shim that broke `rev-parse`
    could not pass as a shim that only broke the count. **RED before**, and red
    against a repair that only discards stderr, which is why both halves of
    §3.6.1 are in the change. No real git behaves this way. The case pins the
    *arm*, not a git.

Mutation cases, each restoring the tree byte-for-byte afterwards:

- Delete the `skipped` check from the summary. Cases 1 and 4 must go red.
- Replace `"$BASE_SHA"` with `origin/main` in the trailer guard. Case 3 must go
  red.
- Make `skip` print `ok`. Cases 1, 2 and 4 must go red.
- Report `SKIP` for the empty range. Case 5 must go red.
- Replace `--base "$BASE_SHA"` with `--base origin/main` at all three range
  gates. Case 9 must go red on all three.
- Replace `--range "${BASE_SHA}..HEAD"` with `--range "origin/main..HEAD"` at
  both trailer gates. Case 9 must go red on both.
- Restore `2>&1` on the count, which is the regression itself. Case 15 must go
  red and case 16 must stay green.
- Make the numeric predicate unmatchable, and separately delete the
  `RANGE_NUMERIC` term from both range arms. Case 16 must go red for each and
  case 15 must stay green.

The last two pairs are the evidence that the two halves of §3.6.1 are not
redundant. If either half covered the other, one of these mutations would leave
the suite green.

## 6. Gates

- `python3 tests/scripts/test_agent_preflight_skip_report.py`, with cases 1
  through 4 shown RED on the unmodified script.
- `bash scripts/agent-preflight.sh`, run on this branch, with the real per-block
  counts recorded and the pinned SHA named. The run must not itself skip a
  block, which is the same claim this row is about.
- `python3 scripts/check-test-registration.py` and
  `python3 tests/scripts/test_check_gate_commands.py`, because the change edits
  the script both of them read. `check-test-registration.py` is the sharper of
  the two here: it executes preflight under a `git` shim that fails every call
  and requires exit 0, so it fails if the default exit status for a skip ever
  flips.
- `python3 tests/scripts/test_agent_gates.py`, because the change edits
  `scripts/agent-ready.py`, which that suite loads.
- No CUDA, GPU, oracle, checkpoint or SACRED gate is implicated. The change
  touches one shell script and one test file, reaches no forward pass, and loads
  no weights.

## 7. Evidence

### 7.1 The defect reproduced on the unmodified script

Scratch repository, stub `python3`, one `HEAD` and two settings of
`refs/remotes/origin/main`. The only difference between the two runs is which
commit that ref names.

| `origin/main` | `ok` lines | `SKIP` lines | banner | exit |
|---|---:|---:|---|---:|
| an ancestor of `HEAD` | 76 | 0 | `All gates green.` | 0 |
| a divergent commit | 74 | 0 | `All gates green.` | 0 |

Two gates left the report and every observable in the output stayed the same.
This is occurrence 3, at the same counts that were seen live.

### 7.2 RED before

`python3 tests/scripts/test_agent_preflight_skip_report.py` on the unmodified
script: `Ran 8 tests`, `FAILED (failures=11)`. Five of the eight are red, and
the three controls in `TheBannerStaysReachableTests` that must stay green
already are. The count case reports the defect in one line:

```
AssertionError: 2 != 0 : 2 gate(s) disappeared from the report and 0 were
reported as skipped. Every gate that stops running has to say so.
```

### 7.3 GREEN after

Same suite on the repaired script: `Ran 8 tests`, `OK`. The first fresh review of
this row added six more cases, taking the suite to `Ran 14 tests`, `OK`. Their
own red-before is §7.7 and §7.8. The second fresh review added two more, so the
suite now reports `Ran 16 tests`, `OK`, and their red-before is §7.9.

The same scratch comparison, rerun against the repaired script. The `ok` totals
are one higher than in §7.1 because the new suite joined `SUITES`:

| `origin/main` | `ok` | `SKIP` | banner | exit |
|---|---:|---:|---|---:|
| an ancestor of `HEAD` | 77 | 0 | `All gates green.` | 0 |
| a divergent commit | 75 | 2 | none | 0 |

The drop of two is now fully accounted for by two `SKIP` lines, and the run
prints the reason and the pinned SHA:

```
2 gate(s) SKIPPED: commit-trailers commit-style
NOT a green preflight: a skipped gate reported nothing about this tree.
```

### 7.4 Mutations

Four mutations, each verified to have applied (`git diff --stat` changed) and to
still compile (`bash -n` returned 0), each restoring the file byte-for-byte
afterwards, confirmed by a sha256 match against the pre-mutation file.

| Mutation | Cases turned red |
|---|---|
| the banner ignores the `skipped` array | 2: the non-ancestor case and the unresolvable-base case |
| the trailer guard reads `origin/main` again instead of `$BASE_SHA` | 1: the mid-run move case |
| `skip()` prints `ok` | 4: both skip cases, the count case, and the exit-code case |
| the empty range reports `SKIP` | 1: the empty-range case |

### 7.5 The row's own preflight run

`bash scripts/agent-preflight.sh --quiet` on this branch after the review
repairs, gated against `origin/main 332aed738a40050fd3fe677bc12ac1bcdc8e4bb5`,
which both range block headings name.

| Block heading | `ok` | `FAIL` | `SKIP` |
|---|---:|---:|---:|
| `Session role:` | 1 | 0 | 0 |
| `Record gates:` | 25 | 1 | 0 |
| `Mutation suites:` | 44 | 1 | 0 |
| `Committed range vs origin/main 332aed738:` | 3 | 0 | 0 |
| `Commit trailers vs origin/main 332aed738:` | 2 | 0 | 0 |
| total | 75 | 2 | 0 |

`SKIP` is 0, so this run did not itself skip a block, which is the same claim
the row is about. The trailer block reports two `ok` rather than two `SKIP`
because the branch merged `origin/main` first, which is the repair the skip
reason names.

The two failures are `check-env-doc` and `test_check_env_doc`, and they are
**pre-existing on `main`**, not caused by this change.
`VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS` and
`VT_MOE_EXPERT_STREAM_SLOT_BYTES` arrived with `3005447f8` (#993) and are
neither documented nor allowlisted. Filed as
[#1000](https://github.com/mudler/vllm.cpp/issues/1000) and owned by
`ENG-EXPERT-STREAM`, and independently as
[#995](https://github.com/mudler/vllm.cpp/issues/995) against the same row by
another branch. The attribution here is measured rather than asserted:
`git diff origin/main...HEAD --stat` names six paths, none of them under `src/`
or `include/`, so this branch cannot have introduced an env var read from those
trees.

### 7.5.1 The three reds measured under `ENOSPC`

The first version of this section said that `test_cpu_x86_llamacpp_floor` failed
and that "two focused gates reported unrelated errors". It named one of the
three, quoted no `df`, no error text and no rerun output. That is thinner than
what this project asks of an attribution, and an unnamed gate cannot be checked
by a reader. The attribution stands. The evidence for it is below.

The three, named, with what each printed in that run:

| Gate | Symptom in the original `ENOSPC` run |
|---|---|
| `test_cpu_x86_llamacpp_floor` | failed. Load-dependent and pre-existing, [#618](https://github.com/mudler/vllm.cpp/issues/618) |
| `python3 scripts/check-test-registration.py` | `ERROR: CMake configure failed while proving required test registration`, then `ERROR: missing required test target test_device_selection in configured codemodel`, exit 1 |
| `python3 tests/scripts/test_check_test_registration.py` | 65 errors, exit 1 |

**Why an out-of-disk instrument accuses the code here.**
`scripts/check-test-registration.py` does its work in a temporary directory at
`:300`, `:489` and `:737`, each under `tempfile.TemporaryDirectory`, and
`_configure_cmake` at `:72-91` runs a real `cmake -S . -B <tempdir>` after
requesting the CMake File API codemodel, then reads the reply back out of that
directory. When the write fails there is no reply to read, the target list is
empty, and the checker reports precisely a **missing cmake target**: a verdict
about the tree, phrased in the vocabulary of the tree, produced by a full disk.

That mechanism is measured rather than argued. Deliberately reproduced on a
1 MiB `tmpfs` filled to zero bytes and pointed at by `TMPDIR`, on this tree, at
this SHA:

```
tmpfs  1.0M  1.0M  0  100%  .../scratchpad/fullfs
$ TMPDIR=<full> python3 scripts/check-test-registration.py          # rc=1
ERROR: CMake configure failed while proving required test registration
ERROR: missing required test target test_device_selection in configured codemodel
ERROR: vllm_cpp_add_test does not create an executable with its configured sources
$ TMPDIR=<full> python3 tests/scripts/test_check_test_registration.py  # rc=1
Ran 52 tests -- FAILED (failures=12)
```

The checker's verdict reproduces word for word. The suite's *count* does not,
because a 1 MiB `tmpfs` is milder pressure than the whole filesystem at zero:
some temporary directories still get created, so cases fail rather than error.
The count is therefore not the evidence and is not offered as such. The
mechanism is, and it reproduces on demand.

**Green on the identical tree with space available**, measured immediately
before this run, `TMPDIR` unset so both use `/tmp`:

```
$ df -h /tmp   ->  447G size, 66G avail, 85% used
$ python3 scripts/check-test-registration.py            # rc=0
OK: required regression tests have executable + CTest registration and the
guard is wired into preflight/CI.
$ python3 tests/scripts/test_check_test_registration.py # rc=0
Ran 52 tests -- OK
```

So a red measured under `ENOSPC` is not a result. Check free disk before
attributing any failure to the code, because a broken instrument does not report
as a broken instrument. It reports as a verdict about the code.

### 7.6 `agent-ready.py` refuses a skip while a human preflight does not

Both scripts copied into one scratch repository, one `HEAD`, `origin/main` set
to a divergent commit, stub `python3` on `PATH`. The only difference between the
two rows is which program asked.

| Caller | Report | Exit |
|---|---|---:|
| `bash scripts/agent-preflight.sh --quiet` | `2 gate(s) SKIPPED: commit-trailers commit-style`, no banner | 0 |
| `python3 scripts/agent-ready.py` | the same report, then `READY FAILED: local preflight did not report every gate green.` | 1 |

`agent-ready.py` never reached the remote question, which is the proof that it
stopped at the local one. On the unrepaired pair it walked straight past the
skip and refused with `REMOTE_UNVERIFIED: error: No such remote 'origin'`
instead, which is case 13's red-before in one line.

### 7.7 Red-before for the review repairs

The new cases run against the tree as it was before this repair, restored with
`git show HEAD:scripts/agent-preflight.sh` and
`git show HEAD:scripts/agent-ready.py` while the new suite stayed in place:
`Ran 14 tests`, `FAILED (failures=7)`.

| Case | Red-before |
|---|---|
| 10, unborn `HEAD` | the three range gates never reported, and `empty, HEAD adds no commits` was printed over them |
| 10, the ancestry reason | `is not an ancestor of HEAD` printed for a query that failed with `fatal: Not a valid object name HEAD` |
| 11 and 12, `--fail-on-skip` | `unknown argument`, exit 2, and the harness precondition caught it as a run that reported nothing |
| 13, the handoff gate | `'READY FAILED: local preflight' not found in "REMOTE_UNVERIFIED: error: No such remote 'origin'"` |

Cases 9 and 14 are green before and after by design. Case 9 is a regression
guard whose red-before is §7.8, and case 14 is the control that proves case 13
is not a script that refuses everything.

### 7.8 The two mutations the suite could not see before

Both are the reviewer's, applied to the repaired script, each verified to have
applied (`git diff --stat` changed) and to still parse (`bash -n` returned 0),
each restored byte-for-byte afterwards and confirmed by a sha256 match.

| Mutation | Sites | Result |
|---|---:|---|
| `--base "$BASE_SHA"` becomes `--base origin/main` | 3 | case 9 red on all three range gates, `FAILED (failures=3)` |
| `--range "${BASE_SHA}..HEAD"` becomes `--range "origin/main..HEAD"` | 2 | case 9 red on both trailer gates, `FAILED (failures=2)` |

Before the argv log, both mutations left the suite at `Ran 8 tests`, `OK`.

Restored: `sha256sum -c` reports `OK` for all three files, and the suite returns
to `Ran 14 tests`, `OK`.

### 7.9 The regression this row introduced, and its repair

**The defect on the shipped script**, `23d28243f`, reproduced end to end before
anything was edited. One scratch repository, stub `python3` on `PATH`,
`origin/main` an ancestor, `HEAD` one commit ahead, and a
`.git/objects/info/alternates` naming a path that does not exist. The only
difference between the rows is which script ran against it:

| Script | `ok` | `FAIL` | `SKIP` | the five range gates | banner | exit |
|---|---:|---:|---:|---|---|---:|
| `ad8dcad8b`, before the `2>&1` | 77 | 0 | 0 | all five `ok` | `All gates green.` | 0 |
| `23d28243f`, as shipped | 72 | 0 | 0 | **absent from the report** | `All gates green.` | 0 |
| this repair | 77 | 0 | 0 | all five `ok` | `All gates green.` | 0 |

The shipped script also wrote two lines to stderr that no reader of the report
sees:

```
agent-preflight.sh: line 320: [: error: unable to normalize alternate object path: ...
1: integer expression expected
```

`git log -S` places the `2>&1` in `05f6dc04e` and nowhere earlier, so the
regression is this row's own and not something it inherited.

**RED before.** The two new cases against the script restored to `23d28243f`
with the new suite in place, the restore verified by `git diff --quiet` and the
file verified to parse with `bash -n`:

```
Ran 16 tests -- FAILED (failures=12)
```

The 12 are the two new cases and their ten `subTest` gates. The other 14 cases
were green, so the red is scoped to what this repair is about:

```
AssertionError: False is not true : commit-trailers did not run although the
count was readable on stdout:
AssertionError: 'empty, HEAD adds no commits' unexpectedly found in ... :
HEAD adds a commit and the run reported an empty range:
AssertionError: True is not false : five unknown gates printed green:
```

**GREEN after.** `Ran 16 tests`, `OK`.

One correct report was red on the first green-after run, and the message was
repaired rather than the assertion. Rewrapping `RANGE_UNKNOWN` had split the
sentence `Unknown is not an empty range.` across a newline, which cases 10 and
16 assert verbatim. The behaviour was right and the prose was wrong. The line is
now kept unwrapped and the script says why beside it.

**Mutations**, each verified to have applied by a diff against the pre-mutation
file rather than against `HEAD`, each verified to parse, each restored
byte-for-byte and confirmed by `sha256sum -c`:

| Mutation | Sites | Result |
|---|---:|---|
| `2>/dev/null` becomes `2>&1`, the regression itself | 1 | case 15 red, `FAILED (failures=6)`. Case 16 stays green |
| the numeric `case` pattern is made unmatchable | 1 | case 16 red, `FAILED (failures=6)`. Case 15 stays green |
| the `RANGE_NUMERIC` term is deleted from both range arms | 2 | case 16 red, `FAILED (failures=6)`. Case 15 stays green |

The last two rows are why the repair has two halves. Each half has a mutation
that only the other half's case detects, so neither is covering for the other.

A first attempt at the second mutation was written with `sed` and a delimiter
that collided with the `|` in the pattern. `sed` exited non-zero, nothing was
edited, and the suite reported `Ran 16 tests`, `OK`. That `OK` was a mutation
that never applied wearing a passing test. The harness was rewritten to diff
against the saved pre-mutation copy and to refuse to report a suite result at
all when the pattern matched zero sites. Every result above ran under that
harness.

### 7.10 Two corrections to the merge commit body

`5b95e221e` is a merge and its body is not amendable, so the corrections live
here. The squash body will be the landed record.

- The body says main brought "#986, #995 and #987". It brought **#986 and
  #995**. `#987` was already in the merge base: `3ce1cf7c7`, this row's own
  base, is `feat(LTX25-RETAKE) ... (#924, #987) (#992)`. Diffing
  `.agents/issue-index.md` at the merge base against each parent shows main's
  side adding exactly two rows.
- The body says "this branch's #998 and #999". There are **three**: #998, #999
  and #1000. The same diff shows all three on the branch side, and all three are
  on `HEAD` at lines 272 through 274.

Both are miscounts in a description of a union merge that was itself correct.
The merged file is unchanged by this correction.

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
