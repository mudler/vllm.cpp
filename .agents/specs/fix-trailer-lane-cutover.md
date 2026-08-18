# FIX-TRAILER-LANE-CUTOVER: excuse one landed message by name, not a prefix of history

**Issue:** [#1262](https://github.com/mudler/vllm.cpp/issues/1262).
**Kind:** checker semantic change. `scripts/check-commit-trailers.py` learns one
narrow exception. No product source is touched and no forward pass is reached.
**Row:** `FIX-TRAILER-LANE-CUTOVER`.

## Now

`281b4bc76c0e635adbc7ed38317035b07c99864d` is on `main` and its message ends

```text
Assisted-by: AGENT:claude-opus-5 CLI
```

The grammar is `AGENT:MODEL [TOOL]`. The bracketed tool is missing. Measured
with the checker itself, from a worktree pinned at `origin/main` `27d5432f9`:

| Command | Result |
|---|---|
| `check-commit-trailers.py --range '281b4bc76~1..origin/main'` | `rc=1`, one offender: `281b4bc76c0e: [attribution] malformed Assisted-by value 'AGENT:claude-opus-5 CLI'` |
| `check-commit-trailers.py --range '281b4bc76..origin/main'` | `rc=0`, `OK: commit trailer contract` |

So the violation is real, and it is exactly one commit carrying exactly one
error.

**It cannot be repaired.** Correcting a landed message rewrites `main`, and
`AGENTS.md` forbids a force push of `main` without exception.

**It does not clear itself.** The main lane at
[`.github/workflows/ci.yml:596-623`](../../.github/workflows/ci.yml) walks
`LAST_GREEN..head`, and `LAST_GREEN` is the last **successfully** gated commit.
A red run never advances it, so every later push re-walks a range that still
contains `281b4bc76` and reports the same red. `ci.yml:74` states that design
deliberately: a cancelled run is lossless because "the next run walks a wider
range and reports the same red". The property that makes a cancellation harmless
is the property that makes an unrepairable red permanent.

## How it got in

The four branch commits were verified and were correct. The repository sets
`squash_merge_commit_message = PR_BODY`, so the landed message came from the
pull request body, which still held the pre-repair value. The guard that reads
the body is `ci.yml:626-635` in `commit-protocol-tag`: it writes `$PR_BODY` to a
file and runs this same checker with `--filled`, so body and commit are held to
one rule by one implementation (#848). At merge time that job was `pending`
because the runner pool was saturated. **It did not fail. It never ran.**

## Scope

In scope:

- One narrow, enumerated exception in `scripts/check-commit-trailers.py`, keyed
  on the full commit oid **and** the exact rendered error string.
- Executable scope proof in `tests/scripts/test_check_commit_trailers.py`.
- The [#1262](https://github.com/mudler/vllm.cpp/issues/1262) row in
  [`issue-index.md`](../issue-index.md), which the issue currently lacks.

Out of scope, deliberately:

- The `Assisted-by` grammar. `ASSISTED_BY` is not touched. Nothing about a
  missing bracket becomes acceptable anywhere.
- `--cutover`. The flag stays exactly as it is and gains no new caller. See
  below for why it is the wrong instrument here rather than an unused one.
- `.github/workflows/ci.yml`. The lane needs no new flag, which is one of the
  reasons this shape was chosen.
- The operator-side pre-merge body check. See `## Owed`.

## Why `--cutover` is the wrong instrument

`--cutover` exists and looks like the escape hatch. It is not one, measured
rather than assumed.

**It does not excuse the commit you name.** `validate_range` asks
`_is_ancestor(cutover, commit)` first, and `git merge-base --is-ancestor X X`
succeeds, so the cutover commit itself is checked **strictly**. Measured:
`--range '281b4bc76~1..origin/main' --cutover 281b4bc76` still fails with the
same one error. Excusing `281b4bc76` requires naming its child
`1dac4f9a70195b282d16c536f319e8b171c925f8` as the cutover, which is already a
counter-intuitive record for a reader to interpret.

**It excuses a prefix of history, not a commit.** With that child as the
cutover, every one of its **2986 ancestors** (`git rev-list --count`) drops to
`strict=False`, which returns no errors at all as long as one
`FOLLOWING_AGENTS_PROTOCOL` paragraph is present. A missing `AI-Assisted`, a
forbidden `Signed-off-by: Claude`, a doubled trailer block: all waived, in
2986 commits nobody re-read. That fails the requirement this row is judged on,
that a **different** violation in the excused commit's own range must still be
caught.

**It is a value that can be moved.** The next unrepairable red on `main` is
"fixed" by advancing one sha. That is the precise failure mode `AGENTS.md`
§"Changing the rules or a checker" exists to prevent, and a reviewer reading a
one-line sha change cannot see what it newly covers.

**No caller passes it on the failing lane.** `ci.yml` invokes the checker with
`--range` alone at `:620` and `:623`; only `scripts/agent-integration.py:107`
passes `--cutover`, from `.agents/policy-cutover`, **a file that does not exist
in the tree**. Reaching for `--cutover` therefore means adding a blanket to the
main lane that has never been there, not using one that is already wired.

## Design

One module-level constant in `scripts/check-commit-trailers.py`:

```python
class LandedException(NamedTuple):
    error: str
    reason: str

LANDED_MESSAGE_EXCEPTIONS: dict[str, LandedException] = {
    "281b4bc76c0e635adbc7ed38317035b07c99864d": LandedException(...),
}
```

`validate_range` computes each commit's errors exactly as before, then drops an
error only when the walked commit's **full oid** is a key and the **rendered
error string is equal** to the registered one. Every other error, in that commit
or any other, is reported unchanged.

Four properties follow, and each is a test:

1. **A different violation in the excused commit still fails.** The match is on
   the exact error text, so a second error in the same commit is reported.
2. **A later violation still fails.** The match is on the oid, so no other
   commit consults the entry.
3. **A new message can never be excused.** The lookup lives in `validate_range`
   only. `validate_commit_message` and the `--message-file` path, which is what
   gates a pull request body before it lands, do not know the constant exists.
   A body carrying this exact malformed value today is still rejected.
4. **It cannot silently widen.** A commit oid covers its message. Registering a
   sha is registering one immutable byte string, so the entry cannot grow to
   cover a message somebody writes later.

The excused error is **printed on every run that applies it**, on a passing run
as well as a failing one, naming the commit, the error, and the reason. An
exception is visible debt, not success, and a reader of the green lane sees the
offender.

## Risks

| Risk | Mitigation |
|---|---|
| The registry grows into a waiver list | `test_the_registry_holds_exactly_the_one_landed_commit` pins the count at 1 and pins the key. Adding an entry is a deliberate, reviewable edit to a test that says why the number is 1. |
| An entry outlives its defect | `test_every_registered_exception_is_live` re-derives each entry against the real commit and fails if the named error is not the error that commit actually produces. |
| The print is mistaken for noise | It names the issue and the word "debt" and is emitted before the verdict line. |
| A shallow clone cannot see the commit | The live-entry test skips with a message naming shallowness as the reason. The registry-shape test never skips, so the count guard holds in every environment. |

## Tests

All in `tests/scripts/test_check_commit_trailers.py`, class
`LandedMessageExceptions`, before the `__main__` guard.

| Case | Proves |
|---|---|
| `test_the_registered_exception_clears_the_real_landed_red` | Green-after on the real commit, and that the applied exception is printed. |
| `test_a_different_violation_in_the_excused_commit_still_fails` | Requirement 2, first half. |
| `test_a_later_violation_is_still_caught` | Requirement 2, second half. |
| `test_the_exception_is_keyed_on_the_exact_error_text` | A near-miss error string excuses nothing. |
| `test_an_unlisted_commit_is_never_excused` | The oid key is load-bearing. |
| `test_a_pull_request_body_is_never_excused` | The `--message-file` path cannot be excused, so this cannot become a way to land a new malformed body. |
| `test_the_registry_holds_exactly_the_one_landed_commit` | Nothing else is excused. |
| `test_every_registered_exception_is_live` | No dead or overbroad entry. |

`scripts/check-pr-size.py` re-runs this file against the **base** checker and
requires it to fail, so the red-before is gated rather than asserted.

## Gates

| Gate | Expected |
|---|---|
| `python3 -m unittest tests.scripts.test_check_commit_trailers` | all cases green |
| `check-commit-trailers.py --range '281b4bc76~1..origin/main'` | `rc=0` **and** the applied exception printed |
| `check-commit-trailers.py --range '281b4bc76..origin/main'` | `rc=0`, unchanged |
| `scripts/agent-preflight.sh --fail-on-skip` | all gates green |

## Stop conditions

- Stop and return `NEEDS_DECISION` if closing the lane needs a second entry.
  One landed defect is a fact; two is a pattern and a different row.
- Stop if the `Assisted-by` grammar has to move. It does not, and if it did the
  answer would be to keep the red.

## Owed

The operator-side pre-merge check is **not** in this row and needs its own:
piping `gh pr view --json body` through `check-commit-trailers.py
--message-file - --filled` before a squash merge. The CI guard at
`ci.yml:626-635` is the right check and it can be outrun by a queued runner,
which is exactly what happened here, so a local belt to that brace is worth
having. It is separated because it changes an operator procedure and a gate
command rather than a checker rule, so it carries its own red-first evidence and
its own reviewer, and because it is not needed to clear the lane. Filed as
[#1263](https://github.com/mudler/vllm.cpp/issues/1263) and owed here.

## Outcome

Filled when the row reaches `DONE`.
