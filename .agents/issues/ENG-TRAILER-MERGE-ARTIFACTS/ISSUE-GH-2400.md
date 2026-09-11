ID: ISSUE-GH-2400
Title: check-commit-style reports an unevaluable range as a style failure, and races a moving main
Row: ENG-TRAILER-MERGE-ARTIFACTS
State: OPEN
Kind: UNKNOWN
GitHub: 2400
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
> `scripts/check-commit-style.py` reports a "could not run" condition as a style
> verdict, and the two commit gates disagree about how to handle a moving base.
>
> ## Reproduction
>
> On one range whose base is not an ancestor of its head:
>
> ```
> $ B=$(git rev-parse origin/main~5); H=$(git rev-parse origin/main~20)
> $ python3 scripts/check-commit-style.py --range "$B..$H"
> commit style check FAILED: range base must be an ancestor of range head   # rc=1
> $ python3 scripts/check-commit-trailers.py --range "$B..$H"
> OK: commit trailer contract                                               # rc=0
> ```
>
> `check-commit-trailers.py` computes the merge base itself (#773) and is robust
> to a base that has moved. `check-commit-style.py` refuses outright.
>
> ## Why it matters
>
> The message says `FAILED` and the exit code is 1, which is the same shape as a
> real style violation. Nothing distinguishes "this range violates the style rule"
> from "I could not evaluate this range at all". A reader chases a violation that
> does not exist; that happened in a session on 2026-08-31 before the cause was
> identified.
>
> The refusal is a race, not a property of the change. `origin/main` moved 36
> commits during a single session on 2026-08-31, so whether this gate evaluates a
> branch at all depends on when it is invoked relative to unrelated merges. A gate
> whose ability to run is timing-dependent does not gate.
>
> The two failure directions are not equal. `commit-trailers` resolving its own
> base means it evaluates the intended commits. `commit-style` refusing means the
> style rule is unenforced for that invocation while appearing to have been
> enforced and failed.
>
> ## Suggested fix
>
> Give `check-commit-style.py` the same base resolution `check-commit-trailers.py`
> already uses, so both gates evaluate the same commit set from the same inputs. If
> a range is genuinely unevaluable, exit with a status distinct from a style
> violation so the two cannot be confused.
>
> A red-before test belongs on the non-ancestor range above: it must stop reporting
> `FAILED` for a range it did not evaluate.
>

## Resolution

-
