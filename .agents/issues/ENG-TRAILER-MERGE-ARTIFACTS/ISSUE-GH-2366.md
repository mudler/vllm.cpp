ID: ISSUE-GH-2366
Title: preflight skips the trailer gates on any branch behind main, forcing a merge instead of a rebase
Row: ENG-TRAILER-MERGE-ARTIFACTS
State: CLOSED
Kind: UNKNOWN
GitHub: 2366
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-30
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `scripts/agent-preflight.sh` runs `commit-trailers` and `commit-style` only when `origin/main` is an ancestor of HEAD. Any task branch whose base is behind main — the ordinary state mid-campaign — gets both gates SKIPPED with the message "Merge origin/main and rerun", and `agent-ready.py` turns that skip into a hard failure through `--fail-on-skip`. The only route to green is merging main into the task branch, which bakes a merge commit into every PR and punishes rebasing an unpushed branch.
>
> ## Root cause
>
> The SKIP is a compensation, and the code says so: `scripts/agent-preflight.sh` drops the gates because `scripts/check-commit-style.py` refuses a non-ancestor base at `validate_range` (`range base must be an ancestor of range head`, check-commit-style.py:125-126). The trailers checker repaired the same defect already — #773/#2157 walk from the merge base, fail-closed on unrelated histories (check-commit-trailers.py:402-407, `pull_request.base.sha` stops being an ancestor as soon as main advances). The style checker never got that port. The preflight comment cites "#999" as owing the repair; **#999 does not exist on this forge** — a dangling anchor, corrected by this issue.
>
> ## Fix
>
> 1. Port the #773 merge-base walk to `check-commit-style.py::validate_range`: `base_oid = merge_base(base, head)`, unrelated histories still fail closed, cutover logic unchanged.
> 2. Drop the ancestry SKIP arm for status 1 in `agent-preflight.sh` (keep the unresolved-base, ancestry-error, and range-count-unknown arms). The gates then judge `${BASE_SHA}..HEAD`, whose commit set is exactly the branch's own commits under any ancestry.
> 3. Update the verbatim-pinned suite(s) with the red-first test: a behind-base range runs both gates instead of skipping.
>
> ## Verified how
>
> - A branch based on an older main (base not an ancestor, own commits present) runs both trailer gates over its own commits and reaches green WITHOUT merging.
> - A rebased branch passes identically.
> - Unrelated histories still refuse (fail closed).
> - Mutation: re-adding the ancestor refusal in the style checker reds the new test.
>
> Owning row: `CHECKER-STYLE-MERGE-BASE`.

## Resolution

GitHub records closing pull request #2370 (https://github.com/mudler/vllm.cpp/pull/2370) merged on 2026-08-31 as commit `5f911cbec3d81cfdf9c8c7fec274f8c837306f60`. GitHub closed issue #2366 on 2026-08-31.
