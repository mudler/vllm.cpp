ID: ISSUE-GH-2407
Title: CI: agent-record fails on every PR because ci.yml still runs a test the issue-index retirement deleted
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2407
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `-`
>
> Owned under `.agents/specs/qwen4-exp-flash-next.md` `## Owed` (W6-CUDA-B found
> it; the fix rides in that wave's pull request, #2391).
>
> ## The defect
>
> `.github/workflows/ci.yml:189` runs
>
> ```
> python3 tests/scripts/test_check_issue_index_append_only.py
> ```
>
> That file does not exist. `7dc2ef1ea`
> ("feat(ENG-RECORD-CONFLICT-SURFACES): W6 -- GitHub is the issue index, and the
> record shape it could never deliver is retired") deleted it along with the rest
> of the append-only index machinery, and left the invocation behind.
>
> ## Consequence
>
> The `agent-record` job **fails on every pull request**, and it fails AFTER the
> gate it exists to run has already passed:
>
> ```
> record anchors: ok=915, stale=28, broken=5  -> rot 33
> agent record OK: ENGINE=173 MODEL=379 QUANT=85 KERNEL=58 BACKEND=87 ANCHOR-ROT=33
> python3: can't open file '.../tests/scripts/test_check_issue_index_append_only.py': [Errno 2] No such file or directory
> ##[error]Process completed with exit code 2.
> ```
>
> So the record gate is green and the job is red, which is the worst shape for a
> check: a reader who trusts the colour concludes the record is broken, and a
> reader who trusts the log concludes the job is noise. The other three scripts in
> that step and the one beside it all exist; this is the only stale line.
>
> ## Fix
>
> Delete the line. There is nothing to run in its place — the append-only index it
> tested is retired, and AGENTS.md now says "GitHub is the index. There is no
> index file."
>

## Resolution

GitHub records `state_reason: not_planned`, closed on 2026-08-31; this disposition makes no implementation claim.
