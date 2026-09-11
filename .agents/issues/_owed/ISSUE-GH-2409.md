ID: ISSUE-GH-2409
Title: CI: agent-record job reds on every PR by invoking a test file 7dc2ef1ea deleted
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2409
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-RECORD-CONFLICT-SURFACES`
>
> `.github/workflows/ci.yml:189` runs a test file that does not exist, so the
> `agent-record` job exits 2 on every pull request in the repository:
>
> ```text
> python3: can't open file '/home/runner/work/vllm.cpp/vllm.cpp/tests/scripts/test_check_issue_index_append_only.py': [Errno 2] No such file or directory
> ##[error]Process completed with exit code 2.
> ```
>
> The job's own gates pass first (`record anchors: ok=915, stale=28, broken=5 ->
> rot 33`, then `Ran 113 tests ... OK`). The red is infrastructure rot, not a
> record failure, and it reads as a record failure to every author who sees it.
>
> `7dc2ef1ea` ("feat(ENG-RECORD-CONFLICT-SURFACES): W6 -- GitHub is the issue
> index, and the record shape it could never deliver is retired") deliberately
> deleted the append-only gate and its suite:
>
> - `scripts/check-issue-index-append-only.py`
> - `tests/scripts/test_check_issue_index_append_only.py`
>
> It moved `.agents/issue-index.md` to `.agents/completed/issue-index.md` and it
> added the replacement, `scripts/agent-issue-index.py` with
> `tests/scripts/test_agent_issue_index.py`. It did not remove the workflow line
> that invokes the deleted suite.
>
> The gate was retired on purpose, so the correct repair is to drop the orphaned
> invocation, not to restore the deleted file. `scripts/agent-preflight.sh:494`
> also still names `check-issue-index-append-only` in a historical comment without
> saying it is gone.
>
> `.github/workflows/ci.yml:478` invokes `tests/scripts/test_agent_issue_index.py`,
> which exists; that line is correct and stays.
>

## Resolution

GitHub links pull request #2414 as closing issue #2409 on 2026-08-31.
