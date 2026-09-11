ID: ISSUE-GH-2390
Title: Roadmap stores stale issue state after index retirement
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: UNKNOWN
GitHub: 2390
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-RECORD-CONFLICT-SURFACES`
>
> `.agents/roadmap_v1.md` still stores a 98-row `## Open issues` table after the tracked issue index was retired. GitHub is the live issue authority, and `scripts/agent-issue-index.py --refresh` already renders an untracked snapshot for offline gates.
>
> The stored table is stale on current `origin/main`: only 8 of its 98 issue numbers remain open, and none of the 29 portfolio rows directly references a current open issue. The table also describes the retired append-only `merge=union` design as current behavior.
>
> This row must remove the stored issue projection, point readers to the existing generated snapshot, and preserve the ordered portfolio table. It must also reconcile stale lifecycle projections and close issues whose described code or record no longer exists.
>
> The campaign includes record gates discovered by the audit. `.github/workflows/ci.yml` still invokes the deleted append-only-index test. Issue #2372 combines two independent compile defects and must split by owning row before repair.
>
> Acceptance:
>
> - `.agents/roadmap_v1.md` stores no issue rows.
> - The roadmap names GitHub as the live issue authority and gives the refresh command.
> - The generated snapshot stays untracked and offline-first.
> - Ownership checks stay diff-scoped. No global unowned-issue count gate is added.
> - Duplicate claimable IDs and vague `PARTIAL` rows are reconciled without converting landed-but-open work to `DONE`.
> - Obsolete issues close with exact code, commit, or falsifying evidence.
> - The issue-index renderer tests, record gates, live-row audit, focused compile gates, and full preflight pass.
>
> One pull request carries the committed spec first and implementation commits after it.

## Resolution

-
