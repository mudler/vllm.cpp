ID: ISSUE-GH-2369
Title: check-agent-record resolves record links against the worktree, so an untracked target gates green locally and red in CI
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: UNKNOWN
GitHub: 2369
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-30
Updated: 2026-08-30
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Observed
>
> PR #2368, head 11f2826eaa, the CI step "Canonical roadmap tables and links are consistent" (`scripts/check-agent-record.py`) failed with the only error in its run:
>
> ```
> ERROR: .agents/benchmark-record.md: dangling link ../docs/bench-evidence/tt-p150-w2-confirmed-20260830.log
> ```
>
> The same gate had passed in the developer worktree immediately before the push.
>
> ## Root cause
>
> `scripts/check-agent-record.py:1120` resolves every record-link candidate with `Path.exists()` against the checkout on disk. `.gitignore` line 34 (`*.log`) kept the W2 evidence log out of the records commit, and the on-disk copy satisfied the gate locally. CI checks out tracked content only, so the link dangles there. The worktree and the CI checkout disagree, and the gate reads the one that hides the defect.
>
> Repro: a detached worktree of the same commit reproduces the single error above; the developer worktree with the untracked file on disk passes.
>
> ## Repair in the W2 flow (#2005)
>
> Force-add the log, the same path the two earlier tt-p150 evidence logs took (`tt-p150-clock-attributed-20260826.log`, `tt-p150-refresh-20260826.log` ride the same ignore rule). No checker change rides that repair.
>
> ## Owed
>
> The checker-semantics fix is its own row: resolve link targets against tracked content (e.g. `git ls-files`) or refuse a link whose target is present but untracked. Checker semantics need a spec, a red-first test — an ignored-but-present target must red, the current suite only covers tracked and absent targets — and fresh review, so it does not ride the evidence repair. Test surface: `tests/scripts/test_agent_record.py`.
>
> Owning row candidate: `CHECKER-AGENT-RECORD-TRACKED-LINKS`.
>
> FOLLOWING_AGENTS_PROTOCOL
>
> Following-Agents-Protocol: true
> AI-Assisted: true
> Assisted-by: AGENT:zai-glm-5.3-flash [maki]

## Resolution

-
