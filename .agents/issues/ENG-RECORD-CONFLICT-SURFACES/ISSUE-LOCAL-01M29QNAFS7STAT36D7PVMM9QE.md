ID: ISSUE-LOCAL-01M29QNAFS7STAT36D7PVMM9QE
Title: agent-issue.py silently discards any issue section outside its fixed schema
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

scripts/agent-issue.py renders an issue file from a fixed IssueRecord schema, so every update and close DELETES any section the schema does not name. Evidence is lost with no warning and no diff shown to the caller.

Mechanism, read in the tree at 97cb6964b:

- IssueRecord carries exactly id, title, row, state, kind, github, mirror, availability, created, updated, closed, problem, resolution (scripts/agent-issue.py:120-140 area).
- _write (:654-674) calls filesystem.atomic_write(path, render_issue_record(record)). The rendered text is generated from those fields ALONE; the file on disk is never read forward.
- So a section such as '## Reconciliation', '## Evidence' or any other heading an agent or a human appended is not represented in the record, is not re-rendered, and disappears on the next update or close.

Observed, not theorised. Closing the local issue on row QUANT-GGUF-IQ4_NL (PR #3149; that record is named in the pull request and is deliberately NOT cited by ID here, because it lives on that branch and a reference to it would not resolve on this one) dropped TWO '## Reconciliation' sections that recorded why the row's scope changed twice: that #3097 had landed the ROCm gather, and that the CUDA arm was never missing because #2419 had already landed it. Both were dated, both cited commits and file:line anchors, and both were the only record of a scope correction. They were restored by hand; nothing in the tool's output said they had gone.

Why this matters more than an ordinary bug. AGENTS.md 'Records' says to move superseded detail into .agents/completed/ and 'Never delete evidence to reduce context'. A tool whose normal success path deletes evidence inverts that rule, and it does so in the one file class the protocol treats as canonical. The failure is also invisible by construction: the caller sees a success, and the loss is only detectable by diffing the file before and after, which nobody does because the tool is the sanctioned way to edit it.

Not filed as a same-flow fix, because the repair is a design question rather than a typo: the tool must decide whether unknown sections are preserved verbatim and where they are re-emitted relative to Problem and Resolution, whether ordering is stable across a round trip, and whether a lossy write should refuse rather than warn. That wants a spec and a red-before test that round-trips a file carrying an unknown section.

## Resolution

-
