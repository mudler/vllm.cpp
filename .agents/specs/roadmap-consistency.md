# Roadmap consistency and local issue authority

Issue: [#2390](https://github.com/mudler/vllm.cpp/issues/2390)

Row: `ENG-RECORD-CONFLICT-SURFACES`

Base: `origin/main` `5a6cafa87914c2b5d8baa01e128ed0b38e552935`

## Scope

This row makes the roadmap, matrices, row specs, and issue records one consistent system.

On 31 August 2026, the developer superseded the GitHub-authoritative snapshot design. Local tracked issue files are canonical. GitHub is an optional mirror.

In scope:

- Keep the accepted removal of the stored `## Open issues` table from `.agents/roadmap_v1.md`.
- Keep the ordered 29-block portfolio table in `.agents/roadmap_v1.md`.
- Keep the accepted repair for the deleted issue-index test invocation.
- Keep the reviewed Task 12 feature projections and Task 13 partial-gap descriptions.
- Specify one tracked file for each issue under `.agents/issues/`.
- Specify stable local and GitHub-derived issue IDs.
- Specify local-first create, update, mirror, import, and close operations.
- Make `scripts/agent-issue-index.py --refresh` derive its view from local files without a network call.
- Add offline gates for local issue integrity and diff-scoped branch references.
- Migrate the union of frozen archive entries and currently reachable GitHub issues into local canonical records.
- Preserve archived ownerless residue in the migration-only `.agents/issues/_intake/` directory.
- Preserve existing unique spec ownership before using `_intake`.
- Reconcile ambiguous ownership before using `_intake`.
- Preserve issue comments only when they contain binding decisions or evidence.
- Reconcile issue and row lifecycle state only when local evidence agrees.

Out of scope for this Task 32 specification correction:

- Implementing or changing `.agents/issues/`, issue tools, migration code, or tests.
- Retrying the Task 26 migration and cutover.
- Restoring issue rows to `.agents/roadmap_v1.md`.
- Making GitHub available to an offline gate.
- Making GitHub the authority when a local file differs.
- Adding a global count ratchet for `_intake` or `_owed`.
- Automatically mirroring comments.
- Product fixes for models, kernels, backends, or serving paths.
- Performance, GPU, model, or oracle measurements.
- Reapplying the unreviewed quantization change that was rolled back.

## Decision record

The developer approved the local-authority correction on 31 August 2026.

The prior design made GitHub authoritative and used an untracked snapshot for offline checks. That design did not provide a locally reviewable canonical record. It also made refresh behavior depend on remote availability.

The corrected design uses tracked, one-file-per-issue records. GitHub mirrors those records when an operator requests a mirror operation. A GitHub failure cannot remove or invalidate a completed local write.

The first Task 26 migration attempt failed review. It placed all 594 archived `Row: -` records under this specification's `## Owed` section. That design recreated the shared ownership lock that the record cutover must remove.

Task 32 corrects migration ownership before Task 26 retries. This decision does not reverse the accepted roadmap cleanup. The roadmap remains free of embedded issue rows and bulk ownership lists.

## Upstream chain

No vLLM behavior defines this record system. This row is local protocol work.

The implementation must update these binding surfaces together:

- `AGENTS.md` issue and record rules.
- `.agents/roadmap_v1.md` issue-view procedure.
- `scripts/agent-issue-index.py`.
- `scripts/check-agent-record.py`.
- The focused record and issue-index tests.

The frozen `.agents/completed/issue-index.md` supplies migration evidence. It does not remain an active issue authority.

## Our baseline

The 31 August 2026 baseline query returned 105 open GitHub issues. The old generated snapshot reported 66 issues without a `Row:` line.

The removed roadmap table contained 98 issue numbers. Only eight were open at the time of removal:

- #41
- #125
- #146
- #193
- #206
- #269
- #332
- #382

The static roadmap table duplicated remote state and had become stale. Task 10 removed those rows and added a checker guard against their return.

Task 11 removed the continuous integration call to `tests/scripts/test_check_issue_index_append_only.py`. The current renderer suite remains `tests/scripts/test_agent_issue_index.py`.

Tasks 12 and 13 remain valid. They converted duplicate feature rows into projections and named engine and model `PARTIAL` gaps. The developer reviewed those changes before changing the issue-authority design.

The unreviewed Task 14 quantization commit was rolled back. This specification does not restore any part of that commit.

Tasks 23 through 25 added schema, operation, projection, migration, and renderer code. The first Task 26 attempt did not land.

Task 26 review found 287 archived `Row: -` issue numbers already cited by other specifications. Of those references, 202 have exactly one existing spec owner. The remaining 85 occur under multiple specs and require reconciliation.

The same review found two lost canonical row mappings. Frozen archive line 365 maps #1168 to `GDN-MOE-BF16-OUT`. Frozen archive line 505 maps #1574 to `BENCH-QWEN38-27B-SOTA`.

Stable reference discovery is incomplete. The current `referenced_issues()` path scans commit messages for `#<number>` and `issues/<number>`. It does not scan changed files or recognize `ISSUE-GH-<number>` and `ISSUE-LOCAL-<ULID>`.

## Port map

No upstream code is ported.

| Item | Current anchor | Required change |
|---|---|---|
| Ordered roadmap | `.agents/roadmap_v1.md` | Keep the accepted issue-table removal. Later, point its procedure to the local-only refresh command. |
| Canonical issue storage | `.agents/issues/` | Add one tracked Markdown file per issue. Do not add a shared tracked index. |
| Row-owned GitHub issue | `.agents/issues/<ROW-ID>/ISSUE-GH-<number>.md` | Use this path when importing an issue that already has a GitHub number and a valid row. |
| Locally created owed issue | `.agents/issues/_owed/ISSUE-LOCAL-<ULID>.md` | Use this path until a canonical row owns the issue. Keep the ID after assignment or mirroring. |
| Migration-only intake | `.agents/issues/_intake/ISSUE-GH-<number>.md` | Preserve only ownerless archived `Row: -` metadata residue. Do not place new or full records here. |
| Issue operations | `scripts/agent-issue.py` | Add explicit `create`, `update`, `mirror`, `import-github`, and `close` operations. |
| Issue renderer | `scripts/agent-issue-index.py` | Make `--refresh` glob local issue files. It performs no network call. |
| Record checker | `scripts/check-agent-record.py` | Validate all local issue files and diff-scoped branch references offline. |
| Generated view | `.agents/issue-index.generated.md` | Keep the derived view untracked and ignored. It is not canonical. |
| Frozen issue history | `.agents/completed/issue-index.md` | Parse it once for migration, then keep it unchanged as migration evidence. |
| GitHub | Issue title, body, and state | Treat these values as an optional mirror of local canonical text. |
| Comments | GitHub issue comments | Do not mirror them automatically. Move binding content into an issue file or owning spec. |
| Renderer test wiring | `.github/workflows/ci.yml` | Keep the accepted Task 11 repair and one current renderer suite. |
| Matrix cleanup | Feature, engine, and model matrices | Keep the reviewed Task 12 and Task 13 results. |

## Design

### Canonical storage

Each issue has one tracked Markdown file.

A GitHub issue imported with a valid row uses:

```text
.agents/issues/<ROW-ID>/ISSUE-GH-<number>.md
```

A locally created issue without row ownership uses:

```text
.agents/issues/_owed/ISSUE-LOCAL-<ULID>.md
```

A row-owned local issue uses the same `ISSUE-LOCAL-<ULID>.md` file name under its row directory. Assigning a row moves the file but does not change its ID.

Mirroring a local issue does not convert `ISSUE-LOCAL-<ULID>` to `ISSUE-GH-<number>`. The command adds the optional GitHub number and keeps the local ID. Importing an existing GitHub issue creates an `ISSUE-GH-<number>` ID.

A migration-only ownerless archive record uses:

```text
.agents/issues/_intake/ISSUE-GH-<number>.md
```

`_intake` is evidence preservation, not ordinary ownership. No new issue or full record may use this directory.

### Canonical schema

Every issue file contains these canonical fields:

```text
ID: ISSUE-GH-<number> | ISSUE-LOCAL-<ULID>
Title: <concise title>
Row: <ROW-ID or ->
State: <OPEN, CLOSED, or UNKNOWN>
Kind: <issue kind or UNKNOWN>
GitHub: <issue number or ->
Mirror: <SYNCED, PENDING, MISSING, or DIVERGED>
Availability: <FULL or METADATA_ONLY>
Created: <YYYY-MM-DD or UNKNOWN>
Updated: <YYYY-MM-DD or UNKNOWN>
Closed: <YYYY-MM-DD, -, or UNKNOWN>

## Problem

<canonical problem statement and evidence>

## Resolution

<canonical resolution or ->
```

All listed fields and both headings are required. `GitHub: -` means that the record has no GitHub number.

`Availability: FULL` requires a non-placeholder `Problem`. Locally created and reachable imported records use `FULL`.

`Availability: METADATA_ONLY` is limited to migrated archive entries whose GitHub issue cannot be read. Such a record uses `State: UNKNOWN`, `Mirror: MISSING`, and `UNKNOWN` for all three dates. Its `Problem` cites the exact frozen archive entry that proves the issue identity. It does not infer state, dates, body text, or resolution from the frozen index.

An `_intake` record has additional schema restrictions. It must use an `ISSUE-GH-<number>` identity, `Row: -`, `State: UNKNOWN`, `Mirror: MISSING`, and `Availability: METADATA_ONLY`. It must use `UNKNOWN` for `Created`, `Updated`, and `Closed`, and `-` for `Resolution`. The frozen archive row must contain `Row: -`, and migration must be unable to recover an owner. Migration must not invent a row, date, state, problem, or resolution.

The intake record's `Problem` is exactly this deterministic form:

```text
Archive: `.agents/completed/issue-index.md:<line>`

### Frozen archive evidence

> <verbatim archived row>
```

`<line>` is the decimal source line of the entry. After the Markdown `> ` prefix, the evidence row must equal that complete archive line exactly. The block contains one quoted row and no other quoted lines.

`State: OPEN` requires dated `Created` and `Updated` values and `Closed: -`. `State: CLOSED` requires dated `Created`, `Updated`, and `Closed` values plus non-placeholder `Resolution` evidence. `State: UNKNOWN` does not require resolution evidence. Open and unknown records may use `-` under `## Resolution`.

`Kind: UNKNOWN` is valid when a reachable issue has no archived kind and no canonical `Kind:` line. Migration records the unknown value instead of guessing from title, labels, or prose.

### Identity invariants

The file basename equals `<ID>.md`. A local ID is immutable after creation.

An `ISSUE-GH-<number>` ID requires `GitHub: <number>`. The number in the ID, filename, and `GitHub:` field must match exactly.

An `ISSUE-LOCAL-<ULID>` ID keeps that ID and filename after row assignment and GitHub mirroring. Adding a GitHub number never renames it.

Each local ID occurs in exactly one file. Each nonempty GitHub number occurs in exactly one file.

`SYNCED`, `MISSING`, and `DIVERGED` require a GitHub number. A record without a GitHub number must use `PENDING`.

### Mirror states

The `Mirror:` field has exactly four values:

- `SYNCED`: A mirror read-back found the same GitHub title, state, and normalized body as the local projection.
- `PENDING`: The local record has no GitHub number and therefore has no remote identity to compare.
- `MISSING`: An explicit network operation could not read the stored GitHub number.
- `DIVERGED`: A numbered record has an unmirrored local change, or an explicit comparison found a readable mismatch.

`SYNCED`, `MISSING`, and `DIVERGED` are invalid without a GitHub number. Offline checks trust the recorded mirror state and never contact GitHub.

### Local-first operations

`scripts/agent-issue.py create` writes and validates the local file first. It allocates an `ISSUE-LOCAL-<ULID>` ID. Without a canonical row, it writes under `_owed`, sets `Mirror: PENDING`, and requires one owning spec. It never writes under `_intake`.

`scripts/agent-issue.py update <file>` accepts changes to `Title`, `Row`, `Kind`, `Problem`, and `Resolution`. It rejects direct changes to identity, mirror metadata, availability, state, and dates. It validates the complete candidate record before replacing the local file. It preserves `ID` and `Created`, sets `Updated` to the operation date, and performs no GitHub write.

If `Row` changes, `update` moves the file to the matching row directory or `_owed` path in the same local operation. The filename and local ID do not change. The surrounding commit must also update the single `_owed` spec reference when ownership changes.

No update may move a record into `_intake`. An update that triages an intake record must move it atomically to a valid row directory or `_owed`. An `_owed` promotion must add exactly one stable `## Owed` spec reference in the same change. Every promotion preserves the issue ID.

After any canonical change, `update` sets `Mirror: DIVERGED` when `GitHub` contains a number. It sets `Mirror: PENDING` when `GitHub: -`.

`scripts/agent-issue.py mirror` projects the local title, state, and body to GitHub. It creates or updates the remote issue only after the local record validates. It then reads the GitHub issue back. The command records `SYNCED` only when the read-back matches the projection. It records `DIVERGED` for a readable mismatch or a failed write to an existing numbered issue. It records `MISSING` when the numbered issue is unavailable. A failed create without an assigned GitHub number remains `PENDING`.

`scripts/agent-issue.py import-github <number>` reads one GitHub issue. If no local record has that GitHub number, it creates an `ISSUE-GH-<number>` file with `Availability: FULL`. It uses the canonical row directory when the imported body names a valid row. Otherwise, it uses `_owed` and requires exactly one owning spec. Import never creates a record in `_intake`.

Import compares the GitHub title, state, and normalized body with the local projection. An exact comparison records `SYNCED`. A difference records `DIVERGED` and reports each mismatch.

Import does not overwrite an existing `FULL` record. It reports differences for operator reconciliation. One explicit exception promotes a matching `METADATA_ONLY` record. Import replaces the archive placeholder with full remote evidence, imports the remote state and dates, and changes `Availability` to `FULL`.

If that record is in `_intake`, import must move it atomically to a valid row directory or `_owed`. An `_owed` move requires exactly one owning spec reference in the same change. Import reports the canonical local promotion before it compares projections.

`scripts/agent-issue.py close` requires resolution evidence. Its local write sets `State: CLOSED`, `Closed`, `Updated`, and `Resolution`. It sets `Mirror: DIVERGED` for a numbered record and `PENDING` otherwise. It then closes an existing GitHub mirror and applies the same read-back rule as `mirror`. A remote failure leaves the local issue closed and never restores `SYNCED`. Close rejects every `_intake` record.

### Deterministic GitHub projection

The GitHub title equals the local `Title` exactly. Dates, `Mirror`, and `Availability` remain local-only.

The projected GitHub body has this exact open-record shape:

```text
Row: <row-or-dash>

Local-Issue: <ID>
Kind: <kind>

<canonical Problem>
```

For `Row: -`, `<row-or-dash>` is the single character `-`. A closed projection appends this section after the problem:

```text

## Resolution

<canonical Resolution>
```

Projection replaces CRLF and CR with LF. It removes trailing spaces and tabs from every line. It removes leading and trailing blank lines from `Problem` and `Resolution`, preserves internal blank lines, and assembles the fixed separators shown above. The final body ends with exactly one newline.

Import compares the GitHub title, GitHub state, and this normalized body. It ignores local-only dates and mirror metadata. `mirror` reads the written issue back and repeats the same comparison before it records `SYNCED`.

`State: UNKNOWN` has no GitHub state projection. `mirror` rejects an unknown record until a successful import supplies the full remote state and changes `Availability` to `FULL`.

### Local index refresh

`scripts/agent-issue-index.py --refresh` globs `.agents/issues/**/*.md`. It validates and renders those local records into `.agents/issue-index.generated.md`.

The refresh command performs no network call. Its result does not depend on `gh`, credentials, GitHub availability, or remote writes.

The generated view remains untracked. It is a convenience projection and offline checker input only where a readable projection helps. Gates validate canonical files directly.

### Ownership and branch references

A row-owned issue file lives under `.agents/issues/<ROW-ID>/`. Its `Row:` value equals the path row, and that row is canonical and claimable.

An `_owed` file uses `Row: -`. Exactly one committed spec lists its stable issue ID under `## Owed`. Assigning a canonical row moves the file out of `_owed` and updates both ownership references in one change.

An `_intake` file also uses `Row: -`, but it requires no `## Owed` reference. Intake preserves migration residue only. Triage must move it atomically to a valid row directory or `_owed`. An `_owed` promotion adds exactly one spec reference in the same change. The move preserves the stable ID.

Branch issue citations remain diff-scoped. The checker scans changed-file content and commit bodies for `ISSUE-GH-<number>`, `ISSUE-LOCAL-<ULID>`, `#<number>`, and `issues/<number>`. Each stable ID must resolve to one local canonical record. A number citation resolves through exactly one canonical `GitHub:` field.

Canonical identity fields are declarations, not citations. Citation discovery excludes the issue record's own `ID: ISSUE-GH-<number>` or `ID: ISSUE-LOCAL-<ULID>` declaration and its own `GitHub: <number>` declaration. The exclusion applies only to those canonical fields in that same issue file.

The exact `### Frozen archive evidence` block required in an intake record is archive evidence, not ownership. Within that exact block in that same intake file, citation discovery excludes `#N` and `issues/N` only when `N` equals the record's own `GitHub:` number. It does not exclude either number form for any other number in the block. It does not exclude either stable-ID form in the block, the record's own number anywhere else in `Problem` or `Resolution`, the archive evidence when copied into another changed file, or any match in a commit body.

After these narrow declaration and archive-evidence exclusions, every discovered stable-ID or number-form citation that resolves to a record under `.agents/issues/_intake/` fails the diff-scoped ownership gate. This applies even when the intake record is otherwise schema-valid. The gate continues to fail until triage moves the record atomically to a valid row directory or an `_owed` file with exactly one owning spec.

Offline validation enforces row, `_owed`, and `_intake` restrictions. It reports current intake debt for operator triage. No gate compares that debt with a shared baseline or fails because a global intake count increased.

No gate counts unowned GitHub issues. Remote changes cannot make an offline branch gate fail.

### Migration

Migration operates on the union of two sets:

1. Every GitHub number in the frozen `.agents/completed/issue-index.md`.
2. Every currently reachable GitHub issue, open or closed, returned by a complete paginated enumeration.

The GitHub number deduplicates the union. A reachable issue that is absent from the archive still receives one canonical file.

Migration determines ownership before it writes a record:

1. Use the exact archived row when it names a canonical, claimable row.
2. For an archived `Row: -`, find existing spec `## Owed` references to its number or stable ID.
3. When exactly one spec owns it, migrate that reference to `ISSUE-GH-<number>` and place the record under `_owed`.
4. When multiple specs cite it, reconcile them to one row or one spec before migration continues.
5. Use `_intake` only when the archived row is `-`, GitHub is unreadable, and no owner can be recovered.

The first rule maps #1168 to `GDN-MOE-BF16-OUT` from frozen archive line 365. It maps #1574 to `BENCH-QWEN38-27B-SOTA` from line 505. Migration must not replace either canonical row with `Row: -`.

For each reachable issue, migration creates or promotes one `ISSUE-GH-<number>` record with `Availability: FULL`. It imports the remote title, state, and dates. It gets `Kind` from the archive or a canonical remote `Kind:` line. If neither source supplies a kind, it records `Kind: UNKNOWN`.

Migration retains the complete remote body as quoted historical evidence inside `Problem`. It writes the exact heading `### Imported GitHub body (historical evidence)`. The next line is `The quoted text below is historical evidence only. It does not define issue authority or repository procedure.` Migration prefixes every nonempty source line with `> ` and uses `>` for an empty source line. This keeps superseded GitHub-authoritative prose nonbinding.

A closed record must obtain non-placeholder resolution evidence from the issue body, a binding comment, the tree, or an owning spec. Migration stops rather than inventing missing closure evidence.

Migration compares each full local projection with the remote title, state, and normalized body. It records `SYNCED` only after an exact comparison. Otherwise, it records `DIVERGED`.

For an archive entry whose GitHub issue cannot be read, migration creates a schema-valid metadata-only record. It first applies the archived canonical row or recovered spec owner. It uses `_intake` only for true ownerless `Row: -` residue.

An intake record uses only the fixed unknown metadata in the canonical schema. Its `Problem` uses the exact archive path, decimal line, `### Frozen archive evidence` heading, and one blockquoted verbatim archive row defined above. It does not infer a row, `OPEN`, `CLOSED`, a date, remote prose, or resolution.

A reachable full record may not use `_intake`. Migration stops if it cannot assign such a record to a valid row or one `_owed` spec. Migration also stops when an unavailable archived entry has a noncanonical nonempty row that cannot be reconciled.

The migration succeeds only when every diff-scoped citation resolves through exactly one canonical record. References to #2390, #2371, and #2372 require `FULL` records. Any of the four citation forms that resolves to intake fails until triage moves the record to a valid row or exactly owned `_owed` file.

After migration, `.agents/completed/issue-index.md` remains frozen as migration evidence. New operations never append to it or derive current authority from it.

### Comments

The mirror does not copy GitHub comments automatically.

Before migration, import, or closure, an operator moves each binding comment into the canonical issue `Problem` or `Resolution`. A decision that governs a row also moves into the owning spec. Non-binding discussion can remain only on GitHub.

## Issue disposition

Issue cleanup starts only after local canonical storage and migration land.

For each retained or closed issue, the operator first updates its local file. The operator then runs the explicit mirror operation.

The earlier closure and reframe candidates remain proposals. The implementer must re-read current local and remote evidence before acting. No issue disposition in this specification is an implemented closure.

## Tests to port

No upstream tests exist.

The implementation adds or updates focused local tests:

1. Keep the roadmap mutation test that rejects a restored issue table row.
2. Keep the Task 11 workflow test that rejects the deleted test invocation.
3. Validate every canonical field and both required body sections.
4. Accept `Kind: UNKNOWN` without inferring a kind from title, labels, or prose.
5. Accept `State: UNKNOWN` only for unavailable metadata-only migration records.
6. Require dated closure and resolution evidence only for `CLOSED`.
7. Reject state or date claims inferred from a frozen archive entry.
8. Reject duplicate local IDs and duplicate nonempty GitHub numbers.
9. Require an `ISSUE-GH-<number>` ID, filename, and `GitHub:` value to match.
10. Prove an `ISSUE-LOCAL-<ULID>` filename survives row assignment and mirroring.
11. Require a GitHub number for `SYNCED`, `MISSING`, and `DIVERGED`.
12. Require `PENDING` when a record has no GitHub number.
13. Reject a row-owned path whose directory differs from `Row:`.
14. Reject a row-owned issue whose row is not canonical and claimable.
15. Reject an `_owed` issue without exactly one owning spec reference.
16. Accept an eligible `_intake` issue without an `## Owed` reference.
17. Prove `agent-issue-index.py --refresh` globs local files and performs no network call.
18. Prove `create` writes a local `ISSUE-LOCAL-<ULID>` record before any mirror attempt.
19. Prove `update <file>` validates before replacement, updates `Updated`, and performs no remote write.
20. Prove a row update moves the path without changing the ID or filename.
21. Prove `update` records `DIVERGED` with a GitHub number and `PENDING` without one.
22. Prove the GitHub projection has exact metadata lines, separators, LF normalization, trailing-whitespace removal, and one final newline.
23. Prove closed projections append the normalized `## Resolution` section and open projections do not.
24. Prove `mirror` reads back title, state, and normalized body before recording `SYNCED`.
25. Prove `import-github` creates a missing full record.
26. Prove `import-github` records `SYNCED` only when title, state, and normalized body match.
27. Prove `import-github` reports differences without overwriting an existing full record.
28. Prove import explicitly promotes an unknown metadata-only record to full remote facts.
29. Prove `close` refuses missing resolution evidence and writes locally before mirroring.
30. Prove migration covers the union of archive entries and all reachable GitHub issues.
31. Prove migration includes a reachable issue that is absent from the archive.
32. Prove migration preserves remote bodies as marked historical evidence.
33. Prove unavailable archive entries remain `UNKNOWN`, `MISSING`, and `METADATA_ONLY`.
34. Prove #2390, #2371, and #2372 resolve after migration.
35. Prove no operation automatically mirrors comments.
36. Accept `_intake` only for an `ISSUE-GH-<number>` archive record with `Row: -`, `UNKNOWN`, `MISSING`, and `METADATA_ONLY`.
37. Require each intake `Problem` to use the exact deterministic archive path, decimal line, `### Frozen archive evidence` heading, and one blockquoted verbatim archive row.
38. Reject an intake record that invents a row, state, date, problem, or resolution.
39. Reject intake when the archived row names a canonical row or an owner can be recovered.
40. Prove create, update, import, and close cannot place a new or full record in intake.
41. Prove intake promotion atomically moves to a valid row or singly owned `_owed` path without changing the ID.
42. Reject an `ISSUE-GH-<number>` citation to intake in changed-file content and in a commit body until triage.
43. Reject an `ISSUE-LOCAL-<ULID>` citation to intake in changed-file content and in a commit body until triage.
44. Reject a `#<number>` citation that resolves through `GitHub:` to intake in changed-file content and in a commit body until triage.
45. Reject an `issues/<number>` citation that resolves through `GitHub:` to intake in changed-file content and in a commit body until triage.
46. Exclude only a canonical issue file's own `ID:` and `GitHub:` declarations from citation discovery, covering both stable-ID declaration forms, and prove the same forms remain discoverable outside those fields.
47. Accept an eligible intake record when its exact frozen-evidence block contains `#N` and `issues/N` for its own `GitHub: N`.
48. Reject both number forms for any other intake number inside that block.
49. Reject the record's own `#N` and `issues/N` elsewhere in its `Problem` or `Resolution`.
50. Reject the number forms when the exact archive evidence is copied into another changed file.
51. Reject either stable-ID form in the frozen-evidence block or elsewhere in an intake record.
52. Prove offline validation reports intake debt without enforcing a shared count ratchet.
53. Prove migration preserves the archived row for #1168 and #1574.
54. Prove migration converts each unique legacy `## Owed` reference to its stable ID.
55. Prove migration blocks ambiguous multi-spec ownership until reconciliation.
56. Prove migration uses intake only after row and spec ownership recovery fail.
57. Reject a bulk archived issue list under this specification's `## Owed` section.

The existing renderer, diff-scoped ownership, and live-row audit tests remain applicable until their clean cutover. The implementation updates them rather than adding a second convention.

## Gates

This Task 32 specification-only wave runs:

```bash
python3 scripts/check-agent-record.py
```

A fresh reviewer must read the immutable specification commit and return `PASS`. Task 32 does not pass on the record checker alone.

The corrected implementation must add semantic issue-file tests for:

- Intake eligibility, the exact deterministic frozen-evidence block, and fixed unknown metadata.
- Hybrid ID, filename, GitHub-number, and uniqueness invariants.
- Path and row equality, canonical row validity, `_owed` ownership, and intake promotion.
- Changed-file content and commit-body citation discovery for all four forms, with only canonical self-declarations and the same intake record's own number forms in its exact frozen-evidence block excluded.
- Acceptance of the required self-evidence block, plus diff-scoped rejection of other numbers in that block, number forms elsewhere in `Problem` or `Resolution`, stable IDs, and copied archive evidence.
- Local-only index refresh and intake-debt reporting without a count ratchet.
- Create, update, mirror, import, and close ordering and intake exclusions.
- Exact GitHub projection and mirror read-back.
- Ordered migration ownership recovery and true-residue intake.
- Canonical archived mappings for #1168 and #1574.

No offline gate or `agent-issue-index.py --refresh` may make a network call.

No GPU, model, oracle, benchmark, or performance gate applies to this record row.

## Evidence

The baseline evidence remains:

- The removed roadmap table had 98 stored issue rows and eight open issues.
- The remote baseline had 105 open issues and 66 without a row.
- The ordered roadmap portfolio had 29 blocks and no direct current-open-issue reference.
- The live-row audit found duplicate feature projections and vague `PARTIAL` rows.
- The continuous integration workflow invoked a deleted issue-index test.

Accepted commits removed the roadmap issue table, repaired test wiring, converted duplicate feature rows into projections, and named engine and model gaps.

Task 26 review rejected a proposal that placed 594 archived `Row: -` records under this specification. That proposal would make every later assignment edit one shared spec.

The review found 287 existing spec references among those records. Exactly 202 had one spec owner, and 85 had multiple possible owners.

The review found only two row mismatches across 831 metadata-only records. Archive line 365 owns #1168 under `GDN-MOE-BF16-OUT`. Archive line 505 owns #1574 under `BENCH-QWEN38-27B-SOTA`.

The review also found incomplete stable reference discovery. `referenced_issues()` reads commit-message number forms but misses changed files and both stable ID forms.

The developer decision on 31 August 2026 is the authority for this specification correction. The prior GitHub-authoritative design is superseded.

## Dependencies

- Tasks 10, 11, 12, and 13 remain reviewed and valid.
- Tasks 23 through 25 landed on this branch before the failed Task 26 attempt.
- Task 32 must land before Task 26 retries.
- The correction to schema parsers, operations, migration, rendering, and focused tests must land before cutover.
- `.agents/completed/issue-index.md` must remain unchanged and readable during migration.
- GitHub access is required only to enumerate and enrich reachable issues during migration and to run explicit mirror operations.
- GitHub access is not a dependency of refresh or any offline gate.
- Each `_owed` issue needs one committed owning spec. An eligible intake record is the only exception.
- Every unique legacy spec owner must keep ownership through the stable-ID migration.
- Every ambiguous multi-spec owner must be reconciled before migration writes the record.
- The local migration must finish before issue closure and reframe work resumes.
- The two compile defects from #2372 still need separate row ownership and gates.
- Record edits must preserve unrelated matrix keys byte-for-byte.

## Work breakdown

The accepted and corrected sequence is:

| Task | Work | Status or completion condition |
|---:|---|---|
| 10 | Remove the stored roadmap issue projection | Accepted and preserved. The roadmap contains no embedded issue rows. |
| 11 | Repair renderer test wiring | Accepted and preserved. CI no longer invokes the deleted suite. |
| 12 | Convert duplicate feature rows into projections | Reviewed and preserved. Backend rows remain canonical. |
| 13 | Name engine and model partial gaps | Reviewed and preserved. Lifecycle states do not change. |
| 14 | Name quantization partial gaps | The unreviewed commit was rolled back. No quantization change is part of the branch. |
| 22 | Revise issue authority specification and claim | Accepted. The specification made tracked issue files canonical. |
| 23 | Add canonical schema parsers and focused tests | Implemented before the cutover attempt. |
| 24 | Add local issue operations and projection | Implemented before the cutover attempt. |
| 25 | Build migration and local renderer paths | Implemented before the cutover attempt. |
| 26 | Migrate and perform the clean cutover | The first attempt failed review because it centralized 594 owners, lost two archive rows, and missed stable changed-file references. Retry only after Task 32 and its implementation correction pass review. |
| 32 | Correct migration intake and ownership | Current specification-and-claim wave. It adds migration-only intake, ordered ownership recovery, coverage for all four citation forms, canonical self-declaration exclusion, and no global debt ratchet. No tooling or issue file changes are included. |
| 26 retry | Migrate and perform the clean cutover | Add union records only after unique owners migrate, ambiguous owners reconcile, canonical archive rows win, and true residue enters intake. Switch every binding surface atomically. |
| 27 | Resume evidence-backed issue cleanup | Each local change precedes its optional GitHub mirror operation. |
| 28 | Split the compile blockers | Each defect has one local issue, row, branch, and focused gate. |
| 29 | Run fresh review and operator gates | Reviewer mutations and the full record gate pass at an immutable row head. |

Task 32 precedes the Task 26 retry even though its task number is higher. Tasks 27 through 29 remain blocked on the successful retry.

## Risks and decisions

### Keep the roadmap free of issue rows

The roadmap stores developer-selected portfolio order. Canonical issue files store issue state without recreating one shared issue table.

### Do not centralize migrated ownership

One specification that owns hundreds of `_owed` records becomes a shared write lock. Migration preserves existing spec ownership instead of moving those references here.

### Keep intake narrow

Intake preserves only ownerless metadata evidence that migration cannot place safely. It is not a backlog, a default directory, or an ownership substitute.

### Report debt without a ratchet

An offline report makes intake debt visible to operators. A shared count threshold would couple unrelated branches and recreate the global gate this design removes.

### Keep hybrid IDs stable

Changing `ISSUE-LOCAL-<ULID>` after mirroring would break commits, specs, and branch references. The GitHub number is optional metadata, not a replacement identity.

### Make remote divergence visible

An import that overwrites a `FULL` record would reverse the authority decision. `DIVERGED` requires an operator to choose and record a reconciliation. The reported promotion of a metadata-only archive placeholder is the only automatic replacement.

### Keep local writes ahead of remote writes

A remote failure must not erase accepted local intent. Commands finish and validate the local change before attempting GitHub.

### Keep refresh offline

Network access would make a local record gate depend on credentials and remote availability. Refresh reads tracked canonical files only.

### Preserve incomplete history

An unreachable GitHub issue does not justify deleting archive evidence. Migration keeps metadata-only records and marks their mirrors `MISSING`.

Migration must recover a canonical archive row or existing spec owner before it uses intake. The intake fallback preserves evidence without guessing ownership.

### Keep imported authority prose nonbinding

Migration quotes each remote body under an exact historical-evidence marker. Remote instructions remain available as evidence but cannot override the local record protocol.

### Preserve unknown archive state

The frozen index proves that an issue identity existed. It does not prove current state, remote dates, or resolution. Unavailable archive records therefore remain `UNKNOWN` and `METADATA_ONLY`.

### Keep one GitHub projection

Exact title, state, and body projection prevents two tools from interpreting mirror equality differently. Dates and mirror metadata stay local.

### Keep comments out of automatic mirroring

Comments contain discussion, automation, and transient status. Binding decisions belong in canonical issue or spec text.

### Preserve reviewed work

The authority correction does not invalidate the roadmap cleanup, test wiring repair, feature projections, or named engine and model gaps.

### Exclude rolled-back quantization prose

The Task 14 commit lacked review under the corrected design. The rollback remains effective until new evidence and review authorize that work.

## Stop conditions

Return `NEEDS_DECISION` if:

- Migration cannot map one archive entry to one stable issue ID.
- A complete GitHub enumeration cannot finish, or a reachable issue cannot receive a full record.
- A reachable or full record would need to enter `_intake`.
- An archived canonical row would be replaced with `Row: -`.
- #1168 does not map to `GDN-MOE-BF16-OUT`, or #1574 does not map to `BENCH-QWEN38-27B-SOTA`.
- A unique existing spec owner would be replaced by this specification or intake.
- Multiple spec owners cannot be reconciled to one row or one spec.
- An intake candidate lacks an archived `Row: -` entry or has a recoverable owner.
- An intake record lacks the exact deterministic archive path, decimal line, `### Frozen archive evidence` heading, or one blockquoted verbatim archive row.
- Intake would require an invented row, state, date, problem, or resolution.
- Create, update, import, or close would place a new or full record in intake.
- Intake promotion cannot move atomically to a valid row or singly owned `_owed` path.
- Intake promotion would change the stable issue ID.
- Any stable-ID citation or any number-form citation outside the same intake record's exact same-number frozen-evidence block would resolve to intake and pass the diff-scoped ownership gate.
- Citation discovery would omit changed-file content, commit bodies, or any of the four forms; count a canonical issue file's own `ID:` or `GitHub:` declaration; reject the required same-number frozen evidence; or exclude a different number, another location in `Problem` or `Resolution`, copied evidence, or a stable ID.
- Offline validation would enforce a shared intake count ratchet.
- Migration would add a bulk archive issue list to this specification's `## Owed`.
- #2390, #2371, or #2372 does not resolve through exactly one canonical GitHub number after migration.
- Two local files claim the same local ID or nonempty GitHub number.
- An `ISSUE-GH-<number>` ID, filename, and `GitHub:` value differ.
- A mirror state that requires GitHub has no GitHub number.
- A row-owned issue cannot name one valid canonical row.
- An `_owed` issue cannot name one owning spec.
- A closed record lacks dated resolution evidence.
- Migration would infer state or dates from the frozen archive.
- An import would need to overwrite local canonical text without an explicit reconciliation decision.
- Migration would assign a guessed kind instead of `Kind: UNKNOWN`.
- Imported remote authority prose lacks the exact historical-evidence marker.
- Migration would need to discard an unreachable or metadata-only archived issue.
- A binding comment cannot be represented in issue or spec text.
- Refresh or an offline checker would require a network call.
- A mirror operation would change an `ISSUE-LOCAL-<ULID>` identity.
- Two operations produce different GitHub projections for the same canonical record.
- The clean cutover would leave GitHub-authoritative instructions in a binding surface.
- Issue tooling or issue files enter this Task 32 specification commit.
- Task 26 retries before the Task 32 correction and focused review pass.
- Rolled-back quantization changes reappear without a new reviewed task.

## Git integration

The developer selected one pull request for the row on 31 August 2026. Task 32 lands as a specification correction before the implementation changes and Task 26 retry.

The remaining commit order is:

1. Task 32 specification and W9 claim correction.
2. Intake, ownership migration, four-form citation discovery, self-declaration exclusion, and focused test corrections.
3. Task 26 atomic record migration and protocol cutover retry.
4. Issue cleanup and compile-issue split.
5. Review repairs.

The compile defects split from #2372 use their own issues and branches. They can land after local canonical issue storage exists.

No merge authority is recorded for the `ENG-RECORD-CONFLICT-SURFACES` issue #2390 follow-up.

## Outcome

Task 26 migrated the immutable union captured at
`2026-09-01T00:13:28+00:00`: 895 frozen archive identities plus 192 completely
paginated reachable GitHub issues, with 64 identities in both sets, produced
1,023 canonical local files. The canonical JSON payload had SHA-256
`e6f0b44054803ccbc544d9fd3b125534948bc847c89dc1eacc1d9c0bb3deef1a`.
The migration did not edit GitHub.

The final collection contains 192 `FULL` and 831 `METADATA_ONLY` records; 118
are `OPEN`, 74 are `CLOSED`, and 831 are `UNKNOWN`; 192 mirrors are `DIVERGED`
and 831 are `MISSING`. Twelve ownerless unavailable archive records remain in
`_intake`. Ninety-eight records retain exactly one stable spec owner under
`_owed`; every other record has a canonical row. Existing unique owners were
rewritten in place, all 32 multi-spec mappings were reconciled, and the rejected
594-entry owner list was not restored.

Archive rows win when still canonical. In particular, ISSUE-GH-1168 remains
under `GDN-MOE-BF16-OUT` and ISSUE-GH-1574 remains under
`BENCH-QWEN38-27B-SOTA`. ISSUE-GH-2390, ISSUE-GH-2371, and ISSUE-GH-2372 are
all `FULL`. The quoted GitHub bodies remain historical evidence only.

ISSUE-GH-1060, ISSUE-GH-1473, ISSUE-GH-1483, and ISSUE-GH-1495 preserve the
exact GitHub `not_planned` actor/date disposition and make no implementation
claim. ISSUE-GH-1651 remains locally `OPEN` and `DIVERGED`: the named pull
request did not merge, the named spec is absent, and the recorded 76.6 tok/s
does not meet its 150 tok/s gate.

The renderer, record checker, roadmap procedure, and repository policy now read
local files as authority. Refresh and validation are offline. The generated
view remains ignored and untracked; `.agents/completed/issue-index.md` remains
frozen migration evidence.

## Owed

- ISSUE-GH-2371: preserve the accepted deleted-test wiring repair during the protocol cutover.
- ISSUE-GH-2372: split the DSA and GLM53 compile defects by owner after local issue migration.

This section must never list the archived residue set. Migration keeps an existing unique spec owner, reconciles ambiguous owners, or uses `_intake` for true residue.

## Now

`ACTIVE`. The local-authority migration and atomic protocol cutover are complete. Twelve unavailable ownerless archive records remain as reported migration intake, without a shared count ratchet. ISSUE-GH-2372 still owns the post-migration split of the DSA and GLM53 compile defects, so the parent row does not move to `DONE`.
