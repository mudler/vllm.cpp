# Retire shared public-document projections

Issue: [#1674](https://github.com/mudler/vllm.cpp/issues/1674).
Row: `ENG-PUBLIC-DOC-PROJECTIONS`.
Base: `origin/main` at `5539686c7` on 22 August 2026.

## Now

Implementation is assembled for fresh review. The shared status projection and
lifecycle checkpoint are removed, the ten benchmark sections are split into
row-owned detail files, and the retained benchmark check is mechanical only.
Next: fresh immutable-head review, repair any findings, then operator gates.

## Scope

This row removes `docs/STATUS.md`. `README.md` becomes the only public project
status overview. Agents read Git, row specs, matrices, and derived commands for
detailed lifecycle state and history.

This row keeps `docs/BENCHMARKS.md` as one compact index. Each public benchmark
owns one file at `docs/benchmarks/<benchmark-id>.md`.

This row removes repository-specific documentation policy gates. It keeps the
documentation site build and mechanical Markdown and link checks.

This row updates specialized gates that read phrases from STATUS or BENCHMARKS.
Those gates must read their canonical release or row records instead.

Product correctness gates, product tests, oracle gates, and performance gates
are out of scope. `docs/FEATURES.md`, `docs/USAGE.md`, model guides, and
reference pages remain public documents.

## Developer decisions

- Delete `docs/STATUS.md`.
- Put the small public project status summary in `README.md`.
- Keep `docs/BENCHMARKS.md` as the only benchmark index.
- Put each detailed benchmark in `docs/benchmarks/<benchmark-id>.md`.
- Stop requiring lifecycle changes to edit public documentation.
- Remove documentation policy gates from continuous integration and preflight.
- Keep the documentation site build and mechanical document validation.
- Use one pull request for the spec and implementation.
- Merge the pull request directly after fresh review and the operator gate.

## Baseline

At the base revision, `docs/STATUS.md` contains 98 lines and 7,290 bytes.
`docs/BENCHMARKS.md` contains 581 lines and 76,162 bytes.

The repository already reduced STATUS from about 3,000 lines. The remaining
page still duplicates lifecycle state from internal records. `AGENTS.md`
requires a STATUS edit whenever a row changes lifecycle state.

`docs/BENCHMARKS.md` remains a shared keyed file. Seven open pull requests edit
STATUS, BENCHMARKS, or both: #361, #1251, #1289, #1362, #1364, #1432, and
#1630. Six of those pull requests are currently conflicting.

The clean base passes `scripts/agent-preflight.sh`. The run includes
`check-doc-checkpoint`, `check-public-doc-tables`, and both mutation suites.

Issue #1585 asked to change the per-commit semantics of
`check-doc-checkpoint.py`. Issue #1520 asked to widen that checker's landing
source allowlist. The developer selected removal of the checker, so #1674
supersedes both issues. Both issues closed on 22 August 2026.

## History

Git history defines the prior motion.

- `39f29b0d7` compacted STATUS and added its ratchet.
- `87308dea3` removed the byte ratchet that coupled every STATUS edit to a
  checker edit. It retained content-quality ratchets.
- `aee6c48d6` split the large USAGE page into task-focused public pages.
- `8a0744ae0` established BENCHMARKS as a public keyed table.
- `7bc9c4bc0` changed the benchmark limit from a shared-file limit to an entry
  limit.

The new design continues the one-writer-per-file rule. It removes the last
mandatory status projection and applies the USAGE split shape to benchmarks.

## Upstream chain

No vLLM implementation defines this repository workflow. The change is local
policy work. `AGENTS.md` section "Changing the rules or a checker" governs the
implementation.

## Design

### Public status

Delete `docs/STATUS.md`. Move only stable, project-level status statements to a
small `README.md` section. Do not move row histories, dated evidence, open gate
details, or per-model limitations into README.

Replace each STATUS link by its true destination:

- use README for the project overview;
- use `docs/FEATURES.md` for shipped capabilities;
- use model guides for model limitations;
- use matrices and row specs for contributor lifecycle state;
- use Git for history.

No compatibility stub remains at `docs/STATUS.md`. A stub would preserve the
path and invite new state prose into the retired surface.

### Public benchmarks

Reduce `docs/BENCHMARKS.md` to one index. Each row contains a stable benchmark
ID, subject, current disposition, and one detail link.

Move each existing benchmark section or logical row to
`docs/benchmarks/<benchmark-id>.md`. A detail file owns its workload, artifact
pins, commands, environment, results, ratios, and limitations.

Use stable lowercase kebab-case IDs. The ID in the index must equal the file
stem. Do not store one benchmark across multiple public files.

The migration preserves every current benchmark claim. It can remove repeated
explanatory prose when the index or detail template states the rule once.

### Internal records

Git remains the history. Matrices and row specs remain the lifecycle records.
`.agents/benchmark-record.md` remains internal measurement evidence.

The public benchmark detail files project selected evidence for users. They do
not become lifecycle records or gate inputs for unrelated rows.

### Gate retirement

Delete `scripts/check-doc-checkpoint.py` and its tests. Remove every preflight,
continuous integration, helper-readiness, and integration call to that script.

Remove STATUS rules from `scripts/check-public-doc-tables.py` and its tests.
Remove benchmark policy rules that enforce prose budgets, section content,
dispositions, or synchronization with internal records.

Delete the checker if no retained public-document rule remains. Otherwise,
rename it to match its remaining mechanical purpose.

Remove specialized phrase matching against STATUS or BENCHMARKS. A release
gate reads the release manifest, state event, or release record that owns the
fact. A row gate reads the row record that owns the fact.

Keep the documentation site build. Keep mechanical Markdown and link checks.
Add one narrow benchmark index check for these mechanical errors:

- duplicate benchmark IDs;
- an index link with no detail file;
- a detail file with no index row;
- a file stem that differs from its benchmark ID.

The narrow check reads only `docs/BENCHMARKS.md` and
`docs/benchmarks/*.md`. It does not read lifecycle records. It does not require
an unrelated pull request to edit the index.

## Open pull request reconciliation

Do not close product pull requests because they edit retired documents. Their
product changes remain independent work.

After this row lands, notify #361, #1251, #1289, #1362, #1364, #1432, and
#1630. Each branch must remove its STATUS edit. Each benchmark claim must move
to a detail file or rebase onto an equivalent migrated file.

PR #1655 changes README only. It is not superseded by this row.

## Tests

The implementer captures red results before each semantic removal or migration.

1. Add a lifecycle-only fixture that changes a matrix or row state without a
   public-document edit. The current checkpoint gate must reject it before the
   removal. The final gate must accept it.
2. Add a benchmark-index fixture with one valid index row and one detail file.
3. Mutate the benchmark ID, delete the detail file, and duplicate the index ID.
   Each mechanical defect must fail.
4. Delete a production call to every removed documentation checker from the
   preflight and continuous integration paths. No test may still expect the
   retired check to run.
5. Build the documentation site. All internal links must resolve after STATUS
   deletion and benchmark migration.
6. Run the full script suite. Product gate registration must remain unchanged.

The reviewer mutates the retained guarantees in a scratch copy. The reviewer
restores the tree byte-for-byte after each mutation.

## Gates

- `scripts/agent-preflight.sh` before edits and before each commit.
- Focused tests for each changed checker or workflow.
- Full `tests/scripts/` suite.
- Documentation site build with the repository command from its workflow.
- `python3 scripts/agent-integration.py --base origin/main`.
- Fresh immutable-head review with mutation evidence.
- Operator rerun of the row gate.
- `python3 scripts/agent-pr-body.py --pr <N>` before merge.

No GPU gate applies. This row changes documentation and repository workflow.

## Work breakdown

### W1: red-first inventory

Map every caller and test for STATUS, BENCHMARKS, `check-doc-checkpoint.py`, and
`check-public-doc-tables.py`. Capture the lifecycle-only red result. Classify
each dependency as policy, mechanical validation, or specialized canonical
record validation.

### W2: retire STATUS

Move the stable overview into README. Delete STATUS. Repair inbound links. Move
specialized checks to canonical records.

### W3: split benchmarks

Create `docs/benchmarks/`. Migrate every public benchmark claim to one detail
file. Reduce BENCHMARKS to the single index.

### W4: retire policy gates

Delete the lifecycle checkpoint. Remove semantic public projection rules.
Remove their workflow and preflight calls. Retain mechanical validation.

### W5: reconcile and verify

Run the focused and full gates. Complete fresh review and repair loops. Update
the seven affected open pull requests by comment after the new paths land.

## Risks

The migration can lose a benchmark claim. Prevent this with a before-and-after
inventory keyed by benchmark ID and section heading.

README can become a new status log. Prevent this by limiting it to stable
project-level statements. No gate requires routine README edits.

Specialized gates can lose a real release guarantee. Move each guarantee to
its canonical record before deleting phrase matching. Do not delete a product
or release obligation with a documentation projection.

Open branches can reintroduce STATUS or the old benchmark table after this row
lands. Notify each affected pull request and require a rebase before merge.

## Git integration

The developer selected one pull request for the spec and implementation on 22
August 2026. Commit order proves that this spec lands before implementation.

The developer authorized a direct merge to `main` after fresh review and the
operator gate. The pull request body check remains required.

## Owed

None at spec time. Add an owned issue here if implementation finds a separate
defect that cannot be fixed in this flow.

## Stop conditions

Return `NEEDS_DECISION` if a specialized gate has no canonical record outside
STATUS or BENCHMARKS. Do not delete that obligation.

Return `NEEDS_DECISION` if the benchmark migration cannot preserve a current
claim under one stable benchmark ID.

Return `NEEDS_CONTEXT` if the documentation site build command or source set is
not available in the repository.

## Outcome

Implementation, repair, current-main reconciliation, and final fresh review are
complete. The final reviewer returned `PASS` at `0dfa517d5` with no findings.

The migration identified 10 logical benchmark sections and preserved all
80,169 normalized characters exactly across 10 detail files. `docs/STATUS.md`
was deleted, and README now owns the stable public project overview.

The lifecycle checkpoint and semantic public-document projection gates were
retired from preflight and continuous integration. The retained
`check-benchmark-index.py` checker validates only the mechanical relationship
between the benchmark index and its detail files. Its production preflight
registration now has an execution-level reachability assertion.

Issues #1520 and #1585 were closed as superseded by #1674. The open product
pull requests listed above remain to be reconciled after this row lands.

All change-related preflight gates pass. The overall full preflight
is not green because the shared host cannot satisfy the quiet window for
`test_cpu_x86_llamacpp_floor`. The observed load was approximately 30 to 46.
This contention is unrelated to the documentation change and is not the older
`tg128_c5` floor result.

The second-review repair preserves nested document paths in the Hugo link
renderer. An output-level test builds the site and verifies that all 10 links
in the benchmark index resolve to emitted detail pages. Replacing the nested
path with `path.Base` in a scratch copy makes the test fail on
`/vllm.cpp/docs/at-a-glance/`. The production renderer passes after the scratch
mutation is removed.

The repair also removes six false links from the internal benchmark record.
Current lifecycle statements now point readers to row-owned specs and matrices.
Historical `docs/STATUS.md` statements remain unlinked historical facts.

The final reviewer proved six guarantees by mutation:

- Deleting `memory.md` fails with a missing-detail error.
- Duplicating the `memory` row and link fails with a duplicate-ID error.
- Adding `orphan.md` fails with an orphan-detail error.
- Changing the link to `./benchmarks/memory.md` is rejected because `memory.md`
  becomes an orphan detail.
- Removing the preflight registration fails the execution-level unit test, and
  the captured argument vector no longer contains the checker.
- Restoring `path.Base` fails the output-level site test on
  `/vllm.cpp/docs/at-a-glance/`.

The current-main reconciliation preserved the
`ENG-EXPERT-STREAM-DEVICE` benchmark detail in
`docs/benchmarks/at-a-glance.md`. The operator gate passed every
change-related check. The unrelated shared-host contention means this result
does not claim fully green integration.
