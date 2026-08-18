# ENG-RECORD-CONFLICT-SURFACES — `MAX_README_CHARS` is the last whole-file lock

Row: `ENG-RECORD-CONFLICT-SURFACES`. Issue
[#498](https://github.com/mudler/vllm.cpp/issues/498).

## The defect

`scripts/check-readme-structure.py:47` sets `MAX_README_CHARS = 30000` and
fails when `README.md` exceeds it. That is a budget on a whole shared file, and
AGENTS.md Records forbids exactly this shape: "Limit an entry, not a shared
file. A shared-file budget forces each addition to remove another entry.
Merging two such edits cleanly is worse than conflicting, because it applies
both removals."

The margin is not theoretical. Measured on `main` at `52ce0c84`:

| Surface | Size | Cap | Free |
|---|---:|---:|---:|
| `README.md` | 29,993 | 30,000 | **7** |

Seven characters. The next README edit reds CI, and two concurrent README edits
that each fit alone merge cleanly into a file that fails.

The issue records that 13 of the last 20 commits touching `README.md` sat under
60 characters free, that `031410e8` landed the file 52 characters over the cap,
and that a whole row (`row/DOCS-README-BUDGET`, #161) plus commit `44206e47`
existed only to pay this rent.

## Precedent — this is the fourth instance, not a new argument

Three budgets of the same shape were already retired for the same reason:

- per-class line budgets, retired 2026-08-10 (AGENTS.md records that nine of
  the previous 22 merged pull requests exceeded the product limit);
- `MAX_CHARS` in `scripts/check-now-current.py` and the `chars` key of
  `STATUS_RATCHET`, removed by `87308dea` under #364;
- `max_chars` on `PageRules` in `scripts/check-public-doc-tables.py`, removed
  under #460 (PR #494), where `docs/BENCHMARKS.md` had 205 characters free.

`87308dea` is the governing precedent and it is explicit about what survives:
it kept "every local per-cell and per-paragraph cap" while removing the
whole-file budget. This change mirrors it.

## Design — delete the file cap, keep the entry caps

Remove `MAX_README_CHARS` and the branch that reads it. Keep, unchanged:

- `MAX_CELL_CHARS = 220` — one table cell is one author's entry;
- `MAX_PARAGRAPH_CHARS = 900` — one prose paragraph is one author's entry;
- the required-section, section-ordering, em-dash, `STATUS_LINK` and
  `CONTRIBUTOR_LINK` shape rules, which count defects rather than length and so
  cannot collide between concurrent PRs.

The anti-drift obligation the cap claimed to serve is preserved by those entry
caps plus `STATUS_LINK`: the README cannot become the status ledger while every
paragraph is bounded at 900 chars, every cell at 220, and the file must point at
`docs/STATUS.md`.

### Rejected: a per-section budget

Issue #498 offers "a per-section or per-paragraph budget". A per-`##`-section cap
is rejected. A README section such as News is written by many pull requests, so
capping it recreates the shared-file lock at smaller granularity and reintroduces
the merge hazard this change exists to remove. A paragraph is the largest unit
one author writes alone, so the per-paragraph cap is the real entry cap and it
already exists. Recorded here because the code cannot record a rejected
alternative.

## Scope

In: `scripts/check-readme-structure.py`,
`tests/scripts/test_check_readme_structure.py`, this spec, the issue-index row.

Out: `README.md` content. This change does not add or remove a single character
of the landing page, so it cannot be confused with buying headroom. Out: the
other retired-budget checkers, which already landed. Out: the remaining
`ENG-RECORD-CONFLICT-SURFACES` surfaces, which #364 tracks.

## Tests

Red-before, in `tests/scripts/test_check_readme_structure.py`:

1. `test_oversized_readme_passes_when_every_entry_is_small` — a README far past
   30,000 chars built from compliant bullets and paragraphs returns no errors.
   RED before the change (the file cap fires), GREEN after.
2. `test_checker_declares_no_whole_file_budget` — asserts
   `MAX_README_CHARS` is absent from the module. RED before, GREEN after. This
   is the anti-regression tooth: it fails if the constant is reintroduced under
   any value.

The existing `test_oversized_readme_fails` is deleted, because it pins the
behavior being retired. Deleting it is the semantic change, and tests 1 and 2
are what replace its coverage.

Mutation evidence required (AGENTS.md, Changing the rules or a checker): with
the entry caps restored to the mutated tree, an oversized paragraph and an
oversized cell must still fail. A change that silently disabled the entry caps
along with the file cap would pass test 1 and must not.

## Gates

- `python3 tests/scripts/test_check_readme_structure.py`
- `python3 scripts/check-readme-structure.py` on the unchanged shipped README
- `scripts/agent-preflight.sh --staged`

No GPU, no build, no model. This is checker semantics and repository policy.

## Risks / decisions

- **Risk: the README drifts back into a status log.** Mitigated by the entry
  caps and `STATUS_LINK`, which is the same mitigation `87308dea` accepted for
  `check-now-current.py`. If drift is later measured rather than feared, the
  answer is a per-entry rule that names the drift, never a byte count on a
  shared file.
- **Decision: no replacement constant.** Adding a larger cap would repeat the
  cycle a fourth time; #161 and `44206e47` are what paying that rent looks like.

## Stop conditions

Stop and return `NEEDS_DECISION` if removing the cap is found to be load-bearing
for a gate outside this checker, or if `README.md` content changes are required
to make the suite green. Neither is expected: the pure function
`readme_errors(text)` is the only consumer.

## Now

Spec committed. Implementation follows in this pull request, red-before test
first.
