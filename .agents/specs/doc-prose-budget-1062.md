# DOC-PROSE-BUDGET-1062: the public pages grew prose where the contract wants rows

**Row:** `DOC-PROSE-BUDGET-1062`
**Issue:** [#1062](https://github.com/mudler/vllm.cpp/issues/1062)
**Base:** `origin/main` `b493f4981`
**Status:** ACTIVE, 2026-08-16

## 1. Scope

`scripts/check-public-doc-tables.py` exits 1 on `origin/main`, so `main` is red
and every branch cut from it inherits the red. The checker runs in the
`agent-record` CI job (`.github/workflows/ci.yml:160`), in
`scripts/agent-preflight.sh`, and in the pre-push hook, so it blocks CI,
preflight, and pushes for everyone.

```
ERROR: the public keyed-table docs are not valid:
  - docs/BENCHMARKS.md has 36 prose paragraphs, over the 35 budget
  - docs/FEATURES.md has 22 prose paragraphs, over the 21 budget
```

**In scope.** The three prose paragraphs [PR #1054](https://github.com/mudler/vllm.cpp/pull/1054)
added to `docs/BENCHMARKS.md` and `docs/FEATURES.md`, moved into table rows on
the same two pages with every fact preserved, plus this spec and the
[issue index](../issue-index.md) row.

**Out of scope.** The checker itself, its constants, and its tests. The
qwen3.5 source change #1054 landed is correct and stays. Also out of scope, and
inherited rather than introduced here: the 13 two-column rows already sitting
inside four-column tables on `docs/BENCHMARKS.md`, and the malformed
`#1003` row in the issue index, which [#1059](https://github.com/mudler/vllm.cpp/issues/1059)
owns.

## 2. Anchors

Local governance surface. There is no vLLM counterpart: vLLM publishes no
keyed-table contract for its public documents.

| What | Where |
|---|---|
| The failing gate | `scripts/check-public-doc-tables.py` |
| The rule the pages break | `check-public-doc-tables.py:17`, `:105` |
| The prose that broke them | `e34d71379`, PR #1054 |
| Last green commit | `283c7e492` |

## 3. Design

The budget is not the defect and is not touched. `check-public-doc-tables.py:17`
and `:105` both state that table ROWS are unbudgeted and that nothing here
budgets the whole file; only prose paragraphs are capped, because rows are a
keyed table's growth mode and prose is its decay mode. Adding a measurement, the
normal operation, was never blocked. Raising the constants or retiring the
budget would make a red gate green by widening its scope, which AGENTS.md
forbids, so neither is done.

What is wrong is the SHAPE of the content, and the repair is to give each fact
the row its own page's schema already has for it.

**`docs/BENCHMARKS.md`.** "Benchmarking is NOT APPLICABLE, the binding gate is
the Apple Clang build" is a DISPOSITION, which is exactly what a row of the
`Open gaps` table records. That table is `Track | Status | Next gate`, and it
already carries a cluster of "no number owed" build-verification dispositions
(`Ampere consumer`, `Pre-Ampere breadth`) that this row joins. The date moves
into the row, which is what the checker's own regrowth-guard message directs
("put the date in the row or the prose"); `DATED_HEADING_RE` reads headings
only.

**`docs/FEATURES.md`.** The fact is that Apple Clang builds the Qwen3.5 MoE
loader with project warnings promoted to errors, and that the loader's
layout-refusal path is platform-invariant. That is a platform and toolchain
fact, so it belongs to `Backends and hardware`, whose Apple row it updates in
place. This follows the convention that table already sets: its `CPU` row's key
cell carries the same class of platform caveat.

Two alternatives were rejected. The `Registered architectures` row for
`Qwen3_5MoeForConditionalGeneration` measures 195 characters in its correctness
cell against a 220 cap, which does not fit the fact, and the sibling
`Qwen3_5ForCausalLM` cell is at exactly 220. Adding a row to the `At a glance`
table would have forced a claim in the `vLLM`, `SGLang`, and `llama.cpp`
columns about a C++ host-toolchain build that no evidence in this change
supports, and the page's own header says those columns are our reading of
documented behavior, so inventing three marks to house one of our facts is
worse than the prose was.

Neither page loses a fact, and neither page merges two paragraphs into one to
slip under a counter, which would game the count while leaving the page in the
shape the contract rejects.

## 4. Risks

- **A row that does not match its table's schema is a new defect.** Each row is
  checked against its own table's column count, not a global one: the `Open
  gaps` row is 3 columns, the `Backends and hardware` row is 5.
- **The cell and row caps still bind.** `MAX_CELL_CHARS` is 220 and
  `MAX_ROW_CHARS` is 600, and both are ENTRY-scoped, so this change pays for
  itself and evicts nobody.
- **Em-dashes are refused by the same checker.** Neither new cell contains one.
- **`docs/BENCHMARKS.md` carries 13 inherited malformed rows.** They are
  measured byte-for-byte identical at `b493f4981` and after this change, so a
  reviewer reading a row-shape scan does not attribute them here.

## 5. Tests and gates

RED first, from a clean detached worktree rather than the shared checkout:

- `python3 scripts/check-public-doc-tables.py` at `b493f4981` exits 1 with both
  budget errors.
- The same checker at `283c7e492` exits 0, and `git diff` proves the checker is
  byte-identical between the two commits, so the pages moved and the gate did
  not.

GREEN after:

- `python3 scripts/check-public-doc-tables.py` exits 0.
- `docs/BENCHMARKS.md` measures 35 prose paragraphs of 35 and
  `docs/FEATURES.md` 21 of 21, both computed with the checker's own
  `_prose_paragraphs`.
- Every table row on both pages splits into its own table's column count on
  unescaped pipes.
- `scripts/agent-preflight.sh` and `scripts/agent-preflight.sh --staged`.

## 6. Evidence

The verbatim red output, the verbatim green output, the paragraph counts, the
per-row shape scan, and the preflight result travel in the pull request body,
which is the landed commit message.

## 7. Stop conditions

Stop and return `NEEDS_DECISION` rather than relaxing any of these:

- A budget constant would have to move.
- A fact from #1054 would have to be dropped, softened, or merged into another
  paragraph.
- A row would have to make a claim about vLLM, SGLang, or llama.cpp that no
  evidence in this change supports.

## 8. Now

`ACTIVE`. Records only: no source, test, or gate file changes.

## 9. Nothing owed

[#1062](https://github.com/mudler/vllm.cpp/issues/1062) is fixed in this flow,
not deferred, so this row OWNS it in the issue index. There is deliberately no
`## Owed` section, because a row that owns an issue and also owes it would be
recorded twice by `scripts/check-agent-record.py`.
