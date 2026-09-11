ID: ISSUE-GH-1055
Title: `main` went RED on `check-public-doc-tables.py` at `e34d71379`: two prose paragraphs added beside a two-character source fix took docs/BENCHMARKS.md to 36 paragraphs of 35 and docs/FEATURES.md to 22 of 21, and the checker runs both in the `pre-push` hook and at `.github/workflows/ci.yml:160`, so every branch in the repository inherited a red it did not cause. Fixed in flow by folding each paragraph into the keyed row its content belongs to, which is what the checker's own message prescribes: the Apple Clang build disposition into the docs/BENCHMARKS.md `Open gaps` table, the Apple Clang platform fact into the docs/FEATURES.md backend table. No paragraph was deleted and no budget was raised. The deeper defect is the budget itself, a whole-page count on a shared file, which AGENTS.md Records names as the anti-pattern (`Limit an entry, not a shared file`); redesigning it is `ENG-RECORD-CONFLICT-SURFACES` scope, whose spec already carries the obligation
Row: ENG-RECORD-CONFLICT-SURFACES
State: UNKNOWN
Kind: bug
GitHub: 1055
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:298`

### Frozen archive evidence

> | [#1055](https://github.com/mudler/vllm.cpp/issues/1055) | `ENG-RECORD-CONFLICT-SURFACES` | `main` went RED on `check-public-doc-tables.py` at `e34d71379`: two prose paragraphs added beside a two-character source fix took docs/BENCHMARKS.md to 36 paragraphs of 35 and docs/FEATURES.md to 22 of 21, and the checker runs both in the `pre-push` hook and at `.github/workflows/ci.yml:160`, so every branch in the repository inherited a red it did not cause. Fixed in flow by folding each paragraph into the keyed row its content belongs to, which is what the checker's own message prescribes: the Apple Clang build disposition into the docs/BENCHMARKS.md `Open gaps` table, the Apple Clang platform fact into the docs/FEATURES.md backend table. No paragraph was deleted and no budget was raised. The deeper defect is the budget itself, a whole-page count on a shared file, which AGENTS.md Records names as the anti-pattern (`Limit an entry, not a shared file`); redesigning it is `ENG-RECORD-CONFLICT-SURFACES` scope, whose spec already carries the obligation | bug |

## Resolution

-
