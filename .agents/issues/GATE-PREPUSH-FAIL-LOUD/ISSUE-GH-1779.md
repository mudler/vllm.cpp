ID: ISSUE-GH-1779
Title: **`.githooks/pre-push` named six checkers and three of them had no file, and its file-test guard skipped each missing one in silence while the hook still exited 0** -- so it presented as six gates and ran three, and `core.hooksPath` is set to `.githooks` here, so it runs on every push. `check-policy.py` and `check-state-record.py` went with `0f3e44eee`, `check-public-doc-tables.py` with #1714; all three are deleted, not renamed. PART 1 FIXED IN FLOW: the loop now refuses a name it cannot find, the three dead names are pruned, the dead `--base` case arm goes with them, and `.githooks/README.md` stops listing the retired table gate. A red-first suite executes the hook against a scratch repository and pins both directions. PART 2 IS NOT FIXED AND STAYS OWED under `## Owed` in `.agents/specs/gate-prepush-fail-loud.md`: 65 specs still name a deleted checker, which needs its own row
Row: GATE-PREPUSH-FAIL-LOUD
State: UNKNOWN
Kind: bug
GitHub: 1779
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:665`

### Frozen archive evidence

> | [#1779](https://github.com/mudler/vllm.cpp/issues/1779) | `GATE-PREPUSH-FAIL-LOUD` | **`.githooks/pre-push` named six checkers and three of them had no file, and its file-test guard skipped each missing one in silence while the hook still exited 0** -- so it presented as six gates and ran three, and `core.hooksPath` is set to `.githooks` here, so it runs on every push. `check-policy.py` and `check-state-record.py` went with `0f3e44eee`, `check-public-doc-tables.py` with #1714; all three are deleted, not renamed. PART 1 FIXED IN FLOW: the loop now refuses a name it cannot find, the three dead names are pruned, the dead `--base` case arm goes with them, and `.githooks/README.md` stops listing the retired table gate. A red-first suite executes the hook against a scratch repository and pins both directions. PART 2 IS NOT FIXED AND STAYS OWED under `## Owed` in `.agents/specs/gate-prepush-fail-loud.md`: 65 specs still name a deleted checker, which needs its own row | bug |

## Resolution

-
