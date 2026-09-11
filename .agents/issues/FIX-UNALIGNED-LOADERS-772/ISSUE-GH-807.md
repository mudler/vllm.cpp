ID: ISSUE-GH-807
Title: **FIXED IN FLOW 2026-08-14, `row/FIX-UNALIGNED-LOADERS-772`.** `docs/FEATURES.md:83-84` carried the row `Safetensors direct load, no conversion` TWICE under the identical key, with two rows that did not agree: one implied the unaligned-read class was handled, the other named three loaders still owed. `fc903b8dd` (the #674 fix) ADDED a row while the second already existed, and adding a row next to an edit of a different row merges cleanly — the "two relocations auto-merge into a duplicate" shape on a keyed projection. Nothing checks `FEATURES.md` for duplicate row keys; `check-public-doc-tables.py` validates structure, not key uniqueness. Collapsed into one accurate row by the #772 fix, which is what made the second row's claim false
Row: FIX-UNALIGNED-LOADERS-772
State: UNKNOWN
Kind: bug
GitHub: 807
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:146`

### Frozen archive evidence

> | [#807](https://github.com/mudler/vllm.cpp/issues/807) | `FIX-UNALIGNED-LOADERS-772` | **FIXED IN FLOW 2026-08-14, `row/FIX-UNALIGNED-LOADERS-772`.** `docs/FEATURES.md:83-84` carried the row `Safetensors direct load, no conversion` TWICE under the identical key, with two rows that did not agree: one implied the unaligned-read class was handled, the other named three loaders still owed. `fc903b8dd` (the #674 fix) ADDED a row while the second already existed, and adding a row next to an edit of a different row merges cleanly — the "two relocations auto-merge into a duplicate" shape on a keyed projection. Nothing checks `FEATURES.md` for duplicate row keys; `check-public-doc-tables.py` validates structure, not key uniqueness. Collapsed into one accurate row by the #772 fix, which is what made the second row's claim false | bug |

## Resolution

-
