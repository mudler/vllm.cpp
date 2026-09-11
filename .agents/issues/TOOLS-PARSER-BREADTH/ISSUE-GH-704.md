ID: ISSUE-GH-704
Title: `README.md` states the tool-parser registry size twice (`:81`, `:219`) and both were stale by two waves: 36 families / 40 names against an actual 38 / 42, re-derived from `tool_parser_names()` (42 entries) and `get_tool_parser` (42 branches over 38 distinct classes). Already wrong at `43a6c5518` (36/40 vs 37/41), so it missed `muse_glimmer` before #608 W1 too. Neither `check-readme-structure.py` nor `check-public-doc-tables.py` cross-checks a prose count against the registry. #649 covers only the engine-matrix row, a different surface. FIXED IN FLOW in #683's review repair
Row: TOOLS-PARSER-BREADTH
State: UNKNOWN
Kind: bug
GitHub: 704
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:198`

### Frozen archive evidence

> | [#704](https://github.com/mudler/vllm.cpp/issues/704) | `TOOLS-PARSER-BREADTH` | `README.md` states the tool-parser registry size twice (`:81`, `:219`) and both were stale by two waves: 36 families / 40 names against an actual 38 / 42, re-derived from `tool_parser_names()` (42 entries) and `get_tool_parser` (42 branches over 38 distinct classes). Already wrong at `43a6c5518` (36/40 vs 37/41), so it missed `muse_glimmer` before #608 W1 too. Neither `check-readme-structure.py` nor `check-public-doc-tables.py` cross-checks a prose count against the registry. #649 covers only the engine-matrix row, a different surface. FIXED IN FLOW in #683's review repair | bug |

## Resolution

-
