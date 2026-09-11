ID: ISSUE-GH-703
Title: `--reasoning-parser inkling` still aborts startup while `--tool-call-parser inkling` resolves, an asymmetry #608 W1 created: upstream registers `inkling` in BOTH registries (`vllm/reasoning/__init__.py:131` -> `InklingParserReasoningAdapter`, the reasoning half of the same `make_adapters(InklingParser)` call whose tool half landed), and `reasoning_parser_names()` has no row for it. Inkling is a thinking dialect, so the reasoning flag is exactly the one an Inkling recipe passes (2 of 157 official recipes do). The fix belongs to #605, whose table already lists `inkling`; filed separately so the asymmetry is visible from the row that created it
Row: SAMPLE-REASONING
State: UNKNOWN
Kind: bug
GitHub: 703
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:197`

### Frozen archive evidence

> | [#703](https://github.com/mudler/vllm.cpp/issues/703) | `SAMPLE-REASONING` | `--reasoning-parser inkling` still aborts startup while `--tool-call-parser inkling` resolves, an asymmetry #608 W1 created: upstream registers `inkling` in BOTH registries (`vllm/reasoning/__init__.py:131` -> `InklingParserReasoningAdapter`, the reasoning half of the same `make_adapters(InklingParser)` call whose tool half landed), and `reasoning_parser_names()` has no row for it. Inkling is a thinking dialect, so the reasoning flag is exactly the one an Inkling recipe passes (2 of 157 official recipes do). The fix belongs to #605, whose table already lists `inkling`; filed separately so the asymmetry is visible from the row that created it | bug |

## Resolution

-
