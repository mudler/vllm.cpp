ID: ISSUE-GH-649
Title: That row's prose still records `tool_parser_names()` 40 / `reasoning_parser_names()` 7; both enumerations have grown since 2026-07-24 and are now **41** (`tool_parsers/abstract.cpp:269`) and **12** (`reasoning_parsers/abstract.cpp:72`). Code and tests are correct — `test_detect.cpp:221` already pins 41 — only the record drifted. Halves belong to two other rows (#608, #605), so it is filed rather than repaired inside #643 (found while implementing #643's review findings)
Row: TOOLS-CALLING-CORE
State: UNKNOWN
Kind: bug
GitHub: 649
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:196`

### Frozen archive evidence

> | [#649](https://github.com/mudler/vllm.cpp/issues/649) | `TOOLS-CALLING-CORE` | That row's prose still records `tool_parser_names()` 40 / `reasoning_parser_names()` 7; both enumerations have grown since 2026-07-24 and are now **41** (`tool_parsers/abstract.cpp:269`) and **12** (`reasoning_parsers/abstract.cpp:72`). Code and tests are correct — `test_detect.cpp:221` already pins 41 — only the record drifted. Halves belong to two other rows (#608, #605), so it is filed rather than repaired inside #643 (found while implementing #643's review findings) | bug |

## Resolution

-
