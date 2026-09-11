ID: ISSUE-GH-1447
Title: **`docs/USAGE.md` said EVERY Qwen3.8 decode figure came from the W0e C ABI harness; the 66.7 s/token streaming-off row of 16 August 2026 came from `vllm-server`, the same binary the section tells the reader to run.** Introduced by #1211 and fixed in the same flow: the sentence is scoped to the W0e and W0f runs and the exception is named, in `docs/USAGE.md` and in the spec paragraph that mirrors it. Provenance read at the source, `.agents/specs/expert-streaming.md:837` (server entry point) and `:905`.
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: record
GitHub: 1447
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:502`

### Frozen archive evidence

> | [#1447](https://github.com/mudler/vllm.cpp/issues/1447) | `ENG-EXPERT-STREAM` | **`docs/USAGE.md` said EVERY Qwen3.8 decode figure came from the W0e C ABI harness; the 66.7 s/token streaming-off row of 16 August 2026 came from `vllm-server`, the same binary the section tells the reader to run.** Introduced by #1211 and fixed in the same flow: the sentence is scoped to the W0e and W0f runs and the exception is named, in `docs/USAGE.md` and in the spec paragraph that mirrors it. Provenance read at the source, `.agents/specs/expert-streaming.md:837` (server entry point) and `:905`. | record |

## Resolution

-
