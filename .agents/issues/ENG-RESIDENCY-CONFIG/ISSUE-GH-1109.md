ID: ISSUE-GH-1109
Title: `docs/ENVIRONMENT.md:50` documented `VT_GGUF_PREFAULT`'s default as **off** while `PrefaultBorrowedSpan` (`qwen3_5_gguf_weights.cpp:37-44`) defaults it **ON** — unset reads as enabled — and no checker compares a documented default against the code that reads it. FIXED IN FLOW while landing #1110 rather than left alone: that row writes the same default into a resolver AND into a config key (`vllm_cpp.mmap.prefault`), so shipping it beside a table stating the opposite would have put the contradiction inside one change. The direction matters to a user, not only to a document: prefault ON is exactly what a model larger than memory must turn OFF, because the prefault reads the whole borrowed tower to populate a page cache that cannot hold it
Row: ENG-RESIDENCY-CONFIG
State: UNKNOWN
Kind: bug
GitHub: 1109
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:338`

### Frozen archive evidence

> | [#1109](https://github.com/mudler/vllm.cpp/issues/1109) | `ENG-RESIDENCY-CONFIG` | `docs/ENVIRONMENT.md:50` documented `VT_GGUF_PREFAULT`'s default as **off** while `PrefaultBorrowedSpan` (`qwen3_5_gguf_weights.cpp:37-44`) defaults it **ON** — unset reads as enabled — and no checker compares a documented default against the code that reads it. FIXED IN FLOW while landing #1110 rather than left alone: that row writes the same default into a resolver AND into a config key (`vllm_cpp.mmap.prefault`), so shipping it beside a table stating the opposite would have put the contradiction inside one change. The direction matters to a user, not only to a document: prefault ON is exactly what a model larger than memory must turn OFF, because the prefault reads the whole borrowed tower to populate a page cache that cannot hold it | bug |

## Resolution

-
