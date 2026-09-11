ID: ISSUE-GH-1287
Title: The record-anchor symbol test asks only whether the cited LINES CONTAIN the symbol, so a COMMENT naming it satisfies the test. `KERNEL-ATTN-MLA-SPARSE` cites `include/vllm/v1/attention/backend.h:271` for `get_kv_cache_shape`, which is a ROCm comment; the declaration is 70 lines down at `:341`, and the anchor passes for the wrong reason. The spec records it as a MEASURED LIMIT rather than a repair, because tightening it would need a parser per language. The second-order cost is the one this issue adds: the ratchet also fails when a bucket FALLS without the baseline being lowered in the same commit, so unrelated drift that parks a comment on a cited line forces the next contributor to bank an improvement that never happened. Worked example on this branch: `72bd06a5a` (a record reconciliation, #535) replaced the `SERVE-ASYNC-LLM` citation `examples/server/main.cpp:230-247` with a bare `examples/server/main.cpp`, so the anchor stopped being counted rather than being repaired, `broken` fell 7 to 6, and the merge had to bank it. NOT #911, which is the different gap that spec BODIES are unpoliced
Row: ENG-RECORD-ANCHOR-RATCHET
State: UNKNOWN
Kind: bug
GitHub: 1287
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:419`

### Frozen archive evidence

> | [#1287](https://github.com/mudler/vllm.cpp/issues/1287) | `ENG-RECORD-ANCHOR-RATCHET` | The record-anchor symbol test asks only whether the cited LINES CONTAIN the symbol, so a COMMENT naming it satisfies the test. `KERNEL-ATTN-MLA-SPARSE` cites `include/vllm/v1/attention/backend.h:271` for `get_kv_cache_shape`, which is a ROCm comment; the declaration is 70 lines down at `:341`, and the anchor passes for the wrong reason. The spec records it as a MEASURED LIMIT rather than a repair, because tightening it would need a parser per language. The second-order cost is the one this issue adds: the ratchet also fails when a bucket FALLS without the baseline being lowered in the same commit, so unrelated drift that parks a comment on a cited line forces the next contributor to bank an improvement that never happened. Worked example on this branch: `72bd06a5a` (a record reconciliation, #535) replaced the `SERVE-ASYNC-LLM` citation `examples/server/main.cpp:230-247` with a bare `examples/server/main.cpp`, so the anchor stopped being counted rather than being repaired, `broken` fell 7 to 6, and the merge had to bank it. NOT #911, which is the different gap that spec BODIES are unpoliced | bug |

## Resolution

-
