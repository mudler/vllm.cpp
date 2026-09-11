ID: ISSUE-GH-1079
Title: All four skip messages in `tests/parity/test_minimax_music3_quant_real.cpp` streamed the case name as a `const char*`, and doctest 2.5.2 stringifies that through its bool overload, so every one printed `SKIP 1` and named no case. The comment above the helpers states the obligation the messages then failed: a gate that silently passes when its asset is absent has not reported. It matters here because the binary reports `6 passed` with `assertions: 0` when the checkpoint is absent, so the message text is all that separates a skipped run from a gated one. Pre-existing on `main` at `100026481`. FIXED IN FLOW while landing [#1073](https://github.com/mudler/vllm.cpp/issues/1073), which rewrote those exact messages and would have carried the defect forward under a changed line; the fix streams `std::string(what)`. Scope measured before fixing: 4 hits, all in this one file
Row: FIX-NAS-PATH-1073
State: UNKNOWN
Kind: bug
GitHub: 1079
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:316`

### Frozen archive evidence

> | [#1079](https://github.com/mudler/vllm.cpp/issues/1079) | `FIX-NAS-PATH-1073` | All four skip messages in `tests/parity/test_minimax_music3_quant_real.cpp` streamed the case name as a `const char*`, and doctest 2.5.2 stringifies that through its bool overload, so every one printed `SKIP 1` and named no case. The comment above the helpers states the obligation the messages then failed: a gate that silently passes when its asset is absent has not reported. It matters here because the binary reports `6 passed` with `assertions: 0` when the checkpoint is absent, so the message text is all that separates a skipped run from a gated one. Pre-existing on `main` at `100026481`. FIXED IN FLOW while landing [#1073](https://github.com/mudler/vllm.cpp/issues/1073), which rewrote those exact messages and would have carried the defect forward under a changed line; the fix streams `std::string(what)`. Scope measured before fixing: 4 hits, all in this one file | bug |

## Resolution

-
