ID: ISSUE-GH-1755
Title: `PhaseLog::RenderText` read `Elapsed()` AFTER `ByStart(Records())`, so the CONSOLE copy of a phase table charged its own copy and its own sort to the `WALL` it printed and to the `unaccounted` row above it. That is [#1569](https://github.com/mudler/vllm.cpp/issues/1569)'s defect on #1569's own sibling emitter: the file copy was repaired and the copy a reader watching a terminal gets was not. The call site was the larger half -- the console block stood at the END of `WriteJson`, after the whole `nlohmann` object was assembled, so it absorbed the JSON build as well: over five `WriteJson` calls on the 8001-record unit timeline `sum(leaf)` held at 0.189 s while the console's `unaccounted` climbed 0.065 -> 0.134 -> 0.200 -> 0.265 -> 0.329 s, about 66 ms of writer work per call charged to a render that had not run. NOTHING COULD SEE IT: `RenderText` prints every total with `%10.3f` and a copy and a sort of 8000 records is 0.12 ms, a quarter of one step of that format, so applying #1569's own one-line repair to its sibling left the suite at `7 | 7 passed` and `100 | 100 passed`. Found by the fresh review of [PR #1711](https://github.com/mudler/vllm.cpp/pull/1711) and fixed in the same flow: the clock is read first in both emitters, the console block moves above the copy, the sort and the build, and the gate takes its bound from `%10.3f`'s own last digit rather than from a wall-clock ratio, over a table large enough for the defect to cross it
Row: LTX25-PHASE-INSTRUMENT
State: UNKNOWN
Kind: bug
GitHub: 1755
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:651`

### Frozen archive evidence

> | [#1755](https://github.com/mudler/vllm.cpp/issues/1755) | `LTX25-PHASE-INSTRUMENT` | `PhaseLog::RenderText` read `Elapsed()` AFTER `ByStart(Records())`, so the CONSOLE copy of a phase table charged its own copy and its own sort to the `WALL` it printed and to the `unaccounted` row above it. That is [#1569](https://github.com/mudler/vllm.cpp/issues/1569)'s defect on #1569's own sibling emitter: the file copy was repaired and the copy a reader watching a terminal gets was not. The call site was the larger half -- the console block stood at the END of `WriteJson`, after the whole `nlohmann` object was assembled, so it absorbed the JSON build as well: over five `WriteJson` calls on the 8001-record unit timeline `sum(leaf)` held at 0.189 s while the console's `unaccounted` climbed 0.065 -> 0.134 -> 0.200 -> 0.265 -> 0.329 s, about 66 ms of writer work per call charged to a render that had not run. NOTHING COULD SEE IT: `RenderText` prints every total with `%10.3f` and a copy and a sort of 8000 records is 0.12 ms, a quarter of one step of that format, so applying #1569's own one-line repair to its sibling left the suite at `7 \| 7 passed` and `100 \| 100 passed`. Found by the fresh review of [PR #1711](https://github.com/mudler/vllm.cpp/pull/1711) and fixed in the same flow: the clock is read first in both emitters, the console block moves above the copy, the sort and the build, and the gate takes its bound from `%10.3f`'s own last digit rather than from a wall-clock ratio, over a table large enough for the defect to cross it | bug |

## Resolution

-
