ID: ISSUE-GH-1760
Title: The `### 10` console call-site gate holds only the JSON BUILD, not the copy and the sort. `M-SITE-MID` slides the `StderrEnabled()` block BELOW `ByStart(Records())` and `Sum(...)` while leaving it ABOVE the `nlohmann` build -- [#1755](https://github.com/mudler/vllm.cpp/issues/1755)'s own class, the console `WALL` charged with the writer's copy and sort -- and it survived **19 of 20 runs** at `test cases: 8 | 8 passed` and `assertions: 120 | 120 passed`, with one run in twenty red at `8 | 7 passed`. A mutation a gate catches once in twenty is one the gate does not catch. The cause is arm (A)'s table SIZE and not its bound: at 16000 records the writer's per-record `phases` build measures 6.5388e-3 to 7.2559e-3 s = 6.54 to 7.26 steps of `%10.3f`, while the copy plus the sort at that same size measures 3.8975e-4 to 1.0344e-3 s = 0.39 to 1.03 steps and straddles the one-step bound. Enlarging the table is measured shut -- `WriteJson` holds ~3.3 KB of `nlohmann` per record while it dumps, so a table big enough costs ~1 GB of resident set through an arm that already costs 141 MB and 2.2 s in CI -- and a new wall-clock tolerance is forbidden by [#1668](https://github.com/mudler/vllm.cpp/issues/1668). NOT [#1718](https://github.com/mudler/vllm.cpp/issues/1718), which is the `instrument_seconds` charge-site class. What would settle it is a bound that does not go through the printed format: a structural assertion over the block's position, or a `RenderText` HANDED the wall it prints. Found by the fresh review of [PR #1711](https://github.com/mudler/vllm.cpp/pull/1711) and reproduced by the session that recorded it
Row: LTX25-PHASE-INSTRUMENT
State: UNKNOWN
Kind: bug
GitHub: 1760
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:652`

### Frozen archive evidence

> | [#1760](https://github.com/mudler/vllm.cpp/issues/1760) | `LTX25-PHASE-INSTRUMENT` | The `### 10` console call-site gate holds only the JSON BUILD, not the copy and the sort. `M-SITE-MID` slides the `StderrEnabled()` block BELOW `ByStart(Records())` and `Sum(...)` while leaving it ABOVE the `nlohmann` build -- [#1755](https://github.com/mudler/vllm.cpp/issues/1755)'s own class, the console `WALL` charged with the writer's copy and sort -- and it survived **19 of 20 runs** at `test cases: 8 \| 8 passed` and `assertions: 120 \| 120 passed`, with one run in twenty red at `8 \| 7 passed`. A mutation a gate catches once in twenty is one the gate does not catch. The cause is arm (A)'s table SIZE and not its bound: at 16000 records the writer's per-record `phases` build measures 6.5388e-3 to 7.2559e-3 s = 6.54 to 7.26 steps of `%10.3f`, while the copy plus the sort at that same size measures 3.8975e-4 to 1.0344e-3 s = 0.39 to 1.03 steps and straddles the one-step bound. Enlarging the table is measured shut -- `WriteJson` holds ~3.3 KB of `nlohmann` per record while it dumps, so a table big enough costs ~1 GB of resident set through an arm that already costs 141 MB and 2.2 s in CI -- and a new wall-clock tolerance is forbidden by [#1668](https://github.com/mudler/vllm.cpp/issues/1668). NOT [#1718](https://github.com/mudler/vllm.cpp/issues/1718), which is the `instrument_seconds` charge-site class. What would settle it is a bound that does not go through the printed format: a structural assertion over the block's position, or a `RenderText` HANDED the wall it prints. Found by the fresh review of [PR #1711](https://github.com/mudler/vllm.cpp/pull/1711) and reproduced by the session that recorded it | bug |

## Resolution

-
