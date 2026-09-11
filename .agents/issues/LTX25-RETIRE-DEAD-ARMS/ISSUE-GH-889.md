ID: ISSUE-GH-889
Title: `kBetaScheduler` shipped as a REACHABLE refusal in the header, `docs/FEATURES.md` and `docs/USAGE.md` with zero product callers: its site sits inside `Ltx2Schedule`, which nothing calls, and the engine calls `Ltx2SigmaSchedule` directly. Upstream constructs `BetaScheduler` nowhere either — all seven ltx-pipelines entry points hard-code `LTX2Scheduler()` — so it is reclassified as a marker rather than wired
Row: LTX25-RETIRE-DEAD-ARMS
State: UNKNOWN
Kind: bug
GitHub: 889
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:233`

### Frozen archive evidence

> | [#889](https://github.com/mudler/vllm.cpp/issues/889) | `LTX25-RETIRE-DEAD-ARMS` | `kBetaScheduler` shipped as a REACHABLE refusal in the header, `docs/FEATURES.md` and `docs/USAGE.md` with zero product callers: its site sits inside `Ltx2Schedule`, which nothing calls, and the engine calls `Ltx2SigmaSchedule` directly. Upstream constructs `BetaScheduler` nowhere either — all seven ltx-pipelines entry points hard-code `LTX2Scheduler()` — so it is reclassified as a marker rather than wired | bug |

## Resolution

-
