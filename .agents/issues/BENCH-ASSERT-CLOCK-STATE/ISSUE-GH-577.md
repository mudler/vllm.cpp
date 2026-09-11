ID: ISSUE-GH-577
Title: Our SSE keepalive (`VT_SERVER_SSE_PING_S`, default 15 s) can drop the SLOWEST requests from benchmark metrics, flattering our own numbers: it made 27B c16 VOID in the 2026-08-13 pin series (93/96 against 96/96)
Row: BENCH-ASSERT-CLOCK-STATE
State: UNKNOWN
Kind: bug
GitHub: 577
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:38`

### Frozen archive evidence

> | [#577](https://github.com/mudler/vllm.cpp/issues/577) | `BENCH-ASSERT-CLOCK-STATE` | Our SSE keepalive (`VT_SERVER_SSE_PING_S`, default 15 s) can drop the SLOWEST requests from benchmark metrics, flattering our own numbers: it made 27B c16 VOID in the 2026-08-13 pin series (93/96 against 96/96) | bug |

## Resolution

-
