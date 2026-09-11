ID: ISSUE-GH-618
Title: `test_cpu_x86_llamacpp_floor`'s contended-leg case is load-dependent: at loadavg 63 the harness exits `NO_QUIET_WINDOW` (4) instead of `GIVING_UP` (2), so the guarantee goes untested and the red reads as a defect in whatever diff is in flight
Row: BACKEND-GATE-CPU-LLAMACPP
State: UNKNOWN
Kind: bug
GitHub: 618
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:20`

### Frozen archive evidence

> | [#618](https://github.com/mudler/vllm.cpp/issues/618) | `BACKEND-GATE-CPU-LLAMACPP` | `test_cpu_x86_llamacpp_floor`'s contended-leg case is load-dependent: at loadavg 63 the harness exits `NO_QUIET_WINDOW` (4) instead of `GIVING_UP` (2), so the guarantee goes untested and the red reads as a defect in whatever diff is in flight | bug |

## Resolution

-
