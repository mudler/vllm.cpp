ID: ISSUE-GH-1363
Title: Thor reports 30 GiB of swap, all free, measured inside `rc run -d thor:gpu0`, so the "zero swap" half of the box's `THIS BOX REBOOTS INSTEAD OF OOM-KILLING` warning is stale. `vm.overcommit_memory=1` is unchanged and the three 2026-08-11 reboots were observed, so the hazard stands and is not relaxed; unresolved are whether the swap is the host's or a container view, whether it changes the failure mode at all, and when it appeared
Row: BACKEND-CUDA-SM110
State: UNKNOWN
Kind: bug
GitHub: 1363
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:644`

### Frozen archive evidence

> | [#1363](https://github.com/mudler/vllm.cpp/issues/1363) | `BACKEND-CUDA-SM110` | Thor reports 30 GiB of swap, all free, measured inside `rc run -d thor:gpu0`, so the "zero swap" half of the box's `THIS BOX REBOOTS INSTEAD OF OOM-KILLING` warning is stale. `vm.overcommit_memory=1` is unchanged and the three 2026-08-11 reboots were observed, so the hazard stands and is not relaxed; unresolved are whether the swap is the host's or a container view, whether it changes the failure mode at all, and when it appeared | bug |

## Resolution

-
