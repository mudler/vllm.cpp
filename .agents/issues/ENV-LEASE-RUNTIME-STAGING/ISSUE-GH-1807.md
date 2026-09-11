ID: ISSUE-GH-1807
Title: A leased `rc` job read its gate checkpoint over CIFS from `/workspace` and every job hand-wrote its own existence-check-then-`cp -a` guard, which accepts a truncated shard a killed copy left behind (`/mnt/nas_share/rc/dedup-gate2/build72.sh:118-125`). Developer direction 2026-08-23: copy NAS -> local once, idempotently. FIXED IN FLOW: `scripts/rc-stage-checkpoint.sh` (manifest-defined completeness via `SHA256SUMS`, marker+size fast path that reads no payload, resumable `.part` copy, post-copy verify, refuses a directory with no manifest), `tests/scripts/test_rc_stage_checkpoint.py` (11 hermetic cases, registered in preflight), and the staging paragraph in `.agents/environment.md`. The two 35B gate checkpoints are staged under `/mnt/nas_share/rc/ckpt/` for `GDN-MOE-PACKED-BA` (#1169)
Row: ENV-LEASE-RUNTIME-STAGING
State: UNKNOWN
Kind: gap
GitHub: 1807
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:668`

### Frozen archive evidence

> | [#1807](https://github.com/mudler/vllm.cpp/issues/1807) | `ENV-LEASE-RUNTIME-STAGING` | A leased `rc` job read its gate checkpoint over CIFS from `/workspace` and every job hand-wrote its own existence-check-then-`cp -a` guard, which accepts a truncated shard a killed copy left behind (`/mnt/nas_share/rc/dedup-gate2/build72.sh:118-125`). Developer direction 2026-08-23: copy NAS -> local once, idempotently. FIXED IN FLOW: `scripts/rc-stage-checkpoint.sh` (manifest-defined completeness via `SHA256SUMS`, marker+size fast path that reads no payload, resumable `.part` copy, post-copy verify, refuses a directory with no manifest), `tests/scripts/test_rc_stage_checkpoint.py` (11 hermetic cases, registered in preflight), and the staging paragraph in `.agents/environment.md`. The two 35B gate checkpoints are staged under `/mnt/nas_share/rc/ckpt/` for `GDN-MOE-PACKED-BA` (#1169) | gap |

## Resolution

-
