ID: ISSUE-GH-1504
Title: test_audit_live_rows is red on main: KERNEL-DFLASH2-SELECTOR-EDGES and KERNEL-TOPK-PAIRS stay ACTIVE after their implementation landed
Row: SPEC-DFLASH2
State: OPEN
Kind: bug
GitHub: 1504
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `SPEC-DFLASH2`
>
> Measured at `origin/main` `cffe59b0` (clean detached worktree, nothing applied):
>
> ```
> $ python3 scripts/audit-live-rows.py
> 248 live rows; 2 abandoned ACTIVE; ...
> $ python3 -m pytest tests/scripts/test_audit_live_rows.py
> AssertionError: Lists differ: ['KERNEL-DFLASH2-SELECTOR-EDGES', 'KERNEL-TOPK-PAIRS'] != []
> ```
>
> Both rows (.agents/kernel-matrix.md:144-145) record their implementation as landed and gated on 2026-08-20 — `KERNEL-DFLASH2-SELECTOR-EDGES` "CPU GATE GREEN ... CUDA VERIFIED", `KERNEL-TOPK-PAIRS` "CPU GATE GREEN ... CUDA RUN" — via SPEC-DFLASH2 W3 (af25bd25 / PR #1497). The rows still carry `ACTIVE`, so `scripts/agent-preflight.sh` fails `test_audit_live_rows` (and `audit-live-rows`) on every tree based at current main.
>
> Fourth instance of the #1469 class (stale ACTIVE after landing; previous: #1469 itself). NOT FIXED IN FLOW by the finder: the repair is a lifecycle move on rows owned by `SPEC-DFLASH2` (#1314), and a lifecycle move owes `docs/STATUS.md`, `docs/BENCHMARKS.md` and the moved row spec's `## Now` — writes outside the finding row's authority. `KERNEL-TOPK-PAIRS` also carries an open CUDA NaN-tie disagreement (#1489) on one row of its device case, which the state move should reconcile or explicitly carry as its open gate.
>
> Found while rebasing row BACKEND-TENSTORRENT-HOST-FREE-FORWARD onto main (its preflight inherits the red); filed in flow.

## Resolution

-
