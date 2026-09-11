ID: ISSUE-GH-1469
Title: **`test_audit_live_rows` is RED on `main`.** Measured on a detached worktree at `origin/main` `cae8ace0c` with nothing applied: `test_shipped_record_has_no_abandoned_active_row` fails with `stale ACTIVE rows remain: ['KERNEL-DFLASH2-GROUPED-CONV']`, 57 tests, 1 failure, and `scripts/agent-preflight.sh` reports both `audit-live-rows` and `test_audit_live_rows` as FAILED. `scripts/audit-live-rows.py` alone returns rc 0 and prints `245 live rows; 1 abandoned ACTIVE`, so the count is the gate and the script's exit status does not carry it. The row is `.agents/kernel-matrix.md:143`, state `ACTIVE`, claim `CLAIM-SPEC-DFLASH2-W2`, and its implementation landed on `main` in `028438e68` (SPEC-DFLASH2 W2, [#1314](https://github.com/mudler/vllm.cpp/issues/1314), PR #1465) - a row still claimed as in-progress whose work is already reachable on `main`. NOT FIXED IN FLOW: the kernel matrix, the row and the DFlash2 claim are outside the authority of the row that found it (`LTX25-DEVICE-RESIDENCY` W0, PR #1441), and a lifecycle move owes `docs/STATUS.md`, `docs/BENCHMARKS.md` and the moved row spec's `## Now`. Found while running the full preflight for that row's gate, where it is an inherited red
Row: KERNEL-DFLASH2-GROUPED-CONV
State: UNKNOWN
Kind: bug
GitHub: 1469
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:511`

### Frozen archive evidence

> | [#1469](https://github.com/mudler/vllm.cpp/issues/1469) | `KERNEL-DFLASH2-GROUPED-CONV` | **`test_audit_live_rows` is RED on `main`.** Measured on a detached worktree at `origin/main` `cae8ace0c` with nothing applied: `test_shipped_record_has_no_abandoned_active_row` fails with `stale ACTIVE rows remain: ['KERNEL-DFLASH2-GROUPED-CONV']`, 57 tests, 1 failure, and `scripts/agent-preflight.sh` reports both `audit-live-rows` and `test_audit_live_rows` as FAILED. `scripts/audit-live-rows.py` alone returns rc 0 and prints `245 live rows; 1 abandoned ACTIVE`, so the count is the gate and the script's exit status does not carry it. The row is `.agents/kernel-matrix.md:143`, state `ACTIVE`, claim `CLAIM-SPEC-DFLASH2-W2`, and its implementation landed on `main` in `028438e68` (SPEC-DFLASH2 W2, [#1314](https://github.com/mudler/vllm.cpp/issues/1314), PR #1465) - a row still claimed as in-progress whose work is already reachable on `main`. NOT FIXED IN FLOW: the kernel matrix, the row and the DFlash2 claim are outside the authority of the row that found it (`LTX25-DEVICE-RESIDENCY` W0, PR #1441), and a lifecycle move owes `docs/STATUS.md`, `docs/BENCHMARKS.md` and the moved row spec's `## Now`. Found while running the full preflight for that row's gate, where it is an inherited red | bug |

## Resolution

-
