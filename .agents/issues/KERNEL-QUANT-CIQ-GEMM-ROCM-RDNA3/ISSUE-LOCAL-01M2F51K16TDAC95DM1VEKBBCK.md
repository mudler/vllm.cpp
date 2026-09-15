ID: ISSUE-LOCAL-01M2F51K16TDAC95DM1VEKBBCK
Title: Classify the RDNA3 WMMA evidence in the permitted per-run layout
Row: KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

The row stores operator and review receipts in nested directories. check-pr-size.py rejects those paths because BENCH_EVIDENCE_RUN permits one run directory followed by a filename. Flatten the receipt filenames, preserve original bytes and source locations, update local links and manifests, and validate the exact integrated diff without changing the checker.

## Resolution

13 September 2026: The landing change flattens all RDNA3 WMMA receipts into the permitted per-run directory. evidence-path-map.json preserves original filenames, byte hashes, and scoped manifest adaptations. Local links and copy manifests resolve the flat paths. Source locations and original measurements remain unchanged. The checker itself is unchanged. Classification of the final integrated diff is recorded with the final records verification receipt.
