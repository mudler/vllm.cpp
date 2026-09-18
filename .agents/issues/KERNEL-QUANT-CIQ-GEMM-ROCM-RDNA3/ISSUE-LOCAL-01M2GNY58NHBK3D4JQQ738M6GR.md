ID: ISSUE-LOCAL-01M2GNY58NHBK3D4JQQ738M6GR
Title: Move RDNA3 WMMA raw evidence out of the pull request diff
Row: KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3
State: CLOSED
Kind: documentation
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-14
Closed: 2026-09-14

## Problem

The RDNA3 WMMA pull request includes 236 evidence files and 42,704 evidence lines for a two-file runtime admission change. The developer requested a compact report with the complete raw evidence preserved in a separately accessible, checksum-verified archive. Keep the same PR, implementation, tests, model results, and all qualifications.

## Resolution

14 September 2026: The linked release archive preserves all 236 original evidence files from d6e40c91f634a041c873c7a04516d55c4d05772a. The operator verified anonymous download, the archive checksum, and every original byte before replacement. docs/bench-evidence/rocm-rdna3-quant-wmma/README.md retains the source revision, archive identity, extraction procedure, correctness results, and performance qualifications. The implementation, tests, and validation harness remain byte-identical. Altered-archive, altered-file, and missing-file integrity checks fail as intended and pass after restoration.
