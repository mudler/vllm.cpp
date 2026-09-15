ID: ISSUE-LOCAL-01M2HK0GFJDRXXAAA8F0014XQQ
Title: Enable and validate rocWMMA attention prefill on gfx1100
Row: BACKEND-ROCM-RDNA3-WMMA-ATTN
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-15
Closed: -

## Problem

The production BF16 SharedK attention prefill kernel uses portable rocWMMA fragments, but both device compilation and host dispatch exclude gfx1100. Enable only physical gfx1100 after red-first correctness, emitted-code inspection, same-binary performance checks, and public end-to-end validation. The developer requests a single-agent implementation and an MR-ready result.

## Resolution

15 September 2026, branch evidence before landing: gfx1100 uses WMMA prefill
by default. Scalar and default prefill pass the unchanged original 96-token
and expanded 256-token workloads with blocks 16 and 32, in graph and eager
modes. The dedicated decoder and prefill have zero VGPR/SGPR spills and zero
private scratch. Physical traces verify production dispatch and graph reuse.
The Gemma 1B checkpoint-dependent regressions now execute and pass.

The continuation issue ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ carries the
scalar arithmetic, decode latency, host memory, and final measurement repairs.
[Final report](../../../docs/bench-evidence/rocm-rdna3-attention-wmma/README.md).
PR #3195 carries the implementation for independent human review. Keep this
issue open until the work lands. Physical RDNA4 remains unavailable.
