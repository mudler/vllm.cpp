ID: ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ
Title: Resolve Gemma 3 4B differences on the expanded primary token gate
Row: BACKEND-ROCM-RDNA3-WMMA-ATTN
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-15
Closed: -

## Problem

Pinned vLLM e126687a9 reproduces all eight distinct 518/1207-token prompts. After linear RoPE and BF16 head corrections, gfx1100 scalar attention differs on 3 of 8 outputs and SharedK WMMA differs on 4 of 8. The original 96-token workload passes both. The broader failures include non-tied primary logits. Retain the full 256-token gate, locate the first intermediate divergence, and keep gfx1100 WMMA opt-in until parity passes. Do not accept this workload as a performance comparison.

## Resolution

15 September 2026, branch evidence before landing: default gfx1100 WMMA and
scalar prefill pass the original 96-token and expanded 256-token gates with
block sizes 16 and 32. Graph and eager controls pass all 16 configurations,
including warm-ups, for 3072 exact output tokens. The scalar repair uses the
measured gfx1100 BF16 DOT2 arithmetic and preserves fully masked tile state.

The dedicated decoder has zero scratch in all four compiled templates and
passes 20 byte-exact primary fixtures. The integrated 20-test regression set
executes the downloaded Gemma 1B checkpoint. Incremental weight staging and
device RoPE construction reduce sampled host memory to about 2 GB. Release
comparisons clear the measured throughput and latency floors. Block-16 decode
is at parity, without a meaningful speed-gain claim. Graph dispatch and input
refresh mutations fail their gates. A live-allocation canary probe detects
removed scratch pinning that the token-only workload did not detect.

[Measured report](../../../docs/bench-evidence/rocm-rdna3-attention-wmma/README.md).
PR #3195 carries the implementation for independent human review. The branch
integrates main ea6da1f80 and retains the superseded unoptimized evidence.
Keep this issue open until the work lands. Physical RDNA4 execution remains
unavailable. Final repetitions pass on one unchanged binary: block-16 decode
is 66.58 versus 66.48 tokens/s and block-32 decode is 70.01 versus 44.65.
Full preflight passes with 657 affected host translation units. The final
HIP-only readback change passes target compilation, repeated regressions,
and the complete sixteen-configuration model gate.
