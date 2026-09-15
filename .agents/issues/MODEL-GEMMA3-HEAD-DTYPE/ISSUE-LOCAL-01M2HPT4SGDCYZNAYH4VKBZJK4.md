ID: ISSUE-LOCAL-01M2HPT4SGDCYZNAYH4VKBZJK4
Title: Project Gemma 3 logits in the resolved BF16 model dtype
Row: MODEL-GEMMA3-HEAD-DTYPE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-15
Closed: -

## Problem

The final Gemma 3 projection writes FP32 directly. Pinned vLLM e126687a9 uses model-dtype BF16 and widens only in the sampler. Real 4B prompts produce different greedy IDs at close logits. Match the projection dtype, retain FP32 at the existing runner output seam, and rerun the unmodified primary token workloads.

## Resolution

15 September 2026, branch evidence before landing. The projection now stores BF16 for tied and untied heads, then widens at the
existing sampler boundary. The focused dtype assertions and combined public
model gates pass.

The original 96-token and expanded 256-token workloads pass with cache blocks
16 and 32, scalar and default WMMA prefill, and graph and eager decode.
[Combined evidence](../../../docs/bench-evidence/rocm-rdna3-attention-wmma/README.md).
PR #3195 carries this prerequisite. Keep the issue open until landing.
