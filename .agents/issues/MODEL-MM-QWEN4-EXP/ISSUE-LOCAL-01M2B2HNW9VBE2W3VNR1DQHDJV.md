ID: ISSUE-LOCAL-01M2B2HNW9VBE2W3VNR1DQHDJV
Title: include/vt/ops.h terse availability tags understate the registered arms
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

`include/vt/ops.h` carries registration claims in two different shapes. The
first is a full sentence — "Registered on kCPU (...) and on kCUDA (...)" — and
MODEL-MM-QWEN4-EXP-ROCM-W3 (PR #3166) swept that population across BOTH comment
sites, the OpId enum block and the function declarations, and corrected every
falsified one. The second shape is a terse tag appended to a shapes paragraph:
`CPU + CUDA.` or `CUDA-only.` with nothing else said. That population was NOT
swept, and this issue owns it.

The tags are falsified the same way the sentences were: a device arm landed
later and nobody came back to the header. Checked against the `RegisterOp` call
sites at `e5531179a` + the W3 comment sweep (a whole-file scan that joins
multi-line `RegisterOp(` calls and includes `.mm`, since a naive one-line grep
misses arms such as `kSigmoidGateBf16`'s kCUDA arm at
`src/vt/cuda/cuda_glue.cu:475`), ten tags in `ops.h` are wrong:

| ops.h | op | tag says | actually registered on |
|---|---|---|---|
| :3368 | `kMoeSiluMul` | CPU + CUDA | kCPU kCUDA kROCM kTENSTORRENT |
| :3612 | `kLayerNorm` | CPU + CUDA | kCPU kCUDA kROCM kVULKAN kTENSTORRENT kMETAL |
| :3619 | `kRelu` | CPU + CUDA | kCPU kCUDA kROCM kVULKAN kTENSTORRENT kMETAL |
| :3636 | `kAdd` | CPU + CUDA | kCPU kCUDA kROCM kVULKAN kTENSTORRENT kMETAL |
| :3693 | `kFusedNormRope` | CPU + CUDA | kCPU kCUDA kROCM |
| :3714 | `kRopeCosSinCache` | CPU + CUDA | kCPU kCUDA kROCM kTENSTORRENT kMETAL |
| :3739 | `kAttnQkNormRopeGate` | CPU + CUDA | kCPU kCUDA kROCM kVULKAN kTENSTORRENT |
| :5638 | the V1 sampler family | "Every op is correctness-grade CPU + CUDA" | every one of them also on kROCM (`src/vt/rocm/rocm_ops.hip:263-288`) |
| :5686 | `kCastBf16` | CPU + CUDA | kCPU kCUDA kROCM kVULKAN kTENSTORRENT kMETAL |
| :5790 | `kQkvSplit` | CPU + CUDA | kCPU kCUDA kROCM kVULKAN kTENSTORRENT kMETAL |

The fresh reviewer of PR #3166 sampled twelve tags and found nine falsified,
including `kGeluTanh` (three arms) which this scan's per-line pattern did not
reach, so the ten above are a LOWER BOUND and the fix must re-scan rather than
work this table.

The same class also exists outside the header: `tests/vt/test_ops_glue.cpp:284`
says "all four of its backends" about an op that has more.

WHY THIS IS NOT PART OF #3166. These tags span many unrelated rows — MoE,
sampler, RoPE, elementwise, cast, the Qwen3-VL vision activations — and
correcting them is that many rows' blast radius, not one ROCm wave's. PR #3166
states in its body that its sweep was scoped to full-sentence registration
claims and that this issue owns the terse tags.

ADJACENT, ALSO NOT FIXED THERE, ALSO OWNED HERE. Four declaration-site
paragraphs still speak of a device arm in the FUTURE tense after that arm
landed: `Qwen4ExpPleConv` ("a CUDA arm may accumulate in f32 and must be gated
against the oracle to say so"), `Qwen4ExpPleGate` ("A CUDA arm that evaluates in
f32 will not inherit that"), `Qwen4ExpGatedResidual` ("A CUDA arm therefore
cannot simply f32-accumulate this reduction"), and `RmsNormGroup`. These are
precision-contract prose rather than registration claims, so they were outside
#3166's scope, but each one's OpId enum paragraph now records the landed kCUDA
and kROCM arms and what accumulator width they actually chose, so the header
reads two ways about the same kernels.

HOW TO CLOSE IT. Do not hand-audit. Build the `(OpId, DeviceType)` map from the
`RegisterOp` call sites (multi-line-aware, `.mm` included, tests and examples
excluded), then hold every availability claim in `ops.h` against it. A claim
that a scan cannot attribute to an OpId is a claim to rewrite, not to skip.

## Resolution

-
