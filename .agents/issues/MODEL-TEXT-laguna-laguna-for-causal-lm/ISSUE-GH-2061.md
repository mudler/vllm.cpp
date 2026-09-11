ID: ISSUE-GH-2061
Title: Laguna's grouped MoE issues two `LqGemmGrouped` calls over the same activation, so it quantizes to Q8_K twice where `vt::MoeGateUpSwiGLUGrouped` quantizes once — W11 measured `QuantizeQ8KKernel` at 12.4% of decode GPU. Also a shared-seam obligation, since AGENTS.md routes mergeable MLP projections through the fused group. Bounded by whether a DYNAMIC UD quant gives both expert towers the same block-quant dtype, which W1 measures before any code
Row: MODEL-TEXT-laguna-laguna-for-causal-lm
State: UNKNOWN
Kind: perf
GitHub: 2061
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:794`

### Frozen archive evidence

> | [#2061](https://github.com/mudler/vllm.cpp/issues/2061) | `MODEL-TEXT-laguna-laguna-for-causal-lm` | Laguna's grouped MoE issues two `LqGemmGrouped` calls over the same activation, so it quantizes to Q8_K twice where `vt::MoeGateUpSwiGLUGrouped` quantizes once — W11 measured `QuantizeQ8KKernel` at 12.4% of decode GPU. Also a shared-seam obligation, since AGENTS.md routes mergeable MLP projections through the fused group. Bounded by whether a DYNAMIC UD quant gives both expert towers the same block-quant dtype, which W1 measures before any code | perf |

## Resolution

-
