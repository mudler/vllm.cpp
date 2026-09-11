ID: ISSUE-GH-2250
Title: Lever #2: measure what bounds `QuantDotGemmGroupedKernel` (Q4_K/Q5_K, 62.1% of Laguna decode GPU) BEFORE tuning it. W11 labelled it "BW-tuning", but the sibling `QuantDotGemmQ8_0Kernel` was measured LATENCY- and LSU-pipe-bound with five structural levers refuted and a recorded floor, so the label is not evidence. W1 is `ncu` counters on the decode path only
Row: MODEL-TEXT-laguna-laguna-for-causal-lm
State: UNKNOWN
Kind: perf
GitHub: 2250
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:875`

### Frozen archive evidence

> | [#2250](https://github.com/mudler/vllm.cpp/issues/2250) | `MODEL-TEXT-laguna-laguna-for-causal-lm` | Lever #2: measure what bounds `QuantDotGemmGroupedKernel` (Q4_K/Q5_K, 62.1% of Laguna decode GPU) BEFORE tuning it. W11 labelled it "BW-tuning", but the sibling `QuantDotGemmQ8_0Kernel` was measured LATENCY- and LSU-pipe-bound with five structural levers refuted and a recorded floor, so the label is not evidence. W1 is `ncu` counters on the decode path only | perf |

## Resolution

-
