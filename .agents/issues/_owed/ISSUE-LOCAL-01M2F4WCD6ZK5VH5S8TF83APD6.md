ID: ISSUE-LOCAL-01M2F4WCD6ZK5VH5S8TF83APD6
Title: Resolve the remaining gfx1100 Qwen3.5-4B Q4_K_M performance gaps
Row: -
State: OPEN
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

The completed RDNA3 WMMA experiment preserves exact tokens and improves its recorded same-binary prefill metric. Native whole-workload throughput and decode remain below the task-pinned primary and llama.cpp comparisons. Tail latency, sampled memory against llama.cpp, matching oracle traces, comparable timing windows, and accepted clock attribution remain unresolved. Establish the missing measurements and close every below-floor axis on the identical retained workload. Do not treat the architecture-admission result as full-model parity or as a performance ceiling.

The next traceable candidates are Gated DeltaNet prefill and scalar remainder work at input lengths 183 and 174. Native `src/vt/rocm/rocm_gdn_scan.hip:55-104` runs a serial token recurrence for prefill and decode, and the native trace contains `GdnScanK`. At the task-pinned primary revision, `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:240-324` selects a chunk-prefill backend whose native method calls `fla_chunk_gated_delta_rule`. These source differences are hypotheses. Matching generated oracle traces and accepted clock windows must establish the executed paths and their costs before ranking them.

## Resolution

-
