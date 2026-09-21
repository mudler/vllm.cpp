ID: ISSUE-LOCAL-01M31EXFQ7NCN3EA6290SH8K2N
Title: ROCm split-KV reduce ignores per-split max when merging partials
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: -

## Problem

PagedAttnDecodeSplitKvReduce in src/vt/rocm/rocm_paged_attn.hip computes m_global from the per-split lse values (line 1318-1325) but the second pass sums o_acc and l_total without rescaling by exp(m_split - m_global); the kernel's own comment ('in practice should track per-split m', line 1334) records the shortcut. The merge is only exact when every split shares the same running max. Found by a fresh reviewer on the HIP 7.1 build fix (#3220); pre-existing since #845 and out of that PR's scope. Fix: mirror vLLM's split-KV merge (paged_attention_v2 reduce kernel) and pin it with a test whose splits have distinct maxima.

## Resolution

-
