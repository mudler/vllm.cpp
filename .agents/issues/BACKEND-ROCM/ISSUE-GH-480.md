ID: ISSUE-GH-480
Title: ROCm: evaluate incremental dual-GPU local-KV sharding for Gemma-4 decode
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 480
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-12
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
> Evaluate an opt-in Gemma-4 ROCm T=1 full-attention decode path that shards the two BF16 KV heads across dual RDNA4 GPUs with local historical KV residency and incremental new-token mirroring.
>
> ## Scope
> - Gemma-4 full-attention layers only
> - `T=1`, one request, BF16 KV, `Hkv=2`, `d=512`
> - GPU0 owns KV head 0; GPU1 owns a persistent local mirror of KV head 1
> - Mirror only newly written K/V entries after initial backfill
> - Run each head against local HBM and merge only peer Q-head outputs
> - Default OFF with synchronous fallback to the existing single-GPU split-KV path
>
> ## Exclusions
> - No FP8 KV in the first experiment
> - No sliding-window layers
> - No remote reads of historical KV
> - No full-cache copy per decode token
> - No production-default change without measured KEEP evidence
>
> ## Gates
> - Build and default/fallback smoke
> - Positive candidate breadcrumb
> - Greedy output identity and Paris/arithmetic quality
> - Short plus long-context paired, order-alternated A/B
> - No GPU faults, page faults, invalid-device errors, NaNs, progressive memory growth, or hidden CPU/offload path
> - KEEP only for at least 10% long-context end-to-end decode improvement with no material short-context regression
>
> ## Evidence origin
> Local architecture audit: `.agents`-external lab reference `gemma4-decode-hotpath-architecture-audit-2026-08-12.md`; implementation must validate source assumptions against the current dirty lab stack before editing.
>

## Resolution

-
