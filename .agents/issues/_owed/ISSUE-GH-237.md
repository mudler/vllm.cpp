ID: ISSUE-GH-237
Title: CUDA BF16 MoE resident cache reuses stale pointers after engine teardown
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 237
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-10
Updated: 2026-08-10
Closed: 2026-08-10

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> Repeated `LoadedEngine` construction and destruction in one CUDA process can reuse a `MoeBlockWeights` address while `MoeBf16ResidentFor` retains the previous engine's device-pointer array in a process-lifetime static map. A later engine can therefore execute the BF16 fast MoE path with stale resident pointers.
>
> ## Reproduction
>
> On current main plus issue #206's exact pretokenized benchmark fixture, run the focused tests in this order:
>
> ```sh
> nix develop .#cuda -c ctest --test-dir build-nix-cuda \
>   -R '^(test_engine_core_proc|test_async_llm|test_bench|test_gdn_decode_fused|test_gdn_prefill_conv|test_ops_gdn)$' \
>   --output-on-failure
> ```
>
> The ordered suite intermittently fails `test_bench`: later synthetic Qwen3.5 engines emit corrupted or zero output-token IDs. The target case alone passed 50/50. Setting `VT_MOE_BF16_FAST=0` made the full `test_bench` suite pass 50/50, isolating the fault to the fast resident-weight path.
>
> ## Expected
>
> Resident CUDA MoE state must be owned by, or invalidated with, the engine/weights lifetime. Recreating an engine in the same process must never reuse stale device pointers, and the default fast path must remain enabled.
>
> ## Scope
>
> - Fix cache ownership/lifetime at the cause.
> - Add a deterministic repeated-engine regression test plus a destructive mutation check.
> - Preserve single-engine output and performance behavior.
> - Discovered while rebasing #206; its PR should reference this issue.
>

## Resolution

GitHub records closing pull request #266 (https://github.com/mudler/vllm.cpp/pull/266) merged on 2026-08-11 as commit `ae78871938afe56307d4d14ac690b952d4ebf855`. GitHub closed issue #237 on 2026-08-10.
