ID: ISSUE-GH-3107
Title: bench: extend matched Strix qualification to c1 c4 and c32
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: UNKNOWN
GitHub: 3107
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-10
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Parent: #3053. Performance campaign: #3076. Owner: current Strix campaign operator.
>
> The developer approved c1, c4, and c32 coverage on 9 September 2026, with a speed target at every tested concurrency. They also approved evenly repeating the original six raw prompts so c32 has enough requests. Keep Qwen3-4B BF16, 128-token greedy decoding, production pinned vLLM, patched SGLang, llama.cpp, and vllm.cpp. No model, precision, oracle-pin, or correctness waiver is included.
>
> The existing harness binds six requests, four request slots, and 8192 KV tokens. Inspect each adapter and its public engine capacity APIs before implementation. Specify a versioned balanced request corpus used identically at c1/c4/c32, required capacity, occupancy evidence, refill/tail behavior, exact output ordering and token counts. Preserve the original six-request evidence as a different workload; do not compare old and expanded throughput as an optimization ratio.
>
> Commit a scoped spec under .agents/specs before fresh test-first implementation. Add CPU regressions and independent mutations for concurrency/capacity, repeated request identities, prompt IDs, settings, missing or reordered outputs, and refusal paths. Require independent review and operator verification, then leased hardware correctness before accepted throughput. Report throughput, latency, and memory separately at each concurrency; retain failures rather than inventing c32 support.
>
> Existing teardown, oracle repeatability, SDK integration, and comparator correctness blockers remain open. This scope does not authorize ignoring zombies, serialized callbacks, eager performance denominators, or claims based on requested rather than actual occupancy.

## Resolution

-
