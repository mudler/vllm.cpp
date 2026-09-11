ID: ISSUE-GH-519
Title: Gemma-4 MTP must not use Qwen speculative KV-cache geometry
Row: SPEC-MTP
State: OPEN
Kind: UNKNOWN
GitHub: 519
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-12
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Bug
>
> Gemma-4 external MTP/speculative decoding is routed through a Qwen-specific speculative KV-cache builder. In `src/vllm/entrypoints/model_loader.cpp`, the speculative path hardcodes `MakeQwen3_5KVCacheSpec(...)` whenever speculation is configured, bypassing the target model's registered `ModelInfo::make_kv_cache` capability.
>
> For Gemma-4 this constructs Qwen `fa_draft` geometry instead of Gemma's heterogeneous full/sliding topology (full: Hkv=2, d=512; sliding: Hkv=8, d=256). Live dual-R9700 evidence shows external Gemma MTP accepts large requests and then wedges during the first 8192-token prefill chunk; the identical spec-off path completes a ~20k-token prompt.
>
> ## Required behavior
>
> - Speculative KV construction is model-capability dispatched.
> - Qwen MTP/DFlash retains its existing `fa_draft` behavior unchanged.
> - Gemma external MTP declares/uses no Qwen draft-KV group and reads target KV only.
> - Spec-off behavior remains byte-identical.
> - No logits-shape/allocation, MoE, kernel, or k>1 changes in this fix.
>
> ## Required gates
>
> 1. CPU regression proving Gemma speculation cannot select the Qwen speculative KV builder / `fa_draft` geometry.
> 2. Qwen speculative KV regression proving existing behavior is unchanged.
> 3. Focused and full CPU gates.
> 4. Exclusive gfx1201 live gate: large first-prefill chunk with Gemma external MTP progresses/completes or fails cleanly, never wedges.
> 5. Interrupted SSE/client disconnect releases the request and leaves the engine schedulable.
> 6. Restore and verify the non-MTP `:8010` production recipe after the exclusive gate.
>
> ## Owning row
>
> `SPEC-MTP-FAMILY` (Gemma-4 family support), not completed Qwen `SPEC-MTP` and not k>1 issue #81.
>

## Resolution

-
