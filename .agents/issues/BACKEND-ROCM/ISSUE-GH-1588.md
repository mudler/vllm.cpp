ID: ISSUE-GH-1588
Title: Characterize Qwen3.5-0.8B CPU against ROCm numerics on gfx1100
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 1588
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-21
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Recorded gap
>
> `.agents/backend-matrix.md` records that Qwen3.5-0.8B runs all-native on
> gfx1100 and that its CPU/ROCm numerical characterization is open. This issue
> owns that characterization.
>
> ## Scope
>
> 1. Run the model end to end on gfx1100 through the ROCm device path and the
>    CPU reference path on identical weights, prompts, and greedy sampling.
> 2. Compare numerics at named layer boundaries, not only at the logits:
>    residual stream, attention output, MLP output, and KV state.
> 3. Audit dtype polarity per `.agents/porting.md`: an `f32` buffer where the
>    reference resolves bf16 passes every token gate and moves twice the
>    bytes. Name each buffer that is wider than upstream and what it costs.
> 4. Record per-boundary maximum deviation and the accepted tolerance in the
>    owning row spec.
>
> ## Blocker
>
> `CHECKPOINT_ROOT` is empty in `.env`, so no checkpoint location is
> recorded. Fetching the checkpoint needs explicit authority from the
> developer. The gate models runnable on this box stay `PENDING` until a
> checkpoint root exists.
>
> ## Gate order
>
> Correctness comes first. Any throughput number for this model waits until
> the declared token gate passes.
>

## Resolution

-
