ID: ISSUE-GH-2433
Title: ROCm: EXL3 runs through two CPU reference operations
Row: QUANT-EXL3
State: OPEN
Kind: UNKNOWN
GitHub: 2433
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `QUANT-EXL3`
>
> The ROCm model sweep ran `llama32-1b-exl3-3bpw` on `strix:gpu0` (`gfx1151`, ROCm 7.2.4) through the public `vllm-cli` path.
>
> The model completed one greedy token, but provider statistics reported two CPU reference operations:
>
> ```text
> [vt reference-tier] op=CastF16 device=rocm:0 ...
> [vt reference-tier] op=Exl3Gemm device=rocm:0 ...
> ```
>
> The run finished in 1.436 seconds for one output token. That number is diagnostic only. `docs/ROCM.md` forbids a performance result with reference-tier hits.
>
> The BF16 control `llama32-1b-instruct-bf16` completed on the same binary and device with zero reference-tier hits. This isolates the gap to the EXL3 path rather than the Llama loader, paged attention, sampler, or base dense model.
>
> Evidence:
>
> - rc job `43267dc3-52c5-43b2-b6b7-538aff68e6b6`
> - `/mnt/nas_share/rc/rocm-model-sweep-v2/llama32-exl3.err`
> - checkpoint `/workspace/ckpt/llama32-1b-exl3-3bpw`
>
> Acceptance:
>
> - Native ROCm `CastF16` and `Exl3Gemm` pass their CPU-oracle operation gates.
> - The same Llama EXL3 checkpoint completes with zero reference-tier hits.
> - Greedy output matches the BF16 target under the existing EXL3 correctness contract.
> - Repeated warm-leg throughput is recorded against the BF16 control with AMD clock attribution.

## Resolution

-
