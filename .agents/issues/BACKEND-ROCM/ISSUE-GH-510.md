ID: ISSUE-GH-510
Title: ROCm: harden compact peer-local KV prefill for Gemma-4
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 510
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-12
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
> Extract and harden the already-validated, opt-in Gemma-4 ROCm Phase 1.1 compact peer-local KV **prefill** path on current main for dual discrete AMD GPUs.
>
> ## Scope
> - Gemma-4 full-attention prefill only
> - BF16 KV, `T=1` request, `Hq=16`, `Hkv=2`, `d=512`
> - Compact active block-table-selected K/V head 1 on the compute GPU
> - Transfer only packed K/V, packed Q heads 8-15, and compact metadata to the peer GPU
> - Execute peer-assigned attention against local peer HBM
> - Merge only packed peer output heads
> - ROCm provider implementation below the portable paged-attention seam
> - Default OFF; existing path remains unchanged when ineligible or disabled
>
> ## Existing evidence
> The dirty lab prototype passed forward+reverse fair-warm A/B on dual R9700, `PREFIX_CACHE=0`, identical unique prompt corpus:
>
> | depth | KEEP median | Phase 1.1 median | gain |
> |---|---:|---:|---:|
> | ~3k | 2074.7 | 2169.5 | +4.57% |
> | ~11k | 1992.4 | 2334.1 | +17.15% |
> | ~18k | 1692.4 | 2137.5 | +26.30% |
>
> Paris, arithmetic, decode smoke, positive execution breadcrumb, and GPU fault gates passed. Modeled transfer at seq=8192 fell by ~14.4x. This issue does not treat the dirty prototype as upstream-ready; it tracks clean extraction and hardening on current main.
>
> ## Required hardening
> - Factor pure eligibility and active-block packing logic for CPU/unit coverage
> - Cover shuffled/sparse blocks, page boundaries, invalid metadata, and byte budgets
> - Persistent resource lifecycle: reuse, growth, teardown, repeated sequential requests
> - Restore compute device on every pre-submit exit
> - Clear only known benign sticky peer-access errors
> - Pre-submit synchronous setup failure may fall back unchanged
> - After any peer copy/kernel submission, failure is fatal; never double-submit the ordinary path
> - Positive execution breadcrumb/counters; dispatch miss invalidates candidate measurements
> - Concurrent-serve non-selection coverage; no progressive VRAM/host-RAM growth
>
> ## Exclusions
> - No sliding-window layers
> - No FP8 KV
> - No decode KV sharding (tracked separately by #480)
> - No c>1 generalization
> - No arbitrary Hkv/d geometry
> - No custom scheduler, serving loop, or model-layer HIP dependency
> - No product-default change in the extraction PR
>
> ## Gates
> 1. RED-first focused tests for eligibility/packing/lifecycle/fallback guarantees
> 2. HIP build plus applicable repository gates
> 3. Paris, arithmetic, READY, decode and tool-call smoke
> 4. Repeated sequential and concurrent non-selection reliability probes
> 5. Forward+reverse/interleaved fair A/B at ~3k/~11k/~18k on current main with `PREFIX_CACHE=0`
> 6. Positive candidate breadcrumb and clean engine-fatal/invalid-device/page-fault/GPU-fault scan
> 7. Restore the exact pre-test `:8010` owner/config automatically
>
> ## KEEP policy
> Retain code only if current-main confirmation preserves correctness and materially reproduces the structural gain. Keep the feature opt-in. Default policy is a separate decision after lifecycle evidence.

## Resolution

-
