ID: ISSUE-GH-2315
Title: ENG-RUNTIME-FOREST-A: model-agnostic layout, capture, draft, and evidence seams
Row: MODEL-SPEC
State: OPEN
Kind: UNKNOWN
GitHub: 2315
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-29
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> The runtime carries three mutually exclusive tensor-layout booleans, Qwen3.5-specific activation and draft types through shared model interfaces, and speculative proposals without provider attribution. Stable-ID records also lack a machine-checked evidence descriptor. These seams prevent model-agnostic draft providers and make invalid layout combinations and silent fallback invisible.
>
> ## Scope
>
> - Replace `Tensor.repacked`, `Tensor.q8_0_aligned`, and `Tensor.elem_kn_repacked` with one exhaustive compact `LayoutId` plus total traits.
> - Replace Qwen3.5-specific shared capture types with a caller-owned, allocation-free `ForwardCapturePlan` and generic model capabilities.
> - Add generic model-resource lookup and draft-provider registration, migrate shared embedding/head borrowing and MTP construction, then remove Qwen-specific registry virtuals.
> - Attribute every existing draft proposal to a compact provider identity and expose per-provider proposed and accepted token counters.
> - Validate evidence descriptors through the existing stable-ID record checker rather than adding a second ledger.
>
> ## Acceptance
>
> - Every old layout flag producer, consumer, copy, and test is migrated. Impossible flag combinations are unrepresentable. `Tensor` size does not increase.
> - Shared registry, runner, loader, and speculative interfaces contain no Qwen3.5 capture, resource, or draft-construction types.
> - Capture bindings and attribution storage allocate no memory per decode step. Ownership of borrowed target resources remains explicit and safe.
> - Existing MTP, DFlash, DSpark, and drafter-chain behavior remains reachable from production entry points. Provider-level counters distinguish fallback or non-use from accepted proposals.
> - Evidence descriptor omissions fail the record checker and its mutation tests.
> - Focused CPU tests, the repository gate, and fresh mutation review pass.
>
> ## Out of scope
>
> Segment topology, topology bytecode, exchange planning, expert-union residency, and accelerator-specific graph replay belong to later Runtime Forest phases.
>

## Resolution

-
