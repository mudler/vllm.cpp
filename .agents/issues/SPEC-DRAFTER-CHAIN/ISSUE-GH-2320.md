ID: ISSUE-GH-2320
Title: SPEC-PROPOSAL-ATTRIBUTION: attribute draft acceptance by provider
Row: SPEC-DRAFTER-CHAIN
State: OPEN
Kind: UNKNOWN
GitHub: 2320
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-29
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Parent: #2315
>
> Owning row: `SPEC-PROPOSAL-ATTRIBUTION`
>
> ## Scope
> Carry a compact proposal-source identity through the existing single-drafter request, scheduler, input-batch, sampling, and accounting path. Record fixed-storage per-provider invocation, proposal, acceptance, and fallthrough counters.
>
> ## Acceptance
> - Every scheduled draft token has one explicit provider identity.
> - Accepted tokens are charged to the provider that proposed them.
> - Empty proposal/fallthrough is distinguishable from provider non-use.
> - Aggregate counters retain current meaning.
> - No string, map, or per-step heap allocation is added to hot accounting paths.
> - The existing one-drafter policy and rejection math remain unchanged.
> - Focused CPU tests prove attribution and counter boundaries.
>
> ## Out of scope
> No multi-provider chain resolution and no changed acceptance algorithm.
>

## Resolution

-
