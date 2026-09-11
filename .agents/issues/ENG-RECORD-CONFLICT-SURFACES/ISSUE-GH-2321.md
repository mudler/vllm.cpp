ID: ISSUE-GH-2321
Title: ENG-EVIDENCE-DESCRIPTOR: validate optimization evidence in stable specs
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: UNKNOWN
GitHub: 2321
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
> Owning row: `ENG-EVIDENCE-DESCRIPTOR`
>
> ## Scope
> Add a machine-checkable optimization evidence descriptor to stable-ID specs and integrate its validation into `scripts/check-agent-record.py`. Encode source posture, capability predicate, firing signal, default state, oracle, benchmark signature, fallback/refusal, and nonclaims without adding a second ledger.
>
> ## Acceptance
> - Descriptor identity must match the owning stable row.
> - New descriptors reject missing or placeholder required fields.
> - Mutation tests prove checks for firing signal, source posture, fallback/refusal, default state, oracle, and benchmark signature.
> - Representative MIRROR, ADAPT, and SYNTHESIS records are encoded.
> - Existing unrelated stable rows remain valid.
>
> ## Out of scope
> No new benchmark runner and no rewrite of historical specs.
>

## Resolution

-
