ID: ISSUE-GH-2319
Title: ENG-MODEL-CAPTURE-RESOURCES: generalize capture and draft resources
Row: MODEL-SPEC
State: OPEN
Kind: UNKNOWN
GitHub: 2319
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
> Owning row: `ENG-MODEL-CAPTURE-RESOURCES`
>
> ## Scope
> Replace Qwen-specific shared model-registry contracts with a generic prepared forward-capture plan, explicit model capture capabilities, typed model-resource lookup with owner retention, and load-time draft-provider registration. Migrate MTP, DFlash, DSpark, shared embedding/head borrowing, and MTP draft construction in one clean cutover.
>
> ## Acceptance
> - Shared runtime headers expose no Qwen-family capture, weight, or draft types.
> - Empty capture plans are default-inert.
> - Prepared captures add no per-step host allocation and no string/map lookup.
> - Resource handles retain the actual storage owner or refuse borrowing.
> - MTP draft construction resolves through the generic provider registry.
> - Every old virtual and attachment path is removed.
> - Focused CPU tests cover capabilities, validation, lifetime, and registration.
>
> ## Out of scope
> No new draft algorithm and no drafter-chain policy.
>

## Resolution

-
