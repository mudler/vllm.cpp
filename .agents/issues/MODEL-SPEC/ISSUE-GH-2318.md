ID: ISSUE-GH-2318
Title: ENG-TENSOR-LAYOUT-ID: replace tensor layout flags with LayoutId
Row: MODEL-SPEC
State: OPEN
Kind: UNKNOWN
GitHub: 2318
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
> Owning row: `ENG-TENSOR-LAYOUT-ID`
>
> ## Scope
> Replace the three independent tensor layout booleans with one exhaustive compact `LayoutId` and total layout traits. Migrate every producer, consumer, copy, view, slice, test, and example.
>
> ## Acceptance
> - Invalid layout combinations are unrepresentable.
> - Layout traits are total and compile-time checked.
> - `sizeof(vt::Tensor)` does not increase.
> - Existing CPU and CUDA dispatch semantics remain unchanged.
> - No legacy boolean, alias, or compatibility accessor remains.
> - Focused CPU tests cover layout identity, propagation, and refusal.
>
> ## Out of scope
> No new tensor format or kernel algorithm.
>

## Resolution

-
