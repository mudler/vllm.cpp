ID: ISSUE-GH-1759
Title: Gemma-4 SWA: per-group physical KV pools and data-plane routing
Row: KV-SLIDING-LOCAL-SPECS
State: OPEN
Kind: UNKNOWN
GitHub: 1759
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-23
Updated: 2026-08-23
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Gemma-4 publishes one oversized FullAttention KV group on current main. Generic sliding infrastructure already exists (`SlidingWindowSpec`, `SlidingWindowManager`, hybrid coordinator, per-layer specs, group-name masks), but all coordinator managers share one `block_pool`. Distinct per-group physical pool sizes, conjunctive per-pool admission, and Gemma data-plane routing of the sliding group's own block table/metadata are absent.
>
> Row: `ROCM-GEMMA4-SWA-PHYSICAL`
>
> This issue tracks the residual only: per-group physical pools plus Gemma publication/data-plane routing. It is not a generic SWA rewrite. Default stays OFF / explicit promotion until CPU and device identity gates. No GPU work on this filing.
>
> Spec-first. Implementation waits on an approved committed spec.
>

## Resolution

-
