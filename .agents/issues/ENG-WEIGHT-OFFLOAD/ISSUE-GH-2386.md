ID: ISSUE-GH-2386
Title: The guard that catches a loader claiming offload support and never consulting the offloader is itself never called
Row: ENG-WEIGHT-OFFLOAD
State: CLOSED
Kind: UNKNOWN
GitHub: 2386
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `VerifyWeightOffloadWasConsulted` has no production caller:
>
> ```
> $ git grep -n "VerifyWeightOffloadWasConsulted" -- src include tests
> include/vllm/model_executor/weight_offloader.h:214   (decl)
> src/vllm/model_executor/weight_offloader.cpp:86      (def)
> tests/vllm/model_executor/test_weight_offloader.cpp:350,357,362,366
> ```
>
> Its sibling `RefuseUnsupportedWeightOffload` **is** wired (`model_loader.cpp:2913`).
>
> ## What it was for, in its own words
>
> `weight_offloader.h:205-209`: *"A model can declare `supports_weight_offload` and then never call `ConsiderWeight`, and the first guard cannot see that lie: the run just offloads nothing."*
>
> `docs/WEIGHT-OFFLOAD.md:173-176` states it in the present tense as shipped behaviour:
>
> > "A model that *claims* support and then never asks the offloader about a single weight is refused too, after load, and reported as a defect in that loader rather than as a configuration error. Zero consulted weights is the only count that can prove that particular lie."
>
> ## Current blast radius, and why it will not stay this small
>
> `supports_weight_offload` defaults to `false` (`model_registry.h:723`) and **no registration sets it true**, so `RefuseUnsupportedWeightOffload` fires first on every configured offload and this second guard is currently unreachable-but-vacuous rather than unreachable-and-wrong.
>
> The moment anyone wires a loader and flips that flag, the promised post-load check silently does not happen — and the failure it guards is by construction silent, so nothing else would report it. The guard is exactly the kind that is only ever needed on the day it is missing.
>
> This is the "nothing lands dead" rule failing on the enforcement mechanism for a related lie, which is why it is worth an issue rather than a note.
>
> Row: `ENG-WEIGHT-OFFLOAD`

## Resolution

Commit `49827821c38e70f4c58907624307431b110f38a1` wires the post-load offload-consultation guard described by this issue; GitHub closed it as COMPLETED on 2026-08-31.
