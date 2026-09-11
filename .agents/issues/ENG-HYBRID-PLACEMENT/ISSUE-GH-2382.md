ID: ISSUE-GH-2382
Title: Hybrid offload places NOTHING: the resolved MoE placement plan is never installed
Row: ENG-HYBRID-PLACEMENT
State: CLOSED
Kind: UNKNOWN
GitHub: 2382
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Re-filed. The original was opened by an automation account that has since been suspended, which HIDES its content rather than deleting it, so the earlier number 404s for every other reader and cannot serve as the owning record. The code history is unaffected.
>
> ## What
>
> `SetActiveMoePlacementPlan` is called by **nothing in `src/`** — only its declaration, its definition, and a unit test. `MoePlacementPlan::Resolve` likewise has no production call site, and the only production consumer of `ResolvePlacementOverrides()` is `ReportDevicePlacement` (`model_loader.cpp`), which **prints and returns**.
>
> So `ActiveMoePlacementPlan()` returns the default on every load, `PlacesAnything()` is always false, and the `RunMoePlaced` / `RunMoePlacedPair` seam is **inert in production on all six wired architecture families**. No expert is ever placed on the CPU.
>
> ## Why it survived three work items
>
> The feature looks complete from every angle that was checked: the config parses (tested), the plan resolves (tested), and the loader **prints** `engine: device placement: N layers on cpu`, so an operator running `VT_CPU_MOE=1` sees the placement confirmed on stderr.
>
> That announcement is the trap — it reports the *resolved* plan, not the *installed* one. A reassuring log line is worse than silence.
>
> And a token gate cannot see it either: with nothing placed, the placed arm is byte-identical to the unplaced arm, so the end-to-end comparison this row owed would have **passed for the wrong reason**.
>
> W2's own comment names the seam: *"W2 RESOLVES AND REPORTS; IT MOVES NOTHING ... W3 owns the routing that reads it."* W3 built the routing and never added the call between them.
>
> ## Fix
>
> Install the plan in `LoadedEngine::FromModelDir` at both config-parse sites (GGUF and safetensors), where the engine device and `num_hidden_layers` are both known and still ahead of all weight I/O — required rather than tidy, since `ResidentWeight` aliases host bytes on a CPU `Dev` and uploads otherwise, so a later install would pay the round trip the placement exists to avoid.
>
> Covered by a reachability test entering through `FromModelDir` rather than constructing the plan. The reachability mutation deletes both call sites and requires that suite RED.
>
> ## Status
>
> Fixed by the PR that closes this issue, which also corrects the gate's invariant: the placed and unplaced arms cannot be token-identical, because a placed layer runs the CPU MoE kernels and this project's cross-device bar for reducing arithmetic is NMSE <= 5e-4, not bitwise equality.
>
> Row: `ENGINE-HYBRID-PLACEMENT`

## Resolution

GitHub records closing pull request #2378 (https://github.com/mudler/vllm.cpp/pull/2378) merged on 2026-08-31 as commit `5f230020fff11c66302810506cfd991e202d52a8`. GitHub closed issue #2382 on 2026-08-31.
