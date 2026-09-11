ID: ISSUE-GH-2434
Title: The connector leaf was re-split by the attention hoist, and its repair is still unowned
Row: MODEL-MM
State: OPEN
Kind: UNKNOWN
GitHub: 2434
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `LTX25-CONNECTOR-REPAIR`
>
> `LTX25-TEXT-COND-DEVICE` (#2354) measured `conditioning.connector.compute` +
> `guiders.connector.compute` at 224.882 s, 43.52% of an LTX-2.5 render, and
> `LTX25-CONNECTOR-GEMM` decomposed that leaf on GB10 into attention 66.5%,
> GEMM 24.8%, residue 8.7%. `VT-CPU-ELEM-DISPATCH` (#2376) then hoisted the
> per-element dtype dispatch out of `AttentionCrossKernel` and measured 11.30x /
> 12.12x on GB10 at exactly those shapes.
>
> So the split the repair would be chosen against no longer describes the tree,
> and the first `## Owed` line of `.agents/specs/ltx25-render-speed-parity.md`
> ("The repair is not here") is still open.
>
> This row re-measures the connector leaf on current `main`, on the machine the
> render was measured on, and repairs what the NEW measurement blames. Byte
> equality is the gate, proven at more than one worker count. A device arm for
> the connector is explicitly out of scope: `Ltx2Attention` interleaves host
> `RmsNormRows` and `Ltx2ApplyRotaryEmb` on raw `float*` between its GEMMs and
> `Ltx2ConnectorForward` reads its weights as host `std::vector<float>`, so that
> is a weight-arm port with its own numerics gate and its own row.
>
> Spec: `.agents/specs/ltx25-connector-repair.md`.
>

## Resolution

-
