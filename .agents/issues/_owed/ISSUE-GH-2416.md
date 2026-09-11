ID: ISSUE-GH-2416
Title: The other 62 vt CPU kernels still resolve a dtype per element, and nobody has measured which ones it costs
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2416
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `VT-CPU-ELEM-SURVEY`
>
> `VT-CPU-ELEM-DISPATCH` (#2376) hoisted the per-element dtype dispatch out of TWO
> of the 64 CPU kernels that call `vt::cpu::LoadF32`/`StoreF32`, and measured
> 8.75x-11.16x for the pair. Its `## Owed` names the remaining 62 and states the
> sizing method: `perf record -e cpu-clock` over a probe that runs THAT op alone
> at a shape a shipped model actually uses, and read `LoadF32`'s self percentage;
> above ~30% the hoist is worth a row.
>
> Nobody has run that. Without the ranking, the only ways forward are a 62-kernel
> sweep nobody can review or a guess about which kernel is next. This row runs the
> measurement, publishes the ranked table as its product, and hoists only what the
> ranking justifies.
>
> It also owes an answer on the second item in that `## Owed`:
> `__attribute__((always_inline))` on `LoadF32` was measured at a further 1.78x
> and NOT taken, because it forces inlining at 231 sites in one 3900-line
> translation unit and no gate there measured the other 62 kernels'
> instruction-cache behaviour. A per-kernel probe set is exactly that missing
> measurement.
>
> Scope: the survey; the hoists the ranking justifies, each with its own
> byte-equality gate; the `always_inline` verdict with its i-cache evidence.
> NOT in scope: hoisting all 62 (that is not one row), and any change that cannot
> be made bit-exact.
>
> #2376 stays open: it owns the remaining kernels this row does not hoist.
>

## Resolution

GitHub links pull request #2426 as closing issue #2416 on 2026-08-31.
