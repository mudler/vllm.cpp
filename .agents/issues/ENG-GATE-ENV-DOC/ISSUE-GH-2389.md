ID: ISSUE-GH-2389
Title: check-env-doc is one-directional: a documented knob whose reader was deleted stays green forever
Row: ENG-GATE-ENV-DOC
State: CLOSED
Kind: UNKNOWN
GitHub: 2389
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
> `scripts/check-env-doc.py:68` computes `scanned - documented - allowlisted`: it catches a variable **read but not documented**. The reverse — a variable **documented but read by nothing** — is unguarded.
>
> ## Why it matters
>
> That is the structural reason two user-facing knobs currently document behaviour the tree does not implement:
>
> - `VT_QWEN35_STAGE_MIN_FREE_FRAC` — its reader was deleted when the staging policy was rewritten to a total-memory rule; the doc row, with a default, a formula and tuning advice, survived. Its only occurrence in compiled code is a comment.
> - `VT_GEMMA4_MLP_MOE_PARALLEL` — no occurrence in `src/`, `include/`, `examples/`, `tools/` or `benchmarks/`. The row self-discloses "Not wired in this PR tip", but it still sits in the user-facing table with a stated default and effect.
>
> A gate that only fires in one direction lets a knob outlive its implementation indefinitely, and the failure is invisible: the operator sets the variable, nothing happens, and nothing says why.
>
> ## Proposed check
>
> Add the reverse assertion — every variable in the user-facing table of `docs/ENVIRONMENT.md` must appear in at least one compiled file under `src/` or `include/` **as more than a comment** (the `VT_QWEN35_STAGE_MIN_FREE_FRAC` case is precisely a comment-only hit, so a naive grep would pass it).
>
> Two known complications, so the check needs a declared escape rather than a naive rule:
>
> - Some variables are legitimately read only by shipped binaries outside `src/` — `VT_BENCH_PRETOKENIZE` is read at `examples/bench/bench_core.h:608`, which is the `vllm-bench` binary its row scopes it to. The search path must include those, or the row must declare where it is read.
> - Some are read through helpers rather than a literal `getenv`: `VT_GGUF_KEEP_QUANT` is read via `EnvOnOr("VT_GGUF_KEEP_QUANT", ...)` at `gguf_keep_quant.cpp:248`. A `getenv`-only scan would report a false positive.
>
> This is a checker semantics change, so per AGENTS.md it needs a spec, a red-before test and green-after evidence, and it must not be made to pass by widening scope.
>
> Row: `ENG-GATE-ENV-DOC`

## Resolution

Commit `b56cffeb5093a765aa666b7b898169ce57d37f9b` adds the reverse environment-documentation assertion described by this issue; GitHub closed it as COMPLETED on 2026-08-31.
