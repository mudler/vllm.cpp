ID: ISSUE-GH-2367
Title: agent-ready cannot pass anywhere: the checker-discovery loop always skips five arg-gated gates against the zero-skip READY contract
Row: ENG-PREFLIGHT-COMPILES
State: OPEN
Kind: UNKNOWN
GitHub: 2367
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-30
Updated: 2026-08-30
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> Since the preflight checker-discovery loop landed (`d574f514a`), every local `python3 scripts/agent-ready.py` fails on every host, for every branch, including a clean checkout of origin/main. The loop runs every `scripts/check-*.py` WITHOUT arguments; five checkers require them and always exit with argparse usage errors, which the loop reports as SKIP ("needs arguments preflight does not supply"):
>
> - `check-pr-size.py` (`--base`/`--head` required)
> - `check-arm-isa-build.py`, `check-cpu-isa-build.py`, `check-cuda-fat-gencode.py` (`--compile-commands` required)
> - `check-triton-aot-multiarch.py` (`--vendored-root` required)
>
> `agent-ready.py` invokes preflight with `--fail-on-skip` and refuses any skip — its documented #998 contract — so READY FAILED everywhere since the loop landed. Reproduce on origin/main `397a7c027`: five SKIPs, READY FAILED.
>
> ## Root cause
>
> Two intended designs contradict:
>
> 1. The discovery loop's stated rule: "A checker that needs arguments this block cannot supply is a SKIP carrying the reason, never silence" — a skip is the HONEST report for an unrunnable gate.
> 2. agent-ready's #998 contract: the pre-handoff gate reads ANY skip as "not ready", because a skip once masked a trailer check that never ran.
>
> Each is right alone; together they make READY unsatisfiable for every branch since the five checkers fell under the loop.
>
> ## Candidate repairs (the fixing row decides with the developer)
>
> - (a) Preflight constructs what it can and names what it cannot: run `check-pr-size.py --base "$BASE_SHA" --head HEAD` locally (always constructible); give the build-artifact gates explicit blocks that run when their input exists (`compile_commands.json` under `build/`, the vendored triton root) and skip naming the absent artifact when it does not. agent-ready's zero-skip contract stays intact; the skip population becomes genuinely host-conditional instead of unconditional.
> - (b) agent-ready learns a stated tolerance class for artifact-gated skips. Weakens #998; needs the developer.
> - (c) A combination.
>
> ## Blocking
>
> Every pre-push READY on every branch and host since `d574f514a`. Found pushing the `tt_clock_state` W2 row, which classified its gate run against this issue as the named blocker.
>
> Owning row (candidate): `CHECKER-READY-ARG-GATES`.

## Resolution

-
