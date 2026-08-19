# CLAIM-ENG-CUDAGRAPH-BREAK-W6

**Row:** `ENG-CUDAGRAPH-BREAK` (engine-matrix, engine core and scheduling).
**Stage:** W6, the eligibility predicate — the last stage of the row.
**Issue:** [#1374](https://github.com/mudler/vllm.cpp/issues/1374).
**Parent:** [#1163](https://github.com/mudler/vllm.cpp/issues/1163).
**Also closes:** [#1020](https://github.com/mudler/vllm.cpp/issues/1020).
**Filed by this stage:** [#1380](https://github.com/mudler/vllm.cpp/issues/1380).
**Spec:** [`.agents/specs/eng-cudagraph-break.md`](../specs/eng-cudagraph-break.md).
**Role:** fresh implementer.
**Base:** `601b576c64d6ffc09112b23130d98af276edd113`.
**Branch:** `row/ENG-CUDAGRAPH-BREAK-W6`.
**Date:** 2026-08-19.

## Scope taken

The graph-eligibility predicate moves out of the two model files that each
re-derived it and into `GPUModelRunner::execute_model`, which names the step's
ACTUAL uniform query length once through `v1::GraphEligibleQueryLen` and ships it
on `ModelForwardInput::uniform_query_len`. Both Qwen3.5 drivers key their slot
ring on `(S, q, spec)`. The widening is bounded by `VT_SPEC_GRAPH_MAX_QLENS`.

## Scope NOT taken, and why

The predicate did NOT move to "eligible except at the break points". Nothing in
this tree captures a prefill or a mixed batch under any predicate, because all
nine drivers are decode drivers. That needs a prefill capture driver nobody has
written, and its benefit is refuted by this spec's own dated D5 measurements on
this hardware. Recorded in the spec's `## Owed` as a row-level item with what
would have to be true first, not as a stage that ran out of window.

## Evidence

- G2 at three levels, five detecting mutations plus an over-fire control, and one
  mutation that was NOT detected and forced a repair. See the spec's `## Gates`
  and the pull-request body.
- G1 re-run on `thor:gpu0` through an `rc` lease: 2066 assertions, 0 failed, 0
  differing on all five migrated drivers. W6 moves no logit.
- The ring key's own device case is blocked by
  [#1380](https://github.com/mudler/vllm.cpp/issues/1380), filed by this stage
  and pre-existing.

## Authority used

One `rc` lease on `thor:gpu0`, released. Push and pull request on
`row/ENG-CUDAGRAPH-BREAK-W6`. No merge, no force-push, no rebase, no `main`.
