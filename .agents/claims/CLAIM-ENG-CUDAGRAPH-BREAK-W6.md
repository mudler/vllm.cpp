# CLAIM-ENG-CUDAGRAPH-BREAK-W6

| Claim | Row IDs | Agent | Worktree / remote dir | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-ENG-CUDAGRAPH-BREAK-W6` | `ENG-CUDAGRAPH-BREAK` (`ACTIVE`) | Claude Code (opus-5), fresh implementer | isolated worktree; one `rc` lease on `thor:gpu0` for the CUDA build and G1, released; no registry, no publication | `row/ENG-CUDAGRAPH-BREAK-W6`, issue [#1374](https://github.com/mudler/vllm.cpp/issues/1374), parent [#1163](https://github.com/mudler/vllm.cpp/issues/1163), also closing [#1020](https://github.com/mudler/vllm.cpp/issues/1020), predecessors [#1192](https://github.com/mudler/vllm.cpp/issues/1192) (W1), [#1261](https://github.com/mudler/vllm.cpp/issues/1261) (W2), [#1291](https://github.com/mudler/vllm.cpp/issues/1291) (W3), [#1307](https://github.com/mudler/vllm.cpp/issues/1307) (W4) and [#1335](https://github.com/mudler/vllm.cpp/issues/1335) (W5) | Owns ONLY W6 of [`eng-cudagraph-break.md`](../specs/eng-cudagraph-break.md): the graph-eligibility predicate in `src/vllm/v1/worker/gpu/runner.cpp`, `v1::GraphEligibleQueryLen` / `ActualUniformDecodeQueryLen` / the dispatch counters in `src/vllm/v1/worker/gpu/cudagraph_dispatch.{h,cpp}`, `ModelForwardInput::uniform_query_len`, the two Qwen3.5 registrations' consumption of it (`qwen3_5_moe.cpp`, `qwen3_5_dense.cpp`), the `(S, q, spec)` slot-ring key and the `VT_SPEC_GRAPH_MAX_QLENS` bound in `src/vllm/model_executor/models/qwen3_5.cpp`, and the three W6 gate levels (`tests/vllm/v1/spec_decode/test_mtp_depth.cpp`, `tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp`, `tests/vllm/v1/worker/gpu/test_cudagraph_dispatch.cpp`) plus the W6 device case in `tests/vllm/models/test_decode_graph_seam_g1_cuda.cpp`. EXCLUDES the PIECEWISE arm and the prefill capture driver it would need (row-owned, and its benefit is refuted by the spec's D5 on this hardware), the async device-token decline and [#1305](https://github.com/mudler/vllm.cpp/issues/1305) (row-owned, needs a `dgx` window with checkpoints), [#1380](https://github.com/mudler/vllm.cpp/issues/1380) (filed by this stage, pre-existing, row-owned), and graph-executable dedup ([#1162](https://github.com/mudler/vllm.cpp/issues/1162)) | `ACTIVE` | 2026-08-19 (fresh implementer) — the predicate moved off `pure_decode`: the runner names the step's ACTUAL uniform query length once and every model reads the answer, instead of two model files re-deriving it in twenty duplicated lines each. #1020 closes on that plus the `(S, q, spec)` ring key, which was a LIVE collision at the base commit rather than the enabler #1020 called it. G2 at three levels with five detecting mutations and an over-fire control; a SIXTH mutation was not detected and forced the verify conjunct into `GraphEligibleQueryLen` where one can move it. **G1 re-run on `thor:gpu0`: 2066 assertions, 0 failed, 0 differing on all five migrated drivers — W6 moves no logit.** **The row's headline did NOT close and that is the stage's result**: no driver here serves a prefill or a mixed batch under any predicate, so 'except at the break points' needs a prefill capture driver nobody has written, whose benefit D5 already refutes. Also found and filed: #1380, a `cudaMalloc` inside a capturing stream on the SECOND parity-ring slot of a speculative shape, after which the queue is poisoned |

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
