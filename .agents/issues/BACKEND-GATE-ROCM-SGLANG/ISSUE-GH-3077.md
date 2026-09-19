ID: ISSUE-GH-3077
Title: test(BACKEND-GATE-ROCM-SGLANG): calibrate a Qwen3-4B correctness proposal
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: bug
GitHub: 3077
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-08
Updated: 2026-09-19
Closed: -

## Problem

Production Qwen3-4B greedy results differ across repeated oracle runs.
The strict token gate remains failing. No numerical tolerance is ratified.
The developer approved capturing original scores, including extraction after replay replaces token IDs without changing logits.
Use `.agents/specs/strix-qwen3-4b-distributional-calibration.md` as the current diagnostic design.
Custom vLLM logits processors are unsuitable for the observed V2 runner.
Use the pinned public raw-logits and trace-replay interfaces, subject to measured production-path equivalence.
Issue 3076 owns the reusable worker-start profiling mechanism and exact ordinary-production baseline evidence.
Issue 3077 owns the mode-aware integration and trace evidence for capture-only and replay modes.
This issue remains open until its diagnostic evidence and ratification proposal land.
The historical text below predates that source investigation and approval.

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Owner: current Strix campaign operator. Parent: #3053. Spec: .agents/specs/strix-qwen3-4b-distributional-calibration.md (commit before implementation).
>
> Pinned production vLLMe126687a9a828d513c01a07cd69f025f27d63280 showed repeated greedy token differences on Qwen3-4B BF16 at c1 and c4. Qualification802b32c9, result SHA1d518a475a79947d9921be92e9893ed74a4831359063284524b3c32a0db4f400, remains NOT_ACCEPTED. User approved scoping a distributional correctness proposal, not numerical tolerances or replacement of the strict gate.
>
> Implement reviewed diagnostic score capture through public incremental generation, not whole-prefix prefill scoring labeled as decode. Ours6e3cbfb exposes a public logits callback; pinned vLLM exposes named importable LogitsProcessor classes, JSON-compatible extra_args and full-vocabulary sample logprobs. Verify these routes execute with the campaign production runner/graphs before calibration. No serialized callables or eager correctness/performance substitution. Compare original prompts plus identical consumed continuation prefixes; capture scores before any forcing. Record actual scheduler shapes and instrumentation equivalence, not requested concurrency alone.
>
> Predeclare oracle-only calibration and held-out validation before viewing candidate scores. Separate c1/c4 and first-prefill/incremental phases. Proposed thresholds, sample counts and statistical rule require subsequent explicit ratification. Existing CUDA/27B/quantized tolerances are not transferable. Stable oracle behavior remains token-exact. Enforce hard failures for prefix/tokenizer/dtype/model changes, nonfinite/missing scores, wrong execution phase, swapped rows, post-forcing capture and skipped callbacks; include corruption controls. A forced output match is not a correctness pass.
>
> Acceptance: source-bound diagnostic design, test-first implementation, independent mutation review, operator leased equivalence and oracle-only evidence, and a written proposed gate with held-out results. This issue does not itself authorize adoption of a tolerance or acceptance of speed. Stop if instrumentation changes the relevant numerical path or production capture cannot be established; name the exact unresolved interface.
>

## Resolution

-
