ID: ISSUE-GH-3102
Title: fix(MODEL-TEXT-qwen3-qwen3-for-causal-lm): diagnose the CPU paged anchor drift
Row: MODEL-TEXT-qwen3-qwen3-for-causal-lm
State: OPEN
Kind: UNKNOWN
GitHub: 3102
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-TEXT-qwen3-qwen3-for-causal-lm`
>
> The CPU `test_qwen3_paged_engine` gate fails on unchanged source at `760c8dd7e27781430b909c3679790c52fe332182`. Both the ordinary Qwen3-0.6B case and its KV-boundary case fail the exact local anchor check at `tests/parity/test_qwen3_paged_engine.cpp:365`.
>
> Prompt 0 is `The capital of France is`. Greedy generation requests 16 tokens. The first five generated IDs are `[12095, 13, 576, 6722, 315]`. At zero-based generated position 5, the committed `our_ids.npy` expects `9625` and the CPU emits `15344`. The boundary case fails on prompt 0 before it reaches its separate boundary exercise.
>
> A fresh baseline build and an independent operator rerun reproduced the same two failures. Both report 28 passing assertions and two failures. The baseline worktree is clean, and the tested executable has SHA256 `46c8a80062219b0135b25ef0668de9b79232fd8f1cf6c173fe4194d8fa6aaf9d`. The F16 implementation's full CTest run encounters the same failure. This establishes that the failure predates that implementation. It does not establish whether the product, artifact selection, or anchor is wrong.
>
> The owning spec is `.agents/specs/first-additive-model-qwen3-dense.md`. The model row records earlier France/Italy differences for Metal and a portable CUDA build. Those observations do not adjudicate this CPU execution. Issue #2671 concerns dump-mode verdicts and does not resolve this ordinary non-dump run. Issue #269 concerns a different gfx1200 ROCm failure.
>
> Evidence in the supplied local workspace is under `/home/vikash/vllm.cpp-rdna3-f16-impl/build-rocm-f16-evidence/`: `cpu-paged-baseline-v1-command.json`, `cpu-paged-baseline-v1-operator.log`, and `cpu-paged-baseline-v1-operator-receipt.json`. The command and receipt record the executable, source revision, environment, and exit status.
>
> The owning model row owes an active-pin, artifact-matched diagnosis and a reviewed correction. Preserve the current fixture, anchor assertion, and tolerance while diagnosing it. Do not refresh the golden solely to make this run green. Keep this full-suite result failing until the cause and correction pass the declared gate.
>

## Resolution

-
