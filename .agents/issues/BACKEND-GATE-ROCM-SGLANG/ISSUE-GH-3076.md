ID: ISSUE-GH-3076
Title: perf(BACKEND-GATE-ROCM-SGLANG): close the Qwen3-4B Strix c4 gap
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: perf
GitHub: 3076
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-08
Updated: 2026-09-20
Closed: -

## Problem

The approved next step is a matched worker-start profile of vllm.cpp and
production vLLM. One launch owner must configure the pinned rocprofiler SDK
before either GPU worker initializes Torch or ROCm, preserve each engine's
production supervisor and worker lifecycle, and finalize bounded trace
artifacts before normal shutdown. The two arms must use identical profiler
categories and the existing pinned Qwen3-4B workload. The resulting traces are
diagnostic evidence, not an accepted timing denominator.

Late attachment is not an admissible substitute. Retained source and process
evidence shows that Torch/Kineto can own profiler registration before a later
attach request. An attachment route that cannot report already captured HIP
graphs cannot establish graph-replay completeness. The earlier production-vLLM
trace also reached graph capture but failed artifact finalization, so its
partial output remains diagnostic only.

The implementation contract is
[`../../specs/strix-qwen3-4b-worker-start-profiling.md`](../../specs/strix-qwen3-4b-worker-start-profiling.md).
It must land before any profiler implementation. The older campaign spec is
historical at commit `b923ac2c4c24e9c608d4a6e02e868538be7536fa`; it is absent
from current `main` and its attach-first execution sequence is superseded for
this issue.

Retained qualification evidence does not bind an immutable worker image. The
design therefore requires a separate no-GPU discovery phase before runtime
closure preparation. That phase derives one runtime-relevant image digest. It
atomically seals the lease, device, boot, operating system, dpkg database, and
the exact 12 live library records in an unaccepted candidate receipt. A
separate operator promotion records that receipt in the tracked campaign
commitment. Replay and preparation load the commitment from reviewed repository
state. Caller-supplied receipt and hash bytes provide integrity only; they do
not provide campaign provenance.

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Owner: current Strix campaign operator. Parent: #3053. Spec: .agents/specs/strix-qwen3-4b-c4-performance.md (commit before implementation).
>
> User approved measurement-first work to bring vllm.cpp at least level with production vLLM at concurrency four on Strix. Two diagnostic qualification corpora gave medians 61.5947184118 versus 79.5022690204 output tokens/s. Correctness failed, so these are NOT_ACCEPTED diagnostic observations, not a benchmark or accepted ratio. Full record: #3053 issuecomment-5586767582.
>
> Keep Qwen3-4B BF16 revision 1cfa9a7208912126459214e8b04321603b3df60c, six raw prompts, 128 greedy tokens, c1/c4, pinned engine baseline6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb and production vLLMe126687a9a828d513c01a07cd69f025f27d63280. Obtain complete same-tool HIP/API/kernel traces before attribution. Trace06 is incomplete and cannot satisfy parity. Source candidates: rocm_embedding.hip127-142 per-step allocation/synchronization/copy/free; rocm.cpp91-97 and qwen3.cpp1184-1186 reject dense graph replay; paged attention BF16/head128/GQA4 does not use the fused-GQA arm. These are unmeasured hypotheses.
>
> Coordinate existing graph #332/PR#2777 and async mirror PR#2779 before duplicating work. #3043 owns the separate 27B workload. Implement only a measured, individually specified lever, preserving invalid-input errors and production reachability. Require smallest red-before public-entry regression, focused and full gates, independent static and mutation review, operator verification, and a same-binary A/B on an idle leased host. Correctness requires the declared exact or separately ratified distributional gate. No eager denominator, candidate-driven tolerance, pin substitution, or ceiling claim. Final acceptance requires matched warmed repeated throughput at c4 >= vLLM, with c1, latency and memory obligations reported; otherwise keep the gap open and name the next traceable hypothesis.
>

## Resolution

-
