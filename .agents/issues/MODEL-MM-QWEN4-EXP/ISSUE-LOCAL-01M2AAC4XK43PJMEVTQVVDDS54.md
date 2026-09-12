ID: ISSUE-LOCAL-01M2AAC4XK43PJMEVTQVVDDS54
Title: The landed sojufx section states a kernel-launch count nobody measured and cites #2336 for a defect #2336 is not about
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

`.agents/specs/qwen4-exp-flash-next.md` §"The sojufx sparkDash measurement" landed on `main` in `80b741290` with three "known bottlenecks explaining the gap". Two of them are not measurements.

**"Each decode step launches ~1,400 kernels, versus 1 for Qwen3.5ForCausalLM models that have a fused decode graph" (spec :266-268).** Nobody counted this tree's launches. The number is a misreading of a comment in `src/vllm/model_executor/models/qwen3_5.cpp:1096`, which says "`ResidentWeight` re-enters the alias branch about 1,361 times per decode step, roughly 70 GiB of counted bytes". Those are weight-residency CALLS, and the comment is about qwen3_5's host-alias instrument on a different checkpoint, not about qwen4_exp and not about kernel launches. The companion "versus 1" is a graph count standing in for a launch count. A launch count needs a profiler, and no trace was taken.

**"Per-step MoE adapter rebuild ([#2336](https://github.com/mudler/vllm.cpp/issues/2336))" (spec :271-274).** The rebuild is real — `qwen4_exp_forward.cpp:673` composes the adapter inside the layer loop — but #2336 is the PLE block, its gate op and the layer loop, and its one `ResidentWeight::d_dev` remark is about the GDN adapter. The spec inherited the citation from the code comment at `qwen4_exp_forward.cpp:656`, which cites "#2336 §3" for the same claim. No issue owned the MoE rebuild until `.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS.md`.

The third, weight aliasing over UMA, is plausible and equally unmeasured; its "~136 GB working set" is a derivation, not a reading.

The result those bottlenecks explain is itself weak. The recorded vllm.cpp arm is 99 decode tokens against a requested 400 — a truncated stream, not a completed benchmark — and the spec's "what each side ran" table credits the vllm.cpp side with "256 blocks x 32 tokens, prefix caching disabled, async scheduling enabled", none of which appears in either launcher the session used: both passed `--max-num-seqs 1 --device cuda` and no other engine flag. The build was a bare `cmake -DVLLM_CPP_CUDA_ARCHITECTURES=121a` with no record of its compiled feature set.

This issue owns the correction: state the three bottlenecks as hypotheses with their evidence status, remove the launch count until a trace supports one, repoint the adapter citation, and mark the run's configuration table to what the launchers actually passed.

## Resolution

-
