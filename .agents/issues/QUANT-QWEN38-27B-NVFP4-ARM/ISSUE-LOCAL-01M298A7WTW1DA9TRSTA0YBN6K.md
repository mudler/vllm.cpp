ID: ISSUE-LOCAL-01M298A7WTW1DA9TRSTA0YBN6K
Title: Ask 2: run the A arms and decide whether VT_MODELOPT_W4A4=1 is correct; the B arms cannot run because their checkpoint is not staged
Row: QUANT-QWEN38-27B-NVFP4-ARM
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: -

## Problem

`.agents/specs/qwen38-27b-quant-arms.md` designs ask 2 as five arms -- A0, A1, B0, B1, A0' -- one binary, one boot each, interleaved, with A0' as the drift control and a decision table written before the run. This issue exists because that experiment now BLOCKS other work and because two of its five arms are not runnable as written.

WHY IT BLOCKS. The NVFP4 tactic draw survey (#2750, #2751, #2752) can only reach the CUTLASS fp4-activation GEMM, where the tactic cache lives, when `VT_MODELOPT_W4A4=1`. At `0` a ModelOpt NVFP4 checkpoint routes to the W4A16 Marlin arm, and `dense_nvfp4_gemm.h` states its own scope: "W4A16 ONLY. This header deliberately does NOT carry the true-W4A4 (fp4-ACTIVATION) path". So the survey's configuration IS arm A1. Draws taken on an activation path that has not been shown token-exact would measure a defect rather than a tactic, which is why ask 2 is ordered first.

WHAT IS RUNNABLE TODAY. A0, A1 and A0' all use `RadixArk/Qwen3.8-27B-NVFP4`, which is staged at `/workspace/ckpt/qwen38-27b-nvfp4-radixark` and is proven on the device: a separate job decoded 20992 tokens at 12.22 tok/s per stream, with the repeat agreeing at 12.21. Read from its tensor index rather than from prose, it ships 401 `input_scale`, 401 `weight_scale` and 193 `weight_scale_2` across 2194 tensors, and `layers.0.mlp.gate_proj` carries both `weight_scale_2` and `input_scale` -- so the opt-in genuinely flips `IsTrueW4A4()`, which is what `uses_nvfp4_w4a4()` reads.

WHAT IS NOT RUNNABLE. B0 and B1 need `nvidia/Qwen3.6-27B-NVFP4` @ `0893e160`. It is NOT staged. The only 3.6-27B artifacts on the share are `qwen3.6-27b-hf` and `qwen36-27b-q4km`, neither of which is that NVFP4 gate model, and this spec separately records that #821's NVFP4 revision is gone from the hub. Fetching it is a large-asset download and needs the developer's authority; it may not be fetchable at all.

THE CONSEQUENCE FOR THE DECISION TABLE, STATED PLAINLY. B1 is what separates "our fp4-activation GEMM is wrong on a shipped format" from "the failure is artifact-specific". Without it, two rows of the four-row table cannot be told apart. But the A half alone answers the question the survey needs: if A1 is TOKEN-EXACT against the pinned oracle, the arm is trustworthy for tactic draws whatever B would have said. If A1 is divergent, the survey must not run on it, and the B arms become necessary to attribute the cause.

THE PRECONDITION THE SPEC ALREADY NAMES, AND IT IS THE ONE THAT MATTERS MOST. "Confirm the fp4 GEMM is present; an arm that silently declined would read as 'W4A4 is fine' and it would be measuring W4A16 twice." A1 must assert it actually took the fp4-activation path before any token comparison is believed.

PROTOCOL. The oracle is pinned vLLM on RadixArk in its production configuration, never `--enforce-eager`; upstream's own arm for that checkpoint is `ModelOptNvFp4LinearMethod`, so the oracle is A1's denominator by construction and A0 is the divergent arm. Token gate: six raw completion prompts, no chat template, greedy both sides, 48 tokens, concurrency 1, `ignore_eos`, compare `vllm-bench --output-token-ids` against the oracle and record the first differing index per prompt, with the tokenizer half asserted separately. No throughput axis is expected to move; if one does, that is a finding.

RELATED. `#2760` owns the wider question and its ask 3 (declaration routing) is deliberately ordered after this. The survey-side capability -- setting the arm and stamping it into every draw record so it can never be inherited from an ambient shell -- is owned separately on row KERNEL-GEMM-BF16.

## Resolution

-
