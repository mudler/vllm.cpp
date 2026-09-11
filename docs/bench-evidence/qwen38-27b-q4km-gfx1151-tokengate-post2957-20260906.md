# The gfx1151 token gate is REPRODUCIBLE, and #2957 is ruled out as its cause

Second independent run of the Qwen3.8-27B Q4_K_M ROCm token gate, four days after
the first, on a different build. Issue
[#2957](https://github.com/mudler/vllm.cpp/pull/2957), row `BACKEND-ROCM`.
Predecessor:
[`qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`](qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md).

**No speed, latency or memory figure appears here.** `AGENTS.md` §Gates admits a
performance result only after the arm's declared token gate passes. It does not
pass. [#2497](https://github.com/mudler/vllm.cpp/issues/2497) has already had one
measurement retracted for exactly that.

## Disposition

**`TOKEN_GATE=FAIL`, tokenizer 0 of 6 divergent, generation 3 of 6 divergent —
the SAME three prompts as the predecessor, at the same first-differing index.**

| | predecessor (2026-09-02) | this run (2026-09-06) |
|---|---|---|
| tree | pre-#2957 | `ac77de7b4`, carries #2957 |
| tokenizer | 6/6 exact | 6/6 exact |
| divergent prompts | **1, 3, 5** | **1, 3, 5** |
| prompt 1 first-diff index | 45 | **45** |
| verdict | FAIL | FAIL |

## 1. #2957 is RULED OUT, by a rule fixed before the run

The interpretation rule was committed in the task briefing before the job ran:
a divergence set that **moves** implicates the term; a set that is **unchanged**
rules it out. The set did not move — not narrowed, not shifted, not even by one
index.

This was a hypothesis worth testing. #2957 narrows `silu(gate)` to the gate dtype
in `SiluMulK`/`MoeSiluMulK` on the MoE path, which is exactly the class of
ROCm-only rounding difference that could move a rank-1/rank-2 boundary, and three
flips in 288 decode steps is the size of effect a missing narrowing produces. It
is simply not the cause here.

**#2957 remains a correct fix and this does not argue against it.** The same job
measured it on the board: 3 test cases, **7061 of 7061 assertions**, 0 skipped,
matching the count derived from the test source BEFORE the run. Its
discrimination arm is what makes it trustworthy — without the fix **6 CHECKs go
red and every one is a `bf16 := true` case**, the 3 that still pass are precisely
the smallest shapes (`{1,4}`, `{1,4}`, `N=8`) with too few elements to witness a
rounding difference, and the ~7052 f32 element-checks are untouched, exactly as
`NarrowTo<float>` being the identity requires. The test detects the defect rather
than agreeing with it.

## 2. The failure is REPRODUCIBLE, which is new and useful

The predecessor established the divergence set once. This run reproduces it
exactly, on a different build, four days later. So the 3-of-6 is a **stable
property of the ROCm arm and not run-to-run noise**.

That matters for method: a stable target can be bisected over candidate terms
with clean signal, one lease per candidate, each returning implicated-or-ruled-out
under the same pre-committed rule. An unstable one could not be.

## 3. What is still unscoped

The predecessor's finding stands untouched: the ROCm tier loses prompts 1/3/5
while the post-#2534 CPU tier loses 1/2/4, disjoint, `TIER_DIVERGENCE_SAME_INDEX=0`.
**Our two own tiers emit different tokens on the same prefix**, so the term is
ROCm-specific and internal, not oracle non-determinism.

[#2966](https://github.com/mudler/vllm.cpp/issues/2966) is the next candidate —
the same alleged missing narrowing in the gelu-mul kernels — and the same method
applies to it.

## Why this gates more than correctness

While this reads FAIL, **no throughput number on `gfx1151` is admissible**, against
llama.cpp or against vLLM — which [#2740](https://github.com/mudler/vllm.cpp/issues/2740)
established does run on this board. The whole Strix speed axis sits behind these
three prompts.

## What ran

rc job `74a5a81c-ecef-4b1c-b93f-eecfa93fe733` on `strix:gpu0`, worker
`rc-worker-lcjhd`, 2026-09-06. Nothing reached the box by `ssh`. Six prompts,
48 tokens, greedy, batch 1, against stock llama.cpp `b10451` with
`ORACLE_TOKENS_OK=6`, `ORACLE_N_VOCAB=248320`, `ORACLE_SYSTEM_INFO=ROCm`. The
verbatim comparison is committed beside this file as
[`qwen38-27b-q4km-gfx1151-tokengate-post2957-20260906/token_gate.txt`](qwen38-27b-q4km-gfx1151-tokengate-post2957-20260906/token_gate.txt).

**One board fault was observed** across the job's legs (`LEGS_OK_fix=2 of 3`, and
a `Memory access fault by GPU node-1 ... Page not present` on the probe arm). The
predecessor recorded 6 of 6 legs clean with zero faults, so the fault rate on this
board is not as settled as that reading suggested. It is recorded here rather than
averaged away; it bears on how many repetitions a future gfx1151 measurement needs.
