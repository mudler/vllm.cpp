ID: ISSUE-GH-2987
Title: Four SACRED gates carry #2805's mute switch: the self-determinism licence is inside an existence guard
Row: -
State: OPEN
Kind: bug
GitHub: 2987
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-05
Updated: 2026-09-18
Closed: -

## Problem

Re-verified against `7fa861392` on 2026-09-18, before this record was
imported. All four sites still carry the defect at the line numbers the
imported body names: `tests/vllm/models/test_deepseek_v2_paged_engine.cpp:229`,
`test_glm4_moe_lite_paged_engine.cpp:141` and
`test_qwen3_32b_nvfp4a16_paged_engine.cpp:251` each still read
`int64_t multi_cells = -1;` immediately above `if (fs::exists(gdir /
"greedy_dist.npy"))`, and `test_qwen3coder_paged_engine.cpp:158` still carries
the bare existence guard. `kMinDeterminismRuns` appears only in
`test_opt_paged_engine.cpp`, so none of the four has the `DK` floor. Owned by
`.agents/specs/sweep-opt-125m.md` § 8e and `## Owed`: the issue spans four
model rows and a `Row:` field holds one value, so `_owed` plus one spec anchor
is the only shape AGENTS.md admits for it.

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `-`
>
> Four SACRED token-exact gates carry the mute switch #2805 names, in the form
> #2805 describes, and #2805 covers only the OPT one.
>
> The shape is a self-determinism licence read from `greedy_dist.npy` inside an
> existence guard, with a `-1` sentinel that reaches the report:
>
> ```cpp
> int64_t multi_cells = -1;
> if (fs::exists(gdir / "greedy_dist.npy")) {
>   ...
>   CHECK(multi_cells == 0);
> }
> ```
>
> Deleting the licence file removes the licence instead of failing the gate, and
> the STRICT bar keeps running unlicensed.
>
> Sites, found by a fresh reviewer of #2981 while verifying the OPT repair:
>
> - `tests/vllm/models/test_deepseek_v2_paged_engine.cpp:229-252`
> - `tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp:141-162`
> - `tests/vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp:251-272`
> - `tests/vllm/models/test_qwen3coder_paged_engine.cpp:158` (guard without the
>   sentinel -- same consequence, one symptom fewer)
>
> None carries a `DK` floor either, so the degenerate `K=1` capture that #2981
> showed is the *reachable* form of this defect passes on all four.
>
> **Why this is filed rather than fixed in #2981.** Each of these is another
> row's SACRED gate. Scoping the OPT fix to the OPT row was correct; widening it
> would have put four rows' correctness bars in one pull request.
>
> ## What closes this
>
> Each of the four gets the #2981 treatment, on its owning row: the licence
> fails closed when its file is absent or unreadable, a `DK` floor, and the
> section placed so it executes where re-captures actually arrive (#2981 moved
> it above the device-only checkpoint guard, because behind that guard the
> licence is unexecuted on every host that skips, including CI).
>
> Each needs its own red-before: remove the licence file and confirm the gate
> reds rather than passing at a reduced assertion count.
>
> ## Evidence
>
> #2981 measured the OPT case end to end: with `greedy_dist.npy` removed the
> pre-fix binary printed
> `STRICT correctness gate: 6/6 prompts token-exact (96/96 tokens) ... (vLLM
> self-determinism: -1 multi-valued cells)` and exited **0** at 34 assertions.
>

## Resolution

-
