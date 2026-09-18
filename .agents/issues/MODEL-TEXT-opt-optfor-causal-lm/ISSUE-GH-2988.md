ID: ISSUE-GH-2988
Title: test_opt_paged_engine: removing greedy_ids.npy deletes the whole SACRED gate into a green at 0 assertions
Row: MODEL-TEXT-opt-optfor-causal-lm
State: OPEN
Kind: bug
GitHub: 2988
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-05
Updated: 2026-09-18
Closed: -

## Problem

Re-verified against `7fa861392` on 2026-09-18, before this record was
imported. Both parts still hold, and one line number has drifted.
`tests/vllm/models/test_opt_paged_engine.cpp:154` still returns from the case
when `greedy_ids.npy` is absent, and `:304`'s
`if (!want_prompt.empty()) CHECK(out.prompt_token_ids == want_prompt)` is
unchanged. The ctest entry has moved from `tests/CMakeLists.txt:3663-3667` to
**`:3871-3875`** and still carries no `FAIL_REGULAR_EXPRESSION`; the seven
`FAIL_REGULAR_EXPRESSION` uses in that file are all on other entries.
`vllm_cpp_add_test` sets `SKIP_RETURN_CODE 77`, but the case returns rather
than exiting 77, so CTest still reads the empty run as a pass. Row
`MODEL-TEXT-opt-optfor-causal-lm` is canonical, and the imported body names it,
so this record is row-owned rather than `_owed`.

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-TEXT-opt-optfor-causal-lm`
>
> #2805 / #2981 closed one mute switch in
> `tests/vllm/models/test_opt_paged_engine.cpp`. Two of the same class survive in
> the same file, and one of them is worse than the one that was fixed.
>
> ## 1. Removing `greedy_ids.npy` deletes the whole SACRED gate into a green
>
> Measured by a fresh reviewer of #2981, on **both** the pre-fix and post-fix
> binaries:
>
> ```
> assertions: 0 | 0 passed | 0 failed | Status: SUCCESS!    rc 0
> ```
>
> The bar itself is behind an existence guard, so removing one committed file
> makes the entire token-exact gate assert nothing and report success. The ctest
> entry at `tests/CMakeLists.txt:3663-3667` carries no
> `FAIL_REGULAR_EXPRESSION`, and the case never returns 77, so nothing outside
> the binary notices either.
>
> `.agents/specs/sweep-opt-125m.md` currently calls this sibling guard "already
> correct". That is too strong: it asserts *nothing* rather than asserting
> something *unlicensed*, which is less bad, not correct. `assertions: 0` is a
> skip wearing a pass.
>
> ## 2. `p3_prompt.i32` absence silently drops a cross-check
>
> Removing it takes the post-fix binary from **43 to 42 assertions**, still
> `SUCCESS`, rc 0 -- `if (!want_prompt.empty()) CHECK(out.prompt_token_ids ==
> want_prompt)` at `:304`. Same absence-means-skip shape. Pre-existing, inside
> the region #2981 left byte-unchanged, so neither introduced nor claimed fixed
> there.
>
> ## What closes this
>
> The golden set is a precondition of the gate, not an option: a missing
> `greedy_ids.npy` or prompt file reds rather than reducing the assertion count.
> Red-before for each: remove the file, confirm the gate reds where it currently
> passes. A `FAIL_REGULAR_EXPRESSION` or an assertion-count floor on the ctest
> entry would also catch the `assertions: 0` shape from outside.
>
> Related: #2987 (the same licence defect in four sibling gates).
>

## Resolution

-
