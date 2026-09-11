ID: ISSUE-GH-2395
Title: main does not build: test_glm_moe_dsa_schedule.cpp passes MlaSharedSelection* where ForwardMlaAttentionBlock takes vt::Tensor*
Row: MODEL-TEXT
State: CLOSED
Kind: UNKNOWN
GitHub: 2395
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-TEXT-GLM-MOE-DSA`
>
> `main` does not compile its test suite.
>
> ```
> tests/vllm/models/test_glm_moe_dsa_schedule.cpp:304:86: error: cannot convert
>   'vllm::mla::MlaSharedSelection*' to 'vt::Tensor*'
> ```
>
> Line 304 is:
>
> ```cpp
> ForwardMlaAttentionBlock(dev, dims, h.weights(), th, tp, kvc, ts, meta, impl, out, shared);
> ```
>
> **Reproduced on clean `origin/main`**, not on any branch. At `f5c707896`, with no
> local change present:
>
> ```sh
> g++ -std=c++20 -fsyntax-only -Iinclude -Isrc -Ithird_party -Itests \
>     -isystem third_party tests/vllm/models/test_glm_moe_dsa_schedule.cpp
> ```
>
> gives the error above. It also fails inside a full `cmake --build`, which is how
> it was found.
>
> **Two concurrently landing rows appear to have collided on one signature.**
> `include/vllm/model_executor/models/mla_attention.h` gained parameters from
> `KV-DSV4-MULTICACHE` (#2323, the `attn_sink`, `6c5336455`) and from this row (the
> shared selection). The call site passes `shared` into a slot that now expects
> `vt::Tensor*`.
>
> **Not fixed here, deliberately.** Which argument the caller means is the owning
> row's knowledge. Reordering them to satisfy the compiler could pass the wrong
> pointer silently, and the type error is the only thing currently preventing that.
> This wants the row that added the parameter, not a drive-by.
>
> Found while building an unrelated row (`VT-CPU-ELEM-DISPATCH`), whose diff
> touches none of `glm`, `mla`, `dsa` or `selection`.
>
> A second, separate build break on `main` was found in the same pass and fixed in
> flow at `5263ac31f`: `tools/bench/ltx2_connector_gemm_probe.cpp` carried shell
> line-continuations inside `//` comments, which `-Wcomment` plus `-Werror`
> rejects. That one was unambiguous; this one is not.
>
> **No checker catches either of them.** The record gates -- trailers, style,
> agent-record, symbol-anchors, env-doc -- all pass on a tree that does not
> compile, because none of them builds anything.
>

## Resolution

GitHub's timeline records closing commit `08fa2f5aa84fa4e97ced1bbc22c6330c38e9d9b5` on 2026-08-31. The commit passes the shared selection to the parameter it names and explicitly closes issue #2395.
