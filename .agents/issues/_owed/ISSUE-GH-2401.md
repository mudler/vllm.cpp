ID: ISSUE-GH-2401
Title: No gate compiles anything before a push: main landed twice today in a state that does not build
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2401
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-PREFLIGHT-COMPILES`
>
> `main` was pushed twice on 2026-08-31 in a state that does not compile, and every
> record gate was green both times.
>
> 1. `5263ac31f` — `tools/bench/ltx2_connector_gemm_probe.cpp` carried shell line
>    continuations at the end of `//` comment lines. `-Wcomment` plus this tree's
>    `-Werror` rejects that. The CMake target existed so the probe could not rot,
>    and it had never been built.
> 2. `08fa2f5aa` — `tests/vllm/models/test_glm_moe_dsa_schedule.cpp:304` passed
>    `MlaSharedSelection*` into `ForwardMlaAttentionBlock`'s `vt::Tensor*`
>    parameter, because two rows landing concurrently each added a trailing
>    optional pointer (#2395).
>
> `scripts/agent-preflight.sh` ran and passed on both trees. It runs 30 record
> checkers and 60 Python suites, and not one of them compiles a single translation
> unit. They validate records, prose, anchors and trailers against a tree that
> does not build.
>
> CI does compile — `cuda-fat-build`, `build-test-cpu`, `build-newest-gcc`, the
> sanitizers — and would have caught both. It lands its verdict up to two hours
> after the push, and both of these reached `main` by a direct push, so the
> verdict arrived after the damage.
>
> The second defect is the harder shape: neither contributing commit's own diff
> contains the broken call. `e799f7d2c` added the parameter to the header;
> `ee5c86031` added the test that calls it. Each side compiled alone. Only the
> merged tree is red, so a gate that compiles "the files this commit changed" does
> not see it — the scope has to follow `#include` edges out of the changed
> headers.
>
> What is owed: a mechanism that compiles what a change can break, before the
> push, at a cost that does not fire on the 30 of the last 60 commits that touch
> no C++ at all.
>
> Found while building `main` for an unrelated reason, twice in one pass.
>

## Resolution

GitHub links pull request #2412 as closing issue #2401 on 2026-08-31.
