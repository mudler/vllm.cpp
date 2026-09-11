ID: ISSUE-GH-2404
Title: windows-msvc is RED on every PR: glm5_next_weights.cpp shadows 'v' and /WX makes it an error
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: UNKNOWN
GitHub: 2404
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> Both Windows jobs fail to COMPILE on every pull request:
>
> ```
> glm5_next_weights.cpp(223,31): warning C4456: declaration of 'v' hides previous local declaration
> glm5_next_weights.cpp(226,31): warning C4456: declaration of 'v' hides previous local declaration
> glm5_next_weights.cpp(223,31): error C2220: the following warning is treated as an error
> ```
>
> `windows-msvc-cpu` and `windows-msvc-vulkan` both die this way; the cause is identical in each.
>
> `KdaHeadCount` declares `const GgufValue* v` in an `if`, then re-declares `v` in each `else if` rung. GCC and Clang do not warn here, so it built green on Linux and landed.
>
> Introduced by `b326ea003` (#2242 / #2292).
>
> ## Why it reached main green, which is the part worth fixing
>
> `windows-msvc` is a PR-only job with no `main` baseline. `main` never runs it, so a Windows-only compile break lands green and is then charged to whichever unrelated pull request runs CI next — every one of them, until someone fixes it.
>
> This is structurally the same failure as [#2389](https://github.com/mudler/vllm.cpp/issues/2389) (`check-env-doc` runs ahead of a push but never on `main`) and as the two env-doc reds before it: a gate that does not run on `main` converts a landed defect into a red on innocent branches. Three instances now, in three different gates.
>
> ## Fix
>
> Rename the shadowed declarations (`v_group`, `v_inner`). Behaviour-preserving, and it restores the Windows build. Fixed in-flow because it blocks an unrelated PR from reaching green, which is exactly the cost the mechanism above imposes.
>
> Row: `ENG-RELEASE-WINDOWS`

## Resolution

The binding comment https://github.com/mudler/vllm.cpp/issues/2404#issuecomment-5480894115 dated 2026-08-31 identifies fix commit `5f230020f` and verifies both Windows jobs compile. It keeps the separate pre-existing API-server crash out of this resolution.
