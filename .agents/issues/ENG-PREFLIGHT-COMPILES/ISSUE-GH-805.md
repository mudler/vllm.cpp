ID: ISSUE-GH-805
Title: main-red (aa3643b6): C++ helper inside extern "C" breaks every clang build (MUSIC3 W6)
Row: ENG-PREFLIGHT-COMPILES
State: CLOSED
Kind: UNKNOWN
GitHub: 805
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-14
Updated: 2026-08-15
Closed: 2026-08-15

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Commit `aa3643b6` (feat(MODEL-MUSIC-MUSIC3): W6) placed the anonymous-namespace helper
>
> ```cpp
> vllm::multimodal::SpeechRegistry& SpeechRegistry() { ... }
> ```
>
> **inside** the `extern "C" {` block of `src/capi/vllm_c.cpp` (lines ~488–1955). Clang rejects a C-linkage function returning a C++ reference:
>
> ```
> src/capi/vllm_c.cpp:1672:35: error: 'SpeechRegistry' has C-linkage specified, but returns
> user-defined type 'vllm::multimodal::SpeechRegistry &' which is incompatible with C
> [-Werror,-Wreturn-type-c-linkage]
> ```
>
> The Windows/MSVC jobs compiled (MSVC is lax about this), which is why CI stayed green on that lane — but every **clang** build is broken on that commit, including the Tenstorrent build (clang-20, -Werror).
>
> **Minimal fix** (applied locally on `row/BACKEND-TENSTORRENT-HOST-FREE-R1` commit `4f0a6f94` to unblock — happy to send as a standalone PR instead): hoist the helper (with its anonymous namespace) to just before the `extern "C" {` block opens. No behavior change; the function is and always was C++, the linkage was accidental.
>
> FYI @ whoever owns the MUSIC3 lane.

## Resolution

Pull request #814 merged on 2026-08-15 as commit `b3d0f3ed5dc8d0918a85b960bca630b311402762` and hoists the C++ helper out of `extern "C"`. GitHub closed issue #805 on the same date.
