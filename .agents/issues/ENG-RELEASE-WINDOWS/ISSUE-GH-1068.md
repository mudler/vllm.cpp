ID: ISSUE-GH-1068
Title: `main` stopped compiling under MSVC at `e34d71379` (#1054), which dropped the `[&kRequired]` capture from the `refuse` lambda in `qwen3_5_weights.cpp` as "the redundant namespace-scope capture". `kMoeExpertLayoutHelp` (`:894`) is namespace-scope and needs no capture; `kRequired` (`:929`) is a function-local `const std::string&` bound to it and IS odr-used in the lambda body, so MSVC rejects it (`error C3493`). Fixed by naming the namespace-scope constant inside the lambda, which satisfies MSVC and keeps the AppleClang diagnostic #1054 removed. It landed green because the guarding gate is a source-TEXT assertion ("rejects `[&kRequired]`, finds `[]`") that passes whether or not the TU compiles, and because `windows-msvc-*` are skipped on `main` (#503) so no baseline existed to regress. An instance of [#503](https://github.com/mudler/vllm.cpp/issues/503)
Row: ENG-RELEASE-WINDOWS
State: UNKNOWN
Kind: bug
GitHub: 1068
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:328`

### Frozen archive evidence

> | [#1068](https://github.com/mudler/vllm.cpp/issues/1068) | `ENG-RELEASE-WINDOWS` | `main` stopped compiling under MSVC at `e34d71379` (#1054), which dropped the `[&kRequired]` capture from the `refuse` lambda in `qwen3_5_weights.cpp` as "the redundant namespace-scope capture". `kMoeExpertLayoutHelp` (`:894`) is namespace-scope and needs no capture; `kRequired` (`:929`) is a function-local `const std::string&` bound to it and IS odr-used in the lambda body, so MSVC rejects it (`error C3493`). Fixed by naming the namespace-scope constant inside the lambda, which satisfies MSVC and keeps the AppleClang diagnostic #1054 removed. It landed green because the guarding gate is a source-TEXT assertion ("rejects `[&kRequired]`, finds `[]`") that passes whether or not the TU compiles, and because `windows-msvc-*` are skipped on `main` (#503) so no baseline existed to regress. An instance of [#503](https://github.com/mudler/vllm.cpp/issues/503) | bug |

## Resolution

-
