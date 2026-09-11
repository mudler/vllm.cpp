ID: ISSUE-GH-2049
Title: **The row spec's settled-decisions list stated the REFUTED QSA mapping**, telling a fresh W4/W5 implementer to build QSA on MiniMax-M3 and calling the DeepSeek-V4 lane "the wrong port" — the exact reverse of the correction recorded in the same file's Port map and Design section, in the matrix row, and in [#1978](https://github.com/mudler/vllm.cpp/issues/1978). Pre-existing on `main`; found while reviewing the W6a merge ([#2019](https://github.com/mudler/vllm.cpp/pull/2019)) and fixed in that same flow per AGENTS.md "Every change starts from an issue". Load-bearing rather than cosmetic: item 2 sits in the section written so an implementer does NOT re-derive it, and a top-down reader hits the stale instruction before the corrected Design section.
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: doc
GitHub: 2049
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:779`

### Frozen archive evidence

> | [#2049](https://github.com/mudler/vllm.cpp/issues/2049) | `MODEL-MM-QWEN4-EXP` | **The row spec's settled-decisions list stated the REFUTED QSA mapping**, telling a fresh W4/W5 implementer to build QSA on MiniMax-M3 and calling the DeepSeek-V4 lane "the wrong port" — the exact reverse of the correction recorded in the same file's Port map and Design section, in the matrix row, and in [#1978](https://github.com/mudler/vllm.cpp/issues/1978). Pre-existing on `main`; found while reviewing the W6a merge ([#2019](https://github.com/mudler/vllm.cpp/pull/2019)) and fixed in that same flow per AGENTS.md "Every change starts from an issue". Load-bearing rather than cosmetic: item 2 sits in the section written so an implementer does NOT re-derive it, and a top-down reader hits the stale instruction before the corrected Design section. | doc |

## Resolution

-
