ID: ISSUE-GH-552
Title: DSA top-k review findings: the `w < topk` guard comment overclaims what it defends, the window clamps and non-positive `topk` are ungated, and `DsaTopkLaunch` swallows its launch error (spec `specs/dsa-topk-bounds.md` §7)
Row: MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 552
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:178`

### Frozen archive evidence

> | [#552](https://github.com/mudler/vllm.cpp/issues/552) | `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` | DSA top-k review findings: the `w < topk` guard comment overclaims what it defends, the window clamps and non-positive `topk` are ungated, and `DsaTopkLaunch` swallows its launch error (spec `specs/dsa-topk-bounds.md` §7) | bug |

## Resolution

-
