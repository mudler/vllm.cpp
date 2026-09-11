ID: ISSUE-GH-476
Title: The 27n fp8-tower arm cannot see a dequant fallback on `out_proj_fp8` or the four `self_attn.*_proj_fp8`: `GdnFp8InProjDebugStats` counts GDN `in_proj` only, and `6603356a` proves that defect is token-invisible
Row: GATE-27B-FP8-TOWER-GOLDEN
State: UNKNOWN
Kind: bug
GitHub: 476
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:81`

### Frozen archive evidence

> | [#476](https://github.com/mudler/vllm.cpp/issues/476) | `GATE-27B-FP8-TOWER-GOLDEN` | The 27n fp8-tower arm cannot see a dequant fallback on `out_proj_fp8` or the four `self_attn.*_proj_fp8`: `GdnFp8InProjDebugStats` counts GDN `in_proj` only, and `6603356a` proves that defect is token-invisible | bug |

## Resolution

-
