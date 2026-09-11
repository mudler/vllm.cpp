ID: ISSUE-GH-470
Title: `PackedGdnDecodeEnvSelected` mirrors only the ENV; the real predicate also requires `in_proj_qkv_fp8.Empty()`, so the 27B gate throws for the wrong reason on any fp8 tower
Row: GATE-27B-FP8-TOWER-GOLDEN
State: UNKNOWN
Kind: bug
GitHub: 470
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:80`

### Frozen archive evidence

> | [#470](https://github.com/mudler/vllm.cpp/issues/470) | `GATE-27B-FP8-TOWER-GOLDEN` | `PackedGdnDecodeEnvSelected` mirrors only the ENV; the real predicate also requires `in_proj_qkv_fp8.Empty()`, so the 27B gate throws for the wrong reason on any fp8 tower | bug |

## Resolution

-
