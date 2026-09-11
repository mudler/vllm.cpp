ID: ISSUE-GH-605
Title: `--reasoning-parser` resolved 10 of upstream's 28 names, so 61 of 76 official-recipe uses aborted at startup — including `qwen3` (18 uses), which the published Qwen3.5/3.6 recipes pass to models we already gate token-exact. W3 runs before W2 on measured demand (W3 covers 43/76, W2 covers 18 and four of its names have zero). **Brick 1 landed:** `qwen3` + `mimo`, coverage 10 -> 12, 20 of the 76 uses
Row: SAMPLE-REASONING
State: UNKNOWN
Kind: feature
GitHub: 605
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:151`

### Frozen archive evidence

> | [#605](https://github.com/mudler/vllm.cpp/issues/605) | `SAMPLE-REASONING` | `--reasoning-parser` resolved 10 of upstream's 28 names, so 61 of 76 official-recipe uses aborted at startup — including `qwen3` (18 uses), which the published Qwen3.5/3.6 recipes pass to models we already gate token-exact. W3 runs before W2 on measured demand (W3 covers 43/76, W2 covers 18 and four of its names have zero). **Brick 1 landed:** `qwen3` + `mimo`, coverage 10 -> 12, 20 of the 76 uses | feature |

## Resolution

-
