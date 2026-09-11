ID: ISSUE-GH-505
Title: `DsaTopkKernel` sizes `chosen[512]`/`picked[64]` by literal while `index_topk` is 512 (Flash) / 1024 (Pro); latent behind `dsa_dense` today, silent stack overflow once the real-geometry DSA residual lands (found while assessing #504)
Row: MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 505
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:177`

### Frozen archive evidence

> | [#505](https://github.com/mudler/vllm.cpp/issues/505) | `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` | `DsaTopkKernel` sizes `chosen[512]`/`picked[64]` by literal while `index_topk` is 512 (Flash) / 1024 (Pro); latent behind `dsa_dense` today, silent stack overflow once the real-geometry DSA residual lands (found while assessing #504) | bug |

## Resolution

-
