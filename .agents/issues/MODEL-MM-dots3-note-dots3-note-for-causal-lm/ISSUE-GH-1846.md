ID: ISSUE-GH-1846
Title: **The released `dots3-note-prev` shard index declares `indexer_rope_layout: "leading"` and `indexer_rope_converted_from: "tail"` in its `metadata` block, and NOTHING reads either key** — `git grep indexer_rope_layout` over vLLM `origin/main` returns nothing. Measured at W2 while reading the whole index. It is the publisher stating how the DSA indexer's `wq_b`/`wk` are laid out along the 128-wide index head, and it agrees with what upstream's code does anyway: `DeepseekV2Indexer` rotates `[..., :rope_dim]` and leaves `[..., rope_dim:]` (`deepseek_v2.py:805,:814`, `rope_dim` 64 of `index_head_dim` 128), which is a LEADING slice. NOT spec §4 trap 2: that one is about which PAIRS rope rotates (GPT-J vs NeoX), this one about which HALF of the head it rotates, and both are numerically silent on a row spec §6.4 says has no oracle. W2 pins both values in an assertion so a re-published checkpoint cannot flip the layout silently; W2 consumes neither, because W2 writes no maths. W3 owes the slice
Row: MODEL-MM-dots3-note-dots3-note-for-causal-lm
State: UNKNOWN
Kind: feature
GitHub: 1846
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:696`

### Frozen archive evidence

> | [#1846](https://github.com/mudler/vllm.cpp/issues/1846) | `MODEL-MM-dots3-note-dots3-note-for-causal-lm` | **The released `dots3-note-prev` shard index declares `indexer_rope_layout: "leading"` and `indexer_rope_converted_from: "tail"` in its `metadata` block, and NOTHING reads either key** — `git grep indexer_rope_layout` over vLLM `origin/main` returns nothing. Measured at W2 while reading the whole index. It is the publisher stating how the DSA indexer's `wq_b`/`wk` are laid out along the 128-wide index head, and it agrees with what upstream's code does anyway: `DeepseekV2Indexer` rotates `[..., :rope_dim]` and leaves `[..., rope_dim:]` (`deepseek_v2.py:805,:814`, `rope_dim` 64 of `index_head_dim` 128), which is a LEADING slice. NOT spec §4 trap 2: that one is about which PAIRS rope rotates (GPT-J vs NeoX), this one about which HALF of the head it rotates, and both are numerically silent on a row spec §6.4 says has no oracle. W2 pins both values in an assertion so a re-published checkpoint cannot flip the layout silently; W2 consumes neither, because W2 writes no maths. W3 owes the slice | feature |

## Resolution

-
