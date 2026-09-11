ID: ISSUE-GH-1394
Title: The CPU paged attention reads `btab[r * bt_row + (j / block_size) * bt_col]` for every `j < seq_lens[r]` without checking that the block table has that many columns, so a caller with a short table gets an out-of-bounds read, a plausible block index out of it, and attention over the WRONG page — silently. `tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp`'s `SpecAttnMeta` supplies one: hardcoded `block_table_num_cols = 1` against shape C's `seq_lens = 24` at `block_size = 16`. Found while fixing [#1380](https://github.com/mudler/vllm.cpp/issues/1380), whose `DevicePool` change moved the bytes after the table and turned the same read into a SIGSEGV on `thor:gpu0` (`gdb` at `src/vt/cpu/cpu_paged_attn.cpp:224` under `FullAttnBlockPaged`). PRE-EXISTING: the case passes at `origin/main` only because the read landed on bytes that decoded to an in-range index. FIXED IN FLOW in both halves — the kernel refuses a short table with one compare per request outside the token loop, and the helper sizes its table for the sequence length it declares. Owned by row `ENG-CUDAGRAPH-BREAK`
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 1394
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:474`

### Frozen archive evidence

> | [#1394](https://github.com/mudler/vllm.cpp/issues/1394) | `ENG-CUDAGRAPH-BREAK` | The CPU paged attention reads `btab[r * bt_row + (j / block_size) * bt_col]` for every `j < seq_lens[r]` without checking that the block table has that many columns, so a caller with a short table gets an out-of-bounds read, a plausible block index out of it, and attention over the WRONG page — silently. `tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp`'s `SpecAttnMeta` supplies one: hardcoded `block_table_num_cols = 1` against shape C's `seq_lens = 24` at `block_size = 16`. Found while fixing [#1380](https://github.com/mudler/vllm.cpp/issues/1380), whose `DevicePool` change moved the bytes after the table and turned the same read into a SIGSEGV on `thor:gpu0` (`gdb` at `src/vt/cpu/cpu_paged_attn.cpp:224` under `FullAttnBlockPaged`). PRE-EXISTING: the case passes at `origin/main` only because the read landed on bytes that decoded to an in-range index. FIXED IN FLOW in both halves — the kernel refuses a short table with one compare per request outside the token loop, and the helper sizes its table for the sequence length it declares. Owned by row `ENG-CUDAGRAPH-BREAK` | bug |

## Resolution

-
