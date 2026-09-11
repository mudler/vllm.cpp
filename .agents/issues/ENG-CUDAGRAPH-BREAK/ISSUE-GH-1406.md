ID: ISSUE-GH-1406
Title: The block-table bound [#1394](https://github.com/mudler/vllm.cpp/issues/1394) adds is CPU-ONLY. Six backends register `OpId::kPagedAttention` (`cpu_paged_attn.cpp:271`, `cuda_paged_attn.cu:2890`, `rocm_ops.hip:148`, `vulkan_ops.cpp:1604`, `metal_ops.mm:1119`, `tenstorrent_ops.cpp:3188`) and the byte-identical `block_table[r * bt_row + (j / block_size) * bt_col]` appears unguarded at `cuda_paged_attn.cu:230,385,521,676,837` and `rocm_paged_attn.hip:226,424,557,765`, with `j` bounded only by `seq_lens[r]`. The obvious repair — one bound at the shared seam `vt::PagedAttention` — is NOT correct as written, and this was measured rather than reasoned: the seam cannot dereference a device `seq_lens`, and its only host-readable quantity `PagedAttentionArgs::max_seq_len` is documented at `include/vt/ops.h:806-812` as a value where **an upper bound is safe**, so a refusal keyed on it rejects a call the contract permits. `row/ENG-CUDAGRAPH-SEAM-SIGSEGV` carried exactly that arm and WITHDREW it for this reason. Three candidate fixes are named in the issue; the smallest that reaches all six backends narrows `max_seq_len` to exact, which the production caller already passes (`v1/attention/backend.cpp:303`, `:57-58`) and which owes a caller audit and a spec
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: gap
GitHub: 1406
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:473`

### Frozen archive evidence

> | [#1406](https://github.com/mudler/vllm.cpp/issues/1406) | `ENG-CUDAGRAPH-BREAK` | The block-table bound [#1394](https://github.com/mudler/vllm.cpp/issues/1394) adds is CPU-ONLY. Six backends register `OpId::kPagedAttention` (`cpu_paged_attn.cpp:271`, `cuda_paged_attn.cu:2890`, `rocm_ops.hip:148`, `vulkan_ops.cpp:1604`, `metal_ops.mm:1119`, `tenstorrent_ops.cpp:3188`) and the byte-identical `block_table[r * bt_row + (j / block_size) * bt_col]` appears unguarded at `cuda_paged_attn.cu:230,385,521,676,837` and `rocm_paged_attn.hip:226,424,557,765`, with `j` bounded only by `seq_lens[r]`. The obvious repair — one bound at the shared seam `vt::PagedAttention` — is NOT correct as written, and this was measured rather than reasoned: the seam cannot dereference a device `seq_lens`, and its only host-readable quantity `PagedAttentionArgs::max_seq_len` is documented at `include/vt/ops.h:806-812` as a value where **an upper bound is safe**, so a refusal keyed on it rejects a call the contract permits. `row/ENG-CUDAGRAPH-SEAM-SIGSEGV` carried exactly that arm and WITHDREW it for this reason. Three candidate fixes are named in the issue; the smallest that reaches all six backends narrows `max_seq_len` to exact, which the production caller already passes (`v1/attention/backend.cpp:303`, `:57-58`) and which owes a caller audit and a spec | gap |

## Resolution

-
