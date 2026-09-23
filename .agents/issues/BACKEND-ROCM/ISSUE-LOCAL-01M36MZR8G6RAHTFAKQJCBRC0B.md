ID: ISSUE-LOCAL-01M36MZR8G6RAHTFAKQJCBRC0B
Title: rocm_paged_attn.hip: unused variable g fails -Werror on clang-17
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

PagedAttnDecodeSplitKvReduce at rocm_paged_attn.hip:1313 declares const int64_t g = h / (hq / num_kv_heads) but never uses g. clang-17 with -Werror treats this as a hard error, blocking the HIP build on Strix (gfx1151, ROCm 5.7). PR #3267 removed the adjacent unused m_split variable but missed this one.

## Resolution

Fixed in PR #3278 (merged as ffd7fbdce). Removed unused variable g at rocm_paged_attn.hip:1313.
