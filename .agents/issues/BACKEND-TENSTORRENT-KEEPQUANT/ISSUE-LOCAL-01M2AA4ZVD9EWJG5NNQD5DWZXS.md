ID: ISSUE-LOCAL-01M2AA4ZVD9EWJG5NNQD5DWZXS
Title: 27B first-pass DRAM OOM: fragmented 4 GiB banks refuse a 1 GiB ternary output despite the W2 decode-plane reclaim
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

The qwen38-gguf-q4km-27b bootstrap (VT_DUMP_IDS=1, main 51c248190 tree, b8b365b15 build) dies 8.5 min into the first forward pass: TT_FATAL Out of Memory allocating 1073725440 B across 8 banks (134217728 B/bank, bank size 4272341376 B; allocated 4002304000 B, free 270037376 B, largest free block 119142976 B) from TernaryDeviceOperation::create_output_tensors (bank_manager.cpp:495). W2's reclaim fixed the decode-plane accumulation; the first pass itself still exhausts and fragments DRAM. Evidence log monitor-1789199424-6d47.

## Resolution

-
