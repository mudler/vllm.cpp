ID: ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW
Title: 27B decode exhausts tt-metal DRAM after ~N requests per process (bank_manager OOM in the keepquant decode path)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

Spec: [tenstorrent-27b-dram-growth](../../specs/tenstorrent-27b-dram-growth.md) (attribution-first, 2026-09-24).

The APEX-I-Nano 27B decode path OOMs on tt-metal at bank_manager.cpp:495 (~268 MB DRAM allocation against ~63 MB largest free block) in Qwen3_5DenseDecodeGraph::Step -> DecodeKeepQuantWordsF32 -> ttnn::where. One 64-request process dies after ~60 requests; 16-request after ~14; 8-request after ~7; single-request processes complete. The growth-with-requests pattern indicates a leak or unbounded allocation accumulation on the decode path. This blocks the INT8DOT wide band sweep (gate 1), the 27B Q4KM sibling gate, and by extension the 27B trace-capture row. Evidence: docs/bench-evidence/tt-int8dot-flip-gates-20260923.md on row/TT-KEEPQUANT-INT8DOT-DEFAULT (2026-09-23/24, P150).

## Resolution

-
