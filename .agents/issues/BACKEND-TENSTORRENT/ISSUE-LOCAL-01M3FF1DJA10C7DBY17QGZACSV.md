ID: ISSUE-LOCAL-01M3FF1DJA10C7DBY17QGZACSV
Title: The dense captured arm never serves a replay: the #2469 continuation port is missing its two post-replay cur_pos increments, so every decode step re-warms eagerly (67.0 s/request of the 35 s/token wall)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-26
Closed: -

## Problem

The 27B step decompose (docs/bench-evidence/tt-27b-step-decompose-20260926.md, VT_TT_STEP_PHASES instrument) measured the captured arm's decode as [cold 33.9 s -> capture 5.7 s (+31.1 s trace wait) -> cold 33.2 s] per request, with boundary=1 at EVERY decode step and VT_DECODE_GRAPH_STATS reporting '4 total replays' for a 12-step run — every capture's own launch, never a served replay. Mechanism: Qwen3_5DenseDecodeGraph::Step updates s.expected_cur_pos = seq_lens[0]-1 (qwen3_5.cpp:12400) and checks (seq_lens[0]-1) == s.expected_cur_pos (qwen3_5.cpp:12395), but the #2469 port from qwen3.cpp is missing the two post-replay increments — qwen3.cpp:971 ('++s.expected_cur_pos; // the replay's in-trace plus_one advanced cur_pos', the replay branch) and qwen3.cpp:1118 ('++s.expected_cur_pos; // the capture step's launch ran the trace once', the capture-tail branch). The predicate is therefore false at every mid-request step, tt_boundary fires, s.graph.Reset() (qwen3_5.cpp:12437) destroys the freshly captured trace, and the slot collapses to cold — two of three steps run the full 64-layer forward EAGERLY at ~33.5 s of host dispatch each (the device work is 1.27 s; the legD probe measured gdn 20.3 ms / fa 18.6 ms per layer). The fix is the two missing increments in the dense driver's do_replay branch and capture-tail branch, mirroring qwen3.cpp:971/:1118; expected value ~-6% TPOT directly (the third step becomes a 31.1 s trace wait instead of a 33.2 s eager pass) and it is the precondition for every captured-arm lever to be measurable (INT8DOT re-measure, M=2 batching, the trace-replay per-command overhead that owns the remaining 31.1 s).

## Resolution

-
