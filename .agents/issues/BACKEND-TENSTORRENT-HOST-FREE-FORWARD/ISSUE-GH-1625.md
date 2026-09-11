ID: ISSUE-GH-1625
Title: Tenstorrent captured decode hangs on the first multi-request run (single-request captured legs work)
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: OPEN
Kind: bug
GitHub: 1625
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`
>
> ## What
>
> With host-free decode + graph capture enabled on the Blackhole P150, a MULTI-REQUEST run hangs deterministically ~10 s into stepping: one tt-metal worker thread spins at 100% CPU while the main thread blocks. Single-request captured runs (the 27.1 tok/s A/B legs, the 80-token captured-vs-eager gate) never hit it.
>
> ## Reproduction (2026-08-21, P150, tree b86e3705f = main 52e328789 + the R5 flip)
>
> - `VT_DUMP_IDS=1 ./build/tests/test_qwen3_paged_engine` (captured default): hangs. Last tt-metal log line is the allocator warning `Allocating device buffers is unsafe due to the existence of an active trace` (allocator.cpp:123), then silence; worker at 100%, main thread sleeping. Killed after 11 min.
> - Same with `VT_TT_RECAPTURE_EVERY=8` (traces destroyed every 8 replays): hangs identically at the same point — per-trace replay count is ruled out.
> - `VLLM_CPP_CUDAGRAPH=0` (host-free EAGER, no capture): same gate completes in 35 s, 125/125 assertions green.
> - Host-free eager decode rate: 10.94/10.95/11.06 tok/s warm (Qwen3-0.6B, batch 1, greedy, --repeat 4, leg 1 discarded — JIT) vs 5.34 pre-flip default; captured single-request is 27.1.
>
> ## Hypotheses (not diagnosed)
>
> - Eager alloc/free churn while a decode trace is live across the request boundary (the class qwen3.cpp's Step() comment records: "eager alloc/free churn around a live trace hung the device ~60 replays in"). The allocator warning supports an eager allocation around a live trace, but RECAPTURE_EVERY=8 bounding live-trace windows did not help, so the churn window may be inside the capture/replay cycle itself rather than between requests.
> - A between-request state interaction the single-request flow never exercises (KV block free + re-alloc, prefill after replay).
>
> ## Consequence
>
> #1604 lands with `support_static_graph_mode()` declined by default on TT (opt-in via `VT_TT_DECODE_CAPTURE`), so the default is the hang-free 11.0 tok/s host-free eager path. Flipping capture back on by default is owned by this issue.
>
> Owner: row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (spec `.agents/specs/tenstorrent-host-free-forward.md`, R5).

## Resolution

-
