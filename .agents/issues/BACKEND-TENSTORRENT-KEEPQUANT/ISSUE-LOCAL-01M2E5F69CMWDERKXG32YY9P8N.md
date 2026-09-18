ID: ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N
Title: EnsureHost downloads the parent plane for a 16-row window on the vllm-bench path (qwen3.8-27B TT)
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

The 27B P150 benchmark through vllm-bench (AsyncLLM entry, production path) fails after load+capture: VT_CHECK in DownloadToHost (src/vt/tenstorrent/tenstorrent_ops.cpp:484) sees got=131072 want=2048, out_shape=1x16x128, ctx=EnsureHost, dev[1024x128 dt=0 lay=1]. The host tensor is a 16-row window; the slot lookup resolved to the parent 1024x128 device buffer. This is the full-extent-window alias family that W4d guarded for the word shadow, permute buffer, and single-chunk windows; the bench's 64-token chunked-prefill shape family reaches an unguarded path. The 16-prompt doctest gate does not exercise it, so the row is gate-green but not benchable. Evidence: /tmp/bench-smoke5.log (TT-SLOT MISMATCH backtrace), smoke3/4 logs record the default-arm 636 MB lm_head OOM that forces the VT_TT_KEEPQUANT_INT8DOT=1 recipe.

## Resolution

-

## Progress 2026-09-13 (evening)

Three distinct legs found while driving the 27B P150 benchmark through
vllm-bench (all evidence in /tmp/bench-smoke{5,8,9,10}.log):

1. ServeActF32 fell through interior windows straight to the host
   fallback, which downloaded the OWNER plane. Fixed: ServeActF32 now
   tries ServeDeviceWindow after the exact shadow (the ServePostConvAB
   pattern), and ServeDeviceWindow's geometry check generalizes to
   contiguous multi-rank views (decode v/g/beta arrive as [1,bh,ud]).
   Red: smoke5 (want 2048 got 131072). Green: smoke8 passes the decode
   serve.
2. EnsureHost had no interior-window path at all for the host-staged
   GdnPrefillKernel reads. Fixed: a contiguous interior view of a
   device-authoritative owner downloads the owner once and copies the
   flat span, WITHOUT marking the owner slot host_current (only the
   window's bytes were filled). Red: smoke8 (want 129024 got 131072).
3. OPEN: a producer commits per-token [1,48,128] device planes into a
   slot whose host buffer spans the whole sequence ([63,...] window at
   delta=12288 = token 2; smoke10 diagnostics). The slot's
   host_current=false then wrongly declares the entire host buffer
   stale although earlier tokens' host bytes are still valid (written
   by the host producer). A download guard cannot fix this; it needs
   the producer call site and either per-plane slot tracking or a
   commit that keeps the host bytes current. Next session: identify
   the committing op (bt in smoke10; addr2line the GdnPrefillKernel /
   dense_attn frames), then fix the tracking granularity.

Not yet committed; the fix lives in the /tmp/row-tt-w3 worktree
(src/vt/tenstorrent/tenstorrent_ops.cpp). The 27B bench remains
blocked on leg 3.
