ID: ISSUE-GH-1105
Title: Tenstorrent host-free decode graph: capture, replay, and on-device state advance
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: OPEN
Kind: feature
GitHub: 1105
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-17
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`
>
> ## Summary
>
> The Tenstorrent Blackhole decode path cannot run a captured mesh-trace while any `to_vector` / host write sits inside the captured region. The trace-runner spike measured that flipping the T=1 hybrid thresholds is not enough: capture aborts on `Reads are not supported during trace capture`.
>
> This issue owns:
>
> - `BACKEND-TENSTORRENT-TRACE-RUNNER` (SPIKE, decision record)
> - `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (ACTIVE: R1-R3b plus R2 on-device `cur_pos` advance)
>
> ## What landed on PR #1053
>
> Env-gated `VT_TT_HOST_FREE_DECODE` decode-graph capture. On a Blackhole P150, Qwen3-0.6B "Hello" at 80 tokens completed 79 replays with no hang, 5.8x vs eager, 22/22 argmax vs the per-step-copy baseline. Default path is inert.
>
> ## Still owed
>
> - Fresh review of the host-free decode graph
> - Operator rerun of the 80-token no-hang gate and the TT golden on card
> - `test_qwen3_paged_engine` still times out under the flag
>
> ## Specs
>
> - `.agents/specs/tenstorrent-host-free-forward.md`
> - `.agents/specs/tenstorrent-host-free-r1.md`
> - `.agents/specs/tenstorrent-host-free-r2.md`
> - `.agents/specs/tenstorrent-trace-runner.md`

## Resolution

-
