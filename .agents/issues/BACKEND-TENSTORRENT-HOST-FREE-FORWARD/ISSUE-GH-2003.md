ID: ISSUE-GH-2003
Title: TT P150 at post-W2c main: the host-hybrid opt-out outperforms the host-free eager DEFAULT by 1.24x — #1604 flip premise inverted
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: perf
GitHub: 2003
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-26
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`
>
> ## What changed
>
> #1604 flipped `VT_TT_HOST_FREE_DECODE` default-ON on 2026-08-21 after measuring
> default **10.94–11.06 tok/s** vs opt-out **5.34** (Qwen3-0.6B, b1, greedy,
> `thalia` P150, base `52e328789`). Re-measured 2026-08-26 at `21fe11cf1`
> (origin/main, post-BACKEND-TENSTORRENT-QWEN35 W2a–W2c):
>
> | arm | median tok/s | n | range |
> |---|---|---|---|
> | default (host-free eager) | **10.822** | 12 | 10.51–11.03 |
> | `VT_TT_HOST_FREE_DECODE=0` (host-hybrid) | **13.369** | 12 | 13.09–13.55 (one outlier leg 11.49) |
>
> Method: order-alternated pairs ×3 (`--repeat 5`, in-process run 1 discarded),
> one `$HOME/gpu.lock` hold, `tt-smi -r` first, box otherwise idle.
> **The opt-out wins 1.24×.** The default arm is unchanged against its
> 2026-08-21 figures (within the band given no clock sampling — see caveats),
> so the movement is ~2.5× improvement in the OPT-OUT arm since then,
> mechanism UNATTRIBUTED. Next traceable step: per-op delta of the host-hybrid
> path between `b86e3705f` and `21fe11cf1`.
>
> ## Why it matters
>
> The shipped default now serves the slower of the two eager arms. Any gate
> that uses the default as denominator inherits the slower figure.
>
> ## Caveats stated plainly
>
> - No clock window was sampled (`tools/bench/gpu_clock_state.py` is
>   NVIDIA-only); all figures here are clock-unattributed. Ordering was
>   alternated to cancel drift, but SM/board-clock attribution is owed.
> - One model shape (b1, 96-token completions), one board (Blackhole P150,
>   aarch64 host), tt-metal HEAD `a3d33028975` (carries our copy_default_tilized
>   patches).
> - The captured opt-in arm (27.1 tok/s single-request on the old tree) is NOT
>   retested; multi-request capture still hangs (#1625), and TT async readback
>   remains #1627.
>
> ## Ownership
>
> Row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`; recorded under the row spec's
> `## Owed` in the same change that lands the refreshed records.
>

## Resolution

GitHub records closing pull request #2368 (https://github.com/mudler/vllm.cpp/pull/2368) merged on 2026-08-31 as commit `511115a1bec95c21756e02fd320628ba4cea9ea3`. GitHub closed issue #2003 on 2026-08-31.
