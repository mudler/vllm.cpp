ID: ISSUE-GH-1696
Title: The landed capture-decline arms are vacuous under an ambient VT_TT_HOST_FREE_DECODE=0: the unit gate reds 833/834 on the opt-out leg
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: 1696
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-22
Updated: 2026-08-23
Closed: 2026-08-23

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> The #1630 squash (`333509dc5`) landed the 3-arm capture-decline block in
> `tests/vt/test_tenstorrent_backend.cpp` ("kTENSTORRENT Platform mirrors the
> registered Backend", ~lines 75-85) — the version whose opt-in arm implicitly
> assumes `HostFreeDecodeEnabled()` is ON:
>
> ```cpp
> CHECK_FALSE(p.support_static_graph_mode());
> ::setenv("VT_TT_DECODE_CAPTURE", "1", 1);
> CHECK(...support_static_graph_mode());   // reds under ambient opt-out
> ```
>
> `support_static_graph_mode()` is the conjunction
> `HostFreeDecodeEnabled() && VT_TT_DECODE_CAPTURE` (`src/vllm/platforms/tenstorrent.cpp:81-84`),
> so under an ambient `VT_TT_HOST_FREE_DECODE=0` — the documented pre-flip
> opt-out — the conjunction is false regardless of the capture env and the
> opt-in CHECK fails:
>
> ```
> VT_TT_HOST_FREE_DECODE=0 ./build/tests/test_tenstorrent_backend
>   ...:83: ERROR: CHECK( ... support_static_graph_mode() ) is NOT correct!
>   assertions: 834 | 833 passed | 1 failed
> ```
>
> Reproduced on thalia (P150) at the pre-merge branch head; the default leg is
> green, which is why CI never sees it. The truth-table repair (pin all four
> conjunction cells, both envs set per cell, ambient state saved/restored) was
> written, red-first-proven (mutation A drops the capture conjunct → cell 2
> reds; mutation B drops the host-free conjunct → cell 4 reds), and verified
> 835/835 on BOTH ambient legs — but the branch push missed the merge window
> by minutes, so main carries the 3-arm version. Found during the #1630 merge
> follow-up; fixed in flow.

## Resolution

GitHub records closing pull request #1699 (https://github.com/mudler/vllm.cpp/pull/1699) merged on 2026-08-23 as commit `8eecc05a96bd07b5a3307b73576c7b5302690d4a`. GitHub closed issue #1696 on 2026-08-23.
