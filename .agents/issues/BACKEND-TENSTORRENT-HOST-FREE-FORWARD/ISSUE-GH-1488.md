ID: ISSUE-GH-1488
Title: test_qwen3_paged_engine TT golden is stale: anchor drift prompt[1] tok=10 (engine=14126, committed=62901), unchanged by the #1476 fix
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: bug
GitHub: 1488
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-21
Closed: 2026-08-21

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Split out of #1476 so the fidelity fix can close without orphaning it.
>
> The TT arm of `tests/parity/test_qwen3_paged_engine.cpp` fails `REQUIRE(anchor_ok)` at `:345`:
>
> ```
> 1 anchor drift prompt[1] tok=10 engine=14126 committed anchor=62901
> ```
>
> Measured on thalia (Blackhole P150, TT Release) **identically before and after the #1476 fix** (same `engine=14126` value both arms, `ctest -R test_qwen3_paged_engine` exit 8 both arms) — so it is NOT the captured-decode page-table/cur_pos defect; this test runs the default path (no `VT_TT_HOST_FREE_DECODE`), which the fix does not touch.
>
> The golden was committed `971d5506` (2026-08-09) and 13 TT ops commits landed Aug 9-18 with no on-card golden re-run, so the anchor is stale rather than proof of a new defect. The near-tie gap protocol must re-adjudicate: run with `VT_DUMP_IDS`, feed the dump to the `qwen3-neartie-gap.py` adjudicator, and either refresh the golden (recording the re-derivation) or escalate the drift as a real divergence.
>
> Owning row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`, listed under `## Owed` in its spec (the operator-gate item that #1476 was filed from).

## Resolution

GitHub records closing pull request #1514 (https://github.com/mudler/vllm.cpp/pull/1514) merged on 2026-08-21 as commit `49c64bbc819fb8240d00253ef91e06232f9955e1`. GitHub closed issue #1488 on 2026-08-21.
