ID: ISSUE-GH-2115
Title: TT host-free opt-out leg (VT_TT_HOST_FREE_DECODE=0) of the Qwen3.5-0.8B sacred e2e drifts one anchor token: prompt[2] tok=1, engine 15039 vs anchor 1814 — deterministic and pre-existing on main
Row: BACKEND-TENSTORRENT-QWEN35
State: CLOSED
Kind: bug
GitHub: 2115
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-27
Updated: 2026-08-28
Closed: 2026-08-28

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Owning row: `BACKEND-TENSTORRENT-QWEN35` (it owns the golden pair). Found during the W4 gate run (#2107), and proven NOT a W4 regression.
>
> ## Measured (2026-08-27, P150, one lock hold per run)
>
> `tests/test_qwen35_paged_engine` under `VT_TT_HOST_FREE_DECODE=0` fails with exactly one anchor drift: prompt[2] token 1, engine produces **15039**, committed anchor says **1814**. Byte-identical across:
>
> 1. base `0ac84a486` (W4 stashed, rebuilt),
> 2. the W4 worktree (run 1),
> 3. the W4 worktree (run 2, reproducibility).
>
> The ambient leg (`VT_TT_HOST_FREE_DECODE` unset) is 16/16 PASS, 0 forward-divergent, max gap 375 mnats — only the opt-out arm drifts. Evidence: `docs/bench-evidence/tt-qwen35-eager-leg2-anchor-drift-20260827.log` (committed on the W4 branch).
>
> ## What this means
>
> The golden pair was re-derived at `c31cad9c1` (16/16 reported); something between that re-derivation and `8f5d4e4ed` moved the opt-out arm's token — candidates include the KV-GDN-STATE-BUDGET landing (`2a42cb369`) or the W2c residency chain, but nothing has bisected it yet. Per the gate's own rule: **bisect the engine change first; only re-derive goldens (VT_DUMP_IDS=1 + qwen3-neartie-gap-transformers.py) after the drift is proven to be a justified numerical change, never to silence the gate.**
>
> ## Owed
>
> - Bisect the drift to the landing that moved it.
> - Either fix the engine defect or re-derive the golden through the sanctioned procedure with the justification recorded.
> - Until then, the W4 spec's gate-3 "=0 leg" line is red on main independent of W4; the W4 PR reports it rather than waives it.

## Resolution

GitHub records closing pull request #2189 (https://github.com/mudler/vllm.cpp/pull/2189) merged on 2026-08-28 as commit `3fe34e2c67f7cf26aaad1bbe4a7d4445c47e296e`. GitHub closed issue #2115 on 2026-08-28.
