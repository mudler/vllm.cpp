ID: ISSUE-GH-2199
Title: BACKEND-TENSTORRENT-QWEN35 spec Now still says W4 is owed after #2118 landed
Row: BACKEND-TENSTORRENT-QWEN35
State: CLOSED
Kind: bug
GitHub: 2199
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-28
Updated: 2026-08-28
Closed: 2026-08-28

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> **`.agents/specs/tenstorrent-qwen35.md`'s `## Now` says "Owed next: **W4 — cut the host staging wall** (this row's active gate)" and "Before W4, reconcile the stale parts of this spec: `## Git integration`'s base, and this section itself" — but W4 landed 2026-08-28 in #2118: `7ba0dfe1a` (levers 1+2, bulk bf16 staging + single-slot resolution, 0.104 → 0.177 tok/s, +70%; `Numel()` 27.09% → 1.76%) with review PASS recorded in `f99116ce2`.** The section's `## Git integration` also still pins base `origin/main` @ `8f5d4e4ed` (2026-08-27), behind the W4 and #2115 landings. The reconciliation the spec itself scheduled did not ride the change that made it stale.
>
> CONSEQUENCE: `scripts/now.py` renders the row's live "Next step" from this section, so the derived position surface tells every reader the row's active gate is a wave that already landed.
>
> FIX (record-only, no product code): rewrite `## Now` to the post-#2118 position — W4 landed with its measured numbers and its not-taken lever recorded; the #1486 teardown fix and the #2115 opt-out-arm pair (each arm gates its own captured pair, both legs doctest 146/146) landed after it; owed next = W3 leftovers (d2h counter completeness, `conv_transposed` fast-path check, tests for both), then the W4 record's named next lever (per-slot persistent device buffer written through the mesh command queue). Bump `## Git integration`'s base to the current tip `3fe34e2c6`. The now-landed `docs/USAGE.md` weights entry leaves that clause obsolete.
>
> Owner: row `BACKEND-TENSTORRENT-QWEN35`, spec `tenstorrent-qwen35.md`.

## Resolution

GitHub records closing pull request #2200 (https://github.com/mudler/vllm.cpp/pull/2200) merged on 2026-08-28 as commit `e216462da2bc8f7be5df8bccfe2434f6a00a5f64`. GitHub closed issue #2199 on 2026-08-28.
