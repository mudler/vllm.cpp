ID: ISSUE-GH-1476
Title: TT host-free captured decode goes degenerate at the first KV block boundary; token-exactness evidence does not reproduce
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: bug
GitHub: 1476
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-21
Closed: 2026-08-21

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Operator gate for #1105 (BACKEND-TENSTORRENT-HOST-FREE-FORWARD), run 2026-08-20 on the P150 host at main `206afb63`, reproduced at the landed SHA `79ff8f31` in fresh clean builds (`VLLM_CPP_CUDA=OFF`, `VLLM_CPP_TENSTORRENT=ON`, tt-metal runtime unchanged since 2026-08-09):
>
> **PASS:** the 80-token no-hang gate. `VT_TT_HOST_FREE_DECODE=1` vllm-cli Qwen3-0.6B "Hello" `--max-tokens 80` completes 79 replays, exit 0, at both SHAs — the ~38-replay wall stays cleared.
>
> **FAIL — fidelity:** captured replay is deterministic degenerate output while host-free eager (same flags, `VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH=0`) is coherent:
> - captured: `...The problem is: "A man is walking on a straight line line line on road Let straight straight A A...` (word salad from ~generated token 30)
> - eager: `...The problem is: "A man is walking on a straight line. He starts at point A, walks 100 meters to the right, then turns 90 degrees...`
>
> Reproduced 5/5 across two independent build dirs at `206afb63`, and 1/1 at `79ff8f31` — so this is **not a main regression**: the landed PR carries it. The recorded 22/22 argmax (m57-vs-m58, 2026-08-16 session checkpoint) predates the final on-device `cur_pos` plus_one integration on the R2 branch and does not reproduce on the landed tree.
>
> Coherence through ~generated token 30 with `block_size=32` and a 1-token prompt points at the first KV block boundary: page-table refresh / on-device `cur_pos` advance across `cols_changed` re-capture (qwen3.cpp `Reset()` path) is the first suspect, ahead of the plus_one ordering itself.
>
> **Also found, same gate run:** `test_qwen3_paged_engine` on TT fails `REQUIRE(anchor_ok)` — anchor drift prompt[1] tok=10 (engine=14126, committed=62901) against the TT golden committed `971d5506` (2026-08-09); 13 TT ops commits landed Aug 9–18 without an on-card golden re-run, so the anchor is stale rather than proof of a new defect, and the near-tie gap protocol must re-adjudicate. A teardown SEGFAULT (`ttnn::Tensor::deallocate_impl` -> `GraphTracker::is_enabled`) followed the failure once on main; not reproduced at the landing SHA.
>
> Owning row: `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (the fidelity item of its owed gate). Fix directions: instrument the replay regime around the first block-boundary re-capture (`VT_TT_TRACE_DEBUG`), verify `WarmPaMeta`/`WarmDecodePos` re-seed after `Reset()`, and re-run the captured-vs-eager diff before any golden work.
>
> Helper note: early per-commit attribution in this session was invalid — a scratch worktree built without the tt-metal `CMAKE_PREFIX_PATH` silently produced a CPU-only binary whose coherent output masked the defect. TT runs are identified by UMD log lines and `Asynchronous scheduling is disabled` (TT never overrides `SupportsAsyncSampledTokenReadback`); CPU says enabled.

## Resolution

GitHub records closing pull request #1498 (https://github.com/mudler/vllm.cpp/pull/1498) merged on 2026-08-21 as commit `d27639e71150557a2d9aafaf9c55108c621ac419`. GitHub closed issue #1476 on 2026-08-21.
