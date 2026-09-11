ID: ISSUE-GH-837
Title: ROCm Gemma-4 hyp B: GetBlas single TLS destroys hipBLAS handle on peer-MoE device hop
Row: ROCM-GEMMA4-GETBLAS-DUALSLOT
State: CLOSED
Kind: bug
GitHub: 837
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-14
Updated: 2026-08-17
Closed: 2026-08-17

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
>
> Stop hipBLAS handle destroy/create on every compute↔expert device hop during Gemma-4 dual-GPU MoE. Port the dirty-lab dual-slot `GetBlas` (`tls_slots[2]`) onto current `origin/main`.
>
> ## Hypothesis (B) — cause unconfirmed
>
> `9772` is an accumulation failure class, not a confirmed GetBlas root. T≥64 prefill-batch peer path is **observed** to call `MatmulBT` → `GetBlas` on the expert queue. Serial T=19 does **not** use this `GetBlas`. Do not claim this row alone clears T=2029.
>
> ## Context
>
> Worktree / `origin/main` `GetBlas` (`src/vt/rocm/rocm_matmul_hipblaslt.hip:72-99`) is a single `static thread_local Tls tls`. On `tls.dev != device` it `hipblasDestroy` + `hipblasCreate`. Donor bytes are pinned under `.agents/evidence/rocm-gemma4-getblas/` (dirty lab `2bb4bd8a` plus uncommitted; HEAD is not a clean donor).
>
> Separate from #697, #838, #839. Independent branch `row/ROCM-GEMMA4-GETBLAS-DUALSLOT`.
>
> Direction: research `713f` / `5071` / `64cb` + hermes `82b2` + coord `25c9`.
>
> ## Scope
>
> - `GetBlas` only: `static thread_local Tls tls` → `tls_slots[2]` + `Tls& tls = tls_slots[(device == 1) ? 1 : 0]`.
> - Preserve existing capture/`hipSetDevice`/`hipblasSetStream` logic inside a slot.
> - Host lifetime seam must observe 0→1→0 and 1→0→1 handle identities (text search is not enough).
>
> ## Out of scope
>
> - Prefill Launch/Finish / PeerPipeTls / DequantCache (#839).
> - Indexed T<63 routing (#838).
> - hipBLASLt product default, FP8×FP8 Lt, #697 / `rocm_paged_attn.hip`.
> - Devices other than 0/1.
>
> ## Gates
>
> - Host lifetime seam GREEN without a GPU; mutations for swapped selector / destroy-on-hop / missing stream rebind / capture-path setDevice must RED.
> - T=1 decode / Paris / arith unchanged vs pre-change binary.
> - T=2029 generate must not be claimed from this row alone.
>
> Kind: bug
>

## Resolution

GitHub records closing pull request #1045 (https://github.com/mudler/vllm.cpp/pull/1045) merged on 2026-08-17 as commit `559973cb8d7691d44554d859bfc4b9e8639abeae`. GitHub closed issue #837 on 2026-08-17.
