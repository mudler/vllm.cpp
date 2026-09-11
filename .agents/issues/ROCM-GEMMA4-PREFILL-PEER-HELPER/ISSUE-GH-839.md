ID: ISSUE-GH-839
Title: ROCm Gemma-4 hyp C: prefill peer GeGLU helper is monolithic; retirement-safe Launch/Finish
Row: ROCM-GEMMA4-PREFILL-PEER-HELPER
State: CLOSED
Kind: perf
GitHub: 839
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-14
Updated: 2026-08-18
Closed: 2026-08-18

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
>
> Replace the monolithic single-TLS `RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice` peer path with lab's resource-managed Launch/Finish + `PeerPipeTls` slots + `DequantCacheSlotFor(expert_dev)` pin that persists until GPU consumption retires. This is the T≥64 / p42k-critical path.
>
> ## Hypothesis (C) — cause unconfirmed
>
> Coord `9772`: 274 matched `moe_prefill_peer_helper` BEGIN/END then accumulation wedge. Individual calls return; repeated invocation is **observed** to wedge. Cause (single-slot TLS vs cache eviction vs GetBlas) is unconfirmed. Do not say repeated invocation "exhausts" single-slot state.
>
> ## Context
>
> `origin/main` helper (`rocm_gemma4_experts.hip:648`) is one function: single `static thread_local Tls tls` / `SameTls tls`, sticky FP8→BF16 dequant, monolithic peer launch + `ev_c`/`ev_e`. Donor bytes are pinned under `.agents/evidence/rocm-gemma4-prefill-peer/` (dirty lab `2bb4bd8a` plus uncommitted; HEAD is not a clean donor).
>
> Donor Launch unpins before `ev_e`; Finish does not host-wait. Product must **not** copy that hole: pin lives in `PeerSlot` until host-observed `ev_e` completion.
>
> `kPeerPipe` stays default OFF.
>
> Separate from #697 and from the T<63 serial route. Independent branch `row/ROCM-GEMMA4-PREFILL-PEER-HELPER`.
>
> Direction: research `713f` / `5071` / `64cb` + coord `25c9`.
>
> ## Scope
>
> - Split peer execution into Launch + Finish; wrapper Launch(slot0)→Finish when pipe off.
> - `PeerPipeTls` slots; `DequantCacheSlotFor(expert_dev)` pin stored on `PeerSlot`.
> - Same-dev dual `SameTls` slots (device 0/1) with the same retirement-safe unpin.
> - `kPeerPipe` default OFF.
>
> ## Out of scope
>
> - Enabling `kPeerPipe` / overlapping slots as a product default.
> - `VT_GEMMA4_PREFILL_FP8_LT`, `VT_GEMMA4_GU_INTERLEAVE`.
> - GetBlas dual-slot (#837; land first or with this as a **separate** head).
> - Indexed T<63; #697 / `rocm_paged_attn.hip`.
> - Diagnostic `STAGE_SYNC` / `PREFILL_TRACE`.
> - Donor unpin-before-`ev_e`.
>
> ## Gates
>
> - Default pipe-off: every Launch has a Finish; pin count returns to 0; pin stays >0 until host-observed `ev_e`.
> - Concurrent-eviction + fail-after-pin/event/wait/copy mutations RED if lifetime is broken.
> - T=2029 generate with `PREFILL_TRACE=1` / `STAGE_SYNC` unset: matched BEGIN/END and HTTP body.
> - T=1 decode / Paris / arith unchanged.
> - p42k only after coord smoke of this row + GetBlas.
>
> Kind: bug
>

## Resolution

GitHub records closing pull request #1047 (https://github.com/mudler/vllm.cpp/pull/1047) merged on 2026-08-18 as commit `0794999eb3d0c41acc5560f58880c4edb6ae069f`. GitHub closed issue #839 on 2026-08-18.
