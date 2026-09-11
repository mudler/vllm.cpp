ID: ISSUE-GH-838
Title: ROCm Gemma-4 hyp A: T=2..63 MoE is observed on serial M1; widen indexed gate
Row: ROCM-GEMMA4-INDEXED-MAX-T
State: CLOSED
Kind: perf
GitHub: 838
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-14
Updated: 2026-08-21
Closed: 2026-08-21

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
>
> Route Gemma-4 FP8 MoE T=2..63 through the existing per-token indexed helpers so short warmups skip today's serial `RunGemma4Fp8TopKOnExpertDevice` path. Port lab `VT_GEMMA4_DECODE_INDEXED_MAX_T` default 63.
>
> ## Hypothesis (A) — cause unconfirmed
>
> T=19 warmup is **observed** on the serial M1 route. That does **not** prove the serial path is the hang cause. Distinct from the T≥64 prefill-helper accumulation class. Do not call the serial path "racy."
>
> ## Context
>
> `origin/main` `gemma4_moe.cpp:735` takes indexed only at `T == 1`. T=2..63 falls through to the serial loop (`:1345`) `RunGemma4Fp8TopKOnExpertDevice`. Donor bytes are pinned under `.agents/evidence/rocm-gemma4-indexed-max-t/` (dirty lab `2bb4bd8a` plus uncommitted; HEAD is not a clean donor). Indexed helpers already exist on main (`rocm_gemma4_experts.hip:543`).
>
> Separate from #697, #837, #839. Independent branch `row/ROCM-GEMMA4-INDEXED-MAX-T`.
>
> Direction: research `713f` / `5071` / `64cb` + coord `25c9`.
>
> ## Scope
>
> - Widen the existing T==1 indexed gate to `1 <= T <= indexed_max_t && T < kPrefillBatchMinT` (64).
> - Env `VT_GEMMA4_DECODE_INDEXED_MAX_T`: unset → 63; `=1` → T=1 only; clamp [1,63].
> - T=1 keeps hipGraph-stable TLS acc; T>1 uses an owned `[T,H]` buffer.
> - Per-token existing indexed helpers only (same-dev and peer).
> - Direct tensor oracle T=2,19,63 × {same-dev, peer} vs today's serial/reference math.
>
> ## Out of scope
>
> - `ExpertGeGLUFp8TopKIndexedBatched` / packed indexed-batch (already REJECTED).
> - `VT_GEMMA4_PREFILL_DEVICE_GROUP`, `VT_GEMMA4_PREFILL_INDEXED_NOSYNC`.
> - Prefill Launch/Finish helper; GetBlas dual-slot; #697.
>
> ## Gates
>
> - Host predicate: T=1 indexed; T=19 indexed at default 63; T=64 not indexed (prefill-batch); env=1 ⇒ T=19 not indexed.
> - Tensor oracle GREEN before claiming A GREEN (Paris/arith is not enough).
> - T=19 generate succeeds; T=1 decode / Paris / arith unchanged.
> - Do not claim p42k from this row.
>
> Kind: bug
>

## Resolution

GitHub records closing pull request #1046 (https://github.com/mudler/vllm.cpp/pull/1046) merged on 2026-08-21 as commit `b806592768c4f985718bd44c62e3cd1d2873b5dd`. GitHub closed issue #838 on 2026-08-21.
