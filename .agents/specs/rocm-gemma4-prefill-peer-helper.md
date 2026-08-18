# Spec: ROCm Gemma-4 prefill peer helper Launch/Finish + dequant-cache

- **Issue:** https://github.com/mudler/vllm.cpp/issues/839
- **Row slug:** `ROCM-GEMMA4-PREFILL-PEER-HELPER` — child of `BACKEND-ROCM` (#41). Separate from #697.
- **Worktree / branch (this unit only):** `/home/don/llms/vllm.cpp-prefill-peer` · `row/ROCM-GEMMA4-PREFILL-PEER-HELPER`
- **Base / recipient:** `origin/main` `3ce5a1dc` `rocm_gemma4_experts.hip:648`
- **Donor bytes:** `.agents/specs/rocm-gemma4-prefill-peer-helper-donor-*.log` slices hashed in [`rocm-gemma4-prefill-peer-helper-donor.md`](rocm-gemma4-prefill-peer-helper-donor.md) (dirty lab `/home/don/llms/vllm.cpp` HEAD `2bb4bd8a` **plus uncommitted**; HEAD is not a clean donor).
- **Implementer:** hermes-vllm. **Reviewer:** research. **Operator/smoke:** coord.
- **Git:** spec-only first (coord `25c9` / research `5071` / BLOCK `64cb`); impl after spec GREEN **on this same row branch**. Independent RED/GREEN from #837/#838. Ordered B then C is allowed; no all-at-once port. One PR per row. No shared `row/ROCM-GEMMA4-XDEV-MOE` landing history.
- **Depends on:** #837 is a separate hypothesis. Operator runs B then C on T=2029. Impl of C may assume B already landed **or** must still be A/B-able alone.
- **Supersedes for review:** `20332292` (BLOCK), preview `c4fbe6e9` (not spec-GREEN), and `231f38cf` (BLOCK `4954`).

## Now

`IMPLEMENTING` — ea9c: product RestoreComputeOrThrow no-op must RED via compile-and-run of the HIP body.

## Adjacent upstream (not this row)

- **#785**: `VT_ROCWMMA_OK` / #697 landing-order only. Do not touch `rocm_paged_attn.hip`.
- **#523 / #509**: custom keep-quant grouped GEMM + `rocm_moe_chain.hip`. No GetBlas/Launch/Finish overlap; docs-only if FEATURES/USAGE touch.
- **#834**: router-lookahead prefetch — adjacent cache policy, not this pin lifetime.

**Not a confirmed fix.** Hypothesis (C) only. `9772` does not isolate Launch/Finish vs dequant-cache vs GetBlas. Repeated invocation is **observed** to wedge after matched BEGIN/END. Cause (single-slot TLS vs cache eviction vs GetBlas destroy) is **unconfirmed**. Do not say repeated invocation "exhausts" single-slot state.

This package stays one row because the lab wrapper is one ownership unit; tests must still fail independently if pin-lifetime or Launch/Finish pairing is broken.

## Upstream / source of the port

No vLLM Python equivalent. Source is the pinned donor slices for **structure**. Product lifetime is **stricter** than the donor (see Design).

| Tree | Symbol | Shape |
|---|---|---|
| `origin/main` `3ce5a1dc` | `RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice` `:648` | one function; same-dev `static thread_local SameTls tls` + sticky; peer `static thread_local Tls tls` + inline copy/GEMM/`ev_c`/`ev_e` |
| hanging `vllm.cpp-bc64fa-r2` `1b1baf43` | `:692` | same monolithic body |
| donor slices | wrapper `:959-1095`; `Launch…` `:1098`; `Finish…` `:1274`; `PeerPipeTls` `:55`; cache core `:60-209`; `DequantCacheSlotFor` `:225` | wrapper Launch(slot0)→Finish; `PrefillDequantCache` Slot/FreeAll/Ensure/GetLocked; same-dev `SameTls tls_slots[2]` + cache pin/unpin |

`kPeerPipe` in lab is **default OFF**. Even OFF, the wrapper + dequant-cache lifetime is the delta (`713f`).

## Symptom this row owns

T≥64 / T=2029 prefill-batch peer path. Coord `9772`: every individual `RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice` returns; 274 BEGIN/END match; then idle + `kfd_wait_on_events`. Accumulation **class**, not a proven single-op hang. p42k-critical. Cause unconfirmed.

## Scope

Port the **resource-managed** lab shape onto main's helper, then close the donor pin hole:

1. Keep the public symbol `RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice` as a wrapper.
2. Same-dev: `SameTls tls_slots[2]`; weights via `DequantCacheSlotFor(compute_dev)` pin → GEMM → **retire → unpin**. No sticky `const void*` in the act TLS.
3. Peer: `LaunchGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, slot=0, …)` then `FinishGemma4Fp8ExpertGeGLUPrefillPeer(...)`.
4. `PeerPipeTls().s[slot]` holds x/y/events/expert queue **and the live cache pin** for that slot.
5. `DequantCacheSlotFor(expert_dev)` pin around expert-side dequant+GEMM; unpin only after GPU consumption retires.
6. `kPeerPipe` / any overlap env **default OFF**. Pipe-on is out of this slice.

Caller in `gemma4_moe.cpp` prefill-batch peer-act chunk loop stays the public wrapper. Do not change gather/scatter or `kPrefillBatchMinT`.

## Out of scope

- Enabling peer-pipe overlap as product default.
- `VT_GEMMA4_PREFILL_FP8_LT` (KEEP-rejected) and `VT_GEMMA4_GU_INTERLEAVE`.
- GetBlas body (#837) — but impl of this row must not ship without #837 landed or bundled as a **separate** immutable head.
- Indexed T<63 (#838).
- `#697` / `rocm_paged_attn.hip`.
- Diagnostic `STAGE_SYNC` / `PREFILL_TRACE` / M1 drains / `hipSetDevice` added only for prints.
- Transplant of dirty-lab layer-split / FIFO / resident-pack extras.
- Copying donor unpin-before-`ev_e`.

## Design

`kPeerPipe` OFF means Launch and Finish are back-to-back on slot 0 — **same event order as today's wait-then-y**, plus slot-scoped scratch.

### Donor hole (must not ship)

Donor Launch (`rocm-gemma4-prefill-peer-helper-donor-launch-finish-1090-1310.log`) unpins at ~1262–1265 immediately after enqueueing GEMMs and **before** `hipEventRecord(tls.ev_e)`. Finish ~1274–1300 enqueues `hipStreamWaitEvent(cst, tls.ev_e)` + peer copy and does **not** host-wait `ev_e`. `DequantCacheSlotFor` is process-wide (220–231), so another worker/stream may evict/rewrite a zero-pin slot while the first expert stream still reads it. "Unpin after Finish" is still insufficient unless Finish proves `ev_e` complete.

Donor `Ensure` (`rocm-gemma4-prefill-peer-helper-donor-dequant-cache-core-60-209.log`) calls `FreeAll` whenever device/I/H/`nslots` disagree with the request. `FreeAll` `hipFree`s every slot and zeros `pins`, including live pins. Product must not copy that.

### Product ownership (required)

Persist the pin in `PeerSlot`:

```
PeerSlot {
  ... existing x/y/eq/ev_c/ev_e/pending_M/edev ...
  int cache_pin = -1;          // slot index inside DequantCache
  int cache_dev = -1;          // device whose DequantCacheSlotFor owns it
}
```

Rules:

1. Launch pins under the same mutex scope as donor `GetLocked` (lock → GetLocked → unlock → GEMM). Store `{cache_pin, cache_dev}` on the `PeerSlot` **before** any GEMM enqueue. Do **not** unpin in Launch.
2. Finish may enqueue `hipStreamWaitEvent` + y-copy, then **must host-wait** `ev_e` (or `hipStreamSynchronize` on the expert stream that recorded it) **before** unpin. `hipStreamWaitEvent` on the compute stream is not a host-side retirement proof.
3. Unpin under lock via `UnpinLocked` only after that host-wait succeeds. Then clear `cache_pin = -1`.
4. Same-dev enqueue-only path: same rule. Record a done-event on the GEMM stream; host-wait it before unpin. "Enqueue then unpin" is forbidden even when compute_dev == expert_dev, unless a written concurrency argument proves no other worker can `GetLocked`/`evict` that slot — default is: **no such argument**, so host-wait.
5. Every Launch/Finish error path after a successful `GetLocked` must retire GPU work **before** unpin:
   - If `ev_e` was successfully recorded: host-wait `ev_e` (or `hipStreamSynchronize` on the stream that recorded it), then unpin.
   - If pin is live and `ev_e` was **not** recorded (fail after pin, after dequant/GEMM enqueue, or `hipEventRecord(ev_e)` itself fails): **`hipStreamSynchronize` the expert stream** (same-dev: the compute/GEMM stream) **then** unpin. "Wait any recorded `ev_e` if valid" is not a retirement proof when the event does not exist.
   - Then restore compute device. No live expert event left. Pin count → 0 except failed-retirement quarantine (pin stays until later observed retire or fatal teardown).
6. Mutex scope still mirrors donor for Get/Unpin (`GetLocked` under lock, GEMM outside, `UnpinLocked` under lock). Do not hold the mutex across `MatmulBT`.
7. **`Ensure` must not reconfigure while any pin is live.** If `dev/I/H/nslots` already match, return true. If they differ and any `slots[i].pins > 0`, return false (reject; do not `FreeAll`). Only call `FreeAll` when pin-count is 0 across the cache. Do not copy donor `Ensure`→`FreeAll` on a live pin. Slot-count comes from `PrefillDequantCacheSlots()` (donor default 1).

### Lifetime invariants (mutation-proven)

- every successful Launch has exactly one Finish;
- Launch fail unpins (or never pinned) after current-stream retire; leftover prior ev_e is never the rollback target; failed retire quarantines;
- Finish restores compute device before return;
- `ev_c` is recorded on compute stream, `ev_e` on expert stream; no cross-device event record;
- cache pin count returns to 0 on success and on every error return where retirement was observed; failed retirement is the sole zero-pin exception (quarantine);
- pin remains >0 from GetLocked until host-observed retirement (`ev_e` complete, or expert/compute stream sync on the pre-record error path);
- a second worker cannot obtain a rewrite/evict of a still-pinned slot;
- `Ensure` never `FreeAll`s a cache that has any `pins > 0`.

## Risks

- Host-wait on every Finish adds latency vs donor enqueue-only unpin. Accept for correctness. Pipe-on overlap is a later spec.
- Without #837, Finish-side `MatmulBT` can still destroy the other GPU's hipBLAS handle.
- Slot 0 only while pipe is off. Do not "simplify" by keeping monolithic TLS.

## Tests

Host-only:

1. Source invariant: product wrapper contains `LaunchGemma4Fp8ExpertGeGLUPrefillPeer` and `FinishGemma4Fp8ExpertGeGLUPrefillPeer`; does not keep a peer `static thread_local Tls tls` as the only storage.
2. Source invariant: `PeerSlot` stores `cache_pin`; Launch does not call `UnpinLocked` before `ev_e` record; Finish host-waits before unpin; `Ensure` returns false when any pin is live instead of `FreeAll`.
3. Pairing: every Launch path that returns true has a Finish; fail-Launch must unpin.
4. `kPeerPipe` default is off (env unset → no overlapping slot 1).

RED: restore monolithic peer TLS → invariant 1 fails. RED: restore donor unpin-before-`ev_e` → invariant 2 fails.

Host or fake-cache mutations (required):

- concurrent eviction: worker B `GetLocked`/rewrite of a zero-pin slot while worker A's expert stream still "reads" (fake outstanding GEMM). Must RED if unpin happened before retirement.
- fail after pin, **after work enqueue** (dequant or GEMM queued), before event record → expert-stream (same-dev: compute-stream) sync then unpin; pin count 0; no live event. A mutation that only fails immediately after pin (no enqueue) does **not** satisfy this case.
- fail at `hipEventRecord(ev_e)` after work enqueue → same stream-sync fallback before unpin.
- fail after event record, before wait → pin count 0 after cleanup, event retired or destroyed safely.
- fail after wait, before copy → pin count 0.
- fail after copy, before unpin → cleanup still unpins (idempotent).
- concurrent `Ensure` with a new I/H or slot-count while any pin is live → `Ensure` returns false, no `FreeAll`, pinned buffers remain. RED if donor `Ensure`→`FreeAll` is restored.

GPU (coord, after #837):

- T=2029, `VT_GEMMA4_PREFILL_TRACE=1`, `VT_GEMMA4_ATTN_STAGE_SYNC` **unset**, `MAX_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=65536`, `NUM_BLOCKS=2048`, `PEER_ACT=1`: matched helper BEGIN/END **and** HTTP body.
- T=1 decode + Paris/arith unchanged.

## Gates

- Host invariants GREEN without GPU.
- No FP8 Lt / GU_INTERLEAVE / pipe-default-on in the impl.
- Operator A/B (`5071`): this is **C**. After B, smoke **T=2029** independently (console+wchan on failure). p42k only after the smallest passing set of A/B/C.
- Default behavior outside Gemma-4 FP8 xdev prefill-batch peer-act is unchanged.
- `#697` files untouched.

## Stop conditions

- Research requires pipe-on default or FP8 Lt — new spec.
- Lab GPU / p42k without coord.
- Impl that edits `rocm_paged_attn.hip`.
- Impl that copies donor unpin-before-`ev_e`.
- Landing this row on a shared branch with #837/#838.

## Evidence

Bus: `0384`, `713f`, `9772`, `25c9`, `5071`, `64cb`, `4954`. Donor bytes hashed in [`rocm-gemma4-prefill-peer-helper-donor.md`](rocm-gemma4-prefill-peer-helper-donor.md) (cache core 60–209 required).

Now: `IMPLEMENTING` — ea9c product RestoreComputeOrThrow no-op compile-and-run RED. `a007ec40` is not a review target.

## Owed

Owned by this row (`ROCM-GEMMA4-PREFILL-PEER-HELPER`) and tracked by
[#839](https://github.com/mudler/vllm.cpp/issues/839), which stays open until each
is discharged. Named here rather than left to be discovered, per AGENTS.md
"Nothing lands dead".

- **The host simulator is a second implementation of Launch/Finish.**
  `include/vt/rocm/rocm_gemma4_prefill_dequant_cache.h` carries `HostAlloc`,
  `PrefillDequantCacheHost`, `HostRetireThenUnpin`, `HostLaunch` and `HostFinish`
  (~250 of its lines), compiled into every HIP build. `HostLaunch`/`HostFinish`
  are hand-written analogues of the product `LaunchGemma4Fp8ExpertGeGLUPrefillPeer`
  and `FinishGemma4Fp8ExpertGeGLUPrefillPeer` in
  `src/vt/rocm/rocm_gemma4_experts.hip`, so the lifetime cases in
  `tests/vt/test_ops_gemma4_prefill_peer.cpp` exercise the analogue and not the
  product. What binds the two today is the source-slice mutation gates in the
  same file (`ProductFinishRetiresCstBeforeReuse`, `CompileFinishCatchTu`,
  `CompileAndRunProductRestore`), which assert against the product TEXT. Owed:
  move the `Host*` half under `tests/`, keeping `PrefillPeerLife`,
  `ChoosePrefillRetire`, `PrefillDequantCacheT`, `ComputeDevGuard`,
  `RestoreFailed`, `SameDevLife`, `OutputCopyGate`, `SlotReusable` and
  `PublishThenRestoreOrThrow` in the header because the `.hip` reaches them.
  Deliberately not done in this change: only a ROCm box compiles the one
  translation unit that consumes the header, so the move cannot be verified
  where this repair was made.

- **The blocking retirement has no measurement — protocol frozen to Researcher
  ca41. Do NOT run the historical async-unpin product arm.** That arm is the
  lifetime bug; using it as a “before” product binary can unpin storage while
  queued GEMMs still consume it. Safe attribution only:
  1. Env-gated measurement-only counters around each product
     `hipEventSynchronize(ev_e)` and `hipStreamSynchronize(cst)`: call count,
     host-block microseconds by kind, and pre-sync
     `hipEventQuery`/`hipStreamQuery` ready-vs-not-ready. No change to
     ordering/lifetime.
  2. Frozen prefill smoke T=2029 on 2×R9700: uninstrumented current-head
     throughput plus instrumented attribution as separate legs. Same
     binary/head/recipe; ≥3 warmups + ≥5 steady samples. Report prefill
     tok/s/TTFT, total request time, wait calls, cumulative wait-host-us,
     us/call, ready fraction, wait-time/request-time ratio. Instrumented
     throughput is diagnostic, not canonical.
  3. If a literal before/after remains mandatory: a **separate synthetic HIP
     microbench** whose buffers stay live until one final stream sync. A waits
     per expert like product; B enqueues the same sequence then waits once.
     Same op count/streams/bytes. Label as overlap upper bound, not product
     throughput. Never disable retirement/unpin in the model path. Never infer
     `before = elapsed - wait_us` as measured throughput.
  4. Isolated `:8012`; never `:8010`; idle window; teardown. Record raw logs +
     HEAD/binary/recipe in `docs/BENCHMARKS.md` + `.agents/benchmark-record.md`.
     Expensive → next-row; the correctness wait stays.

- **`PeerSlot s[1]` SOURCE FIXED (Researcher ca41).** No slot-1 overlap wiring.
  Launch/Finish fail-closed on `slot != 0`. Wrapper still hardcodes `/*slot=*/0`.
  GPU not required for this item.
