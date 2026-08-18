# Spike: on-device decode state advancement (host-free R2)

**Status:** SPIKE (pre-implementation spec) — POL-SPIKE-FIRST.
**Branch:** `row/BACKEND-TENSTORRENT-HOST-FREE-R2` (off `row/BACKEND-TENSTORRENT-HOST-FREE-R1`).
**Owning spec:** `.agents/specs/tenstorrent-host-free-r1.md` (this extends the
host-free decode work; R1 is the captured-replay baseline).

## Problem

R1's captured decode graph is token-exact and 5.8× faster than eager, but
hangs deterministically after ~38–50 consecutive replays of one trace id
(monitor-68). Bisection (monitors 68–72) isolated the cause to the per-replay
**eager host→device copies** into trace-input INT32 buffers:

- `WarmRacIdx`: `copy_to_device` into `update_idxs` + `page_table` (INT32)
- `WarmPaMeta`: `copy_to_device` into `page_table` + `cur_pos` (INT32)

With these OFF (`VT_TT_NO_IDX_WARM`), 79 replays complete cleanly; with them
ON, hang at ~38–50. Root mechanism (code-read): `copy_to_device` → mesh CQ
`write_to_core` perturbs `expected_num_workers_completed`, the same dispatch
counter the trace replay path assigns/`mark_completely_full`s per replay.

**A minimal standalone repro was NOT found** (monitors 73–90): the toxic
write pattern + the exact RAC+sdpa ops, at 1× and 28×, with `blocking=False`
and sharded L1 inputs, all PASS 120 replays. The trigger is specific to the
full diverse-layer driver trace. So the fix must eliminate the interleaved
writes entirely rather than work around a cleanly-reproducible upstream bug.

## Goal

Eliminate the per-replay eager host→device copies of decode state
(`cur_pos`, `update_idxs`, rope cos/sin, decode ids) by advancing that state
**on-device inside the captured trace**, mirroring the upstream
`executor.py` traced-decode pattern (`models/common/models/executor.py:2186`).
After this change, a steady-state replay step should perform **0 eager
host→device copies** (except `page_table`, only when a block boundary is
crossed — rare, not every step).

## Upstream pattern (the reference)

`executor.py:2186` comment: *"the captured trace advanced current_pos/
rot_mat_idxs (in-place plus_one) and wrote the sampled token back into the
persistent token buffer on the previous replay, so tokens/positions need NO
host staging. Refresh only the page table, and only when it actually changed
(block boundaries crossed)."`

Concrete (`executor.py:2310-2351`):
1. Persistent device tensors: `tt_tokens`, `tt_current_pos`, `tt_rot_mat_idxs`,
   `tt_page_table` (allocated once before capture).
2. Captured body: `embed(tokens)` → `decode_forward(..., current_pos, rot_mats,
   page_table)` → (on-device sampling writes token back to `tt_tokens`) →
   `ttnn.plus_one(tt_current_pos)` → `ttnn.plus_one(tt_rot_mat_idxs)`.
3. Steady-state replay: `copy_host_to_device_tensor(page_table, tt_page_table)`
   only if `page_table` changed since last step; `execute_trace(blocking=False)`;
   read back logits/tokens.

`ttnn::operations::experimental::plus_one` exists in our checkout
(`ttnn/cpp/ttnn/operations/experimental/plusone/plusone.hpp:14`) — in-place,
traceable, with `skip_negative_entries`.

## Our driver's gap

Our driver samples on host (downloads logits → CPU argmax → next token), so
the `tt_out_tok` in-place-writeback (which needs on-device sampling) does not
directly apply. But the **position/idx advancement** — the actual toxic class
— is independent of sampling and IS movable on-device:

- `cur_pos = seq_lens - 1`, advances +1 per step → `plus_one` on a persistent
  `cur_pos` tensor captured in the trace.
- `update_idxs = seq_lens - 1` (same value as cur_pos for decode T=1) → derive
  from `cur_pos` on-device (alias or `plus_one` a parallel persistent tensor).
- rope cos/sin indexed by `positions` (= cur_pos) → upstream uses
  `ttnn.plus_one(tt_rot_mat_idxs)`; our `WarmRopeCosSin` recomputes from
  `positions` each step. Movable: capture a `plus_one` on the rope index, or
  capture the rope lookup itself.
- `decode ids` (the sampled token for embedding) — **proven innocent by
  bisection** (`VT_TT_NO_IDS_WARM` alone still hangs; ids OFF + idx ON hangs).
  Still, eliminating it closes the loop: the host-sampled token must reach the
  embedding input. Since we sample on host, this one host→device copy per step
  is unavoidable UNLESS we move sampling on-device. **Defer** — ids is not the
  toxic class, keep the host copy for now (1 copy/step is fine, matching the
  upstream "1 copy when page_table changes" budget).
- `page_table` → refresh from host only when a block boundary is crossed
  (block-table grows). Rare; not every step.

## Design

### Phase 1 (the toxic class): on-device `cur_pos` + `update_idxs`

1. **Persistent `cur_pos` tensor** (INT32, `[B]`, device), allocated once before
   capture. Seeded from host at capture time (first decode step's `seq_lens-1`).
2. **Capture `ttnn::plus_one(cur_pos, skip_negative_entries=true)` at the END of
   the trace body** (after all reads of `cur_pos` in sdpa_decode/RAC), so the
   NEXT replay sees the incremented value. Warm the `plus_one` program before
   capture (program-cache).
3. **`update_idxs`**: for decode T=1, `update_idxs == cur_pos`. Two options:
   (a) alias — pass the same persistent tensor as `update_idxs_tensor` to
   `paged_update_cache` (if the op accepts it without copy); (b) a second
   persistent tensor `+1`'d in parallel. Prefer (a); fall back to (b).
4. **`WarmPaMeta`/`WarmRacIdx` become no-ops in steady state** — they only fire
   on re-seed (first step after prefill, or after a re-capture) to set the
   initial `cur_pos` from host. Gate them on `!replay_regime` (i.e. only on the
   capture/warm step), not every step.

### Phase 2 (page_table): refresh only on change

5. Track `prev_page_table` (host-side). On each step, `if (page_table !=
   prev_page_table) copy_host_to_device(page_table, tt_page_table)`. Block
   boundaries are rare (every `block_size` tokens), so this is ~0 copies/step
   in steady state.

### Phase 3 (rope, deferred)

6. `WarmRopeCosSin` is **not** the toxic class (bisection: rope OFF still hangs
   at 39; idx OFF passes). Defer moving rope on-device to a later iteration
   unless Phase 1+2 don't clear the wall (they should — idx/PAmeta is the sole
   trigger).

### Phase 4 (decode ids, deferred)

7. `WarmDecodeIds` is innocent (ids OFF + idx ON still hangs). Keep the host
   copy for the sampled token (1 copy/step). Moving sampling on-device is a
   larger change and not needed to clear the wall.

## Done-when

- A steady-state replay step performs 0 eager host→device copies of
  `cur_pos`/`update_idxs`/rope/ids (only `page_table` on block-boundary steps).
- `VT_TT_HOST_FREE_DECODE=1` decode of Qwen3-0.6B "Hello" `--max-tokens 80`
  completes 79 replays with **no hang** (currently hangs at ~38–50 with idx
  warm ON, passes only with `VT_TT_NO_IDX_WARM`).
- Token-exact fidelity preserved: 22/22 argmax vs host-free eager (re-run the
  m57-vs-m58 diff protocol).
- No new eager device allocations during replay (the existing
  `InvalidateHostCachesAfterTrace` + pool-sized shadow invariants hold).

## Evidence / gates

- Focused gate: the 80-token no-hang run above + the fidelity diff.
- Full gate: `scripts/agent-preflight.sh` + the TT golden correctness gate
  (unblocks once >75 replays work — currently blocked by the wall).
- Fresh review + operator rerun per POL-REVIEW-FRESH / POL-OPERATOR-VERIFY.

## Risks

- `plus_one` may not be capture-safe in the way we need (in-place write to a
  buffer the trace also reads earlier). Upstream's comment ("These writes are
  AFTER every read of the buffers in this body, so there is no intra-trace
  hazard") suggests it's fine, but verify.
- `paged_update_cache` may not accept an aliased `update_idxs_tensor` (it may
  copy internally). If so, Phase 1(b) needs a second persistent tensor + a
  second `plus_one`.
- Moving `cur_pos` on-device changes the R1 captured-graph semantics; the
  fidelity diff must re-confirm token-exactness.
- This is a larger change than R1's incremental fixes; spec-first, red-test-
  before-fix, focused-then-full gate per policy.

## Stop conditions

- If Phase 1 (on-device cur_pos/update_idxs) does not clear the wall (i.e. the
  hang persists with idx/PAmeta copies eliminated), the trigger is NOT the
  copies but something else in the full driver trace — re-bisect with the new
  minimal copy budget, and reconsider the upstream issue.
- If `plus_one` cannot be captured safely, fall back to a host-side
  `copy_to_device` of `cur_pos` only (1 copy/step) and re-test — if THAT
  clears the wall, the trigger was the multi-copy churn, not the write per se.

## Now

`ACTIVE` as the implementation wave of `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`.
On-device `cur_pos` plus_one is captured in the decode graph. A P150 run of
Qwen3-0.6B "Hello" at 80 tokens completed 79 replays with no hang. Next:
fresh review, then the operator reruns that gate and the TT golden on card.

## First resume command

```sh
cd /home/lu_zero/Sources/vllmcpp-tenstorrent
git status   # confirm clean base on row/BACKEND-TENSTORRENT-HOST-FREE-R1
# Read this spec, then implement Phase 1: persistent cur_pos + plus_one in trace.
```

## Probe results (2026-08-16, monitor-92)

**`ttnn.plus_one` is capture-safe and advances on-device on replay.**
Probe (`scripts/probe_plus_one_capture.py`): persistent INT32 `cur_pos=[5]`,
warm `plus_one` → `[6]`, capture a trace whose body is just `plus_one(cur_pos)`,
replay 5× → `[7,8,9,10,11]`. Clean increment, no capture fatal, no hang.

- Python binding: `ttnn.plus_one` (top-level, NOT `ttnn.experimental.plus_one`).
- C++ API: `ttnn::operations::experimental::plus_one(tensor, sub_core_grids,
  skip_negative_entries)` in
  `ttnn/cpp/ttnn/operations/experimental/plusone/plusone.hpp:14`.
- In-place, traceable, with `skip_negative_entries` (matches upstream
  `plus_one(current_pos, skip_negative_entries=True)`).

⇒ Phase 1 risk #1 (capture-safety) is cleared. Phase 1 risk #2 (aliased
`update_idxs_tensor` in `paged_update_cache`) still to probe during impl.
