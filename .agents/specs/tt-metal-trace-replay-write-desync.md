# Interleaved host→device writes between trace replays desync `expected_num_workers_completed` → completion hang after ~38–50 replays

## Summary

On a single Blackhole P150 (1×1 mesh), replaying a captured mesh trace that
captures ttnn ops (embedding + sdpa_decode + paged_update_cache, ~50 MB trace)
hangs deterministically after ~38–50 consecutive replays of one trace id. The
hang is a futex wait in the post-replay blocking readback: `execute_trace`
returns (the trace is enqueued) but the device-side completion never arrives.

**Root cause (isolated by bisection + code read):** issuing
`ttnn::copy_to_device` (→ `enqueue_write_tensor` → mesh CQ `write_to_core`)
into a buffer the captured trace *reads*, between replays while the trace is
live, perturbs `expected_num_workers_completed` — the same dispatch counter the
trace replay path (`update_worker_state_post_trace_execution`) assigns and
`mark_completely_full()`s per replay. After enough interleaved writes the
accounting desyncs and the next replay's `DISPATCH_WAIT(count)` never
completes. This is the between-replays analogue of the within-trace bug fixed
in #7978 / discussed in #7793.

## Environment

- tt-metal: `a3d330289752192754277638fe5c09eb2fb49763` (2026-08-09)
- Device: single Blackhole P150 (1×1 mesh, `trace_region_size=50 MB`,
  `l1_small_size=DEFAULT_L1_SMALL_SIZE`, 1 CQ)
- Firmware bundle: 19.7.1
- The captured trace is a Qwen3-0.6B decode step (28 transformer layers,
  bf16). The trace captures ttnn ops: `EmbedDeviceIdsInto`, `sdpa_decode`,
  `paged_update_cache`, RmsNorm, RoPE, etc.

## The interleaving pattern (our decode driver)

Per replay step, between `replay_mesh_trace` calls, the driver issues ~9
host→device copies into persistent (pre-allocated, stable-address) buffers:

- 4× rope cos/sin refresh (bf16)
- 2× RAC idx + page_table (INT32 ROW_MAJOR) — read by `paged_update_cache`
- 2× PA meta: page_table + cur_pos (INT32 ROW_MAJOR) — read by `sdpa_decode`
- 1× decode ids (UINT32) — read by the captured embedding

Each copy is `ttnn::copy_to_device(host_tensor, persistent_device_tensor)`
which (per `ttnn/core/tensor/tensor_ops.cpp:182`) calls
`enqueue_write_tensor` → mesh CQ write path → `device_dispatch::write_to_core`
(`fd_mesh_command_queue.cpp:579`).

## Bisection (the toxic class is idx/PA-meta INT32 copies)

Qwen3-0.6B, "Hello", `--max-tokens 80`, `VT_TT_HOST_FREE_DECODE=1`.

We have env-gated skip flags that suppress specific copy classes after the
first capture (stale device content — mechanics test only, numerically wrong):

| skip flag(s) ON | copies still interleaved | result | replays before hang |
|---|---|---|---|
| (none, `VT_TT_RECAPTURE_EVERY=32`) | rope+idx+PAmeta+ids | HANG | 75 (32+32+11, gen3) |
| `VT_TT_NO_ROPE_REFRESH` | idx+PAmeta+ids | HANG | 39 |
| `VT_TT_NO_IDX_WARM` | rope+ids | **PASS** | 79 (RC=0, completed) |
| `VT_TT_NO_IDS_WARM` | rope+idx+PAmeta | HANG | 50 |
| all three (rope+idx+ids) | (none) | **PASS** | 79 (RC=0, completed) |

**Conclusion:** the toxic copy class is the idx/PA-meta refresh
(`WarmRacIdx` + `WarmPaMeta`): `copy_to_device` into the INT32
`page_table`/`update_idxs`/`cur_pos` buffers the captured
`sdpa_decode`/`paged_update_cache` reads. With idx/PA-meta OFF (rope+ids
ON), 79 replays complete cleanly; with idx/PA-meta ON it hangs regardless of
the others. The wall-count variance (39/50/75) is run-to-run timing noise,
not evidence that rope/ids contribute (the gate is binary on idx/PA-meta).

## Upstream mesh-trace machinery is NOT the cause

I ported the upstream `tests/tt_metal/distributed/test_mesh_trace.cpp`
`Sanity` test plus four scratch variants onto this checkout on the same card.
ALL PASS:

- `MeshTraceTestSuite.Sanity`: 10 capture gens × 4 traces × 40 replays +
  release — PASS (21 s).
- `ScratchConsecutiveReplayOneTrace`: 1 program, 120 consecutive replays of
  ONE trace id, no interleaved traffic — PASS.
- `ScratchConsecutiveReplayFinishEach`: same + `Finish()` after every replay
  (mimics our per-step sync) — PASS.
- `ScratchRecaptureGenerations`: 10 gens × 16 replays + release — PASS.
- `ScratchBigTraceReplay`: 50-program trace (~50 MB), 120 replays — PASS (47 s).

The raw `replay_mesh_trace` / `release_trace` / `MeshTraceBuffer` lifetime is
sound. The wall only appears under our interleaving of `copy_to_device` into
trace-input buffers between replays.

## Mechanism (code read)

1. `copy_to_device` (`ttnn/core/tensor/tensor_ops.cpp:182`) →
   `enqueue_write_tensor` → mesh CQ `write_to_core`
   (`tt_metal/distributed/fd_mesh_command_queue.cpp:579`).
2. `write_to_core` and the interleaved-buffer write dispatch params both take
   `expected_num_workers_completed`
   (`tt_metal/impl/buffers/dispatch.cpp:486, 509, 520, 737, 985, 1304, 1350`).
3. The trace replay path `update_worker_state_post_trace_execution`
   (`tt_metal/impl/trace/dispatch.cpp:31-72`) assigns
   `expected_num_workers_completed` from the trace descriptor and calls
   `config_buffer_mgr.mark_completely_full(expected_num_workers_completed[i])`
   per replay (`dispatch.cpp:223`).
4. The trace's embedded `DISPATCH_WAIT(count=expected_workers)` cmds assume
   the count starts from a known state at each replay (the reset mechanism
   described in #7793). Interleaved `write_to_core` between replays perturbs
   this counter; after ~38–50 such perturbations the next replay's
   `DISPATCH_WAIT` never completes.

Writes during *capture* are correctly forbidden
(`TT_FATAL(!trace_id_.has_value(), "Writes are not supported during trace
capture.")` at `fd_mesh_command_queue.cpp:572`), but writes between *replays*
(trace live, not capturing) are allowed and are the trigger.

## Related

- #7793 — the within-trace version of this bug (the `DISPATCH_WAIT(count)` /
  `expected_num_workers_completed` / `num_completion_worker_cores` family).
- #7978 — fix for the within-trace case.
- #19248 — `expected_num_workers_completed` counter-wrap hangs in MeshCQ.
- #17696 — non-deterministic MeshDevice trace replay hangs (blocking
  behavior).

## Ask

1. Is there a sanctioned way to issue host→device writes into trace-input
   buffers between replays without perturbing
   `expected_num_workers_completed`? (e.g. a write path that bypasses the
   dispatch worker-completion accounting, or an explicit counter-reset /
   `Finish` protocol between replays.)
2. If not, should the between-replays case get the same treatment #7978 gave
   the within-trace case — i.e. a count-reset protocol so interleaved writes
   don't desync the next replay's `DISPATCH_WAIT`?

## Workaround (in our driver)

`VT_TT_RECAPTURE_EVERY=N`: destroy + re-capture the graph every N replays.
N=32 gives ~2 healthy capture generations (~75 replays) before the same wall
reappears on the re-captured trace — so the leak also accumulates across
capture cycles, not just within one. Not a viable production fix.

The long-term direction for our driver is to move per-step state advancement
(cur_pos, page_table, update_idx) inside the captured graph (the upstream
`executor.py` traced-decode pattern: 0–1 copies/step), eliminating the
interleaved writes entirely. But the between-replays write-into-trace-input
case seems like a genuine dispatch-accounting gap worth flagging.

## Repro

**Honest status: the trigger does NOT reproduce in any minimal standalone test
I tried.** I can only reproduce it in the full vllm.cpp decode driver.

Minimal repros that all PASS (no hang):
- C++ scratch test in `test_mesh_trace.cpp` (`ScratchInterleavedWriteBetweenReplays`):
  capture an eltwise-binary program, replay 120× with `EnqueueWriteMeshBuffer`
  into its DRAM src buffer between replays — PASS.
- Python script (`repro_trace_replay_write_hang.py`): capture a single
  `ttnn.transformer.paged_scaled_dot_product_attention_decode` (the exact op
  our driver uses) with persistent device page_table/cur_pos, replay 120× with
  `ttnn.copy` into those INT32 buffers between replays (the exact toxic class)
  — PASS. With `blocking=False` + readback sync (matching the driver) — PASS.
- Same script with NUM_OPS=8 and NUM_OPS=28 (28 chained decode calls in one
  trace, matching our layer count) — PASS.
- Adding `ttnn.experimental.paged_update_cache` (the RAC op, with the exact
  height-sharded L1 input `[1,1,32,128]` our driver uses) to the captured
  trace, and refreshing page_table + cur_pos + update_idxs (all three toxic
  buffers) between replays — PASS (both 1× and 28×).

So the trigger is not the op, not the op count, not the RAC op, not the
sharded input, not the ttnn wrapper, not the INT32 write class in isolation,
not blocking=False. It only manifests in the full driver trace, which
differs from the repros in: (1) a diverse op mix (embedding + 28×
[RmsNorm+RoPE+sdpa_decode+paged_update_cache+...] with differing
`num_completion_worker_cores` per op, vs identical chained RAC+sdpa pairs);
(2) our persistent-tensor pool + `from_vector`/`copy_to_device` host-staging
path (vs the repro's `as_tensor` fresh device tensor + `ttnn.copy`).

Filing now with the honest non-minimal framing, in case the between-replays
write case is a known limitation with a sanctioned workaround, and to ask
whether `ttnn::copy_to_device`-into-trace-input-between-replays is expected
to be safe. I can share the full driver reproducer if useful; porting it to
a self-contained tt-metal test would require reconstructing the decode-layer
op mix.
