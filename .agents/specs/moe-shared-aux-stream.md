# ENG-MOE-SHARED-AUX — MoE shared-expert aux-stream decode overlap

Mirror vLLM's decode overlap: run the MoE **shared-expert MLP** on a second CUDA
stream concurrent with the **routed-expert** router/align/grouped-GEMMs on the
main stream, join before the combine. Token-exact by construction (overlap
changes WHEN the independent shared path runs, never WHAT it computes). This is
the largest remaining 35B c1/c2 engine lever.

## Scope

- **In:** the fork/join around the shared-expert MLP inside the committed 35B
  Marlin MoE decode path `MoeBlockFusedMarlinCuda`
  (`src/vllm/model_executor/models/qwen3_5.cpp`). Gated to decode
  (`T <= threshold`) and CUDA, opt-in `VT_MOE_SHARED_AUX_STREAM`, tunable
  `VT_MOE_SHARED_AUX_THRESHOLD` (default 128; GB10 48-SM calibration).
- **Row:** `ENG-MOE-SHARED-AUX` (engine matrix).
- **Out:** the wmma fallback (`MoeBlockFusedCuda`), CPU/GGUF reference path, and
  the 27B dense path (no MoE shared/routed split). vLLM's DP/EP MK-internal
  overlap (`MK_INTERNAL_OVERLAPPED`) and DBO ubatch overlap are separate rows.

## Upstream chain

- `vllm/model_executor/layers/fused_moe/runner/shared_experts.py:99-104`
  (`should_run_shared_in_aux_stream`: `is_cuda() and stream is not None and
  hidden_states.shape[0] <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`),
  `:125-142` (`shared_experts_input.record_stream`, `stream.wait_stream(current)`
  fork; `_run_in_aux_stream` + `current_stream().wait_stream(stream)` join).
- `vllm/utils/multi_stream_utils.py:20-58` `maybe_execute_in_parallel`:
  `event0.record()` on main → `fn0()` on main; on aux: `event0.wait()` → `fn1()`
  → `event1.record()`; then `event1.wait()` on main. A TRT-LLM port
  (`tensorrt_llm/_torch/modules/multi_stream_utils.py`).
- `vllm/utils/torch_utils.py:736-756` `aux_stream()` — ONE global aux stream per
  device, lazily created (not one per layer).
- `vllm/envs.py:260,1837` `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD = 256`
  default.

vLLM's aux tensors are allocated by the **stream-aware** torch caching allocator
(`record_stream`); our scratch `DevicePool` is single-stream only (see Risks).

## Our baseline

- `MoeBlockFusedMarlinCuda` (`qwen3_5.cpp`) — router (`Matmul` +
  `MoeRouterTopK`), `MarlinMoeAlignBlockSize`, grouped GEMMs
  (`MoeGroupedGemmNvfp4Marlin`) → `expert_out`, then the shared expert +
  `MoeCombineGate`/`MoeCombine`. All serial on `d.q`, captured in the decode
  CUDA-graph (`Qwen3_5DecodeGraph`).
- `SharedExpertUngated` (`qwen3_5.cpp`) — the shared-expert MLP producing
  `{sd [T,H] f32, gl [T,1] f32}`; already capture-safe (no host sync / D2H).
- Cross-stream primitives already present: `Backend::CreateQueue`/`CreateEvent`/
  `RecordEvent`/`QueueWaitEvent` (`src/vt/cuda/cuda_backend.cu:77,112-138`),
  used today by the async-output copy stream (ENG-ASYNC-SCHED W3).
- Gap: no second compute stream in the forward; the scratch `DevicePool` assumes
  single-stream ordering.

## Port map

| Upstream | Local |
|---|---|
| `maybe_execute_in_parallel` fork/join | inline fork (before router) + join (before combine) in `MoeBlockFusedMarlinCuda` |
| `aux_stream()` single global stream | `MoeAuxStreamFor(Dev)` — persistent per-device `{Queue, fork Event, done Event}`, lazily created, leaked at exit |
| `hidden.shape[0] <= THRESHOLD` gate | `MoeSharedAuxStreamEnabled() && T <= MoeSharedAuxThreshold()` |
| torch stream-aware allocator (`record_stream`) | `AuxPool()` — a SECOND `DevicePool` for the aux stream, so each pool serves one stream (deviation, see Risks) |

Deviation: we keep the simple non-stream-aware `DevicePool` and instead give the
aux stream its own pool (`AuxPool`), routed via a thread-local `ActivePool()` set
by `ActivePoolScope` during the aux issue; `DBuf` remembers its owning pool so a
buffer allocated in the aux region and freed after the join returns to the right
pool. Recorded here per the discipline rule.

## Tests to port

vLLM has no dedicated unit test for the overlap (it is a runtime-scheduling
transform asserted by end-to-end token-exactness). Our proof mirrors that:

- **Byte-identical:** `test_qwen36_paged_engine` (35B, 315/315) and
  `test_qwen27_paged_engine` (27B, 235/235) run with
  `VT_MOE_SHARED_AUX_STREAM=1` and `=0` (same binary) must be **token
  bit-identical** — the overlap reorders nothing float, so identical tokens ⟺
  identical MoE-block output. (27B is dense, unaffected; included as a
  no-regression guard.)
- **Memory-safety:** `compute-sanitizer memcheck` over the captured decode graph
  with overlap ON, 0 errors (proves the aux-pool isolation removes the
  cross-stream scratch race).
- **Graph capture:** captured (graph-ON) vs eager (`VLLM_CPP_CUDAGRAPH=0`)
  token-exact with overlap ON — the aux fork/join is captured via the event
  edges into the same `cudaStreamBeginCapture(ThreadLocal)` region.

## Gates

1. **CPU:** clean `-Werror` build (overlap code is `#ifdef VT_MARLIN_NVFP4`, but
   the pool routing / `ActivePool` compile unconditionally); full ctest; tools
   unittest 164; record/doc checkers.
2. **Correctness (DGX):** memcheck 0 errors (captured, overlap ON); overlap
   ON==OFF byte-identical (35B 315/315 + 27B 235/235, both env values); captured
   vs eager token-exact.
3. **Perf (DGX, one flock, interleaved same-binary A/B, first drop):** c1/c2/c4
   35B TPOT expect win; c8/c16/c32 expect NEUTRAL (gate closed above threshold).
   Pick threshold (128 vs 64) maximizing the c1-c4 win with zero c8+ regression.

Commands:
- `cmake -B build -DVLLM_CPP_CUTLASS_DIR=… -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DVLLM_CPP_TRITON=ON && cmake --build build -j`
- `VT_MOE_SHARED_AUX_STREAM=1 ./build/tests/test_qwen36_paged_engine` (and `=0`), diff tokens.
- `compute-sanitizer --tool memcheck VT_MOE_SHARED_AUX_STREAM=1 ./build/tests/test_qwen36_paged_engine`.
- in-situ TPOT A/B under one `flock /tmp/gpu`, interleaved ON/OFF, drop first leg.

## Dependencies

- Row block: sits on the committed Marlin MoE decode path (`ENG-MOE-HOSTFREE`,
  `KERNEL-MOE-ROUTING`, the decode CUDA-graph). Toolchain: CUDA 13.0 nvcc,
  CUTLASS (flashinfer), Triton AOT, FA2 sm_121a. Hardware: GB10 (dgx.casa).
  Models: 35B-A3B NVFP4 + 27B NVFP4.

## Work breakdown

1. Aux-pool isolation (`AuxPool`/`ActivePool`/`ActivePoolScope`, `DBuf::pool_`).
2. Persistent aux stream + events (`MoeAuxStream`/`MoeAuxStreamFor`) + predicates
   (`MoeSharedAuxStreamEnabled`/`MoeSharedAuxThreshold`).
3. Fork/join in `MoeBlockFusedMarlinCuda` (both glue-fused and unfused combine).
4. DGX gates + threshold A/B + flip decision.

## Risks / decisions

- **Cross-stream scratch race (the real hazard):** the shared `DevicePool` reuse
  is only safe under single-stream ordering; two streams sharing it would race a
  freed-then-reused block (memcheck-visible). Resolved by the per-stream
  `AuxPool` (deviation above), NOT by making the pool stream-aware — smaller,
  local, and keeps the default path byte-for-byte unchanged.
- **Graph capture of the aux fork:** standard multi-stream capture — the aux
  stream joins the `ThreadLocal` capture via the fork event edge and merges at
  the join event. If `ThreadLocal` rejects the aux fork, fall back to `Relaxed`
  in `BeginCapture` (`cuda_backend.cu`). Verified on first DGX build (recorded in
  the ledger/state).
- Threshold is a calibration, not a behavior choice; vLLM's 256 is for
  large-SM GPUs, GB10 has 48 SMs so overlap only pays at very low concurrency.
- No product calls; vLLM-defined behavior (mirror the overlap) is not reopened.
