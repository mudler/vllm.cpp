# Block-table host-cluster cleanup — mechanical 1:1 mirrors

**Kind:** CPU-first mechanical port cleanup (bit-identical host-side plumbing
mirrors). Executes the FIX dispatch from the completed lost-lanes rescan
([rescan-lost-lanes-2026-07-16.md](rescan-lost-lanes-2026-07-16.md) §1 item c,
§5 item d, §6 item e). No engine algorithm change; every mirror is proven
bit-identical (the same values reach the device).

**Matrix row:** item d advances `ENG-CUDAGRAPH` (decode-graph capture/replay);
this spec is that row's spike link while the cleanup is `ACTIVE` under
`CLAIM-BLOCKTABLE-HOST-CLUSTER`. Items c and e are host-side plumbing under the
same claim (block-table slot fill / SamplingMetadata + SchedulerOutput) with no
dedicated matrix row. The rescan is the accepted upstream spike for the findings;
this file is the structured implementation spike contract + port map + disposition.

## Scope

In: the per-step host-side re-materialization / dead-work found by the rescan
that lives in files this claim owns —

- **(c)** `BlockTable::compute_slot_mapping` writes a tail-pad
  `slot_mapping[num_tokens:max_num_batched_tokens] = PAD_SLOT_ID` every step per
  KV group; the pad is dead (the sole consumer `prepare_inputs` slices
  `[0,total)`, and the decode graph re-pads via `BuildPaddedDecode`). Bound the
  fill to `[0,num_tokens)`.
- **(d)** the decode CUDA-graph capture-size set is the fixed constant
  `{1,2,4,8,16,32,64}`; it MISSES the 24 bucket and over-pads batches 17–31 to
  32 for `max_num_seqs=32`. Derive the set from `max_num_seqs` mirroring vLLM
  `_set_cudagraph_sizes` reduced to the full-decode-cudagraph regime.
- **(e)** `InputBatch::make_sampling_metadata` rebuilds unconditionally per step
  (vLLM caches `self.sampling_metadata`, rebuild gated on `batch_changed`);
  `SchedulerOutput` COPIES the `num_scheduled_tokens` map + `finished_req_ids`
  set where vLLM passes dict refs. Gate the rebuild + `std::move` the copies.

Out (reported for the owning claims, NOT touched here): the rescan §1 sub-levers
that live in `runner.cpp` — (a) the full-width `gather_block_table` /
positions int64→int32 re-materialization and the zero-copy device-view refactor,
and (b) the GDN group gathering the whole 8192-col table to consume only column
0. `runner.cpp` is owned by `CLAIM-ASYNC-SCHED-W3` (async paths) and the GDN
claims; the block-table gather/positions changes there must be done by / with
those owners to avoid an ownership conflict.

## Upstream chain

- (c) `vllm/v1/worker/block_table.py:141-164` + `_compute_slot_mapping_kernel`
  (`:325-381`): the Triton kernel pads `slot_mapping[num_tokens:max_num_tokens]`
  because vLLM's captured decode graph reads the persistent buffer's padded
  tail; our port builds padded inputs explicitly, so the tail-pad is dead.
- (d) `vllm/config/vllm.py::_set_cudagraph_sizes:1667-1770` (`max_cudagraph_
  capture_size = min(max_num_seqs*2, 512, max_num_batched_tokens)`; sizes
  `[1,2,4] + range(8, min(max_cap+1,256), 8) + range(256, max_cap+1, 16)`) and
  `vllm/config/compilation.py:683-684,1438-1444` (the full-decode-cudagraph
  dispatcher caps batches at `max_num_seqs`). For `max_num_seqs=32` the
  decode-relevant set is `{1,2,4,8,16,24,32}`.
- (e) `vllm/v1/worker/gpu_input_batch.py::refresh_metadata:812-830` (rebuild
  `sampling_metadata` only when `batch_update_builder` reports a change);
  `SchedulerOutput` dict fields passed by reference in
  `vllm/v1/core/sched/scheduler.py` (`num_scheduled_tokens`, `finished_req_ids`).

## Our baseline

- (c) `src/vllm/v1/worker/gpu/block_table.cpp:compute_slot_mapping` (pad loop);
  consumer `src/vllm/v1/worker/gpu/prepare_inputs.cpp:160-166` slices `[0,total)`.
- (d) `src/vllm/model_executor/models/qwen3_5.cpp` `kDecodeGraphSizes` /
  `PadToCaptureSize` (anonymous namespace; not unit-testable in place).
- (e) `src/vllm/v1/worker/gpu/input_batch.cpp:make_sampling_metadata` (rebuild
  every call); `src/vllm/v1/core/sched/scheduler.cpp:371,377` (map/set copies).

Honest gap: the runner-side §1 (a)/(b) waste is the largest single copy (the
8192-col GDN gather) but is out of this claim's file ownership — reported only.

## Port map

| Upstream | Local | Change |
|---|---|---|
| `block_table.py` kernel tail-pad | `block_table.cpp::compute_slot_mapping` + `block_table.h` | Drop the `[num_tokens,max)` pad loop; document the tail-pad deviation |
| `_set_cudagraph_sizes` | new `include/vllm/model_executor/models/decode_graph_sizes.h` (inline `DecodeGraphSizes`/`PadToCaptureSize`), consumed by `qwen3_5.cpp` | Derive the capture set from `max_num_seqs`; delete the fixed constant |
| `refresh_metadata` gate | `input_batch.{h,cpp}::make_sampling_metadata` | Cache + `sampling_metadata_dirty_` flag set on add/remove/condense/swap; rebuild-every-step retained only when `!no_penalties()` (output-token freshness) |
| `SchedulerOutput` dict refs | `scheduler.cpp:371,377` | `std::move` the local map + the member set (container plumbing only) |

## Tests to port

Upstream exercises these through `tests/v1/worker/test_gpu_input_batch.py`
(`_make_sampling_metadata`, `refresh_metadata`) and the block-table
`compute_slot_mapping` oracle. Local, per tier (doctest):

- (c) `tests/vllm/v1/worker/test_block_table.cpp` — bounded-fill contract
  (tail untouched, not PAD); `test_prepare_inputs.cpp` — consumer `[0,total)`
  unchanged (bit-identical).
- (d) new `tests/vllm/models/test_decode_graph_sizes.cpp` — the derived set for
  `max_num_seqs` ∈ {1,2,4,8,16,24,32,64}, pad-to-capture mapping, the 24-bucket
  presence for mns=32.
- (e) `tests/vllm/v1/worker/test_input_batch.cpp` — rebuild gated on batch
  change, identical result, penalty-path always-fresh; `tests/vllm/v1/test_scheduler.cpp`
  — SchedulerOutput contents unchanged after the move.

## Gates

- Correctness: bit-identical CPU — full `ctest` battery + `python3 -m unittest
  discover tests/tools`, clean `-Werror` rebuild, record + doc-checkpoint checkers.
- e2e: DGX token-exactness on the final SHA — 27B (235/235 + 16/16) and 35B
  greedy, under one `flock /tmp/gpu`. No A/B (the payoff is measured by the
  dispatched correct-state c2/c8 full-step probe and the next authorized exact
  grid; `benchmark_binding=false`, no speed credit from this change).
- Architectures/backends: CPU-first; the CUDA decode-graph path (item d) is
  DGX-verified by the token-exactness gate (padding rows are masked/inert).

## Dependencies

- `runner.cpp`-resident §1 (a)/(b) are BLOCKED on `CLAIM-ASYNC-SCHED-W3` /
  GDN-claim ownership; done by/with those owners.
- Region-partitioned with `CLAIM-ASYNC-SCHED-W3` on `input_batch.{h,cpp}` and
  `scheduler.{h,cpp}` (distinct lines: SamplingMetadata gating + SchedulerOutput
  move vs the async placeholder/last_sampled plumbing).
- No new toolchain, model, or data.

## Work breakdown

- W-c: block_table tail-pad removal (block_table.cpp + .h + test). One commit.
- W-d: decode-graph capture-size derivation (new header + qwen3_5.cpp + test +
  CMake). One commit.
- W-e: SamplingMetadata cache gating (input_batch.{h,cpp} + test) + SchedulerOutput
  move (scheduler.cpp + test). One commit (two clearly-partitioned parts).
- W-rec: record close (this spec disposition, state, ledger, engine-matrix,
  coordination). Each code commit carries README + BENCHMARKS.

## Risks/decisions

vLLM-defined behavior is not reopened. Product/policy calls: none — every item
is a pure mechanical mirror. Recorded deviation (c): our decode graph owns
padding (`BuildPaddedDecode`), so `compute_slot_mapping` no longer mirrors the
upstream kernel's tail-pad; the buffer tail is never read. Recorded decision (e):
the penalty-active path rebuilds every step (a conservative copy-semantics choice
vs vLLM's reference-held `output_token_ids`); the gate workload is greedy /
`no_penalties`, which gets the full caching win bit-identically. Decision (e2):
`std::move` only (NOT `std::map`→`unordered_map`) to preserve any iteration-order
behavior — the copy is the per-step waste; the container type is not reopened.
