# Lost-lanes parity rescan — 5 lanes re-run (2026-07-16)

**Kind:** read-only source scan + strategy critique (no GPU, no engine change).
Re-runs the six lanes lost by the 2026-07-14 rescan workflow (4 stub outputs +
2 retry-cap deaths; see `parity-rescan-2026-07-14.md` "Coverage caveat") minus
`gdn` (superseded: the correct-state c16 kernel traces measured that lane
directly). 5/5 lanes returned grounded output this time (stub-validation +
one-retry harness; only strategy-challenge needed the retry). Full lane JSON +
synthesis in the session workflow record (`wf_f982abbc`); this file is the
durable disposition. Target was sharpened vs 07-14: the c2–c8 decode means
(2.2–5.5% slower) — the one family with no correct-state attribution.

## Verdicts that stand (dispositioned)

1. **BLOCK-TABLE HOST CLUSTER (3-lane corroboration: attention+host-scheduler+
   kv-cache).** Every decode step re-materializes the full-attention block
   table on the host into 4–5 fresh FULL-WIDTH vectors before the captured H2D:
   `commit_block_table` host→host scalar loop (`block_table.cpp:167-172`),
   `gather_block_table` fresh vector (`runner.cpp:509-519`, called 2×/step),
   `MakeCommonAttentionMetadata` copy (`backend.cpp:57`), BuildPaddedDecode/
   Refresh copies (`qwen3_5.cpp:5087-5089,5164`). Amplifier: no
   `--max-model-len` in the bench arm ⇒ max_model_len 262144 / block 32 ⇒
   cols=8192 ⇒ ~32 KB/row ×5 copies ×2 groups. Sub-levers: GDN group copies the
   whole 8192-col table twice to consume ONLY column 0; `compute_slot_mapping`
   host loop pads to max_num_batched_tokens then the pad is discarded
   (`prepare_inputs.cpp:162-166`) — dead work 2×/step. vLLM: ONE persistent
   CpuGpuBuffer mutated in place, one non-blocking H2D, on-GPU triton
   slot_mapping + gather (`block_table.py:107-205`, `:141-172`). Estimated
   ~50–250 µs/step at c8 aggregate (unmeasured), host-window-conditional.
   → FIX (mechanical mirror) dispatched; probe = scoped engine-thread timers.
2. **Sampler per-step cudaMalloc/cudaFree (+cudaHostAlloc/FreeHost on the async
   path).** `GreedyArgmaxHost` stack-constructs a DeviceBuffer per step
   (`sampler.cpp:42-52`, download() syncs); cudaFree device-syncs. vLLM keeps
   persistent + pinned buffers (`gpu_model_runner.py:873-878`). ~10–50 µs/step
   + a serialization point; the async path adds cudaHostAlloc/FreeHost per step
   (`async_output.cpp:24-25,50`) which claws back part of W3's overlap win.
   → Handed to the W3-throughput owner (its domain).
3. **RMSNorm kernel EFFICIENCY (reframe of the refuted "fusion" lever;
   strategy-lane SC2 + the 49983d8 reconciliation agree).** No fusion gap
   exists (vLLM `fuse_norm_quant=False`, stores bf16 + separate
   `cvt_fp16_to_fp4`, counts 144==144). What remains is per-launch efficiency:
   isolated ours 6.35–9.18 µs (M=16, H=2048–4096) vs vLLM ~3.0 µs/launch
   implied by trace (391 µs / 129); ours single-pass vectorized variant is
   1.27–1.49× but reorders the f32 reduction (token-exactness hazard). The
   c2–c8 relevance argument is NEW and strong: the ~129 launches/step are
   BATCH-INDEPENDENT, so the fixed cost is a LARGER fraction of the c2 mean
   (c2 gap is only ~2.4 ms/step total) than of c16. GPU-busy-robust (survives
   any host/GPU re-attribution). → Same-profiler microbench first (M=2/4/8/16
   both engines, one profiler), then port vLLM's OWN kernel
   (`csrc/layernorm_kernels.cu` family) 1:1 per ground-every-impl; dispatched.
4. **SC1 falsifier (the load-bearing one): the c2–c8 "host-side" attribution
   inherits from a falsified pillar.** The 07-14 rescan's "our GPU kernels net
   faster on the measured window" was measured on contamination-suspect
   pre-slot-fix binaries; decontaminated c16 shows ours GPU-busy is SLOWER
   (+4.65 ms). c2–c8 has NO correct-state same-profiler attribution.
   → task #10 (c2+c8 full-step GPU-busy/idle split, correct-state binary,
   same-profiler) dispatched as the adjudicator; host-only levers are
   conditional on it.
5. **Decode CUDA-graph capture set diverges from vLLM** (`qwen3_5.cpp:5029`
   `{1,2,4,8,16,32,64}` vs vLLM's `[1,2,4]+range(8,…,8)` ⇒ we miss 24 and
   over-pad batches 17–31 to 32). Zero c2–c8 impact (exact buckets both
   sides); bites c16/c32 wave boundaries. Mirror-fidelity fix folded into the
   cleanup dispatch.
6. Tail (opportunistic, µs-scale): SamplingMetadata rebuilt unconditionally +
   3 unconditional penalty-vector copies (`input_batch.cpp:277-282`);
   SchedulerOutput copies maps/sets + `std::map` where vLLM passes dict refs
   (`scheduler.cpp:371,377`); detokenizer O(N²) slow path (overlapped thread —
   PARKED, latent for long outputs, not a c2–c8 lever);
   `remap_gdn_state_slots` re-verified negligible (persisted scratch).

## Stale/corrected strategy-lane claims (recorded so they don't recur)

- "decode-norm-quant-fusion-reconcile-2026-07-16.md does not exist" — STALE:
  it landed at `49983d8` while the lanes were scanning; the reconciliation and
  SC2 AGREE on the efficiency reframe.
- "recurrence lever failed, c16 not closable, nothing in flight" — STALE:
  the occupancy-aware/vendored-cubin redesign is an active in-flight claim.
- SC3 (W3 TTFT risk) — REFINED by `89b329e`: the TTFT shift is closed-loop
  Little's-law repayment of a throughput-neutral TPOT win; the W3 ship gate is
  now THROUGHPUT (≥+1.5% c16 X), which by X=N/W self-corrects TTFT. The
  correct statement of SC3 stands: W3 must never ship on a TPOT-only "win".

## Lanes at parity (scanned, no finding)

Attention decode: FlashInfer/TRT-LLM decode path needs no per-step plan();
neither side builds paged-kv indptr/indices per decode step; workspace pooled;
capture scope equivalent; reorder_batch no-op both sides. Sampling: on-GPU
greedy path structure matches apart from finding 2.

## Implementation disposition (2026-07-16, `CLAIM-BLOCKTABLE-HOST-CLUSTER`)

The mechanical-mirror FIX dispatch for findings §1 (block-table host cluster),
§5 (decode-graph capture set) and §6 (SamplingMetadata / SchedulerOutput tail)
was executed CPU-first, test-first, bit-identical. Structured implementation
spec: [blocktable-host-cluster-cleanup.md](blocktable-host-cluster-cleanup.md).

- **§1 item c — LANDED (`8a717b2`).** `BlockTable::compute_slot_mapping` no
  longer writes the dead `[num_tokens, max_num_batched_tokens)` tail-pad; the
  decode graph re-pads via `BuildPaddedDecode` and the only other consumer slices
  `[0, total)`. Recorded tail-pad deviation. RED→GREEN `test_block_table`.
- **§5 item d — LANDED (`81afc36`).** The decode CUDA-graph capture set is
  derived from `max_num_seqs` in `include/vllm/model_executor/models/decode_graph_sizes.h`
  (`DecodeGraphSizes`/`PadToCaptureSize`), mirroring vLLM `_set_cudagraph_sizes`
  reduced to the full-decode regime. Derived set for `max_num_seqs=32` =
  `{1,2,4,8,16,24,32}` (adds the missing 24 bucket, drops the never-reachable 64;
  batches 17–24 stop over-padding to 32; **+1 captured decode graph**). CUDA-only,
  padding rows inert → token-exact. RED→GREEN `test_decode_graph_sizes`.
- **§6 item e — LANDED (`0c4b41c`).** `InputBatch::make_sampling_metadata` caches
  + rebuilds only on batch change (mirrors vLLM `refresh_metadata`; penalty path
  rebuilds every step for output-token freshness — greedy gate gets the full win
  bit-identically). `SchedulerOutput` `std::move`s the `num_scheduled_tokens` map
  + `finished_req_ids` set (container plumbing only; `std::map` kept). RED→GREEN
  `test_input_batch`.

**§1 items a-runner (full-width `gather_block_table` / positions int64→int32 /
zero-copy device views) and b (GDN group gathering the whole 8192-col table to
read column 0 only): NOT DONE here** — they live in
`src/vllm/v1/worker/gpu/runner.cpp`, owned by `CLAIM-ASYNC-SCHED-W3` (async
paths) and the GDN claims. Reported for those owners; the zero-copy view refactor
also needs a `CommonAttentionMetadata` ABI change (vector → span) spanning
non-owned files and is NOT a mechanical bit-identical mirror. Finding §2 (sampler
alloc, W3 owner) and §3/§4 (RMSNorm efficiency / c2–c8 attribution) are unchanged
and remain with their dispatched owners.

**Perf disposition:** `benchmark_binding=false`, NO speed credit from this change.
The payoff is measured by the dispatched correct-state c2/c8 full-step probe and
the next authorized exact grid. Gate: DGX token-exactness on the final SHA (27B
235/235 + 16/16, 35B); no A/B.
