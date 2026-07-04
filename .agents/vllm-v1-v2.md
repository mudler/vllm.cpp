# vLLM V1/V2 terminology

There is no "V2 engine". vLLM V0 was removed in 2025; everything lives under
`vllm/v1/`. "V2" refers only to the **Model Runner V2** (`vllm/v1/worker/gpu/`
package, "MRV2" in commit messages), an in-progress rewrite of the model
runner *inside* the V1 engine, gated by `VLLM_USE_V2_MODEL_RUNNER` /
`VllmConfig.use_v2_model_runner` (default: on for an allowlist of
architectures, required for DSpark/diffusion).

**We port MRV2**, not the legacy `gpu_model_runner.py` — upstream development
is converging on it.

## Two orthogonal axes: contract (T0) vs storage (M2)

"MRV2" spans **two orthogonal axes**, and "we port MRV2" means the first one:

1. **The scheduler-output CONTRACT + step-input SEMANTICS** — the shape the
   scheduler emits (`prefill_token_ids` on new reqs, resumed folded in
   as-new, cached diffs = `num_computed_tokens` + `new_block_ids`) and the
   step-input build (`query_start_loc` / `seq_lens` / `positions` /
   `slot_mapping` / `logits_indices`). This we MUST match; the scheduler
   already emits it and M1.4 ported it.

2. **The worker STORAGE / staging LAYOUT** — a SEPARATE axis. MRV2's `gpu/`
   package uses GPU-optimized staged tensors (`StagedWriteTensor` /
   `UvaBackedTensor` `BlockTables`, fused-Triton
   `apply_staged_writes`/`gather_block_tables`/`compute_slot_mappings`) plus a
   **transient** per-step `InputBatch` with the persistent state split into
   `req_states` + `BlockTables` on the runner. The V1 path
   (`block_table.py` + `gpu_input_batch.py`) instead has a **persistent**
   `InputBatch` that *holds* the `MultiGroupBlockTable`.

At **T0 (CPU, correctness-grade)** we port axis (1) — the contract — via the
host-array-friendly V1 *algorithm* on axis (2): `block_table.py` +
`gpu_input_batch.py` (persistent `InputBatch` holds `MultiGroupBlockTable`),
which is the identical algorithm to the MRV2 staged `BlockTables` minus the
device wrapper. The MRV2 staged-tensor storage is **deferred to M2**, when the
`vt` device is introduced. The runner we ultimately target is
`gpu/model_runner.py` (it asserts `prefill_token_ids`); the V1 runner's
MRV1-shape admission code (`resumed_from_preemption` / `resumed_req_ids` /
per-req `all_token_ids`) is **NOT** ported — it is dead under our MRV2
scheduler.

## The same two axes apply to the SAMPLER and the RUNNER LOOP (M1.7/M1.8)

The MRV2 sampler (`gpu/sample/sampler.py` + `gpu/sample/states.py`) is **axis (2)
storage**: persistent per-slot GPU buffers (`SamplingStates`/`PenaltiesState`/
`LogitBiasState`/`BadWordsState` on `UvaBackedTensor`s), populated by
`Sampler.add_request` + `apply_staged_writes` at admission and read back at sample
time through `input_batch.idx_mapping`. We do **NOT** port that at T0.

M1.7 ported axis (1) — the sampling **contract** — via the V1 host-array algorithm:
`SamplingMetadata` + `_make_sampling_metadata` (from V1 `gpu_input_batch.py`), built
fresh each step in the **dense `[0, num_reqs)` InputBatch order**. This composes
cleanly because our M1.5 `prepare_inputs` (V1 `_prepare_inputs`) emits `input_ids`/
`positions`/`query_start_loc`/`seq_lens`/`slot_mapping`/`logits_indices` in that
**same single dense order** — so attention metadata (M1.6), the gathered logits
(`hidden_states[logits_indices]`), the `SamplingMetadata` rows, and the sampled-token
write-back all align on ONE order. **No `idx_mapping` slot-indirection is needed at
T0** — that indirection is a consequence of the MRV2 *persistent-slot staged
storage* (axis 2), which we defer to M2. When the `vt`-device staged buffers land
(M2), the sampler moves to persistent-slot buffers + `idx_mapping`, matching
`gpu/sample/sampler.py`.

**One ordering nuance that IS axis-1 (must port at M1.8):** the runner reorders the
batch **decode-first-then-prefill** (`reorder_batch_to_split_decodes_and_prefills`)
before building metadata, because the GDN/mamba split (`split_decodes_and_prefills`,
ported in M1.6 GDN metadata) assumes decodes lead. This is a *semantic* ordering both
V1 and MRV2 need for the hybrid path, not a storage detail — M1.8 applies it so all
four consumers see the same decode-first order. It is **inert for the single-sequence
gate-model bring-up** (batch of 1) and for pure-decode / pure-prefill batches; it only
bites a MIXED decode+prefill batch on the hybrid (GDN) model.
