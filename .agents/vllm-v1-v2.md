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
