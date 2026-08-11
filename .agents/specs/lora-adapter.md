# LoRA / multi-LoRA adapter subsystem (`LORA-RUNTIME`, `LORA-ENDPOINTS`)

Spike + implementation spec for the pure-C++20 1:1 port of vLLM's LoRA adapter
subsystem. Owning rows: [`LORA-RUNTIME`](../engine-matrix.md) and
[`LORA-ENDPOINTS`](../engine-matrix.md) in the LoRA-and-adapters section of the
engine matrix. Claims: `CLAIM-LORA-RUNTIME` (W0+W1) and
`CLAIM-LORA-RUNTIME-W2` in [coordination.md](../coordination.md).

Tracking issue for W2: [#278](https://github.com/mudler/vllm.cpp/issues/278).

Bug found after W2 landed: [#395](https://github.com/mudler/vllm.cpp/issues/395)
— `test_punica_cpu`'s own `RefShrink` reference read past `a_stacked` for the
out-of-range slot the case feeds on purpose, so W2 turned the
`sanitize-cpu (address,undefined)` lane red on `main`. The kernel was correct;
only the reference was missing the `s >= num_slots` half of the guard the case
exists to verify. Fixed in the test, with a mutation showing the case still
fails when the kernel's own guard is dropped. Related coverage gap found in the
same review and tracked separately:
[#400](https://github.com/mudler/vllm.cpp/issues/400).

Pinned oracle `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @ `555967922`
(vLLM 0.26.0.dev0). Every `file:line` below is read from that pin. This is the
#1 HIGH-priority gap from
[vllm-feature-gap-analysis.md](vllm-feature-gap-analysis.md) (LoRA runtime +
endpoints): the whole adapter subsystem is ABSENT in vllm.cpp.

## Scope

- **In (subsystem):** the whole vLLM `vllm/lora/` chain plus its serving
  surface — the LoRA weight containers (`LoRALayerWeights` /
  `PackedLoRALayerWeights` / `LoRAModel`), the punica batched shrink/expand
  apply (`punica_wrapper/` + `ops/torch_ops` + `ops/triton_ops`), the
  LoRA-wrapped linear/embedding/logits layers (`lora/layers/`), adapter
  load/parse (`lora_model.py`, `peft_helper.py`, `utils.py`), the multi-adapter
  LRU manager + slot mapping (`models.py` → `model_manager.py`,
  `worker_manager.py`, `v1/worker/lora_model_runner_mixin.py`), the
  `LoRARequest` plumbing (`request.py`, `resolver.py`), and the OpenAI
  `POST /v1/{load,unload}_lora_adapter` endpoints
  (`entrypoints/serve/lora/api_router.py`).
- **In (this change, W0+W1):** the W0 spike (this file) plus the W1 CPU brick —
  the `LoRALayerWeights` container, the punica CPU shrink/expand ops
  (`bgmv_shrink` / `bgmv_expand` / `bgmv_expand_slice` + `add_lora_linear`),
  and their batched apply on ONE `ReplicatedLinear`-style linear, unit-gated
  vs a double-precision hand reference. CPU-only, `-Werror`.
- **Out (this change):** GPU punica kernels (sgmv/bgmv triton, fp8, fused-MoE),
  TP slicing, the LRU multi-adapter manager, the model-runner integration, the
  loader/safetensors adapter read, the OpenAI endpoints, and the resolver.
  Each is a later W below with its own gate. No GPU, no model e2e in W1.
- **Rows touched:** no NEW counted rows. `LORA-RUNTIME` transitions
  `INVENTORIED` → `ACTIVE` (spike accepted + W1 brick landed with code+test
  anchors + claim). `LORA-ENDPOINTS` stays `INVENTORIED` (scoped W6 below).
  The engine-matrix LoRA-area rollup and Total row move accordingly (ACTIVE
  +1, INVENTORIED −1).

## Upstream chain

Pinned vLLM `555967922`. The complete LoRA surface, grouped:

### Weight containers
- `vllm/lora/lora_weights.py:13` `LoRALayerWeights` — two low-rank matrices
  `lora_a [rank, input]`, `lora_b [output, rank]`, `scaling = alpha/rank`;
  `optimize()` folds scaling into `lora_b` when `scaling != 1`
  (`lora_weights.py:36-42`); `input_dim`/`output_dim` properties
  (`:44-50`); `create_dummy_lora_weights` zeros (`:72-96`).
- `vllm/lora/lora_weights.py:99` `PackedLoRALayerWeights` — packed
  qkv_proj / gate_up_proj (list-of-tensors per sub-module); `pack`
  (`:126-152`), `pack_moe` / `pack_moe_stacked` (`:154-261`), per-slice
  `optimize` (`:263-270`).
- `vllm/lora/lora_model.py:60` `LoRAModel` — `id`, `rank`,
  `loras: dict[module_name -> LoRALayerWeights]`; `from_local_checkpoint`
  (safetensors read + `parse_fine_tuned_lora_name` + `PEFTHelper`),
  `clone`.

### Punica batched apply (the shrink/expand GEMM)
- `vllm/lora/punica_wrapper/punica_base.py:124` `PunicaWrapperBase` — the
  metadata state machine: `_token_lora_indices`, `_sampler_indices`,
  `_embeddings_indices`, sgmv prefill tensors; `update_metadata`
  (`:284-299`) → `convert_mapping` + `compute_meta`; the `add_shrink` /
  `add_expand` / `add_lora_linear` / `add_lora_logits` / `add_lora_embedding`
  contract (`:301-449`).
- `vllm/lora/punica_wrapper/utils.py:15` `compute_meta` — clusters identical
  consecutive LoRA ids into sgmv segments; `no_lora` when the whole batch is
  `-1` (`:40-41`). `convert_mapping` (`:54-160`) — `LoRAMapping` → the four
  index tensors; id→index reverse lookup, `-1` sentinel for "no adapter".
- `vllm/lora/punica_wrapper/punica_cpu.py:22` `PunicaWrapperCPU` — the
  PyTorch-native reference path we mirror on CPU: `_shrink_decode`/`_expand_*`
  → the torch ops; `add_shrink` (`:166`), `add_expand` (`:197`),
  `add_lora_linear` (`:265`, buffer = shrink then expand-add), `add_lora_logits`
  (`:314`).
- `vllm/lora/ops/torch_ops/lora_ops.py` — the exact elementwise math:
  `bgmv_shrink` (`:67`, `out = scaling * einsum("bi,boi->bo")`, overwrite),
  `bgmv_expand` (`:24`, `+=`/`=` over common_len), `bgmv_expand_slice`
  (`:110`, into `y[:, off:off+slice]`), `sgmv_*` = `repeat_interleave` then
  the bgmv (`:7,50,83`).
- `vllm/lora/ops/triton_ops/lora_shrink_op.py`, `lora_expand_op.py`,
  `lora_kernel_metadata.py` (`LoRAKernelMeta`) + fp8 + fused-MoE variants —
  the GPU kernels (GPU W, not W1). Semantics: `lora_id == -1` → kernel early
  exit / no write (mirrored on CPU as "skip", see base_linear.py:257-259 and
  the test CPU reference below).
- `vllm/lora/punica_wrapper/punica_selector.py` — platform → wrapper class.

### LoRA-wrapped layers
- `vllm/lora/layers/base.py` `BaseLayerWithLoRA`; `base_linear.py:70`
  `BaseLinearLayerWithLoRA` — `create_lora_weights` allocates the STACKED
  slots `lora_a_stacked` tuple of `[max_loras, 1, rank, input]` and
  `lora_b_stacked` `[max_loras, 1, output, rank]` (`:129-151`); `set_lora`
  copies one adapter into slot `index` (`:158-184`); `reset_lora` zeros
  (`:153-156`); `apply` → `_apply_lora_to_output` → `punica.add_lora_linear`
  (`:215-238`).
- `replicated_linear.py` (n_slices=1, the W1 vehicle), `column_parallel_linear.py`
  (+ merged qkv/gate_up packing + TP `slice_lora_a/b`),
  `row_parallel_linear.py`, `vocal_parallel_embedding.py`,
  `logits_processor.py`, `fused_moe.py`.

### Adapter load / config
- `vllm/lora/peft_helper.py:20` `PEFTHelper` — parse `adapter_config.json`
  (`r`, `lora_alpha`, `target_modules`, rsLoRA/DoRA guards);
  `vllm_lora_scaling_factor = alpha/r` (or `alpha/sqrt(r)` for rsLoRA,
  `:53-60`); `validate_legal` vs `max_lora_rank` (`:118-132`).
- `vllm/lora/utils.py` — `parse_fine_tuned_lora_name`, `get_lora_id`,
  `is_regex_target_modules`, `get_supported_lora_modules`.
- `vllm/config/lora.py` `LoRAConfig` — `max_lora_rank`, `max_loras`,
  `max_cpu_loras`, `lora_dtype`, `fully_sharded_loras`.

### Multi-adapter manager + runtime plumbing
- `vllm/lora/models.py` (`LoRAModelManager`, `LRUCacheLoRAModelManager`) —
  activate/deactivate slots, LRU eviction, `set_adapter_mapping`.
- `vllm/lora/worker_manager.py` (`WorkerLoRAManager`) — per-worker load/apply.
- `vllm/v1/worker/lora_model_runner_mixin.py:30` `LoRAModelRunnerMixin` —
  `load_lora_model`, `set_active_loras`, `maybe_setup_dummy_loras`; builds the
  `LoRAMapping` from the scheduler batch and calls `punica.update_metadata`.
- `vllm/lora/request.py` `LoRARequest` (id/name/path); `vllm/lora/resolver.py:14`
  `LoRAResolver` / `LoRAResolverRegistry` — dynamic name → adapter resolution.

### Serving endpoints
- `vllm/entrypoints/serve/lora/api_router.py:43,59` — `POST
  /v1/load_lora_adapter`, `POST /v1/unload_lora_adapter`, gated by
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING`; delegates to
  `OpenAIServingModels.{load,unload}_lora_adapter`.
- `vllm/entrypoints/serve/lora/protocol.py` — `LoadLoRAAdapterRequest`
  (`name`, `src`, ...), `UnloadLoRAAdapterRequest`.

## Our baseline

vllm.cpp has NO LoRA subsystem: `grep -ri lora src/ include/` returns only
`cache_salt`/`lora_name` cache-key fields (`KV-PREFIX-CACHE` extra-keys) and
prefix-cache plumbing — no weight container, no punica op, no wrapped layer, no
endpoint. Confirmed by the engine matrix (`LORA-RUNTIME`/`LORA-ENDPOINTS`
`INVENTORIED`) and feature-gap analysis.

Reusable seams already present:
- `include/vllm/model_executor/layers/linear.h` — `LinearMethodBase::Apply` /
  `UnquantizedLinearMethod` (the base-linear apply the LoRA delta is added on
  top of). The W1 brick mirrors `_apply_lora_to_output`: base output first,
  then `add_lora_linear`.
- `src/vt/` op providers (`vt::MatmulBT`, CPU backend) for the future GPU/vt
  punica path. W1 stays in portable C++ (`float`) so the numeric reference is
  exact and CPU-CI-gated; the vt/GPU kernel is a later W.

## Port map

W1 files (this change), all additive:

| New file | Ports FROM (pinned vLLM) |
|---|---|
| `include/vllm/lora/lora_weights.h` | `lora_weights.py:13-96` (`LoRALayerWeights`, `optimize`, `input_dim`/`output_dim`, dummy) |
| `include/vllm/lora/punica.h` | `punica_wrapper/punica_cpu.py:265` (`add_lora_linear`), `ops/torch_ops/lora_ops.py` (bgmv decls), `lora/layers/base_linear.py:100-238` (`LoRALinear` stacked-slot wrapper: create/set/reset/apply) |
| `src/vllm/lora/punica_cpu.cpp` | `ops/torch_ops/lora_ops.py:24-128` (bgmv shrink/expand/expand_slice), `punica_cpu.py:147-312` (shrink/expand/add_lora_linear), `lora_weights.py:36-42` (`Optimize`) |
| `tests/vllm/lora/test_punica_cpu.cpp` | `tests/lora/test_punica_ops.py:36-290`, `tests/lora/test_lora_functions.py` (scaling), `tests/lora/test_layers_utils.py` |

Deviations (recorded per discipline): the `-1` "no adapter" sentinel is
mirrored as SKIP (no contribution) exactly like the triton kernel early-exit
(`base_linear.py:257-259`) and the test CPU references
(`test_punica_ops.py:36-74`), NOT the naive torch `w[-1]` wrap of the
production torch-ops fallback. W1 collapses the stacked "1" layer dim
(`[max_loras, 1, o, i]` → `[max_loras, o, i]`) since a wrapped layer owns one
adapter slot per layer; the extra dim returns with the manager (later W).

## Tests to port

Named traceably after the upstream cases; ported now unless marked SKIP.

- `test_punica_ops.py::check_lora_shrink_kernel` / `_cpu_bgmv_shrink`
  (`:36-48,142-213`) → `test_punica_cpu.cpp` "bgmv_shrink matches
  per-LoRA matmul reference" (multi-adapter batch, scaling 0.5, `-1` skip).
- `test_punica_ops.py::check_lora_expand_kernel` / `_cpu_bgmv_expand`
  (`:51-74,216-290`) → "bgmv_expand add_inputs accumulates onto base";
  "bgmv_expand_slice writes the slice window".
- `test_punica_ops.py` end-to-end shrink→expand → "add_lora_linear equals
  scale * x @ Aᵀ @ Bᵀ added to base output" (double-precision hand reference).
- `test_lora_functions.py` scaling / `optimize` → "Optimize folds scaling into
  lora_b (scaling==1 no-op)".
- `test_layers_utils.py` (`_get_lora_device`) → NOT ported W1 (needs the
  quant-base-layer probe; SKIP-with-reason, tracked to the layers W).
- `test_peft_helper.py`, `test_lora_manager.py`, `test_punica_ops_fp8.py`,
  `test_add_lora.py`, `test_resolver.py`, `test_*_tp.py` model gates → later Ws
  (listed in the breakdown), not owed by W1.

## Gates

The row's runnable CPU gate, in full:

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build-cpu -j 18
ctest --test-dir build-cpu -R test_punica_cpu --output-on-failure
ctest --test-dir build-cpu -R test_lora_layers --output-on-failure
```

- **W1:** CPU `-Werror` clean build; `test_punica_cpu` all cases
  green; RED-first proven (the un-applied / index-`-1` path must FAIL the
  LoRA-applied reference). Numeric tolerance: exact vs a `double` hand
  reference (portable float path, no dtype rounding). Signal:
  RUNTIME-VERIFIED on CPU (the test binary executes).
- **Subsystem (later Ws):** punica op parity vs vLLM's torch_ops on matching
  random data; the GPU kernels token-exact vs the triton kernels; a real
  multi-LoRA model gate (e.g. `Qwen3` dense + adapters, mirroring
  `tests/lora/test_qwen35_densemodel_lora.py`) token-for-token vs the pinned
  vLLM oracle with adapters loaded; endpoint conformance vs the upstream
  router shapes. vLLM throughput parity with LoRA active is the closing bar.

## Dependencies

- Base-linear apply seam (`linear.h`) — present.
- `vt::` op providers — present (future GPU/vt path; W1 does not use them).
- `LoRAConfig` / `LoRARequest` typed configs — not yet ported; W1 does not need
  them (the wrapper takes explicit `max_loras`/`max_rank`/dims). They land with
  the manager/runtime W.
- No new third-party deps. No GPU. No model download for W1.

## Non-overlapping work breakdown

Row-sized, each its own claim/gate. W1 is this change.

- **W1 — CPU punica brick (THIS CHANGE):** `LoRALayerWeights` + bgmv
  shrink/expand/expand_slice + `add_lora_linear` + a `LoRALinear`
  (ReplicatedLinear, n_slices=1) create/set/reset/apply, unit-gated vs a
  double reference. `LORA-RUNTIME` → `ACTIVE`.
- **W2 — packed + column/row/merged layers (LANDED, issue #278):**
  `PackedLoRALayerWeights` (`Pack` + per-slice `Optimize` + `is_packed`), the
  multi-slice `AddShrink`/`AddExpand` (`output_slices` + `offset_start`) and
  multi-slice `AddLoraLinear`, `AddLoraEmbedding`/`AddLoraLogits`, the wrapped
  layer family (replicated, column, row, merged-column/gate_up, qkv,
  merged-qkv, variable-slice) with TP `SliceLoraA`/`SliceLoraB` including the
  fully-sharded (S-LoRA) rank-dim rules, plus embedding and logits LoRA. Ports
  `test_layers.py` through `test_merged_column_parallel_variable_slice`. See
  "W2 as built" below for what is deferred and what is a recorded gap.
- **W3 — metadata + mapping:** `LoRAMapping`, `convert_mapping`,
  `compute_meta`, the sgmv prefill segmentation, `PunicaWrapperBase` state.
  Ports `test_punica_ops.py` sgmv segment cases + `test_layers_utils.py`.
- **W4 — adapter load/config:** `PEFTHelper` (adapter_config.json),
  `parse_fine_tuned_lora_name`, `LoRAModel.from_local_checkpoint`
  (safetensors read). Ports `test_peft_helper.py`, `test_lora_checkpoints.py`.
- **W5 — multi-adapter manager + runtime:** `LoRAModelManager` /
  `LRUCacheLoRAModelManager` (slot activate/evict), `WorkerLoRAManager`,
  `LoRAModelRunnerMixin` batch→mapping. Ports `test_lora_manager.py`,
  `test_worker.py`.
- **W6 — endpoints + resolver (`LORA-ENDPOINTS`):** `LoadLoRAAdapterRequest`/
  `UnloadLoRAAdapterRequest`, `POST /v1/{load,unload}_lora_adapter`,
  `LoRAResolverRegistry`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING` gate. Ports
  `test_resolver.py`, `test_lora_functions.py` endpoint cases.
- **W7 — GPU punica kernels + model gate:** vt/CUDA sgmv/bgmv, fp8, fused-MoE;
  a token-exact multi-LoRA model gate vs the vLLM oracle; throughput parity.

## Risks / decisions

- **Sentinel semantics (`-1`):** decided — mirror the GPU/triton kernel + test
  CPU reference (skip, zero contribution), not the naive torch `w[-1]` wrap.
  Documented inline. Prevents a latent wrong-adapter bug when a base-model
  token co-batches with LoRA tokens.
- **Buffer dtype:** vLLM's punica buffer is float32 (`punica_cpu.py:305-308`);
  W1 uses `float` for the shrink buffer and accumulates in `double` for the
  reference — matching vLLM's "buffer is float32 by default, consistent with
  the triton op".
- **Stacked "1" layer dim collapse:** decided for W1 (one adapter slot per
  layer); the manager W restores it. Flagged so W5 does not assume W1's shape.
- **`optimize()` mutates `lora_b`:** folding scaling into `lora_b` is a
  one-way transform; W1 mirrors it but the `LoRALinear.SetLora` path keeps
  scaling explicit (passes `scaling` to `add_lora_linear`) so both modes are
  exercised, matching vLLM (packed layers `optimize` to scaling==1, plain
  layers keep it).

## W2 as built

Files (all additive except the two `punica` files and the two build lists):

| File | Ports FROM (pinned vLLM) |
|---|---|
| `include/vllm/lora/lora_weights.h` (+) | `lora_weights.py:99-282` (`PackedLoRALayerWeights`: `pack`, per-slice `optimize`, `is_packed`) |
| `include/vllm/lora/punica.h` (+) | `punica_cpu.py:166-236` (`add_shrink`/`add_expand`), `:238-263` (`add_lora_embedding`), `:265-312` (multi-slice `add_lora_linear`), `:314-351` (`add_lora_logits`) |
| `include/vllm/lora/layers.h` (new) | `layers/base_linear.py:70-238`, `layers/column_parallel_linear.py:85-746`, `layers/row_parallel_linear.py:22-177`, `layers/vocal_parallel_embedding.py:17-140`, `layers/logits_processor.py:20-208` |
| `src/vllm/lora/punica_cpu.cpp` (+) | the same punica_cpu.py ranges |
| `src/vllm/lora/layers.cpp` (new) | the same layers/ ranges + `lora_weights.py:126-270` |
| `tests/vllm/lora/test_lora_layers.cpp` (new) | `tests/lora/test_layers.py:116-1035` (through `test_merged_column_parallel_variable_slice`) + `tests/lora/utils.py:29-46` |

**Deferred out of W2, with reason (each is stated in the code it belongs to):**

- **The fully-sharded (S-LoRA) APPLY path** — `_mcp_apply`
  (`column_parallel_linear.py:24-82`) and
  `RowParallelLinearWithShardedLoRA.apply` (`row_parallel_linear.py:118-159`)
  are *defined* by `tensor_model_parallel_all_gather` / `all_reduce` sitting
  between the shrink and the expand. vllm.cpp's TP seam does not yet expose
  those collectives, and a version without them is only correct at
  `tp_size == 1` — a hollow port no gate could catch. The sharded classes
  therefore carry their **slicing** rules (which W2 does own) and inherit the
  unsharded apply, which IS the upstream behaviour at `tp_size == 1` — the
  shrink's `max_rank / tp_size` rows are the whole rank there and the
  all-gather is the identity. At `tp_size > 1` the inherited apply does NOT
  reduce to upstream (the shrink fills part of the buffer, the expand consumes
  all of it; a row-parallel layer's `lora_b` holds only this rank's output
  shard), so `ApplyLoraToOutput` **throws `std::logic_error`** there rather
  than adding a partial delta that looks plausible. A deferral refuses; it does
  not approximate. `CreateLoraWeights` likewise refuses either pairing upstream
  cannot construct — a sharded class without `fully_sharded_loras`, or a plain
  parallel class with it — mirroring `_fully_sharded_can_replace` /
  `_not_fully_sharded_can_replace` (`layers/utils.py:76-101`), which is the
  only thing binding class to flag upstream. The collective-bearing apply lands
  with the TP row.
- **`pack_moe` / `pack_moe_stacked`** (`lora_weights.py:154-261`) — they build
  the per-expert 3-D stacks that only `FusedMoEWithLoRA.set_lora` consumes; they
  belong with the fused-MoE LoRA layer, which is W7.
- **`LoRAConfig`** — W2 keeps taking `max_loras` / `max_lora_rank` /
  `fully_sharded_loras` explicitly, per the W1 decision above; the typed config
  lands with the manager (W5).
- **The metadata that produces the index arrays** — `token_lora_indices`,
  `sampler_indices` and `_embeddings_indices` all come from `convert_mapping`
  (W3). W2's entry points take the arrays as parameters, and the embedding layer
  reproduces `convert_mapping`'s own `slot if lora_id > 0 else 0` derivation for
  the gather (`punica_wrapper/utils.py:114`) from the single per-token slot
  array.

**Harness adaptations in the ported tests** (documented in the test header):
the base linear output is drawn rather than recomputed from a base weight
(our wrapped layer is a delta applier, exactly like `_apply_lora_to_output`
receiving an already-computed `output`); `torch.rand` under `set_random_seed`
becomes a deterministic LCG; upstream's 4096-wide shapes are scaled down while
keeping the shape *structure* (slice counts, rank padding, q-vs-kv widths);
tolerances are upstream's `TOLERANCES[torch.float32] = (5e-3, 5e-3)` applied as
`torch.testing.assert_close`'s `|a-b| <= atol + rtol*|b|`; and the TP slicing
runs at `tp_size` 2/4 directly, since `slice_lora_a`/`slice_lora_b` are pure
functions of `(tp_rank, tp_size)` and upstream's own tests only reach them at
`tp_size == 1`.

**W2 gate:** CPU `-Werror` clean build; `test_lora_layers` 16/16 (4,498
assertions) + `test_punica_cpu` 8/8 (149 assertions); full CPU `ctest` green
apart from the pre-existing failures tracked in #274. RED-first captured as a
compile failure against the absent `layers.h`. Every repair below carries its
own RED-first mutation, and the six mutations a fresh review found SURVIVING
(2026-08-10) each now turn a case red — see the commit message for the table.

**Known gaps, recorded rather than implemented (they are real, and W2 does not
close them):**

- **`ExpandPackedLora` has no test of its own.** It is
  `expand_packed_lora` (`column_parallel_linear.py:266-300`), reached from
  `MergedColumnParallelLinearWithLoRA::SetLora` only when the adapter ships
  fewer groups than the layer has slices. Upstream's own `test_layers.py` never
  builds that checkpoint shape either, so there is nothing to port; the case it
  serves (an `in_proj_qkv`-style fused group) arrives with the adapter LOADER
  in W4, and its test belongs there.
- **`MergedColumnParallelLinearVariableSliceWithLoRA::SetLora` dispatches on a
  SHAPE HEURISTIC, not on a type.** Upstream branches on
  `isinstance(lora_a, torch.Tensor)` (`column_parallel_linear.py:718-746`) — a
  single fused tensor versus a list. `MatList` erases that distinction, so we
  infer it from `lora_a.size() == 1 && lora_b.size() == 1 &&
  lora_b[0].rows != lora_b_rows_[0]`. A one-slice-shaped input whose row count
  happens to equal the first slice's width would take the wrong branch. The
  honest fix is a typed carrier for "one fused tensor" rather than a list of
  one, which belongs with the loader (W4) that produces it; it is an
  UNDOCUMENTED DEVIATION until then, and this is the record of it.
- **`ReindexShardedToFull` has no test reference.** It is the pure half of
  `logits_processor.py:166-183`, and the `sharded_to_full_mapping` it consumes
  is built by the adapter/vocab plumbing that does not exist yet. It is dead
  code today.
- **`MergedQKVParallelLinearWithLoRA`'s `output_sizes_` diverges from upstream
  when `total_num_kv_heads < tp_size`.** We take `total_num_kv_heads *
  head_size` as the unsharded k/v window; upstream's `QKVParallelLinear`
  REPLICATES kv heads in that regime (`num_kv_head_replicas`), so the unsharded
  size it slices against is not the same number. `num_kv_head_replicas` reaches
  `output_ids_` but not `output_sizes_`. No ported case has
  `total_num_kv_heads < tp_size`, so nothing catches it; the replicated-kv TP
  case lands with the TP row that can actually run it.
