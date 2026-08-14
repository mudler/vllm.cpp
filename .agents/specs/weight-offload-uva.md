# Inference-time CPU weight offload (`ENG-WEIGHT-OFFLOAD`)

Port vLLM's weight-offload surface: `cpu_offload_gb` UVA zero-copy offload with
name-segment targeting, plus the layer-group `PrefetchOffloader`. Issue
[#797](https://github.com/mudler/vllm.cpp/issues/797); the dense half of
[#149](https://github.com/mudler/vllm.cpp/issues/149). **Mirror floor** — vLLM
ships all of this at the parity pin `555967922`, so there is no design freedom
here and no secondary oracle: the port is transcription plus our own gates.

**Verdict up front.** The feature is fully specified upstream and its gate is
the easiest kind we have — upstream's own tests assert that offload-ON and
offload-OFF produce *identical output*, which is a token-exact inertness gate we
already know how to run. Three findings shape the port:

- **The non-UVA fallback cannot be CUDA-graphed.** Upstream's own test appends
  `--enforce-eager` whenever UVA is disabled (`test_cpu_offload.py:19-21`),
  because the fallback replaces `module.forward` with a `functional_call`
  wrapper. Since graphs are default-on for us, the fallback arm must refuse
  rather than silently disable graphs.
- **UVA is the wrong backend on GB10, and upstream already knows it.**
  `should_pin_memory()` (`offloader/base.py:23-33`) exists because on
  unified-memory systems pinned memory eats into the same pool the GPU uses —
  upstream names GH200; GB10 is the same class. So the row's own gate hardware
  makes its headline knob inert, exactly as `ENG-EXPERT-STREAM` found for tmpfs.
- **The recorded anchors are stale.** The matrix cites
  `gpu_model_runner.py:445,913`; at the pin those lines are unrelated and the
  real install point is `:939`. Re-derived at HEAD, as
  [`porting.md`](../porting.md) requires.

## Scope

| Field | Content |
|---|---|
| Row ID | `ENG-WEIGHT-OFFLOAD` (engine-matrix). Issue [#797](https://github.com/mudler/vllm.cpp/issues/797), dense half of [#149](https://github.com/mudler/vllm.cpp/issues/149) |
| In | `OffloadConfig` (`offload_backend` + the two sub-configs) as a config surface with vLLM's validator semantics verbatim; the `UVAOffloader` arm (per-parameter budgeted offload, dotted-segment targeting, pinned host copy, device view, the non-UVA fallback); the `PrefetchOffloader` arm (layer grouping, static buffer pool, async H2D prefetch, the sync/join seam); the install seam equivalent to `set_offloader`/`get_offloader().wrap_modules()`; the loader interaction that re-offloads a parameter replaced during weight load; ABI + CLI surface; both upstream test files ported |
| Out | Expert-granular disk paging (`ENG-EXPERT-STREAM`, READY — a different tier and a different granularity); per-tensor-group DEVICE placement and CPU-side expert execution (`ENG-HYBRID-PLACEMENT`, READY); KV offload (`kv_offload/`, already shipped); XPU's `get_xpu_view_from_cpu_tensor` arm; auto-fit style resolution of a budget from free memory (upstream has none — `cpu_offload_gb` is user-supplied) |
| Supported modes | `auto` (default; prefetch if `offload_group_size > 0`, uva if `cpu_offload_gb > 0`, else no-op), `uva`, `prefetch` — the upstream enum exactly, no additions |
| Dispatch behavior | Resolved once at model build. `cpu_offload_gb == 0` and `offload_group_size == 0` ⇒ the no-op offloader ⇒ the existing single-device path, byte-identical. Offloading is opt-in and inert when off, mirroring how `kv_transfer_config` already behaves |
| Regimes served | Discrete-GPU hosts where the model does not fit in VRAM. Inert on unified memory by construction |

## Upstream chain

Pin `555967922`, verified in the local checkout. Provenance note: vLLM's
`offloader/base.py` header records that it was itself *adapted from*
`sgl-project/sglang` `python/sglang/srt/utils/offloader.py`, so the lineage is
SGLang → vLLM → us. vLLM is the mirror source; SGLang is not consulted.

**Config** — `vllm/config/offload.py`:

- `:12` `OffloadBackend = Literal["auto", "uva", "prefetch"]`.
- `:16-44` `UVAOffloadConfig`: `cpu_offload_gb: float = Field(default=0, ge=0)`
  (`:23`), `cpu_offload_params: set[str] = Field(default_factory=set)` (`:34`).
  The docstring fixes the matching semantics: for `"mlp.experts.w2_weight"`,
  `"experts"` and `"experts.w2_weight"` match; `"expert"` and `"w2"` do NOT
  (`:39-43`). Empty set ⇒ offload non-selectively until the byte limit.
- `:48-76` `PrefetchOffloadConfig`: `offload_group_size: int = 0, ge=0` (`:54`),
  `offload_num_in_group: int = 1, ge=1` (`:62`), `offload_prefetch_step: int = 1,
  ge=0` (`:66`), `offload_params: set[str]` (`:70`). Worked example in the
  docstring: `group_size=8, num_in_group=2` offloads layers 6,7,14,15,22,23…
  (`:57`). Note the asymmetry with UVA: an empty `offload_params` here means ALL
  parameters *of each offloaded layer* (`:72-73`), not all parameters globally.
- `:80-93` `OffloadConfig`: `offload_backend = "auto"` (`:83`), `uva` (`:90`),
  `prefetch` (`:93`).
- `:96-136` `validate_offload_config`. Two hard errors — `offload_num_in_group >
  offload_group_size` and `offload_prefetch_step < 1` when prefetch is enabled —
  and three warnings for backend/field mismatch (uva-backend-with-prefetch-fields,
  prefetch-backend-with-uva-fields, auto-with-both, the last stating that
  prefetch wins).

**Selection and install**:

- `offloader/base.py:126-162` `create_offloader`: `auto` ⇒ prefetch if
  `offload_group_size > 0`, elif uva if `cpu_offload_gb > 0`, else `NoopOffloader`.
  UVA is constructed with `cpu_offload_max_bytes = int(cpu_offload_gb * 1024**3)`.
- `offloader/base.py:94-125` `NoopOffloader`, the module-global `_instance`, and
  `get_offloader`/`set_offloader`.
- `v1/worker/gpu_model_runner.py:939` — the single install point,
  `set_offloader(create_offloader(self.offload_config))`. **This supersedes the
  matrix's recorded `:445,913`, which are stale.**
- `model_executor/models/utils.py:816,824` — `make_layers` wraps the layer
  generator in `get_offloader().wrap_modules(...)`, between the PP-missing
  prefix and suffix. That is the ONLY place layers get wrapped.
- `offloader/base.py:46-92` the `BaseOffloader` ABC: `wrap_modules`, `post_init`,
  `sync_prev_onload`, `join_after_forward`, `_wait_for_layer`, `_start_prefetch`.

**UVA arm** — `offloader/uva.py`:

- `:46-49` `pin_memory = should_pin_memory()`; `uva_offloading = is_uva_available()
  and not VLLM_WEIGHT_OFFLOADING_DISABLE_UVA`.
- `:64-76` skip modules with no parameters, modules already on CPU, and everything
  once the budget is spent.
- `:80-84` **per-parameter, not per-module**: the budget check is inside the
  parameter loop and `break`s mid-module, so one module can end up partly
  offloaded. Mirror this exactly — a per-module budget would be a different
  feature with different memory behavior.
- `:91-93` the segment match, implemented as `f".{param}." in f".{name}."`. The
  dot-wrapping on both sides is what makes it a segment match rather than a
  substring match.
- `:97-99` `p.data.to("cpu")` then `.pin_memory()` when enabled.
- `:101-105` when UVA is on, `p.data = get_accelerator_view_from_cpu_tensor(cpu_data)`
  and the parameter is tagged `p._vllm_is_uva_offloaded = True`. When UVA is off,
  `p.data = cpu_data` and nothing is tagged.
- `:110-135` the non-UVA fallback: wrap `module.forward` so each call moves the
  state dict to the device and runs `functional_call(..., tie_weights=False)`.
  The comment at `:123-124` gives the reason for `tie_weights=False` — tied
  weights become untied when `.to(device)` is called per-parameter.
- `:107` the byte counter is incremented from `p.data` AFTER reassignment.

**Loader interaction** — `model_executor/model_loader/utils.py:160-193`: a
context manager records which parameters were UVA-offloaded, moves CPU
parameters to the device for the duration, and on exit **re-offloads any
parameter that was UVA-offloaded but got replaced by a new device tensor**
during loading. Without this, a quantization path that rebuilds a parameter
silently un-offloads it. This is the subtle half of the port.

**CUDA-graph seam** — `compilation/cuda_graph.py:310,324,359` and
`compilation/breakable_cudagraph.py:379,387,421` call
`get_offloader().sync_prev_onload()` and `join_after_forward()` around
capture/replay. The no-op offloader makes these free when offloading is off.

**Env vars** — `envs.py:278-279,1938-1943`:
`VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` and
`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA`, both default `False`, both read as
`bool(int(getenv(..., "0")))`.

**Helpers** — `utils/platform_utils.py:51-57` `is_uva_available()` (currently
`is_pin_memory_available() or current_platform.is_cpu()`, with an explicit
upstream TODO to tighten it); `utils/torch_utils.py:766-776`
`get_accelerator_view_from_cpu_tensor` dispatching to
`get_xpu_view_from_cpu_tensor` / `get_cuda_view_from_cpu_tensor`.

## Our baseline

- **Nothing of this exists.** `grep -ri 'cpu_offload\|offload_backend\|OffloadConfig'`
  over `src/` and `include/` returns only `v1/kv_offload/` (KV blocks, a
  different subject) and two unrelated `UvaBackedTensor` hits in
  `include/vllm/v1/worker/gpu/block_table.h:23` and `runner.h:31`, which are
  host-visible control tensors, not weight offload.
- What we do have that this must reuse rather than duplicate: the tiered-cache
  machinery in `src/vllm/v1/kv_offload/` (an `OffloadingManager` ABC, an LRU
  `cache_policy`, a filesystem tier, an async transfer worker). Its
  `OffloadKey = BlockHashWithGroupId` (`include/vllm/v1/kv_offload/base.h:49`)
  is KV-block-shaped and that file is a 1:1 upstream mirror, so weight residency
  gets a sibling manager sharing `cache_policy`/`fs_io` rather than a mutated key
  type. Upstream keeps them separate too, which settles it.
- `include/vllm.h` already carries the precedent for a nested vLLM config passed
  as JSON: `kv_transfer_config` (`:296`) takes `--kv-transfer-config`'s document
  verbatim, with "NULL or empty ⇒ byte-identical default engine". `OffloadConfig`
  is the same shape and takes the same treatment.
- `include/vllm.h:342-361` documents the whole-engine memory budget and its
  resolution order (`num_blocks` > `kv_cache_memory_bytes` >
  `gpu_memory_utilization`). An offload budget has to state its place in that
  order rather than becoming a fourth independent claim on the same memory.

## Port map

Structural deviations, and why each is forced:

- **No `functional_call`, no module monkey-patching.** vLLM's non-UVA fallback
  rebinds `module.forward` at runtime; we have no equivalent and would not want
  one. Our fallback is an explicit per-layer staged upload before the layer runs.
  This is a mechanism deviation with identical semantics, and it is the reason
  the fallback arm's CUDA-graph refusal (below) is ours to enforce rather than
  inherited.
- **Config lives in the C ABI as JSON**, mirroring `kv_transfer_config`, because
  our ABI has no pydantic and flattening a nested config into C fields would
  diverge the names the moment upstream adds a backend.
- **`is_uva_available()` is re-derived, not copied.** Upstream's version is
  `is_pin_memory_available() or current_platform.is_cpu()` and carries its own
  TODO admitting it is under-specified. We mirror the *behavior* (UVA requires
  pinned memory) and record the divergence if our platform layer answers
  differently.
- **The `_vllm_is_uva_offloaded` tag becomes a property of our weight record**,
  not an attribute monkey-patched onto a tensor, since the loader interaction
  above depends on it surviving a parameter being rebuilt.

Everything else — the enum, the field names, defaults, bounds, the two hard
validation errors, the three warnings, the auto-selection order, the
per-parameter budget break, the dotted-segment match, and the empty-set
asymmetry between the two backends — is transcribed, not designed.

## Tests to port

Both upstream files, preserving parameters and intent:

- `tests/basic_correctness/test_cpu_offload.py:9-29` — the 2x2 matrix over
  `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` and
  `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA`, asserting `--cpu-offload-gb 1` gives the
  same output as no offload. **Preserve `:19-21` verbatim in intent**: when UVA
  is disabled the arm also passes `--enforce-eager`, because graphs only work on
  the UVA path.
- `tests/quantization/test_cpu_offload.py` — offload-ON vs offload-OFF on fp8,
  gptq-marlin, awq-marlin and compressed-tensors w4a16 checkpoints, each skipped
  when the quant method is unsupported. These exist because offload interacts
  with `process_weights_after_loading`, which is precisely the loader
  re-offload path above; the quantized arms are where that path breaks.

Ours in addition:

- The dotted-segment matcher as a table: `"experts"` and `"experts.w2_weight"`
  match `mlp.experts.w2_weight`; `"expert"` and `"w2"` do not; `"w2_weight"` does
  NOT match `mlp.experts.w2_weight_scale`. Asserted by **counting** matches, not
  by spot-checking one name.
- The per-parameter budget break leaves a module partly offloaded — assert the
  partial state explicitly, since a per-module implementation would pass a
  whole-model byte total while behaving differently.
- Validator: both hard errors and all three warnings.
- `auto` selection order, including prefetch-wins-when-both-set.
- Inert: an existing SACRED golden with no offload config is byte-identical.
- Refusal: the non-UVA fallback with graphs enabled.

## Gates

- **Token-exact, offload-ON vs offload-OFF.** This is upstream's own gate shape
  (`compare_two_settings`) and the strong form: offloading is a memory placement
  decision and must not change a single token. Runs against our own goldens; no
  oracle execution needed for correctness, because the invariant is internal.
- **Inertness.** Existing SACRED goldens with the feature unconfigured are
  byte-identical, proving the no-op path is the current path.
- **Memory.** The point of the feature is a smaller device footprint: measure
  resident device bytes with and without `cpu_offload_gb`, and record the ratio.
  A feature that offloads N GiB and does not free N GiB has a defect.
- **Speed, recorded not waived.** Offload trades throughput for capacity, so
  decode tok/s will drop. Record the value at a fixed `cpu_offload_gb`; it is a
  number, not a pass/fail, and it is the honest denominator for
  `ENG-HYBRID-PLACEMENT`'s claim that CPU-side execution beats it.

**Hardware.** Correctness, inertness and the validator gates run anywhere.
The memory and speed gates need a **discrete GPU**: on GB10 the host and device
share one physical pool, so `cpu_offload_gb` frees nothing and the ratio is
meaningless. Same blocker as `ENG-HYBRID-PLACEMENT` W4, and named for the same
reason — so a row that lands everything except its number is not read as done.

## Dependencies

- No hard dependency on `ENG-EXPERT-STREAM` or `ENG-HYBRID-PLACEMENT`; all three
  are independent and compose later. This row is the only one of the three that
  is a pure mirror.
- **Surface reconciliation is owed, once.** `ENG-EXPERT-STREAM`'s spec predates
  vLLM shipping `OffloadConfig` and proposed ds4-style CLI semantics
  (`--ssd-streaming-cache-experts 32GB`) with its own `off`/`nvme` modes. Now
  that upstream has a backend enum whose documented targeting example is
  `"experts"`, the two rows should expose one surface with a disk backend added,
  rather than two vocabularies for one idea. Whichever lands second owns the
  reconciliation; it is not free and it is not this row's alone.
- A discrete-GPU host for the memory and speed gates.

## Work breakdown

| ID | Work | Gate |
|---|---|---|
| W0 | `OffloadConfig` + both sub-configs + the validator, transcribed with bounds, defaults, two errors and three warnings; ABI JSON field + CLI. No offloader yet | validator tests RED-first; inertness golden byte-identical |
| W1 | The offloader seam: ABC, no-op default, install point, and the layer-wrap site equivalent to `make_layers` | inertness unchanged; no-op proven to be the current path |
| W2 | `UVAOffloader`: per-parameter budget, dotted-segment match, pinned host copy, device view | segment-match table; partial-module assertion; token-exact ON vs OFF |
| W3 | The loader re-offload interaction — a parameter rebuilt during load must be re-offloaded | the quantized arms of upstream's test (fp8 + a marlin arm) |
| W4 | The non-UVA fallback as explicit staged upload, and its **refusal** under CUDA graphs | refusal test; token-exact eager |
| W5 | `PrefetchOffloader`: grouping, static buffer pool, async H2D, sync/join seam | grouping table (`group_size=8, num_in_group=2` ⇒ 6,7,14,15,…); token-exact |
| W6 | Memory ratio + recorded decode cost | **BLOCKED on a discrete-GPU rig** |

W0–W5 are all reachable on GB10 because they are correctness work. Only W6 needs
hardware we do not have, which is the opposite balance from
`ENG-HYBRID-PLACEMENT` and the reason this row is the safer of the two to start.

## Risks and decisions

- **UVA is inert on our own gate hardware.** Everything provable here is
  provable on GB10, but the feature's actual purpose — freeing device memory —
  cannot be demonstrated on it at all. The risk is landing a correct feature
  nobody has ever seen do its job. W6 stays open and visible.
- **The loader re-offload path (W3) is where this breaks.** Upstream devotes a
  whole second test file to quantized checkpoints for exactly this reason: a
  parameter rebuilt by `process_weights_after_loading` silently loses its
  offload. Any port that skips W3 will pass bf16 and fail every quantized arm,
  and our quantized arms are the ones users run.
- **Per-parameter vs per-module is easy to get subtly wrong.** The budget break
  sits inside the parameter loop, so a module can be half-offloaded. A
  per-module implementation matches on total bytes and diverges on behavior —
  which is why the test asserts the partial state rather than the total.
- **Two surfaces for one idea** if the `ENG-EXPERT-STREAM` reconciliation keeps
  being deferred. Recorded under Dependencies as owed by whichever lands second.

## Now

`ACTIVE` since 2026-08-14 (`CLAIM-WEIGHT-OFFLOAD-W0A`). **W0a landed**: the
config surface — the backend enum, both sub-configs with upstream's bounds and
defaults, `Validate()` carrying the two hard errors and the three collected
warnings, the dot-anchored segment match, the `int(gb*1024**3)` truncation, the
auto-selection order, the layer grouping, and JSON parsing on the
`kv_transfer_config` precedent. RED-first captured on a compiling stub (11/11
cases, 51/122 assertions red, build rc=0) then green 11/11, 126/126, and
mutation-proven 6/6 with each mutation's compile status reported so a
non-building mutation could not read as a pass.

It is deliberately UNREACHABLE: nothing constructs an `OffloadConfig` yet, which
is why every existing gate is byte-identical by construction. **Owed to finish
W0**: the `include/vllm.h` JSON field and the server CLI flag. Then W1.

One correction the RED pass forced, recorded because the distinction is easy to
lose: `ResolvedBackend()` mirrors `create_offloader` exactly — an EXPLICIT
backend is selected even at a zero budget — and `is_offloading_enabled()` is a
separate predicate for "would anything actually move". Conflating them would let
a zero-budget explicit backend read as offloading-on.

Issue
[#797](https://github.com/mudler/vllm.cpp/issues/797) is the owning issue.
This row is the dense half of [#149](https://github.com/mudler/vllm.cpp/issues/149),
whose CPU-MoE half is `ENG-HYBRID-PLACEMENT` and whose multi-GPU half is
[#147](https://github.com/mudler/vllm.cpp/issues/147) / `BACKEND-DISTRIBUTED-TP`;
#149 closes when all three land, not when this one does.

Next action is W0, which needs no hardware we lack.
