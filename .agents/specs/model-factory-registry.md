# Spike: generic architecture-to-factory + reject-unknown (`MODEL-FACTORY-registry`)

**Row:** `MODEL-FACTORY-registry` (the cross-cutting factory spike in
[model-matrix.md](../model-matrix.md)). Roadmap block
[`ROAD-V1-C2`](../roadmap_v1.md) — the C2 next gate, prerequisite for every new
`MODEL-*` family.
**Parity pin:** `/home/mudler/_git/vllm` @ `e24d1b24fe96a56ba8b0d653efa076d03eb95d6c`.
**Inventory context:** [model-family-inventory.md](model-family-inventory.md)
(all 353 static architecture IDs already inventoried; this leaf closes WBS item 1
"Close `MODEL-FACTORY-registry` with deterministic reject-unknown behavior").

## Scope

**In.** The mechanism that replaces our hardcoded model selection with an
upstream-mirroring registry:

1. A **type-erased architecture-string -> factory** table (C++ mirror of
   `_ModelRegistry.models`, a `dict[str, _BaseRegisteredModel]` keyed by
   `model_arch`). Resolution consumes the complete `config.architectures` list
   in order and returns its first registered entry, exactly as
   `resolve_model_cls` iterates.
2. **Reject-unknown with vLLM's exact error shape** — a C++ mirror of
   `_ModelRegistry._raise_for_unsupported` (`registry.py:1051-1082`): the
   previously-supported, out-of-tree plugin, and default "not supported for
   now. Supported architectures: …" branches, driven by ported static
   `_PREVIOUSLY_SUPPORTED_MODELS` / `_OOT_SUPPORTED_MODELS` tables. The two
   table-driven branches are byte-identical to the full pinned oracle; the
   default branch is byte-identical to a pinned `_ModelRegistry` instantiated
   with the same implemented subset because our supported list is intentionally
   smaller than vLLM's 353-entry list.
3. **Per-family plug-in seam**: each registration carries the four things a
   family needs to be constructed — config parse hook, weight-name map/loader,
   model forward, and KV-cache-spec builder — plus a static **capability
   metadata** struct (mirror of the subset of `_ModelInfo` we consume:
   `is_text_generation_model`, `is_pooling_model`, `is_hybrid`,
   `has_inner_state`, `supports_multimodal`, `score_type`) sufficient to drive
   task-aware loader/runner construction (hybrid-vs-full KV,
   generation-vs-pooling/scoring).
4. **Re-wire the live dispatch**: replace the `IsDenseArch` (`num_experts==0`)
   branch in `model_loader.cpp` with an `architectures`-keyed registry lookup, so
   the two Qwen3.5 gate models resolve THROUGH the registry with no behavior
   change (regression gate below).

**Out (each its own row / spike).** Implementing any NEW family (WBS item 2+);
filling the 353-arch table (we register only implemented targets — reject-unknown
covers the rest, mirroring vLLM's "supported == registered" semantics); the
convert-type/runner-type suffix remap in `_normalize_arch` (task conversion,
deferred with pooling/seq-cls rows); multimodal/pooling/reward task routing beyond
the metadata slot; and the dynamic-Transformers compatibility contract
(inventory-spike WBS item 6). **Explicitly NOT mirrored** (recorded deviations,
Python-runtime-only): lazy import (`_LazyRegisteredModel`), the model-info
subprocess (`_run_in_subprocess`), the on-disk model-info cache, and the
`terratorch` / dynamic-Transformers / OOT-plugin *runtime* resolution branches of
`resolve_model_cls`.

## Upstream chain

| Upstream `file:line` (pin `e24d1b24`) | What it defines / we mirror |
|---|---|
| `registry.py:71-693` | The ten registration dicts (`_TEXT_GENERATION_MODELS` …) mapping `arch -> (module, class)`, merged in declaration order into `_VLLM_MODELS`. Entry format e.g. `"LlamaForCausalLM": ("llama", "LlamaForCausalLM")` (`registry.py:143`). We mirror this table but ONLY for implemented targets. |
| `registry.py:998-1003` | `_ModelRegistry.models: dict[str, _BaseRegisteredModel]`; `get_supported_archs() = self.models.keys()`. Our static table + `SupportedArchs()`. |
| `registry.py:1005-1049` | `register_model` (dedup/overwrite semantics, `<module>:<class>` parse). We mirror via a self-registering static entry; the string-split lazy form is N/A (deviation). |
| `registry.py:1051-1082` | **`_raise_for_unsupported` — THE reject-unknown contract.** Three branches: (a) any arch registered-but-inspection-failed -> "failed to be inspected"; (b) `_PREVIOUSLY_SUPPORTED_MODELS[arch]` -> "was supported in vLLM until v{ver}, and is not supported anymore…"; (c) `_OOT_SUPPORTED_MODELS[arch]` -> "is not supported in-tree anymore. Please install the plugin at {url}…"; (d) default -> "Model architectures {archs} are not supported for now. Supported architectures: {supported}". |
| `registry.py:1166-1190` | `_normalize_arch`: identity fast-path (`arch in self.models -> arch`) we mirror; the suffix remap (`ForSequenceClassification`→base via `try_match_architecture_defaults`) is OUT (task conversion). |
| `registry.py:1244-1296` | `resolve_model_cls`: the core loop `for arch in architectures: … return (cls, arch)`. We mirror the loop + the terminal `_raise_for_unsupported`; skip the `transformers`/`terratorch`/`auto`-fallback branches. |
| `registry.py:746-796` | `_ModelInfo` capability dataclass + `from_model_cls`. We statically encode the consumed subset per registration. |
| `registry.py:701-743` | `_PREVIOUSLY_SUPPORTED_MODELS` (32 arch->version) + `_OOT_SUPPORTED_MODELS` (4 arch->url). Ported verbatim as static maps for message parity. |
| `registry.py:832-967` | `_LazyRegisteredModel` (lazy import, module hash, model-info cache). **Deviation: not ported** — C++ static linkage has no import-time CUDA hazard and no cache need. |
| `registry.py:1396-1404` | `ModelRegistry = _ModelRegistry({arch: _LazyRegisteredModel(...) for … in _VLLM_MODELS})`. Mirrored by a translation-unit static registration table. |
| `include/vllm/transformers_utils/hf_config.h:22-24` | `HfConfig.architectures` (the `architectures` list read from `config.json`/GGUF) — the key our registry resolves on; already present. |

**Runtime trace plan:** none required. Registry resolution is pure host-side
string dispatch; it selects no GPU kernel and no dependency-resolved tactic, so
source inspection is sufficient (unlike the kernel/quant rows). The oracle's exact
error strings are captured by running the pinned pip-vLLM `ModelRegistry` on
unknown archs (see Gates).

## Our baseline

Honest current state (all `file:line` at HEAD):

- `src/vllm/model_executor/models/registry.cpp:10-31` — `ResolveModelForward`
  hardcodes the single string `"Qwen3_5MoeForConditionalGeneration"` ->
  `&Qwen3_5Model::ForwardDense`, throwing `std::runtime_error` for anything else.
  This is the **M0.9 dense single-sequence** signature and is NOT on the live
  batched engine path (comment at `registry.cpp:15-19`), so it does not even
  reject unknowns on the real path.
- `include/vllm/model_executor/models/registry.h:19-30` — `ModelForwardFn` is
  typed `(… const Qwen3_5MoeWeights&, const HfConfig&, vt::Queue&)`: the weights
  type is baked in, so it **cannot type-erase** another family.
- `src/vllm/entrypoints/model_loader.cpp:232-238` `IsDenseArch` (`num_experts==0`)
  + `:266-281` `FromModelDir` — **the ACTUAL live dispatch**: a two-way
  `num_experts` branch that is NOT keyed on `architectures`; it will classify ANY
  dense config as Qwen3.5-dense (`LoadQwen3_5Dense`) rather than rejecting it (the
  gap the inventory spike flagged). Overloaded constructors at
  `model_loader.cpp:157-191` (`Qwen3_5MoeWeights`) / `:196-230`
  (`Qwen3_5DenseWeights`) hardcode the two weight types.
- `include/vllm/v1/worker/gpu/runner.h:125-140,186-190` — `GPUModelRunner` carries
  a `{moe_weights_, dense_weights_}` raw-pointer pair; exactly one is non-null.
  Generalizing this pair to a type-erased model is the real engineering surface.

Gaps: no `architectures`-keyed lookup; no reject-unknown for a valid-but-
unimplemented arch; no capability metadata; no generic factory test.

## Port map

| Upstream surface | Local destination | Notes / deviations |
|---|---|---|
| `_ModelRegistry` + `models` dict + `resolve_model_cls` loop | new `include/vllm/model_executor/models/model_registry.h` + `src/vllm/model_executor/models/model_registry.cpp` (supersede the thin `registry.{h,cpp}`) — `ModelFactory`, `ModelRegistry::Resolve(const HfConfig&)`, `SupportedArchs()` | type-erased factory returns a `LoadedModel` the runner/loader hold behind an interface; files carry the upstream `file:line` header per discipline.md |
| `_raise_for_unsupported` (`registry.py:1051-1082`) | `RaiseForUnsupported(archs, supported)` in the same TU | strings byte-identical; assert against oracle in a T-parity test |
| `_PREVIOUSLY_SUPPORTED_MODELS`, `_OOT_SUPPORTED_MODELS` (`registry.py:701-743`) | static `const` maps in `model_registry.cpp` | ported verbatim (32 + 4 entries); refreshed by the upstream-sync cycle |
| `_ModelInfo` consumed subset (`registry.py:746-796`) | `struct ModelInfo` (POD) attached to each registration | only fields our loader/runner read; extend as task rows land |
| `ModelRegistry = _ModelRegistry({…})` global | one ordered `constexpr ModelRegistration[]` in `model_registry.cpp`, with `REGISTER_VLLM_MODEL(arch, factory, info)` as an entry helper rather than cross-TU static initialization | mirrors the central ordered global-instance pattern without lazy import or static-init-order risk |
| `IsDenseArch` branch + `{moe,dense}` weight pair | `ModelFactory::Build(config, source)` -> type-erased model; `model_loader.cpp` routes on `architectures` via `ModelRegistry::Resolve` | this leaf keeps the two Qwen3.5 weight paths working through the new seam (no runner rewrite beyond the indirection) |
| per-family: config parse / weight-name map / forward / KV-spec builder | the family leaf's `transformers_utils`, loader, `models/<module>.{h,cpp}`, and `MakeKvConfig` variant | filled by each `MODEL-*` leaf, NOT here |

## Tests to port

| Upstream test | Local (tier) | Notes |
|---|---|---|
| `tests/models/test_registry.py:31-67` `test_registry_imports` | `tests/vllm/models/test_model_registry.cpp` (T-unit) | every registered arch resolves to a non-null factory + expected capability flags (`is_text_generation_model` etc.); the CUDA-import assertions are N/A (no lazy import) |
| `tests/models/test_registry.py:87-103` `test_registry_model_property` | same file (T-unit) | `ModelInfo` fields (`supports_multimodal`, `score_type`) match the oracle for each registered arch |
| `tests/models/test_registry.py:151-159` `test_hf_registry_coverage` | same file (T-unit) | every registered arch has an example-config fixture in our test registry |
| `registry.py:1051-1082` (asserted via absence of an upstream test) | NEW negative test (T-unit + T-parity) | empty architecture list; unknown arch -> exact default shape/order against a pinned subset `_ModelRegistry`; `_PREVIOUSLY_SUPPORTED_MODELS` and `_OOT_SUPPORTED_MODELS` entries -> byte-identical full-oracle messages |
| `tests/models/test_initialization.py:50-197` `can_initialize` | per-family init/load test | construct the engine from a dummy config with no disk; **mapped to each family leaf** — the registry implementation commit checks this in SKIPPED with reason `MODEL-FACTORY-registry: no second family yet` until Llama lands |
| `tests/models/test_registry.py:135-148` lazy-modelinfo hash | NOT PORTED | deviation: no lazy import / model-info cache (recorded) |

## Gates

- **Correctness (reject-unknown parity):** the C++ `RaiseForUnsupported` message
  is byte-identical to the pinned vLLM oracle's `_raise_for_unsupported` for
  `_PREVIOUSLY_SUPPORTED_MODELS` / `_OOT_SUPPORTED_MODELS`. Empty lists and a
  garbage or registered-upstream-but-unimplemented arch match a pinned
  `_ModelRegistry` constructed with our implemented-key subset, including the
  supported-key order and formatting. Capability flags per registered arch
  match the oracle `_ModelInfo`.
- **No-regression (the real risk of re-wiring the live path):** after `IsDenseArch`
  is replaced by `ModelRegistry::Resolve`, BOTH gate models stay token-exact —
  re-run the merged greedy gates (35B MoE + 27B dense, 16/16 token-for-token vs
  the pip-vLLM oracle) and confirm no throughput/memory regression per the
  every-axis rule ([gates.md](../gates.md), [benchmark-protocol.md](../benchmark-protocol.md)).
- **CI:** `tests/vllm/models/test_model_registry.cpp` green on CPU;
  `python3 scripts/check-agent-record.py` + `python3 tests/scripts/test_agent_record.py` green.
- **First family (its own leaf, not this row):** e2e token-exact 16/16 vs oracle on
  a tiny checkpoint + the per-family perf/memory every-axis gate.
- **Commands:** `python3 scripts/check-agent-record.py`;
  `python3 tests/scripts/test_agent_record.py`; `ctest -R model_registry`; the
  gate re-runs on dgx under `flock /tmp/gpu` when 2+ GPU agents are active.

## Dependencies

- **Blocks:** every new `MODEL-*` family row — none can enter `READY`/claim until
  this contract lands (it is the plug-in seam they register into).
- **Depends on:** `HfConfig.architectures` (present, `hf_config.h:24`); the
  type-erasure generalization of `GPUModelRunner`'s `{moe,dense}` weight pair and
  `LoadedEngine`'s two constructors (this row owns that indirection). No new
  `KERNEL-*` / `QUANT-*` / `BACKEND-*` dependency for the registry leaf itself.
- **First family (Llama) depends on:** full-attention paged KV (present),
  RMSNorm/RoPE/SwiGLU/matmul `vt::` ops (present, already token-exact in the 27B
  dense gate), a **non-hybrid `MakeKvConfig`** variant (new — single full-attention
  group, no `MambaSpec`/GDN group; contrast `model_loader.cpp:129-155`),
  safetensors + GGUF loaders (present), a tokenizer (present).
- Toolchain/hardware/models/licenses: CPU CI for the registry unit + negative
  tests; a tiny Llama checkpoint (TinyLlama / Llama-3.2-1B, Apache-2.0 / Llama
  license) for the first-family CI gate; dgx + GB10 for the no-regression re-runs.

## Work breakdown

Claim-sized, non-overlapping leaves (registry leaf first, then one leaf per
family). Each family leaf is its own `MODEL-*` matrix row + leaf spike; a family
umbrella never grants ownership of the targets beneath it.

1. **`MODEL-FACTORY-registry` (this leaf, CPU-testable):** type-erased
   `ModelFactory` + central ordered registration table + entry macro +
   `RaiseForUnsupported` (exact strings + the two ported static tables) +
   `ModelInfo` struct; route `model_loader.cpp` on `architectures` via
   `ModelRegistry::Resolve`, replacing `IsDenseArch`; re-register Qwen3.5 MoE +
   dense through it (no new family, no behavior change). Ports the registry unit +
   negative tests. No-regression gate on both gate models.
2. **First family — `MODEL-TEXT-llama-llama-for-causal-lm` (Llama dense):** config
   parse, weight-name map (`q/k/v_proj`→fused `qkv_proj`, `gate/up_proj`→fused
   `gate_up_proj`), `LlamaModel::Forward` reusing the existing
   full-attn + SwiGLU + RMSNorm + RoPE ops (minus Qwen QK-norm, minus GDN),
   non-hybrid KV spec, tokenizer, per-model init + e2e token-exact gate. Unlocks 6
   llama aliases + `MistralModel`/`LlamaModel` embedding entries.
3. **Then one leaf per family, in reach order:** Qwen3 dense (adds per-head QK-norm;
   otherwise our Qwen3.5-dense minus GDN), Mistral (Llama + sliding window; shares
   `ROAD-V1-C5`), Mixtral (Llama attention + `FusedMoE`, reuses our MoE), then
   **Qwen3-Next** (hybrid GDN + MoE — closest to Qwen3.5, largest reuse).

## First-family recommendation — Llama dense (justified)

Recommend **`MODEL-TEXT-llama-llama-for-causal-lm` (`llama.py::LlamaForCausalLM`)**
as the first new family:

- **Highest reach per unit effort — it is the canonical vLLM base class.** Six
  text-generation aliases resolve directly to it: `CwmForCausalLM`
  (`registry.py:87`), `InternLM3ForCausalLM` (`134`),
  `IQuestCoderForCausalLM` (`135`), `LlamaForCausalLM` and its legacy
  `LLaMAForCausalLM` spelling (`143-146`), and `TeleChat3ForCausalLM` (`205`),
  plus the `LlamaModel`/`MistralModel` embedding entries (`224`,`231`). One
  family lights up the most rows.
- **Maximum reuse, minimum new kernel risk — a strict subset of our working
  Qwen3.5-dense forward.** Llama is pure full attention (no hybrid GDN group -> the
  SIMPLEST KV config), standard GQA + RoPE, SwiGLU MLP, RMSNorm — i.e. Qwen3.5-dense
  MINUS the per-head QK-norm and MINUS the GDN/linear-attention layers. Every
  `vt::` op it needs already runs and is token-exact in the 27B dense gate, so it
  exercises the new factory/registry seam against a real second architecture with
  near-zero new kernel work — the ideal shakedown for reject-unknown + the
  per-family plug-in.
- **Cheap, CI-friendly oracle.** Tiny Llama checkpoints (TinyLlama, Llama-3.2-1B)
  run on CPU in CI and under the pinned vLLM oracle, matching gates.md's
  "CI-runnable on CPU (0.6B model)" pattern for a token-exact gate.

## Risks/decisions

- **Product call — register the full 353-arch table as "known but unimplemented"?**
  DECISION: **mirror vLLM** — `supported == registered-and-implemented`; an
  unimplemented arch falls through to the same `_raise_for_unsupported` path. We DO
  port `_PREVIOUSLY_SUPPORTED_MODELS` / `_OOT_SUPPORTED_MODELS` as static data so
  the "supported until v… / install the plugin at …" messages match exactly. This
  is vLLM's own semantics, not a new choice.
- **vLLM-defined, not reopened:** the resolution order (iterate `architectures`,
  first match wins), the exact error strings, and the capability-flag meanings.
- **Supported-list parity boundary:** full vLLM and our implemented subset cannot
  emit the same default supported list by construction. The gate therefore runs
  upstream `_raise_for_unsupported` on a subset registry with the same ordered
  keys; previous/OOT messages still compare directly to the full oracle. This is
  a support-set difference, not relaxed message semantics.
- **Recorded deviations (Python-runtime-only, no C++ analogue):** no lazy import
  (`_LazyRegisteredModel`), no model-info subprocess/cache, no dynamic-Transformers
  / `terratorch` / OOT-plugin runtime fallback. The dynamic-Transformers
  compatibility contract is a separate future row (inventory-spike WBS item 6), not
  this leaf.
- **Type-erasure risk (the real surface):** generalizing the runner's
  `{moe,dense}` weight-pair to a type-erased model. Scoped so THIS leaf keeps the
  two Qwen3.5 weight types working through the new seam; the broader runner
  refactor rides along only as far as the indirection needs.
