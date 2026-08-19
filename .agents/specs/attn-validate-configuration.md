# SPEC — `validate_configuration` and the `supports_*` capability surface

Issue: [#1332](https://github.com/mudler/vllm.cpp/issues/1332)
Owning rows: `BACKEND-ATTN-REGISTRY` (`.agents/backend-matrix.md`, `ANCHOR-BACKFILL`,
claim `CLAIM-ATTN-REGISTRY-1`) owns the selector seam; `BACKEND-CUDA-COMP-FA`
(`PARTIAL`) owns the FA2 arms.
Scope of this change: **M0 and M1 of #1332 only.**

## Now

`BACKEND-ATTN-REGISTRY` stays `ANCHOR-BACKFILL`. This change adds the declared
capability layer the row recorded as deferred and re-anchors four headers onto
the tree pin. It does not move the row's lifecycle state, because the row's
open item — that the resolved name does not drive dispatch — is untouched and is
owed to M4.

## M0 — reconciliation, performed before any code

Checked at `origin/main` `f06b9e93d`:

- `git log --oneline --grep 'BACKEND-ATTN-REGISTRY'` returns one commit,
  `b3a4ca963` (the seam itself). No later landing.
- `git log -S'validate_configuration' --oneline` returns six commits, every one
  of them a spec, audit or spike that *names* the upstream method. `git log
  -S'supports_compute_capability'` returns two, both records-only spikes. The
  capability surface has never been implemented.
- `gh pr list --state open --limit 100`: 24 open pull requests, none touching
  `src/vllm/v1/attention/**`, `include/vllm/v1/attention/**` or
  `include/vllm/platforms/**`.
- `git branch -r | grep -iE 'attn|validate|registry'`: 25 remote branches match,
  every one of them a kernel or model-factory branch (`perf/attn-prefill-*`,
  `opt/decode-paged-attn`, `wmma-attn`, `row/MODEL-FACTORY-registry`,
  `row/KERNEL-ATTN-DECODE-D128`, `row/PERF-CPU-ATTN-DTYPE`). None is a selector
  branch.
- `.agents/NOW.md` names no attention-selector gate.

**Outcome: the gap is real, unowned by any branch in flight, and both rows that
own it already exist.** No new roadmap row is opened.

## Owed — recorded debt this change does NOT close

1. **The selector still routes nothing.** `SelectAttentionBackendName`'s result
   reaches exactly three places in
   `src/vllm/v1/worker/gpu/runner.cpp::GPUModelRunner::initialize_kv_cache`:
   `attn_backend_names_` (`:1002`), the `VT_ATTN_SELECT_LOG` debug print
   (`:1003-1024`), and `vllm::v1::CheckKvCacheShape` (`:1025-1029`).
   `dense_attn::AttnBlock` calls `vt::PagedAttention` unconditionally
   (`include/vllm/model_executor/models/dense_attn_block.h:532-536`) and
   `AttentionBackend::get_impl_cls()` returns `nullptr` for every dense backend.
   The real arm choice is the env-flag + shape + dtype ladder in
   `src/vt/cuda/cuda_paged_attn.cu:2696-2848`. Deleting the selector today would
   leave every emitted token identical. **Owner: #1332 M4, row
   `BACKEND-ATTN-REGISTRY`.**
2. **This layer is necessary and NOT sufficient.** M1 reproduces upstream's
   *declared* capability layer, and that layer is precisely what passed on a
   GB10 (capability 12,1) and then failed at launch: upstream's
   `FlashAttentionBackend.supports_compute_capability` is `capability >= (8,0)`
   (`vllm/v1/attention/backends/flash_attn.py:200-202` @ `5559679229`) while the
   shipped fatbin carries `sm_80` SASS plus `compute_80` PTX and nothing else, so
   every launch needs a driver JIT that fails with
   `cudaErrorUnsupportedPtxVersion`. **A predicate over the DEVICE cannot answer a
   question about the BINARY.** Landing M1 while claiming it fixes selection would
   be the defect #1332 exists to correct. The compiled-arch manifest (M2) and the
   launch probe (M3) are what make the layer sound. **Owner: #1332 M2/M3, row
   `BACKEND-CUDA-COMP-FA`.**

   **The measurement that settles this, and it is not a thought experiment.**
   Run through the pinned oracle on a GB10 board in the fleet (compute capability
   12,1), same wheel, same prompt, one variable:

   | Requested backend | Result |
   |---|---|
   | `FLASHINFER` | `GENERATED: ' Paris. The capital of France is also the capital of the Republic of France.'`, `GEN_RC=0` |
   | default (resolves `FLASH_ATTN`) | dies at the FIRST attention call, `cudaErrorUnsupportedPtxVersion` |

   vLLM's own priority list puts `FLASH_ATTN` at position 0 and `FLASHINFER` at
   position 1 for this device (`cuda.py:156-163`, the `else` arm our table row
   mirrors). **The priority-0 choice is unrunnable on the board and the
   priority-1 choice works, and every filter in this file passes both.** Both
   declare a compute-capability floor at or below 12,1; neither is asked what its
   own fatbin contains. That is the gap, stated as a pair of observations rather
   than as an argument, and it is why this row's M1 must not be reported as
   fixing selection. Motivation only — M2 and M3 close it and are NOT in this
   change.
3. **`AttnSelectorConfig::dtype` is not supplied by the runner.** The production
   call site fills `head_size`, `block_size` and `kv_cache_dtype` from the
   resolved per-layer KV geometry, but the model/query dtype is not available at
   `initialize_kv_cache` (the runner resolves only `ResolveKvCacheDType()` there),
   so `dtype` keeps its `kBF16` default in production and the `supports_dtype`
   predicate is exercised by tests alone. **Owner: #1332 M4.**
4. **The non-MLA sm_100 priority arm gained a `use_non_causal` guard upstream**
   between the two pins; see "Anchor reconciliation" below. Fixed in flow, tracked
   by its own issue, because the anchor being corrected points straight at it.

## Anchor reconciliation

Verified against a checkout of the tree pin. `git rev-parse HEAD` in that
checkout returns `5559679229bc961848b121ccdeaa8fa5d79bec98`, which equals the
`vllm_commit` in the `parity-pin` block of `.agents/upstream-sync.md`. Four
headers cited `@ pin e24d1b24`, the pin retired at W5 (`bc415a3e`).

| File | Cited at `e24d1b24` | Verified at `5559679229` |
|---|---|---|
| `include/vllm/v1/attention/backend.h:1` | `vllm/v1/attention/backend.py` | same file; `validate_configuration` is `:320-393`, the `supports_*` predicates `:154-317` |
| `include/vllm/v1/attention/registry.h:3` | `cuda.py:361-470` (`get_valid_backends` / `get_attn_backend_cls`) | `get_valid_backends` is `:359-394`, `get_attn_backend_cls` is `:397-492` |
| `include/vllm/platforms/cuda_attn_priority.h:2` | `cuda.py:84-176` `_get_backend_priorities`, MLA `:93-142`, non-MLA `:143-166`; `mla/prefill/selector.py:47-76` | `_get_backend_priorities` is `:83-163`, MLA `:93-143`, non-MLA `:144-163`; `selector.py:48-77` |
| `include/vllm/platforms/interface.h:2` | `interface.py:134-229` (`class Platform`) | `class Platform` is `:134-1290`; the cited `:409-439` capability accessors and `:181-187` `supported_dtypes` moved |

**One content drift, not just a line shift.** At `e24d1b24` the non-MLA sm_100
arm was `if device_capability.major == 10`. At `5559679229` it is
`if device_capability.major == 10 and not use_non_causal` (`cuda.py:148`), with
`use_non_causal` added as a fifth `_get_backend_priorities` parameter
(`cuda.py:88`); the comment at `:145-147` gives the reason — SM100f's non-causal
CUTLASS path is known-bad for DFlash. Our table keyed rows on `(use_mla, major)`
alone, so a non-causal request on sm_100 got upstream's causal ordering. Today
this changes no selected name in this tree, because `FLASHINFER` is not
registered and both orderings fall through to `FLASH_ATTN`; it would change one
the moment FlashInfer lands. Fixed here rather than deferred: `use_non_causal` is
a field this change adds to `AttnSelectorConfig` regardless, the fix is one row
plus one predicate in the same header whose anchor is being corrected, and it is
CPU-gateable through the existing `FakeCudaPlatform`.

The MLA rows, the sm_12x two-entry row, and the MLA-prefill lists are unchanged
in content at the new pin and were re-read line by line to confirm it.

**Not reconciled here, recorded instead.** `FlashAttentionBackend::get_kv_cache_shape`
mirrors `flash_attn.py::get_kv_cache_shape` at `e24d1b24`
(`num_blocks, 2, block_size, num_kv_heads, head_size`); at `5559679229` upstream
returns `(num_blocks, num_kv_heads, block_size, 2 * head_size)`. That is the KV
memory format the whole engine allocates and every paged-attention kernel reads,
so re-anchoring it is a kernel campaign, not a comment edit. Left at its
`e24d1b24` anchor with the divergence named in the header. Owner:
`BACKEND-ATTN-REGISTRY`.

## Design

Mirror upstream's structure exactly: predicates are virtual member functions
returning `bool`, `validate_configuration` collects reason strings, and an empty
list means valid.

**Where the config lives.** Upstream's `AttentionSelectorConfig`
(`vllm/v1/attention/selector.py:24-39`) is one named tuple that feeds both
`_get_backend_priorities` and `validate_configuration`. Our equivalent is
`vllm::platforms::AttnSelectorConfig` (`include/vllm/platforms/interface.h`),
already threaded through `Platform::get_attn_backend_priority`. It gains the
remaining upstream fields, every one with upstream's default, so no existing call
site changes meaning.

**`attn_type` is a string.** Upstream's `AttentionType` is a `str` enum and
`supports_attn_type` takes a `str` (`backend.py:292-298`). Our `AttentionType` is
an `enum class` in `vllm::v1`, and `platforms/` must not depend on `v1/`. The
config therefore carries upstream's string, which is the faithful shape, and
`backend.h` gains `AttentionTypeName()` to convert.

**Compute capability is checked only when the platform reports one.** Upstream
runs `validate_configuration` from `CudaPlatform.get_attn_backend_cls`, which
asserts `device_capability is not None` first (`cuda.py:403-404`); `CpuPlatform`
has its own selector (`cpu.py:75-87`) and never reaches this code. Our selector
is shared across every `DeviceType`, and `DeviceCapability::present()` is already
false (`major == -1`) for every non-CUDA platform. The predicate is therefore
applied only when `present()` is true, which reproduces upstream's precondition
rather than inventing a new rule. Recorded because it is the one structural
adaptation in this change.

**The refusal names what failed.** `SelectAttentionBackendName` keeps its two
paths. An explicit override that fails validation throws with the full reason
list, mirroring `cuda.py:416-420`. The priority walk skips an invalid candidate
and, on exhaustion, throws naming each candidate with its reasons, mirroring
`cuda.py:432-446`.

**Per-backend overrides ported.** `FlashAttentionBackend` gets upstream's
`supported_kv_cache_dtypes`, `get_supported_kernel_block_sizes` (`MultipleOf(16)`),
`supports_head_size` (`% 8 == 0 && <= 256`; upstream's FA4 `<= 512` arm is
unreachable here and says so), `supports_compute_capability` (`>= 8.0`),
`supports_sliding_window`, `supports_batch_invariance`, `supports_non_causal`,
`supports_attn_type` (all four types) and the `supports_combination` sink rule.
`TritonMLABackend` keeps `MultipleOf(16)` and its existing `is_mla()`.

**One deliberate divergence from upstream's per-backend data.**
`FlashAttentionBackend::supports_per_head_quant_scales()` is upstream's
`fa_version >= 3`; this tree ships FA2 only (`CMakeLists.txt`
`VLLM_CPP_CUDA_ARCHITECTURES`, `BACKEND-CUDA-COMP-FA` records FA3 as unported),
so it returns `false` — which is upstream's own answer for FA2 and is what makes
the ported `test_per_head_quant_scales_backend_selection` case assert a refusal.

## Tests

`tests/vllm/v1/attention/test_attn_validate_configuration.cpp`, a port of
`tests/kernels/attention/test_attention_selector.py` @ `5559679229`. That file
monkeypatches the device capability and asserts a chosen name, so every case
selected below runs on CPU against the existing `FakeCudaPlatform`, which takes
its capability by constructor argument.

| Upstream case | Ported as |
|---|---|
| `test_backend_selection` (cuda/cpu arms) | selection under a fully populated config still resolves `FLASH_ATTN` on sm_121 and sm_100 |
| `test_fp32_fallback` | an `f32` request finds no valid backend and the refusal names `dtype not supported`; upstream lands on `FLEX_ATTENTION`, which this tree does not register |
| `test_flash_attn` (upstream `pytest.skip`s it) | its five assertions ported as direct predicate cases: capability `(7,5)`, dtype `fp8`, `kv_cache_dtype="fp8"`, `block_size=8`, `head_size=17` |
| `test_per_head_quant_scales_backend_selection` | `FLASH_ATTN` + `use_per_head_quant_scales` is refused and the reason names it |
| `test_non_causal_backend_selection`, `test_non_causal_autoselect_backend` | `FLASH_ATTN` advertises `supports_non_causal()` and a non-causal request selects it |
| `test_flash_attn_rejects_unhandled_kv_cache_dtypes` | all six upstream parameters asserted false |
| `test_flash_attn_accepts_handled_fp8_variants` | both upstream parameters asserted true |
| `test_invalid_backend`, `test_auto_backend_string` | not applicable: our override is a `std::string` with `""` for automatic, so there is no enum lookup to fail |

Plus, beyond upstream: the reason list is asserted to be a LIST — a request that
violates three predicates at once produces three reasons, so a refusal message
cannot collapse to the first failure; the sm_100 `use_non_causal` ordering; and
`AttnSelectorConfig{}` (every default) still validates, which is the
behavior-preserving control for the existing production call sites.

## Gates

`scripts/agent-preflight.sh --fail-on-skip`, plus the focused
`test_attn_validate_configuration` and `test_attn_backend_registry` cases. No GPU:
every predicate is host code and the capability is injected.

## Evidence

CPU Debug build, `-DVLLM_CPP_CUDA=OFF`. Every run below is a direct binary
invocation, never through a pipe, so the reported status is the binary's.

**RED before.** `tests/vllm/v1/attention/test_attn_validate_configuration.cpp`
written first against the config fields alone, with no predicate implemented:

```
[doctest] test cases: 12 |  2 passed | 10 failed | 0 skipped
[doctest] assertions: 22 | 10 passed | 12 failed |
[doctest] Status: FAILURE!            (exit 1)
```

The two that passed are the behavior control (`FLASH_ATTN` still selected) and
the case asserting the sm_100 CAUSAL ordering, which was already correct — the
non-causal half of that same case is one of the ten reds.

**GREEN after**, with the predicate-level cases added:

| Binary | cases | assertions | status |
|---|---:|---:|---|
| `test_attn_validate_configuration` | 21 | 76 | SUCCESS |
| `test_attn_backend_registry` | 17 | 62 | SUCCESS |
| `test_runner` | 19 | 543 | SUCCESS |

**Mutations.** Each applied to the worktree, built, run, then restored from a
`tar` snapshot taken before the pass — not `git checkout`, which reads the index
and would have destroyed the untracked new test file. `git diff --stat` was
printed before and after each one and the line counts returned to their
pre-mutation values, and `compile_rc` was printed for each, because a mutation
that fails to build and a mutation that never applied both read as a pass.

| # | Mutation | `compile_rc` | Result |
|---|---|---:|---|
| M1 | `CandidateInvalidReasons` returns `{}` — the selector stops asking | 0 | RED: `validate_configuration` 12/21, `registry` 15/17 |
| M2 | drop the `capability.present()` precondition | 0 | RED: both files, 1 case each |
| M3 | `supports_block_size` compares equality instead of a multiple | 0 | RED: 4 cases |
| M4 | remove the #1333 causal-only guard | 0 | RED: 1 case |
| M5 | **reachability**: the production geometry stops flowing into the config (`cfg.head_size = 17`) | 0 | RED: `test_runner` **19/19 cases fail** |
| M6 | **reachability**: delete the production config fill entirely | 0 | GREEN: `test_runner` 543/543 |

**M5 is the reachability proof.** Corrupting one field of the config that
`GPUModelRunner::initialize_kv_cache` builds fails every one of the 19
`test_runner` cases, because the refusal propagates out of the runner's own
initialization. The capability layer is therefore reached from a production
entry point with production data, not only from its own unit test.

**M6 is the honest negative, and it is recorded rather than explained away.**
Deleting the fill leaves `test_runner` green, because the defaults still
validate. The layer is reached and executed; no test's failure depends on the
fill being PRESENT, only on its being CORRECT. That is the expected shape while
the selected name still dispatches nothing (`## Owed` item 1), and it is the
second reason this change must not be read as fixing selection.

## Found in flow, filed, not fixed here

[#1353](https://github.com/mudler/vllm.cpp/issues/1353): a full disk makes the
preflight report ten record and policy suites red while the only fault is
`No space left on device`, and the failures read as findings about records. Hit
on this row's own gate run at 100% disk; all ten went green after reclaiming this
row's build tree, with no tree change. Not fixed in this flow because it changes
preflight semantics and adds a refusal path, which `AGENTS.md` routes to the
normal row, spec and fresh-review path. Owner: `ENG-RECORD-ANCHOR-RATCHET`.

## Stop conditions

- Stop and report `NEEDS_DECISION` if a ported predicate would refuse a
  configuration the gate models use today, because that is a behavior change, not
  a capability port.
- Stop if `validate_configuration` cannot be reached from the production call
  site without also wiring dispatch; wiring dispatch is M4 and out of scope.
