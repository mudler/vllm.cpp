# Qwen3.5-0.8B CPU and gfx1100 numerical characterization

Row: `BACKEND-ROCM`.

Issue: [#2773](https://github.com/mudler/vllm.cpp/issues/2773), which supersedes
issue #1588 for implementation traceability.

Branch: `row/BACKEND-ROCM-NUMERICS-1588`.

Active primary oracle pin: vLLM
`e126687a9a828d513c01a07cd69f025f27d63280`.

## Now

`PENDING`. This repaired spec is the only completed deliverable in this commit.
Implementation cannot start until a fresh reviewer passes this immutable spec.
Results cannot be accepted until both correctness prerequisites pass:

1. Issue #2772 and PR #2856 own the default-on ROCm `wvSplitK` sacred-anchor
   repair. The reviewed old base failed prompt 10/token 10 (`369` instead of
   `488`), while `VT_ROCM_SKINNY=0` passed. The operator reran PR #2856 at
   `f06619e4c213e3de28359ee10995e682e8c06932`: CPU mode 2/2, prerequisites
   77/77/77/86, `wvSplitK` 79796/79796, and sacred 137/137 with 15 strict, one
   tied, maximum gap zero, and zero divergence. PR #2856 is still open and the
   contributor lacks merge authority. This prerequisite remains `PENDING`
   until a maintainer lands it and the unchanged default gate passes on the
   implementation base. Disabling skinny GEMM is diagnostic only.
2. The configured source and runnable ROCm wrapper are still at historical
   vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`. The active source object is
   available for read-only inspection, but no runnable active-pin ROCm runtime
   has been supplied or proved. Issue #2794 records repository pin-validation
   context; #2773 itself owns the cache-matched active-pin Qwen3.5-0.8B captures.
   Active-pin capture and token revalidation remain `PENDING` until that runtime
   exists and runs the model under the GPU mutex.

Do not create characterization goldens from a known-regressed local default or
from the historical oracle revision.

## Scope

### In scope

- Add opt-in production-boundary dumps for the residual stream, attention
  output, MLP output, stored K/V cache, and persistent GDN convolution/SSM
  state.
- Compare CPU and gfx1100 on identical weights and identical token prefixes.
- Report descriptive `max_abs`, RMS, and relative-L2 deltas.
- Port applicable active-pin upstream operation tests with their parameters,
  modes, fixtures, failure cases, and exact tolerances.
- Audit runtime dtype, byte width, requested/resolved cache dtype, selected
  attention backend, and provider counts for `auto`, `bfloat16`, and
  `fp8_e4m3`.
- Run cache-matched active-pin vLLM captures and the established end-to-end
  oracle-backed token/near-tie gate after the prerequisites pass.

### Out of scope

- Changing a numerical tolerance after observing results.
- Treating CPU as the conformance oracle for FP8 physical-cache behavior.
- Permanently expanding the sacred 16-prompt gate merely because this
  characterization measures three cache modes. A new permanent case or golden
  may land only when #2773's active-pin capture proves it is necessary and the
  final implementation records the evidence. Broader gate policy needs its own
  issue.
- Enabling ROCm static graph mode; issue #332 owns that work.
- Performance claims or tuning.

## Artifact and oracle pins

Use `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`
from `${CHECKPOINT_ROOT}/Qwen3.5-0.8B`. Before every model run, require:

```sh
sha256sum \
  "${CHECKPOINT_ROOT}/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors" \
  "${CHECKPOINT_ROOT}/Qwen3.5-0.8B/model.safetensors.index.json" \
  "${CHECKPOINT_ROOT}/Qwen3.5-0.8B/config.json"
```

Expected hashes are:

```text
04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696  model.safetensors-00001-of-00001.safetensors
d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e  model.safetensors.index.json
b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204  config.json
```

Resolve the active oracle source with:

```sh
git -C "${VLLM_SOURCE}" rev-parse e126687a9a828d513c01a07cd69f025f27d63280^{commit}
```

The configured checkout may remain detached at the historical revision. Read
active-pin files with `git -C "${VLLM_SOURCE}" show <pin>:<path>`; do not move
that checkout. Source availability does not prove runtime gateability.

## Upstream executing chain

All cross-file anchors name symbols and refer to active pin `e126687a9` unless
explicitly labeled historical:

- `vllm/model_executor/models/qwen3_5.py::Qwen3_5ForCausalLM` defines the text
  model and layer composition.
- `vllm/model_executor/layers/mamba/mamba_utils.py::MambaStateShapeCalculator.gated_delta_net_state_shape`
  defines the GDN convolution and SSM state shapes. It starts at line 258 at
  the active pin. `MambaStateShapeCalculator.mamba2_state_shape` is a different
  architecture and is not the anchor.
- `tests/kernels/mamba/cpu/test_cpu_gdn_ops.py::test_fused_sigmoid_gating_delta_rule_update_cpu`
  and `test_chunk_gated_delta_rule_cpu` define CPU operation tolerances.
- `tests/kernels/mamba/test_gdn_forward_core_split.py::test_forward_core_split_matches_unified`
  defines split/unified state and output comparisons.
- `tests/kernels/attention/test_cache.py::test_reshape_and_cache` defines
  cache-store dtype behavior and FP8 comparison tolerances.
- `tests/kernels/attention/test_cpu_attn.py::varlen_with_paged_kv` starts at
  line 415. Its `_FP8_ATOL` and `_FP8_RTOL` constants at lines 45 and 46
  set `atol` to `0.2` for E4M3 and `0.3` for E5M2, with `rtol=0.1`.
  The comparisons at lines 629 to 653 select these bounds for FP8 and
  `atol=1.5e-2`, `rtol=1e-2` otherwise. Preserve both split and unsplit checks.

The earlier spec cited these surfaces from historical `555967922`; those
citations remain diagnostic history but cannot satisfy the active-pin gate. The
implementer must refresh line evidence from the active object and run the active
runtime before acceptance.

The local production entry is `ModelRegistry::Forward`, which dispatches the
registered `ForwardQwen3_5Dense` factory. Its paged forward reaches
`DenseForwardBody` -> `DenseForwardLayers` -> `RunDenseLayerPaged` ->
`FullAttnBlockPaged` -> `dense_attn::WriteKvCache` -> `vt::ReshapeAndCache` or
`vt::ReshapeAndCacheFp8`. The full-attention caller is
`src/vllm/model_executor/models/qwen3_5.cpp:7758`. `FullAttnBlockPaged` starts
at line 5726 and writes through `dense_attn::WriteKvCache` at
`src/vllm/model_executor/models/qwen3_5.cpp:5909`, before `vt::PagedAttention`
at line 5931. Place the stored-K/V probes after that write. Place GDN probes
after the persistent writes in `GdnBlockPaged` for prefill and decode.

This Qwen3.5 path does not call `dense_attn::AttnBlock`. Issue #2923 under
`Owed` records this existing shared-seam debt. Characterize the actual path
and cite the executing local and active-pin upstream symbols in the evidence.

## Dtype and state contract

The checkpoint resolves model dtype BF16 and explicitly sets
`mamba_ssm_dtype=float32`. The measured surfaces are:

| Surface | Shape per active unit | Physical storage |
|---|---|---|
| Hidden/residual | `[T, 1024]` | BF16 |
| Full-attention K/V | `[T, 2, 256]` per layer | BF16 or FP8 E4M3 |
| GDN convolution state | Per layer and slot: oracle SD `[3, 6144]`, local DS `[6144, 3]` | BF16 |
| GDN SSM state | `[1, 16, 128, 128]` per layer | FP32 |

The active oracle defaults to SD physical convolution storage, with axes
`(state_len, dim)`. DS reverses these axes to `(dim, state_len)`.
`vllm/model_executor/layers/mamba/mamba_utils.py::get_conv_state_layout`, lines
28 to 44, returns `SD` without an override. `_orient_conv_shape`, lines 162
to 166, and `gated_delta_net_state_shape`, lines 268 to 272, produce
`[3, 6144]` per slot for this workload. The kernel consumes a DS view
`[6144, 3]` after the transpose in
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1296` to line 1303.
Local `MakeQwen3_5KVCacheSpec` stores DS directly in
`src/vllm/model_executor/models/qwen3_5_common.cpp:85`. Leave
`VLLM_SSM_CONV_STATE_LAYOUT` unset for the production oracle. Record physical
layout, kernel-view layout, and actual strides on both sides. A transposed
view does not change the underlying physical layout or byte count.

Across six full-attention layers, BF16 K/V use 12,288 bytes per cached token
and FP8 K/V use 6,144 bytes. Across 18 GDN layers, convolution state uses
663,552 bytes and SSM state uses 18,874,368 bytes per active slot. Confirm
allocated bytes from runtime specs; arithmetic alone is not evidence.

For every compared buffer and GEMM output, log the local symbol, active-pin
upstream symbol, runtime dtype, shape, active elements, bytes on each side, and
the reason beside every FP32 exception. An unjustified wider local dtype is a
`DTYPE_WIDTH_MISMATCH` even when tokens agree.

## State dump design

Enable dumps only when `VT_DUMP_ACT` and `VT_DUMP_ACT_SUB` name the same empty,
writable directory. Preserve existing keys and add:

| Key | Production source | Dump dtype | Shape |
|---|---|---|---|
| `state_fa_k` | stored K after `dense_attn::WriteKvCache` | FP32 | `[active_tokens, 512]` |
| `state_fa_v` | stored V after `dense_attn::WriteKvCache` | FP32 | `[active_tokens, 512]` |
| `state_gdn_conv` | persistent convolution cache after update | FP32 | `[active_requests, 18432]` |
| `state_gdn_ssm` | persistent SSM cache after update | FP32 | `[active_requests, 262144]` |

Widen stored BF16 to FP32. Dequantize stored FP8 with the recorded per-layer
scale. The FP32 file format never conceals separately logged physical dtype and
allocation.

Use canonical DS order for `state_gdn_conv`, with `0 <= d < 6144` and
`0 <= s < 3`:

```text
local_dump[r, d * 3 + s] = widen(local[slot_local(r), d, s])
oracle_dump[r, d * 3 + s] = widen(oracle[slot_oracle(r), s, d])
```

Here `r` follows scheduler request order, and each arm resolves its own slot.
Use the recorded strides and storage offsets for each lookup. Record this
mapping in the manifest. The canonical DS dump order does not prescribe
physical storage or make the oracle's transposed view contiguous.

Gather nonnegative slot mappings in input-token order and active GDN indices in
scheduler request order. Refuse unexplained duplicate destinations, missing
rows, partial joins, or capacity-wide dumps. A 24-layer step emits 48 state
blobs, two per layer. Place each probe after its production write and preserve
queue ordering.

## Workload and backend identity

Use one binary and checkpoint copy. Run one request at concurrency one with
`The capital of France is`, greedy sampling, seed 0, MTP disabled, and eight
output tokens. Keep block size, block count, and scheduler budget equal. Run
each arm and cache mode in a separate process and empty directory. Take two
enabled repeats and one dump-disabled identity control.

Log and assert requested cache dtype, resolved storage and FP8 interpretation,
normalized selector input, physical cache dtype, selected attention backend,
and actual provider execution for every arm and mode.
`LoadedEngine::ApplyResolvedCacheDType` in
`src/vllm/entrypoints/model_loader.cpp:1835` applies `ParseCacheDType` from
`include/vllm/v1/kv_cache_dtype.h:58` to the cache spec. The runner constructs
`cfg.kv_cache_dtype` from resolved storage and interpretation at
`src/vllm/v1/worker/gpu/runner.cpp:1590`. `KvCacheDTypeName` maps `kBF16` to
`auto` in `src/vllm/v1/attention/backend.cpp:91` to line 96. Therefore the
production CPU selector receives these values for this BF16 workload:

| Requested cache dtype | Resolved physical storage | Normalized selector input | Selected CPU backend |
|---|---|---|---|
| `auto` | BF16 (`kBF16`) | `auto` | `CPU_ATTN` |
| `bfloat16` | BF16 (`kBF16`) | `auto` | `CPU_ATTN` |
| `fp8_e4m3` | E4M3 bytes (`kI8` plus `kFp8E4M3`) | `fp8_e4m3` | `CPU_ATTN` |

CPU priority starts with `CPU_ATTN` in `src/vllm/platforms/cpu.cpp:51`.
Its accepted selector values include `auto` and `fp8_e4m3` in
`include/vllm/v1/attention/backends/cpu_attn.h:108` to line 110.
`SelectAttentionBackendName` returns the first valid candidate in
`src/vllm/v1/attention/registry.cpp:124` to line 128.
Selection alone does not prove execution. Independently record the executing
attention and cache-store operator providers, selection counts, declines, and
reference-tier hits. The runner's selected name validates cache configuration
but does not dispatch the model's attention call. If main changes this chain
before implementation, update the cited contract before code and obtain fresh
review.

CPU is a diagnostic comparison arm for CPU-versus-ROCm deltas, including FP8
state after dequantization. Active-pin vLLM and its upstream tests are the
conformance authority. CPU results cannot accept or reject FP8 physical-cache
semantics.

Join only steps with identical input token prefixes. Include the first divergent
output step and exclude all later steps. Record the first divergence and number
of excluded steps.

## Descriptive metrics and acceptance

Decode joined blobs to FP32. Let `A` be ROCm and `B` be CPU. Report:

```text
max_abs = max_i(abs(A_i - B_i))
rms     = sqrt(sum_i((A_i - B_i)^2) / N)
rel_l2  = sqrt(sum_i((A_i - B_i)^2)) / sqrt(sum_i(B_i^2))
```

Define relative L2 as zero when both norms are zero and infinity when only `B`
has zero norm. Report the worst index, both manifests, joined/rejected rows, and
excluded steps. Refuse missing or duplicate keys, dtype/shape/byte mismatches,
nonfinite values, and partial joins.

These layer metrics are descriptive. They have no propagated numerical
acceptance envelope, discontinuity multiplier, or post-measurement threshold.
The rejected `E(k)` construction is removed because operation counts do not
bound cancellation, conditioning, nonlinear sensitivity, correlated
reductions, FP8 scaling/saturation, or subnormals.

Acceptance consists of both:

1. Ported active-pin operation tests with their exact upstream per-surface
   tolerances. Examples include GDN `atol=rtol=1e-2` where
   `test_cpu_gdn_ops.py` specifies it, exact convolution-state equality in the
   split-core test, its dtype-specific output/SSM tolerances, and cache/CPU
   attention FP8 tolerances selected by the upstream fixtures. Do not collapse
   them into one project-wide tolerance.
2. The established end-to-end 16-prompt strict-token and ratified 500
   milli-nat near-tie gate against deterministic, cache-matched, active-pin
   vLLM captures. A new physical cache mode needs its own capture and
   teacher-forced gaps before it can be accepted.

If no upstream analogue exists for a layer boundary, report its metrics only.
A future stage-level acceptance rule requires an independently justified bound
or explicit developer ratification before values are inspected.

## Provider and end-to-end contract

For `auto` and `bfloat16`, require the established native operator set,
including nonzero `kReshapeAndCache`, zero `kReshapeAndCacheFp8`, zero declines,
and zero reference-tier hits. The two modes resolve to physical BF16 and must
produce identical local token streams.

For `fp8_e4m3`, replace the store requirement: require nonzero
`kReshapeAndCacheFp8`, zero `kReshapeAndCache`, nonzero native selections for
all other applicable operators, zero declines, and zero reference-tier hits.
Provider sets are mode-specific; never require a BF16 store from a correct FP8
run.

The existing gate remains the permanent gate until #2773 produces reviewed
active-pin evidence for an additional case. Characterization must run all three
modes, but this spec alone does not authorize new permanent goldens.

## Required capture-tool implementation

The current `scripts/qwen3-oracle-capture.py` and
`scripts/qwen3-neartie-gap.py` hard-code `enforce_eager=True` and expose no
cache-dtype argument. Before active-pin capture, add reviewed options equivalent
to:

```text
--kv-cache-dtype {auto,bfloat16,fp8_e4m3}
--execution-mode {production,eager}
--seed INT
--max-tokens INT
--repetitions INT
--model-revision REV
--vllm-revision REV
--provenance-out PATH
```

`production` must instantiate vLLM without `enforce_eager=True`; eager is a
diagnostic arm and never the denominator. The output must record source, wheel,
image and artifact hashes, complete arguments, a hash of the scripts' shared
16-entry `PROMPTS` list, batching, concurrency, sampling, seed, token count,
repetitions, cache mode, resolved
cache dtype, execution mode, and output hash.

Extend `test_qwen35_paged_engine` with future environment inputs
`VT_QWEN35_GATE_DIR` and `VT_QWEN35_KV_CACHE_DTYPE`. The first selects an empty
issue-evidence directory instead of the committed golden directory. The second
sets `EngineParams::kv_cache_dtype` and prints requested, resolved, and physical
dtype. These commands describe the intended interface and data flow after
those changes. They are future commands and will fail today:

```sh
VLLM_PIN=e126687a9a828d513c01a07cd69f025f27d63280
MODEL_REV=2fc06364715b967f1860aea9cf38778875588b17
MODEL="${CHECKPOINT_ROOT}/Qwen3.5-0.8B"
for mode in auto bfloat16 fp8_e4m3; do
  GOLDEN_DIR="evidence/2773/sacred-${mode}"
  test ! -e "${GOLDEN_DIR}"
  mkdir -p "${GOLDEN_DIR}"
  "${VLLM_ORACLE}" scripts/qwen3-oracle-capture.py \
    --model "${MODEL}" --model-revision "${MODEL_REV}" \
    --vllm-revision "${VLLM_PIN}" \
    --kv-cache-dtype "${mode}" --execution-mode production --seed 0 \
    --max-tokens 16 --runs 10 --per-prompt --out-dir "${GOLDEN_DIR}" \
    --provenance-out "evidence/2773/oracle-${mode}.json"
  VT_QWEN35_GATE_DIR="${GOLDEN_DIR}" \
  VT_QWEN35_KV_CACHE_DTYPE="${mode}" \
  VT_DUMP_IDS=1 build-rocm/tests/test_qwen35_paged_engine
  test -s "${GOLDEN_DIR}/our_ids.i32"
  "${VLLM_ORACLE}" scripts/qwen3-neartie-gap.py \
    --model "${MODEL}" --model-revision "${MODEL_REV}" \
    --vllm-revision "${VLLM_PIN}" \
    --kv-cache-dtype "${mode}" --execution-mode production --seed 0 \
    --max-tokens 16 --topk 20 --golden-dir "${GOLDEN_DIR}" \
    --provenance-out "evidence/2773/neartie-${mode}.json"
  VT_QWEN35_GATE_DIR="${GOLDEN_DIR}" \
  VT_QWEN35_KV_CACHE_DTYPE="${mode}" \
    build-rocm/tests/test_qwen35_paged_engine
done
```

The tools must refuse when their `PROMPTS` lists differ from each other or from
`tests/parity/test_qwen35_paged_engine.cpp::Prompts`. Record exact as-run
commands. Do not create a golden unless all 10 repeats are deterministic.

The internal CPU/ROCm state characterization remains eight output tokens as
specified under `Workload and backend identity`. The permanent sacred-gate
candidate uses the existing 16-prompt, 16-output-token regime above. Never use
the eight-token characterization files as sacred-gate goldens.

## Tests and review mutations

The later implementation starts with focused tests that fail because the four
state rows and capture options are absent. It must test row shape/dtype,
nonmonotonic and negative slot mappings, inactive-capacity exclusion, stored
BF16 widening, post-quantization FP8 dequantization, persistent GDN prefill and
decode state, incomplete-step refusal, comparator structural refusals and
zero-denominator rules, dtype-width refusal, backend/dtype logging, and
mode-appropriate provider sets. The checkpoint-backed case must enter through
`LoadedEngine::FromModelDir`.

A fresh reviewer mutates each guarantee in a scratch copy: remove each
production probe, substitute pre-write tensors, remove negative-slot filtering,
sort by physical slot, dump capacity, misreport FP8 as BF16, swap the mode's
cache-store operator, inject a reference-tier hit, remove the production entry
point, and force eager oracle mode. Each focused test must fail for the intended
reason and the reviewer must restore the tree byte for byte.

This spec repair changes no runtime behavior. `IMP-TEST-FIRST` and
`IMP-MUTATE` are future implementation/review gates, not evidence claimed by
this commit.

## Gate order

The later implementation must satisfy, in order:

1. #2772 lands and the unchanged default local gate passes on the chosen base.
2. A runnable active-pin ROCm vLLM runtime is identified and proves this model.
3. Artifact, source, wrapper, wheel, image, prompt, and output hashes are saved.
4. The capture tools fail first for missing options, then pass focused tests.
5. Ported active-pin operation tests pass with unchanged upstream tolerances.
6. Dump/comparator CPU tests and checkpoint-backed CPU trace pass.
7. The controlled full preflight runs; every skip remains `PENDING`.
8. Fresh immutable review detects every required mutation.
9. Two CPU and two gfx1100 trace repeats plus disabled controls pass per mode.
10. Cache-matched active-pin production captures are deterministic and the
    established end-to-end gate passes per measured mode.
11. Mode-appropriate native selections are nonzero, with zero declines and
    zero reference-tier hits.
12. The operator independently reruns focused, full, oracle, and hardware gates.

No GPU result, oracle result, model execution, runtime mutation, or
implementation test is claimed by this spec-only repair.

## Evidence required

Store evidence under an issue-specific durable path and record commit/tree
hashes, clean status, exact commands/statuses, every skip, active oracle and
artifact hashes, compiler/build/ROCm/driver/board identity, GPU mutex evidence,
requested/resolved/physical cache dtype, selected CPU and ROCm attention
backends, provider counts, enabled repeats, disabled controls, manifests, raw
dumps, metric tables, first divergence, excluded steps, upstream tolerance
anchors, red-first output, mutations, fresh review, and operator reruns.

Classify structural and execution failures before interpreting metrics:
`ARTIFACT_MISMATCH`, `INSTRUMENTATION_FAIL`, `NONDETERMINISTIC`,
`STRUCTURE_MISMATCH`, `DTYPE_WIDTH_MISMATCH`, `CORRECTNESS_FAIL`, or
`NONFINITE`. When none applies, report descriptive metrics and the separate
operation/end-to-end acceptance results. Do not invent `WITHIN_DTYPE_ENVELOPE`
or `ORDERING_DRIFT` labels.

## Risks

- Probes can synchronize queues; disabled controls detect output perturbation.
- FP8 boundaries can change codes; compare dequantized values and record bytes.
- Different valid reduction orders can produce descriptive CPU/ROCm deltas.
- Backend fallback can change the diagnostic denominator; explicit backend
  logging exposes it.
- Greedy divergence makes later states incomparable; exclude them.
- The active source object can be read while the active runtime remains absent;
  never convert source availability into a gate pass.

## Stop conditions

- Stop before implementation until this spec receives a fresh `PASS`.
- Keep correctness `PENDING` until #2772 lands and the unchanged default gate
  passes on the implementation base.
- Stop active-pin capture until a runnable active-pin ROCm runtime is proved.
- Stop on a revision/hash/configuration mismatch, dump perturbation,
  nondeterminism, incomplete manifest, unexplained backend, provider decline,
  reference-tier hit, wrong board, missing GPU mutex, or divergent input prefix.
- Stop before changing tolerances or permanent sacred-gate scope after seeing
  results.
- File and assign a new issue before fixing any unexpected defect outside
  #2773.

## Git integration

Use one pull request for the committed spec and later implementation, following
the recorded repository default. Preserve rejected spec commit `7bc2546e9` in
history. The eventual pull request body must name row `BACKEND-ROCM`, link and
close #2773, and carry the required trailers. The spec implementer does not
push, open, or merge that pull request.

## Owed

- A maintainer owes the merge decision for reviewed PR #2856; #2772 remains
  pending until it lands.
- #2773 owes the runnable active-pin Qwen3.5-0.8B ROCm captures, even though
  #2794 supplies repository sync context.
- [#2923](https://github.com/mudler/vllm.cpp/issues/2923), owned by
  `BACKEND-ROCM`, owes routing Qwen3.5 paged attention through
  `dense_attn::AttnBlock`. This tracked exception records existing debt. It
  does not waive or satisfy the shared-seam requirement. Wiring needs its own
  reviewed spec and implementation, outside #2773's instrumentation scope.
- A fresh reviewer owes this repaired spec a verdict.
- A fresh implementer owes the capture options, state probes, tests, comparator,
  and provider checks after the prerequisites pass.
- A fresh implementation reviewer owes static review and every mutation.
- The operator owes the independent controlled and gfx1100 gates.
- The final implementation adds `## Outcome` with measured results, rejected
  alternatives, and reasons for defaults.
- Issue #332 owns the graph-enabled repetition.
