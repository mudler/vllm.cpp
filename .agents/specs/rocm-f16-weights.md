# Retain F16 GGUF weights on ROCm

Row: `BACKEND-ROCM-F16-WEIGHTS`

Issue: [#3092](https://github.com/mudler/vllm.cpp/issues/3092)

Parent: `BACKEND-ROCM`. Base: `6db4bef906859e864c82523c01107473f7dcca29`.

## Now

`ACTIVE`. The implementation retains F16 storage through the Qwen3.5 dense
registry and executes the three ordinary ROCm operations with explicit value
conversion. Focused ownership and physical gfx1100 checks pass. The complete
model and full-suite gates remain failing, as recorded under evidence below.
The parent row remains `ACTIVE`.

The change uses one pull request. Commit this spec before implementation. The
operator owns the GPU, reviews the returned evidence, and reruns the gates.

## Scope

Add ROCm execution for F16 weight storage through ordinary `vt::Matmul`,
`vt::MatmulBT`, and `vt::Embedding`. Make the default Qwen3.5 dense GGUF loader
reach these operations with retained F16 projection and embedding weights.
Preserve the resolved model dtype's arithmetic, including its weight rounding.
The first production witness is Qwen3.5 dense. This is a backend capability,
not a claim that every model loader supports retained F16 weights.

The retained F16 bytes serve two different contracts:

1. An unmarked primitive operand represents its exact F16 values.
2. A retained GGUF model weight represents the file values cast to the resolved
   model dtype before multiplication or gathering. For a BF16 model, this means
   F16 to BF16 rounding before arithmetic, even when an output buffer is F32.

Support F16 weights with F16, BF16, and F32 primitive activations and existing
BF16 or F32 outputs. Preserve the existing BF16 and F32 ordinary GEMM paths.
Keep the current output predicate. F16 model activations and F16 outputs are
outside this change, as are RMSNorm, attention, and a full F16 runtime.

Include the F16 embedding reader because the same default loader admits the
embedding table and a tied output head. Preserve both integer ID widths and
existing invalid-ID behavior. A kernel-only change that the default loader
cannot reach does not satisfy this row.

Exclude stacked expert F16 retention, grouped expert GEMM, block-quantized
GEMM, quantized embedding gather, new skinny kernels, fused alpha/beta GEMM,
and environment or CI policy changes. Preserve existing routing for these
paths. In particular, do not change the concurrent quantized GEMM change
[#2782](https://github.com/mudler/vllm.cpp/pull/2782).

## Upstream chain

Use vLLM `e126687a9a828d513c01a07cd69f025f27d63280`, recorded in
[the active pin](../upstream-sync.md). Source paths below refer to this revision.

- `vllm/model_executor/layers/linear.py:183` creates an unquantized parameter
  with `params_dtype`. `UnquantizedLinearMethod.apply:217` delegates to the
  selected GEMM implementation.
- `vllm/model_executor/layers/utils.py:559` dispatches unquantized GEMM to the
  ROCm implementation at line 563. `rocm_unquantized_gemm_impl:260` selects the
  production implementation. The gfx1x skinny eligibility starts at line 326,
  contiguous activation handling is at line 332, and `F.linear` is the ordinary
  fallback at line 346. The fake implementation at line 349 returns `x.dtype`.
- `vllm/model_executor/layers/vocab_parallel_embedding.py:35` defines
  `UnquantizedEmbeddingMethod`. Parameter creation at line 53 uses
  `params_dtype`, and `embedding:77` calls `F.embedding` on that parameter.
- `vllm/envs.py:1335` defines the default skinny-GEMM setting. Preserve it when
  preparing the oracle. A diagnostic configuration is not the denominator.
- `vllm/model_executor/model_loader/weight_utils.py:1227` defines
  `default_weight_loader`. Its `param.data.copy_` at line 1241 converts loaded
  values to the already allocated parameter dtype. The linear loader at
  `linear.py:572` and parameter loader at `parameter.py:176` carry the same
  parameter-storage contract.

For GGUF, read the complete first-party plugin chain at
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`, recorded in
[the plugin pin](../oracles/vllm-gguf-plugin.md):

1. `vllm_gguf_plugin/weight_utils.py:217`,
   `get_gguf_unquantized_params`, classifies F16 file tensors as unquantized.
   `gguf_quant_weights_iterator_multi:169` yields the native file tensor.
2. `vllm_gguf_plugin/loader.py:69`, `_get_unquantized_modules`, identifies their
   modules. `_prepare_adapter:176` and `load_model:203` carry that selection
   into the quantization configuration.
3. `vllm_gguf_plugin/quantization/config.py:71` returns
   `UnquantizedLinearMethod` at line 82 and `UnquantizedEmbeddingMethod` at
   line 91 for those modules.
4. `loader.py:217` enters `set_default_torch_dtype(model_config.dtype)` before
   model initialization and the weight load at line 229. The vLLM parameter
   loader then performs the conversion described above.

The plugin's `quantization/linear.py:35` unquantized fallback does not prove
that an ordinary F16 file tensor reaches a raw F16 multiplication. The module
selection and destination parameter dtype determine that path first.

The plugin currently records `gateable = no` and names
[#2624](https://github.com/mudler/vllm.cpp/issues/2624) for its Qwen3.8 workload.
This row owns the Qwen3.5 measurement it requires. Importing a wheel or
constructing a configuration does not close either model's gate. The operator
must run the actual active-pin model before accepting a denominator.

The ROCm 7.2.2 [rocBLAS GEMM extension contract](https://rocm.docs.amd.com/projects/rocBLAS/en/docs-7.2.2/reference/extension.html)
lists equal A and B input types. It does not establish BF16 by F16 mixed-input
support. Probe the installed library on gfx1100 and record the accepted A, B,
C, D, compute, and scalar types before choosing a direct call.

## Our baseline

At the pinned base, `DeviceKeepF16Supported` in
`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:170` excludes ROCm.
`GgufLoadPolicy::FromEnv` at line 360 therefore disables retained F16 storage.
`RouteGgufTensor:275` currently uses a general F16 role predicate that includes
stacked experts. Removing only the device exclusion would admit unsupported
expert and embedding consumers.

`MatmulKernelRocm` at `src/vt/rocm/rocm_matmul_hipblaslt.hip:491` and
`MatmulBTKernelRocm:542` accept BF16 pairs or F32 pairs. `ToBlasType` already
knows F16, but that helper does not make the registered entry points accept it.
`src/vt/ops.cpp:118` and line 151 already admit floating inputs and restrict
outputs to BF16 or F32. `EmbeddingKernelRocm` at
`src/vt/rocm/rocm_embedding.hip:109` rejects F16 tables.

The legacy keep-F16 work in [L6 and L7](gguf-keep-quant-loader.md) retains file
bytes and mmap ownership to avoid BF16 expansion copies. L7 also releases dead
repack source pages and prefaults borrowed weights. A persistent converted
weight shadow would erase part of that residency benefit. Transient conversion
must have measured memory costs and cannot be described as free retention.

The local Qwen3.5-4B Q4_K_M checkpoint inspected for this task has no F16
tensors. It is a regression asset and cannot witness the new capability.

## Port map

| Contract | Local surface | Change |
|---|---|---|
| Weight values distinct from storage | `OwnedTensor`, `vt::Tensor` | Optional weight-only value dtype |
| Explicit resolved dtype at admission | `GgufLoadPolicy`, `qwen3_5_dense.cpp` | Supply model dtype in the accepting registry loader |
| Retained ordinary weights and aliases | `qwen3_5_gguf_weights.cpp` | Preserve F16 bytes and value metadata |
| Residency and shared forwarding | `OwnedTensor::View`, both `ResidentWeight` helpers | Preserve operand metadata through every return |
| Primitive validation | `src/vt/ops.cpp` | Validate the new weight contract before dispatch |
| Ordinary GEMM | `rocm_matmul_hipblaslt.hip` | F16 inputs and faithful conversion fallback |
| Embedding | `rocm_embedding.hip` | F16 read followed by the declared value conversion |
| Coverage and registration | `tests/vt`, `tests/vllm`, test CMake lists | Production, dtype, lifetime, and negative tests |

The implementation may add a small ROCm helper file for conversion and scratch
ownership. New tests may use a dedicated `test_rocm_f16_weights` target. Keep
new implementation additive where practical. Do not edit CI workflows.

The implementer's authorized production files are:

- `include/vt/tensor.h`, `src/vt/tensor.cpp`, `include/vt/ops.h`, and
  `src/vt/ops.cpp` for the scoped operand contract.
- `include/vllm/model_executor/models/qwen3_5_weights.h` and
  `src/vllm/model_executor/models/qwen3_5_weights.cpp` for ownership metadata.
- `include/vllm/model_executor/model_loader/gguf_keep_quant.h` and its
  `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp` implementation.
- `include/vllm/model_executor/models/qwen3_5_gguf_weights.h` and
  `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp` for retained storage.
- `src/vllm/model_executor/models/qwen3_5_dense.cpp`,
  `src/vllm/model_executor/models/qwen3_5.cpp`, and
  `include/vllm/model_executor/models/dense_attn_block.h` for the accepting
  registry and existing shared weight consumers.
- `src/vt/rocm/rocm_matmul_hipblaslt.hip`,
  `src/vt/rocm/rocm_embedding.hip`, and a new scoped F16 conversion helper
  beneath `src/vt/rocm` when needed.
- `src/vt/rocm/rocm_backend.hip` for the F16 scratch ownership hooks described
  below. This extension does not change the backend's allocation device policy.
- `CMakeLists.txt`, `tests/CMakeLists.txt`, scoped existing or new tests under
  `tests/vt` and `tests/vllm`, and a new token fixture beneath
  `tests/fixtures/rocm_f16_weights` for registration and behavioral coverage.

The row spec, its issue record, and the public documents named below carry the
resulting records. Changes outside these paths require operator scope review.

### Carry an explicit weight contract

Choose optional `weight_value_dtype` metadata on `OwnedTensor` and
`vt::Tensor`. Unset means that a weight has its storage values. A set value
applies only to the B operand of ordinary GEMM or the table of embedding.
`Tensor::dtype`, `Bytes`, shape, and strides continue to describe storage.
The admitted marker is F16 storage with BF16 or F32 values on ROCm.

The alternative was new `MatmulAttr` and `EmbeddingAttr` arguments. Neither
registered function type currently has attributes. That alternative changes
every provider signature and repeats weight propagation at callers. The
selected operand field keeps the existing shared operation entry points and
limits interpretation to the three named consumers. It is not a global cast
rule for every tensor operation.

Validate the marker before dispatch. Reject it on activation, output, and ID
operands, on an unsupported storage/value pair, or on an unsupported provider.
Audit every reachable consumer of the newly retained weights. No consumer may
ignore a set marker. A missing consumer requires either a scoped extension
here or continued loader refusal for that role.

Carry metadata through copy, move, `View`, and `Slice`, through both
`ResidentWeight` implementations, through all resident return paths, and
through a tied embedding/head alias. Centralize descriptor propagation instead
of hand-copying it at each final return. Host-byte release must retain the
resident operand's metadata and storage dtype. Neither a retained mmap nor a
tied head may acquire a second permanent converted weight allocation.

### Admit only consumers that carry the contract

Extend `GgufLoadPolicy` to accept an explicitly resolved model dtype. The
Qwen3.5 dense registry loader supplies that value. On ROCm, default keep-F16
requires the value and an admitted ordinary matmul or embedding role.

An existing loader that does not supply the value keeps its prior BF16
expansion. Stacked experts and value-transformed tensors keep their prior
routing, regardless of the file's F16 type. Preserve `VT_GGUF_KEEP_F16=0`,
CPU-reference behavior, and CPU and CUDA defaults. An explicit unsupported
model dtype must not silently select BF16 semantics.

The production path is `include/vllm.h` to `LoadedEngine::FromModelDir`, the
GGUF `ModelSource`, `LoadQwen3_5DenseModel`, and the registered dense forward.
The forward reaches `MatmulF32D`, `MatmulBf16D`, ordinary shared attention and
MLP helpers, and `vt::Embedding`. Preserve shared `FusedChain`, `AttnBlock`,
`MlpGateUpMethodBase`, and `MergedGemmGroup` routing where already applicable.
Do not create a separate example-only or model-private F16 GEMM route.

### Preserve arithmetic on ROCm

For a marked BF16-value weight, round F16 to BF16 before multiplication. With
BF16 activations, use a temporary BF16 operand and the existing BF16 dispatch
so the original compute policy remains available. With F32 activations, round
the weight to BF16 first, then upcast exactly when a homogeneous F32 GEMM is
needed. Do not round or overflow F32 activations by casting them to F16.

For unmarked F16 inputs, preserve their exact values. Use direct hipBLAS GEMM
where the installed type contract allows it. Use explicit conversion for an
unsupported mixed pair. An F32 temporary needs a nearby reason that names the
unsupported pair. Preserve F32 accumulation and the requested BF16 or F32
output. A BF16 output cannot be assumed supported merely because F16 inputs
are supported. Record the exact direct and fallback combinations.

Preserve NN contiguity and BT row-strided activation support. Preserve shape,
rank, device, and stride rejection, empty M or N behavior, and K=0 output
zeroing. Respect queue ordering and current-device binding. Scope scratch by
actual device and stream identity. Reuse bounded temporary storage safely
across calls without a permanent cache per weight. An environment-selected
compute override must use matching scalar storage or refuse the unsupported
combination by name. Do not pass F32 alpha bytes as a 16-bit scalar.

Embedding converts each selected F16 value according to the marker before the
requested output conversion. Preserve repeated and boundary IDs, I32 and I64
IDs, empty inputs, and the current invalid-ID error and bounds behavior.

### Release scratch with its queue and graph owners

The operator approved this amendment on 9 September 2026, before its product
edits. The first implementation retained one scratch entry for every unique
queue ID until process exit. Geometric growth bounded an entry, but repeated
engine creation could accumulate an unbounded number of entries. The existing
`GrowOnlyStreamScratch` convention cannot supply the required lifecycle bound.

An active queue owns its current scratch allocation. A graph owns every scratch
allocation referenced by its captured F16 operations. Eager growth releases the
previous allocation when no graph owns it. Queue destruction releases its entry,
and graph destruction releases its captured allocation references. The last
owner releases the allocation after pending device work completes. No permanent
weight shadow or process-lifetime list of destroyed queues is allowed.

Wire this ownership through the existing ROCm backend lifecycle. Cover queue
destruction, the single-graph capture and replay slot, the owned graph-handle
API, and graph deduplication handles. A captured graph must remain valid after
its source queue is destroyed and when replayed on another live queue. Replacing
the single-graph slot releases the prior slot's scratch ownership. Do not change
the public graph handle contract, resource-device policy, or another scratch pool.

Bind allocation release to the allocation's device and preserve the caller's
ambient device. Define lock ordering so stats, enqueue, growth, queue teardown,
and graph teardown cannot invert locks. Do not reclaim a slab solely because
its capture ended, its source queue ended, or a newer eager call uses a larger
slab. Active graph ownership is independent of queue ownership.

The focused lifecycle test creates a queue, executes mixed GEMM, destroys the
queue, and verifies that retained bytes return to the starting value. Repeat
the cycle to expose cumulative retention. A second test captures, grows eagerly,
destroys the source queue, and replays the graph on another queue. Verify both
correct output while the graph remains active and release after its last owner.
Exercise the single-graph and graph-handle APIs, including deduplication.
Record the red result on the initial implementation before repairing ownership.

## Tests to port

Port the applicable ordinary fallback case from
`tests/model_executor/layers/test_rocm_unquantized_gemm.py:152` at the active
vLLM pin. Preserve its F16 inputs, activation shape `[6,64]`, weight shape
`[128,64]`, and `atol=rtol=1e-3`. Document the C++ harness adaptation. The local
seam returns BF16 or F32. For this port, request F32 output, round that result
to F16, and compare it with the upstream F16 result at the original tolerance.
Separately check the unrounded result against F32 accumulation over exactly
widened F16 inputs. Add BF16 as an explicit local regression parameter with
the expected final BF16 rounding, rather than attributing it to this test.
The contiguous-activation and skinny-selection cases at lines 20, 45, 71, 98,
and 135 describe existing specialized dispatch. Keep them as regression
obligations when that dispatch is changed. Do not claim to port new skinny
kernels in this row.

Add tests that expose these guarantees through existing production seams:

- Default Qwen3.5 dense loading of a synthetic F16 GGUF keeps eligible F16
  bytes and metadata, including tied and explicit heads. Enter through the
  registry or public loader. Do not only construct `OwnedTensor` by hand.
- The loaded model's registered forward invokes the retained embedding and
  ordinary projection paths. Use values such as `1 + 2^-10` that distinguish
  raw F16 arithmetic from BF16-rounded weight arithmetic.
- Exercise copy and move, mapped and copied storage, tied aliases, shape views,
  slices, each resident return path, and post-upload host release.
- Cover NN and BT with M=1, M=4, M=6, a larger non-tile multiple, odd N and K,
  row-strided BT input, BF16 and F32 outputs, and all newly admitted input pairs.
- Use F32 activations outside F16's range and values too small for F16. Compare
  against explicit F32 accumulation after the declared weight conversion.
- Cover M=0, N=0, K=0, invalid ranks, shapes, strides, devices, dtype markers,
  and providers. Re-run existing BF16 and F32 cases.
- Cover F16 embedding with both ID widths, repeated IDs, edge rows, invalid
  negative and vocabulary-sized IDs, empty IDs, and BF16 and F32 outputs.
- Verify `KEEP_F16=0`, CPU-reference, unwired ROCm registry loaders, stacked
  experts, and existing CPU and CUDA routing remain correctly selected.

The smallest test must fail on the baseline for the intended production
admission or execution reason. Retain the red command and output before
implementation, then focused green and the full gate.

## Gates

| ID | Requirement | Current result |
|---|---|---|
| F16-G1 | Spec committed before implementation, issue and row agree | Satisfied by this spec commit |
| F16-G2 | Focused red, focused green, and registered CPU/HIP tests | FAILING baseline CPU/HIP suites (#3070, #3102, #3105) and missing native `AttnGateSplit` (#3106); three repaired guards pass |
| F16-G3 | Default public-load and registered-forward reachability | FAILING explicit-head forward at missing native ROCm `AttnGateSplit` (#3106); public retained-weight execution observed |
| F16-G4 | Identical-artifact active-pin oracle and exact token gate | FAILING arithmetic comparison and D1 repeatability; identical-GGUF plugin run remains PENDING |
| F16-G5 | Same-tool executed dtype and dispatch traces | PENDING identical-GGUF primary execution; native/materialized traces captured with existing dtype/policy differences retained |
| F16-G6 | Same-binary A/B and oracle speed, latency, and memory | PENDING F16-G4 |
| F16-G7 | Fresh immutable-head mutation review and operator rerun | PENDING reviewed head |
| F16-G8 | Full repository preflight, no skipped applicable gates | PENDING final operator head; repair preflight and explicit build-check dispositions recorded below |

### Run the model and compare values

Build a real F16 GGUF fixture from the existing Qwen3.5-0.8B asset only after
recording the exact upstream checkpoint repo, revision, file names, sizes, and
hashes. Use the recorded llama.cpp converter pin, not a floating checkout.
The converter revision is `10bf611e533d81f739128304991c5e133c6aebd8`, tag
`b10451`, from [the llama.cpp oracle record](../oracles/llama-cpp.md).
The inspected source is `Qwen/Qwen3.5-0.8B` at
`2fc06364715b967f1860aea9cf38778875588b17`. Its shard
`model.safetensors-00001-of-00001.safetensors` contains 1,746,942,600 bytes and
has SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
All 13 cached files' measured hashes match their recorded metadata, and their
metadata names that same revision. Retain that complete source manifest with
the conversion evidence, including the config and tokenizer files.
Record the converter command, revision, output hash, tensor dtype histogram,
and complete source-to-GGUF tensor mapping with names, shapes, and conversion
rules. Validate the conversion instead of assuming BF16 and F16 interchange.
The operator supplies asset and launcher paths. Repository prose is not a
source of host configuration defaults.

Prefer the active vLLM plus pinned GGUF plugin on the identical GGUF. If that
plugin cannot run this model, keep this row's model gate pending under #3092
and retain the plugin registry's existing #2624 debt. A validated HF
materialization of the exact GGUF tensors may provide the primary arithmetic
comparison only when its complete mapping and dtype conversion are proven.
Do not silently compare a differently rounded original HF checkpoint.
The registered secondary oracle can answer correctness for an ungateable GGUF
path under its recorded policy. It does not override vLLM's model-dtype rule.

Commit `tests/fixtures/rocm_f16_weights/token-workloads.json` before the red
run. Its input arrays follow this fixed definition: request index `r` receives
IDs `1 + 128*r + i` for `i` from 0 through `P-1`. Materialize those arrays in
the fixture and verify every ID is in the model vocabulary. These are explicit
token prompts. Neither arm tokenizes, adds a chat template, or adds BOS.

| Workload | Prompt P | Generated tokens per request | Request batch | Concurrency | Request indices |
|---|---|---|---|---|---|
| F16-D1 | 1 | 32 | 1 | 1 | 0 |
| F16-P16 | 16 | 32 | 1 | 1 | 0 |
| F16-P128 | 128 | 64 | 1 | 1 | 0 |
| F16-C4 | 16 | 32 | 4 | 4 | 0, 1, 2, 3 |

Set temperature to 0, top-p to 1, repetition penalty to 1, presence and
frequency penalties to 0, and seed to 0. Disable stop strings and stop-token
lists. Set `ignore_eos=true` and `min_tokens=max_tokens` to the row's generation
count. Both arms must emit 256 generated IDs across the four workloads.
Capture each request separately and compare complete arrays for exact equality.
Run the matrix twice per arm to establish the oracle's repeatability.

Use `vllm_complete_tokens` for our pre-tokenized public path and vLLM's
`TokensPrompt` on the oracle. Use a HIP-only build with the C ABI's automatic
integer device selection, `device=0`. Assert the resolved device is ROCm and the new
operations use native ROCm providers. The public device field does not accept
the string `"rocm"`, and a successful CPU fallback is not this witness.
For F16-C4, submit all four requests together,
set the sequence capacity to 4, and retain the actual scheduled batch trace.
Do not compare concurrent timing if the effective batching differs. The other
three workloads use sequence capacity 1. The generated real checkpoint and
the small production fixture cover tied and explicit heads between them.

Use the production oracle configuration without `--enforce-eager`. A known
baseline mismatch needs its own evidence and owner. It cannot waive a mismatch
caused by this change.

### Trace the actual operations

Use the same tracing tool on both arms. Record the original F16 storage and
the values' resolved dtype separately. Identify every cast, GEMM, and gather
that executes. Record A, B, C, and D dtypes, output dtype, compute type, scalar
type, entry point, algorithm policy, and resolved template types. Dump generated
kernels before declaring any upstream lever unreachable.

A conversion-plus-BF16 GEMM is an explicit storage adaptation, not a claim of
native F16 invocation parity. Prove that the raw primitive F16 path executes,
and that the marked model path matches the oracle's rounded values. Annotate
every newly introduced F32 model-path buffer with its reason.

### Measure retention and performance

After F16-G4 passes, run an idle-host same-binary A/B with keep-F16 enabled and
disabled. Record at least three alternating legs and retain every raw log.
Measure load time, prefill throughput, decode throughput, end-to-end latency,
peak RSS, peak device memory, persistent weight bytes, and transient scratch
high water. Trace conversion traffic, especially a vocabulary-sized head.

Report values and ratios against the production oracle for every applicable
axis. The parity floor is throughput ratio at least 1.00 and latency and memory
ratios at most 1.00, subject only to the owning row's explicit existing gate.
An axis below its floor remains an open gap with the next traceable hypothesis.
Do not accept a performance result before correctness, hide conversion scratch
inside a weight-memory claim, or call a regression an architectural ceiling.
Preserved resident F16 bytes alone do not prove a reduction in peak memory.

An accepted paired measurement establishes the effect of this storage change
only after its identical-workload and correctness requirements pass. It does
not close the parent backend's existing overall performance gaps. Report both
the same-binary delta and the absolute oracle ratios. Preserve each existing
parent gap's owner and floor. A positive local delta cannot waive an absolute
floor, and token equality cannot waive a dtype or invocation mismatch.

### Verify the immutable head

Run `scripts/agent-preflight.sh` before edits and before the spec commit. On the
implementation head, run the focused registered CPU and ROCm suites, applicable
full CTest suites, and preflight. Record exact configure flags, compiler and HIP
versions, revisions, commands, exit statuses, logs, and omitted gates.

A fresh reviewer mutates each claimed guarantee in a scratch copy. At minimum,
disable loader admission, delete the production forward call, drop metadata
from each residency path, omit BF16 weight rounding, cast F32 activations to
F16, break K=0 zeroing, and misread one ID width. Each corresponding focused
gate must fail. Restore the immutable tree byte-for-byte after every mutation.
The operator reruns the row's gate on the reviewed SHA.

## Dependencies

The existing ROCm backend, shared ordinary operation providers, GGUF loader,
and Qwen3.5 dense registration are available at the base. There is no dependency
on the quantized GEMM pull request. GPU work requires the operator's recorded
host, toolchain, active oracle identity, and exclusive device ownership.
Use the configured fleet lease where applicable and the configured mutex on a
non-fleet device. This spec authoring task runs no GPU work.

## Evidence from 9 September 2026

Evidence files are retained under the implementation worktree's ignored
`build-rocm-f16-evidence` directory. Each operator receipt records its command,
environment, executable hash, mutex boundary, and exit status. Source snapshots
preserve harness revisions whose paths were subsequently rebuilt. Measurements
below do not establish a performance result or close an unresolved gate.

### Red and component checks

The installed `hipblasGemmEx` probe tested all nine F16, BF16, and F32 input
pairs at M=6, N=128, K=64. It used F32 compute, F32 scalars, and the default
algorithm. `blas-probe-run.log` records the result for both requested output types.

| A and B | BF16 output | F32 output |
|---|---|---|
| F16 and F16 | Not supported, status 7 | Success, value 64 |
| BF16 and BF16 | Success, value 64 | Success, value 64 |
| F32 and F32 | Not supported, status 7 | Success, value 64 |
| Each unequal input pair | Not supported, status 7 | Not supported, status 7 |

The implementation therefore uses homogeneous F16 directly only with F32 output.
Marked BF16-value weights convert before the existing BF16 dispatch. Other mixed
pairs widen exactly to F32, after any declared BF16 weight rounding. A requested
BF16 output uses a temporary F32 result when the homogeneous input type requires
that adaptation. Ordinary `VT_ROCM_GEMM_COMPUTE=16f` is refused by name, because
these calls supply F32 scalars. The `16bf` override requires effective BF16 inputs.

Commit `760c8dd7e27781430b909c3679790c52fe332182` added the production admission
test and fixed token fixture before implementation. Its focused admission test
failed because the registry returned BF16 where retained F16 was required.
`production-admission-focused-red.log` preserves that original failure. The first
red invocation omitted the authorized external `TMPDIR`. Its temporary fixture
was removed by the test destructor. Later invocations use the external directory.

A clean rebuild of that commit reproduced the same failure with a hashed binary.
`production-admission-red-reproduced-v1-command.json` records the command and
SHA256 `a3469163bb8f27c9274b8e81e9bad7fa8ffef5f1e19732e9c236ae69f5b9f408`.
The implementation's focused admission test passed 10 assertions. Its complete
CPU GGUF suite passed 56 cases and 10,417 assertions. The ownership and dispatch
contract passed 3 cases and 111 assertions.

The operator ran `test_rocm_f16_weights` on gfx1100. The five-case revision
passed 27,680 assertions, including graph replay after scratch growth and
interleaved streams. Its binary SHA256 is
`d730e89f49fb1bd35bb6ef59e7235c63e79631fc22456c0900c1f69ae3fa7d65`.
`hip-f16-focused-streams-operator-receipt.json` records that run.

`test_rocm_f16_device` passed NN, BT, and embedding on each of two visible
gfx1100 devices. The test allocates and verifies each operand on the intended
device, then changes the ambient device before entering the ordinary operation.
Each operation leaves its queue's device current and returns the required values.
`hip-f16-two-device-v2-operator-receipt.json` records binary SHA256
`67e7758e66f716bd5412fdda8bbd38d7cdbbe6b2d26a749d6d3154f18cbf9da6`.
This scope excludes the existing backend allocation issue #3100.

The pinned upstream ordinary fallback ran on the operator's gfx1100 device.
`oracle-primitive-v1-operator-receipt.json` records the actual F16 `[6,64]`
activation, `[128,64]` weight, and `[6,128]` result. The raw F16 arrays have
combined SHA256
`dbb7f33c2a2c163d06f2129aa31cef108321d5142bbaf37eec3ff194548f6ca8`.
The committed `upstream-fallback.json` retains those arrays. Its capture script
preserves the upstream fallback-selection mocks and `atol=rtol=1e-3`.
Seed 0 makes the original random fixture reproducible. The C++ adaptation
compares NN and BT after the spec's required output rounding. An additional
comparison checks unrounded F32 accumulation. The operator's six-case run
passed 27,692 assertions, recorded in `hip-f16-upstream-fixture-v3-operator-receipt.json`.
Its binary SHA256 is
`a7471e7928698c20f96d739446942ba78e79713646668eb76422cfabd603c108`.

### Scratch lifetime and negative checks

The original scratch registry retained each destroyed queue's allocation. The
operator's four-queue red test retained 8,192 bytes in four entries. The graph
handle red test retained 33,024 bytes after graph destruction, and the single
slot red test accumulated four entries. Numeric replay assertions still passed.
The original binaries, logs, and receipts remain under `snapshots/lifetime-red-v1`.

Commit `e076a533e20935ebfcf829005ce66e59f1d34292` amended the scope before the
lifetime repair. Queue and graph owners now share each referenced allocation.
Eager growth releases an unreferenced old slab. Destroying a queue removes its
entry; captured graphs keep their own allocations until their final owner ends.
The last owner binds the allocation device, calls `hipFree`, and restores the
ambient device. The pool releases owners outside its registry lock.

The operator ran the nine-case lifetime revision: 28,151 assertions passed with
zero skips. Its frozen binary SHA256 is
`3b2d801a71e754f1bfc79e1b6cd9dd2cecaf57b699922b8272d7b6cfb6a516a3`.
A separate process with graph deduplication enabled passed its case's 97
assertions; its runtime reports two graphs sharing one executable. The graph
case retains a shared slab through two handles, destroys the source queue,
replays on another queue, and releases the allocation only after both handles
end. The single-slot case replaces four graphs without cumulative retention.
Receipts reside beside `lifetime-green-v1/manifest.json`.

A test-only linker observer calls the real HIP allocator and records successful
frees with allocation generation IDs. The assertions verify physical release
independently of the product's byte counters. The two-device test verifies
release restores the caller's ambient device. Its strengthened observer also
records actual pointer provenance and the device current at the real free call;
that revision's six-operation control passed under `lifetime-green-v2`.
The final ten-case control passed 28,234 assertions. It includes an unwarmed
capture-growth refusal that leaves capture usable for a subsequent operation.
Its binary SHA256 is
`44c016115552a79426ec39771862fddec8e8a085f9e5f07ef89f03e4a8bedb8f`.

The CPU mutation runs remove twelve distinct ownership, residency, validation,
provider, and production-admission guarantees. Every corrected mutation fails
at runtime, with byte-identical restoration recorded under `mutations-cpu-v1`
and `mutations-cpu-v2`. The first header mutation experiment selected the
original weak inline definition at link time and was not a valid mutation.
The corrected ordering records the chosen object in its link map and includes
a passing unmodified control. No compiler or test tolerance was weakened.

HIP mutation recipes under `mutations-hip-v1`, `mutations-hip-v2`, and
`mutations-hip-v3` separately remove physical free, weight rounding, F32
activation precision, each K=0 zero fill, embedding ID width, operation device
binding, and release device binding or restoration. Each recipe records the
source, exact compile and link commands, binary hash, and byte-exact restoration.
The operator detected all 18 recorded negative invocations, including separate
graph-deduplication arms, in `lifetime-and-numeric-mutations-operator-results.json`.
Each failed at the intended numeric, physical-ownership, or device guarantee.
The capture-growth mutant first failed the expected-error assertion and then
aborted during cleanup; the abort alone is not its mutation evidence.

The physical-free-only mutation ran separately. It removed only `hipFree` while
leaving owner removal and byte counters intact. The three-case run passed 449
assertions and failed ten physical-free assertions. Its log SHA256 is
`ce325e2f7ddd166c3229689d025d7b0eee33a3e82e8c97814b0bacece06fc6a1`,
retained with the operator receipt under `mutations-hip-v1/omit-physical-free`.
These are implementer mutation checks executed by the operator. Fresh review of
the immutable implementation remains F16-G7.

### Model artifacts and the primary run

The stock converter produced `Qwen3.5-0.8B-F16.gguf`, 1,557,662,528 bytes,
with SHA256
`758b5299b027120c3608c43777a89257724d46e26eefac2a8aaad512be15b53f`.
The file contains 335 tensors, comprising 195 F16 tensors and 140 F32 tensors.
`qwen35-08b-source-manifest.json` pins all 13 source files. Conversion validation
compares every emitted tensor's name, shape, storage dtype, and bytes with the
stock converter. Its mapping accounts for 320 text tensors, 15 MTP tensors,
and 153 source vision tensors omitted from this text GGUF.

The first-party plugin does not run this text-only GGUF at the recorded pin.
The complete HF configuration requires the missing vision projector. The exact
text configuration instead reaches a conditional-generation registration that
requires `Qwen3_5Config`, but receives `Qwen3_5TextConfig`. The failed actual
runs remain in `oracle-tokens-cap1-v2-operator-receipt.json` and
`oracle-gguf-text-v4-cap1-operator-receipt.json`. No plugin code, architecture
override after plugin selection, or substitute projector was introduced.

The spec's permitted arithmetic materialization uses the pinned plugin's exact
name mapper and weight transformations. The operator independently checked all
320 emitted tensors against the GGUF and adapter. The plugin excludes the
15 MTP tensors from this text path. `operator-verify-materialization-v1-receipt.json`
records the successful check. `materialization-v1.json` retains every source and
destination name, dtype, shape, transformation, and content hash. The resulting
`model.safetensors` SHA256 is
`b27fbf8cb5d35406b4fce084bdda1ec10de8fb5e3d48f647edc1ebf325fa740e`.
Its exact text configuration, with the native causal-LM architecture, has SHA256
`9530fc375d6751919be82999e46ff9c5d8bc9c3e5d6d678bf2e9d7d220347600`.

`oracle-materialized-v5-cap1-operator-receipt.json` records the actual active-pin
BF16 production run, with eager enforcement disabled. Both oracle repeats of
F16-P16 match the native 32 generated IDs exactly. Both oracle repeats of
F16-P128 are stable, but differ from native at 31 of 64 positions. F16-D1
differs between oracle repeats at 31 of 32 positions. Its repeatability assertion
fails. These observations establish neither a distributional gate nor token parity.

The native public API executes the retained embedding table and marked ordinary
GEMMs. A test-only linker wrapper observes actual native provider calls and invokes
their original implementations. The first observation harness attempted duplicate
provider registration, which preserves the first registration and cannot observe
the calls. Its failed result and source remain under `snapshots/observer-v1`.
The corrected wrapper verifies its selected function before measurement.

The corrected public run emits every capacity-one request but fails D1
repeatability. Its KEEP_F16=0 comparison uses the same binary and matches every
token in all six runs. D1 differs between native repeats at 28 of 32 positions.
F16-P16 and F16-P128 remain stable. `public-cap1-keep-ab-v2-comparison.json`
retains this comparison. KEEP_F16=0 alone does not establish a pre-existing bug,
because that binary still contains the implementation.

An untouched HIP library was built from clean commit `760c8dd7e27781430b909c3679790c52fe332182`.
A separate diagnostic harness uses only baseline-compatible observation fields.
Its adaptation removes unavailable new metadata observations and preserves the
public request parameters, repetitions, assertions, and production calls.
`build-hip-baseline-harness-v3-command.json` records library SHA256
`92f77b07e5cdfc5870117b1098a1b10ae2a09101e6d76fb4d81c90ab2a849c1b`
and binary SHA256
`45f7ffee0a0653187adbcf38a71b6aa203541c4ee811ab93c212c4e8e4eb466f`.
`public-baseline-cap1-v3-operator-receipt.json` records the unchanged baseline's
same D1 repeatability failure. All six baseline token arrays equal both feature
arms, as recorded in `public-baseline-v3-vs-keep0-v2-operator.json`. The feature's
retained-storage choice therefore does not cause these observed differences.

Four fresh-process active-pin diagnostics each run one D1 request with the same
production configuration. Two record scheduler observations; two omit that
instrumentation. All four emit the same 32 IDs, `[198, 1]` repeated 16 times.
`oracle-diagnostic-v6-four-fresh-operator-summary.json` retains this control.
It does not replace or relax the committed repeated-engine workload.

### Capacity-four and current memory observations

The original F16-C4 matrix completed twice in each arm: native KEEP_F16=0,
native KEEP_F16=1, untouched baseline, and the active-pin materialized primary.
Every arm was repeatable. All eight requests and 256 generated IDs matched
exactly. `capacity-four-baseline-comparison-operator.json` and
`capacity-four-primary-comparison-operator.json` retain the operator's independent
comparisons. The current public observer binary SHA256 is
`2e4fda0905334765389653f6781ed79cf0b2c7abc381bc1405b2e2c914dd55a7`;
`capacity-four-v1/manifest.json` pins it beside the unchanged baseline and fixture.

The actual schedules differ. For example, KEEP_F16=1 begins its repeats with
48 then 19 tokens, and 16 then 49 tokens; the primary begins with all 64 prompt
tokens. Each reaches batches of four decode tokens. Exact tokens therefore do
not establish matching invocation counts or a throughput comparison.

The native retained-weight arm observed 12,342 marked ordinary GEMMs across
F16-C4. Its scratch capacity and retained allocation were each 536,870,912 bytes
(512 MiB), with a 508,559,360-byte high-water request and one active queue. The
KEEP_F16=0 control reported zero marked GEMMs and zero F16 scratch. Both arms
reported 2,013,340,416 uploaded device bytes and zero instrumented host-copy
bytes. These counters do not measure total RSS or establish a peak-memory win.
The extra scratch must remain part of any later memory comparison.

The current capacity-one observer preserves counters before its unchanged D1
repeatability failure. It observed 47,872 marked GEMMs and the same 512 MiB
scratch allocation and high-water request. The exact native and primary logs,
commands, and exits remain in
`capacity-four-and-current-observer-operator-results.json`. F16-C4 passes its
arithmetic subcase; the D1/P128 and identical-GGUF failures still hold F16-G4 open.
No speed, latency, memory ratio, or invocation-parity result is accepted.

### Executed dtype and dispatch traces

The operator traced the unchanged capacity-four requests and both repeats with
rocprofv3 1.3.5, revision `6b0e43f341195e203754e08f850e437ff2fc09f9`.
Both arms use the same hashed profiler prefix, copied from existing files and
mounted read-only. The prefix contains no HIP, HSA, hipRTC, or BLAS runtime.
`g5-traces-v1/manifest.json` records its complete file and symlink set.
The application runtimes retain their existing versions and libraries.

The first oracle run is insufficient for API coverage: its forked worker inherits
profiler PID 1, and the parent overwrites the worker API and JSON outputs.
All first-run files remain preserved. The bounded repair uses vLLM's supported
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, per-process filenames, and CSV-only output.
This is a profiling process-launch adaptation, not a performance denominator.
The observer adds real PIDs; model options, graph mode, sampling, every input ID,
and both repeats stay unchanged. The worker finishes writing before shutdown.

The repaired oracle worker is PID 227. Its 248,920 runtime API events cover all
35,343 kernel and 239 memory-copy correlation IDs, with zero missing joins.
Parent and helper processes have separate compiler-only files. The implementer
and operator independently verify that all 256 generated IDs match the traced
native KEEP0/1 arms and the earlier primary run. The operator audit is
`g5-traces-v2/oracle/operator-trace-token-audit.json`, SHA256
`8c08d00475c161d1311dbbcfc953952e09370b3352bb89e96d753b9407b67abc`.

The raw primitive trace executes 18 `rocblas_gemm_ex` calls with F16 A/B,
F32 C/D, and F32 compute. The marked model adds 12,155 conversion launches and
65 `EmbeddingKernel<unsigned short, __hip_bfloat16, int>` launches.
The expanded arm instead uses BF16 embedding storage. Both native model arms
execute the same BLAS dtype groups: 306 BF16-output calls and 5,460 F32-output
calls, each with BF16 A/B and F32 compute. These calls use the standard algorithm,
solution index zero, no flags, and `atomics_not_allowed`.

The materialized primary executes 799 traced BLAS calls with BF16 A/B/C/D,
F32 compute, the standard algorithm, solution index zero, no flags, and
`atomics_allowed`. Both arms execute BF16 `wvSplitK` specializations, with
different resolved templates and invocation counts. The audit retains every
full kernel name, cast/gather name, BLAS dtype tuple, and call count in
`g5-traces-v1/initial-dtype-dispatch-audit.json` and
`g5-traces-v2/dtype-dispatch-audit.json`. F32 alpha/beta types follow the executed
compute type under the rocBLAS ABI (`rocblas-functions.h:21706`); they are not a
separate scalar field emitted by rocprof. The trace does not erase the existing
F32-output, atomics-policy, scheduling, or template differences. No invocation
parity or timing result is accepted. The identical-GGUF plugin trace remains
pending that primary path's execution; the materialized adaptation is explicit.

### Preserve existing view descriptors

An implementer check found that the shared metadata helper rejected empty and
zero-extent `OwnedTensor::View()` descriptors. The isolated CPU regression passes
all three cases on pristine `760c8dd7e` and fails all three on the implementation
before repair. `empty-view-v1/results.json` pins both binaries, libraries, source,
and exact commands. The registered regression fails for the same rank check.

The correction permits rank-zero and zero-extent descriptors. It preserves the
existing distinction: a default `OwnedTensor::Numel()` is zero, while its rank-zero
`vt::Tensor` view has the empty product of one. Shape, strides, byte count, data,
device, and dtype match the baseline. The focused CPU gates pass after repair.
The metadata test separately carries `repacked`, `q8_0_aligned`, and
`elem_kn_repacked` through ownership, borrowing, shape views, and valid slices.
Six scratch mutations remove each flag from `OwnedTensor` and `Tensor` views;
every mutant fails its corresponding assertion. `mutations-cpu-v3/summary.json`
records exit one for all six and byte-exact restoration of source and archive.

The operator reran the corrected HIP metadata, primitives/lifetimes, graph dedup,
and two-device targets: all four pass. Input and output hash checks preserve the
same sources, archive, and executables. The receipt is
`empty-view-v1/hip-focused-operator-receipt.json`; log SHA256 is
`313eaaf053de3f8a97829d244c0943d94a9ceefe88928fef3fd4ce74d42fd615`.

### Full suites

The complete CPU and HIP builds passed with at most four compile jobs.
The full CPU CTest run registered 715 tests. It passed 703, skipped 11,
and failed `test_qwen3_paged_engine`. That test failed two assertions against
the unchanged Qwen3-0.6B token anchor. For `The capital of France is`, generated
position 5 is 15344 where the fixture expects 9625. The KV-boundary case fails
that same prompt before its separate boundary exercise.

A clean baseline build and an independent operator rerun reproduce both failures.
`cpu-paged-baseline-v1-command.json` and its operator receipt record binary SHA256
`46c8a80062219b0135b25ef0668de9b79232fd8f1cf6c173fe4194d8fa6aaf9d`.
Issue #3102 owns the diagnosis under `MODEL-TEXT-qwen3-qwen3-for-causal-lm`.
The anchor, assertion, and model row lifecycle remain unchanged.

The post-correction full CPU rerun produces the same 703 passes, 11 skips,
and one failing target out of 715. `empty-view-v1/cpu-full-command.json`
records its rebuilt archive, exact serial command, exit eight, and hashed log.
The metadata correction introduces no additional CPU suite failure.

The 11 CPU skips are retained in `ctest-cpu-full-v1.log`. They require an
unavailable accelerator or model asset. The operator completed the pre-correction
HIP suite: 691 passed, 12 skipped, and 20 failed out of 723. Its protected inputs
remained unchanged; `full-hip-ctest-v1/operator-receipt.json` retains log SHA256
`ca572d1ed024bf9e53ded34ac377567e5305a2b9aaee6fbee3dcc1f6b4569dc8`.
The new explicit-head case still fails on the separately owned #3098 dependency.
The operator reran all 19 remaining targets against a pristine HIP build under
identical device visibility and serial scheduling. Every failed test name,
normalized error line, and doctest case/assertion summary matches the feature
run. `full-hip-baseline-v1/operator-comparison.json` records the 19/19 match;
the baseline log SHA256 is
`abd5ec9157b4e3d553782d2251348dc23eeb70dc4374586359d21fd271a627cb`.
The existing #3070 owns the backend suite groups; this gfx1100 run does not
satisfy its Strix-specific gate. The HIP Qwen3 anchor failure is distinct from
the CPU failure: `Once upon a time,` generates 264 at position 9 where the
ROCm anchor expects 279. Both standard and boundary cases reproduce it, with
72 of 74 assertions passing. Issue #3105 owns this separate gfx1100 diagnosis.
Fresh immutable-head review remains pending. The staged preflight finished with exit zero, no failed gates, and all 677 in-scope
translation units compiled. Its five argument-required skips are Arm ISA, CPU
ISA, CUDA fat-gencode, PR classification, and Triton AOT multiarch. The explicit
x86 ISA audit passed against the actual CPU build. The attempted Arm audit of
that x86 database fails its expected Arm flags and supplies no Arm build evidence.
PR classification still requires the immutable implementation commit. No skipped
applicable gate is counted as satisfied.

`preflight-lifetime-v2-command.sh`, its log, and its exit file preserve the
complete invocation. The final operator rerun must cover the reviewed head.

The final post-correction staged preflight also exits zero, with no failing
gates and 677 of 677 translation units compiled in 268.8 seconds. It retains
the same five argument-required skips and is not counted as a fully green
preflight. `preflight-empty-view-v1-command.sh`, its exit file, and log SHA256
`2b8d47439eed4c800961f58c651cfdc03c02655a02aadf959e2dd6142c33265c` preserve this run.
The final scoped record and authored `Now` checks pass after the trace evidence
and owned baseline issues are added. Fresh review and the operator's final
immutable-head gate remain required.
The three source citations shifted by this implementation were repaired in their
owning engine and quantization rows. Both implementer and operator checks prove
that every unrelated row remains byte-identical; the record checker returns to
its unchanged 28-stale/5-broken baseline. No checker or baseline was changed.

### Review coverage repair and prerequisite integration

Fresh review of `822ee510cc13fb9e10657da93bb3b941c0700b55` found three
coverage gaps. It found no new product defect in those three guarantees.
The repair enters unsupported model dtypes through `ModelRegistry::Load`,
enters unsupported embedding providers through `vt::Embedding`, and tests
ordinary NN and BT compute overrides independently of graph capture.

The registry cases accept `bfloat16`, `bf16`, and an empty dtype. They reject
explicit `float16` and `float32` with the registered loader's full diagnostic.
Removing only that registry guard fails both new refusal assertions.
The embedding case installs a fake provider that cannot consume the weight
marker. Removing only its provider guard fails the refusal and callback-count
assertions. The provider callback must remain uncalled.
Both cases use scoped environment or provider restoration. An independent
same-process driver covers originally absent, empty, and nonempty environment
values and both original provider states. All five restoration cases pass.

The NN and BT cases each reject `VT_ROCM_GEMM_COMPUTE=16f` and `16bf` by name.
Their mutation removes only the NN guard and leaves BT unchanged. Under the
GPU mutex, the operator runs both on physical gfx1100. Only the two NN refusal
assertions fail in the mutant. The unchanged control passes six assertions.
`operator-focused/nn-compute-refusal-mutant-receipt.json` records log SHA256
`252a9909abbb07e6cab726acce3293964302c4541ec93316575848d63def1414`.
The corresponding control log SHA256 is
`f00e896bebad60db7e12e37c13559f0ea37b9e7be482b8060438606ed1edb716`.
The operator reruns both CPU mutants and controls and verifies every frozen
source and input before and after all six runs.

This branch integrates the reviewed prerequisite
`6a7bcb77637e66df34429208e3a4055e0945a875`, tracked by #3098 and PR #3101.
The integration preserves this branch's F16 `ResidentWeight` metadata changes.
Its `qwen3_5.cpp` bytes match an independent three-way application of the
prerequisite delta from `6db4bef906859e864c82523c01107473f7dcca29`.
All five prerequisite test files match the reviewed prerequisite exactly.
Nine shared keyed records, containing 922 rows, remain byte-identical.
The operator independently verifies these comparisons.

The complete HIP GGUF target now passes 57 of 58 cases and all 10,455
completed assertions. Its remaining case reaches a further existing ROCm refusal: no native
`AttnGateSplit` provider. The explicit-head fixture retains its real gated
attention and reaches `FullAttnBlockPaged` at `qwen3_5.cpp:5813`. It cannot finish that forward.
`operator-focused/gguf-keep-quant-control-receipt.json` records exit one and
log SHA256 `f4d682bdfa800b80c7e25d75b51c31f00a0d3168579534a3e25fff2fda84752a`.
F16-G3 remains failing. The fixture and its expected outcome are unchanged.
Issue #3106 owns this provider gap under `BACKEND-ROCM`.

Repair evidence is under `/home/vikash/.cache/rdna3-f16-repair1`.
`integration/receipt.json` and
`prerequisite-integration-operator-audit.json` retain the integration checks.
`mutations/registry-dtype-refusal/recipe.json` and
`mutations/embedding-provider-guard/recipe.json` record the private archive,
source mutation, exact commands, red exit, and byte-exact restoration.
`cpu-focused/registry-final-receipt.json` records four passing registry cases
and 107 assertions. The complete CPU contract suite passes six cases and
240 assertions. The complete Qwen27, Qwen35, graph, and model-registry CPU
targets pass. `state-restore/receipt.json` records the restoration driver.

The private CPU and HIP build directories reuse only independently hashed,
unchanged donor objects. Their compiler flags match after path normalization.
Every changed translation unit is freshly compiled. Every registered executable
is relinked against the current archive. `donor-reuse.json`, the build receipts,
and `diagnostics-final/receipt.json` retain that provenance. The full CTest
recipe includes the versioned shared library required by the ABI export check.
The first private harness omitted that target; its failure remains recorded,
and `shared-library-final/receipt.json` records the successful correction.

After the runner expectation correction, the complete CPU suite passes 703
tests, skips 12, and fails one of 716. The sole failure remains #3102, with
both original token-anchor diagnostics unchanged. `cpu-full-final/receipt.json`
records the serial command, exit eight, and log SHA256
`f96608c90d4e7ba19d25c71fd307f254c97a185ce90ef04899ebbecef7c78571`.
`cpu-full-final/summary.json` links the exact baseline comparison.
The additional CPU skip is the HIP-only public prerequisite test.

The operator runs every registered HIP test with both devices visible, provider
statistics enabled, and graph dedup disabled. Of 724 tests, 692 pass, 12 skip,
and 20 fail. The added public full-attention completion target passes.
All 19 baseline failures retain the exact normalized error lines and doctest
case and assertion summaries. The remaining GGUF failure is the native
`AttnGateSplit` refusal owned by #3106. Baseline ownership waives no gate.
`frozen-v2/operator-full/results.json` records the mutex, environment, serial
CTest command, exit eight, and log SHA256
`f9b103624eeb4b70b59e633db6bd96417e4b13a9f7992d075303b3c438343007`.
The operator comparison has SHA256
`9fc27383f62597710f4813091f66d764892139159c6b797f372625eadc1ce0af`.
It verifies all 4,157 source files and 1,006 declared inputs before and after.
The implementer independently verifies all 19 comparison entries and that hash.

The final staged preflight exits zero. Every executed gate passes, and all
678 in-scope translation units compile in 270.9 seconds. The complete command
and environment are in `preflight-final-staged.json`; its log SHA256 is
`6532355dac5ee2c2d513bfd8dffc8d1408f129b0b462f5b79229a74a874ba8f9`.
The generic sweep skips five argument-dependent checks. The explicit x86 ISA
audit passes against `build-repair-cpu/compile_commands.json`. Exact-range
classification supplies PR #3095 and the immutable repair head.
ARM ISA, CUDA fat-gencode, and Triton AOT audits have no applicable configured
artifact in this CPU/HIP repair. No architecture-specific instruction unit,
CUDA gencode setting, or Triton artifact changes. These dispositions do not
claim a completely green generic preflight. Final owned-record checks run
after adding this evidence; source and build inputs remain unchanged.

Fresh review of the final repair commit and the operator's final gate remain
required. The row stays `ACTIVE`. Native D1 and primary repeatability failures,
the identical-GGUF plugin refusal, and the traced dtype and policy differences
retain their earlier dispositions. This repair accepts no performance result.

## Work breakdown

1. Commit this spec and the scoped issue and inventory records.
2. A fresh implementer captures the production failing test and dtype fixtures.
3. Implement the minimum complete metadata, loading, GEMM, and gather change.
4. Run focused and full gates, then the identical-artifact model comparison.
5. Capture paired traces and post-correctness performance and memory evidence.
6. A fresh reviewer mutates the immutable head. Fresh implementers repair findings.
7. The operator reruns gates, reads the final pull request body, and publishes
   the merge request for the user's review. The current task does not merge it.

The implementation owns `docs/FEATURES.md` when the capability becomes
reachable. It owns `docs/USAGE.md` for the exact gated checkpoint information
and the remaining refused arms. Publish a benchmark detail and index entry
only when a benchmark is actually accepted. Do not rewrite public documents
for the spec-only commit.

## Risks/decisions

The main risk is numerical: raw F16 file values can differ from those values
rounded to the model's BF16 parameters. Token equality cannot substitute for
tracing that conversion. The selected metadata and explicit cast preserve it.

The second risk is residency: repeated conversion can increase temporary
memory and traffic. Measure that cost before changing defaults. A permanent
BF16 shadow or silent retention regression needs a new decision.

The third risk is incomplete metadata propagation through shared helpers or
tied weights. Limit loader admission to the traced Qwen3.5 dense path and use
negative mutations to prove that every required copy is tested.

The fourth risk is a broader primitive contract being mistaken for a complete
F16 model runtime. This row retains BF16/F32 activation and output contracts.
The existing RMSNorm F16 activation refusal is tracked by #2542. Other missing
full-runtime operations need their own owning issue before that scope expands.

## Owed

- [#3153](https://github.com/mudler/vllm.cpp/issues/3153)
  (`ISSUE-GH-3153`), owned by `BACKEND-ROCM-F16-WEIGHTS`, tracks the
  test_rocm_f16_contract failure where the test expects ViewOn to propagate
  layout markers it deliberately does not. Fixed by correcting the test to
  verify the actual contract.
- [#2773](https://github.com/mudler/vllm.cpp/issues/2773), owned by
  `BACKEND-ROCM`, owes diagnosis of this fixture's unchanged-baseline F16-D1
  repeated-engine failure and F16-P128 primary comparison mismatch. The inputs
  are the committed workload hash
  `907e5f88ded44d10ef4fd44a32b6aa7bae80e7a858a6ddf64a03e63f62a29114`,
  GGUF hash `758b5299b027120c3608c43777a89257724d46e26eefac2a8aaad512be15b53f`,
  and derived primary model hash
  `b27fbf8cb5d35406b4fce084bdda1ec10de8fb5e3d48f647edc1ebf325fa740e`.
  That issue's earlier safetensors evidence did not test this materialization
  or establish the cause of these observations. F16-G4 remains failing or
  pending as measured; ownership does not waive parity.
- [#3070](https://github.com/mudler/vllm.cpp/issues/3070), owned by
  `BACKEND-ROCM`, tracks the existing full HIP suite groups: missing native
  operations, quantization admission, placement and scratch assumptions, and
  async-scheduler expectations. The same async group includes
  `test_qwen3_dense_async_serving`. The gfx1100 baseline pairing here does not
  close that issue's separate Strix gate or authorize changes to unrelated rows.
- [#3105](https://github.com/mudler/vllm.cpp/issues/3105), owned by
  `BACKEND-ROCM`, tracks the reproduced gfx1100 HIP Qwen3 paged-token anchor
  failure. Its prompt, generated position, actual ID, and expected ID differ
  from the separately owned CPU #3102 failure.
- [#3102](https://github.com/mudler/vllm.cpp/issues/3102), owned by
  `MODEL-TEXT-qwen3-qwen3-for-causal-lm`, tracks the reproduced baseline CPU
  paged-token anchor failure. [The owning model spec](first-additive-model-qwen3-dense.md)
  owes the matched active-pin diagnosis and reviewed correction.
- [#3098](https://github.com/mudler/vllm.cpp/issues/3098) owns the existing
  Qwen3.5 full-attention-only consumer checks integrated from reviewed PR #3101.
  It retains the pinned primary GGUF refusal and owns the test-only correction
  to the runner's obsolete refusal expectation.
- [#3106](https://github.com/mudler/vllm.cpp/issues/3106), owned by
  `BACKEND-ROCM`, tracks the missing native `AttnGateSplit` provider reached
  after that prerequisite. The unchanged F16 fixture omits
  `rope.dimension_count`, resolves rotary width zero, and enters the unfused
  `FullAttnBlockPaged` preamble. Its explicit-head forward remains failing.
  A separately committed backend spec and reviewed provider implementation
  must close this gap. This repair changes neither the fixture nor the provider.
- [#3100](https://github.com/mudler/vllm.cpp/issues/3100) owns the existing ROCm
  backend allocation and queue device-binding gap. This row's two-device test
  verifies the three scoped operations with independently verified allocations.
- [#2542](https://github.com/mudler/vllm.cpp/issues/2542), owned by
  `MODEL-MM-QWEN4-EXP` and the `Owed` section of
  [the RMSNorm dtype spec](rmsnorm-gamma-dtype-twins.md), tracks the existing
  CUDA and ROCm RMSNorm F16 activation refusal. It is not ownership of every
  missing F16 runtime operation. F16 model activations and F16 outputs remain
  outside this row.
- Other model registry loaders retain their existing BF16 expansion until they
  explicitly provide and propagate a resolved value dtype. This row makes no
  retained-F16 support claim for them. The owning backend row must track any
  subsequent admission with that model's production gate.
- Stacked experts retain expansion because their grouped block-quantized seam
  does not accept F16 weights. Any later retention needs a separately specified
  grouped F16 consumer and its owning issue before admission.
- #2624 owns the plugin registry's Qwen3.8 gateability measurement. This row
  owns its Qwen3.5 measurement under #3092 and cannot declare that primary path
  gateable from source inspection.

## Row inventory

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `BACKEND-ROCM-F16-WEIGHTS` | vLLM `e126687a9a` ordinary ROCm GEMM and parameter conversion, GGUF plugin `d4c1f0d082` | `MatmulKernelRocm`, `MatmulBTKernelRocm`, `EmbeddingKernelRocm`, `GgufLoadPolicy::FromEnv` | F16-G1 through F16-G8 and dated evidence above | [This spec](rocm-f16-weights.md) | `ACTIVE` | Fresh helper, operator verification | [#3092](https://github.com/mudler/vllm.cpp/issues/3092) |

This per-row inventory is the canonical child record, discovered from the spec
glob. The parent backend-matrix retains its existing state and content.
Add an `Outcome` section only when the row reaches `DONE`, with measured values,
rejected alternatives, and the reasons for the final defaults.
