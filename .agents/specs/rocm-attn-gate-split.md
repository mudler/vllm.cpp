# Complete ROCm unfused gated attention

Row: `BACKEND-ROCM-ATTN-GATE-SPLIT`.

Issue: [#3106](https://github.com/mudler/vllm.cpp/issues/3106).

In-flow test repair: [#3112](https://github.com/mudler/vllm.cpp/issues/3112).

Parent: `BACKEND-ROCM`. Branch: `row/BACKEND-ROCM-ATTN-GATE-SPLIT`.

Spec base: `4137b96369467e925bfdf0738e5bad013c89b58f`.

## Now

`ACTIVE`. Native split and the two zero-width model guards are implemented.
The CPU model regressions pass in both default and explicitly unfused modes.
The operator verifies the split on physical device buffers, both visible devices,
and offsets beyond 32 bits. The F16 head observer follows the actual native
BF16-to-F32 cast. All 58 production cases pass with 10,970 assertions.
The complete HIP suite passes at the immutable head `15d8e1682`.

Fresh review of `15d8e1682` returned PASS with no findings: four CPU guard
mutations redden both model modes, and the six operator GPU mutation gates each
redden their guarantee (G1 reproduces the recorded pre-fix provider refusal at
the production regression, G2 the query/gate layout, G3 the nearest-even BF16
narrowing, G4 the exact gate widening, G5 input preservation, and G6 the
current-device binding). Pristine rebuilt-head reruns of the five receipt gates
exit zero. The operator receipts, review report, and mutation logs are retained
with the row evidence.

The row uses one pull request under the repository default. The parent lifecycle
does not change. Specification commits precede each implementation scope.
The operator owns every GPU invocation and independently reruns the reviewed gate.

## Scope

Implement native ROCm `vt::AttnGateSplit` and complete the unchanged Qwen3.5 F16
registered-forward fixture that reaches it. Preserve all current split validation,
including its contiguous-input requirement. Preserve the loader's resolved
zero-width rotation as a measured no-work model case. Keep shared primitive
validation unchanged.

The split accepts F32 and BF16 input, F32 and BF16 query output, and F32 gate
output. Each head stores its query values followed by its gate values. Query
narrowing uses the existing nearest-even BF16 conversion. Gate widening is exact.
The operation does not apply sigmoid or modify its input.

Exclude F16 activations, narrower gates, strided split admission, fused-preamble
changes, quantized GEMM, PR #2782, scheduler behavior, GDN arithmetic, allocation
policy, CI, global oracle pins, and unrelated HIP baseline failures. Preserve the
F16 fixture, its missing rotary metadata, its weights, and its expected successful
forward. Do not force the fused path or enable host reference fallback.

## Gap verification and history

The operator's frozen HIP log records 57 passing cases of 58, with 10,455 passing
assertions, before the explicit-head case throws:

```text
vt: no kernel for op AttnGateSplit (id 68) on device rocm (type 5)
```

The case at `tests/vllm/test_gguf_keep_quant.cpp:3011` calls
`BuildDenseF16Gguf(DenseDims{}, false, true)`. Its geometry is hidden width 64,
vocabulary 32, two query heads, one KV head, head width 32, and two full-attention
layers. It later repeats with tied weights. The first explicit-head refusal
currently prevents that later execution.

Evidence: `/home/vikash/.cache/rdna3-f16-repair1/operator-focused/gguf-keep-quant-control.log`,
SHA256 `f4d682bdfa800b80c7e25d75b51c31f00a0d3168579534a3e25fff2fda84752a`.
The HIP archive has SHA256
`dae20f3a702462461ae9edc7143e4774fe448882b16fe506441194d446f02bc7`.
The source map has SHA256
`398ad011bd6d16d04f1aa4ada98e0058def422a2b4925b6dd39d90604f838696`.
These are retained prerequisite measurements, not gates run by the spec author.

At the base, `src/vt/rocm/rocm_ops.hip` registers the fused preamble but no
`kAttnGateSplit`. The queried issue is open and has no assignee. No existing
implementation pull request was found. `git log origin/main -S'AttnGateSplit'
-- src/vt/rocm` returns no implementation history. Fetched main
`96c5e4719` contains no repair for this operation.

The CPU implementation is `src/vt/cpu/cpu_ops.cpp:3985`. CUDA's templated split
starts at `src/vt/cuda/cuda_glue.cu:193`. Merged
[PR #2538](https://github.com/mudler/vllm.cpp/pull/2538) preserves BF16 query
width on existing backends. It does not supply a ROCm split provider.

The second refusal follows directly from the executing chain:

1. `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:989` defaults missing
   `rope.dimension_count` to zero.
2. `FullAttnBlockPaged` reads that width at `qwen3_5.cpp:5718`.
3. Its unfused path calls the split at line 5813, followed by both RMSNorm calls.
4. It calls `vt::RopeNeox` unconditionally at line 5826.
5. `src/vt/ops.cpp:1600` requires a strictly positive rotary width.

The ROCm rotation provider at `rocm_dense_basic.hip:645` already returns when
its computed work count is zero. The shared wrapper prevents that case from
reaching it. `git log -S'args.rotary_dim > 0' -- src/vt/ops.cpp` identifies the
original RoPE work and later cache/fusion changes. This specification does not
attribute the refusal to the F16 implementation.

## Primary source and required measurement

Use vLLM `e126687a9a828d513c01a07cd69f025f27d63280`, the active
[primary pin](../upstream-sync.md). Resolve runtime and source paths from the
shared checkout's environment and the operator. A checkout at another revision
is not this reference. Reading the pinned object with `git show` is permitted.

The executing split is in `vllm/model_executor/models/qwen3_next.py:424-443`.
It reshapes query/gate projection values by head, then calls `torch.chunk` at
line 430. Qwen3.5 reuses this attention class. The local F32 gate is an existing
storage adaptation for `vt::SigmoidGateBf16`; it does not represent a new
upstream gate dtype. Preserve that adaptation and the existing BF16 query arm.

The upstream fused-preamble test's reference preparation is
`tests/kernels/test_fused_qk_norm_rope_gate.py:49-67`. Preserve its BF16 input,
seed 13, head width 256, token counts 1, 4, and 37, and query/KV head pairs
24/4 and 16/2 when porting the applicable split preparation. The local test
isolates the split and compares exact values. It does not claim to port the
fused RMSNorm, rotary embedding, or MRoPE portions of that test.

Zero-width source evidence needs execution before it chooses a repair:

- `vllm/model_executor/layers/rotary_embedding/base.py:178-200` preserves the
  entire query/key tail when the rotation slice is empty.
- `base.py:254-271` selects AITER when enabled, otherwise the custom HIP operation.
- `csrc/libtorch_stable/pos_encoding_kernels.cu:49-71` has no rotation iterations
  for zero width, but its launcher at line 168 computes a zero-thread block.
- `tests/kernels/core/test_rotary_embedding.py:33` covers width 32, not zero.
  A zero-width regression is a local extension and must be identified as such.

Run the actual split and zero-width rotation on the existing primary runtime
under its production defaults. Use isolated processes for native-reference and
HIP rotation so a launch error cannot contaminate the next control. Test F32
and BF16 query/key values, width 32, zero rotary width, three tokens, two query
heads, one KV head, and positions `[0,1,2]`. Compare every output byte with the
input and synchronize before accepting the result. Record AITER selection,
runtime revisions, script hash, command, error text, and exit status.

The operator ran eight isolated processes on 9 September 2026. Both split
dtypes, both native-reference rotation dtypes, and both default-dispatch rotation
dtypes pass. Both explicit `forward_hip` calls fail with invalid launch
configuration. The default operation selects `forward_native`, with compilation
custom operations `['+sparse_attn_indexer', 'none']`, and preserves every query/key
byte. No custom-operation override changes those defaults.

The logs and receipts are under
`/home/vikash/.cache/rocm-attn-gate-split-operator`. `results.json` has SHA256
`b2f6b5326f0d34c447f78cee8704952a1cab66792634232b14dd8f5061548787`.
`final-audit.json` records 2,502 input hashes verified before and after execution.
The spec author independently verifies all eight log hashes. The frozen probe
has SHA256 `d5d84245b60dbcc200ba0b64499ce8a78f78e0e03a6c4f1c7c815446ceba5fca`.
These are measured primitive semantics, not a claim that the direct HIP
operation accepts zero width.

This measurement does not establish an upstream model's zero-width configuration
or missing-GGUF-field default. Active `qwen3_next.py:325-330` calls `get_rope`
from model rope parameters. The local HF parser at `hf_config.cpp:569-583`
defaults Qwen3.5's partial factor to 0.25. The GGUF loader instead explicitly
resolves the missing field to zero, unchanged since `1a4db5c3c`.
`MaybeBuildAttnCosSin` at `qwen3_5.cpp:4813` already suppresses rotation-cache
work at zero. Preserve that existing local resolved configuration.

The operator selects an exact local adaptation: skip only the two model
rotation calls when their resolved width is zero. The measured identity
supports this no-work case. Do not claim parity for upstream model construction,
broaden the shared primitive, invent a quarter-width GGUF default, or edit the
fixture. Record this distinction in the implementation and review evidence.

The identical-GGUF plugin remains unable to run the F16 parent workload at its
pinned text registration. This row does not alter the plugin or substitute a
projector. The operator's successful materialized model under the F16 spec is
separate evidence from the operation qualification required here.

## Design and authorized implementation files

Add `src/vt/rocm/rocm_attn_gate_split.hip`. Register its provider in
`src/vt/rocm/rocm_ops.hip`, and add the file to both HIP source lists in
`CMakeLists.txt`. Use the existing queue-device helper, queue stream, conversion
helpers, launch error reporting, and grid-stride conventions. Do not allocate
scratch or synchronize inside the split provider.

For output index `i`, compute `d=i%Dh`, `h=(i/Dh)%Hq`, and `t=i/(Dh*Hq)`.
The source head begins at `t*Hq*2*Dh+h*2*Dh`. Query reads offset `d`; gate
reads offset `Dh+d`. Dispatch all four input/query dtype pairs. Zero elements
return without a kernel launch. Preserve 64-bit indexing and non-block-aligned
sizes. Do not split the complete row into two halves.

`src/vt/ops.cpp:5266-5295` remains the split validation authority. Do not broaden
contiguity or dtypes. Its current error messages remain unchanged.

The selected identity repair changes only the unfused `RopeNeox` calls in
`FullAttnBlock` and `FullAttnBlockPaged` in `src/vllm/model_executor/models/qwen3_5.cpp`,
at lines 5653 and 5826. Call the existing operation when `rot != 0`.
Do not use `rot > 0`, which would silently admit invalid negative widths.
Preserve both RMSNorm calls, the query/key tensors, all metadata validation,
positive rotation behavior, and errors for negative, odd, or oversized widths.
The shared `vt::RopeNeox` contract remains unchanged. No loader default changes.

Add `tests/vt/test_rocm_attn_gate_split.cpp` and register its HIP target in
`tests/CMakeLists.txt`. Add focused CPU/model regression cases for both guarded
call sites where the existing test harness reaches them. The existing
registered-forward test remains the production gate. A separate focused observer
may assert the split
provider executes, but must invoke the real provider and preserve the fixture.

The downstream operations are already registered: both RMSNorm calls,
`RopeNeox`, `CastBf16`, the cache write, `PagedAttention`, and
`SigmoidGateBf16`. Their applicable F32/BF16 operands match the existing provider
contracts. This static check does not establish successful execution. Retain
any next actual refusal and repair it through the same scoped review process.

### Identify the actual F16 output head

The operator's first candidate run completes both F16 forwards and exposes an
observer defect in `tests/vllm/test_gguf_keep_quant.cpp::F16ForwardObservation`.
Its width comparison counts four attention projections and the output head.
Both tied modes report five against the unchanged exact-one assertion.
The log records 58 cases, 57 passing cases, and 10,934 passing assertions.
No retained-versus-expanded numerical assertion fails.

The frozen operator log is
`/home/vikash/.cache/rocm-attn-gate-split-impl/green-focused/gguf-production-full-operator.log`,
SHA256 `cf9cd125bce34008bba2bfd0e51f6fbd018facbc552c31e21a5f0f57fdc89d1a`.
Its adjacent receipt binds the commands and source inputs before and after
execution. This is the original failing result for #3112.
`git log -S'marked_heads'` attributes the observer to `822ee510c`.
The current implementer did not author that observer.

Authorize a scoped observer repair in `tests/vllm/test_gguf_keep_quant.cpp`.
Record the output and weight metadata of each real GEMM call.
Record completed calls after native conversion recursion returns.
`rocm_matmul_hipblaslt.hip:594` and line 665 invoke prepared GEMMs without the
original marker. Completion order preserves the original outer head operand.
Bind the final GEMM output to the device logits returned by the registered
forward, including its pointer, shape, dtype, and device.
For a BF16 head, observe the actual native `CastF32` call between those tensors.
Its source must match the GEMM output. Its destination must match returned logits.
Keep the direct binding for an F32 head.
Count marked head calls by that final GEMM's exact physical weight identity.
Include weight shape, strides, and orientation in the identity.
This distinguishes unrelated equal-width projections and pooled output addresses.
It also preserves tied embedding weights without guessing their loader addresses.

Require device logits for this native production gate.
Keep `marked_heads == 1`, both tied modes, the existing vocabulary, fixture bytes,
native provider checks, marked GEMM and embedding checks, and all numerical checks.
Do not increase the count, use a lower bound, or alter model configuration.
The executing dense head uses `DenseLogitsF32D` at `qwen3_5.cpp:3293-3295`.
Both unchanged GGUF orientations select `MatmulBf16LogitsF32D` at lines 1777-1782.
That helper computes BF16 logits, then calls `vt::CastF32` for the returned storage.
The initial direct-only witness failed this binding and retained that failure.
Its operator log has SHA256
`cca1ca70954d4f425850b3f92e66450c1b2a754c338a7a882444d4f2dbecb505`.
The unrelated MTP head at lines 9203-9211 does not establish this executing chain.

Extend the existing test linker interposition in
`tests/vllm/rocm_f16_native_observer.h` and `tests/CMakeLists.txt` for
`CastF32KernelRocm`. Invoke the real native function and preserve provider selection.
Record the call metadata without changing an operand or fixture.

Repeat the unchanged production case after the observer repair.
In independent scratch mutations, omit its actual final GEMM observation and
remove that call's F16 marker. Independently omit the actual cast observation.
Each mutation must fail the production gate.
Restore every source and build byte after each mutation.
The operator retains GPU authority and independently reruns the reviewed result.
This amendment precedes the observer implementation in Git.

## Tests and gate order

| Gate | Required result |
|---|---|
| SPLIT-G0 | Measured default primitive identity and split values pass; retain the two explicit HIP launch failures and model-config limitation |
| SPLIT-G1 | Small native split case fails before implementation at the missing provider, then passes every applicable dtype and shape |
| SPLIT-G2 | Selected zero-width repair has its own original refusal, identity control, and validation regressions |
| SPLIT-G3 | Unchanged F16 registered forward completes explicit and tied heads, retaining exact keep/expand logits and native calls |
| SPLIT-G4 | Focused CPU/HIP targets, complete applicable CTest suites, and preflight run on the immutable implementation |
| SPLIT-G5 | Fresh static and mutation review passes, followed by the operator's independent gate |

The smallest native red uses `T=2,Hq=2,Dh=2` with unique token/head/query/gate
sentinels. Port the CPU case at `tests/vt/test_ops_glue.cpp:201`. That target
constructs CPU tensors and cannot establish ROCm execution by itself.

The new HIP cases use physical device buffers and assert a native provider.
Cover all four dtype pairs, BF16 rounding midpoints, zero tokens, and ragged
`T=3,Hq=3,Dh=5`. Preserve the BF16 query byte-width and exact gate-value checks
from `test_ops_glue.cpp:239`. Cover the upstream split preparation parameters
specified above with captured active-runtime expected values.

Refuse malformed ranks, mismatched shapes, forbidden F16 input, forbidden gate
dtype, noncontiguous input or outputs, and device mismatches through the shared
entry point. Enqueue producer, split, and output copy on a nondefault queue.
Verify queue-device binding with independently verified allocations on the
operator's two visible devices. Do not rely on the separately owed allocation
policy repair.

For zero rotation, compare the model's normalized query/key values before
attention against the measured identity semantics. Cover both guarded model
paths and nonzero positions. Preserve negative, odd, and oversized model-width
refusals through the existing primitive. Preserve positive-width results and
their existing tolerances. The unmodified shared wrapper still refuses a direct
zero-width primitive call; the model does not issue work for its empty rotation.

The reviewer deletes only the native registration and reruns the split and
production gates. Both must fail. Independently mutate per-head addressing,
query/gate offsets, BF16 query narrowing, and queue binding. Each corresponding
case must detect its defect. For the selected rotation repair, independently
restore each unconditional model call. Its focused production case must fail.
Replace `rot != 0` with `rot > 0` and verify the negative-width regression fails.
Restore all source and build bytes.

The full-suite gate preserves existing CPU #3102 and HIP #3070/#3105 failures
as failing until their own repairs land. Compare pristine baseline diagnostics
under identical scheduling and device visibility. Ownership is not a waiver.
This row owes no throughput claim. Its source-only design stage omits product
red/green, mutations, GPU work, and full builds because no implementation exists.

## Risks and stop conditions

An upstream split is a view, whereas the local seam writes contiguous outputs.
Preserve values and the declared destination widths without pretending the
memory formats or invocation counts match upstream.

Zero rotary width exposes a measured disagreement between default primitive
dispatch and the explicit custom HIP launch. Preserve both results and the
upstream model-configuration limitation. Never conceal that disagreement or
modify the production fixture to select a different path.

Return `NEEDS_CONTEXT` for a missing runtime identity or missing fixture bytes.
Return `NEEDS_DECISION` for a changed primary default result, an additional required
production operation, or a required change outside the authorized files.
Continue independent split work when an additional operation needs scope review.
Do not stop at an implementation finding that a fresh implementer can repair.

## Implementation evidence

All implementation artifacts are under
`/home/vikash/.cache/rocm-attn-gate-split-impl` on the measured host.
The implementer builds in the linked worktree
`/home/vikash/vllm.cpp-rocm-attn-gate-split-impl`.
The original implementation base is `9b3ce386849854f75092ef51288b362acbfcc3d3`.
Spec commits `a6c288a5f` and `8bb7b5439` precede their respective observer repairs.
The implementer runs CPU commands only. The operator owns all HIP execution.

The CPU and HIP builds use Ninja, Release, tests enabled, and examples disabled.
CUDA, Metal, Vulkan, Tenstorrent, the server, and bundled BoringSSL are disabled.
HIP uses `/opt/rocm/lib/llvm/bin/clang++` and `gfx1100`.
The builds use `cmake --build build-split-cpu -- -j4` and
`cmake --build build-split-hip -- -j4`.
Both complete builds exit zero. Frozen requests retain both `CMakeCache.txt`
and `compile_commands.json` with their SHA256 values.

| Gate | Measured result and retained evidence |
|---|---|
| SPLIT-G0 | The original eight-process measurement remains authoritative, including both explicit HIP rotation failures and the model-configuration limitation |
| SPLIT-G1 red | `red/operator-receipt.json` records the smallest split and registered F16 forward both failing at the missing provider before product edits |
| SPLIT-G1 green | `green-repaired/operator-receipt.json` records 974 ordinary assertions, 60 device-binding assertions, and 15 large-index assertions passing |
| SPLIT-G2 red | `cpu-red-receipt.json` binds both original unconditional model refusals, in default and unfused modes, to the exact new test source |
| SPLIT-G2 green | `cpu-green-qwen-default.log` and `cpu-green-qwen-unfused.log` each record 14 passing cases and 835 passing assertions |
| SPLIT-G3 | `green-complete/operator-receipt.json` records all five focused commands passing, including 58 production cases and 10,970 assertions |
| SPLIT-G4 CPU | Pristine CTest has 716 tests and candidate CTest has 717 tests; both exit 8 with only the two existing Qwen3 `anchor_ok` failures |
| SPLIT-G4 HIP | Full build passes; `full-hip-complete/request.json` seals 4,851 inputs for complete operator CTest, which remains pending |
| SPLIT-G5 | The implementer's CPU mutations fail as required; operator HIP mutations, fresh review, and independent operator verification remain pending |

The complete CPU commands use `ctest --test-dir build-split-cpu --output-on-failure -j4`.
Both runs set `GIT_CONFIG_GLOBAL=/dev/null`, an isolated `TMPDIR` and Git ceiling,
and the existing NumPy-only `PYTHONPATH`.
`cpu-full-comparison.json` records identical failure messages and the same pinned
Qwen3 checkpoint. The existing CPU #3102 failure remains `FAILING`.
This comparison does not waive the existing HIP #3070 or #3105 failures.

`mutations-cpu/corrected-link-results.json` records a passing scratch control
and 12 detected mutations. Each model path independently loses its zero guard,
admits negative widths, loses each normalization call, and corrupts each normalized
query or key. `mutations-cpu/restoration.json` verifies the original source,
archive, object, and executable bytes remain unchanged.
The initial scratch links accidentally retained the original whole-archive option.
Their apparent passes are void and retained as `void-original-link-receipt.json`.
The corrected links replace both archive references and detect every defect.
The independent direct-primitive zero mutation also fails its two dtype controls.

The captured upstream split fixture is `tests/fixtures/rocm_attn_gate_split`.
Its generator runs the pinned active runtime with seed 13 and every specified
token/head pair. `cases.bin` contains 3,440,640 bytes with SHA256
`dca91b8782df22399751c4b5f729e8742ac5a7e094dc963be0fb6b64e98cc605`.
The manifest has SHA256
`64adbfcd2552ffbb12655e835d665585da30833e8a0f209e030c1882aeb7a6cc`.
The local F32 gate remains a storage adaptation, and this row makes no speed claim.

The pre-edit preflight passes with the existing five argument-dependent skips.
The CPU ISA audit passes with `build-split-cpu/compile_commands.json`.
The candidate preflight finds one malformed imported symbol citation for #3112.
The corrected citation passes `scripts/check-symbol-anchors.py`.
The candidate preflight exits 1 for that original citation failure, although its
later automatic checker pass sees the corrected citation. Tree compilation passes.
The final preflight rerun and remaining applicable range checks stay pending.
No pending check is a pass.

The final production log has SHA256
`0211aab99387f025ddbd52049919f3aee641a088358d845c9f2eb951c03dcd2b`.
The operator verifies 4,159 input and request hashes before and after execution.
The earlier direct-output and recursive-inner-weight observer failures remain
retained beside their receipts. The standalone model driver's no-argument
invocation exits 77 with usage text and proves no model behavior.

## Inventory

The canonical spec scan owns this child record. No parent matrix or roadmap
lifecycle changes. Update this table and `Now` together when the child moves.

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `BACKEND-ROCM-ATTN-GATE-SPLIT` | vLLM `e126687a9a`, `qwen3_next.py:424-443`, rotary `base.py:178-200` and HIP launcher | `vt::AttnGateSplit`, `vt::RopeNeox`, `FullAttnBlockPaged` | SPLIT-G0 through SPLIT-G5 and retained production red | This file | `ACTIVE` | Fresh implementer, fresh reviewer, operator verification | #3106 |

Add `Outcome` only when the complete production repair reaches `DONE`.
