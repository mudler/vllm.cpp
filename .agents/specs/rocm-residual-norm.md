# Preserve compiled residual expressions on ROCm

Owning row: `BACKEND-ROCM-RESIDUAL-NORM`
Parent row: `BACKEND-ROCM`, whose lifecycle remains `ACTIVE`.
Owner: the delegated residual-norm implementer, fresh reviewer, and coordinating operator.
Issue: [#3103](https://github.com/mudler/vllm.cpp/issues/3103), `ISSUE-GH-3103`.
Base: `9083a5fb35822e86b31cf786e0c80150dfae5478`.
Integration: one pull request under the repository default. Commit this spec before implementation.
The implementation pull request closes #3103 when its required gates pass and it lands.

## Now

State: `SPIKE`. The shared repair is implemented and the focused production witness passes on `gfx1100`.
The committed specification precedes both the captured red and the implementation.
The complete hardware gate and fresh mutation review remain required before acceptance.
The pinned upstream export passes all 264 cases; native comparison passes 528 CPU/ROCm executions and 22,576 assertions.
The operator found a blocking-stream setup error in the new nondefault-stream fixture.
The scoped fixture repair uses an explicitly owned nonblocking stream; its hardware mutation gate remains pending.
The related MoE production token gate remains open; its baseline failed at six generated positions.
No performance result is accepted before the complete declared token gate passes.

## Problem and scope

The compiled primary normalizes an FP32 residual expression before narrowing its result.
The native ROCm path first stores and reloads that expression as BF16.
The compiler also retains the post-attention operands across MoE and recomputes their sum at the next normalization.
Changing one rounding instruction does not preserve this complete lifetime.

Extend the shared residual-normalization and `vt::FusedChain` seams to represent the executing primary expression.
Wire those semantics into the registered Qwen3 MoE production forward.
Keep every physical model residual and activation buffer at the primary's BF16 dtype.
Keep FP32 arithmetic in registers and reduction scratch where the primary uses FP32 arithmetic.
Do not allocate a persistent FP32 residual stream.

Preserve existing operation signatures, recipe defaults, and legacy callers unless a separately proved correction requires a scoped update.
This work changes neither the upstream pin nor CI, checkers, quantized experts, or PR #2782.
It does not repair the separate resource-device issue #3100.
It does not silently broaden this row to attention, LM-head, or grouped-router implementations.
Those gaps still require implementation in the enclosing parity campaign.

## Inventory

| Stable ID | Upstream source | Local anchor | Tests and evidence | State |
|---|---|---|---|---|
| `ROCM-RESIDUAL-EXPR-NORM` | `vllm/ir/ops/layernorm.py::fused_add_rms_norm`, generated `ckic6h6`, `ctj2x6`, and `c5slugd` kernels | `src/vt/rocm/rocm_residual_rmsnorm.hip::ResidualRmsNormRowKernel`, `vt::ResidualRmsNorm` | Captured row-zero values and ordered-expression cases | `SPIKE` |
| `ROCM-RESIDUAL-EXPR-FUSION` | Executed normalization partitions described below | `include/vt/fused_recipe.h::FStep`, `src/vt/ops.cpp::FusedChainCompositeImpl` | Composite/native equivalence and boundary mutations | `SPIKE` |
| `ROCM-RESIDUAL-EXPR-FORWARD` | `vllm/model_executor/models/qwen3_moe.py::Qwen3MoeDecoderLayer` | `src/vllm/model_executor/models/qwen3_moe.cpp::RunMoeLayer`, `ForwardLayers` | Registered loader/forward witness, full token gate, lifetime and wiring mutations | `SPIKE` |

Every inventory item belongs to this spec and #3103.
The parent backend row does not change lifecycle in this slice.

## Pinned source and measured evidence

The primary is vLLM `e126687a9a828d513c01a07cd69f025f27d63280`.
The verified executing source checkout is `/home/vikash/oracle/gfx1100-active-2773/source`.
The older shared `VLLM_SOURCE` checkout reports `5559679229` and is not this evidence's source revision.
These paths identify measured artifacts. They are not environment defaults for another developer.

The primary selects default compilation, production graphs, native IR normalization, `ROCM_ATTN`, and Triton experts.
The resolved engine configuration appears in `oracle-diagnostic-2.log` at lines 15 and 34 to 35.
No eager denominator is allowed.

Evidence root: `/home/vikash/vllm.cpp-rdna3-moe-impl/build-rdna3-moe-hip/evidence`.
The handoff is `norm-gap-handoff/handoff.md`.
Its SHA256 is `9850ee49c93bb82ddfca1a811fb422337c33ea09335e749fbf3c42dfcaa0400a`.
The manifest is `norm-gap-handoff/manifest.json`.
Its SHA256 is `8531cb41f0168d27455b2dc8062a57d5db7afd4e0822013332f398bbf2239c65`.
The spec author independently verified all 14 entries against their byte counts and hashes.

The six generated modules retain complete code, original paths, hashes, and allocation dtypes.
Their full basenames are in the manifest. Short names below identify those immutable entries.

| Module | Executing evidence |
|---|---|
| `c4r2zan` | BF16 embedding, first input norm, QKV, and the Q/K preamble |
| `cz7ckbn` | Layer-zero o_proj, post-attention norm, MoE, next input norm, and QKV |
| `ckic6h6` | Two-input post-attention sum and norm, no residual store |
| `ctj2x6` | Ordered three-input norm and BF16 residual store |
| `cywagh7` | Layer-one o_proj, post-attention norm, MoE, and final norm |
| `c5slugd` | Ordered three-input final norm, no residual consumer |

At the source level, `vllm/ir/ops/layernorm.py::fused_add_rms_norm`, lines 44 to 62, adds in FP32.
It declares BF16 residual and gamma boundaries that the default compiler can eliminate.
The generated executing kernels determine which boundaries survive.
An eager Python expression alone is insufficient evidence for the compiled contract.

At the inspected feature head, `git log -S'ResRound'` attributes the native boundary to `50b0709b3`.
`git log -S'lm_head'` identifies the earlier Qwen3 MoE forward commits `673b2a84a` and `c56ab287d`.
The relevant residual, model, router, and dense-attention files are unchanged between this base and fetched main `96c5e4719`.
The live issue #3103 is open. No open residual repair pull request was found during this audit.

## Required arithmetic and lifetime

Let `f(x)` load a BF16 operand as FP32 and `b(x)` round FP32 to BF16.
Let every arithmetic operation below use FP32, in the shown order.
Let `N(s,w)` compute `b((s * rsqrt(mean(s*s) + eps)) * f(w))`.
There is no BF16 conversion between normalization and gamma multiplication in the observed generated kernels.
The reduction order and reciprocal-square-root implementation require executing-kernel evidence in addition to the expression.

### First input normalization

The first layer consumes the BF16 embedding `r0` and emits `N(f(r0), w_in0)`.
The retained residual is `r0`, still BF16.
The native zero-residual initialization is equivalent only if it preserves these values and the actual output.
The saved first QKV input and output match exactly on both sides.
Preserve that control when adding the new lifetime.

### Post-attention normalization without materialization

For BF16 attention projection `a` and BF16 materialized residual `r`:

```text
s = f(a) + f(r)
n = N(s, w_post)
```

Read `a` and `r` without modifying either input.
Do not store `b(s)` and use it for normalization or for the next layer.
The generated `ckic6h6` kernel loads the two operands at lines 33 to 38.
It normalizes their FP32 sum at lines 39 to 50 and stores BF16 normalized outputs at lines 51 to 52.
Neither output is a residual sum.
The two normalized copies serve the router and expert preparation in the primary.
A shared native buffer can serve both consumers if values and lifetime remain equivalent.

Retain ownership of `a` and `r` through the MoE invocation.
The current local `attn` is a stack-local owning buffer in `RunMoeLayer`.
Either retain that owner until the next-input or final norm, or execute the consuming norm before the owner leaves scope.
The driver can schedule the next normalization immediately after MoE, matching the generated partition, if it prevents duplicate normalization.
A view without its owner is not a complete repair.

### Next input normalization with materialization

For BF16 MoE result `m` and the retained BF16 operands `a` and `r`:

```text
s_attention = f(a) + f(r)
s_next = f(m) + s_attention
r_next = b(s_next)
n_next = N(s_next, w_in_next)
```

Preserve the association `m + (a + r)`.
Do not use `(m + a) + r` or `m + b(a + r)`.
Store `r_next` as BF16 while normalizing `s_next` before its BF16 rounding.
Do not reload the rounded `r_next` for normalization.
The next QKV consumes `n_next` and the next attention partition retains `r_next`.
The generated `ctj2x6` kernel loads all operands at lines 33 to 39.
It evaluates the ordered additions at lines 40 to 43 and stores residual and norm outputs at lines 57 to 58.
`cz7ckbn`, lines 1028 to 1040 and 1076, records the allocations, call, QKV use, and returned residual.

### Final normalization

The final normalization evaluates the same ordered three-term expression.
It emits `N(s_next, w_final)` and has no residual output consumer.
The generated `c5slugd`, lines 33 to 56, overwrites the MoE buffer with its BF16 normalized output.
An independent BF16 output allocation is permissible if ownership and values are equivalent.
Do not create a final residual allocation merely to make the existing in-place API fit.

## Shared design

Add a typed shared operation or explicit descriptor for two-input and ordered three-input residual expressions.
Its inputs are read-only activation operands and gamma.
Its outputs are a required normalized tensor and an optional materialized residual tensor.
The descriptor explicitly identifies operand count, addition order, and residual materialization.
Do not alter the existing provider function-pointer signature through a cast.
Validate shape, dtype, stride, device, output aliases, and invalid descriptor combinations before dispatch.
The new native path must select the complete capability before model execution.
Select the measured compiled lifetime through a shared backend capability or numeric-policy seam.
CPU reference registration alone must not enable that lifetime for other production backends.
Preserve existing CPU and CUDA model defaults unless matching executing-oracle evidence requires a scoped correction.

Represent this operation in `vt::FusedChain` and its composite realization.
The existing composite folds one in-place `kAdd` into `RmsNorm`, at `src/vt/ops.cpp::FusedChainCompositeImpl`, lines 1159 to 1185.
That fold cannot express a read-only sum, three operands, and a separately materialized residual.
Extend the shared representation instead of adding a model-specific HIP sequence.
The current binding has eight operand slots but a recipe step admits three inputs.
A three-activation operation plus gamma needs an explicit, validated representation of all four inputs.
Extend that capacity or add a typed residual-expression descriptor without overloading another field's meaning.

Provide a CPU reference and the ROCm implementation through the same typed operation.
Composite and optimized execution must agree on arithmetic boundaries and observable buffers.
Route both adopted-fusion and explicit-unfused model modes through the new shared semantics.
Keep the old standard and Gemma recipes unchanged for callers that retain their previous contract.
Do not turn a legacy `ResRound` deletion into a global numeric change.

Use row-local FP32 registers or deterministic reduction scratch.
No persistent FP32 activation or residual tensor is allowed on this BF16 path.
If a kernel writes a residual that aliases an input, preserve all needed unrounded values before writing.
Protect reads and stores across threads, including the second normalization pass over wide rows.
Support the admitted shapes with tail masks and deterministic reductions.
Use the queue's device and stream and preserve the caller's ambient device.
Allocate no new storage during graph capture.
Keep every captured buffer alive and stable for graph replay.

## Smallest failing tests

### Production row-zero witness

Start with the unchanged fixture from [the MoE spec](rocm-bf16-moe.md).
Its config SHA256 is `321926020ada026d8dd85f74543dd12b6426301706b4e1f4fbca60a44b871cea`.
Its safetensors SHA256 is `96cd7f30fee496c69782af2813e438e47d7b026598b0f20a05049c522b279af8`.
Preserve every generator seed, tensor order, prompt, sampling value, cohort, and repeat.

Run the test through `ModelRegistry::Load` and `ModelRegistry::Forward`.
Observe layer-zero post-attention norm on step zero of L33/C2/R0.
The actual row-zero attention projection, embedding residual, and gamma match between runtimes.
Require the 128 observed output values to match the captured primary values exactly.
The current native result differs in 33 words, first at element 3:
`-0.1865234375` instead of `-0.185546875`.
The maximum absolute difference is `0.00390625`.
Capture this red before implementing the repair.

The native descriptor file is `native-diagnostic-1/L33-C2-R0.json`.
Files 14, 15, and 16 contain the actual input, residual, and output for this normalization.
The primary input is `oracle-diagnostic-2/L33-C2-R0-dense-1-out.bin`.
The primary normalized output is `oracle-diagnostic-2/L33-C2-R0-dense-2-x.bin`.
Read the gamma from the exact fixture, as `diagnose-residual-row0.py` does.
`residual-row0-diagnostic.json` reproduces every native word with rounding and every primary word without rounding.
Its SHA256 is `421b5855cc35b095635fa7268f98803eb26ed0f267d15e9a5e41e01fa6703d2e`.
An isolated operation replay supplements this production-entry test and cannot replace it.

### Complete expression witnesses

Add exact cases where intermediate BF16 rounding changes the normalized output.
Add three-input cases that distinguish addition association, early materialization, and post-store normalization.
Use exact BF16 operand witnesses before randomized cases:

| Guarantee | `a` | `r` | `m` | Required FP32 expression | Mutated result |
|---|---:|---:|---:|---:|---:|
| Ordered additions | 256 | -256 | `2^-17` | `m + (a+r) = 2^-17` | `(m+a)+r = 0` |
| Deferred materialization | 1 | `2^-8` | `2^-8` | `b(m+(a+r)) = 1.0078125` | `b(m+b(a+r)) = 1` |

These are scalar residual witnesses. Embed them in admitted row shapes and validate the associated normalization independently.
Cover both materialized and absent residual output, and verify every read-only input remains unchanged.
Cover final-output aliasing only when the descriptor explicitly permits it.
Cover the first layer, a subsequent layer, and the final norm through the production forward.
Capture the primary's later residual output and norm outputs on original executing inputs.
Do not treat reconstruction alone as an observed primary output.

Port applicable pinned normalization tests from `tests/kernels/core/test_layernorm.py::test_rms_norm`
and `tests/kernels/ir/test_layernorm.py::TestRMSNorm` and `TestFusedAddRMSNorm`.
Preserve their parameters, fixtures, seeds, tolerances, failures, and pin attribution for every applicable mode.
Record explicit applicability for unimplemented activation dtypes rather than adding a silent skip.
The BF16 compiled-expression exact witnesses supplement the original tests.
Eager native gamma rounding does not define the separately observed compiled expression.

## Verification and mutation gates

1. Commit the spec, then capture the smallest production and component red results.
2. Build the CPU reference and ROCm implementation with recorded compiler and binary hashes.
3. Run shared descriptor tests, existing RMSNorm and fusion regressions, and the new exact witnesses.
4. Run the unchanged complete MoE production token gate against the pinned production oracle.
5. Run explicit fused and unfused modes, graph capture/replay, two streams, wide rows, tails, and device selection checks.
6. Capture actual allocation dtypes, strides, operand ownership, generated kernels, and materialization points on both sides.
7. Run full staged preflight and exact-range record, path, style, and trailer checks.
8. Obtain fresh static and mutation review on the immutable implementation head.
9. Have the operator rerun the applicable hardware gates before acceptance.

The full model workload is L in `{1,3,33}`, concurrency in `{1,2}`, eight greedy tokens, and three repeats.
The unchanged `test_rocm_moe_bf16` consumes the exact fixture and primary `production.json` through its existing environment variables.
The authoritative cohort result is `oracle-selection-6/production.json` under the evidence root.
Its SHA256 is `f3d27a95ba71bddcae38c3defc21bf3f3ff32ba9373a09faf949ead68b74d7ad`.
Add the shared operation witness as `test_ops_residual_rmsnorm` and the production witness to `test_rocm_moe_bf16`.
Use separate CPU and HIP build directories. The focused gate commands are:

```sh
ctest --test-dir build-residual-cpu --output-on-failure -R '^(test_ops_residual_rmsnorm|test_ops_rmsnorm|test_ops_rmsnorm_weight_dtype|test_ops_fused_chain|test_fused_chain_additivity)$'
ctest --test-dir build-residual-hip --output-on-failure -R '^(test_ops_residual_rmsnorm|test_ops_rmsnorm|test_ops_rmsnorm_weight_dtype|test_ops_fused_chain|test_fused_chain_additivity|test_rocm_moe_bf16)$'
scripts/agent-preflight.sh --staged
```

The operator runs the HIP command inside the required mutex or lease.
Set `VT_ROCM_MOE_FIXTURE`, `VT_ROCM_MOE_ORACLE`, and `VT_ROCM_MOE_OUTPUT` from the immutable fixture, primary cohort result, and fresh output location.
Verify each GPU test reports actual native cases and nonzero assertions.
A device-less skip is not a passing ROCm gate.
The native diagnostic command and environment are retained in `native-diagnostic-1-operator-receipt.json`.
The primary diagnostic recipe is `snapshots/oracle-diagnostic-2/command.json` with the operator's read-only runtime-mount adaptation in `oracle-diagnostic-2-operator-receipt.json`.
The operator substitutes fresh output paths and the reviewed binary, retaining model bytes and engine policy.
All GPU execution stays under the operator's recorded device authority and required mutex or lease.

The reviewer must independently mutate each guarantee:

- Round `a+r` before variance.
- Store and reload `b(a+r)` across MoE.
- Normalize from stored `r_next` instead of its unrounded expression.
- Reassociate the three additions.
- Add a BF16 conversion before gamma multiplication.
- Remove the subsequent BF16 residual materialization.
- Modify a read-only input or release the retained attention owner too early.
- Delete the production call site or force the old residual path.
- Bypass descriptor validation or use the wrong stream/device.

Each mutation must fail its intended focused test with nonzero assertions.
The reviewer restores the scratch tree byte-for-byte after each mutation.
A failing full-token baseline is not an adequate witness for a new mutation.
The focused failure must identify the mutated guarantee.

## Dependencies and remaining capture work

The residual repair is implementable from the saved first-stage data and generated lifetime.
Later exact witnesses require an operator capture of the materialized primary residual and normalized output.
This is a measurement task on the available oracle, not an unresolved product decision.

Attention already differs before the residual norm at row 1, element 7 of the first attention output.
For L33/C2/R0, native `0.0888671875` differs from primary `0.08935546875`.
The first QKV input, weight, and output are exact.
The first attention-output comparison contains 2902 differing BF16 words.

The generated `c4r2zan` preamble keeps normalized Q/K values in FP32 until after RoPE and reads a BF16 cos/sin cache.
The native `dense_attn::AttnBlock` stores BF16 Q/K norms before `RopeNeox` with FP32 cos/sin.
Capture actual Q/K immediately before attention, V, cos/sin, KV data and metadata, and the attention output on both sides.
The preserved capture `/home/vikash/.cache/rdna3-moe-attention-operator/results/L33-C2-R0-attention-0.json` and its ten binary siblings hold exactly those inputs and the output, and a validated CPU transcription of `prefix_prefill._fwd_kernel` separated the two hypotheses on them before any native byte of that boundary existed (#3115).
The native capture-and-replay instrument in `tests/vllm/models/test_rocm_moe_bf16.cpp` then measured both terms on the device: 2918 of 8448 words from the attention kernel on identical Q/K/V, and 1569 Q plus 1542 K words from the preamble, with the cos/sin table and the qkv projection byte-exact on both sides.
The pinned normal decoder prefill enters `prefix_prefill.py` through `chunked_prefill_paged_decode.py`.
Its probability-to-value product narrows probabilities to the V dtype.
The native D128/QG1 dispatch retains FP32 probabilities even during prefill.
Replay identical captured Q/K/V through both executing kernels before selecting that repair.
The cached generated attention module is `_fwd_kernel` in Triton cache directory `I7QZRC574ZPUQULYWLLB3LJHQ5RQVEIOAGBPLJ7M5YBI5ZVUMSCA`.
Its launch metadata and generated code are retained beside that module.

The primary also stores BF16 LM-head output, while `ForwardLayers` currently produces F32 logits directly.
Complete primary hidden, weight, and logits inputs exist in `oracle-diagnostic-2/L33-C2-R0-head-6.json` and its binary files.
Replay those exact hidden and weight bytes through native projection with BF16 output, then shared `CastF32` before the current forward result boundary.
The pinned sampler converts logits to FP32 before sampling.
The local host download and graph result views already require FP32.
Changing only the head buffer dtype would leave those views and host copies invalid.
Do not attribute a head mismatch to GEMM until identical hidden inputs are supplied.
The current saved hidden inputs already differ upstream.
A CPU audit of step 6 confirms BF16 narrowing alone preserves the wrong native argmax, token 63.
The primary chooses token 118. Native BF16 top-two values are `63: 0.3359375` and `118: 0.333984375`.
The primary reverses those values, so the earlier hidden-state repairs remain necessary.

DeepSeek-V2 and dots3-note router refusals are implementation debt in the parent backend campaign.
They do not appear in the Qwen3 fixture's executing path.
DeepSeek passes `n_group=1` even for its softmax default, which the current ROCm router rejects.
Dots3-note requires sigmoid scores, bias for selection only, and the one-group form.
Completing those modes needs a separate scoped router repair and production tests.
The broad #41 ownership does not make the implementations complete.

## Owed

This row owns #3103 directly. The related BF16 MoE row keeps its complete token gate open.
No performance or token requirement is waived by native/legacy agreement.
The assigned scoped issues for the attention and head repairs are #3115 (decode attention and Q/K preamble parity) and #3116 (the BF16 LM-head output boundary), both on the BF16 MoE row, filed from the six-position disposition analysis of commit `cc9d4f565`.
The parent `BACKEND-ROCM` issue #41 retains router debt until its scoped repair is assigned.

## Implementation and evidence

The implementation base is the committed spec `9ec19f80b600c9e713f7297638186e986c67a4d1`.
`ResidualNormDesc` distinguishes the two ordered expressions and optional BF16 residual materialization.
The typed operation validates every tensor and permitted exact alias before dispatch.
`FStep` admits four inputs, and the composite realizes the explicit residual opcode through that operation.
The CPU reference does not select the production policy.
The ROCm backend selects the compiled expression; CPU and CUDA retain the materialized policy.
`RunMoeLayer` consumes the retained attention owner immediately after MoE and transfers the next normalized BF16 buffer's ownership.
The final norm omits residual output. No persistent FP32 activation or residual is introduced.

The native kernel uses a deterministic 256-lane FP32 reduction and reloads original operands before each store.
All variance reads finish before any permitted output alias is written.
Wide rows, padded row strides, and tails use the same arithmetic and ownership contract.
The launcher selects the queue's device and stream and restores the caller's ambient device.
The operation allocates no storage during graph capture.

### Red before implementation

Evidence root: `/home/vikash/.cache/residual-norm-impl`.
The `red-freeze` manifest pins unchanged product blobs, tests, model bytes, linked libraries, compiler flags, and binaries.
The CPU red exited 1 with 33 failures among 128 assertions.
Its binary SHA256 is `c5745f016d2033ec1da40d6b98cc29803c3f757d583f81bb9eb85665c97dabbf`.
The operator's production red exited 1 with 33 failures among 400 assertions.
Every captured attention, residual, and gamma word matched the primary before the failed norm comparison.
The production binary SHA256 is `f932a6c7b49c96111fb1868bf043287c69c7c4375beebcf4dd14535b42249024`.
The operator log SHA256 is `3baba8756846635e085587a99443e3b6c3d3268351fda356f9135809eb8fc49d`.

### Focused green and later primary observation

The operator ran `green-freeze-1/command.json` and `component-command.json` on the same physical GPU under the required mutex.
The production witness passed all 399 assertions, including 128 exact primary norm words.
Its binary SHA256 is `aa38607e0b88202e2d65e45f76c790b7f18c2bd6dd3cb62662c91892f6f56e00`.
The component gate passed 10,700 CPU and native ROCm assertions without a device skip.
Its binary SHA256 is `102f34696d79be728dd8c196087f05857c8b91fcc01a2994e6ec1f8fdb705aad`.
The operator verified 60 sealed inputs before and after those runs in `green-freeze-1/operator-receipts.json`.
The production log SHA256 is `e94ea05e06acc04408e37a73ede8bf7acc24071ec132d011755fb5534d6e0d53`.
The component log SHA256 is `63e2cfcf065030259deda3a59cf521db66bcead298d643e7d0b46b3837d1736a`.

The `oracle-probe-v3` observer wraps original compiled launches with unchanged arguments and verifies their source ASTs.
It observes post-attention, next-input, final, and Q/K preamble launches in L33/C1/R0 and L33/C2/R0.
The operator verified ten capture records, 64 binary payloads, and every read-only input before and after the launches.
All 18 production cohorts, 216 tokens, and every logprob matched the paired observation-disabled control.
The observer script SHA256 is `789138a38dc2db167656356b4ffa18cab8e4141a3b96fb5f24f93bdd9f242512`.
The operator log SHA256 is `220ce248825706013fdaa0a640a5b5acd522d8e6d31f9fcc43244036ef3224a5`.
`oracle-probe-v3/operator-output-checks.json` records the independent comparison.
The actual next residual, next norm, and final norm first-row bytes are embedded with full-payload and slice hashes in `tests/support/residual_norm_later_fixture.h`.
The earlier v1 AST-parser failure and v2 singleton-stride serialization failure remain preserved beside the successful v3 evidence.
Neither failed observer is evidence for a primary arithmetic result.

### Test applicability and remaining gates

`tests/vt/residual_norm_upstream.py` executes the pinned core and IR normalization tests and exports their original BF16 fixtures.
It preserves both residual modes, every token count and width, row strides, device count, seed zero, epsilons, and original tolerances.
The C++ operation consumes the exported original native references and compares direct and shared-fusion dispatch exactly.
Weightless normalization is represented by an explicit unit BF16 gamma; plain normalization uses a zero base.
Torch registration and opcheck execute in the exporter. C++ validates its own descriptor and ownership rules.
F16/F32 activation modes are outside this measured BF16 policy and are explicitly refused, never silently skipped.
The new API has no variance-size override, Gemma modifier, or partial-width norm mode.

The six focused CPU suites, including the existing Qwen3 MoE forward control, pass after adding the later witnesses.
`cpu-green-5.log` records that run. External upstream fixtures require `VT_RESIDUAL_NORM_UPSTREAM`; absence is reported as unexecuted.
The operator completed the upstream export: 144 core cases, 60 IR RMS cases, and 60 IR add cases.
All 1,452 payload hashes are verified, and the native comparison passes 528 CPU/ROCm executions with 22,576 assertions.
Expanded hardware tests, complete token equality in both fusion modes, staged preflight, and fresh mutation review remain required.
The original MoE gate still differs at six generated positions in the L33/C2 tail.

The ROCm platform keeps `support_static_graph_mode()` false at this base in `src/vllm/platforms/rocm.cpp:91`.
Production registry decode therefore does not enter the graph driver.
The existing ROCm graph row owns that platform exclusion; this repair does not change it.
The new operation's hardware gate covers capture and repeated replay with live BF16 operands on both devices.
It also checks two queues, ambient-device restoration, wrong-device streams, and a blocked nondefault stream.
No production graph claim is inferred from that component gate.

### Repair the blocked-stream fixture and its invalidated citation

The operator's component run at `94ac5d1742c540fc5cbc2084d0fb74fdfd5dab41` fails before the stream-order check.
The test requests a backend queue, but `RocmBackend::CreateQueue` uses `hipStreamCreate` and returns flags zero.
The existing guarantee requires a stream independent of default-stream synchronization.
The fixture therefore owns an explicit `hipStreamNonBlocking` stream within the existing device scope.
Its release guard synchronizes the callback before destroying callback state or operands during normal and exceptional exits.
The flag assertion and real wrong-stream mutation remain required; the backend default does not change.

The added residual descriptor include moves `vt::Backend` to line 23.
Repair only that citation in the `BACKEND-PLATFORM` matrix row, starting from the complete target file.
The scoped proof requires every other matrix byte to remain unchanged.
The [repair evidence](../../docs/bench-evidence/rocm-residual-norm/README.md)
retains the operator red, record red/green, compiler recipe, upstream results, and unchanged-product proof.
The repaired CPU component binary passes six tests and 8,985 assertions, including all 264 upstream cases.
The hardware fixture, mutation, final preflight, fresh review, and operator rerun remain pending at this checkpoint.

## Stop conditions

Return `NEEDS_CONTEXT` when binding fixture bytes, generated code, primary runtime, or required GPU authority is unavailable.
Continue independent source, CPU, and spec work when a hardware capture awaits the operator.
Return `NEEDS_DECISION` only if executing evidence contradicts the fixed arithmetic or requires a new shared-seam exception.
Do not weaken exact comparisons, alter cohorts, switch the denominator to eager, or widen all residual storage.
A known correctable token mismatch keeps the gate failing and the enclosing campaign active.
Do not mark this row `DONE` or publish accepted performance before its required gates pass.
Add `## Outcome` when the row reaches `DONE`, recording measured behavior and rejected alternatives.
