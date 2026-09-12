# Native BF16 grouped MoE on ROCm

Owning row: `BACKEND-ROCM-BF16-MOE`

Owner: the `BACKEND-ROCM-BF16-MOE` implementer, reviewer, and coordinating operator.
Parent row: `BACKEND-ROCM`, whose lifecycle stays `ACTIVE`.
Issue: [#3094](https://github.com/mudler/vllm.cpp/issues/3094), local record
`ISSUE-GH-3094`.
Origin: [#1928](https://github.com/mudler/vllm.cpp/issues/1928), local record
`ISSUE-GH-1928`.
Base: `6db4bef906859e864c82523c01107473f7dcca29`.
Integration: one pull request, with this spec committed before implementation.
The implementation pull request carries closing keywords for both issues.
This session ends with a published, independently reviewed pull request.
The operator does not merge it without separate developer authorization.

## Now

State: `ACTIVE`. Native providers reach the registered Qwen3 MoE forward path on
`gfx1100`. The legacy grouped suites and eight native boundary, graph, stream,
and two-device cases pass. The production token gate passes under the corrected
whole-sequence-membership rule; the superseded same-configuration comparison
differed at six generated positions across three repeated length-33,
concurrency-2 runs.
The first proved residual-normalization difference belongs to
[#3103](https://github.com/mudler/vllm.cpp/issues/3103), under `BACKEND-ROCM-RESIDUAL-NORM`.
Attention differences (#3115) also require resolution before acceptance; the
head-output boundary is implemented and gated under
[the child spec](rocm-lmhead-bf16.md).

All 60 original upstream component cases pass on both runtimes.
Initial fresh review found three missing test witnesses at `94b8bb0ec`.
The scoped repair covers provider subsets, malformed descriptors, and accepted
weighted and shared numeric modes. Fresh scoped review and the final operator
gate remain pending. Performance is not accepted before a paired decode and
prefill measurement is recorded at this head.

## Problem and scope

ROCm registers neither `kMoeGroupedGemmBf16` nor
`kMoeGroupedGemmBf16GateUpSilu` at the base revision.
The existing BF16 mixture of experts (MoE) fast path therefore remains unavailable
on ROCm. Its eligibility check probes the unfused operation but calls both operations.
The two providers must become available together.

Implement native HIP providers for grouped BF16 matrix multiplication and fused
gate, up, and SiLU multiplication. Route the feature through the existing shared
operations and Qwen3 MoE production path. Preserve resident expert pointer arrays
and the device router output. Do not add a host gather loop.

The pinned ROCm oracle rounds intermediate BF16 tensors differently from the
existing CUDA sibling contract. Add the minimum shared numeric modes needed to
represent those differences. Preserve the current default semantics for every
existing caller that does not select the new mode.

This work is independent of the RDNA3 quantized dot changes in pull request
[#2782](https://github.com/mudler/vllm.cpp/pull/2782). BF16 expert weights do not
reach the quantized RDNA3 dot kernel. Do not depend on that pull request or
change its implementation. Do not change continuous integration configuration,
checker semantics, or the upstream pin.

This row adds a backend capability to an existing model path. It does not add
a model architecture or a quantized arm. GGUF, FP8, integer quantization, expert
parallelism, LoRA, and expert load balancing remain outside this change.
Do not alter their acceptance, refusal, or numeric behavior.

## Inventory

The source anchors in this spec use the immutable revisions stated here.
Line numbers refer to those revisions, before implementation changes them.

| Stable ID | Upstream source | Local anchor | Required test and evidence | State |
|---|---|---|---|---|
| `ROCM-BF16-MOE-GROUPED` | `vllm/model_executor/layers/fused_moe/fused_moe.py:299-610,763-910` | `include/vt/ops.h:3198-3238`, `src/vt/rocm/rocm_moe_grouped_bf16.hip` | `test_ops_moe_grouped_bf16`, `test_rocm_moe_grouped_bf16`, oracle buffers and generated kernel | `ACTIVE` |
| `ROCM-BF16-MOE-GATEUP` | `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:388-409,487-527`, `csrc/libtorch_stable/activation_kernels.cu:44,165-177` | `MoeGroupedGemmBf16GateUpSiluNative` and legacy typed sibling | `test_ops_moe_grouped_bf16_gate_up_silu`, exact BF16 witnesses, `test_rocm_moe_upstream` | `ACTIVE` |
| `ROCM-BF16-MOE-WEIGHTED-DOWN` | `vllm/model_executor/layers/fused_moe/fused_moe.py:593-610`, `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:395-459` | `MoeGroupedGemmBf16Weighted`, `MoeCombinePreweighted` | `test_rocm_moe_grouped_bf16`, route-before-narrowing witness and combine checks | `ACTIVE` |
| `ROCM-BF16-MOE-FORWARD` | `vllm/model_executor/models/qwen3_moe.py:199-237` | `src/vllm/model_executor/models/qwen3_moe_registry.cpp:63-86`, `src/vllm/model_executor/models/qwen3_5.cpp:6869-7010,7198` | `test_rocm_moe_bf16` through load/forward, provider statistics, exact tokens and call-site mutation | `ACTIVE` |

All inventory items belong to this spec and `ISSUE-GH-3094`.
The implementation evidence below distinguishes measured results from remaining gates.

## Upstream contract

### Pin and executing chain

The primary oracle is vLLM at
`e126687a9a828d513c01a07cd69f025f27d63280`.
[Upstream sync](../upstream-sync.md) owns the repository pin.
The inspected checkout contains that exact revision.
Use the operator's recorded oracle runtime and GPU authority from the shared
environment. Machine paths in another spec are not defaults.

The executing source chain is:

1. `vllm/model_executor/models/qwen3_moe.py:199-237` constructs and invokes
   `FusedMoEFactory` for the routed experts.
2. `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:41-130`
   resolves the backend, creates BF16 expert weights, and defines ROCm padding.
3. `vllm/model_executor/layers/fused_moe/oracle/unquantized.py:61-65,208-325`
   orders and selects the unquantized backends.
4. That file's `:329-395` converts weight storage and builds the prepare,
   finalize, and expert operations.
5. `vllm/model_executor/layers/fused_moe/modular_kernel.py:1144,1349`
   allocates activation storage in the model dtype and invokes the expert method.
6. `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:309-330,388-527`
   executes gate/up, activation, and weighted down through BF16 intermediates.
7. `vllm/model_executor/layers/fused_moe/fused_moe.py:763-910` launches the
   kernel with explicit tensor strides and the selected compute type.
8. `vllm/model_executor/layers/fused_moe/fused_moe.py:517-610` accumulates in
   FP32, applies optional route weights, narrows, and stores.
9. `vllm/model_executor/layers/fused_moe/activation.py:196-237` selects
   `torch.ops._C.silu_and_mul` for SiLU.
10. `csrc/libtorch_stable/activation_kernels.cu:44,110,165-177,299` defines
    the BF16 activation and multiplication boundaries.
11. `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:395-459,759` sums
    already weighted BF16 expert results with an FP32 accumulator.

ROCm's source candidate order is AITER, Triton, then batched Triton.
`vllm/_aiter_ops.py:134,1891` restricts the AITER capability to qualifying CDNA
devices. `gfx1100` is RDNA3. Source inspection therefore predicts ordinary Triton
for this unquantized, single-device workload. This prediction is not runtime evidence.

Before choosing the provider's native default, capture the selected oracle backend,
generated kernel, launch arguments, tensor dtypes, shapes, and strides.
Run the identical workload in the pinned engine. A successful config construction
or an unrelated dense-model capture does not satisfy this gate.
If runtime selection differs, reconcile the executing source before implementation.
Do not replace production defaults with `--enforce-eager` for a denominator.

### BF16 boundaries

For the target path, `apply_router_weight_on_input` is false.
Let `b(x)` mean conversion to BF16 with the oracle's rounding behavior.
For token `t`, selected expert `e`, and route weight `r`, the native semantics are:

```text
g = b(dot_fp32(x[t], Wgate[e]))
u = b(dot_fp32(x[t], Wup[e]))
s = b(silu(float(g)))
a = b(float(s) * float(u))
d = b(dot_fp32(a, Wdown[e]) * float(r))
y[t] = b(sum_in_fp32(d for each selected expert))
```

The dot product's order follows the selected kernel and receives the upstream
comparison tolerance. The BF16 conversion points are mandatory independent of
that tolerance. A token match does not prove the memory or arithmetic format.

The existing CUDA fused sibling computes its gate/up intermediates in FP32.
That legacy mode remains byte-identical to its existing unfused composite.
Its FP32 intermediates are a compatibility exception to the native BF16 mode.
Annotate that reason beside each model-path buffer that keeps FP32 storage.
Do not silently make the old composite the new oracle.

The native down operation multiplies the route weight before narrowing to BF16.
The combine operation sums these preweighted values without multiplying again.
Post-store weighting is numerically different and fails the native contract.

### Weight storage and shapes

Upstream stores gate/up weights as `[E,2I,H]` and down weights as `[E,H,I]`.
Its ROCm padding can add 128 BF16 columns when a row occupies a multiple of
512 bytes. The default `VLLM_ROCM_MOE_PADDING` is enabled in `vllm/envs.py:1341`.
Triton conversion preserves the physical strides in
`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:364-370`.
Record those strides in the oracle capture and retain padding cases in the tests.

The local shared ABI uses one device pointer per expert.
Each pointer names a Matmul-B matrix `[K,N]`, indexed as `k * N + n`.
The loader's transpose from checkpoint storage is an existing harness adaptation.
Do not reinterpret a checkpoint `[N,K]` array as the shared matrix format.
The local ABI does not require unused physical padding bytes to match upstream.
It does require BF16 storage, correct logical values, and explicit recorded strides.

## Shared design

### Operations and selection

Register both existing operation IDs for `DeviceType::kROCM` in the same change.
Keep their current typed call signatures and default semantics intact.
Add an explicit shared numeric descriptor or typed sibling operations for native
gate/up rounding, weighted down, and preweighted combine.
If a descriptor changes a function-pointer signature, use a separate typed sibling
instead of casting an old provider to a new signature.
The exact C++ names are implementation choices. The semantic modes are fixed here.

The Qwen3 MoE path selects the complete native capability through the shared
provider seam. Do not add a model-specific HIP kernel or a device-type branch
that changes model arithmetic. Probe every operation required by the native mode.
Never select half of the native sequence and continue with legacy weighting.
Existing callers that do not select this capability keep their defaults.

The existing row map maps pair `p` to its activation row.
A null row map means identity. Expert IDs and row maps remain on the device.
Inputs are contiguous BF16 activations, contiguous I32 indices, and I64 device
pointer arrays. Outputs accept the existing grouped BF16 and FP32 modes.
Preserve validation in `src/vt/ops.cpp:906-962`.
Preserve the zero-pair and zero-output-width no-op behavior.
Use valid router-produced expert IDs. This row does not add sentinel index semantics.

The fused operation accepts separate gate and up pointer arrays.
Use one grouped provider implementation for all callers, including the legacy mode.
Mask incomplete K and N tiles. Preserve repeated experts, empty experts, arbitrary
pair order, repeated rows, and dimensions that are not tile multiples.
Do not assume a particular top-k, expert count, or model geometry.

### Fusion and production reachability

The existing pointer-array BF16 sibling is a tracked exception to a literal
`vt::MergedGemmGroup` instance.
`include/vt/merged_gemm.h:111-118` documents its different weight representation.
[The fusion spec, Tier A4](arch-fusion-fold-plan-2026-07-30.md) records the
shared operation and its original compatibility contract.
This row extends that shared operation with explicit native numeric modes.
It does not create another exception or a per-model expert loop.
Update that local seam comment to distinguish legacy and native numerics when
the implementation introduces the mode.

Model fusion continues through `vt::FusedChain` where that seam applies.
Mergeable dense and shared-expert projections continue through
`layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`.
Do not duplicate their existing model orchestration for this backend.

The production vehicle is `Qwen3MoeForCausalLM` loaded from safetensors.
`src/vllm/model_executor/models/qwen3_moe_registry.cpp:63-86` exposes load and forward.
`src/vllm/model_executor/models/qwen3_moe.cpp:71` delegates to the shared MoE block.
`src/vllm/model_executor/models/qwen3_5.cpp:7198` selects the grouped BF16 path.
The resident gate/up and down calls occur at `:6997-6999`.
The existing `VT_MOE_BF16_FAST` default is enabled at `:909`.
Preserve the rollback setting and validate both sides with the same binary.

DeepSeek and dots3-note contain callers of the shared BF16 operations.
Their production routers require currently refused ROCm modes.
`src/vt/rocm/rocm_moe_router.hip:183` rejects grouped routing, correction bias, and non-softmax
scoring. This row does not claim that either complete model becomes runnable.
Keep that router debt under the parent `BACKEND-ROCM` issue
[#41](https://github.com/mudler/vllm.cpp/issues/41).

### Kernel and scratch lifetime

Place new hardware code in `src/vt/rocm/rocm_moe_grouped_bf16.hip`.
Use `src/vt/cuda/cuda_matmul_nvfp4.cu:943-1004,1413-1573` as a local
layout and deterministic-reduction donor. Its legacy arithmetic is not the
native oracle contract. Attribute every ported upstream kernel section at the pin.

Start with a complete deterministic kernel. Optimize only after correctness passes.
If split-K is used, reduce partials in a fixed order and apply conversion once
at the specified boundary. Do not use atomic output accumulation that changes
results across identical runs.

Key reusable scratch by both device and stream. The null default stream does
not identify a unique device. Serialize allocation and publication across threads.
Preserve allocation lifetime until all users and captured graphs finish.
`src/vt/grow_only_stream_scratch.h` supplies the existing growth and retirement
mechanism. Adapt that mechanism instead of copying CUDA globals with weaker keys.
Allocation failure must not publish partial capacity or invalid pointers.

Prewarm the required capacity before graph capture.
Do not allocate, free, or synchronize during capture.
Preserve captured addresses when another invocation grows scratch.
Validate streams independently and devices separately when two devices are available.
A missing second device leaves that case `PENDING`, not silently skipped.
Direct operation capture tests do not enable model graph capability.
This row does not depend on the separate graph-capability pull request #2777.

## Tests and evidence

### Red before implementation

Create a deterministic tiny safetensors fixture through the existing model loader.
Adapt `tests/vllm/models/test_moe_async_device_ids.cpp:111-203,415-477`.
Enter through `ModelRegistry::Load` and `ModelRegistry::Forward`.
Do not construct an internal MoE block by hand for the reachability claim.

Use this concrete Qwen3 MoE configuration:

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 128,
  "intermediate_size": 128,
  "moe_intermediate_size": 128,
  "shared_expert_intermediate_size": 0,
  "num_experts": 4,
  "num_experts_per_tok": 2,
  "norm_topk_prob": true,
  "decoder_sparse_step": 1,
  "mlp_only_layers": [],
  "hidden_act": "silu",
  "vocab_size": 128,
  "max_position_embeddings": 256,
  "rms_norm_eps": 0.000001,
  "rope_theta": 10000000.0,
  "tie_word_embeddings": false,
  "attention_bias": false,
  "torch_dtype": "bfloat16",
  "bos_token_id": 1,
  "eos_token_id": 127,
  "pad_token_id": 0
}
```

Preserve `BuildTensors` ordering from the fixture at the base revision.
Set its first tensor seed to 7 and increment the seed for each tensor.
Preserve `Bf16Bytes` at `:131-142`, including unsigned 32-bit wraparound and
the existing `F32ToBF16` conversion. Keep projection scale 0.08 and norm scale 0.5.
Only substitute the dimensions and initial seed stated here.
Export the generated safetensors and config once, then use those same bytes in
both runtimes. Record each file's byte count and SHA256.

For each length `L` in `{1,3,33}`, request `r` contains token IDs
`1 + ((11 + 29*r + 17*i) % 126)` for `i` from 0 through `L-1`.
Run request 0 at concurrency 1 and requests 0 and 1 together at concurrency 2.
Repeat each workload 3 times with fresh model state.
Use token-ID prompts with tokenizer initialization skipped on the oracle.
Generate exactly 8 tokens per request through the existing shared device sampler.
Set temperature 0, top-p 1, top-k -1, min-p 0, repetition penalty 1,
presence penalty 0, frequency penalty 0, seed 7, and `ignore_eos=true`.
Set both the minimum and maximum generated token counts to 8.
Record selected expert IDs and assert changing expert pairs across the request set.
Keep repeated expert choices in the component fixtures.

Assert provider selection for grouped down and fused gate/up.
Enable `OpProviderCallStats` and require native selections greater than zero,
zero native declines, and zero CPU fallbacks for the tested expert sequence.
On `gfx1100`, absence of the providers is a failure, not a skipped test.
The red result must fail that reachability assertion on the base revision.
Correct fallback tokens alone do not satisfy this test.

### Port upstream cases

Port `tests/kernels/moe/test_moe.py:test_fused_moe` at `:345` from the pin.
Preserve its seed 7, BF16 fixtures, `atol=0.02`, and `rtol=0`.
Retain these `[M,N,K]` parameter sets:

```text
[1,128,128]
[1,2048,128]
[33,2048,128]
[32768,2048,511]
[40000,1024,1024]
```

Retain expert counts 8, 64, and 192, top-k values 2 and 6, and both padding modes.
Run every applicable single-device BF16 combination from the upstream fixture.
The local pointer-array transpose is the required storage adaptation.
Expert parallel size 4 is inapplicable because this row adds no distributed path.
Tensor-descriptor mode is inapplicable on this hardware by
`vllm/model_executor/layers/fused_moe/utils.py:665-684`.
Record those exclusions with their source rationale.
Do not reduce large M or tail dimensions to make the tests fit.
Name the resource and retain the gate as `PENDING` if a required case cannot run.

Preserve the upstream reference decomposition, activation mode, route-weight
placement, and output comparison. Record generated kernels before interpreting
any unavailable lever. A scratch reference must be labeled as scratch evidence.
The upstream component fixture uses `renormalize=false`. Preserve that mode.
The production fixture uses `norm_topk_prob=true`. Preserve that mode separately.
Keep the upstream `use_compile=false` setting and its graph replay cases where
`N >= 1024` and `K >= 1024` on CUDA-alike hardware, including ROCm.

Add local cases from `tests/vt/test_ops_moe_grouped_bf16.cpp` and
`tests/vt/test_ops_moe_grouped_bf16_gate_up_silu.cpp`.
Cover row-map and identity-map inputs, BF16 and legacy FP32 output, large pair
counts, split-K candidates, K and N tails, repeated experts, and empty experts.
The legacy fused operation must remain byte-identical to its existing composite.
The native mode must match captured oracle boundaries within the upstream gate.
Do not compare native arithmetic only against the legacy composite.

Use boundary values that fail when each BF16 narrowing point is removed.
Use nontrivial route weights that distinguish weighting before and after narrowing.
Assert physical activation and expert-output dtypes independently of token results.
Compare greedy token IDs exactly for the complete production fixture.
Capture finite logits and their differences for diagnosis. Never widen the token
gate because an output difference appears numerically small.

### Scratch and negative mutations

Test repeated launches, simultaneous streams, scratch growth, allocation failure,
prewarmed graph capture, replay after growth, and zero-sized workloads.
Require capture and replay to preserve output and pointer lifetime.

A fresh reviewer applies each applicable mutation to an immutable scratch copy:

| Guarantee | Mutation that must fail |
|---|---|
| Both providers are reachable | Remove each registration separately |
| Production uses the new sequence | Delete its production call site but retain fallback |
| Native mode is selected | Force legacy mode without changing final fallback availability |
| Matrix and row layouts are correct | Transpose the weight stride or ignore the row map |
| Expert selection is correct | Substitute another expert or mishandle an empty expert |
| Tail masks are correct | Remove a K mask and an N mask separately |
| BF16 boundaries are correct | Remove gate/up narrowing and SiLU narrowing separately |
| Weight placement is correct | Move route weighting after down-output narrowing |
| Combine consumes preweighted output | Apply the route weight a second time |
| Scratch survives growth | Free a block still referenced by a captured graph |
| Scratch keys isolate users | Remove the stream or device key where hardware permits |

Record the focused command, nonzero result, first relevant failure, and byte-for-byte
restoration after each mutation. A source inspection does not replace these tests.

## Gates and acceptance

The implementer records exact commands after the test executable names are final.
The evidence must include these obligations:

1. Startup, role, and full `scripts/agent-preflight.sh --staged` at the implementation head.
2. A clean CPU build and existing shared-operation regression tests.
3. A HIP build for `gfx1100`, including the new source in the HIP compile options.
4. The focused native and legacy grouped suites with zero unexpected skips.
5. The pinned oracle's identical production fixture and exact token comparison.
6. Dtype, stride, backend-selection, and generated-kernel evidence on both sides.
7. Scratch, capture, concurrency, and negative-mutation results.
8. A fresh scoped review followed by the operator's own hardware gate.

Use the recorded lease or mutex required for the actual GPU.
Record device identity, ROCm and compiler versions, binary SHA256, revisions,
artifact hashes, launch recipes, sampling configuration, and contention state.
Do not perform GPU work outside the operator's recorded authority.

Only accept performance after the declared correctness gate passes.
Trace both runtimes with the same tool on identical inputs.
For each applicable decode and prefill case, record throughput, latency, peak
device memory, peak host memory, and ratios against the production oracle.
The throughput floor is 1.0 times the oracle. Latency and memory must not exceed
1.0 times the oracle. Keep an unmet axis open with its next traceable hypothesis.
Do not describe an unresolved implementation difference as an architectural ceiling.
Repeat accepted results on an idle device with the same-binary fallback comparison.

Report each gate as satisfied, narrowly waived, pending a named external authority
or resource, or failing. This row cannot become `DONE` with an unresolved required
axis. Add an `## Outcome` section when it reaches `DONE`, including measurements,
rejected approaches, and the reason for each default.

### Exact production token gate (corrected definition)

The gate certifies that the native engine computes the reference function, not
that it reproduces one execution of it. The reference's greedy decode is not
unique across its own legitimate configurations: at length 33, concurrency 1 and
concurrency 2 on the identical prompt, the pinned primary itself emits
`[66,1,70,57,33,81,63,69]` and `[66,1,70,57,33,81,118,66]` respectively (all
three repeats each, `oracle-selection-6/production.json`); the two runs agree on
the first six tokens and differ at positions 6 and 7 of request 0. The developer
ratified this definition on 2026-09-09. A workload passes when the native
sequence equals, for that request, a sequence the reference itself emits under
one of its captured configurations (whole-sequence membership). Positions at
which the captured reference configurations agree remain exact, because
membership in any single reference sequence implies they match. The gate reports
the reference set, the matched configuration, each same-configuration outcome,
and every position where the reference disagrees with itself. It never mixes
positions from different reference sequences, never drops a workload, and never
relaxes a stable position.

Fidelity is a separate criterion, and no gate in this change measures it. It
accepts a native-versus-reference logit difference of at most `1.953125e-3`
(`2^-9`) per logit, which is one BF16 unit in the last place at the logit
magnitude where that maximum was measured. That value is recomputed from the
retained BF16 head-logit diagnostic captures in
`/home/vikash/.cache/rdna3-moe-impl/preserved/oracle-diagnostic-1/`, comparing
request 0 of the concurrency-1 and concurrency-2 captures of one workload at
every step where the generated context still agrees, that is
`L1-C{1,2}-R0-head-0-logits.bin` together with
`L33-C{1,2}-R0-head-{0,1,2,3,4,5}-logits.bin` (BF16, vocabulary 128; request 0 is
row 0 of the concurrency-2 capture). The largest absolute per-logit difference is
exactly `1.953125e-3`, attained at length 33 steps 3, 4, and 5, which is one BF16
ulp at those logits' magnitude, and the greedy argmax is unchanged at every
compared step. The oracle's
own logprob deltas in `oracle-selection-6/production.json`, which reach `2.028e-3`
between its concurrency-1 and concurrency-2 records at positions where their token
sequences agree, are log-probabilities over the vocabulary and are not this logit
band. The attention and Q/K preamble parity work (#3115) and the BF16 LM-head
output boundary work (#3116, [the child spec](rocm-lmhead-bf16.md)) are judged by
this fidelity criterion, not by reproducing one configuration's tie-break.

## Files and authority

The implementation owns the new HIP source, its registration in `rocm_ops.hip`,
the corresponding CMake source and compile-option entries, and focused tests.
It can extend `include/vt/ops.h`, `src/vt/ops.cpp`, provider metadata, shared
MoE descriptors, and the documented fusion seam to represent the fixed numeric modes.
It can update the Qwen3 shared MoE dispatch to select the complete native mode.
Mechanical adapters in existing providers are allowed when required by the shared
API. Those adapters must preserve their current numerics and defaults.

The implementation updates this spec and the issue records as their state changes.
The parent matrix remains unchanged because its lifecycle does not change.
This per-row inventory records the new capability without another shared matrix write.
Update `docs/FEATURES.md` and the operation inventory in `docs/ROCM.md` when the
native backend capability ships. Document a changed command or configuration in
`docs/USAGE.md` only if the implementation changes that public surface.
This row's spec commit makes no shipped capability claim and owes no public rewrite.

## Implementation evidence, 2026-09-09

The implementation starts from committed spec `9202e4c4edc4cf6ef9b3e8da66431effb0fbcee5`.
Its isolated worktree is `/home/vikash/vllm.cpp-rdna3-moe-impl`.
The paths below identify this measured run; they are not environment defaults.
Evidence root: `build-rdna3-moe-hip/evidence/` in that worktree.
The operator holds `/home/vikash/gpu.lock` for every GPU invocation.
The measured device is a Radeon 7900 XTX, `gfx1100`.
Native compilation uses HIP `7.15.26333`, Clang 23, and `-ffp-contract=off`.
The pinned oracle reports Torch `2.12.0+git6bbd260` and HIP `7.2.53211`.

### Design and executing oracle

The shared API adds three typed siblings: native gate/up, weighted down, and
preweighted combine. The existing typed signatures and legacy arithmetic remain
unchanged. Capability selection requires all five grouped/native providers.
The HIP implementation uses one deterministic reduction per output and no scratch
allocation. A device scope selects the queue's device and restores the caller's
ambient device. The stream always comes from that queue.

The pinned production engine selects `TritonExperts` in both layers.
Captured gate/up and down inputs, weights, and outputs are BF16 with two-byte
elements. Gate/up weight strides are `[32768,128,1]`; down strides are
`[16384,128,1]`. The captured MoE kernel uses `BLOCK_SIZE_M=32`,
`BLOCK_SIZE_N=64`, `BLOCK_SIZE_K=128`, and `SPLIT_K=1` for the small fixture.
Its generated `fused_moe_kernel.amdgcn` contains BF16 WMMA instructions.
Native offload code and the compile command are retained in `native-generated/`.

The fixture export preserves the committed generator, including its tensor order:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `config.json` | 719 | `321926020ada026d8dd85f74543dd12b6426301706b4e1f4fbca60a44b871cea` |
| `model.safetensors` | 1123330 | `96cd7f30fee496c69782af2813e438e47d7b026598b0f20a05049c522b279af8` |

The final cohort harness uses pinned `LLM.sleep(level=0)`, `enqueue`, scheduling
wake-up, and `wait_for_completion`. It records the authoritative internal/external
request-ID mapping and asserts the actual prefill and decode batches for every run.
`oracle-selection-6/production.json` has SHA256
`f3d27a95ba71bddcae38c3defc21bf3f3ff32ba9373a09faf949ead68b74d7ad`.
Earlier oracle attempts and the initial uncoordinated scheduling result remain
in evidence. They do not supply the final matched-cohort denominator.

### Measured gates and remaining work

| Gate | Result and evidence |
|---|---|
| Red before implementation | Satisfied. `production-red-2.log` has four missing registration/selection failures and 27960 passing assertions. Its executable SHA256 is `25dfd4e7e58dbe6069c6d1ea20c00ae7249687336645c04389845ceede0e83ae`. |
| CPU build and shared regressions | Satisfied. Six tests pass: model registry, grow-only scratch, provider metadata, native descriptor validation, MoE operations, and grouped router. Log: `/home/vikash/.cache/rdna3-moe-impl/cpu-native-tests.log`. |
| HIP build and legacy arithmetic | Satisfied. Both existing grouped suites pass on ROCm: 7 cases/19 assertions and 3 cases/6 assertions, with no skips. Logs: `test_ops_moe_grouped_bf16-native-1.log` and `test_ops_moe_grouped_bf16_gate_up_silu-native-1.log`. |
| Native numeric boundaries, streams, capture, and devices | Satisfied. `native-boundary-3.log` records 8 cases/2804 assertions, no skips, with devices 0 and 1 visible. It covers opposite ambient device state for native gate/up, weighted down and combine. Executable SHA256: `8bae75c8506b08f2444ed0a4b8379b837608b9261baccdef3c71a97ad0408a33`. |
| Scratch-only allocation and retirement mutations | Narrowly waived for this scratch-free implementation. No allocation, free, capacity publication, retired block, or scratch key exists in these providers. Graph replay after larger shapes and concurrent streams still run. |
| Original upstream component cases | Satisfied. All 60 cases pass on both runtimes, including M=32768/K=511 and M=40000/K=1024 graph cases. The unchanged pinned test supplies seed 7, BF16 fixtures, both padding modes and original tolerances. Raw padded source storage and logical exported strides are retained separately in `upstream-all-2` and its range directories. `upstream-all-2-complete-operator-summary.json` independently checks all 120 case/stage results; its SHA256 is `b5c9d8259ae6c4701ad92647f5e61bdda7ba1ef6e2a2fd6ecb8a9edf99a4af04`. |
| Production provider selection | Satisfied. All three new operations have positive native selections, no declines, no fallbacks, and no CPU selections. The existing two registrations are present. |
| Exact production tokens | Satisfied under the corrected whole-sequence-membership rule (`### Exact production token gate (corrected definition)`, ratified 2026-09-09). Measured native tokens match a captured reference sequence at all 18 workloads, no workload matches NO member, and native request 0 equals the concurrency-1 reference sequence at every length. Request 1 exists only in the concurrency-2 records and matches there. Historical note on the superseded same-configuration comparison: the pinned primary disagrees with itself at length 33/concurrency 2/request 0, where its concurrency-1 record emits `[66,1,70,57,33,81,63,69]` and its concurrency-2 record emits `[66,1,70,57,33,81,118,66]`, at positions 6 and 7 of request 0 on all three repeats. The native run equals the concurrency-1 member token-for-token. The operator receipt for this corrected gate at this head is `/home/vikash/.cache/moe-6fd1650c4-tmp/operator-gpu-positive-da0ce377b.log`: exit 0, 28598 of 28598 assertions passing, every same-configuration outcome printed as a report (the 18 assertions beyond the pre-`REQUIRE` receipt are the request-count requirement, one per matched record). The operator's mutation study at this head restores the superseded same-configuration-only comparison and reddens exactly the three recorded failures; the helper header was restored byte-for-byte (`265e28fe`). The superseded-head red history that the corrected gate answers is `/home/vikash/.cache/residual-norm-repair1/green-cc9d4f565/fusion-1-operator.log`, which records 28541 passing assertions and the three failing same-configuration comparisons, one per repeat. `/home/vikash/.cache/moe-6fd1650c4-tmp/legacy-fused-token-agreement.json` (sha256 `9a97f18b7553c80a`) re-derives the legacy agreement at this head: `VT_FUSED_CHAIN_ADOPT=0` and `=1` both exit 0 for all 18 records and emit the same 216 generated tokens with no mismatching record. The earlier `baseline-production-tokens-comparison.json` was removed with the superseded 41 GB evidence tree during the campaign cleanup; this receipt replaces it. |
| CPU descriptor negative mutations | Satisfied. All nine mutations in `cpu-contract-mutations-2/results.json` fail their intended descriptor or capability assertion. Original source and archive hashes remain equal after each run. The first preparation linked the unchanged CPU whole archive, so that invalid probe is preserved and excluded. |
| Implementer negative mutations | Satisfied. The operator ran all 20 isolated mutations in `negative-mutations-1/run-recipes.json`; every intended defect was detected, with no survivor or timeout. Original source, archive and executable hashes remain unchanged. Each of the eight production mutations adds its specific provider failure beyond the existing token failures. Filtered component mutations select one test with nonzero assertions. `operator-results.json` and per-case receipts preserve the commands and failures. |
| Full staged preflight | Satisfied for executed checks: exit 0, no failures, and 619/619 affected host translation units compiled. The report lists 12 explicit skips, reconciled below; it does not print an all-green claim. Log: `/home/vikash/.cache/rdna3-moe-impl/staged-preflight-1.log`. |
| Supplementary preflight suites | Satisfied. All seven NumPy-dependent suites pass using the existing NumPy 2.1.3 package through task-local links. Both CPU and HIP build databases pass the x86 ISA audit. The first full-site Python path exposed installed vLLM metadata to fake-runtime tests; that failed environment attempt is preserved and excluded. |
| ARM and CUDA build audits | Narrowly waived for this gfx1100 change: no ARM build, CUDA fat binary, or CUDA Triton AOT artifact is produced. The native HIP build and generated gfx1100 object supply backend build evidence. |
| Frozen-head PR path, trailers and style | Pending the local implementation commit and the operator's exact-SHA pre-push checks. |
| Fresh review and final operator gate | Pending the fresh reviewer and coordinating operator at the immutable implementation head. |
| Paired traces, throughput, latency and memory | Failing acceptance prerequisite: no paired decode or prefill measurement is recorded at this head. No performance result or floor is accepted. |

Four additional scratch replays use original captured production MoE inputs,
expert IDs, BF16 weights and route weights for both layers at L33/C1 and L33/C2.
All pass the original absolute tolerance of 0.02. Activated outputs differ in
3/1/5/1 BF16 words, weighted down in 7/0/22/1, and final sums in 2/0/10/1.
Maximum final difference is `6.103515625e-05`. The final comparator is a CPU
reconstruction of the sum of captured weighted BF16 down values. It is not an
observed final Triton tensor. These scratch runs isolate expert arithmetic;
they do not replace the original upstream cases or exact production tokens.

Paired dense captures locate differences before the new experts. The first QKV
input, weights and output match exactly. Attention output first differs at row 1.
Row 0 attention projection remains exact, but post-attention normalization differs
in 33 of 128 values. `norm-gap-handoff/handoff.md` and its manifest preserve the
complete executing compiled lifetime for #3103, including all six generated
modules and their allocation/store dtypes. Their SHA256 values are
`9850ee49c93bb82ddfca1a811fb422337c33ea09335e749fbf3c42dfcaa0400a` and
`8531cb41f0168d27455b2dc8062a57d5db7afd4e0822013332f398bbf2239c65`.
The operator independently reran the algebra witness and checked every manifest hash.

## Test repair after fresh review

Fresh review of `94b8bb0ec67eff82d2860cdda1069e4b0a64bac8` found three P2
coverage gaps. Static review found no additional arithmetic defect.
The review report remains at
`/home/vikash/.cache/rdna3-moe-review/evidence/review-findings.md`.
The repair worktree is `/home/vikash/vllm.cpp-rdna3-moe-repair1`.
Its evidence root is `/home/vikash/.cache/rdna3-moe-repair1/evidence`.
These paths identify retained measurements. They are not environment defaults.

The repair changes two test files and this row's records. Product sources,
CMake, original fixtures, and tolerances remain unchanged.
`protected-before.json` records the product and original archive hashes.

### Added witnesses

The shared contract test checks all 32 availability subsets of the five providers.
It uses uniquely named XPU stubs and the existing provider-disable seam.
Each subset verifies every `OpRegistered` result before checking the complete set.
The cases include the empty set, the legacy-only set, and each missing provider.
The test never dispatches a stub or changes a production device's providers.

Malformed descriptors change one property per subcase.
The new cases cover route rank, stride, and device, and weighted down's common validator.
They cover both combine tensors' strides and devices.
Shared-input cases cover rank, both dimensions, stride, and device.
Each case requires its specific validation error before provider dispatch.

Weighted numeric cases use both output dtypes, nonidentity row maps, and per-pair routes.
A route of 257/512 produces exact FP32 values that BF16 output must narrow.
The shared-input case adds BF16 values into both output types.
Its FP32 result retains 257/512; BF16 rounds that halfway value to 0.5.
Fixed expected values come from exact rational arithmetic, independent of the provider.
`numeric-witness-v2.json` retains that calculation and the counterexamples.

### Repair verification

The repair recreates the reviewer's retained mutations in fresh sources and archives.
The new tests supply the red gates. The unchanged product supplies the green controls.
The CPU mutation set reports ten intended failures across ten distinct binaries.
The conjunction-to-disjunction mutation fails all 30 partial provider subsets.
Each of the nine descriptor mutations fails its corresponding error assertion.
`cpu-mutations/results.json` records commands, filters, exit statuses, and hashes.
Each mutation keeps its changed source and archive in a separate directory.
The original product sources and archives stay unchanged.

A clean CPU configuration and build pass the six declared regressions:
model registry, grow-only scratch, provider metadata, native descriptors,
MoE operations, and grouped router. The coordinator independently confirms
72 passing cases and 2910 assertions in `cpu-operator-results.json`.
The model-registry executable retains its previously disabled `can_initialize` case.
No added contract case skips. Commands and results are retained in
`cpu-configure.log`, `cpu-build.log`, `cpu-test-rebuild.log`, and `cpu-positive.log`.
The focused CTest command is:

```sh
ctest --test-dir build-repair-cpu --output-on-failure -R '^(test_model_registry|test_grow_only_stream_scratch|test_op_provider|test_moe_bf16_native_contract|test_ops_moe|test_ops_moe_router_grouped)$'
```

HIP test compilation preserves the original compiler options and relinks the
unchanged production archive. Its SHA256 is
`a9f57617d3ef79029b2529b216bb370cbd2fc41fa89b1f3f992c5a9c0a7433eb`.
This is a test relink, not a new full HIP product build.
The first GPU batch passes its three controls and detects all four review survivors.
Its frozen sources, commands, binaries, and operator receipts remain under `hip/`
and `hip-mutations/`. The second version strengthens F32 precision witnesses.
The second version passes all three positive controls: 219 contract assertions,
72 weighted-mode assertions, and 42 shared-mode assertions.
All four review survivors fail their intended assertions.
An additional FP32-store narrowing mutation fails both precision witnesses.
`gpu-v2/hip-mutations/run-recipes.json` retains each command and binary hash.
`operator-results.json` and `operator-input-audit.json` in that directory
record the coordinator's results and unchanged source, archive, and binary hashes.
No implementer GPU execution occurred. The coordinator runs all hardware commands
under `flock -n -F /home/vikash/gpu.lock` with the recorded visible-device environment.

The pre-edit full preflight exits 1 with 619/619 host units compiled.
Its only failure is an unchanged symbol-anchor fixture inheriting the parent Git repository.
The isolated rerun passes all 22 tests.
The staged gate uses `GIT_CEILING_DIRECTORIES` to isolate those temporary repositories.
It also uses `GIT_CONFIG_GLOBAL=/dev/null` for the known onboarding fixture.
An external Python wrapper appends `--jobs 4` only to `check-tree-compiles.py`.
All other checker arguments remain unchanged.
The seven NumPy suites use the existing isolated package directory.
The staged full preflight exits 0 with 619/619 host units compiled.
`staged-preflight.log` and `staged-preflight.exit` retain the complete result.
Its five argument-required skips have the dispositions recorded here.
The final receipt-only record edits receive focused record, anchor, and command checks.

The CPU and HIP compile databases pass the x86 ISA audit.
Exact path classification uses the immutable repair commit. Its result is retained
in `path-classification.log` before handoff.
ARM, CUDA fat-binary, and CUDA Triton AOT audits are narrowly waived:
this test repair produces none of those artifacts.
Fresh scoped review and the operator's final verification remain pending.

The original 60 upstream cases and production-token evidence remain authoritative
for the unchanged implementation. This test repair does not rerun that oracle matrix.
The exact model gate failed at six generated positions across three repeats under the
superseded same-configuration comparison, and it passes under the corrected
whole-sequence-membership rule above.
Agreement between all 216 native and legacy tokens does not establish oracle parity.
No performance result is accepted, and the row remains `ACTIVE`.

## Owed

No new unowned issue is introduced by this spec.
The origin issue keeps its existing single owner in
[the device-fit spec](gguf-device-fit-expand-policy.md#owed) until this
implementation closes it. That owner links here for the implementation handoff.
The new row owns its tracking issue directly.
Grouped routing and correction-bias work stays under `BACKEND-ROCM`, issue #41.

- [#3100](https://github.com/mudler/vllm.cpp/issues/3100), owned by
  `BACKEND-ROCM`, tracks shared resource operations that ignore their device index.
  The two-device test selects the resource device in its host setup, checks pointer
  ownership, and then launches each provider with the opposite ambient device.
  This isolates the new provider's device contract without changing shared allocation.
- [#3103](https://github.com/mudler/vllm.cpp/issues/3103), owned by
  `BACKEND-ROCM-RESIDUAL-NORM` in [its repair spec](rocm-residual-norm.md), tracks the compiled Qwen3 MoE residual-normalization lifetime.
  The paired row-zero witness reproduces all 128 native and oracle values separately.
  The native path rounds the residual before variance; the compiled oracle elides
  the post-attention residual store and materializes the next input-norm residual.
  Every materialized residual remains BF16. This row does not change that shared
  normalization path. Its production token gate passes under the corrected rule above.

## Stop conditions

Return `NEEDS_CONTEXT` if the oracle runtime, model bytes, or GPU authority is
unavailable for the next gate. Continue independent CPU work within the spec.
Return `NEEDS_DECISION` if the executing oracle requires semantics outside the
fixed shared modes or if a new shared-seam exception is required.
Do not change unrelated routers, quantized kernels, model architecture, or checkers.
Do not suppress a correctable failure or mark a fallback-only result as native.
Keep a blocked performance axis visible without stopping correctable work.
