# Reconcile GLM-5.3-Flash onto vLLM

Row: `MODEL-MM-GLM53-FLASH`

Issue: [#3045](https://github.com/mudler/vllm.cpp/issues/3045).
Campaign: [glm5-next-flash.md](glm5-next-flash.md).
Base: `e2fb2f06d9944c4bbe66531034479d370df67815`.

## Scope

Reconcile the campaign records after vLLM registered GLM-5.3-Flash.
Expire the transformers algorithm exception under its stated condition.
Preserve its pin and historical evidence. Record the upstream surfaces that
the next device implementation must reconcile.

This change edits this spec, the campaign spec, and the transformers oracle
record. All three are documentation. It changes no product, test, script,
policy, checker, generated file, or continuous integration configuration.
The campaign lifecycle remains `ACTIVE`.

## Upstream anchors

vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) merged
on 3 September 2026 at `98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5`.
The model registry at that object registers `Glm5NextForCausalLM`,
`Glm5NextForConditionalGeneration`, and `Glm5NextMTPModel`.
The global parity pin remains
`e126687a9a828d513c01a07cd69f025f27d63280`, which lacks these registrations.
The merged revision is a fixed source reference ahead of that pin. It is not
an accepted parity denominator or an advance of the global pin.

## Design

Add an explicit current reconciliation before the campaign's original oracle
survey. Label the survey as historical. Update its current `## Now` and
preserve the preceding status as dated history.

Add expiry fields to the `glm5_next` lane and retain its original acceptance,
scope, pin, and evidence. Follow the `qwen4_exp` expiry record's shape.
The expiry changes algorithm authority to vLLM. It does not erase earlier
component comparisons or establish model gateability.

Record source anchors for configuration, model composition, mixture of
experts (MoE), attention, KDA, k-pool caches, and their AMD dispatch.
Name applicable upstream tests and the remaining device-port obligations.

## Risks

Registration does not prove that the model builds or runs. Keep the model
gate `PENDING` until a source build runs the real checkpoint.
Do not infer the resolved runtime dtype, backend, or memory fit from source.
Do not turn historical memory measurements into a permanent hardware ceiling.
The device implementation must reconcile defaults and errors before claiming
equivalence. Numerical agreement alone cannot establish matching memory formats.

## Reconciled source surfaces

Every upstream anchor in this section refers to `98ed0856f3`, not the global
parity pin. These are source observations. No runtime dispatch or numerical
equivalence is claimed. The existing C++ component comparisons remain evidence
against their recorded transformers revision until each vLLM comparison runs.

| Surface | Upstream anchor | Consequence for the next port |
|---|---|---|
| Registration | `vllm/model_executor/models/registry.py:123`, `:417`, `:674` | Three registered architectures end the absence premise |
| Platform entry | `vllm/models/glm5next/__init__.py:9` | Both platforms enter the model under `nvidia/` |
| Configuration | `vllm/transformers_utils/configs/glm5_next.py::Glm5NextTextConfig`, line 11 | Reconcile aliases, defaults, layout, and refusals |
| Layer composition | `vllm/models/glm5next/nvidia/model.py::Glm5NextDecoderLayer`, line 278 | KDA selection, shared mHC ops, and merged MLPs define the device structure |
| Routed experts | `vllm/models/glm5next/nvidia/model.py::Glm5NextMoE`, line 151 | Mirror `FusedMoEFactory`, router dtype, grouped selection, and scale placement |
| MLA | `vllm/models/glm5next/nvidia/attention.py::Glm5NextMLAAttention`, line 404 | Use shared MLA, explicit norm epsilon, and the checkpoint's NoPE mode |
| Indexer cache | `vllm/models/glm5next/nvidia/attention.py::Indexer`, line 208 | Reconcile quantized index cache, bf16 tail cache, and merged projections |
| KDA | `vllm/models/glm5next/nvidia/kda.py::Glm5NextLinearAttention`, line 126 | Reconcile prefill, recurrent decode, and speculative state updates |
| K-pool dispatch | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py::SparseAttnIndexerKpool`, line 880 | Read AMD and NVIDIA providers selected by platform |
| ROCm sparse MLA | `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py::fit_kpool_indices_to_aiter`, line 66 | Preserve tail tokens when fitting the AITER width |

### Configuration and memory formats

The upstream config resolves `mlp_layer_types` from `first_k_dense_replace`
when the explicit layout is absent, at line 181. Its default is 0, at line 42.
The old campaign's hardcoded-three transformers observation does not define
this vLLM behavior. Compare the checkpoint's resolved layout before porting.

Lines 114 and 128 reject disabled indexer layer normalization, compression,
tail selection, and unsupported mHC normalization options. Preserve the five
failure cases in `tests/transformers_utils/test_config.py:54`.

`Glm5NextMoE` defaults `apply_routed_scale_to_output=False`, at model line 158.
Its factory receives `routed_scaling_factor`, at line 239. Follow the factory
and selected quantization method before claiming the scale's executed placement.
`Glm5NextMLP` uses `MergedColumnParallelLinear` and the optional
`SiluAndMulWithClamp`, at lines 116 and 139.

The decoder constructs MLA with `quant_config=None`, at model line 330,
because the checkpoint stores those projections in bf16. The indexer stores
compression positional parameters in fp32 and compression gate weights in bf16,
at attention lines 241 and 246. Its index cache holds FP8 values and scales
inside a uint8 view, at line 283. The tail cache is bf16, at line 294.
These are upstream format observations, not a license to widen model buffers.

KDA obtains state dtypes from `MambaStateDtypeCalculator.kda_state_dtype`,
at KDA line 136. Prefill casts and sigmoids beta in fp32 before
`chunk_kda_with_fused_gate`, at line 537. Decode uses `fused_recurrent_kda`,
at line 571, with the gate and beta transformation inside the kernel.
The import at line 41 selects `amd/ops/third_party/kda` on ROCm and the
corresponding `nvidia/` package otherwise. Both packages own `kernels.py`
and `fused_recurrent.py`. Port their applicable state and rollback semantics.

The mHC path calls shared `MHCPreOp`, `MHCPostOp`, and
`MHCFusedPostPreOp`, at model lines 402 to 404. It passes norm weights to the
pre-operation at line 522. Final stream contraction calls `hc_contract`,
whose definition at `vllm/model_executor/layers/mhc.py:663` averages streams.
That confirms the prior unweighted-collapse choice at the source level.

### K-pool and backend tests

The indexer selects `amd/ops/kpool_compress.py` on ROCm, at
`sparse_attn_indexer_kpool.py:20`. The other branch selects the NVIDIA file.
Both implement compression, paged-tail seeding, batched decode updates, pool
expansion, and tail appending. In the NVIDIA file, the corresponding entry
points begin at lines 258, 411, 612, 713, and 753.
The CUDA indexer requires DeepGEMM, at indexer line 918. Source presence does
not establish that its kernels execute on a particular device.

Port the applicable upstream tests with their parameters and tolerances in
each implementation change. These suites define the first reconciliation work:

| Upstream test surface | Required cases |
|---|---|
| `tests/kernels/test_kpool_decode_update_batched.py:237` | AMD preshuffled cache layout and padded tail stride |
| Same file, lines 295 and 425 | Pool sizes 4 and 16, completion boundaries, padding, plain decode, and 20 fuzz seeds |
| `tests/v1/attention/test_kpool_tail_slot_mapping.py:39` | Circular mapping, empty batches, invalid slots, storage reuse, and interleaved request isolation |
| `tests/v1/attention/test_rocm_glm5next_sparse.py:26` | Best-history and tail preservation, exact-width aliasing, and narrow-input rejection |
| Same file, lines 60 and 102 | Six ROCm route cases, 512 NoPE dimensions, incompatible geometry, and signed-int32 offset overflow |
| `tests/v1/attention/test_sparse_indexer_decode_seq_lens.py` | Per-request decode lengths and sparse index selection |
| `tests/kernels/test_mhc_kernels.py:322` | ROCm fallback normalization and AITER fused normalization |
| `tests/kernels/mamba/test_gdn_prefill_flashinfer.py:22` | Int64 cumulative lengths for the shared FlashInfer prefill path |
| `tests/v1/worker/test_mamba_hybrid_model_state.py` | Hybrid state registration |
| `tests/v1/core/test_kv_cache_utils.py` | Compressed and tail-cache grouping |

The registration includes multimodal and MTP paths. Their source lies in
`nvidia/multimodal.py`, `nvidia/mtp.py`, and
`vllm/transformers_utils/processors/glm5next.py`. Their tests include
`tests/transformers_utils/processors/test_glm5next.py` and
`tests/multimodal/test_video.py`. Multimodal work remains campaign scope.
MTP remains the campaign's explicitly owed arm. This reconciliation implements
neither and does not remove their obligations.

## Next implementation sequence

1. Repair expert placement admission under #3019 using the committed repair spec.
2. For #2410, reconcile the reached config, formats, and upstream tests above.
3. Route the device forward through the shared MLA, fusion, MLP, and sampling seams.
4. Prove production reachability with a test that fails without each call site.
5. Build and run the real oracle under #1998 before declaring its model gateable.
6. Establish correctness before accepting any throughput, latency, or memory ratio.

The last two steps require measured artifacts, resolved runtime configuration,
and an available leased device. A missing resource keeps the named gate
`PENDING`. It does not establish a hardware ceiling or waive an axis.

## Tests and gates

The focused documentation check searches the oracle record for
`expired_by = vllm-project/vllm 98ed0856`.
It must fail before the edit and pass afterward. Removing that line in a
scratch copy must restore the failure. No new persistent checker is needed.

Read the merged registry and the parity-pin registry with the same `git grep`.
Use `Glm4MoeForCausalLM` as the positive control for the pinned registry.
Run `git diff --check`, the record and oracle checks, and the full
`scripts/agent-preflight.sh --staged` before handoff. Report skips separately.
Verify that the global pin and unrelated records remain byte-identical.

No model, GPU, throughput, or latency gate applies to this documentation edit.
Those gates remain owed by the campaign, without a waiver.

## Evidence

On 7 September 2026, `gh pr view 53906 --repo vllm-project/vllm` returned
`MERGED`, `mergedAt=2026-09-03T16:40:36Z`, and the source object above.
`git log -S Glm5Next` on the upstream registry identifies that same commit.
The focused expiry search returned exit 1 before edits.

## Stop conditions

Do not change the global pin, product behavior, unrelated rows, or benchmark
claims. Report any required expansion to the operator.
No push, merge, GPU work, or large download belongs to this change.

## Owed

The campaign issue [#1998](https://github.com/mudler/vllm.cpp/issues/1998)
owns the real-model oracle build and execution. Device wiring remains under
[#2410](https://github.com/mudler/vllm.cpp/issues/2410).
The reconciliation issue closes only when these documentation changes land.

## Now

The record changes implement the scope. The global pin and campaign lifecycle
remain unchanged. The full gate and independent review decide acceptance.
