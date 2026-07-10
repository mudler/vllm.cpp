# Kernel-family inventory spike

Status: accepted inventory spike. This document grounds the 30 practical rows
in [kernel-matrix.md](../kernel-matrix.md) and decomposes them into independent
claims. It is not an implementation or a declaration that a buildable upstream
kernel executed.

Pins: vLLM `e24d1b24`, vllm.cpp baseline `f7ccaa7`, CUTLASS `v4.4.2`,
FlashInfer `0.6.13`, NVIDIA Cutlass DSL `4.5.2`, vLLM FlashAttention
`2c839c33`, and the dependency revisions recorded by the pinned vLLM CMake
files.

## Scope

In scope are exactly 30 stable rows:

| Area | Stable IDs |
|---|---|
| CUDA dispatch | `KERNEL-CUDA-DISPATCH-AOT` |
| Dense/quant GEMM | `KERNEL-GEMM-BF16`, `KERNEL-GEMM-FP8`, `KERNEL-GEMM-NVFP4-W4A4`, `KERNEL-GEMM-MARLIN-W4A16`, `KERNEL-GEMM-INT-LOWBIT`, `KERNEL-GEMM-W4A8`, `KERNEL-GEMM-QUTLASS-MX`, `KERNEL-GEMM-DEEPGEMM` |
| Elementwise/cache | `KERNEL-EW-NORM-ACT`, `KERNEL-EW-NORM-QUANT`, `KERNEL-ROPE-QKNORM`, `KERNEL-KV-CACHE`, `KERNEL-KV-QUANT` |
| Attention | `KERNEL-ATTN-PAGED`, `KERNEL-ATTN-FA2`, `KERNEL-ATTN-FA3-FA4`, `KERNEL-ATTN-FLASHINFER-TRTLLM`, `KERNEL-ATTN-TRITON-FLEX-HPC`, `KERNEL-ATTN-MLA-SPARSE` |
| MoE | `KERNEL-MOE-ROUTING`, `KERNEL-MOE-UNQUANTIZED`, `KERNEL-MOE-QUANTIZED`, `KERNEL-MOE-SPECIAL` |
| Recurrent/engine | `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH`, `KERNEL-SSM-MAMBA`, `KERNEL-SAMPLING`, `KERNEL-COLLECTIVES`, `KERNEL-SPEC-DECODE` |

The rows enumerate practical dispatch families, not every template
instantiation. A spike splits a row before implementation when dtypes,
architectures, dependencies, or tests produce independently gateable support.
The parent ID remains as an umbrella or records explicit superseding IDs.

Out of scope: changing kernels, running hardware, marking an existing family
`DONE`, or using a source-only comparison as runtime evidence.

## Upstream chain

### Orchestration and build surface

The stable extension sources at `/home/mudler/_git/vllm/CMakeLists.txt:375-406`
cover activation, quantization, GPTQ, RoPE/QK-norm, layernorm/fusions,
attention-state merge, sampler/top-k, selective scan, cache, collectives, and
DeepSeek fusions. Per-source architecture flags are applied at `:536-538`.

Dense quantized kernel families and their architecture gates are at
`CMakeLists.txt:468-1061`. The MoE extension inventories router, top-k,
WnA16, permute/unpermute, Marlin, and DSV3 router sources at `:1135-1299`.
Quant method registration and lazy config selection are
`vllm/model_executor/layers/quantization/__init__.py:12-182`; each method's
capability contract begins at `base_config.py:87-126`.

Attention is a runtime registry, not one kernel. The complete pinned enum is
`vllm/v1/attention/backends/registry.py:34-120`; CUDA priority and validation
are `vllm/platforms/cuda.py:89-160,360-493`. It spans FlashAttention,
FlashInfer, TRT-LLM/XQA, Triton, Flex, TurboQuant, HPC, dense/latent/sparse MLA,
ROCm, XPU, CPU, and custom backends.

### Dependency-owned execution

| Family | Dependency ground truth |
|---|---|
| Dense BF16/FP8 | cuBLASLt runtime heuristics; algorithm identity requires trace and heuristic logging |
| FP4/FP8 scaled GEMM and fusions | CUTLASS and FlashInfer CuTe DSL; installed fused FP4 norm kernels at `flashinfer-ref/cute_dsl/add_rmsnorm_fp4quant.py:16-30,91-182` and `rmsnorm_fp4quant.py:16-24,78-87` |
| Flash attention | vLLM FlashAttention dependency plus FA4 CuTe package; version dispatch `fa_utils.py:132-250` |
| FlashInfer/TRT-LLM attention | `vllm/utils/flashinfer.py:206-342,373-511`; selection depends on SM, phase, KV dtype, heads, sinks, speculation, and token count |
| DeepGEMM/QuTLASS | external project source plus runtime JIT/package; `deepgemm.cmake:61-105`, `qutlass.cmake:56-147` |
| GDN | vLLM FLA Triton `fla/ops/chunk*.py`, FlashInfer `gdn_prefill.py`/`gdn_decode.py`, and Blackwell CuTe sources |
| Inductor fusions | generated code dumped with `TORCH_LOGS=output_code` and `TORCH_COMPILE_DEBUG=1` |

Runtime traces in the existing ledger establish only the GB10 gate workload.
They show dependency kernels that static scans missed, including TensorRT-LLM,
FlashAttention, cuBLASLt-selected GEMMs, and generated Triton. Any other row,
shape, model, or architecture is `UNTRACED` until independently captured.

## Our baseline

- [ops.h:30](../../include/vt/ops.h#L30) ends a 53-op CUDA-facing enum and
  [ops.h:160](../../include/vt/ops.h#L160) defines typed registration through
  raw tensor/queue adapters.
- CUDA registers all 53 real op IDs; CPU registers 43. The ten backend-specific
  IDs without CPU registration are `kCastF32`, `kMatmulFp8CublasLt`,
  `kMatmulFp8Cutlass`, `kMatmulNvfp4`, `kMatmulNvfp4Cutlass`,
  `kMoeCombineGate`, `kMoeGroupedGemmNvfp4`,
  `kMoeGroupedGemmNvfp4Marlin`, `kQuantFp8Static`, and
  `kSwizzleBlockscale`. Conceptual references exist for some, but "every op has
  a CPU reference" is not a valid support claim.
- Existing CUDA families live in `src/vt/cuda/cuda_{matmul,ops,cache,
  paged_attn,flash_attn_fa2,moe,gdn,sample,glue}*.cu`; registration anchors and
  tests are recorded row-by-row in the matrix.
- Current implemented slices include BF16/FP8/NVFP4 GEMM, NVFP4 Marlin MoE,
  norms/activations/glue, RoPE, KV insertion, paged attention plus a narrow FA2
  prefill, MoE routing/grouped NVFP4, Qwen GDN, and sampling.
- Current build/runtime evidence is SM121-specific. Cross-architecture support,
  quantized KV, FA3/FA4, FlashInfer/TRT-LLM attention, MLA/sparse attention,
  generic SSM, collectives, and speculative kernels remain open.
- The GDN AOT BF16 and scratch leaves are already claimed as `CLAIM-PR3`; no
  other agent may edit their files or rows until that claim closes.

## Port map

The drop-in objective is to keep the upstream raw pointer, shape, stride,
dtype, capability, and stream contract and replace only the Torch tensor wrapper.

| Upstream family | Local adapter/implementation boundary | Port rule |
|---|---|---|
| stable core ops | `include/vt/ops.h`, `src/vt/cuda/cuda_ops.cu`, focused family TU | keep signature and launch semantics; typed registration catches drift |
| dense/scaled GEMM | `cuda_matmul*.cu`, `cuda_marlin_repack.cu` | mirror capability and ordered kernel selection; preserve weight/scale layouts |
| cache/attention | `cuda_cache.cu`, `cuda_paged_attn.cu`, one TU per external family | separate metadata adapter from copied kernel; no host sync hidden in wrapper |
| MoE | `cuda_moe*.cu`, quantized GEMM TU | port router/permute/GEMM/combine tests as one executable chain, while retaining leaf gates |
| GDN/SSM | `cuda_gdn.cu` and future generic SSM files | keep recurrent-state dtype, slot update, metadata, scratch lifetime, and capture semantics explicit |
| sampling/spec decode | `cuda_sample.cu` and future `cuda_spec_decode.cu` | preserve ordering, tie breaks, RNG/rejection semantics, and device-resident metadata |
| generated/JIT family | vendored per-arch artifacts plus a narrow C launcher | immutable manifest; maintainer-only regeneration; normal builds have no Python |
| collectives | future backend-specific collective adapter | preserve process-group/rank semantics and topology capability checks |

Each ported source carries its upstream path and commit header. A deliberate
fusion or C++-native divergence is recorded in the parity ledger without
changing observable semantics.

## Tests to port

| Rows | Pinned upstream test modules | Local tier / initial skip policy |
|---|---|---|
| CUDA dispatch/AOT | `tests/cuda/test_cuda_context.py`, `test_cuda_compatibility_path.py`; `tests/v1/cudagraph/test_cudagraph_dispatch.py` | host dispatch table plus real-GPU graph test; skip only for absent CUDA hardware |
| BF16/FP8 GEMM | `tests/kernels/quantization/test_cutlass_scaled_mm.py`, `test_scaled_mm_kernel_selection.py`, `test_fp8_quant.py` | CPU/reference op parity, selector, nsys family identity |
| NVFP4/Marlin | `test_nvfp4_scaled_mm.py`, `test_nvfp4_quant.py`, `test_nvfp4_emulation.py`, `test_marlin_gemm.py`, `test_marlin_tile_padding.py` | native/emulation byte/numeric parity and capability errors |
| integer/W4A8/Qutlass/DeepGEMM | `test_machete_mm.py`, `test_cutlass_w4a8_moe.py`, `test_mxfp4_qutlass.py`, `test_nvfp4_qutlass.py`; `tests/kernels/moe/test_deepgemm.py` | checked-in skipped cases until their dependency and matching GPU exist |
| norm/activation/quant | `tests/kernels/core/test_activation.py`, `test_layernorm.py`, `test_fused_quant_layernorm.py`; `tests/kernels/test_fused_quant_activation.py` | CPU/device parity, byte-exact quant, aliasing/in-place cases |
| RoPE/QK-norm | `tests/kernels/core/test_fused_qk_norm_rope.py`, `test_mrope.py`; `tests/kernels/test_fused_qk_norm_rope_gate.py` | scalar oracle then fused-vs-composed device parity |
| KV cache | `tests/kernels/test_cache_kernels.py`, `test_compressor_kv_cache.py`; `tests/quantization/test_per_token_kv_cache.py` | block/stride/skip semantics, quantized dtype and scale coverage |
| paged/FA2 | `tests/kernels/attention/test_attention.py`, `test_cache.py`, `test_flash_attn.py`, `test_prefix_prefill.py` | dense reference, ragged/paged boundary cases, selected-family assertion |
| FA3/FA4/FlashInfer/TRT-LLM | `test_flashinfer.py`, `test_flashinfer_trtllm_attention.py`, `test_use_trtllm_attention.py` | checked-in selector cases; hardware cases skipped with exact missing SM |
| Triton/Flex/HPC/MLA/sparse | `test_triton_prefill_attention.py`, `test_triton_decode_attention.py`, `tests/kernels/test_flex_attention.py`, `test_cutlass_mla_decode.py`, `test_flashmla.py`, `test_flashinfer_mla_decode.py` | split by backend; no umbrella pass hides a missing mode |
| MoE routing/unquantized | `tests/kernels/moe/test_fused_topk.py`, `test_grouped_topk.py`, `test_moe_permute_unpermute.py`, `test_unquantized_backend_selection.py` | port tie, empty expert, padding, permute round-trip, and selection cases |
| quantized/special MoE | `test_block_fp8.py`, `test_nvfp4_moe.py`, `test_mxfp4_moe.py`, `test_moe_kernel_oracle.py`, `test_topk_softplus_sqrt.py` | backend oracle plus each format/architecture leaf |
| GDN/SSM | `tests/kernels/mamba/test_gdn_forward_core_split.py`, `test_gdn_prefill_cutedsl.py`, `test_causal_conv1d.py`, `test_mamba_ssm.py`, `test_mamba_ssm_ssd.py` | sequential oracle, chunked equivalence, state/cache, scratch reuse/capture |
| sampling | `tests/v1/sample/test_sampler.py`, `test_topk_topp_sampler.py`, `test_rejection_sampler.py` | deterministic tie/order tests, distribution tests, Philox parity when claimed |
| collectives | `tests/distributed/test_custom_all_reduce.py`, `test_quick_all_reduce.py`; `tests/kernels/core/test_fused_allreduce_gemma_rms_norm.py` | multi-GPU checked skip until a declared topology exists |
| speculative decode | `tests/v1/spec_decode/test_mtp.py`, `test_eagle_step_kernel.py`, `test_rejection_sampler_utils.py`; `tests/v1/e2e/spec_decode/test_spec_decode.py` | unit kernels first, then MTP/GDN rollback and e2e acceptance/speed |

An upstream case that cannot pass lands as a named skipped test with the row ID,
missing dependency/hardware, and unblocking condition. Porting the implementation
without these cases is incomplete.

## Gates

1. **ABI gate:** copied kernel launcher binds through `vt::` without changing
   its raw pointer/shape/stride/stream semantics; unsupported dtype/shape/SM
   fails before launch.
2. **Correctness gate:** ported upstream tests and a local CPU/reference oracle
   pass across boundary shapes, layouts, dtypes, aliasing, and fallback modes.
3. **Selection gate:** a table-driven host test proves the same capability,
   dtype, quant, phase, and shape priority as pinned vLLM.
4. **Trace gate:** nsys both vLLM and ours on the identical warmed workload;
   `cuda_gpu_kern_sum` records actual names, calls, and steady-state time. For
   cuBLASLt, also record heuristic/algorithm identity; for JIT/codegen, archive
   generated source or manifest hashes.
5. **End-to-end gate:** a representative model/quant/backend matches pinned
   vLLM token-for-token before performance is considered.
6. **Performance gate:** same-box, same-model, same-workload, same-binary A/B,
   fresh vLLM denominator, 2-3 or more repetitions, idle locked GPU. Throughput
   and req/s are no lower; TTFT, TPOT/ITL, and peak memory are no higher.

A family may be `DONE` on one architecture only if the row item explicitly
declares that architecture. Otherwise architecture coverage is split into leaf
rows and the umbrella remains `PARTIAL`.

## Dependencies

| Row block | Required predecessors |
|---|---|
| every CUDA family | `KERNEL-CUDA-DISPATCH-AOT` and its target row in [backend-matrix.md](../backend-matrix.md) |
| quantized GEMM/MoE/KV | owning quantization format loader/repack row plus dependency source pin |
| attention | `KERNEL-KV-CACHE`; quantized modes also require `KERNEL-KV-QUANT` |
| MoE GEMM | `KERNEL-MOE-ROUTING` plus owning dense/quant GEMM family |
| GDN/SSM | metadata/state-cache semantics and target-specific AOT/scratch support |
| speculative decode | sampling/rejection, attention/KV, draft model, and GDN rollback for the gate models |
| collectives | backend runtime, rank/topology API, and a multi-GPU runner |

Shared requirements are the pinned vLLM/dependency sources, model artifacts,
the backend/model/quant matrices, upstream test fixtures, target hardware, nsys,
and the project GPU-lock and benchmark protocols.

## Work breakdown and order

| Order | Claim block | Owned rows/files | Exit |
|---:|---|---|---|
| 0 | Legacy evidence backfill | one existing family row and only its matrix/test anchors | working subset and missing modes are explicit; no implementation edit |
| 1 | Drop-in dispatch spine | `KERNEL-CUDA-DISPATCH-AOT`; `include/vt/ops.h`, backend/dispatch helpers; shared CMake has one lead | per-family/per-arch registration and failure semantics are stable |
| 2 | Current CUDA leaves in parallel | GEMM TUs; EW/RoPE TUs; KV/attention TUs; MoE TUs; sampling TU | each leaf ports upstream tests and produces GB10 selection/trace evidence |
| 3 | Active GDN closure | `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH`; existing `CLAIM-PR3` only | claim returns commit, tests, trace/perf evidence, and reconciled states |
| 4 | T1-driven gaps | FP8 breadth, MTP/spec kernels, sliding-window/KV/attention, Dense/MoE family needs | follows roadmap C2-C8 rather than opportunistic kernel order |
| 5 | Architecture-native families | Marlin/Machete/FA3/FA4/FlashInfer/MLA/DeepGEMM/QuTLASS leaves | opens only after matching backend target spine and hardware claim |
| 6 | Backend expansion | ROCm, MLX/Metal, Vulkan, XPU implementations of the same row contracts | native competitor gates pass; no backend-specific semantics leak upward |

Within order 2, agents may run concurrently because `cuda_matmul*`,
`cuda_ops`/`cuda_glue`, `cuda_cache`/`cuda_paged_attn`, `cuda_moe*`, and
`cuda_sample` are disjoint. Any change to shared `CMakeLists.txt`,
`include/vt/ops.h`, or dispatch registries requires the spine lead or an
explicit ownership handoff.

## Risks and decisions

- vLLM is orchestration; a kernel absent from its `csrc` may live in
  FlashInfer, CUTLASS, cuBLASLt, TensorRT-LLM, DeepGEMM, or generated Inductor
  code. Dependency inspection and traces are mandatory.
- Copied kernels may rely on Torch tensor metadata, allocator behavior, or
  generated layout objects. The adapter replaces wrappers, not algorithmic
  behavior; deviations are recorded and tested.
- Fusion names are not proof of fusion. Generated code bodies and trace launch
  boundaries decide what actually ran.
- Quant format recognition, dequant/repack, native compute, backend coverage,
  and e2e performance remain separate statuses in the quantization matrix.
- Numerical parity is defined by the upstream tests and pinned oracle. A faster
  wrong-precision mode is not an accepted fallback.
- Multi-GPU collectives and many SM-specific families cannot complete on the
  current GB10. They remain `INVENTORIED` or checked-skipped until a declared
  runner exists.

## Count invariants

The owning matrix has exactly 30 rows. At this baseline it contains
10 `ANCHOR-BACKFILL`, 5 `PARTIAL`, 2 `ACTIVE`, and 13 `INVENTORIED` rows.
`CLAIM-PR3` owns the two active rows. When that claim closes, integration updates
their states, owner/commit evidence, lifecycle counts, and any roadmap/README
surface in the same change. A new upstream family changes the row inventory,
tests-to-port table, work order, and counts together.
