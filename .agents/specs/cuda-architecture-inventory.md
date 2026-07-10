# CUDA architecture and backend inventory spike

Status: accepted inventory spike. This document makes the backend rows
claimable after their leaf work is selected; it does not claim runtime support
for any newly inventoried target.

Pins: vLLM `e24d1b24`, vllm.cpp baseline `f7ccaa7`, CUTLASS `v4.4.2`,
FlashInfer `0.6.13`, NVIDIA Cutlass DSL `4.5.2`, and vLLM FlashAttention
`2c839c33`. A competitor benchmark records its own immutable commit/release per
run.

## Scope

In scope:

- all 13 numeric CUDA targets `70,75,80,86,87,89,90,100,101,103,110,120,121`
  and all three vLLM compiler branches;
- all 18 component-specific build/dispatch blocks in
  [backend-matrix.md](../backend-matrix.md), including `a`, `f`, and `+PTX`;
- the vt backend ABI, CPU, ROCm, XPU, TPU, Metal/MLX, Vulkan, and ANE rows;
- correctness, runtime-dispatch, trace, and native-competitor gates; and
- a row-sized order that lets independent agents work without sharing source
  ownership or build directories.

Out of scope for this spike: changing CMake, compiling any target, running a
GPU, declaring model support, selecting product behavior that vLLM already
defines, or treating a successful fatbinary link as execution evidence.

## Upstream chain

### Compiler and target selection

`/home/mudler/_git/vllm/CMakeLists.txt:105-118` defines global target sets of
7, 11, and 9 entries for NVCC `<12.8`, `>=12.8,<13`, and `>=13`. The union is
13. `CMakeLists.txt:201-220` imports PyTorch gencodes but deliberately removes
inherited `+PTX`.

`cmake/utils.cmake:210-263` parses generated-code flags while preserving
architecture suffixes. `:294-345` emits SASS plus PTX for an explicit `+PTX`.
`set_gencode_flags_for_srcs` has a `BUILD_PTX_FOR_ARCH` hook at `:334-344`, but
the pinned tree has no caller, so it is not an available dispatch path.
`cmake/utils.cmake:348-485` performs loose intersection: `f` matches a major
family, `a` preserves architecture-specific targets, SASS fallback is confined
to a major family, and PTX can cross families. SM90+ helper targets are
`cmake/utils.cmake:488-495`.

### Component build selection

The canonical 18 rules and exact source lines are in
[backend-matrix.md](../backend-matrix.md#cuda-component-target-rules). The
important dependency boundaries are:

| Block | Source owner | Selection point |
|---|---|---|
| Stable ops/cache/norm/RoPE/sampling/SSM/collectives | vLLM C++ extension | `CMakeLists.txt:375-406,536-538` |
| Marlin/Machete/AllSpark/DSV3 | vLLM C++ extension and generated sources | `CMakeLists.txt:468-731,1168-1299` |
| CUTLASS scaled-mm/MoE/FP4/W4A8/MLA | CUTLASS plus vLLM launchers | `CMakeLists.txt:737-1061` |
| FlashAttention | vLLM FlashAttention dependency | `cmake/external_projects/vllm_flash_attn.cmake:1-46`; `setup.py:1113-1124` |
| FlashMLA | FlashMLA dependency | `cmake/external_projects/flashmla.cmake:51-142` |
| DeepGEMM | DeepGEMM JIT package | `cmake/external_projects/deepgemm.cmake:61-105` |
| QuTLASS | QuTLASS dependency, one family selector per build | `cmake/external_projects/qutlass.cmake:56-147` |
| FlashInfer/CuTe/fmha/Triton | runtime package or JIT-generated kernel | `requirements/cuda.txt:12-25`; `cmake/external_projects/*.cmake` |

### Runtime selection

Build inclusion is only the first branch. Platform and attention selection run
through `vllm/platforms/cuda.py:89-160,360-493` and
`vllm/v1/attention/backends/registry.py:34-120`. FlashAttention chooses FA2,
FA3, or FA4 by capability and head dimension in `fa_utils.py:132-250`.
FlashInfer's TensorRT-LLM/XQA path additionally gates on SM family, phase, KV
dtype, head divisibility, sinks, speculative decode, and token count in
`vllm/utils/flashinfer.py:373-511`. NVFP4 MoE walks an ordered runtime oracle in
`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:38-276`.

cuBLASLt heuristics, FlashInfer autotuning, dependency JIT, and Inductor codegen
are not visible in a static source inventory. Runtime ground truth therefore
requires nsys for both vLLM and ours on the identical workload, followed by
`nsys stats --report cuda_gpu_kern_sum`. A graphed vLLM trace may include
warmup/JIT/capture work; kernel identities are usable, percentages require a
warmup-excluded steady-state capture.

## Our baseline

- [CMakeLists.txt:37](../../CMakeLists.txt#L37) defaults the whole build to
  `121a`; [CMakeLists.txt:64](../../CMakeLists.txt#L64) applies one global target
  list to every CUDA translation unit.
- CUTLASS and Marlin are enabled only when the target string matches
  `12[01]a` at [CMakeLists.txt:68](../../CMakeLists.txt#L68) and
  [CMakeLists.txt:108](../../CMakeLists.txt#L108).
- FlashAttention-2 is an SM121 gate-specific subset at
  [CMakeLists.txt:428](../../CMakeLists.txt#L428).
- [TritonAOT.cmake:57](../../cmake/TritonAOT.cmake#L57) derives one vendored
  artifact directory; only `triton_aot_vendored/sm_121a/` exists.
- [cuda_backend.cu:119](../../src/vt/cuda/cuda_backend.cu#L119) registers device
  zero and does not validate compute capability before op dispatch.
- The 27B and 35B gate paths have correctness and throughput evidence on the
  GB10/SM121 host. That evidence does not generalize to another target or to
  every registered CUDA op.
- `DeviceType` already reserves Metal, Vulkan, and XPU slots at
  [device.h:11](../../include/vt/device.h#L11); only CPU and CUDA backends are
  registered.

## Port map

| Upstream responsibility | Local destination | Required behavior |
|---|---|---|
| Global target branch and loose intersection | top-level `CMakeLists.txt` plus a focused `cmake/CudaArchitectures.cmake` if decomposition proves useful | preserve compiler-version branches, suffixes, explicit PTX, per-source target filtering |
| Component-specific source target list | owning CUDA target block in CMake | never enable a kernel merely because another TU supports that SM |
| `platforms/cuda.py` capability and backend choice | `src/vt/cuda/cuda_backend.cu` plus narrow dispatch helpers | query device capability once, reject unavailable requested paths, mirror vLLM priority |
| Dependency availability probes | per-family adapter beside the owning `cuda_*.cu` | capability, version, dtype, shape, and feature probes before launch |
| Per-architecture generated code | `src/vt/cuda/triton_aot_vendored/<arch>/MANIFEST` | immutable source/tool/parameter hashes; no Python in a normal build |
| Platform interface | `include/vt/device.h`, `include/vt/backend.h`, `src/vt/backend.cpp` | discrete/unified memory, explicit queue, optional capture, backend-negotiated quant layout |
| Platform tests | `tests/vt/test_*backend*`, per-family CUDA tests, parity/e2e suites | unsupported dispatch fails loudly; supported dispatch records the selected family |
| Competitor harness | `examples/bench/`, server harness, parity ledger | identical model/workload and every-axis result, fresh reference denominator |

Shared CMake/runtime-dispatch files are owned by one architecture-spine lead.
Target agents own only their target manifest/build preset, target-scoped tests,
and row updates until that spine lands.

## Tests to port

Upstream tests are executable specifications. Port or check in a traced skip
with a hardware/toolchain reason; do not silently omit a case.

| Rows | Upstream tests/cases | Local tier |
|---|---|---|
| all CUDA targets | `tests/cuda/test_cuda_compatibility_path.py`, `test_cuda_context.py`, `test_platform_no_cuda_init.py` | configure/build smoke plus device capability/initialization tests |
| architecture dispatch | `tests/kernels/attention/test_attention_selector.py`, `tests/v1/attention/test_attention_backends_selection.py` | unit table for capability/dtype/shape/backend priority |
| SM75/80 and Marlin | `tests/kernels/quantization/test_marlin_gemm.py`, `test_marlin_tile_padding.py`, `test_machete_mm.py` | op parity on matching hardware; compile-only otherwise |
| SM89 FP8 | `test_scaled_mm_kernel_selection.py`, `test_cutlass_scaled_mm.py`, `test_fp8_quant.py` | kernel selection, numerical parity, trace |
| SM90 Hopper | `tests/kernels/attention/test_flash_attn.py`, `test_flashmla.py`; `tests/kernels/moe/test_cutlass_moe.py` | FA3/FlashMLA/MoE selection and model smoke |
| SM100/101/103/110 | `test_flashinfer_trtllm_attention.py`, `test_use_trtllm_attention.py`, `test_cutlass_mla_decode.py` | prefill/decode selector and MLA parity |
| SM120/121 | `test_nvfp4_scaled_mm.py`, `test_nvfp4_quant.py`, `test_flashinfer_sparse_mla_sm120_api.py` | native/emulation parity, selector, current gates |
| ROCm | `tests/kernels/attention/test_rocm_attention_selector.py`, `test_rocm_aiter_unified_attn.py`; `tests/kernels/moe/test_rocm_aiter_topk.py` | HIP op parity then e2e |
| XPU | `tests/kernels/attention/test_xpu_mla_sparse.py`, `tests/lora/test_punica_xpu_ops.py` | loyal platform behavior port |
| CPU | `tests/kernels/attention/test_cpu_attn.py`, `tests/kernels/moe/test_cpu_fused_moe.py`, `tests/quantization/test_cpu_w8a8.py` | correctness plus production thread/quant path |
| platform plugin seam | `tests/plugins_tests/test_platform_plugins.py` | registry and unavailable-backend behavior |

Each CUDA target also runs the local CPU-vs-device op suite, a small-model e2e
smoke, and a model/quant gate representative of the component being claimed.
The 27B/35B files may skip for memory on smaller GPUs; that skip is not support
evidence.

## Gates

For a target row, completion requires all applicable stages:

1. **B0 configure:** intended NVCC accepts the target and only compatible
   component TUs are selected.
2. **B1 build:** clean isolated build, no inherited artifacts, fatbinary
   inspection records SASS/PTX entries.
3. **C1 kernel correctness:** ported upstream cases plus local CPU/oracle parity
   cover dtypes, shapes, boundary values, fallbacks, and unsupported errors.
4. **C2 runtime dispatch:** on real matching hardware, logs and nsys identify
   the intended kernel rather than PTX fallback or an emulation path.
5. **C3 e2e:** representative model/quant/backend output matches pinned vLLM;
   the declared model fits without changing the compared workload.
6. **P performance:** fresh same-box competitor A/B, at least 2-3 repetitions,
   correctness first, total/output throughput and req/s no lower, TTFT/TPOT/ITL
   and peak memory no higher.

CUDA uses pinned vLLM as the mirror floor and SGLang for the requested
low-concurrency serving comparison. CPU and Vulkan use a pinned llama.cpp.
Metal/MLX uses oMLX `v0.5.0rc1` initially and records a new pin when advanced;
llama.cpp Metal is the kernel-level secondary reference. ROCm uses pinned vLLM
and SGLang. XPU uses pinned vLLM, with SGLang added only when the chosen release
supports the identical stack.

Every GPU run holds `/tmp/gpu` for the whole job. A competitor A/B or nsys pair
holds one lock across every arm. Existing hardware is `dgx.casa` (GB10/SM121)
and `192.168.68.103` (M4/16 GB); neither supplies evidence for other CUDA SMs.

## Dependencies

- [kernel-matrix.md](../kernel-matrix.md) for per-family coverage and tests;
- the quantization and model matrices for representative format/model gates;
- an architecture-spine change before independent target ports can merge;
- exact NVCC/toolkit versions required by each compiler branch;
- real target hardware or a named external runner for runtime completion;
- CUTLASS/FlashAttention/FlashInfer/DeepGEMM/QuTLASS source and license pins;
- model artifacts with hashes and sufficient device/unified memory; and
- the shared GPU locking and benchmark protocols.

## Work breakdown and order

| Order | Claim block | Stable rows | Deliverable |
|---:|---|---|---|
| 0 | Evidence backfill | current SM121, CPU, component `PARTIAL` rows | exact anchors, trace provenance, no broadened claim |
| 1 | CUDA architecture spine | `BACKEND-ABI-VT`, all `BACKEND-CUDA-SM*` dependencies | per-TU targets, runtime capability object, explicit fallback/error, per-arch AOT manifest selection |
| 2 | Wave 1 fan-out, parallel after spine | `SM080` A100, `SM090` H100, `SM089` 4090 | generic BF16/FA2 first; then each target's native FP8/FA3/Marlin/CUTLASS leaves |
| 3 | Blackwell data-center families | `SM100`, `SM101`, `SM103`, `SM110` | FA4/TRT-LLM/CUTLASS MLA/DeepGEMM/QuTLASS as hardware allows |
| 4 | Blackwell SM12 family | `SM120`, `SM121` | retain current SM121 gates, add SM120 and close family-specific dispatch gaps |
| 5 | Compatibility lanes | `SM075`, `SM086`, `SM087`, then legacy `SM070` | fallback/Marlin/PTX behavior; SM70 stays in the old-toolchain lane |
| 6 | Backend expansion | ROCm, Metal/MLX, Vulkan, XPU, ANE | roadmap order and native competitor gates; TPU remains parity inventory until scheduled |

Wave 1 preserves the accepted roadmap sequence while allowing A100, H100, and
4090 agents to work in parallel after the common spine. A row can split into
smaller leaves during its spike; dependencies and supersession mappings must be
recorded before parallel claims start.

## Risks and decisions

- Newer NVCC versions remove some old targets; do not erase the older compiler
  branch or substitute PTX without recording the semantic change.
- `a` and `f` are feature-bearing targets, not decorative suffixes. A base-SM
  compile cannot prove architecture-specific instruction availability.
- A global `VT_FP4_MMA_SM120A` definition is unsafe in a multi-arch binary;
  availability must reflect the compiled TU and runtime device.
- Runtime JIT/autotune may select a dependency kernel absent from vLLM `csrc`.
  Source comparison cannot close a runtime row.
- AOT artifacts are per target and generator/tool pin. Copying an SM121 cubin
  into another manifest is invalid.
- Metal/MLX and Vulkan are deliberate vllm.cpp extensions through the mirrored
  platform seam. XPU and ROCm remain loyal vLLM ports. ANE is limited to
  encoder/pooling/fixed-shape draft classes unless its programming model changes.

## Count invariants

The owning matrix must retain 13 CUDA target rows, 18 component-rule rows,
8 platform/ABI rows, and 9 native competitor rows. Compiler branch lists must
remain 7/11/9 with a 13-target union. Upstream changes are applied by adding,
splitting, or superseding stable IDs and updating these counts in the same
change; silent deletion is forbidden.
