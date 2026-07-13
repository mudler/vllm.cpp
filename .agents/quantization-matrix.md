# Quantization matrix

This is the canonical per-scheme support inventory for roadmap C4/D1. The
coverage spike is [specs/quantization-coverage.md](specs/quantization-coverage.md).
Sources are pinned to vLLM `e24d1b24`, llama.cpp `237ad9b` plus the documented
local type-40/41 fork, MLX `4367c73b`, and MLX-LM `a790972f`.

## Status contract

| Column | Meaning |
|---|---|
| `R` | serialization/config is recognized and validated |
| `M` | dequantization/repack/materialization exists and is tested |
| `C` | native quantized compute is dispatched; bf16 expansion is not `C` |
| `E` | representative real-model correctness gate passed |
| `P` | reference-engine performance gate passed on the named backend |

`Y` means grounded support, `-` means absent, and `part` means a documented
subset. A row is `DONE` only when its declared scope has all required stages;
otherwise it remains `PARTIAL` or `INVENTORIED` even if parsing works.

## Cross-cutting execution rows

| ID | Item | Upstream | Our code | Tests/evidence | Spike/spec | State | Owner |
|---|---|---|---|---|---|---|---|
| `QUANT-GGUF-COMPUTE` | Block row (claim the three leaves below, not this row): quantized `vt::` storage plus direct GGUF compute kernels; bf16 expansion does not count | llama.cpp `ggml/src/ggml-cpu/ggml-cpu.c:211-406,1245-1443` tensor traits + `ggml-cpu/quants.c`, `ggml-cpu/repack.cpp` at `237ad9b96` | current loader expands at [qwen3_5_gguf_weights.cpp:105](../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L105); direct path absent | dequant units and same-file APEX gates cover only materialization; CPU perf floor vs same-file llama.cpp measured — decode 54–75×, prefill ≈1,480×, peak RSS 2.7× behind ([ledger B4 row](parity-ledger.md#L290)); vendor-if-faster criterion MET, route = threadpool → compute-in-quant | leaves: [threadpool](specs/gguf-cpu-threadpool.md), [CIQ GEMM](specs/gguf-compute-in-quant-gemm.md), [keep-quant loader](specs/gguf-keep-quant-loader.md); umbrella [coverage spike](specs/quantization-coverage.md) | `READY` | - |
| `QUANT-GGUF-CPU-THREADPOOL` | vt CPU threadpool + chunked parallel op dispatch (GEMM first, then row/batch ops); bit-identical to single-thread; prerequisite leaf. W1-W3 complete; W4 idle-host speed/RSS gate pending | llama.cpp `ggml/src/ggml-cpu/ggml-cpu.c:471-610` (pool/barrier/chunk), `:3024-3390` (workers), `:1155-1443` (mul_mat chunking) at `237ad9b96` | [pool core](../src/vt/cpu/cpu_threadpool.cpp#L78), [GEMM chunking](../src/vt/cpu/cpu_ops.cpp#L88), [row/batch dispatch](../src/vt/cpu/cpu_ops.cpp#L163) | [barrier/chunk/epoch/concurrency tests](../tests/vt/test_cpu_threadpool.cpp#L63), [1/3/20 determinism battery](../tests/vt/test_cpu_threadpool.cpp#L399); full CPU ctest 94/94 at each thread count + TSAN clean; checkpoint-gated model tests absent locally; performance run correctly deferred on a contended host ([ledger recovery row](parity-ledger.md#L296)) | [threadpool leaf](specs/gguf-cpu-threadpool.md) | `GATING` | - |
| `QUANT-GGUF-CIQ-GEMM` | Compute-in-quant GEMM: activation quant (Q8_0/Q8_K) + per-type vec_dot dispatch for Q8_0/Q4_K/Q5_K/Q6_K/Q3_K/Q4_0; portable C++ tier, then x86/Arm SIMD + repack tiers | llama.cpp `ggml/src/ggml-cpu/ggml-cpu.c:211-406` traits table, `ggml-cpu/quants.c:174-860` generic vec_dot, `arch/{x86,arm}/quants.c`, `ggml-cpu/repack.cpp:4153-4830` at `237ad9b96` | absent — all GGUF weights expand to bf16 at [qwen3_5_gguf_weights.cpp:105](../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L105) | [dequant units](../tests/vllm/test_gguf_dequant.cpp#L25) cover layouts only; llama.cpp floor to match/beat measured ([ledger B4 row](parity-ledger.md#L290)) | [CIQ GEMM leaf](specs/gguf-compute-in-quant-gemm.md) | `READY` | - |
| `QUANT-GGUF-KEEPQ-LOADER` | Keep-quantized GGUF loader: block-resident 2-D matmul weights ([N,K], no transpose), per-tensor routing, `VT_CPU_REF` dequant-oracle switch, bench-branch `7c91a42` merge | llama.cpp `src/llama-model-loader.cpp:1047,1385` (file-typed residency), `ggml/src/ggml-cpu/repack.cpp:4727` (repack-at-load hook) at `237ad9b96` | absent — expansion at [qwen3_5_gguf_weights.cpp:105](../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L105); precedent [qwen3_5_weights.h:37](../include/vllm/model_executor/models/qwen3_5_weights.h#L37) | [loader units](../tests/vllm/test_gguf_qwen36_loader.cpp#L1) cover dequant path only; 2.65× RSS gap measured ([ledger B4 row](parity-ledger.md#L290)) | [keep-quant loader leaf](specs/gguf-keep-quant-loader.md) | `READY` | - |
| `QUANT-GGUF-PRESETS` | Representative mixed-file gates for every llama.cpp output preset family | llama.cpp `tools/quantize/quantize.cpp:34-74` | only custom APEX mixed files are executable; no general preset dispatch | [APEX gates](../tests/parity/test_qwen36_gguf_engine.cpp#L143) do not prove llama.cpp preset breadth | [coverage spike](specs/quantization-coverage.md); split exact preset IDs before `READY` | `INVENTORIED` | - |

## 1. llama.cpp / GGUF encodings

Upstream encodings: `ggml/include/ggml.h:389-432`. Our recognition is
[gguf_reader.cpp:191](../src/vllm/model_executor/model_loader/gguf_reader.cpp#L191);
materialization is [gguf_dequant.cpp:246](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L246);
all executable GGUF weights currently expand to bf16 at
[qwen3_5_gguf_weights.cpp:105](../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L105).

| ID | Encoding | ggml ID / upstream role | R | M | C | E | P | State | Our code + tests/evidence | Spike/spec | Owner |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|---|
| `QUANT-GGUF-F32` | F32 | 0 / output | Y | Y | - | Y | - | `PARTIAL` | [dequant dispatch](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L246); [unit](../tests/vllm/test_gguf_dequant.cpp#L25); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L143) | coverage spike; leaf open | - |
| `QUANT-GGUF-F16` | F16 | 1 / output | Y | - | - | - | - | `INVENTORIED` | reader `:197-200`; executable path absent | leaf open | - |
| `QUANT-GGUF-Q4_0` | Q4_0 | 2 / output | Y | Y | - | - | - | `PARTIAL` | [dequant kernel](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L43), [dispatch](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L251); [unit](../tests/vllm/test_gguf_dequant.cpp#L49) | leaf open | - |
| `QUANT-GGUF-Q4_1` | Q4_1 | 3 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-Q5_0` | Q5_0 | 6 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-Q5_1` | Q5_1 | 7 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-Q8_0` | Q8_0 | 8 / output | Y | Y | - | Y | - | `PARTIAL` | [dequant kernel](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L59), [dispatch](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L255); [unit](../tests/vllm/test_gguf_dequant.cpp#L35); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L149) | leaf open | - |
| `QUANT-GGUF-Q2_K` | Q2_K | 10 / output | Y | - | - | - | - | `INVENTORIED` | reader `:209-212`; executable path absent | leaf open | - |
| `QUANT-GGUF-Q3_K` | Q3_K | 11 / output + S/M/L presets | Y | Y | - | Y | - | `PARTIAL` | [dequant kernel](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L72), [dispatch](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L259); [unit](../tests/vllm/test_gguf_dequant.cpp#L164); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L143) | leaf open | - |
| `QUANT-GGUF-Q4_K` | Q4_K | 12 / output + S/M presets | Y | Y | - | Y | - | `PARTIAL` | [dequant kernel](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L119), [dispatch](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L263); [unit](../tests/vllm/test_gguf_dequant.cpp#L77); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L143) | leaf open | - |
| `QUANT-GGUF-Q5_K` | Q5_K | 13 / output + S/M presets | Y | Y | - | Y | - | `PARTIAL` | [dequant kernel](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L148), [dispatch](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L267); [unit](../tests/vllm/test_gguf_dequant.cpp#L105); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L149) | leaf open | - |
| `QUANT-GGUF-Q6_K` | Q6_K | 14 / output | Y | Y | - | Y | - | `PARTIAL` | [dequant kernel](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L185), [dispatch](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L271); [unit](../tests/vllm/test_gguf_dequant.cpp#L134); [Compact gate](../tests/parity/test_qwen36_gguf_engine.cpp#L143), [Balanced gate](../tests/parity/test_qwen36_gguf_engine.cpp#L149) | leaf open | - |
| `QUANT-GGUF-IQ2_XXS` | IQ2_XXS | 16 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-IQ2_XS` | IQ2_XS | 17 / output; IQ2_S preset storage | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-IQ3_XXS` | IQ3_XXS | 18 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-IQ1_S` | IQ1_S | 19 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-IQ4_NL` | IQ4_NL | 20 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-IQ3_S` | IQ3_S | 21 / IQ3 S/XS/M storage | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-IQ2_S` | IQ2_S | 22 / IQ2_M storage | Y | - | - | - | - | `INVENTORIED` | reader `:229-234`; [explicit rejection](../tests/vllm/test_gguf_dequant.cpp#L223) | leaf open | - |
| `QUANT-GGUF-IQ4_XS` | IQ4_XS | 23 / output | Y | - | - | - | - | `INVENTORIED` | reader `:235-240`; [explicit rejection](../tests/vllm/test_gguf_dequant.cpp#L223) | leaf open | - |
| `QUANT-GGUF-IQ1_M` | IQ1_M | 29 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-BF16` | BF16 | 30 / output | Y | - | - | - | - | `INVENTORIED` | reader `:261-264`; executable path absent | leaf open | - |
| `QUANT-GGUF-TQ1_0` | TQ1_0 | 34 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-TQ2_0` | TQ2_0 | 35 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open | - |
| `QUANT-GGUF-MXFP4` | MXFP4 | 39 / MoE output | Y | - | - | - | - | `INVENTORIED` | reader `:265-270`; executable path absent | leaf open | - |
| `QUANT-GGUF-NVFP4` | NVFP4 | 40 / local fork storage | Y | - | - | - | - | `INVENTORIED` | reader `:271-279`; no quantize preset or executable path | [gguf NVFP4 notes](specs/gguf-nvfp4-notes.md) | - |
| `QUANT-GGUF-Q1_0` | Q1_0 | 41 / local fork output | Y | - | - | - | - | `INVENTORIED` | reader `:280-285`; executable path absent | leaf open | - |
Internal dot/scalar tensor types Q8_1, Q8_K, I8/I16/I32/I64/F64 remain parser
concerns, not output-quantization completion rows. Removed GGUF types 4/5,
31-33 and 36-38 are intentionally not roadmap targets.

### llama.cpp output presets

This is a **non-claimable roll-up** under `QUANT-GGUF-PRESETS`; it prevents
encoding support from being mistaken for mixed-preset coverage. Source:
`tools/quantize/quantize.cpp:34-74`. The umbrella spike assigns exact stable
preset IDs before any preset enters `READY` or implementation.

| Preset block | Exact accepted names | Required encodings | Gate state | Spike |
|---|---|---|---|---|
| Scalars | F32, F16, BF16 | matching scalar type | `PARTIAL`: F32 only, custom APEX files | leaf specs open |
| Legacy | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | matching legacy types | `INVENTORIED`: no representative llama-quantize preset gate | leaf specs open |
| K2 | Q2_K, Q2_K_S | Q2_K plus mixed tensor choices | `INVENTORIED` | leaf specs open |
| K3 | Q3_K, Q3_K_S, Q3_K_M, Q3_K_L | Q3_K plus mixed tensor choices | `INVENTORIED` | leaf specs open |
| K4 | Q4_K, Q4_K_S, Q4_K_M | Q4_K plus mixed tensor choices | `INVENTORIED` | leaf specs open |
| K5 | Q5_K, Q5_K_S, Q5_K_M | Q5_K plus mixed tensor choices | `INVENTORIED` | leaf specs open |
| K6 | Q6_K | Q6_K plus mixed tensor choices | `INVENTORIED` | leaf spec open |
| I1 | IQ1_S, IQ1_M | IQ1_S/IQ1_M plus mixed choices | `INVENTORIED` | leaf specs open |
| I2 | IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M | IQ2 variants plus mixed choices | `INVENTORIED` | leaf specs open |
| I3 | IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M | IQ3 variants plus mixed choices | `INVENTORIED` | leaf specs open |
| I4 | IQ4_NL, IQ4_XS | IQ4 variants plus mixed choices | `INVENTORIED` | leaf specs open |
| Ternary/new | TQ1_0, TQ2_0, Q1_0, MXFP4_MOE | matching types | `INVENTORIED` | leaf specs open |

`COPY` is an operation, not a quantization format. Q3_K/Q4_K/Q5_K without a
suffix are aliases for the corresponding `_M` preset.

## 2. vLLM-native and compatibility formats

Registry source: `/home/mudler/_git/vllm/vllm/model_executor/layers/quantization/__init__.py:12-46`.
`DONE` below is deliberately limited to a named gate-model slice.

| ID | Upstream method / scheme | W/A/KV | Backend scope | R | M | C | E | P | State | Our code + tests/evidence | Spike/spec | Owner |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|---|
| `QUANT-NVFP4-MO-W4A16` | ModelOpt NVFP4 W4A16 | W4/A16 | CUDA sm121a, 35B experts | Y | Y | Y | Y | Y | `DONE` | [weights](../src/vllm/model_executor/models/qwen3_5_weights.cpp#L191), [Marlin dispatch](../src/vllm/model_executor/models/qwen3_5.cpp#L2562), [kernel tests](../tests/vt/test_ops_moe_grouped.cpp#L453), [35B gate](../tests/parity/test_qwen36_paged_engine.cpp#L78), [final gate ledger](parity-ledger.md#L284) | [Marlin spike](specs/marlin-dropin-feasibility.md); implementation sequence `240d4a1`..`d6ed102`, accepted at MVP closure `83010c7` | `83010c7` |
| `QUANT-NVFP4-CT-W4A4` | compressed-tensors NVFP4 | W4/A4 | CUDA sm121a, 27B dense | Y | Y | Y | Y | Y | `DONE` | Existing [weights](../src/vllm/model_executor/models/qwen3_5_dense_weights.cpp#L106), [dispatch](../src/vllm/model_executor/models/qwen3_5.cpp#L1054), [quant](../src/vt/cuda/cuda_matmul_nvfp4.cu#L958), [tests](../tests/vt/test_ops_nvfp4_fp4.cpp#L414) and [support ledger](parity-ledger.md#L284) remain. W3-C frozen-plan control is complete; W3-E strict-fails at **39/40 timing + 1/8 memory**. W3-F implements the vLLM/FlashInfer device F32 pointer through [typed validation](../src/vt/ops.cpp#L374), [direct CUDA consumption](../src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu#L924), [model ownership](../src/vllm/model_executor/models/qwen3_5.cpp#L1067) and [ported tests](../tests/vt/test_ops_nvfp4_fp4.cpp#L1099), while `VT_FP4_DEVICE_ALPHA=0` retains staging. CPU Release 103/103 and focused sanitizer suites pass; all DGX results remain pending. `3f256ab` remains 55/124 and no 35B performance is claimed | [W4A4 notes](specs/qwen27b-w4a4-notes.md); MVP closure `83010c7`; [W3-C spike](specs/nvfp4-persistent-plan-cache.md); [W3-F spike](specs/nvfp4-device-alpha.md) | `83010c7` |
| `QUANT-FP8-MO-STATIC` | ModelOpt static per-tensor FP8 | W8/A8 | CUDA sm121a, 35B projections | Y | Y | Y | Y | Y | `DONE` | [weights](../src/vllm/model_executor/models/qwen3_5_weights.cpp#L118), [dispatch](../src/vllm/model_executor/models/qwen3_5.cpp#L627), [kernel tests](../tests/vt/test_ops_fp8_cutlass.cpp#L188), [35B gate](../tests/parity/test_qwen36_paged_engine.cpp#L78), [final gate ledger](parity-ledger.md#L284) | [forward notes](specs/qwen36-forward-notes.md); native path `7b682cc`/`fd2a575`, accepted at MVP closure `83010c7` | `83010c7` |
| `QUANT-FP8-GENERIC` | fp8 + ModelOpt FP8 breadth | static/dynamic, tensor/channel/token/block | all vLLM backends/SM dispatch | part | part | part | part | - | `PARTIAL` | gate slice only: [loader](../src/vllm/model_executor/models/qwen3_5_weights.cpp#L118), [kernel tests](../tests/vt/test_ops_fp8_cutlass.cpp#L188), [35B gate](../tests/parity/test_qwen36_paged_engine.cpp#L78) | leaf spec open | - |
| `QUANT-FP8-PB-WO` | ModelOpt FP8 per-block weight-only | W8/A16 | capability selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-FP8-PCPT` | ModelOpt FP8 per-channel/per-token | W8/A8 | capability selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-MXFP8-MODELOPT` | ModelOpt MXFP8 | W8/A8 | CUDA/ROCm/XPU dispatch | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-MIXED-MODELOPT` | ModelOpt mixed precision | FP8/NVFP4/MXFP8 groups | per-layer | part | part | part | Y | Y | `PARTIAL` | 35B FP8+NVFP4 slice: [loader](../src/vllm/model_executor/models/qwen3_5_weights.cpp#L118), [FP8 tests](../tests/vt/test_ops_fp8_cutlass.cpp#L188), [NVFP4 tests](../tests/vt/test_ops_moe_grouped.cpp#L453), [gate](../tests/parity/test_qwen36_paged_engine.cpp#L78) | leaf spec open | - |
| `QUANT-CT-MXFP4` | compressed-tensors W4A4 MXFP4 | W4/A4 | capability selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-CT-W4A8-FP8` | compressed-tensors W4A8 FP8 | W4/A8 | CUTLASS | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-CT-W4A8-INT8` | compressed-tensors W4A8 INT8 | W4/A8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-CT-W8A8-FP8` | compressed-tensors W8A8 FP8 | W8/A8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-CT-W8A8-INT8` | compressed-tensors W8A8 INT8 | W8/A8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-CT-W8A8-MXFP8` | compressed-tensors W8A8 MXFP8 | W8/A8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-CT-W8A16-FP8` | compressed-tensors FP8 weight-only | W8/A16 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-CT-WNA16` | compressed-tensors WnA16 | symmetric 2-8b; asymmetric 4/8b | Marlin/Machete/etc. | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-CT-WNA8O8` | compressed-tensors WnA8O8 | W2/4/8, static int8 A/O | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-AWQ` | AWQ / AutoAWQ / AWQ-Marlin | W4/A16 | Marlin/CUDA/CPU variants | - | - | - | - | - | `INVENTORIED` | vendored Marlin is NVFP4-only today | leaf spec open | - |
| `QUANT-GPTQ` | GPTQ / AutoGPTQ / GPTQ-Marlin | W2-8/A16 | Marlin/CUDA/CPU variants | - | - | - | - | - | `INVENTORIED` | vendored Marlin is NVFP4-only today | leaf spec open | - |
| `QUANT-BNB` | bitsandbytes | FP4/NF4/INT8 | CUDA library path | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-EXPERTS-I8` | experts-int8 | MoE int8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-MOE-WNA16` | MoE WnA16 | W4/8-A16 | Marlin/Machete/ROCm | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-MXFP4-GPTOSS` | MXFP4 / GPT-OSS MXFP4 | W4/A16 or W4/A4 | model/kernel selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-QUARK` | AMD Quark | mixed int/fp | ROCm | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-TORCHAO` | TorchAO | multiple | compatibility path | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-INC` | Intel INC / AutoRound | WnA16 etc. | CPU/XPU | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-HUMMING` | Humming | weight-only | CUDA | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-DSV4-FP8` | DeepSeek-V4 FP8 | model-specific | CUDA | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-ONLINE` | online FP8/MXFP8/int8 shorthands | runtime W/A | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open | - |
| `QUANT-DEPRECATED` | FBGEMM-FP8, FPQuant | compatibility only | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open; do not prioritize | - |
## 3. KV-cache quantization

Pinned vLLM source: `vllm/config/cache.py:19-36`.

| ID | Item | Upstream | Our code | Tests/evidence | Spike/spec | State | Owner |
|---|---|---|---|---|---|---|---|
| `QUANT-KV-FP8` | fp8, fp8_e4m3, fp8_e5m2 | `vllm/config/cache.py:19-25`; `vllm/model_executor/layers/quantization/kv_cache.py:42-191` | - | no quantized KV cache | `planned: specs/fp8-kv-cache.md` | `INVENTORIED` | - |
| `QUANT-KV-FP8-VENDOR` | fp8_inc, fp8_ds_mla | `vllm/config/cache.py:24-25`; vendor KV implementations selected by attention backend | - | no quantized KV cache | `planned: specs/vendor-fp8-kv-cache.md` | `INVENTORIED` | - |
| `QUANT-KV-TURBO` | k8v4, 4bit_nc, k3v4_nc, 3bit_nc | `vllm/config/cache.py:28-33`; TurboQuant dependency path | - | no quantized KV cache | `planned: specs/turboquant-kv-cache.md` | `INVENTORIED` | - |
| `QUANT-KV-PER-HEAD` | int4/int8/fp8 per-token-head | `vllm/config/cache.py:34`; quantized cache kernels selected by backend | - | no quantized KV cache | `planned: specs/per-head-kv-cache.md` | `INVENTORIED` | - |
| `QUANT-KV-NVFP4` | NVFP4 KV | `vllm/config/cache.py:35`; CUDA FP4 cache kernel family | - | no quantized KV cache | `planned: specs/nvfp4-kv-cache.md` | `INVENTORIED` | - |

## 4. MLX-native quantization

These rows are required for the Apple backend, independent of CUDA/vLLM
serialization. Correctness is against MLX/MLX-LM; performance is against oMLX
and MLX-LM on the same Mac, model, quant and concurrency.

| ID | Item / required matrix | Upstream | Our code | Tests/evidence | Spike/spec | State | Owner |
|---|---|---|---|---|---|---|---|
| `QUANT-MLX-AFFINE-Q2` | affine 2-bit; groups 32/64/128 | MLX `quantized.py:11-95`; `ops.cpp:4624-4693` | - | upstream `test_quantized.py:11-170`; local absent | `planned: specs/mlx-affine-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-AFFINE-Q3` | affine 3-bit; groups 32/64/128 | same MLX primitive/module path | - | upstream `test_quantized.py:11-170`; local absent | `planned: specs/mlx-affine-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-AFFINE-Q4` | affine 4-bit; groups 32/64/128; Q4-G64 first | same MLX primitive/module path | - | upstream `test_quantized.py:11-170`; local absent | `planned: specs/mlx-affine-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-AFFINE-Q5` | affine 5-bit; groups 32/64/128 | same MLX primitive/module path | - | upstream `test_quantized.py:11-170`; local absent | `planned: specs/mlx-affine-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-AFFINE-Q6` | affine 6-bit; groups 32/64/128 | same MLX primitive/module path | - | upstream `test_quantized.py:11-170`; local absent | `planned: specs/mlx-affine-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-AFFINE-Q8` | affine 8-bit; groups 32/64/128 | same MLX primitive/module path | - | upstream `test_quantized.py:11-170`; local absent | `planned: specs/mlx-affine-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-MXFP4` | E2M1 + E8M0; group 32 | MLX MX quantization mode in `ops.cpp:4624-4693` | - | upstream `test_quantized.py` MX cases; local absent | `planned: specs/mlx-mx-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-MXFP8` | E4M3 + E8M0; group 32 | same MLX MX primitive path | - | upstream `test_quantized.py` MX cases; local absent | `planned: specs/mlx-mx-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-NVFP4` | E2M1 + E4M3 scale; group 16; optional global scale | MLX NVFP4 mode in `ops.cpp:4624-4693` | - | upstream `test_quantized.py` NVFP4 cases; local absent | `planned: specs/mlx-nvfp4-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-QQ-NVFP4` | quantized activations + weights; tensor-global restriction separate | MLX quantized-quantized matmul implementation | - | upstream `test_quantized.py:227-251`; local absent | `planned: specs/mlx-qq-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-QQ-MXFP8` | quantized activations + weights; supported QQ shapes | MLX quantized-quantized matmul implementation | - | upstream `test_quantized.py` QQ cases; local absent | `planned: specs/mlx-qq-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-MIXED` | mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6 per-layer recipes | MLX-LM `convert.py:20-227` at `a790972f` | - | upstream MLX-LM conversion tests; local absent | `planned: specs/mlx-mixed-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-FINE` | per-layer bits/group/mode plus skip rules | MLX-LM `convert.py:20-227` at `a790972f` | - | upstream MLX-LM conversion tests; local absent | `planned: specs/mlx-fine-grained-quant.md` | `INVENTORIED` | - |
| `QUANT-MLX-IMPORT` | AWQ/GPTQ 4b, legacy MXFP4, compressed-tensors, BitNet transforms | MLX-LM compatibility transforms at `a790972f` | - | upstream MLX-LM conversion tests; local absent | `planned: specs/mlx-quant-import.md` | `INVENTORIED` | - |

## 5. Policy order

1. Preserve the three gate-specific CUDA rows and generalize their dispatch.
2. Make GGUF compute-in-quant the speed foundation; loader-only breadth cannot
   satisfy llama.cpp parity.
3. Close common GGUF and APEX blocker encodings, then IQ/TQ/Q1 breadth.
4. Close generic vLLM FP8/MX/KV formats.
5. Bring up MLX affine Q4-G64, expand the native matrix, then oMLX gates.
6. Port AWQ/GPTQ/CT integer and the remaining registry methods.
