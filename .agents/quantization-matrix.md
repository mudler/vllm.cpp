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

## 1. llama.cpp / GGUF encodings

Upstream encodings: `ggml/include/ggml.h:389-432`. Our recognition is
[gguf_reader.cpp:191](../src/vllm/model_executor/model_loader/gguf_reader.cpp#L191);
materialization is [gguf_dequant.cpp:246](../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L246);
all executable GGUF weights currently expand to bf16 at
[qwen3_5_gguf_weights.cpp:105](../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L105).

| ID | Encoding | ggml ID / role | R | M | C | E | P | State | Code + test evidence | Leaf spec / owner |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|
| `QUANT-GGUF-F32` | F32 | 0 / output | Y | Y | - | Y | - | `PARTIAL` | dequant `:246-250`; [unit](../tests/vllm/test_gguf_dequant.cpp#L25); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L143) | coverage spike; leaf open |
| `QUANT-GGUF-F16` | F16 | 1 / output | Y | - | - | - | - | `INVENTORIED` | reader `:197-200`; executable path absent | leaf open |
| `QUANT-GGUF-Q4_0` | Q4_0 | 2 / output | Y | Y | - | - | - | `PARTIAL` | dequant `:43-57,251-254`; [unit](../tests/vllm/test_gguf_dequant.cpp#L49) | leaf open |
| `QUANT-GGUF-Q4_1` | Q4_1 | 3 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-Q5_0` | Q5_0 | 6 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-Q5_1` | Q5_1 | 7 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-Q8_0` | Q8_0 | 8 / output | Y | Y | - | Y | - | `PARTIAL` | dequant `:59-70,255-258`; [unit](../tests/vllm/test_gguf_dequant.cpp#L35); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L149) | leaf open |
| `QUANT-GGUF-Q2_K` | Q2_K | 10 / output | Y | - | - | - | - | `INVENTORIED` | reader `:209-212`; executable path absent | leaf open |
| `QUANT-GGUF-Q3_K` | Q3_K | 11 / output + S/M/L presets | Y | Y | - | Y | - | `PARTIAL` | dequant `:72-117,259-262`; [unit](../tests/vllm/test_gguf_dequant.cpp#L164); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L143) | leaf open |
| `QUANT-GGUF-Q4_K` | Q4_K | 12 / output + S/M presets | Y | Y | - | Y | - | `PARTIAL` | dequant `:119-146,263-266`; [unit](../tests/vllm/test_gguf_dequant.cpp#L77); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L143) | leaf open |
| `QUANT-GGUF-Q5_K` | Q5_K | 13 / output + S/M presets | Y | Y | - | Y | - | `PARTIAL` | dequant `:148-183,267-270`; [unit](../tests/vllm/test_gguf_dequant.cpp#L105); [APEX gate](../tests/parity/test_qwen36_gguf_engine.cpp#L149) | leaf open |
| `QUANT-GGUF-Q6_K` | Q6_K | 14 / output | Y | Y | - | Y | - | `PARTIAL` | dequant `:185-219,271-274`; [unit](../tests/vllm/test_gguf_dequant.cpp#L134); both APEX gates | leaf open |
| `QUANT-GGUF-IQ2_XXS` | IQ2_XXS | 16 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-IQ2_XS` | IQ2_XS | 17 / output; IQ2_S preset storage | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-IQ3_XXS` | IQ3_XXS | 18 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-IQ1_S` | IQ1_S | 19 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-IQ4_NL` | IQ4_NL | 20 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-IQ3_S` | IQ3_S | 21 / IQ3 S/XS/M storage | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-IQ2_S` | IQ2_S | 22 / IQ2_M storage | Y | - | - | - | - | `INVENTORIED` | reader `:229-234`; [explicit rejection](../tests/vllm/test_gguf_dequant.cpp#L223) | leaf open |
| `QUANT-GGUF-IQ4_XS` | IQ4_XS | 23 / output | Y | - | - | - | - | `INVENTORIED` | reader `:235-240`; [explicit rejection](../tests/vllm/test_gguf_dequant.cpp#L223) | leaf open |
| `QUANT-GGUF-IQ1_M` | IQ1_M | 29 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-BF16` | BF16 | 30 / output | Y | - | - | - | - | `INVENTORIED` | reader `:261-264`; executable path absent | leaf open |
| `QUANT-GGUF-TQ1_0` | TQ1_0 | 34 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-TQ2_0` | TQ2_0 | 35 / output | - | - | - | - | - | `INVENTORIED` | - | leaf open |
| `QUANT-GGUF-MXFP4` | MXFP4 | 39 / MoE output | Y | - | - | - | - | `INVENTORIED` | reader `:265-270`; executable path absent | leaf open |
| `QUANT-GGUF-NVFP4` | NVFP4 | 40 / local fork storage | Y | - | - | - | - | `INVENTORIED` | reader `:271-279`; no quantize preset or executable path | [gguf NVFP4 notes](specs/gguf-nvfp4-notes.md) |
| `QUANT-GGUF-Q1_0` | Q1_0 | 41 / local fork output | Y | - | - | - | - | `INVENTORIED` | reader `:280-285`; executable path absent | leaf open |

Internal dot/scalar tensor types Q8_1, Q8_K, I8/I16/I32/I64/F64 remain parser
concerns, not output-quantization completion rows. Removed GGUF types 4/5,
31-33 and 36-38 are intentionally not roadmap targets.

### llama.cpp output presets

This table prevents encoding support from being mistaken for mixed-preset
coverage. Source: `tools/quantize/quantize.cpp:34-74`.

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

| ID | Method / scheme | W/A/KV | Backend scope | R | M | C | E | P | State | Our code + tests | Spike / owner |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|
| `QUANT-NVFP4-MO-W4A16` | ModelOpt NVFP4 W4A16 | W4/A16 | CUDA sm121a, 35B experts | Y | Y | Y | Y | Y | `DONE` | [weights](../src/vllm/model_executor/models/qwen3_5_weights.cpp#L191), [Marlin dispatch](../src/vllm/model_executor/models/qwen3_5.cpp#L2562), [kernel tests](../tests/vt/test_ops_moe_grouped.cpp#L453), [35B gate](../tests/parity/test_qwen36_paged_engine.cpp#L78) | [Marlin spike](specs/marlin-dropin-feasibility.md); closing main history |
| `QUANT-NVFP4-CT-W4A4` | compressed-tensors NVFP4 | W4/A4 | CUDA sm121a, 27B dense | Y | Y | Y | Y | Y | `DONE` | [weights](../src/vllm/model_executor/models/qwen3_5_dense_weights.cpp#L106), [dispatch](../src/vllm/model_executor/models/qwen3_5.cpp#L2836), [kernel tests](../tests/vt/test_ops_nvfp4_fp4.cpp#L48), [27B gate](../tests/parity/test_qwen27_paged_engine.cpp#L110) | [W4A4 notes](specs/qwen27b-w4a4-notes.md); closing main history |
| `QUANT-FP8-MO-STATIC` | ModelOpt static per-tensor FP8 | W8/A8 | CUDA sm121a, 35B projections | Y | Y | Y | Y | Y | `DONE` | [weights](../src/vllm/model_executor/models/qwen3_5_weights.cpp#L118), [dispatch](../src/vllm/model_executor/models/qwen3_5.cpp#L627), [kernel tests](../tests/vt/test_ops_fp8_cutlass.cpp#L188), 35B gate | [forward notes](specs/qwen36-forward-notes.md); closing main history |
| `QUANT-FP8-GENERIC` | fp8 + ModelOpt FP8 breadth | static/dynamic, tensor/channel/token/block | all vLLM backends/SM dispatch | part | part | part | part | - | `PARTIAL` | only the gate-specific row above exists | leaf spec open |
| `QUANT-FP8-PB-WO` | ModelOpt FP8 per-block weight-only | W8/A16 | capability selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-FP8-PCPT` | ModelOpt FP8 per-channel/per-token | W8/A8 | capability selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-MXFP8-MODELOPT` | ModelOpt MXFP8 | W8/A8 | CUDA/ROCm/XPU dispatch | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-MIXED-MODELOPT` | ModelOpt mixed precision | FP8/NVFP4/MXFP8 groups | per-layer | part | part | part | Y | Y | `PARTIAL` | 35B exercises FP8 + NVFP4 subset only | leaf spec open |
| `QUANT-CT-MXFP4` | compressed-tensors W4A4 MXFP4 | W4/A4 | capability selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-CT-W4A8-FP8` | compressed-tensors W4A8 FP8 | W4/A8 | CUTLASS | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-CT-W4A8-INT8` | compressed-tensors W4A8 INT8 | W4/A8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-CT-W8A8-FP8` | compressed-tensors W8A8 FP8 | W8/A8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-CT-W8A8-INT8` | compressed-tensors W8A8 INT8 | W8/A8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-CT-W8A8-MXFP8` | compressed-tensors W8A8 MXFP8 | W8/A8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-CT-W8A16-FP8` | compressed-tensors FP8 weight-only | W8/A16 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-CT-WNA16` | compressed-tensors WnA16 | symmetric 2-8b; asymmetric 4/8b | Marlin/Machete/etc. | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-CT-WNA8O8` | compressed-tensors WnA8O8 | W2/4/8, static int8 A/O | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-AWQ` | AWQ / AutoAWQ / AWQ-Marlin | W4/A16 | Marlin/CUDA/CPU variants | - | - | - | - | - | `INVENTORIED` | vendored Marlin is NVFP4-only today | leaf spec open |
| `QUANT-GPTQ` | GPTQ / AutoGPTQ / GPTQ-Marlin | W2-8/A16 | Marlin/CUDA/CPU variants | - | - | - | - | - | `INVENTORIED` | vendored Marlin is NVFP4-only today | leaf spec open |
| `QUANT-BNB` | bitsandbytes | FP4/NF4/INT8 | CUDA library path | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-EXPERTS-I8` | experts-int8 | MoE int8 | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-MOE-WNA16` | MoE WnA16 | W4/8-A16 | Marlin/Machete/ROCm | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-MXFP4-GPTOSS` | MXFP4 / GPT-OSS MXFP4 | W4/A16 or W4/A4 | model/kernel selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-QUARK` | AMD Quark | mixed int/fp | ROCm | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-TORCHAO` | TorchAO | multiple | compatibility path | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-INC` | Intel INC / AutoRound | WnA16 etc. | CPU/XPU | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-HUMMING` | Humming | weight-only | CUDA | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-DSV4-FP8` | DeepSeek-V4 FP8 | model-specific | CUDA | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-ONLINE` | online FP8/MXFP8/int8 shorthands | runtime W/A | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open |
| `QUANT-DEPRECATED` | FBGEMM-FP8, FPQuant | compatibility only | platform selected | - | - | - | - | - | `INVENTORIED` | - | leaf spec open; do not prioritize |

## 3. KV-cache quantization

Pinned vLLM source: `vllm/config/cache.py:19-36`.

| ID | Cache dtype/mode | Our state | Evidence | Spike |
|---|---|---|---|---|
| `QUANT-KV-FP8` | fp8, fp8_e4m3, fp8_e5m2 | `INVENTORIED` | no quantized KV cache | leaf open |
| `QUANT-KV-FP8-VENDOR` | fp8_inc, fp8_ds_mla | `INVENTORIED` | no quantized KV cache | leaf open |
| `QUANT-KV-TURBO` | k8v4, 4bit_nc, k3v4_nc, 3bit_nc | `INVENTORIED` | no quantized KV cache | leaf open |
| `QUANT-KV-PER-HEAD` | int4/int8/fp8 per-token-head | `INVENTORIED` | no quantized KV cache | leaf open |
| `QUANT-KV-NVFP4` | NVFP4 KV | `INVENTORIED` | no quantized KV cache | leaf open |

## 4. MLX-native quantization

These rows are required for the Apple backend, independent of CUDA/vLLM
serialization. Correctness is against MLX/MLX-LM; performance is against oMLX
and MLX-LM on the same Mac, model, quant and concurrency.

| ID | MLX mode | Required matrix | R/M/C/E/P | State | Upstream tests | Leaf spec |
|---|---|---|---|---|---|---|
| `QUANT-MLX-AFFINE-Q2` | affine 2-bit | groups 32/64/128 | `-/-/-/-/-` | `INVENTORIED` | `test_quantized.py:11-170` | open |
| `QUANT-MLX-AFFINE-Q3` | affine 3-bit | groups 32/64/128 | `-/-/-/-/-` | `INVENTORIED` | same | open |
| `QUANT-MLX-AFFINE-Q4` | affine 4-bit | groups 32/64/128; Q4-G64 first | `-/-/-/-/-` | `INVENTORIED` | same | open |
| `QUANT-MLX-AFFINE-Q5` | affine 5-bit | groups 32/64/128 | `-/-/-/-/-` | `INVENTORIED` | same | open |
| `QUANT-MLX-AFFINE-Q6` | affine 6-bit | groups 32/64/128 | `-/-/-/-/-` | `INVENTORIED` | same | open |
| `QUANT-MLX-AFFINE-Q8` | affine 8-bit | groups 32/64/128 | `-/-/-/-/-` | `INVENTORIED` | same | open |
| `QUANT-MLX-MXFP4` | E2M1 + E8M0 | group 32 | `-/-/-/-/-` | `INVENTORIED` | `test_quantized.py` MX cases | open |
| `QUANT-MLX-MXFP8` | E4M3 + E8M0 | group 32 | `-/-/-/-/-` | `INVENTORIED` | same | open |
| `QUANT-MLX-NVFP4` | E2M1 + E4M3 scale | group 16; optional global scale | `-/-/-/-/-` | `INVENTORIED` | same | open |
| `QUANT-MLX-QQ-NVFP4` | quantized activations + weights | tensor-global-scale restriction separate | `-/-/-/-/-` | `INVENTORIED` | `test_quantized.py:227-251` | open |
| `QUANT-MLX-QQ-MXFP8` | quantized activations + weights | supported QQ shapes | `-/-/-/-/-` | `INVENTORIED` | `test_quantized.py` | open |
| `QUANT-MLX-MIXED` | mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6 | per-layer recipe | `-/-/-/-/-` | `INVENTORIED` | MLX-LM conversion tests | open |
| `QUANT-MLX-FINE` | per-layer bits/group/mode + skip | config/import | `-/-/-/-/-` | `INVENTORIED` | MLX-LM conversion tests | open |
| `QUANT-MLX-IMPORT` | AWQ/GPTQ 4b, legacy MXFP4, CT, BitNet | compatibility transforms | `-/-/-/-/-` | `INVENTORIED` | MLX-LM conversion tests | open |

## 5. Policy order

1. Preserve the three gate-specific CUDA rows and generalize their dispatch.
2. Make GGUF compute-in-quant the speed foundation; loader-only breadth cannot
   satisfy llama.cpp parity.
3. Close common GGUF and APEX blocker encodings, then IQ/TQ/Q1 breadth.
4. Close generic vLLM FP8/MX/KV formats.
5. Bring up MLX affine Q4-G64, expand the native matrix, then oMLX gates.
6. Port AWQ/GPTQ/CT integer and the remaining registry methods.
