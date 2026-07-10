# Spike: quantization coverage inventory

**Rows:** `QUANT-*` · **state:** coverage spike complete; leaf implementation
specs still required · **pins:** vLLM `e24d1b24`, llama.cpp `237ad9b` plus the
local NVFP4/Q1_0 fork additions, MLX `4367c73b`, MLX-LM `a790972f`.

This spike defines the complete quantization surface and decomposes it into
row-sized work. It does not make an implementation claim; current support is
grounded in [quantization-matrix.md](../quantization-matrix.md).

## Scope and status dimensions

A serialization format is not equivalent to a production quantization path.
Every leaf is tracked across five independent dimensions:

| Dimension | Completion rule |
|---|---|
| Recognize | Parser/config registry validates the format and byte geometry |
| Materialize | Dequant/repack/scale transform is implemented and oracle-tested |
| Compute | Native quantized kernel is selected; load-time bf16 expansion does not count |
| E2E | Representative real checkpoint passes correctness on the target backend |
| Perf | Same-file/reference-engine benchmark meets the backend acceptance floor |

Mixed-file presets are tracked separately from encodings. One decoder can
enable several presets, but each preset needs a representative real-file gate.

## Upstream chain

| Surface | Authoritative source |
|---|---|
| vLLM method registry | `/home/mudler/_git/vllm/vllm/model_executor/layers/quantization/__init__.py:12-46` |
| vLLM ModelOpt modes | `/home/mudler/_git/vllm/vllm/model_executor/layers/quantization/modelopt.py:104-120` |
| vLLM compressed-tensors schemes | `/home/mudler/_git/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/__init__.py:4-27` |
| vLLM KV cache dtypes | `/home/mudler/_git/vllm/vllm/config/cache.py:19-36` |
| llama.cpp storage types | `/home/mudler/_git/llama.cpp/ggml/include/ggml.h:389-432` |
| llama.cpp output presets | `/home/mudler/_git/llama.cpp/tools/quantize/quantize.cpp:34-74` |
| MLX primitive modes | `mlx/python/src/ops.cpp:4624-4693` at `4367c73b` |
| MLX quantized modules/tests | `mlx/python/mlx/nn/layers/quantized.py:11-95`; `mlx/python/tests/test_quantized.py:11-251` |
| MLX-LM conversion recipes | `mlx_lm/convert.py:20-227` at `a790972f` |

Runtime selection must follow the whole dependency chain. For vLLM this
includes CUTLASS, FlashInfer, Marlin/Machete, DeepGEMM, cuBLASLt and generated
Triton/Inductor paths. A leaf spec must trace which implementation actually ran
on each target architecture before making performance claims.

## Current baseline and corrections

| Finding | Grounded baseline |
|---|---|
| GGUF reader breadth | [gguf_reader.cpp](../../src/vllm/model_executor/model_loader/gguf_reader.cpp) recognizes only the types listed at lines 191-285 |
| GGUF executable dequant | [gguf_dequant.cpp](../../src/vllm/model_executor/model_loader/gguf_dequant.cpp) dispatches only F32, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K and Q6_K at lines 246-279 |
| GGUF compute | [qwen3_5_gguf_weights.cpp](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp) expands weights to bf16 at lines 105-154; no compute-in-quant path exists |
| GGUF F16 | Reader-only; the former feature-matrix claim was incorrect |
| Gate-specific FP8 | Qwen3.6-35B ModelOpt static per-tensor W8A8 is native and gated; generic vLLM FP8 breadth remains open |
| GGUF scope | Qwen3.5-MoE text loader only; not a generic GGUF/model-family claim |

The local llama.cpp checkout is dirty because types 40/41 are fork additions.
Pin that patch as an explicit oracle revision before closing those rows.

## Work decomposition and dependencies

| Order | Row block | Dependency / outcome |
|---:|---|---|
| 1 | `QUANT-GGUF-COMPUTE` | Quantized tensor storage in `vt::` plus llama.cpp-style compute-in-quant dispatch; prerequisite for honest CPU speed parity |
| 2 | Common GGUF scalar/legacy/K types | F16/BF16, Q4_1, Q5_0, Q5_1, Q2_K; then native kernels for already-materialized Q4/Q5/Q6/Q8/K types |
| 3 | Real-file APEX blockers | IQ2_S and IQ4_XS materialization + compute + gates |
| 4 | Remaining IQ/TQ/Q1 types | Port layouts and dot kernels from llama.cpp; one leaf spec per encoding family |
| 5 | GGUF MXFP4/NVFP4 | Reuse native kernels only after scale-layout compatibility is proven |
| 6 | Generic vLLM dispatch | Lift gate-specific FP8/NVFP4 into quant-config/linear-method selection |
| 7 | Generic FP8/MX and KV | FP8, MXFP4/MXFP8 and cache dtype matrix by architecture |
| 8 | MLX native | affine Q4-G64 first, complete affine bits/groups, MX modes, QQ, recipes; oMLX/MLX-LM gates |
| 9 | AWQ/GPTQ/CT integer | Reuse Marlin/CUTLASS adapters after C1 kernel ABI spike |
| 10 | Remaining vLLM registry | bitsandbytes, Quark, TorchAO, INC, Humming and specialized formats |

Each row block must be split into backend-specific claims when kernels differ.

## Tests to port

| Upstream tests | Local tier / obligation |
|---|---|
| llama.cpp `tests/test-quantize-fns.cpp`, `test-quant-type-selection.cpp` | per-encoding layout, selection and dot-product oracle tests |
| llama.cpp `tests/test-quantize-stats.cpp`, `test-quantize-perf.cpp` | quality statistics and microbenchmark evidence; never replace e2e gates |
| llama.cpp `tests/test-gguf.cpp`, `test-gguf-model-data.cpp` | parser/metadata/span and representative-file cases |
| vLLM `tests/kernels/quantization/test_{awq,gptq,marlin_gemm,machete_mm}.py` | weight-only method/kernel parity |
| vLLM `tests/kernels/quantization/test_{fp8_quant,cutlass_scaled_mm,block_fp8}.py` | FP8 quantization and scaled GEMM parity |
| vLLM `tests/kernels/quantization/test_{nvfp4_quant,nvfp4_scaled_mm,flashinfer_nvfp4_scaled_mm}.py` | NVFP4 quant/linear dispatch parity |
| vLLM `tests/models/quantization/test_{fp8,modelopt,nvfp4,mxfp4,mxfp8}.py` | checkpoint/config/e2e model cases |
| vLLM `tests/quantization/test_{compressed_tensors,mixed_precision,online,turboquant}.py` | scheme selection and cache/online breadth |
| MLX `python/tests/test_quantized.py` | affine/MX/NV modes, shapes, gradients where relevant, invalid QQ global-scale case |

A ported case that cannot pass is checked in skipped with a row ID and concrete
reason, per [test-porting.md](../test-porting.md).

## Gates

| Backend / format | Correctness oracle | Performance oracle |
|---|---|---|
| CUDA vLLM formats | pinned pip-vLLM, identical checkpoint/config/seed | vLLM every-axis protocol on the target SM |
| CPU GGUF | pinned same-file llama.cpp, tokens/logits and memory | llama.cpp prompt/decode throughput, TTFT and peak memory; match or beat |
| CUDA GGUF | same-file llama.cpp plus CPU reference | fastest applicable llama.cpp CUDA path and vLLM when the format is shared |
| Apple MLX | MLX primitive/MLX-LM token parity | oMLX and MLX-LM on the same Mac/model/quant/concurrency |

All runs use the exact recipe and repetition rules in
[benchmark-protocol.md](../benchmark-protocol.md). GPU series hold one
`flock /tmp/gpu` for every arm.

## Risks and decisions

- Reader recognition, dequantization and direct compute are not interchangeable.
- Encoding coverage does not imply every llama.cpp mixed preset is gated.
- GGUF scale layouts can differ from safetensors ModelOpt/compressed-tensors
  despite sharing names such as NVFP4/MXFP4.
- MLX quantized matmul has mode-specific restrictions; tensor-global-scale
  NVFP4 QQMM is explicitly rejected upstream and stays a separate cell.
- Quality/perplexity policy is a product decision only where upstream engines
  offer different accepted presets. Dispatch behavior itself mirrors upstream.
