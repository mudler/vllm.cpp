ID: ISSUE-GH-1959
Title: QUANT-QWEN38-27B-NVFP4-ARM: qwen36 compressed-tensors mixed-precision FP8 group_0 is refused, blocking an apples-to-apples vllm.cpp vs vLLM comparison
Row: QUANT-QWEN38-27B-NVFP4-ARM
State: OPEN
Kind: UNKNOWN
GitHub: 1959
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-26
Updated: 2026-08-26
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Blocker
>
> `/data/models/Qwen/Qwen3.6-27B-NVFP4` (the checkpoint nsk/vLLM serves) and its
> `-vllmcpp` sibling declare `quantization_config` = `compressed-tensors`,
> `format: "mixed-precision"`, with two groups:
>
> - `group_0`: FP8 `float-quantized`, per-output-channel `weight_scale` and
>   DYNAMIC per-token activation quantization (`input_activations.dynamic: true,
>   strategy: token`).
> - `group_1`: NVFP4 `nvfp4-pack-quantized`.
>
> Current `main` loads `group_1` but refuses `group_0` in
> `qwen3_5_dense_weights.cpp` via `compressed_tensors::RefusalForHfConfigRaw`
> (because `LoadFp8Raw` cannot represent a per-channel scale + dynamic activation).
> So `vllm-server` exits `fatal` before reading a weight byte.
>
> ## Why it matters
>
> The vllm.cpp vs vLLM (nsk) performance comparison is currently cross-model
> (vllm.cpp runs the AEON MTP checkpoint, vLLM runs qwen36). To make it
> apples-to-apples, vllm.cpp must load the SAME qwen36 checkpoint vLLM serves.
>
> ## Goal
>
> Implement the compressed-tensors FP8 `float-quantized` per-channel + dynamic
> activation arm so qwen36 loads, then re-run the identical evalscope random
> matrix on both engines over the same checkpoint.
>
> ## Owning row
>
> `QUANT-QWEN38-27B-NVFP4-ARM` (the loader's compressed-tensors mixed-precision
> resolution; spec .agents/specs/qwen38-27b-quant-arms.md)

## Resolution

-
