ID: ISSUE-LOCAL-01M2DB952E2T973AHZBYM27PHA
Title: Three prose sites still say ROCm does not register kEmbeddingQuant; #3093 falsified that
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

docs/FEATURES.md:163 asserts 'METAL, VULKAN, ROCM and TENSTORRENT do not [register kEmbeddingQuant] and are still refused (#2394)', and the same claim is repeated verbatim in code comments at src/vllm/model_executor/models/qwen4_exp_weights.cpp:635-641 and src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:232-240. src/vt/rocm/rocm_ops.hip:207 registers it (BACKEND-ROCM-QUANT-GATHER, #3093), the ROCm arm keeps the 26.822 GiB per_layer_token_embd table block-resident on device, and the measurement in docs/bench-evidence/qwen4exp-ple-placement-gfx1151-20260913.md depends on that being true. A fourth, adjacent site is docs/FEATURES.md:112, whose claim that llama.cpp's CUDA get_rows 'dispatches the legacy quants only' is too strong: the llama-cpp-qwen4exp pin admits Q2_K..Q6_K and most IQ types at ggml/src/ggml-cuda/ggml-cuda.cu:4986-5011 and refuses IQ4_NL/MXFP4 only when ne0 % QK_K != 0. All four move together or none does: repairing one copy of a duplicated fact creates the next contradiction. Not fixed here because the correct replacement text is owned by BACKEND-ROCM-QUANT-GATHER's own record, and because this row's change is records-only.

## Resolution

-
