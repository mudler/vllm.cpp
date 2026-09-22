ID: ISSUE-LOCAL-01M3554MJ5K9V3BJ5X84YQAZ70
Title: qwen3 GGUF arch not supported in model_loader dispatch
Row: BACKEND-GATE-ROCM-SGLANG
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-22
Updated: 2026-09-22
Closed: 2026-09-22

## Problem

The GGUF dispatch table in model_loader.cpp supports qwen35, qwen35moe, and qwen3next architectures, but NOT plain qwen3. qwen3_dense.cpp:69 explicitly refuses GGUF loading. This blocks the quant-matched C++-only token gate (llama.cpp Q4_K_M vs vllm.cpp Q4_K_M) needed for BACKEND-GATE-ROCM-SGLANG on Strix, where vLLM and SGLang cannot run due to ROCm 5.7 / torch 2.13.0+rocm7.2 incompatibility.

## Resolution

Fixed in PR #3274 (merged as 3c2fcca33). qwen3 GGUF arch support added: Qwen3HfConfigFromGguf, IsQwen3Gguf, LoadQwen3FromGguf in qwen3_gguf_weights.{h,cpp}. Dispatch entry {"qwen3", &Qwen3HfConfigFromGguf} in model_loader.cpp. kGguf branch in qwen3_dense.cpp LoadQwen3ForCausalLM. CPU build passes with -Werror. Tests: test_qwen3_gguf_weights 5/5 cases 117 assertions, test_model_loader_gguf 13/13, test_qwen3_5_gguf_mtp 4/4. Fresh reviewer mutation-tested 6 guarantees (5 caught immediately, 1 coverage gap found and fixed by fresh implementer).
