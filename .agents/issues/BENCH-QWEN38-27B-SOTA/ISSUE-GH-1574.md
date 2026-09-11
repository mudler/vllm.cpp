ID: ISSUE-GH-1574
Title: **Qwen3.8-27B three-way at each engine's best: one checkpoint ours, vLLM and SGLang all serve, and three of the four recorded NVFP4 blockers are properties of the unsloth artifact rather than of the format.** `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`@`36f717a2` carries 208 `input_scale` tensors (unsloth carries ZERO), an `F32` scalar `weight_scale` on every FP8 module (unsloth's is per-channel BF16, which `ReadF32Scalar` refuses on both count and dtype), and `"dynamic": false` on both weights and activations (unsloth needs a dynamic per-token scheme we cannot represent). Its NVFP4 half is W4A16 g16 rather than unsloth's W4A4. Read from `model.safetensors.index.json` and shard 1's safetensors header by HTTP range request on 2026-08-21. Critical path is the CUDA half of `KV-FP8`, which is owed and whose W1 landed CPU-only; `--kv-cache-dtype fp8` is load-bearing for CORRECTNESS here, not only memory
Row: BENCH-QWEN38-27B-SOTA
State: UNKNOWN
Kind: measurement
GitHub: 1574
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:505`

### Frozen archive evidence

> | [#1574](https://github.com/mudler/vllm.cpp/issues/1574) | `BENCH-QWEN38-27B-SOTA` | **Qwen3.8-27B three-way at each engine's best: one checkpoint ours, vLLM and SGLang all serve, and three of the four recorded NVFP4 blockers are properties of the unsloth artifact rather than of the format.** `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`@`36f717a2` carries 208 `input_scale` tensors (unsloth carries ZERO), an `F32` scalar `weight_scale` on every FP8 module (unsloth's is per-channel BF16, which `ReadF32Scalar` refuses on both count and dtype), and `"dynamic": false` on both weights and activations (unsloth needs a dynamic per-token scheme we cannot represent). Its NVFP4 half is W4A16 g16 rather than unsloth's W4A4. Read from `model.safetensors.index.json` and shard 1's safetensors header by HTTP range request on 2026-08-21. Critical path is the CUDA half of `KV-FP8`, which is owed and whose W1 landed CPU-only; `--kv-cache-dtype fp8` is load-bearing for CORRECTNESS here, not only memory | measurement |

## Resolution

-
