ID: ISSUE-GH-2394
Title: The block-decoding gather arms for METAL, VULKAN, ROCM and TENSTORRENT
Row: -
State: OPEN
Kind: UNKNOWN
GitHub: 2394
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-MM-QWEN4-EXP`
>
> KGATHER ([`.agents/specs/cuda-quant-gather.md`](https://github.com/mudler/vllm.cpp/blob/main/.agents/specs/cuda-quant-gather.md)) gave the CPU and CUDA backends a block-decoding gather and made the capability a registry fact: `vt::Embedding` routes a block-quantized table to `OpId::kEmbeddingQuant`, and the GGUF residency gate is `OpRegistered(kEmbeddingQuant, dev)`.
>
> **Four backends do not register it, so they keep expand-bf16 residency for any gather table.** Each refuses a block table by name in its own `kEmbedding` kernel today, for example `src/vt/tenstorrent/tenstorrent_ops.cpp`: `"tenstorrent kEmbedding: float table, f32/bf16 out"`.
>
> - `src/vt/metal/metal_ops.mm`
> - `src/vt/vulkan/vulkan_ops.cpp`
> - `src/vt/rocm/rocm_ops.hip`
> - `src/vt/tenstorrent/tenstorrent_ops.cpp`
>
> **Why it matters.** On a device that cannot gather quantized, a table expands. For the released `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` the n-gram table goes from **26.822 GiB of IQ4_NL to 95.368 GiB of bf16**, which fits nothing in this fleet — so on those four devices `qwen4_exp` is refused by name at load, ahead of any tensor I/O.
>
> **What a port needs.** `src/vt/cuda/cuda_quant_dequant.cuh` is the reference shape: one codec per block dtype transliterated from `src/vt/cpu/cpu_quant_dequant.cpp` (itself the byte-for-byte port of `ggml-quants.c`'s `dequantize_row_*`), plus a gather kernel. Two things are not optional and are learned rather than obvious:
>
> 1. **Unaligned reads.** A block base is not 4-byte aligned in general (66-byte IQ2_XXS, 110-byte Q3_K, 210-byte Q6_K), and a misaligned multi-byte device load faults rather than being slow.
> 2. **The gate must be an EXACT comparison against the CPU arm**, not a tolerance. A tolerance hides a decode that produces plausible numbers and no crash. `tests/vt/test_cuda_embedding_quant.cpp` is the pattern, including a build guard that refuses to run a stale binary when a mutation fails to compile.
>
> Registering `kEmbeddingQuant` without writing the decoder would convert a clean load-time refusal into a forward-time throw with the whole model resident — the #523 failure — so the registration and the kernel must land together, and only after the arm has executed on that device.

## Resolution

-
