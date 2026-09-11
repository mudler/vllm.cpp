ID: ISSUE-GH-2065
Title: KV-FP8 W6: the ROCm fp8-e4m3 KV cache store and read
Row: KV-FP8
State: CLOSED
Kind: feature
GitHub: 2065
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-27
Updated: 2026-08-28
Closed: 2026-08-28

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> The ROCm arm of the fp8 KV cache (`KV-FP8`). W1 landed the CPU fp8-e4m3
> store + read dequant + config parse. W2 landed the CUDA fp8-e4m3 store +
> paged-attention read (code landed; device gates unexecuted). W3 landed the
> runner integration (half-sized KV blocks, `--kv-cache-dtype`, checkpoint
> scale threading). The ROCm arm is named as owed in the spec's `## Owed`
> section and refused by name in `src/vt/ops.cpp:3835`.
>
> This issue tracks the ROCm fp8-e4m3 K/V store kernel
> (`vt::ReshapeAndCacheFp8` for `DeviceType::kROCM`) and the fp8 dequant on
> the ROCm paged-attention read, mirroring the CUDA W2 arm element-for-element.
>
> ## Why
>
> The fp8 KV cache halves the KV footprint. The ROCm backend (`BACKEND-ROCM`)
> already has 44 registered ops including paged attention and the full GDN
> family, but no fp8 KV arm. An fp8 cache on ROCm currently reaches a float
> kernel and returns silent garbage, which the op wrapper refuses with a
> named-later-brick message. This issue closes that gap.
>
> ## Scope
>
> - **In:** the ROCm fp8-e4m3 K/V store kernel (port of the CUDA
>   `ReshapeAndCacheFp8Kernel` in `src/vt/cuda/cuda_cache.cu:155-226`),
>   the fp8 dequant read in ROCm paged attention (port of `LoadKv` in
>   `src/vt/cuda/cuda_paged_attn.cu:175-185`), the `OpId::kReshapeAndCacheFp8`
>   registration for `DeviceType::kROCM`, and the widening of the
>   `src/vt/ops.cpp` fp8 read refusal to admit `kROCM`.
> - **Out:** fp8_e5m2 compute, per-attention-head scales, the Metal arm,
>   fast-path (tensor-core) fp8 attention kernels, and the memory-halving e2e
>   measurement on a ROCm gate model.
>
> ## Upstream reference
>
> Pinned vLLM `555967922`. The fp8 KV path is vLLM's own csrc:
> `reshape_and_cache_flash_kernel` (`csrc/libtorch_stable/cache_kernels.cu:
> 314-401`) + `CopyWithScaleOp` (`:241-252`); the read dequant
> `scaled_vec_conversion<float, uint8_t>` (`csrc/quantization/w8a8/fp8/
> nvidia/quant_utils.cuh:419-429`). The CUDA arm is the direct template;
> the ROCm arm is elementwise-identical to it.
>
> ## Tracking
>
> Row: `KV-FP8` (engine-matrix). Spec: `.agents/specs/fp8-kv-cache.md`
> (new `## W6` section). PR shape: one PR for spec and implementation
> (repository default).

## Resolution

GitHub records closing pull request #2080 (https://github.com/mudler/vllm.cpp/pull/2080) merged on 2026-08-28 as commit `191f64608a36fced165a4f3202ce088f6b283e67`. GitHub closed issue #2065 on 2026-08-28.
