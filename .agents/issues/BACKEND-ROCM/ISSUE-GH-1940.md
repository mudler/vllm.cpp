ID: ISSUE-GH-1940
Title: ROCm keep-quant still lacks Q4_0, Q5_0, IQ2_XS, IQ4_NL, IQ3_S, IQ4_XS, and MXFP4
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 1940
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-25
Updated: 2026-08-25
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `src/vt/rocm/` ports zero I-quant formats. `MatmulBTQuantKernelRocm`'s refusal
> message at `rocm_grouped_gemm.hip:495` names the gap directly:
> `"ported: Q8_0/Q4_K/Q5_K/Q6_K; owed: Q4_0/Q2_K/Q3_K/IQ2_XXS/IQ3_XXS/IQ2_S/MXFP4"`.
> The grouped launcher's refusal at `:565` repeats the same list.
> `.agents/specs/rocm-gg-keep-quant.md` `## Boundaries` already records this gap,
> but no issue has tracked it until now.
>
> ## What already exists to port from
>
> CUDA and CPU both implement five I-quant formats. Porting each of these five
> is an adaptation of existing source, not a from-scratch port.
>
> | Format | CUDA dot | Oracle |
> |---|---|---|
> | `IQ2_XXS` | `DotIQ2XXS`, `cuda_quant_dot.cu:307` | llama.cpp `quants.c:855`, port note at `cuda_quant_dot.cu:298` |
> | `IQ3_XXS` | `DotIQ3XXS`, `cuda_quant_dot.cu:340` | llama.cpp `quants.c:999` |
> | `IQ2_S` | `DotIQ2S`, `cuda_quant_dot.cu:432` | llama.cpp `quants.c:947`, structure mirrors `DotIQ3XXS` |
> | `IQ1_S` | `DotIQ1S`, `cuda_quant_dot.cu:383` | llama.cpp `quants.c:1099` |
> | `IQ1_XXXS` | `DotIQ1XXXS`, `cuda_quant_dot.cu:407` | `unslothai/llama.cpp` fork `quants.c:1281`, registry id `llama-cpp-unsloth` in `.agents/oracles/`. This is the sub-`IQ1_S` encoding that a published Qwen3.8-2.4T checkpoint stores 96.92% of its routed experts in, per the port note at `cuda_quant_dot.cu:710`. |
>
> CPU and CUDA do not port `IQ1_M`, `IQ2_XS`, `IQ3_S`, `IQ4_NL`, or `IQ4_XS`
> either. Those five need a fresh port from the llama.cpp oracle instead of an
> adaptation.
>
> All five already-ported formats use the same 256-element superblock and the
> same `BlockQ8_K` activation format as `Q4_K`, `Q5_K`, and `Q6_K`
> (`kQK_K`, `rocm_grouped_gemm.hip:41`). A ROCm port of any of them dispatches
> through the identical `nsb = K / 256` decomposition that
> [#1910](https://github.com/mudler/vllm.cpp/issues/1910) covers, including its
> lane-idle defect at small `nsb`, if that row lands first.
>
> ## Why this is more than coverage
>
> [#1910](https://github.com/mudler/vllm.cpp/issues/1910) ports llama.cpp's
> `nwarps`-scaling decode dispatch from `ggml/src/ggml-cuda/mmvq.cu`, pin
> `10bf611e5`, tag `b10451`. That table splits by vec_dot complexity. `Q4_K`,
> `Q5_K`, and `Q6_K` use a linear scale and minimum, and get `nwarps=8` on
> RDNA4. llama.cpp excludes `Q3_K` and the `IQ2` and `IQ3` families by name,
> because their vec_dot does a grid lookup. The comment at `mmvq.cu:387-388`
> states the reason: it regresses from register pressure and lookup table
> contention at higher thread counts.
>
> We have no data on whether that split holds for our own dot product bodies.
> The existing spec `rocm-kquant-nwarps-decode.md` measures only the simple
> side. Porting `IQ2_XXS` or `IQ3_XXS` gives a complex vec_dot body to run the
> same `nwarps` sweep against on real gfx1200 hardware. Both formats are
> already adaptable from the CUDA source above, so this tests llama.cpp's
> split against our own kernel bodies instead of assuming it transfers from a
> different vec_dot implementation on different hardware.
>
> ## What is NOT established
>
> - Whether the HIP translation of `DotIQ2XXS` or `DotIQ3XXS` compiles to
>   efficient code on gfx1200. The grid lookup dequant can use different
>   constant memory or cache behavior on RDNA4 than on the NVIDIA and CDNA
>   hardware that tuned llama.cpp's table.
> - No speed or correctness claim is made here. This issue states scope and
>   motivation only.
> - The keep-quant loader wiring, the dequant table adaptation from
>   `cpu_quant_iq_tables.h` and `cuda_quant_iq_tables.cuh` to a ROCm
>   equivalent, and the correctness tests are all unstarted.
>
> Owned by `BACKEND-ROCM`.
>

## Resolution

-
