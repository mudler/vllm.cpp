# DwarfStar, GGUF

DeepSeek-V4-Flash cannot run on vLLM on a single GB10 at all: every
vLLM-loadable checkpoint is 156 GB or larger against a 119 GiB unified pool, so
the only quant that fits is extreme-low-bit GGUF, which vLLM cannot load here.
GGUF was forced by the hardware. A policy-correct vLLM comparison needs 2x GB10
Sparks with TP2 and is owed.

| Engine | Quant | Decode tok/s | Ratio |
|---|---|---:|---:|
| DwarfStar (`ds4`) | IQ2_XXS mixed | 16.33 | 1.00x |
| **vllm.cpp** (default) | same GGUF | **16.28** | **0.997x, parity** |
| **vllm.cpp** (`VT_V4_RESIDENT_W` default-ON) | same GGUF | **18.69** | **1.144x, byte-exact** |

The default arm is parity, measured same-session clean (2026-08-04, single-load
steady both arms); the earlier 15.87/96% and 17.13 figures are superseded.

Weight residency is the beat-path (2026-08-05, `VT_V4_RESIDENT_W`, default-ON). Env var allowlisted (env-doc gate green).
The dense Q8_0 MLA/shared-expert/lm_head projection tower is read from the GGUF
mmap over ATS/unified memory, which the GB10 GPU reads about 20% slower per-GEMV
than `cudaMalloc`'d device memory. Staging that ~6 GiB tower device-resident once
at load (same bytes, same kernel, same invocation) lifts decode 16.23 to 18.69
(median-of-3, drop_caches), generated ids byte-identical. It is the same lever
that took Laguna to vLLM parity+ (`VT_LAGUNA_RESIDENT_BF16W`).

PEAK RESIDENT is flat at 86.68 GiB in both arms: the staged copy is additive but
the clean mmap file pages evict under the unified pool, so net usage does not
grow. An nsys A/B (identical instance counts) confirms the mechanism is
residency-bound, not latency-bound: per-launch time drops about 20% on every
dense Q8_0 kernel (`QuantDotGemmQ8_0Kernel` 184 to 147 µs, `Q8_0GroupDiagKernel`
212 to 166 µs, `Q8_0PairKernel` 74 to 60 µs). This corrects the earlier
"per-launch GEMV parity / Q8_0 weight-stream floor" framing: our GEMV was
ATS-bound, not at ds4 parity.

Phase-2 staged the routed-expert slabs too (the ~70 GiB IQ2/Q2_K bulk,
`VT_V4_RESIDENT_EXPERTS`, first-touch `cudaMalloc` plus immediate
`madvise(MADV_DONTNEED)` per slab so the transient stays ~flat). It was **measured
NEGATIVE (2026-08-05) and is HELD default-OFF** as a characterized artifact.
Same-binary median-of-3, warm-cancelled steady, drop_caches: OFF (Phase-1) **19.43
tok/s** vs ON **18.76 tok/s** (0.966x, ~3.4% slower), generated ids byte-identical
(md5 equal across all 6 runs), PEAK RESIDENT flat at 86.6 GiB. The move itself
works: host RSS drops 86 to 14 GiB as the mmap pages are reclaimed.

The regression matches the roofline. Unlike the dense Q8_0 tower (63% of DRAM
peak, bandwidth-bound), the grouped-MoE `QuantDotGemmGrouped<IQ2_XXS>`/`<Q2_K>`
kernels run at only ~19-24% of peak (dequant/latency-bound), so weight residency,
a bandwidth lever, cannot help them. It also adds a large one-time graph-capture
cost, and pinning the 70 GiB as `cudaMalloc` (vs evictable mmap file cache) cuts
the unified-pool reclaimable headroom from ~103 to ~30 GiB avail. The lever stays
in the tree, default-OFF, for reproducibility; detail in the benchmark record.
