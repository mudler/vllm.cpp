# ROCm keep-quant expert GEMM — review rework (PR #523)

## What this fixes

The review sweep (localai-bot, 2026-08-13) found the original #523 shape
registered `kMatmulBTQuant` with a loader that flips keep-quant on a BOOLEAN
(`GgufQuantComputeAvailable()` = `OpRegistered(...)`). The original ROCm
provider implemented four formats while the loader admitted a wider set. On a
discrete card with no CPU fallback tier, unsupported blocks could stay
quantized and throw at first forward. The same boolean flipped `keep_f16` on,
and the ROCm `MatmulBT` refuses f16. This caused a second regression.

## The rework

1. **Per-dtype capability in the loader** (`gguf_keep_quant.cpp`):
   `KeepQuantDType` and the keep-f16 default now consult the running device's
   actual support. ROCm keep-quant now supports Q8_0, Q2_K, Q3_K, Q4_K,
   Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ2_S, IQ1_S, and IQ1_XXXS on both
   `kMatmulBTQuant` and `kMatmulBTQuantGrouped`. ROCm keep-f16 is OFF
   (`MatmulBTKernelRocm` accepts bf16/f32 only). Unsupported formats keep the
   pre-existing `expand_bf16` residency — no load fails, no forward throws, and
   `VT_GGUF_KEEP_QUANT=1` on a Q4_0 model is a no-op rather than a regression.
   CUDA/CPU behavior is byte-identical (their sets already cover the CPU list).
2. **Capture-safe scratch**: the per-call `hipMalloc`/`hipFree`/
   `hipStreamSynchronize` on the activation-quant scratch (illegal under
   hipGraph stream capture — blocks #473/#332) becomes a grow-only pool keyed
   by queue identity. `hipMallocAsync` allocates outside capture. A warm block
   remains stable across capture replay and eager reuse. A cold request or
   growth during capture refuses with a pre-warm message. Queue ownership also
   prevents a recycled native stream from inheriting another queue's pointer.
3. **The refusal messages** distinguish the internal GDN formats from the
   wrapper-owned seven-format provider. They name Q4_0, Q5_0, IQ2_XS,
   IQ4_NL, IQ3_S, IQ4_XS, and MXFP4 as the remaining unsupported formats.
4. **Teeth**: the non-grouped `kMatmulBTQuant` gains its own cross-device case
   (it carried the headline mechanism and had no test), and both new cases
   `REQUIRE(OpAvailable(...))` instead of skipping silently when registration
   is dropped. The grouped case keeps its NMSE<=5e-4 vs CPU keep-quant oracle
   bar.
5. `Dp4a` keeps the portable four-MAC body if `__dp4a` is absent on the
   gfx1100 toolchain (verified at build time); if `__dp4a` compiles, use it.

## Gates

- Focused: `test_backend_cross_device` (grouped + non-grouped keep-quant
  cases, REQUIRE-proven registration), red-first by stash-revert.
- Regression: the 0.8B + 0.6B M4 gates; Qwen3.6-35B-A3B Q4_K_M e2e on one
  gfx1100 card (`--max-num-seqs 1`); a Q4_0 GGUF load on ROCm proving no
  regression (expands, generates, no throw).
- Full HIP ctest zero-delta vs base.

## Boundaries

- No change to the ported dot-product cores (review verified them against the
  donor, DotQ6K byte-for-byte).
- Q4_0, Q5_0, IQ2_XS, IQ4_NL, IQ3_S, IQ4_XS, and MXFP4 ROCm kernels remain
  owed. The loader keeps these formats on the named expand-or-refuse path.
