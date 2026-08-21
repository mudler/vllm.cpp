# ROCm keep-quant expert GEMM — review rework (PR #523)

## What this fixes

The review sweep (localai-bot, 2026-08-13) found the original #523 shape
registered `kMatmulBTQuant` with a loader that flips keep-quant on a BOOLEAN
(`GgufQuantComputeAvailable()` = `OpRegistered(...)`), while the ROCm kernel
implements 4 of the 12 formats the loader admits (Q4_0, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ2_S, MXFP4). On a discrete card with no CPU
fallback tier, a Q4_0/Q2_K/IQ2 model that loaded and generated fine before
would keep blocks quantized and throw at first forward. Same boolean flipped
`keep_f16` on, and the ROCm `MatmulBT` refuses f16 — a second regression of a
working path.

## The rework

1. **Per-dtype capability in the loader** (`gguf_keep_quant.cpp`):
   `KeepQuantDType` and the keep-f16 default now consult the running device's
   actual support. ROCm keep-quant supports {Q8_0, Q4_K, Q5_K, Q6_K}
   (kMatmulBTQuant + kMatmulBTQuantGrouped both); ROCm keep-f16 is OFF
   (`MatmulBTKernelRocm` accepts bf16/f32 only). Unsupported formats keep the
   pre-existing `expand_bf16` residency — no load fails, no forward throws, and
   `VT_GGUF_KEEP_QUANT=1` on a Q4_0 model is a no-op rather than a regression.
   CUDA/CPU behavior is byte-identical (their sets already cover the CPU list).
2. **Capture-safe scratch**: the per-call `hipMalloc`/`hipFree`/
   `hipStreamSynchronize` on the activation-quant scratch (illegal under
   hipGraph stream capture — blocks #473/#332) becomes a grow-only per-stream
   pool via `hipMallocAsync`, mirroring the donor's `EnsureScratch` +
   `RetireGraphScratch` (never-freed, because a captured graph may have baked
   the pointer). Also fixes the `qact` leak when `Check()` threw between
   malloc and free.
3. **The refusal messages** name the actually-unported formats (Q4_0, Q2_K,
   Q3_K, IQ2_XXS, IQ3_XXS, IQ2_S, MXFP4) instead of double-listing Q5_K — the
   message is now unreachable in practice (the loader pre-filters) but stays
   correct as the last line of defense.
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
- Q2_K/Q3_K/IQ2/IQ3/MXFP4 ROCm kernels remain owed and are recorded as such
  here and in the refusal messages.
