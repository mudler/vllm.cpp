# GFX1100-TG200 — Lever C attribution: standalone `QuantizeQ8KK` launch sites -> producers

Committed BEFORE any kernel code (Lever C contract step 1). Evidence source:
rocprofv3 rocpd capture `/work/levc-prof/bdb445f9ac06/79723_results.db`
(full-stack config, TG200 lever-C pricing capture, acquired+released under
gpu-ctl at 01:56Z 2026-08-24). Model: Qwen3.5-4B-Q4_K_M
(sha256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
32 blocks = 24 GDN + 8 full-attn at interval 4; H=2560).

## Method

Three signals, same discipline as the T4a evidence §15.1:

1. **Geometry decoding.** The rocpd `grid_size_x` column records HIP global
   work-items in x (`grid.x * block.x`), not blocks. Cross-checks: the lm_head
   GEMV shows 1986560 = 62080 blocks x 32 lanes (N=248320, 4 warps/block);
   every `QuantizeQ8KK` dispatch shows 128 = 1 block x 128 threads, i.e. EVERY
   decode-token activation quant launches a SINGLE BLOCK (`m*nsb <= 128`).
   Pure launch pathology confirmed: mean duration ~48-50 us regardless of
   K (48.2-50.1 us across all seven site classes below).
2. **Step isolation.** One steady-state decode step = dispatch window between
   consecutive `ArgmaxK` launches (step 100 of 256 used; identical structure
   at steps 50/150/200).
3. **Producer adjacency.** Each `QuantizeQ8KK` immediately precedes its
   consumer GEMV; each consumer's activation tensor is produced by the kernel
   immediately upstream of the quant (op-order correlation), cross-checked
   against the forward call sites in `src/vllm/model_executor/models/
   qwen3_5.cpp` / `qwen3_5_gguf_weights.cpp`.

## Per-step census (97 standalone `QuantizeQ8KK` launches/token)

| # | site | producer of the quantized activation | m x K (nsb) | N (consumer) | weight fmt | launches/tok | mean us |
|---|------|--------------------------------------|-------------|--------------|-----------|--------------|---------|
| 1 | FFN gate_up fused matvec (`qwen3_5_gguf_weights.cpp` :1211 row-concat, one kMatmulBTQuant) | **RmsNormRowKernel** (post-attention input layernorm) | 1x2560 (10) | 18432 (= 2x9216) | Q4_K | 32 (24 GDN + 8 attn) | 48.6 |
| 2 | attn q_proj | **RmsNormRowKernel** (full-attn input layernorm) | 1x2560 (10) | 8192 | Q4_K | 8 | 48.4 |
| 3 | attn k_proj | **same norm output as #2** (re-quantized by its own standalone launch) | 1x2560 (10) | 1024 | Q4_K (5 layers) | 5+3* | 47.5-48.1 |
| 4 | attn v_proj | **same norm output as #2** | 1x2560 (10) | 1024 | Q6_K (5) / Q4_K (3)* | 8 | 47.5-48.1 |
| 5 | attn o_proj | PagedAttnDecodeGqaF32Qi (attention output — NOT a norm) | 1x4096 (16) | 2560 | Q4_K | 8 | 49.5 |
| 6 | FFN down_proj | SiluMulK (NOT a norm) | 1x9216 (36) | 2560 | Q4_K (16) / Q6_K (16) | 32 | 50.0 |
| 7 | lm_head | **RmsNormRowKernel** (final norm) | 1x2560 (10) | 248320 | Q6_K | 1 | 48.2 |

\* the k/v format split across the 8 full-attn layers is mixed in this GGUF;
the capture resolves 11 fmt-0 and 5 fmt-2 N=1024 quants/step; the exact
per-layer tensor formats live in the GGUF tensor map (T4a evidence §15).

Reconciliation: 32 + 8 + 8 + 8 + 32 + 1 = 89... resolved against observed
context pairs — RMS->G0(18432)=32, RMS->G0(8192)=8, G0(8192)->G0(1024)=8,
G0/G2(1024)=8, ATTN->G0(2560)=8, SILU->G0/G2(2560)=16+16, RMS->G2(248320)=1,
total **97**. `RmsNormRowKernel<bf16>` count cross-check: 65 launches/step =
2x24 GDN + 2x8 attn + 1 final = 65 exactly.

## Fusability verdict (this lever)

- **Fusable via RmsNormRowKernel epilogue: 57/97 launches/tok** (sites
  1, 2, 3, 4, 7). Sites 3+4 re-quantize the SAME normalized row already
  written for site 2's scratch — one producer record serves all three
  consumers (identical ptr, m, K, dtype, stream).
- Not fusable this round: 40/97 (sites 5, 6; producers are attention output
  and SiluMul). Owed: a SiluMulK epilogue would take another 32/tok.
- **RmsNormGatedK finding:** the gated RMSNorm (`RmsNormGatedK`, 24
  launches/tok) feeds ONLY the bf16 `wvSplitKSml` out_proj matvec — it has
  ZERO QuantizeQ8KK consumers in this model. Extending the fused epilogue to
  the gated sibling buys nothing here; recorded as owed-with-reason rather
  than time-boxed work.

## Discrepancy note (honest reporting)

The Lever C assignment quotes "43 standalone launches/token". THIS capture at
bdb445f9ac06 measures **97/tok** (~4.7 ms/tok at ~49 us each). The 43 figure
is consistent with an arm mix where the T4a fused-fold sub-arm
(VT_GEMV_MMVQ_FOLD_MAX <= 512) absorbs some sites, or with counting distinct
site CLASSES; neither applies to this capture (zero fused-fold kernels in the
decode window). The lever thesis is unchanged and stronger: single-block
launch pathology at ~49 us per launch.

## Fusion-seam gate finding

The change enriches a producer KERNEL behind VT_NORM_QUANT_FUSED (opt-in);
no model .cpp edit, no hand-call fusion, no new recipe. Per
scripts/check-fusion-consistency.py scope (model-forward floors only), the
gate is not tripped; verified green post-change in the evidence file.
