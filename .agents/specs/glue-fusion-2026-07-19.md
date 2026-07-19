# Glue-fusion scoping — two grounded 27B/35B prefill levers (2026-07-19)

Scope: the two grounded prefill levers from the glue-fusion scoping (task #57).
Both are landed/dispositioned here. Owner claim `CLAIM-SIGMOID-GATE-FOLD-1`.

## Lever 1 — sigmoid-gate → o_proj fold (LANDED opt-in)

**Upstream ground.** vLLM folds the full-attention sigmoid output gate
(`out = attn * sigmoid(gate)`) into the adjacent NVFP4 activation quant that feeds
the o_proj GEMM — its Inductor `triton_poi_fused_mul_scaled_fp4_quant_sigmoid_view`.
This is the same activation→quant fusion class as vLLM's
`silu_and_mul_nvfp4_quant` (`ActivationQuantFusionPass`), which we already mirror as
`vt::SiluMulFp4Quant` (down_proj, default ON, measured +2.4%).

**Our standalone op.** `SigmoidGateBf16` (attn·sigmoid(gate) → bf16 `gated`) is
invoked at `qwen3_5.cpp` full-attn blocks (`FullAttnBlock`, `FullAttnBlockPaged`),
feeding the o_proj GEMM (`MatmulNvfp4Bf16D` / `MatmulFp8CutlassD` / `MatmulBf16D`).

**Key architectural finding — only 27B benefits.**
- **27B is true-W4A4** (`o_proj_fp4.IsTrueW4A4()` — alpha>0): the o_proj GEMM
  (`MatmulNvfp4Fp4D`) quantizes its bf16 activation to fp4 via `vt::ScaledFp4Quant`.
  This is the exact vLLM fusion site — fold `attn*sigmoid(gate)` INTO that quant.
- **35B is W4A16** (Marlin, alpha==0) or fp8 o_proj: the GEMM reads bf16
  activations directly — there is NO activation-quant kernel to fuse into. The
  standalone `SigmoidGateBf16` stays; the fusion is inert. (The task's +0.5% 35B
  estimate is not realizable via this fp4-quant fold; a 35B fold would require
  fusing into the attention epilogue or the Marlin activation load — out of scope.)

**Implementation.** NEW `vt::SigmoidGateFp4Quant` op (`kSigmoidGateFp4Quant`):
CUDA `SigmoidGateFp4QuantKernel` (a mirror of `SiluMulFp4QuantKernel` substituting
`attn*sigmoid(gate)` for `silu(gate)*up`, then the exact `ScaledFp4Quant` epilogue),
CPU composite fallback = `SigmoidGateBf16`→bf16→`ScaledFp4Quant` (the correctness
oracle). Model dispatch `SigmoidGateOProjD` fuses on the guarded 27B-W4A4-CUDA path
(gated `FuseSigmoidGateQuantEnabled()`, `VT_FUSE_SIGMOID_QUANT`), else the exact
prior `SigmoidGateBf16` + three-way GEMM. sigmoid·mul is elementwise/non-reducing
⇒ bit-identical (value rounded through bf16 before quant, same as the composite).

**Gates (DGX, `~/work/vllm.cpp-sigmoid-fold`, production flags CUTLASS sm120a +
Marlin + FA2 sm_121a + Triton AOT, clean CUDA `-Werror` 0 warnings, one flock).**
- Byte-exact op test `sigmoid_gate_fp4_quant` **14/14** (CPU f32/bf16 + CUDA f32,
  M×K ∈ {1×64, 37×128, 128×512}) — fused == `SigmoidGateBf16`+`ScaledFp4Quant`.
- Token-exact: 27B `test_qwen27_paged_engine` **235/235 both arms** (default-ON
  fused build + `VT_FUSE_SIGMOID_QUANT=0` fallback); 35B `test_qwen36_paged_engine`
  **315/315** (inert).
- In-situ 27B TTFT A/B (input-1024, output-8, greedy, 3 reps/arm, interleaved):
  c1 med OFF 419.6 / ON 419.0 ms (−0.15%), c2 792.9 / 792.7 ms (−0.03%), prefill
  tok/s ~equal — NEUTRAL within ±0.5% rep noise.

**Disposition: byte-exact + perf-NEUTRAL ⇒ landed OPT-IN** (`VT_FUSE_SIGMOID_QUANT=1`,
default OFF), mirroring the perf-neutral fp8-merged-QKV launch-fusion disposition.
The full-attn o_proj is a small slice of 27B prefill (dominated by GDN + MoE + the
QKV/gate/up GEMMs), so the one-launch + bf16-round-trip saving sits below the noise
floor. Reusable primitive for any future W4A4 gated-projection site.

## Lever 3 — eager Marlin repack (35B) — VERIFIED already-at-load-time, SKIPPED

**Premise to verify.** The first-touch Marlin repack (`BuildMoeMarlinResident`) was
42% of a COLD (no-warmup) 35B prefill trace — does it leak into the binding grid's
measured c1/c2 TTFT?

**Verified NO.** `Qwen3_5Model::PrepareMarlinResident` already builds ALL Marlin
residents (routed experts `BuildMoeMarlinResident`, shared experts, lm_head
`BuildMarlinDenseResident`) eagerly at load, called from the **`GPUModelRunner`
constructor** (`runner.cpp:302/321`) BEFORE any warmup or serving — exactly vLLM's
`process_weights_after_loading`. The forward first-touch build
(`if (!mr.ready) BuildMoeMarlinResident`, `qwen3_5.cpp` `MoeBlockFusedMarlinCuda`)
is a dead fallback in production (`mr.ready` is already true from load).
Corroborated by the `ENG-MOE-LOADSTREAM` load-phase work (per-layer build+free at
load). Empirically an un-warmed 35B c1 bench (6 reqs, input-1024) shows an elevated
first-request TTFT (mean 247.9 / median 178.1 / p99 575.5 ms) — but that ~400 ms
first-request delta is general warmup (FP4 autotune / plan-cache / CUDA-graph
capture / page-in), NOT the multi-second expert repack (a repack leak would add
seconds, and is provably at construction regardless). The binding grid warms with a
1×1024-token request before timing, excluding even that general first-request cost.
The repack is OUTSIDE the measured c1/c2 window ⇒ **no implementation needed, lever
skipped.**
