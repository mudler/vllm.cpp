# Structural prefill-gap scan (2026-07-08) — campaign map

Dynamic 9-agent scan (6 area×lens + adversarial + completeness + synthesis), local
source only (vLLM pin e24d1b24 vs ours). Mandate: hardest STRUCTURAL levers worth
long work; EXCLUDE micro-opts. Full run: workflow wf_11279226-a14.

## The load-bearing arithmetic
fp4 GEMM (42%) at parity; 35B MoE GEMMs are a bit-exact Marlin port; GEMMs ~62% of
prefill → a 0.82× prefill FORCES the eager non-GEMM region to cost ~12–18% more than
vLLM's fused region. Two structural families carry it: (1) the un-fused eager glue
chain, (2) the one GEMM bucket that never got the fp4 treatment — fp8 cuBLASLt.

## Ranked structural levers (impacts OVERLAP — do not sum; all host-side-idle)
| # | Lever | Impact | Effort | Ratio | Portable |
|---|-------|--------|--------|-------|----------|
| 1 | **fp8 cuBLASLt plan-cache + descriptor re-steer** | ~4% | M | HIGH | yes |
| 2 | **GDN WY blocked triangular inverse** | ~2% | L | HIGH | yes |
| 3 ⭐ | **Host-exec glue-chain fusion** (fused_post_conv_prep + add+RMSNorm→quant) | ~10–12% | XL | MED (max abs) | yes* |
| 4 | 35B MoE fused w13 GEMM1 (one grouped GEMM [P,2I] + one silu_mul) | ~2% | M | MED | yes |
| 5 | Merged projection GEMMs (QKV/gate_up/in_proj + scale reconciliation) | ~4% | XL | LOW | yes |

\* #3 portable except the whole-graph buffer-planning slice (Inductor-specific, out of scope).
HARDEST-WORTH-LONG-WORK: #3 and #5. Single recommended next big lever: **#3**.

## Evidence + first attack step
- **#1** ours `src/vt/cuda/cuda_matmul.cu:247` (heuristic re-run/call) + `:215–244`
  (Desc/3×Layout/Preference create+destroy/call) + `:221` TRANSA=T / `:259` weight-as-A.
  vLLM `.../kernels/linear/scaled_mm/pytorch.py:87` (single `_scaled_mm`, plan reused
  in-graph) + `qwen3_next.py:534` `@support_torch_compile`. FIX: cache `(desc,layouts,
  algo)` keyed by shape in `MatmulFp8CublasLt`; re-anchor A=activation + bf16 D to steer
  onto vLLM's TST/TNNN nvjet variant; A/B the two tile variants at prefill shapes.
- **#2** vLLM `fla/ops/solve_tril.py:238` (merge), diag inverses `:302–321`, tensor-core
  Schur merge `:356–390`. Ours `cuda_gdn.cu:1728–1740` (2016-deep serial column-solve @
  64/256 threads), launch `:1971`. FIX: swap ONLY the inverse phase for four 16×16 diag
  forward-sub inverses + six tensor-core Schur merges (21,31,32,41,42,43); reuse vt:: WMMA.
- **#3** vLLM `.../mamba/gdn/qwen_gdn_linear_attn.py:1428` (`fused_post_conv_prep` =
  L2norm+g+beta in one) + `qwen3_next.py:534` (whole-model Inductor). Ours
  `src/vllm/model_executor/models/qwen3_5.cpp:1416–1450` (GdnPostConv/L2Norm/GBeta split)
  + `:1602–1660` (FullAttn dq3/dk3/gatef/dattn/gated each a DBuf round-trip) + `:1753–1835`,
  `:2440–2458`. FIX: build the GDN `fused_post_conv_prep` equiv in the vt:: fused-recipe/
  TDR skeleton — collapse conv→L2norm→g→beta + per-layer add+RMSNorm→(quant) into single
  passes, residual on-chip. PREREQUISITE (cheap): one nsys host-trace of a prefill step →
  GPU-idle-between-launches on GB10. If idle >5% → XL fusion justified; else shrinks to
  pure-HBM core and #1 is the headline.
- **#4** ours `qwen3_5.cpp:2191–2202` (two grouped GEMMs + 2 memsets + MoeSiluMul), repack
  `:2059–2064,2094–2109`; vLLM `marlin_moe.py:116–173`. FIX: concat gate+up along N at load
  → one grouped GEMM size_n=2I → MoeSiluMul reads `[:I]/[I:]`. 35B-only.
- **#5** ours `qwen3_5.cpp:1579` (q/k/v), `:2376` (gate/up), `:1167`+`:1353` (GDN in_proj);
  vLLM `qwen3_next.py:260,542–543,701–703` + scale reconciliation `modelopt.py:518–527`.
  FIX: load-time weight concat + per-projection fp4/fp8 block-scale requant to one alpha +
  output slicing. XL (single alpha must be valid).

## EXCLUDED — do not spend sustained effort
- Refuted/negative: W4A4 MoE fp4-cutlass (35B is W4A16 → Marlin, vLLM too); fp4 cutlass
  GEMM (verbatim vLLM configs, at parity); **chunk_o + recompute_w_u (we're AHEAD via
  fusion + lower-precision apply)**; fused rmsnorm→nvfp4 as a *gap* (vLLM ALSO runs 2
  kernels → single-pass is beat-vLLM, can't close a deficit); piecewise-cudagraph as a
  standalone prefill lever (vLLM's dispatcher returns NONE >512 tokens → its large prefill
  is ALSO eager; launch count only hurts because unfused → folds into #3).
- Micro-opts (mandate excludes): 128-bit RMSNorm loads; swizzle-in-quant; bf16 attn-query/
  GDN-conv dtype; async scheduling (decode); KV vectorize/fp8-write (decode); conv1d tiled
  (just flip `VT_CONV_TILED` after a numerics A/B); MoE router top-k/moe_align (<1%); GDN
  f32 state gather/scatter + host memset (~1.5%, fold into #3).

## Campaign (revised — supersedes the earlier chunk_o/recompute_w_u sequence)
1. delta_h faithful register-tiled port — finish + validate vt::tile component (in flight).
2. **#1 fp8 cuBLASLt plan-cache** — immediate structural down-payment (M, ~4%, high ratio).
3. **nsys host-trace of a prefill step** — size #3 (GPU-idle-between-launches on GB10).
4. **#3 glue-chain fusion** (XL) via fused-recipe skeleton — the big lever, gated on (3).
5. **#2 GDN WY blocked inverse** (L, ~2%) — the algorithm we lack.
6. Reassess with a fresh vLLM-vs-ours profile.

## Implementation refinements (grounded 2026-07-08, post-scan)
- **#1 fp8 plan-cache:** `MatmulFp8CublasLtKernelCuda` (cuda_matmul.cu:204-263) recreates
  desc+3 layouts+preference AND re-runs `cublasLtMatmulAlgoGetHeuristic` (:247) EVERY call.
  FIX: per-device `unordered_map<key,Plan>` keyed by `(m,n,k,out_type)`, storing PERSISTENT
  desc+la+lb+lc+algo (not the RAII Guards — those destroy on scope). On hit skip create+
  heuristic → straight to `cublasLtMatmul`. `valid=false` entry caches the "no fp8 heuristic
  → cutlass fallback" decision too. Leak plans at exit (like ctx.handle). Isolated file.
- **#3 real surface is NARROWER than the scan headline:** our GDN post-conv prep is ALREADY
  fused (`vt::GdnPostConv` under `GlueFuseEnabled`, qwen3_5.cpp:1425 — mirrors
  fla/ops/fused_gdn_prefill_post_conv.py:152). Remaining unfused surface = (a) per-layer
  residual-ADD + RMSNorm (+quant terminal) chains feeding each GEMM — vLLM's Inductor fuses
  add+RMSNorm into one triton kernel, we run add/rmsnorm/quant as separate HBM round-trips;
  (b) FullAttnBlock glue (dq3/dk3/gatef/dattn/gated, qwen3_5.cpp:1602-1660). NOTE (scan [3]):
  rmsnorm→nvfp4 fusion alone is NOT a deficit-closer (vLLM also runs 2 kernels there) — the
  win is the ADD+RMSNorm fuse + residual-on-chip + launch-count reduction across the chain.
  BUILD ON: the TDR fused-recipe skeleton (include/vt/fused_recipe.h — kAdd/kRmsNorm/kMul +
  roles) already exists; #3 Phase-1 = the Tier-1 interpreter kernel walking the add+RMSNorm
  chain in one pass, residual on-chip. GATE on the nsys host-trace (launch-idle) first.
