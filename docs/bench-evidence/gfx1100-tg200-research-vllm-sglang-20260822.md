# GFX1100-TG200 — research notes: vLLM/SGLang mechanisms vs our decode path

Date: 2026-08-22. Sources: vLLM (subagent, web) + SGLang (local shallow clone
at `/home/ghazni/projects/vllm.cpp/sglang-src`, read directly). Purpose: rank
portable quick wins for the TG200 campaign.

## Our measured waste (T1b/T2a captures)

| Item | ms/token | Note |
|---|---|---|
| QuantizeQ8KK activation quant | ~3.4 | 129 launches/token, grid=128 (~16K sb each) where decode m=1 needs grid=1 |
| host dispatch gap | 2.08 | 37 kernel+grid combos per step, GPU idle between |
| PagedAttnOnline bf16 | 1.07 | grid=1, block-wide sync per context token, 8 full-attn layers |
| hipBLASLt Cijk | 0.70 | MT32x32x32 tile at m=1 |
| GdnScan | 0.51 | |

## What the reference engines actually do

### 1. SGLang GGUF path: MMVQ — dequant-in-GEMM, ONE tiny quant per GEMM
(`python/sglang/srt/layers/quantization/gguf.py::fused_mul_mat_gguf`,
kernels `python/sglang/kernels/aot/csrc/quantization/gguf/mmvq.cuh`,
`gguf_kernel.cu`)

- For batch <= mmvq_safe (2-8 rows), SGLang calls `ggml_mul_mat_vec_a8`:
  the ACTIVATION is quantized once to q8_1 by a single small kernel
  (`quantize_row_q8_1_cuda`: one warp per 512-element padded row, wave
  reduce for amax/sum), then `mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
  VDR, vec_dot>` runs one WARP PER OUTPUT ROW of W with the q4_K blocks
  DEQUANTIZED IN REGISTERS via vec_dot_q4_K_q8_1.
- Grid shape: `(ceil(nrows/GGML_CUDA_MMV_Y), nvecs)` with block
  (WARP_SIZE, MMV_Y). At m=1 that is nvecs=1 launch with a handful of
  blocks — no 16K-block quant storm, and NO Q8_K scratch round-trip.
- K-quants q4_K/q5_K/q6_K are first-class (cases 12/13/14 in the
  dispatcher): exactly our formats.

=> The direct port for our engine: replace the QuantizeQ8KK->KQuantGemmK
pair at decode shapes with an MMVQ-style kernel: quantize h [1,K] to
q8_1 (one small launch, or fuse into the previous op), then one
warp-per-output-row kernel over the raw GGUF blocks already resident on
device. This eliminates BOTH the 3.4 ms/token quant storm AND most of
the scratch traffic, while keeping integer-core parity (vec_dot uses the
same dp4a integer dot; only the scale/min handling follows ggml's q8_1
convention, which changes reduction order -> needs near-tie adjudication,
not bit-exactness).

### 2. vLLM W4A16: activations stay bf16 entirely
(gptq_marlin / gptq_triton / awq_triton)

Marlin dequantizes weight tiles inside the GEMM registers; the
activation is never quantized. Same destination as (1) reached from the
other side. Also: gate+up are packed into ONE MergedColumnParallelLinear
GEMM (vllm/model_executor/layers/linear.py), so a dense MLP is
2 GEMMs + 1 activation instead of 3 GEMMs + 2 elementwise ops.

=> Quick win independent of (1): our ffn_gate and ffn_up share the same
input activation; merging them into one keep-quant GEMM halves the
launches AND the quant work for the MLP even before MMVQ lands. The
shared seam for this is `layers::MlpGateUpMethodBase` /
`vt::FusedChain`.

### 3. Graph capture covers the whole step
(vllm/compilation/cuda_graph.py, docs/design/cuda_graphs.md;
sglang decode_cuda_graph_runner.py "full" backend default)

Both engines capture the ENTIRE uniform-decode forward as one graph
(vLLM FULL_AND_PIECEWISE falls back to PIECEWISE only when attention
cannot be captured). One replay launch replaces every per-kernel
dispatch; only sampler/copy-back stays eager in the worst case.

=> Our tree already has the seam: ROCm W1 landed hipGraph capture +
BreakableGraph (rocm_backend.hip; ENG-CUDAGRAPH-BREAK/DEDUP own it),
and platforms/rocm.cpp notes support_static_graph_mode stays false
pending W2. Flipping decode-graph capture ON for this model is the T2b
stage and attacks the whole 2.08 ms gap at once. The Qwen3_5 decode
graph driver already exists for CUDA (qwen3_5.cpp SizeSlot machinery);
the ROCm side needs the graph-enabled flag path exercised on gfx1100.

### 4. Overlap scheduler hides residual host time
(sglang/srt/managers/scheduler.py::event_loop_overlap)

SGLang's overlap loop launches batch N's forward, then processes batch
N-1's results and samples while N is still executing — CPU scheduling
never serializes against GPU compute. Our engine synchronizes per step;
a single-stage overlap (sample/schedule next token while current step
drains) would hide most of whatever host gap remains after graphs.

### 5. RDNA3 specifics

No first-party gfx1100 tuning exists in either engine (AMD CI targets
CDNA; Triton config tables have no gfx1100 entries) — autotune locally.
Notes: prefer wave32 for latency-bound small-N GEMMs but benchmark both
for the dequant-heavy inner loop; gfx1100 LDS is 64KB/workgroup (cap
BLOCK_K when porting marlin-style kernels); no MFMA (WMMA only);
hipBLASLt Cijk tiles are tuned for large batch — at m=1 a custom
N-major skinny GEMM usually beats them.

## Ranked quick wins

1. **MMVQ port** (SGLang mmvq.cuh -> HIP): kills the 3.4 ms/tok quant
   storm + reduces scratch traffic. Biggest single win, self-contained
   in rocm_grouped_gemm.hip. Needs near-tie adjudication (q8_1 vs Q8_K
   convention).
2. **Decode HIP-graph capture** (existing seam, flip on for this model):
   kills up to 2.08 ms/tok of dispatch gap. Engine-level, no numerics
   change.
3. **gate_up merged keep-quant GEMM** (vLLM merged-linear pattern):
   halves MLP launches/quant sites. Rides MlpGateUpMethodBase seam.
4. **PagedAttnOnline -> DecodeGqa coverage** (already partly landed):
   ~0.9 ms/tok remaining.
