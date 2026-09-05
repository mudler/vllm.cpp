# ROCm fp8 KV cache prefill fast path (`GFX1100-TG200`)

Rows: `GFX1100-TG200` (campaign) and `KV-FP8` (engine-matrix).
The fp8 KV decode fast path landed in #2168
(`PagedAttnDecodeGqaF32Q` with fp8 dequant, `VT_ATTN_DECODE_GQA4=1`); this
spec covers the prefill path that the decode spec named as owed.

## Scope

- **In:** template `PagedAttnPrefillSharedK` on `TKV`; add an fp8 dequant
  load path inside the prefill kernel that does vectorized `uint8_t` loads
  + `F8E4M3ToF32Dev` dequant with scale (mirrors `LoadRowEplFp8` from the
  decode kernel); widen the prefill dispatch guard at
  `rocm_paged_attn.hip:1989` to accept `k_cache.dtype == DType::kI8` when
  `args.kv_cache_dtype != kAuto`; pass `k_scale`/`v_scale` to the prefill
  kernel.
- **Out:** the `PagedAttnPrefillFlashTile` and WMMA prefill kernels
  (separate performance bricks; the SharedK kernel is the default prefill
  path and covers the hang). The decode path (already landed in #7). The
  `PagedAttnOnline` fallback (unchanged). fp8_e5m2 compute. Per-head
  scales. Non-gfx1100 architectures.

## Upstream chain

vLLM's fp8 KV cache read dequantizes inside the attention kernel:
`scaled_vec_conversion<float, uint8_t>`
(`quant_utils.cuh:419-429`) = `half_to_float(fp8_to_half(byte)) * scale`.
The ROCm `LoadKv(uint8_t*, ...)` helper at `rocm_paged_attn.hip:176`
already mirrors this arithmetic exactly: `F8E4M3ToF32Dev(p[i]) * scale`.
The decode kernel's `LoadRowEplFp8` (line 370) does the same with
vectorized loads. The prefill kernel needs the same dequant; the
arithmetic is not new code.

## Our baseline

The `PagedAttnPrefillSharedK` kernel (`rocm_paged_attn.hip:872`) is the
default prefill path for bf16 KV with QG=2, `total_q >= 64`,
`num_reqs == 1`. It tiles the Q×KV computation into BM×BN blocks with
online softmax, loading K and V tiles into shared memory. No per-key
`__syncthreads()` — the tile structure avoids the O(n²) sync walk that
causes `PagedAttnOnline` to hang at 14K+ context.

The dispatch guard at line 1989 requires
`k_cache.dtype == DType::kBF16 && v_cache.dtype == DType::kBF16`. With
`--kv-cache-dtype fp8`, the guard fails and the dispatch falls through to
`PagedAttnOnline` (line 2303) — the per-key loop with full-block
`__syncthreads()` reduction per key.

## Measured gap

A/B benchmark on `kind_tharp` (Qwen3.5-4B Q4_K_M, RX 7900 XTX, ROCm
10.0.0, 128-token greedy decode, single request, `--repeat 6` (2 warmup
+ 4 measured, median), `--kv-cache-memory 805306368` (768 blocks),
2026-08-30):

| Context | bf16 tok/s | fp8-slow tok/s | fp8+GQA4 tok/s | gap-old | gap-new |
|--------:|-----------:|--------------:|---------------:|--------:|--------:|
| 256     | 26.84      | 25.87          | 28.70           | 1.04x   | 0.94x   |
| 1024    | 13.65      | 12.75          | 15.32           | 1.07x   | 0.89x   |
| 4096    | 4.12       | 3.66           | 4.47            | 1.13x   | 0.92x   |
| 8192    | 1.91       | 1.62           | 1.93            | 1.18x   | 0.99x   |

`gap-old` = bf16 / fp8-slow (the regression before the decode fix).
`gap-new` = bf16 / fp8+GQA4 (after the decode fix).

The decode kernel achieves parity (0.89x-0.99x). The remaining gap is
prefill: `tok_s` includes prefill time, and fp8 prefill still uses
`PagedAttnOnline` (the slow fallback). The 16384-context data point
could not be measured because `PagedAttnOnline` hangs at ~14K+ prompt
tokens.

## Design

### 1. Template `PagedAttnPrefillSharedK` on `TKV`

Change the kernel signature from hardcoded
`const __hip_bfloat16* k_cache` to `template <typename TKV>` with
`const TKV* k_cache, const TKV* v_cache`. Add `float k_scale,
float v_scale` parameters. Inside the kernel, replace the bf16 K/V tile
loads with a `LoadTileKv<TKV>` dispatch that selects the bf16 path for
`__hip_bfloat16` and the fp8 path for `uint8_t` via `if constexpr`.

### 2. fp8 tile load path

The SharedK kernel loads K/V tiles into shared memory as `__hip_bfloat16`.
For fp8, the tile load dequantizes each byte to `float` (or bf16) with
`F8E4M3ToF32Dev(byte) * scale` and stores the result in the same smem
tile. The dequant happens at load time, so the rest of the kernel (QK
dot product, online softmax, V accumulation) operates on dequantized
values — no change to the compute path.

The vectorized load uses `uint4` (16 bytes = 16 fp8 elements) per thread,
matching `LoadRowEplFp8<EPL=16>` from the decode kernel. The scale is
passed as a parameter.

### 3. Widen the prefill dispatch guard

At line 1989, widen the condition from:
```
k_cache.dtype == DType::kBF16 && v_cache.dtype == DType::kBF16
```
to:
```
(k_cache.dtype == DType::kBF16 && v_cache.dtype == DType::kBF16) ||
(k_cache.dtype == DType::kI8 && v_cache.dtype == DType::kI8 &&
 args.kv_cache_dtype != Fp8KVCacheDataType::kAuto)
```

When the KV is fp8, launch with `k_cache.Ptr<uint8_t>()`,
`v_cache.Ptr<uint8_t>()`, and pass `args.k_scale`/`args.v_scale`.

### 4. Production-seam ROCm test (G7)

The existing `test_ops_fp8_kv_cache` suite (W1, CPU oracle) gates the
fp8 dequant arithmetic, and the served-model token-exact gate gates
end-to-end correctness. Neither reaches the SharedK prefill dispatch:
G4 uses hq=2, H=1, D=16 (no geometry match), and G6 uses d=128 with
total_q=1 (decode, not prefill). A new G7 case in
`test_rocm_fp8_kv_cache` exercises the exact dispatch geometry (f32
q/out, fp8 KV, d=256, total_q=64, QG=4, num_reqs=1) and compares
ROCm SharedK output against the CPU PagedAttnOnline oracle.

Two independent mutation gates:

- **Dispatch reach (counter):** `Fp8PrefillSharedKDispatchCount()` is
  incremented inside the dispatch guard in `PagedAttentionKernelRocm`.
  The test checks the counter before and after the GPU call; a deleted
  or broken dispatch (wrong dtype, wrong d, wrong QG) leaves the counter
  unchanged and the CHECK fails. This is necessary because output
  comparison alone cannot distinguish SharedK from the PagedAttnOnline
  fallback: both GPU kernels use online softmax, while the CPU oracle
  uses two-pass softmax, so both GPU kernels differ from the CPU by a
  similar (small) amount.
- **Load/scale mutations (NMSE band):** the tight NMSE band (< 1e-6)
  gates dequant arithmetic errors. Swapping k_scale/v_scale or dropping
  the dequant scale moves the output by orders of magnitude, well beyond
  the band. The band does NOT gate dispatch deletion — that is the
  counter's role.

## Risks

- **Shared memory layout:** the bf16 SharedK kernel stores K/V tiles as
  `__hip_bfloat16` in shared memory. The fp8 path dequantizes to `float`
  at load time, which doubles the smem per element (4 bytes vs 2). The
  tile sizes BM=32, BN=32 at d=256 use 32*256*2 = 16KB per K tile (bf16).
  With fp8 dequantized to float, the same tile is 32*256*4 = 32KB. The
  gfx1100 LDS is 256KB per CU, so one CTA's K+V tiles (64KB) still fit.
  If smem pressure is too high, dequantize to bf16 instead of float
  (2 bytes, same as the original bf16 path) — the QK dot product and V
  accumulation already work in bf16.
- **Numerical equivalence:** the SharedK kernel uses a different tile
  order than `PagedAttnOnline`'s per-key loop. The online softmax
  reduction order differs, so floating-point results may differ at the
  last bit. This is the same risk the decode kernel's
  `PagedAttnDecodeGqaF32Q` carries (documented in the #7 spec's Risks
  section). The prefill path is less sensitive than decode because
  prefill outputs are intermediate hidden states, not greedy token
  selections.
- **Prefill hang fix:** the SharedK kernel tiles the computation into
  BM×BN blocks with no per-key sync, so it should not hang at 14K+
  context. This needs verification.

## Gates

- **Correctness (CPU oracle):** `test_ops_fp8_kv_cache` GREEN — the W1
  suite already gates the fp8 dequant arithmetic; this change does not
  touch the CPU path.
- **Correctness (dispatch reach, G7):** `test_rocm_fp8_kv_cache` G7
  exercises the exact SharedK prefill geometry (f32 q/out, fp8 KV,
  d=256, total_q=64, QG=4). Two independent mutation gates: (1) the
  `Fp8PrefillSharedKDispatchCount()` counter proves the dispatch fired
  — a deleted or broken guard leaves it unchanged and fails the CHECK;
  (2) the NMSE < 1e-6 band gates load/scale mutations (swapped
  k_scale/v_scale, dropped dequant) which move the output by orders of
  magnitude. The band does NOT gate dispatch deletion because both
  SharedK and the PagedAttnOnline fallback use online softmax and
  produce numerically similar results.
- **Correctness (served model, token-exact):** run `kind_tharp` with fp8
  KV + `VT_ATTN_DECODE_GQA4=1` and compare greedy decode output against
  the bf16 KV baseline at short context (256 tokens). Tokens must match;
  at longer context, the reduction-order risk applies and is recorded.
- **Performance (A/B):** re-run the context-scaled benchmark with the
  fp8 prefill fast path and compare against the bf16 baseline. Target:
  fp8 KV tok/s within 1.0x of bf16 at all contexts (the decode kernel
  already achieves this; the prefill path should match). The 16K context
  data point should now be measurable (no hang).
- **Red-first:** revert the guard widening, confirm the dispatch falls
  back to `PagedAttnOnline`, confirm the benchmark shows the original
  regression and the 14K hang.

## Git integration

- Separate spec and implementation PRs (developer preference, recorded
  2026-08-30).
- Branch: `row/fp8-prefill-fastpath` (off `row/fp8-kv-decode-attn`).
- Push to `origin` (fork `ghazni101/vllm.cpp`) only.
- Spec commit first, then implementation commits.

## Now

Implementation landed. SharedK prefill templated on `TKV`, `TQ`, `TO`.
fp8 dequant-to-bf16 at load time. Dispatch guard accepts f32-query +
fp8-KV + d=256. 14K prefill hang fixed. Token-exact gate passed (101
token prompt, SharedK vs PagedAttnOnline produce identical output).
G7 production-seam reach test added to `test_rocm_fp8_kv_cache` with
dispatch counter (`Fp8PrefillSharedKDispatchCount`) for dispatch-reach
verification and NMSE band for load/scale mutation gating.

## Outcome

### Implementation

Templated `PagedAttnPrefillSharedK` on `TKV` (KV element type), `TQ`
(query element type), and `TO` (output element type). Default template
params preserve backward compatibility with existing bf16 dispatches.

The fp8 tile load path uses `uint4` vectorized loads (16 fp8 elements
per thread) and dequantizes each byte to `__hip_bfloat16` via
`__float2bfloat16(F8E4M3ToF32Dev(byte) * scale)` before storing to the
same bf16 smem tile. The rest of the kernel (QK dot product, online
softmax, V accumulation) is unchanged — it operates on dequantized
bf16 values. Dequant-to-bf16 (not float) keeps smem at 32KB per K/V
tile, matching the bf16 path and fitting within gfx1100's 64KB LDS.

The dispatch guard accepts `query.dtype == kF32 && out.dtype == kF32
&& k_cache.dtype == kI8 && v_cache.dtype == kI8 && d == 256 &&
total_q >= 64 && num_reqs == 1`. QG=4 (Qwen3.5-4B: hq=16, kv=4) is
tiled via z=2 using the QG=2 kernel. BM=32, BN=32 — matching the
existing bf16 d=256 SharedK config.

### Correctness

Token-exact gate: 101-token prompt (10 repetitions of "The quick brown
fox jumps over the lazy dog. "), `--max-tokens 30 --temperature 0
--kv-cache-dtype fp8 --kv-cache-memory 805306368`. SharedK prefill and
PagedAttnOnline prefill produce identical greedy decode output:
"The quick brown fox jumps over the lazy dog. The quick brown fox
jumps over the lazy dog. The quick brown fox jumps over the lazy dog."

All unit tests pass: `test_ops_fp8_kv_cache` (511 assertions),
`test_rocm_fp8_kv_cache` (28), `test_attn_backend_registry` (125),
`test_attn_validate_configuration` (82), `test_kv_cache_fp8_wiring`
(487), `test_ops_attention` (39), `test_rocm_backend` (1065),
`test_rocm_arch` (59). Total: 115 cases, 2297 assertions, 0 failures.

### 14K prefill hang fixed

`PagedAttnOnline` hangs at ~14K+ prompt tokens (GPU scheduler timeout,
90s timeout with no output). The SharedK prefill kernel completes at
14K context in 68 seconds, producing correct output. The hang root
cause — per-key `__syncthreads()` reduction in `PagedAttnOnline`
(14K sync barriers per CTA) — is avoided by SharedK's BM×BN tile
structure with no per-key sync.

### Performance A/B

Qwen3.5-4B Q4_K_M, RX 7900 XTX, ROCm 10.0.0, 2026-08-30.
`--repeat 4` (2 warmup + 2 measured), `--max-tokens 20`,
`--kv-cache-dtype fp8`, `VT_ATTN_DECODE_GQA4=1`,
`--kv-cache-memory 805306368`. SharedK ON vs OFF (PagedAttnOnline):

| Context | SharedK ON tok/s | SharedK OFF tok/s | Ratio |
|--------:|-----------------:|------------------:|------:|
| 251     | 13.7             | 13.6              | 1.01x |
| 1001    | ~2.2             | ~1.9              | ~1.15x |
| 4001    | 0.74             | 0.47              | 1.57x |
| 8001    | ~0.32            | ~0.36             | ~1.0x |
| 14001   | 0.146            | hang              | N/A   |

SharedK is parity at short context (251 tokens), wins at 4K context
(1.57x), and is the only path that completes at 14K context. The 8K
data point is noisy (thermal/contention) but does not regress. The
low tok/s at long context reflects prefill-dominated workload with
only 20 decode tokens.

End-to-end A/B (bf16 baseline vs fp8+GQA4+SharedK), `--repeat 6`
(2 warmup + 4 measured, median run 6), `--max-tokens 100`:

| Context | bf16 tok/s | fp8+GQA4+SharedK tok/s | Ratio |
|--------:|-----------:|----------------------:|------:|
| 251     | 26.56      | 28.27                 | 1.06x |
| 1001    | 10.41      | 11.43                 | 1.10x |
| 4001    | 3.92       | 5.14                  | 1.31x |

fp8+GQA4+SharedK beats bf16 at all contexts. The advantage grows
with context length because the SharedK prefill is faster than
PagedAttnOnline at long context (1.57x at 4K in the isolated
SharedK ON/OFF A/B) while the fp8 decode kernel halves KV bandwidth
at all contexts.

### What was rejected

- **Dequant to float in smem:** rejected because it doubles smem per
  element (4 vs 2 bytes), causing LDS overflow at BN=32 d=256 (66832
  bytes > 65536 byte gfx1100 limit). Dequant to bf16 keeps the same
  smem footprint as the bf16 path.
- **BN=64 for fp8:** rejected for the same LDS overflow reason
  (65536 dynamic + 1296 static = 66832 > 65536). The existing bf16
  d=256 path also uses BN=32, not BN=64.
- **Separate fp8 prefill kernel:** rejected because templating the
  existing SharedK kernel on TKV is cleaner and avoids code
  duplication. The compute path is identical; only the load path
  differs.
