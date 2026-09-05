# ROCm gfx1100: hardware Dp4a for KQuantGemmK (v_dot4_i32_i8)

- Issue: [#2362](https://github.com/mudler/vllm.cpp/issues/2362)
- Row: `BACKEND-ROCM`
- Branch: `row/ROCM-HW-DP4A`

## Now

`DONE` — hardware Dp4a landed. The tiled kernel approach was tried first and
rejected (31% slower — L2 already provides weight reuse; see `## Outcome`).

## Scope

Replace the warp-per-output-element `KQuantGemmK` with a weight-shared tiled kernel for the **prefill regime** (m > 1) on ROCm gfx1100. Decode (m = 1) keeps the existing single-warp kernel unchanged.

### In scope

- `src/vt/rocm/rocm_grouped_gemm.hip`: new `KQuantGemmKTiled` kernel + dispatch logic in `MatmulBTQuantKernelRocm`
- `tests/vt/test_ops_quant_dot.cpp`: add prefill-shape cases (m = 32, 128, 512) that exercise the tiled path
- Q4_K first; Q6_K and Q5_K if the Q4_K result warrants it

### Out of scope

- CUDA path (`QuantDotGemmKernel` in `cuda_quant_dot.cu`) — same bottleneck but separate row
- Decode (m = 1) — already optimized by the TG200 campaign
- Q8_0 weights — already have MultiRow/Prefetch/Subwarp variants
- IQ-quant formats (IQ2_XXS, IQ3_XXS, etc.)

## Upstream anchors

### vLLM (primary oracle)

vLLM uses Marlin for quantized GEMM (`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `marlin_gemm` at `marlin.cu:545`). Marlin is a tiled kernel that reuses weight tiles in shared memory. This kernel is the same design principle applied to the k-quant block format.

### vllm.cpp CUDA reference

`src/vt/cuda/cuda_quant_dot.cu:774` — `QuantDotGemmKernel<W, OutT>`: the identical warp-per-output-element kernel. No tiled K-quant variant exists on the CUDA path either.

`src/vt/cuda/cuda_quant_dot.cu:1118` — `QuantDotGemmQ8_0MultiRowKernel<OutT, NROWS>`: the design template. It shares activation reads across NROWS output columns. The tiled K-quant kernel shares **weight** reads across MROWS activation rows — the inverse tiling, because the bottleneck is weight bandwidth (weight matrix is N×K, activation is M×K, and N >> M for prefill).

### llama.cpp

`ggml-cuda/mmvq.cu` + `vecdotq.cuh`: MMVQ (multi-matrix-vector quantized) kernels for decode. `ggml-cuda/mmvq.cu:vec_dot_q4_K_q8_1_impl_vmmq` is the per-block dot reference. llama.cpp also has `mul_mat` MMQ kernels for prefill that tile across both M and N. Our kernel tiles across M only (weight-shared), which is simpler and sufficient because N is large enough that each block handles one weight column.

### CPU oracle

`src/vt/cpu/cpu_quant_gemm.cpp:MatmulBTQuantKernel` — the bit-exactness reference. The gate is NMSE ≤ 5e-4 vs dequant-f32 (`tests/vt/test_ops_quant_dot.cpp:1000`), same as the existing kernel.

## Design

### Problem

The current `KQuantGemmK` kernel (`rocm_grouped_gemm.hip:416`):

```
warp = blockIdx.x * 4 + threadIdx.y
i = warp / n   (activation row)
j = warp % n   (weight column)
```

Each warp reads an entire weight row (nsb superblocks) from global memory. For m=228, n=4096: 228×4096 = 933,888 warps, each reading the same weight row as 228 others. Weight global memory traffic = m × n × nsb × w_block_bytes. The weight is read m times when it needs to be read once.

### Solution

A weight-shared tiled kernel where a block of MROWS warps handles MROWS activation rows × 1 weight column:

```
blockIdx.x = j              (weight column)
blockIdx.y = i0 / MROWS     (activation row group)
threadIdx.y = warp_id        (0..MROWS-1, one per activation row)
threadIdx.x = lane           (0..31)
```

Weight superblocks are loaded into shared memory by the first warp, then all MROWS warps read from shared memory:

```
__shared__ uint8_t s_w[32 * MAX_BLOCK_BYTES];

for (int64_t sb_base = 0; sb_base < nsb; sb_base += 32) {
    // First warp loads weight superblocks into shared memory
    if (threadIdx.y == 0) {
        int64_t sb = sb_base + lane;
        if (sb < nsb)
            memcpy(s_w + lane * w_block_bytes, w_row + sb * w_block_bytes, w_block_bytes);
    }
    __syncthreads();

    // All warps compute dot using shared weight
    int64_t sb = sb_base + lane;
    if (sb < nsb && i < m)
        partial += DotQ4K(s_w + (sb - sb_base) * w_block_bytes, a_row + sb);
    __syncthreads();
}
// warp reduce, write output — same as existing kernel
```

Weight global memory traffic drops from m × n × nsb × w_block_bytes to n × ceil(m/MROWS) × nsb × w_block_bytes — a factor of MROWS× reduction.

### Why this is bit-exact

The `DotQ4K`/`DotQ5K`/`DotQ6K` functions take `const BlockQ*_K*` pointers. They read the weight block's scales, qs, d, dmin fields and the activation block's qs, bsums, d field. Passing a pointer to shared memory instead of global memory changes only the load source, not the values. The integer dot-product core (`Dp4a` calls, scale multiplication, final float fold) is identical. The warp reduction is identical. The accumulation order per lane is identical (same superblock iteration: lane, lane+32, ...). Therefore the output is bit-identical to the existing kernel.

### MROWS selection

MROWS = 4 (matching the current 4 warps per block). This gives 4× weight bandwidth reduction with the same occupancy. Higher MROWS (8, 16) are possible but increase shared memory pressure and reduce the number of blocks that can be co-scheduled. MROWS = 4 is the conservative starting point; the sweep is in `## Outcome`.

### Shared memory

Per block: 32 × w_block_bytes.
- Q4_K: 32 × 144 = 4,608 bytes
- Q5_K: 32 × 176 = 5,632 bytes
- Q6_K: 32 × 210 = 6,720 bytes

gfx1100 has 64 KB LDS per CU. Even at MROWS=8, shared memory is under 7 KB — no constraint.

### Dispatch

In `MatmulBTQuantKernelRocm`, after the existing `KQuantDecodeCoopWarps` check (which returns 1 for m > 1), add:

```
if (m > 1) {
    // Prefill: use tiled kernel
    const int MROWS = 4;
    dim3 block(32, MROWS);
    dim3 grid(n, (m + MROWS - 1) / MROWS);
    KQuantGemmKTiled<OutT, Fmt><<<grid, block, 0, s>>>(...);
    return;
}
```

The existing single-warp kernel handles m == 1 (decode) and the cooperative kernel handles m == 1 Q6_K with nsb ≤ 32.

## Risks

1. **Bit-exactness**: The kernel must produce identical output to the existing kernel. The design preserves the per-lane superblock iteration order and the integer dot core, so the output should be bit-identical. The gate verifies this. If the compiler reorders shared memory loads differently from global memory loads, the float reassociation could differ — but the integer core is exact and the float scale fold is a single multiply-subtract per superblock, accumulated in the same order.

2. **Occupancy**: MROWS=4 with 32×4=128 threads per block. gfx1100 supports 40 warps per CU (2560 threads). At 4 warps per block, 10 blocks per CU — but shared memory (4.6 KB) and register pressure may limit this. The existing kernel also uses 4 warps per block, so occupancy should be similar.

3. **Edge cases**: When m is not a multiple of MROWS, the last block has idle warps. The kernel guards with `if (i < m) return` — same pattern as the existing kernel's `if (warp >= m * n) return`.

4. **ROCm shared memory memcpy**: `memcpy` in device code should compile to LDS loads. If not, use a manual loop. This is a implementation detail, not a design risk.

## Tests

1. **Existing gate**: `test_ops_quant_dot.cpp` G3 cases (NMSE ≤ 5e-4 vs dequant-f32, bit-exact run-to-run, matches per-row vec_dot). These already cover m = {1, 4, 32, 512} — the tiled path activates at m > 1.

2. **New cases**: Add m = {128, 228} at Q4_K with n = {4096, 2560} to cover the production prefill shapes. These verify the tiled kernel against the CPU oracle at the exact shapes the profile measured.

3. **Token-exact**: Run `vllm-cli` on Qwen3.5-4B Q4_K_M with a fixed prompt and seed, compare output tokens against the baseline (pre-tiled) binary.

4. **A/B throughput**: Run `vllm-bench` at PP 28..1821 on both binaries, same method as the `gfx1100-pp-ab-20260830.md` evidence.

## Gates

1. **ISA gate**: `python3 scripts/check-rocm-dp4a-intrinsic.py` — fails when the `Dp4a` function does not use `__ockl_sdot4`. The CPU-only `ctest -R quant_dot` stays green with the scalar expansion, so this source-level gate is the one that catches a regression. Mutation: `tests/scripts/test_check_rocm_dp4a_intrinsic.py` replaces the intrinsic with the scalar expansion and asserts the checker goes red.
2. **Mutation suite**: `python3 -m unittest tests.scripts.test_check_rocm_dp4a_intrinsic` — 6 cases, including a live-source mutation that replaces `__ockl_sdot4` with the scalar expansion and verifies the checker catches it.
3. `ctest -R 'quant_dot'` — all G3 cases pass (CPU-only; does not catch scalar regression, which is why gate 1 exists)
4. Token-exact vs baseline binary on Qwen3.5-4B Q4_K_M
5. A/B: prefill throughput improvement at PP 228 (the profiled shape) and PP 1821 (the longest prompt)

## Evidence

- Profile: `docs/bench-evidence/gfx1100-pp-ab-20260830.md` — KQuantGemmK at 93.5%, grid/shape breakdown
- A/B baseline: `docs/bench-evidence/pp-ab-baseline.log`, `pp-ab-optimized.log`

## Git integration

One pull request (repository default). Spec committed before implementation.

## Stop conditions

- If the tiled kernel does not improve throughput by at least 1.5× at PP 228, stop and profile to understand why (likely: the kernel is not weight-bandwidth-bound, or L2 already provides sufficient reuse).
- If bit-exactness fails and cannot be restored by matching the accumulation order, stop and document the divergence.
- If the kernel fails to compile or crashes at the production shapes, stop and debug before proceeding.

## Outcome

### Tiled kernel: REJECTED

The weight-shared tiled kernel (`KQuantGemmKTiled`) was implemented and
measured. It was **31% slower** than the baseline at both PP 228 and PP 1821:

| PP | Base TTFT (ms) | Tiled TTFT (ms) | Ratio |
|---|---|---|---|
| 228 | 710 | 933 | 0.76x (slower) |
| 1821 | 5662 | 7397 | 0.77x (slower) |

Root cause: the gfx1100's 6 MB L2 cache already provides weight reuse across
warps reading the same weight row. A Q4_K weight row for K=2560 is 1440 bytes
(10 superblocks × 144 bytes). With 228 warps, the first warp loads it from
global memory and the remaining 227 hit L2. The tiled kernel adds shared
memory copy overhead (byte-by-byte loop) and `__syncthreads()` barriers
without any benefit — the weight is already cached. The stop condition fired.

### Hardware Dp4a: ADOPTED

The actual bottleneck was **software Dp4a**: the `Dp4a` function did 4 int8
multiplies + 4 adds in scalar instructions. Replacing it with the hardware
`v_dot4_i32_i8` instruction (`__ockl_sdot4`) collapses 8 scalar operations
into 1 instruction. The change is a 6-line function body replacement — no
kernel structure change, no shared memory, no synchronization.

Bit-exactness: signed int8×int8→int32 dot product is exact in both hardware
and software. The hardware instruction and the scalar expansion compute the
same integer result. Verified: token IDs identical to baseline, NMSE ≤ 5e-4
vs CPU oracle for all formats (Q4_K, Q5_K, Q6_K, Q8_0).

A/B results (median of 2 runs, Qwen3.5-4B Q4_K_M, RX 7900 XTX, ROCm 7.15):

| PP | Base TTFT (ms) | HW-Dp4a TTFT (ms) | Speedup | Base PT (tok/s) | HW-Dp4a PT (tok/s) | PT gain |
|---|---|---|---|---|---|---|
| 28 | 105.7 | 76.5 | 1.38x | 164.2 | 187.8 | 14.4% |
| 64 | 221.7 | 158.0 | 1.40x | 224.4 | 286.8 | 27.8% |
| 128 | 411.5 | 283.2 | 1.45x | 271.1 | 368.0 | 35.7% |
| 228 | 713.7 | 482.5 | 1.48x | 295.0 | 419.4 | 42.2% |
| 911 | 2790.0 | 1850.0 | 1.51x | 320.7 | 477.6 | 48.9% |
| 1821 | 5692.0 | 3731.0 | 1.53x | 317.5 | 480.9 | 51.5% |

The speedup grows with prompt length: 1.38x at PP 28 to 1.53x at PP 1821.
At PP 1821, prefill throughput rises from 317.5 to 480.9 tok/s — a 51.5%
gain from a 6-line change.

The hardware instruction also benefits decode (m == 1), since `Dp4a` is
called from the same `DotQ4K`/`DotQ5K`/`DotQ6K` functions used by both
prefill and decode kernels.

### What was rejected and why

- **Weight-shared tiled kernel**: L2 cache already provides weight reuse.
  Shared memory copy + sync adds overhead without benefit.
- **MROWS sweep**: moot — the tiled kernel was rejected.
- **Q6_K/Q5_K tiled variants**: moot — the tiled kernel was rejected.

## Owed

- CUDA port of the hardware Dp4a (CUDA already has `__dp4a`; the ROCm path
  was the only one using software Dp4a)
- Decode A/B: the hardware Dp4a also speeds up decode, but the TG200 campaign
  measured decode at ~103 tok/s with the software Dp4a. A re-measurement with
  the hardware Dp4a may move the TG200 target.
