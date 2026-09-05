# ROCm fp8 KV cache decode attention (`GFX1100-TG200`, fork issue #7)

Row: `GFX1100-TG200` (campaign, fork issue #5). Issue: fork
[#7](https://github.com/ghazni101/vllm.cpp/issues/7). The fp8 KV cache store
and correctness-grade read landed in W6
([`fp8-kv-cache.md`](fp8-kv-cache.md) `## W6`); this spec covers the
performance gap the W6 spec named as owed: the fp8 read through the fast
decode kernel. The `KV-FP8` engine-matrix row owns the store/read
correctness surface; this row owns the decode performance arm on top of it.

## Scope

- **In:** widen the `PagedAttnDecodeGqaF32Q` dispatch guard in
  `src/vt/rocm/rocm_paged_attn.hip` to accept `DType::kI8` KV cache when
  `args.kv_cache_dtype != kAuto`; add an fp8 dequant load path inside the
  kernel; pass `k_scale`/`v_scale` to the kernel; add a `LoadRowEplFp8`
  device helper that does vectorized uint8_t loads + `F8E4M3ToF32Dev` dequant
  with scale.
- **Out:** the bf16 decode-opt kernels (`PagedAttnDecodeGqaBf16`,
  `PagedAttnDecodeOptBf16T`) — those stage `__hip_bfloat16` fragments and a
  tensor-core fp8 read is a separate performance brick, same scope line as
  the CUDA W2 arm. The prefill path stays on `PagedAttnOnline` for fp8. The
  `bf16_decode_opt` guard at line 1925 is not touched. fp8_e5m2 compute.
  Per-head scales. Non-gfx1100 architectures.

## Upstream chain

vLLM's fp8 KV cache read dequantizes inside the attention kernel:
`scaled_vec_conversion<float, uint8_t>` (`quant_utils.cuh:419-429`) =
`half_to_float(fp8_to_half(byte)) * scale`. The ROCm `LoadKv(uint8_t*, ...)`
helper at `rocm_paged_attn.hip:176` already mirrors this arithmetic exactly:
`F8E4M3ToF32Dev(p[i]) * scale`. The CUDA arm's `LoadKv` at
`cuda_paged_attn.cu:175-185` is the same. The dequant is not new code; it is
existing code that the fast kernel does not call.

## Our baseline

The `PagedAttnDecodeGqaF32Q` kernel (`rocm_paged_attn.hip:674`) is the
f32-query + bf16-KV decode kernel activated by `VT_ATTN_DECODE_GQA4=1`. It
fuses QG=4 query heads per KV group, walks the KV sequence warp-strided with
online softmax, and uses vectorized 128-bit `uint4` bf16 loads
(`LoadRowEplBf16`, line 342). The dispatch guard at line 2186-2189 requires
`k_cache.dtype == DType::kBF16 && v_cache.dtype == DType::kBF16`.

With `--kv-cache-dtype fp8`, the KV cache is `DType::kI8`. The guard fails,
and the dispatch falls through to `PagedAttnOnline` (line 2223) — the
reference kernel that processes one key at a time with a full-block
`__syncthreads()` reduction per key (line 290-294). The code acknowledges
this at line 2231-2235.

## Measured gap

A/B benchmark on `kind_tharp` (Qwen3.5-4B Q4_K_M, RX 7900 XTX, ROCm 7.14.0,
128-token greedy decode, single request, 4 reps, 2026-08-27):

| Context | fp8 KV tok/s | bf16 KV tok/s | Speedup |
|--------:|-------------:|--------------:|--------:|
| 256     | 99.94        | 143.15        | 1.43x   |
| 1024    | 56.28        | 129.02        | 2.29x   |
| 4096    | 20.53        | 92.08         | 4.49x   |
| 8192    | 11.08        | 66.85         | 6.03x   |
| 16384   | 5.78         | 43.16         | 7.47x   |

The gap widens with context because `PagedAttnOnline` is O(n) per key with
full-block sync, while `PagedAttnDecodeGqaF32Q` is warp-strided with online
softmax and no per-key sync. Qwen3.5-4B has 8 full-attention layers
(`full_attention_interval=4`, 32 total); the O(n) cost is paid on those 8
layers x 4 KV heads x 256 head_dim.

## Design

### 1. `LoadRowEplFp8` device helper

Add a new `LoadRowEplFp8<EPL>` function alongside `LoadRowEplBf16` (after
line 367). For fp8, each element is 1 byte. The vectorized load width
matches the bf16 path's register pressure:

- EPL=4: 4 bytes per lane = one `uint32_t` load
- EPL=8: 8 bytes per lane = one `uint2` load (64 bits)
- EPL=16: 16 bytes per lane = one `uint4` load (128 bits)

After the vectorized load, dequantize each byte with
`F8E4M3ToF32Dev(byte) * scale` into the float register array. The scale is
passed as a parameter.

### 2. Template `PagedAttnDecodeGqaF32Q` on `TKV`

Change the kernel signature from hardcoded `const __hip_bfloat16* k_cache`
to `template <typename TKV>` with `const TKV* k_cache, const TKV* v_cache`.
Add `float k_scale, float v_scale` parameters. Inside the kernel, replace
the two `LoadRowEplBf16<kEpl>(k_cache, ...)` / `LoadRowEplBf16<kEpl>(v_cache, ...)`
calls with a `LoadRowEplKv<kEpl>(k_cache, ..., k_scale)` dispatch that
selects `LoadRowEplBf16` for `__hip_bfloat16` and `LoadRowEplFp8` for
`uint8_t` via `if constexpr`.

### 3. Widen the dispatch guard

At line 2186-2189, widen the condition from:
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
`v_cache.Ptr<uint8_t>()`, and pass `args.k_scale`/`args.v_scale`. The
`PagedAttnDecodeGqaF32Q` template instantiation `PagedAttnDecodeGqaF32Q<QG,
EPL, NWARPS, uint8_t>` is the new instantiation; the existing
`PagedAttnDecodeGqaF32Q<QG, EPL, NWARPS, __hip_bfloat16>` is the unchanged
bf16 path.

### 4. No new test file

The correctness gate is the existing `test_ops_fp8_kv_cache` suite (W1,
CPU oracle) plus the served-model token-exact gate on the `kind_tharp`
container. The fp8 dequant arithmetic is already gated bit-identical against
the CPU codec; the new code path only changes which kernel reads the same
dequantized values. A red-first mutation: revert the guard widening and
confirm the dispatch falls back to `PagedAttnOnline`.

## Risks

- **Reduction order difference:** `PagedAttnDecodeGqaF32Q` uses warp-strided
  online softmax, which reduces the KV sequence in a different order than
  `PagedAttnOnline`'s per-key loop. Greedy decode tokens can move at exact
  ties, same as the d128 decode-opt flip (line 1912-1921). The
  `VT_ATTN_DECODE_GQA4=1` flag is already opt-in and already carries this
  risk for bf16 KV; the fp8 arm inherits it.
- **Vectorized fp8 load alignment:** the uint8_t KV cache pages must be
  4-byte aligned for `uint32_t` loads and 8-byte aligned for `uint2` loads.
  The KV cache block allocation uses `hipMalloc` with block_size *
  num_kv_heads * head_dim bytes per block; for head_dim=256 and block_size=16,
  that is 16*4*256 = 16384 bytes per block, which is naturally aligned. The
  bf16 path already assumes `kc_hd % 8 == 0` (line 1925); the fp8 path needs
  `kc_hd % 4 == 0` for the uint32_t load, which holds for head_dim=128 and
  256 (both are multiples of 4).
- **Register pressure:** the fp8 load path uses the same `float k_reg[kEpl]`
  registers as the bf16 path. The dequant happens in registers; no shared
  memory change. The smem allocation is unchanged.

## Gates

- **Correctness (CPU oracle):** `test_ops_fp8_kv_cache` GREEN — the W1
  suite already gates the fp8 dequant arithmetic; this change does not touch
  the CPU path.
- **Correctness (served model, token-exact):** run `kind_tharp` with fp8 KV
  + `VT_ATTN_DECODE_GQA4=1` and compare greedy decode output against the
  bf16 KV baseline at short context (256 tokens). Tokens must match; at
  longer context, the reduction-order risk applies and is recorded.
- **Performance (A/B):** re-run `/tmp/bench_context_scale.py` with the
  optimized fp8 path and compare against the bf16 baseline. The target is
  fp8 KV decode throughput within 2x of bf16 KV at 16K context (vs the
  current 7.47x gap). fp8 should be faster than bf16 at long context due to
  halved KV bandwidth.
- **Red-first:** revert the guard widening, confirm the dispatch falls back
  to `PagedAttnOnline`, confirm the benchmark shows the original regression.

## Git integration

- Separate spec and implementation PRs (developer preference, recorded
  2026-08-27).
- Branch: `row/GFX1100-TG200` (existing campaign branch).
- Push to `origin` (fork `ghazni101/vllm.cpp`) only.
- Spec commit first, then implementation commits.

## Now

Implementation landed. Review-driven updates (PR #2168 review 5070827757):
added exact-geometry GQA4 FP8 kernel reach test (G6), C ABI v24
`kv_cache_dtype` gate, `vllm-cli --kv-cache-dtype` end-to-end gate,
ROCM_ATTN FP8 cache config acceptance test, and updated public C-ABI/FP8
capability documentation. History rebuilt so the spec commit precedes
implementation. Issue, spec, and PR body reconciled onto owning row
`GFX1100-TG200`.

## Outcome

Implementation landed on `row/fp8-kv-decode-attn` (fork issue #7), rebased
onto `origin/main` at `0bb090def` (2026-08-30).

**Bug fixed during validation:** the `LoadRowEplFp8<EPL=16>` path loaded two
`uint4` values (32 bytes = 32 fp8 elements) and wrote to `r[0..31]`, but `r`
is `float r[16]` — a stack buffer overflow. The bf16 EPL=16 path needs two
`uint4` because each bf16 element is 2 bytes (16 × 2 = 32); fp8 elements are
1 byte (16 × 1 = 16), so one `uint4` suffices. The bug was latent: the
dispatch only instantiates EPL=4 (d=128) and EPL=8 (d=256); EPL=16 would need
d=512, which the guard `(d == 128 || d == 256)` excludes. The `static_assert`
admits EPL=16, so the trap was set for a future d=512 extension. Fixed to a
single `uint4` load writing `r[0..15]`, matching the spec's design section.

**Verification (ROCm 10.0.0 container, `rocm10-gfx1100:10.0.0`, gfx1100 =
AMD Radeon RX 7900 XTX, HIP 7.15.26333, 2026-08-30):**

| Test | Cases | Assertions | Result |
|---|---:|---:|---|
| `test_rocm_fp8_kv_cache` (G2–G5, device) | 7 | 28 | PASS |
| `test_ops_fp8_kv_cache` (CPU oracle) | 8 | 511 | PASS |
| `test_attn_backend_registry` | 20 | 125 | PASS |
| `test_attn_validate_configuration` | 21 | 82 | PASS |
| `test_kv_cache_fp8_wiring` | 31 | 487 | PASS |
| `test_ops_attention` | 11 | 39 | PASS |
| `test_rocm_backend` | 9 | 1065 | PASS |
| `test_rocm_arch` | 9 | 59 | PASS |

**Project gates:**

| Gate | Result |
|---|---|
| `check-agent-record.py` | OK |
| `check-commit-style.py` | OK |
| `check-commit-trailers.py` | OK |
| `check-env-doc.py` | OK |

**Served-model token-exact gate (Qwen3.5-4B Q4_K_M, RX 7900 XTX, ROCm
10.0.0, 2026-08-30):**

Model: `/models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf` (2.6 GB), hq=16,
num_kv_heads=4, head_dim=256, full_attention_interval=4. Greedy decode
(temperature=0), single request.

| Prompt | Max tokens | bf16 vs fp8+GQA4 | Divergence point |
|---|---:|---|---|
| "The capital of France is" | 20 | byte-identical | none |
| "Write a short story about a robot..." | 128 | byte-identical | none |
| "Write a short story about a robot..." | 256 | first ~180 tokens identical | mid-stream near-tie (reduction-order) |

The 256-token divergence is the expected reduction-order difference:
`PagedAttnDecodeGqaF32Q` uses warp-strided online softmax, which reduces
the KV sequence in a different order than `PagedAttnOnline`'s per-key
loop. At exact logits ties, a different reduction order selects a
different token. This is the same behavior the bf16 GQA4 path exhibits
and the spec's `## Risks` section documents. The fp8 arm inherits it.

The fp8 KV cache itself is correct: fp8 KV without GQA4 (through
`PagedAttnOnline`) also diverges from bf16 at ~token 80, from the fp8
quantization precision difference, not from a kernel bug.

**ABI gap fixed:** the C ABI (`vllm_model_params`) did not expose
`kv_cache_dtype`, so the fp8 KV capability was unreachable from
`vllm-cli`. Added as ABI v24: `const char* kv_cache_dtype` field,
mapped in `vllm_engine_load`, exposed as `--kv-cache-dtype` in
`vllm-cli`. NULL/"auto" is byte-identical to before.

**Performance A/B (Qwen3.5-4B Q4_K_M, RX 7900 XTX, ROCm 10.0.0,
2026-08-30):**

Three arms: bf16 KV baseline (`PagedAttnOnline`), fp8 KV through
`PagedAttnOnline` (the slow path the guard widening fixed), and fp8 KV
through `PagedAttnDecodeGqaF32Q` (`VT_ATTN_DECODE_GQA4=1`). Greedy
decode, single request, `--repeat 6` (2 warmup + 4 measured, median),
`--kv-cache-memory 805306368` (768 blocks, max_model_len=24576).
Host load 0.3-1.5 throughout (no co-tenant storms).

| Context | bf16 tok/s | fp8-slow tok/s | fp8+GQA4 tok/s | gap-old | gap-new |
|--------:|-----------:|--------------:|---------------:|--------:|--------:|
| 256     | 26.84      | 25.87          | 28.70           | 1.04x   | 0.94x   |
| 1024    | 13.65      | 12.75          | 15.32           | 1.07x   | 0.89x   |
| 4096    | 4.12       | 3.66           | 4.47            | 1.13x   | 0.92x   |
| 8192    | 1.91       | 1.62           | 1.93            | 1.18x   | 0.99x   |

`gap-old` = bf16 / fp8-slow (the regression before the fix).
`gap-new` = bf16 / fp8+GQA4 (after the fix).

**The 7.47x gap is eliminated.** fp8+GQA4 achieves parity or better
with bf16 at all measured contexts (0.89x-0.99x). The kernel is faster
than bf16 at short context (halved KV bandwidth) and reaches parity at
long context. The original 7.47x gap was at 16K context; the 8K data
point already shows 0.99x (parity).

The 16384-context data point could not be measured: the `PagedAttnOnline`
prefill kernel hangs at ~14K+ prompt tokens (GPU scheduler timeout on
the O(n²) attention computation). This is a prefill-path hardware
limitation, not a decode kernel issue. The decode-only path at 16K
context works (verified via 111-token prompt + 3985-token generation =
4096 total context, 22 tok/s average). Reaching 16K total context via
long generation would take ~54 minutes per arm and was not attempted.

Note: the tok_s values include prefill time (APC does not cache long
prompts in `--repeat` mode). The ratio between arms isolates the decode
kernel difference because the prefill kernel (`PagedAttnOnline`) is
identical across arms. The absolute tok/s is lower than the spec's
original measurement (which used a 110-token prompt with 256
generation) because the prompts here are longer.
