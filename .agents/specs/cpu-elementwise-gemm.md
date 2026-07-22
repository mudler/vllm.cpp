# Leaf spec: elementwise CPU GEMM — dtype specialization + bit-exact SIMD

**Row:** `KERNEL-GEMM-CPU-ELEM` (kernel-matrix.md) · **status:** `ACTIVE` —
**E1-E4 landed** (2026-07-22): the elementwise (f32/f16/bf16) CPU GEMM is
specialized per dtype, multi-accumulator, SIMD-vectorized on AArch64 NEON and
x86-64 SSE2/F16C, and M-blocked, **bit-identical to the historical scalar
kernel on every path**. Binding on dgx aarch64: prefill **3.41×**, decode
**3.11×**, output tokens byte-identical (one md5 across all arms). Against
llama.cpp the CPU position moves **decode 3.38× → 1.03× behind (parity within
3.1 %)**, **prefill 8.20× → 2.34× behind**; RSS unmoved at 2.29× (loader work,
row `L5`) · **upstream pin:** llama.cpp local fork `237ad9b96` (b9892) ·
**parent evidence:** [CPU floor re-measurement](cpu-llamacpp-floor-remeasure-2026-07-22.md)
lever 3 + the [CIQ G4 result](gguf-compute-in-quant-gemm.md) §"Consequences for
the plan", which promoted this to the #1 CPU lever on measured grounds: the
bench file is MIXED (1.615 GiB of `f16` against 1.062 GiB of `q8_0`), so ~60 %
of the weight bytes — including the 970 MiB tied `token_embd`/`lm_head`, the
biggest GEMM in the model — cannot take the quant path at all.

## Scope

- **In scope:** the CPU `kMatmul` (`[K,N]` weight) and `kMatmulBT` (`[N,K]`
  weight) kernels for the elementwise dtypes f32/f16/bf16 — hoisting the
  per-element dtype switch out of the K loop, replacing the single serial f32
  accumulator with 16 independent ones, SIMD micro-kernels per architecture
  with a runtime feature probe, and M-blocking to amortize weight loads.
  Covers every safetensors CPU path and every non-block-quant GGUF tensor.
- **Out of scope:** the block-quantized GEMM (`kMatmulBTQuant`, leaf
  [gguf-compute-in-quant-gemm.md](gguf-compute-in-quant-gemm.md) rows G5-G8);
  RSS/residency (`QUANT-GGUF-KEEPQ-LOADER` **L5** — mmap-in-place + tied-head
  sharing); `BatchedMatmul` and the other CPU kernels (sub-1 % of wall time in
  the attribution profile); any CUDA path.
- **Correctness contract:** BIT-EXACTNESS, not tolerance. Every output element
  keeps the strictly sequential f32 accumulation over `p` with the product
  rounded before the add (`-ffp-contract=off`, CMakeLists.txt:21) that the
  historical kernel had. `VT_CPU_REF=1` and `VT_CPU_MATMUL_TIER=ref` both
  reproduce the historical path.

## Upstream chain

Pinned local fork `/home/mudler/_git/llama.cpp` @ `237ad9b96`.

| Component | Upstream anchor | Content |
|---|---|---|
| bf16 dot | `ggml/src/ggml-cpu/vec.cpp:139` `ggml_vec_dot_bf16` | multi-accumulator mul/add SIMD dot; the AVX2 `LOAD` macro at `:172` is the bf16 widen-by-shift-left-16 we mirror |
| f16 dot | `ggml/src/ggml-cpu/vec.cpp:264` `ggml_vec_dot_f16` | per-ISA widening load + multi-accumulator dot |
| declarations | `ggml/src/ggml-cpu/vec.h:72-73` | what `mul_mat` dispatches to for the two float weight types |
| SIMD abstraction | `ggml/src/ggml-cpu/simd-mappings.h` | `GGML_F16_VEC_LOAD` — the hardware widening convert (`vcvt_f32_f16` on AArch64, `_mm_cvtph_ps` under F16C) |
| chunk worker | `ggml/src/ggml-cpu/ggml-cpu.c:1155-1243` | the 16×16 tile our caller already ports (`blck_0`/`blck_1` = 16 at `:1192-1194`) |
| feature probe discipline | `ggml/src/ggml-cpu/ggml-cpu.c` `ggml_cpu_has_*` | runtime ISA selection with the generic tier always built |

**RECORDED DEVIATION — vectorize across OUTPUT COLUMNS, not along K.** ggml
vectorizes each dot along K and finishes with a horizontal reduce, which
REASSOCIATES the sum: correct, but not bit-comparable to a sequential f32
reduction. We keep ggml's SIMD primitives (the bf16 shift-widen, the f16
hardware convert, separate mul+add rather than FMA) but assign one output
column to each SIMD LANE, so 16 outputs advance together while each one's
reduction stays in `p` order. On the `[N,K]` weight that costs a 4×4 transpose
per 4 elements of K; the M-blocked kernel amortizes it over 4 activation rows.
The result is byte-identical to the scalar reference — which is what the
CUDA-captured goldens, the run-to-run/thread-count determinism contract, and
the `VT_CPU_REF` oracle all require, and what made this change land with **zero
token movement**.

## Our baseline (what was wrong)

- [`MatmulOneChunk`](../../src/vt/cpu/cpu_ops.cpp) called `LoadF32(t, off)` for
  BOTH operands INSIDE the K loop; `LoadF32` switches on `t.dtype`, so the hot
  loop paid two dtype branches per multiply-accumulate.
- The K loop had ONE accumulator, i.e. a loop-carried dependency on a single
  f32 adder: one MAC per FP-add latency. Measured 0.77-0.84 GFLOP/s/thread —
  which is exactly what a ~4-cycle add latency predicts, and 18-24 GFLOP/s
  across 20 threads.
- `F16ToF32` ([dtype.cpp:177](../../src/vt/dtype.cpp#L177)) is a branchy scalar
  bit-twiddle, called once per element.

## Port map

| Upstream | Local | Notes / deviations |
|---|---|---|
| `ggml_vec_dot_bf16` / `_f16` (`vec.cpp:139,264`) + `simd-mappings.h` | new [cpu_matmul_elem.{h,cpp}](../../src/vt/cpu/cpu_matmul_elem.cpp) — tier table `{bt, nk, btm, mr}` × {f32,f16,bf16} | output-column vectorization instead of K vectorization (deviation above); mul+add never FMA |
| `ggml_cpu_has_*` runtime probe | `ElemGemmTier()`, selected once per process; `VT_CPU_MATMUL_TIER=ref\|portable` forces a tier for a SAME-BINARY A/B | portable tier always built and is what CI exercises |
| AArch64 NEON | `Bt16Neon` / `Nk16Neon` / `BtM4Neon` | `vshll_n_u16(v,16)` bf16 widen, `vcvt_f32_f16` f16 widen, `vtrnq_f32`-based 4×4 transpose, MR=4 (16 accumulator vectors + 4 weight vectors fit AArch64's 32 SIMD registers) |
| x86-64 SSE2 (baseline, no probe) + F16C (probed) | `Bt16Sse2` / `Nk16Sse2` / `BtM2Sse2`, `Bt16F16c` / `Nk16F16c` / `BtM2F16c` | `_mm_unpacklo_epi16(zero,v)` is the SSE2 spelling of the bf16 shift-widen; MR=2 because SSE2 has only 16 XMM registers. Wider AVX2/AVX512 remains work row **G5** |
| `ggml-cpu.c:1155-1243` chunk worker | [`MatmulOneChunk`](../../src/vt/cpu/cpu_ops.cpp) rewritten around the tier table; the original loop is kept verbatim as `MatmulOneChunkRef` | same tile, same output set, same order; activation rows widened to f32 ONCE per 16-row tile |

## Tests to port

| Upstream test | Local tier / file | Notes |
|---|---|---|
| llama.cpp `tests/test-backend-ops.cpp` MUL_MAT cases | T-unit [test_ops_matmul_elem.cpp](../../tests/vt/test_ops_matmul_elem.cpp) | our bar is STRICTER than upstream's NMSE: `memcmp` byte-identity against an independent in-test scalar reference, over 3 activation dtypes × 3 weight dtypes × 2 output dtypes × 2 orientations × 14 shapes |
| — (our determinism contract, cpu_threadpool.h) | same file | bit-identical across thread counts 1/2/4/8 |
| — (conversion fidelity) | same file | the ENTIRE 16-bit domain: all 65,536 f16 and bf16 patterns driven through the op, inf/NaN patterns separately so a NaN cannot swallow a sum |
| existing goldens | the whole CPU ctest + the dgx CUDA regression set | unchanged by construction; verified |

## Gates

1. **Bit-exactness:** op output byte-identical to the scalar reference on every
   dtype/orientation/shape/thread-count, and the SIMD widening identical to
   `vt::F16ToF32`/`BF16ToF32` over all 65,536 patterns. **MET** (654 assertions
   green on x86-64 AND on dgx aarch64, and under all three tier settings).
2. **Oracle discipline:** `VT_CPU_REF=1` reproduces the historical tokens.
   **MET** — one md5 `d235db12f2cd304007530286a1755c95` across the BEFORE,
   AFTER and `VT_CPU_REF=1` arms, the same md5 G4 recorded.
3. **No regression:** clean CPU `-Werror` build, full CPU ctest, and the dgx
   CUDA SACRED set all unchanged. **MET** — dgx CUDA build clean with **0
   warnings**; each gate STANDALONE: 27B 235/235, 35B 315/315, Qwen3-Coder 6/6,
   Qwen3-dense 16/16, OPT 6/6, DeepSeek-V2 **223/223 assertions (8/8 prompts)**,
   `test_qwen36_gguf_engine` 28/28, plus `test_gguf_keep_quant`,
   `test_ops_quant_dot`, `test_ops_quant_traits`, `test_gguf_dequant`; golden
   corpus md5 `2965ef5772b556d3f3f86fedf4221b2f` (475 files) identical before and
   after. Dev box full CPU ctest **157/157**. Two transients recorded honestly:
   `test_openai_api_server` and `test_openai_conformance` flaked once under
   `ctest -j2` on the co-tenanted dev box and are green standalone, and the first
   DeepSeek-V2 pass returned the known co-scheduled-memory abort signature (95 of
   223 assertions, 1 failure) while ANOTHER agent was running 27B gates on the
   same box — re-run STANDALONE under the flock it passes 223/223, the same
   signature the CIQ G3 record documents.
4. **Perf:** binding same-binary A/B on the B4 recipe, idle dgx aarch64, one
   `flock`, 3 reps. **MET** — prefill 3.41×, decode 3.11×.
5. **Floor:** match or beat llama.cpp on decode t/s, prefill t/s and peak RSS.
   **DECODE effectively MET (1.03× behind, 3.1 %); PREFILL 2.34× behind;
   RSS 2.29× — both still open** (see "Re-ranked next lever").

## Dependencies

- `QUANT-GGUF-CPU-THREADPOOL` (`ParallelFor` + the mul_mat chunk policy) —
  landed; every number here is at 20 threads and the kernel is called from
  inside its chunk worker.
- `QUANT-GGUF-CIQ-GEMM` **G4** — landed; keep-quant is ON in every arm of this
  A/B, so the measurement isolates the elementwise kernel rather than
  re-measuring G4. G4's `expand_nk` is also why the hot orientation is `[N,K]`,
  which is the orientation the M-blocked micro-kernel targets.
- Hardware: **dgx.casa (GB10, aarch64, 20 cores)** is the ONLY binding host —
  the x86 dev box is `VOID` for timing (co-tenanted). CI and the dev box run the
  portable and SSE2/F16C tiers for correctness. No GPU lock is needed for the
  kernel itself, but the whole build+measure series holds one `flock
  $HOME/gpu.lock` per the benchmark protocol.
- Oracle: llama.cpp fork build at `237ad9b96` (b9892) on the same box/file, plus
  the in-tree scalar reference (`MatmulOneChunkRef`, reachable at runtime via
  `VT_CPU_MATMUL_TIER=ref`) for the bit-exactness gate.
- Models: `Qwen3.5-2B-UD-Q8_K_XL.gguf` for the binding A/B; the APEX 35B
  Compact/Balanced GGUFs for the e2e token gate.

## Work breakdown

| W | Row (claim-sized) | Content | Depends | Status |
|---|---|---|---|---|
| E1 | dtype specialization | hoist the per-element `LoadF32` switch out of the K loop; widen the activation row to f32 once per tile; typed weight micro-kernels | - | **DONE** 2026-07-22 |
| E2 | multi-accumulator portable tier | 16 independent accumulator chains (`kElemLanes`), both orientations, all 9 dtype pairs; always built, CI tier | E1 | **DONE** 2026-07-22 |
| E3 | arch SIMD tiers + runtime probe | AArch64 NEON; x86-64 SSE2 with F16C probed; `VT_CPU_MATMUL_TIER` A/B switch | E2 | **DONE** 2026-07-22 |
| E4 | M-blocking | `ElemBtMFn` — `mr` activation rows share one weight load + transpose (MR=4 NEON, MR=2 SSE2) | E3 | **DONE** 2026-07-22 |
| E5 | x86 AVX2/AVX512 tier | 8/16-wide lanes for the elementwise path; merge with the quant tier's probe | E3 | open — ranks WITH `QUANT-GGUF-CIQ-GEMM` **G5**, and only after a fresh x86 binding host exists (the dev box is `VOID` for timing) |

## Result (2026-07-22)

### Correctness — bit-exact, nothing moved

- Unit gate [test_ops_matmul_elem](../../tests/vt/test_ops_matmul_elem.cpp):
  **5 cases / 654 assertions** green on x86-64 and on dgx **aarch64**, and
  identically green under `VT_CPU_MATMUL_TIER=ref` and `=portable`. The gate is
  `memcmp`, not `Approx`.
- The exhaustive widening case proves the hardware converts are exact
  substitutes: all 65,536 f16 patterns through `vcvt_f32_f16` (AArch64) and
  `_mm_cvtph_ps` (F16C) agree bit-for-bit with `vt::F16ToF32`, **including
  every inf/NaN pattern** — the signaling-NaN quieting we were prepared to have
  to document does not occur.
- **No SIMD path in this change is non-bit-exact**, so the NMSE ≤ 5e-4 fallback
  the plan allowed for was not needed and no golden was regenerated.
- End to end, the BEFORE (`VT_CPU_MATMUL_TIER=ref`), AFTER (production) and
  ORACLE (`VT_CPU_REF=1`) output token streams are **byte-identical**: one md5
  `d235db12f2cd304007530286a1755c95` over 9 runs — the same md5 G4 recorded, so
  the CPU token stream has now been stable across two consecutive kernel
  rewrites.

### Benchmark — binding, dgx.casa (GB10, aarch64, 20 cores), idle

Whole build+measure series under one `flock $HOME/gpu.lock`; transferred by
`git archive` (never rsync); `nvidia-smi` reporting no compute apps and no
co-tenant CPU process at series start. Model `Qwen3.5-2B-UD-Q8_K_XL.gguf`,
llama.cpp `237ad9b96` (b9892). Keep-quant is ON in every arm (the G4 production
default), so this A/B isolates the ELEMENTWISE kernel.

**llama.cpp** (`llama-bench -t 20 -ngl 0 -r 5`): pp512 210.30±0.90,
**pp128 173.28±1.75**, tg128 25.06±0.31, **tg32 24.52±0.45**; isolated tg32
24.54±0.79 with peak RSS 2,934,136 KB = **2.798 GiB**. Two further series on
the same box returned pp128 171.27±3.40 and 173.68±2.22, so the denominator
reproduces. (Two `tg32` legs in later series returned 17.40±9.70 and 3.95±1.83
under page-cache pressure from our own 6.4 GiB arms and are DISCARDED as
contaminated, not averaged in.)

**Ours — SAME BINARY, three arms**, `VLLM_CPP_CPU_THREADS=20`,
`--input-len 128 --output-len 32 --concurrency 1 --seed 0 --temperature 0`,
3 reps each, medians:

| arm | TTFT (ms) | TPOT (ms) | peak RSS (KB) |
|---|---|---|---|
| BEFORE — `VT_CPU_MATMUL_TIER=ref` (the historical kernel) | 5,907.66 | 130.74 | 6,712,412 |
| **AFTER — production default** | **1,730.38** | **42.03** | **6,712,340** |
| ORACLE — `VT_CPU_REF=1` | 2,977.07 | 92.18 | 7,788,252 |

Rep spread on the AFTER arm: TTFT 0.70 %, TPOT 2.3 %. The BEFORE arm
reproduces G4's published AFTER arm (5,970.21 / 130.72 ms / 6,712,368 KB) to
1.1 % / 0.02 % / 28 KB, which is what makes this a valid same-binary A/B.

| axis | BEFORE | AFTER | ours: A/B | vs llama.cpp BEFORE | **vs llama.cpp AFTER** |
|---|---|---|---|---|---|
| Prefill (128 / TTFT) | 21.67 t/s | **73.97 t/s** | **3.41×** | 8.00× behind | **2.34× behind** |
| Decode (1000 / TPOT) | 7.649 t/s | **23.79 t/s** | **3.11×** | 3.21× behind | **1.03× behind** (3.1 %; 1.032× vs isolated tg32) |
| Peak RSS | 6.401 GiB | **6.401 GiB** | 1.00× | 2.29× worse | **2.29× worse** |

The `VT_CPU_REF=1` oracle arm also got 8.3× faster on TTFT and 4.9× on TPOT
versus G4's measurement of it, because the dequant-to-bf16 path runs through
the same elementwise kernel. It still emits the identical tokens.

### Op-level GFLOP/s (bf16 × bf16, 20 threads, best of 3, aarch64)

| shape (M×N×K) | `ref` `[N,K]` | portable | **NEON** | NEON ÷ ref |
|---|---|---|---|---|
| decode qkv 1×3072×2048 | 18.39 | 35.17 | **81.44** | 4.43× |
| decode o 1×2048×2048 | 18.09 | 34.83 | **73.16** | 4.04× |
| decode gate_up 1×12288×2048 | 22.16 | 50.76 | **122.48** | 5.53× |
| decode down 1×2048×6144 | 18.20 | 34.75 | **68.80** | 3.78× |
| decode lm_head 1×248320×2048 | 24.40 | 52.20 | **133.99** | 5.49× |
| prefill qkv 128×3072×2048 | 24.29 | 51.46 | **351.11** | 14.5× |
| prefill o 128×2048×2048 | 24.10 | 51.48 | **346.61** | 14.4× |
| prefill gate_up 128×12288×2048 | 24.28 | 51.57 | **349.98** | 14.4× |
| prefill down 128×2048×6144 | 24.01 | 50.67 | **347.06** | 14.5× |

The `[K,N]` orientation (`kMatmul`, the safetensors CPU path) moves 7.4-19.5 →
33-269 GFLOP/s over the same tiers. On x86-64 (dev box, `VOID` for binding, 8
threads, indicative) the same table is 6.0-6.7 → 38-122 GFLOP/s.

Two readings worth keeping:

1. **The portable tier alone is ~2.1× ref**, with no intrinsics at all. That is
   the multi-accumulator fix — the old kernel was latency-bound on one f32
   adder, not bandwidth- or ISA-bound. Any future backend inherits it.
2. **The elementwise prefill GEMM now reaches 347-351 GFLOP/s against the
   quant tier-0's measured 388-417** at the same shapes. The dtype gap in the
   GEMM is essentially closed; a mixed GGUF no longer has a slow half.

### Re-ranked next lever — prefill is NO LONGER elementwise-GEMM-bound

This is the change's most useful negative result, and it is measured, not
inferred. **E4 (M-blocking) raised the prefill op-level throughput 1.63×
(216 → 351 GFLOP/s) and moved end-to-end TTFT by 0.0 %** (1,702.88 → 1,730.38
ms across series, inside the 0.7-1.5 % rep spread). So:

1. **The 95.37 % `kMatmul` attribution is STALE and must not be reused.** It was
   measured when the elementwise GEMM was 24 GFLOP/s; it is now 350. The next
   CPU lever must be selected from a FRESH op-dispatch profile of the current
   binary at the prefill operating point — the same temporary `vt::GetOp` hook
   the floor re-measurement used. Nothing should be started before that profile.
2. **Prime suspects for the remaining 2.34× prefill**, to be confirmed by that
   profile, not by this document: the QUANT GEMM at M=128 (tier-0 hits 388-417
   GFLOP/s but llama.cpp runs i8mm repack there — rows **G6**/**G7**, which the
   G4 result had parked and this result partially un-parks), and the non-GEMM
   tail, which was under 5 % of a 24 s TTFT and is a materially larger share of
   a 1.73 s one.
3. **RSS is now the largest single deficit at 2.29×**, and it is loader work,
   not kernel work: `QUANT-GGUF-KEEPQ-LOADER` **L5** (mmap-in-place residency +
   sharing the tied `lm_head` with the embedding table) is ~1.9 GiB of the
   3.6 GiB gap.
4. **Decode is at parity** (1.03×, inside llama.cpp's own ±1.8 % run spread) and
   is DRAM-bound from here; it should not receive further kernel effort.
5. **E5 / G5 (x86 AVX2)** stay open but cannot be gated: the x86 box is `VOID`
   for binding numbers. The x86 tier shipped here is SSE2 4-wide, so the
   headroom there is real but unmeasurable on current hardware.

### Reproduction

```sh
# dgx.casa, aarch64, idle, ONE flock for the whole build+measure series
R=$HOME/work/bench-cpu-llama
M=$R/models/Qwen3.5-2B-UD-Q8_K_XL.gguf
LB=$R/llamacpp/build/bin
W=$HOME/work/vllm.cpp-elemgemm        # transferred with `git archive`, never rsync

cmake -S $W -B $W/build-cpu -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build $W/build-cpu -j 20 --target vllm-bench

flock $HOME/gpu.lock sh -c '
  uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  LD_LIBRARY_PATH='"$LB"' '"$LB"'/llama-bench -m '"$M"' -p 512,128 -n 128,32 -t 20 -r 5 -ngl 0
  LD_LIBRARY_PATH='"$LB"' /usr/bin/time -v '"$LB"'/llama-bench -m '"$M"' -p 0 -n 32 -t 20 -r 3 -ngl 0
  for arm in "VT_CPU_MATMUL_TIER=ref" "" "VT_CPU_REF=1"; do
    for rep in 1 2 3; do
      env $arm VLLM_CPP_CPU_THREADS=20 /usr/bin/time -v '"$W"'/build-cpu/examples/vllm-bench \
        --model '"$M"' --num-prompts 1 --input-len 128 --output-len 32 \
        --concurrency 1 --seed 0 --temperature 0 --output-token-ids /tmp/ids.$arm.$rep.json
    done
  done
'
# All nine token files MUST be md5 d235db12f2cd304007530286a1755c95.
```

## Risks/decisions

- **Bit-exactness over peak throughput (decided here).** The obvious extra ~2×
  on the ALU-bound prefill path is FMA (`vfmaq_f32` / `_mm_fmadd_ps`) instead of
  separate mul+add. It is deliberately NOT taken: it changes every rounding and
  would break the byte-identity that lets this change ship without regenerating
  a single golden. If it is ever wanted it belongs in its own opt-in row, gated
  on NMSE ≤ 5e-4 with the tokens re-verified against the llama.cpp oracle — and
  it should be measured first, because E4 already showed prefill is not bound
  by this kernel's ALU work.
- **Thread-local widened-activation scratch.** Up to 16×K f32 per worker
  (~0.4 MB at K=6144, ~8 MB across 20 threads), allocated once for the process
  lifetime. Negligible against a 6.4 GiB peak, and it is why the RSS axis is
  unchanged to within 72 KB.
- **MR is per-tier, not per-shape.** MR=4 on NEON and MR=2 on SSE2 are register-
  file limits, not tuned constants. A shape-adaptive MR was not attempted
  because E4's e2e result says this kernel is no longer the bottleneck.
- **`VT_CPU_MATMUL_TIER`** is a diagnostic/A-B switch, not a supported product
  knob: unset means "the best tier this CPU probes into", which is what
  production runs.
