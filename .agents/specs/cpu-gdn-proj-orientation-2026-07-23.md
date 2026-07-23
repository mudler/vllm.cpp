# CPU prefill: keep the GDN split projections in [N,K] (nk=true) — the last slow-orientation GEMM

**Rows:** `BACKEND-GATE-CPU-LLAMACPP` (backend-matrix) · `QUANT-GGUF-KEEPQ-LOADER`
(the `GgufLoadPolicy::expand_nk` orientation lever this extends) · evidence for
`BENCH-CPU-LLAMA`. **claim:** `CLAIM-CPU-GDN-ORIENT-1`. **base:** `bead27d`.
**upstream pin:** llama.cpp local fork `237ad9b96` (b9892).

## Why (fresh op-dispatch profile of the CURRENT binary, ranked the work)

The prior lever (G6 Arm i8mm) left CPU **decode at llama.cpp parity, prefill
1.44× behind** on the q8_0-dominant `Qwen3.5-2B-UD-Q8_K_XL.gguf`. A fresh
op-dispatch profile of `bead27d` (the temporary `vt::GetOp` wall-time hook,
warm page cache, `--input-len 128 --output-len 1`, 20 threads, dgx aarch64,
total attributed 1070 ms = TTFT) with a per-GEMM shape histogram settled where
prefill time goes:

| op | share | calls | note |
|---|---|---|---|
| **kMatmulBTQuant** | **50.4 %** | 95 | q8_0 quant GEMM (i8mm) |
| **kMatmul** ([K,N], SLOW) | **17.9 %** | 72 | GDN split projections, no M-blocking |
| **kMatmulBT** ([N,K], fast) | 14.9 % | 20 | f16 ffn/attn expanded weights |
| kGdnPrefill | 5.2 % | 18 | threaded already |
| kPagedAttention | 3.2 % | 6 | threaded already |
| everything else | < 6.6 % combined | | |

GEMMs are 83 % of prefill. The shape histogram identified the 72 `kMatmul`
calls EXACTLY: `M=128 N=2048 K=2048 ×36` + `M=128 N=16 K=2048 ×36` = the four
GDN split projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`)
across all 18 GDN layers. They ran the **N-striding `kMatmul`** (no M-blocking,
memory-bandwidth-bound at M=128) instead of the **M-blocked `kMatmulBT`**.

**Root cause.** G4 gave every OTHER expanded weight the file's own [N,K]
orientation (`GgufLoadPolicy::expand_nk`), routing it to the M-blocked
`vt::MatmulBT`. But `LoadGdnGguf` (`qwen3_5_gguf_weights.cpp`) was NOT updated:
its expand branches dequant to [N,K] then **`TransposeBf16` into Matmul-B [K,N]
with nk=false**, i.e. extra load-time work to reach the SLOWER kernel — the exact
defect the floor re-measurement named as "lever 2, weight orientation", left
unclosed for the GDN projections only.

## The lever (gain ÷ effort #1)

Highest gain ÷ effort by a wide margin: LOW effort (a loader-flag change,
bit-identical by construction), moving 17.9 % of prefill from the slow to the
fast kernel. G7 (quant-GEMM repack, 50.4 %) is larger but MEDIUM–HIGH effort and
is the next lever, not this one.

- `GgufLoadPolicy::gdn_expand_nk` (rides `expand_nk`; `VT_GGUF_GDN_NK=0` is the
  narrow same-binary A/B opt-out; `VT_CPU_REF=1` reproduces the historical
  transpose): [gguf_keep_quant.{h,cpp}](../../src/vllm/model_executor/model_loader/gguf_keep_quant.cpp).
- `MakeGdnProj` in [qwen3_5_gguf_weights.cpp](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp):
  with the flag, the projection is KEPT in its dequantized [N=out, K=in] order
  (nk=true) — the V-head reorder (`ReorderVRows`/`ReorderVCols`) is applied to
  the same buffer first and is orthogonal to orientation; without the flag,
  TODAY's transpose to [K,N] is preserved. Applied to `in_proj_qkv`,
  `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`.

**Bit-identical, not near-tie.** `vt::MatmulBT` ([N,K]) and `vt::Matmul` ([K,N])
are the SAME sequential f32 K-reduction differing only in the weight memory
offset (`MatmulOneChunk<kBT>` vs `<false>`, cpu_ops.cpp), and the M-blocked
`BtM4Neon` preserves per-output p-order — so the result is byte-identical to the
transposed load. The float weight VALUES are identical between [N,K] and [K,N]
storage; only indexing differs. On a device whose GEMM picks its algorithm from
the operand layout (cuBLASLt) the flag stays OFF, exactly as `expand_nk` does.

Grounding: llama.cpp keeps every GGUF weight in its native `[out=N, in=K]` disk
order and its `mul_mat` contracts along the contiguous K row
(`ggml/src/ggml-cpu/ggml-cpu.c:1245-1443 @ 237ad9b96`); there is no transpose to
a K-major weight anywhere in its CPU path.

## Result (binding, dgx aarch64, idle, one flock, 2026-07-23)

**Prefill 1.090× faster same-binary; decode 1.09× (TPOT 44.1 → 40.4 ms = llama
tg32 parity); vs llama.cpp pp128 1.44× → 1.32× behind. Output tokens
byte-identical, peak RSS unchanged (3.884 GiB, 1.39×).**

`Qwen3.5-2B-UD-Q8_K_XL.gguf`, `--input-len 128 --output-len 32 --concurrency 1
--seed 0 --temperature 0`, `VLLM_CPP_CPU_THREADS=20`, loadavg-gated < 1.2 inside
one `flock`, 5 reps (cold rep 1 discarded, median of 2–5). llama.cpp `237ad9b96`
refreshed same session (`-t20 -r5 -ngl0`): pp512 211.25, **pp128 176.01±2.04**,
tg32 24.93.

| arm | TTFT median (ms) | prefill t/s | TPOT (ms) |
|---|---|---|---|
| BEFORE (`VT_GGUF_GDN_NK=0`) | 1125.05 | 113.8 | 44.1 |
| **AFTER (default)** | **1031.94** | **124.0** | **40.4** |

**Same-binary prefill 1.090×** — the binding, box-state-robust metric — and a
SECOND gated run corroborates (1.091×: BEFORE 1067.10 → AFTER 977.72 ms). RSS
4,073,020 kB = 3.884 GiB on both arms (the change removes the load-time
transpose but the resident bytes are identical). This session's ABSOLUTE t/s ran
~7 % below the idle-box G6 baseline (BEFORE 113.8 vs the recorded 122.2 t/s for
the identical code — residual load from the same-session build/gate activity, 5-min
loadavg elevated), so the vs-llama ABSOLUTE ratio uses the recorded idle G6
baseline: 122.2 t/s (1.44×) × 1.090 = **133 t/s ≈ 1.32× behind**. Decode 1000/40.4
= 24.75 t/s = **1.01× llama tg32 (at parity)**.

### Correctness — NO token movement

Output-token md5 `d235db12f2cd304007530286a1755c95` is **byte-identical across
all three arms** — AFTER (default), BEFORE (`VT_GGUF_GDN_NK=0`) and ORACLE
(`VT_CPU_REF=1`) — and across thread counts 1/4/20. `VT_CPU_REF=1` reproduces
the historical transpose path exactly. Bit-identity is structural, not a
near-tie: `vt::MatmulBT` and `vt::Matmul` are the same sequential f32
K-reduction differing only in the weight memory offset.

### Fresh post-change profile (same warm hook, prefill)

| op | before | **after** |
|---|---|---|
| kMatmulBTQuant (q8_0 quant GEMM) | 50.7 % | **55.0 %** |
| **kMatmul** ([K,N] slow) | **17.9 % (72 calls)** | **0 % (0 calls) — ELIMINATED** |
| **kMatmulBT** ([N,K] fast) | 14.9 % (20 calls) | **27.7 % (92 calls)** — absorbed the 72 |
| kGdnPrefill | 5.2 % | 5.7 % |
| kPagedAttention | 3.2 % | 3.2 % |

The slow-orientation `kMatmul` is gone from prefill entirely; every elementwise
GEMM now runs the M-blocked `kMatmulBT`. **Next CPU prefill lever = the quant
GEMM (kMatmulBTQuant, now 55 %): G7 repack-at-load for q8_0** (the elementwise
GEMM at 27.7 % is the fast M-blocked kernel already; a further win there needs
FMA, which reassociates and is out of scope under the byte-identity bar).

### Gates (all GREEN)

- Clean CPU `-Werror` build **0 warnings**; CPU unit ctest **136/136** (0 failed;
  the big-model CUDA-goldened engine tests are excluded, as they always are on
  the CPU build — they fail on CPU by design and OOM-risk at -j2).
- `test_qwen36_gguf_engine` STANDALONE **2/2 cases · 28/28 assertions · 16/16
  tokens** on APEX-Compact AND APEX-Balanced vs the same-file llama.cpp oracle
  (exercises the GDN projections through the new orientation at 35B scale).
- CUDA clean `-Werror` build **0 warnings** (the change is CUDA-inert:
  `gdn_expand_nk` is false whenever `expand_nk` is, i.e. on any device that is
  not the CPU quant path, and safetensors never reaches `LoadGdnGguf`).
- Regression set UNCHANGED, each STANDALONE, all pass 0-failed — 27B
  `test_qwen27_paged_engine`, 35B `test_qwen36_paged_engine`, Qwen3-Coder
  `test_qwen3coder_paged_engine`, Qwen3-dense `test_qwen3_paged_engine`, OPT
  `test_opt_paged_engine`, DeepSeek-V2 `test_deepseek_v2_paged_engine`, Llama
  `test_llama_paged_engine` (goldens unchanged; CUDA-inert change).
