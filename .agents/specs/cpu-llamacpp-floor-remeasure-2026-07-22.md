# CPU vs llama.cpp — floor re-measurement, attribution and ranked plan (2026-07-22)

**Rows:** `BACKEND-GATE-CPU-LLAMACPP` (backend-matrix) · `BENCH-CPU-LLAMA`
(specs/competitive-benchmarks.md) · evidence for `QUANT-GGUF-CIQ-GEMM` **G4**,
`QUANT-GGUF-KEEPQ-LOADER` **L4**, `QUANT-GGUF-CPU-THREADPOOL` **W4** ·
**claim:** `CLAIM-BENCH-CPU-LLAMA-REMEASURE-1` ·
**base:** `1cb5f64` · **upstream pin:** llama.cpp local fork `237ad9b96` (b9892).

**Purpose.** The recorded CPU position (README §CPU + the
[B4 decision row](../parity-ledger.md#L291)) was measured 2026-07-10, BEFORE the
threadpool and the whole GGUF compute-in-quant track. This re-measures it,
attributes the gap to a specific op, and ranks the levers on measured numbers.
**No code changed.** The deliverable is measurement + attribution + plan.

## Direct answer

**No — we are not better than llama.cpp on any axis.** We are behind on all
three. But the gap is much smaller than the record claimed on two of the three
axes, and it is now attributable to **one op**.

| Axis | Recorded (B4, 2026-07-10, x86) | **Measured now (binding, aarch64)** | Moved? |
|---|---|---|---|
| Decode | 54–75× behind | **11.6× behind** | YES — ~5–6× better |
| Prefill | ≈1,480× behind | **33.5× behind** | YES — ~44× better |
| Peak RSS | 2.7× worse | **2.65× worse (7.427 vs 2.798 GiB)** | NO — unchanged |

The decode/prefill movement is the landed **threadpool** (W1–W3), which was not
in the B4 arm. The RSS axis has not moved by a single kilobyte, because nothing
on the residency path is on an executed code path yet.

## Binding measurement — dgx.casa (GB10, aarch64, 20 cores), IDLE

Chosen as the binding host because it was genuinely idle (load average 0.01
before the series; `nvidia-smi` reported no compute apps) and the whole series
ran under one `flock $HOME/gpu.lock`. Run-to-run spread on our arm is 0.36 %
(TTFT) / 0.59 % (TPOT), and peak RSS is identical to the kilobyte across reps.

- **Model:** `Qwen3.5-2B-UD-Q8_K_XL.gguf` — `qwen35` dense, 1.94 B params,
  2.68 GiB file; 103 `q8_0` + 56 `f16` + 176 `f32` tensors; `n_embd` 2048,
  `n_layer` 24, `n_ff` 6144, `n_head` 8 (head dim 256), `n_head_kv` 2,
  `n_vocab` 248320.
- **Ours:** `1cb5f64`, transferred by `git archive` (never rsync), built
  `-DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_BUILD_TESTS=OFF
  -DVLLM_CPP_SERVER=OFF`.
- **llama.cpp:** `237ad9b96` (b9892), `-DCMAKE_BUILD_TYPE=Release
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON`.

### llama.cpp arm (`llama-bench -t 20 -ngl 0 -r 5`)

| test | t/s |
|---|---|
| pp512 | 211.16 ± 1.07 |
| pp128 | 174.63 ± 1.75 |
| tg128 | 26.13 ± 0.58 |
| tg32 | 25.80 ± 1.03 |
| tg32 isolated (`-p 0 -n 32 -r 3`, `/usr/bin/time -v`) | 25.16 ± 0.36 |
| peak RSS (isolated tg32) | 2,934,068 KB = **2.798 GiB** |

### Our arm (`vllm-bench --input-len 128 --output-len 32 --concurrency 1 --seed 0 --temperature 0`, `VLLM_CPP_CPU_THREADS=20`, 3 reps)

| rep | TTFT (ms) | TPOT (ms) | duration (s) | peak RSS (KB) |
|---|---|---|---|---|
| 1 | 24,594.52 | 450.68 | 38.57 | 7,788,220 |
| 2 | 24,545.16 | 453.05 | 38.59 | 7,788,220 |
| 3 | 24,633.13 | 450.39 | 38.60 | 7,788,208 |
| **spread** | **0.36 %** | **0.59 %** | 0.08 % | 12 KB |

Derived: prefill 128 / 24.591 s = **5.205 t/s**; decode 1000 / 450.7 ms =
**2.219 t/s**; peak RSS **7.427 GiB**.

### Ratios (binding)

| Axis | llama.cpp | ours | ratio |
|---|---|---|---|
| Prefill (pp128 vs TTFT-derived) | 174.63 t/s | 5.205 t/s | **33.5× behind** |
| Decode (tg32 vs TPOT-derived) | 25.80 t/s | 2.219 t/s | **11.6× behind** (11.3× vs the isolated 25.16) |
| TPOT | 38.8 ms | 450.7 ms | 11.6× worse |
| Peak RSS | 2.798 GiB | 7.427 GiB | **2.65× worse** |

### Threadpool same-binary A/B (`VLLM_CPP_CPU_THREADS=1` vs `=20`)

| | 1 thread | 20 threads | speed-up |
|---|---|---|---|
| TTFT | 306,623.09 ms (0.4175 t/s) | 24,594.52 ms (5.205 t/s) | **12.47×** |
| TPOT | 3,629.84 ms (0.2755 t/s) | 450.68 ms (2.219 t/s) | **8.05×** |
| Peak RSS | 7,788,056 KB | 7,788,220 KB | **1.000×** |

This is the reproduction `QUANT-GGUF-CPU-THREADPOOL` **W4** has been waiting
for since the 2026-07-10 contended-host abort. Against its stated acceptance
(both axes ≥ 10× at 20 threads, RSS ≤ 1.05×, outputs token-exact): **prefill
12.47× PASSES, RSS 1.000× PASSES, decode 8.05× MISSES the 10× bar.** The
decode shortfall is itself a small lever (per-op kick cost at M=1 shapes —
see lever 7).

## x86 dev box — `VOID` for binding, indicative only

The x86 box that carried the original B4 number has persistent co-tenant load
(a 19-hour `python3` watcher at ~30 % of a core, another agent's Go/postgres/
`oci.test` suite, plus dockerd/containerd/kube-apiserver). The protocol makes a
contended run void, and the data show exactly why: llama.cpp `pp128`
261.41 **± 30.95** (±11.8 %) and `tg128` 6.61 **± 1.60** (±24 %), and our rep 3
returned TPOT 1,505.83 ms against 780.67 / 797.67 ms for reps 1–2 — a 1.9×
outlier inside one series. **These numbers are recorded as indicative and are
NOT binding.**

| Axis | llama.cpp | ours (reps 1–2) | indicative ratio |
|---|---|---|---|
| pp128 | 261.41 ± 30.95 t/s | 2.959 t/s | ~88× |
| tg32 | 8.26 ± 0.33 t/s (isolated 8.13 ± 1.22) | 1.267 t/s | ~6.5× |
| peak RSS | 2,935,516 KB = 2.799 GiB | 7,788,880 KB = 7.428 GiB | 2.65× |

The RSS axis is the one x86 figure worth keeping: it is identical to the
aarch64 figure and to B4's, on both boxes and both thread counts.

**The x86 single-thread arm independently validates the B4 baseline.** Running
the *current* binary at `VLLM_CPP_CPU_THREADS=1` on the *same box* B4 used
returns TTFT **586,745.67 ms** and TPOT **6,170.50 ms**, against B4's recorded
559,724 ms / 5,848 ms — a 4.8 % / 5.5 % agreement across twelve days and a
different build. That is the cleanest possible demonstration that (a) the B4
measurement was sound and its number was never an artefact, and (b) the
*entire* movement on the decode and prefill axes is the threadpool, not any
change to the GEMM itself. The x86 same-binary threadpool A/B is 13.6 ×
prefill / 7.8 × decode, consistent with aarch64's 12.47 × / 8.05 ×.

**Rerun recipe when the box is exclusively idle:** the same `bind.sh` series,
gated on `/proc/loadavg` 1-min < 1.5 sustained, with the co-tenant agents
quiesced.

## Attribution — where the time goes

`perf` is unavailable on the dev box (`perf_event_paranoid = 4`, no
passwordless sudo, no gdb/eu-stack/pstack), so attribution was taken with a
**temporary, uncommitted op-dispatch profiler**: a hook in `vt::GetOp`
(`src/vt/ops.cpp`) that attributes the wall time between one `GetOp` and the
next on the same thread to the first op. It therefore charges each op its
kernel time **plus** whatever host glue follows it, and it accounts for all
wall time: total attributed **8.560 s** against a reported benchmark duration
of **8.56 s** (100.0 %). The instrumentation was reverted before the binding
runs, which used the clean binary.

Workload: `--input-len 16 --output-len 4`, `VLLM_CPP_CPU_THREADS=20`, x86.

| op | share | calls | ms/call |
|---|---|---|---|
| **`kMatmul`** | **95.37 %** (8.164 s) | 748 | 10.915 |
| `kMoeSiluMul` | 0.99 % | 96 | 0.882 |
| `kRmsNorm` | 0.83 % | 196 | 0.361 |
| `kGdnPrefill` | 0.73 % | 18 | 3.450 |
| `kRmsNormGated` | 0.50 % | 72 | 0.592 |
| `kCausalConv1dUpdate` | 0.29 % | 54 | 0.458 |
| `kCastBf16` | 0.28 % | 48 | 0.494 |
| `kSigmoidGateBf16` | 0.25 % | 24 | 0.881 |
| `kGdnPostConv` | 0.17 % | 72 | 0.202 |
| `kCausalConv1dFwd` | 0.15 % | 18 | 0.701 |
| `kPagedAttention` | 0.13 % | 24 | 0.447 |
| `kGdnDecode` | 0.12 % | 54 | 0.194 |
| everything else (7 ops) | < 0.21 % combined | | |

**The entire CPU gap is one op: the elementwise bf16 dense GEMM.** Attention,
the GDN recurrence, norms, conv, sampling and all host glue together account
for under 5 %. This is the opposite of the CUDA picture, and it means the CPU
plan is a GEMM plan, not a broad-front optimisation.

### Op-level GEMM microbenchmark (the "what would G4 buy" measurement)

A standalone harness (scratch, not committed) linked against `libvllm.a`,
timing `vt::Matmul` (the `[K,N]` orientation the model actually uses today),
`vt::MatmulBT` (`[N,K]`), and the landed-but-unrouted `vt::MatmulBTQuant`
tier-0 path, at this model's shapes. Best of 3, `VLLM_CPP_CPU_THREADS=20`,
aarch64 (dgx):

| shape (M×N×K) | `kMatmul` `[K,N]` bf16 — **production** | `kMatmulBT` `[N,K]` bf16 | `kMatmulBTQuant` Q8_0 tier-0 | quant ÷ production |
|---|---|---|---|---|
| decode qkv 1×3072×2048 | 11.79 GFLOP/s | 17.91 | **287.54** | 24.4× |
| decode o 1×2048×2048 | 9.22 | 17.73 | **264.39** | 28.7× |
| decode gate_up 1×12288×2048 | 11.01 | 22.28 | **222.22** | 20.2× |
| decode down 1×2048×6144 | 6.11 | 18.48 | **266.50** | 43.6× |
| decode lm_head 1×248320×2048 | 14.66 | 24.68 | **211.43** | 14.4× |
| prefill qkv 128×3072×2048 | 18.42 | 24.62 | **416.79** | 22.6× |
| prefill o 128×2048×2048 | 16.97 | 24.45 | **396.24** | 23.4× |
| prefill gate_up 128×12288×2048 | 14.86 | 24.52 | **393.48** | 26.5× |
| prefill down 128×2048×6144 | 11.93 | 24.24 | **388.34** | 32.6× |

Same harness on x86 (contended, indicative): production 6.1–12.6 GFLOP/s,
Q8_0 tier-0 130–200 GFLOP/s, i.e. 13–20×.

Two facts fall straight out:

1. **The already-landed portable tier-0 quant GEMM is 14–44× (typically ~24×)
   faster than the GEMM production runs**, on the op that is 95.4 % of wall
   time. It is correctness-gated (G3) and simply not called.
2. **Orientation alone costs 1.3–3.0×.** `kMatmulBT` reads the weight row
   contiguously; `kMatmul` strides by N down the K loop. The GGUF loader
   currently dequantises to bf16 **and transposes** into the slower `[K,N]`
   layout (`qwen3_5_gguf_weights.cpp:192-193` — "dequant to bf16 and transpose
   to Matmul-B [K, N]"), even though GGUF's native layout is already `[N,K]`,
   which is exactly the orientation `MatmulBTQuant` needs. The loader is doing
   extra work to reach a slower kernel.

### What G4 would land at (arithmetic on the measured op numbers)

Not measured end to end — this is projection from the table above, stated as
such.

- **Decode:** 24 × (0.044 + 0.032 + 0.226 + 0.094) ms + 4.811 ms (lm_head) =
  **14.3 ms/token ⇒ ~70 t/s** cache-warm upper bound. The real ceiling is DRAM:
  llama.cpp's own 25.80 t/s over a 2.68 GiB working set is ≈ 69 GB/s, so a
  keep-quant decode lands **at llama.cpp's bandwidth-bound rate**, not above it.
- **Prefill:** 24 × (3.864 + 2.710 + 16.373 + 8.295) ms = **750 ms for 128
  tokens ⇒ ~171 t/s** GEMM-only, against llama.cpp's pp128 174.63 t/s — within
  ~2 % before adding our sub-5 % non-GEMM tail.
- **Peak RSS:** keep-quant residency removes the bf16 expansion, so peak should
  fall to ≈ file size + activations ≈ 2.9–3.0 GiB against llama.cpp's 2.798 —
  roughly 1.05×, which is spec gate 5's bar.

So the honest reading is: **G4 is not an incremental step, it is most of the
gap.** The portable tier-0 kernels are already fast enough to put all three
axes within striking distance of the floor on aarch64; the SIMD and repack
tiers are what turn "close" into "at or beyond".

## Confirmed: the landed quant machinery is INERT

Three independent confirmations, all on `1cb5f64`:

1. `grep -rn MatmulBTQuant src/vllm/` returns **nothing** — no model call site.
2. `vt::MatmulBT` (`src/vt/ops.cpp:163`) hard-requires `IsFloat(b.dtype)` and
   dispatches unconditionally to `kMatmulBT`; there is no block-dtype branch.
3. Running with `VT_GGUF_KEEP_QUANT=1` **fails at load on both boxes** with
   `vt: matmul_bt: float inputs and f32/bf16 output required` — the loader can
   now keep blocks resident, and nothing downstream can consume them.

Peak RSS is byte-for-byte the B4 figure on both architectures and both thread
counts, which is the same fact seen from the memory side.

## Ranked levers (gain ÷ effort, all grounded in the numbers above)

| # | Lever | Measured/derived gain | Effort | Notes |
|---|---|---|---|---|
| **1** | **`QUANT-GGUF-CIQ-GEMM` G4** — route model GEMM call sites onto `MatmulBTQuant` + flip keep-quant default ON for GGUF | **14–44× on 95.4 % of wall time ⇒ ~9–17× e2e**, plus peak RSS 7.43 → ~2.9 GiB (2.65× deficit → ~1.05×). Projects to decode ≈ llama.cpp's bandwidth rate and prefill within ~2 % of pp128 | MEDIUM — **every dependency is already DONE** (loader L2/L3, threadpool W1–W3, kernels G1–G3). Work is: stop the loader dequant+transpose for GEMM weights, dispatch `MatmulF32`/`MatmulBf16` on `IsBlockQuant(w.dtype)`, regenerate GGUF engine goldens against the llama.cpp oracle, flip the default | **Do this first, and nothing else before it.** It is the only lever that moves all three axes at once |
| **2** | **Weight orientation** — stop transposing GGUF weights out of their native `[N,K]` into `[K,N]` | **1.3–3.0×** on the same 95.4 % (aarch64; 1.2–1.5× x86) | LOW | **NEW FINDING**, not previously on the plan. It is also a *prerequisite* of lever 1 (`MatmulBTQuant` is BT-only), so fold it into G4 rather than shipping it separately. As a standalone it is a free ~1.5–2× for the bf16 path |
| **3** | **SIMD `vec_dot` for the elementwise bf16/f16 GEMM** | Production GEMM measures 0.77–0.84 GFLOP/s single-thread — roughly two orders of magnitude under one AVX2 core. `MatmulOneChunk` (`cpu_ops.cpp:54-83`) calls `LoadF32(t, off)`, which switches on `t.dtype` **inside the K loop**; llama.cpp has `ggml_vec_dot_bf16`/`_f16` SIMD kernels for exactly this | LOW–MEDIUM | **NEW FINDING.** Covers everything G4 does not: the 56 `f16` tensors in this very file, every safetensors CPU path, and any non-block-quant GEMM. Independent of the quant track |
| **4** | **G5 x86 AVX2/AVX512 tier** | tier-0 measures 130–200 GFLOP/s on x86; llama.cpp's x86 arm is well above that. Expect 2–4× on the quant path | MEDIUM | Mechanical port of `arch/x86/quants.c`. Ranks above G6 because the x86 tier-0 numbers are *relatively* further from the floor than the aarch64 ones |
| **5** | **G6 Arm NEON/dotprod/i8mm tier** | Smaller marginal gain than the spec assumed: aarch64 tier-0 already reaches 211–417 GFLOP/s. Its real value is unlocking `nrows==2` mmla and the GB10/Apple tier | MEDIUM | `nrows` stays pinned at 1 until this lands, with the `ggml-cpu.c:1426-1433` boundary guards |
| **6** | **G7 repack-at-load** | **Re-scope after G4.** Tier-0 Q8_0 at M=128 already hits 388–417 GFLOP/s on aarch64, ~2 % off pp128 parity, so the prefill headroom repack was budgeted for is much smaller than assumed | MEDIUM–HIGH | Do not start before G4's measurement replaces this projection with a number |
| **7** | **Threadpool decode scaling** | Prefill scales 12.47×/20 threads, decode only **8.05×**. The gap is the per-op kick at M=1 shapes (the spec's own recorded fallback: `n_chunks==1` → run inline) | LOW | Also closes `QUANT-GGUF-CPU-THREADPOOL` **W4**, whose reproduction this document supplies |

Levers 4–6 should be re-ranked against a real G4 measurement rather than
against this projection.

## Reproduction

```sh
# ---- binding arm: dgx.casa, aarch64, idle, one flock for the series --------
R=$HOME/work/bench-cpu-llama
M=$R/models/Qwen3.5-2B-UD-Q8_K_XL.gguf
LB=$R/llamacpp/build/bin

# llama.cpp @ 237ad9b96 (b9892)
cmake -S llamacpp -B llamacpp/build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build llamacpp/build -j 16 --target llama-bench

# ours @ 1cb5f64 (transferred with `git archive`, never rsync)
cmake -S vllmcpp -B vllmcpp/build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_BUILD_TESTS=OFF -DVLLM_CPP_SERVER=OFF
cmake --build vllmcpp/build-cpu -j 16 --target vllm-bench

flock $HOME/gpu.lock sh -c '
  uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  LD_LIBRARY_PATH='"$LB"' '"$LB"'/llama-bench -m '"$M"' -p 512,128 -n 128,32 -t 20 -r 5 -ngl 0
  LD_LIBRARY_PATH='"$LB"' /usr/bin/time -v '"$LB"'/llama-bench -m '"$M"' -p 0 -n 32 -t 20 -r 3 -ngl 0
  for rep in 1 2 3; do
    VLLM_CPP_CPU_THREADS=20 /usr/bin/time -v ./build-cpu/examples/vllm-bench --model '"$M"' \
      --num-prompts 1 --input-len 128 --output-len 32 --concurrency 1 --seed 0 --temperature 0
  done
  VLLM_CPP_CPU_THREADS=1 /usr/bin/time -v ./build-cpu/examples/vllm-bench --model '"$M"' \
    --num-prompts 1 --input-len 128 --output-len 32 --concurrency 1 --seed 0 --temperature 0
'

# ---- inertness check (expected to FAIL at load, on either box) -------------
VT_GGUF_KEEP_QUANT=1 ./build-cpu/examples/vllm-bench --model "$M" \
  --num-prompts 1 --input-len 8 --output-len 1 --concurrency 1 --seed 0 --temperature 0
# => vllm-bench: failed: vt: matmul_bt: float inputs and f32/bf16 output required
```

## Honest limits of this measurement

- Single-request (`--concurrency 1`) on one model file. A server-endpoint
  series and a batched operating point still belong to
  `BACKEND-GATE-CPU-LLAMACPP`.
- The x86 arm is `VOID` for binding purposes (contended host), so the
  architecture comparison rests on one clean host. The x86 gap is plausibly
  *larger* than the aarch64 gap (llama.cpp's x86 pp128 is higher and our x86
  op-level numbers are lower), but that is not established here.
- The attribution profile was taken on the contended x86 box with temporary
  instrumentation, at a smaller workload than the binding runs. Its *shares*
  are what matter and they are extreme enough (95.4 % vs < 1 % for everything
  else) that contention cannot change the conclusion.
- The G4 projections are arithmetic on measured op throughput, not an
  end-to-end run. G4 owes the real number on this exact recipe.
