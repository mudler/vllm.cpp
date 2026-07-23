# CPU prefill: thread the two serial non-GEMM kernels (kGdnPrefill + kPagedAttention)

**Rows:** `QUANT-GGUF-CPU-THREADPOOL` (quantization-matrix / kernel-matrix — the
existing `vt::` CPU threadpool leaf; this extends its chunked-dispatch coverage
to the last two single-threaded kernels on the prefill hot path) ·
`BACKEND-GATE-CPU-LLAMACPP` (backend-matrix). **claim:**
`CLAIM-CPU-THREAD-GDN-PAGED-1`. **base:** `4884d03`. **upstream pin:**
llama.cpp local fork `237ad9b96` (b9892).

## Why (measured, not assumed)

After the elementwise-GEMM vectorization (`KERNEL-GEMM-CPU-ELEM`) and the
keep-quant loader (L5), CPU **decode is at llama.cpp parity (1.03×)** but
**prefill is 2.34× behind (73.97 vs 173.28 t/s)**. A fresh op-dispatch profile of
the current binary (idle dgx aarch64) settled where prefill time goes:

- kMatmulBTQuant 37 %, **kGdnPrefill 25 %**, kMatmul 12 %, kMatmulBT 10 %,
  **kPagedAttention 10 %**.
- The GEMM kernels already run at 211–417 GFLOP/s (quant) / 351 (elementwise) —
  NOT the bottleneck.
- **kGdnPrefill (`cpu_ops.cpp`, the GDN linear-attention recurrence) and
  kPagedAttention (`cpu_paged_attn.cpp`) are 35 % of prefill and BOTH ran
  single-threaded** at c1: `GdnPrefillKernel` chunked over SEQUENCES (n=1 at c1),
  `PagedAttentionKernel` had no threadpool at all. That serial 35 % is the gap.

## Scope — thread both, using the EXISTING `ParallelForRows` infra

No new threadpool machinery: reuse `vt::cpu::ParallelForRows` (the 1:1 ggml
flash-attn row-chunk port already in `cpu_threadpool.{h,cpp}`).

### kGdnPrefill — parallel axis = (sequence, value-head)

The gated-delta recurrence is **sequential along a sequence** (state `S` carries
forward in `tok`), so time is NOT a parallel axis. But **value-heads are fully
independent**: head `hv` owns a disjoint state block `state[(s*Hv+hv)]` and
writes disjoint output rows `out[(t*Hv+hv)]`, reading its own q/k/v/g/beta head
slice (GQA key head `hk = hv/(Hv/Hk)`). This is exactly how the GPU GDN chunks
(runner.cpp `Hv`/`Hk` head split; vLLM fla `chunk_gated_delta_rule` parallelizes
`BH = batch*heads`). The old kernel parallelized over sequences only, leaving c1
(n=1) single-threaded even though there are `Hv` heads.

- Extracted `GdnHeadTokenStep` (one head, one token) from `GdnTokenStep`;
  `GdnTokenStep` (used by the decode kernel, batch-parallel) now calls it in a
  head loop — byte-identical, loop order irrelevant (disjoint state per head).
- `GdnPrefillKernel` chunks over `n*Hv` items; each item runs its head's
  time-sequential recurrence. At c1 (Qwen3.5-2B: `Hv=16`) that is 16-way; on the
  27B/35B GDN path `Hv` is larger.

### kPagedAttention — parallel axis = query token (row)

The (request, local-token, q-head) nest produces disjoint output rows
`out[(t*Hq+h)]`. At c1 prefill `num_reqs==1`, so the request loop is serial;
tokens are the embarrassingly-parallel axis (mirrors ggml `flash_attn_ext`
chunking over `neq1*neq2*neq3` = query-pos × head × batch,
`ggml-cpu/ops.cpp:9070-9126`). Flattened the request/local-token nest into the
global query-token index by precomputing, once on the caller (O(total_q)), each
token's absolute position `p`, its request's `seqlen`, and its request row (for
the block table), then `ParallelForRows(total_q)` with the per-head softmax loop
unchanged inside. At c1 prefill (128 tokens) that is 128 rows → full 20-way with
causal-imbalance handled by the 4×-oversubscribed chunk-steal.

## Correctness — numerics-delicate, bit-identity is the bar

Both kernels keep their exact per-element math and per-output f32 reduction
order; parallelism partitions OUTPUT elements only (disjoint state/rows, no
`atomicAdd`, no reassociated reduction) → bit-identical to single-thread by
construction, the same argument the threadpool leaf already makes for the GEMM
and row kernels. Gates:

- Real model: CPU GGUF output-token md5 stays `d235db12f2cd304007530286a1755c95`
  (byte-identical, and run-to-run + across threads 1/2/4/8/20).
- `test_cpu_threadpool` determinism battery EXTENDED with a single-sequence
  `Hv=20>nth` GdnPrefill case (proves head-parallelism at c1) and a 37-token
  causal PagedAttention case; byte-identical at threads 1/3/20.
- `VT_CPU_REF=1` still reproduces the historical path.
- All model regression gates (27B, 35B 315/315, Qwen3-Coder 6/6, Qwen3-dense
  16/16, OPT 6/6, DeepSeek-V2 8/8) UNCHANGED — kGdnPrefill is on the 27B/35B GDN
  path and Qwen3-dense.

## Benchmark (owed on dgx aarch64, idle, one flock, foreground)

Same recipe as the floor re-measurement: `Qwen3.5-2B-UD-Q8_K_XL.gguf`,
`vllm-bench --input-len 128 --output-len 32 --concurrency 1 --seed 0
--temperature 0`, `VLLM_CPP_CPU_THREADS=20`, 3 reps + cold-leg discard,
`git archive` transfer, loadavg-gated inside the lock. Report: prefill
before(`4884d03`)/after + new ratio vs llama.cpp pp128 (173.28±1.75), op-level
thread-scaling 1→20 for both kernels, a FRESH post-change profile + the new
bottleneck.

## Result (binding, dgx aarch64, idle, 2026-07-23)

- **Prefill same-binary 1.382×** (TTFT 1753→1269 ms; 73.0→100.9 t/s). vs
  llama.cpp pp128 (refreshed same session, 177.54±2.16): **2.43× → 1.76× behind**.
- **Decode at parity** — settled TPOT (output-len 96) 40.3→41.0 ms (1.04× vs
  llama tg32 25.30, inside spread). The output-len-32 41.5→44.6 gap is an
  early-decode warmup artifact of the faster TTFT, not a kernel change.
- **Op-scaling 1→20:** kGdnPrefill 7.08× (peaks 7.8× at 8 threads — `Hv=16`
  head cap), kPagedAttention 8.96×.
- **Fresh post-change profile:** the two kernels 35 % → **8.6 %** of prefill
  (kGdnPrefill 25→4.1 %, kPagedAttention 10→4.6 %). **New bottleneck = the GEMMs**
  (kMatmulBTQuant 50 % + kMatmul 16 % + kMatmulBT 14 % = 80 %) ⇒ next CPU prefill
  lever is the SIMD/repack GEMM tiers (G5/G6/G7), not another non-GEMM kernel.
- Byte-identity held (md5 `d235db12f2cd304007530286a1755c95` at threads 1/4/20 +
  `VT_CPU_REF=1`); CPU ctest 158/158; determinism battery extended.

Full repro + tables: [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md) § "thread
kGdnPrefill + kPagedAttention". Status `ACCEPTED`.
