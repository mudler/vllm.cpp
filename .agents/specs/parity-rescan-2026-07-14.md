# Parity rescan and strategy challenge — 2026-07-14

**Kind:** grounded scoping report (diagnostic; changes no lifecycle state and
grants no speed credit). **Method:** 30-agent dynamic workflow
(`wf_002c1692`): binding-grid axis forensics + record grounding, nine
subsystem scans of pinned vLLM (`e24d1b24` / v0.25.0 target `702f481`) and its
dependency chain versus our source, adversarial per-lever verification (16
checked, 9 confirmed-real, of 31 candidates), a strategy challenger and a
completeness critic. **Coverage caveat:** four scan lanes (attention,
host-scheduler, kv-cache, sampling-logits) returned placeholder structured
outputs and two lanes (gdn, strategy-challenge) died at the structured-output
retry cap; those six lanes must be re-run before this report is treated as
exhaustive. The conclusions below rest on the lanes that returned grounded
evidence (axis forensics, trace mining, moe, quant-fusion, cudagraph,
completeness critic) and were verified against source.

## Grounded current position

- The single binding result `3f256ab` (55/124) is **27B-only**: the manifest
  planned {27,35} but the evidence root contains zero 35B artifacts
  (`ratios.json` models==["27"]). 35B performance is entirely unmeasured under
  the v0.25.0 denominator.
- **Axis composition of the 69 failing axes** (failure mass 1-normalized):
  decode latency (TPOT+ITL) **52.2%**, coupled E2EL **12.3%**, coupled
  throughput **12.0%**, **host memory 23.6%**, prefill/TTFT **0.0%** — TTFT
  passes 24/24 and beats vLLM (c16 1.43×, c32 1.45×). Failures concentrate at
  c2/c4/c8 (65.7% of mass); c1 misses are within noise (+0.29–0.89%); c16/c32
  mostly pass (17/20, 18/20). Total-throughput ratios: c1 0.9935, c2 0.9545,
  c4 0.9664, c8 0.9807, c16 1.0279, c32 1.0394.
- Reproducible tail outliers across all three repetitions: c8 p99 ITL
  **1.78×** (~857 vs ~481 ms) and c32 p90 ITL **1.21×** — fat tails, not mean
  shifts, mechanism unconfirmed.
- **Central verified contradiction:** on the only per-kernel timing evidence
  (H1d c16 window, `c498a413-residual-ranking.json`), our GPU kernels are
  collectively **net faster** than vLLM (all-kernels 124.575 vs 128.153
  ms/window, −3.579; GDN recurrence −6.297 is our biggest win), yet wall-clock
  decode is 4–6% slower at c2–c8. The failing decode-coupled mass (76.5%) is
  therefore **host-side** (transport / scheduling / per-step overhead), not
  kernel compute. Caveat: the only c2-adjacent per-kernel comparison is
  cross-profiler and untrustworthy; a same-profiler nsys full-step gap diff at
  c2 is the missing attribution measurement.

## Ranked levers

| # | Lever | Status | Evidence | Axes helped | Effort |
|---:|---|---|---|---|---|
| 1 | **TCP_NODELAY on the SSE server.** `third_party/httplib/httplib.h:142` defaults `CPPHTTPLIB_TCP_NODELAY` to false and no `set_tcp_nodelay(true)` call exists; vLLM serves via uvicorn, which enables TCP_NODELAY by default. One SSE frame per token means tiny per-token writes — the exact Nagle/delayed-ACK pattern, masked at high concurrency. Uniquely fits c2–c8-fail/c16–c32-pass, the fat c8/c32 ITL tails, and the GPU-faster/wall-slower contradiction. Magnitude UNMEASURED (loopback behavior must be observed before crediting) | CONFIRMED diff; gain hypothesis | source-verified both sides | c2–c8 TPOT/ITL/E2EL/throughput + c8 p99, c32 p90 tails; both models | trivial (one line) + one non-binding localhost A/B |
| 2 | **Host-memory mirror.** (a) cheap allocator precheck (malloc_trim(0), MALLOC_ARENA_MAX, smaps_rollup after load) to split live mirror vs glibc arena retention; (b) direct-to-final-device weight streaming replacing the retained 22.920 GiB host mirror (`qwen3_5_dense_weights.cpp:165` owner). Only path to the two hard-failing memory axes (PSS 0.585×, RSS 0.592×; ~46 vs ~27 GiB) | RECORDED (existing diagnosis) | BENCHMARKS.md:54–74 | peak PSS/RSS both models + tail pressure relief | precheck trivial; loader redesign HIGH (needed for 35B regardless) |
| 3 | **Async-scheduling W3 + de-synced sampled-ID D2H** (`ENG-ASYNC-SCHED`, spec async-serving.md): depth-2 overlap and GPU-resident sampled tokens; the state-record scan already selected this for the c2 gap (~4,627 stream syncs; pageable D2H before main-stream sync). Size FIRST with the nsys full-step c2 gap diff — siblings W1/W2/W4 measured neutral (+0.215% with TTFT regression) | RECORDED | engine-matrix `ENG-ASYNC-SCHED` row; vLLM `config/vllm.py:952-1043`, `v1/engine/core.py:519-607`, `worker/gpu/async_utils.py:12-70` | c2–c8 decode axes | medium; own every-axis component |
| 4 | **qkvz merged input projection** (`KERNEL-GEMM-BF16` W2). The de-prioritizing premise ("bf16 GEMM at parity, −0.232 ms") is sign-inverted: bf16-family medians are 51.663 vs 48.798 ms (+2.865, ours slower, cross-profiler caveat); most of that is closed by the landed BA merge, remaining qkvz-only recovery ≈0.476 ms/window | RECORDED (blocked behind packed component) | BENCHMARKS.md:236-249 | c1–c8 decode-shaped axes (launch count + modest ms) | medium; reuse BA pattern |
| 5 | **Tail workstream (diagnose first):** no `cudaStreamCreateWithPriority` / `cudaAccessPolicyWindow` anywhere in the decode path; fixed HTTP worker-pool floor + keep-alive/timeouts audit for SSE at c8–c32. Verify Nagle first — may share the root | PLAUSIBLE | grep-verified absence; `api_server.cpp:44-66` pool | c8 p99 / c32 p90 ITL tails (~19% of mass) | low to diagnose |
| 6 | **[35B-banked, prohibited until 27B 124/124]** shared-expert aux-stream overlap mirroring vLLM `shared_experts.py:66-140` MULTI_STREAM_OVERLAPPED (≤256 tokens); ours runs SharedExpertUngated serially (`qwen3_5.cpp:3670-3684`); vt:: lacks an aux-stream/event fork-join primitive | RECORDED | source-verified both sides | 35B c1–c8 decode (unmeasured) | medium |
| 7 | **[STOP/deprioritized]** kernel micro-levers: PDL (measured-negligible, 2.3 µs median inter-kernel gap under graphs), FP4-producer vectorization W3-H/W3-I (lands on the already-passing c16 window; precedent-negative), pre-capture/widen-grid/embed-in-graph (off-gate robustness items for after 124/124) | CONFIRMED off-axis | H1d window + W3-E/F/G/I strict-fail precedent | none currently failing | — |

## Record corrections

1. **DISPROVEN:** the "fused RMSNorm→NVFP4 producer" / "RMSNorm whole-chain
   fusion" lever and the "+1.810 ms RMSNorm/generated-partitions residual" are
   not parity gaps. vLLM's `RmsNormQuantFusionPass` fuses **FP8 only** (its
   FUSED_OPS carries no nvfp4 keys); the single-GPU nvfp4 path runs a
   standalone `scaled_fp4_quant` on the rmsnorm's bf16 output exactly like
   ours, and `scaled_fp4_quant` is an opaque custom op Inductor cannot codegen
   into surrounding fusions. The +1.81 ms was a cross-profiler artifact.
2. **Sign-inverted premise:** qkvz was de-prioritized on "bf16 GEMM already at
   parity (−0.232 ms/window)"; the recorded medians actually show ours +2.865
   ms slower (cross-profiler caveat; mostly closed by the landed BA merge).
3. **Mis-ranked:** the H1d "fused SiLU→FP4 displaces normal-producer H2"
   selected-candidate ranking is c16-window-derived and off-axis: it targets a
   kernel family we already win, on a concurrency that already passes. It must
   not drive order-0 sequencing.
4. **REFUTED:** the "35B W4A4 auto-select prefill gap" — the 35B checkpoint is
   W4A16, vLLM auto-selects Marlin (`modelopt.py:1401-1413`), the same
   arithmetic class we run; `--moe-backend marlin` is redundant, not causal.
5. **Scope:** "55/124" understates the parity surface: it is 27B-only, and the
   SGLang cache-neutral and shared-prefix floors are dependency-blocked
   (SERVE-ASYNC-LLM incremental streaming; KV-MAMBA-ALIGN).
6. Upstream fatal-log restoration (the c16 diagnostic) confirmed the port had
   dropped `core.py:1233` and `async_llm.py:703-705` — an example of silent
   1:1 drift the scans should watch for.

## Recommended sequence

1. **Unchanged and first:** the c16 packed HTTP-500 diagnostic → repair →
   full 12-leg component rerun (sole unblocker of packed → qkvz → exact grid
   → 35B).
2. **In parallel, now (kernel/component-independent):** land TCP_NODELAY
   (spike + test-first + one non-binding localhost sizing); run the trivial
   memory precheck and scope the weight-streaming loader spike.
3. **Then:** nsys full-step GPU-idle gap diff at c2 to attribute the ~6.5 ms
   between transport / async-sched / residual kernel, before committing W3.
4. **After a passing component:** qkvz, then the single authorized exact-grid
   rerun credits everything together against the 124 axes.
5. **Only after 27B 124/124:** measure the 35B grid (zero evidence today),
   confirm the 35B/SGLang MTP denominator, then the shared-expert overlap.
6. **Stop:** FP4-producer/PDL/fused-norm-quant micro-levers against the gate.

## Follow-ups owed

- Re-run the six lost scan lanes (attention, host-scheduler, kv-cache,
  sampling-logits, gdn, strategy-challenge) with smaller schemas.
- Write `specs/serve-tcp-nodelay.md` (spike gate for lever 1) before its row
  turns ACTIVE; reconstruct per-request `itls[]` from the binding raw root to
  locate the c8 p99 stall cadence (read-only).
- The full workflow report and per-agent journals are retained under the
  session transcript (`wf_002c1692-f41`); this file is the durable record.
