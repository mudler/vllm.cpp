# Spike: FlashInfer-parity NVFP4 small-M dispatch and SM12 tactics

**Row:** `KERNEL-GEMM-NVFP4-W4A4` · **state:** accepted implementation
spike; W1/W2 are measured and W3 is `ACTIVE` under
`CLAIM-NVFP4-SMALL-M-3`; W3-A delayed event timing is immutable
correctness/safety-green, performance-classified and strict-acceptance-failed;
W3-B at clean `d7cdf66` is immutable build/correctness/access-safety-green;
its repeated component is performance-classified and strict-acceptance-failed;
the corrected paired trace is lifecycle-clean and structurally FP4-equivalent,
and the replacement v0.25 node trace is accepted. It promoted W3-D packed QKV,
which is now implementation/
correctness/access-safety green in mutable staging; exact-oracle performance
and immutable evidence remain open.
**Historical implementation pins:** vLLM `e24d1b24fe96`, pip-vLLM `0.24.0`,
installed FlashInfer `0.6.12`, CUTLASS `v4.5.0`, CUDA `13.0.88`, and the
Qwen3.6-27B-NVFP4 gate snapshot recorded in `environment.md`.
**Current performance target:** vLLM v0.25.0 `702f481` with FlashInfer 0.6.13.
The old component/correctness evidence remains valid, but no vLLM 0.24.0
end-to-end ratio binds the active W3 decision.

This accepted spike is the mandatory contract for FP4 plan-cache and kernel
changes. W1 retains its CPU/TSan, exact/legacy sm_121a capture, focused
memcheck and both-27B-model proof. Pushed `bce2627` completes its component,
trace and exact oracle classification: it is structurally effective and
materially improves the 27B grid, but does not pass the strict component or
every-axis gates. The fresh runtime trace promoted the separate 32-tactic W2
family; W2 now implements and correctness-gates that family, merged CT gate/up
semantics and the traced one-input activation/quant producer. Its component is
positive but below the strict timing/memory floor. Clean pushed `b5c6e4f`
completed W2's old-oracle classification: it improved every concurrency and
won c16/c32 total throughput, but c1-c8, TPOT/ITL and host-memory axes remained
red. That ratio is now historical; its paired trace still legitimately
promoted W3 because local runtime selection was dominated by wide tactics while
vLLM resolved the 128x32x256 Stream-K/static pair. Exact `4e1d8ca` remains W1's
immutable before-state.

## Scope

Restore vLLM/FlashInfer-equivalent selection for dense NVFP4 W4A4 GEMMs on
SM120/SM121, starting with the exact 27B online gap. The row owns:

- hybrid `M` tuning buckets, including distinct 1/2/4/8/16 decode buckets;
- one tuning result per complete shape/device/dtype/tactic-set key, with
  single-flight miss handling and capture-safe cache hits;
- FlashInfer's eight SM12 CTA shapes crossed with `swap_ab` and
  static-persistent/Stream-K scheduling (32 tactics, in upstream order);
- the exact dense `gate_up_proj` execution topology those tactics serve in the
  oracle: one N-concatenated gate/up operand and output, one activation quant,
  and vLLM's maximum-of-logical-shards input/weight divisor semantics;
- the trace-promoted full-attention `qkv_proj` topology: concatenate the
  logical Q/K/V packed FP4 weights and linear block scales along N, derive one
  input/weight divisor and alpha from the logical-shard maxima, launch one
  `[M,5120] x [14336,5120]` GEMM, then split `[12288,1024,1024]` BF16 views;
- workspace high-water sizing, forced-tactic diagnostics, warmup and optional
  persistent plan-cache compatibility;
- BF16 gate-model output first, followed by the upstream FP16 output mode
  before the permanent kernel row can close;
- exact per-GEMM, real-model, trace, online latency/throughput and memory gates.

The first implementation checkpoint is deliberately narrower: correct the
bucket identity and single-flight behavior while retaining the current four
wide candidates. The second checkpoint ports the 32-tactic family. They use
separate same-binary toggles and separate A/Bs; a result cannot attribute a
stacked change to one lever.

Out of scope for this spike:

- W4A16 Marlin (`KERNEL-GEMM-MARLIN-W4A16`), FP8, MoE routing, attention,
  scheduler admission, HTTP transport, prefix caching, or new quant formats;
- GDN `qkvz_proj`/`ba_proj` fusion or any other projection merge not selected
  by the accepted node trace. Dense gate/up is W2 and full-attention packed
  QKV is the separately toggled W3-D checkpoint;
- changing FP4 quantization, block-scale layout, alpha semantics, sampling, or
  the 27B correctness contract to manufacture a speed win;
- running 35B performance while any 27B acceptance axis remains below vLLM.

Dispatch follows vLLM. On GB10, `FlashInferCuteDslNvFp4LinearKernel` rejects
SM12, the FlashInfer CUTLASS kernel wins the linear-kernel priority probe, and
its autotuner chooses from all valid tactics for a hybrid `M` bucket. A
missing/unsupported tactic is skipped during warmup; runtime precision never
falls back silently. The existing torch-free emulation path remains a
diagnostic/correctness reference, not the production performance arm.

## Trace-grounded problem

The exact pushed-`a531e05` online campaign is the binding before-state. It ran
cache-off, greedy, closed-loop 1024-input/128-output requests at c1/2/4/8/16/32
for three interleaved repetitions. All 2,016 requests, six memory returns, the
model gate and paired traces passed, but no concurrency passed all 20
performance axes.

Two independent observations localize the FP4 selection defect:

1. Local `NextPow2M` clamps to 16, so actual M=1/2/4/8/16 share one
   `(mp2,N,K)` cache entry. The ascending ladder tunes that entry at M=1 and
   reuses its tactic through c16. Three clean servers started directly at c16
   retuned M=16 and measured mean TPOT **161.747/161.719/161.729 ms**, versus
   **167.484 ms** in the standard ascending series and **161.698 ms** for vLLM.
2. The vLLM dependency trace contains the production tactics absent locally.
   Its dominant W4A4 kernels include CTA **128x32x256** with both
   `StreamKScheduler` (95,232 calls) and `StaticPersistentScheduler` (118,192
   calls). Local code exposes only four N>=128 persistent candidates and no
   `swap_ab` or Stream-K variants. Profiler percentages are not compared across
   the different trace mechanisms; kernel identity and call presence are the
   structural evidence.

Evidence root:
`~/work/vllm.cpp-online-gate/evidence/a531e055f0ef81b1d7296a7cba99d8f09373a265`.
Campaign/trace-status hashes are `24d78fbc…e9d2a` / `1c702ef9…142a`; ours
nsys/kernel hashes are `22d5a0f4…f247d1` / `ab7d0131…c0d6a3`; vLLM
trace/kernel hashes are `83fd0f41…d2a66` / `7056183f…cce417`.

The immutable HTTP-repair campaign at pushed `4e1d8ca` is the clean W1
before-state. All fixed c32 legs complete without an unread socket and the
separate fixed/legacy c32 A/B is steady-state-neutral at 0.999764×, while the
fresh exact total-throughput ratios remain **0.9661/0.9274/0.9378/0.9466/
0.9808/0.9910×** from c1 through c32. That removes transport starvation from
the sampled ladder without closing any concurrency's 20-axis gate, so W1 can
attribute its own exact-bucket/single-flight A/B against a healthy baseline.

### W1 measured classification

Pushed `bce2627` closes the W1 measurement loop without closing the row. The
same-binary exact/legacy c1/2/4/8/16/32 AB/BA/AB series completes 2,016 timed
requests and all six server lifecycles. Exact/legacy total-throughput ratios
are **1.000229/1.001222/1.000618/1.009320/0.999645/1.007172×**, with
**10/16/18/18/5/19 of 20** throughput+latency axes and **1/4** memory axes
passing. C8 and c32 are real wins, while c16 and memory make the iteration a
strict gate failure. Summary/artifact SHA-256 are `de20915a…6239` and
`b0f4b432…dceb`.

The paired exact/legacy nsys capture proves exact plans for M=1/2/4/8/16,
where legacy records only M=1→bucket16 for the small-M family. Across the whole
captured sequence, exact reduces FP4 kernel time **34,754.7→34,391.0 ms
(-1.05%)** and aggregate GPU-kernel time **107,121.5→105,159.8 ms (-1.83%)**.
Profiler throughput is 0.987334× and remains structural-only. Trace
summary/artifact SHA-256 are `d83826db…9c44` and `d7591e84…4811`.

The binding exact 27B campaign validates 12/12 groups, 2,016 requests, six
memory returns, the model gate and paired trace. Median total-throughput ratios
become **0.967983/0.931667/0.940305/0.951590/0.994440/1.007330×**, but only
**4/4/5/4/4/12 of 20** performance axes and **2/4** memory axes pass. Runs,
ratios and trace-status SHA-256 are `06a4bd7a…e41d`, `1e9643e9…c4b9` and
`ef9ce611…3a14`. The fresh oracle trace spends **25.1%** of profiled kernel
time in the missing 128x32x256 Stream-K/static-persistent pair (94,144 and
119,280 calls); ours remains dominated by 256x128x128 persistent. W2 is the
trace-grounded next iteration. 35B stays prohibited.

### W2 correctness localization (2026-07-12)

The complete 32-tactic CUDA build passes all forced raw-GEMM references, graph
replays, sanitizer shapes and a byte-for-byte raw comparison against
FlashInfer. Its first 27B default run nevertheless diverges at layer 1. A
full-layer replay and internal trace localize the first material error to the
dense MLP: attention, residual and post-attention norm agree; the gate/up MLP
output does not. Forced IDs 0/2/4/6 (swap-AB narrow tactics) fail the layer-1
replay, while IDs 1/3/5/7 (ordinary narrow tactics) pass its tolerance.

The end-to-end forced sweep makes the distinction stronger. IDs 1, 3 and 7
produce **16/16** oracle greedy tokens and **9/9** prefill argmax positions;
forced ID 5 preserves the required 6-token prefix but only 8/9 prefill
positions. Wider ordinary IDs are not a correctness substitute. Nsys then
matches the oracle's two dominant 128x32x256 signatures exactly to local IDs 4
and 6: vLLM really runs the swap-AB static/Stream-K kernels. Therefore selecting
an ordinary tactic would be a compensating numerical workaround, not parity.

The missing execution-chain link is the operand topology and global-scale
processing. vLLM constructs one `MergedColumnParallelLinear` for gate/up,
concatenates the two logical weights along N, and processes their scalar CT
divisors as arrays. It warns when the logical shards differ but deliberately
takes `max(weight_global_scale)` and `max(input_global_scale)`, reciprocates
those maxima, computes one alpha, quantizes the activation once and launches
one `[M,K] x [2I,K]` GEMM. Our current dense path launches two `[M,K] x [I,K]`
GEMMs with independent per-checkpoint divisors/alphas. The 27B checkpoint does
contain differing gate/up scalars, so the two programs are not numerically
equivalent even though the raw kernel port is exact.

W2 therefore includes the merged dense gate/up resident and exact maximum-scale
semantics, with `VT_FP4_MERGED_GATE_UP=0` preserving the split W2 arm and
`VT_FP4_FULL_TACTICS=0` preserving W1. The model test must prove default
16/16-vs-oracle before any performance claim. The accepted post-gate v0.25
node trace now identifies merged QKV as the next performance blocker, so W3-D
owns it with `VT_FP4_MERGED_QKV=0` preserving the three-GEMM arm.

The post-repair paired trace exposes one more execution-chain component inside
W2 rather than a new speculative lever. vLLM's production graph replaces the
`SiluAndMul([M,2I] -> [M,I])` plus dynamic NVFP4 quant pair with
`silu_and_mul_nvfp4_quant`; the exact 27B oracle trace contains
`vllm::silu_mul_cvt_fp16_to_fp4<__nv_bfloat16,false>`. Our merged branch still
materializes the BF16 `[M,I]` activation and then runs `ScaledFp4Quant`, even
though the older local fusion only accepts two separately contiguous `[M,I]`
operands and therefore cannot consume the merged `[M,2I]` buffer. W2 includes
the upstream one-input fusion with a dedicated
`VT_FP4_MERGED_SILU_QUANT=0` same-binary fallback. It must preserve the BF16
rounding boundary byte-for-byte and is gated independently before the full W2
component campaign.

### W2 binding classification (2026-07-12)

Clean pushed `b5c6e4fd65cdacea8f378e18ae101ebf521e8f01` completes the W2
campaign under one uncontended model-wide lock. All 12 engine/concurrency
groups, 2,016/2,016 timed requests, six memory returns, the commit-bound 16/16
model gate and paired trace status validate. Median total-throughput ratios
c1→c32 are **0.993275/0.951994/0.965716/0.976001/1.021341/1.021801×**;
performance-axis counts are **4/4/5/4/17/14 of 20**, and memory passes **2/4**.
Median normalized mean-TPOT ratios are
**0.991472/0.941745/0.947586/0.940429/0.982670/0.983680×**. W2 therefore
materially improves the old binding grid and closes total throughput at c16
and c32, but fails the strict every-axis gate.

The new execution trace proves this is a selection gap, not a missing tactic.
All 32 local candidates execute, yet the local profile is dominated by
128x128x128 static-persistent (16.33% of local kernel time) and
256x128x128 static-persistent (7.43%). The exact 128x32x256 family appears only
as a minor selected/tuning slice. vLLM instead spends **25.12%** of captured
kernel time in its 128x32x256 Stream-K/static-persistent pair. Cross-profiler
percentages are not directly compared as wall time; kernel identity and the
different dominant selection are the structural evidence.

Canonical runs/ratios/trace-status SHA-256 are `0056bf62…c5c59`,
`632e087b…192c` and `0190a7e1…ad3e`. Ours nsys/kernel hashes are
`f0599533…9e57` / `d2367ab4…392e`; vLLM trace/kernel hashes are
`db996f39…b41` / `caf8ac9f…258b`. The trace-run output digest is stable for
vLLM and differs across local HTTP repetitions; this remains diagnostic while
the separate commit-bound 16/16 model gate owns correctness. GPU processes are
empty and `/tmp/gpu` is reacquirable after exit. 35B was correctly not run.

### W3 runtime grounding and W3-A staging (2026-07-12)

The production oracle differs from the newer source-only persistent-cache path
in one important way. The installed pip-vLLM 0.24.0 runtime at
`~/venvs/vllm-oracle/lib/python3.12/site-packages/vllm/model_executor/warmup/kernel_warmup.py`
sets `_FLASHINFER_USE_PERSISTENT_CACHE = False`, with an explicit collision
warning for incomplete keys such as `use_8x4_sf_layout`. It therefore enters
`fi_utils.autotune()` in memory, runs one maximum-token dummy forward before
CUDA-graph capture, and lets FlashInfer generate every hybrid bucket up to
2,048. The clean W2 oracle log proves that exact path ran: FlashInfer tuned 16
profiles per real FP4 projection before server readiness. No
`autotune_configs.json` file is part of the production denominator.

Installed FlashInfer's FP4 `TuningConfig` leaves `use_cuda_graph` at its
default `false`. For every tactic it performs three warmups, synchronizes the
stream, launches TensorRT-LLM's one-thread `delayStreamKernel(1000us)`, then
records ten eager executions between CUDA events. Our W2 planner already used
three warmups and ten eager event repeats but omitted the pre-window
synchronize/delay. That is a concrete timing-method mismatch capable of
inverting near-tied tactic choices.

W3-A stages the exact one-thread nanosleep loop and stream boundary in
`cuda_matmul_nvfp4_cutlass.cu`; `VT_FP4_AUTOTUNE_DELAY=0` restores the W2 timing
method in the same binary, and `VT_FP4_AUTOTUNE_VERBOSE=1` reports the timing
protocol with each selected stable tactic ID. Immutable pushed `71f1e89`
clean-builds with CUDA 13.0.88, sm_121a and CUTLASS 4.5. Fresh delayed/off
focused processes each pass 14/14 cases and 18,619/18,619 assertions; both
native 27B arms pass 235/235 plus the full 16/16 oracle stream; delayed
memcheck passes 16,389/16,389 with zero errors. Delayed M=9 selects ID 6
128x32x256 Stream-K for output and merged gate/up and ID 4 static for Q; M=1
keeps the ID 6/4 family on output/gate-up. This closes W3-A correctness/safety
and proves the oracle family is selectable on real shapes.

### W3-A component classification (2026-07-12)

The exact immutable server binary then ran delayed versus
`VT_FP4_AUTOTUNE_DELAY=0` at c16, 96 requests, input 1,024 and output 128 in
delayed/off, off/delayed, delayed/off order under one lock. All 576 requests,
six memory returns and six verified cache evictions pass. Delayed runs are
810.084/811.018/808.693 total tok/s; off runs are
798.974/797.584/813.580. Means are **809.932/803.379 = 1.008156x**, with
0.118%/0.901% CV, but only **13/20 timing** and **2/4 memory** axes satisfy
strict no-regression. Delayed/off mean GPU peaks are 38,069/38,091 MiB and
available-memory drop improves, while delayed PSS/RSS are slightly higher.

The selection evidence is decisive: all processes share 35 plan keys, but only
**5/35** delayed keys retain one tactic ID across all three fresh processes
(off: 8/35). Paired delayed/off ID equality is 14/35, 6/35 and 11/35. The
narrow oracle family appears, but not stably. W3-A therefore receives no
standalone performance acceptance; its faithful timing default remains inside
the active W3 stack. Summary/driver/provenance/tree hashes are
`044bcf6e...e87fc`, `425f8521...e9ae`, `f5caa065...9915` and
`cf4c33c7...360c` under immutable `w3/component-ab`. Clean `b5c6e4f` remains
the binding vLLM denominator. The W3-B implementation below now owns the
pre-serve selection-stability repair and remains active on immutable evidence.
Versioned atomic persistence remains W3-C as an optional local
capability with collision-complete keys and stale rejection; it is not allowed
to stand in for, or silently become the default against, the production oracle
whose file cache is disabled.

### W3-B implementation checkpoint (2026-07-12)

The shared `LoadedEngine` now detects actual true-W4A4 weights rather than a
family name, opens `Nvfp4AutotuneWarmupScope`, and drives one synthetic request
whose prompt length equals the resolved maximum batched-token budget. This runs
before `async_engine()` can create `EngineCoreProc` and before the HTTP server
prints readiness. The request is removed completely from scheduler state. CPU,
non-W4A4, `VT_FP4_AUTOTUNE=0`, `VT_FP4_PLAN_CACHE=0`, or
`VT_FP4_PRE_SERVE_WARMUP=0` skip it.

For each W4A4 GEMM observed at maximum M, the dispatcher mirrors
`get_hybrid_num_tokens_buckets`: powers of two through 256, 256-token steps
through 2,048, 512-token steps through 4,096, then powers of two plus an exact
final maximum. It tunes missing keys against row prefixes of already-max-sized
buffers, returns the maximum-M plan for the real launch, and makes every ready
profile a capture-safe lookup. Completion is rejected unless a maximum-token
W4A4 GEMM was actually observed. A later unknown N/K/M remains legal but emits
a `lazy-miss after pre-serve warmup` diagnostic and increments the public
process-local counter.

CPU focused tests pass 3/3, including the actual-W4A4/BF16 capability split in
`tests/vllm/models/test_model_registry.cpp`. A disposable CUDA
13.0.88/sm_121a/CUTLASS-4.5
stage rooted at `precommit-w3b-195b475` passes exact and legacy focused
processes at **14/14 cases and 26,819/26,819 assertions** each. The full 27B
process passes **235/235 assertions + 16/16 tokens**, materializes **80/80**
profiles (16 buckets × five N/K shapes) into 80 entries and records zero lazy
misses. A separate HTTP process answers `/health` and `/v1/models`; completion
is line 3 and listening is line 6, again with zero misses. Model/server/models
log hashes are `96afc6ed…de401`, `2264a306…7e9e` and `80e10055…8315`.

This was deliberately a non-immutable implementation checkpoint. It granted no
performance credit and did not replace clean `b5c6e4f`; the immutable result
below supersedes its pending safety disposition.

### W3-B immutable correctness/safety checkpoint (2026-07-12)

Clean pushed `d7cdf66db0cfcc53d68d49613623ec6cd3807641` (tree
`24abb109…c488`) builds from a detached, clean source with CUDA 13.0.88,
sm_121a, CUTLASS 4.5 and vendored Triton AOT. Configure/build SHA-256 are
`50047004…e09e3` / `cac56085…fa248`. Registry and dense-loader contracts pass
**12/12 cases, 114/114 assertions** (one existing skip) and **4/4, 29/29**.
Fresh exact and legacy FP4 processes each pass **14/14 cases and
26,819/26,819 assertions**.

The fresh native 27B process passes **235/235 + 16/16**, materializes exactly
**80/80** profiles into 80 entries and reports zero lazy misses. Focused
compute-sanitizer memcheck passes **1/1, 24,586/24,586, zero errors**. A fresh
server answers `/health` and `/v1/models`; warmup completion is log line 3,
listening is line 6, and no lazy miss occurs. Model/memcheck/server log hashes
are `5ea053fe…b6475`, `2ef8f758…b124`, and `04d04fce…6951`;
manifest/provenance are `6f372fbe…89b1` / `1e8db7b7…e936` under
`~/work/vllm.cpp-nvfp4-small-m/d7cdf66…/w3b`. GPU inventories are empty and
the lock is free after the series.

The one-process profile list contains the oracle's narrow 128x32x256 family,
but several choices differ from the disposable process, which reinforces that
one startup cannot prove cross-process stability. This checkpoint therefore
closes immutable build/correctness/access-safety and placement only. The
repeated classification below supersedes its pending component disposition;
`b5c6e4f` remains binding and the 35B run remains prohibited.

### W3-B component and selection-stability classification (2026-07-12)

The same immutable `d7cdf66` server binary ran shipping prewarm versus
`VT_FP4_PRE_SERVE_WARMUP=0` lazy W3-A at c16, 96 requests, input 1,024 and
output 128 in prewarm/lazy, lazy/prewarm, prewarm/lazy order under one
uncontended lock. Every leg cold-evicts the model, corpus and binaries, samples
process/GPU/whole-system memory, archives selected plans, and proves post-exit
memory return. All **576/576** timed requests, six cache evictions and six
memory returns pass; GPU exits empty and `/tmp/gpu` is free.

Prewarm runs are **806.723/806.094/812.555 tok/s** and lazy runs are
**809.209/807.292/808.160 tok/s**. Means are
**808.457/808.220 = 1.000293×**, with **0.360%/0.097% CV**. Strict
no-regression passes only **15/20 timing** axes; failures are mean ITL, mean
TPOT, median TPOT, p90 E2E and p99 TPOT. Memory passes **2/4**: prewarm/lazy
mean PSS is **48,179,005/48,200,672 KiB** and RSS
**48,181,236/48,202,907 KiB**, but GPU peak is **38,048/37,615 MiB** and
available-memory drop **65,615,217/64,893,877 KiB**. Generated-text digests
differ diagnostically across these local runs, as in prior components; the
commit-bound **16/16** immutable model gate owns correctness.

Prewarm produces all **80/80/80** plan keys, while lazy production traffic
touches **35/40/40**. Across all three fresh processes only **20/80** prewarm
keys retain one tactic ID (lazy: **9/30**); paired prewarm/lazy equal IDs are
**13/35, 18/40 and 17/40**. Thus full bucket coverage does not stabilize the
near-tied event measurements. The separate untimed first-request preflight does
reproduce the placement benefit: prewarm/lazy mean first chunk is
**0.779/5.662 s**, and the complete 128-token request is
**14.929/20.249 s**. Shipping prewarm remains because it mirrors production and
moves first-use tuning before readiness, but W3-B receives no steady-state
speed credit and does not replace `b5c6e4f`.

Evidence is
`~/work/vllm.cpp-nvfp4-small-m/d7cdf66db0cfcc53d68d49613623ec6cd3807641/w3b/component-ab`.
Summary/selection/driver/provenance/tree SHA-256 are
`c371848a…cd11`, `fec3bf11…99c8`, `e996e6dd…662`,
`1df8bdbe…23cb` and `85910147…7b6`. The next binding step is a paired
prewarm/lazy nsys capture to compare actual kernel/tactic mix, then select the
next trace-grounded lever before another exact 27B oracle ladder. W3-C remains
optional and 35B remains prohibited.

The first three-arm trace attempt under `w3b/trace-ab-oracle` is **VOID**. The
prewarm server completed its separate warmup plus three retained c16/48 windows
(**144/144**, 810.245/810.860/808.760 tok/s), and wrote an nsys report, but the
post-exit lifecycle validator correctly stopped the series: newly created
client logs/results were nested under the cache-drop root, changing its stable
inventory from **50 to 58 files**. Lazy and vLLM never ran, so neither the
partial throughput nor the capture may be interpreted as a paired result.
Driver/report/tree SHA are `c1162d8f…2743`, `eb8a0996…1cc5c` and
`69a16a4d…b62f`; GPU returned idle and the lock is free. The next reproduction
moves mutable artifacts outside the corpus-only eviction root and writes a new
evidence directory rather than modifying this failed record.

## Upstream chain and execution dependency

### vLLM orchestration

- `vllm/model_executor/kernels/linear/__init__.py:407-420` orders CUDA NVFP4
  backends: CuTe DSL, FlashInfer CUTLASS, native CUTLASS, Marlin, other
  dependency paths, then emulation.
- `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:97-176` capability
  gates FlashInfer CUTLASS, swizzles/pads weights once, accepts pre-quantized
  activation+scale pairs, and invokes `flashinfer_scaled_fp4_mm(...,
  backend="cutlass")`.
- `vllm/model_executor/models/qwen3_5.py:278-288` maps checkpoint
  `gate_proj`/`up_proj` into one `gate_up_proj`; `qwen2_moe.py:75-115` builds a
  `MergedColumnParallelLinear`, launches it once and applies `SiluAndMul`.
- `vllm/model_executor/layers/linear.py:580-636,665-695` concatenates logical
  projection shards along the output dimension and loads scalar scales into
  the fused array.
- `compressed_tensors_w4a4_nvfp4.py:95-138` retains the fused scalar arrays
  through load, then uses the maximum weight divisor and maximum input divisor,
  reciprocates them and computes the one runtime alpha.
- `vllm/compilation/passes/fusion/act_quant_fusion.py:31-40,128-164` registers
  the NVFP4 activation-quant replacement and rewrites a one-input
  `SiluAndMul` followed by `scaled_fp4_quant` to the fused custom op when the
  CUDA symbol is present; `fix_functionalization.py:140-158` preserves its two
  mutated outputs through graph lowering.
- `csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu:27-116,120-163`
  loads the two contiguous halves of `[M,2I]`, computes SiLU and multiply in
  float, rounds through the input half type, and emits packed E2M1 plus the
  swizzled FP8 scale stream in one launch. The exact 27B oracle profile enables
  `fuse_act_quant=True` and executes
  `silu_mul_cvt_fp16_to_fp4<__nv_bfloat16,false>`; this runtime trace, not only
  the available source, makes the fusion part of the production comparison.
- `vllm/config/vllm.py:193-275,1192-1200` enables FlashInfer autotuning for
  optimization levels O1-O3; the production oracle resolves O3 and full/
  piecewise cudagraphs.
- The installed pip-vLLM 0.24.0
  `model_executor/warmup/kernel_warmup.py:123-154` sets
  `_FLASHINFER_USE_PERSISTENT_CACHE = False` because the file-cache key can
  collide, then tunes one maximum-token dummy forward in memory before CUDA
  graph capture. The clean oracle log records 16 generated FP4 profiles per
  projection. This installed file, not a newer source branch, owns the runtime
  denominator.
- The same file retains a disabled persistent branch, while
  `flashinfer_autotune_cache.py:19-41` fingerprints the configuration and
  chooses `autotune_configs.json`. That branch is design reference only until
  its collision warning is removed and the production oracle actually enables
  it.

### FlashInfer 0.6.12 execution owner

The installed dependency under
`~/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/` is the source of
truth for what actually ran:

- `gemm/gemm_base.py:5771-5816` gives FP4's dynamic activation-M dimension the
  hybrid bucket generator and uncapped mapping; scale/output dimensions remain
  constrained to the real M contract.
- `fused_moe/utils.py:212-307` defines the exact mapping: powers of two through
  256, 256-wide steps through 2048, 512-wide steps through 4096, then powers
  of two. Crucially, 1/2/4/8/16 remain distinct.
- `autotuner.py:1343-1426` performs three warmups, synchronizes the current
  stream, launches `delay_kernel(1000)` and measures ten eager iterations with
  events for FP4 (`TuningConfig.use_cuda_graph == false`); `:1428-1584`
  generates all bucket profiles and keys lookup by runner, mapped shapes and
  extras. `data/csrc/nv_internal/tensorrt_llm/kernels/delayStream.cu:24-34`
  owns the one-thread nanosleep loop that removes host timing bias.
- `gemm/gemm_base.py:1300-1340` exposes every tactic returned by
  `fp4_gemm_tactic_num()` and passes the selected integer to the compiled
  runner.
- `data/include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:47-147`
  dispatches DP versus Stream-K and `swap_ab`; `:187-220` enumerates eight
  tiles times two operand orientations times two schedulers.
- The same dependency's `fp4_gemm_template_sm120.h:99-195` constructs the raw
  argument/workspace/capture-safe launch; `:216-327` uses an explicit
  tensor-op/TMA epilogue, block-scaled cooperative mainloop,
  `StaticPersistentScheduler` or `StreamKScheduler`, and the pointer/dimension
  swap needed by `swap_ab`.
- `data/csrc/fp4_gemm_cutlass_sm120.cu:41-75,140-180` binds tactic IDs,
  high-water workspace and BF16/FP16 output to the raw runner.

The exact upstream tactic order is stable for parity and diagnostics:

1. tiles: 128x32x128, 128x32x256, 128x64x128, 128x64x256,
   128x128x128, 128x128x256, 256x128x128, 128x256x128;
2. within each tile: swap-AB DP, ordinary DP, swap-AB Stream-K, ordinary
   Stream-K.

The current local comment that narrow-N tiles matter only when matrix N is
small is rejected by the execution trace: tile N is kernel geometry, while
the Qwen projection N remains large. The oracle selects 128x32 tiles for the
real gate shapes.

## Our baseline

- `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:108-125` retains the two native
  vLLM fallback configs; `:127-165` adds two more wide persistent configs.
- `:217-283` already provides a torch-free raw pointer/shape/scale/alpha/
  workspace/stream launcher, so no PyTorch or FlashInfer runtime dependency is
  required.
- `:329-339` exposes only four candidates. The implementation uses
  `ElementC=OutType`, a block-scaled epilogue builder and persistent scheduling;
  it cannot instantiate the four narrow-N configurations.
- W1 adds `src/vt/cuda/nvfp4_plan_cache.h:26-213`: exact hybrid/legacy mapping,
  a `(bucket,N,K,device,SM,output-dtype,tactic-version)` key, and a per-key
  `tuning -> ready/failed` single-flight cache. Failures wake all waiters, erase
  partial state and permit retry; different keys progress independently.
- `cuda_matmul_nvfp4_cutlass.cu:358-483` defaults to exact buckets,
  `VT_FP4_EXACT_BUCKETS=0` restores the old alias, and a ready hit skips the
  capture query. An uncached captured call fails before events are created.
  Three warmups plus ten real-operand timing iterations remain unchanged.
- `:525-571` supports BF16 output and F32 through a BF16 staging/cast
  deviation. FP16, which FlashInfer supports, is absent.
- `tests/vt/test_ops_nvfp4_fp4.cpp:56-277` covers every hybrid boundary,
  legacy fallback, all key fields, 16-thread single-flight, independent keys,
  failure wake/retry and the pure capture predicate contract. The
  M=96/N=256/K=512 CUDA reference now also captures/replays a ready plan,
  rejects an uncached M=64 plan without invalidating capture, and proves an
  eager retry can tune without partial state. Release passes 100/100 and TSan
  passes 9 cases/615 assertions. Pushed `c8807b0` exact/legacy CUDA capture
  suites pass 10/10 and 18,333/18,333 each; focused memcheck passes 1/1 and
  16,389/16,389 with 0 errors; both 27B model arms pass 1/1 and 234/234.
  Every tactic, workspace growth, forced dispatch, merged gate/up scale
  semantics and broader real projection shapes remain open.

Real 27B W4A4 projection classes to benchmark include:

| Projection | M sweep | N | K |
|---|---:|---:|---:|
| full-attention Q split diagnostic | 1,2,4,8,16,32 | 12,288 | 5,120 |
| full-attention K/V split diagnostic | 1,2,4,8,16,32 | 1,024 | 5,120 |
| full-attention packed QKV production | 1,2,4,8,16,32 | 14,336 | 5,120 |
| attention/GDN output | 1,2,4,8,16,32 | 5,120 | 6,144 |
| dense gate/up split diagnostic | 1,2,4,8,16,32 | 17,408 | 5,120 |
| dense merged gate_up production | 1,2,4,8,16,32 | 34,816 | 5,120 |
| dense down | 1,2,4,8,16,32 | 5,120 | 17,408 |

## Port map

| Upstream concern | Local disposition |
|---|---|
| hybrid bucket generator/mapping (`fused_moe/utils.py:212-307`) | **W1 implemented:** pure helper maps 1/2/4/8/16 independently, preserves 256/2048/4096 phase boundaries, and keys device, architecture, output dtype, N, K, bucket and tactic-set version |
| profile/cache key (`autotuner.py:1438-1584`) | **W1 implemented:** one per-key state machine with condition-variable single-flight; waiters never retune, failures wake/erase/retry, and only complete plans publish |
| capture behavior | **W1 implemented/gated:** a ready lookup is allocation/sync-free; an uncached capture is rejected before tuning; pushed exact/legacy real CUDA capture/replay, invalid-miss teardown and eager-retry suites pass |
| 32 configs (`fp4_gemm_cutlass_template_sm120.h:47-220`) | port the dependency-owned raw template semantics into split local CUDA TUs: explicit TMA epilogue with `ElementC=void`, block-scaled cooperative mainloop, static persistent + Stream-K, both orientations and exact tactic order |
| `swap_ab` | swap A/B and their scale streams plus M/N exactly as FlashInfer, select the column-major epilogue form, and retain user-visible row-major `[M,N]` output |
| merged dense gate/up (`qwen3_5.py`, `qwen2_moe.py`, `linear.py`) | concatenate packed gate/up rows and linear FP8 block scales on device, swizzle the combined scale once, emit BF16 `[M,2I]`, apply `SiluAndMul`, and preserve a split diagnostic arm |
| full-attention packed QKV (`qwen3_5.py:279-288`, `qwen3_next.py:252-270,337-387`, `linear.py:942-1050`) | **W3-D implemented/staging-gated:** one resident packed FP4 Q/K/V operand and concatenated linear-scale stream, compressed-tensors logical-shard maxima, one quantization, one N=14,336 GEMM and row-strided Q/K/V BF16 views consumed without copies; `VT_FP4_MERGED_QKV=0` restores three GEMMs. Mutable staging correctness/safety passes; clean performance remains pending |
| fused CT globals (`compressed_tensors_w4a4_nvfp4.py:95-138`) | compute one input divisor as `max(gate,up)`, one weight multiplier as `1/max(1/gate.scale2,1/up.scale2)`, and one alpha as the product of the two reciprocals; test unequal logical-shard scalars explicitly |
| merged SiLU+NVFP4 quant (`act_quant_fusion.py`, `activation_nvfp4_quant_fusion_kernels.cu`) | add a backend-neutral one-input op over contiguous `[M,2I]`, with CPU composite fallback and a CUDA single-pass producer; wire only the merged true-W4A4 down-projection path, preserve BF16 RN, and retain `VT_FP4_MERGED_SILU_QUANT=0` |
| workspace (`fp4_gemm_template_sm120.h:151-195`) | compute the maximum required bytes across enabled tactics during warmup; acquire queue/device-scoped scratch before capture; no steady-state malloc/free and no undersized fallback |
| timing/warmup/persistence (`kernel_warmup.py:123-220`, `autotuner.py:1343-1426`) | **W3-A and W3-B classified; W3 remains active:** A retains three warmups + stream sync + 1,000-us GPU delay + ten eager repeats; `VT_FP4_AUTOTUNE_DELAY=0` restores W2. B adds exact profile enumeration (`nvfp4_plan_cache.h:53-87`), process scope/stats (`nvfp4_autotune.h:14-42`), all-bucket dispatch and diagnostic misses (`cuda_matmul_nvfp4_cutlass.cu:354-508`), plus actual-W4A4 shared-loader readiness ordering (`model_loader.cpp:211-260`). `VT_FP4_PRE_SERVE_WARMUP=0` restores lazy W3-A. Clean exact/legacy/model/memcheck/server evidence passes at 80/80 profiles, zero lazy misses, 16/16 correctness and zero sanitizer errors. Repeated prewarm/lazy component means are **808.457/808.220 tok/s = 1.000293×** with strict **15/20 timing + 2/4 memory** and only **20/80** stable prewarm IDs; first-use improves **5.662→0.779 s**, so production-faithful prewarm stays without speed credit. Immutable `9cc7191` closes the exact v0.25 grid but strict-fails at **54/124** axes. Replacement `def5f75` passes the node-level contract and promotes W3-D packed QKV from the exact **~240 versus 208 FP4 GEMMs/step** topology gap. W3-C adds collision-complete source/tactic/device/CUDA/CUTLASS/model keys plus atomic load/save/stale rejection as an optional capability; production pip-vLLM disables its file cache, so persistence is not the speed denominator |
| output modes | keep BF16/F32 gate behavior unchanged; add the upstream FP16 epilogue and tests as a separately gated breadth leaf before row closure |
| diagnostics | retain fixed-dispatch/autotune opt-out, add exact-bucket and full-tactic same-binary toggles, forced tactic ID and stable selected-plan reporting; invalid IDs/configs fail loudly |

No implementation copies FlashInfer's Python orchestration. The raw C++/CUDA
dispatcher and templates are Apache-2.0-compatible source ports with upstream
file/commit comments; local `vt::Tensor` validation replaces only tensor
wrappers.

## Tests to port

| Upstream executable specification | Local port/evidence |
|---|---|
| `tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py:26-168` | extend `tests/vt/test_ops_nvfp4_fp4.cpp` with M=1/2/3 and padded/non-padded shapes, BF16 first and FP16 at its breadth leaf, autotune off/on, scale-layout validation, forced tactics and dequantized-matmul tolerance |
| `tests/kernels/quantization/test_nvfp4_scaled_mm.py:17-100` | preserve the native fallback reference and all listed M/N/K shapes so full-tactic work cannot regress the existing two-config path |
| `tests/v1/determinism/test_nvfp4_batch_invariant_scaled_mm.py:31-101` | port as an explicit diagnostic mode in a fresh process; never silently make batch-invariant math the production default if it loses the vLLM performance floor |
| `tests/v1/determinism/test_nvfp4_batch_invariant.py:22-100` | retain the gate model's deterministic common-prefix contract across c1/2/4/8/16/32 and exact native token counts; record full generated text diagnostically because vLLM's own FP4 backends diverge after near ties |
| FlashInfer bucket helpers `fused_moe/utils.py:212-307` | **W1 ported/gated:** table tests for 0,1,2,3,4,8,16,255,256,257,2048,2049,4096,4097 and a bounded max; pushed `c8807b0` gates the real ready-hit/miss/retry capture in both exact and legacy modes |
| FlashInfer tactic enumeration `fp4_gemm_cutlass_template_sm120.h:187-220` | assert 32 stable tactic descriptors/order; force every supported tactic over representative small-M and real Qwen shapes; unsupported configs are reported/skipped only during tuning |
| `Qwen2MoeMLP` + `MergedColumnParallelLinear` (`qwen2_moe.py:75-115`, `linear.py:580-695`) | add a fused-vs-split CUDA gate/up probe over concatenated packed weights/scales, including unequal CT global divisors; pin the exact max-divisor/one-alpha contract and fused BF16 activation |
| `QKVParallelLinear` loader/forward (`qwen3_5.py:279-288`, `qwen3_next.py:252-270,337-387`, `linear.py:942-1050`) | **W3-D staging-gated:** unequal-shard scalar unit; packed-one-GEMM versus three logical CUDA outputs (max diff 0); row-strided preamble/cache tests; default and `VT_FP4_MERGED_QKV=0` real 27B gates (**235/235 + 16/16 each**); focused sanitizer zero errors. The clean nsys 240→208 launch assertion and immutable A/B remain pending |
| `tests/kernels/quantization/test_silu_mul_nvfp4_quant.py:16-73` | port BF16/F32-supported local cases as a byte-exact one-input fused-vs-`SiluAndMul(BF16)+ScaledFp4Quant` CUDA test, including decode, padded-M and real `I=17408` shapes; FP16 remains the declared W4 breadth leaf |
| `tests/compile/passes/test_silu_mul_quant_fusion.py:100-145` | the eager C++ model has no graph-rewrite pass, so gate the equivalent dispatch contract directly: merged true-W4A4 selects one fused producer by default, the env fallback restores two launches, and both preserve 16/16 oracle tokens |
| autotuner timing/cache behavior | **W1 ported/gated; W3-A/W3-B classified:** exact 2,048/4,096 bucket-list assertions plus maximum-M all-profile tuning and M32 capture/replay correctness live at `tests/vt/test_ops_nvfp4_fp4.cpp:92-100,882-922`; clean `d7cdf66` exact/legacy CUDA processes each pass 14/14 + 26,819/26,819; model/server prove 80/80 profiles before readiness with zero misses; memcheck passes 24,586/24,586 with zero errors. W3-B's repeated component proves 80 keys are present each startup but only 20 retain one tactic ID across all three. The corrected old-oracle trace puts FP4 kernel time within 0.63% of vLLM. The binding `9cc7191` v0.25 c1-c32 grid completes with **54/124** axes passing; accepted `def5f75` node tracing selects packed QKV, and stale disk-version/collision rejection belongs to W3-C |

The existing 27B and 35B real-model tests remain mandatory. The 27B test uses
the longest prefix on which vLLM production and emulation agree; it may not be
weakened or replaced with a self-golden. Per-GEMM numerical tolerance and
native token-count correctness are preconditions to all speed claims.

## Gates

1. **Record/build:** source comments cite vLLM and FlashInfer pins; clean CUDA
   13.0.88 `sm_121a` build, warning-as-error, record checker/mutations and
   documentation checkpoint pass. Split-TU compile time and binary-size deltas
   are recorded.
2. **Unit correctness:** bucket/single-flight/cache tests pass; all 32 tactic
   descriptors are present; every supported forced tactic matches the
   dequantized BF16 reference within the upstream tolerance. Merged gate/up
   uses one max-derived input divisor, weight multiplier and alpha and matches
   an explicit upstream-semantics reference with unequal logical-shard scales.
   The one-input fused producer is byte-exact to BF16 `SiluAndMul` followed by
   `ScaledFp4Quant` for decode, padded and real-I shapes. Existing native
   fallback and two-input fused quant tests remain green.
3. **CUDA safety/lifecycle:** compute-sanitizer covers small/padded/real shapes,
   every scheduler/orientation class, workspace growth/reuse and graph replay
   with zero errors. No process-exit scratch leak is added; queue/device
   ownership follows `BACKEND-ABI-VT` or records the existing pool debt.
4. **Real models:** 27B default and each diagnostic fallback pass the committed
   greedy common-prefix gate and exact output counts; 35B 16/16 remains
   unchanged for default/fallback because it does not dispatch this W4A4 GEMM.
5. **Component A/B:** each implementation checkpoint uses one binary and an
   interleaved AB/BA/AB series on the real projection shapes and c1-c32 online
   slices. Default must be reproducibly faster and no individual latency,
   throughput or memory axis may regress versus its fallback.
6. **Execution trace:** nsys ours and the vLLM oracle on the identical c16/48
   workload. The bucket checkpoint must show independent selected plans for
   M=1/2/4/8/16. The tactic checkpoint must show the oracle-selected
   narrow/static/Stream-K family where the same shape selects it; kernel lists,
   calls, workspace and launch behavior are archived.
7. **Binding 27B gate:** rerun the exact full c1/2/4/8/16/32, three-repetition,
   cache-off online campaign against fresh production vLLM after each accepted
   checkpoint. Ours must be no worse on every throughput, TTFT, TPOT/ITL,
   E2E and memory axis. A component win does not waive a red end-to-end axis.
8. **35B/final:** only after every 27B axis passes, run the exact 35B campaign
   and both-model direct-library/online/trace/memory gates. The row remains
   open for FP16/cross-architecture breadth even if the order-0 GB10 gate
   passes.

Every GPU series holds one uncontended `/tmp/gpu` lock for all arms and traces.
The `4e1d8ca` HTTP campaign has completed and releases its immutable build
before any new FP4 GPU command begins.

## Dependencies

- `SERVE-GATE-ONLINE` owns the exact corpus, commands, validation and fresh
  vLLM denominator. Prefix caching stays off and max-seqs/token budget/model
  length/admission/sampling remain unchanged.
- `SERVE-ASYNC-LLM` HTTP capacity is measured first so transport stalls cannot
  contaminate the FP4 A/B. The FP4 source does not alter HTTP or scheduling.
- CUDA 13.0.88, GB10/sm_121a, CUTLASS 4.5.0 and enough build storage for 32
  heavy instantiations are required. SM120 cross-build/runtime becomes a
  backend follow-up; other architectures keep their current dispatch.
- `BACKEND-ABI-VT` defines the future common workspace lifecycle. This repair
  may use the current raw adapter, but it may not invent a conflicting resource
  contract or worsen known teardown debt.
- The Qwen3.6-27B snapshot and exact vLLM/FlashInfer environment are immutable
  benchmark inputs. FlashInfer source is inspected from the installed package;
  no runtime Python dependency is introduced.

## Work breakdown

| Work | Deliverable | State / gate |
|---|---|---|
| W0 | accepted source+trace spike, exact upstream test inventory and before-state | complete in this documentation checkpoint; no runtime result |
| W1 | exact hybrid bucket identity plus complete key and per-key single-flight/capture-miss contract; `VT_FP4_EXACT_BUCKETS=0` restores the aliased baseline | **measured complete, acceptance fail:** all safety/correctness gates pass; component is positive at c8/c32 but fails c16/memory; exact oracle improves yet remains below every-axis floor. Evidence and hashes are in “W1 measured classification” |
| W2 | port exact 8-tile x 2-orientation x 2-scheduler template family and high-water workspace; stable forced IDs; mirror merged dense gate/up plus maximum logical-shard CT divisors; port the traced one-input SiLU+NVFP4-quant producer; `VT_FP4_MERGED_SILU_QUANT=0` restores materialized activation+quant, `VT_FP4_MERGED_GATE_UP=0` restores split W2 and `VT_FP4_FULL_TACTICS=0` restores four-candidate W1 | **measured complete, acceptance fail:** implementation/correctness/safety gates are green; clean `b5c6e4f` improves every concurrency and wins c16/c32 total throughput, but exact ratios/axes/memory remain below the strict floor. Trace proves all tactics exist and promotes selection parity to W3 |
| W3 | A: production-FlashInfer eager timing; B: pre-serve all-bucket in-memory warmup; C: optional versioned persistent plan cache; D: trace-promoted full-attention packed QKV resident/one-GEMM/split-view path with `VT_FP4_MERGED_QKV=0` | **ACTIVE** under `CLAIM-NVFP4-SMALL-M-3`. A/B classifications and `9cc7191`'s exact grid are complete; only 54/124 axes pass. D is implemented and staging correctness/access-safety green: packed/logical max diff 0, row-strided CUDA suites pass, focused sanitizer zero errors, default/split 235/235 +16/16 each, and prewarm shapes reduce 80→64. Clean pushed-SHA A/B and 240→208 re-trace are next; C stays optional |
| W4 | FP16 output, SM120 cross-target, permanent evidence/anchors and final row closure | after order-0 BF16 parity; no broad `DONE` until all declared modes/backends are gated |

W1 and W2 are intentionally separate performance iterations. W3 cannot be
folded into their timed arms because tuning placement changes startup and
first-request behavior. W4 breadth cannot delay an order-0 BF16 speed repair,
but its open state remains visible in the kernel matrix.

## W2 implementation and component classification (2026-07-12)

The W2 staging build uses CUDA 13.0.88, `sm_121a` and CUTLASS 4.5. The full
32-tactic surface is split across bounded translation units and addressed by
stable IDs. Dense gate/up weights and scale streams remain resident in their
merged form, use the maximum logical-shard input/weight divisors and one alpha,
and feed a backend-neutral `SiluAndMulFp4Quant` op. Its CPU implementation is
the composed oracle; CUDA consumes contiguous `[M,2I]` and writes packed FP4
plus the swizzled scale stream in one pass. The three environment fallbacks in
the W2 row isolate the fusion, merged topology and full tactic family without a
second binary.

Focused CPU tests pass **12/12 cases and 885/885 assertions**. The final staged
CUDA NVFP4 slice passes **14/14 and 18,619/18,619**; a focused
compute-sanitizer run passes **1/1 and 16/16** with zero errors and zero leaks.
The one-input producer is byte-exact to BF16 `SiluAndMul` followed by
`ScaledFp4Quant` over F32/BF16 decode, padded and real `I=17408` shapes. The
real dense gate passes **9/9 prefill argmax positions and 16/16 greedy tokens**.
The paged shipping and `VT_FP4_MERGED_SILU_QUANT=0` arms each pass **235/235**
and **16/16 tokens**. Final staged ops/op-parity/paged binary SHA are
`36779505…e9786` / `338a059b…dbc4` / `dc90e5fa…3546`; ops/sanitizer/dense/paged
log SHA are `738d15a5…fcbe` / `61b23535…911b` / `3d2b984f…6595` /
`04a3c872…d29c1`, and GPU process snapshots are empty. Evidence lives at
`~/work/vllm.cpp-nvfp4-small-m/debug/{merged-silu-quant,final-staging}-20260712`.

The full native stream also depends on the separately owned
`KERNEL-GDN-AOT-BF16` correction: vLLM stores GDN core/z at the BF16 model
dtype, while the former local f32 default takes the known near-tie branch once
W2 tactics are active. The existing vendored BF16 path is now default only for
the dense 27B; all fusion A/B arms hold it constant. Its own
`VT_GDN_OUT_BF16=0` component evidence and open gates remain in the kernel
matrix/ledger rather than being attributed to FP4.

The unprofiled cache-off c16/96-request AB/BA/AB series runs the shipping fused
arm at 815.625/812.912/800.256 tok/s and the fallback at
792.337/798.232/801.833 tok/s: means **809.597/797.467 = 1.015211×** with
0.827%/0.491% CV. It passes **17/20** timing axes and **0/4** sampled memory
axes, although all six processes return memory. The three misses are p90 E2E,
p90 TPOT and p99 TPOT; the memory means are sensitive to one 354-MiB-low
fallback GPU sample, so they remain failures rather than being normalized away.
Summary/driver SHA-256 are `cb5e5204…b0ab89` and `3cda0d4c…d7bb0` under
`debug/merged-silu-quant-c16-ab-20260712`.

The bounded paired trace records fused/fallback means
**817.020/800.338 = 1.020843×** and 20/20 timing axes. The fused producer runs
8,557 times for 4.802 s; fallback materializes SiLU 8,390 times for 7.054 s and
adds 8,013 quant calls accounting for 2.643 s of the fallback-only quant time.
That is roughly 4.9 s less boundary-kernel time and about 8,000 fewer launches
in the full capture. Summary SHA-256 is `9933724b…a1318`; fused/fallback nsys
hashes are `094615a1…be22c` and `5efe621c…0a080` under
`debug/merged-silu-quant-trace-20260712`. These trace timings own structural
attribution only. Fresh server autotuning can select different near-tied valid
tactics and generated suffixes; correctness remains the exact op and committed
native model gates, never generated-text equality between independent tuning
sessions.

That component checkpoint did not close W2. The clean `b5c6e4f` campaign is now
historical after the oracle upgrade. W3-A and W3-B same-binary selection A/Bs
are classified and strict-failed, and W3's corrected old-oracle paired trace is
clean and structurally classified. Immutable `9cc7191` now owns the binding
c1/2/4/8/16/32 three-repetition v0.25 27B oracle result: **54/124** axes pass.
Accepted `def5f75` node tracing selects W3-D packed QKV as the next
execution-grounded repair. Keep 35B held until all 27B throughput, latency and
memory axes pass.

### W3-B corrected paired-trace classification (2026-07-12)

Corrected attempt 2 uses a new immutable corpus-only cache root and writes all
mutable clients outside it. One uncontended `/tmp/gpu` lock spans shipping
prewarm, `VT_FP4_PRE_SERVE_WARMUP=0` lazy W3-A, and pinned vLLM. Each arm starts
clean, runs a separate warmup, then keeps only three steady-state c16/48
input-1,024→output-128 ranges with interactive Nsight start/stop. All
**432/432** retained requests pass. All six before/after inventories remain
exactly **49 files** with digest `b1789458…7523`; all three memory returns pass,
the GPU exits empty, and no Nsight session remains.

The SQLite kernel records close the structural W3 question. Prewarm/lazy/vLLM
FP4 GEMM sums are **110.623/114.229/109.932 s**. Prewarm is **3.157%** lower
than lazy and only **0.629%** above vLLM. The 128x32x256 pair is prewarm
**70.333 s / 218,434 calls** versus vLLM **70.986 s / 220,465 calls**. The
remaining scheduler split differs: prewarm spends **65.800 s** in Stream-K and
**4.533 s** in static-persistent, versus vLLM **43.259/27.728 s**. Lazy still
executes all sampled alternative signatures plus **480 / 0.492 s** delay
kernels inside the retained range; prewarm executes no retained delay kernel.
W3-B therefore fixes the original wide-tactic dominance and first-use tuning
contamination, but not cross-process ID stability or exact scheduler mix.

The node-traced client rates are deliberately diagnostic. Prewarm/lazy/vLLM
means are **804.860/810.250/798.324 tok/s** with total-throughput CV
**0.187/0.150/0.012%**. Prewarm is 1.0082x on total throughput and 1.3646x on
normalized TTFT, but only **0.9673x** on normalized mean TPOT/ITL. CUDA-graph
node tracing perturbs scheduling, so none of these rates replaces the clean
full-grid denominator. Evidence is
`~/work/vllm.cpp-nvfp4-small-m/d7cdf66…/w3b/trace-ab-oracle-r2`; tree/driver/
provenance SHA are `2aab1197…a137` / `af29681e…22fbf` /
`dff465b3…bbe3`, and prewarm/lazy/vLLM report SHA are
`a73d6032…1194a` / `9d74b6c8…37ea2` / `f89ffd4a…2d1b8`.

The clean pushed-`3cc490c` exact W3-B c1-c32 attempt is **VOID**. It was stopped
at 28/36 groups, 1,602/2,016 timed requests, four memory returns and no paired
trace after proving that the DGX oracle was vLLM 0.24.0 + FlashInfer 0.6.12,
while the audited v0.25.0 target requires FlashInfer 0.6.13. The owned remote
process group was terminated and GPU/ports/lock were verified idle; no partial
rate binds. The isolated v0.25.0 oracle is validated and atomically active with
rollback preserved. Immutable `9cc7191` has now run the full exact 27B grid and
70/124 axes remain below floor. Its ours trace omitted CUDA-graph child nodes;
the replacement trace below closes that attribution gap. W3-C stays optional
and 35B remains prohibited until every 27B axis passes.

The replacement exact campaign is **complete but failed/open** from immutable clean
`9cc71918dbdc10f014c02feb9bab1d00963a16fe`. Its detached source, fresh sm_121a
server/model-gate build, vLLM 0.25.0 oracle fingerprint and exact c1-c32
source/vLLM corpus views are prepared and hashed under
`~/work/vllm.cpp-online-gate/evidence/9cc71918dbdc10f014c02feb9bab1d00963a16fe`.
One no-GPU recorder command failed before output because it omitted the repo
module path; the corrected module invocation passes. The locked model gate,
36/36 timed groups, 2,016/2,016 requests and six returns pass their evidence
contracts. All 124 axes bind; **54 pass and 70 fail**. Total
ratios c1→c32 are **0.9901/0.9491/0.9633/0.9770/1.0288/1.0467×** and host
PSS/RSS normalize to **0.5855/0.5934×**. Summary hashes are `c46595b8…a894` /
`231ec9fd…7591`. The original trace status/kernel-summary hashes are
`f38b149d…d17`, `8bba1bb1…8f4` and `80999085…d2`, but ours has 1,226 whole-
graph activities and zero graph-node rows. The new contract requires
`--cuda-graph-trace=node` and adds a trace-only one-lock path; accepted
`def5f75` below satisfies it.

### W3-D v0.25 node-trace promotion (2026-07-12)

Immutable `def5f752896036d9b35841a278578fd812f75a0d` runs the exact 27B model
gate plus both c16/48x3 input-1,024/output-128 profilers under one uncontended
lock. The model gate passes in 44.79 seconds; cache inventories and memory
return pass; the GPU, port 8001 and `/tmp/gpu` are idle after exit. Trace
`status.json` is `passed:true` with `cuda_graph_trace=node`. Its SHA-256 is
`c5a07125…11f4`; ours nsys/kernel-summary/SQLite hashes are
`71af83c5…1a36` / `42916a72…36e3` / `7c8aadd2…eae5`; vLLM trace/kernel JSON
hashes are `8c4a267e…4291` / `e4b2d8fe…6a90`.

Ours contains **2,315,412** graph-child kernel rows, **272,354** eager rows and
7,711 distinct graph node IDs. The steady graph executes about **240** FP4
GEMMs per forward: 343,461 graph-child FP4 GEMMs divided by 1,430 graph
`lm_head` markers is 240.182, with the fractional excess attributable to
capture/warmup. vLLM executes exactly **208** FP4 GEMMs per forward: 330,304
divided by 1,588 `ArgMax` step markers. The Qwen3.6-27B topology proves the
difference rather than inferring it from percentages: 64 merged gate-up/down
pairs contribute 128, 16 packed full-attention QKV/output pairs contribute 32,
and 48 GDN outputs contribute 48, totaling 208. Local separate Q/K/V adds two
launches across each of 16 full-attention layers, totaling 240.

This trace promotes exactly one W3-D change: mirror vLLM's
`QKVParallelLinear` loader and forward path. Construct one resident N=14,336
packed FP4 weight and concatenated linear-scale stream from logical
Q/K/V=`12,288/1,024/1,024`; use the compressed-tensors maximum logical-shard
input/weight divisors to derive one quantization and alpha; launch one GEMM;
then expose three non-copying BF16 views. `VT_FP4_MERGED_QKV=0` preserves the
current three-GEMM reference. Unit, CUDA/reference, exact default/fallback 27B,
access-safety and trace-launch-count gates precede an unprofiled same-binary
A/B. The trace earns no speed claim; the binding timing/memory result remains
`9cc7191` at 54/124 axes until that A/B and a fresh full grid say otherwise.

### W3-D implementation/correctness staging checkpoint (2026-07-12)

W3-D now mirrors the whole selected upstream chain. `FullAttnLayerWeights`
retains logical host Q/K/V and two combined device owners. First CUDA use
concatenates packed rows and linear block scales at N=14,336, swizzles the
combined scale once, and computes one input divisor, weight reciprocal and
alpha from the three logical maxima. The forward quantizes once and launches
one BF16 CUTLASS GEMM. Its Q/K/V slices preserve the parent row stride, so
`AttnQkNormRopeGate`, `CastF32` and `ReshapeAndCache` now admit packed inner-
contiguous rows and use the explicit token stride. No split-copy kernel is
introduced. The path requires the already-default fused preamble; an explicit
preamble opt-out or `VT_FP4_MERGED_QKV=0` retains the fully contiguous split
reference. Non-W4A4 and 35B paths are unchanged.

Local CPU build and focused suites pass. Fresh sm_121a CUDA staging at
`~/work/vllm.cpp-packed-qkv-staging` builds the server plus all focused/model
targets. The packed GEMM equals all three logical BF16 outputs with max
absolute difference **0**. CUDA preamble passes **2/2 cases, 14/14 assertions**;
reshape/cache passes **12/12, 4,290/4,290**. Focused packed-GEMM, preamble and
packed-cache compute-sanitizer processes report **zero errors**. The real 27B
default and `VT_FP4_MERGED_QKV=0` processes each pass **235/235 assertions and
16/16 oracle tokens**; default/split prewarm materializes **64/80** profiles.
Default/fallback model-log SHA-256 are `f2215483…dc8d` / `594c3bf7…11fb`;
server and model-test binary hashes are `203994df…4c1c` / `319e6c38…e541`.

This evidence is deliberately non-binding mutable staging. It closes
implementation, component correctness, model correctness and focused access
safety only. It publishes no speed, memory or launch-count result. Commit and
push this checkpoint, clean-build that immutable SHA, then run one lock-held
packed/split same-binary A/B and a node re-trace before updating the binding
54/124 disposition or selecting another lever. 35B stays prohibited.

## Risks and decisions

- **Compile/binary growth:** 32 heavy CUTLASS instantiations can exhaust a
  monolithic nvcc process. Split by tile/scheduler into bounded TUs and record
  build time/object size. Do not prune a traced tactic merely to shorten build.
- **Narrow-N construction:** the current collective builder fails N=32/64
  scale-factor TMA checks. Port FlashInfer's explicit tensor-op epilogue and
  cooperative block-scaled mainloop; do not approximate those shapes with the
  existing builder and then claim tactic parity.
- **Swap-AB correctness:** pointer/scale/M/N swapping changes internal layout
  but not the public `[M,N]` contract. Forced-tactic tests must catch transpose
  or scale-stream mistakes before tuning can select the path.
- **Compensating-tactic trap:** ordinary narrow tactics happen to recover the
  current 27B tokens, but the production trace proves vLLM runs swap IDs 4/6.
  Do not filter swap tactics to hide the split-projection scale mismatch; port
  the merged topology and maximum-divisor semantics and keep the oracle's
  runtime selection.
- **Stream-K workspace:** query the maximum across every enabled tactic and
  reserve it before capture. An allocation, event sync, or plan mutation during
  graph replay is a gate failure.
- **Autotune races:** W1 single-flight now chooses once; failures wake waiters,
  erase partial state and permit retry. CUDA concurrency/capture execution must
  still prove the host contract before the risk is considered gated.
- **Persistent-cache staleness:** include tactic ABI, source revision, device
  capability, CUDA/CUTLASS versions, output dtype and relevant model geometry in
  the fingerprint; reject rather than reinterpret old tactic integers.
- **FP4 near ties:** different valid reduction schedules can change later
  generated tokens. Correctness uses the oracle-common deterministic prefix,
  per-GEMM error bounds and exact native counts. Speed may never be purchased
  by weakening those gates or switching sampling.
- **Attribution:** nsys/torch-profiler timings are structurally informative but
  differently perturbed. Unprofiled interleaved A/B owns performance; traces
  own actual dispatch identity. Re-profile and re-rank after every checkpoint.
