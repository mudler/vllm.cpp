# Spike: FlashInfer-parity NVFP4 small-M dispatch and SM12 tactics

**Row:** `KERNEL-GEMM-NVFP4-W4A4` · **state:** accepted implementation
spike; W1/W2 are measured and W3 is `ACTIVE` under
`CLAIM-NVFP4-SMALL-M-3`; W3-A delayed event timing is immutable
correctness/safety-green and performance-pending.
**Pins:** vLLM
`e24d1b24fe96`, pip-vLLM `0.24.0`, installed FlashInfer `0.6.12`, CUTLASS
`v4.5.0`, CUDA `13.0.88`, and the
Qwen3.6-27B-NVFP4 gate snapshot recorded in `environment.md`.

This accepted spike is the mandatory contract for FP4 plan-cache and kernel
changes. W1 retains its CPU/TSan, exact/legacy sm_121a capture, focused
memcheck and both-27B-model proof. Pushed `bce2627` completes its component,
trace and exact oracle classification: it is structurally effective and
materially improves the 27B grid, but does not pass the strict component or
every-axis gates. The fresh runtime trace promoted the separate 32-tactic W2
family; W2 now implements and correctness-gates that family, merged CT gate/up
semantics and the traced one-input activation/quant producer. Its component is
positive but below the strict timing/memory floor. Clean pushed `b5c6e4f`
completes W2's binding classification: it improves every concurrency and wins
c16/c32 total throughput, but c1-c8, TPOT/ITL and host-memory axes remain red.
The paired trace promotes W3 because local runtime selection is dominated by
wide tactics while vLLM resolves the 128x32x256 Stream-K/static pair. Exact
`4e1d8ca` remains W1's immutable before-state.

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
- merged QKV topology. It remains an independently gateable later lever; the
  merged dense gate/up topology is now in W2 because the forced-tactic
  correctness investigation proved it is part of the oracle's actual FP4
  dispatch contract, not an optional fusion;
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
16/16-vs-oracle before any performance claim. Merged QKV remains outside this
iteration unless the post-gate trace identifies it as the next correctness or
performance blocker.

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
and proves the oracle family is selectable on real shapes, but not performance:
the real-model AB/BA/AB and selection stability remain `PENDING`. Pre-serve all-bucket in-memory
warmup is W3-B. Versioned atomic persistence remains W3-C as an optional local
capability with collision-complete keys and stale rejection; it is not allowed
to stand in for, or silently become the default against, the production oracle
whose file cache is disabled.

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
| full-attention Q | 1,2,4,8,16,32 | 12,288 | 5,120 |
| full-attention K/V | 1,2,4,8,16,32 | 1,024 | 5,120 |
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
| fused CT globals (`compressed_tensors_w4a4_nvfp4.py:95-138`) | compute one input divisor as `max(gate,up)`, one weight multiplier as `1/max(1/gate.scale2,1/up.scale2)`, and one alpha as the product of the two reciprocals; test unequal logical-shard scalars explicitly |
| merged SiLU+NVFP4 quant (`act_quant_fusion.py`, `activation_nvfp4_quant_fusion_kernels.cu`) | add a backend-neutral one-input op over contiguous `[M,2I]`, with CPU composite fallback and a CUDA single-pass producer; wire only the merged true-W4A4 down-projection path, preserve BF16 RN, and retain `VT_FP4_MERGED_SILU_QUANT=0` |
| workspace (`fp4_gemm_template_sm120.h:151-195`) | compute the maximum required bytes across enabled tactics during warmup; acquire queue/device-scoped scratch before capture; no steady-state malloc/free and no undersized fallback |
| timing/warmup/persistence (`kernel_warmup.py:123-220`, `autotuner.py:1343-1426`) | **W3-A immutable correctness/safety green; performance pending:** three warmups + pre-window stream sync + exact 1,000-us GPU delay + ten eager event repeats; `VT_FP4_AUTOTUNE_DELAY=0` restores W2 and verbose output records the selected stable ID. `71f1e89` passes focused/model/memcheck gates and resolves the traced ID 6/4 narrow family on real hot shapes. W3-B tunes every configured hybrid bucket before server readiness and leaves lazy misses diagnostic-only. W3-C adds collision-complete source/tactic/device/CUDA/CUTLASS/model keys plus atomic load/save/stale rejection as an optional capability; production pip-vLLM disables its file cache, so persistence is not the speed denominator |
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
| `tests/kernels/quantization/test_silu_mul_nvfp4_quant.py:16-73` | port BF16/F32-supported local cases as a byte-exact one-input fused-vs-`SiluAndMul(BF16)+ScaledFp4Quant` CUDA test, including decode, padded-M and real `I=17408` shapes; FP16 remains the declared W4 breadth leaf |
| `tests/compile/passes/test_silu_mul_quant_fusion.py:100-145` | the eager C++ model has no graph-rewrite pass, so gate the equivalent dispatch contract directly: merged true-W4A4 selects one fused producer by default, the env fallback restores two launches, and both preserve 16/16 oracle tokens |
| autotuner timing/cache behavior | **W1 ported/gated:** 16-thread same-key one-pass, different-key progress, failure wake/retry/no-partial-state, uncached-capture rejection and ready-hit bypass; pushed real CUDA capture/replay/miss/teardown/retry plus focused memcheck pass. **W3-A immutable correctness/safety green:** fresh delayed/off focused and 27B processes preserve the output/capture contract; delayed memcheck is clean and verbose evidence records real-shape plans. Performance/selection-stability A/B remains pending. Pre-serve bucket coverage belongs to W3-B; stale disk-version/collision rejection belongs to W3-C |

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
| W3 | A: production-FlashInfer eager timing (3 warmups, sync, 1-ms GPU delay, 10 repeats) and selected-plan evidence; B: pre-serve all-bucket in-memory warmup with lazy diagnostic fallback; C: optional versioned persistent plan cache with collision-complete key, atomic load/save/stale rejection and startup/memory evidence | **ACTIVE** under `CLAIM-NVFP4-SMALL-M-3`. W3-A at immutable `71f1e89` is clean-build, focused/model correctness and memcheck green; delayed real-model plans include the traced ID 6/4 narrow family. Same-binary real-model performance/selection-stability A/B remains `PENDING`, so A is not accepted yet. Production pip-vLLM disables its file cache, so W3-C is separately gated and cannot own the oracle speed comparison. After A/B/C, run the paired trace and exact 27B because runtime selection changes |
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

That component checkpoint did not close W2. The subsequent clean `b5c6e4f`
campaign above is now the binding denominator and confirms a strict acceptance
failure with a trace-grounded selection mismatch. W3 owns the next same-binary
selection A/B and exact c1/2/4/8/16/32 three-repetition 27B oracle campaign.
Keep 35B held until all 27B throughput, latency and memory axes pass.

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
