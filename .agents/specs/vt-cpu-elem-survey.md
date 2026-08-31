# SPEC — `VT-CPU-ELEM-SURVEY`: which of the other 62 CPU kernels the per-element dtype dispatch actually costs

Issue: [#2416](https://github.com/mudler/vllm.cpp/issues/2416).
Predecessor: [`.agents/specs/vt-cpu-elem-dispatch.md`](vt-cpu-elem-dispatch.md),
whose `## Owed` names this work, states the sizing method, and leaves
[#2376](https://github.com/mudler/vllm.cpp/issues/2376) open against it.

## The gap

`VT-CPU-ELEM-DISPATCH` hoisted the loop-invariant dtype dispatch out of
`AttentionKernel` and `AttentionCrossKernel` and measured 8.75x-11.16x for the
pair. It hoisted two of the 64 functions that call `vt::cpu::LoadF32` /
`StoreF32`. The other 62 still resolve a dtype per element, and its `## Owed`
gives the method for sizing them verbatim:

> for each candidate, `perf record -e cpu-clock` over a probe that runs THAT op
> alone at a shape a shipped model actually uses, and read `LoadF32`'s self
> percentage. Above ~30% the hoist is worth a row; below it the kernel is bound
> by something else and a hoist buys the fraction the profile names. **Do not
> sweep**: each hoist needs its own byte-equality gate.

Nobody has run it. Without the ranking the only ways forward are a 62-kernel
sweep nobody can review, or a guess about which kernel is next.

**The call sites, counted on this branch's base `43553262c`, and this CORRECTS a
number that was relayed to the developer.** The figure in circulation was "219
sites across 64 kernels in two files", carried forward from the predecessor row.
It is wrong on the count and wrong on the file list. The count is **271 calls
across 67 enclosing functions, all of them in `src/vt/cpu/cpu_ops.cpp`**, and
**`src/vt/cpu/cpu_paged_attn.cpp` contains none** — the five matches there are its
own already-hoisted `KvKind` resolver and its comments, so the predecessor row's
"two files" is one file. Of the 67, five are the shared helpers rather than
kernels (`LoadF32`, `StoreF32`, `AttnResolveOrRefuse`, `FusedLoad`, `FusedStore`)
and two are the pair already hoisted.

The correction is stated rather than quietly substituted because a number that has
been repeated to a human is a claim, and replacing it in silence leaves the wrong
one standing wherever it was repeated. The method is reproducible: attribute every
`LoadF32`/`StoreF32` call expression to the top-level definition it sits inside,
over both files, at that SHA.

**A related negative, verified rather than assumed.** `LTX25-AUDIO-DECODE-COST`
found `LoadF32`, `StoreF32` and `SizeOf` appear ZERO times in
`src/vllm/model_executor/models/ltx2_audio_vae.cpp`. That row's serial-`Conv2d`
defect is the same SHAPE as this one and is NOT this lever, so the audio VAE is
outside this ranking on evidence rather than by omission.

## Scope

IN scope:

- **W1** — a committed probe that runs ONE op alone at a grounded shape, and the
  ranked `LoadF32` self-percentage table over it. **The ranking is this row's
  product**: it lets every future row prioritize by measurement instead of
  guessing, and it is what stops the remaining work becoming a 62-kernel sweep.
- **W2** — hoist ONLY what the ranking justifies, each with its own byte-equality
  gate. One kernel here; see `## Outcome` for why that one and not the next.
- **W3** — the `__attribute__((always_inline))` verdict, with the i-cache
  evidence the predecessor row said it lacked, either way.

OUT of scope, declared rather than approximated:

- **Hoisting the rest of the ranked set.** The measurement says roughly half the
  measured kernels clear the bar. That is a finding, not a licence: each hoist
  needs its own gate and its own mutations, and doing them in one change is the
  sweep the predecessor row forbade. They are in `## Owed`, ranked.
- **Changing any output value.** Byte equality is the bar. A kernel that cannot
  be hoisted bit-exactly stops and is reported.
- **The aarch64 half.** Every number here is x86-64 AVX-512. See `## Owed`.
- **Folding `cpu_paged_attn.cpp`'s private `StoreRowF32` into the shared
  `NarrowRowFromF32` this row adds.** They differ only in the refusal MESSAGE,
  so folding them changes another kernel's observable behaviour and needs its own
  gate. `## Owed`.

## Design

### The instrument

`tools/bench/vt_cpu_elem_survey_probe.cpp`. `--op <name>` runs one op in a loop
with no other `vt` work in the process, so `perf report`'s WHOLE-PROCESS self
percentages are that kernel's and no call-graph attribution is needed. Every case
reaches its kernel through the production `vt::` entry point, never by
constructing the kernel: a probe that called the kernel directly would measure a
function, not the path a model takes. `--list` prints, for every case, the
extents AND the model whose config they come from.

**The shapes are grounded, and that is load-bearing rather than decorative.** The
question a hoist turns on is whether the operand stream is dispatch-bound or
DRAM-bound, and that is a property of the extents, so a synthetic shape ranks
kernels wrongly. Qwen3.6-27B and Qwen3.6-35B-A3B config values come from
[`qwen36-forward-notes.md`](qwen36-forward-notes.md) §1 (read off the real
checkpoints), the GDN derived dims from [`gdn-semantics.md`](gdn-semantics.md)
§1, and the LTX-2.5 connector geometry from
`tools/bench/ltx2_connector_gemm_probe.cpp`. The two token counts are the two
regimes this repo's own harness runs: PREFILL 1024 and DECODE 16
(`tools/bench/online_gate.py:59` `INPUT_LEN`, `:74` `TRACE_CONCURRENCY`).

**Single thread, and why.** The predecessor's profile was taken at the shipped
thread count and carried `Threadpool::Barrier()` at 16.69%. This devbox ran at
loadavg 63-146 throughout, where the same barrier reads 46.77% and swamps
everything else, so a multithreaded ranking here would rank box contention. Every
ranking number below is therefore `VLLM_CPP_CPU_THREADS=1`, which measures the
kernel's instruction mix and not the spin-wait. **The bar is recalibrated onto
that instrument rather than carried across** — see `## Outcome`.

### The hoist

The same transformation `AttentionKernel` and `MatmulOneChunk` already apply in
the same file: resolve the dtype once per call and go through the shared
`WidenRowToF32` / `NarrowRowFromF32` row helpers. `NarrowRowFromF32` is new and
lands beside `WidenRowToF32` in `src/vt/cpu/cpu_matmul_elem.{h,cpp}`, which is
where a caller already looks for the pair.

## Reachability

Nothing new lands unreached. `RmsNormKernel` is already registered and already
reached from `ModelRegistry::Forward` through `vt::RmsNorm` — every text model in
the tree calls it twice per layer, and the `FusedChain` tier-0 walker dispatches
`kRmsNorm` to it. This row changes the body of a reached kernel and adds no entry
point. The gate enters through `vt::RmsNorm`, never by constructing the kernel,
and M7/M8 in `## Outcome` are the reachability mutations: making the new row
helpers inert reds the gate through the production entry point.

`NarrowRowFromF32` is new and is reached by that same call site at its own merge
commit. It is not a staged slice.

## Tests to port

There is no upstream test: upstream's RMSNorm is `torch`. The tests are this
tree's own, and each is an executable observable.

| ID | Assertion |
|---|---|
| T1 | CPU `vt::RmsNorm` is `memcmp`-identical to a per-element reference over the whole dtype matrix the op accepts, at ragged extents, gemma and non-gemma |
| T2 | the residual stream's add / round-on-store / RE-READ order is `memcmp`-identical, in BOTH residual dtypes, and the residual bytes are checked as well as the output bytes |
| T3 | a non-float operand is still refused at the op boundary |
| T4 | every one of the above is `memcmp`-identical at SEVEN worker counts (1, 2, 3, 4, 5, 8, 20), swapped in through `Threadpool::SwapForTesting`, over 64 rows so no two counts produce the same partition |

`tests/vt/test_ops_rmsnorm_elem_dispatch.cpp`. The reference is the ORIGINAL
per-element loop, re-derived from the layout contract in `include/vt/ops.h` and
sharing nothing with `src/vt/cpu` — including its own hand-written f16 and bf16
conversions, because a gate that compared the kernel against a helper the kernel
also uses would prove consistency, not correctness.

**ONE WORKER COUNT CANNOT TEST THIS CLAIM, and that is why T4 exists.** The
hoist's claim is about SUMMATION ORDER: each row's variance stays a serial f32
reduction on one thread and rows stay independent, so the threadpool's row
partition cannot move a single addend. A run at one worker count exercises ONE
partition, and the defect being excluded is a partition-dependent one. `ForRows`
chunks by rows, so a different worker count is a different partition. M9 in
`## Outcome` is the proof that this is not ceremony.

**The dtype matrix is bounded by what the OP accepts.** `vt::RmsNorm` takes x and
weight in {f32, f16, bf16} (`IsFloat`) and out and residual in {f32, bf16}
(`IsOutFloat`, `src/vt/ops.cpp:25`). FOUR tensors carry INDEPENDENT dtypes, which
is exactly the shape a typed table indexed by the wrong operand gets wrong, and
before this row every CPU RmsNorm test ran them all f32 or all bf16.

## Gates

1. `tests/vt/test_ops_rmsnorm_elem_dispatch.cpp` green, and green on a pristine
   base tree too, which is what says the reference is a valid oracle rather than
   a transcription of the new code.
2. The CPU `vt` norm suites green with the SAME case and assertion counts on both
   trees, so nothing was silently skipped.
2b. Byte equality proven at MORE THAN ONE worker count. A single-thread check does
   not test a claim about summation order.
3. A mutation per claimed guarantee, each verified to have APPLIED and BUILT
   before its result is read.
4. Before/after at one thread AND at the shipped thread count, arms interleaved,
   `n >= 3`, with a same-arm control leg and the box's load stated.
5. `scripts/agent-preflight.sh`.

## Risks/decisions

- **The devbox is not idle and cannot be made idle.** Three other agents compiled
  throughout; loadavg ran 63 to 146. Every comparison is INTERLEAVED with a
  same-arm control leg, and the control's own spread is quoted beside every
  ratio. A gap inside its control is not a result.
- **No CMake build tree fits, and a bare `ninja` writes 9.4 GiB with three other
  agents already compiling.** Every binary here is a direct `g++` over the `vt`
  TU set at the project's own flags, `-j 4` at most. `-ffp-contract=off` is not
  optional: the bit-identity contracts depend on it.
- **A mutation that does not build reads as a passing test.** The harness asserts
  the edit changed the file AND that the object compiled and linked before it
  reads a result; its first run reported eight BUILD-FAILED rather than eight
  passes, which is the behaviour that makes the later results readable.
- **This is x86-64 AVX-512 only.** The predecessor's own single-thread claim
  inverted between x86 and aarch64, so nothing here may be read as a GB10
  statement. `## Owed`.

## Stop conditions

- a kernel that cannot be hoisted bit-exactly — report it, never trade correctness;
- ENOSPC, or a build that fails for memory;
- a same-arm control that swallows the effect being claimed.

## Now

`ACTIVE`. W1, W2 and W3 are complete on x86-64. The aarch64 half and the
unhoisted remainder of the ranked set are owed and named in `## Owed`.

## Outcome

### The headline

**The dispatch is not a two-kernel defect and it is not a 62-kernel sweep. It is
a ranked list of 27, and this row publishes it.** Of the **49** unhoisted kernels
measured, **27 read `LoadF32` at or above the recalibrated bar and 22 read below
it**, so nearly half the population is now excluded by measurement rather than by
argument, and the other half is ordered.

**One kernel is hoisted here — `RmsNormKernel` — and it is 12.6x, byte-identical.**
Not because it tops the ranking (it is 9th) but because it is the widest-reaching
entry on it: every text model in the tree calls `vt::RmsNorm` twice per layer, and
the `FusedChain` tier-0 walker dispatches `kRmsNorm` to it. The other 26 stay in
`## Owed`, ranked, because each needs its own byte-equality gate and eight
mutations, and doing them in one change is the sweep the predecessor row forbade.

**`__attribute__((always_inline))` on `LoadF32` is a win in 43 of 48 kernels, up
to 4.58x, and it costs one kernel 15-18%.** The predecessor row deferred it for
want of exactly this measurement. The i-cache regression it feared is real, it is
measurable, and it is confined to `GdnDecodeKernel`.

### The instrument, and the bar it is read against

`tools/bench/vt_cpu_elem_survey_probe.cpp`, `perf record -e cpu-clock -F 1999`,
`VLLM_CPP_CPU_THREADS=1`, one `--op` per run. Self percentages are
whole-process; the probe's setup is amortised by sizing `--reps` so each run is
about three seconds of the op.

**The ~30% bar is recalibrated onto this instrument rather than carried across,
because the two instruments do not read the same number.** The predecessor
profiled at the shipped thread count and read `LoadF32` at 62.68% for
`AttentionCrossKernel`. Rebuilding that exact pre-hoist `cpu_ops.cpp`
(`2fd1d72f0^`, the only commit that touched the file in the range) and profiling
it here reads:

| kernel, PRE-HOIST | `LoadF32` self%, 1 thread | predecessor, shipped threads |
|---|---:|---:|
| `AttentionCrossKernel` | **67.93%** | 62.68% |
| `AttentionKernel` | **72.77%** | — |

So this instrument reads about 1.08x the predecessor's, and the ~30% bar
translates to **~32.5%** here. That is the line the ranking below is cut at.

The same two binaries also re-measure the predecessor's own result on this
instrument, which is what says the calibration is of the same effect: at the
LTX-2.5 connector's video shape, one thread, `AttentionCrossKernel` 57.563 s ->
5.474 s = **10.52x** and `AttentionKernel` 79.975 s -> 4.100 s = **19.51x**. Both
sit inside the 8.75x-11.16x the predecessor measured at the shipped thread count,
and the causal kernel's larger figure is the shape's, not a disagreement.

**Every number in this row is x86-64 AVX-512 on a devbox that was never idle.**
Three other agents compiled throughout and loadavg ran 63 to 146. That is why the
ranking is single-threaded: at the shipped thread count on this box
`Threadpool::Barrier()` reads 46.77% for `RmsNormKernel` against 16.69% in the
predecessor's profile, so a multithreaded ranking here would rank contention.

### W1 — the ranking

49 candidate kernels plus the two already-hoisted ones — 51 probe cases — each run
alone at the shape named beside it. `StoreF32` is printed too, because a kernel whose store
side is heavy pays for the hoist twice.

| # | kernel | LoadF32 self% | StoreF32 self% | probe shape | the model it comes from |
|---:|---|---:|---:|---|---|
| 1 | `BatchedMatmulKernel` | **70.46** | 0.71 | `a[32,1024,128] x b[32,128,128] f32` | LTX-2.5 per-head batched GEMM, 32 heads |
| 2 | `DFlashBlockAttentionKernel` | **57.08** | 1.09 | `q/k/v[256,32,128] f32` | Qwen3-DFlash in-block attention, 16 reqs x 16-token block |
| 3 | `CausalConv1dFwdKernel` | **55.81** | 2.38 | `x[1024,10240] f32, w[10240,4]` | Qwen3.6-27B GDN prefill conv, 1024 tokens |
| 4 | `CastBf16Kernel` | **54.34** | 21.74 | `x[1024,5120] f32 -> bf16` | f32 -> bf16 activation cast, prefill |
| 5 | `RopeFromCacheKernel` | **52.98** | 17.49 | `q[1024,24,256] k[1024,4,256] f32, cache[1024,64]` | Qwen3.6-27B RopeFromCache q/k, prefill |
| 6 | `FusedNormRopeKernel` | **52.23** | 11.79 | `x[1024,576] f32, latent 512 \| pe 64` | DeepSeek-V3 MLA fused kv_a norm+rope, prefill |
| 7 | `RmsNormGroupKernel` | **50.74** | 10.89 | `x[1024,5120] f32, group=128` | Qwen4-exp grouped RMSNorm, prefill |
| 8 | `AttnGateSplitKernel` | **49.87** | 29.30 | `qgate[1024,12288] f32` | Qwen3.6-27B attn_output_gate q\|gate split, prefill |
| 9 | `RmsNormKernel` | **49.06** | 12.05 | `x[1024,5120] f32, w[5120]` | Qwen3.6-27B input_layernorm, prefill |
| 10 | `MulScalarKernel` | **47.09** | 23.43 | `x[1024,5120] f32` | Gemma embedding normalizer, prefill |
| 11 | `AttnQkNormRopeGateKernel` | **47.04** | 15.00 | `qgate[1024,12288] kf[1024,1024] f32` | Qwen3.6-27B fused attention preamble, prefill |
| 12 | `GdnConvSplitKernel` | **43.61** | 29.48 | `conv[1024,10240] f32` | Qwen3.6-27B GDN mixed-qkv split, prefill |
| 13 | `CastF32Kernel` | **42.56** | 25.38 | `x[1024,5120] bf16 -> f32` | bf16 -> f32 GEMM-result cast, prefill |
| 14 | `Mamba2StateUpdateKernel` | **42.30** | 13.58 | `state[16,128,64,128] f32` | NemotronH Mamba2 decode step, c16 |
| 15 | `L2NormKernel` | **40.97** | 16.70 | `x[1024,16,128] f32` | Qwen3.6-27B GDN q/k l2norm, prefill |
| 16 | `QkvSplitKernel` | **40.40** | 36.63 | `qkv[1024,8192] f32` | Qwen3.6-27B merged QKV split, prefill |
| 17 | `GdnPostConvKernel` | **40.15** | 20.16 | `conv[1024,10240] f32` | Qwen3.6-27B fused GDN post-conv prep, prefill |
| 18 | `GdnStateScatterKernel` | **39.58** | 28.42 | `working[16,...] -> cache[64,48,128,128] f32` | Qwen3.6-27B GDN state scatter, c16 |
| 19 | `DFlashGroupedConvKernel` | **37.66** | 3.25 | `x[256,5120] f32, taps 4, groups 40` | Qwen3-DFlash grouped conv prepare, 16 reqs x 16-token block |
| 20 | `GdnStateGatherKernel` | **36.60** | 28.10 | `cache[64,48,128,128] f32 -> working[16,...]` | Qwen3.6-27B GDN state gather, c16 |
| 21 | `SharedExpertGateKernel` | **35.93** | 21.57 | `sd[1024,2048] f32, gl[1024]` | Qwen3.6-35B shared-expert sigmoid gate, prefill |
| 22 | `ConcatMlaNopeRopeKernel` | **35.48** | 34.69 | `nope[1024,128,128] rope[1024,1,64] f32` | DeepSeek-V3 MLA nope\|rope head concat, prefill |
| 23 | `MoeRelu2Kernel` | **35.12** | 24.97 | `x[8192,512] f32` | NemotronH expert relu^2, prefill x top_k |
| 24 | `RmsNormGatedKernel` | **34.85** | 6.92 | `x[16,48,128] f32` | Qwen3.6-27B GDN output gated norm, decode |
| 25 | `CausalConv1dSpecUpdateKernel` | **33.64** | 3.62 | `x[64,10240] f32` | Qwen3.6-27B GDN speculative conv step, c16 x 4 |
| 26 | `MoeCombineKernel` | **33.47** | 2.54 | `expert_out[1024,8,2048] f32` | Qwen3.6-35B weighted expert combine, prefill |
| 27 | `CausalConv1dUpdateKernel` | **33.01** | 4.22 | `x[16,10240] f32, state[16,10240,3]` | Qwen3.6-27B GDN decode conv step, c16 |
| 28 | `CastF16Kernel` | **28.37** | 27.75 | `x[1024,5120] f32 -> f16` | EXL3 activation narrowing cast, prefill |
| 29 | `SigmoidGateBf16Kernel` | **25.03** | 9.34 | `attn/gate[1024,6144] f32 -> bf16` | Qwen3.6-27B attention output gate, prefill |
| 30 | `RmsNormGatedGroupKernel` | **24.75** | 7.50 | `x[1024,4096] f32, n_groups=8` | Mamba2 Mixer2RMSNormGated, prefill |
| 31 | `RmsNormQuantFp8Kernel` | **24.45** | 0.00 | `x[1024,5120] f32 -> e4m3 + bf16` | Qwen3.6-27B fused RMSNorm -> fp8, prefill |
| 32 | `RmsNormGatedQuantFp8Kernel` | **23.55** | 0.00 | `x/gate[16,48,128] f32 -> e4m3` | Qwen3.6-27B GDN gated norm -> fp8, decode |
| 33 | `ScaledFp4QuantKernel` | **22.51** | 0.00 | `x[1024,5120] f32 -> fp4 + e4m3 scales` | Qwen3.6-27B NVFP4 activation quant, prefill |
| 34 | `RopeRotateHead` | **21.32** | 5.02 | `q[1024,24,256] k[1024,4,256] f32, rot=64` | Qwen3.6-27B RopeNeox q/k in place, prefill |
| 35 | `MoeRouterTopKKernel` | **20.41** | 0.00 | `logits[1024,256] f32` | Qwen3.6-35B router softmax top-8 of 256, prefill |
| 36 | `SiluAndMulKernel` | **19.52** | 6.19 | `x[1024,34816] f32 -> out[1024,17408]` | Qwen3.6-27B dense MLP act, prefill |
| 37 | `QuantFp8GroupKernel` | **17.78** | 0.00 | `x[1024,5120] f32, group 128` | DeepSeek block-fp8 per-128-group activation quant, prefill |
| 38 | `MoeSiluMulKernel` | **17.63** | 8.75 | `gate/up[8192,512] f32` | Qwen3.6-35B expert act, prefill x top_k |
| 39 | `QuantFp8StaticKernel` | **15.80** | 0.00 | `x[1024,5120] f32 -> e4m3` | Qwen3.6-27B static per-tensor fp8 activation quant, prefill |
| 40 | `GdnGBetaKernel` | **11.50** | 6.18 | `araw/braw[1024,48] f32` | Qwen3.6-27B GDN decay/gate derivation, prefill |
| 41 | `GeluAndMulKernel` | **11.36** | 3.03 | `x[1024,34816] f32 -> out[1024,17408]` | Gemma GeGLU MLP act, prefill |
| 42 | `MoeRouterGroupedTopKKernel` | **10.62** | 0.00 | `logits[1024,256] f32, groups 8, topk_group 4` | DeepSeek grouped sigmoid router, prefill |
| 43 | `SoftCapKernel` | **7.15** | 6.60 | `x[16,248320] f32` | Gemma-2 final logit soft-cap, decode |
| 44 | `GdnDecodeKernel` | **3.19** | 2.11 | `state[16,48,128,128] f32` | Qwen3.6-27B GDN decode recurrence, c16 |
| 45 | `KdaHeadTokenStep` | **3.18** | 0.39 | `T=64, state[1,48,128,128] f32` | Kimi KDA per-channel decay recurrence, c16 prefill |
| 46 | `EmbeddingKernel` | **0.31** | 0.10 | `table[248320,5120] bf16, ids[1024]` | Qwen3.6-27B embed_tokens gather, prefill |
| 47 | `RopeCosSinCacheKernel` | **0.00** | 6.48 | `cos_sin[1024,64] f32` | Qwen3.6-27B rotary cache build, prefill |
| 48 | `MatmulOneChunk` | **0.00** | 0.72 | `a[1024,4096] x b[4096,4096]^T f32` | LTX-2.5 connector attn projection, 1024 rows |
| 49 | `AttentionKernel` | **0.00** | 0.52 | `q/k/v[1024,32,128] f32, causal` | LTX-2.5 DiT self-attention, video stream |
| 50 | `AttentionCrossKernel` | **0.00** | 0.45 | `q/k/v[1024,32,128] f32` | LTX-2.5 connector cross-attention, video stream |
| 51 | `MulColVecF32Kernel` | **0.00** | 0.00 | `x[1024,10240] f32, col[10240]` | merged fp8 projection per-shard dequant, prefill |

**The population, counted exactly, because the predecessor's "62" and this row's
inventory do not line up on the nose.** The inventory finds 67 enclosing
functions; five are the shared helpers rather than kernels (`LoadF32`,
`StoreF32`, `AttnResolveOrRefuse`, and `FusedChainKernel`'s `FusedLoad` /
`FusedStore`) and two are the pair already hoisted. 49 of the remainder are
measured below and 13 are not, and both lists are enumerated rather than left to
subtraction.

**Reading it.** `AttentionKernel` and `AttentionCrossKernel` at 0.00% are the
predecessor's hoist showing up as the absence of the symbol; they are the
instrument's own control. `RopeCosSinCacheKernel` at 0.00% has no `LoadF32` call
at all (it is store-only), and `MulColVecF32Kernel` and `MatmulOneChunk` reach
specialized f32 paths that never enter the per-element helper. `GdnDecodeKernel`
(3.19%) and `KdaHeadTokenStep` (3.18%) are the clearest below-bar cases in the
set: their recurrences are dominated by the state outer-product, and a hoist
there would buy the 3% the profile names.

**27 kernels are at or above 32.5%.** In rank order: `BatchedMatmulKernel`,
`DFlashBlockAttentionKernel`, `CausalConv1dFwdKernel`, `CastBf16Kernel`,
`RopeFromCacheKernel`, `FusedNormRopeKernel`, `RmsNormGroupKernel`,
`AttnGateSplitKernel`, `RmsNormKernel`, `MulScalarKernel`,
`AttnQkNormRopeGateKernel`, `GdnConvSplitKernel`, `CastF32Kernel`,
`Mamba2StateUpdateKernel`, `L2NormKernel`, `QkvSplitKernel`, `GdnPostConvKernel`,
`GdnStateScatterKernel`, `DFlashGroupedConvKernel`, `GdnStateGatherKernel`,
`SharedExpertGateKernel`, `ConcatMlaNopeRopeKernel`, `MoeRelu2Kernel`,
`RmsNormGatedKernel`, `CausalConv1dSpecUpdateKernel`, `MoeCombineKernel`,
`CausalConv1dUpdateKernel`.

**22 are below it, and that is the half of the result that closes work rather
than opening it.** `CastF16Kernel` (28.37%) down to `EmbeddingKernel` (0.31%) are
bound by something else, and the predecessor's own rule says a hoist there buys
the fraction the profile names. They need no row.

**13 of the enclosing functions the inventory names are NOT measured here, and are
named rather than left implied**:
`Mamba2ChunkScanKernel`, `GdnPackedDecodeKernel`, `GdnSpecDecodeKernel`,
`KdaChunkPrefillKernel`, `DFlashPagedBlockAttentionKernel`,
`Dflash2SelectorEdgesKernel`, `MatmulFp8BlockScaledKernel`,
`MatmulFp8CutlassKernel`, `MatmulNvfp4Fp4Kernel`, `MatmulOneChunkRef`,
`GdnHeadTokenStep` (reached only inside `GdnDecodeKernel`'s 3.19%), and the
`FusedLoad`/`FusedStore` pair inside `FusedChainKernel`. Each needs one `Add(...)`
case in the committed probe and nothing else; they are in `## Owed`.

### W2 — the hoist, and why this kernel

`RmsNormKernel`, 8th on the ranking at 49.06% `LoadF32` + 12.05% `StoreF32`,
chosen over the seven above it because it is the one every model reaches. The
dispatch is now resolved once per call through the shared `WidenRowToF32` and a
new `NarrowRowFromF32`; `w` is widened once per CALL rather than once per row,
and the gemma `+1` folds into that copy.

The profile after, same instrument, same shape: `LoadF32` and `StoreF32` are
**gone from the profile entirely**; the kernel body is 65.65% and
`__memmove_avx512_unaligned_erms` (the f32 arm's row widen) is 21.58%.

**The A/B, interleaved, with a same-arm control leg.** Ten `old` legs and five
`new` legs alternating, `--reps 20`, one thread, loadavg 63-69:

| statistic | old | new | ratio |
|---|---:|---:|---:|
| median of legs | 3.47 s | 0.2745 s | **12.6x** |
| least-contended leg | 1.8237 s | 0.2146 s | **8.5x** |
| most-contended leg | 4.3019 s | 0.7398 s | 5.8x |

The same-arm control (old against old, in the same interleave) ran **0.45x to
1.66x**, so the box's own drift is a factor of 3.7 and the effect is a factor of
8.5 to 12.6. A gap that far outside its own control is a result whatever the box
was doing, and the spread is quoted rather than smoothed.

**At the shipped thread count it is 2.15x, and that is not a contradiction.**
Same interleave, `--reps 30`, twenty threads on a twenty-core box already at
loadavg 73: old median 2.96 s / min 2.1212 s, new median 1.3714 s / min 0.9864 s,
control 0.95x-1.54x. `Threadpool::Barrier()` is 46.77% of that arm before the
change and does not shrink when the kernel does, so the single-thread figure is
the kernel's instruction-level win and the twenty-thread figure is what a
contended box sees. Both are reported because quoting either alone misleads.

### W3 — the correctness evidence

**Byte equality is the whole gate.**
`tests/vt/test_ops_rmsnorm_elem_dispatch.cpp`: **4 cases / 1,812,015
assertions**, every one a `memcmp` against a reference that re-derives the
original per-element loop and shares nothing with `src/vt/cpu`, down to its own
hand-written f16 and bf16 conversions.

**At SEVEN worker counts, not one.** T4 swaps in pools of 1, 2, 3, 4, 5, 8 and 20
through `Threadpool::SwapForTesting` over 64 rows, so no two counts produce the
same row partition, and the reference it is compared against is serial — making it
a parallel-vs-SERIAL identity and not merely parallel-vs-parallel.

**Green on a pristine base tree too**, built from `43553262c`'s own
`cpu_ops.cpp` and `cpu_matmul_elem.cpp`, which is what says the reference is an
oracle rather than a transcription of the new code.

**The norm suite: 59 cases / 1,824,059 assertions, 0 failed** over `test_dtype`,
`test_fused_chain_additivity`, `test_ops_fused_chain`, `test_ops_layernorm`,
`test_ops_mamba2_gated_norm`, `test_ops_rmsnorm`, `test_ops_rms_norm_group`,
`test_rmsnorm_decode_fast`, `test_rmsnorm_gated_fast`,
`test_ops_attention_elem_dispatch` and the new file. **The counts are IDENTICAL
on the pristine base tree, on the changed tree, and on the merged head**, which
is what says nothing was silently skipped rather than silently passing.

**The mutations.** Each is applied to a COPY outside the worktree, so a mutation
cannot be left behind; each is verified to have changed the file, compiled and
linked before its result is read. The harness's first run reported every row
`BUILD-FAILED` rather than every row passing, which is the behaviour that makes
the rest readable. None of the ten is an environment variable, so none can be a
CI lane's permanent configuration wearing a mutation's clothes.

| # | mutation | result |
|---|---|---|
| M1 | split the `sumsq` reduction into two accumulators | KILLED — 81 assertions red |
| M2 | residual: skip the round-trip RE-READ | KILLED — 54 red |
| M3 | fold the gemma `+1` unconditionally | KILLED — 108 red |
| M4 | `NarrowRowFromF32` bf16 truncates instead of round-to-nearest-even | KILLED — 189 red |
| M5 | `NarrowRowFromF32`'s **f16** branch made wrong | **SURVIVED** |
| M6 | `WidenRowToF32`'s f16 branch made wrong | KILLED — 114 red |
| M7 | REACHABILITY: the new hoisted store is inert | KILLED — 216 red |
| M8 | REACHABILITY: the widened `w` copy is ignored | KILLED — 216 red |
| M9 | PARTITION: the row scratch is hoisted out of `ForRows` and shared across workers | KILLED — 215 red |
| M10 | PARTITION: `sumsq` carries across rows inside a chunk | KILLED — 72 red |

**M2 SURVIVED on the first version of this gate, and repairing that is the most
useful thing the mutation set did.** The residual contract is add-in-f32,
round-on-store, RE-READ; dropping the re-read is only observable when the sum
does not survive the round trip. The first operand set drew x and residual from
one scale, `k*2^-6` with `|k| < 128`, whose sums need at most 8 significant bits
— which bf16 holds EXACTLY. The gate was reading a tautology on its most subtle
guarantee. Giving the residual a scale `2^12` above x puts the operands twelve
binades apart, and M2 now dies with 12 assertions. The values stay exactly
representable in f32, f16 and bf16 individually; only the SUM rounds, which is
precisely the thing under test.

**M5 survived and it is a finding, not a hole this row opened.** `vt::RmsNorm`
validates `IsOutFloat(out.dtype)` (`src/vt/ops.cpp:25`), which admits f32 and
bf16 only, so `NarrowRowFromF32`'s f16 branch is UNREACHABLE from this op. No
test here can kill it and this row does not claim one does. M4 is the same
guarantee on the branch that IS reachable, and it dies. This is the predecessor
row's M3b, on the same op boundary, for the same reason.

**M9 is why T4 is not ceremony, and its isolation is the red-first proof.**
Hoisting the row scratch out of `ForRows` shares one buffer across every worker:
a data race that is CORRECT at one thread. With M9 applied and
`VLLM_CPP_CPU_THREADS=1` — the environment a single-worker runner hands the
process — the two dtype-matrix cases pass **753,300 and 367,344 assertions with 0
failed on a broken kernel**, while T4 fails 108. So T4 detects a defect the other
cases structurally cannot see, and the seven worker counts are load-bearing rather
than decorative. M10 is the same point from the other side: carrying `sumsq`
across rows inside a chunk makes the answer depend on where the chunk boundaries
fall, and it dies with 72 red.

**M7 and M8 are the reachability proof.** Making the new store inert, and making
the new widened-`w` copy unused, both red the gate THROUGH `vt::RmsNorm` — so
what the test exercises is the new code on the production path, not a class the
test constructed.

**The refusal moved and its message did not.** The per-element refusal used to be
raised inside a threadpool worker; it is now raised on the calling thread before
any element is read, and it is produced by `LoadF32`/`StoreF32`/`SizeOf`
themselves rather than by a second message that could drift from theirs. That
branch is unreachable through `vt::RmsNorm`, which validates `IsFloat` first, and
T3 asserts the op-boundary refusal rather than claiming a gate on the kernel one.

### W3b — the `always_inline` verdict, which is what the predecessor row could not answer

`__attribute__((always_inline))` on `LoadF32`, against this branch's head, over
every case in the probe. **`perf stat` counts, not wall time**, because the
effect being looked for is a few percent of instruction-cache behaviour and this
box's wall-clock noise is a factor of 3.7. Cycles are the minimum of three runs;
instructions are deterministic; i-cache misses are normalised per kilo-instruction
so the two arms are comparable when the instruction count changes.

**The code-size cost, measured rather than asserted:** `cpu_ops.cpp`'s `.text`
grows 113,995 -> 173,818 bytes = **+52.5%**, and the whole probe binary's text
grows 1,729,517 -> 1,881,108 = +8.8%.

| kernel | cycles base / always_inline | i-cache misses per kilo-insn, base -> ai | Ginsn base -> ai |
|---|---:|---|---|
| `GdnDecodeKernel` | 0.900 | 0.1037 -> 0.1584 | 5.59 -> 5.41 |
| `SoftCapKernel` | 0.980 | 0.0797 -> 0.1333 | 7.52 -> 6.42 |
| `RmsNormKernel` | 0.993 | 0.2843 -> 0.2771 | 1.76 -> 1.76 |
| `CastF16Kernel` | 1.005 | 0.0657 -> 0.0848 | 11.13 -> 8.50 |
| `RopeCosSinCacheKernel` | 1.016 | 0.1285 -> 0.1301 | 10.65 -> 10.65 |
| `MatmulOneChunk` | 1.042 | 0.1205 -> 0.1294 | 40.62 -> 40.62 |
| `RmsNormQuantFp8Kernel` | 1.050 | 0.0515 -> 0.0753 | 11.04 -> 7.77 |
| `QuantFp8StaticKernel` | 1.053 | 0.0720 -> 0.0816 | 14.25 -> 11.63 |
| `AttentionCrossKernel` | 1.055 | 0.0688 -> 0.0896 | 121.52 -> 121.55 |
| `QuantFp8GroupKernel` | 1.062 | 0.0574 -> 0.1431 | 9.38 -> 6.40 |
| `MoeRouterGroupedTopKKernel` | 1.065 | 0.0977 -> 0.0725 | 7.38 -> 5.90 |
| `MoeRelu2Kernel` | 1.090 | 0.1712 -> 0.1752 | 5.68 -> 4.00 |
| `AttentionKernel` | 1.090 | 0.0793 -> 0.0671 | 61.15 -> 61.11 |
| `GeluAndMulKernel` | 1.091 | 0.1827 -> 0.3589 | 11.29 -> 9.08 |
| `GdnGBetaKernel` | 1.113 | 0.0397 -> 0.0642 | 9.89 -> 7.74 |
| `EmbeddingKernel` | 1.119 | 0.9478 -> 0.8786 | 102.32 -> 101.85 |
| `SharedExpertGateKernel` | 1.170 | 0.0401 -> 0.1018 | 6.47 -> 4.66 |
| `GdnStateGatherKernel` | 1.208 | 0.3088 -> 0.3748 | 7.21 -> 5.63 |
| `ScaledFp4QuantKernel` | 1.213 | 0.0879 -> 0.2505 | 8.83 -> 4.60 |
| `ConcatMlaNopeRopeKernel` | 1.230 | 0.1241 -> 0.2439 | 11.31 -> 8.08 |
| `MulColVecF32Kernel` | 1.278 | 0.1368 -> 0.1423 | 3.40 -> 3.40 |
| `QkvSplitKernel` | 1.323 | 0.1607 -> 0.1545 | 6.59 -> 4.90 |
| `MoeRouterTopKKernel` | 1.377 | 0.0446 -> 0.0386 | 8.95 -> 5.93 |
| `RopeFromCacheKernel` | 1.383 | 0.1036 -> 0.3389 | 3.58 -> 2.37 |
| `RopeRotateHead` | 1.400 | 0.2241 -> 0.1205 | 3.92 -> 3.40 |
| `RmsNormGatedQuantFp8Kernel` | 1.435 | 0.0422 -> 0.0670 | 17.63 -> 11.82 |
| `SigmoidGateBf16Kernel` | 1.473 | 0.0634 -> 0.0716 | 11.98 -> 8.52 |
| `MoeSiluMulKernel` | 1.476 | 0.0819 -> 0.0909 | 8.93 -> 6.62 |
| `GdnStateScatterKernel` | 1.568 | 0.5786 -> 0.4510 | 5.99 -> 4.47 |
| `GdnConvSplitKernel` | 1.576 | 0.1077 -> 0.1717 | 12.61 -> 9.41 |
| `GdnPostConvKernel` | 1.590 | 0.1762 -> 0.2215 | 7.31 -> 4.65 |
| `RmsNormGatedGroupKernel` | 1.704 | 0.1121 -> 0.0817 | 7.81 -> 5.54 |
| `AttnGateSplitKernel` | 1.705 | 0.1444 -> 0.1737 | 8.65 -> 6.14 |
| `MulScalarKernel` | 1.709 | 0.0962 -> 0.0954 | 8.54 -> 5.91 |
| `CastBf16Kernel` | 1.789 | 0.0724 -> 0.0494 | 13.60 -> 9.79 |
| `Mamba2StateUpdateKernel` | 1.915 | 0.0749 -> 0.2074 | 10.41 -> 5.91 |
| `L2NormKernel` | 1.946 | 0.1016 -> 0.1468 | 2.86 -> 1.45 |
| `SiluAndMulKernel` | 1.959 | 0.2677 -> 0.1841 | 12.18 -> 8.79 |
| `CastF32Kernel` | 1.971 | 0.0607 -> 0.0505 | 14.43 -> 10.10 |
| `AttnQkNormRopeGateKernel` | 2.025 | 0.1436 -> 0.1966 | 10.89 -> 6.12 |
| `RmsNormGroupKernel` | 2.058 | 0.0714 -> 0.1419 | 8.57 -> 4.14 |
| `RmsNormGatedKernel` | 2.200 | 0.0516 -> 0.0538 | 4.85 -> 2.67 |
| `CausalConv1dUpdateKernel` | 2.521 | 0.0228 -> 0.0289 | 13.73 -> 7.21 |
| `FusedNormRopeKernel` | 2.674 | 0.0816 -> 0.0301 | 12.41 -> 6.10 |
| `MoeCombineKernel` | 2.800 | 0.1265 -> 0.2850 | 7.96 -> 2.75 |
| `CausalConv1dFwdKernel` | 3.023 | 0.1708 -> 0.1939 | 10.64 -> 6.08 |
| `KdaHeadTokenStep` | 3.214 | 0.0868 -> 0.0517 | 7.55 -> 5.55 |
| `BatchedMatmulKernel` | 4.584 | 0.0473 -> 0.0361 | 95.26 -> 35.07 |

**The verdict: 43 wins, 4 neutral, 1 regression.** The wins run to 4.58x
(`BatchedMatmulKernel`), 3.21x, 3.02x and 2.80x, and they are not i-cache effects
in the other direction — they are instruction-count effects. `BatchedMatmulKernel`
executes 95.26 Ginsn out of line and 35.07 Ginsn inlined, because once `LoadF32`
is inlined the loop vectorises. That is the same mechanism the predecessor's
1.78x came from, and it is much larger than 1.78x on the kernels that stream.

**The regression is real and it is one kernel.** `GdnDecodeKernel` reads 0.900 in
the sweep, and re-measured on its own at `--reps 40` with the minimum of five
runs it reads **2.538e9 vs 2.990e9 cycles = 0.849x, 17.8% SLOWER** — with
i-cache misses per kilo-instruction up 0.1037 -> 0.1584 (**+53%**) at an
essentially unchanged instruction count (5.59 -> 5.41 Ginsn). Fewer instructions,
more cycles, more i-cache misses: that is the instruction-cache cost the
predecessor row named, found, and it is confined to the one kernel whose inner
loop is a state outer-product that never benefited from the inline in the first
place.

**`SoftCapKernel`'s 0.980 in the sweep did NOT reproduce.** At `--reps 40`,
minimum of five, it reads 1.022e10 vs 8.673e9 = **1.178x FASTER**. One sweep leg
is not a result on this box, which is why both suspected regressions were
re-measured before either was reported.

**So the idea is not retired and it is not free.** It is a one-line change worth a
median of about 1.4x across 43 CPU kernels, at 52.5% more `cpu_ops.cpp` text and
one kernel 18% slower. It is not landed here because it changes 48 kernels at
once and owes a byte-equality gate across all of them, which is its own row —
this row's obligation was to produce the measurement that decides it, and the
measurement says take it, with `GdnDecodeKernel` excluded or accepted explicitly.

### What this row did NOT do, stated rather than implied

- It hoisted **one** of the 27 kernels above the bar. The other 26 are in
  `## Owed` in rank order with their measured percentages.
- It did not measure 13 of the enclosing functions in the inventory. They are
  named above and in `## Owed`.
- It produced no aarch64 number at all.
- It did not land `always_inline`.

## Owed

- **THE 26 REMAINING KERNELS ABOVE THE BAR**, in rank order with their measured
  `LoadF32` self percentage, so the next row picks by number rather than by
  position in the file: `BatchedMatmulKernel` 70.46,
  `DFlashBlockAttentionKernel` 57.08, `CausalConv1dFwdKernel` 55.81, `CastBf16Kernel` 54.34, `RopeFromCacheKernel` 52.98,
  `FusedNormRopeKernel` 52.23, `RmsNormGroupKernel` 50.74, `AttnGateSplitKernel`
  49.87, `MulScalarKernel` 47.09, `AttnQkNormRopeGateKernel` 47.04,
  `GdnConvSplitKernel` 43.61, `CastF32Kernel` 42.56, `Mamba2StateUpdateKernel`
  42.30, `L2NormKernel` 40.97, `QkvSplitKernel` 40.40, `GdnPostConvKernel` 40.15,
  `GdnStateScatterKernel` 39.58, `DFlashGroupedConvKernel` 37.66,
  `GdnStateGatherKernel` 36.60, `SharedExpertGateKernel` 35.93,
  `ConcatMlaNopeRopeKernel` 35.48, `MoeRelu2Kernel` 35.12, `RmsNormGatedKernel`
  34.85, `CausalConv1dSpecUpdateKernel` 33.64, `MoeCombineKernel` 33.47,
  `CausalConv1dUpdateKernel` 33.01. `RmsNormGroupKernel` is the cheapest of them
  to take next: it is deliberately shaped as `RmsNormKernel` with the reduction
  extent narrowed, so the same helpers and the same gate structure apply.
  Owner: unowned; #2416 stays open against this item.
- **THE 13 UNMEASURED ENCLOSING FUNCTIONS.** `Mamba2ChunkScanKernel`,
  `GdnPackedDecodeKernel`, `GdnSpecDecodeKernel`, `KdaChunkPrefillKernel`,
  `DFlashPagedBlockAttentionKernel`, `Dflash2SelectorEdgesKernel`,
  `MatmulFp8BlockScaledKernel`, `MatmulFp8CutlassKernel`, `MatmulNvfp4Fp4Kernel`,
  `MatmulOneChunkRef`, `GdnHeadTokenStep`, and `FusedChainKernel`'s
  `FusedLoad`/`FusedStore`. Each needs one `Add(...)` case in the committed probe
  and nothing else. Owner: unowned.
- **`__attribute__((always_inline))` ON `LoadF32`: MEASURED, AND THE ANSWER IS
  TAKE IT.** 43 wins of 48 up to 4.58x, 4 neutral, one regression
  (`GdnDecodeKernel`, 0.849x, i-cache misses per kilo-instruction +53%), at
  +52.5% `cpu_ops.cpp` text. It is not landed here because it changes 48 kernels
  at once and owes a byte-equality gate across all of them. Owner: unowned.
- **THE aarch64 HALF. Every number in this row is x86-64 AVX-512.** The tier is
  NEON with a different vector width and different inlining economics, and the
  predecessor row's own single-thread claim inverted between the two
  architectures, so nothing here may be read as a GB10 statement. The repair is
  bit-exact by construction on any ISA, so what is owed is the RATE and the
  RANKING, not the correctness. Reaching those cores needs an `rc` lease.
  Owner: unowned.
- **`cpu_paged_attn.cpp`'s file-private `StoreRowF32` should fold into the shared
  `NarrowRowFromF32` this row adds.** They are the same function; they differ only
  in the refusal MESSAGE, so folding them changes another kernel's observable
  behaviour and needs its own gate. Owner: unowned.
- **A MULTITHREADED RANKING.** Every percentage here is single-threaded, because
  at the shipped thread count on a box at loadavg 63-146 `Threadpool::Barrier()`
  takes 46.77% and a ranking would rank contention. Whether the ORDER changes at
  twenty threads on an idle box is unmeasured, not answered. Owner: unowned.
