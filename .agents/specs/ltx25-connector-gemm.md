# SPEC — `LTX25-CONNECTOR-GEMM`: what the connector's 224.9 s of host f32 GEMM is actually spent on

Issue: **NONE — `REMOTE_UNVERIFIED`.** The `mudler-agent` GitHub account returns
`Your account is suspended` (HTTP 403) on `gh api user`, so no issue could be
opened for this row. The work it takes up is the first `## Owed` item of
[`.agents/specs/ltx25-text-cond-device.md`](ltx25-text-cond-device.md), which
records the same block and the same reason:

> **THE COMPUTE LEVER IS UNOWNED, and it is 43.52% of the render.** 224.882 s of
> host f32 GEMM at ~37 GFLOP/s [...] **No issue was filed for this: the GitHub
> account was suspended mid-row.**

Owner row: `LTX25-CONNECTOR-GEMM`. Predecessors:
[#2296](https://github.com/mudler/vllm.cpp/issues/2296) (the 5.53x reading) and
[#2354](https://github.com/mudler/vllm.cpp/issues/2354) (the weights/compute
split). An issue is owed the moment the account is restored.

## Scope

`LTX25-TEXT-COND-DEVICE` split `conditioning.connector` and found that the time
is **arithmetic, not weight materialization**: connector compute is 224.882 s
over both guidance passes, 43.52% of the render, against 13.707 s for all four
materializations. It named two traceable next steps and moved neither. This row
takes the first one.

IN scope:

- **W1** — establish **BY EXECUTION** which CPU GEMM kernel the connector's
  `vt::MatmulBT` calls actually enter, on both architectures that matter: this
  x86-64 devbox and the GB10 aarch64 cores the 37 GFLOP/s was measured on.
- **W2** — measure the **achievable rate** for the connector's exact GEMM shapes
  in each orientation the tree can express: `MatmulChunked<true>` (what runs),
  `MatmulChunked<false>` over a `[K,N]`-repacked weight (what
  `b.elem_kn_repacked` unlocks), and the historical reference tile
  (`VT_CPU_MATMUL_TIER=ref`).
- **W3** — decompose `Ltx2ConnectorForward` itself, so the claim "the connector
  is GEMM" is measured rather than assumed.
- **W4** — a repair **only if** it is bit-exact and the measurement supports it;
  otherwise the attribution and the next hypothesis.

OUT of scope, declared rather than approximated:

- **A device arm for the connector.** `Ltx2Attention` interleaves host
  `RmsNormRows` and `Ltx2ApplyRotaryEmb` on raw `float*` between its GEMMs and
  `Ltx2ConnectorForward` reads its weights as host `std::vector<float>`, so that
  is a weight-arm port with its own numerics gate, not a queue swap. #2354 says
  so and this row does not contradict it.
- **A published benchmark ID.** `docs/BENCHMARKS.md` gains nothing. A probe of
  one module's GEMM shapes is an instrument, not a benchmark.
- **Changing `vt`'s numeric contract.** Every kernel in `cpu_matmul_elem.cpp`
  keeps each output's K reduction strictly sequential, which is what makes the
  SIMD tiers bit-identical to the scalar reference. Nothing here may weaken it.

## The two predicates, and why reading them is not the deliverable

The brief this row was dispatched with named two static questions. Both are
answerable by reading and **both were read before this spec was written**, so
they are recorded here as inputs rather than as findings:

| question | answer | evidence |
|---|---|---|
| does f32 have an `ElemKind`? | YES | `src/vt/cpu/cpu_matmul_elem.cpp`, `ElemKindOf`, `case DType::kF32` |
| so does the dtype force `MatmulOneChunkRef`? | NO | in `MatmulOneChunk`, `!ElemKindOf(b.dtype, &bk) \|\| !ElemKindOf(a.dtype, &ak) \|\| k <= 0 \|\| ElemGemmUseRef()` is false for f32 |
| is the connector's weight repack-eligible? | YES | `ElemRepackEligible` returns true for f32 with `n, k > 0` |
| does anything repack it? | NO | the only `ElemRepackWeight` caller and the only `elem_kn_repacked = true` assignment are both in `qwen3_5_gguf_weights.cpp` |

**So the connector takes `MatmulChunked<true>`, the BT orientation, through
whatever tier the process resolved.** That is a reading, and this repository has
been wrong before about what a predicate implies at runtime. Two things are still
unknown after it and neither is readable:

1. **Which tier resolves on the GB10's cores.** If `ElemGemmTier()` names `"ref"`
   there — from an environment variable, from a failed feature probe, from a
   build that never compiled the NEON TU — the reference tile is running and
   every sentence above is moot.
2. **What the BT orientation costs.** `MatmulChunked<true>` is not a materialized
   transpose; it is a different read pattern with different locality and, on the
   SIMD tiers, a 4x4 register transpose per weight group. Whether that is 10% or
   3x is a measurement, and `.agents/specs/kernel-gemm-cpu-tiled.md`'s lever 2
   exists precisely because nobody has measured it at these shapes.

## W1/W2/W3 — the instrument

`tools/bench/ltx2_connector_gemm_probe.cpp`, in the shape
`tools/bench/conv1d_scaling_probe.cpp` and `tools/bench/bpe_encode_cost.cpp`
already establish: a probe the build compiles, that CI runs never, and whose own
header carries the run recipe. It has three modes.

- `--mode tier` prints `vt::cpu::ElemGemmTierName()` and the resolved
  `mr`/function-pointer table. This is the **resolver's own output at runtime**,
  which is what W1 asks for and what reading `BuildTier()` cannot give.
- `--mode gemm` runs `vt::MatmulBT` at the connector's exact shapes — `M = 1024`
  and the `(N, K)` pairs the checkpoint's own config implies — in three
  orientations: as shipped, over a `[K,N]`-repacked weight with
  `elem_kn_repacked` set, and under `VT_CPU_MATMUL_TIER=ref`. It reports seconds
  and GFLOP/s per shape and **asserts every arm bit-identical to the reference
  arm**, so a rate is never reported for a kernel that computed something else.
- `--mode connector` runs `Ltx2ConnectorForward` itself at the shipped geometry
  with a configurable layer count, so W3's decomposition is taken on the product
  function rather than on a model of it.

**The shapes are read out of the checkpoint, not assumed.** The
`__metadata__.config` of
`ltx-2.5-22b-dev-transformer-bf16.safetensors` (the manifest-pinned DiT) gives
`connector_num_attention_heads = 32`, `connector_attention_head_dim = 128`,
`connector_num_layers = 8`, `audio_connector_attention_head_dim = 64`,
`connector_apply_gated_attention = true`, `connector_num_learnable_registers = 128`,
`rope_type = split`, `frequencies_precision = float64`. So the video stream is
`inner_dim = 4096` and the audio stream `inner_dim = 2048`, both 8 layers, and
the per-call GEMM total is `12 * dim^2 * rows` per layer:

```
video  8 * 12 * 4096^2 * 1024 = 1.6492e12 MAC
audio  8 * 12 * 2048^2 * 1024 = 0.4123e12 MAC
total                           2.0615e12 MAC = 4.123 TFLOP
```

which is the 4.2 TFLOP #2354 predicted in advance, now read off the checkpoint
instead of estimated.

**The symbol-level proof is `perf`, not a counter.** A counter added to
`MatmulOneChunk` would be a product-code change made to answer a question about
product code, and it would only report the sites it was added to. `perf record -e
cpu-clock` over the probe names the executed symbols — `Bt16*`, `BtM*`, `Nk*`,
`MatmulOneChunkRef`, `AttentionCrossKernel` — and needs no PMU, so it runs inside
a VM guest and inside a lease.

**No full CMake build is possible on this devbox.** `/` is at 99% with 4.7 GB
free and a bare `ninja` writes 9.4 GiB, which is the ENOSPC that has previously
produced FALSE policy refusals in this tree's records. The probe therefore also
carries a **direct `g++` recipe over its own translation-unit set** in its
header, which is the `bpe_encode_cost.cpp` precedent verbatim. The CMake target
exists so the file cannot rot behind a `vt::MatmulBT` or `Ltx2ConnectorForward`
signature change; the recorded runs are taken from the direct compile.

## Tests to port

There is no upstream test. Upstream's connector is a `torch.nn.Module` and its
GEMM is cuBLAS. The tests are this tree's own and each is an executable
observable:

| ID | Assertion | Red before |
|---|---|---|
| T1 | `vt::MatmulBT` over a `[K,N]`-repacked weight with `elem_kn_repacked` set is **byte-identical** to the same GEMM over the un-repacked `[N,K]` weight, at the connector's own shapes | nothing asserts it at a non-square `(N != K)` shape |
| T2 | `ElemRepackWeight` followed by the repacked GEMM equals the direct GEMM for f32 at `N != K` and at a K that is not a multiple of the lane count | — |

T1 is the guarantee any repair in W4 would rest on, written as byte equality
rather than a tolerance, because both orientations accumulate each output over K
in strict increasing order and therefore have no tolerance to argue about.
`tests/vt/test_ops_matmul_elem.cpp` already asserts the tier-vs-reference
identity; what it does not assert is the **orientation** identity at a shape the
connector actually uses.

## Gates

1. `--mode tier` on each architecture measured, output recorded verbatim.
2. `--mode gemm`, `n >= 3` per arm, spread stated, every arm asserted
   bit-identical before any rate is quoted.
3. `perf record` symbol attribution over `--mode connector`, recorded as the
   executed proof of W1.
4. Only if W4 lands a change: `ninja test_ops_matmul_elem` green, the mutation
   for each claimed guarantee, and `scripts/ltx25-text-cond-ab.sh`'s
   `pixel_files_differing=0` plus the #1864 blockiness gate under a lease.
5. `scripts/agent-preflight.sh`.

## Risks/decisions

- **This devbox is not the measured machine.** The 37 GFLOP/s was taken on
  `dgx:gpu0`, a GB10 whose CPU is aarch64; this devbox is an AMD Zen 5 with
  AVX-512. The tier tables differ, `mr` differs (4 on NEON, 2 on SSE2, whatever
  AVX2/AVX512 fill), and an absolute GFLOP/s taken here **is not** a statement
  about the GB10. Where this row quotes a rate it names the machine. The x86
  numbers answer "does the orientation matter" and "is the specialized kernel
  entered"; only a run on the GB10 answers "is 37 GFLOP/s what those cores do".
- **A lease is the only way to reach the GB10 and it is for the CPU.** `rc` is
  the mutex; never `ssh`. The probe needs no GPU, so the lease is short and its
  cost is queue wait, not runtime.
- **A microbenchmark is warm and the render is not.** The connector streams 8 GB
  of f32 weights once per call; a probe that loops one shape measures it out of
  L2/L3. Both forms are reported and the warm one is labelled as warm.
  `warm-probe-loops-are-l2-artefacts` is the failure being avoided.
- **`n = 3` bounds a spread; it does not establish a distribution.** A same-arm
  control is run and reported beside every comparison, because a gap smaller than
  the same-arm spread is not a result.

## Evidence

- `--mode tier` output on each machine, verbatim.
- `perf` symbol table over the connector run.
- Per-shape seconds and GFLOP/s for each orientation, `n >= 3`, spread stated,
  with the bit-identity assertion's own output beside them.
- The machine each number was taken on, its core count, and its clock.
- If W4 lands anything: red-before, green-after, the mutation, byte equality and
  the blockiness verdict.

## Stop conditions

Stop and report, do not work around:

- a speedup that cannot be made bit-exact — report it, never trade correctness
  for it (`AGENTS.md` `## Gates`);
- an unhealthy or unreachable fleet device;
- a same-arm spread that swallows the effect being claimed;
- ENOSPC. The disk is at 99% and a build that fills it makes unrelated checkers
  emit false policy refusals.

## Work breakdown

- **W1** — this spec, the probe, the tier proof.
- **W2** — the orientation and tier rate measurements.
- **W3** — the `Ltx2ConnectorForward` decomposition.
- **W4** — the repair, or the measured negative and the next hypothesis.

## Now

`DONE` on what it set out to measure. W1, W2 and W3 are complete on x86-64 AND
on aarch64, including GB10 itself. W4 is a **measured redirection** rather than a
repair: the lever this row sized is a `vt` seam change owned by
`VT-CPU-ELEM-DISPATCH`, and `## Owed` says so.

## Outcome

### The headline, stated before the evidence because it overturns a landed record

**`conditioning.connector.compute` is not dominated by the GEMM.** Two
independent instruments, sharing no code, put `vt::AttentionCross` at **54% to
67%** of `Ltx2ConnectorForward` against the specialized GEMM micro-kernels at
**25% to 35%**, across three machines, two architectures and five load regimes.
**On GB10 -- the machine the render was measured on, idle -- it is attention
66.5%, GEMM 24.8%, everything else 8.7%.** The connector's attention performs
**4.0% of the layer's arithmetic and takes two thirds of its time**; it costs
**2.69x the GEMM** there. The tier that runs is `neon`, not the reference tile,
and the margin is **54x**.

**So #2354's 37 GFLOP/s is not the GEMM's rate.** That number is
`leaf_seconds / gemm_flops`, which is the GEMM's rate only if the leaf is the
GEMM. **On GB10 itself -- the machine the render was measured on -- the probe
reproduces it: `implied_GFLOPs=36.8` against the render's 37.2, while the GEMM
inside that leaf runs at 129.3 GFLOP/s.** The two constructions agree to 1%
because they are the same construction. The
agreement between #2354's predicted 34 GFLOP/s and its measured 37 was real and
was a coincidence of construction: both sides divided the whole leaf by the
GEMM's flops, so both were bound to agree whatever else was in the leaf.

`a-number-quoted-often-becomes-treated-as-measured` is the shape, and this row is
where it is caught, one row after it was written.

### W1 — which kernel runs, proved by execution

`--mode tier`, run on the devbox, verbatim:

```
tier_name=avx512
elem_gemm_use_ref=0
mr=6
f32_bt=set f32_nk=set f32_btm=set f32_nkm=set
elem_kind_of_f32=1
repack_eligible_f32_4096x4096=1
```

and the resolver moves when told to, which is what says it was read rather than
printed from a constant: `VT_CPU_MATMUL_TIER=ref` gives `tier_name=ref`,
`elem_gemm_use_ref=1`; `=portable` gives `tier_name=portable`, `mr=4`,
`f32_btm=NULL`.

**The symbol-level proof is a `perf` profile of `Ltx2ConnectorForward` itself**,
not of a model of it. `sudo perf record -e cpu-clock -F 199`, two video layers,
devbox at loadavg 10:

| symbol | self |
|---|---:|
| `LoadF32(Tensor const&, long)` | 23.09% |
| `vt::SizeOf(vt::DType)` | 20.49% |
| `BtM6Avx512<kF32>` | 17.40% |
| `AttentionCrossKernel(...)::{lambda(long, long)#1}` | 10.74% |
| `Bt16Avx512<kF32>` | 6.81% |
| `Transpose16(__m512*)` | 6.75% |
| `Threadpool::Barrier()` | 6.10% |
| `Threadpool::PollForWork(...)` | 4.32% |
| `__memmove_avx512_unaligned_erms` | 1.11% |

**`BtM6Avx512` and `Bt16Avx512` ARE the tier's f32 `[N,K]` entry points, so the
specialized kernel is what executes.** `MatmulOneChunkRef` does not appear
anywhere in the flat profile down to a 0.05% limit; `MatmulOneChunk<true>` shows
0.10%, which is the driver frame whose work is in the two inlined tier calls
above it. The `Transpose16` line is the BT orientation's own 4x4-group register
transpose, which is what says the call took `MatmulChunked<true>` rather than the
repacked branch.

An earlier profile of ONE layer taken while the box was at loadavg 40 gives the
same ranking with the threadpool terms inflated (Barrier 14.74%, PollForWork
2.72%) and everything else within 3 points. Both are recorded because the
difference between them is the contention, not the finding.

### W2 — the rate, and the orientation lever refuted on this architecture

`--mode gemm`, M = 1024, the connector's own `(N, K)` set, arms **interleaved**
with a same-arm control leg, n = 3, devbox at loadavg 15-27 (**not idle, and
that is stated rather than smoothed**):

| | seconds for one `RunConnector` call | rate |
|---|---:|---:|
| `MatmulChunked<true>`, the shipped orientation | **26.924** | **153.3 GFLOP/s** |
| `MatmulChunked<false>` over an `ElemRepackWeight`-ed `[K,N]` weight | 55.435 | 74.4 GFLOP/s |

**The repack is 2.06x SLOWER in THAT run -- 1.78x on the idle replicate below,
and 1.78x to 2.06x across all three load regimes -- and every shape is
byte-identical**, so this
is a layout result and not a numerical one. Per shape the ratio is 1.47x to
1.84x on the four large shapes; the same-arm control ran at 0.96x to 1.17x, so
the effect is far outside its own control's spread. The two `heads x dim`
projections (N = 32) go the other way and are 0.4% of the call's flops.

**This qualifies a record.** `include/vt/quant.h` states the elementwise repack
as "measured 1.16x to 1.30x on dgx and BYTE-IDENTICAL". The byte-identity half
reproduces exactly. The speed half does not generalize: it was measured on
aarch64 at another row's shapes, and on x86-64 AVX-512 at the connector's shapes
the same lever is a 2x regression. The plausible mechanism is that the AVX-512
tier's `Transpose16` costs less than the `[K,N]` path's 16 KB-strided weight
walk, which is exactly the trade that inverts between ISAs. **This row DID edit
that header, once the aarch64 run made the question decidable** -- on dgx itself
the lever is 1.22x to 2.70x SLOWER at these shapes, so the sentence keeps its
number and gains the scope it lacked. See `### THE aarch64 RUNS`. The paragraph
above is left as it was written, when the answer was not yet in.

### W3 — the decomposition, and it closes

`--mode connector` and `--mode attn`, same binary, same box, same hour:

| | video (dim 4096) | audio (dim 2048) | x8 layers, both streams |
|---|---:|---:|---:|
| `Ltx2ConnectorForward`, one layer | 8.132 s (spread 12.04%) | 3.366 s (7.18%) | **91.98 s** |
| `vt::AttentionCross` at that layer's shape | 5.361 s (3.07%) | 2.966 s (7.69%) | **66.62 s** |
| the layer's six GEMMs | — | — | **26.92 s** |

**66.62 + 26.92 = 93.54 against 91.98 measured.** These were taken under
contention, and the idle replicate below **supersedes them**: on an idle box the
two named legs are 61.3% and 26.6% and there IS a third term, 12.1%, which is the
per-layer `RmsNormRows`, RoPE, gelu and residual-stream copies. The contended
pair is left standing as it was measured rather than edited away, because the
direction of its error is itself a finding and the idle section states it.

**The efficiency gap is the finding, and it survives at both load levels.** One
video layer's attention is 1.718e10 FLOP against the layer's 4.123e11 of GEMM --
**4.2% of the arithmetic** -- and it takes **58.3% of the layer idle, 65.9%
loaded**. Measured rates idle: the GEMM at **209.6 GFLOP/s**, the attention at
**4.5 GFLOP/s**. A 47x gap between two kernels in the same loop.

**One lever that looks obvious is closed by measurement rather than by argument.**
After `Ltx2ConnectorReplaceRegisters` the additive mask is ALL ZEROS, so a reader
naturally asks whether routing that case to the unbiased `vt::Attention` -- which
`Ltx2Attention` already selects when `args.bias == nullptr` -- would help. It
would not: at the same shape the unbiased op costs **0.94x to 1.03x** of the
biased one, inside both arms' own spread, because `AttentionKernel` reads its
operands through the identical per-element `LoadF32`. Routing on the VALUES would
also be exactly what `ltx2.cpp` refuses in writing ("Route on what the call
MEANS, never on what its numbers happen to be"). It is recorded here so the next
row does not spend a day on it.

**The attention timings move by about 10% between quiet runs and that is stated
rather than smoothed.** Two separate processes on a box at loadavg 10-15 read the
video kernel at 5.361 s and 4.828 s, and the audio kernel at 2.966 s and 2.294 s.
The SHARE this row rests on -- attention against the whole layer -- is taken from
figures measured in the same session, and the hoisted-vs-shipped comparison is
taken inside a single process, which is why neither depends on that drift.

### The IDLE replicate, which supersedes the numbers above and corrects one of them

Every figure in W2 and W3 was taken with other agents compiling on the same 20
cores. At 00:54Z the box reached **loadavg 1.24** for the first time in this
session, and the whole set was retaken there. **These are the primary numbers.**

| | idle | at loadavg ~20 | at loadavg ~45 |
|---|---:|---:|---:|
| connector GEMM set, `MatmulChunked<true>` | **19.688 s / 209.6 GFLOP/s** | 26.924 s / 153.3 | 45.027 s / 91.6 |
| the same over a `[K,N]` repack | 35.041 s / 117.8 | 55.435 s / 74.4 | 88.066 s / 46.9 |
| **`kn/bt`** | **1.78x** | 2.06x | 1.96x |
| `Ltx2ConnectorForward`, video layer | **6.567 s** (spread 2.82%) | 8.132 s (12.04%) | 12.874 s (4.48%) |
| `vt::AttentionCross`, video shape | **3.831 s** (6.16%) | 5.361 s (3.07%) | 7.350 s (12.41%) |

**The repack regression is the most robust thing this row measured.** `kn/bt`
reads 1.78x, 1.96x and 2.06x across three independent load regimes, every shape
byte-identical every time. Whatever else contention did, it did not manufacture
this.

**The idle decomposition of one `RunConnector` call** (8 layers, both streams):

| | seconds | share |
|---|---:|---:|
| `Ltx2ConnectorForward` total | **74.11** | 100% |
| ~ `vt::AttentionCross` | **45.42** | **61.3%** |
| ~ the GEMMs | **19.69** | **26.6%** |
| ~ everything else | 9.00 | 12.1% |

**This CORRECTS the 1.7% closure claimed earlier in this section, and the
direction of the error is instructive.** Under contention the two legs that
matter -- attention and GEMM -- are both 20-thread work, while the residue
(`RmsNormRows` with its f64 accumulator, `Ltx2ApplyRotaryEmb`, the gelu, and
three full `std::vector<float>` copies of the 16 MB residual stream per layer) is
largely SINGLE-threaded and therefore inflates far less. So contention
overstated the two measured legs relative to the residue and made the sum look
tighter than it is. On an idle box the residue is **12.1%**, not 1.7%, and it is
real work rather than measurement slack.

**Every conclusion of this row survives the correction, and one number moves.**
Attention is still **2.3x the GEMM's cost** and still the largest term. The GEMM
still runs at **209.6 GFLOP/s** against the leaf's implied 62.8, so
`leaf_seconds / gemm_flops` is still not the GEMM's rate -- the gap is 3.3x on an
idle box rather than 3.0x on a loaded one. What moves is the hoisted-reference
margin: on an idle box one thread costs **3.174 s** against the shipped kernel's
**3.831 s** on twenty, a **1.21x** lead rather than the 1.42x measured under
load, because the shipped 20-thread kernel gains more from an idle box than a
single-threaded reference can. The claim "one thread beats twenty" holds; the
margin is smaller and the smaller number is the one to quote.

The unbiased-attention control also survives: 0.96x video, 1.15x audio, still
inside the arms' own spread and still no lever.

### Why the attention kernel is 47x off, and what the repair is

`AttentionCrossKernel` (`src/vt/cpu/cpu_ops.cpp`) reads every operand element
through `LoadF32(const Tensor&, int64_t)`, which switches on `t.dtype` and
computes its byte offset with `vt::SizeOf(t.dtype)`. **`vt::SizeOf` is an
out-of-line function in `src/vt/dtype.cpp` and the build enables no LTO**
(`CMakeLists.txt` sets no `INTERPROCEDURAL_OPTIMIZATION` and passes no `-flto`),
so it is a cross-translation-unit call that cannot be inlined away. The profile
shows the consequence directly: `LoadF32` 23.09% plus `SizeOf` 20.49% is
**43.6% of a connector layer spent resolving an element type and an address**,
inside a loop whose body is one multiply and one add.

**That attribution is proved, not inferred.** A `perf` profile of `--mode attn`,
which runs `vt::AttentionCross` and the probe's hoisted reference and NO GEMM at
all, isolates it:

| symbol | self |
|---|---:|
| `LoadF32(Tensor const&, long)` | 36.14% |
| `vt::SizeOf(vt::DType)` | 28.41% |
| `AttentionCrossKernel(...)::{lambda(long, long)#1}` | 15.60% |
| `Threadpool::Barrier()` | 8.18% |
| `ModeAttn(long, int)` -- the hoisted reference, inlined | **4.76%** |
| `Threadpool::PollForWork(...)` | 4.45% |

**64.6% of the attention kernel's own CPU time is resolving an element type and
an address.** The arithmetic and the softmax are the 15.60% line. And the last
column is the same profile's own control: the hoisted reference computes the
identical output for **4.76%** of the process's CPU against the shipped kernel's
80.15%, so the shipped kernel burns **16.8x the CPU for the same result** --
measured in ONE process, with no cross-run drift to argue about.

**The repair is the transformation `MatmulOneChunk` already applies against
`MatmulOneChunkRef`:** resolve the element type once, outside the loops, and walk
typed pointers. It touches no output's accumulation order -- the same indices are
summed in the same sequence -- so it is **bit-exact, not merely close**.

`--mode attn` prices it. `AttnCrossHoisted` in the probe is that transformation,
written in the probe rather than in product code so the headroom could be
measured before anything was changed:

| | shipped kernel, 20 threads | hoisted reference, ONE thread | equality |
|---|---:|---:|---|
| video, heads 32, d 128 | 5.361 s | **3.775 s** | byte-equal |
| audio, heads 32, d 64 | 2.966 s | **2.161 s** | byte-equal |

**A single thread with the dtype hoisted beats twenty threads without it**, on
both streams, with `memcmp`-identical output. That is the headroom, measured, and
it bounds nothing from above: the hoisted form here is scalar and unthreaded.

### The headroom, measured at the shipped thread count

The single-threaded reference above understates the ceiling, because it compares
one thread against twenty. The probe now carries the same hoisted transformation
**on the same thread count the shipped kernel uses**, partitioned over the (head,
query) pairs exactly as `AttentionCrossKernel`'s own `ForRows` partitions them --
each output row independent, each reduction sequential over the same indices, so
it is bit-exact for the same reason:

| | shipped, 20 threads | hoisted, 20 threads | speedup | equality |
|---|---:|---:|---:|---|
| video, heads 32, d 128 | 3.858 s (4.5 GF) | **0.290 s (59.2 GF)** | **13.30x** | byte-equal |
| audio, heads 32, d 64 | 1.961 s (4.4 GF) | **0.205 s (41.9 GF)** | **9.58x** | byte-equal |

**RETRACTED: "the shipped kernel on twenty threads is slower than the hoisted one
on ONE" (3.858 s against 3.261 s).** That held HERE, on AVX-512, and this row
published it as the cleanest statement of the defect's size. It is FALSE on both
aarch64 hosts and it is withdrawn -- see `### RETRACTION` below for the numbers
and for why it was an AVX-512 artifact. The x86 table above stands; the sentence
generalizing from it does not.

**The conditional that stood here is now MEASURED and has moved.** This section
extrapolated "if the aarch64 split resembles the x86 one" to a 2.27x on the
connector and a ~392 s render. It no longer has to: `### The headroom, on GB10`
below measures the split on GB10 itself and reads **2.55x** and **~380 s**. The
conditional is superseded there rather than restated here.

### THE aarch64 RUNS — the machine the render was measured on

Two leases completed and **the primary one is GB10 itself**: `rc-worker-4b8lj`,
2026-08-31T02:33:18Z, `uname=aarch64`, **Cortex-X925, 20 cores**, 119.7 GiB
`MemAvailable`, **loadavg 1.12 at start** -- an idle box, and the same hardware
`LTX25-RENDER-SPEED-PARITY` and #2354 measured the render on. A second,
`rc-worker-n8smh` (14 cores), is a smaller aarch64 part and is reported beside it
because two hosts agreeing is worth more than one.

**W1 is now answered on the machine that matters, and it is not the reference
tile.** `tier.txt`, verbatim:

```
tier_name=neon
elem_gemm_use_ref=0
mr=4
f32_bt=set f32_nk=set f32_btm=set f32_nkm=set
elem_kind_of_f32=1
repack_eligible_f32_4096x4096=1
```

**The margin is 54x, so this is not a close call.** Forced onto the reference
tile the same GEMM set takes **1738.202 s at 2.4 GFLOP/s**; the tier that
actually runs takes **31.906 s at 129.3 GFLOP/s**. The portable tier takes
119.390 s at 34.6 GFLOP/s, so NEON is 3.74x portable.

**And 34.6 is a trap worth naming.** The portable tier's rate is within 7.5% of the
37.2 GFLOP/s #2354 derived from the render, so a reader checking "are we on a
slow tier?" against that number alone would conclude yes. The tier is `neon` and
the GEMM runs at 129.3. The coincidence is arithmetic, not evidence.

**The decomposition on GB10, idle** (8 layers, both streams):

| | seconds | share |
|---|---:|---:|
| `Ltx2ConnectorForward` total | **128.808** | 100% |
| ~ `vt::AttentionCross` | **85.704** | **66.5%** |
| ~ the GEMMs | **31.906** | **24.8%** |
| ~ everything else | 11.198 | 8.7% |

**The closing argument: the probe reproduces #2354's number from the render.**
One video layer reads `implied_GFLOPs=36.8` against the render-derived **37.2**,
on the same architecture, from an independent code path. `leaf_seconds /
gemm_flops` on this machine gives 36.8 while the GEMM inside that leaf runs at
**129.3 GFLOP/s**. The two constructions agree to 1%, and they agree because they
are the same construction -- not because 37 was ever a kernel rate.

**The repack regression replicates on the machine `include/vt/quant.h` names.**
`kn/bt` per shape on GB10: 1.72, 2.06, 1.57, 1.25, 1.51, 1.76, 2.70, 1.22 --
**1.806x overall**, byte-equal on every shape, with same-arm controls at 0.90 to
1.04. Thor reads 1.389x overall. Across two architectures and five load regimes
the repack has never once been faster at these shapes.

**Thor, the second aarch64 host**, agrees on every qualitative point:
`tier_name=neon`, `mr=4`, GEMM 29.337 s / 140.7 GFLOP/s, `kn/bt` 1.389,
connector 84.608 s of which attention is 48.200 s (**57.0%**) and GEMM 29.337 s
(34.7%).

**The unbiased-attention lever stays closed on aarch64 too**: 1.023 and 0.993 on
GB10, 0.974 and 0.986 on Thor.

**A cross-ISA equality signal, stated with its limit.** The connector's printed
checksums are identical on x86-64 AVX-512 and on both aarch64 NEON hosts
(`3.080419` video, `1.145383` audio). That is consistent with the tier
bit-exactness contract holding across ISAs. It is **two floats at six decimal
places**, not a golden, so it is corroboration and not a gate.

### RETRACTION: "the shipped kernel on twenty threads is slower than the hoisted one on ONE"

**That claim is FALSE and it is withdrawn.** It was measured on x86-64 AVX-512,
where it held (hoisted 1 thread 3.261 s against shipped 20 threads 3.858 s), and
this row published it as "the cleanest statement of the defect's size". On
aarch64 it inverts:

| | hoisted, 1 thread | shipped, all threads | verdict |
|---|---:|---:|---|
| x86-64 AVX-512, 20 cores | 3.261 s | 3.858 s | claim HELD |
| **GB10 Cortex-X925, 20 cores** | **8.051 s** | **7.181 s** | **claim FALSE** |
| Thor, 14 cores | 6.163 s | 4.015 s | claim FALSE |

**It was an AVX-512 artifact.** The hoisted reference is written as scalar C++
and the compiler auto-vectorizes its inner dot products; AVX-512 gives it 16
lanes against NEON's 4, so on x86 one hoisted thread could out-run twenty
un-hoisted ones and on Arm it cannot. The generalization was mine, not the
measurement's, and it was made from one architecture.

**What survives is the claim that was always the load-bearing one**, and it
survives on every machine measured: the hoisted transformation at the SHIPPED
thread count is **11.30x** (video) and **12.12x** (audio) on GB10, **8.86x** and
**9.36x** on Thor, **13.30x** and **9.58x** on x86 -- byte-equal every time. The
defect is real, large, and architecture-independent. The retracted sentence
overstated how it presents on one ISA.

### The headroom, on GB10

| | shipped, 20 threads | hoisted, 20 threads | speedup | equality |
|---|---:|---:|---:|---|
| video, heads 32, d 128 | 7.181 s (2.4 GF) | **0.635 s (27.0 GF)** | **11.30x** | byte-equal |
| audio, heads 32, d 64 | 3.532 s (2.4 GF) | **0.292 s (29.5 GF)** | **12.12x** | byte-equal |

**Projected onto the render, now from the right machine.** Attention falls from
85.704 s to 7.416 s, so a `RunConnector` call goes 128.808 s -> **50.520 s**, a
**2.55x** on connector compute. #2354 measured that leaf at 224.882 s in a
516.751 s render, so the same ratio puts the render near **380 s** and the oracle
gap near **4.05x** instead of 5.51x.

**This is still a projection and its assumptions are stated.** The probe's own
connector total (128.808 s) is 14% above the render's measured leaf (112.768 s)
-- synthetic weights, a different valid-token count, and video and audio timed in
separate processes -- so the RATIO is what transfers, not the seconds. No render
was run.

### W4 — a redirection, and why no kernel was changed

The repair is obvious, bit-exact, and prototyped byte-equal. **It is not landed
here, and that is a scope decision rather than a lack of one.**

`AttentionCrossKernel` is a `vt` shared seam. Every model that reaches
`vt::AttentionCross` runs it -- the LTX-2.5 DiT among them -- and
`AttentionKernel` beside it carries the identical defect. Doing this properly is
the `CPU-ELEM-GEMM` shape: keep the current scalar kernel as the reference arm,
add a typed one, gate the two byte-identical, and put a same-binary A/B switch
between them. `AGENTS.md` `## Changing the rules or a checker` and `## Spec
before code` both point that work at its own row with its own spec and a fresh
reviewer, and this row's own scope excludes it.

Three further reasons, each measured rather than asserted:

- ~~**This devbox is not the machine the render was measured on.**~~ **ANSWERED.**
  Every number in the sections above is x86-64 AVX-512, and the prediction made
  here -- that GB10's tier is NEON with `mr = 4` and a weaker GEMM -- is exactly
  what `tier.txt` reports. The split there is no longer an open question:
  `### THE aarch64 RUNS` measures it on GB10 at attention 66.5% / GEMM 24.8%, and
  the reason to keep this bullet is that its LAST clause -- "this row does not
  guess at it" -- is the one the retraction below shows should have been applied
  to the single-thread claim as well.
- **The disk cannot hold a CMake build tree** (`/` at 99%), so the project's full
  suite could not be run against a change to a seam every model uses. What COULD
  be run is `tests/vt/test_ops_attention_cross.cpp`, compiled directly:
  **21 cases / 33 assertions / 0 failed** on this tree, which is the baseline the
  next row starts from and is recorded here so it does not have to re-derive it.
- **The existing byte-identity gate for the orientation lever already exists**
  and needed nothing added. `test_ops_matmul_elem.cpp`'s "load-time [N,K]->[K,N]
  repack is byte-identical" covers T1 for three dtypes at five ragged shapes, and
  the probe's own per-shape `memcmp` extends it to the connector's shapes. **No
  new test was written, because the guarantee was already gated** and a second
  copy is how two rules start.

### What could not be measured

- **The aarch64 side is now MEASURED** and `### THE aarch64 RUNS` carries it.
  This bullet is kept, struck through by that section rather than deleted,
  because it is what the row could honestly say before the leases landed. The
  jobs that answered it are `rc-worker-4b8lj-20260831T023318Z` (GB10) and
  `rc-worker-n8smh-20260831T020213Z` (Thor). The original text follows.
  `rc` jobs `ea2631f3-aed2-46df-b419-3628078f9882`
  (`dgx:gpu0`) and `75800c9e-0b55-406e-9958-ba0048a5a751` (`thor:gpu0`) were
  submitted with the full probe and were still queued at positions 5 and 3 when
  this row was written. `orin:gpu0` was tried first and refused: its `/workspace`
  is local to that host and is NOT the shared NAS the other two mount, which is
  recorded here because it is not written anywhere else.

  **The first aarch64 run LINK-FAILED, and the job design is why that cost
  minutes instead of the whole slot.** `rc` job
  `75800c9e-0b55-406e-9958-ba0048a5a751` reached `thor:gpu0` (`rc-worker-n8smh`,
  2026-08-31T00:42:22Z) -- **aarch64, 14 cores, `sve2 i8mm bf16 svebf16`, max
  2601 MHz, 118 GiB free, loadavg 4.84**. It failed at the link:

  ```
  cpu_quant_repack_arm.cpp:230: undefined reference to
    vt::cpu::InterleaveQ8_0Rows4(...)
  LINK FAILED (vt)
  ```

  The cause is a translation-unit pair this row's hand-picked list split.
  `cpu_quant_repack_arm.cpp` defines `QuantRepackActive` and CALLS
  `InterleaveQ8_0Rows4`; `cpu_quant_repack.cpp` defines `InterleaveQ8_0Rows4` and
  CALLS `QuantRepackActive`. They are mutually complete and `CMakeLists.txt`
  compiles BOTH unconditionally, on every architecture. The recipe compiled the
  `_arm` one only on aarch64 and the other one nowhere.

  **The x86 validation had not caught it, and the reason is worth keeping.** The
  first fix -- adding `cpu_quant_repack.cpp` -- then failed the x86 validation on
  `QuantRepackActive`, which is what revealed that the pair has to be on both
  arches rather than that one file was missing. **Each arch caught a different
  half of the same defect**, so neither validation alone was sufficient and the
  aarch64 failure was worth its minutes.

  A wide-TU link fallback was written, TRIED, and REMOVED rather than shipped:
  every non-per-ISA `src/vt/cpu` TU drags in `cpu_quant_gemm.cpp`, which needs
  `QuantMmlaVecDot` from an ISA-gated file, so the fallback failed its own
  validation. An untested recovery path is not a recovery path, and shipping one
  would have put a second failure mode in front of the measurement.

  The corrected recipe (`run.sh` sha256 `242b8c8d...`) builds and runs W1 end to
  end on x86-64 from the staged tarball. `rc` job
  `6280a660-dd97-471e-bc7f-ef6b0bc62def` re-queues `thor:gpu0`, and
  `ea2631f3-aed2-46df-b419-3628078f9882` on `dgx:gpu0` picks the same file up.
  The failed run's log is left in place as evidence rather than deleted.

  **How to read the result when it lands.** Each run writes `tier.txt`,
  `gemm-default.txt`, `gemm-ref.txt`, `gemm-portable.txt`, and -- if the larger
  binary built -- `connector.txt` and `attn.txt`, under
  `/mnt/nas_share/rc/ltx25-connector-gemm/run/<host>-<stamp>/`, which is
  `/workspace/ltx25-connector-gemm/run/...` from inside the lease. Two questions
  come out of it: whether `tier_name` reads `neon` rather than `ref` or
  `portable`, and whether the `kn/bt` column sits above or below 1 -- the second
  being what decides whether `include/vt/quant.h`'s "1.16x to 1.30x on dgx" needs
  a scope note or a correction.

  **The job builds two binaries, smallest first, and runs the small one before
  attempting the large one.** The tier and orientation questions need only the
  `vt` runtime; the decomposition additionally needs the LTX-2.5 model TUs and
  their audio-VAE closure. A link failure in the larger set would otherwise cost
  the whole queue wait and answer nothing. When the large one does not build the
  log says so in those words, so an ABSENT W3 cannot be read as a measured one.
  The recipe was validated end to end on x86-64 from the staged tarball before
  the jobs were left to queue and both binaries built, so what is untested in it
  is exactly the aarch64 per-ISA branch, which mirrors `CMakeLists.txt`'s own
  `set_source_files_properties` calls. Staged artifacts: `run.sh` sha256
  `242b8c8d...`, `src.tar.gz` sha256 `59465c4c...`, the latter a `git archive` of
  this row's committed head, so the binary that runs under the lease is built
  from the same probe the branch carries.

  **That validation run left a directory in the evidence tree and it is
  quarantined rather than deleted.** `run.sh` derives its output path from
  `hostname`, and the devbox can see the same share the lease workers mount, so
  the build check wrote
  `run/NOT-A-LEASE-devbox-buildcheck-20260831T003240Z/` beside the real runs --
  with a `README.txt` saying in its first line that it is not a fleet
  measurement and that no number may be read out of it. An x86-64 log sitting in
  a directory of aarch64 evidence is exactly the shape of
  `an-instrument-whose-failure-looks-like-a-result`, and renaming it keeps the
  provenance of the validation this section claims.
- **An idle box.** Every devbox number was taken with other agents compiling on
  the same 20 cores, at loadavg 10 to 40. The RATIOS are what this row rests on
  and each carries its own same-arm control; the absolute GFLOP/s are lower
  bounds.
- **A render.** No end-to-end before/after exists, because nothing changed.

### The gates

- `scripts/agent-preflight.sh`: every checker green, `check-commit-trailers`
  `OK: commit trailer contract` and `check-commit-style` `OK: commit writing
  style` over `origin/main..HEAD` after the merge that made them runnable.
- `tests/vt/test_ops_attention_cross.cpp`, compiled directly and run:
  **21 cases / 33 assertions / 0 failed** (the CUDA cases self-skip with `SKIP:
  no CUDA backend registered`, which is what a devbox with no CUDA reports and
  is recorded rather than counted as coverage).
- **One preflight red was MY INSTRUMENT, and it is written down because a
  broken instrument that fails toward a code verdict is how this repository
  loses a day.** `check-windows-portability.py` reported `PowerShell AST parse
  failed: ... GLIBC_2.38 not found (required by .../libstdbuf.so)`. The cause is
  that this row ran preflight under `stdbuf -oL` to defeat its output buffering,
  and `stdbuf` injects `libstdbuf.so` into every child -- including the snap
  `pwsh` the checker shells out to, which links a different libc. Re-run without
  `stdbuf` the same checker prints `Windows portability contract OK`, exit 0. No
  file in this branch is a PowerShell file.

## Owed

- ~~**THE ATTENTION KERNEL'S PER-ELEMENT DTYPE SWITCH IS UNOWNED.**~~ **IT IS
  OWNED: `VT-CPU-ELEM-DISPATCH` implements the repair this row sized, and its
  spec links this one.** What this row hands it: 43.6% of a connector layer is
  `LoadF32` plus a cross-TU `vt::SizeOf`; the hoisted form is byte-equal at the
  shipped thread count and worth **11.30x / 12.12x on GB10**, 8.86x / 9.36x on
  Thor, 13.30x / 9.58x on x86-64. It is not LTX-2.5-specific -- `AttentionKernel`
  carries the same defect and every CPU attention path pays it.
  **That row measured something this row could not**: inlining `vt::SizeOf` alone
  buys 1.35x to 1.93x, while the hand-hoist is worth 8.75x to 11.16x. So the
  cheap repair is necessary and not sufficient, which is this row's own finding
  one level down -- the obvious suspect is real and is not the whole cost.
  Owner: `VT-CPU-ELEM-DISPATCH`.
- **`include/vt/quant.h`'s "1.16x to 1.30x on dgx" is FIXED IN THIS CHANGE, not
  filed.** The queued aarch64 run decided it: on dgx itself the same lever is
  **1.22x to 2.70x SLOWER** at the connector's shapes, 1.806x overall,
  byte-identical throughout. The sentence is contradicted on the machine it
  names, so it gets a scope qualifier and the counter-measurement beside it
  rather than an owner and a wait. `AGENTS.md`: a record edit rides in the change
  whose measurement made it stale, and this row's measurement is what made it
  stale. The byte-identity half of that sentence reproduced exactly and is
  untouched.
- ~~**The aarch64 numbers.**~~ **DISCHARGED.** Both leases ran.
  `rc-worker-4b8lj-20260831T023318Z` is GB10 (Cortex-X925 x20, idle) and
  `rc-worker-n8smh-20260831T020213Z` is Thor. `### THE aarch64 RUNS` carries the
  result and `### RETRACTION` carries the claim they killed.
- **STILL NO ISSUE, and the reason changed under this row.** The account that
  filed for this campaign (`mudler-agent`) was suspended mid-row, and a suspended
  account's content is **hidden, not deleted**. GitHub now works under
  `localai-org-maint-bot` -- `gh api user` returns it, exit 0 -- and **both of
  this campaign's issues fail to resolve under it**, checked rather than assumed:

  ```
  gh issue view 2296 -> Could not resolve to an issue or pull request
  gh issue view 2354 -> Could not resolve to an issue or pull request
  ```

  **That is `REMOTE_UNVERIFIED` and it is NOT evidence either issue is absent.**
  #2296 and #2354 are cited by `ltx25-render-speed-parity.md` and
  `ltx25-text-cond-device.md`, both of which are ON `main`, so the tree says they
  existed. A reader who takes the 404 at face value would file duplicates of two
  live issues and re-open a campaign that is already recorded; that is the
  failure this bullet exists to prevent, and this row did NOT file anything on
  the strength of it. An
  issue for this row is owed the moment someone can open one against the
  campaign's existing thread. Owner: this row.
