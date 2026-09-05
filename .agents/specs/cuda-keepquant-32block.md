# CUDA keep-quant GEMM for the 32-element encodings — `QUANT-CUDA-KEEPQUANT-32B`

Issue: [#2419](https://github.com/mudler/vllm.cpp/issues/2419).
Branch: `row/QUANT-CUDA-KEEPQUANT-32B`.
Extends [`cuda-keepquant-gemm.md`](cuda-keepquant-gemm.md), whose
`KERNEL-QUANT-CIQ-GEMM-CUDA` owns the CUDA `kMatmulBTQuant` provider, and
follows [`cuda-keepquant-iq2xs-iq4xs.md`](cuda-keepquant-iq2xs-iq4xs.md), which
added the last two Q8_K-activation encodings. Discharges the keep-quant half of
the debt [#2380](https://github.com/mudler/vllm.cpp/issues/2380) records under
`## Owed` for `MODEL-MM-QWEN4-EXP`.

## Now

`DONE` on the kernels and their gates. See `## Outcome`.

## The gap, as measured on `84f6fac0a`

`IsCudaKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu:1815`) maps twelve
`DType`s to a `WType`. Every one is a **256-element super-block** encoding whose
`vec_dot` pairs it with a `BlockQ8_K` activation, and the templated GEMM behind
the predicate (`DotSuperblock<W>(const void*, const BlockQ8_K*)`) takes that
activation type in its signature.

**IQ4_NL, Q5_0 and Q4_0 are absent, and cannot be added by writing a `case`.**
They are 32-element encodings that dot a `BlockQ8_0` activation. The mismatch is
in the activation type and the block extent, so this row adds a *second*
templated GEMM beside the first rather than extending it.

Q8_0 is also absent from the predicate but is **not** part of this gap: it has a
dedicated on-device path (`MatmulQ8_0Cuda`, `MatmulQ8_0GroupedCuda`) dispatched
before the predicate is consulted, carrying its own tuned levers and gates. This
row does not touch it, and does not route it through the new template — doing so
would put the DeepSeek-V4 decode levers behind an untuned generic kernel for no
gain this row measures.

### What the three consumers do today

Line numbers in this section are **at `84f6fac0a`**, the base. This change
inserts a lane above `IsCudaKeepQuantSupported` and moves every one of them, so
read them against the base rather than the merged tree.

| Seam | Today, for IQ4_NL / Q5_0 / Q4_0 |
|---|---|
| `MatmulBTQuantKernelCuda:2076` | `cudaStreamSynchronize` then `GetOp(kMatmulBTQuant, kCPU)` over the same tensors. |
| `MatmulBTQuantGroupedKernelCuda:2170` | the same drain-and-fall-back. |
| `MoeGateUpSwiGLUGroupedCuda:2402` | **throws** `moe_gate_up_swiglu: gate/up must be the SAME CUDA keep-quant dtype`. |

The fallback is correct and host-speed where `Backend::Alloc`'s memory is
host-addressable. On CUDA it is a plain `cudaMalloc` (`cuda_backend.cu:104-108`),
so where it is not, the CPU kernel dereferences a device pointer — the SIGSEGV
`cuda-keepquant-iq2xs-iq4xs.md` measured on GB10 for the previous two dtypes.
Either way the sync **invalidates a decode graph capture**, so the fallback is
not merely slow: it is unusable by the captured decode path.

### Why the released artifact needs exactly these

`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S stores its routed-expert towers in
IQ4_NL and Q5_0. That is forced, not chosen: `expert_feed_forward_length` is
**640**, so every routed expert row is indivisible by 256 and no K-quant can
encode it. `qwen4_exp_gguf_weights.cpp:117-119` states this in the tree.

The consequence is sharper than a missing `case`. `ffn_down_exps` is `[E*H, I]`
with `I = 640`, so **K = 640**. The Q8_K seams reject that shape by name
(`K must be a whole number of 256-element Q8_K super-blocks`) *regardless of
dtype*. The 32-element lane is the only lane this checkpoint's MoE can use.

## Scope

**In.** A `BlockQ8_0`-activation templated GEMM for IQ4_NL, Q5_0 and Q4_0:
device dot, dense kernel, grouped kernel, fused grouped gate+up+SwiGLU kernel,
and the dispatch in all three seams above.

**Out.** MXFP4 (the fourth 32-element encoding, `ffn_down` of the DeepSeek-V4
UD-IQ2_M arm). Its device math (`DotMXFP4`) is already written and unreferenced,
so the template this row adds is what it needs — but no artifact this row gates
uses it, and adding an ungated arm is how a dead path lands. Recorded under
`## Owed`.

**Out.** Vectorising the dot with `__dp4a` over packed nibbles (llama.cpp's
`vec_dot_q4_0_q8_1_impl` shape). This row's kernels are scalar loops that mirror
the CPU oracle statement for statement. That is a real throughput ceiling and it
is recorded under `## Owed`; it is not a correctness question, and landing a
correct on-device arm is what removes the host drain.

**Out.** `gguf_keep_quant.cpp::DeviceKeepQuantSupported` and `cuda_ops.cu`
(#2396), the four `qwen4_exp` op kernels (#2391), `qwen4_exp_qsa_block.cpp`
(wave QSADEV).

## Upstream anchors

vLLM implements none of these three `vec_dot`s — it has no GGUF k-quant CPU
reference tier at all — so the secondary-oracle rule applies exactly as it did
for the CPU arm and for `cuda-keepquant-iq2xs-iq4xs.md`. Oracle: llama.cpp
`b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`
([`oracles/llama-cpp.md`](../oracles/llama-cpp.md)).

The **behavioural** oracle for the gate is closer than upstream: it is this
tree's own CPU arm, the same kernel the host drain runs today.

| Ported | From (llama.cpp b10451) | Via (this tree's CPU arm) | To |
|---|---|---|---|
| `ggml_vec_dot_q4_0_q8_0_generic` | `ggml/src/ggml-cpu/quants.c:174` | `cpu_quant_dot.cpp:50 VecDotQ4_0Q8_0` | `cuda_quant_dot.cu::Dot32<W32::kQ4_0>` |
| `ggml_vec_dot_q5_0_q8_0_generic` | `ggml/src/ggml-cpu/quants.c:365` | `cpu_quant_dot.cpp:92 VecDotQ5_0Q8_0` | `cuda_quant_dot.cu::Dot32<W32::kQ5_0>` |
| `ggml_vec_dot_iq4_nl_q8_0_generic` | `ggml/src/ggml-cpu/quants.c:1254` | `cpu_quant_dot.cpp:140 VecDotIQ4_NLQ8_0` | `cuda_quant_dot.cu::Dot32<W32::kIQ4_NL>` |

### The association order is part of the port

The three upstream kernels fold the scales in three *different* orders, and the
CPU arm's comments already say so. They are reproduced rather than normalised:

- Q4_0: `sumf += sumi * F16ToF32(x.d) * F16ToF32(y.d)` — left-associated, the
  integer sum multiplied first.
- Q5_0: `sumf += (F16ToF32(x.d) * F16ToF32(y.d)) * (float)sumi` — scale product
  formed first.
- IQ4_NL: `d = F16ToF32(y.d) * F16ToF32(x.d); sumf += d * (float)(s1 + s2)` —
  scale product first *and* the operands in the opposite order.

`-ffp-contract=off` is CXX-only (`CMakeLists.txt:55`) and never reaches `.cu`,
so nvcc would contract each of these into an FMA that rounds once where upstream
rounds twice. `cuda_quant_dot.cu:620-643` records that being MEASURED here — two
of eight real super-blocks off by 1 and 4 ULP. Every multiply-add in the new
dots is therefore spelled `__fmul_rn` / `__fadd_rn`, as `DotIQ4XS` is.

## Memory format

Checked against the oracle as `.agents/porting.md` requires, because a dtype
that is too wide passes a token gate while moving twice the bytes.

| Encoding | Block | Bytes | Layout |
|---|---|---|---|
| Q4_0 | 32 | 18 | `d` f16 @0, `qs[16]` @2 |
| Q5_0 | 32 | 22 | `d` f16 @0, `qh[4]` @2, `qs[16]` @6 |
| IQ4_NL | 32 | 18 | `d` f16 @0, `qs[16]` @2 |

The kernels read the weight bytes **in place**, at the block stride the loader
already produced, and the accumulator is f32 — the same width as the Q8_K
template's and as the CPU arm's `float sumf`. No dequantised f32 or bf16 copy of
the weight is materialised anywhere on this path.

## Design

A second enum `W32` and a second templated GEMM, beside the Q8_K one:

- `Dot32<W>(const void* wb, const BlockQ8_0* ab)` — one weight block against one
  activation block, mirroring the named CPU function statement for statement.
- `IsCuda32BlockKeepQuantSupported(DType, W32*)` — the new predicate, structured
  exactly like `IsCudaKeepQuantSupported` so the two read as siblings.
- `QuantDotGemm32Kernel<W, OutT>` — dense, one warp per output, lane-strided over
  blocks, 32-lane `__shfl_down_sync` tree reduce. Mirrors
  `QuantDotGemmQ8_0Kernel`.
- `QuantDotGemmGrouped32Kernel<W, OutT>` — weight row `expert_ids[p]*n + j`,
  activation row `bcast ? 0 : p`. Mirrors `QuantDotGemmGroupedQ8_0Kernel`.
- `QuantDotGemmGroupedFusedSwiGLU32Kernel<W>` — gate and up in one warp, the
  same `ClampedSwiGLU` epilogue as the Q8_K fused kernel.

The activation quantiser is **reused, not rewritten**: `QuantizeQ8_0Kernel` /
`QuantizeQ8_0PreqKernel` and the grow-only per-stream `EnsureScratch` are already
the Q8_0-activation prologue, already graph-safe, and already gated.

`K % 32 == 0` replaces `K % 256 == 0` on the new lane, which is what lets K=640
through.

## Risks

1. **A passing token gate proves nothing here.** Addressed by the capture gate
   below; stated again because it is the failure this row exists to make
   visible.
2. **A fixture that never crosses a tile boundary cannot see a cross-tile bug.**
   The dense kernel strides `b = lane; b < nb; b += 32`, so a fixture with
   `nb <= 32` never takes the loop twice and a lane-index defect is invisible.
   The gate therefore includes a shape with `nb > 32`.
3. **Scalar dot is a throughput ceiling.** Named in `## Owed`, not hidden.
4. **Q5_0's `qh` is read as a `uint32_t` from a 2-byte-aligned offset** (@2 of a
   22-byte block). The CPU arm uses `memcpy`; the device dot must not deref a
   `uint32_t*` there. Byte-assembled instead.

## Tests and gates

`tests/vt/test_cuda_quant_dot.cpp`, which is table-driven over `kCases`. Adding
three rows drives every case in the file at once.

**What the gates observe, and what they cannot:**

| Gate | Observes | Cannot observe |
|---|---|---|
| `CUDA keep-quant GEMM == CPU reference and f64 dequant` | that the device numbers match the CPU arm (NMSE) and an independent f64 dequant-and-dot | **whether the device ran at all** — it reads the host fallback as a pass. This is the trap, and this gate is not the discriminator. |
| `CUDA keep-quant runs every kCases dtype INSIDE a stream capture` | **the path.** `cudaStreamSynchronize` on a capturing stream fails, so this is red for exactly the dtypes that drain and green for the ones that do not. Asserts the counted property `captured == std::size(kCases)`. | *which* device kernel ran, or how fast |
| fused-MoE grouped case | that `MoeGateUpSwiGLUGrouped` returns values instead of throwing by name | end-to-end MoE numerics on the real checkpoint |

**MEASURED CORRECTION: on GB10 the capture gate is NOT the observed
discriminator, and this row does not get to claim it was.** The design argument
above stands -- a drained stream cannot be captured -- and it is what makes the
gate's GREEN meaningful after the change. But on the pre-change tree the crash
described above arrives first: `Backend::Alloc` is a plain `cudaMalloc`, the CPU
fallback dereferences a device pointer, and the run dies with SIGSEGV inside the
FIRST case in the file (`test_cuda_quant_dot.cpp:268`, the dense parity gate).
doctest then aborts the binary and reports the remaining **17 cases skipped**,
the capture case at `:1307` among them.

So the capture gate has only ever been observed GREEN here, never RED. The
red-first evidence on this hardware is the crash, not the capture. That is a
stronger red and a weaker gate-attribution at the same time, and both halves are
recorded because only the first half flatters this row.

What would make the capture gate the actual discriminator is a device where
`Backend::Alloc` is host-addressable, so the fallback returns correct numbers at
host speed instead of faulting. No device on this fleet does that, so the claim
stays UNMEASURED rather than false, and it is not offered as evidence.

The parity gate is what says the new kernel is *correct* once it runs.

Neither needs the real checkpoint, and neither can speak for it. What is proven
on a fixture is the dot, the dispatch and the capturability; what is **not**
proven is the released artifact's own tensors flowing through a full forward.
That is stated in `## Owed` rather than implied by a green suite.

## Reachability

`vt::MatmulBTQuant`, `vt::MatmulBTQuantGrouped` and `vt::MoeGateUpSwiGLUGrouped`
are the production entry points; the tests call them, never a kernel directly.
The mutation deletes the new `case` arms from the seam dispatch in a scratch copy
and shows the capture gate's `captured` count fall — a counted property, so a
mutation that never applied cannot read as a pass.

## Owed

- **MXFP4 CUDA keep-quant arm.** `DotMXFP4` exists and is unreferenced; this
  row's template is the seam it needs. Not gated by any artifact this row
  measures. Owner: this row's issue [#2419](https://github.com/mudler/vllm.cpp/issues/2419).
- **`__dp4a`-vectorised 32-block dot.** The scalar loop here is correct and
  on-device; llama.cpp's `vec_dot_q4_0_q8_1_impl` is the shape to port. No
  throughput floor is claimed by this row.
- **An end-to-end `qwen4_exp` CUDA forward on the released checkpoint.** This row
  removes the keep-quant blocker only, and it does NOT claim the forward runs.

  An earlier draft of this entry named the block-decoding n-gram gather as the
  independent blocker, citing #2380. **That is now false, and merging `main`
  is what falsified it** rather than any measurement here:
  [#2396](https://github.com/mudler/vllm.cpp/issues/2396) landed
  `OpId::kEmbeddingQuant` for CUDA (`src/vt/cuda/cuda_ops.cu:4017`) and rewrote
  `DeviceQuantGatherSupported` to ask `vt::OpRegistered(kEmbeddingQuant, dev)`
  (`gguf_keep_quant.cpp:187-189`), so a block-quantized gather table is admitted
  on CUDA and that blocker is gone. The claim is corrected rather than deleted,
  because a spec that quietly drops a refuted sentence teaches nothing.

  What this row can still NOT speak for: the QSA block reads two tables on the
  HOST (`qwen4_exp_qsa_block.cpp:142` says so in the tree), which is the wave
  QSADEV owns and this row must not touch. Whether anything else refuses after
  that is unmeasured HERE and is not this row's to assert.
- **`.agents/quantization-matrix.md` is stale for two of these three, and this row
  did not fix it.** `QUANT-GGUF-Q5_0` and `QUANT-GGUF-IQ4_NL` read `INVENTORIED`
  with `-` in every column and an empty evidence cell, while
  [`docs/FEATURES.md`](../../docs/FEATURES.md) records both as landed for
  `qwen4exp` with a decode bit-exact against llama.cpp `b10451`
  ([#1989](https://github.com/mudler/vllm.cpp/issues/1989)). Neither row changes
  lifecycle state in this change, so no matrix edit is owed by it, and correcting
  the flags means re-verifying another row's work rather than recording this
  one's. Named here so the disagreement is visible rather than silently carried.

## Outcome

Measured on `dgx:gpu0` (GB10, `sm_121a`, CUDA 13.0, `-DVLLM_CPP_CUDA_ARCHITECTURES=121a`)
over the tree `fd3dd549`, whose `src/`, `tests/` and `include/` are byte-identical
to the head this row hands over. One lease, five stages, every rc read literally
from the job log:

| Stage | Tree | seam call sites | `BUILD_RC` | `TEST_RC` | assertions |
|---|---|---:|---:|---:|---:|
| A | pre-change kernel + these tests | 0 | 0 | **139** | 171,600 |
| B | the change | 5 | 0 | **0** | **218,359** |
| C | B, three production call sites DELETED | 1 | 0 | **139** | 171,600 |
| C2 | B, every call site kept, ADMISSION neutered | 5 | 0 | **139** | 171,600 |
| D | B restored | 5 | 0 | **0** | 218,359 |

`RESTORE=IDENTICAL` (byte-for-byte against the staged source), `CONFIGURE_RC=0`.

### What the numbers say

**The pre-change tree does not drain, it FAULTS.** Stage A dies with SIGSEGV in
the first case in the file. So on this hardware the blocker was never "correct
but slow": the released artifact's expert GEMM could not run on CUDA at all.

**Both mutations kill, against a green baseline, by two independent
mechanisms.** C removes the call sites; C2 leaves every call site in place and
makes the predicate refuse. Both land on **exactly** stage A's 171,600
assertions and rc 139. Two different edits reproducing the pre-change count to
the assertion is what rules out a mutation that merely broke something else.

The counted property is the seam occurrence count printed per stage
(5 / 1 / 5), not a patch exit code -- and it earned its keep. The FIRST attempt
at C failed to BUILD (`BUILD_RC[C]=1`) because an unreferenced explicit
specialization is emitted and warned under this TU's `-Werror=all-warnings`,
where an uninstantiated primary template is not. A build failure reads as a
passing test; the harness reported it as `STAGE C BUILD FAILED` and the mutation
was recorded VOID rather than counted. `[[maybe_unused]]` on all five affected
entities is what made C build, and C2 exists so that a mutation would still be
available had it not.

### Ported, and how it was checked

The three device dots reproduce this tree's CPU arm **bit-exactly**, per block
and across a 40-block row, verified on the host independently of any GPU by
compiling the CPU `vec_dot`s into the same translation unit as a transcription
of the device bodies: 0 of 40 blocks differ for each of q4_0, q5_0 and iq4_nl.
That checks the parts most likely to be wrong -- nibble extraction, Q5_0's `qh`
bit splicing, the IQ4_NL codebook index, and the three DIFFERENT scale
association orders upstream uses.

### The bound, and why it is not fitted

`max|diff| <= 8 * nb * FLT_EPSILON * max|cpu|`. Derived from the only difference
between the two sides -- association, the CPU arm summing blocks sequentially
against the kernel's per-lane partials plus a 32-wide warp tree -- rather than
fitted to a fixture. Simulating the kernel's exact reduction order on the host
over nb in {8, 20, 40, 128, 512, 2048} and 24 rows per shape puts the worst case
at **1.3% of the budget**, and the ratio FALLS as nb grows, so it points the way
the error grows. A constant would have tightened as K rose and begun failing on
the long rows this lane exists to serve.

**That 1.3% is a HOST SIMULATION and it understated the device by roughly 9x.**
Measured on GB10, the worst fraction of the bound is **11.8%**. The bound still
holds with ~8.5x headroom and no shape came close to failing, but the number
this spec published first was not the device's, and the difference is the part
the simulation could not model -- the warp shuffle-down reduction, whose
accumulation is a plain `+=` and is contractable where the per-block folds are
`_rn`-guarded. Recorded because a bound quoted at 1.3% invites a later
tightening that the hardware would not survive.

### Which arm the released checkpoint actually takes

Worth stating precisely, because "the qwen4_exp MoE now runs on device" could be
read as all three arms serving it, and that is not what was traced.

`RunQwen4ExpMoeBlock` -> `RunMoeBlock` -> `KqGrouped` -> **`vt::MatmulBTQuantGrouped`**,
three separate grouped GEMMs (gate, up, down). That is the GROUPED arm. The
shared MoE block on this path never calls the fused seam, which
`grep` over `src/vllm/` confirms: the three production callers of
`vt::MoeGateUpSwiGLUGrouped` are `glm5_next_moe.cpp:131`, `laguna.cpp:1032` and
the shared `vt::MergedGemmGroup` seam in `merged_gemm.cpp:29`, none of them this
model's path.

So for the released artifact the load-bearing arms are the **dense** and
**grouped** ones. The **fused** arm removes a real refusal -- that seam has no
fallback behind it and THREW by name for these dtypes, so any model routing
32-block gate/up towers through `vt::MergedGemmGroup` was refused outright --
but **no checkpoint this row measured takes it**. It is reachable through the
production op rather than dead: the fused gate drives it via
`vt::MoeGateUpSwiGLUGrouped`, never by constructing a kernel by hand, and
mutation C deleted that call site along with the other two and turned the gate
red.

### What this row does NOT claim

- **No speed number.** Nothing here measures throughput; the win claimed is that
  the path runs on the device at all.
- **No run of the released checkpoint.** Every gate is a fixture. What the real
  artifact does through a full forward is unmeasured here.
- **The capture gate did not supply the red** -- see the correction above.
- **The scalar dot is a throughput ceiling**, recorded under `## Owed`.
