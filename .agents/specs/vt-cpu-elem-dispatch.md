# SPEC — `VT-CPU-ELEM-DISPATCH`: the per-element dtype dispatch in the CPU attention kernels

Issue: [#2376](https://github.com/mudler/vllm.cpp/issues/2376).
Predecessor: [`.agents/specs/ltx25-connector-gemm.md`](ltx25-connector-gemm.md),
whose `## Owed` names this work and measured it. That row states the defect and
declines to repair it, because `AttentionCrossKernel` is a `vt` seam every model
reaches and a change to it needs its own row, its own spec and a fresh reviewer.
This is that row.

## The defect

`vt::AttentionCross` and `vt::Attention` read every operand element through
`LoadF32` (`src/vt/cpu/cpu_ops.cpp`), which switches on `t.dtype` and multiplies
the offset by `vt::SizeOf(t.dtype)` once per element. Both are loop-invariant.
`vt::SizeOf` was declared at `include/vt/dtype.h` and defined out of line at
`src/vt/dtype.cpp`, and the build enables **no LTO** — `CMakeLists.txt` sets no
`INTERPROCEDURAL_OPTIMIZATION` and passes no `-flto`, confirmed by a `grep` over
`CMakeLists.txt` and `cmake/` that returns nothing — so it was a cross-
translation-unit call the optimizer could not remove. `AttentionCrossKernel`'s
score loop calls `LoadF32` twice per element around a body of one multiply and
one add.

`LoadF32`/`StoreF32` are called at **275 sites across 64 distinct enclosing
functions**, in `src/vt/cpu/cpu_ops.cpp` and `src/vt/cpu/cpu_paged_attn.cpp`,
counted on `origin/main` by walking the two files and attributing each call
expression to the function it sits in. Attention is where it was measured, not
where it lives.

**The predecessor row's `## Owed` says 219 call sites and this row measured 275.**
The 64-function figure reproduces exactly, so the two counts are counting
different things rather than disagreeing about the code; 219 is neither the line
count (236) nor the call count. This row quotes its own number and its own
method, because a number quoted often becomes treated as measured.

## Scope

IN scope:

- **W1** — evaluate the CHEAP repair FIRST and report its number either way:
  make the dispatch inlinable at the source. One change; it reaches all 275
  sites without touching a kernel.
- **W2** — only if W1 falls short, hand-hoist `AttentionCrossKernel` and
  `AttentionKernel`. Those two, and no others.
- **W3** — a byte-equality gate over the dtype matrix the ops actually accept,
  with a mutation per claimed guarantee.

OUT of scope, declared rather than approximated:

- **The other 62 kernels.** Sweeping them is not one row. What they are owed,
  and the method for sizing which of them is worth a row, is in `## Owed`.
- **Changing any output value.** Byte equality is the bar. A change that cannot
  be bit-exact stops and reports; it is never traded for a rate.
- **`include/vt/quant.h`'s `elem_kn_repacked` claim.** The predecessor row owns
  it. This row carries the finding forward in `## Owed` and edits nothing.
- **The aarch64 half.** Every number here is x86-64 AVX-512. See `## Owed`.

## Design

Two repairs, evaluated in cost order, because they are not equal and the cheap
one might have made the expensive one unnecessary.

**(A) Make the per-element helpers inlinable.** `vt::SizeOf` moves into
`include/vt/dtype.h` as `inline`, with its two refusals kept out of line and
cold as `[[noreturn]]` helpers so the hot path is a switch over six integers and
nothing else. The enumeration stays exhaustive with no `default:` label, so
adding a dtype is still a `-Wswitch` error rather than a silent zero. The same
transformation applies to `F16ToF32` and `BF16ToF32`, which every `LoadF32` in
`src/vt/cpu` calls once per element on the reduced-width arms; the two f32 ->
narrow directions stay out of line, being store-side and carrying the
round-to-nearest-even logic. `vt::SizeOf` is not in `cmake/vllm_export.map`, so
no ABI moves.

**(B) Hoist the dispatch out of the two attention kernels.** This is the shape
`MatmulOneChunk` already applies against `MatmulOneChunkRef` in the same file,
and it is not a new one: widen the row that is REUSED to f32 once
(`WidenRowToF32`, which the GEMM already uses and which shares the same
converters), and resolve the STREAMED operand's dtype once per call into a typed
micro-kernel reached through a function pointer — the `ElemGemmTierTable` shape,
one indirect call per key row instead of two switched calls per element.

**Why it is bit-exact, stated as an argument rather than as a hope:**

- `WidenRowToF32` writes exactly the f32 values `LoadF32` returned for the query
  row, through the same `F16ToF32`/`BF16ToF32`, so the multiplicands are the
  same bits;
- `AttnDotT` accumulates over `e` in the same increasing order into one f32
  accumulator. It is a serial float reduction, so with `-ffp-contract=off`
  (CMakeLists.txt:55) and no `-ffast-math` the vectorizer may not reassociate it
  and does not;
- `AttnAccumT` walks the same `(j, e)` order into the same f32 `acc[]`, and each
  `acc[e]` stays its own independent chain over `j` — which is exactly what lets
  it vectorize ACROSS `e` without reordering any sum;
- each output row is independent, so the threadpool partition is unchanged.

No accumulator is split and no sum is reordered. The outputs are
`memcmp`-identical, not close.

## Reachability

Nothing new lands unreached. Both kernels are already registered
(`src/vt/cpu/cpu_ops.cpp`, the `AttentionCrossFn` registration) and already
reached from `ModelRegistry::Forward` through `vt::AttentionCross` and
`vt::Attention` — the LTX-2.5 connector and DiT among many callers. This row
changes the body of a reached kernel and adds no new entry point. The gate enters
through `vt::AttentionCross`/`vt::Attention`, never by constructing the kernel,
and the two reachability mutations in `## Outcome` prove the new micro-kernels
are what executes.

## Tests to port

There is no upstream test: upstream's attention is `torch` and cuBLAS. The tests
are this tree's own, and each is an executable observable.

| ID | Assertion |
|---|---|
| T1 | CPU `vt::AttentionCross` is `memcmp`-identical to a per-element reference over the dtype matrix the op accepts, at ragged shapes, with no bias / `[1,S]` bias / `[Tq,S]` bias |
| T2 | CPU `vt::Attention` likewise, causal and non-causal |
| T3 | a non-float operand is still refused at the op boundary, with the message unchanged |

`tests/vt/test_ops_attention_elem_dispatch.cpp`. The reference is the ORIGINAL
per-element loop, re-derived in the test from the layout contract in
`include/vt/ops.h` and sharing nothing with `src/vt/cpu` — a gate that compared
the kernel against a helper the kernel also uses would prove consistency, not
correctness.

**The dtype matrix is bounded by what the OP accepts, not by what the kernel
could be handed.** `vt::Attention` and `vt::AttentionCross` both require
`IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype`
and `IsOutFloat(out.dtype)`. So there is no mixed-q/k/v row: it would test an
input the production entry point refuses, and an unreachable case that passes is
not evidence. The reachable discriminating axis is `out` disagreeing with the
operands, and every row exercises it.

**This closes a real hole.** Before this row the CPU arm of
`tests/vt/test_ops_attention_cross.cpp` ran f32 operands and nothing else — its
bf16 geometries are CUDA-only — so `AttentionCrossKernel`'s f16 and bf16 element
paths were ungated.

## Gates

1. `tests/vt/test_ops_attention_elem_dispatch.cpp` green, and green on a
   pristine `origin/main` tree too, which is what says the reference is a valid
   oracle rather than a transcription of the new code.
2. The broad CPU `vt` suite green, with the SAME case and assertion counts on
   both trees, so nothing was silently skipped.
3. A mutation per claimed guarantee, each verified to have applied and built
   before its result is read.
4. Byte equality between arms on the measurement binaries, f32 and bf16.
5. Before/after at the SHIPPED thread count, arms interleaved, `n >= 3`, with a
   same-arm control and the box's load stated.
6. `scripts/agent-preflight.sh`.

## Risks/decisions

- **The devbox is not idle and cannot be made idle.** Other agents compile on
  the same 20 cores throughout; loadavg ran 12 to 56. Every comparison is
  therefore INTERLEAVED with a same-arm control leg, so drift lands on all arms
  equally and a gap smaller than the control is not a result.
- **No CMake build tree fits.** `/` reached 1.5 GB free during this row. Every
  binary here is a direct `g++` compile over the `vt` TU set at the project's own
  `-std=c++20 -O2 -ffp-contract=off` and the per-source ISA flags from
  `CMakeLists.txt:1393-1404`. `-ffp-contract=off` is not optional: the
  bit-identity contracts depend on it.
- **`always_inline` on `LoadF32` was measured and NOT taken.** See `## Owed`.
- **This is x86-64 only.** The aarch64 tier has different inlining economics and
  a different vector width. `## Owed` says so.

## Stop conditions

- a speedup that cannot be made bit-exact — report it, never trade correctness;
- ENOSPC;
- a same-arm control that swallows the effect being claimed.

## Now

`ACTIVE`. W1, W2 and W3 are complete on x86-64 and every number below was taken
there. The aarch64 half is owed and named in `## Owed`.

## Outcome

### The headline

**(A) alone was not enough, and that is the answer to the question this row was
told to ask first.** Making `vt::SizeOf` inline removes the cross-translation-
unit call and buys **1.35x to 1.93x**. It does not let the compiler hoist the
switch, so it leaves most of the defect standing. Forcing `LoadF32` inline as
well reaches only **2.6x**. The hand-hoist of the two attention kernels is worth
**8.75x to 11.16x**, byte-identical, across four independent load regimes.

So the cheap repair landed AND the expensive one was still needed. Both are here,
as separate commits in that order, so the measurement that justified the second
is visible in the history rather than asserted after the fact.

### W1 — the cheap repair, measured before any kernel was touched

`vt::SizeOf`, `F16ToF32` and `BF16ToF32` move into `include/vt/dtype.h` as
`inline`. Byte-identical output on every arm.

| arm | video (Tq=S=1024, H=32, D=128) | audio (D=64) |
|---|---:|---:|
| f32, sequential, loadavg 12-31 | 4.768 s -> 3.217 s = **1.48x** | 2.440 s -> 1.510 s = **1.62x** |
| f32, interleaved, loadavg 30-54 | 10.040 s -> 6.507 s = **1.54x** | 5.394 s -> 2.795 s = **1.93x** |
| bf16, interleaved, loadavg 50-56 | 9.195 s -> 6.815 s = **1.35x** | 5.116 s -> 3.465 s = **1.48x** |

**Why it stops there, proved by profile rather than argued.** `perf record -e
cpu-clock` over `vt::AttentionCross` on the inlined tree:

| symbol | self |
|---|---:|
| `LoadF32(Tensor const&, long)` | 62.68% |
| `AttentionCrossKernel(...)::{lambda(long, long)#1}` | 17.79% |
| `Threadpool::Barrier()` | 16.69% |

`vt::SizeOf` is gone from the profile — it inlined. `LoadF32` did not: GCC keeps
it out of line at 231 call sites in one translation unit, so the switch still
executes per element and the call overhead is merely cheaper.

**The obvious follow-up was measured and NOT taken.** Adding
`__attribute__((always_inline))` to `LoadF32` reaches 4.768 s -> 1.810 s
(**2.63x**) and 2.440 s -> 0.974 s (**2.50x**), still byte-identical. So even
fully inlined, the compiler declines to hoist a loop-invariant switch it would
have to version the loop for. It is recorded in `## Owed` rather than landed,
because it forces inlining at 231 sites in a 3900-line TU and this row has no
gate that measures the other 62 kernels' icache behaviour.

### W2 — the hoist, and what it is worth

| regime | loadavg | video | audio | same-arm control |
|---|---|---|---|---|
| f32, sequential | 12-31 | 4.768 -> 0.603 s = **7.90x** | 2.440 -> 0.295 s = **8.27x** | — |
| f32, interleaved, vs base `9fb40279d` | 30-54 | 10.040 -> 0.900 s = **11.16x** | 5.394 -> 0.611 s = **8.82x** | 1.049x / 0.987x |
| bf16, interleaved | 50-56 | 9.195 -> 0.995 s = **9.24x** | 5.116 -> 0.542 s = **9.44x** | 0.935x / 0.955x |
| f32, interleaved, MERGED head vs `origin/main` tip | 60-73 | 12.088 -> 1.382 s = **8.75x** | 6.411 -> 0.677 s = **9.47x** | 1.001x / 0.972x |

**The devbox was never idle and that is stated rather than smoothed.** Other
agents compiled on the same 20 cores throughout; loadavg ran 12 to 73. Every
comparison after the first is INTERLEAVED with a same-arm control leg, and the
control reads 0.935x to 1.049x while the effect reads 7.9x to 11.2x. A gap two
orders of magnitude outside its own control is a result whatever the box was
doing. The ratio's own spread across regimes (7.9x to 11.2x) is real and is why
this row quotes a range, not a single number.

**Single-thread comparisons are not quoted.** The predecessor row's "one thread
with the dtype hoisted beats twenty without it" held on x86 and FAILED on
aarch64, so every figure above is at the shipped thread count on both arms.

### W3 — the correctness evidence

**Byte equality, which is the whole gate.** Every measurement binary dumps its
raw output and every arm is `memcmp`-identical to the pre-change binary, f32 and
bf16, video and audio shapes, including against the `origin/main` tip after the
merge.

**`tests/vt/test_ops_attention_elem_dispatch.cpp`: 4 cases / 114 assertions.**
Green on the changed tree AND green on a pristine `origin/main` tree built from
`git archive`, which is what says the reference is an oracle rather than a
transcription of the new code.

**The broad CPU `vt` suite: 212 cases / 554,046 assertions, 0 failed**, over
attention, attention_cross, matmul_elem, dtype, layernorm, matmul, conv2d/3d/1d,
paged_attn, paged_attn_dtype, dflash_block_attn, mamba2_ssd, attn_relpos, rope,
activation, quant_dot, quant_repack, cast_f16, tensor and reference_tier.
**The counts are IDENTICAL on the pre-change tree and on the merged head**, which
is what says nothing was silently skipped rather than silently passing.

**The mutations.** Each was verified to have applied and to have built before its
result was read, and the tree was restored byte-for-byte after each.

| # | mutation | result |
|---|---|---|
| M1 | split the score reduction into two accumulators | KILLED — 35 assertions red |
| M2 | reverse the `j` order of the value accumulation | KILLED — 29 red |
| M3b | make `StoreF32`'s **f16** branch wrong | **SURVIVED** |
| M3c | make `StoreF32`'s bf16 branch truncate instead of round-to-nearest-even | KILLED — 51 red |
| M4 | widen the query row with the wrong f16 conversion | KILLED — 36 red |
| M5 | inline `SizeOf` returns 4 for bf16 | KILLED — 30 red |
| M6 | REACHABILITY: the new typed dot returns 0 | KILLED — 90 red |
| M7 | REACHABILITY: the new typed accumulator is a no-op | KILLED — 108 red |

**M3b survived and the reason is a finding, not a hole this row opened.** Both
ops validate `IsOutFloat(out.dtype)`, which admits f32 and bf16 only, so
`StoreF32`'s f16 branch is UNREACHABLE from `vt::Attention` and
`vt::AttentionCross`. No test here can kill it and this row does not claim one
does. M3c is the same guarantee on the branch that IS reachable, and it dies.

**M6 and M7 are the reachability proof.** Both new micro-kernels were made inert
and the gate went red through `vt::AttentionCross`/`vt::Attention`, so what the
test exercises is the new code on the production path and not a class the test
constructed.

**The refusal moved and its message did not.** The per-element refusal used to be
thrown inside a threadpool worker and rethrown by `Threadpool::Run`; it is now
raised on the calling thread before any element is read. `AttnResolveOrRefuse`
produces it from `LoadF32`/`SizeOf` themselves rather than from a second message.
That branch is unreachable through the two ops, which validate `IsFloat` first,
and the spec says so instead of claiming a gate on it.

### A finding this row carries but does not own

**`include/vt/quant.h` documents the `elem_kn_repacked` repack as "1.16x to 1.30x
faster on dgx" and that does not hold on x86-64.** The predecessor row measured
it **1.78x / 1.96x / 2.06x SLOWER** across three load regimes at the LTX-2.5
connector's shapes, byte-identical throughout, and **1.25x to 2.06x slower on
aarch64**. The header names a machine the claim appears not to hold on. It needs
a scope qualifier or a correction, it is not this row's kernel, and it is in
`## Owed` with its owner.

## Owed

- **THE OTHER 62 KERNELS.** This row hoisted two of the 64 functions that call
  `LoadF32`/`StoreF32`. The remaining 62 still resolve a dtype per element, and
  (A) gives them 1.35x-1.93x for free while leaving the rest on the table.
  **The sizing method, so the next row prioritizes by measurement rather than
  sweeping:** for each candidate, `perf record -e cpu-clock` over a probe that
  runs THAT op alone at a shape a shipped model actually uses, and read
  `LoadF32`'s self percentage. Above ~30% the hoist is worth a row; below it the
  kernel is bound by something else and a hoist buys the fraction the profile
  names. `AttentionKernel` and `AttentionCrossKernel` read 62.68% and were worth
  9x. Do not sweep: each hoist needs its own byte-equality gate, and the two here
  cost eight mutations between them. Owner: unowned; issue #2376 stays open
  against this item.
- **`__attribute__((always_inline))` ON `LoadF32`, worth a further 1.78x on top
  of (A) and MEASURED.** 4.768 s -> 1.810 s and 2.440 s -> 0.974 s, byte-
  identical. It was not taken because it forces inlining at 231 sites in one
  3900-line TU and no gate here measures the other 62 kernels' instruction-cache
  behaviour. Sizing it is one A/B over the suite above plus a per-kernel probe.
  Owner: unowned.
- **THE aarch64 HALF. Every number in this row is x86-64 AVX-512.** The tier is
  NEON with a different vector width and different inlining economics, and the
  predecessor row's own single-thread claim inverted between the two
  architectures, so nothing here may be read as a GB10 statement. The repair is
  bit-exact by construction on any ISA, so what is owed is the RATE, not the
  correctness. Reaching those cores needs an `rc` lease. Owner: unowned.
- **`include/vt/quant.h`'s "1.16x to 1.30x on dgx" needs a scope qualifier or a
  correction.** Owner: `LTX25-CONNECTOR-GEMM`, whose `## Owed` already carries
  it; recorded here because this row saw the same header and did not edit it.
- **`F32ToF16` and `F32ToBF16` are still out of line.** They are store-side, so
  they cost once per OUTPUT element rather than once per operand element, and
  this row did not measure them. Whether a narrowing-store-heavy kernel pays for
  it is unmeasured, not answered. Owner: unowned.
