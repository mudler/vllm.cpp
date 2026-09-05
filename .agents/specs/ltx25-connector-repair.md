# SPEC — `LTX25-CONNECTOR-REPAIR`: re-split the connector leaf after the attention hoist, then repair what the new split blames

Issue: [#2434](https://github.com/mudler/vllm.cpp/issues/2434).
Owner row: `LTX25-CONNECTOR-REPAIR`.
Predecessors: [#2296](https://github.com/mudler/vllm.cpp/issues/2296) (the 5.53x
render reading), [#2354](https://github.com/mudler/vllm.cpp/issues/2354) (the
weights/compute split), `LTX25-CONNECTOR-GEMM` (the leaf decomposition), and
[#2376](https://github.com/mudler/vllm.cpp/issues/2376) /
[#2416](https://github.com/mudler/vllm.cpp/issues/2416) (the dtype hoist that
invalidated it).

## The gap, and why it is not the one the predecessor left

`.agents/specs/ltx25-render-speed-parity.md`'s first `## Owed` line is still
open in the exact words it was written in: **"The repair is not here."**
`conditioning.connector.compute` + `guiders.connector.compute` = 224.882 s,
**43.52% of a 516.751 s render**, at spreads of 1.70% and 2.07% — the two
quietest rows in that table, against a `denoise` of 2.92%.

`LTX25-CONNECTOR-GEMM` decomposed that leaf on GB10 and blamed
`vt::AttentionCross`: attention 66.5%, GEMM 24.8%, residue 8.7%. **That is no
longer the tree.** `VT-CPU-ELEM-DISPATCH` hoisted the per-element dtype dispatch
out of `AttentionCrossKernel` and measured **11.30x / 12.12x on GB10** at those
same shapes, byte-identical. A repair chosen against the old split would be
chosen against a leg that has already been repaired.

**So this row re-measures before it repairs, and the re-measurement is the first
deliverable rather than a preamble to one.** If the hoist already took most of
the 224.9 s, saying so closes the parent row's owed line and is the best outcome
available.

## Scope

IN scope:

- **W1** — the connector leaf re-measured on current `main`: `Ltx2ConnectorForward`
  in total, `vt::AttentionCross` alone at the same shape, and the connector's
  whole GEMM set, `n >= 3`, interleaved, with the box's load stated per leg. On
  x86-64 here and on GB10 under a lease, because GB10 is the machine the render
  was measured on and because a claim in this campaign has already inverted
  between the two ISAs.
- **W2** — the repair the new split blames, bit-exact, with a red-first
  byte-equality gate and a mutation per claimed guarantee.
- **W3** — the before/after with spread, a same-arm control leg beside it, and
  what it does and does not say about the render.

OUT of scope, declared rather than approximated:

- **A device arm for the connector.** `Ltx2Attention`
  (`src/vllm/model_executor/models/ltx2.cpp`) interleaves host `RmsNormRows` and
  `Ltx2ApplyRotaryEmb` on raw `float*` between its GEMMs, and
  `Ltx2ConnectorForward` reads every weight through `View(...)` over a host
  `std::vector<float>`. That is a weight-arm port with its own numerics gate, not
  a queue swap. #2354 says so, `LTX25-CONNECTOR-GEMM` repeats it, and this row
  does not contradict either. If the measurement points there, this row returns
  the sizing and stops.
- **A published benchmark ID.** `docs/BENCHMARKS.md` gains nothing. One module's
  GEMM shapes on one request is an instrument, not a benchmark.
- **Weakening any numeric contract.** Every kernel in `cpu_matmul_elem.cpp` keeps
  each output's K reduction strictly sequential, which is what makes the SIMD
  tiers bit-identical to the scalar reference. Byte equality is the gate here and
  a speedup that cannot meet it is reported, never taken.
- **A general tiled sgemm.** `.agents/specs/cpu-elem-gemm-wide-isa-and-tiling.md`
  carries `KERNEL-GEMM-CPU-TILED` as a SPIKE with no active row. This row does
  not open it. What it may land is the contained change its own measurement
  prices, and what it may not do is turn into that campaign.

## What the instrument is, and why no new one is written

`tools/bench/ltx2_connector_gemm_probe.cpp` already runs `Ltx2ConnectorForward`
at the shipped geometry (`--mode connector`), `vt::AttentionCross` alone at the
connector's own shape (`--mode attn`), the connector's whole GEMM set
(`--mode gemm`), and the runtime tier resolver (`--mode tier`). It is on `main`,
its shapes are read out of the pinned DiT's `__metadata__.config`, and it was
the instrument the old split was taken with. **Re-measuring with the same
instrument is what makes the before and the after comparable at all**, so this
row adds no probe and changes none of its shapes.

**The render harness pins the binary's sha256 by design and therefore refuses a
rebuild** (`scripts/ltx25-render-speed-repeat.sh`, exit 51). That is correct for
what it measures — a timing of the tree that took the correctness verdict — and
it is why the before/after here is taken on the probe rather than by re-running
that harness against a changed binary. An end-to-end render of a changed arm is
`scripts/ltx25-text-cond-ab.sh`'s job: it builds two arms into SEPARATE build
directories, inverts its identity assertion so it cannot measure one binary
twice, and requires `pixel_files_differing=0`.

## The measured position, before any change

Every number in this section is x86-64 AVX-512 on a devbox at the load stated
beside it. The GB10 half is in `## Outcome`.

**The hoist landed and the attention leg is gone.** `--mode attn` on current
`main` reads the probe's own hoisted reference at **1.06x** (video) and **1.02x**
(audio) against the shipped kernel, where before the hoist the same comparison
read 13.30x and 9.58x here and 11.30x / 12.12x on GB10. The instrument that
found the defect now finds no defect, which is the control this row wanted.

## Design of the repair, stated as a mechanism rather than a patch

`MatmulOneChunk` (`src/vt/cpu/cpu_ops.cpp`) widens a **16-row** activation tile
(`blck_1`, ggml's `mul_mat` chunk tile) and then walks it in blocks of `mr`
activation rows, where `mr` is the tier's M-blocking factor: **6 on AVX-512, 4 on
NEON, 2 on SSE2**. Each `mr` block costs ONE pass over the 16-column weight block
— the load and, on the `[N,K]` orientation, the 4x4-group register transpose that
`Transpose16` performs — and that pass is what the M blocking exists to amortize.

**`mr = 6` does not divide 16.** Rows 0-11 take two M-blocked calls; rows 12-15
fall through to the `mr = 1` path and take **one whole weight pass each**. A
16-row tile therefore performs **6 weight passes instead of the 3 its own
blocking factor allows**, and the four tail rows pay a full transpose apiece.
`kMrNeon = 4` divides 16 exactly, so this specific defect is **AVX-512 only** —
which is stated here because the machine the render was measured on is aarch64
and a reader must not carry the x86 figure onto it.

Two changes, each bit-exact by the same argument the tier already rests on:

1. **Pad the widened tile up to a whole number of `mr` blocks and zero the pad
   rows**, so every activation row goes through the M-blocked kernel. The pad
   rows' outputs are computed and never stored. No output's K reduction order
   moves, because `ElemBtMFn` accumulates lane `l` of row `r` over `p` in strict
   increasing order whatever `mr` is — which is exactly why the tree already
   runs some rows of the same call through `btm` and others through `bt` and
   asserts both byte-identical to `MatmulOneChunkRef`.
2. **Widen the AVX-512 tier's `mr` from 6 to 8.** `BtM6Avx512` holds 16
   transposed weight vectors plus `mr` accumulators plus one broadcast; at
   `mr = 8` that is 25 of 32 ZMM. 8 divides 16, so a full tile becomes **two**
   weight passes.

Both are `vt` shared-seam changes and every CPU f32/f16/bf16 GEMM in the tree
runs through them, which is why the gate below is the whole dtype matrix at more
than one worker count and not the connector's own shape.

## Reachability

Nothing new lands unreached and no new entry point is added. `MatmulOneChunk` is
the body of `vt::MatmulBT` / `vt::Matmul`, reached from `ModelRegistry::Forward`
through every CPU dense projection in the tree, and from `Ltx2ConnectorForward`
through `Linear` on the render's default path. The gate enters through
`vt::MatmulBT`, never by calling the kernel, and the mutation below deletes the
padding so the gate reds through that production entry point.

## Tests to port

There is no upstream test. Upstream's connector is a `torch.nn.Module` and its
GEMM is cuBLAS; ggml's own `mul_mat` has no bit-identity contract to port because
ggml reassociates the K reduction and we deliberately do not
(`src/vt/cpu/cpu_matmul_elem.h`, RECORDED DEVIATION). The tests are this tree's
own and each is an executable observable.

| ID | Assertion |
|---|---|
| T1 | `vt::MatmulBT` is `memcmp`-identical to `MatmulOneChunkRef` at M values that are NOT multiples of `mr` — the rows the padding moves onto a different micro-kernel — over the whole `{f32, f16, bf16}` x `{f32, f16, bf16}` operand matrix |
| T2 | the same at ragged N and ragged K, so the column tail and the K tail are exercised together with the padded M tail |
| T3 | every one of the above is `memcmp`-identical at SEVEN worker counts, so a claim about which thread computes which output is tested against more than one partition |

`tests/vt/test_ops_matmul_elem_mblock.cpp`. The reference arm is
`VT_CPU_MATMUL_TIER=ref`, which is `MatmulOneChunkRef` — the per-element scalar
loop the whole tier hierarchy is defined against — so the oracle shares no code
with the kernel under test.

**ONE WORKER COUNT CANNOT TEST THIS.** The padding changes which rows enter
`btm` and therefore which rows are grouped into one accumulator set. The rows a
chunk covers are set by the threadpool's partition, so a run at one worker count
exercises one partition. `VT-CPU-ELEM-SURVEY`'s M9 is the precedent this rule was
paid for: a deliberately broken kernel passed 753,300 assertions at
`VLLM_CPP_CPU_THREADS=1` and only the worker-count case caught it.

**The operands must be separate enough that the assertion CAN fail.** The same
row's M2 survived its first gate because two operands shared one scale and their
sums fit in 8 significant bits, which bf16 holds exactly. The fixture therefore
fills A and B from independent streams at different magnitudes.

## Gates

1. `tests/vt/test_ops_matmul_elem_mblock.cpp` green, and green on a pristine base
   tree too, which is what says the reference is a valid oracle rather than a
   transcription of the new code.
2. `tests/vt/test_ops_matmul_elem.cpp` green with the SAME case and assertion
   counts on both trees, so nothing was silently skipped.
3. A mutation per claimed guarantee, each verified to have APPLIED and BUILT
   before its result is read.
4. Before/after on `tools/bench/ltx2_connector_gemm_probe.cpp`, arms interleaved,
   two SEPARATE build directories whose binaries are asserted to differ,
   `n >= 3`, a same-arm control leg, and the box's load stated per leg.
5. `scripts/check-tree-compiles.py` and `scripts/agent-preflight.sh`.

## Risks/decisions

- **The devbox is never idle and the load moves DURING a run.** Loadavg ran 25 to
  48 across this row's measurements. Every comparison is interleaved with a
  same-arm control leg and the control's own spread is quoted beside every ratio.
  A gap inside its control is not a result.
- **Contention reweights rather than merely adding noise.** It inflates
  multi-threaded legs relative to single-threaded ones, which is how
  `LTX25-CONNECTOR-GEMM`'s contended decomposition closed to 1.7% when the idle
  one closed to 12.1%. Where this row states a share it names the load it was
  taken at.
- **A claim can invert between architectures**, and in this campaign one already
  has. `mr = 6` is the AVX-512 tier's; `mr = 4` is NEON's and divides 16. The x86
  figure is not a GB10 figure and is never quoted as one.
- **No ceiling is declared.** Where the measurement stops, the next traceable
  hypothesis is named instead.

## Stop conditions

Stop and report, do not work around:

- a speedup that cannot be made bit-exact;
- a same-arm control that swallows the effect being claimed;
- an unhealthy or unreachable fleet device;
- a device port turning out to be the answer — that is `NEEDS_DECISION` with the
  sizing, not a thing to start inside this row;
- ENOSPC. `/` is at 93% and a full CMake build writes 9.4 GiB.

## Work breakdown

- **W1** — this spec and the re-measurement.
- **W2** — the repair and its gates.
- **W3** — the before/after and `## Outcome`.

## Now

`DONE`. W1 is taken on x86-64 and on both `dgx:gpu0` (GB10) and `thor:gpu0`
under leases. W2 is landed, gated on both architectures, and its aarch64 arm is
measured and refused. W3 is the A/B below. What is owed is the end-to-end render
and the GEMM campaign this row deliberately does not open.

## Outcome

### The headline, and it is the one the parent row was owed

**The attention hoist already took most of the 224.9 s, and the connector leaf
is now a GEMM.** Measured with the same probe, at the same shapes, **on GB10
itself -- the machine the render's 516.751 s was measured on** -- under an `rc`
lease on an otherwise idle box, one `RunConnector` call went from
`LTX25-CONNECTOR-GEMM`'s **128.808 s** to **50.34 s**, a **2.56x**, and its
composition inverted:

| GB10, 8 layers x 2 streams | before (`LTX25-CONNECTOR-GEMM`, same host, same probe) | now |
|---|---:|---:|
| `Ltx2ConnectorForward` | **128.808 s** | **50.34 s** |
| ~ `vt::AttentionCross` | 85.704 s (**66.5%**) | **8.47 s (16.8%)** |
| ~ the GEMMs | 31.906 s (24.8%) | 32.86 s (**65.3%**) |
| ~ everything else | 11.198 s (8.7%) | 9.01 s (17.9%) |

**`vt::AttentionCross` is 10.12x smaller and the GEMM did not move** -- 31.906 s
to 32.86 s, inside the 6.6% same-arm spread -- which is the shape a repair to
one leg and not the other has to have. And it lands on the number the
predecessor row predicted before the repair existed: `LTX25-CONNECTOR-GEMM`
projected "128.808 s -> **50.520 s**, a **2.55x**". Measured: **50.34 s,
2.56x**, within 0.4%. A prediction recorded in advance and then met is worth
more than either number alone.

**`thor:gpu0` agrees, on a different aarch64 part.** 84.608 s -> **40.14 s**
(2.11x), attention 48.200 -> **5.51 s** (8.75x), GEMM 29.337 -> 29.98 s
(unchanged), residue 11.6%. Two hosts, same direction, same mechanism.

On x86-64 the same control holds from the other side: the probe's own hoisted
reference, worth 13.30x / 9.58x there before the hoist, is now worth **1.06x /
1.02x**. The instrument that found the defect finds no defect.

**So `.agents/specs/ltx25-render-speed-parity.md`'s first `## Owed` line --
"The repair is not here" -- is answered, and it was answered by
`VT-CPU-ELEM-DISPATCH` rather than by this row.** What this row adds is the
measurement that says so, and one further repair on the leg that is left.

**Projected onto the render, and it is a PROJECTION.** #2354 measured connector
compute at **224.882 s in a 516.751 s render**. At GB10's measured 2.56x that
leaf falls to **87.9 s**, the render to about **380 s**, and the oracle gap from
**5.51x to about 4.05x**. **No render was run.** The probe's connector total is
not the render's leaf -- synthetic weights, a different valid-token count, and
the two streams timed in one process -- so the RATIO transfers and the seconds
do not. The end-to-end re-measurement is under `## Owed` with the reason it is
not here.

### W1 — the leaf, re-measured, and the load it was measured at

**Two leases, and the primary one is GB10.** `rc` job
`a719dc37-580b-4c59-8bd4-0601f17d343f` on `dgx:gpu0`, worker `rc-worker-4b8lj`,
2026-08-31T22:43:00Z, **aarch64, 20 cores, 120.6 GiB `MemAvailable`, loadavg
2.12 at the first leg** -- the probe itself runs 20 threads, so almost all of
the load it ends at is the measurement. The second is `thor:gpu0`, job
`76d149ba-a737-4940-b3f2-b13f623e7512`, worker `rc-worker-n8smh`,
2026-08-31T22:10:42Z, 14 cores, loadavg 5.5 at its first leg.

`tier_name=neon`, `elem_gemm_use_ref=0`, `mr=4` on both; forced to `ref` the
same resolver prints `tier_name=ref`, so it was read rather than printed from a
constant. The portable tier runs the same GEMM set on thor at **50.6 GFLOP/s**
against neon's **137.6**, so the tier that runs is 2.7x the portable one and is
not the reference tile.

**GB10, per leg, `n = 6` for arm A** (three rounds, each carrying a same-arm
control leg):

| | median | spread | legs |
|---|---:|---:|---|
| `Ltx2ConnectorForward`, one video layer | **4.854 s** | 8.4% | 5.189 4.872 4.782 4.814 4.836 4.978 |
| `Ltx2ConnectorForward`, one audio layer | **1.439 s** | 9.2% | 1.419 1.411 1.524 1.477 1.459 1.392 |
| `vt::AttentionCross`, video shape | **0.704 s** | 9.7% | 0.693 0.704 0.690 0.758 0.709 0.704 |
| `vt::AttentionCross`, audio shape | **0.354 s** | 5.9% | 0.353 0.355 0.368 0.371 0.350 0.354 |
| the connector's whole GEMM set | **32.863 s** | 6.6% | 32.906 33.009 34.699 32.514 32.820 32.691 |

**thor, per leg, `n = 6` for arm A:**

| | median | spread | legs |
|---|---:|---:|---|
| video layer | **3.909 s** | 5.9% | 4.056 3.898 3.919 3.824 3.950 3.843 |
| audio layer | **1.110 s** | 11.8% | 1.105 1.118 1.233 1.104 1.114 1.102 |
| `vt::AttentionCross`, video | **0.443 s** | 22.3% | 0.440 0.438 0.443 0.443 0.537 0.532 |
| `vt::AttentionCross`, audio | **0.245 s** | 17.5% | 0.281 0.238 0.244 0.244 0.247 0.268 |
| the whole GEMM set | **29.956 s** | 2.3% | 29.637 29.902 29.930 29.982 29.984 30.323 |

(One further GEMM figure per host -- 81.545 s on thor -- is the FORCED-PORTABLE
leg and is excluded from that median rather than averaged into it.)

**On x86-64 the same re-measurement gives the same shape.** Devbox, 20 cores,
`tier_name=avx512`, `mr=6` before this change: the connector's video layer reads
**3.079 s** where `LTX25-CONNECTOR-GEMM` measured **6.567 s** idle before the
hoist, and `vt::AttentionCross` at that shape reads **0.352 s** against
**3.831 s**. This devbox is shared and its load moved from 3 to 29 across the
session, so its absolute seconds are lower bounds and every ratio below carries
its own control.

### W2 — the repair, and the mechanism it removes

`MatmulOneChunk` widens a 16-row activation tile and walks it in blocks of the
tier's `mr`. One `btm` call is one pass over the 16-column weight block -- the
load, and on `[N,K]` the 16x16 register transpose `Transpose16` -- and that pass
is what M blocking exists to amortize. **`mr = 6` does not divide 16**: rows
0-11 took two M-blocked calls and rows 12-15 fell through to the ONE-ROW kernel,
one whole weight pass and one whole transpose each. Six passes where the
blocking factor allows three.

Two changes, and the second is where the x86 result comes from:

1. The tile is padded up to a whole number of `mr` blocks, the pad rows are
   zeroed, and the store is clamped to the rows the tile holds. **It pads only
   where padding removes a pass.** With `rem = nrows % mr` the tile costs
   `floor(nrows/mr) + rem` passes unpadded and `ceil(nrows/mr)` padded, so
   padding saves `rem - 1` passes and is declined at `rem < 2` and at
   `nrows < mr`. A one-row or two-row decode chunk therefore keeps today's
   kernel exactly, which is the regression this guard exists to refuse: without
   it a single-token GEMV would compute `mr` rows to store one, for no fewer
   weight passes.
2. `kMrAvx512` becomes 8, so a 16-row tile is exactly two passes. The kernel
   holds 16 transposed weight vectors plus `mr` accumulators plus one broadcast:
   25 of 32 ZMM at 8, against 23 at 6.

`kElemMaxMr` is new and every tier `static_assert`s its own `mr` against it,
because `MatmulOneChunk`'s stack accumulator tile is what bounds `mr` and the
tier that sets the value is where the refusal belongs.

### W3 — the measurement, paired and interleaved

**x86-64, on the two binaries that LAND**, sha256 asserted to differ before any
timing. Eight adjacent `(main, change)` pairs on the connector's own GEMM
shapes, each leg `--reps 2` so the probe medians internally, devbox at loadavg
36 to 53 with another agent's work on it throughout:

| | value |
|---|---|
| paired ratios | 1.298 1.294 1.487 1.131 1.288 **0.983** 1.334 1.553 |
| **median** | **1.296x** |
| same-arm control, `A(i+1)/A(i)` | median **1.034**, range **0.782 to 1.231** |
| pooled medians | A 44.727 s, C 35.868 s -> 1.247x |

Seven of eight pairs are above 1.13 and the median is outside the control's
whole range. **The eighth is 0.983 and is printed rather than dropped**: it is
the pair whose `other_hot` counter reads 9, the highest external load in the
run, and a row that quotes only the seven would be selecting on the outcome.

**An earlier replicate on a quieter box agrees**, taken on a pre-guard build of
the same change at loadavg 14 to 29: ten pairs, **median 1.331x**, one outlier
at 0.872 at that run's highest external load, same-arm control median 1.032 and
range 0.896 to 1.203. Two runs, two binaries, two load regimes, 1.30x and 1.33x.

**On aarch64 the same lever is REFUTED ON BOTH HOSTS, and it is not landed.**
The arm that raises `kMrNeon` from 4 to 6 (with the same padding, which on NEON
fires because `16 % 6 = 4`) is **slower on thor and no better on GB10**:

| host | arm A GEMM set | arm C GEMM set | C/A | arm A's own spread |
|---|---:|---:|---:|---:|
| thor | 29.637-30.323, median **29.956** | 31.229 31.442 31.252, median **31.252** | **1.043** | 2.3% |
| GB10 | 32.514-34.699, median **32.863** | 33.063 33.483 33.894, median **33.483** | **1.019** | 6.6% |

On thor it is outside arm A's own spread and on GB10 it is inside it, so the
honest reading is a small regression on one host and nothing on the other. The
connector layer moves with it on neither: 3.978 s against 3.909 s on thor,
4.861 s against 4.854 s on GB10. `kMrNeon` therefore stays 4. **The x86 figure is not carried across**, which is the
failure `LTX25-CONNECTOR-GEMM`'s own `### RETRACTION` section paid for one row
ago: its "one thread beats twenty" held on AVX-512 and was false on both aarch64
hosts.

**THE DECODE SHAPES ESTABLISH NOTHING ON THIS BOX, and that is reported rather
than filled in with the prefill result.** Six pairs at `--reps 10` on each of
the two row counts a decode step uses, at loadavg ~49 with another agent
working:

| rows | paired ratios | median | same-arm control |
|---:|---|---:|---|
| 16 (a `c16` step) | 1.281 0.809 1.397 0.836 1.373 2.240 | 1.327 | median **1.340**, range 0.534-1.681 |
| 1 (one token) | 1.783 1.184 1.269 1.175 0.848 0.832 | 1.180 | median **0.789**, range 0.618-0.994 |

**Each control swallows its own effect**, so neither row is a result. What is
NOT at risk is stated from the source instead: at `nrows = 1` the guard declines
the padding (`nrows < mr`) and `mr_end` equals the tile start, so the M-blocked
loop does not execute and the emitted work is the one-row kernel exactly as
before. M4 corroborates it from the other side -- a perturbation confined to the
padded remainder does not fire there. At `nrows = 16` the tile is two `mr = 8`
passes against six at `mr = 6`, so an improvement is what the mechanism
predicts; the measurement is consistent with one and does not establish it.

**On NEON the padding is INERT at these shapes** and that is stated rather than
implied: `mr = 4` divides the 16-row tile, so `rem = 0` and the guard declines.
What the padding does on aarch64 is cover the ragged tiles a non-multiple `M`
produces, and that is what the byte-equality gate exercises there.

### The correctness evidence

**Byte equality is the whole gate, and it is proven on BOTH architectures.**
`tests/vt/test_ops_matmul_elem_mblock.cpp`: **3 cases / 1,802 assertions / 0
failed** on x86-64, and the SAME counts on a pristine `main` tree, which is what
says the reference is a valid oracle rather than a transcription of the new
code. **The same binary built from this branch's own head inside the GB10 lease
reads 3 / 1,802 / 0 as well, `test_mblock_exit=0`** -- an aarch64 pass is not an
x86 pass and this row did not assume one.
`tests/vt/test_ops_matmul_elem.cpp` is unchanged and green on both trees.

**A mutation per claimed guarantee, each proved to have applied and built before
its result was read.** The harness refuses an unapplied edit and a failed build
in those words, because both otherwise read as a passing test.

| ID | mutation | expected | result |
|---|---|---|---|
| M1 | drop the store clamp, so the pad rows are written out | RED | **RED** — `malloc(): invalid size`, the heap corruption an unclamped store causes |
| M2 | do not zero the pad rows | PASS | **PASS** — see below |
| M3 | reverse the K order of the LAST M-blocked row in `BtM6Avx512` | RED | **RED**, 333 of 1802 |
| M4 | perturb ONLY when `rows_here < mr`, i.e. only on the padded remainder | RED | **RED**, 360 of 1802 |
| M5 | perturb on `ir1_start > 0`, a chunk-position-dependent defect | RED | **RED**, 666 of 1802 |
| M6 | `kMrAvx512 = 9`, past the accumulator tile | BUILD FAILS | **BUILD FAILED**: `static assertion failed: cpu_ops.cpp's accumulator tile bounds mr` |
| M7 | wrong at ONE worker count only (`nth == 5`) | RED | **RED**, 144 of 1802 |

**M4 is the reachability mutation.** It perturbs the padded remainder and
nothing else, and the gate reds through `vt::MatmulBT` / `vt::Matmul` — the
production entry points — so the new path is entered by the shipped call and not
only by a test that constructs it.

**M7 is why T3 exists, and it is the M9 precedent reproduced.** On the M7 binary
the whole-dtype-matrix case passes **864 of 864** assertions and only the
seven-worker-count case fails, **144 of 936**. A single worker count does not
test this claim, and that sentence is now executable rather than argued.

**M2 PASSED and that is recorded rather than dressed up as a guarantee.** Not
zeroing the pad rows changes no output, because the pad rows' products are never
stored. The `memset` is there so a `thread_local` buffer cannot carry a previous
call's operands -- including the inf and nan values
`test_ops_matmul_elem.cpp` feeds -- into the FP pipeline. It is a hygiene
property, not a value property, and this row does not claim a gate it does not
have.

### Where the evidence is

`/mnt/nas_share/rc/ltx25-connector-repair/run/`, which a leased worker sees as
`/workspace/ltx25-connector-repair/run/`:

| directory | host | what is in it |
|---|---|---|
| `rc-worker-4b8lj-20260831T224259Z` | GB10, `dgx:gpu0` | `log.txt`, `tier-{A,C}.txt`, `connector-{A,C}.txt`, `attn-{A,C}.txt`, `gemm-{A,C}.txt`, `gemm-portable.txt`, `test-mblock.txt` |
| `rc-worker-n8smh-20260831T221041Z` | thor, `thor:gpu0` | the same set |

`run.sh` and the two source tarballs it builds from sit beside `run/`.
`src-landed.tar.gz` is a `git archive` of this branch's head, so the binary that
ran the byte-equality gate under the lease is built from the tree this pull
request lands.

### The gates

- `tests/vt/test_ops_matmul_elem_mblock.cpp`: 3 / 1802 / 0, on the changed tree
  and on a pristine base tree, and 3 / 1802 / 0 again on aarch64 inside the GB10
  lease. On the MERGED tree, `test_ops_matmul_elem.cpp` and
  `test_ops_matmul_elem_mblock.cpp` together read **10 cases / 2,487 assertions
  / 0 failed**, which is the pair of suites run against the tree that lands
  rather than against the tree the change was written on.
- Seven mutations, every one matching its prediction.
- The A/B above, two SEPARATE build directories, binaries asserted to differ by
  sha256 before any timing.
- `scripts/check-tree-compiles.py`: **612 of 612** translation units before the
  merge (a header is in scope, so it ran the 1223-unit dependency scan first),
  and **9 of 9** after it, when the range narrowed to this branch's own paths.
- `scripts/agent-preflight.sh`: every checker green after the anchor repair.
  `check-agent-record` failed first at `stale: 29 > baseline 28`, because this
  branch's own edit above `DFlashBlockAttentionKernel` moved it off
  `cpu_ops.cpp:3143`; the anchor is repointed to `:3190` in
  `.agents/kernel-matrix.md` in this change, which is the rule that a record
  edit rides in the change whose edit made it stale.

**One suite fails and it is the box, not this branch.** It is the ONLY failing
gate on the final run. `test_cpu_x86_llamacpp_floor` reds with its own words,
and WHICH of its ten cases reds moves between runs -- `NO_QUIET_WINDOW` in the
run quoted below, `test_a_contended_leg_is_discarded_and_never_summarised` in
the next -- which is itself the signature of a load-dependent refusal rather
than a defect: `NO_QUIET_WINDOW after
30s (busy=113% builders=0 load=50.83 45.74 39.03)`. The harness refuses to
measure without a quiet window and this devbox was at loadavg ~50 with another
agent on it. **The control was run rather than assumed**: the same suite, from
the shared checkout, on a tree carrying none of this branch, fails identically
with `NO_QUIET_WINDOW after 30s (... load=44.90 44.39 39.82)`. And
`git diff --name-only origin/main...HEAD` on this branch names ten paths, none
of them under `tests/scripts/`, `benchmarks/` or that harness. It is a
load-dependent refusal in a harness this row does not touch, and it is not a
verdict about this change.

### What this row does not claim

One request, one geometry, one probe. **No render was run**, so the render-level
figure above is a projection and is labelled as one. Every x86 number is one
shared devbox at a stated load, while both aarch64 hosts were leased and idle.
And the GEMM that is left is **not at a ceiling**: it runs at 137.6
GFLOP/s on 14 Arm cores and 172 to 236 GFLOP/s on 20 AVX-512 ones, against a
mul-then-add roofline several times higher, so what follows is a named
hypothesis and not a limit.

## Owed

- ~~**THE END-TO-END RENDER IS NOT RE-MEASURED, and the harness is why.**~~
  **ANSWERED, and the projection was right on its own leaf and conservative on
  the render.** `LTX25-RENDER-CONFIRM` ([#2457](https://github.com/mudler/vllm.cpp/issues/2457))
  advanced the pin the way this bullet asked -- it built the head, re-took
  #1864's blockiness verdict on that same binary, and only then timed it. `rc`
  job `93a60151` on `dgx:gpu0`: **`VERDICT PASS`** with margins +0.113 and +0.124
  on the two gated ratios, then **302.954 s at n = 3, spread 8.03%**, against
  this row's 518.398 s baseline. That is **1.711x** and **3.230x the oracle**,
  where the projection above said ~380 s and ~4.05x. **The leaf this row
  projected was met**: `conditioning.connector.compute` + `guiders.connector.compute`
  read **81.338 s** against #2354's 224.882 s, a **2.765x**, where 87.9 s was
  projected. The render beat its projection because `4fef1f413`
  (`LTX25-AUDIO-DECODE-COST`) took `decode.audio.mel` from 47.175 to 4.926 s in
  the same range, which no connector projection carried. `denoise` is unchanged
  at 15.129 -> 15.122 s and `load` at 94.483 -> 94.550 s, so the wall moved for
  the leaves it was predicted to move for. Owner: `LTX25-RENDER-CONFIRM`, closed.
- **THE REMAINING GEMM IS THE LEAF, at 65.3% on GB10 and 74.7% on thor, and the
  contained lever does not reach it.** After this change the connector's cost on
  aarch64 is `vt::MatmulBT` at 125.5 GFLOP/s on GB10 and 137.6 on thor, and the M-blocking lever is refuted there.
  The next traceable hypothesis is the one
  `.agents/specs/cpu-elem-gemm-wide-isa-and-tiling.md` already carries as a
  SPIKE with no active row: `KERNEL-GEMM-CPU-TILED`, a K-blocked macro-kernel
  that keeps its accumulators live across K panels. That is bit-exact by the
  same argument this row used -- continuing one f32 accumulator across a split
  `p` loop is the same addition sequence -- and it is what turns the weight from
  something re-read per `mr` rows into something read once per panel. **It is
  not opened here**, because it is a `vt` seam campaign and this row's scope
  excludes it. Owner: unowned; sizing is here.
- **A DEVICE ARM FOR THE CONNECTOR remains the other branch and remains a PORT.**
  `Ltx2Attention` interleaves host `RmsNormRows` and `Ltx2ApplyRotaryEmb` on raw
  `float*` between its GEMMs and `Ltx2ConnectorForward` reads its weights as
  host `std::vector<float>`, so it is a weight-arm port with its own numerics
  gate. #2354 and `LTX25-CONNECTOR-GEMM` both say so and this row does not
  contradict them. Owner: unowned.
- **The `mr` sweep was two points, not a curve.** AVX-512 was measured at 6 and
  8 and NEON at 4 and 6. Whether AVX-512 at some other `mr`, or a restructured
  kernel that decouples `mr` from the register file by spilling the accumulator
  tile to L1, does better is unmeasured. Owner: unowned.
