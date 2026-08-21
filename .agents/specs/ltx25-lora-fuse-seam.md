# LTX25-LORA-FUSE-SEAM: route the LoRA delta product through `vt::Matmul`

**Issue:** [#1202](https://github.com/mudler/vllm.cpp/issues/1202).
**Kind:** defect fix in product code. The arithmetic is already a correct mirror
of upstream; only the execution strategy is wrong.
**Filed by:** [`ltx25-decode-speed.md`](ltx25-decode-speed.md)'s campaign as its
lever 7; this row takes #1202 out of that spec's orbit and owns it.
**Sibling:** [`ltx25-text-linear-seam.md`](ltx25-text-linear-seam.md) did the same
job for the caption projection (#1208). This row reuses its method and reaches a
different member of the same seam, for the reason §3 gives.

## Now

**DONE.** The `(B * strength) @ A` product in
`src/vllm/model_executor/models/ltx2_lora.cpp::Ltx2FuseLoraIntoTensor` is
`vt::Matmul`, the shared row-major GEMM seam. At the shipped DiT's projection
width and the shipped distilled adapter's rank one fused tensor went from
**17.7761 s to 0.1242 s, 143.1x**, and the whole fused buffer is **byte-identical
across the two builds** — the same FNV-1a digest `a23e7f876694c537` on every one
of ten runs across both binaries, not a tolerance.

**The bottleneck moved rather than disappearing, and this row says where.** 59%
of what a fused tensor still costs is the single-threaded loop that adds the
delta back into the weight. That is measured in §5, filed as
[#1254](https://github.com/mudler/vllm.cpp/issues/1254), and carried under
`## Owed` — it is deliberately not fixed here, for the reason §6 gives.

**The GB10 wall this addresses is bounded, not converted.** Every number here is
from a 20-core Zen 5. #1202's measurement is from GB10, and the per-core ratio
between the two boxes is unmeasured — the same gap `ltx25-text-linear-seam.md`
carries. §7 states the bound instead of picking a number inside it.

## 1. What was wrong

`Ltx2FuseLoraIntoTensor` computed the rank-`r` product with a scalar
single-threaded triple loop:

```cpp
for (int64_t o = 0; o < rows; ++o) {
  const uint16_t* brow = bs.data() + o * pair->rank;
  for (int64_t i = 0; i < cols; ++i) {
    float acc = 0.0F;
    for (int64_t k = 0; k < pair->rank; ++k) {
      acc += vt::BF16ToF32(brow[k]) * vt::BF16ToF32(pair->a[k * cols + i]);
    }
    agg[o * cols + i] = vt::F32ToBF16(acc);
  }
}
```

One thread of twenty, no blocking, no SIMD, an out-of-line `vt::BF16ToF32` per
multiply, and an inner operand `pair->a[k * cols + i]` striding by `cols` so
every one of the `rank` innermost loads is its own cache line.

#1202 measured it on `dgx` (GB10, 20 cores) loading the full 21.004 B
transformer with the shipped 8.9 GB distilled adapter: three identical `gdb`
stacks (`vt::BF16ToF32` <- `Ltx2FuseLoraIntoTensor` <- `Ltx2LoadDitFromSafetensors`
<- `Ltx2VideoEngine::Load`), one thread at 99.9% of one core with 19 idle, and an
f32 working set growing 9.432 -> 10.235 GiB over 300-629 s — 2.3% of one pass in
10.4 minutes. Cross-checked against the sum of `out * in * rank` over the 1660
targeted modules, **8.53e12 MAC**.

It sits between `Load` and any generation, so it blocks every LoRA-bearing
pipeline kind on the full model. `one_stage` renders today precisely because
upstream marks it `Full` with no adapter and it never enters this function.

## 2. Upstream anchors, and what must not change

The port mirrors `ltx-core loader/fuse_loras.py` at Lightricks/LTX-2 @
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`. vLLM does not register LTX-2.5, and
`vllm-omni` is `UNPINNED` / `gateable = no`, so `ltx_core` itself is the oracle —
the same standing this row's sibling records.

Three properties are load-bearing, and all three are unchanged by this row:

1. **`B * strength` rounds to bf16 BEFORE the product** (`fuse_loras.py:113`: a
   bf16 tensor times a Python float stays bf16). It is not folded into the
   accumulation. Gated by `test_ltx2_lora`'s "the delta accumulates in BF16, not
   f32" case at rank 192.
2. **Accumulation is f32, the store is bf16** — torch's bf16 matmul. Gated by
   "the matmul RESULT is rounded to bf16 before the weight is added".
3. **`deltas.add_(weight)` then `.to(dtype=weight.dtype)`** (`fuse_loras.py:67-68`),
   which is why the f32 target branch rounds its sum through bf16. Gated by "the
   f32 target branch rounds through the bf16 accumulator".

Both refusals are kept and both keep their tests: the shape-mismatch `Fail`, and
the second-adapter `Fail` over upstream's second aggregation form
`addmm_(B, A, alpha=strength)` (`fuse_loras.py:115`), which rounds differently
and is deliberately unimplemented.

**The accumulator is NOT narrowed or widened by this row.** The sibling row
narrowed f64 -> f32 because `F.linear` accumulates in f32 and the local code had
diverged. Here the code was already f32 and already matched, and `vt::Matmul`'s
contract is f32 accumulation, so there is nothing to reconcile. Changing it would
have been a divergence introduced by a performance row, which is the shape of
defect the dtype-polarity rule exists to catch.

## 3. Why `vt::Matmul` and not `vt::MatmulBT`

The seam has two members and the sibling row took the other one, so the choice
needs a reason rather than a habit.

| | shape | our operands |
|---|---|---|
| `vt::Matmul` | `out[M,N] = a[M,K] @ b[K,N]`, both row-major | `agg[rows, cols] = bs[rows, rank] @ A[rank, cols]` — **exact** |
| `vt::MatmulBT` | `out[M,N] = a[M,K] @ b^T`, `b` is `[N,K]` | would need A as `[cols, rank]`; the checkpoint stores `[rank, cols]` |

`Ltx2LoraFactorPair` holds A as `[rank, in]` row-major because that is how the
adapter stores it (`ltx2_lora.h`, and `fuse_loras.py:196-198` pairs them that
way). So `kMatmul` is the transpose-free member here, exactly as `kMatmulBT` is
the transpose-free member for a torch `nn.Linear` weight. Taking `MatmulBT`
would have meant materializing a transposed copy of A for no gain.

The device is the CPU queue, built locally, for the same reason the sibling
row's is: the fuser is handed raw host buffers by `MaterializeDitTensor` and
has no queue to inherit.

## 4. The correctness gate is byte equality, and it holds

Not a tolerance, and it did not need to be one. Three independent measurements:

**a. The CPU kernel's own contract.** `src/vt/cpu/cpu_matmul_elem.h` records a
deliberate deviation from ggml: ggml vectorizes each dot ALONG K and finishes
with a horizontal reduce, which reassociates the sum; this tree vectorizes ACROSS
OUTPUT COLUMNS instead, so **every output element keeps the same strictly
sequential f32 accumulation over K the scalar kernel had**, with separate mul and
add and never an FMA under the global `-ffp-contract=off`. Its store is
`StoreF32`, which for a bf16 tensor is the same `vt::F32ToBF16` the loop called.
`cpu_threadpool.h`'s determinism contract adds that parallelism partitions
OUTPUT elements only, so results are bit-identical to `n_threads == 1`.

**b. A direct probe, before any of this was written.** A standalone harness
computing both arms in one process, `mismatched = 0` at every shape tried
(`3072x450x3072`, `128x450x3072`, `3072x32x8192`, `37x7x29`, `64x1x64`), and
`mismatched = 0` again under `VLLM_CPP_CPU_THREADS=1` and under
`VT_CPU_MATMUL_TIER=ref`. The tier lever is worth stating because it is the one
configuration in which routing through the seam is *slower* than the loop it
replaced — 18.09 s against 9.84 s at `3072x450x3072` — and it is still
bit-identical, which is the determinism contract holding rather than a
coincidence.

**c. The A/B harness, across two binaries.** §5's runs print an FNV-1a digest of
the whole fused 4096x4096 bf16 buffer. All ten runs, across the before and after
builds, printed `a23e7f876694c537`.

**d. The focused gate on every CPU GEMM tier.** `VT_CPU_MATMUL_TIER` selects the
micro-kernel without a rebuild, and `test_ltx2_lora` is 15/15 with 114 assertions
under `portable`, `sse2`, `avx2`, `avx512` and `ref` alike. This is here because
the obvious objection to a byte-equality gate is that it was written on one
machine's ISA: CI builds arm64, and a gate that only held on AVX-512 would red
there. It holds on the portable tier, which is the one with no arch SIMD at all
and the closest available stand-in for a different micro-kernel. The Arm tier
itself is not exercisable on this box and is left to CI.

So the byte-equality assertion in the gate is a genuine oracle rather than a
tolerance that was widened until it passed, and no existing golden or tolerance
moved.

## 5. The measurement

**Harness.** A scratch A/B driving the **production** fuser —
`Ltx2LoraAdapter::Open` on a real safetensors adapter, then
`vllm::Ltx2FuseLoraIntoTensor` — with **identical source and compiler flags on
both arms**; only `libvllm.a` differs. Geometry is the shipped DiT's projection
width (`inner_dim = num_attention_heads * attention_head_dim = 32 * 128 = 4096`)
and the shipped distilled adapter's rank (450), so `4096 x 450 x 4096` is one
real `to_q`-shaped module: 7.55e9 MAC.

**Recipe.** AMD Ryzen 9 9950X3D, 20 vCPU under KVM, AVX-512; g++ 13.3.0;
`-O2 -std=c++17 -ffp-contract=off`; CMake `Release`,
`-DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF`; box **not idle**
(loadavg 1.73 at the start, 2.09 at the end); five repeats per arm.

| arm | wall (median of 5) | spread | fuser throughput | digest |
|---|---:|---:|---:|---|
| before, scalar triple loop | **17.7761 s** | 17.7324–18.1529 | 0.4247 GMAC/s | `a23e7f876694c537` |
| after, `vt::Matmul` | **0.1242 s** | 0.1235–0.1355 | 60.86 GMAC/s | `a23e7f876694c537` |

**17.7761 s -> 0.1242 s, 143.1x**, one core of twenty becoming many.

**Where the after arm's time actually goes, because it is no longer the GEMM.**
Re-running the same harness at `rank = 1` keeps the output size and makes the
GEMM negligible, so it measures the surrounding per-element work:

| | `rank = 450` | `rank = 1` |
|---|---:|---:|
| after | 0.1242 s | **0.0733 s** |
| before | 17.7761 s | 0.0955 s |

So the GEMM is now ~0.046 s (~164 GMAC/s, matching the standalone probe's 168)
and **~0.073 s — 59% of the call — is the aggregator zero-fill plus the
single-threaded bf16 add-back loop**, three out-of-line conversions per element.
Before this row that loop was 0.5% of the call and correctly ignored. It is
[#1254](https://github.com/mudler/vllm.cpp/issues/1254).

**Decomposition of the 143x**, from the standalone probe at `3072x450x3072`:
`VLLM_CPP_CPU_THREADS=1` gives 27.5x (SIMD micro-kernel, 16-lane output
blocking, inlined widening, and an operand order that stops striding a cache
line per load), and the remaining 14.5x is the threadpool over 20 cores.

**Projection to the full model, stated as a projection.** Across the 1660
targeted modules of the 21.004 B DiT (8.53e12 MAC, and ~1.9e10 output elements
if the whole adapter is rank 450) the product falls from ~20,280 s to ~52 s on
this box, and #1254's loop is ~79 s of the ~131 s that remains. This is
arithmetic on measured rates, not a completed full-model pass — no such pass has
run here.

## 6. What this row deliberately does not do

* **[#1254](https://github.com/mudler/vllm.cpp/issues/1254), the add-back loop**,
  even though it is now the majority of the call and its seam (`vt::Add`) exists
  with the matching contract. The bf16 branch is a clean match; the **f32 branch
  is not**, because it rounds the sum through bf16 before an f32 store to mirror
  `deltas.add_(weight)` on a bf16 aggregator (`fuse_loras.py:67-68`), and
  `vt::Add` with an f32 output would silently skip that rounding and be more
  precise than the oracle. That asymmetry is either a recorded exception or a
  seam extension, and AGENTS.md sends a fix that needs its own argument down the
  normal row path rather than into this flow.
* **[#1210](https://github.com/mudler/vllm.cpp/issues/1210), the two-stage
  rebind.** A two-stage recipe fuses at load, un-fuses for phase 0, and re-fuses
  for phase 1, so the load-time pass is provably wasted. This row makes each pass
  ~143x cheaper and **leaves the wasted round trip exactly where it was**. The two
  fixes are independent and #1210 owns the second.
* **N-adapter fusion.** The second-adapter refusal stays, and `LTX25-IC-LORA`
  still owns it.
* **Any GB10 number.** §7.

## 7. What is not established

* **The GB10-to-x86 per-core ratio.** #1202's stacks, thread count and rate are
  from GB10; every number in §5 is from a 20-core Zen 5. The two are not
  interconvertible without a measurement neither this row nor its sibling has.
  What can be said is direction and shape: the defect is the same code on both
  boxes, the fix removes the same two axes (single thread, scalar inner loop) on
  both, and GB10 has 20 cores as well, so the threading component of the 143x is
  available there. The SIMD component is not transferable — GB10 is Arm and takes
  the NEON tier, not AVX-512. **No GB10 speedup is claimed.**
* **A completed full-model fusion pass.** §5's full-model line is arithmetic on
  measured rates. The 8.53e12 MAC total is #1202's, and #1202 itself extrapolated
  a rate over a 10.4-minute window rather than completing a pass.
* **Whether the fusion is still the dominant pre-generation cost on the full
  model after this row.** It very likely is not — #1254 and #1210 are both larger
  than what is left here — but nothing measured that, and saying so is cheaper
  than a reader assuming it.
* **Anything about the shipped 8.9 GB distilled adapter itself.** No real adapter
  was read on this box; §5's adapter is synthetic at the real geometry.

## 8. Tests and gates

**Focused gate:** `test_ltx2_lora` and `test_ltx2_loader`.

**New case — `test_ltx2_lora`, "the delta product runs on the shared vt::Matmul
seam".** Two assertions that must both hold, because either alone is vacuous
here:

1. **Routing.** `vt::GetOpProviderStats(OpId::kMatmul, DeviceType::kCPU).selections`
   advances by exactly one per fused tensor, and `last_selected` is
   `vt::kNativeProviderName`. This is the assertion that discriminates: the
   arithmetic did not change, so a value check alone passes on the pre-row code.
2. **Byte equality** of the whole fused buffer against a second, independent
   transcription of `fuse_loras.py:113` — the scalar loop this row replaced,
   written out in the test rather than called back into the fuser. At three
   shapes (`19x5x37`, `64x13x96`, `1x192x1`) straddling the seam's 16-wide output
   block and 16-row activation tile, on values with full f32 mantissas so every
   product and every store rounds. Plus a non-vacuity guard that more than three
   quarters of the outputs actually moved off the weight they started from.

**Reachability — `test_ltx2_loader`, in `CheckArmFuses`.** The count above is
taken around `Ltx2LoadDitFromSafetensors`, a production entry point, on all
three arms (FP8, NVFP4, bf16), as an equality against that same call's
`lora_fused_tensors`. `test_ltx2_lora` calls the fuser directly, which proves the
function works and not that a checkpoint load reaches it.

**Red first.** Against the pre-row source, with the tests in place:

* `test_ltx2_lora` — 15 cases / 14 passed / **1 failed**, 113 assertions / 2
  failed. `CHECK(stats.selections == matmuls_before + fused_tensors)` read
  `0 == 3`; `REQUIRE(stats.last_selected != nullptr)` read `nullptr != nullptr`,
  i.e. `kMatmul` on the CPU had never been resolved in that binary at all. The
  byte-equality assertions passed, which is the point — they are the oracle, not
  the discriminator.
* `test_ltx2_loader` — 37 cases / 34 passed / **3 failed**, 64204 assertions / 3
  failed. `CHECK(matmuls_after == matmuls_before + fused.lora_fused_tensors)`
  read `0 == 1` on `fp8`, `nvfp4` and `bf16`.

**Green after.** `test_ltx2_lora` 15/15, 114 assertions; `test_ltx2_loader`
37/37, 64204 assertions. The case count moved on both, so neither is a filter
that matched nothing.

The mutation results are in `## Outcome`.

## 9. Stop conditions

* Stop and report if byte equality fails at any shape: the correctness question
  is then a spec question about reduction order, and it is settled in the spec
  rather than by widening an assertion.
* Stop if the routing assertion cannot be made to red on the pre-row source —
  that would mean it is measuring something other than the routing.
* Do not touch the two-stage rebind (#1210) or the add-back loop (#1254).

## Outcome

**What was measured:** §5. 17.7761 s -> 0.1242 s (143.1x) on one real
projection-shaped module, byte-identical output across both binaries.

**What was rejected, and why:**

* **`vt::MatmulBT`.** §3. The adapter stores A as `[rank, in]`, which is
  `kMatmul`'s `b[K,N]` and not `kMatmulBT`'s `b[N,K]`. The sibling row took
  `MatmulBT` because a torch `nn.Linear` weight is `[out, in]`; the same seam,
  the other member, for the same transpose-free reason.
* **Changing the accumulator.** §2. It was already f32 and already matched the
  oracle. A performance row that silently widened or narrowed it would be the
  dtype-polarity defect.
* **A tolerance instead of byte equality.** §4. Three independent measurements
  said the seam is bit-identical, so a tolerance would have been slack with
  nothing to spend it on — and the place a real reduction-order change would
  later hide.
* **Fixing #1254 in the same flow.** §6. The f32 branch cannot use `vt::Add`
  without dropping a deliberate rounding, and that argument needs its own spec.
* **Hand-rolling threads or SIMD.** Never considered. The seam exists, the
  shared-seam rule is binding, and the CPU kernel already carries the
  determinism contract this row's byte-equality gate depends on.

**Why each default has its value:**

* The guard `pair->rank > 0 && !agg.empty()` is unreachable through
  `Ltx2LoraAdapter::Open` (`ReadFactorAsBf16` refuses an empty factor). It is
  written anyway because the loop it replaced *handled* both cases — a zero trip
  count leaves the zero-seeded accumulator, and `agg` is zero-filled — so the
  replacement is behaviour-preserving rather than merely equivalent where the
  tests happen to look. It also keeps a zero-shaped tensor away from
  `vt::Matmul`, whose contract does not speak to one.
* The queue is constructed locally rather than plumbed through the signature.
  `Ltx2FuseLoraIntoTensor` is handed raw host buffers by the loader and there is
  no queue to inherit; adding a parameter would change the ABI of a function
  whose every caller is on the host.

**Mutations.** Four, each with `BUILT`, the compiler error count, the
`git diff --stat` proving it applied, and the failing assertion by name.
Recorded in `.agents/benchmark-record.md` under this row.

## Owed

| Issue | Lever | State |
|---|---|---|
| [#1254](https://github.com/mudler/vllm.cpp/issues/1254) | the bf16 add-back loop, now 59% of a fused tensor; `vt::Add` has the contract, the f32 branch does not fit it | owed, MEASURED at 0.0733 s of a 0.1242 s call |
| [#1210](https://github.com/mudler/vllm.cpp/issues/1210) | the two-stage rebind still fuses, un-fuses and re-fuses; this row shrinks the constant and leaves the round trip | owed by [`ltx25-decode-speed.md`](ltx25-decode-speed.md), restated here because this row is why the constant changed |

Not owed here, and named so a reader does not go looking: the GB10-to-x86
per-core ratio (§7) is the same open measurement
[`ltx25-text-linear-seam.md`](ltx25-text-linear-seam.md) carries under its own
`## Owed`, and it is not re-filed.
