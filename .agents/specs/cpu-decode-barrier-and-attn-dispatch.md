# CPU backend — the batch-1 decode barrier and the paged-attention dtype switch

**Row:** `PERF-CPU-DECODE-BARRIER`
**Issue:** [#391](https://github.com/mudler/vllm.cpp/issues/391)
**Measurement that motivated it:** [#333](https://github.com/mudler/vllm.cpp/issues/333), landed `6dbedf9f`
**Profile that located it:** `.agents/benchmark-record.md`, 2026-08-06 CPU profile

## 0. What is and is not claimed

**This is a CPU-backend row.** The CUDA path is untouched, and nothing here
bears on our vLLM parity numbers, which live on CUDA.

**vLLM is not the bar here and cannot be.** The pinned oracle carries no
`muse_glimmer`, so that axis stays an open gap by construction (spec
`muse-glimmer.md` §0). The bar for this row is **llama.cpp on the same GGUF**,
labelled secondary, exactly as `.agents/benchmarking.md` requires.

**No ceiling is declared.** The 0.997x prefill tie at 128 tokens is the reason:
the kernels are competitive once overhead is amortised, so the gap is what we
add per operation, and that is addressable rather than intrinsic.

## 1. The evidence

Against llama.cpp on the same 16.7 GB Q4_K_M file, idle GB10, 20 ARM cores,
bands calibrated before any delta was read:

| Workload | Axis | ours | llama.cpp | ratio |
|---|---|---|---|---|
| in128 t=10 r=3 | prefill | 9.94 | 9.97 | **0.997x TIE** |
| in128 t=10 r=3 | decode | 1.31 | 6.41 | 0.204x |
| in128 t=20 r=5 | prefill | 11.63 | 12.94 | 0.898x |
| in512 t=20 r=3 | prefill | 2.23 | 13.13 | 0.170x |
| in512 t=20 r=3 | decode | 0.29 | 5.00 | 0.058x |

**The shape is the finding, not any single ratio.** llama.cpp is flat from 128
to 512 tokens while ours falls 81% (prefill) and 75% (decode) — the signature of
fixed per-operation overhead rather than slow kernels.

And we are not idle. A two-length diff cancelling the load phase puts our decode
at **1987% CPU against llama.cpp's 1810%**, so every core is busy while burning
**5.6x the CPU-seconds per token**. Spinning, not starved.

## 2. Lever 1 — decode is synchronisation-bound

From the 2026-08-06 profile:

> Threadpool synchronisation is **47.15% of decode** (`ThreadReady` +
> `PollForWork` + `Barrier`), rising to **~58%** on the secondary-thread view.
> At M=1 the per-op work is too small to amortise the barrier, so decode is
> **synchronisation-bound, not kernel-bound.**

Site: `src/vt/cpu/cpu_threadpool.{cpp,h}`.

**Explicitly not the defect:** the elementwise GEMM (`BtM4Neon`/`Bt16Neon`,
21.5% prefill / 24.9% decode) is already on its optimised NEON tier and is what
remains once avoidable work is removed. **No GEMM kernel change can reach this
lever** — the profile says so directly, and a change there would be motion
without effect.

Directions worth measuring, none pre-committed: raising the parallel-dispatch
threshold so M=1 ops run inline on the calling thread instead of paying a
barrier; a cheaper wait (spin-then-park) on the critical path; or fusing
adjacent tiny ops so one barrier covers more work.

## 3. Lever 2 — paged attention branches per element

~39% of prefill is the paged-attention inner loop, ~21% of it inside a
per-element dtype switch at `src/vt/cpu/cpu_paged_attn.cpp:29`:

```c++
float LoadF32(const Tensor& t, int64_t elem_offset) {
  switch (t.dtype) {
    case DType::kF32:  return t.Ptr<float>()[elem_offset];
    case DType::kF16:  return F16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_offset]);
```

A branch per element on a value that is **constant for the whole tensor**. Hoist
the dispatch out of the loop — the profile calls this "a known, already-solved
defect class", so mirror however the already-fixed sites in this tree did it
rather than inventing a new shape.

## 4. Gates

**Correctness first, and it is not negotiable for a perf row.**

- Every existing CPU gate stays green, unchanged.
- If a change alters numerics at all, that is a finding to report, not a
  tolerance to widen. A dtype-dispatch hoist should be **bit-identical** by
  construction; if it is not, stop and explain why.
- A threadpool change must not alter results at any batch size. Prove it,
  including at M>1 where the barrier is genuinely needed.

**Speed, per `.agents/benchmarking.md`:**

- llama.cpp on the same GGUF, same harness both sides, same prompts, token
  counts, threads and sampling.
- **Calibrate the noise band from repeated identical legs before interpreting
  any delta.** `r >= 3`; report the spread, not just a median.
- Prefill and decode separately; a single average hides which half moved.
- Same-binary A/B on an idle box, with the box lock held and load recorded.
- Report the **scaling shape** (128 vs 512), since that is what exposed this.

## 5. Risks

- **A barrier change is a correctness risk before it is a speed one.** Running
  ops inline changes ordering and visibility; a wrong relaxation produces
  intermittent wrong output, which is far worse than being slow.
- Measuring on a contended box will manufacture a result. The last attempt was
  stopped twice for exactly this and was right to be.
- Chasing the GEMM because it is large in the profile would be the obvious wrong
  move; §2 says why.

## 6. Stop conditions

- A lever measures neutral or negative → record it as a negative result with the
  number and stop. Negative results are results here.
- A change alters numerics → stop, report, do not widen a tolerance.
- The box is contended → stop and wait rather than publish a manufactured band.

## 7. Outcome

Pending.
