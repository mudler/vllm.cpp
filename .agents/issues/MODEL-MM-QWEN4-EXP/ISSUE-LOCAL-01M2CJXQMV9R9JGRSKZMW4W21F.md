ID: ISSUE-LOCAL-01M2CJXQMV9R9JGRSKZMW4W21F
Title: the ROCm HcGroupedNormKernel is still one-thread-per-group double after the CUDA arm went fp32 block-parallel
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

The W7 change made the CUDA grouped hyper-connection norm fp32 and block-parallel (`src/vt/cuda/cuda_qwen4_exp.cu:300`, one block per `(token, hc stream)` group, a strided per-thread partial folded in a warp-shuffle tree, launched with `GridForGroups(T * hc)` at `:512`). The ROCm twin did NOT move and now carries the exact defect the CUDA arm just removed.

VERIFIED at `ac04275b8`:
- `src/vt/rocm/rocm_qwen4_exp.hip:220` -- `HcGroupedNormKernel` is ONE THREAD PER (token, hc stream): `for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += step)`, and each thread walks its whole group serially, `for (int64_t h = 0; h < H; ++h)`.
- the accumulator is `double` (`double ss`, `__dadd_rn`/`__dmul_rn`, then `__ddiv_rn` narrowed to float), which the file's own header at `:44-55` justifies as inherited from `cpu_qwen4_exp.cpp`.
- the launch is `HcGroupedNormKernel<<<GridFor(T * hc), kBlock, ...>>>` at `:361` -- a grid sized in THREADS over groups, not in blocks over groups.

WHY IT IS NOW WRONG RATHER THAN MERELY DIFFERENT. `src/vllm/model_executor/models/qwen4_exp_hc.h:99-104` states the contract the CUDA arm was moved onto: upstream runs the norm in fp32 (`self._norm(x.float())`) and vLLM likewise (`x = x.float()`); the double belongs to the CPU HOST REFERENCE, and "the device arm is the thing that must be fp32-accumulate and gated against these numbers". The ROCm arm is a device arm and does not meet that contract. Separately, at model width (`hidden` 2560) the one-thread-per-group shape leaves each thread doing a 2560-element serial walk, which is the cost profile the CUDA change measured out.

NOT A STALE-CLAIM BUG. The ROCm kernel contains no `__syncthreads()` at all (`grep -c` returns 0) and no shared memory, so it carries none of the barrier-ordering prose the CUDA arm had to repair. Its header claims only that every kernel in the file is "a grid-stride loop over independent output elements" and wavefront-size agnostic, which is still true of the code as written.

OUT OF SCOPE HERE. The W7 row is CUDA-only and explicitly did not touch `src/vt/rocm/`. This issue records the divergence so it is visible debt; the fix is a separate row's ROCm perf work, and it owes its own red-first evidence and a gfx-family device gate.

## Resolution

FIXED on `row/MODEL-MM-QWEN4-EXP-HCNORM`, spec `.agents/specs/qwen4-exp-rocm-hc-norm.md`.

`HcGroupedNormKernel` in `src/vt/rocm/rocm_qwen4_exp.hip` is now ONE BLOCK PER
`(token, hc stream)` group in fp32 -- a strided per-thread partial folded in a
`__shfl_down` tree and then across the waves, launched with
`GridForGroups(T * hc)`. That is the CUDA arm's shape and upstream's width
(`qwen4_exp_hc.h:99-104`: upstream and vLLM both run the norm on `x.float()`),
so the contract this issue said the ROCm arm did not meet, it now meets. Every
lane constant comes from `warpSize`, so no wavefront size is assumed.

MEASURED on `strix:gpu0` (gfx1151, wave32, ROCm 7.2.4), 2026-09-13, exclusive
leases:
- `HcGroupedNormKernel` falls from **35.68%** of decode kernel time to **0.56%**
  at an unchanged dispatch count (97.0/step) -- 633.5 us -> 6.6 us mean, a 96x
  per-dispatch change.
- decode **5.4345 -> 8.111 tok/s** median, a **1.49x**, four interleaved launches
  on one boot, spreads 1.60% and 2.00%.
- a new five-case device gate,
  `tests/vllm/models/test_qwen4_exp_rocm_reductions.cpp`, holds the shape at
  widths the `hidden = 7` cross-device case cannot express; five structural
  mutations all convict.
- gate on the reviewed head: battery 5/55/0, cross-device 61/84841/0,
  `-tc='*DSA*'` 2/273/0.

Evidence: `docs/bench-evidence/qwen4exp-rocm-hcnorm-gfx1151-20260913/` and its
`w6-f1-redfirst/`. A fresh hostile review accepted the kernel unchanged and its
six findings are repaired in the same branch.

STILL OWED, and recorded under the spec's `## Owed` rather than left here: a
wave64 measurement (only gfx1151 is in this fleet) and a ROCm race check on the
shared-memory shape.
