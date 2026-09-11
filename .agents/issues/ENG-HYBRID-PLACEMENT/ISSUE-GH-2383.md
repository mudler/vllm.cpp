ID: ISSUE-GH-2383
Title: The MoE placement seam assumed bf16 and truncated an f32 block, and the placed branch has no CPU-reachable test
Row: ENG-HYBRID-PLACEMENT
State: CLOSED
Kind: UNKNOWN
GitHub: 2383
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `include/vllm/model_executor/moe_placement_seam.h` hardcoded `vt::DType::kBF16` in six places and never compared it against the block it was handed:
>
> ```cpp
> const size_t bytes = T * H * vt::SizeOf(vt::DType::kBF16);
> ...
> dense_attn::DBuf placed_in(placed, vt::DType::kBF16, {T, H}, staging.data());
> ```
>
> `src/vllm/model_executor/models/kimi_linear_device.cpp:930` builds `DBuf dh2(d, DType::kF32, {T, H})` and hands `dh2.t()` straight to `RunMoePlaced`. On the **placed** branch that copies half the bytes and reinterprets f32 as bf16 — silently, with no error, producing plausible-looking floats rather than a crash. Two further Kimi-Linear sites (`:1231`, `:1616`) pass a stream dtype that is not guaranteed bf16 either.
>
> ## Why it never fired, and why it can now
>
> It could not fire before the placement plan was installed: `placed_on` always equalled the engine device, so the entire placed branch was dead code. Installing the plan is what made it reachable, so it is fixed in that same change rather than left as a latent trap behind a newly-opened door.
>
> ## Why no test caught it, which is the more important half
>
> `RunMoePlaced` derives the engine device from `engine.q.device.type`, and the CPU is the only legal placement target. So on a CPU-only build `placed_on == engine_device` **always**, and the placed branch is unreachable. Every unit test, and all of CI, exercises only the inert path. The placed branch's sole execution is on a GPU box.
>
> That is a coverage hole, not an oversight in one file: an entire branch of a shared seam that six architecture families route through has no automated coverage anywhere a merge gate can see it.
>
> ## Fix
>
> The seam now carries the block's own dtype (`dh.dtype`), sizes the copy-back by the dtype the body actually produced (which need not equal the input's), and no longer hardcodes bf16 anywhere. The `VT_PLACEMENT_DUMP_MOE` hook is dtype-aware too and refuses a dtype it cannot render rather than printing misread bytes for a gate to compute an NMSE over.
>
> `tests/vllm/model_executor/test_placement_dump_dtype.cpp` covers the dump's half — in its own binary, because the env var latches on first use and the existing seam tests would latch it first. Mutation-proven: reverting the dtype branch, compiling clean at rc=0, turns it red at 1 case / 4 assertions.
>
> ## Owed
>
> The seam's placed branch itself remains untested on a CPU-only build, for the structural reason above. Closing that needs either a loopback placement target (a second CPU-backed device the seam would treat as remote) or a GPU-gated test. Not attempted here; named so it is visible debt rather than an assumption that the fix is covered.
>
> Row: `ENGINE-HYBRID-PLACEMENT`

## Resolution

The binding comment https://github.com/mudler/vllm.cpp/issues/2383#issuecomment-5480893545 dated 2026-08-31 identifies fix commit `5f230020f`, its dtype-aware copy-back, and the mutation-proven dump test. It preserves the separate placed-branch coverage debt.
