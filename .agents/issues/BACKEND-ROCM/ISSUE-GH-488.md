ID: ISSUE-GH-488
Title: ROCm gfx1200: PagedAttnOnline is 8.1x slower per call than vLLM's paged-attention kernel (41.1us vs 5.10us)
Row: BACKEND-ROCM
State: CLOSED
Kind: perf
GitHub: 488
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-17
Closed: 2026-08-17

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Summary
>
> On gfx1200, our ROCm paged-attention decode kernel costs **41.1 us per call**
> against vLLM's **5.10 us** on the same board, same model, same workload — an
> **8.1x** per-call gap. It is 11.7% of our decode GPU time versus 2.9% of theirs.
>
> Separate from, and additive to, the skinny-GEMM finding in issue #487:
> that one is about which kernel family we dispatch to, this one is about a
> kernel we wrote ourselves being slow.
>
> ## Measurement
>
> `rocprofv3 --kernel-trace` run **inside the pinned oracle container for both
> sides**, so the same profiler and same ROCm 7.2.3 runtime observe both. Qwen3-0.6B
> bf16, prompt `'The capital of france is'`, 32 generated tokens, greedy. Both
> traces windowed to the decode phase.
>
> | | kernel | calls | avg | share of decode GPU time |
> |---|---|---|---|---|
> | vllm.cpp | `vt::rocm::PagedAttnOnline<bf16,bf16,bf16>` | 868 | **41.11 us** | 11.7% |
> | vLLM `555967922` | `kernel_paged_attention_2d` | 896 | **5.10 us** | 2.9% |
>
> Call counts match (868 vs 896 ≈ 28 layers x 32 tokens), so this is a
> like-for-like per-invocation comparison, not a difference in how often
> attention runs.
>
> For scale, total decode-window GPU busy is 303.7 ms for us against 158.1 ms for
> vLLM. Closing this gap alone would recover roughly 31 ms of that 146 ms
> difference — meaningful, but the GEMV routing issue is the larger share.
>
> ## Where to look
>
> - Ours: `src/vt/rocm/rocm_paged_attn.hip` (1,955 lines), kernel
>   `PagedAttnOnline` — online-softmax, NHD KV via strides, ported from
>   `cpu_paged_attn.cpp` algebra plus the CUDA device-side request lookup.
> - Upstream's decode path at the pin dispatches through
>   `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`, and the kernel that
>   actually ran here is `kernel_paged_attention_2d`.
>
> Note the oracle's own log on this board prints *"Cannot use ROCm custom paged
> attention kernel, falling back to Triton"* — so the 5.10 us figure is vLLM's
> **fallback** path, not its fastest one. The gap to its custom kernel on a
> supported board would presumably be larger still.
>
> ## Not claimed
>
> - **No cause is asserted.** This reports a per-call time difference, not a
>   diagnosis. Occupancy, LDS usage, memory access pattern, and the online-softmax
>   formulation are all untested candidates. A follow-up wants counter collection
>   (`rocprofv3` PMC), not just kernel timing.
> - Single board (RX 9060 XT, gfx1200, RDNA4, ROCm 7.2.3), single run per side.
> - The two kernels are not required to be algorithmically identical; ours may be
>   doing more or different work. Establishing that they compute the same thing
>   for the same inputs is part of any fix, not an assumption of this report.
>
> ## Reproduce
>
> Same procedure as the companion skinny-GEMM issue #487 — our binary runs under
> `rocprofv3` inside the oracle container with `/nix` bind-mounted and
> `LD_LIBRARY_PATH` scoped to the child process.
>

## Resolution

GitHub records closing pull request #767 (https://github.com/mudler/vllm.cpp/pull/767) merged on 2026-08-17 as commit `a7583ac755c7c88dd7de2f8cd2f48c003313d627`. GitHub closed issue #488 on 2026-08-17.
