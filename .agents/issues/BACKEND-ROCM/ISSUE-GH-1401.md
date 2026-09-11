ID: ISSUE-GH-1401
Title: GdnPostConvK: one thread copies the whole value_dim row, setting the kernel's duration (19% of ROCm decode)
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: 1401
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-19
Updated: 2026-08-21
Closed: 2026-08-21

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `GdnPostConvK` spends its whole duration in one thread. The kernel decomposes by
> `(token, head)` into `t * (hk + 1)` items, and the single item with `head == hk`
> copies the ENTIRE `value_dim` row while the other `hk` items do about `4 * dk`
> element-ops each.
>
> At decode (`t = 1`) on Qwen3.5/3.6 geometry — `Hk=16, Hv=32, Dk=Dv=128` — that is:
>
> ```
> value_dim = hv * dv = 4096
> items     = t * (hk + 1) = 17          -> ONE workgroup, 17 of 256 threads active
> head < hk (16 items) : ~512 element-ops each
> head == hk (1 item)  : 4096 element-ops + hv transcendentals
> ```
>
> One lane does 4096 elements while sixteen do 512 — an 8x imbalance on top of
> using one CU of 32. The kernel's duration is set by that lane.
>
> Measured on RX 9060 XT (gfx1200), ROCm 7.2.3: **413-479 us per call, 30 calls
> per token, 11.24 ms/token = 19% of all GPU decode time** — the second largest
> kernel in the profile behind the activation quantizer (#1400, #1294).
>
> Unlike the quantizer there is nothing to hoist: `Ld`/`St` are compile-time
> overloads and the kernel is already templated on all three dtypes. This one is
> purely a decomposition problem.
>
> ## Fix
>
> Give a token `hk` q/k slots, then `v_chunks` copy chunks of `v_chunk` elements,
> then one slot for the gates, with `v_chunk = dk`. Every element operation is
> identical and independent — the v copy is elementwise, the gates are per-`h` —
> so this changes WHICH thread does a given element and never the arithmetic.
>
> ```
> items: t*(hk+1) = 17  ->  t*(hk + value_dim/dk + 1) = 49
> max per-thread element-ops: 4096 -> 512
> ```
>
> ## Measured
>
> Same-binary A/B, `VT_ROCM_GDN_POSTCONV_CHUNK=0` restores the original kernel.
> Kernel figures from `rocprofv3 --kernel-trace --stats`, same token count both
> arms. Decode figures are warm samples with the arms interleaved.
>
> | model | arch | per call OFF -> ON | | decode |
> |---|---|---|---|---|
> | Qwen3.6-14B-A3B-VibeForged Q4_K_M | `qwen35moe` | 468.6 -> 84.1 us | 5.6x | 12.560 -> 14.170 tok/s (1.128x) |
> | Qwen3.6-35B-A3B UD-Q4_K_S | `qwen35moe`, 7 GiB host-resident | 479.1 -> 83.0 us | 5.8x | 7.846 -> 8.579 tok/s (1.093x) |
> | Ornith-1.5-9B Q4_K_M | `qwen35` DENSE | 413.2 -> 76.5 us | 5.4x | 16.275 -> 18.451 tok/s (1.134x) |
>
> The kernel lands at 76-84 us on every model regardless of dense/MoE or
> resident/offloaded, which is what the geometry predicts: per-call cost is set by
> `Hk`/`Hv`/`Dk`/`Dv`, identical across all three.
>
> **Output byte-identical on all three models, both arms.** The dense 9B reaches
> the kernel through `Qwen3_5ForCausalLM` with no MoE block and no offload, so
> nothing else in the path could account for the win.
>
> The 35B's smaller end-to-end ratio is expected: its decode is PCIe-bound from
> offloaded experts, so the GDN win is diluted.
>
> ## Caveats
>
> - Decode-shaped. At larger `t` the original spreads across more items and the
>   imbalance shrinks, so this will not show in a throughput benchmark.
> - One board. `Hk=16, Hv=32, Dk=Dv=128` is the only geometry measured; the fix is
>   shape-general but the numbers are not.
> - Does not address occupancy. `items` goes 17 -> 49, still one workgroup. Full
>   element-parallelism (~8192 items) needs a wave-per-head reduction for the q/k
>   sums and is a separate change.
>

## Resolution

GitHub records closing pull request #1402 (https://github.com/mudler/vllm.cpp/pull/1402) merged on 2026-08-21 as commit `f4ccabbb4d23f216a8679346e479fdf873e22f3b`. GitHub closed issue #1401 on 2026-08-21.
