# sm_120 Qwen3.5 fused GDN decode BV32 RPT2 discriminator

**Rows:** `KERNEL-SSM-MAMBA`, feeding `ROAD-V1-C2-LOCAL-BF16`.
**Lifecycle:** `COMPLETE` — RPT2 was correctness-exact but performance-rejected
and removed at cleanup `f6a0c879141a1887f9fcf8c85a9eb3a8cf8a6275`. BV16 remains the sole opt-in and
BV32 remains the shipped default.

## Gap, anchors and hypothesis

Current cleanup `634ccba70` and result record `3a00d4b82` retain only exact
`VT_GDN_DECODE_BV=16`; BV8/BV24 were slower and removed. The latest same-binary
series measures BV16's dominant grid-y 800 shape at **167.858 us/call** versus
pinned vLLM **128.061 us/call** (**1.311x**). The prior 1.276x run remains
separate variance evidence.

Local `GdnDecodeFusedCuda` -> `DispatchGdnDecodeValueTile` ->
`LaunchGdnDecodeFused{S,NW}` -> `GdnDecodeFusedKernel` is anchored at
`src/vt/cuda/gdn_decode_fused.h:11-67` and
`src/vt/cuda/cuda_gdn.cu:2730-2931`. At Dv=Dk=128, exact BV32 uses NW8 per
value row: grid-x 4, block 256/eight warps and 17,536 bytes dynamic shared.
BV16 obtains block 128/four warps, but doubles grid-x and repeated q/k loads.

Pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` vendors FLA
`fused_sigmoid_gating.py:24-277`: its recurrence is `:122-170`,
`BV=min(next_power_of_2(V),32)` is `:205-212`, grid `(NK,NV,N*HV)` is `:241`,
and `num_warps=4` is `:212,275`. The profiler proves this generated Triton
kernel executes; CUTLASS, cuBLASLt and DeepGEMM are not in this recurrence.

The discriminator combines FLA's BV32/grid-4/four-warp schedule with the
local kernel's exact NW8 arithmetic: each eight-lane row group processes two
independent value rows sequentially. This tests whether BV16's gain came from
four-warp blocks without paying its duplicated grid/q/k traffic.

## Exact design and selector

Add strict opt-in `VT_GDN_DECODE_RPT=2`. Only exact string `"2"` selects RPT2;
unset, empty, whitespace, signs, zero-padding, suffixes/prefixes and every
other value select RPT1. Apply RPT2 only when the selected value tile is BV32
and `dv>=32`; BV16 and `dv<32` remain RPT1 regardless of this environment
value. Existing `VT_GDN_DECODE_BV` semantics are unchanged.

Specialize and unroll RPT at compile time. For RPT2, `groups=ceil(BV/RPT)=16`,
block threads are `groups*NW=128`, and each lane uses
`vi=(tid/NW)+row_iter*groups` for `row_iter=0,1`. Each row retains its same
eight consecutive lanes, Dk slice `[c0,c1)`, shuffle tree, decay, dot, update,
output reduction, output store and state writeback order. Do not change NW,
interleave two rows' recurrence, or reuse one row's accumulator for another.

State-tile load/write use block 128 while preserving the same linear element
mapping and value semantics. The full padded BV32 shared tile remains live, so
Dv=Dk=128 resolves to **grid-x 4, block 128, dynamic shared 17,536 bytes**.
Tail rows stay shuffle-live on zero state and guarded v/output/state accesses.
The existing null indexed-state return and >48 KiB sequential fallback remain
unchanged.

## RED-first tests and mutation review

Extend the CUDA-free contract before product code. It must prove strict RPT
parsing; BV32 RPT1/RPT2 geometry; partial-tile ceiling/tails; `dv<32` fallback;
BV16 unchanged for RPT unset/valid/invalid; and exact callback dispatch into
one RPT1 or RPT2 specialization. Mutations deleting the RPT2 callback,
changing its production binding, row-group count or row stride must be RED.

Extend CUDA GDN coverage to byte-compare candidate output **and full final
state** against both BV32 RPT1 and BV16. Cover production and partial Dv/Dk,
compact and indexed state including a null row, BF16/F32 state and BF16/F32
input/output coverage already supported by the public op. Run both RPT arms in
one process. Mutations that skip the second row, corrupt its stride, remove
the RPT2 production dispatch, or suppress either output/state store must fail.
An actual public CUDA graph node must prove gx=4, bx=128 and smem=17,536.

Focused gates are `test_gdn_decode_fused`, full CUDA `test_ops_gdn`,
`test_qwen35_paged_forward`, staged/full preflight, then fresh static and
targeted-mutation review and independent operator rerun. Every production token
file must equal SHA-256
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`;
any mismatch stops performance work.

## Same-binary decision gate

Under one GPU lease and `flock /tmp/gpu`, use one rebuilt binary and the same
cached Qwen3.5-4B BF16 workload: 128 ShareGPT requests, 128 output tokens,
c32, `max_num_batched_tokens=2048`, 1,280 KV blocks and greedy sampling. Keep
all other selectors identical. Run the counterbalanced sequence
`BV16a -> RPT2a -> RPT2b -> BV16b`, collecting
`nsys --cuda-graph-trace=node`, SQLite, raw summaries, exact token hashes,
binary/head SHA, clocks/contention, peak VRAM and peak PSS.

RPT2 is eligible only if its counterbalanced mean improves grid-y 800 by at
least **1.00%** and improves all fused-decode calls versus BV16, with both
samples moving in the winning direction. It must also be non-regressing on
every enclosing mean: total/output throughput may not decrease; TTFT,
TPOT/ITL, E2E, peak VRAM and peak PSS may not increase. Record values and
ratios for every axis.

If any correctness, geometry, micro or enclosing bar fails, remove the RPT
selector and all product/test arms while preserving the refutation. If all
bars pass, retain RPT2 only as an opt-in; BV16 remains available, shipped
default status and Qwen3.6-27B/35B release gates remain open. No 4B result is
extrapolated to the unavailable release vehicles.

## Result — rejected and removed

Measured product `6ac8bf390` passed the operator gates: portable contract
**4/4 · 125**, CUDA GDN **69/69 · 5,046**, and Qwen3.5 paged-forward
**4/4 · 8**. All four production token files have SHA-256
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.
The intended Dv=Dk=128 launch executed: BV16 grid-x 8 / block 128 /
9,280-byte shared / 50 registers; RPT2 grid-x 4 / block 128 / 17,536-byte
shared / 56 registers.

The counterbalanced `BV16a -> RPT2a -> RPT2b -> BV16b` series contained
1,704 total and 816 grid-y 800 calls. BV16 versus RPT2 means were
**271.577988 -> 372.4151745 ms (+37.1301%)** across all fused decode and
**136.9415415 -> 190.121047 ms (+38.8337%)** at grid-y 800. The enclosing
means also all reject RPT2: total/output throughput
**6732.58/744.47 -> 6709.16/741.88 tok/s**, TTFT
**1026.28 -> 1027.645 ms**, TPOT/ITL **35.00 -> 35.145 ms**, and E2E
**5471.38 -> 5490.87 ms**. Thus RPT2 fails both kernel bars and every enclosing
axis despite exact correctness.

Cleanup `f6a0c879141a1887f9fcf8c85a9eb3a8cf8a6275` removes the selector, specialization and tests.
BV16 remains the sole exact opt-in. Its current-series dominant call is
**167.82 us/call** versus pinned vLLM **128.061 us/call**, about **1.310x**
slower, so default, release, 27B and 35B gates remain open. Evidence is
`/tmp/qwen35-gdn-rpt2-{bv16a,rpt2a,rpt2b,bv16b}-6ac8bf390.*`.
