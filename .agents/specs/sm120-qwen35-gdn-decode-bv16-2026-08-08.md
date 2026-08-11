# sm_120 Qwen3.5 fused GDN decode BV16 — discriminator spike

**Rows:** `KERNEL-SSM-MAMBA`, feeding `ROAD-V1-C2-LOCAL-BF16`.
**Lifecycle:** `READY` as an opt-in local discriminator. The generic Mamba row
remains `INVENTORIED`; Qwen3.6-27B/35B release gates are unavailable here.

## Current-head and intake revalidation

This spike is based on immutable campaign head
`96382c7f5b625f5e5cdf66feebd49d76d334efae`, itself based on
`upstream/main` `ba8d867c9ea1aab0896af9c741924ba9855e0400`. The active local
stack is already bound to PR #155; this records the next discriminator without
changing product code, tests, claims or remotes. A current-tree search finds no
`VT_GDN_DECODE_BV` selector or BV16 implementation.

The local graph-node trace
`/tmp/qwen35-conv-block256-arm1-84c7e23b0.sqlite` resolves the executing kernel
to `GdnDecodeFusedKernel<__nv_bfloat16,__nv_bfloat16,float,8>`. Across all
1,704 calls it takes **295.924017 ms** (**173.664 us** mean), with grid-y 800
dominant at 816 calls / **150.043436 ms** / **183.877 us**. Its launch is
grid `(4,N*Hv,1)`, block 256, 52 registers/thread and 17,536 bytes dynamic
shared memory.

The same-tool pinned-vLLM trace `/tmp/qwen35-async-3f35356e0-vllm.sqlite`
resolves `fused_sigmoid_gating_delta_rule_update_kernel`: all 1,873 calls take
**229.354660 ms** (**122.453 us** mean), and extent 800 takes 840 calls /
**107.571496 ms** / **128.061 us**. It launches grid `(1,4,N*Hv)`, block 128,
96 registers/thread and 512 bytes dynamic shared memory. The unequal total call
counts prohibit treating the all-call totals as a throughput ratio; the shape
and matched workload establish a concrete schedule difference to test.

## Whole execution chain

The local direct-CUDA path is `GdnDecodeFusedCuda` through
`LaunchGdnDecodeFused{S,NW}` to `GdnDecodeFusedKernel` at
`src/vt/cuda/cuda_gdn.cu:2699-2925`. It caps `BV` at 32, computes
`NV=ceil(Dv/BV)`, and launches `BV*NW` threads. For production `Dv=Dk=128` and
the accepted `NW=8`, that is BV32, grid-x 4 and block 256. Each value row owns
eight consecutive lanes; those lanes retain the exact Dk slices and
`__shfl_xor` reduction tree.

Pinned vLLM vendors the Flash Linear Attention Triton operation at
`${VLLM_SOURCE}/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:24-277`:
`BV=min(next_power_of_2(V),32)` at `:208`, grid `(NK,NV,N*HV)` at `:241`, and
`num_warps=4` at `:212,275`. The profiler confirms that generated Triton kernel
executed. This operation does not route through CUTLASS, cuBLASLt or DeepGEMM;
the relevant dependency boundary is the vendored FLA Triton body and its
resolved profiler launch.

The existing `VT_GDN_DECODE_NW=4` probe is not this design. It changed each
row's Dk partition and reduction tree; it was both slower locally and
correctness-invalid: 24/128 streams and 678/16,384 aligned token positions
differed, with SHA-256 `93003ca1...8eebe` instead of the accepted
`83fcdc45...453545`. NW remains 8.

## Isolated hypothesis and design

Keep `NW=8` and every per-row lane, slice, shuffle, arithmetic and store order
unchanged. Change only the independent value tile from BV32 to BV16:

- production grid-x doubles 4→8 while block width halves 256→128;
- one block still has exactly eight lanes for each value row;
- each row's `c0/c1`, decay, dot, update, output reduction and state writeback
  remain byte-for-byte ordered as BV32;
- the shared state slice halves from `32*(Dk+1)` to `16*(Dk+1)` floats, giving
  9,280 bytes at Dk=128 rather than 17,536 bytes;
- value rows are independent, so only value-tile scheduling changes.

The counter-hypothesis is that twice as many blocks repeat q/k staging and
block setup, erasing any occupancy/latency benefit. Measurement decides; no
occupancy claim is inferred from block width alone.

## Selector, geometry and rollback

Add exact opt-in `VT_GDN_DECODE_BV=16`. Only the exact string `"16"` selects
BV16. Unset, empty, whitespace, prefixes/suffixes, numeric lookalikes and every
other value select the shipped BV32 default. This is intentionally stricter
than `atoi` parsing and leaves existing deployments unchanged.

Factor a CUDA-free selector/geometry contract and shared callback dispatcher
for portable tests and the production launcher. For requested width `R`, use
`BV=min(Dv,R)`, `NV=ceil(Dv/BV)`, the existing `NW` corner rule, block threads
`BV*NW_eff`, and shared bytes `(2*Dk + BV*(Dk+1))*sizeof(float)`. Zero dimensions
remain the existing no-launch path. Partial `Dv` uses the existing tail guards;
`Dv<32` retains the existing `NW_eff=1`, so no row group crosses a warp.

The implementation may parameterize or pass BV at runtime, but it must not
duplicate or reorder the kernel recurrence. `VT_GDN_DECODE_BV` is removed with
its code/tests if the discriminator fails. It stays opt-in even on a local win
until repeated local evidence and the unavailable 27B/35B gates close.

## RED-first tests and mutations

Before product implementation, add a CPU-runnable selector/geometry test that
fails until the new contract exists. It must prove:

1. exact `"16"` selects BV16, while unset, `""`, `"16x"`, `"016"`, `" 16"`,
   `"32"` and invalid values select BV32;
2. Dv/Dk=128 with NW8 resolves BV32 to grid-x 4, block 256, 17,536 bytes and
   BV16 to grid-x 8, block 128, 9,280 bytes;
3. partial dimensions round up without dropping rows, and `Dv<32` preserves
   the current one-lane-per-row corner contract;
4. the shared dispatcher invokes exactly one BV16 or BV32 callback, so deleting
   the selector arm or mapping `"16"` to the default is RED.

Extend CUDA GDN coverage with BV16-versus-BV32 byte comparisons for both output
and final state. Cover compact and indexed state, BF16 and F32 state, a partial
value dimension and production Dv=Dk=128; include a null indexed row. The matrix
must exercise both selector arms in one process. Mutations to BV selection,
grid rounding, block width, row mapping, tail guards or either state/output
store must make a focused test fail, followed by byte-for-byte restoration.

## Build and correctness gates

Build the portable selector test, `test_ops_gdn`, `test_qwen35_paged_forward`
and the production `vllm-bench` from the CUDA/Triton/CUTLASS build. Run the
portable test first, then the full CUDA GDN suite and Qwen3.5 paged-forward
gate under the operator's GPU lease, `/tmp/gpu` lock and 22/25 GiB user-systemd
scope. Run staged and full preflight from `nix develop .#cuda`.

The production 128-request greedy token file must remain byte-identical to
SHA-256 `83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.
Any mismatch stops performance work immediately.

## Same-binary performance acceptance

Use one rebuilt binary, one GPU lease/lock and the existing identical cached
Qwen3.5-4B BF16 workload: 128 ShareGPT requests, 128 output tokens, concurrency
32, `max_num_batched_tokens=2048`, 1,280 KV blocks and greedy sampling. Keep the
accepted causal-conv and post-conv selectors identical in both arms. Run the
counterbalanced BV32→BV16→BV16→BV32 order and record graph-node traces, raw
summaries, token hashes, build SHA, clocks/contention and peak VRAM.

BV16 is retained only if the grid-y 800 GdnDecode total improves by **at least
1%** against BV32 and the all-GdnDecode total does not regress. It then owes an
enclosing repeated A/B in which total/output throughput do not decrease and
TTFT, TPOT/ITL, E2E and peak VRAM do not increase. Record both values and ratios
for every axis; a noisy or mixed result fails rather than becoming a speed claim.

If either micro bar or any enclosing axis fails, remove selector, product and
test code and preserve the refutation in the append-only benchmark record and
this spec. If all local bars pass, retain the arm opt-in, complete fresh
implementer/reviewer mutation review, and keep release/default status open on
the hardware-unavailable 27B/35B gates.

## Measured result — ACCEPTED OPT-IN

Product commit `92256e6a9` passes the local acceptance gate. Operator runtime
verification is green: portable selector/geometry **3/3 · 49**, CUDA GDN
**68/68 · 4,699**, and Qwen3.5 paged-forward **4/4 · 8**. Every production
token file in the four-run profile and the memory pair has SHA-256
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.

The counterbalanced same-binary series ran BV32a → BV16a → BV16b → BV32b:

| Arm | all GDN decode, 1,704 calls | grid-y 800, 816 calls | total / output tok/s | TTFT | TPOT / ITL | E2E |
|---|---:|---:|---:|---:|---:|---:|
| BV32a | 295.608238 ms | 149.887342 ms | 6763.51 / 747.89 | 1014.64 ms | 34.90 ms | 5446.55 ms |
| BV16a | 265.417816 ms | 133.690094 ms | 6766.91 / 748.27 | 1013.00 ms | 34.89 ms | 5443.79 ms |
| BV16b | 264.152452 ms | 133.030633 ms | 6779.94 / 749.71 | 1011.37 ms | 34.82 ms | 5433.25 ms |
| BV32b | 295.123586 ms | 149.627524 ms | 6763.94 / 747.94 | 1015.48 ms | 34.89 ms | 5446.15 ms |

The counterbalanced means improve all decode calls
**295.365912→264.785134 ms (-10.3535%)** and the matched grid-y 800 shape
**149.757433→133.360364 ms (-10.9491%)**. The intended launch executed:
grid-x 4 / block 256 / 52 registers became grid-x 8 / block 128 / 52
registers. Local BV16 is still about **1.276x slower** than pinned vLLM on the
matched dominant shape (163.43 versus 128.061 us/call), so that gap stays open.

Including the separately sampled memory pair, the three-pair means move in the
accepted direction on every enclosing axis: total throughput
**6783.437→6791.063 tok/s (+0.1124%)**, output throughput
**750.093→750.940 tok/s (+0.1129%)**, TTFT
**1013.237→1011.490 ms (-0.1724%)**, TPOT/ITL
**34.7867→34.7500 ms (-0.1054%)**, and E2E
**5430.777→5424.630 ms (-0.1132%)**. The memory pair is
BV32→BV16 **13058→13054 MiB** peak GPU memory and
**2,273,490→2,170,880 KiB** peak PSS, both non-regressing.

Evidence lives at
`/tmp/qwen35-gdn-bv{32a,16a,16b,32b}-92256e6a9.{nsys-rep,sqlite,tokens.json}`;
trace SHA-256 values in that order are
`198d1bf843443f69fd1491131a13e7aa95cc136c56e96441aa43c48f267958f5`,
`2f5e47f54faafa70ed8494c3afaa2160e31d311d90e7412cbf9777f2e93e3d18`,
`3b0303a97e2335d5ca2f88eacf5c2e5f063d7d899cb44db3dd7d6a170938d117`,
and `3d7b4a214b14976c95e9e4c944540e91163c7f10e820bb63bf7d0cb9fc9d8b11`.
Memory evidence is
`/tmp/qwen35-gdn-bv{32,16}-mem-92256e6a9.{jsonl,log,tokens.json}`.

Disposition is therefore **ACCEPTED OPT-IN**. `VT_GDN_DECODE_BV=16` remains an
explicit rollback-safe experiment; unset retains BV32. No default or release
credit is claimed while the Qwen3.6-27B/35B gates are hardware-unavailable.
