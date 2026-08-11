# sm_120 Qwen3.5 causal-conv residual — structured spike

**Rows:** `KERNEL-SSM-MAMBA`, feeding `ROAD-V1-C2-LOCAL-BF16`.
**Issue:** [#206](https://github.com/mudler/vllm.cpp/issues/206) (landed by PR [#155](https://github.com/mudler/vllm.cpp/pull/155)).
**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16,
128 ShareGPT requests, 128 output tokens, concurrency 32,
`max_num_batched_tokens=2048`, 1,280 KV blocks, greedy. **Lifecycle:**
implemented and byte-exact; arm 1 is a locally positive opt-in, arm 2 is
falsified, and default/release-model gates remain pending.

## Measured selection

A fresh graph-node `nsys` profile at `7f66792f8`, after the exact-chunk and
opt-in post-conv-tile changes, reports:

| Family | ours | pinned vLLM | ours / vLLM | local excess |
|---|---:|---:|---:|---:|
| prefill causal conv | **234.255 ms** / 1,728 calls | **145.532 ms** / 1,897 calls | **1.6096x** | **88.723 ms** |
| fused post-conv | 122.511 ms / 1,728 calls | 108.035 ms / 1,923 calls | 1.1340x | 14.476 ms |

The next metric is therefore **total GPU time of
`CausalConv1dFwdRegKernel` on the exact c32 production workload**, with the
dominant launch shape used as its first micro-metric. Shape-grouping the same
tool's kernel rows divides that total further:

| Exact programs | Calls | Total | Mean | Share of local conv |
|---:|---:|---:|---:|---:|
| 279 | 576 | 86.052 ms | 149.395 us | 36.7% |
| 280 | 336 | 50.137 ms | 149.218 us | 21.4% |
| 156-158 | 216 | 17.647 ms | 81.699 us | 7.5% |
| all remaining shapes | 600 | 80.419 ms | mixed | 34.3% |

The 279-280-program waves alone are 912 launches and **136.189 ms (58.1%)**,
so their mean launch time is the selected micro-metric. This avoids guessing
from a whole-run aggregate when a launch-shape-local change is available.

Artifacts:

- local report `/tmp/qwen35-next-7f66792f.nsys-rep`, SQLite export
  `/tmp/qwen35-next-7f66792f.sqlite`;
- pinned-vLLM report `/tmp/qwen35-async-3f35356e0-vllm.nsys-rep`, SQLite
  export `/tmp/qwen35-async-3f35356e0-vllm.sqlite`.

Nsight Compute is not present in the selected Nix CUDA environment or the host,
so no hardware-counter claim is made. Installing a profiler package is outside
this campaign's permission envelope. Register counts and launch geometry below
come from the same `nsys` CUPTI rows, not an inferred occupancy claim.

## Whole-chain difference and hypotheses

Pinned vLLM launches Triton's `_causal_conv1d_fwd_kernel` with
`BLOCK_M=8`, `BLOCK_N=256`, four warps and two stages
(`${VLLM_SOURCE}/vllm/model_executor/layers/mamba/ops/causal_conv1d.py:16-63,78-79,692-742`).
On the dominant wave that resolves to `grid=(279,32,1)`, block 128, and 32
registers/thread. Its 279-program launches average 88.346 us.

The local exact-descriptor port already matches `BLOCK_M=8` and enumerates the
same sequence/chunk programs, but retains a runtime convolution width and one
channel per thread: `kConvRegN=128`, `grid=(64,279,1)`, block 128, and 43
registers/thread (`src/vt/cuda/cuda_gdn.cu:702-854`). Its 279-program launches
average 149.395 us. Qwen3.5-4B has 8,192 convolution channels and width four,
so 64 versus 32 feature blocks is exactly the `BLOCK_N` difference.

Two independent hypotheses remain and must be measured separately:

1. **Width specialization.** vLLM's `KERNEL_WIDTH` is a compile-time constant;
   local `k` is runtime and reserves arrays through `kConvRegMaxW+1=9` while
   guarding every unrolled tap. Dispatching the production `k=4` instantiation
   can remove dead taps/branches and reduce register or instruction cost without
   changing the grid.
2. **Two channels per thread.** A 128-thread block can process two coalesced
   128-channel stripes, giving a 256-channel feature tile and 32 feature blocks
   like upstream. It halves block-level descriptor/control work, but duplicates
   each thread's channel-local weights and window. Register pressure can refute
   this even when the grid looks better.

These are ordered discriminators, not one combined patch: first compare the
width-four specialization against the runtime-width kernel at the unchanged
128-channel tile; only then add the two-channel instantiation and compare it
against the specialized one. Unsupported widths and dimensions retain the
current kernel.

## Port and rollback

Add compile-time width/channel-count instantiations of the existing register
window, preserving for each channel the exact operation sequence: bias; taps
`j=0..3`; current SiLU; store; window shift; raw-input final-state writeback.
The experiment must not change exact descriptor metadata, scheduler/model
routing, tensor strides, or post-conv dispatch.

Expose an explicit same-binary experiment selector with three arms:

- `0`: current runtime-width, one-channel kernel (sealed baseline);
- `1`: width-four specialization, one channel per thread;
- `2`: width-four specialization, two channels per thread.

Unset retains arm 0 until the evidence supports a default change. Values or
shapes outside the supported experiment fall back to arm 0. Document the
selector in `docs/ENVIRONMENT.md`; factor its parse/launch-contract predicates
into the portable GDN prefill header so CPU-tier tests can kill accidental
default or grid changes.

## Tests and acceptance

RED-first coverage must prove:

1. the selector defaults to arm 0, accepts only the named arms, and invalid
   values roll back;
2. the production `C=8192,K=4` launch contract is respectively 64/64/32 feature
   blocks with block 128, while partial tiles round up safely;
3. CUDA arms 1 and 2 are byte-identical to arm 0 for output and final state over
   BF16/F32 I/O, initial/fresh state, unequal exact chunks, `T<K-1`, partial
   channels and packed row strides;
4. deleting the width specialization or restoring a 128-channel arm-2 tile
   makes a named structural/mutation test fail;
5. the full CUDA GDN suite and cached Qwen3.5-4B correctness remain green, and
   production token files are byte-identical across all measured arms.

First profile the three arms as the same binary under one `${GPU_LOCK}`. An arm
is retained only if it improves the 279-280-program mean outside run noise and
does not worsen total causal-conv time. The winning arm then owes an enclosing
same-workload profile: total/output throughput, TTFT, TPOT/ITL, E2E and peak
VRAM may not regress. Any default flip remains blocked on repeated local A/B
and the hardware-unavailable 27B/35B release-model gates; the 4B vehicle cannot
substitute for them.

No GEMM claim is made. A later GEMM selection separately owes the four-axis,
same-tool invocation-parity proof.

## Implementation and measured outcome

`VT_CONV_CHANNEL_TILE` now exposes the three specified arms. The production
launcher and portable tests share one callback dispatcher, so deleting either
specialized branch is mutation-pinned without adding a production trace hook.
The CUDA matrix compares output and final state against arm 0 over BF16/F32,
initial/fresh state, exact unequal chunks, short sequences, partial channels
and padded rows. A first serial-stripe arm 2 failed that matrix because stripe
1 could overwrite the shared state before stripe 2 read its initial history;
the accepted kernel preloads both stripes before processing either.

That preload NARROWS the window; it does not close it, and this spec should
not be read as claiming otherwise. The underlying hazard is older and wider
than arm 2: in the exact-chunk mapping only chunk 0 reads the initial
`conv_state` while only the LAST chunk writes the final one, and those are
different, unordered blocks — so whenever `has_initial_state[s]` is set and
`t_len > kConvExactM`, the read and the write race. Upstream does not have
this shape: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:147-150`
does both the prior-state read and the `tl.store` inside the SAME Triton
program, with a `tl.debug_barrier()` between them. The accepted arm's
correctness therefore rests on the initial-state read happening at
instruction ~0, not on any ordering guarantee. Tracked as its own bug rather
than fixed silently here, since it is present on main independently of this
row: [#305](https://github.com/mudler/vllm.cpp/issues/305).

The first production profile series was VOID: the test targets had rebuilt the
CUDA library but `vllm-bench` had not been relinked, and all three traces proved
they still launched `CausalConv1dFwdRegKernel` at grid 64/registers 43. After
explicitly rebuilding `vllm-bench`, the same-binary graph-node series proved
the intended dispatch and measured:

| Arm | Kernel/grid/registers | Conv total | Dominant 279/280 mean | Whole total | TTFT | TPOT / ITL |
|---|---|---:|---:|---:|---:|---:|
| 0 runtime width | runtime / 64 / 43 | 234.605 ms | 149.546 / 149.480 us | 6759.39 tok/s | 1016.69 ms | 34.91 ms |
| 1 K=4, one channel | K4<1> / 64 / 52 | **219.506 ms** | **140.133 / 139.982 us** | **6767.62 tok/s** | **1013.82 ms** | **34.88 ms** |
| 2 K=4, two channels | K4<2> / 32 / 58 | 228.401 ms | 145.586 / 145.468 us | 6757.19 tok/s | 1017.61 ms | 34.91 ms |

Arm 1 improves the selected kernel total **6.436%** and every observed
enclosing axis (total/output +0.122%, TTFT -0.282%, TPOT/ITL -0.086%, E2E
-0.123%). Contrary to the initial register hypothesis, specialization raises
the CUPTI register count 43→52; the win is therefore dead runtime-width work,
not higher occupancy. Arm 2 halves feature blocks but raises registers to 58,
is **4.052% slower than arm 1**, and is neutral/slightly negative end to end;
the 256-channel hypothesis is falsified on sm_120.

All rebuilt-series token files have SHA-256
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.
Portable tests pass 9/9·88, CUDA GDN 67/67·4631, and Qwen3.5 paged-forward
4/4·8. Arm 1 remains opt-in because its enclosing movement is small and the
repeated local plus hardware-unavailable 27B/35B default gates remain open.

Final reports:

- arm 0 `/tmp/qwen35-conv-rebuilt-arm0-565a26fcc.nsys-rep`, SHA-256
  `39d383dd878fc340a3cfaaee79a4addcb4eccb181439e9b4725f724f4569a6eb`;
- arm 1 `/tmp/qwen35-conv-rebuilt-arm1-565a26fcc.nsys-rep`, SHA-256
  `c8799ac0b4cdf997d383fe8a690b223be882dce3b1ee1a6fff35a62d75f7cf85`;
- arm 2 `/tmp/qwen35-conv-rebuilt-arm2-565a26fcc.nsys-rep`, SHA-256
  `3d38793571539864b23688fd9a85966debbf1e7c48fe8a1a2509438a45ee0452`.
