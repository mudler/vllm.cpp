# VT-MATMUL-FP8-BLOCK-CUDA — the block-scaled FP8 GEMM, CUDA arm for `sm_12xa`

Issue: [#1189](https://github.com/mudler/vllm.cpp/issues/1189), milestone **M5**.
Row: `VT-MATMUL-FP8-BLOCK-CUDA`.
Pinned oracle: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`
(`.agents/upstream-sync.md`), asserted as the HEAD of the local checkout before
any `file:line` below was read.

## Scope

Register `vt::MatmulFp8BlockScaled` for `DeviceType::kCUDA` with the
mainloop-scaled CUTLASS kernel upstream dispatches on this architecture. The op,
its signature, its validation and its contract are M2's and do not change; this
row adds one device arm behind them.

```c++
void MatmulFp8BlockScaled(Queue& q, Tensor& out,
                          const Tensor& a_fp8, const Tensor& a_scale,
                          const Tensor& b_fp8, const Tensor& b_scale,
                          int block_n, int block_k);
```

**Out of scope, each owned elsewhere**: merged `gate_up` and QKV (#1189 M6,
concurrently on `row/MODEL-FP8-BLOCK-MERGED`), so nothing here touches
`layers::MlpGateUpMethodBase` or `vt::MergedGemmGroup`; the column-major and
TMA-aligned activation-scale LAYOUTS that `vt::QuantFp8Group` could emit
directly (see `## Owed`); and any speed claim, which needs hardware this row
does not take.

**No GPU lease is taken and no on-hardware run is performed.** The fleet is
contended. What this row establishes is the compile leg and the host-side
dispatch decision; §`## Owed` states plainly what is not established.

## Which implementation actually runs, and why

M2 read the dispatch chain end to end and this row inherits that reading
unchanged. Restated, with the two rows that decide THIS row's kernel body in
bold:

| Step | Where |
|---|---|
| CUDA block-FP8 kernel priority: FlashInfer, DeepGEMM, **CUTLASS**, Marlin, Triton, Humming | `vllm/model_executor/kernels/linear/__init__.py:355-377` |
| DeepGEMM auto-disabled for `qwen3_5_text` on device-capability family 120 | `vllm/utils/deep_gemm.py:27-46` |
| Marlin excluded at `cc >= 89` | `vllm/model_executor/kernels/linear/__init__.py` |
| the selected kernel's apply: `ops.cutlass_scaled_mm(A, B.T, scale_a=As, scale_b=Bs.T)` | `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py::apply_block_scaled_mm` (`:312-326`) |
| `cutlass_scaled_mm` routes every `sm >= 120` device to the sm120 entry | `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:220-226` |
| the sm120 entry hands `cutlass_scaled_mm_blockwise_sm120_fp8` to the shared dispatcher | `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu:13-20` |
| **the blockwise branch, the f32 scale requirement, the `ceil_div` shape checks and the bias refusal** | `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp::dispatch_scaled_mm` (`:15-18,39-55`) |
| **the kernel this row ports, whole** | `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh::cutlass_3x_gemm_fp8_blockwise` |
| **the three tile configs and the M heuristic that selects them** | the same file, `sm120_blockwise_fp8_config_{default,pingpong,swapab}` and `cutlass_gemm_blockwise_sm120_fp8_dispatch` |
| **the caller: mainloop scale pointers, deduced SF layouts, swapped operands** | the same file, `cutlass_gemm_caller_blockwise` |
| CUTLASS 4.5.0's mainloop line | `include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp:712-718` |
| the SF layouts CUTLASS deduces from the problem shape | `include/cutlass/detail/blockwise_scale_layout.hpp::Sm1xxBlockwiseScaleConfig` (`tile_atom_to_shape_SFA`, `tile_atom_to_shape_SFB`) |
| the host-side reroute away from CUTLASS on a misaligned weight | `vllm/_custom_ops.py::cutlass_scaled_mm` (`:790-796`) |
| upstream's own CUTLASS-vs-reference test, its shapes and its criterion | `tests/kernels/quantization/test_block_fp8.py::test_w8a8_block_fp8_cutlass_matmul` (`:156-200`) |

## The constraint that decides correctness

**The scales apply in the mainloop, once per K-block, into an f32 accumulator —
not in an epilogue.** CUTLASS 4.5.0 spells it out at
`sm120_mma_tma_blockwise_scaling.hpp:712-718`:

```text
accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i) * tCrScaleBViewAsC(i);
```

That is the same structure `vt::MatmulFp8BlockScaled`'s CPU arm implements, and
this row therefore does not re-derive any arithmetic. It supplies operands,
scale POINTERS and the deduced scale LAYOUTS, and the collective does the rest.
The two scale pointers are **mainloop** arguments (`ptr_SFA`, `ptr_SFB` in
`cutlass_gemm_caller_blockwise`), never epilogue arguments; the epilogue is a
plain `LinearCombination` with no alpha of its own.

One difference from the CPU arm is real and is not a divergence: CUTLASS
associates the two scale multiplies left to right where the reference forms
their product first (`tests/kernels/quant_utils.py:150-151`). The difference is
at most one f32 ULP per K-block, and upstream's own gate admits it by comparing
exactly these two arms at `rel_diff < 0.001`
(`test_block_fp8.py::test_w8a8_block_fp8_cutlass_matmul`). G2 carries that
criterion verbatim.

## Design

### One TU, three instantiations, mirroring upstream's own dispatch

`src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu` carries the CUTLASS templates and
the registration. It joins `_FP8_CUTLASS_SOURCES` beside
`src/vt/cuda/cuda_matmul_fp8_cutlass.cu`, so it inherits that list's per-source
gencode (`VT_CUTLASS_FP8_ARCHS` = `12.0a,12.1a`), its `-isystem` CUTLASS include
path and its `--expt-relaxed-constexpr` / `--expt-extended-lambda` options. It
is a separate translation unit because the two kernels share no template: the
per-tensor arm builds a dense `CollectiveBuilder` over plain layout tags, and
this one builds the blockwise collective over `cute::tuple<Layout, LayoutSF>`
tags. Putting both in one TU would only serialise their compiles.

The three configs and the heuristic that picks them are upstream's, verbatim:

| Config | Tile | Schedule | Selected when |
|---|---|---|---|
| `swapab` | `128x32x128`, granularity `(128, 1, 128)` | `KernelTmaWarpSpecializedBlockwiseCooperativeSm120` | `M <= 64 \|\| M % 4 != 0` |
| `pingpong` | `64x128x128`, granularity `(1, 128, 128)` | `KernelTmaWarpSpecializedBlockwisePingpongSm120` | otherwise and `M <= 256` |
| `default` | `128x128x128`, granularity `(1, 128, 128)` | `KernelScheduleAuto` | otherwise |

`swap_ab` is not a scheduling nicety and M2's `## Owed` under-described it. With
`ScaleGranularityM = 1` the non-swap activation-scale layout has stride 1 along
M, so a TMA load of it needs a 16-byte-aligned row, which is `M % 4 == 0` for
f32. Upstream's `M % 4 != 0` arm of the heuristic is therefore a correctness
condition, and decode — `M = 1` — takes the swapped path every time. All three
configs land here for that reason.

### The scale layouts CUTLASS deduces, and the one transpose they force

`cutlass_gemm_caller_blockwise` does not read the scale tensors' strides. It
computes `layout_SFA`/`layout_SFB` from the PROBLEM SHAPE through
`Sm120BlockwiseScaleConfig`, so the memory the pointers refer to has to be in
the layout the config deduces. Reading `tile_atom_to_shape_SFA` and
`tile_atom_to_shape_SFB` out of `blockwise_scale_layout.hpp` gives, for both the
swapped and the unswapped configuration:

| Stream | Deduced index of element `(row, k_tile)` | Which memory layout that is |
|---|---|---|
| activation scale | `row + k_tile * M` | **column-major** `[M, k_tiles]` |
| weight scale | `n_block * k_tiles + k_tile` | row-major `[n_tiles, k_tiles]` |

The weight scale is already that: `weight_scale_inv` ships `[n_tiles, k_tiles]`
row-major and M3 keeps the bytes verbatim, which is why upstream can pass
`Bs.T` — a transposed VIEW whose `data_ptr()` is `Bs`'s.

The activation scale is not. `vt::QuantFp8Group` emits row-major
`[M, k_tiles]`, which is the layout M2's contract fixes and M4 allocates, and it
is the layout upstream's own reference arm reads. Upstream solves this one
rung earlier, by asking its quantizer for the other layout —
`CutlassFp8BlockScaledMMKernel` constructs `QuantFP8` with
`column_major_scales=True` (`kernels/linear/scaled_mm/cutlass.py:340-345`), and
its CUTLASS test says so in a comment
(`test_block_fp8.py:190`, "CUTLASS uses column-major format for scales").

**This arm transposes instead**, into per-stream scratch, one `M * k_tiles`
f32 buffer. That is `M * cdiv(K, 128) * 4` bytes and a kernel over the same
element count — 40 floats for a decode step of the target checkpoint's `q_proj`.
Changing the op's contract to carry the column-major layout would change M2's
signature and M4's allocation under a row that owns neither, and it would leave
the CPU arm needing the transpose instead. The layout is recorded under
`## Owed` as the thing that removes this copy, exactly where M1 and M2 already
put it.

### Output dtype

Upstream's blockwise entry accepts bf16 and fp16 only — `assert out_dtype is
torch.bfloat16 or out_dtype is torch.float16` (`_custom_ops.py::cutlass_scaled_mm`).
`vt::MatmulFp8BlockScaled` accepts f32 or bf16, because M2's CPU arm stores
through `.to(output_dtype)` and M4 wires bf16 at every site. This arm therefore
instantiates the collective for **bf16 only** and, for an f32 `out`, casts the
bf16 epilogue value up — byte-for-byte the same thing
`src/vt/cuda/cuda_matmul_fp8_cutlass.cu::MatmulFp8CutlassKernelCuda` does for
its f32 sinks, and recorded here for the same reason: nothing widens the GEMM,
the extra width buys no precision, and a reader must not mistake the f32 sink
for an f32 compute path. One instantiation per config, three in total.

### Ragged edges — supported, with the refusal upstream itself takes

**Ragged block boundaries are SUPPORTED.** `tile_atom_to_shape_SFB` sizes the
scale grid with `ceil_div(N, 128)`, so a short final N-block is expressible and
CUTLASS predicates it. That is not inference: upstream's own CUTLASS test is
`M=32, N=576, K=7168` — `576 = 4*128 + 64` — chosen because DSV3's
`kv_a_proj_with_mqa` has that shape, and it asserts against the same reference
this arm is measured against. G2 ports that case whole.

**The CUTLASS ALIGNMENT floor is refused by name.** `AlignmentA` and
`AlignmentB` are `128 / sizeof_bits<e4m3>` = 16 elements, and both operands
have K as their contiguous or leading extent, so `K % 16 != 0` cannot be
implemented; `AlignmentD` is 8 bf16 elements along N. Upstream draws the same
line one rung higher and reroutes rather than refusing:

```python
cutlass_compatible_b = b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
if current_platform.is_rocm() or not cutlass_compatible_b:
    out = triton_scaled_mm(...)
```

`b` there is `B.T`, so those two are `K % 16` and `N % 16`. There is no Triton
block arm in this tree to reroute to, so this arm **refuses, by name, at the
dispatch boundary**, naming the dimension, its remainder and upstream's own
reroute. A refusal is the honest end of that branch: falling through to a
kernel that mis-slices would be silent, and falling back to the CPU reference
on device pointers is the #960/#844 failure.

`K = 3884` — the other non-round shape in upstream's grid, `3884 % 16 == 12` —
is that refusal, and it is refused on the CUDA arm while it still RUNS on M2's
CPU arm. The two arms of one op therefore have different domains, which is
upstream's situation exactly, and G4 pins both halves.

**`block_n != 128 || block_k != 128` is refused by name.** The scale
granularity is a compile-time template parameter of the collective, upstream
hardcodes 128 in `dispatch_scaled_mm`'s `ceil_div` checks, and every sm120
blockwise config is built at `(1, 128, 128)` or `(128, 1, 128)`. The loader
already refuses a `weight_block_size` other than `[128, 128]` at load
(#1189 M3), so this is the same refusal restated where the kernel can see it,
not a new limit.

### The host-side decision is a pure function, and it is where the gate lands

Everything above that is a DECISION rather than a template — which config a
given M selects, whether a shape is refused and why, and what index the deduced
scale layouts assign — lives in `src/vt/cuda/fp8_block_scaled_dispatch.h`, a
header with no CUDA dependency of any kind. `cuda_matmul_fp8_block_cutlass.cu`
includes it and does nothing else with those questions.

This is the arrangement `src/vt/cuda/fp8_plan_cache.h` and
`src/vt/cuda/graph_safe_scratch.h` already use, and the reason is the same:
this host has no GPU and no `nvcc`, CI has no GPU either, and a decision that
can only be exercised on hardware is a decision nothing in this tree gates. The
predicate, the heuristic and the two layout formulas are the parts that can be
wrong in a way a compile cannot see, so they are the parts that get a red-first
test that runs on every machine.

### The dispatch counter

`vt::cuda::Fp8BlockScaledStats` counts, per config, how many block-scaled GEMMs
this process dispatched, plus the refusals. #1189's gate design records why:
a x1.02 and a x1.10 scale perturbation on the per-tensor fp8 tower were
demonstrably reached and still produced 16/16 identical tokens
(`tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:43-45`), so a token gate
cannot answer "did anything reach this arm". A silent dequant to bf16 is
numerically BETTER than the quantized path and is invisible to every value
comparison in the tree. The counter is not: it advances only from inside this
kernel, after `can_implement` accepted the arguments, and it names WHICH config
ran, so a heuristic that quietly stopped selecting `swapab` for `M = 1` is
visible without a hardware run of its own.

`vllm::dense_fp8_block::BlockGemmCount` at the model layer answers the sibling
question — did the block-wise linear path run at all — and stays as M4 built it.

### What registering this arm changes at the model layer

Nothing in `src/` outside the new TU. `dense_fp8_block::BlockFp8Runnable` asks
`vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, device)`, and
`OpRegistered` means "is there a NATIVE kernel" (`src/vt/op_provider.cpp`), so
registering this arm narrows M4's `Prepare` refusal on a `sm_12xa` build
automatically and leaves it in force on every arch outside
`VT_CUTLASS_FP8_ARCHS`. That is the correct polarity: an arch with no compiled
kernel keeps refusing rather than falling through to the host reference tier
and dereferencing device pointers.

It also means a `sm_12xa` CUDA user stops being refused and starts running a
kernel **that has never executed**. `docs/USAGE.md` and `docs/FEATURES.md` say
so in this change, in those words, and `## Owed` is the record.

### Why it ships registered rather than behind an opt-in flag

The obvious hedge for an unrun kernel is to register it behind an environment
flag, default off. It is rejected, and the reason is that the flag would carry no
information. Neither position has a measurement behind it: OFF is not "the safe
one", it is "the refusal, still", and the refusal is exactly what M5 exists to
remove. A default-off lever would therefore be a switch nobody could ever be told
when to flip, which is the shape `.agents/parity-lever-protocol.md` calls a lever
with no premise, and this tree already carries several of those.

The honest control is a LABEL rather than a flag, and the tree has one:
`cmake/CudaArchFeatures.cmake` already ships `scaledmm-c3x-sm90` and
`cutlass-nvfp4-sm100` as `DERIVED+BUILD-VERIFIED (testing-welcome)` — compiled
and gencode-proven, never run on the board they target. This arm is in exactly
that state and is described in exactly those words in `docs/USAGE.md`,
`docs/FEATURES.md`, the commit body, the pull request body and `## Owed`. What
makes that acceptable rather than reckless is that the first person to run it
gets a written, registered test that says what to compare against and what the
criterion is, instead of a kernel and a shrug.

### `check-cuda-op-arch-gate`: checked, and correctly NOT extended

`scripts/check-cuda-op-arch-gate.py`'s `REQUIRED` set is for ops whose CUDA
registration must not depend on a CUDA FEATURE, and its own docstring excludes
this case: "an op with a genuinely arch-specific kernel (the cutlass GEMMs,
Marlin, FA2) does NOT belong in it, because for those the feature gate is the
correct behaviour and a missing registration is an honest refusal." This kernel
is a CUTLASS GEMM whose collective exists only for the sm120 family. Adding it
would assert the opposite of what this row wants, so the checker is untouched
and G5 pins the reasoning by asserting that the gate still passes with the new
TU present and that the new registration is the only one for its OpId.

## Risks

| Risk | Control |
|---|---|
| the activation scale is passed row-major, so every element after the first K-block reads the wrong scale | G3 pins both deduced layout formulas against a hand-derived table taken from `blockwise_scale_layout.hpp`, not from the implementation; G2 would fail on hardware |
| the transpose is skipped for `k_tiles == 1`, where row-major and column-major coincide, and the bug hides | G3's table includes `k_tiles == 1` AND `k_tiles > 1`, and the kernel has no `k_tiles == 1` special case to skip |
| a ragged N silently mis-slices | ragged N is supported and G2 is upstream's own `N=576` case; G4 asserts no refusal for it |
| a misaligned K takes the kernel anyway and reads past a tile | refused by name; G4 asserts the refusal for `K=3884` and that the message names K and its remainder |
| the M heuristic drifts from upstream, so decode silently stops using `swapab` | G1 pins all three boundaries by value, including `M=1`, `M=64`, `M=65`, `M=66`, `M=68`, `M=256`, `M=257` |
| a block size other than 128x128 reaches a collective built for 128 | refused by name; G4 |
| the kernel is compiled but no arch enables it, so it lands dead | the TU joins `_FP8_CUTLASS_SOURCES`, whose arch cell is `12.0a,12.1a`; the CI `cuda-fat-build` lane builds `121a` and audits per-source gencode |
| the f32 sink is read as an f32 compute path | recorded above and in the TU's header comment; the collective is instantiated for bf16 only |
| a grown workspace is freed while a captured graph still holds its pointer | `RetireGraphScratch`, the same discipline both sibling CUTLASS TUs use |
| the counter advances on a call that then throws, overstating dispatch | incremented after `run` returns `kSuccess`, and G5 asserts the ordering by counting a refused call |
| a reader believes this arm was measured | `## Owed`, the commit body, the pull request body, `docs/USAGE.md` and `docs/FEATURES.md` all say it was not |

## Tests

Two files, and the split is the point.

`tests/vt/test_fp8_block_scaled_dispatch.cpp` — **CPU tier, runs on every
machine including this one**, registered in `tests/CMakeLists.txt` with
`target_include_directories(... ${CMAKE_SOURCE_DIR}/src)` like its
`test_fp8_plan_cache` neighbour.

- **G1** the config heuristic, pinned by value at every boundary of upstream's
  `swap_ab = (M <= 64) || (M % 4 != 0)` then `M <= 256`: `M=1` and `M=2` and
  `M=3` swap because `M <= 64`; `M=65`, `M=66`, `M=67` swap because
  `M % 4 != 0`; `M=68` does not; `M=256` is pingpong; `M=257` is default. The
  expected values are written out, not computed from the function.
- **G2** — **on hardware only, and it is the load-bearing gate.** In
  `tests/vt/test_ops_matmul_fp8_block_cuda.cpp`; see below.
- **G3** the two deduced scale-layout formulas, against a hand-derived table
  read out of `blockwise_scale_layout.hpp::Sm1xxBlockwiseScaleConfig`. For
  `M=3, k_tiles=2` the activation indices are `{0,3,1,4,2,5}` in row-major
  visiting order, and for `n_tiles=3, k_tiles=2` the weight indices are
  `{0,1,2,3,4,5}`. `k_tiles == 1` is included, where the two layouts coincide
  and a transpose bug is invisible.
- **G4** the refusals, each by name and each with the message asserted:
  `K % 16 != 0` (upstream's `K=3884`), `N % 16 != 0`, `block_n != 128`,
  `block_k != 128`, and the three shapes that must NOT be refused —
  `N=576` ragged, `K=7168`, and the target checkpoint's `q_proj`
  `N=12288, K=5120`.
- **G5** the counter's accounting: it advances per config, by one per
  dispatch, on the config the heuristic named; a refusal advances the refusal
  counter and no config counter; and the four counters are read as one
  snapshot so a test cannot mistake a sibling's increment for its own.

The arch-gate reasoning above is checked by RUNNING
`scripts/check-cuda-op-arch-gate.py`, which is in the gate table below. It is
deliberately not restated as a doctest case: the checker already reads the CMake
source lists and every CUDA source, and a C++ test that re-scanned the tree from
a hardcoded path would be a second, weaker copy of it.

`tests/vt/test_ops_matmul_fp8_block_cuda.cpp` — **CUDA tier. It SKIPS on a host
with no device, and a skip is not a pass.** The file is registered so that CI
builds it and a leased box runs it.

- **G2** upstream's `test_w8a8_block_fp8_cutlass_matmul` ported whole:
  `M=32, N=576, K=7168`, `block_size=[128,128]`, `out_dtype=bf16`, scales
  `U(0,1) * 1e-2`, compared against **M2's CPU arm on a CPU queue in the same
  process** with upstream's own criterion, `rel_diff < 0.001`, computed by
  upstream's own formula. Plus a per-element bound and a vacuity guard, because
  a mean-relative criterion cannot see a single wrong element and an all-zero
  output passes any ratio of means.
- **G7** the M sweep: `M` at 1, 7, 8, 32, 83, 200, 512 against the same
  reference, so all three configs are exercised, with the counter asserted to
  show the config `Fp8BlockScaledConfigFor` predicted for each.
- **G8** the f32 sink equals the bf16 result cast up, elementwise.
- **G9** the refusals from §Ragged, on a device, with the message.

## Gates

| Gate | Command |
|---|---|
| focused, host | `ctest -R test_fp8_block_scaled_dispatch --output-on-failure` |
| the M2 arm, unchanged | `ctest -R test_ops_matmul_fp8_block_cpu --output-on-failure` |
| the M1 arm, unchanged | `ctest -R test_ops_quant_fp8_group_cpu` |
| the M4 wiring, unchanged | `ctest -R test_fp8_block_linear --output-on-failure` |
| op provider totality | `ctest -R test_op_provider` |
| the arch gate | `python3 scripts/check-cuda-op-arch-gate.py` |
| record | `scripts/agent-preflight.sh --fail-on-skip` |
| **compile leg** | CI `cuda-fat-build`, which configures `120a;121a` among ten archs with `-DVLLM_CPP_CUTLASS_FETCH=ON` and audits per-source gencode |
| **the on-hardware leg** | NOT RUN. See `## Owed`. |

## Owed

- **No on-hardware run.** This kernel has never executed. G2, G7, G8 and G9 are
  written, registered and RED-by-absence: on this host they report a skip, and a
  skip is not a pass. What is established here is that the TU compiles for
  `sm_120a`/`sm_121a` in CI and that the host-side dispatch decision is
  correct against upstream's own source. What is NOT established is that the
  kernel produces the reference's numbers, or any numbers.
- **No token gate.** `Qwen/Qwen3.8-27B-FP8` has not been run against the pinned
  oracle on this arm, on any device.
- **No speed claim.** None is made anywhere in this change. The row takes no
  lease, so it has no denominator, no clock state and no contention record —
  the three things `.agents/benchmarking.md` requires before a ratio means
  anything.
- **A `sm_12xa` CUDA user is no longer refused.** Registering the arm narrows
  M4's `Prepare` refusal automatically. Until the run above happens, the arm is
  BUILD-VERIFIED ONLY, in the same sense `cmake/CudaArchFeatures.cmake` already
  labels `scaledmm-c3x-sm90` and `cutlass-nvfp4-sm100`. `docs/USAGE.md` and
  `docs/FEATURES.md` carry that label rather than a capability claim.
- **The column-major and TMA-aligned activation-scale layouts**
  (`utils/fp8_utils.py:610-628`), which `vt::QuantFp8Group` could emit and does
  not. M1 shipped row-major only and recorded both as owed; M2 repeated it.
  With the column-major layout this arm's transpose disappears. It stays owed
  because it changes an op's output contract that two landed rows depend on.
- **`process_fp8_weight_block_strategy`**
  (`kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:89-95`) may pad or
  re-lay-out the weight for a particular kernel at
  `process_weights_after_loading`. The CUTLASS blockwise arm needs no transform
  of the `[N,K]` bytes, so none is done; a different kernel would owe one.
- **fp16 output.** Upstream's entry accepts it; `vt::MatmulFp8BlockScaled`'s
  contract does not, and adding it is M2's signature, not this row's.
- **The other blockwise arms.** FlashInfer, DeepGEMM, Marlin, Triton and
  Humming are all in upstream's priority list and none is ported. On the target
  architecture upstream selects none of them either, which is why this row ports
  CUTLASS and nothing else.

## Stop conditions

Stop and report `NEEDS_DECISION` if any of the following holds.

- The pinned oracle's checkout is not at
  `5559679229bc961848b121ccdeaa8fa5d79bec98`. Every anchor above was read at
  that revision, asserted before the first read.
- The op's contract has to change to admit the CUTLASS scale layouts. It does
  not — the transpose above is the price of keeping it — but if a later reading
  says otherwise, that is M2's signature and two landed rows' allocations, not
  a detail of this one.
- CUTLASS 4.5.0 does not carry `KernelTmaWarpSpecializedBlockwise*Sm120` or
  `Sm120BlockwiseScaleConfig`. Both were read at
  `include/cutlass/gemm/dispatch_policy.hpp:947-948` and
  `include/cutlass/detail/blockwise_scale_layout.hpp:286` before this spec was
  written; a tree that fetches a different CUTLASS cannot build this arm and
  must say so rather than widening the fetch pin silently.

Stop and report `NEEDS_CONTEXT` if the work requires a GPU lease. The row is
scoped so that the LANDABLE part does not, and the part that does is `## Owed`
rather than deferred silently.

## Evidence

Taken on `origin/main` at `27d5432f9`, in a linked worktree, Debug, no CUDA
toolkit and no device on the host. **Nothing below is a device measurement, and
the absence is itself recorded rather than glossed.**

**RED.** With both test files present and `src/vt/cuda/fp8_block_scaled_dispatch.h`
deleted — `ls` on it returning `No such file or directory`, which is the proof
available for a file `git` has not yet tracked — the focused build fails,
`compile_rc=1`, with
``tests/vt/test_fp8_block_scaled_dispatch.cpp:42:10: fatal error:
vt/cuda/fp8_block_scaled_dispatch.h: No such file or directory``. Restoring it
from a tar snapshot and rebuilding returns `compile_rc=0`.

The checker half was red first in the same way. `check-cuda-fat-gencode.py` falls
through to `ALL_SMS` for a source it does not know, so with the new test case
added and the checker untouched, `python3 -m unittest
tests.scripts.test_check_cuda_fat_gencode` reported **2 failures**: the new
`test_the_block_scaled_fp8_cutlass_tu_is_audited_as_sm12x`, and the pre-existing
`test_exact_source_and_archive_matrix_passes`, whose message was
``src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu: gencode ['120a', '121a'] !=
expected ['80', ... '121a']`` — the ten-SM fat build audit failing on a
correctly-compiled TU, which is #394's failure in the other direction. With the
checker's two lines added: 8 tests, `OK`.

**GREEN, host tier.** `test_fp8_block_scaled_dispatch` reports 4 cases, 79
assertions, 0 failed, `Status: SUCCESS!`, in 0.00 s under `ctest`. Per block,
each through a `-tc` prefix filter containing no comma, because doctest splits
`-tc` on commas and a name with one yields `0 cases ran` under a `SUCCESS!`
banner:

| Block | Cases | Assertions |
|---|---:|---:|
| G1 the tile-config heuristic | 1 | 22 |
| G3 the deduced scale layouts | 1 | 17 |
| G4 the refusals, and what is NOT refused | 1 | 33 |
| G5 the dispatch counter | 1 | 7 |
| **sum** | **4** | **79** |

The buckets sum to the whole-run count, so no block is silently empty and no
filter selected nothing.

**GREEN, and NOT A RESULT, device tier.** `test_ops_matmul_fp8_block_cuda`
reports 5 cases, 27 assertions, `Status: SUCCESS!` — and **all 27 come from G6
alone**. G2, G7, G8 and G9 each report `assertions: 0` under a `-tc` filter and
each printed `NO CUDA DEVICE: ... #1189 M5's on-hardware leg is OWED, not
passed.` **`assertions: 0` is a skip wearing a pass**, and it is recorded here as
the measurement it is: the load-bearing comparison of this row did not run.

| Block | Cases | Assertions | What that means |
|---|---:|---:|---|
| G6 the grid's own precondition | 1 | 27 | ran, and is host-only by design |
| G2 upstream's CUTLASS case | 1 | **0** | **did not run** |
| G7 the M sweep and the counter | 1 | **0** | **did not run** |
| G8 the f32 sink | 1 | **0** | **did not run** |
| G9 the refusals on a device | 1 | **0** | **did not run** |

**The other declared gates on the same tree**, all `Passed` under `ctest`:
`test_ops_matmul_fp8_block_cpu` (M2 unchanged, 4.79 s),
`test_ops_quant_fp8_group_cpu` (M1 unchanged, 47.61 s),
`test_fp8_block_linear` (M4 unchanged, 1.09 s), `test_op_provider`,
`test_ops_fp8_cpu`. `python3 scripts/check-cuda-op-arch-gate.py` reports
`OK (2 op(s) pinned to an unconditional CUDA TU)` — unchanged, and correctly so:
its `REQUIRED` set is for ops whose CUDA registration must not depend on a
feature, and its own docstring excludes "the cutlass GEMMs", of which this is
one.

### Mutation results

Every mutation printed `compile_rc` **and** evidence that it applied, because a
mutation that fails to build and a mutation that never applied both read as a
passing test. The tree was snapshotted to **tar** and restored from tar, never
with `git checkout -- .`, which reads the index; the restore was verified by
`sha256sum -c` against a digest taken before the first mutation, and every source
was `touch`ed afterwards so `ninja` could not skip the rebuild and leave a
mutated binary to be re-run.

| Mutation | `compile_rc` | Result |
|---|---|---|
| the heuristic's `\|\|` becomes `&&` | 0 | 1 of 4 cases, **9 assertions** fail. This is the one-character bug the two clauses exist to separate: `&&` agrees with `\|\|` on every M below 65 and every M divisible by 4 |
| `M <= 256` tested before the swap | 0 | 9 host assertions AND 1 device-tier assertion — G6's own coverage check notices the grid stopped exercising three configs |
| the activation-scale index loses its stride, `row + k_tile` | **1** | **proves nothing**: `-Werror=unused-parameter` on `m` |
| the same defect with the parameter kept live, `row * m + k_tile` | 0 | **8 assertions** fail. G3's hand-derived table is what catches it |
| the weight-scale index transposed | 0 | 2 assertions fail |
| **`N % 16` becomes `N % 128`** | 0 | 4 host assertions AND 2 device-tier assertions fail. This is the ragged-block claim, measured: the mutation refuses `N = 576`, which is the one shape upstream's own CUTLASS test runs |
| `K % 16` becomes `K % 4` | 0 | 2 assertions fail: `K = 3884` is divisible by 4 and would be accepted into a kernel that cannot tile it |
| the refusal order: alignment asked before block geometry | 0 | 1 assertion fails |
| a refusal counted as a dispatch | 0 | 2 assertions fail. A build whose every call was refused would otherwise look identical to one whose every call ran |
| the `kCount` sentinel guard's `>=` becomes `>` | 0 | 1 assertion fails |
| the K refusal stops naming upstream's reroute | 0 | **NOT CAUGHT** — see below |

Three results are worth keeping.

**One mutation was not caught, and the test was repaired rather than the record
edited.** Deleting the phrase "reroutes it to triton" from the K-alignment
message left the suite green, because the same message's NEXT sentence contains
the bare word "triton" ("there is no triton block arm here") and the assertion
was a bare-word `find`. The assertion now requires the whole clause and
`cutlass_compatible_b` beside it; re-run, the same mutation fails 1 assertion.
That is the assertion count moving from 78 to 79 between the first and final
runs.

**A mutation that fails to build proves nothing, and this row hit it on the
first try.** The obvious spelling of the scale-index bug drops the `m`
parameter, and `-Werror=unused-parameter` rejects it before any test runs. The
mutation was re-expressed to keep the parameter live, which is the only form
that measures anything.

**The ragged-block decision is measured, not asserted.** Two mutations move the
alignment constant in opposite directions and each is caught by a different half
of G4 — the `% 128` one by the shapes that must NOT be refused, the `% 4` one by
the shape that must be. A refusal test that only listed refusals would have
passed the first, which is how a correct kernel gets narrowed until it no longer
runs the checkpoint it was written for.

### The one preflight gate that is red, and why it is not this change

`scripts/agent-preflight.sh --fail-on-skip` on the merge result reports
**1 gate failed: `test_cpu_x86_llamacpp_floor`**, and every other gate `ok` with
no `SKIP` before the verdict line. That is [#618](https://github.com/mudler/vllm.cpp/issues/618),
which the issue index already describes as load-dependent: at high load the
harness exits `NO_QUIET_WINDOW` (4) where the case expects `GIVING_UP` (2).

It is measured as environmental here rather than assumed. The **same gate passed**
earlier in this session, on the same shared checkout at the base SHA, when the
box was quiet. It then failed twice in a row under load, and the harness printed
its own reason the second time: `waiting for quiet: 15s busy=114% builders=0
load=69.45`, at a one-minute load average between 40 and 103. The failing case
lives in `tests/scripts/test_cpu_x86_llamacpp_floor.py`, which this change does
not touch — the only file under `tests/scripts/` it edits is
`test_check_cuda_fat_gencode.py`.

### A shared scratch directory nearly voided this run

The mutation harness was written to the session scratchpad as `mutate.sh`. That
directory is shared across concurrent sessions, and #1189's own M6 row — running
at the same time on `row/MODEL-FP8-BLOCK-MERGED` — wrote its own `mutate.sh` to
the same path while this harness was executing. `bash` reads a script file
incrementally, so a replaced file can make a running job execute another
session's commands against another session's `WT`. The run was killed, the four
files verified byte-for-byte against a `sha256` digest taken beforehand, and the
two repaired mutations re-run from a uniquely named script in a private
subdirectory. Nothing was lost, and the control that caught it is the digest,
not the harness's own "RESTORED" banner.

## Now

`ACTIVE` — M5 of #1189. M1 (`ad5f175e7`), M2 (`770e49486`), M3 (`09597106e`)
and M4 (`281b4bc76`) are `DONE`; M6 is concurrent on
`row/MODEL-FP8-BLOCK-MERGED`.
