# BACKEND-VULKAN-EXL3 — the two operations an EXL3 checkpoint runs on the CPU when the queue is Vulkan

Row: `BACKEND-VULKAN-EXL3`
Issues: [#2530](https://github.com/mudler/vllm.cpp/issues/2530) (primary)
Base SHA: `2e200e2c6` (`origin/main` at claim)
Parent rows: [`QUANT-EXL3`](quant-exl3-shared.md), [`BACKEND-VULKAN`](vulkan-full-support.md)
Sibling row: [`BACKEND-ROCM-EXL3`](backend-rocm-exl3.md) — same donor, same gate
philosophy, different shader language
Matrix: [`.agents/backend-matrix.md`](../backend-matrix.md),
[`.agents/quantization-matrix.md`](../quantization-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** at the pin and has no Vulkan backend at all, so two secondary oracles
apply and neither is the mirror source for the other's half. The trellis FORMAT
is mirrored from [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT). The Vulkan BACKEND SHAPE —
dispatch geometry, the two-view storage model, the committed-SPIR-V route — is
llama.cpp `ggml/src/ggml-vulkan/` @ `237ad9b96`, which is what
`src/vt/vulkan/shaders/vt_common.glsl` already records. The op seam and the
reference tier are this tree's.

## Now

`DONE` for the two ops in scope, gated on `llvmpipe` with no GPU lease. Both are
registered on `kVULKAN`, both are BYTE-identical to the CPU arm, and the
synthetic EXL3 linear that S1 measured at **two** reference-tier hits now
completes with **zero**. §Evidence carries the runs. #2530 is `Refs`, not
`Closes`: its first slice asked for the measurement on a real checkpoint through
`vllm-cli`, and no EXL3 checkpoint exists on this box or on the NAS
(§"What could not be verified").

## S1 — the measurement #2530 asked for first, and its answer

**EXL3 runs on a Vulkan queue today, and it runs entirely on the host.** The
issue called that "strongly implied and NOT established". It is now established.

`ReferenceTierEligible(kVULKAN)` is `true` because
`VulkanBackend::DeviceMemoryIsHostAddressable()` returns `true`
unconditionally (`src/vt/vulkan/vulkan_backend.cpp`), so the portable CPU
reference tier (`src/vt/op_provider.cpp` `RegisterReferenceTier`) installs the
CPU EXL3 kernels onto a Vulkan queue. A synthetic `k = n = 2048`, 3-bit,
codebook-0 linear driven through `vt::Exl3Gemm` on a `kVULKAN` queue produced
exactly the two notices #2433 reported for ROCm, on the same two ops:

```text
[vt op-provider] op=64  device=3 selected=vt-cpu-ref priority=-1000 registered=1
[vt reference-tier] op=CastF16 device=vulkan has NO native kernel; running the
  PORTABLE CPU host kernel ... It is SLOW: this run is not a performance measurement
[vt op-provider] op=132 device=3 selected=vt-cpu-ref priority=-1000 registered=1
[vt reference-tier] op=Exl3Gemm device=vulkan has NO native kernel; ...
reference-tier hits: before=0 after=2 (delta=2)
```

and the Vulkan result was byte-identical to the CPU result — which is not a
parity finding but the mechanism itself, because it WAS the CPU kernel, called
on Vulkan-allocated memory.

So the gap on Vulkan is the same two ops as on ROCm, for the same reason, and
`kExl3HadR128` is likewise unregistered but is not on the dense forward path.

**No timing is quoted from S1 as a ratio.** The dev box was carrying other
agents' builds throughout, and three repeats of the identical binary read
15.9 / 27.4 / 78.8 ms on the CPU leg and 21.5 / 72.1 / 48.2 ms on the Vulkan
leg — legs that run *the same host function* and therefore cannot genuinely
differ by 3x. That spread is contention, `.agents/benchmarking.md` refuses a
number taken under it, and the structural result above does not need one. The
BEFORE/AFTER count of reference-tier hits is the gate, and a count is immune to
box load.

## Scope

Exactly two ops, and the third slice's ambition is deliberately absent.

- **V1 `kCastF16` on Vulkan.** Registered today on CPU (`cpu_ops.cpp` `kCastF16`
  registrar) and CUDA (`cuda_glue.cu` `CastF16KernelCuda`) and, since
  `BACKEND-ROCM-EXL3`, ROCm — while its two siblings `kCastBf16` / `kCastF32`
  have more. `.agents/specs/quant-exl3-shared.md` `## Owed` already records this
  gap in those words; this row discharges the Vulkan quarter of it.
- **V2 `kExl3Gemm` on Vulkan**, transcribed from the portable CPU reference and
  **not** ported from `src/vt/cuda/cuda_exl3.cu`.

Out of scope, each for a stated reason:

- `kExl3MoeMlp`. The CUDA implementation is a persistent cooperative scheduler
  meeting at GRID-WIDE barriers (`cuda_exl3.cu:1659-1662`) and Vulkan has no
  grid sync at any version, so it is a multi-dispatch rewrite rather than a
  transcription. It also serves one checkpoint family, since the tree-wide fused
  contract accepts codebook 1 only (`src/vt/ops.cpp` `Exl3MoeMlp`). Owed, not
  done.
- `kExl3HadR128` as a registered Vulkan op. The shader that performs it EXISTS
  and ships in this row — `vt_exl3_had.comp` is steps 1 and 3 of the GEMM — but
  the op is not on a dense forward path (`kExl3Gemm` does its own Hadamards),
  so registering it would add a reachable-from-nothing entry. The wiring is one
  `RegisterOp` line when a caller appears.
- A cooperative-matrix arm, and the fused decode-inside-tensor-core kernel.
  §"Why the CUDA kernel is not the donor" is the argument; #2530's slices 4 and
  5 own them.

## Why the CUDA kernel is not the donor

The same four structural blockers the ROCm row recorded, plus one Vulkan-only
one, and the first alone settles it.

- `cuda_exl3.cu:103` — `constexpr int kSmemMax = 90 * 1024;`, installed through
  `cudaFuncAttributeMaxDynamicSharedMemorySize`. Vulkan GUARANTEES only 16 KiB
  of `maxComputeSharedMemorySize` and typical devices report 32-48 KiB. The tile
  arithmetic is sized against 90 KiB, so no launch bound recovers it.
- `mma.sync.aligned.m16n8k16` (`:168`, `:1248`). The one cooperative-matrix
  configuration this tree probes is bf16→f32 16x16x16 at subgroup scope
  (`vulkan_context.cpp`); EXL3 needs f16→f32 and f16→f16, neither of which is
  probed, and `m16n8k16` is not a `VkCooperativeMatrixPropertiesKHR` shape.
- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` (`:257`). No SPIR-V equivalent.
- `cp.async` + `commit_group`/`wait_group` (`:241`, `:247`, `:251`), which IS
  the `SH_STAGES` pipeline. Vulkan has no async global→shared copy.
- **Vulkan-only:** a grid-wide barrier. `cudaLaunchCooperativeKernel` has no
  Vulkan form, so the split-K reduction cannot be closed inside one dispatch.

The donor is therefore `src/vt/cpu/cpu_exl3_kernels.cpp` +
`src/vt/cpu/cpu_exl3_dequant.cpp`, which are pure portable C++ with no
intrinsics — exactly what `src/vt/rocm/rocm_exl3.hip` did, and the reason that
row's gate is stronger than the CUDA arm's.

## Design

Three dispatches, step for step as `Exl3GemmKernelCpu` runs it:

```text
1. a_had = had_r_128(A, pre_scale = suh)        vt_exl3_had, HALF_IN/HALF_OUT
2. raw   = a_had @ reconstruct(trellis)          vt_exl3_gemm, f32 accumulate
3. C     = had_r_128(raw, post_scale = svh)      vt_exl3_had, float in
```

**`vt_exl3_had.comp`** — one 32-lane group per 128-block, four groups per
128-invocation workgroup, `h[4][32]` per group in shared memory (2 KiB total
against a 16 KiB guaranteed floor). Lane `t` holds columns `4t..4t+3`, so lane
bit `i` carries Hadamard level `4*i` and levels 4..64 are exactly
`ShuffleHadWarp`'s five xor-partner steps. This is a transcription of
`cpu_exl3_kernels.cpp` `HadBlock128` / `ShuffleHadWarp` / `HadRowBlock`, not a
rewritten butterfly, and that is what makes the byte claim available. The
(half-in, half-out) pair rides SPECIALIZATION CONSTANTS, which is the backend's
established variant axis (`vt_cast.comp` § the dtype pair).

In-place is safe by construction and not by luck: every lane loads its four
inputs before the first `barrier()` and stores after the last one, so no element
is written while any lane may still read it. `vt::Exl3Gemm`'s contract permits
`a_had` to alias `a`, and upstream says the same of its own kernel
("Works inplace if y == x", `hadamard.cu:86`).

**`vt_exl3_gemm.comp`** — one workgroup per (16-column output tile, 8-row
block); one invocation owns one output element. Per k-tile the 128 invocations
decode the tile's 256 codewords into shared memory two each, which is
`Exl3DecodeTile` with its `t` loop unrolled across invocations.

**THE WORKGROUP IS 128 AND NOT THE ROCm ARM's 256.** 128 is the Vulkan-GUARANTEED
`maxComputeWorkGroupInvocations`, and `vt_common.glsl` § VT_TG makes that the
backend's portability floor. 256 would need a device probe and a second module
for the fallback, for a row whose whole claim is correctness. The consequence is
8 rows per workgroup instead of 16 and one extra decode round per invocation;
neither moves a bit.

**THE ACCUMULATION ORDER IS THE HOST'S** — `ti` ascending outermost, `rr`
ascending inside, into one f32 accumulator — which is the host loop nest
(`cpu_exl3_kernels.cpp` `Exl3GemmKernelCpu` step 2) read down a single
accumulator. The host's `if (xv == 0.0f) continue` is KEPT and is not an
optimisation: with `acc == -0.0f`, adding a `+0.0f` product flips the sign of
the zero, so dropping the guard makes this arm differ from its own reference on
exactly the values a byte gate exists to catch.

**fp16 is a SOFTWARE CODEC here, and it is the same one.** This tree's shaders
require only 16-bit STORAGE (`vt_common.glsl:44-45`); there is no fp16
arithmetic extension. `vt_f16_to_f32` / `vt_f32_to_f16` (`vt_common.glsl:135-179`)
are integer transcriptions of `src/vt/dtype.cpp:176-220`, so `RoundHalf`, the
codebook 0/1 fp16 pair-sum and the codebook 2 affine are computed in f32 and
rounded exactly where the host rounds. That is what the host does anyway —
`Exl3DecodeCodeword` widens to `float` and calls `RoundHalf` — so nothing is
approximated to make Vulkan work.

**The f32 `raw` staging buffer** is named as a cost rather than hidden. The CPU
arm holds it in a `std::vector`; a device needs the same `[m, n]` f32 between
steps 2 and 3. It is grow-only and process-wide, allocated through
`VulkanContext::AllocBuffer` and entered in the allocation registry so `Resolve`
finds it, and the batch is flushed before a grow frees the previous buffer
because an open command batch may still reference it. A fused kernel would not
need it; a fused kernel is the later speed row, and this one buys the end of the
reference tier.

Hazard ordering between the three dispatches needs no hand-written barrier: the
committed SPIR-V carries a per-binding `writable_mask` reflected out of the
module (`vulkan_spirv.h`), and `VulkanContext::Dispatch` uses it to decide
whether two dispatches sharing a buffer are independent.

## The shader toolchain premise was stale, and it is stale in BOTH directions

`scripts/gen-vulkan-spirv.py:12-16` records, measured 2026-07-22, that "neither
of our boxes has one — `dgx.casa` and the dev box both ship the Vulkan LOADER
but no `glslc`/`glslangValidator`/`libshaderc`, and neither grants sudo to
install one".

Half of that is now false and the other half is now true for a different reason,
so the comment is corrected in this row rather than left to mislead the next
reader:

- `/usr/bin/glslc` DOES exist on the dev box today — shaderc 2023.8,
  spirv-tools 2023.6, **glslang 14.0.0**.
- It CANNOT regenerate this tree's SPIR-V. `vt_matmul_coopmat.comp:39` requires
  `GL_EXT_bfloat16`, which glslang 14.0.0 does not know:
  `error: '#extension' : extension not supported: GL_EXT_bfloat16`. So
  `gen-vulkan-spirv.py --check` with the system compiler exits on that shader
  before reaching any other.
- The working route needs NO sudo and CI already uses it: the pinned
  `glslang 16.5.0` **release tarball** unpacked into a temp directory
  (`.github/workflows/ci.yml` `vulkan-spirv-freshness`). Run with that on
  `PATH`, `--check` on the unmodified tree prints
  `committed SPIR-V is up to date` — the committed blob is byte-reproducible,
  which is what makes it safe to regenerate here at all.

The regeneration in this row was produced by that exact pinned compiler, so
every pre-existing module's bytes are unchanged and the diff to
`vulkan_spirv.cpp` is the two new modules and nothing else.

## Tests and gates

`tests/vt/test_exl3_vulkan.cpp`, run as
`ctest --test-dir build -R '^test_exl3_vulkan$' --output-on-failure` against a
build configured `-DVLLM_CPP_VULKAN=ON`.

The bound is **ZERO**, on both ops, for the reason the ROCm sibling states: this
is a transcription of the portable reference using no matrix cores and no
split-K, so all three steps run the same IEEE f32 operations on the same values
in the same order. The comparison is INTEGER over the stored output words. A
tolerance here would be slack this arm has not earned and does not need, and it
is what makes the GEMM gate a DECODE gate too — one mis-decoded codeword moves
an f32 accumulation and the stored bits with it.

The claims, and the case that fails if each is broken:

1. `kCastF16` is registered on `kVULKAN`, beside its two siblings.
2. `kCastF16` is byte-identical to the CPU arm from f32 and from bf16, over
   inputs chosen to include an exact fp16 tie of each parity, a value below the
   subnormal floor, a subnormal, an overflow and a signed zero.
3. `kExl3Gemm` is registered on `kVULKAN`.
4. `kExl3Gemm` is byte-identical to the CPU arm over every `(bits, codebook)`
   pair the tree instantiates and the row counts that select each grid arm.
5. The `codebook` argument REACHES the device decode — one trellis, three
   codebooks, three results that must differ from each other on the device
   exactly as they differ on the host. Without this, a kernel that hardcoded one
   codebook would pass claim 4 on every row whose codebook happened to be that
   one.
6. The `bits` argument reaches it, by the same argument on widths.
7. The input transform runs IN PLACE when `a_had` aliases `A` — a claim about
   barriers that no CPU call can exercise.
8. Zero reference-tier hits across a full `Exl3Gemm` on a Vulkan queue,
   measured the way S1 measured two. This is the reachability claim: it is the
   number that moves when the two `RegisterOp` lines are deleted.

Every device case SKIPS when no Vulkan backend is registered, SAYS so, and STILL
ASSERTS the precondition it skipped on. `assertions: 0` printed under `SUCCESS!`
is a skip wearing a pass and this family has paid for that once already.

## Risks

1. **A byte gate against a moving oracle.** The CPU arm is both donor and
   oracle, so a defect transcribed faithfully from a defective host kernel
   passes. Mitigated only in part: `tests/vt/test_exl3_gemm.cpp` gates the host
   arm against an INDEPENDENT f64 reference built from definitions
   (`exl3_fixture.h` `SylvesterH`, `HadRefBlock`), so the host arm is not
   self-gated. This row does not re-derive that.
2. **`llvmpipe` is a software rasteriser.** It executes the same SPIR-V a real
   driver does, so a numeric defect is caught; a driver-specific miscompile or a
   real subgroup width is not. This is the same exposure every Vulkan claim in
   this tree carries and it is not new here. NOT MITIGATED, and stated rather
   than discovered.
3. **The regeneration touches a 1.5 MB generated file.** Mitigated by the
   byte-reproducibility check above: `--check` was green on the unmodified tree
   under the pinned compiler BEFORE any shader was added, so a changed byte in
   an unrelated module would be visible in the diff rather than lost in it.

## Owed

- `kExl3MoeMlp` on Vulkan — see §Scope for why it is a rewrite and not a
  transcription. **Owned by
  [#2765](https://github.com/mudler/vllm.cpp/issues/2765)**, which carries the
  numbers: the tiles need 28672 bytes at bits 3 and 37888 at bits 6 against
  Vulkan's 16384-byte `maxComputeSharedMemorySize` guarantee, and the launcher
  asks for 92160. The shared memory is the SECOND blocker; the first is that the
  group barrier needs a grid sync Vulkan has at no version, which no tile choice
  can supply. #2530 names it under "Not worth doing".
- `kExl3HadR128` registered on Vulkan. The shader exists in this row and the
  wiring is one line; it waits for a caller so nothing lands unreached.
- The real-checkpoint arm of #2530's slice 1 — see below.
- A refusal for a PACKED STRIDED input to the Vulkan cast family. `vt::CastF16`
  and `vt::CastBf16` both tolerate an input whose rows are dense while the row
  stride spans a parent tensor (the merged-QKV view), and `CastKernel` indexes
  FLAT from one byte offset, so it would read such an input as contiguous. This
  is PRE-EXISTING for `kCastBf16` and `kCastF32`, which have been registered on
  this backend since the W0 skeleton, and V1 inherits it rather than introducing
  it. It is not widened here and it is not narrowed here: adding a refusal to the
  one shared kernel would change all three ops' behaviour, which is a different
  row's decision. `src/vt/ops.cpp` records that the merge is CUDA-only, so no
  caller reaches it today.
- A performance number. This row offers none and none is owed by it: the
  reference tier it removes made "slow" the wrong axis, and
  `.agents/benchmarking.md`'s conditions were not met on this box.

## What could not be verified

- **No EXL3 checkpoint exists on this box or on the NAS.**
  `/mnt/nas_share/checkpoints` and `/mnt/nas_share/models` carry no `*exl3*`
  artifact, so the `vllm-cli` measurement #2530's slice 1 describes — the shape
  #2433 ran on `strix:gpu0` — could not be run at all. What IS established is
  the mechanism, on a synthetic linear through the same public seam a checkpoint
  reaches: the tier installs, both ops fall back, and after this row neither
  does. A checkpoint would add an end-to-end token and a throughput ratio; it
  would not change which ops are registered.
- **No real GPU.** Every number and every byte comparison here is `llvmpipe` on
  the dev box. `dgx:gpu0` and `thor:gpu0` carry NVIDIA ICDs and were not leased,
  because the correctness claim does not need one and a throughput claim was
  not offered.

## Evidence

Dev box, `llvmpipe` (`mesa-vulkan-drivers`, `/usr/share/vulkan/icd.d/lvp_icd.json`),
no GPU and no `rc` lease. Build: `cmake -S . -B build -G Ninja
-DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_VULKAN=ON -DVLLM_CPP_CUDA=OFF`,
GCC 13.3.0, `-Werror` clean.

**The focused gate.**

```text
$ ctest --test-dir build -R '^test_exl3_vulkan$' --output-on-failure
1/1 Test #504: test_exl3_vulkan ................. Passed 0.47 sec
100% tests passed, 0 tests failed out of 1

[doctest] test cases:  8 |  8 passed | 0 failed | 0 skipped
[doctest] assertions: 47 | 47 passed | 0 failed |
```

BYTE-EXACT ON THE FIRST DEVICE RUN, `first differing byte = -1` on every arm:

| bits, cb | m | C | what it is | first differing byte |
|---|---|---|---|---|
| 3, 0 | 1 | f16 | a stock exl3 body at the shape a decode step has | -1 |
| 3, 0 | 20 | f16 | three row-blocks, the last one PARTIAL | -1 |
| 6, 0 | 3 | f16 | a stock exl3 lm_head | -1 |
| 3, 1 | 3 | f16 | the SparkInfer DeepSeek-V4 marker | -1 |
| 3, 1 | 3 | f32 | the same, into upstream's `c_fp32` arm | -1 |
| 4, 2 | 3 | f16 | the Qwen3.8-27B mul1 body, 270 of its 272 tensors | -1 |
| 5, 2 | 3 | f16 | its 5-bit tensor, and all 36 of the draft | -1 |
| 6, 2 | 3 | f16 | its mul1 lm_head | -1 |

**BEFORE and AFTER, the same probe and the same shape** (`k = n = 2048`, 3-bit,
codebook 0, `m = 1`), under `VT_OP_PROVIDER_STATS=1`:

| | provider for `CastF16` / `Exl3Gemm` on `device=vulkan` | reference-tier delta |
|---|---|---|
| before | `vt-cpu-ref priority=-1000`, twice | **2** |
| after | `vt-native priority=0`, twice | **0** |

and the AFTER run is byte-identical to the CPU arm at that 2048x2048 shape too
(`first differing f16 word: -1`), which is 64x the elements the suite's own cases
cover.

**The wider suites that this change can move**, all green:
`test_exl3_vulkan`, `test_vulkan_backend` (35 cases / 2140 assertions),
`test_backend_cross_device`, `test_exl3_gemm`, `test_exl3_gemv`, `test_exl3_moe`,
`test_exl3_dequant`, `test_exl3_rocm`, `test_cast_f16` — 9/9.

**SPIR-V regeneration.** `gen-vulkan-spirv.py --check` printed
`committed SPIR-V is up to date` on the UNMODIFIED tree under the pinned glslang
16.5.0 BEFORE anything was added, so the regeneration's diff is +948 / -0 and
every pre-existing module is byte-identical. `--check` is green again after.

### Mutations

Each restored byte-for-byte (sha256 verified), each rebuilt with the test
binary's mtime asserted to have MOVED, and a build failure treated as NOT a
result. Baseline is 8 cases / 47 assertions, all passing.

| # | mutation | result |
|---|---|---|
| M1 | `kCastF16` `RegisterOp` deleted | **RED** 7/1 cases, 44 assertions, 1 failed |
| M2 | `kExl3Gemm` `RegisterOp` deleted | **RED** 7/1 cases, 14 assertions, 1 failed |
| M3 | codebook-0 multiplier `89226354u` -> `...5u` | **RED** 4/4 cases, 8 failed |
| M4 | tail-biting wrap `i1 % words32` -> `i1` | **RED** 3/5 cases, 16 failed |
| M5 | Hadamard level-2 butterfly `s0 - s1` -> `s1 - s0` | **RED** 3/5 cases, 16 failed |
| M6 | `precise` removed from the GEMM accumulator | **GREEN — see below** |
| M7 | the host's `if (xv == 0.0) continue` removed | **GREEN — see below** |
| M8 | the step-1 PRODUCTION CALL SITE deleted | **RED** 3/5 cases, 16 failed |

M1 and M2 red through the REGISTRATION case only; with the op unregistered the
byte cases skip honestly, which is correct — the op cannot run at all — and is
why those cases exist separately.

**M6 and M7 are honest negatives and are reported as such rather than dropped.**

- **M6.** Removing `precise` does not move a byte on `llvmpipe`, because
  `llvmpipe` does not contract the multiply-add. The qualifier is therefore
  DEFENSIVE against a driver that does, and this gate CANNOT demonstrate its
  necessity. It stays, because the host build sets `-ffp-contract=off` for the
  identical reason and a driver that contracts would break the byte claim on
  hardware this row cannot reach. Nobody should read the byte gate as proof that
  the arm is contraction-safe on a real GPU.
- **M7.** Removing the zero guard does not move a byte either, and the reason is
  analytic rather than lucky: the guard only matters when `acc` is already `-0.0`
  and a `+0.0` product is added, `acc` starts at `+0.0`, and f32 addition yields
  `-0.0` only from `(-0.0) + (-0.0)`. The fixture's `a_had` contains no exact
  zeros, so the discriminating value never occurs. The guard is kept because it is
  what the host does, not because this suite proves it.

## Outcome

**What was measured.** #2530's first slice, which asked whether EXL3 runs on a
Vulkan queue at all. It does, entirely on the host, at exactly two reference-tier
ops — the same two #2433 found on ROCm. After this row, zero.

**What was rejected, and why.**

- *Porting `cuda_exl3.cu`.* Rejected on geometry before instruction support:
  90 KiB of shared memory against a 16 KiB Vulkan guarantee. `mma.sync`,
  `ldmatrix`, `cp.async` and the grid-wide barrier are three further independent
  blockers and none of them had to be reached.
- *Decoding to an f32 weight buffer and reusing the existing `kMatmulBT`
  pipeline*, which is what #2530 proposed for this slice. Rejected for two
  reasons. It materialises the full `[k, n]` f32 weight — 32 MiB for one real
  projection — which is the exact thing the CPU reference refuses to do and the
  reason EXL3 exists. And `vt_matmul`'s accumulation order is its own, so the
  gate would have had to be a tolerance. Decoding inside the GEMM keeps the
  host's loop nest and buys byte equality instead, at no extra complexity: the
  shader is shorter than the two it would have replaced.
- *A 256-invocation workgroup*, which the ROCm arm uses. Rejected because 128 is
  the Vulkan-GUARANTEED `maxComputeWorkGroupInvocations`; 256 would need a device
  probe and a second module for the fallback, to buy speed this row does not
  claim.
- *Registering `kExl3HadR128`.* The shader exists and the registration is one
  line, but no dense forward path calls the op. Left owed, and the suite asserts
  the absence so it reads as a decision.

**Why each default has its value.**

- `kExl3InvSqrt128 = 0.088388347648f` is upstream's literal (`hadamard.cu:107`),
  never a recomputed `1/sqrt(128)`, because the two can differ in the last f32
  bit and that bit is the byte gate.
- The workgroup count for the Hadamard is capped at 65535, the Vulkan-guaranteed
  `maxComputeWorkGroupCount[0]`. It is a performance choice only, because the
  shader carries a grid-stride loop.
- The `raw` scratch is grow-only and process-wide, and the batch is FLUSHED
  before a grow frees the previous buffer: dispatches are batched, so the old
  buffer may still be bound by work that has not executed.
- Each GEMM operand is bound through the ONE view it uses. Declaring an unused
  second view would let glslang strip it, and the generator reports that as a
  binding HOLE — a loud failure, but an avoidable one.
