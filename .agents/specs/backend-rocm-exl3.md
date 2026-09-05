# BACKEND-ROCM-EXL3 — the two operations an EXL3 checkpoint still runs on the CPU

Row: `BACKEND-ROCM-EXL3`
Issues: [#2433](https://github.com/mudler/vllm.cpp/issues/2433) (primary)
Base SHA: `c5df2b1ed` (`row/QUANT-EXL3-MUL1`, unmerged at the time of writing)
Parent rows: [`QUANT-EXL3`](quant-exl3-shared.md), [`BACKEND-ROCM`](rocm-backend-w0.md)
Matrix: [`.agents/backend-matrix.md`](../backend-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** at the pin, so the format is mirrored from the registered secondary
oracle [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT). The op seam, the reference
tier and the backend structure are vLLM's and this tree's; exllamav3 supplies
the trellis format only.

**Why the base is not `origin/main`.** Codebook 2 (`mul1`) landed on
`row/QUANT-EXL3-MUL1` and had not merged when this row started. The device arm
here instantiates all three codebooks, so it cannot be written against a tree
that only knows two. When that branch lands, merge `origin/main` and re-verify.

## Now

`DONE` for the two ops in scope, on the one AMD device this fleet has. Both are
registered, both are byte-identical to the CPU arm on `gfx1151`, and the
checkpoint #2433 measured now completes with **zero** reference-tier hits at
**10.0x** the warm throughput of the same tree with the two registrations
disabled. §Evidence carries the run. #2433 is NOT closed: its fourth acceptance
line wants the throughput recorded against the BF16 control with AMD clock
attribution, and the BF16 control HUNG THE GPU in the same lease (§Evidence,
"what did not work"). Three of its four lines are met.

## The gap, as measured

The ROCm model sweep ran `llama32-1b-exl3-3bpw` on `strix:gpu0` (`gfx1151`,
ROCm 7.2.4) through the public `vllm-cli` path (#2433, rc job
`43267dc3-52c5-43b2-b6b7-538aff68e6b6`). One greedy token completed, and
provider statistics reported exactly two CPU reference-tier operations:

```text
[vt reference-tier] op=CastF16 device=rocm:0 ...
[vt reference-tier] op=Exl3Gemm device=rocm:0 ...
```

1.436 s for one output token. The BF16 control `llama32-1b-instruct-bf16`
completed on the same binary and device with **zero** reference-tier hits, which
isolates the gap to the EXL3 path and not to the Llama loader, paged attention,
the sampler or the dense model.

**On a DISCRETE Radeon this is not slowness, it is a refusal.** The reference
tier installs only where `Backend::UnifiedMemory()` is true
([`docs/ROCM.md`](../../docs/ROCM.md) "Understand fallback behavior"), so on a
discrete board `GetOp` throws and an EXL3 checkpoint does not run at all. R2
below is what fixes that. **We cannot test that fix**: `strix:gpu0` is the only
AMD device on the fleet and it is an APU.

## Scope

Exactly two ops.

- **R1 `kCastF16` on ROCm.** Registered today on CPU
  (`src/vt/cpu/cpu_ops.cpp` `kCastF16` registrar) and CUDA
  (`src/vt/cuda/cuda_glue.cu` `CastF16KernelCuda`) while its two siblings
  `kCastBf16`/`kCastF32` have six backends each.
  `.agents/specs/quant-exl3-shared.md` `## Verified on the MERGED tree (2026-09-02) — including the one thing the landing did not check

The row landed on the strength of a device run against the pre-merge tree. Every
ROCm source was byte-identical to what landed, but **the merged tree had never
been compiled under HIP**, and nothing on the dev box compiles `.hip`. That
mattered for one specific coupling: `main`'s new Q6_K `Fmt == 3` arm in
`rocm_grouped_gemm.hip` consumes `DF16ToF32` from the header this row moved it
into. Job `d218dc2d` (`run4.sh`) on `strix:gpu0` closes it.

At `9dcc34cc3`, gfx1151, ROCm 7.2.4, worker `rc-worker-lcjhd`, 784 s:

```
cmake rc=0    ARCH OK: ROCm backend: ENABLED for arch(es) [gfx1151]
build1 rc=0   (the merged tree DOES compile under HIP)
ctest rc=0    (the focused suite still passes on the device)
```

**And the BF16 control ran this time.** It hung the GPU on the previous job
(`HW Exception ... reason :GPU Hang`), which is why #2433's fourth acceptance line
went unmet. Here `bf16_ctrl8 rc=0`, `reference_tier_notices=0`.

| leg, 8 tokens x 3 repeats | wall | ref-tier notices | tok/s |
|---|---|---|---|
| EXL3 BEFORE (both registrations `if (false)`) | 30,313 ms | **2** | 0.847, 0.827, 0.815 |
| EXL3 AFTER (native arm) | 4,306 ms | **0** | 6.995, 8.734, 8.504 |
| BF16 control | 3,240 ms | 0 | — |

**~10.3x warm on the merged tree**, and the two arms produce byte-identical greedy
text -- ` Paris. Paris is known for its famous` from both. The BEFORE leg is the
reachability mutation: disabling the two `RegisterOp` lines restores both the
fallback notices and the 0.83 tok/s.

The bf16 denominator also reframes the result. Before this row EXL3 was ~9.4x
slower than bf16 on this device; it is now within **1.33x** of it, which is the
expected shape for a 3bpw trellis decode rather than a fallback.

**#2433 is still `Refs`, not `Closes`, and the remaining line is narrow.** Its
fourth criterion asks for warm throughput WITH AMD clock attribution. The
throughput is here and the bf16 denominator is here; no `sclk` capture is, and
`.agents/benchmarking.md` is explicit that a number is quotable only with the
clock it was taken at. That capture is what the issue still owes.

Unchanged: every number is a gfx1151 APU figure with unified memory and
generalises to no discrete Radeon -- the case that is actually broken today, and
that this fleet cannot test.

## Owed` already records this gap in
  those words; this row discharges the ROCm quarter of it.
- **R2 `kExl3Gemm` on ROCm**, transcribed from the portable CPU reference and
  **not** ported from `src/vt/cuda/cuda_exl3.cu`.

Out of scope, each for a stated reason:

- `kExl3MoeMlp`. No AMD board on this fleet can hold the artifact that reaches
  it (~99.5 GiB; strix has 67 GB of system RAM), and the tree-wide fused-MoE
  contract is codebook-1-only. Owed, not done.
- `kExl3HadR128` as a registered ROCm op. It is not on a dense forward path, so
  registering it would add a surface nothing reaches. The transform itself IS
  implemented here — `Exl3Gemm` cannot exist without it — as an internal
  device function, and it is gated transitively (§Gates).
- Any matrix-core (`v_mfma`/WMMA), LDS-staged or split-K fast path. This row
  buys CORRECTNESS and the end of the reference tier. Speed is a later row.

## Why the CUDA kernel is not the donor

`src/vt/cuda/cuda_exl3.cu` cannot run on gfx1151, and the reasons are
structural rather than instruction-level. The first one alone settles it:

| Anchor | What it is | Why gfx1151 cannot have it |
|---|---|---|
| `cuda_exl3.cu:103` `constexpr int kSmemMax = 90 * 1024;`, asserted at `:694`, set at `:1893` via `cudaFuncAttributeMaxDynamicSharedMemorySize` | the kernel's shared-memory budget | **AMD LDS is 64 KiB per workgroup.** 90 KiB does not fit, and the tile arithmetic at `:694` is sized against that number, so no flag or launch bound recovers it |
| `:168` and `:1248` `mma.sync.aligned.m16n8k16` | the tensor-core inner product | `m16n8k16` has no AMD shape. RDNA3.5 WMMA is `16x16x16`, and the `FragA`/`FragB`/`FragC` blocking at `:151-155` is NVIDIA's fragment layout, not a width |
| `:257` `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | the fragment load | no equivalent instruction |
| `:241`, `:247`, `:251` `cp.async` / `cp.async.commit_group` / `cp.async.wait_group` | asynchronous global→shared staging, which is what the `SH_STAGES` pipeline at `:694` is made of | gfx1151 has no async global→LDS copy |

`src/vt/rocm/` is hand-written HIP ported from vLLM's ROCm `csrc/`, not a
hipify of our CUDA, and this tree ships no hipify script. That property is
preserved here.

## The donor that IS used

`src/vt/cpu/cpu_exl3_kernels.cpp` and `src/vt/cpu/cpu_exl3_dequant.cpp`, which
are pure portable C++ with no intrinsics. They are therefore both the SOURCE and
the ORACLE, which is what makes this row's gate a byte gate rather than a
tolerance:

| Device function | Transcribed from |
|---|---|
| `HadBlock128` levels 1-2 | `cpu_exl3_kernels.cpp` `HadBlock128`, the `s0/d0/s1/d1` butterfly |
| `HadBlock128` levels 4-64 | `cpu_exl3_kernels.cpp` `ShuffleHadWarp`, five xor-partner steps over 32 lanes, sign flip by XORing the f32 sign bit |
| the four (in, out) width arms and where each scale lands | `cpu_exl3_kernels.cpp` `HadRowBlock` |
| the three-step fused chain | `cpu_exl3_kernels.cpp` `Exl3GemmKernelCpu` |
| the tail-biting 16-bit window read | `cpu_exl3_dequant.cpp` `Exl3TileCodeword` |
| the three codebooks | `cpu_exl3_dequant.cpp` `Exl3DecodeCodeword` |
| the tensor-core permutation | `cpu_exl3_dequant.cpp` `Exl3TileRowMajorIndex` |

All three codebooks are instantiated, because an AMD box has no reason to see
fewer artifacts than an NVIDIA one:

- cb 0, 3INST: `x*89226354u + 64248484u`
- cb 1, MCG: `x*0xCBAC1FEDu`
- cb 2, mul1: `x*0x83DCD12Du`, then the **unsigned byte sum** of the product's
  four bytes into `0x6400`, reinterpreted as an fp16 bit pattern, then the fp16
  affine `k_inv = 0x1eee`, `k_bias = 0xc931`

For the byte sum this file follows the tree's own precedent rather than
inventing one: `src/vt/rocm/rocm_grouped_gemm.hip` `Dp4a` is a scalar
implementation documented as "bit-identical to `__dp4a`", with the hardware
instruction named as "a perf lever, not a correctness requirement". The same
sentence applies to `v_dot4` here.

## The exemplar mirrored

`src/vt/rocm/rocm_grouped_gemm.hip`, which serves `kMatmulBTQuant` — the same
"decode the weight INSIDE the GEMM" shape this op has. From it: the file
header naming the donor and its anchors, `#include <hip/hip_fp16.h>` +
`vt/rocm/rocm_device_bind.h`, the `__device__` bit-exact f16/f32 codec ports
rather than vendor intrinsics, the scalar-with-a-named-perf-lever comment
convention, and the registration through `rocm_ops.hip`.

## Design

### R1 — `CastF16KernelRocm`

A template on the source type, in `rocm_dense_basic.hip` beside the two
siblings, with the same packed-view row handling they have (`row_size`,
`in.stride[0]`) — which the CPU arm does NOT have and the CUDA arm does. It
accepts an f32 or a bf16 source and refuses an f16 one, which is what the
header contract says the op does.

The narrowing round is `DF32ToF16`, the bit-exact port, and not
`__float2half`. The two agree on RNE today; only one of them is a transcription
of `vt::F32ToF16`, and this op's whole gate is byte-equality with that function.

### R2 — `Exl3GemmKernelRocm`, three kernels in `src/vt/rocm/rocm_exl3.hip`

**Step 1, the input Hadamard.** `A[m,k]` fp16 with `pre_scale = suh`, into
`a_had` (which may alias `A`). One 32-lane group per 128-block, four groups per
workgroup; `h[4][32]` in LDS, so the five xor-partner levels are literally
`ShuffleHadWarp` with `__syncthreads()` where the CPU has a `memcpy`. Every
lane reads its four inputs BEFORE the first barrier and writes after the last,
which is what makes the in-place alias safe. LDS cost: 2 KiB.

**Step 2, the GEMM.** Grid `(n/16, ceil(m/16))`, 256 threads. Per k-tile: each
thread decodes ONE of the 256 codewords into `tile[Exl3TileRowMajorIndex(t)]`
in LDS — that is `Exl3DecodeTile` with the loop unrolled across threads — then
thread `(r, cc)` accumulates `acc += xv * tile[rr*16 + cc]` for `rr` ascending,
with `ti` ascending across the outer loop.

**That order is the CPU's order, element for element**, including its
`if (xv == 0.0f) continue;`, which is kept for a reason and not by habit: with
`acc == -0.0f` an added `+0.0f` product would flip the sign of the zero. So the
f32 accumulation is not merely close to the CPU arm, it is the SAME SEQUENCE OF
IEEE OPERATIONS, and the gate can require equality. LDS cost: 1 KiB.

**Step 3, the output Hadamard**, over the f32 `raw` into `c` with
`post_scale = svh`, in the f32→f32 or f32→f16 arm depending on `c.dtype` —
the same two arms `Exl3GemmKernelCpu` picks between.

`raw` is a device scratch allocated per call from the backend allocator. It is
`m*n` f32; for the Llama-3.2-1B EXL3 shapes at batch 1 that is at most 8192
floats. Named as a cost, not hidden: a fused kernel would not need it, and a
fused kernel is the later speed row.

**No matrix cores, no async copies, no cooperative launch, and 3 KiB of LDS
against a 64 KiB budget.** The structural blockers in the table above are not
worked around; they are not encountered.

## Tests

`tests/vt/test_exl3_rocm.cpp`, modelled on the device section of
`tests/vt/test_exl3_gemm.cpp` and sharing `tests/vt/exl3_fixture.h`, so no
second copy of the fixture exists.

1. **Decode, all three codebooks, bit-exact.** The device decode compared to
   `Exl3DecodeTile`/`Exl3DecodeCodeword` as INTEGERS (the fp16 bit patterns),
   asserted equal, not within a tolerance. Reached through `Exl3Gemm` with a
   trellis chosen so the GEMM reduces to the decode.
2. **GEMM against `Exl3GemmKernelCpu`** on identical inputs, asserted
   **byte-equal**. The bound is exact and the justification is in §Design:
   the three steps run the same IEEE f32 operations in the same order. This is
   a stronger claim than the CUDA arm's `1.0e-3`, and it is available only
   because this arm does not use tensor cores.
3. **`CastF16` against the CPU arm**, byte-equal, on an f32 and on a bf16
   source.
4. **Registration**, so a build that compiles the file but fails to register it
   is red.

Every device case SKIPS when no ROCm backend is registered, says so, and STILL
ASSERTS the precondition it skipped on — `assertions: 0` with `SUCCESS!` is a
skip wearing a pass and this family has already paid for it once.

## Gates

Run from the worktree; `ctest`, never a raw binary.

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build -j 2 test_exl3_rocm test_cast_f16 test_exl3_gemm test_exl3_dequant
ctest --test-dir build -R '^(test_exl3_rocm|test_cast_f16|test_exl3_gemm|test_exl3_dequant)$' --output-on-failure
```

On the device, inside an `rc` lease on `strix:gpu0` (never `ssh`):

```sh
cmake -S . -B build-hip -G Ninja -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 \
      -DVLLM_CPP_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build-hip -j 4 test_exl3_rocm
ctest --test-dir build-hip -R '^test_exl3_rocm$' --output-on-failure
VT_OP_PROVIDER_STATS=1 ./build-hip/examples/vllm-cli --model <exl3 ckpt> \
      --prompt "The capital of France is" --max-tokens 8 --temperature 0
```

The acceptance criterion of #2433 is the last line: **zero reference-tier
hits**, `GetReferenceTierHits() == 0`, and `CastF16`/`Exl3Gemm` absent from the
fallback report.

**What a run on `strix:gpu0` can and cannot conclude.** gfx1151 is an RDNA3.5
**APU** with unified memory (`managedMemory=1`, `integrated=1`). Any number from
it is a gfx1151 APU number and generalizes to no discrete Radeon. And because
the reference tier is OFF on a discrete board by design, the discrete arm of R2
is the arm that turns a refusal into a run — and it is the arm no device here
can execute.

## Evidence

Two `rc` leases on `strix:gpu0`, worker `rc-worker-lcjhd`, boot id
`a5bc8128-f6ad-4767-8614-6923f88032e1`, `gfx1151`, ROCm 7.2.4, Ubuntu 24.04
container, Release, `-DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151`.
Configure printed `ROCm backend: ENABLED for arch(es) [gfx1151]`, and
`rocm_exl3.hip.o` carries a real `hipv4-amdgcn-amd-amdhsa--gfx1151` offload
bundle (`llvm-objdump --offloading`), so the file was COMPILED for the board
rather than merely added to a source list.

**The dev box has no HIP toolchain.** `check-tree-compiles` covered 617 C++
translation units and none of the `.hip` ones; every statement below about this
code compiling or running comes from the lease.

### Job 1 -- `ea90cf10-d298-46c3-9298-7a0c257c7f8d`, 767 s

Tree: the index tree `8fbfeba50fa71a579e3f838820ae3c610e5407f7` (the job's
`BASE_SHA` names the SPEC commit `fe6208cbf`, because the implementation was
staged and not yet committed when the tarball was cut; the tarball is the tree,
not that commit).

`ctest -R '^test_exl3_rocm$'`: **7 test cases, 7 passed, 33 assertions, 0
failed.** Every GEMM arm reported `first differing byte = -1` -- byte equality
with the CPU arm, on `(bits, codebook)` = (3,0) at m=1 and m=20, (6,0), (3,1)
into an f16 C and into an f32 C, and (4,2), (5,2), (6,2). The in-place
`a_had == a` case and the codebook-discrimination case passed. Neighbours
unchanged: `test_exl3_dequant`, `test_exl3_gemm`, `test_cast_f16` all pass.

**Mutation table.** Each arm rebuilt one TU and relinked, asserted the binary's
mtime CHANGED before reading a verdict, then restored the file and verified it
with `sha256sum -c`. All four touched files ended byte-identical, and the
restored tree re-ran 7/7.

| # | Mutation | Result | Suite after |
|---|---|---|---|
| M1 | cb 2 multiplier `0x83DCD12D` -> `0x83DCD12E` | DETECTED | 5/7 cases, 29/33 assertions; **only the three cb-2 rows moved**, cb 0 and cb 1 stayed at `-1` |
| M2 | tile permutation `q + 6` -> `q + 5` | DETECTED | 4/7, 21/33 |
| M3 | `kInvSqrt128` `0.088388347648f` -> `0.088388f` (~50 f32 ulps) | DETECTED | 4/7, 21/33 -- **a 1.0e-3 tolerance would have passed this** |
| M4 | `codebook` argument replaced by a hardcoded `1` | DETECTED | 5/7, 22/33 |
| M5 | inner `rr` loop reversed (same algebra, different f32 order) | DETECTED | 4/7, 24/33 -- the ORDER claim is load-bearing |
| M6 | `CastF16` narrowing through `DF32ToBF16` instead of `DF32ToF16` | DETECTED | 6/7, 31/33 |

Job 1 produced **no end-to-end evidence**. It wrapped every generation in
`/usr/bin/time -v`, which is not in this container, so each leg exited 127
before the model ran and then printed a reassuring `reference_tier_notices=0`
computed over an empty log. That is an instrument failure wearing a result and
it is recorded, not quietly re-run.

### Job 3 -- `33f794b2-486b-4d96-a24d-d6e2e4af7a2b`, 136 s

Tree `46e60acef1a0209a72539b65cec647c0434ae4e0`. (Job 2 was killed: it passed
`--device rocm`, and `vllm-cli` accepts `auto|cpu|cuda` and nothing else, so
every leg exited 2 on the usage text. Job 3 passes no `--device` at all, which
is what #2433's own sweep did, and it now refuses to read a number from a log
that contains the usage banner or a nonzero exit.)

Model `llama32-1b-exl3-3bpw` (`turboderp/Llama-3.2-1B-Instruct-exl3` @ 3.0bpw,
`model.safetensors` 1,089,087,416 B), staged from the NAS to worker-local
`/tmp`. `VT_OP_PROVIDER_STATS=1`, prompt `The capital of France is`,
`--temperature 0 --max-num-seqs 1`.

**BEFORE is the same tree with the two `RegisterOp` lines wrapped in
`if (false)`, rebuilt and relinked in the same lease.** One difference, one box,
one binary recipe -- so the delta is the registrations and not the weather.

| Arm | reference-tier notices | 1 token | warm 8-token legs |
|---|---|---|---|
| BEFORE (registrations disabled) | **2** | 1.301 s | 9.711 s, 9.661 s -> 0.824, 0.828 tok/s |
| AFTER (native ROCm) | **0** | 0.300 s | 0.966 s, 0.968 s -> 8.278, 8.262 tok/s |

**10.01x** on the warm leg, **4.34x** on the single-token leg. Against #2433's
own 1.436 s for one token it is 4.8x, cited separately because that number came
from a different tree.

Every op resolved `selected=vt-native` on `device=5`, and the two arms emit the
**identical** greedy continuation, ` Paris. Paris is known for its famous`.
That is the correctness statement on the real artifact: the CPU reference tier
and the native kernel agree token for token on the checkpoint, not only on a
synthetic fixture.

One instrument note. `vllm-cli` is a 26,656-byte client and the registrations
live in `libvllm.so`, so the CLI's sha256 is `43e996fb...` in all three prints
including the BEFORE arm's. The mtime guard and the 10x behavioural difference
are what establish that a different library was linked; the CLI hash does not.

### What did NOT work, on the same board in the same lease

The **BF16 control aborted with a GPU hang**: exit 134, `HW Exception by GPU
node-1 (Agent handle: 0x5f5a094c6ca0) reason :GPU Hang`, with zero
reference-tier ops and every op `selected=vt-native`. Nothing in this row is
implicated -- the EXL3 legs immediately before and after it completed cleanly on
the same GPU -- but it means this row has **no BF16 denominator and no clock
attribution**, which is #2433's fourth acceptance line.

It is also a data point for
[#2511](https://github.com/mudler/vllm.cpp/issues/2511), whose hypothesis is
that "the bf16 arm runs and the Q4_K arm hangs ... points at the quantized
compute path rather than at anything the two arms share". Here a **bf16
safetensors Llama-3.2-1B** hung with the same signature, no quantized path
involved. That weakens the quantized-path framing rather than settling it: this
leg followed two EXL3 generations on the same device within seconds, so prior
GPU state is not excluded. Recorded as an observation, owned by `BACKEND-ROCM`
through #2511, not re-filed.

## Reachability

AGENTS.md "Nothing lands dead" asks whether a production entry point reaches
this code at its own merge commit, and the answer here is not an argument, it is
the measurement above.

`examples/vllm-cli` is a thin client of `include/vllm.h`. It loaded the real
EXL3 checkpoint through the loader and `ModelRegistry::Forward`, and
`VT_OP_PROVIDER_STATS=1` reported every op `selected=vt-native` on `device=5`
with **zero** reference-tier notices. No test constructs `Exl3GemmKernelRocm` by
hand; the suite calls `vt::Exl3Gemm`, which resolves through the op table, and
the model reaches the same entry.

**The mutation a fresh reviewer would run has already been run, and it is the
BEFORE arm.** Deleting the two production call sites -- the `RegisterOp` lines
in `rocm_ops.hip` -- and rebuilding in the same lease did not leave the gate
green: the reference-tier count went 0 -> 2 and throughput went 8.27 -> 0.83
tok/s. A capability that is not reached cannot move those numbers.

## Outcome

- **The donor decision was the whole design, and it held.** Transcribing the
  portable CPU reference rather than porting `cuda_exl3.cu` cost three plain
  kernels and bought a BYTE gate: the arm passed byte-equality on all eight
  `(bits, codebook, m, C dtype)` rows on the FIRST device run. The CUDA arm
  cannot claim that and does not try -- its 1.0e-3 bound exists for
  `mma.m16n8k16`'s unspecified accumulation order, and M3 shows a 50-ulp
  constant error slipping under exactly that kind of bound.
- **`kSmemMax = 90 * 1024` was the decisive fact and it was checkable in
  seconds.** Reading it first is what kept this row from spending its budget
  translating PTX that could never have been launched.
- **Why one thread owns one output element.** It is not a tiling choice. It is
  the only assignment under which a thread's accumulator sees the host's `ti`
  and `rr` sequence, which is what turns "close" into "equal". M5 is the proof:
  reversing the inner loop alone reds the suite.
- **Why a scratch `raw` buffer rather than a fused kernel.** The three steps are
  three launches with an `[m, n]` f32 staging buffer between the GEMM and the
  output Hadamard. Fusing them is what a speed row would do; here it would have
  changed the accumulation order and cost the byte gate for a number this row
  does not claim.
- **The rejected alternative was `__float2half`.** It agrees with
  `vt::F32ToF16` on everything the tests generate. It is still wrong for this
  file, because the gate asserts equality with a specific host function and an
  intrinsic that "agrees" is a coincidence a NaN payload or a compiler version
  can withdraw. The tree already made this call once, in
  `rocm_grouped_gemm.hip`; the codec moved to a header so it is made once.
- **`ccache` earned the second and third leases.** Job 3 reused the extracted
  tree and the build directory after proving both registrations were present
  unmodified at the start of their lines, and finished in 136 s where a fresh
  extract cost eleven minutes.

## Owed

- **`kExl3MoeMlp` on ROCm.** Unreached, unwritten, and unmeasurable on this
  fleet for the memory reason in §Scope.
- **`kExl3HadR128` as a registered ROCm op.** The transform exists; the
  registration does not, because nothing on a dense forward path calls it.
- **The BF16 denominator and the AMD clock attribution**, which are #2433's
  fourth acceptance line. The control hung the GPU (§Evidence), so the 8.27
  tok/s above is an EXL3-vs-EXL3 A/B and NOT a ratio against a bf16 target, and
  no clock state was captured on either leg. Both fall due before any
  competitive claim is made for this arm.
- **A discrete AMD board.** Until one exists here, "EXL3 now runs on discrete
  ROCm" is a code claim and not a measurement.
- **`kCastF16` on Vulkan, Metal and Tenstorrent**, the other three quarters of
  the gap `quant-exl3-shared.md` records.

## Risks/decisions

- **A byte gate is a strong claim and can red on a compiler.** `-ffp-contract=off`
  is already set for HIP in `CMakeLists.txt`, which is what stops an FMA from
  contracting the `acc += xv * w` into a different value. If the byte gate reds,
  the answer is to find the contraction, never to widen the bound to green.
- **Decoding a codebook as the wrong one is invisible to a shape check.** The
  wrong multiplier yields a weight with the right DISTRIBUTION and no
  correlation to the true one. That is why case 1 asserts integer equality on
  the fp16 bit patterns.
- **`strix:gpu0` may not free.** Then the CPU-gated half lands and the device
  half is reported UNMEASURED by name. It is not reported as passing.

## Stop conditions

- The device decode disagrees with `Exl3DecodeTile` on any codeword → stop, and
  re-derive from `codebook.cuh:56-90`. Never tune a constant to green.
- The byte gate needs a tolerance to pass → stop. The design claim in §Design
  is then false and the design, not the gate, is what changes.
- A reference-tier hit remains after both ops register → stop and read the
  provider report before writing a third kernel; the hit names the op.
