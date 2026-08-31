# CUDA keep-quant kernels for IQ2_XS and IQ4_XS — `QUANT-CUDA-IQ4XS-IQ2XS`

Issue: [#2260](https://github.com/mudler/vllm.cpp/issues/2260).
Branch: `row/QUANT-CUDA-IQ4XS-IQ2XS`.
Extends [`cuda-keepquant-gemm.md`](cuda-keepquant-gemm.md), whose
`KERNEL-QUANT-CIQ-GEMM-CUDA` owns the CUDA `kMatmulBTQuant` provider. Discharges
the CUDA half of the `QUANT-GGUF-IQ2_XS` and `QUANT-GGUF-IQ4_XS` rows of
[`quantization-matrix.md`](../quantization-matrix.md), and the premise of O19 in
[`glm5-next-flash.md`](glm5-next-flash.md).

## Now

`DONE`. The CPU arm of both encodings landed with #2247; this row is the device
arm of the same two `vec_dot`s, and it is measured green on `dgx:gpu0` (GB10,
`sm_121a`): 17 of 17 cases, 177,284 of 177,284 assertions, both reachability
mutations killed and the restore proven. See `## Outcome`.

## The gap, as measured on `673464ee1`

`src/vt/cuda/cuda_quant_dot.cu::IsCudaKeepQuantSupported` maps ten `DType`s to a
`WType`: `IQ2_XXS`, `IQ3_XXS`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ2_S`,
`IQ1_S`, `IQ1_XXXS`. **`IQ2_XS` and `IQ4_XS` are absent**, and `git grep -n
'kIQ2_XS\|kIQ4_XS' src/vt/cuda/` returns nothing at all.

An absent dtype does not refuse. The three CUDA consumers of that predicate
behave in two different ways, and the difference is the whole reason this row
exists:

| Seam | Today, for these two dtypes |
|---|---|
| `MatmulBTQuantKernelCuda` | `cudaStreamSynchronize`, then the CPU kernel over the same tensors. **MEASURED on GB10: this SEGFAULTS** when the tensors came from `vt::Backend::Alloc`, which is a plain `cudaMalloc` and is not host-addressable. Where the memory *is* host-accessible it is instead correct, host-speed and **not capturable** — the sync invalidates a decode graph. |
| `MatmulBTQuantGroupedKernelCuda` | the same drain-and-fall-back, with the same two outcomes. |
| `MoeGateUpSwiGLUGroupedCuda` | **throws** `gate/up must be the SAME CUDA keep-quant dtype`. There is no fallback behind that seam. |

`gguf_keep_quant.cpp::DeviceKeepQuantSupported` returns `true` for every dtype
on CUDA precisely because of the first row of that table, so the loader admits
these tensors as keep-quant blocks and the residency is already right. **This
row does not touch that function**, and the reason is recorded rather than
inferred: narrowing it to the CUDA kernel list would push `Q4_0`, `IQ4_NL` and
`MXFP4` back to `expand_bf16` on CUDA, which is a residency regression on
already-shipped models and no part of #2260.

### What each artifact needs, verified against the tree

- **GLM-5.3-Flash `UD-Q2_K_XL`**: 82 `IQ2_XS` + 3 `IQ4_XS` tensors. Both are
  needed. `docs/FEATURES.md:153` currently instructs the reader to use
  `--device cpu` for exactly this reason, citing #2260.
- **GLM-5.3 non-flash `UD-IQ1_S`**: 4 `IQ4_XS` tensors; `IQ1_S`, `IQ3_XXS`,
  `IQ2_XXS`, `Q2_K` and `Q3_K` are already in `IsCudaKeepQuantSupported`. **So
  `IQ4_XS` alone completes that arm's device admission.**

## Upstream anchors

Oracle: llama.cpp `b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`
([`oracles/llama-cpp.md`](../oracles/llama-cpp.md)). vLLM implements neither
encoding, so the secondary oracle rule applies exactly as it did for the CPU arm.

| Ported | From | To |
|---|---|---|
| `ggml_vec_dot_iq2_xs_q8_K_generic` | `b10451 ggml/src/ggml-cpu/quants.c:948` | `cuda_quant_dot.cu::DotIQ2XS` |
| `ggml_vec_dot_iq4_xs_q8_K_generic` | `b10451 ggml/src/ggml-cpu/quants.c:1283` | `cuda_quant_dot.cu::DotIQ4XS` |
| `iq2xs_grid` (512 u64) | `b10451 ggml/src/ggml-common.h:627` | `cuda_quant_iq_tables.cuh::d_iq2xs_grid` |
| `kvalues_iq4nl` (16 i8) | `b10451 ggml/src/ggml-common.h:1007` | `cuda_quant_iq_tables.cuh::d_kvalues_iq4nl` |

The in-tree behavioural reference is the CPU port of those same two functions:
`src/vt/cpu/cpu_quant_dot.cpp::VecDotIQ2_XSQ8_K` and
`src/vt/cpu/cpu_quant_dot.cpp::VecDotIQ4_XSQ8_K`, with the traits rows in
`src/vt/cpu/cpu_quant_traits.cpp` and the block layouts
`src/vt/cpu/cpu_quant_blocks.h::BlockIQ2_XS` (74 B) and `::BlockIQ4_XS` (136 B).

**`IQ4_XS` pairs with `Q8_K`, not `Q8_0`.** Read off `type_traits_cpu` at
`b10451 ggml/src/ggml-cpu/ggml-cpu.c:385-390` (`.vec_dot_type = GGML_TYPE_Q8_K`)
against `:379-384`'s `GGML_TYPE_Q8_0` for `IQ4_NL`. The 16-entry codebook is
shared with `IQ4_NL`; the block geometry is not, and it is the geometry that
picks the activation encoding. This has been got wrong before, so the pairing is
asserted in the test rather than assumed.

## Design

Both encodings are 256-element super-blocks against a `Q8_K` activation, so both
slot into the existing warp-per-output `QuantDotGemmKernel` with no new GEMM
shape, no new activation quantizer and no new launcher. The change is:

1. Two `__device__` codebooks in `cuda_quant_iq_tables.cuh`, mechanically
   derived from `src/vt/cpu/cpu_quant_iq_tables.h` by
   `scripts/gen-cuda-iq-tables.py` rather than hand-transcribed.
   `d_iq2xs_grid` is a `__device__` GLOBAL for the same measured reason
   `d_iq2xxs_grid` and `d_iq2s_grid` are: the 32 lanes of a warp own different
   super-blocks and therefore read different grid rows, and `__constant__`
   serialises a divergent read. `d_kvalues_iq4nl` is `__constant__`, following
   `d_kvalues_mxfp4`: 16 bytes, already resident in every cache.
2. `DotIQ2XS` and `DotIQ4XS`, each a literal transcription of its CPU twin,
   which is itself a 1:1 port of the oracle. **The accumulation order is
   upstream's and is not "improved".** `IQ2_XS` keeps its integer `bsum` and
   folds `0.125` once after the warp reduction, via `FinalFactor<kIQ2_XS>()`,
   exactly as `IQ2_XXS` and `IQ2_S` do. `IQ4_XS` keeps upstream's eight `f32`
   `sumf +=` steps per super-block — one per sub-block, `d1`/`d2` formed before
   the integer sums fold in — because that association is what the oracle's
   golden numbers were produced with.
3. `WType::kIQ2_XS = 10`, `kIQ4_XS = 11`, their `DotSuperblock` specialisations,
   the `FinalFactor` arm for `kIQ2_XS`, the two `IsCudaKeepQuantSupported` rows,
   and a `case` in each of the **three** dispatch switches — dense, grouped, and
   fused gate+up+SwiGLU. #967 is the reason the count is stated: it taught the
   predicate to say yes for two dtypes and extended only ONE switch, which
   turned a named refusal into a launch that never happened and an output tensor
   that kept whatever it held.
4. The drift seal grows two members. `IqTableSnapshot` gains `iq2xs_grid[512]`
   and `kvalues_iq4nl[16]`; `SnapshotIqTablesFromDevice` `static_assert`s each
   extent against `sizeof(d_…)` and copies it out, and the existing seal case
   memcmps both against the CPU tables.

No `f32` widening anywhere: the weight stays in its file blocks, the activation
is `Q8_K` as on CPU, and the output dtype is whatever the caller asked for
(`f32` or `bf16`), unchanged from the other ten encodings.

## Alignment

`BlockIQ2_XS` is 74 B and holds `uint16_t` members; `BlockIQ4_XS` is 136 B and
holds `uint16_t` members at offsets 0 and 2. Every block pointer in the GEMM is
`base + row * (nsb * block_bytes) + sb * block_bytes` with a 256-B-aligned
`cudaMalloc` base, and both 74 and 136 are even, so every `uint16_t` read is
2-byte aligned. This is the same argument that already holds for the 66-B
`BlockIQ2_XXS`. Neither dot does a 4-byte `__dp4a` read, so no stronger
alignment is claimed.

## Tests

**RED first, on the device.** The test additions land in a commit of their own,
BEFORE the kernels, and are built and run on the leased GPU so the failure is
observed rather than predicted. Expected RED: the fused-MoE case throws by name;
the graph-capture case fails because the CPU fallback synchronizes.

| Gate | What it can see that the others cannot |
|---|---|
| `IQ2_XS`/`IQ4_XS` rows in `kCases` | drives all five existing CUDA cases: dense vs CPU oracle at NMSE ≤ 1e-6, dense vs an independent f64 dequantize-then-dot at ≤ 5e-4, grouped vs CPU golden over a POISONED buffer, and the fused gate+up+SwiGLU seam. The last one is today's throw. |
| the oracle's own `vec_dot`, on the DEVICE, over REAL checkpoint bytes | `tests/vt/iq2xs_iq4xs_dot_golden.h` holds llama.cpp `b10451`'s OWN output for 4 super-blocks of `blk.3.ffn_gate_exps.weight` (IQ2_XS) and `blk.11.ffn_down_exps.weight` (IQ4_XS) of the staged artifact. At **k=256 the CUDA result is asserted BIT FOR BIT** against `kIq*DotPerBlockBits[b]`: one super-block means one contributing lane, and the warp tree then only adds exact zeros, so the reduction order cannot differ. At k=1024 four lanes contribute and the tree sums `(v0+v2)+(v1+v3)` where the oracle sums `((v0+v1)+v2)+v3`, so that case is asserted to a bound and the reason is written beside it. **This is the LOWER bound the arm needs**: a wrong grid row, a swapped scale nibble or a mis-shifted `scales_h` pair moves a reduction a little and passes any correlation check. |
| graph capture over `vt::MatmulBTQuant` | the only assertion that separates "ran on the device" from "drained the stream and ran on the host". Today's CPU fallback calls `cudaStreamSynchronize`, which is illegal under capture. |
| the device-codebook seal | a slipped literal in the generated `.cuh` fails on its own instead of waiting for a weight sample to address the drifted entry. |

Every float comparison is guarded with `std::isfinite` before it is believed: a
comparison against `NaN` is false, so an all-`NaN` forward reads as a perfect
match to a mismatch counter.

**Reachability.** The dtype arrives through `GgufFile::OpenOne` on a real header
(the existing `tests/vllm/test_gguf_dequant.cpp` "GgufFile reads IQ2_XS and
IQ4_XS tensors" case builds one), the loader routes it to `kKeepQuant`, and
`vt::MatmulBTQuant` / `vt::MoeGateUpSwiGLUGrouped` are the production ops the
model calls. The reachability mutation deletes the two `case` labels from the
fused-MoE switch in a scratch copy and reruns the focused gate; a green gate
would mean the gate measures the `__device__` function rather than the seam.

## Gates

- `scripts/agent-preflight.sh --fail-on-skip`, zero skips.
- Host C++ suites by hand, with counts: `test_ops_quant_dot`,
  `test_gguf_keep_quant`, `test_gguf_dequant`, `test_cuda_quant_dot` (CPU arm).
- On a leased `dgx:gpu0` (`sm_121a`) worker: a CUDA build and
  `test_cuda_quant_dot`, RED at the test-only commit and GREEN at the
  implementation commit, from the same build directory.
- On the leased box, `scripts/check-cuda-fat-gencode.py --compile-commands` over
  that build's `compile_commands.json`, which no CPU preflight can supply.

**Where the device evidence lands, so a later session does not re-queue it.**
The job is `bash /workspace/cudaiq2260/run.sh` on `dgx:gpu0`, staged from a git
bundle rather than a clone so it has no network dependency, and it writes
`/workspace/cudaiq2260/out/` (= `/mnt/nas_share/rc/cudaiq2260/out/` from the
development box): `run.log` plus `red.txt`, `green.txt`, `mutant_a.txt`,
`mutant_b.txt`, `restored.txt`, `cubin.log`, the two `compile_commands*.json`
and the `src_before/after` hashes. `RED_SHA` and `GREEN_SHA` are pinned in the
script. Read those files before submitting another lease.

### The end-to-end check this row was asked for cannot run, and #2260 is not why

Driving the staged GLM-5.3-Flash `UD-Q2_K_XL` through `--device cuda` was the
decisive test on this row's brief. It is blocked one level above these kernels,
and the block is in another row's code:

```text
src/vllm/model_executor/models/glm5_next_forward.cpp::Glm5NextHostForward
  "this forward is a HOST f32 reference and was handed a non-CPU queue.
   Every glm5_next primitive on this row -- the KDA recurrence, the DSA
   indexer, the mHC blocks, the MoE router and the attention -- is host code,
   and `vt::MoeRouterTopK` dispatches on the queue's device ..."
```

That refusal is unconditional on `queue.device.type != vt::DeviceType::kCPU`,
and it fires before any GEMM. So `--device cuda` on this artifact refuses BY
NAME whether or not these kernels exist, and a CUDA token from this model is not
available to be observed on this row at any effort. **No token is claimed.** The
device arm of that model is owed by
`MODEL-MM-glm5-next-glm5-next-for-conditional-generation` and its campaign
[#1998](https://github.com/mudler/vllm.cpp/issues/1998).

What #2260 does discharge is real and is the prerequisite: the fused MoE seam
stops throwing, the dense and grouped seams stop draining the stream to the
host, and the encodings become capturable. The remaining `--device cpu`
instruction in `docs/FEATURES.md` and `docs/USAGE.md` is therefore kept and its
REASON is corrected, because a record correction that leaves a wrong reason
standing is not a correction.

## Evidence

### The device gate, on `dgx:gpu0` (GB10, `sm_121a`) -- FINAL, attempt 3

**Which box.** `rc-worker-4b8lj` under an `rc` lease on `dgx:gpu0`:
`NVIDIA GB10, GPU-cb5c11ff-4ea1-5472-a9a6-c7a468a4d9f1`, driver `580.173.02`,
aarch64, 20 cores, `nvcc` release 13.0 V13.0.88, built
`-DVLLM_CPP_CUDA_ARCHITECTURES=121a`. 2026-08-31T07:20:29Z to 07:45:54Z.
**No number here transfers to `thor:gpu0`**, which is `sm_110`.

Arch proof, with its denominator: **37 `.cu.o` objects scanned, 37 `sm_121a`
cubins and nothing else.**

| Leg | commit | build | test rc | cases | assertions |
|---|---|---|---|---|---|
| RED | `52daeecea` | 0 | **139 (SIGSEGV)** | 0 passed, **1 failed**, 14 skipped | 66000 |
| GREEN | `0d595d2b6` | 0 | **0** | **17 passed, 0 failed** | **177284, 0 failed** |
| MUT_A fused switch | +mutation | 0 | **1** | 15 passed, **2 failed** | 177047 |
| MUT_B dense switch | +mutation | 0 | **1** | 13 passed, **4 failed** | **71589** |
| RESTORED | `0d595d2b6` | 0 | **0** | **17 passed, 0 failed** | **177284, 0 failed** |

**RED -> GREEN is measured, not asserted.** The RED commit carries the tests and
no kernels; it builds clean and its very first case takes the CPU-fallback path
and SIGSEGVs, aborting the run at 1 failed and 14 skipped. The same tests at the
implementation head pass 17 of 17 cases and 177,284 of 177,284 assertions,
including the bit-for-bit comparison against llama.cpp `b10451`'s own `vec_dot`
output on real GLM-5.3-Flash bytes, the stream-capture case, and the
real-GGUF-header chain. The RED reproduced identically across two independent
leases.

**Both reachability mutations killed, with distinct kill counts.** Deleting the
two `case` labels from the FUSED MoE switch fails 2 cases; deleting them from
the DENSE switch fails 4 and collapses the assertion count from 177,284 to
71,589. Two switches, two separate deletions, two separate kills -- one deletion
would have proved only the site it deleted. Note what kills them: the surviving
`default:` arm THROWS by name, so doctest records failed CASES with no failed
assertions, which is why the case count and not the assertion count is the
signal. That is a stronger kill than a compile error, which is why the mutation
removes a `case` and leaves the file compiling.

**The restore is proven, not assumed.** Each mutation was reverted, sha256'd
against the pre-mutation hash, and REBUILT before being re-run -- a restored
source with a stale binary has read red on a sibling row. RESTORED reproduces
GREEN exactly: 17 cases, 177,284 assertions, 0 failed.

**Sibling suites on the same box:** `test_ops_quant_dot` 249323/0 failed,
`test_gguf_dequant` 7400/0 failed, `test_ops_quant_traits` 6210/0 failed,
`test_gguf_keep_quant` **9 failed of 6470** -- BASE-CAUSED, see below.

**The three gates `agent-preflight.sh` skips for want of a build all PASS here**
on real data: `check-cuda-fat-gencode` 0, `check-cpu-isa-build` 0,
`check-arm-isa-build` 0 -- the last being asked its real question, since this box
is aarch64.

### Attempt 2 is what found the two defects, and it is kept

The run before the one above was GREEN-but-for-5-assertions (15 of 17 cases) and
its RED leg segfaulted. Those two results are what produced the FMA finding and
the fallback-segfault correction recorded below; the table above is the rerun
after both were fixed. Its log is at `/workspace/cudaiq2260/out-attempt2-fma/`
and attempt 1's staging failure at `/workspace/cudaiq2260/out-attempt1-clone-bug/`.
Three leases were spent on this row: one on a staging bug, one that found two
real defects, one green.

### The device gate, on `dgx:gpu0` (GB10, `sm_121a`) -- attempt 2, superseded

Same box, 2026-08-31T02:59:21Z to 03:24:55Z, same 37/37 `sm_121a` cubins.

| Leg | commit | build | test | cases | assertions |
|---|---|---|---|---|---|
| RED | `52daeecea` | 0 | **139 (SIGSEGV)** | 0 passed, 1 failed, 14 skipped | 66000, 0 failed |
| GREEN | `2c5dec2e4` | 0 | 1 | 15 passed, **2 failed** | 177284, **5 failed** |
| MUT_A fused switch | +mutation | 0 | **1** | 13 passed, **4 failed** | 177047, 5 failed |
| MUT_B dense switch | +mutation | 0 | **1** | 13 passed, **4 failed** | **71589**, 0 failed |
| RESTORED | `2c5dec2e4` | 0 | 1 | 15 passed, 2 failed | 177284, 5 failed |

**Reachability mutations: both killed, kill count 2 cases each.** GREEN fails 2
cases (the IQ4_XS rounding, below). Deleting the two `case` labels from the
FUSED MoE switch takes that to 4; deleting them from the DENSE switch also takes
it to 4, and drops the assertion count from 177284 to **71589**, because the
`default:` arm throws by name and aborts the cases mid-run rather than letting
them finish wrong. Two switches, two separate deletions, two separate kills --
one deletion would have proved only the site it deleted. Each restore was
verified byte-identical by sha256 AND rebuilt before being believed, and
RESTORED reproduces GREEN's counts exactly (177284 / 5 / 15), which is what shows
the restore was real and not a stale binary.

**Sibling suites on the same box:** `test_ops_quant_dot` 249323 assertions, 0
failed; `test_gguf_dequant` 0 failed; `test_ops_quant_traits` 7400, 0 failed;
`test_gguf_keep_quant` **9 failed of 6470** -- BASE-CAUSED, see below.

**The three gates `agent-preflight.sh` skips for want of a build all PASS here**
on real data: `check-cuda-fat-gencode` 0, `check-cpu-isa-build` 0,
`check-arm-isa-build` 0 -- the last of those being asked its real question,
since this box is aarch64.

### `test_gguf_keep_quant`'s 9 failures are the CUDA PLATFORM, not this change

All nine are the same assertion shape:
`RouteGgufTensor(..., GgufTensorRole::kEmbeddingTable, ...) == kKeepQuant`
returning `kExpandBf16`. The gather arm's device gate is
`DeviceQuantGatherSupported(dev) { return dev == vt::DeviceType::kCPU; }`, and
`RouteGgufTensor` reads `CurrentPlatform().device_type()` -- which on a CUDA
build running on GB10 is `kCUDA`. The suite encodes the CPU platform's answer.

This row changes neither file: `git diff` over
`gguf_keep_quant.cpp` and `test_gguf_keep_quant.cpp` between the merged `main`
and this head is EMPTY. It is a pre-existing property of running that suite on a
CUDA build, and it is reported rather than repaired because the repair is a
decision about that suite's platform assumptions, not a kernel port's.

### The RED leg SEGFAULTED, and that corrects a claim this spec made

The RED commit crashed with `SIGSEGV` in the FIRST case rather than failing the
fused-MoE assertion this row predicted. The reason matters more than the
prediction: `vt::Backend::Alloc` on CUDA is a plain `cudaMalloc`
(`cuda_backend.cu:104-107`), so its pointers are **not host-addressable**, and
`MatmulBTQuantKernelCuda`'s fallback -- `cudaStreamSynchronize` then the CPU
keep-quant kernel over the same tensors -- dereferences device memory on the
host.

**So "an absent dtype falls back to the CPU and returns correct tokens at host
speed" is not what happens for device-allocated tensors. It crashes.** This spec
and this row's earlier commit messages asserted the slow-but-correct story, and
that story was read off the source comment at
`cuda_quant_dot.cu` ("over the SAME unified-memory tensors"), never measured. It
holds only when the caller's memory really is host-accessible; GB10 has unified
physical memory, which is not the same thing as a `cudaMalloc` pointer being
host-mapped.

**What is NOT established, and is not claimed:** what the production loader
allocates for a kept-quant weight. If it is managed memory the production
fallback works and is merely slow; if it is `cudaMalloc` it crashes. This row
did not trace it, so it says so instead of guessing, and the correction is
limited to what was measured. Either way it strengthens rather than weakens the
case for #2260, and it is the same shape as the extraction lesson below: a
confident reading of a comment, standing in for a measurement.

### The host pre-check below CANNOT SEE A MISSING DECLARATION, and it did not

**Read this before the evidence that follows it, because it bounds all of it.**
The host harness described next reproduced the pinned oracle bit-exactly on all
eight real checkpoint blocks, and the kernels it validated **did not compile for
a device**. The first `sm_121a` build of them failed with four errors:

```text
cuda_quant_dot.cu(482): error: identifier "BlockIQ2_XS" is undefined
cuda_quant_dot.cu(535): error: identifier "BlockIQ4_XS" is undefined
cuda_quant_dot.cu(870): error: identifier "BlockIQ2_XS" is undefined
cuda_quant_dot.cu(874): error: identifier "BlockIQ4_XS" is undefined
```

The anonymous namespace's `using vt::cpu::...` list named seven sibling block
types and neither of these two. The structs existed the whole time
(`cpu_quant_blocks.h:191` and `:205`); only the declarations bringing them into
scope were missing. Fixed on `main` by `8846e3c4b`, merged here.

**Why the harness was structurally incapable of catching it.** It EXTRACTS the
two `__device__` functions by line range and compiles them in a translation unit
it writes itself -- one that does `using namespace vt::cpu;` and therefore has
every block type already in scope. It compiles the ARITHMETIC out of its file
and can never compile the FILE. So it can see a wrong grid index, a swapped
scale nibble or a mis-shifted bit pair, and it cannot see a name that the real
translation unit does not have. **An extraction is not a build**, and the
distinction is invisible while the extraction is passing.

That is this repository's recurring shape -- an instrument that returns a
confident, well-formed answer to a question nobody checked it was asking
([`verification.md`](../verification.md)) -- in the specific form a CUDA port
written on a box with no `nvcc` will keep producing, because extraction is the
only host-side check available. **Anyone doing that again should assume their
harness is blind to scope, includes, qualifiers and linkage, and say so beside
the result** rather than letting a bit-exact number imply more than it measured.

Nothing about the arithmetic evidence below is retracted: the same functions,
unchanged, are the ones `8846e3c4b` made compile. What is retracted is any
reading of it as "these kernels work", which it was never able to support.

**A check that CAN see this class, and its red-before.** The defect is one
set-difference away from visible -- the `Block*` types a `.cu` names against the
ones it declares or qualifies:

```sh
python3 - <<'EOF'
import re
s = open("src/vt/cuda/cuda_quant_dot.cu").read()
have = {m.group(1) for m in re.finditer(r'^using vt::cpu::(Block\w+);', s, re.M)} | \
       {m.group(1) for m in re.finditer(r'vt::cpu::(Block\w+)', s)}
print(sorted({m.group(1) for m in re.finditer(r'\b(Block[A-Z]\w*)\b', s)} - have))
EOF
```

At `6e06e4640` it prints `['BlockIQ2_XS', 'BlockIQ4_XS']`; at `2c5dec2e4` it
prints `[]`. That is a red-before and a green-after on the exact defect, taking
under a second on a box with no CUDA toolkit at all. It is listed under `## Owed`
rather than landed here, because a new checker needs its own spec, registration
and gate, and this row is a kernel port.

### Host pre-check of the ported math, before any lease was spent

`nvcc` is not on the x86 development box, so a bad port would otherwise have
been found only after a queue and a build on the leased GB10. The two new
`__device__` functions were therefore EXTRACTED from
`src/vt/cuda/cuda_quant_dot.cu` by line range — the exact bytes that will be
compiled for the device, not a retyped copy — compiled by `g++` with `__device__`
and `__constant__` defined away, given the GENERATED `cuda_quant_iq_tables.cuh`
as their codebooks, and run against the oracle's own numbers:

```text
iq2_xs block 0: got 0x3FED7254 (1.85505152)    oracle 0x3FED7254  BIT-EXACT
iq2_xs block 1: got 0xBE366684 (-0.178125441)  oracle 0xBE366684  BIT-EXACT
iq2_xs block 2: got 0xBF0B2FD5 (-0.543698609)  oracle 0xBF0B2FD5  BIT-EXACT
iq2_xs block 3: got 0x3FAC2A65 (1.34504378)    oracle 0x3FAC2A65  BIT-EXACT
iq4_xs block 0: got 0xC084FEC1 (-4.15609789)   oracle 0xC084FEC1  BIT-EXACT
iq4_xs block 1: got 0xBFC39E5A (-1.52827001)   oracle 0xBFC39E5A  BIT-EXACT
iq4_xs block 2: got 0x3E173704 (0.147670805)   oracle 0x3E173704  BIT-EXACT
iq4_xs block 3: got 0x3FFF0F4C (1.99265432)    oracle 0x3FFF0F4C  BIT-EXACT
iq2_xs k=1024: tree 0x401E9BFF  oracle-seq 0x401E9C00  margin 2.38e-07  bound 1.87e-06
iq4_xs k=1024: tree 0xC062D199  oracle-seq 0xC062D19B  margin 4.77e-07  bound 3.73e-06
bad=0
```

**Say what this is and what it is not.** It is `x86_64` `g++`, not `sm_121a`
`nvcc`, so it measures the ALGORITHM and not the device: it cannot see a wrong
`__constant__` qualifier, a misaligned device read, a launch that never happens,
or a codebook that failed to reach the GPU. It settles the port's arithmetic
against the pinned oracle before a lease is spent, and the device gates settle
the rest. It also confirmed the k=1024 design decision: the warp tree's order
and the oracle's sequential order differ by ONE ULP on both encodings, so
asserting bit equality against the oracle's total would have been red on the
device for a reason that is not a defect.

### A one-ref bundle has no HEAD, so `git clone` of it checks out nothing

Attempt 1 reached `dgx:gpu0` at 2026-08-31T00:12:03Z as `rc-worker-4b8lj` and
died 14 seconds later, after the CUDA toolkit had already installed:

```text
warning: remote HEAD refers to nonexistent ref, unable to checkout
FATAL: clone produced no CMakeLists.txt
```

`git bundle create <file> <one-branch>` writes that branch and no `HEAD`, so
`git clone <bundle>` succeeds, warns once, and leaves an EMPTY working tree.
Everything before it had passed -- the mount guard read the bundle's sha256 off
the share, `nvcc` reported `release 13.0, V13.0.88`, and `nvidia-smi` reported
`NVIDIA GB10, GPU-cb5c11ff-...`, driver 580.173.02, 20 cores -- so the failure
was purely staging. The fix is `git clone --branch <branch> <bundle>`, plus a
`rev-parse --verify` of BOTH pinned SHAs before any build is spent on either.

**The script's own strictness is what made this cheap rather than confusing.**
The `test -f CMakeLists.txt` guard fired immediately and named the step, instead
of letting `cmake -S` fail later with a message about a missing project. The
fix was verified by cloning the same bundle locally, checking out both SHAs, and
counting `kIQ2_XS|kIQ4_XS` in `cuda_quant_dot.cu`: **0 at the RED commit and 19
at the GREEN one**, which is the same precondition the job greps on the box.

Attempt 1's log is kept at `/workspace/cudaiq2260/out-attempt1-clone-bug/`
rather than deleted.

### The instrument's own precondition, checked before it was trusted

The device oracle-dot case does not feed the oracle's Q8_K activation blocks to
the GEMM -- it cannot, because the CUDA path quantizes the activation itself.
It regenerates the ORACLE's f32 signal from an LCG **restated** inside
`tests/vt/test_cuda_quant_dot.cpp` and lets the device quantizer do the rest. A
one-character drift in that restatement would fail the case on the device for a
reason that is not the kernel, and the reader would have no way to tell the two
apart from the output.

So the link was checked first, with the function's exact bytes lifted out of the
test file by line range:

```text
iq2_xs: our Q8_K of the restated signal vs the ORACLE's activation bytes: IDENTICAL (1168 B)
iq4_xs: our Q8_K of the restated signal vs the ORACLE's activation bytes: IDENTICAL (1168 B)
```

Both seeds, both 1168-byte blobs, byte for byte. The device case therefore fails
only about the kernel.

### The f64-dequant band was PREDICTED before the lease, not discovered on it

The CUDA gate asserts each dtype against an independent f64
dequantize-then-dot at upstream's 5e-4 (`test-backend-ops.cpp:4277`), and that
comparison holds a Q8_K-quantized activation against an f32 reference, so what
it measures is ACTIVATION error and how far that error travels depends on the
weight distribution the encoding produces. `IQ1_S` and `IQ1_XXXS` needed a
per-case ceiling for exactly that reason. A third harness therefore ran the
device dots over the SAME `RandomBlocks` weights, the same `GenerateData`
activation and the same M{1,4,32,512} x N{1,7,16} shapes the device case uses,
with the production `Q8_K` encoder and `BlockToFloat` as the reference:

```text
iq2_xs worst NMSE vs f64 dequant = 1.601e-04 at M=32  N=1   (band 5e-4)  UNDER
iq4_xs worst NMSE vs f64 dequant = 6.027e-05 at M=512 N=1   (band 5e-4)  UNDER
```

Both sit under the shared band with room, so **neither dtype takes a per-case
ceiling and none was added**. Had either come out over, the answer would have
been `NEEDS_DECISION`, not a wider band.

### `land/glm53-flash-and-dsa` conflicted with `main` -- RETIRED by `8846e3c4b`

This row was briefed to merge the landing branch so that a GLM-5.3 end-to-end
check would be possible. That merge was clean. Merging `origin/main` on top of
it is not:

```text
CONFLICT (content): Merge conflict in src/vllm/model_executor/models/model_registry.cpp
```

Two rows narrowed the SAME `multi_kv` refusal independently and neither knows
about the other. `land/glm53-flash-and-dsa` W5b-2c gates it on
`ModelFactory::consumes_multi_kv_cache`; `main`'s `KV-DSV4-MULTICACHE` W5 gates
it on `consumes_multi_kv` through a new `MultiKvRefusalApplies` helper. Taking
either side alone silently un-narrows the other row's model, which is why this
row does not resolve it: picking a side is a semantic change to two other rows'
dispatch and needs their gates, not a kernel port's.

**Measured as base-caused, not asserted.** `git merge-tree --write-tree`
between `origin/land/glm53-flash-and-dsa` and `origin/main`, with NONE of this
row's commits present, reports the identical single conflict on the identical
file. Whoever lands the landing branch reconciles it.

**RETIRED, and kept here because it is why this branch looked behind for
several hours.** `main` at `8846e3c4b` now CONTAINS `land/glm53-flash-and-dsa`,
so whoever landed it reconciled the two narrowings, and merging `origin/main`
into this row is clean. While it stood, this branch could not take `main`, so
`agent-preflight.sh` skipped `commit-trailers` and `commit-style` with
"origin/main is not an ancestor of HEAD" and both were run BY HAND against the
real merge base instead of left silent. Against the merged tree all three now
run normally and report OK.

### `check-pr-size.py` was red on this branch, base-caused -- also RETIRED

```text
ERROR: PR size check could not classify the change:
       unclassified repository path '.agents/scripts/glm53-dsa-first-load.sh'
```

Measured both ways at the time. Against this row's own commits the checker
reported `OK: every explicit path class is within its review budget`. Against
`origin/main..origin/land/glm53-flash-and-dsa` alone, with none of this row's
commits present, it reported the identical error. The unclassified path was
`land/glm53-flash-and-dsa`'s, added by `fe2117c63`.

**Also RETIRED by the same merge.** Against `main` at `8846e3c4b` the checker
reports `OK` for this branch. Both entries are kept rather than deleted because
each was a real red that had to be attributed before it could be dismissed, and
"it went away" is a different claim from "it was never ours".

The generated `d_iq2xs_grid` was checked a second way, independently of the
device seal: its 512 entries serialized little-endian FNV-1a-64 to
`0xc9b1ee61e79909bd`, the digest `cpu_quant_iq_tables.h` states for `kIq2xsGrid`
and `test_ops_quant_dot` re-derives.

## Risks

- **A number from `thor:gpu0` is not a number for `dgx:gpu0`.** `sm_110` and
  `sm_121a` are different targets; every result records its box.
- The k=1024 golden case cannot be bit-exact and saying so is part of the gate,
  not an excuse discovered afterwards.
- `dgx:gpu0` has crashed under long sequences; the lease script is written to be
  resumable and reports `df -h` before and after.

## Stop conditions

`NEEDS_DECISION` rather than widening a tolerance, bending a golden, or claiming
device parity not measured in the same tool. `NEEDS_CONTEXT` if the staged
artifact is absent from the leased worker.

## Outcome

**What was measured.** `IQ2_XS` and `IQ4_XS` run on `dgx:gpu0` (GB10,
`sm_121a`) through all three CUDA keep-quant dispatch seams, reproducing
llama.cpp `b10451`'s own `vec_dot` output BIT FOR BIT on real GLM-5.3-Flash
super-blocks, inside a stream capture, and reached from a real GGUF header
through the production reader and residency decision. 17 of 17 cases, 177,284 of
177,284 assertions, both reachability mutations killed, restore proven by hash
and rebuild. The GB10 job's own summary line reports every expectation met.

**What was rejected, and why.**

- *A tolerance on the IQ4_XS oracle comparison.* The device disagreed by 1 and 4
  ULP and the obvious move was to bound it. That would have permanently hidden
  the actual defect, which was nvcc contracting the accumulation into an FMA and
  thereby computing a different association from upstream's. The cause was
  identified on the host instead (simulating FMA reproduces the device's bits
  exactly) and fixed with `__fadd_rn`/`__fmul_rn`. **The gate that would have
  been widened is the one that found the bug.**
- *`-fmad=false` on the translation unit.* It would have fixed IQ4_XS and
  silently moved the numerics of ten other encodings that nobody measured.
  Scoped intrinsics keep the change to the kernel that needed it.
- *Hoisting IQ4_XS's eight f32 accumulation steps into one integer core.* A
  faster kernel computing a different number.
- *Narrowing `DeviceKeepQuantSupported` to the CUDA kernel list.* It would push
  Q4_0, IQ4_NL and MXFP4 back to `expand_bf16` on CUDA -- a residency regression
  on shipped models, and no part of #2260.
- *Repairing `test_gguf_keep_quant`'s 9 CUDA-platform failures.* Base-caused, and
  the repair is a decision about that suite's platform assumptions.

**Why each default has its value.** `d_iq2xs_grid` is a `__device__` global
because warp lanes read different rows and `__constant__` serialises a divergent
read -- the measured reason its two siblings are; `d_kvalues_iq4nl` stays
`__constant__` at 16 bytes. `FinalFactor<kIQ2_XS>` is `0.125` because the IQ2
codebooks store lanes at a fixed 8x magnitude, and `kIQ4_XS` takes the `1.0`
default because its per-sub-block delta is already folded in as `d * (ls - 32)`.
Neither dtype takes a per-case NMSE ceiling: both were measured under the shared
5e-4 band (1.6e-4 and 6.0e-5) rather than assumed to need one.

**Three leases, and what each bought.** One died on a staging bug (a one-ref
bundle has no HEAD). One found both real defects. One is the green above. The
two defects are the outcome that matters: a device build is the only instrument
that could see either, and both were invisible to a host check that had already
reproduced the oracle bit-exactly on all eight blocks.

## Owed

- **These two dispatch arms are landed but not yet SELECTED by any running
  model, and that is declared rather than implied.** The call sites are live and
  hot -- `vt::MatmulBTQuant`, `vt::MatmulBTQuantGrouped` and
  `vt::MoeGateUpSwiGLUGrouped` run for every keep-quant model on CUDA, and the
  loader already routes both encodings to `kKeepQuant` on that device. What no
  checkpoint reaches today is the DATA: `IQ2_XS` and `IQ4_XS` appear only in the
  two GLM-5.3 artifacts, `GlmMoeDsaForCausalLM` forwards nothing at all, and
  `Glm5NextForConditionalGeneration` refuses a non-CPU queue by name. This is
  the "unselected branch" shape in
  [`reachability.md`](../reachability.md), the wiring is owed by
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, and the campaign
  that tracks it is [#1998](https://github.com/mudler/vllm.cpp/issues/1998).
  The reachability mutation below therefore proves the SEAM, not a model step.
- **A checker for the defect class that reached `main` from this row**: the
  `Block*` types a CUDA translation unit names, minus the ones it declares or
  qualifies, must be empty. The red-before and green-after are recorded above
  (`['BlockIQ2_XS', 'BlockIQ4_XS']` at `6e06e4640`, `[]` at `2c5dec2e4`), it
  needs no CUDA toolkit, and it closes the one gap every host-side check of a
  CUDA port has. Owed here rather than landed because a new checker needs its own
  spec, preflight registration and gate.
- The ROCm arm of both encodings. `rocm-gg-keep-quant.md` owns it and already
  records `IQ2_*` / `IQ3_*` as owed; this row does not touch
  `src/vt/rocm/rocm_grouped_gemm.hip`.
- Speed. This row lands a correct kernel on the same MMVQ warp-per-output
  structure as the other ten encodings and claims no throughput number. No
  `__dp4a` vectorisation of either dot is attempted here; `DotQ4K` and `DotQ5K`
  show what that would look like when a row measures it.
- The `MXFP4` / `Q4_0` / `IQ4_NL` / `Q5_0` Q8_0-activation GEMM variant, which
  is unchanged and still CPU-fallbacks.
