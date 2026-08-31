# The CUDA dequantizing gather — wave KGATHER of `MODEL-MM-QWEN4-EXP`

Row: `MODEL-MM-QWEN4-EXP` · sibling spec:
[`qwen4-exp-flash-next.md`](qwen4-exp-flash-next.md) · the debt this discharges
was recorded by [#2083](https://github.com/mudler/vllm.cpp/issues/2083) under
that row's `## Owed`.

## What this fixes

`vt::Embedding` has taken a block-quantized table since W6a
([#1989](https://github.com/mudler/vllm.cpp/issues/1989)) and decodes ONE ROW
per gathered id, a port of `ggml_compute_forward_get_rows_q`
(`ggml/src/ggml-cpu/ops.cpp:4850` @ `b10451`). That existed on the CPU only.
`EmbeddingKernelCuda` asserted `f32/bf16`, so
`DeviceQuantGatherSupported` (`gguf_keep_quant.cpp`) returned true for `kCPU`
alone and `RouteGgufTensor` sent every gather table on every other device to
`kExpandBf16`.

**That refusal was not a missing optimisation. It was the reason this model has
no GPU arm.** Derived from the committed 1224-tensor manifest of
`unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S`
(`tests/vllm/models/qwen4_exp_gguf_manifest.inc`),
`per_layer_token_embd.weight` is `[320001536, 160]` IQ4_NL:
**28,800,138,240 B = 26.822 GiB** of blocks against
`320001536 * 160 * 2 = 102,400,491,520 B = 95.368 GiB` expanded to bf16, on a
GB10 with ~119.6 GiB for everything. So `qwen4_exp` refused on every non-CPU
device ahead of tensor I/O, and the device-resident quantized table is what
makes a GPU arm possible at all.

`llama.cpp` does NOT have this shape. Its CUDA `get_rows` (`getrows.cu:172`,
`get_rows_cuda`) dispatches F16/BF16/F32/I32 and the LEGACY quants
Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, and aborts on every K-quant and every IQ type; PR
\#27742's `qwen4exp` graph pins the n-gram table to the CPU by tensor class. So
there is no upstream gather to copy, which raises the evidence bar rather than
lowering it.

## Oracles

**vLLM has no implementation.** Its embedding is
`torch.nn.functional.embedding` over a dense tensor; no revision carries a
block-quantized gather, and `qwen4_exp` is not in vLLM at any revision (the row
spec records the live read). So `AGENTS.md` §"When vLLM has no implementation"
applies and the secondary oracle is **llama.cpp** (`llama-cpp`, pinned
`b10451`, `gateable = yes`).

Two deliberate deviations from that oracle, stated here rather than left for a
reader to discover:

1. **The gather shape is ours, not `k_get_rows`'s.** Upstream's CUDA gather
   cannot run this model's encodings (above), so this port takes upstream's
   DECODE and this project's own gather.
2. **The decoders transliterate the SCALAR path, not `convert.cu`.**
   `src/vt/cpu/cpu_quant_dequant.cpp` is the byte-for-byte port of
   `ggml-quants.c`'s `dequantize_row_*`, and it is what the CPU arm runs.
   Upstream's CUDA `dequantize_block_*` reassociate the scale products and are
   NOT bit-identical to their own scalar twins. Matching the scalar order is
   what makes CPU-vs-CUDA an EXACT gate instead of a tolerance — and a
   tolerance is precisely what would hide the failure class this row spent a
   day isolating (dropped repack markers -> NaN -> all-zero logits -> every
   token id 0, nothing thrown, nothing logged).

## Design

- `src/vt/cuda/cuda_quant_dequant.cuh` — one `Dq*` codec per block dtype,
  `Decode(blk, y)` writing `kElems` outputs, plus the grid-stride gather kernel
  `EmbeddingQuantGatherKernel`. Included from INSIDE
  `vt::cuda::{anonymous}` in `cuda_quant_dot.cu`, the only TU that may see
  `cuda_quant_iq_tables.cuh` (it DEFINES the device codebooks, so a second
  includer is a duplicate symbol, not a second copy).
- `src/vt/cuda/cuda_embedding_quant.h` — the two-function seam
  (`EmbeddingQuantSupported`, `LaunchEmbeddingQuant`) plus `EmbeddingQuantErr`,
  the out-of-range record BOTH translation units write. Stated once rather than
  mirrored: a layout that drifted between writer and reader would report a
  wrong id and never fail to compile.
- `cuda_ops.cu` — `EmbeddingKernelCuda` admits a block table, refuses an
  undecodable one BY NAME, and routes it through the SAME deferred error ring
  the float path uses. Its local `EmbeddingErr` becomes an alias of the shared
  record.
- `gguf_keep_quant.cpp` — `DeviceQuantGatherSupported` admits `kCUDA`. LAST,
  and only behind the evidence below: a flipped gate over a broken kernel
  converts a clean refusal into silent garbage.

Three decisions worth their reasons:

**Unaligned loads.** A block base is not 4-byte aligned in general (66-byte
IQ2_XXS, 110-byte Q3_K, 210-byte Q6_K), and a misaligned 32-bit device load is
a FAULT, not a slow load. Every multi-byte read goes through byte-assembly
(`DqU16`/`DqU32`). This is the one systematic difference from the CPU code,
which uses `std::memcpy`.

**Contraction, and the claim this spec had to withdraw.** CXX is compiled
`-ffp-contract=off` (`CMakeLists.txt:55`); nvcc defaults to `--fmad=true`. Q2_K,
Q4_K and Q5_K compute `d*q - m`, exactly the shape a contracting compiler folds
into one FMA, so `DqMulSub` states the two operations with
`__fmul_rn`/`__fsub_rn`. This spec first asserted that the fold "differs in the
last bit". **That was reasoning, not a measurement, and the measurement
falsifies it.** Compiling the host transliteration harness with
`-ffp-contract=fast -mfma` and `DqMulSub` rewritten to `a * b - c` emits 12 FMA
instructions where the guarded build emits 0 — so the probe demonstrably
APPLIED — and changes **nothing**: 0 mismatches across 2,884,352 Q2_K/Q4_K/Q5_K
elements.

The reason is a width argument, and it is measured too. `d` is an f16 widened to
f32, so 11 mantissa bits; the sub-scale is 6 bits and the quant 4 (5 with Q5_K's
high bit), so `d1 = d*sc` and then `d1*q` need at most 21 bits and are EXACTLY
representable. A direct probe finds `d1*q` exact in **3,874,835 of 3,874,835**
random draws. The only rounding left is the subtraction, which an FMA and a
separate multiply-then-subtract perform identically.

So `DqMulSub` is DEFENSIVE, not load-bearing, and the spec says so rather than
keeping a claim that reads as measured. It stays: it costs nothing, and it makes
the invariant explicit instead of leaving the numerics resting on a mantissa-width
argument that a future encoding with a wider scale would silently break. The
consequence for the on-device gate is stated ahead of the run: **mutation M3 is
expected to SURVIVE**, and a survived mutation with a reason is a result, not a
gap. Every other decoder was checked case by case and pairs no multiply with an add.

**One thread per BLOCK, not per row.** The CPU arm parallelises over gathered
rows; a row of the shipped table is five IQ4_NL blocks, so per-row would leave
four fifths of a warp idle on the very table this exists for.

**The device predicate keeps no dtype.** `DeviceQuantGatherSupported(dev)` is
sound only while the CUDA decoder list EQUALS the list `vt::cpu::BlockToFloat`
answers for. That is asserted, not assumed:
`tests/vt/test_cuda_embedding_quant.cpp` sweeps every `DType` and requires
`EmbeddingQuantSupported(dt) == (BlockToFloat(dt) != nullptr)`. A decoder added
to one side alone reddens it.

## Blast radius, stated because it is larger than one model

Flipping the gate changes EVERY GGUF load on CUDA, not only `qwen4_exp`: a
`token_embd.weight` in a block encoding now stays block-resident on the card
instead of expanding to bf16. That is the intended memory win and it is also
the risk. For a bf16 output the two routes should agree bit-for-bit (both
decode with the same scalar expressions and round once to bf16), which is what
the bf16 arm of the parity gate measures; an f32 output is strictly more
precise than the pre-existing expand-then-widen.

## Gates

- Focused: `test_cuda_embedding_quant` — CPU-vs-CUDA **bit-exact** over every
  encoding `BlockToFloat` decodes, at one-block and five-block rows, in f32 and
  bf16 out, i32 and i64 ids, plus the deferred out-of-range report; and the
  decoder-list equality above.
- Policy: `test_gguf_keep_quant` (`the gather's DEVICE gate ...`) and
  `test_qwen4_exp_gguf_weights` (`cuda LOADS, because KGATHER ...`), both
  red-first against the pre-KGATHER predicate.
- Full: `scripts/agent-preflight.sh`.

**Measured before the lease, on the host transliteration harness** (the device
decoders compiled as ordinary C++ with the CUDA vocabulary shimmed and the
codebooks aliased to the CPU tables — a transcription proof, NOT a device run):

| leg | result |
|---|---|
| baseline, 18 codecs | 0 mismatches, max\|diff\| 0, 13,428,192 elements |
| baseline, gather kernel | 0 mismatches over an IQ4_NL [13, 160] table |
| M2 — drop the Q4_K sub-block minimum | **RED**, 457,498 of 968,192, max\|diff\| 4.0295e+06 |
| M4 — stride the block by ELEMENTS not BYTES | **RED**, 896 of 1120 gather elements |
| M3 — allow FMA contraction | **SURVIVES, and provably must** (see Design) |

One instrument failure is recorded rather than hidden: the harness's first run
reported 64 gather mismatches, exactly two blocks of 32. The cause was the shim
defaulting `blockIdx.x` and `threadIdx.x` to 1 alongside the extents, so the
single host thread started at index 2. The defect was the instrument's and read
as the kernel's.

## The gate is a REGISTRY QUERY, and the flip is one registration

The first shape of this row named the device: `dev == kCPU || dev == kCUDA` in
`gguf_keep_quant.cpp`. That failed `scripts/check-device-leakage.py`, which was
right — the GGUF loader is the device-agnostic layer, and a hand-kept per-device
set written into it drifts from the kernels it claims to describe.

The fix is the one the checker's own message names, and it is now in:

- `OpId::kEmbeddingQuant` (`include/vt/ops.h`, appended before `kCount` so no
  existing id shifts).
- `vt::Embedding` routes to it when `IsBlockQuant(table.dtype)`, so a device with
  no block gather refuses BY NAME through the ordinary `GetOp` message instead of
  dispatching into a kernel that asserts on the dtype one frame later.
- `DeviceQuantGatherSupported(dev)` is `vt::OpRegistered(kEmbeddingQuant, dev)`.
  It names no device, exactly as `GgufQuantComputeAvailable()` beside it asks
  about `kMatmulBTQuant`. `check-device-leakage.py` reports rc 0, DSR 32 ==
  baseline 32 — verified, not assumed.
- kCPU registers the same `EmbeddingKernel` under both ids (it already branches
  on the table dtype). The four devices whose gather kernels assert a float table
  by name register neither.

**A consequence worth stating, because it changes what "the flip" means.** The
capability is now the REGISTRATION. There is no separate boolean left to flip:
registering `kEmbeddingQuant` on the CUDA device IS the residency flip. That
registration is therefore WITHHELD — the two lines sit commented in
`cuda_ops.cu` with the reason — until `tests/vt/test_cuda_embedding_quant.cpp`
runs green on a CUDA host. `test_cuda_embedding_quant` asserts
`CHECK_FALSE(OpRegistered(kEmbeddingQuant, kCUDA))` so the registration cannot
arrive without someone deciding to add it, and flipping it is two edits in one
commit: uncomment the registrar, change that one line to `CHECK`.

**So the CUDA gather arm is landed UNREACHED, and this is the notice AGENTS.md
requires.** What is unreached: the block arm of `EmbeddingKernelCuda` and every
decoder in `cuda_quant_dequant.cuh`. The row that owns the wiring:
`MODEL-MM-QWEN4-EXP`. It is listed under `## Owed` below. The GitHub API is
suspended, so no issue number is available to cite and this record carries the
obligation instead.

## Superseded: the earlier blocker, kept for its reasoning

`scripts/check-device-leakage.py` FAILS on this branch, on exactly one line:

    ERROR: DSR REGRESSION in bucket 'kcuda': 1 > baseline 0.
    src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:188
    return dev == vt::DeviceType::kCPU || dev == vt::DeviceType::kCUDA;

That is the flip itself, and nothing else in this change trips it (baseline for
this file is 0 `kCUDA` tokens). The gate is not being pedantic: naming a device
in the device-agnostic loader is the debt it exists to stop, and its message
names the fix — "Ask the op/provider table the question instead
(`vt::OpRegistered` / Platform capability)". `GgufQuantComputeAvailable()` two
screens above already does precisely that for the GEMM arm.

The neighbours do NOT license this. `DeviceKeepQuantSupported` names `kROCM` in
the same file, but the checker counts only `kCUDA`, so those sites are
grandfathered rather than approved — reading them as precedent would be reading
an unmeasured bucket as a permission. Raising the baseline is forbidden by the
checker's own message, and the ALLOWLIST route is for sites that ARE the CUDA
platform leg, which the shared loader is not.

**Three ways to close it, with a recommendation:**

1. **A new `OpId::kEmbeddingQuant`**, registered by the backends that decode a
   block row, dispatched from `vt::Embedding` when the table is block-typed, and
   asked by the loader through `vt::OpRegistered`. This is the principled fix: it
   names no device, it makes the capability a fact of the registry rather than a
   hand-kept list, and it retires the CPU/CUDA list-equality test by construction.
   The cost is an enumerator in `include/vt/ops.h`, which most of the tree
   includes, so it is a near-full rebuild to verify.
2. **A named provider** on the existing `kEmbedding` op, probed with
   `OpProviderNameAt`. No hot-header change and a small rebuild, but it uses a
   provider NAME as a capability channel, which is not what the name is for, and
   it perturbs selection for an op every model calls.
3. **Leave the flip out** and land the kernel alone.

**Recommended: (1), as its own scoped change.** (2) is a workaround that a
reviewer should reject, and (3) is what this branch does today.

**So the flip is NOT landed.** `DeviceQuantGatherSupported` on this branch admits
`kCUDA` and the branch therefore does not pass its own gate; it must not merge in
this state. The kernel below it is complete and independently useful, and
separating the two is what "the flip is last" means in practice.

## A pre-existing flake this row did NOT cause

`test_cpu_x86_llamacpp_floor` FAILS on this development box under load, and it
failed **identically on the untouched baseline** — measured before any file in
this row was edited, on `origin/main` with an empty diff, and again at the end.
The assertion is `self.assertEqual(got.returncode, 2)` getting 4, from
`test_a_contended_leg_is_discarded_and_never_summarised`; the harness discards a
contended leg and retries, and under a 1-minute load average of 35-153 (two
sibling CUDA waves compiling) the retry path returns a different code than the
case expects.

Recorded here because a flake discovered later looks like a regression the last
change caused. It is not this row's, it is load-dependent rather than
tree-dependent, and it is the ONLY preflight failure on this branch that KGATHER
does not own.

## The flip, as a procedure

Staged at `/workspace/kgather/` on the shared NAS and queued on `dgx:gpu0` (job
`38a9b799-7caa-4703-bfe2-4f393f6ac06c`). RED leg `e0188c50b` (decoders and test,
no wiring), GREEN leg the branch head, then four mutations, a restore, and four
sibling suites. The job asserts the DIRECTION of each verdict rather than
printing an rc for a human to interpret, refuses to run a stale binary when a
mutation fails to build, and calls `assertions: 0` a SKIP by name.

When the GREEN leg reports `VERDICT(GREEN)=GREEN` with a non-zero assertion
count, the flip is three edits in ONE commit:

1. `src/vt/cuda/cuda_ops.cu` — uncomment `RegisterCudaBlockGather();` in the
   registrar.
2. `tests/vt/test_cuda_embedding_quant.cpp` — delete
   `RegisterCudaGatherForTest` and its four call sites, and change the registrar
   case's IFF to a plain `CHECK`.
3. This spec and `docs/FEATURES.md` — the CUDA arm becomes reached.

If it reports RED, the kernel is wrong on device and nothing flips.

## Owed

- **The other four devices.** METAL, VULKAN, ROCM and TENSTORRENT gather
  kernels each assert a float table by name; their arms are owed and the
  `qwen4_exp` load-time guard still refuses them BY NAME.
- **Performance.** This wave gates CORRECTNESS only. The decoders read
  byte-wise for alignment safety and one thread decodes a whole block; no
  throughput number is claimed, measured or implied, and no benchmark ID moves.
- **The end-to-end `qwen4_exp` forward on CUDA.** Still refused by
  `ModelRegistry::Forward` and the KV-cache spec (W5c), so this wave removes
  the LOAD-side blocker and does not produce a token on a GPU.
