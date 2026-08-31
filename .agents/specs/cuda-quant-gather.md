# The CUDA dequantizing gather — wave KGATHER of `MODEL-MM-QWEN4-EXP`

Row: `MODEL-MM-QWEN4-EXP` · sibling spec:
[`qwen4-exp-flash-next.md`](qwen4-exp-flash-next.md) · the debt this discharges
was recorded by [#2083](https://github.com/mudler/vllm.cpp/issues/2083) under
that row's Owed section.

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

The gate change reaches EVERY GGUF load on CUDA, not only `qwen4_exp`: a
`token_embd.weight` in a block encoding now stays block-resident on the card
instead of expanding to bf16. The review of #2396 confirmed the second consumer
by name — `qwen3_5_gguf_weights.cpp:786` takes the same residency — so shipped
`qwen35` GGUFs are affected today, not hypothetically.

That is the intended memory win and it is also the risk, and it is why F1
mattered: the suite the first version of this row red on a CUDA build
(`test_gguf_keep_quant`) is the one that governs residency for every GGUF model
on the device, not for one architecture's n-gram table.

**Value behaviour, stated precisely.** For a **bf16 output** the two routes agree
bit for bit: both decode with the same scalar expressions and round once, so the
bf16 store is idempotent over the old expand-then-widen and no token can move.
For an **f32 output** the new path is **strictly more precise** — it decodes
straight to f32 instead of reading a value already rounded through bf16. That is
a value change on already-shipped models, in the direction of accuracy, and it is
recorded rather than left for someone to find in a diff.

`deepseek4` and `laguna` are NOT reached: both consume `token_embd` as a flat
host f32 array, so their loaders narrow the policy for that tensor.

## Gates

- Focused: `test_cuda_embedding_quant` — CPU-vs-CUDA **bit-exact** over every
  encoding `BlockToFloat` decodes, at one-block and five-block rows, in f32 and
  bf16 out, i32 and i64 ids, plus the deferred out-of-range report; and the
  decoder-list equality above.
- Policy: `test_gguf_keep_quant` ("the gather's DEVICE gate is the OP TABLE, not
  a hand-kept device list") and `test_qwen4_exp_gguf_weights` ("cuda follows the
  OP TABLE, not a hardcoded verdict"). Both DERIVE from the registry rather than
  hardcoding a verdict, so they read identically on a CPU-only and a CUDA build
  and cannot become a second competing source of truth about the same fact.
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
registering `kEmbeddingQuant` on the CUDA device IS the residency flip, and
`cuda_ops.cu`'s registrar performs it with one call,
`RegisterCudaBlockGather()`.

That registration was WITHHELD for the length of one commit — the call sat
commented with its reason, and `test_cuda_embedding_quant` asserted the op was
registered exactly when the TEST registered it, so it could not arrive without a
decision. It landed only after the device run below. The ordering was not
ceremony: registering it over decoders no GPU had executed would have routed
EVERY GGUF model's block-typed gather table on CUDA into never-executed code,
with no throw and no log if a decode were wrong.

**The arm is now REACHED**, and the mutation leg is what proves it rather than
asserting it: deleting the production call reds the gate (M1's shape), so the
registrar is the live call site and not a duplicate.

## Superseded: the earlier blocker, kept for its reasoning

**HISTORY, in the past tense.** This section describes the shape KGATHER tried
first and the gate that refused it. It is kept because the reasoning is what
produced the op-table seam above, and deleting it would leave that seam looking
like a preference. `check-device-leakage.py` reports rc 0 on this branch today.

At that earlier shape, `scripts/check-device-leakage.py` FAILED on exactly one
line:

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

## A near-miss worth recording: I edited a script a job was executing

While `rc` job `38a9b799` was running `/workspace/kgather/run.sh` on `dgx:gpu0`, I
edited that same file — adding a fifth mutation to its loop and a fifth suite to
its sibling list. **bash reads a script by BYTE OFFSET as it executes**, so
inserting text ahead of the interpreter's position can make it resume mid-token
and run something nobody wrote. Nothing in that run's output shows corruption and
it completed with coherent verdicts, but "it happened to be fine" is not a
property of the edit.

It is recorded here rather than left in session prose because a later reader
auditing `38a9b799` has no way to know the script changed under it, and would
reasonably assume the file on disk is what ran.

**Nothing in the F1/F2 discharge rests on that job.** Every claim about the head
comes from `28bfb44b` on `thor:gpu0`, submitted from a SEPARATE directory
(`/workspace/kgather-thor/`) with its own bundle, its own `/tmp` path and its own
mutation scripts, started twelve minutes after the last commit it gates. The
isolation was the response to this hazard, not a coincidence.

The rule that follows: a staged job directory is IMMUTABLE from the moment it is
submitted. Change a run by staging a new directory, which is what
`/workspace/kgather-head/` is.

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

## The device gate RAN, and the flip landed on it

`thor:gpu0` (Jetson Thor, **sm_110**, nvcc 13.0.88, aarch64, worker
`rc-worker-n8smh`), 2026-08-31, `rc` job `e53a20f5-f267-4672-8647-3fd82e5f0fe0`.
Both legs built from a staged bundle with no network dependency. Every rc below
is read from the command that produced it, and the job asserts the DIRECTION of
each verdict rather than printing a number for a reader to interpret.

| leg | build rc | test rc | result |
|---|---|---|---|
| RED `e0188c50b` (decoders + test, no wiring) | 0 | 1 | **RED, by name.** 5 cases, 1 passed, 4 threw `vt: cuda embedding: unsupported table dtype (f32/bf16 only)` at `cuda_ops.cu:861`. **32 assertions**, so a real device ran them |
| GREEN `e08bc069d` | 0 | 0 | **6 of 6 cases, 231 of 231 assertions.** All 18 block encodings gathered BIT-EXACTLY against the CPU arm, f32 and bf16 out, i32 and i64 ids, plus the deferred out-of-range report |
| M1 — restore the f32/bf16 refusal | 0 | 1 | **RED** (expected) |
| M2 — drop the Q4_K sub-block minimum | **1** | not run | **UNMEASURED ON DEVICE** ([#2393](https://github.com/mudler/vllm.cpp/issues/2393)) |
| M3 — allow FMA contraction | 0 | 0 | **SURVIVED — as PREDICTED in writing before the run** |
| M4 — stride blocks by elements, not bytes | 0 | 1 | **RED** (expected) |
| RESTORED | 0 | 0 | tree back to GREEN, 6 of 6 |

`assertions: 0` would have been a SKIP and the job says so by name. It never
occurred: 32 on the RED leg and 231 on the GREEN one.

**M2 did not build, and the build guard is why that is a recorded gap rather than
a false pass.** nvcc runs `-Werror=all-warnings`, and replacing `m1` with a
literal `0.0f` left it unreferenced (`error #177-D`). The job refused to run the
test, because the binary on disk was the previous leg's and would have printed a
green describing code nobody compiled. The mutation now reads `m1 * 0.0f`, which
keeps the variable live while still dropping the minimum; the fix is verified
against a control — the new form compiles under `-Werror` at rc 0 and the old
form reproduces the exact failure. The defect class is NOT unmeasured overall:
the host harness reds it at 457,498 of 968,199 elements, max|diff| 4.0295e+06.
Only its DEVICE measurement is owed.

**M3 SURVIVED exactly as predicted**, which is the outcome this spec recorded
before the run. It confirms on nvcc what was measured on g++: `d1*q` is exactly
representable for these encodings, so an FMA and a multiply-then-subtract round
identically. `DqMulSub` is defensive, not load-bearing, on two toolchains.

**Second architecture owed, not claimed.** This ran on sm_110. `dgx:gpu0`
(GB10, sm_121a) is queued as job `38a9b799` and will confirm the target
architecture and close M2 ([#2393](https://github.com/mudler/vllm.cpp/issues/2393)).

## sm_121a confirms it, and M2 is now measured on a device

`dgx:gpu0` (**NVIDIA GB10**, sm_121a, driver 580.173.02, nvcc 13.0.88), `rc` job
`38a9b799-7caa-4703-bfe2-4f393f6ac06c`, 2026-08-31, same script and same staged
bundle as the thor run. This closes the sm_121a half of
[#2393](https://github.com/mudler/vllm.cpp/issues/2393) and M2's device
measurement.

| leg | build rc | test rc | result |
|---|---|---|---|
| RED `e0188c50b` | 0 | 1 | RED by name |
| GREEN | 0 | 0 | **6 of 6 cases, 231 of 231 assertions** |
| M1 restore the refusal | 0 | 1 | **RED** |
| M2 drop the Q4_K minimum | 0 | 1 | **RED — 6863 and 1856 mismatches**, the measurement thor could not make |
| M3 allow FMA contraction | 0 | 0 | **SURVIVED**, predicted, on a second architecture |
| M4 element stride | 0 | 1 | **RED** |
| RESTORED | 0 | 0 | 6 of 6 |

M2's fix worked: `m1 * 0.0f` keeps the local referenced under
`-Werror=all-warnings` and still drops the minimum. So the two architectures now
agree on every mutation verdict.

**M5 correctly REFUSED to apply at that SHA**, and the refusal is the guard
working rather than a gap. M5 deletes the production registration call site,
which at `e08bc069d` was still commented out — zero occurrences — so the script
reported `MUT_APPLY(m5_delete_callsite)=FAILED` and ran nothing. A mutation that
cannot find its target must not report a verdict; M5's real run is on the head,
where the call site exists.

## The HEAD ran, through the PRODUCTION registration

`thor:gpu0` (Jetson Thor, sm_110, nvcc 13.0.88, aarch64), `rc` job
`28bfb44b-1dad-4986-9e78-8e4d69d02ebf`, 2026-08-31. GREEN leg `aeaf7089a` — the
head, with `RegisterCudaBlockGather()` live in the registrar and the test-scope
registration deleted. This is the run the review of #2396 asked for, and it
closes F2.

| leg | build rc | test rc | result |
|---|---|---|---|
| RED `e0188c50b` | 0 | 1 | 4 of 5 cases threw by name, 32 assertions |
| GREEN `aeaf7089a` | 0 | 0 | **7 of 7 cases, 250 of 250 assertions** |
| M1 restore the refusal | 0 | 1 | RED |
| M2 drop the Q4_K minimum | 0 | 1 | RED |
| M3 allow FMA contraction | 0 | 0 | SURVIVED, predicted |
| **M5 DELETE the production call site** | 0 | 1 | **RED — 5 of 7 cases fail, applied 1 -> 0** |
| M4 element stride | 0 | 1 | RED |
| RESTORED | 0 | 0 | 7 of 7 |

**M5 is the finding that mattered.** `cuda_ops.cu` and `cuda_embedding_quant.h`
both said "deleting this call reds that gate" while no leg had ever deleted it.
Now one has: removing the single production registration reds 5 of 7 cases, so
the registrar IS the reachable call site and not a duplicate. The comments state
a measurement instead of an expectation.

**Every sibling suite is rc 0 at the head**, including the two that were red on a
CUDA build one commit earlier: `test_gguf_keep_quant` (F1's 13 assertions) and
`test_qwen4_exp_gguf_weights` (3 cases, 7 assertions). Both flipped 1 -> 0, which
is the prediction this spec recorded BEFORE the run. `test_cuda_quant_dot`,
`test_ops_embedding_quant` and `test_ops_quant_dot` stayed 0.

250 assertions and not 231: the F5 tripwire case adds 19 and runs on every build.

## Two settled by the review, recorded rather than re-proven

**M3's survival is exhaustive.** The reviewer enumerated all 65,536 f16 patterns
and 199,098,368 products, of which ZERO round. So `d1*q` is exactly representable
across the entire domain and not merely on the sampled draws this spec measured;
`DqMulSub` is defensive exactly as written, on both toolchains and for every
input.

**The reach is broader than `qwen4_exp`.** `qwen3_5_gguf_weights.cpp:786` takes
the same residency, so shipped `qwen35` GGUFs now keep `token_embd` block-resident
on CUDA too. That is why F1 mattered as much as it did: the suite it red is the
one that governs residency for every GGUF model on the device, not for one
architecture's n-gram table.

## Two suites were red on CUDA builds, and this row's first evidence could not see it

Both legs of the runs above executed at `e08bc069d`, one commit BEFORE the flip.
That is a real limitation of that evidence and the reviewer of #2396 found it:
the 231 assertions were produced through a test-scope registration the flip later
deleted, so they prove the DECODERS on both architectures and not the WIRING.
The head rerun (`61a5c4fb`, dgx) closes that, and it carries a fifth mutation
(M5) so the reachability claim in `cuda_ops.cu`'s comment is measured rather than
asserted.

Adding `test_qwen4_exp_gguf_weights` to the sibling list — which the same review
asked for — immediately found **three more cases red on a CUDA build**, 7
assertions: the production `load_weights` hook refused the file, the n-gram table
read `bf16` where the case expects `Q8_0`, and two malformed-file cases never
reached their own assertions because the device guard threw first. All three have
the same cause as the `test_gguf_keep_quant` case below — the residency gate
answering false on CUDA — and all three are expected to pass at the head. That
expectation is written here BEFORE the run.

## The flip repaired a red the CUDA builds were already carrying

`test_gguf_keep_quant`'s case "a quantized GATHER TABLE keeps its blocks, per
encoding and per K" asserts `kKeepQuant` unconditionally, and `RouteGgufTensor`
reads `CurrentPlatform().device_type()`. With the predicate at `dev == kCPU`,
that case FAILS on any CUDA build — measured here, 44 cases with 1 failed across
six encodings plus the shipped `[320001536, 160]` IQ4_NL tensor. **That red
predates this row**: it has been there since the gather landed in
[#1989](https://github.com/mudler/vllm.cpp/issues/1989), and nothing noticed
because no CUDA gate ran this suite. The flip repairs it, and the sibling-suite
leg of this job is what surfaced it at all.

The other three sibling suites were clean on the same CUDA build:
`test_cuda_quant_dot` 17 of 17, `test_ops_embedding_quant` 6 of 6,
`test_ops_quant_dot` 32 of 32.

## The HEAD also ran on sm_121a — #2393 is discharged

`dgx:gpu0` (**NVIDIA GB10**, sm_121a, driver 580.173.02, nvcc 13.0.88), job
`ae08d765`, GREEN leg `aeaf7089a` — the head, production registration live:

| leg | build rc | test rc | result |
|---|---|---|---|
| RED `e0188c50b` | 0 | 1 | RED by name |
| GREEN `aeaf7089a` | 0 | 0 | **7 of 7 cases, 250 of 250 assertions** |
| M1 / M2 / M4 | 0 | 1 | RED |
| **M5 delete the production call site** | 0 | 1 | **RED, applied 1 -> 0** |
| M3 allow FMA contraction | 0 | 0 | SURVIVED, predicted |
| RESTORED | 0 | 0 | 7 of 7 |

All five sibling suites rc 0, including `test_gguf_keep_quant` and
`test_qwen4_exp_gguf_weights`.

**So the head is proven on BOTH architectures, with M5, and the last named gap in
[#2393](https://github.com/mudler/vllm.cpp/issues/2393) is closed.** sm_110 and
sm_121a agree on every verdict: seven legs, same directions, same counts.

**Provenance re-checked against the merged head** (`c06547281`, after the
65-commit reconciliation). `git diff aeaf7089a HEAD` over `src/ include/ tests/`
touches 58 files, and the gather-critical surface is BYTE-IDENTICAL across all of
them: `cuda_quant_dequant.cuh`, `cuda_embedding_quant.h`, `cuda_ops.cu`,
`cuda_quant_dot.cu`, `gguf_keep_quant.cpp`, `include/vt/ops.h`, `ops.cpp`,
`test_cuda_embedding_quant.cpp` and `test_gguf_keep_quant.cpp`. Main's
`cpu_ops.cpp` delta does not touch the embedding kernel or its registration. The
one measured file that DID change is `qwen4_exp_weights.cpp`'s refusal string,
recorded above. So the GPU evidence describes this head for everything it
measured.

## Provenance of the GPU evidence, re-checked at the moment of merge

The gated SHA is `aeaf7089a` (thor job `28bfb44b`). At each merge since, the
question is whether anything the GPU measured has moved. Run WIDE, over all of
`src/ include/ tests/`, not over a hand-listed path set — the narrow list this
row was given omits `models/qwen4_exp_weights.cpp`, which this row edits, and
`models/qwen3_5_gguf_weights.cpp`, which main edits.

As of the merge of `e4aa7c527`, `git diff aeaf7089a HEAD` over that whole tree
touches 14 source files, and exactly two matter here:

- **`qwen4_exp_weights.cpp` — mine.** The refusal STRING changed (it no longer
  offers `--device cuda` as a remedy, because that device now loads and then dies
  at a forward this model has no CUDA arm for). `test_qwen4_exp_gguf_weights`
  asserts on message CONTENT, and its GPU verdict was measured against the
  PREVIOUS string. The assertions it makes — that the message names the tensor,
  the device and the word "gather" — hold on both, and the suite is green locally
  after the change, but the GPU reading of that one suite predates this wording.
  Stated rather than glossed.
- **`qwen3_5_gguf_weights.cpp` — main's.** It gained a `prefault` parameter on
  `OwnGgufQuantBlocks`/`OwnGgufF16`. That is a signature change and NOT a
  residency change: the `kEmbeddingTable` route the reviewer identified as this
  row's second consumer is intact, so shipped `qwen35` GGUFs still take the CUDA
  gather residency. Its line moved `:786` -> `:793`, which is the ordinary reason
  to cite a SYMBOL rather than a line.

Nothing else in that delta reaches `src/vt/`, the residency policy, the op table
or either gated test, so the thor head result still describes this tree.

## Landing order with #2397, and the one line that is in NEITHER pull request

[#2397](https://github.com/mudler/vllm.cpp/pull/2397)
(`row/ENG-GGUF-RESIDENCY-RESOLVED-DEVICE`) makes `RouteGgufTensor` take the
engine's RESOLVED device instead of probing `CurrentPlatform()`, and it rewrites
the same two test files this row rewrote. `git merge-tree` CONFLICTS in
`tests/vllm/test_gguf_keep_quant.cpp` and
`tests/vllm/models/test_qwen4_exp_gguf_weights.cpp`.

The two changes AGREE on the principle and differ on the expression:

- this row asks `vt::OpRegistered(vt::OpId::kEmbeddingQuant, CurrentPlatform().device_type())`
- #2397 asks `kRouteDev == vt::DeviceType::kCPU`, with `kRouteDev` the resolved device

**The correct merged form is in neither pull request:**

```cpp
const bool gather_device_capable =
    vt::OpRegistered(vt::OpId::kEmbeddingQuant, kRouteDev);
...
const int gather_kept =
    vt::OpRegistered(vt::OpId::kEmbeddingQuant, kRouteDev) ? 13 : 0;
```

BOTH terms, not one. Taking #2397's side restores F1: `kRouteDev == kCPU` is
exactly the hardcoded device list the flip invalidates, and the suite then reds
13 assertions on a CUDA build with a GPU. Taking this row's side loses the
resolved device and goes back to probing the platform. **A "take theirs", a "take
both", and a clean automatic merge of these hunks are all wrong**, which is why
the conflict being VISIBLE is the good outcome.

Whoever lands second owes those two lines AND a re-run of the CUDA sibling leg
(`test_gguf_keep_quant` and `test_qwen4_exp_gguf_weights` on a GPU), because the
defect they guard against is invisible to every CPU lane.

## Owed

Both entries below are FILED, not merely recorded: the GitHub API is reachable
from this host (`gh api user` authenticates as `localai-org-maint-bot`). An
earlier reading of this row's history said writes were `403`; that was a
generalisation from ONE hidden issue and it is wrong.

- ~~The head on sm_121a~~ — **DISCHARGED**, and
  [#2393](https://github.com/mudler/vllm.cpp/issues/2393) is closed. The head ran
  green on GB10 with M5, and the two architectures agree on every verdict.
- **METAL, VULKAN, ROCM and TENSTORRENT gather arms** —
  [#2394](https://github.com/mudler/vllm.cpp/issues/2394). Each gather kernel
  asserts a float table by name and none registers `kEmbeddingQuant`, so each
  answers false to the residency gate and keeps expand-bf16; the `qwen4_exp`
  loader refuses them by name.
- **Performance.** This wave gates CORRECTNESS only. The decoders read byte-wise
  for alignment safety and one thread decodes a whole block. No throughput number
  is claimed, measured or implied, and no benchmark ID moves.
- **The end-to-end `qwen4_exp` forward on CUDA — and the reason CHANGED under
  this row.** While this branch was open, [#2391](https://github.com/mudler/vllm.cpp/pull/2391)
  landed the last four CUDA op arms, so all six `qwen4_exp` ops plus
  `vt::RmsNormGroup` now register for `kCUDA`. With those and this row's gather
  both landed, **what blocks a CUDA forward is neither op registration nor the
  loader.** The expected shape is a PREDICTION, written here as one because
  nothing has run it: partial dispatch through the PLE, then a NAMED REFUSAL at
  the first QSA layer, because `qwen4_exp_qsa_block.cpp` still reads three
  operands on the HOST — `CheckRopeLayoutsAgree`, `IndexerRows` on the block
  table, and `Qwen4ExpQsaIndex` on `kv_lens`. Owned by the QSADEV wave; cited,
  not touched.
- **`IsCudaKeepQuantSupported` excludes IQ4_NL and Q5_0**
  ([#2423](https://github.com/mudler/vllm.cpp/issues/2423)), which the released
  UD-IQ1_S stores its tensors in — a second wall the synthetic fixture never
  reaches, so no gate in this row could have shown it. Also cited, not touched.
  **No token has come out of a GPU for this model and none is claimed.**
