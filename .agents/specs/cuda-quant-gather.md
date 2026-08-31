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
| baseline, 18 codecs | 0 mismatches, max\|diff\| 0, ~14.2 M elements |
| baseline, gather kernel | 0 mismatches over an IQ4_NL [13, 160] table |
| M2 — drop the Q4_K sub-block minimum | **RED**, 457,498 of 968,192, max\|diff\| 4.0295e+06 |
| M4 — stride the block by ELEMENTS not BYTES | **RED**, 896 of 1120 gather elements |
| M3 — allow FMA contraction | **SURVIVES, and provably must** (see Design) |

One instrument failure is recorded rather than hidden: the harness's first run
reported 64 gather mismatches, exactly two blocks of 32. The cause was the shim
defaulting `blockIdx.x` and `threadIdx.x` to 1 alongside the extents, so the
single host thread started at index 2. The defect was the instrument's and read
as the kernel's.

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
