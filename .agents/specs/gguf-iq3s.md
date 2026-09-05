# GGUF IQ3_S (ggml type 21) — reader trait and row decoder — `QUANT-IQ3S`

Issue: [#2510](https://github.com/mudler/vllm.cpp/issues/2510).
Branch: `row/QUANT-IQ3S`. Owning row: `BACKEND-ROCM`.
Sits beside [`gguf-iquant-dsv4.md`](gguf-iquant-dsv4.md), which owns the IQ2/IQ3
family decoders, and [`cuda-quant-gather.md`](cuda-quant-gather.md), which owns
the CUDA row-gather codec set this change has to extend in step.

## Now

`READY`. The reader, the vt dtype, the CPU row decoder and the CUDA gather
codec have landed; the keep-quant `vec_dot` pair is `## Owed` and #2510 stays
open for it.

## The gap, as measured on `6bf3abb58`

`src/vllm/model_executor/model_loader/gguf_reader.cpp::FindGgmlTraits` has a
case for ggml ids 16 `IQ2_XXS`, 17 `IQ2_XS`, 18 `IQ3_XXS`, 19 `IQ1_S`,
20 `IQ4_NL`, 22 `IQ2_S` and 23 `IQ4_XS`, plus the fork ids 39/40/41/66.
**Id 21, `IQ3_S`, is the one hole in that run**, and `GgufFile::Open` refuses
the whole file on the first tensor that carries it:

```text
vllm_engine_load: gguf: tensor "blk.11.ffn_gate.weight" has unknown ggml type id 21
```

Measured on the staged artifact, not inferred. `/mnt/nas_share/rc/ckpt/
qwen38-27b-ud-q4km/Qwen3.8-27B-UD-Q4_K_M.gguf` (16,464,440,224 B) parses to 866
tensors, of which the type histogram is

| ggml id | name | count |
|---|---|---|
| 0 | F32 | 360 |
| 8 | Q8_0 | 106 |
| 11 | Q3_K | 7 |
| 12 | Q4_K | 104 |
| 13 | Q5_K | 131 |
| 14 | Q6_K | 30 |
| 20 | IQ4_NL | 7 |
| 21 | **IQ3_S** | **4** |
| 23 | IQ4_XS | 117 |

Every id in that table except 21 already has a reader trait. The four blocking
tensors are `blk.11.ffn_gate.weight`, `blk.14.ffn_down.weight`,
`blk.15.ffn_down.weight` and `blk.17.ffn_down.weight`, each `[5120, 17408]` or
`[17408, 5120]` = 89,128,960 elements = 348,160 super-blocks = 38,297,600 bytes,
which cross-checks the 110-byte block stride below.

Four tensors out of 866 cost the entire artifact, and the artifact is the one
the published Strix Halo rows ran ([#2497](https://github.com/mudler/vllm.cpp/issues/2497)
had to substitute the plain `Q4_K_M` because of this).

## Upstream anchors

Oracle: llama.cpp `b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`
([`oracles/llama-cpp.md`](../oracles/llama-cpp.md)). Neither vLLM nor
vLLM-Omni implements the ggml IQ3_S encoding, so the secondary-oracle rule
applies exactly as it did for IQ2_XS/IQ4_XS in
[`cuda-keepquant-iq2xs-iq4xs.md`](cuda-keepquant-iq2xs-iq4xs.md).

| Ported | From (at the pin) | To |
|---|---|---|
| `block_iq3_s` | `ggml/src/ggml-common.h:413-422` | `src/vt/cpu/cpu_quant_blocks.h::BlockIQ3_S`, `gguf_reader.cpp::FindGgmlTraits` case 21 |
| `dequantize_row_iq3_s` | `ggml/src/ggml-quants.c:2607` | `src/vt/cpu/cpu_quant_dequant.cpp::DequantIQ3_S` |
| `iq3s_grid` (512 u32) | `ggml/src/ggml-common.h:1052` | `src/vt/cpu/cpu_quant_iq_tables.h::kIq3sGrid`, `src/vt/cuda/cuda_quant_iq_tables.cuh::d_iq3s_grid` |
| `kmask_iq2xs` | `ggml/src/ggml-common.h` (already in tree) | `kKmaskIq2xs` — reused, not duplicated |
| `ggml_vec_dot_iq3_s_q8_K_generic` | `ggml/src/ggml-cpu/quants.c:1094` | **NOT PORTED — see `## Owed`** |
| `type_traits_cpu[GGML_TYPE_IQ3_S]` | `ggml/src/ggml-cpu/ggml-cpu.c:355-360` (`.vec_dot_type = GGML_TYPE_Q8_K`) | recorded here; the traits row is owed with the `vec_dot` |

**The block is 256 elements in 110 bytes, and the number came from the oracle's
own `sizeof`, not from reading the struct.** The harness described under
`## Evidence` prints `sizeof(block_iq3_s) = 110`, matching
`ggml_half d` (2) + `qs[QK_K/4]` (64) + `qh[QK_K/32]` (8) + `signs[QK_K/8]` (32)
+ `scales[IQ3S_N_SCALE]` (4). The `static_assert` at `ggml-common.h:422` states
the same sum as `sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE`.

**IQ3_S does NOT share IQ3_XXS's codebook.** IQ3_XXS reads `iq3xxs_grid`, a
256-entry u32 table indexed by a plain byte; IQ3_S reads `iq3s_grid`, a
**512-entry** u32 table indexed by a 9-bit value whose ninth bit comes from
`qh`. The two tables are the same element type and the same shape of lookup, so
a decoder written from the IQ3_XXS sibling still runs, still produces plausible
magnitudes, and is wrong. The families also differ in where the sign lives:
IQ3_XXS packs a 7-bit `ksigns_iq2xs` selector into the per-32 `aux32` word,
while IQ3_S carries a **direct sign byte** per lane in its own `signs[32]`
field, like IQ2_S. Nothing in the IQ3_XXS decode transfers.

The scale is also different in kind: IQ3_S has ONE nibble per two 32-element
sub-blocks (`IQ3S_N_SCALE` = 4 bytes for 8 sub-blocks) and the multiplier is
`db = d * (1 + 2*ls)` — an ODD-integer scale with no 0.5 offset and no 0.25
factor, unlike every IQ2 member in this tree.

## Design

Additive throughout; no existing path changes shape.

1. **Reader.** `FindGgmlTraits` case 21 returns `{256, 110, "IQ3_S"}`. This
   alone ends the refusal in #2510: `GgufFile::Open` can size the tensor and the
   file opens.
2. **vt dtype.** `DType::kIQ3_S` is appended to the enum *after* `kIQ4_XS` so no
   existing enumerator value moves, with the geometry row `{256, 110, 21}` in
   `src/vt/dtype.cpp`, the name `"iq3_s"`, a `BlockDTypeFromGgmlTypeId` entry,
   and the storage-only arms in `dtype.h::SizeOf` and `ops.cpp::ToScalarType`.
   Both of those switches are exhaustive with no `default:`, so the compiler
   names every site the new enumerator owes.
3. **Block layout.** `cpu_quant_blocks.h::BlockIQ3_S` with a
   `static_assert(sizeof(...) == 110)`.
4. **Codebook.** `kIq3sGrid[512]` in `cpu_quant_iq_tables.h`, mechanically
   extracted from `ggml-common.h:1052` at the pin.
5. **CPU row decoder.** `DequantIQ3_S`, a 1:1 port of `dequantize_row_iq3_s`,
   wired into `BlockToFloat`. This gives the expand-bf16 load path and the
   quantized *gather* (embedding) path.
6. **CUDA row-gather codec.** `cuda_quant_dequant.cuh::DqIQ3_S` plus the
   `d_iq3s_grid` table and the `VT_DQ_GATHER_TYPES` row. **This is not
   optional.** `KeepQuantGatherDType` admits any dtype `vt::cpu::BlockToFloat`
   answers for, and `DeviceQuantGatherSupported` asks only whether
   `kEmbeddingQuant` is registered on the device — it is dtype-blind. Adding a
   CPU decoder without the CUDA codec would make the loader admit a gather that
   `LaunchEmbeddingQuantTyped` answers with `cudaErrorInvalidValue`. The pinned
   set in `tests/vt/test_cuda_embedding_quant.cpp` exists to make exactly that
   omission red on a CPU-only build, and it does.
7. **ROCm refusal text.** `rocm_grouped_gemm.hip::MatmulBTQuantKernelRocm`
   enumerates the owed weight dtypes by name in its throw. IQ3_S joins that
   list, so the message keeps naming the missing part.

## Per-tier compute disposition

Required by the task and by `AGENTS.md` § "Shared seams". `kMatmulBTQuant`
admission for a GGUF weight is decided by
`gguf_keep_quant.cpp::KeepQuantDType`, which returns false unless
`vt::cpu::HasQuantDotKernel(dt)`.

| Tier | GEMM (`kMatmulBTQuant`) | Gather (`kEmbeddingQuant`) |
|---|---|---|
| CPU | **expands to bf16** — no `VecDotIQ3_SQ8_K`, so `KeepQuantDType` is false and `RouteGgufTensor` returns `kExpandBf16`. Owed. | **native** (`BlockToFloat`) |
| CUDA | **expands to bf16**, for the same reason: the loader never admits the block, so `IsCudaKeepQuantSupported` is never asked. Owed. | **native** (`DqIQ3_S`) |
| ROCm | **expands to bf16.** `DeviceKeepQuantSupported` lists Q8_0/Q4_K/Q5_K/Q6_K only; the throw in `MatmulBTQuantKernelRocm` names IQ3_S among the owed set. Owed. | expands — ROCm registers `kEmbedding` only, so `DeviceQuantGatherSupported` is false. Pre-existing, owed by `cuda-quant-gather.md`. |
| Metal | **expands to bf16.** Same shape as ROCm. Owed. | expands. Pre-existing. |
| Vulkan | **expands to bf16.** Same shape as ROCm. Owed. | expands. Pre-existing. |

**Expansion here is a route, not a silent one.** `RouteGgufTensor` is total,
`Name(GgufResidency)` reports `expand_bf16`, and the routing is asserted per
dtype in `tests/vllm/test_gguf_keep_quant.cpp`. This row adds a case that pins
IQ3_S to `kExpandBf16` *by name*, so the day a `vec_dot` lands, that test reds
and the tier table above has to move with it.

**The residency cost is bounded and measured.** On the staged
`Qwen3.8-27B-UD-Q4_K_M.gguf` the four IQ3_S tensors are 4 x 38,297,600 B =
146.13 MiB packed and 4 x 89,128,960 x 2 B = 680.00 MiB as bf16, a delta of
533.87 MiB on a 15.33 GiB file. Compare
[#1870](https://github.com/mudler/vllm.cpp/issues/1870): a silent expand is a
residency change and not only a speed one, which is why the delta is stated
rather than waved at. On this artifact it is 3.4 % of the file and does not
change what fits; on an artifact whose *experts* were IQ3_S it would, and that
is what the `## Owed` entry is for.

## Why the `vec_dot` is NOT in this change

Stated as a decision, not an omission.

`IsCudaKeepQuantSupported` has no refusal arm: a dtype it does not map falls
through to `cudaStreamSynchronize` plus the **CPU** kernel over the same
tensors. On unified memory that is correct and slow; on a discrete card
`Backend::Alloc` is a plain `cudaMalloc` and the host deref **segfaults** — the
failure `cuda-keepquant-iq2xs-iq4xs.md` measured on GB10. Landing
`VecDotIQ3_SQ8_K` alone would flip `KeepQuantDType(21)` to true, the loader
would start keeping IQ3_S blocks, and every CUDA IQ3_S GEMM would take that
fallback. **The CPU `vec_dot` and the CUDA `WType` arm are therefore one unit,
and this session cannot gate the CUDA half:** the host carries no CUDA toolkit
(`which nvcc` is empty) and no NVIDIA device (`nvidia-smi` is absent), so a CUDA
kernel written here would land uncompiled and ungated. Shipping an ungated
device kernel is worse than owing it. The pair is `## Owed` below with its own
issue.

The gather codec is not in that bind: it is dtype-blind at the device gate, its
omission is *already* red on the CPU lane through the pinned decoder set, and it
carries no fallback-to-host path at all.

## Risks

- **A wrong codebook still decodes.** IQ3_XXS's 256-entry `iq3xxs_grid` and
  IQ3_S's 512-entry `iq3s_grid` are both `uint32_t` tables read four bytes at a
  time. Swapping them produces finite, plausible values. Mitigated by gating
  bit-for-bit against oracle-produced goldens over real checkpoint bytes whose
  indices span 190 distinct values, 73 of them >= 256 — i.e. the `qh` ninth bit
  is actually exercised — rather than against "does not throw".
- **The ninth index bit is easy to drop.** `qh[0] << (8-2*l) & 256` for the even
  lane and `<< (7-2*l)` for the odd one is an asymmetric shift pair. A decoder
  that uses one shift for both still decodes 439 of 512 grid slots correctly on
  average. Same mitigation; the golden set's 73 high-bit indices are what kills
  it.
- **Enum append moves a loop bound.** `test_cuda_embedding_quant.cpp` iterates
  `i <= static_cast<int>(DType::kIQ4_XS)`. Appending `kIQ3_S` after it makes
  that loop stop one short, which would let a MISSING decoder read as present.
  The bound moves to `kIQ3_S` in the same change and the count assertion is what
  catches it.
- **Scale polarity.** IQ3_S's `db = d * (1 + 2*ls)` has neither the `0.5 +` of
  IQ2_S nor its `* 0.25`. A copied IQ2_S scale line yields values off by a
  bounded factor that a correlation check would not see. The golden comparison
  is exact equality on f32 bit patterns, so it does.

## Tests

Red-first, in this order.

1. `tests/vllm/test_gguf_dequant.cpp` — `GgufFile` opens a file carrying a real
   IQ3_S tensor and the reader reports `ggml_type == 21` and
   `nbytes == 4 * 110`. **Red before the reader case**, with the exact
   `unknown ggml type id 21` text from #2510.
2. `tests/vllm/test_gguf_dequant.cpp` — `DequantGgufRowToF32(21, ...)` over the
   four real blocks equals the oracle's 1024 f32 values **bit for bit**, through
   `CheckGgufDequantAgainstOracle`, the same helper the IQ2_XS/IQ4_XS goldens
   use.
3. `tests/vt/test_ops_quant_traits.cpp` — the reader trait and the vt geometry
   are pinned against each other for id 21 (256 elems, 110 bytes, name
   `iq3_s`), and the decode-only class is re-stated with IQ3_S named in it.
   `tests/vt/test_ops_quant_dot.cpp` seals `kIq3sGrid` by FNV-1a 64 digest and
   by its lane alphabet, which is the only thing separating a 512-entry u32
   grid from the 256-entry u32 `kIq3xxsGrid` beside it; IQ3_S has no `vec_dot`
   and therefore no golden dot case, so the seal carries that weight alone.
4. `tests/vt/test_cuda_embedding_quant.cpp` — the pinned CPU row-decoder set
   grows to 19 and lists `kIQ3_S`; the loop bound moves to `kIQ3_S`. This is
   the case that forces the CUDA `VT_DQ_GATHER_TYPES` row.
5. `tests/vllm/test_gguf_keep_quant.cpp` — IQ3_S routes to `kExpandBf16` on a
   weight role *by name*, so the owed `vec_dot` cannot land without this test
   moving.

The upstream tests for this encoding are `ggml/tests/test-quantize-fns.cpp`'s
generic round-trip sweep, which is a *quantizer* gate (`quantize_row_iq3_s`
needs `iq3xs_init_impl`'s neighbour tables) and not a decoder gate. This change
ports no quantizer, so there is no upstream test to preserve; the oracle's own
`dequantize_row_iq3_s` output IS the ported expectation, which is the stronger
of the two.

## Gates

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DVLLM_ENABLE_CUDA=OFF
cmake --build build -j 4 --target vllm_tests_vllm vllm_tests_vt
./build/tests/vllm_tests_vllm -tc='*IQ3_S*'
./build/tests/vllm_tests_vt   -tc='*decoder set*'
scripts/agent-preflight.sh
```

## Evidence

Goldens produced by the ORACLE, from the pin, over REAL checkpoint bytes.
Provenance is carried in `tests/vt/iq3s_golden_vectors.h` and reproduced by:

```sh
git -C <llama.cpp> archive 10bf611e533d81f739128304991c5e133c6aebd8 ggml | tar -x -C $W
gcc -O2 -DGGML_VERSION='"b10451"' -DGGML_COMMIT='"10bf611e5"' \
    -I $W/ggml/include -I $W/ggml/src -I $W/ggml/src/ggml-cpu \
    -o harness harness.c stubs.c $W/ggml/src/ggml-quants.c $W/ggml/src/ggml.c -lm
```

`stubs.c` supplies the five backend symbols `ggml.c` references and a
dequant-only harness never calls; each aborts, so a stub cannot quietly
contribute to a golden value. The harness prints the oracle's own
`sizeof(block_iq3_s)`, which is where the 110 in the reader trait comes from.

**The CUDA gather codec is checked, and it is checked HONESTLY.** This host
carries no CUDA toolkit and no NVIDIA device, so `DqIQ3_S` cannot be compiled by
`nvcc` here and no CUDA gate is claimed. What IS measured is a HOST
TRANSLITERATION: the `struct DqIQ3_S` body is extracted verbatim from
`cuda_quant_dequant.cuh`, the CUDA qualifiers and the two device symbols it
reads (`d_iq3s_grid`, `d_kmask_iq2xs`) are re-declared for the host, and the
result is run over the same four golden blocks. It agrees with the oracle
**1024 of 1024 values, bit-exactly**. That is a logic and syntax check on the
body, not a verdict on the device arm; the device verdict is owed and named as
such.

Inputs: the first 440 bytes (four whole blocks) of `blk.11.ffn_gate.weight` at
absolute offset 4,262,628,128 in
`/mnt/nas_share/rc/ckpt/qwen38-27b-ud-q4km/Qwen3.8-27B-UD-Q4_K_M.gguf`
(sha256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`) —
the exact tensor `GgufFile::Open` refuses today.

## Result

Measured on this branch, x86_64, Debug, `-DVLLM_ENABLE_CUDA=OFF`, clean build
with zero warnings.

**RED, before the reader case, for the intended reason:**

```text
TEST CASE:  DequantGgufRowToF32 IQ3_S row matches the pinned oracle
  ERROR: test case THREW exception: gguf: unknown ggml type id 21
TEST CASE:  GgufFile reads an IQ3_S tensor and it dequants
  ERROR: test case THREW exception: gguf: tensor "blk.11.ffn_gate.weight"
         has unknown ggml type id 21 in /tmp/vllm_gguf_test_...gguf
[doctest] test cases: 2 | 0 passed | 2 failed | 25 skipped
[doctest] assertions: 0 | 0 passed | 0 failed |
```

That second line is the message #2510 reports verbatim.

**The same red and green ON THE REAL ARTIFACT, not only on a synthetic block.**
A synthetic fixture proves the decoder; it cannot prove the file the issue names
now opens. `examples/dump_container` calls `vllm::GgufFile::Open` -- the exact
function that threw -- and prints one line per tensor. Run against
`/mnt/nas_share/rc/ckpt/qwen38-27b-ud-q4km/Qwen3.8-27B-UD-Q4_K_M.gguf`
(16,464,440,224 B, sha256 `322e194f...123482`), reading only the header and a
64 KiB prefix per tensor through the mmap, never copying the 16 GB:

```text
# base reader (case 21 reverted, one TU rebuilt, same binary otherwise)
$ dump_container .../Qwen3.8-27B-UD-Q4_K_M.gguf
dump_container: gguf: tensor "blk.11.ffn_gate.weight" has unknown ggml type id 21
  in /mnt/nas_share/rc/ckpt/qwen38-27b-ud-q4km/Qwen3.8-27B-UD-Q4_K_M.gguf
EXIT=1

# this branch
$ dump_container .../Qwen3.8-27B-UD-Q4_K_M.gguf
blk.11.ffn_gate.weight IQ3_S [17408,5120] 38297600 f73a86bd...
blk.14.ffn_down.weight IQ3_S [5120,17408] 38297600 4a482429...
blk.15.ffn_down.weight IQ3_S [5120,17408] 38297600 759b5656...
blk.17.ffn_down.weight IQ3_S [5120,17408] 38297600 66c38045...
TOTAL 866 tensors
EXIT=0
```

The reader was restored byte-for-byte afterwards and the manifest re-run is
byte-identical to the green above.

**The manifest agrees with an INDEPENDENT census.** A standalone Python GGUF
header parser (not this reader) counts the same 866 tensors and the same
per-type totals, so the C++ side is checked against something that shares no
code with it:

| ggml id | type | count |
|---|---|---|
| 0 | F32 | 360 |
| 13 | Q5_K | 131 |
| 23 | IQ4_XS | 117 |
| 8 | Q8_0 | 106 |
| 12 | Q4_K | 104 |
| 14 | Q6_K | 30 |
| 11 | Q3_K | 7 |
| 20 | IQ4_NL | 7 |
| **21** | **IQ3_S** | **4** |

862 of the 866 were already handled; the 4 that were not refused the other 862.
Both sides also agree on 38,297,600 B per IQ3_S tensor, which is
`89,128,960 / 256 * 110` -- the block geometry this change adds, arrived at from
the file's own byte layout rather than from the struct.

**GREEN, after:**

```text
[doctest] test cases:    2 |    2 passed | 0 failed | 25 skipped
[doctest] assertions: 2052 | 2052 passed | 0 failed |
```

Every affected suite, whole:

| Suite | Cases | Assertions |
|---|---|---|
| `test_gguf_dequant` | 27/27 | 9,452 |
| `test_gguf` | 36/36 | 133 |
| `test_ops_quant_traits` | 10/10 | 6,220 |
| `test_ops_quant_dot` | 33/33 | 253,314 |
| `test_gguf_keep_quant` | 46/46 | 10,301 |
| `test_glm5_next_bridge` | 20/20 | 32,563 |
| `test_cuda_embedding_quant` | exit 77 SKIPPED | no CUDA backend on this host, and it SAYS so rather than printing a 0-assertion pass |

**Mutations.** Each was applied, built, measured, and the tree restored and
re-verified by `md5sum` and by a re-run.

| # | Mutation | Result |
|---|---|---|
| M1 | `kIq3sGrid` -> `kIq3xxsGrid` in `DequantIQ3_S` | 2048 of 2052 assertions FAIL |
| M2 | odd-lane shift `(7 - 2*l)` -> `(8 - 2*l)`, dropping the ninth-bit asymmetry | 358 of 2052 FAIL |
| M3 | IQ2-family scale `d*(0.5+ls)*0.25` in place of `d*(1+2*ls)` | 2048 of 2052 FAIL |
| M4 | REACHABILITY: delete `FindGgmlTraits` case 21 | both cases THROW `unknown ggml type id 21` |
| M5 | give IQ3_S a `vec_dot` row so `HasQuantDotKernel` flips true | `test_ops_quant_traits` 1 case / 2 assertions FAIL, `test_gguf_keep_quant` 1 case / 8 assertions FAIL |

M2 is the one worth reading twice. A symmetric shift pair leaves 1694 of 2052
values correct, so it is exactly the defect a spot check or a tolerance would
pass; only the bit-exact comparison over a golden set carrying 73 high-bit
indices catches it. M4 is the reachability proof this row owes: the smallest
failing test enters through `GgufFile::Open`, the production entry point, and
deleting that call site reds it. M5 is the proof that the expand-bf16
disposition in `## Per-tier compute disposition` is asserted rather than merely
described.

The staged artifact's SHA-256 was re-derived locally over all 16,464,440,224
bytes and equals
`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`, the value
#2510 reports.

**`scripts/agent-preflight.sh --staged`:** 34 of 34 record gates `ok`,
`tree-compiles` `ok` at 808 of 808 translation units in scope, `commit-trailers`
and `commit-style` `ok`, 5 gates SKIPPED (`check-arm-isa-build`,
`check-cpu-isa-build`, `check-cuda-fat-gencode`, `check-pr-size`,
`check-triton-aot-multiarch` — each a pre-existing arch or argument skip this
change neither causes nor repairs, and the run therefore prints `NOT a green
preflight`).

**One gate failed, and it is the host and not this change:
`test_cpu_x86_llamacpp_floor`.** It is a benchmark HARNESS test that refuses to
measure a contended box, and it printed its own reason: `NO_QUIET_WINDOW after
30s (busy=127% builders=0 load=48.06 67.13 55.68)`. Three facts pin the
attribution rather than assert it.

1. The file is byte-identical to `6bf3abb58`'s
   (`md5 7c7cc2fefc8242eb96af9cd737ba75ba` on both sides), it drives
   `scripts/cpu-x86-llamacpp-floor.sh`, and `git diff 6bf3abb58..HEAD --
   scripts/ tests/scripts/` is EMPTY. Nothing in this change is reachable from
   it.
2. **The failing test MOVED between runs.** At load 43 the failure was
   `test_a_contended_leg_is_discarded_and_never_summarised` (rc 4 where 2 was
   expected); at load 48 it was
   `test_g5_load_is_recorded_before_and_after_every_leg` (rc 4 where 0 was
   expected). A defect does not migrate across cases with the load average.
3. **It passes on the identical tree once the box quiets:** `Ran 10 tests in
   4.827s ... OK`, exit 0, at load 14.17 — against 263.674 s and a failure at
   load 48. The 55x wall-clock difference IS the contention the harness is
   refusing to measure through.

No other gate failed.

## Owed

- **`VecDotIQ3_SQ8_K` (CPU) + `WType::kIQ3_S` (CUDA), as ONE unit.** Until both
  land, IQ3_S takes `kExpandBf16` on the GEMM path on every tier. Reason for the
  coupling in `## Why the vec_dot is NOT in this change`. Tracked by
  [#2510](https://github.com/mudler/vllm.cpp/issues/2510), which stays open for
  it; this change closes only the refusal it names.
- **ROCm / Metal / Vulkan native IQ3_S GEMM.** Those tiers implement no IQ
  encoding at all (`DeviceKeepQuantSupported` lists Q8_0/Q4_K/Q5_K/Q6_K for
  ROCm), so IQ3_S joins the standing owed set rather than adding a new one.
  Owned by [`rocm-gg-keep-quant.md`](rocm-gg-keep-quant.md).
- **ROCm / Metal / Vulkan quantized gather.** Pre-existing; those backends
  register `kEmbedding` only. Owned by
  [`cuda-quant-gather.md`](cuda-quant-gather.md).
- **A decode gate on the artifact.** This change makes
  `Qwen3.8-27B-UD-Q4_K_M.gguf` openable. It does not measure a token against
  llama.cpp on it; that is #2497's number and it stays owed there.

## Stop conditions

- Stop and report `NEEDS_DECISION` if the golden decode does not match the
  oracle bit for bit. A tolerance is not the repair; the two sides are the same
  f32 expression, so any difference means a decode parameter diverged.
- Stop if landing the reader case alone would let a tensor reach a device path
  that cannot execute it. It does not — `KeepQuantDType` gates on
  `HasQuantDotKernel`, which stays false — and the `## Tests` entry 5 pins that.
- Do not extend scope to the other unimplemented ggml ids. Id 21 is the one the
  measured artifact needs.
