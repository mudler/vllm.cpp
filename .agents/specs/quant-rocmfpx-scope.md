# QUANT-ROCMFPX-SCOPE — scoping ROCmFPX, and the decision not to port it yet

Issue: [#2463](https://github.com/mudler/vllm.cpp/issues/2463).
Wave: one scoping wave. No product code. The deliverable is this decision and
the trigger that would reverse it.

**Verdict: DO NOT port ROCmFPX now, and DO NOT admit it to the AGENTS.md
secondary-oracle table.** The blocker is not difficulty. The 4-bit encoding is
small, stable and cleanly specified, and the model it ships is one this tree
already loads. The blocker is that nothing on either side of the comparison can
currently be gated: ROCmFPX has no measured build or run here, the artifacts'
own metadata is internally inconsistent, and our ROCm arm has no FP4 compute
kernel of any kind to attach a number to. §11 names the five reasons and the four triggers.

## 1. What ROCmFPX is

**It is a llama.cpp fork, not a new stack**, notwithstanding the repository
description ("This will be the official ROCmFPX stack and studio"). The tree at
`132f08f2a6c5` carries llama.cpp's `AUTHORS`, `ggml/`, `gguf-py/`,
`convert_hf_to_gguf.py` and `include/llama.h`, and merges llama.cpp master
continuously — `f23cce50` (2026-08-31) is "Merge llama.cpp master at 8e53fcefd
into ROCmFPX".

The format is **a set of new ggml type ids in an otherwise ordinary GGUF v3
container**. It is not a repacking of existing k-quants and not a separate
container. `ggml/include/ggml.h:435-446` at `132f08f2a6c5`:

```c
// Downstream ROCmFPX formats live in a reserved range so new upstream
// GGML types can continue to use the compact, append-only sequence.
GGML_TYPE_Q4_0_ROCMFP4      = 100, // dual UE4M3 scales + packed AMD FP4 blocks
GGML_TYPE_Q4_0_ROCMFP4_FAST = 101, // single-scale speed layout
GGML_TYPE_Q6_0_ROCMFPX      = 102, // 6-bit UE4M3-scale layout
GGML_TYPE_Q8_0_ROCMFPX      = 103, // 8-bit UE4M3-scale layout
GGML_TYPE_Q3_0_ROCMFPX      = 104, // 3-bit UE4M3-scale layout
GGML_TYPE_TURBO3_0          = 105, // TurboQuant 3-bit KV cache
GGML_TYPE_TURBO4_0          = 106, // TurboQuant 4-bit KV cache
GGML_TYPE_Q2_0_ROCMFPX      = 107, // 2-bit S40 codebook + dual UE4M3 scales
GGML_TYPE_Q4_0_ROCMI4       = 108, // exact signed-nibble 4-bit + UE4M3 scale
GGML_TYPE_Q5_0_ROCMFPX      = 109, // 5-bit signed linear + dual UE4M3 scales
GGML_TYPE_Q7_0_ROCMFPX      = 110, // 7-bit signed linear + dual UE4M3 scales
GGML_TYPE_COUNT             = 111,
```

The two 4-bit block layouts, `ggml/rocmfp4/rocmfp4.h:13-39`:

```c
#define QK_ROCMFP4 32
typedef struct { uint8_t qs[QK_ROCMFP4/2]; uint8_t e[2]; } block_rocmfp4;      // 18 B / 32 w = 4.50 bpw
typedef struct { uint8_t qs[QK_ROCMFP4/2]; uint8_t e;    } block_rocmfp4_fast; // 17 B / 32 w = 4.25 bpw
```

Both are 32 weights of packed E2M1-derived 4-bit codes; `block_rocmfp4` carries
one unsigned E4M3 scale per 16-weight half, `block_rocmfp4_fast` one for the
whole block. The wider family in `ggml/rocmfpx/rocmfpx.h:53-98` follows the
same shape — `qs[QS_ROCMFPn]` plus one or two UE4M3 scale bytes, always over
`QK_ROCMFPX = 32`.

### 1.1 Two enumerations share one numeric range and do not mean the same thing

`ggml.h`'s `GGML_TYPE_*` (above) and `llama.h`'s `LLAMA_FTYPE_*`
(`include/llama.h:162-183` at the pin) both allocate from 100, and they collide:

| value | as a **ggml type id** (a block layout) | as an **ftype** (a whole-file recipe) |
|---|---|---|
| 100 | `Q4_0_ROCMFP4` | `Q4_0_ROCMFP4` |
| 101 | `Q4_0_ROCMFP4_FAST` | `Q4_0_ROCMFP4_LEAN` |
| 102 | `Q6_0_ROCMFPX` (6-bit) | `Q4_0_ROCMFP4_COHERENT` (4-bit) |
| 103 | `Q8_0_ROCMFPX` (8-bit) | `Q4_0_ROCMFP4_FAST` (4-bit) |
| 106 | `TURBO4_0` (KV cache) | `Q4_0_ROCMFP4_STRIX_LEAN` (weights) |

A reader that takes `general.file_type` for a ggml type id, or the reverse, gets
a plausible wrong answer at every one of those values rather than an error. Any
port must keep the two enumerations strictly apart. This tree already reads only
tensor-info type ids and never dispatches on `general.file_type`, so it is
exposed to the confusion only if a port introduces it.

## 2. `FAST`, `COHERENT` and `STRIX_LEAN` are recipes, not formats

This is the single most load-bearing finding, and it dissolves the apparent
naming disagreement between the two producers.

There are **two** 4-bit block formats (ggml ids 100 and 101). `FAST`,
`COHERENT`, `LEAN` and `STRIX_LEAN` are *ftypes* — per-tensor assignment
recipes over those two formats plus a k-quant embedding. The census (§3) shows
each recipe's composition directly, and the fork's own ftype comments at
`include/llama.h:159-164` agree with it independently:

| ftype | the fork's own description | what the artifact actually contains |
|---|---|---|
| 102 `COHERENT` | "ROCmFP4 with Q6_K token embeddings" | 504 x id 100, 1 x `Q6_K`, 1 x `Q8_0` |
| 103 `FAST` | "ROCmFP4 single-scale speed layout" | 505 x id 101, 1 x `Q8_0` |
| 106 `STRIX_LEAN` | "Strix Halo size-biased K/V recipe" | 82 x id 100 + 422 x id 101, 1 x `Q5_K`, 1 x `Q8_0` |

**Those descriptions are read at `bdceeefc` (2026-08-22), NOT at the pin.** At
`132f08f2a6c5` the per-ftype trailing comments have been stripped and
`include/llama.h:162-183` carries bare values. The values themselves are
identical at both revisions (§7), so the mapping stands; only the prose that
explains it lives at the earlier object. Citing `llama.h:159-164` against the pin
would name lines that do not say this.

So a port that implements ggml ids 100 and 101 reads **every** 4-bit recipe,
present and future, because a new recipe is a new assignment of existing blocks.
Pricing the work per recipe name would have overcounted it by a factor of three.

## 3. Header census — nine artifacts, by HTTP range request

Method: a `Range: bytes=0-12582911` request per file (12 MiB, enough for all of
them; total cost 108 MiB against the 146.5 GiB these nine files hold), parsed with a
from-scratch GGUF header reader. No artifact was downloaded.

Pins: `julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF` @
`1e67b67e0e23324a29a7a4449279685fac364b37`;
`rcmorano/Qwen3.8-27B-ROCMFPX` @ `11b8dc8a8f569f96724ab1d77eedac30b2fbf335`.

Every file is GGUF v3, architecture `qwen35`, **866 tensors**, 360 of them F32.

| producer | file | ftype | ggml type ids present | we enumerate |
|---|---|---|---|---|
| julianmb | `ROCmFP4-FAST` | 103 | `F32` x360, `Q8_0` x1, **101** x505 | no — 101 |
| julianmb | `ROCmFP4-STRIX_LEAN` | 106 | `F32` x360, `Q8_0` x1, `Q5_K` x1, **100** x82, **101** x422 | no — 100, 101 |
| julianmb | `ROCmFP2` | 119 | `F32` x360, `Q6_K` x1, **107** x505 | no — 107 |
| julianmb | `ROCmFP8` | 111 | `F32` x360, **103** x506 | no — 103 |
| rcmorano | `Q4_0_ROCMFP4_COHERENT` | 102 | `F32` x360, `Q8_0` x1, `Q6_K` x1, **100** x504 | no — 100 |
| rcmorano | `Q4_0_ROCMFP4_STRIX_LEAN` | 106 | `F32` x360, `Q8_0` x1, `Q5_K` x1, **100** x82, **101** x422 | no — 100, 101 |
| rcmorano | `Q2_0_ROCMFPX` | 119 | `F32` x360, `Q6_K` x1, **107** x505 | no — 107 |
| rcmorano | `Q6_0_ROCMFPX` | 110 | `F32` x360, **102** x486, **103** x20 | no — 102, 103 |
| rcmorano | `Q8_0_ROCMFPX` | 111 | `F32` x360, **103** x506 | no — 103 |

**Four distinct ROCmFPX ids appear across the nine files: 100, 101, 102, 103,
107.** Ids 104, 105, 106, 108, 109 and 110 are declared by the fork and shipped
by nobody. Nothing is unexplained: every id in every file is either an upstream
id this tree already reads or a ROCmFPX id named in `ggml.h`.

**Every one of the nine refuses at OPEN.** `FindGgmlTraits`
(`gguf_reader.cpp:200-368`) enumerates ids 0, 1, 2, 6, 8, 10-14, 16-20, 22-28,
30, 39, 40, 41 and 66. The tensor-info loop at `gguf_reader.cpp:483-485` fails
the file the first time it meets anything else, before any weight byte is read.

### 3.1 The census reproduces every file size exactly

Not asserted — computed. Summing `(numel / block_elems) * block_bytes` over all
866 tensors, using the block geometry read out of the ROCmFPX headers, and
adding the parsed header length, lands **28 bytes** under the published size of
every one of the nine files. 28 is exactly the GGUF alignment pad
(`header_end = 10995652`, `10995652 mod 32 = 4`, `32 - 4 = 28`).

This is the check that makes the rest of this document load-bearing. It proves
the block layouts in §1 are the layouts these artifacts actually use, rather
than a header we read and hoped applied.

## 4. The two producers ship the same encoding — measured, with one caveat

Comparing `julianmb/ROCmFP4-STRIX_LEAN` against
`rcmorano/Q4_0_ROCMFP4_STRIX_LEAN`, which differ in published size by exactly
256 bytes:

| checked | result |
|---|---|
| tensor names | **identical sets**, 866 for 866 |
| name -> dims | identical for all 866 |
| name -> ggml type id | identical for all 866 |
| tensor data offsets | identical for all 866 |
| `general.file_type` | 106 on both |
| KV keys | julianmb carries 4 more: `quantize.imatrix.{file,dataset,entries_count,chunks_count}` |
| every shared KV value | byte-equal |

The 256-byte delta is fully accounted for: 248 bytes of extra KV plus an 8-byte
shift in the alignment pad. The same holds for `ROCmFP2` vs `Q2_0_ROCMFPX` and
`ROCmFP8` vs `Q8_0_ROCMFPX`, which also differ by exactly 256 bytes and carry
identical type histograms.

**So the naming disagreement is a filename disagreement only.** rcmorano names
files after the ggml type; julianmb after the ftype's marketing label. At the
level a decoder cares about — type ids, block geometry, tensor layout — the two
producers agree exactly, and both use the fork's own ftype numbers correctly.

**The caveat, which matters: the same encoding is not the same bytes.** Reading
256 KiB of real tensor data from three tensors on each side by range request,
the two files disagree on all three:

| tensor | julianmb sha256 (first 24) | rcmorano sha256 (first 24) |
|---|---|---|
| `output.weight` | `ed646f471ae9ba7f6b95caab` | `20186221b025dd8317784d42` |
| `blk.0.ffn_down.weight` | `c2870fa14aa6463b04105386` | `ebdf4e2348df6495119a44ff` |
| `blk.32.ffn_gate.weight` | `f700788c629dc627d0a09760` | `bde22c9c270738639983714e` |

Control, because a byte comparison at a wrong offset differs for free: the F32
`blk.0.attn_norm.weight` read at the same computed offsets is **identical on
both sides** (5120/5120 finite, min 0.8682, max 1.1982, mean 0.9666 on each).
The offsets are right, the unquantized tensors match, and the quantized ones do
not.

The imatrix KV difference is the obvious candidate explanation and is **not
established** — those KVs name Unsloth's imatrix, inherited from the base GGUF,
so they may record provenance rather than an applied calibration. What is
measured is that two files with identical structure hold different values.
Anything gated against "ROCmFPX STRIX_LEAN" must name the producer and revision,
not the recipe.

## 5. This is a checkpoint we already carry a manifest for

`tests/vllm/models/qwen38_27b_q4km_gguf_manifest.inc` freezes a real
`unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10`: GGUF
v3, **866 tensors**, 51 KV, architecture `qwen35`, vocab 248320. Compared against
the census:

| artifact | names equal to our manifest | dims differing | ggml type differing |
|---|---|---|---|
| `jm/ROCmFP4-STRIX_LEAN` | **yes**, 866/866 | 0 | 505 of 866 |
| `jm/ROCmFP4-FAST` | **yes**, 866/866 | 0 | 505 of 866 |
| `rc/Q8_0_ROCMFPX` | **yes**, 866/866 | 0 | 506 of 866 |

Same model, same tensor names, same shapes, same MTP drafter at `blk.64.nextn.*`
with `qwen35.block_count = 65` and `qwen35.nextn_predict_layers = 1`. The **only**
difference is the block format of the quantized tensors.

This cuts the cost estimate hard and is the strongest argument in ROCmFPX's
favour: a port needs **no model work at all**. It is purely a quantization-format
port onto a model arm that already exists. `docs/USAGE.md:623` records that arm
as real — "Q4_K_M text model loads through `--model` and decodes on CPU", against
that same `unsloth` revision — and `qwen35` is registered at
`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:918`.

**But the same line records that the arm is not gated.** The Q4_K_M token gate
against llama.cpp `b10451` **FAILED** on 2026-08-23: tokenizer exact 6/6,
generation divergent 5/6
([evidence](../../docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md),
[#821](https://github.com/mudler/vllm.cpp/issues/821)). So the existing arm that
makes a ROCmFPX port cheap is itself an arm whose decode does not yet match its
oracle. Adding a second quantization to a model whose first one diverges would
build a new format on a foundation that cannot currently tell a format bug from
the divergence it already has. This is reason (5) in §11.

## 6. vLLM at the pin implements nothing comparable

Checked at `5559679229`, per AGENTS.md §"vLLM is the reference".

**vLLM has no in-tree GGUF loader at all.** `git ls-files | grep -i gguf`
returns four paths: one doc and three plugin tests. There is no `gguf.py` under
`vllm/model_executor/layers/quantization/`, and nothing GGUF under
`vllm/model_executor/model_loader/`. `docs/features/quantization/gguf.md` states
it: "GGUF support has migrated to OOT
[vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)".

vLLM does carry `quantization/mxfp4.py`, `utils/mxfp4_utils.py`,
`quark/schemes/quark_ocp_mx.py` and a `turboquant/` package, so 4-bit
OCP-MX-family quantization is in scope for vLLM — but as *safetensors* schemes,
never as GGUF block types, and none of them is ROCmFP4.

`git grep -in 'rocmfp4|rocmfpx|rocmi4'` at the pin returns **rc=1 and zero
bytes**; the control `git grep -il mxfp4` returns rc=0 over 157 files, so the
absence is a real absence and not a broken grep. One near-miss is worth naming
so the next reader does not re-find it: a case-insensitive search for `rocmfp`
DOES hit five files, all of them
`ROCmFP8ScaledMMLinearKernel` (`vllm/model_executor/kernels/linear/scaled_mm/rocm.py:74`).
That is vLLM's ROCm **FP8 scaled-MM** kernel — a substring collision with
"ROCmFP", unrelated to ROCmFPX. vLLM therefore has a ROCm FP8 compute path and
no ROCm FP4 one, which mirrors our own arm's shape.

**Conclusion:** vLLM implements nothing here, so §"When vLLM has no
implementation" is genuinely engaged and a secondary oracle would be admissible
in principle. §10 explains why it is still refused in practice.

## 7. Stability: the 4-bit core is stable, the family is not

The brief's premise was that a repository last pushed today is a moving target.
Measured against the commit history rather than the push date, that is half
right, and the half that is right is not the half that matters most.

| path | commits touching it | most recent |
|---|---|---|
| `ggml/rocmfp4/rocmfp4.h` (ids 100, 101) | **1** | **2026-05-23** `22c4f745` "Add ROCmFP4 quantization support" |
| `ggml/rocmfpx/rocmfpx.h` (ids 102-110) | 4 | 2026-08-29 `a85e8601` "Merge llama.cpp upstream and **establish ROCmFPX formats**" |

**The fork states the guarantee itself.** `include/llama.h:160-161` at the pin:

```c
// ROCmFPX downstream formats. These IDs are part of the on-disk GGUF
// contract and must remain stable across upstream synchronizations.
```

A stated intention is not a measurement, so it is the history below that carries
the finding — but the two agree, and a project that writes that sentence down is
one that understands what it would cost to break it.

**The 4-bit encoding has not changed in over three months.** Nor has its ftype
numbering: `LLAMA_FTYPE_MOSTLY_Q4_0_ROCMFP4*` = 100-106 is byte-identical at
`22c4f745` (2026-05-23), `bdceeefc` (2026-08-22) and the pin (2026-09-01).

The wider family is a different story. Ids 109 and 110 were **added 3 days
before the pin** (`a85e8601`), id 108 **10 days before** (`bdceeefc`). Both
changes were purely additive to the block layouts — no existing struct or
static_assert was altered — but a family still gaining members weekly is not one
to port wholesale.

There was one real numbering inconsistency, and it is worth recording because it
is the shape of defect a port would inherit. The fork duplicates its ftype
enumeration in `ggml.h` (`GGML_FTYPE_*`) and `llama.h` (`LLAMA_FTYPE_*`). At
`bdceeefc` those two disagreed: `Q2_0_ROCMFPX` was 113 in `ggml.h` and 119 in
`llama.h`. `a85e8601` renumbered `ggml.h` to match. **The value written to files
is the `llama.h` one, which never moved**, so no shipped artifact is affected —
both `ROCmFP2` files carry `general.file_type = 119` and are correct. But the
fork carried two contradictory copies of one table for seven days, which is
exactly what AGENTS.md means by not hand-transcribing a table.

Correctness work on the ROCmFPX *kernels* is also still landing daily —
`725c725d` "hip: fix two-scale ROCmFPX MMQ dispatch" and `e3cee826` "vulkan: fix
ROCmFPX MUL_MAT_ID geometry", both 2026-08-31, both within 24 hours of the pin.
The encoding is settled; the implementations that produce and consume it are not.

## 8. The artifacts' own metadata does not survive checking

Three independent problems, all measured, all bearing on whether these files can
serve as a gate denominator.

**`julianmb/config.json` disagrees with its own artifact on 6 of 8 checkable
fields.**

| field | `config.json` | the GGUF | |
|---|---|---|---|
| architecture | `qwen3` | `qwen35` | disagree |
| `vocab_size` | 152064 | 248320 | disagree |
| `hidden_size` | 5120 | 5120 | agree |
| `intermediate_size` | 27648 | 17408 | disagree |
| `num_hidden_layers` | 64 | 65 | disagree |
| `num_attention_heads` | 40 | 24 | disagree |
| `num_key_value_heads` | 8 | 4 | disagree |
| `max_position_embeddings` | 262144 | 262144 | agree |

That file also carries the repository's headline claims —
`"bits_per_weight": 4.26`, `"cooperative_matrix_target": "gfx1151"`,
`"quant_method": "rocmfpx"`. Those claims sit in a document whose every
checkable architectural number is wrong. The `gfx1151` one happens to be
correct (§9); that is not a reason to trust the others.

**`params.json` misreports the file size** it describes: 14,548,483,584 for
`ROCmFP4-FAST` against an actual 14,562,236,384, off by 13.75 MB.

**The provenance metadata still credits the wrong party.** Every ROCmFPX
artifact carries `general.quantized_by = Unsloth` and
`general.repo_url = https://huggingface.co/unsloth`. The requantization to
ROCmFPX did not update either. A file that names its quantizer incorrectly
cannot be cited as evidence of who produced it.

`SHA256SUMS` is present in the julianmb repo and covers all six GGUFs. It was
**not** verified here, because verifying it requires downloading the artifacts,
which this wave's constraints forbid. It is recorded as the better-than-`lfs.oid`
source it is, and as unchecked.

## 9. The target box, measured

`strix:gpu0`, via one `rc run` lease (job `8e5fd317`, 2026-09-01T08:20:09Z).
Never by `ssh`. `/workspace` guarded before use and reported 220 entries.

| | |
|---|---|
| offload arch | **`gfx1151`** |
| ROCm | **7.2.4** |
| Vulkan | RADV `GFX1151`, API 1.4.318 |
| memory | 62 GiB unified, 57 GiB free |
| kernel | 6.14.0-36-generic, Ubuntu 24.04 container |

`rocminfo` and `rocm-smi` are absent inside the job container; `offload-arch` is
the authoritative reading.

**`gfx1151`, not `gfx1200`.** Any number measured here must say so.
`.agents/specs/rocm-gfx1200-m2-correctness.md` names a different target and its
results do not transfer. 62 GiB comfortably fits the 14.8 GiB `STRIX_LEAN` and
the 28.2 GiB `ROCmFP8` artifacts, so capacity is not a constraint on either arm.

## 10. What our tree would need, priced

Cost measured by counting the sites the last block dtype added here (`MXFP4`)
occupies: **14 files** name `kMXFP4`, 64 mention `MXFP4` at all. That is the
per-dtype price, and ROCmFP4 needs two dtypes to read every 4-bit recipe.

| # | work | where | size | note |
|---|---|---|---|---|
| 1 | reader traits for ids 100, 101 | `gguf_reader.cpp:200-368` | **XS** | two `case` arms, `{32,18}` and `{32,17}` |
| 2 | `vt::DType::kROCMFP4`, `kROCMFP4_FAST` | `include/vt/dtype.h:95-121` | **M** | enums are exhaustive with no `default:` by design, so this is a `-Wswitch` error at every site until all ~14 are updated. That is the seam working, and it is the bulk of the cost |
| 3 | block structs + `static_assert` | `src/vt/cpu/cpu_quant_blocks.h` | **XS** | 18 B and 17 B over 32 weights |
| 4 | UE4M3 scale + E2M1 code decode | new | **S** | `rocmfpx_ue4m3_to_fp32` is a 256-entry mapping; **generate it, never transcribe it** |
| 5 | `BlockToFloat` decoder | `cpu_quant_dequant.cpp`, `gguf_dequant.cpp` | **S** | unlocks load-and-expand-bf16 on every backend |
| 6 | CPU `vec_dot` | `cpu_quant_dot.cpp` | **M** | required for the CPU keep-quant path |
| 7 | ROCm dot kernel | `src/vt/rocm/rocm_grouped_gemm.hip` | **L** | see below |
| 8 | `DeviceKeepQuantSupported` admission | `gguf_keep_quant.cpp:128-141` | **XS** | one arm, gated on 7 existing |
| 9 | generated manifest + census test | `scripts/`, `tests/vllm/models/` | **S** | `gen-qwen38-27b-gguf-manifest.py` is the pattern to copy |
| 10 | `docs/USAGE.md` weights block | `docs/USAGE.md` | **XS** | repo + revision + sha256 per arm, per AGENTS.md |

Items 1, 3, 5 and 9 alone would let the artifacts **load** and expand to bf16 on
every backend, which is a real capability and a legitimately small change.

**Item 7 is the one that decides the row, and it is bigger than it looks.**
`DeviceKeepQuantSupported` (`gguf_keep_quant.cpp:128-141`) returns true on
`kROCM` for exactly `{Q8_0, Q4_K, Q5_K, Q6_K}`, and `rocm_grouped_gemm.hip`
implements exactly those four dots. `MXFP4` does appear in that file, but only
inside a refusal — `rocm_grouped_gemm.hip:692` and `:763`:

```
vt rocm: matmul_bt_quant: unsupported weight dtype
(ported: Q8_0/Q4_K/Q5_K/Q6_K; owed: Q4_0/Q2_K/Q3_K/IQ2_XXS/IQ3_XXS/IQ2_S/MXFP4
 -- the loader pre-filters to the ported set, so reaching here is a bug)
```

**Our ROCm arm therefore has no FP4 compute kernel of any kind** — not even for
`MXFP4`, a mainline ggml FP4 type we already decode, which that message and
`.agents/specs/rocm-gg-keep-quant.md` both record as owed alongside Q2_K, Q3_K,
IQ2_* and IQ3_*. A ROCmFP4 dot kernel would be the **first** FP4 ROCm
compute path in this tree, ahead of a format vLLM actually references.

Without item 7, a ROCmFPX model on `strix:gpu0` expands to bf16 and consumes
~2x the memory the format exists to save, and every performance claim the
format makes is unreachable. With it, we would have built an AMD-private FP4
kernel before the upstream-standard one.

## 11. Recommendation, and the triggers that reverse it

**Do not port ROCmFPX now. Do not add it to the AGENTS.md secondary-oracle
table, and do not create `.agents/oracles/rocmfpx.md`.**

`scripts/check-oracle-pins.py:190-196` enforces set equality in both directions
between the AGENTS.md table and this directory, so an oracle file without a
table row fails the gate and a table row is the policy change. That policy change
requires `gateable`, and AGENTS.md is explicit that `gateable = yes` needs a
demonstrated build **and** run: "constructing a config proves nothing." Neither
has been attempted here, and until one is, the honest record is no record —
**not** a file saying `gateable = no`, which would assert an admission this wave
does not recommend.

Five reasons, in the order they bind:

1. **Nothing can be gated yet.** ROCmFPX has no measured build or run in this
   project, on `strix:gpu0` or anywhere. Every number the artifacts advertise —
   36.04 tok/s peak speculative decode, 4.26 bpw — is a README claim in a
   repository whose `config.json` is wrong about the model in 6 of 8 fields.
2. **The compute win is unreachable without the expensive item.** Item 7 is the
   whole point of an AMD FP4 format, it is `L`, and it would land before the
   `MXFP4` ROCm kernel that `rocm-gg-keep-quant.md` already owes for a format
   vLLM references and mainline ggml defines.
3. **The family is still growing.** Two type ids were added 3 days before the
   pin and one 10 days before; kernel correctness fixes landed within 24 hours
   of it. The 4-bit core is stable and the rest is not, so any port must be
   scoped to ids 100 and 101 and must say so.
4. **The demand is one model on one box.** Nine artifacts, one architecture,
   two producers, both requantizing the same Unsloth checkpoint. There is no
   second model and no third producer.
5. **The model arm this would attach to is itself ungated.** The Q4_K_M token
   gate on this exact checkpoint fails 5 of 6 generations against llama.cpp
   `b10451` (§5, #821). A new quant arm added now could not be told apart from
   that divergence, so the port would be unmeasurable even after it worked.

The case in favour is real and should not be lost: the encoding is small and
cleanly specified, the model arm already exists (§5), the 4-bit layouts have
been frozen for three months, and `strix:gpu0` is a live fleet device with 62
GiB that we currently give no quantized ROCm path beyond four k-quants.

**Any of these reverses the recommendation:**

- **T1 — upstream adopts it.** `ggml-org/llama.cpp` merges the ROCmFP4 types, or
  allocates them ids in the mainline sequence. The encoding then stops being one
  fork's private range, and the `llama-cpp` oracle covers it at a release pin.
- **T2 — the MXFP4 ROCm kernel lands.** Once `rocm-gg-keep-quant.md`'s owed
  `MXFP4` dot exists on `gfx1151`, item 7 becomes an adaptation of a working FP4
  ROCm kernel rather than the first one, and item 2's dtype cost is the only
  large term left.
- **T3 — a second independent model appears** in ROCmFP4, from a producer who is
  not requantizing the same Unsloth Qwen3.8-27B. That turns a one-artifact format
  into an ecosystem.
- **T4 — a developer decision** that `strix:gpu0` needs a sub-8-bit weight path
  now. In that case take **only** ids 100 and 101, items 1/3/5/9/10 first as a
  load-and-expand slice with the census test as its gate, and open item 7 as its
  own row behind T2.

## 12. What this wave did NOT do, and why

- **No artifact was downloaded.** Nine headers by range request, 108 MiB total against 146.5 GiB of artifacts.
  Root disk went 19G -> 18G available across the whole wave.
- **`SHA256SUMS` was not verified** (§8). Verifying it requires the artifacts.
- **ROCmFPX was not built or run.** That is the gateability measurement, it
  belongs to the port decision, and doing it would have implied the admission
  this spec recommends against.
- **No `.agents/oracles/rocmfpx.md`** (§11).
- **No product code, no roadmap row, no quantization-matrix row.** The
  recommendation is not to open the row; adding one to say "not doing this"
  would write a shared file to record an absence.
- **The refusal in §3 was read, not executed.** `FindGgmlTraits` enumerates no
  id >= 100 and `gguf_reader.cpp:483-485` fails on the first unknown id; both
  are short and unambiguous. A red-before test would have required a full build
  of this tree to assert a refusal no code path disputes, and would have pinned
  behaviour this wave recommends changing only under T4. Owed as O1 if T4 fires.

## Owed

- [#2463](https://github.com/mudler/vllm.cpp/issues/2463) — this scoping wave.
  Closed by the change that lands this spec. The ROCmFPX port itself is
  **deliberately not owned** by a row: §11 recommends against opening one, and
  T1-T4 name what would change that.
- **O1** — if T4 fires, a red-before test that pins the OPEN-time refusal of a
  ROCmFPX type id before ids 100/101 are added to `FindGgmlTraits`, using
  `tests/vllm/gguf_builder.h`. Not owed now, because the refusal is current
  correct behaviour and this wave recommends leaving it.
- **O2** — `.agents/specs/rocm-gg-keep-quant.md` already owes the `MXFP4` ROCm
  dot kernel. §10 and T2 record that it now also blocks any ROCmFP4 compute arm.
  No new issue is filed; the existing owed item is unchanged in scope.

## Now

`QUANT-ROCMFPX-SCOPE` is a scoping wave and reaches no lifecycle state. It opens
no row. This spec is its whole output.
