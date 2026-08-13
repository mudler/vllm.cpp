# NVFP4 nibble order — two producers, two conventions, one shared dequant

**Row:** `MODEL-DIFFUSION-LTX25` phase L9a (`.agents/specs/ltx-2-5.md` §4.1, §4.2).
**Issue:** [#435](https://github.com/mudler/vllm.cpp/issues/435).
**Branch:** `row/LTX25-L9A-NVFP4-LINEAR`.
**Authority:** operator, 2026-08-13, after L9a returned `NEEDS_DECISION` rather than
building the arm it was briefed to build.
**Status:** spec committed BEFORE implementation, per AGENTS.md §"Spec before code".

---

## 0. Why this needs its own spec

`DequantNvfp4ToBf16` (`nvfp4_dequant.h:59`) is shared by the MiniMax-H3, Laguna
(Qwen3.5), DeepSeek-V4 and Qwen3-32B NVFP4 arms and by the `vt` fp4 GEMM reference.
Phase L9a needs it to read a checkpoint packed the other way round. A change there is
not LTX-2.5 loader work, so it is not made inside the LTX row — it is specified,
gated, and reviewed on its own terms.

## 1. The fact, measured and sourced

E2M1 packs two 4-bit values per byte. **Which logical element lands in which nibble is
a producer convention, and the two producers we consume disagree.**

| Producer | Element `2j` goes in | Source |
|---|---|---|
| **torchao** (`NVFP4Tensor`) | the **LOW** nibble | `torchao/prototype/mx_formats/kernels.py:160` — `(uint8_data[::2] \| uint8_data[1::2] << 4)`; its own inverse at `:137-139` stacks `first_elements = data & 0b1111` first |
| **vLLM** (reads torchao/ModelOpt) | the **LOW** nibble | `nvfp4_emulation_utils.py:321-324` — `low = a_flat & 0x0F` then `torch.stack((low, high))`; the Triton path agrees at `:101-108` (`tl.interleave(low_result, high_result)`) |
| **Lightricks `nvfp4-prequant`** | the **HIGH** nibble | `ltx-kernels/docs/NVFP4.md:27-29` — "`hi_first=True` (default) puts element `2j` in the **high** nibble … Pre-quantized checkpoints used with `nvfp4-prequant` are expected in the default order"; `ltx-core/quantization/nvfp4/linear.py:6` |
| **"Star Ultimate Model Converter Pro"** (`lilcheaty/MiniMax-H3-NVFP4`) | the **HIGH** nibble | already recorded at `minimax_h3.h:1500-1515` |

Our `DequantNvfp4ToBf16` is low-first (`nvfp4_dequant.cpp:74-75`), which is correct for
torchao and ModelOpt and wrong for the other two.

**Read low-first, a high-first file has every adjacent pair transposed.** It is finite,
correctly shaped, and wrong — the exact failure class this project keeps recording. L9a
measured it on the first-party LTX-2.5 NVFP4 DiT against the independent `vonkaiser` FP8
oracle: correct reading corr **0.9956** / 9.46% rel rms; wrong nibble order corr **0.032**.
See `.agents/specs/ltx-2-5.md` §4.1 for the full table and the control.

### 1.1 §4.2 is ANSWERED, and the answer is "no shipped defect"

`ltx-2-5.md` §4.2 recorded an open question: our torchao text-encoder arm reads low-first,
but its gate compared against goldens made by our own low-first helper — consistency, not
correctness. The answer is **low-first**, and the shipped TE arm is correct. This spec's
test work converts that self-referential gate into a source-anchored one so it cannot
regress into a question again.

**What each witness actually witnesses — because "three independent witnesses" overstated
it.** torchao is neither installed nor vendored on this box (`import torchao` →
`ModuleNotFoundError`; no `mx_formats/kernels.py` anywhere on the filesystem), so the two
torchao citations are read from upstream source rather than executed or diffed here:

| Witness | What it proves | Verified how |
|---|---|---|
| `torchao/prototype/mx_formats/kernels.py:160` (`pack_uint4`) | torchao's WRITER is low-first | upstream source only — **not executable on this box** |
| the same file `:137-139` (its unpacker) | torchao's READER agrees with its writer | upstream source only — **not executable on this box** |
| vLLM `nvfp4_emulation_utils.py:321-324` (`break_fp4_bytes`) | the **ModelOpt / compressed-tensors** convention is low-first | pinned tree at `555967922`, read and line-anchored |
| the Triton path at `:101-108` | the same, a second way | pinned tree, read and line-anchored |

The last two are **not** independent witnesses to torchao's convention. `break_fp4_bytes`
is reached only from `dequantize_to_dtype` (`:380`), which serves the ModelOpt and
compressed-tensors NVFP4 paths; vLLM's torchao path (`quantization/torchao.py:290-318`)
delegates to `torchao.quantization.quantize_` and torchao's own tensor subclasses and never
calls it. So vLLM witnesses ModelOpt, torchao's own source witnesses torchao, and the claim
that rests on all of them is the weaker "ModelOpt and torchao agree, on two sources of
differing strength" rather than "three independent confirmations".

This does not move the decision: the default is unchanged, and every caller of the shared
dequant that predates this spec consumes ModelOpt or compressed-tensors bytes — the arm the
strong witness covers. It is recorded so the strength of the anchor is not overstated to
whoever next changes the default.

## 2. Scope

**In.** One parameter on `DequantNvfp4ToBf16` selecting nibble order, **defaulting to
today's low-first behaviour**; the LTX-2.5 loader's producer discrimination and its use of
that parameter; the shape assertion admitting the cuBLAS-padded framing of the swizzled
scale; a correlation gate that proves the DiT dequantizes to the right numbers.

**Out.** H3's `MiniMaxH3Nvfp4SwapNibbles` (§3.1 says why it stays); the fp4-resident
device GEMM path (`cuda_matmul_nvfp4.cu`), which stays low-first-only; MXFP4 and GGUF
NVFP4, which have their own packing and are untouched; the TE's tower wiring, owned by
`row/LTX25-L10-TEXT-TOWER`.

## 3. Design

```cpp
enum class Nvfp4NibbleOrder {
  kLowFirst,   // element 2j in the LOW nibble — torchao, ModelOpt, vLLM. DEFAULT.
  kHighFirst,  // element 2j in the HIGH nibble — Lightricks nvfp4-prequant.
};

void DequantNvfp4ToBf16(const uint8_t* packed, const uint8_t* weight_scale_fp8,
                        float weight_scale_2, int64_t out_dim, int64_t in_dim,
                        uint16_t* out_bf16,
                        Nvfp4NibbleOrder order = Nvfp4NibbleOrder::kLowFirst);
```

A defaulted trailing parameter makes "every existing caller is untouched" true **by
construction** rather than by inspection. The proof obligation in §5 is that the gates
agree.

### 3.1 Why NOT H3's nibble-swap-at-load, which is prior art

`minimax_h3.h:1500-1517` already solved this once, by swapping the two nibbles of every
byte at load so the file becomes standard low-first. That is a genuinely different design
and it was right **for H3**, whose bytes also feed a Marlin fp4-**resident** path: one
transform fixes the bf16 dequant and the device GEMM together.

LTX-2.5 has no fp4-resident arm — `Ltx2StreamDitToDevice` dequantizes to bf16 and uploads
— so a swap would allocate and rewrite a second copy of 9.4 GB of packed weights to feed a
consumer that could just as well have read them in place.

Two mechanisms for one concept is a real smell and it is recorded rather than hidden:
**if LTX-2.5 ever grows an fp4-resident NVFP4 device path, this decision must be revisited
and the load-time normalization adopted**, because a host-side dequant parameter cannot fix
a device GEMM that reads the packed bytes itself. That is the tracked condition, written
where it will be found.

### 3.1.1 The SECOND tracked condition: the resolver does not refuse a linear file

The marker-less arm is an inference, and §3.3 says why no shape test can corroborate it for
this artifact. The consequence, stated exactly rather than as the comfortable version:

**A marker-less NVFP4 checkpoint whose `weight_scale` is stored LINEAR `[N, K/16]` is
RESOLVED as `nvfp4-prequant`, not refused.** For every geometry with `N % 128 == 0` and
`G % 4 == 0` — all 1176 quantized modules of the shipped DiT, gated as
`ambiguous_with_linear == 1176` — the cuBLAS-padded shape and the linear shape are the same
two integers. Such a file gets the 128x4 unswizzle applied to scales that were never
swizzled AND is read high-first. It loads, it renders, and it is wrong.

Linear `[N, K/16]` is the normal case, not a corner one. It is what NVIDIA ModelOpt,
llm-compressor and compressed-tensors all write, and what vLLM's own readers allocate for on
disk (`modelopt.py:1335-1345`, `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:73-76`,
both `[out, in // group_size]` at the pin `555967922`). None of the three emits a
`.torchao_nvfp4` sidecar, so **the marker's absence excludes torchao and nothing else.**

It is resolved this way anyway because there is no better evidence in the file, and that was
checked rather than assumed: the shipped DiT's `__metadata__` carries exactly `config`,
`gemma_source_checkpoint`, `model_version` and `license` — no `quantization_config`, no
producer key, no `nvfp4`/`torchao`/`quant` substring anywhere in the config, and no tensor
name mentioning the quantizer (`MARKER_LIKE == []` over all 7876 tensors, read straight from
the 1,179,408-byte header). MiniMax-H3's community checkpoint DID name its converter in
metadata; this one names nothing. So the decision stands, and it is the **claim** that was
wrong: this resolver refuses every combination the SHAPE can separate, and cannot refuse the
one it cannot see.

**The tracked condition.** Before this loader is pointed at any marker-less NVFP4 artifact
other than the first-party LTX-2.5 DiT, §5.2's correlation gate must be re-run for THAT
artifact against an independent oracle. Passing on the DiT is evidence about the DiT. If a
producer key ever does appear in such a file, key on it and demote this inference — that is
strictly better evidence than the shape, which is worth none here.

### 3.2 The producer discriminator (operator-RATIFIED, 2026-08-13)

The scale layout and the nibble order are decided together, from the **`torchao_nvfp4`
marker's presence or absence**. This is in-file evidence, not provenance guessing: torchao
always writes that marker, so its absence excludes torchao.

| Marker | Required `weight_scale` shape | Resolves to |
|---|---|---|
| present, `is_swizzled_scales=true` | `[32*ceil(N/128), 16*ceil(G/4)]` (`to_blocked` framing) | torchao: unswizzle, **low**-first |
| absent | `[round_up(N,128), round_up(G,4)]` (cuBLAS-padded framing) | `nvfp4-prequant`: unswizzle, **high**-first |
| anything else the shape can SEPARATE | — | **REFUSE BY NAME** |

`G = in_features / 16`. No "probably" branch: a marker with the wrong shape, an absent
marker with the wrong shape, or a marker declaring an unimplemented combination all refuse.

**The third row's qualifier is load-bearing, not hedging.** Because §3.3 makes the padded
framing shape-identical to the LINEAR one for every geometry here, the marker-less arm cannot
refuse a genuinely linear file — it resolves it, and reads it swizzled and high-first. §3.1.1
states that consequence in full and records the condition it imposes.

**Both framings describe the SAME bytes.** `[32*ceil(N/128), 16*ceil(G/4)]` and
`[round_up(N,128), round_up(G,4)]` have equal element counts and the identical layout;
they are two 2-D dresses on one flat buffer. `Ltx2UnswizzleNvfp4BlockScale` is already
framing-agnostic — it keys off logical `rows`/`cols` and a byte count, and its index
formula matches `ltx-kernels/csrc/nvfp4/quantize.cu:26-31` term for term — so **only the
shape assertion changes**, not the permutation.

**The assumption must be legible at the code site.** The marker-absent arm is an inference
from a producer's signature, not a fact read off the file, and the comment there says so
and cites the evidence: the 0.9956 correlation and `ltx-kernels/docs/NVFP4.md:27-29`.

### 3.3 Why the shape cannot discriminate alone

For every quantized layer in the first-party DiT, `N % 128 == 0` and `G % 4 == 0`, so the
cuBLAS-padded shape **numerically equals the linear `[N, G]` shape**. A shape test can
therefore never separate "swizzled, padded framing" from "linear". That is why the marker
is primary and the shape only corroborates, and it is why the original linear-arm brief was
unimplementable as written.

## 4. Risks

| Risk | Mitigation |
|---|---|
| A defaulted parameter silently changes an existing caller | Every NVFP4 gate re-run with **assertion counts compared**, not just Status (§5) |
| The high-first arm is wrong and output is merely finite | The correlation gate (§5.2) requires the 0.9956 / 9.46% signature; finiteness is not accepted as evidence |
| Marker-absent inference generalizes to a file it should not | **NOT mitigated by refusal, and §3.1.1 says so in the tree.** Every combination the SHAPE can separate refuses by name, with no fallback branch — but a marker-less file whose scale is stored LINEAR `[N, K/16]` is shape-identical to the padded framing for every geometry here, so it RESOLVES and is read wrong. The mitigation is §5.2's correlation gate against an independent oracle, owed per artifact, plus the tracked condition at §3.1.1 and the same statement in `ltx2_loader.h` and `docs/USAGE.md` |
| Two nibble mechanisms drift apart | §3.1 records the condition under which H3's is adopted here |
| The correlation gate needs a 21 GB and an 18.7 GB file | It reads byte RANGES from the two headers, materializes ~128 rows, and is opt-in behind an env var like the other shipped-checkpoint cases |

## 5. Tests and gates

### 5.1 RED first — and there are TWO reds, not one

This section originally named one red ("the loader refuses the shape") for a change with two
independent halves. A throw can only ever demonstrate the first half, so stating it as *the*
right reason mislabels the evidence for the second. Both are required:

**(a) The layout red — a THROW.** Before the change the loader knew only the `to_blocked`
framing, so the shipped DiT's cuBLAS-padded `weight_scale` was refused by name and no bytes
were decoded at all. That is a genuine red and it is the one the pre-change tree produces.

**(b) The nibble-order red — a NUMBER, because no throw exists for it.** Fixing (a) alone
makes the file load and produce values that are finite, correctly shaped, correctly scaled
and wrong; §7 gates that absmax is IDENTICAL under both orders, so nothing structural can
fire. The only instrument that separates them is the correlation against the independent FP8
oracle, and the red it must produce is the collapse recorded in §7: corr **-0.00239115**,
rel rms **1.41856**, against **0.994968 / 0.100672** for the correct read.

Red (b) is not run once and discarded. It is a permanent arm of the committed gate
(`test_ltx2_loader.cpp`, the `kLowFirst` arm), for the reason the whole spec exists: the
defect it detects leaves no other trace.

### 5.2 The correlation gate — the one nobody had
The `vonkaiser` FP8 DiT and the first-party NVFP4 DiT quantize the same base weights, so
the FP8 file is an **independent oracle** for the NVFP4 read. The gate:

- dequantizes the same module from both files (rows 0..127),
- asserts Pearson correlation in **[0.99, 0.998]**,
- asserts relative rms error in the **BAND [0.085, 0.115]** — not a ceiling; see below,
- asserts a **control**: the same NVFP4 read against a DIFFERENT module's FP8 weights
  correlates **< 0.2**, so the gate proves it can tell right from wrong rather than
  passing on any two finite arrays,
- asserts the **wrong nibble order** collapses (corr < 0.2, rel rms > 0.5),
- asserts the two defects the band exists for, executed rather than described: a uniform
  **x1.10 group-scale error** leaves corr unmoved to within 1e-6 and pushes rel above the
  band, and the **oracle against itself** lands below the rel floor and above the corr
  ceiling,
- and emits the VALUES, never a boolean.

Without the control the gate is `7.0(c)` again — a fixture that cannot separate a correct
implementation from a plausible wrong one.

**Why rel rms is a BAND (revised after review; the original `<= 0.15` was one-sided).**
rel rms here is a PREDICTED QUANTITY, not an error budget: it is the disagreement two
different quantizations of the same base weights MUST show, measured at 0.100672. A
one-sided ceiling was wrong in both directions, and both were measured on the committed
bytes rather than argued:

| arm | corr | rel rms | `<= 0.15` | band |
|---|---|---|---|---|
| correct read | 0.994968 | 0.100672 | pass | **pass** |
| uniform group scale x1.04 | 0.994968 | 0.111935 | pass | pass |
| uniform group scale x1.05 | 0.994968 | 0.116760 | pass | **RED** |
| uniform group scale x1.09 (in the dequant) | 0.994984 | 0.141100 | pass | **RED** |
| uniform group scale x1.10 (in the dequant) | 0.994964 | 0.150094 | RED by 0.06% | **RED** |
| uniform group scale x0.92 | 0.994968 | 0.122618 | pass | **RED** |
| the FP8 oracle itself ("too good") | 1.0 | 0.0 | pass | **RED** |
| wrong nibble order | -0.00239 | 1.41856 | RED | RED |

Rows marked "in the dequant" are real source mutations of `DequantNvfp4ToBf16`'s
`group_scale`, so the bf16 store-rounding moves with them; the others scale the decoded
floats afterwards, which is why x1.10 reads 0.148922 there and 0.150094 here.

Two facts drive it. **Pearson correlation is scale-invariant** — corr reads 0.994968 to every
printed digit for every multiplier in that table — so rel is the ONLY statistic in the gate
that can see a systematic scale error, and slack spent there is not recoverable elsewhere.
And **rel had no floor**, so an arm that reproduced the oracle exactly — the shape a fixture
takes when the "independent" oracle has quietly become the thing under test — passed with
room to spare. The band's width (+/- ~15% of the measured value) is for compiler and libm
drift only: every input is a committed byte array reduced by a sequential double loop, so
the value is deterministic run to run.

### 5.3 Nibble order, source-anchored
A unit case pinning both orders against hand-computed bytes, plus a generator-side anchor
on torchao's `pack_uint4` line and `ltx-kernels`' `hi_first` documentation, so a convention
change upstream is reported as a source change (§1.1's repair).

### 5.4 The discriminator
Every row of §3.2's table, including each refusal, asserted by name.

### 5.5 Unchanged-behaviour proof (operator's non-negotiable), and its REAL reach

These must be green with **identical case and assertion counts**, reported. Baselines
captured on this branch at `f400413e`, Release, CUDA=OFF.

**A byte-identical count is only evidence for a gate that EXECUTES the changed function.**
That was assumed rather than checked when this section was written, and the check changes
what the sweep is worth. Each gate was instrumented with a call counter on
`DequantNvfp4ToBf16` and then re-run under mutation **M3** — flip the shared default from
`kLowFirst` to `kHighFirst`, i.e. break every defaulting caller at once:

| Gate | Baseline | Live calls into `DequantNvfp4ToBf16` | Under M3 | What the count proves |
|---|---|---|---|---|
| `test_nvfp4_dequant` | 4 / 47 | 6 | **RED** | sensitive |
| `test_gguf_nvfp4` | 14 / 2352 | 6 | **RED** | sensitive |
| `test_ltx2_loader` | 20 / 2363 | 294 | **RED** | sensitive |
| `test_qwen3_forward` | 7 / 1557 | 30 | GREEN | **BLIND — executes it and cannot see the flip** |
| `test_minimax_h3` | 79 / 57395 | 44 | GREEN | **BLIND — 57,395 assertions, none nibble-sensitive** |
| `test_qwen36_weights` | 7 / 45 | 0 | GREEN | vacuous (needs a checkpoint that is absent) |
| `test_ops_nvfp4_matmul` | 4 / 1 | 0 | GREEN | vacuous |
| `test_ops_moe_grouped` | 6 / 3 | 0 | GREEN | vacuous |
| `test_ops_nvfp4_fp4` | 22 / 919 | 0 | GREEN | vacuous |
| `test_ltx2_device` | 13 / 498 | 0 | GREEN | vacuous |

So the honest statement is **three of ten** gates can see a nibble-order change. Five never
call the function, and their identical counts say nothing about it in either direction. Two
call it live and still pass — which is not a defect in this change (the default is unchanged,
so nothing moves) but IS a standing gap in those suites, and the reason "eight byte-identical"
was the wrong summary of this evidence.

**The composition hazard that creates.** H3 reaches low-first by normalizing its bytes at
load (`MiniMaxH3Nvfp4SwapNibbles`, gated behind `MiniMaxH3Nvfp4HighNibbleFirst()`), then
calling the shared dequant with the default. If H3 is ever routed through the new
`kHighFirst` parameter while that swap stays on, the two compose into a double flip — the
original defect, restored — and **nothing in the tree fires**. Tracked as
[#598](https://github.com/mudler/vllm.cpp/issues/598): H3 owes a nibble-sensitive gate. The
instrument already exists; H3's own FL2VA-GGUF correlation is the same independent-oracle
technique §5.2 builds here.

Counts only RISE, and only in the suites this change adds cases to. **If any other
gate moves, STOP.** `Status:` is grepped every run: a thrown case DROPS the assertion
count while still printing "passed".

### 5.6 The real artifact
The first-party NVFP4 DiT stages onto the GPU and forwards, on dgx.casa under
`flock -w 2700 $HOME/gpu.lock`, naming the file, its tensor count and its derived geometry,
with finite non-degenerate output.

## 6. Stop conditions

- The high-first read does not reproduce the correlation signature → STOP; the encoding is
  not merely a nibble order and the premise is wrong again.
- Any existing NVFP4 gate changes case or assertion count → STOP.
- A checkpoint appears that needs a third convention, or a marker/shape combination outside
  §3.2 that the shape CAN separate → refuse by name and report; do not add a branch.
- A marker-less NVFP4 checkpoint other than the first-party DiT is pointed at this loader →
  STOP until §5.2's correlation gate is run for THAT artifact. It will not be refused; §3.1.1
  says why, and passing on the DiT is evidence about the DiT.
- Any temptation to accept "the forward ran and the output was finite" as the correctness
  result → forbidden; §5.2 is the result.

## 7. Outcome

**What was measured.** The first-party NVFP4 DiT is SWIZZLED and HIGH-nibble-first, and both
halves were established against an oracle that is not ours — the `vonkaiser` FP8 DiT of the
same base weights. Re-derived after a full-disk incident on the build host and identical to
every digit:

| reading | rms | corr vs FP8 | rel rms |
|---|---|---|---|
| LINEAR / low-first (what the original brief asked for) | 0.013564 | 0.000414 | 1.786 |
| LINEAR / high-first | 0.013564 | 0.257746 | 1.558 |
| SWIZZLED / low-first (the shipped dequant) | 0.009208 | 0.032296 | 1.394 |
| **SWIZZLED / high-first** | **0.009208** | **0.995560** | **0.0946** |

**What the committed gate reports, transcribed from its own output** (`test_ltx2_loader
--test-case="*NVFP4 DiT agrees*" -s`, Release, CUDA=OFF). An earlier revision of this section
recorded the wrong-order arm as `0.00514`, a number no row of the table above and no line of
the gate produces; the control on the same sentence, `0.00362`, was exact. Corrected against
the running gate rather than re-transcribed:

| arm | corr | rel rms |
|---|---|---|
| oracle (the correct read) | **0.994968** | **0.100672** |
| control (a DIFFERENT module's FP8 weights) | **0.00362335** | — |
| deliberately-wrong nibble order (low-first) | **-0.00239115** | **1.41856** |
| absmax, both orders | identical at 0.0217285 | — |

**What was rejected.**

1. *A linear-layout arm.* The premise was wrong; building it would have made the file load and
   render silently wrong. L9a stopped and returned `NEEDS_DECISION` instead.
2. *Discriminating by SHAPE.* Gated as impossible for this artifact: all 1176 quantized modules
   have a cuBLAS-padded scale shape numerically identical to the linear one
   (`ambiguous_with_linear == 1176`). The marker leads; the shape only corroborates.
3. *H3's nibble-swap-at-load.* Correct for H3, which also feeds a Marlin fp4-resident path, and
   wasteful here: LTX-2.5 dequantizes to bf16, so a swap would rewrite 9.4 GB of packed weights
   to feed a consumer that could read them in place. §3.1 records the condition that reverses
   this decision.
4. *"The forward ran and the output was finite."* The gate asserts absmax is IDENTICAL under
   both nibble orders, so every magnitude summary is blind to the defect by construction. Only
   the correlation separates them.
5. *A one-sided rel-rms tolerance.* Revised after review into the band at §5.2: `<= 0.15` alone
   admitted a +10% uniform group-scale error, and correlation is scale-invariant so nothing
   else in the gate could have caught it.

**Why the default is low-first.** It is what torchao's `pack_uint4` writes
(`kernels.py:160`), what torchao's own unpacker reads (`:137-139`), and what vLLM's readers of
ModelOpt / compressed-tensors bytes assume (`nvfp4_emulation_utils.py:321-324` and `:101-108`).
§1.1 records what each of those witnesses is actually worth — the two torchao lines are upstream
source, not executable on this box, and the two vLLM readers witness the ModelOpt convention
rather than torchao's writer. Every caller predating this change consumes a ModelOpt or
compressed-tensors checkpoint, so a defaulted parameter makes "nothing else moves" true by
construction.

**§4.2 closed, no defect.** torchao is low-first, so the shipped text-encoder arm was already
correct. Its gate is now anchored to upstream sources rather than comparing our helper to
itself. The earlier phrasing — "three independent witnesses" — is corrected in §1.1: two of the
four cited lines are torchao's own writer and reader (one convention, seen twice), and the other
two are vLLM reading ModelOpt.

**Gates.** Ten NVFP4 gates plus nine LTX/video gates green; the only counts that moved are the
suites this change adds cases to (`test_nvfp4_dequant` 4/47 -> 5/69, `test_ltx2_loader`
20/2363 -> 24/4809, the last +16 from the §5.2 re-banding and its two built-in mutation arms).
Mutations run, each with the tree restored byte-for-byte afterwards:

| mutation | result |
|---|---|
| force low-first at the LTX call site | RED (3 cases) |
| resolve a marker-less padded file as torchao | RED (3 cases) |
| reinstate an apostrophe in a refusal | RED (1 case) |
| **M3: flip the shared default to `kHighFirst`** | RED in 3 of 10 gates; **GREEN in the other 7**, two of which execute the changed function 30 and 44 times. §5.5 has the per-gate call counts; the blindness is tracked as [#598](https://github.com/mudler/vllm.cpp/issues/598) |
| **M4: uniform group-scale error, applied inside `DequantNvfp4ToBf16`** | x1.10 -> rel **0.150094**, which the old `<= 0.15` catches by 0.06%. x1.09 -> rel **0.1411**: old bound **GREEN** (45/45, `Status: SUCCESS!`), §5.2 band **RED**. Same binary path, only the bound differs, so the band is strictly stronger. Now a permanent arm of the gate |
| **the FP8 oracle handed to the gate as its own answer** | Old bounds: every CORRELATION assertion passes — corr `1 >= 0.99`, rel `0 <= 0.15`, control `0.0026 < 0.2`, and both wrong-order arms — and only the incidental `got_absmax == wrong_absmax` fires, because `wrong` still reads the real bytes while `got` does not. The assertion that is supposed to BE the result passed. §5.2 band: **RED on the result itself**, `rel >= 0.085` and `corr <= 0.998`. Also now a permanent arm |

**What the ten-gate sweep does NOT prove.** Eight-of-ten byte-identical was reported as the
unchanged-behaviour evidence. Five of those eight never call `DequantNvfp4ToBf16` at all on a
default run, so their identical counts are evidence of nothing about this change in either
direction. The claim that survives is narrower and still sufficient: the default is unchanged,
every pre-existing caller resolves to it by construction, and the three gates that CAN see a
nibble-order change are green.

**On the GPU (dgx.casa, GB10, one `flock -w 2700 $HOME/gpu.lock` hold 02:31:03 to 02:37:15).**
Both arms stage device-resident and forward, `Status: SUCCESS!`:

| arm | file | assertions | output absmax |
|---|---|---|---|
| NVFP4 | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` (18,721,432,024 B, 7876 tensors) | 5626 | video 0.275391 / audio 1.02344 |
| FP8 | `ltx-2.5-22b-distilled-fp8.safetensors` (6124 tensors) | 5624 | video 0.300781 / audio 2.14062 |

The FP8 figures reproduce the previously recorded GB10 run to the digit, which is what makes
the NVFP4 figures next to them trustworthy rather than merely new. The two arms differ because
they are different quantizations AND different configurations: the NVFP4 file declares a config
(`double_precision_rope`, `av_ca_timestep_scale_multiplier = 1000`, both asserted after
adoption), while the FP8 file declares none and runs under manifest defaults.

**Not obtained.** The device case's MESSAGE text (printed tensor count, staging seconds, config
source) was cut by a `tail` in the harness on the first run, and a second run to capture it
timed out on the GPU lock after 45 minutes (`FLOCK_EXIT=1`) against a saturated box. The
underlying facts are asserted rather than printed — the geometry CHECKs are among the 5626 that
passed — and 7876 tensors / 48 blocks / video 4096 / audio 2048 / 21.004B parameters are gated
from the committed manifest. Cosmetic, and not worth further contended GPU time while another
row is rendering.
