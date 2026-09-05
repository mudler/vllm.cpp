# MODEL-QWEN35-EXL3-HEAD — the trellis output head, seen from all three of its readers

Row: `MODEL-QWEN35-EXL3-HEAD`
Issues: [#2495](https://github.com/mudler/vllm.cpp/issues/2495) (item 5's
remainder and item 6);
[#2569](https://github.com/mudler/vllm.cpp/issues/2569) (fixed and closed by
this row)
Base SHA: `a2c7e1962` (`origin/main`)
Parent rows: [`MODEL-QWEN35-GDN-EXL3`](model-qwen35-gdn-exl3.md) →
[`MODEL-QWEN35-EXL3`](model-qwen35-exl3.md) →
[`QUANT-EXL3-MUL1`](quant-exl3-mul1.md) → [`QUANT-EXL3`](quant-exl3-shared.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`. vLLM defines the
MTP head (`vllm/model_executor/models/qwen3_5_mtp.py`), the head-sharing rule
(`v1/worker/gpu/spec_decode/eagle/utils.py::load_eagle_model`) and the logits
seam (`vllm/model_executor/layers/logits_processor.py`). vLLM implements no
EXL3 at the pin, so the trellis FORMAT comes from the registered secondary
oracle [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT), exactly as the parent rows
record. Nothing here re-derives either half.

## Now

`DONE`. An EXL3 `Qwen3_5ForConditionalGeneration` checkpoint's `mtp.*` draft
head loads and runs, and all three readers of the TARGET's trellis `lm_head`
compute with it packed.

## The gap

The head half of #2495 item 5 already landed with the parent row: since
`e6e32d5fb`, `Qwen3_5DenseWeights::lm_head_exl3` exists,
`LoadQwen3_5Dense` fills it (`qwen3_5_dense_weights.cpp:1143`) and
`DenseLogitsF32D` computes with it (`qwen3_5.cpp:3214`). Verified against the
tree at the base SHA, not assumed from the issue text.

What was left is the other half of the same sentence and the same decision seen
from two more places.

### One representation, three readers

The target's head has exactly ONE storage decision. Three call sites read it:

| reader | before this row | production entry |
|---|---|---|
| the target's own logits | EXL3 ✅ (`e6e32d5fb`) | `ModelRegistry::Forward` |
| the MTP draft's `ComputeLogits` | bf16 / NVFP4 only | `LoadedModel::BuildMtpDraft` → the MTP speculator |
| the DFlash/DSpark draft's SHARED head | bf16 / NVFP4 only | `SharedHeadSource::LoadInto` in `FromModelDir` |

Specifying those separately is how a tree ends up with two answers to one
question. They are one row.

### The `mtp.*` half of item 5

`LoadQwen3_5MTP` requires every `mtp.*` tensor to be BF16 (`qwen3_5_mtp.cpp:38`,
mirroring the NVFP4 exclusion at `qwen3_5_mtp.py:86-103`). The published
checkpoint quantizes eight of them.

### What the artifact actually ships

`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` @ revision
`19441ac874c4018295da848e250f23511361cda4`, at
`/mnt/nas_share/models/Qwen3.8-27B-EXL3/target-3.5bpw/`.

The `mtp.*` and `lm_head` rows below were FIRST read from the shard's own
safetensors header by HTTP **range request** (`model-00002-of-00002.safetensors`,
bytes `8..155506`, no payload downloaded), while the shards were still
downloading, and were then RE-READ from the completed local shards. Both
readings agree, and the second is what the numbers below are. `config.json`,
`model.safetensors.index.json` and `quantization_config.json` were read from
the local copies.

`quantization_config.json` declares `head_bits: 6`, `mtp_bits: 4`,
`codebook: "mul1"`, `bpw: 3.5`, `out_scales: "always"`.

**The whole-file inventory, counted from both shards' headers** (2426 tensors,
409 of them with a `.trellis`):

| bits | modules | where |
|---|---|---|
| 3 | 137 | `layers.N.mlp.{gate,up,down}_proj` on 45-46 layers each |
| 4 | 270 | the GDN tower, `self_attn.*`, the other MLP layers, ALL of `mtp.*` |
| 5 | 1 | one `layers.N.self_attn.o_proj` |
| 6 | 1 | `lm_head` |

All 409 carry `.mul1`; NONE carries `.mcg`; and NO tensor in the checkpoint has
`scale` in its name, so `out_scales: "always"` is folded into `svh` and this
tree's no-scales premise holds. The MLP tower is MIXED WIDTH per layer, with one
lone five-bit attention projection among fifteen four-bit ones, which is what
per-tensor bit allocation looks like and is why `bits` is read from the trellis
and never from `bpw`.

That count corrects #2495's item table, which said 272 quantized tensors. It
also surfaces a blocker this row does not fix and does not own: `(bits 3,
codebook 2)` is NOT instantiated on the CUDA GEMM (`cuda_exl3.cu:1981`), so 137
modules of this target refuse by name on a device. That is
[#2574](https://github.com/mudler/vllm.cpp/issues/2574), owned by
`QUANT-EXL3-MUL1`; this row reached the same measurement independently and
[#2575](https://github.com/mudler/vllm.cpp/issues/2575) was closed as its
duplicate. **The model therefore still cannot run end to end on CUDA**, and
nothing in this row changes that. Nothing in this row is three-bit either:
`mtp.*` is uniformly four and the head is six, and `(4,2)`, `(5,2)` and `(6,2)`
are all instantiated.

| tensor | dtype | shape | reading |
|---|---|---|---|
| `lm_head.trellis` | `I16` | `[320, 15520, 96]` | K=5120, N=248320, **bits 6** |
| `lm_head.suh` / `.svh` | `F16` | `[5120]` / `[248320]` | K / N |
| `lm_head.mul1` | `I32` | `[]` | codebook 2 |
| `mtp.fc.trellis` | `I16` | `[640, 320, 64]` | K=10240=2H, N=5120=H, **bits 4** |
| `mtp.layers.0.self_attn.q_proj.trellis` | `I16` | `[320, 768, 64]` | K=5120, N=12288, bits 4 |
| `mtp.layers.0.self_attn.k_proj.trellis` | `I16` | `[320, 64, 64]` | N=1024, bits 4 |
| `mtp.layers.0.self_attn.v_proj.trellis` | `I16` | `[320, 64, 64]` | N=1024, bits 4 |
| `mtp.layers.0.self_attn.o_proj.trellis` | `I16` | `[384, 320, 64]` | K=6144, N=5120, bits 4 |
| `mtp.layers.0.mlp.gate_proj.trellis` | `I16` | `[320, 1088, 64]` | N=17408, bits 4 |
| `mtp.layers.0.mlp.up_proj.trellis` | `I16` | `[320, 1088, 64]` | N=17408, bits 4 |
| `mtp.layers.0.mlp.down_proj.trellis` | `I16` | `[1088, 320, 64]` | K=17408, N=5120, bits 4 |
| every one of those | — | — | carries a `.mul1` marker (codebook 2) |
| `mtp.norm.weight`, `mtp.pre_fc_norm_*.weight`, `mtp.layers.0.{input,post_attention}_layernorm.weight`, `mtp.layers.0.self_attn.{q,k}_norm.weight` | `BF16` | — | the UNQUANTIZED remainder |

Two facts in that table are load-bearing and neither is in the issue text.

**`mtp.fc` is quantized.** The head's fc-cat projection is not remainder; it is
the ninth EXL3 tensor and the one the MTP forward's own precondition
(`qwen3_5.cpp:8740`, `fc.rank == 2 && fc.nk`) refuses outright.

**The `mtp.*` remainder is BF16, not F16.** The parent row had to widen
`LoadModelVectorForScheme` / `LoadMergedBf16RawNK` to accept F16 because
exllamav3 stores the BODY's unquantized linear remainder at fp16. The `mtp.*`
norms are stored BF16, so `LoadBf16Direct` reads them unchanged and this row
widens no dtype. That asymmetry is the artifact's; it is recorded here because
"the parent needed F16, so this one does too" is the plausible wrong guess.

## Design

### D1 — the shared head stays PACKED. This is a mirror, not a product decision.

Every one of the three readers computes against the trellis. None dequantizes.

Upstream reaches this by construction rather than by choosing: the head is an
`nn.Module`, and the draft's `compute_candidates` calls
`LogitsProcessor.get_top_k_tokens(self.lm_head, ...)` → `_apply_head` →
`lm_head.quant_method.apply` (`logits_processor.py:241-286`, `:132-142` @ the
MERGED `vllm-project/vllm#52816` head `b389ac29`), which IS the target's own
logits path. There is no second head and no dequantized copy anywhere in it.
So this is not a fork the developer has to settle; `AGENTS.md` §"vLLM is the
reference" forbids asking.

Three further reasons agree, and are worth stating because the alternative is
cheap to write and expensive to own:

* **Size.** A dequantized copy of this head is `248320 × 5120 × 2` = **2.543
  GB**. `SharedHeadSource`'s own comment records that the borrow-first seam
  (`BorrowStTensorBytes`) was added in #1849 precisely to stop a ~2.54 GB
  anonymous copy of the same tensor. Materializing it back would undo that row
  on the arm this one adds. The packed trellis at 6 bits is
  `320 × 15520 × 96 × 2` = **0.953 GB**, plus 0.5 MB of sign vectors.
* **Exactness.** The DFlash2 selector's whole input is the target head's EXACT
  top-K. A dequantized head does not produce it, which is what
  `RefuseQuantizedDflash2LmHead` refuses. Handing that lane a widened head
  would be the silent-wrong D12 names, arriving through a new door.
* **There is already a lane.** `dense_exl3::Linear` → `layers::Exl3LinearMethod`
  → `dense_attn::Exl3MatmulD` computes with the packed trellis today and writes
  f32 directly. No second matmul is written by this row.

### D2 — the new arguments are REQUIRED, with no default

`SharedHeadSource::LoadInto` already carries two required-with-no-default
arguments and a comment explaining that a `= nullptr` default is what silently
turned the D12 carry off while all 38 dflash/gguf suites stayed green
(`model_loader.cpp:548-568`). The EXL3 owner is added the same way: a fifth
REQUIRED `vllm::Exl3Weight* head_exl3`. Dropping it at a call site is a compile
error, not a green run. `LoadDflashSharedLmHead` gains its fourth parameter on
the same rule.

Both existing call sites are updated to say what they want:

* the DSpark caller (`model_loader.cpp:974`) passes `nullptr` — that lane
  materializes one bf16 head and has nowhere to put a trellis, so it must keep
  refusing an EXL3 target head by name rather than silently loading nothing;
* the DFlash caller (`model_loader.cpp:1052`) passes
  `&draft->weights.lm_head_exl3`.

`LoadQwen3_5MTP`'s resolver overload gains a REQUIRED `has` predicate for the
same reason. An EXL3 probe needs to ask whether `<proj>.trellis` exists, and a
defaulted "assume nothing is quantized" predicate is exactly the silent-off
shape this section is about. Every caller is updated.

### D3 — ONE truth for "is this arm active", keyed on the trellis alone

`Qwen3_5MTPWeights::IsExl3()` returns `!fc_exl3.Empty()`, mirroring
`FullAttnLayerWeights::IsExl3` (keyed on `q_proj_exl3`) and
`DenseMlpWeights::IsExl3` (keyed on `down_proj_exl3`). The loader fills the
group together, so a half-populated container cannot read as an arm; the
predicate answers the same question the loader's own `IsExl3Projection` probe
asked, and no consumer re-derives it from a config scalar.

`Qwen3DFlashWeights` keeps its existing exactly-one-of rule and gains a third
owner: `lm_head`, `lm_head_fp4`, `lm_head_exl3`.

### D4 — probe per projection group, never `quantization_config`

The MTP loader probes `mtp.fc`, `mtp.layers.N.self_attn.q_proj` and
`mtp.layers.N.mlp.gate_proj` independently, exactly as the dense tower's
resolver does. `mtp_bits: 4` is never read: `Exl3Weight::Bits()` takes the width
from the trellis geometry, which is the rule the parent rows established after
measuring a 3.0bpw Llama whose head is six-bit under a config that says 3.

### D5 — branch ORDER mirrors `DenseLogitsF32D`

Every logits site orders its arms `exl3 → fp4 → bf16`, so the three readers of
one head cannot disagree about precedence. `Qwen3_5MTPModel::ComputeLogits` and
`DflashLogitsF32D` are brought onto that order.

### D5b — #2569: the shared read REFUSES when no arm matches

`LoadDflashSharedLmHead` had a bf16 arm and an NVFP4 arm and nothing else, and
when neither matched it fell off the end of its loop and RETURNED, leaving every
owner empty and saying nothing. Its own header comment claimed it "throws,
naming the target shards, when the head is absent"; the throw was actually one
layer up, in `SharedHeadSource::LoadInto`, and it names bf16 tensors.

**This row is what makes that reachable rather than theoretical.** The benchmark
target ships no `lm_head.weight` at all, so on the DSpark lane — which offers no
packed owner of either kind — the read matches nothing, and before this row the
user got "the target's bf16 embed_tokens + lm_head were not found", which names
the wrong tensor and the wrong reason for a head that is present.

The refusal moves to the function that made the routing decision, and it names
the three arms it tried and which of them the calling lane declined to offer.
`LoadInto`'s emptiness check stays: the GGUF arm reaches it too, and it now
includes the trellis owner so a loaded EXL3 head is not read as an absence.

### D6 — the refusals stay exactly as loud

`RefuseQuantizedDflash2LmHead` is NOT widened and NOT touched.
`lm_head_dequantized` keeps its single writer (the GGUF shared-head loader,
which dequantizes `output.weight` on the way in). A packed EXL3 head is the
opposite state — computed with, never widened — so it must not set that flag,
and a case asserts it does not. The GGUF arm never fills the EXL3 owner.

The MoE target ctor of `Qwen3_5MTPModel` passes `nullptr` for the EXL3 head:
`Qwen3_5MoeWeights` has no `lm_head_exl3` field, because no EXL3 MoE checkpoint
is in scope for this row. That is an absence, said out loud, rather than a
field nothing fills.

## Scope

IN:

1. `Qwen3_5MTPWeights` gains `fc_exl3` + `IsExl3()`; `LoadQwen3_5MTP` gains a
   required `has` and an EXL3 rung for `fc`, `self_attn.{q,k,v,o}_proj` and
   `mlp.{gate,up,down}_proj`; its validation becomes arm-aware.
2. `MtpHeadHidden` routes `fc` through `dense_exl3::Linear`;
   `Qwen3_5MTPModel::Forward`'s fc precondition becomes arm-aware.
3. `Qwen3_5MTPModel` shares the target's `lm_head_exl3` and
   `ComputeLogits` computes with it.
4. `SharedHeadSource::LoadInto` and `LoadDflashSharedLmHead` gain a REQUIRED
   EXL3 owner; `Qwen3DFlashWeights::lm_head_exl3`; `DflashLogitsF32D` computes
   with it.
5. #2569: `LoadDflashSharedLmHead` throws by name when NO arm matches, instead
   of returning an empty head. Fixed in this flow because this row's arm is what
   makes the silent case reachable, and closed by it.

OUT:

* The DFlash2 draft's OWN weight loading (#2495 item 7). Only the call-site
  signature updates item 4 forces are made in `qwen3_dflash_weights.cpp`.
* The NVFP4 KV cache, the GDN tower, benchmarks, GPU work.
* `docs/USAGE.md`'s weights row. The 27B checkpoint's tensor payloads were not
  downloaded during this row, so a row asserting a sha256 and a run would assert
  something that did not happen. It stays owed by the parent row.

## Tests

`tests/vllm/models/test_qwen35_exl3.cpp` (the existing synthetic-checkpoint
fixture) and `tests/vllm/v1/spec_decode/test_mtp_speculator.cpp`.

* **H1** an EXL3 `mtp.*` head LOADS: codebook 2, per-tensor width from the
  trellis, geometry from the tensors.
* **H2** it FORWARDS and agrees with the decoded twin — the SAME trellis bytes
  written as bf16 `.weight` tensors — through `Qwen3_5MTPModel::ForwardPaged` +
  `ComputeLogits`, which is the production draft path.
* **H3** the MTP draft computes against a trellis TARGET head: an EXL3 target
  `lm_head` reaches `ComputeLogits` and agrees with the decoded twin.
* **H4** `SharedHeadSource`'s read (through the exported
  `LoadDflashSharedLmHead`) fills the EXL3 owner and ONLY it, from an EXL3
  target; `lm_head_dequantized` stays false and `RefuseQuantizedDflash2LmHead`
  stays silent.
* **H5** the DSpark shape — no EXL3 owner offered — still refuses an EXL3
  target head by name.
* **H6** the bf16 and NVFP4 MTP/shared-head arms are UNCHANGED.

## Gates

```sh
cmake --build build -j 4 --target test_qwen35_exl3 test_mtp_speculator \
      test_qwen3_dflash2_draft
ctest --test-dir build -R 'qwen35_exl3|mtp_speculator|dflash2_draft' \
      --output-on-failure
```

The exit code is the verdict. A doctest `assertions:` line stays green on a
THROWN case, so `rc` is read and reported beside it.

## Mutation table

Each row asserts the mutated file's sha256 changed, that it COMPILED, that the
test binary's mtime MOVED, and that the tree was restored byte-for-byte.

* M1 the `ComputeLogits` EXL3 branch → `if (false)`.
* M2 the `MtpHeadHidden` fc EXL3 branch → `if (false)`.
* M3 the `LoadDflashSharedLmHead` EXL3 rung → `if (false)`.
* M4 `Qwen3_5MTPWeights::IsExl3()` → `return false`.

## Risks

* **R1 — the published checkpoint is still not RUN.** Its shards finished
  downloading during this row and their headers were read in full, so every
  tensor name, dtype, shape, width and marker quoted here is measured. NO
  forward over those weights was executed, on any device: every rel_rms and
  every gate result in this row is the synthetic fixture on a CPU queue. That
  the arm loads this artifact is inference from its headers, not a run.
* **R2 — no DFlash draft in this tree is EXL3.** Item 6's lane is reached when
  the TARGET's head is a trellis, which is independent of the draft's own
  storage; item 7 is what makes the published draft loadable. Until item 7
  lands, the pairing that exercises this lane on a real artifact is a bf16
  draft against an EXL3 target, which no published checkpoint is.
* **R3 — the CUDA GEMV arm.** `(4,2)`, `(5,2)`, `(6,2)` are instantiated for
  the GEMM. The `m<=8` decode GEMV for the mul1 widths remains owed by
  `quant-exl3-mul1.md`; a device decode of this head therefore takes the GEMM.
  Nothing in this row changes that and no device run is claimed.

## Owed

* `docs/USAGE.md`'s weights row for `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`
  (sha256 + arm), owed by the parent row once the artifact is downloaded and
  run. #2495.
* An EXL3 head owner on `Qwen3_5MoeWeights`, if an EXL3 MoE checkpoint ever
  enters scope. #2495.
* #2495 item 7 (the DFlash2 draft's own EXL3 arm), without which the published
  draft cannot be paired with this target.
* [#2574](https://github.com/mudler/vllm.cpp/issues/2574): `(bits 3,
  codebook 2)` on the CUDA GEMM. 137 of the target's 409 quantized modules are
  three-bit `mul1`, and `Exl3ArmInstantiated` does not carry that pair, so the
  target refuses by name on a device the moment its dense MLP tower is reached.
  NOT this row's: it is owned by `QUANT-EXL3-MUL1`, which owns the codebook-2
  instantiation list. Listed here because it is what stands between this row's
  loaders and an end-to-end CUDA run of the artifact.

## Outcome

Measured on the branch AFTER merging `origin/main` TWICE -- 34 commits, then a
further 35 -- with the second merge's `docs/FEATURES.md` conflict resolved by
taking the target-branch cell whole and re-applying this row's one sentence. The
numbers below were reproduced identically on both merged trees, and the ones
quoted are from the second, whose `origin/main` is `6e9fb55c9`. CPU queue,
`RelWithDebInfo`, `-j 2`. No GPU, no lease, no checkpoint run.

### Gate

| binary | exit code | cases | assertions |
|---|---|---|---|
| `test_qwen35_exl3` | **0** | 10 / 10 passed | 3836 |
| `test_qwen3_dflash2_draft` | **0** | 44 / 44 passed | 1491 |
| `test_mtp_speculator` | **0** | 13 / 13 passed, **2 skipped** | 171 |
| `test_mtp_depth` | **0** | 10 / 10 passed | 123 |

The two skips in `test_mtp_speculator` are `test_run_model_reuses_tensor_return_for_mtp`
and `test_run_model_unpacks_tuple_return_for_mtp`, both carrying a literal
`* doctest::skip(true)` since before this row. Nothing this row added is skipped.

The exit code is quoted beside the counts on purpose: a doctest `assertions:`
line stays GREEN on a thrown case, and every mutation below demonstrates exactly
that — M1 reports `2808 | 2808 passed | 0 failed` while the binary exits 1.

Equivalence against the decoded twin — the SAME trellis bytes read as a bf16
`.weight` and run through the same forward:

| case | rel_rms | bound |
|---|---|---|
| `mtp.*` EXL3 paged logits (item 5) | **0.0107** | 5.0e-2 |
| DFlash shared trellis head (item 6) | **0.00164** | 5.0e-2 |

### Mutations

Each row asserts the mutated file's sha256 CHANGED, that it COMPILED
(`BUILD_RC=0` — a build failure is not a red test), that the test binary's
mtime MOVED, and that the tree was restored byte-for-byte afterwards.

| # | mutation | sha | build | mtime | exit | cases |
|---|---|---|---|---|---|---|
| M1 | `ComputeLogits`'s trellis arm -> `false &&` | dc388ac→ff40a38 | 0 | moved | **1** | 9/10 |
| M2 | `MtpHeadHidden`'s `fc` trellis arm -> `false &&` | dc388ac→4321f93 | 0 | moved | **1** | 9/10 |
| M3 | `LoadDflashSharedLmHead`'s trellis rung -> `false &&` | 7a22f95→289a075 | 0 | moved | **1** | 9/10 |
| M4 | `Qwen3_5MTPWeights::IsExl3()` -> `return false` | 2639fcc→9e3094f | 0 | moved | **1** | 9/10 |
| M5 | the #2569 no-arm throw -> silent `return` | 7a22f95→1de8f50 | 0 | moved | **1** | 8/10 |
| M6 | `DflashLogitsF32D`'s trellis arm -> `false &&` | eb27844→88238d5 | 0 | moved | **1** | 43/44 |

M5 is #2569's RED-BEFORE: with the throw removed, both the DSpark-shape case and
the no-arm-at-all case fail, which is the state the tree was in before this row.

**What M1 and M2 prove, and what they do not.** Both red the same case with the
same message, `vt: matmul: rank-2 tensors required` (`ops.cpp:118`) — an
anonymous shape error, because each mutation drops the arm into a bf16 field an
EXL3 load leaves EMPTY. They establish that each call site is load-bearing; they
do NOT establish that the diagnostic is good, and the gate cannot tell the two
sites apart. That is the same limit `model-qwen35-gdn-exl3.md` recorded for its
own reachability mutation. M4 and M3 do produce named refusals
(`qwen3_5 MTP: unexpected rank for mtp.fc.weight`, and #2569's own message).

### Decisions, and why each default has its value

* **PACKED, not dequantized** (D1). A mirror of upstream, not a choice: the head
  is an `nn.Module` and `_apply_head` calls `lm_head.quant_method.apply`. The
  sizes make it a bad choice independently — 2.543 GB dequantized against
  0.953 GB packed at the real 248320x5120x6-bit — and #1849 added the
  borrow-first seam precisely to stop a copy of this tensor.
* **REQUIRED, not defaulted** (D2). Three owners now, three no-default pointer
  parameters. The alternative has already shipped a silent-off arm here once.
* **Probe per group, never `mtp_bits`** (D4). The artifact's own MLP tower is
  mixed 3/4-bit per layer with one lone 5-bit attention projection, which is
  what makes reading width from the trellis rather than from `bpw` load-bearing
  rather than fastidious.
* **`(3,2)` is NOT this row's** (#2574). Measured here, owned by
  `QUANT-EXL3-MUL1`. The artifact still cannot run end to end on CUDA.

### What was NOT verified

* **The published checkpoint was never RUN.** Its 15.4 GB landed during this row
  and both shards' headers were read in full, so every name, dtype, shape, width
  and marker quoted here is measured. No forward over those weights executed, on
  any device.
* **No device run at all.** Every number above is a CPU queue over a synthetic
  fixture whose head is FOUR bits where the artifact's is six.
* **No speed, latency or memory number is claimed on any axis.**
* The full all-targets build was NOT completed: it ran out of disk while linking
  four unrelated `test_ltx2*` binaries. Zero compile errors were reported before
  that point, so the change compiles tree-wide; the four link failures are
  `No space left on device` and nothing else.

## Stop conditions

* `NEEDS_DECISION` if packed-vs-dequantized turns out NOT to be settled by
  upstream. It is settled (D1), so this did not fire.
* Stop if free disk falls below 8 GB.
