# LTX-2.5 IC-LoRA — the adapter this port could not read, and the second blocker the refusal did not name

**Row:** `LTX25-IC-LORA` (a row of the `#644` full-port campaign).
**Issue:** [#923](https://github.com/mudler/vllm.cpp/issues/923).
**Branch:** `row/LTX25-IC-LORA`.
**Parent spec:** [`ltx-2-5.md`](ltx-2-5.md) — operator-owned, NOT edited by this row.
**Upstream root (primary):** Lightricks/LTX-2 @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
verified at the working checkout `/home/mudler/_git/LTX-2` with a clean tree.
**Entry point ported from:** `packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py`.

vLLM is not the reference here and cannot be: vLLM-Omni carries no LTX-2.5, and
`ICLoraPipeline` has no vLLM analogue at all. The oracle is the `ltx-2` pin
recorded in [`../oracles/`](../oracles/), used as the secondary oracle the rule
allows where vLLM implements nothing.

---

## 0. What is claimed, and what is not

This row builds **the LoRA adapter path**: reading an IC-LoRA safetensors file,
reading its metadata, and fusing its delta into the DiT at load, on every dtype
arm the loader materializes. That is the thing the reference refusal names as
absent, and it is the shared prerequisite for `ic_lora.py`, `hdr_ic_lora.py` and
`dubit.py` alike.

It does **not** ship video-to-video. Three statements up front so nothing is
discovered later.

1. **The reference-video arm has TWO further blockers, and neither is the one
   this spec first named.** The refusal named one cause — the LoRA metadata is
   unread — and that cause was true and is closed here. This spec then rewrote it
   onto the token-APPEND machinery, which was true on 2026-08-15 and false on
   2026-08-16: row `LTX25-TOKEN-APPEND` (#930) landed that seam in `c7cb59fbb`
   and the LAST-frame keyframe is now served on it. §6 is rewritten against the
   current tree and now names what genuinely remains — the reference clip's own
   pixel path, and the stage split that gives stage 2 no adapter — recorded as
   [#975](https://github.com/mudler/vllm.cpp/issues/975). Closing the metadata
   half and lifting the refusal would still ship a wrong render.
2. **No render-quality claim, and no real-weights claim.** The evidence here is
   numeric parity against upstream's fusion arithmetic plus mutation gates on
   synthetic fixtures. No IC-LoRA checkpoint was fused on real weights, because
   this row ran without GPU authority (`dgx.casa` was under a long render).
3. **No speed claim.** The parent spec's `PENDING` speed axis is untouched.

## 1. The gap, re-verified against the current tree

Re-derived at `95b7366`, not read out of the record.

| claim | check | result |
|---|---|---|
| the reference conditioning MATH landed | `Ltx2ConditionVideoByReference`, `src/vllm/model_executor/models/ltx2_conditioning.cpp:221` | present, gated by `test_ltx2_vae` |
| the video VAE ENCODER landed and is reachable | `Ltx2ConvVideoEncode`, called at `src/vllm/multimodal/ltx2_video.cpp:1645` | present, reached |
| `ref_video_dir` reaches the engine | `src/capi/vllm_c.cpp:1642`, `include/vllm/multimodal/video_engine.h:96` | present; a DIR of `frame_%06d.ppm`, so no codec is involved |
| arbitrary safetensors `__metadata__` is readable | `SafetensorsFile::Metadata()`, `include/vllm/model_executor/model_loader/safetensors_reader.h:55-57` | present, returns the whole map |
| anything reads a LoRA file | `grep -rn "lora_A\|lora_B" src include` | **only `include/vllm/lora/` — see below** |
| anything fuses a LoRA into LTX weights | `grep -rn kLoraFusion src include` | **only the refusal, `ltx2_pipeline.cpp:1178`** |

**The tree's existing `include/vllm/lora/` is not this mechanism and this row does
not route through it.** It is vLLM's runtime *punica* subsystem — f32,
slot-indexed, per-token, hanging off `LinearMethodBase` for text decode
(`punica.h:15-24`, `layers.h:26-28`). LTX does not apply an adapter at runtime;
it **fuses at load** (`loader/fuse_loras.py:119-150`). Mirroring vLLM's LoRA here
would mirror the wrong upstream. Recorded as a deliberate divergence from the
shared-seam preference, with the reason: the seam cannot express the behaviour,
which is exactly the case AGENTS.md allows extending or bypassing for.

## 2. Upstream anchors

Every anchor below was re-derived at the final tree and asserted unique; §5.3
records the method and the result.

### 2.1 The pipeline

| what | anchor |
|---|---|
| `ICLoraPipeline.__init__` | `ic_lora.py:71-173` |
| stage 1 takes the LoRAs, **stage 2 takes none** | `ic_lora.py:104-125` (`loras=tuple(loras)` at `:108` against `loras=()` at `:119`) |
| reference scale factors read from LoRA metadata | `ic_lora.py:150-173` |
| conflicting factors across LoRAs raise | `ic_lora.py:158-163` and `:167-172` |
| `__call__` | `ic_lora.py:175-349` |
| `conditioning_attention_strength` range check | `ic_lora.py:230-233` |
| reference appended LAST, after the image conditionings | `ic_lora.py:377-402` |
| the CLI's `--lora` | `utils/args.py:600-611` |

### 2.2 The adapter format and its fusion

| what | anchor |
|---|---|
| `LoraPathStrengthAndSDOps(path, strength, sd_ops)` | `loader/primitives.py:160-167` |
| key shape `<prefix>.lora_A.weight` / `.lora_B.weight` → `<prefix>.weight` | `loader/fuse_loras.py:183-186` and `:196-198` |
| the delta: `sum((B * strength) @ A)` | `loader/fuse_loras.py:99-116` |
| **aggregation dtype is bfloat16** | `loader/fuse_loras.py:71` (`bf16_fuse_rule`) |
| the bf16 fuse: `deltas.add_(weight)` then cast to the weight dtype | `loader/fuse_loras.py:61-68` |
| a LoRA naming a key the model lacks is SKIPPED, not an error | `loader/fuse_loras.py:135-137` |
| metadata `reference_downscale_factor`, default 1 | `iclora_utils.py:30-38` |
| metadata `reference_temporal_scale_factor`, default 1 | `iclora_utils.py:41-49` |

### 2.3 The quantized arms

Upstream carries a fuse rule per policy, and **all four aggregate in bfloat16**:

| arm | rule | anchor |
|---|---|---|
| bf16 | add, cast back | `loader/fuse_loras.py:61-71` |
| fp8 scaled-mm | dequant by `weight_scale`, add in f32, **re-quantize** with a fresh scale | `quantization/fp8_scaled_mm.py:167-189` |
| fp8 cast | fused add-round (stochastic on CUDA+Triton) | `quantization/fp8_cast.py:204-239` |
| NVFP4 | dequant to bf16, add, **re-quantize**, emit weight + both scales | `quantization/nvfp4/fuse.py:13-50` |

## 3. Design

### 3.1 The insertion point, and why every arm is served by one of them

Both quantized arms in this tree **dequantize to bf16 inside one function** before
anything else sees a byte. `MaterializeDitTensor`
(`src/vllm/model_executor/models/ltx2_loader.cpp:424-499`) has four branches —
F32 memcpy, BF16 memcpy, `DequantFp8ToBf16` at `:461-469`, and
`Ltx2DequantNvfp4ToBf16` at `:470-498` — and the last two both `return
vt::DType::kBF16`. The header states the policy outright: "The default
materialization is **bf16**, which is the checkpoint's own model dtype"
(`ltx2_loader.h:76-81`).

So the fusion hook goes **immediately after `MaterializeDitTensor` returns**, and
one hook covers FP8, NVFP4, BF16 and F32. It is reached from both callers:
`Ltx2LoadDitFromSafetensors` (`ltx2_loader.cpp:614-621`) and
`Ltx2StreamDitToDevice` (`:667-684`). On the device arm it runs *before*
`backend.Copy`, which preserves that arm's stated invariant that one host buffer
is live at a time (`:668-670`).

**This is a deliberate divergence from upstream's per-arm rules, and it is forced
by an existing design rather than chosen here.** Upstream re-quantizes because it
keeps FP8/NVFP4 weights resident for its quantized kernels; this tree does not
keep them, so there is nothing to re-quantize into. The consequence is recorded
rather than hidden: our fused weight **skips upstream's lossy quantize round
trip** and is therefore slightly more precise than upstream's on the FP8 and
NVFP4 arms. It costs no extra bytes — the weight was already bf16 — so this is
not the too-wide-dtype failure that rule guards against. The tree has no FP8 or
NVFP4 *quantizer* at all (`grep` for one returns nothing; only `DequantFp8ToBf16`
and `Ltx2DequantNvfp4ToBf16` exist), so mirroring the round trip is not available
to be chosen.

### 3.2 The memory format

Mirrored deliberately, per `.agents/porting.md` §"Mirror the memory format".

| ask | upstream answer | here |
|---|---|---|
| what dtype does the delta accumulate in? | **bfloat16** — every one of the four `FuseRule`s sets `aggregation_dtype=torch.bfloat16` | bf16 |
| is the first product rounded differently from the rest? | yes: `matmul(B * strength, A).to(dtype)` first, `addmm_(B, A, alpha=strength)` after (`fuse_loras.py:110-116`) | mirrored |
| what dtype is the fused weight stored as? | the weight's own dtype | the materialized dtype, unchanged |

An f32 accumulator would be the exact defect that rule exists for: the tokens
would still match and every golden would still pass. It is bf16 here on purpose,
and §5.2 mutates that choice to prove the gate can see it.

### 3.3 Surface

- Load extras `lora_path` and `lora_strength` on the `ltx-2.5` family, mirroring
  upstream's `(path, strength)` pair. LoRAs are a **constructor** argument
  upstream (`ic_lora.py:104-114`), not a `__call__` argument, so a load extra is
  the faithful shape and a per-generation field would not be.
- `--lora PATH [STRENGTH]` on `ltx2-gen`, mirroring `utils/args.py:600-611` and
  its `DEFAULT_LORA_STRENGTH`.
- No ABI change. Both extras ride the existing `extra_keys`/`extra_values`
  parallel arrays, which `include/vllm.h:926-927` records as existing for exactly
  this.
- **Exactly one LoRA**, with a second refused by name. Upstream's own `dubit.py`
  enforces the same (`dubit.py:364-365`) and `hdr_ic_lora.py` takes exactly one
  (`hdr_ic_lora.py:271-272`). N-LoRA fusion is recorded under `## Owed` rather than
  half-built; the conflict-detection loop this row ports (`ic_lora.py:155-173`)
  is what N-LoRA needs and is written to take a list already.

### 3.4 `kLoraFusion` stops being a marker

`Ltx2UnportedPipelineFeature::kLoraFusion` is classified as a
declared-out-of-scope marker and its message asserts `DECLARED, NOT REQUESTABLE`
(`ltx2_pipeline.cpp:1178-1181`). Once `lora_path` exists that sentence is false.
Issue [#691](https://github.com/mudler/vllm.cpp/issues/691) records that the
ledger test gates the message **text** and not the property, and predicts this
exact divergence in its own words. So the enumerator is **removed**, not
reclassified: there is no longer an unported LoRA-fusion feature to name, and a
refusal for a served capability is worse than no refusal.

## 4. Risks

| risk | mitigation |
|---|---|
| the fused delta lands on the wrong tensor because LoRA keys carry a ComfyUI prefix the contract strips | normalize onto `Ltx2TensorSpec::name` explicitly and **refuse by name** when a LoRA names a module the contract lacks after normalization, rather than silently skipping — see §4.1 for where this deliberately departs from upstream |
| a LoRA that touches nothing at all fuses green and renders identically | the adapter load refuses when **zero** contract tensors matched; a no-op LoRA is a user error, not a successful load |
| editing `ltx2_video.cpp` above line 1100 shifts the gated READER ANCHORS | the new readers go below the existing ones where possible; the anchors are re-derived at the final tree and the gate is run |
| `docs/FEATURES.md` is a keyed record other agents are editing concurrently | reapply by key and diff unrelated keys byte-for-byte before commit |

### 4.1 One deliberate departure

Upstream **skips** a LoRA key whose target weight is absent
(`fuse_loras.py:135-137`, `if original_weight is None: continue`). It can afford
that: its state dict is the whole model. Here the contract is a fixed enumerated
set with unported modules already stripped, so a skip would silently absorb both
a genuinely-inapplicable key *and* a misnamed one, and the second is the failure
this project keeps paying for. The load therefore refuses by name on an unmatched
key. This is a divergence, it is argued here and in the commit, and it is the
narrower behaviour.

## 5. Tests and gates

### 5.1 Ported from upstream — there is nothing to port, and that is measured

The rule is to port the upstream tests in the same change. **The pinned upstream
ships none, anywhere in the repository**, so the obligation is discharged as
not-applicable rather than skipped. Measured at
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, with a positive control so a null
result cannot be a wrong search term:

| probe | result |
|---|---|
| `find . -name 'test_*.py' -o -name '*_test.py'` | **0** |
| `find . -type d -name 'test*'` | **0** |
| `find . -name conftest.py` | **0** |
| `grep -rl "import pytest\|import unittest" --include='*.py' .` | **0** |
| *control:* `find . -name '*.py'` | **280** |

The control is what makes the zeros admissible: the same traversal that found no
tests found 280 Python files. So the fusion arithmetic is gated against
upstream's expression read from source (`fuse_loras.py:99-116`) and executed by
hand, which is stated as what it is — a source-derived value gate, not a ported
test.

### 5.2 Written here, each RED first

| case | proves |
|---|---|
| the fused weight equals `W + (B*strength) @ A` | the arithmetic, against values computed from the upstream expression |
| the accumulator is bf16, not f32 | a rank-2 adapter whose f32 and bf16 accumulations differ in the stored bf16 result; **this is the memory-format gate** |
| `strength` scales the delta linearly | strength is read, not ignored |
| a LoRA naming an absent module refuses by name | §4.1 |
| a LoRA matching zero tensors refuses by name | §4 |
| the metadata factors are read, absent ⇒ 1 | `iclora_utils.py:30-49` |
| **the matmul RESULT is rounded to bf16 before the weight is added** — rank 2, `acc = 1 + 2^-8` on a bf16 tie, `w = 2^-9`, so a bf16 aggregator stores 1.0 and an f32 one stores 1.0078125 | upstream's `.to(dtype=aggregation_dtype)` at `fuse_loras.py:113`. This is the THIRD rounding the aggregation dtype binds; widening only it left `test_ltx2_lora` 13/13 and `test_ltx2_loader` 31/31, which is what made the case necessary |
| **the reference refusal may not name a seam this engine has** — the case measures `video_tokens` with and without an appending item, and only then forbids the message from blaming the loop | the repair for the review finding that five substring assertions could not go red when the ENGINE changed |
| a second `lora_path` refuses by name | §3.3 |
| **the FP8 arm fuses** — an FP8 fixture plus a LoRA changes the materialized weight | that the hook is after dequant, on the arm most users run |
| **the NVFP4 arm fuses** | the same for NVFP4 |

### 5.3 Reachability

The production entry point is `vllm_video_engine_load` → `LoadVideoEngine` →
`Ltx2VideoEngine::Load` → `Ltx2LoadDitFromSafetensors`. The smallest failing test
enters **there**, through `LoadVideoEngine` with a `lora_path` load extra on the
engine fixture, and asserts the rendered conditioning digest moves. The
reachability mutation deletes the fusion call site in `ltx2_loader.cpp` and
re-runs that case; a green gate would be the finding.

### 5.4 Gate

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Known-red on `main` at the base SHA `95b7366`, proven pre-existing by running
`scripts/agent-preflight.sh` on a byte-for-byte unmodified tree
(`git status --porcelain` empty) before any edit: the seven-checker `#873` family
— `check-release-binary-contract`, `check-release-workflow`,
`check-test-registration`, `test_check_release_binary_contract`,
`test_release_manifest`, `test_release_pipeline`, `test_check_test_registration`.

## 6. What still blocks reference video, re-derived at `c90e3fc02`

**This section was wrong once and is rewritten rather than patched.** Its first
version named the token-APPEND machinery, which row `LTX25-TOKEN-APPEND` (#930)
then landed in `c7cb59fbb`. The determination below is re-derived against the
merged tree, not read out of the earlier record, and it is
[#975](https://github.com/mudler/vllm.cpp/issues/975).

**Ruled OUT, with what ruled each one out.**

- **The DiT is not in the way.** `Ltx2ModalityInput::tokens` is a per-call field,
  and the DiT already accepts a self-attention strength mask.
- **The conditioning item is not in the way.** `Ltx2ConditionVideoByReference`
  (`ltx2_conditioning.cpp:221`) is ported and gated.
- **The phase loop is no longer in the way.** `c7cb59fbb` binds a `target_tokens`
  local, grows `video.tokens` past it on an appending item, carries the grown
  count through denoise, and trims back through `Ltx2ClearConditioning`
  (`ltx_core/tools.py:88-117`) before unpatchify. The LAST-frame keyframe arm is
  SERVED on exactly that machinery, which is the executable proof it exists.
- **`Ltx2LatentState` having no attention-mask field is not in the way for the
  DEFAULT arm.** At `conditioning_attention_strength >= 1.0` with no latent mask,
  upstream sets `attn_mask = None` (`iclora_utils.py:159-160`) and applies
  `ConditioningItemAttentionStrengthWrapper` only `if attn_mask is not None`
  (`:168-169`). The sub-1.0 arm is #932 and is not what blocks this one.

**What IS in the way.**

1. **The reference clip has no pixel path.** Upstream reads the reference at
   `height // scale` by `width // scale` (`iclora_utils.py:116-117`), refuses a
   target either axis of which the factor does not divide (`:112-115`), keeps
   frame 0 and then every Nth frame (`temporal_subsample`, `:87-89`, called at
   `:144`), and encodes the whole clip (`:145-148`). This engine's only
   pixel-to-latent route is `Ltx2LoadImageAndPreprocess` followed by
   `Ltx2ConvVideoEncode` at `frame_count = 1` and the phase's OWN height and
   width, and it refuses an encode returning more than one latent frame. Nothing
   anywhere reads `ref_video_dir` (`src/vllm/multimodal/video_engine.cpp:375`),
   which is a directory of `frame_%06d.ppm`. `Ltx2ConvVideoEncode` already takes
   a `frame_count`, so the encoder is not the gap; the reader, the resize target
   and the subsample are.
2. **The reference item is a STAGE-1 item, and stage 2 must run UNFUSED.**
   `ICLoraPipeline` builds two `DiffusionStage`s from one checkpoint and gives
   stage 1 `loras=tuple(loras)` (`ic_lora.py:108`) and stage 2 `loras=()`
   (`:119`); stage 1 takes `_create_conditionings`, which appends the reference
   item (`:269-278`, `:377-402`), and stage 2 takes plain
   `combined_image_conditionings` with no reference item (`:314-321`). This
   engine holds ONE `Ltx2Dit`, fused at load, that every phase of the recipe
   runs. Serving the arm on the two-phase distilled recipe therefore needs a
   second unfused DiT or a phase-scoped adapter, and serving it on phase 0 alone
   is upstream's `skip_stage_2` (`:302-308`), a different request.

Piece 2 changes how the engine HOLDS its DiT, not how it conditions, so this is
its own row rather than a tail of this one. It also carries a memory decision a
conditioning change does not: a second resident DiT is ~21 B parameters.

## 6.1 What `hdr_ic_lora` and `dubit` would need from this seam

Both were read (not ported) so the adapter path would not have to be rewritten
for them. Neither is in scope; this records what they would ask for, so a later
row can tell an extension from a redesign.

**What this row already gives them.** Both fuse a LoRA at load, and both take
exactly one — `dubit.py:364-365` refuses any other count and
`hdr_ic_lora.py:271-272` hard-codes a single adapter at strength 1.0. So
`Ltx2LoraSpec` + `Ltx2DitLoadOptions::loras` is the right shape for both as it
stands, and neither needs the N-adapter work owed below.

**What they would need added.**

| need | who | why the current shape does not cover it |
|---|---|---|
| the raw `__metadata__` map, not just two typed factors | `hdr_ic_lora` | it reads `hdr_transform` and `use_hdr_transform` (`hdr_ic_lora.py:201-208`) and does NOT use `iclora_utils`' readers at all. `Ltx2LoraAdapter::metadata()` already returns the whole map, so this is **already covered** — which is why the accessor exposes the map rather than the two ints |
| per-stage adapter sets | both | `ic_lora.py:115-119` gives stage 2 `loras=()`, `hdr_ic_lora.py:293-312` gives both stages the same adapter, and `dubit` reuses ONE stage for both. Nothing here hard-codes "stage 2 has no LoRA" — the adapter is a load option on one DiT — so a two-DiT pipeline would carry two `Ltx2DitLoadOptions`. No change to this file |
| an audio conditioning item with negative RoPE positions | `dubit` | `AudioConditionByReferenceLatent` is ported (`ltx2_conditioning.h`), but `dubit.py:335-354` shifts positions by `-aud_dur - 0.04`, and the audio VAE **encoder** has no load path here at all (the reference-audio refusal says so). Unrelated to the LoRA seam |
| a frozen modality and cross-stage latent carry | `dubit` | `ModalitySpec{frozen, noise_scale, initial_latent}` (`dubit.py:321-327`). A pipeline concern, not an adapter one |
| tiled diffusion, per-tile seeds, conditioning slicing | `hdr_ic_lora` | `hdr_ic_lora.py:485-596`. Its stage 2 is a phase LIST with per-phase tiling and a `use_ic_lora` flag, and it SLICES the reference conditioning per tile, dividing the slice indices by `reference_downscale_factor` (`:567-568`). That factor is the one this row reads, so the seam feeds it correctly; the slicing itself is pipeline work |
| a text-context provider that is not the Gemma tower | `hdr_ic_lora` | it loads `video_context` / `audio_context` from a safetensors file and runs no text encoder (`hdr_ic_lora.py:274-281`) |

**The conclusion that matters:** nothing either file needs would change the
adapter reader or the fusion hook. Their requirements land on the pipeline and
the conditioning state, not here. The one place the seam was deliberately shaped
for them is `metadata()` returning the whole map rather than the two factors
`ic_lora.py` happens to want.

## Owed

Each is owed by this row and named in the commit and pull request bodies.

| owed | issue |
|---|---|
| serving the reference-image and reference-video arms: the reference CLIP's own pixel path (read, resize to `height // scale`, temporal subsample, multi-frame encode) and the stage split that gives stage 2 no adapter. §6 derives both. Token-append is NOT part of it any more — #930 landed in `c7cb59fbb` | [#975](https://github.com/mudler/vllm.cpp/issues/975) |
| the `conditioning_attention_mask` / `conditioning_attention_strength < 1.0` arm, which needs `Ltx2LatentState` to carry a mask and `build_attention_mask`'s block structure (`mask_utils.py:170-243`) | [#932](https://github.com/mudler/vllm.cpp/issues/932) |
| N-adapter fusion, which additionally needs upstream's SECOND rounding pattern (`addmm_` with `alpha`, `fuse_loras.py:115`) that this row refuses rather than guesses | [#932](https://github.com/mudler/vllm.cpp/issues/932) |
| GGUF k-quant LoRA fusion — **not applicable** rather than owed: the LTX-2.5 DiT ships FP8 and NVFP4, and no GGUF LTX DiT exists to fuse into | n/a |
| a real-weights IC-LoRA fusion measurement | blocked on GPU authority; `dgx.casa` was under a long render for this row's duration and this row had no GPU authority |

## 8. Stop conditions

- Stop and report `NEEDS_DECISION` if closing the metadata half would require
  lifting the reference refusal before the machinery it needs exists. **This
  fired twice.** The first time it named token-append; §6's first version was the
  result. The second time, in review repair, that cause had itself landed
  (`c7cb59fbb`) and the determination had to be made again from the tree. §6 is
  rewritten, the refusal is rewritten onto the two causes that remain, and
  serving the arm is reported as `NEEDS_DECISION` under
  [#975](https://github.com/mudler/vllm.cpp/issues/975) because a second resident
  DiT is a change to how the engine holds its weights, not a conditioning change.
- Stop if `docs/FEATURES.md` cannot be reapplied by key with unrelated keys
  byte-identical.
- Do not use the GPU. `dgx.casa` was under a long render for this row's duration.

## Now

`ACTIVE` — the adapter path is implemented and gated; the reference arm stays
refused on the two causes §6 derives, which are #975 and are neither of the two
this refusal has previously given.
