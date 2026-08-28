# Vision-tower dtype polarity — a bf16 checkpoint held in host f32

Issue: [#1359](https://github.com/mudler/vllm.cpp/issues/1359).
Row: `ENG-MM-INPUT-PIPELINE` ([engine-matrix.md](../engine-matrix.md)).
Index: [`issue-index.md`](../issue-index.md) line 649.
Sibling spec: [`multimodal-track.md`](multimodal-track.md) §1.5 L3 owns the
tower-SKIP measurement; this spec owns the tower's storage DTYPE. §8 below
states exactly what this row makes stale over there.

This spec is written BEFORE the implementation, and every threshold in §6 is
declared before any number exists. Do not edit a threshold after reading a
result.

## 0. The one-line answer

**Storage-only.** Every vision tower in this tree narrows its host f32 weights
back to bf16 before the first GEMM, so the f32 storage buys nothing and costs
exactly 2x the checkpoint's bytes. Narrowing the storage to bf16 is
**bit-identical**, not merely within tolerance. The fix is a load-path change
and a near-pure win; there is no memory-for-latency trade to weigh, because the
narrowing pass the fix deletes is work we are doing today.

The one exception is small, named, and deliberately kept: `pos_embed_w` (§4.3).

## 1. Scope

**In scope.** The host storage dtype of ported *vision* tower weights, and the
loaders that produce them:

| Tower | Weight struct | Loader that widens | Serves |
|---|---|---|---|
| Qwen3-VL (safetensors) | `include/vllm/model_executor/models/qwen3_vl_vision.h:60-82` | `LoadVisionF32`, `src/vllm/model_executor/models/qwen3_vl.cpp:79-90` | Qwen3-VL-4B, Qwen3.5/3.6-27B dense, Qwen3.6-35B MoE |
| Muse Glimmer | `include/vllm/model_executor/models/muse_glimmer_vision.h:97-119`, `muse_glimmer.h:287` | `Bf16TensorToF32`, `src/vllm/model_executor/models/muse_glimmer_weights.cpp:651-675` | Muse Glimmer 30B |
| CLIP mmproj (GGUF) | shares `Qwen3VLVisionWeights` | `src/vllm/model_executor/models/clip_mmproj_gguf.cpp:199-205` | `--mmproj` engine path |
| MiniMax-H3 vision (GGUF) | shares `Qwen3VLVisionWeights` | `src/vllm/model_executor/models/minimax_h3_vision_gguf.cpp:63-70` | MiniMax-H3 |
| Gemma-4 vision | `include/vllm/model_executor/models/gemma4_vision.h:77-93` | none in `src/` — test-filled only | unreached today |
| Engine holder | `include/vllm/entrypoints/model_loader.h:730` `vision_tower_` | `src/vllm/entrypoints/model_loader.cpp:2590-2591` | `vllm-server` |

**Adjacent, same defect class, NOT authorized by #1359's current title.** The
Whisper/Voxtral *audio* encoder has the identical shape — host f32 storage
(`include/vllm/model_executor/models/whisper_audio.h:68-99`), widened by
`VoxtralStBf16ToF32` (`src/vllm/model_executor/models/voxtral.cpp:381-390`),
computed in bf16 via `ResidentBf16`
(`src/vllm/model_executor/models/whisper_audio.cpp:113-135`). §7 proposes the
scope correction; until the developer rules on it, the audio tower is recorded
here and not changed.

**Out of scope, with a reason.** Gemma-4 *audio*
(`src/vllm/model_executor/models/gemma4_audio.cpp:26-130`) runs scalar host
loops in f32 with no bf16 narrowing anywhere: `Linear` `:26-40`, `ClipLin`
`:43-51`, `RmsNorm` `:53`, `Conv2dK3S2P1` `:92`. There the f32 host values
genuinely ARE the compute dtype. It is shape 2 (§3), and narrowing it is a
different piece of work with a different argument. Likewise the MiniMax-H3
video VAE (`minimax_h3_video_vae_device.cpp:67-80`), Parakeet
(`parakeet_encoder.cpp:113`, `parakeet_transducer.cpp:85`) and Kimi's `WF32`
(`kimi_linear_device.cpp:292`), each of which allocates `DType::kF32` on device.

**Also out of scope.** Removing the tower load entirely (#607 L3, landed) and
the never-read-back tower on the engine (#1358). Both are orthogonal: L3
removes the tower, #1358 says nothing reads it, and this row narrows what is
read when it IS read.

## 2. Upstream anchors, at the parity pin

Pin `5559679229bc961848b121ccdeaa8fa5d79bec98`
([upstream-sync.md](../upstream-sync.md) ` ```parity-pin `), read from the local
checkout at that exact SHA.

- `vllm/model_executor/models/qwen3_vl.py:633-634` — the tower has no dtype of
  its own. `Qwen3_VisionTransformer.dtype` is *defined as*
  `self.patch_embed.proj.weight.dtype`. Whatever the checkpoint loaded as, the
  tower is.
- `vllm/model_executor/model_loader/base_loader.py:53` —
  `with set_default_torch_dtype(model_config.dtype):` wraps the whole model
  construction. There is no ViT-specific override anywhere in the tree; a grep
  for `encoder_dtype`, `vit_dtype`, `vision_dtype` over `vllm/` returns nothing.
- `vllm/model_executor/models/qwen3_vl.py:614-616` — the ViT attention backend
  is even *selected* on `torch.get_default_dtype()`, so the ViT running in the
  model dtype is baked into the dispatch, not only the storage.
- `vllm/model_executor/models/qwen3_vl.py:807` — inputs are cast INTO the
  weight dtype (`x.to(device=self.device, dtype=self.dtype, ...)`), never the
  reverse.
- `vllm/model_executor/models/qwen3_vl.py:2179,2181,2201,2204` — pixel values
  and precomputed embeds are cast to `self.visual.dtype` on both the image and
  the video path.

**Upstream's one f32 use, and it is narrower than ours.**
`vllm/model_executor/models/qwen3_vl.py:298-310` computes the pos-embed bilinear
*coefficients* (`h_idxs`, `w_idxs`) in `torch.float32` — then `:335`
(`weights = weights.to(dtype=dtype)`) casts them DOWN to the model dtype before
`embeds *= weights`, the gather at `:337` reads a model-dtype `nn.Embedding`,
the accumulation at `:339` is model-dtype, and `:344` returns `.to(dtype=dtype)`.
So upstream stores nothing in f32; it computes four scalars per axis in f32 and
immediately narrows. Our host interpolation runs the whole gather and sum in
f32, which is *wider* than upstream (§4.3, §5 risk 2).

**Conclusion.** vLLM resolves one model dtype and the perception encoder
inherits it. AGENTS.md §"Inherit vLLM defaults" makes an `f32` value on the
model path a rare, annotated exception. Our vision towers are 100% f32 with no
annotation. The mirror target is unambiguous and this is not a product decision.

## 3. The feasibility answer, with evidence

The question was: can our vision compute consume bf16, or does the f32 storage
exist because the kernels require it? Three shapes were possible. **The answer
is shape 1 — storage-only — on every vision path in §1.**

**Qwen3-VL family.** `PrepareVisionDeviceWeights`
(`src/vllm/model_executor/models/qwen3_vl_vision.cpp:300-335`) puts every
weight through `MakeDevBf16` (`:132-152`), which calls `ToBf16` (`:86-90`) and
allocates `DType::kBF16`. Per-weight call sites `:308-326`. The compute overload
`:374-378` takes only the device weights, and pixels arrive already bf16
(`const std::vector<uint16_t>& pixel_values_bf16`,
`include/vllm/model_executor/models/qwen3_vl_vision.h:100`). The host-weights
overload `:609-630` is a thin prepare-then-forward wrapper over the same path.
There is no f32 GEMM.

**Muse Glimmer.** `Upload`
(`src/vllm/model_executor/models/muse_glimmer_vision.cpp:116-124`) narrows when
`dt == DType::kBF16`, and `dt` is `cfg.compute_dtype`
(`:444`, `:725`), which defaults to `vt::DType::kBF16`
(`include/vllm/model_executor/models/muse_glimmer_vision.h:86`). Every weight
goes through it: patch/pre-norm `:521-527`, all 50 blocks `:560-576`, `ln_post`
`:691-693`, adapter `:729-730`. The projection narrows in the same way at
`src/vllm/model_executor/models/muse_glimmer_mm.cpp:134-145`.

**GGUF towers.** `clip_mmproj_gguf.cpp:199-205` and
`minimax_h3_vision_gguf.cpp:63-70` dequantize a k-quant row to f32 via
`DequantGgufRowToF32` (`src/vllm/model_executor/model_loader/gguf_dequant.cpp:51`),
then hand the result to the same `Qwen3VLVisionWeights` and the same
`MakeDevBf16`. The f32 is a pure carrier between two conversions.

**Gemma-4 vision.** `MakeDevBf16` at
`src/vllm/model_executor/models/gemma4_vision.cpp:106-125`, applied to the whole
weight set inside the forward at `:250-286`.

### 3.1 Why the narrowing is BIT-IDENTICAL, not "within tolerance"

This is the claim that makes the fix safe, and it is mechanical.

Every safetensors vision tensor is asserted BF16 at the door —
`src/vllm/model_executor/models/qwen3_vl.cpp:82`
(`VT_CHECK(t.dtype == "BF16", ...)`) and
`src/vllm/model_executor/models/muse_glimmer_weights.cpp:660`. So every stored
f32 value is `BF16ToF32(b)` for some `b`.

- `src/vt/dtype.cpp:317` — `BF16ToF32(b)` is `AsF32(uint32_t(b) << 16)`. Exact
  and injective: the low 16 mantissa bits are zero by construction.
- `src/vt/dtype.cpp:319-326` — `F32ToBF16(f)` adds
  `0x7FFF + ((u >> 16) & 1)` and shifts right by 16. For `u = b << 16` the low
  16 bits are zero, so the addend is at most `0x8000` and the carry never
  reaches bit 16. Therefore `F32ToBF16(BF16ToF32(b)) == b` for every `b`.

So on the safetensors paths, replacing "widen at load, narrow at upload" with
"store as loaded, copy at upload" produces the **same device bytes**. On the
GGUF paths the argument differs and must be stated separately: there the value
is `F32ToBF16(dequant_f32)` either way, because the single narrowing simply
moves from upload time to load time. Same function, same input, same output —
provided the implementer reuses `vt::F32ToBF16` rather than writing a second
conversion. Risk 1 in §5 is exactly this.

### 3.2 What the fix removes, which is why §6.3 is a guard and not a budget

Today `MakeDevBf16` / `Upload` allocate an N-element `std::vector<uint16_t>`
and run an N-element `F32ToBF16` pass **per weight, per upload**. With bf16
storage that becomes a direct `Backend::Copy` from the stored bytes. Two towers
pay this per image rather than once, so they gain the most:

- Muse Glimmer has **no** prepare-once path at all — the file says so itself at
  `src/vllm/model_executor/models/muse_glimmer_vision.cpp:25-29` — and
  re-converts and re-uploads the whole 3.58 GiB tower on every forward.
- Gemma-4 vision marshals the whole weight set inside `Gemma4VisionForward`
  (`src/vllm/model_executor/models/gemma4_vision.cpp:250-286`).

The fix therefore cannot cost latency by its own mechanism. It can only remove
a pass. §6.3 measures that prediction rather than assuming it.

## 4. Design

### 4.1 The storage change

Replace `std::vector<float>` with a 16-bit bf16 store on every field of the
structs in §1 (`VisionBlockWeights`, `VisionMergerWeights`,
`Qwen3VLVisionWeights`, `MuseGlimmerVisionBlockWeights`,
`MuseGlimmerVisionWeights`, `MuseGlimmerVisionAdapterWeights`,
`MuseGlimmerVisionTower::projection`), except `pos_embed_w` / `pos_emb` (§4.3).

The repository already has the shape: `std::vector<uint16_t>` raw bf16 bits, as
`Qwen3VLVisionForward`'s own `pixel_values_bf16` parameter uses
(`include/vllm/model_executor/models/qwen3_vl_vision.h:100`) and as
`F32ToBf16Bits` produces (`src/vllm/model_executor/models/qwen3_vl.cpp:103-107`).
Use it rather than inventing a wrapper. `OwnedTensor` is the other candidate and
is what the *text* towers use, but it carries device-residency semantics these
host structs do not have, and swapping it in would enlarge the diff past what
the gate in §6 can attribute.

### 4.2 The loader change

`LoadVisionF32` and `Bf16TensorToF32` stop calling `vt::BF16ToF32` and copy the
bf16 bits straight through. Both must keep reading via `vt::LoadUnaligned`
(`qwen3_vl.cpp:87`) or `std::memcpy`
(`muse_glimmer_weights.cpp:670-673`): the safetensors payload offset carries no
alignment guarantee (#772), and `tests/vllm/models/test_loader_unaligned_offsets.cpp:247`
pins `LoadVisionF32` specifically. A `reinterpret_cast<const uint16_t*>` plus an
indexed read is the regression that test exists to catch.

Rename both functions in the same change (`LoadVisionBf16`,
`Bf16TensorToBf16Bits`), so a reader cannot be told `F32` by the name while the
type says otherwise.

The GGUF loaders keep `DequantGgufRowToF32` and narrow once with
`vt::F32ToBF16` at the end of the `load` lambda
(`clip_mmproj_gguf.cpp:204`, `minimax_h3_vision_gguf.cpp:67`).

### 4.3 `pos_embed_w` stays f32, and carries its reason

`src/vllm/model_executor/models/qwen3_vl_vision.cpp:310` already keeps
`dw->pos_embed_w` host f32 "for per-grid interp", and
`VisionPosEmbedInterpolate` / `MuseGlimmerVisionPosEmbedInterpolate` run the
bilinear gather and sum on the host in f32. This is the one field whose f32
values reach arithmetic without being narrowed first.

Keep it f32. Add the annotation AGENTS.md requires, naming the reason: the
interpolation is a host-side precompute of a table vLLM computes on the GPU, and
running the gather in f32 keeps the precompute out of the tolerance budget the
tower golden is stated against. Cost: 9,437,184 B on Qwen3-VL (0.57% of the
tower) and 3,145,728 B on Muse Glimmer (0.08%).

**Record it as a divergence, not as blessed.** §2 shows upstream gathers in the
model dtype (`qwen3_vl.py:335,337,344`), so our f32 interp is wider than the
mirror. Narrowing it to match would change tower numbers and is not what #1359
asks for. It belongs in `## Owed` with its own issue, not inside this change.

### 4.4 What must NOT change

`MuseGlimmerVisionConfig::compute_dtype` keeps its `kF32` arm
(`muse_glimmer_vision.h:83-86`), and `Upload`
(`muse_glimmer_vision.cpp:116-124`) must grow the widening branch: with bf16
storage and `dt == kF32` it widens back. That path stays bit-identical to today
because the values originated bf16 (§3.1). Dropping the branch would leave the
f32 per-stage gate silently comparing bf16-computed numbers against an f32
reference, absorbed by its own tolerance.

## 5. Risks

1. **A second rounding function.** If the implementer writes a truncating
   narrow instead of reusing `vt::F32ToBF16`, the GGUF towers change numbers and
   no token gate sees it. Gated by §6.4's exact-equality arm and its mutation.
2. **Narrowing `pos_embed` by accident.** On the safetensors path that is still
   bit-identical; on the GGUF path it is not, because the dequant result is a
   genuine f32 that would lose bits before the host interp. §4.3 keeps it f32
   deliberately; a sweeping type change over the struct would take it silently.
3. **The `kF32` arm.** §4.4.
4. **A designed red — WHICH DID NOT FIRE, and that is the finding.**
   `test_spec_carries_both_resident_figures` asserted that `resident(kind)`
   appears in `multimodal-track.md`, and was meant to force that document to move
   with the `* 2`. It could not. Post-fix `resident("qwen3-vl")` collapses onto
   `ondisk("qwen3-vl")`, and `830695424` was ALREADY in that file as the ON-DISK
   figure — so the assertion became a tautology satisfied by a sentence about a
   different quantity. It measured 60/60 green while `multimodal-track.md` went
   on declaring the pre-fix threshold `1495251763 B`, a number a post-#1359 run
   cannot reach, against which a correct change would have been recorded as a
   FAILING axis.

   The repair replaces it with
   `test_spec_carries_the_threshold_the_instrument_applies`, which asserts
   `need(kind) = resident(kind) * MIN_SAVING_FRACTION_PCT / 100` — the quantity a
   run is judged against, present in the spec for no other reason, and therefore
   unable to collapse onto an existing figure the way `resident` did. It is
   scoped to the half-1 DECLARATION rather than to the file, because the spec
   legitimately retains superseded thresholds as the record of runs measured
   against them, and an unscoped search has a false negative there: with the
   widening mutated back to `2` on both the script and the suite's mirror,
   `need("qwen3-vl")` becomes `1495251763`, a number the file still carries as
   the 2026-08-24 figure, and the unscoped form returned OK. Measured, then
   fixed. It still asserts `resident(kind)`, but only where that is not the
   on-disk figure, so the vacuous case is a stated condition rather than
   something that looks like a pass.

   RED-first and mutation evidence, all four restored byte-for-byte by
   `sha256sum -c`:

   | tree | result |
   |---|---|
   | the committed spec, still declaring `1495251763` | **RED**, both kinds, `747625881` and `6918644736` absent |
   | the repaired spec, threshold mutated back to `1495251763` | **RED**: "carries that number in 0 of its 2 half-1 declarations" |
   | the same mutated tree, under the RETIRED assertion re-created verbatim | **GREEN** — which is why it never fired |
   | `WIDEN["qwen3-vl"]` back to `2` in script and mirror, spec untouched | **RED** (unscoped form: green) |

   The general shape is worth keeping: an assertion whose expected value is
   derived from the same quantity the document already states elsewhere stops
   being an assertion the moment those two quantities become equal, and nothing
   in a green run says so.

   The script's `TOWER_RESIDENT_BYTES` (`scripts/mm/tower_skip_rss.sh:451-461`)
   and the test's `resident()` are now per-kind rather than a blanket `* 2`. All
   four surfaces still move in ONE commit or the suite is red. §8.
5. **Peak RSS is not allocation size.** The measurement is
   `/usr/bin/time -v` peak (`scripts/mm/tower_skip_rss.sh:576-581`), so the
   mmap'd source pages and `MaybeReleaseSourcePages` timing sit inside it. If
   §6.1 reads below its band, that is the first cause to check — not a reason to
   move the band.
6. **Unaligned reads.** §4.2.
7. **Blast radius across three model families.** `LoadQwen3VLVisionWeights` is
   shared by Qwen3-VL-4B (`qwen3_vl.cpp:431`), the 27B dense path and the 35B
   MoE path (`qwen3_5_weights.cpp:1757-1771`). One struct change moves all
   three, which is the point — and it means every one of them needs its tower
   golden rerun, not only the vehicle.

## 6. Gates

Correctness before performance. The RSS axis is the reason this issue exists,
because a token gate cannot see a dtype that is too wide — the tokens match and
the goldens pass while the path moves twice the bytes.

### 6.1 The memory gate — DECLARED HERE, BEFORE ANY NUMBER EXISTS

Derived from the checkpoints' own safetensors headers, read 2026-08-28 with a
header-only scan (`8 + header_bytes` prefix; no payload read), exactly as the
existing L3 thresholds were derived.

| Checkpoint | Vision tensors | On disk (all BF16) | Resident TODAY (f32) | Resident AFTER |
|---|---:|---:|---:|---:|
| `qwen3-vl-4b-instruct`, `model.visual.*` | 315 | 830,695,424 B (0.7736 GiB) | 1,661,390,848 B (1.5473 GiB) | 830,695,424 B |
| `muse-glimmer-30b`, `model.vision_{tower,adapter,projection}*` | 809 | 3,843,691,520 B (3.5797 GiB) | 7,687,383,040 B (7.1594 GiB) | 3,843,691,520 B |

Muse Glimmer's 809 decompose as 806 `vision_tower` + 2 `vision_adapter` + 1
`vision_projection`; the whole checkpoint is 1436 tensors / 59,553,253,376 B
(55.4633 GiB), every one BF16, so the encoder is 6.45% of it. Qwen3-VL-4B is
713 tensors / 8,875,631,616 B (8.2661 GiB), the tower 9.36% of it. Both agree
byte-for-byte with the figures already declared at
`scripts/mm/tower_skip_rss.sh:199-200`.

**Threshold, two halves, both required.**

1. On the **default** arm, at the same commit but for the fix, on the same host
   and checkpoint:
   `peak_rss(pre-fix) - peak_rss(post-fix) >= 0.90 x tower_ondisk_bytes.`
   - `qwen3-vl`: **>= 747,625,881 B (0.6963 GiB).**
   - `muse-glimmer`: **>= 3,459,322,368 B (3.2217 GiB).**

   Ninety per cent, for the same reason L3 uses it — allocator granularity, and
   the geometry that is parsed either way — plus one reason specific to this
   change: the loader's per-tensor transient shrinks from `4N` to `2N` bytes
   alongside the store, and the peak includes it.

2. On the **`--language-model-only`** arm, `peak_rss` unchanged within 2%. That
   arm loads no tower, so the fix must not move it. If it does, the measurement
   is about something other than the tower and the axis is VOID, not MET.

An outcome below either half is a FAILING axis, recorded as failing and left
open. The threshold is not renegotiated after the number arrives.

Instrument: `scripts/mm/tower_skip_rss.sh`, whose default/`--language-model-only`
legs are exactly the two arms above, run once per binary. Its verdict logic is
for the *skip*, not for this row, so this row reads the per-leg
`peak RSS default` / `peak RSS lang-model-only` keys and applies the two halves
above. `--model-kind qwen3-vl` uses the `load-only` workload
(`tower_skip_rss.sh:409-428`), which is sufficient here: the widening is a
load-path cost.

### 6.2 The re-declared tower-skip threshold — ALSO declared before the run

After this row lands, the widening is gone, so
`TOWER_RESIDENT_BYTES = TOWER_ONDISK_BYTES` and the `* 2` at
`scripts/mm/tower_skip_rss.sh:431` must go. The L3 skip gate then reads:

| Kind | Threshold after this row | Expected saving |
|---|---:|---:|
| `qwen3-vl` | **747,625,881 B (0.6963 GiB)** | ~830,695,424 B (0.7736 GiB) |
| `muse-glimmer` | **3,459,322,368 B (3.2217 GiB)** | ~3,843,691,520 B (3.5797 GiB) |

The two thresholds coincide with §6.1's because both are 90% of the same
on-disk tower — this row removes exactly what the skip was additionally
removing.

**The published 1.542 GiB Qwen3-VL saving should therefore fall to about
0.77 GiB, and that is CORRECT, not a regression.** The flag will then be
freeing the tower the checkpoint actually ships instead of the tower plus our
widening. Every surface in §8 already carries that prediction; the implementing
wave converts the prediction into the measurement.

Removing the `* 2` is not renegotiating a threshold after seeing a number. It
is correcting a derivation whose premise no longer holds, and this paragraph is
the pre-declaration that authorizes it.

### 6.3 The latency guard

The mechanism can only *remove* a conversion pass (§3.2), so the predicted delta
is `<= 0` on every arm. The guard therefore has a noise band, not a budget:

- **Tower encode.** Median of 5 on one image, same host, same build type, using
  `VLLM_MM_TOWER_PROFILE=1` (`qwen3_vl_vision.cpp:621-629`) to read the
  prepare/forward split, and `tests/vllm/multimodal/bench_qwen3_5_vl_tower.cpp`.
  A regression above **2%** on either half is FAILING and the change does not
  land.
- **Load wall time.** Same recipe, `--device cpu`, load-only.
  A regression above **5%** is FAILING. The band is wider because load time is
  dominated by I/O from CIFS, not by the conversion.

Why 2% and 5% are sufficient: they are noise bands sized to the instrument, and
the guard's job is to catch the case where the change is NOT the storage-only
change this spec describes. A budget would be the wrong instrument, because
there is no trade being made — the fix deletes work rather than moving it.
Report the prepare and forward halves separately; a *win* concentrated in
`prepare` is the positive confirmation that the removed pass is the one §3.2
names.

Muse Glimmer has no oracle denominator (`muse_glimmer_vision.h:5-13`), so this
is a self-A/B against the pre-fix binary, not a parity claim.

### 6.4 Correctness gates

- **Exact-equality arm (new, and this row's real correctness gate).** Capture
  the Qwen3-VL tower output on the pre-fix binary and require the post-fix
  binary to reproduce it **byte-for-byte**. The existing tower goldens
  (`tests/vllm/multimodal/test_qwen3vl_tower.cpp`,
  `tests/vllm/models/test_muse_glimmer_vision.cpp`) are tolerance gates and
  would absorb a rounding change; §3.1 says the correct answer here is
  *identical*, so the gate should assert identity.
  **Red-first mutation:** replace the narrowing with truncation
  (`u >> 16`, dropping the round-to-nearest-even addend at `src/vt/dtype.cpp:323`)
  in a scratch copy. The exact arm must go red while the tolerance arms stay
  green. That difference is the whole argument for adding the arm.
- **Existing towers, unchanged.** `test_qwen3vl_tower`,
  `test_muse_glimmer_vision`, `test_qwen3_5_vl_e2e`,
  `test_qwen3_5_vl_video_e2e`, `test_qwen3_5_moe_vl_hw`,
  `test_clip_mmproj_gguf`, `test_minimax_h3`, `test_gemma4_vision_tower`,
  `test_gemma4_registry_e2e`, `test_tower_skip`, `test_gguf_mmproj_reach`,
  `test_loader_unaligned_offsets`. All three Qwen3-VL-shaped families must be
  rerun, not only the vehicle (§5 risk 7).
- **The f32 arm.** `test_muse_glimmer_vision`'s `compute_dtype = kF32` cases
  must stay green with bf16 storage (§4.4).
- **Full gate.** `scripts/agent-preflight.sh`, which registers both RSS suites
  (`:176`, `:180`).

### 6.5 Reachability

AGENTS.md §"Nothing lands dead": the smallest failing test must enter through a
production entry point.

The storage change is reached from the production loader on both vehicles —
`src/vllm/entrypoints/model_loader.cpp:2590-2591` for Qwen3-VL and
`src/vllm/model_executor/models/muse_glimmer_registry.cpp:88-91` (through
`ModelRegistry`) for Muse Glimmer. The §6.1 measurement runs
`vllm-server --model <checkpoint>` (`scripts/mm/tower_skip_rss.sh:478-480`), so
**the RSS gate is itself the reachability proof**: delete the production load
call and there is nothing left to measure.

State plainly in the PR body that the Qwen3-VL tower is loaded but never read
back on the production path (#1358) and that Muse Glimmer's tower has no
production *caller* (#1566). Neither weakens this row — the bytes are allocated
on the production load path either way, which is exactly the cost being
removed — but a reader must not be left to infer it.

## 7. Blast radius, and the scope correction #1359 needs

#1359's title names Muse Glimmer. The defect spans **six towers, three
loaders and two modalities**, and half of the published Qwen3-VL tower-skip
saving is this defect rather than tower size. Anyone scoping from the title
alone underestimates it by roughly a factor of five in models and by 100% in
bytes on the Qwen3-VL vehicle.

Proposed corrected scope, posted as a comment on #1359 (the title is the
developer's call):

- Muse Glimmer 30B — 809 tensors, 3.580 GiB -> 7.161 GiB. NOT in the landed
  slice; see `## Now` and #2166.
- Qwen3-VL-4B — 315 tensors, 0.774 GiB -> 1.547 GiB.
- Qwen3.5/3.6-27B dense and Qwen3.6-35B MoE — the *same* loader via
  `qwen3_5_weights.cpp:1770`.
- CLIP mmproj GGUF and MiniMax-H3 vision GGUF — the same struct, a dequant
  carrier instead of a widening.
- Gemma-4 vision — same struct shape, no `src/` loader yet, so it is a
  latent instance rather than a live cost.
- Whisper/Voxtral **audio** — the identical defect in a different modality.

If the developer widens #1359 to the defect class, the audio tower rides this
row. If #1359 stays vision-only, the implementing wave files the audio sibling
before touching it, and lists it here under `## Owed`.

## 8. What this row makes stale

The implementing wave must move these in the same change, because the numbers
below are all derived from a `x 2` this row deletes. Risk 4 makes the coupling
a hard red rather than a silent drift.

**Load-bearing (a gate fails otherwise):**

- `scripts/mm/tower_skip_rss.sh:429-431` — `TOWER_RESIDENT_BYTES=$((TOWER_ONDISK_BYTES * 2))`.
  The two `*_TOWER_ONDISK_BYTES` constants at `:199-200` do NOT change; the
  checkpoint's bytes are unchanged.
- `tests/scripts/test_tower_skip_rss_report.py:113-120` — the mirrored
  `resident()` helper, and the boundary constants its `BoundaryTests` derive.
- `.agents/specs/multimodal-track.md` — the DECLARED THRESHOLD, not only the
  resident figure. `test_spec_carries_the_threshold_the_instrument_applies`
  asserts that `need(kind)` appears there for both kinds: `747625881` for
  `qwen3-vl` and `6918644736` for `muse-glimmer`. It replaces
  `test_spec_carries_both_resident_figures`, which went tautological on the
  fixed kind and could no longer fail — see §5 risk 4.

**Prose that would otherwise read the halving as a loss:**

- `.agents/specs/multimodal-track.md:876-887, 889-900, 910-943, 945-952,
  960-969, 996-1003, 1023-1024, 1093, 1902-1912, 1950-1955`
- `.agents/engine-matrix.md:77` — the `ENG-MM-INPUT-PIPELINE` RSS paragraph
- `.agents/roadmap_v1.md:549`
- `.agents/benchmark-record.md:28677-28743`
- `docs/benchmarks/memory.md:16`
- `docs/benchmarks/open-gaps.md:79`
- `docs/guides/multimodal-input.md:164-180` — note `:173-174` says "4.6x" where
  the spec says "4.2x"; both are derived multiples and both move
- `docs/reference/server.md:209`
- `docs/FEATURES.md:240`

**Do NOT hand-edit** `docs/bench-evidence/tower-skip-rss-qwen3vl-thor-20260824{,.legs}.log`.
It is frozen evidence of a run that happened; a rerun produces a new file
beside it.

Per AGENTS.md §"Public documents", a lifecycle change also owes `STATUS`,
`BENCHMARKS` and the moved row spec's `## Now`.

## 9. Evidence produced by this investigation

- Checkpoint headers read 2026-08-28 from
  `/mnt/nas_share/checkpoints/{muse-glimmer-30b,qwen3-vl-4b-instruct}`,
  header-only (`8 + header_bytes`), no payload read. Both totals reproduce the
  figures already declared at `scripts/mm/tower_skip_rss.sh:199-200`
  byte-for-byte, which is an independent check on both.
- Upstream read at the pin from a local checkout verified at
  `5559679229bc961848b121ccdeaa8fa5d79bec98`. Anchors in §2.
- No GPU lease taken; no measurement run. Declaring the thresholds is this
  row's job, measuring them is the implementing wave's.
- Host during the investigation: `df -h /` 55G available (88% used), load
  average 9.00 at the last check. `scripts/agent-preflight.sh` exit 0.

## 10. Stop conditions

- **Any vision consumer found that genuinely computes in f32 on a weight**
  (shape 2): STOP and return `NEEDS_DECISION`. Do not convert per use. Trading
  resident bytes for a per-use conversion is a different piece of work with a
  different justification, and §3 says no such consumer exists on the paths in
  scope.
- **The exact-equality arm goes red**: STOP. The change is not the storage-only
  change this spec authorizes.
- **§6.1 reads below its band on either half**: record FAILING, leave the axis
  open, and name the next hypothesis (start with risk 5). Do not move the band.
- **§6.3 regresses beyond its noise band**: STOP and profile before landing.
  A regression contradicts §3.2's mechanism, so something else changed.
- **The Muse Glimmer measurement needs ~56 GB of worker-local disk** and the box
  has been at 98% used within the last week. Verify free space before staging,
  and report `df -h /` beside every result.

## Now

`ENG-MM-INPUT-PIPELINE` remains `READY`. #1359 stays OPEN: its Qwen3-VL half
landed, its Muse Glimmer half did not.

**Landed.** `Qwen3VLVisionWeights` (and therefore Qwen3-VL-4B, the
Qwen3.5/3.6-27B dense path and the Qwen3.6-35B MoE path, which share
`LoadQwen3VLVisionWeights`), `Gemma4VisionWeights`, and both GGUF tower loaders
now store the checkpoint's own bf16 bits. `LoadVisionF32` became
`LoadVisionBf16`; `Bf16TensorToF32`'s Qwen3-VL analogue kept an f32 spelling only
for the pos-embed table. `MakeDevBf16` on both towers is now a straight `Copy`
of the stored bytes and no longer allocates a per-weight scratch or runs a
per-weight `F32ToBF16` pass.

**Not landed, and §10's first stop condition is why.** Muse Glimmer's tower is
storage-only on its PRODUCTION path and is NOT storage-only on its GATE path.
`MuseGlimmerVisionConfig::compute_dtype`'s `kF32` arm computes on the stored
weight values, and §4.4's ruling that widening back is bit-identical rests on
"the values originated bf16" — true for the loader reading an all-BF16
checkpoint, false for `test_muse_glimmer_vision`, whose weights are a synthetic
f32 LCG that `scripts/mm/muse_glimmer_vision_ref.py:52-61` builds as
`torch.float32` and never rounds. Measured on a scratch tree rather than
predicted: the five f32-arm stages move from rel_l2 1.0-3.0e-07 to 2.164e-03,
2.193e-03, 2.220e-03, 2.892e-03 and 3.462e-03 against a 1e-6 bound. In the same
tree the bf16 arm read `rel_l2=5.951e-03 max_abs=3.675e-02`, byte-for-byte what
it reads today, so the widening IS removable and only the gate stands in the
way. [#2166](https://github.com/mudler/vllm.cpp/issues/2166) owns it.

**The instrument is per-kind because of that split.**
`scripts/mm/tower_skip_rss.sh`'s `TOWER_RESIDENT_BYTES` is now
`TOWER_ONDISK_BYTES` for `qwen3-vl` and `TOWER_ONDISK_BYTES * 2` for
`muse-glimmer`, mirrored by `WIDEN` in
`tests/scripts/test_tower_skip_rss_report.py`. That is one defect fixed on one
of two kinds, not two policies, and keeping the surviving `* 2` visible is what
stops the Muse Glimmer half from being forgotten.

**Nothing is measured yet.** §6.1, §6.2 and §6.3 are all PENDING: the RSS gate
needs a leased host and the staged `qwen3-vl-4b-instruct` checkpoint. The
implementing wave produced the code, the CPU-runnable gates and the harness;
the operator runs

```sh
scripts/mm/tower_skip_rss.sh --model-kind qwen3-vl --device cpu
```

on `thor:gpu0` or `dgx:gpu0` under an `rc` lease, at this commit and at its
parent, and applies §6.1's two halves to the per-leg `peak RSS default` and
`peak RSS lang-model-only` keys. MET is a default-arm reduction of at least
747,625,881 B with the `--language-model-only` arm unchanged within 2%.

## Owed

- [#1359](https://github.com/mudler/vllm.cpp/issues/1359) — its Muse Glimmer
  half, which is where the larger of the two savings is (3.580 GiB -> 7.161 GiB
  today). Blocked on [#2166](https://github.com/mudler/vllm.cpp/issues/2166),
  which owns the golden regeneration the `kF32` per-stage gate needs, and
  separately on the ~56 GB of worker-local disk its RSS leg wants (§10).
- Both §6.1 halves, §6.2's re-declared skip threshold and §6.3's latency band
  are PENDING a leased host. Nothing in this row has been measured.
- [#2173](https://github.com/mudler/vllm.cpp/issues/2173) — **the whole Gemma-4
  vision tower this row narrowed is UNREACHED, and it lands that way.**
  `Gemma4VisionForward` and `Gemma4VisionWeights` have no caller outside
  `tests/vllm/multimodal/test_gemma4_vision_tower.cpp` and
  `test_gemma4_registry_e2e.cpp`; `grep -rn 'gemma4_vision.h' src/ include/`
  returns only `gemma4_vision.cpp` including its own header. It is unreached
  twice over, because the driver that consumes tower output —
  `Gemma4GenerateGreedyViaRegistry`, `gemma4_mm.cpp:165` — takes `mm_projected`
  as a caller-supplied argument, masked-scatters it at `:250-252` without calling
  the tower, and is itself only called from `test_gemma4_registry_e2e.cpp:244`.
  Per `.agents/reachability.md` there is no production call site to delete, so
  the mutation has already answered the question. The `Gemma4VisionWeights` half
  of this row is therefore a storage-dtype correction to a class rather than to a
  capability, and it is landed as a staged slice under AGENTS.md
  §"Nothing lands dead". Owning row for the wiring:
  `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`. The Qwen3-VL half is
  NOT in this state: its tower is loaded on the production load path
  (`LoadQwen3VLWeights` -> `LoadQwen3VLVisionWeights`, proven by the
  call-site-deletion mutation that reds `test_tower_skip`), even though nothing
  reads it back yet ([#1358](https://github.com/mudler/vllm.cpp/issues/1358)) —
  and the bytes this row removes are the bytes that load allocates.
- [#2174](https://github.com/mudler/vllm.cpp/issues/2174) — FIXED IN FLOW, listed
  because it was found here rather than because it is outstanding.
  `gemma4_vision.cpp`'s `MakeDevBf16` copied `bf.size() * sizeof(uint16_t)` into
  a `bytes`-sized allocation with no guard, while the twin it was copied from
  (`qwen3_vl_vision.cpp:137`) grew one in this row. Pre-existing and
  behaviourally unchanged; the guard now mirrors the twin.
- Gemma-4's `Gemma4VisionWeights::position_embedding_table` keeps its host f32
  store for the same reason `pos_embed_w` does — `Gemma4VisionForward` sums its
  x and y rows on the host (`gemma4_vision.cpp:199-210`) and narrows only the
  sum, so narrowing the store would move the result. It is the §4.3 exception in
  a third tower and it rides that entry's reconciliation, not a new one.
- The scope correction proposed in §7 is a comment on #1359, not an edit to its
  title. If the developer keeps #1359 vision-only, the Whisper/Voxtral audio
  instance (`whisper_audio.h:68-99`, `voxtral.cpp:381-390`) needs its own issue
  before it is touched, and it is listed here so it is not lost either way.
- The `pos_embed_w` host-f32 interpolation (§4.3) is WIDER than upstream, which
  narrows to the model dtype at `qwen3_vl.py:335,344`. Kept deliberately and
  annotated by this row; reconciling it onto the mirror needs its own issue,
  because it changes tower numbers.
- Gemma-4 audio (`gemma4_audio.cpp:26-130`) is host-f32 all the way to the
  kernel — a genuine shape-2 case, excluded here with a reason and unowned.
