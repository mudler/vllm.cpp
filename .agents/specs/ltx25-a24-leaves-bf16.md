# LTX-2.5 — the conv video VAE ENCODER's bfloat16 arm (A24, wave 4)

Row: `LTX25-A24-LEAVES-BF16`
Issue: [#2850](https://github.com/mudler/vllm.cpp/issues/2850)
Parent scope: `.agents/specs/ltx25-completion-scope.md` §A.7 (A24), operator-owned
Wave 1: `.agents/specs/ltx25-a24-text-tower-bf16.md` (#2676, merge `8e582a5f9`)
Wave 2: `.agents/specs/ltx25-a24-connector-bf16.md` (#2720, merge `77704c8d0`)
Wave 3: `.agents/specs/ltx25-a24-video-vae-bf16.md` (#2786, merge `c20fb2ba2`) — hard dependency
Oracle: `.agents/oracles/ltx-2.md`, `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
Base: `72858451de5a3c55a29956c7ba4bbc4994641859`

---

## 0. THE WAVE IS ONE COMPONENT, NOT THREE, AND THE THREE MEASUREMENTS THAT SAY SO WERE RUN

This row was dispatched as "the final three": the video VAE encoder, the latent
upsampler and the duration head. Wave 3's central finding was that three of
§A.7's "components" were one unit in this tree, verified rather than inherited.
The same question asked here gives the opposite answer, and it is the answer the
tree gives rather than the one the dispatch assumed.

**Three commands, run at this row's base `72858451d`:**

```sh
grep -c 'Ltx2ConvVideoEncode('     src/vllm/multimodal/ltx2_video.cpp   # 2
grep -c 'Ltx2UpsampleVideoLatent(' src/vllm/multimodal/ltx2_video.cpp   # 3
grep -c 'Ltx2DurationPredict('     src/vllm/multimodal/ltx2_video.cpp   # 0
grep -c 'vt::'                     src/vllm/model_executor/models/ltx2_upsampler.cpp   # 0
grep -c 'Ltx2ConvVideoEncode'      src/vllm/model_executor/models/ltx2_video_vae.cpp   # 2
```

| component | production call sites | routes through wave 3's `vt::DType` seam | verdict |
|---|---:|---|---|
| **video VAE encoder** | **2** (`:3376`, `:4024`) | **yes** — same translation unit as the decoder | **THIS ROW** |
| latent upsampler | 3 (`:3773`, `:3800`, `:5310`) | **no** — 0 `vt::` in 723 lines | **its own row** |
| duration head | **0** | no | **blocked, and not on arithmetic** |

### 0.1 The encoder belongs to wave 3's file and to nothing else

`Ltx2ConvVideoEncode` is **defined inside `ltx2_video_vae.cpp`** (`:1953`), the
decoder's own translation unit, and its header says why: "The primitives are
SHARED with the decoder rather than copied ... A second copy of a causal pad is
the duplicate that goes wrong quietly, because each copy keeps its own green
gate" (`ltx2_video_vae_encoder.h:24-29`). Every stage it calls —
`CausalConv3d`, `ResnetBlock3d`, `AttnBlock3d`, `PixelNorm`, `Silu`,
`VaeKernels(...).group_norm` — already takes the `vt::DType` wave 3 threaded
through it, and the encoder already passes `x.data.dtype()` to three of them
(`:2072`, `:2079`, `:2081`). Wave 3 left two handholds deliberately:
`Ltx2ConvVideoEncode` **refuses a bf16 bag by name** (`:1969-1972`), and
`VaeStore::Host`/`HostBegin` is documented as "an ENCODER-ONLY accessor now"
(`ltx2_video_vae.cpp:309-317`). This row is the continuation those two were
written for. Both are consumed here: the refusal becomes an arm (§5.1), and
`HostBegin` loses its last caller and is deleted, leaving `Host` as what it now
is -- the f32 host branch of `AttnBlock3d`, which BOTH paths reach (§5.2).

### 0.2 The upsampler is a second wave 3, not a leaf

`src/vllm/model_executor/models/ltx2_upsampler.cpp` is **723 lines with zero
`vt::` references**. It has no `VaeStore`, no `VaeKernels`, no `vt::DType`, no
shared-seam routing of any kind: 15 hand-rolled `std::vector<float>` buffers and
its own `Volume`, `Conv3dPad1`, `Conv2dPad1`, `GroupNorm`, `Silu`, blur
downsample and pixel shuffle. Its convolutions are **plain `torch.nn.Conv3d`
with zero padding on every axis including time**, deliberately not the VAE's
`CausalConv3d` (`ltx2_upsampler.h:45-62`), so it cannot borrow wave 3's kernels
even in principle. Giving it a bf16 arm means building a dtype-parameterised
storage model for a file that has none, and then measuring a rounding rule for
each of five kernels — because A24 has now produced **five different norms in
five components** and no rule has yet transferred between two of them. That is a
wave, not a leaf, and folding it in here would make one pull request carry two
unrelated ports.

### 0.3 The duration head cannot land at all, and the obstacle is reachability

`Ltx2DurationPredict` has **zero production call sites**. The engine refuses its
own load extra by name:

```text
multimodal/ltx2_video.cpp:577
  "the 'duration_head_path' extra names '<path>', but the duration head is NOT
   WIRED into this engine: `Ltx2DurationPredict` is ported and gated as a brick
   (ltx2_duration_head.h ...) and nothing here constructs one"
```

AGENTS.md, *Nothing lands dead*: "What lands is reachable from a production
entry point at its own merge commit ... The fresh reviewer mutates for this.
Delete the production call site in a scratch copy and rerun the focused gate."
**There is no production call site to delete.** A bf16 arm for the duration head
could only be gated by a unit test constructing the type by hand, which "proves
that the class works, never that anything reaches it". The head's dtype is not
the head's problem; being unwired is, and that is #611's row
(`.agents/specs/ltx25-retire-dead-arms.md`), not A24's. Wiring it first and
narrowing it second is the only order in which the second step can be gated.

**This is a re-scoping finding and it is reported rather than worked around.**
Two items in this campaign (A8, A18) were mis-scoped by forcing a shape, and
wave 3's own predecessor draft was too. §A.7's eight-row table is
operator-owned; this row does not edit it and returns the finding under
`## Owed`.

---

## 1. Scope

**IN.**

* `src/vllm/model_executor/models/ltx2_video_vae.cpp` — the encoder half only:
  `Patchify`, `SpaceToDepthFold`, `SpaceToDepthDownsample` and
  `Ltx2ConvVideoEncode`, plus the `RequireF32`-shaped refusal at `:1969-1972`
  which becomes an arm.
* `include/vllm/model_executor/models/ltx2_video_vae_encoder.h` — the DTYPE
  block, whose "f32 throughout" sentence this row falsifies.
* `src/vllm/multimodal/ltx2_video.cpp` — the encoder's loader call site
  (`:1657`) asks for `kBF16`, and the render path gains the
  `vae_encode_in_not_bf16` / `vae_encode_not_bf16` counters.
* `include/vllm/multimodal/ltx2_video.h` — those four trace fields.
* `tests/vllm/models/test_ltx2_vae.cpp`, `tests/vllm/multimodal/test_ltx2_video.cpp`.

**OUT, and owed by name in `## Owed`.** The **latent upsampler** (§0.2). The
**duration head** (§0.3). The **FP8/NVFP4** arms, which are A22. The **audio
VAE**, whose f32 §A.7 excludes deliberately. The **CUDA arm** of `kLtx2Vae`,
which wave 3 left owed for a reachability reason this row does not change. The
**bf16 noise draw** ([#2780](https://github.com/mudler/vllm.cpp/issues/2780))
and **`Ltx2TrapezoidalMask1d`**
([#2816](https://github.com/mudler/vllm.cpp/issues/2816)), both the developer's
and both still open.

**HARD DEPENDENCY, satisfied.** `VaeStore`'s `dtype` parameter, `LoadElem`,
`StoreElem`, `RequireVaeDType`, `Upload`/`Download`, and the dtype-aware
`CausalConv3d` / `ResnetBlock3d` / `AttnBlock3d` / `PixelNorm` / `Silu` /
`group_norm` are all wave 3's and are on `main` at `c20fb2ba2`.

---

## 2. Upstream anchors, read at `fd4ded7f` with the checkout's identity asserted

`git -C ~/_git/LTX-2 remote get-url origin` is
`https://github.com/Lightricks/LTX-2.git`, `rev-parse HEAD` is
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, and `git status --porcelain | wc -l`
is `0`. All three were run before anything below was trusted.

| what | anchor | value |
|---|---|---|
| the pipeline's ONE dtype | `ltx-pipelines/src/ltx_pipelines/distilled.py:109` | `self.dtype = torch.bfloat16` |
| the ENCODER is constructed with it | `distilled.py:120-125` | `ImageConditioner(model_paths.video_vae(), self.dtype, ...)` |
| ...and the builder hands it to the module | `utils/blocks.py:985-986` | `self._encoder_builder.build(device=self._device, dtype=self._dtype)` |
| **the input pixels arrive ALREADY bf16** | `utils/helpers.py:285-294` | `load_image_and_preprocess(..., dtype=dtype, ...)` then `video_encoder(image)` |
| **and `forward` never casts them** | `video_vae/video_vae.py:264-336` | no `.to(...)` anywhere on the path |
| the frame-count crop | `video_vae.py:276-286` | warns and crops; does not raise |
| patchify is pure movement | `video_vae/ops.py:6-32` | `rearrange` only |
| the space-to-depth fold, both branches | `video_vae/sampling.py:43-49, 55-61` | `rearrange` only |
| **frame 0 is duplicated at temporal stride 2** | `sampling.py:39-40` | `torch.cat([x[:, :, :1], x], dim=2)`, BEFORE both branches |
| **the skip is a GROUP MEAN** | `sampling.py:50-51` | `rearrange(..., "b (c g) d h w -> b c g d h w")`; `.mean(dim=2)` |
| **and the two branches are ADDED** | `sampling.py:63` | `x = x + x_in` |
| the skip conv is stride 1 | `sampling.py:26-33` | `stride=1`, `out_channels // prod(stride)` |
| a strided CausalConv3d still prepends `k_t - 1` | `video_vae/convolution.py:305-307` | padding decided before the stride |
| **`normalize` NARROWS both stat buffers** | `video_vae/ops.py:81-84` | `(x - mean.view(1,-1,1,1,1).to(x)) / std.view(1,-1,1,1,1).to(x)` |
| the encoder returns the normalized MEANS | `video_vae.py:334-336` | `torch.chunk(sample, 2, dim=1)` then `normalize(means)` |

**There is no f32 pin on the encoder's forward path, measured with a control
rather than asserted from a failed grep.**
`grep -rn 'float32\|\.float()\|autocast' packages/ltx-core/src/ltx_core/model/video_vae/`
returns hits, so the pattern works. Inside `video_vae.py`, `sampling.py` and
`ops.py` the hits are the tiling masks and a loader-time gate fold only. The
audio VAE's argued f32 (`vocoder.py:575-580`) does not extend here, for the same
reason wave 3 gave.

---

## 3. The local side, at `72858451d`

**The encoder already refuses bf16 by name**, and wave 3 wrote that refusal
knowing this row would replace it (`ltx2_video_vae.cpp:1969-1972`). It is a
refusal, not an arm.

**Exactly TWO production routes reach the encoder, and both hand it f32 pixels.**
`multimodal/ltx2_video.cpp:3376` (the retake path, from
`Ltx2ReadFrameDirectory`) and `:4024` (`encode_conditioning_image`, from
`Ltx2LoadImageAndPreprocess`). The second is the direct mirror of
`combined_image_conditionings`. Neither passes a queue, so the encoder is on the
CPU queue on every build of this project.

**The loader feeds it at the f32 default.** `Ltx2LoadVaeWeights(f,
Ltx2VideoVaeEncoderKeyRules())` at `:1657` takes the `compute_dtype = kF32`
default wave 3 added; only the decoder site at `:1646` asks for `kBF16`.

**Four sites are f32 by construction and are the whole change.** `Patchify`
(`:1774`), `SpaceToDepthFold` (`:1805`) and `SpaceToDepthDownsample` (`:1853`,
`:1874`) each `Alloc(..., vt::DType::kF32)` and move elements through
`VaeStore::Host()`, which is the f32-only accessor and now throws on a bf16
store. `Ltx2ConvVideoEncode` allocates its input volume the same way (`:2014`)
and its `normalize` tail (`:2118-2127`) reads `x.data.Host()` as `float*` and
divides by f32 statistics.

---

## 4. What upstream's bf16 encoder actually computes in — EXECUTED, not read

Every table below was produced by RUNNING the pinned modules on CPU with torch
`2.11.0+cu130` on 2026-09-04. Counts are bf16-word mismatches against upstream's
own output; `separating` is how many of the listed alternatives the probe can
tell apart from upstream. **A probe that separates nothing is reported as such
rather than as a confirmation.**

Wave 3's rule holds and is restated because it is what makes these four probes
necessary: five components, five different rules. Nothing here was inherited.

### 4.1 The group mean is a WIDENED accumulate with ONE rounding — and a probe at `group_size = 2` gates nothing

`sampling.py:50-51`. `[1, 8*g, 2, 4, 4]`:

| g | scale | n | f32 accumulate | f64 accumulate | sequential bf16 accumulate | separating |
|---|---|---|---|---|---|---|
| 2 | 2^0 | 256 | **0** | **0** | **0** | **0** |
| 2 | 2^-7 | 256 | **0** | **0** | **0** | **0** |
| 4 | 2^0 | 256 | **0** | **0** | 72 | 1 |
| 4 | 2^-7 | 256 | **0** | **0** | 97 | 1 |
| 8 | 2^0 | 256 | **0** | **0** | 138 | 1 |
| 8 | 2^-7 | 256 | **0** | **0** | 145 | 1 |

The kernel widens internally and rounds only the OUTPUT — wave 3 §4.2's finding
for `torch.mean`, re-run here rather than carried over. `separating = 1`: f32 and
f64 accumulation are indistinguishable and this row does not claim a gate on the
width. **At `group_size = 2` a two-element mean is exact in any order and the
probe is a mute switch**, so the case is laid at `g = 4` and `g = 8`. Our port
already accumulates in `float` (`:1876-1884`); what it lacks is the rounding on
the store.

### 4.2 The skip ADD separates nothing — and the ROUNDING BEFORE it separates everything

Two questions that look like one. `sampling.py:63`.

**(a) A bare bf16 add**, 4096 values, against an f32 and an f64 evaluation each
rounded once:

| scale | n | f32 add, round once | f64 add, round once | separating |
|---|---|---|---|---|
| 2^0 | 4096 | **0** | **0** | **0** |
| 2^-7 | 4096 | **0** | **0** | **0** |
| 2^-14 | 4096 | **0** | **0** | **0** |

`separating = 0` at every scale tested including 2^-14, which is where wave 3's
`PixelNorm` epsilon finally parted. A single add has one rounding point wherever
it is evaluated, so there is no width question here and this row does not invent
one.

**(b) Carrying the group mean in f32 ACROSS the add** — the same operands, with
only the skip's rounding point moving:

| g | scale | n | upstream | group mean kept f32 through the add | separating |
|---|---|---|---|---|---|
| 4 | 2^0 | 256 | **0** | 49 | 1 |
| 4 | 2^-7 | 256 | **0** | 61 | 1 |
| 8 | 2^0 | 256 | **0** | 45 | 1 |
| 8 | 2^-7 | 256 | **0** | 48 | 1 |

**18-24% of the block's output is wrong if the mean's result is not rounded
before the add**, and (a) proves the add itself cannot be blamed for it. This is
wave 3's per-SITE lesson in a fifth place: the rounding point is the STORE, not
the operator.

### 4.3 `PerChannelStatistics.normalize` NARROWS BOTH STAT BUFFERS — and our port keeps them f32

`ops.py:81-84` applies `.to(x)` to both registered buffers. Wave 3 §4.7 measured
`un_normalize`, which is a multiply and an add; this is a **subtract and a
divide**, in the same file, and it was re-run rather than assumed. The f32
buffers were captured BEFORE `.to(bfloat16)` — reading them after narrows them in
place and yields a false 0/0, the trap wave 3's implementer and both its
reviewers hit. `[1, C, 2, 3, 3]`:

| C | scale | n | upstream (stats narrowed) | stats kept f32 (**our port**) | one fused f32 expression | separating |
|---|---|---|---|---|---|---|
| 16 | 2^0 | 288 | **0** | **136** | 129 | 2 |
| 16 | 2^-7 | 288 | **0** | **129** | 147 | 2 |
| 128 | 2^0 | 2304 | **0** | **889** | 975 | 2 |
| 128 | 2^-7 | 2304 | **0** | **931** | 937 | 2 |

`separating = 2`. **39-47% of the encoder's very last arithmetic is wrong if the
activation dtype is flipped without narrowing the statistics**, and no token gate
can see it, because the latent is a `std::vector<float>` on either arm.

### 4.4 The stride does not change the convolution's rule — a CONTROL, not a claim

Wave 3 §4.10 fixed `cpu_conv3d`'s contract (f32 accumulate, bias INSIDE the
accumulator, one rounding on store) against the UNSTRIDED `nn.Conv3d`. The
encoder's `compress_*` blocks convolve at stride 2, which wave 3 never ran. Six
seeds of `[1,32,4,10,10] * [32,32,3,3,3]`:

| stride | n | our contract | % | bias added AFTER rounding | % |
|---|---|---|---|---|---|
| 1 | 24576 | **3** | 0.0122% | 6339 | 25.79% |
| 2 | 3072 | **1** | 0.0326% | 826 | 26.89% |

The residue is the same order as wave 3's 0.0159% and is torch's own blocked
association order, not an accumulator-width error. The bias-placement hypothesis
separates at 26-27%, so the probe is not a mute switch: `separating = 1`. **The
stride changes which output positions exist, not the rule**, and our kernel
already has the rule right.

### 4.5 What is pure MOVEMENT, and therefore has no rule

`patchify` (`ops.py:6-32`) and both space-to-depth folds
(`sampling.py:43-49, 55-61`) are `rearrange` calls. They permute elements and
compute nothing, so a bf16→f32→bf16 round trip through `LoadElem`/`StoreElem` is
bit-exact by construction and no probe can separate anything. Stated so that the
absence of a table here is a finding and not an omission.

---

## 5. Design

**One rule, applied at four sites: the ELEMENT WIDTH follows the volume, and the
rounding point is the STORE.**

1. **`RequireF32`-shaped refusal → `RequireVaeDType`, PLUS a token of its own.**
   `Ltx2ConvVideoEncode`'s `VT_CHECK(weights.dtype == kF32)` becomes wave 3's
   two-width check, so a third width still cannot arrive by silence. It is
   preceded by an entry `VT_CHECK` carrying a message NO OTHER SITE EMITS ("the
   encoder was handed"), because `RequireVaeDType`'s own message is the decode's
   and `VaeStore::Alloc` emits it too, 60-odd lines downstream. A subcase
   asserting the shared text cannot tell the entry from the staging allocation,
   and passes with the entry deleted; §9.6 measures exactly that.
2. **The entry narrowing is `VaeStore::Upload`.** Upstream's pixels are already
   bf16 when `forward` is entered (§2), so the port rounds once at the boundary
   and computes from there. `Ltx2ConvVideoEncode` allocates `x` at the weights'
   dtype and fills it through `Upload` rather than through `Host()`. That was
   `HostBegin()`'s LAST caller, so `HostBegin` is deleted here rather than left
   as an accessor with zero callers whose own doc describes writes that nothing
   performs any more. `Host`'s remaining callers are `AttnBlock3d`'s `!staged`
   branch and nothing else, so its doc and its throw message name that branch
   instead of "the encoder path".
3. **`Patchify` and `SpaceToDepthFold` allocate `Like` their input** and move
   elements with `LoadElem`/`StoreElem`. §4.5 says this is exact.
4. **`SpaceToDepthDownsample`**: `grown` allocates at the input's dtype and the
   frame-0 duplication moves elements the same way; the group mean keeps its
   `float` accumulator (§4.1) and **stores through `StoreElem`**, which is the
   rounding §4.2(b) requires; the final add loads both operands, adds in f32 and
   stores once (§4.2(a) says the add's own width is not observable).
5. **The `normalize` tail narrows the statistics first** (§4.3), then subtracts
   and divides with a rounding after each, and widens once into the public
   `Ltx2LatentVolume`. That widening mirrors `VaeStore::Download` and
   `sample.to(output_dtype)`; it is not a second arithmetic path, because the
   values are already on the bf16 grid when they reach it.
6. **The loader asks for `kBF16` at the encoder's call site only** — one argument
   at `ltx2_video.cpp:1657`, which is the production wiring this row is gated on.

**No new file, no new seam, no parallel path.** Every kernel this touches is one
wave 3 already parameterised.

---

## 6. Tests

**Red first, entering through a production entry point.**

* **`tests/vllm/multimodal/test_ltx2_video.cpp` — the production dtype case.**
  A render that supplies a conditioning image reaches `:4024`. Two assertions,
  in wave 3's shape:
  1. the encoder's INPUT carries sub-bf16 detail — `vae_encode_in_not_bf16 >
     vae_encode_in_values / 2`, with the measured value printed beside the floor,
     because a floor below the real count is a mute switch;
  2. the encoder's OUTPUT latent is entirely bf16-representable —
     `vae_encode_not_bf16 == 0`.
  Assertion 1 is what stops assertion 2 going quietly green on a fixture whose
  numbers happen to land on bf16 grid points, which is the hole A24 sat in.
* **`tests/vllm/models/test_ltx2_vae.cpp` — the three rules, each with its
  rejected hypothesis beside upstream's answer**, from §4.1, §4.2(b) and §4.3.
  Each case asserts the separating count is non-zero before asserting the match,
  so a case that stops separating fails instead of passing.
* **The refusal case** for a third width, so `RequireVaeDType` is not widened by
  accident. It asserts the ENTRY's own token rather than the shared decode
  message, for the reason §5.1 and §9.6 give.

**Mutations the fresh reviewer must run.** Delete the `kBF16` argument at
`ltx2_video.cpp:1657` (the production call site) — the dtype case must red.
Remove the `StoreElem` rounding in the group mean — §4.2(b)'s case must red.
Keep the statistics f32 in `normalize` — §4.3's case must red.

---

## 7. Risks

* **The gated fixture may not reach a `group_size > 2` block**, which would make
  §4.1's shipped case a mute switch. Measured on the fixture config and reported;
  if it does not, the case is laid at a synthetic shape and the fact is stated
  rather than hidden.
* **`VaeStore::Host` throws on a bf16 store**, so any encoder site this row
  misses fails loudly rather than reinterpreting bytes. That is the intended
  polarity and it is wave 3's, not this row's.
* **No real weights and no lease.** Every number here is from the pinned modules
  on synthetic tensors. A real-weights render is owed.
* **The retake path (`:3376`) is not covered by the production dtype case**, which
  goes through `:4024`. Both share one `Ltx2ConvVideoEncode`, so the arithmetic is
  gated once; the second route's own coverage is stated as owed.

---

## 8. Gates

Runnable at the row's head, from the worktree root.

```sh
# 0. The shape verdict of §0, re-derived rather than remembered.
grep -c 'Ltx2ConvVideoEncode('     src/vllm/multimodal/ltx2_video.cpp                 # 2
grep -c 'Ltx2UpsampleVideoLatent(' src/vllm/multimodal/ltx2_video.cpp                 # 3
grep -c 'Ltx2DurationPredict('     src/vllm/multimodal/ltx2_video.cpp                 # 0
grep -c 'vt::'                     src/vllm/model_executor/models/ltx2_upsampler.cpp  # 0

# 1. The production wiring exists and is UNIQUE (an anchor's uniqueness, not its
#    existence -- a count of 1 is what makes the reachability mutation single-site).
grep -c 'Ltx2VideoVaeEncoderKeyRules(), vt::DType::kBF16' src/vllm/multimodal/ltx2_video.cpp  # 1

# 2. The refusal became an arm and no third width slipped in.
grep -c 'RequireVaeDType' src/vllm/model_executor/models/ltx2_video_vae.cpp           # >= 2

# 2b. The encoder's ENTRY refusal has a token of its own, and it is UNIQUE -- which
#     is what lets §6's subcase discriminate the entry from `VaeStore::Alloc`.
grep -rc 'the encoder was handed' src/vllm/model_executor/models/ltx2_video_vae.cpp   # 1

# 2c. `HostBegin` is gone and left no caller behind.
grep -rc 'HostBegin' src/ include/ tests/ examples/                                   # 0

# 3. The focused gate.
cmake --build build -j 2 --target test_ltx2_vae test_ltx2_video
./build/tests/test_ltx2_vae  -tc='*encoder*'
./build/tests/test_ltx2_video -tc='*ENCODER*'

# 4. The full gate.
scripts/agent-preflight.sh --staged
python3 scripts/check-pr-size.py --base origin/main --head HEAD
```

---

## 9. Evidence, as MEASURED

Every figure below was run at this row's head. The mutations were applied to the
worktree and the tree was restored byte-for-byte after each
(`git status --porcelain | wc -l` = 0).

### 9.1 The value gate is BIT-EXACT, which is not wave 3's shape

`./build/tests/test_ltx2_vae -tc='*ENCODER*BF16*'` — 1 passed, **43 of 43
assertions**:

```text
BF16 video encoder: 0 of 64 words differ, max|diff| = 0;
  one-ulp sensitivity 0.0078125; the two upstream arms are 0.00980377 apart;
  defect distances: f32 statistics 0.00390625, unrounded group mean 0.00390625
bf16 encode values wider than bf16: 0 of 64
```

**The band shape wave 3 uses would have been a mute switch here, and the numbers
say so rather than the prose.** Both defect distances (0.00390625) are BELOW the
chain's one-ulp sensitivity (0.0078125), so any band wide enough to admit an
honest port admits both defects. The case asserts that relation
(`DefectStats < UlpSensitivity`, `DefectGroupMean < UlpSensitivity`) so it cannot
quietly stop being true, and holds the port bit-exact instead — which is
available because this fixture is three convolutions deep rather than thirteen.

### 9.2 The red, entering through a production entry point

With the arm implemented and the loader NOT yet wired:

```text
tests/vllm/multimodal/test_ltx2_video.cpp:7108: ERROR:
  CHECK( trace.vae_encode_not_bf16 == 0 ) is NOT correct!
  values: CHECK( 20 == 0 )
  logged: pixels into the encode, wider than bf16: 15352 of 15360
          VAE encode output, wider than bf16: 20 of 20
```

The input floor is `> n / 2` and the measured value is **15352 of 15360
(99.95%)**, printed beside it. After wiring the loader: 1 passed, 7 of 7.

### 9.3 The three mutations, with their literal results

| mutation | `test_ltx2_vae` | `test_ltx2_video` |
|---|---|---|
| **delete the production call site** (the `kBF16` argument at `ltx2_video.cpp:1657`) | **55/55, 3570/3570 GREEN** | **RED**, `CHECK( 20 == 0 )` |
| **the group mean carried in f32 across the add** | **RED**, `CHECK( 39 == 0 )`, max\|diff\| 0.00390625 | 115/115 GREEN |
| **the per-channel statistics taken off the bf16 grid** | **RED**, `CHECK( 1 == 0 )`, max\|diff\| 0.000244141 | 115/115 GREEN |

**The first row is the reachability proof and its GREEN half is the point.** The
whole value suite stays green with nothing reaching the arm — 3570 assertions
that measure a class and not a capability — and only the render-path case moves.
That is what `.agents/reachability.md` asks a mutation to show.

**The second and third rows are the complement.** The render-path counter stays
green under both, because a wrong VALUE computed at the right WIDTH is still
bf16-representable. Neither case can replace the other.

Two notes on the mutations rather than around them. The first mutation deletes a
line, so the derived READER ANCHORS case reds with it; that is the mutation's
shape and not a second finding. The third mutation perturbs each statistic by one
f32 ulp because the stream's own f32 value is not reachable from inside the
function — it is therefore a **lower bound**, and the real defect is 16x larger:
the generator measures it end to end against upstream at 0.00390625 and refuses
to emit a zero.

### 9.4 The generator caught its own author in this repository's named trap

The first form of the f32-statistics defect read the buffers AFTER
`.to(torch.bfloat16)` had narrowed them in place. Both hypotheses became the same
tensor and the probe measured a defect of exactly **0**, which the generator's own
`assert defect_stats > 0` refused to emit. It is the trap wave 3's implementer and
BOTH its reviewers hit. The repaired probe captures the f32 buffers before any
cast and additionally asserts they are OFF the bf16 grid, so a later change to
`param_values` cannot mute the defect in silence.

### 9.5 The rest

* The four probes of §4, with their `separating` counts and the rejected
  hypothesis printed beside upstream's answer in every table.
* The oracle identity assertion of §2, run before any anchor was trusted.
* §0's five `grep -c` measurements, re-run in `## Gates`.
* **The regenerated goldens are a pure addition**: `git diff --stat` on
  `tests/vllm/models/ltx2_vae_goldens.inc` is `48 ++++`, **0 deletions**. Every
  previously committed golden reproduced byte-for-byte from the pinned checkout.


---

### 9.6 The fresh review's repairs, and the mutation that made one of them necessary

The fresh review returned `PASS WITH REPAIRS`. Four were applied.

**F1 — the entry refusal was NOT gated.** §6's subcase reads "at its own entry"
and asserted `doctest::Contains("the decode serves f32")`, which is
`RequireVaeDType`'s message. `VaeStore::Alloc` calls `RequireVaeDType` too, so
that string is emitted at two sites and the subcase could not tell them apart:
with the entry check deleted the whole suite stayed **55/55, 3570/3570 GREEN**.
The repair gives the entry a token no other site emits and asserts that.
Deleting the entry refusal now reds, and the failure names the site that had been
answering all along:

```text
tests/vllm/models/test_ltx2_vae.cpp:1768: ERROR: CHECK_THROWS_WITH_AS( ...
  "the encoder was handed" ... ) threw a DIFFERENT exception! (contents:
  "vt: ltx2 video vae: the decode serves f32 ... it was handed f16 at
   .../ltx2_video_vae.cpp:215")
[doctest] test cases:   55 |   54 passed | 1 failed
[doctest] assertions: 3570 | 3569 passed | 1 failed | Status: FAILURE!
```

`:215` is `RequireVaeDType`'s body, reached from `VaeStore::Alloc`. The mutation
compiled (`cmake --build build --target test_ltx2_vae -j 2`, rc 0) and the tree
was restored byte-for-byte, sha256 `16690621c166f21d` before and after.

**F2 — `docs/FEATURES.md` stated the opposite of what ships.** The wave 3 row's
tail read "the encoder, upsampler and duration head stay f32 and are owed", which
this row falsifies. Its tail now names the two components that remain owed, and
the encoder gets a row of its own in the shape of wave 3's.

**F3 — prose and a runtime message this row made false.** `HostBegin` lost its
last caller when the staging gather replaced it (§5.2) and had **zero** callers in
`src/ include/ tests/ examples/`; it is deleted. `Host`'s doc said every caller is
on the encoder path, and its throw sent a user to a closed row (#2786) with a
false statement; both now name `AttnBlock3d`'s host branch, which is what remains.

**F4 — a wrong upstream anchor at five sites.** `x_in.mean(dim=2)` is
`sampling.py:51`, not `:47-49`, which is `p2=`, `p3=`, `)` of the preceding
`rearrange`. Read again in the pinned checkout at `fd4ded7f`; every citation is
now `sampling.py:50-51`, which spans the group `rearrange` and the mean.

The **nit** was applied too: the `kLtx2VideoEncBf16GroupSize >= 4` comment claimed
to pin what the C++ blocks produce, when the line reads only the generator's
emitted constant. The comment now says what the line does and where the
C++-side half is actually covered.

---

## 10. Stop conditions

* **A rule that does not separate.** If a probe's `separating` is 0 at every
  scale swept, the row reports that and does not claim a gate. §4.2(a) is already
  such a case and is published as one.
* **A `group_size` the fixture cannot reach.** §7's first risk; reported, not
  worked around.
* **Anything needing a GPU lease or real weights.** Refused by name and recorded
  owed. This row implements and gates nothing on a device.

---

## Owed

* **§A.7's eight-row table splits three components this row cannot treat alike**,
  and for two different reasons (§0.2, §0.3).
  `.agents/specs/ltx25-completion-scope.md` is operator-owned; this row reports
  the finding and does not edit it.
* **The LATENT UPSAMPLER's bf16 arm** (`ltx2_upsampler.h:108-112`, `VideoUpsampler`
  at `distilled.py:138-141`), reached at `ltx2_video.cpp:3773`, `:3800` and
  `:5310`. It is a full wave: 723 lines, **zero** `vt::` references, its own
  `Volume` and its own `Conv3dPad1` / `Conv2dPad1` / `GroupNorm` / `Silu` /
  blur-downsample / pixel-shuffle kernels, none of which has a dtype parameter.
  Its convolutions are plain zero-padded `nn.Conv3d`, deliberately not the VAE's
  `CausalConv3d`, so wave 3's kernels do not transfer. **Its temporal arm's
  shipped checkpoint is not on the NAS** (`ltx2_upsampler.h:39-44`), so that arm
  has no real-weight result on any dtype.
* **The DURATION HEAD's bf16 arm** (`ltx2_duration_head.h:55-58`,
  `DurationPredictor` at `distilled.py:163-165`). **No longer blocked on
  reachability.** When this was written the head had zero production call sites
  and `duration_head_path` was refused by name, so a dtype arm could not be gated
  at all — the reachability mutation had no call site to delete. Row
  `LTX25-DURATION-HEAD-WIRE`
  ([#2900](https://github.com/mudler/vllm.cpp/issues/2900),
  `.agents/specs/ltx25-duration-head-wire.md`) supplied the wiring, and
  `Ltx2LoadDurationHeadWeights` already takes the `compute_dtype` the arm needs,
  so what remains is a call-site change plus its fixture. Still owed, and still
  not this row's.
* **§A.7's anchor for the latent upsampler has ROTTED.** The scope doc cites
  `ltx2_upsampler.h:66-70` for that header's DTYPE record; at `72858451d` the
  block is at `:108-112` and line 66 is a bare `//`. The file last moved at
  `5a6f1374a` (2026-09-02). The other two anchors in that row of the table still
  resolve (`ltx2_video_vae_encoder.h:52` and `ltx2_duration_head.h:55`), so this
  is one stale cell and not a systematic drift. The scope doc is operator-owned
  and this row does not edit it.
* **[#2855](https://github.com/mudler/vllm.cpp/issues/2855) — `gen-ltx2-vae-goldens.py`
  emits THREAD-COUNT-DEPENDENT goldens.** Regenerating the committed file at
  `OMP_NUM_THREADS=2` moves 4970 lines; at this box's default core count it is
  byte-identical. So "the goldens regenerate byte-identically" is a check only
  one machine can run, which is not what a golden is for. It is listed HERE
  because this row's own fresh review is what found it, and this row deliberately
  left it out of scope: §6e was checked at both thread counts and is
  thread-invariant, so nothing this row landed depends on it. It needs its own
  spec, because it decides what a golden MEANS here — pin the thread count and
  accept a one-time diff, make the reductions order-independent, or fold the
  thread count into the oracle's identity so a mismatched regeneration refuses.
  Recorded rather than left orphaned: an issue nobody owns is how 701 of them
  accumulated.
* **Whether the audio VAE's argued f32 extends above the vocoder.** §A.7 names
  this as a bound on A24's population and returns it; this row does not resolve
  it.
* **The RETAKE route's own dtype coverage** (`ltx2_video.cpp:3376`). §6's
  production case enters through `:4024`; both share one `Ltx2ConvVideoEncode`, so
  the arithmetic is gated once and the second route is not separately covered.
* **A real-weights bf16 encode against upstream**, which needs a lease and
  `tools/oracle/ltx2_oracle.py`. Every number in §4 is from the pinned modules on
  synthetic tensors.
* **The encoder's share of the VAE checkpoint bytes** is unmeasured, for the same
  reason wave 3 recorded the decoder's as unmeasured: `docs/USAGE.md` records the
  whole file and the key rules take a prefix each.
* **The bf16 noise draw** ([#2780](https://github.com/mudler/vllm.cpp/issues/2780))
  and **`Ltx2TrapezoidalMask1d`**
  ([#2816](https://github.com/mudler/vllm.cpp/issues/2816)) stay the developer's.
  Neither is closed by this row.
* **The CUDA arm of `kLtx2Vae` at bf16**, wave 3's, unchanged here.
* **The FP8 and NVFP4 arms**, which are A22.

## Now

`ACTIVE`, reviewed pull request open. Spec commit `55a98935b` precedes the two
implementation commits `4913a313e` and `e3a669949`, which is the commit order
that proves it came first.
