# LTX-2.5 — the conv video VAE decoder's bfloat16 arm (A24, wave 3)

Row: `LTX25-A24-VIDEO-VAE-BF16`
Issue: [#2786](https://github.com/mudler/vllm.cpp/issues/2786)
Parent scope: `.agents/specs/ltx25-completion-scope.md` §A.7 (A24), operator-owned
Wave 1: `.agents/specs/ltx25-a24-text-tower-bf16.md` (#2676, merge `8e582a5f9`)
Wave 2: `.agents/specs/ltx25-a24-connector-bf16.md` (#2720, merge `77704c8d0`) — hard dependency
Oracle: `.agents/oracles/ltx-2.md`, `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
Base: `77704c8d0a9218d8cedcfadbc888f4a7e96d7e2d`

---

## 0. Three of §A.7's eight components are ONE thing here, and it was verified rather than inherited

§A.7 lists the video VAE decoder, the video VAE device kernels and the tiled-decode
buffer as three independent rows. In this tree they cannot be separated, and the
three measurements that say so were RUN at this row's base:

```sh
grep -c 'RequireF32(' src/vt/cpu/cpu_ltx2_vae.cpp                              # 11
grep -c 'RequireF32(' src/vt/cuda/cuda_ltx2_vae.cu                             # 11
grep -c 'VaeKernels(' src/vllm/model_executor/models/ltx2_video_vae.cpp        # 14
grep -c 'DType::kF32'  src/vllm/model_executor/models/ltx2_video_vae.cpp       # 22
grep -c 'Ltx2VideoDecodeStreaming(' src/vllm/multimodal/ltx2_video.cpp         # 1
```

**(a) The device kernels ARE the decoder's arithmetic on the shipping path.**
Every stage between the convolutions dispatches through `vt::OpId::kLtx2Vae`, and
both arms refuse anything but f32 by name on eleven entry points each.
`ltx2_video_vae.cpp:459` states the rule that makes the CPU arm the shipping one:
"`queue == nullptr` means the CPU queue, NOT 'the old host path'. There is one
code path and the device is a property of the queue." The single production call
passes no queue, so the CPU arm's eleven refusals sit on the production path.

**This is cheap, which is why it is absorbed rather than deferred.** Every entry
in `Ltx2VaeDeviceKernels` already takes a `vt::DType dtype` and already refuses
on it. The work is filling a branch behind a parameter that exists.

**(b) The tiled buffer is inside the wave, by upstream's own construction.**
`conv_video_decoder.py:282` reads `output_dtype = sample.dtype`, `:357` returns
`sample.to(output_dtype)`, `:427-431` allocates the group buffer at
`dtype=latent.dtype` and `:526` allocates the weights buffer as
`torch.zeros_like(buffer)`. Those are the same tensor's dtype, and the tiled path
is the only production route into the decoder.

**(c) The video VAE ENCODER is genuinely separable** — different route, different
weights bag, and the kernels' `dtype` parameter lets an f32 encoder and a bf16
decoder share one table. But `ltx2_video_vae_encoder.h:57` says the encoder's
production arm "is owed with the decoder's", and this row lands the decoder's and
not the encoder's. That sentence becomes false at this merge commit and is
repaired here (§6.6).

---

## 1. Scope

**IN.**

* `include/vllm/model_executor/models/ltx2_video_vae.h`,
  `src/vllm/model_executor/models/ltx2_video_vae.cpp` — `VaeStore`, `Volume`,
  `VaeWeightCache`, `VaeScratch`, `Conv3dThroughSeam`, `VaeAddInPlace`, `Silu`,
  `PixelNorm`, `ApplyNorm`, `TimestepEmbedding`, `FeedSpatialNoise`, `ApplyAdaLn`,
  `AttnBlock3d`, `Ltx2ConvVideoDecode`.
* `include/vllm/model_executor/models/ltx2_video_vae_kernels.h` and
  `src/vt/cpu/cpu_ltx2_vae.cpp` — the eleven CPU bf16 branches behind the
  `vt::DType dtype` each entry already takes.
* `include/vllm/model_executor/models/ltx2_tiling.h`,
  `src/vllm/model_executor/models/ltx2_video_vae_tiled.cpp` — the group buffer,
  the weights buffer and the blend.
* `Ltx2LoadVaeWeights` (`ltx2_loader.cpp`, `ltx2_loader.h`) gains a
  `vt::DType compute_dtype`; only the video-decoder call site asks for `kBF16`.
* `src/vllm/multimodal/ltx2_video.cpp` — the arm selection and the
  `vae_decode_not_bf16` / `vae_decode_values` counters on the render path.
* `scripts/gen-ltx2-vae-goldens.py`, `tests/vllm/models/test_ltx2_vae.cpp`,
  `tests/vllm/multimodal/test_ltx2_video.cpp`.

**OUT, and owed by name in `## Owed`.** The video VAE **encoder**
(`ltx2_video_vae_encoder.h:52-57`, `ImageConditioner` at `distilled.py:120-122`).
The **latent upsampler** (`ltx2_upsampler.h:66-70`, `distilled.py:138-141`) and
the **duration head** (`ltx2_duration_head.h:55-58`, `distilled.py:163-165`). The
**FP8/NVFP4** arms, which are A22. The **audio VAE**, whose f32 §A.7 excludes
deliberately because `ltx2_audio_vae.cpp:7-12` argues it from `vocoder.py:575-580`
rather than owing it. The **diffusion decoder**, refused by name with its own row.
The **CUDA arm of `kLtx2Vae`** — see §5.6, where the reason is a reachability
argument and not a budget one.

`.agents/specs/ltx25-completion-scope.md` is operator-owned and this row does not
edit it. §0's finding is reported under `## Owed` for the operator.

**HARD DEPENDENCY, satisfied.** `Ltx2VaeWeights::bf16`, `::dtype`, `::GetBf16`,
`::HasBf16`, `::Count` and `::Bytes` are wave 2's and are on `main` at
`77704c8d0` (`ltx2_audio_vae.h:68-107`). This row bases on that merge.

---

## 2. Upstream anchors, read at `fd4ded7f` with the checkout's identity asserted

`git -C ~/_git/LTX-2 remote get-url origin` is `https://github.com/Lightricks/LTX-2.git`,
`rev-parse HEAD` is `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, `status --porcelain`
is empty. All three were run before anything below was trusted.

| what | anchor | value |
|---|---|---|
| the pipeline's ONE dtype | `ltx-pipelines/.../distilled.py:109` | `self.dtype = torch.bfloat16` |
| the decoder is constructed with it | `distilled.py:146-149` | `self.dtype` at `:148` |
| the latent follows the WEIGHTS on entry | `conv_video_decoder.py:283-284` | `weights_dtype = next(self.parameters()).dtype`; `sample.to(weights_dtype)` |
| the return follows the LATENT | `conv_video_decoder.py:282, 357` | `output_dtype = sample.dtype`; `return sample.to(output_dtype)` |
| the noise is drawn AT THE ACTIVATION DTYPE | `conv_video_decoder.py:288-294` | `torch.randn(..., dtype=sample.dtype, ...)` |
| the resnet spatial noise too | `resnet.py:112-117` | `torch.randn(spatial_shape, ..., dtype=dtype, ...)`, then `spatial_noise * per_channel_scale`, then `+` |
| the per-channel stats NARROW to the activation | `ops.py:76-79` | `.view(1,-1,1,1,1).to(x)` on both buffers |
| **the timestep itself is built at the activation dtype** | `conv_video_decoder.py:304-305` | `torch.full((B,), self.decode_timestep, ..., dtype=sample.dtype)` |
| **and the scale multiplier narrows to it** | `conv_video_decoder.py:313` | `timestep * self.timestep_scale_multiplier.to(sample)` |
| the timestep embedding takes the activation dtype | `conv_video_decoder.py:331-334` | `hidden_dtype=sample.dtype` |
| the final AdaLN table narrows too | `conv_video_decoder.py:336-337` | `.to(device=sample.device, dtype=sample.dtype)` |
| the resnet AdaLN table narrows | `resnet.py:133-135` | `.to(device=..., dtype=hidden_states.dtype)` |
| PixelNorm is raw eager ops, eps default 1e-8 | `common/normalization.py:22, 37-40` | `mean_sq = torch.mean(x**2, dim, keepdim=True)`; `rms = torch.sqrt(mean_sq + eps)`; `x / rms` |
| `_RMSNorm2D` forms its GAIN first | `video_vae/attention.py:23` | `F.normalize(x, dim=1) * (self.scale * self.gamma)` |
| the tiled buffer takes `latent.dtype` | `conv_video_decoder.py:427-431` | `torch.zeros(..., dtype=latent.dtype)` |
| the tiled WEIGHTS buffer narrows with it | `conv_video_decoder.py:526` | `torch.zeros_like(buffer)` |
| the tile masks stay f32 | `conv_video_decoder.py:548, 554`; `tiling.py:425` | `dtype=torch.float32` |
| the VAE attention backend | `video_vae/attention.py:50, 53` | `AttentionFunction.PYTORCH`, resolved once in `__init__` |

**There is no f32 pin on the conv decoder's forward path, and that is measured
with a control rather than asserted from a failed grep.**
`grep -rn 'float32\|\.float()\|autocast' packages/ltx-core/src/ltx_core/model/video_vae/`
returns hits, so the pattern works. Inside the conv decoder's own files the only
ones are the two **tiling-mask** lines and one **loader-time** gate fold in
`model_configurator.py`. Everything else is `diffusion_video_decoder.py` or
`transformer/` — the `NADiffusionDecoder` this tree refuses by name — or
`video_vae.py`'s ENCODER tiling. So the audio VAE's argued f32
(`vocoder.py:575-580`) does not extend here.

---

## 3. The local side, at `77704c8d0`

**The debt is a header comment, not a refusal, which is why §A.7's sweep could
not see it.** `ltx2_video_vae.h:47-54`, `ltx2_video_vae.cpp:25-44` and `:75-78`
record it, and `:41-44` adds the sentence that makes this row necessary: the
generator's `fill_from_stream` casts every upstream parameter to f32, so the
oracle itself runs f32 and a dtype comparison against it is vacuous by
construction.

**Exactly ONE production route reaches the decoder, it goes through the tiled
path, and it passes NO queue.** `Ltx2VideoEngine`'s render calls
`Ltx2VideoDecodeStreaming` once, inside the `decode.video.chunk` phase scope, with
no `queue` argument. So the conv video VAE decode runs on the CPU queue on every
build of this project, and the CPU kernel arm is the only arm production reaches.
`Ltx2VideoDecode` and `Ltx2ConvVideoDecode` have no caller outside the header,
this file's own definitions and `ltx2_video_vae_tiled.cpp:123`.

**The storage is float-typed throughout.** `VaeStore::Alloc` allocates `n` floats,
`ptr()` returns `float*`, `VaeWeightCache::Get` returns `const float*`, and
`VaeAddInPlace` builds its two `vt::Tensor`s as `vt::DType::kF32` literally.

**The loader widens.** `Ltx2LoadVaeWeights` expands every checkpoint `BF16` tensor
through `Bf16ToF32` into a `std::vector<float>`. Seven call sites share it, so it
gains a `vt::DType compute_dtype` defaulted to `kF32` and only the video-decoder
site asks for `kBF16`.

---

## 4. What upstream's bf16 decoder actually computes in — EXECUTED, not read

Every table below was produced by RUNNING the pinned modules on CPU with torch
`2.11.0+cu130` on 2026-09-03. Counts are bf16-word mismatches against upstream's
own output; `separating` is how many of the listed alternatives the probe can
tell apart from upstream. **A probe that separates nothing is reported as such.**

Three norms in this pipeline now have three different rules: wave 1's tower norm
squares to bf16 before accumulating, wave 2's `rms_norm` is all-f32 with the f32
epsilon, and this decoder's `PixelNorm` is a fully bf16 chain with the **bf16**
epsilon. No single rule covers them and reading the source separates none of them.

### 4.1 `PixelNorm` at bf16 is a FULLY BF16 CHAIN

`normalization.py:37-40` is raw eager arithmetic on the activation tensor, so
every intermediate materializes in bf16. `[2,32,3,5,5]`:

| hypothesis | O(1) | 2^-7 |
|---|---|---|
| **fully bf16 chain, bf16 eps** | **0 / 4800** | **0 / 4800** |
| all-f32 accumulate, f32 eps, divide | 1246 | 1424 |
| all-f32 accumulate, f32 eps, rsqrt-multiply | 1246 | 1424 |
| bf16 square, f32 mean, f32 eps | 1244 | 1434 |
| f64 accumulate | 1246 | 1424 |

`separating = 4` at both scales, so this is not a scale artefact. The port rounds
after each of: the square, the mean, the epsilon add, the sqrt and the divide —
five roundings where the f32 kernel has none.

### 4.2 …but the MEAN inside it is a BLOCKED reduction

`torch.mean` on bf16 does not accumulate in bf16. Against upstream's own
`torch.mean(x**2, dim=1)` on bf16 squares:

| C | f32 accumulate | sequential bf16 accumulate | f64 accumulate |
|---|---|---|---|
| 32 | **0 / 32** | 20 / 32 | **0 / 32** |
| 128 | **0 / 32** | 28 / 32 | **0 / 32** |

The kernel widens internally and rounds only the OUTPUT. "Fully bf16 chain" names
the ROUNDING POINTS, never the accumulator width. `separating = 1`; f32 and f64
accumulation are indistinguishable here and the row says so rather than claiming
a gate on the width.

### 4.3 The PixelNorm epsilon, with the probe BUILT to isolate it

The first form of this probe compared a bf16 chain against an f32 tail, which
moves the whole arithmetic and not the epsilon. Isolated properly — identical
bf16 chain, only the width of the epsilon ADD varying — on `[2,32,2,8,8]`:

| row scale | bf16 eps (upstream) | eps added in f32 then rounded back | eps removed entirely |
|---|---|---|---|
| 2^-0 | **0 / 8192** | 0 | 0 |
| 2^-6 | **0** | 0 | 0 |
| 2^-10 | **0** | 0 | 5579 |
| 2^-12 | **0** | 0 | 8192 |
| 2^-14 | **0** | **842** | 8192 |

The generator ships the same probe at the decoder's own channel count on a
[1,8,2,3,3] volume, where the separating counts are 0, 0 and **7 of 144** for the
width and 0, **144** and **144** for the read. The numbers move with the shape;
the polarity does not, and it is the polarity the case asserts.

`bf16(1e-8) = 1.0011717677116394e-08`.

**This is wave 1's failure written out in advance.** At ordinary magnitude nothing
separates, so a golden taken on the shipped fixture gates the epsilon at zero —
the mute switch. The WIDTH question needs rows at **2^-14**; the "is it read at
all" question needs **2^-10**. The generator lays both, emits the rejected answer
beside upstream's, and refuses to emit when `separating == 0`.

Note the polarity against wave 2: the connector's `rms_norm` epsilon is the
**f32** `1e-6` because it lands in an f32 accumulator; this one is the **bf16**
`1e-8` because it lands in a bf16 add. Same word, opposite answer, two files apart.

### 4.4 `nn.GroupNorm` — the AFFINE narrows, and the first form of this probe could not see it

The first form read the module's parameters AFTER `.to(bfloat16)` had already
narrowed them in place, so both hypotheses were the same tensor and the probe
reported `separating = 0`. Capturing the f32 parameters first:

| hypothesis | C=32 G=4 | C=128 G=32 | C=32 G=1 (`norm3`) |
|---|---|---|---|
| **f64 statistics, bf16-NARROWED affine, one store round (upstream)** | **0 / 1600** | **0 / 24576** | **0 / 1600** |
| torch's own f32 GroupNorm on the widened input, round once | 463 | 6664 | 461 |
| f64 statistics + affine, f32 weight and bias | 463 | 6664 | 461 |
| f32 statistics + affine, f32 weight and bias | 463 | 6664 | 461 |

So the statistics' width is invisible (f32 and f64 agree everywhere) and **the
affine operands' width is 27-29% of the output**. Our kernel already accumulates
in f64 and needs no change there; what it needs is the narrowed weight and bias,
which is what the loader arm delivers. `separating = 3`.

### 4.5 `F.silu` at bf16 is a WIDENED evaluation with one store rounding

| hypothesis | mismatches |
|---|---|
| **`bf16(F.silu(f32))`** | **0 / 4000** |
| `bf16(x/(1+exp(-x)))` evaluated in f32 | **0 / 4000** |
| fully bf16 `x/(1+exp(-x))` | 1155 / 4000 |

`separating = 1`. This is what `Ltx2Silu`'s existing bf16 branch already does
(`cpu_ltx2.cpp`: `Load` to f32, evaluate, `Store` with one rounding), so the SiLU
needs no new code — only the dtype passed to it. The two zero rows are recorded as
NOT separated rather than as two independent confirmations.

### 4.6 The final AdaLN rounds THREE times

`sample * (1 + scale) + shift` on bf16 operands:

| hypothesis | mismatches |
|---|---|
| **three roundings: `(1+sc)`, the multiply, the add** | **0 / 512** |
| all f32, one round | 198 / 512 |
| `(1+sc)` in bf16, then a fused f32 mul-add | 162 / 512 |

`separating = 2`. This is wave 1's add/mul asymmetry in a third place.

### 4.7 `per_channel_statistics.un_normalize` NARROWS THE STAT BUFFERS

`ops.py:76-79` applies `.to(x)` to both registered buffers, so at bf16 the
statistics themselves round first. Our port narrows the f32 stats once per channel
into a `float` and multiplies (`ltx2_video_vae.cpp:1203-1208`), which keeps them
f32. Measured with the f32 buffers captured BEFORE `.to(bfloat16)`:

| hypothesis | C=16 | C=128 |
|---|---|---|
| **stats narrowed to bf16, multiply rounds, add rounds (upstream)** | **0 / 288** | **0 / 4096** |
| stats kept f32, multiply rounds, add rounds (**what our port has**) | **109 / 288** | **1294 / 4096** |
| stats f32, one fused f32 expression | 125 / 288 | 1587 / 4096 |

`separating = 2`. **32% of the decoder's very first arithmetic is wrong if the
activation dtype is flipped without narrowing the statistics**, and no token gate
can see it.

### 4.8 `_RMSNorm2D` — RESOLVED, and only at a width where it can separate

The draft this row started from left `F.normalize` at bf16 unresolved with five
hypotheses rejected and its best 7.4% away, and made it a stop condition. The
sixth hypothesis is that the DENOMINATOR is rounded to bf16 before the divide:
`F.normalize` is `input / input.norm(...).clamp_min(eps)`, and `norm` on a bf16
tensor returns bf16. `[3,64,5,5]`:

| hypothesis | O(1) | 2^-8 |
|---|---|---|
| **f32 accumulate, denominator ROUNDED to bf16, then divide** | **0 / 4800** | **0 / 4800** |
| f32 accumulate, f32 denominator, f32 divide, round once | 1379 | 1469 |
| f64 accumulate, f32 denominator, round once | 1379 | 1469 |
| fully bf16 chain | 749 | 779 |

`separating = 3`. The stop condition is retired with a measurement rather than a
tolerance.

The whole `_RMSNorm2D` block then separates ONLY at a channel count whose square
root is not a power of two, because `self.scale * self.gamma` is formed first:

| hypothesis | C=64 (`sqrt`=8) | C=48 (`sqrt`=6.9282) |
|---|---|---|
| **gain formed in bf16 first, ONE multiply (upstream)** | **0 / 4800** | **0 / 3600** |
| gain formed in f32 then rounded | 0 | 0 |
| normalize, then `*sqrt(C)`, then `*gamma` (**our port's order**) | **0** | **1135 / 3600** |

At C=64 `separating = 0` and the probe gates nothing; at C=48 our port's ordering
is 32% wrong. A probe built on a power-of-two channel count is a mute switch, and
the row states that rather than claiming a gate it does not have.

### 4.9 The VAE attention is served by FLASH, and FLASH is ~38% away from MATH

`AttnBlock3D` pins `AttentionFunction.PYTORCH` (`attention.py:50`) and resolves it
once in `__init__` (`:53`). Its docstring argues that the single-head
`head_dim == in_channels` exceeds FlashAttention's limit so SDPA falls back to an
efficient or math kernel. **That is not what happens on this CPU.** With
`sdpa_kernel`, `[2,1,64,head_dim]`:

| head_dim | bare vs MATH | bare vs FLASH | MATH vs `bf16(f32 attention)` |
|---|---|---|---|
| 64 | 3074 / 8192 (38%) | **0** | 5 / 8192 |
| 128 | 5996 / 16384 (37%) | **0** | 9 / 16384 |
| 256 | 12463 / 32768 (38%) | **0** | 3 / 32768 |

FLASH serves the call at every width tested. `SDPBackend.MATH` is the only
reproducible oracle and it is close to — but **not** bit-equal to — an
f32-accumulated attention, unlike the connector's (wave 2 measured 0/384 there).
So each bf16 attention golden is emitted TWICE, once from the module as
constructed and once with the attention callable pinned to `SDPA_MATH`; the port
is held to MATH and the distance to the unpatched module is printed per arm.

### 4.10 `nn.Conv3d` at bf16 is NOT bit-reproducible from a different reduction order

Upstream's bf16 `F.conv3d` against our `cpu_conv3d` contract (f32 accumulate,
`LoadF32At`/`StoreF32At`, bias INSIDE the accumulator, one rounding on store —
`src/vt/cpu/cpu_conv3d.cpp:59-75, 102-128`), six seeds of
`[1,32,4,9,9] * [32,32,3,3,3]`:

| hypothesis | mismatches |
|---|---|
| f32 accumulate, bias inside, one round (**our CPU kernel**) | 3 / 18816 (0.0159%), max abs diff 0.0078125 |
| f64 accumulate, bias inside, one round | 2 / 3136 on seed 0 against our 3 — no better |
| f32 accumulate, **bias added after rounding** | 5211 / 18816 (rejected) |

The residue is torch's own blocked reduction order, not an accumulator-width
error, and an f64 accumulator does not improve on f32. The bias-placement
hypothesis separates strongly and our kernel already has it right. The 1x1x1
`Linear3d` case is **0 / 4096** — bit-exact — because a one-tap reduction has no
order to differ in.

### 4.11 The PixArt timestep embedding must be computed IN bf16

`conv_video_decoder.py:331-334` passes `hidden_dtype=sample.dtype`. On
`embedding_dim=32` the bf16 module differs from `bf16(the f32 module)` on **12 of
32** values. Our port keeps the frequency table in f64 as "a constant precompute
rather than a data path" (`ltx2_video_vae.cpp:58-61`); at bf16 that choice is
wrong for a third of the embedding, and the annotation moves with the arm.

### 4.12 `scaled_timestep` is a BF16 product, and the draft missed it

`conv_video_decoder.py:304-305` builds the timestep tensor with
`dtype=sample.dtype`, and `:313` multiplies it by
`self.timestep_scale_multiplier.to(sample)`. Both operands are bf16 at bf16. Our
port computes `scaled_timestep` in `double`. Measured on `timestep = 0.05`:

| multiplier | f64 product | bf16 chain | |
|---|---|---|---|
| 1000 | 50.0 | 50.0 | same |
| 1.0009765625 | 0.050048828125 | 0.050048828125 | same |
| 7.3 | 0.365 | 0.365234375 | **DIFFERS** |

A multiplier that is exactly representable in bf16 hides this; the shipped
checkpoint's value is not known to this row (`## Owed`), so the arm narrows both
operands as upstream does rather than relying on the value being benign.

### 4.13 `_feed_spatial_noise` rounds the PRODUCT, then the ADD

`resnet.py:114-117` forms `spatial_noise * per_channel_scale` as its own tensor
and adds it. Two roundings, not one:

| hypothesis | mismatches |
|---|---|
| **the product rounds, then the add rounds** | **0 / 144** |
| one fused f32 expression | 8 / 144 |

`separating = 1`.

### 4.14 The tiled buffer and its WEIGHTS both follow the latent

`buffer` is `dtype=latent.dtype` (`:427-431`) and `weights = torch.zeros_like(buffer)`
(`:526`), so both are bf16. `ltx2_tiling.h:88-94` records the buffer and the masks
and does NOT record the `zeros_like`; that is the record's gap as well as the
tree's, and this row adds the sentence.

`tile(bf16) * mask(f32)` promotes to f32 (`tiling.py:425`) and the `+=` into the
bf16 buffer rounds it straight back, so the promotion is a per-tile intermediate
only. The buffer's dtype is invisible on the first write into a zeroed buffer and
reachable only where more than one tile or temporal group contributes.
`ltx2_tiling.h:70-71` records that one-tile routing is safe below 81 frames and
NOT within 81..120, and `:84-86` records that the 448x256/25f geometry is one
tile. **Whether the shipped bench geometries reach the multi-tile blend is
UNMEASURED and is in `## Owed`, not assumed.**

### 4.16 THREE SCALARS, THREE ANSWERS — found during implementation, not before it

A Python float multiplied into a bf16 tensor does NOT narrow. torch's scalar path
uses `opmath_t<BFloat16> = float`, so the scalar reaches the multiply at f32 and
only the result rounds. On `noise * 0.025 + (1 - 0.025) * sample` over 2000
values, narrowing the two scalars first is wrong on **576 of 2000** — `bf16(0.975)`
is 0.9765625, a whole 2^-9 away.

The same function narrows a REGISTERED BUFFER, because `.to(x)` on a tensor does
(§4.7, 1294 of 4096 if skipped). And `PixelNorm`'s epsilon — also a Python float —
IS narrowed for its ADD (§4.3, 7 of 144 at 2^-14). Three scalars, three answers,
in two files.

**This was not in §4 before implementation and it was the row's one real defect.**
It is A24 wave 1's `t + 1e-6` versus `t * sqrt(8/6)` finding in a third component,
and it is why §10's stop condition on guessing an arithmetic rule is not
decorative.

### 4.15 Simulating the intended C++ loops against upstream

Each kernel loop this row will write was first written as an f32-scalar
simulation with an explicit bf16 round at each intended point and checked against
upstream's own bf16 output, with the alternative it rejects printed beside it:

| loop | as written | rejected alternative |
|---|---|---|
| `PixelNorm` @ 2^-14 | 0 / 144 | f32 epsilon: 15 / 144 |
| `PixelNorm` @ O(1), 2^-7 | 0 / 144 | f32 epsilon: 0 / 144 (does not separate) |
| `un_normalize` | 0 / 144 | f32 statistics: 45 / 144 |
| AdaLN | 0 / 64 | one rounding: 24 / 64 |
| `_RMSNorm2D` C=48 | 0 / 768 | our port's multiply order: 233 / 768 |
| spatial noise | 0 / 144 | fused: 8 / 144 |
| GroupNorm C=32 G=4 | 0 / 1600 | f32 affine: 509 / 1600 |
| GroupNorm C=32 G=1 | 0 / 1600 | f32 affine: 477 / 1600 |

---

## 5. Design

### 5.1 The weights carry their own dtype, on wave 2's struct

`Ltx2LoadVaeWeights` gains `vt::DType compute_dtype = vt::DType::kF32`. At `kBF16`
it populates `Ltx2VaeWeights::bf16` with the checkpoint's own 16-bit words instead
of expanding them through `Bf16ToF32`, and sets `Ltx2VaeWeights::dtype`. A
checkpoint that stores F32 under a `kBF16` request is narrowed once at load, which
is what upstream's `.to(dtype)` does to a module built from an f32 state dict. The
other six call sites keep the default and do not move one byte.

§4.4 and §4.7 are why this is most of the arithmetic work: the GroupNorm affine
and the per-channel statistics are 27-32% of their outputs, and both are fixed by
the weights arriving already narrowed.

### 5.2 The kernel seam is FILLED, not extended

Every `Ltx2VaeDeviceKernels` entry already carries `vt::DType dtype` and every one
refuses. The change is eleven CPU branches behind that existing parameter, plus:

* `vt::Conv3d`'s CPU arm already serves bf16 (`cpu_conv3d.cpp:59-75` widens on
  load and rounds on store), so `Conv3dThroughSeam` only passes the dtype through.
* `vt::Add`'s CPU arm already accepts bf16 -- `vt::Add` admits any `IsOutFloat`
  output (`src/vt/ops.cpp`, the `"add: float in, f32/bf16 out"` check) and
  `AddKernel` (`src/vt/cpu/cpu_layernorm.cpp`) is `LoadF32At`/`StoreF32At` with
  ONE store rounding, which is exactly the rule measured below. So
  `VaeAddInPlace` only passes the dtype through.
* `kLtx2`'s `silu` already has a bf16 branch (`cpu_ltx2.cpp`), and §4.5 shows it
  is the right one, so `Silu` only passes the dtype through.

`VaeStore::Alloc` takes a `vt::DType` and sizes its host bytes by the element
width; `Volume` carries the dtype; `VaeWeightCache` and `VaeScratch` return
`const void*` and serve the requested arm. `VaeAddInPlace`'s two literal
`vt::DType::kF32` tensors take the volume's dtype. **The f32 arm stays reachable**
— it is what every committed golden is measured against, and deleting it deletes
the reference.

Five of the eleven kernels — `depth_to_space`, `frame_slice`, `channel_repeat`,
`unpatchify` and `pad` — are pure shape movement. They copy elements and round
nowhere, so their bf16 branch is the same gather over a 2-byte element.

### 5.3 What the bf16 arm computes

Resolved from the weights, never from a flag, so the arm follows the checkpoint
the way upstream's does:

* the per-channel statistics **narrow to bf16 before the multiply**, and the
  multiply and the add each round (§4.7);
* the noise blend rounds at each step, on the f32 stream this row does NOT change
  (§7, #2780);
* `PixelNorm` rounds after the square, the blocked-f32 mean, the **bf16** epsilon
  add, the sqrt and the divide (§4.1, §4.2, §4.3);
* `GroupNorm` keeps its f64 statistics and applies the **bf16-narrowed** weight
  and bias, one store rounding (§4.4);
* every convolution keeps `cpu_conv3d`'s f32 accumulate with the bias INSIDE the
  accumulator and one rounding on store (§4.10);
* the SiLU is `kLtx2`'s existing bf16 branch (§4.5);
* the timestep is narrowed, the scale multiplier is narrowed, and their product
  rounds (§4.12); the embedding is computed in bf16 with its f64 frequency table
  narrowed at the point it enters the arithmetic (§4.11);
* both AdaLN sites round three times (§4.6);
* `_RMSNorm2D` accumulates the L2 in f32, **rounds the denominator to bf16**,
  divides, and multiplies by a per-channel gain formed as `bf16(sqrt(C) * gamma)`
  (§4.8);
* the attention is held to `SDPBackend.MATH` and its distance to the module as
  constructed is reported (§4.9);
* `unpatchify`, depth-to-space, the frame slice, the channel repeat and the pad
  are shape movement: they narrow with the storage and round nowhere.

### 5.4 The tiled path follows the latent, and the masks do not

`ChunkBuffer` and the weights buffer both take the volume's dtype (§4.14). The
masks stay f32, each `+=` rounds to the buffer's dtype, and the quotient is taken
at the buffer's dtype. `ltx2_tiling.h:88-94` gains the `zeros_like` sentence it is
missing.

### 5.5 The activation CONTAINERS narrow here, unlike wave 2's

Wave 2 kept `std::vector<float>` holding bf16 values and recorded why: its
activations are ~23 MB against the ~4 GB its weights arm returned. **That argument
does not transfer.** The decoder's intermediates are full-resolution video volumes
at the base channel width, so the activations are the cost, and narrowing the
storage is most of the point. That is what forces §5.2.

### 5.6 The CUDA arm keeps refusing, and the reason is REACHABILITY, not budget

`src/vt/cuda/cuda_conv3d.cu:154-156` refuses f16/bf16 storage by name and records
it owed under #1007. Every convolution of the decode goes through `vt::Conv3d`, so
a bf16 volume on a CUDA queue cannot get past `conv_in`. Eleven bf16 branches in
`cuda_ltx2_vae.cu` would therefore be code that nothing can reach — the exact
shape AGENTS.md `## Nothing lands dead` forbids — so this row does not write them.

The `kLtx2Vae` CUDA arm keeps its refusal, with the message updated to name
`cuda_conv3d`'s missing bf16 storage as what has to land first. **The refusal and
the route predicate are the same predicate**, which is the trap this project has
already paid for: the arm is chosen by asking the same question the kernel
refuses on, so a device that cannot serve bf16 is never handed a bf16 volume.
Production is unaffected — §3 measured that the render decodes on the CPU queue on
every build — and a bf16 CUDA decode is recorded owed with the lease it needs.

---

## 6. Tests

### 6.1 RED FIRST, through the production entry point

`Ltx2ConditioningTrace` (`include/vllm/multimodal/ltx2_video.h:624`) already
carries `tower_video_not_bf16` / `tower_video_values` and
`connector_video_not_bf16` / `connector_video_values`. This row adds
`vae_decode_not_bf16` / `vae_decode_values`, sampled with the same
`CountWiderThanBf16` helper **on the decoded chunk inside the
`Ltx2VideoDecodeStreaming` sink** — the one production route (§3), not a
hand-constructed decoder.

The red is captured before `src/` is touched, on the f32 tree, by asserting the
green condition. The literal red and the literal green go in `## Outcome`.

### 6.2 The goldens come from upstream RUN IN BF16

`scripts/gen-ltx2-vae-goldens.py` gains a bf16 half of section 5 running the SAME
arm as the f32 half with the module `.to(torch.bfloat16)` on a bf16 latent. Every
parameter, tolerance and failure case of the f32 half is preserved and one dtype
changes.

**Two harness adaptations, both recorded because both change something.**
`fill_from_stream` must stop casting parameters to f32 on the bf16 half —
`ltx2_video_vae.cpp:41-44` records that this cast is why the existing oracle
cannot see a dtype at all, so leaving it makes the new goldens vacuous by
construction. And the patched `torch.randn` must honour the `dtype=` keyword
upstream passes, returning the shared f32 stream NARROWED, which is this row's
deliberate divergence from `torch.randn(dtype=bfloat16)` (§7, #2780).

The section also emits §4's discriminators as goldens in their own right: the
2^-14 and 2^-10 epsilon probes with their rejected answers, the AdaLN's three
roundings with its two rejected alternatives, the narrowed per-channel statistics
with the f32-stats answer beside them, the narrowed GroupNorm affine with the
f32-affine answer, `_RMSNorm2D` at a channel count whose square root is not a
power of two with our port's ordering as the rejected answer, and the bf16
timestep embedding with the `bf16(f32 module)` answer.

**Every probe prints `separating` and refuses to emit when it is 0.** §4.3, §4.8
and §4.5 are the three places this rule is not decorative.

### 6.3 The value gate

Bit-exactness is attempted first and used where §4 measured it: PixelNorm, both
AdaLN sites, `un_normalize`, GroupNorm, SiLU, `_RMSNorm2D`, the spatial noise, the
timestep embedding and `Linear3d` are all 0-of-N against a stated form and are
gated at `worst == 0`. §4.10's convolution (3/18816) and §4.9's attention are NOT
bit-exact from a different reduction order. For those two the bound is derived
from the format — bf16 unit roundoff `2^-8` relative to the golden's own magnitude
— **never fitted to the result**, and each case prints its measured margin beside
the bound. Wave 2's `## Outcome` is the warning: it planned exactly such a bound,
measured that three of five defect mutations moved less than it, and deleted it.
**If a mutation in §6.5 moves less than the bound, the bound goes and the case is
held bit-exact or reported as ungated.**

### 6.4 The DTYPE gate, which is the one that cannot be faked

1. **The weights.** `Ltx2LoadVaeWeights` is materialized at both dtypes from one
   synthetic checkpoint and `Ltx2VaeWeights::Bytes()` compared: the bf16 arm must
   be **exactly half** and its `tensors` map empty. Measured on the same input, so
   no number is quoted.
2. **The output, in BOTH directions, in one case.** Every float the bf16 decode
   returns must satisfy `BF16ToF32(F32ToBF16(v)) == v`, and the f32 arm on the same
   fixture must **fail** it on most of the stream. An f32 path cannot pass the
   first; a fixture that lost its sub-bf16 detail cannot pass the second.
3. **The intermediates, not only the output.** A decode that computes in f32 and
   rounds once at the exit passes (2) while moving twice the bytes the whole way.
   So the volume's element width is asserted at the kernel boundary: the count of
   `kLtx2Vae` dispatches that received `kBF16` must equal the total dispatch count.
4. **The tiled buffer.** The group buffer's and the weights buffer's element width
   are asserted, and the blend quotient is compared against upstream's
   bf16-weights answer with the f32-weights answer emitted as the rejected one.
5. **At the engine level**, `vae_decode_not_bf16 == 0` on the render path, with the
   f32 arm reading the whole stream.

### 6.5 The mutations a fresh reviewer applies

* **Delete the production call site**: revert the render's loader call to the f32
  arm. §6.4(5) must red; the digests, absmax, frame bytes and determinism checks
  must **not**. This is the reachability proof AGENTS.md `## Nothing lands dead`
  asks for.
* **Keep the per-channel statistics in f32** (§4.7). The decode goldens must red.
* **Keep the GroupNorm affine in f32** (§4.4). The decode goldens must red.
* **Use `1e-8` instead of `bf16(1e-8)`** for PixelNorm (§4.3). Only the 2^-14 probe
  may red; the ordinary-magnitude goldens must stay green — which is what proves
  the probe is the instrument and the goldens are not.
* **Set the PixelNorm epsilon to 0** (§4.3). The 2^-10 probe must red. This
  separates "read" from "read at the right width".
* **Round the AdaLN once instead of three times** (§4.6). The goldens must red.
* **Restore `_RMSNorm2D`'s multiply order** (§4.8). The C=48 probe must red and
  the C=64 control must NOT — which is what proves the width was chosen for a
  reason.
* **Keep `F.normalize`'s denominator in f32** (§4.8). The goldens must red.
* **Compute the timestep embedding in f32 and round** (§4.11). The goldens must red.
* **Leave the tiled weights buffer f32** (§4.14). The multi-tile quotient golden
  must red; the single-tile one must not.
* **Delete §6.4(2)'s round-trip assertion** and run the bf16 cases against the f32
  arm. The value goldens alone must **not** be enough to red it.
* **Delete the `kBF16` dtype argument at three separated kernel dispatch sites.**
  §6.4(3) must red at each. A mutation proves only the sites it deletes.

### 6.6 One record repair this row owes

`ltx2_video_vae_encoder.h:57` says the encoder's production arm "is owed with the
decoder's". This row lands the decoder's and not the encoder's, so that sentence
becomes false at this merge commit and is corrected in the same change, naming the
encoder's own owed row instead.

---

## 7. Risks

* **THE NOISE STREAM IS REPLACED, NOT ROUNDED, AND THIS ROW DOES NOT TOUCH IT.**
  Both draw sites take the activation dtype (`conv_video_decoder.py:288-294`,
  `resnet.py:115`), and `torch.randn(dtype=bfloat16)` at a fixed seed is a
  DIFFERENT sequence from `torch.randn(dtype=float32)` at that seed — not the f32
  stream rounded. Mirroring it would change every captured render digest in this
  repository, including the adherence campaign's byte-identical control. That is
  [#2780](https://github.com/mudler/vllm.cpp/issues/2780) and it belongs to the
  developer. **This row keeps `Ltx2NoiseStream`'s f32 draw and narrows each drawn
  value**, which is a deliberate divergence from upstream, recorded here, in the
  commit that does it, and at the site. #2780 stays open.
* **The ~38% FLASH/MATH gap (§4.9) is larger than the connector's.** Holding the
  port to MATH is correct and is also an admission that the arm as upstream
  actually runs it is not gated. That distance is reported per arm rather than
  hidden.
* **bf16 is lossy and this row makes the decode less precise than it is today.**
  That is the point: upstream's answer is the bf16 one, and
  `ltx2_video_vae.cpp:75-78` already says the f32 file is a correctness reference
  and not the shipping path.
* **The CUDA arm is not measured, and now not written either (§5.6).** The refusal
  and the route predicate are the same predicate, so nothing on any path reaches
  a bf16 CUDA VAE kernel. A bf16 CUDA decode needs `cuda_conv3d`'s bf16 storage
  (#1007) first and a lease to measure.
* **The blend is invisible at one tile (§4.14).** A gate taken only on the shipped
  geometry may never reach the multi-tile effect, so §6.4(4) requires a multi-tile
  fixture, and whether the shipped bench geometries produce one is in `## Owed`.
* **`separating = 0` is reported in three places** (§4.2's accumulator width,
  §4.5's two zero rows, §4.8's C=64 control). Each is recorded as "the probe could
  not part these" rather than as a confirmation.

---

## 8. Gates

Every command is runnable and its output is a number someone can disagree with.
The `grep -c` values were RUN at `77704c8d0`, not written from memory.

```sh
# G0 — identity. Nothing in §2 or §4 is trustworthy if this is another revision.
git -C ~/_git/LTX-2 remote get-url origin      # https://github.com/Lightricks/LTX-2.git
git -C ~/_git/LTX-2 rev-parse HEAD             # fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
git -C ~/_git/LTX-2 status --porcelain         # empty

# G1 — the ONE production route, counted by string. Must stay 1; a second call
#      site means §3's reachability argument no longer holds.
grep -c 'Ltx2VideoDecodeStreaming(' src/vllm/multimodal/ltx2_video.cpp                              # was 1
grep -c 'Ltx2LoadVaeWeights(f, Ltx2VideoVaeDecoderKeyRules()' src/vllm/multimodal/ltx2_video.cpp    # was 1

# G2 — the kernel seam's refusals NARROW on the CPU arm and STAY on the CUDA one.
grep -c 'RequireF32(' src/vt/cpu/cpu_ltx2_vae.cpp    # was 11, becomes 0 (the branch is filled)
grep -c 'RequireF32(' src/vt/cuda/cuda_ltx2_vae.cu   # was 11, STAYS 11 (section 5.6)
grep -n 'this arm serves f32' src/vt/cuda/cuda_conv3d.cu

# G3 — the f32 literals this row replaces. A DROP with no bf16 branch added is a
#      deletion, not a port; read the diff, do not just watch the number fall.
grep -c 'DType::kF32' src/vllm/model_executor/models/ltx2_video_vae.cpp   # was 22

# G4 — the goldens are upstream's, regenerated at the pin, and NOT f32-cast.
python3 scripts/gen-ltx2-vae-goldens.py --ltx2 ~/_git/LTX-2 \
  --out tests/vllm/models/ltx2_vae_goldens.inc && git diff --stat
grep -c 'separating' tests/vllm/models/ltx2_vae_goldens.inc

# G5 — the focused suites, both arms, plus the tiling blend and the engine trace.
ctest --test-dir build -R 'ltx2' --output-on-failure

# G6 — the full gate.
scripts/agent-preflight.sh --staged
python3 scripts/check-pr-size.py --base origin/main --head HEAD
python3 scripts/agent-pr-body.py --pr <N>
```

**G3 needs the warning it carries.** A count that falls proves only that literals
left. `.agents/verification.md`'s standing lesson and this campaign's D11 cluster
are both about records a number agreed with and a tree did not.

---

## 9. Evidence

1. **The fourteen probes of §4**, run against `fd4ded7f` with torch `2.11.0+cu130`
   on CPU on 2026-09-03. Their literal output goes in `## Outcome` beside the
   claim each supports, with the rejected hypothesis and the separating count for
   every one — including the three that separate NOTHING.
2. **§4.15's simulation of every intended C++ loop** against upstream's own bf16
   output, with the rejected alternative beside it.
3. **The regenerated goldens**, whose header carries the upstream revision the
   generator read from git, and whose bf16 half does not cast parameters to f32.
4. **The measured byte counts of §6.4(1)**, printed by the case itself.
5. **The literal red of §6.1 and the literal green after**, in doctest's own output.
6. **The mutations of §6.5** with their literal assertion counts, each restored
   with `git checkout --` and re-verified green.

---

## 10. Stop conditions

* **The noise-stream decision (§7) is not this row's to make.** It is #2780. This
  row keeps the f32 stream and records the divergence; it never implements
  upstream's bf16 draw.
* **A probe stops separating.** Rebuild it; never emit a golden that cannot fail.
  §4.3, §4.5, §4.8's C=64 control and §4.2's accumulator width are the four places
  this is live, and each is stated as a limit rather than hidden.
* **A GPU lease is needed for anything here.** Stop and report; do not take one.
  The CUDA kernel measurement and any real-weights render are on the other side of
  that line.
* **An arm cannot be resolved by measurement.** Refuse it BY NAME with a message
  naming the missing part and record it owed. Never guess an arithmetic rule and
  never fit a tolerance to the port's own output.
* **Upstream turns out to pin f32 anywhere on the conv decoder's forward.** §2
  measured that it does not, with a control. If a re-measure disagrees, the row's
  premise is wrong.

---

## Owed

* **[#2853](https://github.com/mudler/vllm.cpp/issues/2853) — `test_ltx2_video_device_forward`
  is RED**, and this row's refusal is what it walks into. §5.6's device-bf16
  refusal (`ltx2_video_vae.cpp`, "a bf16 decode was requested on device ... only
  the CPU arm serves it") throws inside the pre-existing #1426 fake-accelerator
  case, which asserts as PRECONDITIONS that the platform is `kXPU` and that a
  backend is registered for it — so it drives exactly the path the refusal now
  closes. Found and filed by wave GDNCPUPORT
  ([#2845](https://github.com/mudler/vllm.cpp/issues/2845)), which touches no
  LTX2 file and reproduces it identically under `VT_GDN_CHUNKED=0` and `=1`;
  the refusal string is absent at that branch's pre-merge commit and present at
  `origin/main`. It is listed HERE rather than fixed there because the choice —
  load that case's VAE weights at f32, or land the device bf16 arm this section
  already owes — is this row's, not a GDN kernel row's.
* **The scope reconciliation.** §A.7's eight-row table splits the decoder, the
  device kernels and the tiled buffer into three rows this tree cannot separate
  (§0). `.agents/specs/ltx25-completion-scope.md` is operator-owned; this row
  reports the finding and does not edit it.
* **The noise draw at bf16** — [#2780](https://github.com/mudler/vllm.cpp/issues/2780),
  the developer's. This row's decode narrows the f32 stream instead, which is a
  named divergence from `conv_video_decoder.py:288-294` and `resnet.py:115`.
* **The video VAE ENCODER** (`ltx2_video_vae_encoder.h:52-57`, `ImageConditioner`
  at `distilled.py:120-122`), reached at `ltx2_video.cpp:3108` and `:3756`.
  §6.6 repairs its header's claim that it lands with this row.
* **The latent upsampler** (`ltx2_upsampler.h:66-70`, `distilled.py:138-141`) and
  the **duration head** (`ltx2_duration_head.h:55-58`, `distilled.py:163-165`).
* **The CUDA arm of `kLtx2Vae` at bf16** (§5.6). It needs `cuda_conv3d`'s bf16
  storage (#1007) to be reachable at all, and a lease to be measured.
* **The attention arm upstream ACTUALLY runs at the shipped widths.** The bf16
  goldens are taken under `SDPBackend.MATH`; on the gated fixture that is
  byte-identical to the module as constructed (0.0), but at the shipped widths
  FLASH serves the bare call and is 37-38% of words away from MATH (§4.9). Needs
  a lease and real weights.
* **The convolution's own bf16 rule.** `cpu_conv3d` keeps torch's contract but
  not torch's blocked association order, which is the deep arm's whole residue
  (3 to 5 of 8192 to 24576 at the fixture's shapes). Reproducing the order is its
  own row.
* **The video ENCODER's bf16 arm shares this file's `VaeStore`**, so it is a
  dtype argument away rather than a port -- but it has its own weights bag, its
  own route and no bf16 goldens, and it is not this row's.
* **`cuda_ltx2_vae.cu`'s header claim that it is "NEVER COMPILED, NEVER EXECUTED,
  ANYWHERE IN THIS PROJECT'S REACH"** is falsified by the `cuda-fat-build` CI job
  (`.github/workflows/ci.yml:1119-1156`), which builds the `vllm` target in
  `nvidia/cuda:13.3.0-devel-ubuntu24.04` with the TU unconditionally in the source
  list (`CMakeLists.txt:1854`). "Never EXECUTED" still holds. This row repairs the
  sentence where it edits that block and files the finding here.
* **Whether the shipped bench geometries reach the multi-tile blend** (§4.14),
  which decides whether the tiled buffer's dtype is observable in a gated render.
* **`Ltx2TrapezoidalMask1d` is correctly rounded and `torch.linspace` is not** --
  [#2816](https://github.com/mudler/vllm.cpp/issues/2816). The port builds its
  blend ramps with a double linspace and narrows once; torch's float32 CPU kernel
  computes the head as `start + step*i` and the tail as `end - step*(n-1-i)`, with
  the boundary set by the vectorization width, and the two disagree in the last
  f32 bit for 21 of the 31 lengths in n = 2..32, including 4, 7, 8 and 13. #2816
  carries the swept set; an earlier form of this bullet named only 4, 7 and 13,
  which was a spot check read as a measurement. At f32 the relative error is 6e-8
  against a 5e-6 band and nothing sees it; at bf16 it is a whole word, and a
  6-frame / 2-overlap temporal tile put 2 of 3888 blended outputs one bf16 ulp
  from upstream. §8's fixture uses an 8/4 temporal tile, which does not reach a
  disagreeing length, so the buffer gate measures the buffer. Deciding between
  mirroring torch's vectorization boundary and accepting the divergence with the
  mask case tightened from a band to bit-exactness needs its own row.
* **The tiled decode's own residue at UNEQUAL tile shapes.** §8's latent is six
  frames so that every tile decodes the same (4, 2, 2) latent. At five frames the
  split is [0,4) and [2,5), and the port then sat one bf16 ulp from upstream on 1
  of 3888 outputs -- the same blocked-association term the row records above for
  the convolution, surfacing at a shape the untiled arm does not cover. It is
  bounded by nothing here because §8 gates the equal-shape geometry; the unequal
  one is unmeasured beyond that single reading.
* **The video ENCODER refuses a bf16 bag by name** (`Ltx2ConvVideoEncode`), which
  is a refusal and not an arm. Its bf16 port is owed above.
* **`VaeStore::HostF32` hands a `float*` out of a `std::vector<uint8_t>`**, and
  callers write through it. `-O3` with the default `-fstrict-aliasing` is in the
  flags and `sanitize-cpu` does not catch this class. Alignment holds and every
  access is whole-buffer or through that one accessor, so it works today;
  `std::memcpy` cannot replace it without a write-back, because the encoder's
  gathers mutate the volume in place. The repair is a refactor of the encoder's
  host path, not a line, and it is not this row's.
* **The shipped `timestep_scale_multiplier` value** (§4.12), which decides whether
  the bf16 timestep product differs on the real checkpoint. Needs the checkpoint;
  `CHECKPOINT_ROOT` is not mounted on this box.
* **The decoder's own share of the VAE checkpoint bytes.** `docs/USAGE.md` records
  the whole video VAE file, which holds the encoder and the decoder;
  `Ltx2VideoVaeDecoderKeyRules()` takes only the `decoder.` and
  `per_channel_statistics.` prefixes and the decoder's fraction is UNMEASURED. Do
  not quote a decoder figure until the safetensors header is read.
* **A real-weights bf16 render against upstream**, which needs a lease and
  `tools/oracle/ltx2_oracle.py`.
* **The FP8 and NVFP4 arms**, which are A22.

## Now

`DONE`. Spec commit `f8e7024d3` precedes implementation commit `d1b87687d`.

---

## Outcome

**`DONE`.** Issue [#2786](https://github.com/mudler/vllm.cpp/issues/2786).
Spec commit `f8e7024d3` precedes the implementation commit `d1b87687d`, which is
the commit order that proves it came first.

### What was measured, including where the plan was wrong

* **§0's "one wave" conclusion holds** and was re-verified at this base, not
  inherited: `grep -c 'RequireF32(' src/vt/cpu/cpu_ltx2_vae.cpp` = 11,
  `src/vt/cuda/cuda_ltx2_vae.cu` = 11, `VaeKernels(` = 14, `DType::kF32` in the
  decoder = 22, `Ltx2VideoDecodeStreaming(` = 1.
* **The CUDA arm is NOT written, and §5.6's reason survived contact.** CI DOES
  compile that translation unit (`cuda-fat-build`, which falsifies the file's own
  "NEVER COMPILED" claim and is repaired here), so the reason for not writing it
  is reachability alone: `cuda_conv3d.cu` refuses bf16 storage (#1007) and every
  convolution goes through `vt::Conv3d`, so eleven bf16 branches there would be
  unreachable code.
* **Two probes in the inherited draft were WRONG.** Both read a module's
  parameters AFTER `.to(bfloat16)` had narrowed them in place, so their two
  hypotheses were the same tensor: one reported `nn.GroupNorm` as non-separating
  (it separates on 6664 of 24576) and the other reported `un_normalize`'s stat
  width from a comparison that did not isolate it.
* **The draft's open `F.normalize` stop condition is retired by measurement**
  (§4.8): the denominator is rounded to bf16 before the divide, 0 of 4800 at two
  scales against three separating alternatives.
* **§4.16 is a rule this row did not know it needed**, and it was the one real
  defect: the noise blend's Python-float scalars were being narrowed. The SHALLOW
  arm (§ below) is what caught it, and only because the generator emits the
  rejected answers as TENSORS — the port was equidistant from upstream and from
  two of the three alternatives, which is what pointed at a fourth rule.

### The red and the green, literally

Red, on the f32 tree, through the production entry point:

```text
tests/vllm/multimodal/test_ltx2_video.cpp:7059: ERROR: CHECK( fox.trace.vae_decode_not_bf16 == 0 ) is NOT correct!
  values: CHECK( 27645 == 0 )
  logged: VAE decode output, wider than bf16: 27645 of 27648
[doctest] assertions: 17 | 16 passed | 1 failed |
```

Green, after:

```text
tests/vllm/multimodal/test_ltx2_video.cpp:7059: SUCCESS: CHECK( fox.trace.vae_decode_not_bf16 == 0 ) is correct!
  values: CHECK( 0 == 0 )
  logged: latent into the decode, wider than bf16: 8 of 8
          VAE decode output, wider than bf16: 0 of 27648
```

### How the values are gated, and what is NOT

Three tiers, because one was not enough and saying which is which is the point.

1. **Five per-kernel rules, BIT-EXACT**, each against upstream and against the
   answer it rejects, through the `kLtx2Vae` CPU table: GroupNorm's narrowed
   affine (rejects f32 affine, separating 33/144), `ada_ln`'s three roundings
   (rejects one, 19/64), `spatial_noise`'s product-then-add (rejects fused,
   7/144), `linear_cn`'s bias-seeded accumulator (rejects bias-after, 9/32) and
   `PixelNorm` including its epsilon's WIDTH at three row scales.
2. **A SHALLOW two-convolution arm, BIT-EXACT** (max|diff| == 0), which holds the
   three rules the kernel table does not own: `_RMSNorm2D`'s multiply order
   (rejected answer 0.00390625 away), `un_normalize`'s narrowing (0.0078125) and
   the bf16 timestep embedding (0.00390625). It sits at 8 channels because
   `sqrt(8)` is not representable in bf16 and at C=64 every ordering agrees on
   4800 of 4800.
3. **The deep thirteen-convolution arm, BOUNDED**, at the chain's own measured
   one-ulp response (0.0117188 — how far the decode moves when ONE `conv_in`
   weight changes by ONE bf16 ulp). The port sits at 0.00585938. The bound is
   proven to separate rather than assumed to: the two rejected rules that reach
   this fixture's output are 0.0351562 and 0.015625.

**NOT bit-exact, and why.** torch BLOCKS its convolution reduction and this port
does not; at the fixture's own shapes the two orders disagree on 3 to 5 outputs
of 8192 to 24576. That is the whole of the deep arm's residue.

**A defect that reaches NOTHING at whole-decode level.** The f32 GroupNorm affine
moves the deep arm by exactly 0.0 — no bound of any width could see it — which is
what makes tier 1 load-bearing rather than duplicative.

### Mutations, with their literal results

| mutation | result |
|---|---|
| delete the production `kBF16` argument (`ltx2_video.cpp`) | `vae_decode_not_bf16` 0 -> **27645 of 27648**; **2** of 4972 engine assertions red (the second is the READER-ANCHORS case, `tests/vllm/multimodal/test_ltx2_video.cpp:1473`, because collapsing two lines to one moves the anchors), every digest, absmax, frame byte and determinism check GREEN; `test_ltx2_vae` fully green, which is the unit suite measuring the class rather than the capability |
| GroupNorm: round the normed value before the affine | reds "each kLtx2Vae kernel's BF16 rule" **and the deep arm** |
| `ada_ln`: one rounding instead of three | reds "each kLtx2Vae kernel's BF16 rule" |
| `spatial_noise`: fuse the product and the add | reds "each kLtx2Vae kernel's BF16 rule" |
| `linear_cn`: add the bias after the store rounding | reds "each kLtx2Vae kernel's BF16 rule" |
| `PixelNorm`: f32 epsilon instead of `bf16(1e-8)` | reds the **2^-14 arm ONLY**, at 0.00390625; 2^-0 and 2^-10 stay GREEN |
| `PixelNorm`: epsilon set to 0 | reds the **2^-10 arm** at 0.0625 and 2^-14 at 1.88281; 2^-0 stays GREEN |
| `_RMSNorm2D`: restore the old multiply order | reds the deep AND shallow bf16 arms |
| `un_normalize`: fuse the multiply and the add | reds the deep AND shallow bf16 arms |
| timestep embedding: drop the intermediate rounding | reds the deep AND shallow bf16 arms |
| noise blend: narrow the Python-float scalars | reds **both** arms; the deep arm's `max\|diff\|` is 0.0273438 against a bound of 0.0117188 |

The last four red NOTHING before the shallow arm existed. That is recorded because
it is the reason the shallow arm was built rather than a nicety.

Three more, added by the fresh review's repairs and measured the same way:

| mutation | result |
|---|---|
| `scaled_timestep`: restore the pre-row f64 product | before the repair, GREEN -- 52 cases and 3480 assertions of `test_ltx2_vae` passed. After it, reds the section 5i arm: `max\|diff\|` 0 -> **0.0117188**, landing exactly on the rejected wide-product tensor (its own distance goes 0.0117188 -> 0) |
| tiled decode: force `buffer.Allocate` to `kF32` | before the repair, GREEN -- 10 cases and 915 assertions of `test_ltx2_tiling`, and 114 cases and 4972 assertions of `test_ltx2_video`. After it, `test_ltx2_tiling` is 10/11 with 5 of 971 assertions red: `max\|diff\|` 0 -> **0.00507808**, landing exactly on upstream's own f32-buffer tensor. `test_ltx2_video` stays green, which is the engine suite measuring the render rather than the blend |
| tiled decode: fuse the three separable blend masks (the defect as found) | reds the section 8 arm at **0.001953125**, one bf16 ulp; every f32 tiling case stays green, which is why it survived to the bf16 arm (#2815) |

One more, added by the scoped fresh RE-review, for the refusal this spec had
recorded as ungateable:

| mutation | result |
|---|---|
| delete the bf16-on-a-non-CPU-queue refusal (`ltx2_video_vae.cpp:1473-1481`) | the mutation COMPILES (`BUILD_rc=0`, `libvllm.a` relinked) and reds the new subcase: `test_ltx2_vae -tc='*dtype refusals*'` goes 1/1 cases and 3/3 assertions GREEN to **1 of 1 case failed, 1 of 3 assertions red**, `CHECK_THROWS_WITH_AS ... threw a DIFFERENT exception!` -- the decode runs on past the deleted guard and dies downstream on a missing bf16 parameter. Restored byte-for-byte |
| the SAME subcase with the fake-accelerator registration removed (control) | throws `":177"` "for which no platform is registered", not `":1473"` "only the CPU arm serves it" -- which is the whole content of the retired "cannot be gated on this build" claim, and it holds only in that control |
| delete the ENTRY-POINT `RequireVaeDType` call alone (`ltx2_video_vae.cpp:1465`) | COMPILES and stays **GREEN**, 1 of 1 case and 3 of 3 assertions. `VaeStore::Alloc` (`:250`) refuses the same bag with the same message a few lines later, so the f16 subcase cannot see that call site's deletion. Recorded rather than repaired: forking the text would give one refusal three messages, which is what the single function exists to prevent, and the case now states that it gates the PREDICATE |

### One instrument lost its window and is repaired rather than deleted

`ltx2 a2vid: the distilled adapter rides stage 2 ALONE` measured the adapter's
reach in PPM bytes. On an f32 decode the adapter moved **19 of 146753** — 0.013%,
a few pixels straddling an 8-bit boundary. bfloat16's mantissa is 8 bits too, so
it now moves **0**, and raising the fixture's delta does NOT recover it: at scale
8 the count is 0 as well, because a larger delta pushes both arms into the
writer's clamp. The claim is made one step upstream instead, on a new
`vae_latent_digest`, and the comment says plainly that this is a weaker statement
than "reaches the pixels".

### Records this row's own work falsified, repaired in the same change

* `ltx2_video_vae_encoder.h:57` said the encoder's arm "is owed with the
  decoder's". The decoder's landed and the encoder's did not.
* `ltx2_tiling.h` recorded the buffer and the masks and NOT `zeros_like` — the
  sentence that decides the blend quotient's width.
* `cuda_ltx2_vae.cu` said it was "NEVER COMPILED, NEVER EXECUTED, ANYWHERE IN THIS
  PROJECT'S REACH". The `cuda-fat-build` CI job compiles it on every pull request;
  never EXECUTED still holds.
* `docs/FEATURES.md`'s Conv VAE arithmetic-width row said "STORAGE stays f32;
  bf16 owed".
* **This spec's own `## Owed` said the bf16-on-a-non-CPU-queue refusal "cannot be
  gated on this build", and that was FALSE.** The shadowing by
  `RequirePooledDevice` (`ltx2_video_vae.cpp:1441`, throwing at `:177`) is real
  ONLY in the control condition, when no platform is registered for the device
  type. `vllm::platforms::RegisterPlatform` and `vt::RegisterBackend` are public
  APIs, and `grep -rln 'platforms::RegisterPlatform' tests/` names THIRTEEN other
  test files that already call them on this CPU-only build
  (`tests/vllm/multimodal/test_ltx2_video_device_forward.cpp:190-192` is the one
  the new subcase is shaped after). Doing so here reaches the refusal with no GPU
  and no lease.
  Both arms measured on the same binary: with the fake `kXPU` backend and platform
  registered the decode throws "only the CPU arm serves it" (`:1473`); with the
  registration removed and nothing else changed it throws "for which no platform
  is registered" (`:177`). All three of this row's dtype refusals are now gated in
  "ltx2 vae: the dtype refusals this arm adds are REACHED, not merely written".
* **#2816's disagreement set was a spot check presented as a measurement.** It
  named n = 4, 7 and 13. Swept over n = 2..32 against the port's own `Linspace`
  (`ltx2_tiling.cpp:26-40`), `torch.linspace` disagrees at n = 4, 7, 8, 12, 13,
  14, 15, 16 and 20 through 32 -- 21 of 31 -- and agrees at 2, 3, 5, 6, 9, 10, 11,
  17, 18 and 19; `linspace(1, 0, n)` disagrees on the same set. The polarity is
  unchanged and was re-derived: the port's double-narrowed value is the
  correctly-rounded f32 at every n in the sweep, and the n = 4 pair #2816 quotes
  reproduces exactly. The short list was dangerous in one specific way -- an
  author avoiding "4, 7 and 13" would pick a ramp of 6, whose linspace has 8
  steps, and hit it. The issue now carries the swept set.

### Defaults, and why they have their values

* **`Ltx2LoadVaeWeights`'s `compute_dtype` defaults to `kF32`**, unlike wave 2's
  connector loader, which has no default at all. The connector has ONE caller;
  this has seven, six of which are still f32 ports, and a `kBF16` default would
  hand each of them a bag whose `Get` refuses.
* **`Ltx2VideoFrames::data` stays `std::vector<float>`.** It is the public pixel
  return, three channels wide; the intermediates this row narrowed are the base
  channel width at full resolution. The counter gates that its values are
  bf16-representable, which is the property that matters.
* **`ChunkBuffer` keeps f32 storage and rounds at each accumulation.** Same
  proportion argument: a pixel volume against base-channel-width intermediates.
* **bf16 is refused on a non-CPU queue in ONE place**, asking the same question
  the kernels refuse on, so the refusal and the route predicate cannot drift.
