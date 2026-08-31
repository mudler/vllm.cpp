# SPEC — `LTX25-AUDIO-DECODE-COST`: the 47 s of mel is 31 GMAC through a serial scalar loop

Issue: [#2405](https://github.com/mudler/vllm.cpp/issues/2405).
Owner row: `LTX25-AUDIO-DECODE-COST`.
Parent reading: [`ltx25-render-speed-parity.md`](ltx25-render-speed-parity.md),
which measured the leaf and recorded it under `## Owed` with no hypothesis.

## Scope

IN scope:

- Attribute `decode.audio.mel` by execution: what the 47.171 s is spent on, at
  the shipped checkpoint's own geometry, with the method stated.
- The upstream comparison at the `ltx-2` pin, cited `file:line`.
- One **bit-exact** repair, if one exists.

OUT of scope, declared rather than approximated:

- **`conditioning.connector` and `generate.guiders`.** They are 60.3% of the same
  render and they belong to the parent row's `## Owed`. This row does not touch
  them.
- **The `double` accumulator.** Upstream's `nn.Conv2d` accumulates in the tensor
  dtype, and [#1008](https://github.com/mudler/vllm.cpp/issues/1008) measured and
  moved exactly that for the VIDEO half. Moving this one changes every audio
  golden in the tree, so it is a correctness row with its own red-first evidence
  and it is not this one. Named under `## Owed`.
- **The device arm.** `vt::Conv2d` already exists (`include/vt/ops.h:4236-4258`)
  and is where this convolution should eventually live, exactly as
  [#1007](https://github.com/mudler/vllm.cpp/issues/1007) moved the video half to
  `vt::Conv3d`. That op's contract is a single **f32** accumulator, so routing
  through it is the same golden-changing change as the paragraph above and lands
  with it, not before it.
- **`decode.audio.vocoder`.** 3.823 s of the 50.745 s leaf, 0.74% of the render.
  It is already parallel through `host_parallel::ForOutputRows`
  (`src/vllm/model_executor/models/vocoder1d.cpp:320`), which is why it is small.

## What the leaf actually is

`decode.audio.mel` brackets exactly one call
(`src/vllm/multimodal/ltx2_video.cpp:5548-5551`):

```cpp
phase::Scope mel_phase("decode.audio.mel");
const Ltx2AudioSpectrogram mel = Ltx2AudioDecoderForward(
    im.audio_cfg, im.audio_weights, audio_latent_volume, audio_lc, audio_lf, audio_lm);
```

`im.audio_cfg` is parsed from the checkpoint
(`src/vllm/model_executor/models/ltx2_loader.cpp:1704-1735`). The values were
read from the shipped artifact's own safetensors metadata rather than from the
struct defaults, because three of them differ from the defaults and two of those
change the network's shape:

| key | checkpoint | our default |
|---|---:|---:|
| `ch` | 128 | 128 |
| `ch_mult` | `[1, 2, 4]` | `[1, 2, 4]` |
| `num_res_blocks` | 2 | 2 |
| `mel_bins` (`preprocessing.mel.n_mel_channels`) | **64** | 0 |
| `attn_resolutions` | **`[]`** | `[8, 16, 32]` |
| `mid_block_add_attention` | **false** | true |

Source: the `__metadata__.config` JSON inside
`ltx-2.5-audio-vae-bf16.safetensors`, sha256
`c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5`, which is the
digest `tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json` pins.

**So the shipped audio decoder carries no attention at all**, and every
statement below is about convolution only. A reader who took the header's
defaults would have looked for the cost in `AttnBlock`, which never runs.

The latent at the oracle's request is `[8, 26, 16]`: `kAudioLatentChannels = 8`
and `kAudioLatentMelBins = 16` (`ltx2_video.cpp:2952-2953`), and the frame count
is `AudioLatentShape.from_duration` mirrored at `:3407-3428` —
`round(25 / 24.0 * 16000 / 160 / 4) = 26`. The decoder returns `[2, 101, 64]`,
which is `26 * 4 - 3` mel frames, and at hop 160 into 16 kHz that is the 1.02 s
of audio the parent row quotes.

## The measurement

**Method, stated because the number depends on it.** A standalone harness calls
`Ltx2AudioDecoderForward` at the geometry above with synthetic weights, built
from this tree's own `ltx2_audio_vae.cpp` at `-O2 -ffp-contract=off` — the
project's own contraction pin (`CMakeLists.txt:55`) — with the vocoder and the
GroupNorm arm replaced by stubs that **abort** rather than compute, because the
checkpoint says `norm_type = pixel` and the vocoder is a different phase. A
second build of the same harness carries a per-call timer and MAC counter inside
`Conv2d`. Weight VALUES do not enter the cost: every tap is multiplied and added
whatever it holds, and the trip counts are set by the config and the latent.

This is a CPU measurement and it needed no GPU and no lease.

**The result, on the local devbox (AMD Ryzen 9 9950X3D, 20 visible cores).**

| | seconds | share |
|---|---:|---:|
| `Ltx2AudioDecoderForward` | 24.288 | 100.0% |
| — inside `Conv2d` | **24.216** | **99.7%** |
| — `PixelNorm` | 0.013 | 0.05% |
| — `SiLU` | 0.040 | 0.16% |

28 convolution calls, **31.46 GMAC** (62.9 GFLOP), sustained **1.30 GMAC/s**.
The eleven distinct shapes and their individual rates are in `## Outcome`; none
of them is an outlier, and the two 1x1 `nin_shortcut` convolutions are the
slowest per MAC rather than the fastest.

**1.2 to 1.3 GMAC/s is the number that identifies the defect.** It is one
multiply-accumulate per ~4 cycles at this box's clock, which is the latency of a
dependent scalar FMA chain and nothing else: no SIMD, no instruction-level
parallelism, one thread. The chain is `ci * kh * kw` long — 4608 dependent
`double` FMAs for a 512-channel 3x3 tap — and `-ffp-contract=off` plus a serial
`double` accumulator is exactly the shape a compiler cannot vectorise or
reassociate.

**The GB10 reading is the same defect at that box's clock.** 31.46 GMAC in
47.171 s is 0.667 GMAC/s, the same one-MAC-per-FMA-latency figure on a slower
core. Nothing about the leaf is I/O, allocation or page cache: 99.7% of it is
arithmetic issued at the rate a serial dependency allows.

## Why this is not what upstream does

Lightricks/LTX-2 @ `fd4ded7f`
(`.agents/oracles/ltx-2.md`; the clone this was read in reports that revision at
a clean worktree):

- `packages/ltx-core/src/ltx_core/model/audio_vae/causal_conv_2d.py:61-64` —
  `CausalConv2d.forward` is `F.pad(x, self.padding)` then `self.conv(x)`, where
  `self.conv` is `torch.nn.Conv2d` (`:50-59`).
- `packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py:385-400` —
  `AudioDecoder.forward`, the module this file ports.
- `packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py:117-123` builds
  the decoder with `dtype=self.dtype, device=self.device`, and
- `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1188-1198` is where
  that lands: `self._decoder_builder.build(device=self._device, dtype=self._dtype)`
  inside `gpu_model(...)`, called at `ti2vid_one_stage.py:244`.

So upstream runs these 31.46 GMAC as cuDNN `nn.Conv2d` in the pipeline dtype on
the accelerator. **The oracle's whole process, model load included, is 93.8 s**
(`ltx2_oracle_manifest.json`, `result.render_seconds`), so its audio decode is
bounded far below our 47 s and the cost is not inherent to the algorithm at this
configuration.

**vLLM is not the reference here and the reason is recorded rather than
assumed.** `.agents/oracles/ltx-2.md` establishes that vLLM proper registers
nothing LTX at the parity pin and that vLLM-Omni's `_PIPELINE_RECIPES` stop at
generation 2.3, so `ltx-2` is the admitted secondary oracle for this path under
`AGENTS.md` `## When vLLM has no implementation`.

## The repair, and why it is bit-exact

`Conv2d`'s output loop is partitioned over output **lines** `(oc, y)` through
`host_parallel::ForOutputRows`
(`src/vllm/model_executor/models/host_parallel.h`), the two-line adapter over the
one `vt::cpu` threadpool that `vocoder1d.cpp` and `minimax_music3_ar.cpp` already
use. The whole `ci * kh * kw` reduction stays inside one output element's body,
so every element is produced by the same instruction sequence over the same
values in the same order at any worker count and under any work-stealing
assignment. That is the determinism contract `cpu_threadpool.h:39-43` states for
the CPU backend, inherited rather than restated.

Splitting the reduction axis `ic` instead would make the summation order a
function of the thread count. It is rejected at the site, for the same reason
`ltx2_video_vae.cpp` rejected it under
[#1009](https://github.com/mudler/vllm.cpp/issues/1009).

The **audio encoder shares this primitive** (`Conv2d` is file-private and
`ltx2_audio_vae.cpp:941` is the encoder's call site), so it inherits the change.
That is the same inheritance #1009 recorded for the video encoder.

Nothing else in the leaf is touched. `PixelNorm` and `SiLU` are 0.2% of it
together and parallelising them would be work with no measurable term.

## Tests to port

There is no upstream test for this: upstream calls `nn.Conv2d` and torch's own
suites cover it. The two cases below are ported from the video half's pair for
#1009 in the same file, which is the closest thing to a reference this change
has, and each states its own red-before condition:

| ID | Case | Red before |
|---|---|---|
| A1 | `the AUDIO decode DISPATCHES its convolutions to the CPU threadpool` | yes — the pool's chunk cursor reads 0 on a serial decode |
| A2 | `the AUDIO decode is BIT-IDENTICAL across thread counts` | no, and deliberately so |

**A2 is green before the change and that is stated rather than hidden.** Two runs
of a serial decode are also bit-identical, so a thread-count A/B cannot fail on
an implementation that never threads anything. It is the guarantee half: A1
proves the dispatch happened, A2 proves the dispatch did not change the answer,
and neither is sufficient alone. This is the same division of labour the video
pair carries, and the video pair's own comment says so.

The instrument for A1 is `Threadpool::ChunkAdd(0)`, a non-mutating read of the
pool's shared work-stealing cursor: a fresh pool reads 0, a pool that has run a
partitioned dispatch reads more. The `REQUIRE` before the decode is its own
positive control.

## Gates

1. `cmake --build build --target test_ltx2_vae -j 4 -- -k 0`, then the built
   binary. Every existing audio and video golden in that file must stay green:
   they are the regression gate on the move.
2. A byte comparison of the decoder's output against the pre-change binary, and
   across worker counts, on the shipped geometry.
3. A same-binary A/B on `VLLM_CPP_CPU_THREADS`, n >= 3 per count, interleaved,
   with the box's load stated per leg.
4. `scripts/agent-preflight.sh`.

Gate 3 is a **same-binary** A/B because `.agents/benchmarking.md` `## Two arms
have to BE two arms` refuses a hash-only verdict and no artifact hash in this
tree is falsifiable. `VLLM_CPP_CPU_THREADS=1` short-circuits `ParallelForRows` to
`body(0, nr)` on the caller (`cpu_threadpool.cpp:423-426`), so the one-worker arm
IS the pre-change serial path byte for byte.

No GPU and no lease. The whole leaf is host arithmetic.

## Work breakdown

- **W1** — this spec, committed before the code.
- **W2** — the repair, the two cases, and the measurement.

## Risks/decisions

- **The measurement box is not idle and the ratio is therefore a LOWER bound.**
  Contention reweights rather than merely adding noise, and it inflates the
  multi-threaded arm relative to the single-threaded one — which is exactly the
  direction that would flatter this change if it went the other way. It does not:
  a loaded box under-states a parallel speedup. The load is recorded per leg and
  the figure is quoted as a floor.
- **`ForOutputRows` has a size guard** (`kMinParallelWork = 1 << 16`) and below it
  the body runs inline on the caller. That is a scheduling decision and never a
  numeric one — the same body over the same range — but it means a small enough
  convolution stays serial by design. At the shipped geometry the smallest
  convolution is `conv_out` at 2 output channels; it is 0.01 s of the 24 s.
- **No public benchmark ID.** One geometry, one checkpoint, a CPU leaf. Adding an
  ID on it would be the `a-number-quoted-often-becomes-treated-as-measured`
  failure.

## Evidence

- The per-call `Conv2d` timing and MAC tally at the shipped geometry.
- The eleven convolution shapes, their MACs and their individual rates.
- `memcmp` of the decoder output: pre-change binary against post-change, and
  worker counts 1 / 4 / 16 / 20 against each other.
- The same-binary thread sweep with per-leg load.
- `test_ltx2_vae` case and assertion counts before and after.

## Stop conditions

- A byte difference at any worker count. The whole claim is that thread count
  cannot change the result; a single differing float ends the change.
- A golden that moves. This repair may not reprice one.

## Owed

- **The `double` accumulator.** `nn.Conv2d` accumulates in the tensor dtype;
  this loop accumulates in `double`, which no reference on this path uses. #1008
  measured and fixed precisely this for the video half. It is a golden-changing
  correctness row and it is the larger remaining lever here, because it is what
  unlocks SIMD as well. Owner: this row, through #2405.
- **The device arm through `vt::Conv2d`.** The op exists with CPU and device
  arms and a published f32 accumulation contract (`include/vt/ops.h:4236-4258`),
  and upstream runs this decoder on the accelerator. It lands with the item
  above, because that op's accumulator width is the change above. Owner: this
  row, through #2405.
- **The remaining 206.0 s of the parent row's render** is still a second
  question, and this row moves one term of it. Owner:
  `LTX25-RENDER-SPEED-PARITY`.

## Now

`DONE`, pending review. W1 is this file and W2 is the commit after it, in the
same pull request. The attribution is measured, the repair is byte-identical at
seven worker counts, and the end-to-end GB10 leaf after the change is NOT
measured and is said so in `## Outcome`.

## Outcome

**Where the leaf goes, at the shipped geometry.** 28 convolutions, 31.46 GMAC
(62.9 GFLOP), and the eleven distinct shapes, from the instrumented build:

| shape | seconds | GMAC | GMAC/s |
|---|---:|---:|---:|
| `ci=512 co=512 out=26x16 k=3` | 7.420 | 9.815 | 1.323 |
| `ci=128 co=128 out=101x64 k=3` | 4.103 | 4.766 | 1.162 |
| `ci=256 co=256 out=51x32 k=3` | 3.120 | 4.813 | 1.543 |
| `ci=512 co=512 out=52x32 k=3` | 3.065 | 3.926 | 1.281 |
| `ci=256 co=256 out=102x64 k=3` | 2.993 | 3.850 | 1.287 |
| `ci=256 co=128 out=101x64 k=3` | 1.356 | 1.906 | 1.406 |
| `ci=512 co=256 out=51x32 k=3` | 1.265 | 1.925 | 1.522 |
| `ci=256 co=128 out=101x64 k=1` | 0.537 | 0.212 | 0.394 |
| `ci=512 co=256 out=51x32 k=1` | 0.339 | 0.214 | 0.631 |
| `ci=128 co=2 out=101x64 k=3` | 0.010 | 0.015 | 1.509 |
| `ci=8 co=512 out=26x16 k=3` | 0.009 | 0.015 | 1.621 |
| **`Conv2d` total** | **24.216** | **31.46** | **1.299** |
| `PixelNorm` | 0.013 | | |
| `SiLU` | 0.040 | | |
| **`Ltx2AudioDecoderForward`** | **24.288** | | |

**No shape is an outlier and that is the point.** The rate band is 1.16 to 1.62
GMAC/s across every 3x3 convolution regardless of channel count or output size,
which is what a latency-bound dependent accumulator looks like and what a
bandwidth- or cache-bound loop does not. The two 1x1 `nin_shortcut` convolutions
are the SLOWEST per MAC, at 0.39 and 0.63, because their reduction chain is 9x
shorter per output element and the per-element overhead is a larger share -- the
opposite of what a work-volume explanation predicts.

**The per-element dtype dispatch is NOT the defect here, checked rather than
assumed.** `LoadF32`, `StoreF32` and `vt::SizeOf` appear **zero** times in
`ltx2_audio_vae.cpp`: this loop indexes `std::vector<float>` directly. The 219
call sites `VT-CPU-ELEM-DISPATCH` names are a real and separate lever and none of
them is on this path.

**Byte identity, measured at the shipped geometry.** The decoder output is
byte-identical to the pre-change binary's at worker counts **1, 2, 3, 5, 8, 16
and 20** — seven counts, one `cmp`, no tolerance. Inside the suite, the two new
cases are green and every pre-existing golden is unmoved.

**Red-first, by mutation on the built tree.** Reverting only `Conv2d` to its
serial form and rebuilding that one translation unit takes `test_ltx2_vae` from
**47 cases / 3189 assertions green** to **46/47 with 3188/3189**, the single
failure being the new dispatch case reading a chunk cursor of **0** against
**16**. Every golden is green in both arms, so the change repriced nothing. The
tree was restored and rebuilt to 47/47 afterwards.

**Speed, same-binary A/B on `VLLM_CPP_CPU_THREADS`, interleaved.** Worker count 1
short-circuits `ParallelForRows` to `body(0, nr)` on the caller
(`cpu_threadpool.cpp:423-426`), so that arm IS the pre-change serial path byte
for byte and no second artifact or hash is involved.

| sweep | box loadavg (1 min) | n | 1 worker, median | 20 workers, median | ratio |
|---|---|---:|---:|---:|---:|
| A | 40 - 61 | 3 | 41.30 s | 5.46 s | **7.57x** |
| B | 62 - 90 | 3 | 68.47 s | 11.32 s | **6.05x** |
| C | 67 - 94 | 5 | 56.99 s | 10.80 s | **5.28x** |

Sweep A also read 4 workers at 11.25 s and 16 at 5.85 s.

**The ratio is a FLOOR, and the data says which way the bias runs.** The measured
ratio falls monotonically as the foreign load rises across A, B and C. This is a
shared devbox that carried up to twenty-five foreign compiler processes and
several concurrent preflights from other sessions throughout, and contention
inflates the multi-threaded arm relative to the single-threaded one. So a loaded
box UNDER-states a parallel speedup: 7.57x is the best available lower bound and
an idle-box figure would be higher. The video half's identical change measured
~9x at 16 to 20 workers under #1009, also on a box that was not idle.

**No idle-box reading was obtainable and that is recorded rather than worked
around.** The 1-worker arm's spread reaches 121% in sweep C, which is not a
quotable figure under `.agents/benchmarking.md`; what carries the claim is the
mechanism, the byte identity and the monotone direction, not the decimals.

**Every shipped-geometry convolution crosses the dispatch guard**, so none of
them stays serial by accident: the smallest is `conv_out` at 202 output rows and
73,728 inner-loop trips per row, against `kMinParallelWork` of 65,536.

**What this row does NOT claim.** The end-to-end GB10 render leaf after the
change. `scripts/ltx25-render-speed-repeat.sh` asserts the binary's sha256
against `4b0666ee`'s (its H1), which by design refuses any rebuild, so measuring
the repaired render needs a lease and a harness change and is a separate piece of
work. `decode.audio` was 50.745 s of which mel is 47.171 s and the vocoder
3.823 s; what the leaf reads after this change on that box is **unmeasured**.
