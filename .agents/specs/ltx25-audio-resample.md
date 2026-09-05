# LTX-2.5 — arbitrary-ratio audio resampling (A19)

Row: `LTX25-AUDIO-RESAMPLE`. Campaign: [`ltx-2-5.md`](ltx-2-5.md)
(operator-owned; **not edited by this row**). Issue:
[#2583](https://github.com/mudler/vllm.cpp/issues/2583). Plan:
[`ltx25-completion-scope.md`](ltx25-completion-scope.md) §8 order **7**, gap
**A19**, itself issue
[#2526](https://github.com/mudler/vllm.cpp/issues/2526).

Upstream pins:

| Reference | Registry id | Revision |
|---|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`) | `ltx-2` | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |
| torchaudio, the module `ops.py:40` calls | — | `2.11.0+cu130`, the wheel the `ltx-2` pin resolves and the goldens were generated with |

Read from a local checkout at that revision with `git show <sha>:<path>`, never
from a working tree, because a working tree can differ from the pin.

---

## 0. Honesty statement — what this row does and does not claim

This row makes an audio file at **any** sample rate drive an LTX-2.5 render. It
does not claim A3 (`DubItPipeline`), A18 (reference-audio conditioning delivery),
or any IC-LoRA. It unblocks the first two by removing the refusal that stood in
front of both; it does not implement either, and both stay owed by their own
rows.

**Upstream ships no tests at this pin.** `find . -iname '*test*' -not -path
'./.git/*'` over `Lightricks/LTX-2 @ fd4ded7f` returns empty. So "port the
upstream tests in the same change" has nothing to port from the model author, and
the obligation becomes what §6 does instead: **execute** the upstream module at
the pin and emit its outputs as goldens, plus pin each upstream default against a
`file:line`.

**No render on real weights is claimed.** The row is gated on reduced-dimension
fixtures and executed-upstream goldens, exactly as `LTX25-A2V-AUDIO-INPUT` was.

---

## 1. Scope

In:

* Port `torchaudio.functional.resample` at the defaults `ops.py:40` passes, as an
  arbitrary rational-ratio polyphase sinc-hann resampler.
* Call it from `Ltx2WaveformToLogMel`, which is where upstream calls it.
* Drop the two refusals that stand in front of it, and repair the **nine**
  statements across this tree that misname upstream's filter as kaiser.

Out:

* `sinc_interp_kaiser` and the `beta` knob. Upstream passes neither, and porting
  an unreached branch is the shape `## Nothing lands dead` forbids.
* Non-default `lowpass_filter_width`, `rolloff`. Same reason.
* Compressed audio containers, channel mixing, A3, A18. Each is refused or owed
  elsewhere and none of them is this gap.

---

## 2. Our baseline — the gap, on this tree

`Ltx2WaveformToLogMel` refuses any waveform whose rate is not
`config.target_sample_rate`
(`src/vllm/model_executor/models/ltx2_audio_vae.cpp:1051-1060` at this row's base
`6643b2bbf`; the refusal is gone at head, so the anchor names the tree it
describes), and `Ltx2DecodeAudioWav` refuses the same thing one hop earlier so
the message can name the file
(`src/vllm/model_executor/models/ltx2_audio_input.cpp:96-105`, same base).
Between them, **every** audio input at a rate other than the checkpoint's is
turned away, so `a2vid_two_stage` only accepts a take the user resampled with
some other tool first.

**The refusal misnames the filter, in nine places.** Its message and eight
sibling comments call `torchaudio.functional.resample` "an arbitrary-ratio
polyphase **kaiser** resampler". It is not one. `resample`'s signature defaults
`resampling_method="sinc_interp_hann"` (torchaudio 2.11
`functional/functional.py:1441`), and `_get_sinc_resample_kernel` builds a kaiser
window only on the `else` of `if resampling_method == "sinc_interp_hann"`
(`:1384-1391`). `ops.py:40` passes neither `resampling_method` nor `beta`, so the
window is `cos(t*pi/lpw/2)**2` — a **hann** window, the same family this tree
already ports for the vocoder's BWE stage at `ltx2_audio_vae.cpp:440`
(`Ltx2HannSincResampleFilter1d`), which is exactly that kernel specialized to
`orig_freq == 1`.

So what is missing is the **arbitrary rational ratio**, not the window. Sizing
A19 at M was right; the reason written beside it was wrong, and this row corrects
the reason as well as the code.

The nine sites, read at this row's base `6643b2bbf` because the repair moves
every one of them: `src/vllm/model_executor/models/ltx2_audio_input.cpp:101`,
`src/vllm/model_executor/models/ltx2_audio_vae.cpp:1057`,
`include/vllm/model_executor/models/ltx2_audio_vae_encoder.h:173`,
`include/vllm/multimodal/ltx2_video.h:315`,
`include/vllm/model_executor/models/ltx2_audio_input.h:124`,
`tests/vllm/multimodal/test_ltx2_video.cpp:7518`,
`.agents/specs/ltx25-a2v-audio-input.md:155` and `:463`, and
**`examples/ltx2_gen/main.cpp:179`**.

That last entry is the row's own record defect and not a tenth discovery. This
paragraph said "nine" and listed **eight** until 2026-09-02; the repair covered
the eight it listed, and the ninth shipped unrepaired into a `--help` string. The
list is now enumerated by
`git grep -inE "polyphase kaiser|kaiser resampl" 6643b2bbf -- .`, which returns
exactly these nine, rather than transcribed.

---

## 3. Upstream chain — read end to end, not inferred

| Hop | Upstream | What it settles |
|---|---|---|
| 1 | `a2vid_two_stage.py:196` | `decode_audio_from_file` runs at the file's **native** rate |
| 2 | `decode.py:290-296` | the window (`start_time`, `max_duration`) is applied in samples at the **native** rate, before any resample |
| 3 | `decode.py:173-176` | the waveform is **float32**, `astype(np.float32)`, normalised to [-1, 1] |
| 4 | `a2vid_two_stage.py:200` -> `audio_vae.py:271` | `encode_audio` -> `audio_processor.waveform_to_mel(audio.to(device=device))`; `.to(device=...)` moves, it does not cast |
| 5 | `ops.py:44-49` | `waveform_to_mel` calls `self.resample_audio(audio)` **first**, then the mel transform |
| 6 | `ops.py:36-42` | `resample_audio` returns `audio` unchanged when the rates are equal; otherwise `torchaudio.functional.resample(waveform, orig, target)` and then `.to(device=..., dtype=waveform.dtype)` |
| 7 | `functional.py:1470-1476` | `orig_freq <= 0 or new_freq <= 0` raises; `orig_freq == new_freq` returns the input; `gcd` reduces the ratio |
| 8 | `functional.py:1305-1402` | the kernel: `base_freq = min(o, n) * rolloff`, `width = ceil(lpw * o / base_freq)`, one row per output phase |
| 9 | `functional.py:1405-1432` | zero-pad `(width, width + o)`, `conv1d` stride `o`, transpose-interleave, then **two** closing lines: `target_length = ceil(n * L / o)` with the quotient **narrowed to f32 before the ceil** (`:1427`; `torch.as_tensor` of a Python float takes the default dtype), which lands one sample either side of an exact integer ceil — fewer at 180697 samples for 44100 -> 16000, more at 33554438 for 44100 -> 22050 — and `resampled[..., :target_length]` (`:1428`), a Python slice, so it **clamps** to the `(L // o + 1) * n` columns the convolution produced |
| 10 | `a2vid_two_stage.py:301-303` | the returned soundtrack is the caller's **original** `Audio`, at its **native** rate. The resample never reaches the output |

Hop 10 is why nothing downstream of the encoder changes: the file's own rate
stays the render's `sample_rate`, exactly as today.

---

## 4. Port map

| Upstream | Ours |
|---|---|
| `functional.py:1305-1402` `_get_sinc_resample_kernel` | `Ltx2SincResampleKernel` (file-local), `ltx2_audio_vae.cpp` |
| `functional.py:1405-1432` `_apply_sinc_resample_kernel` | `Ltx2ResampleWaveform`, `ltx2_audio_vae.cpp` |
| `functional.py:1435-1490` `resample` | `Ltx2ResampleWaveform`'s guards and early return |
| `ops.py:36-42` `AudioProcessor.resample_audio` | the call at the head of `Ltx2WaveformToLogMel` |

`Ltx2ResampleWaveform` is declared in
`include/vllm/model_executor/models/ltx2_audio_vae_encoder.h` beside the mel
front-end it belongs to, because upstream keeps them in one module (`ops.py`).

### The dtype, and why it is `float` and not `double`

Upstream resamples in **float32**: hop 3 makes the waveform `np.float32`, hop 4
does not cast it, and `_get_sinc_resample_kernel` is called with
`dtype=waveform.dtype`, so **the filter itself is built in float32** — `idx`,
`t`, the clamp, the window, `sin(t)/t` and the scale, all of it
(`functional.py:1376-1397`).

This is not a detail to round up. Building the same filter in `double` and
storing it as `float` is *more accurate than upstream* and therefore **further
from it**: measured against the pinned oracle, a double-built kernel differs by
up to **3.39746e-06** where a float-built one differs by **1.19209e-07**, a
factor of 28 and 13.6 times the gate's own 2.5e-07 bound. AGENTS.md's *Inherit vLLM defaults* names exactly this failure — a dtype that
is too wide, which no token gate can see. The port mirrors the float32
arithmetic operation for operation, including the association `sinc * (window *
scale)` (`:1397`) and the fact that `scale = base_freq / orig_freq` is computed
in Python `float` (f64) and *then* narrowed (`:1395`).

The one deliberate exception is the convolution **accumulator**, which is
`double`. torch's `conv1d` reduction order over 161 taps is a vectorised
implementation detail that no C++ loop reproduces, so the choice is between two
different roundings; the exact one is chosen and annotated. It is an accumulator,
not a stored dtype: every stored value stays `float`.

### The one shape decision

`Ltx2WaveformToLogMel` takes `[channels, samples]` channel-major. torchaudio
packs the batch and resamples the last axis, so each channel resamples
independently against the same kernel. The port builds the kernel once and walks
the channels, which is that, not an approximation of it.

---

## 5. Reachability — the sentence the records must carry

The production entry point is `vllm_video_generate` -> `VideoEngine::Generate`
-> `Ltx2VideoEngine::Generate` -> `Ltx2DecodeAudioWav`
(`src/vllm/multimodal/ltx2_video.cpp:3063`) -> `Ltx2EncodeAudioToLatent`
(`:3068`) -> `Ltx2WaveformToLogMel`
(`src/vllm/model_executor/models/ltx2_audio_input.cpp:200`) -> the new
`Ltx2ResampleWaveform`. Every hop already exists and is already gated; this row
adds the last one and deletes the two refusals that stood in the chain.

The smallest failing test therefore enters through `LoadVideoEngine` +
`engine->Generate` with a WAV at a rate the fixture's audio VAE does not declare,
per the note at `test_ltx2_video.cpp:7229-7240`. A unit test over
`Ltx2ResampleWaveform` alone would prove the function works and never that a
request can arrive at it.

**Mutation:** delete the `Ltx2ResampleWaveform` call in `Ltx2WaveformToLogMel`
and the video-level case must red. Recorded in `## Outcome`.

---

## 6. Tests to port

Upstream ships none (§0). What this change does instead:

1. **Goldens by execution.** `scripts/gen-ltx2-vae-goldens.py` gains sections
   **8d**, **8e** and **8f**, importing `ltx_core.model.audio_vae.ops.AudioProcessor`
   from the `--ltx2` checkout and running `resample_audio` and `waveform_to_mel`.
   They ride the generator's existing `kLtx2VaeUpstreamRevision` anchor, which the
   C++ suite already asserts against a pinned SHA, so a regeneration against a
   different upstream fails the gate rather than replacing the oracle.
2. **The rate pairs are chosen to reach distinct arms**, not for coverage
   arithmetic. What section 8d ships is 16000 -> 48000 (pure upsample, `o = 1`,
   the degenerate ratio the tree already ports for the vocoder's BWE stage),
   48000 -> 16000 (pure downsample, `o = 3`, `n = 1`, `width = 19`, one phase
   row), 44100 -> 16000 (`gcd = 100`, `o = 441`, `n = 160`, `width = 17`, the
   widest kernel and the pair a real 44.1 kHz take hits), and 16000 -> 16000 (the
   equal-rate early return, which must be a **copy** and not a filtered pass).
   This paragraph named a different four before the port ran; the shipped set is
   the one above, and `RESAMPLE_CASES` in the generator is its record.
2b. **Section 8f reaches the truncation boundary, which 8d cannot.** The 8d arms
   top out at 218 output samples, and upstream's f32-narrowed ceil (hop 9) first
   disagrees with an exact integer ceil at 180697 input samples — 4.097 s at
   44.1 kHz, three orders of magnitude further out. 8f carries six LENGTH-plus-
   tail goldens: 180696 / 180697 / 180698 at 44100 -> 16000, which bracket that
   boundary; 90569 at 22050 -> 16000, where the same effect starts; 33554438 at
   44100 -> 22050, where the narrowing rounds the OTHER way and upstream keeps one
   sample more; and 100663303 at 48000 -> 16000, where it rounds up past the last
   column the convolution produced and upstream's SLICE decides the answer. Each
   arm declares `target_length - exact_ceil` and whether the slice clamps, and the
   generator asserts both against upstream's own numbers **and** asserts the
   returned `.shape[-1]` against `min(target_length, columns)`. An arm that
   stopped discriminating fails generation rather than gating nothing.
3. **A lower bound, not only a tolerance.** A resampler that returned zeros, or
   that returned the input untouched, satisfies a `max|diff| < tol` gate against
   the wrong golden and any shape check. The cases assert the output's absmax is
   positive **and** that a resampled take differs from the same samples read as
   if they were already at the target rate — the exact wrong answer the refusal
   was written to prevent.
4. **The end-to-end case**, at `test_ltx2_video.cpp`, replaces the SUBCASE that
   asserted the refusal: a 44.1 kHz WAV now renders, and its conditioning trace
   carries a non-zero audio latent whose digest **differs** from the same file
   written at the fixture's own rate.

---

## 7. Gates

```sh
cmake --build build -j 4 --target test_ltx2_vae test_ltx2_video
./build/tests/test_ltx2_vae -tc='*resampl*'
./build/tests/test_ltx2_video -tc='*audio*'
scripts/agent-preflight.sh --staged
```

Plus regeneration parity:

```sh
python3 scripts/gen-ltx2-vae-goldens.py --ltx2 ~/_git/LTX-2 \
    --out tests/vllm/models/ltx2_vae_goldens.inc
git diff --exit-code tests/vllm/models/ltx2_vae_goldens.inc
```

---

## 8. Dependencies

None. A19 depends on nothing in §8 of the plan; A3 and A18 depend on it.

---

## 9. Work breakdown

1. Spec (this file), committed first.
2. Generator sections 8d/8e and the regenerated `.inc`.
3. `test_ltx2_vae.cpp` cases, red before the port.
4. `Ltx2ResampleWaveform` plus the `Ltx2WaveformToLogMel` call.
5. Delete the two refusals; repair the nine kaiser statements.
6. `test_ltx2_video.cpp`: the refusal SUBCASE becomes the render case.
7. `docs/FEATURES.md` and `docs/USAGE.md` where they state the rate constraint.

---

## 10. Risks/decisions

| # | Risk | Handling |
|---|---|---|
| R1 | A `double` filter reads as "better" and silently widens the model path | §4 measures both against the oracle and pins float32 with the number. The gate tolerance is tight enough (2.5e-07) that a double-built kernel **fails** it |
| R2 | Removing a refusal admits an input that then fails deeper, with a worse message | The window and channel refusals stay; only the rate one goes. The `samples > n_fft/2` check in `Ltx2WaveformToLogMel` now sees the **resampled** length, which is upstream's order (hop 5) |
| R3 | A tolerance-only gate passes a resampler that returns the input | §6.3's lower bound and difference assertions |
| R4 | The nine kaiser statements are repaired in prose but the sizing document keeps the claim | `ltx25-completion-scope.md` is another row's spec and is not edited here; the correction is recorded on #2583 and in §2 |
| D1 | Should `Ltx2DecodeAudioWav` keep `want_sample_rate`? | No. Upstream's decoder has no target rate (hop 1); the parameter existed only to raise. Removed, and the one call site updated |

---

## Now

`DONE` pending review. Landed on `row/LTX25-AUDIO-RESAMPLE`, issue #2583.

---

## Outcome

### What was measured, and what it decided

**The dtype, which was the one decision this port could have got quietly wrong.**
Three filters were built and compared against the pinned oracle's own
`AudioProcessor.resample_audio` output at four rate pairs:

| Filter built in | max abs diff vs the oracle |
|---|---|
| `double` throughout `Ltx2SincResampleKernel`, narrowed to `float` at the store | **3.39746e-06** |
| torchaudio's own `dtype=None` kernel (`idx` in f64, `t` in the default f32) | 2.37e-06 |
| `float`, mirroring `dtype=waveform.dtype` | **1.19209e-07** |

All three re-measured on 2026-09-02, because the first two rows were first
recorded as 1.03e-05 and 4.71e-06 and neither reproduces. The mutation each row
names is exact: row 1 makes `base_freq`, `scale`, `lpw`, `kPi`, `phase`, `idx`,
`t`, `shaped`, `window` and `sinc` `double` and casts only at
`kernel[j * taps + k] = ...`; it reds the `Wide` arm at 3.39746e-06 and all four
8f tails between 1.55e-06 and 3.09e-06. Row 2 is torchaudio's own
`_get_sinc_resample_kernel(..., dtype=None)` over the same 8d inputs. Row 3 is
this port at the committed head, read by setting `kResampleTol` to `0.0`.

The `double` filter is *more accurate than upstream* and therefore twenty-eight
times further from it. It would have passed a gate written around it and read as
a careful implementation. §4 states the chain that fixes the answer at f32, and
the gate's 2.5e-07 is set so that a widening fails it.

**The tolerance is twice the measured floor, not a hedge.** Setting
`kResampleTol` to `0.0` reds three of the four 8d arms, at 5.96e-08 (Up),
5.96e-08 (Down) and 1.19209e-07 (Wide), and all four 8f tails, at 1.49e-08,
4.47e-08, 2.98e-08 and 7.45e-09. `Same` passes at zero, because it is a copy.
2.5e-07 is 2.10x the worst of those.

**`audio_latent_absmax` cannot gate this, and that is a finding rather than a
gap.** The obvious video-level claim — "the 44.1 kHz take must land nearer the
24 kHz one than the mis-read take does" — was written, run, and **failed**: the
three takes read 1.07194, 1.07293 and 1.07208, a 0.1% spread over three
genuinely different waveforms, because the trace's absmax is dominated by the
encoder's per-channel statistics. The assertion was removed rather than inverted
or loosened, and the reason is recorded where the assertion was.

### The truncation was NOT upstream's, and no arm here could see it

`_apply_sinc_resample_kernel` ends
`target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()`
(`functional.py:1427`). That reads as an integer ceil and this port first shipped
one, `(next * samples + orig - 1) / orig`. It is not one: `torch.as_tensor` of a
**Python float** takes `torch.get_default_dtype()`, which is float32, so the f64
quotient is rounded to f32 **before** the ceil. Where the exact quotient sits
above an integer by less than half an f32 ulp, the narrowing lands on that
integer and upstream keeps one sample FEWER.

Measured at 44100 -> 16000: 180696 samples give 65559 both ways, 180697 give
65559 upstream and 65560 from the exact ceil, 180698 give 65560 both ways. 48102
of the first 60 s worth of lengths diverge, and the density grows with duration.
22050 -> 16000 starts at 90569 and 44100 -> 24000 at 240854. At 48000 -> 16000
(`o = 3`) and 16000 -> 48000 (`o = 1`) it never happens at any audio length,
which is why three of the four 8d arms look clean at any length. One extra output
sample changes the last mel frames' STFT window content and, where
`samples % hop == 0`, the frame count — the audio latent length, and so the
conditioning shape.

**The gate could not see it, and that is the reason 8f exists.** The four 8d
arms top out at 218 output samples. Substituting torchaudio's real float32 ceil
into this port and rebuilding left `*resampl*` 26/26, `*waveform_to_mel*` 9/9 and
`*RESAMPLED*` 14/14 green: the right ratio at the wrong length. Section 8f adds
the four boundary arms of §6.2b, and with them the old formula reds on exactly
the two divergent lengths (65560 vs 65559 and 65720 vs 65719, plus their tails at
0.257 and 0.359) and stays green on the two that bracket them.

The repair mirrors the computation rather than describing it: the int64 product
`next * samples` is exact as a `double` for any length under 2^53 / next, so
`double(next * samples) / double(orig)` is Python's own correctly-rounded `int /
int`; `static_cast<float>` is `as_tensor`, `std::ceil` on the float is
`torch.ceil`, and the cast back to `int64_t` is `.long()`. Checked against
torchaudio over 276060 (ratio, length) pairs — twelve ratios by 23005 lengths,
covering 1 to 20000, the runs around 90569 and 180697, a thousand lengths either
side of 2^24 where f32 no longer holds every integer, and 2^25: zero
divergences.

### The formula was upstream's; the SLICE was not

`_apply_sinc_resample_kernel` ends on **two** lines, and the repair above ported
one of them:

```python
target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
resampled = resampled[..., :target_length]
```

`resampled` carries exactly `blocks * next` columns (`functional.py:1424-1426`),
and a Python slice **clamps**. Where `target_length` exceeds those columns
upstream returns the columns. This port allocated `target`, filled `blocks *
next`, and reported `out_samples = target`, so it emitted a trailing sample
upstream never computed — value-initialised zero, reported as real.

It fires when the f32 narrowing rounds **up** past the exact ceil, the opposite
direction from the one the first repair studied. The slack between the columns
and the exact ceil is `next - ceil(next * (samples % orig) / orig)`, whose
minimum over the residues is `next / orig` in integer division — **zero for
every downsampling ratio**. So any ratio with `next < orig` has lengths whose columns are exactly
the exact ceil, and an upward narrowing there overshoots. Measured: at
48000 -> 16000 and 100663303 samples (2097.2 s), `target_length` is 33554436 and
the convolution produced 33554435. `resample_audio` returns 33554435 on the 8f
input; this port returned 33554436, and its last eight samples were the golden's
window shifted by one and closed with a zero, `max|diff| = 0.321509` against a
`2.5e-07` tolerance. 32000 -> 16000 at 67108869 samples is the same arithmetic:
`target_length` 33554436, columns 33554435.

**On the production path, at a length no default render reaches.** The chain is
`ltx2_gen --audio-path <file>` -> `Ltx2VideoEngine::Generate` ->
`Ltx2DecodeAudioWav` -> `Ltx2EncodeAudioToLatent` -> `Ltx2WaveformToLogMel` ->
`Ltx2ResampleWaveform`, and the row's reachability mutation already proves that
call site is reached. What the mutation cannot say is which INPUTS reach the
clamp, and the answer is: long ones.

The clamp fires only where the output crosses 2^25 with zero column slack.
Scanned over the zero-slack residues of ten ratios, the smallest firing input is
**2097.2 s — 35.0 min** — at 48000 -> 16000, 32000 -> 16000, 96000 -> 16000 and
192000 -> 16000; 25.4 min at 44100 -> 22050; and 18.6 HOURS at 44100 -> 16000,
whose `o = 441` puts the gap 0.36 below the next integer and so needs a far
larger quotient before an f32 ulp can cross it. `Ltx2DecodeAudioWav` caps the
read at `max_duration` (`ltx2_audio_input.cpp:136-139`), which the engine
defaults to the clip's own duration, `frames / fps`
(`ltx2_video.cpp:3063-3069`), so a render has to be over 35 minutes long, or
carry an explicit `--audio-max-duration` at least that large, before any of this
is reached.

**An earlier draft of this section said `--audio-max-duration` is opt-in, and
therefore that the whole file passes through without it. The premise is true and
the conclusion is false**, which is why the error survived a `file:line` cite.
`examples/ltx2_gen/main.cpp:512-515` does only forward `audio_max_duration` when
the flag was given, so the flag IS opt-in at the command line. The default lives
one layer down: `ltx2_video.cpp:3069` reads that extra with
`static_cast<double>(frames) / fps` as its fallback, so an absent flag becomes
the clip's own duration rather than no cap at all. The help text says so in as
many words — "default: the clip's own duration",
`examples/ltx2_gen/main.cpp:187-188`. Recorded rather than quietly corrected,
because reading a default off the layer that FORWARDS a knob instead of the layer
that RESOLVES it is a mistake that repeats, and because the false version made
this repair read as a live user-facing bug, which it is not.

What it is: a parity defect against upstream on a reached path. Where it does
fire, the extra sample moves `frames = 1 + (samples + 2 * pad - n_fft) / hop`
wherever it crosses a hop boundary, which is the conditioning shape — the
consequence the truncation arithmetic exists to get right.

**Why the 276060-pair sweep missed it.** That sweep compared the port's
expression against **the formula**. It never called
`torchaudio.functional.resample(...).shape[-1]`. It validated `:1427`, and the
clamp lives outside the expression, on `:1428`. A reimplementation checked
against a transcription of the reference's arithmetic agrees perfectly and is
still wrong about what the reference *returns*.

The coverage was inverted as well: the two ratios 8f gated, 44100 -> 16000 and
22050 -> 16000, are precisely the safe ones, because their exact ceil is
divisible by 32 or 64 and stays f32-representable. Section 8f was the right
lengths at the wrong ratios, exactly as 8d was the right ratios at the wrong
lengths.

**The repair** is `const int64_t clamped = std::min(target, columns);`, used for
the allocation, both fill bounds, and `out_samples`. The ordinary path is
`target < columns` and still truncates exactly as before; `std::min` changes
nothing there.

**The re-run sweep calls the function.** 496194 (ratio, length) pairs across
nineteen ratios — 495900 lengths under 2 million, plus 294 in six bands around
2^25 where the clamp bites — comparing `min(target, columns)` against
`torchaudio.functional.resample(...).shape[-1]`: zero divergences. The scripts
are not committed; they are two loops over the ratio table above, and the point
that survives is the method, not the file.

**The new arms.** `CeilClamp` is 48000 -> 16000 at 100663303 samples — 8d's own
`Down` ratio at a length 8d cannot reach. `CeilOver` is 44100 -> 22050 at
33554438 samples, where the narrowing rounds up and the columns hold it, so
upstream returns `exact_ceil + 1`; it is there because a clamp to the exact
integer ceil would satisfy `CeilClamp` and fail this. Both expected lengths come
from executing `resample_audio` and reading `.shape[-1]`, not from a formula.

### One more kaiser statement, in product output

The row's first pass repaired eight of the nine statements that misnamed the
filter. The ninth was `examples/ltx2_gen/main.cpp`'s `--help` text, which is not
a comment: it told users that the sample rate "must equal the audio VAE's mel
front-end rate", that "neither is converted, because upstream resamples with a
polyphase kaiser resampler that is not ported here", and that "both mismatches
are refused". All three were false at that head. Repaired, and the sweep re-run
over every git-tracked file rather than over source and specs: `git grep -inE
"(^|[^a-z])kaiser"` now returns only the vocoder's genuine kaiser-sinc
anti-aliasing filter, this row's own prose about the repair, and unrelated
tokenizer and HuggingFace-org byte matches.

### Red before, green after

| | Before | After |
|---|---|---|
| `test_ltx2_vae -tc='*waveform_to_mel*'` | THREW the rate refusal at `ltx2_audio_vae.cpp:1053` (at `6643b2bbf`) | 48/48 cases, 3259/3259 assertions |
| `test_ltx2_video -tc='*RESAMPLED*'` | THREW `'high.wav' is sampled at 44100 Hz ...` | 110/110 cases, 4876/4876 assertions |
| `test_ltx2_vae -tc='*resampl*'`, section 8f | 50/54 with the EXACT INTEGER CEIL in place: `CeilAt` 65560 vs 65559 and its tail 0.256834 out, `CeilAlt` 65720 vs 65719 and its tail 0.358731 out | 54/54 |
| `test_ltx2_vae -tc='*resampl*'`, section 8f's two clamp arms | 66/68 with `:1427` ported and `:1428` NOT: `CeilClamp` 33554436 vs 33554435 and its tail 0.321509 out | 68/68 |

The first two reds are the refusal itself, which is what the row exists to
delete. The third is the truncation repair below, and it is red-first in the
strict sense: the two bracketing arms `CeilBelow` and `CeilAbove` stay GREEN
under the same old formula, so the arm discriminates rather than simply failing.
The same old formula against the test file as it stood BEFORE section 8f leaves
`*resampl*` 26/26, `*waveform_to_mel*` 9/9 and `test_ltx2_video`'s `*RESAMPLED*`
14/14 all green — the measurement that says the gate was blind, rather than an
argument that it was.

The fourth is the slice repair. It is red-first in the same strict sense and one
step further: `CeilOver`, the other new arm, stays GREEN without the clamp,
because upstream's answer there is `exact_ceil + 1` and the unclamped port
already produced it. So the pair discriminates in both directions — `CeilClamp`
fails a port that trusts `:1427` alone, and `CeilOver` fails a port that
"corrects" it by clamping to the exact integer ceil.

### The reachability mutation, and what it proved

Deleting the `Ltx2ResampleWaveform` call inside `Ltx2WaveformToLogMel` and
rebuilding:

* the video-level case reds on `high_trace.audio_latent_digest !=
  misread_trace.audio_latent_digest` — the two takes carry **byte-identical PCM**
  and differ only in the header's rate field, so they produce the same digest
  (`18285287296143238670`) exactly when nothing read the rate;
* the `waveform_to_mel` case reds on the frame count first (38 vs 14) and the mel
  length second (608 vs 224);
* **the section-8d unit case still PASSES.** That is the point of the mutation:
  `Ltx2ResampleWaveform` works whether or not anything calls it, and a test that
  constructs it by hand measures the class. The tree was restored byte-for-byte
  and both suites re-run green.

That video count reads 4876 and not the 4870 this row measured before its last
merge from `main`: `origin/main` added the `dims=2` upsampler cases to the same
suite. Re-read after the merge rather than carried across it, because a merge
falsifies a count without touching a line this row owns.

The mutation was RE-RUN at the truncation-repair head, because the repair moves
the very line the mutation deletes and a mutation proof does not survive an edit
to its own subject. It behaves identically there: 8d/8f 54/54, the mel case red
at 38 vs 14 frames and 608 vs 224, the video case red on the same digest
`18285287296143238670`. `ltx2_audio_vae.cpp` sha256 before and after that
restore: `fd1b68a71594d7bfd28dcb7f16e67e8c0abe91d5c267e089eb0bee3e3755af03`
(the earlier run's subject was the same file at `052047579`, sha256
`88055b01e9c2bf0bb29d826b2c61dc4c8439a1d8ae6d779a46969b50b0c0212b`).

Re-run a third time at the SLICE-repair head, for the same reason: that repair
edits `Ltx2ResampleWaveform`, so it rewrites the file the mutation's subject
lives in and a proof recorded against the old bytes no longer describes these.
It behaves identically again — `*resampl*` 68/68 GREEN under the mutation, the
mel case red at 38 vs 14 frames and 608 vs 224, `test_ltx2_video -tc='*RESAMPLED*'`
red on the same digest `18285287296143238670`, and `test_ltx2_video -tc='*audio*'`
162/162 green, which is why `*RESAMPLED*` and not `*audio*` is the filter that
carries this proof. `ltx2_audio_vae.cpp` sha256 before and after the restore:
`b82b05a913c91d952783224aebc1f72e669f05c5e350c9f8b3e02746688cc78b`. Both builds
exited 0, checked rather than assumed, because on this row a mutation build that
failed `-Werror` once already let three suites re-run stale binaries and print
green.

The mutation's FIRST build failed `-Werror` on the now-unused `sampling_rate`,
and the three suites then re-ran the STALE binaries and printed green — the
shape in which a mutation that never applied reads as a passing test. Recorded
because the reading it produced was indistinguishable from the real one until
the build's exit status was checked.

### Rejected

* **A `double` accumulator for the kernel** — §4, measured above.
* **Porting `sinc_interp_kaiser`** — unreached from `ops.py:40`; it would be a
  branch nothing can enter.
* **Keeping `want_sample_rate` on `Ltx2DecodeAudioWav`** — upstream's decoder has
  no target rate. The parameter existed only to raise, so it went with the raise.
* **Guarding the resample at the call site** (`if (rate != target)`) rather than
  inside `Ltx2ResampleWaveform` — it would have moved upstream's own early return
  (`ops.py:38-39`) off the production path, leaving that arm reachable only from
  a test.

---

## Owed

* **A real-checkpoint A2V render from a 44.1 kHz source.** The gate here is
  reduced-dimension fixtures plus executed-upstream goldens; a full render needs
  the GPU lease and belongs to the campaign's render rows. Inherits
  [#2526](https://github.com/mudler/vllm.cpp/issues/2526).
* **The `sinc_interp_kaiser` arm and the `beta` knob.** Unreached from
  `ops.py:40`; deliberately not ported (§1). Inherits
  [#2526](https://github.com/mudler/vllm.cpp/issues/2526).
* **The audio VAE's f32/bf16 tension.** `ltx2_audio_vae.cpp` argues f32 from
  upstream's vocoder float32 pin (`vocoder.py:575-580`) while upstream builds
  `AudioDecoder` in bf16. This row does not resolve it and does not contradict
  it: the resampler's float32 is argued from `ops.py`'s own chain (§4), not from
  that pin. Inherits [#2526](https://github.com/mudler/vllm.cpp/issues/2526).
